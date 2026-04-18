"""
멀티세션 RAG 챗봇 — Supabase 세션/벡터 저장, OpenAI 임베딩·gpt-4o-mini 스트리밍.
실행: streamlit run multi-session-ref.py (7.MultiService/code 디렉터리에서)
"""

from __future__ import annotations

import logging
import os
import re
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from supabase import Client, create_client

# ---------------------------------------------------------------------------
# 경로 · 환경
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = _THIS_DIR.parent.parent.parent
ENV_PATH = ROOT_DIR / ".env"
if ENV_PATH.is_file():
    pass
elif (_THIS_DIR / ".env").is_file():
    ROOT_DIR = _THIS_DIR
    ENV_PATH = _THIS_DIR / ".env"
else:
    # Streamlit Cloud: repo is read-only; avoid parent chains like /mount
    ROOT_DIR = _THIS_DIR
    ENV_PATH = _THIS_DIR / ".env"

load_dotenv(ENV_PATH)


def _writable_log_dir() -> Path | None:
    """Use a writable directory (e.g. /tmp on Streamlit Cloud)."""
    candidates = [
        ROOT_DIR / "logs",
        Path(tempfile.gettempdir()) / "multi_session_ref_logs",
    ]
    for d in candidates:
        try:
            d.mkdir(parents=True, exist_ok=True)
            probe = d / ".w_probe"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
            return d
        except OSError:
            continue
    return None


def setup_logging() -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    log_dir = _writable_log_dir()
    if log_dir is not None:
        try:
            log_file = log_dir / f"chatbot_{datetime.now().strftime('%Y%m%d')}.log"
            handlers.insert(0, logging.FileHandler(log_file, encoding="utf-8"))
        except OSError:
            pass
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
    )
    for name in ("httpx", "httpcore", "urllib3", "openai", "langchain", "langchain_openai"):
        logging.getLogger(name).setLevel(logging.WARNING)


setup_logging()
log = logging.getLogger("multi_session_rag")


def env_or_warn() -> tuple[str | None, str | None, str | None]:
    import os

    return (
        os.getenv("OPENAI_API_KEY"),
        os.getenv("SUPABASE_URL"),
        os.getenv("SUPABASE_ANON_KEY"),
    )


# ---------------------------------------------------------------------------
# 유틸 (ref.txt)
# ---------------------------------------------------------------------------
def remove_separators(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"~~[^~]+~~", "", text)
    text = re.sub(r"(?m)^\s*[-_=]{3,}\s*$", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Supabase
# ---------------------------------------------------------------------------
@st.cache_resource
def get_supabase(url: str, key: str) -> Client:
    return create_client(url, key)


def list_sessions(sb: Client) -> list[dict[str, Any]]:
    r = sb.table("sessions").select("id,title,updated_at").order("updated_at", desc=True).execute()
    return r.data or []


def fetch_chat_messages(sb: Client, session_id: str) -> list[dict[str, str]]:
    r = (
        sb.table("chat_messages")
        .select("role,content")
        .eq("session_id", session_id)
        .order("id")
        .execute()
    )
    return r.data or []


def replace_chat_messages(sb: Client, session_id: str, history: list[dict[str, str]]) -> None:
    sb.table("chat_messages").delete().eq("session_id", session_id).execute()
    rows = [{"session_id": session_id, "role": m["role"], "content": m["content"]} for m in history]
    if rows:
        sb.table("chat_messages").insert(rows).execute()
    r = sb.table("sessions").select("title").eq("id", session_id).single().execute()
    title = r.data.get("title", "") if r.data else ""
    sb.table("sessions").update({"title": title}).eq("id", session_id).execute()


def create_session(sb: Client, title: str) -> str:
    r = sb.table("sessions").insert({"title": title}).execute()
    if not r.data:
        raise RuntimeError("세션 생성 응답이 비어 있습니다.")
    return r.data[0]["id"]


def delete_session(sb: Client, session_id: str) -> None:
    sb.table("sessions").delete().eq("id", session_id).execute()


def distinct_vector_filenames(sb: Client, session_id: str) -> list[str]:
    r = (
        sb.table("vector_documents")
        .select("file_name")
        .eq("session_id", session_id)
        .execute()
    )
    rows = r.data or []
    seen: set[str] = set()
    out: list[str] = []
    for row in rows:
        fn = row.get("file_name")
        if fn and fn not in seen:
            seen.add(fn)
            out.append(fn)
    return sorted(out)


def copy_vectors_to_session(sb: Client, from_sid: str, to_sid: str, batch_size: int = 10) -> None:
    r = sb.table("vector_documents").select("*").eq("session_id", from_sid).execute()
    rows = r.data or []
    for i in range(0, len(rows), batch_size):
        chunk = rows[i : i + batch_size]
        insert_rows = []
        for row in chunk:
            insert_rows.append(
                {
                    "session_id": to_sid,
                    "file_name": row["file_name"],
                    "content": row["content"],
                    "embedding": row["embedding"],
                    "metadata": row.get("metadata") or {},
                }
            )
        if insert_rows:
            sb.table("vector_documents").insert(insert_rows).execute()


def insert_vector_batches(
    sb: Client,
    session_id: str,
    file_name: str,
    docs: list[Document],
    embeddings: OpenAIEmbeddings,
    batch_size: int = 10,
) -> None:
    texts = [d.page_content for d in docs]
    embs = embeddings.embed_documents(texts)
    rows: list[dict[str, Any]] = []
    for d, e in zip(docs, embs, strict=True):
        rows.append(
            {
                "session_id": session_id,
                "file_name": file_name,
                "content": d.page_content,
                "embedding": e,
                "metadata": d.metadata or {},
            }
        )
    for i in range(0, len(rows), batch_size):
        sb.table("vector_documents").insert(rows[i : i + batch_size]).execute()


def retrieve_by_rpc(
    sb: Client,
    embeddings: OpenAIEmbeddings,
    session_id: str,
    query: str,
    k: int = 5,
) -> list[Document]:
    qe = embeddings.embed_query(query)
    try:
        r = sb.rpc(
            "match_vector_documents",
            {
                "query_embedding": qe,
                "match_count": k,
                "filter_session_id": session_id,
            },
        ).execute()
        data = r.data or []
        docs: list[Document] = []
        for row in data:
            docs.append(
                Document(
                    page_content=row.get("content", ""),
                    metadata={
                        "file_name": row.get("file_name", ""),
                        "similarity": row.get("similarity"),
                    },
                )
            )
        return docs
    except Exception as e:
        log.warning("RPC 검색 실패, 세션 전체 후 필터 폴백: %s", e)
        r2 = sb.table("vector_documents").select("file_name,content").eq("session_id", session_id).execute()
        raw = r2.data or []
        return [
            Document(page_content=x["content"], metadata={"file_name": x.get("file_name", "")})
            for x in raw[: max(k * 4, 20)]
        ]


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------
def get_llm(model_name: str = "gpt-4o-mini", temperature: float = 0.7) -> ChatOpenAI:
    return ChatOpenAI(model=model_name, temperature=temperature, streaming=True)


def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model="text-embedding-3-small")


def generate_session_title(openai_key: str, first_user: str, first_assistant: str) -> str:
    client = OpenAI(api_key=openai_key)
    r = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.3,
        messages=[
            {
                "role": "system",
                "content": "첫 질문과 답변을 바탕으로 짧은 한국어 세션 제목 한 줄만 출력하세요. 따옴표나 부가 설명 없이 제목만.",
            },
            {
                "role": "user",
                "content": f"사용자:\n{first_user[:4000]}\n\n어시스턴트:\n{first_assistant[:4000]}",
            },
        ],
    )
    t = (r.choices[0].message.content or "새 세션").strip()
    return t[:120] if t else "새 세션"


def generate_followup_questions(openai_key: str, answer_text: str) -> str:
    client = OpenAI(api_key=openai_key)
    r = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.4,
        messages=[
            {
                "role": "system",
                "content": "위 답변과 관련해 사용자가 이어서 물어볼 만한 질문을 정확히 3개, 한국어로 번호(1. 2. 3.)만 붙여 출력하세요.",
            },
            {"role": "user", "content": answer_text[:12000]},
        ],
    )
    return (r.choices[0].message.content or "").strip()


def build_lc_messages(
    system_prompt: str,
    history: list[dict[str, str]],
    user_message: str,
) -> list[Any]:
    msgs: list[Any] = [SystemMessage(content=system_prompt)]
    for m in history[-50:]:
        if m["role"] == "user":
            msgs.append(HumanMessage(content=m["content"]))
        else:
            msgs.append(AIMessage(content=m["content"]))
    msgs.append(HumanMessage(content=user_message))
    return msgs


# ---------------------------------------------------------------------------
# 세션 UI 상태
# ---------------------------------------------------------------------------
def init_state() -> None:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "current_session_id" not in st.session_state:
        st.session_state.current_session_id = None
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = []
    if "rag_enabled" not in st.session_state:
        st.session_state.rag_enabled = "RAG 사용"
    if "picker_label" not in st.session_state:
        st.session_state.picker_label = "(새 대화)"


def load_session_into_ui(sb: Client, session_id: str) -> None:
    msgs = fetch_chat_messages(sb, session_id)
    st.session_state.chat_history = [{"role": m["role"], "content": m["content"]} for m in msgs]
    st.session_state.current_session_id = session_id


def format_session_label(title: str, sid: str) -> str:
    return f"{title[:48]} — {str(sid)[:8]}"


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="멀티세션 RAG 챗봇", page_icon="📚", layout="wide")
    init_state()

    openai_key, supa_url, supa_key = env_or_warn()
    missing = []
    if not openai_key:
        missing.append("OPENAI_API_KEY")
    if not supa_url:
        missing.append("SUPABASE_URL")
    if not supa_key:
        missing.append("SUPABASE_ANON_KEY")
    if missing:
        st.error(f"환경 변수가 없습니다: {', '.join(missing)}. {ENV_PATH} 를 확인하세요.")
        return

    sb = get_supabase(supa_url, supa_key)
    embeddings = get_embeddings()
    llm = get_llm("gpt-4o-mini")

    # --- CSS (ref.txt) ---
    st.markdown(
        """
        <style>
        .hdr-title { font-size: 4rem !important; font-weight: 800; text-align: center; margin: 0.2rem 0 1rem 0; }
        .hdr-a { color: #1f77b4 !important; }
        .hdr-b { color: #ffd700 !important; }
        h1 { color: #ff69b4 !important; font-size: 1.4rem !important; }
        h2 { color: #ffd700 !important; font-size: 1.2rem !important; }
        h3 { color: #1f77b4 !important; font-size: 1.1rem !important; }
        div[data-testid="stChatMessage"] { padding: 0.75rem 1rem; border-radius: 12px; margin-bottom: 0.5rem; }
        button[kind="primary"], button[kind="secondary"] {
            background-color: #ff69b4 !important; color: white !important; border: none !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    logo_path = ROOT_DIR / "logo.png"
    c1, c2, c3 = st.columns([1, 3, 1])
    with c1:
        if logo_path.exists():
            st.image(str(logo_path), width=180)
        else:
            st.markdown("### 📚")
    with c2:
        st.markdown(
            '<div class="hdr-title"><span class="hdr-a">멀티세션 RAG</span> <span class="hdr-b">챗봇</span></div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.empty()

    # --- 사이드바 (구분선 없음) ---
    with st.sidebar:
        st.markdown("### 모델")
        st.text("gpt-4o-mini (고정)")

        st.markdown("### RAG (PDF)")
        st.session_state.rag_enabled = st.radio(
            "PDF 검색",
            options=["사용 안 함", "RAG 사용"],
            index=1 if st.session_state.rag_enabled == "RAG 사용" else 0,
            key="rag_radio",
        )

        st.markdown("### PDF 업로드")
        files = st.file_uploader(
            "PDF (다중 선택)",
            type=["pdf"],
            accept_multiple_files=True,
            key="pdf_uploader",
        )
        if st.button("파일 처리하기", type="primary"):
            if not files:
                st.warning("PDF를 선택하세요.")
            else:
                try:
                    need_sid = st.session_state.current_session_id
                    if not need_sid:
                        first_name = files[0].name
                        need_sid = create_session(sb, title=f"문서: {first_name}")
                        st.session_state.current_session_id = need_sid
                    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
                    names: list[str] = []
                    for f in files:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tf:
                            tf.write(f.getvalue())
                            tmp_path = tf.name
                        try:
                            loader = PyPDFLoader(tmp_path)
                            pages = loader.load()
                            for d in pages:
                                d.metadata = dict(d.metadata)
                                d.metadata["file_name"] = f.name
                            splits = splitter.split_documents(pages)
                            for d in splits:
                                d.metadata["file_name"] = f.name
                            insert_vector_batches(sb, need_sid, f.name, splits, embeddings, batch_size=10)
                            names.append(f.name)
                        finally:
                            try:
                                os.unlink(tmp_path)
                            except OSError:
                                pass
                    st.session_state.processed_files = list(
                        dict.fromkeys(st.session_state.processed_files + names)
                    )
                    st.success(f"처리 완료: {', '.join(names)} (세션에 저장됨)")
                except Exception as ex:
                    log.exception("PDF 처리 오류")
                    st.error(f"PDF 처리 오류: {ex}")

        if st.session_state.processed_files:
            st.caption("처리된 파일:")
            for n in st.session_state.processed_files:
                st.text(f"· {n}")

        sessions = list_sessions(sb)
        label_to_id: dict[str, str | None] = {"(새 대화)": None}
        for s in sessions:
            lab = format_session_label(s.get("title") or "(제목 없음)", s["id"])
            label_to_id[lab] = s["id"]

        st.markdown("### 세션 관리")

        def on_pick_change() -> None:
            lab = st.session_state.get("session_pick")
            sid = label_to_id.get(lab) if lab else None
            if sid:
                try:
                    load_session_into_ui(sb, sid)
                except Exception as e:
                    log.exception("세션 로드")
                    st.session_state["_load_err"] = str(e)
            else:
                st.session_state.chat_history = []
                st.session_state.current_session_id = None

        options = list(label_to_id.keys())
        default_label = st.session_state.picker_label
        if default_label not in options:
            default_label = "(새 대화)"
        ix = options.index(default_label) if default_label in options else 0

        st.selectbox(
            "세션 선택 (선택 시 자동 로드)",
            options=options,
            index=ix,
            key="session_pick",
            on_change=on_pick_change,
        )
        st.session_state.picker_label = st.session_state.get("session_pick", "(새 대화)")

        if st.session_state.get("_load_err"):
            st.warning(st.session_state.pop("_load_err"))

        bcol1, bcol2 = st.columns(2)
        with bcol1:
            if st.button("세션저장"):
                h = st.session_state.chat_history
                if len(h) < 2:
                    st.warning("저장하려면 최소 한 번의 질문과 답변이 필요합니다.")
                else:
                    try:
                        fu = next((x["content"] for x in h if x["role"] == "user"), "")
                        fa = next((x["content"] for x in h if x["role"] == "assistant"), "")
                        title = generate_session_title(openai_key, fu, fa)
                        new_id = str(uuid.uuid4())
                        sb.table("sessions").insert({"id": new_id, "title": title}).execute()
                        rows = [
                            {"session_id": new_id, "role": m["role"], "content": m["content"]}
                            for m in h
                        ]
                        if rows:
                            sb.table("chat_messages").insert(rows).execute()
                        old = st.session_state.current_session_id
                        if old:
                            copy_vectors_to_session(sb, old, new_id)
                        st.success("새 세션으로 저장했습니다.")
                    except Exception as ex:
                        log.exception("세션 저장")
                        st.error(str(ex))

        with bcol2:
            if st.button("세션로드"):
                lab = st.session_state.get("session_pick")
                sid = label_to_id.get(lab) if lab else None
                if not sid:
                    st.warning("로드할 세션을 선택하세요.")
                else:
                    try:
                        load_session_into_ui(sb, sid)
                        st.success("세션을 불러왔습니다.")
                    except Exception as ex:
                        st.error(str(ex))

        bcol3, bcol4 = st.columns(2)
        with bcol3:
            if st.button("세션삭제"):
                lab = st.session_state.get("session_pick")
                sid = label_to_id.get(lab) if lab else None
                if not sid:
                    st.warning("삭제할 세션을 선택하세요.")
                else:
                    try:
                        delete_session(sb, sid)
                        if st.session_state.current_session_id == sid:
                            st.session_state.chat_history = []
                            st.session_state.current_session_id = None
                            st.session_state.picker_label = "(새 대화)"
                        st.success("삭제했습니다.")
                        st.rerun()
                    except Exception as ex:
                        st.error(str(ex))

        with bcol4:
            if st.button("화면초기화"):
                st.session_state.chat_history = []
                st.session_state.current_session_id = None
                st.session_state.processed_files = []
                st.session_state.picker_label = "(새 대화)"
                st.success("화면을 초기화했습니다.")
                st.rerun()

        if st.button("vectordb"):
            sid = st.session_state.current_session_id
            if not sid:
                st.info("현재 연결된 세션이 없습니다. PDF 처리 또는 세션 로드 후 사용하세요.")
            else:
                try:
                    fs = distinct_vector_filenames(sb, sid)
                    if not fs:
                        st.text("(이 세션에 저장된 벡터 파일명이 없습니다.)")
                    else:
                        for fn in fs:
                            st.text(fn)
                except Exception as ex:
                    st.error(str(ex))

        st.markdown("### 현재 설정")
        st.text(f"모델: gpt-4o-mini")
        st.text(f"RAG: {st.session_state.rag_enabled}")
        st.text(f"처리된 파일 수: {len(st.session_state.processed_files)}")
        st.text(f"대화 메시지 수: {len(st.session_state.chat_history)}")
        cs = st.session_state.current_session_id
        st.text(f"현재 세션 ID: {cs or '(없음)'}")

    for m in st.session_state.chat_history:
        with st.chat_message(m["role"]):
            st.markdown(remove_separators(m["content"]), unsafe_allow_html=False)

    prompt = st.chat_input("메시지를 입력하세요")
    if not prompt:
        return

    system_base = (
        "당신은 도움이 되는 어시스턴트입니다. 답변은 반드시 Markdown 헤딩(# ## ###)으로 구조화하고, "
        "존댓말로 완전한 문장으로 작성하세요. 구분선(--- 등)과 취소선(~~)은 사용하지 마세요. "
        "참조·출처 표시 문구는 넣지 마세요."
    )

    sid = st.session_state.current_session_id
    use_rag = st.session_state.rag_enabled == "RAG 사용" and sid

    ctx_text = ""
    if use_rag:
        try:
            docs = retrieve_by_rpc(sb, embeddings, sid, prompt, k=5)
            if docs:
                parts = []
                for d in docs:
                    fn = d.metadata.get("file_name", "")
                    parts.append(f"[파일: {fn}]\n{d.page_content}")
                ctx_text = "\n\n".join(parts)
        except Exception as e:
            log.warning("컨텍스트 검색 오류: %s", e)

    if use_rag and ctx_text:
        system_prompt = (
            system_base
            + "\n아래 참고 문맥을 활용해 답하세요. 문맥에 없으면 일반 지식으로 보완해도 됩니다.\n\n[참고 문맥]\n"
            + ctx_text
        )
    else:
        if use_rag and not ctx_text:
            system_prompt = system_base + "\n(벡터 DB에서 관련 청크를 찾지 못했습니다. 일반 지식으로 답하세요.)"
        else:
            system_prompt = system_base

    msgs = build_lc_messages(system_prompt, st.session_state.chat_history, prompt)

    with st.chat_message("user"):
        st.markdown(remove_separators(prompt), unsafe_allow_html=False)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full = ""
        try:
            for chunk in llm.stream(msgs):
                piece = getattr(chunk, "content", None) or ""
                if piece:
                    full += piece
                    placeholder.markdown(remove_separators(full) + "▌")
            main_only = full
            fu = generate_followup_questions(openai_key, main_only)
            extra = "\n\n### 💡 다음에 물어볼 수 있는 질문들\n\n" + fu
            full += extra
            placeholder.markdown(remove_separators(full))
        except Exception as ex:
            log.exception("생성 오류")
            err = f"오류: {ex}"
            full = err
            placeholder.markdown(err)

    final_text = remove_separators(full)
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    st.session_state.chat_history.append({"role": "assistant", "content": final_text})

    # 첫 턴 후 세션이 없으면 자동 생성 + 저장
    try:
        if st.session_state.current_session_id is None and len(st.session_state.chat_history) >= 2:
            fu0 = st.session_state.chat_history[0]["content"]
            fa0 = st.session_state.chat_history[1]["content"]
            t = generate_session_title(openai_key, fu0, fa0)
            new_sid = create_session(sb, title=t)
            st.session_state.current_session_id = new_sid
            replace_chat_messages(sb, new_sid, st.session_state.chat_history)
            st.session_state.picker_label = format_session_label(t, new_sid)
        elif st.session_state.current_session_id:
            replace_chat_messages(sb, st.session_state.current_session_id, st.session_state.chat_history)
            r = (
                sb.table("sessions")
                .select("title")
                .eq("id", st.session_state.current_session_id)
                .single()
                .execute()
            )
            tit = (r.data or {}).get("title") or ""
            st.session_state.picker_label = format_session_label(tit, st.session_state.current_session_id)
    except Exception as e:
        log.warning("자동 저장 실패: %s", e)

    st.rerun()


if __name__ == "__main__":
    main()
