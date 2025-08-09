import os
import io
import asyncio
import tempfile
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

import fitz  # PyMuPDF
import streamlit as st
from dotenv import load_dotenv

# LangChain-like imports
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# ----------------- Setup & config -----------------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")

PDF_DIR = "pdfs"
FAISS_INDEX_PATH = "faiss_index"
os.makedirs(PDF_DIR, exist_ok=True)

try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

if not GOOGLE_API_KEY:
    llm = None
    embedding_model = None
else:
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY)
    embedding_model = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", google_api_key=GOOGLE_API_KEY
    )

DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50
DEFAULT_K = 5

SYSTEM_PROMPT = """You are a knowledgeable assistant.
You can answer questions based on the provided documents and general knowledge.
You must:
- Give clear, concise, well-structured responses.
- Use simple, professional language.
- Ask clarifying questions if the user input is vague.
- Maintain context across the conversation.
"""

# ----------------- Dataclasses & helpers -----------------
@dataclass
class RetrievedChunk:
    page_content: str
    metadata: dict
    score: Optional[float] = None

# ----------------- PDF ingestion / parsing -----------------
def extract_docs_from_pdf_bytes(file_bytes: bytes, filename: str) -> List[Document]:
    docs: List[Document] = []
    try:
        with fitz.open(stream=file_bytes, filetype="pdf") as pdf:
            for i, page in enumerate(pdf, start=1):
                text = page.get_text()
                if text and text.strip():
                    metadata = {"source": filename, "page": i}
                    docs.append(Document(page_content=text, metadata=metadata))
    except Exception as e:
        st.error(f"Failed to parse {filename}: {e}")
    return docs

def save_uploaded_pdf(uploaded_file) -> str:
    path = os.path.join(PDF_DIR, uploaded_file.name)
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path

# ----------------- Vector DB / Retriever -----------------
def build_faiss_index(docs: List[Document], chunk_size: int, chunk_overlap: int) -> FAISS:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    split_docs = splitter.split_documents(docs)
    if embedding_model is None:
        raise RuntimeError("Embedding model not initialized. Set GEMINI_API_KEY in .env.")
    db = FAISS.from_documents(split_docs, embedding_model)
    db.save_local(FAISS_INDEX_PATH)
    return db

def load_or_get_faiss() -> Optional[FAISS]:
    if os.path.exists(FAISS_INDEX_PATH):
        if embedding_model is None:
            st.warning("FAISS index exists but no embedding model configured (set GEMINI_API_KEY).")
            return None
        try:
            db = FAISS.load_local(FAISS_INDEX_PATH, embedding_model, allow_dangerous_deserialization=True)
            return db
        except Exception as e:
            st.error(f"Failed to load FAISS index: {e}")
            return None
    return None

def retriever_from_db(db: FAISS, k: int = DEFAULT_K):
    return db.as_retriever(search_kwargs={"k": k})

# ----------------- Retrieval & LLM helpers -----------------
def retrieve_relevant_chunks(retriever, query: str, metadata_filter: Optional[dict] = None) -> List[RetrievedChunk]:
    if retriever is None:
        return []
    try:
        if metadata_filter:
            docs = retriever.get_relevant_documents(query, filter=metadata_filter)
        else:
            docs = retriever.get_relevant_documents(query)
    except TypeError:
        docs = retriever.get_relevant_documents(query)
    except Exception:
        try:
            docs = retriever.invoke(query)
        except Exception:
            docs = []
    return [RetrievedChunk(page_content=d.page_content, metadata=d.metadata) for d in docs]

def format_sources(retrieved: List[RetrievedChunk]) -> str:
    src_map: Dict[str, set] = {}
    for r in retrieved:
        src = r.metadata.get("source", "unknown")
        page = r.metadata.get("page")
        src_map.setdefault(src, set()).add(str(page) if page else "?")
    lines = []
    for src, pages in src_map.items():
        pages_list = ", ".join(sorted(pages, key=lambda x: int(x) if x.isdigit() else x))
        lines.append(f"- {src} (pages: {pages_list})")
    return "\n".join(lines) if lines else "No sources."

def build_context_text(retrieved: List[RetrievedChunk], max_chars: int = 3500) -> str:
    pieces, total = [], 0
    for r in retrieved:
        text = r.page_content.strip()
        if not text:
            continue
        if total + len(text) > max_chars:
            remaining = max_chars - total
            if remaining > 0:
                pieces.append(text[:remaining])
            break
        pieces.append(text)
        total += len(text)
    return "\n\n---\n\n".join(pieces)

def llm_invoke(messages: List, temperature: float = 0.0) -> str:
    if llm is None:
        raise RuntimeError("LLM not initialized. Set GEMINI_API_KEY in .env.")
    resp = llm.invoke(messages)
    return getattr(resp, "content", str(resp))

# ----------------- High-level operations (tools) -----------------
def rag_answer(query: str, retriever, summary: str, history: List, k: int = DEFAULT_K) -> Tuple[str, List[RetrievedChunk]]:
    retrieved = retrieve_relevant_chunks(retriever, query) if retriever else []
    context_text = build_context_text(retrieved)

    reasoning_prompt = f"""
Answer the user's query based on the provided context (if any).
Structure the answer with headings and bullet points if relevant.

Context:
{context_text}

User Query:
{query}
"""

    msgs = [SystemMessage(content=SYSTEM_PROMPT)]
    if summary:
        msgs.append(SystemMessage(content=f"Conversation summary so far:\n{summary}"))
    msgs.extend(history)
    msgs.append(HumanMessage(content=reasoning_prompt))

    answer = llm_invoke(msgs, temperature=0.0)
    return answer, retrieved

def summarize_documents(retriever, target_doc: Optional[str], summary_style: str = "concise") -> str:
    if retriever is None:
        return "No retriever (index) available for summarization."
    query = "Summarize the document."
    metadata_filter = {"source": target_doc} if target_doc else None
    retrieved = retrieve_relevant_chunks(retriever, query, metadata_filter=metadata_filter)
    context_text = build_context_text(retrieved, max_chars=8000)
    msgs = [SystemMessage(content="You are an assistant that summarizes documents."),
            HumanMessage(content=f"Summarize the following document content in a {summary_style} way:\n\n{context_text}")]
    return llm_invoke(msgs)

# ----------------- Simple Rule-based Agent -----------------
def agent_route_and_act(user_query: str, retriever, st_state) -> Tuple[str, List[RetrievedChunk]]:
    q_lower = user_query.lower()
    if any(tok in q_lower for tok in ["summarize", "summary", "summarise", "summarization", "tl;dr"]):
        target_doc = st_state.get("doc_filter") or None
        summary_text = summarize_documents(retriever, target_doc, summary_style="concise")
        return (f"### Summary ({target_doc or 'All Documents'})\n\n" + summary_text), []
    return rag_answer(user_query, retriever, st_state.get("summary", ""), st_state.get("history", []))

# ----------------- Streamlit UI -----------------
st.set_page_config(page_title="📄 General PDF RAG Bot", layout="wide")
st.title("📄 General Document RAG Chatbot")
st.markdown("Upload PDFs and ask questions about their content.")

# Sidebar
with st.sidebar:
    st.header("Settings & Indexing")
    api_key_input = st.text_input("GEMINI_API_KEY (optional, overrides .env)", type="password")
    if api_key_input:
        os.environ["GEMINI_API_KEY"] = api_key_input
        st.info("Restart app to fully apply new API key.")
    chunk_size = st.number_input("Chunk size", min_value=200, max_value=2000, value=DEFAULT_CHUNK_SIZE, step=50)
    chunk_overlap = st.number_input("Chunk overlap", min_value=0, max_value=500, value=DEFAULT_CHUNK_OVERLAP, step=10)
    k = st.number_input("Retriever top-k", min_value=1, max_value=20, value=DEFAULT_K, step=1)
    uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    if uploaded_files:
        all_docs = []
        for uf in uploaded_files:
            path = save_uploaded_pdf(uf)
            docs = extract_docs_from_pdf_bytes(uf.getvalue(), uf.name)
            all_docs.extend(docs)
            st.success(f"Saved & parsed {uf.name} ({len(docs)} pages parsed).")
        if all_docs:
            try:
                db = build_faiss_index(all_docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                st.success("Rebuilt FAISS index with uploaded documents.")
            except Exception as e:
                st.error(f"Failed to build index: {e}")

# Chat
if "history" not in st.session_state:
    st.session_state.history = []
if "summary" not in st.session_state:
    st.session_state.summary = ""
if "doc_filter" not in st.session_state:
    st.session_state.doc_filter = None

db = load_or_get_faiss()
retriever = retriever_from_db(db, k=int(k)) if db else None

for msg in st.session_state.history:
    with st.chat_message("user" if isinstance(msg, HumanMessage) else "assistant"):
        st.markdown(msg.content)

user_input = st.chat_input(placeholder="Ask a question about your documents...")
if user_input:
    st.session_state.history.append(HumanMessage(content=user_input))
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.spinner("Thinking..."):
        try:
            response, retrieved = agent_route_and_act(user_input, retriever, st.session_state)
        except RuntimeError as e:
            response = f"Error: {e}"
            retrieved = []
    st.session_state.history.append(AIMessage(content=response))
    with st.chat_message("assistant"):
        st.markdown(response)

    with st.expander("Source Documents"):
        if not retrieved:
            st.info("No relevant source documents retrieved.")
        else:
            st.markdown(format_sources(retrieved))
