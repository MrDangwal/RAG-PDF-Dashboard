from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from .chunking import chunk_text
from .llm import get_embeddings
from .pdf_utils import read_pdfs
from .rag import answer_question, format_sources
from .settings import (
    APP_NAME,
    DEFAULT_CHAT_MODEL,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_EMBED_MODEL,
    DEFAULT_TOP_K,
)
from .store import create_index_id, list_indexes, save_metadata
from .vector_store import load_faiss, save_faiss


_DEF_KEY_PLACEHOLDER = ""


def _get_openai_key() -> str:
    if "openai_api_key" not in st.session_state:
        st.session_state.openai_api_key = os.getenv("OPENAI_API_KEY", "")
    return st.session_state.openai_api_key


def _render_sidebar() -> dict:
    st.sidebar.title("RAG Dashboard")

    api_key_input = st.sidebar.text_input(
        "OpenAI API Key",
        type="password",
        value=_DEF_KEY_PLACEHOLDER,
        help="Uses OPENAI_API_KEY env var if left blank.",
    )
    if api_key_input:
        st.session_state.openai_api_key = api_key_input

    model = st.sidebar.text_input("Chat model", value=DEFAULT_CHAT_MODEL)
    embed_model = st.sidebar.text_input("Embedding model", value=DEFAULT_EMBED_MODEL)
    top_k = st.sidebar.number_input("Top K", min_value=1, max_value=20, value=DEFAULT_TOP_K)

    st.sidebar.markdown("---")
    st.sidebar.subheader("RAG Index")

    indexes = list_indexes()
    options = ["Create new"] + [f"{idx.name} ({idx.index_id})" for idx in indexes]
    selection = st.sidebar.selectbox("Choose index", options)

    create_new = selection == "Create new"
    selected = None
    if not create_new:
        selected_id = selection.split("(")[-1].rstrip(")")
        selected = next((idx for idx in indexes if idx.index_id == selected_id), None)
        if selected:
            st.sidebar.caption(
                f"Created: {selected.created_at} | Docs: {selected.doc_count} | Chunks: {selected.chunk_count}"
            )

    return {
        "create_new": create_new,
        "selected": selected,
        "model": model,
        "embed_model": embed_model,
        "top_k": int(top_k),
    }


def _render_new_index(openai_key: str, embed_model: str) -> str | None:
    st.subheader("Create new RAG index")
    name = st.text_input("Index name")
    chunk_size = st.number_input("Chunk size", min_value=200, max_value=2000, value=DEFAULT_CHUNK_SIZE)
    chunk_overlap = st.number_input(
        "Chunk overlap",
        min_value=0,
        max_value=500,
        value=DEFAULT_CHUNK_OVERLAP,
    )
    pdf_files = st.file_uploader(
        "Upload PDFs",
        accept_multiple_files=True,
        type=["pdf"],
    )

    if st.button("Build index", type="primary"):
        if not openai_key:
            st.error("OpenAI API key is required to build embeddings.")
            return None
        if not name:
            st.error("Index name is required.")
            return None
        if not pdf_files:
            st.error("Upload at least one PDF.")
            return None

        with st.spinner("Reading PDFs..."):
            raw_text = read_pdfs(pdf_files)

        if not raw_text.strip():
            st.error("No extractable text found in PDFs.")
            return None

        with st.spinner("Chunking text..."):
            chunks = chunk_text(raw_text, int(chunk_size), int(chunk_overlap))

        embeddings = get_embeddings(openai_api_key=openai_key, model=embed_model)
        index_id = create_index_id(name)
        index_dir = Path("rag_store") / index_id / "faiss"

        with st.spinner("Creating vector store..."):
            save_faiss(chunks, embeddings, index_dir)

        save_metadata(
            index_id=index_id,
            name=name,
            doc_count=len(pdf_files),
            chunk_count=len(chunks),
            embed_model=embed_model,
        )

        st.success("Index created.")
        return index_id

    return None


def _render_chat(
    openai_key: str,
    model: str,
    embed_model: str,
    top_k: int,
    selected,
) -> None:
    if selected is None:
        st.info("Select an existing index or create a new one.")
        return

    st.subheader(f"Chat with: {selected.name}")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    question = st.chat_input("Ask a question")
    if not question:
        return

    if not openai_key:
        st.error("OpenAI API key is required to ask questions.")
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    embeddings = get_embeddings(openai_api_key=openai_key, model=embed_model)
    index_dir = Path("rag_store") / selected.index_id / "faiss"
    vector_store = load_faiss(embeddings, index_dir)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = answer_question(
                question=question,
                vector_store=vector_store,
                openai_api_key=openai_key,
                model=model,
                top_k=top_k,
            )
        st.write(result["answer"])
        sources = format_sources(result.get("sources", []))
        if sources:
            with st.expander("Sources"):
                for item in sources:
                    st.write(item)

    st.session_state.messages.append({"role": "assistant", "content": result["answer"]})


def main() -> None:
    st.set_page_config(page_title=APP_NAME, layout="wide")
    st.title(APP_NAME)

    sidebar = _render_sidebar()
    openai_key = _get_openai_key()

    if sidebar["create_new"]:
        created_id = _render_new_index(openai_key, sidebar["embed_model"])
        if created_id:
            st.session_state.messages = []
    else:
        _render_chat(
            openai_key=openai_key,
            model=sidebar["model"],
            embed_model=sidebar["embed_model"],
            top_k=sidebar["top_k"],
            selected=sidebar["selected"],
        )
