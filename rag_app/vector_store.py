from __future__ import annotations

from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings


def save_faiss(text_chunks: list[str], embeddings: Embeddings, index_dir: Path) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local(str(index_dir))


def load_faiss(embeddings: Embeddings, index_dir: Path) -> FAISS:
    return FAISS.load_local(
        str(index_dir),
        embeddings,
        allow_dangerous_deserialization=True,
    )
