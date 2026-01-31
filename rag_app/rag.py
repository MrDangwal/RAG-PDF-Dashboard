from __future__ import annotations

from typing import Any

from langchain.chains import RetrievalQA
from langchain_core.documents import Document

from .llm import get_chat_llm


def answer_question(
    question: str,
    vector_store,
    openai_api_key: str,
    model: str,
    top_k: int,
) -> dict[str, Any]:
    llm = get_chat_llm(openai_api_key=openai_api_key, model=model)
    retriever = vector_store.as_retriever(search_kwargs={"k": top_k})
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )
    result = qa({"query": question})
    sources = result.get("source_documents", [])
    return {
        "answer": result.get("result", ""),
        "sources": sources,
    }


def format_sources(sources: list[Document]) -> list[str]:
    formatted: list[str] = []
    for doc in sources:
        source = doc.metadata.get("source", "") if hasattr(doc, "metadata") else ""
        snippet = (doc.page_content or "")[:200].replace("\n", " ")
        formatted.append(f"{source} - {snippet}".strip(" -"))
    return formatted
