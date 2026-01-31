from __future__ import annotations

from langchain_openai import ChatOpenAI, OpenAIEmbeddings


def get_embeddings(openai_api_key: str, model: str) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(openai_api_key=openai_api_key, model=model)


def get_chat_llm(openai_api_key: str, model: str, temperature: float = 0) -> ChatOpenAI:
    return ChatOpenAI(openai_api_key=openai_api_key, model=model, temperature=temperature)
