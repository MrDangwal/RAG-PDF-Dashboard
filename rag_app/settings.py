from __future__ import annotations

from pathlib import Path

APP_NAME = "RAG PDF Dashboard"
STORE_DIR = Path("rag_store")
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_EMBED_MODEL = "text-embedding-3-small"
DEFAULT_CHAT_MODEL = "gpt-4o-mini"
DEFAULT_TOP_K = 4
