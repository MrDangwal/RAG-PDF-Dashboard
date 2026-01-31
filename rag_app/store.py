from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from uuid import uuid4

from .settings import STORE_DIR


@dataclass(frozen=True)
class RagIndex:
    index_id: str
    name: str
    created_at: str
    doc_count: int
    chunk_count: int
    embed_model: str

    @property
    def path(self) -> Path:
        return STORE_DIR / self.index_id


def ensure_store_dir() -> None:
    STORE_DIR.mkdir(parents=True, exist_ok=True)


def _meta_path(index_id: str) -> Path:
    return STORE_DIR / index_id / "meta.json"


def create_index_id(name: str) -> str:
    slug = "".join(ch.lower() if ch.isalnum() else "-" for ch in name).strip("-")
    slug = slug or "rag"
    return f"{slug}-{uuid4().hex[:8]}"


def save_metadata(
    index_id: str,
    name: str,
    doc_count: int,
    chunk_count: int,
    embed_model: str,
) -> None:
    ensure_store_dir()
    index_dir = STORE_DIR / index_id
    index_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "index_id": index_id,
        "name": name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "doc_count": doc_count,
        "chunk_count": chunk_count,
        "embed_model": embed_model,
    }
    _meta_path(index_id).write_text(json.dumps(meta, indent=2))


def list_indexes() -> list[RagIndex]:
    ensure_store_dir()
    indexes: list[RagIndex] = []
    for meta_file in STORE_DIR.glob("*/meta.json"):
        try:
            meta = json.loads(meta_file.read_text())
            indexes.append(
                RagIndex(
                    index_id=meta["index_id"],
                    name=meta["name"],
                    created_at=meta["created_at"],
                    doc_count=meta["doc_count"],
                    chunk_count=meta["chunk_count"],
                    embed_model=meta["embed_model"],
                )
            )
        except Exception:
            continue
    indexes.sort(key=lambda item: item.created_at, reverse=True)
    return indexes


def get_index(index_id: str) -> RagIndex | None:
    meta_file = _meta_path(index_id)
    if not meta_file.exists():
        return None
    meta = json.loads(meta_file.read_text())
    return RagIndex(
        index_id=meta["index_id"],
        name=meta["name"],
        created_at=meta["created_at"],
        doc_count=meta["doc_count"],
        chunk_count=meta["chunk_count"],
        embed_model=meta["embed_model"],
    )
