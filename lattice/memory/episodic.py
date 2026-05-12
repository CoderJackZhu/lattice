from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from lattice.memory.base import EmbeddingFn, MemoryItem
from lattice.memory.stores.in_memory import InMemoryVectorStore


class EpisodicMemory:
    def __init__(self, store_path: str, embedding_fn: EmbeddingFn | None = None) -> None:
        self._store_path = Path(store_path).expanduser()
        self._embedding_fn = embedding_fn
        self._items: list[MemoryItem] = []
        self._vector_store: InMemoryVectorStore | None = None
        self._indexed = False

        if embedding_fn:
            self._vector_store = InMemoryVectorStore(embedding_fn)

        self._load()

    def _load(self) -> None:
        if self._store_path.exists():
            try:
                data = json.loads(self._store_path.read_text(encoding="utf-8"))
                self._items = [MemoryItem(**item) for item in data]
            except (json.JSONDecodeError, TypeError):
                self._items = []

    def _save(self) -> None:
        self._store_path.parent.mkdir(parents=True, exist_ok=True)
        data = [asdict(item) for item in self._items]
        self._store_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    async def _ensure_indexed(self) -> None:
        if self._vector_store and not self._indexed and self._items:
            texts = [item.content for item in self._items]
            metadatas = [item.metadata for item in self._items]
            await self._vector_store.add(texts, metadatas)
            self._indexed = True

    async def store(self, item: MemoryItem) -> None:
        item.source = "episodic"
        if not item.timestamp:
            item.timestamp = time.time()
        self._items.append(item)
        if self._vector_store:
            await self._vector_store.add([item.content], [item.metadata])
            self._indexed = True
        self._save()

    async def retrieve(self, query: str, top_k: int = 5) -> list[MemoryItem]:
        if self._vector_store:
            await self._ensure_indexed()
            results = await self._vector_store.search(query, top_k)
            return [
                MemoryItem(
                    content=r.text,
                    score=r.score,
                    metadata=r.metadata,
                    source="episodic",
                )
                for r in results
            ]

        # Fallback: keyword matching
        if not query:
            return self._items[-top_k:]

        keywords = query.lower().split()
        scored: list[tuple[float, MemoryItem]] = []
        for item in self._items:
            content_lower = item.content.lower()
            score = sum(1 for kw in keywords if kw in content_lower)
            if score > 0:
                scored.append((score / len(keywords), item))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            MemoryItem(
                content=item.content,
                score=score,
                metadata=item.metadata,
                timestamp=item.timestamp,
                source="episodic",
            )
            for score, item in scored[:top_k]
        ]

    async def clear(self) -> None:
        self._items.clear()
        if self._vector_store and self._embedding_fn:
            self._vector_store = InMemoryVectorStore(self._embedding_fn)
            self._indexed = False
        self._save()
