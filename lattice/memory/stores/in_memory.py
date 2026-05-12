from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from lattice.memory.base import EmbeddingFn


@dataclass
class SearchResult:
    text: str = ""
    score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class VectorStore(Protocol):
    async def add(self, texts: list[str], metadatas: list[dict[str, Any]]) -> list[str]: ...
    async def search(self, query: str, top_k: int = 5) -> list[SearchResult]: ...
    async def delete(self, ids: list[str]) -> None: ...


@dataclass
class _Document:
    text: str
    embedding: list[float]
    metadata: dict[str, Any]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class InMemoryVectorStore:
    def __init__(self, embedding_fn: EmbeddingFn) -> None:
        self._documents: dict[str, _Document] = {}
        self._embedding_fn = embedding_fn

    async def add(self, texts: list[str], metadatas: list[dict[str, Any]]) -> list[str]:
        embeddings = await self._embedding_fn(texts)
        ids = []
        for text, embedding, metadata in zip(texts, embeddings, metadatas):
            doc_id = uuid.uuid4().hex
            self._documents[doc_id] = _Document(text=text, embedding=embedding, metadata=metadata)
            ids.append(doc_id)
        return ids

    async def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        if not self._documents:
            return []
        query_embedding = (await self._embedding_fn([query]))[0]
        scored: list[tuple[float, _Document]] = []
        for doc in self._documents.values():
            sim = _cosine_similarity(query_embedding, doc.embedding)
            scored.append((sim, doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            SearchResult(text=doc.text, score=sim, metadata=doc.metadata)
            for sim, doc in scored[:top_k]
        ]

    async def delete(self, ids: list[str]) -> None:
        for doc_id in ids:
            self._documents.pop(doc_id, None)
