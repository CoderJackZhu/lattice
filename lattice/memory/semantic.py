from __future__ import annotations

import time

from lattice.memory.base import MemoryItem
from lattice.memory.stores.in_memory import VectorStore


class SemanticMemory:
    def __init__(self, backend: VectorStore) -> None:
        self._backend = backend

    async def store(self, item: MemoryItem) -> None:
        metadata = {
            **item.metadata,
            "source": "semantic",
            "timestamp": item.timestamp or time.time(),
        }
        await self._backend.add([item.content], [metadata])

    async def retrieve(self, query: str, top_k: int = 5) -> list[MemoryItem]:
        results = await self._backend.search(query, top_k)
        return [
            MemoryItem(
                content=r.text,
                score=r.score,
                metadata=r.metadata,
                source="semantic",
            )
            for r in results
        ]

    async def clear(self) -> None:
        pass
