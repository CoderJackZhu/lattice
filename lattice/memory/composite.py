from __future__ import annotations

import asyncio

from lattice.memory.base import Memory, MemoryItem


class CompositeMemory:
    def __init__(self, memories: list[tuple[Memory, float]]) -> None:
        if not memories:
            raise ValueError("CompositeMemory requires at least one memory source")
        self._memories = memories

    async def store(self, item: MemoryItem) -> None:
        primary = self._memories[0][0]
        await primary.store(item)

    async def retrieve(self, query: str, top_k: int = 5) -> list[MemoryItem]:
        tasks = [mem.retrieve(query, top_k) for mem, _ in self._memories]
        results_per_source = await asyncio.gather(*tasks)

        all_items: list[MemoryItem] = []
        for (_, weight), items in zip(self._memories, results_per_source):
            for item in items:
                item.score *= weight
                all_items.append(item)

        all_items.sort(key=lambda x: x.score, reverse=True)
        return all_items[:top_k]

    async def clear(self) -> None:
        await asyncio.gather(*[mem.clear() for mem, _ in self._memories])
