from __future__ import annotations

import copy
import time

from lattice.memory.base import MemoryItem


class WorkingMemory:
    def __init__(self, max_items: int = 20) -> None:
        self._items: list[MemoryItem] = []
        self._max_items = max_items

    async def store(self, item: MemoryItem) -> None:
        item.source = "working"
        if not item.timestamp:
            item.timestamp = time.time()
        self._items.append(item)
        while len(self._items) > self._max_items:
            self._items.pop(0)

    async def retrieve(self, query: str, top_k: int = 5) -> list[MemoryItem]:
        if not query:
            return self._items[-top_k:]

        keywords = query.lower().split()
        scored: list[tuple[float, MemoryItem]] = []
        for item in self._items:
            content_lower = item.content.lower()
            match_count = sum(1 for kw in keywords if kw in content_lower)
            if match_count > 0:
                scored.append((float(match_count), item))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for s, item in scored[:top_k]:
            item_copy = copy.copy(item)
            item_copy.score = s / len(keywords) if keywords else 0.0
            results.append(item_copy)
        return results

    async def clear(self) -> None:
        self._items.clear()
