from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

EmbeddingFn = Callable[[list[str]], Awaitable[list[list[float]]]]


@dataclass
class MemoryItem:
    content: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    score: float = 0.0
    timestamp: float = 0.0
    source: str = ""


@runtime_checkable
class Memory(Protocol):
    async def retrieve(self, query: str, top_k: int = 5) -> list[MemoryItem]: ...
    async def store(self, item: MemoryItem) -> None: ...
    async def clear(self) -> None: ...
