from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Any, Literal, Protocol

from pydantic import BaseModel

from lattice.tool.tool import ToolContext, ToolOutput

ToolExecutor = Callable[[ToolContext, BaseModel], Awaitable[ToolOutput]]


class ToolMiddleware(Protocol):
    async def __call__(
        self, ctx: ToolContext, params: BaseModel, next: ToolExecutor
    ) -> ToolOutput: ...


class RetryMiddleware:
    def __init__(self, max_retries: int = 3, backoff: float = 1.0) -> None:
        self._max_retries = max_retries
        self._backoff = backoff

    async def __call__(
        self, ctx: ToolContext, params: BaseModel, next: ToolExecutor
    ) -> ToolOutput:
        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                return await next(ctx, params)
            except Exception as e:
                last_error = e
                if attempt < self._max_retries:
                    await asyncio.sleep(self._backoff * (2 ** attempt))
        return ToolOutput(content=f"Error after {self._max_retries} retries: {last_error}")


class CacheMiddleware:
    def __init__(self, ttl: int = 300) -> None:
        self._ttl = ttl
        self._cache: dict[str, tuple[ToolOutput, float]] = {}

    async def __call__(
        self, ctx: ToolContext, params: BaseModel, next: ToolExecutor
    ) -> ToolOutput:
        key = f"{params.__class__.__name__}:{params.model_dump_json()}"
        if key in self._cache:
            output, timestamp = self._cache[key]
            if time.time() - timestamp < self._ttl:
                return output
            del self._cache[key]
        result = await next(ctx, params)
        self._cache[key] = (result, time.time())
        return result


class TimeoutMiddleware:
    def __init__(self, timeout: float = 30.0) -> None:
        self._timeout = timeout

    async def __call__(
        self, ctx: ToolContext, params: BaseModel, next: ToolExecutor
    ) -> ToolOutput:
        try:
            return await asyncio.wait_for(next(ctx, params), self._timeout)
        except asyncio.TimeoutError:
            return ToolOutput(content=f"Error: tool execution timed out after {self._timeout}s")


class SandboxMiddleware:
    def __init__(self, sandbox_type: Literal["docker", "subprocess"] = "subprocess") -> None:
        self._sandbox_type = sandbox_type

    async def __call__(
        self, ctx: ToolContext, params: BaseModel, next: ToolExecutor
    ) -> ToolOutput:
        return await next(ctx, params)
