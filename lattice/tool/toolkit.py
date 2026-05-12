from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import BaseModel

from lattice.llm.types import ToolSchema
from lattice.tool.tool import Tool, ToolContext, ToolOutput


class ToolKit:
    def __init__(self, name: str, tools: list[Tool], middleware: list[Any] | None = None) -> None:
        self._name = name
        self._tools = {t.name: t for t in tools}
        self._middleware = middleware or []

    async def execute(self, tool_name: str, params: BaseModel, ctx: ToolContext) -> ToolOutput:
        tool = self._tools.get(tool_name)
        if not tool:
            return ToolOutput(content=f"Error: tool '{tool_name}' not found")

        async def base_executor(c: ToolContext, p: BaseModel) -> ToolOutput:
            return await tool.execute(p, c)

        executor: Callable[[ToolContext, BaseModel], Awaitable[ToolOutput]] = base_executor

        for mw in reversed(self._middleware):
            prev = executor

            async def wrapped(
                c: ToolContext, p: BaseModel, _mw: Any = mw, _next: Any = prev
            ) -> ToolOutput:
                return await _mw(c, p, _next)

            executor = wrapped

        return await executor(ctx, params)

    def get_tools(self) -> list[Tool]:
        return list(self._tools.values())

    def get_schemas(self) -> list[ToolSchema]:
        return [t.to_schema() for t in self._tools.values()]
