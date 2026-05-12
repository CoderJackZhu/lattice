from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass, field
from typing import Any, Callable

from pydantic import BaseModel, create_model

from lattice.llm.types import ToolSchema
from lattice.trace.tracer import Span


@dataclass
class ToolOutput:
    content: str | list[Any] = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolContext:
    agent_name: str = ""
    tool_call_id: str = ""
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)
    trace_span: Span | None = None


class Tool:
    def __init__(
        self,
        name: str,
        description: str,
        parameters: type[BaseModel],
        execute_fn: Callable[..., Any],
    ) -> None:
        self.name = name
        self.description = description
        self.parameters = parameters
        self._execute_fn = execute_fn

    async def execute(self, params: BaseModel, ctx: ToolContext) -> ToolOutput:
        kwargs = params.model_dump()
        result = self._execute_fn(**kwargs)
        if inspect.isawaitable(result):
            result = await result
        if isinstance(result, ToolOutput):
            return result
        return ToolOutput(content=str(result) if result is not None else "")

    def to_schema(self) -> ToolSchema:
        schema = self.parameters.model_json_schema()
        schema.pop("title", None)
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters=schema,
        )


def tool(fn: Callable[..., Any] | None = None, *, description: str | None = None) -> Any:
    def decorator(func: Callable[..., Any]) -> Tool:
        sig = inspect.signature(func)
        hints = {}
        try:
            hints = func.__annotations__.copy()
        except AttributeError:
            pass
        hints.pop("return", None)

        fields: dict[str, Any] = {}
        for param_name, param in sig.parameters.items():
            if param_name in ("self", "cls"):
                continue
            annotation = hints.get(param_name, str)
            if param.default is inspect.Parameter.empty:
                fields[param_name] = (annotation, ...)
            else:
                fields[param_name] = (annotation, param.default)

        params_model = create_model(func.__name__ + "_params", **fields)

        tool_desc = description or func.__doc__ or func.__name__
        tool_desc = tool_desc.strip()

        return Tool(
            name=func.__name__,
            description=tool_desc,
            parameters=params_model,
            execute_fn=func,
        )

    if fn is not None:
        return decorator(fn)
    return decorator
