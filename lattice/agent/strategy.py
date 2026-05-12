from __future__ import annotations

import json
from typing import Protocol

from pydantic import ValidationError

from lattice.agent.types import (
    AgentContext,
    Continue,
    Finish,
    StepResult,
)
from lattice.llm.types import (
    Message,
    StreamEnd,
    TextContent,
    ToolCall,
    ToolResult,
    ToolSchema,
)
from lattice.tool.tool import Tool, ToolContext, ToolOutput


class Strategy(Protocol):
    async def step(self, ctx: AgentContext) -> StepResult: ...


class ReActStrategy:
    async def step(self, ctx: AgentContext) -> StepResult:
        tool_schemas = [t.to_schema() for t in ctx.tools] if ctx.tools else None

        assert ctx.stream_fn is not None, "stream_fn must be set in AgentContext"

        response = None
        async for event in ctx.stream_fn(
            ctx.model,
            ctx.messages,
            system=ctx.system_prompt or None,
            tools=tool_schemas,
        ):
            if isinstance(event, StreamEnd):
                response = event.response

        if response is None:
            return StepResult(
                messages=[],
                action=Finish(output="Error: no response from LLM"),
            )

        assistant_msg = response.message
        tool_calls = [c for c in assistant_msg.content if isinstance(c, ToolCall)]

        if not tool_calls:
            text = "".join(
                c.text for c in assistant_msg.content if isinstance(c, TextContent)
            )
            return StepResult(
                messages=[assistant_msg],
                action=Finish(output=text),
            )

        # Execute tool calls
        tool_results: list[ToolResult] = []
        tools_by_name = {t.name: t for t in ctx.tools} if ctx.tools else {}

        for tc in tool_calls:
            tool_obj = tools_by_name.get(tc.name)
            if not tool_obj:
                tool_results.append(ToolResult(
                    tool_call_id=tc.id,
                    content=f"Error: tool '{tc.name}' not found",
                    is_error=True,
                ))
                continue

            try:
                validated = tool_obj.parameters(**tc.arguments)
            except (ValidationError, TypeError) as e:
                tool_results.append(ToolResult(
                    tool_call_id=tc.id,
                    content=f"Error validating parameters: {e}",
                    is_error=True,
                ))
                continue

            try:
                tool_ctx = ToolContext(tool_call_id=tc.id)
                output = await tool_obj.execute(validated, tool_ctx)
                content = output.content if isinstance(output.content, str) else json.dumps(output.content)
                tool_results.append(ToolResult(
                    tool_call_id=tc.id,
                    content=content,
                ))
            except Exception as e:
                tool_results.append(ToolResult(
                    tool_call_id=tc.id,
                    content=f"Error executing tool: {e}",
                    is_error=True,
                ))

        tool_msg = Message(role="tool", content=tool_results)  # type: ignore[arg-type]
        return StepResult(
            messages=[assistant_msg, tool_msg],
            action=Continue(),
        )


class PlanAndExecuteStrategy:
    def __init__(self, **kwargs: object) -> None:
        raise NotImplementedError("PlanAndExecuteStrategy is available in Phase 2")

    async def step(self, ctx: AgentContext) -> StepResult:
        raise NotImplementedError


class ReflexionStrategy:
    def __init__(self, **kwargs: object) -> None:
        raise NotImplementedError("ReflexionStrategy is available in Phase 2")

    async def step(self, ctx: AgentContext) -> StepResult:
        raise NotImplementedError
