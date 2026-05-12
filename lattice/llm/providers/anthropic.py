from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from typing import Any

from pydantic import BaseModel

from lattice.llm.types import (
    ImageContent,
    Message,
    ModelResponse,
    StreamEnd,
    StreamError,
    StreamEvent,
    StreamStart,
    TextContent,
    TextDelta,
    ThinkingContent,
    ThinkingDelta,
    ToolCall,
    ToolCallDelta,
    ToolCallEnd,
    ToolCallStart,
    ToolResult,
    ToolSchema,
    Usage,
)


def _to_anthropic_messages(messages: list[Message]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []

    for msg in messages:
        if msg.role == "user":
            parts: list[dict[str, Any]] = []
            for c in msg.content:
                if isinstance(c, TextContent):
                    parts.append({"type": "text", "text": c.text})
                elif isinstance(c, ImageContent):
                    parts.append({
                        "type": "image",
                        "source": {"type": "url", "url": c.url},
                    })
            result.append({"role": "user", "content": parts})

        elif msg.role == "assistant":
            parts = []
            for c in msg.content:
                if isinstance(c, TextContent):
                    parts.append({"type": "text", "text": c.text})
                elif isinstance(c, ThinkingContent):
                    parts.append({"type": "thinking", "thinking": c.text})
                elif isinstance(c, ToolCall):
                    parts.append({
                        "type": "tool_use",
                        "id": c.id,
                        "name": c.name,
                        "input": c.arguments,
                    })
            result.append({"role": "assistant", "content": parts})

        elif msg.role == "tool":
            parts = []
            for c in msg.content:
                if isinstance(c, ToolResult):
                    content_val: str | list[dict[str, Any]]
                    if isinstance(c.content, str):
                        content_val = c.content
                    else:
                        content_val = [
                            {"type": "text", "text": p.text}
                            for p in c.content
                            if isinstance(p, TextContent)
                        ]
                    parts.append({
                        "type": "tool_result",
                        "tool_use_id": c.tool_call_id,
                        "content": content_val,
                        "is_error": c.is_error,
                    })
            result.append({"role": "user", "content": parts})

    return result


def _to_anthropic_tools(tools: list[ToolSchema]) -> list[dict[str, Any]]:
    return [
        {
            "name": t.name,
            "description": t.description,
            "input_schema": t.parameters,
        }
        for t in tools
    ]


class AnthropicProvider:
    def __init__(self, api_key: str | None = None) -> None:
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "anthropic package is required. Install with: pip install 'lattice[anthropic]'"
            )
        kwargs: dict[str, Any] = {}
        if api_key:
            kwargs["api_key"] = api_key
        self._client = anthropic.AsyncAnthropic(**kwargs)

    async def stream(
        self,
        model: str,
        messages: list[Message],
        *,
        system: str | None = None,
        tools: list[ToolSchema] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        stop: list[str] | None = None,
        response_format: type[BaseModel] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        ant_messages = _to_anthropic_messages(messages)

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": ant_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = _to_anthropic_tools(tools)
        if stop:
            kwargs["stop_sequences"] = stop

        start_time = time.time()
        yield StreamStart(model=model)

        text_acc = ""
        tool_calls_acc: dict[int, dict[str, Any]] = {}
        usage = Usage()
        stop_reason = ""

        try:
            async with self._client.messages.stream(**kwargs) as stream:
                current_block_idx = -1
                current_block_type = ""

                async for event in stream:
                    event_type = event.type

                    if event_type == "message_start":
                        if hasattr(event, "message") and hasattr(event.message, "usage"):
                            u = event.message.usage
                            usage.input_tokens = getattr(u, "input_tokens", 0)

                    elif event_type == "message_delta":
                        if hasattr(event, "usage") and event.usage:
                            usage.output_tokens = getattr(event.usage, "output_tokens", 0)
                        if hasattr(event, "delta") and hasattr(event.delta, "stop_reason"):
                            stop_reason = event.delta.stop_reason or ""

                    elif event_type == "content_block_start":
                        current_block_idx = event.index
                        block = event.content_block
                        current_block_type = block.type
                        if block.type == "tool_use":
                            tool_calls_acc[current_block_idx] = {
                                "id": block.id,
                                "name": block.name,
                                "arguments": "",
                            }
                            yield ToolCallStart(
                                tool_call_id=block.id,
                                name=block.name,
                            )

                    elif event_type == "content_block_delta":
                        delta = event.delta
                        if delta.type == "text_delta":
                            text_acc += delta.text
                            yield TextDelta(text=delta.text)
                        elif delta.type == "thinking_delta":
                            yield ThinkingDelta(text=delta.thinking)
                        elif delta.type == "input_json_delta":
                            idx = current_block_idx
                            if idx in tool_calls_acc:
                                tool_calls_acc[idx]["arguments"] += delta.partial_json
                                yield ToolCallDelta(
                                    tool_call_id=tool_calls_acc[idx]["id"],
                                    arguments_fragment=delta.partial_json,
                                )

                    elif event_type == "content_block_stop":
                        if current_block_type == "tool_use" and current_block_idx in tool_calls_acc:
                            tc = tool_calls_acc[current_block_idx]
                            try:
                                args = json.loads(tc["arguments"]) if tc["arguments"] else {}
                            except json.JSONDecodeError:
                                args = {}
                            tc["parsed_args"] = args
                            yield ToolCallEnd(
                                tool_call_id=tc["id"],
                                name=tc["name"],
                                arguments=args,
                            )

        except Exception as e:
            yield StreamError(error=str(e))
            return

        # Build final message
        content_list: list[Any] = []
        if text_acc:
            content_list.append(TextContent(text=text_acc))
        for idx in sorted(tool_calls_acc):
            tc = tool_calls_acc[idx]
            args = tc.get("parsed_args", {})
            content_list.append(ToolCall(id=tc["id"], name=tc["name"], arguments=args))

        latency_ms = (time.time() - start_time) * 1000
        stop_reason_map = {"end_turn": "end_turn", "tool_use": "tool_use", "max_tokens": "max_tokens"}
        mapped_reason = stop_reason_map.get(stop_reason, stop_reason)

        final_message = Message(role="assistant", content=content_list)
        model_response = ModelResponse(
            message=final_message,
            usage=usage,
            stop_reason=mapped_reason,
            model=model,
            latency_ms=latency_ms,
        )
        yield StreamEnd(response=model_response)

    async def complete(
        self,
        model: str,
        messages: list[Message],
        *,
        system: str | None = None,
        tools: list[ToolSchema] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        stop: list[str] | None = None,
        response_format: type[BaseModel] | None = None,
    ) -> ModelResponse:
        last_response: ModelResponse | None = None
        async for event in self.stream(
            model, messages, system=system, tools=tools,
            temperature=temperature, max_tokens=max_tokens,
            stop=stop, response_format=response_format,
        ):
            if isinstance(event, StreamEnd):
                last_response = event.response
        if last_response is None:
            raise RuntimeError("Stream ended without a StreamEnd event")
        return last_response
