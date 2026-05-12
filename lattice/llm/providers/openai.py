from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from typing import Any

import openai
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
    ToolCall,
    ToolCallDelta,
    ToolCallEnd,
    ToolCallStart,
    ToolResult,
    ToolSchema,
    Usage,
)


def _to_openai_messages(
    messages: list[Message], system: str | None = None
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    if system:
        result.append({"role": "system", "content": system})

    for msg in messages:
        if msg.role == "user":
            parts: list[dict[str, Any]] = []
            for c in msg.content:
                if isinstance(c, TextContent):
                    parts.append({"type": "text", "text": c.text})
                elif isinstance(c, ImageContent):
                    parts.append({
                        "type": "image_url",
                        "image_url": {"url": c.url},
                    })
            if len(parts) == 1 and parts[0]["type"] == "text":
                result.append({"role": "user", "content": parts[0]["text"]})
            else:
                result.append({"role": "user", "content": parts})

        elif msg.role == "assistant":
            text_parts = []
            tool_calls_out = []
            for c in msg.content:
                if isinstance(c, TextContent):
                    text_parts.append(c.text)
                elif isinstance(c, ThinkingContent):
                    pass
                elif isinstance(c, ToolCall):
                    tool_calls_out.append({
                        "id": c.id,
                        "type": "function",
                        "function": {
                            "name": c.name,
                            "arguments": json.dumps(c.arguments),
                        },
                    })
            entry: dict[str, Any] = {"role": "assistant"}
            content_text = "".join(text_parts)
            if content_text:
                entry["content"] = content_text
            else:
                entry["content"] = None
            if tool_calls_out:
                entry["tool_calls"] = tool_calls_out
            result.append(entry)

        elif msg.role == "tool":
            for c in msg.content:
                if isinstance(c, ToolResult):
                    content_str = c.content if isinstance(c.content, str) else json.dumps(
                        [{"type": "text", "text": p.text} for p in c.content if isinstance(p, TextContent)]
                    )
                    result.append({
                        "role": "tool",
                        "tool_call_id": c.tool_call_id,
                        "content": content_str,
                    })

    return result


def _to_openai_tools(tools: list[ToolSchema]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            },
        }
        for t in tools
    ]


class OpenAIProvider:
    def __init__(self, api_key: str | None = None, base_url: str | None = None) -> None:
        kwargs: dict[str, Any] = {}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        self._client = openai.AsyncOpenAI(**kwargs)

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
        oai_messages = _to_openai_messages(messages, system)

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": oai_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools:
            kwargs["tools"] = _to_openai_tools(tools)
        if stop:
            kwargs["stop"] = stop
        if response_format:
            schema = response_format.model_json_schema()
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": response_format.__name__, "schema": schema},
            }

        start_time = time.time()
        yield StreamStart(model=model)

        text_acc = ""
        tool_calls_acc: dict[int, dict[str, Any]] = {}
        usage = Usage()
        stop_reason = ""

        try:
            response = await self._client.chat.completions.create(**kwargs)
            async for chunk in response:
                if chunk.usage:
                    usage = Usage(
                        input_tokens=chunk.usage.prompt_tokens or 0,
                        output_tokens=chunk.usage.completion_tokens or 0,
                        cache_read_tokens=getattr(chunk.usage, "prompt_tokens_details", None)
                        and getattr(chunk.usage.prompt_tokens_details, "cached_tokens", 0)
                        or 0,
                    )

                if not chunk.choices:
                    continue

                choice = chunk.choices[0]
                if choice.finish_reason:
                    stop_reason = choice.finish_reason

                delta = choice.delta
                if delta.content:
                    text_acc += delta.content
                    yield TextDelta(text=delta.content)

                if delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        idx = tc_delta.index
                        if idx not in tool_calls_acc:
                            tool_calls_acc[idx] = {
                                "id": tc_delta.id or "",
                                "name": tc_delta.function.name if tc_delta.function and tc_delta.function.name else "",
                                "arguments": "",
                            }
                            yield ToolCallStart(
                                tool_call_id=tool_calls_acc[idx]["id"],
                                name=tool_calls_acc[idx]["name"],
                            )
                        if tc_delta.function and tc_delta.function.arguments:
                            tool_calls_acc[idx]["arguments"] += tc_delta.function.arguments
                            yield ToolCallDelta(
                                tool_call_id=tool_calls_acc[idx]["id"],
                                arguments_fragment=tc_delta.function.arguments,
                            )

        except Exception as e:
            yield StreamError(error=str(e))
            return

        # Emit ToolCallEnd for all accumulated tool calls
        parsed_tool_calls: list[ToolCall] = []
        for idx in sorted(tool_calls_acc):
            tc = tool_calls_acc[idx]
            try:
                args = json.loads(tc["arguments"]) if tc["arguments"] else {}
            except json.JSONDecodeError:
                args = {}
            parsed_tool_calls.append(ToolCall(id=tc["id"], name=tc["name"], arguments=args))
            yield ToolCallEnd(tool_call_id=tc["id"], name=tc["name"], arguments=args)

        # Build final message
        content_list: list[Any] = []
        if text_acc:
            content_list.append(TextContent(text=text_acc))
        content_list.extend(parsed_tool_calls)

        latency_ms = (time.time() - start_time) * 1000
        stop_reason_map = {"stop": "end_turn", "tool_calls": "tool_use", "length": "max_tokens"}
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
