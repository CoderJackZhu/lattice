from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Union


# --- Content subtypes ---

@dataclass
class TextContent:
    text: str = ""
    type: Literal["text"] = "text"


@dataclass
class ImageContent:
    url: str = ""
    media_type: str = "image/png"
    type: Literal["image"] = "image"


@dataclass
class ThinkingContent:
    text: str = ""
    type: Literal["thinking"] = "thinking"


@dataclass
class ToolCall:
    id: str = ""
    name: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    type: Literal["tool_call"] = "tool_call"


@dataclass
class ToolResult:
    tool_call_id: str = ""
    content: str | list[TextContent | ImageContent] = ""
    is_error: bool = False
    type: Literal["tool_result"] = "tool_result"


Content = Union[TextContent, ImageContent, ThinkingContent, ToolCall, ToolResult]


# --- Core message and response types ---

@dataclass
class Message:
    role: Literal["user", "assistant", "tool"]
    content: list[Content] = field(default_factory=list)


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0


@dataclass
class ModelResponse:
    message: Message = field(default_factory=lambda: Message(role="assistant"))
    usage: Usage = field(default_factory=Usage)
    stop_reason: str = ""
    model: str = ""
    latency_ms: float = 0.0


@dataclass
class ToolSchema:
    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)


# --- Stream events ---

@dataclass
class StreamStart:
    model: str = ""
    type: Literal["stream_start"] = "stream_start"


@dataclass
class TextDelta:
    text: str = ""
    type: Literal["text_delta"] = "text_delta"


@dataclass
class ThinkingDelta:
    text: str = ""
    type: Literal["thinking_delta"] = "thinking_delta"


@dataclass
class ToolCallStart:
    tool_call_id: str = ""
    name: str = ""
    type: Literal["tool_call_start"] = "tool_call_start"


@dataclass
class ToolCallDelta:
    tool_call_id: str = ""
    arguments_fragment: str = ""
    type: Literal["tool_call_delta"] = "tool_call_delta"


@dataclass
class ToolCallEnd:
    tool_call_id: str = ""
    name: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    type: Literal["tool_call_end"] = "tool_call_end"


@dataclass
class StreamEnd:
    response: ModelResponse = field(default_factory=ModelResponse)
    type: Literal["stream_end"] = "stream_end"


@dataclass
class StreamError:
    error: str = ""
    type: Literal["stream_error"] = "stream_error"


StreamEvent = Union[
    StreamStart,
    TextDelta,
    ThinkingDelta,
    ToolCallStart,
    ToolCallDelta,
    ToolCallEnd,
    StreamEnd,
    StreamError,
]
