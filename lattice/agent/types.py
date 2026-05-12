from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, Union

from lattice.llm.types import Message, StreamEvent, Usage
from lattice.tool.tool import Tool


@dataclass
class AgentContext:
    messages: list[Message]
    tools: list[Tool]
    model: str
    system_prompt: str
    memory_context: list[Any] = field(default_factory=list)
    step_count: int = 0
    max_steps: int = 50
    stream_fn: Callable[..., AsyncIterator[StreamEvent]] | None = None


@dataclass
class AgentEvent:
    type: Literal[
        "agent_start", "agent_end",
        "step_start", "step_end",
        "llm_start", "llm_delta", "llm_end",
        "tool_start", "tool_end",
        "memory_retrieve", "memory_store",
        "error",
    ]
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0


@dataclass
class Continue:
    pass


@dataclass
class Finish:
    output: str = ""


Action = Union[Continue, Finish]


@dataclass
class StepResult:
    messages: list[Message] = field(default_factory=list)
    action: Action = field(default_factory=Continue)


@dataclass
class AgentResult:
    output: str = ""
    messages: list[Message] = field(default_factory=list)
    steps: list[StepResult] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    trace_id: str = ""


class ReflectionJudge(Protocol):
    async def judge(self, output: str, goal: str, context: list[Message]) -> ReflectionVerdict: ...


@dataclass
class ReflectionVerdict:
    passed: bool = False
    feedback: str = ""
    score: float = 0.0
