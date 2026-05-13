from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from lattice.llm.types import Message, ToolSchema
from lattice.memory.base import MemoryItem


@dataclass
class PlanStep:
    id: str = ""
    description: str = ""
    dependencies: list[str] = field(default_factory=list)
    status: Literal["pending", "running", "done", "failed", "skipped"] = "pending"
    result: str | None = None


@dataclass
class Plan:
    goal: str = ""
    steps: list[PlanStep] = field(default_factory=list)

    def ready_steps(self) -> list[PlanStep]:
        resolved_ids = {s.id for s in self.steps if s.status in ("done", "skipped")}
        return [
            s for s in self.steps
            if s.status == "pending" and all(d in resolved_ids for d in s.dependencies)
        ]


@dataclass
class PlanContext:
    available_tools: list[ToolSchema] = field(default_factory=list)
    memory_context: list[MemoryItem] = field(default_factory=list)
    messages: list[Message] = field(default_factory=list)


class Planner(Protocol):
    async def plan(self, goal: str, context: PlanContext) -> Plan: ...
    async def replan(self, plan: Plan, feedback: str, context: PlanContext) -> Plan: ...
