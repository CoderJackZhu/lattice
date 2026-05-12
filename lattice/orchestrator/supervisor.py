from __future__ import annotations

import json
from typing import Any

from lattice.agent.agent import Agent
from lattice.agent.types import AgentResult
from lattice.tool.tool import Tool, ToolContext, ToolOutput, tool


class Supervisor:
    def __init__(
        self,
        coordinator: Agent,
        workers: dict[str, Agent],
        *,
        max_rounds: int = 10,
    ) -> None:
        self._coordinator = coordinator
        self._workers = workers
        self._max_rounds = max_rounds
        self._results: dict[str, str] = {}

        delegate_tool = self._make_delegate_tool()
        self._coordinator.tools = [*self._coordinator.tools, delegate_tool]

    def _make_delegate_tool(self) -> Tool:
        worker_names = list(self._workers.keys())
        supervisor = self

        async def delegate_task(worker: str, task: str) -> str:
            """Delegate a task to a named worker agent. Available workers: """ + ", ".join(worker_names)
            agent = supervisor._workers.get(worker)
            if not agent:
                return f"Error: worker '{worker}' not found. Available: {', '.join(worker_names)}"
            result = await agent.run(task)
            supervisor._results[worker] = result.output
            return result.output

        return tool(delegate_task, description=delegate_task.__doc__ or "Delegate a task to a worker agent")

    async def run(self, input: str) -> AgentResult:
        result = await self._coordinator.run(input)
        return result
