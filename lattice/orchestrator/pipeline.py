from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from lattice.agent.agent import Agent


@dataclass
class PipelineResult:
    outputs: list[str] = field(default_factory=list)
    final_output: str = ""


class Pipeline:
    def __init__(self, agents: list[Agent], *, transform: Any | None = None) -> None:
        self._agents = agents
        self._transform = transform

    async def run(self, input: str) -> PipelineResult:
        outputs: list[str] = []
        current = input

        for agent in self._agents:
            if self._transform:
                current = self._transform(current, outputs)
            result = await agent.run(current)
            outputs.append(result.output)
            current = result.output

        return PipelineResult(outputs=outputs, final_output=outputs[-1] if outputs else "")
