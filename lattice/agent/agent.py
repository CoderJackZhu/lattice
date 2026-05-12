from __future__ import annotations

import copy
import time
from collections.abc import AsyncIterator
from typing import Callable

from lattice.agent.strategy import ReActStrategy, Strategy
from lattice.agent.types import (
    AgentContext,
    AgentEvent,
    AgentResult,
    Continue,
    Finish,
    StepResult,
)
from lattice.llm.provider import registry
from lattice.llm.types import Message, TextContent, Usage
from lattice.tool.tool import Tool


class Agent:
    def __init__(
        self,
        name: str,
        *,
        model: str = "openai:gpt-4o-mini",
        system_prompt: str | Callable[[], str] = "",
        tools: list[Tool] | None = None,
        strategy: Strategy | None = None,
        memory: object | None = None,
        max_steps: int = 50,
        max_tokens_per_step: int = 4096,
    ) -> None:
        self.name = name
        self.model = model
        self.system_prompt = system_prompt
        self.tools = tools or []
        self.strategy = strategy or ReActStrategy()
        self.memory = memory
        self.max_steps = max_steps
        self.max_tokens_per_step = max_tokens_per_step
        self._messages: list[Message] = []
        self._started = False
        self._provider, self._model_name = registry.from_model_id(model)

    def _get_system_prompt(self) -> str:
        if callable(self.system_prompt):
            return self.system_prompt()
        return self.system_prompt

    async def start(self, input: str | Message) -> None:
        self._messages = []
        self._started = True

        if isinstance(input, str):
            user_msg = Message(role="user", content=[TextContent(text=input)])
        else:
            user_msg = input

        self._messages.append(user_msg)

    async def step(self) -> StepResult:
        assert self._started, "Must call start() before step()"

        ctx = AgentContext(
            messages=self._messages,
            tools=self.tools,
            model=self._model_name,
            system_prompt=self._get_system_prompt(),
            step_count=0,
            max_steps=self.max_steps,
            stream_fn=self._provider.stream,
        )

        result = await self.strategy.step(ctx)
        self._messages.extend(result.messages)
        return result

    async def run(self, input: str | Message) -> AgentResult:
        await self.start(input)

        steps: list[StepResult] = []
        total_usage = Usage()
        output = ""

        for step_count in range(self.max_steps):
            step_result = await self.step()
            steps.append(step_result)

            if isinstance(step_result.action, Finish):
                output = step_result.action.output
                break
        else:
            output = "Reached maximum number of steps"

        return AgentResult(
            output=output,
            messages=self._messages,
            steps=steps,
            usage=total_usage,
            trace_id="",
        )

    async def run_stream(self, input: str | Message) -> AsyncIterator[AgentEvent]:
        await self.start(input)

        yield AgentEvent(type="agent_start", data={"name": self.name}, timestamp=time.time())

        for step_count in range(self.max_steps):
            yield AgentEvent(type="step_start", data={"step": step_count}, timestamp=time.time())
            step_result = await self.step()
            yield AgentEvent(type="step_end", data={"step": step_count}, timestamp=time.time())

            if isinstance(step_result.action, Finish):
                break

        yield AgentEvent(type="agent_end", data={"name": self.name}, timestamp=time.time())

    def clone(self) -> Agent:
        new = copy.copy(self)
        new._messages = []
        new._started = False
        return new
