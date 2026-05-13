from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest

from lattice.agent.agent import Agent
from lattice.agent.types import Continue, Finish, StepResult
from lattice.llm.provider import registry
from lattice.llm.types import (
    Message,
    ModelResponse,
    StreamEnd,
    StreamEvent,
    StreamStart,
    TextContent,
    ToolCall,
    Usage,
)
from lattice.tool.tool import Tool, ToolContext, ToolOutput, tool


# --- Mock Provider ---

class MockProvider:
    def __init__(self, responses: list[list[StreamEvent]] | None = None) -> None:
        self._responses = responses or []
        self._call_count = 0
        self.calls: list[dict[str, Any]] = []

    async def stream(
        self,
        model: str,
        messages: list[Message],
        *,
        system: str | None = None,
        tools: Any = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        stop: Any = None,
        response_format: Any = None,
    ) -> AsyncIterator[StreamEvent]:
        self.calls.append({
            "model": model,
            "messages": messages,
            "system": system,
            "tools": tools,
            "max_tokens": max_tokens,
        })
        if self._call_count < len(self._responses):
            events = self._responses[self._call_count]
        else:
            events = self._responses[-1] if self._responses else []
        self._call_count += 1
        for event in events:
            yield event

    async def complete(self, model: str, messages: list[Message], **kwargs: Any) -> ModelResponse:
        async for event in self.stream(model, messages, **kwargs):
            if isinstance(event, StreamEnd):
                return event.response
        raise RuntimeError("No StreamEnd")


def _text_events(text: str) -> list[StreamEvent]:
    msg = Message(role="assistant", content=[TextContent(text=text)])
    response = ModelResponse(message=msg, usage=Usage(input_tokens=10, output_tokens=5), stop_reason="end_turn", model="mock")
    return [StreamStart(model="mock"), StreamEnd(response=response)]


def _tool_call_events(tool_name: str, tool_id: str, arguments: dict[str, Any]) -> list[StreamEvent]:
    msg = Message(role="assistant", content=[
        TextContent(text=""),
        ToolCall(id=tool_id, name=tool_name, arguments=arguments),
    ])
    response = ModelResponse(message=msg, usage=Usage(input_tokens=10, output_tokens=5), stop_reason="tool_use", model="mock")
    return [StreamStart(model="mock"), StreamEnd(response=response)]


@pytest.fixture(autouse=True)
def register_mock():
    registry.register("mock", lambda: MockProvider())
    yield
    if "mock" in registry._factories:
        del registry._factories["mock"]
    for key in list(registry._instances.keys()):
        if key.startswith("mock"):
            del registry._instances[key]


# --- Tests ---

async def test_agent_pure_text_response():
    provider = MockProvider(responses=[_text_events("Hello!")])
    registry._instances["mock"] = provider

    agent = Agent(name="test", model="mock:test-model")
    result = await agent.run("hi")

    assert result.output == "Hello!"
    assert len(result.steps) == 1
    assert isinstance(result.steps[0].action, Finish)


async def test_agent_tool_call_then_finish():
    @tool(description="echo tool")
    async def echo(msg: str) -> str:
        return f"echoed: {msg}"

    provider = MockProvider(responses=[
        _tool_call_events("echo", "tc_1", {"msg": "test"}),
        _text_events("Done! The echo returned: echoed: test"),
    ])
    registry._instances["mock"] = provider

    agent = Agent(name="test", model="mock:test-model", tools=[echo])
    result = await agent.run("echo something")

    assert len(result.steps) == 2
    assert isinstance(result.steps[0].action, Continue)
    assert isinstance(result.steps[1].action, Finish)
    assert "Done!" in result.output


async def test_agent_max_steps():
    provider = MockProvider(responses=[
        _tool_call_events("shell", "tc_1", {"command": "ls"}),
    ])
    registry._instances["mock"] = provider

    @tool(description="always called")
    async def shell(command: str) -> str:
        return "output"

    agent = Agent(name="test", model="mock:test-model", tools=[shell], max_steps=3)
    result = await agent.run("loop forever")

    assert len(result.steps) == 3
    assert "maximum" in result.output.lower()


async def test_agent_tool_not_found():
    provider = MockProvider(responses=[
        _tool_call_events("nonexistent", "tc_1", {}),
        _text_events("Sorry, tool not found"),
    ])
    registry._instances["mock"] = provider

    agent = Agent(name="test", model="mock:test-model", tools=[])
    result = await agent.run("use tool")

    assert len(result.steps) == 2


async def test_agent_clone():
    agent = Agent(name="original", model="mock:test-model", system_prompt="test")
    agent._messages = [Message(role="user", content=[TextContent(text="hi")])]
    agent._started = True

    cloned = agent.clone()
    assert cloned.name == "original"
    assert cloned._messages == []
    assert not cloned._started
    assert cloned.system_prompt == "test"


async def test_agent_start_step():
    provider = MockProvider(responses=[_text_events("response")])
    registry._instances["mock"] = provider

    agent = Agent(name="test", model="mock:test-model")
    await agent.start("hello")
    assert agent._started
    assert len(agent._messages) == 1

    step = await agent.step()
    assert isinstance(step.action, Finish)
    assert step.action.output == "response"


async def test_agent_passes_step_count_and_token_limit_to_strategy():
    provider = MockProvider(responses=[_text_events("response")])
    registry._instances["mock"] = provider

    seen_step_counts: list[int] = []

    class RecordingStrategy:
        async def step(self, ctx) -> StepResult:
            seen_step_counts.append(ctx.step_count)
            async for event in ctx.stream_fn(
                ctx.model,
                ctx.messages,
                max_tokens=ctx.max_tokens_per_step,
            ):
                if isinstance(event, StreamEnd):
                    return StepResult(messages=[event.response.message], action=Finish("done"))
            return StepResult(action=Finish("done"))

    agent = Agent(
        name="test",
        model="mock:test-model",
        strategy=RecordingStrategy(),
        max_tokens_per_step=123,
    )
    await agent.start("hello")
    await agent.step()

    assert seen_step_counts == [0]
    assert provider.calls[0]["max_tokens"] == 123
