from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest

from lattice.agent.strategy import PlanAndExecuteStrategy, ReflexionStrategy
from lattice.agent.types import Continue, Finish, ReflectionVerdict, StepResult
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
from lattice.planner import Plan, PlanStep, StaticPlanner
from lattice.tool.tool import tool, ToolContext


# --- Mock Provider ---

class MockProvider:
    def __init__(self, responses: list[list[StreamEvent]] | None = None) -> None:
        self._responses = responses or []
        self._call_count = 0

    async def stream(
        self, model: str, messages: list[Message], **kwargs: Any,
    ) -> AsyncIterator[StreamEvent]:
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
    resp = ModelResponse(message=msg, usage=Usage(10, 5), stop_reason="end_turn", model="mock")
    return [StreamStart(model="mock"), StreamEnd(response=resp)]


def _tool_call_events(tool_name: str, tool_id: str, arguments: dict) -> list[StreamEvent]:
    msg = Message(role="assistant", content=[ToolCall(id=tool_id, name=tool_name, arguments=arguments)])
    resp = ModelResponse(message=msg, usage=Usage(10, 5), stop_reason="tool_use", model="mock")
    return [StreamStart(model="mock"), StreamEnd(response=resp)]


@pytest.fixture(autouse=True)
def register_mock():
    registry.register("mock", lambda: MockProvider())
    yield
    registry._factories.pop("mock", None)
    for k in list(registry._instances):
        if k.startswith("mock"):
            del registry._instances[k]


# --- Plan tests ---

def test_plan_ready_steps():
    plan = Plan(goal="test", steps=[
        PlanStep(id="s1", description="first", status="done"),
        PlanStep(id="s2", description="second", dependencies=["s1"]),
        PlanStep(id="s3", description="third", dependencies=["s2"]),
    ])
    ready = plan.ready_steps()
    assert len(ready) == 1
    assert ready[0].id == "s2"


def test_plan_ready_steps_parallel():
    plan = Plan(goal="test", steps=[
        PlanStep(id="s1", description="first", status="done"),
        PlanStep(id="s2", description="a", dependencies=["s1"]),
        PlanStep(id="s3", description="b", dependencies=["s1"]),
    ])
    ready = plan.ready_steps()
    assert len(ready) == 2


def test_plan_ready_steps_none_ready():
    plan = Plan(goal="test", steps=[
        PlanStep(id="s1", description="first", dependencies=["s2"]),
        PlanStep(id="s2", description="second", dependencies=["s1"]),
    ])
    assert plan.ready_steps() == []


# --- StaticPlanner tests ---

async def test_static_planner():
    planner = StaticPlanner(steps=["step one", "step two", "step three"])
    plan = await planner.plan("do stuff", context=None)  # type: ignore
    assert len(plan.steps) == 3
    assert plan.steps[0].dependencies == []
    assert plan.steps[1].dependencies == ["step_1"]
    assert plan.steps[2].dependencies == ["step_2"]


async def test_static_planner_replan():
    planner = StaticPlanner(steps=["a"])
    plan = await planner.plan("goal", context=None)  # type: ignore
    replanned = await planner.replan(plan, "fail", context=None)  # type: ignore
    assert replanned is plan


# --- ReflexionStrategy tests ---

class MockJudge:
    def __init__(self, verdicts: list[ReflectionVerdict]) -> None:
        self._verdicts = verdicts
        self._call_count = 0

    async def judge(self, output: str, goal: str, context: list[Message]) -> ReflectionVerdict:
        v = self._verdicts[min(self._call_count, len(self._verdicts) - 1)]
        self._call_count += 1
        return v


async def test_reflexion_passes():
    provider = MockProvider(responses=[_text_events("good answer")])
    registry._instances["mock"] = provider

    judge = MockJudge([ReflectionVerdict(passed=True, feedback="ok", score=1.0)])
    strategy = ReflexionStrategy(judge=judge, max_reflections=3)

    from lattice.agent.agent import Agent
    agent = Agent(name="test", model="mock:m", strategy=strategy)
    result = await agent.run("question")
    assert result.output == "good answer"


async def test_reflexion_retries_then_passes():
    provider = MockProvider(responses=[
        _text_events("bad answer"),
        _text_events("better answer"),
    ])
    registry._instances["mock"] = provider

    judge = MockJudge([
        ReflectionVerdict(passed=False, feedback="try again", score=0.2),
        ReflectionVerdict(passed=True, feedback="ok", score=0.9),
    ])
    strategy = ReflexionStrategy(judge=judge, max_reflections=3)

    from lattice.agent.agent import Agent
    agent = Agent(name="test", model="mock:m", strategy=strategy)
    result = await agent.run("question")
    assert result.output == "better answer"
    assert len(result.steps) >= 2


async def test_reflexion_max_reflections():
    provider = MockProvider(responses=[_text_events("always bad")])
    registry._instances["mock"] = provider

    judge = MockJudge([ReflectionVerdict(passed=False, feedback="nope", score=0.1)])
    strategy = ReflexionStrategy(judge=judge, max_reflections=2)

    from lattice.agent.agent import Agent
    agent = Agent(name="test", model="mock:m", strategy=strategy, max_steps=20)
    result = await agent.run("question")
    assert result.output == "always bad"


# --- Middleware tests ---

async def test_retry_middleware():
    from lattice.tool.middleware import RetryMiddleware
    from lattice.tool.tool import ToolOutput

    call_count = 0

    async def failing_executor(ctx, params):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise RuntimeError("fail")
        return ToolOutput(content="success")

    mw = RetryMiddleware(max_retries=3, backoff=0.01)
    result = await mw(ToolContext(), None, failing_executor)  # type: ignore
    assert result.content == "success"
    assert call_count == 3


async def test_cache_middleware():
    from pydantic import BaseModel
    from lattice.tool.middleware import CacheMiddleware
    from lattice.tool.tool import ToolOutput

    call_count = 0

    class Params(BaseModel):
        x: int = 1

    async def executor(ctx, params):
        nonlocal call_count
        call_count += 1
        return ToolOutput(content=f"result-{call_count}")

    mw = CacheMiddleware(ttl=300)
    r1 = await mw(ToolContext(), Params(x=1), executor)
    r2 = await mw(ToolContext(), Params(x=1), executor)
    assert r1.content == r2.content
    assert call_count == 1


async def test_timeout_middleware():
    import asyncio
    from lattice.tool.middleware import TimeoutMiddleware
    from lattice.tool.tool import ToolOutput

    async def slow_executor(ctx, params):
        await asyncio.sleep(10)
        return ToolOutput(content="done")

    mw = TimeoutMiddleware(timeout=0.05)
    result = await mw(ToolContext(), None, slow_executor)  # type: ignore
    assert "timed out" in result.content
