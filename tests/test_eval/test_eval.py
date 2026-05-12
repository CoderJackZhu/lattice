from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest

from lattice.agent.agent import Agent
from lattice.eval.types import EvalCase, EvalResult, EvalReport
from lattice.eval.evaluators import ExactMatch, Contains, ToolUseEvaluator
from lattice.eval.runner import EvalRunner
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


@pytest.fixture(autouse=True)
def register_mock():
    registry.register("mock", lambda: MockProvider())
    yield
    registry._factories.pop("mock", None)
    for k in list(registry._instances):
        if k.startswith("mock"):
            del registry._instances[k]


# --- ExactMatch tests ---

async def test_exact_match_pass():
    ev = ExactMatch()
    result = await ev.evaluate("42", "42", case=EvalCase(input="q"))
    assert result.passed
    assert result.score == 1.0


async def test_exact_match_fail():
    ev = ExactMatch()
    result = await ev.evaluate("41", "42", case=EvalCase(input="q"))
    assert not result.passed
    assert result.score == 0.0


async def test_exact_match_strips_whitespace():
    ev = ExactMatch()
    result = await ev.evaluate("  42  ", "42", case=EvalCase(input="q"))
    assert result.passed


# --- Contains tests ---

async def test_contains_pass():
    ev = Contains()
    result = await ev.evaluate("The answer is 42!", "42", case=EvalCase(input="q"))
    assert result.passed


async def test_contains_case_insensitive():
    ev = Contains(case_sensitive=False)
    result = await ev.evaluate("Hello World", "hello", case=EvalCase(input="q"))
    assert result.passed


async def test_contains_case_sensitive():
    ev = Contains(case_sensitive=True)
    result = await ev.evaluate("Hello World", "hello", case=EvalCase(input="q"))
    assert not result.passed


async def test_contains_fail():
    ev = Contains()
    result = await ev.evaluate("no match here", "xyz", case=EvalCase(input="q"))
    assert not result.passed


# --- ToolUseEvaluator tests ---

async def test_tool_use_all_present():
    ev = ToolUseEvaluator(required_tools=["shell", "read_file"])
    messages = [
        Message(role="assistant", content=[
            ToolCall(id="1", name="shell", arguments={}),
            ToolCall(id="2", name="read_file", arguments={}),
        ]),
    ]
    result = await ev.evaluate("output", "", case=EvalCase(input="q"), messages=messages)
    assert result.passed
    assert result.score == 1.0


async def test_tool_use_missing():
    ev = ToolUseEvaluator(required_tools=["shell", "read_file"])
    messages = [
        Message(role="assistant", content=[
            ToolCall(id="1", name="shell", arguments={}),
        ]),
    ]
    result = await ev.evaluate("output", "", case=EvalCase(input="q"), messages=messages)
    assert not result.passed
    assert result.score == pytest.approx(0.5)


# --- EvalReport tests ---

def test_eval_report_summary():
    report = EvalReport(
        results=[
            EvalResult(case=EvalCase(input="q1"), output="a", score=1.0, passed=True),
            EvalResult(case=EvalCase(input="q2"), output="b", score=0.0, passed=False),
        ],
        total=2,
        passed=1,
        failed=1,
        avg_score=0.5,
        avg_latency_ms=100,
    )
    summary = report.summary()
    assert "1/2 passed" in summary
    assert "50.00%" in summary


# --- EvalRunner tests ---

async def test_eval_runner_basic():
    provider = MockProvider(responses=[
        _text_events("42"),
        _text_events("50"),
    ])
    registry._instances["mock"] = provider

    agent = Agent(name="test", model="mock:m")
    runner = EvalRunner(agent=agent, evaluator=ExactMatch())

    cases = [
        EvalCase(input="q1", expected="42"),
        EvalCase(input="q2", expected="50"),
    ]
    report = await runner.run(cases)
    assert report.total == 2
    assert report.passed == 2
    assert report.avg_score == 1.0


async def test_eval_runner_mixed():
    provider = MockProvider(responses=[
        _text_events("42"),
        _text_events("wrong"),
    ])
    registry._instances["mock"] = provider

    agent = Agent(name="test", model="mock:m")
    runner = EvalRunner(agent=agent, evaluator=ExactMatch())

    cases = [
        EvalCase(input="q1", expected="42"),
        EvalCase(input="q2", expected="50"),
    ]
    report = await runner.run(cases)
    assert report.total == 2
    assert report.passed == 1
    assert report.failed == 1
