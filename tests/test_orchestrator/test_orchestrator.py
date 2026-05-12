from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest

from lattice.agent.agent import Agent
from lattice.llm.provider import registry
from lattice.llm.types import (
    Message,
    ModelResponse,
    StreamEnd,
    StreamEvent,
    StreamStart,
    TextContent,
    ToolCall,
    ToolResult,
    Usage,
)
from lattice.orchestrator.pipeline import Pipeline, PipelineResult
from lattice.orchestrator.graph import Graph, GraphResult, Node, Edge
from lattice.orchestrator.supervisor import Supervisor


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


# --- Pipeline tests ---

async def test_pipeline_basic():
    provider = MockProvider(responses=[
        _text_events("step 1 output"),
        _text_events("step 2 output"),
    ])
    registry._instances["mock"] = provider

    a1 = Agent(name="a1", model="mock:m")
    a2 = Agent(name="a2", model="mock:m")
    pipeline = Pipeline(agents=[a1, a2])

    result = await pipeline.run("input")
    assert len(result.outputs) == 2
    assert result.outputs[0] == "step 1 output"
    assert result.final_output == "step 2 output"


async def test_pipeline_single_agent():
    provider = MockProvider(responses=[_text_events("only output")])
    registry._instances["mock"] = provider

    agent = Agent(name="solo", model="mock:m")
    pipeline = Pipeline(agents=[agent])

    result = await pipeline.run("input")
    assert result.final_output == "only output"
    assert len(result.outputs) == 1


async def test_pipeline_with_transform():
    provider = MockProvider(responses=[
        _text_events("raw data"),
        _text_events("processed"),
    ])
    registry._instances["mock"] = provider

    a1 = Agent(name="a1", model="mock:m")
    a2 = Agent(name="a2", model="mock:m")
    transformed = False

    def my_transform(current, outputs):
        nonlocal transformed
        if outputs:
            transformed = True
        return current

    pipeline = Pipeline(agents=[a1, a2], transform=my_transform)
    result = await pipeline.run("input")
    assert transformed
    assert result.final_output == "processed"


# --- Graph tests ---

async def test_graph_linear():
    provider = MockProvider(responses=[
        _text_events("node A done"),
        _text_events("node B done"),
    ])
    registry._instances["mock"] = provider

    a1 = Agent(name="a1", model="mock:m")
    a2 = Agent(name="a2", model="mock:m")

    graph = Graph(
        nodes=[
            Node(id="a", agent=a1),
            Node(id="b", agent=a2, dependencies=["a"]),
        ],
        output_node="b",
    )

    result = await graph.run("input")
    assert result.outputs["a"] == "node A done"
    assert result.final_output == "node B done"


async def test_graph_parallel():
    provider = MockProvider(responses=[
        _text_events("branch 1"),
        _text_events("branch 2"),
        _text_events("merged"),
    ])
    registry._instances["mock"] = provider

    a1 = Agent(name="a1", model="mock:m")
    a2 = Agent(name="a2", model="mock:m")
    a3 = Agent(name="a3", model="mock:m")

    graph = Graph(
        nodes=[
            Node(id="n1", agent=a1),
            Node(id="n2", agent=a2),
            Node(id="n3", agent=a3, dependencies=["n1", "n2"]),
        ],
        output_node="n3",
    )

    result = await graph.run("start")
    assert "n1" in result.outputs
    assert "n2" in result.outputs
    assert result.final_output == "merged"


async def test_graph_with_edges():
    provider = MockProvider(responses=[
        _text_events("source output"),
        _text_events("target output"),
    ])
    registry._instances["mock"] = provider

    a1 = Agent(name="a1", model="mock:m")
    a2 = Agent(name="a2", model="mock:m")

    graph = Graph(
        nodes=[
            Node(id="src", agent=a1),
            Node(id="tgt", agent=a2),
        ],
        edges=[Edge(source="src", target="tgt", transform=lambda x: f"transformed: {x}")],
        output_node="tgt",
    )

    result = await graph.run("input")
    assert result.final_output == "target output"


async def test_graph_stuck():
    provider = MockProvider(responses=[_text_events("ok")])
    registry._instances["mock"] = provider

    a1 = Agent(name="a1", model="mock:m")
    a2 = Agent(name="a2", model="mock:m")

    graph = Graph(
        nodes=[
            Node(id="x", agent=a1, dependencies=["y"]),
            Node(id="y", agent=a2, dependencies=["x"]),
        ],
    )

    result = await graph.run("input")
    assert "stuck" in result.final_output


# --- Supervisor tests ---

async def test_supervisor_basic():
    provider = MockProvider(responses=[
        _tool_call_events("delegate_task", "tc1", {"worker": "coder", "task": "write code"}),
        _text_events("worker result"),
        _text_events("All done! The code is ready."),
    ])
    registry._instances["mock"] = provider

    coordinator = Agent(name="coordinator", model="mock:m")
    worker = Agent(name="coder", model="mock:m")

    supervisor = Supervisor(
        coordinator=coordinator,
        workers={"coder": worker},
    )

    result = await supervisor.run("Build a feature")
    assert result.output == "All done! The code is ready."


async def test_supervisor_delegate_tool_injected():
    provider = MockProvider(responses=[_text_events("done")])
    registry._instances["mock"] = provider

    coordinator = Agent(name="coord", model="mock:m")
    worker = Agent(name="w1", model="mock:m")

    supervisor = Supervisor(coordinator=coordinator, workers={"w1": worker})
    tool_names = [t.name for t in supervisor._coordinator.tools]
    assert "delegate_task" in tool_names
