from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable

from lattice.agent.agent import Agent


@dataclass
class Node:
    id: str
    agent: Agent
    dependencies: list[str] = field(default_factory=list)


@dataclass
class Edge:
    source: str
    target: str
    transform: Callable[[str], str] | None = None


@dataclass
class GraphResult:
    outputs: dict[str, str] = field(default_factory=dict)
    final_output: str = ""


class Graph:
    def __init__(
        self,
        nodes: list[Node],
        edges: list[Edge] | None = None,
        *,
        entry_node: str | None = None,
        output_node: str | None = None,
    ) -> None:
        self._nodes = {n.id: n for n in nodes}
        self._edges = edges or []
        self._entry_node = entry_node or (nodes[0].id if nodes else "")
        self._output_node = output_node or (nodes[-1].id if nodes else "")

        if not edges:
            self._deps: dict[str, list[str]] = {n.id: n.dependencies for n in nodes}
        else:
            self._deps = {n.id: [] for n in nodes}
            for edge in self._edges:
                if edge.target in self._deps:
                    self._deps[edge.target].append(edge.source)

        self._edge_map: dict[tuple[str, str], Edge] = {
            (e.source, e.target): e for e in self._edges
        }

    async def run(self, input: str) -> GraphResult:
        outputs: dict[str, str] = {}
        done: set[str] = set()
        pending = set(self._nodes.keys())

        while pending:
            ready = [
                nid for nid in pending
                if all(d in done for d in self._deps.get(nid, []))
            ]

            if not ready:
                remaining = ", ".join(pending)
                return GraphResult(
                    outputs=outputs,
                    final_output=f"Graph stuck: unresolvable dependencies in [{remaining}]",
                )

            tasks: list[asyncio.Task[tuple[str, str]]] = []
            for nid in ready:
                node_input = self._build_input(nid, input, outputs)
                tasks.append(asyncio.create_task(self._run_node(nid, node_input)))

            results = await asyncio.gather(*tasks)
            for nid, output in results:
                outputs[nid] = output
                done.add(nid)
                pending.discard(nid)

        return GraphResult(
            outputs=outputs,
            final_output=outputs.get(self._output_node, ""),
        )

    async def _run_node(self, node_id: str, input: str) -> tuple[str, str]:
        node = self._nodes[node_id]
        result = await node.agent.run(input)
        return node_id, result.output

    def _build_input(self, node_id: str, original_input: str, outputs: dict[str, str]) -> str:
        deps = self._deps.get(node_id, [])
        if not deps:
            return original_input

        parts: list[str] = []
        for dep_id in deps:
            dep_output = outputs.get(dep_id, "")
            edge = self._edge_map.get((dep_id, node_id))
            if edge and edge.transform:
                dep_output = edge.transform(dep_output)
            parts.append(dep_output)

        return "\n\n".join(parts) if parts else original_input
