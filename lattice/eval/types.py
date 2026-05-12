from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class EvalCase:
    input: str
    expected: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)


@dataclass
class EvalResult:
    case: EvalCase
    output: str = ""
    score: float = 0.0
    passed: bool = False
    details: str = ""
    latency_ms: float = 0.0
    steps: int = 0


@dataclass
class EvalReport:
    results: list[EvalResult] = field(default_factory=list)
    total: int = 0
    passed: int = 0
    failed: int = 0
    avg_score: float = 0.0
    avg_latency_ms: float = 0.0

    def summary(self) -> str:
        lines = [
            f"Eval Report: {self.passed}/{self.total} passed ({self.avg_score:.2%} avg score)",
            f"  Average latency: {self.avg_latency_ms:.0f}ms",
        ]
        for r in self.results:
            status = "PASS" if r.passed else "FAIL"
            lines.append(f"  [{status}] {r.case.input[:60]}... -> score={r.score:.2f}")
        return "\n".join(lines)


class Evaluator(Protocol):
    async def evaluate(self, output: str, expected: str, **kwargs: Any) -> EvalResult: ...
