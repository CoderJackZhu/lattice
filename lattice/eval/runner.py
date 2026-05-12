from __future__ import annotations

import asyncio
import time
from typing import Any

from lattice.agent.agent import Agent
from lattice.eval.types import EvalCase, EvalReport, EvalResult, Evaluator


class EvalRunner:
    def __init__(
        self,
        agent: Agent,
        evaluator: Evaluator,
        *,
        concurrency: int = 5,
    ) -> None:
        self._agent = agent
        self._evaluator = evaluator
        self._concurrency = concurrency

    async def run(self, cases: list[EvalCase]) -> EvalReport:
        semaphore = asyncio.Semaphore(self._concurrency)
        tasks = [self._run_case(case, semaphore) for case in cases]
        results = await asyncio.gather(*tasks)

        total = len(results)
        passed = sum(1 for r in results if r.passed)
        avg_score = sum(r.score for r in results) / total if total else 0.0
        avg_latency = sum(r.latency_ms for r in results) / total if total else 0.0

        return EvalReport(
            results=results,
            total=total,
            passed=passed,
            failed=total - passed,
            avg_score=avg_score,
            avg_latency_ms=avg_latency,
        )

    async def _run_case(self, case: EvalCase, semaphore: asyncio.Semaphore) -> EvalResult:
        async with semaphore:
            agent = self._agent.clone()

            start = time.monotonic()
            try:
                agent_result = await agent.run(case.input)
                output = agent_result.output
                messages = agent_result.messages
                steps = len(agent_result.steps)
            except Exception as e:
                output = f"Error: {e}"
                messages = []
                steps = 0

            latency = (time.monotonic() - start) * 1000

            eval_result = await self._evaluator.evaluate(
                output=output,
                expected=case.expected,
                case=case,
                messages=messages,
            )
            eval_result.latency_ms = latency
            eval_result.steps = steps
            return eval_result
