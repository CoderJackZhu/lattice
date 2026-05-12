from __future__ import annotations

from typing import Any

from lattice.eval.types import EvalCase, EvalResult


class ExactMatch:
    async def evaluate(self, output: str, expected: str, **kwargs: Any) -> EvalResult:
        passed = output.strip() == expected.strip()
        return EvalResult(
            case=kwargs.get("case", EvalCase(input="")),
            output=output,
            score=1.0 if passed else 0.0,
            passed=passed,
            details="exact match" if passed else f"expected '{expected}', got '{output}'",
        )


class Contains:
    def __init__(self, case_sensitive: bool = False) -> None:
        self._case_sensitive = case_sensitive

    async def evaluate(self, output: str, expected: str, **kwargs: Any) -> EvalResult:
        if self._case_sensitive:
            passed = expected in output
        else:
            passed = expected.lower() in output.lower()
        return EvalResult(
            case=kwargs.get("case", EvalCase(input="")),
            output=output,
            score=1.0 if passed else 0.0,
            passed=passed,
            details="contains match" if passed else f"'{expected}' not found in output",
        )


class LLMJudge:
    def __init__(self, model: str = "openai:gpt-4o-mini", threshold: float = 0.7) -> None:
        self._model = model
        self._threshold = threshold

    async def evaluate(self, output: str, expected: str, **kwargs: Any) -> EvalResult:
        from lattice.llm.provider import registry
        from lattice.llm.types import Message, StreamEnd, TextContent

        provider, model_name = registry.from_model_id(self._model)

        prompt = (
            f"Evaluate the following output against the expected answer.\n\n"
            f"Expected: {expected}\n"
            f"Actual: {output}\n\n"
            f"Score from 0.0 to 1.0. Reply with ONLY a number."
        )

        messages = [Message(role="user", content=[TextContent(text=prompt)])]
        response = await provider.complete(model_name, messages)

        text = "".join(
            c.text for c in response.message.content if isinstance(c, TextContent)
        )

        try:
            score = float(text.strip())
            score = max(0.0, min(1.0, score))
        except ValueError:
            score = 0.0

        return EvalResult(
            case=kwargs.get("case", EvalCase(input="")),
            output=output,
            score=score,
            passed=score >= self._threshold,
            details=f"LLM judge score: {score:.2f}",
        )


class ToolUseEvaluator:
    def __init__(self, required_tools: list[str]) -> None:
        self._required_tools = required_tools

    async def evaluate(self, output: str, expected: str, **kwargs: Any) -> EvalResult:
        messages = kwargs.get("messages", [])
        from lattice.llm.types import ToolCall

        used_tools: set[str] = set()
        for msg in messages:
            for c in getattr(msg, "content", []):
                if isinstance(c, ToolCall):
                    used_tools.add(c.name)

        missing = [t for t in self._required_tools if t not in used_tools]
        passed = len(missing) == 0
        score = 1.0 - len(missing) / max(len(self._required_tools), 1)

        return EvalResult(
            case=kwargs.get("case", EvalCase(input="")),
            output=output,
            score=score,
            passed=passed,
            details=f"used: {sorted(used_tools)}, missing: {sorted(missing)}" if missing else "all required tools used",
        )
