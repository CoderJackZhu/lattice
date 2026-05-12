from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel

from lattice.llm.types import Message, ModelResponse, StreamEvent, ToolSchema


@runtime_checkable
class LLMProvider(Protocol):
    async def stream(
        self,
        model: str,
        messages: list[Message],
        *,
        system: str | None = None,
        tools: list[ToolSchema] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        stop: list[str] | None = None,
        response_format: type[BaseModel] | None = None,
    ) -> AsyncIterator[StreamEvent]: ...

    async def complete(
        self,
        model: str,
        messages: list[Message],
        *,
        system: str | None = None,
        tools: list[ToolSchema] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        stop: list[str] | None = None,
        response_format: type[BaseModel] | None = None,
    ) -> ModelResponse: ...


class ProviderRegistry:
    def __init__(self) -> None:
        self._factories: dict[str, Callable[..., LLMProvider]] = {}
        self._instances: dict[str, LLMProvider] = {}

    def register(self, name: str, factory: Callable[..., LLMProvider]) -> None:
        self._factories[name] = factory

    def get(self, name: str, **kwargs: Any) -> LLMProvider:
        cache_key = name if not kwargs else f"{name}:{hash(tuple(sorted(kwargs.items())))}"
        if cache_key not in self._instances:
            if name not in self._factories:
                raise ValueError(f"Unknown provider '{name}'. Available: {list(self._factories)}")
            self._instances[cache_key] = self._factories[name](**kwargs)
        return self._instances[cache_key]

    def from_model_id(self, model_id: str) -> tuple[LLMProvider, str]:
        if ":" not in model_id:
            raise ValueError(f"Invalid model ID '{model_id}': expected 'provider:model' format")
        provider_name, model_name = model_id.split(":", 1)
        return self.get(provider_name), model_name


registry = ProviderRegistry()
