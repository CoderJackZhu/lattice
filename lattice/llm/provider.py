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
        self._config_loaded = False
        self._provider_configs: dict[str, dict[str, str]] = {}

    def _ensure_config(self) -> None:
        if self._config_loaded:
            return
        self._config_loaded = True
        try:
            from lattice.llm.config import load_config

            config = load_config()
            for name, pconf in config.providers.items():
                kwargs: dict[str, str] = {}
                if pconf.api_key:
                    kwargs["api_key"] = pconf.api_key
                if pconf.base_url:
                    kwargs["base_url"] = pconf.base_url
                if kwargs:
                    self._provider_configs[name] = kwargs
        except Exception:
            pass

    def register(self, name: str, factory: Callable[..., LLMProvider]) -> None:
        self._factories[name] = factory

    def get(self, name: str, **kwargs: Any) -> LLMProvider:
        self._ensure_config()
        if not kwargs and name in self._provider_configs:
            kwargs = self._provider_configs[name]
        cache_key = name if not kwargs else f"{name}:{hash(frozenset(kwargs.items()))}"
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
