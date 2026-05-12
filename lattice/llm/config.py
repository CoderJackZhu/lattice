from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class ProviderConfig:
    api_key: str = ""
    base_url: str = ""
    type: str = ""


@dataclass
class LatticeConfig:
    models: dict[str, str] = field(default_factory=dict)
    providers: dict[str, ProviderConfig] = field(default_factory=dict)

    def get_model(self, name: str) -> tuple[str, str]:
        model_id = self.models.get(name, name)
        if ":" not in model_id:
            raise ValueError(f"Invalid model ID '{model_id}': expected 'provider:model' format")
        provider, model = model_id.split(":", 1)
        return provider, model


def _substitute_env_vars(text: str) -> str:
    def replacer(match: re.Match[str]) -> str:
        var_name = match.group(1)
        default = match.group(2)
        value = os.environ.get(var_name)
        if value is not None:
            return value
        if default is not None:
            return default
        return match.group(0)

    return re.sub(r"\$\{([^}:]+)(?::([^}]*))?\}", replacer, text)


def load_config(path: str | None = None) -> LatticeConfig:
    search_paths: list[Path] = []
    if path:
        search_paths.append(Path(path))
    search_paths.append(Path("lattice.yaml"))
    search_paths.append(Path.home() / ".lattice" / "config.yaml")

    for p in search_paths:
        if p.exists():
            raw = p.read_text(encoding="utf-8")
            raw = _substitute_env_vars(raw)
            data = yaml.safe_load(raw) or {}
            models = data.get("models", {})
            providers_raw = data.get("providers", {})
            providers = {}
            for name, conf in providers_raw.items():
                if isinstance(conf, dict):
                    providers[name] = ProviderConfig(
                        api_key=conf.get("api_key", ""),
                        base_url=conf.get("base_url", ""),
                        type=conf.get("type", ""),
                    )
            return LatticeConfig(models=models, providers=providers)

    return LatticeConfig()
