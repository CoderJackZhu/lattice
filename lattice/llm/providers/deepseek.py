from __future__ import annotations

import os
from typing import Any

from lattice.llm.providers.openai import OpenAIProvider


class DeepSeekProvider(OpenAIProvider):
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        super().__init__(
            api_key=api_key or os.environ.get("DEEPSEEK_API_KEY"),
            base_url=base_url or "https://api.deepseek.com/v1",
        )
