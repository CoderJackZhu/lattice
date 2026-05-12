from __future__ import annotations

import json
from dataclasses import asdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lattice.trace.tracer import Span


def to_json(spans: list[Span]) -> str:
    return json.dumps([asdict(s) for s in spans], indent=2, ensure_ascii=False)


def to_otlp(spans: list[Span]) -> str:
    raise NotImplementedError("OTLP export is not yet implemented")
