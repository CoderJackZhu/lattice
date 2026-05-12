from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class SpanEvent:
    name: str
    timestamp: float = 0.0
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class Span:
    trace_id: str = ""
    span_id: str = ""
    parent_id: str | None = None
    name: str = ""
    start_time: float = 0.0
    end_time: float | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[SpanEvent] = field(default_factory=list)
    status: Literal["ok", "error"] = "ok"


class Tracer:
    def __init__(self) -> None:
        self._spans: list[Span] = []
        self._active_span: Span | None = None

    def start_span(self, name: str, attributes: dict[str, Any] | None = None) -> Span:
        trace_id = self._active_span.trace_id if self._active_span else uuid.uuid4().hex
        span = Span(
            trace_id=trace_id,
            span_id=uuid.uuid4().hex,
            parent_id=self._active_span.span_id if self._active_span else None,
            name=name,
            start_time=time.time(),
            attributes=attributes or {},
        )
        self._spans.append(span)
        return span

    def end_span(self, span: Span) -> None:
        span.end_time = time.time()

    @asynccontextmanager
    async def span(self, name: str, **attrs: Any) -> AsyncIterator[Span]:
        span = self.start_span(name, attributes=attrs)
        previous = self._active_span
        self._active_span = span
        try:
            yield span
        except Exception as e:
            span.status = "error"
            span.events.append(SpanEvent(
                name="exception",
                timestamp=time.time(),
                attributes={"error": str(e)},
            ))
            raise
        finally:
            self._active_span = previous
            self.end_span(span)

    def export(self, format: Literal["json", "otlp"] = "json") -> str:
        from lattice.trace.exporters import to_json, to_otlp

        if format == "json":
            return to_json(self._spans)
        elif format == "otlp":
            return to_otlp(self._spans)
        raise ValueError(f"Unknown export format: {format}")

    def clear(self) -> None:
        self._spans.clear()
        self._active_span = None


tracer = Tracer()
