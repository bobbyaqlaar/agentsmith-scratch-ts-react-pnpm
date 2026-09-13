"""
runtime/metrics.py — counters and histograms, because spans are the wrong
instrument for a rate.

There were NO OpenTelemetry metrics anywhere in this framework. Everything was
spans, and the numbers an operator actually watches — error rate, cache hit
ratio, p95 time-to-first-token — were therefore only obtainable by scanning
spans. That is expensive at volume, wrong under sampling (a sampled trace store
answers "what fraction failed?" with the fraction of what it kept), and it
degrades exactly as traffic grows.

A counter is not sampled and does not grow with cardinality. The cache hit
ratio the audit asked for is one counter away: `llm_gateway` already knows
whether it hit, and only logged it.

DESIGN, matching runtime/tracing.py so there is one philosophy:

  * every instrument is created lazily and cached, because creating one per
    call leaks meter state;
  * everything no-ops when opentelemetry is absent or no meter provider is
    configured, so tenant code can record unconditionally;
  * nothing here raises into a business path. A metric that breaks the call it
    measures is worse than no metric.

ATTRIBUTES ARE LOW-CARDINALITY ON PURPOSE. tenant, model, role, outcome —
never a run id, never a prompt hash, never a trace id. A metric attribute with
unbounded values creates a new time series per value and takes the backend down
with it; that is what spans are for, and spans carry those already.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

_METER: Any = None
_INSTRUMENTS: dict[str, Any] = {}
_UNAVAILABLE = False


def _meter() -> Any:
    """The framework's meter, or None when metrics are unavailable.

    The failure is cached: without a meter provider this is called on every LLM
    call, and re-attempting a failing import per call is a measurable cost in
    the hot path it exists to measure.
    """
    global _METER, _UNAVAILABLE
    if _METER is not None or _UNAVAILABLE:
        return _METER
    try:
        from opentelemetry import metrics

        _METER = metrics.get_meter("agentsmith.runtime")
    except Exception:
        _UNAVAILABLE = True
        _METER = None
    return _METER


def _instrument(kind: str, name: str, **kwargs: Any) -> Any:
    key = f"{kind}:{name}"
    if key in _INSTRUMENTS:
        return _INSTRUMENTS[key]
    meter = _meter()
    if meter is None:
        _INSTRUMENTS[key] = None
        return None
    try:
        factory = {
            "counter": meter.create_counter,
            "histogram": meter.create_histogram,
        }[kind]
        _INSTRUMENTS[key] = factory(name, **kwargs)
    except Exception:  # fail-open: an unusable instrument is not a call failure
        _INSTRUMENTS[key] = None
    return _INSTRUMENTS[key]


def _record(instrument: Any, value: Any, attributes: Optional[dict]) -> None:
    if instrument is None:
        return
    try:
        clean = {k: v for k, v in (attributes or {}).items() if v is not None}
        if hasattr(instrument, "add"):
            instrument.add(value, clean)
        else:
            instrument.record(value, clean)
    except Exception:  # fail-open: never raise into the path being measured
        pass


# ── the instruments ──────────────────────────────────────────────────────────


def record_llm_call(
    *,
    tenant_id: Optional[str],
    model: Optional[str],
    role: Optional[str],
    outcome: str,
    duration_ms: Optional[float] = None,
    ttft_ms: Optional[float] = None,
    input_tokens: Optional[int] = None,
    output_tokens: Optional[int] = None,
    cost_usd: Optional[float] = None,
    degraded: bool = False,
) -> None:
    """One LLM call, as counters and histograms.

    `outcome` is the dimension that makes an error RATE computable without
    scanning anything: success / degraded / failed, counted on the same series.
    """
    attrs = {
        "tenant.id": tenant_id,
        "llm.model_name": model,
        "llm.gateway.tier": role,
        "outcome": outcome,
        "degraded": degraded,
    }
    _record(_instrument("counter", "agentsmith.llm.calls", unit="1"), 1, attrs)
    if duration_ms is not None:
        _record(
            _instrument("histogram", "agentsmith.llm.duration", unit="ms"),
            duration_ms,
            attrs,
        )
    if ttft_ms is not None:
        _record(
            _instrument("histogram", "agentsmith.llm.ttft", unit="ms"), ttft_ms, attrs
        )
    # Tokens are recorded ONLY when the provider reported them. A streamed call
    # reports none and carries 0/0; adding that zero to a histogram would drag
    # every percentile toward zero while looking like real data — the same
    # distinction the span attributes make with `llm.usage.reported`.
    if input_tokens is not None:
        _record(
            _instrument("histogram", "agentsmith.llm.input_tokens", unit="1"),
            input_tokens,
            attrs,
        )
    if output_tokens is not None:
        _record(
            _instrument("histogram", "agentsmith.llm.output_tokens", unit="1"),
            output_tokens,
            attrs,
        )
    if cost_usd is not None:
        _record(
            _instrument("counter", "agentsmith.llm.cost", unit="USD"), cost_usd, attrs
        )


def record_cache(*, tenant_id: Optional[str], hit: bool) -> None:
    """Idempotency cache hit or miss.

    The gateway has always known this and only logged it, so the hit ratio the
    audit asked for could not be computed at all. One counter with a `hit`
    dimension gives a ratio any backend can divide.
    """
    _record(
        _instrument("counter", "agentsmith.llm.cache", unit="1"),
        1,
        {"tenant.id": tenant_id, "hit": hit},
    )


def record_retry(
    *, tenant_id: Optional[str], model: Optional[str], attempt: int, reason: str
) -> None:
    """One retried provider call.

    `reason` is a COARSE class — "rate_limit", "timeout", "server_error" — not
    the provider's message. A metric attribute carrying free text creates a
    time series per distinct string and takes the backend down; the message
    belongs on the span event, which carries it.
    """
    _record(
        _instrument("counter", "agentsmith.llm.retries", unit="1"),
        1,
        {"tenant.id": tenant_id, "llm.model_name": model, "reason": reason},
    )


def record_retrieval(
    *, tenant_id: Optional[str], backend: str, hits: int, duration_ms: Optional[float] = None
) -> None:
    """One vector-store query. The retrieval hop had no telemetry of any kind."""
    attrs = {"tenant.id": tenant_id, "backend": backend}
    _record(_instrument("counter", "agentsmith.retrieval.queries", unit="1"), 1, attrs)
    _record(
        _instrument("histogram", "agentsmith.retrieval.hits", unit="1"), hits, attrs
    )
    if duration_ms is not None:
        _record(
            _instrument("histogram", "agentsmith.retrieval.duration", unit="ms"),
            duration_ms,
            attrs,
        )


def configure_metrics(exporter: Any = None, *, interval_ms: int = 60_000) -> Any:
    """Install a MeterProvider. Returns it, or None when unavailable.

    Separate from `configure_tracing` because a deployment can reasonably want
    one without the other — metrics to Prometheus, traces to Phoenix — and
    coupling them would force both or neither.
    """
    try:
        from opentelemetry import metrics
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
        from opentelemetry.sdk.resources import Resource
    except ImportError:
        return None

    from runtime.tracing import resource_attributes

    readers = []
    if exporter is not None:
        readers.append(
            PeriodicExportingMetricReader(exporter, export_interval_millis=interval_ms)
        )
    provider = MeterProvider(
        resource=Resource.create(resource_attributes()), metric_readers=readers
    )
    metrics.set_meter_provider(provider)

    # Drop cached instruments: they belong to whatever meter was active when
    # they were created, and keeping them would silently record into the old
    # provider forever.
    _reset_cache()
    return provider


def _reset_cache() -> None:
    """Forget the meter and every instrument. For configure_metrics and tests."""
    global _METER, _UNAVAILABLE
    _METER = None
    _UNAVAILABLE = False
    _INSTRUMENTS.clear()
