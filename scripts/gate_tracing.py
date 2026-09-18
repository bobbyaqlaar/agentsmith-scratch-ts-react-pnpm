"""
scripts/gate_tracing.py — every process-gate decision is a span (pillar 3),
spooled locally and shipped later (.agent-rfc/designs/governance-enforcement.md, G1).

A hook is a short-lived process inside an IDE's time limit. Exporting from it
directly would put the collector on the edit path: a down Phoenix holds every
edit for the exporter's timeout, and a hook that exits before its batch
processor flushes loses the span. So the exporter here writes each batch to
disk as the exact OTLP/HTTP request body, and `ship()` — called only at
session start, stop and sweep, never before an edit — POSTs those bytes with a
short timeout and deletes only what the collector accepted.

Identity and redaction come from runtime/tracing.py and runtime/tenancy.py; this
module adds no second tracer. Tracing never changes a gate's answer: when it
cannot start, the span is a no-op and `status_line()` says why.
"""

from __future__ import annotations

import os
import sys
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from pydantic import BaseModel, Field

MAX_BATCHES = 500
_FRAMEWORK_ROOT = Path(__file__).resolve().parents[1]

_state: dict[str, Any] = {"provider": None, "error": None}


def spool_dir() -> Path:
    base = os.environ.get("AGENTSMITH_STATE_DIR") or str(Path.home() / ".agent-framework" / "state")
    return Path(base) / "gate-spans"


def _dropped_file() -> Path:
    return spool_dir() / "dropped.count"


def dropped_count() -> int:
    try:
        return int(_dropped_file().read_text(encoding="utf-8").strip() or 0)
    except (OSError, ValueError):
        return 0


def _import_runtime() -> None:
    try:
        import runtime.tracing
    except ImportError:
        # A vendored tenant carries runtime/ beside scripts/; the framework
        # checkout does too. Installed mode gets it from the framework venv.
        sys.path.insert(0, str(_FRAMEWORK_ROOT))
        import runtime.tracing  # noqa: F401


def _spool_exporter() -> Any:
    from opentelemetry.exporter.otlp.proto.common.trace_encoder import encode_spans
    from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

    class SpoolExporter(SpanExporter):
        def export(self, spans):  # type: ignore[override]
            try:
                directory = spool_dir()
                directory.mkdir(parents=True, exist_ok=True)
                name = f"{time.time_ns()}-{os.getpid()}"
                partial = directory / f"{name}.tmp"
                partial.write_bytes(encode_spans(spans).SerializeToString())
                partial.replace(directory / f"{name}.pb")  # atomic: ship() never reads half a batch
                _prune(directory)
            except OSError as exc:
                # A read-only or full disk must not turn into a stack trace on a
                # hook's stderr, and must not read as a successful export either.
                _state["error"] = f"spool write failed: {exc}"
                return SpanExportResult.FAILURE
            return SpanExportResult.SUCCESS

        def shutdown(self) -> None:
            pass

    return SpoolExporter()


def _prune(directory: Path) -> None:
    batches = sorted(directory.glob("*.pb"))
    excess = len(batches) - MAX_BATCHES
    if excess <= 0:
        return
    for batch in batches[:excess]:
        batch.unlink(missing_ok=True)
    _dropped_file().write_text(str(dropped_count() + excess), encoding="utf-8")


def _provider() -> Any:
    if _state["provider"] is None:
        _import_runtime()
        from runtime.tracing import configure_tracing

        _state["provider"] = configure_tracing(exporter=_spool_exporter())
    return _state["provider"]


class _NoSpan:
    def set_attribute(self, key: str, value: Any) -> None:
        pass


@contextmanager
def gate_span(event: str, *, root: Path, ide: str, **attributes: Any) -> Iterator[Any]:
    """`agent.gate.<event>` with `agent.role=process-gate`, the repo's tenant
    when it declares one, and the given attributes under `agent.*`."""
    try:
        _provider()
        from runtime.tenancy import agent_context, tenant_id_from_config
        from runtime.tracing import agent_span

        tenant = tenant_id_from_config(root)
    except Exception as exc:  # fail-open: tracing never changes a gate's answer
        _state["error"] = f"{type(exc).__name__}: {exc}"
        yield _NoSpan()
        return
    with agent_context(role="process-gate", tenant_id=tenant):
        with agent_span(f"gate.{event}", tenant_id=tenant, ide=ide, repo=root.name, **attributes) as span:
            yield span


def flush() -> None:
    provider = _state["provider"]
    if provider is not None and hasattr(provider, "force_flush"):
        try:
            provider.force_flush()
        except Exception as exc:  # fail-open
            _state["error"] = f"flush failed: {exc}"


class ShipResult(BaseModel):
    endpoint: str | None
    sent: int = 0
    kept: int = 0
    errors: list[str] = Field(default_factory=list)


def ship(timeout: float = 1.0) -> ShipResult:
    """POST spooled batches to the OTLP traces endpoint; keep any not accepted."""
    batches = sorted(spool_dir().glob("*.pb"))
    try:
        _import_runtime()
        from runtime.otlp import resolve_otlp_endpoint

        endpoint = resolve_otlp_endpoint("traces")
    except Exception as exc:
        return ShipResult(endpoint=None, kept=len(batches), errors=[f"endpoint unresolved: {exc}"])
    if not endpoint:
        return ShipResult(endpoint=None, kept=len(batches))
    result = ShipResult(endpoint=endpoint)
    for batch in batches:
        request = urllib.request.Request(
            endpoint, data=batch.read_bytes(), method="POST",
            headers={"Content-Type": "application/x-protobuf"},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                accepted = 200 <= response.status < 300
        except (urllib.error.URLError, OSError) as exc:
            result.kept += len(batches) - result.sent - result.kept
            result.errors.append(f"{endpoint}: {getattr(exc, 'reason', exc)}")
            return result  # the collector is down: don't pay the timeout once per batch
        if accepted:
            batch.unlink(missing_ok=True)
            result.sent += 1
        else:
            result.kept += 1
    return result


def status_line(result: ShipResult | None = None) -> str:
    """One line for session start. Four states, never one that reads as another."""
    parts = []
    if _state["error"]:
        parts.append(f"gate spans NOT emitted — {_state['error']}")
    if result is not None:
        if result.endpoint is None:
            parts.append(f"gate spans NOT exported (no OTLP endpoint) — "
                         f"{result.kept} batch(es) spooled in {spool_dir()}")
        elif result.errors:
            parts.append(f"gate spans NOT exported — {result.errors[0]}; {result.kept} batch(es) kept for the next try")
        else:
            parts.append(f"gate spans exported to {result.endpoint} ({result.sent} batch(es))")
    dropped = dropped_count()
    if dropped:
        parts.append(f"{dropped} dropped — the spool reached {MAX_BATCHES} batches before a collector took them")
    return "; ".join(parts)


def reset_for_tests() -> None:
    """OTel's global provider is set-once per process; tests need a fresh one."""
    try:
        from opentelemetry import trace
        from opentelemetry.util._once import Once

        trace._TRACER_PROVIDER_SET_ONCE = Once()  # type: ignore[attr-defined]
        trace._TRACER_PROVIDER = None  # type: ignore[attr-defined]
    except Exception:  # fail-open: only tests call this
        pass
    _state["provider"] = None
    _state["error"] = None
