"""
runtime/tracing.py — span helpers for tenant pipeline steps
(TestbedFeedback-2026-07-21 G8).

The gateway emits richly-attributed spans for LLM calls, but a tenant's
NON-LLM steps — tool invocations, scrub counts, judge verdicts, HITL
decisions — had no framework-provided way onto a span. The observability
story is "every token and tool call streamed to Phoenix", yet tool calls
through `ToolRegistry.invoke()` emitted nothing. This closes both halves:
`ToolRegistry.invoke` is instrumented (see tool_registry.py), and
`agent_span()` gives tenant code the same one-liner for its own steps.

Everything degrades to a no-op when opentelemetry isn't installed or no
tracer is configured, exactly like the gateway's own span code — tracing
must never change program behavior or raise into a business path.

    from runtime.tracing import agent_span

    with agent_span("research.sanctions_lookup", tenant_id="acme") as span:
        hits = do_lookup(name)
        span.set_attribute("agent.tool.result_count", len(hits))
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional

# Attribute namespace for agent-step spans, kept distinct from the gateway's
# `llm.gateway.*` so a Phoenix filter can separate tool/step work from LLM work.
_NS = "agent"


def _repo_root() -> Path:
    """Delegates to runtime.config.repo_root — see there for why the marker is
    `.agenticframework` OR `.git`, not `.git` alone.

    There were FIVE of these in three disagreeing variants. A tenant nested
    inside a parent git repo resolved to the parent under the `.git`-only ones
    and to the tenant under the others, so `tenant.yaml` and `models.yaml` were
    loaded from different directories in the same process.
    """
    from runtime.config import repo_root

    return repo_root()


class _NoopSpan:
    """Stand-in when tracing is unavailable — same surface, does nothing."""

    def set_attribute(self, key: str, value: Any) -> None:
        return None

    def record_exception(self, exc: BaseException) -> None:
        return None

    def is_recording(self) -> bool:
        return False


@contextmanager
def agent_span(
    name: str,
    *,
    tenant_id: Optional[str] = None,
    kind: str = "step",
    **attributes: Any,
) -> Iterator[Any]:
    """Open a child span for a tenant pipeline step.

    Records duration and any keyword attributes under `agent.*`; on an
    exception it stamps `agent.error` and re-raises (the step still fails —
    tracing only observes). No-ops cleanly without opentelemetry or an
    active tracer, so tenant code can wrap every step unconditionally.
    """
    tracer = None
    try:
        from opentelemetry import trace

        tracer = trace.get_tracer("agentsmith.runtime")
    except Exception:
        tracer = None

    start = time.perf_counter()
    if tracer is None:
        span: Any = _NoopSpan()
        _stamp(span, name, tenant_id, kind, attributes)
        try:
            yield span
        finally:
            pass
        return

    with tracer.start_as_current_span(f"{_NS}.{name}") as span:
        _stamp(span, name, tenant_id, kind, attributes)
        try:
            yield span
        except Exception as exc:
            try:
                span.set_attribute(f"{_NS}.error", type(exc).__name__)
                span.record_exception(exc)
            except Exception:  # fail-open: tracing must never mask the real error
                pass
            raise
        finally:
            try:
                span.set_attribute(
                    f"{_NS}.duration_ms", (time.perf_counter() - start) * 1000
                )
            except Exception:  # fail-open: never break the step on a tracing write
                pass


def _stamp(span: Any, name: str, tenant_id: Optional[str], kind: str, attrs: dict) -> None:
    try:
        span.set_attribute(f"{_NS}.step", name)
        span.set_attribute(f"{_NS}.kind", kind)
        if tenant_id:
            span.set_attribute("tenant.id", tenant_id)
        for k, v in attrs.items():
            if v is not None:
                span.set_attribute(f"{_NS}.{k}", v)
    except Exception:  # fail-open: attribute writes must never raise into the step
        pass


# The OpenInference/Phoenix payload attribute names, which trace_redactor
# ALREADY scrubs — see its `_PAYLOAD_ATTRIBUTES`. Reused rather than inventing
# `agent.tool.input`, so tool payloads inherit the whole pipeline for free:
# pattern scrubbing in staging, truncation to 50 characters plus an encrypted
# HITL blob in production. A new attribute name would have been a new,
# unscrubbed channel — which is the one thing this must not be.
TOOL_INPUT_ATTRIBUTE = "input.value"
TOOL_OUTPUT_ATTRIBUTE = "output.value"

# Serialisation ceiling, applied BEFORE the redactor sees anything. The
# redactor truncates by profile, but a tool returning a megabyte of JSON should
# never be turned into a megabyte string in the first place — that cost lands
# on the call being traced.
_PAYLOAD_CHARS = 4000


def tool_payloads_enabled() -> bool:
    """Whether tool arguments and results go on the span. OFF by default.

    Opt-in because it is a new egress channel, not because it is unsafe: it
    routes through `trace_redactor` exactly as prompts do. But a tenant should
    say that it wants its tool arguments leaving the process, in the file where
    the rest of its posture is declared, rather than discover it in Phoenix.
    """
    try:
        from runtime.config import as_bool, resolve

        return as_bool(
            resolve(
                "security.trace_tool_payloads",
                env_var="TRACE_TOOL_PAYLOADS",
                default=False,
            )
        )
    except Exception:  # fail-closed: unreadable config records nothing
        return False


def _serialise_payload(value: Any) -> Optional[str]:
    """A payload as text, capped, or None when it cannot be represented.

    None rather than `str(exc)`: a serialisation failure must not put an
    exception message where a caller will read it as the tool's actual input.
    """
    if value is None:
        return None
    try:
        import json

        text = value if isinstance(value, str) else json.dumps(value, default=str)
    except Exception:
        try:
            text = str(value)
        except Exception:
            return None
    if len(text) > _PAYLOAD_CHARS:
        # Marked, not silently clipped — a reader must not take a truncated
        # payload for the whole one.
        return text[:_PAYLOAD_CHARS] + f"…[truncated at {_PAYLOAD_CHARS} chars]"
    return text


def record_tool_call(
    name: str,
    *,
    allowed: bool,
    duration_ms: float,
    error: Optional[str] = None,
    tenant_id: Optional[str] = None,
    args: Any = None,
    result: Any = None,
) -> None:
    """Emit a CHILD span for one tool invocation (`agent.tool.<name>`).

    Called from `ToolRegistry.invoke` so every tool call is visible in
    Phoenix — the allow/deny outcome, duration, and any error — which the
    'every tool call streamed' claim requires but nothing delivered.

    A child span rather than an annotation on the enclosing step, because a
    single step routinely calls several tools (sanctions + registry + media
    in one research step); annotating the current span would let each call
    clobber the previous one's `agent.tool.*` attributes and only the last
    would survive. Nested under the active `agent_span` when there is one,
    so the parent/child structure still shows which step made the call.
    """
    tracer = None
    try:
        from opentelemetry import trace

        if not getattr(trace.get_current_span(), "is_recording", lambda: False)():
            # No active recording span → creating a lone root span per tool
            # call would be noise; only trace tool calls inside a step.
            return
        tracer = trace.get_tracer("agentsmith.runtime")
    except Exception:
        return
    if tracer is None:
        return

    try:
        with tracer.start_as_current_span(f"{_NS}.tool.{name}") as span:
            span.set_attribute("agent.tool.name", name)
            span.set_attribute("agent.tool.allowed", allowed)
            span.set_attribute("agent.tool.duration_ms", duration_ms)
            if tenant_id:
                span.set_attribute("tenant.id", tenant_id)
            if error:
                span.set_attribute("agent.tool.error", error)

            # Arguments and result, when the tenant has asked for them. These
            # land on the names trace_redactor already scrubs, so the profile
            # applies without this module knowing anything about redaction.
            if tool_payloads_enabled():
                payload_in = _serialise_payload(args)
                if payload_in is not None:
                    span.set_attribute(TOOL_INPUT_ATTRIBUTE, payload_in)
                payload_out = _serialise_payload(result)
                if payload_out is not None:
                    span.set_attribute(TOOL_OUTPUT_ATTRIBUTE, payload_out)
    except Exception:  # fail-open: tracing must never break a tool call
        pass


# ── Identity: Resource + on_start stamping (pillar 3) ────────────────────────


def resource_attributes(project_name: Optional[str] = None) -> dict:
    """The PER-PROCESS half of pillar 3 — fixed for the life of the worker.

    These four cannot vary between two spans from the same process, so they
    belong on the OTel Resource where every span inherits them and no call site
    can forget one. `agent.role` and `tenant.id` are deliberately absent: this
    architecture runs many roles in one worker (KYC registers six activities on
    one task queue), and the shared-pool default serves many tenants from one
    process, so either as a Resource attribute would be a confident lie on most
    spans rather than a missing one.

    `project.name` is the one place a repo-derived default is right: it names
    the codebase, nothing partitions on it, and being wrong is cosmetic.
    """
    from runtime.environment import get_environment

    from runtime.config import resolve

    project = resolve(
        "tenant.name",
        explicit=project_name,
        env_var="AGENT_PROJECT_NAME",
        default=None,
    ) or _repo_root().name
    from runtime.version import framework_version

    attrs = {
        "service.name": project,
        "project.name": project,
        "environment": get_environment(),
        # WHICH AGENTSMITH emitted this. Per-process and constant, so unlike
        # `agent.role` and `tenant.id` the Resource is the correct home for it
        # — the reasoning in this docstring points the other way here.
        #
        # It is on the wire because the framework and the tenants that use it
        # have different owners: IT ships AgentSmith, the business ships the
        # tenant and pins a version so IT's cadence cannot move underneath it.
        # IT therefore runs a FLEET on several versions at once, and a version
        # decides what a tenant can emit at all — v1.2.0 has no
        # prompt_identity, no metrics, and no identity processor, so its spans
        # carry no `prompt.system.sha256`, no counters and often no `tenant.id`.
        # Without this attribute that is indistinguishable, on an ops
        # dashboard, from a current tenant that is broken.
        "agentsmith.framework.version": framework_version(),
    }
    # `tenant.owner` is declared in tenant.yaml and was read by nothing, while
    # this attribute came from a shell profile — so on a dev machine every repo
    # reported the same owner and in CI none did. Env still wins as the
    # per-deploy override; the declaration is the default it falls back to.
    from runtime.config import resolve

    owner = resolve("tenant.owner", env_var="AGENT_OWNER_ID", default=None)
    if owner:
        # Omitted when unset rather than "unknown" — see current_identity().
        attrs["agent.owner_id"] = str(owner)
    return attrs


try:
    from opentelemetry.sdk.trace import SpanProcessor as _OTelSpanProcessor
except ImportError:  # tracing is optional
    _OTelSpanProcessor = object  # type: ignore


class AgentIdentityProcessor(_OTelSpanProcessor):
    """Stamps the PER-STEP half of pillar 3 onto every span, at start.

    Subclasses the SDK's SpanProcessor rather than duck-typing it. A plain
    class with on_start/on_end/shutdown/force_flush looks complete and is not:
    the SDK also calls a private `_on_ending`, so a duck-typed processor raises
    AttributeError on the FIRST span it sees. Same choice trace_redactor.py
    made, and for the same reason.

    `on_start` rather than `on_end`: an exporter may sample or drop on the
    attributes, and a span that gains its tenant only at the end has already
    been routed without it.

    This is what makes "every span carries tenant.id" true by construction
    instead of by discipline. It was previously a kwarg on `agent_span()`
    applied under `if tenant_id:` — so a caller who omitted it produced an
    unattributed span, silently, and most callers did.
    """

    def on_start(self, span: Any, parent_context: Any = None) -> None:
        try:
            from runtime.tenancy import current_identity

            for key, value in current_identity().items():
                span.set_attribute(key, value)
        except Exception:  # fail-open: identity must never break a span
            pass

    # on_end / shutdown / force_flush come from the base class — this
    # processor only ever writes at start.


def configure_tracing(
    *,
    project_name: Optional[str] = None,
    exporter: Any = None,
    redact: bool = True,
) -> Any:
    """Install a TracerProvider wired the way pillar 3 requires. Returns it.

    Exists because assembling this by hand is three steps that must all be
    remembered, and the evidence says they are not: KYC Sentinel — the tenant
    built to exercise every layer of this framework — installs NO provider at
    all, so every `agent_span()` in it is a no-op and no span has ever reached
    Phoenix from it. A documented three-step recipe produced zero correct
    setups; a function is harder to half-do.

    Idempotent-ish: OTel's global provider is one-shot, so a second call is
    ignored by the SDK. Returns whatever provider is active either way.
    """
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
    except ImportError:  # fail-open: tracing is optional, the app is not
        return None

    provider = TracerProvider(resource=Resource.create(resource_attributes(project_name)))
    provider.add_span_processor(AgentIdentityProcessor())

    if redact:
        # Ordering matters: identity stamps at start, redaction rewrites at end.
        from runtime.trace_redactor import TraceRedactor

        provider.add_span_processor(TraceRedactor())

    if exporter is not None:
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        provider.add_span_processor(BatchSpanProcessor(exporter))

    trace.set_tracer_provider(provider)
    return trace.get_tracer_provider()



def configure_telemetry(
    *,
    project_name: Optional[str] = None,
    redact: bool = True,
    metrics_interval_ms: int = 60_000,
) -> dict:
    """Install BOTH providers, with exporters resolved from the environment.

    The one call a worker makes. Returns
    `{"tracer_provider": ..., "meter_provider": ...}`; either value is None when
    that half is unavailable.

    WHY THIS EXISTS, and it is the same argument `configure_tracing` already
    makes one function up. `runtime/metrics.py` shipped counters and histograms
    for calls, cache hit/miss, retries, cost and retrieval, every call site was
    correctly placed, and `configure_metrics()` HAD NO CALLER ANYWHERE — not the
    framework worker, not the tenant's, not the example's. Without a
    MeterProvider `metrics.get_meter()` hands back a `_ProxyMeter` whose
    instruments buffer for a real provider that never arrives, so every
    `record_llm_call` in production wrote into nothing. The error rate, the
    cache hit ratio and the TTFT percentiles — the four numbers the
    observability audit says spans are the wrong instrument for — were
    computable nowhere.

    That is precisely the failure `configure_tracing` was written to end, one
    signal over: KYC installed no TracerProvider, so every `agent_span()` in the
    framework's own testbed was a no-op. A documented recipe produced zero
    correct setups then, and separate `configure_tracing`/`configure_metrics`
    calls produced one correct setup out of two now.

    The two remain separately callable — a deployment can legitimately want
    metrics on Prometheus and traces on Phoenix, and coupling them would force
    both or neither. What changes is that the DEFAULT path does both, because
    the evidence is that the assembling-by-hand path does not.
    """
    from runtime.metrics import configure_metrics
    from runtime.otlp import metric_exporter, span_exporter

    tracer_provider = configure_tracing(
        project_name=project_name, exporter=span_exporter(), redact=redact
    )
    meter_provider = configure_metrics(
        exporter=metric_exporter(), interval_ms=metrics_interval_ms
    )
    return {"tracer_provider": tracer_provider, "meter_provider": meter_provider}


# ── Context propagation (W3C trace-context) ──────────────────────────────────


def current_trace_id() -> Optional[str]:
    """The active trace as 32 lowercase hex characters, or None.

    `agent_runs.trace_id` exists to correlate a database row with a trace, and
    `_report_run_status` has accepted a `trace_id` argument since it was
    written — which not one of its nine call sites ever passed. The column was
    NULL for every run ever recorded, and the portal's trace link had nothing
    to link to. This is what fills it.
    """
    try:
        from opentelemetry import trace

        ctx = trace.get_current_span().get_span_context()
        if not ctx.is_valid:
            return None
        return format(ctx.trace_id, "032x")
    except Exception:  # fail-open: correlation is never worth failing a call
        return None


def traceparent() -> Optional[str]:
    """The active context as a W3C `traceparent` value, or None.

    Built by hand rather than via the propagators API because that is one line
    against an optional dependency's optional module, and this must degrade to
    None rather than raise when opentelemetry is absent — which is the normal
    case for a tenant that has not turned tracing on.

        00-<32 hex trace id>-<16 hex span id>-<2 hex flags>
    """
    try:
        from opentelemetry import trace

        ctx = trace.get_current_span().get_span_context()
        if not ctx.is_valid:
            return None
        return (
            f"00-{ctx.trace_id:032x}-{ctx.span_id:016x}-{ctx.trace_flags:02x}"
        )
    except Exception:
        return None


def inject_context(headers: Optional[dict] = None) -> dict:
    """Add `traceparent` to an outgoing request's headers.

    Without this every hop starts its own trace: the worker's spans and the Ops
    Portal's work were separate traces with no edge between them, so "follow
    this request across services" — the entire point of distributed tracing —
    stopped at the process boundary.

    Returns the dict unchanged when there is nothing to propagate, so a caller
    can wrap its headers unconditionally.
    """
    out = dict(headers or {})
    parent = traceparent()
    if parent:
        out["traceparent"] = parent
    return out
