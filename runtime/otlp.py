"""
runtime/otlp.py — where an OTLP signal is sent, decided once.

WHY THIS MODULE EXISTS. Four places built an OTLP exporter from an endpoint
variable, and they disagreed about which variables to read and what to do with
the value:

    scripts/local_agent_stack.py   AGENT_PHOENIX_ENDPOINT -> OTEL_EXPORTER_OTLP_ENDPOINT
    scripts/multi_agent_system.py  AGENT_PHOENIX_ENDPOINT only
    KYC_Sentinel/worker.py         AGENT_PHOENIX_ENDPOINT only
    portal/lib/tracing.ts          the full chain, and the only one that was right

Every Python copy ended with `f"{endpoint.rstrip('/')}/v1/traces"`. The portal's
does not, and its comment says why: this repo's OWN convention — OPERATIONS.md,
docs/team-observability.md, docker-compose.yml, SPECS.md §699 and
`ai-dashboard-start` — sets `OTEL_EXPORTER_OTLP_ENDPOINT` to a full
`…/v1/traces` URL, in the variable the OTLP spec defines as a BASE. So a
consumer that appends unconditionally posts to `/v1/traces/v1/traces` and drops
everything on a 404 that surfaces nowhere.

`local_agent_stack.py` falls back to exactly that variable and appends
unconditionally. The guard was written once, in TypeScript, for the portal, and
the Python sibling reading the same variable never got it — the "fix applied at
one call site and not its identical neighbours" shape, across a language
boundary (review-levers: grep-for-siblings).

THE PORTAL'S VERSION IS THE CANONICAL ONE, ported here rather than a fifth
invention, because it is the only one that handled the trap and the only one
with the full precedence chain. `portal/lib/tracing.ts` cannot import this —
different language, different process — so the two are PINNED by a test that
parses the TypeScript rather than restating it (review-levers: pin-unremovable-duplicates).

ONE THING THE PORTAL'S VERSION COULD NOT HAVE. It only ever resolves traces, so
it never had to consider a base that already names a DIFFERENT signal. Asking
for metrics with `OTEL_EXPORTER_OTLP_ENDPOINT=http://x:6006/v1/traces` set must
not produce `/v1/traces/v1/metrics`. A known signal suffix is stripped before
the requested one is appended — a case that only becomes visible once the two
signals share a resolver.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Mapping, Optional

logger = logging.getLogger(__name__)

# The OTLP/HTTP paths, by signal. Also the set stripped from a base URL that
# already names one — see the module docstring.
SIGNAL_PATHS = {
    "traces": "/v1/traces",
    "metrics": "/v1/metrics",
    "logs": "/v1/logs",
}

# Read in order. The first two are the OTLP spec's; the third is this
# framework's own name for the same thing, which every doc and CI workflow here
# sets. Dropping it would be a silent downgrade for every existing deployment.
_BASE_ENV_VARS = ("OTEL_EXPORTER_OTLP_ENDPOINT", "AGENT_PHOENIX_ENDPOINT")


def resolve_otlp_endpoint(
    signal: str = "traces", env: Optional[Mapping[str, str]] = None
) -> Optional[str]:
    """The full OTLP/HTTP URL for `signal`, or None when nothing is configured.

    None is not a failure. A developer with no collector still gets a provider,
    a Resource and the identity processor, so spans and metrics are correctly
    formed and simply not exported; turning the endpoint on later changes the
    destination and nothing else.
    """
    if signal not in SIGNAL_PATHS:
        raise ValueError(
            f"unknown OTLP signal {signal!r}; expected one of {sorted(SIGNAL_PATHS)}"
        )
    environ = os.environ if env is None else env
    path = SIGNAL_PATHS[signal]

    # A per-signal variable is the operator naming the exact URL. Used as-is:
    # second-guessing it is how you break the one person who read the spec.
    explicit = (environ.get(f"OTEL_EXPORTER_OTLP_{signal.upper()}_ENDPOINT") or "").strip()
    if explicit:
        return explicit

    base = ""
    for var in _BASE_ENV_VARS:
        base = (environ.get(var) or "").strip()
        if base:
            break
    if not base:
        return None

    trimmed = base.rstrip("/")
    for known in SIGNAL_PATHS.values():
        if trimmed.endswith(known):
            # Already names a signal. Strip it — then the requested one is
            # appended below. Same-signal is a no-op round trip; cross-signal
            # is the case that would otherwise yield /v1/traces/v1/metrics.
            trimmed = trimmed[: -len(known)].rstrip("/")
            break
    return f"{trimmed}{path}"


def span_exporter(env: Optional[Mapping[str, str]] = None) -> Any:
    """An OTLP span exporter, or None when unconfigured or unavailable."""
    return _exporter(
        "traces",
        "opentelemetry.exporter.otlp.proto.http.trace_exporter",
        "OTLPSpanExporter",
        env,
    )


def metric_exporter(env: Optional[Mapping[str, str]] = None) -> Any:
    """An OTLP metric exporter, or None when unconfigured or unavailable."""
    return _exporter(
        "metrics",
        "opentelemetry.exporter.otlp.proto.http.metric_exporter",
        "OTLPMetricExporter",
        env,
    )


def _exporter(signal: str, module: str, attr: str, env) -> Any:
    endpoint = resolve_otlp_endpoint(signal, env)
    if not endpoint:
        return None
    try:
        exporter_cls = getattr(__import__(module, fromlist=[attr]), attr)
    except ImportError:
        # SAID, not swallowed. An endpoint is configured, so somebody expects
        # data at it; silently exporting nothing is the failure mode this
        # framework keeps finding. The install fix is one package name.
        logger.warning(
            "%s is configured but opentelemetry-exporter-otlp-proto-http is not "
            "installed — %s will not be exported. pip install "
            "'opentelemetry-exporter-otlp-proto-http>=1.20,<2.0'",
            endpoint,
            signal,
        )
        return None
    return exporter_cls(endpoint=endpoint)
