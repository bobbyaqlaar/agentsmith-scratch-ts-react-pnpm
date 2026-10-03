"""
runtime/test/conftest.py — one in-memory tracer for the whole test session.

OTel's global tracer provider is ONE-SHOT: the first `set_tracer_provider` wins
and every later call is ignored with an "Overriding not allowed" warning. So two
test modules that each install their own provider do not both work — the loser's
exporter is never wired, and it sees the spans of a provider it does not hold.

`test_tracing.py` handled that by skipping when it lost the race. That is safe
but silent, and it made coverage depend on collection order: adding
`test_gateway_span_usage.py` — which also installed a provider — turned five
passing tracing tests into skips, and the suite still reported green. A run that
quietly measures less is the failure mode this repo keeps finding.

Installing it exactly once here removes the race rather than tolerating it, so
no module has to lose and none of them skip.
"""

from __future__ import annotations

import pytest


@pytest.fixture(scope="session")
def _exporter():
    trace = pytest.importorskip("opentelemetry.trace")
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    # Fail rather than skip. Nothing else in this suite installs a provider now,
    # so losing the race means something reintroduced one — and a skip would
    # hide that by turning every span assertion in the session into a no-op.
    assert trace.get_tracer_provider() is provider, (
        "another global TracerProvider was installed before this fixture — "
        "span assertions would silently stop being checked. Install providers "
        "only here."
    )
    return exporter


@pytest.fixture
def spans(_exporter):
    """The session exporter, emptied. Each test asserts on its own spans."""
    _exporter.clear()
    return _exporter


@pytest.fixture
def identity_processor(_exporter):
    """Attach AgentIdentityProcessor to the session provider for one test.

    Shared rather than copied: two test modules need it, and a fixture cloned
    into both is the shape that drifts. Not autouse — most tests assert on
    spans that should NOT be stamped, and a processor that is always on would
    make "an unbound span is unattributed" untestable.
    """
    from opentelemetry import trace

    from runtime.tracing import AgentIdentityProcessor

    processor = AgentIdentityProcessor()
    trace.get_tracer_provider().add_span_processor(processor)
    yield processor
    processor.shutdown()


def pytest_configure(config):
    """Isolate the suite from this machine's `agentsmith` state.

    `agentsmith mode hybrid` and `agentsmith dashboard start` write
    ~/.agent-framework/state/, which the gateway and runtime/otlp.py read after
    the environment. Without this a developer's chosen mode would decide which
    profile these tests resolve. The security harness runs these suites too, so
    it inherits the isolation.
    """
    import os
    import tempfile

    os.environ.setdefault("AGENTSMITH_STATE_DIR", tempfile.mkdtemp(prefix="agentsmith-test-state-"))

    # This checkout's `agentsmith` on PATH, as scripts/test/conftest.py does: a
    # tenant's commits ask it at gate contract 3. One implementation, in
    # scripts/test/provider_shim.py — present in a framework checkout only. This
    # file is vendored into tenants' runtime/test/, where scripts/test/ is not and
    # nothing here commits; loading it unconditionally crashed their security
    # harness at configure time (.agent-rfc/reviews/gate-local-events.md).
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "scripts" / "test" / "provider_shim.py"
    if path.is_file():
        spec = importlib.util.spec_from_file_location("provider_shim", path)
        module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        spec.loader.exec_module(module)  # type: ignore[union-attr]
        module.install()
