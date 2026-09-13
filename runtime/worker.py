"""
runtime/worker.py — Production worker entrypoint.

Starts a Temporal or Celery worker partitioned by tenant.id.
All LLM calls route through llm_gateway.py — cost_router.py is NOT used here.
All spans carry tenant.id, workflow.id, workflow.run_id.

This module intentionally has no domain-specific workflows/activities of its
own — per SPECS.md §25/§28, "framework workflows are never deployed directly
as tenant production code." Tenant repos copy this file's shape and bind
their own workflows/activities, the same way
examples/oil-price-agent/worker.py does (a complete, working reference).

TENANT_WORKER_MODULE is an alternative to copying this file outright: if
set, runtime/worker.py becomes a thin dispatcher that imports that module
and calls its `start_temporal_worker(tenant_id)` / `start_celery_worker(tenant_id)`
function instead of raising — useful for a tenant repo that wants to keep
using runtime/worker.py as its actual entrypoint script (e.g. as a
container CMD) while supplying its own workflow registration as an
importable module rather than a full copy of this file. Without
TENANT_WORKER_MODULE set, the behavior is unchanged: this module cannot run
anything by itself and says so loudly rather than pretending to.

See SPECS.md §25 for the full production runtime specification.
"""

from __future__ import annotations

import importlib
import os
import sys


def main() -> None:
    """
    Worker entrypoint. Reads WORKER_BACKEND from environment:
      - 'temporal' (default): start Temporal worker
      - 'celery':             start Celery worker

    The tenant is resolved by runtime/tenancy.py: AGENT_TENANT_ID, the legacy
    TENANT_ID, or `tenant.id` in .agenticframework/tenant.yaml. Unresolved is
    fatal — a worker with no tenant would write to a shared ledger unattributed.
    """
    # FIRST, before anything reads configuration. The runtime never loaded .env
    # — only scripts/ did — so a worker started outside a shell that had already
    # exported everything silently ran on defaults. Never overwrites, so a
    # container's injected environment still wins.
    from runtime.config import load_env_file
    from runtime.tenancy import TenantUnresolvedError, resolve_tenant_id

    load_env_file()

    # `workflow.engine` was declared in tenant.yaml and read by nothing while
    # this line took the same fact from an environment variable — the same
    # split that left budget.monthly_usd_cap unenforced.
    from runtime.config import resolve

    backend = str(
        resolve("workflow.engine", env_var="WORKER_BACKEND", default="temporal")
    ).lower()

    # Resolves AGENT_TENANT_ID, then the legacy TENANT_ID this line used to read
    # directly, then tenant.yaml's declaration. Still fatal when unresolved —
    # only the number of places it will look has changed.
    try:
        tenant_id = resolve_tenant_id()
    except TenantUnresolvedError as exc:
        print(f"[worker] ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"[worker] Starting {backend} worker for tenant={tenant_id}")

    # Telemetry BEFORE the worker starts, both signals in one call.
    #
    # `configure_tracing` already existed because KYC installed no
    # TracerProvider and every `agent_span()` in the framework's own testbed was
    # therefore a no-op. Metrics were in the identical position and nobody had
    # noticed: `configure_metrics()` had no caller in this repo, in the tenant,
    # or in the example, so every counter and histogram in runtime/metrics.py
    # wrote into a `_ProxyMeter` that was never resolved. Correct call sites,
    # correct attributes, no provider — the error rate and cache hit ratio the
    # observability audit asked for existed nowhere.
    #
    # Exporters resolve from the environment (runtime/otlp.py). Unset is not an
    # error: the providers, the Resource and the identity processor are
    # installed either way, so the signals are correctly formed and simply not
    # shipped anywhere.
    from runtime.tracing import configure_telemetry
    from runtime.version import warn_if_declared_version_differs

    configure_telemetry()

    # After .env and before any work: `framework.version` in tenant.yaml is a
    # DECLARATION that nothing installs from, so it and the installed package
    # can drift silently. Warns, never refuses — a tenant one line ahead of its
    # own declaration is not a safety problem, and refusing would make bumping
    # the pin and the declaration order-dependent.
    warn_if_declared_version_differs()

    # Say what was ignored. A declaration outranks an ambient export, which is
    # deliberate — but an operator who exports something and sees no effect,
    # with nothing said, will reasonably conclude the framework is broken.
    from runtime.config import shadowed_env

    for var, value in sorted(shadowed_env().items()):
        print(
            f"[worker] NOTE: {var}={value!r} in the environment was IGNORED — "
            f"a file declares it. Add it to `env_overrides:` in tenant.yaml to "
            f"let the environment win.",
            file=sys.stderr,
        )

    if backend == "temporal":
        _start_temporal_worker(tenant_id)
    elif backend == "celery":
        _start_celery_worker(tenant_id)
    else:
        print(f"[worker] ERROR: Unknown WORKER_BACKEND={backend!r}", file=sys.stderr)
        sys.exit(1)


def _load_tenant_worker_module():
    module_name = os.environ.get("TENANT_WORKER_MODULE", "")
    if not module_name:
        return None
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        print(
            f"[worker] ERROR: TENANT_WORKER_MODULE={module_name!r} could not be imported: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)


def _start_temporal_worker(tenant_id: str) -> None:
    """
    Delegates to TENANT_WORKER_MODULE.start_temporal_worker(tenant_id) if
    that env var is set. Without it, this module has no workflows/
    activities to register and cannot start anything — see module
    docstring for the two ways to supply them.

    Reference pattern (what examples/oil-price-agent/worker.py implements
    concretely):
      async with await connect_temporal() as client:
          worker = Worker(
              client,
              task_queue=f"agent-tasks-{tenant_id}",
              workflows=[AgentWorkflow],
              activities=[architect_activity, developer_activity, validator_activity],
          )
          await worker.run()
    """
    module = _load_tenant_worker_module()
    if module is None:
        raise NotImplementedError(
            "No workflows/activities registered. Either copy this file's shape into your "
            "tenant repo (see examples/oil-price-agent/worker.py for a complete example) "
            "or set TENANT_WORKER_MODULE=your_module (exposing start_temporal_worker(tenant_id)). "
            "See SPECS.md §25 and runtime/workflows/."
        )
    if not hasattr(module, "start_temporal_worker"):
        raise AttributeError(
            f"TENANT_WORKER_MODULE={module.__name__!r} has no start_temporal_worker(tenant_id) function."
        )
    module.start_temporal_worker(tenant_id)


def _start_celery_worker(tenant_id: str) -> None:
    """
    Delegates to TENANT_WORKER_MODULE.start_celery_worker(tenant_id) if that
    env var is set — same rationale as _start_temporal_worker above.

    Reference pattern:
      app = Celery("agent_worker", broker=os.environ["REDIS_URL"])
      app.conf.task_routes = {
          "runtime.tasks.*": {"queue": f"agent-{tenant_id}"}
      }
      app.worker_main(argv=["worker", "--loglevel=info"])
    """
    module = _load_tenant_worker_module()
    if module is None:
        raise NotImplementedError(
            "No tasks registered. Either copy this file's shape into your tenant repo "
            "or set TENANT_WORKER_MODULE=your_module (exposing start_celery_worker(tenant_id)). "
            "See SPECS.md §25."
        )
    if not hasattr(module, "start_celery_worker"):
        raise AttributeError(
            f"TENANT_WORKER_MODULE={module.__name__!r} has no start_celery_worker(tenant_id) function."
        )
    module.start_celery_worker(tenant_id)


if __name__ == "__main__":
    main()
