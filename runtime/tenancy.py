"""
runtime/tenancy.py — who this work belongs to, resolved once and carried.

Two problems, one module.

RESOLUTION. `.agenticframework/tenant.yaml` has declared `tenant.id` since the
tenant scaffold shipped, and nothing read it. `llm_gateway.py` opens that exact
file for `gateway.routing_overrides` and walks past the id. So every call site
supplied its own — KYC Sentinel hardcoded `tenant_id="kyc-sentinel"` in two
places, a second copy of a declared fact, which is the "config read from two
places will disagree" lesson with the disagreement not yet arrived.

`resolve_tenant_id()` reads the declaration. Precedence, most specific first:

    1. an explicit argument      a shared-pool worker serving tenant A's request
    2. AGENT_TENANT_ID           per-deployment; what a dedicated pool sets
       or TENANT_ID              the older name, still live in the k8s ConfigMap
    3. tenant.yaml -> tenant.id  the declared default
    4. raise                     never a silent default

Step 4 is deliberate. `tenant.id` partitions the budget ledger, the audit log
and the portal's cross-tenant isolation — it is a control, not a label. An
unresolved tenant that quietly became "unknown" would merge two tenants' spend
and two tenants' audit trail into one bucket, and look fine doing it.

Not derived from the repo or directory name, though it is tempting and would
even work here: KYC is `isolation: dedicated` and single-tenant, so its tenant
happens to equal its repo. The framework default is a SHARED pool partitioned by
tenant_id (SPECS.md §23), where one repo serves many tenants — so a repo-derived
id would pass on the tenant you built it against and silently collapse every
shared-pool tenant into one. Production has no repo either: the span comes from a
container with no `.git` and no GITHUB_REPOSITORY, so it would resolve in CI and
fall back exactly where the audit trail matters. And it would make a compliance
identifier mutable by renaming a directory.

CARRYING. `tenant_id` used to be threaded by hand into every `agent_span()`
call and applied under `if tenant_id:` — which is precisely why it was missing
wherever somebody forgot. Here it lives in a contextvar set once at the activity
boundary, and `AgentIdentityProcessor` stamps it onto every span started inside
that scope. No kwarg to omit.

`agent_role` is in the same place for a reason that is specific to this
architecture: one worker process serves MANY roles. KYC's worker registers
intake, research, analyst, approve, dlq and self-correct activities on a single
task queue. So the role cannot be a Resource attribute — a Resource is fixed for
the process, and stamping all six spans with one role would be five confident
lies, which is worse than an absent attribute.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Iterator, Optional

from runtime.config import config_get

TENANT_ENV_VAR = "AGENT_TENANT_ID"

# The name `runtime/worker.py` and the dedicated-tenant ConfigMap have always
# used. Accepted, not renamed: it is live in a k8s manifest, and a rename would
# strand any cluster running the previous chart with a worker that refuses to
# start. AGENT_TENANT_ID wins where both are set — the prefixed name is the
# convention every other framework variable follows.
LEGACY_TENANT_ENV_VAR = "TENANT_ID"

_tenant_id: ContextVar[Optional[str]] = ContextVar("agentsmith_tenant_id", default=None)
_agent_role: ContextVar[Optional[str]] = ContextVar("agentsmith_agent_role", default=None)
_run_id: ContextVar[Optional[str]] = ContextVar("agentsmith_run_id", default=None)


class TenantUnresolvedError(RuntimeError):
    """No tenant could be resolved from argument, environment or tenant.yaml."""


def tenant_id_from_config(root: Optional[Path] = None) -> Optional[str]:
    """`tenant.id` from `.agenticframework/tenant.yaml`, or None.

    None rather than a raise: absence is a normal state for a repo that has not
    been scaffolded, and the caller decides whether that is fatal.
    """
    value = config_get("tenant.id", root)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def resolve_tenant_id(
    explicit: Optional[str] = None, *, root: Optional[Path] = None
) -> str:
    """The tenant this work belongs to. Raises rather than guessing."""
    if explicit and explicit.strip():
        return explicit.strip()

    for var in (TENANT_ENV_VAR, LEGACY_TENANT_ENV_VAR):
        from_env = os.environ.get(var, "").strip()
        if from_env:
            return from_env

    from_config = tenant_id_from_config(root)
    if from_config:
        return from_config

    raise TenantUnresolvedError(
        "No tenant id. Pass one explicitly, set "
        f"{TENANT_ENV_VAR} (or {LEGACY_TENANT_ENV_VAR}), or declare `tenant.id` in "
        ".agenticframework/tenant.yaml. Refusing to default: tenant.id "
        "partitions the budget ledger, the audit log and cross-tenant "
        "isolation, so an unattributed run would merge two tenants' records."
    )


@contextmanager
def agent_context(
    *,
    role: Optional[str] = None,
    tenant_id: Optional[str] = None,
    run_id: Optional[str] = None,
) -> Iterator[None]:
    """Bind identity for everything that runs inside.

    Wrap an activity — the boundary that already knows which role it is — and
    every span started beneath carries the identity, including spans from the
    gateway and from tool calls the tenant has not instrumented itself.

    Restores the previous values on exit, so nesting is safe and one activity
    cannot leak its role into the next on a reused worker thread.
    """
    tokens = [
        (_tenant_id, _tenant_id.set(tenant_id if tenant_id else _tenant_id.get())),
        (_agent_role, _agent_role.set(role if role else _agent_role.get())),
        (_run_id, _run_id.set(run_id if run_id else _run_id.get())),
    ]
    try:
        yield
    finally:
        for var, token in reversed(tokens):
            var.reset(token)


def current_identity() -> dict[str, Any]:
    """The bound identity, omitting anything unset.

    Omitted rather than "unknown": a missing attribute is a visible gap, while
    a plausible placeholder is indistinguishable from a real value and would be
    aggregated as one.
    """
    values = {
        "tenant.id": _tenant_id.get(),
        "agent.role": _agent_role.get(),
        "run.id": _run_id.get(),
    }
    return {k: v for k, v in values.items() if v}


def current_tenant_id() -> Optional[str]:
    return _tenant_id.get()
