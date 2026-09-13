"""
scripts/security/runners/delegating.py — controls whose verification already
exists elsewhere in the repo.

These were the largest category of `runner … not implemented`: 14 of 23
controls skipped, and `skip` counts as green even under `--strict`, so the
harness reported success for a control surface it had never examined.
`SEC-HITL-001` — mandatory human review — was among them, while a live run
showed that gate failing open on a sanctions hit.

Each runner here is a few lines because the check it needs is already written
and already enforced by CI. Adding a second implementation would create a
control that can disagree with the tests, which is worse than no control: it
would eventually report Met while the behaviour regressed.

One module rather than one file per control, deliberately. These are bindings,
not logic; splitting them would make the harness look substantial while saying
the same thing.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

from security.registry import ControlSpec
from security.report import ControlResult
from security.runners._shared import (
    eval_suite_gateable,
    failed,
    framework_component_absent,
    guard_suite,
    node_suite,
    passed,
    pytest_suite,
    tenant_security,
    verify_system,
)


# ── Delegating straight to an existing suite or health check ─────────────────


def hitl_gate(control: ControlSpec, ctx: dict[str, Any]) -> ControlResult:
    """SEC-HITL-001 — `runtime/test/test_hitl_gate.py` already asserts that the
    gate cannot be skipped, that a caller must supply exactly one of
    `gate_activity_name`/`gate_result`, and that a timeout dead-letters rather
    than approving. It runs without Temporal or Postgres, so it is a usable
    control check rather than an integration test."""
    return pytest_suite(control, ctx, "runtime/test/test_hitl_gate.py")


def dlq_check(control: ControlSpec, ctx: dict[str, Any]) -> ControlResult:
    """SEC-DLQ-001 — the dead-letter envelope contract.

    Bound to `test_dead_letter.py`, NOT to `verify_system --check-dlq`. That
    flag is a reachability probe ("DLQ reachable — DATABASE_URL"), so binding
    it would make the control fail whenever Postgres is down — an availability
    check wearing a compliance label, which is the confusion this whole phase
    exists to remove.

    Nor is it bound to `test_hitl_gate.py`, which is SEC-HITL-001's evidence
    and where these assertions used to live. A suite that proves two controls
    proves neither independently: it would have gone green on the HITL gate
    while the recoverable-step producer hand-rolled its own envelope, which is
    exactly the state the move out of that module found.

    What this proves without infrastructure: both producers build the envelope
    through one builder, the builder's keys are what `enqueue` accepts, and the
    generic activity rejects the legacy flattened shape by name. Whether a row
    lands in Postgres is an integration concern and stays out.
    """
    return pytest_suite(control, ctx, "runtime/test/test_dead_letter.py")


def self_correction(control: ControlSpec, ctx: dict[str, Any]) -> ControlResult:
    """SEC-SELF-001 — the recoverable-step / self-correction wrappers.

    Was bound to scripts/test/test_workflow_template_wiring.py — a check of
    workflow YAML consistency (callee references, the hooks/post-checkout
    array matching runtime/cli.py's WORKFLOWS), unrelated to self-correction
    and, being framework-provisioning-relative, unable to even resolve in a
    tenant (it reads workflow-templates/ and hooks/ off its own REPO,
    file-relative — those don't exist in a tenant either). Found while
    working through why SEC-SELF-001 couldn't be vendored into
    AqlaarTeleologyStudio: it never evidenced this control at all, in the
    framework's own self-test or anywhere else. test_self_correction.py
    is the real evidence — infra-free (fakes the gateway, no Temporal), same
    design as test_hitl_gate.py and test_dead_letter.py above.
    """
    return pytest_suite(control, ctx, "runtime/test/test_self_correction.py")


def budget_caps(control: ControlSpec, ctx: dict[str, Any]) -> ControlResult:
    """SEC-BUDGET-001 — budget reservation and the degrade ladder on breach.

    Backends pinned to in-process: the control verifies that a breach degrades
    and ultimately halts, which is code behaviour. Left to inherit the tenant
    .env it picked up `BUDGET_BACKEND=postgres` against a database that was not
    running and reported a control failure for an unrelated reason.
    """
    return pytest_suite(
        control, ctx, "runtime/test/test_llm_gateway_budget.py",
        env={"BUDGET_BACKEND": "memory", "IDEMPOTENCY_BACKEND": "memory"},
    )


def change_gates(control: ControlSpec, ctx: dict[str, Any]) -> ControlResult:
    """SEC-CHANGE-001 — the hooks that enforce RFC/commit discipline.

    `--check-hooks` runs the framework's own `hooks/pre-commit` and
    `hooks/commit-msg` in throwaway repos. A vendored tenant has no `hooks/`,
    so without the guard every tenant failed this control on
    `bash: …/hooks/pre-commit: No such file` — a missing framework file, not a
    tenant's change discipline."""
    absent = framework_component_absent(control, ctx, "hooks")
    if absent is not None:
        return absent
    return verify_system(control, ctx, "--check-hooks")


def rbac_matrix(control: ControlSpec, ctx: dict[str, Any]) -> ControlResult:
    """SEC-RBAC-001 — the portal's role/permission matrix."""
    return node_suite(control, ctx, "test/authz.test.ts", requires=("lib/authz.ts",))


def rag_poison(control: ControlSpec, ctx: dict[str, Any]) -> ControlResult:
    """SEC-RAG-001 — poisoned retrieved context is quarantined before use.

    Retrieval-borne injection is the same attack as a direct prompt injection
    arriving by a different route: the text comes from the corpus, so guarding
    only the user's turn leaves the whole RAG path open. An attacker who can add
    a document writes the instruction once and it arrives inside trusted
    context.

    Scored by `runtime.prompt_guard.scan_documents` over
    `fixtures/rag_poison_base.json`, which pairs every poisoned document with a
    benign twin on the same subject. Both directions count toward one miss rate,
    because a guard that quarantines everything defeats the attack and destroys
    retrieval — and would otherwise score perfectly.

    What this claims: poisoned context is DETECTED and dropped before assembly.
    What it does not claim: that a model would have resisted the instruction had
    the document reached it. Those are different properties and only the first
    is deterministic, which is why this control gates on every commit.
    """
    return guard_suite(control, ctx, "rag_poison", "score_rag_poison_case")


def audit_hmac(control: ControlSpec, ctx: dict[str, Any]) -> ControlResult:
    """SEC-AUDIT-001 — audit-log tamper-evidence, without a database.

    Bound to `test/auditSignature.test.ts`, not `test/auditLog.test.ts`. The
    latter exits at import time without DATABASE_URL, so binding it would have
    made the control fail whenever Postgres was down — which is why it was
    declared a gap rather than wired to something that could not run.

    This proves only the half that needs no infrastructure: a mutated event
    stops verifying, key order in `details` does not change the signature, a
    malformed signature is rejected rather than throwing, and a missing key
    refuses instead of signing with a default. Append-only enforcement is a
    database trigger and is claimed separately by SEC-AUDIT-002.
    """
    return node_suite(
        control, ctx, "test/auditSignature.test.ts", requires=("lib/auditSignature.ts",)
    )


# ── Eval gates: wired and gateable, without running a judge ──────────────────


def eval_golden(control: ControlSpec, ctx: dict[str, Any]) -> ControlResult:
    return eval_suite_gateable(control, ctx, "golden")


def eval_fairness(control: ControlSpec, ctx: dict[str, Any]) -> ControlResult:
    return eval_suite_gateable(control, ctx, "fairness")


def eval_hallucination(control: ControlSpec, ctx: dict[str, Any]) -> ControlResult:
    return eval_suite_gateable(control, ctx, "hallucination")


# ── Tenant-declared controls ─────────────────────────────────────────────────


def tenant_suite(control: ControlSpec, ctx: dict[str, Any]) -> ControlResult:
    """A control the TENANT declares, evidenced by a test suite in its own repo.

    The framework registry cannot enumerate every domain control a tenant needs
    — KYC Sentinel's evidence-mandated rating floor (a sanctions hit forces
    human review whatever the model rated) is a real control with tests and
    documentation that the compliance surface simply could not see.

    Deliberately runs the tenant's own suite rather than importing a declared
    callable: the tests already encode what the control claims, including its
    negative cases, and a second assertion written here could drift from them.
    Tenant registries are additive-only (see load_control_registry), so this
    cannot be used to weaken a framework control.
    """
    if not control.suite:
        return failed(control, "tenant control declares no `suite` to run")
    return pytest_suite(control, ctx, control.suite, base=Path(ctx["tenant_root"]))


def agency_manifest(control: ControlSpec, ctx: dict[str, Any]) -> ControlResult:
    """SEC-AGENCY-001 — the repo declares which actions need a human.

    An artifact control: it verifies the manifest is present, parseable, and
    ACTUALLY EDITED. The shipped template names `example_workflow` /
    `high_impact_step`, and a repo carrying that verbatim has declared nothing
    while appearing compliant — the same defect `risk_register` catches with
    its `RISK-EXAMPLE-*` check, and the same one this framework's own pack had
    before it was filled in.

    Also requires at least one action with `needs_hitl: true`. A manifest where
    nothing needs a human is not a governance record; it is an empty claim.
    """
    import yaml

    path = tenant_security(ctx) / "agency_manifest.yaml"
    if not path.exists():
        return failed(control, f"no agency manifest at {path}")
    try:
        doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        return failed(control, f"agency manifest does not parse: {exc}")

    actions = doc.get("actions") or []
    if not actions:
        return failed(control, "agency manifest declares no actions", path=str(path))

    placeholders = [
        a for a in actions
        if a.get("workflow") == "example_workflow" or a.get("action") == "high_impact_step"
    ]
    if placeholders:
        return failed(
            control,
            f"agency manifest still contains the shipped placeholder "
            f"({len(placeholders)} entry/entries) — nothing has been declared",
            path=str(path),
        )

    gated = [a for a in actions if a.get("needs_hitl") is True]
    if not gated:
        return failed(
            control,
            f"{len(actions)} action(s) declared, none needing human review — "
            f"a manifest where nothing is gated records no governance",
            path=str(path),
        )
    return passed(
        control,
        f"{len(actions)} action(s) declared, {len(gated)} gated on human review",
        path=str(path),
        gated=",".join(sorted(a.get("action", "?") for a in gated)[:3]),
    )


# ── Static: the gateway is the only provider path for workload calls ─────────

# Provider SDKs and the eval-path router. runtime/llm_gateway.py's own
# docstring states the rule this enforces: "Workers MUST NOT import
# cost_router.py directly."
_FORBIDDEN_IN_WORKLOAD = {
    "anthropic", "openai", "groq", "cohere", "mistralai",
    "google.generativeai", "boto3", "cost_router",
}
# Where a tenant's workload code lives. Tests and scripts are exempt: the eval
# harness legitimately uses cost_router, and test doubles import SDKs.
_WORKLOAD_DIRS = ("agents", "workflows", "runtime/workflows")


def gateway_static(control: ControlSpec, ctx: dict[str, Any]) -> ControlResult:
    """SEC-GW-001 — workload code reaches models through the gateway only.

    The map row has always described this as "Static: tenant activities import
    gateway, not raw provider"; nothing performed it. A direct SDK import
    bypasses budget reservation, the degrade ladder, redaction, prompt guard
    and the moderation hook in one step — every gateway control at once — and
    it is invisible at runtime because the call simply succeeds.
    """
    root = Path(ctx["root"])
    offenders: list[str] = []
    for rel in _WORKLOAD_DIRS:
        base = root / rel
        if not base.is_dir():
            continue
        for py in base.rglob("*.py"):
            if "test" in py.parts or py.name.startswith("test_"):
                continue
            try:
                tree = ast.parse(py.read_text(encoding="utf-8", errors="ignore"))
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module]
                for name in names:
                    root_pkg = name.split(".")[0]
                    if name in _FORBIDDEN_IN_WORKLOAD or root_pkg in _FORBIDDEN_IN_WORKLOAD:
                        offenders.append(f"{py.relative_to(root)}: {name}")

    if offenders:
        return ControlResult(
            control_id=control.id,
            status="fail",
            message=f"{len(offenders)} workload import(s) bypass the gateway",
            evidence={"offenders": "; ".join(sorted(set(offenders))[:5])},
        )
    scanned = [d for d in _WORKLOAD_DIRS if (root / d).is_dir()]
    return ControlResult(
        control_id=control.id,
        status="pass",
        message=f"no direct provider imports in {', '.join(scanned) or '(no workload dirs)'}",
        evidence={"dirs": ",".join(scanned)},
    )
