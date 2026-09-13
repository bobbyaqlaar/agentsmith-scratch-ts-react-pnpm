from __future__ import annotations

from collections.abc import Callable
from typing import Any

from security.registry import ControlSpec
from security.report import ControlResult
from security.runners import (
    adversarial_eval,
    delegating,
    moderation_hook,
    noop,
    pii_postcall,
    pii_precall,
    prompt_guard,
    risk_register,
    sovereign_residency,
    sso_revocation,
    structured_output,
    tool_allowlist,
)

Runner = Callable[[ControlSpec, dict[str, Any]], ControlResult]

RUNNERS: dict[str, Runner] = {
    "noop": noop.run,
    "pii_precall": pii_precall.run,
    "pii_postcall": pii_postcall.run,
    "prompt_guard": prompt_guard.run,
    "risk_register": risk_register.run,
    "structured_output": structured_output.run,
    "tool_allowlist": tool_allowlist.run,
    "adversarial_eval": adversarial_eval.run,
    "moderation_hook": moderation_hook.run,
    "sso_revocation": sso_revocation.run,
    "sovereign_residency": sovereign_residency.run,
    # Bindings to verification that already exists — see delegating.py.
    "hitl_gate": delegating.hitl_gate,
    "dlq_check": delegating.dlq_check,
    "self_correction": delegating.self_correction,
    "budget_caps": delegating.budget_caps,
    "change_gates": delegating.change_gates,
    "eval_golden": delegating.eval_golden,
    "eval_fairness": delegating.eval_fairness,
    "eval_hallucination": delegating.eval_hallucination,
    "gateway_static": delegating.gateway_static,
    "tenant_suite": delegating.tenant_suite,
    "rbac_matrix": delegating.rbac_matrix,
    "audit_hmac": delegating.audit_hmac,
    "rag_poison": delegating.rag_poison,
    "agency_manifest": delegating.agency_manifest,
}
