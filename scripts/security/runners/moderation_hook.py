from __future__ import annotations

import os
from typing import Any

from security.registry import ControlSpec
from security.report import ControlResult
from security.runners._shared import framework_root


def run(control: ControlSpec, ctx: dict[str, Any]) -> ControlResult:
    framework_root(ctx)   # sys.path side effect; return value unused

    from runtime import moderation as mod

    mod.reset_output_moderator()
    strict = bool(ctx.get("strict", False)) or os.environ.get("SECURITY_STRICT", "") == "1"
    mode = os.environ.get("MODERATION_HOOK", "").strip().lower()

    # API smoke: register classifier, allow clean, block unsafe.
    mod.register_output_moderator(
        lambda t: mod.ModerationResult(
            allowed="unsafe" not in t.lower(),
            reasons=["policy"] if "unsafe" in t.lower() else [],
        )
    )
    try:
        ok = mod.apply_output_moderation("safe output")
        bad = mod.apply_output_moderation("unsafe payload")
        try:
            mod.apply_output_moderation("unsafe payload", raise_on_block=True)
            raised = False
        except mod.ModerationBlockedError:
            raised = True
    finally:
        mod.reset_output_moderator()

    if not ok.allowed or bad.allowed or not raised:
        return ControlResult(
            control_id=control.id,
            status="fail",
            message="moderator smoke failed",
            evidence={},
        )

    # Prove required mode rejects a genuinely hook-less tenant.
    # use_declared=False isolates this from any hook the tenant HAS declared —
    # otherwise this assertion would fail for exactly the well-configured
    # tenants it is meant to protect (G10).
    rejected_missing_hook = False
    try:
        mod.apply_output_moderation("x", mode="required", use_declared=False)
    except mod.ModerationHookRequiredError:
        rejected_missing_hook = True

    if not rejected_missing_hook:
        return ControlResult(
            control_id=control.id,
            status="fail",
            message="required mode did not raise without hook",
            evidence={},
        )

    # Tenant ownership:
    # - MODERATION_HOOK=required → the tenant must DECLARE a hook the harness
    #   can import and smoke-test (moderation.hook in tenant.yaml, or
    #   MODERATION_HOOK_PATH). Before G10 this branch failed unconditionally,
    #   because an imperative register_output_moderator() call happens in the
    #   worker process and is invisible here — so `required`, the setting
    #   regulated tenants are told to use, could never pass CI.
    # - strict + unset → fail (forces explicit optional/required/off)
    # - optional/off → pass after API smoke (even under SECURITY_STRICT)
    # - unset + non-strict → warn
    if mode == "required":
        declared = mod.declared_hook_path()
        if not declared:
            return ControlResult(
                control_id=control.id,
                status="fail",
                message=(
                    "MODERATION_HOOK=required but no hook declared — set "
                    "moderation.hook in .agenticframework/tenant.yaml "
                    "(module.path:callable) or MODERATION_HOOK_PATH"
                ),
                evidence={"mode": mode},
            )

        try:
            tenant_fn = mod.load_declared_moderator()
        except mod.ModerationHookImportError as exc:
            return ControlResult(
                control_id=control.id,
                status="fail",
                message=f"declared moderation hook unusable: {exc}",
                evidence={"mode": mode, "hook": declared},
            )

        # Smoke the TENANT's classifier, not the framework's lambda: this is
        # what turns SEC-MOD-001 from "the API exists" into "this tenant has
        # a working classifier". Its own policy decides what is unsafe, so
        # only the contract is asserted — a ModerationResult, and a clean
        # string must not be blocked.
        mod.reset_output_moderator()
        mod.register_output_moderator(tenant_fn)
        try:
            clean = mod.apply_output_moderation("The weather forecast for tomorrow.")
        except Exception as exc:
            mod.reset_output_moderator()
            return ControlResult(
                control_id=control.id,
                status="fail",
                message=f"declared hook {declared} raised on benign text: {exc}",
                evidence={"mode": mode, "hook": declared},
            )
        finally:
            mod.reset_output_moderator()

        if not isinstance(clean, mod.ModerationResult):
            return ControlResult(
                control_id=control.id,
                status="fail",
                message=f"declared hook {declared} did not return a ModerationResult",
                evidence={"mode": mode, "hook": declared},
            )
        if not clean.allowed:
            return ControlResult(
                control_id=control.id,
                status="fail",
                message=(
                    f"declared hook {declared} blocked benign text — a classifier "
                    "that blocks everything is not a passing control"
                ),
                evidence={"mode": mode, "hook": declared},
            )

        return ControlResult(
            control_id=control.id,
            status="pass",
            message=f"tenant moderator declared and verified ({declared})",
            evidence={"mode": mode, "hook": declared},
        )
    if mode in ("optional", "off"):
        return ControlResult(
            control_id=control.id,
            status="pass",
            message=f"moderation API smoke ok (MODERATION_HOOK={mode})",
            evidence={"mode": mode},
        )
    if strict:
        return ControlResult(
            control_id=control.id,
            status="fail",
            message="no output moderator registered (strict; set MODERATION_HOOK=optional|required|off)",
            evidence={"mode": "unset"},
        )

    return ControlResult(
        control_id=control.id,
        status="warn",
        message="moderation hook unset (optional) — tenant should register for regulated content",
        evidence={"mode": "unset"},
    )
