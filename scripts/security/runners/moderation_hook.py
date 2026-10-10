from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from security.registry import ControlSpec
from security.report import ControlResult
from security.runners._shared import (
    declared_choice,
    declared_value,
    failed,
    framework_root,
    is_contract,
    passed,
    provider_code_in_scope,
    repository_env,
    repository_python,
)

MODES = ("off", "optional", "required")

# The declared hook, called in the REPOSITORY's interpreter: a contract run never
# imports the repository's code into the provider's process. The hook's own
# policy decides what is unsafe, so only the contract is asserted — it answers,
# with an `allowed`, and lets benign text through.
_HOOK_SMOKE = """
import importlib, json, sys
module, _, name = sys.argv[1].partition(":")
answer = getattr(importlib.import_module(module), name)("The weather forecast for tomorrow.")
allowed = answer.get("allowed") if isinstance(answer, dict) else getattr(answer, "allowed", None)
print(json.dumps({"allowed": allowed if isinstance(allowed, bool) else None}))
"""


def _api_smoke(mod) -> str:
    """The provider's moderation API: '' when it behaves, else what did not."""
    mod.reset_output_moderator()
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
        return "moderator smoke failed"
    try:
        mod.apply_output_moderation("x", mode="required", use_declared=False)
    except mod.ModerationHookRequiredError:
        return ""
    return "required mode did not raise without hook"


def _declared(control: ControlSpec, ctx: dict[str, Any]) -> ControlResult:
    """A contract run: the mode the repository DECLARES, and under `required`
    the hook it declares, answering in its own interpreter."""
    import json
    import subprocess

    if provider_code_in_scope(ctx):
        from runtime import moderation as mod

        broken = _api_smoke(mod)
        if broken:
            return failed(control, broken)
    mode, problem = declared_choice(ctx, "moderation.mode", MODES)
    if problem:
        return failed(control, problem)
    if mode is None:
        return failed(control, "moderation.mode is not declared in .agenticframework/tenant.yaml — declare "
                               "off, optional or required")
    if mode != "required":
        return passed(control, f"moderation.mode {mode!r} declared", mode=mode)
    declared, _ = declared_value(ctx, "moderation.hook")
    hook = str(declared or "").strip()
    if ":" not in hook:
        return failed(control, "moderation.mode is 'required' and moderation.hook names no module.path:callable",
                      mode=mode)
    root = Path(ctx["tenant_root"])
    env = repository_env(Path(ctx["root"]))
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(root), env.get("PYTHONPATH")]))
    try:
        done = subprocess.run([repository_python(), "-c", _HOOK_SMOKE, hook], cwd=root, env=env,
                              capture_output=True, text=True, timeout=120, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return failed(control, f"declared hook {hook} could not run: {exc}", mode=mode, hook=hook)
    last = (done.stdout.strip().splitlines() or [""])[-1]
    try:
        allowed = json.loads(last).get("allowed") if done.returncode == 0 else None
    except (ValueError, AttributeError):
        allowed = None
    if done.returncode != 0:
        return failed(control, f"declared hook {hook} raised on benign text: {done.stderr.strip()[-300:]}",
                      mode=mode, hook=hook)
    if allowed is None:
        return failed(control, f"declared hook {hook} did not answer with an `allowed`", mode=mode, hook=hook)
    if not allowed:
        return failed(control, f"declared hook {hook} blocked benign text — a classifier that blocks everything "
                               "is not a passing control", mode=mode, hook=hook)
    return passed(control, f"moderation.mode 'required'; declared hook {hook} answers", mode=mode, hook=hook)


def run(control: ControlSpec, ctx: dict[str, Any]) -> ControlResult:
    framework_root(ctx)   # sys.path side effect; return value unused
    if is_contract(ctx):
        return _declared(control, ctx)

    from runtime import moderation as mod

    strict = bool(ctx.get("strict", False)) or os.environ.get("SECURITY_STRICT", "") == "1"
    mode = os.environ.get("MODERATION_HOOK", "").strip().lower()

    # API smoke: a classifier allows clean and blocks unsafe, and required mode
    # rejects a genuinely hook-less tenant — use_declared=False isolates that
    # from any hook the tenant HAS declared, or it would fail for exactly the
    # well-configured tenants it is meant to protect (G10).
    broken = _api_smoke(mod)
    if broken:
        return ControlResult(control_id=control.id, status="fail", message=broken, evidence={})

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
