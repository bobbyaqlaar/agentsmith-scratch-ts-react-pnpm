from __future__ import annotations

from pathlib import Path
from typing import Any

from security.registry import ControlSpec
from security.report import ControlResult
from security.runners._shared import failed, framework_root, is_contract, not_applicable, passed, verify_system


def run(control: ControlSpec, ctx: dict[str, Any]) -> ControlResult:
    """SEC-PII-002 — post-call redaction, verified by verify_system.

    ENVIRONMENT=staging is forced by the shared helper: the redaction check
    self-disables under `development`, so running it from a developer shell
    would report Met while checking nothing.

    In a contract run it is the repository's own wire: its declared emitter,
    under the staging and the production profile, with planted probes
    (security/redaction.py).
    """
    if not is_contract(ctx):
        return verify_system(control, ctx, "--check-redaction")
    framework_root(ctx)  # runtime.telemetry_contract, for the loopback receiver
    from security.redaction import check, declared_emitter

    root = Path(ctx["tenant_root"])
    emitter = declared_emitter(root)
    if emitter == "none":
        return not_applicable(control, "the repository declares it emits no telemetry")
    if emitter is None:
        return failed(control, "nothing verifies what this repository's telemetry carries — declare its emitter "
                               'in providers.json `security.emitter`, or `"none"`')
    answers = {environment: check(root, environment, emitter) for environment in ("staging", "production")}
    short = [f"{env}: {a.verdict.replace('_', ' ')} — {a.reason}" for env, a in answers.items() if a.verdict != "pass"]
    if short:
        return failed(control, "; ".join(short), emitter=str(emitter))
    return passed(control, "no probe reached the wire under the staging or the production profile",
                  emitter=str(emitter))
