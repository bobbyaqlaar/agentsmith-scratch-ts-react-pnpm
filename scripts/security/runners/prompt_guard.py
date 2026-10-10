"""SEC-PROMPT-001 runner — prompt-injection detection AND enforcement.

Two things are checked, because passing only the first is how a control
reports "Met" while nothing is actually blocked in production
(TestbedFeedback-2026-07-21 G9):

  1. Detection — the heuristics classify the fixture corpus correctly.
  2. Enforcement — the configured PROMPT_GUARD mode actually blocks. A
     tenant running the observe-first `warn` tier is deliberately NOT
     enforcing, so that is reported as a warn (visible, and a strict-CI
     failure) rather than a silent pass.
"""

from __future__ import annotations

from typing import Any

from security.registry import ControlSpec
from security.report import ControlResult
from security.runners._shared import (
    declared_choice,
    failed,
    framework_root,
    is_contract,
    passed,
    provider_code_in_scope,
    security_fixture,
)

MODES = ("off", "warn", "default", "strict")


def _detection(control: ControlSpec, ctx: dict[str, Any]) -> tuple[int, ControlResult | None]:
    """The heuristics against the provider's own corpus: (cases, failure)."""
    from runtime.prompt_guard import scan_prompt

    cases, problem = security_fixture(control, ctx, "prompt_injection_cases_base.json")
    if problem is not None:
        return 0, problem
    failures: list[str] = []
    for case in cases:
        result = scan_prompt(case["input"])
        expected = bool(case["expect_blocked"])
        if result.blocked != expected:
            failures.append(
                f"{case['id']}: expected blocked={expected} got {result.blocked}"
            )

    if failures:
        return len(cases), ControlResult(
            control_id=control.id,
            status="fail",
            message="; ".join(failures[:5]),
            evidence={"failures": str(len(failures))},
        )
    return len(cases), None


def _declared(control: ControlSpec, ctx: dict[str, Any]) -> ControlResult:
    """A contract run: the posture the repository DECLARES enforces. The
    environment of whoever runs the check is not a declaration; detection is the
    provider's own evidence, checked here only in the provider's repository."""
    from runtime.prompt_guard import is_enforcing

    checked = ""
    if provider_code_in_scope(ctx):
        count, failure = _detection(control, ctx)
        if failure is not None:
            return failure
        checked = f"detection passed {count} cases; "
    mode, problem = declared_choice(ctx, "security.prompt_guard", MODES)
    if problem:
        return failed(control, problem)
    said = "declared" if mode else "not declared — the default"
    mode = mode or "default"
    if not is_enforcing(mode):
        return failed(control, f"{checked}security.prompt_guard is {mode!r} ({said}) — a flagged prompt is not "
                               "blocked; declare `default` or `strict`", mode=mode)
    return passed(control, f"{checked}security.prompt_guard {mode!r} enforces ({said})", mode=mode)


def run(control: ControlSpec, ctx: dict[str, Any]) -> ControlResult:
    framework_root(ctx)   # sys.path side effect; return value unused
    if is_contract(ctx):
        return _declared(control, ctx)

    from runtime.prompt_guard import is_enforcing, resolve_mode

    count, failure = _detection(control, ctx)
    if failure is not None:
        return failure

    # ── Enforcement ──────────────────────────────────────────────────────
    # Detection alone proves the heuristics work, not that anything is
    # blocked. `off` is a real gap; `warn` is a legitimate rollout posture
    # but still not enforcement, so both surface rather than passing.
    mode = resolve_mode()
    evidence = {"cases": str(count), "mode": mode}

    if mode == "off":
        return ControlResult(
            control_id=control.id,
            status="fail",
            message=(
                f"detection passed {count} cases but PROMPT_GUARD=off — "
                "no prompt is scanned in this environment"
            ),
            evidence=evidence,
        )

    if not is_enforcing(mode):
        return ControlResult(
            control_id=control.id,
            status="warn",
            message=(
                f"detection passed {count} cases; PROMPT_GUARD={mode} reports "
                "without blocking (observe-first tier). Set PROMPT_GUARD=default "
                "to enforce before promoting to production."
            ),
            evidence=evidence,
        )

    return ControlResult(
        control_id=control.id,
        status="pass",
        message=f"prompt_guard passed {count} cases; enforcing (mode={mode})",
        evidence=evidence,
    )
