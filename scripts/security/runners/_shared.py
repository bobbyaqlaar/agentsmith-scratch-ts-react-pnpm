"""
scripts/security/runners/_shared.py — the three ways a control runner borrows
an existing verifier, extracted so runners stay thin.

Every check the harness needs already exists somewhere: `verify_system.py` has
the `--check-*` family, `run-evals.py` owns fixture loading and thresholds, and
`runtime/test/` and `scripts/test/` hold suites that already assert the
behaviour a control claims. A runner's job is to point at one of those and
translate its outcome into a ControlResult — not to re-implement the check,
which would give the harness a second opinion that can drift from the one CI
enforces.

Both helpers below were inlined in a single runner each (`pii_postcall` shelled
out to verify_system, `adversarial_eval` loaded run-evals) and would have been
copied into a dozen more.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

from security.registry import ControlSpec
from security.report import ControlResult


def _subprocess_result(
    control: ControlSpec, proc: subprocess.CompletedProcess, ok_msg: str, fail_msg: str
) -> ControlResult:
    if proc.returncode != 0:
        return failed(control, fail_msg, stderr=(proc.stderr or proc.stdout)[:500])
    return passed(control, ok_msg)


def verify_system(
    control: ControlSpec,
    ctx: dict[str, Any],
    flag: str,
    env: Optional[dict[str, str]] = None,
) -> ControlResult:
    """Delegate to `verify_system.py <flag>` — the health check CI already runs.

    ENVIRONMENT is forced to `staging` by default: several checks self-disable
    under `development`, so a control that ran in a developer shell would
    report Met while verifying nothing.
    """
    root = Path(ctx["root"])
    proc = subprocess.run(
        [sys.executable, str(root / "scripts" / "verify_system.py"), flag],
        cwd=root,
        capture_output=True,
        text=True,
        env={**os.environ, "ENVIRONMENT": "staging", **(env or {})},
        check=False,
    )
    return _subprocess_result(
        control, proc, f"verify_system {flag} passed", f"verify_system {flag} failed"
    )


# Settings that configure a TENANT's deployment and guardrail posture rather
# than framework code. The harness runs from the tenant's directory with its
# .env loaded and its CI-set modes exported, so a framework suite would inherit
# them: `MODERATION_HOOK=required` makes the gateway raise when no hook is
# declared, `BUDGET_BACKEND=postgres` points at a database that need not be
# running. Three budget tests failed that way and read as a compliance breach.
_TENANT_RUNTIME_KEYS = (
    "MODERATION_HOOK", "PROMPT_GUARD", "INPUT_GUARDRAIL", "TOOL_ALLOWLIST_STRICT",
    "TOOL_ALLOWLIST_PATH", "PROMPT_DENYLIST_PATH", "SECURITY_STRICT",
    "BUDGET_BACKEND", "IDEMPOTENCY_BACKEND", "DATABASE_URL",
    "AGENT_MONTHLY_USD_CAP", "TENANT_ID", "AI_STACK_MODE", "AGENT_MODEL_PROFILE",
)


def pytest_suite(
    control: ControlSpec,
    ctx: dict[str, Any],
    rel_path: str,
    env: Optional[dict[str, str]] = None,
    select: Optional[str] = None,
    base: Optional[Path] = None,
    python: Optional[str] = None,
) -> ControlResult:
    """Delegate to an existing test module.

    Where a suite already asserts exactly what a control claims, running it is
    strictly better than writing a second check: one behaviour, one assertion,
    and the control cannot quietly disagree with the tests.

    `base` selects which repo the path is relative to — the framework checkout
    by default, or the tenant root for a tenant-declared suite. One helper
    serves both rather than a second subprocess path that could drift.

    Tenant deployment and guardrail settings are stripped first — see
    `_TENANT_RUNTIME_KEYS`. Without that, a framework suite runs under the
    tenant's posture and fails for reasons that have nothing to do with the
    control, turning a compliance check into an availability check. `env` lets
    a caller pin back whatever the suite genuinely needs; `select` passes a
    `-k` expression; `python` is the interpreter — the repository's own, for a
    suite of the repository's in a contract run.
    """
    root = base or Path(ctx["root"])
    target = root / rel_path
    if not target.exists():
        return ControlResult(
            control_id=control.id,
            status="fail",
            message=f"{rel_path} is missing — nothing verifies this control",
            evidence={},
        )
    clean = {k: v for k, v in os.environ.items() if k not in _TENANT_RUNTIME_KEYS}
    cmd = [python or sys.executable, "-m", "pytest", str(target), "-q"]
    if select:
        cmd += ["-k", select]
    run_env = {**clean, "ENVIRONMENT": "staging", **(env or {})}
    proc = subprocess.run(
        cmd,
        cwd=root,
        capture_output=True,
        text=True,
        env=repository_env(Path(ctx["root"]), run_env) if python else run_env,
        check=False,
    )
    # pytest exit 2 = collection/usage error: the suite could not RUN (a
    # missing dependency, an import error), which is categorically different
    # from a suite that ran and failed. Reporting a missing dev package as a
    # control violation is the same availability-as-compliance confusion this
    # phase exists to remove — but it is still a gap, so it fails, with a
    # message that names the cause.
    if proc.returncode == 2:
        return failed(
            control,
            f"{rel_path} could not run — check dependencies, not the control",
            stderr=(proc.stderr or proc.stdout)[-500:],
        )
    return _subprocess_result(
        control, proc, f"{rel_path} passed", f"{rel_path} failed"
    )


def load_run_evals(root: Path):
    """Import `scripts/run-evals.py` as a module.

    Delegates to `_shared.load_script`, the one loader for hyphen-named
    scripts. Reusing run-evals rather than re-reading fixture paths keeps the
    harness and the eval gate on one definition of where fixtures live, which
    suite falls back to which base file, and how a threshold resolves.

    The root arguments still matter: the harness may run from a tenant
    directory, so `runtime.*` and `scripts/*` must be importable before
    run-evals executes.
    """
    for extra in (str(root), str(root / "scripts")):
        if extra not in sys.path:
            sys.path.insert(0, extra)
    from _shared import load_script

    return load_script("run-evals")


def guard_suite(
    control: ControlSpec,
    ctx: dict[str, Any],
    suite: str,
    scorer: str,
    minimum: int = 3,
) -> ControlResult:
    """Run a deterministic guard suite and gate on its miss CEILING.

    Two controls work this way — adversarial prompt injection (SEC-ADV-001) and
    RAG poisoning (SEC-RAG-001). Both score every case locally with
    `runtime.prompt_guard`, both count misses rather than averaging quality, and
    both fail above a threshold instead of below one. Written twice they would
    have been two near-identical files free to drift on the one number that
    matters; the differences are the suite name and the scorer, so those are
    arguments.

    Neither calls a judge, which is why they can gate on every commit while the
    judge-backed eval controls only verify their gate is wired.
    """
    revals = load_run_evals(Path(ctx["root"]))
    cases = revals._load_cases(suite)
    if len(cases) < minimum:
        return failed(control, f"need ≥{minimum} {suite} cases, found {len(cases)}")

    results = [getattr(revals, scorer)(c) for c in cases]
    miss = revals.miss_rate(results)
    limit = revals._resolve_fail_above(suite, None)
    if miss is None:
        # miss_rate returns None when it scored nothing — "zero misses out of
        # zero cases" is not a clean guard result. Guarded above by the
        # len(cases) < minimum check, so unreachable today; handled anyway
        # because the alternative is a TypeError inside a SECURITY control, and
        # because a control that measured nothing must never report pass.
        return failed(control, f"{suite} scored no cases — nothing was probed")
    if miss > limit:
        return ControlResult(
            control_id=control.id,
            status="fail",
            message=f"{suite} miss rate {miss:.3f} > {limit:.3f}",
            evidence={"miss_rate": f"{miss:.3f}", "limit": f"{limit:.3f}"},
        )
    return ControlResult(
        control_id=control.id,
        status="pass",
        message=f"{suite} miss rate {miss:.3f} ≤ {limit:.3f} ({len(cases)} cases)",
        evidence={"miss_rate": f"{miss:.3f}", "cases": str(len(cases))},
    )


def eval_suite_gateable(
    control: ControlSpec, ctx: dict[str, Any], suite: str, minimum: int = 3
) -> ControlResult:
    """A judge-backed eval gate exists, has enough cases to gate, and resolves
    a threshold.

    Deliberately does NOT run the judge. A security control that needs a
    provider credential and a quota would report Gap whenever an account is
    unfunded — turning a compliance check into an availability check, and
    exactly the confusion the eval gates themselves had to be fixed for. The
    verifiable claim is that the gate is wired and would bite; whether quality
    passes is what CI's eval steps report.
    """
    root = Path(ctx["root"])
    revals = load_run_evals(root)
    cases = revals._load_cases(suite)
    if not cases and not revals._evals_path(suite).exists():
        # No fixture file at all — this repo has no such dataset to gate.
        # Falling back to the framework's shipped base would be worse than
        # skipping: it would grade generic cases as if they were this repo's,
        # which is the exact defect pinning `actual_output` was introduced to
        # fix.
        return not_applicable(
            control, f"no {suite} dataset in this repo", suite=suite
        )
    if len(cases) < minimum:
        return ControlResult(
            control_id=control.id,
            status="fail",
            message=f"{suite}: {len(cases)} case(s), need ≥{minimum} to gate",
            evidence={"cases": str(len(cases))},
        )
    threshold = revals._resolve_fail_below(suite, None)
    return ControlResult(
        control_id=control.id,
        status="pass",
        message=f"{suite}: {len(cases)} cases, threshold {threshold:.2f}",
        evidence={"cases": str(len(cases)), "threshold": f"{threshold:.2f}"},
    )


# ── Context and result helpers ───────────────────────────────────────────────
#
# Every runner needs the framework root (usually to import `runtime.*`) and
# builds ControlResults by hand. Five runners each carried their own
# `sys.path.insert(0, str(root))`; ten repeated `Path(ctx["root"])`. Extracted
# so a runner reads as the check it performs rather than its preamble.


def framework_root(ctx: dict[str, Any]) -> Path:
    """The framework checkout, with `runtime.*` importable.

    Runners import guardrail modules to exercise them directly. The path insert
    was duplicated in every runner that does, and omitting it fails only on the
    machines where the framework is not already on sys.path — i.e. not the one
    it was written on.
    """
    root = Path(ctx["root"])
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def tenant_security(ctx: dict[str, Any]) -> Path:
    """The TENANT's `.agent-rfc/security/` directory.

    Distinct from the framework root on purpose: a control that grades the
    framework's own shipped templates while claiming to grade the tenant is the
    defect SEC-TOOL-001 had.
    """
    return Path(ctx["tenant_security"])


def passed(control: ControlSpec, message: str, **evidence: str) -> ControlResult:
    return ControlResult(
        control_id=control.id, status="pass", message=message, evidence=evidence
    )


def failed(control: ControlSpec, message: str, **evidence: str) -> ControlResult:
    return ControlResult(
        control_id=control.id, status="fail", message=message, evidence=evidence
    )


# Prefix that marks a skip as a deliberate judgement rather than a gap. The
# harness reports both as `skip`, and that ambiguity is what let 14 unverified
# controls read as green: "nothing checked this" and "this does not apply here"
# are opposite facts wearing one label.
NOT_APPLICABLE = "not applicable"

# A gap the registry DECLARES. Distinct from a control claiming `met` that
# nothing verifies: one is a tracked deficiency, the other is the map saying
# something untrue. Strict mode punishes the second, not the first — blocking
# on acknowledged gaps makes --strict unusable and creates an incentive to
# mislabel a gap as `met`, which is precisely the failure it exists to catch.
DECLARED_GAP = "gap (declared)"


def not_applicable(control: ControlSpec, message: str, **evidence: str) -> ControlResult:
    """The control is sound but has nothing to govern in THIS repo.

    The framework ships eval fixtures for tenants and holds no golden dataset
    of its own, so SEC-EVAL-001 has nothing to measure when the framework
    grades itself. That is categorically different from a control whose runner
    was never written, and only one of the two should survive a strict run.
    """
    return ControlResult(
        control_id=control.id,
        status="skip",
        message=f"{NOT_APPLICABLE} — {message}",
        evidence=evidence,
    )


def framework_component_absent(
    control: ControlSpec, ctx: dict[str, Any], component: str
) -> Optional[ControlResult]:
    """`not applicable` when a VENDORED install lacks a framework-only component.

    Some controls verify AgentSmith's own components — the Ops Portal
    (`portal/`), the git hooks (`hooks/`) — rather than anything a tenant
    owns. A tenant receives `scripts/`, `runtime/` and `fixtures/security/`
    vendored into its own tree, never those, so the install root there is the
    tenant's and the component is structurally absent. Reporting that as
    `fail: missing portal files` told AqlaarTeleologyStudio its compliance was
    broken when its repo had nothing to do with it: three of its nine strict
    failures were this.

    Returns None — run the check — whenever the component exists OR the
    install root is the framework's own checkout. The second condition is the
    point: in the framework, a missing portal file is exactly what these
    controls exist to catch, and must keep failing.
    """
    root = framework_root(ctx)
    if (root / component).exists():
        return None
    try:
        from runtime.cli import looks_like_framework
    except ImportError:  # fail-open: no runtime to ask, so run the real check
        return None
    if looks_like_framework(root):
        return None
    return not_applicable(
        control,
        f"verifies AgentSmith's own {component}/, which is not vendored into a "
        f"tenant — evidenced by the framework's self-test, not by this repo",
    )


def security_fixture(
    control: ControlSpec, ctx: dict[str, Any], name: str
) -> tuple[list | None, ControlResult | None]:
    """Load `fixtures/security/<name>` — returns (cases, None) or (None, failure).

    Two runners repeated the same eight lines: build the path, fail if absent,
    parse. The absence branch matters more than the parse — a control whose
    fixtures have gone missing must fail rather than iterate an empty list and
    report success on zero cases, which is the quiet way a probe suite stops
    proving anything.
    """
    path = framework_root(ctx) / "fixtures" / "security" / name
    if not path.exists():
        return None, failed(control, f"missing fixture: {path}")
    cases = json.loads(path.read_text(encoding="utf-8"))
    if not cases:
        return None, failed(control, f"fixture {name} is empty — nothing probed")
    return cases, None


# ── A contract run (contract/security/v1) ────────────────────────────────────
#
# `scripts/security_port.py` runs these runners for a repository that declares
# AgentSmith its security provider. Two things differ from the harness run, and
# both are read from the context so one runner serves both: the posture checked
# is the repository's DECLARED one — never the environment of whoever runs the
# check — and the provider's own code (its probe sets, its library) is checked
# only in the provider's own repository (.agent-rfc/designs/security-contract.md).


def is_contract(ctx: dict[str, Any]) -> bool:
    return bool(ctx.get("contract"))


def provider_code_in_scope(ctx: dict[str, Any]) -> bool:
    """Whether the provider's own code is this run's to check: always in the
    harness, and in a contract run only when the repository IS the provider."""
    return not is_contract(ctx) or bool(ctx.get("own"))


def declared_value(ctx: dict[str, Any], dotted: str) -> tuple[Any, str]:
    """(value, problem) at a dotted path of the repository's `tenant.yaml` —
    `(None, "")` when it declares nothing there."""
    import yaml

    path = Path(ctx["tenant_root"]) / ".agenticframework" / "tenant.yaml"
    try:
        node: Any = yaml.safe_load(path.read_text(encoding="utf-8")) if path.is_file() else {}
    except (OSError, yaml.YAMLError) as exc:
        return None, f".agenticframework/tenant.yaml does not parse ({exc})"
    for part in dotted.split("."):
        node = node.get(part) if isinstance(node, dict) else None
    return node, ""


def declared_choice(
    ctx: dict[str, Any], dotted: str, allowed: tuple[str, ...]
) -> tuple[Optional[str], str]:
    """(value, problem) for a word the repository declares in `tenant.yaml`.

    `(None, "")` when it declares nothing. A YAML boolean is a problem, not a
    word: a bare `off` parses as False, and reading that as "off" would turn a
    control off by writing something that was never a valid value — the rule
    `runtime.config.resolve_choice` keeps for the runtime."""
    node, problem = declared_value(ctx, dotted)
    if problem or node is None:
        return None, problem
    if isinstance(node, bool):
        return None, (f"{dotted} is the YAML boolean {str(node).lower()}, not one of {', '.join(allowed)} "
                      "— quote the word")
    value = str(node).strip().lower()
    if value not in allowed:
        return None, f"{dotted}: {node!r} is not one of {', '.join(allowed)}"
    return value, ""


def _evals_port():
    from _shared import load_script

    return load_script("evals_port")


def contract_dataset_gradable(control: ControlSpec, ctx: dict[str, Any], suite: str) -> ControlResult:
    """A judged suite's dataset is the repository's and gradable under
    contract/evals/v1 — enough cases, each with the output the application
    produced. No judge is called: whether quality passes is the eval step's
    answer, not this control's."""
    path = load_run_evals(Path(ctx["root"]))._evals_path(suite)
    if not path.is_file():
        return not_applicable(control, f"no {suite} dataset in this repository", suite=suite)
    port = _evals_port()
    cases, why, _count = port.dataset(path, suite)
    minimum = port.MIN_CASES.get(suite, 3)
    if cases is not None and len(cases) < minimum:
        cases, why = None, f"{suite}: {len(cases)} case(s), need at least {minimum} to gate"
    if cases is None:
        if _declared_not_gradable(ctx, suite):
            # The repository already declared, in its reviewed providers.json,
            # that this suite's `not_gradable` only warns: a tracked gap, shown.
            return ControlResult(control_id=control.id, status="warn",
                                 message=f"{DECLARED_GAP} in providers.json `evals.not_gradable` — {why}",
                                 evidence={"suite": suite})
        return failed(control, why)
    return passed(control, f"{suite}: {len(cases)} cases, each with its output — gradable",
                  cases=str(len(cases)))


def _declared_not_gradable(ctx: dict[str, Any], suite: str) -> bool:
    try:
        declared = json.loads((Path(ctx["tenant_root"]) / ".agenticframework" / "providers.json").read_text(
            encoding="utf-8"))
        return declared["providers"]["evals"]["not_gradable"].get(suite) == "warn"
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return False


def contract_guard_suite(control: ControlSpec, ctx: dict[str, Any], suite: str) -> ControlResult:
    """A guard suite over the repository's own probes, judged as the evals
    contract judges it — never the provider's base cases in its place."""
    import gate_models as gm

    if not load_run_evals(Path(ctx["root"]))._evals_path(suite).is_file():
        return not_applicable(control, f"no {suite} dataset in this repository", suite=suite)
    card = _evals_port().judge(Path(ctx["tenant_root"]), gm.EvalsRequest(suite=suite))
    evidence = {"cases": str(card.cases_total)}
    if card.fail_above is not None:
        evidence["limit"] = f"{card.fail_above:.3f}"
    if card.verdict == "pass":
        return passed(control, f"{suite}: {card.reason}", **evidence)
    return failed(control, f"{suite} {card.verdict.replace('_', ' ')}: {card.reason}", **evidence)


def repository_env(install: Path, base: Optional[dict[str, str]] = None) -> dict[str, str]:
    """The environment for the repository's interpreter, without the provider's
    own source on PYTHONPATH — the setup step puts it there for the provider, and
    a repository's code must import the library version IT pins, not the one
    answering the check."""
    env = dict(os.environ if base is None else base)
    kept = [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p and Path(p).resolve() != install.resolve()]
    if kept:
        env["PYTHONPATH"] = os.pathsep.join(kept)
    else:
        env.pop("PYTHONPATH", None)
    return env


def repository_python() -> str:
    """The repository's own interpreter — `python3` on PATH, the one its CI set
    up with its dependencies. A contract run never imports the repository's code
    into the provider's process."""
    import shutil

    return shutil.which("python3") or sys.executable


def node_suite(
    control: ControlSpec, ctx: dict[str, Any], rel_path: str, requires: tuple[str, ...] = ()
) -> ControlResult:
    """Delegate to a portal test written in TypeScript.

    Three controls now verify portal behaviour this way (SSO revocation, the
    audit log, the RBAC matrix) and the invocation is identical each time:
    `node --experimental-strip-types <file>` from the portal directory.

    `requires` names sibling source files that must exist. A test file present
    while the implementation it exercises has been deleted would pass by
    testing nothing, which is the failure mode a control is least able to
    notice.

    THE LOADER IS NOT OPTIONAL. `lib/` modules import each other with
    extensionless relative specifiers (`./constantTime`), which tsc resolves
    and bare `--experimental-strip-types` does not. `package.json`'s `test` and
    `test:db` scripts pass `--experimental-loader=./test/ts-extension-loader.mjs`
    for exactly this; this runner is a THIRD invocation path and was missed
    when the loader was added to the other two, so SEC-RBAC-001 went red on
    `ERR_MODULE_NOT_FOUND` the moment `lib/authz.ts` gained its first relative
    import. Same lesson as `return 2` for graceful skip: the fix landed at one
    call site and not its siblings.
    """
    absent = framework_component_absent(control, ctx, "portal")
    if absent is not None:
        return absent
    root = framework_root(ctx)
    portal = root / "portal"
    target = portal / rel_path
    loader = portal / "test" / "ts-extension-loader.mjs"
    missing = [p.name for p in (target, *(portal / r for r in requires)) if not p.exists()]
    if missing:
        return failed(control, f"missing portal files: {', '.join(missing)}")
    cmd = ["node", "--experimental-strip-types"]
    if loader.exists():
        # Relative so node resolves it against cwd=portal, matching package.json.
        cmd.append("--experimental-loader=./test/ts-extension-loader.mjs")
    cmd.append(str(target))
    proc = subprocess.run(
        cmd,
        cwd=portal,
        capture_output=True,
        text=True,
        check=False,
    )
    return _subprocess_result(
        control, proc, f"{rel_path} passed", f"{rel_path} failed"
    )
