"""run-security-checks.py — unified security harness for AgentSmith tenant apps."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Literal

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS = _ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# These must follow the sys.path insertion above — the `security` package lives
# under scripts/, which is not importable until that runs.
from security.registry import load_control_registry  # noqa: E402
from security.report import ControlResult, write_evidence_pack  # noqa: E402
from security.runners._shared import DECLARED_GAP  # noqa: E402
from security.runners import RUNNERS  # noqa: E402

Mode = Literal["smoke", "ci", "full"]


def _install_root() -> Path:
    """Root of the checkout this script lives in (file-relative).

    Renamed from `_repo_root` (ReviewFindings-2026-07-18 B4): it shared a
    name with `_shared._repo_root()` but NOT its semantics — that one walks
    up from the *current working directory* to the nearest `.git`, while
    the harness must find its fixtures relative to where it is installed,
    regardless of cwd. Same-name-different-behavior is drift bait; the
    rename records that this is intentional, not a leftover copy."""
    return Path(__file__).resolve().parent.parent


def _tenant_root() -> Path:
    """Root of the repo BEING GRADED — cwd-relative, unlike `_install_root`.

    These two are the same directory only when the framework grades itself.
    A tenant runs the harness out of a framework checkout:

        cd my-tenant && python3 $AGENTSMITH_DIR/scripts/run-security-checks.py

    and the pack under review is the TENANT's `.agent-rfc/security/`, while the
    control registry and templates still come from the install. Resolving both
    from `_install_root()` meant every tenant's `--strict` gate silently graded
    the FRAMEWORK's pack: the authored risk register, agency manifest and tool
    allowlist that `ai-tenant-init` seeds (G5) were never read by anything, and
    a tenant's green SEC-RISK-001 was evidence about a different repo. Same
    walk-up-to-.git semantics as `_shared._repo_root()`.

    `AGENTSMITH_TENANT_ROOT` overrides, for callers that cannot set cwd.
    """
    override = os.environ.get("AGENTSMITH_TENANT_ROOT", "").strip()
    if override:
        return Path(override).resolve()
    cwd = Path.cwd().resolve()
    for parent in [cwd, *cwd.parents]:
        if (parent / ".git").exists():
            return parent
    return cwd


def _tenant_security_dir(root: Path) -> Path:
    return root / ".agent-rfc" / "security"


def _resolve_exit(results: list[ControlResult], strict: bool) -> int:
    """pass=green; fail=red; warn=green unless strict, with two exemptions.

    Both exemptions exist because `warn` and `skip` each carried two opposite
    meanings, and the harness could not tell them apart:

      * a DECLARED gap is a tracked deficiency — it must be visible, but a repo
        honest about its gaps has to be able to pass a strict run;
      * `not applicable` means the control has nothing to govern HERE (the
        framework holds no golden dataset of its own), not that it went
        unchecked.

    An undeclared gap — `met`/`partial` with no runner — is neither, and now
    fails strict. That combination is the map claiming something untrue.
    """
    for r in results:
        if r.status == "fail":
            return 1
        if strict and r.status == "warn" and not r.message.startswith(DECLARED_GAP):
            return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["smoke", "ci", "full"], default="full")
    p.add_argument("--strict", action="store_true")
    p.add_argument("--framework", choices=["owasp", "nist", "atlas", "iso42001"])
    p.add_argument("--evidence-pack", type=Path)
    args = p.parse_args(argv)

    root = _install_root()
    strict = args.strict or os.environ.get("SECURITY_STRICT", "") == "1"
    registry_path = root / "fixtures" / "security" / "control_registry.json"
    controls = load_control_registry(
        registry_path,
        _tenant_security_dir(_tenant_root()) / "control_registry.json",
    )

    if args.mode == "smoke":
        allow = {"SEC-PII-001", "SEC-PII-002", "SEC-AUDIT-001"}
        controls = [c for c in controls if c.id in allow]

    # `root` (install) owns fixtures, templates and the runtime import path;
    # `tenant_root` (cwd) owns the .agent-rfc/security pack under review.
    tenant_root = _tenant_root()
    tenant_security = _tenant_security_dir(tenant_root)
    template_risk = root / "fixtures" / "security" / "templates" / "risk_register.yaml"
    results: list[ControlResult] = []
    ctx = {
        "root": root,
        "tenant_root": tenant_root,
        "tenant_security": tenant_security,
        "mode": args.mode,
        "strict": strict,
        # Framework self-test: validate shipped template when tenant file absent.
        "use_template_fallback": (
            not (tenant_security / "risk_register.yaml").exists()
            and template_risk.exists()
            and args.mode in ("full", "ci")
            and not strict
        ),
    }
    for control in controls:
        if control.status == "gap":
            # Declared and tracked. Visible in every report, but it does not
            # block: a repo honest about its gaps must still be able to pass a
            # strict run, or the only way to go green is to relabel them `met`.
            #
            # The message says "no runner here", not "not yet implemented",
            # because the second is a claim this harness is not in a position
            # to make. It knows only that nothing in THIS run verified the
            # control. SEC-AUDIT-002 — currently the registry's only gap — is
            # implemented (append-only triggers in portal/db/schema.sql) and
            # verified on every push by self-test.yml's Ops Portal job against
            # a live Postgres. It is a gap *here* because this harness runs
            # offline and cannot reach a database. Reporting that as "not yet
            # implemented" understated a shipped control, which is the same
            # class of error as overstating one — read the control's own
            # `mechanism` for what is actually true of it.
            results.append(
                ControlResult(
                    control.id, "warn", f"{DECLARED_GAP} — no runner in this harness", {}
                )
            )
            continue
        runner = RUNNERS.get(control.runner)
        if runner is None:
            # UNDECLARED gap: the registry claims `met`/`partial` and nothing
            # checks it. This used to be `skip`, which passed even --strict —
            # how 14 of 23 controls reported green while never being examined.
            results.append(
                ControlResult(
                    control.id,
                    "fail" if strict else "warn",
                    f"declared {control.status!r} but runner {control.runner!r} is not "
                    f"implemented — nothing verifies this control",
                    {},
                )
            )
            continue
        results.append(runner(control, ctx))

    if args.evidence_pack:
        write_evidence_pack(
            args.evidence_pack, controls, results, args.framework, args.mode
        )

    # Report to stdout, always. This exited silently with a bare status code:
    # CI showed `Process completed with exit code 1` and nothing else, so every
    # diagnosis needed an evidence pack and a local re-run. A harness whose
    # verdict is invisible makes its own failures expensive to act on.
    by_status: dict[str, int] = {}
    for r in results:
        by_status[r.status] = by_status.get(r.status, 0) + 1
    summary = "  ".join(f"{k}={v}" for k, v in sorted(by_status.items()))
    print(f"security harness [{args.mode}{', strict' if strict else ''}]: {summary}")
    for r in results:
        if r.status in ("fail", "warn"):
            print(f"  {r.status.upper():4} {r.control_id:18} {r.message}")
            for key, value in (r.evidence or {}).items():
                print(f"       {key}: {str(value)[:200]}")

    return _resolve_exit(results, strict)


if __name__ == "__main__":
    raise SystemExit(main())
