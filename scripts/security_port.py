"""
scripts/security_port.py — AgentSmith as a security provider (contract/security/v1/protocol.md).

    security_port.py check       this repository's controls, one row each, as a result
    security_port.py redaction   whether planted probes reach this repository's telemetry wire

The repository is the working directory, or the request's `cwd`; one JSON request
on stdin, one result on stdout; exit 0 whenever it answers — the verdict is in
the result and the caller maps it — and exit 3 when this provider cannot run here.

The checks are the security harness's runners, not a second copy. What this adds
is the contract's terms around them: each row says whose evidence it is, and a
control about the provider's own code is not run for a tenant; the posture
checked is the one the repository declares, never the environment of whoever
runs the check; there is no non-strict mode; and a tenant's registry only adds
(.agent-rfc/designs/security-contract.md).
"""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gate_models as gm

EXIT_CANNOT_RUN = 3
# The provider's own installation: its registry, probe sets and library.
INSTALL = Path(__file__).resolve().parent.parent
REGISTRY = INSTALL / "fixtures" / "security" / "control_registry.json"
TENANT_REGISTRY = Path(".agent-rfc") / "security" / "control_registry.json"

# Posture an environment would otherwise set for a check. Cleared for a contract
# run: what a CI step exports is not what the repository declares.
POSTURE_ENV = ("PROMPT_GUARD", "INPUT_GUARDRAIL", "MODERATION_HOOK", "MODERATION_HOOK_PATH",
               "TOOL_ALLOWLIST_STRICT", "TOOL_ALLOWLIST_PATH", "PROMPT_DENYLIST_PATH", "SECURITY_STRICT")


def _version() -> str:
    from runtime.version import framework_version

    return framework_version()


def _own(root: Path) -> bool:
    """Whether the repository is this provider's own checkout — the one place
    its own code is the repository's to check."""
    from runtime.cli import looks_like_framework

    return bool(looks_like_framework(root))


def registry(root: Path) -> tuple[list, set[str], str]:
    """(controls, ids the repository tried to redefine, why it cannot be read).

    The provider's controls, then the repository's own — validated as the
    contract's `registry.schema.json`, and only ADDED: a row with an id the
    provider already checks is refused, and that control fails, because a
    registry the repository under review can edit must not lower its floor."""
    from security.registry import ControlSpec, FrameworkTags, load_control_registry

    controls = load_control_registry(REGISTRY)
    path = root / TENANT_REGISTRY
    if not path.is_file():
        return controls, set(), ""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return [], set(), f"{TENANT_REGISTRY} is not JSON ({exc})"
    if not isinstance(raw, list):
        return [], set(), f"{TENANT_REGISTRY} is not a list of controls"
    rows, problems = [], []
    for index, row in enumerate(raw):
        try:
            rows.append(gm.TenantControl.model_validate(row))
        except gm.ValidationError as exc:
            name = row.get("id") if isinstance(row, dict) and row.get("id") else f"#{index + 1}"
            problems.append(f"{name} ({'; '.join(e['msg'] for e in exc.errors()[:2])})")
    if problems:
        return [], set(), f"{len(problems)} control(s) in {TENANT_REGISTRY} do not match the contract: " \
                          + "; ".join(problems[:5])
    ids = [row.id for row in rows]
    if len(set(ids)) != len(ids):
        return [], set(), f"{TENANT_REGISTRY} names a control twice"
    theirs = {c.id for c in controls}
    clashes = {row.id for row in rows} & theirs
    for row in rows:
        if row.id in clashes:
            continue
        fw = row.frameworks
        controls.append(ControlSpec(
            id=row.id, title=row.title, status=row.status, owner=row.owner,
            frameworks=FrameworkTags(owasp=list(fw.owasp), nist=list(fw.nist), atlas=list(fw.atlas),
                                     iso42001=list(fw.iso42001)),
            runner="tenant_suite", check_type="unit", mechanism=row.mechanism, suite=row.suite,
            subject="repository"))
    return controls, clashes, ""


def _row(control: Any, result: str, message: str, evidence: Optional[dict] = None) -> gm.ControlRow:
    fw = control.frameworks
    return gm.ControlRow(id=control.id, title=control.title, subject=control.subject, result=result,
                         message=message, evidence={str(k): str(v) for k, v in (evidence or {}).items()},
                         frameworks=gm.ControlFrameworks(owasp=fw.owasp, nist=fw.nist, atlas=fw.atlas,
                                                         iso42001=fw.iso42001))


def _checked(control: Any, ctx: dict) -> gm.ControlRow:
    """One control, run under the contract's terms."""
    from security.runners import RUNNERS
    from security.runners._shared import DECLARED_GAP

    if control.subject == "provider" and not ctx["own"]:
        return _row(control, "not_applicable", f"the provider's own code — AgentSmith {ctx['version']}'s own CI is "
                                               "its evidence, not this repository")
    if control.status == "gap":
        return _row(control, "gap", f"{DECLARED_GAP} — nothing checks it here")
    if control.status == "org-owned" and control.runner == "tenant_suite" and not control.suite:
        return _row(control, "not_applicable", "org-owned — evidenced outside this repository, by no suite here")
    runner = RUNNERS.get(control.runner)
    if runner is None:
        return _row(control, "fail", f"declared {control.status!r} and nothing verifies it — runner "
                                     f"{control.runner!r} is not implemented")
    try:
        found = runner(control, ctx)
    except Exception as exc:  # one control raising must not take the others' answers with it
        return _row(control, "fail", f"the check raised {type(exc).__name__}: {str(exc)[:300]}")
    if found.status == "skip":
        return _row(control, "not_applicable", found.message, found.evidence)
    if found.status == "warn":
        # No non-strict mode: a warning is a failure unless it is a gap the registry declares.
        result = "gap" if found.message.startswith(DECLARED_GAP) else "fail"
        return _row(control, result, found.message, found.evidence)
    return _row(control, found.status, found.message, found.evidence)


def check(root: Path, request: gm.SecurityRequest) -> gm.SecurityResult:
    """Every control, or the ones asked about. Runners print to stderr."""
    provider = gm.ProviderName(name="agentsmith", version=_version())
    controls, clashes, unreadable = registry(root)
    if unreadable:
        return gm.SecurityResult(verdict="not_gradable", reason=unreadable, provider=provider)
    if request.controls is not None:
        unknown = sorted(set(request.controls) - {c.id for c in controls})
        if unknown:
            return gm.SecurityResult(verdict="not_gradable", provider=provider,
                                     reason=f"no control {', '.join(unknown)} is checked here")
        controls = [c for c in controls if c.id in set(request.controls)]
    ctx = {"root": INSTALL, "tenant_root": root, "tenant_security": root / ".agent-rfc" / "security",
           "mode": "ci", "strict": True, "use_template_fallback": False, "contract": True,
           "own": _own(root), "version": provider.version}
    rows = []
    for control in controls:
        if control.id in clashes:
            rows.append(_row(control, "fail", f"{TENANT_REGISTRY} redefines this control — a repository's registry "
                                              "only adds controls"))
        else:
            rows.append(_checked(control, ctx))
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.result] = counts.get(row.result, 0) + 1
    failed = [row.id for row in rows if row.result == "fail"]
    reason = "" if not failed else f"{len(failed)} control(s) failed: {', '.join(failed[:8])}" + \
        (f" and {len(failed) - 8} more" if len(failed) > 8 else "")
    return gm.SecurityResult(verdict="fail" if failed else "pass", reason=reason, provider=provider,
                             controls=rows, counts=counts)


def evidence_pack(root: Path, directory: str, result: gm.SecurityResult) -> None:
    """The human-readable pack, beside the result it was made from."""
    from security.registry import ControlSpec, FrameworkTags
    from security.report import ControlResult, write_evidence_pack

    out = Path(directory) if Path(directory).is_absolute() else root / directory
    controls = [ControlSpec(id=r.id, title=r.title, status="met", owner="shared",
                            frameworks=FrameworkTags(owasp=r.frameworks.owasp, nist=r.frameworks.nist,
                                                     atlas=r.frameworks.atlas, iso42001=r.frameworks.iso42001),
                            runner="", check_type="static", mechanism="", subject=r.subject)
                for r in result.controls]
    results = [ControlResult(r.id, r.result, r.message, r.evidence) for r in result.controls]  # type: ignore[arg-type]
    write_evidence_pack(out, controls, results, None, "contract")
    (out / "security_result.json").write_text(result.model_dump_json(by_alias=True, indent=2) + "\n",
                                              encoding="utf-8")


def redaction(root: Path, request: gm.RedactionRequest) -> gm.RedactionResult:
    from security.redaction import check as wire, declared_emitter

    emitter = request.emitter or declared_emitter(root)
    answer = wire(root, request.environment, emitter)
    return gm.RedactionResult(verdict=answer.verdict, reason=answer.reason, environment=request.environment,
                              leaked=list(answer.leaked))


# ── The command ──────────────────────────────────────────────────────────────


def _repository(cwd: Optional[str]) -> Path:
    if cwd:
        return Path(cwd).resolve()
    top = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=False)
    return Path(top.stdout.strip()) if top.returncode == 0 and top.stdout.strip() else Path.cwd()


def _answer(verb: str, root: Path, request: Any) -> Any:
    os.chdir(root)  # the runners resolve the repository's datasets from the working directory
    for name in POSTURE_ENV:
        os.environ.pop(name, None)
    for extra in (str(INSTALL), str(INSTALL / "scripts")):
        if extra not in sys.path:
            sys.path.insert(0, extra)
    with contextlib.redirect_stdout(sys.stderr):  # stdout carries the answer and nothing else
        if verb == "redaction":
            return redaction(root, request)
        result = check(root, request)
        if request.evidence_dir:
            evidence_pack(root, request.evidence_dir, result)
        return result


def main(argv: Optional[list[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args not in (["check"], ["redaction"]):
        print("usage: security_port.py check|redaction   (contract/security/v1/protocol.md)", file=sys.stderr)
        return 2
    verb = args[0]
    model = gm.RedactionRequest if verb == "redaction" else gm.SecurityRequest
    raw = "" if sys.stdin.isatty() else sys.stdin.read()
    try:
        request = model.model_validate_json(raw or "{}")
    except gm.ValidationError as exc:
        error = exc.errors()[0]
        where = ".".join(str(p) for p in error["loc"])
        print(f"security provider: the request is not valid ({where}: {error['msg']})", file=sys.stderr)
        return 2
    root = _repository(request.cwd)
    if not root.is_dir():
        print(f"security provider cannot run here: {root} is not a directory", file=sys.stderr)
        return EXIT_CANNOT_RUN
    try:
        import gate_tracing as gt
    except Exception:  # fail-open: tracing never changes an answer
        gt = None
    if gt is None:
        answer = _answer(verb, root, request)
    else:
        with gt.gate_span(f"security_{verb}", root=root, ide="neutral") as span:
            answer = _answer(verb, root, request)
            if span is not None:
                span.set_attribute("agent.decision", answer.verdict)
        gt.flush()
    print(answer.model_dump_json(by_alias=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
