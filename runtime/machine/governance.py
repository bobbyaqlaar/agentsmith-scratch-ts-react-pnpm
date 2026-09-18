"""
runtime/machine/governance.py — `agentsmith approve` and `agentsmith design new`
(.agent-rfc/designs/governance-enforcement.md, G1).

A design may deviate from a rule only with the owner's approval, and the
approval has to come from a PERSON. Every coding agent — Claude Code, Cursor,
Antigravity, Copilot, Gemini, Codex — runs shell commands without a controlling
terminal, so this reads the confirmation from /dev/tty and never from stdin:
an agent cannot answer it, and cannot fake the record either, because the gate
refuses any record whose channel is not "tty" and because writing
approvals.jsonl by hand is a denied edit.

What this cannot prevent is a person typing an approval an agent asked for.
That is the point — it makes the deviation visible, attributed and dated.
"""

from __future__ import annotations

import io
import os
import secrets
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

from runtime.machine.state import find_script, framework_home

DESIGNS_DIR = ".agent-rfc/designs"


def _candidate_dirs(name: str) -> list[Path]:
    """Where a vendored framework file can be, most specific first: the repo
    being worked in, an explicit checkout, the machine install, and finally this
    module's own checkout — the one that matters when the install is stale."""
    roots = [
        Path.cwd(),
        Path(os.environ.get("AGENTSMITH_DIR", "")) if os.environ.get("AGENTSMITH_DIR") else None,
        framework_home(),
        Path(__file__).resolve().parents[2],
    ]
    return [root / name for root in roots if root is not None]


def _gate_module(name: str):
    """One of the gate's own modules, so the CLI and the gate share a
    definition rather than each carrying one (`one-catalog`)."""
    import importlib

    for directory in _candidate_dirs("scripts"):
        if (directory / f"{name}.py").is_file():
            if str(directory) not in sys.path:
                sys.path.insert(0, str(directory))
            return importlib.import_module(name)
    raise FileNotFoundError(
        f"scripts/{name}.py not found in this repo, $AGENTSMITH_DIR, ~/.agent-framework or the framework "
        "checkout — re-run AgentSmith's install-ai-stack.sh, or set AGENTSMITH_DIR"
    )


def _gate_models():
    return _gate_module("gate_models")


def _registry(gm):
    for candidate in _candidate_dirs("templates"):
        if (candidate / "governance.json").is_file():
            return gm.Registry.model_validate_json((candidate / "governance.json").read_text(encoding="utf-8"))
    raise FileNotFoundError(
        "templates/governance.json not found — run scripts/generate-ide-config.py --registry, "
        "or re-run install-ai-stack.sh"
    )


def _git_identity(root: Path) -> str:
    def config(key: str) -> str:
        result = subprocess.run(["git", "-C", str(root), "config", "--get", key],
                                capture_output=True, text=True, check=False)
        return result.stdout.strip()

    name, email = config("user.name"), config("user.email")
    if name and email:
        return f"{name} <{email}>"
    return name or email or os.environ.get("USER", "unknown")


def _deviation_text(design_text: str, deviation_id: str) -> Optional[str]:
    import re

    section = re.search(r"^## Deviations\b.*?$(.*?)(?=^## |\Z)", design_text, re.M | re.S)
    if not section:
        return None
    for line in section.group(1).splitlines():
        if re.match(r"^-\s*~~", line):
            continue
        if re.match(rf"^-\s*{re.escape(deviation_id)}\b", line):
            return line.strip("- ").strip()
    return None


def approve(design: str, deviation: str, statement: Optional[str], root: Optional[Path] = None) -> int:
    """Record the owner's approval of one deviation. Terminal only."""
    gm = _gate_models()
    root = Path(root or Path.cwd())
    design_path = root / design
    if not design_path.is_file():
        print(f"❌ {design} does not exist", file=sys.stderr)
        return 2
    text = design_path.read_text(encoding="utf-8")
    entry = _deviation_text(text, deviation)
    if entry is None:
        print(f"❌ {design} does not list an active deviation {deviation} in '## Deviations' — "
              "add the entry first, so the approval says what was approved", file=sys.stderr)
        return 2

    # The one thing an agent's shell cannot provide. Two handles, not one
    # "r+": /dev/tty is not seekable, and Python's buffered read-write mode
    # demands that — an "r+" here refused every real terminal too.
    try:
        reader = open("/dev/tty", "r")
        writer = open("/dev/tty", "w")
    except (OSError, ValueError, io.UnsupportedOperation):
        print("❌ an approval is given at a terminal, and this process has none. "
              "Run `agentsmith approve` yourself in a shell — an agent must ask you, not record it for you.",
              file=sys.stderr)
        return 2

    with reader, writer:
        writer.write(
            f"\nDeviation {deviation} of {design}:\n  {entry}\n\n"
            f"Approving lets code that deviates from this rule be committed.\n"
            f"Type {deviation} to approve, anything else to refuse: "
        )
        writer.flush()
        typed = reader.readline().strip()
        if typed != deviation:
            writer.write("not approved — nothing was recorded.\n")
            return 1
        if not statement:
            writer.write("One line on why (recorded with the approval): ")
            writer.flush()
            statement = reader.readline().strip()
        record = gm.Approval(
            id=f"A-{secrets.token_hex(4)}",
            design=design,
            deviation=deviation,
            approver=_git_identity(root),
            approved_at=datetime.now(timezone.utc),
            channel="tty",
            statement=statement or "approved",
        )
        target = root / gm.APPROVALS_FILE
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as handle:
            handle.write(record.model_dump_json() + "\n")
        writer.write(
            f"\n✅ recorded {record.id} in {gm.APPROVALS_FILE}\n"
            f"   Put it in the design: `— approval: {record.id}`\n"
        )
    return 0


def design_new(slug: str, scope: Sequence[str], root: Optional[Path] = None) -> int:
    """Write the design skeleton the gate asks for — every pillar, already asked."""
    gm = _gate_models()
    registry = _registry(gm)
    root = Path(root or Path.cwd())
    target = root / DESIGNS_DIR / f"{slug}.md"
    if target.exists():
        print(f"❌ {target.relative_to(root)} already exists — edit it, or choose another slug", file=sys.stderr)
        return 2
    scope = list(scope) or ["<path or glob this change may touch>"]
    body = [
        "---",
        "status: active",
        "scope:",
        *[f"  - {path}" for path in scope],
        "---",
        f"# {slug.replace('-', ' ').capitalize()}",
        "",
        "## Problem",
        "",
        "What is wrong or missing, and the evidence.",
        "",
        "## Approach",
        "",
        "What you will build, and the decisions that matter.",
        "",
        "## Pillars",
        "",
        "One line per rule: `applies — how`, `n/a — why`, `gap — <backlog id>`, "
        "or `**deviation D<n>**` once the owner has approved it.",
        "",
    ]
    for pillar in registry.pillars:
        if "design" in pillar.check:
            # "TODO" is not one of the four verdicts, so the gate rejects the
            # skeleton until a person answers it. A pre-filled "applies" would
            # have passed unread.
            body.append(f"- P{pillar.id} TODO — {pillar.design_question}")
    body += [
        "",
        "## Deviations",
        "",
        "none",
        "",
        "<!-- Any rule you cannot follow: ASK THE OWNER FIRST, then",
        "     `- D1 — P<id> or lever — what and why — approval: A-xxxxxxxx`,",
        "     where the id comes from the owner running `agentsmith approve`. -->",
        "",
        "## Dependencies",
        "",
        "none",
        "",
        "<!-- Packages added, direct and transitive (diff the lock file). -->",
        "",
        "## Levers",
        "",
        "- `lever-slug` — why it applies and what the design does about it.",
        "",
    ]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(body), encoding="utf-8")
    print(f"✅ Written {target.relative_to(root)} — answer every pillar before you write code")
    return 0


def gates_repair(root: Optional[Path] = None) -> int:
    """`agentsmith gates repair` — what the sweep found, and how to clear it.

    The same sweep the hooks run, so the list a person reads is the list that is
    refusing their commit; a second implementation would eventually disagree
    with the one doing the refusing.
    """
    root = Path(root or Path.cwd())
    script = find_script("process_gate.py", root)
    if script is None:
        print("❌ scripts/process_gate.py not found in this repo or ~/.agent-framework — "
              "re-run install-ai-stack.sh, or set AGENTSMITH_DIR", file=sys.stderr)
        return 2
    return subprocess.run([sys.executable, str(script), "sweep"], cwd=root, check=False).returncode


def gates_list() -> int:
    """The gates this repo's CI declares, as the table the checklist carries."""
    steps = _gate_module("gate_steps")
    found = steps.gates(Path.cwd())
    if not found:
        print(f"gates: no step in .github/workflows carries `{steps.TAG}` — nothing to run. "
              "Tag the steps that are gates, then re-run.")
        return 0
    print(steps.render_table(found, note=False).strip())
    return 0


def gates_run(only: str | None = None, services: bool = False, fail_fast: bool = False,
              allow_install: bool = False) -> int:
    """Run them here. The exit code is CI's answer to the same list, as far as
    a machine without CI's services can give one."""
    steps = _gate_module("gate_steps")
    return steps.run(Path.cwd(), only=only, services=services, fail_fast=fail_fast,
                     allow_install=allow_install)
