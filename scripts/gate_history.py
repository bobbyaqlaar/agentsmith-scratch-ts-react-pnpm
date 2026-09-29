"""
scripts/gate_history.py — pillar 5's log, written by the hooks
(.agent-rfc/designs/governance-enforcement.md, G7).

    record(root, event, detail)     a MAJOR line in .agent-history.log
    record_gates_run(root, …)       what the last local `agentsmith gates` run found

Pillar 5 says to read `.agent-history.log` at session start so past mistakes are
not repeated. Nothing wrote to it: an agent was asked to record its own history
from memory, which is the one source that cannot be trusted to. The stop gate
and the sweep write it now — a blocked turn and a commit that never met a gate
are exactly the facts the next session needs.

The line is the shape `agentsmith check` already reads (`MAJOR` /
`hitl_resolved: false`), so there is one reader, not two.
"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

LOG = ".agent-history.log"
GATES_RUN = "agentsmith/gates-run.json"  # under .git/: local truth about this clone


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def record(root: Path, event: str, detail: str) -> None:
    """Append one unresolved MAJOR entry, unless it repeats the last one.

    The stop hook runs at every turn end, so without that guard one blocked
    change writes a line per turn and the log becomes something nobody reads —
    which is the same as not writing it.
    """
    log = Path(root) / LOG
    entry = {
        "timestamp": _now(),
        "level": "MAJOR",
        "event": event,
        "detail": detail,
        "agent": "process-gate",
        "agent_role": "gate",
        "project": Path(root).name,
        "owner_id": os.environ.get("AGENT_OWNER_ID") or "unknown",
        # Unresolved until a person says otherwise: that is what makes it
        # surface at the next session start instead of scrolling away.
        "hitl_resolved": False,
    }
    try:
        previous = log.read_text(encoding="utf-8").splitlines()[-1] if log.is_file() else ""
    except (OSError, IndexError):
        previous = ""
    if previous:
        try:
            last = json.loads(previous)
            if last.get("event") == event and last.get("detail") == detail \
                    and not last.get("hitl_resolved", True):
                return
        except json.JSONDecodeError:  # fail-open: a corrupt last line must not
            # stop the deduplication check; fall through and append.
            pass
    try:
        with log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")
    except OSError:  # fail-open: a log that cannot be written must not break the
        # hook that writes it.
        pass


def _git_dir(root: Path) -> Optional[Path]:
    found = subprocess.run(["git", "rev-parse", "--absolute-git-dir"], cwd=root,
                           capture_output=True, text=True, check=False).stdout.strip()
    return Path(found) if found else None


def head(root: Path) -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                          capture_output=True, text=True, check=False).stdout.strip()


def record_gates_run(root: Path, passed: int, failed: int, skipped: int,
                     commit: Optional[str] = None, only: Optional[str] = None) -> None:
    """What the last local gates run found, and which commit it ran against.

    In `.git/`, not the work tree: it is a fact about this clone at this
    moment, and committing it would make one machine's run look like everyone's.
    """
    git_dir = _git_dir(root)
    if git_dir is None:
        return
    target = git_dir / GATES_RUN
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({
        "at": _now(),
        "commit": commit if commit is not None else head(root),
        "passed": passed, "failed": failed, "skipped": skipped,
        # A filtered run is recorded AS filtered: one gate passing is not the
        # list passing, and `--governed` must not read it as if it were.
        "only": only,
    }) + "\n", encoding="utf-8")


def last_gates_run(root: Path) -> Optional[dict]:
    git_dir = _git_dir(root)
    if git_dir is None:
        return None
    target = git_dir / GATES_RUN
    if not target.is_file():
        return None
    try:
        return json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
