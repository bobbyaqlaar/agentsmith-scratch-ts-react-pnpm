"""
promote-learning.py — HITL trace promoter: production incident → golden dataset.

Two paths to promotion:
  1. CLI:          python3 scripts/promote-learning.py <case-id> '<input>' '<correct-output>'
  2. Shell alias:  ai-stack-promote <case-id> '<input>' '<correct-output>'

After promotion:
  - Appends the new case to golden_evals.json.
  - Appends the resolution as a learning to custom_judge_criteria.json.
  - Marks the corresponding .agent-history.log entry as hitl_resolved.
  - Writes hitl_resolved_by and hitl_resolved_at.
  - Optionally re-runs the eval scorecard to validate the fix.

This is the production flywheel: every resolved incident compounding
into a calibrated evaluation that catches the same class of failure next time.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional


# ── Paths ─────────────────────────────────────────────────────────────────────

from _shared import (
    _repo_root,
    _iso_now,
    fixtures_path,
    EVALS_FILE,
    GOLDEN_CRITERIA_FILE,
)


# Resolved through the shared tables rather than literal filenames: these two
# used to hardcode "golden_evals.json" and "custom_judge_criteria.json", so a
# rename in run-evals' table would have left this script writing to the old
# name — silently, since it creates what it cannot find.
def _golden_path() -> Path:
    return fixtures_path(EVALS_FILE["golden"])


def _criteria_path() -> Path:
    return fixtures_path(GOLDEN_CRITERIA_FILE)


def _log_path() -> Path:
    return _repo_root() / ".agent-history.log"


# ── Core promoter ─────────────────────────────────────────────────────────────


def promote(
    case_id: str,
    input_query: str,
    correct_output: str,
    expected_tool: str = "any",
    resolution_note: Optional[str] = None,
    resolved_by: Optional[str] = None,
    rerun_evals: bool = True,
) -> dict:
    """
    Promote a production trace to the golden dataset.

    Args:
        case_id:         Unique identifier for the case (may match a log event).
        input_query:     The input that triggered the failure.
        correct_output:  The correct / expected output or behaviour.
        expected_tool:   The tool or pattern expected (optional).
        resolution_note: Human-readable explanation of what the correct fix is.
        resolved_by:     Email / ID of the person who resolved it.
        rerun_evals:     If True, re-run the eval scorecard after promotion.

    Returns:
        Dict with 'case_id', 'golden_count', 'hitl_entries_resolved'.
    """
    resolver = resolved_by or os.environ.get("AGENT_OWNER_ID", "unknown")
    ts = _iso_now()

    # ── 1. Append to golden dataset ───────────────────────────────────────────
    golden = _load_json(_golden_path(), default=[])
    existing_ids = {c.get("id") for c in golden}

    if case_id in existing_ids:
        print(f"⚠️  Case {case_id!r} already in golden dataset. Updating...")
        golden = [c for c in golden if c.get("id") != case_id]

    new_case = {
        "id": case_id,
        "input": input_query,
        "expected_tool": expected_tool,
        "reference_output": correct_output,
        "promoted_at": ts,
        "promoted_by": resolver,
        "resolution_note": resolution_note or "",
    }
    golden.append(new_case)
    _save_json(_golden_path(), golden)
    print(f"✅ Added to golden dataset: {case_id!r}  (total: {len(golden)} cases)")

    # ── 2. Append resolution as judge learning ────────────────────────────────
    if resolution_note:
        criteria = _load_json(
            _criteria_path(),
            default={"name": "Default", "instructions": "", "historical_learnings": []},
        )
        learnings: list = criteria.setdefault("historical_learnings", [])
        learning_entry = f"[{ts}] [{case_id}] {resolution_note}"
        if learning_entry not in learnings:
            learnings.append(learning_entry)
        _save_json(_criteria_path(), criteria)
        print(f"✅ Judge learning appended: {resolution_note[:80]}...")

    # ── 3. Mark .agent-history.log entries as resolved ────────────────────────
    resolved_count = _mark_log_resolved(case_id, resolver, ts)
    if resolved_count:
        print(
            f"✅ Marked {resolved_count} log entry/entries hitl_resolved for event={case_id!r}"
        )
    else:
        print(f"   (No matching unresolved log entries found for event={case_id!r})")

    # ── 4. Re-run evals ───────────────────────────────────────────────────────
    if rerun_evals and len(golden) >= 3:
        print("\n🔄 Re-running eval scorecard to validate fix...")
        try:
            # run-evals.py has a dash in its filename — `from run_evals
            # import run_scorecard` can never resolve (same class of bug as
            # sync-ui-feedback.py's promote_learning import, found by
            # test_promotion_loop.py). load_script loads it by path and
            # caches, so the manual sys.modules bookkeeping this used to do
            # is no longer needed.
            from _shared import load_script

            run_scorecard = load_script("run-evals").run_scorecard

            exit_code = run_scorecard()
            if exit_code == 0:
                print("✅ Evals pass after promotion.")
            else:
                print("⚠️  Evals still failing — additional fixes may be required.")
        except Exception as exc:
            print(f"⚠️  Could not re-run evals: {exc}")

    return {
        "case_id": case_id,
        "golden_count": len(golden),
        "hitl_entries_resolved": resolved_count,
    }


def _mark_log_resolved(event_filter: str, resolver: str, ts: str) -> int:
    """Mark all unresolved MAJOR/CRITICAL entries whose event matches event_filter."""
    log_file = _log_path()
    if not log_file.exists():
        return 0

    updated = 0
    lines: list[str] = []
    with log_file.open("r", encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            try:
                entry = json.loads(raw)
                if (
                    entry.get("event") == event_filter
                    and entry.get("level") in ("MAJOR", "CRITICAL")
                    and not entry.get("hitl_resolved", True)
                ):
                    entry["hitl_resolved"] = True
                    entry["hitl_resolved_by"] = resolver
                    entry["hitl_resolved_at"] = ts
                    raw = json.dumps(entry, default=str)
                    updated += 1
            # fail-open: one malformed JSON-lines entry must not abort resolving the rest;
            # raw line is preserved unchanged below either way
            except Exception:
                pass
            lines.append(raw)

    with log_file.open("w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return updated


# ── Helpers ───────────────────────────────────────────────────────────────────


def _load_json(path: Path, default: object = None) -> object:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        return default
    try:
        with path.open() as fh:
            return json.load(fh)
    except Exception:
        return default


def _save_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(data, fh, indent=2)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Promote a production trace into the golden eval dataset."
    )
    parser.add_argument("case_id", help="Unique case ID (may match log event name)")
    parser.add_argument("input_query", help="Input that caused the issue")
    parser.add_argument(
        "correct_output", help="The correct expected output or behaviour"
    )
    parser.add_argument("--tool", default="any", help="Expected tool name")
    parser.add_argument("--note", default="", help="Resolution explanation")
    parser.add_argument("--resolved-by", default="", help="Resolver email/ID")
    parser.add_argument("--no-rerun", action="store_true", help="Skip eval re-run")
    args = parser.parse_args()

    result = promote(
        case_id=args.case_id,
        input_query=args.input_query,
        correct_output=args.correct_output,
        expected_tool=args.tool,
        resolution_note=args.note or None,
        resolved_by=args.resolved_by or None,
        rerun_evals=not args.no_rerun,
    )
    print(json.dumps(result, indent=2))
