"""
sync-ui-feedback.py — Pulls Phoenix UI annotations and promotes HITL feedback
into the golden eval dataset.

Workflow:
  1. Connect to Arize Phoenix REST API.
  2. Query for spans annotated with thumbs_down / correction labels.
  3. For each annotation that is not yet in golden_evals.json, call promote().
  4. Record sync state to avoid re-processing on subsequent runs.

Called by:
  - ai-test-evals shell function (before running scorecard)
  - GitHub Actions CD workflow (post-deploy sync)

Requires:
  AGENT_PHOENIX_ENDPOINT — Phoenix server URL (default: http://localhost:6006)
  AGENT_OWNER_ID         — Logged as the syncer identity
"""

from __future__ import annotations

import os
import sys
from typing import Any, Optional

PHOENIX_ENDPOINT = os.environ.get("AGENT_PHOENIX_ENDPOINT", "http://localhost:6006")

# Dedupe-list bound (ReviewFindings-2026-07-18 C4): synced_span_ids used to
# grow forever — loaded and rewritten on every run. Only the newest entries
# matter for deduping (Phoenix annotations well behind the cursor don't
# re-surface), so keep the most recent N.
MAX_SYNCED_IDS = 5000


# ── Helpers ───────────────────────────────────────────────────────────────────

from _shared import _repo_root, _iso_now  # noqa: E402,F401 — _repo_root re-exported for tests
from _shared import _load_sync_state, _save_sync_state  # noqa: E402
from _shared import _phoenix_get as _shared_phoenix_get  # noqa: E402


def _load_promote():
    """Load promote() from promote-learning.py by file path.

    The old `from promote_learning import promote` could NEVER resolve —
    the file is `promote-learning.py` (dash, not a valid module name) — so
    every negative annotation raised ModuleNotFoundError inside the loop's
    try/except and was counted as an error instead of promoted. Found by
    scripts/test/test_promotion_loop.py (TestCoverageReview gap 4)."""
    # _shared.load_script loads by path and caches — this had reimplemented
    # both, including its own sys.modules bookkeeping.
    from _shared import load_script

    return load_script("promote-learning").promote


# ── Phoenix API client ────────────────────────────────────────────────────────


def _phoenix_get(path: str, params: Optional[dict] = None) -> Any:
    """Make a GET request to Phoenix REST API."""
    return _shared_phoenix_get(PHOENIX_ENDPOINT, path, params)


def _fetch_annotations() -> list[dict]:
    """
    Fetch all span annotations from Phoenix.
    Returns list of annotation dicts.
    """
    try:
        # Phoenix ≥4.x REST endpoint for annotations
        data = _phoenix_get("/v1/span_annotations")
        return data.get("data", []) if isinstance(data, dict) else data
    except Exception as exc:
        print(f"   ⚠️  Could not fetch annotations: {exc}")
        return []


def _fetch_span(span_id: str) -> Optional[dict]:
    """Fetch span details by ID."""
    try:
        return _phoenix_get(f"/v1/spans/{span_id}")
    except Exception:
        return None


# ── Main sync ─────────────────────────────────────────────────────────────────


def sync() -> dict:
    """
    Pull Phoenix annotations and promote unsynced negative feedback.

    Returns:
        {"synced": int, "skipped": int, "errors": int}
    """
    print(f"🔄 Syncing Phoenix feedback from {PHOENIX_ENDPOINT}...")

    state = _load_sync_state()
    synced_ids = list(state.get("synced_span_ids", []))  # insertion-ordered
    already_synced = set(synced_ids)
    stats = {"synced": 0, "skipped": 0, "errors": 0}

    # Check Phoenix availability
    try:
        _phoenix_get("/healthz")
    except Exception:
        print(f"   ⚠️  Phoenix is offline at {PHOENIX_ENDPOINT}. Skipping sync.")
        return stats

    annotations = _fetch_annotations()
    print(f"   Found {len(annotations)} annotation(s)")

    for annotation in annotations:
        span_id = annotation.get("span_id") or annotation.get("spanId", "")
        label = (annotation.get("label") or annotation.get("name", "")).lower()
        score = annotation.get("score")
        note = annotation.get("explanation") or annotation.get("note", "")

        # Only process negative feedback: thumbs_down, incorrect, correction, fail
        is_negative = (
            "thumbs_down" in label
            or "incorrect" in label
            or "correction" in label
            or "fail" in label
            or (score is not None and float(score) < 0.5)
        )
        if not is_negative:
            stats["skipped"] += 1
            continue

        if span_id in already_synced:
            stats["skipped"] += 1
            continue

        # Fetch the original span to get input/output
        span = _fetch_span(span_id)
        if not span:
            stats["errors"] += 1
            continue

        # Extract input and output from span attributes
        span_attrs = span.get("attributes", {}) or {}
        input_val = (
            span_attrs.get("input.value")
            or span_attrs.get("llm.input_messages", [{}])[0].get("message.content", "")
            if isinstance(span_attrs.get("llm.input_messages"), list)
            else ""
        ) or span.get("name", "unknown_input")

        if not input_val:
            stats["errors"] += 1
            continue

        # Generate case ID from span
        case_id = f"phoenix:{span_id[:12]}"

        try:
            promote = _load_promote()

            promote(
                case_id=case_id,
                input_query=str(input_val)[:500],
                correct_output=note
                or f"[Human annotation: {label}] Correct output needed.",
                expected_tool=span_attrs.get("tool.name", "any"),
                resolution_note=note
                or f"Phoenix annotation: {label} on span {span_id}",
                resolved_by=os.environ.get("AGENT_OWNER_ID", "phoenix-sync"),
                rerun_evals=False,  # Caller decides when to re-run
            )
            already_synced.add(span_id)
            synced_ids.append(span_id)
            stats["synced"] += 1
            print(f"   ✅ Promoted: {case_id} (label={label})")
        except Exception as exc:
            print(f"   ❌ Failed to promote {span_id}: {exc}")
            stats["errors"] += 1

    # Persist updated sync state — newest MAX_SYNCED_IDS only (C4 bound)
    state["synced_span_ids"] = synced_ids[-MAX_SYNCED_IDS:]
    state["last_sync"] = _iso_now()
    _save_sync_state(state)

    print(
        f"\n   Sync complete — promoted: {stats['synced']}, "
        f"skipped: {stats['skipped']}, errors: {stats['errors']}"
    )
    return stats


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    result = sync()
    sys.exit(0 if result["errors"] == 0 else 1)
