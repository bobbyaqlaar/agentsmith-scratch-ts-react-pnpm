"""
shadow-eval.py — Async shadow eval sampler over live production traces
(Product_Archive.md P1c, SPECS.md §9: "An async sampler evaluates 5% of
production traces post-hoc").

Workflow:
  1. List recent spans from the tenant's Phoenix (`environment=production`,
     last --since-hours), via Phoenix's REST `/v1/projects/{project}/spans`
     (same endpoint shape used elsewhere in this script set).
  2. Sample --sample-rate of them deterministically (hash of span_id, so a
     re-run over the same window samples the same spans rather than
     re-rolling the dice).
  3. Judge each sampled span with the exact same judge-calling logic
     run-evals.py uses (scripts/eval_judge.py, factored out for this reuse)
     — no project_response generation step needed, the span already has a
     real production input/output to score.
  4. Write the result back to Phoenix as a span annotation named
     "shadow_eval" with metadata `{"eval.type": "shadow"}` (SPECS.md §9's
     documented tag) via POST /v1/span_annotations — this is Phoenix's
     real mechanism for attaching post-hoc scores to existing spans;
     "experiments" in Phoenix's API are dataset/offline-run objects, not a
     fit for annotating already-ingested production spans.
  5. Track which span_ids have already been shadow-evaluated in
     .agent-rfc/fixtures/sync_state.json (new key, doesn't collide with
     sync-ui-feedback.py's "synced_span_ids" or sync-portal-history.py's
     "synced_history_entry_ids") so a scheduled re-run doesn't re-judge
     the same span.

Shadow evals never auto-promote to the golden dataset — that stays
HITL-gated via `ai-stack-promote`. Failures are surfaced for a human to
review via portal/lib/promotions.ts's suggested-promotion queue, not
applied automatically.

Called by:
  - workflow-templates/shadow-eval.yml (optional, nightly cron — tenants
    opt in by copying the template, same posture as cd-deploy.yml's
    replacement)
  - manually: python3 scripts/shadow-eval.py --sample-rate 0.05

Requires:
  AGENT_PHOENIX_ENDPOINT — Phoenix server URL (default: http://localhost:6006)
  AGENT_JUDGE_MODEL       — judge model id override (unset: the `judge` role
                            in models.yaml, via scripts/_shared.py judge_model())
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).parent))

PHOENIX_ENDPOINT = os.environ.get("AGENT_PHOENIX_ENDPOINT", "http://localhost:6006")
SHADOW_CRITERIA = {
    "name": "Shadow Eval (production sample)",
    "instructions": (
        "Judge whether ACTUAL OUTPUT correctly and safely answers INPUT. "
        "There is no reference output — judge from first principles."
    ),
}


from _shared import _repo_root  # noqa: E402,F401 — re-export for callers/tests
from _shared import _load_sync_state, _save_sync_state  # noqa: E402
from _shared import _phoenix_get as _shared_phoenix_get  # noqa: E402
from _shared import _phoenix_post as _shared_phoenix_post  # noqa: E402
from _shared import judge_model as _resolve_judge_model  # noqa: E402


def _phoenix_get(path: str, params: Optional[dict] = None) -> Any:
    return _shared_phoenix_get(PHOENIX_ENDPOINT, path, params)


def _phoenix_post(path: str, body: dict) -> Any:
    return _shared_phoenix_post(PHOENIX_ENDPOINT, path, body)


def _fetch_production_spans(since_hours: float, project: str = "default") -> list[dict]:
    """List spans for the project, filtered client-side to
    environment=production (the simple-filters REST endpoint doesn't take
    an attribute-filter DSL — that's the GraphQL surface portal/lib/phoenix.ts
    uses instead). Degrades to an empty list on any Phoenix error, same
    posture as sync-ui-feedback.py's _fetch_annotations."""
    from datetime import datetime, timedelta, timezone

    start_time = (datetime.now(timezone.utc) - timedelta(hours=since_hours)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    try:
        data = _phoenix_get(
            f"/v1/projects/{project}/spans",
            params={"start_time": start_time, "limit": 1000},
        )
        spans = data.get("data", []) if isinstance(data, dict) else data
    except Exception as exc:
        print(f"   ⚠️  Could not fetch spans: {exc}")
        return []

    return [
        s
        for s in spans
        if (s.get("attributes") or {}).get("environment") == "production"
    ]


def _sample(spans: list[dict], sample_rate: float) -> list[dict]:
    """Deterministic sampling keyed on span_id so a re-run over an
    overlapping window picks the same spans rather than re-rolling."""
    sampled = []
    for span in spans:
        span_id = span["context"]["span_id"]
        digest = int(hashlib.sha256(span_id.encode("utf-8")).hexdigest(), 16)
        if (digest % 10_000) / 10_000 < sample_rate:
            sampled.append(span)
    return sampled


def _annotate_shadow_result(span_id: str, scored: dict) -> None:
    _phoenix_post(
        "/v1/span_annotations",
        {
            "data": [
                {
                    "name": "shadow_eval",
                    "annotator_kind": "LLM",
                    "span_id": span_id,
                    "result": {
                        "label": "pass" if scored.get("score", 0.0) >= 0.5 else "fail",
                        "score": float(scored.get("score", 0.0)),
                        "explanation": scored.get("quality_notes", ""),
                    },
                    "metadata": {"eval.type": "shadow"},
                }
            ]
        },
    )


def run_shadow_eval(sample_rate: float = 0.05, since_hours: float = 24.0) -> dict:
    from eval_judge import judge_prompt, run_judge

    judge_model = _resolve_judge_model()
    stats = {"sampled": 0, "judged": 0, "failed": 0, "skipped": 0, "errors": 0}

    print(
        f"🌗 Shadow eval — sampling {sample_rate:.0%} of production traces "
        f"(last {since_hours}h) from {PHOENIX_ENDPOINT}"
    )

    spans = _fetch_production_spans(since_hours)
    if not spans:
        print("   ℹ️  No production spans found in window — nothing to sample.")
        return stats

    sampled = _sample(spans, sample_rate)
    stats["sampled"] = len(sampled)
    print(
        f"   {len(spans)} production span(s) in window, {len(sampled)} sampled at {sample_rate:.0%}"
    )

    state = _load_sync_state()
    already_evaluated = set(state.get("shadow_evaluated_span_ids", []))

    for span in sampled:
        span_id = span["context"]["span_id"]
        if span_id in already_evaluated:
            stats["skipped"] += 1
            continue

        attrs = span.get("attributes") or {}
        input_value = attrs.get("input.value", "")
        output_value = attrs.get("output.value", "")
        if not input_value or not output_value:
            stats["skipped"] += 1
            continue

        prompt = judge_prompt(
            instructions=SHADOW_CRITERIA["instructions"],
            historical_text="(none — shadow eval has no reference output)",
            input_text=input_value,
            expected_tool="any",
            reference_output="(none — production span, judged from first principles)",
            actual_output=output_value,
        )
        scored = run_judge(prompt, judge_model)

        try:
            _annotate_shadow_result(span_id, scored)
        except Exception as exc:
            print(f"   ⚠️  Failed to write shadow_eval annotation for {span_id}: {exc}")
            stats["errors"] += 1
            continue

        stats["judged"] += 1
        if scored.get("score", 0.0) < 0.5:
            stats["failed"] += 1
        already_evaluated.add(span_id)
        status = "✅" if scored.get("score", 0.0) >= 0.5 else "❌"
        print(f"   {status} {span_id} score={scored.get('score', 0.0):.2f}")

    state["shadow_evaluated_span_ids"] = list(already_evaluated)
    _save_sync_state(state)

    print(
        f"\n   Judged {stats['judged']}, failed {stats['failed']}, "
        f"skipped {stats['skipped']} (already evaluated or no I/O), errors {stats['errors']}."
    )
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Shadow eval sampler over production traces"
    )
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=0.05,
        help="Fraction of production spans to judge (default: 0.05)",
    )
    parser.add_argument(
        "--since-hours",
        type=float,
        default=24.0,
        help="Window of production spans to consider (default: 24)",
    )
    args = parser.parse_args()
    result = run_shadow_eval(sample_rate=args.sample_rate, since_hours=args.since_hours)
    # Async/post-hoc by design — never blocks or fails the calling job.
    sys.exit(0)
