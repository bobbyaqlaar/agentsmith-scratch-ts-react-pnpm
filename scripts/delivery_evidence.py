"""
scripts/delivery_evidence.py — promote-time evidence pack (JSON + Markdown).

Collects Delivery Model artifacts (eval scorecard, fairness, redaction notes,
guardrail/HITL pointers) into:
  .agent-rfc/fixtures/delivery_evidence.json
  .agent-rfc/fixtures/delivery_evidence.md

Usage:
    python3 scripts/delivery_evidence.py
    python3 scripts/delivery_evidence.py --root /path/to/tenant

Exit 0 always — missing optional items are marked "missing" in the manifest
(soft evidence, not a hard gate). Pair with verify_system.py --check-delivery-model.

Four statuses, and the distinction between the middle two is the point:

    present       the artifact exists and the run that wrote it made a claim
    inconclusive  the artifact exists and the run made NO claim — `no_verdict`,
                  a null `passed`, or a grader other than the one requested
    missing       no artifact, or an unreadable one
    note          a pointer to how the evidence is produced; never a result

`inconclusive` exists because run-evals.py now writes its scorecard on both
exit paths. Before that, "graded and passed" and "graded nothing" both landed
in `present` with a number beside them, in a pack an auditor reads.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, NamedTuple

from _shared import RESULTS_FILE, _repo_root

# delivery_model.py owns where these files live and how they parse. Restating
# `root / ".agenticframework" / "tenant.yaml"` here is the same defect as the
# portal's `lib/tenants.ts` inlining a type that `lib/isolation.ts` exists to
# provide: two copies of one fact, and the pack is the copy that drifts.
from delivery_model import _load_yaml, org_policy_path, tenant_yaml_path


TS_FORMAT = "%Y-%m-%dT%H:%M:%SZ"

# `_shared.fixtures_path` owns this layout but resolves it from the repo root,
# and this script takes an explicit `--root` so a tenant pack can be built from
# anywhere. One statement of the relative path rather than the three it was
# spelled out in — collect_evidence, the row's `path` field, and the writer.
FIXTURES_REL = (".agent-rfc", "fixtures")


def _fixtures_dir(root: Path) -> Path:
    return root.joinpath(*FIXTURES_REL)


def _fixtures_rel(name: str) -> str:
    return "/".join((*FIXTURES_REL, name))

# Status vocabulary. `inconclusive` is not decoration. run-evals.py writes its
# scorecard on BOTH exit paths now, so a run that reported NO VERDICT leaves a
# real file behind that claims nothing — and folding that into `present` made
# "we measured and it held" and "we measured nothing" render identically in an
# auditor-facing pack. Pillar 15: report the states separately and name them.
PRESENT = "present"
INCONCLUSIVE = "inconclusive"
MISSING = "missing"
NOTE = "note"
STATUSES = (PRESENT, INCONCLUSIVE, MISSING, NOTE)


def _iso_now() -> datetime:
    return datetime.now(timezone.utc)


def _item(
    item_id: str,
    label: str,
    status: str,
    path: str | None = None,
    detail: str = "",
) -> dict[str, Any]:
    return {
        "id": item_id,
        "label": label,
        "status": status,  # one of STATUSES
        "path": path,
        "detail": detail,
    }


def _fmt(value: Any) -> str:
    """A number at fixed precision, or a named absence.

    `None` becomes "NOT MEASURED" rather than 0.000 or an empty cell, for the
    reason the whole module exists: a blank reads as "fine", and the metrics
    feeding this pack return None precisely when they had no data.
    """
    if value is None:
        return "NOT MEASURED"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{value:.3f}"
    return str(value)


def _age_suffix(ts: Any, now: datetime) -> str:
    if not isinstance(ts, str):
        return ""
    try:
        ran = datetime.strptime(ts, TS_FORMAT).replace(tzinfo=timezone.utc)
    except ValueError:
        return " (unparseable)"
    return f" ({(now - ran).days}d ago)"


def _provenance(data: dict[str, Any], now: datetime) -> str:
    """When the run happened, how much of it graded, and who graded it.

    The pack stamps its own generation time, which says nothing about the age
    of the scorecards it read — a months-old fixture and a fresh one rendered
    identically. On 2026-08-23 a pack was generated from dry-run failure
    fixtures and dated to the minute it was written.
    """
    ts = data.get("timestamp")
    parts = [f"ran={ts}{_age_suffix(ts, now)}" if ts else "ran=UNRECORDED"]

    graded, total = data.get("cases_graded"), data.get("cases_total")
    if isinstance(graded, int) and isinstance(total, int):
        # A partial grade is the free tier's normal failure and it does not go
        # red — it reports NO VERDICT and exits 0. Show the fraction so the
        # pack says how much of the suite the number rests on.
        parts.append(f"graded={graded}/{total}")

    # What ANSWERED, not what was asked for. A scorecard graded by two models
    # is not a scorecard, and the requested id cannot show a substitution.
    used = data.get("judge_models_used") or data.get("judge_model")
    if isinstance(used, list):
        used = ", ".join(used) if used else None
    if used:
        parts.append(f"judge={used}")

    return " ".join(parts)


def _verdict_status(data: dict[str, Any]) -> tuple[str, str]:
    """Map a scorecard artifact onto (status, verdict phrase).

    `passed` may be null and `verdict` carries which exit path wrote the file.
    The CHANGELOG entry that added them asks every consumer to treat a missing
    or null `passed` as "not a pass" rather than falsy-as-fail — this is that
    consumer, and it was not updated when the producer was.
    """
    verdict = data.get("verdict")
    passed = data.get("passed")
    if verdict == "pass":
        return PRESENT, "verdict=pass"
    if verdict == "fail":
        # Evidence that says no is still evidence. The pack records what was
        # measured, not only what succeeded; a failing scorecard is a finding.
        return PRESENT, "verdict=fail"
    if verdict == "no_verdict":
        return INCONCLUSIVE, "verdict=no_verdict — the run made no claim"
    if verdict is None and isinstance(passed, bool):
        # Predates the `verdict` field, when `passed` was unconditionally a
        # bool and therefore still means what it says.
        return PRESENT, f"passed={passed} (artifact predates `verdict`)"
    return INCONCLUSIVE, f"verdict={verdict!r} passed={passed!r} — not a pass"


def _grader_mismatch(data: dict[str, Any]) -> str | None:
    """The grader that ANSWERED, against the one the run asked for.

    In a normal run these are identical by construction: eval_judge.py stamps
    `judged_by` with the id it was handed. So a mismatch does not mean a model
    was swapped mid-run — it means the artifact did not come off the standard
    path at all. That is how the dry-run FAILURE simulations sitting in
    KYC Sentinel's fixtures on 2026-08-23 got reported as delivery evidence:
    `judge_models_used=['sim']`, route `sim/x`, and a pack that showed neither.

    run-evals.py already fails a scorecard graded by more than one model. It
    cannot catch a single substituted grader, because one is not "more than
    one" — this is that gap, seen from the consumer side.

    Returns None when the graders agree or when there is nothing to compare.
    """
    requested = data.get("judge_model")
    used = data.get("judge_models_used")
    if not requested or not isinstance(used, list) or not used:
        return None
    if set(used) == {requested}:
        return None
    return f"graded by {', '.join(sorted(used))}, not the requested {requested}"


def _golden_detail(data: dict[str, Any]) -> str:
    return f"avg_score={_fmt(data.get('avg_score'))} threshold={_fmt(data.get('threshold'))}"


def _fairness_detail(data: dict[str, Any]) -> str:
    """Worst pair first — that is the number the gate actually compares.

    Parity moved to the worst pair rather than the mean because averaging made
    the suite weaker the more pairs it had: one diverging pair reads 0.750 over
    2 pairs but 0.950 over 10. This pack reported only the mean, which is the
    superseded metric, so a divergence a bigger suite outvoted did not show up
    here at all.
    """
    pairs = data.get("pair_parity")
    worst = min(pairs.values()) if isinstance(pairs, dict) and pairs else None
    return (
        f"fairness={_fmt(data.get('avg_fairness'))} "
        f"worst_pair_parity={_fmt(worst)} "
        f"(mean {_fmt(data.get('avg_pair_parity'))} over {len(pairs) if isinstance(pairs, dict) else 0} pair(s))"
    )


def _hallucination_detail(data: dict[str, Any]) -> str:
    """Three states for detection, mirroring run-evals.py's own report.

    A miss rate of 0.000 means the planted control was flagged. `null` means it
    was not graded — and whether any control was DECLARED is the difference
    between "the control errored" and "this suite has never been asked to
    detect anything". Both used to render as a clean 0.000.
    """
    miss = data.get("hallucination_miss_rate")
    declared = data.get("hallucination_controls_declared")
    if isinstance(miss, (int, float)) and not isinstance(miss, bool):
        detection = f"detection_miss={_fmt(miss)}"
        if miss > 0.0:
            detection += " ❌ a planted hallucination went undetected"
    elif isinstance(declared, int) and declared > 0:
        detection = f"detection_miss=NOT GRADED — {declared} positive control(s) declared, none graded"
    elif declared == 0:
        detection = "detection_miss=NO POSITIVE CONTROL declared in this suite"
    else:
        # No key at all: written by a run-evals that predates persisting it.
        detection = "detection_miss=UNRECORDED — artifact predates the field"
    return f"flagged_claim_rate={_fmt(data.get('hallucination_flag_rate'))} {detection}"


class Scorecard(NamedTuple):
    """One judged suite's row in the pack.

    `suite` is the key into `_shared.RESULTS_FILE` rather than a filename of
    its own. This tuple used to spell the three filenames out, which made it a
    FOURTH copy of a catalog `_shared` already owns — and the failure mode is
    quiet: rename a suite's results file there and the pack reads a path that
    no longer exists, reports `missing`, and an auditor sees "never run" for a
    suite that runs on every push. Same defect as the portal's audit-event
    catalog living in a union plus two hand-kept arrays.
    """

    item_id: str
    label: str
    suite: str
    detail_fn: Callable[[dict[str, Any]], str]
    refresh_cmd: str


SCORECARDS = (
    Scorecard(
        "eval_scorecard",
        "Golden eval scorecard",
        "golden",
        _golden_detail,
        "python3 scripts/run-evals.py --fail-below 0.80",
    ),
    Scorecard(
        "fairness_scorecard",
        "Fairness eval scorecard",
        "fairness",
        _fairness_detail,
        "python3 scripts/run-evals.py --suite fairness --fail-below 0.80",
    ),
    Scorecard(
        # The third judged suite, and the one the pack omitted entirely. A
        # missing line reads as "did not apply" — so the grounding gate, whose
        # detection half is the claim an auditor would most want evidenced,
        # produced no row at all.
        "hallucination_scorecard",
        "Hallucination / grounding scorecard",
        "hallucination",
        _hallucination_detail,
        "python3 scripts/run-evals.py --suite hallucination --fail-below 0.80",
    ),
)

# Fail at import, not by rendering a `missing` row for a suite that ran fine.
_unknown_suites = [c.suite for c in SCORECARDS if c.suite not in RESULTS_FILE]
if _unknown_suites:  # pragma: no cover — guards a rename in _shared.RESULTS_FILE
    raise ImportError(
        f"SCORECARDS names suite(s) absent from _shared.RESULTS_FILE: {_unknown_suites}"
    )


def _scorecard_item(fixtures: Path, now: datetime, card: Scorecard) -> dict[str, Any]:
    filename = RESULTS_FILE[card.suite]
    path = fixtures / filename
    rel = _fixtures_rel(filename)
    if not path.exists():
        return _item(card.item_id, card.label, MISSING, None, f"Run: {card.refresh_cmd}")
    try:
        data = json.loads(path.read_text())
    except Exception as exc:
        return _item(card.item_id, card.label, MISSING, rel, f"unreadable: {exc}")
    if not isinstance(data, dict):
        return _item(card.item_id, card.label, MISSING, rel, "unreadable: not a JSON object")
    status, verdict_phrase = _verdict_status(data)
    mismatch = _grader_mismatch(data)
    if mismatch:
        # A verdict from the wrong grader is not evidence about this tenant's
        # calibrated gate, whichever way it went. Downgrade rather than drop:
        # the row still shows what was found and who found it.
        status = INCONCLUSIVE
        verdict_phrase = f"{verdict_phrase} — ⚠️ {mismatch}"
    # Newlines, not <br> — the JSON manifest is machine-readable and the
    # Markdown renderer is the one place that knows it renders HTML.
    detail = f"{verdict_phrase}\n{card.detail_fn(data)}\n{_provenance(data, now)}"
    return _item(card.item_id, card.label, status, rel, detail)


def collect_evidence(root: Path) -> dict[str, Any]:
    fixtures = _fixtures_dir(root)
    now = _iso_now()
    items: list[dict[str, Any]] = [
        _scorecard_item(fixtures, now, card) for card in SCORECARDS
    ]

    # Redaction: note how to produce evidence (CI runs verify_system --check-redaction)
    env = os.environ.get("ENVIRONMENT", "").strip() or "(unset)"
    items.append(
        _item(
            "redaction_check",
            "Trace redaction compliance",
            NOTE,
            None,
            f"ENVIRONMENT={env}. Produce via: "
            "ENVIRONMENT=staging|production python3 scripts/verify_system.py --check-redaction",
        )
    )

    guardrail = os.environ.get("INPUT_GUARDRAIL", "").strip() or "(unset → env default)"
    items.append(
        _item(
            "input_guardrail",
            "Pre-call PII guardrail mode",
            NOTE,
            "runtime/input_guardrail.py",
            f"INPUT_GUARDRAIL={guardrail}",
        )
    )

    org_policy = org_policy_path(root)
    items.append(
        _item(
            "org_policy",
            "Org delivery policy",
            PRESENT if org_policy.exists() else MISSING,
            ".agenticframework/org-policy.yaml" if org_policy.exists() else None,
            "" if org_policy.exists() else "Copy templates/delivery-model/org-policy.example.yaml",
        )
    )

    # The file existing is not the evidence — the two `delivery.*` keys are.
    # This row said `present` on the file alone and then printed "Set
    # delivery.platform + delivery.data_access_pattern" whether or not they
    # were set, so the same cell served as both a confirmation and an
    # outstanding instruction. A tenant.yaml with neither key read as delivered.
    tenant_path = tenant_yaml_path(root)
    if not tenant_path.exists():
        items.append(
            _item(
                "tenant_yaml",
                "Tenant config",
                MISSING,
                None,
                "Set delivery.platform + delivery.data_access_pattern",
            )
        )
    else:
        delivery = _load_yaml(tenant_path).get("delivery")
        delivery = delivery if isinstance(delivery, dict) else {}
        absent = [k for k in ("platform", "data_access_pattern") if not delivery.get(k)]
        items.append(
            _item(
                "tenant_yaml",
                "Tenant config",
                INCONCLUSIVE if absent else PRESENT,
                ".agenticframework/tenant.yaml",
                f"delivery.{' and delivery.'.join(absent)} not set"
                if absent
                else f"delivery.platform={delivery['platform']} "
                f"delivery.data_access_pattern={delivery['data_access_pattern']}",
            )
        )

    items.append(
        _item(
            "hitl_audit",
            "HITL / audit trail",
            NOTE,
            None,
            "Export Ops Portal GET /api/audit and Phoenix HITL annotations for high-impact flows",
        )
    )

    # Counted per status rather than "everything left over is a note". The old
    # arithmetic derived notes by subtraction, so any status it did not know
    # about was silently absorbed into that bucket — adding `inconclusive`
    # would have been invisible in the summary line while changing the table.
    counts = {status: sum(1 for i in items if i["status"] == status) for status in STATUSES}
    unknown = [i["status"] for i in items if i["status"] not in STATUSES]
    if unknown:  # pragma: no cover — a typo'd status must not vanish
        raise ValueError(f"unknown evidence status(es): {sorted(set(unknown))}")

    return {
        "timestamp": now.strftime(TS_FORMAT),
        "root": str(root),
        "summary": counts,
        "items": items,
    }


def render_markdown(manifest: dict[str, Any]) -> str:
    lines = [
        "# Delivery Model — promote evidence pack",
        "",
        f"Generated: `{manifest.get('timestamp')}`",
        "",
        "Summary: "
        + ", ".join(
            f"**{manifest['summary'].get(status, 0)}** {status}" for status in STATUSES
        )
        + ".",
        "",
        "> `inconclusive` is not a pass. It means the artifact exists and the run",
        "> that wrote it made no claim — a judge that never answered, or a suite",
        "> graded in part. Read it as evidence not yet obtained.",
        "",
        "| ID | Status | Path / detail |",
        "|---|---|---|",
    ]
    for item in manifest["items"]:
        path = item.get("path") or "—"
        detail = (item.get("detail") or "").replace("|", "\\|").replace("\n", "<br>")
        lines.append(
            f"| `{item['id']}` | **{item['status']}** | {path}<br>{detail} |"
        )
    lines.extend(
        [
            "",
            "## How to refresh",
            "",
            "```bash",
            # Derived from SCORECARDS, which is also what a `missing` row tells
            # you to run. Two hand-kept lists of the same commands drift, and
            # the one an auditor follows is whichever they read first.
            *(card.refresh_cmd for card in SCORECARDS),
            "ENVIRONMENT=staging python3 scripts/verify_system.py --check-redaction",
            "python3 scripts/verify_system.py --check-delivery-model",
            "python3 scripts/delivery_evidence.py",
            "```",
            "",
            "See `docs/delivery-model.md` and `docs/iso-42001-control-map.md`.",
            "",
        ]
    )
    return "\n".join(lines)


def write_evidence_pack(root: Path, manifest: dict[str, Any] | None = None) -> dict[str, Path]:
    if manifest is None:
        manifest = collect_evidence(root)
    fixtures = _fixtures_dir(root)
    fixtures.mkdir(parents=True, exist_ok=True)
    json_path = fixtures / "delivery_evidence.json"
    md_path = fixtures / "delivery_evidence.md"
    json_path.write_text(json.dumps(manifest, indent=2) + "\n")
    md_path.write_text(render_markdown(manifest))
    return {"json": json_path, "md": md_path}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write Delivery Model evidence pack")
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Tenant/repo root (default: git root / cwd)",
    )
    args = parser.parse_args(argv)
    root = args.root.resolve() if args.root else _repo_root()
    paths = write_evidence_pack(root)
    print(f"Wrote {paths['json']}")
    print(f"Wrote {paths['md']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
