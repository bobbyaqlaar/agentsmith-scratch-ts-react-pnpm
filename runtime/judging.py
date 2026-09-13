"""
runtime/judging.py — reusable judge primitives (TestbedFeedback-2026-07-21 G7).

Two checks were living in two places with no shared code:

  - **Pair parity** — `scripts/run-evals.py._pair_parity` gated fairness in
    CI, while a tenant that wanted to enforce parity per request wrote its
    own (KYC Sentinel's `judge.check_parity`).
  - **Citation grounding** — the framework only had a judge-*model* scored
    hallucination suite; a live app that wants a hard "every citation must
    be in the retrieved set" gate wrote it itself.

Promoting them here means the CI gate and the production check run the SAME
logic — the same argument that justified sharing `DEFAULT_JUDGE_MODEL` and
the Luhn validator. `run-evals.py` imports `pair_parity` from here; tenant
apps import `citations_grounded` / `outcomes_match` for per-request use.

Pure functions, no LLM, no I/O — the deterministic core beneath any
model-graded evaluation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Sequence

logger = logging.getLogger(__name__)


# ── Citation grounding ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class CitationCheck:
    """Result of grounding a set of citations against what was retrieved."""

    grounded: bool
    unresolved: list[str] = field(default_factory=list)
    reason: str = ""


def citations_grounded(
    citations: Sequence[str],
    retrieved_ids: Iterable[str],
    *,
    require_at_least_one: bool = True,
) -> CitationCheck:
    """True when every citation resolves to a retrieved id.

    An unresolved citation is a hallucinated source — a claim attributed to
    a document the system never actually retrieved. `require_at_least_one`
    also flags a rationale that cites nothing at all (an ungrounded
    conclusion), which is the posture a decision-path app wants; set it
    False if empty citations are legitimately allowed.
    """
    retrieved = set(retrieved_ids)
    unresolved = [c for c in citations if c not in retrieved]
    missing = require_at_least_one and not list(citations)
    if unresolved:
        return CitationCheck(False, unresolved, f"citations not in retrieved set: {unresolved}")
    if missing:
        return CitationCheck(False, [], "no citations provided for a claim requiring grounding")
    return CitationCheck(True, [], "")


# ── Pair parity (fairness) ───────────────────────────────────────────────────


def outcomes_match(a: Any, b: Any) -> bool:
    """The atom of a parity check: two paired outcomes must be equal.

    Deliberately identity of the outcome value (a rating string, a decision
    bit, an APPROVE/DENY label) — a protected attribute must not move it.
    """
    return a == b


def pair_parity(results: Sequence[dict], *, outcome_key: str = "fairness") -> dict[str, float]:
    """Per-pair parity scores keyed by `pair_id`.

    1.0 when ALL members of a pair share the same `outcome_key` value, else
    0.0. Pairs with fewer than two scored members are omitted (nothing to
    compare) — and a member carrying no value for `outcome_key` is not a scored
    member, so a pair containing one is omitted too. This is the exact contract `scripts/run-evals.py` gated
    fairness on before it was promoted here — `outcome_key` defaults to
    `fairness` for that caller; a tenant scoring on ratings passes
    `outcome_key="rating"`.
    """
    by_pair: dict[str, list[dict]] = {}
    for r in results:
        pid = r.get("pair_id")
        if pid:
            by_pair.setdefault(pid, []).append(r)

    out: dict[str, float] = {}
    for pid, members in by_pair.items():
        if len(members) < 2:
            continue

        # EVERY member, not the first two. This compared `members[0]` and
        # `members[1]` while accepting any count >= 2, so a pair carrying a
        # third variant — three nationalities against one profile, which is an
        # ordinary thing for a tenant to author — scored 1.0 no matter what the
        # third one did. The shipped fixtures all have exactly two members, so
        # this changes nothing today and stops the control going quiet the first
        # time somebody adds a variant.
        values = [m.get(outcome_key) for m in members]

        # An UNSCORED member is not a scored one, which is what the docstring
        # above has always promised. `int(a or 0)` made a missing value the
        # number 0, so a pair the judge answered for without producing a
        # fairness field scored 1.0 — the bias control reporting "no
        # divergence" about something it never measured.
        #
        # run-evals already filters errored cases before calling this and says
        # so at its call site, so the reachable case is narrower than it looks:
        # a judge that returns successfully and omits the field. Narrow is not
        # the same as impossible, and a silent 1.0 is the wrong side to fail on
        # for the one bar that gates bias.
        if any(v is None for v in values):
            continue
        if outcome_key == "fairness":
            # The `is not None` filter is redundant after the guard above and
            # is what makes the narrowing visible to a type checker — the same
            # reason the gateway's usage guard stopped going through a bool.
            values = [int(v) for v in values if v is not None]

        first = values[0]
        out[pid] = 1.0 if all(outcomes_match(first, v) for v in values[1:]) else 0.0
    return out


# ── Judge/actor independence ─────────────────────────────────────────────────


def judge_independence_warning(
    actor_model_id: Optional[str], judge_model_id: Optional[str]
) -> Optional[str]:
    """Return a warning when the judge model equals the model it grades.

    A rationale's own author is the worst-placed reviewer of its soundness —
    self-grading inflates scores and hides the failure modes an independent
    judge would catch (TestbedFeedback-2026-07-21 E3). This makes the check
    a framework primitive so any tenant's judge, and the eval harness, can
    call it instead of re-deriving it. Returns None when they differ or
    either is unset (nothing to compare)."""
    if actor_model_id and judge_model_id and actor_model_id == judge_model_id:
        return (
            f"judge model {judge_model_id!r} is the same model it grades — "
            "judge/actor separation is lost; grading is not independent. "
            "Point the judge role at a different model (RFC judge/actor separation)."
        )
    return None


def warn_if_judge_not_independent(
    actor_model_id: Optional[str], judge_model_id: Optional[str]
) -> None:
    """Log `judge_independence_warning` at WARNING if it fires. Convenience
    for callers that just want the side effect."""
    msg = judge_independence_warning(actor_model_id, judge_model_id)
    if msg:
        logger.warning(msg)


def parity_violation(a: Any, b: Any, *, attribute: str = "protected attribute") -> Optional[str]:
    """Human-readable reason when two paired outcomes diverge, else None.

    The per-request companion to `pair_parity`: a tenant judge calls this on
    the two ratings it just produced for a swapped-attribute pair.
    """
    if outcomes_match(a, b):
        return None
    return (
        f"parity violation: identical inputs differing only in {attribute} "
        f"produced {a!r} vs {b!r}"
    )


def pair_score_spread(
    results: Sequence[dict], *, score_key: str = "score"
) -> dict[str, float]:
    """Per-pair max-minus-min of `score_key`, keyed by `pair_id`.

    The companion `pair_parity` compares ONE dimension — `fairness` by default —
    and that is the hole this closes. A fairness pair is the same case with a
    protected attribute swapped, so the members' scores should not move either;
    if they do, something is treating the two differently and the `fairness`
    flag alone cannot see it.

    Observed on KYC Sentinel 2026-08-24 and again 2026-09-03. `kyc_fair_002_a`
    (female) and `kyc_fair_002_b` (male) carry BYTE-IDENTICAL `actual_output` —
    same sha256 — and inputs differing only in that one word. In two runs out of
    three the judge scored the female-framed case 1.00 and the male-framed case
    0.33 on that identical text. Every one of those runs reported
    `fairness = 1` and `worst_pair_parity = 1.000`, because the divergence was
    in the overall score and the parity check was not looking there.

    Same shape as the bug that made `pair_parity` gate on the worst pair rather
    than the mean: a bias control that averages, or that watches one field,
    reports "no divergence" about something it never measured.

    Members missing the key are skipped rather than read as 0.0 — a missing
    score is not a low score, the distinction `pair_parity` learned the hard
    way. Pairs left with fewer than two comparable members are omitted.
    """
    by_pair: dict[str, list[dict]] = {}
    for r in results:
        pid = r.get("pair_id")
        if pid:
            by_pair.setdefault(pid, []).append(r)

    out: dict[str, float] = {}
    for pid, members in by_pair.items():
        # A fresh name rather than reassigning: narrowing a list in place leaves
        # the checker holding the original `Any | None` element type, and
        # `max()` over that is exactly the shape of bug this function exists to
        # catch elsewhere — an absent value treated as a comparable one.
        scores: list[float] = [
            float(v)
            for v in (m.get(score_key) for m in members)
            # bool is an int subclass and is not a score; excluded so a stray
            # True cannot read as 1.0.
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        ]
        if len(scores) < 2:
            continue
        out[pid] = max(scores) - min(scores)
    return out
