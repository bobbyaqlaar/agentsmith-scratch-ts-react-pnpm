"""
runtime/test/test_dead_letter.py — the dead-letter envelope contract
(SEC-DLQ-001 evidence; no Postgres, no Temporal server).

These assertions used to live in `test_hitl_gate.py`, which is SEC-HITL-001's
evidence. One test module cannot be two controls' proof: a green there said
"the HITL gate works", and SEC-DLQ-001 was left reading `verify_system
--check-dlq` — a reachability probe that fails whenever the database is down
and proves nothing about the envelope when it is up. Moved here so each
control points at a suite that fails for its own reason.

What is deliberately NOT here: that a row lands in Postgres. That needs a
database and belongs to the integration suite. What this module proves is the
contract that made the failure path fail in production — the producers and the
consumer agreeing on a shape — and that is pure Python.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "runtime" / "workflows"))

from runtime.dead_letter import DeadLetterQueue, dead_letter_envelope

# The envelope's own keys, named once. Both producers and the consumer are
# checked against this rather than against a list retyped per test.
ENVELOPE_KEYS = {
    "payload",
    "error",
    "tenant_id",
    "reason",
    "workflow_id",
    "gate_id",
    # Added 2026-08-25. `enqueue` is idempotent on task_id and mints a uuid4
    # when given none, so it protected callers that supplied one and nobody
    # else — and `dlq_enqueue_activity` is a Temporal activity, which is
    # at-least-once by construction. A retry after a committed insert wrote a
    # second row and a human saw one failure twice in the portal.
    "task_id",
}


def test_envelope_keys_match_what_enqueue_accepts() -> None:
    """`dead_letter_envelope` produces exactly the keyword arguments
    `DeadLetterQueue.enqueue` takes, so `enqueue(**envelope)` round-trips.

    These were written out by hand at both ends — the producer in
    run_with_hitl_gate, the consumer in dead_letter_activity — with nothing
    connecting them beyond both authors remembering the same six keys. They
    came apart: the HITL timeout path built a flattened payload with no
    `payload` or `tenant_id` key, and the consumer raised KeyError on a gate
    that had just timed out. The failure path failing.
    """
    envelope = dead_letter_envelope(
        payload={"x": 1}, error="boom", tenant_id="t1",
        reason="r", workflow_id="wf", gate_id="g",
    )
    accepted = set(inspect.signature(DeadLetterQueue.enqueue).parameters) - {"self"}
    assert set(envelope) == ENVELOPE_KEYS
    assert set(envelope) <= accepted, (
        f"envelope produces keys enqueue cannot take: {set(envelope) - accepted}"
    )


def test_the_legacy_flattened_shape_is_rejected_by_name() -> None:
    """A tenant passing the flattened shape to the GENERIC activity gets an
    error about the envelope, not about its own business fields.

    Bare `enqueue(**flattened)` says "unexpected keyword argument 'company'",
    which sends the reader looking at their payload rather than at the
    contract — and this only ever happens on a path where a gate has already
    failed.
    """
    import asyncio

    import runtime.workflows.base_workflow as bw

    if not getattr(bw, "_HAS_TEMPORAL", False):
        pytest.skip("dlq_enqueue_activity is only defined when temporalio is installed")

    flattened = {"company": "acme", "region": "eu", "error": "hitl_timeout"}
    with pytest.raises(ValueError) as caught:
        asyncio.run(bw.dlq_enqueue_activity(flattened))
    assert "dead_letter_envelope shape" in str(caught.value)
    assert "company" in str(caught.value)      # names what it actually received
    # And it surfaced without a database: this is a contract error, not an
    # infrastructure one, and it fires on a path where a gate already failed.


def test_every_producer_builds_the_envelope_through_the_builder() -> None:
    """Neither DLQ producer may hand-roll the dict.

    `run_with_recoverable_step` did exactly that — it restated the same six
    keys inline while the HITL path had already been converted to the builder.
    Both happened to agree at the time, which is the problem: nothing would
    have said otherwise if they stopped agreeing, and the first symptom is a
    KeyError on a path that only runs after something has already gone wrong.

    Matched on the AST, not on the source text, so a comment mentioning
    `dead_letter_envelope` cannot satisfy it.
    """
    import ast

    src = (ROOT / "runtime" / "workflows" / "base_workflow.py").read_text()
    tree = ast.parse(src)

    for fname in ("run_with_hitl_gate", "run_with_recoverable_step"):
        fn = next(
            (
                n for n in ast.walk(tree)
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                and n.name == fname
            ),
            None,
        )
        assert fn is not None, f"{fname} not found — did it get renamed?"

        # Every execute_activity call targeting the DLQ must pass a call, not
        # a dict literal.
        for node in ast.walk(fn):
            if not isinstance(node, ast.Call):
                continue
            target = ast.unparse(node.func)
            if "execute_activity" not in target:
                continue
            args = node.args[1:2]          # the activity's input argument
            if not args:
                continue
            activity = ast.unparse(node.args[0])
            if "dlq" not in activity and "dead_letter" not in activity:
                continue
            arg = args[0]
            if isinstance(arg, ast.Dict):
                keys = {k.value for k in arg.keys if isinstance(k, ast.Constant)}
                # The legacy flattened shape is a documented tenant contract
                # (oil-price-agent's own activity) and stays permitted; what
                # must not reappear is a hand-built copy of the ENVELOPE.
                assert not ENVELOPE_KEYS.issubset(keys), (
                    f"{fname} builds the dead-letter envelope inline; call "
                    f"dead_letter_envelope() so the keys live in one module"
                )


def test_envelope_defaults_the_optional_fields_rather_than_omitting_them() -> None:
    """A caller supplying only the required three still produces every key.

    `enqueue` has defaults for the optional fields, so omitting them would
    work — right up until a consumer does `input["gate_id"]`. Emitting the
    full shape every time means the consumer never has to guess whether a
    missing key is absent or None.
    """
    envelope = dead_letter_envelope(payload={"a": 1}, error="e", tenant_id="t")

    assert set(envelope) == ENVELOPE_KEYS
    assert envelope["reason"] is None
    assert envelope["workflow_id"] is None
    assert envelope["gate_id"] is None


# ── The DLQ task id (pass 12) ────────────────────────────────────────────────


def test_the_task_id_is_stable_across_deliveries_of_one_enqueue() -> None:
    """`dlq_enqueue_activity` is a Temporal activity, and activities are
    at-least-once. Without a stable id `enqueue` mints a uuid4 per delivery, so
    a retry after a committed insert wrote a second row and an operator saw one
    failure twice."""
    from runtime.workflows.base_workflow import _dlq_task_id

    first = _dlq_task_id("run-a", "crm-update-gate", 0)
    second = _dlq_task_id("run-a", "crm-update-gate", 0)
    assert first == second


def test_a_reset_or_retried_run_does_not_collide_with_the_previous_one() -> None:
    """Keyed on RUN id, not workflow id.

    A Temporal workflow that is reset or retried keeps its workflow_id and gets
    a new run_id. An id built from workflow_id would collide, and `ON CONFLICT
    DO NOTHING` would silently drop the new run's DLQ entry — trading a
    duplicate row, which is noise, for a missing one, which is a failure nobody
    is told about. The first version of this function made that trade.
    """
    from runtime.workflows.base_workflow import _dlq_task_id

    assert _dlq_task_id("run-a", "g", 0) != _dlq_task_id("run-b", "g", 0)


def test_each_attempt_and_each_gate_files_separately() -> None:
    """run_with_recoverable_step enqueues once per attempt on purpose — a human
    should see each failed fix, not one row that quietly overwrites itself."""
    from runtime.workflows.base_workflow import _dlq_task_id

    ids = {
        _dlq_task_id("run-a", "gate-one", 0),
        _dlq_task_id("run-a", "gate-one", 1),
        _dlq_task_id("run-a", "gate-two", 0),
        _dlq_task_id("run-a", "gate-one", "hitl_timeout"),
    }
    assert len(ids) == 4
