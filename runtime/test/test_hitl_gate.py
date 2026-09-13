"""
runtime/test/test_hitl_gate.py — regression tests for
BaseAgentWorkflow.run_with_hitl_gate.

These drive the method directly with a fake `workflow` module object rather
than a real Temporal test server, so they run everywhere — the Temporal-backed
tests in this directory (test_recoverable_step.py) skip without DATABASE_URL,
and a HITL bypass is not something to leave covered only by a suite that most
runs skip.

What they lock in: the gate must never run the resume activity — the
high-impact action — without either a `needs_hitl` decision that says it isn't
needed, or a recorded human approval. The bug that motivated this: KYC Sentinel
passed the Analyst activity as the gate activity when it had ALREADY run it, so
the gate re-executed a temperature=0.1 frontier call and honoured the second
run's needs_hitl. A re-run that came back unflagged approved the applicant with
no human in the loop at all.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from runtime.workflows import base_workflow
from runtime.workflows.base_workflow import AgentWorkflowResult, BaseAgentWorkflow


class _FakeWorkflowModule:
    """Stand-in for `temporalio.workflow` inside base_workflow's namespace."""

    def __init__(self, approve: bool | None = None, timeout: bool = False) -> None:
        self.executed: list[tuple[str, Any]] = []
        self._approve = approve
        self._timeout = timeout

    async def execute_activity(self, name, payload, **_kwargs):
        self.executed.append((name, payload))
        return {"ran": name}

    async def wait_condition(self, predicate, timeout=None):
        """Honours the predicate, which the first version of this did not.

        It returned `predicate()` unconditionally, so a gate whose condition was
        FALSE carried on into the decision branch exactly as if it had been
        approved — the double was more permissive than `workflow.wait_condition`,
        which blocks until the predicate holds and otherwise raises. No test
        here could tell "an approval arrived" from "no approval arrived and the
        gate resumed anyway", which is the one distinction a HITL gate exists
        to make.

        Modelled synchronously: true now means the signal is already queued
        (normal in Temporal — signals are applied before the wait is reached);
        false means nothing will arrive, which in a real workflow is the
        timeout.
        """
        if self._timeout or not predicate():
            raise TimeoutError("no signal within HITL_SIGNAL_TIMEOUT")
        return True

    def info(self):
        class _Info:
            workflow_id = "wf-test-1"
            # `run_id` as well. A double that carries only the fields the code
            # happened to use when it was written breaks the next change and
            # points nowhere near the cause — the same way this module's
            # `wait_condition` stand-in used to ignore its predicate.
            run_id = "run-test-1"

        return _Info()

    @staticmethod
    def signal(fn=None, **_kwargs):
        return fn if fn is not None else (lambda f: f)


@pytest.fixture()
def fake_workflow(monkeypatch):
    def _install(**kwargs) -> _FakeWorkflowModule:
        fake = _FakeWorkflowModule(**kwargs)
        monkeypatch.setattr(base_workflow, "workflow", fake)
        monkeypatch.setattr(base_workflow, "_HAS_TEMPORAL", True)
        return fake

    return _install


def _gate(wf: BaseAgentWorkflow, **overrides):
    kwargs: dict[str, Any] = {
        "gate_activity_name": None,
        "gate_input": {"profile": "p"},
        "resume_activity_name": "approve_activity",
        "resume_input": {"assessment": "HIGH"},
        "dead_letter_activity_name": "dlq_enqueue_activity",
        "gate_result": {"needs_hitl": True},
    }
    kwargs.update(overrides)
    positional = (
        kwargs.pop("gate_activity_name"),
        kwargs.pop("gate_input"),
        kwargs.pop("resume_activity_name"),
        kwargs.pop("resume_input"),
        kwargs.pop("dead_letter_activity_name"),
    )
    return asyncio.run(wf.run_with_hitl_gate(*positional, **kwargs))


def test_supplied_gate_result_does_not_re_run_any_gate_activity(fake_workflow):
    """The whole point: a caller that already has needs_hitl pays for no
    second execution, so no second execution can disagree with the first."""
    fake = fake_workflow()
    wf = BaseAgentWorkflow()
    wf._hitl_approved = True

    _gate(wf, gate_result={"needs_hitl": True})

    assert [name for name, _ in fake.executed] == ["approve_activity"]


def test_high_risk_gate_result_waits_for_approval_before_resuming(fake_workflow):
    fake = fake_workflow()
    wf = BaseAgentWorkflow()
    wf._hitl_approved = True

    result = _gate(wf, gate_result={"needs_hitl": True})

    assert result == {"ran": "approve_activity"}
    assert fake.executed == [("approve_activity", {"assessment": "HIGH"})]


def test_rejection_never_runs_the_high_impact_action(fake_workflow):
    fake = fake_workflow()
    wf = BaseAgentWorkflow()
    wf._hitl_approved = False

    result = _gate(wf, gate_result={"needs_hitl": True})

    assert result == AgentWorkflowResult(status="failed")
    assert fake.executed == []


def test_gate_result_without_hitl_resumes_directly(fake_workflow):
    fake = fake_workflow()
    wf = BaseAgentWorkflow()

    result = _gate(wf, gate_result={"needs_hitl": False})

    assert result == {"ran": "approve_activity"}
    assert fake.executed == [("approve_activity", {"assessment": "HIGH"})]


def test_gate_activity_name_still_runs_the_gate(fake_workflow):
    """Backward compatibility: the original shape keeps working for callers
    whose gate really is a separate, not-yet-run check."""
    fake = fake_workflow()
    wf = BaseAgentWorkflow()

    result = _gate(wf, gate_activity_name="check_activity", gate_result=None)

    # {"ran": ...} has no needs_hitl, so the gate resumes without pausing.
    assert result == {"ran": "approve_activity"}
    assert [name for name, _ in fake.executed] == ["check_activity", "approve_activity"]


def test_rejects_both_gate_activity_and_gate_result(fake_workflow):
    fake_workflow()
    wf = BaseAgentWorkflow()

    with pytest.raises(ValueError, match="both"):
        _gate(wf, gate_activity_name="check_activity", gate_result={"needs_hitl": True})


def test_rejects_neither_gate_activity_nor_gate_result(fake_workflow):
    fake_workflow()
    wf = BaseAgentWorkflow()

    with pytest.raises(ValueError, match="neither"):
        _gate(wf, gate_activity_name=None, gate_result=None)


def test_timeout_with_tenant_id_emits_the_generic_dlq_envelope(fake_workflow):
    """dlq_enqueue_activity reads payload/error/tenant_id off its input; the
    legacy flattened shape carries none of them and raised KeyError, losing
    the very application the timeout path exists to park."""
    fake = fake_workflow(timeout=True)
    wf = BaseAgentWorkflow()

    result = _gate(wf, gate_result={"needs_hitl": True}, tenant_id="acme", gate_id="g1")

    assert result == AgentWorkflowResult(status="dead_letter")
    name, payload = fake.executed[0]
    assert name == "dlq_enqueue_activity"
    assert payload["payload"] == {"profile": "p"}
    assert payload["error"] == "hitl_timeout"
    assert payload["tenant_id"] == "acme"
    assert payload["gate_id"] == "g1"


def test_timeout_without_tenant_id_keeps_the_legacy_flattened_shape(fake_workflow):
    """examples/oil-price-agent's own dead_letter_activity expects the
    payload's fields inline — that contract must not change under it."""
    fake = fake_workflow(timeout=True)
    wf = BaseAgentWorkflow()

    _gate(wf, gate_result={"needs_hitl": True})

    assert fake.executed[0] == (
        "dlq_enqueue_activity",
        {"profile": "p", "error": "hitl_timeout"},
    )

# The dead-letter envelope's own contract moved to test_dead_letter.py: it is
# SEC-DLQ-001's evidence, and a control cannot be proven by another control's
# suite. What stays here is the HITL gate's use of it — which envelope shape a
# timeout emits — because that is the gate's behaviour, not the DLQ's.


# ── One approval answers one gate (pass 12) ──────────────────────────────────
#
# `_hitl_approved` was a single field that nothing reset, so in a workflow with
# two HITL gates — the shape this class exists to support — the second gate's
# `wait_condition(lambda: self._hitl_approved is not None)` was already true and
# it resumed without anyone approving it. A silent HITL bypass on a high-impact
# action, which is exactly what run_with_hitl_gate's own docstring warns about
# for a different reason.
#
# `_gate_fixes` one method below has been keyed by gate_id from the start, with
# a comment explaining why. Approvals were the sibling that did not get it.


def test_an_approval_does_not_carry_over_to_the_next_gate(fake_workflow):
    fake = fake_workflow()
    wf = BaseAgentWorkflow()
    wf._hitl_approved = True

    first = _gate(wf, gate_result={"needs_hitl": True}, gate_id="gate-one")
    assert first == {"ran": "approve_activity"}

    # The second gate has had no approval of its own. It must wait, and time
    # out into the dead-letter path — not resume on the first gate's answer.
    second = _gate(wf, gate_result={"needs_hitl": True}, gate_id="gate-two")
    assert second == AgentWorkflowResult(status="dead_letter"), (
        "the second gate resumed on the first gate's approval"
    )
    assert [name for name, _ in fake.executed].count("approve_activity") == 1


def test_a_rejection_does_not_carry_over_either(fake_workflow):
    """The other direction: a rejected gate must not silently reject the next
    one, which would look like a workflow failing for no stated reason."""
    fake_workflow()
    wf = BaseAgentWorkflow()
    wf._hitl_approved = False

    assert _gate(wf, gate_result={"needs_hitl": True}, gate_id="gate-one") == (
        AgentWorkflowResult(status="failed")
    )
    assert _gate(wf, gate_result={"needs_hitl": True}, gate_id="gate-two") == (
        AgentWorkflowResult(status="dead_letter")
    )


def test_an_addressed_approval_answers_only_its_own_gate(fake_workflow):
    """`hitl_approved_for(gate_id, approved)` is the form a sender that knows
    which gate it means should use — with two gates waiting, an unaddressed
    approval cannot say."""
    fake = fake_workflow()
    wf = BaseAgentWorkflow()
    wf._gate_approvals["gate-two"] = True

    assert _gate(wf, gate_result={"needs_hitl": True}, gate_id="gate-one") == (
        AgentWorkflowResult(status="dead_letter")
    )
    assert _gate(wf, gate_result={"needs_hitl": True}, gate_id="gate-two") == (
        {"ran": "approve_activity"}
    )
    assert [name for name, _ in fake.executed].count("approve_activity") == 1


# ── No workflow hand-rolls the approval wait ─────────────────────────────────


def test_no_workflow_waits_on_the_approval_field_directly() -> None:
    """Every HITL wait must go through `await_hitl_approval`.

    `examples/oil-price-agent` had its own `wait_condition` on
    `self._hitl_approved is not None` followed by its own read of that field —
    a copy of the base class's control flow that carried the defect the copy
    existed to have: read, never consumed, so one approval satisfied every
    later gate. It has one gate, so nothing broke there; it is also the file a
    tenant pastes into their own repo, where a second gate is ordinary.

    A grep, because the alternative is noticing the next copy by hand.
    """
    root = Path(__file__).resolve().parents[2]
    sources, minimum = _approval_wait_sources(root)
    assert len(sources) >= minimum, f"the sweep found almost nothing to read: {sources}"

    offenders = _approval_wait_offenders(root, sources)
    assert not offenders, (
        "these wait on the approval field directly instead of calling "
        "await_hitl_approval, which consumes it:\n  " + "\n  ".join(offenders)
    )


# Never a tenant's own workflow code: dependency trees, and this runtime's tests
# (which quote the forbidden pattern on purpose).
_SWEEP_SKIP = {"__pycache__", ".git", ".venv", "venv", "node_modules", "site-packages", "test", "tests"}


def _approval_wait_sources(root: Path) -> tuple[list[Path], int]:
    """What the sweep reads, and how many files it must find to mean anything.

    In the framework checkout: `examples/` — the reference implementation a
    tenant copies — and the runtime's own workflows, at least 5 files.

    In a TENANT, where runtime/ is vendored and `examples/` never is: the
    tenant's own Python, which is where a pasted copy of the example actually
    lives. The sweep used to read `examples/` unconditionally, found only
    runtime/workflows' 2 files in AqlaarTeleologyStudio, and failed SEC-HITL-001
    on its own sanity guard. The guard there is the vendored workflows
    themselves — always present, so an empty sweep still fails.
    """
    workflows = [p for p in root.glob("runtime/workflows/*.py") if "__pycache__" not in p.parts]
    if (root / "examples").is_dir():
        examples = [p for p in root.glob("examples/**/*.py") if "__pycache__" not in p.parts]
        return [*examples, *workflows], 5
    tenant = [
        p for p in root.rglob("*.py")
        if not _SWEEP_SKIP.intersection(p.relative_to(root).parts)
        and p.relative_to(root).parts[0] != "scripts"
    ]
    return tenant, max(1, len(workflows))


def _approval_wait_offenders(root: Path, sources: list[Path]) -> list[str]:
    offenders = []
    for path in sources:
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "wait_condition" not in text:
            continue
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("#") or stripped.startswith("*"):
                continue  # prose explaining the rule is not a breach of it
            if "_hitl_approved" in stripped and "lambda" in stripped:
                offenders.append(f"{path.relative_to(root)}: {stripped}")
    return offenders


def test_the_sweep_reads_a_vendored_tenants_own_workflows(tmp_path) -> None:
    """A tenant with runtime/ vendored and no examples/: the sweep must read
    the tenant's own code (and catch a hand-rolled wait there), not fail for
    want of the framework's examples."""
    (tmp_path / "runtime" / "workflows").mkdir(parents=True)
    (tmp_path / "runtime" / "workflows" / "base_workflow.py").write_text("# base\n")
    (tmp_path / "workflows").mkdir()
    pasted = tmp_path / "workflows" / "kyc.py"
    pasted.write_text(
        "await workflow.wait_condition(lambda: self._hitl_approved is not None)\n"
    )

    sources, minimum = _approval_wait_sources(tmp_path)

    assert pasted in sources
    assert len(sources) >= minimum
    assert _approval_wait_offenders(tmp_path, sources) == [
        "workflows/kyc.py: await workflow.wait_condition(lambda: self._hitl_approved is not None)"
    ]
