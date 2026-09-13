from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from runtime.self_correction import propose_corrected_payload, run_self_correction_loop  # type: ignore


class FakeCompletion:
    def __init__(self, text: str) -> None:
        self.text = text


class FakeGateway:
    def __init__(self, text: str) -> None:
        self.text = text
        self.prompts: list[Any] = []

    async def complete(self, *, prompt: Any, model_hint: str) -> FakeCompletion:
        self.prompts.append({"prompt": prompt, "model_hint": model_hint})
        return FakeCompletion(self.text)


@pytest.mark.asyncio
async def test_propose_corrected_payload_parses_json_from_fenced_response() -> None:
    gateway = FakeGateway(
        '```json\n{"customer_id": 102, "status": "active"}\n```'
    )

    result = await propose_corrected_payload(
        gateway,
        {"customer_id": 102, "account_status": "active"},
        "account_status is not a valid property",
    )

    assert result == {"customer_id": 102, "status": "active"}
    assert gateway.prompts[0]["model_hint"] == "developer"


@pytest.mark.asyncio
async def test_run_self_correction_loop_succeeds_after_one_correction() -> None:
    gateway = FakeGateway('{"customer_id": 102, "status": "active"}')
    attempts: list[dict[str, Any]] = []

    async def activity_fn(payload: dict[str, Any]) -> dict[str, Any]:
        attempts.append(payload)
        if "status" not in payload:
            raise ValueError("account_status is not a valid property")
        return {"ok": True, "applied": payload}

    result = await run_self_correction_loop(
        activity_fn=activity_fn,
        payload={"customer_id": 102, "account_status": "active"},
        gateway=gateway,
    )

    assert result == {"ok": True, "applied": {"customer_id": 102, "status": "active"}}
    assert attempts == [
        {"customer_id": 102, "account_status": "active"},
        {"customer_id": 102, "status": "active"},
    ]


@pytest.mark.asyncio
async def test_run_self_correction_loop_returns_sentinel_when_exhausted() -> None:
    gateway = FakeGateway('{"customer_id": 102, "account_status": "active"}')

    async def activity_fn(payload: dict[str, Any]) -> dict[str, Any]:
        raise ValueError("account_status is not a valid property")

    result = await run_self_correction_loop(
        activity_fn=activity_fn,
        payload={"customer_id": 102, "account_status": "active"},
        gateway=gateway,
        max_self_correction_attempts=1,
    )

    assert result == {
        "__self_correction_exhausted__": True,
        "payload": {"customer_id": 102, "account_status": "active"},
        "error": "account_status is not a valid property",
    }


# ── The corrector's own failure must not skip the fallback ───────────────────


def test_a_corrector_that_answers_in_prose_exhausts_rather_than_raising():
    """`propose_corrected_payload` ends in `json.loads`, so a model that
    answers in prose — the ordinary failure of "return ONLY JSON" — raised
    JSONDecodeError straight out of the loop.

    That skipped the `__self_correction_exhausted__` result, which is the whole
    reason a caller gets a structured "could not fix it" instead of an
    exception. In the Temporal twin the skipped fallback is the human DLQ path,
    so the most likely failure of the automatic fixer was the one that stopped
    an application ever reaching a person.
    """
    class Gateway:
        async def complete(self, prompt, model_hint=None):
            class R:
                text = "I'm sorry, I can't produce JSON for that."
            return R()

    async def activity(_payload):
        raise ValueError("schema mismatch: 'amount' must be a number")

    out = asyncio.run(
        run_self_correction_loop(
            activity_fn=activity, payload={"amount": "ten"}, gateway=Gateway()
        )
    )
    assert out["__self_correction_exhausted__"] is True
    assert "self-correction failed to produce a usable payload" in out["error"]


def test_the_payload_handed_on_is_the_last_real_one():
    """Not the wreckage of a correction that did not parse. Whatever reaches
    the fallback is what a human is asked to fix."""
    class Gateway:
        async def complete(self, prompt, model_hint=None):
            class R:
                text = "not json"
            return R()

    async def activity(_payload):
        raise ValueError("nope")

    original = {"amount": "ten"}
    out = asyncio.run(
        run_self_correction_loop(
            activity_fn=activity, payload=original, gateway=Gateway()
        )
    )
    assert out["payload"] == original


def test_a_working_corrector_is_unaffected():
    """The control: the guard must not swallow a correction that worked."""
    class Gateway:
        async def complete(self, prompt, model_hint=None):
            class R:
                text = '{"amount": 10}'
            return R()

    seen = []

    async def activity(payload):
        seen.append(payload)
        if payload.get("amount") != 10:
            raise ValueError("must be a number")
        return {"ok": True}

    out = asyncio.run(
        run_self_correction_loop(
            activity_fn=activity, payload={"amount": "ten"}, gateway=Gateway()
        )
    )
    assert out == {"ok": True}
    assert seen[-1] == {"amount": 10}
