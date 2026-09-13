"""
runtime/test/test_llm_gateway_budget.py — regression test for the budget
reservation race (Product_Archive.md 2.1).

Uses the in-memory backend (no external infra) for the concurrency
assertion — deterministic and fast enough to run on every PR. A separate
manual check (`scripts/verify_system.py --check-idempotency` / `--check-dlq`)
exercises the Postgres backends against a throwaway database, since CI
doesn't have one available by default for this job.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from runtime.llm_gateway import BudgetExceededError, LLMGateway


def _make_gateway(
    cap_usd: float, output_cost_per_token: float, input_cost_per_token: float = 0.01
) -> LLMGateway:
    gw = LLMGateway(tenant_id="acme", budget_cap_usd=cap_usd)
    gw.models = {
        "developer": {
            "id": "test-model",
            "cost_per_input_token": input_cost_per_token,
            "cost_per_output_token": output_cost_per_token,
            "degrade_to": None,
        }
    }
    return gw


def test_the_memory_backend_resets_at_a_month_boundary(monkeypatch):
    """The DEFAULT backend keyed spend by tenant alone, so a monthly cap never
    reset.

    Redis keys `budget:{tenant}:{period}` and Postgres has
    `PRIMARY KEY (tenant_id, period)`; this one accumulated for the life of the
    process. A worker alive across the 1st carried the previous month's spend
    into the new one and eventually refused every call against a cap that
    should have been empty — while `get_budget_status()` reported that lifetime
    figure beside a `period_start` naming the current month.
    """
    from runtime import llm_gateway

    backend = llm_gateway._MemoryBudgetBackend()

    monkeypatch.setattr(llm_gateway, "_current_period", lambda: "2026-08")
    assert backend.try_reserve("acme", 9.0, cap_usd=10.0) is True
    assert backend.get_spend("acme") == 9.0
    # The cap is nearly gone in August...
    assert backend.try_reserve("acme", 5.0, cap_usd=10.0) is False

    monkeypatch.setattr(llm_gateway, "_current_period", lambda: "2026-09")
    # ...and untouched in September.
    assert backend.get_spend("acme") == 0.0
    assert backend.try_reserve("acme", 9.0, cap_usd=10.0) is True

    # August's figure is still August's — the reset is a new key, not a wipe.
    monkeypatch.setattr(llm_gateway, "_current_period", lambda: "2026-08")
    assert backend.get_spend("acme") == 9.0


def test_the_memory_backend_keeps_tenants_apart(monkeypatch):
    """Guard on the guard: if the key were the period alone, every assertion
    above would still pass and every tenant would share one budget."""
    from runtime import llm_gateway

    monkeypatch.setattr(llm_gateway, "_current_period", lambda: "2026-08")
    backend = llm_gateway._MemoryBudgetBackend()
    backend.add_spend("acme", 4.0)
    assert backend.get_spend("globex") == 0.0


@pytest.mark.asyncio
async def test_concurrent_calls_cannot_exceed_cap():
    """N concurrent complete() calls for the same tenant must never let
    combined spend exceed the cap — the old check-then-act pattern allowed
    exactly this."""
    gw = _make_gateway(cap_usd=1.0, output_cost_per_token=0.5)

    async def fake_invoke(cfg, messages, max_tokens, temperature):
        await asyncio.sleep(0.02)
        return "ok", 0, 1  # 1 output token -> $0.5 actual cost at this pricing

    gw._invoke = fake_invoke

    async def run_one():
        try:
            return await gw.complete(prompt="hi", model_hint="developer", max_tokens=1)
        except BudgetExceededError as exc:
            return exc

    results = await asyncio.gather(*[run_one() for _ in range(10)])
    succeeded = [r for r in results if not isinstance(r, BudgetExceededError)]
    final_spend = gw._budget.get_spend("acme")

    # Reservation is conservative (estimated_cost includes input cost too),
    # so the exact number that succeed depends on pricing — the invariant
    # under test is that spend never exceeds the cap, not a specific count.
    assert final_spend <= gw.budget_cap_usd + 1e-9, (
        f"budget overshoot: spent ${final_spend} against cap ${gw.budget_cap_usd} "
        f"with {len(succeeded)} succeeding calls — the reservation race regressed"
    )
    assert len(succeeded) >= 1, "at least one call should succeed under a non-zero cap"


@pytest.mark.asyncio
async def test_reservation_releases_on_invoke_failure():
    """If the provider call raises, the reservation must be released —
    otherwise a transient provider error would permanently burn budget."""
    gw = _make_gateway(cap_usd=1.0, output_cost_per_token=0.5)

    async def failing_invoke(cfg, messages, max_tokens, temperature):
        raise RuntimeError("simulated provider failure")

    gw._invoke = failing_invoke

    with pytest.raises(RuntimeError):
        await gw.complete(prompt="hi", model_hint="developer", max_tokens=1)

    assert gw._budget.get_spend("acme") == 0.0, (
        "reservation was not released after invoke() raised"
    )


@pytest.mark.asyncio
async def test_free_tier_model_bypasses_reservation():
    """A free/local-tier model (cost_per_input_token == 0) must never be
    blocked by budget — see LLMGateway._is_free_tier."""
    gw = _make_gateway(cap_usd=0.0, output_cost_per_token=0.0, input_cost_per_token=0.0)

    async def fake_invoke(cfg, messages, max_tokens, temperature):
        return "ok", 5, 5

    gw._invoke = fake_invoke

    result = await gw.complete(prompt="hi", model_hint="developer", max_tokens=10)
    assert result.text == "ok"
    assert gw._budget.get_spend("acme") == 0.0


@pytest.mark.asyncio
async def test_reservation_reconciles_to_actual_cost():
    """The conservative max_tokens-based estimate must be replaced by the
    real cost after the call, not left as the (larger) estimate."""
    # cap must accommodate the conservative max_tokens-based reservation
    # (100 * (0.1 + 0.1) = 20) even though the actual cost will be far less.
    gw = _make_gateway(
        cap_usd=30.0, output_cost_per_token=0.1, input_cost_per_token=0.1
    )

    async def fake_invoke(cfg, messages, max_tokens, temperature):
        return "ok", 1, 1  # actual cost: 1*0.1 + 1*0.1 = 0.2

    gw._invoke = fake_invoke

    await gw.complete(
        prompt="hi", model_hint="developer", max_tokens=100
    )  # estimate: 100*0.2 = 20 (would exceed cap if not reconciled down)

    spend = gw._budget.get_spend("acme")
    assert abs(spend - 0.2) < 1e-9, (
        f"expected spend reconciled to actual cost 0.2, got {spend}"
    )


@pytest.mark.asyncio
async def test_unreported_usage_bills_the_reservation_not_zero():
    """A provider that omits `usage` must not come out free.

    `parse_response` used to default a missing usage block to 0/0, so
    `cost_usd = 0 * rate + 0 * rate` was exactly $0.00 and the reconcile
    released the whole reservation. An OpenAI-compatible proxy, a shim, or a
    stream without `stream_options.include_usage` charged nothing against the
    monthly cap. complete_stream() already handled this — it keeps the
    try_reserve() amount and flags `cost_estimated` — and complete() did not.
    """
    gw = _make_gateway(cap_usd=30.0, output_cost_per_token=0.1, input_cost_per_token=0.1)

    async def fake_invoke(cfg, messages, max_tokens, temperature):
        return "ok", None, None  # the provider reported nothing

    gw._invoke = fake_invoke

    result = await gw.complete(prompt="hi", model_hint="developer", max_tokens=100)

    # The reservation was 100 * (0.1 + 0.1) = 20, and it stands.
    spend = gw._budget.get_spend("acme")
    assert abs(spend - 20.0) < 1e-9, f"expected the reservation to stand, spend was {spend}"
    assert abs(result.cost_usd - 20.0) < 1e-9
    # And the absence travels: None, not 0, so a consumer summing tokens sees
    # a gap rather than a confident zero.
    assert result.input_tokens is None
    assert result.output_tokens is None


@pytest.mark.asyncio
async def test_a_genuine_zero_is_still_a_zero():
    """The other half of the distinction. A provider that reports 0/0 —
    a cached completion, an empty response it still counted — really did cost
    nothing, and the reservation must be released. If both cases behaved the
    same, the fix above would just be the old bug with new wording."""
    gw = _make_gateway(cap_usd=30.0, output_cost_per_token=0.1, input_cost_per_token=0.1)

    async def fake_invoke(cfg, messages, max_tokens, temperature):
        return "ok", 0, 0  # counted, and the count is zero

    gw._invoke = fake_invoke

    result = await gw.complete(prompt="hi", model_hint="developer", max_tokens=100)

    assert abs(gw._budget.get_spend("acme")) < 1e-9, "a measured zero must release the reservation"
    assert result.cost_usd == 0.0
    assert result.input_tokens == 0
    assert result.output_tokens == 0


@pytest.mark.asyncio
async def test_invoke_retries_transient_errors_with_backoff():
    """_invoke() retries httpx.TransportError/429/5xx (the documented but
    previously-unimplemented "Throttle: exponential backoff" degrade-ladder
    step) and gives up after 3 attempts — tenacity was a required
    dependency from the start for exactly this, but nothing called it
    until this test's corresponding fix landed."""
    import httpx
    from unittest.mock import patch

    gw = _make_gateway(
        cap_usd=1000.0, output_cost_per_token=0.0, input_cost_per_token=0.0
    )
    cfg = gw.models["developer"]
    call_count = {"n": 0}

    async def fake_post(self, url, json=None, headers=None):
        call_count["n"] += 1
        request = httpx.Request("POST", url)
        if call_count["n"] < 3:
            response = httpx.Response(503, request=request, text="Service Unavailable")
            raise httpx.HTTPStatusError("503", request=request, response=response)
        return httpx.Response(
            200,
            request=request,
            json={
                "choices": [{"message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )

    with patch.object(httpx.AsyncClient, "post", fake_post):
        text, _in_tok, _out_tok = await gw._invoke(
            cfg, [{"role": "user", "content": "hi"}], 10, 0.2
        )

    assert call_count["n"] == 3, (
        f"expected 2 failed attempts + 1 success, got {call_count['n']} attempts"
    )
    assert text == "ok"


@pytest.mark.asyncio
async def test_invoke_does_not_retry_non_transient_errors():
    """A 401 (bad credentials) or any other non-429/5xx error fails on the
    first attempt — retrying it would waste the attempt budget on a
    failure that can't succeed differently the second time."""
    import httpx
    from unittest.mock import patch

    gw = _make_gateway(
        cap_usd=1000.0, output_cost_per_token=0.0, input_cost_per_token=0.0
    )
    cfg = gw.models["developer"]
    call_count = {"n": 0}

    async def fake_post_401(self, url, json=None, headers=None):
        call_count["n"] += 1
        request = httpx.Request("POST", url)
        response = httpx.Response(401, request=request, text="invalid api key")
        raise httpx.HTTPStatusError("401", request=request, response=response)

    with (
        patch.object(httpx.AsyncClient, "post", fake_post_401),
        pytest.raises(httpx.HTTPStatusError) as exc_info,
    ):
        await gw._invoke(cfg, [{"role": "user", "content": "hi"}], 10, 0.2)

    assert exc_info.value.response.status_code == 401
    assert call_count["n"] == 1, (
        f"expected exactly 1 attempt (no retry on 401), got {call_count['n']}"
    )


@pytest.mark.asyncio
async def test_invoke_groq_provider_uses_groq_base_url_and_key():
    """provider: groq resolves to Groq's OpenAI-compatible endpoint and
    GROQ_API_KEY by default — not the generic OpenAI fallback's
    api.openai.com/OPENAI_API_KEY, which would silently send Groq-shaped
    requests to the wrong host with the wrong key."""
    import httpx
    from unittest.mock import patch

    gw = _make_gateway(
        cap_usd=1000.0, output_cost_per_token=0.0, input_cost_per_token=0.0
    )
    cfg = {
        "id": "llama-3.3-70b-versatile",
        "provider": "groq",
        "cost_per_input_token": 0,
        "cost_per_output_token": 0,
    }
    seen = {}

    async def fake_post(self, url, json=None, headers=None):
        seen["url"] = url
        seen["headers"] = headers
        request = httpx.Request("POST", url)
        return httpx.Response(
            200,
            request=request,
            json={
                "choices": [{"message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )

    with (
        patch.dict("os.environ", {"GROQ_API_KEY": "gsk_test_key"}),
        patch.object(httpx.AsyncClient, "post", fake_post),
    ):
        text, _, _ = await gw._invoke(cfg, [{"role": "user", "content": "hi"}], 10, 0.2)

    assert text == "ok"
    assert seen["url"] == "https://api.groq.com/openai/v1/chat/completions"
    assert seen["headers"]["Authorization"] == "Bearer gsk_test_key"
