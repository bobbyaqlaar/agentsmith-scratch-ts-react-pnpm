"""
circuit_breaker.py — Dual-tier token velocity + monthly spend circuit breaker.

Tier 1 (burst): 50,000 tokens in any 5-minute rolling window.
Tier 2 (monthly): configurable USD cap (default $150/month).

State is persisted to .agent-rfc/fixtures/token_velocity_cache.json in the
repo root so it survives across shell sessions and agent runs.

To reset manually:
    echo '{"config":{},"monthly_accumulated_spend_usd":0,"current_month_identifier":"","events":[]}' \
      > .agent-rfc/fixtures/token_velocity_cache.json
"""

from __future__ import annotations

import copy
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Moved up from below the config block, where it carried an E402 suppression.
# The constants below now call env_number, so the import has to precede
# them; nothing here imports circuit_breaker back, so the late placement
# was not guarding a cycle.
from _shared import env_number, fixtures_path

# ── Config ────────────────────────────────────────────────────────────────────

BURST_WINDOW_SECONDS = 300  # 5 minutes
# Read through env_number, not int()/float() directly. These are MODULE-LEVEL,
# so a malformed or empty value here is not a bad config value — it is an
# ImportError, raised at `from circuit_breaker import ...` in every consumer.
# Both consumers treat a breaker they cannot import as "proceed unmetered", so
# the failure mode of a stray `AGENT_MONTHLY_USD_CAP=` in a manifest was the
# spend cap quietly ceasing to apply.
BURST_TOKEN_LIMIT = env_number("AGENT_BURST_TOKEN_LIMIT", 50_000, cast=int)
MONTHLY_USD_CAP = env_number("AGENT_MONTHLY_USD_CAP", 150.0)

# Approximate blended cost per token in USD (conservative estimate).
# Override via env for more accurate per-model pricing.
COST_PER_INPUT_TOKEN = env_number("AGENT_COST_PER_INPUT_TOKEN", 0.000003)
COST_PER_OUTPUT_TOKEN = env_number("AGENT_COST_PER_OUTPUT_TOKEN", 0.000015)

# ── State file ────────────────────────────────────────────────────────────────


def _cache_path() -> Path:
    return fixtures_path("token_velocity_cache.json", mkdir=True)


_EMPTY_STATE: dict = {
    "config": {},
    "monthly_accumulated_spend_usd": 0.0,
    "current_month_identifier": "",
    "events": [],  # list of {ts: float, input_tokens: int, output_tokens: int}
}


def _empty_state() -> dict:
    """A fresh empty state, not a view onto the constant.

    `dict(_EMPTY_STATE)` is a SHALLOW copy: the returned dict is new, but
    `state["events"]` is the same list object as `_EMPTY_STATE["events"]`, so
    the first `events.append` mutates the module-level constant and the next
    "empty" state comes back carrying the previous run's events. It matters on
    exactly the path this fallback exists for — a missing or unwritable cache
    file, where `_save_state` fails open and every call re-enters here.
    """
    return copy.deepcopy(_EMPTY_STATE)


def _load_state() -> dict:
    path = _cache_path()
    if not path.exists():
        return _empty_state()
    try:
        with path.open() as fh:
            state = json.load(fh)
        # Ensure all expected keys exist (forward-compatible)
        for k, v in _EMPTY_STATE.items():
            if k not in state:
                state[k] = type(v)()
        return state
    except Exception:
        return _empty_state()


def _save_state(state: dict) -> None:
    try:
        path = _cache_path()
        with path.open("w") as fh:
            json.dump(state, fh, indent=2)
    # fail-open: read-only FS in some CI environments — best effort; does not affect the
    # CircuitBreakerTripped raise path
    except OSError:
        pass


# ── Core audit function ───────────────────────────────────────────────────────


class CircuitBreakerTripped(RuntimeError):
    """Raised when either circuit breaker tier triggers."""

    def __init__(self, tier: str, detail: str) -> None:
        self.tier = tier
        self.detail = detail
        super().__init__(f"[{tier}] {detail}")


def audit_token_velocity_circuit(
    input_tokens: int,
    output_tokens: int,
    *,
    notify: bool = True,
) -> None:
    """
    Record a token usage event and raise CircuitBreakerTripped if either
    tier limit is exceeded.

    Called automatically by AgentLogger.llm_call(); can also be called
    directly by agents that manage their own LLM calls.
    """
    if input_tokens is None or output_tokens is None:
        # A provider that reports no `usage` block gives None since
        # runtime/provider_dispatch.py stopped defaulting it to 0. Reaching the
        # arithmetic below with None raises TypeError, which every caller here
        # wraps in a fail-open `except Exception: pass` — so the call would be
        # unmetered on BOTH tiers and nobody told. Refusing by name gives the
        # caller something it can act on. See scripts/cost_router.py, which
        # does.
        raise ValueError(
            "audit_token_velocity_circuit needs real token counts; got "
            f"input_tokens={input_tokens!r}, output_tokens={output_tokens!r}. "
            "A provider that reported no usage cannot be metered — decide at "
            "the call site whether to estimate or to record the gap, rather "
            "than passing an absence in as if it were a number."
        )

    state = _load_state()
    now = time.time()

    # ── Monthly roll-over ─────────────────────────────────────────────────────
    current_month = datetime.now(timezone.utc).strftime("%Y-%m")
    if state.get("current_month_identifier") != current_month:
        state["current_month_identifier"] = current_month
        state["monthly_accumulated_spend_usd"] = 0.0
        state["events"] = []

    # ── Record this event, and BILL IT, before either tier can raise ──────────
    #
    # The burst check used to sit between the append and the accrual, and it
    # raises. So the tokens of a burst-tripping call were recorded and its
    # DOLLARS were not: the monthly accumulator silently skipped every call
    # that tripped tier 1 — the heaviest bursts, the ones the cap most needs to
    # see. The money was spent either way; the provider had already been
    # called by the time this function runs.
    #
    # Both tiers now measure the same event. Tripping is a decision about what
    # to do next, never a reason to un-record what already happened.
    event = {
        "ts": now,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }
    state["events"].append(event)

    this_cost = (
        input_tokens * COST_PER_INPUT_TOKEN + output_tokens * COST_PER_OUTPUT_TOKEN
    )
    state["monthly_accumulated_spend_usd"] = (
        state.get("monthly_accumulated_spend_usd", 0.0) + this_cost
    )

    cutoff = now - BURST_WINDOW_SECONDS
    window_events = [e for e in state["events"] if e["ts"] >= cutoff]
    window_tokens = sum(e["input_tokens"] + e["output_tokens"] for e in window_events)

    # Prune old events (only keep last 24 h for storage efficiency). After the
    # window sum above, which reads a 5-minute slice and is unaffected.
    state["events"] = [e for e in state["events"] if e["ts"] >= now - 86400]

    monthly_total = state["monthly_accumulated_spend_usd"]
    _save_state(state)

    # ── Tier 1: burst (5-min rolling window) ──────────────────────────────────
    if window_tokens > BURST_TOKEN_LIMIT:
        msg = (
            f"Burst limit exceeded: {window_tokens:,} tokens in last 5 minutes "
            f"(limit: {BURST_TOKEN_LIMIT:,}). Cooling down."
        )
        _notify_if_requested(notify, "BURST", msg)
        raise CircuitBreakerTripped("BURST", msg)

    # ── Tier 2: monthly spend ─────────────────────────────────────────────────
    if monthly_total > MONTHLY_USD_CAP:
        msg = (
            f"Monthly spend cap exceeded: ${monthly_total:.4f} "
            f"(limit: ${MONTHLY_USD_CAP:.2f}). "
            f"Reset: echo '{{...}}' > .agent-rfc/fixtures/token_velocity_cache.json"
        )
        _notify_if_requested(notify, "MONTHLY", msg)
        raise CircuitBreakerTripped("MONTHLY", msg)

    # ── Warn at 80 % monthly ──────────────────────────────────────────────────
    if monthly_total > MONTHLY_USD_CAP * 0.8:
        pct = (monthly_total / MONTHLY_USD_CAP) * 100
        warn_msg = f"Monthly spend at {pct:.0f}%: ${monthly_total:.4f} / ${MONTHLY_USD_CAP:.2f}"
        _notify_if_requested(notify, "WARN", warn_msg)


def _notify_if_requested(notify: bool, tier: str, msg: str) -> None:
    if not notify:
        return
    try:
        from notifier import send_notification

        title = "🚨 Circuit Breaker" if tier != "WARN" else "⚠️ Budget Warning"
        send_notification(
            title, msg, urgency="critical" if tier != "WARN" else "normal"
        )
    except Exception:
        print(f"[circuit_breaker] {tier}: {msg}", file=sys.stderr)


# ── Status query ──────────────────────────────────────────────────────────────


def get_status() -> dict:
    """Return current circuit breaker state as a plain dict."""
    state = _load_state()
    now = time.time()
    cutoff = now - BURST_WINDOW_SECONDS
    window_events = [e for e in state.get("events", []) if e["ts"] >= cutoff]
    window_tokens = sum(e["input_tokens"] + e["output_tokens"] for e in window_events)
    monthly = state.get("monthly_accumulated_spend_usd", 0.0)
    return {
        "burst_tokens_5min": window_tokens,
        "burst_limit": BURST_TOKEN_LIMIT,
        "burst_headroom": max(0, BURST_TOKEN_LIMIT - window_tokens),
        "monthly_spend_usd": round(monthly, 4),
        "monthly_cap_usd": MONTHLY_USD_CAP,
        "monthly_headroom_usd": round(max(0.0, MONTHLY_USD_CAP - monthly), 4),
        "current_month": state.get("current_month_identifier", ""),
    }


def reset_monthly() -> None:
    """Reset monthly accumulator (manual override, requires user confirmation)."""
    state = _load_state()
    state["monthly_accumulated_spend_usd"] = 0.0
    state["current_month_identifier"] = ""
    state["events"] = []
    _save_state(state)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Circuit breaker status and controls")
    parser.add_argument("--status", action="store_true", help="Print current status")
    parser.add_argument(
        "--reset", action="store_true", help="Reset monthly accumulator"
    )
    parser.add_argument(
        "--simulate",
        nargs=2,
        metavar=("INPUT", "OUTPUT"),
        help="Simulate a call: --simulate 1000 500",
    )
    args = parser.parse_args()

    if args.status:
        print(json.dumps(get_status(), indent=2))

    elif args.reset:
        confirm = input("Reset monthly circuit breaker state? (y/n): ")
        if confirm.lower() == "y":
            reset_monthly()
            print("✅ Monthly circuit breaker reset.")
        else:
            print("Cancelled.")

    elif args.simulate:
        try:
            audit_token_velocity_circuit(int(args.simulate[0]), int(args.simulate[1]))
            print(json.dumps(get_status(), indent=2))
        except CircuitBreakerTripped as e:
            print(f"🔴 Circuit tripped [{e.tier}]: {e.detail}")
            sys.exit(1)
    else:
        print(json.dumps(get_status(), indent=2))
