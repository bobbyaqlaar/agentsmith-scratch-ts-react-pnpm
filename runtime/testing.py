"""
runtime/testing.py — test doubles for the LLM gateway.

Why this ships with the framework (TestbedFeedback-2026-07-21 G4): the
KYC Sentinel testbed had to write ~60 lines of FakeGateway before it could
test anything, and every tenant would otherwise reinvent it — each one
drifting from CompletionResult's real shape. Worse, the testbed's
hand-rolled double aliased `complete_stream` to `complete`, which MASKED a
real production crash (G1): a double that is more capable than the real
gateway hides exactly the bugs a testbed exists to find.

So these doubles are deliberately *no more capable than the real thing*:

  - `complete_stream()` refuses to stream for providers the real gateway
    can't stream, unless you opt in to the fallback the real one performs.
  - Budget caps, degrade ladders, and moderation blocks are simulated with
    the same observable results the real gateway returns.

Usage:

    from runtime.testing import FakeGateway

    gw = FakeGateway(responses={"analyst": '{"rating": "LOW"}'})
    result = await gw.complete("...", model_hint="analyst")
    assert gw.calls[-1].model_hint == "analyst"

    # scripted per-call sequence (e.g. broken JSON, then a repair)
    gw = FakeGateway(responses={"analyst": ["not json", '{"rating":"LOW"}']})

    # callables get the prompt and may raise to simulate provider errors
    gw = FakeGateway(responses={"judge": lambda prompt: f"seen {len(prompt)}"})

`RecordingGateway` wraps a REAL gateway and records the same call log —
use it in integration tests where you want live calls but still want to
assert on routing/degrade behavior.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Union

from runtime.llm_gateway import BudgetExceededError, CompletionResult
from runtime.prompt_identity import content_text
from runtime.provider_dispatch import supports_streaming

# What a role's response can be: a fixed string, a queue of strings consumed
# one per call, or a callable taking the prompt text.
ResponseSpec = Union[str, list[str], Callable[[str], str]]


@dataclass
class RecordedCall:
    """One gateway invocation, as the caller made it."""

    model_hint: str
    prompt: str
    streamed: bool = False
    max_tokens: int = 4096
    temperature: float = 0.2
    workflow_id: Optional[str] = None
    idempotency_key: Optional[str] = None


@dataclass
class FakeGateway:
    """Deterministic stand-in for LLMGateway.

    Args:
        responses: role -> response spec. Roles with no entry return
            `default_response`.
        default_response: used for unmapped roles.
        providers: role -> provider name, used only to decide whether
            `complete_stream` can stream (defaults to a streaming-capable
            provider). Set e.g. {"analyst": "bedrock"} to exercise the
            non-streaming path.
        stream_fallback: when True (the real gateway's behavior since G1),
            `complete_stream` on a non-streaming provider falls back to
            `complete` with ttft_ms=None. When False it raises
            NotImplementedError, which is what the gateway did before G1 —
            useful for pinning a tenant's own fallback handling.
        cap_usd / cost_per_call: when cap_usd is set, each call accrues
            cost_per_call and BudgetExceededError is raised once the cap
            would be exceeded — enough to test degrade/halt handling
            without a budget backend.
        ttft_ms: value reported on streamed calls.
    """

    responses: dict[str, ResponseSpec] = field(default_factory=dict)
    default_response: str = "ok"
    providers: dict[str, str] = field(default_factory=dict)
    stream_fallback: bool = True
    tenant_id: str = "test-tenant"
    cap_usd: Optional[float] = None
    cost_per_call: float = 0.0
    ttft_ms: float = 1.0
    calls: list[RecordedCall] = field(default_factory=list)
    spent_usd: float = 0.0

    # ── LLMGateway surface ───────────────────────────────────────────────

    async def complete(
        self,
        prompt: Any,
        model_hint: str = "developer",
        workflow_id: Optional[str] = None,
        idempotency_key: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.2,
        **kwargs: Any,
    ) -> CompletionResult:
        del kwargs
        return self._build_result(
            RecordedCall(
                model_hint=model_hint,
                prompt=_as_text(prompt),
                streamed=False,
                max_tokens=max_tokens,
                temperature=temperature,
                workflow_id=workflow_id,
                idempotency_key=idempotency_key,
            )
        )

    async def complete_stream(
        self,
        prompt: Any,
        model_hint: str = "developer",
        workflow_id: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.2,
        **kwargs: Any,
    ) -> CompletionResult:
        del kwargs
        provider = self.providers.get(model_hint, "openai")
        if not supports_streaming(provider):
            if not self.stream_fallback:
                raise NotImplementedError(
                    f"complete_stream does not support provider {provider!r}"
                )
            return await self.complete(
                prompt,
                model_hint=model_hint,
                workflow_id=workflow_id,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        return self._build_result(
            RecordedCall(
                model_hint=model_hint,
                prompt=_as_text(prompt),
                streamed=True,
                max_tokens=max_tokens,
                temperature=temperature,
                workflow_id=workflow_id,
            )
        )

    def get_budget_status(self) -> dict:
        cap = self.cap_usd if self.cap_usd is not None else float("inf")
        return {
            "tenant_id": self.tenant_id,
            "spent_usd": self.spent_usd,
            "cap_usd": cap,
            "remaining_usd": max(0.0, cap - self.spent_usd),
            "breached": self.spent_usd >= cap,
        }

    # ── Assertion helpers ────────────────────────────────────────────────

    def calls_for(self, model_hint: str) -> list[RecordedCall]:
        return [c for c in self.calls if c.model_hint == model_hint]

    def routes_used(self) -> list[str]:
        """Roles in first-use order — asserts a multi-agent app really did
        route across the tiers it claims to."""
        seen: list[str] = []
        for c in self.calls:
            if c.model_hint not in seen:
                seen.append(c.model_hint)
        return seen

    def assert_prompt_excludes(self, *needles: str) -> None:
        """Every prompt the 'model' saw must not contain these strings —
        the direct way to test a PII/redaction claim."""
        for call in self.calls:
            for needle in needles:
                if needle in call.prompt:
                    raise AssertionError(
                        f"{needle!r} reached the model in a {call.model_hint} prompt"
                    )

    def reset(self) -> None:
        self.calls.clear()
        self.spent_usd = 0.0

    # ── internals ────────────────────────────────────────────────────────
    #
    # `_resolve_text(call) -> str` is THE extension point: subclass and
    # override it for domain-specific scripting (see KYC Sentinel's
    # agents/gateway.py). Everything else here — recording, budget
    # simulation, CompletionResult assembly — should keep working
    # unchanged, which is the point of subclassing rather than rewriting.
    # Names here are deliberately specific (`_build_result`, not
    # `_respond`) so a subclass's own helpers don't collide with them.

    def _build_result(self, call: RecordedCall) -> CompletionResult:
        self.calls.append(call)

        if self.cap_usd is not None and self.cost_per_call:
            if self.spent_usd + self.cost_per_call > self.cap_usd:
                raise BudgetExceededError(
                    f"tenant={self.tenant_id} budget exhausted "
                    f"(${self.spent_usd:.2f}/${self.cap_usd:.2f})"
                )
            self.spent_usd += self.cost_per_call

        text = self._resolve_text(call)
        return CompletionResult(
            text=text,
            model_used=f"fake-{call.model_hint}",
            input_tokens=max(1, len(call.prompt) // 4),
            output_tokens=max(1, len(text) // 4),
            cost_usd=self.cost_per_call,
            degrade_tier=None,
            ttft_ms=self.ttft_ms if call.streamed else None,
        )

    def _resolve_text(self, call: RecordedCall) -> str:
        spec = self.responses.get(call.model_hint, self.default_response)
        if callable(spec):
            return spec(call.prompt)
        if isinstance(spec, list):
            if not spec:
                return self.default_response
            # Consume one per call; the last entry repeats so a test can
            # script "fail once, then succeed forever".
            return spec.pop(0) if len(spec) > 1 else spec[0]
        return spec


class RecordingGateway:
    """Wraps a real LLMGateway and records calls in FakeGateway's format.

    For integration tests that hit live providers but still assert on
    routing, degrade tiers, and what text reached the model.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.calls: list[RecordedCall] = []
        self.results: list[CompletionResult] = []

    async def complete(self, prompt: Any, model_hint: str = "developer", **kw: Any):
        self.calls.append(
            RecordedCall(model_hint=model_hint, prompt=_as_text(prompt), streamed=False)
        )
        result = await self._inner.complete(prompt, model_hint=model_hint, **kw)
        self.results.append(result)
        return result

    async def complete_stream(self, prompt: Any, model_hint: str = "developer", **kw: Any):
        self.calls.append(
            RecordedCall(model_hint=model_hint, prompt=_as_text(prompt), streamed=True)
        )
        result = await self._inner.complete_stream(prompt, model_hint=model_hint, **kw)
        self.results.append(result)
        return result

    def degrade_tiers(self) -> list[Optional[str]]:
        return [r.degrade_tier for r in self.results]

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def _as_text(prompt: Any) -> str:
    """Gateways accept a string or a message list; tests assert on text.

    Content flattening is prompt_identity.content_text, which is what the real
    gateway's prompt hashing uses. This joined `m.get("content", "")` straight
    into a str.join, so any message whose content was a LIST of typed parts —
    the ordinary Anthropic and OpenAI shape — raised TypeError out of
    FakeGateway, on a prompt the real gateway handles. `content: None` did the
    same. A test double that refuses inputs the real thing accepts sends the
    tenant looking for a bug in their own code.
    """
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        return "\n".join(
            content_text(m.get("content")) for m in prompt if isinstance(m, dict)
        )
    return json.dumps(prompt, default=str)
