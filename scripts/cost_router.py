"""
cost_router.py — Prompt complexity analyser → cheapest capable model selector.

Routing logic:
  1. Token count (via tiktoken)
  2. Semantic keyword analysis
  3. Network availability (via network_watchdog)

Route table — each tier resolves in this order:
  1. its AGENT_MODEL_* env var (per-run override, no code change)
  2. the matching role in models.yaml (framework ← tenant ← routing_overrides)
  3. a literal fallback, used only when runtime/ isn't importable

  env var                 registry role
  AGENT_MODEL_ARCHITECT   architect
  AGENT_MODEL_COMPLEX     developer
  AGENT_MODEL_STANDARD    validator
  AGENT_MODEL_FAST        fast
  AGENT_MODEL_LOCAL       fast

(2) is what stops this table drifting. These defaults were hardcoded cloud ids
— claude-sonnet-4-6 / gpt-4o / llama-3.3-70b-versatile / gemma2 / llama3 —
that had no relationship to what models.yaml actually routed to, so the
Layer-1 dev router and the production gateway named different models for the
same tier and neither knew.

Escalation policy: only escalate to a frontier model after two consecutive
failures on the cheaper tier.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _shared import role_model


# Mirrors runtime.provider_dispatch._EXHAUSTION_MARKERS. Duplicated on purpose
# and only reachable when that module is too old to ask: a tenant pins a
# framework VERSION, so scripts/ can be newer than the runtime package it
# imports. Adding the import to call()'s required-import tuple instead would
# turn a version skew into "every LLM call raises ImportError" — the same shape
# as the credential-lookup regression that broke KYC Sentinel's CI.
# scripts/test/test_exhaustion_classification.py asserts the two never drift.
_FALLBACK_EXHAUSTION_MARKERS = (
    "credit balance is too low",     # Anthropic
    "requires more credits",         # OpenRouter 402
    "insufficient credits",          # OpenRouter / misc
    "insufficient_quota",            # OpenAI
    "rate limit",
    "billing",
    "payment required",
    # "429" alone used to be here, and `"429" in msg` matches the digits
    # ANYWHERE: "however you requested 14290 tokens" is a context-length error,
    # a hard user bug, and it was classified as exhaustion — so the gateway
    # degraded through every tier on a malformed prompt and the eval path
    # reported a billing problem that did not exist. Request ids do it too.
    # The phrases below are what a real 429 carries in its body; the status
    # code itself is checked structurally in is_provider_exhausted.
    "too many requests",
    "resource_exhausted",            # Google AI / Vertex
    "quota exceeded",
    "overloaded",
)


# Statuses worth another attempt on the SAME route.
#
# 429 is throttling. 502/503/504 are the provider being transiently
# unavailable, and they are a different failure from exhaustion: the tier still
# has budget, the endpoint just did not serve this request. Retrying is what
# the provider itself asks for — Gemini's 503 body reads "Spikes in demand are
# usually temporary. Please try again later."
#
# Only 429 was here, which cost a real verdict. On 2026-08-31 the first call of
# KYC Sentinel's 7-case hallucination suite took a 503; the other six graded
# cleanly, including the planted ghost-citation control. But a judged gate needs
# every case under the 100%-graded quorum rule, so one un-retried transient
# turned a complete result into NO VERDICT — and on a free tier of ~20 calls a
# day, that is a whole daily window spent for nothing.
#
# 500 is deliberately NOT here. It is what several providers return for a
# request they will reject identically every time (a malformed body, an
# unsupported parameter), and retrying that burns three more calls out of the
# same daily allowance to arrive at the same error, more slowly.
_RETRYABLE_STATUSES = frozenset({429, 502, 503, 504})


def _exhausted(exc: Exception) -> bool:
    """Is this a provider-exhaustion failure (billing / quota / throttling)?

    Classification only — unlike the gateway, this path never degrades on the
    answer. See the call site for why a substituted judge is worse than no
    judge.
    """
    try:
        from runtime.provider_dispatch import is_provider_exhausted

        return is_provider_exhausted(exc)
    except Exception:
        msg = str(exc).lower()
        return any(k in msg for k in _FALLBACK_EXHAUSTION_MARKERS)


# ── Model config ──────────────────────────────────────────────────────────────


def _tier(env_var: str, role: str, fallback: str) -> str:
    override = os.environ.get(env_var, "").strip()
    return override or role_model(role, fallback)


MODEL_ARCHITECT = _tier("AGENT_MODEL_ARCHITECT", "architect", "qwen2.5")
MODEL_COMPLEX = _tier("AGENT_MODEL_COMPLEX", "developer", "llama3.2:3b")
MODEL_STANDARD = _tier("AGENT_MODEL_STANDARD", "validator", "falcon3:3b")
MODEL_FAST = _tier("AGENT_MODEL_FAST", "fast", "smollm2")

# GitHub Models (https://docs.github.com/en/github-models) — free-tier
# OpenAI-compatible inference using a GitHub token instead of a billed
# OPENAI_API_KEY. GITHUB_TOKEN is the automatically-provided token in
# every GitHub Actions run (no extra secret needed there); GITHUB_MODELS_TOKEN
# is the override for local dev (e.g. `export GITHUB_MODELS_TOKEN=$(gh auth token)`).
GITHUB_MODELS_TOKEN = os.environ.get("GITHUB_MODELS_TOKEN") or os.environ.get(
    "GITHUB_TOKEN", ""
)
# Offline fallback — the registry's smallest local tier, so "what runs when the
# network drops" is the same model the gateway's degrade ladder bottoms out on.
MODEL_LOCAL = _tier("AGENT_MODEL_LOCAL", "fast", "smollm2")

# Token thresholds
TOKEN_TIER_HIGH = 8_000
TOKEN_TIER_MEDIUM = 3_000

# Keywords that force a higher-capability model regardless of token count
ARCHITECT_KEYWORDS: list[str] = [
    "architect",
    "system design",
    "rfc",
    "design decision",
    "migration",
    "race condition",
    "security",
    "cryptography",
    "ast",
    "parser",
    "distributed",
]

COMPLEX_KEYWORDS: list[str] = [
    "refactor",
    "optimise",
    "optimize",
    "performance",
    "concurrency",
    "async",
    "deadlock",
    "memory leak",
    "dependency injection",
    "interface design",
]

# ── Token counter ─────────────────────────────────────────────────────────────


def _count_tokens(text: str) -> int:
    """Use tiktoken if available; fall back to word-count heuristic."""
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text.split()) * 4 // 3)


# ── Keyword scorer ────────────────────────────────────────────────────────────


def _keyword_tier(prompt: str) -> Optional[str]:
    lower = prompt.lower()
    for kw in ARCHITECT_KEYWORDS:
        if kw in lower:
            return "architect"
    for kw in COMPLEX_KEYWORDS:
        if kw in lower:
            return "complex"
    return None


# ── Failure tracker (session-scoped) ─────────────────────────────────────────

_consecutive_failures: dict[str, int] = {}

# Module-level dict keyed by model name, only shrinks via record_success for
# individual models — fine for this file's actual usage (dev-mode, one
# process per session, a handful of model names), but unbounded if ever used
# in a long-running process with many distinct/dynamic model ids
# (Product_Archive.md 4.4). This is a cheap upper bound, not an LRU — if it
# ever fires, dropping the whole dict just means the escalation counters
# reset to 0, which is the same as every model's first call ever.
_MAX_TRACKED_MODELS = 256


def record_failure(model: str) -> int:
    """Increment failure counter for model. Returns new count."""
    if (
        len(_consecutive_failures) >= _MAX_TRACKED_MODELS
        and model not in _consecutive_failures
    ):
        _consecutive_failures.clear()
    _consecutive_failures[model] = _consecutive_failures.get(model, 0) + 1
    return _consecutive_failures[model]


def record_success(model: str) -> None:
    """Reset failure counter after a successful call."""
    _consecutive_failures.pop(model, None)


def _should_escalate(model: str) -> bool:
    return _consecutive_failures.get(model, 0) >= 2


# ── Main router ───────────────────────────────────────────────────────────────


class ModelRoute:
    """
    Holds the chosen model and all parameters needed to invoke the LLM API.

    Attributes:
        model:      Model identifier string.
        base_url:   API base URL.
        api_key:    API key (may be empty for Ollama).
        tier:       "architect" | "complex" | "standard" | "fast" | "local"
        is_local:   True if routing to a local Ollama model.
    """

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str,
        tier: str,
        is_local: bool = False,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.tier = tier
        self.is_local = is_local

    def __repr__(self) -> str:
        return f"<ModelRoute model={self.model!r} tier={self.tier!r} local={self.is_local}>"


def route(
    prompt: str,
    task_type: Optional[str] = None,
    force_local: bool = False,
) -> ModelRoute:
    """
    Analyse prompt and return the cheapest capable ModelRoute.

    Args:
        prompt:     The full prompt string (system + user).
        task_type:  Optional hint: "architect" | "code" | "format" | "review".
        force_local: Override to always route to local Ollama.
    """
    # Check network availability
    try:
        from network_watchdog import is_online

        online = is_online() and not force_local
    except Exception:
        online = False

    if not online:
        return _local_route()

    token_count = _count_tokens(prompt)
    kw_tier = _keyword_tier(prompt)

    # Explicit task type overrides
    if task_type == "architect":
        tier = "architect"
    elif task_type == "format":
        tier = "fast"
    elif task_type == "code":
        tier = "standard"
    elif kw_tier:
        tier = kw_tier
    elif token_count > TOKEN_TIER_HIGH:
        tier = "architect"
    elif token_count > TOKEN_TIER_MEDIUM:
        tier = "complex"
    else:
        tier = "standard"

    # Escalation: if the standard/fast model has failed twice, bump up
    if tier == "standard" and _should_escalate(MODEL_STANDARD):
        tier = "complex"
    if tier == "fast" and _should_escalate(MODEL_FAST):
        tier = "standard"

    return _build_cloud_route(tier)


def _build_cloud_route(tier: str) -> ModelRoute:
    groq_key = os.environ.get("GROQ_API_KEY", "")

    if tier == "architect":
        # Provider inferred from AGENT_MODEL_ARCHITECT's actual value (via
        # _route_for_model) rather than hardcoded to Anthropic — previously
        # this tier always posted to api.anthropic.com regardless of what
        # AGENT_MODEL_ARCHITECT was set to, so overriding that env var to a
        # non-Anthropic model id silently sent it to the wrong host with
        # the wrong key. route.tier is relabelled "architect" below since
        # _route_for_model's own generic "forced" tier label is for the
        # force_model param's callers (eval_judge.py), not this one.
        r = _route_for_model(MODEL_ARCHITECT)
        r.tier = "architect"
        return r
    elif tier == "complex":
        r = _route_for_model(MODEL_COMPLEX)
        r.tier = "complex"
        return r
    elif tier == "standard":
        if groq_key:
            return ModelRoute(
                model=MODEL_STANDARD,
                base_url="https://api.groq.com/openai/v1",
                api_key=groq_key,
                tier="standard",
            )
        # Fallback to local Ollama
        return _local_route(model=MODEL_LOCAL)
    else:  # fast
        return _local_route(model=MODEL_FAST)


def _ollama_base_url() -> str:
    """Ollama's OpenAI-compatible base, tolerating OLLAMA_BASE_URL with or
    without the `/v1` suffix.

    runtime/llm_gateway.py reads the same variable as `${OLLAMA_BASE_URL}/v1`,
    appending the suffix itself, so a tenant that sets the bare host — which is
    what Ollama's own docs show — worked on the workload path and 404'd on the
    eval path (`.../chat/completions` instead of `.../v1/chat/completions`).
    One variable, two conventions, and only one of them documented.
    """
    base = (os.environ.get("OLLAMA_BASE_URL") or "http://localhost:11434").rstrip("/")
    return base if base.endswith("/v1") else base + "/v1"


def _local_route(model: Optional[str] = None) -> ModelRoute:
    return ModelRoute(
        model=model or MODEL_LOCAL,
        base_url=_ollama_base_url(),
        api_key="ollama",
        tier="local",
        is_local=True,
    )


def _credential_for(model: str, default_env: str) -> str:
    """API key for an exact model id, honouring a registry role's `api_key_env`.

    Mirrors `LLMGateway._lookup_api_key`: a role's own variable wins when it is
    populated, else the provider default. Without this the two paths disagreed
    about which credential a route uses, and the eval judge was the casualty —
    KYC Sentinel's judge declares `api_key_env: ANTHROPIC_API_KEY_JUDGE` (its
    own account, so a rate limit on the analyst can't also take out its
    reviewer), the gateway honoured it, and this function read
    ANTHROPIC_API_KEY only. Setting exactly the variable the tenant declared
    passed the eval preflight and then sent an empty auth header: 401 on every
    judge call, from a config that is correct everywhere else.

    Falls back silently when the registry can't be read — scripts/ is installed
    to ~/.agent-framework and may run with no runtime/ on the path.
    """
    try:
        from _shared import load_registry
        from runtime.provider_dispatch import credential_env_for_model

        for cfg in (load_registry() or {}).values():
            if cfg.get("id") != model:
                continue
            env = credential_env_for_model(cfg)
            if env and os.environ.get(env, "").strip():
                return os.environ[env]
            break
    except Exception:  # fail-open: fall through to the provider default
        pass
    return os.environ.get(default_env, "")


def _registry_route(model: str) -> Optional[ModelRoute]:
    """Route built from the model's own models.yaml entry, or None if it has no
    entry (or the registry can't be read).

    This is the authoritative path. The substring heuristics below are a
    fallback for ids the registry doesn't declare, and they cannot express a
    provider they were not written for: an id matching none of their patterns
    fell through to `_local_route`, so a Grok or Gemini judge was silently
    routed to localhost Ollama rather than erroring. `grok-4` and
    `gemini-2.5-pro` both did exactly that — and because provenance recorded
    the REQUESTED id, the scorecard would have named a judge that never ran.

    The heuristics were fragile for declared models too:
    `llama-3.3-70b-versatile` routes to Groq only when GROQ_API_KEY happens to
    be set in the process, and to localhost otherwise — a judge silently
    swapped by an unset environment variable.
    """
    try:
        from _shared import load_registry
        from runtime.provider_dispatch import credential_env_for_model

        for cfg in (load_registry() or {}).values():
            if cfg.get("id") != model:
                continue
            provider = cfg.get("provider")
            base_url = cfg.get("endpoint") or _PROVIDER_BASE_URL.get(provider)
            if not base_url:
                return None  # unknown provider with no endpoint — let the heuristics try
            base_url = os.path.expandvars(base_url)
            if provider == "ollama":
                return _local_route(model=model)
            # Via _credential_for, NOT a bare os.environ lookup on the declared
            # variable: a role's `api_key_env` is an opt-in override, and when
            # it is unset the provider default still applies. KYC Sentinel's
            # judge declares ANTHROPIC_API_KEY_JUDGE and documents exactly that
            # ("opt-in, not a hard requirement"). Reading only the declared
            # name sends an EMPTY auth header and 401s every call — the same
            # failure _credential_for was written to fix, reintroduced here.
            from runtime.provider_dispatch import default_api_key_env

            env = credential_env_for_model(cfg)
            # The fallback is the PROVIDER default, not the declared name —
            # passing the declared name makes it fall back to itself, which is
            # still empty and still 401s.
            fallback = default_api_key_env(provider) or ""
            return ModelRoute(
                model=model,
                base_url=base_url,
                api_key=_credential_for(model, fallback) if env else "",
                tier="forced",
            )
    except Exception:  # fail-open: no runtime/ on the path, unreadable registry
        return None
    return None


# Default hosts for registry providers that speak an OpenAI-compatible API (or
# Anthropic's). Mirrors LLMGateway._resolve_endpoint so the workload path and
# the eval path resolve one declared route the same way; a models.yaml
# `endpoint` overrides either.
_PROVIDER_BASE_URL = {
    "anthropic": "https://api.anthropic.com/v1",
    "openai": "https://api.openai.com/v1",
    "groq": "https://api.groq.com/openai/v1",
    "xai": "https://api.x.ai/v1",
    "google_ai": "https://generativelanguage.googleapis.com/v1beta/openai",
    "openrouter": "https://openrouter.ai/api/v1",
    "ollama": "${OLLAMA_BASE_URL}",
}


def _route_for_model(model: str) -> ModelRoute:
    """Build a route for an EXACT model id, bypassing route()'s complexity
    heuristics entirely — for callers (e.g. eval_judge.py's judge model)
    that need a specific, caller-chosen model rather than "whichever tier
    this prompt's length/keywords land on."

    Registry first (`_registry_route`); the id-substring heuristics below only
    run for models models.yaml does not declare.
    """
    declared = _registry_route(model)
    if declared is not None:
        return declared

    anthropic_key = _credential_for(model, "ANTHROPIC_API_KEY")
    openai_key = _credential_for(model, "OPENAI_API_KEY")
    groq_key = _credential_for(model, "GROQ_API_KEY")
    lower = model.lower()

    if "claude" in lower:
        return ModelRoute(
            model=model,
            base_url="https://api.anthropic.com/v1",
            api_key=anthropic_key,
            tier="forced",
        )
    if "gpt" in lower or lower.startswith("o1") or lower.startswith("o3"):
        if GITHUB_MODELS_TOKEN:
            # Free-tier, no OpenAI billing required — prefer this over a
            # possibly-unfunded OPENAI_API_KEY. GitHub Models namespaces
            # OpenAI model ids under "openai/" (confirmed against the live
            # API: bare "gpt-4o" 404s, "openai/gpt-4o" succeeds).
            gh_model = model if "/" in model else f"openai/{model}"
            return ModelRoute(
                model=gh_model,
                base_url="https://models.github.ai/inference",
                api_key=GITHUB_MODELS_TOKEN,
                tier="forced",
            )
        return ModelRoute(
            model=model,
            base_url="https://api.openai.com/v1",
            api_key=openai_key,
            tier="forced",
        )
    if groq_key and ("llama" in lower or "mixtral" in lower or "gemma" in lower):
        return ModelRoute(
            model=model,
            base_url="https://api.groq.com/openai/v1",
            api_key=groq_key,
            tier="forced",
        )
    return _local_route(model=model)


# ── Convenience: call via OpenAI-compatible API ───────────────────────────────


def call(
    prompt: str,
    system: str = "",
    task_type: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 4096,
    force_model: Optional[str] = None,
) -> str:
    """
    Route and invoke the model. Returns the response text.
    Records token usage for circuit breaker.

    force_model: bypass route()'s complexity-tier heuristics and use this
    exact model id (e.g. a configured eval judge model that must not be
    silently swapped for whatever tier the prompt's length/keywords land
    on — see eval_judge.py's run_judge()).
    """
    route_result = (
        _route_for_model(force_model)
        if force_model
        else route(prompt, task_type=task_type)
    )

    # Build messages
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    try:
        import httpx

        try:
            from runtime.provider_dispatch import (
                build_request,
                infer_provider,
                parse_response,
            )
        except ImportError:
            # Add the repo ROOT, not runtime/: the runtime's modules import
            # each other as `runtime.X` (framework G6), so a flat runtime/
            # path no longer resolves them.
            import sys as _sys
            from pathlib import Path as _Path

            _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
            from runtime.provider_dispatch import (
                build_request,
                infer_provider,
                parse_response,
            )

        # Request building / response parsing shared with
        # runtime/llm_gateway.py via runtime/provider_dispatch.py — this
        # used to independently re-derive "is this Anthropic" from the
        # base_url string and build/parse bodies inline, drifting from
        # llm_gateway.py's own copy of the same logic (Product_Archive.md 4.3).
        provider = infer_provider(route_result.base_url)
        path_suffix, headers, body = build_request(
            provider,
            route_result.model,
            messages,
            route_result.api_key,
            max_tokens,
            temperature,
        )
        # cost_router's Anthropic base_url has no /v1 segment (unlike
        # llm_gateway.py's), so its messages endpoint is base_url + "/messages",
        # not base_url + "/v1/messages" — preserve that pre-existing URL shape
        # exactly rather than switching it to provider_dispatch's path.
        url = route_result.base_url.rstrip("/") + (
            "/messages" if provider == "anthropic" else path_suffix
        )

        # Transient-failure retry with FULL JITTER (P11a lesson — do not remove
        # the jitter): a bare `2**n * 5` gives every concurrent CI job identical
        # waits, so they retry in lockstep and re-saturate the provider's
        # rate window together (observed live against Groq's 30 RPM free
        # tier). random.uniform(0, 3) de-synchronizes them.
        # Waits: ~10–13s, ~20–23s, ~40–43s across the 3 retries (4 attempts).
        # Which statuses qualify, and why 500 does not: _RETRYABLE_STATUSES.
        import random
        import time as _time

        max_attempts = 4
        for attempt in range(1, max_attempts + 1):
            resp = httpx.post(url, json=body, headers=headers, timeout=120.0)
            if resp.status_code in _RETRYABLE_STATUSES and attempt < max_attempts:
                wait = (2**attempt) * 5 + random.uniform(0, 3)
                _time.sleep(wait)
                continue
            # `status_code >= 400` rather than httpx's `resp.is_error`: the
            # test doubles in scripts/test/ implement the minimal response
            # surface (status_code / json / text), and depending on an httpx
            # convenience property here would couple this path to the real
            # client for no benefit.
            if resp.status_code >= 400:
                # raise_for_status() reports only the status line, so a 400
                # arrived as "Client error '400 Bad Request'" with no hint at
                # WHICH field was wrong — undebuggable without re-running by
                # hand with the key. Providers put the actionable part in the
                # body ({"error": {"message": "..."}}), so surface it. Bodies
                # are error text, not request echoes: no key material in them
                # (the key travels in a header), and truncation bounds a
                # provider that returns an HTML error page.
                detail = getattr(resp, "text", "") or "(no response body)"
                err = RuntimeError(f"HTTP {resp.status_code} from {url}: {detail[:600]}")
                # Classify, but do NOT degrade. runtime/llm_gateway.py walks the
                # models.yaml degrade_to chain on exhaustion; this path
                # deliberately does not, because its caller is the eval judge
                # and a substituted grader is not a grader — a weaker model
                # emits confident verdicts into the same `score` field, against
                # the same threshold, gating the same merges, and nothing
                # downstream can tell. Failing loudly is the safer default; the
                # scorecard skips with a cause instead of scoring with a
                # stand-in. (See OPERATIONS.md "When a gate blocks, and when it
                # steps aside".)
                if _exhausted(err):
                    raise RuntimeError(
                        f"Provider exhausted for model {route_result.model!r} — "
                        f"billing, quota or throttling, which retrying the same "
                        f"route will not clear. This path does not fall back to "
                        f"another model. {err}"
                    ) from err
                raise err
            break
        data = resp.json()

        text, in_tok, out_tok = parse_response(provider, data)

        # Record token usage for circuit breaker.
        #
        # `parse_response` returns Optional[int] — a provider that omits its
        # `usage` block reports None rather than a fabricated 0. Handing that
        # to the breaker raised TypeError straight into the blanket
        # `except Exception: pass` below, so the call was silently unmetered on
        # BOTH tiers: no tokens toward the 5-minute burst window, no dollars
        # toward the monthly cap. runtime/llm_gateway.py's sibling path already
        # warns and bills the reserved estimate for exactly this case; this
        # call site never got the same treatment because it lives in another
        # package (review-levers: grep-for-siblings).
        #
        # There is no reservation to fall back on here — cost_router has no
        # budget ledger — so the honest outcome is to say the call went
        # unmetered rather than to invent a number for it.
        # Guarded, and its own statement. This import stood bare in the middle
        # of a request path, so any failure to load the breaker — a malformed
        # env var read at its module level, a missing dependency — propagated
        # out of a call whose provider request had already completed and had
        # already been paid for. An unloadable breaker is a bookkeeping fault
        # like any other here: say it, and let the call return.
        try:
            from circuit_breaker import (
                CircuitBreakerTripped,
                audit_token_velocity_circuit,
            )
        except Exception as exc:
            breaker = None
            print(
                f"[cost_router] WARNING: circuit breaker unavailable "
                f"({type(exc).__name__}: {exc}) — this call is NOT counted "
                f"against the burst window or the monthly cap.",
                file=sys.stderr,
            )
        else:
            breaker = audit_token_velocity_circuit

        if breaker is None:
            pass  # already reported above
        elif in_tok is None or out_tok is None:
            print(
                f"[cost_router] WARNING: provider {provider!r} returned no usage "
                f"block for model {route_result.model!r} — this call is NOT counted "
                f"against the burst window or the monthly cap. Both tiers "
                f"under-report by however many such calls are made.",
                file=sys.stderr,
            )
        else:
            try:
                breaker(in_tok, out_tok)
            except CircuitBreakerTripped as tripped:
                # Deliberate, and narrow. The provider call above already
                # completed and was already paid for, so there is nothing left
                # to break; the breaker's job at this call site is to record
                # and to alert (it notifies before raising). Catching only this
                # is the point — the blanket `except Exception: pass` that used
                # to stand here is what hid the TypeError above for as long as
                # it stood.
                print(f"[cost_router] {tripped}", file=sys.stderr)
            except Exception as exc:
                # Still fail-open — a bookkeeping fault must not fail a call
                # the provider already answered — but SAID, not swallowed. The
                # silent version is what let a TypeError stand.
                print(
                    f"[cost_router] WARNING: circuit breaker bookkeeping failed "
                    f"({type(exc).__name__}: {exc}) — this call is unmetered.",
                    file=sys.stderr,
                )

        record_success(route_result.model)
        return text

    except Exception as exc:
        record_failure(route_result.model)
        raise RuntimeError(
            f"LLM call failed [{route_result.tier} / {route_result.model}]: {exc}"
        ) from exc


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import json as _json

    prompt = " ".join(sys.argv[1:]) or "Write a hello world function in Python."
    r = route(prompt)
    print(
        _json.dumps(
            {
                "model": r.model,
                "tier": r.tier,
                "base_url": r.base_url,
                "is_local": r.is_local,
                "estimated_tokens": _count_tokens(prompt),
            },
            indent=2,
        )
    )
