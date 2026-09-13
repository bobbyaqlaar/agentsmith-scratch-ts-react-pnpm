"""
runtime/provider_dispatch.py — shared provider request building and response
parsing for runtime/llm_gateway.py and scripts/cost_router.py.

Before this module existed, llm_gateway.py and cost_router.py each
independently built provider request bodies/headers and parsed responses,
with no shared code (Product_Archive.md 4.3). Only the provider-dispatch
shape is shared here; each caller's own routing/budget/degrade-ladder logic
stays in its own file.

Two tiers of providers:

1. Direct-API providers (`anthropic`, `openai_compatible`) — same host every
   time, static API-key auth. `build_request`/`parse_response` below, used
   by both llm_gateway.py and cost_router.py. llm_gateway.py resolves the
   base_url/api_key itself (optionally overridden per-model via
   `models.yaml`'s `endpoint`/`api_key_env` fields) and prepends it to the
   path these return.

2. Cloud-native providers (`vertex_ai`, `azure_openai`, `bedrock`,
   `huawei_modelarts`) — each needs its own auth scheme (OAuth2 service
   account, api-key + api-version, SigV4, AK/SK signing) and its own
   request/response envelope, not just a different host. These implement
   the `CloudProviderAdapter` protocol below and are looked up via
   `get_cloud_adapter(provider)`. `build_cloud_request`/`parse_cloud_response`
   are the entry points llm_gateway.py calls — each returns/accepts a full
   URL, not just a path, since cloud providers bake project/region/
   deployment/endpoint-id into the URL itself.

A flat `if/elif` per cloud provider was considered and rejected: each cloud
provider's auth, URL shape, and envelope are genuinely independent axes (you
can't share an "is it Bearer-token-shaped" branch across SigV4 and OAuth2),
so a one-class-per-provider Protocol keeps each provider's quirks contained
and makes adding a fifth provider additive rather than another branch
threaded through three concerns at once.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Optional, Protocol

logger = logging.getLogger(__name__)


# ── Direct-API providers (anthropic / openai-compatible) ──────────────────────


def infer_provider(base_url: str) -> str:
    """cost_router.py only has a base_url (no separate provider field) — this
    mirrors its existing "anthropic" in base_url check."""
    return "anthropic" if "anthropic" in base_url else "openai_compatible"


# ── Wire format ───────────────────────────────────────────────────────────────
#
# WHO hosts a model and WHAT SHAPE it speaks are different questions, and
# conflating them is what made OpenRouter awkward to express: it serves Claude,
# Gemini and Llama behind ONE OpenAI-compatible endpoint, so `provider:
# openrouter` says nothing about the envelope while `provider: anthropic`
# implies both a host and the Messages API. A catalog entry can now declare
# `api_format` explicitly; when it does not, the provider's usual format is
# assumed, so every existing entry keeps working unchanged.
API_FORMAT_OPENAI = "openai_chat"
API_FORMAT_ANTHROPIC = "anthropic_messages"

_PROVIDER_DEFAULT_FORMAT = {
    "anthropic": API_FORMAT_ANTHROPIC,
    "openai": API_FORMAT_OPENAI,
    "groq": API_FORMAT_OPENAI,
    "ollama": API_FORMAT_OPENAI,
    "xai": API_FORMAT_OPENAI,
    "google_ai": API_FORMAT_OPENAI,   # AI Studio's OpenAI-compatibility layer
    "openrouter": API_FORMAT_OPENAI,
    "azure_openai": API_FORMAT_OPENAI,
}


def resolve_api_format(cfg: dict) -> str:
    """The wire format a catalog entry speaks.

    An explicit `api_format` wins; otherwise it follows from the provider.
    This is what lets one provider serve several vendors' models
    (OpenRouter) and one vendor be reached through several formats
    (Claude direct = Messages API, Claude via OpenRouter = OpenAI chat).
    """
    declared = (cfg.get("api_format") or "").strip()
    if declared:
        return declared
    return _PROVIDER_DEFAULT_FORMAT.get(cfg.get("provider", "openai"), API_FORMAT_OPENAI)


def build_request(
    provider: str,
    model_id: str,
    messages: list[dict],
    api_key: str,
    max_tokens: int,
    temperature: float = 0.2,
    api_format: Optional[str] = None,
) -> tuple[str, dict, dict]:
    """Returns (url_path, headers, body) for the given provider.

    `api_format` selects the envelope explicitly. When omitted it is derived
    from `provider`, preserving the original behaviour: "anthropic" uses the
    Messages API shape (system pulled out of the messages list into its own
    top-level field), anything else is treated as OpenAI-compatible.
    """
    fmt = api_format or _PROVIDER_DEFAULT_FORMAT.get(provider, API_FORMAT_OPENAI)
    if fmt == API_FORMAT_ANTHROPIC:
        system = (
            "\n".join(m["content"] for m in messages if m["role"] == "system") or None
        )
        user_messages = [m for m in messages if m["role"] != "system"]
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        body: dict[str, Any] = {
            "model": model_id,
            "max_tokens": max_tokens,
            "messages": user_messages,
            # TEMPERATURE WAS DROPPED HERE. The OpenAI branch below has always
            # sent it; this one built a body without it, so every Anthropic-
            # shaped route ran at the provider's default of 1.0 no matter what
            # the caller asked for.
            #
            # The casualty is a control this repo went to some trouble over:
            # scripts/eval_judge.py pins JUDGE_TEMPERATURE = 0.0 so grading is
            # deterministic, and a judge routed to Claude direct — the obvious
            # choice — graded at 1.0 instead. Pinned on the OpenAI routes,
            # discarded on the Anthropic ones, with nothing to show which.
            "temperature": _anthropic_temperature(temperature),
        }
        if system:
            body["system"] = system
        return "/v1/messages", headers, body

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {
        "model": model_id,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    return "/chat/completions", headers, body


def parse_openai_completion(data: dict) -> tuple[str, Optional[int], Optional[int]]:
    """(text, prompt_tokens, completion_tokens) from an OpenAI-shaped response.

    THE COUNTS ARE None WHEN THE PROVIDER DID NOT REPORT THEM. They defaulted
    to 0, which is the one thing this codebase refuses to do anywhere else:
    `agent_runs.input_tokens` is nullable so "not reported" stays
    distinguishable from "used nothing", `_record_span_attributes` writes
    `llm.usage.reported`, and `runtime/metrics` declines to put an unreported
    count into a histogram. All of that guards a distinction this function had
    already destroyed.

    It was not only a dashboard problem. `cost_usd` is computed from these, so
    a provider that omits `usage` — an OpenAI-compatible proxy, a shim, a
    streamed response without `stream_options.include_usage` — produced a cost
    of exactly $0.00, and the budget reconcile then released the whole
    reservation. The call was free as far as the cap was concerned.

    `.get("content") or ""`, not a bare index: OpenAI-compatible providers
    legitimately return `"content": null` — a model that emitted only reasoning
    tokens, produced nothing before a stop, or was cut off by a filter.
    Returning None breaks the (text, int, int) contract and the None then
    travels until something dereferences it: the PII scrubber, a security
    control, died with `TypeError: expected string or bytes-like object`
    several frames from the cause. An empty completion is a normal provider
    outcome and belongs to the caller to interpret, not a crash in a guardrail.

    Shared with the cloud adapters. That hardening was applied to the
    module-level parser while the Azure and Huawei adapters kept byte-identical
    unhardened copies, so a null completion that had already been fixed once
    still crashed on those two routes.
    """
    text = data["choices"][0]["message"].get("content") or ""
    usage = data.get("usage") or {}
    return text, usage.get("prompt_tokens"), usage.get("completion_tokens")


def parse_anthropic_completion(data: dict) -> tuple[str, Optional[int], Optional[int]]:
    """(text, input_tokens, output_tokens) from an Anthropic-shaped response.

    An empty `content` array is this shape's equivalent of a null completion —
    a stop before any block was emitted. Indexing it raises IndexError inside a
    guardrail rather than returning nothing for the caller to interpret.

    Counts are None when unreported, for the reasons on
    `parse_openai_completion` — the two are siblings and were wrong together.
    """
    blocks = data.get("content") or []
    text = (blocks[0].get("text") if blocks else "") or ""
    usage = data.get("usage") or {}
    return text, usage.get("input_tokens"), usage.get("output_tokens")


def parse_response(
    provider: str, data: dict, api_format: Optional[str] = None
) -> tuple[str, Optional[int], Optional[int]]:
    """Returns (text, input_tokens, output_tokens).

    Mirrors build_request: the response envelope must be parsed with the same
    format it was requested in. Claude reached through OpenRouter answers in
    OpenAI shape, not Anthropic shape, so keying this off the provider alone
    would read the wrong fields.
    """
    fmt = api_format or _PROVIDER_DEFAULT_FORMAT.get(provider, API_FORMAT_OPENAI)
    if fmt == API_FORMAT_ANTHROPIC:
        return parse_anthropic_completion(data)
    return parse_openai_completion(data)


# ── Streaming (TTFT path — llm_gateway.complete_stream) ───────────────────────
#
# Streaming lives here with the rest of the per-provider envelope knowledge
# (TestbedFeedback-2026-07-21 G1). Before this, complete_stream() inlined the
# OpenAI delta shape and raised NotImplementedError for everything else — so
# a tenant routing its latency-critical call to Anthropic (the obvious
# design) could not use the TTFT budget at all.

_STREAMING_PROVIDERS = {"openai", "ollama", "groq", "anthropic", "xai", "google_ai", "openrouter"}


_DEFAULT_API_KEY_ENV = {
    "anthropic": "ANTHROPIC_API_KEY",
    "groq": "GROQ_API_KEY",
    "openai": "OPENAI_API_KEY",
    "azure_openai": "AZURE_OPENAI_API_KEY",
    # xAI and Google AI Studio both expose OpenAI-compatible chat/completions,
    # so they need no adapter — only a host and a key variable. Added to give
    # the `judge` role cross-vendor options: a Claude analyst graded by a Claude
    # judge shares a training lineage and RLHF profile, and models rate their
    # own family's output higher. `judge_independence_warning` only catches
    # IDENTICAL ids, so same-vendor judging reads as independent when it isn't.
    "xai": "XAI_API_KEY",
    # OpenRouter — one OpenAI-compatible endpoint fronting many vendors'
    # models. Its ids are namespaced ("anthropic/claude-sonnet-4.5"), and a
    # Claude served this way speaks OpenAI chat, not the Messages API, which
    # is exactly why api_format is declared separately from provider.
    "openrouter": "OPENROUTER_API_KEY",
    # Google AI Studio (generativelanguage), NOT Vertex AI: this is the
    # api-key-in-a-header path. `vertex_ai` below is the same models behind
    # service-account OAuth, and the two are not interchangeable — an AI Studio
    # key cannot authenticate against Vertex.
    "google_ai": "GEMINI_API_KEY",
    # ollama is local and takes a literal "ollama" token — no credential.
    "ollama": None,
    # Cloud-native adapters authenticate through their SDK's credential chain
    # (google-auth ADC, boto3, Huawei AK/SK), not a single API-key env var.
    "vertex_ai": None,
    "bedrock": None,
    "huawei_modelarts": None,
}


def default_api_key_env(provider: str) -> Optional[str]:
    """Env var holding this provider's API key, or None when it needs none.

    One mapping, so "which credential does this role need?" is answerable
    outside the gateway — CI preflights, health checks, an operator asking why
    a route is skipped. It was previously only expressible as the string
    literals inside `LLMGateway._resolve_endpoint`'s if/elif chain, so a
    caller wanting the answer had to hardcode a provider name and guess.
    Unknown providers fall back to OPENAI_API_KEY, matching that chain's
    `else` branch (everything else is OpenAI-compatible).
    """
    return _DEFAULT_API_KEY_ENV.get(provider, "OPENAI_API_KEY")


def credential_env_for_model(cfg: dict) -> Optional[str]:
    """The env var a model config actually reads, honouring `api_key_env`.

    A tenant can point one role at its own account — KYC Sentinel's judge uses
    `api_key_env: ANTHROPIC_API_KEY_JUDGE` so a rate limit on the analyst's
    account can't also take out its reviewer. Anything checking whether a role
    is runnable has to respect that, or it checks the wrong variable and
    reports a correctly-configured route as unavailable.
    """
    default_env = default_api_key_env(cfg.get("provider", "openai"))
    if default_env is None:
        return None
    return cfg.get("api_key_env") or default_env


_EXHAUSTION_MARKERS = (
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


def is_provider_exhausted(exc: Exception) -> bool:
    """True when the provider is unavailable for this key/tier — billing, quota
    or throttling, none of which resolve by retrying the same route.

    Lives here rather than on the gateway because both LLM call paths need the
    same answer and must not drift into two definitions of "exhausted".
    `llm_gateway` acts on it by degrading to the next tier; `cost_router` (the
    eval path) deliberately does NOT degrade — a substituted grader is not a
    grader — and uses it only to say *why* it stopped, so a skipped gate names
    a billing state instead of an opaque status code.

    Matching is on message text, which is why both paths must surface the
    provider's response body rather than `raise_for_status()`'s status line:
    the body carries "credit balance is too low", and without it this returns
    False for the one case it exists to catch. That exact bug has now been
    fixed twice — once on the gateway's streaming path (found live against a
    credit-exhausted key) and once in cost_router.
    """
    # Structured first, where the exception carries a status. A 429 or a 402 is
    # one whatever the body says, and — the point of the change — the digits
    # "429" appearing inside a message are not.
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if status in (402, 429):
        return True

    msg = str(exc).lower()
    return any(k in msg for k in _EXHAUSTION_MARKERS)


# A model id the provider no longer serves. Phrases, not bare status codes:
# "404" appears in request ids and token counts, which is the mistake
# _EXHAUSTION_MARKERS documents above.
_MODEL_GONE_MARKERS = (
    "model_not_found",          # OpenAI / Groq / OpenRouter
    "model not found",
    "does not exist",           # Groq: "The model `x` does not exist"
    "is not available",
    "has been deprecated",      # OpenAI sunset wording
    "has been decommissioned",
    "no longer available",
    "unknown model",
)


def is_model_gone(exc: object) -> bool:
    """True when the configured model itself is the problem — retired, renamed,
    or never existed for this key.

    Distinct from `is_provider_exhausted`, and the distinction is the whole
    point: exhaustion is weather and clears on its own, a withdrawn model is a
    broken configuration in the repo and no later run will fix it. Treating
    them alike is what let Groq's Llama retirement read as a rate limit.

    Both are needed because neither implies the other. A 404 is not exhaustion
    — `is_provider_exhausted` returns False for it, so a decommissioned model
    fell into the generic "judge unreachable" bucket alongside quota errors and
    was reported with quota advice.

    Accepts an exception or a plain string: callers on the eval path hold error
    TEXT read back from a stored scorecard row, not a live exception.
    """
    status = getattr(getattr(exc, "response", None), "status_code", None)
    msg = str(exc).lower()
    # 404 alone is not sufficient — a wrong base URL 404s too — but a 404 whose
    # body talks about the model is decisive.
    if status == 404 and "model" in msg:
        return True
    return any(m in msg for m in _MODEL_GONE_MARKERS)


def supports_streaming(provider: str) -> bool:
    """True when complete_stream can measure TTFT for this provider.

    Cloud-native adapters (vertex_ai / azure_openai / bedrock /
    huawei_modelarts) each have their own auth, URL and event envelope with
    no shared SSE surface, so they stay non-streaming for now — callers
    fall back to the non-streaming path rather than failing.
    """
    return provider in _STREAMING_PROVIDERS and not is_cloud_provider(provider)


def parse_stream_delta(provider: str, data: dict) -> Optional[str]:
    """Extract the incremental text from one decoded SSE `data:` event.

    Returns None for events that carry no text (keep-alives, message/
    content-block start and stop, usage-only deltas) — the caller skips
    those, so TTFT is timed from the first real TOKEN rather than from the
    first protocol frame, which is what the budget is meant to measure.
    """
    if provider == "anthropic":
        # Anthropic Messages streaming: message_start, content_block_start,
        # content_block_delta (the only text-bearing event), ping,
        # content_block_stop, message_delta, message_stop.
        if data.get("type") != "content_block_delta":
            return None
        delta = data.get("delta") or {}
        # text_delta carries prose; input_json_delta (tool use) carries no
        # assistant text and must not be counted as a first token.
        if delta.get("type") not in (None, "text_delta"):
            return None
        return delta.get("text") or None

    choices = data.get("choices") or [{}]
    return (choices[0].get("delta") or {}).get("content") or None


# ── Cloud-native providers (Vertex AI / Azure OpenAI / Bedrock / Huawei ModelArts) ──


class CloudProviderAdapter(Protocol):
    """One implementation per cloud-hosted provider. Each owns its own
    credential acquisition, URL templating, and request/response envelope —
    the three things that differ across SigV4 / OAuth2 / api-key+api-version
    / AK-SK auth and Bedrock/Vertex/Azure/ModelArts envelopes."""

    def build_request(
        self,
        model_id: str,
        messages: list[dict],
        cfg: dict,
        max_tokens: int,
        temperature: float,
    ) -> tuple[str, dict, dict]:
        """Returns (full_url, headers, body)."""
        ...

    def parse_response(self, data: dict) -> tuple[str, Optional[int], Optional[int]]:
        """Returns (text, input_tokens, output_tokens)."""
        ...


def _anthropic_temperature(temperature: float) -> float:
    """A temperature the Anthropic Messages API will accept.

    Its documented range is 0..1, OpenAI's is 0..2. Every caller in this repo
    asks for 0.0, 0.2 or 0.7, so the clamp is a guard rather than a behaviour
    change — but it is a LOUD one: silently sending a different number than the
    caller asked for is the kind of substitution this codebase refuses
    elsewhere, and silently sending an out-of-range one turns a working call
    into a 400.
    """
    if temperature > 1.0:
        logger.warning(
            "temperature=%.2f exceeds the Anthropic Messages API maximum of 1.0 — "
            "sending 1.0. OpenAI-shaped routes accept up to 2.0, so the same "
            "config behaves differently by provider.",
            temperature,
        )
        return 1.0
    return max(0.0, temperature)


def _anthropic_messages_body(
    messages: list[dict],
    max_tokens: int,
    temperature: float,
    anthropic_version: str,
) -> dict:
    """The Anthropic Messages body, for every route that speaks it.

    `anthropic_version` is the only thing that differed between the two
    versions of this that existed: Vertex sends `vertex-2023-10-16`, Bedrock
    `bedrock-2023-05-31`. Bedrock had its own inline copy of the system/user
    split for that one string — a near-duplicate the DRY levers ask to
    parameterise rather than clone, and the reason `temperature` had to be
    forgotten three separate times (here, in Bedrock's copy, and in
    build_request's direct branch) instead of once.

    It also took a `model_id` it never used: Vertex and Bedrock both carry the
    model in the URL, so the parameter was decoration.
    """
    system = "\n".join(m["content"] for m in messages if m["role"] == "system") or None
    user_messages = [m for m in messages if m["role"] != "system"]
    body: dict[str, Any] = {
        "anthropic_version": anthropic_version,
        "max_tokens": max_tokens,
        "messages": user_messages,
        # See build_request's anthropic branch: omitted on every Anthropic
        # route, so all of them ran at the provider's default of 1.0.
        "temperature": _anthropic_temperature(temperature),
    }
    if system:
        body["system"] = system
    return body


class VertexAIAdapter:
    """GCP Vertex AI — Gemini-on-Vertex (`publishers/google/models/...
    :generateContent`) is the default and recommended path: it's Google's
    own first-party model on its own platform, so it doesn't carry the
    region/availability uncertainty of a third-party model (e.g. Claude)
    being hosted on someone else's cloud, which rolls out per-region on its
    own schedule independent of the underlying Vertex AI region's GA status.
    Anthropic-on-Vertex (`publishers/anthropic/models/...:streamRawPredict`)
    is still supported (set `publisher: anthropic`) but deprioritized as
    the default for this provider — use it only if you specifically need
    Claude and have confirmed it's enabled for your project's region.

    Auth: OAuth2 service-account token via `google-auth`. Required
    models.yaml fields: `project`, `region` (defaults to `us-central1`),
    `publisher` (defaults to `google`, i.e. Gemini; set to `anthropic` for
    Claude-on-Vertex). Credentials resolved the standard google-auth way:
    `GOOGLE_APPLICATION_CREDENTIALS` env var pointing at a service-account
    JSON key, or any other `google.auth.default()`-supported source
    (workload identity, gcloud ADC, etc).

    Region note (verified live against a real Vertex AI project): GCP's
    GCC regions — `me-central1` (Doha, Qatar) and `me-central2` (Dammam,
    Saudi Arabia) — do NOT currently serve `gemini-2.5-flash` (confirmed
    404 "Publisher model ... was not found" on both). `us-central1`,
    `europe-west1`, `europe-west4`, and `asia-south1` were confirmed
    working. If GCC-region hosting is a hard requirement, verify model
    availability for that region first via a live call before overriding
    `region` — do not assume any specific GCC region serves a given model.
    """

    _cached_token: str | None = None
    _cached_token_expiry: float = 0.0

    def _get_access_token(self) -> str:
        if self._cached_token and time.time() < self._cached_token_expiry - 60:
            return self._cached_token
        import google.auth
        import google.auth.transport.requests

        credentials, _ = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        credentials.refresh(google.auth.transport.requests.Request())
        VertexAIAdapter._cached_token = credentials.token
        VertexAIAdapter._cached_token_expiry = (
            credentials.expiry.timestamp() if credentials.expiry else time.time() + 3600
        )
        if not VertexAIAdapter._cached_token:
            # `credentials.token` is Optional while this is declared `-> str`, so
            # a refresh that produced nothing would return None and the caller
            # would send `Bearer None` — a 401 that reads as bad credentials
            # rather than as a refresh that silently did not work.
            raise RuntimeError("Vertex AI credential refresh returned no access token")
        return VertexAIAdapter._cached_token

    _DEFAULT_URL_TEMPLATE_ANTHROPIC = (
        "https://{region}-aiplatform.googleapis.com/v1/projects/{project}"
        "/locations/{region}/publishers/anthropic/models/{model_id}:streamRawPredict"
    )
    _DEFAULT_URL_TEMPLATE_GENERIC = (
        "https://{region}-aiplatform.googleapis.com/v1/projects/{project}"
        "/locations/{region}/publishers/{publisher}/models/{model_id}:generateContent"
    )

    def build_request(
        self,
        model_id: str,
        messages: list[dict],
        cfg: dict,
        max_tokens: int,
        temperature: float,
    ) -> tuple[str, dict, dict]:
        project = os.path.expandvars(
            cfg["project"]
        )  # supports ${VAR} so a project id never has to be committed literally
        region = cfg.get(
            "region", "us-central1"
        )  # verified working; GCC regions confirmed NOT to serve gemini-2.5-flash, see class docstring
        publisher = cfg.get(
            "publisher", "google"
        )  # Gemini — first-party on Vertex, no cross-vendor rollout lag
        token = self._get_access_token()
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

        # `url_template` (models.yaml) overrides the default vendor URL shape
        # entirely — needed for a regional API variant, a private VPC
        # endpoint, or a proxy in front of the Vertex AI API. Falls back to
        # the standard public endpoint when not set.
        fmt = {
            "project": project,
            "region": region,
            "publisher": publisher,
            "model_id": model_id,
        }
        if publisher == "anthropic":
            url = cfg.get("url_template", self._DEFAULT_URL_TEMPLATE_ANTHROPIC).format(
                **fmt
            )
            body = _anthropic_messages_body(
                messages, max_tokens, temperature, "vertex-2023-10-16"
            )
        else:
            url = cfg.get("url_template", self._DEFAULT_URL_TEMPLATE_GENERIC).format(
                **fmt
            )
            body = {
                "contents": [
                    {
                        "role": "user" if m["role"] != "assistant" else "model",
                        "parts": [{"text": m["content"]}],
                    }
                    for m in messages
                    if m["role"] != "system"
                ],
                "generationConfig": {
                    "maxOutputTokens": max_tokens,
                    "temperature": temperature,
                },
            }
        return url, headers, body

    def parse_response(self, data: dict) -> tuple[str, Optional[int], Optional[int]]:
        if (
            "content" in data
        ):  # Anthropic-on-Vertex envelope mirrors the direct Messages API shape
            return parse_anthropic_completion(data)
        candidate = data["candidates"][0]
        text = candidate["content"]["parts"][0]["text"]
        usage = data.get("usageMetadata", {})
        return (
            text,
            usage.get("promptTokenCount", 0),
            usage.get("candidatesTokenCount", 0),
        )


class AzureOpenAIAdapter:
    """Azure OpenAI — same chat-completions body shape as direct OpenAI, but
    api-key header (not Bearer), deployment name in the URL path instead of
    a bare model id, and a required `api-version` query param.

    Required models.yaml fields: `resource` (the Azure OpenAI resource
    name), `deployment` (the deployment name — may differ from `id`),
    `api_version` (defaults to "2024-06-01"). `api_key_env` defaults to
    `AZURE_OPENAI_API_KEY`. Optional `url_template` overrides the URL shape
    entirely (e.g. for Azure Gov / sovereign cloud endpoints).
    """

    _DEFAULT_URL_TEMPLATE = (
        "https://{resource}.openai.azure.com/openai/deployments/{deployment}"
        "/chat/completions?api-version={api_version}"
    )

    def build_request(
        self,
        model_id: str,
        messages: list[dict],
        cfg: dict,
        max_tokens: int,
        temperature: float,
    ) -> tuple[str, dict, dict]:
        resource = cfg["resource"]
        deployment = cfg.get("deployment", model_id)
        api_version = cfg.get("api_version", "2024-06-01")
        api_key = os.environ.get(cfg.get("api_key_env", "AZURE_OPENAI_API_KEY"), "")

        url = cfg.get("url_template", self._DEFAULT_URL_TEMPLATE).format(
            resource=resource,
            deployment=deployment,
            api_version=api_version,
            model_id=model_id,
        )
        headers = {"api-key": api_key, "Content-Type": "application/json"}
        body = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        return url, headers, body

    def parse_response(self, data: dict) -> tuple[str, Optional[int], Optional[int]]:
        return parse_openai_completion(data)


class BedrockAdapter:
    """AWS Bedrock — SigV4-signed requests (AWS credentials via boto3/
    botocore, not a bearer token), `bedrock-runtime.{region}.amazonaws.com/
    model/{model-id}/invoke` URL shape, Anthropic-on-Bedrock envelope (its
    own shape, distinct from both the direct Anthropic API and
    Anthropic-on-Vertex).

    Required models.yaml fields: `region` (defaults to `us-east-1`, one of
    Bedrock's original/best-model-coverage regions). AWS credentials
    resolved the standard boto3 way (env vars, shared config file,
    instance/task role) — no per-model credential override, since Bedrock
    access is normally scoped at the IAM-role level, not per model.
    Optional `url_template` overrides the URL shape (e.g. a VPC interface
    endpoint instead of the public `bedrock-runtime` host) — the override
    still gets correctly SigV4-signed since signing happens against
    whatever URL is built, not a hard-coded host.

    GCC region note: AWS's GCC region is `me-central-1` (UAE/Dubai);
    `me-south-1` (Bahrain) is the other Middle East option. Neither has
    been verified to have Bedrock (or the specific foundation model you
    need) enabled — the live test that disproved the equivalent assumption
    for GCP Vertex AI's GCC regions (see VertexAIAdapter docstring) was not
    repeatable here for lack of AWS credentials in the test environment.
    Confirm via a live call before overriding `region` to a GCC value;
    do not assume it works by default.
    """

    _DEFAULT_URL_TEMPLATE = (
        "https://bedrock-runtime.{region}.amazonaws.com/model/{model_id}/invoke"
    )

    def build_request(
        self,
        model_id: str,
        messages: list[dict],
        cfg: dict,
        max_tokens: int,
        temperature: float,
    ) -> tuple[str, dict, dict]:
        import boto3
        from botocore.auth import SigV4Auth
        from botocore.awsrequest import AWSRequest

        region = cfg.get(
            "region", "us-east-1"
        )  # broad model coverage; GCC region unverified, see class docstring
        url = cfg.get("url_template", self._DEFAULT_URL_TEMPLATE).format(
            region=region, model_id=model_id
        )

        body = _anthropic_messages_body(
            messages, max_tokens, temperature, "bedrock-2023-05-31"
        )
        body_bytes = json.dumps(body).encode()

        session = boto3.Session()
        credentials = session.get_credentials()
        request = AWSRequest(
            method="POST",
            url=url,
            data=body_bytes,
            headers={"Content-Type": "application/json"},
        )
        SigV4Auth(credentials, "bedrock", region).add_auth(request)
        headers = dict(request.headers)
        return url, headers, body

    def parse_response(self, data: dict) -> tuple[str, Optional[int], Optional[int]]:
        return parse_anthropic_completion(data)


class HuaweiModelArtsAdapter:
    """Huawei Cloud ModelArts inference — AK/SK request signing (Huawei's
    own "SDK-HMAC-SHA256" scheme, conceptually similar to AWS SigV4 but a
    distinct canonical-request format), hitting a per-deployment custom
    inference endpoint domain.

    NOTE: this is the least-documented of the four cloud providers in
    English-language sources. The endpoint/signing shape below follows
    Huawei's published API Gateway signing algorithm structure but has not
    been validated against a live ModelArts inference endpoint — treat as
    a starting point to verify against current Huawei API docs and a real
    deployment, not as confirmed-correct.

    Required models.yaml fields: `endpoint` (the full custom inference
    endpoint host, e.g. `xxxxx.{region}.modelarts-infer.com`), `region`.
    Credentials: `HUAWEICLOUD_SDK_AK` / `HUAWEICLOUD_SDK_SK` env vars
    (access key / secret key). Optional `path_template` overrides the
    request path (e.g. a differently-versioned or custom inference route)
    — `endpoint` always supplies the host, since the signature covers the
    `Host` header and path together and they must agree.
    """

    _DEFAULT_PATH_TEMPLATE = "/v1/infers/{model_id}/chat/completions"

    def build_request(
        self,
        model_id: str,
        messages: list[dict],
        cfg: dict,
        max_tokens: int,
        temperature: float,
    ) -> tuple[str, dict, dict]:
        import hashlib
        import hmac
        from datetime import datetime, timezone

        endpoint = cfg["endpoint"]
        path = cfg.get("path_template", self._DEFAULT_PATH_TEMPLATE).format(
            model_id=model_id
        )
        url = f"https://{endpoint}{path}"
        body = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        body_bytes = json.dumps(body).encode()

        ak = os.environ.get("HUAWEICLOUD_SDK_AK", "")
        sk = os.environ.get("HUAWEICLOUD_SDK_SK", "")
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        content_sha256 = hashlib.sha256(body_bytes).hexdigest()
        canonical_request = "\n".join(
            [
                "POST",
                path,
                "",
                f"content-type:application/json\nhost:{endpoint}\nx-sdk-date:{timestamp}\n",
                "content-type;host;x-sdk-date",
                content_sha256,
            ]
        )
        string_to_sign = "\n".join(
            [
                "SDK-HMAC-SHA256",
                timestamp,
                hashlib.sha256(canonical_request.encode()).hexdigest(),
            ]
        )
        signature = hmac.new(
            sk.encode(), string_to_sign.encode(), hashlib.sha256
        ).hexdigest()
        headers = {
            "Content-Type": "application/json",
            "X-Sdk-Date": timestamp,
            "Authorization": (
                f"SDK-HMAC-SHA256 Access={ak}, "
                f"SignedHeaders=content-type;host;x-sdk-date, Signature={signature}"
            ),
            "Host": endpoint,
        }
        return url, headers, body

    def parse_response(self, data: dict) -> tuple[str, Optional[int], Optional[int]]:
        return parse_openai_completion(data)


_CLOUD_ADAPTERS: dict[str, CloudProviderAdapter] = {
    "vertex_ai": VertexAIAdapter(),
    "azure_openai": AzureOpenAIAdapter(),
    "bedrock": BedrockAdapter(),
    "huawei_modelarts": HuaweiModelArtsAdapter(),
}


def is_cloud_provider(provider: str) -> bool:
    return provider in _CLOUD_ADAPTERS


def get_cloud_adapter(provider: str) -> CloudProviderAdapter:
    try:
        return _CLOUD_ADAPTERS[provider]
    except KeyError:
        raise ValueError(
            f"Unknown cloud provider {provider!r}. Supported: {sorted(_CLOUD_ADAPTERS)}"
        ) from None


def build_cloud_request(
    provider: str,
    model_id: str,
    messages: list[dict],
    cfg: dict,
    max_tokens: int,
    temperature: float,
) -> tuple[str, dict, dict]:
    """Returns (full_url, headers, body) for a cloud-native provider."""
    return get_cloud_adapter(provider).build_request(
        model_id, messages, cfg, max_tokens, temperature
    )


def parse_cloud_response(provider: str, data: dict) -> tuple[str, Optional[int], Optional[int]]:
    """Returns (text, input_tokens, output_tokens) for a cloud-native provider."""
    return get_cloud_adapter(provider).parse_response(data)
