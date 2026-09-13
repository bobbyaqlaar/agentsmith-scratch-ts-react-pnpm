"""
runtime/llm_gateway.py — Production LLM gateway.

Centralised routing for all production agent LLM calls.
Replaces cost_router.py for production use.

Responsibilities:
  - Accurate per-model pricing (from models.yaml, not blended estimates)
  - Per-tenant budget enforcement (reads from budget store — Postgres or Redis)
  - Degrade ladder on budget/quota breach
  - Audit trail: every call recorded as span attributes
  - Idempotency: duplicate calls short-circuited via idempotency.py

Degrade ladder (on budget breach or provider throttle):
  1. Throttle   — exponential backoff on request rate
  2. Downgrade  — route to cheaper tier in models.yaml
  3. Queue      — delay tasks with exponential backoff
  4. Local      — switch to Ollama if OLLAMA_BASE_URL is configured
  5. Alert      — Ops Portal + Slack/Teams

Workers MUST NOT import cost_router.py directly.

See SPECS.md §29 for full specification.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

from runtime.config import resolve  # noqa: E402
from runtime.moderation import (  # noqa: F401,E402 — re-export block, deliberately after the logger
    ModerationBlockedError,
    ModerationHookRequiredError,
    ModerationResult,
    apply_output_moderation,
    get_output_moderator,
    register_output_moderator,
    reset_output_moderator,
    resolve_mode as resolve_moderation_mode,
)
@dataclass
class CompletionResult:
    """Result of a gateway-routed completion."""

    text: str
    model_used: str
    # None = the provider reported no usage. Not 0 — a caller summing these
    # across runs must be able to see a gap rather than a confident zero, and
    # `cost_usd` is a ceiling rather than a measurement whenever they are None.
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    cost_usd: float
    degrade_tier: Optional[str] = None  # None = nominal; "downgrade" | "local" | etc.
    ttft_ms: Optional[float] = None
    # Guardrail evidence (TestbedFeedback-2026-07-21 G3). The gateway
    # already computes these while scrubbing; before they were exposed here
    # they only reached logs and span attributes, so an app that must
    # record WHAT was redacted in its own decision record — every PDPL /
    # GDPR decision-path app, the exact use case this framework markets —
    # had to re-run the scrub itself and pay for it twice.
    # {"emirates_id": 1, "card": 1, ...}; empty when the guardrail is off.
    guardrail_counts: dict[str, int] = field(default_factory=dict)
    # Prompt-guard heuristics that fired but did NOT block — i.e. populated
    # under PROMPT_GUARD=warn, the observe-first tier added in
    # TestbedFeedback-2026-07-21 G9. Under the blocking modes (`default`,
    # `strict`) a flagged prompt raises instead, so this stays empty on any
    # result they return. Use it to measure a guard rollout against real
    # traffic before switching that tenant to enforcing.
    prompt_guard_reasons: list[str] = field(default_factory=list)


class BudgetExceededError(RuntimeError):
    """Raised when the degrade ladder is exhausted (halt + alert tier, §29)."""


# ── Model registry (§29 Model Registry) ───────────────────────────────────────


def _repo_root() -> Path:
    """Delegates to runtime.config.repo_root — see there for why the marker is
    `.agenticframework` OR `.git`, not `.git` alone.

    There were FIVE of these in three disagreeing variants. A tenant nested
    inside a parent git repo resolved to the parent under the `.git`-only ones
    and to the tenant under the others, so `tenant.yaml` and `models.yaml` were
    loaded from different directories in the same process.
    """
    from runtime.config import repo_root

    return repo_root()


_FRAMEWORK_MODELS_YAML = Path(__file__).resolve().parent / "models.yaml"


def _load_yaml(path: Path) -> dict:
    try:
        import yaml  # type: ignore

        with path.open() as fh:
            return yaml.safe_load(fh) or {}
    except Exception as exc:
        # A malformed registry returned {} and every role fell back to the
        # framework's defaults — a tenant's whole routing table replaced by
        # someone else's, from a YAML typo, with nothing said. The fallback is
        # still right (the gateway must not refuse to start over a comment), but
        # it is a different state from "no registry here".
        logger.error(
            "model registry NOT loaded from %s (%s) — falling back to framework "
            "defaults, so this tenant's routes, costs and degrade ladder are NOT "
            "in effect.",
            path,
            exc,
        )
        return {}


def _active_profile_name(doc: dict) -> str:
    """Which profile a catalog-style registry binds roles from.

    AI_STACK_MODE is what `ai-mode-local` / `ai-mode-hybrid` already export, so
    the shell switch that previously only printed a banner now actually selects
    routes. An explicit AGENT_MODEL_PROFILE wins over it, for selecting a
    profile without changing the machine's mode.
    """
    explicit = os.environ.get("AGENT_MODEL_PROFILE", "").strip()
    if explicit:
        return explicit
    mode = os.environ.get("AI_STACK_MODE", "").strip()
    profiles = doc.get("profiles") or {}
    if mode and mode in profiles:
        return mode
    return str(doc.get("default_profile") or "local")


def _roles_from_doc(doc: dict) -> dict:
    """Flatten either registry shape into {role: cfg}.

    Two shapes are supported on purpose:

      models:   {role: {id, provider, ...}}        # flat — the original
      catalog:  {alias: {id, provider, ...}}       # model REFERENCES
      profiles: {name: {role: alias | {use: alias, ...}}}

    Catalog+profiles separates two things the flat shape conflated: WHICH
    models exist (including closed-weight ones you are not currently routing
    to) and WHICH ROLE uses which. Under the flat shape a cloud model could
    only be present by being wired in, so the framework default had every
    closed-weight entry commented out — visible to a reader, invisible to the
    code, and impossible to switch to without editing YAML.

    Both shapes resolve to the SAME flat {role: cfg} dict, so llm_gateway,
    cost_router and scripts/_shared all consume it unchanged. That is the
    point of doing the resolution here rather than at each call site.
    """
    flat = {role: dict(cfg) for role, cfg in (doc.get("models") or {}).items()}

    catalog = doc.get("catalog") or {}
    profiles = doc.get("profiles") or {}
    if not catalog and not profiles:
        return flat

    bindings = (profiles.get(_active_profile_name(doc)) or {}) if profiles else {}
    for role, binding in bindings.items():
        extras: dict = {}
        if isinstance(binding, dict):
            alias = binding.get("use") or binding.get("model")
            extras = {k: v for k, v in binding.items() if k not in ("use", "model")}
        else:
            alias = binding
        entry = catalog.get(alias)
        if entry is None:
            raise ValueError(
                f"models.yaml: role {role!r} binds to {alias!r}, which is not in "
                f"`catalog`. Known: {sorted(catalog)}"
            )
        # `id` defaults to the catalog KEY, so an entry only needs an explicit
        # id when the provider's name for it differs from the alias — which is
        # the normal case for OpenRouter ("anthropic/claude-sonnet-4.5").
        cfg = {"id": alias, **entry, **extras}
        flat[role] = cfg
    return flat


def degrade_chain(models: dict, model_hint: str) -> list[str]:
    """[model_hint, ...downgrade targets...] following models.yaml degrade_to links.

    Module-level rather than a gateway method because the ladder is also what
    residency depends on: a role can be pinned to an in-border endpoint and
    still leave the jurisdiction on its first overload, by degrading to
    something that is not. The sovereign control (SEC-SOV-001) walks the chain
    with THIS function so it cannot disagree with what the gateway will
    actually do at runtime — a second implementation would be a residency
    check that is correct about a ladder nobody climbs.

    Cycles terminate: a role already seen ends the walk.
    """
    chain = [model_hint]
    current = model_hint
    seen = {model_hint}
    while True:
        nxt = models.get(current, {}).get("degrade_to")
        if not nxt or nxt in seen:
            break
        chain.append(nxt)
        seen.add(nxt)
        current = nxt
    return chain


def load_model_registry() -> dict:
    """
    Load the model registry: framework defaults from runtime/models.yaml,
    overridden by a tenant repo's own models.yaml (if present), overridden
    again by `.agenticframework/tenant.yaml` -> gateway.routing_overrides
    (a per-role model id shorthand, §29).

    Either file may use the flat `models:` shape or `catalog:` + `profiles:`;
    both flatten to {role: cfg} before merging, so the two can be mixed (a
    catalog-style framework default under a flat tenant override, which is
    exactly the current KYC Sentinel arrangement).
    """
    registry: dict = {}
    for role, cfg in _roles_from_doc(_load_yaml(_FRAMEWORK_MODELS_YAML)).items():
        registry[role] = dict(cfg)

    root = _repo_root()
    tenant_models_path = root / "models.yaml"
    if tenant_models_path.exists():
        for role, cfg in _roles_from_doc(_load_yaml(tenant_models_path)).items():
            base = registry.get(role, {})
            # Same model id => the tenant is tweaking fields on the framework's
            # route, so merge. DIFFERENT id => it is a different model, and
            # nothing about the old one carries over. A shallow merge here used
            # to leak the framework entry's fields onto a model they do not
            # describe, silently and in three ways at once:
            #
            #   * `endpoint` — KYC Sentinel's judge (claude-opus-4-8, anthropic)
            #     inherited the framework judge's "${OLLAMA_BASE_URL}/v1", so
            #     the gateway posted Claude requests at the Ollama host. The
            #     eval path escaped only because cost_router used to ignore
            #     `endpoint` entirely.
            #   * `cost_per_*_token` — inheriting a free local tier's zeros
            #     makes a frontier model read as costless to budget reservation
            #     and the spend cap.
            #   * `degrade_to` — deleting the key from the tenant file did NOT
            #     remove the behaviour, because the framework's value showed
            #     through. A judge role explicitly given no fallback still
            #     degraded, to whatever the framework's judge points at.
            #
            # Replacing is also what makes `degrade_to` removable at all: with
            # a merge there is no way to express "no fallback" short of
            # `degrade_to: null`, and a role that omits the key gets one anyway.
            registry[role] = (
                {**base, **cfg}
                if base.get("id") == cfg.get("id")
                else dict(cfg)
            )

    tenant_yaml_path = root / ".agenticframework" / "tenant.yaml"
    if tenant_yaml_path.exists():
        tenant_cfg = _load_yaml(tenant_yaml_path)
        overrides = (tenant_cfg.get("gateway") or {}).get("routing_overrides") or {}
        for role, model_id in overrides.items():
            registry.setdefault(role, {})
            registry[role]["id"] = model_id

    return registry


# ── Budget store (Postgres or Redis backend, §25/§29) ─────────────────────────


def _current_period() -> str:
    """ "YYYY-MM" in UTC — pinned explicitly rather than via bare
    time.strftime("%Y-%m"), which uses the server's LOCAL timezone.
    portal/lib/cost.ts derives the same period via
    `new Date().toISOString().slice(0, 7)`, which is always UTC; a worker
    running in a non-UTC server timezone could otherwise disagree with the
    portal for several hours around a month boundary, putting spend in the
    "wrong" month from the portal's point of view (Product_Archive.md 4.15).
    """
    return time.strftime("%Y-%m", time.gmtime())


@dataclass
class BudgetStatus:
    tenant_id: str
    spent_usd: float
    cap_usd: float
    period_start: str

    @property
    def remaining_usd(self) -> float:
        return max(0.0, self.cap_usd - self.spent_usd)

    @property
    def breached(self) -> bool:
        return self.spent_usd >= self.cap_usd


class _BudgetBackend:
    def get_spend(self, tenant_id: str) -> float:
        raise NotImplementedError

    def add_spend(self, tenant_id: str, amount_usd: float) -> float:
        raise NotImplementedError

    def try_reserve(self, tenant_id: str, amount_usd: float, cap_usd: float) -> bool:
        """Atomically add amount_usd to spend IF the result would not exceed
        cap_usd, in one indivisible operation. Returns True if reserved (the
        amount has already been added to spend) or False if it would have
        breached the cap (nothing was added).

        This exists because the old pattern — a separate get_budget_status()
        read, then an add_spend() write only after the (slow, variable-cost)
        LLM call returns — left a window where N concurrent calls for the
        same tenant could all read "not breached" before any of them
        recorded spend, letting the combined cost of every in-flight call
        blow through the monthly cap (Product_Archive.md 2.1). Callers
        reserve an upper-bound cost estimate via try_reserve() before
        invoking the provider, then reconcile the estimate vs. actual cost
        afterward via add_spend()'s signed delta.
        """
        raise NotImplementedError


class _MemoryBudgetBackend(_BudgetBackend):
    """Single-process budget tracking. Suitable for dev/CI; not for multi-worker prod fleets.

    KEYED BY (tenant, period), like the other two. It was keyed by tenant
    alone, so the monthly cap never reset: Redis keys on
    `budget:{tenant}:{period}` and Postgres has `PRIMARY KEY (tenant_id,
    period)`, and this one — the DEFAULT, since `BUDGET_BACKEND` is unset in
    every deployment that has not chosen otherwise — accumulated for the life
    of the process. A worker alive across the 1st carried December's spend into
    January, and `get_budget_status()` reported that lifetime total next to a
    `period_start` naming the current month. Eventually the cap is "breached"
    by spend from a month that is over.
    """

    def __init__(self) -> None:
        self._spend: dict[tuple[str, str], float] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _key(tenant_id: str) -> tuple[str, str]:
        return (tenant_id, _current_period())

    def get_spend(self, tenant_id: str) -> float:
        with self._lock:
            return self._spend.get(self._key(tenant_id), 0.0)

    def add_spend(self, tenant_id: str, amount_usd: float) -> float:
        with self._lock:
            key = self._key(tenant_id)
            self._spend[key] = self._spend.get(key, 0.0) + amount_usd
            return self._spend[key]

    def try_reserve(self, tenant_id: str, amount_usd: float, cap_usd: float) -> bool:
        with self._lock:
            key = self._key(tenant_id)
            current = self._spend.get(key, 0.0)
            if current + amount_usd > cap_usd:
                return False
            self._spend[key] = current + amount_usd
            return True


class _RedisBudgetBackend(_BudgetBackend):
    def __init__(self) -> None:
        import redis  # type: ignore

        self._client = redis.from_url(os.environ["REDIS_URL"])

    def _key(self, tenant_id: str) -> str:
        period = _current_period()
        return f"agenticframework:budget:{tenant_id}:{period}"

    def get_spend(self, tenant_id: str) -> float:
        val = self._client.get(self._key(tenant_id))
        return float(val) if val else 0.0

    def add_spend(self, tenant_id: str, amount_usd: float) -> float:
        key = self._key(tenant_id)
        new_total = self._client.incrbyfloat(key, amount_usd)
        self._client.expire(key, 40 * 86400)  # budgets are monthly; expire stale keys
        return float(new_total)

    def try_reserve(self, tenant_id: str, amount_usd: float, cap_usd: float) -> bool:
        # INCRBYFLOAT is atomic, but "increment, then check, then maybe
        # undo" is not a single atomic step — between two concurrent
        # INCRBYFLOATs both can observe a total over cap and both roll back,
        # or (the actually dangerous case) both can observe under cap before
        # either's increment is visible to the other... except INCRBYFLOAT
        # itself serializes on the key (Redis commands on one key run one at
        # a time), so the *increment* ordering is always consistent — the
        # post-hoc compensating DECRBYFLOAT below is what makes the
        # reserve-or-release atomic *with respect to the cap*, not the
        # increment itself.
        key = self._key(tenant_id)
        new_total = float(self._client.incrbyfloat(key, amount_usd))
        if new_total > cap_usd:
            self._client.incrbyfloat(key, -amount_usd)
            return False
        self._client.expire(key, 40 * 86400)
        return True


_BUDGET_DDL = """
    CREATE TABLE IF NOT EXISTS llm_gateway_budget (
        tenant_id TEXT NOT NULL,
        period TEXT NOT NULL,
        spent_usd DOUBLE PRECISION NOT NULL DEFAULT 0,
        PRIMARY KEY (tenant_id, period)
    )
"""


class _PostgresBudgetBackend(_BudgetBackend):
    def __init__(self) -> None:
        self._dsn = os.environ["DATABASE_URL"]
        # Once per DSN per process. A gateway is constructed per activity, so
        # this was a DDL round-trip on the hot path of every workflow step.
        from runtime.pg_pool import ensure_schema

        ensure_schema(self._dsn, _BUDGET_DDL, key="llm_gateway_budget")

    def _connect(self):
        # Pooled since ReviewFindings-2026-07-18 C1: try_reserve + add_spend
        # sit inside every gateway LLM call — a fresh TCP+auth handshake per
        # operation was the hottest avoidable cost in the runtime. The
        # returned object's .close() releases back to the pool, so the
        # existing `finally: conn.close()` call sites stay correct as-is.
        from runtime.pg_pool import connect as pg_connect
        return pg_connect(self._dsn)

    def _period(self) -> str:
        return _current_period()

    def get_spend(self, tenant_id: str) -> float:
        conn = self._connect()
        try:
            with conn, conn.cursor() as cur:
                cur.execute(
                    "SELECT spent_usd FROM llm_gateway_budget WHERE tenant_id = %s AND period = %s",
                    (tenant_id, self._period()),
                )
                row = cur.fetchone()
                return float(row[0]) if row else 0.0
        finally:
            conn.close()

    def add_spend(self, tenant_id: str, amount_usd: float) -> float:
        conn = self._connect()
        try:
            with conn, conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO llm_gateway_budget (tenant_id, period, spent_usd)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (tenant_id, period)
                    DO UPDATE SET spent_usd = llm_gateway_budget.spent_usd + EXCLUDED.spent_usd
                    RETURNING spent_usd
                    """,
                    (tenant_id, self._period(), amount_usd),
                )
                new_total = cur.fetchone()[0]
                return float(new_total)
        finally:
            conn.close()

    def try_reserve(self, tenant_id: str, amount_usd: float, cap_usd: float) -> bool:
        # Single atomic statement: the row is inserted/updated and the cap
        # check happens in the same WHERE clause Postgres evaluates under
        # the row lock taken for the UPDATE, so no other transaction's
        # concurrent reserve on the same (tenant_id, period) can interleave
        # between "check" and "act" the way the old read-then-write did.
        conn = self._connect()
        try:
            with conn, conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO llm_gateway_budget (tenant_id, period, spent_usd)
                    VALUES (%s, %s, 0)
                    ON CONFLICT (tenant_id, period) DO NOTHING
                    """,
                    (tenant_id, self._period()),
                )
                cur.execute(
                    """
                    UPDATE llm_gateway_budget
                    SET spent_usd = spent_usd + %s
                    WHERE tenant_id = %s AND period = %s AND spent_usd + %s <= %s
                    RETURNING spent_usd
                    """,
                    (amount_usd, tenant_id, self._period(), amount_usd, cap_usd),
                )
                row = cur.fetchone()
                return row is not None
        finally:
            conn.close()


def _make_budget_backend() -> _BudgetBackend:
    from runtime.environment import env_choice, warn_degraded_default

    # `BUDGET_BACKEND=""` used to raise here. That is what a k8s manifest
    # produces for a declared-but-unset input, and it crashed gateway
    # construction over a variable nobody had set. env_choice treats empty as
    # unset; a genuine typo still raises, because a misspelled backend must not
    # resolve to a default the operator did not choose.
    backend = env_choice(
        "BUDGET_BACKEND", default="memory", allowed=("memory", "redis", "postgres")
    )
    if backend == "redis":
        return _RedisBudgetBackend()
    if backend == "postgres":
        return _PostgresBudgetBackend()

    # The spend cap is per-process on this backend, and it is the default. A
    # fleet of N workers therefore enforces N caps: a $150 monthly limit
    # becomes $150 PER WORKER, with every worker correctly reporting itself
    # under budget. The class docstring has said "not for multi-worker prod
    # fleets" all along; nothing said it where an operator would see it.
    warn_degraded_default(
        "budget-backend-memory",
        "BUDGET_BACKEND is unset: the spend cap is tracked IN PROCESS. Every "
        "worker enforces its own copy of the cap, so a fleet of N workers can "
        "spend N times it. Set BUDGET_BACKEND=redis or =postgres for a shared "
        "ledger.",
    )
    return _MemoryBudgetBackend()


# ── Gateway ───────────────────────────────────────────────────────────────────


class LLMGateway:
    """
    Production LLM gateway. Instantiate once per worker process.

    Usage:
        gateway = LLMGateway(tenant_id="acme")
        result = await gateway.complete(
            prompt=messages,
            model_hint="developer",
            workflow_id="wf-oil-0042",
            idempotency_key="sha256:...",
        )
    """

    def __init__(
        self, tenant_id: Optional[str] = None, budget_cap_usd: Optional[float] = None
    ) -> None:
        # Optional now: `.agenticframework/tenant.yaml` has declared `tenant.id`
        # since the scaffold shipped and nothing read it — this very method
        # loads that file for `gateway.routing_overrides` and walked past the
        # id — so every caller supplied its own. KYC Sentinel carried the same
        # string in two places as a result. resolve_tenant_id() reads the
        # declaration, and RAISES rather than defaulting: this id partitions
        # the budget ledger below, the audit log and cross-tenant isolation.
        from runtime.tenancy import resolve_tenant_id

        self.tenant_id = resolve_tenant_id(tenant_id)
        self.models = load_model_registry()
        # `budget.monthly_usd_cap` has been declared in tenant.yaml since the
        # scaffold shipped and was read by nothing: KYC declared $5 while this
        # line enforced $150 whenever AGENT_MONTHLY_USD_CAP was unset — and it
        # IS unset in production, because .env is not deployed to Cloud Run. A
        # 30x gap between the policy on file and the one in force.
        self.budget_cap_usd = (
            resolve(
                "budget.monthly_usd_cap",
                explicit=budget_cap_usd,
                env_var="AGENT_MONTHLY_USD_CAP",
                default=150.0,
                cast=float,
            )
        )
        self._budget = _make_budget_backend()
        self._idempotency = self._make_idempotency_store()

    @staticmethod
    def _make_idempotency_store():
        try:
            from runtime.idempotency import IdempotencyStore

            return IdempotencyStore()
        except Exception as exc:
            # Missing REDIS_URL/DATABASE_URL, backend lib not installed, etc.
            # Degrades gracefully to "no idempotency" (duplicate-call
            # suppression simply doesn't happen) rather than failing gateway
            # construction — but logged, not silently invisible, since this
            # now means a real backend failed to initialize, not "not
            # implemented yet".
            logger.warning(
                "idempotency store unavailable, duplicate-call suppression disabled: %s",
                exc,
            )
            return None

    # ── Budget ────────────────────────────────────────────────────────────────

    def get_budget_status(self) -> BudgetStatus:
        """Return current budget status for this tenant."""
        spent = self._budget.get_spend(self.tenant_id)
        return BudgetStatus(
            tenant_id=self.tenant_id,
            spent_usd=spent,
            cap_usd=self.budget_cap_usd,
            period_start=f"{_current_period()}-01",
        )

    # ── Degrade ladder (§29) ─────────────────────────────────────────────────

    def _degrade_chain(self, model_hint: str) -> list[str]:
        """[model_hint, ...downgrade targets...] following models.yaml degrade_to links."""
        return degrade_chain(self.models, model_hint)

    @staticmethod
    def _is_free_tier(cfg: dict) -> bool:
        return (
            cfg.get("provider") == "ollama" or cfg.get("cost_per_input_token", 1) == 0
        )

    @staticmethod
    def _is_provider_exhausted(exc: Exception) -> bool:
        """True when the provider itself is unavailable for this key/tier — no point
        retrying; degrade to the next tier instead.  Covers billing, quota, and
        auth errors that will not resolve on their own.

        The markers now live in `runtime/provider_dispatch.py` so the eval path
        (`scripts/cost_router.py`) classifies failures identically — it reports
        exhaustion rather than degrading, but "is this exhaustion?" must have
        one answer. Kept as a method because the degrade loop below and its
        tests both call it.
        """
        from runtime.provider_dispatch import is_provider_exhausted

        return is_provider_exhausted(exc)

    def _resolve_role(
        self, model_hint: str, budget: BudgetStatus
    ) -> tuple[str, Optional[str]]:
        """Walk the degrade ladder. Returns (role, degrade_tier); degrade_tier is None at full strength."""
        if not budget.breached:
            return model_hint, None

        hint_cfg = self.models.get(model_hint, {})
        if self._is_free_tier(hint_cfg):
            # Already on the cheapest/local tier — nothing left to degrade to, and
            # using it doesn't add spend, so the breach doesn't block it.
            return model_hint, None

        chain = self._degrade_chain(model_hint)
        if len(chain) < 2:
            raise BudgetExceededError(
                f"tenant={self.tenant_id} budget exhausted (${budget.spent_usd:.2f}/${budget.cap_usd:.2f}) "
                f"and no cheaper tier available below {model_hint!r}. Halting (alert tier)."
            )

        # Walk the WHOLE chain to the first free tier, not one rung
        # (TestbedFeedback-2026-07-21 G2). Taking only chain[1] meant a
        # caller always asking for the top role degraded to the next PAID
        # tier and then hard-failed its reservation — rung 4 of the
        # documented ladder ("Local — switch to Ollama") was unreachable
        # whenever a paid tier sat between the caller's role and the local
        # one, which is the normal shape of a cost ladder. The budget is
        # already breached here, so a paid rung only buys one more call
        # before failing; a free rung keeps the tenant serving.
        for candidate in chain[1:]:
            cfg = self.models.get(candidate)
            if not cfg:
                continue
            if self._is_free_tier(cfg):
                return candidate, "local"

        # No free tier anywhere in the chain — fall back to the next
        # configured rung (cheaper than the caller's, may still fail its
        # reservation) rather than refusing to serve at all.
        for candidate in chain[1:]:
            if self.models.get(candidate):
                return candidate, "downgrade"

        raise BudgetExceededError(
            f"tenant={self.tenant_id} budget exhausted (${budget.spent_usd:.2f}/${budget.cap_usd:.2f}) "
            f"and no configured tier below {model_hint!r}. Halting (alert tier)."
        )

    # ── Run status reporting (Ops Portal, Product_Archive.md P2a) ─────────

    # Set the first time a run-status POST fails, so the warning below is
    # emitted once per process rather than once per LLM call.
    _run_status_failure_logged = False

    def _report_run_status(
        self,
        run_id: str,
        status: str,
        workflow_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        error_summary: Optional[str] = None,
        *,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        cost_usd: Optional[float] = None,
    ) -> None:
        """Best-effort POST to the Ops Portal's run-status ingest endpoint —
        gated on OPS_PORTAL_URL being set, fails open (logs, never raises)
        on any error. Same philosophy as every other runtime/-to-portal
        call in this codebase (e.g. _ai_audit_log_event in
        install-ai-stack.sh): never let optional observability infra FAIL the
        actual LLM call.

        It can still DELAY one, and the previous version of this sentence said
        "block or fail", which was half true. This is a synchronous POST called
        twice per gateway call — once before the provider request, once after —
        so a portal that is merely slow adds its latency to every call, and the
        first of the two delays the provider request itself. The timeout below
        is what bounds that, and it is deliberately short: this is telemetry,
        and a second of it is already too much to spend on a call that works.

        Making it asynchronous is the real fix and is a decision, not a
        cleanup: it needs a thread or a queue, a shutdown path, and tolerance
        for the two reports arriving out of order — which the portal's ingest
        upsert now has, since it refuses to let a late `running` un-finish a
        run.

        workflow_id is the grouping key portal/lib/runStatus.ts uses to
        aggregate multiple gateway calls within one workflow run (it was
        previously dropped here despite being accepted by the ingest route
        and the agent_runs.workflow_id column — every row landed with
        workflow_id=NULL regardless of what the caller passed to
        complete()).
        """
        ops_portal_url = os.environ.get("OPS_PORTAL_URL")
        sync_token = os.environ.get("OPS_PORTAL_SYNC_TOKEN")
        if not ops_portal_url or not sync_token:
            return
        try:
            import httpx

            # traceparent, so the portal's work joins THIS trace instead of
            # starting its own. Without it the worker and the portal were two
            # unconnected traces and "follow the request across services"
            # stopped at the process boundary.
            from runtime.tracing import current_trace_id, inject_context
            from runtime.version import framework_version

            httpx.post(
                f"{ops_portal_url.rstrip('/')}/api/runs/ingest",
                json={
                    "tenantId": self.tenant_id,
                    "runId": run_id,
                    "workflowId": workflow_id,
                    "status": status,
                    # Defaults to the ACTIVE trace. This argument has existed
                    # since the method was written and not one of its nine call
                    # sites ever passed it, so agent_runs.trace_id was NULL for
                    # every run ever recorded and the portal's trace link had
                    # nothing to link to.
                    "traceId": trace_id or current_trace_id(),
                    "errorSummary": error_summary,
                    # Omitted (null) rather than zeroed when the provider
                    # reported no usage. The ingest route and the column are
                    # both nullable so "not reported" stays distinguishable
                    # from "used nothing" all the way to the dashboard.
                    "inputTokens": input_tokens,
                    "outputTokens": output_tokens,
                    "costUsd": cost_usd,
                    # WHICH AGENTSMITH wrote this row. The portal is operated
                    # by IT and the tenants by the business, on independent
                    # release cadences, so the portal is always looking at a
                    # fleet spanning several framework versions — and a version
                    # decides what fields a tenant can populate at all. Without
                    # it, a NULL cost from a version that never reported cost
                    # and a NULL cost from a broken current tenant are the same
                    # cell. See portal/lib/wireContract.ts.
                    "frameworkVersion": framework_version(),
                },
                headers=inject_context({"Authorization": f"Bearer {sync_token}"}),
                # Was a flat 5s, twice per call: a slow portal could add ten
                # seconds to an LLM call for the sake of a status row. Split so
                # an unreachable host fails fast while a reachable-but-busy one
                # still gets a moment to answer.
                timeout=httpx.Timeout(connect=1.0, read=2.0, write=2.0, pool=1.0),
            )
        except Exception as exc:
            # WARNING once per process, DEBUG after. This was DEBUG-only, so a
            # portal that had stopped accepting reports — a rotated token, a
            # DNS change, a missing dependency — went unnoticed at every log
            # level anyone actually runs, and `agent_runs` simply stopped
            # filling. Once, because it sits on the hot path of every call and
            # a per-call warning would be its own outage.
            if not LLMGateway._run_status_failure_logged:
                LLMGateway._run_status_failure_logged = True
                logger.warning(
                    "run-status reporting to the Ops Portal is failing (further "
                    "occurrences at DEBUG) tenant=%s run_id=%s: %s",
                    self.tenant_id,
                    run_id,
                    exc,
                )
            else:
                logger.debug(
                    "run-status report failed tenant=%s run_id=%s: %s",
                    self.tenant_id,
                    run_id,
                    exc,
                )

    # ── Span attribute recording (§15, §29) ──────────────────────────────────

    def _record_span_attributes(
        self,
        role: str,
        model_id: str,
        degrade_tier: Optional[str],
        workflow_id: Optional[str],
        cost_usd: float,
        ttft_ms: Optional[float] = None,
        *,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        cost_estimated: bool = False,
        started_ns: Optional[int] = None,
        messages: Any = None,
        prompt_template_id: Optional[str] = None,
        outcome: Optional[str] = None,
    ) -> None:
        """Record gateway facts: onto the active span, or onto one we create.

        `is_recording()`, not `is None`. `get_current_span()` never returns
        None — with no active span it returns a NonRecordingSpan whose
        `set_attribute` is a silent no-op, so the old `if span is None: return`
        guard never fired and every attribute below was dropped without a
        signal on any call path not already inside an `agent_span`. That is the
        whole of `tenant.id`, `llm.model_name`, cost and TTFT, silently absent.
        `_record_guardrail_attributes` a few hundred lines down already had the
        correct check; this is its sibling.

        Fixing the guard alone would NOT have changed behaviour — a no-op write
        and an early return drop the attributes equally. What changes it is the
        fallback below: with no parent span, the gateway now emits its own,
        so an LLM call is never simply missing from the trace.

        `started_ns` is required for that fallback and only for it. Without a
        start time the span would be created at report time and read as
        instantaneous, which is worse than absent — it would corrupt every
        latency percentile derived from it. No start time, no synthetic span.
        """
        try:
            from opentelemetry import trace

            span = trace.get_current_span()
            if span is not None and span.is_recording():
                self._stamp_llm_span(
                    span, role, model_id, degrade_tier, workflow_id, cost_usd,
                    ttft_ms, input_tokens, output_tokens, cost_estimated,
                    messages, prompt_template_id,
                )
                self._emit_call_metrics(
                    role, model_id, degrade_tier, cost_usd, ttft_ms,
                    input_tokens, output_tokens, outcome,
                )
                return

            # NOTHING RECORDING — emit our own span rather than dropping the call.
            #
            # `get_current_span()` never returns None; with no active span it
            # returns a NonRecordingSpan whose set_attribute is a silent no-op.
            # So the previous `if span is None: return` never fired and every
            # attribute below went nowhere — not a crash, just an LLM call
            # absent from the trace entirely whenever the tenant had not
            # wrapped it in an `agent_span`. Model, cost, tokens and latency,
            # gone, on the one operation an LLM app most needs to see.
            #
            # `record_tool_call` deliberately declines to do this, because a
            # step makes several tool calls and lone roots would be noise. An
            # LLM call is the opposite case: one per step, and the unit every
            # dashboard is keyed on. A root span here is the standard shape
            # every provider SDK instrumentation emits.
            #
            # start_time makes the duration real. Created at report time
            # without it, the span would read as instantaneous and quietly
            # corrupt every latency percentile computed from it.
            if started_ns is None:
                return
            tracer = trace.get_tracer("agentsmith.runtime")
            span = tracer.start_span(f"llm.{role}", start_time=started_ns)
            try:
                span.set_attribute("llm.gateway.span_source", "gateway")
                self._stamp_llm_span(
                    span, role, model_id, degrade_tier, workflow_id, cost_usd,
                    ttft_ms, input_tokens, output_tokens, cost_estimated,
                    messages, prompt_template_id,
                )
                self._emit_call_metrics(
                    role, model_id, degrade_tier, cost_usd, ttft_ms,
                    input_tokens, output_tokens, outcome,
                )
            finally:
                span.end()
        except Exception:  # fail-open: tracing must never break the actual LLM call
            pass

    def _emit_call_metrics(
        self,
        role: str,
        model_id: str,
        degrade_tier: Optional[str],
        cost_usd: float,
        ttft_ms: Optional[float],
        input_tokens: Optional[int],
        output_tokens: Optional[int],
        outcome: Optional[str] = None,
    ) -> None:
        """Counters and histograms alongside the span.

        `outcome` overrides the success/degraded derivation. A moderation block
        is a real outcome of a real call — it was made and paid for — and
        deriving the dimension from `degrade_tier` alone could only ever say
        "success" for it.

        Spans answer "what happened in this request"; they are the wrong
        instrument for "what fraction of requests failed", which is sampled,
        expensive to scan and grows with traffic. Both, not either.
        """
        from runtime.metrics import record_llm_call

        record_llm_call(
            tenant_id=self.tenant_id,
            model=model_id,
            role=role,
            outcome=outcome or ("degraded" if degrade_tier else "success"),
            ttft_ms=ttft_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            degraded=bool(degrade_tier),
        )

    def _stamp_llm_span(
        self,
        span: Any,
        role: str,
        model_id: str,
        degrade_tier: Optional[str],
        workflow_id: Optional[str],
        cost_usd: float,
        ttft_ms: Optional[float],
        input_tokens: Optional[int],
        output_tokens: Optional[int],
        cost_estimated: bool,
        messages: Any = None,
        prompt_template_id: Optional[str] = None,
    ) -> None:
        """The attribute set, written to whichever span the caller resolved."""
        try:
            span.set_attribute("tenant.id", self.tenant_id)
            span.set_attribute("llm.model_name", model_id)
            span.set_attribute("llm.gateway.tier", role)
            span.set_attribute("llm.gateway.cost_usd", cost_usd)
            # An estimate and a measurement must not read alike. The stream
            # path bills the try_reserve() figure derived from `max_tokens`,
            # which is a ceiling, not what the call actually used.
            span.set_attribute("llm.gateway.cost_estimated", cost_estimated)

            # WHICH PROMPT produced this. The framework recorded the model, the
            # cost, the latency and the verdict, and nothing about what was sent —
            # so "answers got worse last Tuesday" had no column to join against.
            # A digest of the system turn changes when, and only when, a human
            # edits it; the text itself is deliberately not recorded (see
            # runtime/prompt_identity).
            if messages is not None:
                from runtime.prompt_identity import prompt_attributes

                for key, value in prompt_attributes(
                    messages, template_id=prompt_template_id
                ).items():
                    span.set_attribute(key, value)

            # Usage is written ONLY when the provider reported it. A streamed call
            # has no usage in v1, and the CompletionResult carries 0/0 for it —
            # writing that 0 here would make "used no tokens" and "nobody counted"
            # the same number on a dashboard that sums them, which is how a token
            # budget silently undercounts every streamed call.
            reported = input_tokens is not None and output_tokens is not None
            span.set_attribute("llm.usage.reported", reported)
            if input_tokens is not None and output_tokens is not None:
                span.set_attribute("llm.usage.input_tokens", int(input_tokens))
                span.set_attribute("llm.usage.output_tokens", int(output_tokens))
                span.set_attribute(
                    "llm.usage.total_tokens", int(input_tokens) + int(output_tokens)
                )

            if ttft_ms is not None:
                span.set_attribute("llm.gateway.ttft_ms", ttft_ms)
            if degrade_tier:
                span.set_attribute("llm.gateway.degrade_reason", degrade_tier)
            if workflow_id:
                span.set_attribute("workflow.id", workflow_id)
        except Exception:  # fail-open: an attribute write must never break the call
            pass

    # ── Completion ────────────────────────────────────────────────────────────

    async def complete_stream(
        self,
        prompt: Any,
        model_hint: str = "developer",
        workflow_id: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.2,
        **kwargs: Any,
    ) -> CompletionResult:
        """Stream a completion and measure TTFT.

        Supports every direct-API provider — OpenAI-compatible (openai /
        groq / ollama) and Anthropic. Providers with no shared SSE surface
        (the cloud-native adapters) fall back to the non-streaming path
        with `ttft_ms=None` rather than raising: streaming is a latency
        optimisation, never a correctness requirement, so a `models.yaml`
        provider swap must not take a tenant's pipeline down
        (TestbedFeedback-2026-07-21 G1 — this method used to raise
        NotImplementedError for exactly the frontier providers a tenant
        puts on its latency-critical path).
        """
        del kwargs

        budget = self.get_budget_status()
        role, degrade_tier = self._resolve_role(model_hint, budget)

        cfg = self.models.get(role)
        if not cfg:
            raise ValueError(
                f"No model registered for role {role!r}. Check models.yaml."
            )

        provider = cfg.get("provider", "openai")
        from runtime.provider_dispatch import (
            build_request,
            parse_stream_delta,
            supports_streaming,
        )
        if not supports_streaming(provider):
            # Fall back BEFORE reserving budget or reporting run status, so
            # complete() owns the whole call exactly as if it had been
            # invoked directly — no double reservation, no orphaned run row.
            logger.info(
                "provider %r does not support streaming; falling back to complete() "
                "for tenant=%s role=%s (ttft_ms will be None)",
                provider,
                self.tenant_id,
                role,
            )
            return await self.complete(
                prompt,
                model_hint=model_hint,
                workflow_id=workflow_id,
                max_tokens=max_tokens,
                temperature=temperature,
            )

        model_id = cfg["id"]
        messages = self._coerce_messages(prompt)

        # Prompt injection heuristics (SEC-PROMPT-001) — before PII scrub.
        from runtime.prompt_guard import (
            PromptGuardBlockedError,
            apply_prompt_guard,
            is_enforcing as prompt_guard_is_enforcing,
            resolve_mode as resolve_prompt_guard_mode,
        )
        pg_mode = resolve_prompt_guard_mode()
        pg_reasons: list[str] = []
        if pg_mode != "off":
            pg_result = apply_prompt_guard(messages)
            pg_reasons = list(pg_result.reasons)
            if pg_result.blocked:
                # PROMPT_GUARD=warn reports without blocking; every other
                # non-off mode enforces (TestbedFeedback-2026-07-21 G9).
                # is_enforcing() is the single definition of "blocking",
                # shared with the SEC-PROMPT-001 harness.
                if prompt_guard_is_enforcing(pg_mode):
                    logger.warning(
                        "prompt_guard blocked tenant=%s mode=%s reasons=%s",
                        self.tenant_id,
                        pg_mode,
                        pg_result.reasons,
                    )
                    raise PromptGuardBlockedError(
                        f"prompt blocked: {', '.join(pg_result.reasons)}",
                        reasons=pg_result.reasons,
                    )
                logger.warning(
                    "prompt_guard flagged (not blocked) tenant=%s mode=%s reasons=%s "
                    "— call proceeding; findings on CompletionResult.prompt_guard_reasons",
                    self.tenant_id,
                    pg_mode,
                    pg_result.reasons,
                )

        from runtime.input_guardrail import resolve_mode, scrub_messages
        guardrail_mode = resolve_mode()
        messages, guardrail_counts = scrub_messages(messages, mode=guardrail_mode)
        if guardrail_counts:
            logger.info(
                "input_guardrail applied tenant=%s mode=%s counts=%s",
                self.tenant_id,
                guardrail_mode,
                guardrail_counts,
            )

        estimated_cost_usd = max_tokens * (
            cfg.get("cost_per_input_token", 0) + cfg.get("cost_per_output_token", 0)
        )
        reserved = True
        if estimated_cost_usd and not self._is_free_tier(cfg):
            reserved = self._budget.try_reserve(
                self.tenant_id, estimated_cost_usd, self.budget_cap_usd
            )
            if not reserved:
                raise BudgetExceededError(
                    f"tenant={self.tenant_id} budget reservation of ${estimated_cost_usd:.4f} for "
                    f"model_hint={model_hint!r} would exceed cap (${budget.spent_usd:.2f}/${budget.cap_usd:.2f}). "
                    "Concurrent in-flight calls already reserved the remaining budget."
                )

        run_id = (
            f"{workflow_id}-{uuid.uuid4().hex[:8]}"
            if workflow_id
            else f"{self.tenant_id}-{uuid.uuid4().hex[:12]}"
        )
        # Wall-clock start, for the span the gateway emits when nothing else is
        # recording. time_ns (not perf_counter_ns) because OTel start_time is an
        # absolute epoch timestamp, not a monotonic reading.
        started_ns = time.time_ns()
        self._report_run_status(run_id, "running", workflow_id=workflow_id)

        # Shared with _invoke() — this used to be a near-copy that omitted
        # the anthropic branch entirely (TestbedFeedback G1).
        base_url, api_key = self._resolve_endpoint(cfg)

        path, headers, body = build_request(
            provider, model_id, messages, api_key, max_tokens, temperature
        )
        body = {**body, "stream": True}
        url = base_url.rstrip("/") + path

        import httpx

        start = time.perf_counter()
        ttft_ms: Optional[float] = None
        chunks: list[str] = []
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                async with client.stream(
                    "POST", url, json=body, headers=headers
                ) as resp:
                    if not resp.is_success:
                        # Mirrors _invoke()'s _post_with_retry: surface the
                        # provider's actual error body, not just the status
                        # line. resp.raise_for_status() alone raises
                        # httpx.HTTPStatusError whose str() omits the body
                        # entirely (just "Client error '400 Bad Request' for
                        # url ...") — _is_provider_exhausted() pattern-matches
                        # on message text like "credit balance is too low",
                        # which was invisible to it here, so a billing/quota
                        # failure never degraded to the next tier; it just
                        # propagated raw (TestbedFeedback-2026-07-23-style
                        # gap: found running this tenant's Analyst live
                        # against a real, credit-exhausted Anthropic key).
                        await resp.aread()
                        try:
                            err_body = resp.json()
                            err_msg = (
                                err_body.get("error", {}).get("message")
                                or err_body.get("message")
                                or resp.text[:400]
                            )
                        except Exception:
                            err_msg = resp.text[:400]
                        raise RuntimeError(
                            f"LLM API error {resp.status_code} (model={model_id!r}): {err_msg}"
                        )
                    async for line in resp.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        payload = line.removeprefix("data:").strip()
                        if not payload or payload == "[DONE]":
                            continue
                        data = json.loads(payload)
                        # Per-provider envelope lives in provider_dispatch;
                        # non-text events (keep-alives, block start/stop,
                        # usage-only deltas) return None so TTFT is timed
                        # from the first real token, not the first frame.
                        content = parse_stream_delta(provider, data)
                        if not content:
                            continue
                        if ttft_ms is None:
                            ttft_ms = (time.perf_counter() - start) * 1000
                        chunks.append(content)
        except Exception as exc:
            if reserved and estimated_cost_usd:
                self._budget.add_spend(self.tenant_id, -estimated_cost_usd)
            self._report_run_status(
                run_id,
                "failed",
                workflow_id=workflow_id,
                error_summary=str(exc)[:500],
            )
            if self._is_provider_exhausted(exc):
                # The streaming path resolves and tries exactly ONE tier —
                # unlike complete(), it has no degrade-chain loop of its
                # own. A billing/quota/overload failure here used to
                # propagate straight out with no fallback at all, taking
                # down a tenant's whole pipeline over an account issue that
                # complete() already knows how to walk around. Streaming is
                # a latency optimisation, never a correctness requirement
                # (same principle as the supports_streaming() fallback
                # above) — losing ttft_ms must not mean losing the
                # completion, so hand off to complete()'s degrade chain
                # instead of raising.
                logger.warning(
                    "streaming attempt exhausted for role=%r model=%r: %s — "
                    "falling back to complete()'s degrade chain (ttft_ms will be None)",
                    role,
                    model_id,
                    exc,
                )
                return await self.complete(
                    prompt,
                    model_hint=model_hint,
                    workflow_id=workflow_id,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
            raise

        text = "".join(chunks)
        # Stream v1 has no usage tokens; keep the try_reserve() amount as cost.
        cost_usd = estimated_cost_usd if (reserved and estimated_cost_usd) else 0.0

        # Output moderation (SEC-MOD-001) — before success status (audit truth).
        # Caught rather than raised through, for the reason `complete()` spells
        # out: re-raising here jumps over `_record_span_attributes`, and a
        # blocked call that was already charged then emits no cost, no TTFT and
        # no call counter. The streaming path is the sibling that had the same
        # defect and would have kept it.
        blocked: Optional[Exception] = None
        try:
            apply_output_moderation(text, raise_on_block=True)
        except (ModerationBlockedError, ModerationHookRequiredError) as exc:
            blocked = exc

        self._record_span_attributes(
            role,
            model_id,
            degrade_tier,
            workflow_id,
            cost_usd,
            ttft_ms,
            # Stream v1 reports no usage, and cost is the try_reserve() ceiling
            # rather than a measurement. Both facts travel with the span so a
            # consumer is not left inferring them from a zero.
            input_tokens=None,
            output_tokens=None,
            cost_estimated=True,
            started_ns=started_ns,
            messages=messages,
            outcome="blocked" if blocked else None,
        )

        if blocked is not None:
            self._report_run_status(
                run_id,
                "failed",
                workflow_id=workflow_id,
                error_summary=str(blocked)[:500],
            )
            raise blocked

        self._report_run_status(
            run_id,
            "degraded" if degrade_tier else "success",
            workflow_id=workflow_id,
            cost_usd=cost_usd or None,
        )

        return CompletionResult(
            text=text,
            model_used=model_id,
            input_tokens=0,
            output_tokens=0,
            cost_usd=cost_usd,
            degrade_tier=degrade_tier,
            ttft_ms=ttft_ms,
            guardrail_counts=dict(guardrail_counts),
            prompt_guard_reasons=list(pg_reasons),
        )

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
        """
        Route a completion request with per-tenant budget enforcement.

        model_hint options: "architect" | "developer" | "validator" | "fast"
        """
        if idempotency_key and self._idempotency is not None:
            try:
                cached = self._idempotency.get(idempotency_key)
                if cached is not None:
                    from runtime.metrics import record_cache

                    record_cache(tenant_id=self.tenant_id, hit=True)
                    logger.info(
                        "idempotency cache hit tenant=%s key=%s",
                        self.tenant_id,
                        idempotency_key,
                    )
                    cached_result = CompletionResult(**cached)
                    # Re-run moderation on cache hits (SEC-MOD-001) so a newly
                    # registered/stricter hook cannot be bypassed by idempotency.
                    apply_output_moderation(cached_result.text, raise_on_block=True)
                    return cached_result
                from runtime.metrics import record_cache

                record_cache(tenant_id=self.tenant_id, hit=False)
                logger.debug(
                    "idempotency cache miss tenant=%s key=%s",
                    self.tenant_id,
                    idempotency_key,
                )
            except (ModerationBlockedError, ModerationHookRequiredError):
                raise
            except Exception as exc:
                # Now that the backends are real (Postgres/Redis), a failure
                # here is a live infra error (DB down, bad creds), not the
                # old "backend not implemented" case — log it instead of
                # silently treating every failure as a cache miss.
                logger.error(
                    "idempotency lookup failed tenant=%s key=%s: %s",
                    self.tenant_id,
                    idempotency_key,
                    exc,
                )

        budget = self.get_budget_status()
        role, degrade_tier = self._resolve_role(model_hint, budget)

        cfg = self.models.get(role)
        if not cfg:
            raise ValueError(
                f"No model registered for role {role!r}. Check models.yaml."
            )

        model_id = cfg["id"]
        messages = self._coerce_messages(prompt)

        # Prompt injection heuristics (SEC-PROMPT-001) — before PII scrub.
        # PROMPT_GUARD=off|default|strict (default=default). Strict raises inside
        # apply_prompt_guard; default also refuses the provider call here.
        from runtime.prompt_guard import (
            PromptGuardBlockedError,
            apply_prompt_guard,
            is_enforcing as prompt_guard_is_enforcing,
            resolve_mode as resolve_prompt_guard_mode,
        )
        pg_mode = resolve_prompt_guard_mode()
        pg_reasons: list[str] = []
        if pg_mode != "off":
            pg_result = apply_prompt_guard(messages)
            pg_reasons = list(pg_result.reasons)
            if pg_result.blocked:
                # PROMPT_GUARD=warn reports without blocking; every other
                # non-off mode enforces (TestbedFeedback-2026-07-21 G9).
                # is_enforcing() is the single definition of "blocking",
                # shared with the SEC-PROMPT-001 harness.
                if prompt_guard_is_enforcing(pg_mode):
                    logger.warning(
                        "prompt_guard blocked tenant=%s mode=%s reasons=%s",
                        self.tenant_id,
                        pg_mode,
                        pg_result.reasons,
                    )
                    raise PromptGuardBlockedError(
                        f"prompt blocked: {', '.join(pg_result.reasons)}",
                        reasons=pg_result.reasons,
                    )
                logger.warning(
                    "prompt_guard flagged (not blocked) tenant=%s mode=%s reasons=%s "
                    "— call proceeding; findings on CompletionResult.prompt_guard_reasons",
                    self.tenant_id,
                    pg_mode,
                    pg_result.reasons,
                )

        # Pre-call PII scrub (PDPL / FIXES Security & Guardrails) — masks
        # personal data in the prompt before provider invoke. Symmetric to
        # post-call trace_redactor.py. Mode: INPUT_GUARDRAIL or env default.
        from runtime.input_guardrail import resolve_mode, scrub_messages
        guardrail_mode = resolve_mode()
        messages, guardrail_counts = scrub_messages(messages, mode=guardrail_mode)
        if guardrail_counts:
            logger.info(
                "input_guardrail applied tenant=%s mode=%s counts=%s",
                self.tenant_id,
                guardrail_mode,
                guardrail_counts,
            )
        try:
            from opentelemetry import trace as _otel_trace

            span = _otel_trace.get_current_span()
            if span and span.is_recording():
                span.set_attribute("llm.gateway.prompt_guard.mode", pg_mode)
                span.set_attribute("llm.gateway.input_guardrail.mode", guardrail_mode)
                span.set_attribute(
                    "llm.gateway.input_guardrail.redactions",
                    sum(guardrail_counts.values()),
                )
                for kind, n in guardrail_counts.items():
                    span.set_attribute(f"llm.gateway.input_guardrail.{kind}", n)
        except Exception:  # fail-open: tracing must never break the LLM call
            pass

        # Reserve an upper-bound cost estimate atomically before the call,
        # not after — closes the check-then-act race where concurrent calls
        # could all observe "not breached" before any of them recorded
        # spend (Product_Archive.md 2.1). max_tokens bounds output cost
        # exactly; input cost is bounded by the same max_tokens too since we
        # don't know the actual prompt token count until the provider
        # responds — this overestimates input cost, which only makes the
        # gateway degrade *earlier* under contention, never later.
        estimated_cost_usd = max_tokens * (
            cfg.get("cost_per_input_token", 0) + cfg.get("cost_per_output_token", 0)
        )
        reserved = True
        if estimated_cost_usd and not self._is_free_tier(cfg):
            reserved = self._budget.try_reserve(
                self.tenant_id, estimated_cost_usd, self.budget_cap_usd
            )
            if not reserved:
                raise BudgetExceededError(
                    f"tenant={self.tenant_id} budget reservation of ${estimated_cost_usd:.4f} for "
                    f"model_hint={model_hint!r} would exceed cap (${budget.spent_usd:.2f}/${budget.cap_usd:.2f}). "
                    "Concurrent in-flight calls already reserved the remaining budget."
                )

        # run_id is always unique per CALL, never reused across multiple
        # gateway.complete() calls within one workflow run — a workflow
        # that makes 2+ calls (the expected shape for multi-agent/
        # multi-LLM tenant apps, not just the single-call oil-price
        # example) would otherwise have call #2's "running" report
        # re-upsert call #1's already-"success" agent_runs row, resetting
        # finished_at to NULL and making the widget show "running" for a
        # workflow that's actually done. workflow_id is reported
        # separately (see _report_run_status) purely as a grouping key —
        # portal/lib/runStatus.ts aggregates all calls sharing a
        # workflow_id (including concurrent/parallel ones, e.g. fan-out to
        # multiple LLMs) into one widget status rather than relying on
        # row identity to do that grouping.
        run_id = (
            f"{workflow_id}-{uuid.uuid4().hex[:8]}"
            if workflow_id
            else f"{self.tenant_id}-{uuid.uuid4().hex[:12]}"
        )
        # Wall-clock start, for the span the gateway emits when nothing else is
        # recording. time_ns (not perf_counter_ns) because OTel start_time is an
        # absolute epoch timestamp, not a monotonic reading.
        started_ns = time.time_ns()
        self._report_run_status(run_id, "running", workflow_id=workflow_id)

        # Try the chosen tier; on provider-level exhaustion (billing, quota,
        # overload) walk the degrade_to chain rather than failing immediately.
        degrade_chain = self._degrade_chain(role)
        tried: list[str] = []
        text = in_tok = out_tok = None
        last_exc: Exception | None = None
        for attempt_role in degrade_chain:
            attempt_cfg = self.models.get(attempt_role)
            if not attempt_cfg:
                continue
            try:
                text, in_tok, out_tok = await self._invoke(
                    attempt_cfg, messages, max_tokens, temperature
                )
                if attempt_role != role:
                    # Record the tier we actually used
                    degrade_tier = (
                        "local" if self._is_free_tier(attempt_cfg) else "downgrade"
                    )
                    cfg = attempt_cfg
                    model_id = attempt_cfg["id"]
                    logger.warning(
                        "Degraded from %r to %r due to provider error: %s",
                        role,
                        attempt_role,
                        last_exc,
                    )
                last_exc = None
                break
            except Exception as exc:
                tried.append(attempt_role)
                last_exc = exc
                if self._is_provider_exhausted(exc):
                    logger.warning(
                        "Provider exhausted for role=%r model=%r: %s — trying next tier",
                        attempt_role,
                        attempt_cfg.get("id"),
                        exc,
                    )
                    continue
                # Non-exhaustion error (bad prompt, network timeout, etc.) — fail fast
                break

        if last_exc is not None:
            if reserved and estimated_cost_usd:
                self._budget.add_spend(
                    self.tenant_id, -estimated_cost_usd
                )  # release the reservation
            self._report_run_status(
                run_id,
                "failed",
                workflow_id=workflow_id,
                error_summary=str(last_exc)[:500],
            )
            if tried:
                # Name the unset credentials when there are any. "All model
                # tiers exhausted" sends an operator to their provider's
                # billing page; an empty ANTHROPIC_API_KEY sends them to their
                # own environment, which is where the problem is.
                missing = self._unset_key_envs(tried)
                hint = (
                    f" No API key is set for: {', '.join(missing)} — an unset key "
                    "reads as an auth failure, which this gateway treats as tier "
                    "exhaustion."
                    if missing
                    else ""
                )
                raise RuntimeError(
                    f"All model tiers exhausted (tried: {tried}). Last error: {last_exc}.{hint}"
                ) from last_exc
            raise last_exc

        if text is None:
            # NOT unreachable, which is what a first pass at this comment said.
            # The loop `continue`s past any role missing from `self.models`, so
            # a degrade chain whose every role is unconfigured completes with no
            # attempt made, no exception, and `text` still None — and the
            # `last_exc is not None` branch above never fires because nothing
            # ever failed. Nothing was tried.
            #
            # Before this, that fell through to `apply_output_moderation(None)`
            # and a CompletionResult carrying text=None: a caller got an answer
            # object for a call that never happened.
            raise RuntimeError(
                f"No configured model tier for role {role!r}: the degrade chain "
                f"{self._degrade_chain(role)} resolved to nothing in this "
                f"tenant's registry, so no provider was called. Check "
                f"models.yaml for a role entry, or a degrade_to pointing at a "
                f"role that does not exist."
            )

        # WHEN THE PROVIDER REPORTED NO USAGE, the reservation stands as the
        # charge — exactly what complete_stream() already does with its
        # `cost_estimated=True` ("Stream v1 has no usage tokens; keep the
        # try_reserve() amount as cost"). The two paths are siblings and only
        # one of them knew: parse_response used to default a missing `usage`
        # block to 0/0, so this line computed a cost of $0.00 for a real call
        # and the reconcile below released the entire reservation. A provider
        # that omits usage was free, as far as the monthly cap was concerned.
        usage_reported = in_tok is not None and out_tok is not None
        if usage_reported:
            cost_usd = in_tok * cfg.get("cost_per_input_token", 0) + out_tok * cfg.get(
                "cost_per_output_token", 0
            )
        else:
            logger.warning(
                "provider %r returned no usage block for model=%r — billing the "
                "reserved estimate ($%.4f) rather than treating the call as free",
                cfg.get("provider"),
                model_id,
                estimated_cost_usd,
            )
            cost_usd = estimated_cost_usd if (reserved and estimated_cost_usd) else 0.0

        if reserved and estimated_cost_usd:
            # Reconcile: replace the (conservative) reservation with the
            # actual cost. The delta can be negative (actual < estimate,
            # the common case) or positive (rare, e.g. provider returned
            # more output tokens than max_tokens would suggest) — add_spend
            # accepts a signed amount either way. With no usage reported,
            # cost_usd IS the estimate, so the delta is zero and the
            # reservation simply stands.
            delta = cost_usd - estimated_cost_usd
            if delta:
                self._budget.add_spend(self.tenant_id, delta)
        elif cost_usd:
            self._budget.add_spend(self.tenant_id, cost_usd)

        # Output moderation (SEC-MOD-001) — before success status (audit truth).
        #
        # CAUGHT, not raised through. Re-raising here jumped over
        # `_record_span_attributes` below, and that call is what emits the LLM
        # span attributes AND the `agentsmith.llm.*` counters. So a blocked call
        # was charged to the budget ledger a dozen lines up and then emitted no
        # cost attribute, no token attributes and no call counter at all: the
        # ledger and the telemetry disagreed by exactly the moderation-blocked
        # calls, and the `outcome` dimension that exists to make an error rate
        # computable never saw the one outcome a security control produces.
        #
        # The call happened and was paid for, so it is recorded. Only the run
        # STATUS waits for moderation, which is the audit-truth ordering this
        # comment has always been about.
        blocked: Optional[Exception] = None
        try:
            apply_output_moderation(text, raise_on_block=True)
        except (ModerationBlockedError, ModerationHookRequiredError) as exc:
            blocked = exc

        self._record_span_attributes(
            role,
            model_id,
            degrade_tier,
            workflow_id,
            cost_usd,
            input_tokens=in_tok,
            output_tokens=out_tok,
            # The span says the cost is an estimate whenever it is one, the
            # same flag the streaming path sets. A consumer summing
            # `llm.gateway.cost_usd` can then tell a measurement from a ceiling.
            cost_estimated=not usage_reported,
            started_ns=started_ns,
            messages=messages,
            outcome="blocked" if blocked else None,
        )

        if blocked is not None:
            self._report_run_status(
                run_id,
                "failed",
                workflow_id=workflow_id,
                error_summary=str(blocked)[:500],
            )
            raise blocked

        self._report_run_status(
            run_id,
            "degraded" if degrade_tier else "success",
            workflow_id=workflow_id,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=cost_usd,
        )

        result = CompletionResult(
            text=text,
            model_used=model_id,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=cost_usd,
            degrade_tier=degrade_tier,
            guardrail_counts=dict(guardrail_counts),
            prompt_guard_reasons=list(pg_reasons),
        )

        if idempotency_key and self._idempotency is not None:
            try:
                self._idempotency.set(idempotency_key, result.__dict__)
            except Exception as exc:
                logger.error(
                    "idempotency write failed tenant=%s key=%s: %s",
                    self.tenant_id,
                    idempotency_key,
                    exc,
                )

        return result

    @staticmethod
    def _coerce_messages(prompt: Any) -> list[dict]:
        if isinstance(prompt, str):
            return [{"role": "user", "content": prompt}]
        if isinstance(prompt, list):
            return prompt
        raise TypeError(f"prompt must be str or list[dict], got {type(prompt)}")

    @staticmethod
    def _retry_reason(exc: BaseException) -> str:
        """A COARSE class for the metric dimension.

        Never the provider's message: a metric attribute carrying free text
        creates a time series per distinct string. The message goes on the span
        event, where cardinality does not matter.
        """
        text = str(exc).lower()
        if "429" in text or "rate limit" in text:
            return "rate_limit"
        if "timeout" in text or "timed out" in text:
            return "timeout"
        if any(code in text for code in ("500", "502", "503", "504")):
            return "server_error"
        return "transient"

    def _on_retry(self, model_id: str):
        """A tenacity `before_sleep` hook that makes a retry visible.

        The gateway has always retried transient failures with full-jitter
        backoff, and NOTHING said an attempt had happened — so a call retried
        three times looked simply slow. On a free tier where 429s are routine
        that is the common case, not the rare one, and it points every
        investigation at latency when the answer is quota.
        """
        def _hook(retry_state) -> None:
            exc = retry_state.outcome.exception() if retry_state.outcome else None
            reason = self._retry_reason(exc) if exc else "unknown"
            try:
                from opentelemetry import trace

                span = trace.get_current_span()
                if span is not None and span.is_recording():
                    span.add_event(
                        "llm.retry",
                        {
                            "attempt": retry_state.attempt_number,
                            "reason": reason,
                            "sleep_s": float(getattr(retry_state, "idle_for", 0.0) or 0.0),
                            # The full message HERE, not on the metric.
                            "error": str(exc)[:400] if exc else "",
                            "llm.model_name": model_id,
                        },
                    )
            except Exception:  # fail-open: telemetry never breaks a retry
                pass
            try:
                from runtime.metrics import record_retry

                record_retry(
                    tenant_id=self.tenant_id,
                    model=model_id,
                    attempt=retry_state.attempt_number,
                    reason=reason,
                )
            except Exception:
                pass

        return _hook

    @staticmethod
    def _is_retryable_provider_error(exc: BaseException) -> bool:
        """Transient-only: connection/timeout issues, 429 (rate limit), and
        5xx (provider-side fault) — never 4xx other than 429, since a bad
        request/auth/model-id error will fail identically on retry and
        retrying it just burns the attempt budget for no benefit."""
        import httpx

        if isinstance(exc, httpx.TransportError):
            return True
        if isinstance(exc, httpx.HTTPStatusError):
            status = exc.response.status_code
            return status == 429 or status >= 500
        return False

    def _unset_key_envs(self, roles: list[str]) -> list[str]:
        """Env vars the tried tiers needed and did not have.

        `_lookup_api_key` returns "" for an unset variable, so the gateway
        sends an empty credential, the provider answers 401, and
        `_is_provider_exhausted` counts an auth error as exhaustion — by
        design, so a tenant holding only one vendor's key degrades past the
        others. The cost is the diagnosis: with NO key set, every tier
        "exhausts" and the operator is told `All model tiers exhausted`, which
        reads as a capacity or billing problem at the provider. It is a unset
        variable.

        Resolution goes through `_resolve_endpoint` rather than being
        recomputed here — a second implementation of the lookup would be a
        fourth copy of the thing this pass just collapsed into one.
        """
        from runtime.provider_dispatch import _DEFAULT_API_KEY_ENV

        missing: list[str] = []
        for role in roles:
            cfg = self.models.get(role) or {}
            provider = cfg.get("provider", "openai")
            if not _DEFAULT_API_KEY_ENV.get(provider):
                continue  # local or cloud-credential provider — no key to miss
            try:
                _, api_key = self._resolve_endpoint(cfg)
            except Exception:  # a config this broken has a louder problem
                continue
            if not api_key:
                env = str(cfg.get("api_key_env") or _DEFAULT_API_KEY_ENV[provider])
                if env not in missing:
                    missing.append(env)
        return missing

    @staticmethod
    def _resolve_endpoint(cfg: dict) -> tuple[str, str]:
        """(base_url, api_key) for a direct-API provider config.

        base_url/api_key_env are config-driven (models.yaml `endpoint` /
        `api_key_env` fields) so a tenant can point a provider at a proxy,
        a region-pinned host, or a differently-named API key env var
        (e.g. per-tenant keys) without editing this code. The literals
        below are fallbacks for the common case only — direct Anthropic/
        OpenAI calls — not a ceiling on what's supported.

        Shared by _invoke() and complete_stream(). They used to carry
        near-duplicate copies and the streaming one silently omitted the
        anthropic branch, which is part of why streaming never worked for
        it (TestbedFeedback-2026-07-21 G1).
        """
        # The provider -> API-key-env mapping comes from
        # provider_dispatch._DEFAULT_API_KEY_ENV, which this module already
        # imports from. It was spelled out again here as literals in an
        # if/elif chain — a THIRD copy of a catalog that also exists in
        # scripts/_shared.py, and the only one of the three with nothing
        # pinning it (scripts/test/test_judge_model_resolution.py already
        # asserts the other two are equal). Adding a provider meant editing a
        # dict, a mirror, and a branch, and forgetting the branch resolved the
        # new provider's key from OPENAI_API_KEY without a word.
        from runtime.provider_dispatch import _DEFAULT_API_KEY_ENV

        provider = cfg.get("provider", "openai")
        key_env = _DEFAULT_API_KEY_ENV.get(provider) or "OPENAI_API_KEY"

        if provider == "anthropic":
            api_key = LLMGateway._lookup_api_key(cfg, key_env)
            base_url = cfg.get("endpoint") or "https://api.anthropic.com"
        elif provider == "ollama":
            base_url = os.path.expandvars(cfg.get("endpoint", "${OLLAMA_BASE_URL}/v1"))
            # expandvars leaves unset variables as literal "${VAR}" — not a valid URL.
            if not base_url.startswith("http"):
                base_url = (
                    os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434") + "/v1"
                )
            api_key = "ollama"
        elif provider == "groq":
            # Groq's API is OpenAI-compatible (same request/response shape,
            # parse_response's non-anthropic branch handles it) — only the
            # host and API key env var differ from direct OpenAI, same as
            # every other "openai_compatible" provider in this codebase.
            api_key = LLMGateway._lookup_api_key(cfg, key_env)
            base_url = cfg.get("endpoint") or "https://api.groq.com/openai/v1"
        elif provider == "xai":
            # OpenAI-compatible; only host and key differ.
            api_key = LLMGateway._lookup_api_key(cfg, key_env)
            base_url = cfg.get("endpoint") or "https://api.x.ai/v1"
        elif provider == "google_ai":
            # Google AI Studio's OpenAI-compatibility layer. Distinct from the
            # `vertex_ai` adapter, which is the same models behind
            # service-account OAuth — an AI Studio key will not authenticate
            # there, and vice versa.
            api_key = LLMGateway._lookup_api_key(cfg, key_env)
            base_url = (
                cfg.get("endpoint")
                or "https://generativelanguage.googleapis.com/v1beta/openai"
            )
        elif provider == "openrouter":
            # One OpenAI-compatible endpoint fronting many vendors. The model
            # id carries the vendor ("anthropic/claude-sonnet-4.5"), and the
            # envelope is OpenAI chat regardless of whose model it is — which
            # is why api_format is declared per catalog entry rather than
            # inferred from the vendor in the id.
            api_key = LLMGateway._lookup_api_key(cfg, key_env)
            base_url = cfg.get("endpoint") or "https://openrouter.ai/api/v1"
        else:
            api_key = LLMGateway._lookup_api_key(cfg, key_env)
            base_url = cfg.get("endpoint") or "https://api.openai.com/v1"
        return os.path.expandvars(base_url), api_key

    @staticmethod
    def _lookup_api_key(cfg: dict, default_env: str) -> str:
        """Resolve a model's API key, honoring a per-role `api_key_env`
        override (e.g. a tenant giving the judge role its own Anthropic
        account, distinct from the analyst's — RFC-002 "judge/actor
        separation" extended to the account level, not just the model id).

        Falls back to `default_env` when `api_key_env` is configured but its
        target variable isn't populated yet, rather than resolving to an
        empty string and sending a broken auth header — a tenant can roll
        out a dedicated key for one role at a time without both roles going
        dark in between.
        """
        custom_env = cfg.get("api_key_env")
        if custom_env and custom_env != default_env:
            value = os.environ.get(custom_env, "")
            if value:
                return value
        return os.environ.get(default_env, "")

    async def _invoke(
        self, cfg: dict, messages: list[dict], max_tokens: int, temperature: float
    ) -> tuple[str, Optional[int], Optional[int]]:
        """Call the provider for this model config. Returns (text, input_tokens, output_tokens).

        The token counts are OPTIONAL and the annotation used to say they were
        not. `parse_response` returns None when a provider omits its `usage`
        block, so this has always been able to hand back None — and a caller
        trusting `int` did arithmetic on it. That is exactly how
        scripts/cost_router.py came to pass None into the circuit breaker,
        where the TypeError landed in a fail-open handler and the call went
        unmetered on both tiers, silently. The annotation is the fix at the
        source; mypy names every caller that has not caught up.

        Request building / response parsing delegated to
        runtime/provider_dispatch.py, shared with scripts/cost_router.py
        (Product_Archive.md 4.3) — only the base_url/api_key resolution
        below (which legitimately differs: this is the production path with
        its own model registry, cost_router.py has its own env-var-driven
        route table) stays local to this method.

        Retries transient failures with exponential backoff (this module's
        own docstring has documented a "Throttle: exponential backoff on
        request rate" degrade-ladder step from the start — `tenacity` was
        already a required dependency for exactly this, but nothing in the
        codebase actually called it until now). Non-transient errors (bad
        request, auth failure, unknown model) raise immediately — retrying
        those would just waste the attempt budget on a failure that can't
        succeed differently the second time.
        """
        import httpx
        from tenacity import (
            retry,
            retry_if_exception,
            stop_after_attempt,
            wait_exponential,
        )

        from runtime.provider_dispatch import (
            build_cloud_request,
            build_request,
            is_cloud_provider,
            parse_cloud_response,
            parse_response,
        )
        provider = cfg.get("provider", "openai")
        model_id = cfg["id"]

        if is_cloud_provider(provider):
            # Cloud-native providers (vertex_ai/azure_openai/bedrock/
            # huawei_modelarts) need their own auth scheme and URL/envelope
            # shape, not just a different host — provider_dispatch.py's
            # CloudProviderAdapter owns that, and returns a full URL rather
            # than a path since project/region/deployment/endpoint-id are
            # baked into the URL itself.
            url, headers, body = build_cloud_request(
                provider, model_id, messages, cfg, max_tokens, temperature
            )
        else:
            base_url, api_key = self._resolve_endpoint(cfg)

            path, headers, body = build_request(
                provider, model_id, messages, api_key, max_tokens, temperature
            )
            url = base_url.rstrip("/") + path

        @retry(
            retry=retry_if_exception(self._is_retryable_provider_error),
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            reraise=True,
            before_sleep=self._on_retry(model_id),
        )
        async def _post_with_retry() -> dict:
            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(url, json=body, headers=headers)
                if not resp.is_success:
                    # Try to surface a human-readable message from the provider
                    # before falling back to a raw HTTP error.
                    try:
                        err_body = resp.json()
                        err_msg = (
                            err_body.get("error", {}).get("message")
                            or err_body.get("message")
                            or resp.text[:400]
                        )
                    except Exception:
                        err_msg = resp.text[:400]
                    raise RuntimeError(
                        f"LLM API error {resp.status_code} (model={model_id!r}): {err_msg}"
                    )
                return resp.json()

        data = await _post_with_retry()

        # How many attempts it actually took. Recorded even when it is 1, so
        # "this call did not retry" is a fact on the span rather than the
        # absence of one — the same reason an empty retrieval still emits a
        # span.
        try:
            from opentelemetry import trace

            attempts = int(
                getattr(_post_with_retry, "statistics", {}).get("attempt_number", 1)
            )
            span = trace.get_current_span()
            if span is not None and span.is_recording():
                span.set_attribute("llm.gateway.attempts", attempts)
        except Exception:  # fail-open
            pass

        if is_cloud_provider(provider):
            return parse_cloud_response(provider, data)
        return parse_response(provider, data)
