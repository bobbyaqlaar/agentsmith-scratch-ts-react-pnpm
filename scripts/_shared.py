"""
scripts/_shared.py — small helpers duplicated, byte-for-byte in most cases,
across most of scripts/*.py before this consolidation. Import directly
(`from _shared import _repo_root`) — every scripts/*.py file is always
invoked as `python3 scripts/whatever.py`, which puts this directory on
sys.path[0] automatically, the same mechanism scripts/run-evals.py and
scripts/shadow-eval.py already rely on to import scripts/eval_judge.py.

Deliberately NOT shared with runtime/llm_gateway.py's own copy of
_repo_root() — runtime/ is vendored/deployed independently of scripts/
(a tenant repo can carry runtime/ without scripts/ at all), so importing
from here would create a coupling that breaks that independence. The
duplication between scripts/ and runtime/ is a real architectural
boundary, not an oversight; only the duplication *within* scripts/ is
consolidated here.

What lives here, and what deliberately does not:

  load_script      one loader for hyphen-named scripts. Thirteen files
                   hand-rolled the same importlib dance; three had
                   independently reinvented its caching.
  load_registry    the merged model registry, cached per cwd.
  judge_model      the eval judge, registry-first.
  _phoenix_*       one request path; get/post are thin wrappers over it.
  _load_dotenv     .env parsing, inline comments handled.

Not here on purpose:

  * runtime/prompt_guard._denylist_path and
    runtime/tool_registry.default_allowlist_path are structurally identical
    (env var → .agent-rfc/security/<file> → None), but they are runtime
    modules, share no import, and are each usable standalone. Sharing them
    would mean inventing a module for eight lines or coupling two guardrails
    that are deliberately independent.
  * The _FALLBACK_* maps below mirror runtime/provider_dispatch. That
    duplication is a version-skew shim with its own drift tests, not an
    oversight — see their comments.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Optional

# ── Eval-judge model ──────────────────────────────────────────────────────────
#
# Resolution order (see judge_model() below):
#   1. AGENT_JUDGE_MODEL env var — one-off override, no code change
#   2. the `judge` role in the MERGED model registry: framework
#      runtime/models.yaml ← tenant models.yaml ← tenant.yaml
#      gateway.routing_overrides
#   3. DEFAULT_JUDGE_MODEL below — last resort, only when runtime/ isn't
#      importable at all
#
# (2) is why this is no longer a bare constant. A hardcoded id here made the
# judge a *second*, independent model setting: a tenant could declare
# `judge: <model>` in its own models.yaml for its runtime judge and still have
# CI evals graded by whatever this file happened to say. KYC Sentinel is the
# concrete case — it declares an independent judge route, and its scorecard was
# nonetheless being judged by the framework constant. One role, one id, both
# consumers. Before the constant existed at all, run-evals.py / shadow-eval.py
# / verify_system.py each carried their own fallback and drifted apart —
# reading the registry keeps that fixed while removing the duplicate source.
#
# Docs referencing the default: SPECS.md §7/§21, OPERATIONS.md §0,
# UserManual.md §8.
# Kept in step with the `judge` role in runtime/models.yaml so the fallback and
# the registry never name different graders.
DEFAULT_JUDGE_MODEL = "falcon3:3b"

JUDGE_ROLE = "judge"


_REGISTRY_CACHE: dict[str, Optional[dict]] = {}


def role_model(role: str, fallback: str) -> str:
    """Model id for a registry role, or `fallback` when it can't be resolved.

    The single accessor every scripts/*.py should use instead of hardcoding a
    model name. `fallback` is a last resort for scripts-only installs with no
    runtime/ on the path — it is NOT a second source of truth, so keep it equal
    to what models.yaml says for that role.
    """
    registry = load_registry() or {}
    return (registry.get(role) or {}).get("id") or fallback


def role_credential_env(role: str) -> Optional[str]:
    """Env var the given registry role needs, or None if it needs no credential.

    Provider-agnostic on purpose: it reads the MERGED registry, so it follows a
    role wherever the tenant points it. Asking "is ANTHROPIC_API_KEY set?"
    instead — which the KYC Sentinel CI originally did — is wrong twice over:
    it breaks the moment a tenant repoints the role at Groq or a local model,
    and it ignores a role's own `api_key_env` (that tenant's judge declares
    `ANTHROPIC_API_KEY_JUDGE`, so the check was reading a variable the route
    never uses).

    Returns None when the registry is unreadable — callers should treat that
    as "can't tell", not as "no credential needed".
    """
    registry = load_registry()
    if not registry:
        return None
    cfg = registry.get(role)
    if not cfg:
        return None
    try:
        from runtime.provider_dispatch import credential_env_for_model

        return credential_env_for_model(cfg)
    except Exception:
        # The installed runtime predates credential_env_for_model (a tenant
        # pins a framework VERSION, so scripts/ can be newer than the package
        # it imports). Degrade to the same mapping rather than returning None:
        # None means "can't tell, don't skip", which in CI meant running twelve
        # judge calls that all 401'd and failed the build. Broke KYC Sentinel's
        # CI exactly that way — scripts/ from the checkout, runtime/ from the
        # v1.1.0 wheel.
        return _fallback_credential_env(cfg)


# Mirrors runtime.provider_dispatch._DEFAULT_API_KEY_ENV. Duplicated on
# purpose and only reachable when that module is too old to ask — the
# alternative is a hard version coupling between scripts/ and the pinned
# runtime, which is the thing the vendoring boundary exists to avoid.
_FALLBACK_API_KEY_ENV = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "groq": "GROQ_API_KEY",
    "openai": "OPENAI_API_KEY",
    "azure_openai": "AZURE_OPENAI_API_KEY",
    "xai": "XAI_API_KEY",
    "google_ai": "GEMINI_API_KEY",
    "ollama": None,
    "vertex_ai": None,
    "bedrock": None,
    "huawei_modelarts": None,
}


def _fallback_credential_env(cfg: dict) -> Optional[str]:
    default_env = _FALLBACK_API_KEY_ENV.get(cfg.get("provider", "openai"), "OPENAI_API_KEY")
    if default_env is None:
        return None
    return cfg.get("api_key_env") or default_env


def provider_models(provider: str) -> list[str]:
    """Sorted model ids the merged registry routes to a given provider —
    e.g. every `ollama` id, for a "are these pulled?" preflight check."""
    registry = load_registry() or {}
    return sorted(
        {
            cfg["id"]
            for cfg in registry.values()
            if cfg.get("provider") == provider and cfg.get("id")
        }
    )


def load_registry() -> Optional[dict]:
    """The merged model registry, or None when runtime/ isn't available.

    Every scripts/*.py file is invoked as `python3 scripts/whatever.py`, which
    puts `scripts/` on sys.path[0] — NOT the repo root — so a bare
    `import runtime` fails in exactly the normal invocation path and this
    would silently fall back forever. (Caught by running verify_system.py for
    real: it reported the registry as unreadable while pytest, which has the
    root on the path, read it fine.) The insert below is the same one several
    scripts/*.py already do before importing runtime.

    runtime/ is still imported lazily and its absence tolerated: scripts/ is
    installed to ~/.agent-framework/scripts and invoked inside tenant repos
    that may carry no runtime/ at all (the vendoring boundary in this module's
    docstring). Resolving a default must not turn that into a hard dependency —
    hence None, and the caller falls back.
    """
    # Cached per cwd: the merge pulls in a tenant models.yaml and tenant.yaml
    # found from the CURRENT directory, so the answer legitimately differs
    # between repos, but re-parsing three YAML files on every lookup does not.
    cwd = str(Path.cwd().resolve())
    if cwd in _REGISTRY_CACHE:
        return _REGISTRY_CACHE[cwd]

    install_root = Path(__file__).resolve().parent.parent
    if str(install_root) not in sys.path:
        sys.path.insert(0, str(install_root))
    registry: Optional[dict]
    try:
        from runtime.llm_gateway import load_model_registry

        registry = load_model_registry() or None
    except Exception:  # fail-open: no runtime/, or an unreadable registry
        registry = None
    _REGISTRY_CACHE[cwd] = registry
    return registry


def _registry_judge_model() -> Optional[str]:
    """The `judge` role's model id from the merged registry, or None."""
    registry = load_registry()
    if not registry:
        return None
    return (registry.get(JUDGE_ROLE) or {}).get("id") or None


def judge_model() -> str:
    """Resolve the eval-judge model: the `judge` role in models.yaml wins;
    AGENT_JUDGE_MODEL applies only when no role declares one.

    The registry is deliberately FIRST. AGENT_JUDGE_MODEL used to win, on the
    theory that an env var is a convenient one-off override — but an env var is
    ambient, and a line in a shell profile is permanent and invisible. A
    developer profile carrying
    `export AGENT_JUDGE_MODEL="claude-3-5-sonnet-20241022"` silently graded
    every local eval with that model while CI, where the variable is unset,
    used the declared `judge` role. Two graders, one threshold, no signal that
    they differed — and the scores are not comparable across judges, which is
    the whole reason `judge_models_used` provenance exists.

    A config file that a shell profile can override is not a source of truth.
    The variable is still honoured where nothing is declared (a scripts-only
    install with no models.yaml), and an ignored value is logged rather than
    dropped silently, so a deliberate override is not merely puzzling.
    """
    env = os.environ.get("AGENT_JUDGE_MODEL", "").strip()
    declared = _registry_judge_model()
    if declared:
        if env and env != declared:
            import logging

            logging.getLogger(__name__).warning(
                "AGENT_JUDGE_MODEL=%r ignored — the `judge` role in models.yaml "
                "declares %r, and the registry wins so a shell profile cannot "
                "silently regrade a repo. Edit the judge role to change it.",
                env,
                declared,
            )
        return declared
    return env or DEFAULT_JUDGE_MODEL


def _repo_root(start: Optional[Path] = None) -> Path:
    """Nearest ancestor holding `.agenticframework/` or `.git/`; else cwd.

    `start` mirrors `runtime.config.repo_root`'s parameter of the same shape.
    It exists so `_load_dotenv`'s standalone fallback can resolve a caller-given
    root without a third copy of this walk.

    A DELIBERATE mirror of runtime.config.repo_root, not an oversight — the same
    arrangement as `_FALLBACK_EXHAUSTION_MARKERS` below, and for the same reason.
    `scripts/` is machine-installed (~/.agent-framework/scripts) while `runtime/`
    is a pip package; they are not guaranteed to be co-located, and these scripts
    run standalone, before anything has put the install root on sys.path. An
    earlier attempt to import runtime here passed the whole local suite — pytest
    puts the repo root on the path — and failed in CI with
    `ModuleNotFoundError: No module named 'runtime'` on the first standalone
    script invocation.

    `test_shared_root_matches_runtime_root` asserts the two stay identical.

    The marker is `.agenticframework` OR `.git`, not `.git` alone: a tenant
    directory must win over a parent repo containing it, or tenant.yaml and
    models.yaml resolve to different directories in one process.
    """
    cwd = start or Path.cwd()
    for parent in [cwd, *cwd.parents]:
        if (parent / ".agenticframework").is_dir() or (parent / ".git").exists():
            return parent
    return cwd


def _iso_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _tenant_id() -> Optional[str]:
    """Read tenant.id from .agenticframework/tenant.yaml if present.
    Prefers a real YAML parse (handles any valid tenant.yaml shape);
    falls back to a line-regex scan if PyYAML isn't installed, since
    several scripts/*.py callers run in minimal environments."""
    tenant_file = _repo_root() / ".agenticframework" / "tenant.yaml"
    if not tenant_file.exists():
        return None
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(tenant_file.read_text())
        return (data or {}).get("tenant", {}).get("id")
    except ImportError:
        try:
            for line in tenant_file.read_text().splitlines():
                if line.strip().startswith("id:"):
                    return line.split(":", 1)[1].strip()
        # fail-open: best-effort tenant-id lookup; None is a valid "no tenant" result,
        # same as the yaml-parse path below
        except Exception:
            pass
        return None
    except Exception:
        return None


# One sync-state file shared by shadow-eval.py, sync-portal-history.py and
# sync-ui-feedback.py — each keeps its own keys inside it. The load/save
# pair below was copied byte-for-byte in all three before this
# consolidation (ReviewFindings-2026-07-18 B2).
SYNC_STATE_FILE = ".agent-rfc/fixtures/sync_state.json"


def _load_sync_state() -> dict:
    """Read the shared sync-state JSON; {} if missing or unreadable.
    Callers use state.get(key, fallback) for their own keys."""
    import json

    path = _repo_root() / SYNC_STATE_FILE
    if not path.exists():
        return {}
    try:
        with path.open() as fh:
            return json.load(fh)
    except Exception:
        return {}


def _save_sync_state(state: dict) -> None:
    import json

    path = _repo_root() / SYNC_STATE_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(state, fh, indent=2)


def _dotenv_value(raw: str) -> str:
    """Parse one .env value: strip an unquoted inline comment, then quotes.

    Inline comments were previously kept as part of the value, so
    `OLLAMA_BASE_URL=http://localhost:11434   # intake route` set the URL to
    everything including the comment. The request then went to
    `http://localhost:11434   # intake route/chat/completions` and came back
    `405 method not allowed` — a failure that reads like a broken endpoint
    rather than a config parse bug. `KEY=value  # note` is a near-universal
    convention, so this hit a normal, correct-looking .env.

    Follows the usual dotenv rule: an inline comment must be preceded by
    whitespace, so a bare `#` inside an unquoted value survives (passwords and
    URL fragments), and a QUOTED value keeps everything between the quotes —
    `PASS="a#b # c"` is `a#b # c`, not `a#b`.
    """
    raw = raw.strip()
    if raw[:1] in ("'", '"'):
        quote = raw[0]
        end = raw.find(quote, 1)
        if end != -1:
            return raw[1:end]
        return raw[1:]  # unterminated quote — take the rest
    for i, ch in enumerate(raw):
        if ch == "#" and (i == 0 or raw[i - 1] in " \t"):
            return raw[:i].strip()
    return raw


def _load_dotenv(root: Optional[Path] = None) -> None:
    """Best-effort load of repo-root .env into os.environ (no overwrite).

    Prefers runtime.config.load_env_file when `runtime` is importable, and only
    then. This used to be the ONLY loader in the codebase, which is why the
    runtime never saw .env at all: scripts got it, workers did not. Delegating
    fixed that — and reintroduced, three lines below `_repo_root`, the exact
    import that `_repo_root`'s docstring exists to warn against. `scripts/` is
    machine-installed and `runtime/` is a pip package; a standalone script run
    from a tenant directory has no `runtime` on sys.path, and an unconditional
    import here kills the process before it does any work.

    That is not hypothetical. It took out the KYC Sentinel judged-eval split run
    for five consecutive daily windows (2026-08-25 → 08-29). The driver read the
    nonzero exit as "judge unreachable" and logged `NO VERDICT — will retry in a
    later window`, so the symptom presented as an exhausted free-tier quota. The
    golden and hallucination suites graded nothing for five days while the log
    said the gate was merely waiting its turn.

    So: delegate when co-located — the shared `_ENV_FILE` that `resolve()` and
    `shadowed_env()` read is worth keeping single-sourced whenever both halves
    are in one process — and mirror the os.environ half when not. The mirror is
    the same arrangement as `_dotenv_value` and `_FALLBACK_EXHAUSTION_MARKERS`,
    pinned by `test_shared_and_runtime_loaders_agree`.
    """
    try:
        from runtime.config import load_env_file
    except ImportError:  # standalone script: runtime/ is not on sys.path
        _load_dotenv_standalone(root)
        return

    load_env_file(root)


def _load_dotenv_standalone(root: Optional[Path] = None) -> None:
    """The os.environ half of `runtime.config.load_env_file`, for script-only
    processes. A DELIBERATE mirror — see `_load_dotenv`.

    `_ENV_FILE` is deliberately not reproduced. Nothing in a scripts-only
    process can read it: the only readers are `runtime.config.resolve` and
    `shadowed_env`, and if those were importable this branch would not run.
    Keeping a second copy of that dict here would create a state that agrees
    with the runtime's by luck rather than by construction.
    """
    path = _repo_root(root) / ".env"
    if not path.exists():
        return
    try:
        lines = path.read_text().splitlines()
    except OSError:  # fail-open: .env is optional convenience, never fatal
        return

    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, raw = line.partition("=")
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export "):].strip()
        if not key:
            continue
        if key not in os.environ:
            os.environ[key] = _dotenv_value(raw)


def _phoenix_request(
    phoenix_endpoint: str, path: str, *, method: str = "get", **kwargs: Any
) -> Any:
    """One request path for the Phoenix REST API.

    `_phoenix_get` and `_phoenix_post` differed only in the httpx verb and
    whether the payload went as `params` or `json` — the URL construction,
    timeout, status check and error wrapping were identical. Wrapping the error
    with the failing PATH is the part worth keeping in one place: an httpx
    exception alone says a request failed without saying which, and these calls
    are made from three different sync scripts.
    """
    import httpx

    url = f"{phoenix_endpoint.rstrip('/')}{path}"
    try:
        resp = getattr(httpx, method)(url, timeout=30.0, **kwargs)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        raise RuntimeError(f"Phoenix API error [{path}]: {exc}") from exc


def _phoenix_get(
    phoenix_endpoint: str, path: str, params: Optional[dict] = None
) -> Any:
    """GET against a Phoenix REST endpoint."""
    return _phoenix_request(phoenix_endpoint, path, method="get", params=params)


def _phoenix_post(phoenix_endpoint: str, path: str, body: dict) -> Any:
    """POST against a Phoenix REST endpoint."""
    return _phoenix_request(phoenix_endpoint, path, method="post", json=body)


# ── Loading a hyphen-named script as a module ────────────────────────────────


_SCRIPT_MODULE_CACHE: dict[str, Any] = {}


def load_script(name: str, *, cache: bool = True) -> Any:
    """Import `scripts/<name>.py` as a module.

    Several scripts are named with hyphens (`run-evals.py`,
    `promote-learning.py`), which are not importable identifiers, so every
    caller hand-rolled the same four lines of importlib. Fourteen sites did:
    eleven tests, two production scripts, and the security harness — including
    two that had independently reinvented the caching below.

    Cached by default. `run-evals.py` does real work at import (resolves the
    model registry, reads .env) and the security harness loads it once per eval
    control, so re-executing it three times per run was waste that also made
    those controls sensitive to import order.

    Pass `cache=False` where a test needs a genuinely fresh module.
    """
    if cache and name in _SCRIPT_MODULE_CACHE:
        return _SCRIPT_MODULE_CACHE[name]
    import importlib.util

    path = Path(__file__).resolve().parent / f"{name}.py"
    if not path.exists():
        raise FileNotFoundError(f"no such script: {path}")
    # Keep the hyphen-free module name callers already used ("run_evals"), so
    # anything reaching into sys.modules by that name still finds it.
    spec = importlib.util.spec_from_file_location(name.replace("-", "_"), path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    if cache:
        _SCRIPT_MODULE_CACHE[name] = module
    return module



# ── Tenant fixture paths ──────────────────────────────────────────────────────


# Suite → the fixture files it uses. Lives here rather than in run-evals.py
# because run-evals is not the only reader: promote-learning.py resolved the
# same two paths by hardcoding "golden_evals.json" and
# "custom_judge_criteria.json", so renaming a fixture in the table would have
# left it silently writing to the old name. One table, one place to change.
EVALS_FILE = {
    "golden": "golden_evals.json",
    "fairness": "fairness_evals.json",
    "hallucination": "hallucination_evals.json",
    "rag_poison": "rag_poison_evals.json",
}
BASE_FIXTURE = {
    "fairness": "fairness_evals_base.json",
    "hallucination": "hallucination_evals_base.json",
    "rag_poison": "rag_poison_base.json",
}
RESULTS_FILE = {
    "golden": "eval_results.json",
    "fairness": "fairness_eval_results.json",
    "hallucination": "hallucination_eval_results.json",
    "adversarial": "adversarial_eval_results.json",
    "rag_poison": "rag_poison_eval_results.json",
}
GOLDEN_CRITERIA_FILE = "custom_judge_criteria.json"


def fixtures_path(name: str, *, mkdir: bool = False) -> Path:
    """`<repo>/.agent-rfc/fixtures/<name>`.

    Nine call sites spelled this out from `_repo_root()`, and two of them also
    created the directory first. That is a layout the tenant scaffold owns, so
    every copy is a place that has to be found if it ever moves — and the two
    that called mkdir were the two whose absence was a crash rather than an
    empty result, which is not a distinction worth rediscovering per script.

    `mkdir` is opt-in: a reader resolving a path to check whether a fixture
    exists must not bring the directory into being as a side effect.
    """
    path = _repo_root() / ".agent-rfc" / "fixtures" / name
    if mkdir:
        path.parent.mkdir(parents=True, exist_ok=True)
    return path


# ── Request pacing ────────────────────────────────────────────────────────────


class RateLimiter:
    """Space calls at least `1/rps` apart, by sleeping before the next one.

    This is PROACTIVE pacing, and it is a different thing from the 429 retry in
    `cost_router._post_with_retry` — which is REACTIVE: it waits only after a
    provider has already refused. Both are needed and neither substitutes for
    the other. A free-tier key with a per-minute cap will refuse a burst of 12
    judge calls faster than the retry budget (4 attempts) can absorb, so the
    whole suite errors out; and per `run_scorecard`, a suite where every case
    errored reports "judge was unreachable" and returns 0. The run does not
    fail — it never grades at all, which is why the symptom reads as a stuck
    eval rather than a broken one.

    A limiter with `rpm <= 0` is a no-op object rather than None, so callers
    need no branch at the call site.
    """

    __slots__ = ("_last", "_min_interval")

    def __init__(self, rpm: float = 0.0) -> None:
        self._min_interval = 60.0 / rpm if rpm and rpm > 0 else 0.0
        self._last = 0.0

    @property
    def enabled(self) -> bool:
        return self._min_interval > 0.0

    def wait(self) -> float:
        """Block until the next call is allowed. Returns seconds actually slept."""
        if not self.enabled:
            return 0.0
        import time as _time

        now = _time.monotonic()
        earliest = self._last + self._min_interval
        slept = 0.0
        if self._last and now < earliest:
            slept = earliest - now
            _time.sleep(slept)
        self._last = _time.monotonic()
        return slept


def env_number(var: str, default: "float | int", *, cast: Any = float) -> "float | int":
    """A numeric env var, or `default` — never an exception.

    `int(os.environ.get(VAR, "50000"))` looks defaulted and is not: the default
    applies only when the key is ABSENT. `VAR=""` — declared with no value,
    which is what a k8s manifest or a CI matrix produces for an unset input —
    reaches int() as the empty string and raises. At module level, as in
    circuit_breaker.py, that turns an empty environment variable into an
    ImportError for every consumer of the module.

    Falls back rather than raising, and says so on stderr. For a spend cap the
    documented default is the safe value; refusing to load the breaker at all
    is not, because the callers treat an unloadable breaker as "proceed
    unmetered".
    """
    raw = (os.environ.get(var) or "").strip()
    if not raw:
        return default
    try:
        return cast(raw)
    except (TypeError, ValueError):
        print(
            f"⚠️  {var}={raw!r} is not a number — using {default}",
            file=sys.stderr,
        )
        return default


def rate_limiter_from_env(var: str = "EVAL_RPM", default: float = 0.0) -> RateLimiter:
    """Build a RateLimiter from an env var naming requests-per-minute.

    Unset or unparseable means no pacing — the historical behaviour, and the
    right default for a paid key where pacing only costs wall-clock. Set it to
    the provider's documented free-tier limit to make a long suite complete
    instead of exhausting the retry budget partway through.
    """
    raw = (os.environ.get(var) or "").strip()
    if not raw:
        return RateLimiter(default)
    try:
        return RateLimiter(float(raw))
    except ValueError:
        print(f"⚠️  {var}={raw!r} is not a number — ignoring, no pacing applied")
        return RateLimiter(default)
