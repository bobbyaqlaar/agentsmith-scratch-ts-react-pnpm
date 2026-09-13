"""
runtime/config.py — where configuration reaches the runtime from.

Two channels, one precedence, stated once.

`.env` WAS NEVER LOADED BY THE RUNTIME. `scripts/` loads it (via
`_shared._load_dotenv`); `runtime/` — worker, gateway, tenancy, redactor, dead
letter, idempotency — read `os.environ` directly and loaded nothing. So the
runtime worked locally only because the launching shell happened to have the
values exported, which is why `~/.agentsmith-split_eval.sh` opens with
`set -a; . ./.env; set +a`: a per-script workaround for a gap here. Anything
that forgot the incantation silently got defaults.

`tenant.yaml` DECLARED POLICY NOTHING READ. The scaffold has shipped
`budget.monthly_usd_cap`, `workflow.task_queue`, `tenant.owner` and `tenant.id`
since the beginning and only `moderation.hook` was ever wired. KYC Sentinel
declared a $5 monthly cap while the gateway enforced `AGENT_MONTHLY_USD_CAP`
or, unset, a $150 default — and `.env` isn't deployed to Cloud Run, so in
production the declared cap became the default: a 30x breach resting on
somebody remembering `--set-env-vars`. A declared-but-unenforced control is
worse than an absent one, because it reads as a control in an audit.

PRECEDENCE, and why it is not simply "environment wins".

    1. explicit argument          a caller that already knows
    2. .env                       a file the tenant controls, gitignored
    3. tenant.yaml                the declared policy, committed and reviewed
    4. ambient os.environ         ONLY when nothing above declares the key,
                                  or the key is named in `env_overrides`
    5. documented default, or raise

The distinction that matters is between a value the operator DECLARED for this
deployment and one that merely happens to be exported in the shell that
launched the process. Both arrive as `os.environ` and are technically
indistinguishable — so the framework decides by asking whether any file
declares the key.

This was learned the hard way. `install-ai-stack.sh` exported AGENT_OWNER_ID
into ~/.zshrc, advertised as "set once, applies to all projects on this
machine". Under an environment-wins rule that one line silently outranked every
tenant's declared `tenant.owner`, on every repo, forever — and CI, which has no
shell profile, got nothing at all. An ambient channel wearing a deliberate
one's clothes.

So a declaration wins over the ambient environment, and the ambient value is
not discarded silently: `shadowed_env()` reports every key where one was
ignored, and `configure_tracing` / the worker log it once at startup. Silently
ignoring an operator's variable is its own trap.

Where a deployment genuinely must override a declared key — raising a cap
without a redeploy — it says so, in the file, where it can be reviewed:

    env_overrides: [AGENT_MONTHLY_USD_CAP]

Secrets are unaffected. An API key has no declaration anywhere, so rule 4
applies and `--set-secrets` reaches it exactly as before.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

logger = logging.getLogger(__name__)

_UNSET = object()

_TRUE = {"1", "true", "yes", "on"}


def as_bool(value: Any) -> bool:
    """Coerce a flag from either channel.

    YAML gives a real bool; the environment gives a string. Unrecognised text
    is False rather than truthy — `TOOL_ALLOWLIST_STRICT=maybe` must not turn a
    deny-by-default guard on by accident, nor off: False is this flag's
    documented default in every case it is used for.
    """
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in _TRUE


def repo_root(start: Optional[Path] = None) -> Path:
    """Nearest ancestor holding `.agenticframework/` or `.git/`."""
    cwd = start or Path.cwd()
    for parent in [cwd, *cwd.parents]:
        if (parent / ".agenticframework").is_dir() or (parent / ".git").exists():
            return parent
    return cwd


def _dotenv_value(raw: str) -> str:
    """Parse one `.env` value: quotes, then an unquoted inline comment.

    A DELIBERATE mirror of `scripts/_shared._dotenv_value`, pinned by
    `test_dotenv_parsers_agree`. Neither module can import the other: `runtime/`
    ships as a pip package that must not depend on machine-installed scripts,
    and `scripts/` run standalone before anything puts `runtime/` on sys.path —
    importing across that boundary in the other direction is what broke CI on
    every standalone script two commits ago.

    This started as a cruder re-implementation written without checking whether
    one already existed, and the two disagreed on an unterminated quote within a
    day. The logic below is `_shared`'s, which is the considered one:

      * `#` only starts a comment at a word boundary, so `http://h:1#frag`
        keeps its fragment while `KEY=v  # note` does not keep the note. That
        distinction cost a `405 method not allowed` that read like a broken
        endpoint rather than a config parse bug.
      * a QUOTED value keeps everything between the quotes, so
        `PASS="a#b # c"` is `a#b # c`.
      * an unterminated quote takes the rest of the line rather than the quote
        character itself.
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
    return raw.strip()


_ENV_FILE: dict[str, str] = {}
_SHADOWED: dict[str, str] = {}


def load_env_file(root: Optional[Path] = None) -> int:
    """Parse repo-root `.env`. Returns how many keys it declared.

    Values are kept HERE, not merged blindly into `os.environ`, because once
    merged there is no way to tell a key the tenant put in `.env` from one that
    leaked in from a login shell — and those two deserve opposite treatment.

    They are also mirrored into `os.environ` for third-party libraries that read
    it directly (httpx proxies, the OTLP exporter, psycopg), but only where the
    variable is not already set: the mirror must not change what a container was
    configured with. `resolve()` reads the dict, so the mirror's precedence does
    not affect the framework's own resolution.
    """
    path = repo_root(root) / ".env"
    if not path.exists():
        return 0
    try:
        lines = path.read_text().splitlines()
    except OSError:  # fail-open: .env is optional convenience, never fatal
        return 0

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
        value = _dotenv_value(raw)
        _ENV_FILE[key] = value
        if key not in os.environ:
            os.environ[key] = value
    return len(_ENV_FILE)


def shadowed_env() -> dict[str, str]:
    """Keys where an ambient environment value was ignored in favour of a file.

    Read this at startup and log it. An operator who exports something and sees
    no effect, with nothing said, will conclude the framework is broken — and
    they will be half right.
    """
    return dict(_SHADOWED)


_CACHE: dict[Path, dict] = {}


def tenant_config(root: Optional[Path] = None, *, refresh: bool = False) -> dict:
    """Parsed `.agenticframework/tenant.yaml`, or `{}`.

    Cached per root: this is read on several hot-ish paths and the file cannot
    change under a running worker. `refresh=True` is for tests.
    """
    base = repo_root(root)
    if not refresh and base in _CACHE:
        return _CACHE[base]

    path = base / ".agenticframework" / "tenant.yaml"
    doc: dict = {}
    if path.exists():
        try:
            import yaml  # type: ignore

            parsed = yaml.safe_load(path.read_text())
            doc = parsed if isinstance(parsed, dict) else {}
        except Exception as exc:
            # LOUD. Fail-open is right — a worker must not refuse to start over
            # a stray tab — but everything this file declares vanishes at once:
            # the budget cap reverts to the framework's default, the security
            # posture to its defaults, the owner to git. Every one of those is a
            # control, and they would all revert together, silently, from a typo.
            # `tenant.id` is the only key that announces itself, by raising.
            logger.error(
                "tenant.yaml at %s could NOT be parsed (%s) — EVERY declaration "
                "in it is being ignored: budget cap, security posture, owner, "
                "task queue. Framework defaults are in force instead.",
                path,
                exc,
            )
            doc = {}
    _CACHE[base] = doc
    return doc


def config_get(dotted: str, root: Optional[Path] = None) -> Any:
    """`config_get("budget.monthly_usd_cap")` → the declared value, or None."""
    node: Any = tenant_config(root)
    for part in dotted.split("."):
        if not isinstance(node, dict):
            return None
        node = node.get(part)
        if node is None:
            return None
    return node


_OVERRIDES: dict[str, Any] = {}


@contextmanager
def override(**settings: Any) -> Iterator[None]:
    """Deliberate, in-process, scoped override. Precedence 1, same as an
    explicit argument.

    The strict rule below cannot distinguish an operator's `PROMPT_GUARD=off`
    for one debug run from a line that leaked in from a login shell — both are
    just `os.environ`. So it treats a declared key as declared, which is right
    for the ambient case and leaves genuine deliberate overrides with nowhere
    to go. This is that channel: visible in the code that does it, scoped to a
    block, and impossible to leave lying around in a shell profile.

        with override(**{"security.prompt_guard": "off"}):
            ...

    Keys are dotted config paths, so `override(security__prompt_guard=...)`
    is not a thing — pass a dict.
    """
    previous = {k: _OVERRIDES.get(k, _UNSET) for k in settings}
    _OVERRIDES.update(settings)
    try:
        yield
    finally:
        for key, was in previous.items():
            if was is _UNSET:
                _OVERRIDES.pop(key, None)
            else:
                _OVERRIDES[key] = was


def env_overrides(root: Optional[Path] = None) -> set[str]:
    """Env vars this tenant permits to outrank its own declarations.

    Declared in the file so the exception is as reviewable as the rule:

        env_overrides: [AGENT_MONTHLY_USD_CAP]
    """
    declared = config_get("env_overrides", root)
    if isinstance(declared, str):
        return {declared.strip()}
    if isinstance(declared, (list, tuple, set)):
        return {str(x).strip() for x in declared if str(x).strip()}
    return set()


def _cast(value: Any, cast: Any, source: str, dotted: str) -> Any:
    if cast is None:
        return value
    try:
        return cast(value)
    except (TypeError, ValueError) as exc:
        # Loud. A malformed value must not fall through to a lower-precedence
        # source, or whoever set it is silently ignored.
        raise ValueError(f"{dotted}: {value!r} from {source} is not valid") from exc


def resolve_choice(
    dotted: str,
    *,
    env_var: Optional[str] = None,
    allowed: Iterable[str],
    fallback: str,
) -> str:
    """Resolve a setting whose value is one of a fixed set of WORDS.

    Two things go wrong with `str(resolve(...)).strip().lower()`, and both were
    silent.

    YAML 1.1 reads a bare `off` as the boolean False — and `off` is a
    DOCUMENTED value of security.input_guardrail, security.prompt_guard and
    moderation.mode. A tenant writing the value the docs told them to write got
    False, which matches no mode, so their declaration was discarded and the
    fallback applied instead. `on`, `no` and `yes` coerce the same way.

    The boolean is NOT translated back to a word. Mapping False to "off" would
    read `prompt_guard: false` as an instruction to disable the guard, which is
    a way to turn a control off by writing something that was never a valid
    value for it. It says what happened and keeps the fail-closed fallback.

    And an unrecognised value was replaced in silence. A tenant who typed
    `warnn` got the blocking default and no indication their policy had been
    ignored — the posture an auditor reads in tenant.yaml was not the posture
    in force.
    """
    from runtime.environment import warn_once

    raw = resolve(dotted, env_var=env_var, default="")
    options = sorted(allowed)

    if isinstance(raw, bool):
        warn_once(
            f"config-choice-bool:{dotted}",
            f"{dotted} was read as the YAML boolean {raw!r} — YAML 1.1 parses "
            f"bare off/on/yes/no as booleans, and this setting takes a word. "
            f'Quote it (`{dotted.split(".")[-1]}: "off"`). '
            f"Using {fallback!r}; accepted: {', '.join(options)}.",
        )
        return fallback

    text = str(raw).strip().lower()
    if not text:
        return fallback
    if text in options:
        return text

    warn_once(
        f"config-choice-unknown:{dotted}",
        f"{dotted}={text!r} is not a recognised value. Using {fallback!r}; "
        f"accepted: {', '.join(options)}.",
    )
    return fallback


def resolve(
    dotted: str,
    *,
    explicit: Any = None,
    env_var: Optional[str] = None,
    default: Any = _UNSET,
    cast: Any = None,
    root: Optional[Path] = None,
) -> Any:
    """One setting, resolved by the precedence in this module's docstring.

    `default=_UNSET` means there is no safe default: unresolved raises rather
    than inventing one. Use it for anything an auditor would read as a control.
    """
    if explicit is not None:
        return _cast(explicit, cast, "the caller", dotted)

    if dotted in _OVERRIDES:
        return _cast(_OVERRIDES[dotted], cast, "an in-process override", dotted)

    ambient = os.environ.get(env_var, "").strip() if env_var else ""

    # An override the tenant has explicitly permitted outranks its own file.
    if env_var and ambient and env_var in env_overrides(root):
        return _cast(ambient, cast, f"{env_var} (permitted override)", dotted)

    # 2. .env — a file the tenant controls. Outranks the ambient shell, which
    #    is the whole point: an accidental export must not beat a declaration.
    if env_var and env_var in _ENV_FILE and _ENV_FILE[env_var].strip():
        if ambient and ambient != _ENV_FILE[env_var]:
            _SHADOWED[env_var] = ambient
        return _cast(_ENV_FILE[env_var], cast, f"{env_var} in .env", dotted)

    # 3. tenant.yaml — the declared policy.
    declared = config_get(dotted, root)
    if declared is not None:
        if ambient:
            _SHADOWED[env_var] = ambient  # type: ignore[index]
        return _cast(declared, cast, f"{dotted} in tenant.yaml", dotted)

    # 4. Ambient environment — reached only when NO file declares this key,
    #    which is the normal and correct path for secrets and for anything a
    #    platform injects with --set-env-vars.
    if ambient:
        return _cast(ambient, cast, env_var or "the environment", dotted)

    if default is _UNSET:
        raise LookupError(
            f"{dotted} is not set. Pass it explicitly"
            + (f", set {env_var}" if env_var else "")
            + f", or declare `{dotted}` in .agenticframework/tenant.yaml."
        )
    return default
