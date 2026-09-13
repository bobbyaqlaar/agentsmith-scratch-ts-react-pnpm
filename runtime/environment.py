"""
runtime/environment.py — canonical $ENVIRONMENT resolver.

Before this module existed, runtime/trace_redactor.py and
scripts/multi_agent_system.py each read os.environ.get("ENVIRONMENT", ...)
independently, with different fallback philosophy for the same missing/
unrecognized-value condition: trace_redactor.py defaulted to "development"
(the least restrictive redaction profile — full unredacted payloads
exported), while multi_agent_system.py's checkpointer selector only hard-
errors when ENVIRONMENT is explicitly "staging" or "production" with no
DATABASE_URL set — an unrecognized value (e.g. a typo'd "produciton") fell
through to the same permissive MemorySaver path with nothing louder than a
stderr warning (Product_Archive.md 2.8).

get_environment() is fail-closed: missing or unrecognized values resolve to
"production" (the most restrictive profile/path), not "development". Both
callers should import this instead of reading os.environ directly.
"""

from __future__ import annotations

import logging
import os
from typing import Iterable

logger = logging.getLogger(__name__)

_degraded_warned: set[str] = set()

_ALIASES = {
    "development": "development",
    "dev": "development",
    "testing": "development",
    "test": "development",
    "staging": "staging",
    "stage": "staging",
    "production": "production",
    "prod": "production",
}


def get_environment() -> str:
    """Returns one of "development", "staging", "production".

    Fail-closed: an empty/unset ENVIRONMENT, or a value that doesn't match a
    known alias, resolves to "production" — never silently to "development".
    """
    raw = os.environ.get("ENVIRONMENT", "").strip().lower()
    return _ALIASES.get(raw, "production")


def env_choice(var: str, *, default: str, allowed: Iterable[str]) -> str:
    """A backend selector read the same way everywhere.

    Three selectors read three env vars for the same kind of choice and
    disagreed about the same input. `VECTOR_BACKEND=""` fell back to the
    default; `BUDGET_BACKEND=""` and `IDEMPOTENCY_BACKEND=""` raised. A
    variable declared with no value is what a k8s manifest or a CI matrix
    produces for an unset input, so two of the three crashed on a shape the
    third treated as ordinary.

    Empty means UNSET here. A non-empty value that is not allowed still raises,
    which is the other half: a typo'd backend name must not quietly resolve to
    a default the operator did not choose.
    """
    raw = os.environ.get(var, "").strip().lower()
    if not raw:
        return default
    options = list(allowed)
    if raw not in options:
        raise ValueError(
            f"Unknown {var}={raw!r}. Use one of: {', '.join(sorted(options))}."
        )
    return raw


def warn_once(key: str, message: str, level: int = logging.WARNING) -> None:
    """Log `message` the first time `key` is seen, and never again.

    The call sites for these warnings are constructors and resolvers that run
    per worker, per store, or inside a loop, so "say it" has to mean "say it
    once" or it says nothing anyone reads.
    """
    if key in _degraded_warned:
        return
    _degraded_warned.add(key)
    logger.log(level, "%s", message)


def warn_degraded_default(key: str, message: str) -> None:
    """Say once that a default has quietly downgraded a control.

    Several defaults in this framework pick the ephemeral or the fake option:
    an in-process budget ledger, an in-process vector index, a hash embedder
    with no semantic meaning. Each is the right choice for CI and a laptop, and
    each is chosen by DOING NOTHING — so the deployments most likely to be
    running one are the ones that never made a decision about it.

    ERROR outside development because these fail invisibly: no exception, no
    empty result, just a control that has stopped meaning what it says. Once
    per key because the call sites sit in constructors and loops.
    """
    environment = get_environment()
    level = logging.ERROR if environment in {"staging", "production"} else logging.INFO
    warn_once(key, f"{message} [environment={environment}]", level)
