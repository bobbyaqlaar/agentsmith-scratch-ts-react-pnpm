"""
runtime/moderation.py — pluggable output moderation hook (SEC-MOD-001).

Modes (MODERATION_HOOK env):
  off       — never call hook
  optional  — default; no hook → skip/allow; hook present → enforce
  required  — no hook → ModerationHookRequiredError (regulated tenants / strict CI)

Two ways to supply the classifier:

1. `register_output_moderator(fn)` at worker startup — imperative, in-process.
2. **Declared** in `.agenticframework/tenant.yaml` (or `MODERATION_HOOK_PATH`):

       moderation:
         hook: "agents.moderation:classify_output"

   Resolved lazily on first use and registered automatically.

The declaration exists because of TestbedFeedback-2026-07-21 G10: the
SEC-MOD-001 harness runs in a different process from the worker, so an
imperative registration is invisible to it — which made
`MODERATION_HOOK=required` impossible to satisfy in CI, i.e. exactly the
setting regulated tenants are told to use. A committed declaration is
something the harness can read, import, and smoke-test.

Critically, the declaration is also what the RUNTIME loads. If the harness
checked a config key that production ignored, it would be certifying the
wrong thing — the check has to bind to the same source of truth the
gateway uses.
"""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

ModeratorFn = Callable[[str], "ModerationResult"]

_moderator: Optional[ModeratorFn] = None
# Cache of the resolved declaration so a missing/broken hook isn't re-imported
# on every LLM call. Cleared by reset_output_moderator().
_declared_resolved = False


class ModerationHookRequiredError(RuntimeError):
    """Raised when MODERATION_HOOK=required and no moderator is registered."""


class ModerationBlockedError(RuntimeError):
    """Raised when the registered moderator rejects output text."""

    def __init__(self, message: str, reasons: list[str] | None = None) -> None:
        super().__init__(message)
        self.reasons = reasons or []


@dataclass(frozen=True)
class ModerationResult:
    allowed: bool
    reasons: list[str] = field(default_factory=list)
    skipped: bool = False


def reset_output_moderator() -> None:
    """Clear tenant moderator (and the declaration cache) — for tests."""
    global _moderator, _declared_resolved
    _moderator = None
    _declared_resolved = False


def register_output_moderator(fn: ModeratorFn) -> None:
    """Register a tenant classifier for LLM output text."""
    global _moderator
    _moderator = fn


def get_output_moderator() -> Optional[ModeratorFn]:
    return _moderator


# ── Declared hook (SEC-MOD-001 evidence path, G10) ───────────────────────────


class ModerationHookImportError(RuntimeError):
    """Raised when a declared moderation hook cannot be imported or is not
    callable — a broken declaration must be loud, never a silent skip."""


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


def declared_hook_path() -> Optional[str]:
    """Dotted path of the declared moderator, or None.

    `MODERATION_HOOK_PATH` wins so a deployment can point at a different
    classifier without editing the committed tenant.yaml.
    """
    env = os.environ.get("MODERATION_HOOK_PATH", "").strip()
    if env:
        return env

    tenant_yaml = _repo_root() / ".agenticframework" / "tenant.yaml"
    if not tenant_yaml.exists():
        return None
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(tenant_yaml.read_text()) or {}
    except Exception:  # fail-open: an unparsable tenant.yaml means "not declared"
        return None
    hook = (data.get("moderation") or {}).get("hook")
    return str(hook).strip() if hook else None


def load_declared_moderator() -> Optional[ModeratorFn]:
    """Import the declared hook and return it, or None when undeclared.

    Raises ModerationHookImportError when a declaration exists but does not
    resolve — used by both the runtime and the SEC-MOD-001 harness so the
    two agree on what "the tenant has a classifier" means.
    """
    path = declared_hook_path()
    if not path:
        return None

    module_name, sep, attr = path.partition(":")
    if not sep or not attr:
        raise ModerationHookImportError(
            f"moderation hook {path!r} must be 'module.path:callable'"
        )

    # A declared hook is tenant code by definition, so it is resolved
    # relative to the tenant repo root. Without this the harness — which
    # runs from the framework install, not the tenant checkout — could not
    # import it, and every tenant would have to set PYTHONPATH by hand.
    root = str(_repo_root())
    if root not in sys.path:
        sys.path.insert(0, root)

    try:
        import importlib

        module = importlib.import_module(module_name)
    except Exception as exc:
        raise ModerationHookImportError(
            f"cannot import moderation hook module {module_name!r}: {exc}"
        ) from exc

    fn = getattr(module, attr, None)
    if fn is None:
        raise ModerationHookImportError(
            f"moderation hook {path!r}: {module_name!r} has no attribute {attr!r}"
        )
    if not callable(fn):
        raise ModerationHookImportError(f"moderation hook {path!r} is not callable")
    return fn


def _ensure_declared_moderator() -> None:
    """Register the declared hook if nothing is registered yet (idempotent)."""
    global _moderator, _declared_resolved
    if _moderator is not None or _declared_resolved:
        return
    _declared_resolved = True
    fn = load_declared_moderator()
    if fn is not None:
        _moderator = fn
        logger.info("registered declared output moderator: %s", declared_hook_path())


def resolve_mode() -> str:
    from runtime.config import resolve_choice

    return resolve_choice(
        "moderation.mode",
        env_var="MODERATION_HOOK",
        allowed=("off", "optional", "required"),
        fallback="optional",
    )


def apply_output_moderation(
    text: str,
    *,
    raise_on_block: bool = False,
    mode: Optional[str] = None,
    use_declared: bool = True,
) -> ModerationResult:
    """
    Run the registered moderator (if any) against ``text``.

    - off: always allow (skipped)
    - optional + no hook: allow (skipped)
    - required + no hook: raise ModerationHookRequiredError
    - hook present + blocked: return allowed=False; optionally raise

    When nothing is registered imperatively, a hook declared in
    `.agenticframework/tenant.yaml` is resolved and registered first.
    `use_declared=False` skips that — only the SEC-MOD-001 runner needs it,
    to prove `required` still rejects a genuinely hook-less tenant.
    """
    active = mode or resolve_mode()
    if active == "off":
        return ModerationResult(allowed=True, skipped=True)

    if _moderator is None and use_declared:
        _ensure_declared_moderator()

    if _moderator is None:
        if active == "required":
            raise ModerationHookRequiredError(
                "MODERATION_HOOK=required but no output moderator registered"
            )
        return ModerationResult(allowed=True, skipped=True)

    result = _moderator(text)
    if not result.allowed:
        if raise_on_block:
            raise ModerationBlockedError(
                f"output blocked: {', '.join(result.reasons) or 'policy'}",
                reasons=list(result.reasons),
            )
        return ModerationResult(allowed=False, reasons=list(result.reasons), skipped=False)
    return ModerationResult(allowed=True, reasons=list(result.reasons), skipped=False)
