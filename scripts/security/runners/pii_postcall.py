from __future__ import annotations

from typing import Any

from security.registry import ControlSpec
from security.report import ControlResult
from security.runners._shared import verify_system


def run(control: ControlSpec, ctx: dict[str, Any]) -> ControlResult:
    """SEC-PII-002 — post-call redaction, verified by verify_system.

    ENVIRONMENT=staging is forced by the shared helper: the redaction check
    self-disables under `development`, so running it from a developer shell
    would report Met while checking nothing.
    """
    return verify_system(control, ctx, "--check-redaction")
