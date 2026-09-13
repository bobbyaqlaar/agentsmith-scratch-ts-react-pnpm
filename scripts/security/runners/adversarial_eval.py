"""
scripts/security/runners/adversarial_eval.py — SEC-ADV-001.

Prompt-injection / jailbreak probes, scored locally by `runtime.prompt_guard`.
The body is `_shared.guard_suite`, shared with the RAG poisoning control: both
count misses against a ceiling rather than averaging quality against a floor,
and keeping one implementation means the two cannot drift on that number.
"""

from __future__ import annotations

from typing import Any

from security.registry import ControlSpec
from security.report import ControlResult
from security.runners._shared import guard_suite


def run(control: ControlSpec, ctx: dict[str, Any]) -> ControlResult:
    return guard_suite(control, ctx, "adversarial", "score_adversarial_case")
