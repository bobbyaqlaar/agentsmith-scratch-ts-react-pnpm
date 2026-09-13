"""
runtime/input_guardrail.py — pre-call PII scrubbing for LLM prompts.

Symmetric to trace_redactor.py (post-call observability scrubbing): this
module runs **before** provider invoke so personal data is masked in the
decision path (UAE PDPL / FIXES Security & Guardrails).

Modes (INPUT_GUARDRAIL env, else environment-derived default):
  off      — no-op
  default  — framework regex scrubbers (Emirates ID, email, phone, cards)
  custom   — tenant callback registered via register_input_guardrail()

Default when unset: off in development, default in staging/production.
"""

from __future__ import annotations

import copy
import logging
from collections.abc import Callable
from typing import Any, Optional

from runtime.environment import get_environment

logger = logging.getLogger(__name__)

# ── Types ─────────────────────────────────────────────────────────────────────

ScrubFn = Callable[[str], tuple[str, dict[str, int]]]

_custom_scrubber: Optional[ScrubFn] = None

# The shapes themselves live in runtime/pii_patterns.py, shared with
# trace_redactor.py — the pre-export half of this control, which knew about
# neither Emirates IDs nor phone numbers while this half did. Same extraction
# runtime/luhn.py got, for the same reason.
from runtime.pii_patterns import (  # noqa: E402
    CARD_CANDIDATE as _CARD_CANDIDATE,
    EMAIL as _EMAIL,
    EMIRATES_ID_DIGITS as _EMIRATES_ID_DIGITS,
    EMIRATES_ID_HYPHEN as _EMIRATES_ID_HYPHEN,
    PHONE as _PHONE,
    ascii_digits as _ascii_digits,
)


# Shared with trace_redactor.py — one Luhn implementation for both the
# pre-call guard and the post-call redactor (ReviewFindings-2026-07-18 B1).
from runtime.luhn import luhn_valid as _luhn_valid  # noqa: E402 — sited beside the comment explaining the sharing


def reset_input_guardrail() -> None:
    """Clear tenant callback — for tests."""
    global _custom_scrubber
    _custom_scrubber = None


def register_input_guardrail(fn: ScrubFn) -> None:
    """Replace default scrubbing when mode=custom (or call from tenant init)."""
    global _custom_scrubber
    _custom_scrubber = fn


MODES = ("off", "default", "custom")


def resolve_mode() -> str:
    """off | default | custom. Unset means off in development, default elsewhere.

    Goes through resolve_choice so that `security.input_guardrail: off` in
    tenant.yaml — the value this module's own docstring documents — is not
    silently discarded as the YAML boolean False, and so an unrecognised value
    is reported rather than quietly replaced.
    """
    from runtime.config import resolve_choice

    fallback = "off" if get_environment() == "development" else "default"
    return resolve_choice(
        "security.input_guardrail",
        env_var="INPUT_GUARDRAIL",
        allowed=MODES,
        fallback=fallback,
    )


def scrub_text(text: str, mode: Optional[str] = None) -> tuple[str, dict[str, int]]:
    """Scrub a single string. Returns (scrubbed_text, counts_by_type)."""
    active = mode or resolve_mode()
    if active == "off":
        return text, {}
    if active == "custom":
        if _custom_scrubber is None:
            logger.warning(
                "INPUT_GUARDRAIL=custom but no scrubber registered — leaving text unchanged"
            )
            return text, {}
        return _custom_scrubber(text)
    return _default_scrub(text)


def _default_scrub(text: str) -> tuple[str, dict[str, int]]:
    counts: dict[str, int] = {}
    out = text

    # MATCH ON THE ASCII-NORMALISED COPY, SPLICE INTO THE ORIGINAL. `\d`
    # already matched Arabic-Indic digits, so cards were caught; the Emirates ID
    # patterns anchor on a literal `784` and matched nothing when the digits
    # were written `٧٨٤` — which is how a person writes them in Arabic, in a
    # framework whose market is the UAE. `ascii_digits` is a per-character
    # mapping, so a span in the normalised copy addresses the same characters
    # in the text the user actually wrote, and the original is what gets
    # rewritten.
    probe = _ascii_digits(out)

    # Order matters: the hyphenated Emirates ID pattern runs before the
    # bare-digit one so the more specific form is consumed first.
    spans: list[tuple[int, int, str]] = []
    taken: list[tuple[int, int]] = []

    def _claim(start: int, end: int) -> bool:
        if any(start < e and s_ < end for s_, e in taken):
            return False
        taken.append((start, end))
        return True

    for pattern, label, replacement in (
        (_EMIRATES_ID_HYPHEN, "emirates_id", "[REDACTED_EMIRATES_ID]"),
        (_EMIRATES_ID_DIGITS, "emirates_id", "[REDACTED_EMIRATES_ID]"),
        (_EMAIL, "email", "[REDACTED_EMAIL]"),
        (_PHONE, "phone", "[REDACTED_PHONE]"),
    ):
        for m in pattern.finditer(probe):
            if _claim(*m.span()):
                counts[label] = counts.get(label, 0) + 1
                spans.append((m.start(), m.end(), replacement))

    for m in _CARD_CANDIDATE.finditer(probe):
        # Luhn on the normalised digits; a candidate that fails is left alone,
        # and must not be counted.
        if _luhn_valid(m.group(0)) and _claim(*m.span()):
            counts["card"] = counts.get("card", 0) + 1
            spans.append((m.start(), m.end(), "[REDACTED_CARD]"))

    for start, end, replacement in sorted(spans, reverse=True):
        out = out[:start] + replacement + out[end:]
    return out, counts


def detect_pii(text: str) -> dict[str, int]:
    """Count PII occurrences by type WITHOUT rewriting the text.

    `scrub_text` answers "what may I send to the model". This answers "does
    this text contain PII at all", which is what an OUTPUT-side check needs —
    a tenant moderation hook asserting a model didn't reconstruct into its
    answer what the pre-call guard stripped from the prompt, or an audit
    assertion over a stored rationale.

    It delegates to `_default_scrub` and discards the rewritten text, so the
    two sides cannot disagree about what counts as an Emirates ID or a card.
    Re-deriving the patterns in the output check is exactly the divergence
    runtime/luhn.py was extracted to prevent (ReviewFindings-2026-07-18 B1):
    a hand-rolled `(?:\\d[ -]?){13,19}` with no Luhn call flags any long digit
    run — an invoice number, a registry filing reference — that the pre-call
    guard deliberately leaves alone.

    Ignores INPUT_GUARDRAIL mode: "is there PII in this text" is a question
    about the text, not about whether prompts are currently being scrubbed.
    """
    _, counts = _default_scrub(text or "")
    return counts


def scrub_messages(
    messages: list[dict[str, Any]],
    mode: Optional[str] = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """
    Deep-copy messages and scrub string `content` fields.
    Does not mutate the input list.
    """
    active = mode or resolve_mode()
    if active == "off":
        return copy.deepcopy(messages), {}

    total: dict[str, int] = {}
    out = copy.deepcopy(messages)
    for msg in out:
        content = msg.get("content")
        if isinstance(content, str):
            scrubbed, counts = scrub_text(content, mode=active)
            msg["content"] = scrubbed
            for k, v in counts.items():
                total[k] = total.get(k, 0) + v
        elif isinstance(content, list):
            # OpenAI-style multimodal content parts
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    scrubbed, counts = scrub_text(part["text"], mode=active)
                    part["text"] = scrubbed
                    for k, v in counts.items():
                        total[k] = total.get(k, 0) + v
    return out, total
