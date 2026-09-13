"""
runtime/luhn.py — single shared Luhn checksum validator.

Why (ReviewFindings-2026-07-18 B1): `input_guardrail.py` (pre-call PII
scrub) and `trace_redactor.py` (post-call span scrub) each carried their
own `_luhn_valid` with subtly different normalization — one stripped all
non-digits (`\\D`), the other only spaces/hyphens then required
`isdigit()`. On today's card-candidate regexes (digits, spaces, hyphens
only) they agree on every possible match, but the moment either
candidate pattern widens (e.g. dot-separated card formats), the pre-call
guard and post-call redactor would start disagreeing about what counts
as a card — a PII control inconsistency, not just style drift. One
implementation, imported by both, makes that divergence impossible.

Normalization: strip ALL non-digits (the more permissive of the two) —
a validator should not re-encode assumptions about which separators the
upstream candidate regex allows.
"""

from __future__ import annotations

import re

_NON_DIGITS = re.compile(r"\D")


def luhn_valid(candidate: str) -> bool:
    """True if `candidate` contains a 13–19 digit sequence that passes the
    Luhn checksum, after stripping any non-digit separators."""
    digits = _NON_DIGITS.sub("", candidate)
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    for i, ch in enumerate(reversed(digits)):
        n = int(ch)
        if i % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0
