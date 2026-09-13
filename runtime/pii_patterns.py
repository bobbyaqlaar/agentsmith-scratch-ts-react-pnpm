"""
runtime/pii_patterns.py — the PII shapes both sides of the control agree on.

`input_guardrail.py` scrubs a prompt before the model sees it;
`trace_redactor.py` scrubs a span before the exporter sees it. The framework's
own docs call these symmetric controls, and they were not: the pre-call side
knew about Emirates IDs and phone numbers, and the pre-export side did not, so
an Emirates ID that never reached the model still reached Phoenix if a span
attribute carried one. The tenant whose entire subject is Emirates IDs declared
no extra patterns, because nothing told it it had to.

Same reasoning as `runtime/luhn.py`, extracted when the two sides disagreed
about what a card number is. One definition, two consumers.

UNICODE DIGITS ARE PART OF THE SHAPE, not an edge case. This is a framework with
a UAE market and an Emirates ID as a first-class PII type, and `٧٨٤` is how a
person writes 784 in Arabic. `\\d` in Python matches those code points, so the
card pattern caught them all along and `runtime/luhn.py` validates them — but
the Emirates ID pattern anchors on a literal ASCII `784`, so an Emirates ID in
Arabic-Indic numerals matched nothing and went to the model in the clear. The
asymmetry is what made it invisible: anyone testing "do we handle Arabic
numerals" with a card number would have concluded yes.
"""

from __future__ import annotations

import re
import sys
import unicodedata

# Every Unicode decimal digit mapped to its ASCII counterpart. Built rather
# than listed: Arabic-Indic, Extended Arabic-Indic, Devanagari, fullwidth and
# the rest are one table, and a hand-written list is a catalog that drifts.
_DIGIT_TRANSLATION = {
    cp: str(unicodedata.digit(chr(cp)))
    for cp in range(sys.maxunicode + 1)
    if unicodedata.category(chr(cp)) == "Nd"
}


def ascii_digits(text: str) -> str:
    """`text` with every Unicode decimal digit rewritten as ASCII.

    Character-for-character, so a match span in the result addresses exactly the
    same characters in the original — which is what lets a caller detect on the
    normalised copy and redact in the text the user actually wrote.
    """
    return text.translate(_DIGIT_TRANSLATION)


# Emirates ID: 784-XXXX-XXXXXXX-X (hyphenated) or 15 digits starting with 784.
EMIRATES_ID_HYPHEN = re.compile(r"\b784-\d{4}-\d{7}-\d\b")
EMIRATES_ID_DIGITS = re.compile(r"\b784\d{12}\b")
EMAIL = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
# UAE / intl phones: +971… or 00… or long digit runs with optional separators
PHONE = re.compile(
    r"(?<!\d)(?:\+|00)?(?:971[\s-]?)?(?:0?5\d|5\d)[\s-]?\d{3}[\s-]?\d{4}\b"
    r"|(?<!\d)\+\d{10,15}\b"
)
CARD_CANDIDATE = re.compile(r"(?:\d[ -]?){13,19}")
