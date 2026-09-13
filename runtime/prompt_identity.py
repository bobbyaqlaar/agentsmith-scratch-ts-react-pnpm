"""
runtime/prompt_identity.py — which prompt produced this answer.

THE PROBLEM. When answers degrade, the prompt is usually the cause and almost
never the thing you can check. This framework records the model, the cost, the
latency and the verdict, and nothing at all about what was actually sent — so
"it got worse last Tuesday" has no column to join against.

The obvious fix is prompt versioning, which needs a template engine this
framework does not have: `FIXES_AND_CLEANUP.md` records that prompts are inline
f-strings, four of them across KYC Sentinel's agents, and a template engine is
a real piece of work sitting behind its own trigger condition.

WHAT IS AVAILABLE TODAY. Hash the SYSTEM message.

A system prompt is the part that is stable across requests and edited by a
human — which is exactly the shape of a template, whether or not it is stored
as one. The user turn changes every call and hashing it would produce a unique
value per request, useful for nothing. So:

    prompt.system.sha256    changes when, and only when, someone edits the
                            system prompt. This is the join column.
    prompt.template.id      an optional name the caller supplies, for when a
                            hash is not a thing a human can talk about.

That gives "answers degraded when this hash changed" for the cost of a hash,
works with inline f-strings, and survives the eventual move to a real engine
unchanged — the engine would supply `template.id` and this would keep agreeing
with it.

WHAT IS NOT RECORDED, deliberately: the prompt text itself. It is the single
most likely place for PII to enter a span, `trace_redactor` runs later and
would have to scrub it back out, and the hash answers the question the text was
wanted for. A hash is also safe to keep beyond a retention window that the text
is not.
"""

from __future__ import annotations

import hashlib
from typing import Any, Optional

SYSTEM_ROLES = {"system", "developer"}


def content_text(content: Any) -> str:
    """Message content as text, flattening the multipart shape.

    Public because runtime/testing.py needs the same answer. It had its own
    inline version that joined `m.get("content", "")` directly, which raises
    TypeError the moment content is a list — the shape this docstring is about
    — so FakeGateway crashed on a prompt the real gateway accepts. Two
    functions for one question, and only one of them knew about half the
    inputs (review-levers: one-catalog).

    Anthropic and OpenAI both allow `content` to be a list of typed parts. A
    prompt that switches between the two shapes without changing a word must
    not change its hash, or the join column moves for a reason nobody edited.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return "" if content is None else str(content)


def system_prompt(messages: Any) -> Optional[str]:
    """The system/developer turns, concatenated, or None if there are none.

    Concatenated rather than first-only: a caller that splits its instructions
    across two system turns has one prompt, and hashing only the first would
    miss edits to the rest.
    """
    if not isinstance(messages, list):
        return None
    texts = [
        content_text(m.get("content"))
        for m in messages
        if isinstance(m, dict) and str(m.get("role", "")).lower() in SYSTEM_ROLES
    ]
    joined = "\n".join(t for t in texts if t)
    return joined or None


def system_fingerprint(messages: Any) -> Optional[str]:
    """`sha256` of the system prompt, or None when there is not one.

    None, not the hash of an empty string. "This call had no system prompt" and
    "this call had a system prompt that happens to be empty" are different
    facts, and the second has a hash a reader would try to look up.
    """
    text = system_prompt(messages)
    if text is None:
        return None
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def prompt_attributes(
    messages: Any, *, template_id: Optional[str] = None
) -> dict[str, Any]:
    """Span attributes identifying the prompt, without carrying it.

    Everything here is safe to export unredacted: a digest, a count, a caller's
    own label. Absent keys mean absent facts — nothing is defaulted, so a
    missing `prompt.system.sha256` reads as "no system prompt" rather than as
    a value that failed to compute.
    """
    attrs: dict[str, Any] = {}
    fingerprint = system_fingerprint(messages)
    if fingerprint:
        attrs["prompt.system.sha256"] = fingerprint
        attrs["prompt.system.chars"] = len(system_prompt(messages) or "")
    if template_id:
        attrs["prompt.template.id"] = template_id
    if isinstance(messages, list):
        attrs["prompt.message_count"] = len(messages)
    return attrs
