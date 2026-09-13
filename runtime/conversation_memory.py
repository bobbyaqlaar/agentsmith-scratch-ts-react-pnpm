"""
runtime/conversation_memory.py — short-term token-window message buffer.

Keeps a chat-style message list under a configurable token budget.
Eviction: drop oldest messages first until under budget (summarization is v2).

Usage:
    mem = ConversationMemory(token_budget=4000)
    mem.add("user", "Hello")
    messages = mem.as_messages()  # for llm_gateway.complete(messages)
"""

from __future__ import annotations

import functools
import logging
from typing import Any

logger = logging.getLogger(__name__)

# Roles whose messages are never evicted. The system message carries the
# persona, the tool policy and the refusal instructions; truncate-oldest
# deleted it FIRST, because it is first, and the conversation then continued
# with no instruction at all. Nothing failed loudly — the model simply stopped
# being told what it was not allowed to do.
PROTECTED_ROLES = frozenset({"system"})

_fallback_warned = False


@functools.lru_cache(maxsize=8)
def _encoder(encoding_name: str) -> Any:
    """tiktoken's encoder, built once per encoding.

    `get_encoding` ran on every single count, and `_evict` counts the whole
    buffer on every iteration of its loop.
    """
    import tiktoken

    return tiktoken.get_encoding(encoding_name)


def _estimate_tokens(text: str) -> int:
    """Token estimate for when tiktoken is unavailable.

    The rule this replaces was `len(text) // 4`, whose error is not symmetric
    across scripts. Measured against cl100k_base:

        English   12 real vs 18 estimated   +50%
        Arabic    49 real vs 17 estimated   -65%

    Over-counting evicts a message sooner than strictly necessary. UNDER-counting
    by two thirds means a buffer built to sit inside a context window holds
    roughly three times what it believes, and the provider rejects the request —
    in the script this framework's own tenants write in, which is the same
    reason runtime/pii_patterns.py normalises Arabic-Indic digits.

    Three character classes, because one divisor cannot cover them: ASCII words
    run about four characters to the token, punctuation and whitespace tokenize
    much more finely (source code and JSON counted LOW under a flat
    characters/4), and a non-ASCII character is worth close to a whole token.

    WHAT IT IS NOT: exact, or uniformly conservative. Measured across ten
    samples — prose, Arabic, mixed Arabic/digits, Chinese, Hindi, source code,
    JSON, emoji, URLs — this lands between 0.95x and 1.90x of the true count.
    Dense ASCII punctuation (a bare URL) can still read a few percent low. What
    is gone is the systematic two-thirds under-count on non-Latin text; what
    remains is a heuristic, and a character heuristic cannot be made both tight
    on English prose and provably safe on every input. Install the `tokenizer`
    extra for exact counts — this path logs once to say it is estimating.
    """
    if not text:
        return 1
    non_ascii = sum(1 for ch in text if ord(ch) > 127)
    alnum = sum(1 for ch in text if ch.isascii() and ch.isalnum())
    symbols = len(text) - non_ascii - alnum
    return max(1, int(alnum / 4 + symbols * 0.6 + non_ascii * 1.2))


def _count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    global _fallback_warned
    try:
        return len(_encoder(encoding_name).encode(text))
    except Exception as exc:
        # Said once, not per call — a token counter runs in a loop, and a
        # per-call warning would be noise that gets filtered. Said at all
        # because tiktoken is NOT a dependency of the shipped runtime package:
        # for a tenant who installed agentsmith-runtime, this estimate is not a
        # fallback, it is the only path, and nothing announced that.
        if not _fallback_warned:
            _fallback_warned = True
            logger.warning(
                "tiktoken unavailable (%s: %s) — conversation memory is using a "
                "byte/character ESTIMATE for its token budget. Install the "
                "`tokenizer` extra for exact counts.",
                type(exc).__name__,
                exc,
            )
        return _estimate_tokens(text)


class ConversationMemory:
    """In-process short-term memory with truncate-oldest eviction."""

    def __init__(
        self,
        token_budget: int = 4000,
        encoding_name: str = "cl100k_base",
    ) -> None:
        if token_budget < 1:
            raise ValueError("token_budget must be >= 1")
        self.token_budget = token_budget
        self.encoding_name = encoding_name
        self._messages: list[dict[str, Any]] = []

    @property
    def messages(self) -> list[dict[str, Any]]:
        return list(self._messages)

    def add(self, role: str, content: str, **extra: Any) -> None:
        msg: dict[str, Any] = {"role": role, "content": content}
        msg.update(extra)
        self._messages.append(msg)
        self._evict()

    def as_messages(self) -> list[dict[str, Any]]:
        return [{"role": m["role"], "content": m["content"]} for m in self._messages]

    def token_count(self) -> int:
        return sum(
            _count_tokens(str(m.get("content", "")), self.encoding_name)
            for m in self._messages
        )

    def clear(self) -> None:
        self._messages.clear()

    def _evict(self) -> None:
        """Drop the oldest EVICTABLE message until the buffer fits.

        Oldest-first is still the rule; the change is that a protected role is
        not a candidate. The system message is both the first message and the
        one carrying the instructions, so plain truncate-oldest removed the
        agent's persona, its tool policy and its refusals before it touched a
        single turn of conversation — silently, since a shorter list is exactly
        what eviction is supposed to produce.

        Counts are taken ONCE per pass rather than recomputed inside the loop.
        `token_count()` re-encodes the whole buffer, so the old loop was
        quadratic in the number of messages evicted.
        """
        counts = [self._message_tokens(m) for m in self._messages]
        total = sum(counts)

        while total > self.token_budget:
            evictable = [
                i
                for i, m in enumerate(self._messages)
                if m.get("role") not in PROTECTED_ROLES
            ]
            # Keep the most recent evictable message even when it alone busts
            # the budget: there is nothing smaller to fall back to without
            # summarization (v2), and returning an empty conversation would be
            # worse than returning an oversized one.
            if len(evictable) <= 1:
                break
            index = evictable[0]
            total -= counts.pop(index)
            self._messages.pop(index)

        # A buffer that is still over budget here is one whose PROTECTED
        # messages do not fit. Say so: the caller set a budget that cannot hold
        # its own system prompt, and silently serving an oversized prompt is
        # how that reaches them as a provider error instead.
        if total > self.token_budget:
            logger.warning(
                "conversation memory is %d tokens over its %d-token budget and "
                "cannot shrink further — %d protected message(s) plus the most "
                "recent turn exceed it on their own.",
                total - self.token_budget,
                self.token_budget,
                sum(
                    1
                    for m in self._messages
                    if m.get("role") in PROTECTED_ROLES
                ),
            )

    def _message_tokens(self, message: dict[str, Any]) -> int:
        return _count_tokens(str(message.get("content", "")), self.encoding_name)
