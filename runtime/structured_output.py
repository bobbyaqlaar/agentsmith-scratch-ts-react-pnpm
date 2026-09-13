"""
runtime/structured_output.py — extract + validate LLM JSON (SEC-OUTPUT-001).

Parse fenced or bare JSON from model text, then Pydantic model_validate_json.
"""

from __future__ import annotations

import json
import re
from typing import TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)

# The language tag is CAPTURED, not skipped. It used to be `(?:json|JSON)?`,
# which matched a fence opened with any other tag and then left that tag inside
# the captured body — "```python" yielded "python\nprint(1)".
_FENCED = re.compile(
    r"```([A-Za-z0-9_+.-]*)[ \t]*\r?\n?(.*?)```",
    re.DOTALL,
)

_JSON_TAGS = {"json", "json5", "jsonc"}


class StructuredOutputError(ValueError):
    """Raised when LLM text cannot be parsed into the target Pydantic model."""


def _is_json(text: str) -> bool:
    try:
        json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return False
    return True


def _candidates(text: str) -> list[tuple[str, str]]:
    """Every plausible JSON payload in `text`, best first.

    Ordered rather than first-match-wins, because both of the rules this
    replaces picked a block without ever checking whether it was JSON:

    - The FIRST fence won regardless of its tag. A model that shows its working
      in one fence and answers in the next — the ordinary shape for a reasoning
      model — had its working extracted and its answer ignored.
    - `{` was tried before `[`, and an array of objects contains braces, so the
      brace branch always won and returned the array with its brackets stripped:
      `[{"a":1},{"a":2}]` became `{"a":1},{"a":2}`, which parses as nothing. A
      bare JSON array was unreachable, while a fenced one worked — the same
      payload accepted or rejected on how the model chose to format it. The
      docstring claimed arrays were supported; no test covered one.
    """
    out: list[tuple[str, str]] = []
    tagged: list[tuple[str, str]] = []
    untagged: list[tuple[str, str]] = []
    other: list[tuple[str, str]] = []
    for index, match in enumerate(_FENCED.finditer(text), start=1):
        tag = match.group(1).strip().lower()
        body = match.group(2).strip()
        # An empty fence is never the payload. Skipping it also keeps it out of
        # the ORIGIN reported on failure, which would otherwise blame a blank
        # block for a parse error caused by the text after it.
        if not body:
            continue
        if tag in _JSON_TAGS:
            tagged.append((f"```{tag} fence #{index}", body))
        elif not tag:
            untagged.append((f"untagged fence #{index}", body))
        else:
            other.append((f"```{tag} fence #{index}", body))
    # An explicit ```json tag outranks an untagged fence even when the untagged
    # one is also valid JSON and comes first: a model that emits a draft in a
    # bare fence and the answer in a tagged one has said which is which.
    out.extend(tagged)
    out.extend(untagged)
    out.extend(other)

    # Bare: the outermost span for each opener, offered in the order the
    # openers actually appear. Parse-validation alone cannot separate them —
    # for `[{"answer":"a"}]` the brace span is ALSO valid JSON, and taking it
    # returns the element instead of the array. Whichever bracket comes first
    # is the one that opens the outer structure.
    spans: list[tuple[int, str, str]] = []
    for opener, closer in (("{", "}"), ("[", "]")):
        first = text.find(opener)
        last = text.rfind(closer)
        if first != -1 and last > first:
            spans.append((first, f"bare {opener}...{closer} span", text[first : last + 1]))
    out.extend(
        (label, body) for _, label, body in sorted(spans, key=lambda item: item[0])
    )

    out.append(("whole response", text))
    return out


def _extract_json_text(raw: str) -> tuple[str, str]:
    """The first candidate that is actually JSON, and where it came from.

    Falls back to the first candidate when none parse, so the error a caller
    sees points at the payload the model most likely meant rather than at the
    whole response.

    The ORIGIN is returned alongside, and reported on failure, because
    "invalid JSON: Expecting property name" on its own does not say WHICH of a
    model's blocks was tried — with several fences in a response, that is the
    only thing an operator needs and the one thing the message omitted. It
    names the block; it does not quote it. Model output belongs in a redacted
    span, not in an exception string that goes wherever exceptions go.
    """
    text = raw.strip()
    if not text:
        raise StructuredOutputError("empty LLM output")

    candidates = _candidates(text)
    for label, candidate in candidates:
        if _is_json(candidate):
            return label, candidate
    return candidates[0]


def parse_llm_json(raw: str, model: type[T]) -> T:
    """Extract JSON (fenced or bare) and validate against ``model``."""
    origin, candidate = _extract_json_text(raw)
    try:
        # Ensure it's valid JSON first for clearer errors.
        json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise StructuredOutputError(f"invalid JSON in {origin}: {exc}") from exc

    try:
        return model.model_validate_json(candidate)
    except ValidationError as exc:
        raise StructuredOutputError(
            f"schema validation failed for {origin}: {exc}"
        ) from exc
