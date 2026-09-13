from __future__ import annotations

import json
from typing import Any


def _strip_markdown_fences(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped

    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


async def propose_corrected_payload(
    gateway: Any, payload: Any, error: str, model_hint: str = "developer"
) -> Any:
    prompt = [
        {
            "role": "system",
            "content": (
                "You correct invalid JSON payloads for retry. Return ONLY the corrected JSON. "
                "No markdown, no prose, no comments."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "payload": payload,
                    "error": error,
                    "instruction": "Return ONLY corrected JSON for the payload.",
                },
                ensure_ascii=True,
            ),
        },
    ]
    completion = await gateway.complete(prompt=prompt, model_hint=model_hint)
    return json.loads(_strip_markdown_fences(completion.text))


async def run_self_correction_loop(
    *,
    activity_fn: Any,
    payload: Any,
    gateway: Any,
    max_self_correction_attempts: int = 1,
    model_hint: str = "developer",
) -> Any:
    current_payload = payload
    last_error = ""

    try:
        return await activity_fn(current_payload)
    except Exception as exc:
        last_error = str(exc)

    for _ in range(max_self_correction_attempts):
        # The CORRECTOR can fail too, and its most likely failure is the one
        # this loop exists to survive: `propose_corrected_payload` ends in
        # `json.loads`, so a model that answers in prose — the ordinary failure
        # of "return ONLY JSON" — raised JSONDecodeError straight out of here.
        # That skipped the exhaustion result below, which is the whole reason a
        # caller gets a structured "could not fix it" instead of an exception.
        #
        # A failed correction is a failed attempt, not a crash. The payload is
        # left as it was: whatever reaches the fallback should be the last real
        # payload, not the wreckage of a correction that did not parse.
        try:
            current_payload = await propose_corrected_payload(
                gateway, current_payload, last_error, model_hint=model_hint
            )
        except Exception as exc:
            last_error = f"self-correction failed to produce a usable payload: {exc}"
            continue
        try:
            return await activity_fn(current_payload)
        except Exception as exc:
            last_error = str(exc)

    return {
        "__self_correction_exhausted__": True,
        "payload": current_payload,
        "error": last_error,
    }
