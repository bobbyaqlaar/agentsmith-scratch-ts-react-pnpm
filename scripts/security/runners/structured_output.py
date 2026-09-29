from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from security.registry import ControlSpec
from security.report import ControlResult
from security.runners._shared import framework_root


class _SmokeModel(BaseModel):
    answer: str
    score: int


def run(control: ControlSpec, ctx: dict[str, Any]) -> ControlResult:
    framework_root(ctx)   # sys.path side effect; return value unused

    from runtime.structured_output import StructuredOutputError, parse_llm_json

    failures: list[str] = []

    try:
        parsed = parse_llm_json(
            'Here:\n```json\n{"answer":"ok","score":1}\n```',
            _SmokeModel,
        )
        if parsed.answer != "ok" or parsed.score != 1:
            failures.append("fenced parse mismatch")
    except Exception as exc:  # the harness aggregates; a raise loses every other control
        failures.append(f"fenced: {exc}")

    # Not a fail-open: the raise IS this check passing, so the exception is captured
    # and judged rather than handled. Written this way because an empty
    # `except StructuredOutputError: pass` reads as a swallowed error to every
    # reader and to scripts/check_bare_except.py, and the marker that silences it
    # says "fail-open", which this is not.
    raised: Exception | None = None
    try:
        parse_llm_json('{"answer":"ok"}', _SmokeModel)
    except Exception as exc:
        raised = exc
    if raised is None:
        failures.append("invalid schema did not raise")
    elif not isinstance(raised, StructuredOutputError):
        failures.append(f"invalid schema wrong error: {raised}")

    if failures:
        return ControlResult(
            control_id=control.id,
            status="fail",
            message="; ".join(failures[:5]),
            evidence={"failures": str(len(failures))},
        )
    return ControlResult(
        control_id=control.id,
        status="pass",
        message="structured_output smoke ok",
        evidence={},
    )
