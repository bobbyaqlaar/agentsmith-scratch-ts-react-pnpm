from __future__ import annotations

from typing import Any

from security.registry import ControlSpec
from security.report import ControlResult


def run(control: ControlSpec, ctx: dict[str, Any]) -> ControlResult:
    return ControlResult(
        control_id=control.id,
        status="pass",
        message="noop runner",
        evidence={},
    )
