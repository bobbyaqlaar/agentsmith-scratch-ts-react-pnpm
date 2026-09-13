from __future__ import annotations

from typing import Any

from security.registry import ControlSpec
from security.report import ControlResult
from security.runners._shared import framework_root, security_fixture


def run(control: ControlSpec, ctx: dict[str, Any]) -> ControlResult:
    # framework_root inserts the repo ROOT only. The previous version also
    # inserted root/runtime, which would let `import input_guardrail` resolve
    # flat — nothing does that (runtime modules import each other as
    # `runtime.X`), and a bare runtime/ on sys.path can shadow same-named
    # top-level modules. Vestigial from before the package rename.
    framework_root(ctx)   # sys.path side effect; return value unused

    from runtime.input_guardrail import scrub_text

    cases, problem = security_fixture(control, ctx, "pii_probe_cases_base.json")
    if problem is not None:
        return problem
    failures: list[str] = []
    for case in cases:
        scrubbed, _counts = scrub_text(case["input"], mode="default")
        for needle in case.get("must_not_contain", []):
            if needle in scrubbed:
                failures.append(f"{case['id']}: still contains {needle!r}")
        for needle in case.get("must_contain", []):
            if needle not in scrubbed:
                failures.append(f"{case['id']}: missing {needle!r}")

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
        message=f"scrubbed {len(cases)} probe cases",
        evidence={"cases": str(len(cases))},
    )
