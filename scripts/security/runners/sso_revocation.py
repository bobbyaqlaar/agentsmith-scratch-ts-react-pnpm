from __future__ import annotations

from typing import Any

from security.registry import ControlSpec
from security.report import ControlResult
from security.runners._shared import failed, framework_component_absent, framework_root, node_suite


def run(control: ControlSpec, ctx: dict[str, Any]) -> ControlResult:
    """SEC-SSO-001 — the portal fails closed when revocation cannot be checked.

    Two halves, and only the second is generic. The snippet assertions below are
    specific to revocation semantics: they check the mode switch and its
    fail-closed branch are actually present, so the test file cannot pass by
    exercising an implementation that has been gutted. Running the portal test
    is the part three controls now share, and lives in `node_suite`.
    """
    absent = framework_component_absent(control, ctx, "portal")
    if absent is not None:
        return absent
    portal = framework_root(ctx) / "portal"
    mode_file = portal / "lib" / "ssoRevocationMode.ts"
    middleware = portal / "middleware.ts"

    missing = [p.name for p in (mode_file, middleware) if not p.exists()]
    if missing:
        return failed(control, f"missing portal SSO files: {', '.join(missing)}")

    text = mode_file.read_text(encoding="utf-8") + "\n" + middleware.read_text(encoding="utf-8")
    absent = [
        s for s in ("SSO_REVOCATION_MODE", "fail-closed", "fail-open", "503")
        if s not in text
    ]
    if absent:
        return failed(
            control, f"SSO revocation mode snippets missing: {', '.join(absent)}"
        )

    return node_suite(
        control, ctx, "test/ssoRevocation.test.ts", requires=("lib/ssoRevocationMode.ts",)
    )
