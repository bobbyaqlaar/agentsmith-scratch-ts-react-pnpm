from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

from security.registry import ControlSpec
from security.report import ControlResult
from security.runners._shared import failed, framework_root, passed, tenant_security


# How a tenant registers a tool. Both forms appear in the shipped templates and
# in KYC Sentinel: the decorator, and the imperative call.
#   @tool(name="x", ...)          registry.register(fn, name="x", ...)
_REGISTRARS = {"tool", "register"}


def _registered_tool_names(root: Path) -> dict[str, str]:
    """Tool names the tenant actually registers → the file registering them.

    Read statically rather than by importing the tenant's modules: importing
    executes tenant code inside the harness, and a tenant whose imports need
    credentials or a database would fail this control for reasons that have
    nothing to do with its allowlist.
    """
    found: dict[str, str] = {}
    for rel in ("agents", "workflows", "tools"):
        base = root / rel
        if not base.is_dir():
            continue
        for py in base.rglob("*.py"):
            if "test" in py.parts or py.name.startswith("test_"):
                continue
            try:
                tree = ast.parse(py.read_text(encoding="utf-8", errors="ignore"))
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                label = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
                if label not in _REGISTRARS:
                    continue
                for kw in node.keywords:
                    if kw.arg == "name" and isinstance(kw.value, ast.Constant):
                        found.setdefault(str(kw.value.value), py.name)
    return found


def run(control: ControlSpec, ctx: dict[str, Any]) -> ControlResult:
    """SEC-TOOL-001 — the TENANT's allowlist governs the TENANT's tools.

    This used to load `fixtures/security/templates/tool_allowlist.yaml` — the
    framework's own template — and register two invented tools against it. That
    proves `ToolRegistry` denies an unlisted name, which is a framework unit
    test, not a control: it passed identically whether the tenant had an
    allowlist, had an empty one, or registered a dozen tools none of which
    appeared on it.

    Verified now:
      1. the tenant has its own allowlist, not the shipped template;
      2. every tool the tenant registers resolves the way that allowlist says —
         listed names invoke, unlisted names raise ToolNotAllowedError;
      3. at least one registered tool is NOT allowlisted, so enforcement is
         demonstrated rather than asserted. An allowlist naming everything is
         indistinguishable from having none, and the deny path is the half that
         can regress without anyone noticing.
    """
    framework_root(ctx)  # makes runtime.* importable
    allowlist_path = tenant_security(ctx) / "tool_allowlist.yaml"

    from runtime.tool_registry import ToolNotAllowedError, ToolRegistry, load_allowlist

    if not allowlist_path.exists():
        return failed(
            control,
            f"no tenant allowlist at {allowlist_path} — tools are ungoverned",
            expected=str(allowlist_path),
        )

    allowed = load_allowlist(allowlist_path)
    registered = _registered_tool_names(allowlist_path.parents[2])

    # An EMPTY allowlist is correct posture for a repo that registers no tools —
    # under strict mode it denies everything, which is what the framework's own
    # pack deliberately declares. It is only a defect when tools exist to govern.
    if not allowed:
        if not registered:
            return passed(
                control,
                "no tools registered and an empty allowlist — strict mode denies all",
                allowlist=str(allowlist_path),
            )
        return failed(
            control,
            f"allowlist is empty but {len(registered)} tool(s) are registered — "
            f"strict mode denies every one of them",
            allowlist=str(allowlist_path),
            registered=",".join(sorted(registered)[:5]),
        )

    if not registered:
        # Allowlist has entries but nothing registers tools here. Exercise both
        # paths against the tenant's OWN allowlist rather than a shipped template.
        registered = {sorted(allowed)[0]: "(allowlist)", "__unlisted_probe__": "(probe)"}

    registry = ToolRegistry(allowlist_path=allowlist_path, strict=True)
    failures: list[str] = []
    denied: list[str] = []

    for name in sorted(registered):
        registry.register(lambda: "ok", name=name, description="SEC-TOOL-001 probe")
        should_allow = name in allowed
        try:
            registry.invoke(name, {})
            if not should_allow:
                failures.append(f"{name}: not allowlisted but invoked")
        except ToolNotAllowedError:
            denied.append(name)
            if should_allow:
                failures.append(f"{name}: allowlisted but denied")
        except Exception as exc:  # anything else is a real failure
            failures.append(f"{name}: unexpected {type(exc).__name__}: {exc}")

    if failures:
        return failed(
            control,
            "; ".join(failures[:5]),
            allowlist=str(allowlist_path),
            registered=str(len(registered)),
        )
    if not denied:
        return failed(
            control,
            f"all {len(registered)} registered tool(s) are allowlisted — the deny "
            f"path is never exercised, so enforcement is asserted, not shown",
            allowlist=str(allowlist_path),
        )
    return passed(
        control,
        f"{len(registered)} registered tool(s) governed by the tenant allowlist; "
        f"{len(denied)} denied under strict",
        allowlist=str(allowlist_path),
        registered=str(len(registered)),
        denied=",".join(sorted(denied)[:3]),
    )
