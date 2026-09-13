from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator

from security.registry import ControlSpec
from security.report import ControlResult

_SCHEMA_PATH = (
    Path(__file__).resolve().parents[1] / "schemas" / "risk_register.schema.json"
)


def _resolve_path(ctx: dict[str, Any]) -> Path:
    override = ctx.get("risk_register_path")
    if override is not None:
        return Path(override)
    tenant_security = Path(ctx["tenant_security"])
    return tenant_security / "risk_register.yaml"


def _template_path(ctx: dict[str, Any]) -> Path:
    """The shipped placeholder register — validating IT is legitimate (that is
    the use_template_fallback path below); shipping it AS your register is not."""
    return (
        Path(ctx["root"]) / "fixtures" / "security" / "templates" / "risk_register.yaml"
    )


def run(control: ControlSpec, ctx: dict[str, Any]) -> ControlResult:
    path = _resolve_path(ctx)
    strict = bool(ctx.get("strict", False))

    if not path.exists():
        # Framework self-test may validate the shipped template when no tenant file.
        template = _template_path(ctx)
        if ctx.get("use_template_fallback") and template.exists():
            path = template
        else:
            status = "fail" if strict else "warn"
            return ControlResult(
                control_id=control.id,
                status=status,
                message=f"missing risk register: {path}",
                evidence={"path": str(path)},
            )

    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        return ControlResult(
            control_id=control.id,
            status="fail",
            message=f"invalid YAML: {exc}",
            evidence={"path": str(path)},
        )

    if not isinstance(data, dict):
        return ControlResult(
            control_id=control.id,
            status="fail",
            message="risk register root must be a mapping",
            evidence={"path": str(path)},
        )

    schema = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(data), key=lambda e: list(e.path))
    if errors:
        first = errors[0]
        loc = "/".join(str(p) for p in first.path) or "(root)"
        return ControlResult(
            control_id=control.id,
            status="fail",
            message=f"schema invalid at {loc}: {first.message}",
            evidence={"path": str(path), "error_count": str(len(errors))},
        )

    entries = data.get("entries") or []

    # Schema validity is not the same as having a risk register. The shipped
    # template is a placeholder by design ("Example residual risk — replace
    # with organisation-owned entry"), and it validates perfectly — so a repo
    # that seeded the pack via G5 and never filled it in passed --strict and
    # published an evidence pack whose SEC-RISK-001 row cited RISK-EXAMPLE-001.
    # That is the failure mode a compliance artifact exists to prevent. The
    # check is for the shipped sentinel id only, not a judgement about whether
    # real entries are any good — content truth stays a human's job.
    placeholders = [
        str(e.get("id"))
        for e in entries
        if isinstance(e, dict) and str(e.get("id", "")).startswith("RISK-EXAMPLE-")
    ]
    if placeholders and path != _template_path(ctx):
        if strict:
            return ControlResult(
                control_id=control.id,
                status="fail",
                message=(
                    f"risk register still contains un-edited template entries "
                    f"({', '.join(placeholders[:3])}) — replace them with "
                    f"organisation-owned risks"
                ),
                evidence={"path": str(path), "placeholders": str(len(placeholders))},
            )
        return ControlResult(
            control_id=control.id,
            status="warn",
            message=f"risk register is still the shipped template ({len(placeholders)} example entries)",
            evidence={"path": str(path), "placeholders": str(len(placeholders))},
        )

    return ControlResult(
        control_id=control.id,
        status="pass",
        message=f"risk register valid ({len(entries)} entries)",
        evidence={"path": str(path), "entries": str(len(entries))},
    )
