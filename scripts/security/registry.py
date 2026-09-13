from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional


CheckType = Literal["unit", "integration", "eval", "artifact", "static", "live"]
ControlStatus = Literal["met", "partial", "gap", "org-owned"]
Owner = Literal["framework", "tenant", "shared"]


@dataclass(frozen=True)
class FrameworkTags:
    owasp: list[str]
    nist: list[str]
    atlas: list[str]
    iso42001: list[int]


@dataclass(frozen=True)
class ControlSpec:
    id: str
    title: str
    status: ControlStatus
    owner: Owner
    frameworks: FrameworkTags
    runner: str
    check_type: CheckType
    mechanism: str
    # Tenant-declared test path, for controls whose evidence lives in the
    # tenant repo. Only the `tenant_suite` runner reads it.
    suite: Optional[str] = None


def load_control_registry(
    path: Path, tenant_path: Optional[Path] = None
) -> list[ControlSpec]:
    """Framework controls, plus any the tenant declares.

    Mirrors the `models.yaml` merge (framework ← tenant) that already exists in
    this codebase, and closes the same gap: a tenant with a domain-specific
    control had nowhere to declare it. KYC Sentinel's evidence-mandated rating
    floor — a sanctions hit forces human review regardless of what the model
    rated — is a real control with tests and documentation that the compliance
    surface could not see.

    A tenant may only ADD. Redefining a framework control id raises, because a
    registry a tenant can edit is one where a tenant can quietly downgrade
    `SEC-HITL-001` to `noop` and keep a green harness. Additive-only means the
    framework's floor cannot be lowered from the repo being graded.
    """
    raw = json.loads(path.read_text())
    if tenant_path is not None and tenant_path.exists():
        framework_ids = {row["id"] for row in raw}
        tenant_raw = json.loads(tenant_path.read_text())
        clashes = sorted({r["id"] for r in tenant_raw} & framework_ids)
        if clashes:
            raise ValueError(
                f"tenant control registry redefines framework control(s): "
                f"{clashes}. Tenant registries are additive — a framework "
                f"control cannot be weakened from the repo under review."
            )
        raw = raw + tenant_raw
    seen: set[str] = set()
    out: list[ControlSpec] = []
    for row in raw:
        cid = row["id"]
        if cid in seen:
            raise ValueError(f"duplicate control id: {cid}")
        seen.add(cid)
        fw = row.get("frameworks", {})
        out.append(
            ControlSpec(
                id=cid,
                title=row["title"],
                status=row["status"],
                owner=row["owner"],
                frameworks=FrameworkTags(
                    owasp=list(fw.get("owasp", [])),
                    nist=list(fw.get("nist", [])),
                    atlas=list(fw.get("atlas", [])),
                    iso42001=[int(x) for x in fw.get("iso42001", [])],
                ),
                runner=row["runner"],
                check_type=row["check_type"],
                mechanism=row["mechanism"],
                suite=row.get("suite"),
            )
        )
    return out
