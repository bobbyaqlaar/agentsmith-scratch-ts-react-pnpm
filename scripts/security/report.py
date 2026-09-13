from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from security.registry import ControlSpec


@dataclass
class ControlResult:
    control_id: str
    status: Literal["pass", "fail", "skip", "warn"]
    message: str
    evidence: dict[str, str]


def _results_by_id(results: list[ControlResult]) -> dict[str, ControlResult]:
    return {r.control_id: r for r in results}


def _filter_controls(
    controls: list[ControlSpec],
    framework: str | None,
) -> list[ControlSpec]:
    if framework is None:
        return list(controls)
    out: list[ControlSpec] = []
    for c in controls:
        tags = c.frameworks
        if framework == "owasp" and tags.owasp:
            out.append(c)
        elif framework == "nist" and tags.nist:
            out.append(c)
        elif framework == "atlas" and tags.atlas:
            out.append(c)
        elif framework == "iso42001" and tags.iso42001:
            out.append(c)
    return out


def _status_for(control: ControlSpec, by_id: dict[str, ControlResult]) -> str:
    r = by_id.get(control.id)
    # "not run", never "skip". `skip` in this vocabulary means the control has
    # nothing to govern HERE — a deliberate not-applicable, which _resolve_exit
    # treats as green. A control with no result at all went UNEXAMINED, and
    # rendering the two identically is the conflation that let 14 of 23
    # controls report clean while nothing checked them. Unreachable today
    # (main appends one result per control), which is not a property a
    # refactor preserves.
    return r.status if r else "not run"


def _message_for(control: ControlSpec, by_id: dict[str, ControlResult]) -> str:
    r = by_id.get(control.id)
    return r.message if r else "no result"


def _render_table(
    title: str,
    controls: list[ControlSpec],
    by_id: dict[str, ControlResult],
    tag_column: str,
    tag_values: Callable[[ControlSpec], Sequence[object]],
) -> str:
    lines = [
        f"# {title}",
        "",
        f"| Control ID | {tag_column} | Registry | Result | Message |",
        "|---|---|---|---|---|",
    ]
    for c in controls:
        tags = ", ".join(str(x) for x in tag_values(c)) or "—"
        lines.append(
            f"| `{c.id}` | {tags} | {c.status} | {_status_for(c, by_id)} | "
            f"{_message_for(c, by_id)} |"
        )
    lines.append("")
    return "\n".join(lines)


def _render_security_report_md(
    controls: list[ControlSpec],
    by_id: dict[str, ControlResult],
    mode: str | None = None,
) -> str:
    lines = [
        "# Security harness report",
        "",
        # Stated even when it is the full run, so its absence is never the
        # thing a reader has to notice.
        f"Mode: `{mode or 'unrecorded'}`"
        + (
            "  — **a smoke run examines three controls; the rest were not "
            "attempted, not passed.**"
            if mode == "smoke"
            else ""
        ),
        "",
        "| Control ID | Title | Registry | Owner | Result | Message |",
        "|---|---|---|---|---|---|",
    ]
    for c in controls:
        lines.append(
            f"| `{c.id}` | {c.title} | {c.status} | {c.owner} | "
            f"{_status_for(c, by_id)} | {_message_for(c, by_id)} |"
        )
    lines.append("")
    # Summary counts
    counts: dict[str, int] = defaultdict(int)
    for c in controls:
        counts[_status_for(c, by_id)] += 1
    lines.extend(
        [
            "## Summary",
            "",
            *(f"- **{k}**: {v}" for k, v in sorted(counts.items())),
            "",
        ]
    )
    return "\n".join(lines)


def write_evidence_pack(
    out_dir: Path,
    controls: list[ControlSpec],
    results: list[ControlResult],
    framework: str | None = None,
    mode: str | None = None,
) -> None:
    """Write the auditor-facing pack.

    `mode` is recorded because `smoke` narrows the registry to three controls
    before anything runs. Without it the pack is indistinguishable from a full
    run that happens to have three controls — every one green, and no line
    saying the other twenty were never attempted. An absent control reads as an
    absent risk.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    by_id = _results_by_id(results)
    scoped = _filter_controls(controls, framework)

    payload = {
        "mode": mode,
        "framework_filter": framework,
        "controls": [asdict(r) for r in results],
        "registry": [
            {
                "id": c.id,
                "title": c.title,
                "status": c.status,
                "owner": c.owner,
                "runner": c.runner,
                "frameworks": {
                    "owasp": c.frameworks.owasp,
                    "nist": c.frameworks.nist,
                    "atlas": c.frameworks.atlas,
                    "iso42001": c.frameworks.iso42001,
                },
            }
            for c in controls
        ],
    }
    (out_dir / "security_report.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )

    (out_dir / "security_report.md").write_text(
        _render_security_report_md(scoped, by_id, mode),
        encoding="utf-8",
    )

    owasp_controls = [c for c in controls if c.frameworks.owasp]
    if framework == "owasp":
        owasp_controls = scoped
    (out_dir / "owasp_llm_top10.md").write_text(
        _render_table(
            "OWASP LLM Top 10",
            owasp_controls,
            by_id,
            "OWASP",
            lambda c: c.frameworks.owasp,
        ),
        encoding="utf-8",
    )

    nist_controls = [c for c in controls if c.frameworks.nist]
    if framework == "nist":
        nist_controls = scoped
    (out_dir / "nist_ai_rmf.md").write_text(
        _render_table(
            "NIST AI RMF",
            nist_controls,
            by_id,
            "NIST",
            lambda c: c.frameworks.nist,
        ),
        encoding="utf-8",
    )

    atlas_controls = [c for c in controls if c.frameworks.atlas]
    if framework == "atlas":
        atlas_controls = scoped
    (out_dir / "mitre_atlas.md").write_text(
        _render_table(
            "MITRE ATLAS",
            atlas_controls,
            by_id,
            "ATLAS",
            lambda c: c.frameworks.atlas,
        ),
        encoding="utf-8",
    )

    iso_controls = [c for c in controls if c.frameworks.iso42001]
    if framework == "iso42001":
        iso_controls = scoped
    (out_dir / "iso_42001.md").write_text(
        _render_table(
            "ISO/IEC 42001",
            iso_controls,
            by_id,
            "ISO theme",
            lambda c: c.frameworks.iso42001,
        ),
        encoding="utf-8",
    )
