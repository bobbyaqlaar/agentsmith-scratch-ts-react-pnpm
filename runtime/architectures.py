"""
runtime/architectures.py — the structural styles a tenant can start from, and
the agentic overlay (.agent-rfc/designs/tenant-architecture.md).

    catalogue()                         the validated templates/architectures.yaml
    resolve_style(name)                 a style id, from its id or a common name
    render_design_md(...)               docs/DESIGN.md for a new tenant
    render_architecture(...)            its architecture sections; `target=True` for `tenant adopt`
    render_scaffold_design(...)         the design for the commit that arms the gates
    render_scaffold_rfc(...)            the tenant's first RFC, as a template to fill in
    session_start_line(style, agentic)  one line every agent session starts with

The catalogue is the one home of what each style is; `tenant init` renders it
and nothing else reads it.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field


def _catalogue_path() -> Path:
    """The catalogue, found where the rest of `tenant init` finds framework
    files: $AGENTSMITH_DIR, then the machine install, then this checkout. An
    installed `agentsmith-runtime` package carries no templates/ of its own."""
    candidates = [
        Path(os.environ["AGENTSMITH_DIR"]) if os.environ.get("AGENTSMITH_DIR") else None,
        Path.home() / ".agent-framework",
        Path(__file__).resolve().parent.parent,
    ]
    for root in candidates:
        if root is not None and (root / "templates" / "architectures.yaml").is_file():
            return root / "templates" / "architectures.yaml"
    raise FileNotFoundError("templates/architectures.yaml not found in $AGENTSMITH_DIR, ~/.agent-framework "
                            "or this checkout — re-run install-ai-stack.sh")

# Where a stack's source lives. `<src>` in the catalogue is replaced by this.
SOURCE_ROOT = {"python-fastapi": "app/", "ts-react": "src/", "go": "internal/"}


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Layer(_Frozen):
    name: str = Field(min_length=1)
    path: str = Field(min_length=1)
    holds: str = Field(min_length=1)
    may_depend_on: list[str] = Field(default_factory=list)


class Style(_Frozen):
    name: str = Field(min_length=1)
    aliases: list[str] = Field(default_factory=list)
    summary: str = Field(min_length=1)
    fits: str = Field(min_length=1)
    layers: list[Layer] = Field(min_length=1)
    rules: list[str] = Field(min_length=1)
    tests: str = Field(min_length=1)
    watch: list[str] = Field(default_factory=list)


class Component(_Frozen):
    name: str = Field(min_length=1)
    path: str = Field(min_length=1)
    holds: str = Field(min_length=1)


class Agentic(_Frozen):
    name: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    components: list[Component] = Field(min_length=1)
    rules: list[str] = Field(min_length=1)
    placement: dict[str, str]


class Catalogue(_Frozen):
    styles: dict[str, Style] = Field(min_length=1)
    agentic: Agentic


@lru_cache(maxsize=1)
def catalogue() -> Catalogue:
    return Catalogue.model_validate(yaml.safe_load(_catalogue_path().read_text(encoding="utf-8")))


def resolve_style(name: str) -> str:
    """The style id for `name` — its id, or a common name for it (`clean`,
    `ports-and-adapters`, `n-tier`, …). Refuses anything else, listing the ids."""
    wanted = name.strip().lower()
    for style_id, style in catalogue().styles.items():
        if wanted == style_id or wanted in (a.lower() for a in style.aliases):
            return style_id
    raise ValueError(f"unknown architecture {name!r} — choose one of {', '.join(catalogue().styles)}")


def _paths(text: str, stack: str, source_root: Optional[str] = None) -> str:
    return text.replace("<src>/", source_root or SOURCE_ROOT.get(stack, "app/"))


def render_design_md(tenant_id: str, stack: str, style: Optional[str], agentic: bool) -> str:
    """`docs/DESIGN.md` for a new tenant: the living picture, starting from its structure."""
    lines = [f"# {tenant_id} — design", "", "The living picture of this system: its structure, and why.", ""]
    return "\n".join(lines) + render_architecture(stack, style, agentic)


def render_architecture(stack: str, style: Optional[str], agentic: bool, *, target: bool = False,
                        source_root: Optional[str] = None) -> str:
    """The architecture sections of `docs/DESIGN.md`. `target` renders them for a
    repository that already has code (`tenant adopt`): the style is the direction
    the code moves in, not a description of it. `source_root` replaces the
    stack's default (`app/`, …) when the repository keeps its code elsewhere."""
    lines = ["## Architecture (target)" if target else "## Architecture", ""]
    if target:
        lines += ["The structure this repository is moving towards. Existing code may not follow it yet;",
                  "each design that touches that code says how it moves.", ""]
    if style is None:
        if target:
            lines += [f"No structural style was chosen at adoption. Choose one of {', '.join(catalogue().styles)}",
                      "before the first design that restructures code, and record it here.", ""]
        else:
            lines += [
                "No structural style was chosen when this repository was scaffolded. Choose one before",
                "the first design that adds code — re-run `agentsmith tenant init <id> --architecture",
                f"<style> --force` with one of: {', '.join(catalogue().styles)}.",
                "",
            ]
    else:
        s = catalogue().styles[style]
        lines += [f"**Style: {s.name}.** {s.summary}", "", f"**Why this style.** {s.fits}", ""]
        lines += ["### Layers", "", "| Layer | Where | Holds | May depend on |", "|---|---|---|---|"]
        for layer in s.layers:
            depends = ", ".join(layer.may_depend_on) or "nothing"
            lines.append(f"| {layer.name} | `{_paths(layer.path, stack, source_root)}` | {layer.holds} | {depends} |")
        lines += ["", "Anything a layer does not list is a dependency pointing the wrong way.", ""]
        lines += ["### Rules", ""] + [f"- {rule}" for rule in s.rules] + [""]
        lines += ["### Tests", "", s.tests, ""]
        if s.watch:
            lines += ["### What to watch for", ""] + [f"- {item}" for item in s.watch] + [""]
    if agentic:
        a = catalogue().agentic
        lines += ["## The agent layer", "", a.summary, ""]
        if style is not None:
            lines += [f"**In this style:** {a.placement[style]}", ""]
        lines += ["| Component | Where | Holds |", "|---|---|---|"]
        for component in a.components:
            lines.append(f"| {component.name} | `{_paths(component.path, stack, source_root)}` | {component.holds} |")
        lines += ["", "### Rules", ""] + [f"- {rule}" for rule in a.rules] + [""]
    return "\n".join(lines)


# What the scaffold commit truthfully does for each pillar. A pillar the
# registry adds later, and any not listed here, gets the scaffold's honest
# default: it adds no code that pillar governs.
_SCAFFOLD_DEFAULT = "n/a — the scaffold adds no code this pillar governs; the first design that adds some answers it"


def _scaffold_answers(stack: str, style: Optional[str], agentic: bool, adopted: bool = False) -> dict[int, str]:
    workflow = ".github/workflows/agentsmith-gates.yml" if adopted else f".github/workflows/ci-{stack}.yml"
    agent = ("applies — `docs/DESIGN.md` › The agent layer fixes where agents, tools and workflows live, "
             "and their rules; no agent is written yet")
    return {
        1: ("applies — `docs/DESIGN.md` records the structural style before any code"
            if style else "applies — `docs/DESIGN.md` records that no style is chosen yet, and how to choose one"),
        2: f"n/a — the {'adoption' if adopted else 'scaffold'} adds workflows and configuration, no package",
        4: (f"applies — `docs/DESIGN.md` says where tests go; `{workflow}` runs them" if not adopted else
            "applies — the repository's own CI keeps running its tests; `docs/DESIGN.md` says where tests go"),
        7: "applies — `AGENTS.md` and `CLAUDE.md` carry this stack's rules",
        9: agent if agentic else "n/a — not an agentic application",
        10: ("applies — `docs/DESIGN.md` › The agent layer requires every model call to go through the gateway"
             if agentic else "n/a — not an agentic application; no model calls"),
        11: ("applies — `docs/DESIGN.md` › The agent layer: retrieved content and tool output are data, never "
             "instructions" if agentic else _SCAFFOLD_DEFAULT),
        12: f"applies — nothing generated holds a credential; `{workflow}` reads secrets by name",
        13: ("applies — the gates are armed (`.githooks/`); this commit's review exemption is verified against "
              "`.agenticframework/scaffold.json`" + ("; the hooks the repository ran before still run, after the "
                                                       "gate (`.githooks/chain`)" if adopted else "")),
        14: "applies — `.agent-rfc/fixtures/knowledge_graph.json` is generated in this commit",
        16: ("applies — `docs/DESIGN.md` › The agent layer: a failed step parks for a human, never guessed at"
             if agentic else _SCAFFOLD_DEFAULT),
    }


def render_scaffold_rfc(tenant_id: str, stack: str, intake: Optional[dict] = None) -> str:
    """The tenant's first RFC, at `.agent-rfc/NNN-*.md`.

    Two jobs. It is the spec `docs/UserManual.md` › Writing Agent Specifications
    asks for before an agent edits anything, in the `NNN-short-description.md`
    convention and with the three sections that page calls the minimum. And it
    satisfies `hooks/pre-commit` Guardrail 4, which looks for at least one `*.md`
    at `.agent-rfc/` DEPTH 1 under an enterprise org policy — the scaffold's own
    design sits at depth 2, so a scaffolded tenant could not make its first
    commit on such a machine (.agent-rfc/designs/scaffold-rfc-and-vouched-skip.md).

    A template, and it says so. Nothing here is a real requirement: the
    acceptance criteria are unchecked on purpose, so a reader can tell an
    unfilled RFC from a filled one at a glance.

    With `intake` — the `rfc` of a portal intake, already validated by
    runtime/intake.py — it is the author's RFC instead: the same three sections
    at the same path, filled, and without the TEMPLATE banner, because it is no
    longer one (.agent-rfc/designs/portal-intake-pull.md). The author's text is
    the Markdown body and nothing else: it reaches no YAML, workflow or shell.
    """
    if intake is not None:
        files = ("\n".join(f"- {f}" for f in intake["files_to_modify"])
                 or "- None named yet — add them before an agent works from this.")
        criteria = "\n".join(f"- [ ] {c}" for c in intake["acceptance_criteria"])
        return f"""# RFC 001 — {tenant_id}

## Objective

{intake["objective"]}

## Files to Modify

{files}

## Acceptance Criteria

{criteria}

<!-- Written by `agentsmith tenant init --from` from the author's portal intake.
     Stack: {stack}. The process gates need a design per change as well as this
     RFC: `agentsmith design new <slug> --scope <glob>` writes that skeleton.
     See AgentSmith docs/process-gates.md. -->
"""
    return f"""# RFC 001 — {tenant_id}

    ⚠️  TEMPLATE — written by `agentsmith tenant init`. Replace every section
        below before an agent works from it. Indented deliberately: a warning in
        column 0 fails a tenant build (.github/scratch-tenants/build.sh).

## Objective

What this change is for, in the words of whoever asked for it. One paragraph.

## Files to Modify

- `path/to/the/file` — what changes in it and why

## Acceptance Criteria

- [ ] The behaviour someone can check, not the code that produces it
- [ ] What must keep working that works today
- [ ] How a reviewer will know this is done

<!-- Stack: {stack}. The process gates need a design per change as well as this
     RFC: `agentsmith design new <slug> --scope <glob>` writes that skeleton.
     See AgentSmith docs/process-gates.md. -->
"""


def extend_design_scope(text: str, paths: Iterable[str]) -> str:
    """`text` with each of `paths` its front matter's `scope:` list lacks appended
    to that list; everything else as it was, and `text` itself when nothing is
    missing — so a second call changes nothing.

    For `agentsmith sync`, which keeps the arming design rather than rewriting it
    but commits files that design never listed: the hooks a tenant armed before
    they existed, and the manifest (.agent-rfc/designs/sync-adds-missing-hooks.md).
    A text with no front matter or no `scope:` list is returned unchanged: there
    is no list to extend, and inventing one would rewrite someone's design."""
    if not text.startswith("---\n"):
        return text
    end = text.find("\n---", 4)
    if end < 0:
        return text
    lines = text[4:end].split("\n")
    try:
        start = lines.index("scope:")
    except ValueError:
        return text
    # An item as the gate reads one (scripts/process_gate.py front_matter): any
    # indentation. Reading only `  - ` would miss a hand-indented list, append it
    # all again, and leave duplicates.
    item = re.compile(r"^(\s+-\s+)(.+?)\s*$")
    stop = start + 1
    while stop < len(lines) and item.match(lines[stop]):
        stop += 1
    found = [item.match(line) for line in lines[start + 1:stop]]
    listed = {m.group(2) for m in found if m}
    missing = [path for path in dict.fromkeys(paths) if path not in listed]
    if not missing:
        return text
    lead = found[-1].group(1) if found and found[-1] else "  - "
    lines[stop:stop] = [f"{lead}{path}" for path in missing]
    return "---\n" + "\n".join(lines) + text[end:]


def render_scaffold_design(tenant_id: str, stack: str, style: Optional[str], agentic: bool,
                           files: Iterable[str], pillars: Iterable[dict], adopted: bool = False) -> str:
    """The design for the commit that arms the gates: `.agent-rfc/designs/scaffold.md`
    for `tenant init`, `.agent-rfc/designs/adoption.md` for `tenant adopt`.

    Scoped to exactly the files the command wrote, and `done`: it covers that
    one commit and authorises no edit after it. `pillars` are the registry's —
    every one the registry asks a design to answer is answered.
    """
    answers = _scaffold_answers(stack, style, agentic, adopted)
    shape = catalogue().styles[style].name if style else "no structural style yet"
    kind = f"{shape}, agentic" if agentic else shape
    scope = "\n".join(f"  - {f}" for f in sorted(files))
    pillar_lines = [
        f"- P{p['id']} {answers.get(p['id'], _SCAFFOLD_DEFAULT)}"
        for p in pillars if "design" in (p.get("check") or [])
    ]
    if adopted:
        title = f"Adopt {tenant_id} ({kind})"
        about = ("Generated by `agentsmith tenant adopt`. It covers the adoption commit only; the next change to "
                 "gated code — existing code included — needs a design of its own.")
        problem = ("An existing repository comes under the gates. What it has — its code, hooks, CI, agent rules "
                   "and design document — stays; the gates are added around it.")
        approach = (f"`agentsmith tenant adopt {tenant_id}` gated the code it found, armed the hooks and chained the "
                    "ones the repository already ran, merged the gates into the IDE configs and the rules into "
                    "the rule files it already had, added `.github/workflows/agentsmith-gates.yml` beside its own "
                    f"CI, and recorded the target structure in `docs/DESIGN.md`: {kind}.")
        command, commit = "tenant adopt", "the commit that arms the gates"
        first, armed = "next", "this commit"
    else:
        title = f"Scaffold {tenant_id} ({kind})"
        about = ("Generated by `agentsmith tenant init`. It covers the scaffold commit only; the next change to\n"
                 "gated code needs a design of its own.")
        problem = "A new repository needs its gates, its CI and its structure decided before any code is written."
        approach = (f"`agentsmith tenant init {tenant_id} --stack {stack}` wrote the gate configuration and armed "
                    "hooks,\nthe IDE hook configs and rule files, the CI workflows, a knowledge graph, and "
                    f"`docs/DESIGN.md`\nrecording the structure: {kind}.")
        command, commit = "tenant init", "the first commit"
        first, armed = "first", "the first commit"
    return f"""---
status: done
scope:
{scope}
---
# {title}

{about}

## Problem

{problem}

## Approach

{approach} `.agenticframework/scaffold.json` holds a SHA-256 of every file `{command}` wrote, which is
what lets this commit's review be `n/a: generated scaffold` — the gate accepts that only on
{commit}, and only for files that still match.

## Pillars

{chr(10).join(pillar_lines)}

## Deviations

none

## Dependencies

None added by the {'adoption' if adopted else 'scaffold'}.

## Levers

- `design-before-code` — the structure is decided and written before the {first} line of code.
- `gate-integrity` — the gates are armed from {armed}, including for this one.
- `declared-vs-enforced` — the review exemption is checked against file hashes, not taken on trust.
"""


def session_start_line(style: Optional[str], agentic: bool) -> Optional[str]:
    """The line every agent session in this repository starts with, or None."""
    if style is None and not agentic:
        return None
    parts = []
    if style is not None:
        s = catalogue().styles[style]
        parts.append(f"Architecture: {s.name} — {s.rules[0]}")
    if agentic:
        parts.append("Agentic: every model call goes through the gateway, and tools are deny-by-default")
    return "; ".join(parts) + ". See docs/DESIGN.md › Architecture."
