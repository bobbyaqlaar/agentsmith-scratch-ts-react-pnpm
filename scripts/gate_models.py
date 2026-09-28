"""
scripts/gate_models.py — the governance registry and the records the process
gate reads beyond Problem / Approach / Levers (.agent-rfc/designs/
governance-enforcement.md, G1).

    Registry        templates/governance.json, compiled from
                    templates/agent-rules.yaml by generate-ide-config.py --registry
    ## Pillars       one answer per pillar the registry marks `design`
    ## Deviations    `none`, or entries that each resolve to an owner approval
    approvals.jsonl  written only by `agentsmith approve` (a terminal, never an agent)
    ## Sign-off      the docs/validation-checklist.md Step 4 block, complete

Pydantic V2 (pillar 7). Everything here validates on the receiving side and
returns named errors; process_gate.py decides what blocks.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

__all__ = [
    "Allowance",
    "Approval",
    "Artifact",
    "Artifacts",
    "Decision",
    "Deviation",
    "Extends",
    "GateEvent",
    "Ide",
    "Pillar",
    "PillarPolicy",
    "Records",
    "Registry",
    "Signoff",
    "ValidationError",
    "check_approvals",
    "check_evidence",
    "check_pillars",
    "check_signoff",
    "parse_approvals",
    "parse_deviations",
]

APPROVALS_FILE = ".agenticframework/approvals.jsonl"
CheckKind = Literal["design", "review", "mechanical"]
# The two ids the records cite each other by: a deviation in a design, and the
# approval the owner recorded at a terminal.
_DEVIATION_ID = r"(?:[A-Z]+-)?D\d+"
_APPROVAL_ID = r"A-[0-9a-f]{8}"


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Pillar(_Frozen):
    id: int = Field(ge=1)
    name: str = Field(min_length=1)
    check: list[CheckKind] = Field(min_length=1)
    design_question: str | None = None
    rule: str | None = None


class Allowance(_Frozen):
    """One file this repo is not held to one mechanical check for.

    `why` is required because an exemption nobody explained is one nobody can
    review, and `approval` is what lets a new one appear at all (gate_pillars
    .transition_problems).
    """

    check: str = Field(min_length=1)
    path: str = Field(min_length=1)
    why: str = Field(min_length=1)
    approval: str | None = Field(default=None, pattern=rf"^{_APPROVAL_ID}$")


class PillarPolicy(_Frozen):
    """`pillars` in .agenticframework/process-gates.json — whether this repo is
    held to the evidence rule and the mechanical checks, and what it is not
    held to yet.

    Unknown keys are refused (`allowed` is not `allow`, and a policy nobody
    reads is worse than none), which is why the `_about` every other block in
    that file carries has to be a field here.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    about: str | None = Field(default=None, alias="_about")
    mode: Literal["off", "report", "enforce"] = "off"
    allow: list[Allowance] = Field(default_factory=list)


class GateEvent(_Frozen):
    """What an IDE is asking the gate about, once the dialect is off it.

    Six IDEs send six payload shapes for the same three questions; the checks
    read this and nothing else, so a seventh IDE cannot change what a rule
    means (scripts/gate_ides.py).
    """

    kind: Literal["edit", "shell", "other"]
    paths: list[str] = Field(default_factory=list)
    command: str | None = None
    cwd: str | None = None
    stop_active: bool = False


class Decision(_Frozen):
    """The gate's answer, in the neutral profile of the gate contract
    (contract/gate/v1/). `gate_ides` renders the same four answers into each
    IDE's dialect; this is the shape a provider outside this repository writes.

    allow    the edit or the turn may proceed
    deny     it may not, and `text` says why
    block    the turn is refused (a stop gate's `deny`, which IDEs spell apart)
    context  not a refusal: `text` is what the session should start knowing
    """

    decision: Literal["allow", "deny", "block", "context"]
    text: str = ""


class Ide(_Frozen):
    """One IDE's hook wiring, as the registry declares it."""

    id: str = Field(min_length=1)
    config: str = Field(min_length=1)
    events: dict[str, str] = Field(default_factory=dict)
    fail_closed: bool = False
    note: str | None = None


class Signoff(_Frozen):
    groups: list[str] = Field(min_length=1)
    fields: list[str] = Field(min_length=1)


class Records(_Frozen):
    design_sections: list[str] = Field(min_length=1)
    signoff: Signoff
    # What every IDE rule file tells an agent at the start of a change; the
    # gate does not read it, the generator renders it into six files.
    design_start: list[str] = Field(default_factory=list)


class Extends(_Frozen):
    """A tenant's own additions — `extends` in .agenticframework/process-gates.json.
    Kept apart from the registry so re-syncing the framework never overwrites it."""

    pillars: list[Pillar] = Field(default_factory=list)
    # Only keys something reads live here: a declared key nothing consumes is a
    # rule a tenant believes is in force (`declared-vs-enforced`).
    #   session_start  extra lines in every agent's session-start context (process_gate.py)
    #   rules_extra    repo notes appended to every generated rule file (generate-ide-config.py)
    #   test_command   what those files name as this repo's test command
    #   artifacts      this repo's own document layout, over the framework's
    session_start: list[str] = Field(default_factory=list)
    rules_extra: list[str] = Field(default_factory=list)
    test_command: str | None = None
    artifacts: "Artifacts | None" = None


class Artifact(_Frozen):
    """One document type a repo is allowed exactly one of."""

    id: str = Field(min_length=1)
    path: str | None = Field(default=None)
    patterns: list[str] = Field(default_factory=list)
    required: bool = True
    # A record that only grows — the archive, the review log, the changelog.
    # Its numbered entries never move, so a pointer to one by number names it;
    # in any other document a bare number is a position (cross-reference rule).
    append_only: bool = False
    note: str | None = None


class Artifacts(_Frozen):
    types: list[Artifact] = Field(default_factory=list)
    reference: list[str] = Field(default_factory=list)
    ignored: list[str] = Field(default_factory=list)

    def merged(self, extra: "Artifacts | None") -> "Artifacts":
        """A repo's own `extends.artifacts` over the framework's defaults.

        A type it names again replaces that type — `path: null` says this repo
        has none — and anything else is added. Reference and ignored globs are
        additive: a repo knows its own documentation, and removing one of the
        framework's would quietly stop checking a file.
        """
        if extra is None:
            return self
        by_id = {a.id: a for a in self.types}
        for artifact in extra.types:
            by_id[artifact.id] = artifact
        return Artifacts(
            types=list(by_id.values()),
            reference=[*self.reference, *extra.reference],
            ignored=[*self.ignored, *extra.ignored],
        )


class Registry(_Frozen):
    model_config = ConfigDict(frozen=True, extra="ignore")  # "_about" and later sections

    version: str
    pillars: list[Pillar] = Field(min_length=1)
    records: Records
    artifacts: Artifacts = Field(default_factory=Artifacts)
    ides: list[Ide] = Field(default_factory=list)

    def pillar_ids(self) -> set[int]:
        return {p.id for p in self.pillars}

    def merged(self, extends: Extends | None) -> Registry:
        if extends is None:
            return self
        merged_artifacts = self.artifacts.merged(extends.artifacts)
        if not extends.pillars:
            return self.model_copy(update={"artifacts": merged_artifacts})
        clash = sorted(self.pillar_ids() & {p.id for p in extends.pillars})
        if clash:
            raise ValueError(
                "extends redefines " + ", ".join(f"P{i}" for i in clash)
                + " — a tenant may add pillars, not replace them"
            )
        return self.model_copy(update={"pillars": [*self.pillars, *extends.pillars],
                                        "artifacts": merged_artifacts})


# ── ## Deviations ────────────────────────────────────────────────────────────


class Deviation(_Frozen):
    id: str
    text: str
    approval_id: str | None


def parse_deviations(section: str) -> tuple[list[Deviation], list[str]]:
    """-> (active deviations, errors). Struck-through entries are withdrawn."""
    stripped = section.strip()
    if re.match(r"^(?:-\s*)?none\b", stripped, re.I):
        return [], []
    if not stripped:
        return [], ["'## Deviations' is empty — write `none`, or one entry per deviation"]
    found: list[Deviation] = []
    errors: list[str] = []
    for line in stripped.splitlines():
        if re.match(r"^-\s*~~", line):
            continue
        entry = re.match(rf"^-\s*({_DEVIATION_ID})\b(.*)$", line)
        if not entry:
            continue
        ident, rest = entry.group(1), entry.group(2)
        approval = re.search(rf"approval:\s*({_APPROVAL_ID})\b", rest)
        if any(d.id == ident for d in found):
            errors.append(f"deviation {ident} is listed twice")
            continue
        if not approval:
            errors.append(
                f"deviation {ident} has no owner approval — the owner runs `agentsmith approve` in a terminal "
                "and the entry ends `approval: A-xxxxxxxx`"
            )
        found.append(Deviation(id=ident, text=rest.strip(" —-"), approval_id=approval.group(1) if approval else None))
    if not found and not errors:
        errors.append("'## Deviations' lists no entry — write `none`, or `- D1 — rule — what and why — approval: A-…`")
    return found, errors


# ── Approvals ────────────────────────────────────────────────────────────────


class Approval(_Frozen):
    id: str = Field(pattern=rf"^{_APPROVAL_ID}$")
    design: str = Field(min_length=1)
    deviation: str = Field(pattern=rf"^{_DEVIATION_ID}$")
    approver: str = Field(min_length=1)
    approved_at: datetime
    channel: Literal["tty"]
    statement: str = Field(min_length=1)


def parse_approvals(text: str | None) -> tuple[list[Approval], list[str]]:
    approvals: list[Approval] = []
    errors: list[str] = []
    for number, line in enumerate((text or "").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            approvals.append(Approval.model_validate(json.loads(line)))
        except (json.JSONDecodeError, ValidationError) as exc:
            detail = exc.errors()[0]["msg"] if isinstance(exc, ValidationError) else exc.msg
            errors.append(f"{APPROVALS_FILE} line {number} is not an approval record: {detail}")
    return approvals, errors


def check_approvals(deviations: list[Deviation], approvals: list[Approval], design: str) -> list[str]:
    by_id = {a.id: a for a in approvals}
    errors = []
    for deviation in deviations:
        if deviation.approval_id is None:
            continue  # parse_deviations already named it
        approval = by_id.get(deviation.approval_id)
        if approval is None:
            errors.append(
                f"deviation {deviation.id} cites {deviation.approval_id}, which is not in {APPROVALS_FILE} — "
                f"the owner records it with `agentsmith approve {design} {deviation.id}`"
            )
        elif approval.design != design or approval.deviation != deviation.id:
            errors.append(
                f"deviation {deviation.id} cites {approval.id}, which approves {approval.deviation} of "
                f"{approval.design}, not {deviation.id} of {design}"
            )
    return errors


# ── ## Pillars ───────────────────────────────────────────────────────────────

_ANSWER = re.compile(
    r"^-\s*(?P<ids>P\d+(?:\s*,\s*P?\d+)*)\s+(?P<verdict>\*\*[^*]+\*\*|\S+(?:\s+—)?)(?P<rest>.*)$"
)
# Anything that was MEANT to answer a pillar, however it is punctuated, so a line
# `_ANSWER` cannot read is quoted rather than silently dropped.
_MEANT_AS_ANSWER = re.compile(r"^-\s*P\d+\b")


def pillar_kinds(section: str) -> dict[str, str]:
    """Each answered pillar's kind — applies, n/a, gap, deviation — read by the
    same pattern `check_pillars` judges. `unrecognised` for an answer that
    pattern matches but that is none of those, so it is shown, not dropped."""
    kinds: dict[str, str] = {}
    for line in section.splitlines():
        match = _ANSWER.match(line.strip())
        if not match:
            continue
        verdict = match.group("verdict").strip("* ").rstrip(" —").lower()
        kind = "deviation" if verdict.startswith("deviation") else (
            verdict if verdict in ("applies", "n/a", "gap") else "unrecognised")
        for number in re.findall(r"\d+", match.group("ids")):
            kinds[f"P{number}"] = kind
    return kinds


def _detail(rest: str) -> str:
    return re.sub(r"^\s*[—–-]+\s*", "", rest).strip()


def check_pillars(section: str, registry: Registry, deviations: list[Deviation]) -> list[str]:
    errors: list[str] = []
    answered: set[int] = set()
    known = {p.id: p for p in registry.pillars}
    deviation_ids = {d.id for d in deviations}
    for line in section.splitlines():
        match = _ANSWER.match(line.strip())
        if not match:
            continue
        ids = [int(i) for i in re.findall(r"\d+", match.group("ids"))]
        verdict = match.group("verdict").strip("* ").rstrip(" —")
        detail = _detail(match.group("rest"))
        label = match.group("ids")
        for pid in ids:
            if pid not in known:
                errors.append(f"'## Pillars' answers P{pid}, which the registry does not define")
            answered.add(pid)
        deviation = re.match(rf"^deviation\s+({_DEVIATION_ID})$", verdict)
        if deviation:
            if deviation.group(1) not in deviation_ids:
                errors.append(f"{label} cites deviation {deviation.group(1)}, which '## Deviations' "
                              "does not list as active")
        elif verdict in ("applies", "n/a"):
            if not detail:
                errors.append(f"{label} {verdict} — the answer says why (`{label} {verdict} — how or why`)")
        elif verdict == "gap":
            if not re.search(r"\b[A-Z][A-Z0-9]*-\d+\b", detail):
                errors.append(f"{label} gap — a declared gap names its backlog id (`{label} gap — PB-123`)")
        else:
            errors.append(f"{label} '{verdict}' — each answer is applies, n/a, gap or deviation")
    # A line that MEANT to answer a pillar and did not parse. Without this, an
    # answer with a colon where a space belongs — `- P3: applies — ...` — is
    # dropped by `_ANSWER` and the pillar reads as unanswered, so the author is
    # told they omitted something that is on the page. Same fault as `_PASS` in
    # scripts/process_gate.py (.agent-rfc/designs/sibling-sweep.md).
    for line in section.splitlines():
        stripped = line.strip()
        if _MEANT_AS_ANSWER.match(stripped) and not _ANSWER.match(stripped):
            errors.append(
                f"this line did not parse as a pillar answer — it must be "
                f"'- P<n> applies|n/a|gap|**deviation D<n>** — <why>': {stripped!r}"
            )
    for pillar in registry.pillars:
        if "design" in pillar.check and pillar.id not in answered:
            errors.append(f"'## Pillars' does not answer P{pillar.id} {pillar.name} — {pillar.design_question}")
    return errors


def check_evidence(section: str, resolve) -> list[str]:
    """Every `applies` answer names something that can be looked up.

    Sixteen lines of "it applies" is a form. An answer carries a token in
    backticks — a path, a test id, a span name — and `resolve` says whether
    anything of that name exists. It proves the token resolves, not that it is
    the right one: a name someone else can look up is falsifiable, prose is not.

    `n/a` is a reason there is nothing to name, `gap` already names a backlog
    id, and a deviation already resolves to an approval, so only `applies`
    carries evidence. One resolving token is enough: an answer may quote a flag
    or a word in backticks beside the evidence, and flagging those would push
    everyone to write the evidence without them.
    """
    errors: list[str] = []
    for line in section.splitlines():
        match = _ANSWER.match(line.strip())
        if not match:
            continue
        verdict = match.group("verdict").strip("* ").rstrip(" —")
        if verdict != "applies":
            continue
        label = match.group("ids")
        tokens = re.findall(r"`([^`]+)`", match.group("rest"))
        if not tokens:
            errors.append(
                f"{label} applies — the answer names nothing that can be looked up; name the path, test "
                "or span it applies to in backticks (`scripts/thing.py`, `test_it_retries`)"
            )
        elif not any(resolve(token) for token in tokens):
            errors.append(
                f"{label} applies — {', '.join(f'`{t}`' for t in tokens)} names nothing this commit "
                "tracks: no such path, and no source file holds that text"
            )
    return errors


# ── ## Sign-off ──────────────────────────────────────────────────────────────


def _signoff_section(text: str) -> str | None:
    match = re.search(r"^## Sign-off\b.*?$(.*?)(?=^## |\Z)", text, re.M | re.S)
    return match.group(1) if match else None


def check_signoff(text: str, registry: Registry) -> list[str]:
    spec = registry.records.signoff
    section = _signoff_section(text)
    if section is None:
        return ["records no '## Sign-off' block (docs/validation-checklist.md Step 4)"]
    errors = []
    for number, group in enumerate(spec.groups, start=1):
        line = re.search(rf"^\s*Group {number}\b.*$", section, re.M)
        if not line:
            errors.append(f"sign-off has no line for Group {number} · {group}")
            continue
        marked = re.findall(r"\[x\]\s*(checked|n/a|gap)\b\s*[:—–-]?\s*([^\[]*)", line.group(0), re.I)
        if len(marked) != 1:
            errors.append(f"sign-off Group {number} · {group} must mark exactly one of "
                          "checked / n/a / gap")
            continue
        verdict, reason = marked[0][0].lower(), marked[0][1].strip()
        if verdict != "checked" and (not reason or set(reason) <= set("_ ")):
            errors.append(f"sign-off Group {number} · {group} is {verdict} without saying why")
    for field in spec.fields:
        line = re.search(rf"^\s*{re.escape(field)}[^:\n]*:\s*(.*)$", section, re.M)
        if not line:
            errors.append(f"sign-off has no '{field}' line")
        elif not line.group(1).strip() or set(line.group(1).strip()) <= set("_ "):
            errors.append(f"sign-off '{field}' is blank")
    return errors
