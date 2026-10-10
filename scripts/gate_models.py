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
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

__all__ = [
    "RECORD_LIMITS",
    "AgencyManifest",
    "Allowance",
    "Approval",
    "Artifact",
    "Artifacts",
    "ControlRow",
    "Decision",
    "DecisionV2",
    "DevRecord",
    "Deviation",
    "EvalsPort",
    "EvalsRequest",
    "EvalsThresholds",
    "Extends",
    "GateEvent",
    "GateEventV2",
    "GateEventV3",
    "Ide",
    "KgEdge",
    "KgImpact",
    "KgNode",
    "KnowledgeGraph",
    "Pillar",
    "PillarPolicy",
    "RecordCommit",
    "Records",
    "RedactionRequest",
    "RedactionResult",
    "Registry",
    "RiskRegister",
    "RulesCheck",
    "RulesFile",
    "RulesFileState",
    "RulesPort",
    "RulesRender",
    "RulesRequest",
    "Scorecard",
    "SecurityPort",
    "SecurityRequest",
    "SecurityResult",
    "Signoff",
    "TenantControl",
    "ToolAllowlist",
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


class GateEventV2(GateEvent):
    """A gate event in contract 2 (contract/gate/v2/): v1's, plus `range` —
    the commits from `base` to `head`, which the `ci` event asks about. A
    `base` of all zeros is a new branch: the head commit alone. v1's model is
    left as it was, because contract/gate/v1/ is published and still served."""

    kind: Literal["edit", "shell", "other", "range"]
    base: str | None = None
    head: str | None = None


class DecisionV2(Decision):
    """A contract-2 answer. The four decisions are v1's; `report` (markdown)
    and `annotations` (one problem a line) are for the person reading CI, and a
    provider with nothing more than the verdict to say leaves both empty."""

    report: str = ""
    annotations: list[str] = Field(default_factory=list)


class GateEventV3(GateEventV2):
    """A gate event in contract 3 (contract/gate/v3/): v2's, plus `commit` —
    may this commit be made, judged on the staged change and `message` — and
    `push` — may this history leave the machine. v2's model is left as it was."""

    kind: Literal["edit", "shell", "other", "range", "commit", "push"]
    message: str | None = None
    amend: bool = False


class KgNode(BaseModel):
    """One node of a repository's knowledge graph: a file or a guardrail."""

    model_config = ConfigDict(extra="allow")

    id: str = Field(min_length=1)
    node_type: str | None = None


class KgEdge(BaseModel):
    """One edge: `source` IMPORTS `target`, and the like."""

    model_config = ConfigDict(extra="allow")

    source: str
    target: str
    edge_type: str


class KnowledgeGraph(BaseModel):
    """`.agent-rfc/fixtures/knowledge_graph.json`, as a provider reads it to
    scope a review (contract/gate/v3/knowledge_graph.schema.json). The fields
    the review-scope hash depends on are pinned; a builder may add more."""

    model_config = ConfigDict(extra="allow")

    nodes: list[KgNode] = Field(default_factory=list)
    edges: list[KgEdge] = Field(default_factory=list)
    links: list[KgEdge] = Field(default_factory=list)


class KgImpact(_Frozen):
    """What `<gate> kg impact` answers: the files a review must read (the change
    plus one hop of dependents), the lever groups, the changed files the graph
    does not know yet, and the scope's name — the review's `KG query:` line."""

    files: list[str]
    groups: list[int]
    unknown: list[str] = Field(default_factory=list)
    query: str = Field(pattern=r"^kg:[0-9a-f]{12}$")


# ── The record a gate provider sends a portal (contract/record/v1/) ─────────
#
# One request body. The limits are the receiver's, published so a sender can
# keep inside them; `record.schema.json` is generated from these models, and the
# portal's validator (portal/lib/devIngest.ts) is held to that file by its tests
# (.agent-rfc/designs/record-contract.md). Unknown keys are allowed: a receiver
# drops what it does not read, so a sender adding one breaks nobody.

RECORD_LIMITS = {"commits": 500, "subject": 1000, "text": 4000, "list": 200, "designs": 1000,
                 "body_bytes": 2_000_000}
RECORD_VERDICTS = ("passed", "failed", "passed_with_notes", "not_gated", "before_adoption")
RECORD_PILLAR_KINDS = ("applies", "n/a", "gap", "deviation", "unrecognised")

_Sha = Annotated[str, Field(pattern=r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")]
_Text = Annotated[str, Field(max_length=RECORD_LIMITS["text"])]
_Short = Annotated[str, Field(max_length=32)]
_Verdict = Literal["passed", "failed", "passed_with_notes", "not_gated", "before_adoption"]
_PillarKind = Literal["applies", "n/a", "gap", "deviation", "unrecognised"]
_PillarId = Annotated[str, Field(pattern=r"^P\d{1,3}$")]


class _Record(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)


class RecordDeviation(_Record):
    id: _Short
    text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    approval_id: _Short | None = None


class _DesignFields(_Record):
    title: _Text | None = None
    status: _Short | None = None
    scope: list[_Text] = Field(default_factory=list, max_length=RECORD_LIMITS["list"])
    # `propertyNames` as well as the key pattern: a receiver refuses a key that is
    # not a pillar, so the published schema must too, or a sender it passes is refused.
    pillars: dict[_PillarId, _PillarKind] = Field(
        default_factory=dict, max_length=RECORD_LIMITS["list"],
        json_schema_extra={"propertyNames": {"pattern": r"^P\d{1,3}$"}})
    deviations: list[RecordDeviation] = Field(default_factory=list, max_length=RECORD_LIMITS["list"])


class RecordDesign(_DesignFields):
    """The design a commit names: resolved, or the reason it could not be."""

    ref: _Text
    path: _Text | None = None
    resolved: bool
    reason: _Text | None = None


class RecordHeadDesign(_DesignFields):
    """A design document as it stands at the head."""

    path: _Text


class RecordPass(_Record):
    n: int = Field(ge=0)
    findings: int = Field(ge=0)


class RecordReview(_Record):
    ref: _Text
    path: _Text | None = None
    resolved: bool
    reason: _Text | None = None
    passes: list[RecordPass] = Field(default_factory=list, max_length=RECORD_LIMITS["list"])
    signed_off: bool | None = None
    kg_query: Annotated[str, Field(max_length=64)] | None = None


class RecordCommit(_Record):
    commit: _Sha
    parent: _Sha | None = None
    subject: Annotated[str, Field(max_length=RECORD_LIMITS["subject"])]
    author_name: Annotated[str, Field(max_length=320)] | None = None
    author_email: Annotated[str, Field(max_length=320)] | None = None
    committed_at: Annotated[str, Field(max_length=64)] | None = None
    adopted: bool
    gated: bool
    verdict: _Verdict
    errors: list[_Text] = Field(default_factory=list, max_length=RECORD_LIMITS["list"])
    notes: list[_Text] = Field(default_factory=list, max_length=RECORD_LIMITS["list"])
    repairs: list[_Sha] = Field(default_factory=list, max_length=RECORD_LIMITS["list"])
    design: RecordDesign | None = None
    review: RecordReview | None = None


class DevRecord(_Record):
    """One request body: what a gate provider decided about each commit of a
    range, and the designs at its head (on the last part of a long range)."""

    schema_version: Literal[1] = Field(alias="schema")
    head: _Sha
    commits: list[RecordCommit] = Field(max_length=RECORD_LIMITS["commits"])
    designs: list[RecordHeadDesign] | None = Field(default=None, max_length=RECORD_LIMITS["designs"])
    ci_run_url: Annotated[str, Field(pattern=r"^https?://")] | None = None


# ── The rules a provider renders for a tenant's agents (contract/rules/v1/) ─
#
# `render` names each file, its text, how it is placed and what it is; `check`
# says whether the repository still holds what `render` would place. The path
# rule is the contract's own: a rules provider writes what agents read, never a
# repository's hooks, declarations or CI — and the CALLER enforces it, on every
# path, before writing any (.agent-rfc/designs/rules-contract.md).

RULES_PLACEMENTS = ("whole", "block")
RULES_KINDS = ("instructions", "supporting")
RULES_STATES = ("current", "drifted", "absent")
RULES_FORBIDDEN = (".git", ".githooks", ".agenticframework", ".github/workflows", ".github/actions")
RULES_LIMITS = {"files": 200, "text": 1_000_000}


def _any_case(text: str) -> str:
    """`text` as a pattern blind to letter case, in a form JSON Schema readers
    share — a case-insensitive filesystem opens `.GIT/hooks` as `.git/hooks`."""
    return "".join(f"[{c.lower()}{c.upper()}]" if c.isalpha() else re.escape(c) for c in text)


# Relative, normalised (no empty, `.` or `..` segment, no backslash or control
# character), and
# not inside a forbidden directory, whatever its case.
RULES_PATH = (
    r"^(?!/)(?!.*//)(?!.*/$)(?!(?:.*/)?\.\.?(?:/|$))"
    + "(?!(?:" + "|".join(_any_case(d) for d in RULES_FORBIDDEN) + r")(?:/|$))"
    + r"[^\\\x00-\x1f]+$"
)


class RulesRequest(BaseModel):
    """What a caller sends on stdin. Unknown keys are ignored, so a caller that
    adds one breaks no provider."""

    model_config = ConfigDict(extra="allow")

    cwd: str | None = None


class RulesFile(_Frozen):
    model_config = ConfigDict(frozen=True, extra="forbid", regex_engine="python-re")

    path: str = Field(pattern=RULES_PATH, max_length=512)
    text: str = Field(max_length=RULES_LIMITS["text"])
    placement: Literal["whole", "block"]
    kind: Literal["instructions", "supporting"]


class RulesRender(_Frozen):
    files: list[RulesFile] = Field(max_length=RULES_LIMITS["files"])

    @model_validator(mode="after")
    def _one_entry_per_path(self) -> "RulesRender":
        seen: set[str] = set()
        for entry in self.files:
            if entry.path.lower() in seen:
                raise ValueError(f"{entry.path} is rendered twice")
            seen.add(entry.path.lower())
        return self


class RulesFileState(_Frozen):
    path: str
    state: Literal["current", "drifted", "absent"]


class RulesCheck(_Frozen):
    decision: Literal["allow", "deny"]
    text: str = ""
    report: str = ""
    files: list[RulesFileState] = Field(default_factory=list)


class RulesPort(_Frozen):
    """`providers.rules` in .agenticframework/providers.json."""

    command: str = Field(min_length=1)
    version: str | None = None
    contract: Literal[1] | None = None
    setup: str | None = Field(default=None, min_length=1)


# ── The evals a provider judges for a tenant (contract/evals/v1/) ──────────
#
# The tenant owns its datasets and its application's outputs; the provider
# judges them and answers with a scorecard. Cases carry what the suite reads and
# may carry more (`extra="allow"`): a tenant's own annotations break nothing
# (.agent-rfc/designs/evals-contract.md).

EVAL_SUITES = ("golden", "fairness", "hallucination", "adversarial", "rag_poison")
# Suites a model judges: each case must carry the output the tenant's app produced.
JUDGED_SUITES = ("golden", "fairness", "hallucination")
EVAL_VERDICTS = ("pass", "fail", "no_verdict", "not_gradable")


class _Case(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str = Field(min_length=1)


class GoldenCase(_Case):
    input: str = Field(min_length=1)
    actual_output: str = Field(min_length=1)
    reference_output: str | None = None
    expected_tool: str | None = None


class FairnessCase(GoldenCase):
    pair_id: str = Field(min_length=1)
    protected_attribute: str = Field(min_length=1)
    attribute_value: str = Field(min_length=1)


class ContextDocument(BaseModel):
    """One retrieved document, as a retrieval layer produces it: the judge
    quotes its `text` under its `id` (or `title`)."""

    model_config = ConfigDict(extra="allow")

    text: str = Field(min_length=1)
    id: str | None = None
    title: str | None = None


class HallucinationCase(GoldenCase):
    # The three shapes the judge renders (eval_judge._as_context).
    retrieved_context: str | list[str | ContextDocument] | None = None
    expect_hallucination: bool = False
    score_hallucination: bool = True


class AdversarialCase(_Case):
    input: str = Field(min_length=1)
    expect: Literal["block", "flag", "safe"]


class RagPoisonCase(_Case):
    query: str | None = None  # reported, not scored
    document: str = Field(min_length=1)
    # Its scorer reads anything but `quarantine` as `safe`: a typo must not.
    expect: Literal["quarantine", "safe"]
    pair_id: str | None = None


EVAL_CASES = {"golden": GoldenCase, "fairness": FairnessCase, "hallucination": HallucinationCase,
              "adversarial": AdversarialCase, "rag_poison": RagPoisonCase}


class EvalsRequest(BaseModel):
    """What a caller sends on stdin. Unknown keys are ignored."""

    model_config = ConfigDict(extra="allow")

    suite: Literal["golden", "fairness", "hallucination", "adversarial", "rag_poison"]
    fail_below: float | None = Field(default=None, ge=0.0, le=1.0)
    fail_above: float | None = Field(default=None, ge=0.0, le=1.0)
    cwd: str | None = None


class Scorecard(BaseModel):
    """A provider's answer to `run`. The keys below are the contract; a provider
    adds its own (AgentSmith's carry every field its scorecard always had, which
    the security harness, the promotion loop and the evidence pack read)."""

    model_config = ConfigDict(extra="allow")

    schema_version: Literal[1] = Field(default=1, alias="schema")
    suite: Literal["golden", "fairness", "hallucination", "adversarial", "rag_poison"]
    verdict: Literal["pass", "fail", "no_verdict", "not_gradable"]
    reason: str = ""
    threshold: float | None = None
    fail_above: float | None = None
    cases_total: int = Field(default=0, ge=0)
    cases_graded: int = Field(default=0, ge=0)


class EvalsPort(_Frozen):
    """`providers.evals` in .agenticframework/providers.json. `no_verdict` and
    `not_gradable` name the suites where that verdict only warns — an exception to
    closed-in-CI the tenant declares, in an always-governed file."""

    command: str = Field(min_length=1)
    version: str | None = None
    contract: Literal[1] | None = None
    no_verdict: dict[Literal["golden", "fairness", "hallucination", "adversarial", "rag_poison"],
                     Literal["warn"]] = Field(default_factory=dict)
    not_gradable: dict[Literal["golden", "fairness", "hallucination", "adversarial", "rag_poison"],
                       Literal["warn"]] = Field(default_factory=dict)


# ── The security a provider checks for a tenant (contract/security/v1/) ──────
#
# The tenant owns its security pack and its declared posture; the provider
# checks them and answers with one row per control, each saying whose evidence
# it is. The pack files are the tenant's, so they validate here rather than in
# whichever runner reads them (.agent-rfc/designs/security-contract.md).

SECURITY_SUBJECTS = ("repository", "provider")
SECURITY_RESULTS = ("pass", "fail", "gap", "not_applicable")
SECURITY_VERDICTS = ("pass", "fail", "not_gradable")
REDACTION_VERDICTS = ("pass", "fail", "not_gradable", "not_applicable")
# SEC-<AREA>-NNN, where a tenant's area may have parts (SEC-KYC-FLOOR-001).
CONTROL_ID = r"^SEC-[A-Z0-9]+(?:-[A-Z0-9]+)*-[0-9]{3}$"


class ControlFrameworks(_Frozen):
    owasp: list[str] = Field(default_factory=list)
    nist: list[str] = Field(default_factory=list)
    atlas: list[str] = Field(default_factory=list)
    iso42001: list[int] = Field(default_factory=list)


class TenantControl(BaseModel):
    """One row of a tenant's `.agent-rfc/security/control_registry.json` — a
    control the tenant adds, evidenced by a suite in its own repository. Only
    adds: an id the provider already checks is refused where the rows are read."""

    model_config = ConfigDict(extra="allow")

    id: str = Field(pattern=CONTROL_ID)
    title: str = Field(min_length=1)
    status: Literal["met", "partial", "gap", "org-owned"]
    owner: Literal["framework", "tenant", "shared"] = "tenant"
    frameworks: ControlFrameworks = Field(default_factory=ControlFrameworks)
    suite: str | None = Field(default=None, min_length=1)
    mechanism: str = ""

    @model_validator(mode="after")
    def _evidenced(self) -> "TenantControl":
        if self.status in ("met", "partial") and not self.suite:
            raise ValueError(f"{self.id} claims {self.status!r} and names no `suite` that evidences it")
        return self


class RiskEntry(_Frozen):
    id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    severity: Literal["low", "medium", "high", "critical"]
    mitigations: list[str]
    control_ids: list[Annotated[str, Field(pattern=CONTROL_ID)]]


class RiskRegister(_Frozen):
    """`.agent-rfc/security/risk_register.yaml`."""

    version: int | None = Field(default=None, ge=1)
    entries: list[RiskEntry] = Field(min_length=1)


class AgencyAction(BaseModel):
    model_config = ConfigDict(extra="allow")

    workflow: str = Field(min_length=1)
    action: str = Field(min_length=1)
    needs_hitl: bool
    notes: str | None = None


class AgencyManifest(BaseModel):
    """`.agent-rfc/security/agency_manifest.yaml` — which actions need a human."""

    model_config = ConfigDict(extra="allow")

    version: int | None = Field(default=None, ge=1)
    actions: list[AgencyAction] = Field(min_length=1)


class AllowedTool(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str = Field(min_length=1)
    allowed: bool = True


class ToolAllowlist(BaseModel):
    """`.agent-rfc/security/tool_allowlist.yaml` — an empty list denies every tool."""

    model_config = ConfigDict(extra="allow")

    tools: list[AllowedTool] = Field(default_factory=list)


class SecurityRequest(BaseModel):
    """What a caller sends `check` on stdin. Unknown keys are ignored."""

    model_config = ConfigDict(extra="allow")

    controls: list[Annotated[str, Field(pattern=CONTROL_ID)]] | None = None
    evidence_dir: str | None = Field(default=None, min_length=1)
    cwd: str | None = None


class ProviderName(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str = Field(min_length=1)
    version: str = Field(min_length=1)


class ControlRow(BaseModel):
    """One control in a `check` result."""

    model_config = ConfigDict(extra="allow")

    id: str = Field(pattern=CONTROL_ID)
    title: str = ""
    subject: Literal["repository", "provider"]
    result: Literal["pass", "fail", "gap", "not_applicable"]
    message: str = ""
    frameworks: ControlFrameworks = Field(default_factory=ControlFrameworks)
    evidence: dict[str, str] = Field(default_factory=dict)


class SecurityResult(BaseModel):
    """A provider's answer to `check`. A provider adds keys of its own."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    schema_version: Literal[1] = Field(default=1, alias="schema")
    verdict: Literal["pass", "fail", "not_gradable"]
    reason: str = ""
    provider: ProviderName
    controls: list[ControlRow] = Field(default_factory=list)
    counts: dict[Literal["pass", "fail", "gap", "not_applicable"], int] = Field(default_factory=dict)


class RedactionRequest(BaseModel):
    """What a caller sends `redaction` on stdin. Unknown keys are ignored."""

    model_config = ConfigDict(extra="allow")

    environment: Literal["staging", "production"]
    emitter: str | None = Field(default=None, min_length=1)
    cwd: str | None = None


class RedactionResult(BaseModel):
    """A provider's answer to `redaction`: whether a probe reached the wire. A
    leak is named by the probe's kind, never its text."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    schema_version: Literal[1] = Field(default=1, alias="schema")
    verdict: Literal["pass", "fail", "not_gradable", "not_applicable"]
    reason: str = ""
    environment: Literal["staging", "production"]
    leaked: list[str] = Field(default_factory=list)


class SecurityPort(_Frozen):
    """`providers.security` in .agenticframework/providers.json. `emitter` is the
    command that exports one representative run of the repository's telemetry —
    `"none"` when it emits none."""

    command: str = Field(min_length=1)
    version: str | None = None
    contract: Literal[1] | None = None
    emitter: str | None = Field(default=None, min_length=1)


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
    #   otel_endpoint  the collector those files name (contract/rules/v1: a render
    #                  reads declarations, never the environment)
    #   artifacts      this repo's own document layout, over the framework's
    #   evals          per-suite thresholds an evals provider applies (contract/evals/v1)
    session_start: list[str] = Field(default_factory=list)
    rules_extra: list[str] = Field(default_factory=list)
    test_command: str | None = None
    otel_endpoint: str | None = Field(default=None, pattern=r"^https?://[^\s]+$")
    artifacts: "Artifacts | None" = None
    evals: dict[Literal["golden", "fairness", "hallucination", "adversarial", "rag_poison"],
                "EvalsThresholds"] = Field(default_factory=dict)


class EvalsThresholds(_Frozen):
    """One suite's bars: a floor on the average score, a ceiling on a rate."""

    fail_below: float | None = Field(default=None, ge=0.0, le=1.0)
    fail_above: float | None = Field(default=None, ge=0.0, le=1.0)


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
