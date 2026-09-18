#!/usr/bin/env python3
"""
scripts/process_gate.py — design before code, review before merge, enforced.

docs/design-review-checklist.md and docs/review-levers.md were written down and
skipped: nothing put them in an agent's context, and nothing checked. This is
the one implementation every enforcement layer calls, in AgentSmith and in any
tenant that adopts it (docs/process-gates.md; designs:
.agent-rfc/designs/process-gates.md, process-gates-tenants.md):

    session-start   Claude Code SessionStart hook — states the rules and gates
    pre-edit        Claude Code PreToolUse hook — denies an edit to a gated
                    path that no active design note covers
    stop            Claude Code Stop hook — blocks ending a turn with gated
                    changes that have no clean review newer than them
    commit-msg F    .githooks/commit-msg — the staged commit needs Design: and
                    Review: trailers that resolve
    ci              CI — the same per-commit checks over a pushed range, plus a
                    CHANGELOG rule where the repo declares one

What a repo gates is declared in its own `.agenticframework/process-gates.json`
(CONFIG below). A repo without one has not adopted the gates: the local hooks
do nothing there, and `ci` fails, because CI running the gate means the repo
had adopted it.

Runs in the framework environment — Python 3.11+ with pydantic and
opentelemetry — which .githooks/process-gate resolves ($AGENTSMITH_PYTHON,
$AGENTSMITH_DIR/.venv, ~/.agent-framework/.venv, a vendored repo's .venv). The
records it reads are Pydantic models (scripts/gate_models.py) and every decision
is a spooled span (scripts/gate_tracing.py): the framework follows its own
pillars 3 and 7 with no exception (.agent-rfc/designs/governance-enforcement.md).
This file alone stays parseable by an older interpreter so that, run by one, it
can say why it cannot gate (exit 3) instead of dying on a syntax error.

What no gate here can do is judge whether a design is GOOD. They prove a design
and a clean review exist, are scoped to the change, cite real levers, and are
as fresh as the change — quality still needs a reviewer who is not the builder.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

# The gate must not change the tree it checks: importing its own modules would
# otherwise drop scripts/__pycache__/ into the repo, which the stop and commit
# gates then see as gated changes nobody made.
sys.dont_write_bytecode = True
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

EXIT_UNUSABLE = 3  # the interpreter cannot run the gate; the launcher tries the next one
UNUSABLE: Optional[str] = None
if sys.version_info < (3, 11):
    UNUSABLE = f"Python {sys.version.split()[0]} at {sys.executable} is older than 3.11"
else:
    try:
        import gate_history as gh
        import gate_ides as gi
        import gate_models as gm
        import gate_kg as lkg
        import gate_shell as gsh
        import gate_pillars as gp
        import gate_tracing as gt
    except Exception as _exc:  # named below, not swallowed: the launcher prints it
        UNUSABLE = f"{sys.executable} cannot import the gate's models or tracing ({type(_exc).__name__}: {_exc})"

CONFIG = ".agenticframework/process-gates.json"
DESIGNS_DIR = ".agent-rfc/designs"
REVIEWS_DIR = ".agent-rfc/reviews"
FRAMEWORK_PREFIX = "@framework/"
# The directory this script was installed from: an AgentSmith checkout, or
# ~/.agent-framework. `@framework/<path>` in a config resolves against it.
FRAMEWORK_ROOT = Path(__file__).resolve().parents[1]

SMALL_CHANGE_LINES = 20
# Where the graph lives, read at the commit being checked like every record.
KG_FIXTURE = ".agent-rfc/fixtures/knowledge_graph.json"
_KG_QUERY = re.compile(r"^\s*KG query[^:\n]*:\s*(\S+)\s*$", re.M)
Reader = Callable[[str], Optional[str]]


# ── Globs ────────────────────────────────────────────────────────────────────

_GLOB_CACHE: Dict[str, "re.Pattern[str]"] = {}


def glob_match(path: str, pattern: str) -> bool:
    """GitHub-style glob: `**` crosses directories, `*` and `?` do not."""
    rx = _GLOB_CACHE.get(pattern)
    if rx is None:
        out, i = [], 0
        while i < len(pattern):
            if pattern.startswith("**", i):
                out.append(".*")
                i += 2
            elif pattern[i] == "*":
                out.append("[^/]*")
                i += 1
            elif pattern[i] == "?":
                out.append("[^/]")
                i += 1
            else:
                out.append(re.escape(pattern[i]))
                i += 1
        rx = _GLOB_CACHE[pattern] = re.compile("".join(out) + r"\Z")
    return bool(rx.match(path))


def _any(path: str, patterns) -> bool:
    return any(glob_match(path, p) for p in patterns)


# ── Configuration ────────────────────────────────────────────────────────────


class Config:
    """A repo's declaration of what the gates cover (`CONFIG`)."""

    def __init__(self, data: dict) -> None:
        self.gated: List[str] = list(data.get("gated") or [])
        self.not_gated: List[str] = list(data.get("not_gated") or [])
        self.levers_doc: str = data.get("levers_doc") or "docs/review-levers.md"
        self.design_checklist: str = data.get("design_checklist") or "docs/design-review-checklist.md"
        changelog = data.get("changelog") or {}
        self.changelog_file: Optional[str] = changelog.get("file")
        self.changelog_paths: List[str] = list(changelog.get("paths") or [])
        self.changelog_except: List[str] = list(changelog.get("except") or [])
        # The rules registry: the framework's by default — beside the running
        # script, which in a vendored tenant is the tenant's own synced copy.
        self.registry: str = data.get("registry") or f"{FRAMEWORK_PREFIX}templates/governance.json"
        # Declaring `registry` is how a repo ADOPTS the registry's design-time
        # requirements — pillars, deviations, dependencies, the sign-off block.
        # A commit whose own config does not declare one predates them, and no
        # design written then could satisfy them, so `ci` judges it by the rules
        # it was made under. Same rule as a repo that had not adopted the gates
        # at all: every commit is judged by the config it carries.
        self.registry_declared: bool = "registry" in data
        # One artifact per type, and the cross-reference rule that travels with
        # it: "off" (default), "report" or "enforce". The check ships in G5a and
        # the documents move in G5b, so a repo that has not migrated says
        # "report" — it is told what to fix without being stopped from fixing it.
        self.artifacts_mode: str = str(data.get("artifacts") or "off")
        # Where the records live: "legacy" is a file per change under
        # .agent-rfc/, "single" is a section in the design artifact and entries
        # in the review log. A repo keeps exactly one convention, so a change
        # can never be recorded in a place the gate does not read.
        self.records_mode: str = str(data.get("records") or "legacy")
        # Whether this repo is held to the pillar checks G6 makes mechanical,
        # and to evidence in its designs' pillar answers. It lives here, not in
        # the shared registry, because THIS file is read at the commit being
        # checked: a requirement added to the registry would judge every commit
        # ever made by rules that did not exist when they were made.
        # Whether a review must name the scope it covered, as the knowledge
        # graph computes it. Same three modes and the same reason as the two
        # above: this file is read at the commit being checked.
        self.kg_mode: str = str(data.get("knowledge_graph") or "off")
        self.pillars_declared: bool = "pillars" in data
        self.pillar_policy, self.pillar_problems = gp.parse_policy(data.get("pillars"))
        self.extends_data = data.get("extends")

    def problems(self) -> List[str]:
        errors = []
        if not self.gated:
            errors.append(f"{CONFIG} declares no gated paths")
        elif not self.is_gated(CONFIG):
            errors.append(f"{CONFIG} must gate itself, or the gates can be switched off unreviewed")
        if self.changelog_file and not self.changelog_paths:
            errors.append(f"{CONFIG} names a changelog file but no paths that require it")
        errors.extend(f"{CONFIG} {problem}" for problem in self.pillar_problems)
        if self.extends_data is not None:
            try:
                gm.Extends.model_validate(self.extends_data)
            except gm.ValidationError as exc:
                errors.append(f"{CONFIG} 'extends' is invalid: {exc.errors()[0]['loc']} {exc.errors()[0]['msg']}")
        return errors

    def load_registry(self, read: Reader) -> Tuple[Optional["gm.Registry"], List[str]]:
        """-> (registry merged with `extends`, problems). A missing or invalid
        registry is a problem, never a reason to check less."""
        text = self.doc_text(self.registry, read)
        if text is None:
            return None, [f"the rules registry {self.display(self.registry)} does not exist — "
                          "run scripts/generate-ide-config.py --registry, or re-sync AgentSmith"]
        try:
            registry = gm.Registry.model_validate_json(text)
            extends = gm.Extends.model_validate(self.extends_data) if self.extends_data is not None else None
            return registry.merged(extends), []
        except (gm.ValidationError, ValueError) as exc:
            return None, [f"the rules registry {self.display(self.registry)} is invalid: {exc}"]

    def is_gated(self, path: str) -> bool:
        return _any(path, self.gated) and not _any(path, self.not_gated)

    def needs_changelog(self, path: str) -> bool:
        return bool(self.changelog_file) and _any(path, self.changelog_paths) \
            and not _any(path, self.changelog_except) and not path.endswith(".md")

    def doc_text(self, value: str, read: Reader) -> Optional[str]:
        """A repo path is read at the commit being checked; `@framework/…`
        beside this script, because an installed-mode tenant carries no copy."""
        if value.startswith(FRAMEWORK_PREFIX):
            path = FRAMEWORK_ROOT / value[len(FRAMEWORK_PREFIX):]
            return path.read_text(encoding="utf-8") if path.is_file() else None
        return read(value)

    def display(self, value: str) -> str:
        if value.startswith(FRAMEWORK_PREFIX):
            return f"{value[len(FRAMEWORK_PREFIX):]} in your AgentSmith checkout or ~/.agent-framework"
        return value


def parse_config(text: Optional[str]) -> Tuple[Optional[Config], List[str]]:
    """-> (config, problems). (None, []) when there is no config: not adopted."""
    if text is None:
        return None, []
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        return None, [f"{CONFIG} is not valid JSON: {exc}"]
    if not isinstance(data, dict):
        return None, [f"{CONFIG} must be a JSON object"]
    config = Config(data)
    return config, config.problems()


def lever_slugs(levers_text: str) -> set:
    return set(re.findall(r"^- `([a-z0-9-]+)`", levers_text, re.M))


# ── Records ──────────────────────────────────────────────────────────────────


def front_matter(text: str) -> Tuple[Dict[str, object], str]:
    """The small YAML subset the records use: `key: value` and `key:` + `  - item`."""
    if not text.startswith("---\n"):
        return {}, text
    end = text.find("\n---", 4)
    if end < 0:
        return {}, text
    meta: Dict[str, object] = {}
    key = None
    for line in text[4:end].splitlines():
        item = re.match(r"^\s+-\s+(.+?)\s*$", line)
        if item and key is not None:
            value = meta.setdefault(key, [])
            if isinstance(value, list):
                value.append(item.group(1))
            continue
        pair = re.match(r"^([A-Za-z_][\w-]*):\s*(.*?)\s*$", line)
        if pair:
            key = pair.group(1)
            meta[key] = pair.group(2) if pair.group(2) else []
    return meta, text[end + 4:]


def _section(body: str, heading: str) -> Optional[str]:
    match = re.search(rf"^## {re.escape(heading)}\b.*?$(.*?)(?=^## |\Z)", body, re.M | re.S)
    return match.group(1) if match else None


# What a design had to carry before the registry existed (pre-G1 commits).
PRE_REGISTRY_SECTIONS = ["Problem", "Approach", "Levers"]


def evidence_problems(body: str, evidence: "gp.Resolver") -> List[str]:
    """The one place the '## Pillars' section is handed to the evidence rule:
    `enforce` reads it as errors and `report` as notes, and they must be reading
    the same thing (`one-verdict`)."""
    return gm.check_evidence(_section(body, "Pillars") or "", evidence)


def check_design(
    text: str,
    known_slugs: set,
    levers_doc: str,
    registry: "gm.Registry",
    approvals: List["gm.Approval"],
    design_path: str,
    adopted: bool = True,
    evidence: Optional["gp.Resolver"] = None,
) -> List[str]:
    errors = []
    meta, body = front_matter(text)
    if not meta:
        return ["has no front matter (--- status: active|done, scope: [globs] ---)"]
    if meta.get("status") not in ("active", "done"):
        errors.append(f"status is {meta.get('status')!r}, expected active or done")
    scope = meta.get("scope")
    if not isinstance(scope, list) or not scope:
        errors.append("scope lists no paths")
    for heading in (registry.records.design_sections if adopted else PRE_REGISTRY_SECTIONS):
        if _section(body, heading) is None:
            errors.append(f"has no '## {heading}' section")
    deviations, deviation_errors = gm.parse_deviations(_section(body, "Deviations") or "")
    if _section(body, "Deviations") is not None:
        errors.extend(deviation_errors)
        errors.extend(gm.check_approvals(deviations, approvals, design_path))
    if _section(body, "Pillars") is not None:
        errors.extend(gm.check_pillars(_section(body, "Pillars") or "", registry, deviations))
        if evidence is not None:
            errors.extend(evidence_problems(body, evidence))
    # Backticked names in the section that are real levers. Other backticked
    # names (a file, a flag) are allowed alongside; what is required is that
    # the checklist was worked and at least one lever named.
    cited = set(re.findall(r"`([a-z0-9]+(?:-[a-z0-9]+)+)`", _section(body, "Levers") or ""))
    if not cited & known_slugs:
        errors.append(f"'## Levers' cites no lever from {levers_doc} — work its checklist and name what applied")
    return errors


def design_scope(text: str) -> List[str]:
    scope = front_matter(text)[0].get("scope")
    return list(scope) if isinstance(scope, list) else []


_PASS = re.compile(r"^##\s+Pass\s+(\d+)\s+[—–-]+\s+findings:\s*(\d+)\s*$", re.M)


def check_review(text: str, registry: "gm.Registry", adopted: bool = True) -> List[str]:
    passes = [(int(n), int(k)) for n, k in _PASS.findall(text)]
    if not passes:
        return ["records no passes ('## Pass N — findings: K')"]
    errors = []
    numbers = [n for n, _ in passes]
    if numbers != list(range(1, len(numbers) + 1)):
        errors.append(f"passes are numbered {numbers}, expected 1..{len(numbers)} in order")
    if passes[-1][1] != 0:
        errors.append(
            f"last pass (Pass {passes[-1][0]}) reports {passes[-1][1]} finding(s) — fix them and run another pass"
        )
    elif adopted:
        # Clean passes are half of done; the sign-off states per group what was
        # checked, what did not apply and what is a declared gap.
        errors.extend(gm.check_signoff(text, registry))
    return errors


def load_approvals(read: Reader) -> Tuple[List["gm.Approval"], List[str]]:
    return gm.parse_approvals(read(gm.APPROVALS_FILE))


# ── Trailers ─────────────────────────────────────────────────────────────────


def trailer(message: str, name: str) -> Optional[str]:
    found = re.findall(rf"^{name}:[ \t]*(.*?)[ \t]*$", message, re.M | re.I)
    return found[-1] if found else None


def resolve_record(value: str, directory: str, single: bool = False,
                   artifact: str = "") -> Tuple[Optional[str], Optional[str]]:
    """-> (path, error). The value arrives from a commit message, so it names
    either a Markdown file directly inside `directory` (legacy) or a slug in
    this repo's record artifact (single). A repo keeps one convention; the
    other is refused by name, so nobody records a change where nothing reads."""
    if single:
        path, slug = split_record_ref(value)
        if path != artifact or not slug:
            return None, f"must name {artifact}#<slug> in this repo, got {value!r}"
        return f"{path}#{slug}", None
    if "#" in value:
        return None, (f"must name a file directly under {directory}/, got {value!r} — "
                      "this repo keeps a record per change, not sections in one artifact")
    path = value.strip()
    if path.startswith("./"):
        path = path[2:]
    parent, _, name = path.rpartition("/")
    if parent != directory or not name.endswith(".md") or "/" in name or name.startswith("."):
        return None, f"must name a file directly under {directory}/, got {value!r}"
    return path, None


def kg_problems(files: List[str], review_text: str, read: Reader) -> List[str]:
    """The review names the scope it covered, and the scope is recomputed here.

    The line is a hash of the impacted FILE SET — the change plus one hop of
    dependents — so it says which scope was reviewed, not which bytes. A
    missing graph is its own answer: "no graph" and "the scope matches" are
    different facts (`ambiguous-signals`).
    """
    graph_text = read(KG_FIXTURE)
    if graph_text is None:
        return [f"{KG_FIXTURE} is not in this commit, so the scope of the review cannot be checked — "
                "build it (`python3 scripts/map_codebase.py`) and commit it"]
    try:
        graph = json.loads(graph_text)
    except json.JSONDecodeError as exc:
        return [f"{KG_FIXTURE} is not valid JSON ({exc}) — rebuild it with scripts/map_codebase.py"]
    # The files the change LEAVES. A deleted path is nothing a reviewer can read,
    # and whether it is listed at all depends on who asked git: `git diff` detects
    # renames and names only the new path, `git diff-tree` does not and names
    # both. Found when a commit that renamed four documents passed the commit gate
    # and failed CI with a different hash for the same change.
    expected = lkg.impact(graph, [f for f in files if read(f) is not None])
    found = _KG_QUERY.search(review_text)
    if not found:
        return ["records no 'KG query:' line — run `python3 scripts/local_knowledge_graph.py --impact "
                f"--base HEAD`, read what it lists, and put its hash in the sign-off ({expected.query})"]
    if found.group(1) != expected.query:
        return [f"'KG query: {found.group(1)}' is not the scope of this change ({expected.query}) — "
                f"{len(expected.files)} file(s) are in it, including "
                + ", ".join(expected.files[:3]) + ("…" if len(expected.files) > 3 else "")]
    return []


def check_change(
    files: List[str],
    gated_lines: int,
    message: str,
    read: Reader,
    config: Config,
    added: Optional[List[str]] = None,
    previous: Optional[Reader] = None,
    evidence: Optional["gp.Resolver"] = None,
) -> Tuple[List[str], List[str]]:
    """One commit's worth of files against its message. -> (errors, notes)."""
    # The cross-reference rule is about documents, which are mostly ungated, so
    # it runs before the gated-paths shortcut below.
    xref: List[str] = []
    if config.artifacts_mode in ("report", "enforce") and added:
        xref = cross_reference_problems(added)
    gated = sorted(f for f in files if config.is_gated(f))
    if not gated:
        if config.artifacts_mode == "enforce":
            return xref, []
        return [], [f"artifacts (report): {problem}" for problem in xref]
    errors: List[str] = list(xref) if config.artifacts_mode == "enforce" else []
    notes: List[str] = [] if config.artifacts_mode == "enforce" else [
        f"artifacts (report): {problem}" for problem in xref
    ]
    small = gated_lines <= SMALL_CHANGE_LINES
    known_slugs = lever_slugs(config.doc_text(config.levers_doc, read) or "")
    levers_shown = config.display(config.levers_doc)
    registry, registry_errors = config.load_registry(read)
    if registry is None:
        return registry_errors, []
    approvals, approval_errors = load_approvals(read)
    errors.extend(approval_errors)

    # What this commit does to the policy it inherited. The ratchet runs in
    # every mode: it guards the policy itself, which `report` does not exempt
    # anyone from.
    policy = config.pillar_policy
    previous_config = parse_config(previous(CONFIG))[0] if previous is not None else None
    inherited = previous_config.pillar_policy if previous_config is not None \
        and previous_config.pillars_declared else None
    errors.extend(gp.transition_problems(inherited, policy, approvals))
    design_body = ""

    def na(name: str, value: str) -> bool:
        match = re.match(r"^n/?a\s*[:—-]\s*(\S.*)$", value, re.I)
        if not match:
            return False
        if small:
            notes.append(f"{name}: n/a ({match.group(1)}) — {gated_lines} gated line(s)")
        else:
            errors.append(
                f"{name}: n/a is allowed only for changes of at most {SMALL_CHANGE_LINES} gated lines; "
                f"this one changes {gated_lines}"
            )
        return True

    single = config.records_mode == "single"
    design_wanted = f"{DESIGN_ARTIFACT}#<slug>" if single else f"{DESIGNS_DIR}/<slug>.md"
    review_wanted = f"{REVIEW_ARTIFACT}#<slug>" if single else f"{REVIEWS_DIR}/<slug>.md"

    design_value = trailer(message, "Design")
    if design_value is None:
        errors.append(f"missing 'Design: {design_wanted}' trailer (gated paths: {', '.join(gated[:5])}"
                      f"{' …' if len(gated) > 5 else ''})")
    elif not na("Design", design_value):
        path, err = resolve_record(design_value, DESIGNS_DIR, single, DESIGN_ARTIFACT)
        file_path, slug = split_record_ref(path or "")
        document = read(file_path) if file_path else None
        text = design_section(document or "", slug or "") if (single and document is not None) else document
        if err:
            errors.append(f"Design: {err}")
        elif document is None:
            errors.append(f"Design: {file_path} does not exist in this commit")
        elif text is None:
            errors.append(f"Design: {file_path} has no '## {ACTIVE_CHANGE}{slug}' section in this commit")
        else:
            design_errors = check_design(text, known_slugs, levers_shown, registry, approvals, path,
                                         config.registry_declared,
                                         evidence if policy.mode == "enforce" else None)
            errors.extend(f"Design: {path} {e}" for e in design_errors)
            if policy.mode == "report" and evidence is not None:
                notes.extend(f"pillars (report): Design: {path} {e}"
                             for e in evidence_problems(front_matter(text)[1], evidence))
            design_body = front_matter(text)[1]
            scope = design_scope(text)
            uncovered = [f for f in gated if not _any(f, scope)]
            if scope and uncovered:
                errors.append(f"Design: {path} scope does not cover {', '.join(uncovered)}")

    # The pillars this repo is held to in the code. After the design, because
    # `P2-dependencies` reads the '## Dependencies' section that design carries.
    if policy.mode in ("report", "enforce"):
        mechanical = gp.mechanical_problems(
            gated, read, registry, policy, previous, _section(design_body, "Dependencies") or "")
        if policy.mode == "enforce":
            errors.extend(mechanical)
        else:
            notes.extend(f"pillars (report): {problem}" for problem in mechanical)

    review_value = trailer(message, "Review")
    if review_value is None:
        errors.append(f"missing 'Review: {review_wanted}' trailer")
    elif not na("Review", review_value):
        path, err = resolve_record(review_value, REVIEWS_DIR, single, REVIEW_ARTIFACT)
        file_path, slug = split_record_ref(path or "")
        document = read(file_path) if file_path else None
        text = review_entries(document or "", slug or "") if (single and document is not None) else document
        if err:
            errors.append(f"Review: {err}")
        elif document is None:
            errors.append(f"Review: {file_path} does not exist in this commit")
        elif text is None:
            errors.append(f"Review: {file_path} records no passes for {slug} in this commit")
        else:
            errors.extend(f"Review: {path} {e}" for e in check_review(text, registry, config.registry_declared))
            if config.kg_mode in ("report", "enforce"):
                scope = [f"Review: {path} {e}" for e in kg_problems(files, text, read)]
                (errors if config.kg_mode == "enforce" else notes).extend(
                    scope if config.kg_mode == "enforce" else [f"knowledge graph (report): {s}" for s in scope])
            if file_path not in files:
                errors.append(
                    f"Review: {file_path} is not changed in this commit — a review older than the change "
                    "cannot vouch for it; record the pass that covers this change"
                )
    return errors, notes


# ── git ──────────────────────────────────────────────────────────────────────


def git(*args: str, cwd: Optional[Path] = None, check: bool = True) -> str:
    result = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=False)
    if check and result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)}: {result.stderr.strip()}")
    return result.stdout


def repo_root(start: Optional[str] = None) -> Path:
    out = git("rev-parse", "--show-toplevel", cwd=Path(start) if start else None, check=False).strip()
    return Path(out) if out else Path(start or os.getcwd())


def _reader_at(root: Path, rev: str) -> Reader:
    """`rev` "" reads the index; otherwise a commit."""
    def read(path: str) -> Optional[str]:
        result = subprocess.run(
            ["git", "show", f"{rev}:{path}"], cwd=root, capture_output=True, text=True, check=False
        )
        return result.stdout if result.returncode == 0 else None
    return read


def _worktree_reader(root: Path) -> Reader:
    def read(path: str) -> Optional[str]:
        target = root / path
        return target.read_text(encoding="utf-8") if target.is_file() else None
    return read


def _added_lines(diff: str) -> List[str]:
    """The lines a diff adds, without the `+`. `-U0` keeps this to what changed."""
    return [line[1:] for line in diff.splitlines() if line.startswith("+") and not line.startswith("+++")]


def _gated_lines(numstat: str, config: Config) -> int:
    total = 0
    for line in numstat.splitlines():
        parts = line.split("\t")
        if len(parts) >= 3 and config.is_gated(parts[2]):
            added, removed = parts[0], parts[1]
            # A binary file ("-") counts as more than a small change.
            total += int(added) if added.isdigit() else SMALL_CHANGE_LINES + 1
            total += int(removed) if removed.isdigit() else 0
    return total


# ── Subcommands ──────────────────────────────────────────────────────────────


def _worktree_config(root: Path) -> Tuple[Optional[Config], List[str]]:
    return parse_config(_worktree_reader(root)(CONFIG))


def active_designs(root: Path, config: Config) -> List[Tuple[str, str, List[str]]]:
    """-> [(relpath, text, errors)] for designs with status: active. A registry
    or approvals file that cannot be read makes every design incomplete."""
    read = _worktree_reader(root)
    slugs = lever_slugs(config.doc_text(config.levers_doc, read) or "")
    registry, shared_errors = config.load_registry(read)
    approvals, approval_errors = load_approvals(read)
    shared_errors = shared_errors + approval_errors
    def checked(rel: str, text: str) -> Tuple[str, str, List[str]]:
        errors = list(shared_errors)
        if registry is not None:
            errors += check_design(text, slugs, config.display(config.levers_doc), registry, approvals, rel,
                                   config.registry_declared)
        return rel, text, errors

    found = []
    if config.records_mode == "single":
        document = read(DESIGN_ARTIFACT) or ""
        for slug in _slug_sections(document, ACTIVE_CHANGE):
            text = design_section(document, slug) or ""
            if front_matter(text)[0].get("status") == "active":
                found.append(checked(f"{DESIGN_ARTIFACT}#{slug}", text))
        return found
    for path in sorted((root / DESIGNS_DIR).glob("*.md")):
        text = path.read_text(encoding="utf-8")
        if front_matter(text)[0].get("status") == "active":
            found.append(checked(f"{DESIGNS_DIR}/{path.name}", text))
    return found


# What the last decision was, for the span. Read from the decision itself, not
# inferred from the text a hook printed: a formatting change to that JSON would
# quietly relabel every deny as an allow (`ambiguous-signals`).
_DECISION: Dict[str, Optional[str]] = {"value": None, "rule": None}
# Which IDE asked, and so which dialect the answer is written in. Set once in
# main() from --ide / $AGENTSMITH_IDE; the rules below never look at it.
_IDE: Dict[str, str] = {"value": gi.DEFAULT_IDE if not UNUSABLE else "claude"}


def _record(decision: str, rule: Optional[str] = None) -> None:
    _DECISION["value"] = decision
    if rule:
        _DECISION["rule"] = rule


def _deny(reason: str) -> None:
    _record("deny")
    print(gi.render(_IDE["value"], "deny", reason))


def cmd_pre_edit(payload: dict) -> int:
    try:
        event = gi.parse(_IDE["value"], "pre-edit", payload)
    except gi.Unreadable as unreadable:
        # Fail closed: a write this cannot read is refused, naming what it saw.
        _deny(str(unreadable))
        return 0
    if event.kind == "shell" and event.command:
        # The same agent that cannot edit a gated file can type `git commit
        # --no-verify`. This refuses the obvious ways round; the sweep is what
        # catches the rest (gate_shell.py states that limit).
        hooks_path = git("config", "--get", "core.hooksPath",
                         cwd=repo_root(event.cwd), check=False).strip() or ".githooks"
        refusal = gsh.refusal(event.command, hooks_path=hooks_path)
        if refusal:
            _record("deny", "shell-bypass")
            _deny(refusal)
        return 0
    if event.kind != "edit" or not event.paths:
        return 0  # a read, or a tool that changes nothing
    target = event.paths[0]
    root = repo_root(event.cwd)
    try:
        rel = Path(target).resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return 0  # outside this repository
    config, problems = _worktree_config(root)
    if problems:
        # A broken config is not an absent one: the repo adopted the gates.
        if rel != CONFIG:
            # It cannot say what is gated, so nothing but the config itself may change.
            _deny(f"the process-gate config is broken — {'; '.join(problems)}. Fix {CONFIG} first.")
        return 0
    if rel == gm.APPROVALS_FILE:
        # No design scope can unlock this one: the record of the owner's
        # permission is written by `agentsmith approve` at a terminal. An agent
        # that could edit the file could approve its own deviation.
        _deny(
            f"{rel} records the owner's approvals and is never edited directly — "
            "ask the owner to run `agentsmith approve <design> <deviation>` in a terminal."
        )
        return 0
    if config is None or not config.is_gated(rel):
        return 0
    designs = active_designs(root, config)
    covering = [(p, errs) for p, text, errs in designs if _any(rel, design_scope(text))]
    if [p for p, errs in covering if not errs]:
        return 0
    if covering:
        detail = "; ".join(f"{p}: {'; '.join(errs)}" for p, errs in covering)
        _deny(f"{rel} is covered by a design note that is not complete — {detail}.")
        return 0
    _deny(
        f"{rel} is a gated path and no active design note covers it. Design before code: "
        f"work {config.display(config.design_checklist)}, write {DESIGNS_DIR}/<slug>.md "
        "(front matter status: active, scope: globs covering this file; sections ## Problem, ## Approach, "
        "## Pillars (one answer per pillar), ## Deviations (none, or each with the owner's "
        "approval), "
        f"## Dependencies, ## Levers citing levers from {config.display(config.levers_doc)}) — "
        "`agentsmith design new <slug>` writes the skeleton — then retry. Ask the owner before any deviation. "
        "See AgentSmith's docs/process-gates.md."
    )
    return 0


def _uncommitted_gated(root: Path, config: Config) -> List[str]:
    out = git("status", "--porcelain", "-uall", cwd=root, check=False)
    paths = []
    for line in out.splitlines():
        path = line[3:]
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        path = path.strip('"')
        if config.is_gated(path):
            paths.append(path)
    return sorted(set(paths))


def stop_problems(root: Path) -> List[str]:
    config, problems = _worktree_config(root)
    if problems:
        return problems
    if config is None:
        return []
    changed = _uncommitted_gated(root, config)
    if not changed:
        return []
    designs = active_designs(root, config)
    by_design: Dict[str, List[str]] = {}
    for path in changed:
        covering = [p for p, text, errs in designs if not errs and _any(path, design_scope(text))]
        if not covering:
            problems.append(f"{path}: no complete active design note covers it")
        for design in covering:
            by_design.setdefault(design, []).append(path)
    for design, paths in sorted(by_design.items()):
        review = f"{REVIEWS_DIR}/{Path(design).name}"
        review_path = root / review
        if not review_path.is_file():
            problems.append(f"{design}: no review record at {review} for {', '.join(paths)}")
            continue
        registry, registry_errors = config.load_registry(_worktree_reader(root))
        errs = (registry_errors if registry is None else
                check_review(review_path.read_text(encoding="utf-8"), registry, config.registry_declared))
        if errs:
            problems.append(f"{review}: {'; '.join(errs)}")
            continue
        newest = max(((root / p).stat().st_mtime for p in paths if (root / p).exists()), default=0.0)
        if review_path.stat().st_mtime < newest:
            problems.append(
                f"{review}: last updated before the newest change it covers — run a review pass "
                f"against {config.display(config.levers_doc)} over {', '.join(paths)} and record it"
            )
    return problems


def cmd_stop(payload: dict) -> int:
    event = gi.parse(_IDE["value"], "stop", payload)
    root = repo_root(event.cwd)
    unreviewed = stop_problems(root)
    # A commit that skipped the gate is as unreviewed as an uncommitted change,
    # and the end of a turn is a touchpoint like any other.
    swept, sweep_report, _failing = sweep(root)
    problems = [*unreviewed, *([line.strip() for line in sweep_report if line.strip()] if swept else [])]
    if not problems:
        return 0
    _record("block", "review-before-done" if not swept else "bypass-sweep")
    # Pillar 5's log, written by the hook rather than by an agent's memory of
    # what happened. Unresolved, so the next session start surfaces it.
    gh.record(root, "bypass_found" if swept and not unreviewed else "stop_gate_blocked",
              "; ".join(problems)[:400])
    heading = "Commits that never passed the gate" if swept and not unreviewed else "Unreviewed gated changes"
    text = f"{heading}:\n- " + "\n- ".join(problems)
    # Blocking a turn that is already being blocked could loop forever, so a
    # repeat warns instead. The commit and CI gates still hold.
    print(gi.render(_IDE["value"], "block", text + "\nSee AgentSmith's docs/process-gates.md.",
                    repeat=event.stop_active))
    return 0


def cmd_session_start(payload: dict) -> int:
    root = repo_root(gi.parse(_IDE["value"], "session-start", payload).cwd)
    config, problems = _worktree_config(root)
    if config is None and not problems:
        return 0  # this repository has not adopted the gates
    if problems:
        lines = [
            f"⚠️ The process-gate config is broken: {'; '.join(problems)}. "
            f"Every edit except to {CONFIG} is denied until it is fixed — it cannot say what is gated."
        ]
    else:
        designs = active_designs(root, config)
        registry, _ = config.load_registry(_worktree_reader(root))
        checklist, levers = config.display(config.design_checklist), config.display(config.levers_doc)
        lines = [
            "This repository enforces its build discipline mechanically (AgentSmith docs/process-gates.md).",
            f"1. Design before code: before editing code, work {checklist} and write "
            f"{DESIGNS_DIR}/<slug>.md (status: active, scope globs, ## Problem / ## Approach / ## Pillars / "
            "## Deviations / ## Dependencies / ## Levers; `agentsmith design new <slug>` writes it). "
            "Answer every pillar. If any rule must be deviated from, STOP and ask the owner first: a deviation "
            "counts only with an approval the owner records in a terminal (`agentsmith approve`). "
            "Edits to gated paths without a complete design are denied.",
            f"2. Review before done: after building, run review passes against {levers}, verify each "
            f"finding in code, fix, and record every pass in {REVIEWS_DIR}/<slug>.md as "
            "'## Pass N — findings: K' until a pass finds 0, then a complete '## Sign-off' block "
            "(docs/validation-checklist.md Step 4). Ending a turn with unreviewed changes is blocked.",
            "3. Every commit touching gated paths carries 'Design: <design path>' and 'Review: <review path>' "
            "trailers, and changes the review record in that same commit. CI checks every pushed commit"
            + (f", and {config.changelog_file} for the paths that need it." if config.changelog_file else "."),
            f"Gated paths are declared in {CONFIG}. Bash-made edits are not caught by the edit gate — "
            "the stop, commit and CI gates still see them.",
        ]
        extends = config.extends_data or {}
        lines.extend(str(line) for line in (extends.get("session_start") or []))
        swept, sweep_report, _failing = sweep(root)
        if swept:
            lines.append("⚠️ The bypass sweep found commits that never passed the gate — commits and pushes are "
                         "refused until they are repaired (`agentsmith gates repair`):")
            lines.extend(line.strip() for line in sweep_report if line.strip().startswith(("❌", "✅", "ℹ️", "Repair")))
        if registry is not None:
            answerable = [p for p in registry.pillars if "design" in p.check]
            lines.append(
                f"Pillars a design must answer ({len(answerable)}): "
                + ", ".join(f"P{p.id} {p.name}" for p in answerable)
            )
        if designs:
            # An incomplete design is listed as incomplete: it unlocks nothing,
            # and a bare list of names reads as "these are in force".
            listed = []
            for path, _text, errors in designs:
                listed.append(f"{path} — INCOMPLETE, unlocks nothing: {errors[0]}" if errors else path)
            lines.append("Active designs: " + "; ".join(listed))
        if config.kg_mode in ("report", "enforce"):
            # The graph, read rather than recommended: an agent told to run a
            # script mostly does not, and a summary it can act on is the point
            # of having the graph at all.
            graph_text = _worktree_reader(root)(KG_FIXTURE)
            uncommitted = _uncommitted_gated(root, config)
            if graph_text and uncommitted:
                try:
                    found = lkg.impact(json.loads(graph_text), uncommitted)
                except json.JSONDecodeError:
                    found = None
                if found is not None:
                    lines.append(
                        f"Knowledge graph — this change touches {len(found.files)} file(s): "
                        + ", ".join(found.files[:12]) + ("…" if len(found.files) > 12 else "")
                        + (f". Lever groups: {', '.join(str(g) for g in found.groups)}" if found.groups else "")
                        + f". The review's sign-off carries `KG query: {found.query}`."
                    )
            elif not graph_text:
                lines.append(f"⚠️ {KG_FIXTURE} is missing — build it with "
                             "`python3 scripts/map_codebase.py`, and commit it.")
        if not (root / "AGENTS.md").is_file() and (root / "scripts/generate-ide-config.py").is_file():
            # Other agents (Codex, Cursor, Gemini, Copilot) read these, not this hook.
            lines.append(
                "⚠️ This clone has no generated agent files (AGENTS.md, .cursorrules, …): run "
                "`python3 scripts/generate-ide-config.py --repo-root .` (needs pyyaml; the files are gitignored)."
            )
    if git("config", "--get", "core.hooksPath", cwd=root, check=False).strip() != ".githooks":
        # The sweep re-arms a clone that carries .githooks, and says so there.
        # Reaching here means it could not: no launcher to point at.
        lines.append("⚠️ The commit gate is not armed in this clone and cannot be: it has no .githooks/process-gate. "
                     "Re-sync AgentSmith into this repo, then run `git config core.hooksPath .githooks`.")
    telemetry = gt.status_line(gt.ship())
    if telemetry:
        lines.append(telemetry)
    print(gi.render(_IDE["value"], "context", "\n".join(lines)))
    return 0


_EMPTY_TREE = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"


def cmd_commit_msg(message_file: str, amend: bool = False) -> int:
    root = repo_root()
    read = _reader_at(root, "")  # the index: what this commit will contain
    config, problems = parse_config(read(CONFIG))
    if problems:
        print(f"❌ process gate: commit blocked — {'; '.join(problems)}", file=sys.stderr)
        return 1
    if config is None:
        return 0  # the commit does not carry a config: this repo has not adopted the gates
    message = "\n".join(
        line for line in Path(message_file).read_text(encoding="utf-8").splitlines() if not line.startswith("#")
    )
    # The commit being created replaces HEAD when amending, so its changes are
    # measured from HEAD's parent (or from nothing, for a root commit).
    base = ["HEAD"]
    if amend:
        parent = git("rev-parse", "--verify", "-q", "HEAD^", cwd=root, check=False).strip()
        base = [parent or _EMPTY_TREE]
    elif not git("rev-parse", "--verify", "-q", "HEAD", cwd=root, check=False).strip():
        base = [_EMPTY_TREE]
    # --no-renames: CI lists files with `git diff-tree`, which does not detect
    # renames, so it sees a rename's OLD path as a deletion. `git diff` detects
    # them and hid it — renaming a gated file away escaped the design's scope
    # here and was caught only after the push. Both gates now see the same list.
    files = [f for f in git("diff", "--cached", "--name-only", "--no-renames", *base, cwd=root).splitlines() if f]
    lines = _gated_lines(git("diff", "--cached", "--numstat", "--no-renames", *base, cwd=root), config)

    added = _added_lines(git("diff", "--cached", "-U0", *base, cwd=root, check=False))
    # What this commit is measured against: the config it inherits and the lock
    # files as they were. The index, for the commit's own side, because that is
    # what it will contain — a file added by the same change is evidence the
    # moment it is staged.
    previous = _reader_at(root, base[0])
    evidence = gp.evidence_resolver(root, "") if config.pillar_policy.mode != "off" else None
    errors, notes = check_change(files, lines, message, read, config, added, previous, evidence)

    # A commit that skipped the gate blocks the next commit — unless the next
    # commit is the repair. pre-commit cannot decide that (no message yet), so
    # it is decided here, where the message says which commits it repairs.
    _code, sweep_report, failing = sweep(root)
    if failing:
        claimed = {sha for line in _REPAIRS.findall(message)
                   for token in re.split(r"[\s,]+", line) if token
                   for sha in [git("rev-parse", "--verify", "--quiet", f"{token}^{{commit}}",
                                   cwd=root, check=False).strip()] if sha}
        unrepaired = [f for f in failing if f[0] not in claimed]
        if unrepaired:
            errors.extend([
                *[line.strip() for line in sweep_report if line.strip().startswith("❌")],
                f"this commit must repair them: add a `Repairs: {unrepaired[0][0][:12]}` trailer "
                "(one per commit) to the message that brings them under a design and review",
            ])

    for note in notes:
        print(f"ℹ️  process gate: {note}")
    if errors:
        _record("block", "design-and-review-trailers")
        print("❌ process gate: commit blocked (AgentSmith docs/process-gates.md)", file=sys.stderr)
        for error in errors:
            print(f"   - {error}", file=sys.stderr)
        return 1
    return 0


def _range_commits(root: Path, base: str, head: str) -> Tuple[List[str], Optional[str]]:
    if not base or set(base) == {"0"}:
        return [head], "no base commit (new branch or first push): checked the head commit only"
    if subprocess.run(["git", "cat-file", "-e", f"{base}^{{commit}}"], cwd=root, check=False).returncode != 0:
        return [head], f"base {base[:12]} is not in this history (force-push?): checked the head commit only"
    out = git("rev-list", "--reverse", "--no-merges", f"{base}..{head}", cwd=root)
    return [c for c in out.splitlines() if c], None


def _report(lines: List[str], annotations: List[str]) -> None:
    text = "\n".join(lines)
    print(text)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as fh:
            fh.write(text + "\n")
    for annotation in annotations:
        print(annotation)



# ── Checking commits ─────────────────────────────────────────────────────────

Failures = List[Tuple[str, str, List[str]]]


def check_commits(root: Path, commits: List[str]) -> Tuple[Failures, List[Tuple[str, str, str]],
                                                           List[Tuple[str, str]], set, int]:
    """Every commit against the config and records it carries.

    One implementation for `ci` (a pushed range) and `sweep` (whatever reached
    this machine without passing the commit gate): two would answer the same
    question differently, and the sweep exists precisely to catch what the other
    layers missed.
    -> (failures, escapes, unadopted, files seen, commits touching gated paths)
    """
    failures: Failures = []
    escapes: List[Tuple[str, str, str]] = []
    unadopted: List[Tuple[str, str]] = []
    range_files: set = set()
    gated_commits = 0
    for commit in commits:
        listed = git("diff-tree", "--no-commit-id", "--name-only", "-r", "--root", commit, cwd=root)
        files = [f for f in listed.splitlines() if f]
        range_files.update(files)
        message = git("log", "-1", "--format=%B", commit, cwd=root)
        subject = message.splitlines()[0] if message else ""
        read = _reader_at(root, commit)
        # Each commit is judged by the config it carries: a commit from before
        # adoption is listed, not failed.
        config, problems = parse_config(read(CONFIG))
        if problems:
            failures.append((commit, subject, problems))
            continue
        if config is None:
            unadopted.append((commit, subject))
            continue
        if not any(config.is_gated(f) for f in files):
            continue
        gated_commits += 1
        lines = _gated_lines(git("diff-tree", "--no-commit-id", "--numstat", "-r", "--root", commit, cwd=root), config)
        added = _added_lines(git("show", "--format=", "-U0", "--root", commit, cwd=root, check=False))
        parent = git("rev-parse", "--verify", "-q", f"{commit}^", cwd=root, check=False).strip()
        previous = _reader_at(root, parent) if parent else None
        evidence = gp.evidence_resolver(root, commit) if config.pillar_policy.mode != "off" else None
        errors, notes = check_change(files, lines, message, read, config, added, previous, evidence)
        if errors:
            failures.append((commit, subject, errors))
        escapes.extend((commit, subject, n) for n in notes)

    return failures, escapes, unadopted, range_files, gated_commits


def cmd_ci(base: str, head: str) -> int:
    root = repo_root()
    head_config, head_problems = parse_config(_reader_at(root, head)(CONFIG))
    if head_config is None or head_problems:
        why = "; ".join(head_problems) or (
            f"{CONFIG} is missing at {head[:12]} — this CI runs the process gate, so the repo adopted it, "
            "and a missing config means the gates were removed"
        )
        _report(["## Process gates", "", f"- ❌ {why}"], [f"::error title=Process gate::{why}"])
        return 1

    commits, caveat = _range_commits(root, base, head)
    failures, escapes, unadopted, range_files, gated_commits = check_commits(root, commits)

    changelog_error = None
    needing = sorted(f for f in range_files if head_config.needs_changelog(f))
    if needing and head_config.changelog_file not in range_files:
        changelog_error = (
            f"changed paths that need a {head_config.changelog_file} entry "
            f"({', '.join(needing[:6])}{' …' if len(needing) > 6 else ''}), and {head_config.changelog_file} "
            "was not updated in this range"
        )

    report = [
        "## Process gates",
        "",
        f"Checked {len(commits)} commit(s), {gated_commits} touching gated paths"
        f"{' — ' + caveat if caveat else ''}.",
    ]
    for commit, subject, errors in failures:
        report.append(f"- ❌ `{commit[:10]}` {subject}")
        report.extend(f"  - {e}" for e in errors)
    if changelog_error:
        report.append(f"- ❌ {changelog_error}")
    for commit, subject, note in escapes:
        report.append(f"- ⚠️ `{commit[:10]}` {subject} — {note}")
    for commit, subject in unadopted:
        report.append(f"- ℹ️ `{commit[:10]}` {subject} — before this repo adopted the gates; not checked")
    art_code, art_lines = artifacts_report(root, head_config)
    report.extend(f"- {'❌' if art_code else 'ℹ️'} {line}" for line in art_lines)
    if not failures and not changelog_error and not art_code:
        report.append("- ✅ every gated commit carries a resolving Design and a clean, same-commit Review")
    annotations = [f"::error title=Process gate {c[:10]}::{e}" for c, _s, errs in failures for e in errs]
    if changelog_error:
        annotations.append(f"::error title=Process gate::{changelog_error}")
    _report(report, annotations)
    return 1 if failures or changelog_error or art_code else 0





# ── records: single — one design artifact, one review log ────────────────────
#
# The rules do not change with the shape, so a section is normalised into what
# the legacy checkers already read: the fenced `governance` block becomes front
# matter and `### X` becomes `## X`. A second set of checkers would be a second
# set of rules (`no-copy-paste`).

DESIGN_ARTIFACT = "docs/DESIGN.md"
REVIEW_ARTIFACT = "docs/REVIEW_LOG.md"
ACTIVE_CHANGE = "Active change: "


def _slug_sections(text: str, prefix: str) -> Dict[str, str]:
    """`## <prefix><slug>` sections, by slug, in document order."""
    found: Dict[str, str] = {}
    for match in re.finditer(rf"^## {re.escape(prefix)}(\S+).*?$(.*?)(?=^## |\Z)", text, re.M | re.S):
        found.setdefault(match.group(1).strip(), match.group(2))
    return found


def design_section(text: str, slug: str) -> Optional[str]:
    """One `## Active change: <slug>` section, as a legacy design note."""
    body = _slug_sections(text, ACTIVE_CHANGE).get(slug)
    if body is None:
        return None
    fence = re.search(r"^```governance\s*$(.*?)^```\s*$", body, re.M | re.S)
    if fence is None:
        # No front matter: check_design says what is missing, in its own words.
        return re.sub(r"^### ", "## ", body, flags=re.M)
    rest = body[: fence.start()] + body[fence.end():]
    return "---\n" + fence.group(1).strip("\n") + "\n---\n" + re.sub(r"^### ", "## ", rest, flags=re.M)


def review_entries(text: str, slug: str) -> Optional[str]:
    """This change's `## <slug> — Pass N` and `## <slug> — Sign-off` entries, as
    a legacy review record. One log holds every change, so the slug is what
    keeps another change's clean pass from vouching for this one."""
    entries = re.findall(rf"^## {re.escape(slug)}\s+[—–-]\s+(.*?)$(.*?)(?=^## |\Z)", text, re.M | re.S)
    if not entries:
        return None
    return "\n".join(f"## {heading}\n{body}" for heading, body in entries)


def split_record_ref(value: str) -> Tuple[Optional[str], Optional[str]]:
    path, _, slug = value.strip().partition("#")
    return (path or None), (slug or None)


# ── One artifact per type, and cross-references ──────────────────────────────

ARTIFACT_MODES = ("off", "report", "enforce")

# A pointer into another document's numbering — `SPECS.md §23`, `DESIGN.md#L120`.  <!-- xref: example -->
# Section numbers move on the next edit of the document they point into; a
# heading name or the document alone does not.
_XREF = re.compile(r"[\w./-]+\.md\s*(?:§|#L)\s*[\w.]*\d")
# What a line that must SHOW a bad pointer carries. Greppable, so the
# exemptions can be counted.
_XREF_EXEMPT = "<!-- xref: example -->"


def _glob_match(path: str, pattern: str) -> bool:
    """Match a repo path against a registry glob. `PurePosixPath.match` handles
    a bare filename pattern (`*BACKLOG*.md` anywhere); fnmatch handles a rooted
    one (`docs/reference/*.md`, `.agent-rfc/**/*.md`)."""
    from fnmatch import fnmatch
    from pathlib import PurePosixPath

    try:
        if PurePosixPath(path).match(pattern):
            return True
    except ValueError:
        pass
    return fnmatch(path, pattern) or fnmatch(path, pattern.replace("**/", "*"))


def artifact_problems(root: Path, registry: "gm.Registry") -> List[str]:
    """Where this repo has more than one document of a type, a stray, or a gap.

    Reads what git tracks, not the filesystem: an untracked scratch file is
    nobody's record, and a document that is not committed governs nothing.
    """
    tracked = [f for f in git("ls-files", "*.md", cwd=root, check=False).splitlines() if f]
    artifacts = registry.artifacts
    canonical = {a.path: a for a in artifacts.types if a.path}
    problems: List[str] = []

    for artifact in artifacts.types:
        if artifact.path and artifact.required and artifact.path not in tracked:
            problems.append(f"{artifact.path} is missing — every repo keeps one {artifact.id}")

    for path in tracked:
        if path in canonical or any(_glob_match(path, g) for g in artifacts.reference + artifacts.ignored):
            continue
        matched = [a for a in artifacts.types for pattern in a.patterns if _glob_match(path, pattern)]
        if matched:
            governs = matched[0].path or f"this repo declares no {matched[0].id}"
            problems.append(f"{path} is a second {matched[0].id} — {governs} governs; fold it in and delete it")
        else:
            problems.append(
                f"{path} is neither an artifact nor declared reference documentation — "
                "fold it into the artifact that owns it, or declare it in `extends.artifacts.reference`"
            )
    return problems


def cross_reference_problems(added: List[str]) -> List[str]:
    """Pointers into another document's numbering, among the lines a change adds.

    Added lines only: a repo adopting the rule has pointers already, and failing
    all of them would block the very migrations that remove them (G5b).
    """
    problems = []
    for line in added:
        if _XREF_EXEMPT in line:
            continue
        found = _XREF.search(line)
        if found:
            problems.append(
                f"a new line points into another document's section numbers ({found.group(0).strip()}) — "
                "numbers move on the next edit; name the document, or a heading inside it. "
                f"An example that must show one carries {_XREF_EXEMPT}"
            )
    return problems


def artifacts_report(root: Path, config: Config, read: Optional[Reader] = None) -> Tuple[int, List[str]]:
    """-> (code, lines). `report` never fails; `enforce` fails on any problem."""
    if config.artifacts_mode not in ("report", "enforce"):
        return 0, []
    registry, registry_errors = config.load_registry(read or _reader_at(root, ""))
    if registry is None:
        return 1, [f"artifacts: {'; '.join(registry_errors)}"]
    found = artifact_problems(root, registry)
    if not found:
        return 0, []
    lines = [f"artifacts ({config.artifacts_mode}): {len(found)} problem(s)"]
    lines += [f"  - {problem}" for problem in found]
    return (1 if config.artifacts_mode == "enforce" else 0), lines


def cmd_artifacts() -> int:
    """Run by hand and by CI. It reads the WORKING TREE, not the index: someone
    asking "what does this repo look like now?" means the files in front of
    them. The sweep and commit-msg read the index, which is what those commits
    will contain. The verdict itself comes from `artifacts_report`, so this and
    the sweep can never disagree about what `report` means (`one-verdict`)."""
    root = repo_root()
    config, problems = _worktree_config(root)
    if config is None:
        print("artifacts: this repo has not adopted the process gates — nothing to check")
        return 0
    if problems:
        print(f"artifacts: {CONFIG} is unusable — " + "; ".join(problems))
        return 1
    if config.artifacts_mode not in ARTIFACT_MODES:
        print(f"artifacts: `artifacts` must be one of {', '.join(ARTIFACT_MODES)}, "
              f"not {config.artifacts_mode!r}")
        return 1
    if config.artifacts_mode == "off":
        print("artifacts: off for this repo — it has not declared a document layout yet "
              "(set `artifacts` to report or enforce in " + CONFIG + ")")
        return 0

    code, lines = artifacts_report(root, config, _worktree_reader(root))
    if not lines:
        print(f"artifacts: one file per type, no strays ({config.artifacts_mode})")
        return code
    print("\n".join(lines))
    if config.artifacts_mode == "report":
        print("  reported, not blocked: this repo is in `report` until its documents are consolidated")
    return code


def cmd_pillars() -> int:
    """What this repo owns today, for the pillars a script can check.

    The commit gate checks the files a change touches; this checks everything
    tracked, which is the question someone asking about the repo is asking —
    and the list to fix or to seed the allowlist from at adoption. Both call the
    same checks, so they cannot disagree about what a rule means.
    """
    root = repo_root()
    config, problems = _worktree_config(root)
    if config is None:
        print("pillars: this repo has not adopted the process gates — nothing to check")
        return 0
    if problems:
        print(f"pillars: {CONFIG} is unusable — " + "; ".join(problems))
        return 1
    policy = config.pillar_policy
    if policy.mode == "off":
        print("pillars: off for this repo — it is not held to the mechanical checks yet "
              f"(set `pillars` to report or enforce in {CONFIG})")
        return 0
    registry, registry_errors = config.load_registry(_worktree_reader(root))
    if registry is None:
        print("pillars: " + "; ".join(registry_errors))
        return 1
    active = gp.active_checks(registry)
    if not active:
        print(f"pillars: {config.display(config.registry)} marks no pillar `mechanical` — "
              "nothing was checked, which is not the same as passing")
        return 0
    found = gp.repo_problems(root, registry, policy)
    # Both kinds of exemption, counted: an allowlist entry the owner approved,
    # and a line that says it must hold a credential-shaped string. A number
    # that is printed is a number someone can watch grow.
    exempt = f"{len(policy.allow)} allowlisted, {gp.exemption_count(root)} line marker(s)"
    if not found:
        print(f"pillars: every tracked file passes {', '.join(sorted(active))} "
              f"({policy.mode}, {exempt})")
        return 0
    print(f"pillars ({policy.mode}): {len(found)} problem(s) across what this repo tracks ({exempt})")
    print("\n".join(f"  - {problem}" for problem in found))
    if policy.mode == "report":
        print("  reported, not blocked: this repo is in `report`")
    else:
        print("  a commit is refused only for the files it touches — fix them, or have the owner "
              "approve an allowlist entry for each")
    return 1 if policy.mode == "enforce" else 0


# ── The sweep ────────────────────────────────────────────────────────────────
#
# The commit gate is skippable: `--no-verify`, an unarmed clone, a rebase, a
# cherry-pick, or git run from a shell no IDE gates. Branch protection would
# catch it on the way out, and the owner ruled out depending on a GitHub plan
# (D3, approved 2026-09-15). So the check runs again, locally, at every
# touchpoint: pre-commit, pre-push, session start and stop. A commit that
# slipped past is found at the next thing anyone does in the repo.
#
# `.git/agentsmith/verified` remembers what has already been checked, so a
# sweep costs one pass over what is new. It lives in .git — it is this
# machine's record of what it verified, not shared history.

VERIFIED_CAP = 5000
# How many unverified commits one sweep checks. Overridable so the batching
# itself is testable without making 200 commits.
SWEEP_BATCH = int(os.environ.get("AGENTSMITH_SWEEP_BATCH") or 200)
VERIFIED_REL = "agentsmith/verified"


def _git_dir(root: Path) -> Path:
    out = git("rev-parse", "--git-dir", cwd=root, check=False).strip() or ".git"
    path = Path(out)
    return path if path.is_absolute() else root / path


def load_verified(root: Path) -> dict:
    """What this machine has already checked. A missing or unreadable store
    means "never swept", which initialises rather than re-checking history."""
    try:
        data = json.loads((_git_dir(root) / VERIFIED_REL).read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("shas"), list):
            return {"version": 1, "shas": [str(s) for s in data["shas"]], "initialised": True}
    except (OSError, json.JSONDecodeError, ValueError):
        pass
    return {"version": 1, "shas": [], "initialised": False}


def save_verified(root: Path, shas: List[str]) -> int:
    """Keep the newest VERIFIED_CAP; -> how many were dropped, never silently."""
    dropped = max(0, len(shas) - VERIFIED_CAP)
    path = _git_dir(root) / VERIFIED_REL
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": 1, "shas": shas[-VERIFIED_CAP:], "updated_at": _now()}
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return dropped


def _now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def local_commits(root: Path, limit: int = 2000) -> List[str]:
    """Every commit reachable from a local branch or HEAD, oldest first. Remote
    refs are not swept: what someone else pushed is their CI's business, and a
    fetch would otherwise make this machine responsible for their history."""
    out = git("rev-list", "--reverse", "--no-merges", f"--max-count={limit}",
              "--branches", "HEAD", cwd=root, check=False)
    return [c for c in out.splitlines() if c]


_REPAIRS = re.compile(r"^Repairs:[ \t]*(.+?)[ \t]*$", re.M | re.I)


def repairs_claimed(root: Path, commit: str) -> List[str]:
    """Full shas this commit says it repairs, as resolved by git. A trailer
    naming something git does not have repairs nothing — otherwise any commit
    could clear any finding by claiming it."""
    message = git("log", "-1", "--format=%B", commit, cwd=root, check=False)
    claimed: List[str] = []
    for line in _REPAIRS.findall(message):
        for token in re.split(r"[\s,]+", line):
            if not token:
                continue
            resolved = git("rev-parse", "--verify", "--quiet", f"{token}^{{commit}}", cwd=root, check=False).strip()
            if resolved:
                claimed.append(resolved)
    return claimed


def rearm_hooks(root: Path, config: Optional[Config]) -> Optional[str]:
    """An unarmed clone is one of the ways a commit skips the gate. Where the
    repo has adopted the gates and carries the launcher, re-arm it — and say so.
    A repo that never adopted them is left alone: git's machine-wide template
    applies these hooks to every `git init` on the machine."""
    if config is None or not (root / ".githooks" / "process-gate").is_file():
        return None
    current = git("config", "core.hooksPath", cwd=root, check=False).strip()
    if current == ".githooks":
        return None
    git("config", "core.hooksPath", ".githooks", cwd=root, check=False)
    return f"re-armed core.hooksPath = .githooks (was {current or 'unset'})"


def sweep(root: Path) -> Tuple[int, List[str], Failures]:
    """Re-check everything local that has not been verified.
    -> (code, report, the commits still failing)."""
    config, problems = parse_config(_reader_at(root, "")(CONFIG))
    report: List[str] = []
    if config is None and not problems:
        return 0, ["sweep: this repo has not adopted the process gates — nothing to check"], []
    armed = rearm_hooks(root, config)
    if armed:
        report.append(f"sweep: {armed}")
    if problems:
        return 1, [*report, f"sweep: {CONFIG} is unusable — " + "; ".join(problems)], []

    art_code, art_lines = artifacts_report(root, config)
    report.extend(art_lines)

    store = load_verified(root)
    commits = local_commits(root)
    if not store["initialised"]:
        dropped = save_verified(root, commits)
        report.append(
            f"sweep: initialised — {len(commits)} commit(s) of existing history recorded as the starting point "
            f"and NOT swept; everything from here on is checked"
            + (f" ({dropped} older sha(s) beyond the cap were not recorded)" if dropped else "")
        )
        return art_code, report, []

    known = set(store["shas"])
    candidates = [c for c in commits if c not in known]
    deferred = 0
    if len(candidates) > SWEEP_BATCH:
        # This runs at every commit, push, session start and turn end, and each
        # commit costs several git calls. Fetching a long branch must not turn
        # the next session start into a minute of silence: take the oldest
        # batch, say how many are left, and take the rest next time. Nothing is
        # skipped — an unchecked commit stays unverified.
        deferred = len(candidates) - SWEEP_BATCH
        candidates = candidates[:SWEEP_BATCH]
    if not candidates:
        report.append("sweep: nothing new since the last sweep")
        return art_code, report, []

    failures, _escapes, unadopted, _files, gated = check_commits(root, candidates)
    failed = {commit for commit, _subject, _errors in failures}
    passed = [c for c in candidates if c not in failed]
    repaired = {sha for c in passed for sha in repairs_claimed(root, c)} & failed
    still_failing = [(c, s, e) for c, s, e in failures if c not in repaired]

    verified = store["shas"] + [c for c in candidates if c not in failed or c in repaired]
    dropped = save_verified(root, verified)

    report.append(f"sweep: checked {len(candidates)} new commit(s), {gated} touching gated paths"
                  + (f"; {deferred} more will be swept next time" if deferred else ""))
    if dropped:
        report.append(f"  ℹ️ the verified record keeps the newest {VERIFIED_CAP} commits; {dropped} older sha(s) "
                      "were dropped and would be re-checked if they are still reachable")
    for commit, subject, _errors in failures:
        if commit in repaired:
            report.append(f"  ✅ {commit[:10]} {subject} — repaired by a later commit")
    for commit, subject in unadopted:
        report.append(f"  ℹ️ {commit[:10]} {subject} — before this repo adopted the gates; not checked")
    for commit, subject, errors in still_failing:
        report.append(f"  ❌ {commit[:10]} {subject} — this commit did not pass the gate:")
        report.extend(f"       {e}" for e in errors)
    if still_failing:
        report.append(
            "  Repair, never rewrite: commit the design and review records that cover those changes, with a "
            "`Repairs: <sha>` trailer naming each commit above (`agentsmith gates repair` lists them). "
            "Until then a commit must repair them, and pushes are refused."
        )
        return 1, report, still_failing
    return art_code, report, []


def cmd_sweep(report_only: bool = False) -> int:
    """`--report` is what pre-commit runs: the commit being made may be the
    repair, and its message — the only place that says so — does not exist yet.
    commit-msg makes that call; this re-arms the hooks and says what is pending."""
    code, report, _failures = sweep(repo_root())
    print("\n".join(report))
    return 0 if report_only else code


_SPAN_EVENTS = {"session-start": "session_start", "pre-edit": "pre_edit", "stop": "stop",
                "commit-msg": "commit_msg", "ci": "ci", "sweep": "sweep", "artifacts": "artifacts",
                "pillars": "pillars"}


def _traced(command: str, root: Path, run: Callable[[], int]) -> int:
    """Run one subcommand inside its span; the exit code and whether it
    denied or blocked are recorded from what it actually printed."""
    import contextlib
    import io

    captured = io.StringIO()
    ide = os.environ.get("AGENTSMITH_IDE") or "unknown"
    with gt.gate_span(_SPAN_EVENTS[command], root=root, ide=ide) as span:
        with contextlib.redirect_stdout(captured):
            code = run()
        out = captured.getvalue()
        decision = _DECISION["value"] or ("block" if code != 0 else "allow")
        span.set_attribute("agent.decision", decision)
        if _DECISION["rule"]:
            span.set_attribute("agent.rule", _DECISION["rule"])
        span.set_attribute("agent.exit_code", code)
    sys.stdout.write(out)
    if command in ("session-start", "stop", "ci"):
        gt.flush()
        if command != "session-start":
            gt.ship()
    else:
        gt.flush()
    return code


def main(argv: Optional[List[str]] = None) -> int:
    if UNUSABLE:
        print(f"process gate cannot run here: {UNUSABLE}. It needs the framework environment "
              "(Python 3.11+ with pydantic and opentelemetry) — run AgentSmith's install-ai-stack.sh, "
              "or point AGENTSMITH_PYTHON at an interpreter that has them.", file=sys.stderr)
        return EXIT_UNUSABLE
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("artifacts", "pillars"):
        sub.add_parser(name)
    for name in ("session-start", "pre-edit", "stop"):
        hook = sub.add_parser(name)
        hook.add_argument("--ide", default=None, choices=gi.IDES,
                          help="which IDE is asking (default: $AGENTSMITH_IDE, then claude)")
    sweep_cmd = sub.add_parser("sweep")
    sweep_cmd.add_argument("--report", action="store_true",
                           help="report and re-arm, but do not refuse (pre-commit; commit-msg decides)")
    msg = sub.add_parser("commit-msg")
    msg.add_argument("--amend", action="store_true", help="the commit replaces HEAD (git commit --amend)")
    msg.add_argument("message_file")
    ci = sub.add_parser("ci")
    ci.add_argument("--base", default="")
    ci.add_argument("--head", default="HEAD")
    args = parser.parse_args(argv)

    if args.command == "artifacts":
        return _traced("artifacts", repo_root(), cmd_artifacts)
    if args.command == "pillars":
        return _traced("pillars", repo_root(), cmd_pillars)
    if args.command in ("session-start", "pre-edit", "stop"):
        try:
            _IDE["value"] = gi.resolve(args.ide, os.environ)
        except ValueError as exc:
            print(f"process gate: {exc}", file=sys.stderr)
            return 1
        raw = sys.stdin.read() if not sys.stdin.isatty() else ""
        handler = {"session-start": cmd_session_start, "pre-edit": cmd_pre_edit, "stop": cmd_stop}[args.command]
        try:
            payload = json.loads(raw) if raw.strip() else {}
            return _traced(args.command, repo_root(payload.get("cwd")), lambda: handler(payload))
        except Exception as exc:  # a hook must answer, whatever broke
            if args.command != "pre-edit":
                print(f"process gate {args.command} could not run: {exc!r}", file=sys.stderr)
                return 0
            # Fail CLOSED. Claude Code treats a crashed PreToolUse hook as a
            # non-blocking error and lets the edit through — so a gate that
            # cannot evaluate would silently stop gating.
            _deny(f"process gate could not evaluate this edit ({exc!r}) — "
                  "fix process_gate.py or its input before editing gated paths")
            return 0
    if args.command == "sweep":
        return _traced("sweep", repo_root(), lambda: cmd_sweep(args.report))
    if args.command == "commit-msg":
        return _traced("commit-msg", repo_root(), lambda: cmd_commit_msg(args.message_file, amend=args.amend))
    return _traced("ci", repo_root(), lambda: cmd_ci(args.base, args.head))


if __name__ == "__main__":
    sys.exit(main())
