"""
generate-ide-config.py — generates .cursorrules, CLAUDE.md, and Antigravity
skill files from templates/agent-rules.yaml, the single source of truth
(docs/DESIGN.md › Ten Operational Pillars, Antigravity Integration).

Called by the post-checkout hook instead of embedding IDE-rule content
inline — editing templates/agent-rules.yaml is now the only way to change
what gets written to .cursorrules / CLAUDE.md / .agents/skills/*/skill.md.

Idempotent: never overwrites an existing .cursorrules, CLAUDE.md, or
skill.md (same behavior as the hook's previous `[ ! -f ... ]` guards).

Usage:
    python3 generate-ide-config.py \\
        --repo-root /path/to/repo \\
        --rules-file /path/to/templates/agent-rules.yaml \\
        --stack ts-react|python-fastapi|go|generic \\
        --project-name myproject --owner-id me@example.com \\
        --otel-endpoint http://localhost:6006 --test-cmd "pytest" \\
        --framework-version 1.0.0
"""

from __future__ import annotations

import argparse
import difflib
import json
import sys
from pathlib import Path


def _load_rules(rules_file: Path) -> dict:
    try:
        import yaml  # type: ignore
    except ImportError:
        print(
            f"❌ generate-ide-config.py requires pyyaml, which {sys.executable} lacks. "
            "The git hooks run it with ~/.agent-framework/.venv/bin/python — re-run "
            "install-ai-stack.sh to build that environment. IDE config was NOT regenerated.",
            file=sys.stderr,
        )
        sys.exit(1)
    with rules_file.open() as fh:
        return yaml.safe_load(fh) or {}


def _playbook_skills(rules: dict) -> list[dict]:
    """Skills shaped as a doc pointer (`design_review`, `validation_review`)
    rather than a pillar compression — distinguished by carrying `doc`, which
    a pillars-shaped skill never does. Kept as one function so every renderer
    that needs this list gets it the same way (`parameterize-dont-clone`)."""
    return [s for s in rules.get("skills", []) if s.get("doc")]


def _playbook_location(doc: str) -> str:
    """Where a generated file should tell the reader to find `doc`.

    `doc` is a path under the FRAMEWORK's own docs/ — not vendored into a
    tenant repo or written by the post-checkout hook the way .cursorrules /
    CLAUDE.md / etc. are. Only scripts/, templates/, and (as of this pair)
    these two named files reach ~/.agent-framework — see install-ai-stack.sh.
    So there is no single resolvable path; there are two, and the reader
    picks whichever matches how AgentSmith got onto this machine.
    """
    filename = doc.rsplit("/", 1)[-1]
    return (
        f"`$AGENTSMITH_DIR/{doc}` if you're on a live framework checkout, "
        f"or `~/.agent-framework/docs/{filename}` if AgentSmith was installed "
        f"from the package"
    )


PROCESS_GATES_CONFIG = ".agenticframework/process-gates.json"


def _repo_extends(repo_root: Path) -> dict:
    """A repo's own `extends` block. Its notes belong in the generated rule
    files, not hand-edited into them: a hand edit drifts, and the drift check
    then reports every regeneration as a conflict."""
    config = repo_root / PROCESS_GATES_CONFIG
    if not config.is_file():
        return {}
    try:
        return json.loads(config.read_text(encoding="utf-8")).get("extends") or {}
    except (json.JSONDecodeError, OSError):
        return {}


def _tenant_declaration(repo_root: Path) -> dict:
    """`tenant:` and `framework:` from .agenticframework/tenant.yaml.

    Without this the defaults were a git-remote name and "unknown@unknown", so
    CI's drift check generated different files from a developer's run and the
    check could only pass by committing the placeholder. The repo already
    declares all three; read them (`single-source-of-truth`). Parsed with a
    two-level scan rather than pyyaml — this runs where pyyaml may be absent,
    and the file is written by `agentsmith tenant init`, not by hand.
    """
    config = repo_root / ".agenticframework" / "tenant.yaml"
    if not config.is_file():
        return {}
    values: dict = {}
    section = None
    for line in config.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("#") or not line.strip():
            continue
        if not line.startswith((" ", "\t")):
            section = line.split(":", 1)[0].strip()
            continue
        if section in ("tenant", "framework") and ":" in line:
            key, _, value = line.strip().partition(":")
            value = value.strip().strip('"\'')
            if value:
                values[f"{section}.{key.strip()}"] = value
    return values


def _repo_notes_block(ctx: dict) -> str:
    notes = ctx.get("rules_extra") or []
    if not notes:
        return ""
    return "\n".join(["## This repository", "", *[f"- {note}" for note in notes], ""])


def _design_start_lines(rules: dict) -> list[str]:
    """The ask-the-owner block, from agent-rules.yaml `records.design_start`.

    Every renderer surfaces it: a rule that only Claude Code's file carries is a
    rule Cursor, Antigravity, Copilot, Gemini and Codex never see, and the
    owner asked for permission-to-deviate in ANY IDE.
    """
    records = rules.get("records") or {}
    return [" ".join(str(line).split()) for line in records.get("design_start", [])]


DESIGN_START_HEADING = "## Design start — before you write code"


def _design_start_block(rules: dict, heading: str = DESIGN_START_HEADING) -> str:
    lines = _design_start_lines(rules)
    if not lines:
        return ""
    bullets = [f"- {line}" for line in lines]
    return "\n".join([heading, "", *bullets, ""])


def render_registry(rules: dict) -> str:
    """templates/governance.json — the rules the process gate enforces, as JSON.

    The hooks run without pyyaml, so they cannot read agent-rules.yaml; a
    hand-kept JSON beside it would be a second rule set. This compiles the one
    source instead, and --registry --check-only fails when the two differ.
    """
    registry = {
        "_about": "Generated from templates/agent-rules.yaml by scripts/generate-ide-config.py --registry. "
        "Do not edit; read by scripts/process_gate.py (.agent-rfc/designs/governance-enforcement.md).",
        "version": str((rules.get("meta") or {}).get("version", "0")),
        "pillars": [
            {
                "id": p["id"],
                "name": p["name"],
                "check": list(p.get("check") or []),
                "design_question": p.get("design_question"),
                "rule": " ".join(str(p.get("rule", "")).split()),
            }
            for p in rules.get("pillars", [])
        ],
        "records": rules.get("records") or {},
        "artifacts": rules.get("artifacts") or {},
        "ides": rules.get("ides") or [],
    }
    return json.dumps(registry, indent=2, ensure_ascii=False) + "\n"


def _registry_mode(rules_file: Path, rules: dict, check_only: bool) -> int:
    target = rules_file.with_name("governance.json")
    expected = render_registry(rules)
    if check_only:
        actual = target.read_text(encoding="utf-8") if target.exists() else None
        if actual == expected:
            print(f"✅ {target.name} matches {rules_file.name}")
            return 0
        state = 'is missing' if actual is None else 'has drifted'
        print(f"❌ {target} {state} — run generate-ide-config.py --registry")
        if actual is not None:
            sys.stdout.writelines(difflib.unified_diff(
                actual.splitlines(keepends=True), expected.splitlines(keepends=True),
                fromfile=f"committed/{target.name}", tofile=f"generated/{target.name}",
            ))
        return 1
    target.write_text(expected, encoding="utf-8")
    print(f"✅ Written {target} from {rules_file.name}")
    return 0


def _gates_mode(repo_root: Path, check_only: bool) -> int:
    """The gates table in docs/validation-checklist.md, from the workflow tags.

    Step 3 of that checklist used to carry four commands somebody typed out, in
    a repo whose CI runs thirty steps — the duplicate `run-the-gates-ci-lists`
    exists to stop. It is generated between markers now, from the one place the
    gates are declared (`pin-unremovable-duplicates`).
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import gate_steps as gs

    target = repo_root / "docs" / "validation-checklist.md"
    text = target.read_text(encoding="utf-8")
    if gs.BEGIN not in text or gs.END not in text:
        print(f"❌ {target} has no {gs.BEGIN} / {gs.END} markers to write the gates table between")
        return 1
    head, rest = text.split(gs.BEGIN, 1)
    _old, tail = rest.split(gs.END, 1)
    expected = head + gs.BEGIN + "\n" + gs.render_table(gs.gates(repo_root)) + gs.END + tail
    if check_only:
        if text == expected:
            print("✅ validation-checklist.md gates table matches the workflow tags")
            return 0
        print(f"❌ {target} has drifted from the `{gs.TAG}` tags — "
              "run generate-ide-config.py --gates")
        sys.stdout.writelines(difflib.unified_diff(
            text.splitlines(keepends=True), expected.splitlines(keepends=True),
            fromfile="committed/validation-checklist.md", tofile="generated/validation-checklist.md",
        ))
        return 1
    target.write_text(expected, encoding="utf-8")
    print(f"✅ Written the gates table in {target} from the workflow tags")
    return 0


def _hooks_mode(repo_root: Path, check_only: bool) -> int:
    """The IDE hook configs, from the registry's `ides`.

    Written only where the config schema is verified — a file in a shape nobody
    has confirmed looks like enforcement and may be ignored in silence. The
    adapters read and answer all six dialects either way.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import gate_ides as gi

    # The tenant's declared choice, so this generator agrees with `tenant init`,
    # `tenant adopt` and `agentsmith sync` instead of reporting the IDE they
    # deliberately did not write as "missing".
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    try:
        from runtime.config import chosen_ides

        wanted = chosen_ides(repo_root, gi.GENERATED)
    except Exception:  # fail-open: a checkout without runtime/ still generates
        wanted = tuple(gi.GENERATED)

    problems = 0
    for ide in wanted:
        target = repo_root / gi.ADAPTERS[ide].config_path
        existing = json.loads(target.read_text(encoding="utf-8")) if target.is_file() else None
        expected = json.dumps(gi.render_config(ide, existing), indent=2, ensure_ascii=False) + "\n"
        actual = target.read_text(encoding="utf-8") if target.is_file() else None
        if check_only:
            if actual == expected:
                print(f"✅ {gi.ADAPTERS[ide].config_path} matches the registry")
                continue
            problems += 1
            print(f"❌ {gi.ADAPTERS[ide].config_path} "
                  f"{'is missing' if actual is None else 'has drifted'} — run generate-ide-config.py --hooks")
            if actual is not None:
                sys.stdout.writelines(difflib.unified_diff(
                    actual.splitlines(keepends=True), expected.splitlines(keepends=True),
                    fromfile=f"committed/{ide}", tofile=f"generated/{ide}"))
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(expected, encoding="utf-8")
        print(f"✅ Written {gi.ADAPTERS[ide].config_path} from the registry")
    not_generated = [i for i in gi.IDES if i not in gi.GENERATED]
    print(f"ℹ️  Not generated (no verified config schema yet, the adapter still reads them): "
          f"{', '.join(not_generated)}")
    return 1 if problems else 0


def _render_cursorrules(rules: dict, stack: str, ctx: dict[str, str]) -> str:
    lines = [
        "# AgentSmith — Agent Guardrails",
        f"# Project: {ctx['project_name']} | Owner: {ctx['owner_id']}",
        f"# Auto-generated by AgentSmith v{ctx['framework_version']} "
        f"from templates/agent-rules.yaml — do not edit this file directly.",
        "",
    ]

    pillar_titles = {
        1: "REQUIREMENTS & DESIGN",
        2: "BUILD ARCHITECTURE (Ponytail Rules)",
        3: "TRACING & EVALUATIONS",
        4: "TESTING GUARDRAILS",
        5: "OPERATIONS & SELF-IMPROVEMENT",
        6: "INTERFACE CONSTRAINTS (Caveman Compression)",
        7: "STACK-SPECIFIC RULES",
        8: "OBSERVABILITY WIRE",
        9: "MULTI-AGENT ORCHESTRATION",
        10: "COST-OPTIMISATION ROUTING",
    }

    for pillar in rules.get("pillars", []):
        pid = pillar["id"]
        title = pillar_titles.get(pid, pillar["name"].upper())
        rule_text = pillar["rule"].strip()
        # Substitute the same placeholders the old inline heredocs used.
        rule_text = (
            rule_text.replace("$AGENT_PHOENIX_ENDPOINT", ctx["otel_endpoint"])
            .replace("{{OWNER_ID}}", ctx["owner_id"])
            .replace("{{PROJECT_NAME}}", ctx["project_name"])
        )
        lines.append(f"## {pid}. {title}")
        lines.append(f"- {rule_text}")
        if pid == 3:
            lines.append(
                f"- OTel endpoint for this project: {ctx['otel_endpoint']}/v1/traces"
            )
        if pid == 4:
            lines.append(f"- Test command for this project: {ctx['test_cmd']}")
        lines.append("")

    block = _design_start_block(rules, "## DESIGN START — BEFORE YOU WRITE CODE")
    if block:
        lines.append(block)
    notes = _repo_notes_block(ctx)
    if notes:
        lines.append(notes.replace("## This repository", "## THIS REPOSITORY"))
    stack_def = (rules.get("stacks") or {}).get(stack)
    next_no = len(rules.get("pillars", [])) + 1
    if stack_def:
        # Numbered after the last pillar, not hardcoded. This said "11" from
        # when there were ten pillars; adding four gave the file two sections
        # numbered 11 — the addendum and Untrusted Content — which is exactly
        # the kind of quiet inconsistency an agent reading the rules trips on.
        lines.append(f"## {next_no}. {stack.upper()} ADDENDUM")
        for r in stack_def.get("additional_rules", []):
            lines.append(f"- {r}")
        lines.append("")
        next_no += 1

    playbooks = _playbook_skills(rules)
    if playbooks:
        lines.append(f"## {next_no}. DESIGN & VALIDATION PLAYBOOKS")
        for pb in playbooks:
            lines.append(
                f"- **{pb['title']}** ({pb['trigger']}) — {_playbook_location(pb['doc'])}."
            )
        lines.append("")

    return "\n".join(lines)


def _render_claude_md(rules: dict, ctx: dict[str, str]) -> str:
    playbooks = _playbook_skills(rules)
    playbook_section = ""
    if playbooks:
        pb_lines = "\n".join(
            f"- **{pb['title']}** ({pb['trigger']}) — {_playbook_location(pb['doc'])}."
            for pb in playbooks
        )
        playbook_section = f"""
## Design & Validation Playbooks
{pb_lines}
"""
    design_start = _design_start_block(rules)
    return f"""# AgentSmith — Claude Code Instructions
# Project: {ctx["project_name"]} | Owner: {ctx["owner_id"]}
# Auto-generated from templates/agent-rules.yaml — do not edit this file directly.

## Compliance
Follow all rules in `.cursorrules` exactly. They are not suggestions.

{design_start}
{_repo_notes_block(ctx)}
## Session Start Checklist
1. Read `.agent-history.log` — surface any unresolved MAJOR/CRITICAL entries to the user.
2. Check `.agent-rfc/` — confirm a spec exists before touching any source file.
3. Scope the change with the Knowledge Graph: `agentsmith gate kg impact` — the files to read, the `KG query:` hash.

## Test Command
```
{ctx["test_cmd"]}
```

## Observability
- OTel endpoint: {ctx["otel_endpoint"]}/v1/traces
- Emit spans for every tool call, LLM invocation, and file write.
{playbook_section}
## Output Format
Terse by default — code, commands, data. Skip preambles and filler summaries.
Being terse is not being silent: state plainly what failed and why you think so,
any risk you are taking, and any assumption you had to make. The session-start
escalation above is worthless if it arrives as a code block nobody can read.
"""


def _render_agents_md(rules: dict, stack: str, ctx: dict[str, str]) -> str:
    """AGENTS.md — the convention Codex reads, and increasingly the cross-tool one.

    Deliberately self-contained rather than a pointer to `.cursorrules`. CLAUDE.md
    can say "follow .cursorrules" because Claude Code reliably opens neighbouring
    files on request; an agent that only ingests AGENTS.md at session start would
    get a rule file consisting of one redirect it may never follow. The cost is
    some duplication between two generated files, which is acceptable because both
    are generated — they cannot drift from each other, only from the YAML, and
    --check-only catches that.
    """
    pillars = "\n".join(
        f"{p['id']}. **{p['name']}** — {' '.join(str(p['rule']).split())}"
        for p in rules.get("pillars", [])
    )
    stack_def = (rules.get("stacks") or {}).get(stack) or {}
    addendum = "\n".join(f"- {r}" for r in stack_def.get("additional_rules", [])) or "- (none for this stack)"
    playbooks = _playbook_skills(rules)
    playbook_section = ""
    if playbooks:
        pb_lines = "\n".join(
            f"- **{pb['title']}** ({pb['trigger']}) — {_playbook_location(pb['doc'])}."
            for pb in playbooks
        )
        playbook_section = f"""
## Design & Validation Playbooks
{pb_lines}
"""
    return f"""# AGENTS.md — {ctx["project_name"]}
<!-- Owner: {ctx["owner_id"]} -->
<!-- Auto-generated from templates/agent-rules.yaml — do not edit this file directly. -->

These are the operating rules for any coding agent working in this repository.
They are not style preferences; each one exists because its absence caused a
real failure. Where a rule explains itself, that reasoning is the point — follow
the intent, not just the letter.

## Session start
1. Read `.agent-history.log` and surface unresolved MAJOR/CRITICAL entries.
2. Confirm a spec exists in `.agent-rfc/` before editing any source file.
3. Scope the change with the Knowledge Graph (`agentsmith gate kg impact`): the files to read and the `KG query:` hash.

## Rules
{pillars}

{_design_start_block(rules)}
{_repo_notes_block(ctx)}
## Stack ({stack})
{addendum}
{playbook_section}
## Test command
```
{ctx["test_cmd"]}
```

## Observability
Emit OTel spans for every tool call, LLM invocation and file write to
{ctx["otel_endpoint"]}/v1/traces.
"""


def _render_copilot_instructions(rules: dict, stack: str, ctx: dict[str, str]) -> str:
    """`.github/copilot-instructions.md` — repo-wide instructions for GitHub Copilot.

    Deliberately the SHORT one. Copilot prepends this to requests rather than
    reading it once at session start, so length is a recurring cost paid on every
    completion, and a fourteen-pillar essay would crowd out the code the model is
    supposed to be looking at. The pillars are compressed to their imperative —
    the reasoning that makes AGENTS.md worth reading is exactly what does not
    survive that budget.

    That trade means Copilot gets the rules but not the arguments, so this file
    points at AGENTS.md for anything a reader needs to weigh rather than obey.
    """
    lines = []
    for p in rules.get("pillars", []):
        # First sentence only: it carries the instruction, the rest carries the why.
        first = " ".join(str(p["rule"]).split()).split(". ")[0].rstrip(".")
        lines.append(f"- **{p['name']}**: {first}.")
    stack_def = (rules.get("stacks") or {}).get(stack) or {}
    stack_rules = "\n".join(f"- {r}" for r in stack_def.get("additional_rules", []))
    # One line, not a bullet per playbook — the budget argument above applies
    # here exactly as it does to the pillars: name that they exist and where,
    # leave the "why" and the trigger detail to AGENTS.md.
    playbook_names = ", ".join(pb["title"].split(" — ")[0] for pb in _playbook_skills(rules))
    playbook_line = (
        f"\nDesign/validation playbooks ({playbook_names}): see `AGENTS.md`.\n"
        if playbook_names
        else ""
    )
    return f"""<!-- Auto-generated from templates/agent-rules.yaml — do not edit directly. -->
# Copilot instructions — {ctx["project_name"]}

Follow these when suggesting or editing code in this repository. Full reasoning
for each rule is in `AGENTS.md`; this file is the condensed form Copilot sees on
every request.

{chr(10).join(lines)}

{_design_start_block(rules)}
{_repo_notes_block(ctx)}
## {stack} specifics
{stack_rules or "- (none for this stack)"}
{playbook_line}
Tests: `{ctx["test_cmd"]}`
"""


def _render_gemini_md(rules: dict, stack: str, ctx: dict[str, str]) -> str:
    """GEMINI.md — read by Gemini CLI at session start, like CLAUDE.md.

    Full-length, same as AGENTS.md: this is loaded once per session rather than
    per request, so the reasoning earns its tokens. Kept as its own renderer
    rather than aliasing AGENTS.md because the session-start contract differs —
    Gemini CLI resolves its tool names through this file, so the header says so
    explicitly rather than leaving a Claude-shaped instruction to be guessed at.
    """
    body = _render_agents_md(rules, stack, ctx)
    body = body.replace(
        f'# AGENTS.md — {ctx["project_name"]}',
        f'# GEMINI.md — {ctx["project_name"]}',
        1,
    )
    return body.replace(
        "These are the operating rules for any coding agent working in this repository.",
        "These are the operating rules for any coding agent working in this repository.\n"
        "Gemini CLI: tool names in these rules follow the Claude Code convention —\n"
        "map them to your equivalents rather than skipping the rule.",
        1,
    )


def _render_skill(skill: dict, rules: dict, ctx: dict[str, str]) -> str:
    lines = [
        f"# {skill['title']}",
        "## Trigger",
        skill["trigger"],
        "",
        "## Instructions",
    ]

    if skill.get("doc"):
        # Doc-pointer skill: the content lives in the referenced file, not
        # compressed from pillars — see the comment beside these two entries
        # in templates/agent-rules.yaml for why it isn't inlined here.
        summary = " ".join(str(skill.get("summary", "")).split())
        lines.append(f"- Before proceeding, read {_playbook_location(skill['doc'])}.")
        if summary:
            lines.append(f"- {summary}")
        lines.append(f"- Owner: {ctx['owner_id']} | Project: {ctx['project_name']}")
        return "\n".join(lines) + "\n"

    if 1 in skill.get("pillars", []):
        lines.append("- " + "\n- ".join(_design_start_lines(rules)) if _design_start_lines(rules) else "")
    pillar_by_id = {p["id"]: p for p in rules.get("pillars", [])}
    for pid in skill.get("pillars", []):
        pillar = pillar_by_id.get(pid)
        if not pillar:
            continue
        rule_text = (
            pillar["rule"]
            .strip()
            .replace("$AGENT_PHOENIX_ENDPOINT", ctx["otel_endpoint"])
        )
        lines.append(f"- ({pillar['name']}) {rule_text}")
    lines.append(f"- OTel endpoint: {ctx['otel_endpoint']}/v1/traces")
    lines.append(f"- Owner: {ctx['owner_id']} | Project: {ctx['project_name']}")
    return "\n".join(lines) + "\n"


def _detect_stack(repo_root: Path) -> tuple[str, str]:
    """(stack, default test command) from what the repository holds — the order
    `tenant adopt` and hooks/post-checkout use, and the lock file the tests run
    under (a uv project runs `uv run pytest`, a pnpm one `pnpm test`)."""
    if (repo_root / "package.json").exists():
        if (repo_root / "pnpm-lock.yaml").exists():
            return "ts-react", "CI=true pnpm test"
        return "ts-react", "CI=true npm test"
    if any((repo_root / name).exists() for name in ("pyproject.toml", "Pipfile", "setup.py")) \
            or list(repo_root.glob("requirements*.txt")):
        return "python-fastapi", "uv run pytest" if (repo_root / "uv.lock").exists() else "pytest"
    if (repo_root / "go.mod").exists():
        return "go", "go test -race ./..."
    return "generic", "echo 'No test command configured'"


DEFAULT_OTEL_ENDPOINT = "http://localhost:6006"


def declared_context(repo_root: Path) -> dict:
    """What the rule files say about THIS repository, from what it commits.

    contract/rules/v1: a render reads the repository's declarations and content
    and nothing else — not the environment, not the git remote — so CI and the
    developer who committed the files render the same bytes. It used to read
    AGENT_OWNER_ID, AGENT_PHOENIX_ENDPOINT and FRAMEWORK_VERSION, and a runner
    with one of them set reported drift nobody had made.
    """
    extends = _repo_extends(repo_root)
    declared = _tenant_declaration(repo_root)
    detected_stack, detected_test_cmd = _detect_stack(repo_root)
    return {
        "stack": detected_stack,
        "rules_extra": extends.get("rules_extra") or [],
        "project_name": declared.get("tenant.name") or declared.get("tenant.id") or "unnamed",
        "owner_id": declared.get("tenant.owner") or "unknown@unknown",
        "otel_endpoint": extends.get("otel_endpoint") or DEFAULT_OTEL_ENDPOINT,
        "test_cmd": extends.get("test_command") or detected_test_cmd,
        "framework_version": declared.get("framework.version") or "1.0.0",
    }


HISTORY_SEED = (
    "# .agent-history.log — append-only record of agent sessions.\n"
    "# Read at session start (pillar 5): unresolved MAJOR/CRITICAL entries\n"
    "# are things a previous session could not finish, and repeating them\n"
    "# is the failure this log exists to prevent.\n"
    "# Written by scripts/agent_logger.py; safe to read, do not rewrite.\n"
)


def seed_history(repo_root: Path) -> bool:
    """Create .agent-history.log if it is missing. Pillar 5 tells every agent to
    read it on session start; a fresh repo had none, so the rule pointed at
    nothing — and a missing file reads as "no history" rather than "not wired
    up yet". Tenant data, not a rule file: never overwritten, never rendered."""
    path = repo_root / ".agent-history.log"
    if path.exists():
        return False
    path.write_text(HISTORY_SEED)
    return True


def render_files(repo_root: Path, rules: dict, ctx: dict) -> list[tuple[str, str, str]]:
    """Every rule file, as `(path, text, kind)`: the five instruction files an
    IDE reads at session start, then the skill files. One list for the default
    mode, --check-only and the rules port (scripts/rules_port.py), so there is
    one renderer."""
    stack = ctx["stack"]
    files = [
        (".cursorrules", _render_cursorrules(rules, stack, ctx), "instructions"),
        ("CLAUDE.md", _render_claude_md(rules, ctx), "instructions"),
        ("AGENTS.md", _render_agents_md(rules, stack, ctx), "instructions"),
        ("GEMINI.md", _render_gemini_md(rules, stack, ctx), "instructions"),
        (".github/copilot-instructions.md", _render_copilot_instructions(rules, stack, ctx), "instructions"),
    ]
    files += [(f".agents/skills/{skill['id']}/skill.md", _render_skill(skill, rules, ctx), "supporting")
              for skill in rules.get("skills", [])]
    return files


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--rules-file", default=None)
    ap.add_argument("--stack", default=None)
    ap.add_argument("--project-name", default=None)
    ap.add_argument("--owner-id", default=None)
    ap.add_argument("--otel-endpoint", default=None)
    ap.add_argument("--test-cmd", default=None)
    ap.add_argument("--framework-version", default=None)
    ap.add_argument(
        "--write",
        action="store_true",
        help="Regenerate the rule files even where they exist. The default never "
        "overwrites (first provisioning); a governed repo that tracks them regenerates with this.",
    )
    ap.add_argument(
        "--check-only",
        action="store_true",
        help="Don't write files — regenerate in memory and diff against what's "
        "already committed. Exits 1 if .cursorrules/CLAUDE.md/skill.md have "
        "drifted from templates/agent-rules.yaml (Pillar 6/7 CI gate).",
    )
    ap.add_argument(
        "--hooks",
        action="store_true",
        help="Regenerate the IDE hook configs from governance.json's `ides`; with --check-only, "
        "exit 1 when a committed one has drifted.",
    )
    ap.add_argument(
        "--gates",
        action="store_true",
        help="Regenerate the gates table in docs/validation-checklist.md from the "
        "`# agentsmith:gate` tags in .github/workflows; with --check-only, exit 1 when it has drifted.",
    )
    ap.add_argument(
        "--registry",
        action="store_true",
        help="Compile agent-rules.yaml into governance.json beside it (the process gate's registry); "
        "with --check-only, exit 1 when the committed one differs.",
    )
    args = ap.parse_args()

    repo_root = Path(args.repo_root)

    rules_file = (
        Path(args.rules_file)
        if args.rules_file
        else repo_root / "templates" / "agent-rules.yaml"
    )
    if not rules_file.exists():
        if args.check_only:
            print(f"ℹ️  No rules file at {rules_file} — IDE config drift check skipped.")
            sys.exit(0)
        print(
            f"❌ Rules file not found: {rules_file}. IDE config was NOT regenerated.",
            file=sys.stderr,
        )
        sys.exit(1)

    rules = _load_rules(rules_file)
    if args.hooks:
        sys.exit(_hooks_mode(Path(args.repo_root).resolve(), args.check_only))
    if args.gates:
        sys.exit(_gates_mode(Path(args.repo_root).resolve(), args.check_only))
    if args.registry:
        sys.exit(_registry_mode(rules_file, rules, args.check_only))
    # What the repository declares, overridden only by what the caller passed
    # explicitly (hooks/post-checkout provisions a fresh clone that way).
    ctx = declared_context(repo_root)
    for key, value in (("stack", args.stack), ("project_name", args.project_name), ("owner_id", args.owner_id),
                       ("otel_endpoint", args.otel_endpoint), ("test_cmd", args.test_cmd),
                       ("framework_version", args.framework_version)):
        if value:
            ctx[key] = value
    stack = ctx["stack"]

    if args.check_only:
        sys.exit(0 if _check_drift(repo_root, rules, ctx) else 1)

    for rel, text, kind in render_files(repo_root, rules, ctx):
        target = repo_root / rel
        if args.write or not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text)
            if kind == "instructions":
                print(f"✅ Written {rel} ({stack}) from agent-rules.yaml")

    if seed_history(repo_root):
        print("✅ Seeded .agent-history.log")

    if rules.get("skills"):
        print(
            f"✅ Written Antigravity skill files ({', '.join(s['id'] for s in rules['skills'])}) from agent-rules.yaml"
        )


def _check_drift(repo_root: Path, rules: dict, ctx: dict) -> bool:
    """Print a diff for any committed IDE config file that no longer matches
    what agent-rules.yaml would generate. Returns True if clean (no drift)."""
    clean = True
    for rel, expected, _kind in render_files(repo_root, rules, ctx):
        path = repo_root / rel
        if not path.exists():
            print(f"ℹ️  {rel} not generated yet — skipping drift check.")
            continue
        actual = path.read_text()
        if actual == expected:
            print(f"✅ {rel} matches agent-rules.yaml")
            continue
        clean = False
        print(f"❌ {rel} has drifted from agent-rules.yaml:")
        sys.stdout.writelines(difflib.unified_diff(
            actual.splitlines(keepends=True),
            expected.splitlines(keepends=True),
            fromfile=f"committed/{path.name}",
            tofile=f"generated/{path.name}",
        ))
    return clean


if __name__ == "__main__":
    main()
