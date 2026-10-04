"""
runtime/adopt.py — `agentsmith tenant adopt`: an existing repository comes
under the gates without losing what it has (.agent-rfc/designs/tenant-adopt.md).

    plan_adoption(...)        what is there and what adopt would do; writes nothing
    adopt(plan)               does it; returns every path it wrote or changed
    commit_command(written)   the adoption commit, staging those paths by name
    prior_hooks_dir(root)     the hooks directory the repository ran before the gates
    chain_hooks(root, prior)  keep those hooks running behind the gates (also `tenant init`)
    merge_rules_block(...)    generated agent rules inside a file the repository already had

`tenant init` decides a layout for an empty repository; adopt has to find the
one a repository already has, and add the gates around it: merge, never skip,
never overwrite, and leave its CI alone.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

# The hooks the gate owns, copied from the framework's .githooks; every other
# hook the prior directory has gets a stub that runs it through `chain`.
from runtime.cli import GATE_HOOKS

ADOPTION_DESIGN = ".agent-rfc/designs/adoption.md"
PROVIDERS = ".agenticframework/providers.json"
# The contract version a repository adopted today speaks, and this framework's
# own major as the range it expects (contract/gate/v1/protocol.md).
GATE_CONTRACT = 3
# AgentSmith's own CI setup step: what a tenant pins to pin the provider in CI
# (contract/gate/v2/protocol.md, .agent-rfc/designs/gate-contract-ci.md).
SETUP_ACTION = "bobbyaqlaar/AgentSmith/.github/actions/setup-agentsmith"
GATES_WORKFLOW = ".github/workflows/agentsmith-gates.yml"
# Weekly, it brings the repository up to the framework's latest release and
# opens a pull request (.agent-rfc/designs/sync-pull-request.md). Written here
# so a tenant hears about an upgrade without anyone remembering to look.
SYNC_WORKFLOW = ".github/workflows/agentsmith-sync.yml"

# Client-side hooks git runs that a stub may stand in for. A known list, so a
# helper file beside the hooks (husky.sh, a README) is never mistaken for one.
CLIENT_HOOKS = frozenset({
    "applypatch-msg", "pre-applypatch", "post-applypatch", "pre-merge-commit", "prepare-commit-msg",
    "post-commit", "pre-rebase", "post-checkout", "post-merge", "post-rewrite", "reference-transaction",
    "pre-auto-gc", "post-index-change", "push-to-checkout", "sendemail-validate",
})
# The machine's own hooks (install-ai-stack.sh copies them into every `git init`)
# that provision a repository: post-checkout vendors framework code and writes
# CI workflows, post-commit tags, pushes and re-maps the graph. `tenant init`
# wants both; an adopted repository keeps its own layout and CI, so adopt does
# not chain them (`provisioning=False`). Its guardrail hooks still run.
PROVISIONING_HOOKS = frozenset({"post-checkout", "post-commit"})
STUB = """#!/usr/bin/env bash
# Written by `agentsmith tenant init` / `tenant adopt`: runs this hook from the
# directory this repository used before the gates were armed (.githooks/chain).
exec bash "$(dirname "${BASH_SOURCE[0]}")/chain" "$(basename "${BASH_SOURCE[0]}")" "$@"
"""

# The rules port (contract/rules/v1/protocol.md): what this repository declares
# as its rules provider renders the files every IDE's agent reads, and adopt and
# sync place them by the contract's rules. AgentSmith's own command is answered
# in-process — the install running adopt IS that provider.
RULES_COMMAND = "agentsmith rules"
RULES_CONTRACT = 1

SOURCE_EXTENSIONS = {
    "python-fastapi": (".py",),
    "ts-react": (".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs"),
    "go": (".go",),
}
# Top-level directories that hold source files but are not this repository's code.
_NOT_CODE = frozenset({"node_modules", "vendor", "dist", "build", "docs", "venv", "site-packages", "third_party"})
_TEST_DIRS = frozenset({"test", "tests", "spec", "__tests__", "e2e"})
# A directory holding AgentSmith's own code, copied in by hooks/post-checkout —
# the markers the hook itself keys on. It is the framework's, not the
# repository's: gated, the next re-vendoring would need a design and a review of
# framework code (.agent-rfc/designs/installed-architectures.md).
VENDORED_MARKERS = {"scripts": "run-security-checks.py", "runtime": "llm_gateway.py"}


class AdoptError(ValueError):
    """This repository is not one adopt is for, or it cannot tell enough about it."""


class RulesUnavailable(RuntimeError):
    """The declared rules provider gave no render a caller may place — nothing is written."""


@dataclass
class Plan:
    tenant_id: str
    root: Path
    stack: str
    style: Optional[str]
    agentic: bool
    gated: list[str]
    source_root: Optional[str]
    prior_hooks: Optional[Path]
    framework_ref: str
    actions: list[tuple[str, str]] = field(default_factory=list)  # (path, create|merge|append|leave)
    warnings: list[str] = field(default_factory=list)


# ── git ──────────────────────────────────────────────────────────────────────


def _git(root: Path, *args: str) -> str:
    done = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, check=False)
    return done.stdout.strip() if done.returncode == 0 else ""


def _executable_hooks(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.iterdir()
                  if p.is_file() and not p.name.endswith(".sample") and os.access(p, os.X_OK))


def prior_hooks_dir(root: Path) -> Optional[Path]:
    """The hooks directory this repository ran before the gates: `core.hooksPath`
    if it names anything but `.githooks`, else `.git/hooks` when it holds more
    than git's samples. None when there is nothing to keep running."""
    configured = _git(root, "config", "--get", "core.hooksPath")
    if configured and configured.rstrip("/") != ".githooks":
        path = Path(configured).expanduser()
        path = path if path.is_absolute() else root / path
        return path if path.is_dir() else None
    common = _git(root, "rev-parse", "--git-common-dir")
    if not common:
        return None
    hooks = (Path(common) if Path(common).is_absolute() else root / common) / "hooks"
    return hooks if _executable_hooks(hooks) else None


def is_machine_hook(hook: Path) -> bool:
    """One of AgentSmith's own hooks/ — each says so in its second line."""
    try:
        return hook.read_text(encoding="utf-8", errors="replace").splitlines()[1].startswith("# AgentSmith:")
    except (OSError, IndexError):
        return False


def stub_names(prior: Path, provisioning: bool = True) -> list[str]:
    """The hooks in `prior` that get a stub in .githooks: every client hook the
    gate does not own, less the machine's provisioning hooks unless wanted."""
    return [hook.name for hook in _executable_hooks(prior)
            if hook.name in CLIENT_HOOKS and hook.name not in GATE_HOOKS
            and (provisioning or hook.name not in PROVISIONING_HOOKS or not is_machine_hook(hook))]


def chain_hooks(root: Path, prior: Path, provisioning: bool = True) -> list[str]:
    """Keep `prior`'s hooks running behind the gates: record the directory in
    `agentsmith.chainHooksPath` (read by .githooks/chain), and write a stub for
    each hook the gate does not own (`stub_names`). Returns the stubs written.

    A post-commit that runs tags and may push on its own; a repository that has
    not been pushing on commit must not start, so `agentsmith.autopush` is set
    to false when it is unset."""
    _git(root, "config", "agentsmith.chainHooksPath", str(prior))
    written = []
    for name in stub_names(prior, provisioning):
        stub = root / ".githooks" / name
        stub.parent.mkdir(parents=True, exist_ok=True)
        stub.write_text(STUB, encoding="utf-8")
        stub.chmod(0o755)
        written.append(f".githooks/{name}")
    if ".githooks/post-commit" in written and not _git(root, "config", "--get", "agentsmith.autopush"):
        _git(root, "config", "agentsmith.autopush", "false")
        print(f"  ! {prior}/post-commit tags and pushes on commit; set `agentsmith.autopush false` here so a "
              "commit does not start publishing. `git config agentsmith.autopush true` turns it back on.")
    return written


# ── Detection ────────────────────────────────────────────────────────────────


def _detect_stack(root: Path) -> str:
    """The same order scripts/generate-ide-config.py and hooks/post-checkout use."""
    if (root / "package.json").is_file():
        return "ts-react"
    if any((root / name).is_file() for name in ("pyproject.toml", "Pipfile", "setup.py")) \
            or list(root.glob("requirements*.txt")):
        return "python-fastapi"
    if (root / "go.mod").is_file():
        return "go"
    raise AdoptError("cannot tell this repository's stack (no package.json, pyproject.toml, requirements*.txt, "
                     "Pipfile, setup.py or go.mod) — pass --stack")


def vendored_dirs(root: Path) -> list[str]:
    """Top-level directories that hold vendored AgentSmith code, by its own markers."""
    return sorted(name for name, marker in VENDORED_MARKERS.items() if (root / name / marker).is_file())


def _detect_code(root: Path, stack: str) -> tuple[list[str], Optional[str]]:
    """(gated globs, the source root for docs/DESIGN.md) from what git tracks.
    Vendored framework directories are left out (`vendored_dirs`)."""
    extensions = SOURCE_EXTENSIONS[stack]
    vendored = set(vendored_dirs(root))
    dirs: set[str] = set()
    root_exts: set[str] = set()
    for path in _git(root, "ls-files").splitlines():
        if not path.endswith(extensions):
            continue
        head, _, rest = path.partition("/")
        if not rest:
            root_exts.add(Path(path).suffix)
        elif not head.startswith(".") and head not in _NOT_CODE and head not in vendored:
            dirs.add(head)
    globs = [f"{d}/**" for d in sorted(dirs)] + [f"*{ext}" for ext in sorted(root_exts)]
    code = sorted(d for d in dirs if d not in _TEST_DIRS)
    return globs, (f"{code[0]}/" if len(code) == 1 else None)


def plan_adoption(tenant_id: str, root: Path, *, stack: Optional[str] = None, architecture: Optional[str] = None,
                  agentic: bool = False, gate: Optional[Sequence[str]] = None,
                  framework_ref: Optional[str] = None) -> Plan:
    """What is in `root` and what adopt would do to it. Writes nothing."""
    from runtime import architectures
    from runtime.cli import ALWAYS_GATED, STACKS, _default_framework_version, looks_like_framework, \
        validate_tenant_id

    root = Path(root)
    validate_tenant_id(tenant_id)
    marker = looks_like_framework(root)
    if marker:
        raise AdoptError(f"{root} looks like the AgentSmith framework itself ({marker})")
    if not _git(root, "rev-parse", "--show-toplevel"):
        raise AdoptError(f"{root} is not a git repository")
    if not _git(root, "rev-parse", "--verify", "-q", "HEAD"):
        raise AdoptError(f"{root} has no commit yet — a new repository is `agentsmith tenant init`'s")
    if (root / ".agenticframework" / "process-gates.json").exists():
        committed = subprocess.run(["git", "-C", str(root), "cat-file", "-e",
                                    "HEAD:.agenticframework/process-gates.json"], capture_output=True, check=False)
        if committed.returncode == 0:
            raise AdoptError(f"{root} already carries .agenticframework/process-gates.json — it is already under "
                             "the gates; change that file under a design instead")
        raise AdoptError(f"{root} has an uncommitted .agenticframework/process-gates.json — an adoption that did not "
                         "finish? `git status` lists what it wrote; remove those files and run adopt again")
    if (root / ADOPTION_DESIGN).exists():
        raise AdoptError(f"{root / ADOPTION_DESIGN} exists — adopt writes the adoption's design there; move yours")
    for rel in (".claude/settings.json", ".cursor/hooks.json"):
        if (root / rel).is_file():
            try:
                json.loads((root / rel).read_text(encoding="utf-8"))
            except ValueError as exc:
                raise AdoptError(f"{root / rel} is not valid JSON ({exc}) — fix it first; adopt merges into it") \
                    from exc
    if stack is not None and stack not in STACKS:
        raise AdoptError(f"stack must be one of {STACKS}, got {stack!r}")
    stack = stack or _detect_stack(root)
    style = architectures.resolve_style(architecture) if architecture else None
    detected, source_root = _detect_code(root, stack)
    globs = list(gate) if gate else detected
    if not globs:
        raise AdoptError(f"found no tracked {'/'.join(SOURCE_EXTENSIONS[stack])} files to gate — pass --gate GLOB")

    from runtime.cli import _framework_dir, missing_gate_hooks

    framework = _framework_dir()
    if framework is None:
        raise AdoptError("AgentSmith's scripts are not in $AGENTSMITH_DIR, ~/.agent-framework or this "
                         "checkout — run install-ai-stack.sh")
    missing = missing_gate_hooks(framework)
    if missing:
        raise AdoptError(f"{framework}/.githooks/ has no {', '.join(missing)} — the gates cannot be armed "
                         "from this install, and adopting without them would leave this repository with no "
                         "hooks at all. Re-run install-ai-stack.sh from a current AgentSmith checkout")
    for name in GATE_HOOKS:
        mine = root / ".githooks" / name
        if mine.exists():
            raise AdoptError(f"{mine} exists and is not AgentSmith's — move this repository's hooks to another "
                             "directory and point core.hooksPath there; adopt will keep them running")

    plan = Plan(tenant_id=tenant_id, root=root, stack=stack, style=style, agentic=agentic,
                gated=[*globs, *ALWAYS_GATED], source_root=source_root, prior_hooks=prior_hooks_dir(root),
                framework_ref=framework_ref or f"v{_default_framework_version()}")

    def fate(rel: str, present: str) -> str:
        return present if (root / rel).exists() else "create"

    actions = [(".agenticframework/tenant.yaml", fate(".agenticframework/tenant.yaml", "leave")),
               (".agenticframework/process-gates.json", "create"),
               (PROVIDERS, fate(PROVIDERS, "leave"))]
    actions += [(f".githooks/{name}", "create") for name in GATE_HOOKS]
    if plan.prior_hooks is not None:
        actions += [(f".githooks/{name}", "create") for name in stub_names(plan.prior_hooks, provisioning=False)]
        skipped = sorted(set(stub_names(plan.prior_hooks)) - set(stub_names(plan.prior_hooks, provisioning=False)))
        if skipped:
            plan.warnings.append(f"AgentSmith's own {' and '.join(skipped)} in {plan.prior_hooks} are not chained: "
                                 "they vendor framework code, write CI workflows and tag commits, and an adopted "
                                 "repository keeps its own layout and CI. Its other hooks run behind the gates")
    actions += [(".claude/settings.json", fate(".claude/settings.json", "merge")),
                (".cursor/hooks.json", fate(".cursor/hooks.json", "leave"))]
    try:
        for file in rendered_rules(root, framework).files:
            actions.append((file.path, fate(file.path, "regenerate" if file.placement == "whole" else "merge")))
    except RulesUnavailable as exc:
        plan.warnings.append(f"the rule files will not be written: {exc}")
    actions.append((".agent-history.log", fate(".agent-history.log", "leave")))
    design = root / "docs" / "DESIGN.md"
    if not design.exists():
        actions.append(("docs/DESIGN.md", "create"))
    else:
        has_target = "## Architecture (target)" in design.read_text(encoding="utf-8")
        actions.append(("docs/DESIGN.md", "leave" if has_target else "append"))
    workflows = root / ".github" / "workflows"
    existing = sorted(p.name for p in workflows.glob("*.y*ml")) if workflows.is_dir() else []
    actions += [(f".github/workflows/{name}", "leave") for name in existing if name != Path(GATES_WORKFLOW).name]
    actions.append((GATES_WORKFLOW, fate(GATES_WORKFLOW, "leave")))
    actions.append((SYNC_WORKFLOW, fate(SYNC_WORKFLOW, "leave")))
    actions += [(".agent-rfc/fixtures/knowledge_graph.json", fate(".agent-rfc/fixtures/knowledge_graph.json",
                                                                   "merge")),
                (ADOPTION_DESIGN, "create"), (".agenticframework/scaffold.json", "create")]
    plan.actions = actions

    vendored = vendored_dirs(root)
    if vendored:
        plan.warnings.append(
            f"{', '.join(name + '/' for name in vendored)} holds vendored AgentSmith code (its own "
            "post-checkout put it there) and is not gated — gating it would make the next re-vendoring "
            "need a design and a review of framework code. This repository's own code is gated")
    package = root / "package.json"
    if package.is_file() and "husky" in package.read_text(encoding="utf-8"):
        plan.warnings.append("package.json runs husky, which points core.hooksPath back at .husky on install and "
                             "disarms the gates; after an install, `git config core.hooksPath .githooks` re-arms "
                             "them (the husky hooks keep running through the chain)")
    if _git(root, "status", "--porcelain"):
        plan.warnings.append("the working tree has uncommitted changes; the adoption commit stages only what "
                             "adopt writes")
    return plan


def describe(plan: Plan) -> str:
    """The plan, as the person deciding reads it."""
    lines = [f"Adopting {plan.root} as tenant '{plan.tenant_id}'",
             f"  stack          {plan.stack}",
             f"  architecture   {plan.style or 'none chosen'}{' + agentic' if plan.agentic else ''}",
             f"  gated          {', '.join(plan.gated)}",
             f"  prior hooks    {plan.prior_hooks or 'none'}"
             + (" — kept running behind the gates" if plan.prior_hooks else ""),
             f"  gates workflow runs AgentSmith {plan.framework_ref}", ""]
    width = max(len(action) for _, action in plan.actions)
    lines += [f"  {action:<{width}}  {path}" for path, action in plan.actions]
    lines += [f"\n  ! {warning}" for warning in plan.warnings]
    return "\n".join(lines)


# ── Writing ──────────────────────────────────────────────────────────────────


def rules_port(framework: Optional[Path] = None):
    """The framework's scripts/rules_port.py: the contract's placement rules and
    AgentSmith's own render. Loaded from the framework, as the gate's modules are."""
    from runtime.cli import _framework_dir

    # The named framework, else the install this module came from, else the
    # resolved one — the order `_actions_dir` uses, for the reason it gives.
    candidates: list[Optional[Path]] = [Path(framework)] if framework is not None else \
        [Path(__file__).resolve().parent.parent, _framework_dir()]
    source = next((c / "scripts" / "rules_port.py" for c in candidates
                   if c is not None and (c / "scripts" / "rules_port.py").is_file()), None)
    if source is None:
        raise RulesUnavailable("this AgentSmith install has no scripts/rules_port.py — re-run install-ai-stack.sh")
    loaded = _RULES_PORTS.get(source)
    if loaded is None:
        import importlib.util

        spec = importlib.util.spec_from_file_location(f"rules_port_{len(_RULES_PORTS)}", source)
        if spec is None or spec.loader is None:
            raise RulesUnavailable(f"{source} cannot be loaded as a module")
        loaded = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(loaded)
        _RULES_PORTS[source] = loaded
    return loaded


_RULES_PORTS: dict = {}


def declared_rules(root: Path) -> Optional[str]:
    """The `rules` port this repository declares: `"none"`, a command, or None."""
    try:
        port = json.loads((root / PROVIDERS).read_text(encoding="utf-8"))["providers"].get("rules")
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None
    if port == "none":
        return "none"
    command = port.get("command") if isinstance(port, dict) else None
    return command if isinstance(command, str) and command.strip() else None


def rendered_rules(root: Path, framework: Optional[Path] = None):
    """What the declared rules provider renders for `root`, validated — every
    path checked against the contract before anything is written. Raises
    RulesUnavailable rather than return half a render."""
    port = rules_port(framework)
    declared = declared_rules(root)
    if declared == "none":
        return port.gm.RulesRender(files=[])
    if declared is None or declared == RULES_COMMAND:
        try:
            return port.render(root)
        except Exception as exc:  # a render that failed is not an empty one
            raise RulesUnavailable(f"AgentSmith's render failed ({type(exc).__name__}: {exc})") from exc
    try:
        # Split on whitespace, as the launcher splits it in CI: one command,
        # read the same way by every caller.
        done = subprocess.run([*declared.split(), "render"], cwd=root,
                              input=json.dumps({"cwd": str(root)}), capture_output=True, text=True, check=False,
                              timeout=120)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RulesUnavailable(f"the declared rules provider `{declared}` did not run ({exc})") from exc
    try:
        return port.gm.RulesRender.model_validate_json(done.stdout)
    except port.gm.ValidationError as exc:
        first = exc.errors()[0] if exc.errors() else {"loc": (), "msg": str(exc)}
        raise RulesUnavailable(
            f"the declared rules provider `{declared}` gave no render this repository may place (exit "
            f"{done.returncode}: {'.'.join(str(p) for p in first['loc'])} {first['msg']})") from exc


def setup_reference(ref: Optional[str] = None) -> str:
    """AgentSmith's CI setup step at `ref` (default: this release)."""
    from runtime.cli import _default_framework_version

    return f"{SETUP_ACTION}@{ref or 'v' + _default_framework_version()}"


def declared_setup(root: Path) -> Optional[str]:
    """The `setup` this repository's declaration names for its gate, if any."""
    try:
        gate = json.loads((root / PROVIDERS).read_text(encoding="utf-8"))["providers"]["gate"]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    setup = gate.get("setup") if isinstance(gate, dict) else None
    return setup if isinstance(setup, str) and setup else None


def workflow_setup(root: Path, ref: Optional[str] = None) -> str:
    """The setup step the gates workflow runs: another provider's, as declared,
    and never rewritten; AgentSmith's own, at `ref` — so a sync to a new release
    moves the tenant's CI to it."""
    declared = declared_setup(root)
    if declared and not declared.startswith(SETUP_ACTION + "@"):
        return declared
    return setup_reference(ref)


def providers_declaration(command: str = "agentsmith gate", setup: Optional[str] = None) -> str:
    """Who governs this repository, as `contract/gate/v3/providers.schema.json`
    describes it. Named rather than implied: the hooks and CI ask the
    declaration, and another platform's command goes here instead
    (.agent-rfc/designs/provider-resolution.md, gate-contract-ci.md)."""
    from runtime.cli import _default_framework_version

    major = _default_framework_version().split(".")[0]
    return json.dumps({
        "_about": "Who governs this repository. The hooks and CI ask this before anything else; "
                  "`\"gate\": \"none\"` declares the repository ungoverned. See "
                  "contract/gate/v3/protocol.md.",
        "contract": GATE_CONTRACT,
        "providers": {"gate": {"command": command, "version": f"^{major}", "setup": setup or setup_reference()},
                      "rules": {"command": RULES_COMMAND, "version": f"^{major}", "contract": RULES_CONTRACT}},
    }, indent=2) + "\n"


def merge_rules_block(existing: str, generated: str) -> str:
    """`existing` with the generated rules in one marked block — replacing the
    block a previous run left, never adding a second (the contract's `block`
    placement)."""
    port = rules_port()
    return port.place(existing, port.gm.RulesFile(path="CLAUDE.md", text=generated, placement="block",
                                                  kind="instructions"))


def _write_rules(plan: Plan, framework: Path) -> list[str]:
    """Place what the declared rules provider renders: a file the repository
    lacks is created, one it has gets the provider's block."""
    try:
        rendered = rendered_rules(plan.root, framework)
    except RulesUnavailable as exc:
        print(f"  ! rule files not written: {exc}", file=sys.stderr)
        return []
    written = rules_port(framework).write(plan.root, rendered)
    # Not a rule file, so not in a render — but the plan promises it, and
    # pillar 5 tells every agent to read it.
    if rules_port(framework).seed_history(plan.root):
        written.append(".agent-history.log")
    return written


def adopt(plan: Plan) -> list[str]:
    """Carry out `plan`. Returns every path written or changed, for the commit."""
    from runtime import architectures
    from runtime.cli import SCAFFOLD_MANIFEST, _framework_dir, _process_gates_config, _templates_dir, \
        install_gate_hooks, tenant_yaml, write_scaffold_records

    framework = _framework_dir()
    if framework is None:
        raise AdoptError("AgentSmith's scripts are not in $AGENTSMITH_DIR, ~/.agent-framework or this checkout — "
                         "run install-ai-stack.sh")
    root, written = plan.root, []

    def put(rel: str, body: str) -> None:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(body, encoding="utf-8")
        written.append(rel)

    if not (root / ".agenticframework" / "tenant.yaml").exists():
        put(".agenticframework/tenant.yaml", tenant_yaml(plan.tenant_id))
    put(".agenticframework/process-gates.json",
        _process_gates_config(plan.stack, architectures.session_start_line(plan.style, plan.agentic), plan.gated))
    if not (root / PROVIDERS).exists():
        put(PROVIDERS, providers_declaration(setup=setup_reference(plan.framework_ref)))

    written += install_gate_hooks(root, framework, prior=plan.prior_hooks, provisioning=False)

    sys.path.insert(0, str(framework / "scripts"))
    import gate_ides  # type: ignore

    from runtime.config import chosen_ides

    for ide in chosen_ides(root, gate_ides.GENERATED):
        rel = gate_ides.ADAPTERS[ide].config_path
        existing = json.loads((root / rel).read_text(encoding="utf-8")) if (root / rel).is_file() else None
        if existing is not None and ide != "claude":
            continue  # its config is rewritten whole, so one the repository has is left alone
        put(rel, json.dumps(gate_ides.render_config(ide, existing), indent=2) + "\n")

    written += _write_rules(plan, framework)

    design = root / "docs" / "DESIGN.md"
    section = architectures.render_architecture(plan.stack, plan.style, plan.agentic, target=design.exists(),
                                                source_root=plan.source_root)
    if not design.exists():
        put("docs/DESIGN.md", f"# {plan.tenant_id} — design\n\nThe living picture of this system: its structure, "
                              f"and why.\n\n{section}")
    elif "## Architecture (target)" not in design.read_text(encoding="utf-8"):
        put("docs/DESIGN.md", design.read_text(encoding="utf-8").rstrip("\n") + "\n\n" + section)

    for workflow in (GATES_WORKFLOW, SYNC_WORKFLOW):
        if (root / workflow).exists():
            continue
        name = Path(workflow).name
        template = next((d / name for d in (_templates_dir(), framework / "workflow-templates")
                         if d is not None and (d / name).is_file()), None)
        if template is None:
            print(f"  ! {name} template not found — re-run install-ai-stack.sh", file=sys.stderr)
            continue
        put(workflow, template.read_text(encoding="utf-8").replace(
            "{{PROVIDER_SETUP}}", setup_reference(plan.framework_ref)))

    graph = ".agent-rfc/fixtures/knowledge_graph.json"
    mapper = framework / "scripts" / "map_codebase.py"
    if mapper.is_file():
        subprocess.run([sys.executable, str(mapper), "--quiet"], cwd=root, capture_output=True, check=False)
    if not (root / graph).is_file():
        put(graph, json.dumps({"directed": True, "multigraph": False, "graph": {}, "nodes": [], "links": []},
                              indent=2) + "\n")
    else:
        written.append(graph)

    written = list(dict.fromkeys(written))
    written += write_scaffold_records(root, plan.tenant_id, plan.stack, plan.style, plan.agentic, written,
                                      force=True, design=ADOPTION_DESIGN, adopted=True)
    if SCAFFOLD_MANIFEST not in written:
        raise AdoptError("the adoption manifest was not written — see the messages above")
    return written


def commit_command(written: Sequence[str], rfc: Optional[str] = None) -> str:
    """The adoption commit: what adopt wrote, staged by name — `git add -A`
    would sweep in whatever else the working tree holds.

    `rfc` is an `RFC-NNN` reference when the repository has one. It goes in a
    trailer because hooks/commit-msg requires such a reference under an
    enterprise org policy and greps the whole message, and because the subject
    has 72 characters to spend
    (.agent-rfc/designs/scaffold-rfc-and-vouched-skip.md).
    """
    paths = " ".join(shlex.quote(p) for p in written)
    refs = f" -m \"Refs: {rfc}\"" if rfc else ""
    return (f"git add -- {paths} && git commit -m \"chore: adopt AgentSmith gates\" "
            f"-m \"Design: {ADOPTION_DESIGN}\" -m \"Review: n/a: generated scaffold\"{refs}")
