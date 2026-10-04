"""
runtime/sync.py — `agentsmith sync`: bring a tenant's copies of this framework
up to date (.agent-rfc/designs/framework-sync.md).

    plan_sync(root)     what is stale, without writing
    sync(plan)          write it; returns the paths it wrote
    commit_command(...) the commit, which the tenant's own gates accept

A tenant holds the gate hooks, the IDE hook configs, the generated rule files
and — vendored tenants — the framework's own code. `agentsmith upgrade`
refreshed the last group only, so the rest drifted silently: the gate hooks
changed six times in 90 days and no tenant saw any of it.

Everything here writes what THIS framework owns and nothing else. The tenant's
code, its CI beyond the gates workflow, and its documents are not touched, which
is what lets the commit be reviewed by hash rather than by hand.
"""

from __future__ import annotations

import hashlib
import json
import re
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

SYNC_DESIGN = ".agent-rfc/designs/adoption.md"
MANIFEST = ".agenticframework/scaffold.json"
GENERATED_BY = "agentsmith sync"


class SyncError(ValueError):
    """This repository is not one `sync` can bring up to date."""


@dataclass
class Plan:
    root: Path
    tenant_id: str
    stack: str
    vendored: bool
    framework: Path
    version: str
    stale: list[str] = field(default_factory=list)
    added: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _manifest(root: Path) -> dict:
    try:
        return json.loads((root / MANIFEST).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def is_adopted(root: Path) -> bool:
    """A repository `tenant adopt` brought in: never vendored into, by either
    path (`agentsmith sync` and `agentsmith upgrade` both read this)."""
    return str(_manifest(root).get("generated_by", "")).endswith("tenant adopt")


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ownership(root: Path, rel: str) -> str:
    """Whose file is this? The manifest is the record, and there are three
    answers (.agent-rfc/designs/sync-merged-files.md):

        ours      the framework wrote it and nobody has touched it — refresh it
        edited    the framework wrote it and the tenant changed it — leave it, and say so
        theirs    never ours — leave it, silently
    """
    recorded = (_manifest(root).get("files") or {}).get(rel)
    if recorded is None:
        return "theirs"
    here = root / rel
    if not here.is_file():
        return "ours"  # it was ours and is gone: writing it back is the refresh
    return "ours" if _digest(here) == recorded else "edited"


def plan_sync(root: Path, *, tenant_id: Optional[str] = None, framework: Optional[Path] = None) -> Plan:
    """What `sync` would refresh here. Writes nothing.

    `framework` names the copy to sync FROM; the default is the one this
    install resolves. A tenant is stale when the framework moved on, so the two
    are deliberately separate."""
    from runtime.cli import GATE_HOOKS, _default_framework_version, _framework_dir, looks_like_framework, \
        missing_gate_hooks

    root = Path(root)
    # First, before anything is read: the framework carries process-gates.json
    # like any governed repository, so the check below let a sync scaffold a
    # tenant into the framework itself (.agent-rfc/designs/framework-sync-refuses-framework.md).
    # No override, unlike `tenant init`'s: there is nothing to sync it FROM.
    marker = looks_like_framework(root)
    if marker:
        raise SyncError(f"{root} is AgentSmith's own checkout ({marker}): it is what `sync` copies FROM, "
                        "not a tenant. Run it in a tenant repository, or pass --root")
    if not (root / ".agenticframework" / "process-gates.json").is_file():
        raise SyncError(f"{root} is not under the gates — `agentsmith tenant adopt` brings a repository in")
    framework = Path(framework) if framework is not None else _framework_dir()
    if framework is None:
        raise SyncError("AgentSmith's scripts are not in $AGENTSMITH_DIR, ~/.agent-framework or this "
                        "checkout — run install-ai-stack.sh")
    missing = missing_gate_hooks(framework)
    if missing:
        raise SyncError(f"{framework}/.githooks/ has no {', '.join(missing)} — this install cannot arm the "
                        "gates, and a sync from it would disarm this repository. Re-run install-ai-stack.sh")

    manifest = _manifest(root)
    plan = Plan(root=root, tenant_id=tenant_id or str(manifest.get("tenant") or root.name),
                stack=str(manifest.get("stack") or "python-fastapi"), vendored=not is_adopted(root),
                framework=framework, version=_default_framework_version())
    # Stale: a file this framework owns whose copy here differs from the
    # framework's. A sync adds nothing new to a repository that did not ask for
    # it — except the gate's own hooks, which a repository that armed the gates
    # did ask for: one armed before `pre-commit`, `pre-push` and `chain` existed
    # gets them, and they go in the manifest and the commit like any refresh
    # (.agent-rfc/designs/sync-adds-missing-hooks.md).
    for hook in (framework / ".githooks").iterdir():
        here = root / ".githooks" / hook.name
        if here.is_file() and hook.is_file() and _digest(here) != _digest(hook):
            plan.stale.append(f".githooks/{hook.name}")
    plan.added = [f".githooks/{hook}" for hook in GATE_HOOKS
                  if (framework / ".githooks" / hook).is_file() and not (root / ".githooks" / hook).exists()]
    if {".githooks/pre-commit", ".githooks/pre-push"} & set(plan.added):
        plan.notes.append("new here: the bypass sweep runs on every commit and push from now on, and a push is "
                          "refused while a commit that skipped the gate is outstanding — `agentsmith gates repair`")

    # The files the tenant shares with the framework. `ownership` decides: a
    # file they have edited is named and left, never clobbered.
    shared, regions = _shared_files(plan)
    for rel, text in shared.items():
        state = ownership(root, rel)
        here = root / rel
        if (rel in _MERGED or rel in regions) and here.is_file():
            # Their file, our region: `text` already keeps everything that is
            # theirs, so the file's own hash is not the question.
            if here.read_text(encoding="utf-8") != text:
                plan.stale.append(rel)
            continue
        if state == "edited":
            plan.notes.append(f"{rel} was edited here since the framework wrote it — left alone. "
                              "Revert it and re-run to take the framework's version")
        elif state == "ours" and (not here.is_file() or here.read_text(encoding="utf-8") != text):
            plan.stale.append(rel)
    if plan.vendored:
        plan.notes.append("vendored tenant: scripts/, runtime/, templates/ and fixtures/ are refreshed too")
    else:
        plan.notes.append("adopted repository: nothing is vendored into it")
    return plan


# Files where the framework owns a REGION and the tenant owns the rest, so the
# file's own hash is not the question: Claude's settings, where `render_config`
# replaces the hooks and keeps permissions and everything else. Judging it by the
# whole file would freeze a tenant's gate wiring the moment they edited their own
# permissions (.agent-rfc/designs/sync-merged-files.md). The rule files a
# provider renders as a `block` are judged the same way, per file, by
# `_shared_files`. Cursor's config is not here: the framework renders it whole,
# so a tenant who rewrote it owns it.
_MERGED = (".claude/settings.json",)

CONFIG = ".agenticframework/process-gates.json"
# What `tenant adopt` used to write for each document: this install's layout, in
# the tenant's declaration. Exactly these values become `"provider"`; any other
# value is the tenant's choice (.agent-rfc/designs/rules-contract.md).
LEGACY_DOCUMENTS = {"registry": "@framework/templates/governance.json",
                    "levers_doc": "@framework/docs/review-levers.md",
                    "design_checklist": "@framework/docs/design-review-checklist.md"}


def provider_documents(text: str) -> str:
    """`process-gates.json` with each legacy `@framework/` document named
    `"provider"` instead — edited in place, so the rest of the file is byte for
    byte what the tenant committed."""
    for key, legacy in LEGACY_DOCUMENTS.items():
        text = re.sub(rf'("{key}"\s*:\s*)"{re.escape(legacy)}"', r'\1"provider"', text)
    return text


def _shared_files(plan: Plan) -> tuple[dict[str, str], set[str]]:
    """What the framework would write today for the files a tenant also edits:
    the IDE hook configs, the rule files, the gates workflow and the gate config —
    and which of them it owns only a region of."""
    import json as _json
    import sys as _sys

    from runtime.adopt import GATES_WORKFLOW, PROVIDERS, SYNC_WORKFLOW, RulesUnavailable, providers_declaration, \
        rendered_rules, rules_port, workflow_setup

    root, framework = plan.root, plan.framework
    shared: dict[str, str] = {}
    regions: set[str] = set()
    # The rule files, as the declared rules provider renders them and placed by
    # the contract's rules: a `block` keeps everything outside it (contract/rules/v1).
    try:
        port = rules_port(framework)
        for file in rendered_rules(root, framework).files:
            here = root / file.path
            existing = here.read_text(encoding="utf-8") if here.is_file() else None
            shared[file.path] = port.place(existing, file)
            if file.placement == "block":
                regions.add(file.path)
    except RulesUnavailable as exc:
        plan.notes.append(f"rule files not refreshed — nothing of them is written: {exc}")

    config = root / CONFIG
    if config.is_file():
        before = config.read_text(encoding="utf-8")
        after = provider_documents(before)
        if after != before:
            shared[CONFIG] = after
            regions.add(CONFIG)
            plan.notes.append(f"{CONFIG}: documents named by this install's paths now name `\"provider\"` — "
                              "the gate provider's own, wherever it is installed")

    _sys.path.insert(0, str(framework / "scripts"))
    try:
        import gate_ides as gi  # type: ignore

        from runtime.config import chosen_ides

        for ide in chosen_ides(root, gi.GENERATED):
            rel = gi.ADAPTERS[ide].config_path
            here = root / rel
            existing = _json.loads(here.read_text(encoding="utf-8")) if here.is_file() else None
            shared[rel] = _json.dumps(gi.render_config(ide, existing), indent=2) + "\n"
    except Exception as exc:  # a framework too old to render them says so, and the rest still syncs
        plan.notes.append(f"IDE hook configs not refreshed ({exc})")

    for workflow in (GATES_WORKFLOW, SYNC_WORKFLOW):
        template = _workflow_template(framework, Path(workflow).name)
        if template is not None and (root / workflow).is_file():
            # The gates workflow carries the provider's setup step, pinned, so a
            # tenant that syncs starts running the new release in CI — unless it
            # declares another provider's step, which is never rewritten. The
            # sync workflow follows the latest release and has nothing to substitute.
            shared[workflow] = template.read_text(encoding="utf-8").replace(
                "{{PROVIDER_SETUP}}", workflow_setup(root, f"v{plan.version}"))
    # The declaration itself, moved to gate contract 2 while it is the one the
    # framework wrote; a tenant that edited it — another provider, `none` — owns
    # it, and `ownership` leaves it alone and says so.
    if (root / PROVIDERS).is_file():
        shared[PROVIDERS] = providers_declaration(setup=workflow_setup(root, f"v{plan.version}"))
    return shared, regions


def _workflow_template(framework: Path, name: str) -> Optional[Path]:
    from runtime.cli import _templates_dir

    for directory in (_templates_dir(), framework / "workflow-templates"):
        if directory is not None and (directory / name).is_file():
            return directory / name
    return None


def describe(plan: Plan) -> str:
    lines = [f"Syncing {plan.root} with AgentSmith {plan.version}", ""]
    changes = [f"  stale   {path}" for path in plan.stale] + [f"  add     {path}" for path in plan.added]
    lines += changes or ["  nothing stale in what this framework owns"]
    lines += [f"\n  ! {note}" for note in plan.notes]
    return "\n".join(lines)


def sync(plan: Plan) -> list[str]:
    """Refresh what this framework owns here. Returns the paths written."""
    from runtime.adopt import PROVIDERS, providers_declaration
    from runtime.cli import install_gate_hooks, write_scaffold_records

    root, written = plan.root, []

    # The hooks, from the one implementation `init` and `adopt` use. `force`,
    # because refreshing a stale copy is the point.
    install_gate_hooks(root, plan.framework, force=True, provisioning=False)
    shared, _regions = _shared_files(plan)
    for rel in plan.stale:
        text = shared.get(rel)
        if text is None:
            continue  # a hook: `install_gate_hooks` has just rewritten it
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    written += plan.stale + plan.added
    if not (root / PROVIDERS).exists():
        (root / PROVIDERS).write_text(providers_declaration(), encoding="utf-8")
        written.append(PROVIDERS)

    if plan.vendored:
        from runtime.machine.upgrade import upgrade

        upgrade(root, plan.version, out=lambda _line: None)

    # Only what actually moved: `install_gate_hooks` rewrites every hook, and a
    # commit listing files whose bytes did not change is noise a reviewer reads
    # past — and a manifest entry nobody needed.
    written = [path for path in dict.fromkeys(written) if _changed(root, path)]
    if not written:
        return []
    written += write_scaffold_records(root, plan.tenant_id, plan.stack, None, False, written,
                                      force=True, design=SYNC_DESIGN, adopted=not plan.vendored,
                                      generated_by=GENERATED_BY)
    return written


def _changed(root: Path, rel: str) -> bool:
    """Is this path different from what the repository has committed?"""
    done = subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--", rel],
                          capture_output=True, text=True, check=False)
    return bool(done.stdout.strip())


def commit_command(written: list[str], version: str) -> str:
    paths = " ".join(shlex.quote(path) for path in written)
    return (f"git add -- {paths} && git commit -m \"chore(framework): sync AgentSmith {version}\" "
            f"-m \"Design: {SYNC_DESIGN}\" -m \"Review: n/a: framework sync {version}\"")
