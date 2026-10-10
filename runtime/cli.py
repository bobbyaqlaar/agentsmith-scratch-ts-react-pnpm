"""
runtime/cli.py — `agentsmith`, the framework's entry point.

WHY THIS EXISTS

Everything a user does with AgentSmith went through fifteen zsh functions and
~61 lines that `install-ai-stack.sh` appended to `~/.zshrc`. That has four
defects for something meant to be a product:

  drift        the tenant scaffold lived in the installer AND in every user's
               shell profile. Editing the repo left the machine writing the old
               one, which happened twice in a single day.
  OS lock-in   zsh functions exist only in an interactive macOS/Linux shell.
               Windows has none, a Dockerfile `RUN` has none, CI has none.
  untestable   not one of the fifteen was covered by a test, because they only
               exist after an interactive install.
  invasive     appending to a login shell is the most intrusive thing the
               installer does, and why re-running it feels risky.

A console script has none of those. `pip install agentsmith-runtime` and
`agentsmith tenant init` works in PowerShell, in a container, in a CI step —
with no profile to source and nothing to un-edit on uninstall.

MIGRATION, finished. Every function is now a subcommand here; the logic lives
in runtime/machine/ (.agent-rfc/designs/agentsmith-cli.md). The installer no
longer writes a profile block, and removes the old one.

WHAT A CHILD PROCESS CANNOT DO is export into its parent shell, which is how
the functions "set" a mode. Nothing needs to any more: `agentsmith mode` writes
~/.agent-framework/state/mode, which the gateway and the git hooks read — so
the mode reaches IDEs, git GUIs and hooks, which an export never did.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Optional, Sequence

# The scaffold's shape is a product decision, so it lives here as data rather
# than as a heredoc: a test can assert on it, and there is exactly one copy.
STACKS = ("python-fastapi", "ts-react", "go")
ISOLATIONS = ("shared", "dedicated")


# A tenant id has to survive being written into YAML, read back, and spliced
# into an environment variable NAME (HITL_ENCRYPTION_KEY_<TENANT>). This is the
# set that does all three. Deliberately narrow: this runs once, when a tenant is
# created, which is the only moment the id is free to change.
TENANT_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def validate_tenant_id(tenant_id: str) -> str:
    """The tenant id, or a ValueError explaining what it has to look like."""
    if not isinstance(tenant_id, str) or not tenant_id.strip():
        raise ValueError("tenant id must be a non-empty string")
    if not TENANT_ID_PATTERN.match(tenant_id):
        raise ValueError(
            f"tenant id {tenant_id!r} must be 1-64 characters of letters, digits, "
            f"dot, underscore or hyphen, starting with a letter or digit. It is "
            f"written into YAML, resolved back by runtime/tenancy.py, and spliced "
            f"into the HITL_ENCRYPTION_KEY_<TENANT> variable name."
        )
    return tenant_id


def _default_framework_version() -> str:
    """The release to declare, without a source-checkout marker.

    framework_version() reports `1.3.0+src` from a checkout, which is the right
    answer for "what is running" and the wrong one to write into a tenant's
    config: the tenant is declaring which RELEASE it targets. The literal that
    used to be here was a second copy of the version number and would have
    drifted from pyproject.toml at the next bump.
    """
    from runtime.version import SOURCE_SUFFIX, framework_version

    version = framework_version()
    return version[: -len(SOURCE_SUFFIX)] if version.endswith(SOURCE_SUFFIX) else version


def tenant_yaml(
    tenant_id: str, *, isolation: str = "shared", framework_version: Optional[str] = None,
    ides: Optional[Sequence[str]] = None,
) -> str:
    """The scaffolded `.agenticframework/tenant.yaml`, as text.

    Every key below is READ by something — that is the entry criterion, learned
    from shipping five that were not. `environments:` with `phoenix_namespace`,
    `eval_fail_below` and `redaction_profile` used to be here and was removed:
    nothing read any of it, and two of the three were actively misleading.

    Modes are QUOTED because YAML 1.1 parses a bare `off` as the boolean false,
    which would silently mean something no mode name matches.

    So is the TENANT ID, for the same reason and for a while it was not: the
    quoting was applied to the values someone had thought about rather than to
    the class of problem. `agentsmith tenant init off` wrote `id: off`, YAML
    read False, and runtime/tenancy.py resolved the tenant to the string
    "False" — which then keyed the spend ledger, the HITL encryption variable
    and every span. `123` became an int the same way.
    """
    validate_tenant_id(tenant_id)
    quoted_id = json.dumps(tenant_id)  # a JSON string is a valid YAML scalar
    # Omitted unless declared: an absent key reads as "every verified IDE",
    # which is what runtime.config.chosen_ides falls back to, so writing the
    # full list by default would turn a default into a pin.
    workspace = (
        "\nworkspace:\n"
        "  # Which IDEs get a generated hook config here, read by "
        "`runtime.config.chosen_ides`\n"
        "  # for `tenant init`, `tenant adopt` and `agentsmith sync`. Remove the key to\n"
        "  # let every IDE with a verified config schema be written.\n"
        f"  ides: [{', '.join(json.dumps(i) for i in ides)}]\n"
    ) if ides else ""
    return f"""# {tenant_id} — AgentSmith tenant configuration (docs/DESIGN.md › Tenancy Model)
#
# Generated by `agentsmith tenant init`. Every key here is read by the runtime;
# each `security.*` line is overridden by the environment variable named beside
# it, and a declaration outranks an ambient export — see runtime/config.py.

tenant:
  id: {quoted_id}
  name: {quoted_id}
  # A MARKER, not a control: no code reads it. It tells an operator which
  # manifests to apply (runtime/k8s/dedicated-tenant/ for `dedicated`).
  isolation: {isolation}
  # owner: you@example.com        # else `git config user.email`

framework:
  version: "{framework_version or _default_framework_version()}"
{workspace}
# Security posture. Quoted on purpose: YAML 1.1 reads a bare `off` as false.
security:
  prompt_guard: "default"           # off|warn|default|strict  (PROMPT_GUARD)
  input_guardrail: "default"        # off|default|custom       (INPUT_GUARDRAIL)
  tool_allowlist_strict: false      # deny-by-default tools    (TOOL_ALLOWLIST_STRICT)
  ip_redaction: false               # scrub IPs from spans     (ENABLE_IP_REDACTION)
  # Tool arguments and results on the span. OFF by default: it is a new
  # egress channel. When on, payloads go on the attribute names
  # trace_redactor already scrubs, so the profile applies.
  trace_tool_payloads: false        #                          (TRACE_TOOL_PAYLOADS)

moderation:
  mode: "optional"                  # off|optional|required    (MODERATION_HOOK)
  # hook: mypackage.moderation:classify_output

budget:
  monthly_usd_cap: 150              # AGENT_MONTHLY_USD_CAP overrides

workflow:
  engine: temporal                  # WORKER_BACKEND overrides
  task_queue: {tenant_id}           # TASK_QUEUE overrides

delivery:
  platform: on-prem                 # see docs/delivery-model.md
  data_access_pattern: api-only

# Environment variables permitted to outrank the declarations above. Empty by
# default: an ambient export must not silently relax a declared posture.
# env_overrides: [AGENT_MONTHLY_USD_CAP]
"""


# ── tenant init ──────────────────────────────────────────────────────────────


def _templates_dir() -> Optional[Path]:
    """Where `agentsmith tenant init` and `tenant adopt` copy CI workflows from.

    **This file's own location first**, then the machine install. The order used
    to be the other way round under a docstring claiming it was this way, and
    the code did the opposite of what the sentence promised: running from a
    checkout served the machine's STALE templates. Found by adopting a throwaway
    repository after fixing `agentsmith-gates.yml` — the tenant got the old file,
    because the install had one, while `agentsmith-sync.yml` got the fix, because
    the install did not. A fixed template silently failed to reach a tenant, and
    the adopt tests passed because they were reading the same stale copy.

    An installed `agentsmith` resolves to the same directory either way: there
    `__file__` is already under ~/.agent-framework. The fallback is for a
    vendored tenant, whose `runtime/` has no `workflow-templates/` beside it.
    """
    for candidate in (
        Path(__file__).resolve().parent.parent / "workflow-templates",
        Path.home() / ".agent-framework" / "workflow-templates",
    ):
        if candidate.is_dir():
            return candidate
    return None


class FrameworkRootError(RuntimeError):
    """Refusing to scaffold a tenant into the framework's own checkout."""


# Markers that identify the framework's own repository. `pyproject.toml`
# declaring the package name is the definitive one — that string exists in
# exactly one project. The others are corroborating: a tenant repo has no
# reason to ship the installer or the workflow templates it is a consumer of.
FRAMEWORK_MARKERS = (
    ("pyproject.toml", 'name = "agentsmith-runtime"'),
    ("install-ai-stack.sh", None),
    ("workflow-templates", None),
    ("templates/agent-rules.yaml", None),
)


def looks_like_framework(root: Path) -> Optional[str]:
    """The marker that says this is the framework's checkout, or None.

    Written after scaffolding a tenant into the framework repo by accident:
    `tenant init` defaults --root to the working directory, and running it from
    the wrong terminal is a two-second mistake. Twelve tests went red because
    the stray `tenant.yaml` declared a security posture that then outranked the
    environment those tests set — loud, but only because a precedence change
    two days earlier happened to make it so. It should not depend on that.

    Requires TWO markers, not one. A single check on `install-ai-stack.sh`
    would refuse in a repo that merely vendored the installer, and a guard that
    fires on legitimate work is one people learn to bypass.
    """
    found: list[str] = []
    for name, needle in FRAMEWORK_MARKERS:
        path = root / name
        if not path.exists():
            continue
        if needle is None:
            found.append(name)
        else:
            try:
                if needle in path.read_text(encoding="utf-8"):
                    found.append(f"{name} declares {needle!r}")
            except OSError:
                continue
    return " and ".join(found) if len(found) >= 2 else None


# A callee missing from this list makes GitHub reject the whole workflow as
# invalid, not just the missing job — so they ship together with their caller.
WORKFLOWS = (
    "cd-staging.yml",
    "cd-production.yml",
    "eval-scorecard.yml",
    "eval-fairness.yml",
    "eval-hallucination.yml",
    "eval-ttft-live.yml",
    "eval-security.yml",
)


def gate_ides_module(framework: Optional[Path] = None):
    """`scripts/gate_ides` — the IDE registry — or None where it cannot be
    resolved.

    Resolved through the framework directory rather than imported at module
    scope, because `runtime/` ships vendored into tenants where `scripts/` sits
    elsewhere. None where it is unresolvable, which is how callers tell "nothing
    to check against" from "nothing chosen"; both readers below fail open, since
    a missing registry must not refuse a scaffold.
    """
    framework = framework or _framework_dir()
    if framework is None:
        return None
    sys.path.insert(0, str(framework / "scripts"))
    try:
        import gate_ides  # type: ignore

        return gate_ides
    except Exception:  # fail-open: an unresolvable registry must not block a scaffold
        return None


def init_tenant(
    tenant_id: str,
    root: Path,
    *,
    stack: str = "python-fastapi",
    isolation: str = "shared",
    force: bool = False,
    allow_framework_root: bool = False,
    architecture: Optional[str] = None,
    agentic: bool = False,
    ides: Optional[Sequence[str]] = None,
    rfc: Optional[dict] = None,
) -> list[str]:
    """Scaffold a tenant. Returns the paths written, relative to `root`.

    `architecture` is a structural style (or a common name for one — `clean`,
    `n-tier`, …) and `agentic` adds the agent layer on top of it
    (.agent-rfc/designs/tenant-architecture.md). Both shape docs/DESIGN.md and
    the design written for the scaffold commit.

    Never overwrites without `force`: a tenant.yaml is edited by hand after
    generation, and silently replacing it would discard a declared posture.

    Refuses to write into the framework's own checkout — see
    `looks_like_framework`. `force` does NOT bypass that: it means "replace
    files I already own", which is a different question from "write into the
    wrong repository", and overloading one flag onto both is how a guard gets
    disabled by someone solving an unrelated problem.
    """
    registry = gate_ides_module()
    # Only when the registry resolved: without it we cannot tell a wrong name
    # from a right one, and refusing on that basis would turn a missing install
    # into a rejected argument.
    if registry is not None:
        verified = tuple(registry.GENERATED)
        for ide in ides or ():
            if ide in verified:
                continue
            # `denied-vs-missing`: a typo and a real IDE whose config shape is
            # unconfirmed are different answers, and telling someone their
            # spelling is an unverified schema sends them to the wrong fix.
            known = ide in registry.ADAPTERS and ide != registry.NEUTRAL
            reason = (
                "no verified config schema, so no config can be written for it — the gate "
                "still reads and answers its events; it is the CONFIG FILE's shape that is "
                "unconfirmed"
                if known else "not an IDE this framework knows"
            )
            raise ValueError(f"--ide {ide}: {reason}. Choose from {', '.join(verified)}.")

    marker = None if allow_framework_root else looks_like_framework(root)
    if marker:
        raise FrameworkRootError(
            f"{root} looks like the AgentSmith framework itself ({marker}).\n"
            f"A tenant scaffolded here would declare a security posture and a "
            f"budget that the framework's own tests then run against.\n"
            f"Run this in the tenant's repository, pass --root, or override "
            f"with --allow-framework-root if you really mean it."
        )
    # Every argument is checked before the first mkdir. The id used to be
    # validated inside tenant_yaml(), which runs after `.agenticframework/`
    # exists — so a rejected id left a directory behind and the next run found
    # a half-scaffolded tenant.
    validate_tenant_id(tenant_id)
    if isolation not in ISOLATIONS:
        raise ValueError(f"isolation must be one of {ISOLATIONS}, got {isolation!r}")
    if stack not in STACKS:
        raise ValueError(f"stack must be one of {STACKS}, got {stack!r}")
    from runtime import architectures

    style = architectures.resolve_style(architecture) if architecture else None
    from runtime.adopt import prior_hooks_dir

    # Before anything re-points core.hooksPath: the hooks this repository runs
    # now — the machine's, copied by `git init` — keep running behind the gates.
    prior = prior_hooks_dir(root)

    written: list[str] = []
    cfg_dir = root / ".agenticframework"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg = cfg_dir / "tenant.yaml"
    if cfg.exists() and not force:
        print("  = .agenticframework/tenant.yaml exists — left untouched")
    else:
        cfg.write_text(tenant_yaml(tenant_id, isolation=isolation, ides=ides))
        written.append(".agenticframework/tenant.yaml")

    templates = _templates_dir()
    if templates is None:
        print(
            "  ! no workflow-templates found — CI workflows not written. "
            "Re-run install-ai-stack.sh, or use a checkout.",
            file=sys.stderr,
        )
        return written

    wf_dir = root / ".github" / "workflows"
    wf_dir.mkdir(parents=True, exist_ok=True)
    for name in (f"ci-{stack}.yml", *WORKFLOWS):
        src = templates / name
        if not src.is_file():
            print(f"  ! template missing, skipped: {name}", file=sys.stderr)
            continue
        dest = wf_dir / name
        if dest.exists() and not force:
            print(f"  = .github/workflows/{name} exists — left untouched")
            continue
        # Same substitution hooks/post-checkout makes; copied verbatim, the
        # workflows carried a literal `{{TENANT_ID}}`.
        dest.write_text(src.read_text(encoding="utf-8").replace("{{TENANT_ID}}", tenant_id), encoding="utf-8")
        written.append(f".github/workflows/{name}")

    written += _copy_composite_actions(root)
    written += _provision_governance(
        root, force=force,
        design_md=architectures.render_design_md(tenant_id, stack, style, agentic),
        session_start=architectures.session_start_line(style, agentic),
        prior=prior,
    )
    written += _vendor(root, prior)
    written += write_scaffold_records(root, tenant_id, stack, style, agentic, written, force, rfc=rfc)
    return written


def _all_files(root: Path) -> set[str]:
    return {p.relative_to(root).as_posix() for p in root.rglob("*")
            if p.is_file() and ".git" not in p.relative_to(root).parts[:1]}


def _vendor(root: Path, prior: Optional[Path]) -> list[str]:
    """Run the machine's post-checkout once, before the manifest is written, so
    the framework code it vendors is part of the scaffold the manifest vouches
    for (.agent-rfc/designs/tenant-adopt.md). Vendoring stays the hook's job —
    this only makes it happen before the first commit instead of never.
    Returns the files it added."""
    import subprocess as sp

    from runtime.adopt import is_machine_hook

    hook = prior / "post-checkout" if prior is not None else None
    # The machine's hook only: the manifest vouches for what it writes, and
    # AgentSmith can vouch for its own vendoring, not for another tool's output.
    if hook is None or not os.access(hook, os.X_OK) or not is_machine_hook(hook):
        return []
    before = _all_files(root)
    done = sp.run([str(hook), "0" * 40, "0" * 40, "1"], cwd=root, capture_output=True, text=True, check=False)
    if done.returncode != 0:
        print(f"  ! {hook} failed while vendoring: {(done.stderr or done.stdout).strip()[-300:]}", file=sys.stderr)
    return sorted(_all_files(root) - before)


SCAFFOLD_DESIGN = ".agent-rfc/designs/scaffold.md"
# Depth 1, not under designs/: hooks/pre-commit Guardrail 4 requires at least
# one *.md directly under .agent-rfc/ where an org policy exists, and the
# scaffold design is one level too deep to answer it
# (.agent-rfc/designs/scaffold-rfc-and-vouched-skip.md).
SCAFFOLD_RFC = ".agent-rfc/001-scaffold.md"
SCAFFOLD_MANIFEST = ".agenticframework/scaffold.json"


def _scaffold_files(root: Path, written: list[str]) -> list[str]:
    """Every file behind `written` — a composite action is listed as its directory."""
    files: set[str] = set()
    for rel in written:
        path = root / rel
        if path.is_dir():
            files.update(p.relative_to(root).as_posix() for p in path.rglob("*") if p.is_file())
        elif path.is_file():
            files.add(rel)
    return sorted(files)


def write_scaffold_records(root: Path, tenant_id: str, stack: str, style: Optional[str], agentic: bool,
                           written: list[str], force: bool, design: str = SCAFFOLD_DESIGN,
                           adopted: bool = False, generated_by: Optional[str] = None,
                           rfc: Optional[dict] = None) -> list[str]:
    """The design for the commit that arms the gates, and the manifest that
    lets its review be `n/a: generated scaffold` (scripts/process_gate.py
    scaffold_problems) — for `tenant init`, or `tenant adopt` when `adopted`.

    The manifest lists only files this run wrote: a file the tenant already had
    is theirs, and the gate asks for a real review of it. A file adopt merged
    into is listed as merged — its whole new content is what the hash vouches for."""
    import hashlib

    framework = _framework_dir()
    if framework is None:
        return []
    # The tenant's first RFC. Written before `files` is computed so the manifest
    # records it: not required for the `n/a: generated scaffold` escape —
    # .agent-rfc/** is not a gated path — but `--force` uses the manifest to know
    # what is its own to replace.
    rfc_path = root / SCAFFOLD_RFC
    existing_rfcs = sorted(root.glob(".agent-rfc/*.md")) if (root / ".agent-rfc").is_dir() else []
    if existing_rfcs:
        # Never drop a stub among real ones, and never overwrite an edited
        # template — including on --force.
        print(f"  = {existing_rfcs[0].relative_to(root)} exists — no RFC template written")
    else:
        from runtime import architectures

        rfc_path.parent.mkdir(parents=True, exist_ok=True)
        rfc_path.write_text(architectures.render_scaffold_rfc(tenant_id, stack, rfc), encoding="utf-8")
        written = [*written, SCAFFOLD_RFC]

    files = _scaffold_files(root, written)
    # A re-run skips what exists and is not its to replace (the composite
    # actions, vendored code); an earlier manifest vouched for those, and still
    # does while each is byte-for-byte what it recorded.
    manifest_path = root / SCAFFOLD_MANIFEST
    earlier: dict = {}
    if manifest_path.is_file():
        try:
            earlier = json.loads(manifest_path.read_text(encoding="utf-8")).get("files") or {}
        except (ValueError, AttributeError):
            earlier = {}
    files = sorted(set(files) | {
        f for f, digest in earlier.items()
        if (root / f).is_file() and hashlib.sha256((root / f).read_bytes()).hexdigest() == digest})
    records: list[str] = []
    # Returned as well as recorded: callers stage what this returns, and an RFC
    # that only reached the manifest was left untracked by `tenant adopt`
    # (test_tenant_adopt.py: 'every file adopt wrote was staged by name').
    if SCAFFOLD_RFC in written:
        records.append(SCAFFOLD_RFC)
    registry_path = framework / "templates" / "governance.json"
    design_path = root / design
    if not registry_path.is_file():
        # An install from before the registry existed: the gate it runs cannot
        # read its rules either. Say so rather than write a design that answers
        # pillars nobody can list.
        print(f"  ! {design} not written: {registry_path} is missing — re-run "
              "install-ai-stack.sh from a current AgentSmith checkout", file=sys.stderr)
    elif design_path.exists() and generated_by:
        # A sync keeps the design the arming commit wrote — rewriting it would make
        # every sync look like a new design — but its commit must be covered by it,
        # and a sync can commit what the arming commit never wrote: the hooks a
        # tenant armed before they existed, and the manifest wherever the tenant
        # gates it. Scope lines only; the prose is the arming commit's
        # (.agent-rfc/designs/sync-adds-missing-hooks.md).
        from runtime import architectures

        before = design_path.read_text(encoding="utf-8")
        after = architectures.extend_design_scope(before, [*written, SCAFFOLD_MANIFEST])
        if after != before:
            design_path.write_text(after, encoding="utf-8")
            records.append(design)
        else:
            print(f"  = {design} already covers this sync")
    elif design_path.exists() and not force:
        print(f"  = {design} exists — left untouched")
    else:
        from runtime import architectures

        design_path.parent.mkdir(parents=True, exist_ok=True)
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        # The manifest is in the commit this design covers, and a tenant may gate it.
        design_path.write_text(architectures.render_scaffold_design(
            tenant_id, stack, style, agentic, [*files, SCAFFOLD_MANIFEST], registry.get("pillars", []),
            adopted=adopted), encoding="utf-8")
        records.append(design)
    command = generated_by or ("agentsmith tenant adopt" if adopted else "agentsmith tenant init")
    manifest = {
        "_about": f"What `{command}` wrote, by SHA-256. The review of the commit that arms the gates may be "
                  "`n/a: generated scaffold` only while every gated file still matches — see AgentSmith "
                  "docs/process-gates.md.",
        "generated_by": command,
        "framework_version": _default_framework_version(),
        "tenant": tenant_id,
        "stack": stack,
        "architecture": style,
        "agentic": agentic,
        "files": {f: hashlib.sha256((root / f).read_bytes()).hexdigest() for f in files},
    }
    (root / SCAFFOLD_MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    records.append(SCAFFOLD_MANIFEST)
    return records


# The design-and-review gates, provisioned rather than left as a page in a
# manual (.agent-rfc/designs/governance-enforcement.md, G7). A control a tenant
# has to install by hand is a control most tenants do not have.
GATED_BY_STACK = {
    "python-fastapi": ["app/**", "agents/**", "workflows/**", "scripts/**", "runtime/**", "test/**"],
    "go": ["cmd/**", "internal/**", "pkg/**", "scripts/**"],
    "ts-react": ["src/**", "app/**", "lib/**", "scripts/**"],
}
ALWAYS_GATED = [".github/**", ".githooks/**", ".agenticframework/process-gates.json",
                ".agenticframework/providers.json", ".claude/settings.json", ".cursor/hooks.json"]


def _process_gates_config(stack: str, session_start: Optional[str] = None,
                          gated: Optional[list[str]] = None) -> str:
    """A tenant's gate config.

    The design and review gates are LIVE from the first commit. `artifacts`,
    `pillars` and `knowledge_graph` are `off`: a tenant switched to enforce on
    day one is refused its first commit for documents it has not written and a
    graph it has not built, and the first thing anyone does then is take the
    gates out. Each is turned on deliberately.
    """
    config: dict[str, object] = {
        "_about": "What the process gates cover here — AgentSmith docs/process-gates.md. "
                  "This file is gated itself, so switching a gate off takes a design and a review.",
        "gated": gated or [*GATED_BY_STACK.get(stack, GATED_BY_STACK["python-fastapi"]), *ALWAYS_GATED],
        "not_gated": ["**.md", ".agent-rfc/**", "**/node_modules/**"],
        # The gate provider's own documents, named without naming where it is
        # installed: a provider's layout has no place in a tenant's declaration.
        # Declared, not omitted — an absent `registry` means a config that
        # predates the pillar requirements (.agent-rfc/designs/rules-contract.md).
        "registry": "provider",
        "artifacts": "off",
        "pillars": "off",
        "knowledge_graph": "off",
        "levers_doc": "provider",
        "design_checklist": "provider",
    }
    if session_start:
        # The structural style and its first rule, in every agent session's context.
        config["extends"] = {"session_start": [session_start]}
    return json.dumps(config, indent=2) + "\n"


def _framework_dir() -> Optional[Path]:
    for candidate in (Path(os.environ["AGENTSMITH_DIR"]) if os.environ.get("AGENTSMITH_DIR") else None,
                      Path.home() / ".agent-framework",
                      Path(__file__).resolve().parent.parent):
        if candidate is not None and (candidate / "scripts" / "process_gate.py").is_file():
            return candidate
    return None


GATE_HOOKS = ("process-gate", "commit-msg", "pre-commit", "pre-push", "chain")


def missing_gate_hooks(framework: Path, raising: bool = False) -> list[str]:
    """The gate hooks this framework cannot supply. `raising` turns a non-empty
    answer into the error the caller should not write past
    (.agent-rfc/designs/installed-architectures.md)."""
    missing = [hook for hook in GATE_HOOKS if not (framework / ".githooks" / hook).is_file()]
    if missing and raising:
        raise FileNotFoundError(
            f"{framework}/.githooks/ has no {', '.join(missing)} — the gates cannot be armed from this "
            "install. Re-run install-ai-stack.sh from a current AgentSmith checkout, or point "
            "$AGENTSMITH_DIR at one."
        )
    return missing


def install_gate_hooks(root: Path, framework: Path, prior: Optional[Path] = None, force: bool = False,
                       provisioning: bool = True) -> list[str]:
    """The gate's hooks, copied in and armed; `prior`'s hooks chained behind
    them (runtime/adopt.py chain_hooks — `provisioning=False` leaves out the
    machine's post-checkout and post-commit). Returns the paths written."""
    import subprocess as sp

    missing_gate_hooks(framework, raising=True)
    written = []
    for hook in GATE_HOOKS:
        source = framework / ".githooks" / hook
        if source.is_file() and (force or not (root / ".githooks" / hook).exists()):
            (root / ".githooks").mkdir(parents=True, exist_ok=True)
            shutil.copy(source, root / ".githooks" / hook)
            (root / ".githooks" / hook).chmod(0o755)
            written.append(f".githooks/{hook}")
    # Armed, not merely present: a hook family nobody points git at is the
    # `implemented-not-invoked` failure this whole programme is about. Armed only
    # once the hooks are there: a hooks path overrides .git/hooks, so arming an
    # empty directory leaves a repository with no hooks at all — which is what a
    # machine install without .githooks/ used to do, silently.
    sp.run(["git", "-C", str(root), "config", "core.hooksPath", ".githooks"], check=False)
    if prior is not None:
        from runtime.adopt import chain_hooks

        written += chain_hooks(root, prior, provisioning=provisioning)
    return written


def _provision_governance(root: Path, force: bool = False, design_md: Optional[str] = None,
                          session_start: Optional[str] = None, prior: Optional[Path] = None) -> list[str]:
    """The gates, armed, plus what they read: the config, the hooks, the IDE
    hook configs, the generated rule files, a committed graph and the stubs."""
    import subprocess as sp

    written: list[str] = []
    framework = _framework_dir()

    def put(rel: str, body: str) -> None:
        target = root / rel
        if target.exists() and not force:
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body, encoding="utf-8")
        written.append(rel)

    stack = "python-fastapi"
    for name in GATED_BY_STACK:
        if (root / ".github" / "workflows" / f"ci-{name}.yml").is_file():
            stack = name
            break
    put(".agenticframework/process-gates.json", _process_gates_config(stack, session_start))

    if framework is not None:
        written += install_gate_hooks(root, framework, prior=prior, force=force)

        try:
            gate_ides = gate_ides_module(framework)

            from runtime.config import chosen_ides

            for ide in chosen_ides(root, gate_ides.GENERATED):
                rel = gate_ides.ADAPTERS[ide].config_path
                existing = json.loads((root / rel).read_text(encoding="utf-8")) \
                    if (root / rel).is_file() else None
                body = json.dumps(gate_ides.render_config(ide, existing), indent=2) + "\n"
                if existing is None or force:
                    put(rel, body)
        except Exception as exc:  # a scaffold that half-works says which half
            print(f"  ! IDE hook configs not written ({exc})", file=sys.stderr)

        for script, args in (
            (framework / "scripts" / "generate-ide-config.py",
             ["--repo-root", str(root), "--rules-file", str(framework / "templates" / "agent-rules.yaml")]),
            (framework / "scripts" / "map_codebase.py", ["--quiet"]),
        ):
            if script.is_file():
                done = sp.run([sys.executable, str(script), *args], cwd=root,
                              capture_output=True, text=True, check=False)
                if done.returncode != 0:
                    print(f"  ! {script.name} did not complete: {done.stderr.strip()[:200]}", file=sys.stderr)
        graph = root / ".agent-rfc" / "fixtures" / "knowledge_graph.json"
        if not graph.is_file():
            # A repo with no code yet still gets a graph: an empty one is a
            # fact ("nothing mapped"), and a missing file is a gap the
            # governed check would report forever.
            put(".agent-rfc/fixtures/knowledge_graph.json", json.dumps(
                {"directed": True, "multigraph": False, "graph": {}, "nodes": [], "links": []},
                indent=2) + "\n")
        else:
            written.append(".agent-rfc/fixtures/knowledge_graph.json")

    put("README.md", f"# {root.name}\n\nWhat this is, and how to run it.\n")
    put("docs/PRODUCT_BACKLOG.md", "# Product backlog\n\nOne row per item: what, why, status.\n")
    if (root / "docs" / "DESIGN.md").exists() and not force and design_md is not None:
        print("  = docs/DESIGN.md exists — left untouched; its architecture section is not written")
    put("docs/DESIGN.md", design_md or "# Design\n\nThe living picture of this system.\n")
    put("docs/REVIEW_LOG.md", "# Review log\n\nAppend-only: one section per change.\n")
    return written


def _actions_dir() -> Optional[Path]:
    """Composite actions the cd-* workflows call as `./.github/actions/<name>`.

    This file's location first, then the machine install — the same order, for
    the same reason, as `_templates_dir`, and it was wrong here too. On the
    machine this was found on, ~/.agent-framework/github-actions held THREE of
    the five actions: `install-python-deps` and `rollback-notify` were added to
    the checkout and never re-installed. A tenant adopted there got workflows
    calling two actions that were not copied in, which fails at the first
    `uses:` with "Can't find 'action.yml'" — a whole-workflow rejection, not a
    step failure. CI never saw it because the scratch tenants run
    install-ai-stack.sh first, so their install is never stale.
    """
    for candidate in (
        Path(__file__).resolve().parent.parent / ".github" / "actions",
        Path.home() / ".agent-framework" / "github-actions",
    ):
        if candidate.is_dir():
            return candidate
    return None


def _copy_composite_actions(root: Path) -> list[str]:
    """`uses: ./.github/actions/<name>` resolves inside the TENANT's repo, so the
    actions must be copied in beside the workflows that call them.

    hooks/post-checkout always did this; `tenant init` wrote cd-staging.yml and
    cd-production.yml without it, and both fail at their first `uses:` with
    "Can't find 'action.yml'". Per action, never overwriting — a tenant may
    have adjusted one (KYC Sentinel diffs its gcp-auth against the framework's
    in CI precisely because they are copies).
    """
    source = _actions_dir()
    if source is None:
        print(
            "  ! no composite actions found — cd-staging.yml/cd-production.yml will fail "
            "at ./.github/actions/*. Re-run install-ai-stack.sh, or use a checkout.",
            file=sys.stderr,
        )
        return []
    written = []
    for action in sorted(p for p in source.iterdir() if p.is_dir()):
        dest = root / ".github" / "actions" / action.name
        if dest.exists():
            continue
        shutil.copytree(action, dest)
        written.append(f".github/actions/{action.name}")
    return written


# ── commands ─────────────────────────────────────────────────────────────────


def _rfc_reference(root: Path) -> Optional[str]:
    """`RFC-001` for the first RFC at `.agent-rfc/` depth 1, or None.

    hooks/commit-msg requires an RFC-NNN somewhere in the message under an
    enterprise org policy, and it greps the WHOLE message — so the reference goes
    in a trailer, clear of the 72-character subject rule. Read off disk rather
    than assumed to be 001: a tenant that already had RFCs keeps them, and the
    scaffold writes none (.agent-rfc/designs/scaffold-rfc-and-vouched-skip.md).
    """
    rfc_dir = root / ".agent-rfc"
    if not rfc_dir.is_dir():
        return None
    for path in sorted(rfc_dir.glob("*.md")):
        number = path.name.split("-", 1)[0]
        if number.isdigit():
            return f"RFC-{number}"
    return None


# What a portal intake decides, so --from refuses each of them rather than
# letting one silently win over the other.
_INTAKE_FLAGS = (("--stack", "stack"), ("--isolation", "isolation"), ("--architecture", "architecture"),
                 ("--agentic", "agentic"), ("--ide", "ide"))


def _cmd_tenant_init(args: argparse.Namespace) -> int:
    """`agentsmith tenant init` — from arguments, or from a portal intake
    (`--from`; runtime/intake.py), which is fetched and validated first and
    consumed only once the scaffold, its RFC included, has landed."""
    intake = None
    if args.from_intake is not None:
        given = [flag for flag, attr in _INTAKE_FLAGS if getattr(args, attr)]
        if args.tenant_id is not None or given:
            print(f"agentsmith: --from takes the tenant id and its options from the intake; drop "
                  f"{', '.join(([args.tenant_id] if args.tenant_id else []) + given)}", file=sys.stderr)
            return 2
        from runtime import intake as intakes

        try:
            intake = intakes.fetch(args.from_intake)
        except intakes.IntakeError as exc:
            print(f"agentsmith: {exc}", file=sys.stderr)
            return exc.exit_code
        record = intake.record
        args.tenant_id, args.stack, args.isolation = record["tenant_id"], record["stack"], record["isolation"]
        args.architecture, args.agentic, args.ide = record["architecture"], record["agentic"], record["ides"] or None
        print(f"Intake {record['intake_id']}: tenant '{args.tenant_id}' — scaffolding it here.")
    elif args.tenant_id is None:
        print("agentsmith: give the tenant's id, or --from <intake> to take it from a portal intake", file=sys.stderr)
        return 2
    args.stack = args.stack or "python-fastapi"
    args.isolation = args.isolation or "shared"

    root = Path(args.root).resolve() if args.root else Path.cwd()
    code = _scaffold_tenant(args, root, intake.record["rfc"] if intake else None)
    if code != 0 or intake is None:
        return code
    # Consume only once the intake has fully landed. The scaffold never
    # overwrites an RFC, so in a repository that already had one the author's
    # text was not written — and consuming would burn the only copy of it.
    from runtime import architectures

    expected = architectures.render_scaffold_rfc(args.tenant_id, args.stack, intake.record["rfc"])
    landed = root / SCAFFOLD_RFC
    if not (landed.is_file() and landed.read_text(encoding="utf-8") == expected):
        # --force does not help: write_scaffold_records never writes an RFC
        # beside an existing one, forced or not.
        blocking = ", ".join(p.relative_to(root).as_posix() for p in sorted(root.glob(".agent-rfc/*.md"))) or "an RFC"
        print(f"\n  ⚠️  The intake's RFC was not written: {blocking} was already here, and the scaffold never "
              f"writes an RFC beside another. Intake {intake.record['intake_id']} is left unused so its text is "
              f"not lost — move {blocking} out of .agent-rfc/, run the same command again, then merge the two.")
        return 0
    print(intake.consume())
    return 0


def _scaffold_tenant(args: argparse.Namespace, root: Path, intake_rfc: Optional[dict]) -> int:
    try:
        written = init_tenant(
            args.tenant_id,
            root,
            stack=args.stack,
            isolation=args.isolation,
            force=args.force,
            allow_framework_root=args.allow_framework_root,
            architecture=args.architecture,
            agentic=args.agentic,
            ides=args.ide,
            rfc=intake_rfc,
        )
    except FrameworkRootError as exc:
        print(f"agentsmith: refusing to scaffold here.\n{exc}", file=sys.stderr)
        return 3
    except (ValueError, FileNotFoundError) as exc:
        # FileNotFoundError: a machine install missing a template the catalogue
        # needs. It names the file and the fix; a traceback would not.
        print(f"agentsmith: {exc}", file=sys.stderr)
        return 2
    for path in written:
        print(f"  + {path}")
    print(f"\nTenant '{args.tenant_id}' scaffolded ({args.stack}, {args.isolation}"
          f"{', ' + args.architecture if args.architecture else ''}{', agentic' if args.agentic else ''}).")
    if SCAFFOLD_MANIFEST in written:
        # The RFC reference is unconditional, not enterprise-only: it is true
        # everywhere (the RFC exists), it keeps this output the same on every
        # machine, and without it an enterprise tenant's first commit is refused
        # by hooks/commit-msg even once Guardrail 4 is satisfied.
        rfc = _rfc_reference(root)
        refs = f' -m "Refs: {rfc}"' if rfc else ""
        print(
            "\nThe gates are armed, and this scaffold is their first commit. Commit it exactly as written:\n"
            f'  git add -A && git commit -m "chore: scaffold {args.tenant_id}" '
            f'-m "Design: {SCAFFOLD_DESIGN}" -m "Review: n/a: generated scaffold"{refs}\n'
            "The gate accepts that review only while every scaffolded file is unchanged; add code in the "
            "next commit, under a design of its own."
        )
    if ".agenticframework/tenant.yaml" in written:
        # Declared in docs/UserManual.md, the portal's audit route and its event
        # types since the shell-function days, and written by nothing once the
        # scaffold moved here.
        import os

        from runtime.machine.policy import audit_log_event

        audit_log_event(
            "tenant_created",
            os.environ.get("AGENT_OWNER_ID") or "unknown",
            args.tenant_id,
            {"stack": args.stack, "isolation": args.isolation},
        )
    # The generated workflows run `python3 scripts/*.py` from THIS repo, and
    # those import runtime.* and read fixtures/. Vendoring them is
    # hooks/post-checkout's job — one implementation, not a second copy here
    # to drift — and `init_tenant` ran it when this repository had it. When it
    # did not, say plainly that the scaffold is not usable yet, and how to fix
    # it BEFORE the first commit: vendored after it, the code needs a review.
    if not (root / "scripts" / "run-security-checks.py").exists():
        print(
            "\nNot done yet: scripts/, runtime/ and fixtures/ are not vendored, and every "
            "scripts/*.py step in the workflows above fails until they are.\n"
            "This repository's hooks do not include AgentSmith's post-checkout, which vendors them. "
            "Before the first commit, run `git init` here (it adds the machine's hooks and changes "
            f"nothing else), then re-run `agentsmith tenant init {args.tenant_id} ... --force`."
        )
    if args.isolation == "dedicated":
        print("Apply runtime/k8s/dedicated-tenant/ to provision its own worker pool.")
    return 0


def _cmd_tenant_adopt(args: argparse.Namespace) -> int:
    """`agentsmith tenant adopt` — runtime/adopt.py. The plan is printed first,
    and nothing is written without a yes: at a terminal, asked; elsewhere, --yes."""
    from runtime.adopt import AdoptError, adopt, commit_command, describe, plan_adoption

    root = Path(args.root).resolve() if args.root else Path.cwd()
    try:
        plan = plan_adoption(args.tenant_id, root, stack=args.stack, architecture=args.architecture,
                             agentic=args.agentic, gate=args.gate, framework_ref=args.framework_ref)
    except (ValueError, FileNotFoundError) as exc:
        # AdoptError, an unknown style, a bad tenant id, or a machine install
        # missing a template — each says what to do; none is a traceback.
        print(f"agentsmith: {exc}", file=sys.stderr)
        return 2
    print(describe(plan))
    if not args.yes:
        if not sys.stdin.isatty():
            print("\nagentsmith: not a terminal, so nothing was written — re-run with --yes to adopt",
                  file=sys.stderr)
            return 2
        if input("\nAdopt this repository? [y/N] ").strip().lower() not in ("y", "yes"):
            print("Nothing written.")
            return 1
    try:
        written = adopt(plan)
    except AdoptError as exc:
        print(f"agentsmith: {exc}", file=sys.stderr)
        return 1
    print()
    for path in written:
        print(f"  + {path}")
    from runtime.machine.policy import audit_log_event

    audit_log_event("tenant_created", os.environ.get("AGENT_OWNER_ID") or "unknown", args.tenant_id,
                    {"stack": plan.stack, "adopted": True})
    print(
        f"\nTenant '{args.tenant_id}' adopted. The gates are armed; commit what adopt wrote, exactly as written:\n"
        f"  {commit_command(written, _rfc_reference(plan.root))}\n"
        "The gate accepts that review only on this commit and only while every file still matches. From the next "
        "commit on, a change to gated code — existing code included — needs a design and a review.\n"
        "The gates workflow checks out AgentSmith with this run's own token; if AgentSmith is private, "
        "set the AGENTSMITH_READ_TOKEN repository secret (Contents: read)."
    )
    return 0


def _provider_scripts() -> Optional[Path]:
    """The scripts this provider answers with: its own installation's. The
    repository being asked about is used only when it IS the framework checkout
    — a tenant's vendored copy is the tenant's, possibly stale, and a provider
    answering with it would judge the tenant by code the tenant holds
    (.agent-rfc/designs/gate-local-events.md)."""
    here = Path.cwd()
    if looks_like_framework(here) and (here / "scripts" / "process_gate.py").is_file():
        return here / "scripts"
    framework = _framework_dir()
    if framework is not None and (framework / "scripts" / "process_gate.py").is_file():
        return framework / "scripts"
    return None


def _cmd_gate(args: argparse.Namespace) -> int:
    """`agentsmith gate <event>` — AgentSmith as a gate provider
    (contract/gate/v3/protocol.md). The neutral profile over the same decision
    path every IDE dialect goes through: one implementation of what a rule means.

    Exit 3 — the contract's "this provider cannot run here" — when the gate
    itself is not on this machine, so a caller can try the next provider."""
    import subprocess as sp

    scripts = _provider_scripts()
    if scripts is None:
        print("agentsmith gate: no process_gate.py in $AGENTSMITH_DIR or ~/.agent-framework — "
              "run install-ai-stack.sh", file=sys.stderr)
        return 3
    gate = scripts / "process_gate.py"
    if args.event == "kg":
        return _gate_kg(scripts, args.verb, args.base)
    # The dialect the caller asked for, or the contract's neutral profile. A
    # provider that serves IDE hooks accepts --ide and translates; conformance
    # pins the neutral profile (contract/gate/v1/protocol.md). `ci` is contract
    # 2's; `commit` and `push` are contract 3's — each answered by the same run
    # the hook or CI made by path (contract/gate/v3/protocol.md).
    command = {
        "ci": ["ci", "--decision"],
        "commit": ["commit-msg", "--decision"],
        "push": ["sweep", "--decision"],
    }.get(args.event, [args.event, "--ide", args.ide or "neutral"])
    done = sp.run([sys.executable, str(gate), *command],
                  input=sys.stdin.read() if not sys.stdin.isatty() else "{}",
                  capture_output=True, text=True, check=False)
    sys.stderr.write(done.stderr)
    if done.returncode == 3:
        return 3
    # The hooks say "allow" by staying silent, which a caller cannot tell from a
    # crash that printed nothing. The contract's profile is explicit, so the
    # adapter says it (contract/gate/v1/protocol.md).
    if done.returncode == 0 and not done.stdout.strip():
        print(json.dumps({"decision": "allow", "text": ""}))
        return 0
    sys.stdout.write(done.stdout)
    return done.returncode


def _gate_kg(scripts: Path, verb: Optional[str], base: Optional[str]) -> int:
    """`agentsmith gate kg build|impact` — contract 3's knowledge-graph verbs:
    build the repository's graph, or name a change's review scope as JSON
    (contract/gate/v3/protocol.md). `impact` is the commit being made unless
    `--base` names a ref — the scope the commit gate checks a review against."""
    import subprocess as sp

    if verb == "build":
        return sp.run([sys.executable, str(scripts / "map_codebase.py"), "--quiet"], check=False).returncode
    if verb == "impact":
        scope = ["--base", base] if base else ["--staged"]
        return sp.run([sys.executable, str(scripts / "local_knowledge_graph.py"), "--impact", "--json", *scope],
                      check=False).returncode
    print("agentsmith gate kg: build | impact [--staged | --base REF]", file=sys.stderr)
    return 2


def _cmd_rules(args: argparse.Namespace) -> int:
    """`agentsmith rules render [--write] | check` — AgentSmith as a rules
    provider (contract/rules/v1/protocol.md), answered by this installation's
    own scripts, never a tenant's vendored copy — as `gate` is."""
    import subprocess as sp

    scripts = _provider_scripts()
    if scripts is None or not (scripts / "rules_port.py").is_file():
        print("agentsmith rules: no rules_port.py in $AGENTSMITH_DIR or ~/.agent-framework — "
              "run install-ai-stack.sh", file=sys.stderr)
        return 3
    if args.write and args.verb != "render":
        print("agentsmith rules: --write goes with render", file=sys.stderr)
        return 2
    done = sp.run([sys.executable, str(scripts / "rules_port.py"), args.verb, *(["--write"] if args.write else [])],
                  input=sys.stdin.read() if not sys.stdin.isatty() else "{}", capture_output=True, text=True,
                  check=False)
    sys.stderr.write(done.stderr)
    sys.stdout.write(done.stdout)
    return done.returncode


def _cmd_evals_run(args: argparse.Namespace) -> int:
    """`agentsmith evals run [--suite S]` — AgentSmith as an evals provider
    (contract/evals/v1/protocol.md), answered by this installation's own
    scripts. The request is stdin; `--suite` writes it for a person at a
    terminal. Exit 0 whenever it answers: the verdict is in the scorecard."""
    import subprocess as sp

    scripts = _provider_scripts()
    if scripts is None or not (scripts / "evals_port.py").is_file():
        print("agentsmith evals: no evals_port.py in $AGENTSMITH_DIR or ~/.agent-framework — "
              "run install-ai-stack.sh", file=sys.stderr)
        return 3
    request = sys.stdin.read() if not sys.stdin.isatty() else ""
    if args.suite:
        try:
            body = json.loads(request) if request.strip() else {}
        except ValueError:
            body = None
        if not isinstance(body, dict):
            print("agentsmith evals: stdin is not a JSON request", file=sys.stderr)
            return 2
        request = json.dumps({**body, "suite": args.suite})
    # stderr is the person's report and streams as the suite runs; stdout is the scorecard.
    done = sp.run([sys.executable, str(scripts / "evals_port.py"), args.verb], input=request or "{}",
                  stdout=sp.PIPE, text=True, check=False)
    sys.stdout.write(done.stdout)
    return done.returncode


def _cmd_security(args: argparse.Namespace) -> int:
    """`agentsmith security check|redaction` — AgentSmith as a security provider
    (contract/security/v1/protocol.md), answered by this installation's own
    scripts. The request is stdin; the flags write it for a person at a
    terminal. Exit 0 whenever it answers: the verdict is in the result."""
    import subprocess as sp

    scripts = _provider_scripts()
    if scripts is None or not (scripts / "security_port.py").is_file():
        print("agentsmith security: no security_port.py in $AGENTSMITH_DIR or ~/.agent-framework — "
              "run install-ai-stack.sh", file=sys.stderr)
        return 3
    request = sys.stdin.read() if not sys.stdin.isatty() else ""
    flags = {"environment": args.environment, "emitter": args.emitter, "evidence_dir": args.evidence_dir,
             "controls": args.control or None}
    given = {key: value for key, value in flags.items() if value}
    if given:
        try:
            body = json.loads(request) if request.strip() else {}
        except ValueError:
            body = None
        if not isinstance(body, dict):
            print("agentsmith security: stdin is not a JSON request", file=sys.stderr)
            return 2
        request = json.dumps({**body, **given})
    # stderr is the person's report; stdout is the result.
    done = sp.run([sys.executable, str(scripts / "security_port.py"), args.verb], input=request or "{}",
                  stdout=sp.PIPE, text=True, check=False)
    sys.stdout.write(done.stdout)
    return done.returncode


def _security_conformance(args: argparse.Namespace) -> int:
    """`agentsmith conformance --port security --provider CMD` (contract/security/v1/protocol.md)."""
    import tempfile

    from runtime import conformance as rc

    if not args.provider:
        print("agentsmith conformance --port security: --provider COMMAND is required", file=sys.stderr)
        return 2
    with tempfile.TemporaryDirectory() as tmp:
        try:
            report = rc.run_security(args.provider, Path(tmp) / "fixture")
        except FileNotFoundError as exc:
            print(f"agentsmith: {exc}", file=sys.stderr)
            return 2
    print(report.render())
    return 0 if report.passed else 1


def _evals_conformance(args: argparse.Namespace) -> int:
    """`agentsmith conformance --port evals --provider CMD` (contract/evals/v1/protocol.md)."""
    import tempfile

    from runtime import conformance as rc

    if not args.provider:
        print("agentsmith conformance --port evals: --provider COMMAND is required", file=sys.stderr)
        return 2
    with tempfile.TemporaryDirectory() as tmp:
        try:
            report = rc.run_evals(args.provider, Path(tmp) / "fixture")
        except FileNotFoundError as exc:
            print(f"agentsmith: {exc}", file=sys.stderr)
            return 2
    print(report.render())
    return 0 if report.passed else 1


def _rules_conformance(args: argparse.Namespace) -> int:
    """`agentsmith conformance --port rules --provider CMD` (contract/rules/v1/protocol.md)."""
    import tempfile

    from runtime import conformance as rc

    if not args.provider:
        print("agentsmith conformance --port rules: --provider COMMAND is required", file=sys.stderr)
        return 2
    with tempfile.TemporaryDirectory() as tmp:
        try:
            report = rc.run_rules(args.provider, Path(tmp) / "fixture")
        except FileNotFoundError as exc:
            print(f"agentsmith: {exc}", file=sys.stderr)
            return 2
    print(report.render())
    return 0 if report.passed else 1


def _telemetry_conformance(args: argparse.Namespace) -> int:
    """`agentsmith conformance --port telemetry --export FILE | --emitter CMD`
    (contract/telemetry/v1/protocol.md)."""
    from runtime import conformance as rc

    if bool(args.export) == bool(args.emitter):
        print("agentsmith conformance --port telemetry: give --export FILE or --emitter COMMAND", file=sys.stderr)
        return 2
    try:
        report = rc.run_telemetry_export(Path(args.export)) if args.export else rc.run_telemetry_emitter(args.emitter)
    except (OSError, ValueError) as exc:
        print(f"agentsmith: {exc}", file=sys.stderr)
        return 2
    print(report.render())
    return 0 if report.passed else 1


def _record_conformance(args: argparse.Namespace) -> int:
    """`agentsmith conformance --port record --sender CMD | --receiver URL`
    (contract/record/v1/protocol.md)."""
    import tempfile

    from runtime import conformance as rc

    if bool(args.sender) == bool(args.receiver):
        print("agentsmith conformance --port record: give --sender COMMAND or --receiver URL", file=sys.stderr)
        return 2
    try:
        if args.receiver:
            token = os.environ.get(rc.RECEIVER_TOKEN_ENV, "").strip()
            if not token:
                print(f"agentsmith conformance: set {rc.RECEIVER_TOKEN_ENV} to a test token the receiver issued",
                      file=sys.stderr)
                return 2
            report = rc.run_record_receiver(args.receiver, token)
        else:
            with tempfile.TemporaryDirectory() as tmp:
                report = rc.run_record_sender(args.sender, Path(tmp) / "fixture",
                                              args.url_env or rc.SENDER_URL_ENV, args.token_env or rc.SENDER_TOKEN_ENV)
    except FileNotFoundError as exc:
        print(f"agentsmith: {exc}", file=sys.stderr)
        return 2
    print(report.render())
    return 0 if report.passed else 1


def _cmd_conformance(args: argparse.Namespace) -> int:
    """`agentsmith conformance --provider "<command>"` — does that command
    satisfy the gate contract? Run it against another platform's adapter, or
    against this one (scripts/test/test_gate_contract.py does)."""
    import tempfile

    from runtime.conformance import run

    if args.port == "record":
        return _record_conformance(args)
    if args.port == "evals":
        return _evals_conformance(args)
    if args.port == "security":
        return _security_conformance(args)
    if args.port == "rules":
        return _rules_conformance(args)
    if args.port == "telemetry":
        return _telemetry_conformance(args)
    if not args.provider:
        print("agentsmith conformance: --provider is required for the gate contract", file=sys.stderr)
        return 2
    with tempfile.TemporaryDirectory() as tmp:
        try:
            report = run(args.provider, Path(tmp) / "fixture", args.contract)
        except FileNotFoundError as exc:
            print(f"agentsmith: {exc}", file=sys.stderr)
            return 2
    print(report.render())
    return 0 if report.passed else 1


def _cmd_sync(args: argparse.Namespace) -> int:
    """`agentsmith sync` — runtime/sync.py. Brings this repository's copies of
    the framework up to date; the commit it prints passes the repository's own
    gates, because every file in it is one the framework wrote."""
    from runtime.sync import SyncError, commit_command, describe, plan_sync, sync

    root = Path(args.root).resolve() if args.root else Path.cwd()
    try:
        plan = plan_sync(root)
    except (SyncError, FileNotFoundError) as exc:
        print(f"agentsmith: {exc}", file=sys.stderr)
        return 2
    print(describe(plan))
    if not plan.stale and not plan.added and not plan.vendored:
        print("\nNothing to do.")
        return 0
    if not args.yes:
        if not sys.stdin.isatty():
            print("\nagentsmith: not a terminal, so nothing was written — re-run with --yes to sync",
                  file=sys.stderr)
            return 2
        if input("\nSync this repository? [y/N] ").strip().lower() not in ("y", "yes"):
            print("Nothing written.")
            return 1
    written = sync(plan)
    if not written:
        print("\nNothing to do.")
        return 0
    print()
    for path in written:
        print(f"  ~ {path}")
    print(f"\nSynced with AgentSmith {plan.version}. Commit it as written:\n"
          f"  {commit_command(written, plan.version)}\n"
          "The gate accepts that review while every file in the commit is one the framework wrote.")
    return 0


def _cmd_doctor(args: argparse.Namespace) -> int:
    """Delegates to verify_system, which already owns every check.

    It used to look beside this file (`<package>/../scripts/`), which exists in a
    checkout and never in an installed package — so `agentsmith doctor` could
    only ever run from a clone.
    """
    import subprocess

    from runtime.machine.state import find_script

    script = find_script("verify_system.py")
    if script is None:
        print("agentsmith: verify_system.py not found in ./scripts/ or ~/.agent-framework/scripts/", file=sys.stderr)
        return 1
    return subprocess.run([sys.executable, str(script), *args.checks], check=False).returncode


# `shellenv` was removed: it printed `export AI_STACK_MODE=…` for a profile to
# eval, and an exported mode outranks the machine's mode file for every process
# that shell starts — the opposite of what `agentsmith mode` is for.


def _cmd_mode(args: argparse.Namespace) -> int:
    from runtime.machine import ops

    return ops.mode(args.mode)


def _cmd_check(args: argparse.Namespace) -> int:
    from runtime.machine import ops

    return ops.check()


def _cmd_status(args: argparse.Namespace) -> int:
    from runtime.machine import ops

    return ops.status()


def _cmd_models(args: argparse.Namespace) -> int:
    from runtime.machine import ops

    return ops.models("judge" if args.judge else "ollama")


def _cmd_dashboard(args: argparse.Namespace) -> int:
    from runtime.machine import ops

    return ops.dashboard_start() if args.action == "start" else ops.dashboard_stop()


def _cmd_evals(args: argparse.Namespace) -> int:
    """`agentsmith evals` — sync HITL feedback from Phoenix, then run the
    scorecard; `agentsmith evals run` — the evals contract's verb."""
    if getattr(args, "verb", None) == "run":
        return _cmd_evals_run(args)
    if getattr(args, "suite", None):
        print("agentsmith evals: --suite goes with run", file=sys.stderr)
        return 2
    from runtime.machine import ops

    return ops.evals()


def _cmd_promote(args: argparse.Namespace) -> int:
    from runtime.machine import ops

    return ops.promote(args.case_id, args.query, args.output)


def _cmd_tenant_promote(args: argparse.Namespace) -> int:
    from runtime.machine import ops

    return ops.tenant_promote(args.tenant_id, args.from_env, args.to_env)


def _cmd_tenant_onprem(args: argparse.Namespace) -> int:
    from runtime.machine import ops

    return ops.onprem_scaffold()


def _cmd_upgrade(args: argparse.Namespace) -> int:
    from runtime.machine.upgrade import upgrade

    return upgrade(Path.cwd(), args.to or _default_framework_version())


def _cmd_scrub(args: argparse.Namespace) -> int:
    from runtime.machine import ops

    return ops.scrub(args.directory, args.yes)


def _cmd_uninstall(args: argparse.Namespace) -> int:
    from runtime.machine import ops

    if args.legacy_profile_only:
        return ops.remove_profile_blocks()
    return ops.uninstall(args.yes, args.purge)


def _cmd_gates_repair(args: argparse.Namespace) -> int:
    from runtime.machine import governance

    return governance.gates_repair()


def _cmd_gates_list(args: argparse.Namespace) -> int:
    from runtime.machine import governance

    return governance.gates_list()


def _cmd_gates_run(args: argparse.Namespace) -> int:
    from runtime.machine import governance

    return governance.gates_run(only=args.only, services=args.services, fail_fast=args.fail_fast,
                                allow_install=args.allow_install)


def _cmd_hooks_bypass_check(args: argparse.Namespace) -> int:
    """Exit 0 when the hooks may be bypassed. Called by the four git hooks when a
    bypass is requested and an org policy file exists — see runtime/machine/policy.py."""
    from runtime.machine.policy import bypass_decision

    decision = bypass_decision()
    print(f"AgentSmith: {decision.message}", file=sys.stderr)
    return 0 if decision.allowed else 1


def _cmd_purge_idempotency(args: argparse.Namespace) -> int:
    """Delete idempotency rows past their TTL.

    `idempotency_keys.expires_at` is only read in the lookup's WHERE clause, so
    an expired row stops being returned and never stops existing — the table
    grows by one row per gateway call. `IdempotencyStore.purge_expired` has
    existed the whole time with no caller anywhere; this is the caller, so the
    Day-2 task in docs/UserManual.md › Maintain (Day-2 Operations) can name a command instead of raw SQL.
    """
    try:
        from runtime.idempotency import IdempotencyStore

        deleted = IdempotencyStore().purge_expired()
    except Exception as exc:
        print(f"agentsmith: idempotency purge failed: {exc}", file=sys.stderr)
        return 1
    if deleted < 0:
        print("backend expires its own keys — nothing to purge")
    else:
        print(f"purged {deleted} expired idempotency key(s)")
    return 0


def _cmd_version(args: argparse.Namespace) -> int:
    try:
        from importlib.metadata import version

        print(version("agentsmith-runtime"))
    except Exception:
        print("unknown (not installed as a package)")
    return 0


def governance_designs_dir() -> str:
    from runtime.machine.governance import DESIGNS_DIR

    return DESIGNS_DIR


def _cmd_approve(args: argparse.Namespace) -> int:
    from runtime.machine import governance

    return governance.approve(args.design, args.deviation, args.statement)


def _cmd_design_new(args: argparse.Namespace) -> int:
    from runtime.machine import governance

    return governance.design_new(args.slug, args.scope)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agentsmith", description="AgentSmith framework CLI"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    tenant = sub.add_parser("tenant", help="tenant scaffolding").add_subparsers(
        dest="tenant_command", required=True
    )
    init = tenant.add_parser("init", help="scaffold .agenticframework/ and CI workflows")
    init.add_argument("tenant_id", nargs="?", default=None,
                      help="the tenant's id — or omit it and give --from, whose intake names it")
    init.add_argument(
        "--from",
        dest="from_intake",
        default=None,
        metavar="INTAKE",
        help="scaffold from a portal intake: its stack, options, IDEs and first RFC. Reads "
        "AGENTSMITH_PORTAL_URL, and AGENTSMITH_INTAKE_TOKEN (asked for at a terminal when unset)",
    )
    # Defaults of None, applied in _cmd_tenant_init, so that a flag the author
    # actually typed can be told from one they did not — which --from needs to
    # refuse rather than silently override.
    init.add_argument("--stack", default=None, choices=list(STACKS), help="default: python-fastapi")
    init.add_argument("--isolation", default=None, choices=list(ISOLATIONS), help="default: shared")
    init.add_argument(
        "--architecture",
        default=None,
        metavar="STYLE",
        help="structural style: layered, modular-monolith, hexagonal, microservice, event-driven "
        "(also clean, ports-and-adapters, onion, n-tier)",
    )
    init.add_argument(
        "--agentic",
        action="store_true",
        help="add the agent layer: agents, allowlisted tools, the model gateway, durable workflows, evals",
    )
    init.add_argument("--root", default=None, help="target repo (default: cwd)")
    init.add_argument(
        "--ide",
        action="append",
        default=None,
        metavar="NAME",
        help="write a hook config for this IDE only (repeatable; default: every IDE "
        "with a verified config schema). Recorded as workspace.ides in tenant.yaml, "
        "so `agentsmith sync` keeps the choice.",
    )
    init.add_argument("--force", action="store_true", help="overwrite existing files")
    init.add_argument(
        "--allow-framework-root",
        action="store_true",
        help="scaffold even where the framework's own checkout is detected",
    )
    init.set_defaults(func=_cmd_tenant_init)

    adopt = tenant.add_parser("adopt", help="bring an existing repository under the gates, keeping what it has")
    adopt.add_argument("tenant_id")
    adopt.add_argument("--stack", default=None, choices=list(STACKS), help="default: detected")
    adopt.add_argument("--architecture", default=None, metavar="STYLE",
                       help="the structural style the code moves towards, recorded in docs/DESIGN.md")
    adopt.add_argument("--agentic", action="store_true", help="record the agent layer too")
    adopt.add_argument("--gate", action="append", default=None, metavar="GLOB",
                       help="a path to gate (repeatable); replaces what detection found")
    adopt.add_argument("--framework-ref", default=None,
                       help="the AgentSmith tag the gates workflow runs (default: this install's version)")
    adopt.add_argument("--root", default=None, help="target repo (default: cwd)")
    adopt.add_argument("--yes", action="store_true", help="adopt without asking (required off a terminal)")
    adopt.set_defaults(func=_cmd_tenant_adopt)

    sync_cmd = sub.add_parser("sync", help="bring this repository's copies of the framework up to date")
    sync_cmd.add_argument("--root", default=None, help="the repository (default: cwd)")
    sync_cmd.add_argument("--yes", action="store_true", help="sync without asking (required off a terminal)")
    sync_cmd.set_defaults(func=_cmd_sync)

    gate = sub.add_parser("gate", help="answer a gate event (contract/gate/v3 — the neutral profile)")
    gate.add_argument("event", choices=("session-start", "pre-edit", "stop", "ci", "commit", "push", "kg"))
    gate.add_argument("verb", nargs="?", choices=("build", "impact"), help="for kg: build or impact")
    gate.add_argument("--staged", action="store_true", help="kg impact: the commit being made (the default)")
    gate.add_argument("--base", default=None, metavar="REF", help="kg impact: diff the working tree against REF")
    gate.add_argument("--ide", default=None,
                      help="the dialect the payload is in (default: the contract's neutral profile)")
    gate.set_defaults(func=_cmd_gate)

    rules = sub.add_parser("rules", help="render or check the rule files agents read (contract/rules/v1)")
    rules.add_argument("verb", choices=("render", "check"))
    rules.add_argument("--write", action="store_true",
                       help="render: place the files here instead of printing them (not a contract verb)")
    rules.set_defaults(func=_cmd_rules)

    conformance = sub.add_parser("conformance", help="does a command or a portal satisfy a contract?")
    conformance.add_argument("--port", choices=("gate", "record", "rules", "telemetry", "evals", "security"),
                             default="gate",
                             help="the contract: the gate, the rules, the evals or security (a provider command), the "
                                  "record (a sender or a receiver), or telemetry (an emitter or an export)")
    conformance.add_argument("--export", metavar="FILE",
                             help="telemetry: an OTLP/JSON export to judge — one object, or one per line")
    conformance.add_argument("--emitter", metavar="COMMAND",
                             help="telemetry: a command to run with its OTLP export pointed at a loopback receiver")
    conformance.add_argument("--provider", metavar="COMMAND",
                             help='the provider to test, e.g. "agentsmith gate" or "agentsmith rules"')
    conformance.add_argument("--sender", metavar="COMMAND",
                             help="record: a gate provider that sends records, run against a loopback receiver")
    conformance.add_argument("--receiver", metavar="URL",
                             help="record: a portal's ingest address; the token is read from GOVERNANCE_RECORD_TOKEN")
    conformance.add_argument("--url-env", default=None, metavar="NAME",
                             help="record --sender: the variable the provider reads its receiver from")
    conformance.add_argument("--token-env", default=None, metavar="NAME",
                             help="record --sender: the variable the provider reads its token from")
    conformance.add_argument("--contract", type=int, default=1, choices=(1, 2, 3),
                             help="the gate contract version to score against (2 adds ci; 3 commit, push and kg)")
    conformance.set_defaults(func=_cmd_conformance)

    promote_tenant = tenant.add_parser("promote", help="gate on staging evals, then open the develop → main PR")
    promote_tenant.add_argument("tenant_id")
    promote_tenant.add_argument("--from", dest="from_env", required=True)
    promote_tenant.add_argument("--to", dest="to_env", required=True)
    promote_tenant.set_defaults(func=_cmd_tenant_promote)

    onprem = tenant.add_parser("onprem-scaffold", help="copy the on-prem deployment template into ./deploy/onprem/")
    onprem.set_defaults(func=_cmd_tenant_onprem)

    doctor = sub.add_parser("doctor", help="run verify_system checks")
    doctor.add_argument("checks", nargs="*", help="e.g. --check-kg --check-hooks")
    doctor.set_defaults(func=_cmd_doctor)

    mode = sub.add_parser("mode", help="set the machine's mode (no argument: print it)")
    mode.add_argument("mode", nargs="?", choices=["local", "hybrid", "off"])
    mode.set_defaults(func=_cmd_mode)

    sub.add_parser("check", help="health check: Phoenix, Ollama or API keys, unresolved log entries").set_defaults(
        func=_cmd_check
    )
    sub.add_parser("status", help="mode, hooks, Phoenix, judge, owner, network").set_defaults(func=_cmd_status)

    models = sub.add_parser("models", help="the models the merged registry routes to")
    which = models.add_mutually_exclusive_group()
    which.add_argument("--ollama", action="store_true", help="Ollama ids, space-separated (default)")
    which.add_argument("--judge", action="store_true", help="the judge model in effect")
    models.set_defaults(func=_cmd_models)

    dashboard = sub.add_parser("dashboard", help="start or stop Phoenix (the shared Docker stack when present)")
    dashboard.add_argument("action", choices=["start", "stop"])
    dashboard.set_defaults(func=_cmd_dashboard)

    evals = sub.add_parser("evals", help="sync HITL feedback from Phoenix, then run the eval scorecard; "
                                         "`evals run` judges one suite (contract/evals/v1)")
    evals.add_argument("verb", nargs="?", choices=("run",))
    evals.add_argument("--suite", choices=("golden", "fairness", "hallucination", "adversarial", "rag_poison"),
                       help="run: the suite to judge, when no request comes on stdin")
    evals.set_defaults(func=_cmd_evals)

    security = sub.add_parser("security", help="check this repository's security pack and posture, or what its "
                                               "telemetry carries on the wire (contract/security/v1)")
    security.add_argument("verb", choices=("check", "redaction"))
    security.add_argument("--control", action="append", metavar="ID",
                          help="check: only this control (repeatable)")
    security.add_argument("--evidence-dir", metavar="DIR", help="check: write the evidence pack here")
    security.add_argument("--environment", choices=("staging", "production"),
                          help="redaction: the profile to check (required for redaction)")
    security.add_argument("--emitter", metavar="COMMAND",
                          help="redaction: the emitter to run, instead of the one providers.json declares")
    security.set_defaults(func=_cmd_security)

    promote = sub.add_parser("promote", help="promote a fix to the golden dataset and re-run evals")
    promote.add_argument("case_id")
    promote.add_argument("query")
    promote.add_argument("output")
    promote.set_defaults(func=_cmd_promote)

    upgrade = sub.add_parser("upgrade", help="refresh this vendored tenant's framework copy and commit it")
    upgrade.add_argument("--to", default=None, help="version to declare (default: the installed release)")
    upgrade.set_defaults(func=_cmd_upgrade)

    scrub = sub.add_parser("scrub", help="delete generated IDE-rule files under a directory, after listing them")
    scrub.add_argument("directory", nargs="?", default=None)
    scrub.add_argument("--yes", action="store_true", help="do not ask for confirmation")
    scrub.set_defaults(func=_cmd_scrub)

    uninstall = sub.add_parser("uninstall", help="restore git config and remove the command from this machine")
    uninstall.add_argument("--yes", action="store_true", help="do not ask for confirmation")
    uninstall.add_argument("--purge", action="store_true", help="also remove ~/.agent-framework and ~/.git_templates")
    uninstall.add_argument(
        "--legacy-profile-only", action="store_true",
        help="only remove the shell-function block older installs appended to a shell profile",
    )
    uninstall.set_defaults(func=_cmd_uninstall)

    gates = sub.add_parser("gates", help="the local gates").add_subparsers(dest="gates_command", required=True)
    gates.add_parser(
        "repair", help="list commits that never passed the gate, and how to bring them under one"
    ).set_defaults(func=_cmd_gates_repair)
    gates.add_parser(
        "list", help="the gates this repo's CI declares (the `# agentsmith:gate` steps)"
    ).set_defaults(func=_cmd_gates_list)
    gates_run = gates.add_parser(
        "run", help="run this repo's CI gates here, and say which ones could not run"
    )
    gates_run.add_argument("--only", default=None, help="only the gates whose name contains this text")
    gates_run.add_argument(
        "--services", action="store_true",
        help="the service containers CI starts (a database, a broker) are running here",
    )
    gates_run.add_argument("--fail-fast", action="store_true", help="stop at the first failure")
    gates_run.add_argument(
        "--allow-install", action="store_true",
        help="also run the dependency-install lines CI needs (they change this environment; "
        "by default they are dropped and named)",
    )
    gates_run.set_defaults(func=_cmd_gates_run)

    hooks = sub.add_parser("hooks", help="internal: called by the git hooks").add_subparsers(
        dest="hooks_command", required=True
    )
    hooks.add_parser("bypass-check", help="exit 0 when the org policy allows a hook bypass").set_defaults(
        func=_cmd_hooks_bypass_check
    )

    purge = sub.add_parser(
        "purge-idempotency", help="delete idempotency rows past their TTL"
    )
    purge.set_defaults(func=_cmd_purge_idempotency)

    approve = sub.add_parser(
        "approve",
        help="record the owner's approval of one deviation in a design (asks at the terminal)",
    )
    approve.add_argument("design", help=f"e.g. {governance_designs_dir()}/<slug>.md")
    approve.add_argument("deviation", help="the id in the design's '## Deviations' section, e.g. D1")
    approve.add_argument("--statement", default=None, help="one line on why (asked for if omitted)")
    approve.set_defaults(func=_cmd_approve)

    design = sub.add_parser("design", help="design records").add_subparsers(dest="design_command", required=True)
    design_new = design.add_parser("new", help="write the design skeleton the process gate asks for")
    design_new.add_argument("slug")
    design_new.add_argument("--scope", action="append", default=[],
                            help="path or glob this change may touch (repeatable)")
    design_new.set_defaults(func=_cmd_design_new)

    ver = sub.add_parser("version", help="installed framework version")
    ver.set_defaults(func=_cmd_version)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    # `doctor` forwards verify_system's own flags (`--check-kg`). argparse will
    # not hand an option-looking token to a positional, so `agentsmith doctor
    # --check-kg` failed with "unrecognized arguments" — for as long as the
    # command existed. Everything else still rejects unknown arguments.
    args, extra = parser.parse_known_args(argv)
    if extra:
        if args.command != "doctor":
            parser.error(f"unrecognized arguments: {' '.join(extra)}")
        args.checks = [*args.checks, *extra]
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
