"""
runtime/machine/ops.py — the operator commands `agentsmith` runs: mode, check,
status, models, dashboard, evals, promote, tenant promote, on-prem scaffold,
scrub and uninstall.

Each was a shell function in `~/.zshrc`; the docstring of each names what
changed in the port and why. Commands print for a human and return an exit
code; runtime/cli.py wires them to argparse.
"""

from __future__ import annotations

import importlib
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from types import ModuleType
from typing import Callable, Mapping, Optional, Sequence

from runtime.machine import state
from runtime.machine.policy import (
    approvers_text,
    audit_log_event,
    bypass_decision,
    org_policy_hooks,
    org_policy_path,
)
from runtime.machine.state import find_script, framework_home

Out = Callable[[str], None]

DEFAULT_PHOENIX = "http://localhost:6006"
OLLAMA_TAGS = "http://localhost:11434/api/tags"


def _env(env: Optional[Mapping[str, str]]) -> Mapping[str, str]:
    return os.environ if env is None else env


def phoenix_endpoint(env: Optional[Mapping[str, str]] = None) -> str:
    return (_env(env).get("AGENT_PHOENIX_ENDPOINT") or DEFAULT_PHOENIX).rstrip("/")


def effective_mode(env: Optional[Mapping[str, str]] = None) -> Optional[str]:
    """AI_STACK_MODE from the environment, else the machine's mode file."""
    explicit = (_env(env).get("AI_STACK_MODE") or "").strip()
    return explicit or state.read_mode(env)


def _http_ok(url: str, timeout: float = 3.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return 200 <= response.status < 400
    except urllib.error.HTTPError as exc:
        return 300 <= exc.code < 400
    except Exception:
        return False


def _git_global(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "config", "--global", *args], capture_output=True, text=True, check=False)


def _shared() -> Optional[ModuleType]:
    """The machine's `scripts/_shared.py` — the registry lookups it owns.

    Imported rather than re-implemented: judge and model resolution is
    cwd-aware (a tenant's models.yaml overrides the framework's) and has a
    precedence of its own that must not be rebuilt here.
    """
    for scripts in (framework_home() / "scripts", Path(os.environ.get("AGENTSMITH_DIR", "")) / "scripts"):
        if (scripts / "_shared.py").is_file():
            if str(scripts) not in sys.path:
                sys.path.insert(0, str(scripts))
            try:
                return importlib.import_module("_shared")
            except Exception:
                return None
    return None


def required_ollama_models() -> Optional[list[str]]:
    """The Ollama ids the merged registry routes to; None when it cannot be read."""
    shared = _shared()
    if shared is None or shared.load_registry() is None:
        return None
    return list(shared.provider_models("ollama"))


def judge_model() -> Optional[str]:
    shared = _shared()
    try:
        return shared.judge_model() if shared else None
    except Exception:
        return None


def _run_script(name: str, args: Sequence[str], out: Out, *, quiet: bool = False) -> int:
    """Run a framework script with this environment's interpreter — see find_script."""
    script = find_script(name)
    if script is None:
        out(f"❌ {name} not found in ./scripts/ or {framework_home() / 'scripts'} — run install-ai-stack.sh")
        return 127
    cmd = [sys.executable, str(script), *args]
    if quiet:
        return subprocess.run(cmd, capture_output=True, text=True, check=False).returncode
    return subprocess.run(cmd, check=False).returncode


# ── mode ────────────────────────────────────────────────────────────────────


def mode(target: Optional[str], out: Out = print, env: Optional[Mapping[str, str]] = None) -> int:
    """`agentsmith mode [local|hybrid|off]`.

    Writes `state/mode`, which the gateway and the hooks read — an export used
    to reach one terminal. `init.templateDir` is re-linked (local/hybrid) or
    unset (off) only on a developer install: the shell functions did it on an
    enterprise install too, which promises never to touch git's global config.
    """
    if target is None:
        current = effective_mode(env)
        source = "environment" if (_env(env).get("AI_STACK_MODE") or "").strip() else "state file"
        out(f"{current} ({source})" if current else "not set — the registry's default profile applies")
        return 0

    developer = state.read_install_mode(env) == "developer"
    if target == "off":
        decision = bypass_decision(env)
        if not decision.allowed:
            out(f"🛑 {decision.message}")
            return 1
        state.write_mode("disabled", env)
        if developer:
            _git_global("--unset", "init.templateDir")
        out("🔒 AI Stack: DISABLED — hooks muted" + (", templates unlinked" if developer else ""))
        if decision.message != "no org policy on this machine":
            out(f"⚠️  {decision.message}")
        return 0

    state.write_mode(target, env)
    if developer:
        _git_global("init.templateDir", str(Path.home() / ".git_templates"))
    banner = {"local": "🍃 LOCAL OFFLINE mode (Ollama)", "hybrid": "💎 HYBRID CLOUD mode (Claude + cost routing)"}
    out(f"AI Stack: {banner[target]} — recorded for every process on this machine")
    if (_env(env).get("AI_STACK_MODE") or "").strip() not in ("", target):
        out(f"⚠️  AI_STACK_MODE={_env(env)['AI_STACK_MODE']} is exported in this shell and still wins here.")
    check(out, env)
    return 0


# ── check / status / models ─────────────────────────────────────────────────


def _unresolved_log_entries(log: Path) -> list[str]:
    entries: list[str] = []
    try:
        lines = log.read_text(encoding="utf-8").splitlines()
    except OSError:
        return entries
    for line in lines:
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(e, dict) and e.get("level") in ("MAJOR", "CRITICAL") and not e.get("hitl_resolved", True):
            entries.append(
                f"   [{e['level']}] {e.get('timestamp', '')}  {e.get('event', '')}  "
                f"({e.get('agent', '')} / {e.get('project', '')})"
            )
    return entries


def check(out: Out = print, env: Optional[Mapping[str, str]] = None) -> int:
    """`agentsmith check` — Phoenix, the engine for the mode, unresolved log entries.

    A registry that cannot be read is reported as such. The shell function
    fell back to a hand-kept model list, which had already drifted once and
    told users to pull models nothing routes to.
    """
    environ = _env(env)
    current = effective_mode(env)
    out(f"🩺 AgentSmith Health Check — Mode: [{current or 'not set'}]")
    failed = False

    endpoint = phoenix_endpoint(env)
    if _http_ok(endpoint):
        out(f"   ✅ [TRACER]  Phoenix is live at {endpoint}")
    else:
        out("   ⚠️  [TRACER]  Phoenix is offline. Run: agentsmith dashboard start")
        failed = True

    if current == "disabled":
        out("   ⚠️  [HOOKS]   mode is disabled — hooks muted. Run: agentsmith mode local")
        failed = True
    elif current in (None, "local"):
        try:
            with urllib.request.urlopen(OLLAMA_TAGS, timeout=3) as response:
                names = {m.get("name", "") for m in json.load(response).get("models", [])}
            out("   ✅ [ENGINE]  Ollama daemon responding")
            required = required_ollama_models()
            if required is None:
                out("   ⚠️  [MODEL]   models.yaml could not be read — required models not checked")
                failed = True
            for model in required or []:
                # Exact id, or the same id with an implicit ":latest" — never a substring.
                if model in names or f"{model}:latest" in names:
                    out(f"   ✅ [MODEL]   {model} loaded")
                else:
                    out(f"   ⚠️  [MODEL]   {model} not found — run: ollama pull {model}")
                    failed = True
        except Exception:
            out("   ❌ [ENGINE]  Ollama is offline — run: ollama serve")
            failed = True
    elif current == "hybrid":
        for key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
            if environ.get(key):
                out(f"   ✅ [CLOUD]   {key.split('_')[0].title()} key present")
            else:
                out(f"   ⚠️  [CLOUD]   {key} not set")
                failed = True

    unresolved = _unresolved_log_entries(Path(".agent-history.log"))
    if unresolved:
        out(f"   🔴 Unresolved MAJOR/CRITICAL issues: {len(unresolved)}")
        for line in unresolved:
            out(line)
        out("   → Run 'agentsmith promote' or resolve via Phoenix UI.")
        failed = True

    if environ.get("OPS_PORTAL_URL"):
        if _run_script("sync-portal-history.py", [], out, quiet=True) != 0:
            out("   ⚠️  [PORTAL]  history sync to OPS_PORTAL_URL failed")

    out("   🎉 All checks passed — environment ready" if not failed
        else "   🛑 Health check failed — resolve issues above before running agents")
    return 1 if failed else 0


def _online() -> bool:
    try:
        with socket.create_connection(("1.1.1.1", 53), timeout=1):
            return True
    except OSError:
        return False


def status(out: Out = print, env: Optional[Mapping[str, str]] = None) -> int:
    environ = _env(env)
    current = effective_mode(env)
    out("────────────────────────────────────────────────")
    out(f"  Mode:      {current or 'not set'}")
    requested = current == "disabled" or environ.get("DISABLE_AI_STACK") == "true"
    if not requested:
        hooks = "on"
    elif org_policy_path().exists():
        # The hooks ask the policy per run; `status` does not, so as not to
        # write an audit event just for looking.
        hooks = "bypass requested — the org policy decides on each hook run"
    else:
        hooks = "muted"
    out(f"  Hooks:     {hooks}")
    out(f"  Install:   {state.read_install_mode(env)}  ({framework_home()})")
    out(f"  Phoenix:   {phoenix_endpoint(env)}")
    out(f"  Judge:     {judge_model() or 'unresolved'}   (from models.yaml, not the environment)")
    out(f"  Owner:     {environ.get('AGENT_OWNER_ID') or 'tenant.yaml tenant.owner, else git config user.email'}")
    out(f"  Network:   {'🌐 ONLINE' if _online() else '❌ OFFLINE (local fallback armed)'}")
    out("────────────────────────────────────────────────")
    return 0


def models(which: str, out: Out = print) -> int:
    """`agentsmith models --ollama` (for `ollama pull $(…)`) or `--judge`."""
    if which == "judge":
        judge = judge_model()
        if not judge:
            out("agentsmith: the judge model could not be resolved from models.yaml")
            return 1
        out(judge)
        return 0
    required = required_ollama_models()
    if required is None:
        out("agentsmith: models.yaml could not be read")
        return 1
    out(" ".join(required))
    return 0


# ── dashboard ───────────────────────────────────────────────────────────────


def _phoenix_pid_file(env: Optional[Mapping[str, str]] = None) -> Path:
    return state.state_dir(env) / "phoenix.pid"


def dashboard_start(out: Out = print, env: Optional[Mapping[str, str]] = None) -> int:
    """`agentsmith dashboard start` — the shared Docker stack, else standalone Phoenix.

    Records the endpoint in `state/phoenix-endpoint`, which runtime/otlp.py reads
    after the environment: the shell function exported OTEL_EXPORTER_OTLP_ENDPOINT
    into one terminal.
    """
    environ = _env(env)
    compose = framework_home() / "observability" / "docker-compose.yml"
    opted_out = Path(".agenticframework/no-shared-infra").is_file()
    endpoint = phoenix_endpoint(env)
    if opted_out:
        out("ℹ  This repo has .agenticframework/no-shared-infra — using a standalone Phoenix, not the shared stack.")

    if shutil.which("docker") and compose.is_file() and not opted_out:
        out("📊 Starting the standing stack (Phoenix + Postgres + Ops Portal)...")
        rc = subprocess.run(["docker", "compose", "up", "-d"], cwd=compose.parent, check=False).returncode
        if rc != 0:
            out("❌ docker compose up failed — see the output above")
            return rc
        state.write_phoenix_endpoint(endpoint, env)
        out(f"🚀 Phoenix    → open {endpoint}")
        out(f"🚀 Ops Portal → open http://localhost:{environ.get('OPS_PORTAL_PORT', '3000')}")
        out("   (shared across every repo on this machine — opt out per repo with:")
        out("    touch .agenticframework/no-shared-infra)")
        return 0

    uvx = shutil.which("uvx")
    if not uvx:
        out("❌ Standalone Phoenix needs uv (uvx): brew install uv — or install Docker for the shared stack.")
        return 1
    cmd = [uvx, "--from", "arize-phoenix", "phoenix", "serve", "--port", environ.get("AGENT_PHOENIX_PORT", "6006")]
    if environ.get("AGENT_PHOENIX_DB_URL"):
        cmd += ["--database-url", environ["AGENT_PHOENIX_DB_URL"]]
    log = framework_home() / "logs" / "phoenix.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    out(f"📊 Starting Arize Phoenix at {endpoint} (standalone — no Docker stack)...")
    with log.open("ab") as fh:
        proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
    pid_file = _phoenix_pid_file(env)
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text(f"{proc.pid}\n", encoding="utf-8")
    state.write_phoenix_endpoint(endpoint, env)
    out(f"🚀 Dashboard starting → open {endpoint}  (log: {log})")
    return 0


def dashboard_stop(out: Out = print, env: Optional[Mapping[str, str]] = None) -> int:
    compose = framework_home() / "observability" / "docker-compose.yml"
    state.clear_phoenix_endpoint(env)
    if shutil.which("docker") and compose.is_file():
        out("🔒 Stopping the standing stack (Phoenix + Postgres + Ops Portal)...")
        # No -v: named volumes (traces, portal data) survive a stop.
        rc = subprocess.run(["docker", "compose", "down"], cwd=compose.parent, check=False).returncode
        out("✅ Stack offline (data preserved — 'docker compose down -v' there to wipe it)" if rc == 0
            else "❌ docker compose down failed — see the output above")
        return rc

    out("🔒 Stopping Phoenix...")
    pid_file = _phoenix_pid_file(env)
    try:
        pid = int(pid_file.read_text(encoding="utf-8").strip())
        # A pid file outlives a reboot, and the number is reused: signal the
        # group only if that process is still the Phoenix this command started.
        command = subprocess.run(["ps", "-p", str(pid), "-o", "command="], capture_output=True, text=True,
                                 check=False).stdout
        if "phoenix" not in command:
            raise ProcessLookupError(pid)
        os.killpg(pid, signal.SIGTERM)  # the whole session: uvx and the server it spawned
    except (OSError, ValueError):
        # No pid file, or the process is gone. A Phoenix started by the old
        # shell function has no pid file at all.
        for pattern in ("phoenix serve", "phoenix.server.main"):
            subprocess.run(["pkill", "-f", pattern], capture_output=True, check=False)
    pid_file.unlink(missing_ok=True)
    out("✅ Dashboard offline")
    return 0


# ── evals / promote ─────────────────────────────────────────────────────────


def evals(out: Out = print, env: Optional[Mapping[str, str]] = None) -> int:
    if not _http_ok(phoenix_endpoint(env)):
        out("🔄 Phoenix offline — starting dashboard first...")
        if dashboard_start(out, env) == 0:
            time.sleep(2)
    out("🔄 Syncing HITL feedback from Phoenix UI...")
    if _run_script("sync-ui-feedback.py", [], out, quiet=True) != 0:
        out("⚠️  HITL feedback sync failed — scoring without it")
    out("🎯 Running eval scorecard...")
    return _run_script("run-evals.py", [], out)


def promote(case_id: str, query: str, output: str, out: Out = print, env: Optional[Mapping[str, str]] = None) -> int:
    rc = _run_script("promote-learning.py", [case_id, query, output], out)
    if rc != 0:
        out("❌ promote-learning.py failed — evals not re-run")
        return rc
    out("🔄 Re-running evals to validate fix...")
    return evals(out, env)


# ── tenant lifecycle ────────────────────────────────────────────────────────


def _tenant_yaml_id(path: Path) -> Optional[str]:
    try:
        import yaml

        doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return None
    value = (doc.get("tenant") or {}).get("id") if isinstance(doc, dict) else None
    return None if value is None else str(value)


def tenant_promote(tenant_id: str, from_env: str, to_env: str, out: Out = print) -> int:
    """`agentsmith tenant promote <id> --from staging --to production`.

    The id is compared with the PARSED `tenant.id` — the shell function had
    moved off substring matching already; YAML parsing also reads a quoted id
    (which `tenant init` writes) as the id rather than as `"acme"`.
    """
    if (from_env, to_env) != ("staging", "production"):
        out("❌ Only staging → production promotion is supported (no cross-tenant or cross-stage jumps)")
        return 1
    tenant_yaml = Path(".agenticframework/tenant.yaml")
    if not tenant_yaml.is_file():
        out("❌ No .agenticframework/tenant.yaml in current repo — run from the tenant repo root")
        return 1
    actual = _tenant_yaml_id(tenant_yaml)
    if actual != tenant_id:
        out(f"❌ tenant.yaml id ('{actual}') does not match '{tenant_id}' — "
            "promotion is always within the same tenant repo")
        return 1

    out("🔎 Verifying staging eval gate...")
    if _run_script("run-evals.py", ["--fail-below", "0.75"], out) != 0:
        out("🛑 Staging eval gate failed — promotion blocked")
        return 1
    out("✅ Staging eval gate passed")

    if not shutil.which("gh"):
        out("❌ gh CLI required to open the develop → main promotion PR")
        return 1
    out(f"🚀 Opening promotion PR: develop → main for tenant '{tenant_id}'...")
    rc = subprocess.run(
        [
            "gh", "pr", "create",
            "--title", f"promote({tenant_id}): staging → production",
            "--body", "Auto-generated by `agentsmith tenant promote`. Staging eval gate passed. "
                      "Requires review approval before merge "
                      "(see docs/DESIGN.md › Per-Tenant Lifecycle and Promotion).",
            "--base", "main", "--head", "develop",
        ],
        check=False,
    ).returncode
    if rc != 0:
        return rc
    audit_log_event("hitl_promotion", os.environ.get("AGENT_OWNER_ID") or "unknown", tenant_id,
                    {"from": from_env, "to": to_env})
    return 0


def onprem_scaffold(out: Out = print) -> int:
    """Copy the on-prem deployment template into ./deploy/onprem/ (opt-in).

    Its audit event was `onprem_deploy_scaffolded`, which the portal's audit
    endpoint rejects (not an AUDIT_EVENT_TYPE) — every one landed in the local
    fallback log as a failed write. It is a `config_change` now.
    """
    if not Path(".git").exists():
        out("❌ Not a git repository — run inside the tenant repo root")
        return 1
    src = framework_home() / "templates" / "onprem-deploy"
    if not src.is_dir():
        out(f"❌ No onprem-deploy template at {src} — run install-ai-stack.sh first")
        return 1
    dest = Path("deploy/onprem")
    if dest.exists():
        out(f"⚠️  {dest} already exists — leaving untouched. Delete it first to re-scaffold.")
        return 1
    shutil.copytree(src, dest)
    audit_log_event("config_change", os.environ.get("AGENT_OWNER_ID") or "unknown", Path.cwd().name,
                    {"action": "onprem_deploy_scaffolded"})
    out(f"🏗  Scaffolded on-prem deployment template at {dest}/")
    out(f"   Next: cp {dest}/.env.example {dest}/.env, set APP_IMAGE_PROD, then")
    out(f"   {dest}/scripts/up.sh — see {dest}/README.md for the canary/shadow/")
    out(f"   air-gapped bundling workflow and {dest}/kubernetes/README.md for Helm.")
    return 0


# ── maintenance ─────────────────────────────────────────────────────────────


def _confirm(prompt: str, yes: bool) -> bool:
    if yes:
        return True
    try:
        return input(prompt).strip().lower() in ("y", "yes")
    except EOFError:
        return False


# (name, is_dir, max_depth) — the generated IDE-rule artefacts. Depth counts
# like `find -maxdepth`: the target directory itself is depth 0.
SCRUB_TARGETS = (
    (".cursorrules", False, 3),
    ("CLAUDE.md", False, 3),
    ("AGENTS.md", False, 3),
    ("GEMINI.md", False, 3),
    (".github/copilot-instructions.md", False, 4),  # the file only, never .github/
    (".agents", True, 3),
)


def scrub_matches(target: Path) -> list[Path]:
    matches: list[Path] = []
    base_depth = len(target.parts)
    for root, dirs, files in os.walk(target):
        here = Path(root)
        depth = len(here.parts) - base_depth
        for name, is_dir, max_depth in SCRUB_TARGETS:
            if "/" in name:
                parent, leaf = name.split("/")
                if here.name == parent and leaf in files and depth + 1 <= max_depth:
                    matches.append(here / leaf)
            elif is_dir and name in dirs and depth + 1 <= max_depth:
                matches.append(here / name)
            elif not is_dir and name in files and depth + 1 <= max_depth:
                matches.append(here / name)
        if depth + 1 >= 4:
            dirs[:] = []
        else:
            dirs[:] = [d for d in dirs if d != ".agents"]  # listed whole, not descended
    return sorted(set(matches))


def scrub(directory: Optional[str], yes: bool, out: Out = print) -> int:
    """List every exact path first, then ask — a confirmation naming only the
    top directory (say `$HOME`) gave no idea it reached every sibling project."""
    target = Path(directory) if directory else Path.cwd()
    if not target.is_dir():
        out(f"❌ Directory not found: {target}")
        return 1
    matches = scrub_matches(target)
    if not matches:
        out(f"✨ Nothing to scrub under {target}")
        return 0
    out("🧹 WARNING: This will permanently delete the following paths:")
    for path in matches:
        out(f"   {path}")
    if not _confirm(f"   Confirm deletion of the {len(matches)} path(s) above? (y/n): ", yes):
        out("❌ Cancelled")
        return 1
    for path in matches:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink(missing_ok=True)
        out(f"   removed: {path}")
    out("✨ Scrub complete — framework re-provisions on next git init")
    return 0


PROFILE_FILES = (".zshrc", ".bashrc", ".profile")
_BLOCK_START = re.compile(r"^# >>> (AgentSmith|AgenticFramework) managed block")


def remove_profile_blocks(home: Optional[Path] = None, out: Out = print) -> int:
    """Remove the managed shell-function block the installer used to append.

    Markers are matched as whole lines: the block itself contains the marker
    text inside ai-stack-uninstall's sed pattern, and an unanchored end match
    stopped THERE, leaving the rest of the block behind. The blank line the
    installer wrote before the block goes with it (each `--force` re-install
    used to leave one more). A block with a start and no end, or the pre-marker
    legacy block, is reported and never edited blind. Returns 1 when a block
    was left in place.
    """
    home = home or Path.home()
    left_in_place = 0
    for name in PROFILE_FILES:
        rc = home / name
        if not rc.is_file():
            continue
        lines = rc.read_text(encoding="utf-8").splitlines(keepends=True)
        kept: list[str] = []
        removed = 0
        i = 0
        while i < len(lines):
            m = _BLOCK_START.match(lines[i])
            if not m:
                kept.append(lines[i])
                i += 1
                continue
            end_marker = f"# <<< {m.group(1)} managed block <<<"
            end = next((j for j in range(i + 1, len(lines)) if lines[j].rstrip("\n") == end_marker), None)
            if end is None:
                out(f"⚠️  {rc}: an AgentSmith block starts at line {i + 1} with no end marker — not edited.")
                left_in_place = 1
                kept.extend(lines[i:])
                break
            if kept and not kept[-1].strip():
                kept.pop()
            removed += 1
            i = end + 1
        if removed:
            shutil.copy2(rc, rc.with_name(rc.name + ".agentsmith-bak"))
            rc.write_text("".join(kept), encoding="utf-8")
            out(f"✅ Removed the AgentSmith shell functions from {rc} (backup: {rc.name}.agentsmith-bak)")
        elif "AI AGENT FRAMEWORK CONTROLLER" in "".join(kept):
            out(f"⚠️  {rc} has AgentSmith shell functions without managed-block markers — remove them by hand.")
            left_in_place = 1
    return left_in_place


def uninstall(yes: bool, purge: bool, out: Out = print, env: Optional[Mapping[str, str]] = None) -> int:
    """`agentsmith uninstall` — restore git config, remove the command and any
    legacy shell block; `--purge` also removes ~/.agent-framework and ~/.git_templates."""
    home = framework_home()
    out("🗑  AgentSmith: machine-level uninstall. This will:")
    out("   - Restore git init.templateDir to its pre-install value (or unset it)")
    out("   - Remove ~/.local/bin/agentsmith and any AgentSmith block in your shell profile")
    out("   - " + ("Remove ~/.agent-framework and ~/.git_templates" if purge
                     else "Keep ~/.agent-framework (pass --purge to remove it)"))
    try:
        hooks = org_policy_hooks()
    except Exception:
        hooks = {"bypass_policy": "disabled"}
    if hooks and hooks.get("bypass_policy") == "disabled":
        out("⚠️  This machine has an enterprise org policy with bypass_policy: disabled.")
        out("   Uninstalling removes hook enforcement entirely — not a sanctioned break-glass")
        out(f"   bypass. IT should be notified: {approvers_text(hooks)}")
    if not _confirm("   Confirm uninstall? (y/n): ", yes):
        out("❌ Cancelled")
        return 1

    previous = ""
    try:
        previous = (home / "previous_template_dir").read_text(encoding="utf-8").strip()
    except OSError:  # fail-open: no recorded value means "unset it", handled below
        pass
    if previous:
        _git_global("init.templateDir", previous)
        out(f"✅ Restored git init.templateDir → {previous}")
    else:
        _git_global("--unset", "init.templateDir")
        out("✅ Unset git init.templateDir (no prior value was recorded)")

    link = Path.home() / ".local" / "bin" / "agentsmith"
    if link.is_symlink() and str(home) in os.path.realpath(link):
        link.unlink()
        out(f"✅ Removed {link}")
    remove_profile_blocks(out=out)

    if purge:
        shutil.rmtree(home, ignore_errors=True)
        shutil.rmtree(Path.home() / ".git_templates", ignore_errors=True)
        out("✅ Removed ~/.agent-framework and ~/.git_templates")
    out("🎯 Uninstall complete.")
    return 0
