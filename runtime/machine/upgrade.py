"""
runtime/machine/upgrade.py — `agentsmith upgrade`: refresh a vendored tenant's
copy of the framework and commit it.

Ported from the `ai-stack-upgrade` shell function, keeping every defect that
function had already been fixed for (each is a test in
scripts/test/test_ai_stack_upgrade.py):

- untracked first-time files are committed (`git status --porcelain`, not
  `git diff --quiet`, which ignores them);
- the framework's own scripts/test/ is pruned in a staging copy, never after
  copying — pruning in place deleted a tenant's own scripts/test/;
- a tenant's own `runtime` package is never merged into;
- runtime/test/ is cut down to the suites the security harness delegates to;
- a tenant that depends on agentsmith-runtime as a package is not vendored into.

And one the function still had: with no `--to`, it declared
`${FRAMEWORK_VERSION:-1.1.0}` — a variable the shell profile never set — so a
bare upgrade rewrote tenant.yaml to 1.1.0. It now declares the installed
framework's release.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Optional

from runtime.machine.policy import audit_log_event
from runtime.machine.state import framework_home

# The suites run-security-checks.py delegates to — the same list as
# TENANT_RUNTIME_TESTS in hooks/post-checkout (test_workflow_template_wiring.py).
TENANT_RUNTIME_TESTS = (
    "conftest.py",
    "test_hitl_gate.py",
    "test_dead_letter.py",
    "test_llm_gateway_budget.py",
    "test_self_correction.py",
)

_PACKAGE_PIN = re.compile(r"""^[\s"']*agentsmith-runtime""", re.M)


def _files_under(root: Path, skip_name: str = "") -> list[str]:
    return sorted(
        p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and p.name != skip_name
    )


def _copy_tree_merge(src: Path, dest: Path) -> None:
    shutil.copytree(src, dest, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__"))


def _depends_on_runtime_package(repo: Path) -> bool:
    candidates = [*repo.glob("requirements*.txt"), repo / "pyproject.toml"]
    for path in candidates:
        try:
            if path.is_file() and _PACKAGE_PIN.search(path.read_text(encoding="utf-8")):
                return True
        except OSError:
            continue
    return False


def _write_ruff_isolation(repo: Path, directory: str, files: list[str]) -> None:
    """Exclude the vendored files from the tenant's own ruff config.

    Only a file AgentSmith wrote, or none: a tenant's own directory config is
    never touched. Same content hooks/post-checkout's write_vendored_ruff_config
    produces.
    """
    target = repo / directory
    config = target / "ruff.toml"
    if config.is_file() and not config.read_text(encoding="utf-8").startswith("# Written by AgentSmith"):
        return
    if (target / ".ruff.toml").exists() or (target / "pyproject.toml").exists():
        return
    extend = ""
    if (repo / "ruff.toml").is_file():
        extend = 'extend = "../ruff.toml"'
    elif (repo / ".ruff.toml").is_file():
        extend = 'extend = "../.ruff.toml"'
    elif (repo / "pyproject.toml").is_file() and re.search(
        r"^\[tool\.ruff", (repo / "pyproject.toml").read_text(encoding="utf-8"), re.M
    ):
        extend = 'extend = "../pyproject.toml"'
    lines = [
        f"# Written by AgentSmith when it vendored {directory}/ — see hooks/post-checkout.",
        "# Excludes the vendored files from this repo's ruff check/format; your own",
        "# files here keep your rules. `agentsmith upgrade` regenerates this file.",
    ]
    if extend:
        lines.append(extend)
    lines.append("extend-exclude = [")
    lines += [f'  "{name.replace(chr(34), chr(92) + chr(34))}",' for name in sorted(files) if name]
    lines.append("]")
    config.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _tenant_id(tenant_yaml: Path) -> str:
    try:
        import yaml

        doc = yaml.safe_load(tenant_yaml.read_text(encoding="utf-8")) or {}
        return str((doc.get("tenant") or {}).get("id") or "")
    except Exception:
        return ""


def upgrade(
    repo: Path,
    target_version: str,
    *,
    out: Callable[[str], None] = print,
    home: Optional[Path] = None,
) -> int:
    home = home or framework_home()
    tenant_yaml = repo / ".agenticframework" / "tenant.yaml"
    if not tenant_yaml.is_file():
        out("❌ No .agenticframework/tenant.yaml in current repo — run from the tenant repo root")
        return 1
    if not (repo / ".git").exists():
        out("❌ Not a git repository")
        return 1

    # Same guard as hooks/post-checkout's "installed mode": vendoring runtime/
    # into the root of a repo that pins the package would shadow the pin.
    if _depends_on_runtime_package(repo):
        out("ℹ️  This repo depends on agentsmith-runtime as a package — nothing to vendor.")
        out("   Upgrade by bumping the agentsmith-runtime pin (and framework.version in tenant.yaml) instead.")
        return 0

    vendor_src = home / "scripts"
    if not vendor_src.is_dir() or not any(vendor_src.iterdir()):
        out(f"❌ No vendored scripts found at {vendor_src} — run install-ai-stack.sh on this machine first")
        return 1

    out(f"📦 Upgrading vendored scripts to v{target_version}...")
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp) / "scripts"
        _copy_tree_merge(vendor_src, stage)
        shutil.rmtree(stage / "test", ignore_errors=True)
        vendored_scripts = _files_under(stage)
        _copy_tree_merge(stage, repo / "scripts")
    out(f"✅ Copied vendored scripts from {vendor_src}")

    runtime_src = home / "runtime"
    runtime = repo / "runtime"
    if runtime.is_dir() and not (runtime / "llm_gateway.py").is_file():
        out("⚠️  runtime/ exists and is not AgentSmith's — NOT upgraded. Rename yours, or install")
        out("   AgentSmith's runtime as a package; scripts/ will import YOUR runtime.* meanwhile.")
    elif runtime_src.is_dir() and any(runtime_src.iterdir()):
        _copy_tree_merge(runtime_src, runtime)
        shutil.rmtree(runtime / ".hitl_blobs", ignore_errors=True)
        shutil.rmtree(runtime / "test", ignore_errors=True)
        (runtime / "test").mkdir()
        for suite in TENANT_RUNTIME_TESTS:
            if (runtime_src / "test" / suite).is_file():
                shutil.copy2(runtime_src / "test" / suite, runtime / "test" / suite)
        out(f"✅ Copied vendored runtime/ from {runtime_src}")
    else:
        out(f"⚠️  No vendored runtime/ found at {runtime_src} — skipping. Re-run")
        out("   install-ai-stack.sh from a live checkout to pick it up.")

    if (repo / "scripts").is_dir():
        _write_ruff_isolation(repo, "scripts", vendored_scripts)
    if (runtime / "llm_gateway.py").is_file():
        _write_ruff_isolation(repo, "runtime", _files_under(runtime, skip_name="ruff.toml"))

    # templates/: the rules source and the registry the vendored gate reads as
    # `@framework/templates/governance.json` — which, for a vendored tenant,
    # resolves inside the tenant. Without these the gate cannot load its rules.
    templates_src = home / "templates"
    if templates_src.is_dir() and any(templates_src.iterdir()):
        _copy_tree_merge(templates_src, repo / "templates")
        out(f"✅ Copied vendored templates/ from {templates_src}")
    else:
        out(f"⚠️  No vendored templates/ found at {templates_src} — the process gate will have no rules registry.")

    security_src = home / "fixtures" / "security"
    if security_src.is_dir() and any(security_src.iterdir()):
        _copy_tree_merge(security_src, repo / "fixtures" / "security")
        out(f"✅ Copied vendored fixtures/security/ from {security_src}")
    else:
        out(f"⚠️  No vendored fixtures/security/ found at {security_src} — skipping.")
    base_fixtures = sorted((home / "fixtures").glob("*_base.json")) if (home / "fixtures").is_dir() else []
    for fixture in base_fixtures:
        (repo / "fixtures").mkdir(exist_ok=True)
        shutil.copy2(fixture, repo / "fixtures" / fixture.name)

    text = tenant_yaml.read_text(encoding="utf-8")
    tenant_yaml.write_text(re.sub(r"^  version: .*$", f'  version: "{target_version}"', text, flags=re.M))
    out(f'✅ Updated .agenticframework/tenant.yaml -> framework.version: "{target_version}"')

    # Only paths that exist: `git add` on a pathspec matching nothing aborts
    # the whole command, not just that path.
    vendor_paths = ["scripts", ".agenticframework/tenant.yaml"]
    if (repo / "templates").is_dir():
        vendor_paths.append("templates")
    if (runtime / "llm_gateway.py").is_file():
        vendor_paths.append("runtime")  # never a tenant's own runtime/
    if (repo / "fixtures" / "security").is_dir():
        vendor_paths.append("fixtures/security")
    if (repo / "fixtures").is_dir() and any((repo / "fixtures").glob("*_base.json")):
        vendor_paths.append("fixtures/*_base.json")  # a pathspec: the base fixtures only

    def git(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=False)

    if not git("status", "--porcelain", "--", *vendor_paths).stdout.strip():
        out(f"ℹ️  No changes — {' '.join(vendor_paths)} already match v{target_version}")
        return 0

    message = f"chore(framework): upgrade AgentSmith to v{target_version}"
    git("add", "--", *vendor_paths)
    committed = git("commit", "-m", message)
    if committed.returncode != 0:
        out(committed.stdout + committed.stderr)
        out("❌ git commit failed (blocked by a hook, GPG-sign required and unavailable, etc.) —")
        out("   scripts/ and tenant.yaml were updated and staged but NOT committed. Fix the issue and re-run:")
        out(f'   git commit -m "{message}"')
        return 1
    out(f"✅ Committed: {message}")

    audit_log_event(
        "config_change",
        os.environ.get("AGENT_OWNER_ID") or "unknown",
        _tenant_id(tenant_yaml),
        {"action": "framework_upgrade", "version": target_version},
    )
    out("")
    out("🎯 Upgrade complete. Push and open a PR per your branch protection rules (no direct push to main).")
    return 0
