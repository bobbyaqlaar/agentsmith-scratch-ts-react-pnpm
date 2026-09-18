"""
runtime/machine/state.py — where the framework lives on this machine, and the
state `agentsmith mode` and the installer record there.

State is a FILE, not an environment variable, because an export reaches only
the shell that ran it. `~/.agent-framework/state/mode` is read by the gateway
(`runtime/llm_gateway.py:_active_profile_name`) and the four git hooks, so a
mode chosen in a terminal is the mode an IDE, a git GUI and a hook see too.

Each file holds one word, so a hook can read it with `cat` and no parser.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping, Optional

# `local` and `hybrid` name model profiles in models.yaml; `disabled` mutes the
# hooks. One vocabulary, because it used to be two variables (AI_STACK_MODE and
# DISABLE_AI_STACK) that could disagree.
MODES = ("local", "hybrid", "disabled")
INSTALL_MODES = ("developer", "enterprise")

# Overrides the state directory. For test isolation and sandboxes — the hooks
# honour it too, so a process and the hooks it triggers read the same state.
STATE_DIR_ENV = "AGENTSMITH_STATE_DIR"


def framework_home() -> Path:
    """`~/.agent-framework` — what install-ai-stack.sh installs into."""
    return Path.home() / ".agent-framework"


def state_dir(env: Optional[Mapping[str, str]] = None) -> Path:
    """The state directory. The override is looked up in `env`, then in the
    process environment — a caller passing a hand-built env mapping (a test, the
    OTLP resolver) must not escape an isolation the process already set."""
    override = ((env or {}).get(STATE_DIR_ENV) or os.environ.get(STATE_DIR_ENV) or "").strip()
    return Path(override) if override else framework_home() / "state"


def _read_word(name: str, allowed: tuple[str, ...], env: Optional[Mapping[str, str]]) -> Optional[str]:
    """The recorded word, or None when absent, unreadable or not one we wrote.

    A hand-edited or truncated file is treated as "not chosen" rather than
    trusted: every reader falls back to its own default, which is what an
    absent file already means.
    """
    try:
        word = (state_dir(env) / name).read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return word if word in allowed else None


def _write_word(name: str, word: str, allowed: tuple[str, ...], env: Optional[Mapping[str, str]]) -> Path:
    if word not in allowed:
        raise ValueError(f"{name} must be one of {allowed}, got {word!r}")
    directory = state_dir(env)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    tmp = path.with_suffix(".tmp")
    tmp.write_text(word + "\n", encoding="utf-8")
    tmp.replace(path)  # atomic: a hook reading mid-write sees the old word or the new one
    return path


def read_mode(env: Optional[Mapping[str, str]] = None) -> Optional[str]:
    return _read_word("mode", MODES, env)


def write_mode(mode: str, env: Optional[Mapping[str, str]] = None) -> Path:
    return _write_word("mode", mode, MODES, env)


def read_install_mode(env: Optional[Mapping[str, str]] = None) -> str:
    """`developer` unless the installer recorded `enterprise` (it writes the file
    itself, in Step 7).

    An install that predates this file was a developer install in every case
    that matters here: the shell functions it replaced set `init.templateDir`
    unconditionally.
    """
    return _read_word("install-mode", INSTALL_MODES, env) or "developer"


def read_phoenix_endpoint(env: Optional[Mapping[str, str]] = None) -> Optional[str]:
    """The Phoenix `agentsmith dashboard start` last started, while it runs.

    runtime/otlp.py reads it after the environment, so a process started from
    an IDE exports traces to the dashboard the terminal started. The shell
    function exported OTEL_EXPORTER_OTLP_ENDPOINT into that one terminal.
    """
    try:
        value = (state_dir(env) / "phoenix-endpoint").read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return value if value.startswith(("http://", "https://")) else None


def write_phoenix_endpoint(endpoint: str, env: Optional[Mapping[str, str]] = None) -> None:
    if not endpoint.startswith(("http://", "https://")):
        raise ValueError(f"phoenix endpoint must be an http(s) URL, got {endpoint!r}")
    directory = state_dir(env)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "phoenix-endpoint").write_text(endpoint + "\n", encoding="utf-8")


def clear_phoenix_endpoint(env: Optional[Mapping[str, str]] = None) -> None:
    (state_dir(env) / "phoenix-endpoint").unlink(missing_ok=True)


def find_script(name: str, cwd: Optional[Path] = None) -> Optional[Path]:
    """A framework script: the tenant's vendored copy, else the machine's.

    Exactly one is returned, never "try one, then the other". The shell
    functions ran `python3 scripts/X.py || python3 ~/.agent-framework/scripts/X.py`,
    so a tenant script that FAILED was re-run from the framework's copy — an
    eval gate could pass on code the tenant does not run.
    """
    local = (cwd or Path.cwd()) / "scripts" / name
    if local.is_file():
        return local
    installed = framework_home() / "scripts" / name
    return installed if installed.is_file() else None
