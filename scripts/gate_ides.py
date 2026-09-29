"""
scripts/gate_ides.py — one gate, six dialects (.agent-rfc/designs/
governance-enforcement.md, G2a).

The rules are one implementation. What differs between IDEs is the shape of the
payload on stdin and the shape of the answer on stdout, and both live here:

    parse(ide, event, payload) -> GateEvent     what the IDE is asking about
    render(ide, decision, text) -> str          the answer, in its dialect
    ADAPTERS[ide].config_path                   where its hook config goes

A seventh IDE is a table entry, a fixture and a test — never a second gate.

**Verified.** Claude Code by use: it is the hook family this repo runs. Cursor
against the vendor's docs, re-read 2026-09-17 — `preToolUse` with
`matcher: "Write"` (there is no before-edit hook; `afterFileEdit` fires after
the write), `permission: allow|deny` with `user_message`/`agent_message`,
`followup_message` on `stop`, `additional_context` on `sessionStart`, and
`workspace_roots` where there is no `cwd`. The other four follow the design's
pinned table, verified by the owner on 2026-09-15 and not re-verified here;
each carries a golden payload fixture, which is where a real session's payload
gets pinned.

**A write payload this cannot read is denied**, naming the keys it saw. Field
names are the part most likely to be wrong, and a parser that shrugged would
leave a silent hole in exactly the IDEs nobody here tests daily.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, Mapping, NamedTuple, Optional

import gate_models as gm

IDES = ("claude", "cursor", "antigravity", "copilot", "gemini", "codex")
DEFAULT_IDE = "claude"

# Tool names that WRITE. Everything else is allowed through by the edit gate —
# a read is not a change, and G2b is what looks at shell commands.
_EDIT_TOOLS = {"edit", "write", "multiedit", "notebookedit", "create_file", "apply_patch",
               "str_replace_editor", "replace", "write_file"}
_SHELL_TOOLS = {"bash", "shell", "run_shell_command", "terminal", "run_command", "exec"}
# Where a path hides, across the six dialects. Spellings, not guesses at
# semantics: each is a field one of these IDEs documents for its write tool.
_PATH_KEYS = ("file_path", "filePath", "notebook_path", "notebookPath", "path",
              "absolute_path", "absolutePath", "target_file", "targetFile", "file")
_COMMAND_KEYS = ("command", "cmd", "script")


class Unreadable(ValueError):
    """A payload that says it is a write but names no path this can find."""


def _first(mapping: Mapping[str, Any], keys) -> Optional[str]:
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _tool_input(payload: Mapping[str, Any]) -> Dict[str, Any]:
    for key in ("tool_input", "toolInput", "input", "arguments", "args", "params"):
        value = payload.get(key)
        if isinstance(value, dict):
            return value
        if isinstance(value, str):  # Cursor's MCP hooks pass JSON as a string
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed
    return {}


def _tool_name(payload: Mapping[str, Any]) -> str:
    return str(_first(payload, ("tool_name", "toolName", "tool", "name")) or "")


def _root(payload: Mapping[str, Any]) -> Optional[str]:
    """The repository. Cursor's stop and sessionStart carry no `cwd` at all —
    they carry `workspace_roots`, and a gate that cannot find the repo checks
    nothing."""
    direct = _first(payload, ("cwd", "workingDirectory", "working_directory", "project_dir",
                              "projectDir", "workspace_root", "workspaceRoot", "rootPath"))
    if direct:
        return direct
    for key in ("workspace_roots", "workspaceRoots", "workspaceFolders", "roots"):
        value = payload.get(key)
        if isinstance(value, list) and value:
            first = value[0]
            if isinstance(first, str):
                return first
            if isinstance(first, dict):
                found = _first(first, ("path", "uri", "fsPath"))
                if found:
                    return found.replace("file://", "")
    return None


def _generic(event: str, payload: Mapping[str, Any]) -> gm.GateEvent:
    """What every dialect has in common. The per-IDE entries below say what it
    does differently, rather than each restating all of this (`no-copy-paste`)."""
    cwd = _root(payload)
    if event == "stop":
        repeat = bool(payload.get("stop_hook_active") or payload.get("stopHookActive")
                      or (payload.get("loop_count") or 0))
        return gm.GateEvent(kind="other", cwd=cwd, stop_active=repeat)
    if event == "session-start":
        return gm.GateEvent(kind="other", cwd=cwd)

    tool = _tool_name(payload).lower()
    arguments = _tool_input(payload)
    command = _first(arguments, _COMMAND_KEYS) or _first(payload, _COMMAND_KEYS)
    if tool in _SHELL_TOOLS or (not tool and command):
        return gm.GateEvent(kind="shell", cwd=cwd, command=command)
    path = _first(arguments, _PATH_KEYS) or _first(payload, _PATH_KEYS)
    if tool in _EDIT_TOOLS or (not tool and path):
        if not path:
            raise Unreadable(
                f"a {tool or 'write'} payload named no file this gate could find. Keys seen: "
                f"{sorted(arguments) or sorted(payload)}. The edit is denied rather than allowed "
                "through — pin this payload in scripts/test/fixtures/ide-payloads/ and the "
                "adapter will read it."
            )
        return gm.GateEvent(kind="edit", cwd=cwd, paths=[path])
    return gm.GateEvent(kind="other", cwd=cwd)


# ── the answers ──────────────────────────────────────────────────────────────


def _claude_render(decision: str, text: str, repeat: bool) -> str:
    if decision == "deny":
        return json.dumps({"hookSpecificOutput": {
            "hookEventName": "PreToolUse", "permissionDecision": "deny", "permissionDecisionReason": text}})
    if decision == "context":
        return json.dumps({"hookSpecificOutput": {
            "hookEventName": "SessionStart", "additionalContext": text}})
    if repeat:
        return json.dumps({"systemMessage": "⚠️ Turn ended with " + text})
    return json.dumps({"decision": "block", "reason": text})


def _cursor_render(decision: str, text: str, repeat: bool) -> str:
    if decision == "deny":
        return json.dumps({"permission": "deny", "user_message": text, "agent_message": text})
    if decision == "context":
        return json.dumps({"additional_context": text})
    if repeat:
        return json.dumps({"additional_context": "⚠️ Turn ended with " + text})
    return json.dumps({"followup_message": text})


def _gemini_render(decision: str, text: str, repeat: bool) -> str:
    if decision == "deny":
        return json.dumps({"decision": "deny", "reason": text})
    if decision == "context":
        return json.dumps({"additionalContext": text})
    if repeat:
        return json.dumps({"systemMessage": "⚠️ Turn ended with " + text})
    return json.dumps({"decision": "deny", "reason": text})


def _antigravity_render(decision: str, text: str, repeat: bool) -> str:
    if decision == "deny":
        return json.dumps({"decision": "deny", "reason": text})
    if decision == "context":
        return json.dumps({"additionalContext": text})
    if repeat:
        return json.dumps({"systemMessage": "⚠️ Turn ended with " + text})
    return json.dumps({"decision": "block", "reason": text})


def _codex_render(decision: str, text: str, repeat: bool) -> str:
    if decision == "deny":
        return json.dumps({"decision": "block", "reason": text})
    if decision == "context":
        return json.dumps({"context": text})
    if repeat:
        return json.dumps({"message": "⚠️ Turn ended with " + text})
    return json.dumps({"decision": "block", "reason": text})


class Adapter(NamedTuple):
    """One IDE: where its config goes, how it asks, how it must be answered."""

    config_path: str
    events: Dict[str, str]  # our event -> the IDE's hook name
    parse: Callable[[str, Mapping[str, Any]], gm.GateEvent]
    render: Callable[[str, str, bool], str]
    # Does the EDIT GATE fail closed for this IDE — is an edit refused when the
    # gate cannot run? A governance fact a tenant picking an IDE needs, not a
    # config detail: the mechanism differs per IDE (Cursor's failClosed key,
    # Claude's shell fallback, the contract's fall-through), which is why no
    # single line of code can read it and scripts/test/test_fail_closed_declared.py
    # checks each one's own mechanism instead
    # (.agent-rfc/designs/fail-closed-has-a-reader.md).
    fail_closed: bool
    note: str

    def retarget(self, payload: dict, root: str, path: str) -> dict:
        """This IDE's fixture, pointed at a real repo and file."""
        return retarget(payload, root, path)


# The neutral profile of contract/gate/v1: the event as the checks read it, and
# the decision as `gm.Decision`. It is a CONTRACT, not an IDE — it is deliberately
# absent from IDES, which the registry pins one-for-one — and it is how a tenant
# that named a provider, or another platform's adapter, talks to the gate.
def _neutral_parse(event: str, payload: Mapping[str, Any]) -> gm.GateEvent:
    try:
        return gm.GateEvent.model_validate(dict(payload))
    except Exception as exc:  # a payload in someone else's dialect, or malformed
        # Unreadable is what every adapter raises for "I cannot read this", and
        # the gate answers it with a refusal rather than a traceback: a provider
        # handed a payload it does not understand must fail closed, not crash.
        raise Unreadable(
            f"this is not a {NEUTRAL} gate event (contract/gate/v1/event.schema.json): {exc}"
        ) from exc


def _neutral_render(decision: str, text: str, repeat: bool) -> str:
    return gm.Decision(decision="block" if decision == "block" else decision, text=text).model_dump_json()


NEUTRAL = "neutral"

ADAPTERS: Dict[str, Adapter] = {
    NEUTRAL: Adapter("", {"session-start": "session-start", "pre-edit": "pre-edit", "stop": "stop"},
                     _neutral_parse, _neutral_render, True,
                     "contract/gate/v1 — the profile a provider implements; no IDE sends it"),
    "claude": Adapter(".claude/settings.json",
                      {"session-start": "SessionStart", "pre-edit": "PreToolUse", "stop": "Stop"},
                      _generic, _claude_render, True,
                      "verified by use — this repo runs it; .githooks/process-gate's pre-edit "
                      "fallback prints a deny, which is what makes it fail closed"),
    "cursor": Adapter(".cursor/hooks.json",
                      {"session-start": "sessionStart", "pre-edit": "preToolUse", "stop": "stop"},
                      _generic, _cursor_render, True,
                      "vendor docs 2026-09-17; no before-edit hook, so the edit gate is preToolUse "
                      "with matcher Write, and sessionStart cannot block"),
    "antigravity": Adapter(".agents/hooks.json",
                           {"session-start": "SessionStart", "pre-edit": "PreToolUse", "stop": "Stop"},
                           _generic, _antigravity_render, False,
                           "design's pinned table (owner, 2026-09-15); fails open, so git and the sweep hold"),
    "copilot": Adapter(".github/hooks/agentsmith.json",
                       {"session-start": "sessionStart", "pre-edit": "preToolUse", "stop": "stop"},
                       _generic, _claude_render, False,
                       "reads Claude's shape but ignores matchers and spells tool fields in camelCase"),
    "gemini": Adapter(".gemini/settings.json",
                      {"session-start": "SessionStart", "pre-edit": "PreToolUse", "stop": "AfterAgent"},
                      _generic, _gemini_render, False,
                      "design's pinned table (owner, 2026-09-15)"),
    "codex": Adapter(".codex/hooks.json",
                     {"session-start": "SessionStart", "pre-edit": "PreToolUse", "stop": "Stop"},
                     _generic, _codex_render, False,
                     "design's pinned table (owner, 2026-09-15); session context travels via AGENTS.md"),
}


def resolve(explicit: Optional[str], environ: Mapping[str, str]) -> str:
    """`--ide`, then $AGENTSMITH_IDE, then Claude Code — the one this repo runs."""
    name = (explicit or environ.get("AGENTSMITH_IDE") or DEFAULT_IDE).strip().lower()
    if name not in ADAPTERS:
        raise ValueError(f"no adapter for IDE {name!r} — one of {', '.join(IDES)}")
    return name


def parse(ide: str, event: str, payload: Mapping[str, Any]) -> gm.GateEvent:
    return ADAPTERS[ide].parse(event, payload)


def render(ide: str, decision: str, text: str, repeat: bool = False) -> str:
    return ADAPTERS[ide].render(decision, text, repeat)


def retarget(payload: dict, root: str, path: str) -> dict:
    """A fixture, pointed at a real repo and file — so a test can run the gate
    end to end on the payload shape an IDE actually sends."""
    updated = json.loads(json.dumps(payload))
    for key in ("cwd", "workingDirectory", "working_directory", "project_dir", "projectDir"):
        if key in updated:
            updated[key] = root
    for key in ("workspace_roots", "workspaceRoots", "roots"):
        if isinstance(updated.get(key), list) and updated[key]:
            updated[key] = [root]
    arguments = _tool_input(updated)
    for key in _PATH_KEYS:
        if key in arguments:
            arguments[key] = path
    for container in ("tool_input", "toolInput", "input", "arguments"):
        if isinstance(updated.get(container), dict):
            updated[container] = arguments
    return updated


# ── the hook configs ─────────────────────────────────────────────────────────
#
# Generated, so that a hook existing means it runs (`implemented-not-invoked`).
# Only where the config SCHEMA is verified: a file in a shape nobody has
# confirmed looks like enforcement and may be ignored in silence, which is
# worse than no file at all. Claude Code's is verified by use and Cursor's
# against the vendor's docs; the other four wait for a real session to confirm
# theirs, and `agentsmith` says so rather than writing something hopeful.
LAUNCHER = ".githooks/process-gate"
GENERATED = ("claude", "cursor")


def _command(event: str, ide: str, root_var: str = "") -> str:
    prefix = f'bash "{root_var}/{LAUNCHER}"' if root_var else f"bash {LAUNCHER}"
    return f"{prefix} {event} --ide {ide}"


def render_config(ide: str, existing: Optional[dict] = None) -> dict:
    """This IDE's hook config, ready to write. `existing` keeps whatever else
    the file holds — Claude's carries attribution and permissions, which are
    not this generator's to own."""
    if ide == "cursor":
        return {
            "version": 1,
            "hooks": {
                "sessionStart": [{"command": _command("session-start", ide)}],
                # There is no before-edit hook in Cursor; preToolUse fires
                # before every tool and the matcher narrows it to writes.
                "preToolUse": [{"command": _command("pre-edit", ide), "matcher": "Write",
                                "failClosed": ADAPTERS[ide].fail_closed, "timeout": 30}],
                # The shell surface (G2b). Cursor gives it its own hook, with
                # the command and cwd; the same subcommand reads both.
                "beforeShellExecution": [{"command": _command("pre-edit", ide),
                                          "failClosed": ADAPTERS[ide].fail_closed, "timeout": 30}],
                "stop": [{"command": _command("stop", ide), "timeout": 60}],
            },
        }
    if ide == "claude":
        config = json.loads(json.dumps(existing or {}))
        deny = json.dumps({"hookSpecificOutput": {
            "hookEventName": "PreToolUse", "permissionDecision": "deny",
            "permissionDecisionReason": (
                "The process gate could not run (.githooks/process-gate missing, or it failed before "
                "answering). Edits are denied until it runs - see AgentSmith docs/process-gates.md."),
        }})
        config["hooks"] = {
            "SessionStart": [{"hooks": [
                {"type": "command", "command": _command("session-start", ide, "$CLAUDE_PROJECT_DIR"),
                 "timeout": 30}]}],
            # The shell surface fails OPEN when the launcher cannot run, unlike
            # the edit gate below. A broken gate that refuses every shell
            # command leaves no way to repair the install from inside the IDE,
            # and the sweep and commit gate still hold behind it.
            "PreToolUse": [{"matcher": "Bash", "hooks": [
                {"type": "command", "command": _command("pre-edit", ide, "$CLAUDE_PROJECT_DIR"),
                 "timeout": 30, "statusMessage": "Checking the shell gate"}]},
                {"matcher": "Edit|Write|MultiEdit|NotebookEdit", "hooks": [
                {"type": "command",
                 # The fallback is the fail-closed half: a launcher that cannot
                 # run must still produce a deny, or the gate silently stops.
                 "command": _command("pre-edit", ide, "$CLAUDE_PROJECT_DIR")
                            + " || printf '%s' '" + deny.replace("'", "'\\''") + "'",
                 "timeout": 30, "statusMessage": "Checking the design gate"}]}],
            "Stop": [{"hooks": [
                {"type": "command", "command": _command("stop", ide, "$CLAUDE_PROJECT_DIR"),
                 "timeout": 60}]}],
        }
        return config
    raise ValueError(
        f"no verified config schema for {ide} — its hook config is not generated. The adapter reads and "
        "answers it (scripts/gate_ides.py); what is missing is a confirmed shape for the file itself."
    )
