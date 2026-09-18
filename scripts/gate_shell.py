"""
scripts/gate_shell.py — the shell surface (.agent-rfc/designs/
governance-enforcement.md, G2b).

The edit gate watches an IDE's edit tools. The same agent can open a terminal
and type `git commit --no-verify`, and until this nothing said no until the
sweep found it afterwards.

    refusal(command, hooks_path) -> the reason to refuse, or None

It refuses four things: skipping the commit gate, pointing git at other hooks,
writing to what the gate is made of, and recording an approval from a shell.

**Stated limit, because it decides how much this is worth.** It reads the
command an IDE is about to run — not what that command does. `bash -c "$(…)"`,
a script file, a shell alias and a Makefile target all reach git without
passing through here. That is why G3's sweep exists and why the commit and CI
gates are the ones that cannot be talked around. This makes the obvious bypass
visible and costly, not impossible.
"""

from __future__ import annotations

import re
import shlex
from typing import List, Optional, Sequence

SEE = "See AgentSmith's docs/process-gates.md."
# What the gate is made of. Editing one of these is how you would switch it off,
# so a shell write to one is refused the same way the edit gate refuses it.
PROTECTED = (
    ".githooks/",
    ".agenticframework/approvals.jsonl",
    ".agenticframework/process-gates.json",
    ".claude/settings.json",
    ".cursor/hooks.json",
    ".agents/hooks.json",
    ".gemini/settings.json",
    ".codex/hooks.json",
    ".github/hooks/",
)
# Programs that change a file named on their command line. `cat` and `grep` are
# deliberately absent: reading the gate is fine, and a check that refused it
# would be refusing people who are trying to understand the thing.
WRITERS = {"rm", "mv", "cp", "tee", "truncate", "chmod", "chown", "install", "ln", "dd", "shred"}
_IN_PLACE = {"sed", "perl"}  # only with -i


def _segments(command: str) -> List[List[str]]:
    """One shell line, split where one command ends and the next begins.

    Tokenised FIRST, so quoting is respected: a bypass written inside quotes —
    a test fixture, an echo, a here-doc handed to an interpreter — is one
    argument to some other program, not a command being run. Splitting the raw
    text refused the repo's own tests for this rule, which is how this came to
    be tokenised.

    An unbalanced quote is not a line this can read; it falls back to the crude
    split rather than pretending it parsed.
    """
    separators = {"&&", "||", ";", "|", "&"}
    try:
        # `punctuation_chars` makes the lexer break on `;`, `|` and `&` the way
        # a shell does — `shlex.split` alone leaves `true;` as one token, and
        # the command after it was never looked at.
        lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
        lexer.whitespace_split = True
        tokens = list(lexer)
    except ValueError:
        return [part.split() for part in re.split(r"&&|\|\||;|\|", command) if part.strip()]
    segments: List[List[str]] = [[]]
    for token in tokens:
        if token in separators:
            segments.append([])
        else:
            segments[-1].append(token)
    return [segment for segment in segments if segment]


def _redirect_targets(segment: str) -> List[str]:
    return re.findall(r">>?\s*([^\s;|&]+)", segment)


def _protects(path: str) -> bool:
    # `lstrip("./")` would eat the leading dot of `.githooks/…` — every path
    # here starts with one, so that silently matched nothing.
    cleaned = path.strip("'\"")
    while cleaned.startswith("./"):
        cleaned = cleaned[2:]
    return any(cleaned == protected or cleaned.startswith(protected) for protected in PROTECTED)


def _git_refusal(words: Sequence[str], hooks_path: str) -> Optional[str]:
    """`git …` — the three ways to run it without the hooks."""
    arguments = list(words[1:])
    # `git -c core.hooksPath=… <anything>`: the hooks are off for that one call.
    for argument in arguments:
        if argument.startswith("core.hooksPath=") or argument.startswith("core.hookspath="):
            return (f"`git -c {argument}` runs this command with the hooks turned off. The commit gate "
                    f"is not optional; the sweep finds what skips it and the next commit has to repair it. {SEE}")
    words_after_options = [a for a in arguments if not a.startswith("-")]
    subcommand = words_after_options[0] if words_after_options else ""

    if subcommand == "config":
        values = [a for a in arguments if not a.startswith("-")][1:]
        if any(a.lower() == "core.hookspath" for a in values):
            wanted = values[values.index(next(v for v in values if v.lower() == "core.hookspath")) + 1:]
            target = wanted[0].strip("'\"") if wanted else ""
            if target != hooks_path:
                return (f"`git config core.hooksPath {target or '<empty>'}` points this repo away from "
                        f"{hooks_path}, which is where its gates live. Re-arming it — "
                        f"`git config core.hooksPath {hooks_path}` — is allowed and is what the sweep asks for. {SEE}")
        return None

    if subcommand in ("commit", "merge", "rebase", "cherry-pick", "revert"):
        if "--no-verify" in arguments or "-n" in arguments:
            return (f"`git {subcommand} --no-verify` skips the commit gate. Nothing is saved by it: the sweep "
                    "finds the commit on the next hook run and refuses every push until it is repaired. "
                    f"Fix what the gate is refusing instead. {SEE}")
    if subcommand == "push" and "--no-verify" in arguments:
        return ("`git push --no-verify` skips the pre-push sweep, which is the layer that catches commits "
                f"that never met a gate. CI checks the same range after the push. {SEE}")
    return None


def refusal(command: str, hooks_path: str = ".githooks") -> Optional[str]:
    """Why this shell command must not run, or None to let it through."""
    for words in _segments(command):
        if not words:
            continue
        segment = " ".join(words)
        program = words[0].rsplit("/", 1)[-1]

        if program == "git":
            found = _git_refusal(words, hooks_path)
            if found:
                return found

        if program == "agentsmith" and len(words) > 1 and words[1] == "approve":
            return ("`agentsmith approve` records the owner's permission and asks for it at a terminal — "
                    "an agent's shell has none, so this cannot complete here. Ask the owner to run it. " + SEE)

        # Writing to what the gate is made of: by redirection, or by a program
        # that takes the path as an argument.
        for target in _redirect_targets(segment):
            if _protects(target):
                return (f"writing to {target.strip(chr(39)+chr(34))} changes the gate itself. It is a gated path: "
                        f"it changes under a design and a review like anything else. {SEE}")
        if program in WRITERS or (program in _IN_PLACE and any(a.startswith("-i") for a in words[1:])):
            for argument in words[1:]:
                if not argument.startswith("-") and _protects(argument):
                    return (f"`{program} {argument}` changes the gate itself. It is a gated path: it changes "
                            f"under a design and a review like anything else. {SEE}")
    return None
