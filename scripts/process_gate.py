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

Stdlib only and Python 3.9-compatible: hooks run whatever `python3` is on PATH,
and on a stock Mac that is 3.9 without pyyaml.

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

CONFIG = ".agenticframework/process-gates.json"
DESIGNS_DIR = ".agent-rfc/designs"
REVIEWS_DIR = ".agent-rfc/reviews"
FRAMEWORK_PREFIX = "@framework/"
# The directory this script was installed from: an AgentSmith checkout, or
# ~/.agent-framework. `@framework/<path>` in a config resolves against it.
FRAMEWORK_ROOT = Path(__file__).resolve().parents[1]

SMALL_CHANGE_LINES = 20
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

    def problems(self) -> List[str]:
        errors = []
        if not self.gated:
            errors.append(f"{CONFIG} declares no gated paths")
        elif not self.is_gated(CONFIG):
            errors.append(f"{CONFIG} must gate itself, or the gates can be switched off unreviewed")
        if self.changelog_file and not self.changelog_paths:
            errors.append(f"{CONFIG} names a changelog file but no paths that require it")
        return errors

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


def check_design(text: str, known_slugs: set, levers_doc: str = "docs/review-levers.md") -> List[str]:
    errors = []
    meta, body = front_matter(text)
    if not meta:
        return ["has no front matter (--- status: active|done, scope: [globs] ---)"]
    if meta.get("status") not in ("active", "done"):
        errors.append(f"status is {meta.get('status')!r}, expected active or done")
    scope = meta.get("scope")
    if not isinstance(scope, list) or not scope:
        errors.append("scope lists no paths")
    for heading in ("Problem", "Approach", "Levers"):
        if _section(body, heading) is None:
            errors.append(f"has no '## {heading}' section")
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


def check_review(text: str) -> List[str]:
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
    return errors


# ── Trailers ─────────────────────────────────────────────────────────────────


def trailer(message: str, name: str) -> Optional[str]:
    found = re.findall(rf"^{name}:[ \t]*(.*?)[ \t]*$", message, re.M | re.I)
    return found[-1] if found else None


def resolve_record(value: str, directory: str) -> Tuple[Optional[str], Optional[str]]:
    """-> (path, error). The value arrives from a commit message; it is only
    ever allowed to name a Markdown file directly inside `directory`."""
    path = value.strip()
    if path.startswith("./"):
        path = path[2:]
    parent, _, name = path.rpartition("/")
    if parent != directory or not name.endswith(".md") or "/" in name or name.startswith("."):
        return None, f"must name a file directly under {directory}/, got {value!r}"
    return path, None


def check_change(
    files: List[str],
    gated_lines: int,
    message: str,
    read: Reader,
    config: Config,
) -> Tuple[List[str], List[str]]:
    """One commit's worth of files against its message. -> (errors, notes)."""
    gated = sorted(f for f in files if config.is_gated(f))
    if not gated:
        return [], []
    errors: List[str] = []
    notes: List[str] = []
    small = gated_lines <= SMALL_CHANGE_LINES
    known_slugs = lever_slugs(config.doc_text(config.levers_doc, read) or "")
    levers_shown = config.display(config.levers_doc)

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

    design_value = trailer(message, "Design")
    if design_value is None:
        errors.append(f"missing 'Design: {DESIGNS_DIR}/<slug>.md' trailer (gated paths: {', '.join(gated[:5])}"
                      f"{' …' if len(gated) > 5 else ''})")
    elif not na("Design", design_value):
        path, err = resolve_record(design_value, DESIGNS_DIR)
        text = read(path) if path else None
        if err:
            errors.append(f"Design: {err}")
        elif text is None:
            errors.append(f"Design: {path} does not exist in this commit")
        else:
            errors.extend(f"Design: {path} {e}" for e in check_design(text, known_slugs, levers_shown))
            scope = design_scope(text)
            uncovered = [f for f in gated if not _any(f, scope)]
            if scope and uncovered:
                errors.append(f"Design: {path} scope does not cover {', '.join(uncovered)}")

    review_value = trailer(message, "Review")
    if review_value is None:
        errors.append(f"missing 'Review: {REVIEWS_DIR}/<slug>.md' trailer")
    elif not na("Review", review_value):
        path, err = resolve_record(review_value, REVIEWS_DIR)
        text = read(path) if path else None
        if err:
            errors.append(f"Review: {err}")
        elif text is None:
            errors.append(f"Review: {path} does not exist in this commit")
        else:
            errors.extend(f"Review: {path} {e}" for e in check_review(text))
            if path not in files:
                errors.append(
                    f"Review: {path} is not changed in this commit — a review older than the change "
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
    """-> [(relpath, text, errors)] for designs with status: active."""
    read = _worktree_reader(root)
    slugs = lever_slugs(config.doc_text(config.levers_doc, read) or "")
    found = []
    for path in sorted((root / DESIGNS_DIR).glob("*.md")):
        text = path.read_text(encoding="utf-8")
        if front_matter(text)[0].get("status") == "active":
            errors = check_design(text, slugs, config.display(config.levers_doc))
            found.append((f"{DESIGNS_DIR}/{path.name}", text, errors))
    return found


def _deny(reason: str) -> None:
    print(json.dumps({"hookSpecificOutput": {
        "hookEventName": "PreToolUse", "permissionDecision": "deny", "permissionDecisionReason": reason,
    }}))


def cmd_pre_edit(payload: dict) -> int:
    tool_input = payload.get("tool_input") or {}
    target = tool_input.get("file_path") or tool_input.get("notebook_path")
    if not target:
        return 0
    root = repo_root(payload.get("cwd"))
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
        "(front matter status: active, scope: globs covering this file; sections ## Problem, "
        f"## Approach, ## Levers citing levers from {config.display(config.levers_doc)}), then retry. "
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
        errs = check_review(review_path.read_text(encoding="utf-8"))
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
    root = repo_root(payload.get("cwd"))
    problems = stop_problems(root)
    if not problems:
        return 0
    text = "Unreviewed gated changes:\n- " + "\n- ".join(problems)
    if payload.get("stop_hook_active"):
        # Blocking again could loop forever. The commit and CI gates still hold.
        print(json.dumps({"systemMessage": "⚠️ Turn ended with " + text}))
    else:
        print(json.dumps({"decision": "block", "reason": text + "\nSee AgentSmith's docs/process-gates.md."}))
    return 0


def cmd_session_start(payload: dict) -> int:
    root = repo_root(payload.get("cwd"))
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
        checklist, levers = config.display(config.design_checklist), config.display(config.levers_doc)
        lines = [
            "This repository enforces its build discipline mechanically (AgentSmith docs/process-gates.md).",
            f"1. Design before code: before editing code, work {checklist} and write "
            f"{DESIGNS_DIR}/<slug>.md (status: active, scope globs, ## Problem / ## Approach / ## Levers). "
            "Edits to gated paths without one are denied.",
            f"2. Review before done: after building, run review passes against {levers}, verify each "
            f"finding in code, fix, and record every pass in {REVIEWS_DIR}/<slug>.md as "
            "'## Pass N — findings: K' until a pass finds 0. Ending a turn with unreviewed changes is blocked.",
            "3. Every commit touching gated paths carries 'Design: <design path>' and 'Review: <review path>' "
            "trailers, and changes the review record in that same commit. CI checks every pushed commit"
            + (f", and {config.changelog_file} for the paths that need it." if config.changelog_file else "."),
            f"Gated paths are declared in {CONFIG}. Bash-made edits are not caught by the edit gate — "
            "the stop, commit and CI gates still see them.",
        ]
        if designs:
            lines.append("Active designs: " + ", ".join(p for p, _, _ in designs))
        if not (root / "AGENTS.md").is_file() and (root / "scripts/generate-ide-config.py").is_file():
            # Other agents (Codex, Cursor, Gemini, Copilot) read these, not this hook.
            lines.append(
                "⚠️ This clone has no generated agent files (AGENTS.md, .cursorrules, …): run "
                "`python3 scripts/generate-ide-config.py --repo-root .` (needs pyyaml; the files are gitignored)."
            )
    if git("config", "--get", "core.hooksPath", cwd=root, check=False).strip() != ".githooks":
        lines.append("⚠️ The commit gate is not armed in this clone: run `git config core.hooksPath .githooks`.")
    print(json.dumps({"hookSpecificOutput": {"hookEventName": "SessionStart", "additionalContext": "\n".join(lines)}}))
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
    files = [f for f in git("diff", "--cached", "--name-only", *base, cwd=root).splitlines() if f]
    lines = _gated_lines(git("diff", "--cached", "--numstat", *base, cwd=root), config)

    errors, notes = check_change(files, lines, message, read, config)
    for note in notes:
        print(f"ℹ️  process gate: {note}")
    if errors:
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
    failures: List[Tuple[str, str, List[str]]] = []
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
        errors, notes = check_change(files, lines, message, read, config)
        if errors:
            failures.append((commit, subject, errors))
        escapes.extend((commit, subject, n) for n in notes)

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
    if not failures and not changelog_error:
        report.append("- ✅ every gated commit carries a resolving Design and a clean, same-commit Review")
    annotations = [f"::error title=Process gate {c[:10]}::{e}" for c, _s, errs in failures for e in errs]
    if changelog_error:
        annotations.append(f"::error title=Process gate::{changelog_error}")
    _report(report, annotations)
    return 1 if failures or changelog_error else 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("session-start", "pre-edit", "stop"):
        sub.add_parser(name)
    msg = sub.add_parser("commit-msg")
    msg.add_argument("--amend", action="store_true", help="the commit replaces HEAD (git commit --amend)")
    msg.add_argument("message_file")
    ci = sub.add_parser("ci")
    ci.add_argument("--base", default="")
    ci.add_argument("--head", default="HEAD")
    args = parser.parse_args(argv)

    if args.command in ("session-start", "pre-edit", "stop"):
        raw = sys.stdin.read() if not sys.stdin.isatty() else ""
        handler = {"session-start": cmd_session_start, "pre-edit": cmd_pre_edit, "stop": cmd_stop}[args.command]
        try:
            payload = json.loads(raw) if raw.strip() else {}
            return handler(payload)
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
    if args.command == "commit-msg":
        return cmd_commit_msg(args.message_file, amend=args.amend)
    return cmd_ci(args.base, args.head)


if __name__ == "__main__":
    sys.exit(main())
