"""
scripts/gate_steps.py — the gates CI lists, found once and run locally
(.agent-rfc/designs/governance-enforcement.md, G6c).

    gates(root)        every step tagged `# agentsmith:gate`, from every
                       workflow under .github/workflows and workflow-templates
    runnable(step)     can this one run here, and if not, exactly why
    run(root)          run them, one line each, and a count that separates
                       passed from skipped
    render_table(...)  the same list as the table docs/validation-checklist.md
                       carries, generated so there is no second copy to drift

`run-the-gates-ci-lists` says to run the list CI has, not the one you remember.
The list nobody could run was thirty steps in a workflow; the one in the
checklist was four commands somebody typed out once. This reads the workflow.

The tag goes on a step that has a name and a `run:`:

    - name: "ruff"
      # agentsmith:gate
      run: ruff check scripts/

    - name: "ShellCheck"
      # agentsmith:gate needs=shellcheck
      run: shellcheck hooks/*

A tag on a `uses:` step is an error: an action is not a script anyone can run
here, and a tag that quietly does nothing is worse than no tag at all.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple

TAG = "# agentsmith:gate"
# A job's `services:` is a coarse answer — it has steps that never touch the
# database. A step says so for itself with `# agentsmith:gate no-services`.
NO_SERVICES = "no-services"
BEGIN = "<!-- BEGIN agentsmith:gates -->"
END = "<!-- END agentsmith:gates -->"
# This repo's own CI. `workflow-templates/` is tagged too, but those are
# TEMPLATES — they are a tenant's CI once synced, not this repo's, and running
# them here would run a tenant's build. A test parses them where they live.
WORKFLOW_DIRS = (".github/workflows",)

# What a local run can resolve, written out rather than evaluated. GitHub
# expressions are a language; this is the handful of them that have an honest
# local answer. Everything else makes the step skipped, by name.
_EXPRESSIONS = {
    "github.workspace": lambda root: str(root),
    "github.sha": lambda root: _git(root, "rev-parse", "HEAD"),
    "github.event.pull_request.head.sha || github.sha": lambda root: _git(root, "rev-parse", "HEAD"),
    "github.event.pull_request.base.sha || github.event.before": lambda root: _merge_base(root),
}
_EXPRESSION = re.compile(r"\$\{\{\s*(.*?)\s*\}\}")


def _git(root: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True,
                          check=False).stdout.strip()


def _merge_base(root: Path) -> str:
    """What CI would call the base of this range: where this branch left the
    upstream one. Falls back to the previous commit, which is what a one-commit
    branch's range is anyway."""
    upstream = _git(root, "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}") or "origin/main"
    return _git(root, "merge-base", upstream, "HEAD") or _git(root, "rev-parse", "HEAD~1")


class Step(NamedTuple):
    """One tagged step: what CI runs, and what it would take to run it here."""

    workflow: str
    job: str
    name: str
    script: str
    env: Dict[str, str]
    needs: Optional[str]
    services: bool
    working_directory: Optional[str]


# ── reading the workflows ────────────────────────────────────────────────────
#
# The tag is a COMMENT, and a YAML parse drops comments. So the structure comes
# from the parser (scripts are block scalars, env is a mapping — not things to
# re-implement with regexes) and the tags come from a text scan that maps each
# tagged line to the step name above or below it.


def _item_bounds(lines: List[str], tag_line: int) -> Tuple[int, int]:
    """The list item (the step) that encloses this line.

    Bounded on purpose: searching outwards for the nearest `name:` walks out of
    the step and finds the JOB's name, and a tag would then be attributed to a
    step that does not exist — or, worse, to a real step somewhere else with
    that name.
    """
    indent = len(lines[tag_line]) - len(lines[tag_line].lstrip())
    start = 0
    for number in range(tag_line, -1, -1):
        stripped = lines[number].lstrip()
        if stripped.startswith("- ") or stripped == "-":
            if len(lines[number]) - len(stripped) < indent:
                start = number
                break
    else:
        start = tag_line
    item_indent = len(lines[start]) - len(lines[start].lstrip())
    end = len(lines)
    for number in range(start + 1, len(lines)):
        line = lines[number]
        if not line.strip():
            continue
        current = len(line) - len(line.lstrip())
        if current < item_indent or (current == item_indent and line.lstrip().startswith("-")):
            end = number
            break
    return start, end


def _tagged_names(text: str) -> Dict[str, Optional[str]]:
    """-> {step name: the tag's argument}. A tag on a step with no name raises:
    silently dropping it would be a gate nobody runs, and guessing the name
    from outside the step would be worse."""
    lines = text.splitlines()
    found: Dict[str, Optional[str]] = {}
    for number, line in enumerate(lines):
        if TAG not in line:
            continue
        argument = line.split(TAG, 1)[1].strip() or None
        start, end = _item_bounds(lines, number)
        name = None
        for candidate in lines[start:end]:
            match = re.match(r"^\s*-?\s*name:\s*(.+?)\s*$", candidate)
            if match:
                name = match.group(1).strip('"\'')
                break
        if name is None:
            raise ValueError(f"a `{TAG}` tag sits on a step with no name — name it, "
                             "or nothing can report which gate ran")
        found[name] = argument
    return found


def _steps_of(document: dict):
    for job in (document.get("jobs") or {}).values():
        if not isinstance(job, dict):
            continue
        for step in job.get("steps") or []:
            if isinstance(step, dict):
                yield job, step


def _workflow_files(root: Path, dirs: Tuple[str, ...] = WORKFLOW_DIRS) -> List[Path]:
    found: List[Path] = []
    for directory in dirs:
        base = root / directory
        if base.is_dir():
            found.extend(sorted(p for p in base.glob("*.yml")))
            found.extend(sorted(p for p in base.glob("*.yaml")))
    return found


def gates(root: Path, dirs: Tuple[str, ...] = WORKFLOW_DIRS) -> List[Step]:
    """Every tagged step, in the order the workflows list them."""
    import yaml

    found: List[Step] = []
    for path in _workflow_files(root, dirs):
        text = path.read_text(encoding="utf-8")
        if TAG not in text:
            continue
        tagged = _tagged_names(text)
        document = yaml.safe_load(text) or {}
        seen = set()
        for job, step in _steps_of(document):
            name = str(step.get("name") or "")
            if name not in tagged:
                continue
            seen.add(name)
            if "run" not in step:
                raise ValueError(f"`{name}` in {path.name} carries `{TAG}` but runs nothing — "
                                 "an action is not a script this can run")
            env = {str(k): str(v) for k, v in {**(job.get("env") or {}),
                                               **(step.get("env") or {})}.items()}
            found.append(Step(
                workflow=str(path.relative_to(root)),
                job=str(job.get("name") or "").strip('"\''),
                name=name,
                script=str(step["run"]),
                env=env,
                needs=tagged[name],
                services=bool(job.get("services")),
                working_directory=step.get("working-directory")
                or (job.get("defaults") or {}).get("run", {}).get("working-directory"),
            ))
        missing = set(tagged) - seen
        if missing:
            raise ValueError(f"{path.name} tags {', '.join(sorted(missing))}, which is not a step "
                             "with that name — the tag and the step have drifted apart")
    return found


# ── what can run here ────────────────────────────────────────────────────────


def _resolve(value: str, root: Path) -> Tuple[Optional[str], Optional[str]]:
    """-> (value with expressions filled in, the one that could not be)."""
    unresolved = None

    def swap(match: "re.Match[str]") -> str:
        nonlocal unresolved
        expression = match.group(1)
        if expression in _EXPRESSIONS:
            return _EXPRESSIONS[expression](root)
        unresolved = expression
        return match.group(0)

    filled = _EXPRESSION.sub(swap, value)
    return (None, unresolved) if unresolved else (filled, None)


def runnable(step: Step, root: Path, services: bool) -> Tuple[bool, str]:
    """Can this gate run here? The reason is the message a person reads, so it
    names the thing that is missing rather than saying "skipped"."""
    for value in (step.script, *step.env.values(), step.working_directory or ""):
        _filled, unresolved = _resolve(value, root)
        if unresolved:
            return False, f"needs CI context — ${{{{ {unresolved} }}}} has no local answer"
    if step.services and not services and step.needs != NO_SERVICES:
        return False, ("needs services this job starts in CI (a database, a broker) — start them and "
                       "re-run with --services")
    if step.needs and step.needs != NO_SERVICES:
        want = step.needs.split("=", 1)[1] if "=" in step.needs else step.needs
        if want.startswith("env:"):
            if not os.environ.get(want[4:]):
                return False, f"needs ${want[4:]}, which is not set here"
        elif shutil.which(want) is None:
            return False, f"needs {want}, which is not installed here"
    return True, ""


# ── running them ─────────────────────────────────────────────────────────────


def run(root: Path, only: Optional[str] = None, services: bool = False,
        fail_fast: bool = False, allow_install: bool = False, repo: Optional[Path] = None) -> int:
    """Run every gate that can run here. -> the exit code the caller returns.

    Passed, failed and skipped are three counts, never two: a gate that could
    not run is not a gate that passed (`ambiguous-signals`). A tool that is not
    installed is an INFRASTRUCTURE answer and lands in `skipped`; only a gate
    that ran and said no is a failure (P13).
    """
    root = Path(root)
    where = Path(repo or root)
    steps = [s for s in gates(root) if not only or only.lower() in s.name.lower()]
    if not steps:
        print(f"gates: nothing tagged `{TAG}`" + (f" matches {only!r}" if only else ""))
        return 0
    passed = failed = skipped = 0
    for step in steps:
        can, why = runnable(step, where, services)
        if not can:
            print(f"  ⏭️  {step.name} — skipped: {why}")
            skipped += 1
            continue
        script, _ = _resolve(step.script, where)
        dropped: List[str] = []
        if not allow_install:
            script, dropped = _without_installs(script or "")
        environment = dict(os.environ)
        for key, value in step.env.items():
            resolved, _ = _resolve(value, where)
            environment[key] = resolved if resolved is not None else value
        directory, _ = _resolve(step.working_directory or str(where), where)
        print(f"  ▶️  {step.name}  ({step.workflow} · {step.job})")
        for line in dropped:
            print(f"      (not run — this is not a fresh runner: {line})")
        result = subprocess.run(["bash", "-eo", "pipefail", "-c", script or ""],
                                cwd=directory or str(where), env=environment, check=False)
        if result.returncode == 0:
            passed += 1
            print(f"  ✅ {step.name}")
        elif result.returncode == COULD_NOT_RUN:
            skipped += 1
            print(f"  ⏭️  {step.name} — skipped: a command it runs is not installed here "
                  "(exit 127). That is this machine, not this code.")
        else:
            failed += 1
            print(f"  ❌ {step.name} — exit {result.returncode}")
            if fail_fast:
                break
    print(f"\ngates: {passed} passed, {failed} failed, {skipped} skipped "
          f"(of {len(steps)} tagged in the workflows)")
    # What ran here, and against which commit — `verify_system.py --governed`
    # reads it to tell "green" from "green for a different commit".
    try:
        import gate_history

        gate_history.record_gates_run(where, passed=passed, failed=failed, skipped=skipped, only=only)
    except Exception:  # fail-open: recording the run must not fail the run
        pass
    return 1 if failed else 0


# ── the table the checklist carries ──────────────────────────────────────────


# A CI runner starts empty, so a step installs its tooling before checking
# anything. This machine is not a fresh runner: running those lines would
# change whatever environment happens to be active, which is not what someone
# asking "do the gates pass?" agreed to. They are dropped, out loud.
_INSTALLS = ("pip install", "pip3 install", "uv pip install", "npm ci", "npm install",
             "pnpm install", "yarn install", "go install", "apt-get", "brew install")
COULD_NOT_RUN = 127  # bash: command not found


def _without_installs(script: str) -> Tuple[str, List[str]]:
    """-> (what to run, the install lines dropped)."""
    kept, dropped = [], []
    for line in script.splitlines():
        if any(line.strip().startswith(prefix) for prefix in _INSTALLS):
            dropped.append(line.strip())
        else:
            kept.append(line)
    return "\n".join(kept), dropped


def _escape(text: str) -> str:
    """A pipe inside a cell ends the cell — `xargs -0 | python3` would split a
    row into two columns and the table would render as nonsense."""
    return text.replace("|", "\\|")


def render_table(steps: List[Step], note: bool = True) -> str:
    """Generated into docs/validation-checklist.md between the markers. The
    checklist used to carry four commands somebody typed out, against a
    workflow with thirty steps (`pin-unremovable-duplicates`)."""
    lines = [""]
    if note:  # for the file; a terminal does not need to be told not to edit it
        lines += [f"<!-- Generated by scripts/generate-ide-config.py --gates from the `{TAG}` tags."
                  " Do not edit by hand. -->", ""]
    lines += [
        "| Gate | Workflow · job | Runs | Locally |",
        "|---|---|---|---|",
    ]
    for step in steps:
        # A one-line step is worth quoting; a heredoc flattened with semicolons
        # is not a command anyone can run, and printing it as one would be a
        # small lie in a table about what runs. The job column names the file.
        script = [line for line in step.script.strip().splitlines() if line.strip()]
        command = f"`{_escape(script[0])}`" if len(script) == 1 else "(script)"
        locally = "yes"
        if step.needs == NO_SERVICES:
            locally = "yes"
        elif step.services:
            locally = "needs services"
        elif step.needs:
            want = step.needs.split("=", 1)[1] if "=" in step.needs else step.needs
            locally = f"needs {want}"
        lines.append(f"| {_escape(step.name)} | `{step.workflow}` · {_escape(step.job)} | "
                     f"{command} | {locally} |")
    lines.append("")
    return "\n".join(lines)
