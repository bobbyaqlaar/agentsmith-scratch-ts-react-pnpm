"""
runtime/conformance.py — does this command satisfy the gate contract?
(.agent-rfc/designs/gate-port.md, contract/gate/v1/ and v2/ protocol.md)

    cases(contract)                    what the contract asks, as data
    fixture(contract)                  the repository those cases are asked about
    run(provider, workdir, contract)   build it, replay them, report

Each version is scored by its own directory: version 2's cases include
version 1's, so a provider that passes 2 passes 1, and a version-1 provider
keeps the score it had (.agent-rfc/designs/gate-contract-ci.md).

A gate decision is about a repository, so the contract carries the repository
too: the fixture is built here, from the contract's own data, and the provider
is run inside it with the event on stdin. Nothing in this module knows what
AgentSmith would answer — it compares a provider's answer with the contract's,
which is what lets a second platform use it.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

CONTRACT_VERSION = 1  # the default a run is scored against; 2 adds `ci`
CONTRACTS = (1, 2)
DECISIONS = ("allow", "deny", "block", "context")
# A provider saying "not me": the caller tries the next one. Never a decision.
CANNOT_RUN = 3


def _contract_dir(contract: int = CONTRACT_VERSION) -> Path:
    """The contract, beside this package or in the framework it came from —
    the same order `runtime/architectures.py` uses for its catalogue."""
    if contract not in CONTRACTS:
        raise ValueError(f"gate contract {contract} does not exist — choose from {', '.join(map(str, CONTRACTS))}")
    candidates = [
        Path(os.environ["AGENTSMITH_DIR"]) if os.environ.get("AGENTSMITH_DIR") else None,
        Path(__file__).resolve().parent.parent,
        Path.home() / ".agent-framework",
    ]
    for root in candidates:
        if root is not None and (root / "contract" / "gate" / f"v{contract}" / "cases.json").is_file():
            return root / "contract" / "gate" / f"v{contract}"
    raise FileNotFoundError(
        f"contract/gate/v{contract}/ not found in $AGENTSMITH_DIR, beside this package, or "
        "~/.agent-framework — re-run install-ai-stack.sh"
    )


class Case(BaseModel):
    """One case from `cases.json` — data this code did not write, so it is
    validated on the way in rather than trusted (pillar 7)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    event_name: str
    event: dict
    expect: str
    why: str = ""
    arrange: dict = Field(default_factory=dict)


@dataclass(frozen=True)
class Result:
    case: Case
    ok: bool
    actual: Optional[str] = None
    why: str = ""
    unavailable: bool = False


@dataclass(frozen=True)
class Report:
    provider: str
    results: list[Result]
    contract: int = CONTRACT_VERSION

    @property
    def passed(self) -> bool:
        return bool(self.results) and all(r.ok for r in self.results)

    def render(self) -> str:
        lines = [f"gate contract v{self.contract} — {self.provider}", ""]
        for result in self.results:
            mark = "✅" if result.ok else "❌"
            lines.append(f"  {mark} {result.case.name}" + (f" — {result.why}" if result.why else ""))
        kept = sum(1 for r in self.results if r.ok)
        lines += ["", f"{kept}/{len(self.results)} cases" + ("" if self.passed else " — not conformant")]
        return "\n".join(lines)


def cases(contract: int = CONTRACT_VERSION) -> list[Case]:
    data = json.loads((_contract_dir(contract) / "cases.json").read_text(encoding="utf-8"))
    return [Case.model_validate({k: v for k, v in case.items() if not k.startswith("_")})
            for case in data["cases"]]


def fixture(contract: int = CONTRACT_VERSION) -> dict:
    return json.loads((_contract_dir(contract) / "fixture.json").read_text(encoding="utf-8"))


def build_fixture(workdir: Path, contract: int = CONTRACT_VERSION) -> Path:
    """The contract's repository, freshly built. `--template=` so the machine's
    own git hooks stay out of a run that is about the provider, not the machine.
    Version 2's fixture adds a tagged `history` after the first commit (tagged
    `fixture`), so its `ci` cases can name a range."""
    spec = fixture(contract)
    root = Path(workdir)
    root.mkdir(parents=True, exist_ok=True)
    for rel, body in spec["files"].items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body, encoding="utf-8")
    commit = spec.get("commit") or {}
    git = ["git", "-C", str(root), "-c", f"user.name={commit.get('name', 'conformance')}",
           "-c", f"user.email={commit.get('email', 'c@x')}"]
    subprocess.run(["git", "init", "-q", "-b", "main", "--template=", str(root)], check=True)
    subprocess.run([*git, "add", "-A"], check=True, capture_output=True)
    subprocess.run([*git, "commit", "-q", "-m", commit.get("message", "fixture")], check=True, capture_output=True)
    if spec.get("history"):
        subprocess.run([*git, "tag", "fixture"], check=True, capture_output=True)
    for step in spec.get("history") or []:
        for rel, body in step["files"].items():
            target = root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(body, encoding="utf-8")
        subprocess.run([*git, "add", "-A"], check=True, capture_output=True)
        subprocess.run([*git, "commit", "-q", "-m", step["message"]], check=True, capture_output=True)
        subprocess.run([*git, "tag", step["tag"]], check=True, capture_output=True)
    return root


def _ask(provider: str, case: Case, root: Path) -> tuple[int, str, str]:
    done = subprocess.run([*shlex.split(provider), case.event_name], input=json.dumps(case.event),
                          capture_output=True, text=True, cwd=root, check=False)
    return done.returncode, done.stdout.strip(), done.stderr.strip()


def _judge(case: Case, code: int, out: str, err: str) -> Result:
    if code == CANNOT_RUN:
        return Result(case, ok=False, unavailable=True,
                      why=f"cannot run here (exit {CANNOT_RUN}): {err.splitlines()[0] if err else 'no reason given'}")
    try:
        answer = json.loads(out)
        decision, text = answer["decision"], answer.get("text", "")
    except (ValueError, KeyError, TypeError):
        first = (out or err or "").splitlines()
        return Result(case, ok=False, why=f"not a decision: {first[0] if first else '(no output)'}")
    if decision not in DECISIONS:
        return Result(case, ok=False, actual=str(decision), why=f"not a decision: {decision!r}")
    if decision != case.expect:
        return Result(case, ok=False, actual=decision, why=f"expected {case.expect}, got {decision}")
    if decision in ("deny", "block") and not text.strip():
        return Result(case, ok=False, actual=decision, why=f"{decision} with no reason a person could read")
    return Result(case, ok=True, actual=decision)


def run(provider: str, workdir: Path, contract: int = CONTRACT_VERSION) -> Report:
    """Build the fixture, replay every case against `provider`, and report."""
    root = build_fixture(Path(workdir), contract)
    results = []
    for case in cases(contract):
        for rel, body in (case.arrange.get("write") or {}).items():
            (root / rel).write_text(body, encoding="utf-8")
        results.append(_judge(case, *_ask(provider, case, root)))
    return Report(provider=provider, results=results, contract=contract)
