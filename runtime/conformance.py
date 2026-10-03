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
import re
import shlex
import subprocess
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

CONTRACT_VERSION = 1  # the default a run is scored against; 2 adds `ci`
CONTRACTS = (1, 2, 3)
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
    # A verb's arguments, after the event name: `kg impact` is ["impact"].
    args: list[str] = Field(default_factory=list)


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
    done = subprocess.run([*shlex.split(provider), case.event_name, *case.args], input=json.dumps(case.event),
                          capture_output=True, text=True, cwd=root, check=False)
    return done.returncode, done.stdout.strip(), done.stderr.strip()


def _judge_impact(case: Case, out: str) -> Result:
    """`kg impact` answers a scope, not a decision: the staged files among those
    to read, lever groups, and a `kg:` hash (contract/gate/v3/kg_impact.schema.json)."""
    try:
        answer = json.loads(out)
        files, groups, query = answer["files"], answer["groups"], answer["query"]
    except (ValueError, KeyError, TypeError):
        first = out.splitlines()
        return Result(case, ok=False, why=f"not a scope: {first[0] if first else '(no output)'}")
    staged = set(case.arrange.get("stage") or [])
    if not (isinstance(files, list) and isinstance(groups, list) and staged <= set(files)):
        return Result(case, ok=False, actual="impact", why=f"the staged files {sorted(staged)} are not among {files}")
    if not (isinstance(query, str) and re.fullmatch(r"kg:[0-9a-f]{12}", query)):
        return Result(case, ok=False, actual="impact", why=f"not a kg: hash: {query!r}")
    return Result(case, ok=True, actual="impact")


def _judge(case: Case, code: int, out: str, err: str) -> Result:
    if code == CANNOT_RUN:
        return Result(case, ok=False, unavailable=True,
                      why=f"cannot run here (exit {CANNOT_RUN}): {err.splitlines()[0] if err else 'no reason given'}")
    if case.expect == "impact":
        return _judge_impact(case, out)
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


def _arrange(root: Path, arrange: dict) -> None:
    """Put the fixture in the state a case asks about: a tagged commit checked
    out (`checkout`, discarding what earlier cases wrote), files written, and
    files staged — for the `commit` and `kg` cases of contract 3."""
    # No hook override: the fixture was made with `--template=` and has no
    # .githooks/, so its own commits meet no gate — which is the point of `commit`.
    git = ["git", "-C", str(root), "-c", "user.name=conformance", "-c", "user.email=c@x"]
    if arrange.get("checkout"):
        subprocess.run([*git, "checkout", "-q", "-f", "--detach", arrange["checkout"]], check=True, capture_output=True)
        subprocess.run([*git, "reset", "-q"], check=True, capture_output=True)
    for tag in arrange.get("drop") or []:
        subprocess.run([*git, "tag", "-d", tag], check=False, capture_output=True)
    for rel, body in (arrange.get("write") or {}).items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body, encoding="utf-8")
    if arrange.get("stage"):
        subprocess.run([*git, "add", "--", *arrange["stage"]], check=True, capture_output=True)
    if arrange.get("commit"):
        # A commit made WITHOUT the gate — the fixture's own git, no hooks — which is
        # exactly the history the `push` event exists to stop leaving the machine.
        for rel, body in arrange["commit"]["files"].items():
            (root / rel).write_text(body, encoding="utf-8")
        subprocess.run([*git, "add", "-A"], check=True, capture_output=True)
        subprocess.run([*git, "commit", "-q", "-m", arrange["commit"]["message"]], check=True, capture_output=True)


def run(provider: str, workdir: Path, contract: int = CONTRACT_VERSION) -> Report:
    """Build the fixture, replay every case against `provider`, and report."""
    root = build_fixture(Path(workdir), contract)
    results = []
    for case in cases(contract):
        _arrange(root, case.arrange)
        results.append(_judge(case, *_ask(provider, case, root)))
    return Report(provider=provider, results=results, contract=contract)


# ── The record contract (contract/record/v1/protocol.md) ─────────────────────
#
# Two parties, two runs. A receiver is sent the contract's cases and judged by
# status; a sender is run against the gate fixture with a loopback receiver in
# front of it, and judged by what it sent and how it answered
# (.agent-rfc/designs/record-contract.md).

RECORD_CONTRACT = 1
# What AgentSmith's provider reads; another provider names its own (--url-env, --token-env).
SENDER_URL_ENV = "AGENTSMITH_PORTAL_URL"
SENDER_TOKEN_ENV = "AGENTSMITH_PORTAL_INGEST_TOKEN"
RECEIVER_TOKEN_ENV = "GOVERNANCE_RECORD_TOKEN"


def _record_dir() -> Path:
    for root in (Path(os.environ["AGENTSMITH_DIR"]) if os.environ.get("AGENTSMITH_DIR") else None,
                 Path(__file__).resolve().parent.parent, Path.home() / ".agent-framework"):
        if root is not None and (root / "contract" / "record" / f"v{RECORD_CONTRACT}" / "cases.json").is_file():
            return root / "contract" / "record" / f"v{RECORD_CONTRACT}"
    raise FileNotFoundError(f"contract/record/v{RECORD_CONTRACT}/ not found in $AGENTSMITH_DIR, beside this "
                            "package, or ~/.agent-framework — re-run install-ai-stack.sh")


class RecordCase(BaseModel):
    """One case from contract/record/v1/cases.json — data, validated on the way in."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    token: str
    expect: int
    why: str = ""
    set: dict = Field(default_factory=dict)
    repeat_commits: int = 0
    pad: int = 0


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    why: str = ""


@dataclass(frozen=True)
class RecordReport:
    party: str
    target: str
    checks: list[Check]

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(c.ok for c in self.checks)

    def render(self) -> str:
        lines = [f"record contract v{RECORD_CONTRACT} — {self.party}: {self.target}", ""]
        lines += [f"  {'✅' if c.ok else '❌'} {c.name}" + (f" — {c.why}" if c.why else "") for c in self.checks]
        kept = sum(1 for c in self.checks if c.ok)
        lines += ["", f"{kept}/{len(self.checks)} checks" + ("" if self.passed else " — not conformant")]
        return "\n".join(lines)


def record_cases() -> list[RecordCase]:
    data = json.loads((_record_dir() / "cases.json").read_text(encoding="utf-8"))
    return [RecordCase.model_validate({k: v for k, v in case.items() if not k.startswith("_")})
            for case in data["cases"]]


def record_body(case: RecordCase) -> bytes:
    """The fixture's record with the one thing this case changes."""
    import copy

    body = copy.deepcopy(json.loads((_record_dir() / "fixture.json").read_text(encoding="utf-8"))["record"])
    for path, value in case.set.items():
        *parents, last = path.split(".")
        node = body
        for key in parents:
            node = node[int(key)] if isinstance(node, list) else node[key]
        if isinstance(node, list):
            node[int(last)] = value
        else:
            node[last] = value
    if case.repeat_commits:
        body["commits"] = [body["commits"][0]] * case.repeat_commits
    if case.pad:
        body["_pad"] = "x" * case.pad
    return json.dumps(body).encode("utf-8")


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[override]
        return None  # the redirect's status is the answer; nothing follows it


def run_record_receiver(url: str, token: str) -> RecordReport:
    """Send every case to the receiver at `url` and judge each by its status."""
    import secrets
    import urllib.error

    opener = urllib.request.build_opener(_NoRedirect)
    checks = []
    for case in record_cases():
        headers = {"content-type": "application/json"}
        if case.token == "valid":
            headers["authorization"] = f"Bearer {token}"
        elif case.token == "unknown":
            headers["authorization"] = f"Bearer conformance-unknown-{secrets.token_hex(8)}"
        request = urllib.request.Request(url, data=record_body(case), method="POST", headers=headers)
        try:
            with opener.open(request, timeout=60) as response:
                status = response.status
        except urllib.error.HTTPError as exc:
            status = exc.code
        except (urllib.error.URLError, OSError) as exc:
            checks.append(Check(case.name, False, f"no answer: {exc}"))
            continue
        checks.append(Check(case.name, status == case.expect,
                            "" if status == case.expect else f"expected {case.expect}, got {status}"))
    return RecordReport(party="receiver", target=url, checks=checks)


def run_record_sender(provider: str, workdir: Path, url_env: str = SENDER_URL_ENV,
                      token_env: str = SENDER_TOKEN_ENV) -> RecordReport:
    """Run the provider's `ci` four times, against a loopback receiver that
    stores, redirects, refuses the token, and is unwell."""
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    received: list[tuple[str, bytes]] = []
    elsewhere: list[str] = []
    mode = {"status": 200, "location": ""}

    class Receiver(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("content-length", 0)))
            if self.server is other:
                elsewhere.append(self.headers.get("authorization", ""))
            else:
                received.append((self.headers.get("authorization", ""), body))
            status = 200 if self.server is other else mode["status"]
            payload = json.dumps({"stored": 1} if status == 200 else {"error": "conformance"}).encode()
            self.send_response(status)
            if status in (301, 302, 307, 308):
                self.send_header("location", mode["location"] + self.path)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Receiver)
    other = ThreadingHTTPServer(("127.0.0.1", 0), Receiver)
    for s in (server, other):
        threading.Thread(target=s.serve_forever, daemon=True).start()
    root = build_fixture(Path(workdir), 2)
    token = "conformance-record-token"
    env = {**os.environ, url_env: f"http://127.0.0.1:{server.server_port}", token_env: token}

    def decide() -> Optional[str]:
        done = subprocess.run([*shlex.split(provider), "ci"], cwd=root, env=env, capture_output=True, text=True,
                              input=json.dumps({"kind": "range", "base": "fixture", "head": "clean"}), check=False)
        try:
            return str(json.loads(done.stdout)["decision"])
        except (ValueError, KeyError, TypeError):
            return None

    checks = []
    try:
        mode.update(status=200)
        decision = decide()
        problem = _record_schema_problem([body for _auth, body in received])
        checks.append(Check("a stored record is sent, and satisfies the schema",
                            decision == "allow" and bool(received) and problem is None,
                            f"decision {decision}, {len(received)} request(s){'; ' + problem if problem else ''}"
                            if not (decision == "allow" and received and problem is None) else ""))
        bearer = bool(received) and all(auth == f"Bearer {token}" for auth, _body in received)
        checks.append(Check("it carries the token as a bearer", bearer))
        received.clear()
        mode.update(status=302, location=f"http://127.0.0.1:{other.server_port}")
        decision = decide()
        checks.append(Check("a redirect is not followed, and fails the answer",
                            decision == "deny" and not elsewhere,
                            "" if decision == "deny" and not elsewhere else
                            f"decision {decision}; the redirect target was sent {len(elsewhere)} request(s)"))
        mode.update(status=401)
        decision = decide()
        checks.append(Check("a refused token fails the answer", decision == "deny", f"decision {decision}"
                            if decision != "deny" else ""))
        mode.update(status=503)
        decision = decide()
        checks.append(Check("an unwell receiver does not fail the answer", decision == "allow",
                            f"decision {decision}" if decision != "allow" else ""))
    finally:
        server.shutdown()
        other.shutdown()
    return RecordReport(party="sender", target=provider, checks=checks)


def _record_schema_problem(bodies: list[bytes]) -> Optional[str]:
    """Why a sent body does not satisfy record.schema.json, or None."""
    import jsonschema

    schema = json.loads((_record_dir() / "record.schema.json").read_text(encoding="utf-8"))
    for number, body in enumerate(bodies, start=1):
        try:
            jsonschema.validate(json.loads(body), schema)
        except (ValueError, jsonschema.ValidationError) as exc:
            return f"request {number}: {getattr(exc, 'message', exc)}"
    return None
