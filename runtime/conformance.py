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


# ── The rules contract (contract/rules/v1/) ──────────────────────────────────
#
# A rules provider renders the files every IDE's agent reads and checks that a
# repository still holds them. The cases are asked about a fixture repository
# built fresh for each one; placing a render uses the contract's own placement
# rules — the ones every caller uses (.agent-rfc/designs/rules-contract.md).

RULES_CONTRACT = 1


def _rules_dir() -> Path:
    for root in (Path(os.environ["AGENTSMITH_DIR"]) if os.environ.get("AGENTSMITH_DIR") else None,
                 Path(__file__).resolve().parent.parent, Path.home() / ".agent-framework"):
        if root is not None and (root / "contract" / "rules" / f"v{RULES_CONTRACT}" / "cases.json").is_file():
            return root / "contract" / "rules" / f"v{RULES_CONTRACT}"
    raise FileNotFoundError(f"contract/rules/v{RULES_CONTRACT}/ not found in $AGENTSMITH_DIR, beside this "
                            "package, or ~/.agent-framework — re-run install-ai-stack.sh")


class RulesCase(BaseModel):
    """One case from contract/rules/v1/cases.json — data, validated on the way in."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    verb: str = Field(min_length=1)
    arrange: list[dict] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    expect: dict
    why: str = ""


@dataclass(frozen=True)
class RulesReport:
    provider: str
    checks: list[Check]

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(c.ok for c in self.checks)

    def render(self) -> str:
        lines = [f"rules contract v{RULES_CONTRACT} — provider: {self.provider}", ""]
        lines += [f"  {'✅' if c.ok else '❌'} {c.name}" + (f" — {c.why}" if c.why else "") for c in self.checks]
        kept = sum(1 for c in self.checks if c.ok)
        lines += ["", f"{kept}/{len(self.checks)} cases" + ("" if self.passed else " — not conformant")]
        return "\n".join(lines)


def rules_cases() -> list[RulesCase]:
    data = json.loads((_rules_dir() / "cases.json").read_text(encoding="utf-8"))
    return [RulesCase.model_validate(case) for case in data["cases"]]


def rules_fixture() -> dict:
    return json.loads((_rules_dir() / "fixture.json").read_text(encoding="utf-8"))


def _placement():
    """scripts/rules_port.py beside the contract: the placement rules a caller
    applies, one implementation for adopt, sync and this runner."""
    import importlib.util

    source = _rules_dir().parents[2] / "scripts" / "rules_port.py"
    spec = importlib.util.spec_from_file_location("rules_port_conformance", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_rules_fixture(workdir: Path) -> Path:
    import shutil

    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True)
    for rel, text in rules_fixture()["files"].items():
        (workdir / rel).parent.mkdir(parents=True, exist_ok=True)
        (workdir / rel).write_text(text, encoding="utf-8")
    subprocess.run(["git", "init", "-q", "--template=", str(workdir)], check=True)
    return workdir


_JUNK_ENV = ("AGENT_OWNER_ID", "AGENT_PHOENIX_ENDPOINT", "FRAMEWORK_VERSION")


def _ask_rules(provider: str, verb: str, root: Path, env: dict[str, str]) -> tuple[int, str]:
    base = {k: v for k, v in os.environ.items() if k not in _JUNK_ENV}
    try:
        done = subprocess.run([*shlex.split(provider), verb], cwd=root, input=json.dumps({"cwd": str(root)}),
                              capture_output=True, text=True, check=False, timeout=120, env={**base, **env})
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 127, str(exc)
    return done.returncode, done.stdout


def _schema_problem(name: str, answer: object) -> Optional[str]:
    import jsonschema

    schema = json.loads((_rules_dir() / name).read_text(encoding="utf-8"))
    try:
        jsonschema.validate(answer, schema)
    except jsonschema.ValidationError as exc:
        return f"{'/'.join(str(p) for p in exc.absolute_path) or '(root)'}: {exc.message[:200]}"
    return None


def _render(provider: str, root: Path, env: Optional[dict] = None) -> tuple[Optional[dict], str]:
    """(the render, why not)."""
    code, out = _ask_rules(provider, "render", root, env or {})
    try:
        answer = json.loads(out)
    except ValueError:
        return None, f"no JSON answer (exit {code})"
    problem = _schema_problem("rendered.schema.json", answer)
    if problem:
        return None, f"not a render the schema allows — {problem}"
    paths = [f["path"].lower() for f in answer["files"]]
    if len(paths) != len(set(paths)):
        return None, "a path is rendered twice"
    return answer, ""


def _arrange_rules(provider: str, root: Path, steps: list[dict], fixture: dict) -> Optional[str]:
    """Apply a case's `arrange`; returns the path an edit or delete touched."""
    port = _placement()
    target = None
    rendered: Optional[dict] = None
    for step in steps:
        op = step["op"]
        if op == "place":
            rendered, why = _render(provider, root)
            if rendered is None:
                raise ValueError(f"could not place the render: {why}")
            for file in rendered["files"]:
                path = root / file["path"]
                existing = path.read_text(encoding="utf-8") if path.is_file() else None
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(port.place(existing, port.gm.RulesFile.model_validate(file)), encoding="utf-8")
        elif op == "edit":
            pick = next((f for f in (rendered or {}).get("files", []) if f["placement"] == step["placement"]), None)
            if pick is None:
                return None  # this provider places nothing that way: the case does not apply
            target = pick["path"]
            path = root / target
            text = path.read_text(encoding="utf-8")
            if step["where"] == "outside":
                text = text.rstrip("\n") + "\n\nThe tenant's own line, after the block.\n"
            elif step["placement"] == "block":
                head, marker, rest = text.partition("<!-- agentsmith:rules:begin")
                line, newline, body = rest.partition("\n")
                text = head + marker + line + newline + "A line nobody rendered.\n" + body
            else:
                text = text + "A line nobody rendered.\n"
            path.write_text(text, encoding="utf-8")
        elif op == "delete":
            pick = next((f for f in (rendered or {}).get("files", []) if f["kind"] == "instructions"), None)
            if pick is None:
                return None
            target = pick["path"]
            (root / target).unlink()
        elif op == "note":
            config = root / ".agenticframework" / "process-gates.json"
            data = json.loads(config.read_text(encoding="utf-8"))
            data["extends"]["rules_extra"][0] = fixture["notes"][0] + " (changed)"
            config.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        else:
            raise ValueError(f"unknown arrange op {op!r}")
    return target


def _judge_render(case: RulesCase, provider: str, root: Path, fixture: dict) -> Check:
    want = case.expect.get("render")
    rendered, why = _render(provider, root, case.env)
    if rendered is None:
        return Check(case.name, False, why)
    files = rendered["files"]
    if want == "valid":
        if not any(f["kind"] == "instructions" for f in files):
            return Check(case.name, False, "no instructions file rendered")
        return Check(case.name, True)
    if want == "same-as-without-env":
        plain, why = _render(provider, root)
        if plain is None:
            return Check(case.name, False, why)
        same = json.dumps(plain, sort_keys=True) == json.dumps(rendered, sort_keys=True)
        return Check(case.name, same, "" if same else f"the render changed with {', '.join(sorted(case.env))} set")
    if want == "notes":
        missing = [f"{f['path']} lacks note {i + 1}" for f in files if f["kind"] == "instructions"
                   for i, note in enumerate(fixture["notes"]) if note not in f["text"]]
        return Check(case.name, not missing, "; ".join(missing[:4]))
    if want == "hand-written-not-whole":
        taken = [f for f in files if f["path"] == fixture["hand_written"] and f["placement"] == "whole"]
        return Check(case.name, not taken, f"{fixture['hand_written']} would be replaced whole" if taken else "")
    raise ValueError(f"unknown render expectation {want!r}")


def _judge_check(case: RulesCase, provider: str, root: Path, target: Optional[str]) -> Check:
    code, out = _ask_rules(provider, "check", root, case.env)
    try:
        answer = json.loads(out)
    except ValueError:
        return Check(case.name, False, f"no JSON answer (exit {code})")
    problem = _schema_problem("check.schema.json", answer)
    if problem:
        return Check(case.name, False, f"not a check result the schema allows — {problem}")
    if answer["decision"] != case.expect["decision"]:
        return Check(case.name, False, f"answered {answer['decision']}, the contract says {case.expect['decision']}")
    if answer["decision"] == "deny" and not answer.get("text"):
        return Check(case.name, False, "a deny must say why")
    if "target" in case.expect:
        found = {f["path"]: f["state"] for f in answer.get("files", [])}.get(target)
        if found != case.expect["target"]:
            return Check(case.name, False, f"{target} is reported {found or 'not at all'}, "
                                           f"the contract says {case.expect['target']}")
        if case.expect["target"] == "drifted" and target not in answer.get("text", ""):
            return Check(case.name, False, f"the deny does not name {target}")
    return Check(case.name, True)


def run_rules(provider: str, workdir: Path) -> RulesReport:
    """Every case of contract/rules/v1/cases.json against `provider`."""
    fixture = rules_fixture()
    checks = []
    for case in rules_cases():
        root = build_rules_fixture(workdir)
        try:
            target = _arrange_rules(provider, root, case.arrange, fixture)
        except ValueError as exc:
            checks.append(Check(case.name, False, str(exc)))
            continue
        if any(step["op"] in ("edit", "delete") for step in case.arrange) and target is None:
            checks.append(Check(case.name, True, "not applicable: this provider places nothing that way"))
            continue
        if case.expect.get("answer") == "none":
            code, out = _ask_rules(provider, case.verb, root, case.env)
            try:
                json.loads(out)
                answered = bool(out.strip())
            except ValueError:
                answered = False
            checks.append(Check(case.name, not answered,
                                f"answered an unknown verb (exit {code})" if answered else ""))
        elif case.verb == "render":
            checks.append(_judge_render(case, provider, root, fixture))
        else:
            checks.append(_judge_check(case, provider, root, target))
    return RulesReport(provider, checks)


# ── The telemetry contract (contract/telemetry/v1/) ──────────────────────────
#
# A wire contract: the emitter is judged by what it EXPORTS, never by its code.
# `--export` reads an OTLP/JSON file; `--emitter` runs a command against a
# loopback OTLP receiver. `telemetry_cases()` is the judge's own test — each
# case changes one thing in the golden export that the judge must notice
# (.agent-rfc/designs/telemetry-contract.md).


@dataclass(frozen=True)
class TelemetryReport:
    source: str
    checks: list
    notes: list[str]

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(c.ok for c in self.checks)

    def render(self) -> str:
        from runtime.telemetry_contract import TELEMETRY_CONTRACT

        lines = [f"telemetry contract v{TELEMETRY_CONTRACT} — {self.source}", ""]
        lines += [f"  {'✅' if c.ok else '❌'} {c.name}" + (f" — {c.why}" if c.why else "") for c in self.checks]
        lines += [f"  ℹ️  {note}" for note in self.notes]
        kept = sum(1 for c in self.checks if c.ok)
        lines += ["", f"{kept}/{len(self.checks)} checks" + ("" if self.passed else " — not conformant")]
        return "\n".join(lines)


def _judge_agrees():
    """Does this judge decide the contract's own cases? Checked first on every
    run: a report from a judge that disagrees with the contract it names is not
    a verdict on the emitter."""
    from runtime.telemetry_contract import Check

    wrong = [f"{case.name}: {why}" for case in telemetry_cases() for ok, why in [judge_case(case)] if not ok]
    return Check("the judge decides the contract's own cases", not wrong, "; ".join(wrong[:2]))


def run_telemetry_export(path: Path) -> TelemetryReport:
    from runtime.telemetry_contract import exports_from_file, judge

    verdict = judge(exports_from_file(path))
    return TelemetryReport(f"export: {path}", [_judge_agrees(), *verdict.checks], verdict.notes)


def run_telemetry_emitter(command: str, timeout: int = 300) -> TelemetryReport:
    """Run `command` with its OTLP export pointed at a loopback receiver, and
    judge what arrived. Its own destinations are removed from its environment,
    so nothing it emits leaves this machine during the run."""
    from runtime.telemetry_contract import OTLP_DESTINATIONS, Check, LoopbackCollector, judge

    with LoopbackCollector() as collector:
        env = {k: v for k, v in os.environ.items() if k not in OTLP_DESTINATIONS}
        env.update({"OTEL_EXPORTER_OTLP_ENDPOINT": collector.endpoint, "OTEL_EXPORTER_OTLP_PROTOCOL": "http/protobuf",
                    "OTEL_METRIC_EXPORT_INTERVAL": "1000"})
        try:
            done = subprocess.run(shlex.split(command), env=env, capture_output=True, text=True, check=False,
                                  timeout=timeout)
            ran = Check("the emitter ran", done.returncode == 0,
                        "" if done.returncode == 0 else f"exit {done.returncode}: {done.stderr.strip()[-300:]}")
        except (OSError, subprocess.TimeoutExpired) as exc:
            ran = Check("the emitter ran", False, str(exc))
        exports, problems = list(collector.exports), list(collector.problems)
    verdict = judge(exports)
    checks = [_judge_agrees(), ran, *verdict.checks]
    if problems:
        checks.append(Check("every body it sent was OTLP", False, "; ".join(problems[:3])))
    return TelemetryReport(f"emitter: {command}", checks, verdict.notes)


class TelemetryCase(BaseModel):
    """One case from contract/telemetry/v1/cases.json — data, validated on the way in."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    change: list[dict] = Field(default_factory=list)
    expect: dict


def _telemetry_dir() -> Path:
    from runtime.telemetry_contract import contract_dir

    return contract_dir()


def telemetry_cases() -> list[TelemetryCase]:
    data = json.loads((_telemetry_dir() / "cases.json").read_text(encoding="utf-8"))
    return [TelemetryCase.model_validate(case) for case in data["cases"]]


def telemetry_exports(case: TelemetryCase) -> list[dict]:
    """fixture.json's exports with the case's changes applied, in order."""
    import copy

    exports = copy.deepcopy(json.loads((_telemetry_dir() / "fixture.json").read_text(encoding="utf-8"))["exports"])

    def edit(attributes: list, change: dict) -> list:
        kept = [a for a in attributes if a.get("key") != change.get("drop")]
        for key, value in (change.get("set") or {}).items():
            kept = [a for a in kept if a.get("key") != key] + [{"key": key, "value": value}]
        return kept

    for change in case.change:
        op = change["op"]
        if op == "empty":
            exports = []
            continue
        for export in exports:
            for block in export.get("resourceSpans", []) + export.get("resourceMetrics", []):
                if op == "resource":
                    resource = block.setdefault("resource", {})
                    resource["attributes"] = edit(resource.get("attributes", []), change)
                for scope in block.get("scopeSpans", []):
                    for span in scope.get("spans", []):
                        if op == "spans" or (op == "span" and span.get("name", "").startswith(change["match"])):
                            span["attributes"] = edit(span.get("attributes", []), change)
                for scope in block.get("scopeMetrics", []):
                    for metric in scope.get("metrics", []):
                        if op == "metric" and metric.get("name") == change["match"]:
                            metric["unit"] = change["unit"]
        if op not in ("resource", "span", "spans", "metric"):
            raise ValueError(f"unknown change {op!r}")
    return exports


def judge_case(case: TelemetryCase) -> tuple[bool, str]:
    """Does AgentSmith's judge decide `case` as the contract says? (ok, why not)"""
    from runtime.telemetry_contract import judge

    verdict = judge(telemetry_exports(case))
    failed = sorted(c.name for c in verdict.checks if not c.ok)
    if case.expect.get("passed"):
        if failed:
            return False, f"judged non-conformant: {', '.join(failed)}"
    elif failed != [case.expect["failed"]]:
        return False, f"failed {failed or 'nothing'}, the contract says exactly {case.expect['failed']!r}"
    note = case.expect.get("note")
    if note and not any(note in n for n in verdict.notes):
        return False, f"the report does not note {note}"
    return True, ""


# ── The evals contract (contract/evals/v1/) ─────────────────────────────────
#
# An evals provider judges a tenant's dataset and answers with a scorecard. The
# judge it calls is the runner's: a stub on loopback that answers with the
# scores the fixture's fixed outputs carry, so a conformance run calls no model
# and never drifts (.agent-rfc/designs/evals-contract.md).

EVALS_CONTRACT = 1
STUB_JUDGE_URL_ENV = "EVALS_STUB_JUDGE_URL"
STUB_JUDGE_KEY_ENV = "EVALS_STUB_JUDGE_KEY"
STUB_JUDGE_KEY = "conformance-stub-judge"
# What a runner's own environment may carry that would decide an eval for it.
_EVALS_JUNK_ENV = ("EVAL_FAIL_BELOW", "FAIRNESS_FAIL_BELOW", "HALLUCINATION_FAIL_ABOVE", "ADVERSARIAL_FAIL_ABOVE",
                   "RAG_POISON_FAIL_ABOVE", "FAIRNESS_PARITY_FAIL_BELOW", "FAIRNESS_SCORE_SPREAD_FAIL_ABOVE",
                   "AGENT_JUDGE_MODEL", "AGENT_MODEL_PROFILE", "AI_STACK_MODE", "EVAL_RPM")


def _evals_dir() -> Path:
    for root in (Path(os.environ["AGENTSMITH_DIR"]) if os.environ.get("AGENTSMITH_DIR") else None,
                 Path(__file__).resolve().parent.parent, Path.home() / ".agent-framework"):
        if root is not None and (root / "contract" / "evals" / f"v{EVALS_CONTRACT}" / "cases.json").is_file():
            return root / "contract" / "evals" / f"v{EVALS_CONTRACT}"
    raise FileNotFoundError(f"contract/evals/v{EVALS_CONTRACT}/ not found in $AGENTSMITH_DIR, beside this "
                            "package, or ~/.agent-framework — re-run install-ai-stack.sh")


class EvalsCase(BaseModel):
    """One case from contract/evals/v1/cases.json — data, validated on the way in."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    verb: str = Field(min_length=1)
    suite: str = Field(min_length=1)
    dataset: Optional[str]
    request: dict = Field(default_factory=dict)
    env: dict[str, str] = Field(default_factory=dict)
    judge: str = Field(default="up", pattern=r"^(up|down)$")
    expect: dict
    why: str = ""


@dataclass(frozen=True)
class EvalsReport:
    provider: str
    checks: list[Check]

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(c.ok for c in self.checks)

    def render(self) -> str:
        lines = [f"evals contract v{EVALS_CONTRACT} — provider: {self.provider}", ""]
        lines += [f"  {'✅' if c.ok else '❌'} {c.name}" + (f" — {c.why}" if c.why else "") for c in self.checks]
        kept = sum(1 for c in self.checks if c.ok)
        lines += ["", f"{kept}/{len(self.checks)} cases" + ("" if self.passed else " — not conformant")]
        return "\n".join(lines)


def evals_cases() -> list[EvalsCase]:
    data = json.loads((_evals_dir() / "cases.json").read_text(encoding="utf-8"))
    return [EvalsCase.model_validate(case) for case in data["cases"]]


def evals_fixture() -> dict:
    return json.loads((_evals_dir() / "fixture.json").read_text(encoding="utf-8"))


def build_evals_fixture(workdir: Path, case: EvalsCase) -> Path:
    import shutil

    fixture = evals_fixture()
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True)
    for rel, text in fixture["files"].items():
        (workdir / rel).parent.mkdir(parents=True, exist_ok=True)
        (workdir / rel).write_text(text, encoding="utf-8")
    if case.dataset is not None:
        dataset = fixture["datasets"][case.dataset]
        if dataset["suite"] != case.suite:
            raise ValueError(f"case {case.name!r} asks {case.suite} about a {dataset['suite']} dataset")
        path = workdir / fixture["paths"][case.suite]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dataset["cases"], indent=2) + "\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", "--template=", str(workdir)], check=True)
    return workdir


class StubJudge:
    """An OpenAI-compatible judge on loopback. It answers every chat completion
    with a verdict built from the markers in the request — `SCORE=`,
    `HALLUCINATION=`, `FAIRNESS=`, the first of each — and refuses (401) a call
    without its key. Use as a context manager; `url` is the base a client
    appends `/chat/completions` to."""

    def __init__(self, key: str = STUB_JUDGE_KEY) -> None:
        import threading
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        markers = {name: re.compile(rf"{name}=([0-9]+(?:\.[0-9]+)?)")
                   for name in ("SCORE", "HALLUCINATION", "FAIRNESS")}
        self.calls = 0
        judge = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                judge.calls += 1
                body = self.rfile.read(int(self.headers.get("Content-Length") or 0)).decode("utf-8", "replace")
                if self.headers.get("Authorization", "") != f"Bearer {key}":
                    return self._send(401, {"error": {"message": "the stub judge needs its key"}})
                found = {name: float(m.group(1)) for name, rx in markers.items() if (m := rx.search(body))}
                score = found.get("SCORE", 0.0)
                verdict: dict = {"correctness": 1 if score >= 0.5 else 0, "tool_accuracy": 1, "score": score,
                                 "quality_notes": "stub judge"}
                if "HALLUCINATION" in found:
                    verdict["hallucination"] = found["HALLUCINATION"]
                if "FAIRNESS" in found:
                    verdict["fairness"] = int(found["FAIRNESS"])
                try:
                    model = json.loads(body).get("model", "stub-judge")
                except ValueError:
                    model = "stub-judge"
                self._send(200, {"id": f"stub-{judge.calls}", "object": "chat.completion", "model": model,
                                 "choices": [{"index": 0, "finish_reason": "stop",
                                              "message": {"role": "assistant", "content": json.dumps(verdict)}}],
                                 "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}})

            def _send(self, status: int, payload: dict) -> None:
                data = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *_args) -> None:
                pass

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}/v1"
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def __enter__(self) -> "StubJudge":
        self._thread.start()
        return self

    def __exit__(self, *_exc) -> None:
        self._server.shutdown()
        self._server.server_close()


def _closed_port_url() -> str:
    """A loopback URL nothing listens on: a judge that does not answer."""
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    return f"http://127.0.0.1:{port}/v1"


def _ask_evals(provider: str, case: EvalsCase, root: Path, judge_url: str) -> tuple[int, str]:
    base = {k: v for k, v in os.environ.items() if k not in _EVALS_JUNK_ENV}
    env = {**base, STUB_JUDGE_URL_ENV: judge_url, STUB_JUDGE_KEY_ENV: STUB_JUDGE_KEY, **case.env}
    request = {"suite": case.suite, "cwd": str(root.resolve()), **case.request}
    try:
        done = subprocess.run([*shlex.split(provider), case.verb], cwd=root, input=json.dumps(request),
                              capture_output=True, text=True, check=False, timeout=300, env=env)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 127, str(exc)
    return done.returncode, done.stdout


def _evals_schema_problem(answer: object) -> Optional[str]:
    import jsonschema

    schema = json.loads((_evals_dir() / "scorecard.schema.json").read_text(encoding="utf-8"))
    try:
        jsonschema.validate(answer, schema)
    except jsonschema.ValidationError as exc:
        return f"{'/'.join(str(p) for p in exc.absolute_path) or '(root)'}: {exc.message[:200]}"
    return None


def _judge_scorecard(case: EvalsCase, code: int, out: str) -> Check:
    try:
        answer = json.loads(out)
    except ValueError:
        return Check(case.name, False, f"no JSON answer (exit {code})")
    if code != 0:
        return Check(case.name, False, f"answered, but exited {code} — a provider that answers exits 0")
    problem = _evals_schema_problem(answer)
    if problem:
        return Check(case.name, False, f"not a scorecard the schema allows — {problem}")
    if answer["suite"] != case.suite:
        return Check(case.name, False, f"a scorecard for {answer['suite']}, asked about {case.suite}")
    want = case.expect["verdict"]
    if answer["verdict"] != want:
        why = f" ({answer['reason']})" if answer.get("reason") else ""
        return Check(case.name, False, f"answered {answer['verdict']}{why}, the contract says {want}")
    if want != "pass" and not answer.get("reason"):
        return Check(case.name, False, f"a {want} must say why")
    for bar in ("threshold", "fail_above"):
        if bar in case.expect and answer.get(bar) != case.expect[bar]:
            return Check(case.name, False, f"applied {bar} {answer.get(bar)}, the contract says {case.expect[bar]}")
    return Check(case.name, True)


def run_evals(provider: str, workdir: Path) -> EvalsReport:
    """Every case of contract/evals/v1/cases.json against `provider`."""
    checks = []
    with StubJudge() as judge:
        for case in evals_cases():
            try:
                root = build_evals_fixture(workdir, case)
            except (KeyError, ValueError) as exc:
                checks.append(Check(case.name, False, f"the contract's own data is broken: {exc}"))
                continue
            code, out = _ask_evals(provider, case, root, judge.url if case.judge == "up" else _closed_port_url())
            if case.expect.get("answer") == "none":
                answered = bool(out.strip())
                checks.append(Check(case.name, not answered,
                                    f"answered an unknown verb (exit {code})" if answered else ""))
            else:
                checks.append(_judge_scorecard(case, code, out))
    return EvalsReport(provider, checks)


# ── The security contract (contract/security/v1/) ──────────────────────────
#
# A security provider checks a repository's own pack, posture and wire, and
# answers with one row per control. The fixture is a tenant, never the
# provider's own repository, so every row about the provider's own code must be
# reported not applicable; and its emitter is the fixture's, posting OTLP/JSON
# with the standard library, so no case depends on a provider's installation
# (.agent-rfc/designs/security-contract.md).

SECURITY_CONTRACT = 1
# What a runner's own environment may carry that would decide a check for it.
_SECURITY_JUNK_ENV = ("PROMPT_GUARD", "INPUT_GUARDRAIL", "MODERATION_HOOK", "MODERATION_HOOK_PATH",
                      "TOOL_ALLOWLIST_STRICT", "TOOL_ALLOWLIST_PATH", "PROMPT_DENYLIST_PATH", "SECURITY_STRICT",
                      "AGENTSMITH_TENANT_ROOT")


def _security_dir() -> Path:
    for root in (Path(os.environ["AGENTSMITH_DIR"]) if os.environ.get("AGENTSMITH_DIR") else None,
                 Path(__file__).resolve().parent.parent, Path.home() / ".agent-framework"):
        if root is not None and (root / "contract" / "security" / f"v{SECURITY_CONTRACT}" / "cases.json").is_file():
            return root / "contract" / "security" / f"v{SECURITY_CONTRACT}"
    raise FileNotFoundError(f"contract/security/v{SECURITY_CONTRACT}/ not found in $AGENTSMITH_DIR, beside this "
                            "package, or ~/.agent-framework — re-run install-ai-stack.sh")


class SecurityCase(BaseModel):
    """One case from contract/security/v1/cases.json — data, validated on the way in."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    verb: str = Field(min_length=1)
    request: dict = Field(default_factory=dict)
    files: dict[str, Optional[str]] = Field(default_factory=dict)
    env: dict[str, str] = Field(default_factory=dict)
    expect: dict
    why: str = ""


@dataclass(frozen=True)
class SecurityReport:
    provider: str
    checks: list[Check]

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(c.ok for c in self.checks)

    def render(self) -> str:
        lines = [f"security contract v{SECURITY_CONTRACT} — provider: {self.provider}", ""]
        lines += [f"  {'✅' if c.ok else '❌'} {c.name}" + (f" — {c.why}" if c.why else "") for c in self.checks]
        kept = sum(1 for c in self.checks if c.ok)
        lines += ["", f"{kept}/{len(self.checks)} cases" + ("" if self.passed else " — not conformant")]
        return "\n".join(lines)


def security_cases() -> list[SecurityCase]:
    data = json.loads((_security_dir() / "cases.json").read_text(encoding="utf-8"))
    return [SecurityCase.model_validate(case) for case in data["cases"]]


def security_fixture() -> dict:
    return json.loads((_security_dir() / "fixture.json").read_text(encoding="utf-8"))


def contract_controls() -> list[str]:
    """The repository controls every provider checks (controls.json)."""
    data = json.loads((_security_dir() / "controls.json").read_text(encoding="utf-8"))
    return [row["id"] for row in data["controls"]]


def build_security_fixture(workdir: Path, case: SecurityCase) -> Path:
    import shutil

    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True)
    files = {**security_fixture()["files"], **case.files}
    for rel, text in files.items():
        if text is None:
            continue
        (workdir / rel).parent.mkdir(parents=True, exist_ok=True)
        (workdir / rel).write_text(text, encoding="utf-8")
    subprocess.run(["git", "init", "-q", "--template=", str(workdir)], check=True)
    return workdir


def _ask_security(provider: str, case: SecurityCase, root: Path) -> tuple[int, str]:
    base = {k: v for k, v in os.environ.items() if k not in _SECURITY_JUNK_ENV}
    request = {"cwd": str(root.resolve()), **case.request}
    try:
        done = subprocess.run([*shlex.split(provider), case.verb], cwd=root, input=json.dumps(request),
                              capture_output=True, text=True, check=False, timeout=600, env={**base, **case.env})
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 127, str(exc)
    return done.returncode, done.stdout


def _security_schema_problem(answer: object, name: str) -> Optional[str]:
    import jsonschema

    schema = json.loads((_security_dir() / name).read_text(encoding="utf-8"))
    try:
        jsonschema.validate(answer, schema)
    except jsonschema.ValidationError as exc:
        return f"{'/'.join(str(p) for p in exc.absolute_path) or '(root)'}: {exc.message[:200]}"
    return None


def _judge_security(case: SecurityCase, code: int, out: str) -> Check:
    try:
        answer = json.loads(out)
    except ValueError:
        return Check(case.name, False, f"no JSON answer (exit {code})")
    if code != 0:
        return Check(case.name, False, f"answered, but exited {code} — a provider that answers exits 0")
    redaction = case.verb == "redaction"
    problem = _security_schema_problem(answer, "redaction-result.schema.json" if redaction else "result.schema.json")
    if problem:
        return Check(case.name, False, f"not a result the schema allows — {problem}")
    want = case.expect["verdict"]
    if answer["verdict"] != want:
        why = f" ({answer['reason']})" if answer.get("reason") else ""
        return Check(case.name, False, f"answered {answer['verdict']}{why}, the contract says {want}")
    if want != "pass" and not answer.get("reason"):
        return Check(case.name, False, f"a {want} must say why")
    if redaction:
        if answer["environment"] != case.request.get("environment"):
            return Check(case.name, False, f"answered for {answer['environment']}, asked about "
                                           f"{case.request.get('environment')}")
        if "leaked" in case.expect and sorted(answer.get("leaked") or []) != sorted(case.expect["leaked"]):
            return Check(case.name, False, f"named {answer.get('leaked')} as leaked, the contract says "
                                           f"{case.expect['leaked']}")
        return Check(case.name, True)
    if want == "not_gradable":
        return Check(case.name, True)
    rows = {row["id"]: row for row in answer["controls"]}
    if not case.request.get("controls"):
        missing = [cid for cid in contract_controls() if rows.get(cid, {}).get("subject") != "repository"]
        if missing:
            return Check(case.name, False, f"no repository row for {', '.join(missing[:5])}")
    ran = [row["id"] for row in answer["controls"]
           if row["subject"] == "provider" and row["result"] != "not_applicable"]
    if ran:
        return Check(case.name, False, f"ran the provider's own code for a tenant: {', '.join(ran[:5])}")
    for cid, result in (case.expect.get("rows") or {}).items():
        got = rows.get(cid, {}).get("result")
        if got != result:
            said = f" ({rows[cid].get('message', '')[:160]})" if cid in rows else ""
            return Check(case.name, False, f"{cid} is {got or 'absent'}{said}, the contract says {result}")
    if case.expect.get("only") and set(rows) != set(case.expect.get("rows") or {}):
        return Check(case.name, False, f"answered {len(rows)} controls, asked about {len(case.expect['rows'])}")
    return Check(case.name, True)


def run_security(provider: str, workdir: Path) -> SecurityReport:
    """Every case of contract/security/v1/cases.json against `provider`."""
    checks = []
    for case in security_cases():
        root = build_security_fixture(workdir, case)
        code, out = _ask_security(provider, case, root)
        if case.expect.get("answer") == "none":
            answered = bool(out.strip())
            checks.append(Check(case.name, not answered, f"answered an unknown verb (exit {code})" if answered else ""))
        else:
            checks.append(_judge_security(case, code, out))
    return SecurityReport(provider, checks)
