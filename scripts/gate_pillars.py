"""
scripts/gate_pillars.py — the pillars a script can check by itself, and the
policy that says whether this repo is held to them (G6a,
.agent-rfc/designs/governance-enforcement.md).

    parse_policy            `pillars` in .agenticframework/process-gates.json
    transition_problems     the ratchet: the mode only strengthens, and an
                            allowlist entry needs the owner's approval to appear
    mechanical_problems     the checks, over the files one commit touches
    repo_problems           the same checks over everything the repo tracks
    evidence_resolver       does this token name anything? (pillar answers)

Which checks exist is the registry's to say: a check runs only while its pillar
is marked `mechanical` in governance.json. Which repo is held to them is the
repo's, in a file that is read at the commit being checked — a requirement
added to the shared registry would judge every commit ever made by it.
"""

from __future__ import annotations

import ast
import re
import subprocess
from pathlib import Path
from typing import Callable, Dict, List, NamedTuple, Optional, Sequence, Set, Tuple

import gate_models as gm

MODES = ("off", "report", "enforce")
_STRENGTH = {mode: rank for rank, mode in enumerate(MODES)}

Resolver = Callable[[str], bool]
Reader = Callable[[str], Optional[str]]


# ── the policy ───────────────────────────────────────────────────────────────


def parse_policy(value: object) -> Tuple[gm.PillarPolicy, List[str]]:
    """-> (policy, problems). A bare string is the mode: a repo with nothing to
    allowlist should not have to write an object to say `enforce`."""
    if value is None:
        return gm.PillarPolicy(), []
    data = {"mode": value} if isinstance(value, str) else value
    try:
        return gm.PillarPolicy.model_validate(data), []
    except gm.ValidationError as exc:
        error = exc.errors()[0]
        where = ".".join(str(part) for part in error["loc"]) or "pillars"
        # The value it was given, not only the rule it broke: "mode: Input
        # should be 'off', 'report' or 'enforce'" does not say what is there.
        given = repr(error.get("input"))
        return gm.PillarPolicy(), [f"`pillars` is invalid: {where}: {error['msg']} (got {given})"]


def transition_problems(
    old: Optional[gm.PillarPolicy],
    new: gm.PillarPolicy,
    approvals: Sequence[gm.Approval],
) -> List[str]:
    """What this commit does to the policy it inherited.

    A repo declaring one for the first time is seeding it — there is nothing to
    weaken yet. After that the mode only strengthens and the allowlist only
    shrinks, which is also why dropping the key is a weakening: without that,
    the allowlist could be widened by switching the policy off and on again.

    Stated limit: `old` is None when the parent commit carried no policy AND
    when its config could not be parsed at all, so a broken config in between
    reads as a seed. That commit is itself refused by every gate, and the sweep
    finds it if it was made with --no-verify — but it is the one seam here.
    """
    if old is None:
        return []
    problems: List[str] = []
    if _STRENGTH[new.mode] < _STRENGTH[old.mode]:
        problems.append(
            f"`pillars` would weaken from {old.mode!r} to {new.mode!r} — a gate is not turned down to "
            "pass a change (P13); the owner decides that, not the change that fails"
        )
    known = {approval.id for approval in approvals}
    before = {(entry.check, entry.path) for entry in old.allow}
    for entry in new.allow:
        if (entry.check, entry.path) in before:
            continue
        if entry.approval is None:
            problems.append(
                f"`pillars.allow` adds {entry.check} for {entry.path} — an allowlist only shrinks; "
                "the owner records the exception in a terminal (`agentsmith approve`) and the entry "
                "carries `\"approval\": \"A-xxxxxxxx\"`"
            )
        elif entry.approval not in known:
            problems.append(
                f"`pillars.allow` cites {entry.approval} for {entry.check} {entry.path}, which is not in "
                f"{gm.APPROVALS_FILE}"
            )
    return problems


# ── what the checks read ─────────────────────────────────────────────────────


class Source:
    """One file as a check sees it: its text, the text at the parent commit,
    the design covering the change, and what the repo tracks.

    The parse is done once and shared — six checks over one file should cost
    one `ast.parse`, and the text checks should cost none.
    """

    def __init__(self, path: str, text: str, previous: Optional[str] = None,
                 design: str = "", read: Optional[Reader] = None) -> None:
        self.path = path
        self.text = text
        self.previous = previous
        self.design = design
        # The same reader the file came from, for a rule that has to look at a
        # neighbouring file — `next.config.*` above a component. It reads the
        # commit being checked, so the answer is that commit's, not today's.
        self.read: Reader = read or (lambda _path: None)
        self._tree: object = _UNPARSED

    @property
    def tree(self) -> Optional[ast.AST]:
        """The syntax tree, or None when this is not Python or does not parse.
        A file that does not parse is ruff's and the test run's to report."""
        if self._tree is _UNPARSED:
            try:
                self._tree = ast.parse(self.text) if self.path.endswith(".py") else None
            except SyntaxError:
                self._tree = None
        return self._tree  # type: ignore[return-value]


_UNPARSED = object()


def _decorator_name(node: ast.expr) -> str:
    """`@app.get("/x")` -> "app.get"; `@traced` -> "traced"."""
    if isinstance(node, ast.Call):
        return _decorator_name(node.func)
    if isinstance(node, ast.Attribute):
        return f"{_decorator_name(node.value)}.{node.attr}"
    if isinstance(node, ast.Name):
        return node.id
    return ""


_ROUTE_METHODS = {"get", "post", "put", "patch", "delete", "head", "options"}
_COMMAND_DECORATORS = {"command"}
_TRACING_DECORATORS = {"traced", "trace", "instrument", "agent_span"}
_SPAN_CALLS = {"agent_span", "gate_span", "start_as_current_span", "start_span"}


def _entrypoint(node: ast.AST) -> Optional[str]:
    """"route", "CLI command", or None — what a decorator declares this to be."""
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return None
    kind = None
    for decorator in node.decorator_list:
        name = _decorator_name(decorator)
        last = name.rsplit(".", 1)[-1]
        if "." in name and last in _ROUTE_METHODS:
            kind = "route"
        elif "." in name and last in _COMMAND_DECORATORS:
            kind = "CLI command"
    return kind


def _is_dataclass(node: ast.ClassDef) -> bool:
    return any(_decorator_name(d).rsplit(".", 1)[-1] == "dataclass" for d in node.decorator_list)


def _annotation_name(node: Optional[ast.expr]) -> str:
    return _decorator_name(node).rsplit(".", 1)[-1] if node is not None else ""


# ── the checks ───────────────────────────────────────────────────────────────


# What a validated source looks like: the output of a Pydantic model's own
# dump. `X(**json.loads(raw))` is not this, which is the case that matters.
# `model_dump` only — Pydantic V2's dump, which is what this repo uses. v1's
# `.dict()` would match any object with a `dict()` method, which is a hole.
_VALIDATED_CALLS = {"model_dump"}


def _validated(node: ast.expr) -> bool:
    """Is this expression the dump of a model something already validated?"""
    if isinstance(node, ast.Call):
        if _decorator_name(node.func).rsplit(".", 1)[-1] in _VALIDATED_CALLS:
            return "." in _decorator_name(node.func)  # a method on something, not a bare dict()
        return False
    # `{k: v for ...}` over a dump, which is how a converter drops a field.
    if isinstance(node, (ast.DictComp, ast.Dict)):
        return any(_validated(inner) for inner in ast.walk(node) if isinstance(inner, ast.Call))
    return False


def _p7_pydantic(src: Source) -> List[str]:
    """P7: a model at a boundary is a Pydantic model.

    Narrowed in G6b to what the rule is actually protecting — an object built
    out of data the code did not write, which is where validation belongs. A
    dataclass built by keyword from values in the same module validates
    nothing, so requiring Pydantic there is ritual.

    Unpacking a Pydantic model's own `model_dump()` is not that either: the
    values came out of a model this code just validated. Flagging it would
    teach people to write every field out at the call site to quiet a checker,
    which is worse code and one more copy of the field list.

    Stated limit: a boundary this cannot see is a dataclass filled field by
    field from a parsed payload.
    """
    tree = src.tree
    if tree is None:
        return []
    models = {node.name: node for node in ast.walk(tree)
              if isinstance(node, ast.ClassDef) and _is_dataclass(node)}
    if not models:
        return []
    found: Dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = _decorator_name(node.func).rsplit(".", 1)[-1]
            unpacked = any(k.arg is None and not _validated(k.value) for k in node.keywords) or \
                any(isinstance(a, ast.Starred) and not _validated(a.value) for a in node.args)
            if name in models and unpacked and name not in found:
                found[name] = (f"`{name}` is a dataclass built from data the code did not write "
                               f"(line {node.lineno}) — a model at a boundary is a Pydantic BaseModel, "
                               "which validates what it is handed")
        if _entrypoint(node) == "route":
            arguments = list(node.args.args) + list(node.args.kwonlyargs)  # type: ignore[union-attr]
            for argument in arguments:
                name = _annotation_name(argument.annotation)
                if name in models and name not in found:
                    found[name] = (f"`{name}` is a dataclass used as a request body "
                                   f"(line {argument.lineno}) — a model at a boundary is a Pydantic "
                                   "BaseModel, which validates what it is handed")
    return [found[name] for name in sorted(found)]


def _p7_async(src: Source) -> List[str]:
    """P7: a route handler is `async def`. A synchronous one holds the event
    loop for every other request in flight, which looks like a slow dependency
    rather than like the bug it is."""
    tree = src.tree
    if tree is None:
        return []
    return [f"`{node.name}` is a route declared `def` (line {node.lineno}) — a synchronous handler "
            "blocks the event loop for every other request; `async def`"
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and _entrypoint(node) == "route"]


def _p3_tracing(src: Source) -> List[str]:
    """P3: an entrypoint emits a span.

    Stated limit: an entrypoint that is not declared by a decorator is not
    found, and a handler that delegates to a helper which traces reads as
    untraced. The design-time question covers both; this covers the shape a
    script can see.
    """
    tree = src.tree
    if tree is None:
        return []
    problems = []
    for node in ast.walk(tree):
        kind = _entrypoint(node)
        if not kind:
            continue
        if any(_decorator_name(d).rsplit(".", 1)[-1] in _TRACING_DECORATORS
               for d in node.decorator_list):  # type: ignore[attr-defined]
            continue
        if not any(isinstance(inner, ast.Call)
                   and _decorator_name(inner.func).rsplit(".", 1)[-1] in _SPAN_CALLS
                   for inner in ast.walk(node)):
            problems.append(
                f"`{node.name}` is a {kind} (line {node.lineno}) that opens no span — "  # type: ignore[attr-defined]
                "wrap it in `agent_span(...)`, or every call through it is invisible")
    return problems


# Where a provider SDK belongs. These are part of the rule, not entries in an
# allowlist: "no provider SDK outside the gateway" is one rule, and writing the
# exception down twice would let the two drift. Matched by suffix so a tenant's
# vendored copy counts.
GATEWAY_FILES = ("runtime/llm_gateway.py", "runtime/provider_dispatch.py", "scripts/cost_router.py")
PROVIDER_MODULES = ("openai", "anthropic", "cohere", "mistralai", "litellm", "ollama", "groq",
                    "together", "replicate", "google.generativeai", "vertexai")


def _imported_modules(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name, node.lineno
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            yield node.module, node.lineno


def _p10_gateway(src: Source) -> List[str]:
    """P10: provider SDKs live behind the gateway, where the budget store, the
    degrade ladder and the audit trail are. A call around it is not cheaper —
    it is unbudgeted and unrecorded."""
    tree = src.tree
    if tree is None or src.path.endswith(GATEWAY_FILES):
        return []
    problems = []
    for module, line in _imported_modules(tree):
        if any(module == provider or module.startswith(provider + ".") for provider in PROVIDER_MODULES):
            problems.append(f"imports the provider SDK `{module}` (line {line}) — model calls route through "
                            f"{GATEWAY_FILES[0]}, which is where the budget, the degrade ladder and the "
                            "audit trail are")
    return problems


_TS_ANY = re.compile(r"(?::\s*any\b|\bas\s+any\b|<any>|\bany\[\])")


def _code_lines(text: str):
    """(number, line) for the lines of a TypeScript file that are not comments.
    A line reader, not a parser — `#` is a private field here, not a comment."""
    in_block = False
    for number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if in_block:
            if "*/" in stripped:
                in_block = False
            continue
        if stripped.startswith("/*"):
            in_block = "*/" not in stripped
            continue
        if stripped.startswith("//") or stripped.startswith("*"):
            continue
        yield number, line


def _p7_ts_any(src: Source) -> List[str]:
    """P7: no `any` in TypeScript — it turns the type checker off for
    everything downstream of it.

    Stated limit: a line, not a parse. A string containing the text reads as a
    violation.
    """
    return [f"uses `any` (line {number}) — `unknown` with a narrowing check, or the real type; "
            "`any` switches off every check below it"
            for number, line in _code_lines(src.text) if _TS_ANY.search(line)]


_CLIENT_ONLY = re.compile(r"\buse(State|Effect|Ref|Memo|Callback|Context|Reducer|Router)\b|\bon[A-Z]\w+=")
_NEXT_CONFIG = ("next.config.js", "next.config.mjs", "next.config.ts", "next.config.cjs")


def _in_next_project(path: str, read: "Reader") -> bool:
    """`'use client'` is a Next.js directive: asking a Vite app for it would be
    asking for a mistake."""
    parts = path.split("/")[:-1]
    while True:
        prefix = "/".join(parts)
        if any(read("/".join([prefix, name]) if prefix else name) is not None for name in _NEXT_CONFIG):
            return True
        if not parts:
            return False
        parts.pop()


def _p7_use_client(src: Source) -> List[str]:
    """P7: a component that uses hooks or DOM events runs on the client and has
    to say so, or Next renders it on the server and the build fails there
    instead of here."""
    if not src.path.endswith(".tsx") or not _in_next_project(src.path, src.read):
        return []
    body = src.text.lstrip()
    if body.startswith('"use client"') or body.startswith("'use client'"):
        return []
    found = _CLIENT_ONLY.search(src.text)
    if not found:
        return []
    return [f"uses `{found.group(0).rstrip('=')}` without `'use client'` — a component with hooks or "
            "event handlers runs on the client and declares it on its first line"]


# A shape, not a check against any issuer: a credential that matches none of
# these is not found, and the design-time question still asks.
_SECRETS = (
    ("an Anthropic key", re.compile(r"sk-ant-[A-Za-z0-9_\-]{16,}")),
    ("an OpenAI key", re.compile(r"\bsk-[A-Za-z0-9]{32,}")),
    ("an AWS access key id", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("a private key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("a GitHub token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}")),
    ("a Slack token", re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}")),
    ("a Google API key", re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b")),
)
# What a line that MUST hold one carries — a redaction test, a fixture that
# proves the scrubber works. Greppable, so the exemptions can be counted.
NOT_A_SECRET = "not-a-secret:"


def _p12_secrets(src: Source) -> List[str]:
    """P12: a credential belongs in .env and the CI secret store, never in a
    file git tracks.

    The marker counts on the line itself or on the line above it: the lines
    that need it are long ones inside a fixture, and a rule that only reads the
    end of the line would push them past every formatter this repo runs.
    """
    problems = []
    lines = src.text.splitlines()
    for number, line in enumerate(lines, start=1):
        if NOT_A_SECRET in line or (number > 1 and NOT_A_SECRET in lines[number - 2]):
            continue
        for name, pattern in _SECRETS:
            if pattern.search(line):
                problems.append(
                    f"line {number} looks like {name} — credentials are read by variable name from "
                    f".env and the CI secret store. A line that must hold one (a redaction test) ends "
                    f"`# {NOT_A_SECRET} <why>`")
                break
    return problems


_REQUIREMENT = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*==", re.M)
_UV_PACKAGE = re.compile(r'^\s*name\s*=\s*"([^"]+)"', re.M)
_NODE_PACKAGE = re.compile(r'^\s*"node_modules/([^"]+)":', re.M)


def _packages(text: Optional[str]) -> Set[str]:
    if not text:
        return set()
    found = {name.lower() for name in _REQUIREMENT.findall(text)}
    found |= {name.lower() for name in _NODE_PACKAGE.findall(text)}
    for chunk in text.split("[[package]]")[1:]:
        named = _UV_PACKAGE.search(chunk)
        if named:  # the first `name =` after the header is the package's own
            found.add(named.group(1).lower())
    return found


def _p2_dependencies(src: Source) -> List[str]:
    """P2: a package this change adds to a lock file is named in the design's
    `## Dependencies`. That section exists to answer which dependencies — direct
    and transitive — a change brings in; the lock file is the answer."""
    added = _packages(src.text) - _packages(src.previous)
    if not added:
        return []
    named = src.design.lower()
    return [f"adds `{package}`, which the change's '## Dependencies' does not name — list every "
            "package the lock gains, direct and transitive, or the section is a guess"
            for package in sorted(added) if package not in named]


class Check(NamedTuple):
    """One rule a script can decide.

    `reads` are basename globs — which files this rule is about. `tests` says
    whether test files count: a fixture is not a model, but a real key in a
    test is leaked exactly as far as one in a module. `per_change` marks a rule
    that compares against the parent commit, so it has no answer for "what does
    this repo look like now".
    """

    pillar: int
    reads: Tuple[str, ...]
    run: Callable[[Source], List[str]]
    tests: bool = False
    per_change: bool = False


CHECKS: Dict[str, Check] = {
    "P2-dependencies": Check(2, ("requirements*.txt", "requirements*.lock", "uv.lock", "poetry.lock",
                                 "package-lock.json", "pnpm-lock.yaml"), _p2_dependencies,
                             tests=True, per_change=True),
    "P3-tracing": Check(3, ("*.py",), _p3_tracing),
    "P7-async": Check(7, ("*.py",), _p7_async),
    "P7-pydantic": Check(7, ("*.py",), _p7_pydantic),
    "P7-ts-any": Check(7, ("*.ts", "*.tsx"), _p7_ts_any),
    "P7-use-client": Check(7, ("*.tsx",), _p7_use_client),
    "P10-gateway": Check(10, ("*.py",), _p10_gateway),
    "P12-secrets": Check(12, (), _p12_secrets, tests=True),
}

# Files whose bytes are not text: a credential scan over a PNG finds nothing
# and reads the whole file to do it.
_BINARY = (".png", ".jpg", ".jpeg", ".gif", ".ico", ".pdf", ".woff", ".woff2", ".ttf", ".otf",
           ".zip", ".gz", ".tar", ".whl", ".so", ".dylib", ".bin")


def active_checks(registry: "gm.Registry") -> Dict[str, Check]:
    """The checks this registry turns on. Empty is a real answer: a repo whose
    registry marks no pillar `mechanical` was not checked, which is not the same
    as passing (`ambiguous-signals`)."""
    mechanical = {p.id for p in registry.pillars if "mechanical" in p.check}
    return {name: check for name, check in CHECKS.items() if check.pillar in mechanical}


def first_party(path: str) -> bool:
    """Code this repo is judged on. A fixture is not a model and a test double
    is not a route: flagging test files would teach every repo to allowlist its
    test directory. `P12-secrets` opts back in — see `Check.tests`."""
    parts = path.split("/")
    name = parts[-1]
    return not (name.startswith("test_") or name.endswith(("_test.py", ".test.ts", ".test.tsx"))
                or name == "conftest.py" or {"test", "tests", "__tests__"} & set(parts[:-1]))


def _reads(check: Check, path: str) -> bool:
    from fnmatch import fnmatch

    name = path.split("/")[-1]
    if not check.tests and not first_party(path):
        return False
    if path.endswith(_BINARY):
        return False
    return not check.reads or any(fnmatch(name, pattern) for pattern in check.reads)


def mechanical_problems(
    files: Sequence[str],
    read: Reader,
    registry: "gm.Registry",
    policy: gm.PillarPolicy,
    previous: Optional[Reader] = None,
    design: str = "",
    per_change: bool = True,
) -> List[str]:
    """The checks over the files one commit touches — whole-file, not the added
    lines: how a module is built is not answerable line by line. A repo adopts
    without fixing what it already owns, and pays when it next edits a file."""
    checks = active_checks(registry)
    if not checks:
        return []
    allowed = {(entry.check, entry.path) for entry in policy.allow}
    problems: List[str] = []
    for path in sorted(files):
        wanted = {name: check for name, check in checks.items()
                  if (name, path) not in allowed and (per_change or not check.per_change)
                  and _reads(check, path)}
        if not wanted:
            continue
        text = read(path)
        if text is None:
            continue  # deleted in this commit
        source = Source(path, text, previous(path) if previous else None, design, read)
        for name, check in sorted(wanted.items()):
            problems.extend(f"{name}: {path} {problem}" for problem in check.run(source))
    return problems


def repo_problems(root: Path, registry: "gm.Registry", policy: gm.PillarPolicy) -> List[str]:
    """The same checks over everything the repo tracks: what it would have to
    fix or allowlist, which is what someone asking about the repo means.

    The per-change checks sit this out — "which packages did this change add"
    has no answer when the question is about the whole repo.
    """
    tracked = [f for f in _git(root, "ls-files").splitlines() if f]

    def read(path: str) -> Optional[str]:
        full = root / path
        try:
            return full.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return None

    return mechanical_problems(tracked, read, registry, policy, per_change=False)


def exemption_count(root: Path) -> int:
    """How many lines are exempted by the `not-a-secret:` marker — counted and
    printed, so the number is visible rather than quietly growing.

    A line counts when it carries the marker AND would otherwise be flagged.
    Counting markers instead would count this rule's own documentation, and an
    explanation of an exemption is not an exemption someone took.
    """
    total = 0
    for path in _git(root, "ls-files").splitlines():
        if not path or path.endswith(_BINARY):
            continue
        try:
            lines = (root / path).read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError):
            continue
        for number, line in enumerate(lines, start=1):
            marked = NOT_A_SECRET in line or (number > 1 and NOT_A_SECRET in lines[number - 2])
            if marked and any(pattern.search(line) for _name, pattern in _SECRETS):
                total += 1
    return total


# ── evidence tokens ──────────────────────────────────────────────────────────


def _git(root: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=False).stdout


def evidence_resolver(root: Path, rev: str) -> Resolver:
    """Does this token name anything? A tracked path or glob, or text a tracked
    source file holds — one rule covering a path, a test id and a span name.

    Markdown is searched for paths but not for content: the records are
    markdown, and a design that could resolve its own token would be certifying
    itself. `rev` "" is the index, what the commit being made will contain.
    """
    from fnmatch import fnmatch

    listing = ["ls-files"] if not rev else ["ls-tree", "-r", "--name-only", rev]
    tracked = [f for f in _git(root, *listing).splitlines() if f]
    known = set(tracked)
    seen: Dict[str, bool] = {}

    def resolve(token: str) -> bool:
        if token in seen:
            return seen[token]
        found = token in known or any(fnmatch(path, token) for path in tracked)
        if not found:
            grep = ["grep", "-q", "-F", "-e", token] + ([rev] if rev else ["--cached"]) + ["--", ":!*.md"]
            found = subprocess.run(["git", *grep], cwd=root, capture_output=True, check=False).returncode == 0
        seen[token] = found
        return found

    return resolve
