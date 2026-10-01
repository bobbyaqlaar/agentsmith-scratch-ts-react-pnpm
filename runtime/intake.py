"""
runtime/intake.py — pull a tenant intake from the portal, for
`agentsmith tenant init --from <id>` (.agent-rfc/designs/portal-intake-pull.md;
wire format contract/intake/v1/).

The portal never writes into a repository. It holds an author's answers and
hands out one token for them; this module fetches the record onto the author's
machine, `tenant init` scaffolds from it, and only then is it consumed — so the
first commit is still made locally and still vouched by the scaffold manifest.

Standard library only: `runtime/` is vendored into tenants. It follows
scripts/send_dev_record.py's discipline, and is stricter in one place:

- `https`, or `http` only to localhost — checked before the token is sent.
- NO redirects. `urllib` carries the Authorization header across one, so a
  redirecting address would hand the token to wherever it points.
- A bounded timeout and a response-size cap.
- Every field is re-validated here against this CLI's own rules. What the
  portal accepted is not trusted for that reason: the portal's checks are for
  the author's convenience, these are the control.

Exit codes, so a script can tell the two apart: 2 when the author must change
something (not configured, refused, an invalid record — each named), 4 when the
portal is unreachable or failing and the same command is the retry.
"""

from __future__ import annotations

import getpass
import json
import os
import re
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

URL_VAR = "AGENTSMITH_PORTAL_URL"
TOKEN_VAR = "AGENTSMITH_INTAKE_TOKEN"

CHANGE_SOMETHING = 2
RETRY = 4

# contract/intake/v1/record.schema.json, mirrored because contract/ is not
# vendored into tenants; runtime/test/test_intake.py pins each to the schema.
SCHEMA_VERSION = 1
APP_ID = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")
INTAKE_ID = re.compile(r"^[0-9]{1,19}$")
ARCHITECTURE = re.compile(r"^[a-z][a-z-]{0,39}$")
LIMITS = {"objective": 4000, "criteria": 20, "files": 50, "item": 500}
FIELDS = frozenset({"schema_version", "intake_id", "tenant_id", "stack", "isolation", "architecture",
                    "agentic", "ides", "rfc", "expires_at"})
RFC_FIELDS = frozenset({"objective", "acceptance_criteria", "files_to_modify"})

TIMEOUT_SECONDS = 30
# The largest valid record is about 40 KB of text; this leaves room for JSON
# escaping and refuses anything that is plainly not an intake.
MAX_BYTES = 256 * 1024

# Newline and tab are the only control characters a paragraph needs.
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


class IntakeError(Exception):
    """Why the intake could not be used, and the exit code that says whether to
    change something or retry."""

    def __init__(self, message: str, exit_code: int = CHANGE_SOMETHING):
        super().__init__(message)
        self.exit_code = exit_code


def safe_portal_url(url: str) -> bool:
    """https anywhere; http only to this machine. The same rule
    scripts/send_dev_record.py applies before it sends a token — pinned to it by
    runtime/test/test_intake.py, since the two ship to different places."""
    parsed = urlparse(url)
    return parsed.scheme == "https" or (parsed.scheme == "http" and parsed.hostname in ("localhost", "127.0.0.1"))


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[override]
        raise IntakeError(
            f"the portal redirected ({code}) to {newurl} — not following it, because the token would go "
            f"with it. Set {URL_VAR} to the portal's final address."
        )


_OPENER = urllib.request.build_opener(_NoRedirect)


def _clean(value: Any, limit: int, field: str) -> str:
    if not isinstance(value, str):
        raise IntakeError(f"invalid intake: {field} is not text")
    cleaned = _CONTROL.sub("", value).strip()
    if not 1 <= len(cleaned) <= limit:
        raise IntakeError(f"invalid intake: {field} must be 1–{limit} characters")
    return cleaned


def _lines(value: Any, low: int, high: int, field: str) -> list[str]:
    if not isinstance(value, list) or not low <= len(value) <= high:
        raise IntakeError(f"invalid intake: {field} must be a list of {low}–{high} items")
    # One item, one line: a newline inside a list item would end the item and
    # start a heading or a paragraph of its own in the RFC.
    return [" ".join(_clean(v, LIMITS["item"], field).split()) for v in value]


def validate_record(record: Any, intake_id: str) -> dict[str, Any]:
    """The record, re-checked field by field against this CLI's own rules, or an
    IntakeError naming the first field that fails."""
    from runtime.cli import ISOLATIONS, STACKS, validate_tenant_id

    if not isinstance(record, dict):
        raise IntakeError("invalid intake: the portal's answer is not a JSON object")
    if record.get("schema_version") != SCHEMA_VERSION:
        raise IntakeError(
            f"this agentsmith reads intake records of version {SCHEMA_VERSION}; the portal sent "
            f"{record.get('schema_version')!r}. Upgrade agentsmith, or ask for the intake again."
        )
    unknown = set(record) - FIELDS
    missing = FIELDS - set(record)
    if unknown or missing:
        raise IntakeError(f"invalid intake: not a v1 record (unexpected {sorted(unknown)}, missing {sorted(missing)})")
    if record["intake_id"] != intake_id:
        raise IntakeError(f"invalid intake: asked for intake {intake_id}, "
                          f"the portal answered with {record['intake_id']!r}")

    tenant_id = record["tenant_id"]
    try:
        validate_tenant_id(tenant_id)
    except ValueError as exc:
        raise IntakeError(f"invalid intake: {exc}") from exc
    if not APP_ID.match(tenant_id):
        raise IntakeError(f"invalid intake: tenant id {tenant_id!r} is not one the portal can register as an app")
    if record["stack"] not in STACKS:
        raise IntakeError(f"invalid intake: stack {record['stack']!r} is not one of {', '.join(STACKS)}")
    if record["isolation"] not in ISOLATIONS:
        raise IntakeError(f"invalid intake: isolation {record['isolation']!r} is not one of {', '.join(ISOLATIONS)}")
    architecture = record["architecture"]
    if architecture is not None and not (isinstance(architecture, str) and ARCHITECTURE.match(architecture)):
        raise IntakeError("invalid intake: architecture is not a style name")
    if not isinstance(record["agentic"], bool):
        raise IntakeError("invalid intake: agentic is not true or false")
    ides = record["ides"]
    if not isinstance(ides, list) or not all(isinstance(i, str) for i in ides) or len(set(ides)) != len(ides):
        raise IntakeError("invalid intake: ides is not a list of distinct names")

    rfc = record["rfc"]
    if not isinstance(rfc, dict) or set(rfc) != RFC_FIELDS:
        raise IntakeError(f"invalid intake: rfc must have exactly {', '.join(sorted(RFC_FIELDS))}")
    return {
        "intake_id": intake_id,
        "tenant_id": tenant_id,
        "stack": record["stack"],
        "isolation": record["isolation"],
        "architecture": architecture,
        "agentic": record["agentic"],
        # Which IDE names are real is init_tenant's --ide check, one place for both paths.
        "ides": ides,
        "rfc": {
            "objective": _clean(rfc["objective"], LIMITS["objective"], "rfc.objective"),
            "acceptance_criteria": _lines(rfc["acceptance_criteria"], 1, LIMITS["criteria"], "rfc.acceptance_criteria"),
            "files_to_modify": _lines(rfc["files_to_modify"], 0, LIMITS["files"], "rfc.files_to_modify"),
        },
    }


def _call(method: str, url: str, token: str) -> dict[str, Any]:
    request = urllib.request.Request(
        url, method=method, data=b"{}" if method == "POST" else None,
        headers={"authorization": f"Bearer {token}", "accept": "application/json",
                 **({"content-type": "application/json"} if method == "POST" else {})},
    )
    try:
        with _OPENER.open(request, timeout=TIMEOUT_SECONDS) as response:
            body = response.read(MAX_BYTES + 1)
    except urllib.error.HTTPError as exc:
        reason = exc.read(4096).decode("utf-8", "replace")
        try:
            reason = json.loads(reason).get("error", reason)
        except (ValueError, AttributeError):  # fail-open: keep the raw text already in `reason`
            pass
        if exc.code >= 500:
            raise IntakeError(f"the portal failed ({exc.code}): {reason} — run the same command again", RETRY) from exc
        hint = {401: f" — check {TOKEN_VAR}: it is the token shown once when the intake was created"}.get(exc.code, "")
        raise IntakeError(f"the portal refused ({exc.code}): {reason}{hint}") from exc
    except (urllib.error.URLError, OSError) as exc:
        raise IntakeError(f"the portal was not reachable at {url} ({exc}) — run the same command again", RETRY) from exc
    if len(body) > MAX_BYTES:
        raise IntakeError(f"invalid intake: the portal's answer is larger than {MAX_BYTES} bytes")
    try:
        return json.loads(body)
    except ValueError as exc:
        raise IntakeError("invalid intake: the portal's answer is not JSON") from exc


@dataclass
class Intake:
    """A validated record, and what is needed to consume it once it has landed."""

    record: dict[str, Any]
    _base: str
    # Out of the repr: a dataclass prints every field, so a traceback or a debug
    # print of this object would otherwise print the token.
    _token: str = field(repr=False)

    def consume(self) -> str:
        """Marks the intake used. Never raises: by now the scaffold is written,
        and a failure here must not read as a failed scaffold."""
        try:
            _call("POST", f"{self._base}/api/dev/scaffold/{self.record['intake_id']}/consume", self._token)
        except IntakeError as exc:
            # Indented: a column-0 ⚠️ fails a tenant build (.github/scratch-tenants/build.sh).
            return (f"  ⚠️  The scaffold is complete, but intake {self.record['intake_id']} "
                    f"was not marked used ({exc}). "
                    f"It expires by itself; nothing needs undoing.")
        return f"Intake {self.record['intake_id']} marked used in the portal."


def _token() -> str:
    # Literal names, not the constants: scripts/test/test_env_var_documentation.py
    # finds an environment read by its literal, and a constant would hide it.
    token = os.environ.get("AGENTSMITH_INTAKE_TOKEN", "").strip()
    if token:
        return token
    if sys.stdin.isatty():
        # Asked, not exported: an `export` lands in shell history as surely as an argument.
        token = getpass.getpass("Intake token (shown once when the intake was created): ").strip()
        if token:
            return token
    raise IntakeError(f"not configured: set {TOKEN_VAR} to the intake's token, "
                      "or run this at a terminal to be asked for it")


def fetch(intake_id: str) -> Intake:
    """The intake's validated record, or an IntakeError."""
    if not INTAKE_ID.match(intake_id):
        # Before any URL is built: `--from ../admin` is a bad argument, not a path.
        raise IntakeError(f"--from {intake_id!r}: an intake id is digits, as the portal shows it")
    base = os.environ.get("AGENTSMITH_PORTAL_URL", "").strip().rstrip("/")
    if not base:
        raise IntakeError(f"not configured: set {URL_VAR} to the portal's address")
    if not safe_portal_url(base):
        raise IntakeError(f"{URL_VAR} must be https (or http to localhost) — not sending the token to {base}")
    token = _token()
    record = validate_record(_call("GET", f"{base}/api/dev/scaffold/{intake_id}", token), intake_id)
    return Intake(record, base, token)

