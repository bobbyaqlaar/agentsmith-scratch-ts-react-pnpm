#!/usr/bin/env python3
"""
scripts/send_dev_record.py FILE — send the process gate's record to the portal's
Dev workspace (portal phase 1; AgentSmith docs/process-gates.md).

    python3 scripts/process_gate.py ci --base "$BASE" --head "$HEAD" --json dev-record.json
    python3 scripts/send_dev_record.py dev-record.json

Reads AGENTSMITH_PORTAL_URL and AGENTSMITH_PORTAL_INGEST_TOKEN (the app's token,
issued under the portal's Administration › Apps). Standard library only, so any
CI job that can run the gate can run this.

How it ends, and why:
- not configured → a notice, exit 0: a repository without a portal is not a
  failed build;
- the portal refuses the record (4xx) → its reason as an error, exit 1: the
  record or the token is wrong, and nobody learns it from a green tick;
- the portal is unreachable or failing (5xx) → a warning, exit 0: the gate's
  verdict stands on its own, and the next push sends the record again;
- a plain-http portal other than localhost → refused before the token is sent;
- a portal address that redirects → refused before a second request exists,
  because urllib copies the Authorization header onto the redirected request:
  the token would go wherever the redirect points.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from typing import Callable, Optional
from urllib.parse import urlparse

# The portal's per-request limit (portal/lib/devIngest.ts DEV_LIMITS.commits).
CHUNK = 500


class _Redirected(Exception):
    pass


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Refuses instead of following: urllib's redirect_request copies the
    Authorization header onto the next request, so following would hand the
    ingest token to wherever the redirect points. The same opener
    runtime/intake.py uses (.agent-rfc/designs/send-dev-record-redirects.md)."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[override]
        raise _Redirected(f"{code} to {newurl}")


_OPENER = urllib.request.build_opener(_NoRedirect)


def _say(level: str, text: str) -> None:
    """A GitHub annotation inside Actions; a plain line anywhere else."""
    prefix = f"::{level}::" if os.environ.get("GITHUB_ACTIONS") == "true" else f"{level}: "
    print(prefix + text)


def _parts(document: dict) -> list[dict]:
    """The record in parts of at most CHUNK commits, oldest first. The designs
    ride with the last part only: an earlier part's head is not the range's
    head, and the portal keeps whichever designs came with the newest head."""
    commits = document.get("commits") or []
    if len(commits) <= CHUNK:
        return [document]
    parts = []
    for start in range(0, len(commits), CHUNK):
        part = {k: v for k, v in document.items() if k not in ("commits", "designs")}
        part["commits"] = commits[start:start + CHUNK]
        part["head"] = part["commits"][-1].get("commit", document.get("head"))
        if start + CHUNK >= len(commits) and "designs" in document:
            part["designs"] = document["designs"]
            part["head"] = document.get("head")
        parts.append(part)
    return parts


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__.strip().splitlines()[0])
        return 2
    try:
        document: Optional[dict] = json.loads(open(argv[1], encoding="utf-8").read())
        unread = ""
    except (OSError, json.JSONDecodeError) as exc:
        document, unread = None, str(exc)
    return send(document, _say, unread)


def send(document: Optional[dict], say: Callable[[str, str], None], unread: str = "") -> int:
    """Send `document` and report through `say(level, text)`; 0 or 1 as the
    endings in this module's docstring, in the same order — configuration first.
    `None` with `unread` is a record that could not be read. Imported by the
    gate's contract-2 `ci` answer (.agent-rfc/designs/gate-contract-ci.md), so a
    tenant's workflow runs no record step of its own."""
    url = os.environ.get("AGENTSMITH_PORTAL_URL", "").strip()
    token = os.environ.get("AGENTSMITH_PORTAL_INGEST_TOKEN", "").strip()
    if not url or not token:
        say("notice", "the gate's record was not sent: set AGENTSMITH_PORTAL_URL and "
                       "AGENTSMITH_PORTAL_INGEST_TOKEN to show this repository in the portal's Dev workspace")
        return 0
    parsed = urlparse(url)
    if parsed.scheme != "https" and not (parsed.scheme == "http" and parsed.hostname in ("localhost", "127.0.0.1")):
        say("error", f"AGENTSMITH_PORTAL_URL must be https (or http to localhost) — not sending the token to {url}")
        return 1
    if document is None:
        say("warning", f"no gate record to send ({unread}) — did the gate step run?")
        return 0

    endpoint = url.rstrip("/") + "/api/dev/ingest"
    stored = 0
    for part in _parts(document):
        request = urllib.request.Request(
            endpoint, data=json.dumps(part).encode("utf-8"), method="POST",
            headers={"authorization": f"Bearer {token}", "content-type": "application/json"},
        )
        try:
            with _OPENER.open(request, timeout=30) as response:
                stored += json.loads(response.read() or b"{}").get("stored", 0)
        except _Redirected as exc:
            # The operator's fix, not an outage: a warning here would let a green
            # build hide that every record since was going nowhere.
            say("error", f"the portal redirected ({exc}) — not following it, because the token would go "
                          "with it. Set AGENTSMITH_PORTAL_URL to the portal's final address.")
            return 1
        except urllib.error.HTTPError as exc:
            reason = exc.read().decode("utf-8", "replace")[:500]
            try:
                reason = json.loads(reason).get("error", reason)
            except (ValueError, AttributeError):  # fail-open: the body was not the
                # JSON we hoped for; keep the raw text already in `reason`.
                pass
            if 400 <= exc.code < 500:
                say("error", f"the portal refused the gate's record ({exc.code}): {reason}")
                return 1
            say("warning", f"the portal could not store the gate's record ({exc.code}): {reason}")
            return 0
        except (urllib.error.URLError, OSError) as exc:
            say("warning", f"the portal was not reachable at {url} ({exc}) — the record was not sent")
            return 0
    say("notice", f"the gate's record reached the portal: {stored} commit(s) stored")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
