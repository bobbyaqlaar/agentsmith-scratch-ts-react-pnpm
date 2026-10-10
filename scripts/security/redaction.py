"""
scripts/security/redaction.py — what a repository's telemetry carries on the
wire, checked with planted probes (contract/security/v1/protocol.md, `redaction`).

The repository's declared emitter runs once with its OTLP export pointed at a
receiver on 127.0.0.1 — its own destinations and collector credential removed —
`ENVIRONMENT` set to the profile asked about, and the probes in
`SECURITY_REDACTION_PROBES`. It promises one span carrying
`security.redaction.probe: true` with the probes in `input.value`, through
whatever redaction it applies. A probe that arrives verbatim, anywhere in what
was exported, is a leak — named by its kind, never its text.

This judges what reaches the wire, not what a helper returns: the check it
replaces for a tenant scrubbed three strings with AgentSmith's own redactor and
said nothing about what the tenant exports (.agent-rfc/designs/security-contract.md).
"""

from __future__ import annotations

import json
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

PROBES_ENV = "SECURITY_REDACTION_PROBES"
INSTALL = Path(__file__).resolve().parents[2]  # the provider's own checkout
PROBE_ATTRIBUTE = "security.redaction.probe"


@dataclass(frozen=True)
class Probe:
    kind: str
    text: str  # what the emitter is handed
    core: str  # what must not arrive


# contract/security/v1/probes.json, held equal by a test. None is a credential:
# each matches what a redactor scrubs, and none matches the shapes the P12 check
# looks for in a tracked file (the API-key probe is shorter than a real key).
PROBES = (
    Probe("api key", "sk-redactionprobe0000000001", "sk-redactionprobe0000000001"),
    Probe("bearer token", "Authorization: Bearer redaction-probe-token-0001", "redaction-probe-token-0001"),
    Probe("email address", "redaction.probe@example.com", "redaction.probe@example.com"),
    Probe("card number", "4111 1111 1111 1111", "4111 1111 1111 1111"),
)


@dataclass(frozen=True)
class Redaction:
    verdict: str  # pass | fail | not_gradable | not_applicable
    reason: str
    leaked: tuple[str, ...] = ()


def declared_emitter(root: Path) -> Optional[str]:
    """`providers.security.emitter` in the repository's providers.json, or None."""
    try:
        declared = json.loads((root / ".agenticframework" / "providers.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    port = (declared.get("providers") or {}).get("security") if isinstance(declared, dict) else None
    emitter = port.get("emitter") if isinstance(port, dict) else None
    return (str(emitter).strip() or None) if emitter is not None else None


def check(root: Path, environment: str, emitter: Optional[str], timeout: int = 300) -> Redaction:
    """Run `emitter` for `environment`'s profile and judge what arrived."""
    from runtime.telemetry_contract import OTLP_DESTINATIONS, LoopbackCollector, flatten

    if emitter is None:
        return Redaction("not_gradable", "the repository declares no emitter — name the command that exports one "
                                         'run of its telemetry, or `"emitter": "none"`')
    if emitter == "none":
        return Redaction("not_applicable", "the repository declares it emits no telemetry")
    from security.runners._shared import repository_env

    with LoopbackCollector() as collector:
        # The emitter is the repository's code: its OTLP destinations removed,
        # and the provider's own source off its path.
        env = {k: v for k, v in repository_env(INSTALL).items() if k not in OTLP_DESTINATIONS}
        env.update({"OTEL_EXPORTER_OTLP_ENDPOINT": collector.endpoint, "OTEL_EXPORTER_OTLP_PROTOCOL": "http/protobuf",
                    "OTEL_METRIC_EXPORT_INTERVAL": "1000", "ENVIRONMENT": environment,
                    PROBES_ENV: json.dumps([probe.text for probe in PROBES])})
        ran = ""  # why the emitter did not finish cleanly, judged after the wire
        try:
            done = subprocess.run(shlex.split(emitter), cwd=root, env=env, capture_output=True, text=True,
                                  check=False, timeout=timeout)
            if done.returncode != 0:
                tail = done.stderr.strip().splitlines()[-1:] or [""]
                ran = f"the emitter exited {done.returncode}: {tail[0][-300:]}"
        except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
            ran = f"the emitter did not run to the end ({exc})"
        exports, problems = list(collector.exports), list(collector.problems)
    # What reached the wire first: a leak is the stronger fact, whatever the
    # emitter did after it sent one — exit badly, hang, or crash.
    wire = json.dumps(exports, ensure_ascii=False)
    leaked = tuple(probe.kind for probe in PROBES if probe.core in wire)
    if leaked:
        return Redaction("fail", f"{', '.join(leaked)} reached the wire unredacted under the {environment} profile",
                         leaked)
    if ran:
        return Redaction("not_gradable", ran)
    if not exports:
        why = f" ({problems[0]})" if problems else ""
        return Redaction("not_gradable", f"nothing arrived from the emitter{why}")
    planted = any(span.attributes.get(PROBE_ATTRIBUTE) == ("bool", True)
                  for emitted in flatten(exports) for span in emitted.spans)
    if not planted:
        return Redaction("not_gradable", f"no span carried `{PROBE_ATTRIBUTE}` — the emitter did not plant the "
                                         f"probes it is handed in {PROBES_ENV}")
    return Redaction("pass", f"no probe reached the wire under the {environment} profile")
