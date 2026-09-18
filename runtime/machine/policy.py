"""
runtime/machine/policy.py — the enterprise org policy, break-glass tokens, the
audit log, and the one decision about whether the hooks may be bypassed.

The policy file (`~/.agent-framework/agenticframework-org.yaml`, deployed by
MDM — enterprise/README.md) declares `hooks.bypass_policy`. It used to be read
only by `ai-stack-off`, while every hook exited on `DISABLE_AI_STACK=true` from
any environment — so `DISABLE_AI_STACK=true git commit` skipped a policy of
`disabled` without asking it. `bypass_decision` is now called by the hooks
themselves (`agentsmith hooks bypass-check`) and by `agentsmith mode off`.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

from runtime.machine.state import framework_home

ORG_POLICY_FILE = "agenticframework-org.yaml"

# What portal/app/api/audit/append accepts (portal/lib/auditSignature.ts —
# test_machine_policy.py parses it). Anything else is rejected with a 400.
AUDIT_EVENT_TYPES = ("hook_bypass", "hitl_promotion", "config_change", "tenant_created")


def org_policy_path() -> Path:
    return framework_home() / ORG_POLICY_FILE


class PolicyUnreadable(RuntimeError):
    """The policy file exists but cannot be read as a policy."""


def org_policy_hooks(path: Optional[Path] = None) -> Optional[dict[str, Any]]:
    """The policy's `hooks:` mapping; None when there is no policy file.

    A file that exists but does not parse raises PolicyUnreadable rather than
    reading as "no policy": the caller is deciding whether to let a bypass
    through, and a corrupt policy must not be the way to get one.
    """
    path = path or org_policy_path()
    if not path.exists():
        return None
    try:
        import yaml  # a dependency of the runtime package; lazy to keep hooks fast

        doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        raise PolicyUnreadable(f"{path}: {exc}") from exc
    hooks = doc.get("hooks") if isinstance(doc, dict) else None
    if hooks is not None and not isinstance(hooks, dict):
        raise PolicyUnreadable(f"{path}: `hooks` is not a mapping")
    return hooks or {}


def approvers_text(hooks: Mapping[str, Any]) -> str:
    approvers = hooks.get("break_glass_approvers") or []
    if isinstance(approvers, str):
        return approvers
    return ", ".join(str(a) for a in approvers) or "it-sec@example.com"


def validate_break_glass_token(token: str, key: str, now: Optional[float] = None) -> tuple[bool, str]:
    """(valid, reason). Token format: `<actor>:<expires_epoch>.<hex_hmac_sha256>`.

    Signed with BREAK_GLASS_HMAC_KEY, a secret IT distributes out of band —
    never the per-use token itself. The same HMAC-over-shared-secret pattern as
    the portal's widget tokens and audit signatures.
    """
    if not key:
        return False, (
            "break-glass tokens cannot be validated on this machine (BREAK_GLASS_HMAC_KEY not "
            "configured). Contact IT to provision it."
        )
    payload, dot, signature = token.rpartition(".")
    if not dot or not payload or not signature:
        return False, "malformed break-glass token (expected <actor>:<expires_epoch>.<signature>)"
    expected = hmac.new(key.encode(), payload.encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, signature):
        return False, "break-glass token signature is invalid — this is not a token IT issued"
    expiry = payload.rpartition(":")[2]
    if not expiry.isdigit() or (now if now is not None else time.time()) > int(expiry):
        return False, "break-glass token has expired — request a new one from IT"
    return True, "valid"


def audit_log_event(
    event_type: str,
    actor_id: str,
    tenant_id: str,
    details: Mapping[str, Any],
    env: Optional[Mapping[str, str]] = None,
) -> None:
    """Best-effort audit write (docs/DESIGN.md › Enterprise Install and Compliance Pack). Never raises, never blocks.

    SPECS promises bypass events reach the immutable audit log unconditionally,
    so when the portal is unconfigured or the write fails the event goes to
    `~/.agent-framework/local-audit-fallback.log` instead of vanishing.
    """
    if event_type not in AUDIT_EVENT_TYPES:
        raise ValueError(f"audit event type must be one of {AUDIT_EVENT_TYPES}, got {event_type!r}")
    environ = os.environ if env is None else env
    url = (environ.get("OPS_PORTAL_URL") or "").rstrip("/")
    token = environ.get("AUDIT_LOG_WRITE_TOKEN") or ""
    body = {"eventType": event_type, "actorId": actor_id, "tenantId": tenant_id, "details": dict(details)}

    reason = "ops_portal_not_configured"
    if url and token:
        request = urllib.request.Request(
            f"{url}/api/audit/append",
            data=json.dumps(body).encode(),
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                if 200 <= response.status < 300:
                    return
        except Exception:  # fail-open: the event is kept locally below instead
            pass
        reason = "ops_portal_write_failed"

    record = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        **body,
        "reason": reason,
    }
    try:
        log = framework_home() / "local-audit-fallback.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
    except OSError:  # fail-open: auditing never blocks the command it records
        pass


@dataclass(frozen=True)
class BypassDecision:
    allowed: bool
    message: str


def bypass_decision(
    env: Optional[Mapping[str, str]] = None,
    now: Optional[float] = None,
    policy_path: Optional[Path] = None,
) -> BypassDecision:
    """May the hooks be bypassed right now? Records the attempt when a policy applies.

    No policy file: yes — a developer machine, today's behaviour. `disabled`:
    no. `break-glass`: only with a valid, unexpired AI_BREAK_GLASS_TOKEN. A
    policy file that cannot be read: no.
    """
    environ = os.environ if env is None else env
    actor = environ.get("AGENT_OWNER_ID") or "unknown"
    try:
        hooks = org_policy_hooks(policy_path)
    except PolicyUnreadable as exc:
        return BypassDecision(False, f"the org policy could not be read ({exc}) — hooks stay on")
    if hooks is None:
        return BypassDecision(True, "no org policy on this machine")

    policy = str(hooks.get("bypass_policy") or "")
    if policy == "disabled":
        audit_log_event("hook_bypass", actor, "", {"result": "denied", "policy": "disabled"}, environ)
        return BypassDecision(
            False,
            f"enterprise policy: hook bypass is DISABLED. Contact IT for a break-glass procedure: "
            f"{approvers_text(hooks)}",
        )
    if policy == "break-glass":
        token = environ.get("AI_BREAK_GLASS_TOKEN") or ""
        if not token:
            audit_log_event(
                "hook_bypass", actor, "", {"result": "denied", "policy": "break-glass", "reason": "no_token"}, environ
            )
            return BypassDecision(
                False,
                f"enterprise policy: hook bypass requires a break-glass token from {approvers_text(hooks)}; "
                f"set AI_BREAK_GLASS_TOKEN=<token> for the command",
            )
        valid, reason = validate_break_glass_token(token, environ.get("BREAK_GLASS_HMAC_KEY") or "", now)
        if not valid:
            audit_log_event(
                "hook_bypass", actor, "", {"result": "denied", "policy": "break-glass", "reason": "invalid_token"},
                environ,
            )
            return BypassDecision(False, reason)
        audit_log_event("hook_bypass", actor, "", {"result": "approved", "policy": "break-glass"}, environ)
        return BypassDecision(True, "break-glass bypass used — logged to the enterprise audit log")
    if policy:
        # A value this code does not know — `disable`, `Disabled`, a newer
        # policy name — was read as "no restriction". An org that wrote a
        # policy meant to restrict something; refuse until it says what.
        return BypassDecision(
            False,
            f"the org policy's bypass_policy {policy!r} is not one of disabled, break-glass — refusing the bypass",
        )
    return BypassDecision(True, "org policy sets no bypass restriction (bypass_policy unset)")
