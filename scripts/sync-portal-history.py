"""
sync-portal-history.py — Pushes .agent-history.log entries to the Ops
Portal's history sync endpoint (Product_Archive.md P1b).

Workflow:
  1. Read .agent-history.log (JSONL), skip entries already synced
     (tracked in .agent-rfc/fixtures/sync_state.json, a different key than
     sync-ui-feedback.py's own "synced_span_ids" so the two scripts don't
     collide on the same state file).
  2. Derive a stable entryId per line (the raw log doesn't carry one) and
     POST the batch to {OPS_PORTAL_URL}/api/sync/history — the exact body
     shape portal/app/api/sync/history/route.ts expects.
  3. Record which entries were synced so a re-run doesn't resend them.

Called by:
  - cd-staging.yml / cd-production.yml (post-deploy step, optional)
  - ai-stack-check, when OPS_PORTAL_URL is configured

Requires:
  OPS_PORTAL_URL         — Ops Portal base URL. Unset = skip silently (exit 0).
  OPS_PORTAL_SYNC_TOKEN  — Bearer token for the sync endpoint. Unset = skip silently.
  .agenticframework/tenant.yaml — tenant id (same `id:` field ai-tenant-promote reads)
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from typing import Any, Optional

OPS_PORTAL_URL = os.environ.get("OPS_PORTAL_URL", "")
OPS_PORTAL_SYNC_TOKEN = os.environ.get("OPS_PORTAL_SYNC_TOKEN", "")
HISTORY_LOG_FILE = ".agent-history.log"


from _shared import _repo_root, _tenant_id  # noqa: E402
from _shared import _load_sync_state, _save_sync_state  # noqa: E402


def _load_tenant_yaml() -> dict:
    """Reads + parses .agenticframework/tenant.yaml once — shared by
    _budget_cap_usd() and _replay_webhook_config() below, which both used
    to open and yaml.safe_load() this same small file independently.
    Returns {} if the file is missing or fails to parse (callers decide
    how loudly to warn about a parse failure; this just never raises)."""
    tenant_yaml = _repo_root() / ".agenticframework" / "tenant.yaml"
    if not tenant_yaml.exists():
        return {}
    try:
        import yaml  # type: ignore

        return yaml.safe_load(tenant_yaml.read_text()) or {}
    except Exception as exc:
        print(
            f"⚠️  Failed to parse .agenticframework/tenant.yaml ({exc}) — "
            f"budget cap / replay webhook sync skipped this run."
        )
        return {}


def _budget_cap_usd(tenant_yaml_data: dict) -> Optional[float]:
    """Reads gateway.budget_cap_usd from tenant.yaml (Product_Archive.md
    P2b) — an optional, user-added section (same optionality as the
    already-documented gateway.routing_overrides), not something
    ai-tenant-init writes by default. Returns None if the gateway section
    or the key is missing — never raises, since this is a nice-to-have
    display value, not something that should break the sync."""
    try:
        # `budget.monthly_usd_cap` FIRST — that is the key LLMGateway actually
        # enforces. This function read only `gateway.budget_cap_usd`, so a
        # tenant declaring the enforced key (KYC declares $5) pushed no cap at
        # all to the portal: the dashboard showed nothing while $5 was in
        # force. Two keys for one concept, disagreeing exactly as the appendix
        # warns. The old key is still accepted so a tenant using it keeps
        # working, but it is the fallback, not the source.
        budget = tenant_yaml_data.get("budget") or {}
        value = budget.get("monthly_usd_cap")
        if value is None:
            value = (tenant_yaml_data.get("gateway") or {}).get("budget_cap_usd")
        return float(value) if value is not None else None
    except Exception:
        return None


def _replay_webhook_config(tenant_yaml_data: dict) -> Optional[dict]:
    """Reads hitl.replay_webhook_url/replay_webhook_secret from tenant.yaml
    (HITL/DLQ redesign) — same optionality and same "never raises" posture
    as _budget_cap_usd above. Both keys are required together: a URL with
    no secret can't be signed, a secret with no URL has nowhere to send.
    Deliberately per-tenant (read from THIS repo's own tenant.yaml) so a
    human-in-the-loop fix is always routed to the specific team running
    this tenant's worker, never a shared cross-tenant endpoint."""
    try:
        from urllib.parse import urlparse

        hitl = tenant_yaml_data.get("hitl") or {}
        url, secret = hitl.get("replay_webhook_url"), hitl.get("replay_webhook_secret")
        if not url and not secret:
            return None
        if bool(url) != bool(secret):
            print(
                "⚠️  tenant.yaml's hitl section has only one of replay_webhook_url/"
                "replay_webhook_secret set — both are required together (a URL with no "
                "secret can't be signed; a secret with no URL has nowhere to send). Neither will be synced."
            )
            return None
        if urlparse(url).scheme not in ("http", "https"):
            print(
                f"⚠️  hitl.replay_webhook_url={url!r} is not a valid http(s) URL — skipping sync."
            )
            return None
        return {"replayWebhookUrl": url, "replayWebhookSecret": secret}
    except Exception as exc:
        print(
            f"⚠️  Failed to parse tenant.yaml's hitl section ({exc}) — replay webhook sync skipped this run."
        )
        return None


def _entry_id(entry: dict) -> str:
    """The raw .agent-history.log line has no stable id field — derive one
    from the fields that make a line unique, so re-syncing the same line
    twice upserts (per the portal's ON CONFLICT (tenant_id, entry_id))
    instead of creating a duplicate row."""
    basis = json.dumps(
        {
            "timestamp": entry.get("timestamp"),
            "level": entry.get("level"),
            "event": entry.get("event"),
        },
        sort_keys=True,
    )
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()[:24]


def sync() -> dict:
    stats = {"synced": 0, "skipped": 0, "errors": 0}

    if not OPS_PORTAL_URL or not OPS_PORTAL_SYNC_TOKEN:
        print(
            "ℹ️  OPS_PORTAL_URL/OPS_PORTAL_SYNC_TOKEN not set — skipping portal history sync."
        )
        return stats

    tenant_id = _tenant_id()
    if not tenant_id:
        print(
            "ℹ️  No .agenticframework/tenant.yaml — skipping portal history sync (not a tenant repo)."
        )
        return stats

    log_path = _repo_root() / HISTORY_LOG_FILE
    if not log_path.exists():
        print(f"ℹ️  No {HISTORY_LOG_FILE} yet — nothing to sync.")
        return stats

    state = _load_sync_state()
    already_synced = set(state.get("synced_history_entry_ids", []))

    to_send: list[dict] = []
    for line in log_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            stats["errors"] += 1
            continue

        entry_id = _entry_id(entry)
        if entry_id in already_synced:
            stats["skipped"] += 1
            continue

        to_send.append(
            {
                "entryId": entry_id,
                "level": entry.get("level", "INFO"),
                "event": entry.get("event", ""),
                "timestamp": entry.get("timestamp", ""),
                "hitlResolved": entry.get("hitl_resolved", True),
                "raw": entry,
            }
        )

    tenant_yaml_data = _load_tenant_yaml()
    budget_cap_usd = _budget_cap_usd(tenant_yaml_data)
    replay_webhook = _replay_webhook_config(tenant_yaml_data)

    if not to_send and budget_cap_usd is None and replay_webhook is None:
        print(
            f"✅ Nothing new to sync (skipped {stats['skipped']} already-synced entries)."
        )
        return stats

    try:
        import httpx

        payload: dict[str, Any] = {"tenantId": tenant_id, "entries": to_send}
        if budget_cap_usd is not None:
            payload["budgetCapUsd"] = budget_cap_usd
        if replay_webhook is not None:
            payload.update(replay_webhook)
        resp = httpx.post(
            f"{OPS_PORTAL_URL.rstrip('/')}/api/sync/history",
            json=payload,
            headers={"Authorization": f"Bearer {OPS_PORTAL_SYNC_TOKEN}"},
            timeout=30.0,
        )
        resp.raise_for_status()
        written = resp.json().get("written", len(to_send))
        for e in to_send:
            already_synced.add(e["entryId"])
        stats["synced"] = written
        print(
            f"✅ Synced {written} entr{'y' if written == 1 else 'ies'} to {OPS_PORTAL_URL}"
        )
    except Exception as exc:
        print(f"⚠️  Portal history sync failed (non-fatal): {exc}")
        stats["errors"] += 1
        return stats  # don't persist state — these entries should be retried next run

    state["synced_history_entry_ids"] = list(already_synced)
    _save_sync_state(state)
    return stats


if __name__ == "__main__":
    result = sync()
    # Never fail the calling CD job over this — optional infra, same
    # philosophy as _ai_audit_log_event in install-ai-stack.sh.
    sys.exit(0)
