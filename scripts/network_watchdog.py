"""
network_watchdog.py — Network connectivity probe with automatic offline fallback.

Pings 1.1.1.1:53 (Cloudflare DNS). On failure, switches the active LLM
endpoint to the local Ollama instance and notifies the agent.

Used by cost_router.py and local_agent_stack.py to decide which model
tier to use before every LLM call.
"""

from __future__ import annotations

import os
import socket
import threading
import time
from typing import Optional

# ── Config ────────────────────────────────────────────────────────────────────

PROBE_HOST = "1.1.1.1"
PROBE_PORT = 53
PROBE_TIMEOUT = 2.0  # seconds

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_API_KEY = os.environ.get("OLLAMA_API_KEY", "ollama")
CLOUD_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
CLOUD_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# How often to re-probe when in offline mode (seconds)
RECHECK_INTERVAL = 30

# ── State ─────────────────────────────────────────────────────────────────────

_lock = threading.Lock()
_online: Optional[bool] = None  # None = not yet probed
_last_check: float = 0.0


# ── Core probe ────────────────────────────────────────────────────────────────


def is_online(force: bool = False) -> bool:
    """
    Return True if the machine has internet connectivity.
    Result is cached for RECHECK_INTERVAL seconds.
    Pass force=True to bypass the cache and re-probe immediately.
    """
    global _online, _last_check
    with _lock:
        now = time.time()
        if not force and _online is not None and (now - _last_check) < RECHECK_INTERVAL:
            return _online

        try:
            sock = socket.create_connection(
                (PROBE_HOST, PROBE_PORT), timeout=PROBE_TIMEOUT
            )
            sock.close()
            was_online = _online
            _online = True
            _last_check = now
            if was_online is False:
                # Recovered from offline — notify
                _notify_recovery()
            return True
        except OSError:
            was_online = _online
            _online = False
            _last_check = now
            if was_online is not False:
                # Just went offline — notify
                _notify_offline()
            return False


def get_llm_endpoint() -> dict:
    """
    Return {"base_url": ..., "api_key": ...} for the currently active
    LLM backend — cloud when online, Ollama when offline.
    """
    if is_online():
        return {"base_url": CLOUD_BASE_URL, "api_key": CLOUD_API_KEY}
    return {"base_url": OLLAMA_BASE_URL, "api_key": OLLAMA_API_KEY}


# `require_online()` and the background keepalive thread were removed here
# (2026-08-26). Three public functions, no caller anywhere — not in this repo,
# not in a workflow, not in a doc. `start_background_watcher`'s own docstring
# said "call once at agent startup to enable proactive offline detection", and
# nothing ever did, so the detection it described had never run;
# `stop_background_watcher` was `pass`, kept "for API symmetry" with it.
# `is_online()` below is called on demand and caches, which is what every real
# caller actually wanted.


# ── Notifications ─────────────────────────────────────────────────────────────


def _notify(title: str, body: str, urgency: str, fallback: str) -> None:
    """Desktop notification, degrading to stderr.

    The `except Exception` is the point: this runs on a network transition, so
    the notifier itself may be exactly what is unavailable. A watchdog that
    raised while reporting an outage would turn a degraded network into a dead
    watchdog, which is when it is most needed.
    """
    try:
        from notifier import send_notification

        send_notification(title, body, urgency=urgency)
    except Exception:  # fail-open: reporting must not depend on the thing being reported
        import sys

        print(f"[network_watchdog] {fallback}", file=sys.stderr)


def _notify_offline() -> None:
    _notify(
        "📡 Network Offline",
        "AgentSmith switched to LOCAL mode (Ollama).",
        "normal",
        "OFFLINE — falling back to Ollama",
    )


def _notify_recovery() -> None:
    _notify(
        "✅ Network Restored",
        "AgentSmith is back online — cloud models available.",
        "low",
        "ONLINE — cloud models available",
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    status = is_online(force=True)
    endpoint = get_llm_endpoint()
    print(
        json.dumps(
            {
                "online": status,
                "active_endpoint": endpoint["base_url"],
            },
            indent=2,
        )
    )
