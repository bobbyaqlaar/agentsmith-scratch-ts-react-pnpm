"""
runtime/temporal_client.py — one way to reach the Temporal server.

Seven call sites connected to Temporal, and they did not agree:

  * `runtime/worker.py` read `os.environ["TEMPORAL_ADDRESS"]` — a KeyError,
    not a message, when unset — while others defaulted to localhost.
  * Only `examples/oil-price-agent/*` honoured `TEMPORAL_TLS` at all. The
    framework's own shipped worker and KYC Sentinel's ignored it, so a
    deployment against a TLS-terminating Temporal Cloud endpoint connected
    without TLS and the setting appeared to work because nothing complained.
  * Those three compared it to the literal `"true"`, while OPERATIONS.md
    documents `TEMPORAL_TLS="1"`. Following the documentation therefore
    produced `use_tls=False` — a documented security switch that silently did
    nothing, everywhere it was read.
  * Only `replay_webhook_server` bounded the connect. Without it an
    unreachable server hangs for the OS TCP timeout, often 2+ minutes, which
    reads as "the app is stuck" rather than "Temporal is down".

Each of those is defensible alone; together they meant the same environment
produced four different behaviours depending on which entry point ran.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Optional

# Accepted truthy spellings live in runtime/config.as_bool. This module had its
# own `_TRUTHY = {"1", "true", "yes", "on"}` and config.py grew an identical set
# under a different name — two catalogs of the same fact, which is how one of
# them ends up accepting a spelling the other rejects. The reasoning is worth
# keeping: "1" is what OPERATIONS.md documents, "true" is what the example
# scripts checked for, and a flag that turns a security control ON must not
# depend on which spelling the reader happened to copy.

DEFAULT_ADDRESS = "localhost:7233"
DEFAULT_CONNECT_TIMEOUT = 10.0


def tls_enabled(env: Optional[dict] = None) -> bool:
    """Whether TEMPORAL_TLS asks for TLS, accepting every documented spelling."""
    from runtime.config import as_bool

    return as_bool((env or os.environ).get("TEMPORAL_TLS", ""))


def temporal_address(env: Optional[dict] = None) -> str:
    """TEMPORAL_ADDRESS, or the local default.

    Deliberately not `os.environ[...]`: a missing address should produce a
    connection error naming the host it tried, not a KeyError from inside a
    worker's startup path.
    """
    return (env or os.environ).get("TEMPORAL_ADDRESS", "").strip() or DEFAULT_ADDRESS


async def connect(
    address: Optional[str] = None,
    *,
    timeout: float = DEFAULT_CONNECT_TIMEOUT,
    **kwargs: Any,
):
    """Connect to Temporal with the address, TLS setting and timeout resolved
    consistently. Extra kwargs pass through to `Client.connect`.

    The timeout is bounded rather than indefinite for the reason
    replay_webhook_server documented and the other six call sites did not
    inherit: a server that is down should fail in seconds with its address in
    the message.
    """
    from temporalio.client import Client

    target = address or temporal_address()
    kwargs.setdefault("tls", tls_enabled())
    try:
        return await asyncio.wait_for(Client.connect(target, **kwargs), timeout=timeout)
    except asyncio.TimeoutError as exc:
        raise TimeoutError(
            f"Temporal did not answer at {target} within {timeout:g}s "
            f"(tls={kwargs['tls']}). Check TEMPORAL_ADDRESS and that the "
            f"server is reachable."
        ) from exc
