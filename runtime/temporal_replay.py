"""
runtime/temporal_replay.py — concrete Temporal `replay_handler` for
runtime/dead_letter.py's DeadLetterQueue.

Without this, `DeadLetterQueue.replay()` only flips a DB status flag —
"replayed" in name only, since the original workflow execution that
produced the entry has already terminated and there's nothing left to
resume (see Product_Archive.md's HITL/DLQ redesign). This module closes
that gap for Temporal specifically: it signals the *live, still-parked*
workflow identified by `DLQEntry.workflow_id` — which only exists because
`runtime/workflows/base_workflow.py`'s `run_with_recoverable_step` keeps
the workflow alive on failure instead of terminating it — with the
(possibly human-edited) payload, resuming exactly where it failed.

Usage (inside your Temporal worker process, which already has a Client):

    from temporal_replay import make_temporal_replay_handler
    from dead_letter import DeadLetterQueue

    dlq = DeadLetterQueue(replay_handler=make_temporal_replay_handler(temporal_client))
    dlq.replay(task_id, override_payload=edited_payload)  # -> signals the live workflow

See runtime/replay_webhook_server.py for the reference HTTP receiver a
tenant runs to let the Ops Portal trigger this remotely (the portal itself
has no Temporal client — see that module's docstring for why).
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from dead_letter import DLQEntry
    from temporalio.client import Client

logger = logging.getLogger(__name__)


def make_temporal_replay_handler(client: "Client") -> Callable[["DLQEntry"], None]:
    """Builds a synchronous `replay_handler` (the shape DeadLetterQueue.replay()
    expects) that signals the live workflow at `entry.workflow_id` with
    `entry.payload` via the `human_fix_payload` signal
    (base_workflow.BaseAgentWorkflow.human_fix_payload(gate_id, fix)).

    Wrapped in asyncio.run() internally because DeadLetterQueue itself is
    fully synchronous (plain psycopg2) — this bridges the one async call
    (the Temporal client signal) without forcing the rest of the DLQ API
    to become async. Must not be called from inside a running event loop
    (e.g. from within a Temporal workflow/activity) for that reason — call
    it from a plain sync context, such as the webhook receiver in
    runtime/replay_webhook_server.py.
    """

    def handler(entry: "DLQEntry") -> None:
        if not entry.workflow_id:
            logger.warning(
                "DLQ entry task_id=%s has no workflow_id — nothing to signal "
                "(it wasn't created by run_with_recoverable_step; falling back "
                "to recording the replay attempt without resuming anything).",
                entry.task_id,
            )
            return
        if not entry.gate_id:
            logger.warning(
                "DLQ entry task_id=%s has a workflow_id but no gate_id — "
                "human_fix_payload requires both to route the fix to the "
                "right parked gate. Skipping signal.",
                entry.task_id,
            )
            return

        async def _signal() -> None:
            handle = client.get_workflow_handle(entry.workflow_id)
            await handle.signal(
                "human_fix_payload", args=[entry.gate_id, entry.payload]
            )

        asyncio.run(_signal())
        logger.info(
            "Signaled workflow_id=%s gate_id=%s with replayed payload for task_id=%s",
            entry.workflow_id,
            entry.gate_id,
            entry.task_id,
        )

    return handler
