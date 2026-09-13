"""
runtime/pg_pool.py — process-wide Postgres connection pool shared by the
runtime's Postgres-backed stores (`llm_gateway._PostgresBudgetBackend`,
`idempotency._PostgresBackend`, `dead_letter.DeadLetterQueue`,
`vector_store.PgVectorStore`).

Keep that list complete: it named three stores while there were four, and the
missing one — PgVectorStore — was still opening a raw connection per operation
long after this module existed. An enumeration that is allowed to go stale
reads as "these are the Postgres stores", which is how the omission survived.

Why this exists (ReviewFindings-2026-07-18 C1): each store used to open a
fresh `psycopg2.connect()` per operation — budget try_reserve + add_spend
alone put 2–3 TCP + auth round-trips inside EVERY gateway LLM call. The
SQL each store runs was already atomic; only connection reuse was missing.

Design:
  - One `ThreadedConnectionPool` per DSN, created lazily on first use
    (psycopg2 stays an optional dependency — nothing imports it at module
    import time, same convention as the stores' own lazy `_connect`).
  - `connect(dsn)` returns a connection *proxy* whose `.close()` returns
    the underlying connection to the pool instead of closing it. This is
    deliberate: every existing call site follows the
    `conn = self._connect() ... finally: conn.close()` pattern, so the
    stores keep their exact code shape — only their `_connect` changed.
    psycopg2's own `putconn` rolls back a connection returned mid-
    transaction and closes one whose status is unknown (broken), so a
    call site that raised before commit cannot poison the next borrower.
  - Each borrowed connection is pinged (`SELECT 1`) before hand-off;
    dead ones (e.g. Postgres restarted since last use) are discarded and
    replaced. This keeps the old per-call-connect behavior's resilience —
    a DB restart never fails a store operation that a fresh connection
    would have served. The ping is a warm-socket round-trip, still far
    cheaper than the TCP + TLS + auth handshake it replaces.
  - Pool exhausted (burst > PG_POOL_MAX concurrent borrows) → fall back
    to a direct one-shot connection, i.e. degrade to exactly the old
    behavior rather than block or fail.

Env:
  PG_POOL_MAX — max pooled connections per DSN (default 5). Keep
  (workers × PG_POOL_MAX) under the server's max_connections budget;
  the portal's own pg pool (portal/lib/db.ts, max 10) shares the same
  database.
"""

from __future__ import annotations

import os
import threading
from typing import Any, Optional

_pools: dict[str, Any] = {}
_pools_lock = threading.Lock()


def _max_conn() -> int:
    try:
        return max(1, int(os.environ.get("PG_POOL_MAX", "5")))
    except ValueError:
        return 5


def _get_pool(dsn: str):
    from psycopg2.pool import ThreadedConnectionPool  # type: ignore

    with _pools_lock:
        pool = _pools.get(dsn)
        if pool is None or getattr(pool, "closed", False):
            pool = ThreadedConnectionPool(1, _max_conn(), dsn)
            _pools[dsn] = pool
        return pool


class _PooledConnection:
    """Proxy over a psycopg2 connection: `close()` returns it to the pool.

    `with conn:` (transaction scope) and every attribute/method delegate
    to the real connection, so call sites can't tell the difference from
    a direct `psycopg2.connect()` result.
    """

    def __init__(self, pool: Any, conn: Any) -> None:
        self._pool = pool
        self._conn: Optional[Any] = conn

    def close(self) -> None:
        conn, self._conn = self._conn, None
        if conn is None:  # double-close is a no-op, same as psycopg2
            return
        try:
            # putconn rolls back an in-progress transaction and discards
            # broken connections (transaction status unknown) itself.
            self._pool.putconn(conn)
        except Exception:  # fail-open: pool already closed/full — don't leak the socket
            try:
                conn.close()
            except Exception:  # fail-open: connection already dead; nothing left to release
                pass

    @property
    def closed(self) -> bool:
        return self._conn is None or bool(self._conn.closed)

    def __enter__(self):
        return self._conn.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self._conn.__exit__(exc_type, exc_val, exc_tb)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._conn, name)


def connect(dsn: str):
    """Borrow a pooled connection (ping-validated); see module docstring.

    Drop-in replacement for `psycopg2.connect(dsn)` at the stores'
    `_connect()` sites — the returned object's `.close()` releases back
    to the pool instead of tearing down the connection.
    """
    import psycopg2  # type: ignore

    try:
        pool = _get_pool(dsn)
    except Exception:
        # Pool construction failed (e.g. DB briefly unreachable at first
        # borrow) — old behavior: let the direct connect raise or serve.
        return psycopg2.connect(dsn)

    # Discard dead pooled connections until a live one is found. Bounded
    # by pool size + 1 so a fully-stale pool (DB restart) converges to
    # fresh connections instead of looping.
    for _ in range(_max_conn() + 1):
        try:
            conn = pool.getconn()
        except Exception:
            # Pool exhausted — degrade to the pre-pool per-call behavior.
            return psycopg2.connect(dsn)
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
            conn.rollback()  # leave no open transaction from the ping
            return _PooledConnection(pool, conn)
        except Exception:  # fail-open: stale/broken pooled conn — discard, try next
            try:
                pool.putconn(conn, close=True)
            except Exception:  # fail-open: pool refused the return; close directly
                try:
                    conn.close()
                except Exception:  # fail-open: already closed
                    pass
    return psycopg2.connect(dsn)


def close_all() -> None:
    """Close every pool (tests / worker shutdown)."""
    with _pools_lock:
        for pool in _pools.values():
            try:
                pool.closeall()
            except Exception:  # fail-open: shutdown path; a dead pool is already closed
                pass
        _pools.clear()


# ── One-time schema bootstrap ─────────────────────────────────────────────────

_MIGRATED: set[tuple[str, str]] = set()


def ensure_schema(dsn: str, ddl: "str | list[str]", *, key: str) -> None:
    """Run idempotent DDL once per (dsn, key) per process.

    Three backends bootstrap their own table on construction — the DLQ, the
    idempotency store and the gateway's budget ledger — and only the DLQ
    remembered it had already done so. The other two re-ran CREATE TABLE IF NOT
    EXISTS on every construction, and `LLMGateway(...)` is built per activity:
    on Postgres that is two extra DDL round-trips before any real work, on the
    hot path of every workflow step.

    Cost is not the whole of it. DDL takes a brief table-level lock even when it
    is a no-op, so a burst of workers each constructing a gateway at once
    serialise on a statement that had nothing left to do — the failure shows up
    as latency under concurrency, which is the hardest kind to attribute.

    Keyed by (dsn, key) rather than dsn alone: one database legitimately holds
    all three tables, and a per-DSN flag would let whichever backend ran first
    suppress the others' migrations.

    `ddl` may be one statement or a list of them, applied in order.

    The cache is per PROCESS and deliberately not persisted — a fresh process
    against a database that has been reset must still create its tables.
    """
    token = (dsn, key)
    if token in _MIGRATED:
        return
    statements = [ddl] if isinstance(ddl, str) else list(ddl)
    conn = connect(dsn)
    try:
        with conn, conn.cursor() as cur:
            # A list rather than one semicolon-joined string: the DLQ's
            # migration is CREATE TABLE followed by three ALTER ... ADD COLUMN
            # IF NOT EXISTS, and relying on multi-statement execute() would
            # make the bootstrap depend on driver behaviour that differs
            # between psycopg2 and psycopg3.
            for statement in statements:
                cur.execute(statement)
    finally:
        conn.close()
    _MIGRATED.add(token)


def reset_migration_cache() -> None:
    """Forget which schemas have been bootstrapped. For tests only."""
    _MIGRATED.clear()
