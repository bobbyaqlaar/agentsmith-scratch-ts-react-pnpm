"""
agent_logger.py — AgentSmith structured log writer.

Writes JSON-Lines to stdout and appends to .agent-history.log in the repo root.
Enforces 4-level severity model: INFO / MINOR / MAJOR / CRITICAL.
  - INFO / MINOR: pruned via FIFO at 10,000 entries (post-commit hook handles pruning).
  - MAJOR / CRITICAL: never pruned until hitl_resolved: true.

Calls audit_token_velocity_circuit() on every LLM invocation log entry.
All entries carry full agent identity and project attribution.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from typing import Any, Literal, Optional

# ── Types ─────────────────────────────────────────────────────────────────────

Level = Literal["INFO", "MINOR", "MAJOR", "CRITICAL"]

# ── Helpers ───────────────────────────────────────────────────────────────────
# _repo_root/_iso_now/_tenant_id used to be defined here; consolidated into
# _shared.py since they were byte-for-byte duplicated across most of
# scripts/*.py (this file was the canonical version _shared.py was lifted
# from — see that module's docstring for why it's not also shared with
# runtime/llm_gateway.py's separate copy).
from _shared import _repo_root, _iso_now, _tenant_id  # noqa: E402


def _project_name() -> str:
    root = _repo_root()
    remote = ""
    try:
        import subprocess

        remote = subprocess.check_output(
            ["git", "remote", "get-url", "origin"],
            text=True,
            cwd=root,
        ).strip()
        if remote:
            return remote.rstrip("/").split("/")[-1].removesuffix(".git")
    # fail-open: no git remote / not a git repo / git not installed all fall back to the
    # dir name below
    except Exception:
        pass
    return root.name


# ── Core logger ───────────────────────────────────────────────────────────────


def _git_config(key: str) -> Optional[str]:
    """`git config user.email`, or None. The dev-session fallback."""
    try:
        out = subprocess.run(
            ["git", "config", "--get", key], capture_output=True, text=True, timeout=5, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = out.stdout.strip()
    return value or None


def _owner(dotted: str, env_var: str, git_key: str) -> str:
    """Who is running this: the tenant's declaration, an explicit override, or git.

    The git fallback is what lets `AGENT_OWNER_ID` leave the shell profile. The
    installer used to export it there, which looks like a per-deploy override
    and is actually AMBIENT: it applied to every repo on the machine and, under
    the standard precedence, silently outranked whatever each tenant declared as
    its owner. One developer with one address never notices; a second person or
    a second tenant does.

    Outside a tenant — the dev-session tooling this logger mostly serves — git
    already knows the answer and needs no configuration at all.

    The import is guarded because `scripts/` is machine-installed and `runtime/`
    ships as a pip package: a standalone invocation from a tenant directory has
    no `runtime` on sys.path, and an unguarded import here killed AgentLogger's
    constructor — the same defect, and the same day, as the one `_load_dotenv`
    in `_shared` documents at length.

    The degraded path deliberately does NOT rebuild `resolve`'s precedence from
    parts. That chain is four layers deep (permitted override → .env →
    tenant.yaml → ambient) and its ORDER is the load-bearing part; a
    hand-reconstruction that put .env and tenant.yaml the wrong way round would
    reintroduce exactly the ambient-outranks-declaration bug described above,
    and it would do it silently. Without the runtime, tenant.yaml and .env are
    not readable as configuration at all, so what is left is the ambient
    environment and git — in that order, and named as such on the record. This
    is authorship metadata on a log line, not a control, and the alternative is
    a dead process.
    """
    try:
        from runtime.config import resolve
    except ImportError:  # standalone script: runtime/ is not on sys.path
        ambient = os.environ.get(env_var, "").strip()
        return ambient or (_git_config(git_key) or "unknown")

    declared = resolve(dotted, env_var=env_var, default=None)
    return str(declared) if declared else (_git_config(git_key) or "unknown")


class AgentLogger:
    """
    Structured logger for agent sessions.

    Usage:
        logger = AgentLogger(agent_name="Developer", agent_role="subagent",
                             orchestrator="Supervisor")
        logger.info("tool_invoked", tool="write_file", path="src/api.py")
        logger.major("empty_catch_detected", file="src/handler.py", line=42)
    """

    def __init__(
        self,
        agent_name: str,
        agent_role: Literal["orchestrator", "subagent", "standalone"] = "standalone",
        orchestrator: Optional[str] = None,
        session_id: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        self.agent_name = agent_name
        self.agent_role = agent_role
        self.orchestrator = orchestrator
        self.session_id = session_id or str(uuid.uuid4())
        self.model = model or os.environ.get("AGENT_DEFAULT_MODEL", "unknown")
        self.owner_id = _owner("tenant.owner", "AGENT_OWNER_ID", "user.email")
        self.owner_name = _owner("tenant.owner_name", "AGENT_OWNER_NAME", "user.name")
        self.project = _project_name()
        self.tenant_id = _tenant_id()
        self._log_path = _repo_root() / ".agent-history.log"

    # ── Public API ────────────────────────────────────────────────────────────

    def info(self, event: str, **kwargs: Any) -> dict:
        return self._write("INFO", event, **kwargs)

    def minor(self, event: str, **kwargs: Any) -> dict:
        return self._write("MINOR", event, **kwargs)

    def major(self, event: str, **kwargs: Any) -> dict:
        return self._write("MAJOR", event, **kwargs)

    def critical(self, event: str, **kwargs: Any) -> dict:
        return self._write("CRITICAL", event, **kwargs)

    def llm_call(
        self,
        event: str,
        input_tokens: int,
        output_tokens: int,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> dict:
        """Log an LLM invocation and run circuit breaker check."""
        entry = self._write(
            "INFO",
            event,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=model or self.model,
            **kwargs,
        )
        # Circuit breaker — import lazily to avoid circular deps.
        #
        # Fail-open, as before: the log entry above is already written and a
        # breaker fault must not retract it. But the blanket silent `pass` made
        # every fault indistinguishable from "under both limits" — including a
        # caller passing None token counts, which a provider that omits its
        # `usage` block now produces. Same split as scripts/cost_router.py: the
        # TRIP is an expected outcome and is reported as one; anything else is
        # a fault and says so.
        # The import is its OWN try, and the call is a separate one. With both
        # inside a single try, a failed import left `CircuitBreakerTripped`
        # unbound while still being named by the first except clause — Python
        # evaluates those clauses in order, so resolving the name raised
        # UnboundLocalError and that escaped the try entirely. The handler
        # written to keep this fail-open was the thing that raised, and it did
        # so for a plain `AGENT_MONTHLY_USD_CAP=` in the environment.
        try:
            from circuit_breaker import (
                CircuitBreakerTripped,
                audit_token_velocity_circuit,
            )
        except Exception as exc:
            print(
                f"[agent_logger] WARNING: circuit breaker unavailable "
                f"({type(exc).__name__}: {exc}) — this call is unmetered.",
                file=sys.stderr,
            )
        else:
            try:
                audit_token_velocity_circuit(input_tokens, output_tokens)
            except CircuitBreakerTripped as tripped:
                print(f"[agent_logger] {tripped}", file=sys.stderr)
            except Exception as exc:
                print(
                    f"[agent_logger] WARNING: circuit breaker bookkeeping failed "
                    f"({type(exc).__name__}: {exc}) — this call is unmetered.",
                    file=sys.stderr,
                )
        return entry

    # ── Internal ──────────────────────────────────────────────────────────────

    def _write(self, level: Level, event: str, **kwargs: Any) -> dict:
        entry: dict[str, Any] = {
            "timestamp": _iso_now(),
            "level": level,
            "event": event,
            "agent": self.agent_name,
            "agent_role": self.agent_role,
            "session_id": self.session_id,
            "project": self.project,
            "model": self.model,
            "owner_id": self.owner_id,
            "owner_name": self.owner_name,
        }
        if self.tenant_id:
            entry["tenant_id"] = self.tenant_id
        if self.orchestrator:
            entry["orchestrator"] = self.orchestrator
        if level in ("MAJOR", "CRITICAL"):
            entry["hitl_resolved"] = False
            entry["hitl_resolved_by"] = None
            entry["hitl_resolved_at"] = None
        entry.update(kwargs)

        line = json.dumps(entry, default=str)

        # stdout
        print(line, flush=True)

        # append to .agent-history.log
        try:
            with self._log_path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except (
            OSError
        ):  # fail-open: read-only filesystem (CI without checkout) — stdout only
            pass

        return entry

    def resolve_hitl(self, event_filter: str, resolved_by: Optional[str] = None) -> int:
        """
        Mark all unresolved MAJOR/CRITICAL entries whose 'event' matches
        event_filter as resolved.  Returns the count of updated entries.
        """
        if not self._log_path.exists():
            return 0
        resolver = resolved_by or self.owner_id
        ts = _iso_now()
        updated = 0
        lines: list[str] = []
        with self._log_path.open("r", encoding="utf-8") as fh:
            for raw in fh:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    entry = json.loads(raw)
                    if (
                        entry.get("event") == event_filter
                        and entry.get("level") in ("MAJOR", "CRITICAL")
                        and not entry.get("hitl_resolved", True)
                    ):
                        entry["hitl_resolved"] = True
                        entry["hitl_resolved_by"] = resolver
                        entry["hitl_resolved_at"] = ts
                        raw = json.dumps(entry, default=str)
                        updated += 1
                # fail-open: one malformed JSON-lines entry must not abort resolving the
                # rest; raw line is preserved unchanged below either way
                except Exception:
                    pass
                lines.append(raw)
        with self._log_path.open("w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
        return updated

    def unresolved_issues(self) -> list[dict]:
        """Return all unresolved MAJOR/CRITICAL entries for the current project."""
        if not self._log_path.exists():
            return []
        results = []
        with self._log_path.open("r", encoding="utf-8") as fh:
            for raw in fh:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    entry = json.loads(raw)
                    if (
                        entry.get("project") == self.project
                        and entry.get("level") in ("MAJOR", "CRITICAL")
                        and not entry.get("hitl_resolved", True)
                    ):
                        results.append(entry)
                # fail-open: one malformed JSON-lines entry must not abort scanning the
                # rest of the log for unresolved issues
                except Exception:
                    pass
        return results


# ── Module-level convenience instance ─────────────────────────────────────────

# `get_logger()` and its module-level singleton were removed here (2026-08-26):
# no caller anywhere, while every real call site constructs `AgentLogger(...)`
# directly. A process-wide singleton is also the wrong shape for this class —
# one worker serves many agent roles, and the first caller would have fixed the
# role for every later one.


# ── CLI: dump unresolved issues ───────────────────────────────────────────────

if __name__ == "__main__":
    logger = AgentLogger("CLI", "standalone")
    issues = logger.unresolved_issues()
    if not issues:
        print(json.dumps({"status": "clean", "project": logger.project}))
        sys.exit(0)
    for issue in issues:
        print(json.dumps(issue))
    sys.exit(1)
