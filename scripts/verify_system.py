"""
verify_system.py — AgentSmith installation health checker.

Validates:
  1. Python version (≥3.11)
  2. Required packages importable
  3. Git hooks installed and executable
  4. Phoenix connectivity
  5. Network / Ollama status
  6. Agent identity configured
  7. Unresolved MAJOR/CRITICAL log entries in current project

Used by:
  - ai-stack-check shell function
  - GitHub Actions CI (optional smoke-test step)

Exit codes:
  0 = all checks passed
  1 = one or more checks failed
"""

from __future__ import annotations

import importlib
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

REQUIRED_PACKAGES = [
    "phoenix",  # arize-phoenix
    "opentelemetry",  # opentelemetry-sdk
    "networkx",  # networkx>=3.0
    "langgraph",  # langgraph>=0.2 (soft)
    "tiktoken",  # cost router token counting
    "httpx",  # LLM API calls
    "plyer",  # notifications
    "tenacity",  # retry logic
]

SOFT_PACKAGES = {"langgraph"}  # warn-only; not hard requirement


from _shared import (  # noqa: E402
    _repo_root,
    fixtures_path,
    judge_model as _resolve_judge_model,
)


def _required_ollama_models() -> list[str]:
    """Ollama model ids the merged registry actually routes to.

    Read from models.yaml rather than hardcoded, for the same reason the eval
    judge is (scripts/_shared.py:judge_model()): a second hand-maintained list
    of model names drifts silently, and this one had — it still named
    llama3/mistral/gemma2 long after the registry moved on, so the check told
    users to pull models nothing used while the ones the gateway needs went
    unverified. Falls back to that historical list only if runtime/ isn't
    importable (scripts-only install).
    """
    from _shared import provider_models

    # Same helper install-ai-stack.sh's ai-stack-required-models shells out to,
    # so the health check and this check can never disagree about what to pull.
    return provider_models("ollama") or ["qwen2.5", "llama3.2:3b", "falcon3:3b", "smollm2"]


def _check(label: str, ok: bool, detail: str = "", warn_only: bool = False) -> bool:
    if ok:
        print(f"  ✅  {label}")
    elif warn_only:
        print(f"  ⚠️   {label}" + (f" — {detail}" if detail else ""))
    else:
        print(f"  ❌  {label}" + (f" — {detail}" if detail else ""))
    return ok or warn_only


def run_checks() -> bool:
    failures = 0

    print("═══════════════════════════════════════════════════")
    print("  AgentSmith System Verification")
    print("═══════════════════════════════════════════════════\n")

    # ── 1. Python version ─────────────────────────────────────────────────────
    print("Python environment:")
    major, minor = sys.version_info.major, sys.version_info.minor
    ok = major >= 3 and minor >= 11
    if not _check(f"Python {major}.{minor}", ok, "Requires ≥3.11"):
        failures += 1
    print()

    # ── 2. Required packages ──────────────────────────────────────────────────
    print("Required packages:")
    for pkg in REQUIRED_PACKAGES:
        is_soft = pkg in SOFT_PACKAGES
        try:
            mod = importlib.import_module(pkg)
            version = getattr(mod, "__version__", "?")
            _check(f"{pkg} ({version})", True, warn_only=is_soft)
        except ImportError:
            msg = f"pip install {pkg}"
            if not _check(pkg, False, msg, warn_only=is_soft):
                failures += 1
    print()

    # ── 3. Git hooks ──────────────────────────────────────────────────────────
    print("Git hooks:")
    template_dir = Path.home() / ".git_templates" / "hooks"
    for hook in ["pre-commit", "commit-msg", "post-commit", "post-checkout"]:
        hook_path = template_dir / hook
        ok = hook_path.exists() and os.access(hook_path, os.X_OK)
        if not _check(f"hook: {hook}", ok, f"Expected at {hook_path}"):
            failures += 1
    # templateDir global config
    try:
        configured_dir = subprocess.check_output(
            ["git", "config", "--global", "init.templateDir"],
            text=True,
        ).strip()
        ok = str(template_dir) in configured_dir or configured_dir.endswith(
            ".git_templates"
        )
        if not _check("git init.templateDir set", ok, f"Got: {configured_dir!r}"):
            failures += 1
    except subprocess.CalledProcessError:
        if not _check("git init.templateDir set", False, "Not configured"):
            failures += 1
    print()

    # ── 4. Phoenix connectivity ───────────────────────────────────────────────
    print("Observability:")
    phoenix_endpoint = os.environ.get("AGENT_PHOENIX_ENDPOINT", "http://localhost:6006")
    try:
        import httpx

        resp = httpx.get(phoenix_endpoint, timeout=3.0)
        phoenix_ok = resp.status_code in (
            200,
            301,
            302,
            404,
        )  # 404 = server up, unknown path
    except Exception:
        phoenix_ok = False
    if not _check(
        f"Phoenix @ {phoenix_endpoint}",
        phoenix_ok,
        "Run: ai-dashboard-start",
        warn_only=True,
    ):
        pass  # warn-only: offline phoenix is allowed
    print()

    # ── 5. Network / Ollama ───────────────────────────────────────────────────
    print("Network & local models:")
    try:
        s = socket.create_connection(("1.1.1.1", 53), timeout=2)
        s.close()
        _check("Internet connectivity", True)
    except OSError:
        _check(
            "Internet connectivity", False, "Offline — local mode only", warn_only=True
        )

    try:
        import httpx

        resp = httpx.get("http://localhost:11434/api/tags", timeout=3.0)
        ollama_ok = resp.status_code == 200
        models = [m["name"] for m in resp.json().get("models", [])] if ollama_ok else []
        _check(
            "Ollama daemon",
            ollama_ok,
            "Run: ollama serve" if not ollama_ok else "",
            warn_only=True,
        )
        for required_model in _required_ollama_models():
            # Exact id, or the same id with an implicit ":latest" tag — not a
            # substring match. The old hardcoded ["llama3", "mistral",
            # "gemma2"] list drifted from models.yaml AND matched loosely, so
            # "llama3" reported present because llama3.2:latest happened to be
            # installed, while the models the registry actually routes to went
            # unchecked.
            present = any(
                m == required_model or m.split(":")[0] == required_model.split(":")[0]
                if ":" not in required_model
                else m == required_model
                for m in models
            )
            _check(
                f"  model: {required_model}",
                present,
                f"ollama pull {required_model}",
                warn_only=True,
            )
    except Exception:
        _check("Ollama daemon", False, "Run: ollama serve", warn_only=True)
    print()

    # ── 6. Agent identity ─────────────────────────────────────────────────────
    print("Identity:")
    owner_id = os.environ.get("AGENT_OWNER_ID", "")
    owner_name = os.environ.get("AGENT_OWNER_NAME", "")
    if not _check(
        "AGENT_OWNER_ID set", bool(owner_id), "export AGENT_OWNER_ID=you@example.com"
    ):
        failures += 1
    if not _check(
        "AGENT_OWNER_NAME set", bool(owner_name), "export AGENT_OWNER_NAME='Your Name'"
    ):
        failures += 1
    judge = _resolve_judge_model()
    _check(f"AGENT_JUDGE_MODEL: {judge}", True)
    print()

    # ── 7. Unresolved MAJOR/CRITICAL in current project ───────────────────────
    print("Project log:")
    root = _repo_root()
    log_file = root / ".agent-history.log"
    if not log_file.exists():
        _check("No .agent-history.log (new project)", True)
    else:
        unresolved = []
        with log_file.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    if entry.get("level") in ("MAJOR", "CRITICAL") and not entry.get(
                        "hitl_resolved", True
                    ):
                        unresolved.append(entry)
                # fail-open: one malformed JSON-lines entry must not abort scanning the
                # rest of the log for unresolved issues
                except Exception:
                    pass

        if unresolved:
            _check(
                f"{len(unresolved)} unresolved MAJOR/CRITICAL issue(s)",
                False,
                "Run 'ai-stack-promote' or resolve in Phoenix UI",
            )
            for entry in unresolved[:5]:
                print(
                    f"       [{entry['level']}] {entry.get('timestamp', '')}  {entry.get('event', '')}"
                )
            failures += 1
        else:
            _check(".agent-history.log clean (no unresolved issues)", True)
    print()

    # ── Summary ───────────────────────────────────────────────────────────────
    print("═══════════════════════════════════════════════════")
    if failures == 0:
        print("  🎉  All checks passed — environment ready")
    else:
        print(f"  🛑  {failures} check(s) failed — resolve issues above")
    print("═══════════════════════════════════════════════════")

    return failures == 0


def check_security() -> bool:
    """
    Smoke subset of the unified security harness (SEC-PII-001/002 + audit).

    Delegates to scripts/run-security-checks.py --mode smoke.
        python3 scripts/verify_system.py --check-security
    """
    root = Path(__file__).resolve().parent.parent
    script = root / "scripts" / "run-security-checks.py"

    print("═══════════════════════════════════════════════════")
    print("  Security Harness Smoke Check")
    print("═══════════════════════════════════════════════════\n")

    if not script.exists():
        _check("run-security-checks.py present", False, f"missing {script}")
        return False

    proc = subprocess.run(
        [sys.executable, str(script), "--mode", "smoke"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    detail = (proc.stderr or proc.stdout or "").strip()
    if len(detail) > 400:
        detail = detail[:400] + "…"
    ok = proc.returncode == 0
    _check("security harness smoke (--mode smoke)", ok, detail)
    print()
    print("═══════════════════════════════════════════════════")
    if ok:
        print("  🎉  Security smoke passed")
    else:
        print("  🛑  Security smoke failed")
    print("═══════════════════════════════════════════════════")
    return ok


def check_redaction() -> bool:
    """
    CI validation for §27 trace redaction. Runs the active redaction profile
    (from $ENVIRONMENT) against fixture payloads containing known secret/PII
    patterns and fails if any raw pattern survives scrubbing.

    TWO LEVELS, and the second is the one that matters. The fixture loop below
    calls `redactor._scrub(...)` with a string — it proves the PATTERN LIBRARY
    works, and it proved that happily throughout the period when the redactor
    was not redacting anything at all: `on_end` raised TypeError on every real
    span because a finished span's attributes are immutable, and this gate
    never called `on_end`. It also passed while every SEQUENCE-valued attribute
    exported unscrubbed, because it only ever handed the scrubber a `str`.

    So a span now goes through a real TracerProvider with the redactor attached
    and the assertions are made on what the EXPORTER received. A control is
    what reaches the wire, not what a helper returns.

    Used by cd-staging.yml / cd-production.yml:
        python3 scripts/verify_system.py --check-redaction
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from runtime.trace_redactor import TraceRedactor

    redactor = TraceRedactor()
    fixtures = [
        "Authorization: Bearer sk-ant-abcdefghijklmnopqrstuvwxyz0123456789",
        "contact support at someone@example.com about order 4111-1111-1111-1111",
        "raw key sk-abcdefghijklmnopqrstuvwxyz0123456789",
    ]

    print("═══════════════════════════════════════════════════")
    print(f"  Redaction Compliance Check (profile: {redactor.profile})")
    print("═══════════════════════════════════════════════════\n")

    if redactor.profile == "none":
        _check(
            "ENVIRONMENT is development/testing — redaction check skipped",
            True,
            warn_only=True,
        )
        print("═══════════════════════════════════════════════════")
        return True

    failures = 0
    for fixture in fixtures:
        scrubbed = redactor._scrub(
            fixture, hash_identifiers=(redactor.profile == "staging")
        )
        if redactor.profile == "production":
            scrubbed = redactor._truncate(scrubbed)
        leaked = (
            ("sk-" in scrubbed) or ("@example.com" in scrubbed) or ("4111" in scrubbed)
        )
        if not _check(
            f"fixture scrubbed: {fixture[:40]}...", not leaked, f"leaked → {scrubbed!r}"
        ):
            failures += 1

    # ── The control, not the helper ──────────────────────────────────────────
    try:
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
            InMemorySpanExporter,
        )
    except ImportError:
        _check(
            "opentelemetry not installed — the end-to-end span check cannot run "
            "(nothing is exporting spans on this host either)",
            True,
            warn_only=True,
        )
    else:
        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(TraceRedactor())
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        try:
            with provider.get_tracer("verify_system").start_as_current_span("probe") as span:
                span.set_attribute("input.value", fixtures[1])
                # A SEQUENCE, because that is a first-class OTel attribute type
                # and the scrub loop used to skip it entirely.
                span.set_attribute("docs", [fixtures[0], fixtures[2]])
            spans = exporter.get_finished_spans()
        except Exception as exc:  # a redactor that raises out of span.end()
            _check(f"a real span survives export ({exc})", False)
            failures += 1
            spans = []

        if not _check("the redactor emitted a span at all", bool(spans)):
            failures += 1
        else:
            attrs = dict(spans[0].attributes)
            rendered = " ".join(
                part
                for value in attrs.values()
                for part in (value if isinstance(value, (list, tuple)) else [value])
                if isinstance(part, str)
            )
            for marker, what in (("sk-", "an API key"), ("@example.com", "an email"), ("4111", "a card number")):
                if not _check(
                    f"{what} does not survive to the exporter",
                    marker not in rendered,
                    f"leaked → {rendered[:120]!r}",
                ):
                    failures += 1

    print()
    print("═══════════════════════════════════════════════════")
    if failures == 0:
        print("  🎉  Redaction compliance passed")
    else:
        print(f"  🛑  {failures} fixture(s) leaked raw secret/PII patterns")
    print("═══════════════════════════════════════════════════")
    return failures == 0


def check_idempotency() -> bool:
    """
    CI validation for the idempotency store (runtime/idempotency.py) against
    a real Postgres — requires DATABASE_URL pointing at a throwaway database.

        DATABASE_URL=postgresql://test:test@localhost:5432/test \
            IDEMPOTENCY_BACKEND=postgres python3 scripts/verify_system.py --check-idempotency
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    os.environ.setdefault("IDEMPOTENCY_BACKEND", "postgres")
    from runtime.idempotency import IdempotencyStore, make_key

    print("═══════════════════════════════════════════════════")
    print("  Idempotency Store Check")
    print("═══════════════════════════════════════════════════\n")

    failures = 0
    try:
        store = IdempotencyStore()
        key = make_key({"check": "verify_system", "nonce": time.time()})

        if not _check("cache miss before first write", store.get(key) is None):
            failures += 1

        store.set(key, {"ok": True}, ttl_seconds=60)
        if not _check("cache hit after write", store.get(key) == {"ok": True}):
            failures += 1

        store.set(key, {"ok": True}, ttl_seconds=0)
        time.sleep(1)
        if not _check("entry expired after ttl_seconds=0", store.get(key) is None):
            failures += 1
    except Exception as exc:
        _check("idempotency store reachable", False, str(exc))
        failures += 1

    print()
    print("═══════════════════════════════════════════════════")
    print(
        "  🎉  Idempotency check passed"
        if failures == 0
        else f"  🛑  {failures} check(s) failed"
    )
    print("═══════════════════════════════════════════════════")
    return failures == 0


def check_dlq() -> bool:
    """
    CI validation for the dead-letter queue (runtime/dead_letter.py) against
    a real Postgres — requires DATABASE_URL pointing at a throwaway database.

        DATABASE_URL=postgresql://test:test@localhost:5432/test \
            python3 scripts/verify_system.py --check-dlq
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from runtime.dead_letter import DeadLetterQueue

    print("═══════════════════════════════════════════════════")
    print("  Dead-Letter Queue Check")
    print("═══════════════════════════════════════════════════\n")

    failures = 0
    try:
        dlq = DeadLetterQueue()
        tenant_id = f"verify-system-{int(time.time())}"
        entry = dlq.enqueue(
            payload={"check": "verify_system"}, error="synthetic", tenant_id=tenant_id
        )

        pending = [e.task_id for e in dlq.list(tenant_id=tenant_id, status="pending")]
        if not _check(
            "enqueued entry appears in pending list", entry.task_id in pending
        ):
            failures += 1

        replay_calls = []
        dlq_with_handler = DeadLetterQueue(
            replay_handler=lambda e: replay_calls.append(e.task_id)
        )
        dlq_with_handler.replay(entry.task_id)
        if not _check(
            "replay_handler invoked with the right task_id",
            replay_calls == [entry.task_id],
        ):
            failures += 1

        replayed = [e.task_id for e in dlq.list(tenant_id=tenant_id, status="replayed")]
        if not _check("entry moved to replayed status", entry.task_id in replayed):
            failures += 1

        entry2 = dlq.enqueue(
            payload={"check": "discard"}, error="synthetic", tenant_id=tenant_id
        )
        dlq.discard(entry2.task_id)
        discarded = [
            e.task_id for e in dlq.list(tenant_id=tenant_id, status="discarded")
        ]
        if not _check(
            "discarded entry moved to discarded status", entry2.task_id in discarded
        ):
            failures += 1

        try:
            dlq.replay("nonexistent-task-id")
            _check(
                "replay() raises KeyError for unknown task_id", False, "did not raise"
            )
            failures += 1
        except KeyError:
            _check("replay() raises KeyError for unknown task_id", True)
    except Exception as exc:
        _check("DLQ reachable", False, str(exc))
        failures += 1

    print()
    print("═══════════════════════════════════════════════════")
    print(
        "  🎉  DLQ check passed"
        if failures == 0
        else f"  🛑  {failures} check(s) failed"
    )
    print("═══════════════════════════════════════════════════")
    return failures == 0


def check_hooks() -> bool:
    """
    CI validation for the developer opt-in + enterprise RFC gate
    (Product_Archive.md P1a) — simulates both scenarios in throwaway git
    repos so a regression in hooks/pre-commit / hooks/commit-msg fails CI
    instead of only being caught by hand.
    """
    import shutil
    import subprocess
    import tempfile

    repo_root = Path(__file__).resolve().parent.parent
    hooks_dir = repo_root / "hooks"

    print("═══════════════════════════════════════════════════")
    print("  Hook Opt-In / Enterprise RFC Gate Check")
    print("═══════════════════════════════════════════════════\n")

    failures = 0

    def _run(cmd: list, cwd: Path, env: dict) -> subprocess.CompletedProcess:
        return subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, check=False)

    def _git_repo(tmp: Path, name: str) -> Path:
        repo = tmp / name
        repo.mkdir()
        subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"], cwd=repo, check=True
        )
        subprocess.run(["git", "config", "user.name", "test"], cwd=repo, check=True)
        return repo

    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)

        # Scenario 1: no .agenticframework/enabled, no tenant.yaml, no org
        # policy -> pre-commit must no-op even on a staged file that would
        # otherwise trip guardrail 1 (unresolved AI marker).
        repo1 = _git_repo(tmp, "repo1")
        (repo1 / "file.py").write_text("# TODO: agent fix this\n")
        subprocess.run(["git", "add", "file.py"], cwd=repo1, check=True)
        env1 = {**os.environ, "HOME": str(tmp / "fake_home_1")}
        result1 = _run(["bash", str(hooks_dir / "pre-commit")], cwd=repo1, env=env1)
        if not _check(
            "unopted-in repo: pre-commit no-ops (exit 0) despite a tripping pattern",
            result1.returncode == 0,
            result1.stdout + result1.stderr,
        ):
            failures += 1

        # Scenario 2: .agenticframework/enabled present, no org policy ->
        # pre-commit DOES enforce guardrail 1.
        repo2 = _git_repo(tmp, "repo2")
        (repo2 / ".agenticframework").mkdir()
        (repo2 / ".agenticframework" / "enabled").touch()
        (repo2 / "file.py").write_text("# TODO: agent fix this\n")
        subprocess.run(["git", "add", "."], cwd=repo2, check=True)
        env2 = {**os.environ, "HOME": str(tmp / "fake_home_2")}
        result2 = _run(["bash", str(hooks_dir / "pre-commit")], cwd=repo2, env=env2)
        if not _check(
            "opted-in repo: pre-commit enforces guardrail 1 (exit 1)",
            result2.returncode == 1,
            result2.stdout + result2.stderr,
        ):
            failures += 1

        # Scenario 3: enterprise org policy present + opted-in, commit
        # message has no RFC-NNN -> commit-msg blocks.
        repo3 = _git_repo(tmp, "repo3")
        (repo3 / ".agenticframework").mkdir()
        (repo3 / ".agenticframework" / "enabled").touch()
        fake_home_3 = tmp / "fake_home_3"
        (fake_home_3 / ".agent-framework").mkdir(parents=True)
        (fake_home_3 / ".agent-framework" / "agenticframework-org.yaml").write_text(
            "hooks:\n  bypass_policy: disabled\n"
        )
        msg_file = tmp / "msg_no_rfc.txt"
        msg_file.write_text("feat(x): no rfc reference here\n")
        env3 = {**os.environ, "HOME": str(fake_home_3)}
        result3 = _run(
            ["bash", str(hooks_dir / "commit-msg"), str(msg_file)], cwd=repo3, env=env3
        )
        if not _check(
            "enterprise + opted-in, no RFC-NNN in message: commit-msg blocks (exit 1)",
            result3.returncode == 1,
            result3.stdout + result3.stderr,
        ):
            failures += 1

        # Scenario 4: same as 3, but message DOES reference RFC-001 -> passes.
        msg_file2 = tmp / "msg_with_rfc.txt"
        msg_file2.write_text("feat(x): add thing (RFC-001)\n")
        result4 = _run(
            ["bash", str(hooks_dir / "commit-msg"), str(msg_file2)], cwd=repo3, env=env3
        )
        if not _check(
            "enterprise + opted-in, RFC-001 in message: commit-msg passes (exit 0)",
            result4.returncode == 0,
            result4.stdout + result4.stderr,
        ):
            failures += 1

        shutil.rmtree(tmp / "fake_home_1", ignore_errors=True)
        shutil.rmtree(tmp / "fake_home_2", ignore_errors=True)

    print()
    print("═══════════════════════════════════════════════════")
    print(
        "  🎉  Hook gate check passed"
        if failures == 0
        else f"  🛑  {failures} check(s) failed"
    )
    print("═══════════════════════════════════════════════════")
    return failures == 0


def check_history_sync() -> bool:
    """
    CI/manual validation for scripts/sync-portal-history.py against a real
    running Ops Portal (Product_Archive.md P1b) — requires OPS_PORTAL_URL
    and OPS_PORTAL_SYNC_TOKEN pointing at one, run from a throwaway tenant
    repo (with .agenticframework/tenant.yaml and a fixture .agent-history.log).

        OPS_PORTAL_URL=http://localhost:3000 OPS_PORTAL_SYNC_TOKEN=... \
            python3 scripts/verify_system.py --check-history-sync
    """
    import subprocess
    import tempfile

    print("═══════════════════════════════════════════════════")
    print("  History Sync Check")
    print("═══════════════════════════════════════════════════\n")

    ops_portal_url = os.environ.get("OPS_PORTAL_URL", "")
    sync_token = os.environ.get("OPS_PORTAL_SYNC_TOKEN", "")
    if not ops_portal_url or not sync_token:
        _check(
            "OPS_PORTAL_URL and OPS_PORTAL_SYNC_TOKEN set",
            False,
            "required for this check",
        )
        return False

    sync_script = Path(__file__).resolve().parent / "sync-portal-history.py"
    failures = 0

    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        subprocess.run(["git", "init", "-q"], cwd=tmp, check=True)
        tenant_id = f"verify-history-sync-{int(time.time())}"
        (tmp / ".agenticframework").mkdir()
        (tmp / ".agenticframework" / "tenant.yaml").write_text(
            f"tenant:\n  id: {tenant_id}\n"
        )
        (tmp / ".agent-history.log").write_text(
            '{"timestamp":"2026-01-01T00:00:00Z","level":"CRITICAL","event":"verify_check","hitl_resolved":false}\n'
        )

        env = {
            **os.environ,
            "OPS_PORTAL_URL": ops_portal_url,
            "OPS_PORTAL_SYNC_TOKEN": sync_token,
        }
        result1 = subprocess.run(
            ["python3", str(sync_script)],
            cwd=tmp,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        if not _check(
            "first sync run exits 0",
            result1.returncode == 0,
            result1.stdout + result1.stderr,
        ):
            failures += 1

        result2 = subprocess.run(
            ["python3", str(sync_script)],
            cwd=tmp,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        idempotent = "Nothing new to sync" in result2.stdout
        if not _check(
            "second run is idempotent (no resend)", idempotent, result2.stdout
        ):
            failures += 1

    print()
    print("═══════════════════════════════════════════════════")
    print(
        "  🎉  History sync check passed"
        if failures == 0
        else f"  🛑  {failures} check(s) failed"
    )
    print("═══════════════════════════════════════════════════")
    return failures == 0


def _kg_shape(kg_path: Path) -> Optional[str]:
    """The graph's meaning, with the parts that legitimately churn removed.

    Compared instead of raw bytes for two reasons:

    * `last_modified` is a file mtime. `actions/checkout` stamps the working
      tree at checkout time, so on CI every node's mtime differs from the
      committed graph's and a byte compare would report drift on every run —
      a gate that always fails is as useless as one that never does.
    * Node and edge order is an artifact of a directory walk, not of content.

    What remains — which files exist, their language, their symbols, and the
    import edges between them — is exactly what goes stale when someone edits
    code and does not regenerate.

    KNOWN LIMIT: map_codebase skips files whose stored mtime still matches, at
    1-second granularity, so an edit made within the same second as the last
    regeneration is not re-parsed and this check passes. Irrelevant in CI —
    actions/checkout stamps every file fresh, so nothing is ever skipped — and
    self-correcting locally on the next run once the clock ticks.
    """
    if not kg_path.exists():
        return None
    try:
        data = json.loads(kg_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None

    tracked = _git_tracked_files()
    if tracked is None:
        return None

    # GIT-TRACKED CodebaseFile NODES ONLY, and no Guardrail nodes. The mapper
    # walks the filesystem, so the graph legitimately differs between machines
    # and CI is the one that differs most:
    #
    #   portal/next-env.d.ts   written by `next build`, gitignored. Present on
    #                          any machine that has built the portal, absent on
    #                          a fresh runner — the `purged: 1` in CI.
    #   rfc:fixtures/…         Guardrail nodes are sourced from .agent-rfc docs,
    #                          some of which are GENERATED and gitignored
    #                          (delivery_evidence.md). Worse, they store an
    #                          ABSOLUTE source_file path, which cannot match
    #                          across machines at all.
    #
    # Comparing those made the gate fail on every CI run — a false-positive
    # gate, which is worse than the weak one it replaced, because the fix is to
    # delete it. What is left is exactly what goes stale when someone edits
    # committed code without regenerating.
    nodes = sorted(
        (
            n.get("id"),
            n.get("language"),
            tuple(sorted(n.get("symbols") or [])),
        )
        for n in data.get("nodes", [])
        if n.get("node_type") == "CodebaseFile" and n.get("id") in tracked
    )
    raw_edges = data.get("edges", data.get("links", []))
    edges = sorted(
        (e.get("source"), e.get("target"), e.get("edge_type") or e.get("type"))
        for e in raw_edges
        if e.get("source") in tracked and e.get("target") in tracked
    )
    return json.dumps({"nodes": nodes, "edges": edges}, sort_keys=True, default=str)


def _git_tracked_files() -> Optional[set[str]]:
    """Repo-relative paths git knows about, or None if git cannot answer.

    None makes the caller skip the drift comparison rather than guess. Treating
    "git unavailable" as "nothing is tracked" would compare two empty shapes
    and report the graph up to date having examined no nodes at all.
    """
    try:
        proc = subprocess.run(
            ["git", "ls-files"],
            cwd=_repo_root(),
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return set(proc.stdout.split())


def check_kg() -> bool:
    """
    CI validation for the Knowledge Graph (Product_Archive.md P10a, Pillar 2).
    Runs map_codebase.py against the framework's own codebase and asserts the
    resulting graph is non-empty with at least the known scripts/ file nodes.

        python3 scripts/verify_system.py --check-kg
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import map_codebase  # type: ignore

    print("═══════════════════════════════════════════════════")
    print("  Knowledge Graph Check (self-test.yml)")
    print("═══════════════════════════════════════════════════\n")

    kg_path = fixtures_path("knowledge_graph.json")
    # Captured BEFORE run_map() overwrites it. Without this the check only ever
    # proved the mapper runs: it regenerated the graph and then asserted things
    # about the graph it had just written, so a committed file stale by two
    # releases passed every time. Observed 2026-08-24 — the committed graph was
    # 703 lines behind the portal work and the gate had been green throughout.
    before = _kg_shape(kg_path)

    # force=True: this check's regeneration is the reference the committed
    # graph is measured against, so it must not inherit whatever the
    # incremental path last left behind. Without it a graph corrupted by
    # anything other than a file edit compares equal to itself forever.
    stats = map_codebase.run_map(force=True)
    print(f"  ℹ️   map_codebase.py: {stats}")

    failures = 0

    if not _check("knowledge_graph.json written", kg_path.exists()):
        failures += 1
        print()
        print("═══════════════════════════════════════════════════")
        print(f"  🛑  {failures} check(s) failed")
        print("═══════════════════════════════════════════════════")
        return False

    with kg_path.open() as fh:
        data = json.load(fh)

    nodes = data.get("nodes", [])
    node_ids = {n.get("id") for n in nodes}
    edges = data.get("edges", data.get("links", []))

    if not _check("Knowledge Graph is non-empty", len(nodes) > 0):
        failures += 1

    # The drift check the name promised — on SHAPE, not bytes. See _kg_shape:
    # a byte compare would fail every CI run, because actions/checkout stamps
    # working-tree mtimes at checkout time and every `last_modified` in a
    # freshly built graph would differ from the committed one.
    if before is not None and before != _kg_shape(kg_path):
        # run_map() above has ALREADY rewritten the file, so the working tree
        # is now correct and a second run of this check goes green. Say so, or
        # "just run it again" looks like a flaky gate instead of a reminder
        # that the fix is still uncommitted.
        _check(
            "knowledge_graph.json was stale — it has just been regenerated in "
            "place; COMMIT .agent-rfc/fixtures/knowledge_graph.json. "
            "(Re-running this check will now pass without the commit.)",
            False,
        )
        failures += 1
    else:
        _check("knowledge_graph.json is up to date with the code", True)

    known_files = {
        "scripts/map_codebase.py",
        "scripts/verify_system.py",
        "scripts/local_knowledge_graph.py",
    }
    missing = known_files - node_ids
    if not _check(
        f"known scripts/ file nodes present ({len(known_files)} expected)",
        not missing,
        f"missing: {missing}",
    ):
        failures += 1

    print(f"  ℹ️   {len(nodes)} nodes, {len(edges)} edges")

    print()
    print("═══════════════════════════════════════════════════")
    print(
        "  🎉  Knowledge Graph check passed"
        if failures == 0
        else f"  🛑  {failures} check(s) failed"
    )
    print("═══════════════════════════════════════════════════")
    return failures == 0


def check_onprem_deploy() -> bool:
    """
    Syntax/shape validation for templates/onprem-deploy/ (OPERATIONS.md
    D.6) — no live cluster or Docker daemon required beyond `docker
    compose config` and `helm template`'s own dry-run rendering. Renders
    both proxy engines' configs with canary+shadow+with-db all enabled
    (the densest combination) and asserts the output is valid YAML with
    the expected weighted/mirror shape, then does the same for the Helm
    chart across both proxyEngine values.

        python3 scripts/verify_system.py --check-onprem-deploy
    """
    import shutil
    import subprocess
    import tempfile

    import yaml

    print("═══════════════════════════════════════════════════")
    print("  On-Prem Deploy Template Check")
    print("═══════════════════════════════════════════════════\n")

    base = _repo_root() / "templates" / "onprem-deploy"
    failures = 0

    if not base.exists():
        _check("templates/onprem-deploy/ exists", False)
        return False
    _check("templates/onprem-deploy/ exists", True)

    have_docker = shutil.which("docker") is not None
    have_helm = shutil.which("helm") is not None

    if have_docker:
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            env_file = tmp / ".env"
            env_file.write_text(
                "APP_IMAGE_PROD=ghcr.io/acme/app:abc1234\n"
                "APP_IMAGE_CANARY=ghcr.io/acme/app:canary\n"
                "APP_IMAGE_SHADOW=ghcr.io/acme/app:shadow\n"
                "APP_PORT=8080\nCANARY_WEIGHT_PERCENT=15\nSHADOW_MIRROR_PERCENT=100\n"
            )
            # render scripts read ./.env relative to the template dir — copy
            # the fixture env there for the duration of this check.
            shutil.copy(env_file, base / ".env")
            for engine in ("traefik", "envoy"):
                render_script = base / "scripts" / f"render-{engine}-config.py"
                r = subprocess.run(
                    ["python3", str(render_script)],
                    cwd=base,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                rendered_ok = r.returncode == 0
                if not _check(
                    f"{engine}: render script exits 0", rendered_ok, r.stdout + r.stderr
                ):
                    failures += 1

                cfg_file = (
                    base / "proxy" / "traefik" / "dynamic.rendered.yml"
                    if engine == "traefik"
                    else base / "proxy" / "envoy" / "envoy.rendered.yaml"
                )
                if cfg_file.exists():
                    try:
                        yaml.safe_load(cfg_file.read_text())
                        _check(f"{engine}: rendered config is valid YAML", True)
                    except yaml.YAMLError as e:
                        _check(
                            f"{engine}: rendered config is valid YAML", False, str(e)
                        )
                        failures += 1
                    cfg_file.unlink()

                result = subprocess.run(
                    [
                        "docker",
                        "compose",
                        "--env-file",
                        str(base / ".env"),
                        "-f",
                        str(base / "docker-compose.yml"),
                        "-f",
                        str(base / f"docker-compose.{engine}.yml"),
                        "--profile",
                        "canary",
                        "--profile",
                        "shadow",
                        "--profile",
                        "with-db",
                        "config",
                        "--quiet",
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if not _check(
                    f"{engine}: docker compose config validates",
                    result.returncode == 0,
                    result.stderr,
                ):
                    failures += 1
            (base / ".env").unlink(missing_ok=True)
    else:
        _check(
            "docker available for compose validation",
            False,
            "skipped — docker not installed",
            warn_only=True,
        )

    if have_helm:
        chart = base / "kubernetes"
        result = subprocess.run(
            ["helm", "lint", str(chart)], capture_output=True, text=True, check=False
        )
        if not _check(
            "helm lint passes", result.returncode == 0, result.stdout + result.stderr
        ):
            failures += 1
        for set_args, label in [
            ([], "defaults"),
            (
                [
                    "--set",
                    "canary.enabled=true",
                    "--set",
                    "shadow.enabled=true",
                    "--set",
                    "withDb.enabled=true",
                    "--set",
                    "withDb.credentialsSecretName=db-creds",
                ],
                "canary+shadow+db",
            ),
            (["--set", "proxyEngine=envoy-gateway"], "envoy-gateway"),
        ]:
            result = subprocess.run(
                ["helm", "template", "verify-check", str(chart), *set_args],
                capture_output=True,
                text=True,
                check=False,
            )
            ok = result.returncode == 0
            if ok:
                try:
                    list(yaml.safe_load_all(result.stdout))
                except yaml.YAMLError as e:
                    ok = False
                    result.stderr += str(e)
            if not _check(
                f"helm template ({label}) renders valid YAML", ok, result.stderr
            ):
                failures += 1
    else:
        _check(
            "helm available for chart validation",
            False,
            "skipped — helm not installed",
            warn_only=True,
        )

    print()
    print("═══════════════════════════════════════════════════")
    print(
        "  🎉  On-prem deploy check passed"
        if failures == 0
        else f"  🛑  {failures} check(s) failed"
    )
    print("═══════════════════════════════════════════════════")
    return failures == 0


def check_delivery_model() -> bool:
    """
    Soft gate for Enterprise Delivery Model (Product_Archive.md).

    If `.agenticframework/org-policy.yaml` defines `delivery_model`, warn when
    tenant `delivery.platform` / `data_access_pattern` are missing or not in
    the approved catalog. Never fails CI (warn-only).

        python3 scripts/verify_system.py --check-delivery-model
    """
    scripts_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(scripts_dir))
    from delivery_model import check_tenant_against_policy  # type: ignore
    from _shared import _repo_root  # type: ignore

    print("═══════════════════════════════════════════════════")
    print("  Delivery Model Soft Gate")
    print("═══════════════════════════════════════════════════\n")

    result = check_tenant_against_policy(_repo_root())
    status = result["status"]
    if status == "skip":
        _check(
            "No delivery_model in org-policy — soft gate skipped",
            True,
            warn_only=True,
        )
    elif status == "ok":
        _check("Tenant delivery.* matches approved catalog", True)
    else:
        for warning in result.get("warnings") or []:
            _check(warning, False, warn_only=True)
        _check(
            "Delivery Model soft gate complete (warnings above; CI not failed)",
            True,
            warn_only=True,
        )

    print()
    print("═══════════════════════════════════════════════════")
    print("  Soft gate finished (ok_for_ci=True)")
    print("  Evidence pack: python3 scripts/delivery_evidence.py")
    print("═══════════════════════════════════════════════════")
    return True  # soft gate — never hard-fail


MODES = [
    ("--check-security", "security harness registry + control coverage"),
    ("--check-redaction", "trace redaction profile for $ENVIRONMENT"),
    ("--check-idempotency", "idempotency store round-trip"),
    ("--check-dlq", "dead-letter queue replay path"),
    ("--check-history-sync", "agent history ↔ portal sync state"),
    ("--check-hooks", "hook opt-in / enterprise RFC gate"),
    ("--check-onprem-deploy", "on-prem bundle renders and self-checks"),
    ("--check-kg", "Knowledge Graph freshness (drift gate)"),
    ("--check-delivery-model", "delivery-model evidence soft gate"),
]


def _usage() -> None:
    """`--help` used to fall through to the full verification run.

    Every other entry point in scripts/ uses argparse and answers `--help` in
    the ordinary way. This one dispatches on `sys.argv` — nine `in sys.argv`
    tests — so `--help` matched none of them, fell through to `run_checks()`,
    and printed a multi-second system scan. Exit 0, no usage, and the nine
    modes below documented in five different .md files and discoverable from
    the tool itself nowhere.

    Which is the same shape as the rest of this file's subject matter: a
    command that quietly does something other than what it was asked, and
    reports success.
    """
    print("usage: verify_system.py [MODE]\n")
    print("Runs the full system verification when given no mode.\n")
    print("Modes (each runs exactly one check and exits non-zero on failure):")
    width = max(len(f) for f, _ in MODES)
    for flag, what in MODES:
        print(f"  {flag:<{width}}  {what}")
    print("\nThese are single-purpose CI entry points, not composable flags:")
    print("the first one matched wins and the rest are ignored.")


if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        _usage()
        sys.exit(0)
    if "--check-redaction" in sys.argv:
        sys.exit(0 if check_redaction() else 1)
    if "--check-security" in sys.argv:
        sys.exit(0 if check_security() else 1)
    if "--check-idempotency" in sys.argv:
        sys.exit(0 if check_idempotency() else 1)
    if "--check-dlq" in sys.argv:
        sys.exit(0 if check_dlq() else 1)
    if "--check-history-sync" in sys.argv:
        sys.exit(0 if check_history_sync() else 1)
    if "--check-hooks" in sys.argv:
        sys.exit(0 if check_hooks() else 1)
    if "--check-onprem-deploy" in sys.argv:
        sys.exit(0 if check_onprem_deploy() else 1)
    if "--check-kg" in sys.argv:
        sys.exit(0 if check_kg() else 1)
    if "--check-delivery-model" in sys.argv:
        sys.exit(0 if check_delivery_model() else 1)
    ok = run_checks()
    sys.exit(0 if ok else 1)
