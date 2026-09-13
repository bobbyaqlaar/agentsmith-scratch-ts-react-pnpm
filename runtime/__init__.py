"""
AgentSmith — Production Runtime Layer

This package contains production-grade components for running agents at scale.
It is intentionally separate from the `scripts/` dev-lifecycle package.

Components:
  worker.py         — Temporal/Celery worker entrypoint, partitioned by tenant.id
  llm_gateway.py    — Centralised LLM routing with per-tenant budgets and degrade ladder
  trace_redactor.py — Environment-aware OTLP span scrubbing before export
  idempotency.py    — Idempotency key store and deduplication
  dead_letter.py    — Dead-letter queue and replay API
  workflows/        — Reference durable workflow definitions

These components are NOT used in developer IDE sessions (dev/hybrid mode).
For dev sessions, use scripts/multi_agent_system.py or scripts/local_agent_stack.py.

See SPECS.md §25 for the full production runtime specification.
"""
