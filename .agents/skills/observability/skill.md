# OTel Span Emission
## Trigger
Before any LLM call or tool invocation

## Instructions
- (Tracing and Evaluations) All execution must emit OpenTelemetry spans to http://localhost:6006/v1/traces. Every span must carry: agent.name, agent.role, agent.owner_id, tenant.id, llm.model_name, project.name, environment. Input/output content policy: full in dev, scrubbed in staging, minimal in prod.
- (Observability Wire) .cursorrules and CLAUDE.md include explicit OTLP endpoint instructions. OTEL_EXPORTER_OTLP_ENDPOINT is set in the shell when dashboard starts.
- OTel endpoint: http://localhost:6006/v1/traces
- Owner: unknown@unknown | Project: agentsmith-scratch-ts-react-pnpm
