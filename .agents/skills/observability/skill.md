# OTel Span Emission
## Trigger
Before any LLM call or tool invocation

## Instructions
- (Tracing and Evaluations) All execution must emit OpenTelemetry spans to http://localhost:6006/v1/traces. Every span must carry: agent.name, agent.role, agent.owner_id, tenant.id, llm.model_name, project.name, environment. Input/output content policy: full in dev, scrubbed in staging, minimal in prod.
- (Observability Wire) Telemetry is OTLP with the attributes and instruments contract/telemetry/v1 catalogues, and `governance.telemetry.contract` on the Resource. The endpoint comes from the standard OTEL_EXPORTER_OTLP_* variables, never from a path into a provider's install; AgentSmith's runtime library reads them through runtime/otlp.py, and plain OpenTelemetry reads them too. `agentsmith conformance --port telemetry` judges what an emitter exports.
- OTel endpoint: http://localhost:6006/v1/traces
- Owner: unknown@unknown | Project: agentsmith-scratch-ts-react-pnpm
