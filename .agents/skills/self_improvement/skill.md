# Log Monitoring and HITL Escalation
## Trigger
On session start and on repeated failures

## Instructions
- (Operations and Self-Improvement) On every session start, read .agent-history.log to avoid repeating past mistakes. After two consecutive identical tool failures, log MAJOR, halt, and escalate.
- (Multi-Agent Orchestration) Dev sessions may use LangGraph MemorySaver. Production workers must use Temporal or Celery — MemorySaver is prohibited.
- OTel endpoint: http://localhost:6006/v1/traces
- Owner: unknown@unknown | Project: agentsmith-scratch-ts-react-pnpm
