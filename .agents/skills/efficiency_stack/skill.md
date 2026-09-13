# AgentSmith Lifecycle Guardrails
## Trigger
Before writing any code or making file changes

## Instructions
- (Requirements and Design) An .agent-rfc/ directory must exist. Do not modify source files unless a markdown spec exists in .agent-rfc/. Before any change, produce a step-by-step implementation blueprint.
- (Build Architecture (Ponytail)) Run a 5-step analysis before creating new files: (1) does this already exist? (2) is there a native library? (3) what is the minimal change? (4) what does this affect? (5) are there downstream graph dependencies? No unapproved third-party dependencies.
- (Testing Guardrails) Every logical change requires a corresponding unit or integration test. Never clear or skip existing tests to force green coverage.
- (Operations and Self-Improvement) On every session start, read .agent-history.log to avoid repeating past mistakes. After two consecutive identical tool failures, log MAJOR, halt, and escalate.
- (Interface Constraints (Caveman Compression)) Default to code blocks, data structures, terminal commands and variables. Skip pleasantries, preambles and meta-summaries — they cost tokens and add nothing. This removes filler; it does not withhold judgement. Always say, in plain sentences, what failed and why you think so, a risk you are taking, an assumption you had to make, or the reason you stopped. Pillar 5 tells you to escalate after two identical failures, and an escalation nobody can read is not an escalation. Terse by default, explicit when something is wrong.
- OTel endpoint: http://localhost:6006/v1/traces
- Owner: unknown@unknown | Project: agentsmith-scratch-ts-react-pnpm
