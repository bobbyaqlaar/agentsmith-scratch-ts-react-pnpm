# Validation & Test Checklist — before merge
## Trigger
After implementation, before opening a PR or claiming a slice done

## Instructions
- Before proceeding, read `$AGENTSMITH_DIR/docs/validation-checklist.md` if you're on a live framework checkout, or `~/.agent-framework/docs/validation-checklist.md` if AgentSmith was installed from the package.
- It works docs/review-levers.md group by group against the CHANGE, sets the testing obligations (mutation-check anything load-bearing, re-pin stale fixtures), and runs the gates CI actually lists — then signs off per group as checked, not applicable, or a declared gap.
- Owner: unknown@unknown | Project: agentsmith-scratch-ts-react-pnpm
