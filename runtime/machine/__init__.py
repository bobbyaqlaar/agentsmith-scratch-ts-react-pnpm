"""
runtime/machine — the operator commands behind `agentsmith`, and the machine
state they share with the hooks and the gateway.

These were eighteen shell functions `install-ai-stack.sh` appended to
`~/.zshrc`. A function in a login shell exists for that shell only: a git GUI,
an IDE, CI and Claude Code's hooks never saw `ai-mode-hybrid`'s export, and
nothing could test `ai-stack-upgrade` without carving it out of a heredoc. See
.agent-rfc/designs/agentsmith-cli.md.

Import-light on purpose: `agentsmith hooks bypass-check` runs inside a git hook,
so nothing here imports a third-party package at module level.
"""
