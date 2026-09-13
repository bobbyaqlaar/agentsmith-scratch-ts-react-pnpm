"""
runtime/workflows/ — Reference durable workflow definitions.

This package contains reference workflow patterns only.
Tenant repos define their OWN production workflow files.
Do NOT deploy examples from this directory as tenant production code.

See SPECS.md §25 and examples/oil-price-agent/workflows/ for a complete
domain reference built on top of this pattern.

Reference workflows:
  base_workflow.py — BaseAgentWorkflow: HITL pause/resume signal handling with
                      a DLQ-routed timeout. Subclass this; don't deploy it.
"""
