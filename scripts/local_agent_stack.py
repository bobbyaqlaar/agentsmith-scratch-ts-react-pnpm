"""
local_agent_stack.py — Pure Python multi-agent pipeline (no framework dependency).

Three-node pipeline:
  Architect  →  Developer  →  Validator

All LLM calls route through cost_router.call().
All events are logged via agent_logger.AgentLogger.
All spans are emitted to Phoenix via OpenTelemetry.

This is the stack used when LangGraph is unavailable or explicitly opted out.
For the LangGraph implementation see multi_agent_system.py.

Usage:
    from local_agent_stack import run_pipeline
    result = run_pipeline(
        task="Build a FastAPI endpoint that returns oil price predictions.",
        spec_file=".agent-rfc/001-oil-price-tracker.md",
    )
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional

# ── OpenTelemetry setup ───────────────────────────────────────────────────────


def _setup_otel(project_name: str, session_id: str) -> Any:
    """Initialise OTLP tracer. Returns the tracer or a no-op if unavailable."""
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.sdk.resources import Resource

        # runtime/otlp.py owns the endpoint, for all four callers that used to
        # own it separately. This one's fallback chain ended at
        # OTEL_EXPORTER_OTLP_ENDPOINT — the variable OPERATIONS.md,
        # docker-compose.yml and `ai-dashboard-start` all set to a full
        # `…/v1/traces` URL — and then appended `/v1/traces` to whatever it
        # found. portal/lib/tracing.ts had documented that exact trap and
        # guarded against it; this sibling, reading the same variable in the
        # same repo, never got the guard.
        from runtime.otlp import span_exporter

        from agent_logger import _tenant_id

        resource_attrs = {
            "service.name": project_name,
            "project.name": project_name,
            "agent.session_id": session_id,
        }
        tenant_id = _tenant_id()
        if tenant_id:
            resource_attrs["tenant.id"] = tenant_id
        resource = Resource.create(resource_attrs)
        provider = TracerProvider(resource=resource)
        exporter = span_exporter()
        if exporter is not None:
            provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        return trace.get_tracer("agenticframework")
    except Exception:
        return _NoopTracer()


class _NoopSpan:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def set_attribute(self, *_):
        pass

    def set_status(self, *_):
        pass

    def record_exception(self, *_):
        pass


class _NoopTracer:
    def start_as_current_span(self, *_, **__):
        return _NoopSpan()


# ── State ─────────────────────────────────────────────────────────────────────


@dataclass
class PipelineState:
    task: str
    spec: str = ""
    plan: str = ""
    code: str = ""
    validation: str = ""
    status: Literal["pending", "running", "success", "needs_revision", "failed"] = (
        "pending"
    )
    revision_count: int = 0
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    messages: list[dict] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


# ── System prompts ────────────────────────────────────────────────────────────

_ARCHITECT_SYSTEM = """You are the Architect agent in an AgentSmith pipeline.
Your role is to produce a precise, step-by-step implementation blueprint for the
Developer agent. Rules:
- Reference exact file paths that need to be created or modified.
- Number every step.
- Map each step to a requirement in the spec.
- Output raw structured text only — no conversational padding.
- Flag any ambiguity as a numbered OPEN ITEM at the end."""

_DEVELOPER_SYSTEM = """You are the Developer agent in an AgentSmith pipeline.
Your role is to produce clean, production-ready code from the Architect's blueprint. Rules:
- Output code only. No explanations outside inline comments.
- Handle all error paths explicitly — no empty catch blocks, no bare except clauses.
- Every function must have a typed signature.
- Follow every pillar in .cursorrules exactly."""

_VALIDATOR_SYSTEM = """You are the Validator agent in an AgentSmith pipeline.
Your role is to review the Developer's code against the Architect's blueprint and the
original spec. You must respond with a JSON object only:
{
  "verdict": "PASS" | "FAIL",
  "issues": ["<issue 1>", "<issue 2>"],
  "revision_hint": "<what the Developer must fix, or empty string if PASS>"
}"""


# ── Agent nodes ───────────────────────────────────────────────────────────────


def _architect_node(state: PipelineState, tracer: Any, logger: Any) -> PipelineState:
    from cost_router import call as llm_call

    prompt = f"""SPEC:\n{state.spec}\n\nTASK:\n{state.task}"""
    if state.revision_count > 0:
        prompt += f"\n\nPREVIOUS VALIDATION ISSUES:\n{state.validation}"

    with tracer.start_as_current_span("architect") as span:
        span.set_attribute("agent.name", "Architect")
        span.set_attribute("agent.role", "orchestrator")
        span.set_attribute(
            "agent.owner_id", os.environ.get("AGENT_OWNER_ID", "unknown")
        )
        span.set_attribute("input.value", prompt[:1000])
        try:
            plan = llm_call(prompt, system=_ARCHITECT_SYSTEM, task_type="architect")
            state.plan = plan
            span.set_attribute("output.value", plan[:1000])
            logger.info("architect_plan_generated", tokens=len(plan.split()))
        except Exception as exc:
            logger.major("architect_failed", error=str(exc))
            span.record_exception(exc)
            state.status = "failed"
    return state


def _developer_node(state: PipelineState, tracer: Any, logger: Any) -> PipelineState:
    from cost_router import call as llm_call

    prompt = f"""BLUEPRINT:\n{state.plan}\n\nORIGINAL TASK:\n{state.task}"""
    if state.revision_count > 0:
        prompt += f"\n\nREVISION REQUIRED:\n{state.validation}"

    with tracer.start_as_current_span("developer") as span:
        span.set_attribute("agent.name", "Developer")
        span.set_attribute("agent.role", "subagent")
        span.set_attribute(
            "agent.owner_id", os.environ.get("AGENT_OWNER_ID", "unknown")
        )
        span.set_attribute("input.value", prompt[:1000])
        try:
            code = llm_call(prompt, system=_DEVELOPER_SYSTEM, task_type="code")
            state.code = code
            span.set_attribute("output.value", code[:1000])
            logger.info("developer_code_generated", tokens=len(code.split()))
        except Exception as exc:
            logger.major("developer_failed", error=str(exc))
            span.record_exception(exc)
            state.status = "failed"
    return state


def _validator_node(state: PipelineState, tracer: Any, logger: Any) -> PipelineState:
    from cost_router import call as llm_call

    prompt = f"SPEC:\n{state.spec}\n\nBLUEPRINT:\n{state.plan}\n\nCODE:\n{state.code}"

    with tracer.start_as_current_span("validator") as span:
        span.set_attribute("agent.name", "Validator")
        span.set_attribute("agent.role", "subagent")
        span.set_attribute(
            "agent.owner_id", os.environ.get("AGENT_OWNER_ID", "unknown")
        )
        span.set_attribute("input.value", prompt[:1000])
        try:
            raw = llm_call(prompt, system=_VALIDATOR_SYSTEM, task_type="review")
            # Extract JSON from response (model may wrap in markdown)
            import re

            m = re.search(r"\{.*\}", raw, re.DOTALL)
            verdict_json = (
                json.loads(m.group(0))
                if m
                else {"verdict": "FAIL", "issues": [raw], "revision_hint": raw}
            )

            state.validation = json.dumps(verdict_json, indent=2)
            span.set_attribute("output.value", state.validation)

            if verdict_json.get("verdict") == "PASS":
                state.status = "success"
                logger.info("validator_passed", revision=state.revision_count)
            else:
                state.status = "needs_revision"
                state.revision_count += 1
                logger.minor(
                    "validator_revision_requested",
                    revision=state.revision_count,
                    issues=verdict_json.get("issues"),
                )
        except Exception as exc:
            logger.major("validator_failed", error=str(exc))
            span.record_exception(exc)
            state.status = "failed"
    return state


# ── Pipeline orchestrator ─────────────────────────────────────────────────────

MAX_REVISIONS = 2


def run_pipeline(
    task: str,
    spec_file: Optional[str] = None,
    spec_text: Optional[str] = None,
    project_name: Optional[str] = None,
) -> dict[str, Any]:
    """
    Run the Architect → Developer → Validator pipeline.

    Args:
        task:         Natural-language task description.
        spec_file:    Path to the RFC markdown spec file.
        spec_text:    Inline spec text (used if spec_file not provided).
        project_name: Overrides auto-detected project name.

    Returns:
        Dict with keys: status, code, plan, validation, session_id, revisions.
    """
    from agent_logger import AgentLogger

    session_id = str(uuid.uuid4())
    pname = project_name or Path.cwd().name

    logger = AgentLogger("Supervisor", agent_role="orchestrator", session_id=session_id)
    tracer = _setup_otel(pname, session_id)

    # Load spec
    spec = spec_text or ""
    if spec_file and Path(spec_file).exists():
        spec = Path(spec_file).read_text(encoding="utf-8")

    state = PipelineState(task=task, spec=spec, session_id=session_id)
    state.status = "running"
    logger.info("pipeline_start", task=task[:200], spec_chars=len(spec))

    with tracer.start_as_current_span("pipeline") as root_span:
        root_span.set_attribute("project.name", pname)
        root_span.set_attribute("agent.session_id", session_id)
        root_span.set_attribute(
            "agent.owner_id", os.environ.get("AGENT_OWNER_ID", "unknown")
        )
        if logger.tenant_id:
            root_span.set_attribute("tenant.id", logger.tenant_id)

        while state.status in ("running", "needs_revision"):
            if state.revision_count > MAX_REVISIONS:
                logger.major(
                    "max_revisions_exceeded",
                    revisions=state.revision_count,
                    last_validation=state.validation,
                )
                state.status = "failed"
                break

            state = _architect_node(state, tracer, logger)
            if state.status == "failed":
                break

            state = _developer_node(state, tracer, logger)
            if state.status == "failed":
                break

            state = _validator_node(state, tracer, logger)

            if (
                state.status == "needs_revision"
                and state.revision_count > MAX_REVISIONS
            ):
                logger.major(
                    "max_revisions_exceeded",
                    revisions=state.revision_count,
                    last_validation=state.validation,
                )
                state.status = "failed"
                break

    logger.info(
        "pipeline_end",
        status=state.status,
        revisions=state.revision_count,
    )

    return {
        "status": state.status,
        "session_id": state.session_id,
        "revisions": state.revision_count,
        "plan": state.plan,
        "code": state.code,
        "validation": state.validation,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run local agent pipeline")
    parser.add_argument("task", help="Task description")
    parser.add_argument("--spec", help="Path to .agent-rfc/ spec file")
    parser.add_argument("--project", help="Project name override")
    args = parser.parse_args()

    result = run_pipeline(
        task=args.task,
        spec_file=args.spec,
        project_name=args.project,
    )
    print(json.dumps({k: v for k, v in result.items() if k != "code"}, indent=2))
    if result.get("code"):
        print("\n--- GENERATED CODE ---")
        print(result["code"])
