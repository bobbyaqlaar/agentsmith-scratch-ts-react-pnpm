"""
multi_agent_system.py — LangGraph ≥0.2 multi-agent pipeline with HITL.

Graph topology:
  Architect → Developer → Validator
              ↑                ↓ (needs_revision, max 2 retries)
              └────────────────┘
                     ↓ (PASS or max retries exhausted)
                  [END]

HITL interrupt: inserted after Validator FAIL on revision 2.
All nodes emit OTel spans to Phoenix.
State is a TypedDict — no persistent DB required for dev mode.

Requires:
    langgraph>=0.2
    langchain-core
    langchain-anthropic or langchain-openai
"""

from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Literal, Optional, TypedDict

# scripts/ so `_shared` resolves however this module is loaded (invoked as a
# file, sys.path[0] is already scripts/; imported from a test, it is not).
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from runtime.environment import get_environment
except ImportError:
    # scripts/ is invoked as a file (python3 scripts/foo.py), so sys.path[0]
    # is scripts/, not the repo root. Add the ROOT — not runtime/ — so the
    # runtime imports as a package; its modules import each other that way
    # (framework G6), which a flat runtime/ path can no longer satisfy.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from runtime.environment import get_environment

# Guard import
try:
    from langgraph.graph import StateGraph, END
    from langgraph.checkpoint.memory import MemorySaver

    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False

# ── State definition ──────────────────────────────────────────────────────────


class AgentState(TypedDict):
    task: str
    spec: str
    plan: str
    code: str
    validation: str
    verdict: Literal["PASS", "FAIL", "PENDING"]
    revision_count: int
    status: Literal["running", "success", "needs_revision", "failed", "hitl_pending"]
    session_id: str
    owner_id: str
    project: str
    issues: list[str]
    revision_hint: str
    hitl_approved: bool


def _default_state(task: str, spec: str, project: str) -> AgentState:
    return AgentState(
        task=task,
        spec=spec,
        plan="",
        code="",
        validation="",
        verdict="PENDING",
        revision_count=0,
        status="running",
        session_id=str(uuid.uuid4()),
        owner_id=os.environ.get("AGENT_OWNER_ID", "unknown"),
        project=project,
        issues=[],
        revision_hint="",
        hitl_approved=False,
    )


# ── OTel tracer ───────────────────────────────────────────────────────────────


def _get_tracer(project: str, session_id: str) -> Any:
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.sdk.resources import Resource

        # See local_agent_stack._setup_otel — one resolver, four callers. This
        # copy read only AGENT_PHOENIX_ENDPOINT and defaulted to localhost, so
        # it also ignored OTEL_EXPORTER_OTLP_TRACES_ENDPOINT entirely.
        from runtime.otlp import span_exporter

        from agent_logger import _tenant_id

        resource_attrs = {
            "service.name": project,
            "project.name": project,
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
        return trace.get_tracer("agenticframework.langgraph")
    except Exception:

        class _Noop:
            def start_as_current_span(self, *a, **k):
                class _Span:
                    def __enter__(self):
                        return self

                    def __exit__(self, *_):
                        pass

                    def set_attribute(self, *_):
                        pass

                    def record_exception(self, *_):
                        pass

                return _Span()

        return _Noop()


# ── LangChain model factory ───────────────────────────────────────────────────


def _get_model(role: Literal["architect", "developer", "validator"]) -> Any:
    """
    Return a LangChain chat model appropriate for the given role.
    Falls back through Anthropic → OpenAI → local Ollama.
    """
    # Same resolution order as cost_router.py: env override → the matching
    # models.yaml role → literal fallback (scripts-only install). These were
    # hardcoded to claude-3-5-sonnet-20241022 / gpt-4o / llama3 — ids that had
    # drifted from both models.yaml AND from cost_router's own copy of the
    # same table, so three places named three different architect models.
    from _shared import role_model

    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    architect_model = os.environ.get("AGENT_MODEL_ARCHITECT", "").strip() or role_model(
        "architect", "qwen2.5"
    )
    complex_model = os.environ.get("AGENT_MODEL_COMPLEX", "").strip() or role_model(
        "developer", "llama3.2:3b"
    )
    local_model = os.environ.get("AGENT_MODEL_LOCAL", "").strip() or role_model(
        "fast", "smollm2"
    )

    if role == "architect" and anthropic_key:
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=architect_model, api_key=anthropic_key, temperature=0.2
        )
    elif role in ("developer", "validator") and openai_key:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=complex_model, api_key=openai_key, temperature=0.1)
    else:
        # Local Ollama via OpenAI-compatible API
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=local_model,
            base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
            api_key="ollama",
            temperature=0.1,
        )


# ── Node implementations ──────────────────────────────────────────────────────

_ARCHITECT_PROMPT = """You are the Architect agent. Produce a numbered step-by-step
implementation blueprint from the spec and task below.
Map each step to a spec requirement. Flag ambiguities as OPEN ITEMs.
Output structured text only — no conversational padding.

SPEC:
{spec}

TASK:
{task}
{revision_section}"""

_DEVELOPER_PROMPT = """You are the Developer agent. Produce clean, production-ready code
from the blueprint below. Output code only — no explanations except inline comments.
Handle all error paths explicitly. All functions must have typed signatures.

BLUEPRINT:
{plan}

TASK:
{task}
{revision_section}"""

_VALIDATOR_PROMPT = """You are the Validator agent. Review the code against the blueprint
and spec. Respond with a JSON object ONLY:
{{
  "verdict": "PASS" | "FAIL",
  "issues": ["<issue>"],
  "revision_hint": "<what Developer must fix, or empty string if PASS>"
}}

SPEC:
{spec}

BLUEPRINT:
{plan}

CODE:
{code}"""


def _run_llm(model: Any, prompt: str) -> str:
    from langchain_core.messages import HumanMessage

    response = model.invoke([HumanMessage(content=prompt)])
    return response.content


def architect_node(state: AgentState) -> AgentState:
    from agent_logger import AgentLogger

    logger = AgentLogger("Architect", "orchestrator", session_id=state["session_id"])

    revision_section = ""
    if state["revision_count"] > 0:
        revision_section = f"\n\nPREVIOUS VALIDATION ISSUES:\n{state['validation']}"

    prompt = _ARCHITECT_PROMPT.format(
        spec=state["spec"],
        task=state["task"],
        revision_section=revision_section,
    )

    model = _get_model("architect")
    try:
        plan = _run_llm(model, prompt)
        logger.info("architect_plan_generated", tokens=len(plan.split()))
        return {**state, "plan": plan}
    except Exception as exc:
        logger.major("architect_failed", error=str(exc))
        return {**state, "status": "failed"}


def developer_node(state: AgentState) -> AgentState:
    from agent_logger import AgentLogger

    logger = AgentLogger(
        "Developer",
        "subagent",
        orchestrator="Architect",
        session_id=state["session_id"],
    )

    revision_section = ""
    if state["revision_count"] > 0:
        revision_section = f"\n\nREVISION REQUIRED:\n{state['validation']}"

    prompt = _DEVELOPER_PROMPT.format(
        plan=state["plan"],
        task=state["task"],
        revision_section=revision_section,
    )

    model = _get_model("developer")
    try:
        code = _run_llm(model, prompt)
        logger.info("developer_code_generated", tokens=len(code.split()))
        return {**state, "code": code}
    except Exception as exc:
        logger.major("developer_failed", error=str(exc))
        return {**state, "status": "failed"}


def validator_node(state: AgentState) -> AgentState:
    from agent_logger import AgentLogger

    logger = AgentLogger(
        "Validator",
        "subagent",
        orchestrator="Architect",
        session_id=state["session_id"],
    )

    prompt = _VALIDATOR_PROMPT.format(
        spec=state["spec"],
        plan=state["plan"],
        code=state["code"],
    )

    model = _get_model("validator")
    try:
        import re

        raw = _run_llm(model, prompt)
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        result = (
            json.loads(m.group(0))
            if m
            else {"verdict": "FAIL", "issues": [raw], "revision_hint": raw}
        )

        verdict = result.get("verdict", "FAIL")
        issues = result.get("issues", [])
        hint = result.get("revision_hint", "")
        rev_count = state["revision_count"] + (1 if verdict == "FAIL" else 0)

        new_status: Literal["success", "needs_revision", "hitl_pending"] = (
            "success"
            if verdict == "PASS"
            else "hitl_pending"
            if rev_count > 2
            else "needs_revision"
        )

        if verdict == "PASS":
            logger.info("validator_passed", revision=state["revision_count"])
        else:
            logger.minor("validator_revision_requested", issues=issues)

        return {
            **state,
            "validation": json.dumps(result, indent=2),
            "verdict": verdict,
            "issues": issues,
            "revision_hint": hint,
            "revision_count": rev_count,
            "status": new_status,
        }
    except Exception as exc:
        from agent_logger import AgentLogger

        AgentLogger("Validator", "subagent", session_id=state["session_id"]).major(
            "validator_failed", error=str(exc)
        )
        return {**state, "status": "failed"}


def hitl_node(state: AgentState) -> AgentState:
    """HITL interrupt — surface issues to the user and wait for approval."""
    from agent_logger import AgentLogger
    from notifier import notify_hitl_required

    logger = AgentLogger("HITL", "standalone", session_id=state["session_id"])
    logger.major(
        "hitl_escalation",
        issues=state["issues"],
        revision_count=state["revision_count"],
        project=state["project"],
    )
    notify_hitl_required(
        agent="Validator",
        event="max_revisions_exceeded",
        detail="\n".join(state["issues"]),
        project=state["project"],
    )
    return {**state, "status": "hitl_pending"}


# ── Edge routers ──────────────────────────────────────────────────────────────


def route_after_validator(state: AgentState) -> str:
    s = state["status"]
    if s == "success":
        return END
    elif s == "needs_revision":
        return "architect"
    elif s == "hitl_pending":
        return "hitl"
    return END


def route_after_hitl(state: AgentState) -> str:
    if state.get("hitl_approved"):
        return "developer"
    return END


# ── Checkpointer (§25, §28: MemorySaver is dev-only — prohibited in production) ─


def _get_checkpointer() -> Any:
    """
    Select a LangGraph checkpointer based on $ENVIRONMENT (resolved via the
    canonical, fail-closed runtime/environment.py:get_environment() — see
    Product_Archive.md 2.8). NOTE: "unset" now resolves to "production",
    not "development" — local/IDE runs must set ENVIRONMENT=development
    explicitly to get MemorySaver; an unset var no longer defaults to it.

    development: in-memory MemorySaver (acceptable — IDE sessions only).
    staging / production (including unset/unrecognized, fail-closed): Postgres-backed
        checkpointer, required for durability across worker restarts. Falling
        back to MemorySaver here would silently
        lose HITL pause state on crash, so it is a hard error instead.
    """
    # Uses the same canonical, fail-closed resolver as trace_redactor.py
    # (runtime/environment.py) so an unset/unrecognized $ENVIRONMENT is
    # treated identically by both — previously this defaulted missing/
    # unrecognized values to "development" independently of trace_redactor's
    # own (different) default, which could silently diverge under the exact
    # same misconfiguration (Product_Archive.md 2.8).
    environment = get_environment()
    database_url = os.environ.get("DATABASE_URL", "")

    if environment in ("staging", "production"):
        if not database_url:
            raise RuntimeError(
                f"ENVIRONMENT={environment!r} requires a Postgres checkpointer — "
                "set DATABASE_URL. MemorySaver is prohibited outside development "
                "(see SPECS.md §25, §28). Note: production agent runs should use "
                "runtime/worker.py + Temporal, not this LangGraph dev/hybrid path."
            )
        try:
            from langgraph.checkpoint.postgres import PostgresSaver
            from psycopg import Connection
            from psycopg.rows import dict_row
        except ImportError as exc:
            raise RuntimeError(
                "langgraph-checkpoint-postgres is required for staging/production. "
                "Run: pip install langgraph-checkpoint-postgres"
            ) from exc

        # NOTE: PostgresSaver.from_conn_string() is a @contextmanager generator —
        # calling __enter__() on it without ever calling __exit__() leaves nothing
        # holding a reference to that generator. Once it's garbage collected,
        # Python throws GeneratorExit into it, running its `with Connection.connect(...)`
        # cleanup and closing the connection out from under the returned saver
        # (observed as `psycopg.OperationalError: the connection is closed` on the
        # very next checkpoint read). Open the connection directly instead and keep
        # it alive for the lifetime of the saver.
        conn = Connection.connect(
            database_url, autocommit=True, prepare_threshold=0, row_factory=dict_row
        )
        saver = PostgresSaver(conn)
        saver.setup()
        return saver

    print(
        "[multi_agent_system] WARNING: using MemorySaver (in-memory checkpointer) — "
        "dev-only, state is lost on process exit. Not valid for staging/production.",
        file=sys.stderr,
    )
    return MemorySaver()


# ── Graph builder ─────────────────────────────────────────────────────────────


def build_graph(with_hitl_interrupt: bool = True) -> Any:
    if not LANGGRAPH_AVAILABLE:
        raise ImportError(
            "langgraph >=0.2 is required. Run: pip install 'langgraph>=0.2'"
        )

    builder = StateGraph(AgentState)
    builder.add_node("architect", architect_node)
    builder.add_node("developer", developer_node)
    builder.add_node("validator", validator_node)
    builder.add_node("hitl", hitl_node)

    builder.set_entry_point("architect")
    builder.add_edge("architect", "developer")
    builder.add_edge("developer", "validator")
    builder.add_conditional_edges(
        "validator",
        route_after_validator,
        {
            "architect": "architect",
            "hitl": "hitl",
            END: END,
        },
    )
    builder.add_conditional_edges(
        "hitl",
        route_after_hitl,
        {
            "developer": "developer",
            END: END,
        },
    )

    checkpointer = _get_checkpointer()
    interrupts = ["hitl"] if with_hitl_interrupt else []
    return builder.compile(checkpointer=checkpointer, interrupt_before=interrupts)


# ── Public run function ───────────────────────────────────────────────────────


def run_pipeline(
    task: str,
    spec_file: Optional[str] = None,
    spec_text: Optional[str] = None,
    project_name: Optional[str] = None,
    with_hitl: bool = True,
) -> dict[str, Any]:
    """
    Run the LangGraph multi-agent pipeline.

    Returns:
        Dict with: status, code, plan, validation, session_id, revisions.
    """
    if not LANGGRAPH_AVAILABLE:
        # Graceful fallback to pure Python stack
        from local_agent_stack import run_pipeline as py_pipeline

        return py_pipeline(
            task, spec_file=spec_file, spec_text=spec_text, project_name=project_name
        )

    from pathlib import Path as _Path

    pname = project_name or _Path.cwd().name
    spec = spec_text or ""
    if spec_file and _Path(spec_file).exists():
        spec = _Path(spec_file).read_text(encoding="utf-8")

    graph = build_graph(with_hitl_interrupt=with_hitl)
    state = _default_state(task=task, spec=spec, project=pname)
    config = {"configurable": {"thread_id": state["session_id"]}}

    from agent_logger import _tenant_id

    tracer = _get_tracer(pname, state["session_id"])
    with tracer.start_as_current_span("langgraph_pipeline") as root:
        root.set_attribute("project.name", pname)
        root.set_attribute("agent.session_id", state["session_id"])
        root.set_attribute("agent.owner_id", state["owner_id"])
        tenant_id = _tenant_id()
        if tenant_id:
            root.set_attribute("tenant.id", tenant_id)

        final = graph.invoke(state, config=config)

    return {
        "status": final.get("status", "unknown"),
        "session_id": final.get("session_id", ""),
        "revisions": final.get("revision_count", 0),
        "plan": final.get("plan", ""),
        "code": final.get("code", ""),
        "validation": final.get("validation", ""),
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Run LangGraph multi-agent pipeline")
    parser.add_argument("task", help="Task description")
    parser.add_argument("--spec", help="Path to RFC spec file")
    parser.add_argument("--project", help="Project name")
    parser.add_argument("--no-hitl", action="store_true", help="Disable HITL interrupt")
    args = parser.parse_args()

    result = run_pipeline(
        task=args.task,
        spec_file=args.spec,
        project_name=args.project,
        with_hitl=not args.no_hitl,
    )
    print(json.dumps({k: v for k, v in result.items() if k != "code"}, indent=2))
    if result.get("code"):
        print("\n--- GENERATED CODE ---")
        print(result["code"])
