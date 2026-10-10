"""
runtime/telemetry_contract.py — what a governed application's telemetry carries
(contract/telemetry/v1/protocol.md, .agent-rfc/designs/telemetry-contract.md).

    load_catalogue()             contract/telemetry/v1/attributes.json, validated
    exports_from_bytes(...)      an OTLP body (protobuf or JSON) as OTLP/JSON dicts
    judge(exports)               the contract's checks over what was exported
    LoopbackCollector()          an OTLP/HTTP receiver on 127.0.0.1, for `--emitter`

A wire contract: a tenant emits OTLP and runs nothing of a provider's. The
catalogue is the published shape; this module reads exported telemetry — never
the code that produced it — so an emitter on plain OpenTelemetry is judged
exactly as the runtime library is.
"""

from __future__ import annotations

import gzip
import json
import os
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

TELEMETRY_CONTRACT = 1
CONTRACT_ATTRIBUTE = "governance.telemetry.contract"
# A body larger than this is refused, and so is a run that sends more in all:
# the receiver reads bodies it did not write.
MAX_BODY_BYTES = 8_000_000
MAX_TOTAL_BYTES = 64_000_000
# An emitter's own destinations and the collector credential, removed from its
# environment for a loopback run so nothing it emits leaves the machine.
OTLP_DESTINATIONS = ("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "OTEL_EXPORTER_OTLP_METRICS_ENDPOINT",
                     "OTEL_EXPORTER_OTLP_LOGS_ENDPOINT", "AGENT_PHOENIX_ENDPOINT", "OTEL_EXPORTER_OTLP_HEADERS")

Where = Literal["resource", "span", "span_name", "event", "metric", "metric_attribute"]
ValueType = Literal["string", "int", "double", "number", "bool", "string[]"]


class CatalogueEntry(BaseModel):
    """One name the contract defines, or one family of names (`family`)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(pattern=r"^[a-z][a-z0-9_]*(\.[a-z0-9_]+)*\.?$")
    where: Where
    family: bool = False
    # A family whose members the emitter names itself (`agent_span(**kw)` writes
    # a tenant's own keys under `agent.`); a closed family's members are fixed by
    # the emitter that writes them (`llm.gateway.input_guardrail.<kind>`).
    open: bool = False
    type: Optional[ValueType] = None
    requirement: Literal["required", "required_in_run", "conditional", "optional"] = "optional"
    # `required` on a span applies only to spans whose name starts with this.
    applies_to: Optional[str] = None
    # `conditional`: present exactly when every attribute named here has that value.
    when: dict[str, Any] = Field(default_factory=dict)
    payload: bool = False
    instrument: Optional[Literal["counter", "histogram", "gauge", "up_down_counter"]] = None
    unit: Optional[str] = None
    values: list[str] = Field(default_factory=list)
    since: Optional[str] = Field(default=None, pattern=r"^\d+\.\d+\.\d+$")
    meaning: str = Field(min_length=1)

    @model_validator(mode="after")
    def _consistent(self) -> "CatalogueEntry":
        if self.family != self.name.endswith("."):
            raise ValueError(f"{self.name}: a family's name ends with '.', and only a family's")
        if self.open and not self.family:
            raise ValueError(f"{self.name}: only a family is open")
        if (self.requirement == "conditional") != bool(self.when):
            raise ValueError(f"{self.name}: `when` goes with a conditional requirement, and only with one")
        if self.where == "metric" and (self.instrument is None or self.unit is None):
            raise ValueError(f"{self.name}: an instrument names its kind and unit")
        if self.where in ("resource", "span", "metric_attribute") and self.type is None and not self.family:
            raise ValueError(f"{self.name}: an attribute names its type")
        return self


class Catalogue(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    contract: Literal[1]
    entries: list[CatalogueEntry] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique(self) -> "Catalogue":
        seen: set[tuple[str, str]] = set()
        for entry in self.entries:
            key = (entry.where, entry.name)
            if key in seen:
                raise ValueError(f"{entry.where} {entry.name} is catalogued twice")
            seen.add(key)
        return self

    def lookup(self, where: str, name: str) -> Optional[CatalogueEntry]:
        """The exact entry for `name`, else the longest family that holds it."""
        exact = next((e for e in self.entries if e.where == where and e.name == name and not e.family), None)
        if exact is not None:
            return exact
        families = [e for e in self.entries if e.where == where and e.family and name.startswith(e.name)]
        if not families:
            return None
        return max(families, key=lambda e: len(e.name))

    def of(self, where: str) -> list[CatalogueEntry]:
        return [e for e in self.entries if e.where == where]


def contract_dir() -> Path:
    for root in (Path(os.environ["AGENTSMITH_DIR"]) if os.environ.get("AGENTSMITH_DIR") else None,
                 Path(__file__).resolve().parent.parent, Path.home() / ".agent-framework"):
        here = root / "contract" / "telemetry" / f"v{TELEMETRY_CONTRACT}" if root is not None else None
        if here is not None and (here / "attributes.json").is_file():
            return here
    raise FileNotFoundError(f"contract/telemetry/v{TELEMETRY_CONTRACT}/ not found in $AGENTSMITH_DIR, beside this "
                            "package, or ~/.agent-framework — re-run install-ai-stack.sh")


def load_catalogue(directory: Optional[Path] = None) -> Catalogue:
    path = (directory or contract_dir()) / "attributes.json"
    return Catalogue.model_validate_json(path.read_text(encoding="utf-8"))


# ── Reading OTLP ─────────────────────────────────────────────────────────────


def exports_from_bytes(signal: str, body: bytes, content_type: str = "", encoding: str = "") -> dict:
    """One OTLP/HTTP request body as an OTLP/JSON dict. Raises ValueError on a
    body that is not one — the caller reports it, never crashes on it."""
    if encoding.strip().lower() == "gzip":
        try:
            body = gzip.decompress(body)
        except OSError as exc:
            raise ValueError(f"a gzip body that does not decompress ({exc})") from exc
    if "json" in content_type.lower():
        try:
            data = json.loads(body)
        except ValueError as exc:
            raise ValueError(f"a JSON body that does not parse ({exc})") from exc
        if not isinstance(data, dict):
            raise ValueError("an OTLP/JSON body must be an object")
        return data
    try:
        from google.protobuf.json_format import MessageToDict
        from opentelemetry.proto.collector.logs.v1.logs_service_pb2 import ExportLogsServiceRequest
        from opentelemetry.proto.collector.metrics.v1.metrics_service_pb2 import ExportMetricsServiceRequest
        from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
    except ImportError as exc:  # the exporter package brings these; say what is missing
        raise ValueError(f"cannot decode OTLP protobuf here ({exc}) — send OTLP/JSON") from exc
    message = {"traces": ExportTraceServiceRequest, "metrics": ExportMetricsServiceRequest,
               "logs": ExportLogsServiceRequest}[signal]()
    try:
        message.ParseFromString(body)
    except Exception as exc:  # protobuf raises its own DecodeError
        raise ValueError(f"a protobuf body that does not decode ({type(exc).__name__})") from exc
    return MessageToDict(message)


def exports_from_file(path: Path) -> list[dict]:
    """An OTLP/JSON export: one object, or one per line (a collector's file exporter)."""
    text = path.read_text(encoding="utf-8")
    try:
        whole = json.loads(text)
        return [whole] if isinstance(whole, dict) else list(whole)
    except ValueError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]


def _value(raw: dict) -> tuple[str, Any]:
    """An OTLP AnyValue as (type, python value)."""
    if "stringValue" in raw:
        return "string", raw["stringValue"]
    if "boolValue" in raw:
        return "bool", bool(raw["boolValue"])
    if "intValue" in raw:
        return "int", int(raw["intValue"])
    if "doubleValue" in raw:
        return "double", float(raw["doubleValue"])
    if "arrayValue" in raw:
        items = [_value(v) for v in (raw["arrayValue"].get("values") or [])]
        kinds = {kind for kind, _ in items}
        return (f"{kinds.pop()}[]" if len(kinds) == 1 else "array"), [v for _, v in items]
    return "unknown", None


def _attributes(raw: Optional[list]) -> dict[str, tuple[str, Any]]:
    return {a["key"]: _value(a.get("value") or {}) for a in raw or [] if isinstance(a, dict) and "key" in a}


@dataclass
class Span:
    name: str
    span_id: str
    parent_id: str
    attributes: dict[str, tuple[str, Any]]
    events: list[str] = field(default_factory=list)


@dataclass
class Metric:
    name: str
    unit: str
    instrument: str
    points: list[dict[str, tuple[str, Any]]]


@dataclass
class Emitted:
    resource: dict[str, tuple[str, Any]]
    spans: list[Span] = field(default_factory=list)
    metrics: list[Metric] = field(default_factory=list)


_INSTRUMENTS = {"sum": "counter", "histogram": "histogram", "gauge": "gauge",
                "exponentialHistogram": "histogram"}


def flatten(exports: list[dict]) -> list[Emitted]:
    """OTLP/JSON dicts as one `Emitted` per Resource."""
    out: list[Emitted] = []
    for export in exports:
        for block in export.get("resourceSpans") or []:
            emitted = Emitted(_attributes((block.get("resource") or {}).get("attributes")))
            for scope in block.get("scopeSpans") or []:
                for raw in scope.get("spans") or []:
                    emitted.spans.append(Span(
                        name=str(raw.get("name", "")), span_id=str(raw.get("spanId", "")),
                        parent_id=str(raw.get("parentSpanId", "") or ""),
                        attributes=_attributes(raw.get("attributes")),
                        events=[str(e.get("name", "")) for e in raw.get("events") or []]))
            out.append(emitted)
        for block in export.get("resourceMetrics") or []:
            emitted = Emitted(_attributes((block.get("resource") or {}).get("attributes")))
            for scope in block.get("scopeMetrics") or []:
                for raw in scope.get("metrics") or []:
                    kind = next((k for k in _INSTRUMENTS if k in raw), "")
                    instrument = _INSTRUMENTS.get(kind, "unknown")
                    if kind == "sum" and not raw["sum"].get("isMonotonic", False):
                        instrument = "up_down_counter"
                    points = [_attributes(p.get("attributes")) for p in (raw.get(kind) or {}).get("dataPoints") or []]
                    emitted.metrics.append(Metric(str(raw.get("name", "")), str(raw.get("unit", "")), instrument,
                                                  points))
            out.append(emitted)
    return out


# ── Judging ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    why: str = ""


@dataclass
class Judgement:
    checks: list[Check]
    notes: list[str]

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(c.ok for c in self.checks)


def _typed(expected: Optional[str], actual: str) -> bool:
    if expected is None:
        return True
    if expected == "number":
        return actual in ("int", "double")
    return expected == actual


def _in_runs(spans: list[Span]) -> set[str]:
    """Span ids inside a run: carrying `run.id`, or descended from one that does."""
    by_id = {s.span_id: s for s in spans}
    inside: set[str] = set()
    for span in spans:
        node: Optional[Span] = span
        seen: set[str] = set()
        while node is not None and node.span_id not in seen:
            seen.add(node.span_id)
            if "run.id" in node.attributes:
                inside.add(span.span_id)
                break
            node = by_id.get(node.parent_id)
    return inside


def _holds(when: dict[str, Any], attributes: dict[str, tuple[str, Any]]) -> bool:
    return all(name in attributes and attributes[name][1] == value for name, value in when.items())


def _first(problems: list[str], limit: int = 4) -> str:
    more = f" (and {len(problems) - limit} more)" if len(problems) > limit else ""
    return "; ".join(problems[:limit]) + more


def judge(exports: list[dict], catalogue: Optional[Catalogue] = None) -> Judgement:
    """Every check of contract/telemetry/v1/protocol.md over `exports`."""
    catalogue = catalogue or load_catalogue()
    emitted = flatten(exports)
    spans = [s for e in emitted for s in e.spans]
    metrics = [m for e in emitted for m in e.metrics]
    checks: list[Check] = []
    notes: list[str] = []

    checks.append(Check("something was exported", bool(spans or metrics),
                        "" if spans or metrics else "no span and no metric arrived — nothing was judged"))

    resources = [e.resource for e in emitted]
    contract_problems = []
    for resource in resources:
        found = resource.get(CONTRACT_ATTRIBUTE)
        if found is None:
            contract_problems.append(f"a Resource without {CONTRACT_ATTRIBUTE} — a pre-contract emitter")
        elif found[1] != TELEMETRY_CONTRACT:
            contract_problems.append(f"a Resource claiming {CONTRACT_ATTRIBUTE}={found[1]!r}, not {TELEMETRY_CONTRACT}")
    checks.append(Check("the Resource names the contract it speaks", not contract_problems,
                        _first(sorted(set(contract_problems)))))

    # The contract attribute is judged on its own, above: a pre-contract emitter
    # is one finding, not two.
    required = [e for e in catalogue.of("resource") if e.requirement == "required" and e.name != CONTRACT_ATTRIBUTE]
    missing = sorted({e.name for resource in resources for e in required if e.name not in resource})
    checks.append(Check("the Resource carries every required attribute", not missing,
                        f"missing: {', '.join(missing)}" if missing else ""))

    inside = _in_runs(spans)
    identity = [e for e in catalogue.of("span") if e.requirement == "required_in_run"]
    unidentified = [f"{s.name} lacks {e.name}" for s in spans if s.span_id in inside
                    for e in identity if e.name not in s.attributes]
    checks.append(Check("every span inside a run carries the run's identity", not unidentified,
                        _first(unidentified)))

    kinds = [e for e in catalogue.of("span") if e.requirement == "required" and e.applies_to]
    incomplete = [f"{s.name} lacks {e.name}" for s in spans for e in kinds
                  if s.name.startswith(e.applies_to or "") and e.name not in s.attributes]
    checks.append(Check("every span carries what its kind requires", not incomplete, _first(incomplete)))

    conditional = [e for e in catalogue.of("span") if e.requirement == "conditional"]
    wrong_presence = []
    for span in spans:
        for entry in conditional:
            if entry.applies_to and not span.name.startswith(entry.applies_to):
                continue
            holds, present = _holds(entry.when, span.attributes), entry.name in span.attributes
            if holds and not present:
                wrong_presence.append(f"{span.name} lacks {entry.name} though {entry.when} holds")
            elif present and not holds:
                wrong_presence.append(f"{span.name} carries {entry.name} without {entry.when}")
    checks.append(Check("conditional attributes appear exactly when their condition holds", not wrong_presence,
                        _first(wrong_presence)))

    mistyped, unknown, payloads = [], set(), set()
    for where, items in (("resource", resources), ("span", [s.attributes for s in spans]),
                         ("metric_attribute", [p for m in metrics for p in m.points])):
        for attributes in items:
            for name, (kind, _value_) in attributes.items():
                known = catalogue.lookup(where, name)
                if known is None:
                    unknown.add(f"{where} {name}")
                    continue
                if not _typed(known.type, kind):
                    mistyped.append(f"{where} {name} is {kind}, catalogued {known.type}")
                if known.values and kind == "string" and _value_ not in known.values:
                    mistyped.append(f"{where} {name}={_value_!r}, not one of {', '.join(known.values)}")
                if known.payload:
                    payloads.add(name)
    checks.append(Check("every catalogued attribute has its catalogued type", not mistyped,
                        _first(sorted(set(mistyped)))))

    wrong_instruments = []
    for metric in metrics:
        instrument = catalogue.lookup("metric", metric.name)
        if instrument is None:
            unknown.add(f"metric {metric.name}")
            continue
        if instrument.instrument != metric.instrument or (instrument.unit is not None
                                                          and instrument.unit != metric.unit):
            wrong_instruments.append(f"{metric.name} is a {metric.instrument} in {metric.unit or 'no unit'}, "
                                     f"catalogued a {instrument.instrument} in {instrument.unit}")
    checks.append(Check("every catalogued instrument has its kind and unit", not wrong_instruments,
                        _first(wrong_instruments)))

    for span in spans:
        if catalogue.lookup("span_name", span.name) is None:
            unknown.add(f"span name {span.name}")
    if unknown:
        notes.append(f"not in the catalogue (allowed, ignored by readers): {_first(sorted(unknown), 8)}")
    if payloads:
        notes.append(f"payload attributes emitted — their redaction is C7's to verify: {', '.join(sorted(payloads))}")
    notes.append(f"judged {len(spans)} span(s), {len(inside)} inside a run, and {len(metrics)} metric(s) "
                 f"under {len(resources)} Resource(s)")
    return Judgement(checks, notes)


# ── A receiver for `--emitter` ───────────────────────────────────────────────


class LoopbackCollector:
    """OTLP/HTTP on 127.0.0.1, holding what it is sent. Every body is untrusted:
    size-capped, decoded defensively, and a body that does not decode is kept
    as a problem rather than raised."""

    def __init__(self) -> None:
        self.exports: list[dict] = []
        self.problems: list[str] = []
        self._total = 0
        self._lock = threading.Lock()
        collector = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_: Any) -> None:  # quiet: the report says what matters
                return None

            def do_POST(self) -> None:
                signal = {"/v1/traces": "traces", "/v1/metrics": "metrics", "/v1/logs": "logs"}.get(self.path)
                length = int(self.headers.get("Content-Length") or 0)
                if signal is None or length <= 0 or length > MAX_BODY_BYTES:
                    self.send_response(404 if signal is None else 413)
                    self.end_headers()
                    return
                body = self.rfile.read(length)
                collector._accept(signal, body, self.headers.get("Content-Type", ""),
                                  self.headers.get("Content-Encoding", ""))
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b"{}")

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def endpoint(self) -> str:
        return f"http://127.0.0.1:{self._server.server_address[1]}"

    def _accept(self, signal: str, body: bytes, content_type: str, encoding: str) -> None:
        with self._lock:
            self._total += len(body)
            if self._total > MAX_TOTAL_BYTES:
                self.problems.append(f"more than {MAX_TOTAL_BYTES} bytes sent in all — the rest was not read")
                return
            if signal == "logs":
                return
            try:
                self.exports.append(exports_from_bytes(signal, body, content_type, encoding))
            except ValueError as exc:
                self.problems.append(f"{signal}: {exc}")

    def __enter__(self) -> "LoopbackCollector":
        self._thread.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self._server.shutdown()
        self._server.server_close()
