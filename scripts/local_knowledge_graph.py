"""
local_knowledge_graph.py — Per-repo NetworkX knowledge graph.

Node types:
  - CodebaseFile        — source file with AST-extracted symbols
  - Guardrail           — agent rule / constraint from .agent-rfc/
  - ProductionIncident  — HITL-promoted log entry (MAJOR/CRITICAL, resolved)

Edges:
  - IMPORTS             — file-to-file import relationship
  - IMPLEMENTS          — file implements guardrail
  - CAUSED_INCIDENT     — file linked to a production incident

Persisted to .agent-rfc/fixtures/knowledge_graph.json (node-link format).
Updated by map_codebase.py on every commit and checkout.

Migration note: pre-0.4 graphs stored at .agent-rfc/knowledge_graph.json.
The post-checkout hook moves the file to the fixtures path on first run.

Usage:
    from local_knowledge_graph import AgentKnowledgeGraph
    kg = AgentKnowledgeGraph()
    context = kg.fetch_subgraph_context_window("src/api/routes.py", hops=2)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, Optional

try:
    import networkx as nx
    from networkx.readwrite import json_graph as nx_json

    NX_AVAILABLE = True
except ImportError:
    NX_AVAILABLE = False

NodeType = Literal["CodebaseFile", "Guardrail", "ProductionIncident"]


# ── Helpers ───────────────────────────────────────────────────────────────────

from _shared import _iso_now, fixtures_path  # noqa: E402


def _graph_path() -> Path:
    return fixtures_path("knowledge_graph.json", mkdir=True)


# ── Core class ────────────────────────────────────────────────────────────────


class AgentKnowledgeGraph:
    """
    Per-repository knowledge graph backed by NetworkX DiGraph.

    All mutating methods auto-persist to disk.  Reads are in-memory.
    """

    def __init__(self) -> None:
        if not NX_AVAILABLE:
            raise ImportError(
                "networkx >=3.0 is required. Run: pip install 'networkx>=3.0'"
            )
        self._path = _graph_path()
        self._g: nx.DiGraph = self._load()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load(self) -> "nx.DiGraph":
        if self._path.exists():
            try:
                with self._path.open() as fh:
                    data = json.load(fh)
                return nx_json.node_link_graph(data, directed=True, multigraph=False)
            except Exception as exc:
                # Falling through to a fresh empty graph silently here would
                # mean save() later overwrites a corrupted-but-recoverable
                # file with an empty one, destroying the existing graph with
                # no signal as to why — at minimum the user should see this.
                import sys

                print(
                    f"⚠️  Knowledge Graph at {self._path} is unreadable ({exc}); starting from an empty graph.",
                    file=sys.stderr,
                )
        return nx.DiGraph()

    def save(self) -> None:
        data = nx_json.node_link_data(self._g)
        with self._path.open("w") as fh:
            json.dump(data, fh, indent=2, default=str)

    # ── Node operations ───────────────────────────────────────────────────────

    def upsert_file(
        self,
        rel_path: str,
        *,
        language: str = "unknown",
        symbols: Optional[list[str]] = None,
        last_modified: Optional[str] = None,
    ) -> None:
        """Add or update a CodebaseFile node."""
        self._g.add_node(
            rel_path,
            node_type="CodebaseFile",
            language=language,
            symbols=symbols or [],
            last_modified=last_modified or _iso_now(),
        )
        self.save()

    def upsert_guardrail(
        self,
        rule_id: str,
        title: str,
        source_file: str,
        pillar: Optional[int] = None,
    ) -> None:
        """Add or update a Guardrail node."""
        self._g.add_node(
            rule_id,
            node_type="Guardrail",
            title=title,
            source_file=source_file,
            pillar=pillar,
            created_at=self._g.nodes.get(rule_id, {}).get("created_at", _iso_now()),
        )
        self.save()

    def inject_production_learning(
        self,
        incident_id: str,
        *,
        event: str,
        agent: str,
        file_hint: Optional[str] = None,
        resolution_summary: str = "",
        resolved_by: str = "",
        resolved_at: Optional[str] = None,
    ) -> None:
        """
        Add a ProductionIncident node from a HITL-resolved log entry.
        Optionally links the incident to the related source file.
        """
        self._g.add_node(
            incident_id,
            node_type="ProductionIncident",
            event=event,
            agent=agent,
            resolution_summary=resolution_summary,
            resolved_by=resolved_by,
            resolved_at=resolved_at or _iso_now(),
        )
        if file_hint and self._g.has_node(file_hint):
            self._g.add_edge(file_hint, incident_id, edge_type="CAUSED_INCIDENT")
        self.save()

    def add_import(self, source: str, target: str) -> None:
        """Record that source file imports target file."""
        if not self._g.has_node(source):
            self.upsert_file(source)
        if not self._g.has_node(target):
            self.upsert_file(target)
        self._g.add_edge(source, target, edge_type="IMPORTS")
        self.save()

    def link_file_to_guardrail(self, file_path: str, rule_id: str) -> None:
        """Mark a file as implementing a guardrail."""
        if self._g.has_node(file_path) and self._g.has_node(rule_id):
            self._g.add_edge(file_path, rule_id, edge_type="IMPLEMENTS")
            self.save()

    def remove_file(self, rel_path: str) -> None:
        """Remove a stale CodebaseFile node (e.g. deleted file)."""
        if self._g.has_node(rel_path):
            self._g.remove_node(rel_path)
            self.save()

    # ── Query operations ──────────────────────────────────────────────────────

    def fetch_subgraph_context_window(
        self,
        anchor_path: str,
        hops: int = 2,
        max_nodes: int = 40,
    ) -> dict[str, Any]:
        """
        Return a context window dict describing nodes within `hops` edges
        of anchor_path. Used to inject relevant codebase context into an
        agent's system prompt before editing.

        Returns:
            {
                "anchor": str,
                "nodes": [{"id": ..., "node_type": ..., ...}, ...],
                "edges": [{"source": ..., "target": ..., "edge_type": ...}, ...],
                "guardrails": [...],
                "incidents": [...],
            }
        """
        if not self._g.has_node(anchor_path):
            return {
                "anchor": anchor_path,
                "nodes": [],
                "edges": [],
                "guardrails": [],
                "incidents": [],
            }

        # BFS from anchor
        reachable = nx.single_source_shortest_path_length(
            self._g.to_undirected(), anchor_path, cutoff=hops
        )
        subgraph_nodes = sorted(reachable, key=lambda n: reachable[n])[:max_nodes]
        sg = self._g.subgraph(subgraph_nodes)

        nodes_out = []
        guardrails = []
        incidents = []

        for node_id, attrs in sg.nodes(data=True):
            row = {"id": node_id, **attrs}
            nodes_out.append(row)
            if attrs.get("node_type") == "Guardrail":
                guardrails.append(row)
            elif attrs.get("node_type") == "ProductionIncident":
                incidents.append(row)

        edges_out = [{"source": s, "target": t, **d} for s, t, d in sg.edges(data=True)]

        return {
            "anchor": anchor_path,
            "nodes": nodes_out,
            "edges": edges_out,
            "guardrails": guardrails,
            "incidents": incidents,
        }

    def find_files_by_symbol(self, symbol: str) -> list[str]:
        """Return all file paths that export or define the given symbol name."""
        return [
            node_id
            for node_id, attrs in self._g.nodes(data=True)
            if attrs.get("node_type") == "CodebaseFile"
            and symbol in attrs.get("symbols", [])
        ]

    def impacted_files(self, rel_path: str) -> list[str]:
        """Return files that import rel_path (direct reverse-dependency lookup)."""
        return [
            src
            for src, tgt, data in self._g.in_edges(rel_path, data=True)
            if data.get("edge_type") == "IMPORTS"
        ]

    def stats(self) -> dict[str, int]:
        counts: dict[str, int] = {
            "total": self._g.number_of_nodes(),
            "edges": self._g.number_of_edges(),
        }
        for nt in ("CodebaseFile", "Guardrail", "ProductionIncident"):
            counts[nt] = sum(
                1 for _, d in self._g.nodes(data=True) if d.get("node_type") == nt
            )
        return counts

    def as_json(self) -> str:
        return json.dumps(nx_json.node_link_data(self._g), indent=2, default=str)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AgentKnowledgeGraph CLI")
    parser.add_argument("--stats", action="store_true", help="Print node/edge counts")
    parser.add_argument(
        "--context", metavar="FILE", help="Fetch subgraph context for FILE"
    )
    parser.add_argument("--symbol", metavar="SYMBOL", help="Find files defining SYMBOL")
    parser.add_argument(
        "--hops", type=int, default=2, help="Subgraph hop depth (default: 2)"
    )
    args = parser.parse_args()

    kg = AgentKnowledgeGraph()

    if args.stats:
        print(json.dumps(kg.stats(), indent=2))
    elif args.context:
        print(
            json.dumps(
                kg.fetch_subgraph_context_window(args.context, hops=args.hops),
                indent=2,
                default=str,
            )
        )
    elif args.symbol:
        files = kg.find_files_by_symbol(args.symbol)
        print(json.dumps({"symbol": args.symbol, "files": files}, indent=2))
    else:
        print(json.dumps(kg.stats(), indent=2))
