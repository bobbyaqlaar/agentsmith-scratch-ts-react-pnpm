"""
scripts/gate_kg.py — what a change touches, read from the graph
(.agent-rfc/designs/governance-enforcement.md, G4).

Kept apart from local_knowledge_graph.py on purpose: that module reaches for
networkx and the scripts/ helpers, and the process gate runs in every tenant's
git hooks with pydantic and OpenTelemetry and nothing else
(scripts/requirements-gate.txt). This reads the node-link JSON directly, so the
gate and the CLI share one definition of "the scope of this change" without the
gate paying for the rest.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Iterable, NamedTuple, Optional, Sequence

# ── Impact: what a change touches, and the scope a review covers ─────────────
#
# Read straight from the node-link JSON, with no networkx: every git hook in
# every tenant would otherwise pay for the import, and the gate's own
# dependency list (scripts/requirements-gate.txt) is pydantic and OpenTelemetry
# on purpose. The CLI and the process gate call this same function
# (.agent-rfc/designs/governance-enforcement.md, G4).


# Which review-lever groups a kind of file pulls in. A route is a screen and a
# session; a gate script is safety and signal integrity. Derived from what the
# change touches, so a reviewer is pointed at the groups that apply rather than
# reading all seven every time.
_GROUPS = (
    ("portal/app/api/", (2, 6, 7)),
    ("app/api/", (2, 6, 7)),
    ("/routes/", (2, 6, 7)),
    ("portal/", (5, 7)),
    (".tsx", (5, 7)),
    ("scripts/process_gate", (2, 6)),
    ("scripts/gate_", (2, 6)),
    (".githooks/", (2, 6)),
    ("runtime/", (2, 3)),
    ("scripts/", (2, 4)),
    ("workflow-templates/", (4, 6)),
    (".github/workflows/", (4, 6)),
)


class Impact(NamedTuple):
    """The scope of one change: what to read, which levers, and a name for it."""

    files: list
    groups: list
    unknown: list
    query: str


def _dependents(graph: dict, path: str) -> Iterable[str]:
    for link in graph.get("links") or graph.get("edges") or []:
        if link.get("edge_type") == "IMPORTS" and link.get("target") == path:
            yield link["source"]


def impact(graph: dict, changed: Sequence[str]) -> Impact:
    """The files a reviewer must read for this change, and the hash naming them.

    One hop of dependents, not the transitive closure: two hops out from a
    shared helper is most of the repo, and a scope nobody can read is one
    nobody does.
    """
    known = {node["id"] for node in graph.get("nodes") or []}
    files = set(changed)
    for path in changed:
        files.update(_dependents(graph, path))
    groups = set()
    for path in files:
        for marker, applies in _GROUPS:
            if marker in path:
                groups.update(applies)
                break
    ordered = sorted(files)
    # The SCOPE, not the diff's bytes: a content hash would go stale on the next
    # keystroke and turn the sign-off line into something pasted unread.
    digest = hashlib.sha256("\n".join(ordered).encode("utf-8")).hexdigest()[:12]
    return Impact(files=ordered, groups=sorted(groups),
                  unknown=sorted(f for f in changed if f not in known), query=f"kg:{digest}")


def changed_files(base: str, root: Optional[Path] = None) -> list:
    """What `git diff --name-only <base>` says this change touches."""
    result = subprocess.run(["git", "diff", "--name-only", base], cwd=root,
                            capture_output=True, text=True, check=False)
    return [line for line in result.stdout.splitlines() if line]


def render_impact(found: Impact) -> str:
    lines = [f"Files to read ({len(found.files)}):"]
    lines += [f"  - {path}" for path in found.files]
    if found.unknown:
        lines.append("Not in the graph yet (new files): " + ", ".join(found.unknown))
    lines.append("Lever groups that apply: "
                 + (", ".join(str(group) for group in found.groups) or "none derived"))
    lines.append(f"KG query: {found.query}")
    return "\n".join(lines)
