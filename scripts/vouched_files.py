#!/usr/bin/env python3
"""
scripts/vouched_files.py — which staged files the scaffold manifest vouches for.

Prints one path per line: staged, listed in `.agenticframework/scaffold.json`,
and byte-for-byte what that manifest recorded. `hooks/pre-commit` skips its
guardrails for exactly these, because they are files AgentSmith wrote into the
tenant and the tenant cannot fix
(.agent-rfc/designs/scaffold-rfc-and-vouched-skip.md).

Two properties this must have, and both are about what it does when it cannot
answer:

  * A skip is granted only by a positive SHA-256 match. Edit a vendored file and
    the hash differs, so it is checked again like any other file.
  * Every failure prints NOTHING and exits 0 — no manifest, unreadable manifest,
    not a git repository, `git cat-file` failing. An empty list means the hook
    checks everything, which is what it did before this existed. Losing this
    script cannot disable a guardrail; it can only make the hook stricter.

Hashes the STAGED blob, not the working tree: the guardrails run over what is
about to be committed, and a file edited after `git add` must not inherit the
vouch its staged copy earned.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


def _git(*args: str, root: Path | None = None) -> bytes | None:
    try:
        done = subprocess.run(["git", *args], cwd=str(root) if root else None, capture_output=True, check=False)
    except OSError:  # fail-open: no git on PATH means no vouch, so nothing is skipped
        return None
    return done.stdout if done.returncode == 0 else None


def _staged_blobs(root: Path, paths: list[str]) -> dict[str, bytes]:
    """Each path's staged content, in one `git cat-file --batch` rather than one
    process per path: a tenant's first commit stages the whole vendored
    framework, which is 167 files."""
    if not paths:
        return {}
    try:
        done = subprocess.run(
            ["git", "cat-file", "--batch"],
            cwd=str(root),
            input="".join(f":{p}\n" for p in paths).encode("utf-8"),
            capture_output=True,
            check=False,
        )
    except OSError:
        return {}
    if done.returncode != 0:
        return {}

    out: dict[str, bytes] = {}
    buf, pos = done.stdout, 0
    for path in paths:
        end = buf.find(b"\n", pos)
        if end == -1:
            break
        header = buf[pos:end].decode("utf-8", "replace").split()
        pos = end + 1
        if len(header) != 3:  # "<oid> missing" for a path not in the index
            continue
        try:
            size = int(header[2])
        except ValueError:
            break
        out[path] = buf[pos : pos + size]
        pos += size + 1  # the newline git writes after the body
    return out


def vouched(root: Path) -> list[str]:
    try:
        manifest = json.loads((root / ".agenticframework" / "scaffold.json").read_text(encoding="utf-8"))
        recorded = manifest.get("files") or {}
        if not isinstance(recorded, dict):
            return []
    except (OSError, ValueError, AttributeError):
        # fail-open: no manifest, or not a manifest — vouch for nothing
        return []

    staged_out = _git("diff", "--cached", "--name-only", "--diff-filter=d", root=root)
    if staged_out is None:
        return []
    staged = staged_out.decode("utf-8", "replace").split("\n")
    # Exact string membership, never a glob or a path join: the manifest is
    # repository content, and it must not be able to name anything outside what
    # this commit actually stages.
    candidates = [p for p in staged if p and p in recorded]

    blobs = _staged_blobs(root, candidates)
    return [path for path in candidates if path in blobs and hashlib.sha256(blobs[path]).hexdigest() == recorded[path]]


def main() -> int:
    top = _git("rev-parse", "--show-toplevel")
    if top is None:
        return 0
    root = Path(top.decode("utf-8", "replace").strip())
    for path in vouched(root):
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
