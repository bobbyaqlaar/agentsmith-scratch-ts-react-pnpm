"""Where a security artefact lives, for the two runtime modules that look.

`prompt_guard` and `tool_registry` each resolved their own file the same way —
env override, else a conventional path under `.agent-rfc/security/`, else None —
differing only in the variable name and the filename. Two copies of a lookup
rule is two places to change when the convention moves, and the second one is
the one that gets missed.

Kept in `runtime/` rather than `scripts/_shared.py` because these run inside a
tenant's process at request time; `scripts/` is tooling and is not importable
from an installed package.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

SECURITY_DIR = Path(".agent-rfc") / "security"


def security_artefact_path(env_var: str, filename: str) -> Optional[Path]:
    """Resolve one security artefact, or None when the tenant has not authored it.

    An explicit env var always wins and is returned WITHOUT an existence check:
    if someone points at a file that is not there, that is a misconfiguration
    worth surfacing at the point of use, not silently falling back to a
    convention they were trying to override.

    The conventional path is existence-checked, because absence there is the
    normal state for a tenant that has not opted into the control.
    """
    override = os.environ.get(env_var, "").strip()
    if override:
        return Path(override)
    candidate = SECURITY_DIR / filename
    return candidate if candidate.exists() else None
