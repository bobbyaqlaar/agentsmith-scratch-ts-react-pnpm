"""
runtime/version.py — which AgentSmith a tenant is actually running.

WHY IT HAS TO BE ON THE WIRE. AgentSmith and the tenants that use it have
different owners and different release cadences: IT operations ships the
framework and the Ops Portal; the business ships the tenant app and pins a
framework version so IT's cadence cannot move underneath it. That separation is
the design, and the pin is what makes it work.

The consequence nobody had accounted for is that IT then operates a FLEET of
tenants on different framework versions at once, and a version determines what
telemetry a tenant is even capable of emitting. A tenant pinned to v1.2.0 emits
no `prompt.system.sha256` (prompt_identity did not exist), no metrics at all
(neither did metrics.py), and `tenant.id` only when a caller remembered the
kwarg — there was no identity processor. On an ops dashboard that is
indistinguishable from a current tenant that is BROKEN.

`framework.version` was declared in `.agenticframework/tenant.yaml` by the
scaffold and read by nothing — the same declared-but-unenforced shape as
`tenant.id`, `budget.monthly_usd_cap` and `workflow.engine` before those were
closed. And a declaration in the tenant's own repo could not answer this
anyway: what matters is the version of the code that is RUNNING, not the one
the repo says it wants.

INSTALLED VERSUS CHECKED-OUT, and this is the part that must not be smoothed
over. `importlib.metadata` answers for a package installed from a release
artifact. A framework checkout on `sys.path` — `AGENTSMITH_DIR`, which is how
tenant CI and local development run — has a `pyproject.toml` whose version is
whatever the last release set, and `main` can be far ahead of it. Reporting a
bare "1.2.0" from a working copy that is twenty-odd changelog sections past the
tag would be a confident lie of exactly the kind this module exists to prevent.

So a checkout reports `1.2.0+src`. The `+src` is SemVer build metadata, it
compares equal for ordering, and it says the one thing an operator needs: this
is not a released artifact, so the number alone does not tell you what it
emits. `UNKNOWN` when even that cannot be determined — never a guess.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DIST_NAME = "agentsmith-runtime"

#: What `framework_version()` returns when nothing can answer. Never a number:
#: a wrong version is aggregated with real fleet data, an absent one is a gap
#: an operator can see.
UNKNOWN = "unknown"

#: Marks a version resolved from a source checkout rather than an installed
#: release artifact. See the module docstring.
SOURCE_SUFFIX = "+src"

_cached: Optional[str] = None


def _installed_version() -> Optional[str]:
    try:
        from importlib.metadata import version

        return version(DIST_NAME)
    except Exception:  # PackageNotFoundError, or no importlib.metadata at all
        return None


def _checkout_version() -> Optional[str]:
    """The `version = "x.y.z"` line of the pyproject.toml above this file.

    Parsed rather than tomllib-loaded so this works on any supported Python and
    cannot fail on an unrelated syntax error elsewhere in the file — the same
    regex `scripts/test/test_version_consistency.py` already pins.
    """
    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    try:
        text = pyproject.read_text(encoding="utf-8")
    except OSError:
        return None
    m = re.search(r'^version = "(\d+\.\d+\.\d+)"', text, re.M)
    return m.group(1) if m else None


def framework_version() -> str:
    """The running framework's version. Cached — it cannot change mid-process."""
    global _cached
    if _cached is not None:
        return _cached

    installed = _installed_version()
    if installed:
        _cached = installed
        return _cached

    checkout = _checkout_version()
    _cached = f"{checkout}{SOURCE_SUFFIX}" if checkout else UNKNOWN
    return _cached


def is_source_checkout(version: Optional[str] = None) -> bool:
    """Whether a reported version came from a working copy rather than a release.

    Exposed because a consumer reasoning about what a version EMITS has to treat
    these differently: a released 1.2.0 emits exactly what 1.2.0 emitted, and a
    1.2.0+src emits whatever that working copy happened to contain.
    """
    return (version if version is not None else framework_version()).endswith(
        SOURCE_SUFFIX
    )


# ── Declared versus installed ─────────────────────────────────────────────────

_mismatch_warned = False


def _series(version: str) -> Optional[tuple[int, int]]:
    """`(major, minor)` from `1.3.0`, `1.3.x`, `1.3.0+src`, or None."""
    v = version.strip()
    m = re.match(r"^(\d+)\.(\d+)\.", v) or re.match(r"^(\d+)\.(\d+)\.[xX*]$", v)
    return (int(m.group(1)), int(m.group(2))) if m else None


def warn_if_declared_version_differs(root: Optional[Path] = None) -> Optional[str]:
    """Warn once when the running framework crosses a MAJOR boundary from the
    version this tenant declares it was built against.

    Returns the warning text, or None when there is nothing to say — so a caller
    can surface it its own way and a test can assert on it without capturing
    logs.

    WHAT THIS IS AND IS NOT. It is not a check that two strings agree. A first
    version compared MINOR series and warned on any disagreement, which made it
    a bookkeeping alarm: a config string differing from an installed package is
    not a risk to anybody, and a warning that fires when nothing is wrong is one
    an operator learns to skip past.

    The obligations run in one direction and it is worth being explicit about
    them, because the check follows from them:

      * A TENANT conforms to AgentSmith's specs. It does that irrespective of
        version, and it does not owe anyone a config string kept in sync with
        whatever IT installed.
      * AGENTSMITH maintains backward compatibility for tenants already in
        production. Inside a major series that is a promise, not a hope — so a
        tenant declaring `1.3.x` and running 1.4 or 1.9 is the system working,
        and silence is the correct output.

    What breaks that promise is a MAJOR release, which is what the compatibility
    matrix exists to describe. That is the only case worth a tenant operator's
    attention, so it is the only case this warns about.

    ONE HISTORICAL EXCEPTION, and it is the reason to read this paragraph rather
    than trust the rule: 1.3.0 shipped five breaking changes as a MINOR, cut that
    way knowingly when the framework's only consumers were two repositories under
    one owner. So this check is silent on a 1.2.x → 1.3.x move that can genuinely
    break a tenant. The obligation holds from 1.3.0 forward; that upgrade needs
    the compatibility matrix read by hand. Special-casing it here was considered
    and rejected — a version check carrying a table of past mistakes is a table
    nobody maintains.

    WARNS, NEVER RAISES. Even across a major, refusing to start would take a
    running tenant down at upgrade time on the strength of a config string —
    and the tenant's own tests, not this, are what establish whether it still
    works. `resolve_tenant_id` refuses because an unattributed run corrupts the
    budget ledger and the audit trail; this is not that.
    """
    global _mismatch_warned
    if _mismatch_warned:
        return None

    running = framework_version()
    if running == UNKNOWN:
        _mismatch_warned = True
        text = (
            "AgentSmith cannot determine its own version — neither an installed "
            "`agentsmith-runtime` distribution nor a readable pyproject.toml. "
            "Telemetry will report the framework version as 'unknown'."
        )
        logger.warning(text)
        return text

    from runtime.config import config_get

    declared = config_get("framework.version", root)
    if declared is None or not str(declared).strip():
        return None

    want = _series(str(declared))
    have = _series(running)
    if want is None:
        _mismatch_warned = True
        text = (
            f"framework.version={declared!r} in .agenticframework/tenant.yaml is "
            f"not a version this can read (expected e.g. '1.3.x' or '1.3.0'); "
            f"the running framework is {running}."
        )
        logger.warning(text)
        return text
    if have is None or want[0] == have[0]:
        # Same major. Backward compatibility within a major is AgentSmith's
        # obligation, so any minor or patch difference here is the framework
        # keeping its side of the bargain — there is nothing to report.
        return None

    _mismatch_warned = True
    if have[0] > want[0]:
        text = (
            f"AgentSmith {running} is running, but this tenant declares it was "
            f"built against framework.version={declared!r}. That is a MAJOR "
            f"version boundary, which is where AgentSmith's backward-compatibility "
            f"promise ends — see the compatibility matrix in the framework's "
            f"CHANGELOG.md for what changed between {want[0]}.x and {have[0]}.x, "
            f"and re-run this tenant's own tests before trusting the upgrade."
        )
    else:
        text = (
            f"AgentSmith {running} is running, but this tenant declares "
            f"framework.version={declared!r} — a NEWER major than what is "
            f"installed. Anything this tenant uses that arrived in "
            f"{want[0]}.x will be missing; expect ImportError or absent "
            f"behaviour rather than a graceful degrade."
        )
    logger.warning(text)
    return text


def _reset_cache() -> None:
    """For tests only."""
    global _cached, _mismatch_warned
    _cached = None
    _mismatch_warned = False
