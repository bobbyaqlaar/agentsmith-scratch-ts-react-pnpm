"""
scripts/security/runners/sovereign_residency.py — SEC-SOV-001.

Data residency, checked as a DECLARATION rather than a reachability probe.

`verify_sovereign_endpoint.py` is a live probe needing Hugging Face credentials.
Binding a compliance control to it would fail the control whenever an account
was unfunded or an endpoint was briefly down — an availability check wearing a
compliance label, which is why SEC-SOV-001 was left a declared gap instead.

But residency is not a reachability property in the first place. What matters
is whether any role CAN leave the jurisdiction, and the honest way to ask that
is of the registry, not of one endpoint that happens to answer today.

The degrade ladder is the reason this is worth doing at all. A live probe checks
the primary endpoint — which is precisely the one that is in-border. Residency
leaks on the *fallback*: a role pinned to an in-border Ollama that degrades to a
hosted API leaves the country on its first overload, and the probe stays green
throughout, because the endpoint it probes never changed. So the chain is walked
with `llm_gateway.degrade_chain`, the same function the gateway uses at runtime.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from security.registry import ControlSpec
from security.report import ControlResult
from security.runners._shared import failed, framework_root, not_applicable, passed

# Providers that can be operated inside a jurisdiction the tenant controls.
#
# `ollama` is self-hosted by definition. The OpenAI-compatible providers are
# here because "sovereign" is a property of the ENDPOINT, not the vendor: an
# in-border deployment speaking the OpenAI wire format is a normal Pattern B
# sovereign install (templates/uae-sovereign/README.md), and refusing it would
# make the control wrong about a supported configuration. What those entries
# must carry is an explicit `endpoint` — see below.
_SELF_HOSTABLE = {"ollama", "vllm", "openai_compatible", "azure_openai", "vertex_ai"}

# Providers with no in-border deployment story: the vendor's own multi-tenant
# API, wherever that runs. A sovereign profile routing to one of these has left
# the jurisdiction regardless of what any endpoint probe reports.
_HOSTED_ONLY = {"anthropic", "openai", "groq", "openrouter", "xai", "google_ai", "bedrock"}


def _endpoint_is_declared(cfg: dict) -> bool:
    """A self-hostable provider still has to say WHERE.

    `provider: azure_openai` with no endpoint is the vendor default, which is
    not in-border by accident. An endpoint referencing an env var (the template
    uses `${OLLAMA_BASE_URL}/v1`) counts as declared: where it points is a
    deployment decision, and the control's job is to ensure the registry does
    not silently fall back to a vendor default.
    """
    return bool(str(cfg.get("endpoint") or "").strip())


def _violations(roles: dict) -> list[str]:
    """Every way a role can serve traffic from outside the boundary."""
    problems: list[str] = []
    from runtime.llm_gateway import degrade_chain

    for role in sorted(roles):
        for step, target in enumerate(degrade_chain(roles, role)):
            cfg = roles.get(target)
            if cfg is None:
                # A degrade target that resolves to nothing is not a safe
                # failure: the ladder simply ends early, and the caller gets
                # whatever the gateway does when it runs out of rungs. Naming
                # it here beats discovering it during an outage.
                problems.append(
                    f"{role}: degrades to {target!r}, which is not a declared role"
                )
                continue
            provider = str(cfg.get("provider") or "").strip() or "(unset)"
            via = "" if step == 0 else f" (via degrade → {target})"
            if provider in _HOSTED_ONLY:
                problems.append(
                    f"{role}: provider {provider!r} is a hosted multi-tenant API "
                    f"with no in-border deployment{via}"
                )
            elif provider not in _SELF_HOSTABLE:
                problems.append(
                    f"{role}: provider {provider!r} is not a recognised "
                    f"self-hostable provider{via}"
                )
            elif not _endpoint_is_declared(cfg):
                problems.append(
                    f"{role}: provider {provider!r} declares no endpoint, so it "
                    f"resolves to the vendor default{via}"
                )
    return problems


def run(control: ControlSpec, ctx: dict[str, Any]) -> ControlResult:
    template = framework_root(ctx) / "templates" / "uae-sovereign" / "models.yaml"
    if not template.exists():
        # The framework ships the template; a tenant checkout may not carry it.
        # Reporting a gap there would claim the tenant had failed a residency
        # check it never opted into.
        return not_applicable(
            control, "no templates/uae-sovereign/models.yaml in this checkout"
        )

    import sys

    root = framework_root(ctx)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from runtime.llm_gateway import _load_yaml, _roles_from_doc

    doc = _load_yaml(Path(template))
    if not doc:
        return failed(control, f"{template.name} is empty or unparseable")

    try:
        # Resolves either registry shape, so the check follows the template
        # across a migration to catalog+profiles without being rewritten.
        roles = _roles_from_doc(doc)
    except ValueError as exc:
        return failed(control, f"{template.name}: {exc}")

    if not roles:
        return failed(control, f"{template.name} declares no roles")

    problems = _violations(roles)
    if problems:
        return failed(
            control,
            f"{len(problems)} residency violation(s) in the sovereign template: "
            + "; ".join(problems),
        )

    return passed(
        control,
        f"all {len(roles)} sovereign roles stay in-border across the full degrade "
        f"ladder ({', '.join(sorted(roles))})",
    )
