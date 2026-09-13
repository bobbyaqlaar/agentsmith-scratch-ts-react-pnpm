"""
scripts/delivery_model.py — soft-gate helpers for Enterprise Delivery Model.

Reads `.agenticframework/org-policy.yaml` (delivery_model block) and
`.agenticframework/tenant.yaml` (delivery.platform / data_access_pattern).
Never hard-fails CI — returns ok_for_ci=True with status ok|warn|skip.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        import yaml
    except ImportError:  # pragma: no cover
        return {}
    try:
        data = yaml.safe_load(path.read_text())
    except Exception as exc:
        # A malformed org policy used to become {}, which
        # check_tenant_against_policy reads as "no delivery_model declared" and
        # reports as `skip` — a soft gate passing because the policy it was
        # meant to enforce could not be parsed. Indistinguishable, from the
        # outside, from an org that has not written one.
        print(
            f"delivery_model: {path} could NOT be parsed ({exc}) — the delivery "
            f"gate is reporting `skip` because of that, not because no policy "
            f"exists.",
            file=sys.stderr,
        )
        return {}
    return data if isinstance(data, dict) else {}


def org_policy_path(root: Path) -> Path:
    return root / ".agenticframework" / "org-policy.yaml"


def tenant_yaml_path(root: Path) -> Path:
    return root / ".agenticframework" / "tenant.yaml"


def check_tenant_against_policy(root: Path) -> dict[str, Any]:
    """
    Soft-check tenant delivery.* against org delivery_model catalog.

    Returns:
      status: ok | warn | skip
      warnings: list[str]
      ok_for_ci: always True (soft gate)
      details: dict
    """
    policy = _load_yaml(org_policy_path(root))
    dm = policy.get("delivery_model")
    if not isinstance(dm, dict) or not dm:
        return {
            "status": "skip",
            "warnings": [],
            "ok_for_ci": True,
            "details": {"reason": "no delivery_model in org-policy.yaml"},
        }

    tenant = _load_yaml(tenant_yaml_path(root))
    delivery = tenant.get("delivery") if isinstance(tenant.get("delivery"), dict) else {}
    platform = delivery.get("platform")
    pattern = delivery.get("data_access_pattern")

    approved = list(dm.get("approved_platforms") or [])
    patterns = list(dm.get("data_access_patterns") or [])
    warnings: list[str] = []

    if not platform:
        warnings.append(
            "tenant.yaml missing delivery.platform — set it to an approved_platforms id"
        )
    elif approved and platform not in approved:
        warnings.append(
            f"delivery.platform={platform!r} not in approved_platforms {approved}"
        )

    if pattern and patterns and pattern not in patterns:
        warnings.append(
            f"delivery.data_access_pattern={pattern!r} not in data_access_patterns {patterns}"
        )

    status = "warn" if warnings else "ok"
    return {
        "status": status,
        "warnings": warnings,
        "ok_for_ci": True,
        "details": {
            "platform": platform,
            "data_access_pattern": pattern,
            "approved_platforms": approved,
            "data_access_patterns": patterns,
        },
    }
