#!/usr/bin/env python3
"""
scripts/verify_judge_route.py — prove a configured judge is reachable, and that
it is reached where the registry says, before trusting it to gate merges.

Why this exists: a misconfigured judge used to fail SILENTLY rather than
loudly. `cost_router` substring-matched the model id and fell through to
localhost Ollama for anything it did not recognise, so a Grok or Gemini judge
was served by a local model under its own name. Nothing in a scorecard revealed
it — the recorded provenance was the requested id.

Checks, in order (each is a separate failure you would otherwise debug blind):
  1. the `judge` role resolves from the merged registry
  2. its credential variable is set
  3. it routes to the host its provider implies, not to localhost
  4. a real round-trip returns parseable JSON

Usage:
    python3 scripts/verify_judge_route.py
    python3 scripts/verify_judge_route.py --model gemini-2.5-pro
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _shared import _load_dotenv, judge_model, load_registry, role_credential_env


def main() -> int:
    _load_dotenv()
    ap = argparse.ArgumentParser(description="Verify the judge route end to end")
    ap.add_argument("--model", help="override the registry's judge id")
    args = ap.parse_args()

    model = args.model or judge_model()
    registry = load_registry() or {}
    cfg = next((c for c in registry.values() if c.get("id") == model), {})
    provider = cfg.get("provider", "(undeclared — falling back to id heuristics)")

    print(f"  judge model:   {model}")
    print(f"  provider:      {provider}")

    if not cfg:
        print(
            "  ⚠️  This id is not declared in any models.yaml role. Routing will "
            "fall back to\n      name-substring heuristics, which resolve an "
            "unrecognised id to LOCAL Ollama."
        )

    import os

    from cost_router import _route_for_model

    env = role_credential_env("judge")
    if env:
        present = bool(os.environ.get(env, "").strip())
        print(f"  credential:    {env} {'✅ set' if present else '❌ NOT SET'}")
        if not present:
            print(f"\n  ❌ Export {env} and re-run.")
            return 1
    else:
        print("  credential:    none required (local route)")

    route = _route_for_model(model)
    print(f"  resolved host: {route.base_url}")

    declared_cloud = provider not in ("ollama", None) and cfg
    if declared_cloud and route.is_local:
        print(
            f"\n  ❌ {model} is declared as {provider!r} but resolved to a LOCAL "
            f"route.\n     A judge silently served by Ollama is the exact failure "
            f"this script exists to catch."
        )
        return 1

    print("\n  Calling the judge with a trivial scoring prompt…")
    try:
        from cost_router import call as llm_call

        raw = llm_call(
            'Score this trivially. Respond with JSON only: {"score": 1.0}',
            system="You are a strict technical evaluator. Respond with JSON only.",
            task_type="review",
            force_model=model,
        )
    except Exception as exc:
        print(f"  ❌ Call failed: {exc}")
        return 1

    snippet = raw.strip()[:200]
    print(f"  response:      {snippet!r}")

    # An empty or non-JSON reply is a FAILURE, not a pass. eval_judge scores a
    # verdict it cannot parse as 0.0 with no error set — which reads as "the
    # application failed every case" rather than "the judge said nothing".
    import re

    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        print(
            "\n  ❌ Reachable, but returned no JSON object"
            f"{' (empty response)' if not raw.strip() else ''}.\n"
            "     eval_judge scores an unparseable verdict as 0.0 with no error "
            "set, so this\n     judge would report a perfect run as a total "
            "failure. Check the model is\n     pulled/available and honours a "
            "JSON-only instruction."
        )
        return 1
    try:
        verdict = json.loads(m.group(0))
    except Exception as exc:
        print(f"\n  ❌ Found a JSON-looking block that does not parse: {exc}")
        return 1

    print(f"  ✅ Parseable verdict {verdict} — this judge can score a scorecard.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
