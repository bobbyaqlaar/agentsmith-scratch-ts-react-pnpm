#!/usr/bin/env python3
"""Live-verify UAE sovereign / Falcon model endpoints (same bar as vertex_gemini notes).

Default: Ollama OpenAI-compatible API with Falcon 3 tags
  falcon3:3b, falcon3:1b

  OLLAMA_BASE_URL=http://127.0.0.1:11434 python3 scripts/verify_sovereign_endpoint.py
  python3 scripts/verify_sovereign_endpoint.py --ollama
  HF_TOKEN=... python3 scripts/verify_sovereign_endpoint.py --hf
  HF_TOKEN=... python3 scripts/verify_sovereign_endpoint.py --hf --local-only

Exit 0 if at least one model returns text (or HTTP 200 with empty body for
tiny models); non-zero otherwise.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _sovereign_ollama_models() -> tuple[str, ...]:
    """Ollama model ids from the UAE sovereign starter registry.

    Sourced from templates/uae-sovereign/models.yaml — the file this script
    exists to verify — rather than a hardcoded tuple beside it. That registry
    IS the sovereign profile's definition; a smoke test that checks different
    tags than the profile declares proves nothing about the profile.

    Falls back to the historical pair if the template is missing or unreadable,
    so the script still runs from a partial install.
    """
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(
            (ROOT / "templates" / "uae-sovereign" / "models.yaml").read_text(
                encoding="utf-8"
            )
        )
        ids = {
            cfg["id"]
            for cfg in (data or {}).get("models", {}).values()
            if isinstance(cfg, dict)
            and cfg.get("provider") == "ollama"
            and cfg.get("id")
        }
        if ids:
            # Largest first: the profile's primary route before its fallback.
            return tuple(sorted(ids, reverse=True))
    except Exception:  # fail-open: a smoke test must not die on a missing template
        pass
    return ("falcon3:3b", "falcon3:1b")  # model-literal-ok: fallback == the template's own ids


OLLAMA_MODELS = _sovereign_ollama_models()
HF_MODELS = (
    "tiiuae/Falcon-E-3B-Base-prequantized",
    "tiiuae/Falcon-H1-Tiny-R-0.6B-pre-GRPO",
)
DEFAULT_HF_BASE = "https://router.huggingface.co/v1"
DEFAULT_OLLAMA_BASE = "http://127.0.0.1:11434"


# _load_dotenv lives in _shared.py (ReviewFindings-2026-07-18 B3). This
# script's old private copy raised on unreadable .env; the shared version
# is best-effort (never fatal), matching the other callers.
from _shared import _load_dotenv  # noqa: E402


def _chat_url(base_url: str) -> str:
    """Normalize to .../v1/chat/completions (OLLAMA_BASE_URL has no /v1)."""
    base = base_url.rstrip("/")
    if base.endswith("/v1"):
        return base + "/chat/completions"
    if base.endswith("/chat/completions"):
        return base
    return base + "/v1/chat/completions"


def _chat(
    base_url: str,
    model: str,
    prompt: str,
    token: str | None = None,
    timeout: float = 120.0,
) -> str:
    url = _chat_url(base_url)
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 32,
            "temperature": 0,
        }
    ).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "agentsmith-verify-sovereign/1.0",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, data=body, method="POST", headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    choices = data.get("choices") or []
    if not choices:
        return ""
    msg = choices[0].get("message") or {}
    return (msg.get("content") or "").strip()


def _ollama_base() -> str:
    return (os.environ.get("OLLAMA_BASE_URL") or DEFAULT_OLLAMA_BASE).strip()


def _generate_local_hf(token: str, model: str, prompt: str) -> str:
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model, token=token)
    mdl = AutoModelForCausalLM.from_pretrained(model, dtype=torch.float32, token=token)
    mdl.eval()
    text = prompt
    if getattr(tok, "chat_template", None):
        try:
            text = tok.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            text = prompt
    inputs = tok(text, return_tensors="pt")
    inputs = {k: v for k, v in inputs.items() if k != "token_type_ids"}
    with torch.no_grad():
        out = mdl.generate(**inputs, max_new_tokens=16, do_sample=False)
    return tok.decode(out[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="*", default=[])
    parser.add_argument("--ollama", action="store_true", help="Verify Ollama Falcon 3 (default)")
    parser.add_argument("--hf", action="store_true", help="Verify Hugging Face Falcon ids")
    parser.add_argument("--local-only", action="store_true", help="HF: skip router, local transformers only")
    parser.add_argument("--router-only", action="store_true", help="HF: router only")
    args = parser.parse_args()

    _load_dotenv(ROOT)
    use_hf = args.hf
    use_ollama = args.ollama or not use_hf

    prompt = "Say the word ok"
    ok: list[tuple[str, str]] = []
    errors: list[str] = []

    if use_ollama and not use_hf:
        models = args.models or list(OLLAMA_MODELS)
        base = _ollama_base()
        for model in models:
            try:
                text = _chat(base, model, prompt)
                print(f"OK  ollama {model}  reply={text[:80]!r}")
                ok.append((model, "ollama"))
            except Exception as e:
                detail = ""
                if isinstance(e, urllib.error.HTTPError):
                    detail = e.read().decode("utf-8", errors="replace")[:200]
                    print(f"FAIL ollama {model}  HTTP {e.code}: {detail}", file=sys.stderr)
                else:
                    print(f"FAIL ollama {model}  {type(e).__name__}: {e}", file=sys.stderr)
                errors.append(f"{model}: {e}")

    if use_hf:
        token = (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or "").strip()
        if not token:
            print("FAIL: HF_TOKEN missing for --hf mode.", file=sys.stderr)
            return 2
        models = args.models or list(HF_MODELS)
        hf_base = (os.environ.get("HF_BASE_URL") or DEFAULT_HF_BASE).strip()
        for model in models:
            if not args.local_only:
                try:
                    text = _chat(hf_base, model, prompt, token=token)
                    print(f"OK  router {model}  reply={text[:80]!r}")
                    ok.append((model, "hf_router"))
                    continue
                except Exception as e:
                    print(f"SKIP router {model}  {e}", file=sys.stderr)
                    if args.router_only:
                        errors.append(f"{model}: {e}")
                        continue
            if args.router_only:
                continue
            try:
                text = _generate_local_hf(token, model, prompt)
                print(f"OK  local  {model}  reply={text[:80]!r}")
                ok.append((model, "local_transformers"))
            except Exception as e:
                print(f"FAIL local  {model}  {type(e).__name__}: {e}", file=sys.stderr)
                errors.append(f"{model}: {e}")

    today = date.today().isoformat()
    if ok:
        parts = [f"{m} via {how}" for m, how in ok]
        print(
            f"\n# Live-verified ({today}): {', '.join(parts)}. "
            "For national data keep OLLAMA_BASE_URL in-border."
        )
        return 0

    print("\nFAIL: no model returned a completion.", file=sys.stderr)
    if errors:
        print("Errors: " + "; ".join(errors), file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
