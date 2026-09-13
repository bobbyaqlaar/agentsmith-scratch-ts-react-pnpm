#!/usr/bin/env python3
"""
verify_ttft.py — Live Ollama TTFT gate (Reliability pack v1).

Streams a tiny prompt against Ollama's OpenAI-compatible API, prints ttft_ms,
and exits non-zero when over budget or unreachable.

Exit codes:
  0 — TTFT within TTFT_FAIL_ABOVE_MS
  1 — TTFT exceeds TTFT_FAIL_ABOVE_MS
  2 — connection / stream failure (Ollama unreachable or no first token)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import httpx

from _shared import _repo_root, _load_dotenv, role_model  # noqa: F401 — _repo_root kept for callers

DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_FAIL_ABOVE_MS = 2000
# The registry's smallest local tier, so a TTFT smoke test measures a model the
# stack actually routes to. Was hardcoded `falcon3:1b`, which models.yaml never
# referenced at all — the number it produced described nothing in the system.
DEFAULT_MODEL = role_model("fast", "smollm2")


def _resolve_base_url() -> str:
    raw = os.environ.get("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL).strip()
    return raw or DEFAULT_OLLAMA_BASE_URL


def _resolve_fail_above_ms() -> float:
    raw = os.environ.get("TTFT_FAIL_ABOVE_MS", str(DEFAULT_FAIL_ABOVE_MS)).strip()
    return float(raw or DEFAULT_FAIL_ABOVE_MS)


def measure_ttft_ms(base_url: str, model: str) -> float:
    """POST a streaming chat completion; return ms to first content delta."""
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    body = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with one word: hi"}],
        "stream": True,
        "max_tokens": 32,
        "temperature": 0.0,
    }
    start = time.perf_counter()
    ttft_ms: float | None = None

    with httpx.Client(timeout=120.0) as client:
        with client.stream("POST", url, json=body) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line.startswith("data:"):
                    continue
                payload = line.removeprefix("data:").strip()
                if not payload or payload == "[DONE]":
                    continue
                data = json.loads(payload)
                content = (
                    data.get("choices", [{}])[0].get("delta", {}).get("content")
                )
                if not content:
                    continue
                ttft_ms = (time.perf_counter() - start) * 1000
                break

    if ttft_ms is None:
        raise RuntimeError("stream ended without a content chunk")
    return ttft_ms


def main(argv: list[str] | None = None) -> int:
    _load_dotenv()
    parser = argparse.ArgumentParser(description="Verify live Ollama TTFT budget")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Ollama model id (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--fail-above-ms",
        type=float,
        default=None,
        help="Fail when ttft_ms exceeds this (default: TTFT_FAIL_ABOVE_MS env)",
    )
    args = parser.parse_args(argv)

    base_url = _resolve_base_url()
    fail_above_ms = (
        args.fail_above_ms
        if args.fail_above_ms is not None
        else _resolve_fail_above_ms()
    )

    try:
        ttft_ms = measure_ttft_ms(base_url, args.model)
    except (httpx.ConnectError, httpx.ConnectTimeout, httpx.NetworkError) as exc:
        print(f"connection failed: {exc}", file=sys.stderr)
        return 2
    except httpx.HTTPStatusError as exc:
        print(
            f"ollama HTTP {exc.response.status_code}: {exc.response.text[:400]}",
            file=sys.stderr,
        )
        return 2
    except Exception as exc:
        print(f"ttft check failed: {exc}", file=sys.stderr)
        return 2

    print(f"ttft_ms={ttft_ms:.1f}")
    if ttft_ms > fail_above_ms:
        print(
            f"TTFT {ttft_ms:.1f}ms exceeds budget {fail_above_ms:.0f}ms",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
