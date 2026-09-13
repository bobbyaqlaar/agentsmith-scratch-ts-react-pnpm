"""
runtime/embeddings.py — pluggable text embedders for vector retrieval.

EMBEDDER env:
  hash (DEFAULT — a deterministic fake with no semantic meaning; see below)
  sentence-transformers | st  — local SentenceTransformer model

EMBEDDING_MODEL — model id for sentence-transformers (default all-MiniLM-L6-v2)

THE DEFAULT IS A FAKE. HashEmbedder hashes text; it does not model meaning. It
exists so CI and a laptop can exercise the retrieval path without downloading a
model, and it is what `make_embedder()` returns when EMBEDDER is unset — which
is the documented usage in vector_store.py's own docstring:

    store = make_vector_store()
    hits = store.query("commodity volatility", k=5)

Measured on that path, unrelated text outscores a paraphrase:

    "sanctions screening returned a confirmed match"
      vs "the applicant appears on a sanctions list"   -0.025
      vs "the weather in Dubai is hot today"           +0.043

and the query "is this person on a sanctions list?" ranks "today is sunny in
Abu Dhabi" FIRST. Nothing errors, the scores look like scores, and the
retrieved context goes to the model as though it meant something.

This header used to read "default in tests / when unset and sentence-
transformers missing", describing a fallback chain that does not exist — nothing
here ever tries sentence-transformers on its own.

Using the fake outside development now says so once, at ERROR level. It stays a
warning rather than a refusal: tenants are running this default in production
today, and the framework does not break them outside a major release.
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import struct
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text."""
        ...

    @property
    def identity(self) -> str:
        """A stable name for whatever produced these vectors.

        HashEmbedder and all-MiniLM-L6-v2 both emit 384 dimensions, so a store
        cannot tell them apart by shape: vectors written by one and queried by
        the other produce confident nonsense and no error anywhere. This is the
        value a store records to notice that.
        """
        ...


def _warn_if_fake_outside_development() -> None:
    """Say once that retrieval is not semantic.

    Delegates to runtime.environment.warn_degraded_default — this was the first
    of these warnings and briefly the only one; the budget ledger and the vector
    index have the same shape, so the once-per-key and level-by-environment
    logic lives in one place rather than three.
    """
    from runtime.environment import warn_degraded_default

    warn_degraded_default(
        "embedder-hash",
        "EMBEDDER is unset or 'hash': retrieval is using HashEmbedder, a "
        "deterministic FAKE with no semantic meaning. Vector search will return "
        "ranked but arbitrary results. Set EMBEDDER=sentence-transformers (and "
        "install the `embeddings` extra) for real retrieval.",
    )


class HashEmbedder:
    """Deterministic fake embedder for tests/CI — no model download."""

    def __init__(self, dim: int = 384) -> None:
        if dim < 8:
            raise ValueError("dim must be >= 8")
        self.dim = dim
        _warn_if_fake_outside_development()

    @property
    def identity(self) -> str:
        return f"hash:{self.dim}"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._one(t) for t in texts]

    def _one(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        # Expand digest to dim floats in [-1, 1], then L2-normalize
        vals: list[float] = []
        seed = digest
        while len(vals) < self.dim:
            for i in range(0, len(seed) - 3, 4):
                if len(vals) >= self.dim:
                    break
                (n,) = struct.unpack(">I", seed[i : i + 4])
                vals.append((n / 0xFFFFFFFF) * 2.0 - 1.0)
            seed = hashlib.sha256(seed).digest()
        norm = math.sqrt(sum(v * v for v in vals)) or 1.0
        return [v / norm for v in vals]


class SentenceTransformerEmbedder:
    """Local embeddings via sentence-transformers (lazy import)."""

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or os.environ.get(
            "EMBEDDING_MODEL", "all-MiniLM-L6-v2"
        )
        self._model: Any = None

    @property
    def identity(self) -> str:
        return f"sentence-transformers:{self.model_name}"

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is not installed. "
                "pip install 'sentence-transformers>=3.0,<4.0' "
                "or set EMBEDDER=hash for tests."
            ) from exc
        self._model = SentenceTransformer(self.model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        # The embedding hop emitted nothing, and it is the one most likely to
        # be the latency this framework's users are hunting: a local
        # sentence-transformers model can take longer than the LLM call it
        # feeds. Instrumented on the REAL embedder only — HashEmbedder is
        # arithmetic, and a span per call would be noise per SPECS' own
        # "trace what can be slow" rule.
        from runtime.tracing import agent_span

        with agent_span(
            "embedding.encode", kind="embedding", model=self.model_name, count=len(texts)
        ) as span:
            self._load()
            assert self._model is not None
            vectors = self._model.encode(texts, normalize_embeddings=True)
            out = [list(map(float, row)) for row in vectors]
            try:
                if out:
                    span.set_attribute("agent.embedding.dimensions", len(out[0]))
            except Exception:  # fail-open
                pass
            return out


def make_embedder() -> Embedder:
    kind = os.environ.get("EMBEDDER", "").strip().lower()
    if kind in {"st", "sentence-transformers", "sentence_transformers", "local"}:
        return SentenceTransformerEmbedder()
    if kind in {"", "hash", "fake", "test"}:
        return HashEmbedder()
    raise ValueError(f"Unknown EMBEDDER={kind!r}; use hash or sentence-transformers")
