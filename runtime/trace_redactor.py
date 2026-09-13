"""
runtime/trace_redactor.py — Environment-aware OTLP span scrubbing.

Acts as an OpenTelemetry SpanProcessor. Intercepts spans before export
and applies the active redaction profile based on $ENVIRONMENT.

Redaction profiles (see SPECS.md §27):
  development  — full capture (up to 1,000 chars)
  staging      — PII/secret patterns stripped; structure preserved; hashed identifiers
  production   — minimal: hashed/truncated to 50 chars; full payload in encrypted HITL blob only

Usage:
    from runtime.trace_redactor import TraceRedactor
    provider = TracerProvider()
    provider.add_span_processor(TraceRedactor())   # reads ENVIRONMENT from env

Note on mutating spans in on_end(): the OTel SDK's `ReadableSpan` exposes
attributes as a read-only mapping by contract, but the only point at which a
processor can intercept a span before export is `on_end`. This processor
mutates the span's internal `_attributes` dict directly — the standard
workaround used by redaction/scrubbing processors in the OTel Python
ecosystem, since there is no public "rewrite before export" hook.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
from pathlib import Path
from typing import Any, Optional

from runtime.environment import get_environment

logger = logging.getLogger(__name__)

try:
    from opentelemetry.sdk.trace import (
        ReadableSpan,
        Span,
        SpanProcessor as _OTelSpanProcessor,
    )

    _HAS_OTEL = True
except ImportError:
    _HAS_OTEL = False
    _OTelSpanProcessor = object  # type: ignore

# The span attribute set by runtime/llm_gateway.py and the agent scripts on
# every span — the authoritative per-span tenant identity.
_TENANT_ATTRIBUTE = "tenant.id"


# ── Default secret/PII pattern library (§27) ──────────────────────────────────

_SECRET_PATTERNS = [
    re.compile(r"sk-ant-[A-Za-z0-9\-]{20,}"),  # Anthropic API keys
    re.compile(r"sk-[A-Za-z0-9]{20,}"),  # OpenAI API keys
    re.compile(r"Bearer\s+[A-Za-z0-9\-_.~+/]+=*"),  # Bearer tokens
    re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"),  # Email addresses
]

# Candidate credit-card-shaped digit runs; validated with Luhn before redaction.
from runtime.pii_patterns import (  # noqa: E402
    CARD_CANDIDATE as _CARD_CANDIDATE,
    EMIRATES_ID_DIGITS as _EMIRATES_ID_DIGITS,
    EMIRATES_ID_HYPHEN as _EMIRATES_ID_HYPHEN,
    PHONE as _PHONE,
    ascii_digits as _ascii_digits,
)


def _splice(text, probe, pattern, marker_fn, only_if=None):
    """Replace `pattern`'s matches in `probe` at the same spans in `text`.

    The two are the same length by construction (`ascii_digits` maps character
    for character), so a span found in one addresses the other exactly.
    """
    out, shift = text, 0
    for m in pattern.finditer(probe):
        if only_if is not None and not only_if(m.group(0)):
            continue
        replacement = marker_fn(m)
        start, end = m.start() + shift, m.end() + shift
        out = out[:start] + replacement + out[end:]
        shift += len(replacement) - (m.end() - m.start())
    return out

# Disabled by default in staging per §27 ("optional — disabled by default in staging").
_IP_PATTERN = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")

_REDACTED_MARKER = "[REDACTED]"

# Attributes that carry free-text payloads worth redacting/truncating.
_PAYLOAD_ATTRIBUTES = {"input.value", "output.value"}

# Attributes whose value is bounded and non-sensitive BY CONSTRUCTION, and which
# the production profile therefore must not truncate.
#
# `prompt.system.sha256` is a digest — runtime/prompt_identity records it
# precisely so the prompt itself never reaches a span, and it is the join column
# for "answers changed when this hash changed". Truncating it to 50 characters
# produced a string that is not a sha256 of anything and matches nothing
# computed anywhere else, so the one attribute designed to be safe in production
# was the one production broke.
_UNTRUNCATED_ATTRIBUTES = {"prompt.system.sha256"}


# Shared with input_guardrail.py — one Luhn implementation for both the
# pre-call guard and the post-call redactor (ReviewFindings-2026-07-18 B1).
# The shared version strips ALL non-digit separators (this module's old
# copy stripped only spaces/hyphens — identical on today's
# _CARD_CANDIDATE matches, divergent the moment that regex widens).
from runtime.luhn import luhn_valid as _luhn_valid  # noqa: E402 — sited beside the comment explaining the sharing


def _redact_credit_cards(text: str) -> str:
    def _sub(match: "re.Match") -> str:
        return _REDACTED_MARKER if _luhn_valid(match.group(0)) else match.group(0)

    return _CARD_CANDIDATE.sub(_sub, text)


def _hash8(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()[:8]


def _load_extra_patterns() -> list:
    """Tenant repos extend the pattern library via .agenticframework/redaction-patterns.yaml (§27).

    Anchored on `runtime.config.repo_root()`, which is what every other module
    in this package uses. This walked up from `Path.cwd()` with its own
    stop-at-`.git` rule — a SIXTH root finder, written before the others were
    consolidated and missed when they were.

    The consequence is not stylistic: which patterns load depended on the
    process's working directory. A worker started from a subdirectory outside
    the repo, or from `/`, found no file and returned an empty list — the same
    value as "this tenant declared no extra patterns". The parse-error path
    below is deliberately LOUD for exactly that reason, and the not-found path
    was silent, so a tenant's PII patterns could stop applying with nothing
    anywhere to say so.
    """
    from runtime.config import repo_root

    root = repo_root()
    for parent in [root, *root.parents]:
        candidate = parent / ".agenticframework" / "redaction-patterns.yaml"
        if candidate.exists():
            try:
                import yaml  # type: ignore

                data = yaml.safe_load(candidate.read_text()) or {}
                return [re.compile(p) for p in data.get("patterns", [])]
            except Exception as exc:
                # LOUD. A typo in a tenant's pattern file used to return an
                # empty list, indistinguishable from "this tenant declared no
                # extra patterns" — so the file silently stopped contributing
                # and every payload it was written to scrub went to Phoenix
                # less redacted than the tenant believed. The framework's own
                # patterns still apply, so this degrades rather than fails, but
                # it must not degrade quietly: it is a compliance control.
                logger.error(
                    "redaction patterns NOT loaded from %s (%s) — this tenant's "
                    "extra PII patterns are NOT being applied; framework defaults "
                    "only. Fix the file: spans are being exported meanwhile.",
                    candidate,
                    exc,
                )
                return []
        if (parent / ".git").exists():
            break
    return []


def _describe_extra_patterns(count: int) -> None:
    """Say, once, what the tenant pattern file contributed.

    Zero patterns is a legitimate state and an alarming one, and they look
    identical from the outside. Logged at INFO so an operator reading startup
    can see which of the two they have, rather than inferring it from what does
    not get redacted later.
    """
    if count:
        logger.info("redaction: %d tenant pattern(s) loaded from .agenticframework/", count)
    else:
        logger.info(
            "redaction: no tenant patterns loaded (no .agenticframework/"
            "redaction-patterns.yaml found from %s) — framework defaults only",
            repr(str(Path.cwd())),
        )


# ── Encrypted HITL blob storage (§27) ─────────────────────────────────────────


class HITLBlobStore:
    """
    Stores the full, unredacted payload for production spans flagged for HITL
    review. Encrypted with AES-256-GCM using a per-tenant key.

    The per-tenant key is HITL_ENCRYPTION_KEY_<TENANT>, where <TENANT> is the
    tenant id upper-cased with every non-alphanumeric character replaced by an
    underscore — `kyc-sentinel` becomes HITL_ENCRYPTION_KEY_KYC_SENTINEL. The
    id was previously spliced in raw, which is not a settable variable name for
    the hyphenated ids everyone actually uses, so those tenants fell through to
    the fleet-wide key without a word. Falling back is still allowed and is now
    reported.

    Storage backend: local filesystem by default (HITL_BLOB_DIR, default
    runtime/.hitl_blobs/); set HITL_BLOB_S3_BUCKET to write to S3 instead.

    TTL is recorded in blob metadata (default 90 days, §27) — actual expiry
    is enforced by an external lifecycle job (S3 lifecycle rule, or a cron
    that purges local blobs older than their recorded TTL), not by this class.
    """

    def __init__(self, tenant_id: str) -> None:
        self.tenant_id = tenant_id

    @staticmethod
    def env_suffix(tenant_id: str) -> str:
        """A tenant id as the tail of an environment variable NAME.

        The per-tenant variable used to be `HITL_ENCRYPTION_KEY_{id.upper()}`
        with the id spliced in raw, which is not a variable name for any id
        containing a hyphen or a dot — `HITL_ENCRYPTION_KEY_KYC-SENTINEL` is
        something a shell refuses to set. The conventional tenant id is exactly
        that shape (this framework's own testbed is `kyc-sentinel`), so for
        those tenants the per-tenant key was UNREACHABLE and every one of them
        silently shared the fleet-wide HITL_ENCRYPTION_KEY — in the one store
        that holds unredacted payloads on purpose, while the class docstring
        and OPERATIONS.md both promise a per-tenant key.
        """
        return re.sub(r"[^A-Za-z0-9]", "_", tenant_id).upper()

    def _key(self) -> bytes:
        from runtime.environment import warn_degraded_default

        def configured(name: str) -> Optional[str]:
            """The value, or None. Whitespace-only counts as unset.

            The value is returned RAW. It is hashed to derive the key, so
            stripping it would change the derived key and make every blob
            already written with it undecryptable.
            """
            value = os.environ.get(name)
            return value if value and value.strip() else None

        suffix = self.env_suffix(self.tenant_id)
        # The legacy spelling is still honoured: for an id with no punctuation
        # the two are identical, and for one WITH punctuation it was never
        # settable, so nothing that works today stops working.
        raw = configured(f"HITL_ENCRYPTION_KEY_{suffix}") or configured(
            f"HITL_ENCRYPTION_KEY_{self.tenant_id.upper()}"
        )
        if raw is None:
            raw = configured("HITL_ENCRYPTION_KEY")
            if raw is not None:
                warn_degraded_default(
                    f"hitl-shared-key:{self.tenant_id}",
                    f"tenant={self.tenant_id!r} is encrypting HITL compliance "
                    f"blobs with the FLEET-WIDE HITL_ENCRYPTION_KEY. Anyone who "
                    f"can decrypt one tenant's payloads can decrypt this one's. "
                    f"Set HITL_ENCRYPTION_KEY_{suffix} for a per-tenant key.",
                )
        if raw is None:
            raise RuntimeError(
                f"No HITL encryption key configured for tenant={self.tenant_id!r}. "
                f"Set HITL_ENCRYPTION_KEY_{suffix} or HITL_ENCRYPTION_KEY."
            )
        # Derive a 32-byte key regardless of input length/encoding.
        return hashlib.sha256(raw.encode("utf-8")).digest()

    def put(self, ref: str, plaintext: str, ttl_days: int = 90) -> str:
        """Encrypt and store plaintext under ref. Returns the blob reference.

        Raises RuntimeError immediately for configuration errors (missing
        encryption key) — silently swallowing that case used to leave a span
        with an `hitl_blob_ref` pointing at a blob that was never written,
        defeating the "full payload preserved for compliance review"
        guarantee with zero error or alert (Product_Archive.md 2.3).
        Storage I/O errors (disk full, S3 unreachable) are logged at ERROR
        and swallowed — a transient storage outage still shouldn't break
        trace export, but it must not be invisible either.
        """
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        import os as _os
        import json
        import time

        key = self._key()  # raises RuntimeError if unconfigured — let it propagate
        nonce = _os.urandom(12)
        aesgcm = AESGCM(key)
        ciphertext = aesgcm.encrypt(nonce, plaintext.encode("utf-8"), None)

        bucket = os.environ.get("HITL_BLOB_S3_BUCKET")
        blob = {
            "nonce": nonce.hex(),
            "ciphertext": ciphertext.hex(),
            "tenant_id": self.tenant_id,
            "created_at": time.time(),
            "ttl_days": ttl_days,
        }
        try:
            if bucket:
                import boto3  # type: ignore

                s3 = boto3.client("s3")
                s3.put_object(
                    Bucket=bucket,
                    Key=f"hitl/{self.tenant_id}/{ref}.json",
                    Body=json.dumps(blob),
                )
            else:
                blob_dir = Path(
                    os.environ.get(
                        "HITL_BLOB_DIR",
                        str(Path(__file__).resolve().parent / ".hitl_blobs"),
                    )
                )
                blob_dir = blob_dir / self.tenant_id
                blob_dir.mkdir(parents=True, exist_ok=True)
                (blob_dir / f"{ref}.json").write_text(json.dumps(blob))
            return ref
        except OSError as exc:
            logger.error(
                "HITL blob persistence failed for tenant=%s ref=%s: %s",
                self.tenant_id,
                ref,
                exc,
            )
            return ref
        except Exception as exc:  # e.g. boto3 ClientError — not importable to catch by name unconditionally
            logger.error(
                "HITL blob persistence failed for tenant=%s ref=%s: %s",
                self.tenant_id,
                ref,
                exc,
            )
            return ref

    def _blob_path(self, ref: str) -> Path:
        blob_dir = Path(
            os.environ.get(
                "HITL_BLOB_DIR",
                str(Path(__file__).resolve().parent / ".hitl_blobs"),
            )
        )
        return blob_dir / self.tenant_id / f"{ref}.json"

    def get(self, ref: str) -> Optional[str]:
        """Decrypt and return the payload stored under `ref`, or None if absent.

        The counterpart to put(), and until now there was none: nothing in this
        framework could read a HITL blob back. The store exists so an unredacted
        payload survives for compliance review, and the only way to satisfy such
        a request was to reverse-engineer the format from put() — read the JSON,
        hex-decode nonce and ciphertext, SHA-256 the environment key, AES-GCM
        open. An operator doing that under time pressure, on the one artifact
        that holds the data the redactor deliberately removed from everywhere
        else, is not a procedure anyone should have to invent.

        A wrong key raises rather than returning nonsense: AES-GCM authenticates,
        so a failed tag is the difference between "this blob belongs to another
        tenant" and "this blob is corrupt", and both need saying out loud.

        Only the local filesystem backend is implemented here. put() also writes
        to S3 when HITL_BLOB_S3_BUCKET is set, and that path has no test and no
        S3 in CI to get one; claiming to read it back would be claiming more
        than is verified.
        """
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        import json

        if os.environ.get("HITL_BLOB_S3_BUCKET"):
            raise NotImplementedError(
                "HITLBlobStore.get() reads the local filesystem backend only; "
                "HITL_BLOB_S3_BUCKET is set. Fetch the object from S3 and decrypt "
                "it with the same derivation (see this method's source)."
            )

        path = self._blob_path(ref)
        if not path.exists():
            return None

        blob = json.loads(path.read_text())
        key = self._key()  # raises RuntimeError if unconfigured — let it propagate
        try:
            plaintext = AESGCM(key).decrypt(
                bytes.fromhex(blob["nonce"]),
                bytes.fromhex(blob["ciphertext"]),
                None,
            )
        except Exception as exc:
            raise RuntimeError(
                f"HITL blob {ref!r} for tenant={self.tenant_id!r} did not decrypt "
                f"with the configured key. Either the key changed or the blob "
                f"belongs to a different tenant. Underlying: {type(exc).__name__}"
            ) from exc
        return plaintext.decode("utf-8")


def _make_blob_ref(trace_id: str, span_id: str, attr_key: str) -> str:
    # span_id is required: a single trace with multiple independently
    # HITL-flagged sibling spans (e.g. Architect/Developer/Validator) would
    # otherwise all compute the same ref `{trace_id}.{attr_key}` and the last
    # write wins, permanently overwriting the earlier spans' encrypted
    # payloads before anyone reviews them (Product_Archive.md 2.2).
    return f"{trace_id}.{span_id}.{attr_key}"


# ── Span processor ────────────────────────────────────────────────────────────


def _writable_attributes(span: Any) -> Optional[dict]:
    """The span's attribute mapping, in a form that can be written to.

    CORRECTED 2026-08-25. An earlier commit (a18c848) claimed this helper fixed
    a live defect — that `span.end()` always handed processors an immutable
    `BoundedAttributes`, so every span raised `TypeError` out of `end()` and
    nothing was ever redacted. That is NOT what happens on the SDK this repo
    runs (1.42.x), and re-checking it against a real `span.end()` is what showed
    the diagnosis was wrong:

      * `BoundedAttributes.__init__` does default to `immutable=True`, so any
        hand-CONSTRUCTED one refuses writes — which is what the original probe
        exercised;
      * but a live span's attributes are built with `immutable=False`, and
        `Span._readable_span()` passes that same object through rather than
        copying it. Writes in `on_end` succeed, and always did.

    So the redaction control was working. What this helper actually buys is
    tolerance of the immutable shape where it does occur — a hand-built
    `ReadableSpan`, and any SDK version that copies attributes on the way out —
    and it costs one more layer of private access. That is worth keeping; the
    claim that it repaired a broken control was not true and is corrected here
    rather than left for a future reader to inherit.
    """
    attributes = getattr(span, "_attributes", None)
    if attributes is None:
        return None
    inner = getattr(attributes, "_dict", None)
    return inner if isinstance(inner, dict) else (attributes or None)


def _string_parts(value: Any) -> Optional[list[str]]:
    """The string content of an attribute, or None when there is none to scrub.

    OTel attribute values are `str | bool | int | float` or a homogeneous
    SEQUENCE of those. The scrub loop only ever handled the first `str` case, so
    every sequence-valued attribute passed through untouched — which is not a
    corner: `set_attribute("retrieved_docs", [...])` is ordinary, and this is
    the module that stops a document going to Phoenix with an email in it.

    A sequence containing a non-string is left alone rather than partly
    processed: mixed types are not a shape this scrubber understands, and
    guessing at one is how a redactor starts corrupting data instead of
    protecting it.
    """
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)) and value and all(isinstance(v, str) for v in value):
        return list(value)
    return None


def _rebuild(original: Any, parts: list[str]) -> Any:
    """Put scrubbed parts back in the shape they came from.

    Written back through `_attributes._dict`, which bypasses the SDK's own
    coercion — so a tuple that goes out as a list changes the exported type of
    an attribute for no reason.
    """
    if isinstance(original, str):
        return parts[0]
    return tuple(parts) if isinstance(original, tuple) else parts


class TraceRedactor(_OTelSpanProcessor):
    """
    SpanProcessor that scrubs sensitive data before export.

    Inherits from opentelemetry.sdk.trace.SpanProcessor (when available) so
    that newer SDK lifecycle hooks (e.g. `_on_ending`, added after this
    interface was first stabilised) get a safe no-op default instead of
    raising AttributeError when the SDK's TracerProvider calls them.

    profile:
      "none"       — no scrubbing (development)
      "staging"    — strip patterns; preserve structure; hash flagged identifiers
      "production" — strip patterns, truncate to 50 chars; full payload stashed
                      in an encrypted HITL blob
    """

    def __init__(
        self, profile: Optional[str] = None, tenant_id: Optional[str] = None
    ) -> None:
        env = profile or get_environment()
        self.profile = {
            "development": "none",
            "staging": "staging",
            "production": "production",
        }.get(env, "production")  # fail closed: unrecognized -> strictest profile
        # Fallback only — used when a span carries no tenant.id attribute at
        # all. The authoritative source is per-span, resolved in on_end()
        # below; binding tenant_id once here (at __init__/process-construction
        # time) was the actual cross-tenant leak: on a shared worker pool
        # processing spans for multiple tenants in one process, every
        # HITL-flagged span got encrypted with whichever tenant's key the
        # processor happened to be constructed with (Product_Archive.md 1.2).
        # Resolved the way every other module resolves it. This line read
        # os.environ["TENANT_ID"] directly and was the last place in the repo
        # doing that (tool_registry.py's comment counted five others before it).
        # Reading it raw got two things wrong, and both land on the fallback
        # that decides which tenant's key a HITL compliance blob is written
        # under — the same binding this class's __init__ comment calls the
        # original cross-tenant leak:
        #
        #   AGENT_TENANT_ID, the variable tenancy.py names FIRST and the docs
        #   call primary, was not read at all — a deployment setting only that
        #   one fell through to "unknown".
        #
        #   TENANT_ID="" — set-but-empty, routine in k8s manifests and CI
        #   matrices — yielded "" rather than the intended default, because
        #   os.environ.get only substitutes when the key is ABSENT.
        #
        # Non-fatal by design: resolve_tenant_id raises rather than guessing,
        # which is right for a worker refusing to start and wrong for a span
        # processor, where it would take down telemetry for an unset variable.
        try:
            from runtime.tenancy import resolve_tenant_id

            self.default_tenant_id = resolve_tenant_id(tenant_id)
        except Exception:
            self.default_tenant_id = "unknown"
        from runtime.config import as_bool, resolve

        self.enable_ip_redaction = as_bool(
            resolve("security.ip_redaction", env_var="ENABLE_IP_REDACTION", default=False)
        )
        self._extra_patterns = _load_extra_patterns()
        _describe_extra_patterns(len(self._extra_patterns))
        self._blob_stores: dict[str, HITLBlobStore] = {}

    def _blob_store_for(self, tenant_id: str) -> HITLBlobStore:
        store = self._blob_stores.get(tenant_id)
        if store is None:
            store = HITLBlobStore(tenant_id)
            self._blob_stores[tenant_id] = store
        return store

    def on_start(self, span: "Span", parent_context=None) -> None:  # type: ignore[override]
        pass  # No action on start — scrubbing happens once the span's final attributes are known.

    def on_end(self, span: "ReadableSpan") -> None:  # type: ignore[override]
        # See _writable_attributes: a finished span's attributes are immutable
        # in this SDK, and writing to them raised TypeError out of span.end().
        if self.profile == "none" or not _HAS_OTEL:
            return

        attributes = _writable_attributes(span)
        if not attributes:
            return

        trace_id = (
            format(span.context.trace_id, "032x")
            if getattr(span, "context", None)
            else "unknown"
        )
        span_id = (
            format(span.context.span_id, "016x")
            if getattr(span, "context", None)
            else "unknown"
        )
        tenant_id = attributes.get(_TENANT_ATTRIBUTE) or self.default_tenant_id

        for key, value in list(attributes.items()):
            # STRINGS AND SEQUENCES OF STRINGS. The loop used to `continue` on
            # anything that was not a `str`, and a sequence of strings is a
            # first-class OTel attribute type the SDK accepts without comment —
            # so `span.set_attribute("docs", [...])` carrying an email, an API
            # key or a card number was exported whole, in every profile,
            # production included. Verified against a real span before and
            # after; see test_trace_redactor.
            strings = _string_parts(value)
            if strings is None:
                continue

            if self.profile == "staging":
                attributes[key] = _rebuild(
                    value, [self._scrub(part, hash_identifiers=True) for part in strings]
                )
            elif self.profile == "production":
                scrubbed = [self._scrub(part, hash_identifiers=False) for part in strings]
                if key in _PAYLOAD_ATTRIBUTES and sum(len(p) for p in scrubbed) > 50:
                    ref = _make_blob_ref(trace_id, span_id, key)
                    try:
                        # The ORIGINAL, rendered — a sequence payload has to
                        # reach the compliance blob whole, like a string one.
                        self._blob_store_for(tenant_id).put(ref, "\n".join(strings))
                        attributes[f"{key}.hitl_blob_ref"] = ref
                    except RuntimeError as exc:
                        # Missing HITL_ENCRYPTION_KEY for this tenant — log
                        # loudly rather than silently truncating the payload
                        # with a dangling blob ref nothing ever wrote to.
                        logger.error(
                            "HITL blob NOT written for tenant=%s ref=%s: %s — payload truncated "
                            "without compliance backup.",
                            tenant_id,
                            ref,
                            exc,
                        )
                if key in _UNTRUNCATED_ATTRIBUTES:
                    attributes[key] = _rebuild(value, scrubbed)
                else:
                    attributes[key] = _rebuild(
                        value, [self._truncate(part, max_chars=50) for part in scrubbed]
                    )

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _scrub(self, text: str, hash_identifiers: bool) -> str:
        """Apply the pattern library to a string. hash_identifiers=True replaces
        matches with a short hash (staging — structure preserved, identifiers
        recoverable for correlation); False replaces with a flat marker
        (production — no information retained outside the encrypted blob)."""
        marker_fn = (
            (lambda m: f"[REDACTED:{_hash8(m.group(0))}]")
            if hash_identifiers
            else (lambda m: _REDACTED_MARKER)
        )

        # ASCII-normalised for MATCHING only; the text written out is the
        # original. `\d` already matched Arabic-Indic digits, but the Emirates
        # ID shapes anchor on a literal `784`, so `٧٨٤-…` matched nothing. See
        # runtime/pii_patterns.ascii_digits — a per-character mapping, so a span
        # here addresses the same characters there.
        probe = _ascii_digits(text)

        # The PERSONAL identifiers, from the same module input_guardrail.py
        # reads. They were absent here entirely: this half of the control knew
        # about API keys, bearer tokens, email and cards, and nothing about an
        # Emirates ID or a phone number — so an identifier the pre-call guard
        # stripped from a prompt still left the process on a span attribute.
        # The framework's own documentation calls the two symmetric, and the
        # tenant whose whole subject is Emirates IDs declared no extra patterns,
        # because nothing told it that it had to.
        for pattern in (_EMIRATES_ID_HYPHEN, _EMIRATES_ID_DIGITS, _PHONE):
            text = _splice(text, probe, pattern, marker_fn)
            probe = _ascii_digits(text)

        for pattern in (*_SECRET_PATTERNS, *self._extra_patterns):
            text = pattern.sub(marker_fn, text)

        probe = _ascii_digits(text)
        text = _splice(
            text, probe, _CARD_CANDIDATE, marker_fn, only_if=_luhn_valid
        )

        if self.enable_ip_redaction:
            text = _IP_PATTERN.sub(marker_fn, text)

        return text

    def _truncate(self, text: str, max_chars: int = 50) -> str:
        if len(text) > max_chars:
            return text[:max_chars] + "…[truncated]"
        return text
