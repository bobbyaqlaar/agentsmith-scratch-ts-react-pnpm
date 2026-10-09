#!/usr/bin/env python3
"""
scripts/mutation_check.py — curated mutation testing.

A MUTATION is a single deliberate edit that breaks one property the code is
meant to hold. If no test fails, that property is unasserted — the suite is
green for a reason other than the one it claims. A mutation that does not
change the file, or that runs against a failing baseline, proves nothing.

This is the hand-picked half of the framework's mutation testing. Every entry
below was written while fixing the defect it names, and each one FAILED at
least one test at the moment it was added; keeping them means a later edit
cannot quietly remove a guard whose tests would then still pass. The
exhaustive half is mutmut (see pyproject.toml [tool.mutmut] and the
non-blocking CI job), which enumerates mutations nobody thought to write.

Usage:
    python3 scripts/mutation_check.py              # every suite
    python3 scripts/mutation_check.py --list       # names only, run nothing
    python3 scripts/mutation_check.py structured_output prompt_guard

Exit codes: 0 all mutations caught · 1 something survived, a target is stale,
or the baseline was already failing.

WHEN A TARGET GOES STALE. "target absent" means the code was refactored and
the mutation no longer applies. That is not a false alarm — the evidence for
that property has expired. Re-point the entry at the new code and confirm it
still fails a test; do not delete it because it stopped matching.
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Mutation:
    """One broken property.

    `before` must appear in `path` exactly `expect` times, and every occurrence
    is replaced. Pinning the count is the point: if it drifts, the code moved
    and the evidence for this property needs re-confirming, which is a
    different thing from the mutation having become wrong.
    """

    name: str
    path: str
    before: str
    after: str
    expect: int = 1


@dataclass(frozen=True)
class Suite:
    """A group of mutations and the tests that must notice them.

    `watch`, when set, lists globs of the files whose change could break what
    the suite defends; `--changed-since` then skips the suite when none of them
    changed (.agent-rfc/designs/mutation-ci-scope.md). A suite without `watch`
    runs every time. Only a suite slow enough to matter declares one.
    """

    name: str
    tests: tuple[str, ...]
    mutations: tuple[Mutation, ...]
    watch: tuple[str, ...] = ()


CATALOGUE: tuple[Suite, ...] = (
    Suite(
        name="pii",
        tests=(
            "runtime/test/test_input_guardrail.py",
            "runtime/test/test_trace_redactor.py",
            "runtime/test/test_luhn_parity.py",
        ),
        mutations=(
            Mutation(
                "digits are not normalised — an Emirates ID in Arabic-Indic "
                "numerals stops being PII",
                "runtime/pii_patterns.py",
                "return text.translate(_DIGIT_TRANSLATION)",
                "return text",
            ),
            Mutation(
                "the redactor forgets the personal identifiers it was taught",
                "runtime/trace_redactor.py",
                "for pattern in (_EMIRATES_ID_HYPHEN, _EMIRATES_ID_DIGITS, _PHONE):",
                "for pattern in ():",
            ),
            Mutation(
                "the blob stores the SCRUBBED text instead of the original — the "
                "compliance guarantee becomes a no-op",
                "runtime/trace_redactor.py",
                '                        self._blob_store_for(tenant_id).put(ref, "\\n".join(strings))',
                "                        self._blob_store_for(tenant_id).put(ref, \"\\n\".join(scrubbed))",
            ),
            Mutation(
                "a blob that does not decrypt is returned as nonsense instead of raising",
                "runtime/trace_redactor.py",
                "        except Exception as exc:\n"
                "            raise RuntimeError(\n"
                '                f"HITL blob {ref!r} for tenant={self.tenant_id!r} did not decrypt "',
                "        except Exception as exc:  # noqa\n"
                "            return None  # type: ignore[return-value]\n"
                "            raise RuntimeError(\n"
                '                f"HITL blob {ref!r} for tenant={self.tenant_id!r} did not decrypt "',
            ),
            Mutation(
                "get() silently reads the local directory even when S3 is the backend",
                "runtime/trace_redactor.py",
                '        if os.environ.get("HITL_BLOB_S3_BUCKET"):\n            raise NotImplementedError(',
                '        if False:\n            raise NotImplementedError(',
            ),
            Mutation(
                "the tenant id is spliced into the key variable raw — a hyphenated "
                "tenant silently shares the fleet key again",
                "runtime/trace_redactor.py",
                '        return re.sub(r"[^A-Za-z0-9]", "_", tenant_id).upper()',
                "        return tenant_id.upper()",
            ),
            Mutation(
                "falling back to the fleet-wide HITL key stops being reported",
                "runtime/trace_redactor.py",
                "                warn_degraded_default(\n"
                '                    f"hitl-shared-key:{self.tenant_id}",',
                "                _unused = (\n"
                '                    f"hitl-shared-key:{self.tenant_id}",',
            ),
            Mutation(
                "the key value is stripped before hashing — every existing blob "
                "becomes undecryptable",
                "runtime/trace_redactor.py",
                "            return value if value and value.strip() else None",
                "            return value.strip() if value and value.strip() else None",
            ),
            Mutation(
                "the fallback tenant goes back to reading TENANT_ID raw",
                "runtime/trace_redactor.py",
                "            self.default_tenant_id = resolve_tenant_id(tenant_id)",
                '            self.default_tenant_id = tenant_id or os.environ.get("TENANT_ID", "unknown")',
            ),
            Mutation(
                "an unresolvable tenant becomes the empty string, not 'unknown'",
                "runtime/trace_redactor.py",
                '            self.default_tenant_id = "unknown"',
                '            self.default_tenant_id = ""',
            ),
        ),
    ),
    Suite(
        name="tenant_scaffold",
        tests=("runtime/test/test_cli.py",),
        mutations=(
            Mutation(
                "the tenant id goes back into the YAML unquoted — `init off` writes "
                "a boolean",
                "runtime/cli.py",
                "  id: {quoted_id}\n  name: {quoted_id}",
                "  id: {tenant_id}\n  name: {tenant_id}",
            ),
            Mutation(
                "any tenant id is accepted again",
                "runtime/cli.py",
                "    if not TENANT_ID_PATTERN.match(tenant_id):",
                "    if False:",
            ),
            Mutation(
                "an empty tenant id is accepted",
                "runtime/cli.py",
                "    if not isinstance(tenant_id, str) or not tenant_id.strip():",
                "    if False:",
            ),
            Mutation(
                "validation moves back after the first mkdir, leaving a partial "
                "scaffold behind",
                "runtime/cli.py",
                "    validate_tenant_id(tenant_id)\n    if isolation not in ISOLATIONS:",
                "    if isolation not in ISOLATIONS:",
            ),
            Mutation(
                "the declared version is a hardcoded literal again",
                "runtime/cli.py",
                "  version: \"{framework_version or _default_framework_version()}\"",
                '  version: "1.3.0"',
            ),
            Mutation(
                "the source-checkout marker leaks into the tenant's declared version",
                "runtime/cli.py",
                "    return version[: -len(SOURCE_SUFFIX)] if version.endswith(SOURCE_SUFFIX) else version",
                "    return version",
            ),
        ),
    ),
    Suite(
        name="testing_double",
        tests=(
            "runtime/test/test_testing_double_parity.py",
            "runtime/test/test_prompt_identity.py",
        ),
        mutations=(
            Mutation(
                "the double goes back to its own inline flattening — a multimodal "
                "prompt raises TypeError out of FakeGateway",
                "runtime/testing.py",
                "            content_text(m.get(\"content\")) for m in prompt if isinstance(m, dict)",
                "            m.get(\"content\", \"\") for m in prompt if isinstance(m, dict)",
            ),
            Mutation(
                "content_text stops flattening typed parts",
                "runtime/prompt_identity.py",
                "    if isinstance(content, list):",
                "    if False:",
            ),
            Mutation(
                "None content stops being empty text",
                "runtime/prompt_identity.py",
                '    return "" if content is None else str(content)',
                "    return str(content)",
            ),
        ),
    ),
    Suite(
        name="idempotency_key",
        tests=("runtime/test/test_idempotency_key.py",),
        mutations=(
            Mutation(
                "default=str returns — a set of strings keys differently in every "
                "process and the crash-retry pays twice",
                "runtime/idempotency.py",
                "    canonical = json.dumps(payload, sort_keys=True, default=_canonical)",
                "    canonical = json.dumps(payload, sort_keys=True, default=str)",
            ),
            Mutation(
                "sets stop being sorted — iteration order decides the key again",
                "runtime/idempotency.py",
                "        return sorted(\n"
                "            value, key=lambda item: json.dumps(item, sort_keys=True, default=_canonical)\n"
                "        )",
                "        return list(value)",
            ),
            Mutation(
                "sort_keys goes away — dict insertion order decides the key",
                "runtime/idempotency.py",
                "    canonical = json.dumps(payload, sort_keys=True, default=_canonical)",
                "    canonical = json.dumps(payload, sort_keys=False, default=_canonical)",
            ),
            Mutation(
                "an unstable payload is accepted instead of refused",
                "runtime/idempotency.py",
                "    raise UnstableIdempotencyKey(",
                "    return str(value)\n    raise UnstableIdempotencyKey(",
            ),
        ),
    ),
    Suite(
        name="replay_webhook",
        tests=("runtime/test/test_replay_webhook_signature.py",),
        mutations=(
            Mutation(
                "the tolerance window is removed — a captured request is valid forever",
                "runtime/replay_webhook_server.py",
                "    if drift > SIGNATURE_TOLERANCE_SECONDS:",
                "    if False:",
            ),
            Mutation(
                "a non-finite timestamp skips the window again (NaN beats every "
                "comparison)",
                "runtime/replay_webhook_server.py",
                "    if not math.isfinite(sent_at):\n        return False, \"malformed timestamp\"",
                "    if False:\n        return False, \"malformed timestamp\"",
            ),
            Mutation(
                "the timestamp leaves the signed material — replaying a body with a "
                "fresh timestamp works",
                "runtime/replay_webhook_server.py",
                '    signed = timestamp_header.encode() + b"." + body',
                "    signed = body",
            ),
            Mutation(
                "the tolerance is widened to a day",
                "runtime/replay_webhook_server.py",
                "SIGNATURE_TOLERANCE_SECONDS = 300",
                "SIGNATURE_TOLERANCE_SECONDS = 86400",
            ),
            Mutation(
                "signature comparison stops being constant-time",
                "runtime/replay_webhook_server.py",
                "    if not hmac.compare_digest(expected, signature_header[len(\"sha256=\") :]):",
                '    if expected != signature_header[len("sha256=") :]:',
            ),
        ),
    ),
    Suite(
        name="prompt_guard",
        tests=(
            "runtime/test/test_prompt_guard.py",
            "scripts/test/test_security_prompt_guard_enforcement.py",
        ),
        mutations=(
            Mutation(
                "matching goes back to raw text — every Unicode evasion works again",
                "runtime/prompt_guard.py",
                "probe = _normalise(text)",
                "probe = text",
            ),
            Mutation(
                "separators accept whitespace only — 'ignore-all-previous' passes",
                "runtime/prompt_guard.py",
                r"[\s\-_]+",
                r"\s+",
                # One per pattern in the module; all of them have to go, or the
                # remaining ones still catch the hyphenated form.
                expect=10,
            ),
            Mutation(
                "ZWJ/ZWNJ count as padding — ordinary Persian and emoji trip the guard",
                "runtime/prompt_guard.py",
                '_LEGITIMATE_FORMAT_CHARS = {"\\u200c", "\\u200d"}',
                "_LEGITIMATE_FORMAT_CHARS = set()",
            ),
        ),
    ),
    Suite(
        name="structured_output",
        tests=("runtime/test/test_structured_output.py",),
        mutations=(
            Mutation(
                "an untagged fence outranks an explicit ```json one",
                "runtime/structured_output.py",
                "    out.extend(tagged)\n    out.extend(untagged)\n    out.extend(other)",
                "    out.extend(untagged)\n    out.extend(tagged)\n    out.extend(other)",
            ),
            Mutation(
                "bare spans stop being ordered by opener — `[{...}]` loses its brackets",
                "runtime/structured_output.py",
                "sorted(spans, key=lambda item: item[0])",
                "spans",
            ),
            Mutation(
                "the first candidate wins without being parsed",
                "runtime/structured_output.py",
                "    for label, candidate in candidates:\n"
                "        if _is_json(candidate):\n"
                "            return label, candidate\n"
                "    return candidates[0]",
                "    return candidates[0]",
            ),
            Mutation(
                "_is_json calls everything JSON",
                "runtime/structured_output.py",
                "    except (json.JSONDecodeError, ValueError):\n        return False",
                "    except (json.JSONDecodeError, ValueError):\n        return True",
            ),
            Mutation(
                "every fence tag counts as json — ```python hijacks extraction",
                "runtime/structured_output.py",
                "        if tag in _JSON_TAGS:",
                "        if True:",
            ),
            Mutation(
                "the fence tag stops being captured",
                "runtime/structured_output.py",
                r'r"```([A-Za-z0-9_+.-]*)[ \t]*\r?\n?(.*?)```"',
                r'r"```(?:json|JSON)?()[ \t]*\r?\n?(.*?)```"',
            ),
            Mutation(
                "an empty fence becomes a candidate and gets blamed for the failure",
                "runtime/structured_output.py",
                "        if not body:\n            continue",
                "        if False:\n            continue",
            ),
            Mutation(
                "a parse failure stops naming the block it tried",
                "runtime/structured_output.py",
                'f"invalid JSON in {origin}: {exc}"',
                'f"invalid JSON: {exc}"',
            ),
            Mutation(
                "a schema failure stops naming the block it tried",
                "runtime/structured_output.py",
                'f"schema validation failed for {origin}: {exc}"',
                'f"schema validation failed: {exc}"',
            ),
            Mutation(
                "the error quotes the model's payload into the log",
                "runtime/structured_output.py",
                'f"invalid JSON in {origin}: {exc}"',
                'f"invalid JSON in {candidate}: {exc}"',
            ),
        ),
    ),
    Suite(
        name="circuit_breaker",
        tests=(
            "scripts/test/test_circuit_breaker.py",
            "scripts/test/test_breaker_fail_open.py",
            "scripts/test/test_cost_router.py",
        ),
        mutations=(
            Mutation(
                "the burst limit goes back to a bare int() — an empty var kills the import",
                "scripts/circuit_breaker.py",
                'BURST_TOKEN_LIMIT = env_number("AGENT_BURST_TOKEN_LIMIT", 50_000, cast=int)',
                'BURST_TOKEN_LIMIT = int(os.environ.get("AGENT_BURST_TOKEN_LIMIT", "50000"))',
            ),
            Mutation(
                "the monthly cap goes back to a bare float()",
                "scripts/circuit_breaker.py",
                'MONTHLY_USD_CAP = env_number("AGENT_MONTHLY_USD_CAP", 150.0)',
                'MONTHLY_USD_CAP = float(os.environ.get("AGENT_MONTHLY_USD_CAP", "150.0"))',
            ),
            Mutation(
                "env_number stops treating a declared-but-empty var as unset",
                "scripts/_shared.py",
                '    raw = (os.environ.get(var) or "").strip()\n    if not raw:\n        return default',
                "    raw = os.environ.get(var, str(default))",
            ),
            Mutation(
                "a malformed limit falls back silently",
                "scripts/_shared.py",
                '        print(\n'
                '            f"⚠️  {var}={raw!r} is not a number — using {default}",\n'
                '            file=sys.stderr,\n'
                '        )\n'
                '        return default',
                "        return default",
            ),
            Mutation(
                "the breaker import moves back inside the one try — the fail-open "
                "handler raises UnboundLocalError",
                "scripts/agent_logger.py",
                "        try:\n"
                "            from circuit_breaker import (\n"
                "                CircuitBreakerTripped,\n"
                "                audit_token_velocity_circuit,\n"
                "            )\n"
                "        except Exception as exc:",
                '        try:\n'
                '            from circuit_breaker import (\n'
                '                CircuitBreakerTripped,\n'
                '                audit_token_velocity_circuit,\n'
                '            )\n'
                '            audit_token_velocity_circuit(input_tokens, output_tokens)\n'
                '        except CircuitBreakerTripped as tripped:\n'
                '            print(f"[agent_logger] {tripped}", file=sys.stderr)\n'
                '        except Exception as exc:',
            ),
        ),
    ),
    Suite(
        name="conversation_memory",
        tests=("runtime/test/test_memory_and_vector.py",),
        mutations=(
            Mutation(
                "nothing is protected — eviction deletes the system prompt first",
                "runtime/conversation_memory.py",
                'PROTECTED_ROLES = frozenset({"system"})',
                "PROTECTED_ROLES = frozenset()",
            ),
            Mutation(
                "the guard swallows the budget — nothing is ever evicted",
                "runtime/conversation_memory.py",
                "            if len(evictable) <= 1:\n                break",
                "            if len(evictable) <= 99:\n                break",
            ),
            Mutation(
                "the estimate goes back to chars//4 — Arabic reads 65% low",
                "runtime/conversation_memory.py",
                "    return max(1, int(alnum / 4 + symbols * 0.6 + non_ascii * 1.2))",
                "    return max(1, len(text) // 4)",
            ),
            Mutation(
                "the buffer stops saying it cannot shrink",
                "runtime/conversation_memory.py",
                "        if total > self.token_budget:\n            logger.warning(",
                "        if False:\n            logger.warning(",
            ),
        ),
    ),
    Suite(
        name="gate_binary_files",
        tests=(
            "scripts/test/test_gate_kg.py::"
            "test_a_commit_carrying_a_binary_file_is_judged_by_both_gates_not_crashed_on",
            "scripts/test/test_process_gate.py::"
            "test_the_working_tree_reader_returns_a_binary_file_rather_than_crashing",
        ),
        mutations=(
            Mutation(
                "the index and commit reader decodes strictly again — a commit carrying an image "
                "crashes the gate and CI",
                "scripts/process_gate.py",
                '        result = subprocess.run(["git", "show", f"{rev}:{path}"], cwd=root, '
                'capture_output=True, check=False)\n'
                '        return result.stdout.decode("utf-8", errors="replace") '
                'if result.returncode == 0 else None',
                '        result = subprocess.run(["git", "show", f"{rev}:{path}"], cwd=root, '
                'capture_output=True, text=True, '
                'check=False)\n'
                "        return result.stdout if result.returncode == 0 else None",
            ),
            Mutation(
                "the working-tree reader decodes strictly again — the stop gate crashes on an image",
                "scripts/process_gate.py",
                '        return target.read_bytes().decode("utf-8", errors="replace") if target.is_file() else None',
                '        return target.read_text(encoding="utf-8") if target.is_file() else None',
            ),
        ),
    ),
    Suite(
        name="send_dev_record",
        tests=("scripts/test/test_send_dev_record.py",),
        mutations=(
            Mutation(
                "redirects are followed — and urllib carries the ingest token to wherever they point",
                "scripts/send_dev_record.py",
                "_OPENER = urllib.request.build_opener(_NoRedirect)",
                "_OPENER = urllib.request.build_opener()",
            ),
            Mutation(
                "a redirect reads as a warning, so a green build hides a misconfigured address",
                "scripts/send_dev_record.py",
                "                          \"with it. Set AGENTSMITH_PORTAL_URL to the portal's final address.\")\n"
                "            return 1",
                "                          \"with it. Set AGENTSMITH_PORTAL_URL to the portal's final address.\")\n"
                "            return 0",
            ),
        ),
    ),
    Suite(
        name="intake",
        tests=("runtime/test/test_intake.py",),
        mutations=(
            Mutation(
                "the token goes to any address, plain http included",
                "runtime/intake.py",
                '    return parsed.scheme == "https" or '
                '(parsed.scheme == "http" and parsed.hostname in ("localhost", "127.0.0.1"))',
                "    return True",
            ),
            Mutation(
                "redirects are followed — and urllib carries the Authorization header with them",
                "runtime/intake.py",
                "_OPENER = urllib.request.build_opener(_NoRedirect)",
                "_OPENER = urllib.request.build_opener()",
            ),
            Mutation(
                "the token is in the Intake's repr, so a traceback or a debug print shows it",
                "runtime/intake.py",
                "    _token: str = field(repr=False)",
                '    _token: str = ""',
            ),
            Mutation(
                "an intake id reaches the URL unchecked — `--from ../admin` becomes a path",
                "runtime/intake.py",
                "    if not INTAKE_ID.match(intake_id):",
                "    if False:",
            ),
            Mutation(
                "a tenant id the portal could never register is scaffolded anyway",
                "runtime/intake.py",
                "    if not APP_ID.match(tenant_id):",
                "    if False:",
            ),
            Mutation(
                "the stack is trusted because the portal accepted it",
                "runtime/intake.py",
                '    if record["stack"] not in STACKS:',
                "    if False:",
            ),
            Mutation(
                "fields the CLI does not know are accepted silently",
                "runtime/intake.py",
                "    if unknown or missing:",
                "    if missing:",
            ),
            Mutation(
                "a newline in an acceptance criterion starts a heading of its own in the RFC",
                "runtime/intake.py",
                '    return [" ".join(_clean(v, LIMITS["item"], field).split()) for v in value]',
                '    return [_clean(v, LIMITS["item"], field) for v in value]',
            ),
            Mutation(
                "a portal that is down reads as something the author must change, not a retry",
                "runtime/intake.py",
                "        if exc.code >= 500:",
                "        if False:",
            ),
            Mutation(
                "an intake is consumed though its RFC never landed — the author's text is lost",
                "runtime/cli.py",
                '    if not (landed.is_file() and landed.read_text(encoding="utf-8") == expected):',
                "    if False:",
            ),
            Mutation(
                "--from and a flag it decides are both accepted, and one silently wins",
                "runtime/cli.py",
                "        if args.tenant_id is not None or given:",
                "        if False:",
            ),
        ),
    ),
    Suite(
        name="chosen_ides",
        tests=("runtime/test/test_chosen_ides.py",),
        mutations=(
            Mutation(
                "an unreadable declaration wires NO editor instead of every one — a "
                "typo in tenant.yaml silently leaves the pre-edit gate unconsulted",
                "runtime/config.py",
                "    if not isinstance(declared, list):\n        return every",
                "    if not isinstance(declared, list):\n        return ()",
            ),
            Mutation(
                "a declaration naming nothing available returns empty rather than "
                "falling back — `ides: [gemini]` disarms both verified editors",
                "runtime/config.py",
                "    return tuple(ide for ide in every if ide in declared) or every",
                "    return tuple(ide for ide in every if ide in declared)",
            ),
            Mutation(
                "the declaration is read from the cache again, so the file `tenant "
                "init` writes mid-run is not seen and the choice is discarded",
                "runtime/config.py",
                'tenant_config(root, refresh=True).get("workspace")',
                'tenant_config(root).get("workspace")',
            ),
            Mutation(
                "the file's own strings reach the path instead of the framework's "
                "constants — `ides` becomes an arbitrary path source",
                "runtime/config.py",
                "    return tuple(ide for ide in every if ide in declared) or every",
                "    return tuple(declared) or every",
            ),
        ),
    ),
    Suite(
        name="config_choices",
        tests=("runtime/test/test_config_choices.py",),
        mutations=(
            Mutation(
                "a YAML boolean is no longer noticed — a bare `off` is discarded "
                "in silence again",
                "runtime/config.py",
                "    if isinstance(raw, bool):",
                "    if False:",
            ),
            Mutation(
                "the boolean is translated to a word — `prompt_guard: false` "
                "disables the guard",
                "runtime/config.py",
                '            f"Using {fallback!r}; accepted: {\', \'.join(options)}.",\n'
                "        )\n"
                "        return fallback",
                '            f"Using {fallback!r}; accepted: {\', \'.join(options)}.",\n'
                "        )\n"
                '        return "off"',
            ),
            Mutation(
                "an unrecognised value goes back to being replaced silently",
                "runtime/config.py",
                '    warn_once(\n        f"config-choice-unknown:{dotted}",',
                '    _unused = (\n        f"config-choice-unknown:{dotted}",',
            ),
            Mutation(
                "unset starts warning too — the signal drowns in noise",
                "runtime/config.py",
                "    if not text:\n        return fallback",
                "    if False:\n        return fallback",
            ),
            Mutation(
                "the input guard's development fallback becomes the production one",
                "runtime/input_guardrail.py",
                '    fallback = "off" if get_environment() == "development" else "default"',
                '    fallback = "default"',
            ),
        ),
    ),
    Suite(
        name="degraded_defaults",
        tests=(
            "runtime/test/test_degraded_defaults.py",
            "runtime/test/test_memory_and_vector.py",
            "runtime/test/test_llm_gateway_budget.py",
        ),
        mutations=(
            Mutation(
                "an empty selector stops meaning unset — BUDGET_BACKEND='' crashes again",
                "runtime/environment.py",
                '    raw = os.environ.get(var, "").strip().lower()\n    if not raw:\n        return default',
                "    raw = os.environ.get(var, default).lower()",
            ),
            Mutation(
                "a typo'd backend silently resolves to the default",
                "runtime/environment.py",
                "    if raw not in options:\n        raise ValueError(",
                "    if False:\n        raise ValueError(",
            ),
            Mutation(
                # Re-pointed when warn_once was extracted: the level choice moved
                # but the property it defends did not.
                "a degraded default is logged at the same level everywhere",
                "runtime/environment.py",
                '    level = logging.ERROR if environment in {"staging", "production"} '
                "else logging.INFO",
                "    level = logging.INFO",
            ),
            Mutation(
                "one degraded default silences the others",
                "runtime/environment.py",
                "    if key in _degraded_warned:\n        return\n    _degraded_warned.add(key)",
                "    if _degraded_warned:\n        return\n    _degraded_warned.add(key)",
            ),
            Mutation(
                "the warning repeats on every call",
                "runtime/environment.py",
                "    if key in _degraded_warned:\n        return\n    _degraded_warned.add(key)",
                "    _degraded_warned.add(key)",
            ),
            Mutation(
                "the per-worker spend cap stops announcing itself",
                "runtime/llm_gateway.py",
                '    warn_degraded_default(\n        "budget-backend-memory",',
                '    _unused = (\n        "budget-backend-memory",',
            ),
            Mutation(
                "the in-process vector index stops announcing itself",
                "runtime/vector_store.py",
                '    warn_degraded_default(\n        "vector-backend-memory",',
                '    _unused = (\n        "vector-backend-memory",',
            ),
            Mutation(
                "VECTOR_BACKEND aliases are narrowed — a tenant on =pgvector breaks",
                "runtime/vector_store.py",
                '_VECTOR_ALIASES = ("memory", "mem", "inmemory", "postgres", "pgvector", "pg")',
                '_VECTOR_ALIASES = ("memory", "postgres")',
            ),
            Mutation(
                "the fake embedder stops announcing itself",
                "runtime/embeddings.py",
                '    warn_degraded_default(\n        "embedder-hash",',
                '    _unused = (\n        "embedder-hash",',
            ),
            Mutation(
                "the embedder identity collapses to its dimension",
                "runtime/embeddings.py",
                '        return f"hash:{self.dim}"',
                '        return f"{self.dim}"',
            ),
            Mutation(
                "the retrieval span stops naming the embedder",
                "runtime/vector_store.py",
                '        if embedder:\n            span.set_attribute("agent.retrieval.embedder", embedder)',
                "        pass",
            ),
            Mutation(
                "_identity_of raises on an embedder that predates `identity`",
                "runtime/vector_store.py",
                "    try:\n        value = embedder.identity\n    except Exception:\n        return None",
                "    value = embedder.identity",
            ),
        ),
    ),
    # .agent-rfc/designs/tenant-adopt.md — an existing repository under the
    # gates, the hooks it ran kept behind them, and `tenant init` vendoring.
    # Its tests build real repositories through real hooks, about twelve
    # minutes in CI, so it runs when what it protects changes
    # (.agent-rfc/designs/mutation-ci-scope.md).
    Suite(
        name="tenant_adopt",
        tests=(
            "scripts/test/test_hook_chain.py",
            "scripts/test/test_hook_visibility_override.py",
            "scripts/test/test_vouched_files.py",
            "scripts/test/test_tenant_adopt.py",
            "scripts/test/test_scaffold_review.py",
            "scripts/test/test_installed_runtime_tenant.py",
        ),
        watch=(
            ".githooks/*",
            "hooks/*",
            "runtime/adopt.py",
            "runtime/cli.py",
            "runtime/architectures.py",
            "templates/architectures.yaml",
            "templates/governance.json",
            "scripts/process_gate.py",
            "scripts/gate_*.py",
            "scripts/generate-ide-config.py",
            "scripts/vouched_files.py",
            "scripts/rules_port.py",
            "workflow-templates/agentsmith-gates.yml",
            "scripts/test/test_hook_chain.py",
            "scripts/test/test_hook_visibility_override.py",
            "scripts/test/test_vouched_files.py",
            "scripts/test/test_tenant_adopt.py",
            "scripts/test/test_scaffold_review.py",
            "scripts/test/test_installed_runtime_tenant.py",
            "scripts/test/test_scratch_tenants.py",
            "scripts/mutation_check.py",
        ),
        mutations=(
            Mutation(
                "the vouched note counts the manifest it never checked against itself",
                "scripts/process_gate.py",
                "    checked = [path for path in gated if path != SCAFFOLD_MANIFEST]",
                "    checked = gated",
            ),
            Mutation(
                "the chain runs the prior hook even when it points back at .githooks — it calls itself forever",
                ".githooks/chain",
                '[ "$(cd "$dir" 2>/dev/null && pwd -P)" != "$here" ] || exit 0',
                "true",
            ),
            Mutation(
                # Not `|| exit $?` removed: the hook runs under `set -e`, so that
                # mutation is equivalent and survived. Running the chain first is
                # the real break.
                "the prior commit-msg runs before the gate, even for a message the gate refuses",
                ".githooks/commit-msg",
                '"$hook_dir/process-gate" commit-msg "${amend[@]+"${amend[@]}"}" "$msg_file" || exit $?',
                'bash "$hook_dir/chain" commit-msg "$msg_file"\n'
                '"$hook_dir/process-gate" commit-msg "${amend[@]+"${amend[@]}"}" "$msg_file" || exit $?',
            ),
            Mutation(
                "pre-push stops chaining — the repository's own pre-push silently stops running",
                ".githooks/pre-push",
                'exec bash "$hook_dir/chain" pre-push "$@"',
                "exit 0",
            ),
            Mutation(
                "any commit may claim to be the one that arms the gates",
                "scripts/process_gate.py",
                "    if not arming:\n",
                "    if False:\n",
            ),
            Mutation(
                "only a root commit arms the gates again — an adoption commit is refused",
                "scripts/process_gate.py",
                "arming=previous is None or previous(CONFIG) is None)",
                "arming=previous is None)",
            ),
            Mutation(
                "a vendored gate looks for @framework/ files only beside itself",
                "scripts/process_gate.py",
                "    roots.append(Path.home() / \".agent-framework\")\n    return next(",
                "    roots = roots[:1]\n    return next(",
            ),
            Mutation(
                "adopt chains the machine's post-checkout and post-commit — they vendor into the repository",
                "runtime/adopt.py",
                "prior=plan.prior_hooks, provisioning=False)",
                "prior=plan.prior_hooks, provisioning=True)",
            ),
            Mutation(
                "the rules block is appended again on every run",
                "scripts/rules_port.py",
                "    if _BLOCK.search(existing):",
                "    if False:",
            ),
            Mutation(
                "adopt replaces the repository's Claude settings instead of merging into them",
                "runtime/adopt.py",
                "        put(rel, json.dumps(gate_ides.render_config(ide, existing), indent=2) + \"\\n\")",
                "        put(rel, json.dumps(gate_ides.render_config(ide, None), indent=2) + \"\\n\")",
            ),
            Mutation(
                "an explicit autopush setting is overwritten",
                "runtime/adopt.py",
                'and not _git(root, "config", "--get", "agentsmith.autopush"):',
                ":",
            ),
            Mutation(
                "tenant init no longer runs the machine's post-checkout before its first commit",
                "runtime/cli.py",
                "    written += _vendor(root, prior)\n",
                "",
            ),
            Mutation(
                "a vouch survives a tenant editing the file — the skip becomes a path "
                "allowlist instead of a hash match",
                "scripts/vouched_files.py",
                "if path in blobs and hashlib.sha256(blobs[path]).hexdigest() == recorded[path]",
                "if path in blobs",
            ),
            Mutation(
                "a manifest that is not a mapping is trusted anyway",
                "scripts/vouched_files.py",
                "        if not isinstance(recorded, dict):\n            return []",
                "        if False:\n            return []",
            ),
            Mutation(
                "a declared AGENTSMITH_TENANT_VISIBILITY loses to detection — a public "
                "fixture stops tracking the IDE configs its tenants track",
                "hooks/post-checkout",
                'if [ -z "$VIS_SOURCE" ] && echo "$REMOTE_URL" | grep -q "github.com"; then',
                'if echo "$REMOTE_URL" | grep -q "github.com"; then',
            ),
            Mutation(
                "a misspelt AGENTSMITH_TENANT_VISIBILITY is silently read as private",
                "hooks/post-checkout",
                '  *) echo "⚠️  AGENTSMITH_TENANT_VISIBILITY=',
                '  *) VIS_SOURCE=declared; echo "',
            ),
            Mutation(
                "the machine's post-checkout vendors into an adopted repository",
                "hooks/post-checkout",
                "   && grep -qs '\"generated_by\": \"agentsmith tenant adopt\"' "
                "\"$REPO_ROOT/.agenticframework/scaffold.json\"; then",
                "   && false; then",
            ),
            Mutation(
                "an unfinished adoption is told it is already under the gates",
                "runtime/adopt.py",
                "        if committed.returncode == 0:",
                "        if True:",
            ),
            Mutation(
                "a rule file AgentSmith generated gets the same rules appended again",
                "scripts/rules_port.py",
                '        placement = "whole" if existing is None or _ours(rel, existing, titles) else "block"',
                '        placement = "whole" if existing is None else "block"',
            ),
            Mutation(
                "tenant init runs any post-checkout and vouches for what it wrote",
                "runtime/cli.py",
                " or not is_machine_hook(hook):",
                ":",
            ),
            Mutation(
                "a malformed settings file is found only after adopt has started writing",
                "runtime/adopt.py",
                "    for rel in (\".claude/settings.json\", \".cursor/hooks.json\"):\n"
                "        if (root / rel).is_file():",
                "    for rel in ():\n        if (root / rel).is_file():",
            ),
            Mutation(
                "a --force re-run stops vouching for what an earlier run wrote",
                "runtime/cli.py",
                '            earlier = json.loads(manifest_path.read_text(encoding="utf-8")).get("files") or {}',
                "            earlier = {}",
            ),
            Mutation(
                "adopt gates the vendored framework code it finds",
                "runtime/adopt.py",
                "and head not in vendored:",
                "and head not in ():",
            ),
            Mutation(
                "adopt writes over a repository even when the gates cannot be armed",
                "runtime/adopt.py",
                "    missing = missing_gate_hooks(framework)\n    if missing:",
                "    missing = []\n    if missing:",
            ),
            Mutation(
                "the gate's note stops naming the command that wrote the files",
                "scripts/process_gate.py",
                'as `{_manifest_author(read)}` wrote them")',
                'as `agentsmith tenant init` wrote them")',
            ),
        ),
    ),
    # .agent-rfc/designs/installed-architectures.md — what an installed machine
    # (not a checkout) must carry for `tenant init` / `tenant adopt` to work.
    Suite(
        name="installed_machine",
        tests=(
            "scripts/test/test_installer_templates.py",
            "runtime/test/test_cli.py",
        ),
        mutations=(
            Mutation(
                "the installer stops shipping the gate's hooks — every tenant is armed at an empty directory",
                "install-ai-stack.sh",
                'cp -r "$INSTALLER_DIR/.githooks/." "$GITHOOKS_DIR/"',
                'true "$INSTALLER_DIR"',
            ),
            Mutation(
                "the installer stops shipping the architecture catalogue",
                "install-ai-stack.sh",
                'cp "$INSTALLER_DIR/templates/architectures.yaml" "$FRAMEWORK_DIR/templates/architectures.yaml"',
                "true",
            ),
            Mutation(
                "an install with no gate hooks arms core.hooksPath anyway",
                "runtime/cli.py",
                "    missing_gate_hooks(framework, raising=True)",
                "    missing_gate_hooks(framework)",
            ),
            Mutation(
                "the release stops shipping the gate's hooks",
                ".github/workflows/release.yml",
                "tar -czf dist/githooks.tar.gz -C .githooks .",
                "true",
            ),
        ),
    ),

    # .agent-rfc/designs/gate-port.md — the contract a provider satisfies, and
    # the suite that proves it. Fast: the fixture is five events in a temp repo.
    Suite(
        name="gate_contract",
        tests=(
            "scripts/test/test_gate_contract.py",
            "runtime/test/test_conformance.py",
            "scripts/test/test_provider_resolution.py",
            "scripts/test/test_tenant_adopt.py",
        ),
        mutations=(
            Mutation(
                "the adapter goes back to saying `allow` by staying silent",
                "runtime/cli.py",
                '    if done.returncode == 0 and not done.stdout.strip():\n'
                '        print(json.dumps({"decision": "allow", "text": ""}))\n'
                "        return 0\n",
                "",
            ),
            Mutation(
                "the neutral profile answers in an IDE's dialect",
                "scripts/gate_ides.py",
                "    return gm.Decision(decision=\"block\" if decision == \"block\" else decision, "
                "text=text).model_dump_json()",
                "    return _claude_render(decision, text, repeat)",
            ),
            Mutation(
                "a refusal with no reason passes conformance",
                "runtime/conformance.py",
                '    if decision in ("deny", "block") and not text.strip():',
                "    if False:",
            ),
            Mutation(
                "a provider that printed no decision is taken as having answered — a provider too old "
                "for the event leaves the repository ungated",
                ".githooks/process-gate",
                '      if [ -n "$answer" ]; then',
                "      if true; then",
            ),
            Mutation(
                "a repository that declared itself ungoverned falls back to the framework anyway",
                ".githooks/process-gate",
                '    if [ "$provider" = "none" ]; then',
                "    if false; then",
            ),
            Mutation(
                "the declaration is ignored and the framework's own paths always win",
                ".githooks/process-gate",
                '    provider="${GOVERNANCE_PROVIDER:-$(declared_gate)}"',
                '    provider=""',
            ),
            Mutation(
                "adopt stops declaring who governs the repository",
                "runtime/adopt.py",
                "    if not (root / PROVIDERS).exists():\n"
                "        put(PROVIDERS, providers_declaration(setup=setup_reference(plan.framework_ref)))",
                "    pass",
            ),
            Mutation(
                "a provider that cannot run is scored as a wrong answer",
                "runtime/conformance.py",
                "    if code == CANNOT_RUN:",
                "    if False:",
            ),
        ),
    ),

    # .agent-rfc/designs/framework-sync.md — one command keeps a tenant current,
    # and the tenant's own gates accept the commit it prints.
    # .agent-rfc/designs/gate-contract-ci.md — gate contract 2: a tenant's CI
    # asks its declared provider, and a check with no decision never falls back.
    Suite(
        name="gate_contract_v2",
        tests=(
            "scripts/test/test_gate_contract_v2.py",
            "scripts/test/test_gate_contract.py",
        ),
        mutations=(
            Mutation(
                "ci at contract 2 falls back to the framework's own gate when the provider gives no decision",
                ".githooks/process-gate",
                'if [ "$(declared_contract)" -ge 2 ]; then',
                "if false; then",
            ),
            Mutation(
                "the launcher reads every declaration as contract 1",
                ".githooks/process-gate",
                '    *) echo "$n" ;;',
                "    *) echo 1 ;;",
            ),
            Mutation(
                "a range the provider refused passes CI",
                ".githooks/process-gate",
                'if decision == "allow":',
                'if decision in ("allow", "deny"):',
            ),
            Mutation(
                "anything is put into the range event as a ref",
                ".githooks/process-gate",
                "      *[!A-Za-z0-9._/~^@{}-]*)",
                "      __never_a_ref__)",
            ),
            Mutation(
                "a portal that refuses the record no longer fails the answer — a wrong token hides behind green",
                "scripts/process_gate.py",
                'decision="deny" if code or sent else "allow"',
                'decision="deny" if code else "allow"',
            ),
            Mutation(
                "a ci event that is not a range is judged anyway",
                "scripts/process_gate.py",
                '                if event.kind != "range":',
                "                if False:",
            ),
            Mutation(
                "the declaration is governed only when a config lists it — an unreviewed `none` turns CI off",
                "scripts/process_gate.py",
                "        return path in ALWAYS_GOVERNED or self.lists(path)",
                "        return self.lists(path)",
            ),
        ),
    ),
    # .agent-rfc/designs/record-contract.md — the record a gate provider sends a
    # portal: one model, both sides held to it, and conformance that can fail.
    Suite(
        name="telemetry_contract",
        tests=("scripts/test/test_telemetry_contract.py",),
        mutations=(
            Mutation(
                "an emitter that exported nothing is judged conformant",
                "runtime/telemetry_contract.py",
                '    checks.append(Check("something was exported", bool(spans or metrics),',
                '    checks.append(Check("something was exported", True,',
            ),
            Mutation(
                "spans inside a run are never asked for their identity",
                "runtime/telemetry_contract.py",
                "    unidentified = [f\"{s.name} lacks {e.name}\" for s in spans if s.span_id in inside",
                "    unidentified = [f\"{s.name} lacks {e.name}\" for s in spans if False",
            ),
            Mutation(
                "a span is inside a run only if it carries run.id itself — children escape",
                "runtime/telemetry_contract.py",
                "            node = by_id.get(node.parent_id)",
                "            node = None",
            ),
            Mutation(
                "conditional attributes are never checked",
                "runtime/telemetry_contract.py",
                "            holds, present = _holds(entry.when, span.attributes), entry.name in span.attributes",
                "            holds = present = True",
            ),
            Mutation(
                "a catalogued attribute of the wrong type passes",
                "runtime/telemetry_contract.py",
                "                if not _typed(known.type, kind):",
                "                if False:",
            ),
            Mutation(
                "any name under any family counts as catalogued",
                "runtime/telemetry_contract.py",
                "        families = [e for e in self.entries if e.where == where and e.family "
                "and name.startswith(e.name)]",
                "        families = [e for e in self.entries if e.where == where and e.family]",
            ),
            Mutation(
                "the receiver reads a body of any size",
                "runtime/telemetry_contract.py",
                "                if signal is None or length <= 0 or length > MAX_BODY_BYTES:",
                "                if signal is None or length <= 0:",
            ),
            Mutation(
                "the CI setup shim answers with a vendored tenant's own CLI",
                ".github/actions/setup-agentsmith/action.yml",
                'exec "%s" -P -m runtime.cli',
                'exec "%s" -m runtime.cli',
            ),
            Mutation(
                "the weekly sync runs a vendored tenant's own sync",
                "workflow-templates/agentsmith-sync.yml",
                "run: python3 -P -m runtime.cli sync --yes",
                "run: python3 -m runtime.cli sync --yes",
            ),
            Mutation(
                "the runtime library stops naming the contract it speaks",
                "runtime/tracing.py",
                '        "governance.telemetry.contract": 1,\n',
                "",
            ),
        ),
    ),
    Suite(
        name="rules_contract",
        tests=("scripts/test/test_rules_contract.py",),
        mutations=(
            Mutation(
                "a check passes a block whatever its region says",
                "scripts/rules_port.py",
                '    return "current" if held is not None and held == file.text.rstrip() else "drifted"',
                '    return "current" if held is not None else "drifted"',
            ),
            Mutation(
                "the render reads the owner from the environment — CI and a developer disagree",
                "scripts/generate-ide-config.py",
                '        "owner_id": declared.get("tenant.owner") or "unknown@unknown",',
                '        "owner_id": __import__("os").environ.get("AGENT_OWNER_ID") or declared.get("tenant.owner") '
                'or "unknown@unknown",',
            ),
            Mutation(
                "a rules provider may write the repository's hooks",
                "scripts/gate_models.py",
                'RULES_FORBIDDEN = (".git", ".githooks", ".agenticframework", ".github/workflows", ".github/actions")',
                'RULES_FORBIDDEN = (".git", ".agenticframework", ".github/workflows", ".github/actions")',
            ),
            Mutation(
                "the caller writes what a provider renders without checking a path",
                "runtime/adopt.py",
                "        return port.gm.RulesRender.model_validate_json(done.stdout)",
                "        return port.gm.RulesRender.model_construct(files=[port.gm.RulesFile.model_construct(**f) "
                "for f in json.loads(done.stdout)['files']])",
            ),
            Mutation(
                "a declared rules provider that gives no answer passes the CI step",
                ".githooks/process-gate",
                "    ask_provider check '{}' \"$where\" rules\n    exit $?",
                "    ask_provider check '{}' \"$where\" rules\n    exit 0",
            ),
            Mutation(
                "`provider` is read as a repository path — the gate finds no registry",
                "scripts/process_gate.py",
                '    return f"{FRAMEWORK_PREFIX}{PROVIDER_DOCUMENTS[key]}" if value == PROVIDER_VALUE else value',
                "    return value",
            ),
            Mutation(
                "sync renames a document the tenant chose, not only what adopt wrote",
                "runtime/sync.py",
                """        text = re.sub(rf'("{key}"\\s*:\\s*)"{re.escape(legacy)}"', r'\\1"provider"', text)""",
                """        text = re.sub(rf'("{key}"\\s*:\\s*)"@framework/[^"]*"', r'\\1"provider"', text)""",
            ),
        ),
    ),
    Suite(
        name="launcher_declaration",
        tests=("scripts/test/test_gate_contract_v3.py", "scripts/test/test_rules_contract.py"),
        mutations=(
            Mutation(
                "a nested map is read as its parent's keys — a port's own contract leaks out",
                ".githooks/process-gate",
                '        if (c == "{") { d++; obj[d] = 1; key[d] = ""; got[d] = 0; i++; continue }',
                '        if (c == "{") { if (d < 2) d++; obj[d] = 1; key[d] = ""; got[d] = 0; i++; continue }',
            ),
            Mutation(
                "the top-level contract beats the gate entry's own",
                ".githooks/process-gate",
                '  n="$(decl_get providers.gate.contract)"\n  [ -n "$n" ] || n="$(decl_get contract)"',
                '  n="$(decl_get contract)"\n  [ -n "$n" ] || n="$(decl_get providers.gate.contract)"',
            ),
            Mutation(
                "a port declared none reads as its command",
                ".githooks/process-gate",
                '  if [ "$(decl_get "providers.$1")" = "none" ]; then',
                '  if false; then',
            ),
        ),
    ),
    Suite(
        name="env_file_credentials",
        tests=("runtime/test/test_config.py", "scripts/test/test_env_file_credentials.py"),
        mutations=(
            Mutation(
                "a stale export beats the credential the repository declares",
                "runtime/config.py",
                "        elif value and os.environ[key] != value and is_credential(key):",
                "        elif False:",
            ),
            Mutation(
                "env_overrides no longer lets the shell win for a credential",
                "runtime/config.py",
                "    if key in env_overrides(root):\n        return\n    os.environ[key] = value",
                "    if False:\n        return\n    os.environ[key] = value",
            ),
            Mutation(
                "the warning shows the declared key",
                "runtime/config.py",
                '        print(f"⚠️  {key} in this shell differs from {path}',
                '        print(f"⚠️  {key}={value} in this shell differs from {path}',
            ),
            Mutation(
                "the startup note shows a credential's record as a value",
                "runtime/config.py",
                "    return [f\"{var}{'' if value == REDACTED else '=' + repr(value)} in the environment",
                "    return [f\"{var}{'=' + repr(value)} in the environment",
            ),
            Mutation(
                "script-only processes keep the stale export",
                "scripts/_shared.py",
                "        elif value and os.environ[key] != value and _CREDENTIAL_NAME.search(key) \\",
                "        elif False and os.environ[key] != value and _CREDENTIAL_NAME.search(key) \\",
            ),
            Mutation(
                "doctor prints the profile line, value and all",
                "scripts/verify_system.py",
                '                found.append(f"~/{name}:{number} {match.group(1)}")',
                '                found.append(f"~/{name}:{number} {line.strip()}")',
            ),
        ),
    ),
    Suite(
        name="evals_contract",
        tests=("scripts/test/test_evals_contract.py",),
        mutations=(
            Mutation(
                "a judge that never answered is reported as a pass",
                "scripts/evals_port.py",
                '    elif said == "no_verdict":\n        verdict = "no_verdict"',
                '    elif said == "no_verdict":\n        verdict = "pass"',
            ),
            Mutation(
                "a judged case without its output is graded",
                "scripts/gate_models.py",
                "    input: str = Field(min_length=1)\n    actual_output: str = Field(min_length=1)",
                "    input: str = Field(min_length=1)\n    actual_output: str | None = None",
            ),
            Mutation(
                "the runner's environment reaches the scoring's bars",
                "scripts/evals_port.py",
                "    for name in THRESHOLD_ENV:\n        os.environ.pop(name, None)",
                "    for name in ():\n        os.environ.pop(name, None)",
            ),
            Mutation(
                "the declaration beats the request's bar",
                "scripts/evals_port.py",
                "(request.fail_below, declared.fail_below, registry_fail_below)",
                "(declared.fail_below, request.fail_below, registry_fail_below)",
            ),
            Mutation(
                "a withdrawn judge model reads as weather, not a broken configuration",
                "scripts/evals_port.py",
                '    elif said == "no_verdict" and code == 1:',
                "    elif False:",
            ),
            Mutation(
                "the launcher passes a suite that was not gradable",
                ".githooks/process-gate",
                'if declared == "warn":',
                'if declared == "warn" or verdict == "not_gradable":',
            ),
            Mutation(
                "retrieved documents are refused — a retrieval tenant's dataset is not gradable",
                "scripts/gate_models.py",
                "    retrieved_context: str | list[str | ContextDocument] | None = None",
                "    retrieved_context: list[str] | str | None = None",
            ),
            Mutation(
                "a rag_poison typo is read as safe",
                "scripts/gate_models.py",
                '    expect: Literal["quarantine", "safe"]',
                "    expect: str = Field(min_length=1)",
            ),
            Mutation(
                "every judged refusal blames a missing output",
                "scripts/evals_port.py",
                '        output = " — a judged case carries the output the application produced" if outputless else ""',
                '        output = " — a judged case carries the output the application produced" '
                'if suite in gm.JUDGED_SUITES else ""',
            ),
            Mutation(
                "the provider's CI setup cannot reach a judge",
                "scripts/requirements-gate.txt",
                "\nhttpx>=0.25,<1.0",
                "",
            ),
            Mutation(
                "a warning declared for one suite excuses another",
                ".githooks/process-gate",
                "    declared = port.get(verdict, {}).get(suite) if isinstance(port, dict) else None",
                "    declared = next(iter(port.get(verdict, {}).values()), None) if isinstance(port, dict) else None",
            ),
        ),
    ),
    Suite(
        name="record_contract",
        tests=("scripts/test/test_record_contract.py",),
        mutations=(
            Mutation(
                "the gate stops validating the record before it sends it",
                "scripts/process_gate.py",
                "            gm.DevRecord.model_validate(part)",
                "            pass",
            ),
            Mutation(
                "a range named by ref records the ref, not the commit — a record no receiver can store",
                "scripts/process_gate.py",
                '    only = git("rev-parse", "--verify", f"{head}^{{commit}}", cwd=root, check=False).strip() or head',
                "    only = head",
            ),
            Mutation(
                "the receiver suite passes a receiver whatever it answers",
                "runtime/conformance.py",
                "        checks.append(Check(case.name, status == case.expect,",
                "        checks.append(Check(case.name, True,",
            ),
            Mutation(
                "the sender suite passes a sender that sends nothing",
                "runtime/conformance.py",
                '                            decision == "allow" and bool(received) and problem is None,',
                '                            decision == "allow",',
            ),
            Mutation(
                "the sender suite passes a sender that follows a redirect with the token",
                "runtime/conformance.py",
                '                            decision == "deny" and not elsewhere,',
                "                            True,",
            ),
            Mutation(
                "the sender suite passes a sender that sends no token",
                "runtime/conformance.py",
                '        bearer = bool(received) and all(auth == f"Bearer {token}" for auth, _body in received)',
                "        bearer = True",
            ),
        ),
    ),
    # .agent-rfc/designs/gate-local-events.md — gate contract 3: a tenant's commit
    # and push ask its declared provider, and the knowledge graph is the gate's.
    Suite(
        name="gate_contract_v3",
        tests=("scripts/test/test_gate_contract_v3.py",),
        mutations=(
            Mutation(
                "commit and push at contract 3 fall back to the framework's own gate",
                ".githooks/process-gate",
                '      local_by_contract "$@"\n      exit $?',
                "      true\n      exit $?",
            ),
            Mutation(
                "pre-commit at contract 3 asks the provider for a push decision it does not need",
                ".githooks/process-gate",
                '      case " $* " in *" --report "*) return 0 ;; esac',
                "      :",
            ),
            Mutation(
                "the tenant's hook keeps applying the subject rule at contract 3 — policy back in the shim",
                ".githooks/commit-msg",
                "  [3-9]|[1-9][0-9]) ;;",
                "  __never__) ;;",
            ),
            Mutation(
                "the provider drops the subject rule the hook handed it",
                "scripts/process_gate.py",
                "    if not re.match(COMMIT_SUBJECT, subject):",
                "    if False:",
            ),
            Mutation(
                "a push carrying a commit that skipped the gate is allowed",
                "scripts/process_gate.py",
                'decision="deny" if code else "allow",',
                'decision="allow",',
            ),
            Mutation(
                "the provider answers with a tenant's vendored copy of the gate",
                "runtime/cli.py",
                "    if looks_like_framework(here) and",
                "    if True and",
            ),
            Mutation(
                "kg impact scopes the working tree, not the commit — the review's hash disagrees with the gate's",
                "scripts/local_knowledge_graph.py",
                "            graph, changed = staged_graph(), staged_files()",
                '            graph, changed = staged_graph(), changed_files("HEAD")',
            ),
        ),
    ),
    Suite(
        name="framework_sync",
        tests=(
            "scripts/test/test_framework_sync.py",
            "scripts/test/test_sync_workflow.py",
            "scripts/test/test_tenant_adopt.py",
            "scripts/test/test_workflow_template_wiring.py",
        ),
        mutations=(
            Mutation(
                "a tenant must hold a secret to check out a PUBLIC provider",
                "workflow-templates/agentsmith-sync.yml",
                "          token: ${{ secrets.AGENTSMITH_READ_TOKEN || github.token }}",
                "          token: ${{ secrets.AGENTSMITH_READ_TOKEN }}",
            ),
            Mutation(
                "any commit may claim to be a framework sync",
                "scripts/process_gate.py",
                "    elif _SYNC_REVIEW.match(review_value):\n        problems = manifest_problems(gated, read)",
                "    elif _SYNC_REVIEW.match(review_value):\n        problems = []",
            ),
            Mutation(
                "a sync claims to have written files it did not touch",
                "runtime/sync.py",
                "    written = [path for path in dict.fromkeys(written) if _changed(root, path)]",
                "    written = list(dict.fromkeys(written))",
            ),
            Mutation(
                "a stale hook is not noticed, so a sync never refreshes anything",
                "runtime/sync.py",
                "        if here.is_file() and hook.is_file() and _digest(here) != _digest(hook):",
                "        if False:",
            ),
            Mutation(
                "a file the tenant edited is refreshed anyway — their work is clobbered",
                "runtime/sync.py",
                '        if state == "edited":',
                "        if False:",
            ),
            Mutation(
                "a rule file is rewritten whole, losing the tenant's own prose around the block",
                "runtime/sync.py",
                "            shared[file.path] = port.place(existing, file)",
                "            shared[file.path] = file.text",
            ),
            Mutation(
                "Claude's settings are judged by the whole file, so a tenant's own permissions "
                "freeze their gate wiring",
                "runtime/sync.py",
                "        if (rel in _MERGED or rel in regions) and here.is_file():",
                "        if rel in regions and here.is_file():",
            ),
            Mutation(
                "adopt stops writing the sync workflow — a tenant never hears it is behind",
                "runtime/adopt.py",
                "    for workflow in (GATES_WORKFLOW, SYNC_WORKFLOW):\n        if (root / workflow).exists():",
                "    for workflow in (GATES_WORKFLOW,):\n        if (root / workflow).exists():",
            ),
            Mutation(
                "the sync workflow is not kept current, so it proposes from a stale copy of itself",
                "runtime/sync.py",
                "    for workflow in (GATES_WORKFLOW, SYNC_WORKFLOW):\n        template = _workflow_template(",
                "    for workflow in (GATES_WORKFLOW,):\n        template = _workflow_template(",
            ),
            Mutation(
                "upgrade vendors into an adopted repository again",
                "runtime/machine/upgrade.py",
                "    if is_adopted(repo):",
                "    if False:",
            ),
            Mutation(
                "the manifest stops recording which command wrote it",
                "runtime/cli.py",
                '    command = generated_by or ("agentsmith tenant adopt" if adopted else "agentsmith tenant init")',
                '    command = "agentsmith tenant adopt" if adopted else "agentsmith tenant init"',
            ),
            Mutation(
                "sync scaffolds a tenant into the framework's own checkout again",
                "runtime/sync.py",
                "    marker = looks_like_framework(root)\n    if marker:\n        raise SyncError(",
                "    marker = None\n    if marker:\n        raise SyncError(",
            ),
            Mutation(
                "upgrade copies an install over the framework's own scripts/ and runtime/",
                "runtime/machine/upgrade.py",
                "    marker = looks_like_framework(repo)",
                "    marker = None",
            ),
            Mutation(
                "a tenant armed before the sweep is not planned the hooks it lacks",
                "runtime/sync.py",
                '    plan.added = [f".githooks/{hook}" for hook in GATE_HOOKS',
                '    plan.added = [f".githooks/{hook}" for hook in () and GATE_HOOKS',
            ),
            Mutation(
                "the hooks a sync adds are armed but left out of its manifest and commit",
                "runtime/sync.py",
                "    written += plan.stale + plan.added",
                "    written += plan.stale",
            ),
            Mutation(
                "the manifest is asked to vouch for itself, so a tenant gating it can never take a sync",
                "scripts/process_gate.py",
                "        if path == SCAFFOLD_MANIFEST:\n            # The check's own input",
                "        if False:\n            # The check's own input",
            ),
            Mutation(
                "a sync commits files its arming design does not cover",
                "runtime/cli.py",
                "        after = architectures.extend_design_scope(before, [*written, SCAFFOLD_MANIFEST])",
                "        after = before",
            ),
            Mutation(
                "sync rewrites another provider's declared setup step to AgentSmith's",
                "runtime/adopt.py",
                '    if declared and not declared.startswith(SETUP_ACTION + "@"):',
                "    if False:",
            ),
            Mutation(
                "sync never moves an untouched declaration to contract 2",
                "runtime/sync.py",
                "    if (root / PROVIDERS).is_file():",
                "    if False:",
            ),
            Mutation(
                "adopt declares an older gate contract, so a new tenant's commits and pushes never ask its provider",
                "runtime/adopt.py",
                "GATE_CONTRACT = 3\n",
                "GATE_CONTRACT = 2\n",
            ),
            Mutation(
                "a scaffold design leaves out the manifest committed beside it",
                "runtime/cli.py",
                "            tenant_id, stack, style, agentic, [*files, SCAFFOLD_MANIFEST], registry.get(",
                "            tenant_id, stack, style, agentic, files, registry.get(",
            ),
        ),
    ),
)


def _backup_dir() -> Path:
    """Somewhere outside the repo to keep pristine copies during a dirty run."""
    backup = Path(tempfile.mkdtemp(prefix="mutation_check_backup_"))
    return backup


def _dirty_catalogue_files(suites: tuple[Suite, ...]) -> list[str]:
    """Catalogue files with uncommitted changes.

    This harness REWRITES the files it mutates. Running it over uncommitted work
    risks losing that work, and a run killed before its restore (a timeout, a
    SIGKILL) leaves the mutation in place — which is how a stray edit reaches a
    commit. Refusing to start on a dirty file covers both: it protects work in
    progress, and it is what surfaces the residue of a previous crashed run.
    """
    paths = sorted({m.path for suite in suites for m in suite.mutations})
    if not paths:
        # `git status -- ` with no paths reports the whole repository, which a
        # run that mutates nothing (every suite skipped) has no reason to refuse.
        return []
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain", "--", *paths],
            cwd=REPO, capture_output=True, text=True, check=True,
        )
    except Exception:
        return []  # not a git checkout, or no git — nothing to assert
    return [line[3:] for line in result.stdout.splitlines() if line.strip()]


def _install_restore_handlers(target: Path, original: str) -> None:
    """Put the file back if this run is interrupted.

    Covers Ctrl-C and SIGTERM. A SIGKILL cannot be caught, which is why the
    dirty-file check above exists as the backstop.
    """

    def _restore(signum, frame):  # pragma: no cover - signal path
        target.write_text(original, encoding="utf-8")
        print(f"\n  interrupted — restored {target.relative_to(REPO)}", file=sys.stderr)
        raise SystemExit(130)

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _restore)
        except (ValueError, OSError):  # fail-open: signal.signal only works on the
            # main thread; without the handler an interrupt just skips the restore.
            pass  # pragma: no cover - non-main thread


def _pytest(tests: tuple[str, ...], *, first_failure: bool = False) -> subprocess.CompletedProcess:
    """Run a suite's tests. `first_failure` stops at the first failing test:
    right for a mutation run, where one failure already means "caught" and the
    rest of the run decides nothing — a survivor fails nothing, so it still runs
    every test. Never for a baseline, which must show EVERY test passes
    (.agent-rfc/designs/mutation-first-failure.md)."""
    args = [sys.executable, "-m", "pytest", *tests, "-q", "-p", "no:cacheprovider"]
    if first_failure:
        args.append("-x")
    if _has_timeout_plugin():
        # A mutation can turn a loop into an infinite one. Without a per-test
        # timeout that hangs the whole run rather than reporting a survivor.
        args.append("--timeout=120")
    # A fresh bytecode cache per run. Python trusts a cached .pyc whose recorded
    # source mtime (whole seconds) and size match the file; a same-length mutation
    # written within the same second as the last write of that file — the restore
    # of the previous mutation, say — matches, and the test runs the stale
    # original: a false survivor. -x made runs fast enough to hit it
    # (.agent-rfc/designs/mutation-first-failure.md).
    with tempfile.TemporaryDirectory(prefix="mutation-pycache-") as cache:
        env = {**os.environ, "PYTHONPYCACHEPREFIX": cache}
        # check=False deliberately: a NON-ZERO exit is the good outcome here. It
        # means the suite noticed the mutation, which is the entire point.
        return subprocess.run(args, cwd=REPO, capture_output=True, text=True, check=False, env=env)


def _has_timeout_plugin() -> bool:
    try:
        import pytest_timeout  # noqa: F401
    except Exception:
        return False
    return True


def _last_line(proc: subprocess.CompletedProcess) -> str:
    lines = [ln for ln in proc.stdout.strip().splitlines() if ln.strip()]
    return lines[-1] if lines else "(no output)"


def run_suite(suite: Suite) -> list[str]:
    """Run one suite. Returns a list of problems; empty means every mutation died."""
    problems: list[str] = []

    # A dirty baseline makes every result below meaningless — this has happened.
    baseline = _pytest(suite.tests)
    if baseline.returncode != 0:
        return [
            f"{suite.name}: BASELINE ALREADY FAILING — {_last_line(baseline)}. "
            f"Nothing below this line can be trusted; fix the suite first."
        ]

    for mutation in suite.mutations:
        target = REPO / mutation.path
        original = target.read_text(encoding="utf-8")

        occurrences = original.count(mutation.before)
        if occurrences != mutation.expect:
            problems.append(
                f"{suite.name}: STALE TARGET ({occurrences} matches, expected "
                f"{mutation.expect}) — {mutation.name}"
                f"\n      in {mutation.path}; re-point it at the new code, do not delete it."
            )
            continue

        previous_handlers = (
            signal.getsignal(signal.SIGINT),
            signal.getsignal(signal.SIGTERM),
        )
        _install_restore_handlers(target, original)
        try:
            target.write_text(original.replace(mutation.before, mutation.after),
                              encoding="utf-8")
            if target.read_text(encoding="utf-8") == original:
                problems.append(f"{suite.name}: MUTATION DID NOT APPLY — {mutation.name}")
                continue
            result = _pytest(suite.tests, first_failure=True)
        finally:
            target.write_text(original, encoding="utf-8")
            for sig, handler in zip(
                (signal.SIGINT, signal.SIGTERM), previous_handlers, strict=True
            ):
                try:
                    signal.signal(sig, handler)
                except (ValueError, OSError):  # fail-open: as above — not the main
                    # thread, so there was no handler of ours to put back.
                    pass  # pragma: no cover

        if result.returncode == 0:
            problems.append(
                f"{suite.name}: SURVIVED — {mutation.name}"
                f"\n      the tests pass with this broken; the property is unasserted."
            )
        else:
            print(f"  caught   {mutation.name[:88]}")

    return problems


_NO_COMMIT = "0" * 40


def changed_files(ref: str, cwd: Path = REPO) -> "set[str] | None":
    """The files changed since `ref` — committed, uncommitted or new — or None when that cannot be
    told: no ref, the all-zeros SHA GitHub sends for a new branch, or a commit
    this clone does not have (a force-push, a shallow checkout). None is
    "unknown", never "nothing changed"."""
    if not ref or ref == _NO_COMMIT:
        return None
    known = subprocess.run(["git", "-C", str(cwd), "rev-parse", "--verify", "-q", f"{ref}^{{commit}}"],
                           capture_output=True, text=True, check=False)
    if known.returncode != 0:
        return None
    # `ref` against the working tree, not HEAD, plus untracked files: locally
    # the edit in progress is the change. In CI the tree is clean, so this is
    # ref..HEAD there.
    diff = subprocess.run(["git", "-C", str(cwd), "diff", "--name-only", "--no-renames", ref],
                          capture_output=True, text=True, check=False)
    new = subprocess.run(["git", "-C", str(cwd), "ls-files", "--others", "--exclude-standard"],
                         capture_output=True, text=True, check=False)
    if diff.returncode != 0 or new.returncode != 0:
        return None
    return {line for line in (diff.stdout + new.stdout).splitlines() if line}


def select_suites(suites: "tuple[Suite, ...]", changed: "set[str] | None"
                  ) -> "tuple[list[Suite], list[tuple[Suite, str]]]":
    """-> (suites to run, (suite, why) skipped). A suite without `watch` always
    runs, and so does every suite when `changed` is None (unknown)."""
    run: list[Suite] = []
    skipped: list[tuple[Suite, str]] = []
    for suite in suites:
        if changed is None or not suite.watch or any(
                fnmatch.fnmatchcase(path, glob) for path in changed for glob in suite.watch):
            run.append(suite)
        else:
            skipped.append((suite, "none of the files it watches changed"))
    return run, skipped


def main() -> int:
    # One line, one write: under CI stdout is a pipe and block-buffered, so whole
    # groups of suites landed at one timestamp and the step's log could not say
    # which suite took the time (.agent-rfc/designs/mutation-first-failure.md).
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("suites", nargs="*", help="suite names (default: all)")
    parser.add_argument("--list", action="store_true", help="list suites and exit")
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="run over uncommitted changes (the normal case while fixing something); "
        "pristine copies are saved beside the repo first",
    )
    parser.add_argument(
        "--changed-since",
        metavar="REF",
        default=None,
        help="skip a suite that declares `watch` when none of its files changed between REF and HEAD; "
        "a REF this clone cannot use runs every suite",
    )
    args = parser.parse_args()

    selected = CATALOGUE
    if args.suites:
        known = {s.name for s in CATALOGUE}
        unknown = set(args.suites) - known
        if unknown:
            print(f"unknown suite(s): {', '.join(sorted(unknown))}", file=sys.stderr)
            print(f"known: {', '.join(sorted(known))}", file=sys.stderr)
            return 1
        selected = tuple(s for s in CATALOGUE if s.name in set(args.suites))

    skipped: list[tuple[Suite, str]] = []
    if args.changed_since is not None:
        changed = changed_files(args.changed_since)
        if changed is None:
            print(f"ℹ️  cannot tell what changed since {args.changed_since or '(no base)'!s} — "
                  "running every suite")
        run, skipped = select_suites(selected, changed)
        selected = tuple(run)
        for suite, why in skipped:
            print(f"⏭️  {suite.name} skipped — {why} since {args.changed_since}")

    if args.list:
        for suite in selected:
            print(f"{suite.name:22} {len(suite.mutations):2} mutations  "
                  f"{len(suite.tests)} test file(s)")
        return 0

    dirty = _dirty_catalogue_files(selected)
    if dirty and args.allow_dirty:
        # The author's own case: you cannot mutation-test a fix before you
        # commit it, and requiring a commit first means committing unverified
        # code. The default stays refuse — this is the deliberate exception —
        # and a pristine copy goes to disk so that even a SIGKILL, which no
        # handler can catch, leaves something to restore from.
        backup = _backup_dir()
        for path in dirty:
            destination = backup / path.replace("/", "__")
            destination.write_text((REPO / path).read_text(encoding="utf-8"),
                                   encoding="utf-8")
        print(f"⚠️  running over {len(dirty)} uncommitted file(s); "
              f"pristine copies saved to {backup}\n", file=sys.stderr)
    elif dirty:
        print(
            "🛑  refusing to run: these files have uncommitted changes and this "
            "harness rewrites them.\n",
            file=sys.stderr,
        )
        for path in dirty:
            print(f"      {path}", file=sys.stderr)
        print(
            "\n    Commit or stash them first, or pass --allow-dirty (it saves "
            "pristine\n    copies before it starts). If you did not edit these, a "
            "previous run was\n    killed before it could restore one — check the diff "
            "before discarding it.",
            file=sys.stderr,
        )
        return 1

    problems: list[str] = []
    total = 0
    for suite in selected:
        print(f"\n── {suite.name} ({len(suite.mutations)} mutations)")
        total += len(suite.mutations)
        problems.extend(run_suite(suite))

    print()
    if problems:
        print(f"🛑  {len(problems)} problem(s) across {total} mutation(s):\n")
        for problem in problems:
            print(f"  {problem}")
        return 1
    note = f"; {len(skipped)} suite(s) skipped, named above" if skipped else ""
    if not selected:
        print(f"ℹ️  no suite ran{note}")
        return 0
    print(f"✅  {total} mutations, all caught{note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
