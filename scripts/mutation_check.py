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
            "workflow-templates/agentsmith-gates.yml",
            "scripts/test/test_hook_chain.py",
            "scripts/test/test_tenant_adopt.py",
            "scripts/test/test_scaffold_review.py",
            "scripts/test/test_installed_runtime_tenant.py",
            "scripts/test/test_scratch_tenants.py",
            "scripts/mutation_check.py",
        ),
        mutations=(
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
                "runtime/adopt.py",
                "    if _RULES_BLOCK.search(existing):",
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
                "runtime/adopt.py",
                "target.write_text(text if generated_by_agentsmith(existing) else",
                "target.write_text(text if False else",
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
                "    if not (root / PROVIDERS).exists():\n        put(PROVIDERS, providers_declaration())",
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
        except (ValueError, OSError):  # pragma: no cover - non-main thread
            pass


def _pytest(tests: tuple[str, ...]) -> subprocess.CompletedProcess:
    args = [sys.executable, "-m", "pytest", *tests, "-q", "-p", "no:cacheprovider"]
    if _has_timeout_plugin():
        # A mutation can turn a loop into an infinite one. Without a per-test
        # timeout that hangs the whole run rather than reporting a survivor.
        args.append("--timeout=120")
    # check=False deliberately: a NON-ZERO exit is the good outcome here. It
    # means the suite noticed the mutation, which is the entire point.
    return subprocess.run(args, cwd=REPO, capture_output=True, text=True, check=False)


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
            result = _pytest(suite.tests)
        finally:
            target.write_text(original, encoding="utf-8")
            for sig, handler in zip(
                (signal.SIGINT, signal.SIGTERM), previous_handlers, strict=True
            ):
                try:
                    signal.signal(sig, handler)
                except (ValueError, OSError):  # pragma: no cover
                    pass

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
