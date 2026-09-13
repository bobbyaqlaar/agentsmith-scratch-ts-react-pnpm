"""
scripts/check_bare_except.py — AST-based detector for empty exception
handlers ("swallowed" errors), used by hooks/pre-commit's Guardrail 2.

Replaces a regex check (`except\\s*(\\w+)?:\\s*$`) that matched the *header
line* of almost any `except` clause — since Python's `except X:` line
always ends right at the colon regardless of how long or short the body
underneath is, that regex flagged ordinary multi-line handlers (e.g.
`except KeyError:\\n    raise ValueError(...) from None`) as "bare except
with no handler body" while doing nothing to verify the body was actually
empty. Parsing the real syntax tree instead of pattern-matching text lines
is the only way to tell "this handler's body is genuinely a no-op" from
"this handler's body just happens to start on the next line," which is
true of essentially all idiomatically-formatted Python.

A handler is flagged only when its body is exactly one statement and that
statement is `pass` or a bare `...` (Ellipsis) expression — i.e. it does
nothing at all. `except Exception: pass` is sometimes an intentional,
documented fail-open pattern (e.g. runtime/llm_gateway.py's
_record_span_attributes, where a tracing failure must never break the
actual LLM call) — append `# fail-open: <reason>` to the `except` line to
opt that specific handler out, the same way the rest of this codebase
uses inline suppression comments rather than a blanket exemption list.

Deliberately NOT spelled `# noqa: ...` — ruff treats any `# noqa:` comment
as ITS OWN suppression directive and demands a comma-separated list of
ruff rule codes after the colon, so `# noqa: bare-except — reason` (this
checker's first convention) made ruff itself emit an "Invalid noqa
directive" warning on every annotated line. `# fail-open: ...` carries the
same meaning without colliding with ruff's own comment syntax.
"""

from __future__ import annotations

import ast
import sys


def _is_noop_body(body: list[ast.stmt]) -> bool:
    if len(body) != 1:
        return False
    stmt = body[0]
    if isinstance(stmt, ast.Pass):
        return True
    if (
        isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Constant)
        and stmt.value.value is Ellipsis
    ):
        return True
    return False


def find_violations(source: str, path: str) -> list[tuple[str, int]]:
    try:
        tree = ast.parse(source, filename=path)
    except SyntaxError as exc:
        # Reporting the syntax error is not this checker's job — py_compile and
        # ruff both do it. Returning [] silently IS its job to get right: an
        # empty list means "examined, clean", and a file this checker could not
        # read has not been examined at all. Say so, then defer.
        print(
            f"check_bare_except: SKIPPED {path} — could not parse it "
            f"({exc.msg} at line {exc.lineno}), so it was NOT checked for bare "
            f"excepts. Fix the syntax error and re-run.",
            file=sys.stderr,
        )
        return []

    lines = source.splitlines()
    violations = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler):
            continue
        if not _is_noop_body(node.body):
            continue
        # The except header can span multiple lines (a long exception-type
        # tuple, or a formatter wrapping a trailing comment onto its own
        # line) — node.lineno is only where `except` itself starts, so
        # search the whole header span (up to the body's first line) for
        # the suppression marker, not just that one line.
        body_start = node.body[0].lineno if node.body else node.lineno + 1
        header_lines = lines[node.lineno - 1 : max(node.lineno, body_start - 1)]
        # ALSO the comment lines immediately ABOVE the except, because a reason
        # worth writing is often longer than what fits after `except Exception:`.
        # Requiring it inline put twelve of these over any sane line limit and
        # set this checker against E501 — two standards in one repo that could
        # not both be satisfied, which is how a rule gets waived. Only an
        # unbroken run of comment lines counts: a `fail-open:` note further up,
        # separated by code, is about something else.
        above = []
        i = node.lineno - 2
        while i >= 0 and lines[i].lstrip().startswith("#"):
            above.append(lines[i])
            i -= 1
        if any("fail-open:" in line for line in header_lines + above):
            continue
        violations.append((path, node.lineno))
    return violations


def main(argv: list[str]) -> int:
    failed = False
    for path in argv:
        try:
            with open(path, encoding="utf-8") as f:
                source = f.read()
        except OSError:
            continue
        for file_path, lineno in find_violations(source, path):
            print(
                f"{file_path}:{lineno}: empty except handler (pass/... only) — log or re-raise, "
                f"or append '# fail-open: <reason>' to the except line if this is an intentional fail-open"
            )
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
