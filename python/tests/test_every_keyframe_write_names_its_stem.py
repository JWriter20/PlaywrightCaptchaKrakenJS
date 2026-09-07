"""`write_keyframes` takes a REQUIRED keyword-only `stem`, and one caller forgot it.

The animated path has two entries, and only one of them was exercised by a test.
`_solve_animated` passes `stem="challenge"`; the SPECULATIVE escalation — the
one that fires when a still answer is dropped mid-solve and the recording is
promoted — called `write_keyframes(kfset, keyframe_dir)` and died on

    TypeError: write_keyframes() missing 1 required keyword-only argument: 'stem'

every single time it was reached. Measured on Tier 3 (2026-09-07, routed Abyss,
both ports): 18 of 42 pair failures were this one TypeError, across every
animated family. It reads in the report as the SOLVER failing to solve an
animated captcha, which is the expensive way to be wrong — the model was
answering fine and the number said the model was the problem.

WHY A STATIC CHECK. Reaching that line for real needs a browser, a live widget
that cycles, AND a still answer that gets dropped part-way; the sibling path is
what a fixture normally takes, which is exactly why this one went unexercised.
The defect is not subtle behaviour, it is an argument that is not there — so the
cheap check that could not have missed it is the right one, in the style
`test_empty_answer_still_submits.py` already uses on this same file.

Asserted against the REAL signature rather than a hardcoded name, so the day
`stem` stops being required this test stops demanding it instead of going stale.
"""
import ast
import inspect
from pathlib import Path

from captchakraken.keyframes import write_keyframes

SRC = Path(__file__).resolve().parents[1] / "src" / "captchakraken"


def _required_kwonly() -> set:
    sig = inspect.signature(write_keyframes)
    return {name for name, p in sig.parameters.items()
            if p.kind is inspect.Parameter.KEYWORD_ONLY
            and p.default is inspect.Parameter.empty}


def _call_sites():
    """(file, lineno, {kwargs}) for every `write_keyframes(...)` in the client."""
    out = []
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = (func.id if isinstance(func, ast.Name)
                    else func.attr if isinstance(func, ast.Attribute) else None)
            if name != "write_keyframes":
                continue
            # `**kwargs` forwarding counts as supplying everything — it cannot
            # be checked statically and is not what this bug was.
            if any(k.arg is None for k in node.keywords):
                continue
            out.append((path.name, node.lineno,
                        {k.arg for k in node.keywords}))
    return out


def test_the_client_calls_write_keyframes_somewhere():
    # Guards the test itself: a rename would otherwise make it vacuously green.
    assert _call_sites(), "no write_keyframes call sites found — did it move?"


def test_every_call_site_passes_every_required_keyword():
    required = _required_kwonly()
    assert "stem" in required, (
        "write_keyframes no longer requires `stem`; this test is pinning a "
        "signature that has changed — update it deliberately.")

    missing = [
        (fname, lineno, sorted(required - kwargs))
        for fname, lineno, kwargs in _call_sites()
        if required - kwargs
    ]
    assert not missing, (
        "write_keyframes() is missing required keyword-only argument(s) at:\n"
        + "\n".join(f"  {f}:{ln} missing {names}" for f, ln, names in missing)
        + "\nThis raises TypeError the moment that line runs, and on the "
          "animated path it reports as the solver failing to solve.")
