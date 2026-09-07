"""Regression, Python half: the frame gate was off on every real animated board.

The JS twin is `js/src/cycling-board-waits-for-its-screen.test.ts` and carries
the full measurement. In short: the gate skipped whenever the clip sliced
`even`, on the reasoning that `even` means no state recurs. It does not — it
means the slicer could not PROVE recurrence, and for a board whose loop is
longer than a 4s burst it never can. 32 of 60 real clips sliced `even` while
sitting on 2-3 steady screens, so the driver clicked whichever screen happened
to be up and GeeTest's cycling line art solved 1/14.

Both ports drive the same fixtures under Tier 3 and CLAUDE.md 1c requires them
to behave the same, so both are fixed and both are pinned.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from captchakraken.page_solver import PageSolver, PageSolverConfig  # noqa: E402


def _solver(screens: int, mode: str = "even") -> PageSolver:
    s = PageSolver(PageSolverConfig(keyframe_wait_poll_ms=1))
    s._keyframe_mode = mode
    s._keyframe_steady_screens = screens
    return s


def test_a_board_that_holds_screens_is_waited_for(monkeypatch, tmp_path):
    """The live failure: every real geetest_v4_svg burst is this shape."""
    s = _solver(3)
    probes = {"n": 0}

    # **kwargs, because `_wait_for_keyframe` passes `animations="allow"` — the
    # gate watches motion and must not freeze it. A stub pinned to the old
    # signature fails as an AssertionError about waiting, which is the wrong
    # thing to go debugging.
    def fake_shot(_el, path, *_a, **_k):
        probes["n"] += 1
        Path(path).write_bytes(b"x")

    monkeypatch.setattr(s, "_screenshot", fake_shot)
    monkeypatch.setattr("captchakraken.page_solver._unlink", lambda *_: None)

    import cv2
    import numpy as np
    ref = tmp_path / "kf.png"
    cv2.imwrite(str(ref), np.zeros((40, 40, 3), dtype=np.uint8))
    monkeypatch.setattr(cv2, "imread", lambda *_a, **_k: np.zeros((40, 40, 3), dtype=np.uint8))

    assert s._wait_for_keyframe(object(), str(ref), (0.5, 0.5)) is True
    assert probes["n"] >= 1, "the gate must actually look at the widget"


def test_a_one_way_animation_still_does_not_wait(monkeypatch, tmp_path):
    """hcaptcha_rotating_obj_video and the other four continuous types.

    Measured 0 steady screens for 60 of 60 of those clips, so the 2026-08-19
    finding — 6.0s of a 28.8s solve spent waiting for a frame that could not
    return — keeps its fix.
    """
    s = _solver(0)
    looked = {"n": 0}
    monkeypatch.setattr(s, "_screenshot",
                        lambda *_a, **_k: looked.__setitem__("n", looked["n"] + 1))
    assert s._wait_for_keyframe(object(), str(tmp_path / "kf.png"), (0.5, 0.5)) is False
    assert looked["n"] == 0, "a clip with no steady screens must not be polled at all"


def test_the_pointer_is_parked_before_the_gate_opens(monkeypatch):
    """The ordering that makes the gate worth having.

    A humanised move is 274-647ms across the widget against a 1500ms dwell, so
    travelling after the match could still land the click on the next screen.
    """
    s = _solver(3)
    events: list = []

    class Hand:
        def move(self, _p, to): events.append(("move", tuple(round(v, 3) for v in to)))
        def click(self, _p, to): events.append(("click", tuple(round(v, 3) for v in to)))
        def pause(self, _k): pass

    s._human = Hand()
    monkeypatch.setattr(s, "_wait_for_keyframe",
                        lambda *_a, **_k: events.append(("gate", None)) or True)

    box = {"x": 100.0, "y": 200.0, "width": 300.0, "height": 400.0}
    s._click_when_frame_matches(
        object(), object(),
        {"target_bounding_box": [0.4, 0.4, 0.6, 0.6]}, box, "/tmp/kf.png")

    kinds = [e[0] for e in events]
    assert kinds == ["move", "gate", "click"], f"wrong order: {kinds}"
    assert events[0][1] == events[2][1], (
        f"parked and pressed different points: {events[0][1]} vs {events[2][1]}")


def test_the_budget_holds_one_worst_case_cycle():
    """Dwell max 2.7s x 3 screens = 8.1s. A shorter budget cannot catch it."""
    assert PageSolverConfig().keyframe_wait_timeout_ms >= 8_100


def test_both_ports_grant_the_same_video_budget():
    """`video_budget_ms` is derived from the wait on both sides.

    A fixture that passed on one port and timed out on the other would read as
    a driver bug, which is the whole reason the constants are pinned together.
    """
    js = (Path(__file__).resolve().parents[2] / "js" / "src" / "solver.ts").read_text()
    assert "keyframeWaitTimeoutMs: 9_000" in js, (
        "the JS default moved without this one; video_budget_ms would disagree")
