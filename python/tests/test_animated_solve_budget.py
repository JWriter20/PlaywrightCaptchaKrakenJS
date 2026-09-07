"""An escalation to recording must fit in the budget, or say why it cannot.

`overall_solve_timeout_ms` is sized for ROUNDS — "a round costs ~4-7s, so six is
the budget". Recording an animated challenge is not a round: it is a fixed extra
stage costing the burst, the slice, one multi-image inference and the wait for
the widget to come back to the chosen frame. Nothing in the 45 s was set aside
for it, so a solve that escalated late ran the clock out MID-BURST and reported
a timeout — a message about the model being slow, for a budget that never had
room for what the solver had just decided to do.

Measured 2026-08-22, Tier 3 run 32596340560: every python-port failure on
hcaptcha_fish_swim_different, hcaptcha_number_with_highest_value_video and
hcaptcha_tile_flip_video was "exceeded overall_solve_timeout_ms during recording
the animated challenge" at 45.7-52.7 s, on fixtures that solve in 11-20 s
whenever the still path happens to answer them.
"""
from __future__ import annotations

import time

import pytest

from captchakraken.page_solver import CaptchaSolveError, PageSolver, PageSolverConfig


#: A two-frame burst. Every test here is about the BUDGET arithmetic, not about
#: the geometry, and a real 4 s burst per call turned this file into 16 s of
#: sleeping.
FAST_BURST = {"video_burst_duration_ms": 200, "video_burst_fps": 10}


def _solver(**overrides) -> PageSolver:
    return PageSolver(config=PageSolverConfig(**{**FAST_BURST, **overrides}))


class _Element:
    """An element whose screenshot always works, instantly."""

    def __init__(self) -> None:
        self.shots = 0

    def screenshot(self, **kwargs) -> None:
        self.shots += 1


def test_the_budget_is_derived_from_what_a_recording_actually_costs():
    """A longer burst carries its own budget rather than reintroducing the bug.

    The whole failure was a fixed number that had no relationship to the work,
    so a constant here would be the same mistake with a friendlier value.
    """
    # Derived from the burst's CEILING since 2026-09-07. A burst now runs until
    # the board's cycle closes rather than for a fixed 4s — a 4s window cannot
    # contain the 5.3s cycle measured on GeeTest svg, so it was omitting one
    # screen of three — and the grant has to cover the longest one it will sit
    # through, not the shortest.
    cfg = PageSolverConfig()
    assert cfg.video_budget_ms() == (cfg.video_burst_max_ms
                                     + cfg.keyframe_wait_timeout_ms
                                     + cfg.video_extra_inference_ms)

    longer = PageSolverConfig(video_burst_max_ms=cfg.video_burst_max_ms * 2)
    assert longer.video_budget_ms() - cfg.video_budget_ms() == cfg.video_burst_max_ms


def test_recording_extends_the_deadline_once_and_only_once(monkeypatch):
    """Once per solve. A grant per burst would make the deadline unbounded on a
    puzzle that re-records, which is the opposite failure and a worse one."""
    solver = _solver()
    solver._reset_animated_state()
    start = time.monotonic() * 1000.0
    solver._deadline_ms = start + solver.config.overall_solve_timeout_ms
    monkeypatch.setattr(solver, "_screenshot", lambda *a, **k: None)

    for _ in range(3):
        with pytest.raises(Exception):
            # No real frames come back, so it raises after the grant — which is
            # the part under test.
            solver._record_keyframes(_Element())

    granted = solver._deadline_ms - (start + solver.config.overall_solve_timeout_ms)
    assert round(granted) == solver.config.video_budget_ms()


def test_a_caller_who_turned_recording_off_gets_no_extension(monkeypatch):
    """`video_solve_enabled=False` already means "fail fast rather than spend the
    recording time"; silently extending that caller's deadline would ignore the
    one switch they used to say so."""
    solver = _solver(video_solve_enabled=False)
    solver._reset_animated_state()
    start = time.monotonic() * 1000.0
    solver._deadline_ms = start + solver.config.overall_solve_timeout_ms
    monkeypatch.setattr(solver, "_screenshot", lambda *a, **k: None)

    with pytest.raises(Exception):
        solver._record_keyframes(_Element())
    assert solver._deadline_ms == start + solver.config.overall_solve_timeout_ms


def test_a_burst_is_never_abandoned_partway_for_being_over_budget(monkeypatch):
    """The regression test.

    A half-recorded burst is WORTHLESS — the slicer reads the clip's temporal
    structure, so stopping at frame 27 of 40 does not give a shorter answer, it
    gives a recording that may not contain the screen the answer is on. The old
    per-frame `_check_deadline` threw away both the frames and the seconds spent
    making them. Here the deadline is already blown when the burst starts, and
    every frame must still be taken.
    """
    solver = _solver()
    solver._reset_animated_state()
    # Deep in the red: even after the grant there is no budget left at all.
    solver._deadline_ms = time.monotonic() * 1000.0 - 10_000
    solver._video_budget_granted = True  # pretend the grant already happened

    element = _Element()
    monkeypatch.setattr(solver, "_screenshot", lambda *a, **k: element.screenshot())

    with pytest.raises(CaptchaSolveError) as excinfo:
        solver._record_keyframes(element)

    # It refused BEFORE recording anything, rather than stopping halfway.
    assert element.shots == 0
    assert "not starting one that would be cut off" in str(excinfo.value)
    # …and it names the knobs, because "your budget cannot fit a recording" is
    # only actionable if you know which number to move.
    assert "overall_solve_timeout_ms" in str(excinfo.value)


def test_the_default_budget_is_enough_for_an_escalation_on_the_last_round():
    """The arithmetic the live failures came down to.

    Five still rounds at ~7 s is 35 s — inside the 45 s cap, and exactly where
    the animated probe arms after two identical answers. The recording that
    follows has to fit in what is left, and before this change it could not.
    """
    cfg = PageSolverConfig()
    spent_on_rounds = (cfg.max_solve_loops - 1) * 7_000
    left = cfg.overall_solve_timeout_ms - spent_on_rounds + cfg.video_budget_ms()
    assert left >= cfg.video_burst_duration_ms + cfg.keyframe_wait_timeout_ms
