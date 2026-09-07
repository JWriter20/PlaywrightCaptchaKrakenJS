"""
Hermetic tests for the Python page driver.

No browser, no GPU, no network. The driver duck-types the Playwright surface,
which is exactly what makes this testable: a fake page that implements the same
handful of methods exercises the real state machine. These cover the decisions
that were expensive to learn in the TypeScript driver and that a port is most
likely to get wrong.
"""

from __future__ import annotations

import struct
import time
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from captchakraken.humanize import MobileHumanizer, MouseHumanizer  # noqa: E402
from captchakraken.page_solver import (  # noqa: E402
    AnimatedChallengeError,
    CaptchaSolveError,
    NoCaptchaFoundError,
    PageSolver,
    PageSolverConfig,
    UnsupportedChallengeError,
    _aggregate,
    _as_dict,
    _read_png_dimensions,
)
from captchakraken.solver import UnsupportedCaptchaError  # noqa: E402
from captchakraken.trajectory import generate_trajectory  # noqa: E402


# ---------------------------------------------------------------- fake page


class FakeElement:
    def __init__(
        self,
        src: str = "",
        visible: bool = True,
        box: Optional[Dict[str, float]] = None,
        frame: Optional["FakeFrame"] = None,
        text: str = "",
    ) -> None:
        self._src = src
        self._visible = visible
        self._box = box or {"x": 100.0, "y": 100.0, "width": 400.0, "height": 400.0}
        self._frame = frame
        self._text = text
        self.screenshots = 0

    def get_attribute(self, name: str) -> Optional[str]:
        return {"src": self._src, "value": None}.get(name)

    def is_visible(self) -> bool:
        return self._visible

    def bounding_box(self) -> Dict[str, float]:
        return self._box

    def content_frame(self) -> Optional["FakeFrame"]:
        return self._frame

    def text_content(self) -> str:
        return self._text

    def scroll_into_view_if_needed(self) -> None:
        pass

    def screenshot(self, path: str, **_: Any) -> None:
        self.screenshots += 1
        _write_png(path, 400, 400)


class FakeFrame:
    def __init__(self, elements: Optional[Dict[str, FakeElement]] = None) -> None:
        self._elements = elements or {}

    def query_selector(self, selector: str) -> Optional[FakeElement]:
        for key, element in self._elements.items():
            if key in selector or selector in key:
                return element
        return None

    def wait_for_selector(self, *_: Any, **__: Any) -> None:
        pass

    def wait_for_function(self, *_: Any, **__: Any) -> None:
        pass


class FakeMouse:
    def __init__(self) -> None:
        self.moves: List[tuple] = []
        self.clicks = 0
        self.presses = 0

    def move(self, x: float, y: float) -> None:
        self.moves.append((x, y))

    def down(self) -> None:
        self.presses += 1

    def up(self) -> None:
        self.clicks += 1


class FakePage:
    def __init__(self, elements: Optional[Dict[str, FakeElement]] = None) -> None:
        self._elements = elements or {}
        self.mouse = FakeMouse()
        self.viewport_size = {"width": 1280, "height": 800}

    def query_selector(self, selector: str) -> Optional[FakeElement]:
        return self._elements.get(selector)

    def query_selector_all(self, selector: str) -> List[FakeElement]:
        element = self._elements.get(selector)
        return [element] if element else []

    def eval_on_selector(self, *_: Any, **__: Any) -> str:
        return ""

    # camoufox reports viewport_size None and the driver then asks the page.
    # Default: report nothing, so tests exercise the "unknown viewport" branch.
    inner_size: Optional[Dict[str, float]] = None

    def evaluate(self, *_: Any, **__: Any) -> Optional[Dict[str, float]]:
        return self.inner_size


def _write_png(path: str, width: int, height: int) -> None:
    """A byte-valid PNG header — enough for _read_png_dimensions and file I/O."""
    ihdr = struct.pack(">II", width, height) + b"\x08\x02\x00\x00\x00"
    with open(path, "wb") as handle:
        handle.write(b"\x89PNG\r\n\x1a\n")
        handle.write(struct.pack(">I", 13) + b"IHDR" + ihdr + b"\x00\x00\x00\x00")


RECAPTCHA_BFRAME = 'iframe[src*="recaptcha/api2/bframe"]'
RECAPTCHA_ANCHOR = 'iframe[src*="recaptcha/api2/anchor"]'
HCAPTCHA_CHALLENGE = 'iframe[src*="hcaptcha"][src*="frame=challenge"]'
HCAPTCHA_CHECKBOX = 'iframe[src*="hcaptcha"][src*="frame=checkbox"]'


#: Truthy stand-in for the CaptchaSolver, so `__init__` does not build a real
#: one (it would open an HTTP session to vLLM). Replaced with None immediately.
_NO_MODEL = object()


def _solver(**overrides: Any) -> PageSolver:
    """A PageSolver whose model half is stubbed — no CaptchaSolver constructed."""
    # The REAL __init__, with a sentinel standing in for the model half, so
    # per-solve state added there cannot go missing here. Hand-listing the
    # attributes instead is what it used to do, and every attribute added since
    # broke this helper with an AttributeError raised nowhere near the
    # behaviour under test (`_budget`, most recently).
    solver = PageSolver(config=PageSolverConfig(**overrides), solver=_NO_MODEL)
    solver._solver = None
    # Deliberately NOT __init__'s value: these tests call the mouse helpers
    # directly, and an unseeded cursor takes the seeding path first. Set on the
    # HUMANIZER, which is where that flag lives — setting it on the solver would
    # silently do nothing and the seeding move would come back.
    if hasattr(solver._human, "_cursor_seeded"):
        solver._human._cursor_seeded = True
    return solver


# ---------------------------------------------------------------- detection


class TestDetection:
    def test_prefers_the_open_challenge_over_the_checkbox(self):
        # An open image challenge and an unchecked anchor can be on the page at
        # once. Solving the anchor again just reopens what is already open.
        challenge = FakeElement(src="https://google.com/recaptcha/api2/bframe?k=x")
        anchor = FakeElement(src="https://google.com/recaptcha/api2/anchor?k=x")
        page = FakePage({RECAPTCHA_BFRAME: challenge, RECAPTCHA_ANCHOR: anchor})
        assert _solver().detect_captcha(page) is challenge

    def test_ignores_an_already_checked_recaptcha_anchor(self):
        frame = FakeFrame({".recaptcha-checkbox-checked": FakeElement()})
        anchor = FakeElement(src="recaptcha/api2/anchor", frame=frame)
        page = FakePage({RECAPTCHA_ANCHOR: anchor})
        assert _solver().detect_captcha(page) is None

    def test_hcaptcha_checkbox_solved_via_aria_checked_without_a_token(self):
        # Demo pages don't always populate h-captcha-response, so the anchor's
        # visual state is the necessary tie-breaker, not a nicety.
        frame = FakeFrame({'#checkbox[aria-checked="true"]': FakeElement()})
        checkbox = FakeElement(src="https://hcaptcha.com/?frame=checkbox", frame=frame)
        page = FakePage({HCAPTCHA_CHECKBOX: checkbox})
        assert _solver().detect_captcha(page) is None
        assert _solver().is_captcha_solved(page) is True

    def test_invisible_recaptcha_is_not_an_interactive_widget(self):
        # v3 / invisible-v2 injects only an anchor with size=invisible and never
        # a challenge frame. Treating it as "still loading" hangs for the whole
        # render-wait budget on every v3 page.
        anchor = FakeElement(src="https://google.com/recaptcha/api2/anchor?k=x&size=invisible")
        page = FakePage({RECAPTCHA_ANCHOR: anchor})
        assert _solver().has_interactive_widget_in_dom(page) is False

    def test_visible_v2_anchor_is_an_interactive_widget(self):
        anchor = FakeElement(src="https://google.com/recaptcha/api2/anchor?k=x")
        page = FakePage({RECAPTCHA_ANCHOR: anchor})
        assert _solver().has_interactive_widget_in_dom(page) is True


class TestSolveLoop:
    def test_no_widget_at_all_fails_fast(self):
        with pytest.raises(NoCaptchaFoundError):
            _solver().solve(FakePage())

    def test_nothing_left_after_interacting_is_success(self):
        # detect_captcha() == None means two different things depending on
        # whether we have interacted. Post-interaction it means solved.
        solver = _solver()
        page = FakePage()
        solver._has_interacted_probe = True
        challenge = FakeElement(src="recaptcha/api2/bframe")
        pages = [challenge, None]

        def fake_detect(_page):
            return pages.pop(0) if pages else None

        solver.detect_captcha = fake_detect  # type: ignore[method-assign]
        solver._solve_single = lambda *_: (True, [])  # type: ignore[method-assign]
        solver.is_captcha_solved = lambda _page: False  # type: ignore[method-assign]
        solver._is_challenge_freshly_rendered = lambda _page: False  # type: ignore[method-assign]
        solver._has_recaptcha_underselect_error = lambda _page: False  # type: ignore[method-assign]

        result = solver.solve(page)
        assert result.is_solved is True

    def test_no_interaction_with_captcha_still_present_aborts(self):
        # Otherwise the loop spins until the overall timeout doing nothing.
        solver = _solver()
        solver.detect_captcha = lambda _page: FakeElement(src="recaptcha/api2/bframe")  # type: ignore[method-assign]
        solver._solve_single = lambda *_: (False, [])  # type: ignore[method-assign]
        solver.is_captcha_solved = lambda _page: False  # type: ignore[method-assign]
        solver._has_recaptcha_underselect_error = lambda _page: False  # type: ignore[method-assign]
        with pytest.raises(CaptchaSolveError, match="no interactions"):
            solver.solve(FakePage())

    def test_unsupported_on_the_first_frame_is_definitive(self):
        # Before any interaction there is no transitional-frame excuse: fail.
        solver = _solver()
        solver.detect_captcha = lambda _page: FakeElement(src="https://hcaptcha.com/?frame=challenge")  # type: ignore[method-assign]

        def raise_unsupported(*_):
            raise UnsupportedCaptchaError("nope")

        solver._solve_single = raise_unsupported  # type: ignore[method-assign]
        with pytest.raises(UnsupportedChallengeError):
            solver.solve(FakePage())

    def test_unsupported_mid_solve_retries_instead_of_aborting(self):
        # The "solves round 1, dies on round 2" bug: after interacting, an
        # `unsupported` verdict is usually a not-yet-settled next round.
        solver = _solver(max_unsupported_resolves=2)
        solver.detect_captcha = lambda _page: FakeElement(src="https://hcaptcha.com/?frame=challenge")  # type: ignore[method-assign]
        solver._wait_for_element_settled = lambda _el: "settled"  # type: ignore[method-assign]
        solver.is_captcha_solved = lambda _page: False  # type: ignore[method-assign]
        solver._is_challenge_freshly_rendered = lambda _page: False  # type: ignore[method-assign]
        solver._has_recaptcha_underselect_error = lambda _page: False  # type: ignore[method-assign]

        calls = {"n": 0}

        def flaky(*_):
            calls["n"] += 1
            if calls["n"] == 1:
                return True, []          # interact once
            raise UnsupportedCaptchaError("transitional blank frame")

        solver._solve_single = flaky  # type: ignore[method-assign]
        with pytest.raises(UnsupportedChallengeError):
            solver.solve(FakePage())
        # Retried rather than aborting on the first unsupported after interaction.
        assert calls["n"] >= 3

    def test_stale_handle_after_submit_is_retried_not_fatal(self):
        # hCaptcha swaps the iframe between rounds while we hold the old handle.
        solver = _solver(max_stale_element_retries=2, stale_element_backoff_ms=1)
        solver.detect_captcha = lambda _page: FakeElement(src="https://hcaptcha.com/?frame=challenge")  # type: ignore[method-assign]
        solver.is_captcha_solved = lambda _page: False  # type: ignore[method-assign]
        solver._is_challenge_freshly_rendered = lambda _page: False  # type: ignore[method-assign]
        solver._has_recaptcha_underselect_error = lambda _page: False  # type: ignore[method-assign]

        calls = {"n": 0}

        def flaky(*_):
            calls["n"] += 1
            if calls["n"] == 1:
                return True, []
            raise RuntimeError("Element is not attached to the DOM")

        solver._solve_single = flaky  # type: ignore[method-assign]
        with pytest.raises(RuntimeError):
            solver.solve(FakePage())
        assert calls["n"] >= 3  # retried the stale handle before giving up

    def test_stale_handle_before_any_interaction_is_surfaced(self):
        # A first-frame failure is a real problem, not a round transition.
        solver = _solver()
        solver.detect_captcha = lambda _page: FakeElement(src="https://hcaptcha.com/?frame=challenge")  # type: ignore[method-assign]

        def raise_detached(*_):
            raise RuntimeError("Element is not attached to the DOM")

        solver._solve_single = raise_detached  # type: ignore[method-assign]
        with pytest.raises(RuntimeError):
            solver.solve(FakePage())

    def test_underselect_error_retries_once_then_aborts(self):
        # reCAPTCHA does NOT refresh tiles on this error, so without the
        # missed-tiles prompt the model answers `done` forever and the solve
        # loops until timeout.
        solver = _solver()
        solver.detect_captcha = lambda _page: FakeElement(src="recaptcha/api2/bframe")  # type: ignore[method-assign]
        solver.is_captcha_solved = lambda _page: False  # type: ignore[method-assign]
        solver._is_challenge_freshly_rendered = lambda _page: False  # type: ignore[method-assign]
        # `select-more` specifically: the banner where the tiles do NOT refresh.
        # `dynamic-more` is deliberately NOT fatal — it is the dynamic 3x3's
        # normal progress — and is pinned in
        # test_recaptcha_dynamic_more_is_not_an_error.py.
        solver._recaptcha_banner_kind = lambda _page: "select-more"  # type: ignore[method-assign]

        seen_retry_modes: List[Optional[str]] = []

        def record(_page, _el, retry_mode):
            seen_retry_modes.append(retry_mode)
            return True, []

        solver._solve_single = record  # type: ignore[method-assign]
        with pytest.raises(CaptchaSolveError, match="under-selection"):
            solver.solve(FakePage())
        # First pass has no hint; the retry carries missed-tiles.
        assert seen_retry_modes[0] is None
        assert "missed-tiles" in seen_retry_modes

    def test_solved_signal_short_circuits_the_loop(self):
        solver = _solver()
        solver.detect_captcha = lambda _page: FakeElement(src="recaptcha/api2/bframe")  # type: ignore[method-assign]
        solver._solve_single = lambda *_: (True, [])  # type: ignore[method-assign]
        solver.is_captcha_solved = lambda _page: True  # type: ignore[method-assign]
        assert solver.solve(FakePage()).is_solved is True


class TestHumanizationMode:
    """The mode is a choice of INPUT DEVICE, and the solver must honour it
    everywhere — not just at the one call site somebody remembered."""

    def _driven(self, mode, backend=None):
        """Run one full round against a click answer, and report what was
        dispatched."""
        cfg: Dict[str, Any] = {"humanization": mode}
        if backend is not None:
            cfg = {"humanizer": MobileHumanizer(backend=backend)}
        solver = _solver(**cfg)
        element = FakeElement(src="recaptcha/api2/bframe")
        page = FakePage({RECAPTCHA_BFRAME: element})
        solver.detect_captcha = lambda _page: element  # type: ignore[method-assign]
        solver._solve_single = lambda *_: (  # type: ignore[method-assign]
            solver._execute_click(
                page,
                {"target_bounding_box": [0.1, 0.1, 0.3, 0.3]},
                element.bounding_box(),
            ),
            [],
        ) and (True, [])
        solver.is_captcha_solved = lambda _page: True  # type: ignore[method-assign]
        solver.solve(page)
        return page

    def test_mobile_never_reaches_for_the_mouse(self):
        # THE regression. A mousemove at a touch-only widget is the wrong event,
        # not a weaker one: the page's touch handlers never fire, and the solve
        # fails for a reason nothing reports.
        backend = _RecordingTouch()
        page = self._driven("mobile", backend=backend)
        assert page.mouse.moves == [] and page.mouse.presses == 0
        assert backend.kinds == ["down", "move", "up"]

    #: `_execute_click` insets 10% and picks a random point inside the bbox
    #: (0.1..0.3 of a 400px element, offset 100), so the landing point is a
    #: band, not a constant.
    _TARGET = (148.0, 212.0)

    def _lands_on_target(self, at) -> bool:
        lo, hi = self._TARGET
        return lo <= at[0] <= hi and lo <= at[1] <= hi

    def test_none_spends_one_move_per_gesture(self):
        page = self._driven("none")
        assert len(page.mouse.moves) == 1
        assert self._lands_on_target(page.mouse.moves[0])

    def test_mouse_is_still_a_full_trajectory(self):
        page = self._driven("mouse")
        assert len(page.mouse.moves) > 5
        assert self._lands_on_target(page.mouse.moves[-1])


class _RecordingTouch:
    """Stands in for a CDP/Appium backend — see humanize.TouchBackend."""

    name = "recording"

    def __init__(self) -> None:
        self.kinds: List[str] = []

    def down(self, x: float, y: float) -> None:
        self.kinds.append("down")

    def move(self, path) -> None:
        self.kinds.append("move")

    def up(self, x: float, y: float) -> None:
        self.kinds.append("up")


class TestFreshnessGuard:
    def test_reuses_the_answer_when_the_frame_held_still(self):
        solver = _solver()
        solver._frame_changed_since = lambda *_: False  # type: ignore[method-assign]
        calls = {"n": 0}

        def query(_path):
            calls["n"] += 1
            return [{"action": "done"}], [{"total_tokens": 5}]

        actions, usage = solver._solve_frame_freshness_guarded(FakeElement(), "/tmp/x.png", query)
        assert calls["n"] == 1
        assert actions == [{"action": "done"}]
        assert usage == [{"total_tokens": 5}]

    def test_resolves_and_merges_usage_when_the_frame_moved(self):
        # Both queries really happened and both were billed, so both must appear
        # — dropping the stale round's usage would under-report real spend.
        solver = _solver(max_stale_frame_resolves=1)
        solver._frame_changed_since = lambda *_: True  # type: ignore[method-assign]
        solver._screenshot = lambda *a, **k: _write_png(a[1] if len(a) > 1 else k["path"], 4, 4)  # type: ignore[method-assign]
        calls = {"n": 0}

        def query(_path):
            calls["n"] += 1
            return [{"action": f"round{calls['n']}"}], [{"total_tokens": calls["n"]}]

        actions, usage = solver._solve_frame_freshness_guarded(FakeElement(), "/tmp/x.png", query)
        assert calls["n"] == 2
        assert actions == [{"action": "round2"}]  # the DEVELOPED frame's answer
        assert usage == [{"total_tokens": 1}, {"total_tokens": 2}]

    def test_disabled_guard_never_requeries(self):
        solver = _solver(stale_frame_resolve_enabled=False)
        solver._frame_changed_since = lambda *_: True  # type: ignore[method-assign]
        calls = {"n": 0}

        def query(_path):
            calls["n"] += 1
            return [{"action": "done"}], []

        solver._solve_frame_freshness_guarded(FakeElement(), "/tmp/x.png", query)
        assert calls["n"] == 1


class TestGridGeometry:
    GRID = [
        [0, 0, 100, 100], [100, 0, 200, 100], [200, 0, 300, 100],
        [0, 100, 100, 200], [100, 100, 200, 200], [200, 100, 300, 200],
        [0, 200, 100, 300], [100, 200, 200, 300], [200, 200, 300, 300],
    ]

    def test_bbox_maps_to_a_row_major_one_indexed_cell(self):
        solver = _solver()
        # Centre of the middle cell -> cell 5.
        assert solver._bbox_to_cell([0.4, 0.4, 0.6, 0.6], self.GRID, 300, 300) == 5
        # Top-left -> cell 1.
        assert solver._bbox_to_cell([0.0, 0.0, 0.2, 0.2], self.GRID, 300, 300) == 1
        # Bottom-right -> cell 9.
        assert solver._bbox_to_cell([0.85, 0.85, 1.0, 1.0], self.GRID, 300, 300) == 9

    def test_bbox_outside_every_cell_is_none(self):
        # A gutter hit must not be silently attributed to a neighbouring tile;
        # callers click the raw bbox and skip per-tile tracking.
        sparse = [[0, 0, 10, 10]]
        assert _solver()._bbox_to_cell([0.9, 0.9, 0.95, 0.95], sparse, 300, 300) is None

    def test_clicked_cells_are_watched_first(self):
        # Ordering matters: the just-clicked tiles are the ones whose fade tells
        # us the refresh began.
        assert _solver()._order_by_priority([3, 7, 1], [7, 1]) == [7, 1, 3]

    def test_priority_cells_not_loading_are_dropped(self):
        assert _solver()._order_by_priority([3], [7, 1]) == [3]


class TestHelpers:
    def test_png_dimensions_from_ihdr(self, tmp_path):
        path = str(tmp_path / "a.png")
        _write_png(path, 321, 654)
        assert _read_png_dimensions(path) == (321, 654)

    def test_png_dimensions_rejects_non_png(self, tmp_path):
        path = str(tmp_path / "a.png")
        Path(path).write_bytes(b"not a png at all......")
        assert _read_png_dimensions(path) is None

    def test_aggregate_sums_rounds(self):
        assert _aggregate([
            {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
            {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
        ]) == [{"rounds": 2, "prompt_tokens": 15, "completion_tokens": 3, "total_tokens": 18}]

    def test_aggregate_of_nothing_is_empty(self):
        assert _aggregate([]) == []

    def test_as_dict_accepts_pydantic_actions(self):
        from captchakraken.action_types import WaitAction

        assert _as_dict(WaitAction(action="wait", duration_ms=50))["duration_ms"] == 50

    def test_as_dict_passes_plain_dicts_through(self):
        assert _as_dict({"action": "done"}) == {"action": "done"}


class TestTrajectory:
    def test_lands_exactly_on_target(self):
        # Texture is fine everywhere except the last sample: drift there means
        # every click lands off-target.
        points, _ = generate_trajectory((0, 0), (640, 480))
        assert points[-1] == (640, 480)

    def test_timings_are_cumulative_and_monotonic(self):
        # The driver sleeps against these as absolute offsets, exactly as the TS
        # driver does. Per-step deltas here would make the cursor teleport.
        _, timings = generate_trajectory((0, 0), (800, 600))
        assert timings[0] == 0
        assert all(b >= a for a, b in zip(timings, timings[1:]))

    def test_path_is_not_a_straight_line(self):
        points, _ = generate_trajectory((0, 0), (600, 0))
        # A straight lerp would keep every y at 0.
        assert any(abs(y) > 1.0 for _, y in points[:-1])

    def test_zero_length_move_is_a_single_sample(self):
        points, timings = generate_trajectory((10, 10), (10, 10))
        assert points == [(10, 10)] and timings == [0.0]

    def test_longer_moves_take_longer(self):
        # Fitts's law, not a fixed step count.
        _, short = generate_trajectory((0, 0), (40, 0))
        _, long = generate_trajectory((0, 0), (1200, 0))
        assert long[-1] > short[-1]


class TestAlreadySolved:
    def test_an_already_satisfied_captcha_returns_solved_not_an_error(self):
        # The best outcome must not be an exception. A good stealth browser
        # often clears reCAPTCHA on the checkbox alone, so the widget is in the
        # DOM, nothing is detectable to solve, and we have not interacted —
        # previously the render-wait branch then raised NoCaptchaFoundError.
        solver = _solver()
        solver.detect_captcha = lambda _page: None  # type: ignore[method-assign]
        solver.is_captcha_solved = lambda _page: True  # type: ignore[method-assign]
        solver.has_interactive_widget_in_dom = lambda _page: True  # type: ignore[method-assign]
        assert solver.solve(FakePage()).is_solved is True

    def test_an_unrendered_widget_still_waits_then_fails(self):
        # The other side of the same branch must keep working: present but not
        # solved and not rendered -> wait, then fail rather than hang.
        solver = _solver()
        solver.detect_captcha = lambda _page: None  # type: ignore[method-assign]
        solver.is_captcha_solved = lambda _page: False  # type: ignore[method-assign]
        solver.has_interactive_widget_in_dom = lambda _page: True  # type: ignore[method-assign]
        with pytest.raises(NoCaptchaFoundError):
            solver.solve(FakePage())


class TestDeadline:
    def test_a_slow_single_attempt_cannot_overrun_the_budget(self):
        # The budget must be a real budget. Checking it only at the top of each
        # attempt means one slow attempt overruns it without bound — a camoufox
        # session ran past ten minutes against a nominal 120s timeout because
        # nothing looked at the clock again until the attempt returned.
        solver = _solver(overall_solve_timeout_ms=50)

        def slow_single(_page, _el, _retry):
            time.sleep(0.2)  # already past the 50ms budget
            solver._check_deadline("test")
            return True, []

        solver.detect_captcha = lambda _page: FakeElement(src="recaptcha/api2/bframe")  # type: ignore[method-assign]
        solver._solve_single = slow_single  # type: ignore[method-assign]
        solver.is_captcha_solved = lambda _page: False  # type: ignore[method-assign]
        with pytest.raises(CaptchaSolveError, match="exceeded overall_solve_timeout_ms"):
            solver.solve(FakePage())

    def test_the_deadline_is_cleared_between_solves(self):
        # A deadline left set from a previous solve would make the next one fail
        # instantly.
        solver = _solver(overall_solve_timeout_ms=50)
        solver.detect_captcha = lambda _page: None  # type: ignore[method-assign]
        solver.is_captcha_solved = lambda _page: True  # type: ignore[method-assign]
        solver.solve(FakePage())
        assert solver._deadline_ms is None
        solver._check_deadline("outside a solve")  # must not raise


class TestTracePathClamping:
    """Lives on `humanize.MouseHumanizer` now, but it is the same rule and the
    same camoufox deadlock it exists to avoid."""

    def _points(self, page, pts):
        human = MouseHumanizer()
        human._trace(page, pts, [0.0] * len(pts))
        return page.mouse.moves

    def test_falls_back_to_the_page_when_viewport_size_is_none(self):
        # camoufox reports viewport_size None. We must NOT give up and skip
        # clamping — an out-of-window destination wedges its juggler (mouse.move
        # never returns). Ask the page for the real window instead.
        page = FakePage()
        page.viewport_size = None
        page.inner_size = {"width": 800.0, "height": 600.0}
        assert self._points(page, [(5000.0, 4000.0)]) == [(799.0, 599.0)]

    def test_no_clamping_when_the_viewport_is_genuinely_unknowable(self):
        # camoufox reports viewport_size None. Clamping to a GUESSED viewport
        # puts coordinates exactly on an edge that isn't the real one, and an
        # exact-edge coordinate deadlocks camoufox's humanised-mouse patch
        # (upstream #225) — mouse.move never returns and the solve hangs at 0%
        # CPU. Points already come from a trajectory between on-screen elements.
        page = FakePage()
        page.viewport_size = None
        assert self._points(page, [(5000.0, 4000.0)]) == [(5000.0, 4000.0)]

    def test_clamping_insets_off_the_exact_edge(self):
        # When we DO know the viewport, an out-of-range point lands just inside
        # the boundary rather than exactly on it.
        page = FakePage()
        page.viewport_size = {"width": 800, "height": 600}
        moves = self._points(page, [(-50.0, -50.0), (5000.0, 5000.0)])
        assert moves == [(1.0, 1.0), (799.0, 599.0)]

    def test_in_range_points_are_untouched(self):
        page = FakePage()
        page.viewport_size = {"width": 800, "height": 600}
        assert self._points(page, [(400.0, 300.0)]) == [(400.0, 300.0)]


# ------------------------------------------------- typing and sliding


class _Scope:
    """A challenge frame that answers ONLY the selectors it was given.

    Deliberately not FakeFrame, whose loose substring matching would let a
    generic pattern satisfy a vendor lookup and hide the ordering these tests
    exist to pin.
    """

    def __init__(self, mapping: Dict[str, "FakeElement"]) -> None:
        self._mapping = mapping
        self.asked: List[str] = []

    def query_selector(self, selector: str) -> Optional["FakeElement"]:
        self.asked.append(selector)
        return self._mapping.get(selector)


class _Keyboard:
    def __init__(self) -> None:
        self.typed: List[str] = []
        self.pressed: List[str] = []

    def type(self, text: str) -> None:
        self.typed.append(text)

    def press(self, key: str) -> None:
        self.pressed.append(key)


class _Mouse(FakeMouse):
    """Records the ORDER of press/move/release, which is the whole contract of
    a slide: a release in the wrong place is a wrong answer submitted."""

    def __init__(self) -> None:
        super().__init__()
        self.log: List[tuple] = []

    def move(self, x: float, y: float) -> None:
        super().move(x, y)
        self.log.append(("move", x, y))

    def down(self) -> None:
        self.log.append(("down", self.moves[-1][0] if self.moves else None, None))

    def up(self) -> None:
        super().up()
        self.log.append(("up", self.moves[-1][0] if self.moves else None, None))


def _typing_page() -> "FakePage":
    page = FakePage()
    page.mouse = _Mouse()
    page.keyboard = _Keyboard()
    return page


class TestTextCaptchas:
    def test_the_vendor_box_wins_over_the_generic_one(self):
        """Selector order is not cosmetic. The generic tail matches any text
        input in the scope; on a widget that also carries, say, an audio-answer
        field, taking the first generic hit puts the code in the wrong box."""
        from captchakraken.page_solver import TEXT_INPUT_SELECTORS

        vendor = FakeElement(box={"x": 10.0, "y": 10.0, "width": 100.0, "height": 20.0})
        generic = FakeElement(box={"x": 10.0, "y": 60.0, "width": 100.0, "height": 20.0})
        scope = _Scope({"input.mtcap-inputtext": vendor, "input[type=text]": generic})
        assert _solver()._find_control(scope, TEXT_INPUT_SELECTORS) is vendor

    def test_the_generic_fallback_still_finds_an_unnamed_box(self):
        """Most pages are this case: our own Tier 3 fixtures render neither
        vendor's class names, and vendors rename theirs without notice."""
        from captchakraken.page_solver import TEXT_INPUT_SELECTORS

        box = FakeElement()
        scope = _Scope({"input[type=text]": box})
        assert _solver()._find_control(scope, TEXT_INPUT_SELECTORS) is box

    def test_an_invisible_box_is_not_the_box(self):
        from captchakraken.page_solver import TEXT_INPUT_SELECTORS

        hidden = FakeElement(visible=False)
        real = FakeElement()
        scope = _Scope({"input#captchaCode": hidden, "input[type=text]": real})
        assert _solver()._find_control(scope, TEXT_INPUT_SELECTORS) is real

    def test_the_code_is_typed_character_by_character(self):
        page = _typing_page()
        scope = _Scope({"input[type=text]": FakeElement()})
        assert _solver()._execute_type(page, scope, {"text": "aB3d"}) is True
        assert page.keyboard.typed == ["a", "B", "3", "d"]

    def test_the_mouse_travels_to_the_box_before_typing(self):
        """Focus by click, at a real coordinate — not element.fill(), which
        sets the value with no pointer event and no keystrokes at all."""
        page = _typing_page()
        field = FakeElement(box={"x": 200.0, "y": 300.0, "width": 120.0, "height": 24.0})
        _solver()._execute_type(page, _Scope({"input[type=text]": field}), {"text": "x"})
        pressed_at = next(x for kind, x, _ in page.mouse.log if kind == "down")
        assert 200.0 <= pressed_at <= 320.0

    def test_a_retry_clears_the_previous_attempt(self):
        """Round 2 arrives with round 1's answer still in the box. Typing over
        the top of it appends, and submits a string the model never read."""
        page = _typing_page()
        _solver()._execute_type(page, _Scope({"input[type=text]": FakeElement()}), {"text": "ok"})
        assert page.keyboard.pressed and page.keyboard.pressed[0] == "Control+A"

    def test_no_box_means_nothing_was_done(self):
        """Must report False, not True: the caller counts this as an
        interaction, and a solve that 'interacted' without acting spins the
        outer loop until the deadline instead of failing."""
        page = _typing_page()
        assert _solver()._execute_type(page, _Scope({}), {"text": "abc"}) is False
        assert page.keyboard.typed == []

    def test_an_empty_answer_is_not_typed(self):
        page = _typing_page()
        assert _solver()._execute_type(page, _Scope({"input[type=text]": FakeElement()}),
                                       {"text": ""}) is False


class TestSlideGeometry:
    """The algebra behind the closed loop, in isolation.

    Each probe measures  width = piece_width + ratio x offset,  where the width
    spans the piece's original left edge to its current right edge.
    """

    def test_two_probes_recover_both_unknowns(self):
        # piece 40px wide, follows the handle 1:1.
        piece_w, ratio = PageSolver._solve_slide_geometry([(24.0, 64.0), (64.0, 104.0)], 400.0)
        assert (round(piece_w, 6), round(ratio, 6)) == (40.0, 1.0)

    def test_a_geared_slider_is_measured_not_assumed(self):
        # Tencent-style: the piece moves further than the handle. Assuming 1:1
        # here would stop the piece short of the gap every time.
        piece_w, ratio = PageSolver._solve_slide_geometry([(20.0, 70.0), (60.0, 150.0)], 400.0)
        assert (round(piece_w, 6), round(ratio, 6)) == (30.0, 2.0)

    def test_one_probe_falls_back_to_a_stated_one_to_one(self):
        piece_w, ratio = PageSolver._solve_slide_geometry([(24.0, 64.0)], 400.0)
        assert (piece_w, ratio) == (40.0, 1.0)

    def test_an_absurd_ratio_is_rejected_rather_than_steered_by(self):
        # A redraw between probes makes the two widths unrelated. A ratio of
        # ~0.02 solved from that would demand a handle offset of thousands of
        # pixels — off the track, into the page, and on camoufox a hung move.
        _, ratio = PageSolver._solve_slide_geometry([(24.0, 64.0), (64.0, 65.0)], 400.0)
        assert ratio == 1.0

    def test_a_piece_wider_than_the_widget_is_not_a_piece(self):
        piece_w, _ = PageSolver._solve_slide_geometry([(24.0, 390.0), (64.0, 430.0)], 400.0)
        assert piece_w is None

    def test_no_measurements_at_all_is_reported_as_such(self):
        assert PageSolver._solve_slide_geometry([], 400.0) == (None, 1.0)


class TestSlideDriver:
    """The control loop, with the CV stubbed by a simulation of a real slider.

    The simulated widget: a 40px piece starting 10px from the left, carried 1:1
    by a handle. `_track_piece` returns what the real one would — the union of
    the vacated ground and the piece's current position.
    """

    PIECE_LEFT, PIECE_W = 10.0, 40.0

    def _rig(self, target_px: float, widget_w: float = 400.0, dpr: float = 1.0):
        """`dpr` is the screen's device-pixel ratio, and it is the one thing in
        here that is not the same number twice: the shots come back in DEVICE
        pixels while every coordinate the loop steers by is a CSS pixel. Both
        are simulated in that space, and the masks the loop hands the CV are
        recorded on `page.excludes` so a test can read what space they arrived
        in."""
        solver = _solver()
        page = _typing_page()
        handle = FakeElement(box={"x": 120.0, "y": 420.0, "width": 40.0, "height": 30.0})
        element = FakeElement(box={"x": 100.0, "y": 100.0, "width": widget_w, "height": 400.0})
        element.screenshot = lambda path, **_: _write_png(  # type: ignore[method-assign]
            path, int(round(widget_w * dpr)), int(round(400.0 * dpr)))
        scope = _Scope({".geetest_slider_button": handle})
        start_x = handle._box["x"] + handle._box["width"] / 2
        page.excludes = []  # type: ignore[attr-defined]

        def fake_track(_element, _before, _after, exclude):
            page.excludes.append(list(exclude))  # type: ignore[attr-defined]
            offset = page.mouse.moves[-1][0] - start_x
            right = self.PIECE_LEFT + self.PIECE_W + offset
            return [int(self.PIECE_LEFT * dpr), 0, int(round(right * dpr)), int(20 * dpr)]

        solver._track_piece = fake_track  # type: ignore[method-assign]
        frac = target_px / widget_w
        action = {"target_bounding_box": [frac, 0.4, frac, 0.6]}
        return solver, page, element, scope, action, start_x

    def test_the_handle_is_pressed_not_the_gap(self):
        """The model names the SLOT. Pressing there grabs the background image;
        nothing moves, and the puzzle is failed without a single error."""
        solver, page, element, scope, action, _ = self._rig(target_px=150.0)
        solver._execute_slide(page, element, scope, action, element._box)
        pressed_at = next(x for kind, x, _ in page.mouse.log if kind == "down")
        assert 120.0 <= pressed_at <= 160.0  # inside the handle, not near 250

    def test_the_piece_is_steered_onto_the_slot(self):
        # piece centre = PIECE_LEFT + PIECE_W/2 + offset = 30 + offset,
        # so a slot at 150px within the widget wants offset 120.
        solver, page, element, scope, action, start_x = self._rig(target_px=150.0)
        solver._execute_slide(page, element, scope, action, element._box)
        released_at = next(x for kind, x, _ in reversed(page.mouse.log) if kind == "move")
        assert abs((released_at - start_x) - 120.0) <= solver.config.slide_tolerance_px

    def test_the_button_is_not_released_before_the_piece_arrives(self):
        """Releasing IS the submit on every one of these puzzles — there is no
        Verify button to reconsider at. A release during the probes would
        submit the probe offset as the answer."""
        solver, page, element, scope, action, _ = self._rig(target_px=150.0)
        solver._execute_slide(page, element, scope, action, element._box)
        kinds = [k for k, _, _ in page.mouse.log]
        assert kinds.count("down") == 1 and kinds.count("up") == 1
        assert kinds.index("up") == len(kinds) - 1

    def test_a_geared_widget_still_lands(self):
        solver, page, element, scope, action, start_x = self._rig(target_px=200.0)

        def geared(_element, _before, _after, _exclude):
            offset = page.mouse.moves[-1][0] - start_x
            return [int(self.PIECE_LEFT), 0,
                    int(round(self.PIECE_LEFT + self.PIECE_W + 2.0 * offset)), 20]

        solver._track_piece = geared  # type: ignore[method-assign]
        solver._execute_slide(page, element, scope, action, element._box)
        released_at = next(x for kind, x, _ in reversed(page.mouse.log) if kind == "move")
        # piece centre = 30 + 2*offset -> 200 wants offset 85.
        assert abs((released_at - start_x) - 85.0) <= solver.config.slide_tolerance_px

    def test_a_hidpi_screen_still_lands_the_piece(self):
        """Regression, 2026-08-27. Tier 3's mobile arm drove this fixture on a
        Pixel 7 — device-pixel ratio 2.625 — and the slider missed on every
        attempt while the mouse arm solved the same three boards. The loop
        measures the piece in the SHOT's pixels and steers the handle in the
        PAGE's, and those are the same number only at 1x. At 2.625 the widths
        come back 2.6x too wide, `_solve_slide_geometry` rejects them as larger
        than the widget, and the drive falls back to open-loop guessing."""
        solver, page, element, scope, action, start_x = self._rig(target_px=150.0, dpr=2.625)
        solver._execute_slide(page, element, scope, action, element._box)
        released_at = next(x for kind, x, _ in reversed(page.mouse.log) if kind == "move")
        assert abs((released_at - start_x) - 120.0) <= solver.config.slide_tolerance_px

    def test_the_handle_mask_is_in_the_shot_s_pixel_space(self):
        """`changed_bbox` masks in IMAGE pixels. A CSS-pixel band handed to it on
        a 2.625x screen masks a strip of empty card ABOVE the handle and leaves
        the handle itself unmasked — the largest moving thing in frame, and what
        the loop would then measure as the piece."""
        solver, page, element, scope, action, _ = self._rig(target_px=150.0, dpr=2.625)
        solver._execute_slide(page, element, scope, action, element._box)
        # The handle's own band, in the shot's pixels: y 420..450 page = 320..350
        # within the element, x across the full 400-wide widget.
        x1, y1, x2, y2 = page.excludes[0]
        assert y1 <= 320.0 * 2.625 and y2 >= 350.0 * 2.625
        assert x1 == 0.0 and x2 == pytest.approx(400.0 * 2.625, abs=1.0)

    def test_a_sliderless_widget_drags_the_piece_itself(self):
        """Lemin's 'cropped' puzzle has no track — you drag the piece onto the
        gap. Same answer from the model, because the two are indistinguishable
        in the picture; different gesture entirely."""
        solver = _solver()
        page = _typing_page()
        piece = FakeElement(box={"x": 140.0, "y": 220.0, "width": 40.0, "height": 40.0})
        element = FakeElement(box={"x": 100.0, "y": 100.0, "width": 400.0, "height": 400.0})
        scope = _Scope({".lemin-cropped-puzzle-piece": piece})
        action = {"target_bounding_box": [0.5, 0.3, 0.5, 0.4]}
        assert solver._execute_slide(page, element, scope, action, element._box) is True
        released_at = next(x for kind, x, _ in reversed(page.mouse.log) if kind == "move")
        assert abs(released_at - (100.0 + 200.0)) < 1.0  # dragged to the slot itself

    def test_nothing_draggable_at_all_reports_no_action(self):
        solver = _solver()
        page = _typing_page()
        element = FakeElement()
        action = {"target_bounding_box": [0.5, 0.3, 0.5, 0.4]}
        assert solver._execute_slide(page, element, _Scope({}), action, element._box) is False
        assert [k for k, _, _ in page.mouse.log if k == "down"] == []

    def test_an_invisible_piece_never_stops_the_drag_hanging(self):
        """When the CV cannot resolve the piece at all, the loop must still
        release the mouse. A slide that returns with the button held wedges
        every later input event on the page."""
        solver, page, element, scope, action, _ = self._rig(target_px=150.0)
        solver._track_piece = lambda *_: None  # type: ignore[method-assign]
        solver._execute_slide(page, element, scope, action, element._box)
        assert [k for k, _, _ in page.mouse.log][-1] == "up"


class TestPieceTracking:
    """The CV primitive itself, against images with a known moving rectangle."""

    def _frame(self, path, piece_x, handle_x):
        import numpy as np
        try:
            import cv2
        except ImportError:  # pragma: no cover
            pytest.skip("cv2 not available")
        img = np.zeros((120, 400, 3), dtype=np.uint8)
        img[20:60, piece_x:piece_x + 40] = 255           # the piece
        img[90:110, handle_x:handle_x + 30] = 200        # the slider handle
        cv2.imwrite(str(path), img)

    def test_the_union_spans_the_vacated_ground_to_the_new_edge(self, tmp_path):
        from captchakraken.tool_calls.track_piece import changed_bbox

        a, b = tmp_path / "a.png", tmp_path / "b.png"
        self._frame(a, piece_x=10, handle_x=10)
        self._frame(b, piece_x=70, handle_x=70)
        bbox = changed_bbox(str(a), str(b), exclude=[0, 85, 400, 115])
        assert bbox is not None
        # left = the piece's ORIGINAL left edge, right = its CURRENT right edge.
        assert bbox[0] == 10 and bbox[2] == 110
        assert bbox[2] - bbox[0] == 40 + 60  # piece width + travel

    def test_the_handle_is_masked_out_of_the_measurement(self, tmp_path):
        """Without the mask the handle is the widest moving thing in frame and
        the loop steers by the handle's own position — which is always exactly
        where it was told to go, so the piece never converges."""
        from captchakraken.tool_calls.track_piece import changed_bbox

        a, b = tmp_path / "a.png", tmp_path / "b.png"
        self._frame(a, piece_x=10, handle_x=10)
        self._frame(b, piece_x=30, handle_x=300)
        masked = changed_bbox(str(a), str(b), exclude=[0, 85, 400, 115])
        unmasked = changed_bbox(str(a), str(b))
        assert masked is not None and masked[2] == 70      # piece only
        assert unmasked is not None and unmasked[2] == 330  # handle dominates

    def test_a_still_frame_reports_nothing_moved(self, tmp_path):
        from captchakraken.tool_calls.track_piece import changed_bbox

        a, b = tmp_path / "a.png", tmp_path / "b.png"
        self._frame(a, piece_x=10, handle_x=10)
        self._frame(b, piece_x=10, handle_x=10)
        assert changed_bbox(str(a), str(b), exclude=[0, 85, 400, 115]) is None


class TestSlideSubmitPolicy:
    """A completed slide has ALREADY submitted."""

    def _run(self, actions):
        solver = _solver()
        page = _typing_page()
        handle = FakeElement(box={"x": 120.0, "y": 420.0, "width": 40.0, "height": 30.0})
        verify = FakeElement(box={"x": 300.0, "y": 500.0, "width": 80.0, "height": 30.0})
        scope = _Scope({".geetest_slider_button": handle, ".button-submit": verify})
        element = FakeElement(box={"x": 100.0, "y": 100.0, "width": 400.0, "height": 400.0})
        element._frame = scope

        solver._settle_or_animated = lambda _e: False           # type: ignore[method-assign]
        solver._solve_frame_freshness_guarded = (               # type: ignore[method-assign]
            lambda _el, shot, fn: fn(shot))
        solver._get_solution = lambda *_a, **_k: (actions, [])   # type: ignore[method-assign]
        solver._track_piece = lambda *_: [10, 0, 200, 20]        # type: ignore[method-assign]

        performed, _ = solver._solve_single(page, element, None)
        return performed, [k for k, _, _ in page.mouse.log]

    def test_a_slide_is_not_followed_by_a_verify_click(self):
        """Letting go of the handle is the gesture these puzzles grade, and none
        of them ships a Verify button. Anything the generic finder turns up
        afterwards belongs to the HOST page — pressing it submits the form the
        captcha was guarding, while the verdict is still in flight."""
        from captchakraken.action_types import DragAction

        slide = DragAction(action="drag", source_bounding_box=None,
                           target_bounding_box=[0.35, 0.4, 0.4, 0.6])
        performed, kinds = self._run([slide])
        assert performed is True
        assert kinds.count("down") == 1, "the only press should be the slider handle"
        assert kinds.count("up") == 1

    def test_the_gate_does_not_suppress_an_ordinary_submit(self):
        """`not slid` must narrow the submit policy for slides ONLY. A `done`
        answer still has to press Verify to advance the round — that is how the
        driver gets past a challenge it had nothing to say about, and
        suppressing it would hang every such round."""
        from captchakraken.action_types import DoneAction

        performed, kinds = self._run([DoneAction(action="done")])
        assert kinds.count("down") == 1, "the Verify click"
        # Reported as an interaction, though `done` performed no action of its
        # own: the caller aborts any round that reports none, so returning
        # False here would submit the puzzle and then give up on it a line
        # later. See TestInlineWidgetSubmit.
        assert performed is True

    def test_a_slide_action_constructs_at_all(self):
        """`DragAction(source_bounding_box=None)` used to raise ValidationError:
        the field defaulted to None but was not typed Optional, and pydantic
        validates a value that is PASSED even when it equals the default. Every
        slide would have died converting a correct answer into an action."""
        from captchakraken.action_types import DragAction

        action = DragAction(action="drag", source_bounding_box=None,
                            target_bounding_box=[0.1, 0.2, 0.3, 0.4])
        assert action.source_bounding_box is None


class TestSlideProbeBookkeeping:
    """The correction loop must steer from the offset its READING belongs to.

    Regression: the base was indexed by how many probes succeeded
    (`probes[len(widths) - 1]`). When probe 1 fails to resolve and probe 2
    works, that indexes to probe 1's offset while the measurement came from
    probe 2's — so the very first correction is computed against a position the
    piece was never at, and the puzzle is failed with a confident-looking log.
    """

    def test_the_correction_base_follows_the_last_successful_reading(self):
        solver = _solver()
        page = _typing_page()
        handle = FakeElement(box={"x": 120.0, "y": 420.0, "width": 40.0, "height": 30.0})
        element = FakeElement(box={"x": 100.0, "y": 100.0, "width": 400.0, "height": 400.0})
        scope = _Scope({".geetest_slider_button": handle})
        start_x = 140.0
        probes = solver.config.slide_probe_offsets_px

        # Probe 1 resolves nothing (a spinner, a redraw); probe 2 does.
        calls = {"n": 0}

        def flaky(_element, _before, _after, _exclude):
            calls["n"] += 1
            if calls["n"] == 1:
                return None
            offset = page.mouse.moves[-1][0] - start_x
            return [10, 0, int(round(10 + 40 + offset)), 20]

        solver._track_piece = flaky  # type: ignore[method-assign]
        target_px = 150.0
        frac = target_px / 400.0
        solver._execute_slide(page, element, scope,
                              {"target_bounding_box": [frac, 0.4, frac, 0.6]}, element._box)

        released_at = next(x for kind, x, _ in reversed(page.mouse.log) if kind == "move")
        # piece centre = 30 + offset, so a slot at 150 wants offset 120 —
        # whichever probe happened to be the one that resolved.
        assert abs((released_at - start_x) - 120.0) <= solver.config.slide_tolerance_px
        assert probes[0] != probes[-1], "the probes must differ or this proves nothing"


class TestTypedAnswerIsSubmitted:
    """A code the model read has to be SENT, not just typed.

    The submit policy was:

        should_submit = not slid and (
            not performed_action or puzzle_source == "hcaptcha" or is_recaptcha_one_shot
        )

    A distorted-text captcha is `puzzle_source == "unknown"` — that is precisely
    how `text_mode` is detected, since neither hCaptcha nor reCAPTCHA has ever
    served a typed challenge — and typing sets `performed_action = True`. So
    every clause was False and Verify was never pressed. The driver read the
    code correctly, typed it into the box, left it sitting there, and looped
    until the overall deadline reported "captcha still detected".

    BotDetect, MTCaptcha and Yandex all ship a submit button, so this affected
    the whole family in production, not only the Tier 3 fixtures where it was
    found.
    """

    def _run(self, actions):
        solver = _solver()
        page = _typing_page()
        field = FakeElement(box={"x": 120.0, "y": 300.0, "width": 200.0, "height": 30.0})
        verify = FakeElement(box={"x": 300.0, "y": 500.0, "width": 80.0, "height": 30.0})
        scope = _Scope({"input[type=text]": field, ".button-submit": verify})
        element = FakeElement(box={"x": 100.0, "y": 100.0, "width": 400.0, "height": 400.0})
        element._frame = scope

        solver._settle_or_animated = lambda _e: False            # type: ignore[method-assign]
        solver._solve_frame_freshness_guarded = (                 # type: ignore[method-assign]
            lambda _el, shot, fn: fn(shot))
        solver._get_solution = lambda *_a, **_k: (actions, [])    # type: ignore[method-assign]

        performed, _ = solver._solve_single(page, element, None)
        return performed, [k for k, _, _ in page.mouse.log], page

    def test_typing_is_followed_by_a_verify_click(self):
        from captchakraken.action_types import TypeAction

        performed, kinds, _ = self._run([TypeAction(action="type", text="5T63")])
        assert performed is True
        # One press to focus the field, one to submit it.
        assert kinds.count("down") == 2, (
            "the typed code was never submitted — only the field was clicked"
        )

    def test_a_click_answer_is_submitted_too(self):
        """A selection is an answer, and the widget drew a button to send it
        with.

        This used to assert the opposite, on the reasoning that these boards
        fade and re-round rather than being one-shot. They do not — the ones
        that grade themselves mid-selection draw no submit control at all, so
        `verify_button` is None and nothing is pressed either way. What the
        exclusion cost was a whole extra model call per puzzle, spent asking a
        board we had already answered correctly whether it was `done`."""
        from captchakraken.action_types import ClickAction

        performed, kinds, _ = self._run(
            [ClickAction(action="click", target_bounding_boxes=[[0.4, 0.4, 0.5, 0.5]])])
        assert performed is True
        # One press for the tile, one for Verify.
        assert kinds.count("down") == 2, "the selection was never submitted"

    def test_button_discovery_never_escapes_the_widget(self):
        """What keeps the press ours to make.

        The submit of the FORM a captcha guards sits on the host page, and
        pressing it mid-solve would send that form while the verdict is still in
        flight. Nothing about the ANSWER can rule that out — the protection is
        that discovery is relative to `scope`, the vendor's own widget
        container, so a button outside it is unreachable by construction. That
        is the invariant, and it is why a click answer can now be submitted
        without a `typed` gate in front of it.
        """
        from captchakraken.action_types import ClickAction

        solver = _solver()
        page = _typing_page()
        # A host-page "Submit", on the PAGE and outside the widget.
        page._elements["button"] = FakeElement(
            box={"x": 600.0, "y": 700.0, "width": 80.0, "height": 30.0}, text="Submit")
        element = FakeElement(box={"x": 100.0, "y": 100.0, "width": 400.0, "height": 400.0})
        element._frame = None
        element.query_selector = lambda _sel: None   # type: ignore[method-assign]

        solver._settle_or_animated = lambda _e: False            # type: ignore[method-assign]
        solver._solve_frame_freshness_guarded = (                 # type: ignore[method-assign]
            lambda _el, shot, fn: fn(shot))
        solver._get_solution = lambda *_a, **_k: (                # type: ignore[method-assign]
            [ClickAction(action="click", target_bounding_boxes=[[0.4, 0.4, 0.5, 0.5]])], [])

        solver._solve_single(page, element, None)
        kinds = [k for k, _, _ in page.mouse.log]
        assert kinds.count("down") == 1, (
            "pressed a Submit that belongs to the page, not to the captcha"
        )

    def test_a_widget_that_is_not_an_iframe_still_gets_its_verify_pressed(self):
        """The Verify button was looked up ONLY inside `element.content_frame()`.

        A BotDetect/MTCaptcha/Yandex widget is markup on the host page, not a
        vendor iframe, so `content_frame()` is None, `verify_button` was never
        even queried, and the typed code went nowhere — while `scope`
        (`content_frame() or element`) had already found the text box one line
        earlier. Two different containers for two halves of the same
        interaction.

        Searching `scope` rather than `page`: it is the captcha container, so a
        stray "Submit" belonging to the form the captcha guards stays out of
        reach.
        """
        from captchakraken.action_types import TypeAction

        solver = _solver()
        page = _typing_page()
        field = FakeElement(box={"x": 120.0, "y": 300.0, "width": 200.0, "height": 30.0})
        verify = FakeElement(box={"x": 300.0, "y": 500.0, "width": 80.0, "height": 30.0},
                             text="Verify")
        element = FakeElement(box={"x": 100.0, "y": 100.0, "width": 400.0, "height": 400.0})
        # No content_frame: the widget is on the host page.
        element._frame = None
        element._children = {"input[type=text]": field}
        element.query_selector = lambda sel: (                  # type: ignore[method-assign]
            field if "input" in sel else (verify if "button" in sel else None))

        solver._settle_or_animated = lambda _e: False            # type: ignore[method-assign]
        solver._solve_frame_freshness_guarded = (                 # type: ignore[method-assign]
            lambda _el, shot, fn: fn(shot))
        solver._get_solution = lambda *_a, **_k: (                # type: ignore[method-assign]
            [TypeAction(action="type", text="5T63")], [])

        performed, _ = solver._solve_single(page, element, None)
        kinds = [k for k, _, _ in page.mouse.log]
        assert performed is True
        assert kinds.count("down") == 2, (
            "no Verify press on a non-iframe widget — the typed code was never sent"
        )


class TestSliderlessFreeDrag:
    """Lemin's "cropped": a jigsaw with no rail, where the piece itself is
    carried onto the photo.

    The slide path handles it by dragging the draggable piece straight to the
    model's slot — but it moved to the target's X while HOLDING THE PIECE'S OWN
    Y, which is the tray. On a rail that is right and not even a choice: the
    handle only travels horizontally. Here the slot is 250 px up the card, so
    the piece slid along the tray and stopped, and every attempt submitted a
    piece that had never left it.

    Only reachable from a SOURCELESS drag, which `lemin_cropped` no longer
    emits — but every model published before that data change still does, and
    the shipped client has to keep working with them.
    """

    def _run(self, target_bbox):
        piece = FakeElement(box={"x": 130.0, "y": 620.0, "width": 60.0, "height": 60.0})
        scope = _Scope({".lemin-cropped-puzzle-piece": piece})
        element = FakeElement(box={"x": 100.0, "y": 100.0, "width": 400.0, "height": 600.0})
        element._frame = scope

        solver = _solver()
        page = _typing_page()
        ok = solver._execute_slide(
            page, element, scope,
            {"target_bounding_box": target_bbox, "source_bounding_box": None},
            element.bounding_box())
        return ok, page.mouse.log

    def test_the_piece_is_carried_to_the_slot_not_along_the_tray(self):
        # Slot at 40% across, 30% down a 400x600 widget pinned at (100, 100):
        # x = 100 + 160 = 260, y = 100 + 180 = 280.
        ok, log = self._run([0.375, 0.283, 0.425, 0.317])
        assert ok is True
        release_y = [y for kind, _, y in log if kind == "move"][-1]
        assert abs(release_y - 280.0) < 2.0, (
            f"released at y={release_y}: the piece was dragged along the tray "
            f"(y≈650) instead of up to the slot (y≈280)"
        )


class TestInlineWidgetSubmit:
    """The answer goes into an inline widget and is never sent.

    Eight vendors render into the HOST PAGE rather than a vendor iframe
    (GeeTest, Yidun, Tencent, Yandex, Lemin, Prosopo, MTCaptcha, BotDetect), so
    `element.content_frame()` is None for all of them. The Verify button was
    looked up ONLY inside that frame, so on those widgets it was never even
    SEARCHED FOR — the solver clicked the right tiles, or dropped the piece in
    the right hole, and then sat there re-solving a puzzle it had already
    answered until the loop cap or the deadline.

    Measured on the Tier 3 fixtures (run 20260805-235956), it is the single
    biggest failure class: `lemin_cropped` and `prosopo_grid_3x3` abort with
    "performed no interactions", and eleven more types burn all ten solve
    loops. Every one of them is an inline vendor.
    """

    def _run(self, actions, *, verify_named="Verify"):
        """One `_solve_single` against an inline widget that draws a Verify."""
        verify = FakeElement(box={"x": 300.0, "y": 500.0, "width": 80.0, "height": 30.0},
                             text=verify_named)
        element = FakeElement(box={"x": 100.0, "y": 100.0, "width": 400.0, "height": 400.0})
        element._frame = None  # inline: markup on the host page, no iframe
        # The widget's own button, and nothing else — `scope` is the container,
        # so a host-page Submit is out of reach by construction.
        element.query_selector = lambda sel: (              # type: ignore[method-assign]
            verify if "button" in sel else None)

        solver = _solver()
        page = _typing_page()
        solver._settle_or_animated = lambda _e: False        # type: ignore[method-assign]
        solver._solve_frame_freshness_guarded = (             # type: ignore[method-assign]
            lambda _el, shot, fn: fn(shot))
        solver._get_solution = lambda *_a, **_k: (actions, [])  # type: ignore[method-assign]

        performed, _ = solver._solve_single(page, element, None)
        return performed, [k for k, _, _ in page.mouse.log]

    def test_done_on_an_inline_widget_presses_verify(self):
        """`prosopo_grid_3x3`, exactly as traced: round 1 clicks four tiles,
        round 2 the model answers `done` because everything matching is already
        selected — and `done` performs no action, so the outer loop's
        "performed no interactions" guard fired and aborted a puzzle whose only
        remaining step was to press the button."""
        from captchakraken.action_types import DoneAction

        performed, kinds = self._run([DoneAction(action="done")])
        assert kinds.count("down") == 1, (
            "the tiles were selected and the widget was never submitted"
        )
        # The press IS the interaction. Reporting False here re-arms the very
        # guard this fixes, one loop later.
        assert performed is True

    def test_a_placed_piece_is_submitted(self):
        """`lemin_cropped`, exactly as traced: the model drags the piece into
        the gap and then, round after round, nudges it by a pixel — because
        nothing ever tells the widget to grade it. A piece placement has no
        count to reach and no release to be judged on, so unlike a click round
        it will never be followed by a `done`."""
        from captchakraken.action_types import DragAction

        performed, kinds = self._run([DragAction(
            action="drag",
            source_bounding_box=[0.20, 0.83, 0.23, 0.86],
            target_bounding_box=[0.70, 0.64, 0.73, 0.67])])
        assert performed is True
        # One press-drag-release for the piece, one press for Verify.
        assert kinds.count("down") == 2, "the piece was placed and never submitted"

    def test_a_click_round_is_submitted_without_waiting_for_done(self):
        """The round that used to cost a model call.

        `prosopo_grid_3x3` answered all four tiles correctly on its first call,
        then spent 6.2 s of a 13.8 s solve on a second call whose entire content
        was `done`, purely so the press had a round to happen on."""
        from captchakraken.action_types import ClickAction

        performed, kinds = self._run(
            [ClickAction(action="click", target_bounding_boxes=[[0.4, 0.4, 0.5, 0.5]])])
        assert performed is True
        assert kinds.count("down") == 2, "the selection waited for a `done` round"
