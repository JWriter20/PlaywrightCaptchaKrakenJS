"""
Humanisation is a pluggable INPUT DEVICE, not a realism dial.

These pin the three things that made it worth extracting from `page_solver`:

  - **mobile emits touch and NOTHING else.** A mousemove at a touch-only widget
    is the wrong event, not a weaker one — the page's touch handlers never fire
    and the solve fails for a reason nothing reports. So the strongest test here
    asserts an ABSENCE: driving a whole solve in mobile mode must not touch
    `page.mouse` once.
  - **the Appium payload is the W3C one.** It is built by hand rather than
    through a client's action builder (so the package imports no Selenium), and
    a typo in it fails on a real handset and nowhere else.
  - **the mouse mode did not change.** It was measured; a refactor is not the
    place to revisit it.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from captchakraken import humanize  # noqa: E402
from captchakraken.humanize import (  # noqa: E402
    AppiumTouchBackend,
    MobileHumanizer,
    MouseHumanizer,
    NullHumanizer,
    PAUSE_KINDS,
    TouchBackend,
    resolve,
)
from captchakraken.page_solver import PageSolverConfig  # noqa: E402
from captchakraken.trajectory import generate_swipe  # noqa: E402


@pytest.fixture(autouse=True)
def _no_sleeping(monkeypatch):
    """Every mode's pauses are real sleeps; a test suite must not pay them."""
    monkeypatch.setattr(humanize, "_delay", lambda ms: None)


# ------------------------------------------------------------------ fakes


class RecordingMouse:
    def __init__(self) -> None:
        self.events: List[Any] = []

    def move(self, x: float, y: float) -> None:
        self.events.append(("move", x, y))

    def down(self) -> None:
        self.events.append(("down",))

    def up(self) -> None:
        self.events.append(("up",))


class RecordingPage:
    def __init__(self) -> None:
        self.mouse = RecordingMouse()
        self.viewport_size = {"width": 800, "height": 600}


class RecordingTouch(TouchBackend):
    name = "recording"

    def __init__(self) -> None:
        self.events: List[Any] = []

    def down(self, x: float, y: float) -> None:
        self.events.append(("down", x, y))

    def move(self, path) -> None:
        self.events.append(("move", len(path)))

    def up(self, x: float, y: float) -> None:
        self.events.append(("up", x, y))

    @property
    def kinds(self) -> List[str]:
        return [e[0] for e in self.events]


class RatioPage(RecordingPage):
    """A page that answers `window.devicePixelRatio`, or refuses to."""

    def __init__(self, dpr: Optional[float] = 1.0) -> None:
        super().__init__()
        self._dpr = dpr

    def evaluate(self, expression: str) -> Any:
        assert "devicePixelRatio" in expression
        if self._dpr is None:
            raise RuntimeError("no execution context")
        return self._dpr


class RecordingDriver:
    """A WebDriver that records the raw action chains it is asked to perform."""

    def __init__(self) -> None:
        self.chains: List[Any] = []

    def execute(self, command: str, params: Dict[str, Any]) -> None:
        assert command == "actions"
        self.chains.append(params["actions"])


# ------------------------------------------------------------- resolution


class TestResolve:
    def test_an_explicit_object_wins_over_everything(self, monkeypatch):
        monkeypatch.setenv("CAPTCHA_HUMANIZATION", "mobile")
        mine = NullHumanizer()
        cfg = PageSolverConfig(humanization="mouse", humanizer=mine)
        assert resolve(cfg) is mine

    def test_code_beats_the_environment(self, monkeypatch):
        # Deliberate, and the opposite of this package's model-identity vars:
        # which mode is right is a property of the PAGE, so an env var must not
        # silently flip a desktop solve to touch dispatch.
        monkeypatch.setenv("CAPTCHA_HUMANIZATION", "mobile")
        assert resolve(PageSolverConfig(humanization="none")).name == "none"

    def test_the_environment_is_read_when_the_code_says_nothing(self, monkeypatch):
        monkeypatch.setenv("CAPTCHA_HUMANIZATION", "mobile")
        assert resolve(PageSolverConfig()).name == "mobile"

    def test_the_default_is_the_historical_behaviour(self, monkeypatch):
        monkeypatch.delenv("CAPTCHA_HUMANIZATION", raising=False)
        assert resolve(PageSolverConfig()).name == "mouse"

    def test_a_typo_names_the_alternatives(self, monkeypatch):
        monkeypatch.delenv("CAPTCHA_HUMANIZATION", raising=False)
        with pytest.raises(ValueError) as exc:
            resolve(PageSolverConfig(humanization="touch"))
        assert "mobile" in str(exc.value) and "humanizer" in str(exc.value)

    def test_the_starting_position_is_honoured(self, monkeypatch):
        monkeypatch.delenv("CAPTCHA_HUMANIZATION", raising=False)
        assert resolve(PageSolverConfig(starting_mouse_position=(30.0, 40.0))).at == (30.0, 40.0)

    def test_every_mode_answers_the_whole_pause_vocabulary(self, monkeypatch):
        # An unknown kind must yield no wait rather than raise, so adding a
        # pause site cannot break a humanizer written against an older release.
        for mode in (MouseHumanizer(), MobileHumanizer(), NullHumanizer()):
            for kind in PAUSE_KINDS + ("a-kind-added-next-year",):
                assert mode._pause_ms(kind) >= 0.0


# ------------------------------------------------------------------ mouse


class TestMouse:
    def test_a_click_is_a_trajectory_then_a_press(self):
        page, human = RecordingPage(), MouseHumanizer(start=(10.0, 10.0))
        human.click(page, (400.0, 300.0))
        kinds = [e[0] for e in page.mouse.events]
        # Many moves (the trajectory), then exactly one down/up at the end.
        assert kinds[-2:] == ["down", "up"]
        assert kinds.count("move") > 5
        assert human.at == (400.0, 300.0)

    def test_it_still_hovers(self):
        assert MouseHumanizer().hovers is True


# ----------------------------------------------------------------- mobile


class TestMobile:
    def _human(self):
        backend = RecordingTouch()
        return MobileHumanizer(backend=backend), backend

    def test_a_move_with_no_finger_down_dispatches_nothing(self):
        human, backend = self._human()
        human.move(RecordingPage(), (300.0, 200.0))
        assert backend.events == []
        # …but the position is still recorded: the next touch lands there.
        assert human.at == (300.0, 200.0)

    def test_a_tap_wobbles_between_touchstart_and_touchend(self):
        # A tap with zero movement in between is a synthetic tap. The contact
        # centroid of a real finger rolls a pixel or two under pressure.
        human, backend = self._human()
        human.click(RecordingPage(), (120.0, 90.0))
        assert backend.kinds == ["down", "move", "up"]
        assert backend.events[0][1:] == (120.0, 90.0)

    def test_a_drag_travels_while_touching(self):
        human, backend = self._human()
        human.drag(RecordingPage(), (10.0, 10.0), (300.0, 140.0))
        assert backend.kinds == ["down", "move", "up"]
        assert backend.events[1][1] > 5   # a whole swipe path, not one jump
        assert human.at == (300.0, 140.0)

    def test_it_never_touches_the_mouse(self):
        page, (human, _) = RecordingPage(), self._human()
        human.click(page, (50.0, 50.0))
        human.drag(page, (50.0, 50.0), (200.0, 200.0))
        assert page.mouse.events == []

    def test_it_does_not_hover(self):
        # There is no cursor to rest anywhere, so every hover-for-realism
        # behaviour has to switch itself off.
        assert MobileHumanizer().hovers is False

    def test_reset_lifts_a_finger_a_previous_solve_left_down(self):
        # W3C input state is per SESSION: a solve that timed out inside the
        # slider leaves the pointer down, and the next one would start from a
        # finger already on the glass.
        human, backend = self._human()
        human.press(RecordingPage())
        backend.events.clear()
        human.reset(RecordingPage())
        assert backend.kinds == ["up"]
        human.reset(RecordingPage())          # idempotent
        assert backend.kinds == ["up"]

    def test_typing_clears_through_the_element_not_control_a(self):
        # There is no Control key on a phone keyboard, and no `page.keyboard` at
        # all on an Appium element.
        class Field:
            def __init__(self) -> None:
                self.cleared = 0
                self.keys: List[str] = []

            def clear(self) -> None:
                self.cleared += 1

            def send_keys(self, ch: str) -> None:
                self.keys.append(ch)

        human, _ = self._human()
        field = Field()
        assert human.type_text(RecordingPage(), field, "ab7") is True
        assert field.cleared == 1 and field.keys == ["a", "b", "7"]


# ------------------------------------------------------------------- none


class TestNone:
    def test_one_move_per_gesture_and_no_dwell(self):
        page, human = RecordingPage(), NullHumanizer(start=(10.0, 10.0))
        human.click(page, (400.0, 300.0))
        assert page.mouse.events == [("move", 400.0, 300.0), ("down",), ("up",)]

    def test_typing_is_one_fill(self):
        class Field:
            def __init__(self) -> None:
                self.value: Optional[str] = None

            def fill(self, text: str) -> None:
                self.value = text

        field = Field()
        assert NullHumanizer().type_text(RecordingPage(), field, "xyz") is True
        assert field.value == "xyz"


# ----------------------------------------------------------------- appium


class TestAppiumBackend:
    def test_the_chain_is_a_w3c_touch_pointer(self):
        driver = RecordingDriver()
        AppiumTouchBackend(driver).down(12.0, 34.0)
        (pointer,) = driver.chains[0]
        assert pointer["type"] == "pointer"
        assert pointer["parameters"] == {"pointerType": "touch"}
        assert [a["type"] for a in pointer["actions"]] == ["pointerMove", "pointerDown"]
        assert (pointer["actions"][0]["x"], pointer["actions"][0]["y"]) == (12, 34)
        assert pointer["actions"][0]["origin"] == "viewport"

    def test_a_leg_is_one_chain_with_per_sample_durations(self):
        # Batched on purpose: pacing a 90Hz path over the wire is not pacing.
        driver = RecordingDriver()
        AppiumTouchBackend(driver).move([(1.0, 2.0, 11.0), (3.0, 4.0, 12.6)])
        assert len(driver.chains) == 1
        actions = driver.chains[0][0]["actions"]
        assert [a["duration"] for a in actions] == [11, 13]

    def test_press_and_release_are_separate_performs(self):
        # W3C input state persists per session, which is what lets the slider
        # press, screenshot, steer, screenshot and only then release.
        driver = RecordingDriver()
        backend = AppiumTouchBackend(driver)
        backend.down(0.0, 0.0)
        backend.move([(5.0, 5.0, 8.0)])
        backend.up(5.0, 5.0)
        assert len(driver.chains) == 3
        assert driver.chains[-1][0]["actions"] == [{"type": "pointerUp", "button": 0}]

    def test_css_pixels_are_mapped_onto_the_device(self):
        # A real handset wants screen pixels; the two differ by devicePixelRatio
        # and by whatever chrome sits above the webview.
        driver = RecordingDriver()
        AppiumTouchBackend(driver, scale=3.0, origin=(0.0, 132.0)).down(10.0, 20.0)
        move = driver.chains[0][0]["actions"][0]
        assert (move["x"], move["y"]) == (30, 192)

    def test_a_driver_that_speaks_neither_call_says_so(self):
        with pytest.raises(RuntimeError) as exc:
            AppiumTouchBackend(object()).down(0.0, 0.0)
        assert "execute" in str(exc.value)


class TestAppiumScaleIsNotGuessed:
    """The transform is the one thing here that fails SILENTLY.

    A wrong `scale` does not raise on either side of the wire: the chain is
    valid W3C, the device performs it, and the finger lands somewhere else. The
    solve then fails looking exactly like a model that cannot read the puzzle —
    the same shape as the DPR bug in the slider's control loop, one seam over.
    So an unset scale on a session that reports it is not 1 is refused rather
    than dispatched, and `origin` is named in the refusal because it cannot be
    measured from here at all.
    """

    def test_a_hidpi_session_with_no_transform_refuses(self):
        driver = RecordingDriver()
        backend = AppiumTouchBackend(driver, page=RatioPage(3.0))
        with pytest.raises(RuntimeError) as exc:
            backend.down(10.0, 20.0)
        message = str(exc.value)
        assert "devicePixelRatio 3" in message and "scale" in message
        assert "origin" in message          # names the half it cannot measure
        assert not driver.chains            # refused, not dispatched

    def test_an_explicit_scale_is_taken_as_the_callers_word(self):
        # The caller who has already mapped the coordinates says so with an
        # explicit scale, and is not second-guessed by a ratio we read.
        driver = RecordingDriver()
        AppiumTouchBackend(driver, scale=1.0, page=RatioPage(3.0)).down(10.0, 20.0)
        move = driver.chains[0][0]["actions"][0]
        assert (move["x"], move["y"]) == (10, 20)

    def test_a_1x_session_needs_no_transform(self):
        driver = RecordingDriver()
        AppiumTouchBackend(driver, page=RatioPage(1.0)).down(10.0, 20.0)
        assert driver.chains

    def test_a_page_that_cannot_be_asked_is_not_refused(self):
        # Absent evidence is not evidence of a mismatch. Failing a solve over an
        # inability to MEASURE is the mirror of the bug this guards against —
        # same call `_shot_scale` makes when a screenshot will not read.
        driver = RecordingDriver()
        AppiumTouchBackend(driver, page=RatioPage(None)).down(10.0, 20.0)
        assert driver.chains

    def test_no_page_at_all_is_not_refused(self):
        driver = RecordingDriver()
        AppiumTouchBackend(driver).down(10.0, 20.0)
        assert driver.chains

    def test_the_ratio_is_read_once_not_per_gesture(self):
        # It is a round trip into the page, and a solve makes hundreds of these.
        page = RatioPage(1.0)
        reads = []
        original = page.evaluate
        page.evaluate = lambda e: (reads.append(e), original(e))[1]  # type: ignore[method-assign]
        backend = AppiumTouchBackend(RecordingDriver(), page=page)
        backend.down(1.0, 1.0)
        backend.move([(2.0, 2.0, 5.0)])
        backend.up(2.0, 2.0)
        assert len(reads) == 1

    def test_the_factory_hands_the_backend_its_page(self):
        # The check is worthless if the wiring never passes a page through.
        backend = humanize.touch_backend_for(RatioPage(3.0), RecordingDriver())
        with pytest.raises(RuntimeError):
            backend.down(0.0, 0.0)


# ------------------------------------------------------------------ swipe


class TestSwipe:
    def test_it_honours_the_trajectory_contract(self):
        points, timings = generate_swipe((10.0, 10.0), (400.0, 260.0))
        assert len(points) == len(timings)
        assert timings[0] == 0.0
        assert all(b >= a for a, b in zip(timings, timings[1:]))
        assert points[-1] == (400.0, 260.0)

    def test_a_finger_does_not_overshoot(self):
        # The mouse model's most recognisable tell is a hand arriving past a
        # target it cannot see under the cursor. A finger occludes its own
        # target and commits.
        #
        # STATED AS A DISTRIBUTION, AND SEEDED, because the per-sample jitter is
        # GAUSSIAN and therefore unbounded: "no draw ever exceeds 3.0 px" is not
        # a property this generator has, it is a property it has 99.9% of the
        # time. Measured over 20,000 unseeded draws on 2026-09-06 — p50 0.12,
        # p99 2.24, p99.9 2.97, max 4.33 — so the old form (20 draws, no seed)
        # went red on 1.9% of runs, and 3.7% of CI runs across two Python
        # versions. A gate that reddens once every twenty-seven runs for no
        # reason is a gate people learn to re-run rather than read, which is
        # exactly how a real overshoot would get waved through.
        #
        # The seed makes it a measurement rather than a dice roll. A change that
        # shifts the distribution still fails it; a change that draws an unlucky
        # tail no longer does.
        random.seed(20260907)
        overshoot = sorted(
            max(x for x, _ in generate_swipe((0.0, 0.0), (600.0, 0.0))[0]) - 600.0
            for _ in range(2000)
        )
        # The gesture as a whole commits: the overwhelming majority land on or
        # short of the target, and the tail is jitter rather than a hand
        # sailing past.
        assert overshoot[int(len(overshoot) * 0.99)] <= 3.0
        assert overshoot[-1] <= 8.0

    def test_a_zero_length_swipe_is_one_sample(self):
        assert generate_swipe((5.0, 5.0), (5.0, 5.0)) == ([(5.0, 5.0)], [0.0])
