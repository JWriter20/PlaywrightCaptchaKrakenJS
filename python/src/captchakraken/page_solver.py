"""
Browser page driver — the Python mirror of `js/src/solver.ts`.

WHAT THIS IS
The Python port used to be image-in / actions-out: you handed `CaptchaSolver` a
PNG and got back click/drag actions, and something else had to own the browser.
Everything that actually drives a page — finding the challenge iframe, waiting
for tiles to paint, clicking, submitting, deciding whether the vendor accepted
— lived only in the TypeScript driver. This module closes that gap so a Python
caller (camoufox's Python API, plain Playwright, patchright) can solve a captcha
end to end without a Node process.

THE SPLIT, WHICH IS THE SAME ON BOTH SIDES
  vision / CV / prompting  -> Python (`solver.py`, `planner.py`, `tool_calls/`)
  page driving + clicking  -> the driver (this file, or solver.ts)

The TS driver reaches the Python half by spawning the CLI and talking JSON over
a pipe. This driver calls the very same functions in-process — no subprocess, no
serialisation, no persistent CV worker to leak. That is the only intended
difference between the two drivers: everything about WHAT to click, WHEN a frame
is settled, and WHETHER a puzzle is supported is shared code, so the two cannot
drift on the parts that decide accuracy.

STRUCTURAL TYPING, NO BROWSER DEPENDENCY
Like `playwright-types.ts`, this module imports no browser package. The caller
passes whatever Playwright-compatible `page` they have; we duck-type the slice
we use. Importing `playwright` here would force it into every consumer's tree
and break across version skew, and camoufox users already have their own.

SYNC ONLY, FOR NOW
This mirrors the synchronous Playwright API (`Camoufox()`, `sync_playwright()`),
which is camoufox's headline Python interface. An async mirror is mechanical but
is NOT written yet — see `solve_captcha_on_page`'s docstring. Calling this from
an async event loop will not work, because sync Playwright handles cannot be
driven from one.
"""

from __future__ import annotations

import hashlib
import os
import random
import re
import shutil
import sys
import tempfile
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .action_types import CaptchaAction
from .solver import CaptchaSolver, UnsupportedCaptchaError
from .timing import PhaseBudget, timings_enabled
from .humanize import Humanizer, resolve as resolve_humanizer

DEBUG = os.getenv("CAPTCHA_DEBUG", "0") == "1"

# Read by planner.py, which turns it into the X-CK-Session header. Kept as a
# module constant so the name is defined in exactly one place per port.
_SESSION_ENV = "CAPTCHA_KRAKEN_SESSION"


def _log(message: str) -> None:
    # flush=True is load-bearing, not tidiness. A solve is a minutes-long
    # sequence of slow steps, and Python block-buffers stdout whenever it is not
    # a TTY — so piped or redirected (CI logs, `> run.log`, a supervisor) the
    # progress lines all appear at once when the process exits. A run that is
    # working then looks indistinguishable from one that is hung, which is
    # exactly the wrong signal from the one output people watch.
    print(f"[captchakraken] {message}", flush=True)


def _debug(message: str) -> None:
    if DEBUG:
        print(f"[captchakraken:debug] {message}", flush=True)


def _delay(ms: float) -> None:
    if ms > 0:
        time.sleep(ms / 1000.0)


def _tmp_png(prefix: str) -> str:
    fd, path = tempfile.mkstemp(prefix=f"{prefix}_", suffix=".png")
    os.close(fd)
    return path


def _unlink(path: Optional[str]) -> None:
    if path and os.path.exists(path):
        try:
            os.unlink(path)
        except OSError:
            pass  # best-effort; a leaked temp frame is not worth failing a solve


# --------------------------------------------------------------------------
# Errors — the taxonomy the outer loop branches on. The TS driver tags plain
# Errors with `.animated` / `.unsupported` booleans; Python gets real types,
# which is the same distinction expressed properly.
# --------------------------------------------------------------------------


class CaptchaSolveError(Exception):
    """Base for every failure this driver raises."""


class AnimatedChallengeError(CaptchaSolveError):
    """An animated challenge we could not RECORD.

    Note the narrowed meaning. "The challenge never stops moving" is no longer a
    failure: `video_solve_enabled` records the widget, slices the recording into
    keyframes and solves those. This is raised only when that path cannot get an
    artifact to work with — the element refuses to screenshot, or the recording
    decodes to nothing.
    """


class UnsupportedChallengeError(CaptchaSolveError):
    """A settled frame the model reports it cannot solve (e.g. hCaptcha click/drag)."""


class NoCaptchaFoundError(CaptchaSolveError):
    """No interactive widget — reCAPTCHA v3/invisible, or a click-triggered challenge."""


class PageClosedError(CaptchaSolveError):
    """The page, context or browser went away mid-solve.

    Split out from staleness deliberately. These two used to share one pattern,
    which meant a DEAD target was retried three times as if it were a live one
    mid-transition, and the error that finally escaped named whichever call ran
    last — `ElementHandle.screenshot`, usually — instead of saying the page had
    closed. That cost a real debugging session on the headed reCAPTCHA run.
    """


# Playwright surfaces these as messages, not types, so both drivers match on
# text. Kept as one pattern so the two lists can be diffed against each other.
#
# "Target closed" is NOT here on purpose — see `_CLOSED_TARGET_RE`. Everything in
# this list describes a handle that went stale while the PAGE IS STILL ALIVE, so
# re-detecting next round is the right response.
_STALE_HANDLE_RE = re.compile(
    r"Timeout .*exceeded|not visible|not attached|detached",
    re.IGNORECASE,
)

# A target that no longer exists. Nothing can be re-detected on it, so this is
# terminal rather than retryable.
_CLOSED_TARGET_RE = re.compile(
    r"Target (?:page, context or browser )?(?:has been )?closed|Session closed",
    re.IGNORECASE,
)


def answer_needs_element_box(actions: List[Dict[str, Any]]) -> bool:
    """Does this answer need the widget's box to be performed?

    Only a coordinate-bearing action does. ``done`` means "nothing left to
    click", which is exactly the answer a vendor that closes on success will be
    showing no widget for.

    An ALLOW-LIST of what needs nothing, not a list of what does: a coordinate
    action added later and forgotten here must default to needing the box,
    because that direction raises instead of clicking at the origin. Twin of
    `answerNeedsElementBox` in the JS port (rule 1c) — if one moves, move both.
    """
    return any((a or {}).get("action") != "done" for a in actions)


@dataclass
class SolveResult:
    """Mirror of the TS `SolveResult`."""

    is_solved: bool
    final_mouse_position: Tuple[float, float]
    token_usage: List[Dict[str, Any]] = field(default_factory=list)
    #: Where this solve's wall-clock went, in milliseconds per phase — the same
    #: partition `CAPTCHA_TIMINGS=1` prints, handed to the caller instead of to
    #: stderr. `{}` when the result was built outside a solve.
    #:
    #: Returned rather than only logged because "the solve took 12s" is not
    #: actionable and "the settle monitor spent 4s of it" is, and a caller
    #: measuring a fleet cannot scrape stderr from inside its own process.
    phases: Dict[str, float] = field(default_factory=dict)


@dataclass
class _GridSession:
    """Cached geometry for one reCAPTCHA grid puzzle. Mirror of `GridSession`."""

    grid_boxes: List[Sequence[int]]
    element_box: Dict[str, float]
    scale_x: float
    scale_y: float
    screenshot_w: int
    screenshot_h: int


def settle_verdict(samples, *, settle_frames: int, animated_after_ms: int) -> str:
    """The pixel-settle rule, over `(elapsed_ms, moved)` polls.

    Split out of the polling loop so the RULE can be tested against recorded
    timelines without a browser — see `tests/test_animated_is_detected.py`,
    which pins what it does to each shipped animated puzzle.

    Note what it cannot do: 'animated' needs a poll that MOVED at or after
    `animated_after_ms`, so any `settle_frames` run of stillness before then
    ends the wait as 'settled' first. Every animated puzzle we ship rests
    between screens for longer than that, which is why the caller also probes.
    """
    still_streak = 0
    for elapsed_ms, moved in samples:
        if moved:
            still_streak = 0
            if elapsed_ms >= animated_after_ms:
                return "animated"
        else:
            still_streak += 1
            if still_streak >= settle_frames:
                return "settled"
    return "timeout"


#: How unlike the chosen keyframe the widget must look before the gate calls it
#: a different board, and over how many polls. Measured on GeeTest svg: two
#: SCREENS of one board differ by 0.0056, a different board by 0.77 — two orders
#: of magnitude, so this is a plateau rather than a knife edge. Several polls
#: because a frame caught mid-swap can read high for a moment. Must match
#: NOT_THIS_BOARD_DIFF / _POLLS in the JS port.
#: How much the widget must change during inference to count as MOVING. A far
#: tighter number than `stale_frame_diff_threshold` (0.02) because it answers a
#: different question: 0.02 asks "did this answer go stale", this asks "is this
#: the same picture", which is the noise floor. Using the stale threshold for
#: both made the guard blind to GeeTest svg, whose screens differ by 0.0056.
#: Must match MOVED_DURING_INFERENCE_DIFF in the JS port.
_MOVED_DURING_INFERENCE_DIFF = 0.002
_NOT_THIS_BOARD_DIFF = 0.5
_NOT_THIS_BOARD_POLLS = 3


@dataclass
class PageSolverConfig:
    """
    Tunables, named to match the TS `CaptchaKrakenConfig` keys (snake_cased) so a
    value tuned on one driver can be found on the other.
    """

    # ── how the driver moves ───────────────────────────────────────────────
    #: "mouse" (default), "mobile" or "none". See `humanize.py`.
    #:
    #: Not a realism dial — a choice of INPUT DEVICE. "mobile" dispatches touch
    #: events with finger kinematics, and on a touch-only widget that is the
    #: difference between the page's handlers firing and not. "none" is the
    #: shortest legal path to the same DOM effect, for fixtures and for callers
    #: who humanise somewhere else in their stack.
    #:
    #: None means unset: `CAPTCHA_HUMANIZATION`, else "mouse".
    humanization: Optional[str] = None
    #: Your own `humanize.Humanizer`. Overrides `humanization` entirely — the
    #: driver then makes no decisions about pointer motion at all.
    humanizer: Optional[Humanizer] = None
    #: mobile only. The thing that is actually TOUCHED, when it is not the page
    #: object — an Appium/Selenium WebDriver on a real handset. Left None, the
    #: mode dispatches CDP touch events at the page it was given.
    touch_driver: Optional[Any] = None
    #: mobile only. CSS-pixel -> device-pixel transform for `touch_driver`,
    #: e.g. `{"scale": 3.0, "origin": (0, 132)}`. See AppiumTouchBackend; the
    #: default identity is right for browser emulation.
    touch_transform: Optional[Dict[str, Any]] = None
    #: Where the pointer starts. Setting it stops the first gesture of a solve
    #: crossing the whole window from the origin.
    starting_mouse_position: Optional[Tuple[float, float]] = None

    # ── which model answers ────────────────────────────────────────────────
    #: Force ONE expert of a routed model — "pixel", "grid", "video" or "text".
    #:
    #: A routed model (Abyss) is four adapters behind one endpoint, and the
    #: router is the prompt family the request is about to send, so the default
    #: (None) already reaches the right one with no help from the caller. This
    #: is the override: serve one arm, drive only the puzzles it owns, which is
    #: what a per-arm benchmark needs and what a licence holder pinning a single
    #: expert wants.
    #:
    #: Refused at construction against a model that serves one adapter — every
    #: model published so far — because a run that quietly measured the
    #: generalist and reported it as the expert is a number nobody can catch.
    #: None means unset: `CAPTCHA_EXPERT`, else route by family.
    expert: Optional[str] = None

    # 45s, and the loop count that fits inside it rather than one that needs
    # policing by the clock. A round costs ~4-7s once the waits below are paid,
    # so six rounds is the budget; ten never fitted and only ever expressed
    # itself as a timeout. The cap is a BACKSTOP — `max_no_progress_rounds` is
    # what is supposed to end a hopeless solve, and a solve that reaches this
    # number is a bug report, not a normal outcome.
    max_solve_loops: int = 6
    overall_solve_timeout_ms: int = 45_000
    #: Consecutive rounds producing the SAME answer before the solve is
    #: abandoned.
    #:
    #: At temperature 0 the model is a function of the picture, so an identical
    #: answer means an identical picture — and every answer this driver produces
    #: is EXECUTED, so the previous one already ran and moved nothing. Repeating
    #: it cannot do better; it just spends a round.
    #:
    #: Measured on recaptcha_grid_4x4 (fixture seed 20260730): the model answers
    #: `[2,6,7,9,10]` on round 1 and then `[2,6,7,10]` on rounds 2 THROUGH 10 —
    #: nine identical click sets, each one clicked, each one rejected, ending in
    #: "still detected after 10 solve loops" at 66.1s. Of that, 39.0s was pure
    #: waiting. Stopping on the second repeat ends it at round 4.
    #:
    #: 2 rather than 1: the first repeat also ARMS the animated probe, which is
    #: the one recovery worth trying (a cycling board reads as a still and
    #: answers the same way every round). One more round buys that chance.
    max_no_progress_rounds: int = 2
    post_solve_delay_ms: int = 1_200
    # How long to keep watching for a SUCCESS after submitting.
    #
    # That is the window's whole job, and it is why a wrong answer spends all of
    # it: the vendors emit nothing on a wrong answer, they just re-deal. Sizing
    # it therefore means asking how late a real success can arrive.
    #
    # MEASURED over 34 successful rounds across 20 puzzle types (fixture server,
    # two seeds each): min 3ms, p50 360ms, p90 378ms, max 528ms.
    #
    # Read that with care — those values are quantized by `..._poll_ms`, not by
    # any vendor. The clusters sit at one poll (~200ms), two polls (~370ms, the
    # `widget_gone >= 2` confirmation) and three (~520ms). So the measurement
    # mostly describes THIS loop, and the fixture's verdict is a local fetch,
    # which means real hCaptcha/reCAPTCHA round-trips are absent from it
    # entirely.
    #
    # 1000ms is therefore the measured worst case (528ms, ~264ms once the poll
    # below is halved) plus deliberate headroom for a vendor round-trip nobody
    # has measured yet. Down from 2500ms, which was spent in full on every
    # unsuccessful round — 25.5s of one 66.1s recaptcha_grid_4x4 solve.
    #
    # To replace the headroom with a number: drive the real vendor pages via
    # `tests/live-solve/src/demo-targets.ts` and re-read this distribution.
    post_solve_outcome_timeout_ms: int = 1_000
    #: Poll interval inside that window.
    #:
    #: The success signals are polled, so detection latency is a MULTIPLE of
    #: this — and the one that matters most, "the widget is gone", needs two
    #: consecutive polls to confirm, so it cannot be seen in less than 2x. At
    #: 150ms that put a 300ms floor under every inline vendor's success. 75ms
    #: halves it, and the poll is cheap: `detect_captcha` measures ~3ms.
    post_solve_outcome_poll_ms: int = 75
    element_screenshot_timeout_ms: int = 8_000
    max_unsupported_resolves: int = 3
    max_stale_element_retries: int = 3
    stale_element_backoff_ms: int = 900

    # Freshness guard — re-solve if the frame moved during inference.
    stale_frame_resolve_enabled: bool = True
    stale_frame_diff_threshold: float = 0.02
    max_stale_frame_resolves: int = 2

    # Pixel-settle monitor.
    settle_poll_ms: int = 220
    settle_frames: int = 2
    settle_timeout_ms: int = 9_000
    animated_challenge_after_ms: int = 4_500
    settle_diff_threshold: float = 0.01
    post_submit_change_timeout_ms: int = 4_000

    # ── Animated challenges ────────────────────────────────────────────────
    # A challenge that never settles is RECORDED and solved from keyframes rather
    # than abandoned. Off switch for callers who would rather fail fast than spend
    # the recording time.
    video_solve_enabled: bool = True
    # Whether a repeated answer may escalate to a RECORDING of the widget.
    #
    # A board that cycles reads as static to `_wait_for_element_settled` (it
    # rests between screens for far longer than the 440ms that rule stops at),
    # gets answered from whichever screen the screenshot caught, and then
    # answers identically every round. That repeat is the signal, and the
    # response is to record a burst and let the SLICER decide: a widget that
    # really is static slices to one keyframe and is solved as the still it is,
    # so the escalation costs one burst and can never produce a worse answer.
    #
    # Off restores the old behaviour for a caller who would rather re-read a
    # still than spend `video_burst_duration_ms` finding out.
    animated_probe_enabled: bool = True
    # Burst geometry. Deliberately identical to the collector's
    # (`_collect_common.BURST_DURATION_MS` / `BURST_FPS` in the finetune repo), so a
    # challenge recorded here is the same shape of artifact the model trained on —
    # same clip length, same frame rate, therefore the same keyframe slicing.
    video_burst_duration_ms: int = 4_000
    #: Hard ceiling on a burst that never repeats a screen. The burst normally
    #: ends when the board's cycle closes; this bounds a continuous animation,
    #: which never closes one. Must match `videoBurstMaxMs` in the JS port.
    video_burst_max_ms: int = 12_000
    #: Record while the still screenshot is being read, so a cycling board is
    #: known before its answer is acted on. Must match `speculativeBurstEnabled`
    #: in the JS port.
    speculative_burst_enabled: bool = True
    video_burst_fps: int = 10
    # How long to wait for the widget to return to the keyframe the model chose,
    # before clicking anyway. Bounded because the alternative is worse: these
    # puzzles cycle, so the state DOES come back — but if the recording caught a
    # one-off transition it never will, and a click on the model's coordinates is
    # still a better use of the remaining budget than a timeout.
    # 9000, not 6000. A GeeTest svg board dwells p50 1.5s / p75 2.0s / max 2.7s
    # per screen, so a 3-screen cycle runs 4.5s median and 8.1s worst case — a
    # 6s budget could not catch the worst case however well aimed. Must match
    # SOLVE_DEFAULTS.keyframeWaitTimeoutMs in the JS port: `video_budget_ms` is
    # derived from it on both sides and the two must not disagree.
    keyframe_wait_timeout_ms: int = 9_000
    keyframe_wait_poll_ms: int = 120
    #: Extra wall clock granted ONCE, the first time a solve escalates to a
    #: recording. NOT a looser `overall_solve_timeout_ms`.
    #:
    #: That budget is sized for ROUNDS — "a round costs ~4-7s, so six is the
    #: budget". The recording path is not a round. It is a fixed extra stage
    #: costing the burst, the slice, one MULTI-IMAGE inference (six keyframes,
    #: several times a still's) and the wait for the widget to come back round
    #: to the frame the model chose. Nothing in the 45s was ever set aside for
    #: it, so an escalation started at ~35s ran the clock out mid-burst and
    #: reported a TIMEOUT — which reads as a slow model rather than as a budget
    #: with no room for the thing the solver had just decided to do.
    #:
    #: Measured 2026-08-22, Tier 3 run 32596340560: EVERY python-port failure on
    #: hcaptcha_fish_swim_different, hcaptcha_number_with_highest_value_video and
    #: hcaptcha_tile_flip_video was "exceeded overall_solve_timeout_ms during
    #: recording the animated challenge", at 45.7-52.7s elapsed. The same three
    #: fixtures solve in 11-20s on the rounds where the still path happens to
    #: answer them, so it is the escalation that does not fit, not the puzzle.
    #:
    #: Granted only when `video_solve_enabled` — a caller who wants a hard
    #: deadline turns recording off, which is the switch that already means
    #: "fail fast rather than spend the recording time".
    video_extra_inference_ms: int = 8_000

    def video_budget_ms(self) -> int:
        """What one escalation to recording is allowed to cost, on top.

        DERIVED, so a longer burst or a longer keyframe wait carries its own
        budget instead of quietly reintroducing the timeout this exists to fix.
        """
        return (self.video_burst_max_ms + self.keyframe_wait_timeout_ms
                + self.video_extra_inference_ms)

    # Grid load / dynamic-refresh timing.
    # How long to wait for hCaptcha's task images to paint before screenshotting
    # anyway. Best-effort by design — the screenshot happens either way — so it
    # is bounded by what a loading tile plausibly costs, not by a generous
    # default. 8s was two-thirds of what a whole solve is now allowed to take.
    hcaptcha_images_timeout_ms: int = 3_000

    grid_load_poll_interval_ms: int = 250
    grid_load_timeout_ms: int = 8_000
    recaptcha_max_dynamic_rounds: int = 8
    recaptcha_fade_onset_grace_ms: int = 4_000
    recaptcha_dynamic_fade_poll_ms: int = 250
    recaptcha_dynamic_fade_wait_ms: int = 6_000
    recaptcha_tile_hover_enabled: bool = True

    # ── puzzle-piece sliders (see _execute_slide) ──────────────────────────
    # How far to nudge the handle, in px, to learn the piece's width and how
    # fast it follows. Two probes because two unknowns; far enough apart that
    # the difference between the two widths is signal rather than rounding, and
    # both small enough to stay on the shortest track observed (~250px).
    slide_probe_offsets_px: Tuple[float, ...] = (24.0, 64.0)
    # Stop steering once the piece is this close. Tighter than any vendor's
    # accept window, so the limit on solving is the model's slot estimate.
    slide_tolerance_px: float = 2.0
    # Corrections after the probes. Each costs a screenshot with the mouse held
    # down; two is enough for a linear system, and the third would only be
    # chasing a measurement that is not going to converge.
    slide_max_corrections: int = 2


# Vendors with no checkbox/challenge split — one container, one interactive
# surface. Checked in detect_captcha() after the five hard-coded reCAPTCHA /
# hCaptcha / Turnstile checks above, so those keep first refusal. Selectors
# lifted from src/captchaCollection/sources.py, which already drives these 8
# vendors nightly in the collector. Mirror of VENDOR_WIDGET_LOCATORS in
# solver.ts — keep both in the same order with the same selectors.
VENDOR_WIDGET_LOCATORS = [
    {"puzzle_source": "geetest", "selectors": [".geetest_box", ".geetest_panel_box", ".geetest_popup_window", ".geetest_widget"]},
    # Tencent renders IN THE HOST DOCUMENT since 2026-08-11; before that it was
    # `iframe#tcaptcha_iframe_dy`. Both shapes are listed, in-page first,
    # because the iframe build is still deployed on sites pinning an older
    # widget and costs nothing to keep. Dropping the in-page selectors is what
    # made the client blind to every live Tencent captcha for twelve days.
    #
    # `iframe[id^="tcaptcha"]`, NOT `[id*=...]`. The substring form also matched
    # MTCaptcha's `mtcaptcha-iframe-1` — "mtcaptcha" contains "tcaptcha" — so
    # Tencent's entry was silently the only thing detecting MTCaptcha at all,
    # which is why `.mtcap` matching nothing went unnoticed. Anchored, it still
    # matches the real `tcaptcha_iframe_dy` and claims nobody else's widget.
    {"puzzle_source": "tencent", "selectors": ['#tcaptcha_transform_dy', '#tCaptchaDyContent', '.tencent-captcha-dy__content', 'iframe#tcaptcha_iframe_dy', 'iframe[id^="tcaptcha"]', 'iframe[src*="captcha.gtimg.com"]', 'iframe[src*="captcha.qq.com"]']},
    {"puzzle_source": "yidun", "selectors": [".yidun_panel", ".yidun"]},
    # Yandex SmartCaptcha is IFRAMED, and `.CheckboxCaptcha` is a class inside
    # that frame's document — `query_selector` does not cross a frame boundary,
    # so the host page never carried it and this entry matched nothing, ever.
    # Challenge frame first, anchor second, exactly like reCAPTCHA above.
    # `.CheckboxCaptcha` stays for an inline embed that renders it directly.
    {"puzzle_source": "yandex", "selectors": ['iframe[src*="smartcaptcha.yandexcloud.net/advanced"]', 'iframe[src*="smartcaptcha.yandexcloud.net"]', ".CheckboxCaptcha"]},
    {"puzzle_source": "lemin", "selectors": ["#lemin-cropped-captcha", ".lemin-captcha-popup"]},
    {"puzzle_source": "prosopo", "selectors": [".prosopo-modalInner", ".procaptcha-checkbox"]},
    # MTCaptcha is IFRAMED too: `<div class="mtcaptcha"><iframe
    # id="mtcaptcha-iframe-1" src="https://service.mtcaptcha.com/...">`. `.mtcap`
    # is the class prefix used INSIDE that frame (`mtcap-inputtext`), and a CSS
    # class selector matches whole tokens, so it never matched the host page's
    # `class="mtcaptcha"` either. Kept last for an inline embed.
    {"puzzle_source": "mtcaptcha", "selectors": ['iframe[src*="service.mtcaptcha.com"]', 'iframe[id^="mtcaptcha-iframe"]', ".mtcaptcha", ".mtcap"]},
    {"puzzle_source": "botdetect", "selectors": [".BDC_CaptchaDiv"]},
]


# ── which vendor's code the page LOADED, as opposed to what it rendered ─────
#
# A selector is a claim about markup, and markup is the half of a captcha the
# vendor is free to rewrite overnight — it is an anti-bot surface, so churn is
# the point. The REQUEST is far more stable: Tencent moved its widget out of an
# iframe and renamed every class on 2026-08-11, and went on fetching from
# `turing.captcha.gtimg.com` and `turing.captcha.qcloud.com` throughout. The
# substring `captcha.gtimg.com` — already in the table above, on a selector that
# stopped matching — never left the wire.
#
# This is a TRIPWIRE, not a detection path. A URL says which vendor is on the
# page; it cannot say where the widget is, and the solver needs an ELEMENT to
# screenshot, to scope its controls against, and to drag. What it buys is the
# difference between the two things a null detection can mean:
#
#   · no captcha here (reCAPTCHA v3 / invisible / not triggered yet) — fine, and
#     what the message used to guess unconditionally;
#   · the vendor's code is loaded and running and NOTHING WE LOOK FOR MATCHES —
#     i.e. the DOM moved under us. That is a bug report, and it took twelve days
#     to notice the last time because the client said the first thing while
#     meaning the second.
#
# Hosts measured 2026-08-24 by loading each vendor's demo page and recording
# every third-party request (see scripts/check_vendor_selectors.py --hosts).
#: The two vendors with bespoke handling in this file. EVERYTHING else shares the
#: generic path, and the two behaviours below are gated on "not one of these" —
#: spelled as this set rather than as `== "unknown"`, which is what they used to
#: say. The two were identical only while `unknown` was the sole third value, so
#: the day anyone reports a vendor by name — to constrain which grid shapes it may
#: be solved as, say — `== "unknown"` silently turns OFF typed-challenge detection
#: for MTCaptcha, Yandex and BotDetect (all three ARE typed captchas) and the
#: animated settle probe for GeeTest and Tencent. Neither failure raises; the text
#: captcha just becomes unsolvable and the animated board gets answered from one
#: arbitrary frame.
VENDORS_WITH_BESPOKE_HANDLING = frozenset({"hcaptcha", "recaptcha"})

VENDOR_URL_MARKERS = [
    {"puzzle_source": "hcaptcha", "hosts": ["hcaptcha.com"]},
    {"puzzle_source": "recaptcha", "hosts": ["google.com/recaptcha", "recaptcha.net"]},
    {"puzzle_source": "turnstile", "hosts": ["challenges.cloudflare.com"]},
    {"puzzle_source": "geetest", "hosts": ["geetest.com"]},
    {"puzzle_source": "tencent", "hosts": ["captcha.gtimg.com", "captcha.qcloud.com"]},
    {"puzzle_source": "yidun", "hosts": ["dun.163.com", "cstaticdun.126.net",
                                         "necaptcha.nosdn.127.net"]},
    {"puzzle_source": "yandex", "hosts": ["smartcaptcha.yandexcloud.net"]},
    {"puzzle_source": "lemin", "hosts": ["leminnow.com"]},
    {"puzzle_source": "prosopo", "hosts": ["prosopo.io"]},
    {"puzzle_source": "mtcaptcha", "hosts": ["mtcaptcha.com"]},
    # BotDetect is deliberately absent and must stay absent. It is a SELF-HOSTED
    # library — the image comes from the application's own origin
    # (`BotDetectCaptcha.ashx`), so there is no vendor host to see and a URL
    # tripwire cannot cover it. Recording that is the point: an empty result for
    # a BotDetect page means "this check does not apply", not "no captcha".
]


# ── where the answer goes, when it is not a click ───────────────────────────
#
# Both tables are ordered VENDOR-FIRST, GENERIC-LAST, and the driver takes the
# first visible match. That order is the whole design: a named vendor selector
# is unambiguous, while the generic patterns are guesses that happen to be right
# most of the time. Trying the guess first would, on a page that hosts a captcha
# *and* a login form, type the captcha's answer into the username box.
#
# The generic tail is not a nicety either — it is what actually fires on most
# pages. Vendors rename these classes without notice (they are anti-bot
# surfaces, so churn is the point), and our own Tier 3 fixtures render neither
# vendor's DOM. Anything that only worked via the vendor list would be a feature
# that passes review and fails in the field.

# A distorted-text captcha's answer box. The three vendor entries are the three
# types in instructions.py::TEXT_TYPES.
#
# Split in two because the two halves are safe in different places. The NAMED
# half identifies a captcha's answer box wherever it sits; the GENERIC half only
# means "the text box in this widget" and is trustworthy only when the scope
# already is the widget. `_answer_box` widens past the widget for BotDetect and
# may take the named half with it, never the generic one.
TEXT_INPUT_VENDOR_SELECTORS = [
    # BotDetect — the input is application-defined, so match the id fragment its
    # own docs and samples use. These three are what the nightly collector
    # already drives (src/captchaCollection/sources.py).
    "input[id*=captchaCode]",
    "input#captchaCode",
    "input[id*=validateCaptcha]",
    ".BDC_CaptchaDiv input[type=text]",
    # MTCaptcha
    "input.mtcap-inputtext",
    ".mtcap input[type=text]",
    # Yandex SmartCaptcha
    ".AdvancedCaptcha-Input input",
    "input.Textinput-Control",
    'input[name="rep"]',
]

TEXT_INPUT_GENERIC_SELECTORS = [
    # Generic — an input the page itself labels as the captcha answer.
    'input[name*="captcha" i]',
    'input[id*="captcha" i]',
    'input[aria-label*="captcha" i]',
    'input[placeholder*="code" i]',
    'input[autocomplete="off"][type=text]',
    # Last resort: the only text box in the widget. Scoped to the challenge
    # frame/container by the caller, never to the whole page — see _find_control.
    "input[type=text]",
    "input:not([type])",
    "input[type=tel]",
    "textarea",
]

TEXT_INPUT_SELECTORS = TEXT_INPUT_VENDOR_SELECTORS + TEXT_INPUT_GENERIC_SELECTORS

# The handle you drag on a puzzle-piece slider. NOT the piece: on every one of
# these vendors the piece is inert decoration that the handle carries, so a
# drag starting on the piece moves nothing at all.
SLIDER_HANDLE_SELECTORS = [
    # GeeTest v3 / v4
    ".geetest_slider_button",
    ".geetest_btn",
    ".geetest_slider .geetest_arrow",
    # Tencent. The post-2026-08-11 knob is a bare div — no id, no role=slider,
    # no aria-valuenow — so every generic fallback below misses it and it has to
    # be named. The three after it are the pre-redesign widget.
    ".tencent-captcha-dy__slider-block",
    "#tcaptcha_drag_thumb",
    ".tc-slider-normal",
    "[id*=slideBlock]",
    # Yidun (NetEase)
    ".yidun_slider",
    ".yidun_jigsaw",
    # Lemin
    ".lemin-slider-handle",
    "#lemin-cropped-captcha .slider",
    # Generic — an ARIA slider, or a class that says handle/thumb/button on a
    # track. `[draggable=true]` is deliberately absent: it is the HTML5
    # drag-and-drop opt-in, which fires dragstart rather than pointermove, and
    # no slider captcha uses it.
    '[role="slider"]',
    "[aria-valuenow]",
    '[class*="slider"][class*="btn"]',
    '[class*="slider"][class*="button"]',
    '[class*="slide"][class*="handle"]',
    '[class*="drag"][class*="thumb"]',
]

# Fallback for the sliderless members of the family. Lemin's "cropped" puzzle
# has no track at all — you drag the piece itself onto the gap — and the model
# answers it with the same sourceless drag, because from the picture the two are
# indistinguishable. Tried only after SLIDER_HANDLE_SELECTORS finds nothing.
DRAGGABLE_PIECE_SELECTORS = [
    ".lemin-cropped-puzzle-piece",
    "#lemin-cropped-captcha canvas + canvas",
    '[class*="puzzle"][class*="piece"]',
    '[class*="jigsaw"]',
]


class PageSolver:
    """
    Drives one Playwright-compatible `page` through a captcha.

        from playwright.sync_api import sync_playwright
        from captchakraken.page_solver import PageSolver

        solver = PageSolver()
        result = solver.solve(page)
        if result.is_solved: ...

    One instance is reusable across solves; state is reset per `solve()`.
    """

    def __init__(
        self,
        config: Optional[PageSolverConfig] = None,
        solver: Optional[CaptchaSolver] = None,
        **solver_kwargs: Any,
    ) -> None:
        self.config = config or PageSolverConfig()
        # One CaptchaSolver for the whole driver: it owns the planner, which
        # accumulates token usage and holds the HTTP session to vLLM.
        #
        # `expert` reaches the solver from the config, so the knob is in the
        # same object as every other tunable and matches the TS port's
        # `CaptchaKrakenConfig.expert`. An explicit kwarg still wins — a caller
        # who built the argument list themselves has said the more specific
        # thing — and a caller who passed their OWN solver has already made
        # this decision inside it.
        if self.config.expert is not None:
            solver_kwargs.setdefault("expert", self.config.expert)
        self._solver = solver or CaptchaSolver(**solver_kwargs)
        #: Every gesture goes through here, and it owns the pointer position.
        #: See humanize.py — the driver names gestures, this decides what
        #: events they are and how long they take.
        self._human: Humanizer = resolve_humanizer(self.config)
        self._last_submit_frame_hash: Optional[str] = None
        # Absolute deadline for the current solve, in the same clock as
        # time.monotonic() * 1000. None outside a solve.
        self._deadline_ms: Optional[float] = None
        # Animation detection, all reset per solve by `_reset_animated_state`.
        # `_known_animated` latches: once a widget has been SEEN to animate,
        # every later round takes the keyframe path without re-probing.
        self._known_animated = False
        self._animated_probe_armed = False
        self._animated_probe_done = False
        # One-shot: the recording path buys its own budget the first time it is
        # entered, and never again in the same solve. See video_budget_ms.
        self._video_budget_granted = False
        # THE ONE recording and THE ONE answer for the animated board on screen.
        # A cycling board needs a single inference: the burst shows the model
        # every screen, the answer names the cell and the screen it is on, and
        # the gate holds the click until that screen is back. None of that
        # changes between rounds, so asking again buys nothing and costs a burst
        # plus a multi-image inference. Mirrors `animatedPlan` in the JS port.
        self._animated_plan: Optional[Tuple[List[str], str, List[CaptchaAction],
                                            List[Dict[str, Any]]]] = None
        self._keyframe_mode: Optional[str] = None
        # How many distinct steady screens the current clip sits on; 0 if it is
        # not that kind of clip. THE GATE KEYS ON THIS, NOT ON `_keyframe_mode`
        # — see the JS twin's `keyframeSteadyScreens` and `steady_screens()` in
        # keyframes.py. `even` means the slicer could not PROVE recurrence, not
        # that the board has nothing to come back to.
        self._keyframe_steady_screens: int = 0
        # Repeat detection; see `max_no_progress_rounds`.
        self._last_answer_sig: Optional[str] = None
        self._no_progress_rounds = 0
        # Per-solve phase accounting; see `_phase`.
        self._budget: Optional[PhaseBudget] = None

    @property
    def _last_mouse(self) -> Tuple[float, float]:
        """Where the pointer is. Owned by the humanizer, because a mode that
        dispatches no motion (mobile, between taps) still has to answer this."""
        return self._human.at

    @_last_mouse.setter
    def _last_mouse(self, at: Tuple[float, float]) -> None:
        self._human.at = (float(at[0]), float(at[1]))

    @contextmanager
    def _phase(self, name: str):
        """Attribute the enclosed wall-clock to `name` for this solve's budget.

        A no-op outside a solve, so helpers stay callable from tests and from
        the CLI without a budget having been opened.
        """
        if self._budget is None:
            yield
            return
        with self._budget.phase(name):
            yield

    @staticmethod
    def _answer_signature(actions: Sequence[Any], retry_mode: Optional[str]) -> Optional[str]:
        """A stable identity for "the answer this round is about to execute".

        Keyed on the retry mode too: the missed-tiles retry deliberately re-asks
        about the same board and its answer legitimately overlaps the previous
        one, so counting that as a repeat would abandon the one path built to
        recover from an under-selection.

        Returns None when the actions cannot be summarised, which is treated as
        "not a repeat" — an unreadable answer must never be the reason a solve
        is abandoned.
        """
        try:
            parts = []
            for raw in actions:
                a = _as_dict(raw)
                parts.append((
                    a.get("action"),
                    _round_pts(a.get("target_bounding_boxes")),
                    _round_pts(a.get("target_bounding_box")),
                    _round_pts(a.get("target_coordinates")),
                    _round_pts(a.get("source_bounding_box")),
                    a.get("text"),
                ))
            return repr((retry_mode, parts))
        except Exception:  # noqa: BLE001 — a signature is an optimisation, not a contract
            return None

    def _note_answer(self, actions: Sequence[Any], retry_mode: Optional[str]) -> None:
        """Count consecutive identical answers; see `max_no_progress_rounds`."""
        sig = self._answer_signature(actions, retry_mode)
        if sig is not None and sig == self._last_answer_sig:
            self._no_progress_rounds += 1
            _log(f"[no-progress] the model returned the same answer again "
                 f"({self._no_progress_rounds}/{self.config.max_no_progress_rounds}) — "
                 f"the previous one already ran and changed nothing")
            # A board that reads the same every round is the signature of a
            # CYCLING challenge answered as a still. Give the recording path a
            # chance before giving up on the solve entirely.
            self._arm_animated_probe()
        else:
            self._no_progress_rounds = 0
            self._last_answer_sig = sig

    def _reset_animated_state(self) -> None:
        self._known_animated = False
        self._animated_probe_armed = False
        self._animated_probe_done = False
        self._video_budget_granted = False
        self._discard_animated_plan()
        self._keyframe_mode = None
        self._keyframe_steady_screens = 0
        self._last_answer_sig: Optional[str] = None
        self._no_progress_rounds = 0

    def _check_deadline(self, where: str) -> None:
        """
        Enforce `overall_solve_timeout_ms` from INSIDE the long-running loops.

        The TypeScript driver checks its budget only at the top of each attempt,
        which means the budget is not really a budget: one slow attempt overruns
        it without bound, because nothing looks at the clock again until the
        attempt returns. Observed in practice — a camoufox session ran past ten
        minutes against a nominal 120 s timeout, and the check at the top of the
        loop never got a turn to fire.

        Called at the points that can legitimately spin for a long time: each
        action executed, and each round of the dynamic grid driver.
        """
        if self._deadline_ms is None:
            return
        if time.monotonic() * 1000.0 > self._deadline_ms:
            raise CaptchaSolveError(
                f"captcha solve exceeded overall_solve_timeout_ms "
                f"({self.config.overall_solve_timeout_ms}ms) during {where}"
            )

    # ------------------------------------------------------------------
    # Shared-half bridges. Each of these is the in-process equivalent of one
    # CLI subcommand the TS driver shells out for.
    # ------------------------------------------------------------------

    def _find_grid(self, image_path: str) -> Optional[List[Sequence[int]]]:
        """`captchakraken find-grid` — row-major cell boxes in screenshot pixels."""
        from .tool_calls.find_grid import find_grid

        try:
            return find_grid(image_path)
        except Exception as exc:  # pragma: no cover - CV failure is never fatal
            _debug(f"find_grid failed: {exc}")
            return None

    def _has_movement(self, path_a: str, path_b: str, threshold: float) -> bool:
        """`captchakraken check-movement`."""
        from .image_processor import ImageProcessor

        try:
            return bool(ImageProcessor.detect_movement(path_a, path_b, threshold))
        except Exception as exc:
            _debug(f"detect_movement failed: {exc}")
            return False

    def _grid_cell_states(
        self, path_a: str, path_b: str, grid_boxes: Sequence[Sequence[int]]
    ) -> Optional[Dict[str, List[int]]]:
        """
        `captchakraken grid-cell-states-fixed`.

        Always the FIXED variant: the dynamic refresh blanks tiles to near-white,
        which makes find_grid fail on that frame, and a self-detecting call would
        then report "no grid" — which a naive caller misreads as "nothing
        loading", i.e. solved. Passing the cached boxes keeps empty/changing/
        selected correct while tiles are blank.
        """
        from .cli import _compute_grid_cell_states

        try:
            return _compute_grid_cell_states(path_a, path_b, list(grid_boxes))
        except Exception as exc:
            _debug(f"grid_cell_states failed: {exc}")
            return None

    def _get_solution(
        self, image_path: str, puzzle_source: str, retry_mode: Optional[str],
        text_mode: bool = False,
    ) -> Tuple[List[CaptchaAction], List[Dict[str, Any]]]:
        """
        The model query. In-process where TS spawns the CLI.

        Token usage is read off the planner by DELTA rather than reset, because
        the planner is shared across the whole solve and the caller wants a
        per-round figure that still sums to the session total.
        """
        planner = self._solver.planner
        before = len(planner.token_usage)
        actions = self._solver.solve(
            image_path, puzzle_source=puzzle_source, retry_mode=retry_mode,
            text_mode=text_mode,
        )
        usage = [dict(u) for u in planner.token_usage[before:]]
        if not isinstance(actions, list):
            actions = [actions]
        return actions, usage

    def _get_keyframe_solution(
        self, keyframe_paths: Sequence[str]
    ) -> Tuple[List[CaptchaAction], List[Dict[str, Any]]]:
        """The model query for an animated challenge. Usage read by DELTA, as above.

        One request for the whole keyframe set, not one per frame: the model has to
        compare the frames to find what differs between them, which it can only do
        with all of them in a single context. Per-frame queries would also cost N
        billable rounds for one puzzle.
        """
        planner = self._solver.planner
        before = len(planner.token_usage)
        actions = self._solver.solve_keyframes(keyframe_paths)
        usage = [dict(u) for u in planner.token_usage[before:]]
        if not isinstance(actions, list):
            actions = [actions]
        return actions, usage

    # ------------------------------------------------------------------
    # Mouse
    # ------------------------------------------------------------------

    def _smooth_move(self, page: Any, x: float, y: float) -> None:
        """Travel to a point. What that MEANS is the humanizer's business — on
        a touchscreen with no finger down it is a bookkeeping update and emits
        nothing at all."""
        with self._phase("mouse"):
            self._human.move(page, (x, y))

    def _move_to_element(self, page: Any, element: Any, padding_percentage: float = 25.0) -> None:
        # BOUNDED. Playwright's default timeout is 30s, and this is called once
        # per action plus once per submit — on a challenge iframe that is
        # mid-animation, scrolling "waits for stability" and burns the full 30s
        # every time, which turned a ~5s solve loop into minutes during live
        # testing. The element is already on screen in every real case here (we
        # just screenshotted it), so a short bound loses nothing: on timeout we
        # move to wherever it currently is.
        try:
            element.scroll_into_view_if_needed(timeout=2_000)
        except TypeError:
            # An adapter whose signature takes no timeout (e.g. the Puppeteer
            # bridge). Fall back rather than fail the solve.
            try:
                element.scroll_into_view_if_needed()
            except Exception:
                pass
        except Exception:
            pass
        box = element.bounding_box()
        if not box:
            raise CaptchaSolveError("element has no bounding box")
        pad = padding_percentage / 100.0
        pad_x, pad_y = box["width"] * pad, box["height"] * pad
        target_x = box["x"] + pad_x + random.random() * (box["width"] - 2 * pad_x)
        target_y = box["y"] + pad_y + random.random() * (box["height"] - 2 * pad_y)
        self._smooth_move(page, target_x, target_y)

    def _move_and_click(self, page: Any, element: Any) -> None:
        self._move_to_element(page, element)
        with self._phase("mouse"):
            self._human.click(page, self._last_mouse)

    def _click_point_for(
        self, action: Dict[str, Any], element_box: Dict[str, float]
    ) -> Optional[Tuple[float, float]]:
        """Where in the element this click lands, in element-relative pixels.

        Split out of `_execute_click` because on an animated board the point has
        to be chosen ONCE and used three times — to park the pointer, to ask
        about the right neighbourhood, and to press. Choosing it inside the
        click (it is randomised within the target) meant the gate watched the
        bbox centre while the press landed somewhere else.
        """
        bbox = action.get("target_bounding_box")
        coords = action.get("target_coordinates")
        if bbox:
            min_x, min_y, max_x, max_y = (float(v) for v in bbox)
            px_min_x = min_x * element_box["width"]
            px_max_x = max_x * element_box["width"]
            px_min_y = min_y * element_box["height"]
            px_max_y = max_y * element_box["height"]
            pad_x = (px_max_x - px_min_x) * 0.1
            pad_y = (px_max_y - px_min_y) * 0.1
            return (
                (px_min_x + pad_x) + random.random() * ((px_max_x - pad_x) - (px_min_x + pad_x)),
                (px_min_y + pad_y) + random.random() * ((px_max_y - pad_y) - (px_min_y + pad_y)),
            )
        if coords:
            return (float(coords[0]) * element_box["width"],
                    float(coords[1]) * element_box["height"])
        return None

    def _click_when_frame_matches(
        self, page: Any, element: Any, action: Dict[str, Any],
        element_box: Dict[str, float], await_keyframe: str,
    ) -> None:
        """Click one target on an ANIMATED board, at the moment its screen is up.

        The ORDER is the point of it: park on the target, open the gate, press
        in place — so the only thing between "the right screen is showing" and
        the click is a mouse-down. It used to wait and then click, and the click
        begins with a humanised move of 274-647ms against a 1500ms median dwell,
        so the gate could succeed and the press still land on the next screen.
        Moving to a point the pointer already occupies is a no-op, so parking
        early costs nothing.
        """
        rel = self._click_point_for(action, element_box)
        if rel is None:
            _log("click action without coordinates or bounding box; skipping")
            return
        at = (element_box["x"] + rel[0], element_box["y"] + rel[1])
        with self._phase("mouse"):
            self._human.move(page, at)
        self._wait_for_keyframe(
            element, await_keyframe,
            (rel[0] / element_box["width"], rel[1] / element_box["height"]),
        )
        with self._phase("mouse"):
            self._human.click(page, at)

    def _execute_click(
        self, page: Any, action: Dict[str, Any], element_box: Dict[str, float]
    ) -> None:
        bbox = action.get("target_bounding_box")
        coords = action.get("target_coordinates")
        if bbox:
            min_x, min_y, max_x, max_y = (float(v) for v in bbox)
            px_min_x = min_x * element_box["width"]
            px_max_x = max_x * element_box["width"]
            px_min_y = min_y * element_box["height"]
            px_max_y = max_y * element_box["height"]
            # 10% inset, so a click never lands on a tile's border.
            pad_x = (px_max_x - px_min_x) * 0.1
            pad_y = (px_max_y - px_min_y) * 0.1
            rel_x = (px_min_x + pad_x) + random.random() * ((px_max_x - pad_x) - (px_min_x + pad_x))
            rel_y = (px_min_y + pad_y) + random.random() * ((px_max_y - pad_y) - (px_min_y + pad_y))
        elif coords:
            rel_x = float(coords[0]) * element_box["width"]
            rel_y = float(coords[1]) * element_box["height"]
        else:
            _log("click action without coordinates or bounding box; skipping")
            return

        with self._phase("mouse"):
            self._human.click(page, (element_box["x"] + rel_x, element_box["y"] + rel_y))

    def _execute_drag(
        self, page: Any, action: Dict[str, Any], element_box: Dict[str, float]
    ) -> None:
        def center(bbox: Sequence[float]) -> Tuple[float, float]:
            return (
                element_box["x"] + ((float(bbox[0]) + float(bbox[2])) / 2) * element_box["width"],
                element_box["y"] + ((float(bbox[1]) + float(bbox[3])) / 2) * element_box["height"],
            )

        src = center(action["source_bounding_box"])
        dst = center(action["target_bounding_box"])
        with self._phase("mouse"):
            self._human.drag(page, src, dst)

    # ------------------------------------------------------------------
    # Typing and sliding
    # ------------------------------------------------------------------

    def _find_control(self, scope: Any, selectors: Sequence[str]) -> Optional[Any]:
        """First VISIBLE match for `selectors`, tried in order.

        `scope` is the challenge frame, or — for the vendors that render into
        the host page rather than an iframe — the widget element itself. Never
        the page: the generic tail of both selector tables would otherwise
        happily match a login form's text box or a carousel's drag handle
        somewhere else on the document, and the answer would go there.
        """
        for selector in selectors:
            try:
                element = scope.query_selector(selector)
            except Exception:
                continue  # a selector this adapter can't parse must not end the search
            if self._visible(element):
                return element
        return None

    def _answer_box(self, scope: Any, element: Any = None) -> Optional[Any]:
        """Where a distorted-text captcha's answer goes.

        Inside the widget first. If the widget holds no text box at all, widen
        ONCE to its enclosing <fieldset>/<form> and look again for a
        VENDOR-NAMED box only.

        BotDetect is why, and nothing else needs it. BotDetect is a self-hosted
        LIBRARY, so the host application owns the layout: measured on
        captcha.com 2026-08-24, `.BDC_CaptchaDiv` — the element `detect_captcha`
        returns — is 280x50 and holds the image alone, while `#captchaCode` sits
        in a sibling `<div class="validationDiv">` under the enclosing
        <fieldset>. Scoped to the widget the solver found no box, `text_mode`
        stayed False, and it read the code correctly and never typed it. A clean
        solve thrown away, and indistinguishable in any report from a model that
        cannot read warped text.

        The widened pass is restricted to TEXT_INPUT_VENDOR_SELECTORS on
        purpose. The generic tail (`input[type=text]`, `textarea`,
        `input:not([type])`) is what makes the in-widget lookup work for every
        other vendor, and OUTSIDE the widget it is exactly how a captcha's
        answer ends up in a login form's username box — the failure
        `_find_control`'s docstring already warns about. One level, named
        selectors, and never the page.
        """
        inside = self._find_control(scope, TEXT_INPUT_SELECTORS)
        if inside is not None or element is None:
            return inside
        # Nearest first: a <fieldset> inside a <form> is the tighter box, and
        # `query_selector` on a union would hand back whichever comes first in
        # document order — the outer one.
        for axis in ("ancestor::fieldset[1]", "ancestor::form[1]"):
            try:
                host = element.query_selector(f"xpath={axis}")
            except Exception:
                continue
            if host is None:
                continue
            found = self._find_control(host, TEXT_INPUT_VENDOR_SELECTORS)
            if found is not None:
                _log("widget holds no answer box; using the vendor-named one "
                     "in its enclosing form")
                return found
        return None

    def _execute_type(self, page: Any, scope: Any, action: Dict[str, Any],
                      element: Any = None) -> bool:
        """Put the model's reading of a distorted-text captcha into its box."""
        text = str(action.get("text") or "")
        if not text:
            return False
        field = self._answer_box(scope, element)
        if field is None:
            _log("type action, but no text box in the widget; skipping")
            return False

        # Tapping the box is what focuses it — and on a phone it is also what
        # raises the keyboard, so this is not decoration on either device.
        self._move_and_click(page, field)
        if not self._human.type_text(page, field, text):
            return False
        _log(f"typed {len(text)} character(s) into the captcha field")
        return True

    def _track_piece(
        self, element: Any, before: str, after: str, exclude: Sequence[float]
    ) -> Optional[Sequence[int]]:
        """`captchakraken track-piece` — box of what moved, handle masked out."""
        from .tool_calls.track_piece import changed_bbox

        try:
            self._screenshot(element, after, timeout_ms=self.config.element_screenshot_timeout_ms)
            return changed_bbox(before, after, exclude)
        except Exception as exc:
            _debug(f"track_piece failed: {exc}")
            return None

    @staticmethod
    def _shot_scale(shot: str, css_width: float) -> float:
        """Device pixels per CSS pixel, read off the shot we are about to
        measure. 1.0 when the image cannot be read — an unreadable shot is
        already the `_track_piece` returns-None path, and guessing a ratio
        would steer the handle by it."""
        dims = _read_png_dimensions(shot)
        if not dims or css_width <= 0:
            return 1.0
        return dims[0] / css_width

    def _execute_slide(
        self,
        page: Any,
        element: Any,
        scope: Any,
        action: Dict[str, Any],
        element_box: Dict[str, float],
    ) -> bool:
        """Drive a puzzle-piece slider until the PIECE reaches the model's slot.

        The model is asked for one thing here — the centre of the gap — because
        it is the only thing the picture can tell it. What it cannot know is how
        far the handle must travel to put the piece there: the handle is
        elsewhere on the widget, and the ratio between the two is a vendor
        implementation detail that several of them deliberately vary.

        So this is closed-loop, not a calculation. Press the handle, nudge it
        twice by known amounts, and watch the screen:

            union(before, after) spans the piece's ORIGINAL left edge to its
            CURRENT right edge, so its width is  piece_width + ratio × nudge.

        Two nudges, two widths, two unknowns — solve for both, then steer the
        remaining distance and re-measure. The mouse is not released until the
        piece is home, because on every one of these puzzles releasing IS the
        submit; there is no Verify button to reconsider at.

        Returns False if there is nothing here to drag, leaving the caller's
        normal no-op handling to deal with it.
        """
        target_x = (
            (float(action["target_bounding_box"][0]) + float(action["target_bounding_box"][2])) / 2
        ) * element_box["width"]

        handle = self._find_control(scope, SLIDER_HANDLE_SELECTORS)
        if handle is None:
            # No track — the sliderless members of the family (Lemin's
            # "cropped") want the piece dragged directly. Same answer from the
            # model, because the two look identical; different gesture. Nothing
            # to close a loop on, since the piece is under the cursor and moves
            # with it one for one.
            piece = self._find_control(scope, DRAGGABLE_PIECE_SELECTORS)
            if piece is None:
                _log("slide action, but the widget has neither a slider nor a draggable piece")
                return False
            box = piece.bounding_box()
            if not box:
                return False
            # BOTH axes. The rail members travel horizontally and nothing else,
            # so the handle's own y is the only y there is — but a free drag
            # carries the piece across the card, and holding the piece's row
            # here slid it along the TRAY and released it there, 250 px below
            # the slot, on every attempt.
            target_y = (
                (float(action["target_bounding_box"][1])
                 + float(action["target_bounding_box"][3])) / 2
            ) * element_box["height"]
            _log("no slider track; dragging the piece to the slot directly")
            with self._phase("mouse"):
                self._human.drag(
                    page,
                    (box["x"] + box["width"] / 2, box["y"] + box["height"] / 2),
                    (element_box["x"] + target_x, element_box["y"] + target_y),
                )
            return True

        hbox = handle.bounding_box()
        if not hbox:
            return False
        start_x = hbox["x"] + hbox["width"] / 2
        hold_y = hbox["y"] + hbox["height"] / 2

        # Mask the whole horizontal BAND the handle runs in, not just where it
        # is now: it is about to move across that band, and most vendors fill
        # the track behind it as it goes. Either would otherwise be the largest
        # moving thing in frame, and we would track the handle instead of the
        # piece.
        pad = max(4.0, hbox["height"] * 0.35)
        band = [
            0.0,
            hbox["y"] - element_box["y"] - pad,
            element_box["width"],
            hbox["y"] + hbox["height"] - element_box["y"] + pad,
        ]

        shots = [_tmp_png("slide") for _ in range(4)]
        try:
            self._move_to_element(page, handle, padding_percentage=30.0)
            self._human.press(page)
            self._human.pause("grab")
            self._screenshot(element, shots[0],
                             timeout_ms=self.config.element_screenshot_timeout_ms)

            # CSS pixels out here, DEVICE pixels inside the shots. The same
            # number on a 1x desktop and 2.625x apart on a phone, which is why
            # this loop measured a slider correctly for a year and then missed
            # every attempt the moment Tier 3 grew a mobile arm: the mask landed
            # above the handle, the widths came back 2.6x too wide, and
            # `_solve_slide_geometry` threw them out as wider than the widget.
            # Measured from the shot rather than asked of the page, for the same
            # reason the grid path does it (`_GridSession.scale_x`): what the CV
            # reads is the image, whatever the window thinks its ratio is.
            scale = self._shot_scale(shots[0], element_box["width"])
            exclude = [v * scale for v in band]

            probes = self.config.slide_probe_offsets_px
            widths: List[Tuple[float, float]] = []
            last_box = None
            for offset, shot in zip(probes, shots[1:]):
                self._smooth_move(page, start_x + offset, hold_y)
                self._human.pause("probe")
                box = self._track_piece(element, shots[0], shot, exclude)
                if box is not None:
                    widths.append((float(offset), float(box[2] - box[0]) / scale))
                    last_box = box

            piece_w, ratio = self._solve_slide_geometry(widths, element_box["width"])
            if last_box is None or piece_w is None:
                # Never saw the piece — a canvas the screenshot cannot separate,
                # a widget that redraws wholesale, or a press the handle refused.
                # Fall back on the geometry every one of these puzzles shares:
                # piece and handle both start flush left, so the handle's travel
                # is the piece's travel.
                _log("slider: piece never resolved on screen; steering by handle travel alone")
                self._smooth_move(page, start_x + (target_x - (start_x - element_box["x"])), hold_y)
            else:
                # The offset `last_box` was MEASURED at — not `probes[-1]`, and
                # not indexed by how many measurements succeeded. If the first
                # probe failed to resolve and the second worked, those two
                # disagree, and steering from a base the reading does not belong
                # to sends the piece somewhere neither the model nor the screen
                # asked for.
                offset = float(widths[-1][0])
                for _ in range(self.config.slide_max_corrections):
                    piece_center = (last_box[2] / scale - piece_w / 2.0)
                    error = target_x - piece_center
                    if abs(error) <= self.config.slide_tolerance_px:
                        break
                    offset += error / ratio
                    self._smooth_move(page, start_x + offset, hold_y)
                    self._human.pause("probe")
                    box = self._track_piece(element, shots[0], shots[3], exclude)
                    if box is None:
                        break  # ran out of track; release where we are
                    last_box = box
                _debug(f"slider: piece_w={piece_w:.1f} ratio={ratio:.3f} scale={scale:.3f} "
                       f"final_center={last_box[2] / scale - piece_w / 2.0:.1f} "
                       f"target={target_x:.1f}")

            # Settle before letting go. A release in the same tick as the last
            # move reads as a machine, and some vendors sample the final
            # milliseconds of the gesture.
            self._human.pause("settle")
        finally:
            try:
                self._human.release(page)
            except Exception:
                pass
            for shot in shots:
                _unlink(shot)
        return True

    @staticmethod
    def _solve_slide_geometry(
        widths: Sequence[Tuple[float, float]], widget_width: float
    ) -> Tuple[Optional[float], float]:
        """Piece width and handle-to-piece travel ratio, from probe measurements.

        Each measurement is (handle offset, width of what changed), and
        width = piece_width + ratio × offset. Two of them determine both.

        With only one usable measurement the system is underdetermined, so ratio
        is ASSUMED to be 1 — true of every vendor observed, and the assumption
        is stated here rather than buried as a default. A ratio solved from
        implausible measurements (a redraw, a piece that hit the wall between
        probes) is rejected the same way: better a 1:1 guess that overshoots and
        gets corrected than a ratio of 0.02 that sends the handle off the track.
        """
        if not widths:
            return None, 1.0
        piece_w: Optional[float] = None
        ratio = 1.0
        if len(widths) >= 2:
            (o1, w1), (o2, w2) = widths[0], widths[-1]
            if o2 != o1:
                candidate = (w2 - w1) / (o2 - o1)
                if 0.2 <= candidate <= 3.0:
                    ratio = candidate
                    piece_w = w1 - ratio * o1
        if piece_w is None:
            o, w = widths[-1]
            piece_w = w - ratio * o
        # A piece narrower than a few pixels, or wider than half the widget, is
        # a measurement of something else.
        if not 3.0 <= piece_w <= widget_width * 0.6:
            return None, ratio
        return piece_w, ratio

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    @staticmethod
    def _visible(element: Any) -> bool:
        try:
            return bool(element and element.is_visible())
        except Exception:
            return False

    def _has_non_empty_field_value(self, page: Any, selector: str) -> bool:
        try:
            element = page.query_selector(selector)
            if not element:
                return False
            value = element.get_attribute("value")
            if value is None:
                value = page.eval_on_selector(
                    selector, "node => (typeof node.value === 'string' ? node.value : '')"
                )
            return bool(value and str(value).strip())
        except Exception:
            return False

    def _is_recaptcha_anchor_checked(self, anchor_iframe: Any) -> bool:
        try:
            frame = anchor_iframe.content_frame()
            if not frame:
                return False
            checked = frame.query_selector(".recaptcha-checkbox-checked")
            return self._visible(checked)
        except Exception:
            return False

    def _is_hcaptcha_anchor_checked(self, anchor_iframe: Any) -> bool:
        # hCaptcha sets <div id="checkbox" aria-checked="true"> on success. Demo
        # pages don't always populate h-captcha-response, so this visual state is
        # the necessary tie-breaker rather than a nicety.
        try:
            frame = anchor_iframe.content_frame()
            if not frame:
                return False
            return self._visible(frame.query_selector('#checkbox[aria-checked="true"]'))
        except Exception:
            return False

    def has_interactive_widget_in_dom(self, page: Any) -> bool:
        """
        Broader than `detect_captcha`: is a widget in the DOM AT ALL, even if it
        has not rendered? Distinguishes "still loading, wait for it" from
        "reCAPTCHA v3 / invisible, fail fast" — two cases a null detection
        cannot tell apart.
        """
        try:
            for anchor in page.query_selector_all('iframe[src*="recaptcha/api2/anchor"]'):
                src = anchor.get_attribute("src") or ""
                # v3 / invisible-v2 injects only an anchor, with size=invisible,
                # and never a challenge frame. Excluded here so we fail fast.
                if not re.search(r"[?&]size=invisible", src):
                    return True
            if page.query_selector('iframe[src*="recaptcha/api2/bframe"]'):
                return True
            if page.query_selector('iframe[src*="hcaptcha"][src*="frame=checkbox"]'):
                return True
            if page.query_selector('iframe[src*="hcaptcha"][src*="frame=challenge"]'):
                return True
        except Exception:
            pass
        return False

    def detect_captcha(self, page: Any) -> Optional[Any]:
        """Open challenges first, then unsolved checkboxes. Mirror of `detectCaptcha`."""
        recaptcha_challenge = page.query_selector('iframe[src*="recaptcha/api2/bframe"]')
        if self._visible(recaptcha_challenge):
            return recaptcha_challenge

        # Match the URL fragment, not the title: hCaptcha's ANCHOR title also
        # says "hCaptcha security challenge", so a title match mis-classifies it.
        hcaptcha_challenge = page.query_selector('iframe[src*="hcaptcha"][src*="frame=challenge"]')
        if self._visible(hcaptcha_challenge):
            return hcaptcha_challenge

        recaptcha_checkbox = page.query_selector('iframe[src*="recaptcha/api2/anchor"]')
        if self._visible(recaptcha_checkbox) and not self._is_recaptcha_anchor_checked(
            recaptcha_checkbox
        ):
            return recaptcha_checkbox

        hcaptcha_checkbox = page.query_selector('iframe[src*="hcaptcha"][src*="frame=checkbox"]')
        if self._visible(hcaptcha_checkbox):
            has_token = self._has_non_empty_field_value(page, '[name="h-captcha-response"]')
            if not has_token and not self._is_hcaptcha_anchor_checked(hcaptcha_checkbox):
                return hcaptcha_checkbox

        turnstile_iframe = page.query_selector('iframe[src*="challenges.cloudflare.com"]')
        if self._visible(turnstile_iframe) and not self._has_non_empty_field_value(
            page, '[name="cf-turnstile-response"]'
        ):
            return turnstile_iframe

        # Closed shadow roots hide the iframe; the container is still findable.
        turnstile_container = page.query_selector(".cf-turnstile")
        if self._visible(turnstile_container) and not self._has_non_empty_field_value(
            page, '[name="cf-turnstile-response"]'
        ):
            return turnstile_container

        # Vendors with one interactive surface (no checkbox/challenge split) —
        # GeeTest, Tencent, Yidun, Yandex, Lemin, Prosopo, MTCaptcha, BotDetect.
        for entry in VENDOR_WIDGET_LOCATORS:
            for selector in entry["selectors"]:
                el = page.query_selector(selector)
                if self._visible(el):
                    return el

        return None

    def vendors_on_the_wire(self, page: Any) -> List[str]:
        """Which vendors' code this page actually loaded.

        Read out of the page's own resource timing plus the `src`/`href` of
        everything it linked, so it works whenever it is called — a request
        listener would have to have been attached before navigation, and by the
        time a solve fails that ship has sailed.

        Best-effort by construction: the resource-timing buffer is finite and a
        page may have cleared it. An empty answer is "nothing seen", never "no
        captcha here".
        """
        try:
            names = page.evaluate(
                """() => {
                  const out = [];
                  try {
                    for (const e of performance.getEntriesByType('resource')) out.push(e.name);
                  } catch (err) {}
                  for (const el of document.querySelectorAll('script[src],iframe[src],link[href],img[src]')) {
                    out.push(el.getAttribute('src') || el.getAttribute('href') || '');
                  }
                  return out;
                }"""
            )
        except Exception:
            return []
        blob = " ".join(str(n) for n in (names or []))
        return [entry["puzzle_source"] for entry in VENDOR_URL_MARKERS
                if any(host in blob for host in entry["hosts"])]

    def _no_widget_message(self, page: Any) -> str:
        """Why nothing was found — the two cases told apart.

        The old text guessed "reCAPTCHA v3 / invisible" every time, including
        the times a vendor had simply moved its DOM. See VENDOR_URL_MARKERS.
        """
        base = "no interactive captcha widget detected"
        loaded = self.vendors_on_the_wire(page)
        if not loaded:
            return (f"{base} (no vendor captcha code loaded on this page — "
                    "likely reCAPTCHA v3 / invisible, or a click-triggered "
                    "challenge that has not been triggered)")
        return (f"{base}, BUT {'/'.join(loaded)} code IS loaded and running on "
                "this page. The vendor's markup no longer matches anything in "
                "VENDOR_WIDGET_LOCATORS — re-measure with "
                "scripts/check_vendor_selectors.py and update BOTH solver ports")

    def is_captcha_solved(self, page: Any) -> bool:
        """
        The vendor's definitive DONE signal — anchor checked or token populated.

        Needed because after the final submit hCaptcha keeps the challenge iframe
        VISIBLE for a couple of seconds while it verifies. Treating that frame as
        a fresh puzzle burns ~18s re-running the pipeline on a closing frame.
        """
        try:
            # The token FIRST, and unconditionally. It is a hidden field on the
            # PAGE, not inside the widget, so it never needed the anchor iframe
            # to be on screen — and the moment it matters most is precisely when
            # the anchor is NOT on screen, because hCaptcha keeps its challenge
            # overlay up for a couple of seconds after the winning submit.
            # Gating this on `_visible(anchor)` meant the one signal that was
            # already true went unread, and the loop ground on against a frame
            # being torn down. See tests/test_solved_detection.py.
            if self._has_non_empty_field_value(page, '[name="h-captcha-response"]'):
                return True
            if self._has_non_empty_field_value(page, '[name="g-recaptcha-response"]'):
                return True
            # Turnstile. `detect_captcha` already reads this exact field to
            # decide a Turnstile widget is UNSOLVED, so the signal was known to
            # one half of the driver and ignored by the other: a solved
            # Turnstile reported unsolved here and the loop ground through the
            # readiness waits and a wasted inference until the widget happened
            # to disappear. Same defect the hCaptcha token had.
            if self._has_non_empty_field_value(page, '[name="cf-turnstile-response"]'):
                return True

            # Anchor state is the fallback, and this one DOES need the iframe:
            # it is read out of the anchor's own document.
            hc = page.query_selector('iframe[src*="hcaptcha"][src*="frame=checkbox"]')
            if self._visible(hc) and self._is_hcaptcha_anchor_checked(hc):
                return True
            rc = page.query_selector('iframe[src*="recaptcha/api2/anchor"]')
            if self._visible(rc) and self._is_recaptcha_anchor_checked(rc):
                return True
        except Exception:
            pass
        return False

    def _is_challenge_freshly_rendered(self, page: Any) -> bool:
        """
        A NEXT ROUND (prompt painted) as opposed to a frame animating closed
        (prompt already gone). Lets a multi-round solve move on immediately
        instead of waiting out the full post-submit window.
        """
        try:
            hc = page.query_selector('iframe[src*="hcaptcha"][src*="frame=challenge"]')
            if self._visible(hc):
                # "Prompt painted" alone does NOT mean a next round. hCaptcha
                # leaves the round you just answered on screen while it
                # verifies, so this fired on the CLOSING frame and broke the
                # post-submit poll out of its solved-check after ~0ms — which
                # then committed the solver to ~21s of readiness waits and a
                # full inference against a frame that was about to be destroyed.
                # The frame must have actually CHANGED since the submit; we
                # already snapshot exactly that at submit time.
                if self._last_submit_frame_hash:
                    current = self._element_frame_hash(hc)
                    if current and current == self._last_submit_frame_hash:
                        return False
                frame = hc.content_frame()
                prompt = frame.query_selector(".prompt-text") if frame else None
                if self._visible(prompt):
                    text = prompt.text_content() or ""
                    if text.strip():
                        return True
            rc = page.query_selector('iframe[src*="recaptcha/api2/bframe"]')
            if self._visible(rc):
                frame = rc.content_frame()
                instructions = (
                    frame.query_selector(".rc-imageselect-instructions, #rc-imageselect")
                    if frame
                    else None
                )
                if self._visible(instructions):
                    return True
        except Exception:
            pass
        return False

    #: Which of reCAPTCHA's bframe banners is showing, by selector.
    #:
    #: ONE ENTRY PER MEANING. They share a corner and a look and they are three
    #: different facts; see `_recaptcha_banner_kind`.
    _RECAPTCHA_BANNERS = (
        (".rc-imageselect-error-select-more", "select-more"),
        (".rc-imageselect-error-dynamic-more", "dynamic-more"),
        (".rc-imageselect-incorrect-response", "rejected"),
    )

    @staticmethod
    def _banner_is_fatal_after_retry(kind: Optional[str]) -> bool:
        """May a REPEAT of this banner end the solve?

        Only where repeating means stuck. `dynamic-more` repeating is what a
        dynamic board's normal progress looks like, and treating it as fatal is
        the bug this split was made to fix.
        """
        return kind in ("select-more", "rejected")

    def _recaptcha_banner_kind(self, page: Any) -> Optional[str]:
        """WHICH of reCAPTCHA's three bframe banners is showing, if any.

        `rejected`     "Please try again."  The answer was wrong; a fresh board
                       follows, so the useful response is to solve that one.
        `select-more`  "Please select all matching images."  Under-selected, and
                       the tiles do NOT refresh — so a driver that re-submits
                       the same answer loops until the session times out. The
                       missed-tiles retry and the abort exist for this case.
        `dynamic-more` "Please also check the new images."  NOT AN ERROR. It is
                       the dynamic 3x3's normal flow: cleared tiles fade out,
                       replacements fade in, and the widget says so — on
                       essentially every round of that variant.

        This used to be a boolean over all three, which made the third
        indistinguishable from the two genuine errors and armed the one-retry
        abort with the sentence that means "you are doing fine". Every dynamic
        board therefore died at round two. Measured 2026-09-06: three of three
        `recaptcha_3x3_fade` attempts stopped at exactly boards=2, while
        `recaptcha_4x4` on the same run passed at 2, 3 and 5 boards.
        """
        try:
            bframe = page.query_selector('iframe[src*="recaptcha/api2/bframe"]')
            if not bframe:
                return None
            frame = bframe.content_frame()
            if not frame:
                return None
            for selector, kind in self._RECAPTCHA_BANNERS:
                element = frame.query_selector(selector)
                if not element:
                    continue
                # reCAPTCHA toggles these via aria-hidden on a wrapper, so
                # presence + non-empty text is the reliable test.
                if self._visible(element):
                    text = element.text_content() or ""
                    if text.strip():
                        return kind
        except Exception:
            pass
        return None

    def _get_verify_button(self, frame: Any) -> Optional[Any]:
        # `.//` — RELATIVE. `scope` is an ElementHandle whenever the widget is
        # markup on the host page rather than a vendor iframe (every
        # distorted-text captcha), and a document-rooted `//button` does not
        # resolve against an element handle: the query returned None even with
        # the button sitting inside that very element, so a typed code was never
        # submitted. On a Frame the context node is the document, where `.//` and
        # `//` mean the same thing, so the vendor paths are unaffected.
        for text in ("Verify", "Next", "Submit", "Skip"):
            lowered = text.lower()
            try:
                button = frame.query_selector(
                    f"xpath=.//button[contains(translate(., "
                    f"'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '{lowered}')]"
                    f" | .//div[@role='button' and contains(translate(., "
                    f"'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '{lowered}')]"
                )
                if self._visible(button):
                    return button
            except Exception:
                pass
        try:
            recaptcha_verify = frame.query_selector("#recaptcha-verify-button")
            if self._visible(recaptcha_verify):
                return recaptcha_verify
            hcaptcha_verify = frame.query_selector(".button-submit")
            if self._visible(hcaptcha_verify):
                return hcaptcha_verify
            # GeeTest: `<div class="geetest_submit geetest_disable">OK</div>`.
            #
            # Invisible to BOTH shapes above — a bare div carries no
            # role="button", and "OK" is on none of the four word lists. So
            # nothing was ever pressed on the puzzles that need pressing, and a
            # GeeTest board does not grade until you do: the solve loop re-read
            # the same unchanged panel and re-answered it identically until the
            # round cap. Ordered icon-click scored 0/31 and 0/13 that way while
            # the model was answering CORRECTLY (measured 2026-08-19). A driver
            # that discards a right answer is indistinguishable from a model
            # that cannot solve the puzzle, which is how this hid.
            #
            # Matched by CLASS, not by the word: `geetest_submit_tips` sits
            # beside it and also reads "OK", and pressing the tooltip does
            # nothing. `.geetest_submit` is a distinct token, so it cannot
            # match it. Mirrors getVerifyButton in js/src/solver.ts — CLAUDE.md
            # 1c, the two ports must behave the same.
            geetest_submit = frame.query_selector(".geetest_submit")
            if self._visible(geetest_submit):
                return geetest_submit
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------
    # Frame readiness
    # ------------------------------------------------------------------

    def _screenshot(self, element: Any, path: str, timeout_ms: Optional[int] = None,
                    animations: str = "disabled") -> None:
        """
        Short timeout, and animations disabled unless the caller needs the motion.

        Playwright's default 30s stability wait hangs per-screenshot on a
        closing/animating challenge element; that alone made a multi-round solve
        take ~115s. Failing fast and skipping a frame is strictly better.

        `animations` is 'disabled' everywhere the goal is a STABLE STILL, which
        is almost everywhere. It is not a formality: the flag fast-forwards
        finite animations and FREEZES infinite ones, which is exactly what a
        model-facing screenshot wants and exactly what a recording must not do.
        The burst and the animation probe pass 'allow' — see `solver.ts`
        `recordKeyframeBurst`, which learned this first: GeeTest's svg board
        cycles in CSS, so a burst taken with animations disabled captured the
        same picture forty times, the slicer honestly reported `mode=static`,
        and the solve went back to answering a single still. hCaptcha hid it,
        because it animates in canvas and the flag does not touch canvas.
        """
        element.screenshot(
            path=path,
            timeout=timeout_ms if timeout_ms is not None else 2_500,
            animations=animations,
        )

    def _element_frame_hash(self, element: Any) -> Optional[str]:
        path = _tmp_png("fh")
        try:
            self._screenshot(element, path)
            with open(path, "rb") as handle:
                return hashlib.sha1(handle.read()).hexdigest()
        except Exception:
            return None
        finally:
            _unlink(path)

    def _wait_for_element_settled(self, element: Any) -> str:
        """
        Poll until the element's pixels stop changing.

        Returns 'settled' | 'animated' | 'timeout'. This is a PIXEL settle; the
        caller pairs it with the DOM-level image wait so a static loading frame
        (a spinner on grey, below the diff threshold) isn't mistaken for painted
        tiles.
        """
        cfg = self.config
        start = time.monotonic() * 1000.0
        previous: Optional[str] = None
        frames: List[str] = []
        # This loop only COLLECTS polls; the verdict is `settle_verdict`'s, so
        # the recorded timelines in tests/test_animated_is_detected.py exercise
        # the same rule that runs here rather than a copy of it that can drift.
        samples: List[Tuple[float, bool]] = []
        try:
            while (time.monotonic() * 1000.0) - start < cfg.settle_timeout_ms:
                self._check_deadline("waiting for the challenge to settle")
                path = _tmp_png("settle")
                try:
                    self._screenshot(element, path)
                except Exception:
                    _unlink(path)
                    _delay(cfg.settle_poll_ms)
                    continue
                frames.append(path)
                if previous:
                    moved = self._has_movement(previous, path, cfg.settle_diff_threshold)
                    if len(frames) > 1:
                        _unlink(frames.pop(0))
                    samples.append(((time.monotonic() * 1000.0) - start, moved))
                    verdict = settle_verdict(
                        samples,
                        settle_frames=cfg.settle_frames,
                        animated_after_ms=cfg.animated_challenge_after_ms,
                    )
                    if verdict != "timeout":
                        return verdict
                previous = frames[-1]
                _delay(cfg.settle_poll_ms)
            return "timeout"
        finally:
            for path in frames:
                _unlink(path)

    # ------------------------------------------------------------------
    # Animated challenges
    # ------------------------------------------------------------------

    def _settle_or_animated(self, element: Any) -> bool:
        """Wait for the widget to settle; return whether it is animated instead.

        "Never settles" stopped being a failure. hCaptcha's "select the odd animal"
        fades its sprites on independent cycles and its "unique motion pattern"
        puzzle spins identical meshes — those challenges are animated BY DESIGN, and
        the answer only exists across frames. True here routes the caller to the
        recording path.

        `video_solve_enabled=False` restores the old behaviour for callers who would
        rather fail fast than spend the recording time.
        """
        # Already established for this widget — don't re-probe every round.
        if self._known_animated:
            return True

        with self._phase("settle"):
            verdict = self._wait_for_element_settled(element)
        if verdict != "animated":
            # 'settled' is not proof of static. The settle rule stops at the
            # first `settle_frames` of stillness, and every animated puzzle we
            # ship rests between screens for longer than that, so it calls all
            # of them settled (tests/test_animated_is_detected.py has the
            # measured timelines).
            #
            # What decides it instead is a RECORDING, taken once a round has
            # already answered the same thing twice — the signature of a
            # cycling board read as a still. The recording is self-checking: if
            # the widget really is static it slices to one keyframe and the
            # caller solves that frame as the still it is. So there is nothing
            # to be careful about here, and nothing to observe first.
            if not self._animated_probe_armed or self._animated_probe_done:
                return False
            self._animated_probe_done = True
            self._animated_probe_armed = False
            _log("[animated] the same answer came back twice — recording the "
                 "challenge to see whether it is cycling")

        # Latch either route. Re-deciding every round would re-record a widget
        # we have already SEEN animate, and on a puzzle that rests between
        # screens a later recording can land inside a hold, slice to one frame,
        # and flip the solve back onto the still path halfway through.
        self._known_animated = True

        if not self.config.video_solve_enabled:
            raise AnimatedChallengeError(
                "the challenge never settles and video_solve_enabled is off"
            )
        _log("[animated] challenge is animated — solving it from keyframes")
        return True

    def _arm_animated_probe(self) -> None:
        """Round N finished without finishing the CHALLENGE — probe next round.

        Deliberately not armed for round 1. A static puzzle solves there and
        never pays anything, which is what keeps the common case at the ~530ms
        settle it has today. Only a widget that survived a round is worth
        spending a cycle of observation on.
        """
        if self.config.animated_probe_enabled and not self._animated_probe_done:
            self._animated_probe_armed = True

    def _should_speculate(self, puzzle_source: str, text_mode: bool) -> bool:
        """Should this round record while it asks?

        reCAPTCHA is excluded for the same reason it is excluded from the
        animated path at all: its dynamic 3x3 REPLACES tiles in place and has
        its own fade gates, so a burst there would film a fade and call it a
        cycle. A distorted-text round is excluded because the answer is a
        string, not a place, and no recording helps read one.
        """
        cfg = self.config
        if not cfg.video_solve_enabled:
            return False
        if not cfg.speculative_burst_enabled:
            return False
        if puzzle_source == "recaptcha":
            return False
        return not text_mode

    def _speculate(
        self, element: Any, shot: str, puzzle_source: str,
        retry_mode: Optional[str], text_mode: bool,
    ) -> Tuple[List[CaptchaAction], List[Dict[str, Any]], Optional[str]]:
        """Read the still and film the widget at once.

        Returns `(actions, usage, keyframe_dir)`. `keyframe_dir` is None when
        the board turned out to be still — the caller then behaves exactly as
        it always has.
        """
        from concurrent.futures import ThreadPoolExecutor

        import cv2

        from .keyframes import extract_keyframes, write_keyframes

        cfg = self.config
        fps = max(1, int(cfg.video_burst_fps))
        floor_frames = max(1, round(cfg.video_burst_duration_ms / (1000.0 / fps)))
        total = max(floor_frames, round(cfg.video_burst_max_ms / (1000.0 / fps)))
        interval = 1.0 / fps

        frames: List[Any] = []
        order: List[str] = []
        last_digest: Optional[str] = None
        cycle_closed = False
        probe = _tmp_png("spec")

        pool = ThreadPoolExecutor(max_workers=1)
        try:
            # `_get_solution` DIRECTLY, not through the freshness guard.
            #
            # Two reasons, and the first is a hard constraint: the guard
            # screenshots the element to decide whether the answer went stale,
            # and a Playwright call from this worker would violate the sync
            # API's thread affinity. Only the planner's HTTP request may cross.
            #
            # The second is that the guard is redundant here. It exists to ask
            # "did the widget change while the model was reading it" — which is
            # exactly what the burst on the main thread is answering, at the
            # noise floor, from every frame rather than from one diff at the
            # end.
            fut = pool.submit(
                self._get_solution, shot, puzzle_source, retry_mode, text_mode)
            with self._phase("burst"):
                for i in range(total):
                    start = time.monotonic()
                    try:
                        self._screenshot(element, probe, animations="allow")
                        img = cv2.imread(probe)
                    except Exception as exc:  # noqa: BLE001 — a dropped frame is not fatal
                        _debug(f"speculative frame {i} failed: {exc}")
                        img = None
                    if img is not None:
                        frames.append(img)
                        try:
                            with open(probe, "rb") as fh:
                                d = hashlib.sha1(fh.read()).hexdigest()
                        except Exception:  # noqa: BLE001
                            d = None
                        if d is not None and d != last_digest:
                            if d in order and len(order) >= 2:
                                cycle_closed = True
                            elif d not in order:
                                order.append(d)
                            last_digest = d
                    moved = len(order) >= 2
                    # Still, and the answer has landed: nothing more to film.
                    if not moved and fut.done():
                        break
                    # Moving, and the cycle has closed: every screen is in hand.
                    if moved and cycle_closed and len(frames) >= floor_frames:
                        _log(f"[animated] cycle closed after {len(frames) * interval:.1f}s "
                             f"({len(order)} screens); stopping the burst")
                        break
                    wait = interval - (time.monotonic() - start)
                    if wait > 0:
                        time.sleep(wait)

            if len(order) < 2:
                actions, usage = fut.result()
                return actions, usage, None

            _log("[animated] the widget moved while the model was reading it — "
                 "dropping the still answer and finishing the recording.")
            fut.result()          # drained, not used: it describes a screen that has gone
            # THE ESCALATION BUYS ITS OWN BUDGET, once per solve — and only now,
            # because a speculative burst that turned out to be unnecessary must
            # not extend the deadline of a still solve.
            if cfg.video_solve_enabled and not self._video_budget_granted:
                self._video_budget_granted = True
                if self._deadline_ms is not None:
                    self._deadline_ms += cfg.video_budget_ms()
                    _log(f"[animated] +{cfg.video_budget_ms()}ms for the recording path")
            kfset = extract_keyframes(frames, fps=float(fps))
            self._keyframe_mode = kfset.mode
            self._keyframe_steady_screens = kfset.steady_screens
            keyframe_dir = tempfile.mkdtemp(prefix="ckkf_")
            paths = write_keyframes(kfset, keyframe_dir)
            with self._phase("inference"):
                actions, usage = self._get_keyframe_solution(paths)
            self._animated_plan = (paths, keyframe_dir, actions, usage)
            return actions, usage, keyframe_dir
        finally:
            pool.shutdown(wait=True)
            _unlink(probe)

    def _discard_animated_plan(self) -> None:
        """Forget the recorded answer, and delete the frames it was holding."""
        plan = self._animated_plan
        self._animated_plan = None
        if plan and plan[1]:
            shutil.rmtree(plan[1], ignore_errors=True)

    def _record_keyframes(self, element: Any) -> Tuple[List[str], str]:
        """Record the widget and return `(keyframe_paths, temp_dir)`.

        Screenshots the element at the collector's burst geometry (4 s @ 10 fps),
        then hands the frames to the SAME slicer the training data was cut with
        (`keyframes.extract_keyframes`). That shared code path is the whole point:
        the model answers with a frame NUMBER, and a number only means something if
        the live set was sliced the way the trained set was.

        Frames are kept in memory and sliced from there — no intermediate mp4. The
        old pipeline encoded one, and every clip it produced was mp4v, a codec the
        serving side may or may not decode. Skipping the encode removes that whole
        class of silent failure along with the disk round-trip.

        The caller owns `temp_dir` and must remove it once the actions are done
        with: the returned paths are read back by the wait gate on every poll, so
        cleaning up any earlier would break the click.

        Raises AnimatedChallengeError if nothing could be captured.
        """
        from .keyframes import extract_keyframes, write_keyframes

        cfg = self.config
        fps = max(1, int(cfg.video_burst_fps))
        # A BURST MUST OUTLAST ONE FULL CYCLE OR IT OMITS A SCREEN — and may
        # omit the one the answer is on. CLAUDE.md states that rule; the
        # COLLECTOR was fixed for it and the driver was not.
        #
        # Measured on the live GeeTest svg board, 2026-09-07: three screens
        # holding 1.5-1.9s each, FULL PERIOD 5.3s, against a 4000ms window. So
        # the model was shown two screens of three on about half of attempts,
        # and when the target sat on the third it was being asked a question
        # with the answer removed.
        #
        # Fixed by watching rather than by a bigger constant: the frames of one
        # screen are byte-identical, so a digest per frame says exactly when the
        # board returns to a screen already recorded — the cycle is closed and
        # there is nothing left to capture. A continuous animation never
        # repeats and stops at the floor, keeping today's behaviour.
        floor_frames = max(1, round(cfg.video_burst_duration_ms / (1000.0 / fps)))
        total = max(floor_frames,
                    round(cfg.video_burst_max_ms / (1000.0 / fps)))
        interval = 1.0 / fps

        # THE ESCALATION BUYS ITS OWN BUDGET, once per solve.
        #
        # `overall_solve_timeout_ms` counts rounds, and this is not a round —
        # see `video_budget_ms` for the arithmetic and for what it cost not to
        # have it. Granted here rather than where the probe arms because this is
        # the one place every path into a recording goes through.
        if cfg.video_solve_enabled and not self._video_budget_granted:
            self._video_budget_granted = True
            if self._deadline_ms is not None:
                self._deadline_ms += cfg.video_budget_ms()
                _log(f"[animated] +{cfg.video_budget_ms()}ms for the recording path")

        # Checked ONCE, before the first frame — never per frame.
        #
        # A HALF-RECORDED BURST IS WORTHLESS: the slicer reads a clip's temporal
        # structure, so stopping at frame 27 of 40 does not produce a shorter
        # answer, it produces a recording that may not contain the screen the
        # answer is on. Aborting mid-way therefore threw away the whole burst
        # AND the ~3s already spent making it, to report a timeout. The burst is
        # also fixed-length and short, so it is not one of the places that "can
        # legitimately spin for a long time" that `_check_deadline` is for.
        #
        # If the budget cannot fit one, say THAT — a caller who has tightened
        # `overall_solve_timeout_ms` below what a recording costs has asked for
        # something impossible, and should hear it rather than watch a burst die
        # partway through every time.
        if self._deadline_ms is not None:
            left = self._deadline_ms - time.monotonic() * 1000.0
            if left < cfg.video_burst_duration_ms:
                raise CaptchaSolveError(
                    f"only {left:.0f}ms of the {cfg.overall_solve_timeout_ms}ms solve "
                    f"budget is left and an animated recording needs "
                    f"{cfg.video_burst_duration_ms}ms — not starting one that would "
                    f"be cut off mid-way. Raise overall_solve_timeout_ms or "
                    f"video_extra_inference_ms, or set video_solve_enabled=False."
                )

        frames: List[Any] = []
        order: List[str] = []          # distinct screens, in first-seen order
        last_digest: Optional[str] = None
        cycle_closed = False

        remaining = max(0, total - len(frames))
        shot = _tmp_png("burst")
        # A burst that runs far past its own length is a hung screenshot, not a
        # tight budget — bounded separately so the two cannot be confused.
        burst_deadline = (time.monotonic() * 1000.0
                          + 3 * cfg.video_burst_duration_ms + 5_000)
        try:
            for i in range(remaining):
                if time.monotonic() * 1000.0 > burst_deadline:
                    raise CaptchaSolveError(
                        f"the animated recording stalled: {i} of {remaining} frames "
                        f"in {3 * cfg.video_burst_duration_ms + 5000}ms. The widget "
                        f"is not screenshotting."
                    )
                start = time.monotonic()
                try:
                    # animations ALLOWED — see `_screenshot`. The JS port has
                    # done this since the GeeTest svg burst came back as 40
                    # copies of one picture; this port never did, so every
                    # CSS-animated vendor was recorded frozen.
                    self._screenshot(element, shot, animations="allow")
                except Exception as exc:  # noqa: BLE001 — a dropped frame is not fatal
                    _debug(f"burst frame {i} failed: {exc}")
                else:
                    import cv2

                    img = cv2.imread(shot)
                    if img is not None:
                        frames.append(img)
                        try:
                            with open(shot, "rb") as fh:
                                d = hashlib.sha1(fh.read()).hexdigest()
                        except Exception:  # noqa: BLE001 — no digest, no early stop
                            d = None
                        if d is not None and d != last_digest:
                            # A screen already recorded, coming back after
                            # another one: the loop has closed and every screen
                            # is now in the clip.
                            if d in order and len(order) >= 2:
                                cycle_closed = True
                            elif d not in order:
                                order.append(d)
                            last_digest = d
                # Past the floor with the cycle closed — more frames are the
                # same screens again, paid for in budget the solve needs.
                if cycle_closed and len(frames) >= floor_frames:
                    _log(f"[animated] cycle closed after {len(frames) * interval:.1f}s "
                         f"({len(order)} screens); stopping the burst")
                    break
                # Drift-corrected: a slow screenshot must not stretch the clip, or
                # the recording covers more wall-clock than the model trained on and
                # a cycle's period lands differently across the frames.
                wait = interval - (time.monotonic() - start)
                if wait > 0 and i < remaining - 1:
                    time.sleep(wait)
        finally:
            _unlink(shot)

        if not frames:
            raise AnimatedChallengeError(
                "could not record the animated challenge (no frame screenshotted)"
            )
        _log(f"[animated] recorded {len(frames)} frames at {fps}fps")

        kfset = extract_keyframes(frames, fps=float(fps))
        temp_dir = tempfile.mkdtemp(prefix="ck_keyframes_")
        paths = write_keyframes(kfset, temp_dir, stem="challenge")
        _log(f"[animated] sliced to {len(paths)} keyframe(s) (mode={kfset.mode})")
        # The wait gate consults this: `even` means the extractor found no state
        # that RECURS, so there is nothing for the page to come back to.
        self._keyframe_mode = kfset.mode
        self._keyframe_steady_screens = kfset.steady_screens
        return [str(p) for p in paths], temp_dir

    def _wait_for_keyframe(self, element: Any, keyframe_path: str,
                           point_norm: Tuple[float, float]) -> bool:
        """Hold until the widget looks like `keyframe_path` around `point_norm`.

        This is the reason an animated answer names a frame. The model picked the
        moment its target was visible; the coordinates are only correct at that
        moment. Clicking as soon as the answer arrives lands on whatever the sprite
        happens to be doing — for a cross-fade, usually background.

        Compares only the neighbourhood of the action point, with the same box and
        the same metric the training label was chosen with
        (`keyframes.region_box` / `region_diff_ratio`). Local rather than
        whole-frame because everything ELSE in these puzzles is also moving: a
        whole-frame match would need every unrelated sprite to align too, and would
        essentially never open.

        Returns whether the state was reached. On timeout the caller clicks anyway
        — see `keyframe_wait_timeout_ms`.

        NOT ATTEMPTED on an `even` clip. `keyframes.py` picks that mode precisely
        when the clip never revisits a picture it has already shown — a slow
        one-way change, a rotation, a cross-fade — so there is no state for the
        page to come BACK to and the wait can only ever run out. It is not a
        cheap failure: the gate polls the full `keyframe_wait_timeout_ms` (6s)
        PER CLICK before giving up and clicking anyway.
        And `even` is not the rare case. All 116 real clips under
        `cleanSamples/test/raw/**/keyframes/` are `even`; `cycle` has never once
        fired on real footage, which keyframes.py itself records ("real state
        separations top out around 0.007, well under this"). So on real traffic
        this gate was 6s of dead time on every animated click, always followed by
        the same click it would have made immediately. Measured on
        `hcaptcha_rotating_obj_video`: 6.0s of a 28.8s solve, closest region diff
        0.0721 against a 0.05 tolerance, then it clicked and solved.

        The gate stays for `cycle`/`static`, where a state genuinely does recur
        and waiting is the difference between clicking the sprite and clicking
        the background.
        """
        import cv2

        from .keyframes import MATCH_REGION_TOLERANCE, region_box, region_diff_ratio

        # NOT `mode == "even"`. A board that holds a few screens is worth
        # waiting for whether or not one 4s burst was long enough to catch it
        # repeating — and it never is, which is why every real animated capture
        # slices `even` and this gate was off on all of them. What is genuinely
        # not worth waiting for is a clip with NO steady screens: a rotation, a
        # one-way fade, a sprite crossing. Measured: 0 steady screens for all
        # five continuous hCaptcha video types, 2-3 for every real GeeTest svg.
        if self._keyframe_steady_screens < 2:
            _log(f"[animated] clip sits on {self._keyframe_steady_screens} steady "
                 f"screen(s); nothing to come back to, acting on the model's "
                 f"frame without waiting")
            return False

        ref = cv2.imread(keyframe_path)
        if ref is None:
            _debug(f"keyframe {keyframe_path} unreadable; not waiting")
            return False
        box = region_box(ref.shape[1::-1], point_norm)

        cfg = self.config
        deadline = (time.monotonic() * 1000.0) + cfg.keyframe_wait_timeout_ms
        probe = _tmp_png("kfwait")
        best = 1.0
        polls = 0
        try:
            while (time.monotonic() * 1000.0) < deadline:
                self._check_deadline("waiting for the challenge keyframe")
                try:
                    # animations="allow": the motion is the thing being watched.
                    # "disabled" FREEZES an infinite CSS animation, so the gate
                    # would photograph one frozen screen on every poll and could
                    # never see the board come round. The JS twin carries the
                    # full note.
                    self._screenshot(element, probe, animations="allow")
                    live = cv2.imread(probe)
                except Exception as exc:  # noqa: BLE001
                    _debug(f"keyframe probe failed: {exc}")
                    live = None
                if live is not None:
                    d = region_diff_ratio(ref, live, box)
                    best = min(best, d)
                    if d <= MATCH_REGION_TOLERANCE:
                        _log(f"[animated] widget matched the chosen keyframe (diff={d:.4f})")
                        return True
                    # NOT THIS BOARD AT ALL — stop waiting for a screen of it.
                    # Two SCREENS of one board differ by 0.0056 (measured on
                    # GeeTest svg); a different board reads 0.77. So a diff this
                    # large means the puzzle we recorded is gone — solved,
                    # refreshed or replaced — and the rest of the budget buys
                    # nothing. See the JS twin for the full note.
                    polls += 1
                    if polls >= _NOT_THIS_BOARD_POLLS and best > _NOT_THIS_BOARD_DIFF:
                        _log(f"[animated] the widget no longer resembles the recorded "
                             f"board (best diff={best:.4f} over {polls} polls); not "
                             f"waiting out the budget")
                        self._discard_animated_plan()
                        return False
                _delay(cfg.keyframe_wait_poll_ms)
        finally:
            _unlink(probe)
        # NEVER SAW IT. Either the board moved on to a different challenge or
        # the answer was for a screen this widget does not show. Both mean the
        # plan is spent, so the next round records afresh rather than
        # re-clicking a cell chosen from pictures that are gone.
        self._discard_animated_plan()
        _log(f"[animated] widget never matched the chosen keyframe within "
             f"{cfg.keyframe_wait_timeout_ms}ms (closest diff={best:.4f}); "
             f"clicking on the model's coordinates anyway, and recording "
             f"afresh next round")
        return False

    def _wait_for_change_since(self, element: Any, since_hash: str) -> bool:
        """After a submit the frame MUST change (next round, or closing)."""
        cfg = self.config
        start = time.monotonic() * 1000.0
        while (time.monotonic() * 1000.0) - start < cfg.post_submit_change_timeout_ms:
            current = self._element_frame_hash(element)
            if current and current != since_hash:
                return True
            _delay(cfg.settle_poll_ms)
        return False

    def _wait_for_hcaptcha_challenge_images(self, challenge_iframe: Any) -> None:
        """
        Block until the challenge's task images have actually painted.

        hCaptcha sets each tile's background-image once the asset loads, and
        ships an empty `url("")` placeholder before it arrives. Best-effort: a
        timeout falls through to the screenshot, where the existing fail-fast
        path still covers a genuinely unsupported puzzle.

        "NOTHING TO WAIT FOR" IS READY. The last clause used to be
        `return !!(example && ...)`, which is false when there is no example
        image — so a challenge with no tile grid, no canvas and no example
        polled until the timeout and then carried on regardless. Measured on
        `hcaptcha_number_with_highest_value_video`: 24.0s of a 45.2s solve, the
        full 8s three times over, more than half the budget spent asking a
        question about elements that were not on the page. A readiness gate can
        only report on what it can see; with nothing to check it has no opinion,
        and "no opinion" must not read as "not ready".
        """
        cfg = self.config
        try:
            frame = challenge_iframe.content_frame()
            if not frame:
                return
            frame.wait_for_selector(".prompt-text", state="visible",
                                    timeout=cfg.hcaptcha_images_timeout_ms)
            frame.wait_for_function(
                """() => {
                    const tiles = Array.from(
                        document.querySelectorAll('.task-image .image, .task .image'));
                    if (tiles.length > 0) {
                        return tiles.every((el) => {
                            const bg = getComputedStyle(el).backgroundImage;
                            return bg && bg !== 'none' && !/url\\(["']?["']?\\)/.test(bg);
                        });
                    }
                    const canvas = document.querySelector('canvas');
                    if (canvas && canvas.width > 0 && canvas.height > 0) return true;
                    const example = document.querySelector(
                        '.challenge-example img, .image-wrapper img');
                    if (example) return example.complete && example.naturalWidth > 0;
                    return true;   // nothing here to be waited for
                }""",
                timeout=cfg.hcaptcha_images_timeout_ms,
            )
        except Exception:
            pass  # timed out or detached mid-load; screenshot anyway

    def _wait_for_grid_cells_loaded(self, element: Any) -> bool:
        """
        reCAPTCHA fades new tiles in over ~1s, on first load and on every
        in-place refresh. Screenshotting mid-fade feeds the model a partial grid.
        """
        cfg = self.config
        start = time.monotonic() * 1000.0
        frames: List[str] = []
        try:
            while (time.monotonic() * 1000.0) - start < cfg.grid_load_timeout_ms:
                self._check_deadline("waiting for grid cells to load")
                path = _tmp_png("gridpoll")
                try:
                    self._screenshot(element, path)
                except Exception:
                    _unlink(path)
                    _delay(cfg.grid_load_poll_interval_ms)
                    continue
                frames.append(path)
                if len(frames) >= 2:
                    boxes = self._find_grid(frames[-1])
                    if boxes:
                        states = self._grid_cell_states(frames[-2], frames[-1], boxes)
                        if (
                            states
                            and not states["empty"]
                            and not states["changing"]
                            and states["loaded"]
                        ):
                            return True
                    _unlink(frames.pop(0))
                _delay(cfg.grid_load_poll_interval_ms)
            return False
        except Exception:
            return False
        finally:
            for path in frames:
                _unlink(path)

    def _get_grid_boxes(self, element: Any) -> Optional[Dict[str, Any]]:
        """
        Detect the grid ONCE per puzzle session. Geometry is stable across the
        in-place dynamic refresh (only tile images change), so callers cache it.
        """
        path = _tmp_png("findgrid")
        try:
            self._screenshot(element, path, timeout_ms=self.config.element_screenshot_timeout_ms)
            boxes = self._find_grid(path)
            if not boxes or len(boxes) not in (9, 16):
                return None
            dims = _read_png_dimensions(path)
            if not dims:
                return None
            return {
                "boxes": [list(b) for b in boxes],
                "size": 4 if len(boxes) == 16 else 3,
                "screenshot_w": dims[0],
                "screenshot_h": dims[1],
            }
        except Exception:
            return None
        finally:
            _unlink(path)

    # ------------------------------------------------------------------
    # Freshness guard
    # ------------------------------------------------------------------

    def _frame_changed_since(self, element: Any, prior_path: str, threshold: float) -> bool:
        probe = _tmp_png("freshcheck")
        try:
            self._screenshot(element, probe)
            return self._has_movement(prior_path, probe, threshold)
        except Exception:
            return False
        finally:
            _unlink(probe)

    def _solve_frame_freshness_guarded(
        self,
        element: Any,
        initial_shot: str,
        run_query: Callable[[str], Tuple[List[CaptchaAction], List[Dict[str, Any]]]],
    ) -> Tuple[List[CaptchaAction], List[Dict[str, Any]]]:
        """
        Never act on a stale frame.

        If the frame changed WHILE the model was generating, its answer describes
        an undeveloped frame whose tiles no longer line up — so re-screenshot and
        re-solve on the developed one. Token usage from every attempt is merged,
        because every attempt was really billed.
        """
        cfg = self.config
        owned: List[str] = []
        merged_usage: List[Dict[str, Any]] = []
        try:
            current = initial_shot
            actions, usage = run_query(current)
            merged_usage.extend(usage)
            if not cfg.stale_frame_resolve_enabled:
                return actions, merged_usage

            # DID THE PICTURE MOVE WHILE THE MODEL READ IT? Asked once per
            # round, at the noise floor rather than the staleness threshold. A
            # still puzzle answers no for the price of one frame diff; a
            # cycling one answers yes and stops being treated as a still a
            # whole round earlier than "the same answer came back twice"
            # allows. Acting on one observation is safe because the recording
            # is self-checking — see `_settle_or_animated`.
            if self._frame_changed_since(element, current,
                                         _MOVED_DURING_INFERENCE_DIFF):
                self._arm_animated_probe()
                _log("[freshness] the widget moved while the model was reading "
                     "it, with nothing clicked — recording it rather than "
                     "answering another still.")

            changed_during_inference = 0
            for attempt in range(cfg.max_stale_frame_resolves):
                if not self._frame_changed_since(element, current, cfg.stale_frame_diff_threshold):
                    break  # frame held still through inference — answer is valid
                # CHANGED AGAIN, having already re-solved once and touched
                # nothing. One change is a board still developing — tiles
                # fading in — which is what the re-solve below is for. A SECOND
                # change in the same round is a board that does not stop, and
                # re-solving it is futile: each answer is for a screen that will
                # be gone before the click.
                #
                # This is the earliest the driver can know. Both ports otherwise
                # wait for the same answer to come back twice, which costs a
                # whole extra round and its inference — ~10s of a 40s solve
                # spent rediscovering what the guard has just watched happen.
                changed_during_inference += 1
                if changed_during_inference >= 2:
                    self._arm_animated_probe()
                    _log("[freshness] the frame changed twice during inference with "
                         "nothing clicked — this board cycles; recording it rather "
                         "than re-solving a screen that has gone.")
                    return actions, merged_usage
                fresh = _tmp_png("freshsolve")
                try:
                    self._screenshot(element, fresh)
                except Exception:
                    _unlink(fresh)
                    break  # can't grab a fresh frame — act on what we have
                owned.append(fresh)
                _log(
                    f"[freshness] frame changed during inference "
                    f"(re-solve {attempt + 1}/{cfg.max_stale_frame_resolves})"
                )
                current = fresh
                actions, usage = run_query(current)
                merged_usage.extend(usage)
            return actions, merged_usage
        finally:
            for path in owned:
                _unlink(path)

    # ------------------------------------------------------------------
    # reCAPTCHA 3x3 dynamic driver
    # ------------------------------------------------------------------

    def _cell_center_page(self, cell: int, session: _GridSession) -> Tuple[float, float]:
        x1, y1, x2, y2 = session.grid_boxes[cell - 1]
        return (
            session.element_box["x"] + ((x1 + x2) / 2) * session.scale_x,
            session.element_box["y"] + ((y1 + y2) / 2) * session.scale_y,
        )

    def _hover_cell(self, page: Any, session: _GridSession, cell: int) -> None:
        # A hover is mimicry of a resting CURSOR, so on a device that has none
        # it is not weaker mimicry — it is a mousemove at a touch-only widget.
        if not self._human.hovers:
            return
        first = session.grid_boxes[0]
        cell_w = (first[2] - first[0]) * session.scale_x
        cell_h = (first[3] - first[1]) * session.scale_y
        cx, cy = self._cell_center_page(cell, session)
        self._smooth_move(
            page,
            cx + (random.random() - 0.5) * cell_w * 0.4,
            cy + (random.random() - 0.5) * cell_h * 0.4,
        )

    @staticmethod
    def _bbox_to_cell(
        bbox: Sequence[float], grid_boxes: Sequence[Sequence[int]], width: int, height: int
    ) -> Optional[int]:
        cx = ((float(bbox[0]) + float(bbox[2])) / 2) * width
        cy = ((float(bbox[1]) + float(bbox[3])) / 2) * height
        for index, (x1, y1, x2, y2) in enumerate(grid_boxes):
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                return index + 1  # 1-indexed, row-major, matching find_grid
        return None

    @staticmethod
    def _order_by_priority(loading: Sequence[int], priority: Sequence[int]) -> List[int]:
        remaining = set(loading)
        ordered: List[int] = []
        for cell in priority:
            if cell in remaining:
                ordered.append(cell)
                remaining.discard(cell)
        ordered.extend(sorted(remaining))
        return ordered

    def _watch_clicked_tiles(
        self, page: Any, element: Any, session: _GridSession, priority: Sequence[int] = ()
    ) -> Tuple[List[int], bool]:
        """
        Watch the just-clicked tiles until the widget says what it did with them.

        Returns `(loading, chipped)`. Two answers because reCAPTCHA gives a click
        one of exactly two replies, and they are the two kinds of board:

          * the small blue CHIP in the tile's top-left corner — the photo was
            KEPT. Nothing is on its way in, the selection is the answer, and the
            caller should press Verify (`chipped`).
          * the photo blanking or dissolving under a large centred check — the
            tile is being SWAPPED, and what lands may match too, so the board has
            to be read again (`loading`).

        A widget that swaps one clicked cell swaps them all, so the two never
        share a board and one look at the tiles we just clicked settles it. The
        chip is what `detect_selected_cells` reports as `selected` — it looks for
        it in the top-left corner only, behind a circularity and a centroid test
        a centred check fails — and we already read it every poll.

        `chipped` needs EVERY watched tile, because a partial reading is a
        misread and the two mistakes do not cost the same: calling a swapping
        board finished submits half an answer and burns the attempt, while
        calling a chipped board unfinished costs one inference. It is also only
        ever OUR click that a chip can be reporting: tiles already wearing one
        are filtered out of the model's answer before we click, so a chip on a
        watched tile arrived in response to this round.

        The blank/fade transition LAGS the click by a beat: reCAPTCHA holds a
        clicked tile selected (old image visible) for ~1-3s and only then blanks
        it. A single snapshot right after clicking therefore sees nothing and
        concludes the puzzle is finished. Hovers a clicked tile each poll so the
        cursor keeps moving during the wait.
        """
        cfg = self.config
        watch = list(priority) if priority else None
        start = time.monotonic() * 1000.0
        frames: List[str] = []
        hover_index = 0
        try:
            first = _tmp_png("loadchk")
            self._screenshot(element, first)
            frames.append(first)

            while (time.monotonic() * 1000.0) - start < cfg.recaptcha_fade_onset_grace_ms:
                self._check_deadline("watching for the tile refresh")
                iteration_start = time.monotonic() * 1000.0
                if priority:
                    try:
                        self._hover_cell(page, session, priority[hover_index % len(priority)])
                    except Exception:
                        pass
                    hover_index += 1
                # Enforce a minimum inter-frame gap: the in-process CV call is
                # near-instant, so without this the polls fire back-to-back on
                # near-identical frames and miss a slow fade.
                elapsed = (time.monotonic() * 1000.0) - iteration_start
                if elapsed < cfg.recaptcha_dynamic_fade_poll_ms:
                    _delay(cfg.recaptcha_dynamic_fade_poll_ms - elapsed)

                path = _tmp_png("loadchk")
                self._screenshot(element, path)
                frames.append(path)

                states = self._grid_cell_states(frames[-2], frames[-1], session.grid_boxes)
                # Chip first: a chip landing on a tile ZOOMS its photo out, which
                # reads as `changing` on the same frame that shows the chip. Test
                # the swap first and every chipped board looks like a swapping
                # one for as long as the animation runs.
                if priority and set((states or {}).get("selected", [])).issuperset(priority):
                    return [], True
                in_scope = lambda c: watch is None or c in watch  # noqa: E731
                empty = [c for c in (states or {}).get("empty", []) if in_scope(c)]
                changing = [c for c in (states or {}).get("changing", []) if in_scope(c)]
                loading = sorted(set(empty) | set(changing))
                if loading:
                    return self._order_by_priority(loading, priority), False
                _unlink(frames.pop(0))
            return [], False
        except Exception as exc:
            _debug(f"fade-onset error: {exc}")
            return [], False
        finally:
            for path in frames:
                _unlink(path)

    def _wait_for_any_clicked_tile_loaded(
        self, page: Any, element: Any, session: _GridSession, fading_cells: Sequence[int]
    ) -> bool:
        """Wait for at least one blank/fading cell to finish reloading."""
        if not fading_cells:
            return True
        cfg = self.config
        start = time.monotonic() * 1000.0
        frames: List[str] = []
        hover_index = 0
        try:
            while (time.monotonic() * 1000.0) - start < cfg.recaptcha_dynamic_fade_wait_ms:
                self._check_deadline("waiting for a tile to reload")
                iteration_start = time.monotonic() * 1000.0
                if cfg.recaptcha_tile_hover_enabled:
                    try:
                        self._hover_cell(
                            page, session, fading_cells[hover_index % len(fading_cells)]
                        )
                    except Exception:
                        pass
                    hover_index += 1
                elapsed = (time.monotonic() * 1000.0) - iteration_start
                if elapsed < cfg.recaptcha_dynamic_fade_poll_ms:
                    _delay(cfg.recaptcha_dynamic_fade_poll_ms - elapsed)

                path = _tmp_png("fadepoll")
                self._screenshot(element, path)
                frames.append(path)

                if len(frames) >= 2:
                    states = self._grid_cell_states(frames[-2], frames[-1], session.grid_boxes)
                    if states and [c for c in fading_cells if c in states["loaded"]]:
                        return True
                    _unlink(frames.pop(0))
            return False
        except Exception as exc:
            _debug(f"wait-load error: {exc}")
            return False
        finally:
            for path in frames:
                _unlink(path)

    def _solve_recaptcha_grid(
        self,
        page: Any,
        element: Any,
        retry_mode: Optional[str],
        grid: Dict[str, Any],
        element_box: Dict[str, float],
    ) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        Multi-round driver for reCAPTCHA 3x3 dynamic puzzles. One call = one
        puzzle session.

        The shared Python half is authoritative about WHAT to do — it runs the
        blue-badge detector, filters already-selected and still-loading tiles,
        and returns `click` / `wait` / `done`. This driver owns the human-like
        WAITING that logic cannot: after a click round it hovers the just-clicked
        tiles and waits for at least one to finish reloading before re-solving,
        so we never burn a model call on a grid that is still mid-fade.

        Only a board that SWAPS a clicked tile out is worth those extra rounds.
        One that ticks the tile and keeps the photo has been fully answered by
        the round that clicked it — same as the 4x4 — so `_watch_clicked_tiles`
        reports the chip and this submits there and then. Rounds 2..N exist for
        the fading board and nothing else.

        Submits on `done`, and on a board that ticked our clicks — never on a
        round-cap exit, which is left to the outer loop to re-detect and decide.
        """
        cfg = self.config
        session = _GridSession(
            grid_boxes=grid["boxes"],
            element_box=element_box,
            scale_x=element_box["width"] / grid["screenshot_w"],
            scale_y=element_box["height"] / grid["screenshot_h"],
            screenshot_w=grid["screenshot_w"],
            screenshot_h=grid["screenshot_h"],
        )

        clicked_order: List[int] = []
        performed_action = False
        should_submit = False
        all_usage: List[Dict[str, Any]] = []
        pending_retry = retry_mode

        for round_index in range(1, cfg.recaptcha_max_dynamic_rounds + 1):
            # Each round can legitimately spend ~10s waiting on fades, so eight
            # of them plus the model calls can outlast the whole solve budget.
            self._check_deadline(f"recaptcha grid round {round_index}")
            # Round 1 skips it: `_solve_single` has just waited for these cells,
            # read the grid boxes off the loaded board and handed over, and
            # nothing has touched the widget in between. Rounds 2..N still wait,
            # because by then THIS driver has clicked and the tiles really are
            # reloading. Measured 2148ms x2 on one solve.
            if round_index > 1:
                with self._phase("grid-load"):
                    self._wait_for_grid_cells_loaded(element)
            shot = _tmp_png("recap")
            action: Optional[Dict[str, Any]] = None
            try:
                with self._phase("screenshot"):
                    self._screenshot(element, shot,
                                     timeout_ms=cfg.element_screenshot_timeout_ms)
                retry_for_round = pending_retry
                pending_retry = None  # only round 1 carries the inbound hint
                with self._phase("inference"):
                    actions, usage = self._solve_frame_freshness_guarded(
                        element,
                        shot,
                        lambda image_path: self._get_solution(
                            image_path, "recaptcha", retry_for_round),
                    )
                all_usage.extend(usage)
                action = _as_dict(actions[0]) if actions else None
            finally:
                _unlink(shot)

            if not action or action.get("action") == "done":
                _log(f"[recaptcha-grid] round {round_index}: done; submitting.")
                should_submit = True
                break

            if action.get("action") == "wait":
                _log(f"[recaptcha-grid] round {round_index}: waiting for tiles.")
                with self._phase("fade-wait"):
                    loading, _ = self._watch_clicked_tiles(
                        page, element, session, clicked_order)
                    self._wait_for_any_clicked_tile_loaded(
                        page, element, session, loading)
                continue

            if action.get("action") == "click":
                bboxes = action.get("target_bounding_boxes") or (
                    [action["target_bounding_box"]] if action.get("target_bounding_box") else []
                )
                if not bboxes:
                    # Malformed click: treat as a soft wait so we never submit
                    # prematurely, and re-solve next round.
                    _log(f"[recaptcha-grid] round {round_index}: click with no bboxes; re-solving.")
                    _delay(500)
                    continue

                clicked_this_round: List[int] = []
                for bbox in bboxes:
                    cell = self._bbox_to_cell(
                        bbox, session.grid_boxes, session.screenshot_w, session.screenshot_h
                    )
                    self._execute_click(page, {"target_bounding_box": bbox}, element_box)
                    if cell is not None:
                        clicked_order.append(cell)
                        clicked_this_round.append(cell)
                    self._human.pause("between")
                performed_action = True
                _log(
                    f"[recaptcha-grid] round {round_index}: clicked {len(bboxes)} tile(s) "
                    f"-> cells {clicked_this_round}."
                )

                with self._phase("fade-wait"):
                    loading, chipped = self._watch_clicked_tiles(
                        page, element, session, clicked_this_round
                    )
                if chipped or not loading:
                    # Either the widget ticked our clicks and kept the photos
                    # (a board that does that is answered), or nothing faded
                    # within the grace window. Submit rather than paying for
                    # another round to be told the same thing.
                    _log(
                        f"[recaptcha-grid] round {round_index}: "
                        f"{'tiles chipped' if chipped else 'nothing loading'}; submitting."
                    )
                    should_submit = True
                    break
                with self._phase("fade-wait"):
                    self._wait_for_any_clicked_tile_loaded(
                        page, element, session, loading)
                continue

            _log(f"[recaptcha-grid] round {round_index}: unexpected action; re-solving.")

        if should_submit:
            frame = element.content_frame()
            if frame:
                verify = self._get_verify_button(frame)
                if verify:
                    _log("[recaptcha-grid] clicking Verify to submit.")
                    self._move_and_click(page, verify)
                    # The press IS an interaction, and saying so is load-bearing
                    # — the same rule `_solve_single` learned on
                    # prosopo_grid_3x3, in the other driver, on another day. A
                    # `done` round clicks no tile, so without this the one
                    # answer shape that submits and does nothing else reports
                    # having done nothing: the caller then takes the
                    # no-interaction wait instead of polling for the verdict
                    # (~1s of dead time on every static reCAPTCHA), and raises
                    # "performed no interactions" if the widget has not
                    # finished tearing down — on an answer correctly sent.
                    performed_action = True

        return performed_action, all_usage

    # ------------------------------------------------------------------
    # One pass over a rendered challenge
    # ------------------------------------------------------------------

    def _solve_single(
        self, page: Any, element: Any, retry_mode: Optional[str]
    ) -> Tuple[bool, List[Dict[str, Any]]]:
        src = None
        try:
            src = element.get_attribute("src")
        except Exception:
            pass
        src = src or ""

        # The vendor hint routes to the right pipeline. It matters: hCaptcha
        # click puzzles must never go through grid detection, because find_grid
        # false-positives on the header/footer bands.
        if "hcaptcha.com" in src:
            puzzle_source = "hcaptcha"
        elif "recaptcha/api2" in src:
            puzzle_source = "recaptcha"
        else:
            puzzle_source = "unknown"

        # Everything the answer might have to be delivered INTO — a text box, a
        # slider handle — is looked up against this, never against the page.
        # For the iframed vendors it is the challenge document; for the ones
        # that render into the host page (GeeTest, Yidun, BotDetect, …) it is
        # the widget element, whose subtree is the same boundary.
        scope = element.content_frame() or element

        # Does this puzzle want a STRING rather than a place to click? Only the
        # DOM can say. The picture cannot: BotDetect's warped code and
        # hCaptcha's "click the matching character" are the same genre of image
        # and want opposite answers. Restricted to `unknown` because neither
        # hCaptcha nor reCAPTCHA has ever served a typed challenge, so a match
        # inside one of their frames would be a false positive by definition.
        text_mode = (
            puzzle_source not in VENDORS_WITH_BESPOKE_HANDLING
            and self._answer_box(scope, element) is not None
        )
        if text_mode:
            _log("widget has a text box; solving as a distorted-text captcha")

        # hCaptcha REUSES the challenge iframe across rounds: after a submit it
        # briefly shows the previous round, then a spinner, then the next one.
        # Screenshotting any of those transitional frames feeds the model a
        # blank/stale grid it correctly calls "unsupported" — which used to abort
        # the whole solve on round 2.
        is_animated = False
        if puzzle_source == "hcaptcha" and "frame=challenge" in src:
            if self._last_submit_frame_hash:
                with self._phase("await-next-round"):
                    self._wait_for_change_since(element, self._last_submit_frame_hash)
                self._last_submit_frame_hash = None
            with self._phase("hcaptcha-images"):
                self._wait_for_hcaptcha_challenge_images(element)
            is_animated = self._settle_or_animated(element)
            # Those three waits total up to ~21s and none of them watches for
            # success, so the vendor's token routinely lands DURING them. Ask
            # once more before spending an inference: this is the last free
            # moment to notice the captcha is already accepted, and the
            # inference is the single most expensive thing in the loop.
            if self.is_captcha_solved(page):
                _log("solved while waiting for the next round; skipping inference.")
                return False, []
        elif puzzle_source not in VENDORS_WITH_BESPOKE_HANDLING:
            # Non-hCaptcha, non-reCAPTCHA widgets (GeeTest, Tencent, …). The settle
            # probe was never run for these, so an animated one — GeeTest's svg board
            # cycles its glyph set — was screenshotted mid-cycle and answered from
            # whatever single moment we happened to catch. reCAPTCHA is excluded on
            # purpose: it has its own readiness gate below, its grids are never
            # animated, and a second probe would only add latency to a path that
            # already works.
            is_animated = self._settle_or_animated(element)

        # Only the image-challenge frame holds a grid. Running grid detection on
        # the anchor checkbox just burns an 8s timeout before the click.
        is_recaptcha_challenge = puzzle_source == "recaptcha" and "recaptcha/api2/bframe" in src
        if is_recaptcha_challenge:
            with self._phase("grid-load"):
                self._wait_for_grid_cells_loaded(element)
            grid = self._get_grid_boxes(element)
            if grid and grid["size"] == 3:
                # Only a 3x3 ever refreshes its tiles in place, so only a 3x3
                # can need more than one round — and whether THIS one does is
                # decided inside the driver, by what the widget does with the
                # first click. A 4x4 never refreshes: it falls through to the
                # ordinary click-then-submit path below.
                element_box = element.bounding_box()
                if element_box:
                    return self._solve_recaptcha_grid(
                        page, element, retry_mode, grid, element_box
                    )

        shot = _tmp_png("captcha")
        performed_action = False
        slid = False
        placed = False
        clicked = False
        typed = False
        all_usage: List[Dict[str, Any]] = []
        keyframe_dir: Optional[str] = None
        have_shot = False
        try:
            if is_animated:
                # The freshness guard is deliberately SKIPPED here. It re-solves when
                # the frame changes during inference, and an animated challenge
                # changes by definition — every attempt would be judged stale and the
                # whole re-solve budget would burn without ever acting. The frame
                # number in the answer is the real guard: it names the state to act
                # in, and `_execute_click` waits for it.
                if self._animated_plan is not None:
                    # ONE burst, ONE inference, for as long as this board is up.
                    keyframes, keyframe_dir, actions, all_usage = self._animated_plan
                    _log("[animated] reusing the recorded answer — "
                         "same board, same screens")
                    reused_plan = True
                else:
                    reused_plan = False
                    with self._phase("burst"):
                        keyframes, keyframe_dir = self._record_keyframes(element)
                # THE RECORDING IS THE TEST. A burst of a widget that turns out
                # not to move slices to a single keyframe (`mode=static`), and
                # that frame is exactly the still this round would have taken
                # anyway — so a wrong guess about "is it animated" costs the
                # burst and nothing else, and cannot produce a wrong answer.
                #
                # This replaced a separate probe that watched the widget FIRST
                # and only recorded if it saw motion. That paid for the
                # observation twice over: up to `animated_probe_ms` to decide,
                # then the burst to act, and it still had to be long enough to
                # outlast the longest hold (8.2s measured on
                # number_with_highest_value_video) or it answered "static" from
                # inside one — which it did, live, burning 9s of a 45s budget to
                # reach the wrong conclusion about a puzzle that was animating.
                if len(keyframes) < 2:
                    _log("[animated] the recording shows one picture; "
                         "solving it as a still")
                    is_animated = False
                    # Reuse the frame we just recorded rather than taking
                    # another screenshot of the same motionless widget.
                    shutil.copyfile(keyframes[0], shot)
                    have_shot = True
                    # Not a cycling board after all, so there is no plan to keep.
                    self._discard_animated_plan()
                    shutil.rmtree(keyframe_dir, ignore_errors=True)
                    keyframe_dir = None
                elif not reused_plan:
                    with self._phase("inference"):
                        actions, all_usage = self._get_keyframe_solution(keyframes)
                    self._animated_plan = (keyframes, keyframe_dir,
                                           actions, all_usage)

            if not is_animated and self._should_speculate(puzzle_source, text_mode):
                # ASK AND WATCH AT THE SAME TIME. The JS twin carries the full
                # note; in short, the burst is the only thing that can tell a
                # cycling board from a still one AND is also the recording that
                # answers it, so it runs while the screenshot is being read.
                #
                # A still board sees one picture, keeps its answer and drops the
                # frames — identical behaviour and identical cost, because the
                # recording happened inside a wait we were already making. A
                # moving board's still answer is discarded UNREAD, because it
                # describes a screen that has gone.
                #
                # THE MODEL CALL GOES TO A WORKER THREAD AND THE SCREENSHOTS DO
                # NOT. Playwright's sync API is thread-affine, so every browser
                # call stays on this thread; the planner's HTTP request touches
                # no browser and is the only thing handed off. Exactly one model
                # call is in flight at a time, so the shared planner sees no
                # concurrency either.
                if not have_shot:
                    with self._phase("screenshot"):
                        self._screenshot(
                            element, shot,
                            timeout_ms=self.config.element_screenshot_timeout_ms)
                actions, all_usage, keyframe_dir = self._speculate(
                    element, shot, puzzle_source, retry_mode, text_mode)
                if keyframe_dir is not None:
                    is_animated = True

            elif not is_animated:
                if not have_shot:
                    with self._phase("screenshot"):
                        self._screenshot(
                            element, shot,
                            timeout_ms=self.config.element_screenshot_timeout_ms)
                with self._phase("inference"):
                    actions, all_usage = self._solve_frame_freshness_guarded(
                        element,
                        shot,
                        lambda image_path: self._get_solution(
                            image_path, puzzle_source, retry_mode, text_mode=text_mode
                        ),
                    )

            # The box turns NORMALISED coordinates into page ones, so only an
            # action that carries coordinates needs it. A "done" answer carries
            # none — it means "nothing left to click" — and several vendors
            # CLOSE the challenge the moment they accept it, so the widget is
            # legitimately gone by the time that answer arrives. Demanding a box
            # for it failed the solve at the moment it succeeded: measured
            # 2026-09-07 on gt4.geetest.com, 22 of 22 live GeeTest attempts, each
            # banked as the model being wrong. The JS port carries the identical
            # rule (rule 1c) and `answer_needs_element_box` is its twin.
            element_box = element.bounding_box()
            if not element_box and answer_needs_element_box(actions):
                raise CaptchaSolveError("could not get bounding box of captcha element")

            # Recorded at the moment of EXECUTION, which is what makes a repeat
            # mean something: this exact answer is about to be performed, so if
            # it matches the last one, the last one already ran and the page is
            # still asking the same question.
            self._note_answer(actions, retry_mode)
            _log(f"executing {len(actions)} action(s)")
            frame = element.content_frame()
            verify_button = None

            for raw_action in actions:
                self._check_deadline("action execution")
                action = _as_dict(raw_action)
                kind = action.get("action")
                if kind == "click":
                    bboxes = action.get("target_bounding_boxes") or (
                        [action["target_bounding_box"]]
                        if action.get("target_bounding_box")
                        else []
                    )
                    if not bboxes and not action.get("target_coordinates"):
                        _log("click action has no bboxes or coordinates; skipping")
                        continue
                    # On an animated challenge, hold each click until the widget is
                    # back in the state the model answered about. Per-click, not once
                    # per action: these puzzles keep cycling, so by the time click 2
                    # comes round the state has moved on again.
                    await_kf = action.get("await_keyframe")
                    if bboxes:
                        for bbox in bboxes:
                            one = {"target_bounding_box": bbox}
                            if await_kf:
                                self._click_when_frame_matches(
                                    page, element, one, element_box, await_kf)
                            else:
                                self._execute_click(page, one, element_box)
                            self._human.pause("between")
                    else:
                        self._execute_click(page, action, element_box)
                    performed_action = True
                    clicked = True
                elif kind == "drag" and not action.get("source_bounding_box"):
                    # No source — a puzzle-piece slider. What you grab is not
                    # what has to arrive, so this cannot go through
                    # _execute_drag: pressing the gap the model named and
                    # dragging from there picks up nothing at all.
                    if self._execute_slide(page, element, scope, action, element_box):
                        performed_action = True
                        slid = True
                elif kind == "drag":
                    # Wait on the SOURCE: the piece has to be there to be picked up.
                    # The destination is not gated — by the time the mouse arrives the
                    # animation has moved on regardless, and a drop is judged by where
                    # it lands, not by what the slot looked like on pickup.
                    await_kf = action.get("await_keyframe")
                    if await_kf and action.get("source_bounding_box"):
                        self._wait_for_keyframe(
                            element, await_kf, _bbox_center(action["source_bounding_box"])
                        )
                    self._execute_drag(page, action, element_box)
                    performed_action = True
                    placed = True
                elif kind == "type":
                    if self._execute_type(page, scope, action, element):
                        performed_action = True
                        typed = True
                elif kind == "wait":
                    duration = int(action.get("duration_ms") or 0)
                    if duration > 0:
                        _delay(duration)
                        performed_action = True


            # Resolve the widget's submit control AFTER the action loop, not
            # inside it. An empty plan — the correct answer to reCAPTCHA 3x3's
            # `none_present` variation, where nothing matches and the control
            # reads SKIP — never enters that loop, so the lookup never ran, the
            # press below found `verify_button` None, and the round aborted on
            # 'performed no interactions'. The finder was never the problem:
            # 'Skip' has always been in its list. It was simply never called for
            # the one answer shape that performs no other action.
            # `scope` when there is no vendor iframe. Eight vendors render
            # into the HOST PAGE — GeeTest, Yidun, Tencent, Yandex, Lemin,
            # Prosopo, MTCaptcha, BotDetect — so `content_frame()` is None
            # for all of them and the button was never even SEARCHED FOR,
            # while the text box and the slider handle it sits beside were
            # both found through `scope` a few lines above. Two containers
            # for two halves of one interaction.
            #
            # This used to be gated on `typed`, for fear of turning up the
            # submit of the FORM the captcha guards. `scope` is the widget
            # container and the xpaths are RELATIVE, so that button is out
            # of reach by construction; what the gate actually did was make
            # every non-typed inline puzzle unsubmittable. Measured on the
            # Tier 3 fixtures: 4 pairs aborting outright and 11 more types
            # burning all ten solve loops on a puzzle they had answered on
            # the first one. The press itself is still bounded by
            # `should_submit` below, which is where the hazard belongs.
            # RESOLVED, not travelled to. `_move_and_click` below moves to the
            # button itself and picks its own random point inside it, so moving
            # here first bought a second humanised trajectory that ended a few
            # dozen pixels from where the first one stopped, plus a second
            # bounded scroll-into-view. Two hops onto one button is slower AND
            # is not something a hand does. A round that decides not to submit
            # no longer walks to the control either.
            lookup = frame or (scope if not slid else None)
            if lookup is not None:
                verify_button = self._get_verify_button(lookup)

            # Submit policy: press the widget's own submit control whenever we
            # have put an ANSWER into it — a selection, a placed piece, a typed
            # code — or when we had nothing to do and want the round to advance.
            #
            # Two exclusions, and they are the whole rule:
            #
            #   a completed SLIDE has already submitted. Letting go of the handle
            #     is the gesture these puzzles grade; none of them ships a Verify
            #     button, so anything the generic finder turns up afterwards
            #     belongs to the host page, and pressing it would submit the form
            #     the captcha guards while the verdict is still in flight.
            #   a round that only WAITED has answered nothing. Submitting an
            #     empty board spends the attempt on a puzzle we were about to
            #     solve.
            #
            # hCaptcha and the reCAPTCHA 4x4 used to be named here as one-shot
            # special cases; they are ordinary click rounds and this covers them.
            # (reCAPTCHA 3x3 never reaches here — it returned above, to the
            # driver that owns its fade-and-re-round rounds.)
            #
            # A click round DID used to be excluded, on the reasoning that these
            # boards re-round and a half-made selection spends the attempt. They
            # do not: the ones that grade themselves mid-selection draw no submit
            # control at all, so `verify_button` is None and nothing is pressed
            # either way. What the exclusion actually bought was a whole extra
            # model call per puzzle, spent asking a board we had already answered
            # correctly whether it was `done` — 6.2 s of prosopo_grid_3x3's 13.8 s,
            # on a selection that scored 1.0 on the first call.
            answered = clicked or placed or typed
            should_submit = not slid and (answered or not performed_action)
            if should_submit and verify_button:
                _log(f"clicking Verify to submit ({puzzle_source}).")
                self._move_and_click(page, verify_button)
                # The press IS an interaction, and saying so is load-bearing:
                # the caller aborts a round that reports none, so submitting a
                # `done` answer and then returning False re-arms the very guard
                # this satisfies — the puzzle is sent and the solve gives up on
                # it one line later, which is what `prosopo_grid_3x3` did.
                performed_action = True
                # Snapshot at submit time so the NEXT attempt waits for the real
                # transition before treating whatever is on screen as fresh.
                self._last_submit_frame_hash = self._element_frame_hash(element)
        finally:
            _unlink(shot)
            # Only now: the wait gate re-reads the keyframe PNG on every poll, so
            # removing the directory any earlier would break the click it is gating.
            # ...and not while the PLAN still holds them: the next round
            # re-executes this same answer, and the gate reads these PNGs on
            # every poll.
            held = self._animated_plan[1] if self._animated_plan else None
            if keyframe_dir and keyframe_dir != held:
                shutil.rmtree(keyframe_dir, ignore_errors=True)

        return performed_action, all_usage

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def watch(self, page: Any, **options: Any) -> "CaptchaWatcher":
        """
        A watcher that solves captchas on `page` as they appear.

        Mirrors the TS `solver.watch(page)` in intent, but NOT in whether it
        blocks: this returns an idle watcher and you choose how to drive it,
        because a sync Playwright handle cannot be driven from a worker thread.

            solver.watch(page).run()              # blocking; hold the page clean
            w = solver.watch(page)
            while my_loop():
                w.poll_once()                     # cooperative; you own the cadence

        Options are `CaptchaWatcher`'s fields: `interval_ms`, `max_solves`,
        `error_backoff_ms`, `on_solved`, `on_error`. Nothing is injected into
        the page; see watcher.py.
        """
        from .watcher import CaptchaWatcher

        return CaptchaWatcher(solver=self, page=page, **options)

    def solve(self, page: Any) -> SolveResult:
        """
        Solve whatever captcha is on `page`.

        Returns a `SolveResult`; raises `NoCaptchaFoundError`,
        `UnsupportedChallengeError`, `AnimatedChallengeError`, or
        `CaptchaSolveError` on the failure modes named in each type.
        """
        start = time.monotonic() * 1000.0
        cumulative_usage: List[Dict[str, Any]] = []
        self._last_submit_frame_hash = None
        self._deadline_ms = start + self.config.overall_solve_timeout_ms
        self._human.reset(page)
        self._reset_animated_state()
        self._budget = PhaseBudget()

        # Mint one session id for the WHOLE solve, exactly as the TS driver does
        # per `solve()`. The planner turns it into `X-CK-Session`, which is what
        # lets the hosted gateway group this captcha's 1..N inference rounds into
        # a single billable attempt. Without it a multi-round dynamic 3x3 bills
        # as several attempts instead of one, so this is a billing-correctness
        # requirement, not telemetry.
        #
        # Restored rather than left set: a stale id leaking into the NEXT solve
        # would merge two separate captchas into one attempt — the opposite
        # error, and the one that under-bills.
        previous_session = os.environ.get(_SESSION_ENV)
        session_id = str(uuid.uuid4())
        os.environ[_SESSION_ENV] = session_id
        # The verdict this driver reports back, and it starts FALSE. A solve
        # that raises never sets it, which is the right default: the widget did
        # not accept, and the boards that made it raise are exactly the ones
        # worth keeping. See planner.report_outcome.
        solved_for_report = False
        try:
            result = self._solve_impl(page, start, cumulative_usage)
            # Attached HERE rather than at each of the seven `return
            # SolveResult(...)` sites, so a new early exit cannot forget it.
            result.phases = dict(self._budget.totals)
            result.phases["total"] = self._budget.elapsed_ms()
            solved_for_report = bool(getattr(result, "is_solved", False))
            return result
        finally:
            # Printed on the way out of every solve, success or failure — a
            # solve that FAILED is exactly the one whose time you want itemised.
            if timings_enabled() and self._budget is not None:
                print(self._budget.report(), file=sys.stderr)
            self._deadline_ms = None
            # BEFORE the env is restored, so the id reported is this solve's.
            # Best-effort and never raised out of a `finally` — an exception
            # here would replace the caller's real error, or their real result,
            # with one about telemetry.
            try:
                self._solver.planner.report_outcome(session_id, solved_for_report)
            except Exception as exc:  # noqa: BLE001
                _log(f"[outcome] could not report: {exc}")
            if previous_session is None:
                os.environ.pop(_SESSION_ENV, None)
            else:
                os.environ[_SESSION_ENV] = previous_session

    def _solve_impl(
        self,
        page: Any,
        start: float,
        cumulative_usage: List[Dict[str, Any]],
    ) -> SolveResult:
        cfg = self.config

        pending_retry_mode: Optional[str] = None
        already_retried_underselect = False
        unsupported_retries = 0
        stale_element_retries = 0
        has_interacted = False
        render_waits = 0
        # Strictly FEWER than the solve loops. A render wait consumes an
        # attempt, so at parity the loop runs out first and the branch below
        # never fires — "no interactive captcha widget", which is the correct
        # and benign answer for a reCAPTCHA v3 / invisible page, is reported
        # instead as "still detected after N solve loops". That is a hard error
        # for a caller who catches NoCaptchaFoundError to mean "nothing here",
        # and it appeared the moment max_solve_loops came down from 10 to 6.
        max_render_waits = min(6, cfg.max_solve_loops - 1)

        for attempt in range(1, cfg.max_solve_loops + 1):
            # Round 1 did not finish the challenge. Before spending another
            # still inference on it, let this round's settle check probe whether
            # the widget is animated — the settle rule alone cannot tell, and
            # answering an animated puzzle from one still is how a solve burns
            # all ten loops without ever being able to succeed.
            if attempt >= 2:
                self._arm_animated_probe()
            if (time.monotonic() * 1000.0) - start > cfg.overall_solve_timeout_ms:
                raise CaptchaSolveError(
                    f"captcha solve timed out after {cfg.overall_solve_timeout_ms}ms "
                    f"(attempt {attempt}/{cfg.max_solve_loops})"
                )

            # Ask the DOM whether we are DONE before asking what to solve next.
            # Only once we have interacted: before that a populated token is the
            # "already satisfied" case handled below, and this ordering would
            # skip the render-wait a fresh widget needs.
            #
            # `detect_captcha` is the expensive call — it settles pixels and
            # screenshots the element — and after the winning submit it returns
            # the challenge frame hCaptcha is TEARING DOWN. Solving that frame
            # cannot succeed; it just runs until the handle goes stale, which
            # measured 19-33s of dead time at the end of every run while the
            # answer had already been accepted. `is_captcha_solved` is a couple
            # of cheap DOM reads and is authoritative, so it goes first.
            if has_interacted and self.is_captcha_solved(page):
                _log("captcha reports solved; finishing.")
                return SolveResult(True, self._last_mouse, _aggregate(cumulative_usage))

            with self._phase("detect"):
                element = self.detect_captcha(page)
            if not element:
                # Two-stage. A null detection splits into two very different
                # cases and treating them alike is how you either hang on a
                # v3 page or give up on a slow-rendering widget.
                if has_interacted:
                    _log("no supported captcha remains after interaction; considering solved.")
                    return SolveResult(True, self._last_mouse, _aggregate(cumulative_usage))
                # Already satisfied before we touched anything — the widget is
                # present but the vendor has passed it (anchor checked / token
                # populated). Common with a good stealth browser: camoufox often
                # clears reCAPTCHA on the checkbox alone, with no challenge ever
                # shown. Without this, the render-wait branch below sits for
                # ~6s and then raises "no interactive captcha widget", turning
                # the BEST outcome into an exception the caller has to catch.
                if self.is_captcha_solved(page):
                    _log("captcha already satisfied; nothing to solve.")
                    return SolveResult(True, self._last_mouse, _aggregate(cumulative_usage))
                if self.has_interactive_widget_in_dom(page) and render_waits < max_render_waits:
                    render_waits += 1
                    _log(
                        f"widget in DOM but not yet rendered; waiting "
                        f"({render_waits}/{max_render_waits})."
                    )
                    _delay(800 + random.random() * 300)
                    continue
                raise NoCaptchaFoundError(self._no_widget_message(page))

            _log(f"--- captcha solve loop {attempt}/{cfg.max_solve_loops} ---")
            retry_mode_this_loop = pending_retry_mode
            pending_retry_mode = None

            try:
                did_interact, usage = self._solve_single(page, element, retry_mode_this_loop)
            except AnimatedChallengeError:
                raise
            except UnsupportedCaptchaError as unsupported:
                # A settled frame the model cannot solve is normally definitive.
                # BUT mid-solve, a transitional blank frame produces the same
                # verdict — that was the "solves round 1, dies on round 2" bug.
                if has_interacted and unsupported_retries < cfg.max_unsupported_resolves:
                    unsupported_retries += 1
                    current = self.detect_captcha(page)
                    with self._phase("settle"):
                        _settled = self._wait_for_element_settled(current)
                    if current and _settled == "animated":
                        # Used to be terminal. Now it just means the next round is an
                        # animated puzzle: retry the loop and `_solve_single` takes the
                        # recording path. `unsupported_retries` still bounds it, so a
                        # widget that is animated AND unsolvable cannot spin here.
                        if not cfg.video_solve_enabled:
                            raise AnimatedChallengeError(
                                "the challenge never settles and video_solve_enabled is off"
                            )
                        _log('"unsupported" mid-solve and the next round is animated; '
                             "retrying into the recording path.")
                        continue
                    _log(
                        f'"unsupported" mid-solve; settled and retrying '
                        f"({unsupported_retries}/{cfg.max_unsupported_resolves})."
                    )
                    continue
                # The solver's OWN message, not a guess about what it saw.
                # This used to substitute "likely an hCaptcha click/drag
                # puzzle" for every unsupported verdict, including the ones
                # that already said exactly what was wrong — "prompt
                # generation 1 has no distorted-text prompt", say, which names
                # both the cause and the fix. Reporting a wrong guess in place
                # of a right answer costs whoever reads the gate an
                # investigation, every time.
                raise UnsupportedChallengeError(
                    f"cannot solve this kind of captcha — {unsupported}"
                ) from unsupported
            except Exception as exc:
                # A stale/detached handle after a submit is a TRANSITION, not a
                # dead puzzle: hCaptcha swapped in the next round while we held
                # the old iframe. Only after interacting — a first-frame failure
                # is a genuine problem worth surfacing.
                message = str(exc)
                closed = bool(_CLOSED_TARGET_RE.search(message))
                if has_interacted and (closed or _STALE_HANDLE_RE.search(message)):
                    # ASK THE VENDOR FIRST. The handle most often went stale
                    # BECAUSE the answer was accepted: reCAPTCHA tears the
                    # challenge iframe down the instant it takes an answer, and
                    # hCaptcha swaps the frame out. Re-entering the pipeline
                    # here spends a whole round — detect, screenshot, infer — on
                    # a puzzle that no longer exists, and races the teardown.
                    # Headless usually won that race; headed usually lost it and
                    # surfaced `TargetClosedError` from whatever ran next.
                    #
                    # The top-of-loop check is not enough on its own: it runs
                    # only after the backoff, and it does DOM reads that throw
                    # on a target that is already gone.
                    try:
                        if self.is_captcha_solved(page):
                            _log("captcha reports solved; finishing.")
                            return SolveResult(
                                True, self._last_mouse, _aggregate(cumulative_usage)
                            )
                    except Exception:  # noqa: BLE001
                        # Can't consult the vendor — the page is gone. Fall
                        # through; `closed` decides between a named error and a
                        # retry, and neither wants this exception in its place.
                        pass

                    if closed:
                        raise PageClosedError(
                            "the page, context or browser closed mid-solve, after the "
                            "answer had been submitted but before the vendor's verdict "
                            "could be read — the solve may in fact have succeeded"
                        ) from exc

                    if stale_element_retries < cfg.max_stale_element_retries:
                        stale_element_retries += 1
                        _log(
                            f"stale challenge handle after submit; re-detecting next round "
                            f"({stale_element_retries}/{cfg.max_stale_element_retries})."
                        )
                        _delay(cfg.stale_element_backoff_ms)
                        continue
                raise

            has_interacted = has_interacted or did_interact

            # Give up on a solve that is repeating itself, rather than letting
            # the clock do it. The alternative is not "one more chance" — it is
            # the same click, again, until `overall_solve_timeout_ms`, which is
            # how a hopeless captcha came to cost 66s and how half of Tier 3's
            # wall-clock was being spent on attempts that could not succeed.
            if self._no_progress_rounds >= cfg.max_no_progress_rounds:
                raise CaptchaSolveError(
                    f"no progress: the model returned the same answer "
                    f"{self._no_progress_rounds + 1} times running and the challenge "
                    f"is still up (attempt {attempt}/{cfg.max_solve_loops})"
                )
            render_waits = 0
            cumulative_usage.extend(usage)

            # ONE wait after a round, polled, whichever kind of round it was.
            #
            # There used to be two: this poll when the round interacted, and a
            # flat `_delay(post_solve_delay_ms + jitter)` when it did not. The
            # flat sleep observed NOTHING — it ran to the end and only then
            # asked whether the widget was still there — so a round that had in
            # fact finished the captcha paid 1200-1500ms to find that out. The
            # poll below already asks exactly that question and answers it the
            # moment it becomes true.
            #
            # The two windows keep different LENGTHS, because different things
            # size them: `post_solve_outcome_timeout_ms` covers how late a
            # vendor's success signal can arrive (measured — see the constant),
            # while `post_solve_delay_ms` is the dwell a round that answered
            # nothing takes before deciding the page has settled.
            window_ms = (cfg.post_solve_outcome_timeout_ms if did_interact
                         else cfg.post_solve_delay_ms + random.random() * 300)
            # Poll for the vendor's SOLVED signal before re-entering the
            # pipeline. hCaptcha keeps the challenge visible for a couple of
            # seconds while verifying; without this the loop re-solves that
            # closing frame and burns ~18s. Only ever early-RETURNS on a
            # definitive signal, so it cannot loop.
            deadline = time.monotonic() * 1000.0 + window_ms
            solved = False
            widget_gone = 0
            _verdict_t0 = time.perf_counter()
            while time.monotonic() * 1000.0 < deadline:
                if self.is_captcha_solved(page):
                    solved = True
                    break
                # The eight inline vendors have no response token, so
                # `is_captcha_solved` — which reads only the hCaptcha and
                # reCAPTCHA anchors — can never fire for them and this loop
                # ran out its whole 2.5s budget on EVERY round, waiting for
                # a signal that cannot arrive. Measured on geetest_v4_slide:
                # 5.2s of a 12.3s solve, spent after the puzzle was already
                # answered, with the widget sitting there visibly solved.
                #
                # "The widget is gone" is the completion signal for those
                # vendors and is already the authority immediately after
                # this loop, so this only reaches the same verdict sooner —
                # confirmed over two polls so a frame caught mid-swap
                # between rounds cannot read as a solve.
                if self.detect_captcha(page) is None:
                    widget_gone += 1
                    if widget_gone >= 2:
                        solved = True
                        break
                else:
                    widget_gone = 0
                if self._is_challenge_freshly_rendered(page):
                    break  # next round is up; go solve it now
                _delay(cfg.post_solve_outcome_poll_ms)
            _verdict_ms = (time.perf_counter() - _verdict_t0) * 1000.0
            if self._budget is not None:
                self._budget.add(
                    "await-verdict" if did_interact else "post-submit-delay",
                    _verdict_ms)
            # How long a SUCCESS actually took to show itself. This is the
            # only number that can size `post_solve_outcome_timeout_ms`: the
            # window exists to catch a late success, so it needs to cover
            # the slowest real one and nothing beyond it. A round that ends
            # any other way spends the whole window by construction — there
            # is no signal on a wrong answer — so its duration says nothing
            # about how long the window ought to be.
            if solved:
                _log(f"[verdict] success signal arrived after {_verdict_ms:.0f}ms")
                return SolveResult(True, self._last_mouse, _aggregate(cumulative_usage))

            _banner = self._recaptcha_banner_kind(page)
            if self._banner_is_fatal_after_retry(_banner):
                if already_retried_underselect:
                    raise CaptchaSolveError(
                        "reCAPTCHA still showing the under-selection error after retry; "
                        "aborting (model unable to identify the missed tile)"
                    )
                _log("reCAPTCHA under-selection error; retrying with missed-tiles prompt.")
                pending_retry_mode = "missed-tiles"
                already_retried_underselect = True

            if not self.detect_captcha(page):
                return SolveResult(True, self._last_mouse, _aggregate(cumulative_usage))

            if not did_interact:
                raise CaptchaSolveError(
                    "captcha still detected but the solver performed no interactions; "
                    "aborting to avoid an infinite loop"
                )

        raise CaptchaSolveError(
            f"captcha still detected after {cfg.max_solve_loops} solve loops"
        )


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _bbox_center(bbox: Sequence[float]) -> Tuple[float, float]:
    """Centre of an [x1, y1, x2, y2] 0–1 box, as the (x, y) 0–1 point the keyframe
    wait gate compares around. The solver builds these boxes as a small square
    around the model's point, so the centre recovers that point exactly."""
    x1, y1, x2, y2 = (float(v) for v in bbox)
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _round_pts(value: Any, places: int = 3) -> Any:
    """Coordinates rounded, for comparing two answers for sameness.

    Rounded rather than compared exactly because the same tile chosen twice can
    differ in the last float digit after the normalise/clamp round-trip, and a
    repeat that reads as "different" is a repeat that costs a round.
    """
    if isinstance(value, (int, float)):
        return round(float(value), places)
    if isinstance(value, (list, tuple)):
        return [_round_pts(v, places) for v in value]
    return value


def _as_dict(action: Any) -> Dict[str, Any]:
    """Actions arrive as pydantic models in-process, or dicts from JSON."""
    if isinstance(action, dict):
        return action
    for method in ("model_dump", "dict"):
        if hasattr(action, method):
            try:
                return getattr(action, method)()
            except Exception:
                pass
    return {"action": getattr(action, "action", None)}


def _aggregate(usage: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Mirror of `aggregateTokenUsage`: one summed row, plus the raw rounds."""
    if not usage:
        return []
    total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for entry in usage:
        for key in total:
            total[key] += int(entry.get(key) or 0)
    return [{"rounds": len(usage), **total}]


def _read_png_dimensions(path: str) -> Optional[Tuple[int, int]]:
    """
    Width/height from the IHDR chunk, so no image-size dependency is needed.
    PNG signature is 8 bytes, IHDR length+type another 8, then two big-endian
    uint32s at offsets 16 and 20.
    """
    try:
        with open(path, "rb") as handle:
            header = handle.read(24)
        if len(header) < 24 or header[1:4] != b"PNG":
            return None
        width = int.from_bytes(header[16:20], "big")
        height = int.from_bytes(header[20:24], "big")
        return (width, height) if width and height else None
    except OSError:
        return None


def solve_captcha_on_page(page: Any, **kwargs: Any) -> SolveResult:
    """
    One-shot convenience wrapper mirroring the TS `new CaptchaKrakenSolver().solve(page)`.

    Synchronous only. The async Playwright API needs a parallel implementation
    (`await` at every call site) rather than a wrapper — sync Playwright handles
    cannot be driven from inside an event loop, so `asyncio.to_thread` would not
    save this. Async support is not yet written.
    """
    return PageSolver(**kwargs).solve(page)
