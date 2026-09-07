"""Reduce a recorded captcha clip to the few frames the model is shown.

VERBATIM PORT. Everything below the docstring is byte-identical to
`src/video/keyframes.py` in the CaptchaKrakenFinetune repo, and
`scripts/check_prompt_parity.py` there fails the build if the two ever diverge.

That is not tidiness, it is a correctness requirement. The model is trained on
keyframes cut by that code and answers with a frame NUMBER indexing into them. If
this copy sliced a live recording differently — a cycle collapsing here but not
there, a different keyframe budget — the number would name a picture that does not
exist, and the driver would wait for a state the page never reaches. Read that
file's docstring for the algorithm and the reasoning behind every threshold.

In this repo the module has one extra job: `region_box` / `region_diff_ratio` are
also the solver's wait-for-state gate. The driver holds the mouse until the live
page's neighbourhood around the click point matches the chosen keyframe's, using
the same metric and the same box the label was chosen with.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

# ── extraction parameters ───────────────────────────────────────────────────
# Defaults are tuned against the collector's burst geometry (10 fps; 4 s by
# default, but `Source.burst_duration_ms` raises it per vendor so a burst covers
# a full animation cycle, so clip length varies). They are all
# fractions or frame counts, so they survive a change of clip length; only
# MIN_HOLD_FRAMES is absolute, and it is deliberately a wall-clock idea (~0.3 s).

DEFAULT_MAX_KEYFRAMES = 6

# Two frames differing in fewer than this fraction of pixels are "the same
# picture". A false "same" merges two distinct states, which loses an answer, so
# this is the most consequential number in the file.
#
# MEASURED, not guessed. Over all 20 real clips in cleanSamples/test/raw, the
# consecutive-frame differences fall into two clusters with a 64x gap:
#
#   same picture (compression + antialiasing)  <= 0.000067   (0.067 per mille)
#   smallest genuine state change               >= 0.004282  (4.28  per mille)
#
# 0.001 sits inside that gap — 15x above the noise floor, 4.3x below the faintest
# real swap — and the slicing is identical anywhere from 0.0005 to 0.002, so this
# is a plateau rather than a knife edge.
#
# It used to be 0.005, which was ABOVE the smallest real state change. That is not
# a tuning nicety: a 340x384 geetest icon board swaps one tile and changes ~0.5% of
# the frame, so half of its real transitions read as "the same picture". One clip
# (ssqr8) merged to a single state and was declared `static` — one still for a clip
# with three pictures — and another (zxpan) emitted 2 of its 3. The old value was
# calibrated against 64x64 fixtures whose patch covers 14% of the frame; a real
# captcha animates a sprite covering under 1% of the widget.
DEFAULT_STEADY_RATIO = 0.001

# Two states must differ by at least this much for a clip to be called a `cycle`
# and COLLAPSED to one frame per state. Deliberately NOT scaled down alongside
# `steady_ratio`: the two thresholds guard opposite risks. `steady_ratio` decides
# "is this the same picture", where being too generous LOSES a state — so it tracks
# the measured noise floor. This one licenses the aggressive collapse, where being
# too generous discards the middle of a clip, so it stays conservative. Real state
# separations top out around 0.007, well under this, so `cycle` does not fire on any
# real footage collected so far; `even` + dedup reaches the same answer for a rested
# clip without having to decide whether it recurs. Lowering this to reach cycle mode
# would buy nothing and risk the one failure that cannot be detected downstream.
DEFAULT_DISTINCT_RATIO = 0.02

# A hold has to last long enough to be a deliberate resting state, not a frame
# that happened to repeat mid-animation. 3 frames @ 10 fps ≈ 0.3 s.
DEFAULT_MIN_HOLD_FRAMES = 3

# Holds must account for at least this much of the clip. A 3-state cycle with
# fast swaps is ~0.9 held; a cross-fade is ~0.0. The gap is wide, so this
# threshold is not delicate.
DEFAULT_MIN_STEADY_COVERAGE = 0.5

# The pixel-difference threshold inside the similarity metric. Not tunable per
# call on purpose — it is the one number shared with every other movement check
# in the project (see module docstring).
_PIXEL_DELTA = 30

# Half-width of the neighbourhood around an action point, as a fraction of the
# image, used both to CHOOSE which keyframe a label belongs to and — in the live
# solver — to decide the page now looks like that keyframe. One constant on
# purpose: "the frame this answer is valid in" and "the moment it is safe to
# click" have to be the same question, or the solver waits for a state the label
# never described. 0.06 ≈ a 12%-of-frame box, big enough to contain a sprite and
# small enough that unrelated motion elsewhere doesn't veto the click.
MATCH_REGION_HALF = 0.06

# How different the region may be and still count as "the page looks like this
# keyframe". Deliberately looser than DEFAULT_STEADY_RATIO: the live page carries
# antialiasing and cursor artefacts a rendered keyframe does not, and a gate that
# never opens is worse than one that opens a frame early.
MATCH_REGION_TOLERANCE = 0.05

MANIFEST_NAME = "keyframes.json"
KEYFRAME_DIR_NAME = "keyframes"


@dataclass(frozen=True)
class KeyframeParams:
    """Everything that decides how a clip is sliced. Recorded in the manifest so
    a keyframe set on disk can be traced back to the settings that produced it —
    a threshold change is otherwise invisible and silently mixes two slicings
    into one training set."""

    max_keyframes: int = DEFAULT_MAX_KEYFRAMES
    steady_ratio: float = DEFAULT_STEADY_RATIO
    distinct_ratio: float = DEFAULT_DISTINCT_RATIO
    min_hold_frames: int = DEFAULT_MIN_HOLD_FRAMES
    min_steady_coverage: float = DEFAULT_MIN_STEADY_COVERAGE
    # Never hand the model two copies of one picture (see `_distinct_indices`).
    # Present as a param, not a hard-coded behaviour, for the reason every other
    # number here is: it lands in the manifest, so a set cut before this existed
    # has params that no longer match and gets re-cut instead of silently reused.
    dedupe: bool = True

    def as_dict(self) -> Dict[str, Any]:
        return {
            "max_keyframes": self.max_keyframes,
            "steady_ratio": self.steady_ratio,
            "distinct_ratio": self.distinct_ratio,
            "min_hold_frames": self.min_hold_frames,
            "min_steady_coverage": self.min_steady_coverage,
            "dedupe": self.dedupe,
        }


@dataclass
class Keyframe:
    """One image the model will be shown.

    `number` is 1-based and is what the model returns as `"frame"`. It is NOT the
    index in the source clip: a 3-state cycle emits frames 1, 2, 3 even if their
    source indices are 4, 17 and 29. `source_index` keeps the provenance so the
    generator's ground-truth frame can be snapped onto a keyframe number and so a
    human labeller can scrub back to the right moment.
    """

    number: int
    source_index: int
    timestamp_ms: float
    image: Optional[np.ndarray] = None  # BGR uint8; None once written to disk


@dataclass
class KeyframeSet:
    mode: str  # "cycle" | "even" | "static"
    keyframes: List[Keyframe]
    source_frames: int
    fps: float
    params: KeyframeParams = field(default_factory=KeyframeParams)
    # cycle mode only: per-source-frame state id (0-based, aligned with
    # `keyframes`). Empty for the other modes. Lets a caller answer "which
    # keyframe was the clip showing at source frame i?" without re-diffing.
    frame_states: List[int] = field(default_factory=list)
    # How many distinct steady screens the clip sits on; 0 if it is not that
    # kind of clip. INDEPENDENT OF `mode`, and the difference is the whole
    # point: `mode` says what the slicer could PROVE about recurrence from one
    # burst, this says what the board looks like. A three-screen captcha whose
    # loop is longer than the recording is `even` with `steady_screens == 3`.
    # The live driver reads this to decide whether waiting for a named screen
    # is meaningful; reading `mode` for that put the wait off on 100% of real
    # animated captchas. See `steady_screens()`.
    steady_screens: int = 0

    def __len__(self) -> int:
        return len(self.keyframes)

    def number_for_source_index(self, source_index: int) -> int:
        """The keyframe a given source frame belongs to (1-based), by POSITION.

        In cycle mode this is exact — the frame's own state. Otherwise it is the
        nearest keyframe in time. Prefer `number_for_frame` whenever the pixels are
        available: nearest-in-time is not the same question the solver asks, and a
        transition frame (state -1) has no state of its own to fall back on.
        """
        if not self.keyframes:
            raise ValueError("keyframe set is empty")
        if self.mode == "cycle" and 0 <= source_index < len(self.frame_states):
            state = self.frame_states[source_index]
            if state >= 0:
                return state + 1
        return min(
            self.keyframes, key=lambda k: abs(k.source_index - source_index)
        ).number

    def number_for_frame(
        self,
        frame: np.ndarray,
        *,
        point_norm: Optional[Tuple[float, float]] = None,
        source_index: Optional[int] = None,
    ) -> int:
        """The keyframe that LOOKS most like `frame`, 1-based.

        This — not nearest-in-time — is how a label's frame number is chosen,
        because it asks the same question the live solver asks. At solve time the
        driver holds the mouse until the page's neighbourhood around the click
        point matches the chosen keyframe; so the frame we write into the label
        must be the one whose *appearance* matches the moment the answer was read
        off, or the driver ends up waiting for a state where the target isn't
        there.

        `point_norm` (x, y in 0..1) restricts the comparison to the neighbourhood
        of the action point, again matching the solver. Without it the whole frame
        is compared, which on a multi-sprite puzzle is dominated by sprites that
        have nothing to do with the answer.

        `source_index` only breaks ties, keeping the choice stable and preferring
        the temporally closest of two equally-similar keyframes.
        """
        if not self.keyframes:
            raise ValueError("keyframe set is empty")
        box = region_box(frame.shape[1::-1], point_norm) if point_norm else None

        def cost(kf: Keyframe) -> Tuple[float, int]:
            if kf.image is None:
                raise ValueError(f"keyframe {kf.number} has no image to compare")
            d = region_diff_ratio(kf.image, frame, box)
            tie = abs(kf.source_index - source_index) if source_index is not None else 0
            return (d, tie)

        return min(self.keyframes, key=cost).number

    def manifest(self, *, stem: str, filenames: Sequence[str]) -> Dict[str, Any]:
        return {
            "stem": stem,
            "mode": self.mode,
            "steady_screens": self.steady_screens,
            "fps": self.fps,
            "source_frames": self.source_frames,
            "params": self.params.as_dict(),
            "keyframes": [
                {
                    "number": kf.number,
                    "file": name,
                    "source_index": kf.source_index,
                    "timestamp_ms": round(kf.timestamp_ms, 1),
                }
                for kf, name in zip(self.keyframes, filenames)
            ],
        }


# ── similarity ──────────────────────────────────────────────────────────────


def region_box(
    size_wh: Tuple[int, int],
    point_norm: Tuple[float, float],
    half: float = MATCH_REGION_HALF,
) -> Tuple[int, int, int, int]:
    """`(x1, y1, x2, y2)` pixel box of half-width `half` around a 0..1 point.

    Clamped to the image and never empty: a point on the very edge of the frame
    (a target flush against the border) would otherwise produce a zero-area box,
    and a comparison over no pixels reads as a perfect match — the gate would open
    immediately on any state at all.
    """
    w, h = int(size_wh[0]), int(size_wh[1])
    cx, cy = float(point_norm[0]) * w, float(point_norm[1]) * h
    rx, ry = max(1.0, half * w), max(1.0, half * h)
    x1 = max(0, min(w - 1, int(round(cx - rx))))
    y1 = max(0, min(h - 1, int(round(cy - ry))))
    x2 = max(x1 + 1, min(w, int(round(cx + rx))))
    y2 = max(y1 + 1, min(h, int(round(cy + ry))))
    return x1, y1, x2, y2


def region_diff_ratio(
    a: np.ndarray, b: np.ndarray, box: Optional[Tuple[int, int, int, int]] = None
) -> float:
    """`frame_diff_ratio` over a sub-rectangle (the whole frame when `box` is None).

    Mismatched shapes short-circuit before cropping, so a resized live screenshot
    reads as "completely different" rather than as a crop of a different geometry.
    """
    if a is None or b is None or a.shape != b.shape:
        return 1.0
    if box is None:
        return frame_diff_ratio(a, b)
    x1, y1, x2, y2 = box
    return frame_diff_ratio(a[y1:y2, x1:x2], b[y1:y2, x1:x2])


def frame_diff_ratio(a: np.ndarray, b: np.ndarray) -> float:
    """Fraction of pixels that differ by more than 30 in any channel.

    Mismatched shapes short-circuit to 1.0 ("completely different"), matching
    `_collect_common.max_movement_ratio`. Do not swap in a perceptual metric
    without changing the solver's wait gate too — see module docstring.
    """
    if a is None or b is None:
        return 1.0
    if a.shape != b.shape:
        return 1.0
    diff = cv2.absdiff(a, b)
    if diff.ndim == 3:
        diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, thr = cv2.threshold(diff, _PIXEL_DELTA, 255, cv2.THRESH_BINARY)
    h, w = thr.shape[:2]
    if h * w == 0:
        return 1.0
    return cv2.countNonZero(thr) / float(h * w)


# ── decoding ────────────────────────────────────────────────────────────────


def decode_video(path: str | os.PathLike) -> Tuple[List[np.ndarray], float]:
    """Every frame of a clip, in order, plus its fps.

    Reads with `cv2.VideoCapture` rather than trusting CAP_PROP_FRAME_COUNT,
    which lies on the `mp4v` files this project writes (the muxer puts `moov`
    last and the count comes back as 0 often enough to matter).
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"could not open clip: {path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frames: List[np.ndarray] = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        cap.release()
    if not frames:
        raise ValueError(f"clip decoded to zero frames: {path}")
    if not fps or fps <= 0 or fps > 240:
        fps = 10.0  # the collector's BURST_FPS; a sane floor beats a NaN
    return frames, fps


# ── extraction ──────────────────────────────────────────────────────────────


def _anchor_runs(frames: Sequence[np.ndarray], steady_ratio: float) -> List[Tuple[int, int]]:
    """Partition the clip into maximal runs that all match the run's FIRST frame.

    Anchored on the first frame rather than chained frame-to-frame on purpose: a
    slow pan has every consecutive diff under the threshold while the endpoints
    are nothing alike, so a chained walk would call the whole pan one steady
    state. Comparing against the anchor cannot drift.
    """
    runs: List[Tuple[int, int]] = []
    i = 0
    n = len(frames)
    while i < n:
        j = i
        while j + 1 < n and frame_diff_ratio(frames[i], frames[j + 1]) <= steady_ratio:
            j += 1
        runs.append((i, j))
        i = j + 1
    return runs


def _medoid(frames: Sequence[np.ndarray], indices: Sequence[int]) -> int:
    """The index in `indices` whose frame is most typical of the group.

    For a hold this is nearly arbitrary (the frames are near-identical by
    construction) but it keeps a half-faded edge frame from representing the
    state when a run happens to straddle a transition boundary.
    """
    if len(indices) <= 2:
        return indices[0]
    best_idx, best_cost = indices[0], float("inf")
    for a in indices:
        cost = sum(frame_diff_ratio(frames[a], frames[b]) for b in indices if b != a)
        if cost < best_cost:
            best_idx, best_cost = a, cost
    return best_idx


def _even_indices(n: int, count: int) -> List[int]:
    """`count` indices spread across 0..n-1, endpoints included, deduped."""
    if n <= count:
        return list(range(n))
    if count <= 1:
        return [0]
    out = [int(round(k * (n - 1) / (count - 1))) for k in range(count)]
    seen: List[int] = []
    for i in out:
        if i not in seen:
            seen.append(i)
    return seen


def _distinct_indices(
    frames: Sequence[np.ndarray], candidates: Sequence[int], params: KeyframeParams
) -> List[int]:
    """Drop candidates that repeat a picture already kept, then spend the freed
    budget on frames that are genuinely new.

    WHY THIS IS NEEDED EVEN THOUGH `even` MODE IS ALREADY "CORRECT". Even spacing
    asks a question about TIME, and a clip that rests answers it with duplicates: a
    geetest icon board holding three states for 12/20/8 frames, sampled at
    0/8/16/23/31/39, returns state A twice and state B three times. Six images, three
    pictures. That is not merely wasteful of context — it makes the `frame` label
    AMBIGUOUS. If frames 1 and 2 are the same picture, an answer naming either is
    equally right, but grading compares against one number and the solver waits for
    one of them. A duplicate is a silently unanswerable question.

    Why this and not a better `cycle` detector. Distinguishing "three discrete states
    shown once" from "smooth motion coarsely quantised" is not decidable from
    `frame_diff_ratio`: the metric saturates once a sprite clears its old position,
    so both shapes produce equal-distance, high-coverage "holds" and the pairwise
    geometry that would separate them is destroyed (a 2px/frame pan and a real
    3-state board both score ~0.55 on every path statistic tried). Deduplication
    sidesteps the question entirely, because it is safe in the one direction that
    matters: it only ever removes a picture that is ALREADY in the set, so no frame
    carrying an answer can be lost. Under-sampling — the failure this module exists
    to prevent — is impossible here by construction.

    The backfill exists because dedup frees budget and a clip can have more states
    than even spacing happened to land on. `symza` holds nine; sampling six and
    deduplicating leaves four, so four of its pictures were never offered. Walking
    the clip in order for anything distinct from everything kept spends the returned
    slots on new pictures instead of retiring them.

    Returns indices in ascending order, so keyframe numbers stay a time order.
    """
    kept: List[int] = []

    def is_new(i: int) -> bool:
        return not any(
            frame_diff_ratio(frames[i], frames[k]) <= params.steady_ratio for k in kept
        )

    for i in candidates:
        if is_new(i):
            kept.append(i)
    if len(kept) < params.max_keyframes:
        for i in range(len(frames)):
            if len(kept) >= params.max_keyframes:
                break
            if i not in kept and is_new(i):
                kept.append(i)
    return sorted(kept)


def _steady_screens(
    frames: Sequence[np.ndarray], params: KeyframeParams
) -> Optional[Tuple[List[Tuple[int, int]], List[int], List[int]]]:
    """`(holds, state_medoid_indices, state_per_hold)` if the clip is a set of
    steady held pictures, else None.

    Split out of `_detect_cycle` because the answer is wanted TWICE and the two
    questions are not the same one:

      - "is this a cycle?" — needs a state to be seen coming BACK, which takes
        more than one pass through the loop.
      - "does this board sit on a few stable screens?" — needs only one pass,
        and is what the live driver has to know in order to decide whether
        waiting for a particular screen is meaningful.

    Answering only the first left the second unasked. See `steady_screens`.
    """
    n = len(frames)
    runs = _anchor_runs(frames, params.steady_ratio)
    holds = [(s, e) for (s, e) in runs if (e - s + 1) >= params.min_hold_frames]
    if not holds:
        return None

    covered = sum(e - s + 1 for s, e in holds)
    if covered / float(n) < params.min_steady_coverage:
        return None

    # Merge holds that show the same picture. Order of first appearance is the
    # order the model will see, so a cycle that revisits state 1 does not mint a
    # fourth image for it.
    state_reps: List[int] = []          # medoid source index per state
    hold_state: List[int] = []          # state id per hold, parallel to `holds`
    for s, e in holds:
        rep = _medoid(frames, list(range(s, e + 1)))
        for k, existing in enumerate(state_reps):
            if frame_diff_ratio(frames[rep], frames[existing]) <= params.steady_ratio:
                hold_state.append(k)
                break
        else:
            if len(state_reps) >= params.max_keyframes:
                return None  # too many distinct holds to be a tidy cycle
            state_reps.append(rep)
            hold_state.append(len(state_reps) - 1)

    if len(state_reps) < 1:
        return None

    # NO `distinct_ratio` HERE, deliberately. The merge above already separated
    # these states by more than `steady_ratio`, which is the measured noise
    # floor — that is the whole of "are these different pictures".
    #
    # `distinct_ratio` (0.02) is twenty times coarser and exists to license
    # `cycle`'s COLLAPSE to one frame per state, where being too generous throws
    # away the middle of a clip. Applying it here instead answers a question
    # nobody asked and answers it wrongly: measured on a live GeeTest svg burst,
    # the three screens are separated by 0.0056 and 0.0050 — thin line art, a
    # few glyph strokes change — so a 0.02 bar reports a board with three
    # obvious screens as having none. keyframes.py's own comment already said
    # so ("real state separations top out around 0.007, well under this") and
    # read it as evidence the check was harmless.
    return holds, state_reps, hold_state


def steady_screens(frames: Sequence[np.ndarray], params: KeyframeParams) -> int:
    """How many distinct steady screens this clip sits on. 0 if it does not.

    THE LIVE DRIVER'S QUESTION, and it is not `mode`. A board that advances
    through three held pictures gives a burst that contains ONE pass, so
    `_detect_cycle` cannot prove it recurs and the clip is sliced `even` — which
    is right for slicing (dedup reaches the same stills) and was being read by
    the driver as "nothing recurs, do not wait for a screen". Every real
    geetest_v4_svg clip is exactly this shape: 2-3 holds covering 95-100% of the
    burst, mutually distinct, and the page absolutely does come back around.

    A one-way animation — the case the no-wait path exists for — is nearly all
    transition, so it has no qualifying holds and this returns 0.
    """
    got = _steady_screens(frames, params)
    return len(got[1]) if got else 0


def _detect_cycle(
    frames: Sequence[np.ndarray], params: KeyframeParams
) -> Optional[Tuple[List[int], List[int]]]:
    """`(state_medoid_indices, per_frame_state)` if the clip is a cycle of steady
    pictures, else None.

    `per_frame_state[i]` is the 0-based state of source frame i, or -1 for a
    transition frame that belongs to no hold.
    """
    n = len(frames)
    got = _steady_screens(frames, params)
    if got is None:
        return None
    holds, state_reps, hold_state = got

    # A cycle REVISITS. More holds than states means some picture came back, which
    # is what "the video is just a shift between N steady images" describes. Equal
    # counts mean every hold was a new picture — a one-way progression (a slow
    # cross-fade reads exactly like this, see module docstring), where collapsing
    # to one frame per hold would discard the frames carrying the answer.
    #
    # The cost of this rule is a genuine cycle whose period is long enough that
    # only ONE pass fits in the recording — every real geetest_v4_svg clip
    # is exactly this — so it falls back to `even`. That used to mean 6 frames of a
    # 3-picture clip; since `even` deduplicates it now means 3, which is the same
    # answer `cycle` would have given, reached without having to decide whether the
    # clip recurs. The rule stays because dropping it WOULD be unsafe: a slow
    # one-way fade also decomposes into 2 long holds at full coverage (verified),
    # and collapsing that to 2 frames discards the middle of the clip. Dedup cannot
    # make that mistake; a permissive cycle detector can.
    if len(holds) <= len(state_reps) and len(state_reps) > 1:
        return None

    # Every pair of states must be clearly different before the clip is
    # COLLAPSED to one frame per state — that collapse is what this threshold
    # licenses, and being too generous with it discards the middle of a clip.
    # `steady_screens` deliberately does not apply it; see `_steady_screens`.
    for a in range(len(state_reps)):
        for b in range(a + 1, len(state_reps)):
            if frame_diff_ratio(frames[state_reps[a]], frames[state_reps[b]]) < params.distinct_ratio:
                return None

    per_frame = [-1] * n
    for (s, e), state in zip(holds, hold_state):
        for i in range(s, e + 1):
            per_frame[i] = state
    return state_reps, per_frame


def _drop_smeared(
    frames: Sequence[np.ndarray], indices: Sequence[int], params: KeyframeParams
) -> List[int]:
    """Drop an index caught BETWEEN two holds, when both of them are already
    represented in the cut.

    These clips hold a picture, swap, hold the next. An evenly spaced sample can
    land on the instant of the swap — a board part-way between two states, which
    is a picture the puzzle never actually shows. If the model already has the
    board before the swap AND the board after it, that frame adds nothing but a
    smear; if either endpoint is missing, it is the only thing standing in for a
    state and must stay.

    THE MACHINERY ALREADY EXISTED AND THIS PATH NEVER ASKED IT. `_anchor_runs`
    plus `min_hold_frames` is how `_detect_cycle` separates holds from
    transitions (it labels the latter -1 in `frame_states`), and
    `min_hold_frames` had exactly one reader — inside `_detect_cycle`. On the
    `even` path the parameter was declared, serialised into every manifest, and
    inert.

    WHY THE TEST IS "BOTH NEIGHBOURING HOLDS ARE KEPT" AND NOT A PIXEL DIFF.
    Two earlier versions were measured against both trees and both were wrong:

      * *Snap to the nearest held frame* turned `[0, 16, 39]` into `[0, 16]` on
        six eval clips, dropping a SCREEN. A page that never reaches the model is
        unanswerable whenever the target lived on it.
      * *Drop when some other still is within `distinct_ratio`* cut
        `hcaptcha_tile_flip_video` from 6 stills to 3, because two boards of that
        puzzle differ by ONE TILE — well inside `distinct_ratio` — so a diff
        threshold cannot tell "the same picture again" from "the next board".
        EXPECTED_SLICING says it directly: fewer than 6 "means dedup merged two
        boards that differ only by the tile mid-flip, and that tile is frequently
        the answer".

    Anchoring on the HOLDS either side asks the question that actually matters —
    does the model already see both ends of this swap — and needs no threshold of
    its own.

    A clip with no holds at all (a rotation, a continuous fade, anything in
    permanent motion) has no transitions to speak of and is returned untouched.
    """
    runs = _anchor_runs(frames, params.steady_ratio)
    holds = [(s, e) for (s, e) in runs if (e - s + 1) >= params.min_hold_frames]
    if not holds:
        return list(indices)               # nothing holds; nothing is a smear

    # Holds showing the same picture are ONE state — the same merge
    # `_detect_cycle` does, at the same threshold. It matters here because a
    # cycle revisits: on a real `icon_2x2` clip the hold at 53..74 is the
    # opening board coming back, so the still at index 0 already covers it.
    # Counting holds instead of states kept a smear whose neighbouring picture
    # was on screen the whole time.
    state_of_hold: List[int] = []
    reps: List[int] = []
    for start, end in holds:
        rep = _medoid(frames, list(range(start, end + 1)))
        for k, existing in enumerate(reps):
            if frame_diff_ratio(frames[rep], frames[existing]) <= params.steady_ratio:
                state_of_hold.append(k)
                break
        else:
            reps.append(rep)
            state_of_hold.append(len(reps) - 1)

    def hold_of(i: int) -> Optional[int]:
        for k, (s, e) in enumerate(holds):
            if s <= i <= e:
                return k
        return None

    covered = {state_of_hold[h] for h in (hold_of(i) for i in indices)
               if h is not None}
    kept: List[int] = []
    for i in indices:
        if hold_of(i) is not None:
            kept.append(i)
            continue
        before = max((k for k, (s, e) in enumerate(holds) if e < i), default=None)
        after = min((k for k, (s, e) in enumerate(holds) if s > i), default=None)
        # Only a swap BETWEEN two kept holds is redundant. A transition at the
        # very start or end of a clip has an endpoint that was never recorded,
        # so it is evidence of a state and stays.
        if before is not None and after is not None \
                and state_of_hold[before] in covered \
                and state_of_hold[after] in covered:
            continue
        kept.append(i)
    return kept


def extract_keyframes(
    frames: Sequence[np.ndarray],
    *,
    fps: float = 10.0,
    params: Optional[KeyframeParams] = None,
) -> KeyframeSet:
    """Slice a decoded clip into the frames the model will be shown.

    `frames` are BGR uint8 arrays in capture order (what `cv2` hands back and
    what the collector's burst produces).
    """
    if not frames:
        raise ValueError("no frames to extract keyframes from")
    p = params or KeyframeParams()
    n = len(frames)

    def _ms(i: int) -> float:
        return (i / fps) * 1000.0 if fps > 0 else 0.0

    cycle = _detect_cycle(frames, p)
    if cycle is not None:
        reps, per_frame = cycle
        mode = "static" if len(reps) == 1 else "cycle"
        return KeyframeSet(
            steady_screens=len(reps),
            mode=mode,
            keyframes=[
                Keyframe(number=k + 1, source_index=idx, timestamp_ms=_ms(idx),
                         image=frames[idx])
                for k, idx in enumerate(reps)
            ],
            source_frames=n,
            fps=fps,
            params=p,
            frame_states=per_frame,
        )

    indices = _even_indices(n, p.max_keyframes)
    # BEFORE dedup, so a smear does not spend budget a real state could use...
    indices = _drop_smeared(frames, indices, p)
    if p.dedupe:
        # Applied to `even` only. `cycle` states are distinct by construction (the
        # merge in `_detect_cycle` uses this same threshold) and `static` is one
        # frame, so there is nothing to remove; and dropping a cycle keyframe here
        # would desynchronise `frame_states`, which indexes keyframes by position.
        indices = _distinct_indices(frames, indices, p)
        # ...AND AFTER, because `_distinct_indices` BACKFILLS. Dedup frees
        # budget and the backfill walks the clip for anything not already
        # represented — which is exactly what a smeared frame is, so it gets
        # picked straight back up. Measured on a 3-hold fixture: filtering only
        # before dedup returned [0, 10, 12, 21, 25] with 10 and 21 the two
        # smears the first pass had just removed.
        indices = _drop_smeared(frames, indices, p)
    return KeyframeSet(
        # Asked even though the cycle test refused: the two questions differ,
        # and this is the one the driver needs answered.
        steady_screens=steady_screens(frames, p),
        mode="even",
        keyframes=[
            Keyframe(number=k + 1, source_index=idx, timestamp_ms=_ms(idx),
                     image=frames[idx])
            for k, idx in enumerate(indices)
        ],
        source_frames=n,
        fps=fps,
        params=p,
    )


def extract_keyframes_from_video(
    path: str | os.PathLike, *, params: Optional[KeyframeParams] = None
) -> KeyframeSet:
    frames, fps = decode_video(path)
    return extract_keyframes(frames, fps=fps, params=params)


# ── on-disk layout ──────────────────────────────────────────────────────────
#
#   <media_dir>/keyframes/<stem>/frame_01.png ... frame_0N.png
#   <media_dir>/keyframes/<stem>/keyframes.json
#
# Derived, regenerable, and ADDITIVE — the clip stays canonical next to it. That
# matters most under `cleanSamples/test/`, which is irreplaceable hand-labeled
# data we never rewrite (CLAUDE.md § Data trees): extraction only ever creates a
# new sibling directory there.


def keyframe_dir_for(media_path: str | os.PathLike) -> Path:
    """Where this clip's keyframes live. Pure path arithmetic, no I/O."""
    p = Path(media_path)
    return p.parent / KEYFRAME_DIR_NAME / p.stem


def _frame_filename(number: int) -> str:
    return f"frame_{number:02d}.png"


def write_keyframes(
    kfset: KeyframeSet, out_dir: str | os.PathLike, *, stem: str
) -> List[Path]:
    """Write `frame_NN.png` + `keyframes.json` into `out_dir`. Returns the PNG
    paths in model order.

    Stale files from a previous slicing are removed: a clip re-sliced from 6
    keyframes down to 3 would otherwise leave `frame_04..06.png` behind, and
    `read_keyframe_paths` globs the directory — the model would be handed six
    images while the manifest and the label said three.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for stale in out.glob("frame_*.png"):
        stale.unlink()

    paths: List[Path] = []
    names: List[str] = []
    for kf in kfset.keyframes:
        name = _frame_filename(kf.number)
        target = out / name
        if kf.image is None:
            raise ValueError(f"keyframe {kf.number} has no image to write")
        cv2.imwrite(str(target), kf.image)
        paths.append(target)
        names.append(name)

    with (out / MANIFEST_NAME).open("w") as f:
        json.dump(kfset.manifest(stem=stem, filenames=names), f, indent=2)
    return paths


def load_manifest(kf_dir: str | os.PathLike) -> Optional[Dict[str, Any]]:
    p = Path(kf_dir) / MANIFEST_NAME
    if not p.exists():
        return None
    try:
        with p.open() as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    return data if isinstance(data, dict) else None


def read_keyframe_paths(kf_dir: str | os.PathLike) -> List[Path]:
    """The keyframe PNGs in model order, or [] if the directory isn't usable.

    Order comes from the manifest when present and from the zero-padded filenames
    otherwise; both give frame 1 first. A manifest naming a file that isn't there
    yields [] rather than a short list, because a gap would silently renumber
    every later frame and invalidate the stored `frame` label.
    """
    d = Path(kf_dir)
    if not d.is_dir():
        return []
    manifest = load_manifest(d)
    if manifest and isinstance(manifest.get("keyframes"), list):
        out: List[Path] = []
        for entry in sorted(manifest["keyframes"], key=lambda e: e.get("number", 0)):
            p = d / str(entry.get("file", ""))
            if not p.exists():
                return []
            out.append(p)
        return out
    return sorted(d.glob("frame_*.png"))


def materialize_keyframes(
    media_path: str | os.PathLike,
    *,
    params: Optional[KeyframeParams] = None,
    force: bool = False,
) -> List[Path]:
    """The keyframes for a clip, extracting them on first use.

    Idempotent and cheap on the second call: an existing manifest whose params
    match is reused as-is. A params CHANGE re-slices, because reusing a set cut
    with different thresholds would mix two slicings in one dataset — the failure
    the manifest records params to prevent.
    """
    media = Path(media_path)
    kf_dir = keyframe_dir_for(media)
    p = params or KeyframeParams()

    if not force:
        manifest = load_manifest(kf_dir)
        if manifest and manifest.get("params") == p.as_dict():
            existing = read_keyframe_paths(kf_dir)
            if existing:
                return existing

    kfset = extract_keyframes_from_video(media, params=p)
    return write_keyframes(kfset, kf_dir, stem=media.stem)
