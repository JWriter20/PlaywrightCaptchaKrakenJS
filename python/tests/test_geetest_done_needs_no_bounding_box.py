"""Regression: a ``done`` answer must not need the widget's bounding box.

The box exists to turn NORMALISED coordinates into page ones. ``done`` carries
none — it means "nothing left to click" — so requiring a box for it fails the
solve at exactly the moment it succeeded, because several vendors CLOSE the
challenge as soon as they accept it.

MEASURED 2026-09-07 on gt4.geetest.com. Loop 1 clicked three tiles; loop 2
returned ``{"action": "done"}``; the panel was already ``display:none`` because
GeeTest had accepted them; the driver raised "could not get bounding box of
captcha element". Every live GeeTest attempt died there — 22 of 22 across four
puzzles — and each was recorded as the model getting it wrong. A probe held the
same element for 20s with a screenshot and a wandering pointer and it never
moved, which is what ruled out the panel merely timing out.

The JS port carries the identical rule; `js/src/geetest-done-needs-no-bounding-box.test.ts`
pins the same cases (rule 1c).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from captchakraken.page_solver import answer_needs_element_box  # noqa: E402


def test_a_done_only_answer_needs_no_box():
    assert answer_needs_element_box([{"action": "done"}]) is False
    assert answer_needs_element_box([{"action": "done"}, {"action": "done"}]) is False


def test_every_coordinate_action_still_needs_one():
    for action in ("click", "drag", "type", "slide", "move"):
        assert answer_needs_element_box([{"action": action}]) is True, action


def test_a_done_mixed_with_real_work_still_needs_one():
    # Required if ANY action needs it — a mixed answer must not be let through
    # on the strength of its last element.
    assert answer_needs_element_box([{"action": "done"}, {"action": "click"}]) is True
    assert answer_needs_element_box([{"action": "click"}, {"action": "done"}]) is True


def test_an_unrecognised_action_defaults_to_needing_one():
    # Allow-list, not deny-list: a coordinate action added later and not listed
    # must raise loudly rather than click at the origin.
    assert answer_needs_element_box([{"action": "some_future_gesture"}]) is True
    assert answer_needs_element_box([{}]) is True


def test_an_empty_answer_needs_nothing():
    assert answer_needs_element_box([]) is False
