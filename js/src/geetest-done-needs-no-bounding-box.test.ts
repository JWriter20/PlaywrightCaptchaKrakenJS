/**
 * Regression: a `done` answer must not need the widget's bounding box.
 *
 * The box exists to turn NORMALISED coordinates into page ones. `done` carries
 * no coordinates — it means "nothing left to click" — so demanding a box for
 * it fails the solve at exactly the moment it succeeded, because several
 * vendors CLOSE the challenge as soon as they accept it.
 *
 * MEASURED 2026-09-07 on gt4.geetest.com, with the trace:
 *
 *   --- Captcha Solve Loop 1/6 ---
 *   CLI -> {"action":"click","target_bounding_boxes":[[...],[...],[...]]}
 *   Executing 1 actions.
 *   --- Captcha Solve Loop 2/6 ---
 *   CLI -> {"action":"done"}
 *   solve threw: Could not get bounding box of captcha element
 *   element now: box=NULL vis=false attached=true css=none/...
 *
 * The panel was `display:none` because GeeTest had ACCEPTED the three tiles.
 * Every live GeeTest attempt died there — 22 of 22 across four puzzles — and
 * every one was banked as the model getting it wrong. A probe held the same
 * element for 20s with a screenshot and a wandering pointer and it never
 * moved, which is what ruled out the panel simply timing out.
 *
 * The allow-list direction matters: a coordinate action with no box still
 * throws, because that one genuinely cannot be performed.
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';

import { answerNeedsElementBox, isStaleHandleError } from './solver';

test('a done-only answer needs no box', () => {
  assert.equal(answerNeedsElementBox([{ action: 'done' }]), false);
  assert.equal(answerNeedsElementBox([{ action: 'done' }, { action: 'done' }]), false);
});

test('every coordinate-bearing action still needs one', () => {
  for (const action of ['click', 'drag', 'type', 'slide', 'move']) {
    assert.equal(answerNeedsElementBox([{ action }]), true, action);
  }
});

test('a done mixed with real work still needs one', () => {
  // The box is required if ANY action needs it — a mixed answer must not be
  // allowed through on the strength of its last element.
  assert.equal(answerNeedsElementBox([{ action: 'done' }, { action: 'click' }]), true);
  assert.equal(answerNeedsElementBox([{ action: 'click' }, { action: 'done' }]), true);
});

test('an unrecognised action defaults to needing one', () => {
  // Allow-list, not deny-list: a coordinate action added later and not listed
  // here must fail loudly rather than click at the origin.
  assert.equal(answerNeedsElementBox([{ action: 'some_future_gesture' }]), true);
  assert.equal(answerNeedsElementBox([{}]), true);
});

test('an empty answer needs nothing', () => {
  assert.equal(answerNeedsElementBox([]), false);
});

/*
 * The other half of the same vendor behaviour.
 *
 * A `done` answer is what the model returns when the board is finished, and
 * the fix above lets it through. But the model does not always say `done`: on
 * gt4.geetest.com's gobang board it answered loop 2 with a CLICK, against a
 * screenshot taken a moment before GeeTest closed the accepted panel. That is
 * not the model being wrong — it is the widget moving on, which is the same
 * thing hCaptcha does when it swaps in the next round, and the driver already
 * has a bounded re-detect for it.
 */

test('a vanished bounding box is the widget moving on', () => {
  assert.equal(isStaleHandleError('Could not get bounding box of captcha element'), true);
});

test('the shapes hCaptcha produces are still recognised', () => {
  for (const m of [
    'Timeout 3000ms exceeded',
    'element is not visible',
    'Element is not attached to the DOM',
    'Target closed',
  ]) {
    assert.equal(isStaleHandleError(m), true, m);
  }
});

test('a real failure is not mistaken for a transition', () => {
  // These must keep failing the solve: the widget was there, the answer
  // reached the vendor, and the vendor said no.
  for (const m of [
    'Captcha still detected after 6 solve loops',
    'No progress: the model returned the same answer 3 times running',
    'Cannot solve this kind of captcha — UNSUPPORTED_CAPTCHA',
    'account is out of credits',
  ]) {
    assert.equal(isStaleHandleError(m), false, m);
  }
});
