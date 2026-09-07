/**
 * Regression: "Please also check the new images." is PROGRESS, not a rejection.
 *
 * reCAPTCHA puts three different sentences in the same corner of the bframe,
 * and the driver used to read all three through one predicate
 * (`hasRecaptchaUnderselectError`) that answered a boolean:
 *
 *   .rc-imageselect-incorrect-response   "Please try again."
 *        -> the answer was WRONG. A fresh board follows.
 *   .rc-imageselect-error-select-more    "Please select all matching images."
 *        -> UNDER-SELECTED. The board does NOT refresh, so a driver that
 *           re-submits the same answer loops until it times out — which is why
 *           the one-retry-then-abort rule exists and is correct HERE.
 *   .rc-imageselect-error-dynamic-more   "Please also check the new images."
 *        -> RIGHT SO FAR. This is the dynamic 3x3's normal flow: cleared tiles
 *           fade out, replacements fade in, and the widget says so. It appears
 *           on essentially every round of that variant.
 *
 * Folding the third into the other two made the abort fire on the SECOND round
 * of every dynamic board, because the sentence that means "keep going" was
 * being counted as the same failure twice running. Measured 2026-09-06 on
 * google.com/recaptcha/api2/demo with the client's loop budget raised to 12:
 * all three `recaptcha_3x3_fade` attempts ended `client-stopped` at exactly
 * boards=2, while `recaptcha_4x4` on the same run passed at boards 2, 3 and 5.
 * The vendor was still dealing; we were the ones who stopped.
 *
 * The Python driver carries the identical table and the identical latch, so
 * `python/tests/test_recaptcha_dynamic_more_is_not_an_error.py` pins the same
 * three sentences. If one moves, move both (rule 1c).
 */

import { test } from 'node:test';
import assert from 'node:assert/strict';

import { CaptchaKrakenSolver } from './solver';

/** A page whose bframe shows exactly one of reCAPTCHA's banners. */
function pageShowing(selector: string | null, text = 'banner text'): any {
  const frame = {
    $: async (sel: string) =>
      sel === selector
        ? { isVisible: async () => true, textContent: async () => text }
        : null,
  };
  return {
    $: async (sel: string) =>
      sel === 'iframe[src*="recaptcha/api2/bframe"]'
        ? { contentFrame: async () => frame }
        : null,
  };
}

const solver = () => new CaptchaKrakenSolver({ apiKey: 'test' }) as any;

test('"Please try again" is a rejection', async () => {
  const kind = await solver().recaptchaBannerKind(
    pageShowing('.rc-imageselect-incorrect-response', 'Please try again.'),
  );
  assert.equal(kind, 'rejected');
});

test('"Please select all matching images" is an under-selection', async () => {
  const kind = await solver().recaptchaBannerKind(
    pageShowing('.rc-imageselect-error-select-more', 'Please select all matching images.'),
  );
  assert.equal(kind, 'select-more');
});

test('"Please also check the new images" is PROGRESS, not an error', async () => {
  const kind = await solver().recaptchaBannerKind(
    pageShowing('.rc-imageselect-error-dynamic-more', 'Please also check the new images.'),
  );
  assert.equal(
    kind,
    'dynamic-more',
    'the dynamic variant\'s "more images arrived" notice must not be classified with the two ' +
      'genuine errors — doing so aborts every dynamic board at round two',
  );
});

test('a clean board reports no banner', async () => {
  assert.equal(await solver().recaptchaBannerKind(pageShowing(null)), null);
});

/**
 * The one that actually cost us the variant: the abort latch must not be armed
 * by progress. Two consecutive dynamic-more notices are an ordinary two-round
 * dynamic solve.
 */
test('dynamic-more never arms the abort latch', async () => {
  const s = solver();
  const page = pageShowing('.rc-imageselect-error-dynamic-more', 'Please also check the new images.');
  assert.equal(s.bannerIsFatalAfterRetry(await s.recaptchaBannerKind(page)), false);
  assert.equal(s.bannerIsFatalAfterRetry(await s.recaptchaBannerKind(page)), false);
});

test('a repeated under-selection IS fatal after one retry', async () => {
  const s = solver();
  const page = pageShowing('.rc-imageselect-error-select-more', 'Please select all matching images.');
  assert.equal(s.bannerIsFatalAfterRetry(await s.recaptchaBannerKind(page)), true);
});
