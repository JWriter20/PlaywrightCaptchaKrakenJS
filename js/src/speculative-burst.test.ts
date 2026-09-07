/**
 * Ask and watch at the same time.
 *
 * The screenshot goes to the model and the recording starts in the same breath,
 * because the burst is the only thing that can tell a cycling board from a
 * still one AND is also the recording that answers it. Whichever the board
 * turns out to be, the answer arrives about one inference from now:
 *
 *   still  — one picture seen, the still answer stands, frames dropped. Costs
 *            nothing: the burst happened inside a wait we were already making.
 *   moving — the still answer is discarded UNREAD (it describes a screen that
 *            has gone), the burst runs to the end of the cycle, and the
 *            multi-image answer is used. Two inference CALLS, about one
 *            inference of wall-clock.
 *
 * The shape before this learned the same thing in whole ROUNDS — answer a
 * still, act on it, notice nothing moved, answer another still, then record.
 * Measured at ~15s of a 40.3s solve, and three inference calls for a puzzle
 * whose entire task is one click.
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';

import { CaptchaKrakenSolver } from './solver';

const STILL = { actions: [{ action: 'click', target_bounding_boxes: [[0.1, 0.1, 0.2, 0.2]] }], token_usage: [] };
const VIDEO = { actions: [{ action: 'click', target_bounding_boxes: [[0.4, 0.4, 0.5, 0.5]], frame: 2 }], token_usage: [] };

/** A solver whose recording either sees movement or does not. */
function speculating(moved: boolean) {
  const solver: any = new CaptchaKrakenSolver({});
  const calls: string[] = [];
  let abandoned = false;
  let finished = false;
  solver.startKeyframeBurst = () => ({
    moved: () => moved,
    abandon: async () => { abandoned = true; },
    finish: async () => { finished = true; return '/tmp/ck_burst_fake'; },
  });
  solver.getSolution = async () => { calls.push('still'); return STILL; };
  solver.getAnimatedSolution = async () => { calls.push('video'); return VIDEO; };
  solver.captchaFrameChangedSince = async () => false;
  solver.answerFor = async (_k: string, run: () => any) => run();
  return { solver, calls, state: () => ({ abandoned, finished }) };
}

test('a still board pays exactly one inference and drops the recording', async () => {
  const { solver, calls, state } = speculating(false);
  const rec = solver.startKeyframeBurst();
  const still = await solver.solveFrameFreshnessGuarded(
    {} as any, '/tmp/s.png', () => solver.getSolution());
  assert.equal(rec.moved(), false);
  await rec.abandon();
  assert.deepEqual(calls, ['still'], 'a still board asked the model more than once');
  assert.equal(state().abandoned, true, 'the frames of a still board were kept');
  assert.equal(state().finished, false, 'a still board finished a recording it did not need');
  assert.ok(still);
});

test('a moving board drops the still answer and finishes the recording', async () => {
  const { solver, calls, state } = speculating(true);
  const rec = solver.startKeyframeBurst();
  await solver.solveFrameFreshnessGuarded({} as any, '/tmp/s.png', () => solver.getSolution());
  assert.equal(rec.moved(), true);
  const dir = await rec.finish();
  const video = await solver.getAnimatedSolution(dir);
  assert.deepEqual(calls, ['still', 'video'],
    'the still call is expected — it is the one that overlaps the recording');
  assert.equal(state().finished, true);
  assert.equal(video.actions[0].frame, 2, 'the answer acted on must be the video one');
});

test('reCAPTCHA never speculates', () => {
  // Its dynamic 3x3 replaces tiles in place and has its own fade gates; a burst
  // there would film a fade and call it a cycle.
  const solver: any = new CaptchaKrakenSolver({});
  assert.equal(solver.shouldSpeculate('recaptcha', false), false);
  assert.equal(solver.shouldSpeculate('hcaptcha', false), true);
  assert.equal(solver.shouldSpeculate('unknown', false), true);
});

test('a distorted-text round never speculates', () => {
  // The answer is a string, not a place. Nothing in a recording helps read one.
  const solver: any = new CaptchaKrakenSolver({});
  assert.equal(solver.shouldSpeculate('unknown', true), false);
});

test('it can be turned off, and off is the old behaviour', () => {
  const solver: any = new CaptchaKrakenSolver({ speculativeBurstEnabled: false });
  assert.equal(solver.shouldSpeculate('hcaptcha', false), false);
  const noVideo: any = new CaptchaKrakenSolver({ videoSolveEnabled: false });
  assert.equal(noVideo.shouldSpeculate('hcaptcha', false), false);
});

test('the speculative round does not wander the cursor over what it is filming', () => {
  // withIdleWander drifts the pointer across the widget during inference. Every
  // frame would then have a mouse in a different place, which reads as a board
  // with a dozen screens — and one live burst reported exactly that.
  const fs = require('node:fs') as typeof import('fs');
  const path = require('node:path') as typeof import('path');
  const src = fs.readFileSync(path.resolve(__dirname, '..', 'src', 'solver.ts'), 'utf8');
  const start = src.indexOf('} else if (this.shouldSpeculate(');
  const end = src.indexOf('} else {', start);
  assert.ok(start > 0 && end > start, 'could not find the speculative branch');
  const branch = src.slice(start, end);
  assert.ok(
    !/withIdleWander[\s\S]{0,200}getSolution/.test(branch),
    'the still inference inside the speculative branch wanders the cursor',
  );
});
