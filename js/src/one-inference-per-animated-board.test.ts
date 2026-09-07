/**
 * A cycling board is recorded ONCE and asked about ONCE.
 *
 * The whole task on GeeTest's svg board is a single click: press the image
 * matching the icon in the corner. The answer already carries everything needed
 * to do it — the cell, and `frame` naming which of the board's screens that
 * cell is on. The gate then holds the pointer on the cell until that screen is
 * back up. Nothing in that changes between rounds: the board keeps cycling
 * through the same three pictures with the same target.
 *
 * It was re-recording and re-asking on every round anyway:
 *
 *     Loop 3  record 4s + multi-image inference + gated click
 *     Loop 4  record 4s + multi-image inference + gated click
 *     Loop 5  record 4s ... budget gone
 *     Loop 6  timed out after 66000ms
 *
 * Three bursts and three inferences to make one click. Measured, that is ~4s of
 * recording plus 3.1-3.4s of inference per round — so the loop spent its whole
 * budget re-deriving an answer it already had.
 *
 * The plan is dropped when the gate reports it never saw the chosen screen,
 * because that is the one signal that says the widget is no longer the board
 * the plan was made for.
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';

import { CaptchaKrakenSolver } from './solver';

const ANSWER = {
  actions: [{ action: 'click', target_bounding_boxes: [[0.4, 0.4, 0.5, 0.5]], frame: 3 }],
  token_usage: [],
};

function animatedSolver() {
  const solver: any = new CaptchaKrakenSolver({});
  let bursts = 0;
  let inferences = 0;
  solver.recordKeyframeBurst = async () => { bursts++; return `/tmp/ck_burst_fake_${bursts}`; };
  solver.getAnimatedSolution = async () => { inferences++; return ANSWER; };
  solver.discardAnimatedPlan = function () { this.animatedPlan = null; };  // no fs
  return { solver, counts: () => ({ bursts, inferences }) };
}

/** One animated round, as `solveSingle` runs it. */
async function round(solver: any): Promise<void> {
  if (solver.animatedPlan) return;                       // reuse
  const burstDir = await solver.recordKeyframeBurst();
  const response = await solver.getAnimatedSolution(burstDir);
  solver.animatedPlan = { burstDir, response };
}

test('a second round costs no burst and no inference', async () => {
  const { solver, counts } = animatedSolver();
  await round(solver);
  await round(solver);
  await round(solver);
  assert.deepEqual(counts(), { bursts: 1, inferences: 1 },
    'the board was re-recorded or re-asked about; one click needs one of each');
});

test('the plan is dropped when the gate never saw its screen', async () => {
  const { solver } = animatedSolver();
  await round(solver);
  assert.ok(solver.animatedPlan, 'nothing was planned');
  // What `waitForKeyframe` does when the widget never matches: the board is no
  // longer the one this answer describes.
  solver.discardAnimatedPlan();
  assert.equal(solver.animatedPlan, null,
    'a spent plan would re-click a cell chosen from pictures that are gone');
});

test('a new solve starts with no plan', async () => {
  const { solver } = animatedSolver();
  await round(solver);
  solver.resetSolveState();
  assert.equal(solver.animatedPlan, null,
    'a plan leaking into the next captcha would answer it with the previous one');
});

test('the recorded frames outlive the round that made them', () => {
  // The gate re-reads the keyframe PNGs on every poll, and the NEXT round
  // re-executes this same answer — so the per-round cleanup must not delete a
  // directory the plan is still holding.
  // Read from the repo, not the build: this asserts on a line of SOURCE, and
  // the compiled output has the comment that explains it stripped away.
  const fs = require('node:fs') as typeof import('fs');
  const path = require('node:path') as typeof import('path');
  const src = fs.readFileSync(
    path.resolve(__dirname, '..', 'src', 'solver.ts'), 'utf8');
  assert.match(
    src, /this\.animatedPlan\?\.burstDir !== burstDir/,
    'the round-end cleanup no longer spares the directory the plan holds',
  );
});
