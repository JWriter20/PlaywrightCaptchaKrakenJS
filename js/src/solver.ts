// Playwright API types only — and our OWN structural copies, not a browser
// package's. The solver never launches a browser; the caller hands us a `Page`
// from whichever Playwright-compatible launcher they chose (vanilla `playwright`,
// `patchright`, `camoufox-js`, …). Typing against any one of those would pull it
// into consumers' trees and break across version skew, so instead we duck-type
// the exact slice of the Playwright surface the solver uses. Every real
// Playwright `Page`/`Frame`/`ElementHandle` structurally satisfies these. See
// playwright-types.ts.
import {
  PlaywrightPage as Page,
  PlaywrightElementHandle as ElementHandle,
  PlaywrightFrame as Frame,
} from './playwright-types';
import { watchPage, CaptchaWatcher, WatchOptions } from './watcher';
import { Humanizer, resolveHumanizer } from './humanize.js';
import { exec, execFile, spawn, spawnSync, ChildProcessWithoutNullStreams } from 'child_process';
import { promisify } from 'util';
import * as fs from 'fs';
import * as path from 'path';
import * as os from 'os';
import { createHash, randomUUID } from 'crypto';
import { PhaseBudget, timingsEnabled } from './timing';
import { CaptchaKrakenConfig, SolverResult, ClickAction, DragAction, TypeAction, CaptchaAction, SolveResult, CliResponse, TokenUsage, Vector, SolveStepEvent } from './types';
import { aggregateTokenUsage } from './token-usage';
import { parseApiError } from './errors';
import { DEFAULT_RECAPTCHA_MAX_DYNAMIC_ROUNDS } from './limits';
import { resolvePythonCommand } from './python-command';
import { buildSolveArgs, redactCommand, solveEnv } from './cli-invocation';
import { solveSlideGeometry } from './slide-geometry';
import { getBundledCliRoot, resolveLoraName } from './model-name';

const execAsync = promisify(exec);
const execFileAsync = promisify(execFile);

/**
 * Centre of an [x1, y1, x2, y2] 0–1 box, as the (x, y) 0–1 point the animated
 * wait gate compares around. The solver builds these boxes as a small square
 * around the model's point, so the centre recovers that point exactly.
 */
function bboxCenter(bbox: [number, number, number, number]): [number, number] {
  return [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2];
}

/**
 * Is `command` runnable? Probed with `--version`, which every candidate
 * interpreter answers cheaply and without side effects. `shell: false` so a
 * command name can never be interpreted by /bin/sh.
 */
function commandExists(command: string): boolean {
  try {
    const probe = spawnSync(command, ['--version'], { stdio: 'ignore', shell: false });
    return !probe.error && probe.status === 0;
  } catch {
    return false;
  }
}

function getVenvPython(cliRoot: string): string | null {
  const venvDir = path.join(cliRoot, '.venv');
  const candidates = [
    path.join(venvDir, 'bin', 'python'),
    path.join(venvDir, 'bin', 'python3'),
    path.join(venvDir, 'Scripts', 'python.exe'),
    path.join(venvDir, 'Scripts', 'python'),
  ];
  for (const c of candidates) {
    if (fs.existsSync(c)) return c;
  }
  return null;
}

// Env for spawning the python CLI. Prepend the bundled `python/src` to
// PYTHONPATH so `python -m captchakraken.cli` imports even when the postinstall
// `pip install` was skipped or failed (best-effort bootstrap). `extra` carries
// per-invocation values (currently the solve session id) and is applied last so
// it wins over the inherited environment.
function cliEnv(cliRoot: string, extra?: NodeJS.ProcessEnv): NodeJS.ProcessEnv {
  const srcDir = path.join(cliRoot, 'src');
  const existing = process.env.PYTHONPATH;
  return {
    ...process.env,
    PYTHONPATH: existing ? `${srcDir}${path.delimiter}${existing}` : srcDir,
    ...extra,
  };
}

// Simple Vector interface for internal use moved to types.ts

/** Cached geometry for one reCAPTCHA 3x3 dynamic-puzzle session. */
interface GridSession {
  /** Grid cell boxes in SCREENSHOT pixel space, row-major, 0-indexed array. */
  gridBoxes: number[][];
  /** Playwright element box in PAGE css px (for mouse coords). */
  elementBox: { x: number; y: number; width: number; height: number };
  /** screenshot px -> page px. */
  scaleX: number;
  scaleY: number;
  /** Screenshot dimensions the gridBoxes were computed against. */
  screenshotW: number;
  screenshotH: number;
}

/** Per-cell grid state from grid-cell-states-fixed (1-indexed cell numbers). */
interface GridCellStates {
  empty: number[];
  changing: number[];
  loaded: number[];
  selected: number[];
}

const log = (message: string, ...args: any[]) => console.log(`[Solver] ${message}`, ...args);
const delay = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

/**
 * Lifecycle state of the challenge the solver is driving. Tracked so behaviours
 * can be gated on it — in particular so we never feed a mid-transition or
 * still-loading frame to the model.
 *
 *   Detecting → Loading → Ready → Solving → Acting → Submitting → Transitioning
 *   → (Loading → …) | Solved
 *
 * `Animated` used to be the terminal "it's a video, give up" exit. It no longer
 * is: a challenge that never settles gets RECORDED and solved from keyframes, so
 * the normal path absorbs it. The state now means only that the recording was
 * impossible, or that `videoSolveEnabled` is off.
 */
enum CaptchaState {
  Detecting = 'detecting',
  Loading = 'loading',
  Ready = 'ready',
  Solving = 'solving',
  Acting = 'acting',
  Submitting = 'submitting',
  Transitioning = 'transitioning',
  Solved = 'solved',
  Animated = 'animated',
}

/**
 * Vendors with no checkbox/challenge split — one container, one interactive
 * surface. Checked in detectCaptcha() after the five hard-coded reCAPTCHA /
 * hCaptcha / Turnstile checks above, so those keep first refusal. Selectors
 * lifted from src/captchaCollection/sources.py, which already drives these 8
 * vendors nightly in the collector. Mirror of PYTHON_VENDOR_WIDGET_LOCATORS in
 * page_solver.py — keep both in the same order with the same selectors.
 */
const VENDOR_WIDGET_LOCATORS: ReadonlyArray<{ puzzleSource: string; selectors: string[] }> = [
  { puzzleSource: 'geetest', selectors: ['.geetest_box', '.geetest_panel_box', '.geetest_popup_window', '.geetest_widget'] },
  // Tencent renders IN THE HOST DOCUMENT since 2026-08-11; before that it was
  // `iframe#tcaptcha_iframe_dy`. Both shapes are listed, in-page first, because
  // the iframe build is still deployed on sites pinning an older widget and
  // costs nothing to keep. Dropping the in-page selectors is what made the
  // client blind to every live Tencent captcha for twelve days.
  //
  // `iframe[id^="tcaptcha"]`, NOT `[id*=...]`. The substring form also matched
  // MTCaptcha's `mtcaptcha-iframe-1` — "mtcaptcha" contains "tcaptcha" — so
  // Tencent's entry was silently the only thing detecting MTCaptcha at all,
  // which is why `.mtcap` matching nothing went unnoticed. Anchored, it still
  // matches the real `tcaptcha_iframe_dy` and claims nobody else's widget.
  { puzzleSource: 'tencent', selectors: ['#tcaptcha_transform_dy', '#tCaptchaDyContent', '.tencent-captcha-dy__content', 'iframe#tcaptcha_iframe_dy', 'iframe[id^="tcaptcha"]', 'iframe[src*="captcha.gtimg.com"]', 'iframe[src*="captcha.qq.com"]'] },
  { puzzleSource: 'yidun', selectors: ['.yidun_panel', '.yidun'] },
  // Yandex SmartCaptcha is IFRAMED, and `.CheckboxCaptcha` is a class inside
  // that frame's document — querySelector does not cross a frame boundary, so
  // the host page never carried it and this entry matched nothing, ever.
  // Challenge frame first, anchor second, exactly like reCAPTCHA above.
  // `.CheckboxCaptcha` stays for an inline embed that renders it directly.
  { puzzleSource: 'yandex', selectors: ['iframe[src*="smartcaptcha.yandexcloud.net/advanced"]', 'iframe[src*="smartcaptcha.yandexcloud.net"]', '.CheckboxCaptcha'] },
  { puzzleSource: 'lemin', selectors: ['#lemin-cropped-captcha', '.lemin-captcha-popup'] },
  { puzzleSource: 'prosopo', selectors: ['.prosopo-modalInner', '.procaptcha-checkbox'] },
  // MTCaptcha is IFRAMED too: `<div class="mtcaptcha"><iframe
  // id="mtcaptcha-iframe-1" src="https://service.mtcaptcha.com/...">`. `.mtcap`
  // is the class prefix used INSIDE that frame (`mtcap-inputtext`), and a CSS
  // class selector matches whole tokens, so it never matched the host page's
  // `class="mtcaptcha"` either. Kept last for an inline embed.
  { puzzleSource: 'mtcaptcha', selectors: ['iframe[src*="service.mtcaptcha.com"]', 'iframe[id^="mtcaptcha-iframe"]', '.mtcaptcha', '.mtcap'] },
  { puzzleSource: 'botdetect', selectors: ['.BDC_CaptchaDiv'] },
];

/**
 * Which vendor's code the page LOADED, as opposed to what it rendered.
 * Mirrors VENDOR_URL_MARKERS in page_solver.py — CLAUDE.md 1c.
 *
 * A selector is a claim about markup, and markup is the half of a captcha the
 * vendor is free to rewrite overnight — it is an anti-bot surface, so churn is
 * the point. The REQUEST is far more stable: Tencent moved its widget out of an
 * iframe and renamed every class on 2026-08-11, and went on fetching from
 * `turing.captcha.gtimg.com` and `turing.captcha.qcloud.com` throughout. The
 * substring `captcha.gtimg.com` — already in the table above, on a selector
 * that stopped matching — never left the wire.
 *
 * This is a TRIPWIRE, not a detection path. A URL says which vendor is on the
 * page; it cannot say where the widget is, and the solver needs an ELEMENT to
 * screenshot, to scope its controls against, and to drag. What it buys is
 * telling apart the two things a null detection can mean: no captcha here, or
 * the vendor's code is loaded and running and nothing we look for matches.
 *
 * Hosts measured 2026-08-24 from each vendor's demo page.
 */
/**
 * The two vendors with bespoke handling in this file. EVERYTHING else shares the
 * generic path, and the two behaviours below are gated on "not one of these" —
 * spelled as this set rather than as `=== 'unknown'`, which is what they used to
 * say. The two were identical only while `unknown` was the sole third value, so
 * the day anyone reports a vendor by name — to constrain which grid shapes it may
 * be solved as, say — `=== 'unknown'` silently turns OFF typed-challenge detection
 * for MTCaptcha, Yandex and BotDetect (all three ARE typed captchas) and the
 * animated settle probe for GeeTest and Tencent. Neither failure throws; the text
 * captcha just becomes unsolvable and the animated board gets answered from one
 * arbitrary frame. Mirrors VENDORS_WITH_BESPOKE_HANDLING in page_solver.py.
 */
const VENDORS_WITH_BESPOKE_HANDLING: ReadonlySet<string> = new Set(['hcaptcha', 'recaptcha']);

const VENDOR_URL_MARKERS: ReadonlyArray<{ puzzleSource: string; hosts: string[] }> = [
  { puzzleSource: 'hcaptcha', hosts: ['hcaptcha.com'] },
  { puzzleSource: 'recaptcha', hosts: ['google.com/recaptcha', 'recaptcha.net'] },
  { puzzleSource: 'turnstile', hosts: ['challenges.cloudflare.com'] },
  { puzzleSource: 'geetest', hosts: ['geetest.com'] },
  { puzzleSource: 'tencent', hosts: ['captcha.gtimg.com', 'captcha.qcloud.com'] },
  { puzzleSource: 'yidun', hosts: ['dun.163.com', 'cstaticdun.126.net', 'necaptcha.nosdn.127.net'] },
  { puzzleSource: 'yandex', hosts: ['smartcaptcha.yandexcloud.net'] },
  { puzzleSource: 'lemin', hosts: ['leminnow.com'] },
  { puzzleSource: 'prosopo', hosts: ['prosopo.io'] },
  { puzzleSource: 'mtcaptcha', hosts: ['mtcaptcha.com'] },
  // BotDetect is deliberately absent and must stay absent: it is a SELF-HOSTED
  // library, so the image comes from the application's own origin and there is
  // no vendor host to see. An empty result for a BotDetect page means "this
  // check does not apply", not "no captcha".
];

/**
 * Where the answer goes, when it is not a click. Mirrors TEXT_INPUT_SELECTORS /
 * SLIDER_HANDLE_SELECTORS / DRAGGABLE_PIECE_SELECTORS in page_solver.py — keep
 * both in the same order with the same selectors.
 *
 * Ordered VENDOR-FIRST, GENERIC-LAST, and the driver takes the first visible
 * match. That order is the design: a named vendor selector is unambiguous,
 * while the generic patterns are guesses that happen to be right most of the
 * time. Trying the guess first would, on a page hosting a captcha *and* a login
 * form, type the captcha's answer into the username box.
 *
 * The generic tail is not a nicety either — it is what actually fires on most
 * pages. Vendors rename these classes without notice, and our own Tier 3
 * fixtures render neither vendor's DOM.
 */
// Split in two because the two halves are safe in different places. The NAMED
// half identifies a captcha's answer box wherever it sits; the GENERIC half only
// means "the text box in this widget" and is trustworthy only when the scope
// already is the widget. `answerBox` widens past the widget for BotDetect and
// may take the named half with it, never the generic one.
const TEXT_INPUT_VENDOR_SELECTORS: ReadonlyArray<string> = [
  // BotDetect — the input is application-defined, so match the id fragment its
  // own docs and samples use (the three the nightly collector already drives).
  'input[id*=captchaCode]',
  'input#captchaCode',
  'input[id*=validateCaptcha]',
  '.BDC_CaptchaDiv input[type=text]',
  // MTCaptcha
  'input.mtcap-inputtext',
  '.mtcap input[type=text]',
  // Yandex SmartCaptcha
  '.AdvancedCaptcha-Input input',
  'input.Textinput-Control',
  'input[name="rep"]',
];

const TEXT_INPUT_GENERIC_SELECTORS: ReadonlyArray<string> = [
  // Generic — an input the page itself labels as the captcha answer.
  'input[name*="captcha" i]',
  'input[id*="captcha" i]',
  'input[aria-label*="captcha" i]',
  'input[placeholder*="code" i]',
  'input[autocomplete="off"][type=text]',
  // Last resort: the only text box in the widget. Scoped to the challenge
  // frame/container by the caller, never to the whole page — see findControl.
  'input[type=text]',
  'input:not([type])',
  'input[type=tel]',
  'textarea',
];

const TEXT_INPUT_SELECTORS: ReadonlyArray<string> = [
  ...TEXT_INPUT_VENDOR_SELECTORS, ...TEXT_INPUT_GENERIC_SELECTORS,
];

/**
 * The handle you drag on a puzzle-piece slider. NOT the piece: on every one of
 * these vendors the piece is inert decoration that the handle carries, so a
 * drag starting on the piece moves nothing at all.
 */
const SLIDER_HANDLE_SELECTORS: ReadonlyArray<string> = [
  // GeeTest v3 / v4
  '.geetest_slider_button',
  '.geetest_btn',
  '.geetest_slider .geetest_arrow',
  // Tencent. The post-2026-08-11 knob is a bare div — no id, no role=slider,
  // no aria-valuenow — so every generic fallback below misses it and it has to
  // be named. The three after it are the pre-redesign widget.
  '.tencent-captcha-dy__slider-block',
  '#tcaptcha_drag_thumb',
  '.tc-slider-normal',
  '[id*=slideBlock]',
  // Yidun (NetEase)
  '.yidun_slider',
  '.yidun_jigsaw',
  // Lemin
  '.lemin-slider-handle',
  '#lemin-cropped-captcha .slider',
  // Generic — an ARIA slider, or a class that says handle/thumb/button on a
  // track. `[draggable=true]` is deliberately absent: it is the HTML5
  // drag-and-drop opt-in, which fires dragstart rather than pointermove, and no
  // slider captcha uses it.
  '[role="slider"]',
  '[aria-valuenow]',
  '[class*="slider"][class*="btn"]',
  '[class*="slider"][class*="button"]',
  '[class*="slide"][class*="handle"]',
  '[class*="drag"][class*="thumb"]',
];

/**
 * Fallback for the sliderless members of the family. Lemin's "cropped" puzzle
 * has no track at all — you drag the piece itself onto the gap — and the model
 * answers it with the same sourceless drag, because from the picture the two
 * are indistinguishable. Tried only after SLIDER_HANDLE_SELECTORS finds nothing.
 */
const DRAGGABLE_PIECE_SELECTORS: ReadonlyArray<string> = [
  '.lemin-cropped-puzzle-piece',
  '#lemin-cropped-captcha canvas + canvas',
  '[class*="puzzle"][class*="piece"]',
  '[class*="jigsaw"]',
];

/**
 * Puzzle-piece slider tuning. Mirrors the `slide_*` fields of
 * PageSolverConfig in page_solver.py.
 */
const SLIDE_PROBE_OFFSETS_PX = [24, 64];
const SLIDE_TOLERANCE_PX = 2;
const SLIDE_MAX_CORRECTIONS = 2;

/**
 * The two numbers that bound a solve, together, because they only make sense
 * together: `maxSolveLoops` is the count that FITS inside the timeout at the
 * ~4-7s a round costs, and the timeout is the backstop for when it does not.
 * Named rather than inlined as `??` defaults so the relationship between them
 * can be pinned by a test — see no-progress.test.ts. The Python port keeps the
 * same pair on `PageSolverConfig`.
 */
/**
 * Does this answer need the widget's box to be performed?
 *
 * Only a coordinate-bearing action does. `done` means "nothing left to click"
 * and is the answer a vendor that closes on success will be showing no widget
 * for, so requiring a box for it fails the solve at the moment it succeeded.
 *
 * Written as an ALLOW-LIST of what needs nothing rather than a list of what
 * does: a new coordinate action added later must default to needing the box,
 * because that is the direction where being wrong throws instead of clicking
 * nowhere.
 */
export function answerNeedsElementBox(actions: ReadonlyArray<{ action?: string }>): boolean {
  return actions.some((a) => a?.action !== 'done');
}

/**
 * Is this the challenge TRANSITIONING under us, rather than a dead puzzle?
 *
 * hCaptcha swaps in the next round while we hold the old iframe, so a
 * screenshot on it fails "not visible" / "Timeout" / "not attached". GeeTest
 * does the mirror image: it CLOSES the panel the moment it accepts an answer,
 * so the widget we are holding loses its layout box between the screenshot and
 * the click. Both are the widget moving on, and the answer to both is to
 * re-detect on the next round rather than to fail the solve.
 *
 * The caller gates this on having ALREADY INTERACTED, which is what keeps it
 * honest: a first-frame failure is a genuine problem and still surfaces.
 *
 * Measured 2026-09-07 on gt4.geetest.com — every live GeeTest attempt ended
 * "Could not get bounding box of captcha element" on a panel the vendor had
 * just accepted, and each was recorded as the model being wrong.
 */
export function isStaleHandleError(message: string): boolean {
  return /Timeout .*exceeded|not visible|not attached|detached|Target closed|bounding box of captcha element/i
    .test(message);
}

/**
 * How unlike the chosen keyframe the widget has to look before the gate calls
 * it a different board, and over how many polls.
 *
 * Measured on GeeTest svg: two SCREENS of one board differ by 0.0056 (thin line
 * art, a few glyph strokes); a different board reads 0.77. Two orders of
 * magnitude apart, so this threshold is a plateau rather than a knife edge.
 * Several polls, not one, because a transition frame caught mid-swap can read
 * high for a moment.
 */
const NOT_THIS_BOARD_DIFF = 0.5;
/**
 * How much the widget must change during inference to count as MOVING.
 *
 * A separate, much tighter number than `staleFrameDiffThreshold` (0.02),
 * because it answers a different question. 0.02 asks "did this answer go
 * stale" — did tiles move enough that the coordinates no longer land. This asks
 * "is this picture the same picture", which is the noise floor, and the two are
 * an order of magnitude apart.
 *
 * Using the stale threshold for both made the guard BLIND to the board it most
 * needed to see: GeeTest svg's screens differ by 0.0056, comfortably under
 * 0.02, so a board cycling all the way through a 2.7s inference reported no
 * change at all. Third time this file has applied a coarse threshold to a fine
 * question (see `distinct_ratio` in keyframes.py and NOT_THIS_BOARD_DIFF).
 *
 * 0.002 sits above the measured frame-to-frame noise floor (0.001) and well
 * under the 0.0056 separation.
 */
const MOVED_DURING_INFERENCE_DIFF = 0.002;
const NOT_THIS_BOARD_POLLS = 3;

export const SOLVE_DEFAULTS = {
  maxSolveLoops: 6,
  overallSolveTimeoutMs: 45_000,
  /**
   * How long to hold a click waiting for the screen the model answered about.
   *
   * 9000, not the 6000 it was. A GeeTest svg board dwells p50 1.5s / p75 2.0s /
   * max 2.7s per screen, so a 3-screen cycle runs 4.5s median and 8.1s worst
   * case — a 6s budget could not catch the worst case however well aimed, and
   * gave up one screen short. Named here rather than left as a literal in two
   * places because `video_budget_ms` is derived from it on both ports and the
   * arithmetic has to agree.
   */
  keyframeWaitTimeoutMs: 9_000,
} as const;

export class CaptchaKrakenSolver {
  private config: CaptchaKrakenConfig;
  /** Extra ms this solve has been granted for a recording; see recordKeyframeBurst. */
  private videoBudgetMs: number = 0;
  private videoBudgetGranted: boolean = false;
  /**
   * Every gesture goes through here, and it owns the pointer position. See
   * humanize.ts — the solver names gestures, this decides what events they are
   * and how long they take.
   */
  private human: Humanizer;
  private imageCounter: number = 0; // Track images sent to CLI for debugging
  private sessionDebugDir: string | null = null;
  // onStep instrumentation: monotonic step index + solve-start wall clock.
  // Reset at the top of each solveImpl() so indices/elapsed are per-solve.
  private stepIndex: number = 0;
  private solveStartMs: number = 0;
  // Dedicated dump dir for the reCAPTCHA 3x3 dynamic driver — frames + a JSONL
  // state log so the click/fade/wait flow can be diagnosed offline. Always set
  // (independent of CAPTCHA_DEBUG) so we capture the hard-to-reproduce timing.
  private gridDebugDir: string | null = null;
  private gridDebugSeq: number = 0;
  // Persistent CV worker (`python -m captchakraken.cli serve`) — one long-lived process
  // that answers find-grid / grid-cell-states polls over stdin/stdout, so the
  // hot poll loops pay one ~0.4s interpreter+cv2 import ONCE instead of per poll.
  private cvWorker: ChildProcessWithoutNullStreams | null = null;
  private cvWorkerReady: Promise<boolean> | null = null;
  private cvWorkerSeq: number = 0;
  private cvWorkerPending: Map<number, { resolve: (v: any) => void; reject: (e: any) => void }> = new Map();
  private cvWorkerBuf: string = '';
  // Per-solve cache of model responses keyed by (screenshot content hash +
  // puzzle source + retry mode). If the page hasn't changed since we last asked
  // the model about it, re-querying is wasted work — reuse the prior answer.
  // Cleared at the top of each solve. See getSolution().
  private solutionCache: Map<string, CliResponse> = new Map();
  // Set when a screenshot we have ALREADY answered comes back this solve.
  //
  // Every answer getSolution returns is executed — there is no speculative
  // call — so an identical picture on a later round cannot mean "nothing
  // changed, reuse it". It means the answer we already tried changed nothing.
  // On a board that CYCLES (GeeTest's svg variant advances through screens and
  // dwells ~1.5s on each) the pixels come back around, the settle probe reads
  // that dwell as static, and the driver replayed a failed answer every round:
  // 81 solve loops, 12 model calls, 69 cache hits, 0 solves across 16 live
  // attempts. So a repeat is treated as EVIDENCE the still reading was wrong,
  // and the challenge is re-solved from a recorded burst. See
  // repeated-answer.test.ts.
  private repeatedAnswerSeen = false;
  /**
   * The ONE recording and the ONE answer for the animated challenge on screen.
   *
   * A cycling board needs a single inference and no more. The burst shows the
   * model every screen; the answer names the cell AND the screen it is on
   * (`frame` / `await_keyframe`); the gate then holds the click until that
   * screen is back up. Nothing about that changes between rounds — the board
   * keeps cycling through the same pictures with the same target — so asking
   * again buys nothing and costs a burst plus a multi-image inference.
   *
   * It was asking again every round. Loops 3, 4 and 5 each re-recorded and
   * re-inferred, which is what exhausted a 66s budget on a puzzle whose whole
   * task is one click.
   *
   * Dropped when the gate reports it never saw the chosen screen: that means
   * the widget is no longer the board this plan was made for, which is exactly
   * when a fresh recording is warranted.
   */
  private animatedPlan: { burstDir: string; response: CliResponse } | null = null;
  // Slicing mode of the burst the current answer came from, as reported by
  // `solve-animated`. `waitForKeyframe` reads it: `even` means the slicer found
  // no state that RECURS, so there is nothing for the page to come back to and
  // the gate can only ever time out. See test_even_clips_do_not_wait.py.
  private keyframeMode: string | null = null;
  /**
   * How many distinct steady screens the current clip sits on; 0 if it is not
   * that kind of clip. THE GATE KEYS ON THIS, NOT ON `keyframeMode`.
   *
   * `even` was being read as "no state recurs, do not wait". It does not mean
   * that — it means the slicer could not PROVE recurrence, which for a board
   * whose loop is longer than a 4s burst it never can. 32 of 60 real clips
   * sliced `even` while sitting on 2-3 steady screens, so the gate was off on
   * every real animated captcha and the driver clicked whichever screen was up.
   */
  private keyframeSteadyScreens = 0;
  /**
   * When the current solve must be over, in `Date.now()` terms. 0 outside one.
   *
   * The budget used to be checked ONLY at the top of each solve loop, so any
   * single round could overrun it without limit — and an animated round is the
   * long one: a 4s burst of 40 element screenshots, a multi-image inference,
   * then a gated click per target. Measured on GeeTest svg, 2026-09-07: rounds
   * that re-recorded ran past a 66s budget to the harness's own 180s ceiling
   * and were reported "hung, not slow", which is a message about the model for
   * a driver that simply never looked at the clock.
   *
   * The Python port has always checked mid-wait (`_check_deadline`), so this
   * was also a cross-port divergence on exactly the puzzles Tier 3 times.
   */
  private solveDeadlineAt = 0;
  // Repeat detection; see `maxNoProgressRounds`. `lastAnswerSig` is the answer
  // the previous round EXECUTED, `noProgressRounds` how many rounds running
  // have re-executed it.
  private lastAnswerSig: string | null = null;
  private noProgressRounds = 0;
  // Current challenge lifecycle state (see CaptchaState). Diagnostic + used to
  // gate behaviours; transitions are logged via gridDebug when CAPTCHA_DEBUG=1.
  private state: CaptchaState = CaptchaState.Detecting;
  // Content hash of the challenge frame at the moment we last clicked Submit.
  // On the next attempt we wait for the frame to CHANGE from this (the expected
  // post-submit transition) before treating it as a fresh puzzle — so the shift
  // itself is never screenshotted and mis-read as a blank/unsupported frame.
  // Cleared once consumed. See solveSingle().
  private lastSubmitFrameHash: string | null = null;
  // Groups every inference round of ONE captcha into a single attempt for the
  // hosted API. A dynamic reCAPTCHA 3x3 re-solves after each click round, so one
  // solve() can fire up to `recaptchaMaxDynamicRounds` model calls; sharing a
  // session id lets the gateway bill them as one capped attempt rather than N
  // independent ones. Null outside a solve; ignored entirely when self-hosting.
  private solveSessionId: string | null = null;

  /** Interpreter + CLI root, resolved once. See resolveCli. */
  private cliCache: { cliRoot: string; py: string } | null = null;

  /** The LoRA name models.json calls `latest`, read once. See askModel. */
  private loraNameCache: string | null = null;

  /**
   * Phase accounting for the CURRENT or MOST RECENT solve; see timing.ts.
   * Null before the first solve, and readable after a failed one.
   */
  budget: PhaseBudget | null = null;

  /**
   * Attribute the enclosed wall-clock to `name` for this solve's budget.
   *
   * A no-op outside a solve, so every helper stays callable from tests and from
   * the watcher without a budget having been opened.
   */
  private ph<T>(name: string, fn: () => Promise<T>): Promise<T> {
    return this.budget ? this.budget.phase(name, fn) : fn();
  }

  constructor(config: CaptchaKrakenConfig = {}) {
    this.config = config;
    this.human = resolveHumanizer({
      ...config,
      startingMousePosition: config.startingMousePosition ?? { x: 100, y: 100 },
    });
  }

  /**
   * Where the pointer is. Owned by the humanizer, because a mode that
   * dispatches no motion (mobile, between taps) still has to answer this.
   */
  private get lastMousePosition(): Vector {
    return { x: this.human.at[0], y: this.human.at[1] };
  }

  private set lastMousePosition(v: Vector) {
    this.human.at = [v.x, v.y];
  }

  async solve(page: Page): Promise<SolveResult | void> {
    this.solveSessionId = randomUUID();
    this.budget = new PhaseBudget();
    // FALSE unless the solve returns a solved result. A solve that throws never
    // sets it, which is the right default: the widget did not accept, and the
    // boards that made it throw are exactly the ones worth keeping.
    let solvedForReport = false;
    try {
      const result = await this.solveImpl(page);
      // Attached HERE rather than at each of the four `isSolved:` sites, so a
      // new early exit cannot forget it.
      if (result) result.phases = this.budget.toObject();
      solvedForReport = result?.isSolved === true;
      return result;
    } finally {
      // Printed on the way out of every solve, success or failure — a solve
      // that FAILED is exactly the one whose time you want itemised.
      if (timingsEnabled()) console.error(this.budget!.report());
      // NOT cleared. A solve that FAILED is exactly the one whose time you want
      // itemised, and by the time the caller catches the error the result — the
      // only thing `phases` rides on — does not exist. `solve()` replaces it on
      // entry, so the next solve cannot read the last one's. Same lifetime as
      // page_solver.py's `_budget`.
      // Always shut the persistent CV worker down when a solve ends (success,
      // failure, or timeout) so we never leak a python process between solves.
      this.teardownCvWorker();
      this.cvWorkerReady = null;
      // BEFORE the id is cleared, so the report names this solve.
      this.reportOutcome(this.solveSessionId, solvedForReport);
      // Clear last: a stale id leaking into the NEXT solve would merge two
      // separate captchas into one billable attempt.
      this.solveSessionId = null;
    }
  }

  /**
   * Tell the hosted API whether the widget accepted — the one fact the server
   * cannot see for itself, since a wrong answer reaches it as a 200 with
   * well-formed JSON in it.
   *
   * THROUGH THE PYTHON CLI, not a `fetch` here. The endpoint, the credential,
   * the extra-header handling and the "one 404 and stop asking" rule all live
   * in `planner.report_outcome`; a second copy in TypeScript would agree with
   * it until one of them was edited. That is the drift `model-name.ts` exists
   * to prevent, on the same seam, and this port already crosses that subprocess
   * boundary for every model call.
   *
   * NOT AWAITED, and detached. The solve is over and its result is decided, so
   * nothing here may add latency to what the caller gets back or keep the
   * process alive past it. `unref()` is what makes the second true.
   */
  private reportOutcome(sessionId: string | null, solved: boolean): void {
    if (!sessionId) return;
    // CHECKED HERE, NOT ONLY IN PYTHON. The Python side honours the same
    // variable, but this port spawns a PROCESS to ask it — and against an
    // endpoint with no such route (a local vLLM, every fixture run) that
    // process exists only to be told 404. Python's "one 404 and stop asking"
    // latch cannot help here either, because each solve is a fresh process.
    // Tier 3 sets this, and drives ~100 solves per run on one box.
    if (process.env.CAPTCHA_REPORT_OUTCOME === '0') return;
    try {
      const { cliRoot, py } = this.resolveCli();
      const child = spawn(
        py,
        ['-m', 'captchakraken.cli', 'report-outcome', sessionId, solved ? 'solved' : 'failed'],
        { cwd: cliRoot, env: cliEnv(cliRoot), detached: true, stdio: 'ignore' },
      );
      // A report that cannot be sent is not the caller's problem, and an
      // unhandled 'error' event on a spawn is an uncaught exception in Node.
      child.on('error', () => {});
      child.unref();
    } catch {
      // resolveCli throws when the bundled engine is missing, which is already
      // a loud failure everywhere it matters. Not here.
    }
  }

  /**
   * Install an auto-solver: probe `page` on a timer and solve any captcha that
   * becomes visible, until `stop()`.
   *
   * Returns immediately — the watching happens in the background, so your
   * automation carries on and captchas are handled underneath it.
   *
   * Nothing is injected into the page on any platform; see watcher.ts for why,
   * and for the note on Camoufox, where the probe's DOM reads land in its
   * isolated world by default.
   *
   * ```typescript
   * const solver = new CaptchaKrakenSolver();
   * const watcher = solver.watch(page);
   * await page.goto('https://example.com/protected');   // solved as it appears
   * await watcher.stop();
   * ```
   */
  watch(page: Page, options: WatchOptions = {}): CaptchaWatcher {
    return watchPage(this, page, options);
  }

  private async solveImpl(page: Page): Promise<SolveResult | void> {
    const maxSolveLoops = this.config.maxSolveLoops ?? SOLVE_DEFAULTS.maxSolveLoops;
    const postSolveDelayMs = this.config.postSolveDelayMs ?? 1200;
    const overallSolveTimeoutMs =
      this.config.overallSolveTimeoutMs ?? SOLVE_DEFAULTS.overallSolveTimeoutMs;

    const start = Date.now();
    let cumulativeTokenUsage: TokenUsage[] = [];
    this.imageCounter = 0;
    this.stepIndex = 0;
    this.solveStartMs = start;
    await this.human.reset(page);
    this.resetSolveState();
    this.setState(CaptchaState.Detecting);

    // Initialize session debug directory if debugging is enabled
    if (process.env.CAPTCHA_DEBUG === '1') {
      const cliRoot = this.config.repoPath ?? getBundledCliRoot();
      const debugRunsDir = path.join(cliRoot, '..', 'debug_runs');
      if (!fs.existsSync(debugRunsDir)) {
        fs.mkdirSync(debugRunsDir, { recursive: true });
      }
      const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, -5);
      this.sessionDebugDir = path.join(debugRunsDir, `solve_${timestamp}`);
      fs.mkdirSync(this.sessionDebugDir, { recursive: true });
      log(`Session debug directory: ${this.sessionDebugDir}`);
    }

    // Set to "missed-tiles" for the next iteration when we detect that the
    // captcha vendor rejected our submission with an under-selection error
    // (reCAPTCHA "Please select all matching images"). Used once, then
    // cleared. If the error appears again after the retry, we abort —
    // burning loops on a stuck model only delays the inevitable fail.
    let pendingRetryMode: string | null = null;
    let alreadyRetriedRecaptchaError = false;
    // A blank/transitioning frame that slips past the settle gate can make the
    // model return "unsupported". If we've already interacted (so we're mid
    // multi-round, not on a genuinely unsupported first frame), actively wait
    // for the challenge to settle and retry — up to this budget — before
    // declaring the whole puzzle unsupported. One retry isn't enough when the
    // next round loads slowly (that's why some solves failed round 2 and others
    // didn't — it was a race).
    let unsupportedRetries = 0;
    const maxUnsupportedRetries = this.config.maxUnsupportedReSolves ?? 3;
    // A stale/detached challenge handle: after a submit, hCaptcha swaps the
    // challenge iframe for the next round while we still hold the old handle, so
    // a screenshot on it hangs then fails "not visible". Not a real failure —
    // re-detect the fresh challenge and retry, up to this budget.
    let staleElementRetries = 0;
    const maxStaleElementRetries = this.config.maxStaleElementRetries ?? 3;
    // Track whether we've interacted with the captcha at least once. Before any
    // interaction, a null detectCaptcha() means "not rendered yet", not "solved".
    let hasInteracted = false;
    // Bounded wait for an in-DOM-but-still-rendering widget (Stage 1).
    let renderWaits = 0;
    // Strictly FEWER than the solve loops. A render wait consumes an attempt,
    // so at parity the loop runs out first and the "no interactive captcha
    // widget" branch below never fires — the correct, benign answer for a
    // reCAPTCHA v3 / invisible page gets reported as "still detected after N
    // solve loops" instead. Mirrors page_solver.py's max_render_waits.
    const MAX_RENDER_WAITS = Math.min(6, maxSolveLoops - 1);

    for (let attempt = 1; attempt <= maxSolveLoops; attempt++) {
      // `videoBudgetMs` is 0 until this solve records something, so a solve
      // that never escalates gets exactly the deadline the caller configured.
      const budgetMs = overallSolveTimeoutMs + this.videoBudgetMs;
      // Published so the burst and the keyframe gate can stop on their own
      // rather than overrunning and being noticed a round later.
      this.solveDeadlineAt = start + budgetMs;
      if (Date.now() - start > budgetMs) {
        throw new Error(
          `Captcha solve timed out after ${budgetMs}ms (attempt ${attempt}/${maxSolveLoops})`
          + (this.videoBudgetMs ? `, including ${this.videoBudgetMs}ms granted for recording an animated challenge` : '')
          + '.');
      }

      /*
       * ASK THE VENDOR BEFORE DOING ANY MORE WORK.
       *
       * Once we have interacted, the cheapest possible question is "did that
       * already work?" — a hidden response field on the page, no iframe, no
       * screenshot, microseconds. Skipping it meant a solved reCAPTCHA still
       * paid a full re-detect on the challenge iframe the vendor was tearing
       * down: settle probe, grid poll, each screenshot waiting on an element
       * that would never go stable again, ~10s after the answer was already
       * accepted.
       *
       * Gated on hasInteracted for the same reason the post-interaction branch
       * below is: before we touch anything, a response token belongs to
       * somebody else's earlier solve and says nothing about this one.
       */
      if (hasInteracted && await this.isCaptchaSolved(page)) {
        console.log('Vendor reports solved; returning without another detect pass.');
        return {
          isSolved: true,
          finalMousePosition: this.lastMousePosition,
          tokenUsage: aggregateTokenUsage(cumulativeTokenUsage),
        };
      }

      const captchaElement = await this.ph('detect', () => this.detectCaptcha(page));
      if (!captchaElement) {
        // Two-stage detection. detectCaptcha() returns null when there's no
        // VISIBLE, unsolved widget — but that splits into two cases:
        if (hasInteracted) {
          // We already clicked/solved something and now nothing actionable
          // remains → treat as solved.
          console.log('No supported captcha found (post-interaction); considering solved.');
          return {
            isSolved: true,
            finalMousePosition: this.lastMousePosition,
            tokenUsage: aggregateTokenUsage(cumulativeTokenUsage)
          };
        }

        // No interaction yet. Stage 1: is an interactive widget present in the
        // DOM but simply not finished rendering?
        if (await this.hasInteractiveWidgetInDom(page) && renderWaits < MAX_RENDER_WAITS) {
          renderWaits++;
          console.log(
            `Captcha widget present in DOM but not yet rendered; waiting `
            + `(${renderWaits}/${MAX_RENDER_WAITS}).`
          );
          await delay(800 + Math.random() * 300);
          continue;
        }

        // No interactive widget in the DOM (reCAPTCHA v3 / invisible, or an
        // hCaptcha that only triggers on user action), or it never rendered.
        // Fail fast rather than burning the whole loop budget.
        throw new Error(await this.noWidgetMessage(page));
      }

      console.log(`\n--- Captcha Solve Loop ${attempt}/${maxSolveLoops} ---`);
      const retryModeThisLoop = pendingRetryMode;
      pendingRetryMode = null;

      let didInteract: boolean;
      let tokenUsage: TokenUsage[];
      try {
        ({ didInteract, tokenUsage } = await this.solveSingle(
          page, captchaElement, attempt, retryModeThisLoop,
        ));
      } catch (e: any) {
        // `.animated` no longer means "the challenge moves" — moving challenges are
        // recorded and solved from keyframes. It now means the RECORDING itself was
        // impossible (the element refused to screenshot), or that the caller turned
        // the path off with `videoSolveEnabled: false`. Either way there is nothing
        // left to try.
        if (e?.animated) {
          this.setState(CaptchaState.Animated);
          throw new Error(
            `Animated challenge could not be solved: ${e.message ?? 'recording failed'}`
          );
        }
        // Stage 2: we screenshotted a settled frame and the CLI says the puzzle
        // TYPE is unsupported (e.g. hCaptcha click/drag). Normally a definitive
        // verdict — fail fast. BUT if we've already interacted, a transient
        // blank/transition frame can still produce this; re-settle + retry once
        // before giving up (fixes the "solves round 1, dies on round 2" case).
        if (e?.unsupported) {
          if (hasInteracted && unsupportedRetries < maxUnsupportedRetries) {
            unsupportedRetries++;
            // Almost certainly a not-yet-settled next round. Actively wait for
            // it to settle before retrying; if it never settles it's animated.
            const el = await this.detectCaptcha(page);
            if (el) {
              const s = await this.ph('settle', () => this.waitForElementSettled(el));
              if (s === 'animated') {
                // Used to be terminal. Now it just means the next round is an
                // animated puzzle: retry the loop and solveSingle takes the
                // recording path. `unsupportedRetries` still bounds it, so a
                // widget that is animated AND unsolvable cannot spin here.
                if (this.config.videoSolveEnabled === false) {
                  this.setState(CaptchaState.Animated);
                  throw new Error(
                    'Animated/video challenge detected — the puzzle never settles '
                    + 'and videoSolveEnabled is off.'
                  );
                }
                console.log(
                  '"unsupported" mid-solve and the next round is animated; '
                  + 'retrying into the recording path.',
                );
                continue;
              }
            }
            console.log(
              `"unsupported" mid-solve (not-yet-settled next round); settled and `
              + `retrying (${unsupportedRetries}/${maxUnsupportedRetries}).`,
            );
            continue;
          }
          // The solver's OWN message, not a guess about what it saw. This
          // used to substitute "likely an hCaptcha click/drag puzzle" for
          // every unsupported verdict, including the ones that already said
          // exactly what was wrong and how to fix it. A wrong guess reported
          // in place of a right answer costs whoever reads the gate an
          // investigation, every time.
          throw new Error(`Cannot solve this kind of captcha — ${e?.message ?? e}`);
        }
        // Stale/detached challenge handle: hCaptcha swapped in the next round
        // while we held the old iframe, so a screenshot on it fails "not
        // visible" / "Timeout" / "not attached". This is a transition, not a
        // dead puzzle — back off, then let the loop re-detect the fresh
        // challenge. (Only after we've interacted; a first-frame failure is a
        // genuine problem worth surfacing.)
        const emsg = String((e && (e as any).message) || e);
        if (
          hasInteracted
          && staleElementRetries < maxStaleElementRetries
          && isStaleHandleError(emsg)
        ) {
          staleElementRetries++;
          console.log(
            `stale challenge handle after submit ("${emsg.split('\n')[0]}"); `
            + `re-detecting next round (${staleElementRetries}/${maxStaleElementRetries}).`,
          );
          await delay(this.config.staleElementBackoffMs ?? 900);
          continue;
        }
        throw e;
      }
      hasInteracted = hasInteracted || didInteract;

      // Give up on a solve that is repeating itself, rather than letting the
      // clock do it. The alternative is not "one more chance" — it is the same
      // click, again, until overallSolveTimeoutMs.
      if (this.noProgressRounds >= (this.config.maxNoProgressRounds ?? 2)) {
        throw new Error(
          `No progress: the model returned the same answer ${this.noProgressRounds + 1} `
          + `times running and the challenge is still up `
          + `(attempt ${attempt}/${maxSolveLoops}). Total usage: `
          + JSON.stringify(aggregateTokenUsage(cumulativeTokenUsage))
        );
      }
      renderWaits = 0;
      cumulativeTokenUsage.push(...tokenUsage);

      // ONE wait after a round, polled, whichever kind of round it was.
      //
      // There used to be two: this poll when the round interacted, and a flat
      // `delay(postSolveDelayMs + jitter)` when it did not. The flat sleep
      // observed NOTHING — it ran to the end and only then asked whether the
      // widget was still there — so a round that had in fact finished the
      // captcha paid 1200-1500ms to find that out. The poll below already asks
      // exactly that question and answers it the moment it becomes true.
      //
      // The two windows keep different LENGTHS, because different things size
      // them: postSolveOutcomeTimeoutMs covers how late a vendor's success
      // signal can arrive (measured — see types.ts), while postSolveDelayMs is
      // the dwell a round that answered nothing takes before deciding the page
      // has settled. Mirrors page_solver.py; the ports must not disagree about
      // how long a round costs.
      //
      // hCaptcha keeps the challenge iframe visible for a couple of seconds
      // while it verifies the final answer; without this window the loop
      // re-entered the pipeline on that closing frame and burned ~18s. It ONLY
      // early-returns on a definitive signal, so it cannot loop.
      const settleMs = didInteract
        ? (this.config.postSolveOutcomeTimeoutMs ?? 1000)
        : postSolveDelayMs + Math.random() * 300;
      {
        const deadline = Date.now() + settleMs;
        const verdictT0 = Date.now();
        let solved = false;
        let widgetGone = 0;
        while (Date.now() < deadline) {
          if (await this.isCaptchaSolved(page)) { solved = true; break; }
          // The eight inline vendors have NO response token, so
          // `isCaptchaSolved` — which reads only the hCaptcha and reCAPTCHA
          // anchors — can never fire for them, and this loop ran out its whole
          // window on EVERY round waiting for a signal that cannot arrive.
          // "The widget is gone" is their completion signal and is already the
          // authority immediately after this loop, so this only reaches the
          // same verdict sooner — confirmed over two polls so a frame caught
          // mid-swap between rounds cannot read as a solve. page_solver.py has
          // had this since the geetest_v4_slide measurement (5.2s of a 12.3s
          // solve, spent after the puzzle was already answered); this port had
          // not, which is most of why it measured slower on those vendors.
          if (!(await this.detectCaptcha(page))) {
            widgetGone += 1;
            if (widgetGone >= 2) { solved = true; break; }
          } else {
            widgetGone = 0;
          }
          // A fresh next round has rendered → stop waiting, go solve it now
          // (keeps multi-round solves fast instead of burning the full window).
          if (await this.isChallengeFreshlyRendered(page)) break;
          await delay(this.config.postSolveOutcomePollMs ?? 75);
        }
        this.budget?.add(didInteract ? 'await-verdict' : 'post-submit-delay',
                         Date.now() - verdictT0);
        // How long a SUCCESS actually took to show itself — the only number
        // that can size postSolveOutcomeTimeoutMs. A round that ends any other
        // way spends the whole window by construction.
        if (solved) console.log(`[verdict] success signal arrived after ${Date.now() - verdictT0}ms`);
        if (solved) {
          return {
            isSolved: true,
            finalMousePosition: this.lastMousePosition,
            tokenUsage: aggregateTokenUsage(cumulativeTokenUsage),
          };
        }
      }

      // Detect reCAPTCHA's under-selection error banner. If present, the
      // vendor rejected our last submission because the model missed at
      // least one matching tile. Set the retry flag for the next loop so
      // the CLI augments the grid prompt with an explicit "you missed
      // some" instruction. If we've already retried once and the error is
      // STILL showing, bail — the model is stuck and the loop will just
      // keep flipping between "done" and Verify until timeout.
      const banner = await this.recaptchaBannerKind(page);
      if (banner && this.bannerIsFatalAfterRetry(banner)) {
        if (alreadyRetriedRecaptchaError) {
          throw new Error(
            'reCAPTCHA still showing the under-selection error after retry; '
            + 'aborting (model unable to identify the missed tile). Total usage: '
            + JSON.stringify(aggregateTokenUsage(cumulativeTokenUsage))
          );
        }
        console.log('reCAPTCHA returned under-selection error; retrying with missed-tiles prompt.');
        pendingRetryMode = 'missed-tiles';
        alreadyRetriedRecaptchaError = true;
      }

      const after = await this.detectCaptcha(page);
      if (!after) {
        return {
          isSolved: true,
          finalMousePosition: this.lastMousePosition,
          tokenUsage: aggregateTokenUsage(cumulativeTokenUsage)
        };
      }

      // If we didn't actually interact and captcha is still detected, don't spin forever.
      if (!didInteract) {
        throw new Error(`Captcha still detected but solver performed no interactions; aborting to avoid an infinite loop. Total usage: ${JSON.stringify(aggregateTokenUsage(cumulativeTokenUsage))}`);
      }
    }

    throw new Error(`Captcha still detected after ${maxSolveLoops} solve loops. Total usage: ${JSON.stringify(aggregateTokenUsage(cumulativeTokenUsage))}`);
  }

  /**
   * Fire the optional onStep observer with a fresh screenshot of the captcha
   * element. No-op (beyond a cheap early return) when no callback is set, so it
   * stays off the critical path in normal runs. The emitted PNG is owned by the
   * callback — we never delete it. Best-effort: a screenshot or callback error
   * never fails the solve.
   */
  private async emitStep(
    captchaElement: ElementHandle,
    stage: SolveStepEvent['stage'],
    label: string,
    puzzleSource: SolveStepEvent['puzzleSource'],
    frameRole: SolveStepEvent['frameRole'],
    attempt: number,
    meta?: Record<string, any>,
  ): Promise<void> {
    const cb = this.config.onStep;
    if (!cb) return;
    this.stepIndex++;
    let screenshotPath: string | null = path.join(
      os.tmpdir(),
      `step_${this.stepIndex}_${Date.now()}_${Math.floor(Math.random() * 1e9)}.png`,
    );
    try {
      // ITS OWN, SHORT TIMEOUT — not `elementScreenshotTimeoutMs`, which is
      // sized for the picture the MODEL reads and is 8s. This one is an
      // observability snapshot, and `animations: 'disabled'` makes Playwright
      // wait for the element to stop moving before it will take it. On a widget
      // that is still animating that wait ran to the full 8s, per step:
      // measured 8.0s of a 12.0s mtcaptcha_text solve, spent photographing a
      // text box for a trace. An observer must never cost more than the action
      // it is observing, and a missed frame in a trace costs nothing.
      await captchaElement.screenshot({
        path: screenshotPath,
        timeout: this.config.stepScreenshotTimeoutMs ?? 2000,
        animations: 'disabled',
      });
    } catch {
      screenshotPath = null;
    }
    try {
      await cb({
        index: this.stepIndex,
        stage,
        label,
        screenshotPath,
        puzzleSource,
        frameRole,
        attempt,
        elapsedMs: this.solveStartMs ? Date.now() - this.solveStartMs : 0,
        meta,
      });
    } catch (e: any) {
      log(`onStep callback threw (ignored): ${e?.message ?? e}`);
    }
  }

  private async solveSingle(page: Page, captchaElement: ElementHandle, attempt: number, retryMode: string | null = null): Promise<{ didInteract: boolean, tokenUsage: TokenUsage[] }> {
    // Vendor hint constrains which grid SHAPES the CLI may solve a detection as.
    // find_grid false-positives on the header+footer bands of hCaptcha's click
    // puzzles, and hCaptcha ships only a 3x3 — so a 16-cell lattice on one is a
    // contradiction and is dropped back to the click path. It is not a blanket
    // skip: hcaptcha_grid_3x3_property is a real grid and still solves as one.
    // Anything that is not hCaptcha or reCAPTCHA reports 'unknown' and is
    // allowed every shape (GeeTest and Prosopo both ship real 3x3 grids).
    const src = await captchaElement.getAttribute('src').catch(() => null);
    const puzzleSource = src && src.includes('hcaptcha.com')
      ? 'hcaptcha'
      : src && src.includes('recaptcha/api2')
        ? 'recaptcha'
        : 'unknown';

    // Distinguish the anchor "I'm not a robot" checkbox from the open image
    // challenge so recorders can drop the (useless) pre-challenge checkbox
    // screenshots and keep only the real puzzle. reCAPTCHA: anchor = api2/anchor,
    // challenge = api2/bframe. hCaptcha: anchor = frame=checkbox, challenge =
    // frame=challenge. (Note puzzleSource alone can't tell hCaptcha's checkbox
    // from its challenge — both srcs contain hcaptcha.com.)
    const frameRole: SolveStepEvent['frameRole'] =
      !src ? 'unknown'
        : src.includes('recaptcha/api2/bframe') || src.includes('frame=challenge')
          ? 'challenge'
          : src.includes('recaptcha/api2/anchor') || src.includes('frame=checkbox')
            ? 'checkbox'
            : 'unknown';

    // Everything the answer might have to be delivered INTO — a text box, a
    // slider handle — is looked up against this, never against the page. For
    // the iframed vendors it is the challenge document; for the ones that
    // render into the host page (GeeTest, Yidun, BotDetect, …) it is the widget
    // element, whose subtree is the same boundary.
    const scope: Frame | ElementHandle = (await captchaElement.contentFrame()) ?? captchaElement;

    // Does this puzzle want a STRING rather than a place to click? Only the DOM
    // can say. The picture cannot: BotDetect's warped code and hCaptcha's
    // "click the matching character" are the same genre of image and want
    // opposite answers. Restricted to `unknown` because neither hCaptcha nor
    // reCAPTCHA has ever served a typed challenge, so a match inside one of
    // their frames would be a false positive by definition.
    const textMode = !VENDORS_WITH_BESPOKE_HANDLING.has(puzzleSource)
      && (await this.answerBox(scope, captchaElement)) !== null;
    if (textMode) {
      console.log('Widget has a text box; solving as a distorted-text captcha.');
    }

    // hCaptcha swaps the challenge images in asynchronously — the iframe is
    // "visible" the instant the frame opens, but the task tiles paint a beat
    // later, and on multi-round puzzles it REUSES the same iframe: after a
    // submit it briefly shows the previous round, then a loading spinner, then
    // the next round. Screenshotting any of those transitional frames feeds the
    // model a blank/stale grid it correctly reports as "unsupported" — which
    // used to abort the whole solve on round 2. Gate on the challenge state:
    //   1. If we just submitted, wait for the frame to actually CHANGE (the
    //      transition starting) so we're past the previous round.
    //   2. Wait for the tiles to paint (DOM) AND for the pixels to stop moving
    //      (settle monitor). If it never settles, it's an animated/video puzzle.
    let isAnimated = false;
    if (puzzleSource === 'hcaptcha' && src && src.includes('frame=challenge')) {
      if (this.lastSubmitFrameHash) {
        this.setState(CaptchaState.Transitioning);
        await this.ph('await-next-round', () => this.waitForChangeSince(captchaElement, this.lastSubmitFrameHash as string));
        this.lastSubmitFrameHash = null;
      }
      this.setState(CaptchaState.Loading);
      await this.ph('hcaptcha-images', () => this.waitForHcaptchaChallengeImages(captchaElement));
      const settle = await this.ph('settle', () => this.waitForElementSettled(captchaElement));
      if (settle === 'animated') {
        // A challenge that never settles is animated BY DESIGN — hCaptcha's
        // "select the odd animal" fades its sprites on independent cycles, and
        // "unique motion pattern" spins identical meshes. This used to end the
        // solve; it now routes to the recording path below.
        if (this.config.videoSolveEnabled === false) {
          const e: any = new Error(
            'ANIMATED_CHALLENGE: the challenge never settles and videoSolveEnabled is off.',
          );
          e.animated = true;
          throw e;
        }
        console.log('[animated] challenge never settles — recording it');
        isAnimated = true;
      }
      // Those three waits total ~21s and none of them watches for success, so
      // the vendor's token routinely lands DURING them. Ask once more before
      // spending an inference: last free moment to notice the captcha is
      // already accepted, and inference is the most expensive step in the loop.
      if (await this.isCaptchaSolved(page)) {
        console.log('[captchakraken] solved while waiting for the next round; skipping inference.');
        return { didInteract: false, tokenUsage: [] };
      }
      this.setState(CaptchaState.Ready);
    } else if (!VENDORS_WITH_BESPOKE_HANDLING.has(puzzleSource)
               && this.config.videoSolveEnabled !== false) {
      // Non-hCaptcha, non-reCAPTCHA widgets (GeeTest, Tencent, …). The settle probe
      // was never run for these, so an animated one — GeeTest's svg board cycles its
      // glyph set — was screenshotted mid-cycle and answered from whatever single
      // moment we happened to catch. reCAPTCHA is excluded on purpose: it has its own
      // readiness gate below, its grids are never animated, and a second probe would
      // only add latency to a path that already works.
      if (await this.ph('settle', () => this.waitForElementSettled(captchaElement)) === 'animated') {
        console.log('[animated] challenge never settles — recording it');
        isAnimated = true;
      }
    }

    /*
     * A board that cycles but DWELLS defeats the settle probe: it holds each
     * screen still for far longer than the ~440ms of stillness that declares a
     * challenge static, so it is read as a picture and answered from whichever
     * screen we caught. The tell arrives a round later, when a screenshot we
     * have already answered comes back — proof that the answer ran and moved
     * nothing. Record it from here on instead of guessing at another still.
     */
    if (!isAnimated && this.shouldRetryAsAnimated(puzzleSource)) {
      console.log('[animated] a picture we already answered came back — recording it');
      isAnimated = true;
    }

    // Only the image-challenge frame (bframe) holds a grid. The anchor checkbox
    // (api2/anchor) has none — running the grid settle/detect on it just wastes
    // an 8s timeout + a find-grid subprocess before the checkbox click. Gate the
    // grid handling to the bframe.
    const isRecaptchaChallenge = puzzleSource === 'recaptcha'
      && !!src && src.includes('recaptcha/api2/bframe');

    // reCAPTCHA fades new tiles in over ~1s (initial load and the in-place
    // dynamic refresh after a click). Screenshotting mid-fade feeds the LoRA a
    // blank/partial grid. Poll until the grid's cells have settled before
    // grabbing the frame. Best-effort — falls through on timeout. The in-place
    // refresh re-enters solveSingle each loop, so this guard covers it too.
    // Grid size the solver establishes for this challenge, surfaced in the
    // baseline step's meta so callers (e.g. the demo recorder) can bucket
    // reCAPTCHA attempts into 3x3 vs 4x4 without scraping debug logs.
    let establishedGridSize: number | null = null;
    if (isRecaptchaChallenge) {
      await this.ph('grid-load', () => this.waitForGridCellsLoaded(captchaElement));
      // Only a 3x3 reCAPTCHA ever refreshes its tiles in place (blank/fade →
      // new image), so only a 3x3 can need the multi-round driver: click →
      // hover/wait for fades → re-solve. Whether THIS one does is decided inside
      // the driver, by what the widget does with the first click. A 4x4 never
      // refreshes and is one-shot like hCaptcha: click all matching tiles, then
      // submit in the same pass. Falls through if the grid can't be established.
      const grid = await this.getGridBoxes(captchaElement);
      if (grid && grid.size === 3) {
        establishedGridSize = 3;
        const elementBox = await captchaElement.boundingBox();
        if (elementBox) {
          return this.solveRecaptchaGrid(page, captchaElement, attempt, retryMode, grid, elementBox);
        }
      } else if (grid && grid.size === 4) {
        establishedGridSize = 4;
      }
    }

    // 1. Take Screenshot
    const screenshotPath = path.join(os.tmpdir(), `captcha_${Date.now()}_${Math.floor(Math.random() * 1e9)}.png`);
    await this.ph('screenshot', () => captchaElement.screenshot({
      path: screenshotPath,
      timeout: this.config.elementScreenshotTimeoutMs ?? 8000,
      animations: 'disabled',
    }));

    // Save image to debug directory if debugging is enabled
    this.saveImageForDebug(screenshotPath);

    // Baseline screenshot before any action is taken. Emitted once per solve
    // (the first time we reach a one-shot/checkbox screenshot); later loops are
    // covered by the post-action 'submit'/'round' steps.
    if (this.stepIndex === 0) {
      await this.emitStep(captchaElement, 'initial', 'initial (pre-action)', puzzleSource, frameRole, attempt,
        establishedGridSize ? { gridSize: establishedGridSize } : undefined);
    }

    let performedAction = false;
    let slid = false;
    let placed = false;
    let clicked = false;
    let typed = false;
    let allTokenUsage: TokenUsage[] = [];
    let burstDir: string | null = null;

    try {
      // 2. Call CLI — while the model generates (the main idle window), drift
      //    the cursor over the challenge like a human weighing the options,
      //    instead of freezing it in place. Wrapped in the freshness guard: if
      //    the frame changes mid-inference (a tile fades in), the answer is for
      //    a stale frame, so we re-screenshot and re-solve on the developed one.
      let response: CliResponse;
      if (isAnimated) {
        // ONE burst, ONE inference, for as long as this board is on screen.
        if (this.animatedPlan) {
          burstDir = this.animatedPlan.burstDir;
          response = this.animatedPlan.response;
          console.log('[animated] reusing the recorded answer — same board, same screens');
        } else {
          burstDir = await this.ph('burst', () => this.recordKeyframeBurst(captchaElement));
          response = await this.ph('inference', () => this.withIdleWander(page, captchaElement, () =>
            this.getAnimatedSolution(burstDir as string)));
          this.animatedPlan = { burstDir, response };
        }
      } else if (this.shouldSpeculate(puzzleSource, textMode)) {
        /*
         * ASK AND WATCH AT THE SAME TIME.
         *
         * The screenshot goes to the model and the recording starts in the same
         * breath. Whichever the board turns out to be, the answer arrives about
         * one inference from now:
         *
         *   still  — the recording saw one picture, so the still answer stands
         *            and the frames are dropped. Identical to the old behaviour
         *            and identical in cost; the burst happened inside a wait we
         *            were making anyway.
         *   moving — the still answer is discarded UNREAD, because it describes
         *            a screen that has already gone. The burst runs on to the
         *            end of the cycle and the multi-image answer is used. Two
         *            inference CALLS, roughly one inference of wall-clock.
         *
         * The old shape learned the same thing in whole ROUNDS — answer a
         * still, act, notice nothing moved, answer another, then record — which
         * measured ~15s of a 40s solve.
         *
         * NO IDLE WANDER while speculating: the cursor drifts across the widget
         * during inference, and every frame of the burst would have a mouse
         * pointer in a different place. That is a new picture each time, which
         * reads as a board with a dozen screens.
         */
        const rec = this.startKeyframeBurst(captchaElement);
        let still: CliResponse | null = null;
        let stillError: unknown = null;
        try {
          still = await this.ph('inference', () => this.solveFrameFreshnessGuarded(
            captchaElement, screenshotPath,
            (imagePath) => this.getSolution(imagePath, puzzleSource, retryMode, textMode),
          ));
        } catch (e) {
          stillError = e;   // rethrown below only if the board turns out still
        }

        if (!rec.moved()) {
          await rec.abandon();
          if (stillError) throw stillError;
          response = still as CliResponse;
        } else {
          console.log(
            '[animated] the widget moved while the model was reading it — dropping '
            + 'the still answer and finishing the recording.',
          );
          isAnimated = true;
          this.repeatedAnswerSeen = true;   // so later rounds go straight to it
          burstDir = await this.ph('burst', () => rec.finish());
          response = await this.ph('inference', () => this.withIdleWander(page, captchaElement, () =>
            this.getAnimatedSolution(burstDir as string)));
          this.animatedPlan = { burstDir, response };
        }
      } else {
        response = await this.ph('inference', () => this.solveFrameFreshnessGuarded(
          captchaElement, screenshotPath,
          (imagePath) => this.withIdleWander(page, captchaElement, () =>
            this.getSolution(imagePath, puzzleSource, retryMode, textMode)),
        ));
      }
      const actions = response.actions;
      allTokenUsage = response.token_usage;

      // Archive debug artifacts if enabled
      this.archiveLatestDebugRun(attempt, actions);

      // 3. Execute Actions
      const actionList = Array.isArray(actions) ? actions : [actions];

      /*
       * The box translates NORMALISED coordinates into page ones, so it is
       * needed only by an action that carries coordinates. A `done` answer
       * carries none — it means "nothing left to click", and falls through to
       * the submit block below.
       *
       * DEMANDING IT UNCONDITIONALLY TURNED SUCCESSES INTO ERRORS. Several
       * vendors CLOSE the challenge the moment it is satisfied, so the widget
       * is legitimately gone by the time the model says `done` — and the throw
       * landed on the one answer that proves the solve worked. Measured
       * 2026-09-07 on gt4.geetest.com: loop 1 clicked three tiles, loop 2
       * returned `{"action":"done"}`, the panel went `display:none` because
       * GeeTest had accepted it, and the driver reported
       * "Could not get bounding box of captcha element". Every live GeeTest
       * attempt failed that way — 22 of 22 across four puzzles — and each was
       * recorded as the model being wrong.
       *
       * Without a box, a `done` answer now reaches the submit block and then
       * the next loop, where `detectCaptcha` finds nothing and the existing
       * post-interaction rule reports the solve. A coordinate action with no
       * box still throws, because that one genuinely cannot be performed.
       */
      const elementBox = await captchaElement.boundingBox();
      if (!elementBox && answerNeedsElementBox(actionList)) {
        throw new Error('Could not get bounding box of captcha element');
      }
      // Read at the point of use rather than asserted once: every reader below
      // sits inside a coordinate action's branch, which the guard above has
      // already made unreachable without a box. Keeping the check here means
      // the type is honest AND the original error survives for the case that
      // genuinely cannot be performed.
      const requireBox = () => {
        if (!elementBox) throw new Error('Could not get bounding box of captcha element');
        return elementBox;
      };

      // Recorded at the moment of EXECUTION, which is what makes a repeat mean
      // something: this exact answer is about to be performed, so if it matches
      // the last one, the last one already ran and the page is still asking the
      // same question.
      this.noteAnswer(actionList, retryMode);
      console.log(`Executing ${actionList.length} actions.`);
      const frame = await captchaElement.contentFrame();
      let verifyButton: ElementHandle | null = null;

      for (const action of actionList) {
        if (action.action === 'click') {
          const c = action as ClickAction;
          // v2 emits `target_bounding_boxes` (plural). v1 fields kept as fallbacks.
          const bboxes: Array<[number, number, number, number]> = c.target_bounding_boxes
            ?? (c.target_bounding_box ? [c.target_bounding_box] : []);
          if (!bboxes.length && !c.target_coordinates) {
            console.warn('Click action has no bboxes or coordinates', c);
            continue;
          }
          if (bboxes.length) {
            for (const bbox of bboxes) {
              // On an animated challenge, hold each click until the widget is back
              // in the state the model answered about. Per-click, not once per
              // action: these puzzles keep cycling, so by the time click 2 comes
              // round the state has moved on again.
              const one = { ...c, target_bounding_box: bbox } as ClickAction;
              if (c.await_keyframe) {
                await this.clickWhenFrameMatches(page, captchaElement, one, requireBox(), c.await_keyframe);
              } else {
                await this.executeClick(page, captchaElement, one, requireBox());
              }
              await this.human.pause('between');
            }
          } else {
            await this.executeClick(page, captchaElement, c, requireBox());
          }
          performedAction = true;
          clicked = true;
          await this.emitStep(captchaElement, 'click', `clicked ${bboxes.length || 1} target(s)`, puzzleSource, frameRole, attempt, { bboxes });
        } else if (action.action === 'drag' && !(action as DragAction).source_bounding_box) {
          // No source — a puzzle-piece slider. What you grab is not what has to
          // arrive, so this cannot go through executeDrag: pressing the gap the
          // model named and dragging from there picks up nothing at all.
          if (await this.executeSlide(page, captchaElement, scope, action as DragAction, requireBox())) {
            performedAction = true;
            slid = true;
            await this.emitStep(captchaElement, 'drag', 'slid the piece into the slot', puzzleSource, frameRole, attempt, { action });
          }
        } else if (action.action === 'drag') {
          const d = action as DragAction;
          // Wait on the SOURCE: the piece has to be there to be picked up. The
          // destination is not gated — by the time the mouse arrives the animation
          // has moved on regardless, and a drop is judged by where it lands, not by
          // what the slot looked like on pickup.
          if (d.await_keyframe && d.source_bounding_box) {
            await this.waitForKeyframe(captchaElement, d.await_keyframe, ...bboxCenter(d.source_bounding_box));
          }
          await this.executeDrag(page, captchaElement, action as any, requireBox());
          performedAction = true;
          placed = true;
          await this.emitStep(captchaElement, 'drag', 'drag', puzzleSource, frameRole, attempt, { action });
        } else if (action.action === 'type') {
          if (await this.executeType(page, scope, action as TypeAction, captchaElement)) {
            performedAction = true;
            typed = true;
            await this.emitStep(captchaElement, 'type', 'typed the code', puzzleSource, frameRole, attempt, { action });
          }
        } else if (action.action === 'wait') {
          if ((action as any).duration_ms > 0) {
            console.log(`Waiting for ${(action as any).duration_ms}ms as requested by CLI`);
            await delay((action as any).duration_ms);
            performedAction = true;
            await this.emitStep(captchaElement, 'wait', `waited ${(action as any).duration_ms}ms`, puzzleSource, frameRole, attempt, { action });
          }
        }
      }

      // Resolve the widget's submit control AFTER the action loop, not inside
      // it. An empty plan — the correct answer to reCAPTCHA 3x3's
      // `none_present` variation, where nothing matches and the control reads
      // SKIP — never enters that loop, so the lookup never ran, the press below
      // found `verifyButton` null, and the round aborted on 'performed no
      // interactions'. The finder was never the problem: 'Skip' has always been
      // in its list. It was simply never called for the one answer shape that
      // performs no other action.
      // `scope` when there is no vendor iframe. Eight vendors render into the
      // HOST PAGE — GeeTest, Yidun, Tencent, Yandex, Lemin, Prosopo,
      // MTCaptcha, BotDetect — so `contentFrame()` is null for all of them
      // and the button was never even SEARCHED FOR, while the text box and
      // the slider handle it sits beside were both found through `scope`
      // above. Two containers for two halves of one interaction.
      //
      // `scope` is the widget container and getVerifyButton's xpaths are
      // RELATIVE, so the submit of the FORM the captcha guards is out of
      // reach by construction. The press itself is bounded by
      // `shouldClickSubmit` below, which is where that hazard belongs.
      const lookup = frame ?? (slid ? null : scope);
      if (lookup) verifyButton = await this.getVerifyButton(lookup);
      // RESOLVED, not travelled to. `moveAndClick` below moves to the button
      // itself, and it picks its own random point inside it — so moving here
      // first bought a second humanised trajectory that ended a few dozen
      // pixels from where the first one stopped, plus a second bounded
      // scroll-into-view. Two hops to one button is slower AND is not
      // something a hand does. A round that decides not to submit no longer
      // walks to the control either.
      // 'done' actions fall through to the submit block below, same as before.

      // Submit policy: press the widget's own submit control whenever we have
      // put an ANSWER into it — a selection, a placed piece, a typed code — or
      // when we had nothing to do and want the round to advance.
      //
      // Two exclusions, and they are the whole rule:
      //
      //   a completed SLIDE has already submitted. Letting go of the handle is
      //     the gesture these puzzles grade; none of them ships a Verify
      //     button, so anything the generic finder turns up afterwards belongs
      //     to the HOST page, and pressing it would submit the form the captcha
      //     guards while the verdict is still in flight.
      //   a round that only WAITED has answered nothing.
      //
      // hCaptcha and the reCAPTCHA 4x4 used to be named here as one-shot
      // special cases; they are ordinary click rounds and this covers them.
      // (reCAPTCHA 3x3 never reaches here — solveRecaptchaGrid owns its
      // fade-and-re-round rounds.)
      //
      // A click round used to be excluded, on the reasoning that these boards
      // re-round and a half-made selection spends the attempt. They do not: the
      // ones that grade themselves mid-selection draw no submit control, so
      // verifyButton is null and nothing is pressed either way. What the
      // exclusion bought was an extra model call per puzzle, asking a board we
      // had already answered correctly whether it was `done`.
      const answered = clicked || placed || typed;
      const shouldClickSubmit = !slid && (answered || !performedAction);
      if (shouldClickSubmit && verifyButton) {
        console.log(performedAction
          ? `Actions executed; clicking Verify to submit (${puzzleSource}).`
          : 'No active actions performed (empty or done). Checking for Verify/Next button...');
        await this.moveAndClick(page, verifyButton);
        // The press IS an interaction, and saying so is load-bearing: the
        // caller aborts a round that reports none, so submitting a `done`
        // answer and then reporting false re-arms the very guard this
        // satisfies — the puzzle is sent and the solve gives up on it one line
        // later, which is what `prosopo_grid_3x3` did.
        performedAction = true;
        await this.emitStep(captchaElement, 'submit', 'submitted (Verify/Next)', puzzleSource, frameRole, attempt);
        // Snapshot the frame at submit time so the NEXT attempt waits for the
        // real transition (next round loading / frame closing) before treating
        // whatever is on screen as a fresh puzzle. See the hCaptcha gate above.
        this.setState(CaptchaState.Submitting);
        this.lastSubmitFrameHash = await this.elementFrameHash(captchaElement);
      }
    } finally {
      // Cleanup
      if (fs.existsSync(screenshotPath)) {
        fs.unlinkSync(screenshotPath);
      }
      // Only now: the wait gate re-reads the keyframe PNGs (which live inside this
      // directory) on every poll, so removing it any earlier would break the click
      // it is gating.
      // ...and not while the PLAN still holds them: the next round re-executes
      // this same answer, and the gate reads these PNGs on every poll.
      if (burstDir && this.animatedPlan?.burstDir !== burstDir) {
        try { fs.rmSync(burstDir, { recursive: true, force: true }); } catch { /* best-effort */ }
      }
    }

    return { didInteract: performedAction, tokenUsage: allTokenUsage };
  }

  private async getVerifyButton(frame: Frame | ElementHandle): Promise<ElementHandle | null> {
    let submitted = false;

    // 1. Try generic button selectors by text
    //
    // `.//` — RELATIVE. `frame` is an ElementHandle whenever the widget is
    // markup on the host page rather than a vendor iframe (all eight inline
    // vendors), and a document-rooted `//button` does not resolve against an
    // element handle: the query returns nothing even with the button sitting
    // inside that very element. On a Frame the context node is the document,
    // where `.//` and `//` mean the same thing, so the vendor paths are
    // unaffected — and scoping is the point on an element, since a
    // document-rooted match would reach the host page's own form submit.
    const buttonTexts = ['Verify', 'Next', 'Submit', 'Skip'];
    for (const text of buttonTexts) {
      try {
        // Case-insensitive contains for text
        const btn = await frame.$(
          `xpath=.//button[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '${text.toLowerCase()}')] | .//div[@role="button" and contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '${text.toLowerCase()}')]`
        );
        if (btn && await btn.isVisible()) {
          return btn;
        }
      } catch (e) {
        // Ignore locator errors
      }
    }

    if (!submitted) {
      // 2. Try specific ID (Recaptcha)
      const recaptchaVerify = await frame.$('#recaptcha-verify-button');
      if (recaptchaVerify && await recaptchaVerify.isVisible()) {
        return recaptchaVerify;
      }
    }

    if (!submitted) {
      // 3. Try specific class (hCaptcha)
      const hcaptchaVerify = await frame.$('.button-submit');
      if (hcaptchaVerify && await hcaptchaVerify.isVisible()) {
        return hcaptchaVerify;
      }
    }

    if (!submitted) {
      // 4. GeeTest: `<div class="geetest_submit geetest_disable">OK</div>`.
      //
      // Invisible to BOTH shapes above — a bare div carries no role="button",
      // and "OK" is on none of the four word lists. So nothing was ever pressed
      // on the puzzles that need pressing, and since a GeeTest board does not
      // grade until you do, the solve loop re-read the same unchanged panel and
      // re-answered it identically until the round cap. Ordered icon-click
      // scored 0/31 and 0/13 that way while the model was answering correctly:
      // measured on 2026-08-19, three returned points landed on the three
      // reference icons in order and the cursor arrived within 0.005 normalised
      // of each. A driver that discards a right answer is indistinguishable
      // from a model that cannot solve the puzzle, which is how this hid.
      //
      // Matched by CLASS, not by the word: `geetest_submit_tips` sits beside it
      // and also reads "OK", and pressing the tooltip does nothing at all.
      // `.geetest_submit` is a distinct class token, so it cannot match it.
      const geetestSubmit = await frame.$('.geetest_submit');
      if (geetestSubmit && await geetestSubmit.isVisible()) {
        return geetestSubmit;
      }
    }
    return null;
  }

  private async hasNonEmptyFieldValue(page: Page, selector: string): Promise<boolean> {
    try {
      const el = await page.$(selector);
      if (!el) return false;
      const value = await page.$eval(selector, node => {
        const anyNode = node as any;
        return typeof anyNode.value === 'string' ? anyNode.value : '';
      });
      return typeof value === 'string' && value.trim().length > 0;
    } catch {
      return false;
    }
  }

  /**
   * WHICH of reCAPTCHA's three bframe banners is showing, if any.
   *
   * They share a corner and a look and they mean different things:
   *
   *   `rejected`      "Please try again."  The answer was wrong; a fresh board
   *                   follows, so the useful response is to solve that one.
   *   `select-more`   "Please select all matching images."  Under-selected, and
   *                   the tiles do NOT refresh — so a driver that re-submits
   *                   the same answer loops until the session times out. This
   *                   is the one the missed-tiles retry and the abort exist for.
   *   `dynamic-more`  "Please also check the new images."  NOT AN ERROR. It is
   *                   the dynamic 3x3's normal flow: cleared tiles fade out,
   *                   replacements fade in, and the widget says so — on
   *                   essentially every round of that variant.
   *
   * This used to return a boolean over all three, which made the third one
   * indistinguishable from the first two and armed the one-retry abort with the
   * sentence that means "you are doing fine". Every dynamic board therefore
   * died at round two while the vendor was still dealing. Measured 2026-09-06:
   * three of three `recaptcha_3x3_fade` attempts stopped at exactly boards=2,
   * against `recaptcha_4x4` passing at 2, 3 and 5 boards on the same run.
   */
  /**
   * May a repeat of this banner end the solve?
   *
   * Only for the banners where repeating means STUCK. `dynamic-more` repeating
   * means the board is still being cleared, which is progress, and treating it
   * as fatal is the bug this pair of methods was split to fix.
   */
  private bannerIsFatalAfterRetry(
    kind: 'rejected' | 'select-more' | 'dynamic-more' | null,
  ): boolean {
    return kind === 'select-more' || kind === 'rejected';
  }

  private async recaptchaBannerKind(
    page: Page,
  ): Promise<'rejected' | 'select-more' | 'dynamic-more' | null> {
    try {
      const bframe = await page.$('iframe[src*="recaptcha/api2/bframe"]');
      if (!bframe) return null;
      const frame = await bframe.contentFrame();
      if (!frame) return null;
      // One selector per MEANING. Order is irrelevant — reCAPTCHA shows one.
      const banners = [
        ['.rc-imageselect-error-select-more', 'select-more'],
        ['.rc-imageselect-error-dynamic-more', 'dynamic-more'],
        ['.rc-imageselect-incorrect-response', 'rejected'],
      ] as const;
      for (const [sel, kind] of banners) {
        const el = await frame.$(sel);
        if (el) {
          // reCAPTCHA toggles these elements between visible / hidden via
          // an `aria-hidden` attribute on a wrapper — checking isVisible()
          // alone misses cases where the element is in the layout tree but
          // currently being faded in. Treat presence + non-empty text as
          // enough.
          const visible = await el.isVisible().catch(() => false);
          const text = (await el.textContent().catch(() => null)) ?? '';
          if (visible && text.trim().length > 0) return kind;
        }
      }
      return null;
    } catch {
      return null;
    }
  }

  private async isRecaptchaAnchorChecked(anchorIframe: ElementHandle): Promise<boolean> {
    try {
      const frame = await anchorIframe.contentFrame();
      if (!frame) return false;
      const checked = await frame.$('.recaptcha-checkbox-checked');
      return !!(checked && await checked.isVisible());
    } catch {
      return false;
    }
  }

  private async isHcaptchaAnchorChecked(anchorIframe: ElementHandle): Promise<boolean> {
    // hCaptcha's anchor sets <div id="checkbox" aria-checked="true"> when
    // the puzzle has been solved. We use this as a solve signal because the
    // h-captcha-response token isn't always populated on demo pages.
    try {
      const frame = await anchorIframe.contentFrame();
      if (!frame) return false;
      const ariaChecked = await frame.$('#checkbox[aria-checked="true"]');
      return !!(ariaChecked && await ariaChecked.isVisible());
    } catch {
      return false;
    }
  }

  /**
   * True the moment the vendor reports the whole captcha solved — the anchor
   * checkbox flipped to checked, or the response token got populated. This is a
   * definitive "done" signal that a lingering, animating-closed challenge frame
   * is not: after the final submit, hCaptcha keeps the challenge iframe VISIBLE
   * for a couple of seconds while it verifies, so treating that frame as a fresh
   * puzzle (the old behavior) burned ~18s re-running the pipeline on it.
   */
  private async isCaptchaSolved(page: Page): Promise<boolean> {
    try {
      // The token FIRST, and unconditionally. It is a hidden field on the PAGE,
      // not inside the widget, so it never needed the anchor iframe to be on
      // screen — and the moment it matters most is precisely when the anchor is
      // NOT on screen, because hCaptcha keeps its challenge overlay up for a
      // couple of seconds after the winning submit. Gating this on the anchor's
      // visibility meant the one signal that was already true went unread, and
      // the loop ground on against a frame being torn down. Mirrors the Python
      // driver; pinned by tests/test_solved_detection.py in the finetune repo.
      if (await this.hasNonEmptyFieldValue(page, '[name="h-captcha-response"]')) return true;
      if (await this.hasNonEmptyFieldValue(page, '[name="g-recaptcha-response"]')) return true;
      // Turnstile. detectCaptcha already reads this exact field to decide a
      // Turnstile widget is UNSOLVED, so the signal was known to one half of
      // the driver and ignored by the other. Mirrors the Python driver.
      if (await this.hasNonEmptyFieldValue(page, '[name="cf-turnstile-response"]')) return true;

      // Anchor state is the fallback, and this one DOES need the iframe: it is
      // read out of the anchor's own document.
      const hc = await page.$('iframe[src*="hcaptcha"][src*="frame=checkbox"]');
      if (hc && await hc.isVisible().catch(() => false)) {
        if (await this.isHcaptchaAnchorChecked(hc)) return true;
      }
      const rc = await page.$('iframe[src*="recaptcha/api2/anchor"]');
      if (rc && await rc.isVisible().catch(() => false)) {
        if (await this.isRecaptchaAnchorChecked(rc)) return true;
      }
    } catch { /* fall through */ }
    return false;
  }

  /**
   * True when an image challenge is open AND has actually rendered its prompt —
   * i.e. a fresh round we should solve, as opposed to a frame animating closed
   * (whose prompt has already gone). Used to tell "next round" from "solved,
   * closing" after a submit without waiting out a fixed timeout.
   */
  private async isChallengeFreshlyRendered(page: Page): Promise<boolean> {
    try {
      const hc = await page.$('iframe[src*="hcaptcha"][src*="frame=challenge"]');
      if (hc && await hc.isVisible().catch(() => false)) {
        // "Prompt painted" alone does NOT mean a next round. hCaptcha leaves
        // the round you just answered on screen while it verifies, so this
        // fired on the CLOSING frame and broke the post-submit poll out of its
        // solved-check immediately — committing the solver to ~21s of
        // readiness waits and a full inference against a dying frame. The
        // frame must have actually CHANGED since the submit, which we already
        // snapshot at submit time. Mirrors the Python driver.
        if (this.lastSubmitFrameHash) {
          const current = await this.elementFrameHash(hc).catch(() => null);
          if (current && current === this.lastSubmitFrameHash) return false;
        }
        const frame = await hc.contentFrame();
        const prompt = frame && await frame.$('.prompt-text');
        if (prompt && await prompt.isVisible().catch(() => false)) {
          const text = (await prompt.textContent().catch(() => '')) ?? '';
          if (text.trim().length > 0) return true;
        }
      }
      const rc = await page.$('iframe[src*="recaptcha/api2/bframe"]');
      if (rc && await rc.isVisible().catch(() => false)) {
        // The SAME gate as hCaptcha above, and for the same reason. reCAPTCHA
        // also leaves the answered board on screen while it verifies, and its
        // instructions stay visible throughout — so "instructions are showing"
        // fired on the closing frame and reported a fresh round that was not
        // coming. This branch simply never got the fix when the hCaptcha one
        // did.
        if (this.lastSubmitFrameHash) {
          const current = await this.elementFrameHash(rc).catch(() => null);
          if (current && current === this.lastSubmitFrameHash) return false;
        }
        const frame = await rc.contentFrame();
        const instr = frame && await frame.$('.rc-imageselect-instructions, #rc-imageselect');
        if (instr && await instr.isVisible().catch(() => false)) return true;
      }
    } catch { /* fall through */ }
    return false;
  }


  /**
   * Block until the hCaptcha challenge frame's task images have actually
   * painted, so we don't screenshot a blank/half-loaded grid.
   *
   * hCaptcha renders each grid tile as a `.task-image .image` div whose
   * `background-image` is set once the asset loads; the prompt sits in
   * `.prompt-text`. Image-select (click/drag) challenges use a single
   * `.challenge-example` / `canvas` surface instead. We wait for either family
   * to be present AND for the background-image URLs to be populated (not the
   * empty `url("")` placeholder hCaptcha ships before the asset arrives).
   *
   * Best-effort: a timeout or a missing content frame just falls through to the
   * screenshot rather than throwing — the existing fail-fast path still covers a
   * genuinely unsupported puzzle.
   */
  private async waitForHcaptchaChallengeImages(challengeIframe: ElementHandle): Promise<void> {
    try {
      const frame = await challengeIframe.contentFrame();
      if (!frame) return;

      // Prompt must be present and non-empty first — it's the cheapest signal
      // that the challenge frame has rendered its content at all.
      await frame.waitForSelector('.prompt-text', {
        state: 'visible',
        timeout: this.config.hcaptchaImagesTimeoutMs ?? 3000,
      });

      // Then wait for the actual imagery to load. Grid tiles expose a
      // background-image; click/drag puzzles expose a canvas or example image.
      await frame.waitForFunction(() => {
        const tiles = Array.from(
          document.querySelectorAll('.task-image .image, .task .image'),
        ) as HTMLElement[];
        if (tiles.length > 0) {
          // Every visible tile must have a real background-image URL.
          return tiles.every((el) => {
            const bg = getComputedStyle(el).backgroundImage;
            return bg && bg !== 'none' && !/url\(["']?["']?\)/.test(bg);
          });
        }
        // Non-grid (click/drag) challenge: a painted canvas or loaded example img.
        const canvas = document.querySelector('canvas');
        if (canvas instanceof HTMLCanvasElement && canvas.width > 0 && canvas.height > 0) {
          return true;
        }
        const example = document.querySelector(
          '.challenge-example img, .image-wrapper img',
        ) as HTMLImageElement | null;
        if (example) return example.complete && example.naturalWidth > 0;
        // NOTHING TO WAIT FOR IS READY. This used to be
        // `return !!(example && ...)`, false when there was no example image at
        // all — so a challenge with no tile grid, no canvas and no example
        // polled out the whole timeout and then carried on regardless.
        // Measured in the Python port, which had the identical clause: 24.0s of
        // a 45.2s solve, the full window three times, spent asking about
        // elements that were not on the page. A gate can only report on what it
        // can see; with nothing to check it has no opinion, and no opinion must
        // not read as "not ready".
        return true;
      }, { timeout: this.config.hcaptchaImagesTimeoutMs ?? 3000 });
    } catch {
      // Timed out or frame detached mid-load; fall through to the screenshot.
    }
  }

  /**
   * Stage-1 detection: is an *interactive* captcha widget present in the DOM at
   * all — even if its iframe hasn't finished rendering yet?
   *
   * This is deliberately broader than detectCaptcha (which only returns a
   * VISIBLE, not-yet-solved element). We use it to distinguish two cases that
   * detectCaptcha() === null cannot tell apart:
   *
   *   - A reCAPTCHA-v2 / hCaptcha widget IS in the DOM but is still loading
   *     (iframe present, glyph not painted) → we should WAIT for it.
   *   - There is no interactive widget — reCAPTCHA v3 (score-based, invisible)
   *     or an hCaptcha that only triggers on a user action → we must FAIL FAST.
   *
   * reCAPTCHA v3 injects only `iframe[src*="recaptcha/api2/anchor"]` with
   * `size=invisible` in the src, and never an `api2/bframe` challenge frame, so
   * we exclude the invisible variant here.
   */
  /**
   * Which vendors' code this page actually loaded.
   *
   * Read out of the page's own resource timing plus the `src`/`href` of
   * everything it linked, so it works whenever it is called — a request
   * listener would have to have been attached before navigation, and by the
   * time a solve fails that ship has sailed.
   *
   * Best-effort by construction: the resource-timing buffer is finite and a
   * page may have cleared it. An empty answer is "nothing seen", never "no
   * captcha here". Mirrors `vendors_on_the_wire` in page_solver.py.
   */
  public async vendorsOnTheWire(page: Page): Promise<string[]> {
    let names: string[] = [];
    try {
      // `$eval('html', ...)` rather than `evaluate`: PlaywrightPage is a
      // structural subset carrying exactly the members the solver uses, and
      // widening it would oblige every adapter (puppeteer-adapter.ts) to grow a
      // member for one diagnostic. The callback runs in the page either way.
      names = await page.$eval('html', () => {
        const out: string[] = [];
        try {
          for (const e of performance.getEntriesByType('resource')) out.push(e.name);
        } catch (err) { /* buffer unavailable */ }
        for (const el of Array.from(
          document.querySelectorAll('script[src],iframe[src],link[href],img[src]'),
        )) {
          out.push(el.getAttribute('src') || el.getAttribute('href') || '');
        }
        return out;
      });
    } catch {
      return [];
    }
    const blob = (names ?? []).join(' ');
    return VENDOR_URL_MARKERS
      .filter(({ hosts }) => hosts.some((h) => blob.includes(h)))
      .map(({ puzzleSource }) => puzzleSource);
  }

  /**
   * Why nothing was found — the two cases told apart. The old text guessed
   * "reCAPTCHA v3 / invisible" every time, including the times a vendor had
   * simply moved its DOM. See VENDOR_URL_MARKERS.
   */
  private async noWidgetMessage(page: Page): Promise<string> {
    const base = 'No interactive captcha widget detected';
    const loaded = await this.vendorsOnTheWire(page);
    if (!loaded.length) {
      return `${base} (no vendor captcha code loaded on this page — likely `
        + 'reCAPTCHA v3 / invisible, or a click-triggered challenge that has '
        + 'not been triggered). Failing fast.';
    }
    return `${base}, BUT ${loaded.join('/')} code IS loaded and running on this `
      + 'page. The vendor\'s markup no longer matches anything in '
      + 'VENDOR_WIDGET_LOCATORS — re-measure with '
      + 'scripts/check_vendor_selectors.py and update BOTH solver ports.';
  }

  public async hasInteractiveWidgetInDom(page: Page): Promise<boolean> {
    // reCAPTCHA v2 anchor, but NOT the invisible (v3 / invisible-v2) variant.
    const recaptchaAnchors = await page.$$('iframe[src*="recaptcha/api2/anchor"]');
    for (const a of recaptchaAnchors) {
      const src = (await a.getAttribute('src')) ?? '';
      if (!/[?&]size=invisible/.test(src)) return true;
    }
    // reCAPTCHA challenge frame present at all → definitely interactive.
    if (await page.$('iframe[src*="recaptcha/api2/bframe"]')) return true;

    // hCaptcha checkbox or challenge frame present (visible or not yet).
    if (await page.$('iframe[src*="hcaptcha"][src*="frame=checkbox"]')) return true;
    if (await page.$('iframe[src*="hcaptcha"][src*="frame=challenge"]')) return true;

    // The eight inline vendors, same table detectCaptcha uses. Without them
    // this answered "no widget" for every GeeTest / Yidun / Yandex / Lemin /
    // Prosopo / MTCaptcha / BotDetect / Tencent page whose puzzle had not
    // painted yet — and the caller reads that as "reCAPTCHA v3 / invisible"
    // and throws "No interactive captcha widget detected. Failing fast." in
    // under a second, instead of granting the render wait this method exists
    // to grant. The Python port has no such fail-fast, so the two ports
    // disagreed on every inline vendor (CLAUDE.md 1c) and Tier 3 scored it as
    // an unsolvable puzzle on the JS side only.
    //
    // Presence, not visibility — "in the DOM but not finished rendering" is
    // the entire question here; detectCaptcha still does the visibility check.
    for (const { selectors } of VENDOR_WIDGET_LOCATORS) {
      for (const selector of selectors) {
        if (await page.$(selector)) return true;
      }
    }

    return false;
  }

  public async detectCaptcha(page: Page): Promise<ElementHandle | null> {
    // Prioritize open challenges (the grid/images) over the initial checkbox

    // Recaptcha Challenge
    const recaptchaChallenge = await page.$('iframe[src*="recaptcha/api2/bframe"]');
    if (recaptchaChallenge && await recaptchaChallenge.isVisible()) return recaptchaChallenge;

    // hCaptcha Challenge — match the `frame=challenge` URL fragment.
    // The anchor iframe's title is "Widget containing checkbox for hCaptcha
    // security challenge" so a title-based fallback would mis-classify it
    // as the challenge frame. The URL fragment is unambiguous.
    const hcaptchaChallenge = await page.$('iframe[src*="hcaptcha"][src*="frame=challenge"]');
    if (hcaptchaChallenge && await hcaptchaChallenge.isVisible()) return hcaptchaChallenge;

    // Recaptcha Checkbox
    const recaptchaCheckbox = await page.$('iframe[src*="recaptcha/api2/anchor"]');
    if (recaptchaCheckbox && await recaptchaCheckbox.isVisible()) {
      // If it's already checked, consider it solved and continue searching.
      const checked = await this.isRecaptchaAnchorChecked(recaptchaCheckbox);
      if (!checked) return recaptchaCheckbox;
    }

    // hCaptcha Checkbox (anchor) — match the `frame=checkbox` URL fragment.
    const hcaptchaCheckbox = await page.$('iframe[src*="hcaptcha"][src*="frame=checkbox"]');
    if (hcaptchaCheckbox && await hcaptchaCheckbox.isVisible()) {
      // Solved if EITHER the h-captcha-response token is set OR the anchor
      // has flipped to aria-checked="true". Demo pages don't always populate
      // the token, so the visual state is the necessary tie-breaker.
      const hasToken = await this.hasNonEmptyFieldValue(page, '[name="h-captcha-response"]');
      const checked = await this.isHcaptchaAnchorChecked(hcaptchaCheckbox);
      if (!hasToken && !checked) return hcaptchaCheckbox;
    }

    // Cloudflare Turnstile
    // Try iframe first (if visible/open)
    const cloudflareIframe = await page.$('iframe[src*="challenges.cloudflare.com"]');
    if (cloudflareIframe && await cloudflareIframe.isVisible()) {
      const hasToken = await this.hasNonEmptyFieldValue(page, '[name="cf-turnstile-response"]');
      if (!hasToken) return cloudflareIframe;
    }

    // Fallback to container for closed shadow roots
    const cloudflareContainer = await page.$('.cf-turnstile');
    if (cloudflareContainer && await cloudflareContainer.isVisible()) {
      const hasToken = await this.hasNonEmptyFieldValue(page, '[name="cf-turnstile-response"]');
      if (!hasToken) return cloudflareContainer;
    }

    // Vendors with one interactive surface (no checkbox/challenge split) —
    // GeeTest, Tencent, Yidun, Yandex, Lemin, Prosopo, MTCaptcha, BotDetect.
    for (const { selectors } of VENDOR_WIDGET_LOCATORS) {
      for (const selector of selectors) {
        const el = await page.$(selector);
        if (el && await el.isVisible()) return el;
      }
    }

    return null;
  }

  /**
   * Initialize a fresh dump directory for one reCAPTCHA 3x3 dynamic-driver
   * session. Frames + a state.jsonl log land here so the click/fade/wait timing
   * can be replayed offline. Gated on CAPTCHA_DEBUG=1 — the per-frame dumps and
   * extra state queries add latency, so they stay off in normal runs. Set
   * CAPTCHA_DEBUG=1 to capture them when diagnosing timing. Best-effort.
   */
  private initGridDebug(): void {
    if (process.env.CAPTCHA_DEBUG !== '1') {
      this.gridDebugDir = null;
      return;
    }
    try {
      const cliRoot = this.config.repoPath ?? getBundledCliRoot();
      const base = path.join(cliRoot, 'latestDebugRun_grid');
      const stamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, -5);
      this.gridDebugDir = path.join(base, `griddrv_${stamp}_${Math.floor(Math.random() * 1e6)}`);
      this.gridDebugSeq = 0;
      fs.mkdirSync(this.gridDebugDir, { recursive: true });
      console.log(`[grid-debug] dumping driver frames + state to: ${this.gridDebugDir}`);
    } catch (e) {
      this.gridDebugDir = null;
      console.warn(`[grid-debug] could not init debug dir: ${e}`);
    }
  }

  /**
   * Log a structured event for the grid driver: prints a one-line summary to the
   * console and appends a JSON record to state.jsonl. If `framePath` is given,
   * copies that frame into the dump dir under a sequenced, labeled name so the
   * record can be matched to the exact pixels the detector saw. Best-effort.
   */
  private gridDebug(event: string, data: Record<string, any> = {}, framePath?: string): void {
    // No-op unless grid debugging is active (CAPTCHA_DEBUG=1). Keeps the verbose
    // per-poll trace + frame dumps off the hot path in normal runs.
    if (!this.gridDebugDir) return;
    const seq = ++this.gridDebugSeq;
    console.log(`[grid-debug #${seq}] ${event} ${JSON.stringify(data)}`);
    try {
      let savedFrame: string | undefined;
      if (framePath && fs.existsSync(framePath)) {
        savedFrame = `${String(seq).padStart(3, '0')}_${event}.png`;
        fs.copyFileSync(framePath, path.join(this.gridDebugDir, savedFrame));
      }
      const record = { seq, t: new Date().toISOString(), event, ...data, frame: savedFrame };
      fs.appendFileSync(path.join(this.gridDebugDir, 'state.jsonl'), JSON.stringify(record) + '\n');
    } catch {
      // best-effort; never fail the solve over debug I/O
    }
  }

  private saveImageForDebug(imagePath: string): void {
    // Check if CAPTCHA_DEBUG is enabled
    const debugEnabled = process.env.CAPTCHA_DEBUG === '1';
    if (!debugEnabled) {
      return;
    }

    try {
      const cliRoot = this.config.repoPath ?? getBundledCliRoot();
      // Save input images to a separate directory that won't be cleared by the Python CLI
      // The Python CLI clears latestDebugRun, so we use a sibling directory
      const inputImagesDir = path.join(cliRoot, 'latestDebugRun_inputs');

      // Ensure input images directory exists
      if (!fs.existsSync(inputImagesDir)) {
        fs.mkdirSync(inputImagesDir, { recursive: true });
      }

      // Increment counter and save with a descriptive name
      this.imageCounter++;
      const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, -5);
      const debugImageName = `input_${String(this.imageCounter).padStart(3, '0')}_${timestamp}.png`;
      const debugImagePath = path.join(inputImagesDir, debugImageName);

      // Copy the image to debug directory
      fs.copyFileSync(imagePath, debugImagePath);
      console.log(`[DEBUG] Saved input image to: ${debugImagePath}`);
    } catch (error) {
      // Don't fail the solve if debug save fails
      console.warn(`[DEBUG] Failed to save image for debugging: ${error}`);
    }
  }

  private archiveLatestDebugRun(attempt: number, actions: SolverResult): void {
    if (!this.sessionDebugDir) return;

    try {
      const cliRoot = this.config.repoPath ?? getBundledCliRoot();
      const latestDebugDir = path.join(cliRoot, 'latestDebugRun');
      const inputImagesDir = path.join(cliRoot, 'latestDebugRun_inputs');

      const attemptDir = path.join(this.sessionDebugDir, `attempt_${attempt}`);
      fs.mkdirSync(attemptDir, { recursive: true });

      // Archive CLI artifacts if they exist
      if (fs.existsSync(latestDebugDir)) {
        fs.cpSync(latestDebugDir, attemptDir, { recursive: true });
        fs.rmSync(latestDebugDir, { recursive: true, force: true });
      }

      // Archive input images if they exist
      if (fs.existsSync(inputImagesDir)) {
        const archivedInputsDir = path.join(attemptDir, 'inputs');
        fs.mkdirSync(archivedInputsDir, { recursive: true });
        fs.cpSync(inputImagesDir, archivedInputsDir, { recursive: true });
        fs.rmSync(inputImagesDir, { recursive: true, force: true });
      }

      // Add actions info to the attempt directory
      fs.writeFileSync(
        path.join(attemptDir, 'actions_result.json'),
        JSON.stringify(actions, null, 2)
      );

      console.log(`[DEBUG] Archived attempt ${attempt} debug artifacts to: ${attemptDir}`);
    } catch (error) {
      console.warn(`[DEBUG] Failed to archive debug artifacts: ${error}`);
    }
  }

  /**
   * Resolve the bundled CaptchaKraken CLI root and the python interpreter to
   * run it with. Prefers the packaged venv python (postinstall bootstrap),
   * falling back to the configured/`python` command. Throws if the CLI folder
   * is missing — callers that must not throw (e.g. runCliTool) wrap this.
   */
  /**
   * The adapter name models.json calls `latest`, resolved once per solver.
   *
   * `resolveLoraName` reads models.json (and pinned_model.json) off disk
   * SYNCHRONOUSLY, and it was doing that on every inference for a file that
   * cannot change inside one process.
   */
  private loraName(cliRoot: string): string {
    if (this.loraNameCache === null) this.loraNameCache = resolveLoraName({ cliRoot });
    return this.loraNameCache;
  }

  private resolveCli(): { cliRoot: string; py: string } {
    // MEMOISED. `resolvePythonCommand` probes the interpreter with a
    // `spawnSync(cmd, ['--version'])`, which blocks the whole Node event loop —
    // Playwright's socket pump and every timer with it — and this was called on
    // EVERY inference, every CV fallback and every worker start. On the
    // npm-install default (no bundled venv, nothing configured) that is up to
    // two blocking process spawns per model call, for an answer that cannot
    // change inside one process.
    if (this.cliCache) return this.cliCache;
    const { repoPath, pythonCommand } = this.config;
    const cliRoot = repoPath ?? getBundledCliRoot();
    if (!fs.existsSync(cliRoot)) {
      throw new Error(
        `CaptchaKraken CLI folder not found at ${cliRoot}. ` +
        `If you installed from npm, ensure the package ships 'python/'.`
      );
    }
    // Was `getVenvPython(cliRoot) ?? 'python'`. Debian-family systems have no
    // bare `python`, so with no bundled venv every solve died at
    // `/bin/sh: 1: python: not found` — before reaching the model, and only on
    // the JS side, which made it read as an endpoint fault. See python-command.ts.
    const py = resolvePythonCommand({
      configured: pythonCommand,
      venvPython: getVenvPython(cliRoot),
      exists: commandExists,
    });
    this.cliCache = { cliRoot, py };
    return this.cliCache;
  }

  /**
   * Run an OpenCV tool subcommand of the CLI (e.g. `grid-cell-states a.png
   * b.png`) and return its parsed single-line JSON. These subcommands print
   * exactly one JSON object on stdout (timing records go to stderr), so we
   * parse the whole trimmed stdout. Best-effort: returns `{}` on any failure so
   * polling callers can treat it as "inconclusive, keep going" without throwing.
   */
  private async runCliTool(args: string[]): Promise<any> {
    try {
      const { cliRoot, py } = this.resolveCli();
      // Use execFile (no shell) so args containing JSON / brackets / spaces —
      // e.g. the grid_boxes payload for grid-cell-states-fixed — are passed
      // literally without any shell quoting/globbing hazards.
      const { stdout } = await execFileAsync(py, ['-m', 'captchakraken.cli', ...args], {
        cwd: cliRoot,
        env: cliEnv(cliRoot),
        maxBuffer: 10 * 1024 * 1024,
      });
      return JSON.parse(stdout.trim());
    } catch {
      return {};
    }
  }

  /**
   * Lazily start the persistent CV worker (`python -m captchakraken.cli serve`) and resolve
   * once it has imported cv2/numpy and emitted its `{"ready":true}` handshake.
   * Returns false if it can't be started (caller then falls back to one-shot
   * subprocesses). Idempotent: subsequent calls await the same readiness promise.
   */
  private ensureCvWorker(): Promise<boolean> {
    if (this.cvWorkerReady) return this.cvWorkerReady;
    this.cvWorkerReady = new Promise<boolean>((resolve) => {
      try {
        const { cliRoot, py } = this.resolveCli();
        const proc = spawn(py, ['-m', 'captchakraken.cli', 'serve'], { cwd: cliRoot, env: cliEnv(cliRoot) });
        this.cvWorker = proc;

        let settled = false;
        const fail = () => {
          if (!settled) { settled = true; resolve(false); }
          this.teardownCvWorker();
        };

        proc.stdout.on('data', (chunk: Buffer) => {
          this.cvWorkerBuf += chunk.toString();
          let nl: number;
          while ((nl = this.cvWorkerBuf.indexOf('\n')) >= 0) {
            const line = this.cvWorkerBuf.slice(0, nl).trim();
            this.cvWorkerBuf = this.cvWorkerBuf.slice(nl + 1);
            if (!line) continue;
            let msg: any;
            try { msg = JSON.parse(line); } catch { continue; }
            if (!settled && msg.ready === true) { settled = true; resolve(true); continue; }
            if (typeof msg.id === 'number' && this.cvWorkerPending.has(msg.id)) {
              const p = this.cvWorkerPending.get(msg.id)!;
              this.cvWorkerPending.delete(msg.id);
              if (msg.ok) p.resolve(msg.result);
              else p.reject(new Error(msg.error || 'cv worker error'));
            }
          }
        });
        proc.on('error', fail);
        proc.on('exit', () => {
          // Reject any in-flight requests so callers fall back rather than hang.
          for (const [, p] of this.cvWorkerPending) p.reject(new Error('cv worker exited'));
          this.cvWorkerPending.clear();
          fail();
        });
        // Bounded readiness wait — if imports stall, fall back to one-shot.
        setTimeout(() => { if (!settled) { settled = true; resolve(false); } }, 8000);
      } catch {
        resolve(false);
      }
    });
    return this.cvWorkerReady;
  }

  /** Send one request to the CV worker and await its JSON result. Throws on any
   *  worker failure so callers can fall back to the one-shot path. */
  private cvWorkerRequest(payload: Record<string, any>, timeoutMs = 10000): Promise<any> {
    const proc = this.cvWorker;
    if (!proc || proc.exitCode !== null) return Promise.reject(new Error('cv worker not running'));
    const id = ++this.cvWorkerSeq;
    return new Promise<any>((resolve, reject) => {
      const timer = setTimeout(() => {
        if (this.cvWorkerPending.delete(id)) reject(new Error('cv worker request timeout'));
      }, timeoutMs);
      this.cvWorkerPending.set(id, {
        resolve: (v) => { clearTimeout(timer); resolve(v); },
        reject: (e) => { clearTimeout(timer); reject(e); },
      });
      try {
        proc.stdin.write(JSON.stringify({ id, ...payload }) + '\n');
      } catch (e) {
        this.cvWorkerPending.delete(id);
        clearTimeout(timer);
        reject(e);
      }
    });
  }

  /** Kill the worker and clear state. Safe to call repeatedly. */
  private teardownCvWorker(): void {
    const proc = this.cvWorker;
    this.cvWorker = null;
    if (proc) { try { proc.kill(); } catch { /* best-effort */ } }
  }

  /**
   * Run a CV tool through the persistent worker when available, falling back to a
   * one-shot `runCliTool` subprocess otherwise. `cmd`/`payload` map to the
   * worker's protocol; `fallbackArgs` is the equivalent one-shot argv. Worker
   * results are wrapped to match the one-shot JSON shape:
   *   - grid-cell-states[-fixed]: worker returns the states object directly, or
   *     {grid:null}; the one-shot returns the same shape, so just pass through.
   *   - find-grid: worker returns the array (or null) as `result`.
   * Best-effort: never throws.
   */
  private async runCvTool(cmd: string, payload: Record<string, any>, fallbackArgs: string[]): Promise<any> {
    try {
      if (await this.ensureCvWorker()) {
        const result = await this.cvWorkerRequest({ cmd, ...payload });
        return result;
      }
    } catch {
      // fall through to one-shot
    }
    return this.runCliTool(fallbackArgs);
  }

  /**
   * Block until a reCAPTCHA grid's cells have settled — none blank, none
   * mid-fade — before we screenshot it for the model. reCAPTCHA fades new tiles
   * in over ~1s; capturing mid-fade feeds the LoRA a blank/partial grid.
   *
   * We poll: screenshot the challenge element, keep the last two frames, and ask
   * the CLI's batched `grid-cell-states` (one subprocess per poll) which cells
   * are empty/changing/loaded. We return as soon as every cell is loaded, or on
   * timeout. Best-effort, mirroring `waitForHcaptchaChallengeImages`: never
   * throws, and falls through on timeout so a stuck/odd grid still proceeds to
   * the normal screenshot path. Temp frames are always cleaned up.
   */
  private async waitForGridCellsLoaded(
    captchaElement: ElementHandle,
    opts?: { intervalMs?: number; timeoutMs?: number },
  ): Promise<boolean> {
    const interval = opts?.intervalMs ?? this.config.gridLoadPollIntervalMs ?? 250;
    const timeout = opts?.timeoutMs ?? this.config.gridLoadTimeoutMs ?? 8000;
    const start = Date.now();
    const frames: string[] = [];
    const tmp = () => path.join(
      os.tmpdir(),
      `gridpoll_${Date.now()}_${Math.floor(Math.random() * 1e9)}.png`,
    );
    try {
      while (Date.now() - start < timeout) {
        const f = tmp();
        /*
         * AN EXPLICIT, SHORT TIMEOUT — and the loop's own budget is not one.
         *
         * Playwright defaults to 30s and waits for the element to be visible
         * and stable first. Right after the grid driver clicks Verify, this
         * element is a challenge iframe the vendor is tearing down, so that
         * wait runs to the full default. The `while` above is consulted only
         * BETWEEN iterations, so one hung screenshot sails straight past the
         * 8s cap it looks protected by: measured live, submit at 24.5s and the
         * solved verdict at 64.7s, a 38.5s tail on a solve whose real work took
         * twelve seconds — paid on every multi-round reCAPTCHA.
         *
         * waitForElementSettled already carries this fix and the note
         * explaining it; this poll was written in the same shape without it.
         */
        const left = timeout - (Date.now() - start);
        try {
          await captchaElement.screenshot({
            path: f,
            timeout: Math.max(500, Math.min(2500, left)),
            animations: 'disabled',
          });
        } catch {
          // The challenge cannot be photographed: it is gone or going. There is
          // no grid here to finish loading, so stop rather than spend the rest
          // of the budget proving it again.
          return false;
        }
        frames.push(f);

        if (frames.length >= 2) {
          const a = frames[frames.length - 2];
          const b = frames[frames.length - 1];
          const res = await this.runCvTool('grid-cell-states', { a, b }, ['grid-cell-states', a, b]);
          // `{grid: null}` => grid not painted yet; keep polling. A real grid
          // result with no empty/changing cells and >=1 loaded cell => settled.
          const gridFound = res && res.grid !== null && Array.isArray(res.loaded);
          if (
            gridFound
            && Array.isArray(res.empty) && res.empty.length === 0
            && Array.isArray(res.changing) && res.changing.length === 0
            && res.loaded.length > 0
          ) {
            return true;
          }
          // Drop the older frame so disk use stays bounded to one prior frame.
          const stale = frames.shift();
          if (stale && fs.existsSync(stale)) fs.unlinkSync(stale);
        }

        await delay(interval);
      }
      return false;
    } catch {
      return false;
    } finally {
      for (const f of frames) {
        if (fs.existsSync(f)) {
          try { fs.unlinkSync(f); } catch { /* best-effort cleanup */ }
        }
      }
    }
  }

  /**
   * Read a PNG's pixel dimensions from its IHDR chunk (bytes 16-23, big-endian).
   * Avoids pulling in an image-size dependency. Returns null if the file isn't a
   * readable PNG.
   */
  private readPngDimensions(filePath: string): { width: number; height: number } | null {
    try {
      const fd = fs.openSync(filePath, 'r');
      try {
        const buf = new Uint8Array(24);
        const read = fs.readSync(fd, buf, 0, 24, 0);
        if (read < 24) return null;
        // PNG signature is 8 bytes; IHDR length+type is 8 more; then width/height
        // as big-endian uint32s at byte offsets 16 and 20.
        const isPng = buf[1] === 0x50 && buf[2] === 0x4e && buf[3] === 0x47; // "PNG"
        if (!isPng) return null;
        const beU32 = (o: number) => (buf[o] << 24 | buf[o + 1] << 16 | buf[o + 2] << 8 | buf[o + 3]) >>> 0;
        const width = beU32(16);
        const height = beU32(20);
        if (!width || !height) return null;
        return { width, height };
      } finally {
        fs.closeSync(fd);
      }
    } catch {
      return null;
    }
  }

  /**
   * Detect the reCAPTCHA grid once for a puzzle session: screenshot the element,
   * run `find-grid`, and read the screenshot's pixel dimensions. Grid boxes are
   * pixel coords in SCREENSHOT space (not page CSS space). Returns null if no
   * grid is detected. The geometry is stable across the in-place dynamic refresh
   * (only tile images change), so callers cache the result for the session.
   */
  private async getGridBoxes(
    captchaElement: ElementHandle,
  ): Promise<{ boxes: number[][]; size: 3 | 4; screenshotW: number; screenshotH: number } | null> {
    const f = path.join(os.tmpdir(), `findgrid_${Date.now()}_${Math.floor(Math.random() * 1e9)}.png`);
    try {
      // Explicit timeout — the challenge may be mid-teardown and Playwright's
      // 30s default waits for it to go stable. See screenshot-timeouts.test.ts.
      await captchaElement.screenshot({ path: f, timeout: 2500, animations: 'disabled' });
      const res = await this.runCvTool('find-grid', { image: f }, ['find-grid', f]);
      if (!Array.isArray(res) || (res.length !== 9 && res.length !== 16)) {
        return null;
      }
      const dims = this.readPngDimensions(f);
      if (!dims) return null;
      return {
        boxes: res as number[][],
        size: res.length === 16 ? 4 : 3,
        screenshotW: dims.width,
        screenshotH: dims.height,
      };
    } catch {
      return null;
    } finally {
      if (fs.existsSync(f)) {
        try { fs.unlinkSync(f); } catch { /* best-effort cleanup */ }
      }
    }
  }

  /**
   * Map a model-returned normalized bbox (fractions of the element/screenshot)
   * to a 1-indexed grid cell. Uses the bbox center, converts to screenshot
   * pixels, and returns the cell whose pixel box contains it. Cell numbering is
   * row-major (matches the CLI's find_grid output). Returns null if the center
   * falls outside every cell (e.g. in a gutter) — callers click the raw bbox
   * anyway and skip per-tile tracking.
   */
  private bboxToCell(
    bbox: [number, number, number, number],
    gridBoxes: number[][],
    screenshotW: number,
    screenshotH: number,
  ): number | null {
    const [x1, y1, x2, y2] = bbox;
    const cx = ((x1 + x2) / 2) * screenshotW;
    const cy = ((y1 + y2) / 2) * screenshotH;
    for (let i = 0; i < gridBoxes.length; i++) {
      const [bx1, by1, bx2, by2] = gridBoxes[i];
      if (cx >= bx1 && cx <= bx2 && cy >= by1 && cy <= by2) {
        return i + 1; // 1-indexed
      }
    }
    return null;
  }

  /**
   * Center of a 1-indexed grid cell in PAGE pixel space, for mouse moves.
   * Converts the cached screenshot-pixel box to page coords via the session's
   * scaleX/scaleY (screenshot px -> page px) and element origin.
   */
  private cellCenterPage(cell: number, session: GridSession): { x: number; y: number } {
    const [x1, y1, x2, y2] = session.gridBoxes[cell - 1];
    const cxPx = (x1 + x2) / 2;
    const cyPx = (y1 + y2) / 2;
    return {
      x: session.elementBox.x + cxPx * session.scaleX,
      y: session.elementBox.y + cyPx * session.scaleY,
    };
  }

  /** Smooth-move the mouse over one cell's center with intra-cell jitter. */
  private async hoverCell(page: Page, session: GridSession, cell: number): Promise<void> {
    // A hover is mimicry of a resting CURSOR, so on a device that has none it
    // is not weaker mimicry — it is a mousemove at a touch-only widget.
    if (!this.human.hovers) return;
    const cellWPage = (session.gridBoxes[0][2] - session.gridBoxes[0][0]) * session.scaleX;
    const cellHPage = (session.gridBoxes[0][3] - session.gridBoxes[0][1]) * session.scaleY;
    const center = this.cellCenterPage(cell, session);
    const jitterX = (Math.random() - 0.5) * cellWPage * 0.4;
    const jitterY = (Math.random() - 0.5) * cellHPage * 0.4;
    await this.performSmoothMove(page, center.x + jitterX, center.y + jitterY);
  }

  /**
   * Query per-cell grid state using the SESSION'S CACHED grid boxes via the
   * `grid-cell-states-fixed` CLI command. This is critical: the dynamic refresh
   * blanks tiles to near-white, which makes find_grid fail on that frame, so the
   * self-detecting `grid-cell-states` would return {grid:null} mid-fade and a
   * naive caller would misread that as "nothing loading / solved". Passing the
   * cached boxes keeps empty/changing/selected correct even while tiles are
   * blank. Returns null only on a genuine CLI failure. Best-effort.
   */
  private async gridCellStates(
    session: GridSession,
    frameA: string,
    frameB: string,
  ): Promise<GridCellStates | null> {
    const boxesJson = JSON.stringify(session.gridBoxes);
    const res = await this.runCvTool(
      'grid-cell-states-fixed',
      { a: frameA, b: frameB, grid_boxes: session.gridBoxes },
      ['grid-cell-states-fixed', frameA, frameB, boxesJson],
    );
    if (!res || !Array.isArray(res.empty)) return null;
    return {
      empty: res.empty ?? [],
      changing: res.changing ?? [],
      loaded: res.loaded ?? [],
      selected: res.selected ?? [],
    };
  }

  /** Order a loading set so `priority` cells (just-clicked) come first. */
  private orderByPriority(loading: number[], priority: number[]): number[] {
    const set = new Set(loading);
    const ordered: number[] = [];
    for (const c of priority) {
      if (set.has(c)) { ordered.push(c); set.delete(c); }
    }
    for (const c of set) ordered.push(c);
    return ordered;
  }

  /**
   * Watch the just-clicked tiles until the widget says what it did with them.
   *
   * Two answers because reCAPTCHA gives a click one of exactly two replies, and
   * they are the two kinds of board:
   *
   *   - `chipped`: the small blue chip landed in the tiles' top-left corners —
   *     the photos were KEPT. Nothing is on its way in, the selection is the
   *     answer, and the caller should press Verify.
   *   - `loading`: the photos are blanking or dissolving under a large centred
   *     check — those tiles are being SWAPPED, and what lands may match too, so
   *     the board has to be read again.
   *
   * A widget that swaps one clicked cell swaps them all, so the two never share
   * a board and one look at the tiles we just clicked settles it. The chip is
   * what the CV layer reports as `selected` (top-left corner only, behind a
   * circularity and a centroid test a centred check fails), and we already read
   * it on every poll. `chipped` needs EVERY watched tile: a partial reading is a
   * misread, and calling a swapping board finished submits half an answer and
   * burns the attempt, where calling a chipped board unfinished costs one
   * inference.
   *
   * The blank/fade transition lags the click by a beat, so a single snapshot
   * right after clicking misses it (the tile still shows its old image — not yet
   * white, not yet changing). We poll consecutive frames and mark a cell loading
   * if it is `empty` (≥97% near-white) OR `changing` (>2% pixels differ). HOVERS
   * a clicked tile each poll so the mouse keeps moving (no unnatural pauses).
   * Returns as soon as either verdict is in, or `{loading: [], chipped: false}`
   * if the whole window passes with nothing happening (→ solved). Logs every
   * poll + frame.
   */
  private async watchClickedTiles(
    page: Page,
    captchaElement: ElementHandle,
    session: GridSession,
    priority: number[] = [],
  ): Promise<{ loading: number[]; chipped: boolean }> {
    const grace = this.config.recaptchaFadeOnsetGraceMs ?? 4000;
    const interval = this.config.recaptchaDynamicFadePollMs ?? 250;
    const start = Date.now();
    const frames: string[] = [];
    const tmp = () => path.join(os.tmpdir(), `loadchk_${Date.now()}_${Math.floor(Math.random() * 1e9)}.png`);
    // We care specifically about the tiles we just clicked (priority). reCAPTCHA
    // holds them selected (old image visible) for ~1-3s, THEN blanks them to swap
    // in a replacement. So we must watch the CLICKED cells across the whole grace
    // window — the onset is delayed, not immediate.
    const watch = priority.length ? priority : null; // null => watch all cells
    this.gridDebug('fade-onset:start', { grace, priority, watching: watch ?? 'all' });
    let hoverIdx = 0;
    try {
      const first = tmp();
      try {
        await captchaElement.screenshot({ path: first, timeout: 2500, animations: 'disabled' });
      } catch {
        // Cannot photograph the challenge: it is gone or going, so no tile is
        // loading and none was chipped. Same verdict as an empty window.
        return { loading: [], chipped: false };
      }
      frames.push(first);
      this.gridDebug('fade-onset:baseline', {}, first);

      let polls = 0;
      while (Date.now() - start < grace) {
        // Keep the mouse moving over a clicked tile during the wait, and enforce
        // a minimum inter-frame gap so the change detector has a real diff (the
        // worker query is near-instant, so without this polls could fire back-to-
        // back on near-identical frames and miss a slow fade).
        const iterStart = Date.now();
        if (priority.length) {
          await this.hoverCell(page, session, priority[hoverIdx % priority.length]).catch(() => {});
          hoverIdx++;
        }
        const elapsed = Date.now() - iterStart;
        if (elapsed < interval) await delay(interval - elapsed);
        const f = tmp();
        /*
         * Explicit timeout: this element is a challenge iframe that may be
         * mid-teardown, and Playwright's 30s default waits for it to become
         * stable first. See screenshot-timeouts.test.ts.
         */
        try {
          await captchaElement.screenshot({ path: f, timeout: 2500, animations: 'disabled' });
        } catch {
          break; // challenge gone — let the post-loop verdict stand
        }
        frames.push(f);
        polls++;

        const a = frames[frames.length - 2];
        const b = frames[frames.length - 1];
        const st = await this.gridCellStates(session, a, b);
        // Restrict the loading signal to the cells we clicked (if known): a
        // background tile changing is irrelevant; a clicked tile going blank/
        // changing means the refresh has begun.
        const inScope = (c: number) => !watch || watch.includes(c);
        const emptyW = (st?.empty ?? []).filter(inScope);
        const changingW = (st?.changing ?? []).filter(inScope);
        this.gridDebug('fade-onset:poll', {
          poll: polls, elapsedMs: Date.now() - start,
          watchedEmpty: emptyW, watchedChanging: changingW,
          empty: st?.empty ?? null, changing: st?.changing ?? null,
          loaded: st?.loaded ?? null, selected: st?.selected ?? null,
        }, b);
        // Chip first: a chip landing on a tile ZOOMS its photo out, which reads
        // as `changing` on the very frame that shows the chip. Test the swap
        // first and every chipped board looks like a swapping one for as long as
        // that animation runs.
        const selected = st?.selected ?? [];
        if (priority.length && priority.every(c => selected.includes(c))) {
          this.gridDebug('fade-onset:chipped', { chipped: priority, afterMs: Date.now() - start });
          return { loading: [], chipped: true };
        }
        const loading = [...new Set([...emptyW, ...changingW])];
        if (loading.length) {
          const ordered = this.orderByPriority(loading, priority);
          this.gridDebug('fade-onset:loading-detected', { loading: ordered, afterMs: Date.now() - start });
          return { loading: ordered, chipped: false };
        }

        const stale = frames.shift();
        if (stale && fs.existsSync(stale)) fs.unlinkSync(stale);
      }
      this.gridDebug('fade-onset:none', { afterMs: Date.now() - start, polls });
      return { loading: [], chipped: false };
    } catch (e) {
      this.gridDebug('fade-onset:error', { error: String(e) });
      return { loading: [], chipped: false };
    } finally {
      for (const f of frames) {
        if (fs.existsSync(f)) { try { fs.unlinkSync(f); } catch { /* best-effort */ } }
      }
    }
  }

  /**
   * After loading is detected, wait until at least one of the given blank/fading
   * cells reaches the `loaded` state, HOVERING those cells (in order) the whole
   * time so the mouse never sits still. Returns true once a tile loads, false on
   * timeout (caller proceeds anyway). Uses the session's cached grid boxes so it
   * works while tiles are blank. Logs every poll + frame.
   */
  private async waitForAnyClickedTileLoaded(
    page: Page,
    captchaElement: ElementHandle,
    session: GridSession,
    fadingCells: number[],
  ): Promise<boolean> {
    if (!fadingCells.length) return true;
    const interval = this.config.recaptchaDynamicFadePollMs ?? 250;
    const timeout = this.config.recaptchaDynamicFadeWaitMs ?? 6000;
    const hoverEnabled = this.config.recaptchaTileHoverEnabled ?? true;
    const start = Date.now();
    const frames: string[] = [];
    const tmp = () => path.join(os.tmpdir(), `fadepoll_${Date.now()}_${Math.floor(Math.random() * 1e9)}.png`);
    this.gridDebug('wait-load:start', { fadingCells, timeout, interval });
    let hoverIdx = 0;
    let polls = 0;
    try {
      while (Date.now() - start < timeout) {
        // Move over a fading tile each iteration — human waiting for the image.
        // Always enforce a minimum inter-frame gap so the change detector has a
        // real diff to work with even when the (now near-instant) worker query
        // would otherwise let polls fire back-to-back.
        const iterStart = Date.now();
        if (hoverEnabled) {
          await this.hoverCell(page, session, fadingCells[hoverIdx % fadingCells.length]).catch(() => {});
          hoverIdx++;
        }
        const elapsed = Date.now() - iterStart;
        if (elapsed < interval) await delay(interval - elapsed);
        const f = tmp();
        /*
         * Explicit timeout: this element is a challenge iframe that may be
         * mid-teardown, and Playwright's 30s default waits for it to become
         * stable first. See screenshot-timeouts.test.ts.
         */
        try {
          await captchaElement.screenshot({ path: f, timeout: 2500, animations: 'disabled' });
        } catch {
          break; // challenge gone — nothing left to wait for
        }
        frames.push(f);

        if (frames.length >= 2) {
          const a = frames[frames.length - 2];
          const b = frames[frames.length - 1];
          const st = await this.gridCellStates(session, a, b);
          polls++;
          const loadedNow = st ? fadingCells.filter(c => st.loaded.includes(c)) : [];
          this.gridDebug('wait-load:poll', {
            poll: polls, elapsedMs: Date.now() - start,
            empty: st?.empty ?? null, changing: st?.changing ?? null,
            loaded: st?.loaded ?? null, loadedTargets: loadedNow,
          }, b);
          if (loadedNow.length) {
            this.gridDebug('wait-load:loaded', { loadedNow, afterMs: Date.now() - start });
            return true;
          }
          const stale = frames.shift();
          if (stale && fs.existsSync(stale)) fs.unlinkSync(stale);
        }
      }
      this.gridDebug('wait-load:timeout', { afterMs: Date.now() - start, polls });
      return false;
    } catch (e) {
      this.gridDebug('wait-load:error', { error: String(e) });
      return false;
    } finally {
      for (const f of frames) {
        if (fs.existsSync(f)) { try { fs.unlinkSync(f); } catch { /* best-effort */ } }
      }
    }
  }

  /**
   * Multi-round driver for reCAPTCHA 3x3 dynamic puzzles ("click all X" where
   * tiles refresh in place). One invocation = one puzzle session.
   *
   * The CLI is authoritative about WHAT to do — it runs the blue-badge detector,
   * filters out already-selected and still-loading tiles, and returns one of:
   *   - `click`: click these tiles (already filtered to fresh, ready tiles)
   *   - `wait` : nothing to click yet, tiles are still loading — do NOT submit
   *   - `done` : nothing matching remains — submit (click Verify)
   *
   * This driver owns the HUMAN-LIKE WAITING the CLI can't: after a click round,
   * and on a `wait`, it hovers the just-clicked / currently blank+fading tiles
   * (in click order) and waits for at least one to finish reloading before
   * re-screenshotting and re-solving — so we don't burn a solver call on a grid
   * that's still mid-fade.
   *
   * Only a board that SWAPS a clicked tile out is worth those extra rounds. One
   * that ticks the tile and keeps the photo has been fully answered by the round
   * that clicked it — same as the 4x4 — so `watchClickedTiles` reports the chip
   * and this submits there and then. Rounds 2..N exist for the fading board and
   * nothing else. It submits on `done`, and on a board that ticked our clicks.
   *
   * Returns the same shape as solveSingle so the outer solve loop — including the
   * under-selection retry and post-solve detectCaptcha — wraps it unchanged.
   */
  private async solveRecaptchaGrid(
    page: Page,
    captchaElement: ElementHandle,
    attempt: number,
    retryMode: string | null,
    grid: { boxes: number[][]; size: 3 | 4; screenshotW: number; screenshotH: number },
    elementBox: { x: number; y: number; width: number; height: number },
  ): Promise<{ didInteract: boolean; tokenUsage: TokenUsage[] }> {
    const maxRounds =
      this.config.recaptchaMaxDynamicRounds ?? DEFAULT_RECAPTCHA_MAX_DYNAMIC_ROUNDS;

    const session: GridSession = {
      gridBoxes: grid.boxes,
      elementBox,
      scaleX: elementBox.width / grid.screenshotW,
      scaleY: elementBox.height / grid.screenshotH,
      screenshotW: grid.screenshotW,
      screenshotH: grid.screenshotH,
    };

    const clickedOrder: number[] = [];
    let performedAction = false;
    let shouldSubmit = false;
    const allTokenUsage: TokenUsage[] = [];
    let pendingRetry = retryMode;

    this.initGridDebug();
    this.gridDebug('session:init', {
      attempt, retryMode, size: grid.size,
      screenshotW: grid.screenshotW, screenshotH: grid.screenshotH,
      scaleX: session.scaleX, scaleY: session.scaleY,
      elementBox, gridBoxes: session.gridBoxes,
    });

    for (let round = 1; round <= maxRounds; round++) {
      // 1. Settle and screenshot.
      //
      // Round 1 skips the wait: `solveSingle` has just paid it, read the grid
      // boxes off the loaded board and handed over, and nothing has touched
      // the widget in between. Rounds 2..N still wait, because by then THIS
      // driver has clicked and the tiles really are reloading.
      if (round > 1) await this.ph('grid-load', () => this.waitForGridCellsLoaded(captchaElement));
      const shotA = path.join(os.tmpdir(), `recap_${Date.now()}_${Math.floor(Math.random() * 1e9)}.png`);
      try {
        // Explicit timeout — after a Verify this element is being torn down,
        // and the 30s default waits for it to go stable first. See
        // screenshot-timeouts.test.ts.
        await this.ph('screenshot', () => captchaElement.screenshot({ path: shotA, timeout: 2500, animations: 'disabled' }));
      } catch {
        // The board is gone. Stop driving rounds and let the outer loop
        // re-detect — which, having interacted, reads "nothing left" as solved.
        break;
      }
      this.saveImageForDebug(shotA);
      // Per-round boundary snapshot for onStep observers. Round 1's snapshot is
      // the baseline (pre-action) for the 3x3 dynamic path.
      await this.emitStep(captchaElement, round === 1 ? 'initial' : 'round', `round-${round}:pre-solve`, 'recaptcha', 'challenge', attempt, { round });
      // Log the grid state the model is about to see — diagnostic only, so we
      // skip the extra state query unless grid debugging is active (keeps it off
      // the critical path in normal runs).
      if (this.gridDebugDir) {
        const preState = await this.gridCellStates(session, shotA, shotA);
        this.gridDebug(`round-${round}:pre-solve`, {
          round, pendingRetry,
          empty: preState?.empty ?? null, changing: preState?.changing ?? null,
          loaded: preState?.loaded ?? null, selected: preState?.selected ?? null,
          clickedOrder: [...clickedOrder],
        }, shotA);
      }

      let action: CaptchaAction | null = null;
      try {
        // 2. Solve. The CLI returns a single action for grid puzzles. Guarded
        //    against a mid-inference tile fade: if the grid changes while the
        //    model generates, re-screenshot and re-solve on the developed frame
        //    (the dynamic 3x3 puzzle is the case this matters most for).
        const retryForThisRound = pendingRetry;
        pendingRetry = null; // only the first round carries the inbound retry hint
        const response = await this.ph('inference', () => this.solveFrameFreshnessGuarded(
          captchaElement, shotA,
          (imagePath) => this.getSolution(imagePath, 'recaptcha', retryForThisRound),
        ));
        this.archiveLatestDebugRun(attempt, response.actions);
        allTokenUsage.push(...response.token_usage);
        const actionList = Array.isArray(response.actions) ? response.actions : [response.actions];
        action = actionList[0] ?? null;
        this.gridDebug(`round-${round}:action`, { action });
      } finally {
        if (fs.existsSync(shotA)) {
          try { fs.unlinkSync(shotA); } catch { /* best-effort cleanup */ }
        }
      }

      // 3. Dispatch on the action type.
      if (!action || action.action === 'done') {
        // Nothing matching remains → submit.
        console.log(`[recaptcha-grid] round ${round}: done; submitting.`);
        this.gridDebug(`round-${round}:done`, {});
        shouldSubmit = true;
        break;
      }

      if (action.action === 'wait') {
        // Tiles are still loading; the CLI explicitly told us NOT to submit.
        // Find what's loading, hover it, and wait for at least one to settle.
        console.log(`[recaptcha-grid] round ${round}: CLI says wait (${(action as any).duration_ms ?? 0}ms).`);
        await this.ph('fade-wait', async () => {
          const { loading } = await this.watchClickedTiles(page, captchaElement, session, clickedOrder);
          await this.waitForAnyClickedTileLoaded(page, captchaElement, session, loading);
        });
        continue;
      }

      if (action.action === 'click') {
        const c = action as ClickAction;
        const bboxes = c.target_bounding_boxes
          ?? (c.target_bounding_box ? [c.target_bounding_box] : []);
        if (!bboxes.length) {
          // Malformed click with no targets — treat as a soft wait so we don't
          // submit prematurely; re-solve next round.
          console.warn(`[recaptcha-grid] round ${round}: click action with no bboxes; re-solving.`);
          this.gridDebug(`round-${round}:click-no-bboxes`, {});
          await delay(500);
          continue;
        }

        // 4. Click the tiles in order, tracking cell numbers for hover ordering.
        const clickedThisRound: number[] = [];
        for (const bbox of bboxes) {
          const cell = this.bboxToCell(bbox, session.gridBoxes, session.screenshotW, session.screenshotH);
          await this.executeClick(page, captchaElement, { action: 'click', target_bounding_box: bbox } as ClickAction, elementBox);
          if (cell != null) {
            clickedOrder.push(cell);
            clickedThisRound.push(cell);
          }
          await this.human.pause('between');
        }
        performedAction = true;
        console.log(`[recaptcha-grid] round ${round}: clicked ${bboxes.length} tile(s) -> cells ${JSON.stringify(clickedThisRound)}.`);
        this.gridDebug(`round-${round}:clicked`, { bboxes, clickedThisRound });
        await this.emitStep(captchaElement, 'click', `round-${round}:clicked ${bboxes.length} tile(s)`, 'recaptcha', 'challenge', attempt, { round, clickedThisRound, bboxes });

        // 5. The clicked tiles either wear the chip (the widget kept the photo:
        //    this board is answered) or go blank / fade out for a replacement
        //    (dynamic puzzle: read it again). reCAPTCHA's reply lags the click,
        //    so we watch a grace window, not a single instant-after snapshot.
        const { loading, chipped } = await this.ph('fade-wait', () => this.watchClickedTiles(page, captchaElement, session, clickedThisRound));
        if (chipped || !loading.length) {
          // Either the widget ticked our clicks and kept the photos — a board
          // that does that is fully answered by the round that clicked it — or
          // nothing loaded within the grace window. Submit rather than paying
          // for another round to be told the same thing.
          console.log(`[recaptcha-grid] round ${round}: ${chipped ? 'tiles chipped' : 'no tiles loading'} after click; submitting.`);
          this.gridDebug(`round-${round}:${chipped ? 'chipped-submit' : 'no-loading-submit'}`, {});
          shouldSubmit = true;
          break;
        }
        // Tiles are reloading — wait (while hovering) for at least one to settle
        // before re-solving, so we don't feed the model a mid-fade grid.
        console.log(`[recaptcha-grid] round ${round}: tiles loading ${JSON.stringify(loading)}; waiting.`);
        await this.ph('fade-wait', () => this.waitForAnyClickedTileLoaded(page, captchaElement, session, loading));
        continue;
      }

      // Unexpected action type for a grid (drag/type) — re-solve.
      console.warn(`[recaptcha-grid] round ${round}: unexpected action '${(action as any).action}'; re-solving.`);
      this.gridDebug(`round-${round}:unexpected-action`, { action });
    }

    // Submit: click Verify if present (no-op if the grid is gone). Only when the
    // CLI signalled `done` or the widget chipped our clicks — never on a
    // timeout/round-cap exit, which leaves the outer loop to re-detect and
    // decide.
    if (shouldSubmit) {
      const frame = await captchaElement.contentFrame();
      if (frame) {
        const verifyButton = await this.getVerifyButton(frame);
        if (verifyButton) {
          console.log('[recaptcha-grid] clicking Verify to submit.');
          await this.moveAndClick(page, verifyButton);
          // The press IS an interaction, and saying so is load-bearing: the
          // caller polls for the vendor's verdict on a round that interacted
          // and sleeps postSolveDelayMs flat on one that did not — and throws
          // 'performed no interactions' if the widget outlives that sleep. A
          // `done` round clicks no tile, so without this line the one answer
          // shape that submits and nothing else reports having done nothing.
          performedAction = true;
          await this.emitStep(captchaElement, 'submit', 'submitted (Verify)', 'recaptcha', 'challenge', attempt);
          // Snapshot the submitted frame, exactly as the one-shot path does.
          // Without it isChallengeFreshlyRendered reads the CLOSING board as a
          // fresh round, breaks the post-submit solved-poll on its first tick,
          // and commits the solver to a full re-detect against a dying iframe —
          // measured at 13s per solve, on top of a 12s solve.
          this.lastSubmitFrameHash = await this.elementFrameHash(captchaElement).catch(() => null);
        }
      }
    }

    return { didInteract: performedAction, tokenUsage: allTokenUsage };
  }

  /**
   * Record the animated challenge and return the directory holding the burst.
   *
   * The driver owns the browser, so it does the recording; the CLI does the
   * slicing (`solve-animated`). One zero-padded PNG per frame, because the slicer
   * sorts by name and reads the clip's temporal structure — `frame_9.png` sorting
   * after `frame_10.png` would shuffle the burst and turn a detectable cycle into
   * noise.
   *
   * Geometry comes from config and defaults to the collector's (4s @ 10fps), so a
   * challenge recorded here is the same shape of artifact the model trained on.
   * The caller must remove the directory when the actions are done with — the
   * keyframes the wait gate re-reads on every poll live inside it.
   */
  private async recordKeyframeBurst(captchaElement: ElementHandle): Promise<string> {
    const rec = this.startKeyframeBurst(captchaElement);
    return rec.finish();
  }

  /**
   * Start recording NOW, and let the caller decide later what it was for.
   *
   * This is what makes the speculative path possible. The burst is the only
   * thing that can tell a cycling board from a still one, and it is also the
   * recording that answers it — so it is started at the same moment the still
   * screenshot goes to the model, and the two run together.
   *
   *   - the board never moves  -> `moved()` stays false, the still answer is
   *     used, and the frames are thrown away. The recording cost nothing but
   *     CPU, because it happened inside an inference we were already waiting
   *     for.
   *   - the board moves        -> the still answer is DISCARDED unread, the
   *     burst runs on to the end of the cycle, and the multi-image answer is
   *     the one acted on. Two inference CALLS, but only about one inference of
   *     wall-clock, because the first ran during the recording.
   *
   * The old shape paid for that knowledge in whole ROUNDS: answer a still, act
   * on it, notice nothing changed, answer another still, and only then record.
   * ~15s of a 40s solve to learn something the first four frames already knew.
   */
  private startKeyframeBurst(captchaElement: ElementHandle): {
    /** Has the widget shown more than one picture yet? */
    moved: () => boolean;
    /** Stop now and delete the frames — the board turned out to be still. */
    abandon: () => Promise<void>;
    /** Run to the end of the cycle and return the directory. */
    finish: () => Promise<string>;
  } {
    const fps = Math.max(1, this.config.videoBurstFps ?? 10);
    const durationMs = this.config.videoBurstDurationMs ?? 4000;
    const floorFrames = Math.max(1, Math.round(durationMs / (1000 / fps)));
    const ceilingMs = this.config.videoBurstMaxMs ?? 12_000;
    const total = Math.max(floorFrames, Math.round(ceilingMs / (1000 / fps)));
    const intervalMs = 1000 / fps;

    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'ck_burst_'));
    const order: string[] = [];        // distinct screens, in first-seen order
    let captured = 0;
    let lastDigest: string | null = null;
    let cycleClosed = false;
    let stopped = false;
    let runToEnd = false;              // set by finish(): keep going past the floor

    const loop = (async () => {
      for (let i = 0; i < total && !stopped; i++) {
        const started = Date.now();
        const frame = path.join(dir, `frame_${String(i).padStart(4, '0')}.png`);
        try {
          await captchaElement.screenshot({
            path: frame,
            timeout: this.config.elementScreenshotTimeoutMs ?? 8000,
            // ANIMATIONS STAY ON. Everywhere else in this file screenshots are
            // taken with animations: 'disabled', which is right when the goal is
            // a stable still — it fast-forwards finite animations and FREEZES
            // infinite ones. Here the motion IS the subject, so freezing it
            // records the same picture forty times: GeeTest's svg board cycles
            // in CSS, and the slicer correctly reported the burst as
            // `mode=static` and cut it to a single keyframe, putting us back to
            // answering a still. hCaptcha's animated challenges hid this because
            // they animate in canvas, which that flag does not touch.
            animations: 'allow',
          });
          captured++;
          try {
            const d = createHash('sha1').update(fs.readFileSync(frame)).digest('hex');
            if (d !== lastDigest) {
              // A screen already recorded, coming back after another one: the
              // loop has closed and every screen is now in the clip.
              if (order.includes(d) && order.length >= 2) cycleClosed = true;
              else if (!order.includes(d)) order.push(d);
              lastDigest = d;
            }
          } catch { /* a digest we could not take just means no early stop */ }
        } catch {
          // A dropped frame costs a sample, not the recording.
        }
        // Past the floor and the cycle has closed — anything more is the same
        // screens again, paid for in wall-clock the solve budget needs.
        if (runToEnd && cycleClosed && i + 1 >= floorFrames) {
          console.log(
            `[animated] cycle closed after ${((i + 1) * intervalMs / 1000).toFixed(1)}s `
            + `(${order.length} screens); stopping the burst`,
          );
          break;
        }
        // Drift-corrected: a slow screenshot must not stretch the clip, or the
        // burst covers more wall-clock than the model trained on and a cycle's
        // period lands differently across the frames.
        const wait = intervalMs - (Date.now() - started);
        if (wait > 0 && i < total - 1) await delay(wait);
      }
    })();

    return {
      // More than one picture seen. This is the ONLY question the speculative
      // caller needs answered, and the burst answers it for free.
      moved: () => order.length >= 2,

      abandon: async () => {
        stopped = true;
        try { await loop; } catch { /* the recording never fails a solve */ }
        fs.rmSync(dir, { recursive: true, force: true });
      },

      finish: async () => {
        runToEnd = true;
        // THE ESCALATION BUYS ITS OWN BUDGET, once per solve.
        //
        // `overallSolveTimeoutMs` counts rounds and a recording is not a round.
        // Granted HERE rather than when the burst starts, because a speculative
        // burst that turns out to be unnecessary must not extend the deadline
        // of a still solve. Recorded as an EXTENSION rather than by loosening
        // the config, so a solve that never escalates keeps the deadline the
        // caller asked for. Same one-shot, same total, same reason as
        // page_solver.py's `video_budget_ms`.
        if (this.config.videoSolveEnabled !== false && !this.videoBudgetGranted) {
          this.videoBudgetGranted = true;
          this.videoBudgetMs =
            (this.config.videoBurstMaxMs ?? 12_000) +
            (this.config.keyframeWaitTimeoutMs ?? SOLVE_DEFAULTS.keyframeWaitTimeoutMs) +
            (this.config.videoExtraInferenceMs ?? 8000);
        }
        try { await loop; } catch { /* fall through to the captured-count check */ }
        if (!captured) {
          fs.rmSync(dir, { recursive: true, force: true });
          const e: any = new Error(
            'ANIMATED_CHALLENGE: could not record the animated challenge (no frame screenshotted).',
          );
          e.animated = true;
          throw e;
        }
        console.log(`[animated] recorded ${captured} frames at ${fps}fps -> ${dir}`);
        return dir;
      },
    };
  }

  /**
   * Slice a recorded burst into keyframes and solve them in ONE model request.
   *
   * Deliberately not routed through `solveFrameFreshnessGuarded`. That guard
   * re-solves when the frame changes during inference, and an animated challenge
   * changes by definition — every attempt would be judged stale and the whole
   * re-solve budget would burn without ever acting. The `frame` in the answer is
   * the real guard: it names the state to act in, and `waitForKeyframe` enforces it.
   *
   * Also not deduped by screenshot hash: there is no single screenshot, and two
   * recordings of the same widget are never byte-identical anyway.
   */
  private async getAnimatedSolution(framesDir: string): Promise<CliResponse> {
    // resolveCli() FIRST: the default model is read out of the bundled engine's
    // models.json, so the name and the prompt generation it selects come from
    // the same place the Python port reads them. See model-name.ts.
    const { cliRoot, py } = this.resolveCli();
    const {
      model = this.loraName(cliRoot),
      apiKey = process.env.CAPTCHA_KRAKEN_API_KEY ?? process.env.VLLM_API_KEY,
    } = this.config;

    const args = [
      '-m', 'captchakraken.cli', 'solve-animated',
      '--frames-dir', framesDir,
      '--fps', String(this.config.videoBurstFps ?? 10),
      '--model', model,
    ];
    // NOT `args.push('--api-key', apiKey)`: a flag value is argv just as much as
    // a positional is, and argv is world-readable on Linux. Env, same as the
    // still-image path.

    try {
      // execFile (no shell): the temp dir path is ours but still goes through
      // literally, with no quoting hazard.
      const { stdout, stderr } = await execFileAsync(py, args, {
        cwd: cliRoot,
        env: solveEnv(
          cliEnv(cliRoot, this.solveSessionId ? { CAPTCHA_KRAKEN_SESSION: this.solveSessionId } : undefined),
          apiKey,
        ),
        maxBuffer: 10 * 1024 * 1024,
      });
      if (stderr) console.error('CaptchaKraken CLI stderr:', stderr);
      const parsed = JSON.parse(stdout.trim());
      this.keyframeMode = parsed.keyframe_mode ?? null;
      this.keyframeSteadyScreens = parsed.steady_screens ?? 0;
      console.log(
        `[animated] ${parsed.source_frames} frames -> ${(parsed.keyframes ?? []).length} `
        + `keyframe(s) (mode=${parsed.keyframe_mode})`,
      );
      return { actions: parsed.actions ?? [], token_usage: parsed.token_usage ?? [] };
    } catch (error: any) {
      const stderr: string = error.stderr ?? '';
      if (/"unsupported"\s*:\s*true/.test(stderr)) {
        const e = new Error('UNSUPPORTED_CAPTCHA: Cannot solve this animated captcha');
        (e as any).unsupported = true;
        throw e;
      }
      const apiError = parseApiError(stderr);
      if (apiError) throw apiError;
      console.error('Error executing CaptchaKraken solve-animated:', error);
      throw new Error(`Failed to execute the animated captcha solver: ${error.message}`);
    }
  }

  /**
   * Hold until the widget looks like `keyframePath` around the 0–1 point (cx, cy).
   *
   * This is the reason an animated answer names a frame. The model picked the
   * moment its target was visible, and the coordinates are only correct at that
   * moment; clicking as soon as the answer arrives lands on whatever the sprite
   * happens to be doing, which for a cross-fade is usually background.
   *
   * Only the neighbourhood of the action point is compared, with the same box and
   * metric the training label's frame was chosen with. Local rather than
   * whole-frame because everything ELSE in these puzzles is also moving: a
   * whole-frame match would need every unrelated sprite to align too, and would
   * essentially never open.
   *
   * Never throws. Returns whether the state was reached; on timeout the caller
   * clicks anyway (see `keyframeWaitTimeoutMs`).
   *
   * NOT ATTEMPTED on an `even` clip. The slicer picks that mode precisely when
   * the clip never revisits a picture it has already shown — a rotation, a
   * one-way fade, a sprite crossing — so there is no state to come back to and
   * this can only run out its full 6s, PER CLICK, before clicking the
   * coordinates it already had. It is also the normal case, not a corner: all
   * 116 real clips under cleanSamples/test/raw are `even` and `cycle` has never
   * fired on real footage. Measured on hcaptcha_rotating_obj_video: 6.0s of a
   * 28.8s solve, closest region diff 0.0721 against a 0.05 tolerance, then the
   * same click, then solved. Kept for `cycle`/`static`, where the state does
   * come back and waiting is the difference between the sprite and background.
   */
  private async waitForKeyframe(
    captchaElement: ElementHandle,
    keyframePath: string,
    cx: number,
    cy: number,
  ): Promise<boolean> {
    // NOT `mode === 'even'`. See `keyframeSteadyScreens`: a board that holds a
    // few screens is worth waiting for whether or not one burst was long enough
    // to catch it repeating. What is NOT worth waiting for is a clip with no
    // steady screens at all — a rotation, a one-way fade, a sprite crossing —
    // where the gate can only ever time out. Measured: 0 steady screens for all
    // five continuous hCaptcha video types, 2-3 for every real GeeTest svg.
    if (this.keyframeSteadyScreens < 2) {
      console.log(
        `[animated] clip sits on ${this.keyframeSteadyScreens} steady screen(s); `
        + "nothing to come back to, acting on the model's frame without waiting",
      );
      return false;
    }
    const timeout = this.config.keyframeWaitTimeoutMs ?? SOLVE_DEFAULTS.keyframeWaitTimeoutMs;
    const interval = this.config.keyframeWaitPollMs ?? 120;
    // Never past the solve's own deadline: waiting for a screen we no longer
    // have time to click is pure overrun. Mirrors `_check_deadline` in the
    // Python port's wait loop.
    const deadline = Math.min(
      Date.now() + timeout,
      this.solveDeadlineAt || Number.MAX_SAFE_INTEGER,
    );
    let polls = 0;
    const probe = path.join(os.tmpdir(), `ck_kfwait_${Date.now()}_${Math.floor(Math.random() * 1e9)}.png`);
    let best = 1;
    try {
      while (Date.now() < deadline) {
        try {
          await captchaElement.screenshot({
            path: probe,
            timeout: this.config.elementScreenshotTimeoutMs ?? 8000,
            // 'allow', NOT 'disabled'. THE MOTION IS THE THING BEING WATCHED.
            //
            // Playwright's `disabled` fast-forwards finite animations and
            // FREEZES infinite ones — so a board cycling in CSS is photographed
            // in the same frozen state on every poll, and a gate whose whole
            // job is to notice the screen changing could never notice it. It
            // either matched immediately (on the frozen frame, i.e. whichever
            // screen the freeze happened to catch) or ran the full budget out.
            //
            // `recordKeyframeBurst` already learned this and says so; the probe
            // that watches for the SAME motion was still freezing it.
            animations: 'allow',
          });
          const r = await this.runCvTool(
            'match-region',
            { ref: keyframePath, live: probe, cx, cy },
            ['match-region', keyframePath, probe, String(cx), String(cy)],
          );
          if (typeof r?.diff === 'number') best = Math.min(best, r.diff);
          if (r?.match) {
            console.log(`[animated] widget matched the chosen keyframe (diff=${r.diff.toFixed(4)})`);
            return true;
          }
          // NOT THIS BOARD AT ALL — stop waiting for a screen of it.
          //
          // The screens of one board differ from each other by very little:
          // 0.0056 measured on GeeTest svg, where a few glyph strokes change.
          // A different board reads 0.77. So a diff this large is not "the
          // wrong screen is up", it is "the puzzle we recorded is gone" —
          // solved, refreshed, or replaced — and the remaining budget buys
          // nothing. It was spending the full 9s on this, once per solve,
          // AFTER the click that had already succeeded.
          polls++;
          if (polls >= NOT_THIS_BOARD_POLLS && best > NOT_THIS_BOARD_DIFF) {
            console.log(
              `[animated] the widget no longer resembles the recorded board `
              + `(best diff=${best.toFixed(4)} over ${polls} polls); not waiting out the budget`,
            );
            this.discardAnimatedPlan();
            return false;
          }
        } catch {
          // A failed probe is one lost poll, not a failed solve.
        }
        await delay(interval);
      }
    } finally {
      if (fs.existsSync(probe)) { try { fs.unlinkSync(probe); } catch { /* best-effort */ } }
    }
    // NEVER SAW IT. Either the board moved on to a different challenge, or the
    // answer was for a screen this widget does not show. Both mean the plan is
    // spent, so the next round records afresh rather than re-clicking a cell
    // chosen from pictures that are gone.
    this.discardAnimatedPlan();
    console.log(
      `[animated] widget never matched the chosen keyframe within ${timeout}ms `
      + `(closest diff=${best.toFixed(4)}); clicking on the model's coordinates anyway, `
      + `and recording afresh next round`,
    );
    return false;
  }

  /**
   * Per-solve state that must not leak into the next challenge on the page.
   *
   * A repeat is a fact about ONE challenge. Carrying it forward would make the
   * captcha after a cycling one record a burst it does not need.
   */
  private resetSolveState(): void {
    this.solutionCache.clear();
    this.repeatedAnswerSeen = false;
    this.discardAnimatedPlan();
    this.lastSubmitFrameHash = null;
    this.keyframeMode = null;
    this.keyframeSteadyScreens = 0;
    this.solveDeadlineAt = 0;
    this.lastAnswerSig = null;
    this.noProgressRounds = 0;
    // Per SOLVE, not per process: a grant leaking into the next captcha would
    // silently hand a still puzzle 18s it was never meant to have.
    this.videoBudgetMs = 0;
    this.videoBudgetGranted = false;
  }

  /**
   * A stable identity for "the answer this round is about to execute".
   *
   * Keyed on the retry mode too: the missed-tiles retry deliberately re-asks
   * about the same board and its answer legitimately overlaps the previous one,
   * so counting that as a repeat would abandon the one path built to recover
   * from an under-selection.
   *
   * Coordinates are ROUNDED rather than compared exactly — the same tile chosen
   * twice can differ in the last float digit after the normalise/clamp
   * round-trip, and a repeat that reads as "different" is a repeat that costs a
   * round. Returns null when the answer cannot be summarised, which counts as
   * "not a repeat": an unreadable answer must never be why a solve is
   * abandoned. Mirrors page_solver.py `_answer_signature`.
   */
  private static answerSignature(actions: any[], retryMode: string | null): string | null {
    const round3 = (v: any): any => {
      if (typeof v === 'number') return Math.round(v * 1000) / 1000;
      if (Array.isArray(v)) return v.map(round3);
      return v ?? null;
    };
    try {
      return JSON.stringify([retryMode ?? null, actions.map((a: any) => [
        a?.action ?? null,
        round3(a?.target_bounding_boxes),
        round3(a?.target_bounding_box),
        round3(a?.target_coordinates),
        round3(a?.source_bounding_box),
        a?.text ?? null,
      ])]);
    } catch {
      return null;
    }
  }

  /** Count consecutive identical answers; see `maxNoProgressRounds`. */
  private noteAnswer(actions: any[], retryMode: string | null): void {
    const sig = CaptchaKrakenSolver.answerSignature(actions, retryMode);
    if (sig !== null && sig === this.lastAnswerSig) {
      this.noProgressRounds++;
      console.log(
        `[no-progress] the model returned the same answer again `
        + `(${this.noProgressRounds}/${this.config.maxNoProgressRounds ?? 2}) — `
        + `the previous one already ran and changed nothing`,
      );
      // A board that reads the same every round is the signature of a CYCLING
      // challenge answered as a still. Let the recording path have a go before
      // giving up on the solve entirely.
      this.repeatedAnswerSeen = true;
    } else {
      this.noProgressRounds = 0;
      this.lastAnswerSig = sig;
    }
  }

  /**
   * The answer for this picture, asking the model only if we have not already.
   *
   * The cache saving is real and is kept: a byte-identical picture costs no
   * second inference. What changed is what a hit MEANS. It used to mean
   * "nothing has changed, so this answer still stands"; it actually means the
   * answer already ran and moved nothing, because every answer this returns is
   * executed. So the hit is served (it is free) and recorded, and the round
   * after it stops guessing at a still frame — see `shouldRetryAsAnimated`.
   */
  private async answerFor(cacheKey: string, ask: () => Promise<CliResponse>): Promise<CliResponse> {
    const cached = this.solutionCache.get(cacheKey);
    if (cached) {
      console.log(
        '[dedup] this exact picture was already answered and the answer already ran — '
        + 'the challenge is cycling, not still; re-solving it as animated.',
      );
      this.repeatedAnswerSeen = true;
      // Reuse the actions but drop the token usage (no new tokens were spent).
      return { actions: cached.actions, token_usage: [] };
    }
    const fresh = await ask();
    this.solutionCache.set(cacheKey, fresh);
    return fresh;
  }

  /**
   * Should this round be recorded and solved from keyframes instead of read as
   * a still?
   *
   * reCAPTCHA is excluded deliberately. Its dynamic 3x3 REPLACES tiles in place
   * and has its own multi-round driver with its own fade gates; its grids are
   * never animated, so escalating there would swap a path that works for one
   * that cannot read a grid.
   */
  /** Forget the recorded answer, and delete the frames it was holding. */
  private discardAnimatedPlan(): void {
    const dir = this.animatedPlan?.burstDir;
    this.animatedPlan = null;
    if (dir && fs.existsSync(dir)) {
      try { fs.rmSync(dir, { recursive: true, force: true }); } catch { /* best-effort */ }
    }
  }

  /**
   * Should this round record while it asks?
   *
   * reCAPTCHA is excluded for the same reason it is excluded from the animated
   * path at all: its dynamic 3x3 REPLACES tiles in place, has its own
   * multi-round driver with its own fade gates, and its grids are never
   * animated — so a burst there would film a fade and call it a cycle.
   *
   * Distorted-text rounds are excluded because the answer is a string, not a
   * place, and nothing about a recording helps read one.
   */
  private shouldSpeculate(puzzleSource: 'hcaptcha' | 'recaptcha' | 'unknown', textMode: boolean): boolean {
    if (this.config.videoSolveEnabled === false) return false;
    if (this.config.speculativeBurstEnabled === false) return false;
    if (puzzleSource === 'recaptcha') return false;
    if (textMode) return false;
    return true;
  }

  private shouldRetryAsAnimated(puzzleSource: 'hcaptcha' | 'recaptcha' | 'unknown'): boolean {
    if (!this.repeatedAnswerSeen) return false;
    if (puzzleSource === 'recaptcha') return false;
    return this.config.videoSolveEnabled !== false;
  }

  private async getSolution(imagePath: string, puzzleSource: 'hcaptcha' | 'recaptcha' | 'unknown' = 'unknown', retryMode: string | null = null, textMode = false): Promise<CliResponse> {
    // v2 ships a single provider: the CaptchaKraken vLLM server via the bundled
    // CaptchaKraken CLI. The CLI's planner reads VLLM_BASE_URL and the bearer
    // token (CAPTCHA_KRAKEN_API_KEY, falling back to VLLM_API_KEY) from the
    // environment; we also forward the key explicitly as a CLI arg below so it
    // works even when the subprocess doesn't inherit it. The LoRA name defaults
    // to whatever models.json calls `latest` — NOT a literal; a hardcoded
    // `captcha` here is what sent generation-1 prompts to a generation-2
    // adapter. Override via CAPTCHA_LORA_NAME.

    // Dedup on the screenshot's bytes under the same prompt (puzzle source +
    // retry mode). A hit costs no inference; what it MEANS is answerFor's job,
    // and it is not "nothing changed" — see the note on repeatedAnswerSeen.
    let cacheKey: string | null = null;
    try {
      const imgHash = createHash('sha1').update(fs.readFileSync(imagePath)).digest('hex');
      cacheKey = `${imgHash}|${puzzleSource}|${retryMode ?? ''}|${textMode ? 'text' : ''}`;
    } catch {
      cacheKey = null; // hashing failed — fall through to a normal query
    }
    if (cacheKey) return this.answerFor(cacheKey, () => this.askModel(imagePath, puzzleSource, retryMode, textMode));
    return this.askModel(imagePath, puzzleSource, retryMode, textMode);
  }

  /** One inference: build the CLI invocation, run it, parse what comes back. */
  private async askModel(imagePath: string, puzzleSource: 'hcaptcha' | 'recaptcha' | 'unknown' = 'unknown', retryMode: string | null = null, textMode = false): Promise<CliResponse> {
    // resolveCli() FIRST — see getAnimatedSolution.
    const { cliRoot, py } = this.resolveCli();
    const {
      model = this.loraName(cliRoot),
      apiKey = process.env.CAPTCHA_KRAKEN_API_KEY ?? process.env.VLLM_API_KEY,
    } = this.config;

    // An ARGV ARRAY for execFile, not a string for a shell.
    //
    // This was `cmdParts.join(' ')` handed to `exec`, so the bearer token sat in
    // argv — world-readable at /proc/<pid>/cmdline for the life of the solve —
    // and the same string was then printed to stdout, putting it in CI logs and
    // scrollback. The key now travels in the environment (see cli-invocation.ts),
    // and dropping the shell also removes the quoting hazard: `"${imagePath}"`
    // was hand-quoted, which a path containing a quote defeats.
    //
    // The vendor hint goes on as --puzzle-source=<vendor>; the CLI's argparse
    // falls through to the flag form for unknown trailing args.
    const args = buildSolveArgs({
      imagePath,
      model,
      puzzleSource,
      retryMode,
      // The DOM said this puzzle has a text box, so the CLI must send the
      // distorted-text prompt and skip grid detection. The picture alone cannot
      // decide this — see the textMode note in solveSingle.
      textMode,
      // Undefined unless the caller pinned one, which is the normal case: a
      // routed model picks its own expert from the prompt family the CLI is
      // about to send.
      expert: this.config.expert,
    });

    console.log(
      `Executing CaptchaKraken CLI: ${redactCommand([py, ...args].join(' '), apiKey)}`
    );

    try {
      const { stdout, stderr } = await execFileAsync(py, args, {
        cwd: cliRoot,
        // Only this call reaches the model, so it's the only one that needs the
        // session id — the other cliEnv() call sites run pure-OpenCV subcommands
        // that never touch the inference endpoint. `solveEnv` adds the bearer
        // token here, which is what keeps it out of argv.
        env: solveEnv(
          cliEnv(cliRoot, this.solveSessionId ? { CAPTCHA_KRAKEN_SESSION: this.solveSessionId } : undefined),
          apiKey,
        ),
        maxBuffer: 10 * 1024 * 1024 // Increase buffer for large outputs if needed
      });

      console.log('CaptchaKraken CLI stdout:', stdout);
      if (stderr) {
        console.error('CaptchaKraken CLI stderr:', stderr);
      }

      if (!stdout.trim()) {
        throw new Error(`CLI returned empty output. Stderr: ${stderr}`);
      }

      try {
        const lines = stdout.trim().split('\n');
        let actions: SolverResult = [];
        let tokenUsage: TokenUsage[] = [];

        for (const line of lines) {
          try {
            const parsed = JSON.parse(line);

            // Handle new format { actions: ..., token_usage: ... }
            if (parsed.actions !== undefined && parsed.token_usage !== undefined) {
              actions = parsed.actions;
              tokenUsage = parsed.token_usage;
              break;
            }

            // Fallback for old format or list of actions
            if (Array.isArray(parsed)) {
              actions = parsed;
            } else if (parsed.action && (parsed.target_bounding_box || parsed.target_coordinates || parsed.action === 'wait')) {
              actions = [parsed];
            }
          } catch (e) {
            // Not json or not relevant
          }
        }

        // Caching is answerFor's, so that what a hit MEANS lives in one place.
        return { actions, token_usage: tokenUsage };
      } catch (parseError) {
        throw new Error(`Failed to parse CLI output: ${stdout}\nStderr: ${stderr}`);
      }

    } catch (error: any) {
      // The CLI emits {"unsupported": true} (exit 2) when the current frame is
      // neither a grid nor a checkbox — e.g. an hCaptcha click/drag puzzle.
      // Surface that as a distinct error the solve loop can recognize and fail
      // fast on (only the puzzle TYPE is unsupported; a not-yet-rendered widget
      // is handled separately by DOM-presence waiting before we ever get here).
      const stderr: string = error.stderr ?? '';
      if (/"unsupported"\s*:\s*true/.test(stderr)) {
        const e = new Error('UNSUPPORTED_CAPTCHA: Cannot solve this kind of captcha');
        (e as any).unsupported = true;
        throw e;
      }

      // The hosted API refused the solve and said why (out of credits, rate
      // limited, attempt abandoned…). The CLI exits 3 with the already-worded
      // sentence plus the machine-readable fields; both are rethrown as-is.
      //
      // Wrapping this in "Failed to execute captcha solver CLI: Command failed
      // …" — which is what happens to every other non-zero exit — would bury a
      // billing problem under a sentence about a subprocess. camoufox surfaces
      // whatever we throw, so this is the message its users actually read.
      const apiError = parseApiError(stderr);
      if (apiError) throw apiError;

      console.error('Error executing CaptchaKraken CLI:', error);
      if (error.stdout) console.log('CLI stdout on error:', error.stdout);
      if (error.stderr) console.error('CLI stderr on error:', error.stderr);
      throw new Error(`Failed to execute captcha solver CLI: ${error.message}`);
    }
  }

  /**
   * Query the model for `initialShot`, then guard against the captcha frame
   * having changed DURING inference. reCAPTCHA/hCaptcha fade fresh tiles in over
   * ~1s; if new imagery painted while the model was generating, its answer
   * targets a stale ("undeveloped") frame and its tile picks / bboxes no longer
   * line up with what's on screen. After the model returns we re-screenshot the
   * element and diff it against the frame we sent (reusing the `check-movement`
   * primitive); if it moved beyond the threshold we discard the answer and
   * re-solve on the fresh frame, up to `maxStaleFrameReSolves` times, then act on
   * the latest answer rather than spin.
   *
   * `runQuery` performs the actual model call for a given screenshot path (the
   * caller wraps it in idle-wander where appropriate); it is re-invoked with the
   * fresh path on each re-solve. Token usage from every query — including
   * discarded ones — is accumulated, since those tokens were really spent. The
   * caller owns `initialShot`; every fresh frame captured here is created and
   * cleaned up here. Best-effort: a screenshot/diff failure falls through to the
   * answer already in hand.
   */
  private async solveFrameFreshnessGuarded(
    captchaElement: ElementHandle,
    initialShot: string,
    runQuery: (imagePath: string) => Promise<CliResponse>,
  ): Promise<CliResponse> {
    const enabled = this.config.staleFrameReSolveEnabled !== false;
    const threshold = this.config.staleFrameDiffThreshold ?? 0.02;
    const maxReSolves = this.config.maxStaleFrameReSolves ?? 2;

    const ownedFrames: string[] = []; // fresh frames WE captured (never initialShot)
    const mergedUsage: TokenUsage[] = [];
    // Declared out here so the `finally` can remove it.
    const liveAnchor = path.join(
      os.tmpdir(), `ck_move_${Date.now()}_${Math.floor(Math.random() * 1e9)}.png`);
    try {
      let currentPath = initialShot;

      // An anchor for the movement question, taken LIVE — the screenshot sent
      // to the model was taken with animations frozen, so comparing against it
      // asks "does the live widget differ from a frozen one", which is a
      // different and much noisier question than "did it move".
      let haveAnchor = false;
      try {
        await captchaElement.screenshot({ path: liveAnchor, timeout: 2500, animations: 'allow' });
        haveAnchor = true;
      } catch { /* no anchor, no movement check — never fails a solve */ }

      let response = await runQuery(currentPath);
      mergedUsage.push(...response.token_usage);

      if (!enabled) return response;

      // DID THE PICTURE MOVE WHILE THE MODEL READ IT? Asked once per round, at
      // the noise floor rather than the staleness threshold.
      //
      // A still puzzle answers no and pays one cheap frame diff. A cycling one
      // answers yes, and the driver can stop pretending it is a still a whole
      // round earlier than the "same answer came back twice" rule allows —
      // which cost two still inferences and ~15s of a 40s solve.
      //
      // Acting on ONE observation is safe because the recording is
      // self-checking: a widget that turns out not to move slices to a single
      // keyframe and is solved as the still it is, for the price of one burst.
      // page_solver.py makes that argument in `_settle_or_animated`.
      if (!this.repeatedAnswerSeen && haveAnchor
          && await this.captchaFrameChangedSince(
               captchaElement, liveAnchor, MOVED_DURING_INFERENCE_DIFF, 'allow')) {
        this.repeatedAnswerSeen = true;
        console.log(
          '[freshness] the widget moved while the model was reading it, with '
          + 'nothing clicked — recording it rather than answering another still.',
        );
      }

      let changedDuringInference = 0;
      for (let i = 0; i < maxReSolves; i++) {
        if (!(await this.captchaFrameChangedSince(captchaElement, currentPath, threshold))) {
          break; // frame held still through inference — the answer is valid.
        }
        // CHANGED AGAIN, having already re-solved once and touched nothing.
        //
        // One change is a board still developing — tiles fading in — which is
        // what the re-solve below is for. A SECOND change in the same round is
        // a board that does not stop, and re-solving it is futile: each answer
        // is for a screen that will be gone before the click. Recording it is
        // what works, and this is the earliest the driver can know.
        //
        // It used to learn it a round later, from the same answer coming back
        // twice — two still inferences and ~10s of a 40s solve spent
        // rediscovering something the guard had already watched happen.
        if (++changedDuringInference >= 2) {
          this.repeatedAnswerSeen = true;
          console.log(
            '[freshness] the frame changed twice during inference with nothing '
            + 'clicked — this board cycles; recording it rather than re-solving '
            + 'a screen that has gone.',
          );
          return { actions: response.actions, token_usage: mergedUsage };
        }
        const fresh = path.join(
          os.tmpdir(),
          `freshsolve_${Date.now()}_${Math.floor(Math.random() * 1e9)}.png`,
        );
        try {
          await captchaElement.screenshot({ path: fresh, timeout: 2500, animations: 'disabled' });
        } catch {
          break; // can't grab a fresh frame — act on the answer we have.
        }
        ownedFrames.push(fresh);
        this.saveImageForDebug(fresh);
        console.log(
          `[freshness] captcha frame changed during inference `
          + `(re-solve ${i + 1}/${maxReSolves}); the prior answer was for a stale `
          + `frame — re-querying on the developed one.`,
        );
        this.gridDebug('freshness:stale-frame', { reSolve: i + 1, threshold });
        currentPath = fresh;
        response = await runQuery(currentPath);
        mergedUsage.push(...response.token_usage);
      }
      return { actions: response.actions, token_usage: mergedUsage };
    } finally {
      if (fs.existsSync(liveAnchor)) { try { fs.unlinkSync(liveAnchor); } catch { /* best-effort */ } }
      for (const f of ownedFrames) {
        if (fs.existsSync(f)) {
          try { fs.unlinkSync(f); } catch { /* best-effort cleanup */ }
        }
      }
    }
  }

  /**
   * Screenshot the captcha element to a throwaway temp frame and diff it against
   * `priorPath` (the frame we sent the model). Returns true when the frame
   * changed beyond `threshold` — i.e. tiles faded in / refreshed since. Reuses
   * the persistent CV worker's `check-movement` (falling back to a one-shot
   * subprocess), the same frame-diff the settle detectors use. Cleans up its own
   * temp frame. Best-effort: any screenshot/diff failure returns false so the
   * caller acts on the answer it already has rather than spinning.
   */
  private async captchaFrameChangedSince(
    captchaElement: ElementHandle,
    priorPath: string,
    threshold: number,
    animations: 'allow' | 'disabled' = 'disabled',
  ): Promise<boolean> {
    const probe = path.join(
      os.tmpdir(),
      `freshcheck_${Date.now()}_${Math.floor(Math.random() * 1e9)}.png`,
    );
    try {
      await captchaElement.screenshot({ path: probe, timeout: 2500, animations: 'disabled' });
      const res = await this.runCvTool(
        'check-movement',
        { a: priorPath, b: probe, threshold },
        ['check-movement', priorPath, probe, String(threshold)],
      );
      return !!(res && res.has_movement);
    } catch {
      return false;
    } finally {
      if (fs.existsSync(probe)) {
        try { fs.unlinkSync(probe); } catch { /* best-effort cleanup */ }
      }
    }
  }

  /** Record a challenge-state transition. No-op for behaviour; logs the change
   *  (via gridDebug, so only when CAPTCHA_DEBUG=1) for offline diagnosis. */
  private setState(next: CaptchaState, note?: string): void {
    if (this.state === next) return;
    this.gridDebug('state', { from: this.state, to: next, ...(note ? { note } : {}) });
    this.state = next;
  }

  /**
   * Screenshot the challenge element to a throwaway file and return its sha1
   * content hash (or null on failure). Used to tell whether the frame has
   * changed (e.g. the post-submit transition to the next round). Cleans up.
   */
  private async elementFrameHash(el: ElementHandle): Promise<string | null> {
    const f = path.join(os.tmpdir(), `fh_${Date.now()}_${Math.floor(Math.random() * 1e9)}.png`);
    try {
      await el.screenshot({ path: f, timeout: 2500, animations: 'disabled' });
      return createHash('sha1').update(fs.readFileSync(f)).digest('hex');
    } catch {
      return null;
    } finally {
      if (fs.existsSync(f)) { try { fs.unlinkSync(f); } catch { /* best-effort */ } }
    }
  }

  /**
   * Monitor the challenge element until its pixels stop changing (settled), or
   * until it's clear it never will (animated / video). Polls screenshots and
   * diffs consecutive frames with `check-movement` (the persistent CV worker).
   * Returns:
   *   'settled'  — `settleFrames` consecutive frame-pairs showed no movement;
   *                safe to screenshot for the model.
   *   'animated' — it kept moving right up to `animatedChallengeAfterMs` without
   *                ever settling → very likely a video/animated puzzle.
   *   'timeout'  — neither happened within `settleTimeoutMs` (proceed best-effort).
   *
   * Note this is a *pixel* settle; the caller pairs it with the DOM-level
   * `waitForHcaptchaChallengeImages` so a static loading frame (spinner on grey,
   * below the pixel threshold) isn't mistaken for painted tiles. Cleans up.
   */
  private async waitForElementSettled(
    el: ElementHandle,
    opts?: { pollMs?: number; settleFrames?: number; maxMs?: number; animatedAfterMs?: number; threshold?: number },
  ): Promise<'settled' | 'animated' | 'timeout'> {
    const pollMs = opts?.pollMs ?? this.config.settlePollMs ?? 220;
    const settleFrames = opts?.settleFrames ?? this.config.settleFrames ?? 2;
    const maxMs = opts?.maxMs ?? this.config.settleTimeoutMs ?? 9000;
    const animatedAfterMs = opts?.animatedAfterMs ?? this.config.animatedChallengeAfterMs ?? 4500;
    const threshold = opts?.threshold ?? this.config.settleDiffThreshold ?? 0.01;
    const start = Date.now();
    let prev: string | null = null;
    let stillStreak = 0;
    const frames: string[] = [];
    const tmp = () => path.join(os.tmpdir(), `settle_${Date.now()}_${Math.floor(Math.random() * 1e9)}.png`);
    try {
      while (Date.now() - start < maxMs) {
        const f = tmp();
        // Short timeout + disabled animations: a closing/animating challenge
        // element otherwise makes Playwright's default 30s stability wait hang
        // per screenshot (that's what made a multi-round solve take ~115s). Fail
        // fast and skip the frame instead.
        try { await el.screenshot({ path: f, timeout: 2500, animations: 'disabled' }); }
        catch { await delay(pollMs); continue; }
        frames.push(f);
        if (prev) {
          const res = await this.runCvTool(
            'check-movement', { a: prev, b: f, threshold }, ['check-movement', prev, f, String(threshold)],
          );
          const moved = !!(res && res.has_movement);
          stillStreak = moved ? 0 : stillStreak + 1;
          const stale = frames.shift();
          if (stale && fs.existsSync(stale)) { try { fs.unlinkSync(stale); } catch { /* best-effort */ } }
          if (stillStreak >= settleFrames) return 'settled';
          // Still moving this late in → it's not just loading; call it animated.
          if (moved && (Date.now() - start) >= animatedAfterMs) return 'animated';
        }
        prev = frames[frames.length - 1];
        await delay(pollMs);
      }
      return 'timeout';
    } finally {
      for (const f of frames) {
        if (fs.existsSync(f)) { try { fs.unlinkSync(f); } catch { /* best-effort */ } }
      }
    }
  }

  /**
   * After a submit we EXPECT the challenge frame to change (advance to the next
   * round, or close because it was accepted). Poll until the element's content
   * hash differs from `sinceHash` — the transition beginning — so we never
   * screenshot/solve the pre-transition frame again. Returns true once it
   * changed, false if it never changed within `postSubmitChangeTimeoutMs` (e.g.
   * it was already the final state). Best-effort.
   */
  private async waitForChangeSince(
    el: ElementHandle,
    sinceHash: string,
    opts?: { pollMs?: number; maxMs?: number },
  ): Promise<boolean> {
    const pollMs = opts?.pollMs ?? this.config.settlePollMs ?? 220;
    const maxMs = opts?.maxMs ?? this.config.postSubmitChangeTimeoutMs ?? 4000;
    const start = Date.now();
    while (Date.now() - start < maxMs) {
      const h = await this.elementFrameHash(el);
      if (h && h !== sinceHash) return true;
      await delay(pollMs);
    }
    return false;
  }

  /**
   * Run `fn` (typically the model query) while idly drifting the cursor over
   * the captcha, so the mouse behaves like a human weighing the options instead
   * of freezing during inference. Uses the same humanizer as real clicks, so a
   * mode with no cursor skips it entirely; cancelled the instant `fn` resolves.
   * Best-effort — any wander error is swallowed and never fails the solve.
   * Disable via config.idleMouseWander.
   */
  private async withIdleWander<T>(
    page: Page,
    element: ElementHandle,
    fn: () => Promise<T>,
  ): Promise<T> {
    // Same rule as hoverCell: drifting a cursor that does not exist would emit
    // mousemove at a touch-only widget.
    if (this.config.idleMouseWander === false || !this.human.hovers) return fn();
    let box: { x: number; y: number; width: number; height: number } | null = null;
    try { box = await element.boundingBox(); } catch { box = null; }
    if (!box || box.width < 20 || box.height < 20) return fn();
    const b = box;

    let stop = false;
    // The dwell is INTERRUPTIBLE, and that is not a detail. `fn` is the model
    // call; the moment it returns there is a click to make, and the finally
    // below waits for this loop to notice. With a plain `delay` it noticed only
    // when the timer ran out, so every inference was followed by up to 540ms of
    // waiting for a pause nobody was watching — paid once per round, and the
    // Python port pays none of it because it does not wander at all. Same
    // wake-able-sleep shape as watcher.ts.
    const waker: { fn: (() => void) | null } = { fn: null };
    const nap = (ms: number) => new Promise<void>((resolve) => {
      const timer = setTimeout(() => { waker.fn = null; resolve(); }, ms);
      waker.fn = () => { clearTimeout(timer); waker.fn = null; resolve(); };
    });
    const pad = 0.18; // keep drift inside the tile area, off the extreme edges
    const wander = (async () => {
      // brief pause before the first drift — don't lurch the instant we ask
      await nap(120 + Math.random() * 180);
      while (!stop) {
        const tx = b.x + b.width * (pad + Math.random() * (1 - 2 * pad));
        const ty = b.y + b.height * (pad + Math.random() * (1 - 2 * pad));
        try {
          await this.performSmoothMove(page, tx, ty);
        } catch {
          break;
        }
        if (stop) break;
        await nap(180 + Math.random() * 360); // human dwell between glances
      }
    })();

    try {
      return await fn();
    } finally {
      stop = true;
      waker.fn?.();
      await wander.catch(() => {});
    }
  }

  // Simplified move function with smooth movement
  async move(
    page: Page,
    selectorOrElement: string | ElementHandle,
    options: { paddingPercentage?: number } = {}
  ): Promise<void> {
    let elem: ElementHandle | null = null;
    if (typeof selectorOrElement === 'string') {
      elem = await page.waitForSelector(selectorOrElement, { state: 'visible', timeout: 10000 });
    } else {
      elem = selectorOrElement;
    }

    if (!elem) {
      throw new Error(`Element not found: ${selectorOrElement}`);
    }

    // BOUNDED. Playwright's default is 30s and it waits for the element to be
    // STABLE — not animating — before it will scroll. This runs once per action
    // and once per submit, so on a challenge that is mid-animation it burned
    // the full default every time: measured 10.1s of a 12.0s mtcaptcha_text
    // solve, spent scrolling to a text box that was already on screen. The
    // element is on screen in every real case here (we just screenshotted it),
    // so a short bound loses nothing: on timeout we move to wherever it is.
    // page_solver.py has had this since it turned a ~5s solve loop into minutes
    // in live testing; this port never got it.
    try {
      await elem.scrollIntoViewIfNeeded({ timeout: 2000 });
    } catch {
      /* an unscrolled element is still where boundingBox says it is */
    }

    const box = await elem.boundingBox();
    if (!box) {
      throw new Error(`Element has no bounding box: ${selectorOrElement}`);
    }

    // Default padding 25% to stay well inside the element
    const padding = (options.paddingPercentage || 25) / 100;
    const padX = box.width * padding;
    const padY = box.height * padding;

    // Pick a random point within the padded area
    const targetX = box.x + padX + Math.random() * (box.width - 2 * padX);
    const targetY = box.y + padY + Math.random() * (box.height - 2 * padY);

    await this.performSmoothMove(page, targetX, targetY);
  }

  async moveAndClick(page: Page, element: ElementHandle) {
    await this.move(page, element);
    await this.ph('mouse', () => this.human.click(page, this.human.at));
  }

  /**
   * Travel to a point. What that MEANS is the humanizer's business — on a
   * touchscreen with no finger down it is a bookkeeping update and emits
   * nothing at all.
   */
  private async performSmoothMove(page: Page, x: number, y: number) {
    await this.ph('mouse', () => this.human.move(page, [x, y]));
  }

  /**
   * Where in the element this click lands, in element-relative pixels.
   *
   * Split out of `executeClick` because on an animated board the point has to
   * be chosen ONCE and then used three times — to park the pointer, to ask
   * `match-region` about the right neighbourhood, and to press. Choosing it
   * inside the click (it is randomised within the target) meant the gate
   * watched the bbox centre while the press landed somewhere else.
   */
  private clickPointFor(
    action: ClickAction,
    elementBox: { x: number, y: number, width: number, height: number },
  ): [number, number] | null {
    if (action.target_bounding_box) {
      const [minX, minY, maxX, maxY] = action.target_bounding_box;
      const pixelMinX = minX * elementBox.width;
      const pixelMaxX = maxX * elementBox.width;
      const pixelMinY = minY * elementBox.height;
      const pixelMaxY = maxY * elementBox.height;
      const paddingX = (pixelMaxX - pixelMinX) * 0.1;
      const paddingY = (pixelMaxY - pixelMinY) * 0.1;
      const safeMinX = pixelMinX + paddingX;
      const safeMaxX = pixelMaxX - paddingX;
      const safeMinY = pixelMinY + paddingY;
      const safeMaxY = pixelMaxY - paddingY;
      return [
        safeMinX + Math.random() * (safeMaxX - safeMinX),
        safeMinY + Math.random() * (safeMaxY - safeMinY),
      ];
    }
    if (action.target_coordinates) {
      const [xPct, yPct] = action.target_coordinates;
      return [xPct * elementBox.width, yPct * elementBox.height];
    }
    return null;
  }

  /**
   * Click one target on an ANIMATED board, at the moment its screen is up.
   *
   * The order is the point of it. The pointer is parked on the target FIRST,
   * then the gate opens, then the press happens in place — so the only thing
   * between "the right screen is showing" and the click is a mouse-down.
   *
   * It used to wait and then call `executeClick`, which begins with a humanised
   * move: 274ms p10 / 398ms p50 / 647ms max across a 340x384 widget, against a
   * 1500ms median dwell. A quarter to a third of the screen's life spent
   * travelling after we had already confirmed it was the right one — so the
   * gate could do its job and the click still land on the next screen.
   * `move()` short-circuits when it is already at the target, so parking early
   * costs nothing and removes all of it.
   */
  private async clickWhenFrameMatches(
    page: Page,
    element: ElementHandle,
    action: ClickAction,
    elementBox: { x: number, y: number, width: number, height: number },
    awaitKeyframe: string,
  ): Promise<void> {
    const rel = this.clickPointFor(action, elementBox);
    if (!rel) {
      console.warn('Click action received without coordinates or bounding box', action);
      return;
    }
    const at: [number, number] = [elementBox.x + rel[0], elementBox.y + rel[1]];
    await this.ph('mouse', () => this.human.move(page, at));
    // The gate watches the neighbourhood of the point we are about to press,
    // not the middle of the box — those differ on a wide target, and the one
    // that matters is where the finger is.
    await this.waitForKeyframe(
      element, awaitKeyframe,
      rel[0] / elementBox.width, rel[1] / elementBox.height,
    );
    await this.ph('mouse', () => this.human.click(page, at));
  }

  private async executeClick(
    page: Page,
    element: ElementHandle,
    action: ClickAction,
    elementBox: { x: number, y: number, width: number, height: number }
  ) {
    let relativeX: number;
    let relativeY: number;

    if (action.target_bounding_box) {
      // Pick random point in padding
      const [minX, minY, maxX, maxY] = action.target_bounding_box;

      const pixelMinX = minX * elementBox.width;
      const pixelMaxX = maxX * elementBox.width;
      const pixelMinY = minY * elementBox.height;
      const pixelMaxY = maxY * elementBox.height;

      // Apply padding (10%)
      const paddingX = (pixelMaxX - pixelMinX) * 0.1;
      const paddingY = (pixelMaxY - pixelMinY) * 0.1;

      const safeMinX = pixelMinX + paddingX;
      const safeMaxX = pixelMaxX - paddingX;
      const safeMinY = pixelMinY + paddingY;
      const safeMaxY = pixelMaxY - paddingY;

      // Random position
      relativeX = safeMinX + Math.random() * (safeMaxX - safeMinX);
      relativeY = safeMinY + Math.random() * (safeMaxY - safeMinY);
    } else if (action.target_coordinates) {
      // [x, y] percentages
      const [xPct, yPct] = action.target_coordinates;
      relativeX = xPct * elementBox.width;
      relativeY = yPct * elementBox.height;
    } else {
      console.warn('Click action received without coordinates or bounding box', action);
      return;
    }

    await this.ph('mouse', () => this.human.click(page, [elementBox.x + relativeX, elementBox.y + relativeY]));
  }

  private async executeDrag(
    page: Page,
    _element: ElementHandle,
    action: { source_bounding_box: [number, number, number, number]; target_bounding_box: [number, number, number, number] },
    elementBox: { x: number, y: number, width: number, height: number }
  ) {
    const bboxCenter = (bbox: [number, number, number, number]) => {
      const cx = elementBox.x + ((bbox[0] + bbox[2]) / 2) * elementBox.width;
      const cy = elementBox.y + ((bbox[1] + bbox[3]) / 2) * elementBox.height;
      return { x: cx, y: cy };
    };
    const src = bboxCenter(action.source_bounding_box);
    const dst = bboxCenter(action.target_bounding_box);
    await this.ph('mouse', () => this.human.drag(page, [src.x, src.y], [dst.x, dst.y]));
  }

  // ────────────────────────────────────────────────────────── typing + sliding
  // Mirrors _find_control / _execute_type / _execute_slide in page_solver.py.

  /**
   * First VISIBLE match for `selectors`, tried in order.
   *
   * `scope` is the challenge frame, or — for the vendors that render into the
   * host page rather than an iframe — the widget element itself. Never the
   * page: the generic tail of both selector tables would otherwise happily
   * match a login form's text box or a carousel's drag handle somewhere else on
   * the document, and the answer would go there.
   */
  private async findControl(
    scope: Frame | ElementHandle,
    selectors: ReadonlyArray<string>,
  ): Promise<ElementHandle | null> {
    for (const selector of selectors) {
      try {
        const el = await scope.$(selector);
        if (el && await el.isVisible()) return el;
      } catch {
        // A selector this adapter can't parse must not end the search.
      }
    }
    return null;
  }

  /**
   * Where a distorted-text captcha's answer goes.
   *
   * Inside the widget first. If the widget holds no text box at all, widen ONCE
   * to its enclosing <fieldset>/<form> and look again for a VENDOR-NAMED box
   * only.
   *
   * BotDetect is why, and nothing else needs it. BotDetect is a self-hosted
   * LIBRARY, so the host application owns the layout: measured on captcha.com
   * 2026-08-24, `.BDC_CaptchaDiv` — the element detectCaptcha returns — is
   * 280x50 and holds the image alone, while `#captchaCode` sits in a sibling
   * `<div class="validationDiv">` under the enclosing <fieldset>. Scoped to the
   * widget the solver found no box, textMode stayed false, and it read the code
   * correctly and never typed it. A clean solve thrown away, and
   * indistinguishable in any report from a model that cannot read warped text.
   *
   * The widened pass is restricted to TEXT_INPUT_VENDOR_SELECTORS on purpose.
   * The generic tail is what makes the in-widget lookup work for every other
   * vendor, and OUTSIDE the widget it is exactly how a captcha's answer ends up
   * in a login form's username box. One level, named selectors, never the page.
   *
   * Mirrors `_answer_box` in page_solver.py — CLAUDE.md 1c.
   */
  private async answerBox(
    scope: Frame | ElementHandle,
    element?: ElementHandle | null,
  ): Promise<ElementHandle | null> {
    const inside = await this.findControl(scope, TEXT_INPUT_SELECTORS);
    if (inside !== null || !element) return inside;
    // Nearest first: a <fieldset> inside a <form> is the tighter box, and a
    // union query hands back whichever comes first in document order — the
    // outer one.
    for (const axis of ['ancestor::fieldset[1]', 'ancestor::form[1]']) {
      let host: ElementHandle | null = null;
      try {
        host = await element.$(`xpath=${axis}`);
      } catch {
        continue;
      }
      if (!host) continue;
      const found = await this.findControl(host, TEXT_INPUT_VENDOR_SELECTORS);
      if (found) {
        console.log('Widget holds no answer box; using the vendor-named one in '
          + 'its enclosing form.');
        return found;
      }
    }
    return null;
  }

  /** Put the model's reading of a distorted-text captcha into its box. */
  private async executeType(
    page: Page,
    scope: Frame | ElementHandle,
    action: TypeAction,
    element?: ElementHandle | null,
  ): Promise<boolean> {
    const text = action.text ?? '';
    if (!text) return false;
    const field = await this.answerBox(scope, element);
    if (!field) {
      console.warn('Type action, but no text box in the widget; skipping.');
      return false;
    }

    // Tapping the box is what focuses it — and on a phone it is also what
    // raises the keyboard, so this is not decoration on either device.
    await this.moveAndClick(page, field);
    if (!(await this.human.typeText(page, field, text))) return false;
    console.log(`Typed ${text.length} character(s) into the captcha field.`);
    return true;
  }

  /** `captchakraken track-piece` — box of what moved, handle masked out. */
  private async trackPiece(
    element: ElementHandle,
    beforePath: string,
    afterPath: string,
    exclude: [number, number, number, number],
  ): Promise<[number, number, number, number] | null> {
    try {
      await element.screenshot({
        path: afterPath,
        timeout: this.config.elementScreenshotTimeoutMs ?? 8000,
        animations: 'disabled',
      });
      const res = await this.runCvTool(
        'track-piece',
        { before: beforePath, after: afterPath, exclude },
        ['track-piece', beforePath, afterPath, JSON.stringify(exclude)],
      );
      return res && res.bbox ? res.bbox : null;
    } catch (e) {
      console.warn('track-piece failed:', e);
      return null;
    }
  }

  /**
   * Drive a puzzle-piece slider until the PIECE reaches the model's slot.
   *
   * The model is asked for one thing here — the centre of the gap — because it
   * is the only thing the picture can tell it. What it cannot know is how far
   * the handle must travel to put the piece there: the handle is elsewhere on
   * the widget, and the ratio between the two is a vendor implementation detail
   * that several of them deliberately vary.
   *
   * So this is closed-loop, not a calculation. Press the handle, nudge it twice
   * by known amounts, and watch the screen: union(before, after) spans the
   * piece's ORIGINAL left edge to its CURRENT right edge, so its width is
   * pieceWidth + ratio x nudge. Two nudges, two widths, two unknowns — solve for
   * both, then steer the remaining distance and re-measure. The mouse is not
   * released until the piece is home, because on every one of these puzzles
   * releasing IS the submit; there is no Verify button to reconsider at.
   *
   * Returns false if there is nothing here to drag, leaving the caller's normal
   * no-op handling to deal with it.
   */
  /**
   * Device pixels per CSS pixel, read off the shot we are about to measure.
   * 1 when the image cannot be read — an unreadable shot is already the
   * trackPiece-returns-null path, and guessing a ratio would steer by it.
   */
  private shotScale(shot: string, cssWidth: number): number {
    const dims = this.readPngDimensions(shot);
    if (!dims || cssWidth <= 0) return 1;
    return dims.width / cssWidth;
  }

  private async executeSlide(
    page: Page,
    element: ElementHandle,
    scope: Frame | ElementHandle,
    action: DragAction,
    elementBox: { x: number, y: number, width: number, height: number },
  ): Promise<boolean> {
    const targetX = ((action.target_bounding_box[0] + action.target_bounding_box[2]) / 2)
      * elementBox.width;

    const handle = await this.findControl(scope, SLIDER_HANDLE_SELECTORS);
    if (!handle) {
      // No track — the sliderless members of the family (Lemin's "cropped")
      // want the piece dragged directly. Same answer from the model, because
      // the two look identical; different gesture. Nothing to close a loop on,
      // since the piece is under the cursor and moves with it one for one.
      const piece = await this.findControl(scope, DRAGGABLE_PIECE_SELECTORS);
      const box = piece ? await piece.boundingBox() : null;
      if (!box) {
        console.warn('Slide action, but the widget has neither a slider nor a draggable piece.');
        return false;
      }
      // BOTH axes. The rail members travel horizontally and nothing else, so
      // the handle's own y is the only y there is — but a free drag carries the
      // piece across the card, and holding the piece's row here slid it along
      // the TRAY and released it there, well below the slot, every time.
      const targetY = ((action.target_bounding_box[1] + action.target_bounding_box[3]) / 2)
        * elementBox.height;
      console.log('No slider track; dragging the piece to the slot directly.');
      await this.human.drag(
        page,
        [box.x + box.width / 2, box.y + box.height / 2],
        [elementBox.x + targetX, elementBox.y + targetY],
      );
      return true;
    }

    const hbox = await handle.boundingBox();
    if (!hbox) return false;
    const startX = hbox.x + hbox.width / 2;
    const holdY = hbox.y + hbox.height / 2;

    // Mask the whole horizontal BAND the handle runs in, not just where it is
    // now: it is about to move across that band, and most vendors fill the
    // track behind it as it goes. Either would otherwise be the largest moving
    // thing in frame, and we would track the handle instead of the piece.
    const pad = Math.max(4, hbox.height * 0.35);
    const band: [number, number, number, number] = [
      0,
      hbox.y - elementBox.y - pad,
      elementBox.width,
      hbox.y + hbox.height - elementBox.y + pad,
    ];

    const shots = Array.from({ length: 4 }, (_, i) =>
      path.join(os.tmpdir(), `slide_${Date.now()}_${i}_${Math.floor(Math.random() * 1e9)}.png`));
    try {
      await this.move(page, handle, { paddingPercentage: 30 });
      await this.human.press(page);
      await this.human.pause('grab');
      await element.screenshot({
        path: shots[0],
        timeout: this.config.elementScreenshotTimeoutMs ?? 8000,
        animations: 'disabled',
      });

      // CSS pixels out here, DEVICE pixels inside the shots. The same number on
      // a 1x desktop and 2.625x apart on a phone, which is why this loop
      // measured a slider correctly for a year and then missed every attempt
      // the moment Tier 3 grew a mobile arm: the mask landed above the handle,
      // the widths came back 2.6x too wide, and solveSlideGeometry threw them
      // out as wider than the widget. Measured from the shot rather than asked
      // of the page, for the same reason the grid path does it (`scaleX` in
      // driveRecaptchaGrid): what the CV reads is the image, whatever the
      // window thinks its ratio is.
      const scale = this.shotScale(shots[0], elementBox.width);
      const exclude = band.map((v) => v * scale) as [number, number, number, number];

      const widths: Array<[number, number]> = [];
      let lastBox: [number, number, number, number] | null = null;
      for (let i = 0; i < SLIDE_PROBE_OFFSETS_PX.length; i++) {
        const offset = SLIDE_PROBE_OFFSETS_PX[i];
        await this.performSmoothMove(page, startX + offset, holdY);
        await this.human.pause('probe');
        const box = await this.trackPiece(element, shots[0], shots[i + 1], exclude);
        if (box) {
          widths.push([offset, (box[2] - box[0]) / scale]);
          lastBox = box;
        }
      }

      const { pieceWidth, ratio } = solveSlideGeometry(widths, elementBox.width);
      if (!lastBox || pieceWidth === null) {
        // Never saw the piece — a canvas the screenshot cannot separate, a
        // widget that redraws wholesale, or a press the handle refused. Fall
        // back on the geometry every one of these puzzles shares: piece and
        // handle both start flush left, so the handle's travel is the piece's.
        console.warn('Slider: piece never resolved on screen; steering by handle travel alone.');
        await this.performSmoothMove(page, startX + (targetX - (startX - elementBox.x)), holdY);
      } else {
        // The offset lastBox was MEASURED at — not the final probe, and not
        // indexed by how many measurements succeeded. If the first probe failed
        // to resolve and the second worked, those two disagree, and steering
        // from a base the reading does not belong to sends the piece somewhere
        // neither the model nor the screen asked for.
        let offset = widths[widths.length - 1][0];
        for (let i = 0; i < SLIDE_MAX_CORRECTIONS; i++) {
          const pieceCentre = lastBox[2] / scale - pieceWidth / 2;
          const error = targetX - pieceCentre;
          if (Math.abs(error) <= SLIDE_TOLERANCE_PX) break;
          offset += error / ratio;
          await this.performSmoothMove(page, startX + offset, holdY);
          await this.human.pause('probe');
          const box = await this.trackPiece(element, shots[0], shots[3], exclude);
          if (!box) break;  // ran out of track; release where we are
          lastBox = box;
        }
      }

      // Settle before letting go. A release in the same tick as the last move
      // reads as a machine, and some vendors sample the final milliseconds of
      // the gesture.
      await this.human.pause('settle');
    } finally {
      try { await this.human.release(page); } catch { /* the page may have navigated */ }
      for (const shot of shots) {
        try { if (fs.existsSync(shot)) fs.unlinkSync(shot); } catch { /* best-effort */ }
      }
    }
    return true;
  }
}
