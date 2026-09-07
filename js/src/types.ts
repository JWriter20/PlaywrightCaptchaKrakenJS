import type { Humanizer, HumanizationMode, TouchTransform } from './humanize.js';

/**
 * Lifecycle event emitted by {@link CaptchaKrakenConfig.onStep}. One is fired
 * before any interaction (`stage: 'initial'`) and one after every executed
 * action (click batch, drag, wait, submit) plus per dynamic-grid round, so a
 * caller can record the exact sequence of intermediate stages, time them, and
 * count steps without scraping CAPTCHA_DEBUG dumps.
 */
export interface SolveStepEvent {
  /** 1-based monotonically increasing step index across the whole solve. */
  index: number;
  /**
   * Coarse stage kind:
   *  - `initial`  : screenshot taken before any action (baseline)
   *  - `click`    : after a click (or batch of clicks) was executed
   *  - `drag`     : after a drag was executed (a slider counts as one)
   *  - `type`     : after a distorted-text answer was typed into its box
   *  - `wait`     : after a CLI-requested wait elapsed
   *  - `submit`   : after the Verify/Next/Submit button was clicked
   *  - `round`    : a dynamic reCAPTCHA 3x3 round boundary (pre-solve snapshot)
   */
  stage: 'initial' | 'click' | 'drag' | 'type' | 'wait' | 'submit' | 'round';
  /** Short human label, e.g. "round-2:clicked 3 tile(s)". */
  label: string;
  /**
   * Absolute path to a PNG screenshot of the captcha element at this step.
   * The file is owned by the callback once emitted — the solver does NOT
   * delete it (copy/move it where you need it). Null if the screenshot failed.
   */
  screenshotPath: string | null;
  /** Detected puzzle vendor, if known. */
  puzzleSource?: 'hcaptcha' | 'recaptcha' | 'unknown';
  /**
   * Which captcha frame this screenshot is of:
   *  - `checkbox`  : the anchor "I'm not a robot" widget (no puzzle yet)
   *  - `challenge` : the open image/grid challenge frame (the real puzzle)
   *  - `unknown`   : could not be determined from the frame src
   * Recorders that only want the actual solve (not the pre-challenge checkbox
   * clicks) filter to `challenge`.
   */
  frameRole?: 'checkbox' | 'challenge' | 'unknown';
  /** Outer solve-loop attempt this step belongs to. */
  attempt: number;
  /** ms since solve() started. */
  elapsedMs: number;
  /** Free-form per-stage detail (action payload, clicked cells, etc.). */
  meta?: Record<string, any>;
}

export interface CaptchaKrakenConfig {
  /**
   * How the driver MOVES — a choice of input DEVICE, not a realism dial.
   *
   *  - `'mouse'`  (default) Bezier arcs, Fitts's-law durations, overshoot.
   *  - `'mobile'` touch events with finger kinematics. On a touch-only widget
   *               this is the difference between the page's handlers firing and
   *               not; a mousemove there is the wrong event, not a weaker one.
   *               Needs a Chromium-family page launched with `hasTouch: true`,
   *               or an Appium/WebdriverIO driver in {@link touchDriver}.
   *  - `'none'`   the shortest legal path to the same DOM effect. Much faster,
   *               and detectable by anything scoring pointer telemetry — for
   *               fixtures, self-hosted targets, and stacks that humanise
   *               elsewhere.
   *
   * Unset falls back to `CAPTCHA_HUMANIZATION`, then `'mouse'`. Set
   * {@link humanizer} instead to supply your own implementation.
   *
   * @example
   * ```typescript
   * const context = await browser.newContext({ ...devices['Pixel 7'], hasTouch: true });
   * const solver = new CaptchaKrakenSolver({ humanization: 'mobile' });
   * await solver.solve(await context.newPage());
   * ```
   */
  humanization?: HumanizationMode;

  /**
   * Your own {@link Humanizer}. Overrides {@link humanization} entirely — the
   * driver then makes no decisions about pointer motion at all.
   *
   * Use it when you already model your users' input (a hardware pointer, a
   * device farm, a recorded trace), or when the browser humanises for you:
   * camoufox's `humanize` juggler re-humanises every `mouse.move()` it is
   * handed, and composing both measured 82.1s against 13.4s on one solve.
   *
   * @example
   * ```typescript
   * import { NullHumanizer, type Humanizer } from 'captchakraken';
   *
   * class MyPointer extends NullHumanizer implements Humanizer {
   *   async move(page, to) { await myHardwareMouse.glideTo(to[0], to[1]); this.at = to; }
   * }
   * new CaptchaKrakenSolver({ humanizer: new MyPointer() });
   * ```
   */
  humanizer?: Humanizer;

  /**
   * `humanization: 'mobile'` only. The thing that is actually TOUCHED, when it
   * is not the page object — an Appium / WebdriverIO / Selenium driver on a real
   * handset. Left unset, the mode dispatches CDP touch events at the page it
   * was given, which is what browser mobile emulation wants.
   */
  touchDriver?: any;

  /**
   * `humanization: 'mobile'` only. CSS-pixel → device-pixel transform for
   * {@link touchDriver}. `scale` is usually `window.devicePixelRatio`; `origin`
   * is the top-left of the webview in screen coordinates. The default identity
   * is right for emulation and for callers who map coordinates themselves.
   */
  touchTransform?: TouchTransform;

  /**
   * Optional observer fired at each intermediate solve stage. Receives a
   * baseline screenshot before any action and one after every executed action.
   * Use it to capture intermediate-stage screenshots, count steps, and time
   * each phase. The callback may be async; the solver awaits it. Errors thrown
   * by the callback are swallowed (never fail a solve because logging failed).
   *
   * The PNG at `event.screenshotPath` is owned by the callback once emitted —
   * the solver will not delete it.
   */
  onStep?: (event: SolveStepEvent) => void | Promise<void>;

  /**
   * Path to the bundled CaptchaKraken CLI root.
   *
   * Usually you do NOT need to set this. If omitted, the solver will auto-resolve the
   * `python/` directory (the captchakraken package) shipped inside this npm package.
   */
  repoPath?: string;
  /**
   * Command to run python (default: 'python' or 'python3').
   */
  pythonCommand?: string;
  /**
   * vLLM LoRA name to invoke.
   *
   * Defaults to whatever `models.json` calls `latest` (via CAPTCHA_LORA_NAME if
   * set) — NOT a literal. The name selects the PROMPT GENERATION as well as the
   * weights, so a constant here would pin the prompts to one generation while
   * `latest` moved on. Override only if you've registered a different module
   * with the vLLM server; see model-name.ts.
   */
  model?: string;
  /**
   * Force ONE expert of a ROUTED model — 'pixel' | 'grid' | 'video' | 'text'.
   *
   * A routed model (Abyss) is four LoRA adapters behind one endpoint, and the
   * router is the prompt family each request is about to send — so leaving this
   * unset already reaches the right expert. This is the override: serve one
   * arm, drive only the puzzles it owns, which is what a per-arm benchmark
   * needs and what a licence holder pinning a single expert wants.
   *
   * Refused against a model that serves a single adapter — every model
   * published so far — rather than ignored, because a run that quietly measured
   * the generalist while reporting an expert is a number nobody can catch.
   *
   * Mirrors the Python `PageSolverConfig.expert`; also settable as
   * CAPTCHA_EXPERT, which this forwards to the CLI ahead of.
   */
  expert?: string;
  /**
   * Bearer token for the vLLM server (also picked up from VLLM_API_KEY env).
   */
  apiKey?: string;

  /**
   * Starting mouse position (default: { x: 100, y: 100 }).
   * HIGHLY RECOMMENDED to set this, prevents jumping around of the cursor when solving.
   */
  startingMousePosition?: { x: number, y: number };


  /**
   * Automatically re-check for newly opened / next-step captchas after each solve
   * attempt (e.g., clicking a checkbox opens an image challenge).
   *
   * The loop count that FITS inside overallSolveTimeoutMs, rather than one that
   * needs policing by the clock: a round costs ~4-7s, so six rounds is the 45s
   * budget. A backstop, not the normal way a hopeless solve ends — that is
   * maxNoProgressRounds. Reaching this number is a bug report.
   *
   * Default: 6
   */
  maxSolveLoops?: number;

  /**
   * Consecutive rounds producing the SAME answer before the solve is abandoned.
   *
   * At temperature 0 the model is a function of the picture, so an identical
   * answer means an identical picture — and every answer this driver produces is
   * EXECUTED, so the previous one already ran and moved nothing. Repeating it
   * cannot do better; it just spends a round.
   *
   * 2 rather than 1 because the first repeat also escalates to a RECORDING (see
   * shouldRetryAsAnimated), which is the one recovery worth trying: a cycling
   * board reads as a still and answers the same way every round. One more round
   * buys that chance.
   *
   * Default: 2
   */
  maxNoProgressRounds?: number;

  /**
   * Delay (ms) after executing actions before re-detecting captchas.
   * Useful to allow challenge frames / new images to appear.
   *
   * Default: 1200
   */
  postSolveDelayMs?: number;

  /**
   * Overall time limit (ms) for the entire solve loop.
   *
   * Default: 45000
   */
  overallSolveTimeoutMs?: number;

  /**
   * Poll interval (ms) for the reCAPTCHA grid-cell-load wait — how often the
   * solver re-screenshots the grid while waiting for tiles to stop fading in.
   * Doubles as the inter-frame gap for the settle change-detector, so keep it
   * comfortably above zero.
   *
   * Default: 250
   */
  gridLoadPollIntervalMs?: number;

  /**
   * Overall timeout (ms) for the reCAPTCHA grid-cell-load wait. On timeout the
   * solver proceeds to screenshot anyway (best-effort).
   *
   * Default: 8000
   */
  gridLoadTimeoutMs?: number;

  /**
   * reCAPTCHA 3x3 dynamic puzzles only. After clicking a round of tiles, the
   * max time (ms) to wait for at least one clicked blank/fading tile to finish
   * loading before re-screenshotting and re-solving. On timeout the solver
   * proceeds anyway (best-effort, backstopped by overallSolveTimeoutMs).
   *
   * Default: 6000
   */
  recaptchaDynamicFadeWaitMs?: number;

  /**
   * reCAPTCHA 3x3 dynamic puzzles only. Minimum gap (ms) between the two frames
   * the fade detectors diff, and the poll cadence for waitForAnyClickedTileLoaded
   * / currentLoadingCells. Kept comfortably above zero so two consecutive frames
   * during a slow fade differ enough for the change detector to fire (frames
   * captured back-to-back can look identical mid-fade and read as "loaded").
   *
   * Default: 250
   */
  recaptchaDynamicFadePollMs?: number;

  /**
   * reCAPTCHA 3x3 dynamic puzzles only. Grace window (ms) after clicking during
   * which the solver watches the clicked tiles for the ONSET of a blank/fade
   * before deciding the puzzle is solved. reCAPTCHA keeps a clicked tile
   * SELECTED (showing its old image + a blue badge) for a couple of seconds and
   * only THEN blanks it to swap in a replacement, so the window must comfortably
   * exceed that delay or we submit while the refresh is still pending. If no
   * clicked tile goes blank/changing within this window, the puzzle is treated
   * as solved.
   *
   * Default: 4000
   */
  recaptchaFadeOnsetGraceMs?: number;

  /**
   * reCAPTCHA 3x3 dynamic puzzles only. Cap on the number of
   * click → refresh → re-solve rounds within a single puzzle, independent of
   * maxSolveLoops.
   *
   * Default: 8
   */
  recaptchaMaxDynamicRounds?: number;

  /**
   * reCAPTCHA 3x3 dynamic puzzles only. When true, the solver hovers the mouse
   * over the just-clicked blank/fading tiles (in click order) while waiting for
   * them to reload, mimicking a human. Disable to skip the hover behavior.
   *
   * Default: true
   */
  recaptchaTileHoverEnabled?: boolean;

  /**
   * While the model is generating a solution (the main idle window), drift the
   * cursor over the challenge area with human-like trajectories instead of
   * leaving it frozen. Cancelled the instant the model responds. Set false to
   * keep the cursor still during inference.
   *
   * Default: true
   */
  idleMouseWander?: boolean;

  /**
   * After an action, the max time (ms) to watch for the solve outcome — the
   * vendor's solved signal (checkbox checked / response token) or a freshly
   * rendered next round — before falling back to a full re-detect. Returning as
   * soon as the solved signal appears avoids re-entering the solve pipeline on a
   * challenge frame that is merely animating closed.
   *
   * That is the window's whole job, and it is why a WRONG answer spends all of
   * it: the vendors emit nothing on a wrong answer, they just re-deal. So
   * sizing it means asking how late a real SUCCESS can arrive — measured over
   * 34 successful rounds across 20 puzzle types on the fixture server: p50
   * 360ms, max 528ms. 1000ms is that worst case plus headroom for a vendor
   * round-trip nobody has measured yet. Down from 2500, which was spent in full
   * on every unsuccessful round.
   *
   * Default: 1000
   */
  postSolveOutcomeTimeoutMs?: number;

  /**
   * Poll interval (ms) inside that window.
   *
   * The success signals are polled, so detection latency is a MULTIPLE of this.
   * Each poll is a couple of cheap DOM reads.
   *
   * Default: 75
   */
  postSolveOutcomePollMs?: number;

  /**
   * Guard against the captcha frame changing WHILE the model is generating.
   * reCAPTCHA/hCaptcha fade fresh tiles in over ~1s; if new imagery paints
   * between the screenshot we send the model and the moment its answer returns,
   * that answer describes a stale ("undeveloped") frame and its tile picks /
   * bboxes no longer line up with what's on screen. When enabled, the solver
   * re-screenshots the frame after inference, and if it changed it discards the
   * stale answer and re-solves on the fresh frame (see maxStaleFrameReSolves).
   *
   * Default: true
   */
  staleFrameReSolveEnabled?: boolean;

  /**
   * Fraction of pixels (0–1) that must differ between the frame sent to the
   * model and a screenshot taken right after inference for the frame to count
   * as "changed during inference" (a tile faded in / refreshed). Below this,
   * the frame is treated as unchanged and the answer is used as-is. Reuses the
   * same frame-diff primitive as the reCAPTCHA settle detector.
   *
   * Default: 0.02
   */
  staleFrameDiffThreshold?: number;

  /**
   * Max number of times to re-screenshot + re-solve when the frame keeps
   * changing during inference, before giving up and acting on the latest answer
   * (better to act than to spin). Set 0 to detect-and-log without re-solving.
   *
   * Default: 2
   */
  maxStaleFrameReSolves?: number;

  // ── Settle monitor (challenge-state gating) ───────────────────────────────
  // The solver tracks the challenge's lifecycle state and refuses to send a
  // mid-transition / still-loading frame to the model. These tune the pixel
  // "has it settled yet?" monitor that gates that decision (it reuses the same
  // check-movement frame-diff as the stale-frame guard).

  /**
   * Fraction of pixels (0–1) that must differ between two consecutive challenge
   * frames for it to still count as "moving" (loading/animating) rather than
   * settled.
   *
   * Default: 0.01
   */
  settleDiffThreshold?: number;

  /**
   * How long to wait for hCaptcha's task images to paint before screenshotting
   * anyway. Best-effort by design — the screenshot happens either way — so it
   * is bounded by what a loading tile plausibly costs. Default: 3000
   */
  hcaptchaImagesTimeoutMs?: number;

  /** Poll interval (ms) for the settle monitor. Default: 220 */
  settlePollMs?: number;

  /**
   * Consecutive still frame-pairs required before the challenge is declared
   * settled and safe to screenshot for the model.
   *
   * Default: 2
   */
  settleFrames?: number;

  /** Overall timeout (ms) for the settle monitor before it gives up. Default: 9000 */
  settleTimeoutMs?: number;

  /**
   * If the challenge keeps changing continuously for at least this long without
   * ever settling, it's treated as an **animated / video** challenge (surfaced
   * distinctly, not as "unsupported"). Static grids settle in ~1–2s; a video
   * never does.
   *
   * Default: 4500
   */
  animatedChallengeAfterMs?: number;

  // ── Animated challenges ───────────────────────────────────────────────────
  // A challenge that never settles is animated BY DESIGN (hCaptcha fades sprites
  // on independent cycles; GeeTest's svg board cycles its glyphs). Those are
  // RECORDED and solved from keyframes rather than abandoned.

  /**
   * Record and solve animated challenges. When false, a challenge that never
   * settles fails with `.animated = true` as it did before — for callers who
   * would rather fail fast than spend the recording time.
   *
   * Default: true
   */
  videoSolveEnabled?: boolean;

  /**
   * Length (ms) and frame rate of the burst recorded from an animated challenge.
   *
   * Do not tune these casually. They are deliberately identical to the geometry
   * the training corpus was collected at (4000ms @ 10fps), and the keyframe
   * slicer reads the clip's temporal structure — a different length or rate
   * changes where a cycle's period lands across the frames, so the live set gets
   * sliced differently from the trained set and the model's frame number stops
   * meaning what the driver thinks it means.
   *
   * Defaults: 4000 / 10
   */
  videoBurstDurationMs?: number;
  videoBurstFps?: number;

  /**
   * How long (ms) to wait for the widget to return to the keyframe the model
   * chose, before clicking anyway; and the poll interval while waiting.
   *
   * Bounded because the alternative is worse: these puzzles cycle, so the state
   * does come back — but if the recording caught a one-off transition it never
   * will, and a click on the model's coordinates is a better use of the
   * remaining budget than a timeout.
   *
   * Defaults: 6000 / 120
   */
  keyframeWaitTimeoutMs?: number;
  keyframeWaitPollMs?: number;
  /**
   * Hard ceiling on a burst that never repeats a screen. The burst normally
   * ends when the cycle closes; this bounds a continuous animation, which never
   * closes one. Default 12000ms.
   */
  videoBurstMaxMs?: number;
  /**
   * Record while the still screenshot is being read, so a cycling board is
   * known before its answer is acted on. Default on. Off restores the older
   * shape, which learned the same thing two rounds later.
   */
  speculativeBurstEnabled?: boolean;

  /**
   * Extra wall clock (ms) granted ONCE, the first time a solve escalates to a
   * recording. NOT a looser `overallSolveTimeoutMs`.
   *
   * That budget counts ROUNDS — `maxSolveLoops` x ~7000ms, which
   * no-progress.test.ts pins. A recording is not a round: it is a fixed extra
   * stage costing the burst, the slice, one MULTI-IMAGE inference (six
   * keyframes, several times a still's) and the wait for the widget to come
   * back round to the chosen frame. Nothing in the 45s was set aside for it, so
   * an escalation late in a solve simply ran the clock out and reported a
   * timeout — which reads as a slow model rather than as a budget with no room
   * for what the solver had just decided to do.
   *
   * Measured 2026-08-22, Tier 3 run 32596340560: hcaptcha_click_image_by_traits,
   * hcaptcha_connect_path and hcaptcha_grid_3x3_property each failed this port
   * with "Captcha solve timed out after 45000ms (attempt 6/6)" at 49-59s, and
   * solve in 14-20s on the rounds the still path answers them.
   *
   * The total granted is `videoBurstDurationMs + keyframeWaitTimeoutMs + this`,
   * derived so a longer burst carries its own budget — page_solver.py's
   * `video_budget_ms` is the same arithmetic, since the two ports must not
   * disagree about how long a video solve may take.
   *
   * Granted only when `videoSolveEnabled`: a caller who wants a hard deadline
   * turns recording off, which already means "fail fast rather than spend the
   * recording time".
   *
   * Default: 8000
   */
  videoExtraInferenceMs?: number;

  /**
   * After clicking Submit/Verify, the solver EXPECTS the frame to change (advance
   * to the next round, or close because it was accepted). This is how long (ms)
   * to wait for that transition to begin before re-evaluating — so the shift
   * itself is never screenshotted and mis-read as a fresh (blank) puzzle.
   *
   * Default: 4000
   */
  postSubmitChangeTimeoutMs?: number;

  /**
   * Per-call timeout (ms) for element screenshots of the challenge iframe.
   * Playwright's default is 30000, which means a stale/transitioning handle
   * (e.g. hCaptcha swapped in the next round while we held the old iframe)
   * hangs the whole solve for 30s before failing. Bounding it lets a stale
   * handle fail fast so the solve loop can re-detect the fresh challenge.
   *
   * Default: 8000
   */
  elementScreenshotTimeoutMs?: number;

  /**
   * How long an `onStep` observer's screenshot may take, ms (default 2000).
   *
   * Deliberately separate from `elementScreenshotTimeoutMs`, which is sized for
   * the picture the MODEL reads. This one is a trace snapshot: it is taken with
   * `animations: 'disabled'`, which makes Playwright wait for the element to
   * stop moving first, and on a still-animating widget that ran to the full 8s
   * on every step — 8.0s of a measured 12.0s solve. An observer must never cost
   * more than the action it observes, and a missed frame in a trace costs
   * nothing. Ignored entirely when no `onStep` is set.
   */
  stepScreenshotTimeoutMs?: number;

  /**
   * After a submit, hCaptcha may replace the challenge iframe for the next
   * round, detaching the handle we just detected ("element is not visible").
   * This is a transition, not a dead puzzle: how many times to back off,
   * re-detect the fresh challenge, and retry before giving up.
   *
   * Default: 3
   */
  maxStaleElementRetries?: number;

  /**
   * Backoff (ms) before re-detecting after a stale-element screenshot failure,
   * giving the round transition time to finish.
   *
   * Default: 900
   */
  staleElementBackoffMs?: number;

  /**
   * When the model reports "unsupported" *after we've already interacted* (i.e.
   * mid multi-round), it's almost always a not-yet-settled next round rather
   * than a genuinely unsupported puzzle. This is how many times to wait for the
   * challenge to settle and re-solve before giving up. (A single retry loses a
   * race when the next round loads slowly.)
   *
   * Default: 3
   */
  maxUnsupportedReSolves?: number;
}

export interface BoundingBox {
  0: number; // min_x
  1: number; // min_y
  2: number; // max_x
  3: number; // max_y
}

/**
 * Fields present ONLY on actions from an animated challenge.
 *
 * Why an action carries a picture: on an animated puzzle the target is visible
 * only part of the time, so the coordinates are correct only while the widget
 * looks the way it did in that keyframe. The driver holds the mouse until the
 * live neighbourhood around the click point matches the same neighbourhood of
 * `await_keyframe`, then clicks. Without the wait, a click on a fading sprite
 * lands on background.
 *
 * Both are set together or not at all — a number with no image cannot be waited
 * on, and an image with no number cannot be reported. Absent on every still
 * puzzle, where there is no moment to wait for.
 */
export interface AnimatedActionFields {
  /** Absolute path to the keyframe the model chose to act on. */
  await_keyframe?: string | null;
  /** Its 1-based number in the keyframe set that was sent. */
  frame?: number | null;
}

export interface ClickAction extends AnimatedActionFields {
  action: 'click';
  /**
   * One or more normalized [x1, y1, x2, y2] bboxes (0–1 fractions of the
   * screenshot). Each entry produces one click. Emitted by v2 CLI for both
   * grid selections and click-puzzle points.
   */
  target_bounding_boxes?: Array<[number, number, number, number]>;
  /** Legacy v1 fields kept for backwards-compat with older CLI builds. */
  target_number?: number | null;
  target_bounding_box?: [number, number, number, number] | null;
  target_coordinates?: [number, number] | null;
}

export interface DragAction extends AnimatedActionFields {
  action: 'drag';
  /**
   * Null on a PUZZLE-PIECE SLIDER. Not a missing field — the shape of the
   * answer: what you grab (a handle, elsewhere on the widget) is not what has
   * to arrive (the piece), and how far one moves the other is a vendor detail
   * no picture reveals. The driver reads a null source as "find the slider and
   * close the loop on the piece" (see executeSlide).
   */
  source_bounding_box: [number, number, number, number] | null;
  target_bounding_box: [number, number, number, number];
}

/** Distorted-text captchas: the answer is a string, not a place. */
export interface TypeAction {
  action: 'type';
  text: string;
}

export interface DoneAction {
  action: 'done';
}

export interface WaitAction {
  action: 'wait';
  duration_ms: number;
}

export type CaptchaAction = ClickAction | WaitAction | DragAction | TypeAction | DoneAction;

export type SolverResult = CaptchaAction | CaptchaAction[];

export interface TokenUsage {
  input_tokens: number;
  output_tokens: number;
  cached_input_tokens?: number;
  model: string;
}

export interface CliResponse {
  actions: SolverResult;
  token_usage: TokenUsage[];
}

export interface Vector {
  x: number;
  y: number;
}

export interface SolveResult {
  isSolved: boolean;
  finalMousePosition: Vector;
  tokenUsage: {
    modelName: string;
    inputTokens: number;
    outputTokens: number;
    cachedInputTokens: number;
    estimatedCost: number;
  };
  /**
   * Where this solve's wall-clock went, in milliseconds per phase — the same
   * partition `CAPTCHA_TIMINGS=1` prints, handed to the caller instead of to
   * stderr. Mirrors `SolveResult.phases` on the Python port.
   *
   * Returned rather than only logged because "the solve took 12s" is not
   * actionable and "the settle monitor spent 4s of it" is, and a caller
   * measuring a fleet cannot scrape another process's stderr.
   */
  phases?: Record<string, number>;
}
