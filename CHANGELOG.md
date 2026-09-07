# Changelog

All notable changes to CaptchaKraken are documented here. This project follows
semantic versioning; v2 is a major, **breaking** release.

## [2.10.0] - 2026-09-07

The hosted API could tell you a solve failed. It could not tell you WHERE.

### Added

- **`X-CK-Site` — the hostname of the page being solved.** Sent per solve by
  both drivers, from `page.url`, and read only by the hosted gateway. It is what
  makes a solve rate actionable: one vendor rolling out a new variant shows up
  as one host's rate falling, where the aggregate does not move at all.

  **The hostname and nothing else.** Not the path, not the query, not a
  fragment, no `user:pass@` prefix, no port. A captcha lives on a login or a
  checkout page, so the URL is PARSED and the host taken out of it rather than a
  URL being sent and trimmed later — `https://shop.example.com/account/reset?
  token=9f3c1a` is reported as `shop.example.com`. Pinned in both ports by
  `test_only_the_hostname_leaves_the_machine.py` and
  `site-is-only-the-hostname.test.ts`.

  A page with no host — `about:blank`, a `file://` fixture, a page that closed
  mid-solve — sends no header rather than a guess, and nothing here can raise: a
  solve must not fail over telemetry.

- **`CAPTCHA_REPORT_SITE=0` turns it off**, on its own. Separate from
  `CAPTCHA_REPORT_OUTCOME` because they are separate disclosures: the outcome is
  a fact about our model and the site is a fact about your business, and one
  switch for both would price that choice at "tell us nothing". Turning off
  either leaves the other working. Self-hosted users send neither, unchanged.

Nothing was removed or renamed; `contract.json` grows two env-var entries.

## [2.8.0] - 2026-09-06

A model can now be a MIXTURE rather than a single adapter, and a model can now
say its weights are not yours to download. Both are facts about the model that
used to live in the caller's head.

### Added

- **Expert routing — `experts` in `models.json`.** A routed model is several
  LoRAs behind one endpoint, and the thing that picks between them is the
  **prompt family the request is about to send** (`grid`, `pixel`, `video`,
  `text`). That router already existed, so a request names its own expert with
  no DOM access, no image classifier and no cooperation from the caller. The
  family is a per-REQUEST argument, never solver state: a real solve changes
  family mid-solve, e.g. a board that reads as a grid and then a click round
  after the grid is rejected.

  **ABSENT MEANS NOT ROUTED, and that is the whole compatibility story.** Every
  model published to date declares no `experts`, so the client sends the single
  `lora_name` on every request exactly as it did before this field existed, and
  the bytes on the wire are identical.

- **Pin one arm — `expert=` on `CaptchaSolver` / `PageSolverConfig.expert` /
  `CaptchaKrakenConfig.expert`, `--expert`, `CAPTCHA_EXPERT`.** What a per-arm
  benchmark wants. Pinning against a single-adapter model **raises at
  construction** rather than being ignored: a run that quietly measured the
  generalist and reported it as the expert is a number nobody can catch.

- **`availability: "private"` — a third value, because the weights now exist.**
  `public` / `private` / `licensed` answer one question: can these weights be
  obtained, and by whom. `private` means the bytes are on the Hub and a token
  this account authorised opens them; `licensed` means there is no repo at all.
  So `fetch` refuses `licensed` and does **not** refuse `private` — an early
  refusal is a kindness for a model you could never download and a lie for one
  you can. `plan()` reports `needs_auth`, so `--dry-run` says "this needs a
  token" before the 401 rather than after it. An unrecognised value reads as
  `licensed`: both halves fail closed.

### Fixed

- **Grid detection estimates the pitch over gaps that could be a cell**, and
  refuses a shape no vendor ships. The previous pass needed every interior
  divider to be visible, so a board whose middle cells happened to be flat was
  read as no grid at all — the solver then saw nothing to click. Measured over
  the reCAPTCHA 3x3 corpus, this also lowers phantom selections on fresh boards
  (15 -> 12 over 1062 tiles); each phantom was a tile the solver would refuse
  to click.
- **Keyframe stills caught between two boards are dropped.** A frame sampled
  during the vendor's transition holds half of each board and is answerable as
  neither.

### Changed

- **The hosted `captcha` alias means the model we actually serve.** It had gone
  on naming the v1.1 weights that share the name as a local vLLM alias, so a
  hosted caller resolved prompts for a model the endpoint had stopped serving.
- **An unmapped or unrecognised prompt family degrades to the generalist,
  never a 400.** A caller prompting in their own words is the expected case for
  a shipped model; refusing one cost every distorted-text solve in production
  for a day.

## [2.7.0] - 2026-08-29

Humanisation stops being something the driver *is* and becomes something it
*has*. It used to be wired straight into the solver: every gesture was a
`page.mouse.*` call with a Bezier trajectory in front of it and a random sleep
behind it, and there was no way to ask for anything else — which meant no way to
drive a phone, and no way to turn it off when your stack already did it.

### Added

- **`humanization: 'mouse' | 'mobile' | 'none'`** (`humanization=` in Python),
  and **`humanizer`** for one of your own. Unset reads `CAPTCHA_HUMANIZATION`,
  then defaults to `mouse` — which is byte-for-byte what the driver did before,
  so nothing moves for an existing caller. The env var deliberately loses to
  anything set in code: which mode is right is a property of the page you are
  driving, not a deployment decision.

- **`'mobile'` — real touch events with finger kinematics.** It never touches
  `page.mouse`. A mousemove at a touch-only widget is the *wrong event*, not a
  weaker one: the page's touch handlers never fire and the solve fails for a
  reason nothing reports. What it does instead:
  - **no hover.** A move with nothing on the glass dispatches nothing at all,
    and the reCAPTCHA tile-hover and the idle drift during inference switch
    themselves off rather than emitting a cursor that does not exist.
  - **taps wobble.** A finger held for 90 ms does not report one unchanging
    coordinate — the contact centroid rolls a pixel or two under pressure. A tap
    with zero movement between `touchstart` and `touchend` is a synthetic tap.
  - **`generate_swipe`**, a separate motion model rather than the mouse one
    retuned. A finger leaves fast and brakes late (where a mouse hand is
    symmetric), bows far less, wanders low-frequency instead of jittering with
    speed, and **never overshoots** — that tell is a hand arriving past a target
    it cannot see under the cursor, and a finger occludes its own target.
  - Fitts's law with touch constants: a wider effective target (a fingertip is
    the 44 pt / 48 dp both platform guidelines are built around) and a slower
    intercept, so every gesture takes longer than its mouse equivalent.

- **Appium / WebdriverIO / Selenium support**, via `touchDriver`
  (`touch_driver=`). Gestures go out as **W3C pointer actions** with
  `pointerType: "touch"`, and a whole swipe is **one action chain with
  per-sample durations** — the kinematics are reproduced by the device rather
  than by a loop round-tripping over the wire, over which a 90 Hz path is not
  90 Hz at all. Press and release stay separate calls, because W3C input state
  persists per session and that is what lets the puzzle-slider driver press,
  screenshot, steer, screenshot and only then let go. Nothing imports Selenium:
  the payload is the raw protocol one.
  `touchTransform` / `touch_transform` maps CSS pixels onto screen pixels
  (`{scale, origin}`) — upstream is whatever `boundingBox()` returned, a handset
  wants device coordinates, and the difference is `devicePixelRatio` plus the
  chrome above the webview. Neither is guessable from inside the solver, so it
  is a parameter and not a heuristic.

- **`'none'`** — one move, one press, one release, and text in a single `fill()`.
  Roughly an order of magnitude faster on a slider, and detectable by anything
  scoring pointer telemetry. That is the trade, stated plainly: it is for your
  own fixtures, and for stacks that humanise elsewhere. Camoufox is the concrete
  case — its `humanize` juggler re-humanises every `mouse.move()` it is handed,
  and composing the two measured **82.1 s against 13.4 s** on one geetest slide.

- **`startingMousePosition` is now honoured on the first gesture too.** The
  camoufox origin-seeding step used to overwrite it with the window centre.

- **`SolveResult.phases`** — where a solve's wall clock went, in milliseconds
  per phase (`detect`, `settle`, `inference`, `mouse`, `await-verdict`, …), on
  both ports. The same partition `CAPTCHA_TIMINGS=1` prints to stderr, handed
  back to you instead: "the solve took 12s" is not actionable and "the settle
  monitor spent 4s of it" is. Phases nest, so one inside a differently-named one
  counts under both — the cursor drifting over the widget *while* the model
  generates is genuinely mouse time and inference time — and the totals can
  therefore exceed the elapsed time.

- **`stepScreenshotTimeoutMs`** (default 2000) — how long an `onStep`
  observer's snapshot may take. Separate from `elementScreenshotTimeoutMs`,
  which is sized for the picture the model reads. Ignored entirely when no
  `onStep` is set.
### Fixed

- **`captchakraken.__version__` said `2.6.0` while 2.6.1 was on PyPI and npm.**
  Nothing compared it against the manifest pip actually reads, so every caller
  branching on it — and every bug report quoting it — named a release that was
  not the one running. Now checked by a test.

- **The MCP server told every client it was `0.1.0`** while npm published
  `0.1.2`. `serverInfo.version` is what an MCP client logs and what a user
  quotes in a bug report, so the drift pointed every report at the wrong
  release. The CI handshake smoke now compares the advertised version against
  `package.json` and fails the build on a mismatch — the literal had nothing
  comparing it to anything, which is why it drifted twice without notice.

- **`CaptchaKrakenAPIError` is exported from the Python package root**, as it
  already was from the TypeScript one. `docs/hosted-api.md` tells every caller
  to branch on its `.code`, and that recipe did not work in one of the two
  languages the docs claim parity for: the type existed only at
  `captchakraken.errors`. Purely additive — the old import path still works, and
  `errors` imports nothing but `typing`, so this adds no dependency floor.

### Changed

- **The pointer position moved out of the solver** and into the humanizer, in
  both ports — a mode that dispatches no motion still has to answer where the
  next gesture starts. `SolveResult.finalMousePosition` /
  `final_mouse_position` are unchanged.

- **Every inter-gesture wait is named** (`tap`, `between`, `grab`, `drop`,
  `probe`, `settle`, `key`) and comes from the mode's own table, rather than
  from a literal at each call site. A finger dwells on a tap about four times as
  long as a mouse button is held, and neither number means anything to a caller
  who turned humanisation off. An unknown name yields no wait rather than
  raising, so a new pause site cannot break a humanizer written against an older
  release.

- **Solves are markedly faster, without moving any humanisation.** Nine waits
  came out of the driver, all of them found by asking `SolveResult.phases`
  where the time was going on solves that already *worked*. Across the full
  fixture suite, both ports: median solve 6.8s -> 5.7s, p95 33.6s -> 21.6s.
  No pause table, trajectory constant or gesture changed.

  - the reCAPTCHA grid driver now reports the Verify press as an interaction,
    so the caller polls for the vendor's verdict instead of taking the
    no-interaction wait — and can no longer abort a correctly submitted answer
    with "performed no interactions";
  - its round 1 no longer re-waits for a grid the caller has just waited for;
  - there is ONE wait after a round and it is polled, not slept: a round that
    finished the captcha now ends when the widget goes, not 1200ms later;
  - the widget's submit control is travelled to once, not twice;
  - the planner keeps ONE pooled connection instead of re-dialling the endpoint
    per inference (measured 258ms fresh vs 144ms pooled);
  - `ensure_server` decides an endpoint is remote before asking it for /health.

  Node-only, and the reason that port measured seconds slower per solve:
  `scrollIntoViewIfNeeded` and the `onStep` observer's snapshot are both now
  bounded (they inherited Playwright's 30s and the model's 8s budget, and both
  wait for the widget to stop animating — 8.0s of one measured 12.0s solve);
  the cursor drift during inference is interruptible; the interpreter and the
  adapter name are resolved once per solver rather than per inference; and the
  widget disappearing now ends the post-submit window for the eight vendors
  that ship no response token, as it already did in Python.

## [2.6.0] - 2026-08-19

A solve-timing release. Nothing here changes what the model answers; it changes
how much of a solve is spent waiting for things that cannot happen. The measured
end of it: a `recaptcha_grid_4x4` attempt that used to run 66.1s and fail now
gives up at round 4, and Tier 3 stops spending half its wall-clock on attempts
that had no way to succeed.

### Changed

- **The solve budget is 45s over 6 rounds**, down from 120s over 10
  (`overallSolveTimeoutMs` / `overall_solve_timeout_ms`, `maxSolveLoops` /
  `max_solve_loops`). A round costs ~4-7s once the waits below are paid, so six
  rounds is what fits; ten never did, and only ever expressed itself as a
  timeout. The cap is now a BACKSTOP — reaching it is a bug report, not a normal
  outcome.

- **The post-submit outcome window is 1000ms, polled every 75ms**
  (`postSolveOutcomeTimeoutMs` / `post_solve_outcome_timeout_ms`, and the new
  `postSolveOutcomePollMs` / `post_solve_outcome_poll_ms`). A WRONG answer spends
  the whole window by construction — the vendors emit nothing on a wrong answer,
  they just re-deal — so the only number that can size it is how late a real
  SUCCESS arrives: p50 360ms, max 528ms over 34 successful rounds across 20
  puzzle types. 2500ms was being spent in full on every unsuccessful round,
  25.5s of one 66.1s solve.

- **hCaptcha's image wait is bounded at 3s**, not 8 (`hcaptchaImagesTimeoutMs` /
  `hcaptcha_images_timeout_ms`). It is best-effort — the screenshot happens
  either way — so it should cost what a loading tile plausibly costs, not
  two-thirds of a whole solve.

### Added

- **A solve that repeats itself is abandoned** (`maxNoProgressRounds` /
  `max_no_progress_rounds`, default 2). At temperature 0 the model is a function
  of the picture, and every answer this driver produces is EXECUTED — so the same
  answer twice means the previous one already ran and the page is still asking
  the same question. Measured: nine identical click sets, each clicked, each
  rejected, ending at the round cap. 2 rather than 1 because the first repeat
  also escalates to a RECORDING, which is the one recovery still worth trying:
  a board that cycles reads as a still and answers the same way every round.

- **A per-solve phase budget**, printed to stderr under `CAPTCHA_TIMINGS=1`
  (Python). "The solve took 77s" is not actionable; "the settle monitor spent
  31s of it" is. Always accumulated, only printed under the flag, and printed on
  the way out of a FAILED solve too — that is the one whose time you want
  itemised.

### Fixed

- **A missing Appium `touchTransform` scale is now refused instead of
  dispatched.** The transform is the one part of the touch path that fails
  silently: hand it a wrong (or absent) `scale` and nothing raises anywhere —
  the W3C chain is valid, the device performs it, the finger lands somewhere
  else, and the solve fails looking exactly like a model that cannot read the
  puzzle. Same shape as the slider bug below, one seam over. A `touchDriver`
  with no `scale` on a page reporting `devicePixelRatio != 1` now raises on the
  first gesture and names the value to pass. An explicit `scale: 1` is taken as
  the caller's word that the coordinates are already mapped — unset and an
  explicit 1 are different facts, and only the first is checked. An unreadable
  ratio falls through to the identity: absent evidence is not evidence of a
  mismatch. `origin` cannot be measured from inside the page, so it is named in
  the refusal rather than guessed at.

- **The slider missed every attempt on a phone.** `executeSlide` closes a loop
  between two pixel spaces — it steers the handle in CSS pixels (`boundingBox`,
  the probe offsets, the slot the model named) and MEASURES the piece in the
  screenshot's pixels, where the CV masks the handle and reports what moved.
  Those are the same number on a 1x desktop, so the loop was right for a year
  and the two spaces were never told apart. At a device-pixel ratio of 2.625 the
  mask landed on a strip of empty card ABOVE the handle, leaving the handle the
  largest moving thing in frame; the widths came back 2.6x too wide;
  `solveSlideGeometry` threw them out as wider than the widget; and the drive
  fell back to open-loop guessing. Measured on the geetest_v3_slide fixture:
  0/2 ports on a Pixel 7 against 2/2 on the same three boards under a mouse,
  and 2/2 after. The scale is read off the shot rather than asked of the page —
  what the CV measures is the image, whatever the window thinks its ratio is.
  Both ports, same commit.

- **GeeTest's OK button was never pressed.** It ships as
  `<div class="geetest_submit geetest_disable">OK</div>` — invisible to both
  shapes the button finder knew, because a bare div carries no `role="button"`
  and "OK" is on none of the four word lists. A GeeTest board does not grade
  until you press it, so the loop re-read the same unchanged panel and
  re-answered it identically until the round cap: ordered icon-click scored
  0/31 and 0/13 while the model was answering CORRECTLY. Matched by CLASS, not
  by the word — `geetest_submit_tips` sits beside it, also reads "OK", and does
  nothing when pressed.

- **A readiness gate with nothing to check no longer reads as "not ready".**
  The last clause asked whether hCaptcha's example image had loaded and returned
  false when there was no example image at all, so a challenge with no tile
  grid, no canvas and no example polled out the full timeout and carried on
  regardless — 24.0s of a 45.2s solve, three times over. A gate can only report
  on what it can see; with nothing to check it has no opinion, and no opinion
  must not read as "not ready".

- **An `even` clip no longer waits 6s per click for a state that never
  recurs.** The slicer picks that mode precisely when the clip never revisits a
  picture it has already shown, so the keyframe gate can only run out its full
  window before clicking the coordinates it already had. It is also the normal
  case, not a corner: all 116 real clips are `even` and `cycle` has never fired
  on real footage. The gate stays for `cycle`/`static`, where a state does come
  back and waiting is the difference between the sprite and the background.

- **The Python port recorded CSS-animated widgets frozen.** Its burst took every
  frame with `animations="disabled"`, which fast-forwards finite animations and
  FREEZES infinite ones — right for a model-facing still, fatal for a recording.
  GeeTest's svg board came back as forty copies of one picture, the slicer
  honestly reported `mode=static`, and the solve went back to answering a single
  still. The TS port has passed `allow` since it hit this first; hCaptcha hid it
  there, because it animates in canvas and the flag does not touch canvas.

- **A widget that never renders reports "no captcha" again, not "still detected
  after N loops".** The render-wait cap was a flat 6 and a render wait consumes
  an attempt, so at `maxSolveLoops` 6 the loop ran out first and the branch never
  fired — turning the correct, benign answer for a reCAPTCHA v3 / invisible page
  into a hard error for anyone catching `NoCaptchaFoundError`. It is now tied to
  the loop count in both ports.

## [2.5.0] - 2026-08-18

### Added
- **Solve captchas as they appear.** `solver.watch(page)` (TS) and
  `PageSolver.watch(page)` (Python) install a watcher that probes the page and
  solves any challenge that becomes visible, so a script no longer has to know
  where a captcha might interrupt it. Exposed through camoufox as
  `watchCaptcha` / `watch_captcha`.

  It injects **nothing** into the page. The obvious build — a `MutationObserver`
  signalling out through an exposed binding — reacts faster and is the one
  design that cannot be stealthy everywhere, because a binding is a function on
  `window` and an observer is script a vendor can enumerate. Instead it drives
  the existing `detectCaptcha()` from the driver side on a timer, so there is no
  new detection surface on any launcher; under camoufox those DOM reads land in
  the sandboxed Juggler world for free, since that is camoufox's default for all
  Playwright evaluation. The cost is reaction time bounded by `interval_ms`
  rather than by the mutation.

  Python's blocks (`run()`) where TypeScript's returns a handle, and offers
  `poll_once()` for callers with their own loop: a sync Playwright handle cannot
  be driven from a worker thread, so a background watcher is not available to it.

### Fixed
- **Only a FADING reCAPTCHA is worth a second look.** The 3x3 driver re-solved
  every board until the model said `done`, so an ordinary "select all images
  with X" cost two inferences: one to answer it, and one to be told there was
  nothing left. Only the board that SWAPS a clicked tile out needs that, because
  only there does a click deal a photo nobody has read yet.

  The widget says which kind it is, in the reply it gives a click. A tile it
  KEEPS gets the small blue chip in its top-left corner; a tile it is REPLACING
  gets a large blue check across the middle and dissolves to white under it. A
  widget that swaps one clicked cell swaps them all, so the two are never on one
  board and one look at the tiles we just clicked settles it — and the chip is
  already what `detect_selected_cells` reports as `selected`, from the same
  per-poll CV call the driver was making anyway. Chip on every tile we clicked →
  press Verify, same as a 4x4.

  Ordering matters: a chip landing ZOOMS the photo out, which reads as
  `changing` on the very frame that shows the chip. Testing for the swap first —
  which is all the driver used to do — makes a chipped board look like a
  swapping one for as long as that animation runs, which is how the wasted round
  got spent. The verdict also needs EVERY clicked tile, because the two mistakes
  do not cost the same: calling a swapping board finished submits half an answer
  and burns the attempt, while calling a chipped board unfinished costs one
  inference, which is what the old code paid every time.

- **A photograph is not a selection badge.** `detect_selected_cells` runs BOTH
  vendors' badge tests on every tile, so hCaptcha's — a small blue-teal disc
  with a white glyph, top-right — was applied to reCAPTCHA boards too, and it
  was two pixel COUNTS and nothing else: >=8 teal-ish pixels anywhere in the
  corner patch, >=2 near-white pixels anywhere in the same patch. Blue sky with
  a white pole in it satisfies both. The tile is then reported as already
  selected, `solver._solve_grid` drops it from the model's answer, and it is
  never clicked — a correct answer thrown away on a board where nothing was
  selected at all. On the synthetic corpus that cost a target tile on 9% of
  clean boards; on the real reCAPTCHA eval captures it fired on 3 boards of 39.

  What a badge has and a photograph does not is that the white BELONGS TO the
  teal mark — inside the disc, or hugging its rim. The counts stay as a cheap
  gate and the verdict now needs >=2 white pixels within 2px of the teal blob's
  convex hull. Phantom selections over 3051 tile corners of boards with nothing
  selected: 74 -> 15. Rendered badges found: 1200/1200, both the filled-disc and
  white-ringed renderings, 9-16px, colour-jittered and JPEG'd.

  Two pixels of slack, not zero, because the ringed rendering draws the white ON
  the rim and fragments the teal under it — a strict inside-the-hull test finds
  a quarter of those. Nor a delta-E on the badge's colour, which is the obvious
  tightening and the wrong one: pinning the hue drops recall to 40% under
  jitter, because we do not know that colour to a delta-E's precision. The
  reCAPTCHA corner chip was measured at the same time and is clean — 0 false
  positives on those same corners — and is unchanged.

- **The Puppeteer adapter was never actually verified.** Its header claimed it
  was "verified against Puppeteer 24.x" while nothing in the package touched it:
  there was no test for it, and Tier 3 drives camoufox on both ports. It is now
  covered by unit tests for every API delta it bridges, plus a live check that
  launches real Puppeteer and real Playwright and drives every member of the
  structural page surface through them (skipped when those libraries are not
  installed — this package still ships with zero browser dependencies).

  That verification found a real gap: `fromPuppeteer` did not forward
  `isClosed`, which the new watcher reads to end its loop. A Puppeteer-driven
  watcher would have polled a closed page forever.

- **Distorted-text captchas are typed, and puzzle-piece sliders are dragged.**
  Two answer families the model was already trained to produce and the driver
  silently threw away. `ActionPlanner._normalize_pixel` dropped both — a
  `{"action": "type", "text": …}` answer has no coordinate for any branch to key
  off, and the drag branch required BOTH ends, so the sourceless drag the
  "FOR PUZZLE PIECE SLIDER PUZZLES" clause explicitly asks for parsed to
  nothing. A perfectly correct answer became "unsupported".

  **Text.** The driver decides from the DOM, not the picture: a visible text box
  in the challenge selects the distorted-text prompt (`--text-mode` on the CLI)
  and skips grid detection, because BotDetect's boxed glyphs are exactly the
  lattice `find_grid` looks for. The answer is typed character by character with
  jittered gaps, after a real pointer move and click — `fill()` would set the
  value with no keystrokes at all, and these are the vendors that score cadence.
  A retry clears the box first, so round 2 cannot append to round 1.

  **Sliders.** Closed-loop, not a calculation. The model gives the centre of the
  gap — all the picture can tell it — but not how far the handle must travel to
  put the piece there, which is a ratio several vendors deliberately vary. So the
  driver presses the handle, nudges it twice by known amounts, and watches:
  `union(before, after)` spans the piece's original left edge to its current
  right edge, so its width is `piece_width + ratio x nudge`. Two nudges, two
  unknowns, solved; then it steers the remainder and re-measures. The mouse is
  not released until the piece is home, because on every one of these puzzles
  **releasing is the submit** — which is also why a completed slide is never
  followed by a Verify click.

  New surface: `TypeAction` (JS), a nullable `DragAction.source_bounding_box`
  (null = slider), `TEXT_INPUT_SELECTORS` / `SLIDER_HANDLE_SELECTORS` /
  `DRAGGABLE_PIECE_SELECTORS` (vendor-first, generic-last — the generic tail is
  what fires on most real pages), the `slide_*` knobs on `PageSolverConfig`,
  `tool_calls/track_piece.py`, and CLI `track-piece` (also a `serve` cmd, since
  it runs several times per drag with the button held). `prompts.py` gained the
  generation-2 `text` family, which the client had been missing since the
  finetune repo defined it.

  **Requires a generation-2 model.** Generation 1 — including the currently
  served `CaptchaKraken_v1.1` — has no text prompt and no slider clause, so a v1
  model is never asked for either answer. Text captchas now report
  `UnsupportedCaptchaError` naming that reason rather than clicking at random.
- **Animated challenges are solved instead of skipped.** hCaptcha's "select the
  odd animal" (sprites cross-fading on independent cycles) and "unique motion
  pattern" (identical meshes, only the rotation differs) carry none of their
  answer in any single frame. The driver now records the widget for 4 s at 10 fps,
  reduces the recording to the few stills that carry the answer, and sends those
  as **one multi-image request**.

  The answer gains a `"frame"` naming which still it acted on, and the driver
  **holds the mouse until the widget looks like that frame again** before
  clicking — comparing only the neighbourhood of the click point, because
  everything else on these puzzles is moving too. Without that wait, a click on a
  fading sprite lands on background.

  New surface: `CaptchaSolver.solve_keyframes()`,
  `ActionPlanner.get_keyframe_actions()`, `keyframes.py` (a verbatim copy of the
  training repo's slicer — the two are checked byte-for-byte in CI, because the
  model answers with a frame NUMBER and a solver that sliced differently would
  wait for a picture that does not exist). Actions carry `await_keyframe` +
  `frame`. CLI: `solve-animated --frames-dir DIR` and `match-region` (also a
  `serve` cmd, since the wait gate polls it every ~120 ms).

  Config: `video_solve_enabled` / `videoSolveEnabled` (default on),
  `video_burst_duration_ms` / `video_burst_fps`, `keyframe_wait_timeout_ms` /
  `keyframe_wait_poll_ms`, and the camelCase equivalents on the TS side.

  **The clip is never sent to the model.** Every mp4 this project can write is
  MPEG-4 Part 2, whose decodability on the serving side was never verified, and a
  clip cannot carry a frame number in the first place. Frames stay in memory and
  are sliced there, so no encode happens on the solve path at all.

  Accuracy depends on the adapter: the pipeline is in place, and an adapter
  trained on the keyframe format is what makes the answers good.

### Changed
- **License bumped to CaptchaKraken Source-Available License v1.1.** One new
  restriction and one new clarification, both about stealth browsers.

  v1.0 listed "stealth / anti-detection browsers and browser automation
  frameworks" as an illustrative *permitted* commercial use. That gave away the
  thing worth keeping: any antidetect browser vendor could ship CaptchaKraken as
  a built-in feature and market the solve as theirs. **§3(d)** now requires a
  commercial license to embed, bundle, preinstall, fetch on demand, or advertise
  the Software as a captcha-solving capability of a stealth/antidetect browser,
  profile manager, or automation platform **distributed to third parties**.

  It is a restriction on **distribution, not use**, and **§2(c)** says so
  explicitly, because the ambiguity would otherwise land on exactly the people
  this project is for: pointing the solver at Camoufox, Puppeteer, Playwright, or
  any stealth browser to automate *your own* work stays unrestricted, commercial
  or not. If you are the one clicking "run", nothing changed for you.

  **Forward-only.** v1.1 governs releases distributed under it. Copies obtained
  under v1.0 — including the already-published `CaptchaKrakenV1_Lora`,
  `Sunlight-AWQ-4bit`, and `Twilight-FP8` weights — remain governed by v1.0; a
  grant already made cannot be revoked by editing a file. `docs/licensing.md`
  carries the plain-English version and a using-vs-shipping table.
- **`AnimatedChallengeError` / `.animated` narrowed.** It used to mean "the
  challenge never settles, give up". It now means "an animated challenge we could
  not RECORD" — the element refused to screenshot, or `video_solve_enabled` is
  off. A moving challenge is no longer a failure.
- The Python driver now runs the settle probe for `puzzle_source == "unknown"`
  (GeeTest, Tencent, …) as well as hCaptcha. It never did, so an animated
  non-hCaptcha widget was screenshotted mid-cycle and answered from whatever
  single moment happened to be caught. reCAPTCHA is deliberately excluded: it has
  its own readiness gate and its grids are never animated.
- `"unsupported"` mid-solve followed by a never-settling next round used to be
  terminal. It now retries into the recording path, still bounded by
  `max_unsupported_resolves`.
- The frame-freshness guard is skipped for animated challenges. It re-solves when
  the frame changes during inference, and these change by definition — every
  attempt would be judged stale and the whole re-solve budget would burn without
  ever acting. The `frame` in the answer is the guard that replaces it.
- `CaptchaSolver.solveVideo()` still aliases `solve()` (one still, one answer) and
  is NOT redirected to `solve_keyframes()`: callers of that name pass a single
  media path, and reinterpreting it as "record and slice" would change what an
  existing integration does. New code calls `solve_keyframes()` explicitly.

## [2.4.0] — 2026-07-29

The release that makes the **hosted** API usable by someone who has never heard
of vLLM. Nothing here changes self-hosting.

### Added
- **`captchakraken-mcp`, a new npm package**, published from this repo as a
  third workspace (`mcp/`). Signs you in with GitHub, mints a solving key, and
  reports balance, usage and a top-up link. `npx captchakraken-mcp`.

  It is a **separate install** from `captchakraken`, deliberately: `npx`
  resolves by package name, and the MCP's dependencies have no business on
  every browser-driver install. It also keeps its own version line — 0.1.0
  here — so a solver change does not force an MCP release.
- `CaptchaKrakenAPIError` is exported from the TypeScript port, carrying
  `code`, `resolutionUrl` and `retryAfterSeconds`. Branch on `code`; the prose
  is not a contract.
- The credentials file may now carry the **endpoint** alongside the key
  (`VLLM_BASE_URL=` or `CAPTCHA_KRAKEN_BASE_URL=`).

### Changed
- **Hosted-API refusals now explain themselves.** Running out of credits used
  to produce `vLLM 402 Payment Required at https://api.captchakraken.com/...`,
  naming infrastructure the user never installed. Out of credits, rate limited,
  account suspended, request too large, and abandoned attempts now each produce
  a sentence naming CaptchaKraken and the URL that resolves it, in both ports.
  429 honours `Retry-After`.

  An unrecognised code still produces a useful message — it carries the
  server's own text through — rather than falling back to something generic.
- **`create_api_key` no longer returns the secret.** It writes
  `~/.captchakraken/credentials` at 0600 and reports the path, so a live key
  cannot reach an agent transcript. It writes the endpoint at the same time.
- **A key from the MCP no longer needs `VLLM_BASE_URL` set.** `base_url()`
  reads the credentials file when the env var is absent, so a hosted user's
  first solve reaches `api.captchakraken.com` instead of dialling a local port
  with nothing behind it. An explicit `VLLM_BASE_URL` still wins, and a machine
  with no credentials file still defaults to localhost.

### Unchanged, on purpose
- Self-hosting. A local vLLM sends no error envelope, so it still produces the
  old message, bearer-token hint and all. A bare-token credentials file still
  yields no endpoint, so nobody who hand-wrote a local key into that file gets
  silently redirected at our servers.

## [2.3.0] — 2026-07-24

### Added
- **A Python page driver — you can now solve captchas from Python end to end.**
  Until now the Python port was image-in / actions-out: you handed it a PNG and
  got back click/drag actions, and something else had to own the browser.
  Everything that actually drives a page lived only in the TypeScript driver, so
  Python callers could not use the solver against a live site at all.

  ```python
  from playwright.sync_api import sync_playwright
  from captchakraken import PageSolver

  result = PageSolver().solve(page)   # detects, solves, clicks, submits
  ```

  `captchakraken.page_solver` mirrors `js/src/solver.ts`: the same detection
  order, the same freshness guard, the same multi-round driver for reCAPTCHA's
  dynamic 3×3, the same under-selection retry, and the same submit policy.
  Verified against live reCAPTCHA — on a real dynamic 3×3 it drives the rounds
  and submits on `done`, exactly as the TypeScript driver does.

  **The split is identical on both sides**: vision, CV and prompting stay in
  Python (`solver.py`, `planner.py`, `tool_calls/`); the driver only finds the
  challenge and clicks. The TypeScript driver reaches the Python half by
  spawning the CLI; this one calls the same functions in-process — no
  subprocess, no CV worker to leak. That is the *only* intended difference, so
  the two cannot drift on anything that decides accuracy.

  Imports no browser package: pass any Playwright-compatible page (`playwright`,
  `patchright`, camoufox) and it duck-types the slice it needs.

  **Synchronous only.** An async mirror is not yet written — a sync Playwright
  handle cannot be driven from inside an event loop, so `AsyncCamoufox` and
  `async_playwright` users are not covered by this release.
- **`camoufox` integration.** `pip install "camoufox[captcha]"` then
  `from camoufox.captcha import solve_captcha`. Requests are tagged
  `camoufox/<version>` for attribution.
- **Human mouse trajectories in Python** (`captchakraken.trajectory`). Same
  `(points, cumulative_timings)` contract as the TypeScript driver's cursory-ts
  call, so pacing is driver-independent: Fitts's-law duration, Bezier arc,
  ease-in-out velocity, speed-scaled jitter, and overshoot-and-correct on longer
  moves. An independent implementation, not a port — cursory-ts selects from a
  bundled corpus of recorded human traces, which is that package's own asset.

### Fixed
- `__version__` in the Python package said `2.0.0` through the entire 2.2.x
  line. It now tracks the real version.

Three more, all found by running the new driver against live reCAPTCHA rather
than against fixtures — none of them were visible in the hermetic tests:

- **An already-passed captcha raised instead of returning.** If the vendor had
  already cleared the widget, nothing was detectable to solve and we had not
  interacted, so the render-wait branch ran out and raised
  `NoCaptchaFoundError`. That is the *common* case behind a good stealth
  browser — camoufox frequently clears reCAPTCHA on the checkbox alone — so the
  best possible outcome was being reported as an error the caller had to catch.
- **`scroll_into_view_if_needed()` inherited Playwright's 30 s default** and runs
  once per action plus once per submit. On a challenge iframe that is
  mid-animation it waits for stability and burns the full 30 s each time, which
  turned a ~5 s solve loop into minutes. Now bounded to 2 s; the element has
  just been screenshotted, so there is nothing to lose.
- **`overall_solve_timeout_ms` was not actually a budget.** It was checked only
  at the top of each attempt, so a single slow attempt overran it without bound
  — nothing looked at the clock again until that attempt returned. A camoufox
  session was observed running past ten minutes against a nominal 120 s timeout.
  The deadline is now enforced inside the long-running loops (each action
  executed, each round of the dynamic grid driver), so the configured budget is
  the real ceiling. *The TypeScript driver has the same structural gap and has
  not been changed here.*
- **Mouse moves could wedge camoufox permanently.** camoufox humanises every
  `mousemove` into its own trajectory, guards the intermediate points against
  the window bounds, and then dispatches the requested destination *unguarded*
  ("always finish exactly on the requested destination"). A destination outside
  the window fires as an exit event rather than `eMouseMove`, so no
  hit-renderer ack returns; dispatch is serialised on a process-global
  activation chain, so that one missing ack hangs **every later input event
  forever**. The symptom is `page.mouse.move()` never returning — 0% CPU, no
  in-flight work, a solve that looks dead. Same failure family as camoufox #225.
  The driver now resolves the real window (`viewport_size`, falling back to
  asking the page for `innerWidth/innerHeight`, since camoufox reports
  `viewport_size = None`) and clamps a pixel inside it. With this, three
  consecutive live reCAPTCHA solves through camoufox returned real tokens.
- **Progress output was invisible when piped.** Python block-buffers stdout when
  it is not a TTY, so every line of a minutes-long solve appeared at once on
  exit — a working run looked exactly like a hung one in a log file or CI.

## [2.2.1] — 2026-07-24

Point release. No API breaks; every change below is either a fix for a puzzle
class that silently failed, or an opt-in header/credential path that is absent
unless you deliberately set it.

### Fixed
- **Drag puzzles failed as "unsupported" against the current adapter.** The
  LoRA was retrained on the content schema (`action: drag`, `drags[]`,
  lowercase) while `planner.py` still asked for the legacy output/PascalCase
  schema. The model answered in a hybrid, the parser dropped it, and every drag
  puzzle failed — with no test going red. The inference prompt is now synced to
  the trained schema, and the parser accepts every `simulate_drag` shape the
  model emits rather than one canonical form.
- **30-second stale-element hangs in the JS driver.** A handle captured before
  the frame re-rendered was awaited until the Playwright timeout; the solver now
  detects the detach and re-acquires instead of blocking the whole solve.

### Added
- **Pinned serving manifest** (`pinned_model.json`): the base model, the LoRA
  adapter and revision, and the SHA-256 of each serving prompt, asserted in CI
  (`tests/test_pinned_model.py`). Editing a serving prompt now fails CI until
  someone consciously re-pins, which forces the question "does the pinned
  adapter still expect this prompt?" — the check that would have caught the drag
  regression above on the day it landed.
- **Credentials file.** When neither `CAPTCHA_KRAKEN_API_KEY` nor `VLLM_API_KEY`
  is set in the environment, the key is read from
  `~/.captchakraken/credentials` (override the directory with
  `CAPTCHA_KRAKEN_STATE_DIR`). Env always wins, so nothing that works today
  changes; this only removes the need to keep a bearer token in your shell
  profile.
- **Hosted-API attribution headers**, both optional and absent unless set, so
  self-hosted users are unaffected:
  - `X-CK-Client` (from `CAPTCHA_KRAKEN_CLIENT`) — which integration issued the
    solve, e.g. `camoufox/0.4.11`. Attribution only: it is caller-supplied and
    is never priced on.
  - `X-CK-Session` (from `CAPTCHA_KRAKEN_SESSION`) — groups the 1..N inference
    rounds of one captcha into a single billable attempt. The JS driver mints a
    UUID per `solve()` and reuses it across every CLI invocation in that solve.
  Both values are sanitized before they reach the wire — a CR/LF in the
  environment would otherwise splice arbitrary headers into the request.
- **Fleet-routing priority header.** Setting `CAPTCHA_REQUEST_PRIORITY` to a
  positive int sends `X-JH-Priority: <n>`, which a fleet front-end can route on
  to keep throwaway traffic (e.g. a CI gate) off the production GPU.
  Deliberately a header, not vLLM's request-body `priority` field, which is
  lower-is-higher and would misorder against the server's own scheduling
  classes.

### CI
- The hermetic suite now runs in full on every PR; the dead v1-only tests are
  skipped **visibly** rather than silently collected. New coverage: solver
  contract, grid parse, routing headers, credentials file, pinned manifest.

## [2.2.0] — 2026-07-15

### Added
- **Freshness guard — never act on a stale frame.** reCAPTCHA/hCaptcha fade
  fresh tiles in over ~1s; if the frame changed *while the model was
  generating*, its answer described an "undeveloped" frame whose tiles no longer
  lined up. The solver now re-screenshots after every model query and diffs it
  against the frame it sent (`check-movement`); on a change it discards the stale
  answer and re-solves on the developed frame. Covers both the one-shot path and
  the reCAPTCHA 3×3 dynamic driver. Tunable via `staleFrameReSolveEnabled` /
  `staleFrameDiffThreshold` / `maxStaleFrameReSolves`; `check-movement` was added
  to the persistent CV worker so the check runs on the warm process.
- **Unified `captchakraken fetch` updater.** One command pulls the latest model
  from the HuggingFace org (https://huggingface.co/CaptchaKraken) *and* upgrades
  the vLLM serving stack, then restarts a running local server. Flags:
  `--weights-only`, `--engine-only`, `--no-restart`, `--dry-run`. Shell
  equivalent: `./setup.sh --update`.
- **Documentation hub.** Most of the README moved into a browsable
  [`docs/`](docs/README.md) tree (self-hosting, usage, how-it-works, performance,
  roadmap, licensing); the README is now a slim overview + quickstart.

### Fixed
- **License metadata corrected.** Both ports previously declared `GPL-3.0`, which
  contradicted (and would have *overridden* with a permissive license) the
  source-available `LICENSE` that prohibits selling the solve. The npm and PyPI
  packages now declare the CaptchaKraken Source-Available License and ship the
  `LICENSE` file.

### CI
- The Python job is now a **no-regression gate**: grid detection + the freshness
  check + the fetch command, run on every PR (still hermetic — no GPU/network).

## [2.0.0] — 2026-06-07

### ⚠️ Breaking
- **Complete solver rewrite.** v2 replaces the v1 architecture (SAM3 grounding +
  general multi-provider LLMs: Gemini / OpenRouter / Ollama) with a single
  purpose-built **Qwen3.5-9B grid LoRA** served on a local **vLLM** server.
- **Solver API / config changed.** `apiProvider`, multi-provider `model`, and
  provider `apiKey` options are gone. The solver now reads just two env vars —
  `VLLM_BASE_URL` and `CAPTCHA_KRAKEN_API_KEY` — and defaults to the published
  grid LoRA, so `new CaptchaKrakenSolver()` needs no model/provider.
- v1's `transformers` / `torch` / SAM3 dependencies are removed from the solver
  venv (available on the `v1-old-architecture` branch).

### Added
- **Bring-your-own browser — zero browser dependency.** The package no longer
  depends on any browser library in any form: the vendored `camoufox-js`,
  `patchright`, and `patchright-core` are gone (not even devDependencies). The
  public API types `solve(page)` against an implementation-neutral, self-contained
  structural Playwright `Page` interface defined by the package itself (not
  imported from `playwright-core`), so any Playwright-compatible launcher works —
  vanilla `playwright`, `patchright`, `camoufox-js`, etc. Install whichever one
  you want yourself and hand the solver its `Page`. (The live solve-and-record
  tests moved to the parent `CaptchaKrakenFinetune` repo, which owns the launcher.)
- **Puppeteer support via `fromPuppeteer()` adapter.** Puppeteer isn't
  Playwright-API-compatible, so the package exports a thin `fromPuppeteer(page)`
  wrapper that bridges the few differing methods (`viewport`/`viewportSize`,
  `waitForTimeout`, `getAttribute`/`textContent`/`scrollIntoView` via `evaluate`,
  selector-state options). Wrap a Puppeteer page once and pass it to `solve()`.
  All four launchers (Playwright, Patchright, camoufox-js, Puppeteer) are tested
  end-to-end against the live reCAPTCHA demo.
- **`install.sh`** — one-command, hardware-gated setup. Detects NVIDIA VRAM or
  Apple-silicon unified memory, picks FP8 8-bit (≥22 GB) vs AWQ 4-bit (11–22 GB),
  refuses to install below the serve floor (with download-anyway / get-notified
  options), pulls base + grid LoRA, and writes `captchakraken.env`.
- **Source-available LICENSE** — build *with* the model (scrapers, stealth
  browsers, data collection); don't sell captcha-solving as a service or ship
  thin wrappers. See [LICENSE](LICENSE).
- **CONTRIBUTING.md** and a **CI workflow** (`.github/workflows/ci.yml`) running
  hermetic grid-detection tests + a TypeScript build on every PR (no GPU/network).
- **Hermetic grid-detection tests** (`test_grid_detection_ci.py`) that synthesize
  grids in memory — the CI guard for the core `find_grid` invariant.
- **Demo recorder** (`tests/record_demos.spec.ts`) — drives a real browser
  against the live model, tags reCAPTCHA attempts 3×3 vs 4×4, skips/retries
  out-of-scope hCaptcha puzzles, and records videos of successful solves plus a
  per-type solve-rate summary.

### Solver / model
- Grid LoRA (CaptchaKraken's grid adapter) exact-tile accuracy on held-out
  real data: reCAPTCHA 3×3 **94.7%**, hCaptcha 3×3 property **86.7%**,
  reCAPTCHA 4×4 **76.2%** (overall **85.8%**).
- reCAPTCHA dynamic 3×3 (multi-round in-place refresh) and 4×4 one-shot grids,
  hCaptcha 3×3 property grids, and the checkbox / Turnstile flows are solved
  end-to-end. Non-grid hCaptcha puzzles are detected and safely skipped.

### Coming soon
- Hosted cloud API (no GPU required), smaller quantizations, and non-grid
  hCaptcha puzzle support. See the README roadmap.
