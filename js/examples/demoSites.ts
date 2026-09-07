/**
 * Solve several real captcha sites in one run, in one browser, and print a
 * table of what happened.
 *
 *   source ../captchakraken.env        # VLLM_BASE_URL + CAPTCHA_KRAKEN_API_KEY
 *   npx tsx examples/demoSites.ts
 *   npx tsx examples/demoSites.ts recaptcha geetest-slide     # pick a subset
 *   npx tsx examples/demoSites.ts --list
 *
 * WHY ONE BROWSER FOR ALL OF THEM. Launching Holo on a virtual-GPU display is
 * tens of seconds — paid once here instead of once per site. It is also closer
 * to what a real integration does: one long-lived browser meeting captchas as
 * they come, rather than a fresh fingerprint per puzzle, which is a much easier
 * problem than the one customers have.
 *
 * Each site gets its OWN context, though, so a token banked on one vendor
 * cannot wave us through on the next. A demo that silently stopped solving
 * because the first solve satisfied everything afterwards would be the most
 * flattering possible bug.
 */
import { CaptchaKrakenSolver } from '../src/index';
import type { SolveResult } from '../src/types';
import { resolveLauncher, launchOptions, displayMode } from './launcher';

interface Site {
  id: string;
  vendor: string;
  what: string;
  url: string;
  /** Extra settle time for vendors whose widget paints late. */
  settleMs?: number;
  /**
   * Selectors to click, in order, to reveal the puzzle — the VISITOR'S click,
   * not the solver's.
   *
   * GeeTest's demo pages put the widget behind a button: until it is pressed
   * the markup is in the DOM but nothing is drawn, and the solver correctly
   * reports that it can find no interactive widget. That is a property of the
   * demo page, not of the vendor's integration — on a real site the button is
   * whatever the user was already clicking — so pressing it belongs to the
   * harness. reCAPTCHA and hCaptcha need nothing here: the solver drives their
   * checkbox itself, which is why the two shipped single-vendor examples
   * navigate and solve with no setup at all.
   */
  open?: string[];
  /** Selectors that mean the puzzle is now up; any one is enough. */
  ready?: string[];
}

/**
 * The sites. Public demo pages the vendors host themselves, so running this
 * costs nobody anything and hits no customer's property.
 *
 * ORDERED, AND reCAPTCHA IS LAST ON PURPOSE — but not hidden. Its demo page
 * deals a dynamic board perhaps half the time: tiles fade out and are replaced,
 * and it only ends when the board comes back clean, which runs past the
 * client's shipped 45s budget often enough that it is genuinely a coin flip.
 * That is a real property of the product and it has its own recorded figure in
 * the deck. Leading with it means the first thing the room sees is the one
 * puzzle most likely to time out, which tells them less about the model than
 * the other three do. Run `demoSites.ts recaptcha` to go straight at it.
 */
const SITES: Site[] = [
  {
    id: 'hcaptcha',
    vendor: 'hCaptcha',
    what: 'grid or drag, boards in pairs',
    url: 'https://accounts.hcaptcha.com/demo',
    settleMs: 3000,
  },
  {
    id: 'geetest-slide',
    vendor: 'GeeTest v4',
    what: 'slide a piece into its notch',
    url: 'https://gt4.geetest.com/demov4/slide-popup-en.html',
    settleMs: 2200,
    open: ['.geetest_btn', '.geetest_btn_click', '.geetest_radar_btn', '.geetest_holder'],
    ready: ['.geetest_popup_box', '.geetest_popup_window', '.geetest_box', '.geetest_panel'],
  },
  {
    id: 'geetest-icon',
    vendor: 'GeeTest v4',
    what: 'click named icons in order',
    url: 'https://gt4.geetest.com/demov4/icon-popup-en.html',
    settleMs: 2200,
    open: ['.geetest_btn', '.geetest_btn_click', '.geetest_radar_btn', '.geetest_holder'],
    ready: ['.geetest_popup_box', '.geetest_popup_window', '.geetest_box', '.geetest_panel'],
  },
  {
    id: 'recaptcha',
    vendor: 'reCAPTCHA',
    what: 'tile grid, dealt until satisfied',
    url: process.env.RECAPTCHA_URL ?? 'https://www.google.com/recaptcha/api2/demo',
    settleMs: 3000,
  },
];

const fmtMs = (ms: number) => (ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${ms}ms`);

interface Outcome {
  site: Site;
  solved: boolean;
  /** Wall clock from `goto` to verdict — page load and the reveal click included. */
  totalMs: number;
  /** Just `solve()`. THIS is the number the benchmarks report. */
  solveMs: number;
  tokensIn: number;
  tokensOut: number;
  note?: string;
}

/**
 * Why an attempt did not end in a solve.
 *
 * THREE OF THESE ARE NOT THE MODEL, and saying so is the point. A refused
 * session, a dead endpoint and a widget that never appeared all produce "not
 * solved", and scoring them as misses is how a transport fault gets read as a
 * weak model.
 */
/** For the summary table: one short line, not the solver's full report. */
function brief(note: string | undefined, width = 46): string {
  if (!note) return 'not solved';
  // The client appends a usage blob to some errors, and a JSON object pasted
  // into a fixed-width table destroys the table. The full text has already
  // been printed above this, in full, as the attempt happened.
  const first = note.split(/\.\s|\. Total usage|; /)[0].trim();
  return first.length > width ? first.slice(0, width - 1) + '…' : first;
}

function explain(err: unknown, result?: SolveResult | void): string {
  const msg = err instanceof Error ? err.message : err ? String(err) : '';
  const low = msg.toLowerCase();
  if (/econnrefused|fetch failed|connection refused|max retries/.test(low)) {
    return 'never reached the model — endpoint unreachable (check VLLM_BASE_URL)';
  }
  if (/401|403|unauthor|api key/.test(low)) return 'endpoint rejected the credential';
  if (/out of credits|too many times without settling/.test(low)) {
    return 'the vendor declined to serve a puzzle';
  }
  if (/timeout|exceeded/.test(low)) return 'the widget never became interactable';
  if (/no captcha|not find|unsupported/.test(low)) return 'no challenge appeared on the page';
  if (result && !result.isSolved) return 'answered, vendor did not accept';
  return msg || 'unknown';
}

/** Press whatever this demo page puts in front of its widget. */
async function reveal(page: any, site: Site): Promise<void> {
  for (const sel of site.open ?? []) {
    const el = await page.waitForSelector(sel, { state: 'visible', timeout: 8000 }).catch(() => null);
    if (!el) continue;
    const clicked = await el.click({ timeout: 4000 }).then(() => true).catch(() => false);
    if (!clicked) continue;
    for (const panel of site.ready ?? []) {
      const up = await page.waitForSelector(panel, { state: 'visible', timeout: 12_000 }).catch(() => null);
      if (up) return;
    }
    return;
  }
}

async function main(): Promise<void> {
  const argv = process.argv.slice(2);
  if (argv.includes('--list')) {
    for (const s of SITES) console.log(`  ${s.id.padEnd(16)} ${s.vendor.padEnd(12)} ${s.url}`);
    return;
  }
  const picked = argv.filter((a) => !a.startsWith('-'));
  const sites = picked.length ? SITES.filter((s) => picked.includes(s.id)) : SITES;
  if (!sites.length) {
    console.error(`no such site. Known: ${SITES.map((s) => s.id).join(', ')}`);
    process.exitCode = 2;
    return;
  }

  const { launch, from, name } = await resolveLauncher();
  const rule = '─'.repeat(66);
  console.log(`\n${rule}`);
  console.log('  CaptchaKraken — live solve');
  console.log(rule);
  console.log(`  browser   : ${name}  (${displayMode()})`);
  console.log(`  from      : ${from}`);
  console.log(`  endpoint  : ${process.env.VLLM_BASE_URL ?? 'http://127.0.0.1:8000/v1'}`);
  console.log(`  model     : ${process.env.CAPTCHA_LORA_NAME ?? process.env.MODEL ?? '(client default)'}`);
  console.log(`  sites     : ${sites.length}`);
  console.log(`${rule}\n`);

  const browser = await launch(launchOptions());
  const results: Outcome[] = [];
  const solver = new CaptchaKrakenSolver();

  try {
    for (const site of sites) {
      process.stdout.write(`  ${site.vendor} — ${site.what}\n    ${site.url}\n    solving… `);
      // A fresh context per site: see the file header on why a shared one
      // would flatter the result.
      const context = await browser.newContext({ viewport: null });
      const t0 = Date.now();
      // TWO CLOCKS, BECAUSE THEY ANSWER DIFFERENT QUESTIONS. `solveMs` is what
      // the recorded medians measure and the only figure comparable to them.
      // `totalMs` also carries the page load, the settle and the demo page's
      // own reveal click — real time a visitor waits, but not the model's.
      // Printing one number for both makes the live demo look like it
      // contradicts the deck, which is the last thing it should do.
      let s0 = t0;
      let solved = false;
      let note: string | undefined;
      let result: SolveResult | void = undefined;
      try {
        const page = await context.newPage();
        await page.goto(site.url, { waitUntil: 'domcontentloaded', timeout: 60_000 });
        await page.waitForTimeout(site.settleMs ?? 2500);
        if (site.open) await reveal(page, site);
        s0 = Date.now();
        result = await solver.solve(page);
        solved = !!result && result.isSolved;
        if (!solved) note = explain(undefined, result);
      } catch (err) {
        note = explain(err);
      } finally {
        await context.close().catch(() => {});
      }
      const totalMs = Date.now() - t0;
      const solveMs = Date.now() - s0;
      const out = result ? result.tokenUsage.outputTokens : 0;
      const inp = result ? result.tokenUsage.inputTokens : 0;
      console.log(
        solved
          ? `\x1b[32m✓ solved\x1b[0m in ${fmtMs(solveMs)}  (page open to done: ${fmtMs(totalMs)} · ${inp} in / ${out} out)\n`
          : `\x1b[31m✗ ${note}\x1b[0m  after ${fmtMs(solveMs)}\n`,
      );
      results.push({ site, solved, totalMs, solveMs, tokensIn: inp, tokensOut: out, note });
    }
  } finally {
    await browser.close().catch(() => {});
  }

  const ok = results.filter((r) => r.solved);
  console.log(rule);
  console.log(`  ${'vendor'.padEnd(13)} ${'puzzle'.padEnd(30)} ${'solve'.padStart(7)} ${'total'.padStart(7)}   result`);
  console.log(rule);
  for (const r of results) {
    console.log(
      `  ${r.site.vendor.padEnd(13)} ${r.site.what.slice(0, 30).padEnd(30)} ` +
        `${fmtMs(r.solveMs).padStart(7)} ${fmtMs(r.totalMs).padStart(7)}   ` +
        `${r.solved ? '\x1b[32m✓\x1b[0m solved' : `\x1b[31m✗\x1b[0m ${brief(r.note)}`}`,
    );
  }
  console.log(rule);
  const median = ok.length
    ? [...ok.map((r) => r.solveMs)].sort((a, b) => a - b)[Math.floor(ok.length / 2)]
    : 0;
  console.log(
    `  ${ok.length}/${results.length} solved` +
      (ok.length ? `   median solve ${fmtMs(median)}` : ''),
  );
  console.log(`${rule}\n`);
  process.exitCode = ok.length === results.length ? 0 : 1;
}

main().catch((err) => {
  console.error(`\n${err instanceof Error ? err.message : err}\n`);
  process.exitCode = 1;
});
