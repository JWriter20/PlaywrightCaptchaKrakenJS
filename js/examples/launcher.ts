/**
 * Resolve the stealth browser the examples drive.
 *
 * HOLO FIRST, CAMOUFOX AS THE FALLBACK. Holo is the successor to the camoufox
 * fork — one repo holding the hardened Firefox and its launcher — and it is
 * what production drives. An example that demonstrates the solver on a browser
 * nobody ships is demonstrating something other than the product.
 *
 * Both are the same launcher shape: an async function returning a Playwright
 * `Browser`. That is the only thing any example asks of either, which is what
 * makes the fallback honest rather than a second code path.
 *
 * Resolution is by ENVIRONMENT rather than by a declared dependency, on
 * purpose. The client is deliberately browser-agnostic — it types Playwright
 * structurally and imports no browser at all — so making every consumer of
 * `captchakraken` install Firefox tooling to satisfy an example would be
 * backwards. The examples own the browser; the package does not.
 */

/** Where a launcher might live, in preference order. */
function candidates(): string[] {
  const out: string[] = [];
  const path = require('node:path');

  // An explicit checkout wins. It is the only entry that survives moving to
  // another machine, and it is how you drive a build that is newer than
  // anything published.
  for (const envVar of ['HOLO_TS_PATH', 'CAMOUFOX_TS_PATH']) {
    const raw = process.env[envVar]?.trim();
    if (!raw) continue;
    const p = path.isAbsolute(raw) ? raw : path.resolve(process.cwd(), raw);
    out.push(p.endsWith('.js') ? p : path.join(p, 'dist', 'index.js'));
  }
  // Then whatever is installed, newest name first.
  out.push('@jobharvest/holo', 'holo', 'camoufox', 'camoufox-js');
  return out;
}

export interface ResolvedLauncher {
  launch: (opts: any) => Promise<any>;
  /** Which candidate answered — printed so a demo can never lie about this. */
  from: string;
  name: 'Holo' | 'Camoufox';
}

export async function resolveLauncher(): Promise<ResolvedLauncher> {
  const { pathToFileURL } = require('node:url');
  const path = require('node:path');
  const tried: string[] = [];

  for (const spec of candidates()) {
    try {
      const url = spec.startsWith('.') || path.isAbsolute(spec)
        ? pathToFileURL(spec).href
        : spec;
      const mod: any = await import(url);
      const holo = mod.Holo ?? mod.default?.Holo;
      const camoufox = mod.Camoufox ?? mod.default?.Camoufox;
      if (holo) return { launch: holo, from: spec, name: 'Holo' };
      if (camoufox) return { launch: camoufox, from: spec, name: 'Camoufox' };
      tried.push(`${spec} (loaded, exports neither Holo nor Camoufox)`);
    } catch (e: any) {
      tried.push(`${spec} (${e?.code ?? e?.name ?? 'failed'})`);
    }
  }

  throw new Error(
    'Could not resolve a Holo or Camoufox launcher. Tried:\n' +
      tried.map((t) => `  - ${t}`).join('\n') +
      '\n\nPoint HOLO_TS_PATH at a built Holo checkout (run `npm run build` there),' +
      '\nor CAMOUFOX_TS_PATH at the typescript/ package of a camoufox checkout.',
  );
}

/**
 * How the browser should be displayed.
 *
 * `virtual-gpu` is a headless Xorg bound to a REAL card, and it is not the same
 * picture as true headless: headless rasterises WebGL and canvas in software,
 * so a captcha that reads the GPU sees a machine no visitor has. That shows up
 * as harder challenges, which reads as a worse model.
 */
export function displayMode(): boolean | 'virtual' | 'virtual-gpu' {
  const v = (process.env.HEADLESS ?? 'true').trim().toLowerCase();
  if (v === 'virtual' || v === 'virtual-gpu') return v;
  if (v === '0' || v === 'false') return false;
  return true;
}

/** Launch options shared by every example, so they cannot drift apart. */
export function launchOptions(): Record<string, any> {
  const executablePath =
    process.env.HOLO_EXECUTABLE_PATH ||
    process.env.CAMOUFOX_BINARY ||
    process.env.CAMOUFOX_EXECUTABLE_PATH;
  return {
    headless: displayMode(),
    // HUMANIZE defaults OFF. The solver already walks its own 60-point
    // trajectory; the launcher's humanize juggler re-humanises each of those 60
    // micro-moves, turning one straight line into 60 nested traversals — 25-52s
    // per click round instead of ~5s, which overruns the solve budget and
    // reports a solvable captcha as unsolved.
    humanize: process.env.HUMANIZE === '1',
    geoip: false,
    ...(executablePath ? { executable_path: executablePath } : {}),
  };
}
