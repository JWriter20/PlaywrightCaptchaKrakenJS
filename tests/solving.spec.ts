import { test, expect } from '@playwright/test';
import { CaptchaKrakenSolver } from '../src/solver.js';
import * as path from 'path';
import * as dotenv from 'dotenv';
import * as fs from 'fs';
import { Camoufox } from 'camoufox-js';

// Load environment variables
dotenv.config();

// Determine REPO_PATH with fallback to submodule
let REPO_PATH = process.env.CAPTCHA_KRAKEN_REPO_PATH;
if (!REPO_PATH) {
  const submodulePath = path.resolve(__dirname, '..', 'CaptchaKraken-cli');
  if (fs.existsSync(submodulePath) && fs.existsSync(path.join(submodulePath, 'src', 'cli.py'))) {
    REPO_PATH = submodulePath;
  }
}

// Determine PYTHON_COMMAND with fallback to local venv
let PYTHON_COMMAND = process.env.PYTHON_COMMAND;
if (!PYTHON_COMMAND && REPO_PATH) {
  // Check for venv in root
  const rootVenv = path.resolve(__dirname, '..', '.venv', 'bin', 'python');
  if (fs.existsSync(rootVenv)) {
    PYTHON_COMMAND = rootVenv;
  } else {
    // Check for venv in submodule
    const submoduleVenv = path.join(REPO_PATH, '.venv', 'bin', 'python');
    if (fs.existsSync(submoduleVenv)) {
      PYTHON_COMMAND = submoduleVenv;
    }
  }
}
// Final fallback to system python
if (!PYTHON_COMMAND) {
  PYTHON_COMMAND = 'python3';
}

const MODEL = process.env.MODEL || 'gemini-2.5-flash-lite';
const API_PROVIDER = (process.env.API_PROVIDER || 'gemini') as 'ollama' | 'gemini' | 'openrouter';
const API_KEY = process.env.API_KEY || (API_PROVIDER === 'openrouter' ? process.env.OPENROUTER_KEY : process.env.GEMINI_API_KEY);

// Skip tests if REPO_PATH is not configured
const testWithSolver = test.extend<{ solver: CaptchaKrakenSolver }>({
  browser: [async ({ }, use) => {
    const browser = await Camoufox({ headless: false });
    await use(browser);
    await browser.close();
  }, { scope: 'worker' }],

  solver: async ({ }, use) => {
    if (!REPO_PATH || !fs.existsSync(REPO_PATH)) {
      console.warn('Skipping solving tests: CAPTCHA_KRAKEN_REPO_PATH not set or invalid.');
      test.skip();
    }

    const solver = new CaptchaKrakenSolver({
      repoPath: REPO_PATH!,
      pythonCommand: PYTHON_COMMAND,
      model: MODEL,
      apiProvider: API_PROVIDER,
      apiKey: API_KEY,
      // Multi-step captchas often require several iterations (checkbox -> challenge -> verify).
      // The solver now loops internally; keep these generous for real-world tests.
      maxSolveLoops: 15,
      postSolveDelayMs: 2000,
      overallSolveTimeoutMs: 180_000
    });

    await use(solver);
  }
});

testWithSolver.describe('Real World Solving Tests', () => {
  // Increase timeout for real solving (AI models can be slow)
  testWithSolver.slow();

  testWithSolver('Recaptcha (Google Demo) - Solve', async ({ page, solver }) => {
    await page.goto('https://nopecha.com/captcha/recaptcha#moderate');

    // Attempt to solve (internal loop handles checkbox->challenge->verify)
    await solver.solve(page as any);

    // Final Verification
    const anchorFrame = page.frames().find(f => f.url().includes('recaptcha/api2/anchor'));
    const isChecked = await anchorFrame?.locator('.recaptcha-checkbox-checked').count();
    expect(isChecked).toBeGreaterThan(0);
  });

  testWithSolver('hCaptcha (Demo) - Solve', async ({ page, solver }) => {
    await page.goto('https://democaptcha.com/demo-form-eng/hcaptcha.html');

    // Attempt to solve (internal loop handles checkbox->challenge->verify)
    await solver.solve(page as any);

    const response = await page.$eval('[name="h-captcha-response"]', el => (el as HTMLTextAreaElement).value);
    expect(response).toBeTruthy();
  });

  testWithSolver('Cloudflare Turnstile (2Captcha Demo) - Solve', async ({ page, solver }) => {
    await page.goto('https://2captcha.com/demo/cloudflare-turnstile');

    // Turnstile often solves automatically, but sometimes requires a click.
    // solver.solve will find the widget and click if needed (if it identifies a click target).
    // Note: Cloudflare usually doesn't have "click targets" inside the widget in the same way,
    // it's just one big button or auto.
    // The CLI needs to support identifying the "Verify you are human" checkbox area.

    await solver.solve(page as any);

    // Wait for success
    // On this demo page, the success is often indicated by the token input being filled
    // or the widget state changing.

    // Poll for token
    await expect(async () => {
      const val = await page.$eval('[name="cf-turnstile-response"]', el => (el as HTMLInputElement).value);
      expect(val).not.toContain('DUMMY'); // The initial value is DUMMY TOKEN in the example HTML sometimes?
      // Actually the example in prompt had "XXXX.DUMMY.TOKEN.XXXX".
      expect(val).not.toContain('DUMMY.TOKEN');
      expect(val).toBeTruthy();
    }).toPass({ timeout: 15000 });
  });
});

