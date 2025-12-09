import { test, expect } from '@playwright/test';
import { CaptchaKrakenSolver } from '../src/solver.js';
import * as path from 'path';
import * as dotenv from 'dotenv';
import * as fs from 'fs';

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
const API_PROVIDER = (process.env.API_PROVIDER || 'gemini') as 'ollama' | 'gemini';
const API_KEY = process.env.API_KEY || process.env.GEMINI_API_KEY;

// Skip tests if REPO_PATH is not configured
const testWithSolver = test.extend<{ solver: CaptchaKrakenSolver }>({
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
      apiKey: API_KEY
    });

    await use(solver);
  }
});

testWithSolver.describe('Real World Solving Tests', () => {
  // Increase timeout for real solving (AI models can be slow)
  testWithSolver.slow();

  testWithSolver('Recaptcha (Google Demo) - Solve', async ({ page, solver }) => {
    await page.goto('https://google.com/recaptcha/api2/demo');

    // Attempt to solve
    await solver.solve(page as any);

    // If it was a challenge, we might need another solve loop or check.
    // Usually the first click opens it, then we need to solve the images.
    // The detectCaptcha inside solve() handles prioritizing the challenge if visible.

    // Check if challenge appeared
    const challengeFrame = await page.$('iframe[src*="recaptcha/api2/bframe"]');
    if (challengeFrame && await challengeFrame.isVisible()) {
      console.log('Challenge appeared, attempting to solve challenge grid...');
      // Loop a few times for multi-step challenges
      for (let i = 0; i < 5; i++) {
        await page.waitForTimeout(2000); // Wait for images to load/fade
        // Check if we are done
        const isChecked = await page.frames().find(f => f.url().includes('recaptcha/api2/anchor'))
          ?.$('.recaptcha-checkbox-checked');

        if (isChecked && await isChecked.isVisible()) {
          console.log('Recaptcha checked!');
          break;
        }

        await solver.solve(page as any);
      }
    }

    // Final Verification
    const anchorFrame = page.frames().find(f => f.url().includes('recaptcha/api2/anchor'));
    const isChecked = await anchorFrame?.locator('.recaptcha-checkbox-checked').count();
    expect(isChecked).toBeGreaterThan(0);
  });

  testWithSolver('hCaptcha (Demo) - Solve', async ({ page, solver }) => {
    await page.goto('https://democaptcha.com/demo-form-eng/hcaptcha.html');

    // 1. Click checkbox (handled by solve detection priority)
    await solver.solve(page as any);

    // 2. Wait for challenge
    try {
      await page.waitForSelector('iframe[src*="hcaptcha.com"][src*="frame=challenge"]', { timeout: 5000 });
    } catch {
      // Maybe it just checked instantly (low security env)
    }

    // 3. Loop solve
    for (let i = 0; i < 5; i++) {
      await page.waitForTimeout(2000);

      // Check success in response field
      const response = await page.$eval('[name="h-captcha-response"]', el => (el as HTMLTextAreaElement).value);
      if (response) {
        console.log('hCaptcha solved! Token:', response.substring(0, 20) + '...');
        break;
      }

      // Detect and solve
      // If challenge is closed/done, solve() might log "No supported captcha found" and return
      await solver.solve(page as any);
    }

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

