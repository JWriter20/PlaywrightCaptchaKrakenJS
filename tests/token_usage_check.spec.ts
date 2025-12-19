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
  const rootVenv = path.resolve(__dirname, '..', '.venv', 'bin', 'python');
  if (fs.existsSync(rootVenv)) {
    PYTHON_COMMAND = rootVenv;
  } else {
    const submoduleVenv = path.join(REPO_PATH, '.venv', 'bin', 'python');
    if (fs.existsSync(submoduleVenv)) {
      PYTHON_COMMAND = submoduleVenv;
    }
  }
}
if (!PYTHON_COMMAND) {
  PYTHON_COMMAND = 'python3';
}

const MODEL = process.env.MODEL || 'gemini-2.5-flash-lite';
const API_PROVIDER = (process.env.API_PROVIDER || 'gemini') as 'ollama' | 'gemini';
const API_KEY = process.env.API_KEY || process.env.GEMINI_API_KEY;

const testWithSolver = test.extend<{ solver: CaptchaKrakenSolver }>({
  browser: [async ({ }, use) => {
    const browser = await Camoufox({ headless: false });
    await use(browser);
    await browser.close();
  }, { scope: 'worker' }],

  solver: async ({ }, use) => {
    if (!REPO_PATH || !fs.existsSync(REPO_PATH)) {
      test.skip();
    }

    const solver = new CaptchaKrakenSolver({
      repoPath: REPO_PATH!,
      pythonCommand: PYTHON_COMMAND,
      model: MODEL,
      apiProvider: API_PROVIDER,
      apiKey: API_KEY,
      maxSolveLoops: 10,
      postSolveDelayMs: 2000,
      overallSolveTimeoutMs: 120_000
    });

    await use(solver);
  }
});

testWithSolver.describe('Token Usage Reporting', () => {
  testWithSolver.slow();

  testWithSolver('Should report non-zero token usage after solving', async ({ page, solver }) => {
    // We use a page with a simple captcha that is likely to trigger at least one LLM call
    await page.goto('https://nopecha.com/captcha/recaptcha#moderate');

    let result;
    let tokenUsage;
    try {
      result = await solver.solve(page as any);
      tokenUsage = result?.tokenUsage;
    } catch (e: any) {
      console.log('Solve failed, checking if token usage is in error message.');
      const match = e.message.match(/Total usage: (\{.*\})/);
      if (match) {
        tokenUsage = JSON.parse(match[1]);
      }
    }

    console.log('Token Usage Found:', JSON.stringify(tokenUsage, null, 2));

    expect(tokenUsage).toBeDefined();
    if (tokenUsage) {
      if (API_PROVIDER === 'gemini') {
        console.log(`Model: ${tokenUsage.modelName}`);
        console.log(`Input Tokens: ${tokenUsage.inputTokens}`);
        console.log(`Output Tokens: ${tokenUsage.outputTokens}`);
        console.log(`Estimated Cost: $${tokenUsage.estimatedCost}`);

        // We expect non-zero if an LLM was actually used
        // In my previous run, I saw loop 2 and 4 and 6 and 8 and 10 used tokens
        expect(tokenUsage.inputTokens).toBeGreaterThan(0);
        expect(tokenUsage.outputTokens).toBeGreaterThan(0);
      }
    }
  });
});

