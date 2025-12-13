# PlaywrightCaptchaKrakenJS

A Patchright (Playwright) wrapper for [CaptchaKraken-cli](https://github.com/JWriter20/CaptchaKraken-cli) to solve captchas (Recaptcha, hCaptcha, Cloudflare Turnstile) using AI vision models.

## Prerequisites

1.  **Node.js** and **npm**.
2.  **Python 3.8+** installed.

## Installation

```bash
npm install playwright-captcha-kraken-js patchright-core
```

If you're cloning this repository, initialize the git submodule:

```bash
git submodule update --init --recursive
```

Then install the Python dependencies for CaptchaKraken-cli:

```bash
cd CaptchaKraken-cli
pip install -r requirements.txt
cd ..
```

**Note:** Setup your environment variables (API keys) in `.env` if needed or pass them in config.

## Usage

```typescript
import { chromium } from 'patchright';
import { CaptchaKrakenSolver } from 'playwright-captcha-kraken-js';

(async () => {
  const browser = await chromium.launch({ headless: false });
  const page = await browser.newPage();
  
  // Configure the solver
  const solver = new CaptchaKrakenSolver({
    pythonCommand: 'python3',               // Python executable
    // Optional overrides:
    // repoPath: './CaptchaKraken-cli',     // Default: './CaptchaKraken-cli'
    // model: 'gemini-2.5-flash-lite',      // Default
    // apiProvider: 'gemini',               // Default
    // apiKey: 'YOUR_API_KEY'               // Defaults to process.env.GEMINI_API_KEY
  });

  await page.goto('https://www.google.com/recaptcha/api2/demo');

  // Attempt to solve the captcha
  // This will detect the captcha, screenshot it, call the CLI, and execute clicks.
  await solver.solve(page);

  // You might need to call solve() multiple times for multi-step captchas.
  // Example loop:
  /*
  for (let i = 0; i < 3; i++) {
    await solver.solve(page);
    await page.waitForTimeout(2000);
  }
  */

  await browser.close();
})();
```

## Configuration

| Option | Type | Default | Description |
|---|---|---|---|
| `repoPath` | `string` | `'./CaptchaKraken-cli'` | Path to the `CaptchaKraken-cli` directory. |
| `pythonCommand` | `string` | `'python'` | Python command to use. |
| `model` | `string` | `'gemini-2.5-flash-lite'` | The vision model to use. |
| `apiProvider` | `'ollama' \| 'gemini'` | `'gemini'` | The API provider. |
| `apiKey` | `string` | `process.env.GEMINI_API_KEY` | API Key (required for Gemini). |

**Note:** A special finetuned AI model for improved accuracy will be available soon.

## Testing

To run the tests:

```bash
npm test
```

### End-to-End Solving Tests

To run the real-world solving tests (which connect to your local `CaptchaKraken-cli`), you can optionally provide environment variables to override defaults:

```bash
MODEL="llama3.2-vision" \
API_PROVIDER="ollama" \
npx playwright test tests/solving.spec.ts
```

Note: The tests will automatically use `./CaptchaKraken-cli` as the default path. You can override it with `CAPTCHA_KRAKEN_REPO_PATH` if needed.

These tests will:
1.  Navigate to demo pages (Recaptcha, hCaptcha, Turnstile).
2.  Attempt to solve the captcha using your local CLI setup.
3.  Verify that the solution was accepted.
