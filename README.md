# PlaywrightCaptchaKrakenJS

A Patchright (Playwright) wrapper for [CaptchaKraken-cli](https://github.com/JWriter20/CaptchaKraken-cli) to solve captchas (Recaptcha, hCaptcha, Cloudflare Turnstile) using AI vision models.

## Current Capabilities

Right now, we can reliably solve:
- **Checkbox captchas**: ~100% success rate
- **Image captchas**: ~60% success rate (work in progress with finetuning vision models to improve this)

Other kinds of captchas have not really been tested. Development will primarily focus on reCAPTCHA, Cloudflare Turnstile, and hCaptcha.

## Prerequisites

1.  **Node.js** and **npm**.
2.  **Python 3.10+** installed.

## Installation

```bash
npm install playwright-captcha-kraken-js patchright-core
```

If you're cloning this repository, initialize the git submodule:

```bash
git submodule update --init --recursive
```

On install, this package will automatically create a local venv at `CaptchaKraken-cli/.venv` and install
Python dependencies via an `npm postinstall` hook.

- **Skip python setup**: set `CAPTCHA_KRAKEN_SKIP_PYTHON_SETUP=1`
- **Use a specific python**: set `CAPTCHA_KRAKEN_PYTHON=/path/to/python3`

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
    // Optional overrides:
    // repoPath: '/absolute/path/to/CaptchaKraken-cli', // Usually not needed (auto-resolved from npm package)
    // model: 'gemini-2.5-flash-lite',      // Default
    // apiProvider: 'gemini',               // Default
    // apiKey: 'YOUR_API_KEY',              // Defaults to process.env.GEMINI_API_KEY
    // pythonCommand: 'python3',            // Usually not needed (auto-uses venv python)
  });

  await page.goto('https://www.google.com/recaptcha/api2/demo');

  // Attempt to solve the captcha
  // This will detect the captcha, screenshot it, call the CLI, and execute clicks.
  // It will also automatically re-check for newly opened next-step challenges
  // (e.g., checkbox -> image grid) and keep solving until solved (or loop limit).
  await solver.solve(page);

  await browser.close();
})();
```

## Configuration

| Option | Type | Default | Description |
|---|---|---|---|
| `repoPath` | `string` | *(auto)* | Path to the bundled `CaptchaKraken-cli` directory (usually not needed). |
| `pythonCommand` | `string` | *(auto)* | Python command to use. Usually not needed - automatically uses the venv python created during installation. |
| `model` | `string` | `'gemini-2.5-flash-lite'` | The vision model to use. |
| `apiProvider` | `'ollama' \| 'gemini'` | `'gemini'` | The API provider. |
| `apiKey` | `string` | `process.env.GEMINI_API_KEY` | API Key (required for Gemini). |
| `maxSolveLoops` | `number` | `10` | Max number of detect→solve iterations in a single `solve()` call. |
| `postSolveDelayMs` | `number` | `1200` | Delay after each iteration before re-detecting. |
| `overallSolveTimeoutMs` | `number` | `120000` | Overall time limit for the whole `solve()` call. |

## Coming Soon

We're actively working on several improvements to enhance the solver's capabilities:

- **OpenRouter integration**: Support for OpenRouter API, enabling you to use any vision model available on their platform (GPT-4 Vision, Claude, etc.)
- **Stronger default model**: Replacing `gemini-2.5-flash-lite` with a more capable default model that better handles complex image recognition tasks
- **Improved reCAPTCHA accuracy**: Finetuned custom model to boost reCAPTCHA image captcha success rate from ~60% to 95%+
- **hCaptcha complex drag puzzle solving**: Full support for hCaptcha's drag-and-drop puzzle challenges

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
