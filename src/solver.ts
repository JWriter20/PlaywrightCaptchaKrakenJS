import { Page, ElementHandle, Frame } from 'patchright-core';
import { generate_trajectory } from 'cursory-ts';
import { exec } from 'child_process';
import { promisify } from 'util';
import * as fs from 'fs';
import * as path from 'path';
import * as os from 'os';
import { CaptchaKrakenConfig, SolverResult, ClickAction, CaptchaAction } from './types';

const execAsync = promisify(exec);

function getBundledCliRoot(): string {
  // When installed from npm, this file is in `<pkgRoot>/dist` (compiled) or `<pkgRoot>/src` (dev).
  // The bundled python project sits at `<pkgRoot>/CaptchaKraken-cli`.
  return path.resolve(__dirname, '..', 'CaptchaKraken-cli');
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

// Simple Vector interface for internal use
interface Vector {
  x: number;
  y: number;
}

interface TimedVector {
  x: number;
  y: number;
  timestamp?: number;
}

const log = (message: string, ...args: any[]) => console.log(`[Solver] ${message}`, ...args);
const delay = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

export class CaptchaKrakenSolver {
  private config: CaptchaKrakenConfig;
  private lastMousePosition: Vector = { x: 100, y: 100 }; // Start at safe position

  constructor(config: CaptchaKrakenConfig) {
    this.config = config;
  }

  async solve(page: Page): Promise<void> {
    // 1. Detect Captcha
    const captchaElement = await this.detectCaptcha(page);
    if (!captchaElement) {
      console.log('No supported captcha found.');
      return;
    }

    // 2. Take Screenshot
    const screenshotPath = path.join(os.tmpdir(), `captcha_${Date.now()}.png`);
    await captchaElement.screenshot({ path: screenshotPath });

    try {
      // 3. Call CLI
      const actions = await this.getSolution(screenshotPath);

      // 4. Execute Actions
      const actionList = Array.isArray(actions) ? actions : [actions];

      // We need the element's bounding box to translate coordinates
      const elementBox = await captchaElement.boundingBox();
      if (!elementBox) {
        throw new Error('Could not get bounding box of captcha element');
      }

      console.log(`Executing ${actionList.length} actions.`);

      let performedAction = false;
      for (const action of actionList) {
        if (action.action === 'click') {
          await this.executeClick(page, captchaElement, action as ClickAction, elementBox);
          // Small delay between clicks
          await delay(Math.random() * 500 + 200);
          performedAction = true;
        } else if (action.action === 'wait') {
          if (action.duration_ms > 0) {
            console.log(`Waiting for ${action.duration_ms}ms as requested by CLI`);
            await delay(action.duration_ms);
            performedAction = true;
          }
        }
      }

      if (!performedAction) {
        console.log('No active actions performed (empty or done). Checking for Verify/Next button...');
        const contentFrame = await captchaElement.contentFrame();
        if (contentFrame) {
          // 1. Try generic button selectors by text
          const buttonTexts = ['Verify', 'Next', 'Submit', 'Skip'];
          for (const text of buttonTexts) {
            try {
              // Using XPath to find buttons or inputs with specific text/value
              // Case insensitive contains for text, or value attribute
              const btn = await contentFrame.$(`xpath=//button[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '${text.toLowerCase()}')] | //div[@role="button" and contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '${text.toLowerCase()}')]`);
              if (btn && await btn.isVisible()) {
                console.log(`Clicking button with text "${text}"`);
                await this.moveAndClick(page, btn);
                return;
              }
            } catch (e) {
              // Ignore locator errors
            }
          }

          // 2. Try specific ID (Recaptcha)
          const recaptchaVerify = await contentFrame.$('#recaptcha-verify-button');
          if (recaptchaVerify && await recaptchaVerify.isVisible()) {
            console.log('Clicking Recaptcha Verify/Next button by ID');
            await this.moveAndClick(page, recaptchaVerify);
            return;
          }

          // 3. Try specific class (hCaptcha)
          const hcaptchaVerify = await contentFrame.$('.button-submit');
          if (hcaptchaVerify && await hcaptchaVerify.isVisible()) {
            console.log('Clicking hCaptcha Verify/Submit button by Class');
            await this.moveAndClick(page, hcaptchaVerify);
            return;
          }
        }
      }
    } finally {
      // Cleanup
      if (fs.existsSync(screenshotPath)) {
        fs.unlinkSync(screenshotPath);
      }
    }
  }

  public async detectCaptcha(page: Page): Promise<ElementHandle | null> {
    // Prioritize open challenges (the grid/images) over the initial checkbox

    // Recaptcha Challenge
    const recaptchaChallenge = await page.$('iframe[src*="recaptcha/api2/bframe"]');
    if (recaptchaChallenge && await recaptchaChallenge.isVisible()) return recaptchaChallenge;

    // hCaptcha Challenge
    // Try matching by src (frame=challenge)
    const hcaptchaChallenge = await page.$('iframe[src*="hcaptcha.com"][src*="frame=challenge"]');
    if (hcaptchaChallenge && await hcaptchaChallenge.isVisible()) return hcaptchaChallenge;

    // Fallback: title containing "content" or "challenge" (sometimes title varies)
    const hcaptchaChallengeTitle = await page.$('iframe[src*="hcaptcha.com"][title*="challenge"]');
    if (hcaptchaChallengeTitle && await hcaptchaChallengeTitle.isVisible()) return hcaptchaChallengeTitle;

    // Recaptcha Checkbox
    const recaptchaCheckbox = await page.$('iframe[src*="recaptcha/api2/anchor"]');
    if (recaptchaCheckbox && await recaptchaCheckbox.isVisible()) return recaptchaCheckbox;

    // hCaptcha Checkbox
    const hcaptchaCheckbox = await page.$('iframe[src*="hcaptcha.com"]:not([title*="challenge"])');
    if (hcaptchaCheckbox && await hcaptchaCheckbox.isVisible()) return hcaptchaCheckbox;

    // Cloudflare Turnstile
    // Try iframe first (if visible/open)
    const cloudflareIframe = await page.$('iframe[src*="challenges.cloudflare.com"]');
    if (cloudflareIframe && await cloudflareIframe.isVisible()) return cloudflareIframe;

    // Fallback to container for closed shadow roots
    const cloudflareContainer = await page.$('.cf-turnstile');
    if (cloudflareContainer && await cloudflareContainer.isVisible()) return cloudflareContainer;

    return null;
  }

  private async getSolution(imagePath: string): Promise<SolverResult> {
    const {
      repoPath,
      pythonCommand = 'python',
      model = 'gemini-2.5-flash-lite',
      apiProvider = 'gemini',
      apiKey = process.env.GEMINI_API_KEY
    } = this.config;

    const cliRoot = repoPath ?? getBundledCliRoot();
    if (!fs.existsSync(cliRoot)) {
      throw new Error(
        `CaptchaKraken CLI folder not found at ${cliRoot}. ` +
        `If you installed from npm, ensure the package ships 'CaptchaKraken-cli/'.`
      );
    }

    // Prefer the packaged venv python if present (postinstall bootstrap), otherwise fall back.
    const venvPython = getVenvPython(cliRoot);
    const py = venvPython ?? pythonCommand;

    const cmdParts = [
      py,
      '-m',
      'src.cli',
      `"${imagePath}"`,
      model,
      apiProvider
    ];

    if (apiKey) {
      cmdParts.push(apiKey);
    }

    const command = cmdParts.join(' ');
    console.log(`Executing CaptchaKraken CLI: ${command}`);

    try {
      const { stdout, stderr } = await execAsync(command, {
        cwd: cliRoot,
        env: process.env,
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
        const allActions: any[] = [];
        let foundAny = false;

        for (const line of lines) {
          try {
            const parsed = JSON.parse(line);
            if (Array.isArray(parsed)) {
              allActions.push(...parsed);
              foundAny = true;
            } else if (parsed.action && (parsed.target_bounding_box || parsed.target_coordinates || parsed.action === 'wait')) {
              allActions.push(parsed);
              foundAny = true;
            }
          } catch (e) {
            // Not json or not relevant
          }
        }

        if (!foundAny) {
          try {
            const parsed = JSON.parse(stdout);
            if (Array.isArray(parsed)) {
              allActions.push(...parsed);
            } else if (parsed.action && (parsed.target_bounding_box || parsed.target_coordinates)) {
              allActions.push(parsed);
            }
          } catch (e) {
            // Ignore
          }
        }

        console.log('CaptchaKraken parsed result:', JSON.stringify(allActions, null, 2));

        return allActions as SolverResult;
      } catch (parseError) {
        throw new Error(`Failed to parse CLI output: ${stdout}\nStderr: ${stderr}`);
      }

    } catch (error: any) {
      console.error('Error executing CaptchaKraken CLI:', error);
      if (error.stdout) console.log('CLI stdout on error:', error.stdout);
      if (error.stderr) console.error('CLI stderr on error:', error.stderr);
      throw new Error(`Failed to execute captcha solver CLI: ${error.message}`);
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

    await elem.scrollIntoViewIfNeeded();

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
    await page.mouse.down();
    await delay(Math.random() * 50 + 20);
    await page.mouse.up();
  }

  private async performSmoothMove(page: Page, x: number, y: number) {
    // Generate trajectory using cursory-ts with 60Hz frequency for better control
    const [points, timings] = generate_trajectory(
      [this.lastMousePosition.x, this.lastMousePosition.y],
      [x, y],
      60 // 60 points per second
    );

    const SPEED_MULTIPLIER = 1;

    const vectors: TimedVector[] = [];

    for (let i = 0; i < points.length; i++) {
      vectors.push({
        x: points[i][0],
        y: points[i][1],
        timestamp: timings[i] / SPEED_MULTIPLIER // timings are cumulative from start
      });
    }

    await this.tracePath(page, vectors);
  }

  private async tracePath(page: Page, vectors: TimedVector[]) {
    // Get viewport for clamping
    let viewport: { width: number, height: number } = { width: 1920, height: 1080 };
    try {
      const vp = page.viewportSize();
      if (vp) viewport = vp;
    } catch (e) { }

    const startTime = Date.now();

    for (let i = 0; i < vectors.length; i++) {
      const v = vectors[i];

      try {
        // Clamp coordinates to viewport
        const clampedX = Math.max(0, Math.min(v.x, viewport.width));
        const clampedY = Math.max(0, Math.min(v.y, viewport.height));

        // Move mouse
        await page.mouse.move(clampedX, clampedY);

        // Update last position
        this.lastMousePosition = { x: clampedX, y: clampedY };

        // Calculate delay to match target timestamp
        if (v.timestamp !== undefined) {
          const targetTime = startTime + v.timestamp;
          const now = Date.now();
          const delayMs = targetTime - now;

          if (delayMs > 0) {
            await delay(delayMs);
          }
        }
      } catch (error) {
        // Check if page closed or other fatal errors if needed, otherwise ignore
        const errorMessage = error instanceof Error ? error.message : String(error);
        if (errorMessage.includes('Target closed') || errorMessage.includes('Session closed')) {
          log('Warning: could not move mouse, page or session closed.');
          return;
        }
      }
    }
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

    const absoluteX = elementBox.x + relativeX;
    const absoluteY = elementBox.y + relativeY;

    // Use the shared smooth move method
    await this.performSmoothMove(page, absoluteX, absoluteY);

    // Perform click
    await page.mouse.down();
    await page.waitForTimeout(Math.random() * 30 + 20); // Random hold duration
    await page.mouse.up();
  }
}
