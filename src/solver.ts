import { Page, ElementHandle, Frame } from 'patchright-core';
import { generate_trajectory } from 'cursory-ts';
import { exec } from 'child_process';
import { promisify } from 'util';
import * as fs from 'fs';
import * as path from 'path';
import * as os from 'os';
import { CaptchaKrakenConfig, SolverResult, ClickAction } from './types';

const execAsync = promisify(exec);

export class CaptchaKrakenSolver {
  private config: CaptchaKrakenConfig;
  private lastMousePosition: [number, number] = [0, 0];

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

      for (const action of actionList) {
        if (action.action === 'click') {
          await this.executeClick(page, captchaElement, action, elementBox);
          // Small delay between clicks
          await new Promise(resolve => setTimeout(resolve, Math.random() * 500 + 200));
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

    const cmdParts = [
      pythonCommand,
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

    try {
      const { stdout, stderr } = await execAsync(command, {
        cwd: repoPath,
        maxBuffer: 10 * 1024 * 1024 // Increase buffer for large outputs if needed
      });

      if (!stdout.trim()) {
        // If stdout is empty, maybe check stderr or throw
        throw new Error(`CLI returned empty output. Stderr: ${stderr}`);
      }

      try {
        // Find the JSON in stdout. There might be logs.
        // Assuming the last line is the JSON or the JSON is the main output.
        // The user's CLI prints json.dumps(action_data) at the end.
        // But there might be other prints if the python code is chatty.
        // We'll try to find the last valid JSON array or object.
        const lines = stdout.trim().split('\n');
        let jsonResult: any = null;

        // Try parsing from the last line backwards
        for (let i = lines.length - 1; i >= 0; i--) {
          try {
            const parsed = JSON.parse(lines[i]);
            if (Array.isArray(parsed) || (parsed.action && (parsed.target_bounding_box || parsed.target_coordinates))) {
              jsonResult = parsed;
              break;
            }
          } catch (e) {
            // Not json
          }
        }

        if (!jsonResult) {
          // Fallback: try parsing the whole output
          jsonResult = JSON.parse(stdout);
        }

        return jsonResult as SolverResult;
      } catch (parseError) {
        throw new Error(`Failed to parse CLI output: ${stdout}\nStderr: ${stderr}`);
      }

    } catch (error: any) {
      throw new Error(`Failed to execute captcha solver CLI: ${error.message}`);
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
      // [min_x, min_y, max_x, max_y] (percentages)
      const [minX, minY, maxX, maxY] = action.target_bounding_box;

      // Calculate pixel coordinates relative to element
      const pixelMinX = minX * elementBox.width;
      const pixelMaxX = maxX * elementBox.width;
      const pixelMinY = minY * elementBox.height;
      const pixelMaxY = maxY * elementBox.height;

      // Apply padding (e.g. 10% inside the box to avoid edges)
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

    // Use cursory-ts to generate realistic mouse movement
    const [points, timings] = generate_trajectory(this.lastMousePosition, [absoluteX, absoluteY]);

    for (let i = 0; i < points.length; i++) {
      const point = points[i];
      // timing in cursory is delay in ms to wait *before* this point? or duration?
      // "generate_trajectory(start, end) -> (points, timings)"
      // Based on typical python library:
      // timings[i] corresponds to time to wait.
      // Let's assume timings[i] is the delay to wait before/after moving to point[i].
      // Usually these libraries return steps.
      const delay = timings[i];
      await page.mouse.move(point[0], point[1]);
      if (delay > 0) {
        // cursory-ts timings might be small, let's just wait.
        // In python example: time.sleep(t / 1000)
        await page.waitForTimeout(delay);
      }
    }

    // Perform click
    await page.mouse.down();
    await page.waitForTimeout(Math.random() * 50 + 20); // Random hold duration
    await page.mouse.up();

    // Update last known position
    this.lastMousePosition = [absoluteX, absoluteY];
  }
}

