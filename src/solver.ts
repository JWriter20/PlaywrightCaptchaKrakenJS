import { Page, ElementHandle, Frame } from 'patchright-core';
import { generate_trajectory } from 'cursory-ts';
import { exec } from 'child_process';
import { promisify } from 'util';
import * as fs from 'fs';
import * as path from 'path';
import * as os from 'os';
import { CaptchaKrakenConfig, SolverResult, ClickAction, CaptchaAction, SolveResult, CliResponse, TokenUsage, Vector } from './types';
import { aggregateTokenUsage } from './token-usage';

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

// Simple Vector interface for internal use moved to types.ts

interface TimedVector {
  x: number;
  y: number;
  timestamp?: number;
}

const log = (message: string, ...args: any[]) => console.log(`[Solver] ${message}`, ...args);
const delay = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

export class CaptchaKrakenSolver {
  private config: CaptchaKrakenConfig;
  private lastMousePosition: Vector; // Start at safe position
  private imageCounter: number = 0; // Track images sent to CLI for debugging
  private sessionDebugDir: string | null = null;

  constructor(config: CaptchaKrakenConfig) {
    this.config = config;
    this.lastMousePosition = config.startingMousePosition ?? { x: 100, y: 100 };
  }

  async solve(page: Page): Promise<SolveResult | void> {
    const maxSolveLoops = this.config.maxSolveLoops ?? 10;
    const postSolveDelayMs = this.config.postSolveDelayMs ?? 1200;
    const overallSolveTimeoutMs = this.config.overallSolveTimeoutMs ?? 120_000;

    const start = Date.now();
    let cumulativeTokenUsage: TokenUsage[] = [];
    this.imageCounter = 0;

    // Initialize session debug directory if debugging is enabled
    if (process.env.CAPTCHA_DEBUG === '1') {
      const cliRoot = this.config.repoPath ?? getBundledCliRoot();
      const debugRunsDir = path.join(cliRoot, '..', 'debug_runs');
      if (!fs.existsSync(debugRunsDir)) {
        fs.mkdirSync(debugRunsDir, { recursive: true });
      }
      const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, -5);
      this.sessionDebugDir = path.join(debugRunsDir, `solve_${timestamp}`);
      fs.mkdirSync(this.sessionDebugDir, { recursive: true });
      log(`Session debug directory: ${this.sessionDebugDir}`);
    }

    for (let attempt = 1; attempt <= maxSolveLoops; attempt++) {
      if (Date.now() - start > overallSolveTimeoutMs) {
        throw new Error(`Captcha solve timed out after ${overallSolveTimeoutMs}ms (attempt ${attempt}/${maxSolveLoops}).`);
      }

      const captchaElement = await this.detectCaptcha(page);
      if (!captchaElement) {
        console.log('No supported captcha found.');
        return {
          isSolved: true,
          finalMousePosition: this.lastMousePosition,
          tokenUsage: aggregateTokenUsage(cumulativeTokenUsage)
        };
      }

      console.log(`\n--- Captcha Solve Loop ${attempt}/${maxSolveLoops} ---`);
      const { didInteract, tokenUsage } = await this.solveSingle(page, captchaElement, attempt);
      cumulativeTokenUsage.push(...tokenUsage);

      // Let the page update (challenge frame open, images refresh, verification, etc.)
      await delay(postSolveDelayMs + Math.random() * 300);

      const after = await this.detectCaptcha(page);
      if (!after) {
        return {
          isSolved: true,
          finalMousePosition: this.lastMousePosition,
          tokenUsage: aggregateTokenUsage(cumulativeTokenUsage)
        };
      }

      // If we didn't actually interact and captcha is still detected, don't spin forever.
      if (!didInteract) {
        throw new Error(`Captcha still detected but solver performed no interactions; aborting to avoid an infinite loop. Total usage: ${JSON.stringify(aggregateTokenUsage(cumulativeTokenUsage))}`);
      }
    }

    throw new Error(`Captcha still detected after ${maxSolveLoops} solve loops. Total usage: ${JSON.stringify(aggregateTokenUsage(cumulativeTokenUsage))}`);
  }

  private async solveSingle(page: Page, captchaElement: ElementHandle, attempt: number): Promise<{ didInteract: boolean, tokenUsage: TokenUsage[] }> {
    // 1. Take initial Screenshot to detect movement
    const initialScreenshotPath = path.join(os.tmpdir(), `captcha_init_${Date.now()}.png`);
    await captchaElement.screenshot({ path: initialScreenshotPath });

    let inputPath = initialScreenshotPath;
    let isVideo = false;

    // Check for movement
    const hasMovement = await this.checkMovement(captchaElement, initialScreenshotPath);
    if (hasMovement) {
      console.log('Movement detected in captcha, capturing video...');
      try {
        const videoPath = await this.captureVideo(captchaElement);
        inputPath = videoPath;
        isVideo = true;
      } catch (e) {
        console.warn('Failed to capture video, falling back to image:', e);
      }
    }

    // Save input to debug directory if debugging is enabled
    this.saveImageForDebug(inputPath);

    let performedAction = false;
    let allTokenUsage: TokenUsage[] = [];

    try {
      // 2. Call CLI with either image or video
      const response = await this.getSolution(inputPath);
      const actions = response.actions;
      allTokenUsage = response.token_usage;

      // Archive debug artifacts if enabled
      this.archiveLatestDebugRun(attempt, actions);

      // 3. Execute Actions
      const actionList = Array.isArray(actions) ? actions : [actions];

      // We need the element's bounding box to translate coordinates
      const elementBox = await captchaElement.boundingBox();
      if (!elementBox) {
        throw new Error('Could not get bounding box of captcha element');
      }

      console.log(`Executing ${actionList.length} actions.`);

      for (const action of actionList) {
        if (action.action === 'click') {
          await this.executeClick(page, captchaElement, action as ClickAction, elementBox);
          // Small delay between clicks
          await delay(Math.random() * 20 + 30);
          performedAction = true;
        } else if (action.action === 'wait') {
          if ((action as any).duration_ms > 0) {
            console.log(`Waiting for ${(action as any).duration_ms}ms as requested by CLI`);
            await delay((action as any).duration_ms);
            performedAction = true;
          }
        }
      }

      if (!performedAction) {
        // ... existing button clicking code ...
        console.log('No active actions performed (empty or done). Checking for Verify/Next button...');
        const contentFrame = await captchaElement.contentFrame();
        if (contentFrame) {
          // 1. Try generic button selectors by text
          const buttonTexts = ['Verify', 'Next', 'Submit', 'Skip'];
          for (const text of buttonTexts) {
            try {
              // Case-insensitive contains for text
              const btn = await contentFrame.$(
                `xpath=//button[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '${text.toLowerCase()}')] | //div[@role="button" and contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '${text.toLowerCase()}')]`
              );
              if (btn && await btn.isVisible()) {
                console.log(`Clicking button with text "${text}"`);
                await this.moveAndClick(page, btn);
                performedAction = true;
                break;
              }
            } catch (e) {
              // Ignore locator errors
            }
          }

          if (!performedAction) {
            // 2. Try specific ID (Recaptcha)
            const recaptchaVerify = await contentFrame.$('#recaptcha-verify-button');
            if (recaptchaVerify && await recaptchaVerify.isVisible()) {
              console.log('Clicking Recaptcha Verify/Next button by ID');
              await this.moveAndClick(page, recaptchaVerify);
              performedAction = true;
            }
          }

          if (!performedAction) {
            // 3. Try specific class (hCaptcha)
            const hcaptchaVerify = await contentFrame.$('.button-submit');
            if (hcaptchaVerify && await hcaptchaVerify.isVisible()) {
              console.log('Clicking hCaptcha Verify/Submit button by Class');
              await this.moveAndClick(page, hcaptchaVerify);
              performedAction = true;
            }
          }
        }
      }
    } finally {
      // Cleanup
      if (fs.existsSync(initialScreenshotPath)) {
        fs.unlinkSync(initialScreenshotPath);
      }
      if (inputPath !== initialScreenshotPath && fs.existsSync(inputPath)) {
        fs.unlinkSync(inputPath);
      }
    }

    return { didInteract: performedAction, tokenUsage: allTokenUsage };
  }

  private async checkMovement(captchaElement: ElementHandle, firstScreenshotPath: string): Promise<boolean> {
    const secondScreenshotPath = path.join(os.tmpdir(), `captcha_check_${Date.now()}.png`);
    await delay(150); // Wait a bit for movement
    try {
      await captchaElement.screenshot({ path: secondScreenshotPath });
    } catch (e) {
      return false; // Element might have disappeared
    }

    try {
      const cliRoot = this.config.repoPath ?? getBundledCliRoot();
      const venvPython = getVenvPython(cliRoot);
      const py = venvPython ?? (this.config.pythonCommand || 'python');

      const command = `${py} -m src.cli check-movement "${firstScreenshotPath}" "${secondScreenshotPath}"`;
      const { stdout } = await execAsync(command, { cwd: cliRoot });
      const result = JSON.parse(stdout);
      return !!result.has_movement;
    } catch (e) {
      console.warn('Movement check failed, assuming static:', e);
      return false;
    } finally {
      if (fs.existsSync(secondScreenshotPath)) fs.unlinkSync(secondScreenshotPath);
    }
  }

  private async captureVideo(captchaElement: ElementHandle): Promise<string> {
    const framesDir = path.join(os.tmpdir(), `captcha_frames_${Date.now()}`);
    fs.mkdirSync(framesDir, { recursive: true });

    const videoPath = path.join(os.tmpdir(), `captcha_video_${Date.now()}.webm`);

    try {
      // Capture 4 frames
      for (let i = 0; i < 4; i++) {
        await captchaElement.screenshot({ path: path.join(framesDir, `frame_${i}.png`) });
        if (i < 3) await delay(100);
      }

      // Use ffmpeg to create a webm video
      // -framerate 5: 5 frames per second
      // -i frame_%d.png: input pattern
      // -c:v libvpx-vp9: codec
      // -crf 35: quality (reasonable for small files)
      // -b:v 0: required for CRF in vp9
      // -vf scale: downscale for speed if needed
      const command = `ffmpeg -y -framerate 5 -i "${path.join(framesDir, 'frame_%d.png')}" -c:v libvpx-vp9 -crf 35 -b:v 0 -vf "scale='min(400,iw)':-1" "${videoPath}"`;
      await execAsync(command);

      return videoPath;
    } catch (e) {
      console.error('ffmpeg failed:', e);
      throw e;
    } finally {
      // Cleanup frames
      if (fs.existsSync(framesDir)) {
        fs.rmSync(framesDir, { recursive: true, force: true });
      }
    }
  }

  private async hasNonEmptyFieldValue(page: Page, selector: string): Promise<boolean> {
    try {
      const el = await page.$(selector);
      if (!el) return false;
      const value = await page.$eval(selector, node => {
        const anyNode = node as any;
        return typeof anyNode.value === 'string' ? anyNode.value : '';
      });
      return typeof value === 'string' && value.trim().length > 0;
    } catch {
      return false;
    }
  }

  private async isRecaptchaAnchorChecked(anchorIframe: ElementHandle): Promise<boolean> {
    try {
      const frame = await anchorIframe.contentFrame();
      if (!frame) return false;
      const checked = await frame.$('.recaptcha-checkbox-checked');
      return !!(checked && await checked.isVisible());
    } catch {
      return false;
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
    if (recaptchaCheckbox && await recaptchaCheckbox.isVisible()) {
      // If it's already checked, consider it solved and continue searching.
      const checked = await this.isRecaptchaAnchorChecked(recaptchaCheckbox);
      if (!checked) return recaptchaCheckbox;
    }

    // hCaptcha Checkbox
    const hcaptchaCheckbox = await page.$('iframe[src*="hcaptcha.com"]:not([title*="challenge"])');
    if (hcaptchaCheckbox && await hcaptchaCheckbox.isVisible()) {
      // If we already have a token, treat as solved and continue searching.
      const hasToken = await this.hasNonEmptyFieldValue(page, '[name="h-captcha-response"]');
      if (!hasToken) return hcaptchaCheckbox;
    }

    // Cloudflare Turnstile
    // Try iframe first (if visible/open)
    const cloudflareIframe = await page.$('iframe[src*="challenges.cloudflare.com"]');
    if (cloudflareIframe && await cloudflareIframe.isVisible()) {
      const hasToken = await this.hasNonEmptyFieldValue(page, '[name="cf-turnstile-response"]');
      if (!hasToken) return cloudflareIframe;
    }

    // Fallback to container for closed shadow roots
    const cloudflareContainer = await page.$('.cf-turnstile');
    if (cloudflareContainer && await cloudflareContainer.isVisible()) {
      const hasToken = await this.hasNonEmptyFieldValue(page, '[name="cf-turnstile-response"]');
      if (!hasToken) return cloudflareContainer;
    }

    return null;
  }

  private saveImageForDebug(imagePath: string): void {
    // Check if CAPTCHA_DEBUG is enabled
    const debugEnabled = process.env.CAPTCHA_DEBUG === '1';
    if (!debugEnabled) {
      return;
    }

    try {
      const cliRoot = this.config.repoPath ?? getBundledCliRoot();
      // Save input images to a separate directory that won't be cleared by the Python CLI
      // The Python CLI clears latestDebugRun, so we use a sibling directory
      const inputImagesDir = path.join(cliRoot, 'latestDebugRun_inputs');

      // Ensure input images directory exists
      if (!fs.existsSync(inputImagesDir)) {
        fs.mkdirSync(inputImagesDir, { recursive: true });
      }

      // Increment counter and save with a descriptive name
      this.imageCounter++;
      const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, -5);
      const extension = path.extname(imagePath) || '.png';
      const debugImageName = `input_${String(this.imageCounter).padStart(3, '0')}_${timestamp}${extension}`;
      const debugImagePath = path.join(inputImagesDir, debugImageName);

      // Copy the image to debug directory
      fs.copyFileSync(imagePath, debugImagePath);
      console.log(`[DEBUG] Saved input image to: ${debugImagePath}`);
    } catch (error) {
      // Don't fail the solve if debug save fails
      console.warn(`[DEBUG] Failed to save image for debugging: ${error}`);
    }
  }

  private archiveLatestDebugRun(attempt: number, actions: SolverResult): void {
    if (!this.sessionDebugDir) return;

    try {
      const cliRoot = this.config.repoPath ?? getBundledCliRoot();
      const latestDebugDir = path.join(cliRoot, 'latestDebugRun');
      const inputImagesDir = path.join(cliRoot, 'latestDebugRun_inputs');

      const attemptDir = path.join(this.sessionDebugDir, `attempt_${attempt}`);
      fs.mkdirSync(attemptDir, { recursive: true });

      // Archive CLI artifacts if they exist
      if (fs.existsSync(latestDebugDir)) {
        fs.cpSync(latestDebugDir, attemptDir, { recursive: true });
        fs.rmSync(latestDebugDir, { recursive: true, force: true });
      }

      // Archive input images if they exist
      if (fs.existsSync(inputImagesDir)) {
        const archivedInputsDir = path.join(attemptDir, 'inputs');
        fs.mkdirSync(archivedInputsDir, { recursive: true });
        fs.cpSync(inputImagesDir, archivedInputsDir, { recursive: true });
        fs.rmSync(inputImagesDir, { recursive: true, force: true });
      }

      // Add actions info to the attempt directory
      fs.writeFileSync(
        path.join(attemptDir, 'actions_result.json'),
        JSON.stringify(actions, null, 2)
      );

      console.log(`[DEBUG] Archived attempt ${attempt} debug artifacts to: ${attemptDir}`);
    } catch (error) {
      console.warn(`[DEBUG] Failed to archive debug artifacts: ${error}`);
    }
  }

  private async getSolution(imagePath: string): Promise<CliResponse> {
    const {
      repoPath,
      pythonCommand = 'python',
      apiProvider = 'gemini',
      model = apiProvider === 'openrouter' ? 'google/gemini-2.0-flash-lite-preview-02-05:free' : 'gemini-2.5-flash-lite',
      apiKey = apiProvider === 'openrouter' ? process.env.OPENROUTER_KEY : (apiProvider === 'gemini' ? process.env.GEMINI_API_KEY : undefined)
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
        let actions: SolverResult = [];
        let tokenUsage: TokenUsage[] = [];

        for (const line of lines) {
          try {
            const parsed = JSON.parse(line);

            // Handle new format { actions: ..., token_usage: ... }
            if (parsed.actions !== undefined && parsed.token_usage !== undefined) {
              actions = parsed.actions;
              tokenUsage = parsed.token_usage;
              break;
            }

            // Fallback for old format or list of actions
            if (Array.isArray(parsed)) {
              actions = parsed;
            } else if (parsed.action && (parsed.target_bounding_box || parsed.target_coordinates || parsed.action === 'wait')) {
              actions = [parsed];
            }
          } catch (e) {
            // Not json or not relevant
          }
        }

        return { actions, token_usage: tokenUsage };
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
    await delay(Math.random() * 20 + 20);
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
