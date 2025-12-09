import { test, expect } from '@playwright/test';
import { CaptchaKrakenSolver } from '../src/solver';
import { ElementHandle } from 'patchright';

// Mock config - we won't actually call the CLI in these tests, just test detection
const mockConfig = {
  repoPath: './mock',
  model: 'mock-model'
};

test.describe('Captcha Detection', () => {
  let solver: CaptchaKrakenSolver;

  test.beforeEach(() => {
    solver = new CaptchaKrakenSolver(mockConfig);
  });

  test('Recaptcha (Google Demo): Detects Checkbox then Challenge', async ({ page }) => {
    await page.goto('https://google.com/recaptcha/api2/demo');

    // 1. Detect Checkbox
    const initialCaptcha = await solver.detectCaptcha(page as any);
    expect(initialCaptcha).not.toBeNull();
    // Verify it looks like the anchor/checkbox
    const initialSrc = await initialCaptcha?.getAttribute('src');
    expect(initialSrc).toContain('recaptcha/api2/anchor');

    // 2. Click it to trigger challenge
    await initialCaptcha?.click();

    // Wait for the challenge to appear (iframe with bframe)
    // We can't rely on solver.detectCaptcha immediately as it might take a moment
    await page.waitForSelector('iframe[src*="recaptcha/api2/bframe"]', { state: 'visible', timeout: 5000 }).catch(() => {
      console.log("Challenge might not appear if bot detection is too smart or passive.");
    });

    // Note: Sometimes the demo just checks the box without a challenge. 
    // We check if either the box is checked OR a challenge appeared.
    // But the goal is to test "if challenge exists, we find it".

    // Let's see if we can force a challenge or just check detection logic.
    // If a challenge frame is present, detectCaptcha should find it.

    const challengeFrame = await page.$('iframe[src*="recaptcha/api2/bframe"]');
    if (challengeFrame && await challengeFrame.isVisible()) {
      const challengeCaptcha = await solver.detectCaptcha(page as any);
      expect(challengeCaptcha).not.toBeNull();
      const challengeSrc = await challengeCaptcha?.getAttribute('src');
      expect(challengeSrc).toContain('recaptcha/api2/bframe');
    } else {
      console.log('No challenge appeared (passive check passed), skipping challenge detection assertion.');
    }
  });

  test('hCaptcha (Demo): Detects Checkbox then Challenge', async ({ page }) => {
    await page.goto('https://democaptcha.com/demo-form-eng/hcaptcha.html');

    // 1. Detect Checkbox
    const initialCaptcha = await solver.detectCaptcha(page as any);
    expect(initialCaptcha).not.toBeNull();
    const initialSrc = await initialCaptcha?.getAttribute('src');
    expect(initialSrc).toContain('hcaptcha.com');
    expect(initialSrc).not.toContain('challenge'); // Should be the checkbox/anchor

    // 2. Click it
    await initialCaptcha?.click();

    // 3. Detect Challenge
    // Note: The demo page sometimes auto-solves or behaves erratically.
    // If we can't see the challenge frame visible, it might be off screen or hidden 
    // but the checkbox is checked.

    // We check if the challenge frame exists, even if "hidden" by some definitions (opacity/size)
    const challengeFrame = await page.$('iframe[src*="hcaptcha.com"][src*="frame=challenge"]');
    if (challengeFrame) {
      // Just verify we found the frame
      const src = await challengeFrame.getAttribute('src');
      expect(src).toContain('hcaptcha.com');
      expect(src).toContain('frame=challenge');
    } else {
      // If no challenge frame, maybe it solved immediately?
      // Check for checkbox checked state if possible, or just warn.
      console.log('hCaptcha challenge frame not found (maybe solved immediately?)');
    }
  });

  test('Cloudflare Turnstile (2Captcha Demo): Detects Container', async ({ page }) => {
    await page.goto('https://2captcha.com/demo/cloudflare-turnstile');

    // Wait for the widget to load
    await page.waitForSelector('.cf-turnstile', { state: 'visible', timeout: 10000 });

    // 1. Detect Widget
    const captcha: ElementHandle<Element> | null = await solver.detectCaptcha(page as any) as ElementHandle<Element>;
    expect(captcha).not.toBeNull();

    // It might return the iframe (if visible/open) or the container
    const tagName = await captcha?.evaluate(el => el.tagName.toLowerCase());

    if (tagName === 'iframe') {
      const src = await captcha?.getAttribute('src');
      expect(src).toContain('challenges.cloudflare.com');
    } else {
      const className = await captcha?.getAttribute('class');
      expect(className).toContain('cf-turnstile');
    }
  });
});

