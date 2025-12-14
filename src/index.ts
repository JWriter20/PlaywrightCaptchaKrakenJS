export * from './types';
export { CaptchaKrakenSolver } from './solver';

/**
 * Solve a captcha on a Playwright page.
 * 
 * @param page - The Playwright page containing the captcha
 * @param config - Configuration for the CaptchaKraken solver
 * @returns Promise that resolves when the captcha solving is complete
 * 
 * @example
 * ```typescript
 * import { solve } from 'playwrightcaptchakrakenjs';
 * 
 * await solve(page, {
 *   apiKey: process.env.GEMINI_API_KEY
 * });
 * ```
 */
