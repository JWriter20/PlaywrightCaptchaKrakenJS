export interface CaptchaKrakenConfig {
  /**
   * Path to the bundled CaptchaKraken CLI root.
   *
   * Usually you do NOT need to set this. If omitted, the solver will auto-resolve the
   * `CaptchaKraken-cli/` directory shipped inside this npm package.
   */
  repoPath?: string;
  /**
   * Command to run python (default: 'python' or 'python3').
   */
  pythonCommand?: string;
  /**
   * Model to use (default: 'gemini-2.5-flash-lite').
   */
  model?: string;
  /**
   * API provider (default: 'gemini').
   */
  apiProvider?: 'ollama' | 'gemini' | 'openrouter';
  /**
   * API Key for the provider (if required).
   */
  apiKey?: string;

  /**
   * Starting mouse position (default: { x: 100, y: 100 }).
   * HIGHLY RECOMMENDED to set this, prevents jumping around of the cursor when solving.
   */
  startingMousePosition?: { x: number, y: number };


  /**
   * Automatically re-check for newly opened / next-step captchas after each solve
   * attempt (e.g., clicking a checkbox opens an image challenge).
   *
   * Default: 10
   */
  maxSolveLoops?: number;

  /**
   * Delay (ms) after executing actions before re-detecting captchas.
   * Useful to allow challenge frames / new images to appear.
   *
   * Default: 1200
   */
  postSolveDelayMs?: number;

  /**
   * Overall time limit (ms) for the entire solve loop.
   *
   * Default: 120000 (2 minutes)
   */
  overallSolveTimeoutMs?: number;
}

export interface BoundingBox {
  0: number; // min_x
  1: number; // min_y
  2: number; // max_x
  3: number; // max_y
}

export interface ClickAction {
  action: 'click';
  target_number: number | null;
  target_bounding_box: [number, number, number, number] | null;
  target_coordinates: [number, number] | null;
}

export interface WaitAction {
  action: 'wait';
  duration_ms: number;
}

export type CaptchaAction = ClickAction | WaitAction;

export type SolverResult = CaptchaAction | CaptchaAction[];

export interface TokenUsage {
  input_tokens: number;
  output_tokens: number;
  cached_input_tokens?: number;
  model: string;
}

export interface CliResponse {
  actions: SolverResult;
  token_usage: TokenUsage[];
}

export interface Vector {
  x: number;
  y: number;
}

export interface SolveResult {
  isSolved: boolean;
  finalMousePosition: Vector;
  tokenUsage: {
    modelName: string;
    inputTokens: number;
    outputTokens: number;
    cachedInputTokens: number;
    estimatedCost: number;
  };
}
