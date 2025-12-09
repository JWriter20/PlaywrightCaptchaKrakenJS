export interface CaptchaKrakenConfig {
  /**
   * Path to the CaptchaKraken-cli repository root.
   */
  repoPath: string;
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
  apiProvider?: 'ollama' | 'gemini';
  /**
   * API Key for the provider (if required).
   */
  apiKey?: string;
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

export type SolverResult = ClickAction | ClickAction[];

