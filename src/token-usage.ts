import { TokenUsage } from './types';

export interface ModelPricing {
  inputPricePerM: number;
  outputPricePerM: number;
  cachedInputPricePerM: number;
}

export const PRICING: Record<string, ModelPricing> = {
  'gemini-3-pro-preview': {
    inputPricePerM: 2.00,
    outputPricePerM: 12.00,
    cachedInputPricePerM: 0.20,
  },
  'gemini-3-flash-preview': {
    inputPricePerM: 0.50,
    outputPricePerM: 3.00,
    cachedInputPricePerM: 0.05,
  },
  'gemini-2.5-pro': {
    inputPricePerM: 1.25,
    outputPricePerM: 10.00,
    cachedInputPricePerM: 0.125,
  },
  'gemini-2.5-flash': {
    inputPricePerM: 0.30,
    outputPricePerM: 2.50,
    cachedInputPricePerM: 0.03,
  },
  'gemini-2.5-flash-preview': {
    inputPricePerM: 0.30,
    outputPricePerM: 2.50,
    cachedInputPricePerM: 0.03,
  },
  'gemini-2.5-flash-lite': {
    inputPricePerM: 0.10,
    outputPricePerM: 0.40,
    cachedInputPricePerM: 0.01,
  },
  'gemini-2.5-flash-lite-preview': {
    inputPricePerM: 0.10,
    outputPricePerM: 0.40,
    cachedInputPricePerM: 0.01,
  },
  'gemini-2.0-flash': {
    inputPricePerM: 0.10,
    outputPricePerM: 0.40,
    cachedInputPricePerM: 0.025,
  },
  'gemini-2.5-computer-use-preview-10-2025': {
    inputPricePerM: 1.25,
    outputPricePerM: 10.00,
    cachedInputPricePerM: 0.025,
  }
};

export function estimateCost(usage: TokenUsage): number {
  const model = usage.model;
  // Default to gemini-2.5-flash pricing if unknown
  const pricing = PRICING[model] || PRICING['gemini-2.5-flash'];

  const inputCost = (usage.input_tokens / 1_000_000) * pricing.inputPricePerM;
  const outputCost = (usage.output_tokens / 1_000_000) * pricing.outputPricePerM;
  const cachedCost = ((usage.cached_input_tokens || 0) / 1_000_000) * pricing.cachedInputPricePerM;

  return inputCost + outputCost + cachedCost;
}

export function aggregateTokenUsage(usages: TokenUsage[]) {
  if (usages.length === 0) {
    return {
      modelName: 'none',
      inputTokens: 0,
      outputTokens: 0,
      cachedInputTokens: 0,
      estimatedCost: 0,
    };
  }

  const modelName = usages[0].model;
  let inputTokens = 0;
  let outputTokens = 0;
  let cachedInputTokens = 0;
  let estimatedCost = 0;

  for (const usage of usages) {
    inputTokens += usage.input_tokens;
    outputTokens += usage.output_tokens;
    cachedInputTokens += usage.cached_input_tokens || 0;
    estimatedCost += estimateCost(usage);
  }

  return {
    modelName,
    inputTokens,
    outputTokens,
    cachedInputTokens,
    estimatedCost,
  };
}

