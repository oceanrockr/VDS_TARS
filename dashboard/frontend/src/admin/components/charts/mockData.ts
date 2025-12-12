/**
 * T.A.R.S. Admin Dashboard - Mock Chart Data
 *
 * Mock data generators for testing chart components.
 */

import {
  RewardDataPoint,
  LossDataPoint,
  EntropyDataPoint,
  ExplorationDataPoint,
  NashConvergenceDataPoint,
  TrialScoreDataPoint,
  HyperparamImportance,
  ParetoFrontierPoint,
} from './types';

// ============================================================================
// AGENT TRAINING MOCK DATA
// ============================================================================

export function generateMockRewardData(
  numEpisodes: number = 100,
  targetReward: number = 500
): RewardDataPoint[] {
  const data: RewardDataPoint[] = [];
  let baseReward = 0;

  for (let i = 0; i < numEpisodes; i++) {
    // Simulate learning progress with noise
    const progress = i / numEpisodes;
    const expectedReward = targetReward * progress;
    const noise = (Math.random() - 0.5) * 100;
    const reward = Math.max(0, expectedReward + noise);

    data.push({
      episode: i + 1,
      timestamp: new Date(Date.now() - (numEpisodes - i) * 60000).toISOString(),
      reward,
    });

    baseReward = reward;
  }

  return data;
}

export function generateMockLossData(
  numEpisodes: number = 100
): LossDataPoint[] {
  const data: LossDataPoint[] = [];

  for (let i = 0; i < numEpisodes; i++) {
    const progress = i / numEpisodes;
    const decay = Math.exp(-progress * 3);

    data.push({
      episode: i + 1,
      timestamp: new Date(Date.now() - (numEpisodes - i) * 60000).toISOString(),
      valueLoss: (Math.random() * 0.5 + 0.5) * decay,
      policyLoss: (Math.random() * 0.3 + 0.3) * decay,
      tdError: (Math.random() * 0.2 + 0.2) * decay,
      criticLoss: (Math.random() * 0.4 + 0.4) * decay,
      actorLoss: (Math.random() * 0.35 + 0.35) * decay,
      totalLoss: (Math.random() * 1.0 + 1.0) * decay,
    });
  }

  return data;
}

export function generateMockEntropyData(
  numEpisodes: number = 100
): EntropyDataPoint[] {
  const data: EntropyDataPoint[] = [];
  const targetEntropy = -2.0;

  for (let i = 0; i < numEpisodes; i++) {
    const progress = i / numEpisodes;
    const decay = Math.exp(-progress * 2);
    const entropy = targetEntropy * (1 - decay) + Math.random() * 0.5 * decay;

    data.push({
      episode: i + 1,
      timestamp: new Date(Date.now() - (numEpisodes - i) * 60000).toISOString(),
      entropy,
      targetEntropy,
    });
  }

  return data;
}

export function generateMockExplorationData(
  numEpisodes: number = 100
): ExplorationDataPoint[] {
  const data: ExplorationDataPoint[] = [];

  for (let i = 0; i < numEpisodes; i++) {
    const progress = i / numEpisodes;
    const epsilon = Math.max(0.01, 1.0 - progress * 0.99);
    const temperature = Math.max(0.1, 2.0 - progress * 1.9);
    const noiseStd = Math.max(0.05, 0.5 - progress * 0.45);

    data.push({
      episode: i + 1,
      timestamp: new Date(Date.now() - (numEpisodes - i) * 60000).toISOString(),
      epsilon,
      temperature,
      noiseStd,
      explorationRate: (epsilon + temperature / 2 + noiseStd) / 2.5,
    });
  }

  return data;
}

export function generateMockNashConvergenceData(
  numIterations: number = 50
): NashConvergenceDataPoint[] {
  const data: NashConvergenceDataPoint[] = [];
  const threshold = 0.001;

  for (let i = 0; i < numIterations; i++) {
    const progress = i / numIterations;
    const decay = Math.exp(-progress * 4);

    data.push({
      iteration: i + 1,
      timestamp: new Date(Date.now() - (numIterations - i) * 120000).toISOString(),
      nashGap: threshold * 100 * decay + Math.random() * threshold * 10,
      exploitability: threshold * 80 * decay + Math.random() * threshold * 8,
      convergenceThreshold: threshold,
    });
  }

  return data;
}

// ============================================================================
// AUTOML MOCK DATA
// ============================================================================

export function generateMockTrialData(
  numTrials: number = 50
): TrialScoreDataPoint[] {
  const data: TrialScoreDataPoint[] = [];
  let bestValue = -Infinity;

  for (let i = 0; i < numTrials; i++) {
    const progress = i / numTrials;
    const exploration = Math.random() > 0.7; // 30% exploration

    // Simulate TPE optimizer improving over time
    const expectedValue = exploration
      ? Math.random() * 100
      : bestValue + (Math.random() - 0.3) * 50;

    const value = Math.max(0, Math.min(100, expectedValue));
    bestValue = Math.max(bestValue, value);

    // Simulate state distribution
    let state: 'COMPLETE' | 'RUNNING' | 'FAIL' | 'PRUNED' = 'COMPLETE';
    const rand = Math.random();
    if (rand < 0.1) state = 'FAIL';
    else if (rand < 0.2) state = 'PRUNED';

    data.push({
      trialNumber: i + 1,
      timestamp: new Date(Date.now() - (numTrials - i) * 180000).toISOString(),
      value,
      state,
      params: {
        learning_rate: Math.random() * 0.01,
        gamma: 0.9 + Math.random() * 0.099,
        batch_size: [32, 64, 128, 256][Math.floor(Math.random() * 4)],
        hidden_dim: [64, 128, 256, 512][Math.floor(Math.random() * 4)],
      },
    });
  }

  return data;
}

export function generateMockHyperparamImportance(): HyperparamImportance[] {
  const params = [
    { name: 'learning_rate', base: 0.8 },
    { name: 'gamma', base: 0.6 },
    { name: 'batch_size', base: 0.4 },
    { name: 'hidden_dim', base: 0.3 },
    { name: 'entropy_coef', base: 0.25 },
    { name: 'value_coef', base: 0.2 },
    { name: 'n_steps', base: 0.15 },
    { name: 'tau', base: 0.1 },
    { name: 'target_update_freq', base: 0.08 },
    { name: 'clip_range', base: 0.05 },
  ];

  return params.map(p => ({
    paramName: p.name,
    importance: p.base + Math.random() * 0.1,
    stdDev: Math.random() * 0.05,
  }));
}

export function generateMockParetoFrontier(
  numPoints: number = 50
): ParetoFrontierPoint[] {
  const data: ParetoFrontierPoint[] = [];

  for (let i = 0; i < numPoints; i++) {
    // Generate points with trade-off between objectives
    const obj1 = Math.random() * 100;
    const obj2 = Math.random() * 100;

    data.push({
      trialNumber: i + 1,
      objective1: obj1,
      objective2: obj2,
      params: {
        learning_rate: Math.random() * 0.01,
        gamma: 0.9 + Math.random() * 0.099,
      },
      isOnFrontier: false, // Will be calculated by chart component
    });
  }

  return data;
}

// ============================================================================
// COMPREHENSIVE MOCK DATA SETS
// ============================================================================

export const MOCK_DATA = {
  // DQN Agent
  dqn: {
    reward: generateMockRewardData(200, 450),
    loss: generateMockLossData(200),
    entropy: generateMockEntropyData(200),
    exploration: generateMockExplorationData(200),
  },

  // A2C Agent
  a2c: {
    reward: generateMockRewardData(150, 380),
    loss: generateMockLossData(150),
    entropy: generateMockEntropyData(150),
    exploration: generateMockExplorationData(150),
  },

  // PPO Agent
  ppo: {
    reward: generateMockRewardData(180, 500),
    loss: generateMockLossData(180),
    entropy: generateMockEntropyData(180),
    exploration: generateMockExplorationData(180),
  },

  // DDPG Agent
  ddpg: {
    reward: generateMockRewardData(220, 420),
    loss: generateMockLossData(220),
    exploration: generateMockExplorationData(220),
  },

  // Multi-agent Nash
  nash: generateMockNashConvergenceData(60),

  // AutoML
  automl: {
    trials: generateMockTrialData(100),
    importance: generateMockHyperparamImportance(),
    pareto: generateMockParetoFrontier(80),
  },
};
