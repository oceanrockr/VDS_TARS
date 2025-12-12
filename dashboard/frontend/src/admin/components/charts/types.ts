/**
 * T.A.R.S. Admin Dashboard - Chart Type Definitions
 *
 * TypeScript interfaces for all chart data types used in the admin dashboard.
 */

// ============================================================================
// AGENT TRAINING METRICS
// ============================================================================

export interface RewardDataPoint {
  episode: number;
  timestamp: string;
  reward: number;
  rollingMean?: number;
  rollingStd?: number;
}

export interface LossDataPoint {
  episode: number;
  timestamp: string;
  valueLoss?: number;
  policyLoss?: number;
  tdError?: number;
  criticLoss?: number;
  actorLoss?: number;
  totalLoss?: number;
}

export interface EntropyDataPoint {
  episode: number;
  timestamp: string;
  entropy: number;
  targetEntropy?: number;
}

export interface ExplorationDataPoint {
  episode: number;
  timestamp: string;
  epsilon?: number;
  temperature?: number;
  noiseStd?: number;
  explorationRate?: number;
}

export interface NashConvergenceDataPoint {
  iteration: number;
  timestamp: string;
  nashGap: number;
  exploitability: number;
  convergenceThreshold: number;
}

// ============================================================================
// AUTOML METRICS
// ============================================================================

export interface TrialScoreDataPoint {
  trialNumber: number;
  timestamp: string;
  value: number;
  params: Record<string, any>;
  state: 'COMPLETE' | 'RUNNING' | 'FAIL' | 'PRUNED';
}

export interface HyperparamImportance {
  paramName: string;
  importance: number;
  stdDev?: number;
}

export interface ParetoFrontierPoint {
  trialNumber: number;
  objective1: number;
  objective2: number;
  params: Record<string, any>;
  isOnFrontier: boolean;
}

export interface OptimizationTrajectory {
  iteration: number;
  timestamp: string;
  bestValue: number;
  currentValue: number;
  explorationPhase: boolean;
}

// ============================================================================
// HYPERSYNC METRICS
// ============================================================================

export interface HyperSyncProposalMetric {
  proposalId: string;
  timestamp: string;
  agentName: string;
  oldReward: number;
  newReward: number;
  improvement: number;
  status: 'PENDING' | 'APPROVED' | 'REJECTED';
}

export interface HyperSyncApprovalRate {
  timestamp: string;
  approvalRate: number;
  totalProposals: number;
  approvedCount: number;
}

// ============================================================================
// SYSTEM METRICS
// ============================================================================

export interface SystemHealthMetric {
  timestamp: string;
  cpuUsage: number;
  memoryUsage: number;
  diskUsage?: number;
  networkLatency?: number;
}

export interface APIKeyUsageMetric {
  timestamp: string;
  keyName: string;
  requestCount: number;
  errorCount: number;
  rateLimitHits: number;
}

export interface JWTMetric {
  timestamp: string;
  issuedCount: number;
  verifiedCount: number;
  failedCount: number;
  activeTokens: number;
}

// ============================================================================
// CHART CONFIGURATION
// ============================================================================

export interface ChartConfig {
  width?: number;
  height?: number;
  margin?: {
    top?: number;
    right?: number;
    bottom?: number;
    left?: number;
  };
  showGrid?: boolean;
  showLegend?: boolean;
  showTooltip?: boolean;
  responsive?: boolean;
  syncId?: string; // For syncing multiple charts
}

export interface ChartColors {
  primary: string;
  secondary: string;
  success: string;
  warning: string;
  error: string;
  info: string;
  neutral: string;
  [key: string]: string;
}

export interface TooltipFormatterOptions {
  precision?: number;
  unit?: string;
  prefix?: string;
  suffix?: string;
}
