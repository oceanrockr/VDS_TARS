/**
 * T.A.R.S. Admin Dashboard - Chart Utilities
 *
 * Shared utilities for chart components including theme, formatters, and helpers.
 */

import { ChartColors, TooltipFormatterOptions } from './types';

// ============================================================================
// THEME COLORS
// ============================================================================

export const CHART_COLORS: ChartColors = {
  primary: '#1976d2',
  secondary: '#dc004e',
  success: '#4caf50',
  warning: '#ff9800',
  error: '#f44336',
  info: '#2196f3',
  neutral: '#9e9e9e',

  // Extended palette for multi-line charts
  blue: '#1976d2',
  red: '#f44336',
  green: '#4caf50',
  orange: '#ff9800',
  purple: '#9c27b0',
  teal: '#009688',
  pink: '#e91e63',
  indigo: '#3f51b5',
  amber: '#ffc107',
  cyan: '#00bcd4',
};

export const GRADIENT_COLORS = {
  blueGradient: ['#1976d2', '#42a5f5'],
  greenGradient: ['#4caf50', '#81c784'],
  redGradient: ['#f44336', '#e57373'],
  purpleGradient: ['#9c27b0', '#ba68c8'],
  orangeGradient: ['#ff9800', '#ffb74d'],
};

// ============================================================================
// DEFAULT CHART CONFIGURATION
// ============================================================================

export const DEFAULT_CHART_CONFIG = {
  width: undefined, // Responsive by default
  height: 300,
  margin: {
    top: 5,
    right: 30,
    bottom: 20,
    left: 20,
  },
  showGrid: true,
  showLegend: true,
  showTooltip: true,
  responsive: true,
};

export const COMPACT_CHART_CONFIG = {
  ...DEFAULT_CHART_CONFIG,
  height: 200,
  margin: {
    top: 5,
    right: 20,
    bottom: 15,
    left: 15,
  },
  showLegend: false,
};

export const LARGE_CHART_CONFIG = {
  ...DEFAULT_CHART_CONFIG,
  height: 400,
  margin: {
    top: 10,
    right: 40,
    bottom: 30,
    left: 30,
  },
};

// ============================================================================
// TOOLTIP FORMATTERS
// ============================================================================

export function formatTooltipValue(
  value: number,
  options?: TooltipFormatterOptions
): string {
  const {
    precision = 2,
    unit = '',
    prefix = '',
    suffix = '',
  } = options || {};

  const formatted = value.toFixed(precision);
  return `${prefix}${formatted}${unit}${suffix}`;
}

export function formatTimestamp(timestamp: string): string {
  try {
    const date = new Date(timestamp);
    return date.toLocaleString('en-US', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  } catch {
    return timestamp;
  }
}

export function formatShortTimestamp(timestamp: string): string {
  try {
    const date = new Date(timestamp);
    return date.toLocaleTimeString('en-US', {
      hour: '2-digit',
      minute: '2-digit',
    });
  } catch {
    return timestamp;
  }
}

export function formatReward(value: number): string {
  return formatTooltipValue(value, { precision: 2 });
}

export function formatLoss(value: number): string {
  return formatTooltipValue(value, { precision: 4 });
}

export function formatPercentage(value: number): string {
  return formatTooltipValue(value * 100, { precision: 1, suffix: '%' });
}

export function formatEpisode(value: number): string {
  return `Episode ${value}`;
}

export function formatTrial(value: number): string {
  return `Trial ${value}`;
}

// ============================================================================
// DATA PROCESSING
// ============================================================================

/**
 * Calculate rolling mean for a series of values
 */
export function calculateRollingMean(
  data: number[],
  window: number = 10
): number[] {
  const result: number[] = [];

  for (let i = 0; i < data.length; i++) {
    const start = Math.max(0, i - window + 1);
    const subset = data.slice(start, i + 1);
    const mean = subset.reduce((sum, val) => sum + val, 0) / subset.length;
    result.push(mean);
  }

  return result;
}

/**
 * Calculate rolling standard deviation
 */
export function calculateRollingStd(
  data: number[],
  window: number = 10
): number[] {
  const result: number[] = [];

  for (let i = 0; i < data.length; i++) {
    const start = Math.max(0, i - window + 1);
    const subset = data.slice(start, i + 1);
    const mean = subset.reduce((sum, val) => sum + val, 0) / subset.length;
    const variance =
      subset.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) /
      subset.length;
    result.push(Math.sqrt(variance));
  }

  return result;
}

/**
 * Add rolling statistics to reward data
 */
export function enhanceRewardData<T extends { reward: number }>(
  data: T[],
  window: number = 10
): (T & { rollingMean: number; rollingStd: number })[] {
  const rewards = data.map(d => d.reward);
  const rollingMean = calculateRollingMean(rewards, window);
  const rollingStd = calculateRollingStd(rewards, window);

  return data.map((d, i) => ({
    ...d,
    rollingMean: rollingMean[i],
    rollingStd: rollingStd[i],
  }));
}

/**
 * Calculate Pareto frontier for multi-objective optimization
 */
export function calculateParetoFrontier(
  points: { objective1: number; objective2: number }[]
): boolean[] {
  const n = points.length;
  const isOnFrontier: boolean[] = new Array(n).fill(false);

  for (let i = 0; i < n; i++) {
    let dominated = false;

    for (let j = 0; j < n; j++) {
      if (i === j) continue;

      // Point j dominates point i if it's better on both objectives
      const betterObj1 = points[j].objective1 >= points[i].objective1;
      const betterObj2 = points[j].objective2 >= points[i].objective2;
      const strictlyBetter =
        points[j].objective1 > points[i].objective1 ||
        points[j].objective2 > points[i].objective2;

      if (betterObj1 && betterObj2 && strictlyBetter) {
        dominated = true;
        break;
      }
    }

    isOnFrontier[i] = !dominated;
  }

  return isOnFrontier;
}

/**
 * Downsample data for better chart performance
 */
export function downsampleData<T>(
  data: T[],
  maxPoints: number = 500
): T[] {
  if (data.length <= maxPoints) {
    return data;
  }

  const step = Math.ceil(data.length / maxPoints);
  const downsampled: T[] = [];

  for (let i = 0; i < data.length; i += step) {
    downsampled.push(data[i]);
  }

  // Always include the last point
  if (downsampled[downsampled.length - 1] !== data[data.length - 1]) {
    downsampled.push(data[data.length - 1]);
  }

  return downsampled;
}

/**
 * Get domain (min/max) for Y-axis with padding
 */
export function getYDomain(
  data: number[],
  padding: number = 0.1
): [number, number] {
  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min;

  return [
    min - range * padding,
    max + range * padding,
  ];
}

/**
 * Get adaptive tick count based on data size
 */
export function getAdaptiveTickCount(dataLength: number): number {
  if (dataLength < 10) return dataLength;
  if (dataLength < 50) return 10;
  if (dataLength < 100) return 20;
  return 30;
}

// ============================================================================
// CHART HELPERS
// ============================================================================

/**
 * Get color based on value threshold
 */
export function getThresholdColor(
  value: number,
  thresholds: { warning: number; error: number }
): string {
  if (value >= thresholds.error) {
    return CHART_COLORS.error;
  } else if (value >= thresholds.warning) {
    return CHART_COLORS.warning;
  } else {
    return CHART_COLORS.success;
  }
}

/**
 * Get color based on trend (positive/negative)
 */
export function getTrendColor(value: number): string {
  return value >= 0 ? CHART_COLORS.success : CHART_COLORS.error;
}

/**
 * Format large numbers (K, M, B notation)
 */
export function formatLargeNumber(value: number): string {
  if (value >= 1e9) {
    return (value / 1e9).toFixed(1) + 'B';
  } else if (value >= 1e6) {
    return (value / 1e6).toFixed(1) + 'M';
  } else if (value >= 1e3) {
    return (value / 1e3).toFixed(1) + 'K';
  }
  return value.toString();
}

/**
 * Get responsive chart dimensions based on container
 */
export function getResponsiveDimensions(
  containerWidth: number,
  aspectRatio: number = 16 / 9
): { width: number; height: number } {
  return {
    width: containerWidth,
    height: containerWidth / aspectRatio,
  };
}
