/**
 * T.A.R.S. Admin Dashboard - Reward Curve Chart
 *
 * Displays agent reward progression with rolling mean and confidence bands.
 */

import React, { useMemo } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Area,
  ComposedChart,
  ReferenceLine,
} from 'recharts';
import { RewardDataPoint, ChartConfig } from './types';
import {
  CHART_COLORS,
  DEFAULT_CHART_CONFIG,
  formatReward,
  formatEpisode,
  enhanceRewardData,
  downsampleData,
  getYDomain,
} from './utils';
import ChartWrapper from './ChartWrapper';

interface RewardCurveChartProps {
  data: RewardDataPoint[];
  config?: Partial<ChartConfig>;
  title?: string;
  subtitle?: string;
  loading?: boolean;
  error?: string;
  showRollingMean?: boolean;
  showConfidenceBand?: boolean;
  rollingWindow?: number;
  targetReward?: number; // Optional target line
}

export const RewardCurveChart: React.FC<RewardCurveChartProps> = ({
  data,
  config,
  title = 'Reward Curve',
  subtitle,
  loading = false,
  error,
  showRollingMean = true,
  showConfidenceBand = true,
  rollingWindow = 10,
  targetReward,
}) => {
  const finalConfig = { ...DEFAULT_CHART_CONFIG, ...config };

  // Process data with rolling statistics
  const processedData = useMemo(() => {
    if (data.length === 0) return [];

    let enhanced = showRollingMean
      ? enhanceRewardData(data, rollingWindow)
      : data;

    // Downsample if too many points
    enhanced = downsampleData(enhanced, 500);

    // Add confidence band (mean ± std)
    if (showConfidenceBand && showRollingMean) {
      return enhanced.map(d => ({
        ...d,
        upperBound: (d.rollingMean || 0) + (d.rollingStd || 0),
        lowerBound: (d.rollingMean || 0) - (d.rollingStd || 0),
      }));
    }

    return enhanced;
  }, [data, showRollingMean, showConfidenceBand, rollingWindow]);

  // Calculate Y-axis domain
  const yDomain = useMemo(() => {
    if (processedData.length === 0) return undefined;
    const rewards = processedData.map(d => d.reward);
    return getYDomain(rewards, 0.15);
  }, [processedData]);

  // Custom tooltip
  const CustomTooltip = ({ active, payload }: any) => {
    if (!active || !payload || payload.length === 0) return null;

    const data = payload[0].payload;
    return (
      <div
        style={{
          backgroundColor: 'white',
          padding: '10px',
          border: '1px solid #ccc',
          borderRadius: '4px',
          fontSize: '12px',
        }}
      >
        <p style={{ margin: '0 0 5px 0', fontWeight: 'bold' }}>
          {formatEpisode(data.episode)}
        </p>
        <p style={{ margin: '2px 0', color: CHART_COLORS.blue }}>
          Reward: {formatReward(data.reward)}
        </p>
        {data.rollingMean !== undefined && (
          <p style={{ margin: '2px 0', color: CHART_COLORS.orange }}>
            Rolling Mean: {formatReward(data.rollingMean)}
          </p>
        )}
        {data.rollingStd !== undefined && (
          <p style={{ margin: '2px 0', color: CHART_COLORS.neutral }}>
            Std Dev: {formatReward(data.rollingStd)}
          </p>
        )}
      </div>
    );
  };

  return (
    <ChartWrapper
      title={title}
      subtitle={subtitle || `Last ${data.length} episodes (window=${rollingWindow})`}
      loading={loading}
      error={error}
      config={finalConfig}
    >
      <ResponsiveContainer width="100%" height={finalConfig.height}>
        <ComposedChart
          data={processedData}
          margin={finalConfig.margin}
        >
          {finalConfig.showGrid && (
            <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
          )}
          <XAxis
            dataKey="episode"
            label={{ value: 'Episode', position: 'insideBottom', offset: -10 }}
            tick={{ fontSize: 12 }}
          />
          <YAxis
            label={{ value: 'Reward', angle: -90, position: 'insideLeft' }}
            tick={{ fontSize: 12 }}
            domain={yDomain}
          />
          {finalConfig.showTooltip && <Tooltip content={<CustomTooltip />} />}
          {finalConfig.showLegend && (
            <Legend
              wrapperStyle={{ fontSize: '12px' }}
              verticalAlign="top"
              height={36}
            />
          )}

          {/* Confidence band (mean ± std) */}
          {showConfidenceBand && showRollingMean && (
            <Area
              type="monotone"
              dataKey="upperBound"
              stackId="1"
              stroke="none"
              fill={CHART_COLORS.orange}
              fillOpacity={0.1}
              name="Confidence Band"
            />
          )}

          {/* Rolling mean line */}
          {showRollingMean && (
            <Line
              type="monotone"
              dataKey="rollingMean"
              stroke={CHART_COLORS.orange}
              strokeWidth={2}
              dot={false}
              name="Rolling Mean"
            />
          )}

          {/* Raw reward points */}
          <Line
            type="monotone"
            dataKey="reward"
            stroke={CHART_COLORS.blue}
            strokeWidth={1}
            dot={{ r: 2 }}
            name="Episode Reward"
          />

          {/* Target reward reference line */}
          {targetReward !== undefined && (
            <ReferenceLine
              y={targetReward}
              stroke={CHART_COLORS.success}
              strokeDasharray="5 5"
              label={{ value: 'Target', position: 'right', fontSize: 12 }}
            />
          )}
        </ComposedChart>
      </ResponsiveContainer>
    </ChartWrapper>
  );
};

export default RewardCurveChart;
