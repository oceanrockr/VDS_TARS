/**
 * T.A.R.S. Admin Dashboard - Nash Convergence Chart
 *
 * Displays Nash equilibrium convergence metrics for multi-agent RL.
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
  ReferenceLine,
} from 'recharts';
import { NashConvergenceDataPoint, ChartConfig } from './types';
import {
  CHART_COLORS,
  DEFAULT_CHART_CONFIG,
  formatTooltipValue,
  downsampleData,
  getYDomain,
} from './utils';
import ChartWrapper from './ChartWrapper';

interface NashConvergenceChartProps {
  data: NashConvergenceDataPoint[];
  config?: Partial<ChartConfig>;
  title?: string;
  subtitle?: string;
  loading?: boolean;
  error?: string;
}

export const NashConvergenceChart: React.FC<NashConvergenceChartProps> = ({
  data,
  config,
  title = 'Nash Equilibrium Convergence',
  subtitle,
  loading = false,
  error,
}) => {
  const finalConfig = { ...DEFAULT_CHART_CONFIG, ...config };

  // Process data
  const processedData = useMemo(() => {
    if (data.length === 0) return [];
    return downsampleData(data, 500);
  }, [data]);

  // Calculate Y-axis domain
  const yDomain = useMemo(() => {
    if (processedData.length === 0) return undefined;
    const allValues = processedData.flatMap(d => [d.nashGap, d.exploitability]);
    return getYDomain(allValues, 0.15);
  }, [processedData]);

  // Get convergence threshold (assume constant)
  const threshold = processedData[0]?.convergenceThreshold;

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
          Iteration {data.iteration}
        </p>
        <p style={{ margin: '2px 0', color: CHART_COLORS.blue }}>
          Nash Gap: {formatTooltipValue(data.nashGap, { precision: 6 })}
        </p>
        <p style={{ margin: '2px 0', color: CHART_COLORS.red }}>
          Exploitability: {formatTooltipValue(data.exploitability, { precision: 6 })}
        </p>
        <p style={{ margin: '2px 0', color: CHART_COLORS.success }}>
          Threshold: {formatTooltipValue(data.convergenceThreshold, { precision: 6 })}
        </p>
      </div>
    );
  };

  return (
    <ChartWrapper
      title={title}
      subtitle={subtitle || 'Lower values indicate better convergence'}
      loading={loading}
      error={error}
      config={finalConfig}
    >
      <ResponsiveContainer width="100%" height={finalConfig.height}>
        <LineChart data={processedData} margin={finalConfig.margin}>
          {finalConfig.showGrid && (
            <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
          )}
          <XAxis
            dataKey="iteration"
            label={{ value: 'Iteration', position: 'insideBottom', offset: -10 }}
            tick={{ fontSize: 12 }}
          />
          <YAxis
            label={{ value: 'Value', angle: -90, position: 'insideLeft' }}
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

          {/* Convergence threshold reference line */}
          {threshold !== undefined && (
            <ReferenceLine
              y={threshold}
              stroke={CHART_COLORS.success}
              strokeDasharray="5 5"
              label={{ value: 'Threshold', position: 'right', fontSize: 12 }}
            />
          )}

          {/* Nash gap line */}
          <Line
            type="monotone"
            dataKey="nashGap"
            stroke={CHART_COLORS.blue}
            strokeWidth={2}
            dot={{ r: 2 }}
            name="Nash Gap"
          />

          {/* Exploitability line */}
          <Line
            type="monotone"
            dataKey="exploitability"
            stroke={CHART_COLORS.red}
            strokeWidth={2}
            dot={{ r: 2 }}
            name="Exploitability"
          />
        </LineChart>
      </ResponsiveContainer>
    </ChartWrapper>
  );
};

export default NashConvergenceChart;
