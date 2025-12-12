/**
 * T.A.R.S. Admin Dashboard - Entropy Chart
 *
 * Displays policy entropy over training to monitor exploration vs exploitation.
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
import { EntropyDataPoint, ChartConfig } from './types';
import {
  CHART_COLORS,
  DEFAULT_CHART_CONFIG,
  formatTooltipValue,
  formatEpisode,
  downsampleData,
  getYDomain,
} from './utils';
import ChartWrapper from './ChartWrapper';

interface EntropyChartProps {
  data: EntropyDataPoint[];
  config?: Partial<ChartConfig>;
  title?: string;
  subtitle?: string;
  loading?: boolean;
  error?: string;
  showTargetEntropy?: boolean;
}

export const EntropyChart: React.FC<EntropyChartProps> = ({
  data,
  config,
  title = 'Policy Entropy',
  subtitle,
  loading = false,
  error,
  showTargetEntropy = true,
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
    const entropies = processedData.map(d => d.entropy);
    return getYDomain(entropies, 0.15);
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
        <p style={{ margin: '2px 0', color: CHART_COLORS.purple }}>
          Entropy: {formatTooltipValue(data.entropy, { precision: 4 })}
        </p>
        {data.targetEntropy !== undefined && (
          <p style={{ margin: '2px 0', color: CHART_COLORS.orange }}>
            Target: {formatTooltipValue(data.targetEntropy, { precision: 4 })}
          </p>
        )}
      </div>
    );
  };

  return (
    <ChartWrapper
      title={title}
      subtitle={subtitle || 'Higher entropy = more exploration'}
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
            dataKey="episode"
            label={{ value: 'Episode', position: 'insideBottom', offset: -10 }}
            tick={{ fontSize: 12 }}
          />
          <YAxis
            label={{ value: 'Entropy', angle: -90, position: 'insideLeft' }}
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

          {/* Target entropy reference line */}
          {showTargetEntropy &&
            processedData.length > 0 &&
            processedData[0].targetEntropy !== undefined && (
              <ReferenceLine
                y={processedData[0].targetEntropy}
                stroke={CHART_COLORS.orange}
                strokeDasharray="5 5"
                label={{ value: 'Target', position: 'right', fontSize: 12 }}
              />
            )}

          {/* Entropy line */}
          <Line
            type="monotone"
            dataKey="entropy"
            stroke={CHART_COLORS.purple}
            strokeWidth={2}
            dot={{ r: 2 }}
            name="Policy Entropy"
          />
        </LineChart>
      </ResponsiveContainer>
    </ChartWrapper>
  );
};

export default EntropyChart;
