/**
 * T.A.R.S. Admin Dashboard - Exploration Metrics Chart
 *
 * Displays exploration parameters (epsilon, temperature, noise, etc.) over training.
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
} from 'recharts';
import { ExplorationDataPoint, ChartConfig } from './types';
import {
  CHART_COLORS,
  DEFAULT_CHART_CONFIG,
  formatTooltipValue,
  formatEpisode,
  downsampleData,
} from './utils';
import ChartWrapper from './ChartWrapper';

interface ExplorationChartProps {
  data: ExplorationDataPoint[];
  config?: Partial<ChartConfig>;
  title?: string;
  subtitle?: string;
  loading?: boolean;
  error?: string;
}

export const ExplorationChart: React.FC<ExplorationChartProps> = ({
  data,
  config,
  title = 'Exploration Metrics',
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

  // Determine which metrics are available
  const hasEpsilon = processedData.some(d => d.epsilon !== undefined);
  const hasTemperature = processedData.some(d => d.temperature !== undefined);
  const hasNoiseStd = processedData.some(d => d.noiseStd !== undefined);
  const hasExplorationRate = processedData.some(
    d => d.explorationRate !== undefined
  );

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
        {data.epsilon !== undefined && (
          <p style={{ margin: '2px 0', color: CHART_COLORS.blue }}>
            Epsilon: {formatTooltipValue(data.epsilon, { precision: 4 })}
          </p>
        )}
        {data.temperature !== undefined && (
          <p style={{ margin: '2px 0', color: CHART_COLORS.red }}>
            Temperature: {formatTooltipValue(data.temperature, { precision: 4 })}
          </p>
        )}
        {data.noiseStd !== undefined && (
          <p style={{ margin: '2px 0', color: CHART_COLORS.green }}>
            Noise Std: {formatTooltipValue(data.noiseStd, { precision: 4 })}
          </p>
        )}
        {data.explorationRate !== undefined && (
          <p style={{ margin: '2px 0', color: CHART_COLORS.purple }}>
            Exploration Rate: {formatTooltipValue(data.explorationRate, { precision: 4 })}
          </p>
        )}
      </div>
    );
  };

  return (
    <ChartWrapper
      title={title}
      subtitle={subtitle || 'Exploration strategy decay over training'}
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
            label={{ value: 'Value', angle: -90, position: 'insideLeft' }}
            tick={{ fontSize: 12 }}
            domain={[0, 'auto']}
          />
          {finalConfig.showTooltip && <Tooltip content={<CustomTooltip />} />}
          {finalConfig.showLegend && (
            <Legend
              wrapperStyle={{ fontSize: '12px' }}
              verticalAlign="top"
              height={36}
            />
          )}

          {/* Exploration metric lines */}
          {hasEpsilon && (
            <Line
              type="monotone"
              dataKey="epsilon"
              stroke={CHART_COLORS.blue}
              strokeWidth={2}
              dot={false}
              name="Epsilon (ε-greedy)"
            />
          )}
          {hasTemperature && (
            <Line
              type="monotone"
              dataKey="temperature"
              stroke={CHART_COLORS.red}
              strokeWidth={2}
              dot={false}
              name="Temperature"
            />
          )}
          {hasNoiseStd && (
            <Line
              type="monotone"
              dataKey="noiseStd"
              stroke={CHART_COLORS.green}
              strokeWidth={2}
              dot={false}
              name="Noise Std Dev"
            />
          )}
          {hasExplorationRate && (
            <Line
              type="monotone"
              dataKey="explorationRate"
              stroke={CHART_COLORS.purple}
              strokeWidth={2}
              dot={false}
              name="Exploration Rate"
            />
          )}
        </LineChart>
      </ResponsiveContainer>
    </ChartWrapper>
  );
};

export default ExplorationChart;
