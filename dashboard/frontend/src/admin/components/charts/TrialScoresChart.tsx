/**
 * T.A.R.S. Admin Dashboard - AutoML Trial Scores Chart
 *
 * Displays optimization trial scores over time with state indicators.
 */

import React, { useMemo } from 'react';
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts';
import { TrialScoreDataPoint, ChartConfig } from './types';
import {
  CHART_COLORS,
  DEFAULT_CHART_CONFIG,
  formatTooltipValue,
  formatTrial,
  downsampleData,
  getYDomain,
} from './utils';
import ChartWrapper from './ChartWrapper';

interface TrialScoresChartProps {
  data: TrialScoreDataPoint[];
  config?: Partial<ChartConfig>;
  title?: string;
  subtitle?: string;
  loading?: boolean;
  error?: string;
  showBestLine?: boolean;
}

export const TrialScoresChart: React.FC<TrialScoresChartProps> = ({
  data,
  config,
  title = 'AutoML Trial Scores',
  subtitle,
  loading = false,
  error,
  showBestLine = true,
}) => {
  const finalConfig = { ...DEFAULT_CHART_CONFIG, ...config };

  // Process data by state
  const processedData = useMemo(() => {
    if (data.length === 0) return { complete: [], pruned: [], failed: [] };

    // Separate by state
    const complete = data.filter(d => d.state === 'COMPLETE');
    const pruned = data.filter(d => d.state === 'PRUNED');
    const failed = data.filter(d => d.state === 'FAIL');

    return {
      complete: downsampleData(complete, 300),
      pruned: downsampleData(pruned, 100),
      failed: downsampleData(failed, 100),
    };
  }, [data]);

  // Calculate best value so far
  const bestValues = useMemo(() => {
    if (data.length === 0) return [];

    const sorted = [...data].sort((a, b) => a.trialNumber - b.trialNumber);
    let bestSoFar = -Infinity;
    const result: { trialNumber: number; bestValue: number }[] = [];

    sorted.forEach(trial => {
      if (trial.state === 'COMPLETE' && trial.value > bestSoFar) {
        bestSoFar = trial.value;
      }
      result.push({ trialNumber: trial.trialNumber, bestValue: bestSoFar });
    });

    return downsampleData(result, 200);
  }, [data]);

  // Calculate Y-axis domain
  const yDomain = useMemo(() => {
    if (data.length === 0) return undefined;
    const values = data
      .filter(d => d.state === 'COMPLETE')
      .map(d => d.value);
    if (values.length === 0) return undefined;
    return getYDomain(values, 0.15);
  }, [data]);

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
          maxWidth: '300px',
        }}
      >
        <p style={{ margin: '0 0 5px 0', fontWeight: 'bold' }}>
          {formatTrial(data.trialNumber)}
        </p>
        <p style={{ margin: '2px 0', color: CHART_COLORS.blue }}>
          Value: {formatTooltipValue(data.value, { precision: 4 })}
        </p>
        <p style={{ margin: '2px 0', color: CHART_COLORS.neutral }}>
          State: {data.state}
        </p>
        {data.params && (
          <div style={{ marginTop: '5px', fontSize: '10px' }}>
            <p style={{ margin: '2px 0', fontWeight: 'bold' }}>Params:</p>
            {Object.entries(data.params)
              .slice(0, 3)
              .map(([key, value]) => (
                <p key={key} style={{ margin: '1px 0 1px 10px' }}>
                  {key}: {String(value).substring(0, 20)}
                </p>
              ))}
            {Object.keys(data.params).length > 3 && (
              <p style={{ margin: '1px 0 1px 10px' }}>
                ... {Object.keys(data.params).length - 3} more
              </p>
            )}
          </div>
        )}
      </div>
    );
  };

  return (
    <ChartWrapper
      title={title}
      subtitle={subtitle || `${data.length} trials total`}
      loading={loading}
      error={error}
      config={finalConfig}
    >
      <ResponsiveContainer width="100%" height={finalConfig.height}>
        <ScatterChart margin={finalConfig.margin}>
          {finalConfig.showGrid && (
            <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
          )}
          <XAxis
            type="number"
            dataKey="trialNumber"
            name="Trial"
            label={{ value: 'Trial Number', position: 'insideBottom', offset: -10 }}
            tick={{ fontSize: 12 }}
          />
          <YAxis
            type="number"
            dataKey="value"
            name="Score"
            label={{ value: 'Objective Value', angle: -90, position: 'insideLeft' }}
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

          {/* Best value line */}
          {showBestLine && bestValues.length > 0 && (
            <Scatter
              data={bestValues}
              line={{ stroke: CHART_COLORS.success, strokeWidth: 2 }}
              shape={() => null}
              name="Best So Far"
            />
          )}

          {/* Trial points by state */}
          <Scatter
            data={processedData.complete}
            fill={CHART_COLORS.blue}
            name="Complete"
          />
          <Scatter
            data={processedData.pruned}
            fill={CHART_COLORS.orange}
            name="Pruned"
          />
          <Scatter
            data={processedData.failed}
            fill={CHART_COLORS.red}
            name="Failed"
          />
        </ScatterChart>
      </ResponsiveContainer>
    </ChartWrapper>
  );
};

export default TrialScoresChart;
