/**
 * T.A.R.S. Admin Dashboard - Pareto Frontier Chart
 *
 * Displays Pareto frontier for multi-objective optimization.
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
  Line,
  LineChart,
  ComposedChart,
} from 'recharts';
import { ParetoFrontierPoint, ChartConfig } from './types';
import {
  CHART_COLORS,
  DEFAULT_CHART_CONFIG,
  formatTooltipValue,
  calculateParetoFrontier,
} from './utils';
import ChartWrapper from './ChartWrapper';

interface ParetoFrontierChartProps {
  data: ParetoFrontierPoint[];
  config?: Partial<ChartConfig>;
  title?: string;
  subtitle?: string;
  loading?: boolean;
  error?: string;
  objective1Label?: string;
  objective2Label?: string;
  showFrontierLine?: boolean;
}

export const ParetoFrontierChart: React.FC<ParetoFrontierChartProps> = ({
  data,
  config,
  title = 'Pareto Frontier',
  subtitle,
  loading = false,
  error,
  objective1Label = 'Objective 1',
  objective2Label = 'Objective 2',
  showFrontierLine = true,
}) => {
  const finalConfig = { ...DEFAULT_CHART_CONFIG, ...config };

  // Process data: calculate Pareto frontier
  const processedData = useMemo(() => {
    if (data.length === 0) return { frontier: [], dominated: [] };

    // If isOnFrontier is not provided, calculate it
    let dataWithFrontier = data;
    if (data.some(d => d.isOnFrontier === undefined)) {
      const frontierFlags = calculateParetoFrontier(data);
      dataWithFrontier = data.map((d, i) => ({
        ...d,
        isOnFrontier: frontierFlags[i],
      }));
    }

    const frontier = dataWithFrontier.filter(d => d.isOnFrontier);
    const dominated = dataWithFrontier.filter(d => !d.isOnFrontier);

    // Sort frontier points for line drawing
    const sortedFrontier = [...frontier].sort(
      (a, b) => a.objective1 - b.objective1
    );

    return { frontier: sortedFrontier, dominated };
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
          Trial {data.trialNumber}
        </p>
        <p style={{ margin: '2px 0', color: CHART_COLORS.blue }}>
          {objective1Label}: {formatTooltipValue(data.objective1, { precision: 4 })}
        </p>
        <p style={{ margin: '2px 0', color: CHART_COLORS.red }}>
          {objective2Label}: {formatTooltipValue(data.objective2, { precision: 4 })}
        </p>
        <p
          style={{
            margin: '2px 0',
            color: data.isOnFrontier ? CHART_COLORS.success : CHART_COLORS.neutral,
          }}
        >
          {data.isOnFrontier ? 'On Frontier ✓' : 'Dominated'}
        </p>
        {data.params && (
          <div style={{ marginTop: '5px', fontSize: '10px' }}>
            <p style={{ margin: '2px 0', fontWeight: 'bold' }}>Params:</p>
            {Object.entries(data.params)
              .slice(0, 2)
              .map(([key, value]) => (
                <p key={key} style={{ margin: '1px 0 1px 10px' }}>
                  {key}: {String(value).substring(0, 15)}
                </p>
              ))}
          </div>
        )}
      </div>
    );
  };

  return (
    <ChartWrapper
      title={title}
      subtitle={
        subtitle ||
        `${processedData.frontier.length} solutions on frontier, ${processedData.dominated.length} dominated`
      }
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
            dataKey="objective1"
            name={objective1Label}
            label={{
              value: objective1Label,
              position: 'insideBottom',
              offset: -10,
            }}
            tick={{ fontSize: 12 }}
          />
          <YAxis
            type="number"
            dataKey="objective2"
            name={objective2Label}
            label={{
              value: objective2Label,
              angle: -90,
              position: 'insideLeft',
            }}
            tick={{ fontSize: 12 }}
          />
          {finalConfig.showTooltip && <Tooltip content={<CustomTooltip />} />}
          {finalConfig.showLegend && (
            <Legend
              wrapperStyle={{ fontSize: '12px' }}
              verticalAlign="top"
              height={36}
            />
          )}

          {/* Dominated points */}
          <Scatter
            data={processedData.dominated}
            fill={CHART_COLORS.neutral}
            fillOpacity={0.4}
            name="Dominated Solutions"
          />

          {/* Frontier points */}
          <Scatter
            data={processedData.frontier}
            fill={CHART_COLORS.success}
            name="Pareto Frontier"
            line={
              showFrontierLine
                ? { stroke: CHART_COLORS.success, strokeWidth: 2 }
                : undefined
            }
          />
        </ScatterChart>
      </ResponsiveContainer>
    </ChartWrapper>
  );
};

export default ParetoFrontierChart;
