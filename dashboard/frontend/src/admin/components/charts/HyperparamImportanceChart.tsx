/**
 * T.A.R.S. Admin Dashboard - Hyperparameter Importance Chart
 *
 * Displays hyperparameter importance from AutoML optimization.
 */

import React, { useMemo } from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Cell,
} from 'recharts';
import { HyperparamImportance, ChartConfig } from './types';
import {
  CHART_COLORS,
  DEFAULT_CHART_CONFIG,
  formatTooltipValue,
} from './utils';
import ChartWrapper from './ChartWrapper';

interface HyperparamImportanceChartProps {
  data: HyperparamImportance[];
  config?: Partial<ChartConfig>;
  title?: string;
  subtitle?: string;
  loading?: boolean;
  error?: string;
  maxParams?: number;
}

export const HyperparamImportanceChart: React.FC<
  HyperparamImportanceChartProps
> = ({
  data,
  config,
  title = 'Hyperparameter Importance',
  subtitle,
  loading = false,
  error,
  maxParams = 10,
}) => {
  const finalConfig = { ...DEFAULT_CHART_CONFIG, ...config };

  // Process data: sort by importance and take top N
  const processedData = useMemo(() => {
    if (data.length === 0) return [];

    const sorted = [...data].sort((a, b) => b.importance - a.importance);
    return sorted.slice(0, maxParams);
  }, [data, maxParams]);

  // Generate colors based on importance
  const getBarColor = (importance: number) => {
    const maxImportance = Math.max(...processedData.map(d => d.importance));
    const ratio = importance / maxImportance;

    if (ratio > 0.7) return CHART_COLORS.error;
    if (ratio > 0.4) return CHART_COLORS.warning;
    if (ratio > 0.2) return CHART_COLORS.info;
    return CHART_COLORS.neutral;
  };

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
          {data.paramName}
        </p>
        <p style={{ margin: '2px 0', color: CHART_COLORS.blue }}>
          Importance: {formatTooltipValue(data.importance, { precision: 4 })}
        </p>
        {data.stdDev !== undefined && (
          <p style={{ margin: '2px 0', color: CHART_COLORS.neutral }}>
            Std Dev: {formatTooltipValue(data.stdDev, { precision: 4 })}
          </p>
        )}
      </div>
    );
  };

  return (
    <ChartWrapper
      title={title}
      subtitle={
        subtitle || `Top ${Math.min(maxParams, data.length)} most important parameters`
      }
      loading={loading}
      error={error}
      config={finalConfig}
    >
      <ResponsiveContainer width="100%" height={finalConfig.height}>
        <BarChart
          data={processedData}
          layout="vertical"
          margin={finalConfig.margin}
        >
          {finalConfig.showGrid && (
            <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
          )}
          <XAxis
            type="number"
            label={{ value: 'Importance', position: 'insideBottom', offset: -10 }}
            tick={{ fontSize: 12 }}
          />
          <YAxis
            type="category"
            dataKey="paramName"
            width={120}
            tick={{ fontSize: 11 }}
          />
          {finalConfig.showTooltip && <Tooltip content={<CustomTooltip />} />}

          <Bar dataKey="importance" name="Importance">
            {processedData.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={getBarColor(entry.importance)} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </ChartWrapper>
  );
};

export default HyperparamImportanceChart;
