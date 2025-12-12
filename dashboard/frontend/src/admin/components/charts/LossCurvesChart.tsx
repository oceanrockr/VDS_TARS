/**
 * T.A.R.S. Admin Dashboard - Loss Curves Chart
 *
 * Displays multiple loss metrics (value, policy, TD error, critic, actor) for RL agents.
 */

import React, { useMemo, useState } from 'react';
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
import { Box, ToggleButtonGroup, ToggleButton, Typography } from '@mui/material';
import { LossDataPoint, ChartConfig } from './types';
import {
  CHART_COLORS,
  DEFAULT_CHART_CONFIG,
  formatLoss,
  formatEpisode,
  downsampleData,
  getYDomain,
} from './utils';
import ChartWrapper from './ChartWrapper';

interface LossCurvesChartProps {
  data: LossDataPoint[];
  config?: Partial<ChartConfig>;
  title?: string;
  subtitle?: string;
  loading?: boolean;
  error?: string;
  showValueLoss?: boolean;
  showPolicyLoss?: boolean;
  showTDError?: boolean;
  showCriticLoss?: boolean;
  showActorLoss?: boolean;
  showTotalLoss?: boolean;
}

export const LossCurvesChart: React.FC<LossCurvesChartProps> = ({
  data,
  config,
  title = 'Loss Curves',
  subtitle,
  loading = false,
  error,
  showValueLoss = true,
  showPolicyLoss = true,
  showTDError = true,
  showCriticLoss = true,
  showActorLoss = true,
  showTotalLoss = false,
}) => {
  const finalConfig = { ...DEFAULT_CHART_CONFIG, ...config };

  // State for toggling loss types
  const [visibleLosses, setVisibleLosses] = useState<string[]>(() => {
    const visible: string[] = [];
    if (showValueLoss) visible.push('value');
    if (showPolicyLoss) visible.push('policy');
    if (showTDError) visible.push('td');
    if (showCriticLoss) visible.push('critic');
    if (showActorLoss) visible.push('actor');
    if (showTotalLoss) visible.push('total');
    return visible;
  });

  // Process data
  const processedData = useMemo(() => {
    if (data.length === 0) return [];
    return downsampleData(data, 500);
  }, [data]);

  // Calculate Y-axis domain based on visible losses
  const yDomain = useMemo(() => {
    if (processedData.length === 0) return undefined;

    const allValues: number[] = [];
    processedData.forEach(d => {
      if (visibleLosses.includes('value') && d.valueLoss !== undefined)
        allValues.push(d.valueLoss);
      if (visibleLosses.includes('policy') && d.policyLoss !== undefined)
        allValues.push(d.policyLoss);
      if (visibleLosses.includes('td') && d.tdError !== undefined)
        allValues.push(d.tdError);
      if (visibleLosses.includes('critic') && d.criticLoss !== undefined)
        allValues.push(d.criticLoss);
      if (visibleLosses.includes('actor') && d.actorLoss !== undefined)
        allValues.push(d.actorLoss);
      if (visibleLosses.includes('total') && d.totalLoss !== undefined)
        allValues.push(d.totalLoss);
    });

    if (allValues.length === 0) return undefined;
    return getYDomain(allValues, 0.15);
  }, [processedData, visibleLosses]);

  // Handle toggle
  const handleToggle = (
    event: React.MouseEvent<HTMLElement>,
    newVisibleLosses: string[]
  ) => {
    // Require at least one loss type to be visible
    if (newVisibleLosses.length > 0) {
      setVisibleLosses(newVisibleLosses);
    }
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
          {formatEpisode(data.episode)}
        </p>
        {data.valueLoss !== undefined && (
          <p style={{ margin: '2px 0', color: CHART_COLORS.blue }}>
            Value Loss: {formatLoss(data.valueLoss)}
          </p>
        )}
        {data.policyLoss !== undefined && (
          <p style={{ margin: '2px 0', color: CHART_COLORS.red }}>
            Policy Loss: {formatLoss(data.policyLoss)}
          </p>
        )}
        {data.tdError !== undefined && (
          <p style={{ margin: '2px 0', color: CHART_COLORS.green }}>
            TD Error: {formatLoss(data.tdError)}
          </p>
        )}
        {data.criticLoss !== undefined && (
          <p style={{ margin: '2px 0', color: CHART_COLORS.purple }}>
            Critic Loss: {formatLoss(data.criticLoss)}
          </p>
        )}
        {data.actorLoss !== undefined && (
          <p style={{ margin: '2px 0', color: CHART_COLORS.orange }}>
            Actor Loss: {formatLoss(data.actorLoss)}
          </p>
        )}
        {data.totalLoss !== undefined && (
          <p style={{ margin: '2px 0', color: CHART_COLORS.neutral }}>
            Total Loss: {formatLoss(data.totalLoss)}
          </p>
        )}
      </div>
    );
  };

  // Actions (toggle buttons)
  const actions = (
    <ToggleButtonGroup
      value={visibleLosses}
      onChange={handleToggle}
      aria-label="visible losses"
      size="small"
    >
      {showValueLoss && (
        <ToggleButton value="value" aria-label="value loss">
          <Typography variant="caption">Value</Typography>
        </ToggleButton>
      )}
      {showPolicyLoss && (
        <ToggleButton value="policy" aria-label="policy loss">
          <Typography variant="caption">Policy</Typography>
        </ToggleButton>
      )}
      {showTDError && (
        <ToggleButton value="td" aria-label="td error">
          <Typography variant="caption">TD</Typography>
        </ToggleButton>
      )}
      {showCriticLoss && (
        <ToggleButton value="critic" aria-label="critic loss">
          <Typography variant="caption">Critic</Typography>
        </ToggleButton>
      )}
      {showActorLoss && (
        <ToggleButton value="actor" aria-label="actor loss">
          <Typography variant="caption">Actor</Typography>
        </ToggleButton>
      )}
      {showTotalLoss && (
        <ToggleButton value="total" aria-label="total loss">
          <Typography variant="caption">Total</Typography>
        </ToggleButton>
      )}
    </ToggleButtonGroup>
  );

  return (
    <ChartWrapper
      title={title}
      subtitle={subtitle || `Last ${data.length} episodes`}
      loading={loading}
      error={error}
      config={finalConfig}
      actions={actions}
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
            label={{ value: 'Loss', angle: -90, position: 'insideLeft' }}
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

          {/* Loss lines */}
          {visibleLosses.includes('value') && (
            <Line
              type="monotone"
              dataKey="valueLoss"
              stroke={CHART_COLORS.blue}
              strokeWidth={2}
              dot={false}
              name="Value Loss"
            />
          )}
          {visibleLosses.includes('policy') && (
            <Line
              type="monotone"
              dataKey="policyLoss"
              stroke={CHART_COLORS.red}
              strokeWidth={2}
              dot={false}
              name="Policy Loss"
            />
          )}
          {visibleLosses.includes('td') && (
            <Line
              type="monotone"
              dataKey="tdError"
              stroke={CHART_COLORS.green}
              strokeWidth={2}
              dot={false}
              name="TD Error"
            />
          )}
          {visibleLosses.includes('critic') && (
            <Line
              type="monotone"
              dataKey="criticLoss"
              stroke={CHART_COLORS.purple}
              strokeWidth={2}
              dot={false}
              name="Critic Loss"
            />
          )}
          {visibleLosses.includes('actor') && (
            <Line
              type="monotone"
              dataKey="actorLoss"
              stroke={CHART_COLORS.orange}
              strokeWidth={2}
              dot={false}
              name="Actor Loss"
            />
          )}
          {visibleLosses.includes('total') && (
            <Line
              type="monotone"
              dataKey="totalLoss"
              stroke={CHART_COLORS.neutral}
              strokeWidth={2}
              dot={false}
              name="Total Loss"
            />
          )}
        </LineChart>
      </ResponsiveContainer>
    </ChartWrapper>
  );
};

export default LossCurvesChart;
