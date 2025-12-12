/**
 * T.A.R.S. Admin Dashboard - Chart Wrapper Component
 *
 * Reusable wrapper component for all charts with consistent styling and error handling.
 */

import React from 'react';
import { Box, Paper, Typography, CircularProgress, Alert } from '@mui/material';
import { ChartConfig } from './types';
import { DEFAULT_CHART_CONFIG } from './utils';

interface ChartWrapperProps {
  title?: string;
  subtitle?: string;
  loading?: boolean;
  error?: string;
  children: React.ReactNode;
  config?: Partial<ChartConfig>;
  actions?: React.ReactNode; // Optional action buttons (refresh, export, etc.)
  height?: number;
}

export const ChartWrapper: React.FC<ChartWrapperProps> = ({
  title,
  subtitle,
  loading = false,
  error,
  children,
  config,
  actions,
  height,
}) => {
  const finalConfig = { ...DEFAULT_CHART_CONFIG, ...config };
  const chartHeight = height || finalConfig.height;

  return (
    <Paper
      elevation={2}
      sx={{
        p: 2,
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
      }}
    >
      {/* Header */}
      {(title || actions) && (
        <Box
          sx={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            mb: 1,
          }}
        >
          <Box>
            {title && (
              <Typography variant="h6" component="h3" gutterBottom>
                {title}
              </Typography>
            )}
            {subtitle && (
              <Typography variant="caption" color="text.secondary">
                {subtitle}
              </Typography>
            )}
          </Box>
          {actions && <Box>{actions}</Box>}
        </Box>
      )}

      {/* Chart Content */}
      <Box
        sx={{
          flex: 1,
          position: 'relative',
          minHeight: chartHeight,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
        }}
      >
        {loading ? (
          <CircularProgress />
        ) : error ? (
          <Alert severity="error" sx={{ width: '100%' }}>
            {error}
          </Alert>
        ) : (
          <Box sx={{ width: '100%', height: '100%' }}>{children}</Box>
        )}
      </Box>
    </Paper>
  );
};

export default ChartWrapper;
