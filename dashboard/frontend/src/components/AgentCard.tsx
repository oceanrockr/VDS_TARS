/**
 * Agent Card Component
 * Displays agent status, metrics, and reward history
 */
import React from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
  Chip,
  LinearProgress,
} from '@mui/material';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

interface AgentCardProps {
  agentId: string;
  agentType: string;
  stateDim: number;
  actionDim: number;
  currentEpisode: number;
  recentRewardMean: number;
  loss: number;
  rewardHistory?: number[];
}

const AgentCard: React.FC<AgentCardProps> = ({
  agentId,
  agentType,
  stateDim,
  actionDim,
  currentEpisode,
  recentRewardMean,
  loss,
  rewardHistory = [],
}) => {
  // Format reward history for chart
  const chartData = rewardHistory.map((reward, index) => ({
    step: index,
    reward,
  }));

  // Get agent color based on type
  const getAgentColor = (type: string) => {
    switch (type) {
      case 'DQN':
        return '#00bcd4';
      case 'A2C':
        return '#4caf50';
      case 'PPO':
        return '#ff9800';
      case 'DDPG':
        return '#9c27b0';
      default:
        return '#607d8b';
    }
  };

  const agentColor = getAgentColor(agentType);

  return (
    <Card sx={{ bgcolor: 'background.paper', borderTop: `4px solid ${agentColor}` }}>
      <CardContent>
        {/* Header */}
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Typography variant="h6" component="div">
            {agentId.toUpperCase()}
          </Typography>
          <Chip label={agentType} size="small" sx={{ bgcolor: agentColor, color: 'white' }} />
        </Box>

        {/* Metrics Grid */}
        <Box sx={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 2, mb: 2 }}>
          <Box>
            <Typography variant="caption" color="text.secondary">
              State Dim
            </Typography>
            <Typography variant="body1">{stateDim}</Typography>
          </Box>
          <Box>
            <Typography variant="caption" color="text.secondary">
              Action Dim
            </Typography>
            <Typography variant="body1">{actionDim}</Typography>
          </Box>
          <Box>
            <Typography variant="caption" color="text.secondary">
              Episode
            </Typography>
            <Typography variant="body1">{currentEpisode}</Typography>
          </Box>
          <Box>
            <Typography variant="caption" color="text.secondary">
              Avg Reward
            </Typography>
            <Typography variant="body1" color={recentRewardMean > 0.7 ? 'success.main' : 'warning.main'}>
              {recentRewardMean.toFixed(3)}
            </Typography>
          </Box>
        </Box>

        {/* Loss Progress */}
        <Box sx={{ mb: 2 }}>
          <Typography variant="caption" color="text.secondary">
            Loss: {loss.toFixed(4)}
          </Typography>
          <LinearProgress
            variant="determinate"
            value={Math.min(loss * 100, 100)}
            sx={{ mt: 0.5, height: 6, borderRadius: 3 }}
          />
        </Box>

        {/* Reward History Chart */}
        {chartData.length > 0 && (
          <Box sx={{ height: 150, mt: 2 }}>
            <Typography variant="caption" color="text.secondary" gutterBottom>
              Reward History (Last {chartData.length} steps)
            </Typography>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                <XAxis dataKey="step" stroke="#888" tick={{ fontSize: 10 }} />
                <YAxis stroke="#888" tick={{ fontSize: 10 }} domain={[0, 1]} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#1a2332', border: '1px solid #333' }}
                  labelStyle={{ color: '#fff' }}
                />
                <Line
                  type="monotone"
                  dataKey="reward"
                  stroke={agentColor}
                  strokeWidth={2}
                  dot={false}
                  animationDuration={300}
                />
              </LineChart>
            </ResponsiveContainer>
          </Box>
        )}
      </CardContent>
    </Card>
  );
};

export default AgentCard;
