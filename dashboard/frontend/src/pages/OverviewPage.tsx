import React from 'react';
import { Grid, Typography } from '@mui/material';
import AgentCard from '../components/AgentCard';

const OverviewPage: React.FC = () => {
  const mockAgents = [
    { agentId: 'policy', agentType: 'DQN', stateDim: 32, actionDim: 10, currentEpisode: 150, recentRewardMean: 0.78, loss: 0.023, rewardHistory: Array(50).fill(0).map((_, i) => 0.5 + Math.random() * 0.3) },
    { agentId: 'consensus', agentType: 'A2C', stateDim: 16, actionDim: 5, currentEpisode: 150, recentRewardMean: 0.82, loss: 0.015, rewardHistory: Array(50).fill(0).map((_, i) => 0.6 + Math.random() * 0.3) },
    { agentId: 'ethical', agentType: 'PPO', stateDim: 24, actionDim: 8, currentEpisode: 150, recentRewardMean: 0.75, loss: 0.031, rewardHistory: Array(50).fill(0).map((_, i) => 0.55 + Math.random() * 0.25) },
    { agentId: 'resource', agentType: 'DDPG', stateDim: 20, actionDim: 1, currentEpisode: 150, recentRewardMean: 0.88, loss: 0.012, rewardHistory: Array(50).fill(0).map((_, i) => 0.7 + Math.random() * 0.25) },
  ];

  return (
    <>
      <Typography variant="h4" gutterBottom>T.A.R.S. Multi-Agent Orchestration</Typography>
      <Grid container spacing={3}>
        {mockAgents.map(agent => (
          <Grid item xs={12} md={6} lg={3} key={agent.agentId}>
            <AgentCard {...agent} />
          </Grid>
        ))}
      </Grid>
    </>
  );
};

export default OverviewPage;
