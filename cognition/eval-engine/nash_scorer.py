"""
Nash Scorer - Multi-agent Nash equilibrium scoring.
"""
import sys
import os
from typing import Dict
import numpy as np

# Import NashEquilibriumSolver from orchestration-agent
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'orchestration-agent'))
from nash import NashEquilibriumSolver

from models import MetricsResult


class NashScorer:
    """Compute Nash equilibrium scores for multi-agent evaluation."""

    def __init__(self, nash_solver: NashEquilibriumSolver):
        self.solver = nash_solver

    async def compute_agent_nash_scores(
        self,
        agent_results: Dict[str, MetricsResult]
    ) -> Dict[str, float]:
        """
        Compute Nash scores for each agent.

        Args:
            agent_results: Dict mapping agent_type -> MetricsResult

        Returns:
            Dict mapping agent_type -> nash_score (0.0-1.0)
        """
        # Build payoff matrix from agent rewards
        payoff_matrix = self.build_payoff_matrix(agent_results)

        # Solve for Nash equilibrium
        equilibrium = self.solver.solve(payoff_matrix)

        # Return Nash scores for each agent
        return {
            agent: equilibrium.scores[i]
            for i, agent in enumerate(agent_results.keys())
        }

    def build_payoff_matrix(
        self,
        agent_results: Dict[str, MetricsResult]
    ) -> np.ndarray:
        """
        Build N×N payoff matrix from agent rewards.

        Matrix[i][j] = reward of agent i when interacting with agent j
        For single-agent envs, use mean_reward on diagonal.

        Returns:
            N×N numpy array
        """
        agents = list(agent_results.keys())
        n = len(agents)
        matrix = np.zeros((n, n))

        for i, agent_i in enumerate(agents):
            for j, agent_j in enumerate(agents):
                if i == j:
                    # Diagonal: agent's own mean reward
                    matrix[i][j] = agent_results[agent_i].mean_reward
                else:
                    # Off-diagonal: normalized interaction reward
                    matrix[i][j] = self.normalize_rewards(
                        agent_results[agent_i].mean_reward,
                        agent_results[agent_j].mean_reward
                    )

        return matrix

    def normalize_rewards(
        self,
        reward_i: float,
        reward_j: float
    ) -> float:
        """
        Normalize rewards for payoff matrix.

        For single-agent environments, use average of rewards.
        """
        return (reward_i + reward_j) / 2.0

    def compute_conflict_score(
        self,
        agent_results: Dict[str, MetricsResult]
    ) -> float:
        """
        Compute conflict score (0.0-1.0) based on reward variance.
        High conflict = high variance across agents.

        Returns:
            Coefficient of variation (std / mean)
        """
        rewards = [result.mean_reward for result in agent_results.values()]
        mean_reward = np.mean(rewards)

        if mean_reward == 0:
            return 0.0

        return float(np.std(rewards) / mean_reward)
