"""
Agent Evaluation Worker - Core evaluation execution.
"""
import sys
import os
from typing import Dict, Any, Optional, List
import asyncio
import time
from datetime import datetime
import gymnasium as gym
import numpy as np

# Import agents from orchestration-agent
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'orchestration-agent'))
from dqn import DQNAgent
from a2c import A2CAgent
from ppo import PPOAgent
from ddpg import DDPGAgent

# Import from parent package
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from environment_manager import EnvironmentCache
from metrics_calculator import MetricsCalculator
from regression_detector import RegressionDetector
from models import EpisodeResult, MetricsResult, RegressionResult, BaselineRecord


class AgentEvaluationWorker:
    """Execute agent evaluation in isolated async tasks."""

    def __init__(
        self,
        env_manager: EnvironmentCache,
        metrics_calc: MetricsCalculator,
        regression_detector: RegressionDetector
    ):
        self.env_manager = env_manager
        self.metrics_calc = metrics_calc
        self.regression_detector = regression_detector

    async def evaluate_agent_in_env(
        self,
        agent_type: str,
        agent_state: Dict[str, Any],
        hyperparameters: Dict[str, float],
        environment: str,
        num_episodes: int,
        baseline: Optional[BaselineRecord]
    ) -> Dict[str, Any]:
        """
        Execute full agent evaluation.

        Steps:
        1. Load agent model from state dict
        2. Get/create environment from cache
        3. Run N episodes, collect results
        4. Calculate metrics
        5. Detect regression vs baseline
        6. Return comprehensive results

        Args:
            agent_type: Type of agent (DQN, A2C, PPO, DDPG)
            agent_state: Serialized agent weights
            hyperparameters: Hyperparameters to use
            environment: Environment ID (e.g., "CartPole-v1")
            num_episodes: Number of episodes to run
            baseline: Optional baseline for regression detection

        Returns:
            {
                "environment": str,
                "episodes": List[EpisodeResult],
                "metrics": MetricsResult,
                "regression": RegressionResult,
                "duration_seconds": float
            }
        """
        start_time = time.time()

        # 1. Load agent model
        agent = self._load_agent_model(agent_type, agent_state, hyperparameters)

        # 2. Get environment from cache
        env = await self.env_manager.get_or_create_env(environment)

        # 3. Run episodes and collect results
        episodes = []
        for i in range(num_episodes):
            episode_result = await self._run_episode(agent, env, i)
            episodes.append(episode_result)

        # 4. Compute action distribution for entropy calculation
        action_dist = self._compute_action_distribution(episodes)

        # 5. Calculate metrics
        metrics = self.metrics_calc.compute_all_metrics(episodes, action_dist)

        # 6. Detect regression vs baseline
        if baseline:
            regression = self.regression_detector.should_trigger_rollback(metrics, baseline)
        else:
            regression = RegressionResult(
                detected=False,
                regression_score=0.0,
                should_rollback=False,
                failed_checks=[]
            )

        # 7. Calculate duration
        duration = time.time() - start_time

        return {
            "environment": environment,
            "episodes": episodes,
            "metrics": metrics,
            "regression": regression,
            "duration_seconds": duration
        }

    def _load_agent_model(
        self,
        agent_type: str,
        state_dict: Dict[str, Any],
        hyperparameters: Dict[str, float]
    ) -> Any:
        """
        Load agent (DQN, A2C, PPO, DDPG) from state dict.

        Args:
            agent_type: Type of agent
            state_dict: Serialized agent weights (can be empty for new agent)
            hyperparameters: Hyperparameters to use

        Returns:
            Agent instance (DQNAgent, A2CAgent, PPOAgent, or DDPGAgent)
        """
        if agent_type == "DQN":
            agent = DQNAgent(**hyperparameters)
            if state_dict:
                agent.load_state_dict(state_dict)
            return agent
        elif agent_type == "A2C":
            agent = A2CAgent(**hyperparameters)
            if state_dict:
                agent.load_state_dict(state_dict)
            return agent
        elif agent_type == "PPO":
            agent = PPOAgent(**hyperparameters)
            if state_dict:
                agent.load_state_dict(state_dict)
            return agent
        elif agent_type == "DDPG":
            agent = DDPGAgent(**hyperparameters)
            if state_dict:
                agent.load_state_dict(state_dict)
            return agent
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")

    async def _run_episode(
        self,
        agent: Any,
        env: gym.Env,
        episode_num: int
    ) -> EpisodeResult:
        """
        Run single episode, return reward/steps/loss.

        Args:
            agent: Agent instance
            env: Gymnasium environment
            episode_num: Episode number (for tracking)

        Returns:
            EpisodeResult with total_reward, steps, success, loss
        """
        obs, info = env.reset()
        total_reward = 0.0
        steps = 0
        done = False
        truncated = False
        losses = []

        while not (done or truncated):
            # Select action from agent
            action = agent.select_action(obs)

            # Step environment
            next_obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

            # If agent has training enabled, collect loss
            if hasattr(agent, 'get_loss'):
                loss = agent.get_loss()
                if loss is not None:
                    losses.append(loss)

            obs = next_obs

        # Determine success (environment-specific)
        success = self._is_episode_successful(env, total_reward, steps)

        # Calculate mean loss
        mean_loss = float(np.mean(losses)) if losses else None

        return EpisodeResult(
            episode_num=episode_num,
            total_reward=float(total_reward),
            steps=steps,
            success=success,
            loss=mean_loss
        )

    def _is_episode_successful(
        self,
        env: gym.Env,
        total_reward: float,
        steps: int
    ) -> bool:
        """
        Determine if episode was successful (environment-specific).

        Rules:
        - CartPole-v1: reward >= 195 (solved threshold)
        - LunarLander-v2: reward >= 200
        - MountainCar-v0: steps < 200 (reached goal quickly)
        """
        # Get environment ID
        if hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'spec'):
            env_id = env.unwrapped.spec.id
        else:
            env_id = str(env)

        # Environment-specific success criteria
        if "CartPole" in env_id:
            return total_reward >= 195
        elif "LunarLander" in env_id:
            return total_reward >= 200
        elif "MountainCar" in env_id:
            return steps < 200
        else:
            # Default: use reward threshold
            return total_reward > 0

    def _compute_action_distribution(
        self,
        episodes: List[EpisodeResult]
    ) -> np.ndarray:
        """
        Compute action distribution across all episodes.

        For entropy calculation. Returns array of action counts.

        Note: This is a simplified version that returns uniform distribution.
        In a full implementation, actions would be tracked during episodes.
        """
        # Assume 4 discrete actions (typical for CartPole, etc.)
        # Return uniform distribution as placeholder
        return np.ones(4)
