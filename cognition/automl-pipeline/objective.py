"""
T.A.R.S. AutoML Real Objective Function
Integrates with multi-agent orchestration for actual training-based evaluation

This module provides the real objective function that Optuna uses to evaluate
hyperparameters by running actual agent training episodes.

Author: T.A.R.S. Cognitive Team
Version: v0.9.4-alpha
"""

import logging
import time
from typing import Dict, Any, Callable
import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class MultiAgentObjective:
    """
    Real objective function for multi-agent hyperparameter optimization.

    Instead of using mock rewards, this class trains agents for a fixed number
    of episodes and returns the actual validation performance.
    """

    def __init__(
        self,
        n_train_episodes: int = 50,
        n_eval_episodes: int = 10,
        max_steps_per_episode: int = 200,
        use_quick_mode: bool = False,
    ):
        """
        Initialize objective function.

        Args:
            n_train_episodes: Number of training episodes
            n_eval_episodes: Number of evaluation episodes
            max_steps_per_episode: Maximum steps per episode
            use_quick_mode: If True, use fewer episodes for faster optimization
        """
        self.n_train_episodes = n_train_episodes if not use_quick_mode else 10
        self.n_eval_episodes = n_eval_episodes if not use_quick_mode else 5
        self.max_steps_per_episode = max_steps_per_episode

        logger.info(
            f"MultiAgentObjective initialized: "
            f"train_eps={self.n_train_episodes}, "
            f"eval_eps={self.n_eval_episodes}, "
            f"quick_mode={use_quick_mode}"
        )

    def create_dqn_objective(self, state_dim: int = 32, action_dim: int = 10) -> Callable:
        """
        Create objective function for DQN agent.

        Args:
            state_dim: State space dimension
            action_dim: Action space dimension

        Returns:
            Callable that takes hyperparameters and returns validation reward
        """
        def objective_fn(params: Dict[str, Any]) -> float:
            """Train and evaluate DQN with given hyperparameters."""
            try:
                from cognition.orchestration_agent.dqn import DQNAgent

                # Create agent with trial hyperparameters
                agent = DQNAgent(
                    state_dim=state_dim,
                    action_dim=action_dim,
                    learning_rate=params.get("learning_rate", 0.001),
                    gamma=params.get("gamma", 0.95),
                    epsilon_start=params.get("epsilon_start", 1.0),
                    epsilon_end=params.get("epsilon_end", 0.01),
                    epsilon_decay=params.get("epsilon_decay", 0.995),
                    batch_size=params.get("batch_size", 64),
                    buffer_size=params.get("buffer_size", 10000),
                    target_update_freq=params.get("target_update", 10),
                    use_dueling=params.get("double_dqn", False),
                    use_prioritized_replay=params.get("prioritized_replay", True),
                )

                # Train agent
                train_rewards = self._train_dqn_agent(agent, self.n_train_episodes)

                # Evaluate agent
                eval_rewards = self._evaluate_agent(
                    agent,
                    lambda s: agent.select_action(s, explore=False),
                    self.n_eval_episodes
                )

                # Return mean evaluation reward
                return float(np.mean(eval_rewards))

            except Exception as e:
                logger.error(f"DQN objective failed: {e}")
                return -1000.0  # Return very low score on failure

        return objective_fn

    def create_a2c_objective(self, state_dim: int = 16, action_dim: int = 5) -> Callable:
        """Create objective function for A2C agent."""
        def objective_fn(params: Dict[str, Any]) -> float:
            """Train and evaluate A2C with given hyperparameters."""
            try:
                from cognition.orchestration_agent.a2c import A2CAgent

                agent = A2CAgent(
                    state_dim=state_dim,
                    action_dim=action_dim,
                    learning_rate=params.get("learning_rate", 0.0007),
                    gamma=params.get("gamma", 0.99),
                    gae_lambda=params.get("gae_lambda", 0.95),
                )

                # Train
                train_rewards = self._train_actor_critic_agent(
                    agent,
                    self.n_train_episodes,
                    params.get("n_steps", 5)
                )

                # Evaluate
                eval_rewards = self._evaluate_agent(
                    agent,
                    lambda s: agent.select_action(s)[0],  # Return action only
                    self.n_eval_episodes
                )

                return float(np.mean(eval_rewards))

            except Exception as e:
                logger.error(f"A2C objective failed: {e}")
                return -1000.0

        return objective_fn

    def create_ppo_objective(self, state_dim: int = 24, action_dim: int = 8) -> Callable:
        """Create objective function for PPO agent."""
        def objective_fn(params: Dict[str, Any]) -> float:
            """Train and evaluate PPO with given hyperparameters."""
            try:
                from cognition.orchestration_agent.ppo import PPOAgent

                agent = PPOAgent(
                    state_dim=state_dim,
                    action_dim=action_dim,
                    learning_rate=params.get("learning_rate", 0.0003),
                    gamma=params.get("gamma", 0.99),
                    clip_epsilon=params.get("clip_epsilon", 0.2),
                    n_epochs=params.get("n_epochs", 10),
                )

                # Train
                train_rewards = self._train_actor_critic_agent(
                    agent,
                    self.n_train_episodes,
                    params.get("batch_size", 64)
                )

                # Evaluate
                eval_rewards = self._evaluate_agent(
                    agent,
                    lambda s: agent.select_action(s)[0],
                    self.n_eval_episodes
                )

                return float(np.mean(eval_rewards))

            except Exception as e:
                logger.error(f"PPO objective failed: {e}")
                return -1000.0

        return objective_fn

    def create_ddpg_objective(self, state_dim: int = 20, action_dim: int = 1) -> Callable:
        """Create objective function for DDPG agent."""
        def objective_fn(params: Dict[str, Any]) -> float:
            """Train and evaluate DDPG with given hyperparameters."""
            try:
                from cognition.orchestration_agent.ddpg import DDPGAgent

                agent = DDPGAgent(
                    state_dim=state_dim,
                    action_dim=action_dim,
                    actor_lr=params.get("actor_lr", 0.0001),
                    critic_lr=params.get("critic_lr", 0.001),
                    gamma=params.get("gamma", 0.99),
                    tau=params.get("tau", 0.005),
                )

                # Train
                train_rewards = self._train_ddpg_agent(agent, self.n_train_episodes)

                # Evaluate
                eval_rewards = self._evaluate_agent(
                    agent,
                    lambda s: agent.select_action(s, add_noise=False),
                    self.n_eval_episodes
                )

                return float(np.mean(eval_rewards))

            except Exception as e:
                logger.error(f"DDPG objective failed: {e}")
                return -1000.0

        return objective_fn

    def _train_dqn_agent(self, agent, n_episodes: int) -> list:
        """
        Train DQN agent in a simulated environment.

        This uses a simplified multi-agent coordination environment.
        """
        episode_rewards = []

        for episode in range(n_episodes):
            state = self._reset_environment(agent.state_dim)
            episode_reward = 0
            done = False
            steps = 0

            while not done and steps < self.max_steps_per_episode:
                # Select action
                action = agent.select_action(state)

                # Simulate environment step
                next_state, reward, done = self._step_environment(
                    state, action, agent.action_dim
                )

                # Store transition
                agent.store_transition(state, action, reward, next_state, done)

                # Train
                if len(agent.memory) >= agent.batch_size:
                    loss = agent.train_step()

                state = next_state
                episode_reward += reward
                steps += 1

            episode_rewards.append(episode_reward)

            # Decay epsilon
            agent.epsilon = max(
                agent.epsilon_end,
                agent.epsilon * agent.epsilon_decay
            )

        return episode_rewards

    def _train_actor_critic_agent(self, agent, n_episodes: int, n_steps: int) -> list:
        """Train Actor-Critic agent (A2C/PPO)."""
        episode_rewards = []

        for episode in range(n_episodes):
            state = self._reset_environment(agent.state_dim)
            episode_reward = 0
            done = False
            steps = 0

            # Collect trajectories
            states, actions, rewards, log_probs, values = [], [], [], [], []

            while not done and steps < self.max_steps_per_episode:
                # Select action
                action, log_prob, value = agent.select_action(state)

                # Simulate environment step
                next_state, reward, done = self._step_environment(
                    state, action, agent.action_dim
                )

                # Store
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                log_probs.append(log_prob)
                values.append(value)

                state = next_state
                episode_reward += reward
                steps += 1

                # Update every n_steps
                if len(states) >= n_steps or done:
                    agent.train_step(
                        states, actions, rewards, log_probs, values, done
                    )
                    states, actions, rewards, log_probs, values = [], [], [], [], []

            episode_rewards.append(episode_reward)

        return episode_rewards

    def _train_ddpg_agent(self, agent, n_episodes: int) -> list:
        """Train DDPG agent."""
        episode_rewards = []

        for episode in range(n_episodes):
            state = self._reset_environment(agent.state_dim)
            episode_reward = 0
            done = False
            steps = 0

            while not done and steps < self.max_steps_per_episode:
                # Select action (continuous)
                action = agent.select_action(state, add_noise=True)

                # Simulate environment step
                next_state, reward, done = self._step_continuous_environment(
                    state, action
                )

                # Store transition
                agent.store_transition(state, action, reward, next_state, done)

                # Train
                if len(agent.memory) >= agent.batch_size:
                    agent.train_step()

                state = next_state
                episode_reward += reward
                steps += 1

            episode_rewards.append(episode_reward)

        return episode_rewards

    def _evaluate_agent(
        self,
        agent,
        policy_fn: Callable,
        n_episodes: int
    ) -> list:
        """
        Evaluate agent performance without training.

        Args:
            agent: Agent instance
            policy_fn: Function that takes state and returns action
            n_episodes: Number of evaluation episodes

        Returns:
            List of episode rewards
        """
        eval_rewards = []

        for episode in range(n_episodes):
            state = self._reset_environment(agent.state_dim)
            episode_reward = 0
            done = False
            steps = 0

            while not done and steps < self.max_steps_per_episode:
                # Get action from policy (no exploration)
                action = policy_fn(state)

                # Step environment
                if isinstance(action, (int, np.integer)):
                    next_state, reward, done = self._step_environment(
                        state, action, agent.action_dim
                    )
                else:
                    next_state, reward, done = self._step_continuous_environment(
                        state, action
                    )

                state = next_state
                episode_reward += reward
                steps += 1

            eval_rewards.append(episode_reward)

        return eval_rewards

    def _reset_environment(self, state_dim: int) -> np.ndarray:
        """
        Reset simulated multi-agent environment.

        Returns:
            Initial state vector
        """
        # Simulate initial state (e.g., system metrics, agent statuses)
        return np.random.randn(state_dim) * 0.5

    def _step_environment(
        self,
        state: np.ndarray,
        action: int,
        action_dim: int
    ) -> tuple:
        """
        Simulate one step in discrete action environment.

        Args:
            state: Current state
            action: Discrete action
            action_dim: Action space dimension

        Returns:
            (next_state, reward, done)
        """
        # Simplified multi-agent coordination dynamics
        # In production, this would interface with actual orchestration service

        # State transition (simplified)
        next_state = state + np.random.randn(len(state)) * 0.1
        next_state += (action / action_dim - 0.5) * 0.3  # Action influence

        # Reward function (multi-agent coordination objectives)
        # 1. System stability (low variance)
        stability_reward = -np.std(next_state) * 0.5

        # 2. Performance (high mean, clipped)
        performance_reward = np.clip(np.mean(next_state), -1, 1)

        # 3. Action efficiency (penalize extreme actions)
        efficiency_penalty = -abs(action - action_dim / 2) / action_dim * 0.2

        reward = stability_reward + performance_reward + efficiency_penalty

        # Episode termination (random or based on state)
        done = np.random.rand() < 0.02 or np.max(np.abs(next_state)) > 5.0

        return next_state, reward, done

    def _step_continuous_environment(
        self,
        state: np.ndarray,
        action: np.ndarray
    ) -> tuple:
        """Simulate one step in continuous action environment."""
        # State transition
        next_state = state + np.random.randn(len(state)) * 0.1
        next_state += action * 0.5  # Action influence

        # Reward (resource allocation optimization)
        # Penalize over/under allocation
        target_allocation = 0.7
        allocation_error = np.abs(action[0] - target_allocation)
        reward = 1.0 - allocation_error

        # Penalize instability
        reward -= np.std(next_state) * 0.3

        # Episode termination
        done = np.random.rand() < 0.02 or np.max(np.abs(next_state)) > 5.0

        return next_state, reward, done


# ============================================================================
# Factory Functions
# ============================================================================

def create_objective(agent_type: str, use_quick_mode: bool = False) -> Callable:
    """
    Create objective function for a given agent type.

    Args:
        agent_type: One of ['dqn', 'a2c', 'ppo', 'ddpg']
        use_quick_mode: If True, use fewer episodes for faster trials

    Returns:
        Callable objective function for Optuna

    Raises:
        ValueError: If agent_type is unknown
    """
    objective_factory = MultiAgentObjective(use_quick_mode=use_quick_mode)

    if agent_type == "dqn":
        return objective_factory.create_dqn_objective(state_dim=32, action_dim=10)
    elif agent_type == "a2c":
        return objective_factory.create_a2c_objective(state_dim=16, action_dim=5)
    elif agent_type == "ppo":
        return objective_factory.create_ppo_objective(state_dim=24, action_dim=8)
    elif agent_type == "ddpg":
        return objective_factory.create_ddpg_objective(state_dim=20, action_dim=1)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
