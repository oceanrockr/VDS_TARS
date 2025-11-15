"""
A2C Agent Implementation

Implements Advantage Actor-Critic algorithm with GAE for consensus agent.

Loss function:
    L = L_policy + 0.5 * L_value - 0.01 * H(π)

Where:
    - L_policy: Policy gradient loss with advantage
    - L_value: Mean squared error for value function
    - H(π): Entropy bonus for exploration

Author: T.A.R.S. Cognitive Team
Version: v0.9.2-alpha
"""

import logging
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .network import A2CNetwork


logger = logging.getLogger(__name__)


class A2CAgent:
    """
    A2C (Advantage Actor-Critic) Agent for consensus decision-making.

    Features:
    - Shared backbone with actor-critic heads
    - Generalized Advantage Estimation (GAE) with λ=0.95
    - Entropy bonus for exploration
    - Combined loss optimization

    Args:
        state_dim (int): Dimension of state space
        action_dim (int): Dimension of action space
        learning_rate (float): Learning rate for optimizer
        gamma (float): Discount factor
        gae_lambda (float): GAE lambda parameter
        value_loss_coef (float): Coefficient for value loss
        entropy_coef (float): Coefficient for entropy bonus
        max_grad_norm (float): Maximum gradient norm for clipping
        device (str): Device to run on ('cpu' or 'cuda')
    """

    def __init__(
        self,
        state_dim: int = 16,
        action_dim: int = 5,
        learning_rate: float = 0.0007,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        device: str = "cpu"
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.device = device

        # Initialize network
        self.network = A2CNetwork(
            state_dim=state_dim,
            action_dim=action_dim
        ).to(device)

        # Optimizer
        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=learning_rate
        )

        # Statistics tracking
        self.episode_count = 0
        self.step_count = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self.training_losses = deque(maxlen=100)
        self.policy_losses = deque(maxlen=100)
        self.value_losses = deque(maxlen=100)
        self.entropies = deque(maxlen=100)

        # Episode buffer
        self.reset_episode_buffer()

        logger.info(
            f"A2C Agent initialized: state_dim={state_dim}, action_dim={action_dim}, "
            f"lr={learning_rate}, gamma={gamma}, gae_lambda={gae_lambda}"
        )

    def reset_episode_buffer(self):
        """Reset episode buffer for collecting trajectories."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

    def select_action(
        self,
        state: np.ndarray,
        deterministic: bool = False
    ) -> Tuple[int, float, float]:
        """
        Select action using current policy.

        Args:
            state (np.ndarray): Current state
            deterministic (bool): If True, select greedy action

        Returns:
            Tuple containing:
                - action (int): Selected action
                - log_prob (float): Log probability of action
                - value (float): State value estimate
        """
        self.network.eval()

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action_probs, value = self.network(state_tensor)

            if deterministic:
                action = torch.argmax(action_probs, dim=-1)
            else:
                dist = torch.distributions.Categorical(action_probs)
                action = dist.sample()

            # Calculate log probability
            dist = torch.distributions.Categorical(action_probs)
            log_prob = dist.log_prob(action)

        self.network.train()

        return (
            action.item(),
            log_prob.item(),
            value.item()
        )

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        done: bool,
        log_prob: float,
        value: float
    ):
        """
        Store transition in episode buffer.

        Args:
            state (np.ndarray): State
            action (int): Action taken
            reward (float): Reward received
            done (bool): Episode done flag
            log_prob (float): Log probability of action
            value (float): Value estimate
        """
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.step_count += 1

    def compute_gae(
        self,
        next_value: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Generalized Advantage Estimation (GAE).

        GAE formula:
            δ_t = r_t + γ*V(s_{t+1}) - V(s_t)
            A_t = Σ_{l=0}^∞ (γλ)^l * δ_{t+l}

        Args:
            next_value (float): Value of next state (0 if terminal)

        Returns:
            Tuple containing:
                - advantages (np.ndarray): Computed advantages
                - returns (np.ndarray): Discounted returns
        """
        advantages = []
        gae = 0.0

        # Reverse iteration through episode
        values = self.values + [next_value]

        for t in reversed(range(len(self.rewards))):
            # TD error
            delta = (
                self.rewards[t] +
                self.gamma * values[t + 1] * (1 - self.dones[t]) -
                values[t]
            )

            # GAE update
            gae = delta + self.gamma * self.gae_lambda * (1 - self.dones[t]) * gae
            advantages.insert(0, gae)

        # Convert to numpy arrays
        advantages = np.array(advantages, dtype=np.float32)
        returns = advantages + np.array(self.values, dtype=np.float32)

        return advantages, returns

    def train_step(
        self,
        next_state: Optional[np.ndarray] = None,
        normalize_advantages: bool = True
    ) -> Dict[str, float]:
        """
        Perform one training step using collected trajectories.

        Args:
            next_state (np.ndarray, optional): Next state for bootstrap
            normalize_advantages (bool): Whether to normalize advantages

        Returns:
            Dict containing training metrics
        """
        if len(self.states) == 0:
            logger.warning("No transitions collected, skipping training step")
            return {}

        # Compute next state value for bootstrapping
        if next_state is not None and not self.dones[-1]:
            with torch.no_grad():
                next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
                _, next_value = self.network(next_state_tensor)
                next_value = next_value.item()
        else:
            next_value = 0.0

        # Compute GAE
        advantages, returns = self.compute_gae(next_value)

        # Normalize advantages
        if normalize_advantages and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Convert to tensors
        states_tensor = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions_tensor = torch.LongTensor(self.actions).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        old_log_probs_tensor = torch.FloatTensor(self.log_probs).to(self.device)

        # Forward pass
        log_probs, entropy, values = self.network.evaluate_actions(
            states_tensor,
            actions_tensor
        )

        # Policy loss (negative because we want to maximize)
        policy_loss = -(log_probs * advantages_tensor).mean()

        # Value loss (MSE)
        value_loss = F.mse_loss(values.squeeze(), returns_tensor)

        # Entropy bonus (negative because we want to maximize entropy)
        entropy_loss = -entropy.mean()

        # Total loss
        total_loss = (
            policy_loss +
            self.value_loss_coef * value_loss +
            self.entropy_coef * entropy_loss
        )

        # Optimization step
        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
        self.optimizer.step()

        # Track statistics
        self.training_losses.append(total_loss.item())
        self.policy_losses.append(policy_loss.item())
        self.value_losses.append(value_loss.item())
        self.entropies.append(entropy.mean().item())

        # Store episode statistics
        episode_reward = sum(self.rewards)
        episode_length = len(self.rewards)
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)
        self.episode_count += 1

        # Metrics
        metrics = {
            "total_loss": total_loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.mean().item(),
            "episode_reward": episode_reward,
            "episode_length": episode_length,
            "mean_advantage": advantages.mean(),
            "mean_return": returns.mean(),
        }

        # Reset episode buffer
        self.reset_episode_buffer()

        return metrics

    def get_statistics(self) -> Dict[str, any]:
        """
        Get agent statistics.

        Returns:
            Dict containing various statistics
        """
        stats = {
            "episode_count": self.episode_count,
            "step_count": self.step_count,
            "avg_loss": np.mean(self.training_losses) if self.training_losses else None,
            "avg_policy_loss": np.mean(self.policy_losses) if self.policy_losses else None,
            "avg_value_loss": np.mean(self.value_losses) if self.value_losses else None,
            "avg_entropy": np.mean(self.entropies) if self.entropies else None,
        }

        if self.episode_rewards:
            recent_10 = self.episode_rewards[-10:]
            recent_100 = self.episode_rewards[-100:]

            stats["avg_reward_last_10"] = np.mean(recent_10)
            stats["avg_reward_last_100"] = np.mean(recent_100)
            stats["max_reward"] = max(self.episode_rewards)
            stats["min_reward"] = min(self.episode_rewards)

        if self.episode_lengths:
            stats["avg_episode_length"] = np.mean(self.episode_lengths[-100:])

        return stats

    def save(self, filepath: str):
        """Save agent state to file."""
        torch.save({
            "network_state_dict": self.network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "episode_count": self.episode_count,
            "step_count": self.step_count,
            "episode_rewards": list(self.episode_rewards),
            "episode_lengths": list(self.episode_lengths),
        }, filepath)
        logger.info(f"A2C agent saved to {filepath}")

    def load(self, filepath: str):
        """Load agent state from file."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.network.load_state_dict(checkpoint["network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.episode_count = checkpoint["episode_count"]
        self.step_count = checkpoint["step_count"]
        self.episode_rewards = checkpoint["episode_rewards"]
        self.episode_lengths = checkpoint["episode_lengths"]
        logger.info(f"A2C agent loaded from {filepath}")


# Import F for MSE loss
import torch.nn.functional as F
