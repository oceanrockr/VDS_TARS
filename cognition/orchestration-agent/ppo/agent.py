"""
PPO Agent Implementation

Implements Proximal Policy Optimization with clipped surrogate objective
for ethical oversight and fairness-aware decision-making.

Loss function:
    L = L_CLIP + c1 * L_VF - c2 * S[π](s)

Where:
    - L_CLIP: Clipped surrogate objective
    - L_VF: Value function loss
    - S[π]: Entropy bonus

Author: T.A.R.S. Cognitive Team
Version: v0.9.2-alpha
"""

import logging
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .network import PPONetwork


logger = logging.getLogger(__name__)


class RolloutBuffer:
    """
    Buffer for storing rollout trajectories.

    Stores complete trajectories for PPO updates with advantages
    and returns pre-computed.
    """

    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.log_probs = []
        self.advantages = []
        self.returns = []

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        done: bool,
        value: float,
        log_prob: float
    ):
        """Add transition to buffer."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
        self.log_probs.append(log_prob)

    def compute_returns_and_advantages(
        self,
        last_value: float,
        gamma: float = 0.99,
        gae_lambda: float = 0.95
    ):
        """Compute returns and advantages using GAE."""
        advantages = []
        gae = 0.0

        values = self.values + [last_value]

        for t in reversed(range(len(self.rewards))):
            delta = (
                self.rewards[t] +
                gamma * values[t + 1] * (1 - self.dones[t]) -
                values[t]
            )
            gae = delta + gamma * gae_lambda * (1 - self.dones[t]) * gae
            advantages.insert(0, gae)

        self.advantages = advantages
        self.returns = [adv + val for adv, val in zip(advantages, self.values)]

    def get(self) -> Dict[str, np.ndarray]:
        """Get all data as numpy arrays."""
        return {
            "states": np.array(self.states, dtype=np.float32),
            "actions": np.array(self.actions, dtype=np.int64),
            "log_probs": np.array(self.log_probs, dtype=np.float32),
            "advantages": np.array(self.advantages, dtype=np.float32),
            "returns": np.array(self.returns, dtype=np.float32),
            "values": np.array(self.values, dtype=np.float32),
        }

    def clear(self):
        """Clear all buffers."""
        self.__init__()

    def __len__(self):
        return len(self.states)


class PPOAgent:
    """
    PPO (Proximal Policy Optimization) Agent for ethical oversight.

    Features:
    - Clipped surrogate objective (ε = 0.2)
    - Multiple epochs of mini-batch updates
    - KL-divergence monitoring and adaptive clipping
    - Early stopping based on KL divergence
    - Value function clipping (optional)
    - Entropy bonus for exploration

    Args:
        state_dim (int): Dimension of state space
        action_dim (int): Dimension of action space
        learning_rate (float): Learning rate for optimizer
        gamma (float): Discount factor
        gae_lambda (float): GAE lambda parameter
        clip_epsilon (float): PPO clipping parameter
        value_loss_coef (float): Coefficient for value loss
        entropy_coef (float): Coefficient for entropy bonus
        max_grad_norm (float): Maximum gradient norm for clipping
        n_epochs (int): Number of epochs per update
        batch_size (int): Mini-batch size for updates
        target_kl (float): Target KL divergence for early stopping
        clip_value_loss (bool): Whether to clip value loss
        device (str): Device to run on ('cpu' or 'cuda')
    """

    def __init__(
        self,
        state_dim: int = 24,
        action_dim: int = 8,
        learning_rate: float = 0.0003,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        n_epochs: int = 10,
        batch_size: int = 64,
        target_kl: Optional[float] = 0.015,
        clip_value_loss: bool = False,
        device: str = "cpu"
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.target_kl = target_kl
        self.clip_value_loss = clip_value_loss
        self.device = device

        # Initialize network
        self.network = PPONetwork(
            state_dim=state_dim,
            action_dim=action_dim
        ).to(device)

        # Optimizer
        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=learning_rate,
            eps=1e-5
        )

        # Rollout buffer
        self.buffer = RolloutBuffer()

        # Statistics tracking
        self.episode_count = 0
        self.step_count = 0
        self.update_count = 0
        self.episode_rewards = []
        self.episode_lengths = []

        # Training metrics
        self.training_losses = deque(maxlen=100)
        self.policy_losses = deque(maxlen=100)
        self.value_losses = deque(maxlen=100)
        self.entropies = deque(maxlen=100)
        self.kl_divergences = deque(maxlen=100)
        self.clip_fractions = deque(maxlen=100)
        self.explained_variances = deque(maxlen=100)

        logger.info(
            f"PPO Agent initialized: state_dim={state_dim}, action_dim={action_dim}, "
            f"lr={learning_rate}, gamma={gamma}, clip_epsilon={clip_epsilon}"
        )

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

            if deterministic:
                action_probs, value = self.network(state_tensor)
                action = torch.argmax(action_probs, dim=-1)
                dist = torch.distributions.Categorical(action_probs)
                log_prob = dist.log_prob(action)
            else:
                action, log_prob, _, value = self.network.get_action_and_value(
                    state_tensor
                )

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
        Store transition in rollout buffer.

        Args:
            state (np.ndarray): State
            action (int): Action taken
            reward (float): Reward received
            done (bool): Episode done flag
            log_prob (float): Log probability of action
            value (float): Value estimate
        """
        self.buffer.add(state, action, reward, done, value, log_prob)
        self.step_count += 1

    def update(
        self,
        next_state: Optional[np.ndarray] = None,
        normalize_advantages: bool = True
    ) -> Dict[str, float]:
        """
        Perform PPO update using collected rollouts.

        Args:
            next_state (np.ndarray, optional): Next state for bootstrap
            normalize_advantages (bool): Whether to normalize advantages

        Returns:
            Dict containing training metrics
        """
        if len(self.buffer) == 0:
            logger.warning("No transitions collected, skipping update")
            return {}

        # Compute last value for bootstrapping
        if next_state is not None and not self.buffer.dones[-1]:
            with torch.no_grad():
                next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
                _, last_value = self.network(next_state_tensor)
                last_value = last_value.item()
        else:
            last_value = 0.0

        # Compute returns and advantages
        self.buffer.compute_returns_and_advantages(
            last_value,
            self.gamma,
            self.gae_lambda
        )

        # Get buffer data
        data = self.buffer.get()

        # Normalize advantages
        if normalize_advantages:
            data["advantages"] = (
                (data["advantages"] - data["advantages"].mean()) /
                (data["advantages"].std() + 1e-8)
            )

        # Convert to tensors
        states = torch.FloatTensor(data["states"]).to(self.device)
        actions = torch.LongTensor(data["actions"]).to(self.device)
        old_log_probs = torch.FloatTensor(data["log_probs"]).to(self.device)
        advantages = torch.FloatTensor(data["advantages"]).to(self.device)
        returns = torch.FloatTensor(data["returns"]).to(self.device)
        old_values = torch.FloatTensor(data["values"]).to(self.device)

        # Multiple epochs of optimization
        total_loss_sum = 0.0
        policy_loss_sum = 0.0
        value_loss_sum = 0.0
        entropy_sum = 0.0
        kl_sum = 0.0
        clip_frac_sum = 0.0
        n_updates = 0

        for epoch in range(self.n_epochs):
            # Generate random indices for mini-batches
            indices = np.arange(len(states))
            np.random.shuffle(indices)

            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]

                # Get mini-batch
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                batch_old_values = old_values[batch_indices]

                # Evaluate actions
                log_probs, entropy, values = self.network.evaluate_actions(
                    batch_states,
                    batch_actions
                )

                # Policy loss (clipped surrogate objective)
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(
                    ratio,
                    1.0 - self.clip_epsilon,
                    1.0 + self.clip_epsilon
                ) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                if self.clip_value_loss:
                    # Clipped value loss
                    values_clipped = batch_old_values + torch.clamp(
                        values.squeeze() - batch_old_values,
                        -self.clip_epsilon,
                        self.clip_epsilon
                    )
                    value_loss_1 = F.mse_loss(values.squeeze(), batch_returns)
                    value_loss_2 = F.mse_loss(values_clipped, batch_returns)
                    value_loss = torch.max(value_loss_1, value_loss_2)
                else:
                    value_loss = F.mse_loss(values.squeeze(), batch_returns)

                # Entropy loss
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
                nn.utils.clip_grad_norm_(
                    self.network.parameters(),
                    self.max_grad_norm
                )
                self.optimizer.step()

                # Track metrics
                with torch.no_grad():
                    # KL divergence (approximate)
                    kl = (batch_old_log_probs - log_probs).mean()

                    # Clip fraction
                    clip_frac = ((ratio - 1.0).abs() > self.clip_epsilon).float().mean()

                total_loss_sum += total_loss.item()
                policy_loss_sum += policy_loss.item()
                value_loss_sum += value_loss.item()
                entropy_sum += entropy.mean().item()
                kl_sum += kl.item()
                clip_frac_sum += clip_frac.item()
                n_updates += 1

            # Early stopping based on KL divergence
            if self.target_kl is not None:
                avg_kl = kl_sum / n_updates
                if avg_kl > 1.5 * self.target_kl:
                    logger.info(f"Early stopping at epoch {epoch + 1} due to KL={avg_kl:.4f}")
                    break

        # Average metrics
        avg_total_loss = total_loss_sum / n_updates
        avg_policy_loss = policy_loss_sum / n_updates
        avg_value_loss = value_loss_sum / n_updates
        avg_entropy = entropy_sum / n_updates
        avg_kl = kl_sum / n_updates
        avg_clip_frac = clip_frac_sum / n_updates

        # Explained variance
        with torch.no_grad():
            y_pred = old_values.cpu().numpy()
            y_true = returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = 1 - np.var(y_true - y_pred) / (var_y + 1e-8)

        # Track statistics
        self.training_losses.append(avg_total_loss)
        self.policy_losses.append(avg_policy_loss)
        self.value_losses.append(avg_value_loss)
        self.entropies.append(avg_entropy)
        self.kl_divergences.append(avg_kl)
        self.clip_fractions.append(avg_clip_frac)
        self.explained_variances.append(explained_var)
        self.update_count += 1

        # Store episode statistics
        episode_reward = sum(self.buffer.rewards)
        episode_length = len(self.buffer.rewards)
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)
        self.episode_count += 1

        # Clear buffer
        self.buffer.clear()

        # Return metrics
        metrics = {
            "total_loss": avg_total_loss,
            "policy_loss": avg_policy_loss,
            "value_loss": avg_value_loss,
            "entropy": avg_entropy,
            "kl_divergence": avg_kl,
            "clip_fraction": avg_clip_frac,
            "explained_variance": explained_var,
            "episode_reward": episode_reward,
            "episode_length": episode_length,
            "n_updates": n_updates,
        }

        return metrics

    def get_statistics(self) -> Dict[str, any]:
        """Get agent statistics."""
        stats = {
            "episode_count": self.episode_count,
            "step_count": self.step_count,
            "update_count": self.update_count,
            "avg_loss": np.mean(self.training_losses) if self.training_losses else None,
            "avg_policy_loss": np.mean(self.policy_losses) if self.policy_losses else None,
            "avg_value_loss": np.mean(self.value_losses) if self.value_losses else None,
            "avg_entropy": np.mean(self.entropies) if self.entropies else None,
            "avg_kl": np.mean(self.kl_divergences) if self.kl_divergences else None,
            "avg_clip_fraction": np.mean(self.clip_fractions) if self.clip_fractions else None,
            "avg_explained_var": np.mean(self.explained_variances) if self.explained_variances else None,
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
            "update_count": self.update_count,
            "episode_rewards": list(self.episode_rewards),
            "episode_lengths": list(self.episode_lengths),
        }, filepath)
        logger.info(f"PPO agent saved to {filepath}")

    def load(self, filepath: str):
        """Load agent state from file."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.network.load_state_dict(checkpoint["network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.episode_count = checkpoint["episode_count"]
        self.step_count = checkpoint["step_count"]
        self.update_count = checkpoint["update_count"]
        self.episode_rewards = checkpoint["episode_rewards"]
        self.episode_lengths = checkpoint["episode_lengths"]
        logger.info(f"PPO agent loaded from {filepath}")
