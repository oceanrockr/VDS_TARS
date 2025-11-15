"""
DDPG Agent Implementation

Implements Deep Deterministic Policy Gradient for continuous control
in resource management scenarios.

Features:
- Actor-Critic with target networks
- Soft target updates (τ = 0.005)
- Experience replay buffer
- Ornstein-Uhlenbeck exploration noise

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

from .network import DDPGActor, DDPGCritic
from .noise import OUNoise, AdaptiveOUNoise


logger = logging.getLogger(__name__)


class ReplayBuffer:
    """
    Experience replay buffer for DDPG.

    Stores transitions and samples random mini-batches for training.

    Args:
        capacity (int): Maximum number of transitions to store
        state_dim (int): Dimension of state space
        action_dim (int): Dimension of action space
    """

    def __init__(
        self,
        capacity: int = 100000,
        state_dim: int = 20,
        action_dim: int = 1
    ):
        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.position = 0
        self.size = 0

        # Pre-allocate arrays
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """Add transition to buffer."""
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position] = float(done)

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """
        Sample random mini-batch.

        Args:
            batch_size (int): Size of mini-batch

        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
        """
        indices = np.random.randint(0, self.size, size=batch_size)

        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices]
        )

    def __len__(self) -> int:
        return self.size


class DDPGAgent:
    """
    DDPG (Deep Deterministic Policy Gradient) Agent for resource management.

    Features:
    - Continuous action space for resource scaling
    - Actor-Critic architecture with target networks
    - Soft target updates with τ = 0.005
    - Ornstein-Uhlenbeck exploration noise
    - Experience replay

    Args:
        state_dim (int): Dimension of state space
        action_dim (int): Dimension of action space
        actor_lr (float): Learning rate for actor
        critic_lr (float): Learning rate for critic
        gamma (float): Discount factor
        tau (float): Soft update parameter
        buffer_size (int): Replay buffer capacity
        batch_size (int): Mini-batch size
        noise_type (str): Type of noise ('ou', 'adaptive_ou', 'gaussian')
        noise_sigma (float): Noise standard deviation
        device (str): Device to run on ('cpu' or 'cuda')
    """

    def __init__(
        self,
        state_dim: int = 20,
        action_dim: int = 1,
        actor_lr: float = 0.0001,
        critic_lr: float = 0.001,
        gamma: float = 0.99,
        tau: float = 0.005,
        buffer_size: int = 100000,
        batch_size: int = 64,
        noise_type: str = "ou",
        noise_sigma: float = 0.2,
        device: str = "cpu"
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.device = device

        # Actor networks
        self.actor = DDPGActor(
            state_dim=state_dim,
            action_dim=action_dim
        ).to(device)

        self.actor_target = DDPGActor(
            state_dim=state_dim,
            action_dim=action_dim
        ).to(device)

        # Initialize target with same weights
        self.actor_target.load_state_dict(self.actor.state_dict())

        # Critic networks
        self.critic = DDPGCritic(
            state_dim=state_dim,
            action_dim=action_dim
        ).to(device)

        self.critic_target = DDPGCritic(
            state_dim=state_dim,
            action_dim=action_dim
        ).to(device)

        # Initialize target with same weights
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(),
            lr=actor_lr
        )

        self.critic_optimizer = optim.Adam(
            self.critic.parameters(),
            lr=critic_lr,
            weight_decay=1e-2
        )

        # Replay buffer
        self.replay_buffer = ReplayBuffer(
            capacity=buffer_size,
            state_dim=state_dim,
            action_dim=action_dim
        )

        # Exploration noise
        if noise_type == "ou":
            self.noise = OUNoise(
                size=action_dim,
                sigma=noise_sigma
            )
        elif noise_type == "adaptive_ou":
            self.noise = AdaptiveOUNoise(
                size=action_dim,
                sigma_start=noise_sigma
            )
        elif noise_type == "gaussian":
            from .noise import GaussianNoise
            self.noise = GaussianNoise(
                size=action_dim,
                sigma=noise_sigma
            )
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")

        # Statistics tracking
        self.episode_count = 0
        self.step_count = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self.actor_losses = deque(maxlen=100)
        self.critic_losses = deque(maxlen=100)
        self.q_values = deque(maxlen=100)

        logger.info(
            f"DDPG Agent initialized: state_dim={state_dim}, action_dim={action_dim}, "
            f"actor_lr={actor_lr}, critic_lr={critic_lr}, gamma={gamma}, tau={tau}"
        )

    def select_action(
        self,
        state: np.ndarray,
        add_noise: bool = True,
        noise_scale: float = 1.0
    ) -> np.ndarray:
        """
        Select action using current policy.

        Args:
            state (np.ndarray): Current state
            add_noise (bool): Whether to add exploration noise
            noise_scale (float): Scale factor for noise

        Returns:
            np.ndarray: Selected action (clipped to [0, 1])
        """
        self.actor.eval()

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action = self.actor(state_tensor).cpu().numpy()[0]

        self.actor.train()

        # Add exploration noise
        if add_noise:
            noise = self.noise.sample() * noise_scale
            action = action + noise

        # Clip to valid range
        action = np.clip(action, 0.0, 1.0)

        return action

    def store_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """
        Store transition in replay buffer.

        Args:
            state (np.ndarray): State
            action (np.ndarray): Action
            reward (float): Reward
            next_state (np.ndarray): Next state
            done (bool): Episode done flag
        """
        self.replay_buffer.add(state, action, reward, next_state, done)
        self.step_count += 1

    def train_step(self) -> Optional[Dict[str, float]]:
        """
        Perform one training step.

        Returns:
            Dict containing training metrics, or None if buffer too small
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        # Sample mini-batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # ==================== Update Critic ====================

        # Compute target Q-value
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, next_actions)
            target_q = rewards + (1 - dones) * self.gamma * target_q

        # Current Q-value
        current_q = self.critic(states, actions)

        # Critic loss
        critic_loss = F.mse_loss(current_q, target_q)

        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        # ==================== Update Actor ====================

        # Actor loss (negative Q-value)
        actor_actions = self.actor(states)
        actor_loss = -self.critic(states, actor_actions).mean()

        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()

        # ==================== Soft Update Target Networks ====================

        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)

        # Track metrics
        self.actor_losses.append(actor_loss.item())
        self.critic_losses.append(critic_loss.item())
        self.q_values.append(current_q.mean().item())

        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "q_value": current_q.mean().item(),
            "target_q": target_q.mean().item(),
        }

    def _soft_update(
        self,
        source: nn.Module,
        target: nn.Module
    ):
        """
        Soft update target network parameters.

        θ_target = τ * θ_source + (1 - τ) * θ_target

        Args:
            source (nn.Module): Source network
            target (nn.Module): Target network
        """
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                self.tau * source_param.data + (1.0 - self.tau) * target_param.data
            )

    def end_episode(self, episode_reward: float, episode_length: int):
        """
        Called at end of episode to update statistics.

        Args:
            episode_reward (float): Total episode reward
            episode_length (int): Episode length
        """
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)
        self.episode_count += 1
        self.noise.reset()

    def get_statistics(self) -> Dict[str, any]:
        """Get agent statistics."""
        stats = {
            "episode_count": self.episode_count,
            "step_count": self.step_count,
            "buffer_size": len(self.replay_buffer),
            "avg_actor_loss": np.mean(self.actor_losses) if self.actor_losses else None,
            "avg_critic_loss": np.mean(self.critic_losses) if self.critic_losses else None,
            "avg_q_value": np.mean(self.q_values) if self.q_values else None,
        }

        # Add noise sigma if adaptive
        if isinstance(self.noise, AdaptiveOUNoise):
            stats["noise_sigma"] = self.noise.get_sigma()

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
            "actor_state_dict": self.actor.state_dict(),
            "actor_target_state_dict": self.actor_target.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "critic_target_state_dict": self.critic_target.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            "episode_count": self.episode_count,
            "step_count": self.step_count,
            "episode_rewards": list(self.episode_rewards),
            "episode_lengths": list(self.episode_lengths),
        }, filepath)
        logger.info(f"DDPG agent saved to {filepath}")

    def load(self, filepath: str):
        """Load agent state from file."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.actor_target.load_state_dict(checkpoint["actor_target_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.critic_target.load_state_dict(checkpoint["critic_target_state_dict"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
        self.episode_count = checkpoint["episode_count"]
        self.step_count = checkpoint["step_count"]
        self.episode_rewards = checkpoint["episode_rewards"]
        self.episode_lengths = checkpoint["episode_lengths"]
        logger.info(f"DDPG agent loaded from {filepath}")
