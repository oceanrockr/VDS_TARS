"""
DQN Agent with Double DQN and Prioritized Replay

Implements the core training loop for deep Q-learning with:
- Double DQN for reduced overestimation
- Prioritized experience replay
- Target network with soft updates

Author: T.A.R.S. Cognitive Team
Version: v0.9.1-alpha
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Optional, Tuple
import logging
from pathlib import Path

from .network import DQNNetwork, DuelingDQNNetwork
from .memory_buffer import PrioritizedReplayBuffer, ReplayBuffer


logger = logging.getLogger(__name__)


class DQNAgent:
    """
    Deep Q-Network agent with Double DQN and prioritized replay.

    Attributes:
        state_dim (int): State space dimensionality
        action_dim (int): Number of discrete actions
        device (torch.device): Computing device (CPU/GPU)
        policy_net (nn.Module): Main Q-network for action selection
        target_net (nn.Module): Target Q-network for stable training
        optimizer (optim.Optimizer): Network optimizer
        memory (PrioritizedReplayBuffer): Experience replay buffer
    """

    def __init__(
        self,
        state_dim: int = 32,
        action_dim: int = 10,
        learning_rate: float = 0.001,
        gamma: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        batch_size: int = 64,
        buffer_size: int = 10000,
        target_update_freq: int = 10,
        tau: float = 0.005,
        use_dueling: bool = False,
        use_prioritized_replay: bool = True,
        device: Optional[str] = None
    ):
        """
        Initialize DQN agent.

        Args:
            state_dim: State vector dimension
            action_dim: Number of actions
            learning_rate: Adam optimizer learning rate
            gamma: Discount factor for future rewards
            epsilon_start: Initial exploration rate
            epsilon_end: Minimum exploration rate
            epsilon_decay: Epsilon decay factor per episode
            batch_size: Training batch size
            buffer_size: Replay buffer capacity
            target_update_freq: Episodes between target network updates
            tau: Soft update parameter (1.0 = hard update)
            use_dueling: Use dueling DQN architecture
            use_prioritized_replay: Use prioritized experience replay
            device: Computing device ('cuda' or 'cpu')
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.tau = tau
        self.use_prioritized_replay = use_prioritized_replay

        # Device setup
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logger.info(f"DQN Agent initialized on device: {self.device}")

        # Network initialization
        NetworkClass = DuelingDQNNetwork if use_dueling else DQNNetwork

        self.policy_net = NetworkClass(state_dim, action_dim).to(self.device)
        self.target_net = NetworkClass(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network always in eval mode

        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)

        # Loss function
        self.criterion = nn.SmoothL1Loss(reduction='none')  # Huber loss

        # Replay buffer
        if use_prioritized_replay:
            self.memory = PrioritizedReplayBuffer(capacity=buffer_size)
        else:
            self.memory = ReplayBuffer(capacity=buffer_size)

        # Training statistics
        self.episode_count = 0
        self.total_steps = 0
        self.training_losses = []
        self.episode_rewards = []

    def select_action(self, state: np.ndarray, epsilon: Optional[float] = None) -> int:
        """
        Select action using epsilon-greedy policy.

        Args:
            state: Current state vector
            epsilon: Exploration rate (uses self.epsilon if None)

        Returns:
            Selected action index
        """
        if epsilon is None:
            epsilon = self.epsilon

        state_tensor = torch.FloatTensor(state).to(self.device)
        return self.policy_net.get_action(state_tensor, epsilon)

    def store_experience(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """
        Store experience in replay buffer.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Terminal flag
        """
        self.memory.add(state, action, reward, next_state, done)

    def train_step(self) -> Optional[float]:
        """
        Perform one training step (sample batch and update network).

        Returns:
            Training loss if buffer has enough samples, None otherwise
        """
        if len(self.memory) < self.batch_size:
            return None

        # Sample batch
        if self.use_prioritized_replay:
            (states, actions, rewards, next_states, dones,
             weights, indices) = self.memory.sample(self.batch_size)

            states = states.to(self.device)
            actions = actions.to(self.device)
            rewards = rewards.to(self.device)
            next_states = next_states.to(self.device)
            dones = dones.to(self.device)
            weights = weights.to(self.device)
        else:
            states, actions, rewards, next_states, dones = self.memory.sample(
                self.batch_size
            )
            states = states.to(self.device)
            actions = actions.to(self.device)
            rewards = rewards.to(self.device)
            next_states = next_states.to(self.device)
            dones = dones.to(self.device)
            weights = torch.ones_like(rewards).to(self.device)

        # Current Q-values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))

        # Double DQN: use policy network to select actions, target network to evaluate
        with torch.no_grad():
            # Select best actions using policy network
            next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)

            # Evaluate using target network
            next_q_values = self.target_net(next_states).gather(1, next_actions)

            # Compute target Q-values
            target_q_values = rewards.unsqueeze(1) + \
                              (1 - dones.unsqueeze(1)) * self.gamma * next_q_values

        # Compute loss (element-wise for prioritized replay)
        loss_elements = self.criterion(current_q_values, target_q_values)
        loss = (loss_elements * weights.unsqueeze(1)).mean()

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)

        self.optimizer.step()

        # Update priorities in replay buffer
        if self.use_prioritized_replay:
            td_errors = (current_q_values - target_q_values).detach().cpu().numpy()
            self.memory.update_priorities(indices, td_errors.flatten())

        loss_value = loss.item()
        self.training_losses.append(loss_value)

        return loss_value

    def update_target_network(self, soft_update: bool = True) -> None:
        """
        Update target network.

        Args:
            soft_update: Use soft update (Polyak averaging) if True, hard copy if False
        """
        if soft_update:
            # Soft update: θ_target = τ * θ_policy + (1 - τ) * θ_target
            for target_param, policy_param in zip(
                self.target_net.parameters(),
                self.policy_net.parameters()
            ):
                target_param.data.copy_(
                    self.tau * policy_param.data + (1 - self.tau) * target_param.data
                )
        else:
            # Hard update: θ_target = θ_policy
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self) -> None:
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def end_episode(self, episode_reward: float) -> None:
        """
        Mark end of episode and perform bookkeeping.

        Args:
            episode_reward: Total reward accumulated in episode
        """
        self.episode_count += 1
        self.episode_rewards.append(episode_reward)

        # Update target network periodically
        if self.episode_count % self.target_update_freq == 0:
            self.update_target_network(soft_update=True)

        # Decay epsilon
        self.decay_epsilon()

        logger.debug(
            f"Episode {self.episode_count}: "
            f"Reward={episode_reward:.2f}, Epsilon={self.epsilon:.3f}"
        )

    def get_statistics(self) -> Dict[str, float]:
        """
        Get training statistics.

        Returns:
            Dictionary of statistics
        """
        stats = {
            "episode_count": self.episode_count,
            "total_steps": self.total_steps,
            "epsilon": self.epsilon,
            "buffer_size": len(self.memory),
        }

        if self.episode_rewards:
            stats["avg_reward_last_10"] = np.mean(self.episode_rewards[-10:])
            stats["avg_reward_last_100"] = np.mean(self.episode_rewards[-100:])
            stats["max_reward"] = np.max(self.episode_rewards)

        if self.training_losses:
            stats["avg_loss_last_100"] = np.mean(self.training_losses[-100:])
            stats["current_loss"] = self.training_losses[-1]

        return stats

    def save(self, path: str) -> None:
        """
        Save agent state to disk.

        Args:
            path: File path for checkpoint
        """
        checkpoint = {
            "policy_net_state_dict": self.policy_net.state_dict(),
            "target_net_state_dict": self.target_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "episode_count": self.episode_count,
            "total_steps": self.total_steps,
            "epsilon": self.epsilon,
            "episode_rewards": self.episode_rewards,
            "training_losses": self.training_losses,
        }

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)
        logger.info(f"Agent checkpoint saved to {path}")

    def load(self, path: str) -> None:
        """
        Load agent state from disk.

        Args:
            path: File path for checkpoint
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.policy_net.load_state_dict(checkpoint["policy_net_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_net_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.episode_count = checkpoint["episode_count"]
        self.total_steps = checkpoint["total_steps"]
        self.epsilon = checkpoint["epsilon"]
        self.episode_rewards = checkpoint["episode_rewards"]
        self.training_losses = checkpoint["training_losses"]

        logger.info(f"Agent checkpoint loaded from {path}")
