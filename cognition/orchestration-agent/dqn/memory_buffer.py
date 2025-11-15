"""
Prioritized Experience Replay Buffer for DQN

Implements prioritized sampling based on TD-error magnitude.
Reference: "Prioritized Experience Replay" (Schaul et al., 2016)

Author: T.A.R.S. Cognitive Team
Version: v0.9.1-alpha
"""

import numpy as np
import torch
from collections import namedtuple
from typing import Tuple, List
import random


# Experience tuple
Experience = namedtuple(
    'Experience',
    ['state', 'action', 'reward', 'next_state', 'done']
)


class SumTree:
    """
    Sum tree data structure for efficient prioritized sampling.

    Stores priorities in a binary tree for O(log n) sampling.
    """

    def __init__(self, capacity: int):
        """
        Initialize sum tree.

        Args:
            capacity: Maximum number of experiences to store
        """
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write_idx = 0
        self.n_entries = 0

    def _propagate(self, idx: int, change: float) -> None:
        """
        Propagate priority change up the tree.

        Args:
            idx: Tree index to update
            change: Priority delta
        """
        parent = (idx - 1) // 2
        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, s: float) -> int:
        """
        Retrieve leaf index for a given cumulative sum.

        Args:
            idx: Current tree index
            s: Target cumulative sum

        Returns:
            Leaf index containing the target sum
        """
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self) -> float:
        """Get total priority sum."""
        return self.tree[0]

    def add(self, priority: float, data: Experience) -> None:
        """
        Add experience with priority.

        Args:
            priority: Experience priority (TD-error magnitude)
            data: Experience tuple
        """
        idx = self.write_idx + self.capacity - 1
        self.data[self.write_idx] = data
        self.update(idx, priority)

        self.write_idx = (self.write_idx + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, idx: int, priority: float) -> None:
        """
        Update priority for a tree index.

        Args:
            idx: Tree index
            priority: New priority value
        """
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, s: float) -> Tuple[int, float, Experience]:
        """
        Sample experience based on cumulative sum.

        Args:
            s: Target cumulative sum (0 to total())

        Returns:
            Tuple of (tree_index, priority, experience)
        """
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


class PrioritizedReplayBuffer:
    """
    Prioritized experience replay buffer.

    Samples experiences with probability proportional to their TD-error.
    """

    def __init__(
        self,
        capacity: int = 10000,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 0.001,
        epsilon: float = 1e-6
    ):
        """
        Initialize prioritized replay buffer.

        Args:
            capacity: Maximum buffer size
            alpha: Priority exponent (0 = uniform, 1 = fully prioritized)
            beta: Importance sampling exponent (0 = no correction, 1 = full correction)
            beta_increment: Beta annealing rate
            epsilon: Small constant to avoid zero priorities
        """
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        self.max_priority = 1.0

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """
        Add experience to buffer with maximum priority.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Terminal flag
        """
        experience = Experience(state, action, reward, next_state, done)
        priority = self.max_priority ** self.alpha
        self.tree.add(priority, experience)

    def sample(self, batch_size: int) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
        torch.Tensor, List[int]
    ]:
        """
        Sample batch with prioritized sampling.

        Args:
            batch_size: Number of experiences to sample

        Returns:
            Tuple of (states, actions, rewards, next_states, dones, weights, indices)
        """
        batch = []
        indices = []
        priorities = []

        # Divide priority range into segments
        segment = self.tree.total() / batch_size

        # Anneal beta towards 1.0
        self.beta = min(1.0, self.beta + self.beta_increment)

        for i in range(batch_size):
            # Sample from each segment
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)

            idx, priority, data = self.tree.get(s)
            batch.append(data)
            indices.append(idx)
            priorities.append(priority)

        # Compute importance sampling weights
        priorities = np.array(priorities)
        sampling_probs = priorities / self.tree.total()
        weights = (self.tree.n_entries * sampling_probs) ** (-self.beta)
        weights /= weights.max()  # Normalize

        # Convert to tensors
        states = torch.FloatTensor(np.array([e.state for e in batch]))
        actions = torch.LongTensor(np.array([e.action for e in batch]))
        rewards = torch.FloatTensor(np.array([e.reward for e in batch]))
        next_states = torch.FloatTensor(np.array([e.next_state for e in batch]))
        dones = torch.FloatTensor(np.array([e.done for e in batch]))
        weights_tensor = torch.FloatTensor(weights)

        return states, actions, rewards, next_states, dones, weights_tensor, indices

    def update_priorities(self, indices: List[int], td_errors: np.ndarray) -> None:
        """
        Update priorities based on TD-errors.

        Args:
            indices: Tree indices to update
            td_errors: TD-error magnitudes
        """
        for idx, error in zip(indices, td_errors):
            priority = (abs(error) + self.epsilon) ** self.alpha
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)

    def __len__(self) -> int:
        """Get current buffer size."""
        return self.tree.n_entries


class ReplayBuffer:
    """
    Standard (non-prioritized) experience replay buffer.

    Used as baseline comparison for prioritized replay.
    """

    def __init__(self, capacity: int = 10000):
        """
        Initialize replay buffer.

        Args:
            capacity: Maximum buffer size
        """
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """Add experience to buffer."""
        experience = Experience(state, action, reward, next_state, done)

        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience

        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """
        Sample random batch.

        Args:
            batch_size: Number of experiences to sample

        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
        """
        batch = random.sample(self.buffer, batch_size)

        states = torch.FloatTensor(np.array([e.state for e in batch]))
        actions = torch.LongTensor(np.array([e.action for e in batch]))
        rewards = torch.FloatTensor(np.array([e.reward for e in batch]))
        next_states = torch.FloatTensor(np.array([e.next_state for e in batch]))
        dones = torch.FloatTensor(np.array([e.done for e in batch]))

        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        """Get current buffer size."""
        return len(self.buffer)
