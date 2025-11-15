"""
DQN Neural Network Architecture for Policy Agent

Implements a 3-layer MLP with configurable hidden sizes for deep Q-learning.
Architecture: 32 → 64 → 64 → 10 with ReLU activations.

Author: T.A.R.S. Cognitive Team
Version: v0.9.1-alpha
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class DQNNetwork(nn.Module):
    """
    Deep Q-Network for policy optimization.

    Architecture:
        - Input layer: state_dim (default 32)
        - Hidden layer 1: 64 units, ReLU
        - Hidden layer 2: 64 units, ReLU
        - Output layer: action_dim (default 10)

    Attributes:
        state_dim (int): Dimensionality of state space
        action_dim (int): Number of discrete actions
        hidden_sizes (Tuple[int, int]): Hidden layer sizes
    """

    def __init__(
        self,
        state_dim: int = 32,
        action_dim: int = 10,
        hidden_sizes: Tuple[int, int] = (64, 64)
    ):
        """
        Initialize DQN network.

        Args:
            state_dim: Input state vector dimension
            action_dim: Number of output actions
            hidden_sizes: Tuple of hidden layer sizes
        """
        super(DQNNetwork, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_sizes = hidden_sizes

        # Network layers
        self.fc1 = nn.Linear(state_dim, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], action_dim)

        # Initialize weights with Xavier uniform
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize network weights using Xavier uniform initialization."""
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)

        # Initialize biases to zero
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            state: Input state tensor [batch_size, state_dim]

        Returns:
            Q-values for each action [batch_size, action_dim]
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values

    def get_action(self, state: torch.Tensor, epsilon: float = 0.0) -> int:
        """
        Select action using epsilon-greedy policy.

        Args:
            state: Current state tensor [state_dim]
            epsilon: Exploration probability (0 = greedy, 1 = random)

        Returns:
            Selected action index
        """
        if torch.rand(1).item() < epsilon:
            # Random exploration
            return torch.randint(0, self.action_dim, (1,)).item()
        else:
            # Greedy exploitation
            with torch.no_grad():
                q_values = self.forward(state.unsqueeze(0))
                return q_values.argmax(dim=1).item()


class DuelingDQNNetwork(nn.Module):
    """
    Dueling DQN architecture with separate value and advantage streams.

    Implements the dueling architecture from:
    "Dueling Network Architectures for Deep Reinforcement Learning" (Wang et al., 2016)

    Q(s, a) = V(s) + (A(s, a) - mean(A(s, a')))
    """

    def __init__(
        self,
        state_dim: int = 32,
        action_dim: int = 10,
        hidden_sizes: Tuple[int, int] = (64, 64)
    ):
        """
        Initialize Dueling DQN network.

        Args:
            state_dim: Input state vector dimension
            action_dim: Number of output actions
            hidden_sizes: Tuple of hidden layer sizes
        """
        super(DuelingDQNNetwork, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        # Shared feature extraction layers
        self.fc1 = nn.Linear(state_dim, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])

        # Value stream
        self.value_fc = nn.Linear(hidden_sizes[1], 1)

        # Advantage stream
        self.advantage_fc = nn.Linear(hidden_sizes[1], action_dim)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize network weights."""
        for layer in [self.fc1, self.fc2, self.value_fc, self.advantage_fc]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with dueling architecture.

        Args:
            state: Input state tensor [batch_size, state_dim]

        Returns:
            Q-values combining value and advantage streams
        """
        # Shared features
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        # Value stream: V(s)
        value = self.value_fc(x)

        # Advantage stream: A(s, a)
        advantage = self.advantage_fc(x)

        # Combine streams: Q(s, a) = V(s) + (A(s, a) - mean(A(s, a')))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))

        return q_values

    def get_action(self, state: torch.Tensor, epsilon: float = 0.0) -> int:
        """
        Select action using epsilon-greedy policy.

        Args:
            state: Current state tensor [state_dim]
            epsilon: Exploration probability

        Returns:
            Selected action index
        """
        if torch.rand(1).item() < epsilon:
            return torch.randint(0, self.action_dim, (1,)).item()
        else:
            with torch.no_grad():
                q_values = self.forward(state.unsqueeze(0))
                return q_values.argmax(dim=1).item()
