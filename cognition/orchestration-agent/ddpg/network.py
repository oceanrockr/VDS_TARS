"""
DDPG Neural Network Architectures

Separate actor and critic networks for continuous control.

Actor Network:
    Input (20-dim state) → FC (256, ReLU) → FC (256, ReLU) → FC (1 action, Tanh) → Scale to [0, 1]

Critic Network:
    State (20-dim) + Action (1-dim) → FC (256, ReLU) → FC (256, ReLU) → FC (1 Q-value)

Author: T.A.R.S. Cognitive Team
Version: v0.9.2-alpha
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class DDPGActor(nn.Module):
    """
    DDPG Actor network for continuous action selection.

    The actor network outputs a deterministic action given a state.
    Output is passed through tanh and scaled to [0, 1] for resource scaling.

    Args:
        state_dim (int): Dimension of state space
        action_dim (int): Dimension of action space (continuous)
        hidden_dim (int): Size of hidden layers
        action_low (float): Minimum action value
        action_high (float): Maximum action value
    """

    def __init__(
        self,
        state_dim: int = 20,
        action_dim: int = 1,
        hidden_dim: int = 256,
        action_low: float = 0.0,
        action_high: float = 1.0
    ):
        super(DDPGActor, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_low = action_low
        self.action_high = action_high
        self.action_scale = (action_high - action_low) / 2.0
        self.action_bias = (action_high + action_low) / 2.0

        # Network layers
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights with uniform distribution."""
        nn.init.uniform_(self.fc1.weight, -1/np.sqrt(self.state_dim), 1/np.sqrt(self.state_dim))
        nn.init.uniform_(self.fc1.bias, -1/np.sqrt(self.state_dim), 1/np.sqrt(self.state_dim))

        nn.init.uniform_(self.fc2.weight, -1/np.sqrt(256), 1/np.sqrt(256))
        nn.init.uniform_(self.fc2.bias, -1/np.sqrt(256), 1/np.sqrt(256))

        nn.init.uniform_(self.fc3.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.fc3.bias, -3e-3, 3e-3)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through actor network.

        Args:
            state (torch.Tensor): State tensor of shape (batch_size, state_dim)

        Returns:
            torch.Tensor: Action tensor of shape (batch_size, action_dim)
                         scaled to [action_low, action_high]
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))

        # Scale to action range
        action = self.action_scale * x + self.action_bias

        return action


class DDPGCritic(nn.Module):
    """
    DDPG Critic network for Q-value estimation.

    The critic estimates Q(s, a) - the expected return for taking
    action a in state s.

    Args:
        state_dim (int): Dimension of state space
        action_dim (int): Dimension of action space
        hidden_dim (int): Size of hidden layers
    """

    def __init__(
        self,
        state_dim: int = 20,
        action_dim: int = 1,
        hidden_dim: int = 256
    ):
        super(DDPGCritic, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        # State pathway
        self.fc1 = nn.Linear(state_dim, hidden_dim)

        # State + action pathway
        self.fc2 = nn.Linear(hidden_dim + action_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights with uniform distribution."""
        nn.init.uniform_(self.fc1.weight, -1/np.sqrt(self.state_dim), 1/np.sqrt(self.state_dim))
        nn.init.uniform_(self.fc1.bias, -1/np.sqrt(self.state_dim), 1/np.sqrt(self.state_dim))

        nn.init.uniform_(self.fc2.weight, -1/np.sqrt(256 + self.action_dim), 1/np.sqrt(256 + self.action_dim))
        nn.init.uniform_(self.fc2.bias, -1/np.sqrt(256 + self.action_dim), 1/np.sqrt(256 + self.action_dim))

        nn.init.uniform_(self.fc3.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.fc3.bias, -3e-3, 3e-3)

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through critic network.

        Args:
            state (torch.Tensor): State tensor of shape (batch_size, state_dim)
            action (torch.Tensor): Action tensor of shape (batch_size, action_dim)

        Returns:
            torch.Tensor: Q-value of shape (batch_size, 1)
        """
        # Process state
        x = F.relu(self.fc1(state))

        # Concatenate with action
        x = torch.cat([x, action], dim=1)

        # Process combined representation
        x = F.relu(self.fc2(x))
        q_value = self.fc3(x)

        return q_value


# Import numpy for weight initialization
import numpy as np
