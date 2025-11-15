"""
A2C Neural Network Architecture

Actor-Critic network with shared backbone for consensus decision-making.

Architecture:
    Input (16-dim state)
      → Shared: FC1 (32, ReLU) → FC2 (32, ReLU)
      → Actor: FC (5 actions, Softmax)
      → Critic: FC (1 value output)

Author: T.A.R.S. Cognitive Team
Version: v0.9.2-alpha
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class A2CNetwork(nn.Module):
    """
    Advantage Actor-Critic network with shared feature extractor.

    The network consists of:
    1. Shared backbone: Extracts features from state
    2. Actor head: Outputs action probabilities
    3. Critic head: Outputs state value estimate

    Args:
        state_dim (int): Dimension of state space
        action_dim (int): Dimension of action space (discrete)
        hidden_dim (int): Size of hidden layers (default: 32)
    """

    def __init__(
        self,
        state_dim: int = 16,
        action_dim: int = 5,
        hidden_dim: int = 32
    ):
        super(A2CNetwork, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        # Shared backbone
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        # Actor head (policy)
        self.actor = nn.Linear(hidden_dim, action_dim)

        # Critic head (value function)
        self.critic = nn.Linear(hidden_dim, 1)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights using orthogonal initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.

        Args:
            state (torch.Tensor): State tensor of shape (batch_size, state_dim)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - action_probs: Action probabilities (batch_size, action_dim)
                - value: State value estimate (batch_size, 1)
        """
        # Shared feature extraction
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        # Actor: action probabilities
        action_logits = self.actor(x)
        action_probs = F.softmax(action_logits, dim=-1)

        # Critic: state value
        value = self.critic(x)

        return action_probs, value

    def get_action_and_value(
        self,
        state: torch.Tensor,
        action: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action, log probability, entropy, and value for given state.

        Args:
            state (torch.Tensor): State tensor
            action (torch.Tensor, optional): Specific action to evaluate

        Returns:
            Tuple containing:
                - action: Selected action (if action=None) or input action
                - log_prob: Log probability of the action
                - entropy: Policy entropy
                - value: State value estimate
        """
        action_probs, value = self.forward(state)

        # Create distribution
        dist = torch.distributions.Categorical(action_probs)

        # Sample action if not provided
        if action is None:
            action = dist.sample()

        # Calculate log probability and entropy
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action, log_prob, entropy, value

    def evaluate_actions(
        self,
        state: torch.Tensor,
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for given states (used during training).

        Args:
            state (torch.Tensor): Batch of states
            action (torch.Tensor): Batch of actions

        Returns:
            Tuple containing:
                - log_prob: Log probabilities of actions
                - entropy: Policy entropy
                - value: State value estimates
        """
        action_probs, value = self.forward(state)

        # Create distribution
        dist = torch.distributions.Categorical(action_probs)

        # Calculate log probability and entropy
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return log_prob, entropy, value


class A2CSharedNetwork(nn.Module):
    """
    Alternative A2C architecture with larger shared network.

    This version uses a deeper shared backbone for more complex
    feature extraction before splitting into actor and critic.

    Args:
        state_dim (int): Dimension of state space
        action_dim (int): Dimension of action space
        hidden_dims (list): List of hidden layer dimensions
    """

    def __init__(
        self,
        state_dim: int = 16,
        action_dim: int = 5,
        hidden_dims: list = [64, 64, 32]
    ):
        super(A2CSharedNetwork, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        # Build shared layers
        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        self.shared = nn.Sequential(*layers)

        # Actor and critic heads
        self.actor = nn.Linear(prev_dim, action_dim)
        self.critic = nn.Linear(prev_dim, 1)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights with orthogonal initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through network."""
        features = self.shared(state)

        action_logits = self.actor(features)
        action_probs = F.softmax(action_logits, dim=-1)

        value = self.critic(features)

        return action_probs, value
