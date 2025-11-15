"""
PPO Neural Network Architecture

Policy and value networks with shared encoder for ethical decision-making.

Architecture:
    Input (24-dim state)
      → Shared Encoder: FC1 (64, ReLU) → FC2 (64, ReLU)
      → Policy Network: FC (32, ReLU) → FC (8 actions, Softmax)
      → Value Network: FC (32, ReLU) → FC (1 value output)

Author: T.A.R.S. Cognitive Team
Version: v0.9.2-alpha
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class PPONetwork(nn.Module):
    """
    PPO network with shared encoder and separate policy/value heads.

    The network uses a shared encoder to extract features, then splits
    into separate policy and value networks for better specialization.

    Args:
        state_dim (int): Dimension of state space
        action_dim (int): Dimension of action space (discrete)
        hidden_dim (int): Size of hidden layers
        encoder_dim (int): Size of encoder output
    """

    def __init__(
        self,
        state_dim: int = 24,
        action_dim: int = 8,
        hidden_dim: int = 64,
        encoder_dim: int = 64
    ):
        super(PPONetwork, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.encoder_dim = encoder_dim

        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, encoder_dim),
            nn.ReLU()
        )

        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(encoder_dim, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim)
        )

        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(encoder_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights using orthogonal initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(
        self,
        state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.

        Args:
            state (torch.Tensor): State tensor of shape (batch_size, state_dim)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - action_probs: Action probabilities (batch_size, action_dim)
                - value: State value estimate (batch_size, 1)
        """
        # Shared encoding
        features = self.encoder(state)

        # Policy: action probabilities
        action_logits = self.policy_net(features)
        action_probs = F.softmax(action_logits, dim=-1)

        # Value: state value
        value = self.value_net(features)

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

    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        """
        Get only the value estimate for given state.

        Args:
            state (torch.Tensor): State tensor

        Returns:
            torch.Tensor: Value estimate
        """
        features = self.encoder(state)
        value = self.value_net(features)
        return value


class PPOSeparateNetwork:
    """
    PPO with completely separate policy and value networks.

    This version doesn't share parameters between policy and value,
    which can be useful when the two objectives require very different
    feature representations.

    Args:
        state_dim (int): Dimension of state space
        action_dim (int): Dimension of action space
        hidden_dim (int): Size of hidden layers
    """

    def __init__(
        self,
        state_dim: int = 24,
        action_dim: int = 8,
        hidden_dim: int = 64
    ):
        super(PPOSeparateNetwork, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        # Separate policy network
        self.policy = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

        # Separate value network
        self.value = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights with orthogonal initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through both networks."""
        action_logits = self.policy(state)
        action_probs = F.softmax(action_logits, dim=-1)
        value = self.value(state)
        return action_probs, value
