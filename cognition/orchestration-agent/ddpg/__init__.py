"""
DDPG (Deep Deterministic Policy Gradient) Agent for Resource Management

This module implements the Resource Agent using DDPG algorithm with:
- Actor-Critic architecture for continuous control
- Target networks with soft updates (τ = 0.005)
- Ornstein-Uhlenbeck exploration noise
- Experience replay buffer
- Continuous action space (0.0 - 1.0 scaling factors)

Author: T.A.R.S. Cognitive Team
Version: v0.9.2-alpha
"""

from .network import DDPGActor, DDPGCritic
from .agent import DDPGAgent
from .noise import OUNoise

__all__ = ["DDPGActor", "DDPGCritic", "DDPGAgent", "OUNoise"]
