"""
PPO (Proximal Policy Optimization) Agent for Ethical Oversight

This module implements the Ethical Agent using PPO algorithm with:
- Dual policy and value networks with shared encoder
- Clipped surrogate objective (ε = 0.2)
- KL-divergence penalty for stability
- Adaptive learning rate
- Fairness reward integration

Author: T.A.R.S. Cognitive Team
Version: v0.9.2-alpha
"""

from .network import PPONetwork
from .agent import PPOAgent

__all__ = ["PPONetwork", "PPOAgent"]
