"""
A2C (Advantage Actor-Critic) Agent for Consensus

This module implements the Consensus Agent using A2C algorithm with:
- Shared backbone network (16 → 32 → 32)
- Separate actor and critic heads
- Generalized Advantage Estimation (GAE)
- Combined loss with entropy bonus

Author: T.A.R.S. Cognitive Team
Version: v0.9.2-alpha
"""

from .network import A2CNetwork
from .agent import A2CAgent

__all__ = ["A2CNetwork", "A2CAgent"]
