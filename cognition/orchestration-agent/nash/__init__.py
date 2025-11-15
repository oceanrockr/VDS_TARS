"""
Nash Equilibrium Solver for Multi-Agent Coordination

This module implements Nash equilibrium computation for strategic coordination:
- Lemke-Howson algorithm for 2-player games
- Iterative best-response dynamics for n-player games
- Pareto optimality filtering
- Support-enumeration method for small games

Author: T.A.R.S. Cognitive Team
Version: v0.9.2-alpha
"""

from .solver import NashSolver, NashResult
from .pareto import ParetoFrontier

__all__ = ["NashSolver", "NashResult", "ParetoFrontier"]
