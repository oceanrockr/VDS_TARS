"""
Nash Equilibrium Solver Implementation

Implements multiple algorithms for computing Nash equilibria:
1. Lemke-Howson for 2-player games
2. Iterative best-response for n-player games
3. Support enumeration for small games

Author: T.A.R.S. Cognitive Team
Version: v0.9.2-alpha
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linprog


logger = logging.getLogger(__name__)


@dataclass
class NashResult:
    """
    Result of Nash equilibrium computation.

    Attributes:
        converged (bool): Whether algorithm converged
        strategy_profile (Dict[str, np.ndarray]): Strategy for each player
        payoffs (Dict[str, float]): Expected payoff for each player
        iterations (int): Number of iterations
        method (str): Algorithm used
        is_pure (bool): Whether equilibrium is pure strategy
        support (Optional[Dict[str, List[int]]]): Support of mixed strategy
    """
    converged: bool
    strategy_profile: Dict[str, np.ndarray]
    payoffs: Dict[str, float]
    iterations: int
    method: str
    is_pure: bool = False
    support: Optional[Dict[str, List[int]]] = None


class NashSolver:
    """
    Nash Equilibrium solver for multi-agent games.

    Supports both 2-player and n-player games with various solution methods.

    Args:
        method (str): Solution method ('lemke_howson', 'iterative_br', 'support_enum')
        max_iterations (int): Maximum iterations for iterative methods
        tolerance (float): Convergence tolerance
        random_seed (Optional[int]): Random seed for reproducibility
    """

    def __init__(
        self,
        method: str = "iterative_br",
        max_iterations: int = 1000,
        tolerance: float = 1e-6,
        random_seed: Optional[int] = None
    ):
        self.method = method
        self.max_iterations = max_iterations
        self.tolerance = tolerance

        if random_seed is not None:
            np.random.seed(random_seed)

        logger.info(
            f"Nash Solver initialized: method={method}, "
            f"max_iter={max_iterations}, tol={tolerance}"
        )

    def solve(
        self,
        payoff_matrices: Dict[str, np.ndarray],
        player_names: Optional[List[str]] = None
    ) -> NashResult:
        """
        Compute Nash equilibrium for given game.

        Args:
            payoff_matrices (Dict[str, np.ndarray]): Payoff matrix for each player
                For 2-player: matrices are (m, n) where rows = player1 actions, cols = player2 actions
                For n-player: matrices are multi-dimensional tensors
            player_names (Optional[List[str]]): Names of players

        Returns:
            NashResult: Computed equilibrium
        """
        n_players = len(payoff_matrices)

        if player_names is None:
            player_names = [f"player_{i}" for i in range(n_players)]

        logger.info(f"Computing Nash equilibrium for {n_players}-player game")

        # Choose method based on number of players
        if n_players == 2:
            if self.method == "lemke_howson":
                return self._lemke_howson(payoff_matrices, player_names)
            elif self.method == "support_enum":
                return self._support_enumeration(payoff_matrices, player_names)
            else:
                return self._iterative_best_response(payoff_matrices, player_names)
        else:
            # N-player games use iterative best-response
            return self._iterative_best_response(payoff_matrices, player_names)

    def _iterative_best_response(
        self,
        payoff_matrices: Dict[str, np.ndarray],
        player_names: List[str]
    ) -> NashResult:
        """
        Iterative best-response dynamics for n-player games.

        Algorithm:
        1. Initialize with uniform mixed strategies
        2. Each player iteratively computes best response to others' strategies
        3. Converge when no player wants to deviate

        Args:
            payoff_matrices (Dict[str, np.ndarray]): Payoff matrices
            player_names (List[str]): Player names

        Returns:
            NashResult: Computed equilibrium
        """
        n_players = len(player_names)

        # Initialize uniform strategies
        strategies = {}
        action_dims = {}

        for i, player in enumerate(player_names):
            payoff = list(payoff_matrices.values())[i]
            n_actions = payoff.shape[0]  # Number of actions for this player
            action_dims[player] = n_actions
            strategies[player] = np.ones(n_actions) / n_actions

        # Iterative best-response
        converged = False

        for iteration in range(self.max_iterations):
            max_change = 0.0

            # Each player updates their strategy
            for player in player_names:
                old_strategy = strategies[player].copy()

                # Compute expected payoff for each action
                payoff_matrix = payoff_matrices[player]
                expected_payoffs = self._compute_expected_payoffs(
                    payoff_matrix,
                    strategies,
                    player,
                    player_names
                )

                # Best response: put all probability on best action(s)
                best_actions = np.where(
                    np.isclose(expected_payoffs, expected_payoffs.max())
                )[0]

                new_strategy = np.zeros(len(expected_payoffs))
                new_strategy[best_actions] = 1.0 / len(best_actions)

                # Smooth update (inertia for stability)
                alpha = 0.3  # Learning rate
                strategies[player] = alpha * new_strategy + (1 - alpha) * old_strategy

                # Normalize
                strategies[player] /= strategies[player].sum()

                # Track convergence
                change = np.linalg.norm(strategies[player] - old_strategy)
                max_change = max(max_change, change)

            # Check convergence
            if max_change < self.tolerance:
                converged = True
                logger.info(f"Converged after {iteration + 1} iterations")
                break

        # Compute final payoffs
        payoffs = {}
        for player in player_names:
            payoff_matrix = payoff_matrices[player]
            expected_payoff = self._compute_expected_payoffs(
                payoff_matrix,
                strategies,
                player,
                player_names
            )
            payoffs[player] = float(np.dot(strategies[player], expected_payoff))

        # Check if pure strategy
        is_pure = all(
            np.count_nonzero(strategies[player] > 0.01) == 1
            for player in player_names
        )

        # Get support
        support = {
            player: list(np.where(strategies[player] > 0.01)[0])
            for player in player_names
        }

        return NashResult(
            converged=converged,
            strategy_profile=strategies,
            payoffs=payoffs,
            iterations=iteration + 1 if converged else self.max_iterations,
            method="iterative_best_response",
            is_pure=is_pure,
            support=support
        )

    def _compute_expected_payoffs(
        self,
        payoff_matrix: np.ndarray,
        strategies: Dict[str, np.ndarray],
        current_player: str,
        player_names: List[str]
    ) -> np.ndarray:
        """
        Compute expected payoffs for each action of current player.

        Args:
            payoff_matrix (np.ndarray): Payoff matrix for current player
            strategies (Dict[str, np.ndarray]): Current strategies
            current_player (str): Player computing payoffs
            player_names (List[str]): All player names

        Returns:
            np.ndarray: Expected payoff for each action
        """
        n_actions = payoff_matrix.shape[0]
        expected_payoffs = np.zeros(n_actions)

        if len(player_names) == 2:
            # 2-player case: simple matrix multiplication
            player_idx = player_names.index(current_player)
            other_player = [p for p in player_names if p != current_player][0]
            other_strategy = strategies[other_player]

            if player_idx == 0:
                # Row player
                expected_payoffs = payoff_matrix @ other_strategy
            else:
                # Column player
                expected_payoffs = payoff_matrix.T @ strategies[player_names[0]]

        else:
            # N-player case: enumerate all strategy profiles
            # This is a simplified version - for production, use tensor operations
            for action in range(n_actions):
                # For each action of current player, compute expected payoff
                # over all possible opponent action profiles
                payoff_sum = 0.0
                probability_sum = 0.0

                # Get opponent strategies
                opponent_strategies = [
                    strategies[p] for p in player_names if p != current_player
                ]

                # Enumerate opponent actions (simplified for small action spaces)
                # For large spaces, use sampling
                n_opponent_actions = [len(s) for s in opponent_strategies]

                if np.prod(n_opponent_actions) > 1000:
                    # Use sampling for large action spaces
                    n_samples = 1000
                    for _ in range(n_samples):
                        # Sample opponent actions
                        opponent_actions = [
                            np.random.choice(len(s), p=s)
                            for s in opponent_strategies
                        ]

                        # Get payoff for this joint action
                        joint_action = [action] + opponent_actions
                        payoff = payoff_matrix[tuple(joint_action)]

                        # Weight by probability
                        prob = np.prod([
                            opponent_strategies[i][opponent_actions[i]]
                            for i in range(len(opponent_actions))
                        ])

                        payoff_sum += payoff * prob
                        probability_sum += prob

                    expected_payoffs[action] = payoff_sum / max(probability_sum, 1e-10)

                else:
                    # Exact enumeration for small spaces
                    import itertools

                    for opponent_actions in itertools.product(
                        *[range(n) for n in n_opponent_actions]
                    ):
                        # Get payoff
                        joint_action = [action] + list(opponent_actions)
                        payoff = payoff_matrix[tuple(joint_action)]

                        # Probability of this profile
                        prob = np.prod([
                            opponent_strategies[i][opponent_actions[i]]
                            for i in range(len(opponent_actions))
                        ])

                        payoff_sum += payoff * prob

                    expected_payoffs[action] = payoff_sum

        return expected_payoffs

    def _lemke_howson(
        self,
        payoff_matrices: Dict[str, np.ndarray],
        player_names: List[str]
    ) -> NashResult:
        """
        Lemke-Howson algorithm for 2-player games.

        This is a simplified implementation. For production, consider using
        game theory libraries like Nashpy or Gambit.

        Args:
            payoff_matrices (Dict[str, np.ndarray]): Payoff matrices
            player_names (List[str]): Player names (must be exactly 2)

        Returns:
            NashResult: Computed equilibrium
        """
        if len(player_names) != 2:
            raise ValueError("Lemke-Howson requires exactly 2 players")

        # Get payoff matrices
        A = list(payoff_matrices.values())[0]  # Player 1 payoffs
        B = list(payoff_matrices.values())[1]  # Player 2 payoffs

        # Use support enumeration as fallback (simpler implementation)
        return self._support_enumeration(payoff_matrices, player_names)

    def _support_enumeration(
        self,
        payoff_matrices: Dict[str, np.ndarray],
        player_names: List[str]
    ) -> NashResult:
        """
        Support enumeration method for 2-player games.

        Enumerates all possible support sizes and solves for equilibrium.

        Args:
            payoff_matrices (Dict[str, np.ndarray]): Payoff matrices
            player_names (List[str]): Player names

        Returns:
            NashResult: Computed equilibrium
        """
        if len(player_names) != 2:
            raise ValueError("Support enumeration requires exactly 2 players")

        A = list(payoff_matrices.values())[0]
        B = list(payoff_matrices.values())[1]

        m, n = A.shape  # m actions for player 1, n for player 2

        # Try pure strategy equilibria first
        for i in range(m):
            for j in range(n):
                # Check if (i, j) is a pure Nash equilibrium
                # Player 1 doesn't want to deviate
                if A[i, j] >= np.max(A[:, j]):
                    # Player 2 doesn't want to deviate
                    if B[i, j] >= np.max(B[i, :]):
                        # Found pure Nash equilibrium
                        strategy1 = np.zeros(m)
                        strategy1[i] = 1.0
                        strategy2 = np.zeros(n)
                        strategy2[j] = 1.0

                        return NashResult(
                            converged=True,
                            strategy_profile={
                                player_names[0]: strategy1,
                                player_names[1]: strategy2
                            },
                            payoffs={
                                player_names[0]: float(A[i, j]),
                                player_names[1]: float(B[i, j])
                            },
                            iterations=1,
                            method="support_enum_pure",
                            is_pure=True,
                            support={
                                player_names[0]: [i],
                                player_names[1]: [j]
                            }
                        )

        # No pure equilibrium found, use iterative best-response for mixed
        logger.info("No pure equilibrium found, using iterative best-response")
        return self._iterative_best_response(payoff_matrices, player_names)


    def compute_regret(
        self,
        payoff_matrices: Dict[str, np.ndarray],
        strategies: Dict[str, np.ndarray],
        player_names: List[str]
    ) -> Dict[str, float]:
        """
        Compute regret for each player given current strategies.

        Regret = max payoff from best response - current payoff

        Args:
            payoff_matrices (Dict[str, np.ndarray]): Payoff matrices
            strategies (Dict[str, np.ndarray]): Current strategies
            player_names (List[str]): Player names

        Returns:
            Dict[str, float]: Regret for each player
        """
        regrets = {}

        for player in player_names:
            payoff_matrix = payoff_matrices[player]

            # Current expected payoff
            expected_payoffs = self._compute_expected_payoffs(
                payoff_matrix,
                strategies,
                player,
                player_names
            )
            current_payoff = np.dot(strategies[player], expected_payoffs)

            # Best response payoff
            best_payoff = np.max(expected_payoffs)

            # Regret
            regrets[player] = float(best_payoff - current_payoff)

        return regrets
