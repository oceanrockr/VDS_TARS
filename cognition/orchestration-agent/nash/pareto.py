"""
Pareto Frontier Computation

Implements Pareto optimality filtering for multi-objective optimization
and conflict resolution in multi-agent systems.

Author: T.A.R.S. Cognitive Team
Version: v0.9.2-alpha
"""

import logging
from typing import Dict, List, Tuple

import numpy as np


logger = logging.getLogger(__name__)


class ParetoFrontier:
    """
    Pareto frontier computation for multi-objective optimization.

    A solution is Pareto optimal if no other solution is better in all
    objectives simultaneously.

    Args:
        epsilon (float): Tolerance for dominance comparison
    """

    def __init__(self, epsilon: float = 1e-6):
        self.epsilon = epsilon

    def is_dominated(
        self,
        point: np.ndarray,
        other_point: np.ndarray,
        maximize: bool = True
    ) -> bool:
        """
        Check if point is dominated by other_point.

        Args:
            point (np.ndarray): Point to check
            other_point (np.ndarray): Potential dominating point
            maximize (bool): Whether objectives should be maximized

        Returns:
            bool: True if point is dominated by other_point
        """
        if maximize:
            # other_point dominates if it's >= in all dimensions
            # and strictly > in at least one
            all_geq = np.all(other_point >= point - self.epsilon)
            some_greater = np.any(other_point > point + self.epsilon)
            return all_geq and some_greater
        else:
            # For minimization, flip the comparison
            all_leq = np.all(other_point <= point + self.epsilon)
            some_less = np.any(other_point < point - self.epsilon)
            return all_leq and some_less

    def compute_frontier(
        self,
        points: np.ndarray,
        maximize: bool = True,
        return_indices: bool = False
    ) -> np.ndarray:
        """
        Compute Pareto frontier from set of points.

        Args:
            points (np.ndarray): Array of shape (n_points, n_objectives)
            maximize (bool): Whether objectives should be maximized
            return_indices (bool): Whether to return indices instead of points

        Returns:
            np.ndarray: Pareto-optimal points or their indices
        """
        n_points = points.shape[0]
        is_pareto = np.ones(n_points, dtype=bool)

        for i in range(n_points):
            if not is_pareto[i]:
                continue

            # Check if point i is dominated by any other point
            for j in range(n_points):
                if i == j or not is_pareto[j]:
                    continue

                if self.is_dominated(points[i], points[j], maximize):
                    is_pareto[i] = False
                    break

        if return_indices:
            return np.where(is_pareto)[0]
        else:
            return points[is_pareto]

    def select_solution(
        self,
        pareto_points: np.ndarray,
        weights: np.ndarray = None,
        method: str = "weighted_sum"
    ) -> int:
        """
        Select single solution from Pareto frontier.

        Args:
            pareto_points (np.ndarray): Pareto-optimal points
            weights (np.ndarray): Importance weights for objectives
            method (str): Selection method ('weighted_sum', 'max_min', 'centroid')

        Returns:
            int: Index of selected solution
        """
        if len(pareto_points) == 0:
            raise ValueError("No Pareto points provided")

        if len(pareto_points) == 1:
            return 0

        n_objectives = pareto_points.shape[1]

        if weights is None:
            weights = np.ones(n_objectives) / n_objectives

        if method == "weighted_sum":
            # Weighted sum of objectives
            scores = np.dot(pareto_points, weights)
            return int(np.argmax(scores))

        elif method == "max_min":
            # Maximize minimum objective (fairness)
            min_objectives = np.min(pareto_points, axis=1)
            return int(np.argmax(min_objectives))

        elif method == "centroid":
            # Closest to centroid of frontier
            centroid = np.mean(pareto_points, axis=0)
            distances = np.linalg.norm(pareto_points - centroid, axis=1)
            return int(np.argmin(distances))

        else:
            raise ValueError(f"Unknown selection method: {method}")

    def compute_hypervolume(
        self,
        pareto_points: np.ndarray,
        reference_point: np.ndarray = None
    ) -> float:
        """
        Compute hypervolume indicator of Pareto frontier.

        The hypervolume is the volume of the region dominated by the
        Pareto frontier (relative to a reference point).

        Args:
            pareto_points (np.ndarray): Pareto-optimal points
            reference_point (np.ndarray): Reference point for hypervolume

        Returns:
            float: Hypervolume value
        """
        if len(pareto_points) == 0:
            return 0.0

        n_objectives = pareto_points.shape[1]

        if reference_point is None:
            # Use point slightly worse than worst in frontier
            reference_point = np.min(pareto_points, axis=0) - 0.1

        # For 2D case, use simple algorithm
        if n_objectives == 2:
            # Sort by first objective
            sorted_indices = np.argsort(pareto_points[:, 0])
            sorted_points = pareto_points[sorted_indices]

            hypervolume = 0.0
            prev_x = reference_point[0]

            for point in sorted_points:
                width = point[0] - prev_x
                height = point[1] - reference_point[1]
                hypervolume += width * height
                prev_x = point[0]

            return float(hypervolume)

        else:
            # For higher dimensions, use approximation
            # (exact computation is expensive)
            logger.warning(
                f"Hypervolume for {n_objectives}D is approximated"
            )

            # Monte Carlo approximation
            n_samples = 10000
            samples = np.random.uniform(
                reference_point,
                np.max(pareto_points, axis=0),
                size=(n_samples, n_objectives)
            )

            # Check which samples are dominated by frontier
            dominated = np.zeros(n_samples, dtype=bool)

            for point in pareto_points:
                # Sample is dominated if it's worse in all dimensions
                dominated |= np.all(samples <= point, axis=1)

            # Estimate volume
            unit_volume = np.prod(
                np.max(pareto_points, axis=0) - reference_point
            )
            hypervolume = unit_volume * np.mean(dominated)

            return float(hypervolume)

    def resolve_conflicts(
        self,
        agent_payoffs: Dict[str, float],
        candidate_solutions: List[Dict[str, float]],
        fairness_weight: float = 0.3
    ) -> int:
        """
        Resolve conflicts between agents by selecting best solution.

        Args:
            agent_payoffs (Dict[str, float]): Current payoffs for each agent
            candidate_solutions (List[Dict[str, float]]): Candidate solutions
            fairness_weight (float): Weight for fairness objective

        Returns:
            int: Index of selected solution
        """
        if len(candidate_solutions) == 0:
            raise ValueError("No candidate solutions provided")

        # Convert to array
        agent_names = list(agent_payoffs.keys())
        n_agents = len(agent_names)
        n_candidates = len(candidate_solutions)

        points = np.zeros((n_candidates, n_agents))

        for i, solution in enumerate(candidate_solutions):
            for j, agent in enumerate(agent_names):
                points[i, j] = solution.get(agent, 0.0)

        # Compute Pareto frontier
        pareto_indices = self.compute_frontier(
            points,
            maximize=True,
            return_indices=True
        )

        pareto_points = points[pareto_indices]

        logger.info(
            f"Found {len(pareto_indices)} Pareto-optimal solutions "
            f"out of {n_candidates} candidates"
        )

        # Select solution balancing total payoff and fairness
        weights = np.ones(n_agents) / n_agents
        weights *= (1 - fairness_weight)

        # Add fairness component (favor solutions with less variance)
        scores = np.dot(pareto_points, weights)

        # Fairness penalty (higher variance = lower score)
        variances = np.var(pareto_points, axis=1)
        fairness_scores = 1.0 / (1.0 + variances)
        fairness_scores /= fairness_scores.max()

        total_scores = scores + fairness_weight * fairness_scores

        best_pareto_idx = int(np.argmax(total_scores))
        best_candidate_idx = int(pareto_indices[best_pareto_idx])

        logger.info(
            f"Selected solution {best_candidate_idx} with "
            f"payoffs: {candidate_solutions[best_candidate_idx]}"
        )

        return best_candidate_idx
