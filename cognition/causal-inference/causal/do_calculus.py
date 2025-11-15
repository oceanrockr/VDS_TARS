"""
Do-Calculus Engine for Causal Effect Estimation

Implements Pearl's do-calculus for computing interventional distributions
from observational data and causal graphs.

Reference: "Causality" (Pearl, 2009)

Author: T.A.R.S. Cognitive Team
Version: v0.9.1-alpha
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Set, Optional, Tuple
import networkx as nx
from scipy.stats import norm
import logging


logger = logging.getLogger(__name__)


class DoCalculusEngine:
    """
    Engine for computing causal effects using do-calculus.

    Implements:
    - Backdoor adjustment for confounding
    - Do-operator for interventions
    - Average treatment effect (ATE) estimation
    """

    def __init__(self, causal_graph: nx.DiGraph):
        """
        Initialize do-calculus engine.

        Args:
            causal_graph: Causal DAG learned from data
        """
        self.graph = causal_graph

    def estimate_effect(
        self,
        data: pd.DataFrame,
        treatment: str,
        outcome: str,
        treatment_value: float,
        adjustment_set: Optional[List[str]] = None
    ) -> Tuple[float, float]:
        """
        Estimate causal effect of treatment on outcome.

        Computes E[Y | do(X = treatment_value)]

        Args:
            data: Observational data
            treatment: Treatment variable name
            outcome: Outcome variable name
            treatment_value: Value to set treatment to
            adjustment_set: Variables to adjust for (auto-detected if None)

        Returns:
            Tuple of (effect_estimate, standard_error)
        """
        logger.info(f"Estimating effect of {treatment}={treatment_value} on {outcome}")

        # Find adjustment set if not provided
        if adjustment_set is None:
            adjustment_set = self._find_backdoor_set(treatment, outcome)
            logger.info(f"Using backdoor adjustment set: {adjustment_set}")

        if adjustment_set is None:
            raise ValueError(
                f"No valid adjustment set found for {treatment} -> {outcome}. "
                "Effect may not be identifiable."
            )

        # Backdoor adjustment formula
        # E[Y | do(X=x)] = Σ_z E[Y | X=x, Z=z] P(Z=z)

        effects = []
        weights = []

        # Stratify by adjustment set
        if adjustment_set:
            for adj_values, group_data in data.groupby(adjustment_set):
                if len(group_data) < 5:  # Skip small strata
                    continue

                # Estimate E[Y | X=treatment_value, Z=adj_values]
                # Using linear regression within stratum
                effect = self._estimate_conditional_expectation(
                    group_data, treatment, outcome, treatment_value
                )

                # Weight by P(Z=adj_values)
                weight = len(group_data) / len(data)

                effects.append(effect)
                weights.append(weight)
        else:
            # No adjustment needed (no confounders)
            effect = self._estimate_conditional_expectation(
                data, treatment, outcome, treatment_value
            )
            effects = [effect]
            weights = [1.0]

        # Weighted average
        if not effects:
            raise ValueError("Insufficient data for effect estimation")

        causal_effect = np.average(effects, weights=weights)

        # Estimate standard error
        # Using delta method approximation
        std_error = np.std(effects) / np.sqrt(len(effects))

        logger.info(
            f"Estimated causal effect: {causal_effect:.4f} ± {std_error:.4f}"
        )

        return causal_effect, std_error

    def _find_backdoor_set(
        self,
        treatment: str,
        outcome: str
    ) -> Optional[List[str]]:
        """
        Find a minimal backdoor adjustment set.

        A set Z satisfies the backdoor criterion if:
        1. Z blocks all backdoor paths from X to Y
        2. Z does not contain any descendants of X

        Args:
            treatment: Treatment variable
            outcome: Outcome variable

        Returns:
            List of adjustment variables, or None if not identifiable
        """
        # Find all backdoor paths (paths with arrow into treatment)
        backdoor_paths = self._find_backdoor_paths(treatment, outcome)

        if not backdoor_paths:
            # No confounding, no adjustment needed
            return []

        # Find minimal set that blocks all backdoor paths
        # Start with parents of treatment (common approach)
        parents = list(self.graph.predecessors(treatment))

        # Check if parents block all backdoor paths
        if self._blocks_all_paths(parents, backdoor_paths):
            # Remove descendants of treatment
            valid_parents = [
                p for p in parents
                if not nx.has_path(self.graph, treatment, p)
            ]
            return valid_parents

        # If parents insufficient, try all ancestors of treatment
        ancestors = nx.ancestors(self.graph, treatment)
        ancestors_list = list(ancestors)

        if self._blocks_all_paths(ancestors_list, backdoor_paths):
            return ancestors_list

        # No valid adjustment set found
        return None

    def _find_backdoor_paths(
        self,
        treatment: str,
        outcome: str
    ) -> List[List[str]]:
        """
        Find all backdoor paths from treatment to outcome.

        A backdoor path is a path with an arrow into the treatment.

        Args:
            treatment: Treatment variable
            outcome: Outcome variable

        Returns:
            List of backdoor paths
        """
        backdoor_paths = []

        # Get all simple paths in undirected version
        undirected = self.graph.to_undirected()

        try:
            all_paths = list(nx.all_simple_paths(
                undirected, treatment, outcome, cutoff=10
            ))
        except nx.NetworkXNoPath:
            return []

        for path in all_paths:
            # Check if path starts with arrow into treatment
            if len(path) >= 2:
                # Check if edge is treatment <- path[1]
                if self.graph.has_edge(path[1], treatment):
                    backdoor_paths.append(path)

        return backdoor_paths

    def _blocks_all_paths(
        self,
        adjustment_set: List[str],
        paths: List[List[str]]
    ) -> bool:
        """
        Check if adjustment set blocks all paths.

        Args:
            adjustment_set: Variables to condition on
            paths: Paths to block

        Returns:
            True if all paths are blocked
        """
        for path in paths:
            if not self._path_blocked(path, adjustment_set):
                return False
        return True

    def _path_blocked(
        self,
        path: List[str],
        adjustment_set: List[str]
    ) -> bool:
        """
        Check if a path is blocked by the adjustment set.

        A path is blocked if it contains a non-collider in the adjustment set
        or a collider not in the adjustment set.

        Args:
            path: Path to check
            adjustment_set: Variables to condition on

        Returns:
            True if path is blocked
        """
        for i in range(1, len(path) - 1):
            prev_node = path[i - 1]
            curr_node = path[i]
            next_node = path[i + 1]

            # Check if curr_node is a collider (-> curr_node <-)
            is_collider = (
                self.graph.has_edge(prev_node, curr_node) and
                self.graph.has_edge(next_node, curr_node)
            )

            if is_collider:
                # Collider blocks path unless in adjustment set
                if curr_node not in adjustment_set:
                    return True
            else:
                # Non-collider blocks path if in adjustment set
                if curr_node in adjustment_set:
                    return True

        return False

    def _estimate_conditional_expectation(
        self,
        data: pd.DataFrame,
        treatment: str,
        outcome: str,
        treatment_value: float
    ) -> float:
        """
        Estimate E[Y | X=treatment_value] using linear regression.

        Args:
            data: Data for estimation
            treatment: Treatment variable
            outcome: Outcome variable
            treatment_value: Value to set treatment to

        Returns:
            Estimated conditional expectation
        """
        if len(data) < 5:
            # Not enough data, return mean
            return data[outcome].mean()

        # Fit linear regression: Y ~ X
        X = data[treatment].values.reshape(-1, 1)
        Y = data[outcome].values

        # Least squares estimation
        beta = np.linalg.lstsq(
            np.hstack([np.ones((len(X), 1)), X]),
            Y,
            rcond=None
        )[0]

        # Predict at treatment_value
        prediction = beta[0] + beta[1] * treatment_value

        return prediction

    def estimate_ate(
        self,
        data: pd.DataFrame,
        treatment: str,
        outcome: str,
        treatment_values: Tuple[float, float] = (0.0, 1.0)
    ) -> Tuple[float, float]:
        """
        Estimate Average Treatment Effect (ATE).

        ATE = E[Y | do(X=1)] - E[Y | do(X=0)]

        Args:
            data: Observational data
            treatment: Treatment variable
            outcome: Outcome variable
            treatment_values: Tuple of (control_value, treatment_value)

        Returns:
            Tuple of (ate_estimate, standard_error)
        """
        control_value, treated_value = treatment_values

        # Estimate effects for both values
        effect_treated, se_treated = self.estimate_effect(
            data, treatment, outcome, treated_value
        )

        effect_control, se_control = self.estimate_effect(
            data, treatment, outcome, control_value
        )

        # ATE = E[Y|do(X=1)] - E[Y|do(X=0)]
        ate = effect_treated - effect_control

        # Standard error (assuming independence)
        se_ate = np.sqrt(se_treated**2 + se_control**2)

        logger.info(f"Estimated ATE: {ate:.4f} ± {se_ate:.4f}")

        return ate, se_ate

    def compute_confidence_interval(
        self,
        effect: float,
        std_error: float,
        confidence: float = 0.95
    ) -> Tuple[float, float]:
        """
        Compute confidence interval for causal effect.

        Args:
            effect: Point estimate
            std_error: Standard error
            confidence: Confidence level (default 95%)

        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        z_score = norm.ppf((1 + confidence) / 2)
        margin = z_score * std_error

        lower = effect - margin
        upper = effect + margin

        return lower, upper
