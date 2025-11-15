"""
Counterfactual Reasoning Module

Implements counterfactual inference for "what-if" questions using
structural causal models.

Reference: "Causality" (Pearl, 2009), Chapter 7

Author: T.A.R.S. Cognitive Team
Version: v0.9.1-alpha
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import networkx as nx
import logging


logger = logging.getLogger(__name__)


class CounterfactualEngine:
    """
    Engine for counterfactual reasoning.

    Answers questions like:
    "What would the outcome have been if we had intervened differently?"

    Uses the three-step process:
    1. Abduction: Infer unobserved variables from observed data
    2. Action: Modify model with intervention
    3. Prediction: Predict outcome under intervention
    """

    def __init__(self, causal_graph: nx.DiGraph):
        """
        Initialize counterfactual engine.

        Args:
            causal_graph: Causal DAG
        """
        self.graph = causal_graph

    def compute_counterfactual(
        self,
        data: pd.DataFrame,
        observed: Dict[str, float],
        intervention: Dict[str, float],
        outcome: str
    ) -> Tuple[float, float]:
        """
        Compute counterfactual outcome.

        Question: "Given observed values, what would outcome be if we had intervened?"

        Example:
            observed = {"max_replicas": 10, "violation_rate": 0.15}
            intervention = {"max_replicas": 8}
            outcome = "violation_rate"
            Result: "violation_rate would have been 0.18"

        Args:
            data: Historical observational data
            observed: Observed variable values in actual world
            intervention: Hypothetical intervention
            outcome: Outcome variable to predict

        Returns:
            Tuple of (counterfactual_value, uncertainty)
        """
        logger.info(f"Computing counterfactual: observed={observed}, "
                   f"intervention={intervention}, outcome={outcome}")

        # Step 1: Abduction - estimate unobserved exogenous variables
        exogenous_values = self._abduction(data, observed)

        # Step 2: Action - modify model with intervention
        modified_graph = self._intervention_graph(intervention)

        # Step 3: Prediction - simulate outcome under intervention
        counterfactual_value, uncertainty = self._predict_counterfactual(
            data,
            modified_graph,
            exogenous_values,
            intervention,
            outcome
        )

        logger.info(f"Counterfactual {outcome}: {counterfactual_value:.4f} ± {uncertainty:.4f}")

        return counterfactual_value, uncertainty

    def _abduction(
        self,
        data: pd.DataFrame,
        observed: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Abduction step: Infer unobserved variables from observations.

        Estimates exogenous noise terms that would produce observed values.

        Args:
            data: Historical data
            observed: Observed variable values

        Returns:
            Dictionary of inferred exogenous values
        """
        exogenous = {}

        # For each observed variable, estimate its exogenous component
        for var, obs_value in observed.items():
            # Get parents of variable
            parents = list(self.graph.predecessors(var))

            if not parents:
                # Root node: exogenous value is the observed value
                exogenous[var] = obs_value
            else:
                # Estimate structural equation residual
                # Simplified: use regression residual as proxy for exogenous term

                # Get parent values from observed dict
                parent_values = [observed.get(p) for p in parents]

                if None in parent_values:
                    # Missing parent values, use historical mean
                    exogenous[var] = data[var].mean() - obs_value
                else:
                    # Estimate expected value given parents
                    expected_value = self._estimate_structural_function(
                        data, var, parents, parent_values
                    )

                    # Exogenous component is residual
                    exogenous[var] = obs_value - expected_value

        return exogenous

    def _intervention_graph(
        self,
        intervention: Dict[str, float]
    ) -> nx.DiGraph:
        """
        Create intervention graph by removing incoming edges to intervened variables.

        Args:
            intervention: Variables to intervene on

        Returns:
            Modified causal graph
        """
        # Copy original graph
        modified_graph = self.graph.copy()

        # Remove incoming edges to intervened variables
        for var in intervention.keys():
            # Remove all edges X -> var
            predecessors = list(modified_graph.predecessors(var))
            for pred in predecessors:
                modified_graph.remove_edge(pred, var)

        return modified_graph

    def _predict_counterfactual(
        self,
        data: pd.DataFrame,
        modified_graph: nx.DiGraph,
        exogenous: Dict[str, float],
        intervention: Dict[str, float],
        outcome: str
    ) -> Tuple[float, float]:
        """
        Predict counterfactual outcome using modified graph.

        Args:
            data: Historical data
            modified_graph: Graph with intervention applied
            exogenous: Inferred exogenous values
            intervention: Intervention values
            outcome: Variable to predict

        Returns:
            Tuple of (predicted_value, uncertainty)
        """
        # Topologically sort variables for causal order
        try:
            causal_order = list(nx.topological_sort(modified_graph))
        except nx.NetworkXError:
            # Graph has cycle, use original order
            causal_order = list(modified_graph.nodes())

        # Compute values for all variables in causal order
        computed_values = intervention.copy()

        for var in causal_order:
            if var in intervention:
                # Intervened variable, use intervention value
                continue

            # Get parents
            parents = list(modified_graph.predecessors(var))

            if not parents:
                # Root node, use exogenous value
                computed_values[var] = exogenous.get(var, data[var].mean())
            else:
                # Compute using structural function
                parent_values = [computed_values.get(p, data[p].mean()) for p in parents]

                expected_value = self._estimate_structural_function(
                    data, var, parents, parent_values
                )

                # Add exogenous component
                computed_values[var] = expected_value + exogenous.get(var, 0.0)

        # Get outcome value
        counterfactual_value = computed_values.get(outcome, data[outcome].mean())

        # Estimate uncertainty using historical variance
        uncertainty = data[outcome].std() / np.sqrt(len(data))

        return counterfactual_value, uncertainty

    def _estimate_structural_function(
        self,
        data: pd.DataFrame,
        variable: str,
        parents: List[str],
        parent_values: List[float]
    ) -> float:
        """
        Estimate structural function: Y = f(parents) + noise

        Uses linear regression as approximation.

        Args:
            data: Historical data
            variable: Variable to predict
            parents: Parent variables
            parent_values: Values of parent variables

        Returns:
            Estimated value
        """
        if not parents:
            return data[variable].mean()

        # Fit linear regression
        X = data[parents].values
        Y = data[variable].values

        if len(X) < 5:
            return data[variable].mean()

        # Add intercept
        X_with_intercept = np.hstack([np.ones((len(X), 1)), X])

        # Least squares
        try:
            beta = np.linalg.lstsq(X_with_intercept, Y, rcond=None)[0]

            # Predict
            parent_array = np.array([1.0] + parent_values)
            predicted = parent_array @ beta

            return predicted

        except np.linalg.LinAlgError:
            # Singular matrix, return mean
            return data[variable].mean()

    def compute_probability_of_sufficiency(
        self,
        data: pd.DataFrame,
        treatment: str,
        outcome: str,
        treatment_values: Tuple[float, float] = (0.0, 1.0)
    ) -> float:
        """
        Compute Probability of Sufficiency (PS).

        PS = P(Y_1 = 1 | Y_0 = 0, X = 1)
        "Probability that treatment was sufficient to cause outcome"

        Args:
            data: Historical data
            treatment: Treatment variable
            outcome: Outcome variable
            treatment_values: (control, treated) values

        Returns:
            Probability of sufficiency
        """
        control, treated = treatment_values

        # Filter to treated units with positive outcome
        treated_data = data[
            (data[treatment] == treated) & (data[outcome] == 1)
        ]

        if len(treated_data) == 0:
            return 0.0

        # Count how many would have had Y=0 under control
        counterfactuals = []

        for _, row in treated_data.iterrows():
            observed = row.to_dict()
            intervention = {treatment: control}

            try:
                cf_value, _ = self.compute_counterfactual(
                    data, observed, intervention, outcome
                )
                counterfactuals.append(cf_value)
            except Exception as e:
                logger.warning(f"Counterfactual computation failed: {e}")
                continue

        if not counterfactuals:
            return 0.0

        # Probability that counterfactual outcome would be 0
        ps = np.mean(np.array(counterfactuals) < 0.5)

        return ps
