"""
Causal Discovery Module using PC Algorithm

Implements the Peter-Clark (PC) algorithm for learning causal graphs
from observational data.

Reference: "Causation, Prediction, and Search" (Spirtes et al., 2000)

Author: T.A.R.S. Cognitive Team
Version: v0.9.1-alpha
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Set, Optional
from itertools import combinations
import logging
import networkx as nx
from scipy.stats import chi2_contingency, pearsonr


logger = logging.getLogger(__name__)


class PCAlgorithm:
    """
    Peter-Clark algorithm for causal discovery.

    Learns a causal DAG from observational data using conditional
    independence tests.

    Attributes:
        alpha (float): Significance level for independence tests
        method (str): Independence test method ('pearson' or 'chi2')
        max_cond_vars (int): Maximum number of conditioning variables
    """

    def __init__(
        self,
        alpha: float = 0.05,
        method: str = "pearson",
        max_cond_vars: int = 3
    ):
        """
        Initialize PC algorithm.

        Args:
            alpha: Significance level for independence tests
            method: Independence test method
            max_cond_vars: Maximum conditioning set size
        """
        self.alpha = alpha
        self.method = method
        self.max_cond_vars = max_cond_vars

    def learn_graph(self, data: pd.DataFrame) -> nx.DiGraph:
        """
        Learn causal graph from data.

        Args:
            data: Observational data (rows=samples, columns=variables)

        Returns:
            Directed acyclic graph (DAG)
        """
        logger.info(f"Learning causal graph from {len(data)} samples, "
                   f"{len(data.columns)} variables")

        # Step 1: Start with complete undirected graph
        graph = self._initialize_complete_graph(data.columns.tolist())

        # Step 2: Remove edges using conditional independence tests
        graph = self._skeleton_discovery(data, graph)

        # Step 3: Orient edges
        graph = self._orient_edges(graph)

        logger.info(f"Discovered graph with {graph.number_of_nodes()} nodes, "
                   f"{graph.number_of_edges()} edges")

        return graph

    def _initialize_complete_graph(self, variables: List[str]) -> nx.Graph:
        """
        Create complete undirected graph.

        Args:
            variables: List of variable names

        Returns:
            Complete undirected graph
        """
        graph = nx.Graph()
        graph.add_nodes_from(variables)

        # Add all possible edges
        for var1, var2 in combinations(variables, 2):
            graph.add_edge(var1, var2)

        return graph

    def _skeleton_discovery(
        self,
        data: pd.DataFrame,
        graph: nx.Graph
    ) -> nx.Graph:
        """
        Discover graph skeleton using conditional independence.

        Args:
            data: Observational data
            graph: Initial complete graph

        Returns:
            Skeleton graph (undirected)
        """
        # Store conditioning sets for each edge
        separating_sets: Dict[Tuple[str, str], Set[str]] = {}

        # Iterate through increasing conditioning set sizes
        for cond_size in range(self.max_cond_vars + 1):
            edges_to_remove = []

            for var1, var2 in list(graph.edges()):
                # Get neighbors of var1 (excluding var2)
                neighbors = set(graph.neighbors(var1)) - {var2}

                # Test all conditioning sets of current size
                for cond_set in combinations(neighbors, min(cond_size, len(neighbors))):
                    # Test conditional independence
                    is_independent, p_value = self._test_independence(
                        data, var1, var2, list(cond_set)
                    )

                    if is_independent:
                        # Mark edge for removal
                        edges_to_remove.append((var1, var2))
                        separating_sets[(var1, var2)] = set(cond_set)
                        logger.debug(
                            f"Removing edge {var1}-{var2} "
                            f"(conditioning on {cond_set}, p={p_value:.4f})"
                        )
                        break

            # Remove independent edges
            graph.remove_edges_from(edges_to_remove)

        return graph

    def _test_independence(
        self,
        data: pd.DataFrame,
        var1: str,
        var2: str,
        cond_vars: List[str]
    ) -> Tuple[bool, float]:
        """
        Test conditional independence: var1 _||_ var2 | cond_vars

        Args:
            data: Observational data
            var1: First variable
            var2: Second variable
            cond_vars: Conditioning variables

        Returns:
            Tuple of (is_independent, p_value)
        """
        if self.method == "pearson":
            return self._pearson_test(data, var1, var2, cond_vars)
        elif self.method == "chi2":
            return self._chi2_test(data, var1, var2, cond_vars)
        else:
            raise ValueError(f"Unknown independence test: {self.method}")

    def _pearson_test(
        self,
        data: pd.DataFrame,
        var1: str,
        var2: str,
        cond_vars: List[str]
    ) -> Tuple[bool, float]:
        """
        Partial correlation test for conditional independence.

        Uses Fisher's Z-transform for continuous variables.

        Args:
            data: Observational data
            var1: First variable
            var2: Second variable
            cond_vars: Conditioning variables

        Returns:
            Tuple of (is_independent, p_value)
        """
        if not cond_vars:
            # Unconditional test
            corr, p_value = pearsonr(data[var1], data[var2])
            return p_value > self.alpha, p_value

        # Partial correlation via regression residuals
        # Regress var1 on cond_vars
        X_cond = data[cond_vars].values
        y1 = data[var1].values
        y2 = data[var2].values

        # Simple linear regression (least squares)
        beta1 = np.linalg.lstsq(X_cond, y1, rcond=None)[0]
        beta2 = np.linalg.lstsq(X_cond, y2, rcond=None)[0]

        # Compute residuals
        residual1 = y1 - X_cond @ beta1
        residual2 = y2 - X_cond @ beta2

        # Test independence of residuals
        corr, p_value = pearsonr(residual1, residual2)

        return p_value > self.alpha, p_value

    def _chi2_test(
        self,
        data: pd.DataFrame,
        var1: str,
        var2: str,
        cond_vars: List[str]
    ) -> Tuple[bool, float]:
        """
        Chi-squared test for categorical variables.

        Args:
            data: Observational data
            var1: First variable
            var2: Second variable
            cond_vars: Conditioning variables

        Returns:
            Tuple of (is_independent, p_value)
        """
        if not cond_vars:
            # Unconditional chi-squared test
            contingency = pd.crosstab(data[var1], data[var2])
            _, p_value, _, _ = chi2_contingency(contingency)
            return p_value > self.alpha, p_value

        # Conditional chi-squared test
        # Stratify by conditioning variables
        p_values = []

        for cond_values, group_data in data.groupby(cond_vars):
            if len(group_data) < 5:  # Skip small strata
                continue

            contingency = pd.crosstab(group_data[var1], group_data[var2])
            _, p_val, _, _ = chi2_contingency(contingency)
            p_values.append(p_val)

        if not p_values:
            return False, 1.0  # Not enough data

        # Combine p-values using Fisher's method
        combined_p = self._combine_p_values(p_values)

        return combined_p > self.alpha, combined_p

    def _combine_p_values(self, p_values: List[float]) -> float:
        """
        Combine p-values using Fisher's method.

        Args:
            p_values: List of p-values

        Returns:
            Combined p-value
        """
        # Fisher's method: -2 * sum(log(p_i)) ~ chi2(2k)
        test_stat = -2 * np.sum(np.log(np.array(p_values) + 1e-10))
        df = 2 * len(p_values)

        from scipy.stats import chi2
        combined_p = 1 - chi2.cdf(test_stat, df)

        return combined_p

    def _orient_edges(self, graph: nx.Graph) -> nx.DiGraph:
        """
        Orient edges in skeleton graph to produce DAG.

        Uses Meek's rules for edge orientation.

        Args:
            graph: Undirected skeleton graph

        Returns:
            Directed acyclic graph
        """
        # Convert to directed graph (initially bidirected)
        dag = nx.DiGraph()
        dag.add_nodes_from(graph.nodes())

        # Add bidirected edges
        for edge in graph.edges():
            dag.add_edge(edge[0], edge[1])
            dag.add_edge(edge[1], edge[0])

        # Apply Meek's rules to orient edges
        # Rule 1: Orient v-structures (colliders)
        dag = self._orient_v_structures(dag, graph)

        # Rules 2-4: Propagate orientations
        changed = True
        while changed:
            changed = False
            changed |= self._meek_rule_2(dag)
            changed |= self._meek_rule_3(dag)
            changed |= self._meek_rule_4(dag)

        return dag

    def _orient_v_structures(
        self,
        dag: nx.DiGraph,
        skeleton: nx.Graph
    ) -> nx.DiGraph:
        """
        Orient v-structures (colliders): X -> Z <- Y where X and Y not adjacent.

        Args:
            dag: Directed graph with bidirected edges
            skeleton: Undirected skeleton

        Returns:
            DAG with v-structures oriented
        """
        for node in dag.nodes():
            # Get pairs of neighbors
            neighbors = list(dag.predecessors(node))

            for x, y in combinations(neighbors, 2):
                # Check if x and y are not adjacent (v-structure)
                if not skeleton.has_edge(x, y):
                    # Orient as x -> node <- y
                    if dag.has_edge(node, x):
                        dag.remove_edge(node, x)
                    if dag.has_edge(node, y):
                        dag.remove_edge(node, y)

                    logger.debug(f"Oriented v-structure: {x} -> {node} <- {y}")

        return dag

    def _meek_rule_2(self, dag: nx.DiGraph) -> bool:
        """
        Meek's rule 2: If X -> Y - Z and X not adjacent to Z, orient Y -> Z.

        Args:
            dag: Partially oriented DAG

        Returns:
            True if any edge was oriented
        """
        changed = False

        for y in dag.nodes():
            predecessors = list(dag.predecessors(y))
            successors = list(dag.successors(y))

            for x in predecessors:
                if not dag.has_edge(y, x):  # x -> y directed
                    for z in successors:
                        if dag.has_edge(z, y):  # y - z undirected
                            if not dag.has_edge(x, z) and not dag.has_edge(z, x):
                                # Orient y -> z
                                dag.remove_edge(z, y)
                                changed = True

        return changed

    def _meek_rule_3(self, dag: nx.DiGraph) -> bool:
        """
        Meek's rule 3: If X - Y - Z with X -> Z and Y - Z, orient Y -> Z.

        Args:
            dag: Partially oriented DAG

        Returns:
            True if any edge was oriented
        """
        changed = False

        for y in dag.nodes():
            neighbors = list(dag.successors(y))

            for z in neighbors:
                if dag.has_edge(z, y):  # y - z undirected
                    # Find x such that x -> z and x - y
                    predecessors_z = list(dag.predecessors(z))

                    for x in predecessors_z:
                        if not dag.has_edge(z, x):  # x -> z directed
                            if dag.has_edge(x, y) and dag.has_edge(y, x):  # x - y undirected
                                # Orient y -> z
                                dag.remove_edge(z, y)
                                changed = True
                                break

        return changed

    def _meek_rule_4(self, dag: nx.DiGraph) -> bool:
        """
        Meek's rule 4: If X - Y - Z with X -> W -> Z, orient Y -> Z.

        Args:
            dag: Partially oriented DAG

        Returns:
            True if any edge was oriented
        """
        changed = False

        for y in dag.nodes():
            neighbors = list(dag.successors(y))

            for z in neighbors:
                if dag.has_edge(z, y):  # y - z undirected
                    # Find x, w such that x - y, x -> w -> z
                    predecessors_z = list(dag.predecessors(z))

                    for w in predecessors_z:
                        if not dag.has_edge(z, w):  # w -> z directed
                            predecessors_w = list(dag.predecessors(w))

                            for x in predecessors_w:
                                if not dag.has_edge(w, x):  # x -> w directed
                                    if dag.has_edge(x, y) and dag.has_edge(y, x):  # x - y undirected
                                        # Orient y -> z
                                        dag.remove_edge(z, y)
                                        changed = True
                                        break

        return changed
