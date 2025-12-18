"""
Advanced Correlation & Temporal Intelligence Engine

This module implements a time-aware, direction-aware, and causality-oriented
analytics layer for cross-repository trend analysis. It extends the correlation
engine with:

1. Time-Lagged Correlation Analysis - Correlations with positive/negative lags
2. Directional Influence Scoring - Identify leader → follower relationships
3. Propagation Path Detection - Detect chains like repo-A → repo-B → repo-C
4. Causality Heuristics - Rule-based temporal precedence and monotonic propagation

Architecture:
- LaggedCorrelation: Correlation result at a specific time lag
- InfluenceScore: Repo's influence score based on leading behavior
- PropagationEdge: Single edge in propagation graph
- PropagationPath: Complete propagation chain
- TemporalAnomaly: Detected temporal pattern anomaly
- TemporalIntelligenceReport: Complete output with all analytics

Exit Codes (130-139):
- 130: Success, no temporal risks
- 131: Temporal correlations detected (non-critical)
- 132: Critical propagation risk detected
- 133: Config error
- 134: Parsing error
- 199: General temporal intelligence error

Input:
    org-health-report.json from Phase 14.8 Task 1
    trend-correlation-report.json from Phase 14.8 Task 3 (optional)

Output:
    temporal-intelligence-report.json

Version: 1.0.0
Phase: 14.8 Task 4
"""

import json
import logging
import math
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set

logger = logging.getLogger(__name__)


# ============================================================================
# Exit Codes (130-139)
# ============================================================================

EXIT_TEMPORAL_SUCCESS = 130
EXIT_TEMPORAL_CORRELATIONS_FOUND = 131
EXIT_CRITICAL_PROPAGATION_RISK = 132
EXIT_TEMPORAL_CONFIG_ERROR = 133
EXIT_TEMPORAL_PARSE_ERROR = 134
EXIT_GENERAL_TEMPORAL_ERROR = 199


# ============================================================================
# Custom Exceptions
# ============================================================================

class TemporalIntelligenceError(Exception):
    """Base exception for temporal intelligence engine errors."""
    exit_code = EXIT_GENERAL_TEMPORAL_ERROR


class TemporalIntelligenceConfigError(TemporalIntelligenceError):
    """Configuration error in temporal intelligence engine."""
    exit_code = EXIT_TEMPORAL_CONFIG_ERROR


class TemporalIntelligenceParseError(TemporalIntelligenceError):
    """Failed to parse input reports."""
    exit_code = EXIT_TEMPORAL_PARSE_ERROR


# ============================================================================
# Enums
# ============================================================================

class InfluenceDirection(Enum):
    """Direction of influence between repositories."""
    LEADER = "leader"          # This repo leads others
    FOLLOWER = "follower"      # This repo follows others
    BIDIRECTIONAL = "bidirectional"  # Mutual influence
    INDEPENDENT = "independent"      # No significant influence


class PropagationType(Enum):
    """Type of propagation pattern detected."""
    LINEAR = "linear"          # A → B → C
    BRANCHING = "branching"    # A → B, A → C
    CONVERGING = "converging"  # A → C, B → C
    CYCLIC = "cyclic"          # A → B → A (rare, indicates mutual causation)


class TemporalAnomalyType(Enum):
    """Type of temporal anomaly detected."""
    RAPID_PROPAGATION = "rapid_propagation"
    DELAYED_CASCADE = "delayed_cascade"
    SYNCHRONIZED_LAG_PATTERN = "synchronized_lag_pattern"
    LEADER_DETERIORATION = "leader_deterioration"
    FOLLOWER_IMPACT = "follower_impact"
    SYSTEMIC_PROPAGATION = "systemic_propagation"
    CYCLIC_DEPENDENCY = "cyclic_dependency"


class TemporalSeverity(Enum):
    """Severity of temporal anomalies."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    def __lt__(self, other: "TemporalSeverity") -> bool:
        order = {self.LOW: 0, self.MEDIUM: 1, self.HIGH: 2, self.CRITICAL: 3}
        return order[self] < order[other]

    def __le__(self, other: "TemporalSeverity") -> bool:
        order = {self.LOW: 0, self.MEDIUM: 1, self.HIGH: 2, self.CRITICAL: 3}
        return order[self] <= order[other]


# ============================================================================
# Data Classes - Lagged Correlation
# ============================================================================

@dataclass
class LaggedCorrelation:
    """
    Correlation result between two repos at a specific time lag.

    A positive lag means repo_a leads repo_b by that many intervals.
    A negative lag means repo_b leads repo_a.
    """
    repo_a_id: str
    repo_b_id: str

    # Lag configuration
    lag: int  # Positive = A leads B, Negative = B leads A
    lag_description: str = ""

    # Correlation metrics
    correlation_coefficient: float = 0.0
    p_value_estimate: float = 1.0  # Estimated significance

    # Direction inference
    leader_repo_id: str = ""
    follower_repo_id: str = ""
    lag_intervals: int = 0  # Absolute lag in intervals

    # Quality metrics
    sample_size: int = 0
    is_significant: bool = False
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "repo_a_id": self.repo_a_id,
            "repo_b_id": self.repo_b_id,
            "lag": self.lag,
            "lag_description": self.lag_description,
            "correlation_coefficient": self.correlation_coefficient,
            "p_value_estimate": self.p_value_estimate,
            "leader_repo_id": self.leader_repo_id,
            "follower_repo_id": self.follower_repo_id,
            "lag_intervals": self.lag_intervals,
            "sample_size": self.sample_size,
            "is_significant": self.is_significant,
            "confidence": self.confidence
        }


@dataclass
class InfluenceScore:
    """
    Influence score for a repository based on its leading behavior.

    Higher scores indicate repos that tend to lead others (early warning indicators).
    """
    repo_id: str
    repo_name: str

    # Influence metrics
    influence_score: float = 0.0  # 0-100 scale
    influence_rank: int = 0
    influence_direction: InfluenceDirection = InfluenceDirection.INDEPENDENT

    # Leadership metrics
    repos_led: int = 0  # Number of repos this repo leads
    repos_following: int = 0  # Number of repos this repo follows
    avg_lead_lag: float = 0.0  # Average lag when leading
    avg_follow_lag: float = 0.0  # Average lag when following

    # Strength metrics
    leadership_strength: float = 0.0  # Average correlation when leading
    follower_strength: float = 0.0  # Average correlation when following

    # Risk implications
    systemic_importance: float = 0.0  # How important for org health
    early_warning_potential: float = 0.0  # Usefulness as early warning

    # Detail lists
    led_repos: List[str] = field(default_factory=list)
    following_repos: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "repo_id": self.repo_id,
            "repo_name": self.repo_name,
            "influence_score": self.influence_score,
            "influence_rank": self.influence_rank,
            "influence_direction": self.influence_direction.value,
            "repos_led": self.repos_led,
            "repos_following": self.repos_following,
            "avg_lead_lag": self.avg_lead_lag,
            "avg_follow_lag": self.avg_follow_lag,
            "leadership_strength": self.leadership_strength,
            "follower_strength": self.follower_strength,
            "systemic_importance": self.systemic_importance,
            "early_warning_potential": self.early_warning_potential,
            "led_repos": self.led_repos,
            "following_repos": self.following_repos
        }


# ============================================================================
# Data Classes - Propagation
# ============================================================================

@dataclass
class PropagationEdge:
    """
    Single edge in the propagation graph.

    Represents a directed influence relationship: source → target.
    """
    source_repo_id: str
    target_repo_id: str

    # Edge properties
    lag_intervals: int = 0
    correlation_strength: float = 0.0
    confidence: float = 0.0

    # Causality heuristics score
    causality_score: float = 0.0  # 0-1, higher = more likely causal

    # Edge type
    is_direct: bool = True  # True if no intermediate repos
    intermediate_repos: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "source_repo_id": self.source_repo_id,
            "target_repo_id": self.target_repo_id,
            "lag_intervals": self.lag_intervals,
            "correlation_strength": self.correlation_strength,
            "confidence": self.confidence,
            "causality_score": self.causality_score,
            "is_direct": self.is_direct,
            "intermediate_repos": self.intermediate_repos
        }


@dataclass
class PropagationPath:
    """
    Complete propagation chain across multiple repositories.

    Example: repo-A → repo-B → repo-C → repo-D
    """
    path_id: str
    path_type: PropagationType = PropagationType.LINEAR

    # Path structure
    repo_sequence: List[str] = field(default_factory=list)
    edges: List[PropagationEdge] = field(default_factory=list)

    # Path metrics
    total_lag: int = 0  # Total intervals from start to end
    path_length: int = 0  # Number of hops
    avg_edge_strength: float = 0.0
    path_confidence: float = 0.0

    # Impact assessment
    source_repo_id: str = ""
    terminal_repos: List[str] = field(default_factory=list)
    affected_repo_count: int = 0
    potential_impact: str = ""

    # Risk indicators
    involves_critical_repos: bool = False
    estimated_propagation_time: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "path_id": self.path_id,
            "path_type": self.path_type.value,
            "repo_sequence": self.repo_sequence,
            "edges": [e.to_dict() for e in self.edges],
            "total_lag": self.total_lag,
            "path_length": self.path_length,
            "avg_edge_strength": self.avg_edge_strength,
            "path_confidence": self.path_confidence,
            "source_repo_id": self.source_repo_id,
            "terminal_repos": self.terminal_repos,
            "affected_repo_count": self.affected_repo_count,
            "potential_impact": self.potential_impact,
            "involves_critical_repos": self.involves_critical_repos,
            "estimated_propagation_time": self.estimated_propagation_time
        }


# ============================================================================
# Data Classes - Temporal Anomalies
# ============================================================================

@dataclass
class TemporalAnomaly:
    """
    Detected temporal pattern anomaly.

    Represents a notable temporal pattern that may indicate risk propagation.
    """
    anomaly_id: str
    anomaly_type: TemporalAnomalyType
    severity: TemporalSeverity

    # Description
    title: str
    message: str
    timestamp: str

    # Affected entities
    affected_repos: List[str] = field(default_factory=list)
    affected_count: int = 0
    propagation_path_id: str = ""

    # Trigger metrics
    trigger_metric: str = ""
    trigger_value: float = 0.0
    trigger_threshold: float = 0.0

    # Temporal details
    lag_pattern: str = ""
    propagation_speed: str = ""
    time_to_impact: str = ""

    # Evidence
    evidence: List[str] = field(default_factory=list)
    supporting_correlations: List[str] = field(default_factory=list)

    # Recommendations
    recommended_actions: List[str] = field(default_factory=list)
    monitoring_suggestions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "anomaly_id": self.anomaly_id,
            "anomaly_type": self.anomaly_type.value,
            "severity": self.severity.value,
            "title": self.title,
            "message": self.message,
            "timestamp": self.timestamp,
            "affected_repos": self.affected_repos,
            "affected_count": self.affected_count,
            "propagation_path_id": self.propagation_path_id,
            "trigger_metric": self.trigger_metric,
            "trigger_value": self.trigger_value,
            "trigger_threshold": self.trigger_threshold,
            "lag_pattern": self.lag_pattern,
            "propagation_speed": self.propagation_speed,
            "time_to_impact": self.time_to_impact,
            "evidence": self.evidence,
            "supporting_correlations": self.supporting_correlations,
            "recommended_actions": self.recommended_actions,
            "monitoring_suggestions": self.monitoring_suggestions
        }


# ============================================================================
# Data Classes - Configuration
# ============================================================================

@dataclass
class TemporalThresholds:
    """Configurable thresholds for temporal intelligence engine."""

    # Lag configuration
    max_lag: int = 3  # Maximum lag to analyze (±max_lag)
    min_lag: int = -3  # Minimum lag (usually -max_lag)

    # Correlation thresholds
    min_lagged_correlation: float = 0.5
    min_significant_correlation: float = 0.6
    strong_correlation_threshold: float = 0.7

    # Influence scoring
    min_influence_score: float = 30.0
    high_influence_threshold: float = 70.0
    min_repos_led: int = 2

    # Propagation detection
    min_path_confidence: float = 0.5
    max_path_length: int = 5
    min_causality_score: float = 0.4

    # Anomaly detection
    rapid_propagation_threshold: int = 1  # Lag intervals
    systemic_propagation_threshold: int = 3  # Min repos affected
    leader_deterioration_threshold: float = 0.15  # Score drop

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TemporalThresholds":
        """Create from dictionary."""
        return cls(
            max_lag=data.get("max_lag", 3),
            min_lag=data.get("min_lag", -3),
            min_lagged_correlation=data.get("min_lagged_correlation", 0.5),
            min_significant_correlation=data.get("min_significant_correlation", 0.6),
            strong_correlation_threshold=data.get("strong_correlation_threshold", 0.7),
            min_influence_score=data.get("min_influence_score", 30.0),
            high_influence_threshold=data.get("high_influence_threshold", 70.0),
            min_repos_led=data.get("min_repos_led", 2),
            min_path_confidence=data.get("min_path_confidence", 0.5),
            max_path_length=data.get("max_path_length", 5),
            min_causality_score=data.get("min_causality_score", 0.4),
            rapid_propagation_threshold=data.get("rapid_propagation_threshold", 1),
            systemic_propagation_threshold=data.get("systemic_propagation_threshold", 3),
            leader_deterioration_threshold=data.get("leader_deterioration_threshold", 0.15)
        )


@dataclass
class TemporalIntelligenceConfig:
    """Configuration for the Temporal Intelligence Engine."""

    # Input paths
    org_report_path: Path
    correlation_report_path: Optional[Path] = None

    # Output
    output_path: Optional[Path] = None

    # Thresholds
    thresholds: TemporalThresholds = field(default_factory=TemporalThresholds)

    # Analysis options
    compute_lagged_correlations: bool = True
    compute_influence_scores: bool = True
    compute_propagation_paths: bool = True
    detect_temporal_anomalies: bool = True

    # Behavior flags
    verbose: bool = False
    summary_only: bool = False

    # CI/CD flags
    fail_on_critical_propagation: bool = False
    fail_on_any_temporal_patterns: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "org_report_path": str(self.org_report_path),
            "correlation_report_path": str(self.correlation_report_path) if self.correlation_report_path else None,
            "output_path": str(self.output_path) if self.output_path else None,
            "thresholds": self.thresholds.to_dict(),
            "compute_lagged_correlations": self.compute_lagged_correlations,
            "compute_influence_scores": self.compute_influence_scores,
            "compute_propagation_paths": self.compute_propagation_paths,
            "detect_temporal_anomalies": self.detect_temporal_anomalies,
            "verbose": self.verbose,
            "summary_only": self.summary_only,
            "fail_on_critical_propagation": self.fail_on_critical_propagation,
            "fail_on_any_temporal_patterns": self.fail_on_any_temporal_patterns
        }


# ============================================================================
# Data Classes - Report
# ============================================================================

@dataclass
class TemporalIntelligenceSummary:
    """Summary statistics for temporal intelligence analysis."""

    # Lagged correlation summary
    total_repo_pairs: int = 0
    lagged_correlations_computed: int = 0
    significant_lagged_correlations: int = 0
    leader_follower_pairs: int = 0

    # Influence summary
    repos_with_influence: int = 0
    high_influence_repos: int = 0
    leader_repos: int = 0
    follower_repos: int = 0

    # Propagation summary
    propagation_paths_detected: int = 0
    linear_paths: int = 0
    branching_paths: int = 0
    longest_path_length: int = 0
    repos_in_paths: int = 0

    # Anomaly summary
    total_anomalies: int = 0
    critical_anomalies: int = 0
    high_anomalies: int = 0
    systemic_risks: int = 0

    # Org-level indicators
    avg_propagation_lag: float = 0.0
    org_interconnectedness: float = 0.0  # How connected repos are temporally

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class TemporalIntelligenceReport:
    """
    Complete Temporal Intelligence Report.

    Contains all temporal analysis results including:
    - Lagged correlations
    - Influence rankings
    - Propagation paths
    - Temporal anomalies
    - Recommendations
    """
    # Report metadata
    report_id: str
    generated_at: str
    org_report_path: str
    correlation_report_path: str = ""

    # Summary
    summary: TemporalIntelligenceSummary = field(default_factory=TemporalIntelligenceSummary)

    # Lagged correlations
    lagged_correlations: List[LaggedCorrelation] = field(default_factory=list)
    optimal_lag_matrix: Dict[str, Dict[str, int]] = field(default_factory=dict)

    # Influence rankings
    influence_scores: List[InfluenceScore] = field(default_factory=list)
    leader_ranking: List[str] = field(default_factory=list)

    # Propagation paths
    propagation_paths: List[PropagationPath] = field(default_factory=list)
    propagation_graph: Dict[str, List[str]] = field(default_factory=dict)

    # Temporal anomalies
    anomalies: List[TemporalAnomaly] = field(default_factory=list)

    # Recommendations
    recommendations: List[Dict[str, Any]] = field(default_factory=list)
    monitoring_priorities: List[str] = field(default_factory=list)

    # Source report metadata
    org_health_status: str = "unknown"
    org_health_score: float = 0.0
    total_repos: int = 0

    # Performance
    analysis_duration_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "report_id": self.report_id,
            "generated_at": self.generated_at,
            "org_report_path": self.org_report_path,
            "correlation_report_path": self.correlation_report_path,
            "summary": self.summary.to_dict(),
            "lagged_correlations": [lc.to_dict() for lc in self.lagged_correlations],
            "optimal_lag_matrix": self.optimal_lag_matrix,
            "influence_scores": [s.to_dict() for s in self.influence_scores],
            "leader_ranking": self.leader_ranking,
            "propagation_paths": [p.to_dict() for p in self.propagation_paths],
            "propagation_graph": self.propagation_graph,
            "anomalies": [a.to_dict() for a in self.anomalies],
            "recommendations": self.recommendations,
            "monitoring_priorities": self.monitoring_priorities,
            "org_health_status": self.org_health_status,
            "org_health_score": self.org_health_score,
            "total_repos": self.total_repos,
            "analysis_duration_ms": self.analysis_duration_ms
        }


# ============================================================================
# Repo Trend Series (from correlation engine, simplified)
# ============================================================================

@dataclass
class RepoTimeSeries:
    """Simplified time series data for a repository."""
    repo_id: str
    repo_name: str
    scores: List[float] = field(default_factory=list)
    timestamps: List[str] = field(default_factory=list)
    current_score: float = 0.0
    current_trend: str = "unknown"
    risk_tier: str = "low"

    def get_score_at(self, index: int) -> Optional[float]:
        """Get score at specific index, or None if out of bounds."""
        if 0 <= index < len(self.scores):
            return self.scores[index]
        return None


# ============================================================================
# Lagged Correlation Engine
# ============================================================================

class LaggedCorrelationEngine:
    """
    Computes time-lagged correlations between repository trend series.

    For each pair of repos (A, B), computes correlation at various lags:
    - lag=0: synchronous correlation
    - lag=+k: A leads B by k intervals
    - lag=-k: B leads A by k intervals
    """

    def __init__(self, config: TemporalIntelligenceConfig):
        """Initialize lagged correlation engine."""
        self.config = config
        self.thresholds = config.thresholds
        self._lagged_correlations: List[LaggedCorrelation] = []
        self._optimal_lags: Dict[str, Dict[str, int]] = {}

    def compute_all_lagged_correlations(
        self,
        repo_series: Dict[str, RepoTimeSeries]
    ) -> List[LaggedCorrelation]:
        """
        Compute lagged correlations for all repo pairs.

        Args:
            repo_series: Dictionary of repo_id -> RepoTimeSeries

        Returns:
            List of significant LaggedCorrelation results
        """
        self._lagged_correlations = []
        self._optimal_lags = {rid: {} for rid in repo_series}

        repo_ids = list(repo_series.keys())
        n = len(repo_ids)

        for i in range(n):
            repo_a_id = repo_ids[i]
            series_a = repo_series[repo_a_id]

            for j in range(i + 1, n):
                repo_b_id = repo_ids[j]
                series_b = repo_series[repo_b_id]

                # Compute correlations at all lags
                lag_results = self._compute_lagged_correlation_pair(
                    series_a, series_b
                )

                # Find optimal lag (highest absolute correlation)
                if lag_results:
                    best = max(lag_results, key=lambda x: abs(x.correlation_coefficient))
                    if best.is_significant:
                        self._lagged_correlations.append(best)
                        self._optimal_lags[repo_a_id][repo_b_id] = best.lag
                        self._optimal_lags[repo_b_id][repo_a_id] = -best.lag

        logger.info(f"Computed {len(self._lagged_correlations)} significant "
                   f"lagged correlations")

        return self._lagged_correlations

    def _compute_lagged_correlation_pair(
        self,
        series_a: RepoTimeSeries,
        series_b: RepoTimeSeries
    ) -> List[LaggedCorrelation]:
        """
        Compute correlations at all lags for a pair of repos.

        Args:
            series_a: First repo's time series
            series_b: Second repo's time series

        Returns:
            List of LaggedCorrelation at each lag
        """
        results = []
        scores_a = series_a.scores
        scores_b = series_b.scores

        min_len = min(len(scores_a), len(scores_b))
        if min_len < 3:
            return results

        for lag in range(self.thresholds.min_lag, self.thresholds.max_lag + 1):
            corr = self._compute_correlation_at_lag(scores_a, scores_b, lag)

            if corr is not None:
                # Determine leader/follower
                if lag > 0:
                    leader = series_a.repo_id
                    follower = series_b.repo_id
                elif lag < 0:
                    leader = series_b.repo_id
                    follower = series_a.repo_id
                else:
                    leader = ""
                    follower = ""

                # Create lag description
                if lag > 0:
                    lag_desc = f"{series_a.repo_id} leads {series_b.repo_id} by {lag} interval(s)"
                elif lag < 0:
                    lag_desc = f"{series_b.repo_id} leads {series_a.repo_id} by {-lag} interval(s)"
                else:
                    lag_desc = "Synchronous correlation"

                # Estimate significance
                is_sig = abs(corr) >= self.thresholds.min_significant_correlation
                confidence = min(1.0, abs(corr) / self.thresholds.strong_correlation_threshold)

                lc = LaggedCorrelation(
                    repo_a_id=series_a.repo_id,
                    repo_b_id=series_b.repo_id,
                    lag=lag,
                    lag_description=lag_desc,
                    correlation_coefficient=corr,
                    leader_repo_id=leader,
                    follower_repo_id=follower,
                    lag_intervals=abs(lag),
                    sample_size=min_len - abs(lag),
                    is_significant=is_sig,
                    confidence=confidence
                )
                results.append(lc)

        return results

    def _compute_correlation_at_lag(
        self,
        x: List[float],
        y: List[float],
        lag: int
    ) -> Optional[float]:
        """
        Compute Pearson correlation at specific lag.

        If lag > 0: correlate x[:-lag] with y[lag:]
        If lag < 0: correlate x[-lag:] with y[:lag]
        If lag = 0: correlate x with y directly
        """
        if lag > 0:
            x_aligned = x[:-lag] if lag < len(x) else []
            y_aligned = y[lag:] if lag < len(y) else []
        elif lag < 0:
            abs_lag = abs(lag)
            x_aligned = x[abs_lag:] if abs_lag < len(x) else []
            y_aligned = y[:-abs_lag] if abs_lag < len(y) else []
        else:
            min_len = min(len(x), len(y))
            x_aligned = x[:min_len]
            y_aligned = y[:min_len]

        if len(x_aligned) < 2 or len(y_aligned) < 2:
            return None

        min_len = min(len(x_aligned), len(y_aligned))
        x_aligned = x_aligned[:min_len]
        y_aligned = y_aligned[:min_len]

        return self._pearson_correlation(x_aligned, y_aligned)

    def _pearson_correlation(self, x: List[float], y: List[float]) -> float:
        """Compute Pearson correlation coefficient."""
        n = len(x)
        if n < 2:
            return 0.0

        mean_x = sum(x) / n
        mean_y = sum(y) / n

        cov = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
        std_x = math.sqrt(sum((xi - mean_x) ** 2 for xi in x))
        std_y = math.sqrt(sum((yi - mean_y) ** 2 for yi in y))

        if std_x == 0 or std_y == 0:
            return 0.0

        return cov / (std_x * std_y)

    def get_lagged_correlations(self) -> List[LaggedCorrelation]:
        """Get computed lagged correlations."""
        return self._lagged_correlations

    def get_optimal_lag_matrix(self) -> Dict[str, Dict[str, int]]:
        """Get matrix of optimal lags between repo pairs."""
        return self._optimal_lags


# ============================================================================
# Influence Scoring Engine
# ============================================================================

class InfluenceScoringEngine:
    """
    Computes influence scores for repositories based on temporal leadership.

    A repo has high influence if:
    - It leads many other repos (lagged correlation with positive lag)
    - Its leading correlations are strong
    - Changes in this repo precede changes in others consistently
    """

    def __init__(self, config: TemporalIntelligenceConfig):
        """Initialize influence scoring engine."""
        self.config = config
        self.thresholds = config.thresholds
        self._influence_scores: Dict[str, InfluenceScore] = {}

    def compute_influence_scores(
        self,
        repo_series: Dict[str, RepoTimeSeries],
        lagged_correlations: List[LaggedCorrelation]
    ) -> List[InfluenceScore]:
        """
        Compute influence scores for all repositories.

        Args:
            repo_series: Dictionary of repo_id -> RepoTimeSeries
            lagged_correlations: List of lagged correlations

        Returns:
            List of InfluenceScore sorted by influence (descending)
        """
        self._influence_scores = {}

        # Initialize scores for all repos
        for repo_id, series in repo_series.items():
            self._influence_scores[repo_id] = InfluenceScore(
                repo_id=repo_id,
                repo_name=series.repo_name
            )

        # Process lagged correlations
        for lc in lagged_correlations:
            if not lc.is_significant or lc.lag == 0:
                continue

            leader_id = lc.leader_repo_id
            follower_id = lc.follower_repo_id

            if not leader_id or not follower_id:
                continue

            if leader_id in self._influence_scores:
                score = self._influence_scores[leader_id]
                score.repos_led += 1
                score.led_repos.append(follower_id)

            if follower_id in self._influence_scores:
                score = self._influence_scores[follower_id]
                score.repos_following += 1
                score.following_repos.append(leader_id)

        # Compute influence scores
        for repo_id, score in self._influence_scores.items():
            self._compute_final_influence_score(score, lagged_correlations)

        # Sort by influence score
        sorted_scores = sorted(
            self._influence_scores.values(),
            key=lambda s: s.influence_score,
            reverse=True
        )

        # Assign ranks
        for rank, score in enumerate(sorted_scores, 1):
            score.influence_rank = rank

        logger.info(f"Computed influence scores for {len(sorted_scores)} repos")

        return sorted_scores

    def _compute_final_influence_score(
        self,
        score: InfluenceScore,
        lagged_correlations: List[LaggedCorrelation]
    ) -> None:
        """Compute final influence score for a repo."""
        # Calculate leadership metrics
        leader_correlations = [
            lc for lc in lagged_correlations
            if lc.leader_repo_id == score.repo_id and lc.is_significant
        ]

        follower_correlations = [
            lc for lc in lagged_correlations
            if lc.follower_repo_id == score.repo_id and lc.is_significant
        ]

        if leader_correlations:
            score.leadership_strength = sum(
                abs(lc.correlation_coefficient) for lc in leader_correlations
            ) / len(leader_correlations)
            score.avg_lead_lag = sum(
                lc.lag_intervals for lc in leader_correlations
            ) / len(leader_correlations)

        if follower_correlations:
            score.follower_strength = sum(
                abs(lc.correlation_coefficient) for lc in follower_correlations
            ) / len(follower_correlations)
            score.avg_follow_lag = sum(
                lc.lag_intervals for lc in follower_correlations
            ) / len(follower_correlations)

        # Compute influence score (0-100)
        # Formula: weighted combination of leadership factors
        leadership_factor = min(score.repos_led * 15, 40)  # Max 40 points
        strength_factor = score.leadership_strength * 30  # Max 30 points
        consistency_factor = (1 - (score.repos_following / max(score.repos_led + 1, 1))) * 20  # Max 20 points
        early_warning_factor = min(score.avg_lead_lag * 5, 10) if score.repos_led > 0 else 0  # Max 10 points

        score.influence_score = min(100, (
            leadership_factor +
            strength_factor +
            consistency_factor +
            early_warning_factor
        ))

        # Determine influence direction
        if score.repos_led > score.repos_following and score.repos_led >= self.thresholds.min_repos_led:
            score.influence_direction = InfluenceDirection.LEADER
        elif score.repos_following > score.repos_led and score.repos_following >= self.thresholds.min_repos_led:
            score.influence_direction = InfluenceDirection.FOLLOWER
        elif score.repos_led > 0 and score.repos_following > 0:
            score.influence_direction = InfluenceDirection.BIDIRECTIONAL
        else:
            score.influence_direction = InfluenceDirection.INDEPENDENT

        # Systemic importance (how many repos affected directly/indirectly)
        score.systemic_importance = min(100, score.repos_led * 20 + score.leadership_strength * 30)

        # Early warning potential
        if score.repos_led >= 2 and score.leadership_strength >= 0.6:
            score.early_warning_potential = min(100, (
                score.repos_led * 15 +
                score.leadership_strength * 40 +
                score.avg_lead_lag * 10
            ))

    def get_influence_scores(self) -> Dict[str, InfluenceScore]:
        """Get influence scores dictionary."""
        return self._influence_scores


# ============================================================================
# Propagation Graph Builder
# ============================================================================

class PropagationGraphBuilder:
    """
    Builds and analyzes propagation graphs from lagged correlations.

    Detects chains like: repo-A → repo-B → repo-C
    Where → indicates temporal precedence with significant correlation.
    """

    def __init__(self, config: TemporalIntelligenceConfig):
        """Initialize propagation graph builder."""
        self.config = config
        self.thresholds = config.thresholds
        self._edges: List[PropagationEdge] = []
        self._graph: Dict[str, List[str]] = {}
        self._paths: List[PropagationPath] = []
        self._path_counter = 0

    def build_propagation_graph(
        self,
        lagged_correlations: List[LaggedCorrelation],
        influence_scores: Dict[str, InfluenceScore]
    ) -> Dict[str, List[str]]:
        """
        Build directed propagation graph.

        Args:
            lagged_correlations: Lagged correlations with leader/follower info
            influence_scores: Influence scores for repos

        Returns:
            Adjacency list representation of propagation graph
        """
        self._edges = []
        self._graph = {}

        for lc in lagged_correlations:
            if not lc.is_significant or lc.lag == 0:
                continue

            leader = lc.leader_repo_id
            follower = lc.follower_repo_id

            if not leader or not follower:
                continue

            # Compute causality score using heuristics
            causality_score = self._compute_causality_score(lc, influence_scores)

            if causality_score >= self.thresholds.min_causality_score:
                edge = PropagationEdge(
                    source_repo_id=leader,
                    target_repo_id=follower,
                    lag_intervals=lc.lag_intervals,
                    correlation_strength=abs(lc.correlation_coefficient),
                    confidence=lc.confidence,
                    causality_score=causality_score
                )
                self._edges.append(edge)

                # Add to adjacency list
                if leader not in self._graph:
                    self._graph[leader] = []
                self._graph[leader].append(follower)

        logger.info(f"Built propagation graph with {len(self._edges)} edges")

        return self._graph

    def _compute_causality_score(
        self,
        lc: LaggedCorrelation,
        influence_scores: Dict[str, InfluenceScore]
    ) -> float:
        """
        Compute causality score using rule-based heuristics.

        Heuristics:
        1. Temporal precedence (lag > 0)
        2. Correlation strength
        3. Consistent leadership pattern
        4. No symmetric correlation (excludes coincidental)
        """
        score = 0.0

        # Factor 1: Temporal precedence (0.3 max)
        if lc.lag_intervals > 0:
            score += min(0.3, lc.lag_intervals * 0.1)

        # Factor 2: Correlation strength (0.3 max)
        score += abs(lc.correlation_coefficient) * 0.3

        # Factor 3: Consistent leadership (0.2 max)
        leader_score = influence_scores.get(lc.leader_repo_id)
        if leader_score and leader_score.influence_direction == InfluenceDirection.LEADER:
            score += 0.2
        elif leader_score and leader_score.repos_led > 0:
            score += 0.1

        # Factor 4: Asymmetry bonus (0.2 max)
        # Leader leads more than it follows
        if leader_score and leader_score.repos_led > leader_score.repos_following:
            score += 0.2

        return min(1.0, score)

    def detect_propagation_paths(
        self,
        repo_series: Dict[str, RepoTimeSeries]
    ) -> List[PropagationPath]:
        """
        Detect propagation paths in the graph.

        Finds chains where changes propagate through multiple repos.
        """
        self._paths = []
        visited_starts = set()

        # Find source nodes (repos with outgoing but no/few incoming edges)
        incoming_count = {}
        for target_list in self._graph.values():
            for target in target_list:
                incoming_count[target] = incoming_count.get(target, 0) + 1

        # Identify source nodes
        sources = [
            repo_id for repo_id in self._graph
            if incoming_count.get(repo_id, 0) == 0
        ]

        # If no pure sources, use repos with high outgoing/incoming ratio
        if not sources:
            sources = [
                repo_id for repo_id in self._graph
                if len(self._graph[repo_id]) > incoming_count.get(repo_id, 0)
            ]

        # DFS from each source to find paths
        for source in sources:
            if source in visited_starts:
                continue
            visited_starts.add(source)

            paths_from_source = self._find_paths_from(source, set())
            for path_repos in paths_from_source:
                if len(path_repos) >= 2:
                    path = self._create_propagation_path(path_repos, repo_series)
                    if path.path_confidence >= self.thresholds.min_path_confidence:
                        self._paths.append(path)

        # Classify path types
        self._classify_paths()

        logger.info(f"Detected {len(self._paths)} propagation paths")

        return self._paths

    def _find_paths_from(
        self,
        current: str,
        visited: Set[str],
        max_depth: int = 5
    ) -> List[List[str]]:
        """Find all paths from current node using DFS."""
        if max_depth == 0 or current in visited:
            return [[current]]

        visited = visited | {current}
        paths = []

        neighbors = self._graph.get(current, [])
        if not neighbors:
            return [[current]]

        for neighbor in neighbors:
            if neighbor not in visited:
                sub_paths = self._find_paths_from(
                    neighbor,
                    visited,
                    max_depth - 1
                )
                for sub_path in sub_paths:
                    paths.append([current] + sub_path)

        if not paths:
            paths = [[current]]

        return paths

    def _create_propagation_path(
        self,
        repo_sequence: List[str],
        repo_series: Dict[str, RepoTimeSeries]
    ) -> PropagationPath:
        """Create PropagationPath from repo sequence."""
        self._path_counter += 1

        # Find edges for this path
        edges = []
        total_lag = 0

        for i in range(len(repo_sequence) - 1):
            source = repo_sequence[i]
            target = repo_sequence[i + 1]

            # Find corresponding edge
            for edge in self._edges:
                if edge.source_repo_id == source and edge.target_repo_id == target:
                    edges.append(edge)
                    total_lag += edge.lag_intervals
                    break

        # Calculate path metrics
        avg_strength = sum(e.correlation_strength for e in edges) / len(edges) if edges else 0.0
        path_confidence = sum(e.causality_score for e in edges) / len(edges) if edges else 0.0

        # Check for critical repos
        involves_critical = any(
            repo_series.get(r, RepoTimeSeries(r, r)).risk_tier in ("high", "critical")
            for r in repo_sequence
        )

        path = PropagationPath(
            path_id=f"path_{self._path_counter:03d}",
            repo_sequence=repo_sequence,
            edges=edges,
            total_lag=total_lag,
            path_length=len(repo_sequence) - 1,
            avg_edge_strength=avg_strength,
            path_confidence=path_confidence,
            source_repo_id=repo_sequence[0],
            terminal_repos=[repo_sequence[-1]],
            affected_repo_count=len(repo_sequence),
            involves_critical_repos=involves_critical,
            estimated_propagation_time=f"{total_lag} interval(s)"
        )

        return path

    def _classify_paths(self) -> None:
        """Classify paths by type."""
        source_targets = {}  # source -> list of terminals

        for path in self._paths:
            source = path.source_repo_id
            if source not in source_targets:
                source_targets[source] = set()
            source_targets[source].update(path.terminal_repos)

        for path in self._paths:
            source = path.source_repo_id
            terminals = source_targets.get(source, set())

            if len(terminals) > 1:
                path.path_type = PropagationType.BRANCHING
            else:
                path.path_type = PropagationType.LINEAR

    def get_edges(self) -> List[PropagationEdge]:
        """Get propagation edges."""
        return self._edges

    def get_graph(self) -> Dict[str, List[str]]:
        """Get propagation graph."""
        return self._graph

    def get_paths(self) -> List[PropagationPath]:
        """Get propagation paths."""
        return self._paths


# ============================================================================
# Temporal Anomaly Detector
# ============================================================================

class TemporalAnomalyDetector:
    """
    Detects temporal anomalies using rule-based heuristics.

    Anomaly types:
    - Rapid propagation: Changes spread quickly (low lag)
    - Delayed cascade: Changes propagate slowly but eventually affect many
    - Leader deterioration: High-influence repo declining
    - Systemic propagation: Many repos affected through propagation
    """

    def __init__(self, config: TemporalIntelligenceConfig):
        """Initialize temporal anomaly detector."""
        self.config = config
        self.thresholds = config.thresholds
        self._anomalies: List[TemporalAnomaly] = []
        self._anomaly_counter = 0

    def detect_anomalies(
        self,
        repo_series: Dict[str, RepoTimeSeries],
        lagged_correlations: List[LaggedCorrelation],
        influence_scores: List[InfluenceScore],
        propagation_paths: List[PropagationPath]
    ) -> List[TemporalAnomaly]:
        """
        Detect temporal anomalies.

        Args:
            repo_series: Repository time series data
            lagged_correlations: Lagged correlation results
            influence_scores: Influence scores
            propagation_paths: Detected propagation paths

        Returns:
            List of detected TemporalAnomaly
        """
        self._anomalies = []
        timestamp = datetime.utcnow().isoformat()

        # Detect rapid propagation
        self._detect_rapid_propagation(propagation_paths, timestamp)

        # Detect leader deterioration
        self._detect_leader_deterioration(repo_series, influence_scores, timestamp)

        # Detect systemic propagation
        self._detect_systemic_propagation(propagation_paths, repo_series, timestamp)

        # Detect synchronized lag patterns
        self._detect_synchronized_lag_pattern(lagged_correlations, timestamp)

        # Sort by severity
        self._anomalies.sort(key=lambda a: (
            {"critical": 0, "high": 1, "medium": 2, "low": 3}.get(a.severity.value, 4)
        ))

        logger.info(f"Detected {len(self._anomalies)} temporal anomalies")

        return self._anomalies

    def _generate_anomaly_id(self) -> str:
        """Generate unique anomaly ID."""
        self._anomaly_counter += 1
        return f"temporal_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{self._anomaly_counter:04d}"

    def _detect_rapid_propagation(
        self,
        paths: List[PropagationPath],
        timestamp: str
    ) -> None:
        """Detect rapid propagation patterns."""
        for path in paths:
            if path.path_length >= 2 and path.total_lag <= self.thresholds.rapid_propagation_threshold:
                severity = TemporalSeverity.HIGH
                if path.involves_critical_repos:
                    severity = TemporalSeverity.CRITICAL

                anomaly = TemporalAnomaly(
                    anomaly_id=self._generate_anomaly_id(),
                    anomaly_type=TemporalAnomalyType.RAPID_PROPAGATION,
                    severity=severity,
                    title=f"Rapid Propagation Detected: {path.source_repo_id}",
                    message=f"Changes from {path.source_repo_id} propagate to {path.affected_repo_count} "
                           f"repos in only {path.total_lag} interval(s)",
                    timestamp=timestamp,
                    affected_repos=path.repo_sequence,
                    affected_count=path.affected_repo_count,
                    propagation_path_id=path.path_id,
                    trigger_metric="propagation_lag",
                    trigger_value=float(path.total_lag),
                    trigger_threshold=float(self.thresholds.rapid_propagation_threshold),
                    lag_pattern=f"{path.total_lag} interval(s) for {path.path_length} hops",
                    propagation_speed="rapid",
                    time_to_impact=path.estimated_propagation_time,
                    evidence=[
                        f"Path: {' → '.join(path.repo_sequence)}",
                        f"Total lag: {path.total_lag} interval(s)",
                        f"Path confidence: {path.path_confidence:.2f}"
                    ],
                    recommended_actions=[
                        f"Monitor {path.source_repo_id} as early warning indicator",
                        "Implement circuit breakers for rapid cascade prevention",
                        "Investigate shared dependencies in propagation path"
                    ],
                    monitoring_suggestions=[
                        f"Set alerts on {path.source_repo_id} health score drops",
                        f"Track correlation between {path.source_repo_id} and downstream repos"
                    ]
                )
                self._anomalies.append(anomaly)

    def _detect_leader_deterioration(
        self,
        repo_series: Dict[str, RepoTimeSeries],
        influence_scores: List[InfluenceScore],
        timestamp: str
    ) -> None:
        """Detect deterioration in high-influence repos."""
        for score in influence_scores:
            if score.influence_score < self.thresholds.high_influence_threshold:
                continue

            series = repo_series.get(score.repo_id)
            if not series or len(series.scores) < 2:
                continue

            # Check for declining trend
            if series.current_trend.lower() == "declining":
                # Calculate drop percentage
                if len(series.scores) >= 2:
                    score_drop = (series.scores[0] - series.scores[-1]) / max(series.scores[0], 1)

                    if score_drop >= self.thresholds.leader_deterioration_threshold:
                        severity = TemporalSeverity.HIGH
                        if score.repos_led >= 5 or series.risk_tier in ("high", "critical"):
                            severity = TemporalSeverity.CRITICAL

                        anomaly = TemporalAnomaly(
                            anomaly_id=self._generate_anomaly_id(),
                            anomaly_type=TemporalAnomalyType.LEADER_DETERIORATION,
                            severity=severity,
                            title=f"High-Influence Repo Declining: {score.repo_id}",
                            message=f"{score.repo_id} (influence score: {score.influence_score:.1f}) "
                                   f"is declining and leads {score.repos_led} other repos",
                            timestamp=timestamp,
                            affected_repos=[score.repo_id] + score.led_repos[:5],
                            affected_count=1 + score.repos_led,
                            trigger_metric="leader_score_drop",
                            trigger_value=score_drop * 100,
                            trigger_threshold=self.thresholds.leader_deterioration_threshold * 100,
                            evidence=[
                                f"Influence score: {score.influence_score:.1f}",
                                f"Repos led: {score.repos_led}",
                                f"Score drop: {score_drop*100:.1f}%",
                                f"Led repos: {', '.join(score.led_repos[:3])}"
                            ],
                            recommended_actions=[
                                f"Prioritize stabilization of {score.repo_id}",
                                f"Monitor downstream repos: {', '.join(score.led_repos[:3])}",
                                "Consider preventive measures for follower repos"
                            ],
                            monitoring_suggestions=[
                                f"Set critical alert on {score.repo_id} health",
                                "Track lagged correlation changes"
                            ]
                        )
                        self._anomalies.append(anomaly)

    def _detect_systemic_propagation(
        self,
        paths: List[PropagationPath],
        repo_series: Dict[str, RepoTimeSeries],
        timestamp: str
    ) -> None:
        """Detect systemic propagation risk."""
        # Find paths affecting many repos
        high_impact_paths = [
            p for p in paths
            if p.affected_repo_count >= self.thresholds.systemic_propagation_threshold
        ]

        if not high_impact_paths:
            return

        # Aggregate affected repos
        all_affected = set()
        source_repos = set()
        for path in high_impact_paths:
            all_affected.update(path.repo_sequence)
            source_repos.add(path.source_repo_id)

        # Check for declining sources
        declining_sources = [
            s for s in source_repos
            if repo_series.get(s, RepoTimeSeries(s, s)).current_trend.lower() == "declining"
        ]

        if declining_sources:
            severity = TemporalSeverity.HIGH
            if len(all_affected) >= 5:
                severity = TemporalSeverity.CRITICAL

            anomaly = TemporalAnomaly(
                anomaly_id=self._generate_anomaly_id(),
                anomaly_type=TemporalAnomalyType.SYSTEMIC_PROPAGATION,
                severity=severity,
                title="Systemic Propagation Risk Detected",
                message=f"{len(declining_sources)} declining source repo(s) may impact "
                       f"{len(all_affected)} repos through propagation",
                timestamp=timestamp,
                affected_repos=list(all_affected),
                affected_count=len(all_affected),
                trigger_metric="systemic_propagation_scope",
                trigger_value=float(len(all_affected)),
                trigger_threshold=float(self.thresholds.systemic_propagation_threshold),
                evidence=[
                    f"Declining sources: {', '.join(declining_sources)}",
                    f"Total affected repos: {len(all_affected)}",
                    f"Propagation paths: {len(high_impact_paths)}"
                ],
                recommended_actions=[
                    "Implement org-wide monitoring for systemic risk",
                    f"Stabilize source repos: {', '.join(declining_sources)}",
                    "Consider isolation strategies for critical repos"
                ],
                monitoring_suggestions=[
                    "Create dashboard for propagation path health",
                    "Set alerts on source repo deterioration"
                ]
            )
            self._anomalies.append(anomaly)

    def _detect_synchronized_lag_pattern(
        self,
        lagged_correlations: List[LaggedCorrelation],
        timestamp: str
    ) -> None:
        """Detect synchronized lag patterns across multiple pairs."""
        # Group correlations by lag
        lag_groups: Dict[int, List[LaggedCorrelation]] = {}
        for lc in lagged_correlations:
            if lc.is_significant and lc.lag != 0:
                lag = abs(lc.lag)
                if lag not in lag_groups:
                    lag_groups[lag] = []
                lag_groups[lag].append(lc)

        # Check for concentrated patterns at specific lags
        for lag, correlations in lag_groups.items():
            if len(correlations) >= 3:  # At least 3 pairs at same lag
                affected_repos = set()
                for lc in correlations:
                    affected_repos.add(lc.repo_a_id)
                    affected_repos.add(lc.repo_b_id)

                anomaly = TemporalAnomaly(
                    anomaly_id=self._generate_anomaly_id(),
                    anomaly_type=TemporalAnomalyType.SYNCHRONIZED_LAG_PATTERN,
                    severity=TemporalSeverity.MEDIUM,
                    title=f"Synchronized Lag Pattern at {lag} Interval(s)",
                    message=f"{len(correlations)} repo pairs show synchronized "
                           f"correlation at lag {lag}",
                    timestamp=timestamp,
                    affected_repos=list(affected_repos),
                    affected_count=len(affected_repos),
                    trigger_metric="synchronized_pairs",
                    trigger_value=float(len(correlations)),
                    lag_pattern=f"Lag {lag}: {len(correlations)} pairs",
                    evidence=[
                        f"Pairs at lag {lag}: {len(correlations)}",
                        f"Repos involved: {len(affected_repos)}"
                    ],
                    recommended_actions=[
                        "Investigate shared factors causing synchronized patterns",
                        f"Review activities occurring {lag} interval(s) apart"
                    ]
                )
                self._anomalies.append(anomaly)

    def get_anomalies(self) -> List[TemporalAnomaly]:
        """Get detected anomalies."""
        return self._anomalies


# ============================================================================
# Main Engine
# ============================================================================

class TemporalIntelligenceEngine:
    """
    Main orchestrator for the Advanced Correlation & Temporal Intelligence Engine.

    This class:
    1. Loads data from org health and correlation reports
    2. Computes lagged correlations
    3. Calculates influence scores
    4. Builds propagation graphs and detects paths
    5. Identifies temporal anomalies
    6. Generates temporal intelligence report
    7. Returns appropriate exit code
    """

    def __init__(self, config: TemporalIntelligenceConfig):
        """Initialize the temporal intelligence engine."""
        self.config = config

        if config.verbose:
            logging.basicConfig(
                level=logging.DEBUG,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            logger.setLevel(logging.DEBUG)
        else:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s'
            )

        # Initialize components
        self.lagged_correlation_engine = LaggedCorrelationEngine(config)
        self.influence_scoring_engine = InfluenceScoringEngine(config)
        self.propagation_builder = PropagationGraphBuilder(config)
        self.anomaly_detector = TemporalAnomalyDetector(config)

        # Results storage
        self._repo_series: Dict[str, RepoTimeSeries] = {}
        self._lagged_correlations: List[LaggedCorrelation] = []
        self._influence_scores: List[InfluenceScore] = []
        self._propagation_paths: List[PropagationPath] = []
        self._anomalies: List[TemporalAnomaly] = []
        self._org_report: Optional[Dict[str, Any]] = None

    def run(self) -> Tuple[TemporalIntelligenceReport, int]:
        """
        Run the complete temporal intelligence pipeline.

        Returns:
            Tuple of (TemporalIntelligenceReport, exit_code)
        """
        start_time = datetime.utcnow()

        try:
            logger.info("=" * 80)
            logger.info("ADVANCED CORRELATION & TEMPORAL INTELLIGENCE ENGINE")
            logger.info("=" * 80)

            # Step 1: Load org report
            logger.info("\nStep 1: Loading org health report...")
            self._load_org_report()
            logger.info(f"  Loaded report: {self.config.org_report_path}")

            # Step 2: Extract time series
            logger.info("\nStep 2: Extracting time series data...")
            self._extract_time_series()
            logger.info(f"  Extracted {len(self._repo_series)} repo time series")

            if len(self._repo_series) < 2:
                logger.warning("Insufficient repos for temporal analysis")
                return self._minimal_report(start_time), EXIT_TEMPORAL_SUCCESS

            # Step 3: Compute lagged correlations
            if self.config.compute_lagged_correlations:
                logger.info("\nStep 3: Computing lagged correlations...")
                self._lagged_correlations = self.lagged_correlation_engine.compute_all_lagged_correlations(
                    self._repo_series
                )
                logger.info(f"  Found {len(self._lagged_correlations)} significant lagged correlations")

            # Step 4: Compute influence scores
            if self.config.compute_influence_scores:
                logger.info("\nStep 4: Computing influence scores...")
                self._influence_scores = self.influence_scoring_engine.compute_influence_scores(
                    self._repo_series,
                    self._lagged_correlations
                )
                logger.info(f"  Computed scores for {len(self._influence_scores)} repos")

            # Step 5: Build propagation graph and detect paths
            if self.config.compute_propagation_paths:
                logger.info("\nStep 5: Building propagation graph...")
                influence_dict = self.influence_scoring_engine.get_influence_scores()
                self.propagation_builder.build_propagation_graph(
                    self._lagged_correlations,
                    influence_dict
                )
                self._propagation_paths = self.propagation_builder.detect_propagation_paths(
                    self._repo_series
                )
                logger.info(f"  Detected {len(self._propagation_paths)} propagation paths")

            # Step 6: Detect temporal anomalies
            if self.config.detect_temporal_anomalies:
                logger.info("\nStep 6: Detecting temporal anomalies...")
                self._anomalies = self.anomaly_detector.detect_anomalies(
                    self._repo_series,
                    self._lagged_correlations,
                    self._influence_scores,
                    self._propagation_paths
                )
                logger.info(f"  Detected {len(self._anomalies)} anomalies")

            # Step 7: Generate report
            logger.info("\nStep 7: Generating temporal intelligence report...")
            report = self._generate_report()

            # Calculate duration
            duration = (datetime.utcnow() - start_time).total_seconds() * 1000
            report.analysis_duration_ms = duration

            # Step 8: Write output
            if self.config.output_path:
                logger.info("\nStep 8: Writing report...")
                self._write_report(report)
                logger.info(f"  Wrote report to {self.config.output_path}")

            # Determine exit code
            exit_code = self._determine_exit_code(report)

            logger.info("\n" + "=" * 80)
            logger.info("TEMPORAL INTELLIGENCE ANALYSIS COMPLETE")
            logger.info("=" * 80)
            logger.info(f"Repos Analyzed: {report.total_repos}")
            logger.info(f"Lagged Correlations: {report.summary.significant_lagged_correlations}")
            logger.info(f"Leader Repos: {report.summary.leader_repos}")
            logger.info(f"Propagation Paths: {report.summary.propagation_paths_detected}")
            logger.info(f"Anomalies: {report.summary.total_anomalies}")
            logger.info(f"Exit Code: {exit_code}")
            logger.info("=" * 80)

            return report, exit_code

        except TemporalIntelligenceParseError as e:
            logger.error(f"Failed to parse input report: {e}")
            return self._error_report(str(e)), EXIT_TEMPORAL_PARSE_ERROR
        except TemporalIntelligenceConfigError as e:
            logger.error(f"Configuration error: {e}")
            return self._error_report(str(e)), EXIT_TEMPORAL_CONFIG_ERROR
        except Exception as e:
            logger.error(f"Temporal intelligence error: {e}")
            import traceback
            traceback.print_exc()
            return self._error_report(str(e)), EXIT_GENERAL_TEMPORAL_ERROR

    def _load_org_report(self) -> None:
        """Load org health report."""
        report_path = self.config.org_report_path

        if not report_path.exists():
            raise TemporalIntelligenceParseError(
                f"Org health report not found: {report_path}"
            )

        try:
            with open(report_path, 'r', encoding='utf-8') as f:
                self._org_report = json.load(f)

            if "repositories" not in self._org_report:
                raise TemporalIntelligenceParseError(
                    "Org report missing 'repositories' field"
                )

        except json.JSONDecodeError as e:
            raise TemporalIntelligenceParseError(
                f"Invalid JSON in org health report: {e}"
            )

    def _extract_time_series(self) -> None:
        """Extract time series data from org report."""
        repositories = self._org_report.get("repositories", [])

        for repo_data in repositories:
            repo_id = repo_data.get("repo_id", "")
            if not repo_id:
                continue

            # Extract scores from trend history
            scores = []
            timestamps = []

            trend_history = repo_data.get("trend_history", [])
            if isinstance(repo_data.get("trends"), dict):
                trend_history = repo_data["trends"].get("history", trend_history)
                trend_history = repo_data["trends"].get("data_points", trend_history)

            for hist in trend_history:
                scores.append(float(hist.get("score", hist.get("repository_score", 0.0))))
                timestamps.append(hist.get("timestamp", ""))

            # Add current score
            current_score = float(repo_data.get("repository_score", 0.0))
            scores.append(current_score)
            timestamps.append(datetime.utcnow().isoformat())

            series = RepoTimeSeries(
                repo_id=repo_id,
                repo_name=repo_data.get("repo_name", repo_id),
                scores=scores,
                timestamps=timestamps,
                current_score=current_score,
                current_trend=repo_data.get("trends", {}).get("overall_trend", "unknown")
                    if isinstance(repo_data.get("trends"), dict) else "unknown",
                risk_tier=repo_data.get("risk_tier", "low")
            )

            self._repo_series[repo_id] = series

    def _generate_report(self) -> TemporalIntelligenceReport:
        """Generate the temporal intelligence report."""
        timestamp = datetime.utcnow().isoformat()

        # Build summary
        leader_repos = [s for s in self._influence_scores
                       if s.influence_direction == InfluenceDirection.LEADER]
        follower_repos = [s for s in self._influence_scores
                        if s.influence_direction == InfluenceDirection.FOLLOWER]
        high_influence = [s for s in self._influence_scores
                        if s.influence_score >= self.config.thresholds.high_influence_threshold]

        # Unique repos in paths
        repos_in_paths = set()
        for path in self._propagation_paths:
            repos_in_paths.update(path.repo_sequence)

        # Average propagation lag
        avg_lag = 0.0
        if self._propagation_paths:
            avg_lag = sum(p.total_lag for p in self._propagation_paths) / len(self._propagation_paths)

        summary = TemporalIntelligenceSummary(
            total_repo_pairs=len(self._repo_series) * (len(self._repo_series) - 1) // 2,
            lagged_correlations_computed=len(self._lagged_correlations),
            significant_lagged_correlations=len([lc for lc in self._lagged_correlations if lc.is_significant]),
            leader_follower_pairs=len([lc for lc in self._lagged_correlations
                                       if lc.leader_repo_id and lc.follower_repo_id]),
            repos_with_influence=len([s for s in self._influence_scores if s.influence_score > 0]),
            high_influence_repos=len(high_influence),
            leader_repos=len(leader_repos),
            follower_repos=len(follower_repos),
            propagation_paths_detected=len(self._propagation_paths),
            linear_paths=len([p for p in self._propagation_paths
                            if p.path_type == PropagationType.LINEAR]),
            branching_paths=len([p for p in self._propagation_paths
                               if p.path_type == PropagationType.BRANCHING]),
            longest_path_length=max((p.path_length for p in self._propagation_paths), default=0),
            repos_in_paths=len(repos_in_paths),
            total_anomalies=len(self._anomalies),
            critical_anomalies=len([a for a in self._anomalies
                                   if a.severity == TemporalSeverity.CRITICAL]),
            high_anomalies=len([a for a in self._anomalies
                               if a.severity == TemporalSeverity.HIGH]),
            systemic_risks=len([a for a in self._anomalies
                               if a.anomaly_type == TemporalAnomalyType.SYSTEMIC_PROPAGATION]),
            avg_propagation_lag=avg_lag,
            org_interconnectedness=len(repos_in_paths) / max(len(self._repo_series), 1)
        )

        # Build recommendations
        recommendations = self._generate_recommendations()

        # Build leader ranking
        leader_ranking = [s.repo_id for s in self._influence_scores[:10]]

        # Monitoring priorities
        monitoring_priorities = []
        for score in self._influence_scores[:5]:
            if score.influence_score >= self.config.thresholds.min_influence_score:
                monitoring_priorities.append(score.repo_id)

        report = TemporalIntelligenceReport(
            report_id=f"temporal_intelligence_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            generated_at=timestamp,
            org_report_path=str(self.config.org_report_path),
            correlation_report_path=str(self.config.correlation_report_path) if self.config.correlation_report_path else "",
            summary=summary,
            lagged_correlations=self._lagged_correlations,
            optimal_lag_matrix=self.lagged_correlation_engine.get_optimal_lag_matrix(),
            influence_scores=self._influence_scores,
            leader_ranking=leader_ranking,
            propagation_paths=self._propagation_paths,
            propagation_graph=self.propagation_builder.get_graph(),
            anomalies=self._anomalies,
            recommendations=recommendations,
            monitoring_priorities=monitoring_priorities,
            org_health_status=self._org_report.get("org_health_status", "unknown"),
            org_health_score=self._org_report.get("org_health_score", 0.0),
            total_repos=len(self._repo_series)
        )

        return report

    def _generate_recommendations(self) -> List[Dict[str, Any]]:
        """Generate recommendations based on analysis."""
        recommendations = []
        rec_id = 0

        # Critical anomaly recommendations
        critical_anomalies = [a for a in self._anomalies
                            if a.severity == TemporalSeverity.CRITICAL]
        if critical_anomalies:
            rec_id += 1
            recommendations.append({
                "id": f"rec_{rec_id:03d}",
                "priority": "critical",
                "title": "Address Critical Temporal Risks",
                "message": f"{len(critical_anomalies)} critical temporal anomaly(ies) detected "
                          f"requiring immediate attention",
                "actions": critical_anomalies[0].recommended_actions[:3],
                "affected_repos": critical_anomalies[0].affected_repos[:5]
            })

        # High-influence monitoring
        high_influence = [s for s in self._influence_scores
                        if s.influence_score >= self.config.thresholds.high_influence_threshold]
        if high_influence:
            rec_id += 1
            recommendations.append({
                "id": f"rec_{rec_id:03d}",
                "priority": "high",
                "title": "Monitor High-Influence Repositories",
                "message": f"{len(high_influence)} high-influence repo(s) should be prioritized for monitoring",
                "actions": [
                    f"Set up health alerts for {high_influence[0].repo_id}",
                    "Monitor lagged correlations for early warning",
                    "Review dependencies between leader and follower repos"
                ],
                "affected_repos": [s.repo_id for s in high_influence[:5]]
            })

        # Propagation path risk
        long_paths = [p for p in self._propagation_paths if p.path_length >= 3]
        if long_paths:
            rec_id += 1
            recommendations.append({
                "id": f"rec_{rec_id:03d}",
                "priority": "medium",
                "title": "Review Long Propagation Chains",
                "message": f"{len(long_paths)} propagation path(s) with 3+ hops detected",
                "actions": [
                    "Implement circuit breakers at intermediate repos",
                    "Add isolation mechanisms to prevent cascade",
                    "Consider architectural refactoring to reduce coupling"
                ],
                "affected_repos": long_paths[0].repo_sequence[:5] if long_paths else []
            })

        return recommendations

    def _write_report(self, report: TemporalIntelligenceReport) -> None:
        """Write report to JSON file."""
        try:
            output_path = Path(self.config.output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report.to_dict(), f, indent=2, default=str)

        except Exception as e:
            raise TemporalIntelligenceError(f"Failed to write report: {e}")

    def _determine_exit_code(self, report: TemporalIntelligenceReport) -> int:
        """Determine appropriate exit code."""
        # Check for critical propagation risk
        if report.summary.critical_anomalies > 0:
            if self.config.fail_on_critical_propagation:
                return EXIT_CRITICAL_PROPAGATION_RISK

        # Check for any temporal patterns
        if report.summary.significant_lagged_correlations > 0:
            if self.config.fail_on_any_temporal_patterns:
                return EXIT_TEMPORAL_CORRELATIONS_FOUND
            return EXIT_TEMPORAL_CORRELATIONS_FOUND

        return EXIT_TEMPORAL_SUCCESS

    def _minimal_report(self, start_time: datetime) -> TemporalIntelligenceReport:
        """Generate minimal report when insufficient data."""
        duration = (datetime.utcnow() - start_time).total_seconds() * 1000

        return TemporalIntelligenceReport(
            report_id=f"temporal_intelligence_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            generated_at=datetime.utcnow().isoformat(),
            org_report_path=str(self.config.org_report_path),
            org_health_status=self._org_report.get("org_health_status", "unknown") if self._org_report else "unknown",
            org_health_score=self._org_report.get("org_health_score", 0.0) if self._org_report else 0.0,
            total_repos=len(self._repo_series),
            analysis_duration_ms=duration
        )

    def _error_report(self, error_message: str) -> TemporalIntelligenceReport:
        """Generate error report."""
        return TemporalIntelligenceReport(
            report_id=f"temporal_intelligence_error_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            generated_at=datetime.utcnow().isoformat(),
            org_report_path=str(self.config.org_report_path),
            recommendations=[{
                "id": "error",
                "priority": "critical",
                "title": "Analysis Error",
                "message": error_message,
                "actions": ["Review error and retry"],
                "affected_repos": []
            }]
        )


# ============================================================================
# Utility Functions
# ============================================================================

def create_default_thresholds() -> TemporalThresholds:
    """Create default temporal thresholds."""
    return TemporalThresholds()


def load_temporal_config(config_path: Path) -> TemporalThresholds:
    """
    Load temporal thresholds from configuration file.

    Args:
        config_path: Path to YAML or JSON config file

    Returns:
        TemporalThresholds object
    """
    if config_path.suffix in ('.yaml', '.yml'):
        try:
            import yaml
            with open(config_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
        except ImportError:
            raise TemporalIntelligenceConfigError(
                "PyYAML not installed. Install with: pip install pyyaml"
            )
    else:
        with open(config_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

    return TemporalThresholds.from_dict(data.get("thresholds", {}))
