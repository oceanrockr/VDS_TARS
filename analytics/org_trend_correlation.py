"""
Multi-Repository Trend Correlation Engine

This module implements a cross-repository trend correlation layer that:
1. Analyzes trend patterns across multiple repositories
2. Computes pairwise correlation coefficients between repo trends
3. Detects synchronized declines, shared volatility, and correlated issue spikes
4. Clusters related repositories based on trend similarity
5. Generates anomalies and predictive indicators

Architecture:
- RepoTrendSeries: Time-series data for a repository's health metrics
- TrendCorrelation: Pairwise correlation result between two repositories
- CorrelationCluster: Group of repositories with correlated trends
- CrossRepoAnomaly: Detected cross-repository anomaly or pattern
- TrendCorrelationReport: Complete output with correlations, clusters, anomalies

Exit Codes (120-129):
- 120: Success, no concerning correlations
- 121: Correlations found (non-critical)
- 122: Critical cross-repo anomaly detected
- 123: Config error
- 124: Parsing error
- 199: General correlation error

Input:
    org-health-report.json from Phase 14.8 Task 1

Output:
    trend-correlation-report.json

Version: 1.0.0
Phase: 14.8 Task 3
"""

import json
import logging
import math
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set, Union

logger = logging.getLogger(__name__)


# ============================================================================
# Exit Codes (120-129)
# ============================================================================

EXIT_CORRELATION_SUCCESS = 120
EXIT_CORRELATIONS_FOUND = 121
EXIT_CRITICAL_ANOMALY = 122
EXIT_CORRELATION_CONFIG_ERROR = 123
EXIT_CORRELATION_PARSE_ERROR = 124
EXIT_GENERAL_CORRELATION_ERROR = 199


# ============================================================================
# Custom Exceptions
# ============================================================================

class TrendCorrelationError(Exception):
    """Base exception for trend correlation engine errors."""
    exit_code = EXIT_GENERAL_CORRELATION_ERROR


class TrendCorrelationConfigError(TrendCorrelationError):
    """Configuration error in correlation engine."""
    exit_code = EXIT_CORRELATION_CONFIG_ERROR


class TrendCorrelationParseError(TrendCorrelationError):
    """Failed to parse org health report."""
    exit_code = EXIT_CORRELATION_PARSE_ERROR


# ============================================================================
# Enums
# ============================================================================

class TrendDirection(Enum):
    """Direction of a repository's health trend."""
    IMPROVING = "improving"
    STABLE = "stable"
    DECLINING = "declining"
    UNKNOWN = "unknown"

    @classmethod
    def from_string(cls, value: str) -> "TrendDirection":
        """Convert string to TrendDirection."""
        value_lower = value.lower() if value else "unknown"
        try:
            return cls(value_lower)
        except ValueError:
            return cls.UNKNOWN


class CorrelationType(Enum):
    """Type of correlation detected between repositories."""
    POSITIVE = "positive"          # Repos move in same direction
    NEGATIVE = "negative"          # Repos move in opposite directions
    SYNCHRONIZED_DECLINE = "synchronized_decline"   # Both declining together
    SYNCHRONIZED_IMPROVEMENT = "synchronized_improvement"  # Both improving together
    SHARED_VOLATILITY = "shared_volatility"  # Both have high volatility
    ISSUE_SPIKE_CORRELATED = "issue_spike_correlated"  # Issue counts correlated
    NONE = "none"


class AnomalySeverity(Enum):
    """Severity of a detected cross-repo anomaly."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    def __lt__(self, other: "AnomalySeverity") -> bool:
        order = {self.LOW: 0, self.MEDIUM: 1, self.HIGH: 2, self.CRITICAL: 3}
        return order[self] < order[other]

    def __le__(self, other: "AnomalySeverity") -> bool:
        order = {self.LOW: 0, self.MEDIUM: 1, self.HIGH: 2, self.CRITICAL: 3}
        return order[self] <= order[other]

    @classmethod
    def from_string(cls, value: str) -> "AnomalySeverity":
        """Convert string to AnomalySeverity."""
        value_lower = value.lower() if value else "low"
        try:
            return cls(value_lower)
        except ValueError:
            return cls.LOW


class AnomalyType(Enum):
    """Type of cross-repository anomaly detected."""
    SYNCHRONIZED_DECLINE = "synchronized_decline"
    EMERGING_RISK_CLUSTER = "emerging_risk_cluster"
    LEADING_INDICATOR = "leading_indicator"
    CASCADING_FAILURE = "cascading_failure"
    SHARED_VOLATILITY = "shared_volatility"
    CORRELATED_ISSUE_SPIKE = "correlated_issue_spike"
    DIVERGENCE = "divergence"
    RECOVERY_PATTERN = "recovery_pattern"


class ClusterMethod(Enum):
    """Method used for clustering repositories."""
    HIERARCHICAL = "hierarchical"
    DENSITY = "density"
    THRESHOLD = "threshold"


# ============================================================================
# Data Classes - Trend Series
# ============================================================================

@dataclass
class TrendDataPoint:
    """
    Single data point in a repository's trend series.

    Represents a snapshot of repository health at a specific time.
    """
    timestamp: str
    score: float
    health_status: str
    critical_issues: int = 0
    total_issues: int = 0
    trend_direction: str = "unknown"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrendDataPoint":
        """Create from dictionary."""
        return cls(
            timestamp=data.get("timestamp", ""),
            score=float(data.get("score", data.get("repository_score", 0.0))),
            health_status=data.get("health_status", data.get("overall_health", "unknown")),
            critical_issues=int(data.get("critical_issues", 0)),
            total_issues=int(data.get("total_issues", 0)),
            trend_direction=data.get("trend_direction", "unknown")
        )


@dataclass
class RepoTrendSeries:
    """
    Time-series trend data for a single repository.

    Contains historical health metrics used for correlation analysis.
    """
    repo_id: str
    repo_name: str

    # Time series data points
    data_points: List[TrendDataPoint] = field(default_factory=list)

    # Current state snapshot
    current_score: float = 0.0
    current_health_status: str = "unknown"
    current_trend_direction: str = "unknown"
    current_critical_issues: int = 0
    risk_tier: str = "low"

    # Computed statistics
    score_mean: float = 0.0
    score_std: float = 0.0
    score_min: float = 0.0
    score_max: float = 100.0
    score_volatility: float = 0.0

    # Trend metrics
    trend_slope: float = 0.0  # Positive = improving, negative = declining
    intervals_declining: int = 0
    intervals_improving: int = 0
    intervals_stable: int = 0

    # Issue metrics
    issue_trend_slope: float = 0.0
    max_issue_spike: int = 0

    def compute_statistics(self) -> None:
        """Compute statistical metrics from data points."""
        if not self.data_points:
            return

        scores = [dp.score for dp in self.data_points]
        n = len(scores)

        # Basic statistics
        self.score_mean = sum(scores) / n
        self.score_min = min(scores)
        self.score_max = max(scores)

        # Standard deviation
        if n > 1:
            variance = sum((s - self.score_mean) ** 2 for s in scores) / (n - 1)
            self.score_std = math.sqrt(variance)

        # Volatility (coefficient of variation)
        if self.score_mean > 0:
            self.score_volatility = (self.score_std / self.score_mean) * 100

        # Trend slope (simple linear regression)
        if n > 1:
            x_mean = (n - 1) / 2
            numerator = sum((i - x_mean) * (scores[i] - self.score_mean) for i in range(n))
            denominator = sum((i - x_mean) ** 2 for i in range(n))
            if denominator > 0:
                self.trend_slope = numerator / denominator

        # Count intervals by direction
        for dp in self.data_points:
            direction = dp.trend_direction.lower()
            if direction == "declining":
                self.intervals_declining += 1
            elif direction == "improving":
                self.intervals_improving += 1
            else:
                self.intervals_stable += 1

        # Issue metrics
        issues = [dp.critical_issues for dp in self.data_points]
        if len(issues) > 1:
            issue_changes = [issues[i] - issues[i-1] for i in range(1, len(issues))]
            self.max_issue_spike = max(issue_changes) if issue_changes else 0

            # Issue trend slope
            issue_mean = sum(issues) / len(issues)
            x_mean = (len(issues) - 1) / 2
            num = sum((i - x_mean) * (issues[i] - issue_mean) for i in range(len(issues)))
            denom = sum((i - x_mean) ** 2 for i in range(len(issues)))
            if denom > 0:
                self.issue_trend_slope = num / denom

    def get_score_series(self) -> List[float]:
        """Get list of scores as numeric series."""
        return [dp.score for dp in self.data_points]

    def get_issue_series(self) -> List[int]:
        """Get list of critical issue counts as numeric series."""
        return [dp.critical_issues for dp in self.data_points]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "repo_id": self.repo_id,
            "repo_name": self.repo_name,
            "data_points": [dp.to_dict() for dp in self.data_points],
            "current_score": self.current_score,
            "current_health_status": self.current_health_status,
            "current_trend_direction": self.current_trend_direction,
            "current_critical_issues": self.current_critical_issues,
            "risk_tier": self.risk_tier,
            "score_mean": self.score_mean,
            "score_std": self.score_std,
            "score_min": self.score_min,
            "score_max": self.score_max,
            "score_volatility": self.score_volatility,
            "trend_slope": self.trend_slope,
            "intervals_declining": self.intervals_declining,
            "intervals_improving": self.intervals_improving,
            "intervals_stable": self.intervals_stable,
            "issue_trend_slope": self.issue_trend_slope,
            "max_issue_spike": self.max_issue_spike
        }

    @classmethod
    def from_repo_data(
        cls,
        repo_data: Dict[str, Any],
        trend_history: Optional[List[Dict[str, Any]]] = None
    ) -> "RepoTrendSeries":
        """
        Create RepoTrendSeries from repository data in org-health-report.

        Args:
            repo_data: Repository data from org-health-report.json
            trend_history: Optional historical trend data points

        Returns:
            RepoTrendSeries instance
        """
        series = cls(
            repo_id=repo_data.get("repo_id", ""),
            repo_name=repo_data.get("repo_name", repo_data.get("repo_id", "")),
            current_score=float(repo_data.get("repository_score", 0.0)),
            current_health_status=repo_data.get("health_status", "unknown"),
            current_trend_direction=repo_data.get("trends", {}).get("overall_trend", "unknown")
                if isinstance(repo_data.get("trends"), dict)
                else "unknown",
            current_critical_issues=int(repo_data.get("critical_issues", 0)),
            risk_tier=repo_data.get("risk_tier", "low")
        )

        # Add current state as a data point
        current_point = TrendDataPoint(
            timestamp=datetime.utcnow().isoformat(),
            score=series.current_score,
            health_status=series.current_health_status,
            critical_issues=series.current_critical_issues,
            total_issues=int(repo_data.get("total_issues", 0)),
            trend_direction=series.current_trend_direction
        )
        series.data_points.append(current_point)

        # Add historical data points if available
        if trend_history:
            for hist in trend_history:
                dp = TrendDataPoint.from_dict(hist)
                series.data_points.insert(0, dp)  # Insert at beginning (oldest first)

        # Compute statistics
        series.compute_statistics()

        return series


# ============================================================================
# Data Classes - Correlation Results
# ============================================================================

@dataclass
class TrendCorrelation:
    """
    Pairwise correlation result between two repositories.

    Captures the statistical correlation between their trend series
    and classifies the correlation type.
    """
    repo_a_id: str
    repo_b_id: str

    # Correlation metrics
    pearson_coefficient: float = 0.0  # -1 to 1
    spearman_coefficient: float = 0.0  # -1 to 1 (rank-based)
    issue_correlation: float = 0.0  # Correlation of issue counts

    # Classification
    correlation_type: CorrelationType = CorrelationType.NONE
    correlation_strength: str = "none"  # weak, moderate, strong
    is_significant: bool = False

    # Additional context
    shared_decline_periods: int = 0
    shared_improvement_periods: int = 0
    volatility_similarity: float = 0.0  # How similar their volatility is

    # Risk implications
    combined_risk_score: float = 0.0
    risk_implications: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "repo_a_id": self.repo_a_id,
            "repo_b_id": self.repo_b_id,
            "pearson_coefficient": self.pearson_coefficient,
            "spearman_coefficient": self.spearman_coefficient,
            "issue_correlation": self.issue_correlation,
            "correlation_type": self.correlation_type.value,
            "correlation_strength": self.correlation_strength,
            "is_significant": self.is_significant,
            "shared_decline_periods": self.shared_decline_periods,
            "shared_improvement_periods": self.shared_improvement_periods,
            "volatility_similarity": self.volatility_similarity,
            "combined_risk_score": self.combined_risk_score,
            "risk_implications": self.risk_implications
        }


@dataclass
class CorrelationCluster:
    """
    Group of repositories with correlated trends.

    Represents repositories that exhibit similar health patterns
    and may share common factors affecting their health.
    """
    cluster_id: str
    cluster_name: str

    # Member repositories
    repo_ids: List[str] = field(default_factory=list)
    repo_count: int = 0

    # Cluster characteristics
    cluster_method: ClusterMethod = ClusterMethod.THRESHOLD
    avg_internal_correlation: float = 0.0
    cluster_density: float = 0.0  # How tightly correlated

    # Trend summary
    dominant_trend: str = "unknown"  # Most common trend in cluster
    percent_declining: float = 0.0
    percent_improving: float = 0.0
    avg_score: float = 0.0
    avg_risk_score: float = 0.0

    # Risk assessment
    cluster_risk_tier: str = "low"
    is_risk_cluster: bool = False  # True if majority high-risk

    # Contextual info
    potential_shared_factors: List[str] = field(default_factory=list)
    recommended_actions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "cluster_id": self.cluster_id,
            "cluster_name": self.cluster_name,
            "repo_ids": self.repo_ids,
            "repo_count": self.repo_count,
            "cluster_method": self.cluster_method.value,
            "avg_internal_correlation": self.avg_internal_correlation,
            "cluster_density": self.cluster_density,
            "dominant_trend": self.dominant_trend,
            "percent_declining": self.percent_declining,
            "percent_improving": self.percent_improving,
            "avg_score": self.avg_score,
            "avg_risk_score": self.avg_risk_score,
            "cluster_risk_tier": self.cluster_risk_tier,
            "is_risk_cluster": self.is_risk_cluster,
            "potential_shared_factors": self.potential_shared_factors,
            "recommended_actions": self.recommended_actions
        }


# ============================================================================
# Data Classes - Anomalies
# ============================================================================

@dataclass
class CrossRepoAnomaly:
    """
    Detected cross-repository anomaly or pattern.

    Represents a notable pattern detected across multiple repositories
    that may indicate systemic issues or shared risk factors.
    """
    anomaly_id: str
    anomaly_type: AnomalyType
    severity: AnomalySeverity

    # Description
    title: str
    message: str
    timestamp: str

    # Affected repos
    affected_repos: List[str] = field(default_factory=list)
    affected_count: int = 0

    # Metrics that triggered detection
    trigger_metric: str = ""
    trigger_value: float = 0.0
    trigger_threshold: float = 0.0

    # Additional context
    details: Dict[str, Any] = field(default_factory=dict)
    evidence: List[str] = field(default_factory=list)

    # Predictive indicators
    is_predictive: bool = False
    predicted_impact: str = ""
    confidence: float = 0.0

    # Recommendations
    recommended_actions: List[str] = field(default_factory=list)

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
            "trigger_metric": self.trigger_metric,
            "trigger_value": self.trigger_value,
            "trigger_threshold": self.trigger_threshold,
            "details": self.details,
            "evidence": self.evidence,
            "is_predictive": self.is_predictive,
            "predicted_impact": self.predicted_impact,
            "confidence": self.confidence,
            "recommended_actions": self.recommended_actions
        }


# ============================================================================
# Data Classes - Configuration
# ============================================================================

@dataclass
class CorrelationThresholds:
    """
    Configurable thresholds for correlation detection.
    """
    # Correlation coefficient thresholds
    min_correlation_weak: float = 0.3
    min_correlation_moderate: float = 0.5
    min_correlation_strong: float = 0.7

    # Significance threshold
    significance_threshold: float = 0.5  # Minimum to consider significant

    # Clustering thresholds
    min_cluster_correlation: float = 0.6  # Minimum to include in cluster
    min_cluster_size: int = 2
    max_cluster_size: int = 50

    # Anomaly detection thresholds
    synchronized_decline_threshold: float = 0.20  # >= 20% repos declining together
    synchronized_decline_min_repos: int = 3
    volatility_similarity_threshold: float = 0.8  # High volatility similarity
    issue_spike_correlation_threshold: float = 0.6

    # Leading indicator thresholds
    leading_indicator_lag: int = 1  # Number of intervals to check lag
    leading_indicator_correlation: float = 0.7

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CorrelationThresholds":
        """Create from dictionary."""
        return cls(
            min_correlation_weak=data.get("min_correlation_weak", 0.3),
            min_correlation_moderate=data.get("min_correlation_moderate", 0.5),
            min_correlation_strong=data.get("min_correlation_strong", 0.7),
            significance_threshold=data.get("significance_threshold", 0.5),
            min_cluster_correlation=data.get("min_cluster_correlation", 0.6),
            min_cluster_size=data.get("min_cluster_size", 2),
            max_cluster_size=data.get("max_cluster_size", 50),
            synchronized_decline_threshold=data.get("synchronized_decline_threshold", 0.20),
            synchronized_decline_min_repos=data.get("synchronized_decline_min_repos", 3),
            volatility_similarity_threshold=data.get("volatility_similarity_threshold", 0.8),
            issue_spike_correlation_threshold=data.get("issue_spike_correlation_threshold", 0.6),
            leading_indicator_lag=data.get("leading_indicator_lag", 1),
            leading_indicator_correlation=data.get("leading_indicator_correlation", 0.7)
        )


@dataclass
class TrendCorrelationConfig:
    """
    Configuration for the Trend Correlation Engine.
    """
    # Input
    org_report_path: Path

    # Output
    output_path: Optional[Path] = None

    # Thresholds
    thresholds: CorrelationThresholds = field(default_factory=CorrelationThresholds)

    # Analysis options
    compute_clusters: bool = True
    detect_anomalies: bool = True
    compute_leading_indicators: bool = True

    # Behavior flags
    verbose: bool = False
    summary_only: bool = False

    # CI/CD flags
    fail_on_critical_anomaly: bool = False
    fail_on_any_correlations: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "org_report_path": str(self.org_report_path),
            "output_path": str(self.output_path) if self.output_path else None,
            "thresholds": self.thresholds.to_dict(),
            "compute_clusters": self.compute_clusters,
            "detect_anomalies": self.detect_anomalies,
            "compute_leading_indicators": self.compute_leading_indicators,
            "verbose": self.verbose,
            "summary_only": self.summary_only,
            "fail_on_critical_anomaly": self.fail_on_critical_anomaly,
            "fail_on_any_correlations": self.fail_on_any_correlations
        }


# ============================================================================
# Data Classes - Report
# ============================================================================

@dataclass
class CorrelationSummary:
    """Summary statistics for correlation analysis."""
    total_repo_pairs: int = 0
    significant_correlations: int = 0
    positive_correlations: int = 0
    negative_correlations: int = 0

    # Synchronized patterns
    synchronized_decline_pairs: int = 0
    synchronized_improvement_pairs: int = 0
    shared_volatility_pairs: int = 0

    # Cluster summary
    total_clusters: int = 0
    largest_cluster_size: int = 0
    risk_clusters: int = 0

    # Anomaly summary
    total_anomalies: int = 0
    critical_anomalies: int = 0
    high_anomalies: int = 0
    predictive_indicators: int = 0

    # Org-level indicators
    avg_inter_repo_correlation: float = 0.0
    correlation_density: float = 0.0  # % of pairs with significant correlation

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class TrendCorrelationReport:
    """
    Complete Trend Correlation Report.

    Contains all correlation analysis results including:
    - Correlation matrix
    - Repository clusters
    - Detected anomalies
    - Predictive indicators
    - Recommendations
    """
    # Report metadata
    report_id: str
    generated_at: str
    org_report_path: str

    # Summary
    summary: CorrelationSummary = field(default_factory=CorrelationSummary)

    # Trend series for all repos
    repo_series: List[Dict[str, Any]] = field(default_factory=list)

    # Correlation matrix (significant pairs only)
    correlations: List[TrendCorrelation] = field(default_factory=list)

    # Full correlation matrix (for export)
    correlation_matrix: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Clusters
    clusters: List[CorrelationCluster] = field(default_factory=list)

    # Anomalies
    anomalies: List[CrossRepoAnomaly] = field(default_factory=list)

    # Predictive indicators
    predictive_indicators: List[Dict[str, Any]] = field(default_factory=list)

    # Recommendations
    recommendations: List[Dict[str, Any]] = field(default_factory=list)

    # Source org report summary
    org_health_status: str = "unknown"
    org_health_score: float = 0.0
    total_repos: int = 0

    # Metadata
    analysis_duration_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "report_id": self.report_id,
            "generated_at": self.generated_at,
            "org_report_path": self.org_report_path,
            "summary": self.summary.to_dict(),
            "repo_series": self.repo_series,
            "correlations": [c.to_dict() for c in self.correlations],
            "correlation_matrix": self.correlation_matrix,
            "clusters": [c.to_dict() for c in self.clusters],
            "anomalies": [a.to_dict() for a in self.anomalies],
            "predictive_indicators": self.predictive_indicators,
            "recommendations": self.recommendations,
            "org_health_status": self.org_health_status,
            "org_health_score": self.org_health_score,
            "total_repos": self.total_repos,
            "analysis_duration_ms": self.analysis_duration_ms
        }


# ============================================================================
# Trend Loader
# ============================================================================

class TrendLoader:
    """
    Loads and prepares trend data from org-health-report.json.

    Extracts repository health snapshots and converts them to
    time-series format for correlation analysis.
    """

    def __init__(self, config: TrendCorrelationConfig):
        """Initialize trend loader."""
        self.config = config
        self._org_report: Optional[Dict[str, Any]] = None
        self._repo_series: Dict[str, RepoTrendSeries] = {}

    def load_org_report(self) -> Dict[str, Any]:
        """
        Load org health report from JSON file.

        Returns:
            Parsed org health report

        Raises:
            TrendCorrelationParseError: If loading fails
        """
        report_path = self.config.org_report_path

        if not report_path.exists():
            raise TrendCorrelationParseError(
                f"Org health report not found: {report_path}"
            )

        try:
            with open(report_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Validate required fields
            if "repositories" not in data and "repos_loaded" not in data:
                raise TrendCorrelationParseError(
                    "Org report missing 'repositories' field"
                )

            self._org_report = data
            return data

        except json.JSONDecodeError as e:
            raise TrendCorrelationParseError(
                f"Invalid JSON in org health report: {e}"
            )
        except Exception as e:
            raise TrendCorrelationParseError(
                f"Failed to load org health report: {e}"
            )

    def extract_trend_series(self) -> Dict[str, RepoTrendSeries]:
        """
        Extract trend series for all repositories.

        Returns:
            Dictionary mapping repo_id to RepoTrendSeries
        """
        if not self._org_report:
            self.load_org_report()

        repositories = self._org_report.get("repositories", [])

        for repo_data in repositories:
            repo_id = repo_data.get("repo_id", "")
            if not repo_id:
                continue

            # Extract trend history if available
            trend_history = self._extract_trend_history(repo_data)

            # Create series
            series = RepoTrendSeries.from_repo_data(repo_data, trend_history)
            self._repo_series[repo_id] = series

            logger.debug(f"Loaded trend series for {repo_id}: "
                        f"{len(series.data_points)} points")

        logger.info(f"Extracted trend series for {len(self._repo_series)} repos")
        return self._repo_series

    def _extract_trend_history(
        self,
        repo_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Extract historical trend data points from repository data.

        Args:
            repo_data: Repository data from org report

        Returns:
            List of historical data point dictionaries
        """
        history = []

        # Check for trend_history field (if explicitly provided)
        if "trend_history" in repo_data:
            history = repo_data["trend_history"]

        # Check trends field for historical data
        elif "trends" in repo_data and isinstance(repo_data["trends"], dict):
            trends = repo_data["trends"]
            if "history" in trends:
                history = trends["history"]
            elif "data_points" in trends:
                history = trends["data_points"]

        return history

    def get_repo_series(self, repo_id: str) -> Optional[RepoTrendSeries]:
        """Get trend series for a specific repository."""
        return self._repo_series.get(repo_id)

    def get_all_series(self) -> Dict[str, RepoTrendSeries]:
        """Get all loaded trend series."""
        return self._repo_series

    def get_org_report(self) -> Optional[Dict[str, Any]]:
        """Get the loaded org report."""
        return self._org_report


# ============================================================================
# Correlation Matrix Builder
# ============================================================================

class CorrelationMatrixBuilder:
    """
    Builds correlation matrix between repository trend series.

    Computes pairwise Pearson and Spearman correlation coefficients
    and classifies correlation types.
    """

    def __init__(self, config: TrendCorrelationConfig):
        """Initialize correlation matrix builder."""
        self.config = config
        self.thresholds = config.thresholds
        self._matrix: Dict[str, Dict[str, float]] = {}
        self._correlations: List[TrendCorrelation] = []

    def build_matrix(
        self,
        repo_series: Dict[str, RepoTrendSeries]
    ) -> Dict[str, Dict[str, float]]:
        """
        Build full correlation matrix for all repository pairs.

        Args:
            repo_series: Dictionary of repo_id -> RepoTrendSeries

        Returns:
            Nested dictionary representing correlation matrix
        """
        repo_ids = list(repo_series.keys())
        n = len(repo_ids)

        # Initialize matrix
        self._matrix = {rid: {} for rid in repo_ids}

        # Compute pairwise correlations
        for i in range(n):
            repo_a_id = repo_ids[i]
            series_a = repo_series[repo_a_id]

            for j in range(i, n):
                repo_b_id = repo_ids[j]
                series_b = repo_series[repo_b_id]

                if i == j:
                    # Self-correlation is 1.0
                    self._matrix[repo_a_id][repo_b_id] = 1.0
                else:
                    # Compute correlation
                    correlation = self._compute_correlation(series_a, series_b)

                    # Store in matrix (symmetric)
                    self._matrix[repo_a_id][repo_b_id] = correlation.pearson_coefficient
                    self._matrix[repo_b_id][repo_a_id] = correlation.pearson_coefficient

                    # Store detailed correlation if significant
                    if correlation.is_significant:
                        self._correlations.append(correlation)

        logger.info(f"Built correlation matrix: {n}x{n} "
                   f"({len(self._correlations)} significant pairs)")

        return self._matrix

    def _compute_correlation(
        self,
        series_a: RepoTrendSeries,
        series_b: RepoTrendSeries
    ) -> TrendCorrelation:
        """
        Compute correlation between two repository trend series.

        Args:
            series_a: First repository's trend series
            series_b: Second repository's trend series

        Returns:
            TrendCorrelation with computed metrics
        """
        result = TrendCorrelation(
            repo_a_id=series_a.repo_id,
            repo_b_id=series_b.repo_id
        )

        # Get score series
        scores_a = series_a.get_score_series()
        scores_b = series_b.get_score_series()

        # Align series lengths (use shorter length)
        min_len = min(len(scores_a), len(scores_b))
        if min_len < 2:
            # Not enough data for correlation
            return result

        scores_a = scores_a[-min_len:]
        scores_b = scores_b[-min_len:]

        # Compute Pearson correlation
        result.pearson_coefficient = self._pearson_correlation(scores_a, scores_b)

        # Compute Spearman correlation
        result.spearman_coefficient = self._spearman_correlation(scores_a, scores_b)

        # Compute issue correlation
        issues_a = series_a.get_issue_series()[-min_len:]
        issues_b = series_b.get_issue_series()[-min_len:]
        if sum(issues_a) > 0 or sum(issues_b) > 0:
            result.issue_correlation = self._pearson_correlation(
                [float(i) for i in issues_a],
                [float(i) for i in issues_b]
            )

        # Compute shared decline/improvement periods
        for i in range(len(series_a.data_points)):
            if i >= len(series_b.data_points):
                break
            dir_a = series_a.data_points[i].trend_direction.lower()
            dir_b = series_b.data_points[i].trend_direction.lower()

            if dir_a == "declining" and dir_b == "declining":
                result.shared_decline_periods += 1
            elif dir_a == "improving" and dir_b == "improving":
                result.shared_improvement_periods += 1

        # Compute volatility similarity
        if series_a.score_volatility > 0 and series_b.score_volatility > 0:
            vol_ratio = min(series_a.score_volatility, series_b.score_volatility) / \
                       max(series_a.score_volatility, series_b.score_volatility)
            result.volatility_similarity = vol_ratio

        # Classify correlation
        result.correlation_type, result.correlation_strength = self._classify_correlation(result)

        # Determine significance
        abs_corr = abs(result.pearson_coefficient)
        result.is_significant = abs_corr >= self.thresholds.significance_threshold

        # Compute combined risk
        result.combined_risk_score = (
            float(series_a.risk_tier in ("high", "critical")) * 50 +
            float(series_b.risk_tier in ("high", "critical")) * 50
        )

        # Risk implications
        if result.shared_decline_periods > 0 and result.combined_risk_score > 0:
            result.risk_implications.append(
                f"Both repos declining together ({result.shared_decline_periods} periods)"
            )

        if result.correlation_type == CorrelationType.SYNCHRONIZED_DECLINE:
            result.risk_implications.append(
                "Strong synchronized decline pattern"
            )

        return result

    def _pearson_correlation(
        self,
        x: List[float],
        y: List[float]
    ) -> float:
        """Compute Pearson correlation coefficient."""
        n = len(x)
        if n < 2:
            return 0.0

        # Means
        mean_x = sum(x) / n
        mean_y = sum(y) / n

        # Covariance and standard deviations
        cov = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
        std_x = math.sqrt(sum((xi - mean_x) ** 2 for xi in x))
        std_y = math.sqrt(sum((yi - mean_y) ** 2 for yi in y))

        if std_x == 0 or std_y == 0:
            return 0.0

        return cov / (std_x * std_y)

    def _spearman_correlation(
        self,
        x: List[float],
        y: List[float]
    ) -> float:
        """Compute Spearman rank correlation coefficient."""
        n = len(x)
        if n < 2:
            return 0.0

        # Convert to ranks
        rank_x = self._rank(x)
        rank_y = self._rank(y)

        # Pearson correlation of ranks
        return self._pearson_correlation(rank_x, rank_y)

    def _rank(self, values: List[float]) -> List[float]:
        """Convert values to ranks."""
        n = len(values)
        indexed = [(v, i) for i, v in enumerate(values)]
        indexed.sort(key=lambda x: x[0])

        ranks = [0.0] * n
        for rank, (_, original_idx) in enumerate(indexed, 1):
            ranks[original_idx] = float(rank)

        return ranks

    def _classify_correlation(
        self,
        correlation: TrendCorrelation
    ) -> Tuple[CorrelationType, str]:
        """
        Classify correlation type and strength.

        Args:
            correlation: TrendCorrelation with computed metrics

        Returns:
            Tuple of (CorrelationType, strength_string)
        """
        coef = correlation.pearson_coefficient
        abs_coef = abs(coef)

        # Determine strength
        if abs_coef >= self.thresholds.min_correlation_strong:
            strength = "strong"
        elif abs_coef >= self.thresholds.min_correlation_moderate:
            strength = "moderate"
        elif abs_coef >= self.thresholds.min_correlation_weak:
            strength = "weak"
        else:
            return CorrelationType.NONE, "none"

        # Determine type
        if coef >= self.thresholds.min_correlation_moderate:
            # Positive correlation
            if correlation.shared_decline_periods >= 2:
                return CorrelationType.SYNCHRONIZED_DECLINE, strength
            elif correlation.shared_improvement_periods >= 2:
                return CorrelationType.SYNCHRONIZED_IMPROVEMENT, strength
            else:
                return CorrelationType.POSITIVE, strength

        elif coef <= -self.thresholds.min_correlation_moderate:
            return CorrelationType.NEGATIVE, strength

        # Check for shared volatility
        if correlation.volatility_similarity >= self.thresholds.volatility_similarity_threshold:
            return CorrelationType.SHARED_VOLATILITY, strength

        # Check for issue correlation
        if abs(correlation.issue_correlation) >= self.thresholds.issue_spike_correlation_threshold:
            return CorrelationType.ISSUE_SPIKE_CORRELATED, strength

        return CorrelationType.POSITIVE if coef > 0 else CorrelationType.NEGATIVE, strength

    def get_matrix(self) -> Dict[str, Dict[str, float]]:
        """Get the correlation matrix."""
        return self._matrix

    def get_significant_correlations(self) -> List[TrendCorrelation]:
        """Get list of significant correlations."""
        return self._correlations


# ============================================================================
# Cluster Detector
# ============================================================================

class ClusterDetector:
    """
    Detects clusters of correlated repositories.

    Uses threshold-based clustering with density refinement.
    """

    def __init__(self, config: TrendCorrelationConfig):
        """Initialize cluster detector."""
        self.config = config
        self.thresholds = config.thresholds
        self._clusters: List[CorrelationCluster] = []
        self._cluster_counter = 0

    def detect_clusters(
        self,
        repo_series: Dict[str, RepoTrendSeries],
        correlation_matrix: Dict[str, Dict[str, float]],
        correlations: List[TrendCorrelation]
    ) -> List[CorrelationCluster]:
        """
        Detect repository clusters based on correlation patterns.

        Args:
            repo_series: Dictionary of repo_id -> RepoTrendSeries
            correlation_matrix: Full correlation matrix
            correlations: List of significant correlations

        Returns:
            List of detected CorrelationCluster objects
        """
        self._clusters = []

        # Build adjacency list of strongly correlated repos
        adjacency = self._build_adjacency(correlations)

        # Find connected components (clusters)
        visited = set()

        for repo_id in adjacency:
            if repo_id in visited:
                continue

            # BFS to find connected component
            cluster_repos = self._find_connected_component(repo_id, adjacency, visited)

            # Only create cluster if meets minimum size
            if len(cluster_repos) >= self.thresholds.min_cluster_size:
                cluster = self._create_cluster(
                    cluster_repos, repo_series, correlation_matrix
                )
                self._clusters.append(cluster)

        logger.info(f"Detected {len(self._clusters)} clusters")

        return self._clusters

    def _build_adjacency(
        self,
        correlations: List[TrendCorrelation]
    ) -> Dict[str, Set[str]]:
        """Build adjacency list from significant correlations."""
        adjacency: Dict[str, Set[str]] = {}

        for corr in correlations:
            if abs(corr.pearson_coefficient) < self.thresholds.min_cluster_correlation:
                continue

            # Add bidirectional edges
            if corr.repo_a_id not in adjacency:
                adjacency[corr.repo_a_id] = set()
            if corr.repo_b_id not in adjacency:
                adjacency[corr.repo_b_id] = set()

            adjacency[corr.repo_a_id].add(corr.repo_b_id)
            adjacency[corr.repo_b_id].add(corr.repo_a_id)

        return adjacency

    def _find_connected_component(
        self,
        start_repo: str,
        adjacency: Dict[str, Set[str]],
        visited: Set[str]
    ) -> List[str]:
        """Find connected component using BFS."""
        component = []
        queue = [start_repo]

        while queue:
            repo_id = queue.pop(0)
            if repo_id in visited:
                continue

            visited.add(repo_id)
            component.append(repo_id)

            # Add neighbors to queue
            if repo_id in adjacency:
                for neighbor in adjacency[repo_id]:
                    if neighbor not in visited:
                        queue.append(neighbor)

        return component

    def _create_cluster(
        self,
        repo_ids: List[str],
        repo_series: Dict[str, RepoTrendSeries],
        correlation_matrix: Dict[str, Dict[str, float]]
    ) -> CorrelationCluster:
        """Create a cluster object with computed metrics."""
        self._cluster_counter += 1

        cluster = CorrelationCluster(
            cluster_id=f"cluster_{self._cluster_counter:03d}",
            cluster_name=f"Correlation Cluster {self._cluster_counter}",
            repo_ids=repo_ids,
            repo_count=len(repo_ids),
            cluster_method=ClusterMethod.THRESHOLD
        )

        # Compute internal correlation
        internal_corrs = []
        for i, repo_a in enumerate(repo_ids):
            for repo_b in repo_ids[i+1:]:
                if repo_a in correlation_matrix and repo_b in correlation_matrix[repo_a]:
                    internal_corrs.append(correlation_matrix[repo_a][repo_b])

        if internal_corrs:
            cluster.avg_internal_correlation = sum(internal_corrs) / len(internal_corrs)
            # Density: ratio of strong correlations
            strong = sum(1 for c in internal_corrs if c >= self.thresholds.min_correlation_strong)
            cluster.cluster_density = strong / len(internal_corrs) if internal_corrs else 0.0

        # Compute trend summary
        declining = 0
        improving = 0
        scores = []
        risk_scores = []
        high_risk_count = 0

        for repo_id in repo_ids:
            if repo_id not in repo_series:
                continue

            series = repo_series[repo_id]

            direction = series.current_trend_direction.lower()
            if direction == "declining":
                declining += 1
            elif direction == "improving":
                improving += 1

            scores.append(series.current_score)

            # Risk tracking
            if series.risk_tier in ("high", "critical"):
                high_risk_count += 1
                risk_scores.append(80.0)
            elif series.risk_tier == "medium":
                risk_scores.append(50.0)
            else:
                risk_scores.append(20.0)

        n = len(repo_ids)
        cluster.percent_declining = declining / n if n > 0 else 0.0
        cluster.percent_improving = improving / n if n > 0 else 0.0
        cluster.avg_score = sum(scores) / len(scores) if scores else 0.0
        cluster.avg_risk_score = sum(risk_scores) / len(risk_scores) if risk_scores else 0.0

        # Dominant trend
        if declining > improving and declining > (n - declining - improving):
            cluster.dominant_trend = "declining"
        elif improving > declining and improving > (n - declining - improving):
            cluster.dominant_trend = "improving"
        else:
            cluster.dominant_trend = "stable"

        # Risk tier
        if high_risk_count >= n / 2:
            cluster.cluster_risk_tier = "high"
            cluster.is_risk_cluster = True
        elif high_risk_count > 0:
            cluster.cluster_risk_tier = "medium"
        else:
            cluster.cluster_risk_tier = "low"

        # Recommendations
        if cluster.is_risk_cluster:
            cluster.recommended_actions.append(
                f"Investigate shared factors in {cluster.repo_count} correlated high-risk repos"
            )

        if cluster.percent_declining > 0.5:
            cluster.recommended_actions.append(
                "Review common dependencies or configurations causing decline"
            )
            cluster.potential_shared_factors.append("Common declining pattern")

        return cluster

    def get_clusters(self) -> List[CorrelationCluster]:
        """Get detected clusters."""
        return self._clusters


# ============================================================================
# Anomaly Detector
# ============================================================================

class AnomalyDetector:
    """
    Detects cross-repository anomalies using rule-based detection.

    Identifies patterns such as:
    - Synchronized declines
    - Emerging risk clusters
    - Leading indicators
    - Correlated issue spikes
    """

    def __init__(self, config: TrendCorrelationConfig):
        """Initialize anomaly detector."""
        self.config = config
        self.thresholds = config.thresholds
        self._anomalies: List[CrossRepoAnomaly] = []
        self._anomaly_counter = 0

    def detect_anomalies(
        self,
        repo_series: Dict[str, RepoTrendSeries],
        correlations: List[TrendCorrelation],
        clusters: List[CorrelationCluster]
    ) -> List[CrossRepoAnomaly]:
        """
        Detect cross-repository anomalies.

        Args:
            repo_series: Dictionary of repo_id -> RepoTrendSeries
            correlations: List of significant correlations
            clusters: Detected clusters

        Returns:
            List of detected CrossRepoAnomaly objects
        """
        self._anomalies = []
        timestamp = datetime.utcnow().isoformat()

        # Detect synchronized decline
        self._detect_synchronized_decline(repo_series, timestamp)

        # Detect emerging risk clusters
        self._detect_emerging_risk_clusters(clusters, repo_series, timestamp)

        # Detect leading indicators
        if self.config.compute_leading_indicators:
            self._detect_leading_indicators(repo_series, correlations, timestamp)

        # Detect shared volatility
        self._detect_shared_volatility(repo_series, correlations, timestamp)

        # Detect correlated issue spikes
        self._detect_correlated_issue_spikes(repo_series, correlations, timestamp)

        # Sort by severity
        self._anomalies.sort(key=lambda a: (
            {"critical": 0, "high": 1, "medium": 2, "low": 3}.get(a.severity.value, 4)
        ))

        logger.info(f"Detected {len(self._anomalies)} anomalies")

        return self._anomalies

    def _generate_anomaly_id(self) -> str:
        """Generate unique anomaly ID."""
        self._anomaly_counter += 1
        return f"anomaly_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{self._anomaly_counter:04d}"

    def _detect_synchronized_decline(
        self,
        repo_series: Dict[str, RepoTrendSeries],
        timestamp: str
    ) -> None:
        """Detect synchronized decline across multiple repos."""
        declining_repos = [
            repo_id for repo_id, series in repo_series.items()
            if series.current_trend_direction.lower() == "declining"
        ]

        total_repos = len(repo_series)
        decline_ratio = len(declining_repos) / total_repos if total_repos > 0 else 0.0

        if (decline_ratio >= self.thresholds.synchronized_decline_threshold and
            len(declining_repos) >= self.thresholds.synchronized_decline_min_repos):

            # Determine severity
            if decline_ratio >= 0.40:
                severity = AnomalySeverity.CRITICAL
            elif decline_ratio >= 0.30:
                severity = AnomalySeverity.HIGH
            else:
                severity = AnomalySeverity.MEDIUM

            anomaly = CrossRepoAnomaly(
                anomaly_id=self._generate_anomaly_id(),
                anomaly_type=AnomalyType.SYNCHRONIZED_DECLINE,
                severity=severity,
                title="Synchronized Decline Across Repositories",
                message=f"{len(declining_repos)} repos ({decline_ratio*100:.1f}%) "
                       f"exhibit simultaneous declining trends",
                timestamp=timestamp,
                affected_repos=declining_repos,
                affected_count=len(declining_repos),
                trigger_metric="percent_declining",
                trigger_value=decline_ratio,
                trigger_threshold=self.thresholds.synchronized_decline_threshold,
                evidence=[
                    f"{len(declining_repos)} of {total_repos} repos are declining",
                    f"Decline ratio: {decline_ratio*100:.1f}%"
                ],
                recommended_actions=[
                    "Investigate common factors across declining repos",
                    "Review recent org-wide changes or deployments",
                    "Check for shared dependencies or infrastructure issues",
                    "Consider rollback if decline correlates with recent release"
                ]
            )
            self._anomalies.append(anomaly)

    def _detect_emerging_risk_clusters(
        self,
        clusters: List[CorrelationCluster],
        repo_series: Dict[str, RepoTrendSeries],
        timestamp: str
    ) -> None:
        """Detect clusters forming new risk patterns."""
        for cluster in clusters:
            if not cluster.is_risk_cluster:
                continue

            if cluster.percent_declining > 0.5:
                severity = AnomalySeverity.HIGH
                if cluster.repo_count >= 5:
                    severity = AnomalySeverity.CRITICAL

                anomaly = CrossRepoAnomaly(
                    anomaly_id=self._generate_anomaly_id(),
                    anomaly_type=AnomalyType.EMERGING_RISK_CLUSTER,
                    severity=severity,
                    title=f"Emerging Risk Cluster: {cluster.cluster_name}",
                    message=f"Cluster of {cluster.repo_count} correlated repos "
                           f"showing high risk ({cluster.percent_declining*100:.0f}% declining)",
                    timestamp=timestamp,
                    affected_repos=cluster.repo_ids,
                    affected_count=cluster.repo_count,
                    trigger_metric="cluster_risk",
                    trigger_value=cluster.avg_risk_score,
                    details={
                        "cluster_id": cluster.cluster_id,
                        "avg_correlation": cluster.avg_internal_correlation,
                        "dominant_trend": cluster.dominant_trend
                    },
                    evidence=[
                        f"Cluster size: {cluster.repo_count} repos",
                        f"Internal correlation: {cluster.avg_internal_correlation:.2f}",
                        f"Declining: {cluster.percent_declining*100:.0f}%"
                    ],
                    recommended_actions=[
                        f"Focus remediation efforts on {cluster.cluster_name}",
                        "Investigate shared dependencies within cluster",
                        "Consider cluster-wide improvement initiative"
                    ]
                )
                self._anomalies.append(anomaly)

    def _detect_leading_indicators(
        self,
        repo_series: Dict[str, RepoTrendSeries],
        correlations: List[TrendCorrelation],
        timestamp: str
    ) -> None:
        """
        Detect repos whose trend changes precede others.

        A leading indicator is when one repo's decline precedes
        another correlated repo's decline.
        """
        # Find repos that started declining before correlated repos
        for corr in correlations:
            if not corr.is_significant:
                continue

            series_a = repo_series.get(corr.repo_a_id)
            series_b = repo_series.get(corr.repo_b_id)

            if not series_a or not series_b:
                continue

            # Check if one repo's decline preceded the other
            if (series_a.current_trend_direction.lower() == "declining" and
                series_b.current_trend_direction.lower() == "declining"):

                # Check decline timing from data points
                a_decline_start = self._find_decline_start(series_a)
                b_decline_start = self._find_decline_start(series_b)

                if a_decline_start is not None and b_decline_start is not None:
                    if a_decline_start < b_decline_start:
                        # Repo A's decline preceded Repo B's
                        self._create_leading_indicator_anomaly(
                            leader=series_a,
                            follower=series_b,
                            lag=b_decline_start - a_decline_start,
                            timestamp=timestamp
                        )

    def _find_decline_start(self, series: RepoTrendSeries) -> Optional[int]:
        """Find the index where declining trend started."""
        for i, dp in enumerate(series.data_points):
            if dp.trend_direction.lower() == "declining":
                return i
        return None

    def _create_leading_indicator_anomaly(
        self,
        leader: RepoTrendSeries,
        follower: RepoTrendSeries,
        lag: int,
        timestamp: str
    ) -> None:
        """Create anomaly for leading indicator pattern."""
        anomaly = CrossRepoAnomaly(
            anomaly_id=self._generate_anomaly_id(),
            anomaly_type=AnomalyType.LEADING_INDICATOR,
            severity=AnomalySeverity.MEDIUM,
            title=f"Leading Indicator: {leader.repo_id} → {follower.repo_id}",
            message=f"Decline in {leader.repo_id} preceded decline in "
                   f"{follower.repo_id} by {lag} interval(s)",
            timestamp=timestamp,
            affected_repos=[leader.repo_id, follower.repo_id],
            affected_count=2,
            trigger_metric="leading_indicator_lag",
            trigger_value=float(lag),
            is_predictive=True,
            predicted_impact=f"Changes in {leader.repo_id} may predict "
                           f"future changes in {follower.repo_id}",
            confidence=0.7,
            evidence=[
                f"{leader.repo_id} decline started first",
                f"Lag: {lag} interval(s)",
                f"Both repos now declining"
            ],
            recommended_actions=[
                f"Monitor {leader.repo_id} as early warning indicator",
                f"When {leader.repo_id} improves, expect {follower.repo_id} to follow",
                "Investigate causal relationship between repos"
            ]
        )
        self._anomalies.append(anomaly)

    def _detect_shared_volatility(
        self,
        repo_series: Dict[str, RepoTrendSeries],
        correlations: List[TrendCorrelation],
        timestamp: str
    ) -> None:
        """Detect repos with correlated high volatility."""
        high_volatility_repos = [
            repo_id for repo_id, series in repo_series.items()
            if series.score_volatility > 15.0  # High volatility threshold
        ]

        if len(high_volatility_repos) >= 3:
            # Check if they're correlated
            correlated_volatile = set()
            for corr in correlations:
                if (corr.repo_a_id in high_volatility_repos and
                    corr.repo_b_id in high_volatility_repos and
                    corr.volatility_similarity >= self.thresholds.volatility_similarity_threshold):
                    correlated_volatile.add(corr.repo_a_id)
                    correlated_volatile.add(corr.repo_b_id)

            if len(correlated_volatile) >= 3:
                anomaly = CrossRepoAnomaly(
                    anomaly_id=self._generate_anomaly_id(),
                    anomaly_type=AnomalyType.SHARED_VOLATILITY,
                    severity=AnomalySeverity.MEDIUM,
                    title="Correlated High Volatility Pattern",
                    message=f"{len(correlated_volatile)} repos exhibit "
                           f"correlated high volatility",
                    timestamp=timestamp,
                    affected_repos=list(correlated_volatile),
                    affected_count=len(correlated_volatile),
                    trigger_metric="volatility_similarity",
                    trigger_value=self.thresholds.volatility_similarity_threshold,
                    evidence=[
                        f"{len(correlated_volatile)} repos with high volatility",
                        "Volatility patterns are correlated"
                    ],
                    recommended_actions=[
                        "Investigate shared factors causing instability",
                        "Review release/deployment patterns",
                        "Consider stabilization initiative"
                    ]
                )
                self._anomalies.append(anomaly)

    def _detect_correlated_issue_spikes(
        self,
        repo_series: Dict[str, RepoTrendSeries],
        correlations: List[TrendCorrelation],
        timestamp: str
    ) -> None:
        """Detect correlated issue spikes across repos."""
        issue_spike_pairs = []

        for corr in correlations:
            if abs(corr.issue_correlation) >= self.thresholds.issue_spike_correlation_threshold:
                series_a = repo_series.get(corr.repo_a_id)
                series_b = repo_series.get(corr.repo_b_id)

                if series_a and series_b:
                    if series_a.max_issue_spike > 0 and series_b.max_issue_spike > 0:
                        issue_spike_pairs.append((corr.repo_a_id, corr.repo_b_id))

        if issue_spike_pairs:
            # Flatten unique repos
            affected = set()
            for a, b in issue_spike_pairs:
                affected.add(a)
                affected.add(b)

            if len(affected) >= 2:
                anomaly = CrossRepoAnomaly(
                    anomaly_id=self._generate_anomaly_id(),
                    anomaly_type=AnomalyType.CORRELATED_ISSUE_SPIKE,
                    severity=AnomalySeverity.HIGH,
                    title="Correlated Issue Spikes Detected",
                    message=f"{len(affected)} repos show correlated issue spikes",
                    timestamp=timestamp,
                    affected_repos=list(affected),
                    affected_count=len(affected),
                    trigger_metric="issue_correlation",
                    trigger_value=self.thresholds.issue_spike_correlation_threshold,
                    evidence=[
                        f"{len(issue_spike_pairs)} repo pairs with correlated issues",
                        "Issue counts moving together"
                    ],
                    recommended_actions=[
                        "Investigate common root cause for issues",
                        "Check for shared dependencies with bugs",
                        "Review recent deployments affecting multiple repos"
                    ]
                )
                self._anomalies.append(anomaly)

    def get_anomalies(self) -> List[CrossRepoAnomaly]:
        """Get detected anomalies."""
        return self._anomalies


# ============================================================================
# Main Engine
# ============================================================================

class TrendCorrelationEngine:
    """
    Main orchestrator for the Multi-Repository Trend Correlation Engine.

    This class:
    1. Loads trend data from org health report
    2. Builds correlation matrix between repositories
    3. Detects repository clusters
    4. Identifies cross-repo anomalies
    5. Generates predictive indicators
    6. Produces correlation report
    7. Returns appropriate exit code
    """

    def __init__(self, config: TrendCorrelationConfig):
        """Initialize the trend correlation engine."""
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
        self.loader = TrendLoader(config)
        self.matrix_builder = CorrelationMatrixBuilder(config)
        self.cluster_detector = ClusterDetector(config)
        self.anomaly_detector = AnomalyDetector(config)

        # Results storage
        self._repo_series: Dict[str, RepoTrendSeries] = {}
        self._correlation_matrix: Dict[str, Dict[str, float]] = {}
        self._correlations: List[TrendCorrelation] = []
        self._clusters: List[CorrelationCluster] = []
        self._anomalies: List[CrossRepoAnomaly] = []

    def run(self) -> Tuple[TrendCorrelationReport, int]:
        """
        Run the complete trend correlation pipeline.

        Returns:
            Tuple of (TrendCorrelationReport, exit_code)
        """
        start_time = datetime.utcnow()

        try:
            logger.info("=" * 80)
            logger.info("MULTI-REPOSITORY TREND CORRELATION ENGINE")
            logger.info("=" * 80)

            # Step 1: Load org report and extract trend series
            logger.info("\nStep 1: Loading org health report...")
            org_report = self.loader.load_org_report()
            logger.info(f"  Loaded report: {self.config.org_report_path}")

            # Step 2: Extract trend series
            logger.info("\nStep 2: Extracting trend series...")
            self._repo_series = self.loader.extract_trend_series()
            logger.info(f"  Extracted {len(self._repo_series)} repo series")

            if len(self._repo_series) < 2:
                logger.warning("Insufficient repos for correlation analysis")
                return self._minimal_report(org_report, start_time), EXIT_CORRELATION_SUCCESS

            # Step 3: Build correlation matrix
            logger.info("\nStep 3: Building correlation matrix...")
            self._correlation_matrix = self.matrix_builder.build_matrix(self._repo_series)
            self._correlations = self.matrix_builder.get_significant_correlations()
            logger.info(f"  Found {len(self._correlations)} significant correlations")

            # Step 4: Detect clusters
            if self.config.compute_clusters:
                logger.info("\nStep 4: Detecting clusters...")
                self._clusters = self.cluster_detector.detect_clusters(
                    self._repo_series,
                    self._correlation_matrix,
                    self._correlations
                )
                logger.info(f"  Detected {len(self._clusters)} clusters")

            # Step 5: Detect anomalies
            if self.config.detect_anomalies:
                logger.info("\nStep 5: Detecting anomalies...")
                self._anomalies = self.anomaly_detector.detect_anomalies(
                    self._repo_series,
                    self._correlations,
                    self._clusters
                )
                logger.info(f"  Detected {len(self._anomalies)} anomalies")

            # Step 6: Generate report
            logger.info("\nStep 6: Generating correlation report...")
            report = self._generate_report(org_report)

            # Calculate duration
            duration = (datetime.utcnow() - start_time).total_seconds() * 1000
            report.analysis_duration_ms = duration

            # Step 7: Write output
            if self.config.output_path:
                logger.info("\nStep 7: Writing report...")
                self._write_report(report)
                logger.info(f"  Wrote report to {self.config.output_path}")

            # Determine exit code
            exit_code = self._determine_exit_code(report)

            logger.info("\n" + "=" * 80)
            logger.info("TREND CORRELATION ANALYSIS COMPLETE")
            logger.info("=" * 80)
            logger.info(f"Repos Analyzed: {report.total_repos}")
            logger.info(f"Significant Correlations: {report.summary.significant_correlations}")
            logger.info(f"Clusters: {report.summary.total_clusters}")
            logger.info(f"Anomalies: {report.summary.total_anomalies}")
            logger.info(f"Exit Code: {exit_code}")
            logger.info("=" * 80)

            return report, exit_code

        except TrendCorrelationParseError as e:
            logger.error(f"Failed to parse org health report: {e}")
            return self._error_report(str(e)), EXIT_CORRELATION_PARSE_ERROR
        except TrendCorrelationConfigError as e:
            logger.error(f"Configuration error: {e}")
            return self._error_report(str(e)), EXIT_CORRELATION_CONFIG_ERROR
        except Exception as e:
            logger.error(f"Trend correlation error: {e}")
            import traceback
            traceback.print_exc()
            return self._error_report(str(e)), EXIT_GENERAL_CORRELATION_ERROR

    def _generate_report(self, org_report: Dict[str, Any]) -> TrendCorrelationReport:
        """Generate the correlation report."""
        timestamp = datetime.utcnow().isoformat()

        # Build summary
        summary = CorrelationSummary(
            total_repo_pairs=len(self._repo_series) * (len(self._repo_series) - 1) // 2,
            significant_correlations=len(self._correlations),
            positive_correlations=sum(
                1 for c in self._correlations
                if c.correlation_type in (CorrelationType.POSITIVE,
                                          CorrelationType.SYNCHRONIZED_DECLINE,
                                          CorrelationType.SYNCHRONIZED_IMPROVEMENT)
            ),
            negative_correlations=sum(
                1 for c in self._correlations
                if c.correlation_type == CorrelationType.NEGATIVE
            ),
            synchronized_decline_pairs=sum(
                1 for c in self._correlations
                if c.correlation_type == CorrelationType.SYNCHRONIZED_DECLINE
            ),
            synchronized_improvement_pairs=sum(
                1 for c in self._correlations
                if c.correlation_type == CorrelationType.SYNCHRONIZED_IMPROVEMENT
            ),
            shared_volatility_pairs=sum(
                1 for c in self._correlations
                if c.correlation_type == CorrelationType.SHARED_VOLATILITY
            ),
            total_clusters=len(self._clusters),
            largest_cluster_size=max((c.repo_count for c in self._clusters), default=0),
            risk_clusters=sum(1 for c in self._clusters if c.is_risk_cluster),
            total_anomalies=len(self._anomalies),
            critical_anomalies=sum(
                1 for a in self._anomalies if a.severity == AnomalySeverity.CRITICAL
            ),
            high_anomalies=sum(
                1 for a in self._anomalies if a.severity == AnomalySeverity.HIGH
            ),
            predictive_indicators=sum(1 for a in self._anomalies if a.is_predictive)
        )

        # Calculate correlation density
        if summary.total_repo_pairs > 0:
            summary.correlation_density = (
                summary.significant_correlations / summary.total_repo_pairs
            )

        # Average correlation
        if self._correlations:
            summary.avg_inter_repo_correlation = sum(
                abs(c.pearson_coefficient) for c in self._correlations
            ) / len(self._correlations)

        # Build recommendations
        recommendations = self._generate_recommendations()

        # Build predictive indicators
        predictive_indicators = [
            {
                "anomaly_id": a.anomaly_id,
                "type": a.anomaly_type.value,
                "affected_repos": a.affected_repos,
                "predicted_impact": a.predicted_impact,
                "confidence": a.confidence
            }
            for a in self._anomalies if a.is_predictive
        ]

        report = TrendCorrelationReport(
            report_id=f"trend_correlation_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            generated_at=timestamp,
            org_report_path=str(self.config.org_report_path),
            summary=summary,
            repo_series=[s.to_dict() for s in self._repo_series.values()],
            correlations=self._correlations,
            correlation_matrix=self._correlation_matrix,
            clusters=self._clusters,
            anomalies=self._anomalies,
            predictive_indicators=predictive_indicators,
            recommendations=recommendations,
            org_health_status=org_report.get("org_health_status", "unknown"),
            org_health_score=org_report.get("org_health_score", 0.0),
            total_repos=len(self._repo_series)
        )

        return report

    def _generate_recommendations(self) -> List[Dict[str, Any]]:
        """Generate recommendations based on analysis."""
        recommendations = []
        rec_id = 0

        # Critical anomaly recommendations
        critical_anomalies = [a for a in self._anomalies
                            if a.severity == AnomalySeverity.CRITICAL]
        if critical_anomalies:
            rec_id += 1
            recommendations.append({
                "id": f"rec_{rec_id:03d}",
                "priority": "critical",
                "title": "Address Critical Cross-Repo Anomalies",
                "message": f"{len(critical_anomalies)} critical anomaly(ies) detected "
                          f"requiring immediate attention",
                "actions": critical_anomalies[0].recommended_actions[:3],
                "affected_repos": critical_anomalies[0].affected_repos[:5]
            })

        # Risk cluster recommendations
        risk_clusters = [c for c in self._clusters if c.is_risk_cluster]
        if risk_clusters:
            rec_id += 1
            total_repos = sum(c.repo_count for c in risk_clusters)
            recommendations.append({
                "id": f"rec_{rec_id:03d}",
                "priority": "high",
                "title": "Investigate Risk Clusters",
                "message": f"{len(risk_clusters)} risk cluster(s) identified "
                          f"affecting {total_repos} repos",
                "actions": [
                    "Identify shared dependencies within clusters",
                    "Review common configurations or patterns",
                    "Prioritize cluster-wide improvements"
                ],
                "affected_repos": risk_clusters[0].repo_ids[:5] if risk_clusters else []
            })

        # Synchronized decline recommendations
        sync_decline_count = sum(
            1 for c in self._correlations
            if c.correlation_type == CorrelationType.SYNCHRONIZED_DECLINE
        )
        if sync_decline_count > 2:
            rec_id += 1
            recommendations.append({
                "id": f"rec_{rec_id:03d}",
                "priority": "high",
                "title": "Address Synchronized Decline Pattern",
                "message": f"{sync_decline_count} repo pairs showing "
                          f"synchronized decline patterns",
                "actions": [
                    "Investigate org-wide changes affecting multiple repos",
                    "Check shared infrastructure or dependencies",
                    "Review recent org-wide policy changes"
                ],
                "affected_repos": []
            })

        # Leading indicator recommendations
        leading_indicators = [a for a in self._anomalies if a.is_predictive]
        if leading_indicators:
            rec_id += 1
            recommendations.append({
                "id": f"rec_{rec_id:03d}",
                "priority": "medium",
                "title": "Monitor Leading Indicators",
                "message": f"{len(leading_indicators)} leading indicator(s) identified "
                          f"for predictive monitoring",
                "actions": [
                    "Set up monitoring for leading indicator repos",
                    "Use leading repos as early warning signals",
                    "Investigate causal relationships"
                ],
                "affected_repos": leading_indicators[0].affected_repos[:3] if leading_indicators else []
            })

        return recommendations

    def _write_report(self, report: TrendCorrelationReport) -> None:
        """Write report to JSON file."""
        try:
            output_path = Path(self.config.output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report.to_dict(), f, indent=2, default=str)

        except Exception as e:
            raise TrendCorrelationError(f"Failed to write report: {e}")

    def _determine_exit_code(self, report: TrendCorrelationReport) -> int:
        """Determine appropriate exit code."""
        # Check for critical anomalies
        if report.summary.critical_anomalies > 0:
            if self.config.fail_on_critical_anomaly:
                return EXIT_CRITICAL_ANOMALY

        # Check for any correlations
        if report.summary.significant_correlations > 0:
            if self.config.fail_on_any_correlations:
                return EXIT_CORRELATIONS_FOUND
            return EXIT_CORRELATIONS_FOUND

        return EXIT_CORRELATION_SUCCESS

    def _minimal_report(
        self,
        org_report: Dict[str, Any],
        start_time: datetime
    ) -> TrendCorrelationReport:
        """Generate minimal report when insufficient data."""
        duration = (datetime.utcnow() - start_time).total_seconds() * 1000

        return TrendCorrelationReport(
            report_id=f"trend_correlation_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            generated_at=datetime.utcnow().isoformat(),
            org_report_path=str(self.config.org_report_path),
            org_health_status=org_report.get("org_health_status", "unknown"),
            org_health_score=org_report.get("org_health_score", 0.0),
            total_repos=len(self._repo_series),
            repo_series=[s.to_dict() for s in self._repo_series.values()],
            analysis_duration_ms=duration
        )

    def _error_report(self, error_message: str) -> TrendCorrelationReport:
        """Generate error report."""
        return TrendCorrelationReport(
            report_id=f"trend_correlation_error_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
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

def create_default_thresholds() -> CorrelationThresholds:
    """Create default correlation thresholds."""
    return CorrelationThresholds()


def load_correlation_config(config_path: Path) -> CorrelationThresholds:
    """
    Load correlation thresholds from configuration file.

    Args:
        config_path: Path to YAML or JSON config file

    Returns:
        CorrelationThresholds object
    """
    if config_path.suffix in ('.yaml', '.yml'):
        try:
            import yaml
            with open(config_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
        except ImportError:
            raise TrendCorrelationConfigError(
                "PyYAML not installed. Install with: pip install pyyaml"
            )
    else:
        with open(config_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

    return CorrelationThresholds.from_dict(data.get("thresholds", {}))
