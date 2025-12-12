"""
Organization Health Governance Engine - Multi-Repository Health Aggregation & SLO Engine

This module implements an org-level health governance layer that:
1. Aggregates health data from multiple repositories
2. Evaluates SLO/SLA policies across the org
3. Computes org-level risk tiers and metrics
4. Produces comprehensive org health reports with recommendations

Architecture:
- OrgHealthConfig: Configuration for aggregation and SLO evaluation
- RepositoryHealthSnapshot: Per-repo health state (dashboard + alerts + trends)
- SloPolicy / SloEvaluationResult: SLO policy definitions and evaluation results
- OrgHealthAggregator: Core aggregation and evaluation engine
- OrgHealthReport: Comprehensive output with rollups, SLO status, recommendations

Exit Codes (90-99):
- 90: Success, no SLO violations
- 91: SLO violations detected
- 92: Org risk >= HIGH tier threshold
- 93: No repos discovered / loaded
- 94: Config error
- 95: Data aggregation error
- 99: General org-health error

Directory Structure Expected:
    org-health/
      repo-a/
        dashboard/health-dashboard.json
        alerts/alerts.json
        trends/trend-report.json
      repo-b/
        ...

Version: 1.0.0
Phase: 14.8 Task 1
"""

import json
import logging
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set, Callable, Union

logger = logging.getLogger(__name__)


# ============================================================================
# Exit Codes (90-99)
# ============================================================================

EXIT_ORG_SUCCESS = 90
EXIT_SLO_VIOLATIONS = 91
EXIT_HIGH_ORG_RISK = 92
EXIT_NO_REPOS_DISCOVERED = 93
EXIT_CONFIG_ERROR = 94
EXIT_AGGREGATION_ERROR = 95
EXIT_GENERAL_ORG_ERROR = 99


# ============================================================================
# Custom Exceptions
# ============================================================================

class OrgHealthError(Exception):
    """Base exception for org health errors."""
    exit_code = EXIT_GENERAL_ORG_ERROR


class ConfigError(OrgHealthError):
    """Configuration error."""
    exit_code = EXIT_CONFIG_ERROR


class NoReposDiscoveredError(OrgHealthError):
    """No repositories discovered or loaded."""
    exit_code = EXIT_NO_REPOS_DISCOVERED


class AggregationError(OrgHealthError):
    """Data aggregation failed."""
    exit_code = EXIT_AGGREGATION_ERROR


class SloViolationError(OrgHealthError):
    """SLO violations detected."""
    exit_code = EXIT_SLO_VIOLATIONS


# ============================================================================
# Enums
# ============================================================================

class HealthStatus(Enum):
    """Repository health status levels."""
    GREEN = "green"
    YELLOW = "yellow"
    RED = "red"
    UNKNOWN = "unknown"

    @classmethod
    def from_string(cls, value: str) -> "HealthStatus":
        """Convert string to HealthStatus."""
        value_lower = value.lower() if value else "unknown"
        try:
            return cls(value_lower)
        except ValueError:
            return cls.UNKNOWN


class RiskTier(Enum):
    """Organization risk tier levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    def __lt__(self, other: "RiskTier") -> bool:
        """Compare risk tiers."""
        order = {self.LOW: 0, self.MEDIUM: 1, self.HIGH: 2, self.CRITICAL: 3}
        return order[self] < order[other]

    def __le__(self, other: "RiskTier") -> bool:
        order = {self.LOW: 0, self.MEDIUM: 1, self.HIGH: 2, self.CRITICAL: 3}
        return order[self] <= order[other]


class TrendDirection(Enum):
    """Trend direction for repository health."""
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


class SloOperator(Enum):
    """Operators for SLO condition evaluation."""
    EQUALS = "=="
    NOT_EQUALS = "!="
    LESS_THAN = "<"
    LESS_THAN_OR_EQUALS = "<="
    GREATER_THAN = ">"
    GREATER_THAN_OR_EQUALS = ">="


# ============================================================================
# Data Classes - SLO Configuration
# ============================================================================

@dataclass
class RepoSelector:
    """
    Selector for filtering repositories in SLO policies.

    Supports:
    - tags: List of tags to match (repo must have any of these tags)
    - id_pattern: Regex pattern to match repo_id
    - all: If True, matches all repos
    """
    tags: List[str] = field(default_factory=list)
    id_pattern: Optional[str] = None
    all: bool = False

    def matches(self, repo_id: str, repo_tags: List[str]) -> bool:
        """
        Check if a repository matches this selector.

        Args:
            repo_id: Repository identifier
            repo_tags: Tags associated with the repository

        Returns:
            True if repo matches the selector criteria
        """
        # If 'all' is set, match everything
        if self.all:
            return True

        # Check id pattern
        if self.id_pattern:
            if re.match(self.id_pattern, repo_id):
                return True

        # Check tags
        if self.tags:
            if any(tag in repo_tags for tag in self.tags):
                return True

        # No criteria matched
        return False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RepoSelector":
        """Create from dictionary."""
        return cls(
            tags=data.get("tags", []),
            id_pattern=data.get("id_pattern"),
            all=data.get("all", False)
        )


@dataclass
class SloPolicy:
    """
    Service Level Objective policy definition.

    An SLO evaluates a specific metric across selected repositories
    against a target value using an operator.

    Supported metrics:
    - health_status: Computed as percent of repos with status == target
    - percent_green: Percentage of repos with GREEN status
    - percent_yellow_or_better: Percentage of repos with GREEN or YELLOW status
    - critical_issues: Max critical issues across matching repos
    - total_issues: Max total issues across matching repos
    - repository_score: Min/avg/max repository score
    - trend_improving: Percentage of repos with IMPROVING trend
    """
    id: str
    description: str
    repo_selector: RepoSelector
    metric: str
    target: float
    operator: SloOperator = SloOperator.GREATER_THAN_OR_EQUALS
    enabled: bool = True

    # Optional aggregation mode for multi-repo metrics
    # Supported: "any", "all", "avg", "min", "max", "percent"
    aggregation: str = "percent"

    # Severity if SLO is violated
    violation_severity: str = "medium"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "description": self.description,
            "repo_selector": self.repo_selector.to_dict(),
            "metric": self.metric,
            "target": self.target,
            "operator": self.operator.value,
            "enabled": self.enabled,
            "aggregation": self.aggregation,
            "violation_severity": self.violation_severity
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SloPolicy":
        """Create from dictionary."""
        # Parse operator
        op_str = data.get("operator", ">=")
        operator_map = {
            "==": SloOperator.EQUALS,
            "!=": SloOperator.NOT_EQUALS,
            "<": SloOperator.LESS_THAN,
            "<=": SloOperator.LESS_THAN_OR_EQUALS,
            ">": SloOperator.GREATER_THAN,
            ">=": SloOperator.GREATER_THAN_OR_EQUALS
        }
        operator = operator_map.get(op_str, SloOperator.GREATER_THAN_OR_EQUALS)

        return cls(
            id=data["id"],
            description=data.get("description", ""),
            repo_selector=RepoSelector.from_dict(data.get("repo_selector", {"all": True})),
            metric=data["metric"],
            target=float(data["target"]),
            operator=operator,
            enabled=data.get("enabled", True),
            aggregation=data.get("aggregation", "percent"),
            violation_severity=data.get("violation_severity", "medium")
        )


@dataclass
class SloEvaluationResult:
    """
    Result of evaluating a single SLO policy.

    Contains the computed metric value, whether the SLO is satisfied,
    and details about which repositories violated the policy.
    """
    slo_id: str
    slo_description: str
    satisfied: bool
    current_value: float
    target_value: float
    operator: str

    # Repositories that were evaluated
    repos_evaluated: int = 0

    # Repositories that violated the SLO (for per-repo constraints)
    violating_repos: List[str] = field(default_factory=list)

    # Additional context
    metric: str = ""
    details: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


# ============================================================================
# Data Classes - Repository Health Snapshot
# ============================================================================

@dataclass
class AlertSummary:
    """Summary of alerts for a repository."""
    total_alerts: int = 0
    critical_alerts: int = 0
    error_alerts: int = 0
    warning_alerts: int = 0
    info_alerts: int = 0

    @classmethod
    def from_alert_report(cls, data: Dict[str, Any]) -> "AlertSummary":
        """Create from alert report JSON."""
        return cls(
            total_alerts=data.get("total_alerts", 0),
            critical_alerts=data.get("critical_alerts", 0),
            error_alerts=data.get("error_alerts", 0),
            warning_alerts=data.get("warning_alerts", 0),
            info_alerts=data.get("info_alerts", 0)
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class TrendSummary:
    """Summary of trends for a repository."""
    overall_trend: TrendDirection = TrendDirection.UNKNOWN
    trend_confidence: float = 0.0
    predicted_next_score: float = 0.0
    score_volatility: float = 0.0
    total_anomalies: int = 0
    total_warnings: int = 0

    @classmethod
    def from_trend_report(cls, data: Dict[str, Any]) -> "TrendSummary":
        """Create from trend report JSON."""
        return cls(
            overall_trend=TrendDirection.from_string(data.get("overall_trend", "unknown")),
            trend_confidence=data.get("trend_confidence", 0.0),
            predicted_next_score=data.get("predicted_next_score", 0.0),
            score_volatility=data.get("score_volatility", 0.0),
            total_anomalies=data.get("total_anomalies", 0),
            total_warnings=data.get("total_warnings", 0)
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "overall_trend": self.overall_trend.value,
            "trend_confidence": self.trend_confidence,
            "predicted_next_score": self.predicted_next_score,
            "score_volatility": self.score_volatility,
            "total_anomalies": self.total_anomalies,
            "total_warnings": self.total_warnings
        }


@dataclass
class RepositoryHealthSnapshot:
    """
    Complete health snapshot for a single repository.

    Aggregates data from:
    - Health Dashboard (Task 8)
    - Alert Report (Task 9)
    - Trend Report (Task 10)
    """
    repo_id: str
    repo_name: str

    # Core health metrics from dashboard
    health_status: HealthStatus = HealthStatus.UNKNOWN
    repository_score: float = 0.0
    total_issues: int = 0
    critical_issues: int = 0
    missing_artifacts: int = 0
    corrupted_artifacts: int = 0

    # Version info
    latest_version: Optional[str] = None
    version_count: int = 0

    # Alert summary
    alerts: AlertSummary = field(default_factory=AlertSummary)

    # Trend summary
    trends: TrendSummary = field(default_factory=TrendSummary)

    # Tags for SLO filtering
    tags: List[str] = field(default_factory=list)

    # Computed metrics
    normalized_score: float = 0.0  # 0-100 normalized
    risk_tier: RiskTier = RiskTier.LOW
    risk_score: float = 0.0  # Composite risk score

    # Data availability flags
    has_dashboard: bool = False
    has_alerts: bool = False
    has_trends: bool = False

    # Timestamps
    dashboard_timestamp: Optional[str] = None
    alerts_timestamp: Optional[str] = None
    trends_timestamp: Optional[str] = None

    # SLO evaluation results for this repo
    slo_violations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "repo_id": self.repo_id,
            "repo_name": self.repo_name,
            "health_status": self.health_status.value,
            "repository_score": self.repository_score,
            "total_issues": self.total_issues,
            "critical_issues": self.critical_issues,
            "missing_artifacts": self.missing_artifacts,
            "corrupted_artifacts": self.corrupted_artifacts,
            "latest_version": self.latest_version,
            "version_count": self.version_count,
            "alerts": self.alerts.to_dict(),
            "trends": self.trends.to_dict(),
            "tags": self.tags,
            "normalized_score": self.normalized_score,
            "risk_tier": self.risk_tier.value,
            "risk_score": self.risk_score,
            "has_dashboard": self.has_dashboard,
            "has_alerts": self.has_alerts,
            "has_trends": self.has_trends,
            "dashboard_timestamp": self.dashboard_timestamp,
            "alerts_timestamp": self.alerts_timestamp,
            "trends_timestamp": self.trends_timestamp,
            "slo_violations": self.slo_violations
        }


# ============================================================================
# Data Classes - Configuration
# ============================================================================

@dataclass
class RiskTierThresholds:
    """
    Thresholds for computing risk tiers.

    Risk score is computed from multiple factors and then mapped to tiers.
    """
    # Score thresholds (below these values = higher risk)
    low_score_threshold: float = 80.0  # Below = MEDIUM
    medium_score_threshold: float = 60.0  # Below = HIGH
    high_score_threshold: float = 40.0  # Below = CRITICAL

    # Critical issues thresholds
    critical_issues_medium: int = 1  # >= triggers MEDIUM
    critical_issues_high: int = 3  # >= triggers HIGH
    critical_issues_critical: int = 5  # >= triggers CRITICAL

    # Alert thresholds
    critical_alerts_escalate: int = 1  # Any critical alert escalates tier

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RiskTierThresholds":
        """Create from dictionary."""
        return cls(
            low_score_threshold=data.get("low_score_threshold", 80.0),
            medium_score_threshold=data.get("medium_score_threshold", 60.0),
            high_score_threshold=data.get("high_score_threshold", 40.0),
            critical_issues_medium=data.get("critical_issues_medium", 1),
            critical_issues_high=data.get("critical_issues_high", 3),
            critical_issues_critical=data.get("critical_issues_critical", 5),
            critical_alerts_escalate=data.get("critical_alerts_escalate", 1)
        )


@dataclass
class OrgHealthConfig:
    """
    Configuration for the Organization Health Aggregator.

    Specifies:
    - Root directory containing per-repo artifacts
    - SLO policies to evaluate
    - Risk tier thresholds
    - Output configuration
    """
    # Input paths
    root_dir: Path

    # Default artifact file names (can be overridden per-repo)
    dashboard_filename: str = "health-dashboard.json"
    alerts_filename: str = "alerts.json"
    trends_filename: str = "trend-report.json"

    # Default subdirectory names within each repo
    dashboard_subdir: str = "dashboard"
    alerts_subdir: str = "alerts"
    trends_subdir: str = "trends"

    # Output configuration
    output_path: Optional[Path] = None

    # SLO policies
    slo_policies: List[SloPolicy] = field(default_factory=list)

    # Risk tier configuration
    risk_thresholds: RiskTierThresholds = field(default_factory=RiskTierThresholds)

    # Behavior flags
    fail_on_slo_violation: bool = True
    fail_on_critical_org_risk: bool = True
    verbose: bool = False

    # Filtering
    repo_filter: Optional[List[str]] = None  # If set, only load these repos

    # Repo metadata (tags per repo)
    repo_tags: Dict[str, List[str]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "root_dir": str(self.root_dir),
            "dashboard_filename": self.dashboard_filename,
            "alerts_filename": self.alerts_filename,
            "trends_filename": self.trends_filename,
            "dashboard_subdir": self.dashboard_subdir,
            "alerts_subdir": self.alerts_subdir,
            "trends_subdir": self.trends_subdir,
            "output_path": str(self.output_path) if self.output_path else None,
            "slo_policies": [p.to_dict() for p in self.slo_policies],
            "risk_thresholds": self.risk_thresholds.to_dict(),
            "fail_on_slo_violation": self.fail_on_slo_violation,
            "fail_on_critical_org_risk": self.fail_on_critical_org_risk,
            "verbose": self.verbose,
            "repo_filter": self.repo_filter,
            "repo_tags": self.repo_tags
        }

    @classmethod
    def from_yaml_file(cls, yaml_path: Path, root_dir: Path) -> "OrgHealthConfig":
        """
        Load configuration from YAML file.

        Args:
            yaml_path: Path to YAML configuration file
            root_dir: Root directory for org health artifacts

        Returns:
            OrgHealthConfig instance
        """
        try:
            import yaml
        except ImportError:
            # Fallback to JSON parsing if PyYAML not available
            raise ConfigError("PyYAML not installed. Install with: pip install pyyaml")

        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)

            return cls._from_config_dict(data, root_dir)

        except Exception as e:
            raise ConfigError(f"Failed to load config from {yaml_path}: {e}")

    @classmethod
    def from_json_file(cls, json_path: Path, root_dir: Path) -> "OrgHealthConfig":
        """
        Load configuration from JSON file.

        Args:
            json_path: Path to JSON configuration file
            root_dir: Root directory for org health artifacts

        Returns:
            OrgHealthConfig instance
        """
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            return cls._from_config_dict(data, root_dir)

        except Exception as e:
            raise ConfigError(f"Failed to load config from {json_path}: {e}")

    @classmethod
    def _from_config_dict(cls, data: Dict[str, Any], root_dir: Path) -> "OrgHealthConfig":
        """Create config from dictionary."""
        # Parse SLO policies
        slo_policies = []
        for policy_data in data.get("slo_policies", []):
            slo_policies.append(SloPolicy.from_dict(policy_data))

        # Parse risk thresholds
        risk_thresholds = RiskTierThresholds.from_dict(
            data.get("risk_thresholds", {})
        )

        return cls(
            root_dir=root_dir,
            dashboard_filename=data.get("dashboard_filename", "health-dashboard.json"),
            alerts_filename=data.get("alerts_filename", "alerts.json"),
            trends_filename=data.get("trends_filename", "trend-report.json"),
            dashboard_subdir=data.get("dashboard_subdir", "dashboard"),
            alerts_subdir=data.get("alerts_subdir", "alerts"),
            trends_subdir=data.get("trends_subdir", "trends"),
            output_path=Path(data["output_path"]) if data.get("output_path") else None,
            slo_policies=slo_policies,
            risk_thresholds=risk_thresholds,
            fail_on_slo_violation=data.get("fail_on_slo_violation", True),
            fail_on_critical_org_risk=data.get("fail_on_critical_org_risk", True),
            verbose=data.get("verbose", False),
            repo_filter=data.get("repo_filter"),
            repo_tags=data.get("repo_tags", {})
        )


# ============================================================================
# Data Classes - Org Health Report
# ============================================================================

@dataclass
class OrgMetrics:
    """Aggregated metrics across all repositories."""
    # Repository counts
    total_repos: int = 0
    repos_green: int = 0
    repos_yellow: int = 0
    repos_red: int = 0
    repos_unknown: int = 0

    # Risk tier distribution
    repos_low_risk: int = 0
    repos_medium_risk: int = 0
    repos_high_risk: int = 0
    repos_critical_risk: int = 0

    # Trend distribution
    repos_improving: int = 0
    repos_stable: int = 0
    repos_declining: int = 0
    repos_trend_unknown: int = 0

    # Score statistics
    avg_score: float = 0.0
    min_score: float = 0.0
    max_score: float = 100.0
    median_score: float = 0.0

    # Issue totals
    total_issues: int = 0
    total_critical_issues: int = 0
    max_issues_per_repo: int = 0
    max_critical_per_repo: int = 0

    # Alert totals
    total_alerts: int = 0
    total_critical_alerts: int = 0

    # Data availability
    repos_with_dashboard: int = 0
    repos_with_alerts: int = 0
    repos_with_trends: int = 0

    # Derived KPIs
    percent_green: float = 0.0
    percent_yellow_or_better: float = 0.0
    percent_improving: float = 0.0
    percent_meeting_min_score: float = 0.0  # >= 80

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class RepoRisk:
    """Risk summary for a single repository (for top risk repos list)."""
    repo_id: str
    repo_name: str
    risk_tier: str
    risk_score: float
    health_status: str
    repository_score: float
    trend_direction: str
    critical_issues: int
    critical_alerts: int
    reason_codes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class Recommendation:
    """Actionable recommendation based on org health analysis."""
    recommendation_id: str
    priority: str  # critical, high, medium, low
    title: str
    message: str
    affected_repos: List[str] = field(default_factory=list)
    suggested_actions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class OrgHealthReport:
    """
    Complete Organization Health Report.

    Aggregates health data from all repositories and provides:
    - Org-level health status and score
    - SLO compliance summary
    - Risk tier distribution
    - Top risk repositories
    - Actionable recommendations
    """
    # Report metadata
    report_id: str
    generated_at: str
    root_dir: str

    # Org-level status
    org_health_status: str = "unknown"
    org_health_score: float = 0.0
    org_risk_tier: str = "unknown"

    # Aggregated metrics
    metrics: OrgMetrics = field(default_factory=OrgMetrics)

    # SLO evaluation
    total_slos: int = 0
    slos_satisfied: int = 0
    slos_violated: int = 0
    slo_results: List[SloEvaluationResult] = field(default_factory=list)

    # Repository snapshots
    repositories: List[Dict[str, Any]] = field(default_factory=list)

    # Top risk repositories (sorted by risk)
    top_risk_repos: List[RepoRisk] = field(default_factory=list)

    # Policy violations summary
    policy_violations: List[Dict[str, Any]] = field(default_factory=list)

    # Recommendations
    recommendations: List[Recommendation] = field(default_factory=list)

    # Execution metadata
    repos_discovered: int = 0
    repos_loaded: int = 0
    repos_failed: int = 0
    analysis_duration_ms: float = 0.0

    # Load errors
    load_errors: List[Dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "report_id": self.report_id,
            "generated_at": self.generated_at,
            "root_dir": self.root_dir,
            "org_health_status": self.org_health_status,
            "org_health_score": self.org_health_score,
            "org_risk_tier": self.org_risk_tier,
            "metrics": self.metrics.to_dict(),
            "total_slos": self.total_slos,
            "slos_satisfied": self.slos_satisfied,
            "slos_violated": self.slos_violated,
            "slo_results": [r.to_dict() for r in self.slo_results],
            "repositories": self.repositories,
            "top_risk_repos": [r.to_dict() for r in self.top_risk_repos],
            "policy_violations": self.policy_violations,
            "recommendations": [r.to_dict() for r in self.recommendations],
            "repos_discovered": self.repos_discovered,
            "repos_loaded": self.repos_loaded,
            "repos_failed": self.repos_failed,
            "analysis_duration_ms": self.analysis_duration_ms,
            "load_errors": self.load_errors
        }


# ============================================================================
# Organization Health Aggregator
# ============================================================================

class OrgHealthAggregator:
    """
    Core aggregator for organization-level health governance.

    Responsibilities:
    1. Discover repositories under the root directory
    2. Load health data (dashboard, alerts, trends) for each repo
    3. Compute per-repo risk scores and tiers
    4. Evaluate SLO policies across repositories
    5. Compute org-level metrics and status
    6. Generate recommendations
    7. Produce comprehensive OrgHealthReport
    """

    def __init__(self, config: OrgHealthConfig):
        """
        Initialize the org health aggregator.

        Args:
            config: Configuration for aggregation and SLO evaluation
        """
        self.config = config

        if config.verbose:
            logger.setLevel(logging.DEBUG)

        # Repository snapshots
        self._repositories: Dict[str, RepositoryHealthSnapshot] = {}

        # Computed results
        self._slo_results: List[SloEvaluationResult] = []
        self._org_metrics: Optional[OrgMetrics] = None
        self._recommendations: List[Recommendation] = []

        # Tracking
        self._discovered_repos: List[str] = []
        self._load_errors: List[Dict[str, str]] = []

    # ========================================================================
    # Discovery & Loading
    # ========================================================================

    def discover_repositories(self) -> List[str]:
        """
        Discover repositories under the root directory.

        Scans the root directory for subdirectories that contain
        health artifacts (dashboard, alerts, trends).

        Returns:
            List of discovered repository IDs (directory names)

        Raises:
            NoReposDiscoveredError: If no repositories are found
        """
        root = self.config.root_dir

        if not root.exists():
            raise NoReposDiscoveredError(f"Root directory does not exist: {root}")

        if not root.is_dir():
            raise NoReposDiscoveredError(f"Root path is not a directory: {root}")

        discovered = []

        for item in root.iterdir():
            if not item.is_dir():
                continue

            # Skip hidden directories
            if item.name.startswith('.'):
                continue

            # Check if this looks like a repo directory (has any health artifacts)
            has_dashboard = self._check_artifact_exists(item, "dashboard")
            has_alerts = self._check_artifact_exists(item, "alerts")
            has_trends = self._check_artifact_exists(item, "trends")

            if has_dashboard or has_alerts or has_trends:
                repo_id = item.name
                discovered.append(repo_id)
                logger.debug(f"Discovered repo: {repo_id}")

        # Apply filter if configured
        if self.config.repo_filter:
            discovered = [r for r in discovered if r in self.config.repo_filter]

        self._discovered_repos = discovered

        if not discovered:
            raise NoReposDiscoveredError(
                f"No repositories discovered in {root}. "
                "Expected subdirectories with health artifacts."
            )

        logger.info(f"Discovered {len(discovered)} repository(ies)")
        return discovered

    def _check_artifact_exists(self, repo_dir: Path, artifact_type: str) -> bool:
        """
        Check if a specific artifact type exists for a repository.

        Args:
            repo_dir: Repository directory path
            artifact_type: "dashboard", "alerts", or "trends"

        Returns:
            True if artifact exists
        """
        if artifact_type == "dashboard":
            subdir = self.config.dashboard_subdir
            filename = self.config.dashboard_filename
        elif artifact_type == "alerts":
            subdir = self.config.alerts_subdir
            filename = self.config.alerts_filename
        elif artifact_type == "trends":
            subdir = self.config.trends_subdir
            filename = self.config.trends_filename
        else:
            return False

        artifact_path = repo_dir / subdir / filename
        return artifact_path.exists()

    def load_repository_health(self, repo_id: str) -> Optional[RepositoryHealthSnapshot]:
        """
        Load health data for a single repository.

        Loads and merges data from:
        - Dashboard (required for meaningful analysis)
        - Alerts (optional)
        - Trends (optional)

        Args:
            repo_id: Repository identifier (directory name)

        Returns:
            RepositoryHealthSnapshot or None if loading failed
        """
        repo_dir = self.config.root_dir / repo_id

        if not repo_dir.exists():
            self._load_errors.append({
                "repo_id": repo_id,
                "error": f"Repository directory not found: {repo_dir}"
            })
            return None

        # Initialize snapshot
        snapshot = RepositoryHealthSnapshot(
            repo_id=repo_id,
            repo_name=repo_id,  # Default to repo_id, can be overridden from dashboard
            tags=self.config.repo_tags.get(repo_id, [])
        )

        # Load dashboard
        dashboard_data = self._load_dashboard(repo_id, repo_dir)
        if dashboard_data:
            self._populate_from_dashboard(snapshot, dashboard_data)

        # Load alerts
        alerts_data = self._load_alerts(repo_id, repo_dir)
        if alerts_data:
            self._populate_from_alerts(snapshot, alerts_data)

        # Load trends
        trends_data = self._load_trends(repo_id, repo_dir)
        if trends_data:
            self._populate_from_trends(snapshot, trends_data)

        # Compute derived metrics
        self._compute_risk_score(snapshot)

        # Store in cache
        self._repositories[repo_id] = snapshot

        return snapshot

    def _load_dashboard(self, repo_id: str, repo_dir: Path) -> Optional[Dict[str, Any]]:
        """Load dashboard JSON for a repository."""
        dashboard_path = repo_dir / self.config.dashboard_subdir / self.config.dashboard_filename

        if not dashboard_path.exists():
            logger.debug(f"No dashboard for {repo_id}: {dashboard_path}")
            return None

        try:
            with open(dashboard_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.debug(f"Loaded dashboard for {repo_id}")
            return data
        except Exception as e:
            self._load_errors.append({
                "repo_id": repo_id,
                "error": f"Failed to load dashboard: {e}"
            })
            return None

    def _load_alerts(self, repo_id: str, repo_dir: Path) -> Optional[Dict[str, Any]]:
        """Load alerts JSON for a repository."""
        alerts_path = repo_dir / self.config.alerts_subdir / self.config.alerts_filename

        if not alerts_path.exists():
            logger.debug(f"No alerts for {repo_id}: {alerts_path}")
            return None

        try:
            with open(alerts_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.debug(f"Loaded alerts for {repo_id}")
            return data
        except Exception as e:
            self._load_errors.append({
                "repo_id": repo_id,
                "error": f"Failed to load alerts: {e}"
            })
            return None

    def _load_trends(self, repo_id: str, repo_dir: Path) -> Optional[Dict[str, Any]]:
        """Load trends JSON for a repository."""
        trends_path = repo_dir / self.config.trends_subdir / self.config.trends_filename

        if not trends_path.exists():
            logger.debug(f"No trends for {repo_id}: {trends_path}")
            return None

        try:
            with open(trends_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.debug(f"Loaded trends for {repo_id}")
            return data
        except Exception as e:
            self._load_errors.append({
                "repo_id": repo_id,
                "error": f"Failed to load trends: {e}"
            })
            return None

    def _populate_from_dashboard(
        self,
        snapshot: RepositoryHealthSnapshot,
        data: Dict[str, Any]
    ) -> None:
        """Populate snapshot fields from dashboard data."""
        snapshot.has_dashboard = True
        snapshot.dashboard_timestamp = data.get("scan_timestamp") or data.get("generated_at")

        # Core health metrics
        snapshot.health_status = HealthStatus.from_string(data.get("overall_health", "unknown"))
        snapshot.repository_score = float(data.get("repository_score", 0.0))
        snapshot.total_issues = int(data.get("total_issues", 0))
        snapshot.critical_issues = int(data.get("critical_issues", 0))
        snapshot.missing_artifacts = int(data.get("missing_artifacts", 0))
        snapshot.corrupted_artifacts = int(data.get("corrupted_artifacts", 0))

        # Version info
        snapshot.latest_version = data.get("latest_version")
        versions_health = data.get("versions_health", [])
        snapshot.version_count = len(versions_health)

        # Normalized score (already 0-100 in dashboard)
        snapshot.normalized_score = snapshot.repository_score

        # Override repo_name if available
        if data.get("repository_name"):
            snapshot.repo_name = data["repository_name"]

    def _populate_from_alerts(
        self,
        snapshot: RepositoryHealthSnapshot,
        data: Dict[str, Any]
    ) -> None:
        """Populate snapshot fields from alerts data."""
        snapshot.has_alerts = True
        snapshot.alerts_timestamp = data.get("generated_at")
        snapshot.alerts = AlertSummary.from_alert_report(data)

    def _populate_from_trends(
        self,
        snapshot: RepositoryHealthSnapshot,
        data: Dict[str, Any]
    ) -> None:
        """Populate snapshot fields from trends data."""
        snapshot.has_trends = True
        snapshot.trends_timestamp = data.get("generated_at")
        snapshot.trends = TrendSummary.from_trend_report(data)

    def _compute_risk_score(self, snapshot: RepositoryHealthSnapshot) -> None:
        """
        Compute risk score and tier for a repository.

        Risk is computed from:
        - Health score (inverse: lower score = higher risk)
        - Critical issues count
        - Critical alerts count
        - Trend direction (declining = higher risk)
        """
        thresholds = self.config.risk_thresholds

        # Base risk from score (0-100, inverted)
        score_risk = (100 - snapshot.repository_score) / 100 * 40  # Max 40 points

        # Risk from critical issues
        issue_risk = min(snapshot.critical_issues * 10, 30)  # Max 30 points

        # Risk from critical alerts
        alert_risk = min(snapshot.alerts.critical_alerts * 15, 20)  # Max 20 points

        # Risk from trend
        trend_risk = 0
        if snapshot.trends.overall_trend == TrendDirection.DECLINING:
            trend_risk = 10  # Max 10 points

        # Total risk score (0-100)
        snapshot.risk_score = min(100, score_risk + issue_risk + alert_risk + trend_risk)

        # Determine risk tier
        snapshot.risk_tier = self._determine_risk_tier(snapshot, thresholds)

    def _determine_risk_tier(
        self,
        snapshot: RepositoryHealthSnapshot,
        thresholds: RiskTierThresholds
    ) -> RiskTier:
        """Determine risk tier based on multiple factors."""
        # Start with LOW risk
        tier = RiskTier.LOW

        # Check score thresholds
        if snapshot.repository_score < thresholds.high_score_threshold:
            tier = RiskTier.CRITICAL
        elif snapshot.repository_score < thresholds.medium_score_threshold:
            tier = max(tier, RiskTier.HIGH)
        elif snapshot.repository_score < thresholds.low_score_threshold:
            tier = max(tier, RiskTier.MEDIUM)

        # Check critical issues
        if snapshot.critical_issues >= thresholds.critical_issues_critical:
            tier = RiskTier.CRITICAL
        elif snapshot.critical_issues >= thresholds.critical_issues_high:
            tier = max(tier, RiskTier.HIGH)
        elif snapshot.critical_issues >= thresholds.critical_issues_medium:
            tier = max(tier, RiskTier.MEDIUM)

        # Check critical alerts
        if snapshot.alerts.critical_alerts >= thresholds.critical_alerts_escalate:
            if tier < RiskTier.HIGH:
                tier = max(tier, RiskTier.HIGH)

        return tier

    def load_all_repositories(self) -> int:
        """
        Load health data for all discovered repositories.

        Returns:
            Number of repositories successfully loaded

        Raises:
            NoReposDiscoveredError: If no repos have been discovered
        """
        if not self._discovered_repos:
            self.discover_repositories()

        loaded = 0
        for repo_id in self._discovered_repos:
            snapshot = self.load_repository_health(repo_id)
            if snapshot:
                loaded += 1

        logger.info(f"Loaded {loaded}/{len(self._discovered_repos)} repositories")
        return loaded

    # ========================================================================
    # SLO Evaluation
    # ========================================================================

    def evaluate_slos(self) -> List[SloEvaluationResult]:
        """
        Evaluate all configured SLO policies.

        Returns:
            List of SloEvaluationResult for each policy
        """
        results = []

        for policy in self.config.slo_policies:
            if not policy.enabled:
                continue

            result = self._evaluate_single_slo(policy)
            results.append(result)

            if not result.satisfied:
                logger.warning(f"SLO '{policy.id}' VIOLATED: {result.current_value} {policy.operator.value} {result.target_value}")
            else:
                logger.debug(f"SLO '{policy.id}' satisfied: {result.current_value} {policy.operator.value} {result.target_value}")

        self._slo_results = results
        return results

    def _evaluate_single_slo(self, policy: SloPolicy) -> SloEvaluationResult:
        """
        Evaluate a single SLO policy.

        Args:
            policy: SLO policy to evaluate

        Returns:
            SloEvaluationResult with evaluation outcome
        """
        # Find matching repositories
        matching_repos = self._filter_repos_by_selector(policy.repo_selector)

        if not matching_repos:
            return SloEvaluationResult(
                slo_id=policy.id,
                slo_description=policy.description,
                satisfied=True,  # No repos to evaluate = vacuously true
                current_value=0.0,
                target_value=policy.target,
                operator=policy.operator.value,
                repos_evaluated=0,
                metric=policy.metric,
                details="No repositories matched the selector"
            )

        # Compute metric value
        current_value, violating_repos = self._compute_slo_metric(
            policy, matching_repos
        )

        # Evaluate condition
        satisfied = self._check_slo_condition(
            current_value, policy.target, policy.operator
        )

        return SloEvaluationResult(
            slo_id=policy.id,
            slo_description=policy.description,
            satisfied=satisfied,
            current_value=current_value,
            target_value=policy.target,
            operator=policy.operator.value,
            repos_evaluated=len(matching_repos),
            violating_repos=violating_repos,
            metric=policy.metric,
            details=f"Evaluated {len(matching_repos)} repos"
        )

    def _filter_repos_by_selector(
        self,
        selector: RepoSelector
    ) -> List[RepositoryHealthSnapshot]:
        """Filter repositories matching a selector."""
        matching = []

        for repo_id, snapshot in self._repositories.items():
            if selector.matches(repo_id, snapshot.tags):
                matching.append(snapshot)

        return matching

    def _compute_slo_metric(
        self,
        policy: SloPolicy,
        repos: List[RepositoryHealthSnapshot]
    ) -> Tuple[float, List[str]]:
        """
        Compute the metric value for an SLO across repositories.

        Args:
            policy: SLO policy with metric definition
            repos: Repositories to compute metric over

        Returns:
            Tuple of (metric_value, list_of_violating_repo_ids)
        """
        metric = policy.metric.lower()
        violating = []

        if metric == "percent_green":
            green_count = sum(1 for r in repos if r.health_status == HealthStatus.GREEN)
            value = green_count / len(repos) if repos else 0.0
            # Track non-green repos as violating
            for r in repos:
                if r.health_status != HealthStatus.GREEN:
                    violating.append(r.repo_id)
            return value, violating

        elif metric == "percent_yellow_or_better":
            ok_count = sum(
                1 for r in repos
                if r.health_status in (HealthStatus.GREEN, HealthStatus.YELLOW)
            )
            value = ok_count / len(repos) if repos else 0.0
            for r in repos:
                if r.health_status not in (HealthStatus.GREEN, HealthStatus.YELLOW):
                    violating.append(r.repo_id)
            return value, violating

        elif metric == "critical_issues":
            # Max critical issues across repos (for <= target constraint)
            max_issues = 0
            for r in repos:
                if r.critical_issues > max_issues:
                    max_issues = r.critical_issues
                if r.critical_issues > policy.target:
                    violating.append(r.repo_id)
            return float(max_issues), violating

        elif metric == "total_issues":
            # Max total issues
            max_issues = max((r.total_issues for r in repos), default=0)
            for r in repos:
                if r.total_issues > policy.target:
                    violating.append(r.repo_id)
            return float(max_issues), violating

        elif metric == "repository_score":
            # Min score (for >= target constraint)
            scores = [r.repository_score for r in repos]
            if policy.aggregation == "min":
                value = min(scores) if scores else 0.0
            elif policy.aggregation == "max":
                value = max(scores) if scores else 0.0
            elif policy.aggregation == "avg":
                value = sum(scores) / len(scores) if scores else 0.0
            else:
                value = min(scores) if scores else 0.0

            for r in repos:
                if r.repository_score < policy.target:
                    violating.append(r.repo_id)
            return value, violating

        elif metric == "percent_improving":
            improving = sum(
                1 for r in repos
                if r.trends.overall_trend == TrendDirection.IMPROVING
            )
            value = improving / len(repos) if repos else 0.0
            for r in repos:
                if r.trends.overall_trend != TrendDirection.IMPROVING:
                    violating.append(r.repo_id)
            return value, violating

        else:
            logger.warning(f"Unknown SLO metric: {metric}")
            return 0.0, []

    def _check_slo_condition(
        self,
        current: float,
        target: float,
        operator: SloOperator
    ) -> bool:
        """Check if SLO condition is satisfied."""
        if operator == SloOperator.EQUALS:
            return current == target
        elif operator == SloOperator.NOT_EQUALS:
            return current != target
        elif operator == SloOperator.LESS_THAN:
            return current < target
        elif operator == SloOperator.LESS_THAN_OR_EQUALS:
            return current <= target
        elif operator == SloOperator.GREATER_THAN:
            return current > target
        elif operator == SloOperator.GREATER_THAN_OR_EQUALS:
            return current >= target
        else:
            return False

    # ========================================================================
    # Org-Level Metrics
    # ========================================================================

    def compute_org_metrics(self) -> OrgMetrics:
        """
        Compute aggregated metrics across all repositories.

        Returns:
            OrgMetrics with all aggregated statistics
        """
        repos = list(self._repositories.values())
        metrics = OrgMetrics()

        if not repos:
            return metrics

        metrics.total_repos = len(repos)

        # Health status distribution
        for r in repos:
            if r.health_status == HealthStatus.GREEN:
                metrics.repos_green += 1
            elif r.health_status == HealthStatus.YELLOW:
                metrics.repos_yellow += 1
            elif r.health_status == HealthStatus.RED:
                metrics.repos_red += 1
            else:
                metrics.repos_unknown += 1

        # Risk tier distribution
        for r in repos:
            if r.risk_tier == RiskTier.LOW:
                metrics.repos_low_risk += 1
            elif r.risk_tier == RiskTier.MEDIUM:
                metrics.repos_medium_risk += 1
            elif r.risk_tier == RiskTier.HIGH:
                metrics.repos_high_risk += 1
            elif r.risk_tier == RiskTier.CRITICAL:
                metrics.repos_critical_risk += 1

        # Trend distribution
        for r in repos:
            trend = r.trends.overall_trend
            if trend == TrendDirection.IMPROVING:
                metrics.repos_improving += 1
            elif trend == TrendDirection.STABLE:
                metrics.repos_stable += 1
            elif trend == TrendDirection.DECLINING:
                metrics.repos_declining += 1
            else:
                metrics.repos_trend_unknown += 1

        # Score statistics
        scores = [r.repository_score for r in repos]
        metrics.avg_score = sum(scores) / len(scores) if scores else 0.0
        metrics.min_score = min(scores) if scores else 0.0
        metrics.max_score = max(scores) if scores else 0.0
        metrics.median_score = self._compute_median(scores)

        # Issue totals
        metrics.total_issues = sum(r.total_issues for r in repos)
        metrics.total_critical_issues = sum(r.critical_issues for r in repos)
        metrics.max_issues_per_repo = max((r.total_issues for r in repos), default=0)
        metrics.max_critical_per_repo = max((r.critical_issues for r in repos), default=0)

        # Alert totals
        metrics.total_alerts = sum(r.alerts.total_alerts for r in repos)
        metrics.total_critical_alerts = sum(r.alerts.critical_alerts for r in repos)

        # Data availability
        metrics.repos_with_dashboard = sum(1 for r in repos if r.has_dashboard)
        metrics.repos_with_alerts = sum(1 for r in repos if r.has_alerts)
        metrics.repos_with_trends = sum(1 for r in repos if r.has_trends)

        # Derived KPIs
        metrics.percent_green = (
            metrics.repos_green / metrics.total_repos * 100
            if metrics.total_repos > 0 else 0.0
        )
        metrics.percent_yellow_or_better = (
            (metrics.repos_green + metrics.repos_yellow) / metrics.total_repos * 100
            if metrics.total_repos > 0 else 0.0
        )
        metrics.percent_improving = (
            metrics.repos_improving / metrics.total_repos * 100
            if metrics.total_repos > 0 else 0.0
        )
        metrics.percent_meeting_min_score = (
            sum(1 for r in repos if r.repository_score >= 80.0) / metrics.total_repos * 100
            if metrics.total_repos > 0 else 0.0
        )

        self._org_metrics = metrics
        return metrics

    def _compute_median(self, values: List[float]) -> float:
        """Compute median of a list of values."""
        if not values:
            return 0.0
        sorted_values = sorted(values)
        n = len(sorted_values)
        mid = n // 2
        if n % 2 == 0:
            return (sorted_values[mid - 1] + sorted_values[mid]) / 2
        return sorted_values[mid]

    # ========================================================================
    # Recommendations
    # ========================================================================

    def generate_recommendations(self) -> List[Recommendation]:
        """
        Generate actionable recommendations based on org health analysis.

        Returns:
            List of prioritized recommendations
        """
        recommendations = []
        rec_id = 0

        # Get metrics and repos
        metrics = self._org_metrics or self.compute_org_metrics()
        repos = list(self._repositories.values())

        # 1. Critical risk repos need immediate attention
        critical_repos = [r for r in repos if r.risk_tier == RiskTier.CRITICAL]
        if critical_repos:
            rec_id += 1
            recommendations.append(Recommendation(
                recommendation_id=f"rec_{rec_id:03d}",
                priority="critical",
                title="Address Critical Risk Repositories",
                message=f"{len(critical_repos)} repository(ies) are at CRITICAL risk level and require immediate attention.",
                affected_repos=[r.repo_id for r in critical_repos],
                suggested_actions=[
                    "Review critical issues in each repository",
                    "Investigate root cause of low health scores",
                    "Consider rollback if integrity is compromised",
                    "Escalate to team leads if necessary"
                ]
            ))

        # 2. High risk repos with declining trends
        declining_high_risk = [
            r for r in repos
            if r.risk_tier in (RiskTier.HIGH, RiskTier.CRITICAL)
            and r.trends.overall_trend == TrendDirection.DECLINING
        ]
        if declining_high_risk:
            rec_id += 1
            recommendations.append(Recommendation(
                recommendation_id=f"rec_{rec_id:03d}",
                priority="high",
                title="Stop Declining Trends in High-Risk Repos",
                message=f"{len(declining_high_risk)} high-risk repository(ies) have declining health trends.",
                affected_repos=[r.repo_id for r in declining_high_risk],
                suggested_actions=[
                    "Prioritize health improvement for these repos",
                    "Review recent changes causing degradation",
                    "Consider pausing new releases until stabilized"
                ]
            ))

        # 3. Repos with critical alerts
        repos_with_critical_alerts = [
            r for r in repos if r.alerts.critical_alerts > 0
        ]
        if repos_with_critical_alerts:
            rec_id += 1
            recommendations.append(Recommendation(
                recommendation_id=f"rec_{rec_id:03d}",
                priority="high",
                title="Resolve Critical Alerts",
                message=f"{len(repos_with_critical_alerts)} repository(ies) have active critical alerts.",
                affected_repos=[r.repo_id for r in repos_with_critical_alerts],
                suggested_actions=[
                    "Review alert details in each repository",
                    "Address underlying issues causing alerts",
                    "Update monitoring if alerts are false positives"
                ]
            ))

        # 4. SLO violations
        violated_slos = [r for r in self._slo_results if not r.satisfied]
        if violated_slos:
            rec_id += 1
            all_violating_repos = set()
            for slo in violated_slos:
                all_violating_repos.update(slo.violating_repos)
            recommendations.append(Recommendation(
                recommendation_id=f"rec_{rec_id:03d}",
                priority="high",
                title="Address SLO Violations",
                message=f"{len(violated_slos)} SLO(s) are currently violated.",
                affected_repos=list(all_violating_repos),
                suggested_actions=[
                    f"Review SLO '{slo.slo_id}': {slo.slo_description}" for slo in violated_slos[:3]
                ] + [
                    "Plan improvements to meet SLO targets"
                ]
            ))

        # 5. High volatility repos
        volatile_repos = [
            r for r in repos
            if r.trends.score_volatility > 10.0  # Configurable threshold
        ]
        if volatile_repos:
            rec_id += 1
            recommendations.append(Recommendation(
                recommendation_id=f"rec_{rec_id:03d}",
                priority="medium",
                title="Investigate High Volatility Repositories",
                message=f"{len(volatile_repos)} repository(ies) show high score volatility.",
                affected_repos=[r.repo_id for r in volatile_repos],
                suggested_actions=[
                    "Review release patterns and frequency",
                    "Investigate sources of instability",
                    "Consider more frequent health monitoring"
                ]
            ))

        # 6. Repos missing data
        repos_missing_trends = [r for r in repos if not r.has_trends]
        if repos_missing_trends:
            rec_id += 1
            recommendations.append(Recommendation(
                recommendation_id=f"rec_{rec_id:03d}",
                priority="low",
                title="Enable Trend Analysis",
                message=f"{len(repos_missing_trends)} repository(ies) lack trend data.",
                affected_repos=[r.repo_id for r in repos_missing_trends],
                suggested_actions=[
                    "Set up regular dashboard snapshot collection",
                    "Enable trend analysis in CI/CD pipeline"
                ]
            ))

        self._recommendations = recommendations
        return recommendations

    # ========================================================================
    # Report Generation
    # ========================================================================

    def generate_org_health_report(self) -> OrgHealthReport:
        """
        Generate complete organization health report.

        Returns:
            OrgHealthReport with all aggregated data and analysis
        """
        start_time = datetime.utcnow()

        # Ensure all data is computed
        if not self._repositories:
            raise AggregationError("No repositories loaded. Call load_all_repositories() first.")

        if self._org_metrics is None:
            self.compute_org_metrics()

        if not self._slo_results:
            self.evaluate_slos()

        if not self._recommendations:
            self.generate_recommendations()

        # Compute org-level status
        org_status, org_score = self._compute_org_status()
        org_risk = self._compute_org_risk_tier()

        # Build top risk repos list
        top_risk = self._build_top_risk_list(limit=10)

        # Build policy violations list
        policy_violations = [
            {
                "slo_id": r.slo_id,
                "description": r.slo_description,
                "current_value": r.current_value,
                "target_value": r.target_value,
                "violating_repos": r.violating_repos
            }
            for r in self._slo_results if not r.satisfied
        ]

        # Calculate duration
        duration = (datetime.utcnow() - start_time).total_seconds() * 1000

        report = OrgHealthReport(
            report_id=f"org_health_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            generated_at=datetime.utcnow().isoformat(),
            root_dir=str(self.config.root_dir),
            org_health_status=org_status,
            org_health_score=org_score,
            org_risk_tier=org_risk.value,
            metrics=self._org_metrics,
            total_slos=len(self._slo_results),
            slos_satisfied=sum(1 for r in self._slo_results if r.satisfied),
            slos_violated=sum(1 for r in self._slo_results if not r.satisfied),
            slo_results=self._slo_results,
            repositories=[r.to_dict() for r in self._repositories.values()],
            top_risk_repos=top_risk,
            policy_violations=policy_violations,
            recommendations=self._recommendations,
            repos_discovered=len(self._discovered_repos),
            repos_loaded=len(self._repositories),
            repos_failed=len(self._load_errors),
            analysis_duration_ms=duration,
            load_errors=self._load_errors
        )

        return report

    def _compute_org_status(self) -> Tuple[str, float]:
        """
        Compute org-level health status and score.

        Returns:
            Tuple of (status_string, score_float)
        """
        metrics = self._org_metrics

        # Org score is weighted average
        org_score = metrics.avg_score

        # Status based on distribution
        if metrics.repos_red > 0 or metrics.repos_critical_risk > 0:
            status = "red"
        elif metrics.repos_yellow > metrics.repos_green:
            status = "yellow"
        elif org_score < 60:
            status = "yellow"
        elif org_score < 40:
            status = "red"
        else:
            status = "green"

        return status, org_score

    def _compute_org_risk_tier(self) -> RiskTier:
        """Compute org-level risk tier."""
        metrics = self._org_metrics

        if metrics.repos_critical_risk > 0:
            return RiskTier.CRITICAL
        if metrics.repos_high_risk >= 2 or metrics.total_critical_issues >= 5:
            return RiskTier.HIGH
        if metrics.repos_medium_risk >= 3:
            return RiskTier.MEDIUM
        return RiskTier.LOW

    def _build_top_risk_list(self, limit: int = 10) -> List[RepoRisk]:
        """Build sorted list of top risk repositories."""
        repos = list(self._repositories.values())

        # Sort by risk score descending
        repos.sort(key=lambda r: r.risk_score, reverse=True)

        top_risk = []
        for r in repos[:limit]:
            # Build reason codes
            reasons = []
            if r.health_status == HealthStatus.RED:
                reasons.append("health_red")
            if r.critical_issues > 0:
                reasons.append(f"critical_issues:{r.critical_issues}")
            if r.alerts.critical_alerts > 0:
                reasons.append(f"critical_alerts:{r.alerts.critical_alerts}")
            if r.trends.overall_trend == TrendDirection.DECLINING:
                reasons.append("declining_trend")
            if r.repository_score < 60:
                reasons.append(f"low_score:{r.repository_score:.1f}")

            top_risk.append(RepoRisk(
                repo_id=r.repo_id,
                repo_name=r.repo_name,
                risk_tier=r.risk_tier.value,
                risk_score=r.risk_score,
                health_status=r.health_status.value,
                repository_score=r.repository_score,
                trend_direction=r.trends.overall_trend.value,
                critical_issues=r.critical_issues,
                critical_alerts=r.alerts.critical_alerts,
                reason_codes=reasons
            ))

        return top_risk


# ============================================================================
# Main Orchestrator - OrgHealthEngine
# ============================================================================

class OrgHealthEngine:
    """
    Main orchestrator for org-level health governance.

    Runs the complete aggregation pipeline:
    1. Discover repositories
    2. Load health data
    3. Evaluate SLOs
    4. Compute metrics
    5. Generate recommendations
    6. Produce report
    7. Return exit code
    """

    def __init__(self, config: OrgHealthConfig):
        """Initialize the org health engine."""
        self.config = config

        if config.verbose:
            logging.basicConfig(
                level=logging.DEBUG,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        else:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s'
            )

        self.aggregator = OrgHealthAggregator(config)

    def run(self) -> Tuple[OrgHealthReport, int]:
        """
        Run the complete org health analysis pipeline.

        Returns:
            Tuple of (OrgHealthReport, exit_code)
        """
        start_time = datetime.utcnow()

        try:
            logger.info("=" * 80)
            logger.info("ORGANIZATION HEALTH GOVERNANCE ENGINE")
            logger.info("=" * 80)

            # Step 1: Discover repositories
            logger.info("\nStep 1: Discovering repositories...")
            repos = self.aggregator.discover_repositories()
            logger.info(f"  Discovered {len(repos)} repository(ies)")

            # Step 2: Load health data
            logger.info("\nStep 2: Loading repository health data...")
            loaded = self.aggregator.load_all_repositories()
            logger.info(f"  Loaded {loaded}/{len(repos)} repositories")

            if loaded == 0:
                raise NoReposDiscoveredError("Failed to load any repository health data")

            # Step 3: Compute metrics
            logger.info("\nStep 3: Computing org-level metrics...")
            metrics = self.aggregator.compute_org_metrics()
            logger.info(f"  Average Score: {metrics.avg_score:.1f}")
            logger.info(f"  GREEN: {metrics.repos_green} | YELLOW: {metrics.repos_yellow} | RED: {metrics.repos_red}")

            # Step 4: Evaluate SLOs
            logger.info("\nStep 4: Evaluating SLO policies...")
            slo_results = self.aggregator.evaluate_slos()
            satisfied = sum(1 for r in slo_results if r.satisfied)
            violated = len(slo_results) - satisfied
            logger.info(f"  SLOs: {len(slo_results)} total | {satisfied} OK | {violated} VIOLATED")

            # Step 5: Generate recommendations
            logger.info("\nStep 5: Generating recommendations...")
            recommendations = self.aggregator.generate_recommendations()
            logger.info(f"  Generated {len(recommendations)} recommendation(s)")

            # Step 6: Generate report
            logger.info("\nStep 6: Generating org health report...")
            report = self.aggregator.generate_org_health_report()

            # Update duration
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
            logger.info("ORG HEALTH ANALYSIS COMPLETE")
            logger.info("=" * 80)
            logger.info(f"Org Health Status: {report.org_health_status.upper()}")
            logger.info(f"Org Health Score: {report.org_health_score:.1f}")
            logger.info(f"Org Risk Tier: {report.org_risk_tier.upper()}")
            logger.info(f"SLOs: {report.slos_satisfied} satisfied, {report.slos_violated} violated")
            logger.info(f"Exit Code: {exit_code}")
            logger.info("=" * 80)

            return report, exit_code

        except NoReposDiscoveredError as e:
            logger.error(f"No repositories discovered: {e}")
            return self._error_report(str(e)), EXIT_NO_REPOS_DISCOVERED
        except ConfigError as e:
            logger.error(f"Configuration error: {e}")
            return self._error_report(str(e)), EXIT_CONFIG_ERROR
        except AggregationError as e:
            logger.error(f"Aggregation error: {e}")
            return self._error_report(str(e)), EXIT_AGGREGATION_ERROR
        except Exception as e:
            logger.error(f"Org health error: {e}")
            return self._error_report(str(e)), EXIT_GENERAL_ORG_ERROR

    def _write_report(self, report: OrgHealthReport) -> None:
        """Write org health report to JSON file."""
        try:
            output_path = Path(self.config.output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report.to_dict(), f, indent=2, default=str)

        except Exception as e:
            raise AggregationError(f"Failed to write report: {e}")

    def _determine_exit_code(self, report: OrgHealthReport) -> int:
        """Determine appropriate exit code based on report and config."""
        # Check for critical org risk
        if (
            report.org_risk_tier == "critical"
            and self.config.fail_on_critical_org_risk
        ):
            return EXIT_HIGH_ORG_RISK

        # Check for SLO violations
        if report.slos_violated > 0 and self.config.fail_on_slo_violation:
            return EXIT_SLO_VIOLATIONS

        return EXIT_ORG_SUCCESS

    def _error_report(self, error_message: str) -> OrgHealthReport:
        """Generate an error report."""
        return OrgHealthReport(
            report_id=f"org_health_error_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            generated_at=datetime.utcnow().isoformat(),
            root_dir=str(self.config.root_dir),
            load_errors=[{"error": error_message}]
        )


# ============================================================================
# Utility Functions
# ============================================================================

def load_slo_config(config_path: Path) -> List[SloPolicy]:
    """
    Load SLO policies from a configuration file.

    Args:
        config_path: Path to YAML or JSON config file

    Returns:
        List of SloPolicy objects
    """
    if config_path.suffix in ('.yaml', '.yml'):
        try:
            import yaml
            with open(config_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
        except ImportError:
            raise ConfigError("PyYAML not installed. Install with: pip install pyyaml")
    else:
        with open(config_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

    policies = []
    for policy_data in data.get("slo_policies", []):
        policies.append(SloPolicy.from_dict(policy_data))

    return policies


def create_default_slo_policies() -> List[SloPolicy]:
    """
    Create a default set of SLO policies.

    Returns:
        List of default SloPolicy objects
    """
    return [
        SloPolicy(
            id="org-percent-green",
            description="At least 80% of repositories should be GREEN",
            repo_selector=RepoSelector(all=True),
            metric="percent_green",
            target=0.80,
            operator=SloOperator.GREATER_THAN_OR_EQUALS,
            violation_severity="medium"
        ),
        SloPolicy(
            id="core-repos-green",
            description="All core repositories must be GREEN or YELLOW",
            repo_selector=RepoSelector(tags=["core"]),
            metric="percent_yellow_or_better",
            target=1.0,
            operator=SloOperator.EQUALS,
            violation_severity="high"
        ),
        SloPolicy(
            id="max-critical-issues",
            description="No repository may have more than 5 critical issues",
            repo_selector=RepoSelector(all=True),
            metric="critical_issues",
            target=5,
            operator=SloOperator.LESS_THAN_OR_EQUALS,
            violation_severity="critical"
        ),
        SloPolicy(
            id="min-score-threshold",
            description="All repositories should have score >= 50",
            repo_selector=RepoSelector(all=True),
            metric="repository_score",
            target=50.0,
            operator=SloOperator.GREATER_THAN_OR_EQUALS,
            aggregation="min",
            violation_severity="high"
        )
    ]
