"""
Repository Health Trend Analyzer - Time-Series Engine

This module implements a comprehensive time-series analysis layer for repository health:
1. Stores and indexes historical dashboard snapshots
2. Computes score trends via linear regression
3. Calculates moving averages and volatility metrics
4. Detects anomalies using statistical methods (z-score)
5. Generates predictive health scores with confidence intervals
6. Produces early warning indicators for proactive monitoring

Architecture:
- HealthHistoryStore: Persists and retrieves dashboard snapshots
- TrendAnalyzer: Core time-series computation engine
- TrendReport: Comprehensive output dataclass
- TrendEngine: Main orchestrator

Exit Codes (80-89):
- 80: Trend analysis successful
- 81: Not enough history (minimum snapshots not met)
- 82: Invalid snapshot (corrupted or malformed)
- 83: Time series computation error
- 84: Prediction error
- 85: History store read/write error
- 86: Chart generation error
- 89: General trend analysis failure

Statistical Methods:
- Linear Regression: Ordinary Least Squares (OLS)
- Moving Averages: Simple Moving Average (SMA)
- Volatility: Standard Deviation
- Anomaly Detection: Z-score with configurable threshold
- Prediction: Linear extrapolation with confidence intervals

Version: 1.0.0
Phase: 14.7 Task 10
"""

import json
import logging
import math
import os
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union, Iterator

logger = logging.getLogger(__name__)


# ============================================================================
# Exit Codes (80-89)
# ============================================================================

EXIT_TREND_SUCCESS = 80
EXIT_INSUFFICIENT_HISTORY = 81
EXIT_INVALID_SNAPSHOT = 82
EXIT_COMPUTATION_ERROR = 83
EXIT_PREDICTION_ERROR = 84
EXIT_HISTORY_STORE_ERROR = 85
EXIT_CHART_GENERATION_ERROR = 86
EXIT_GENERAL_TREND_ERROR = 89


# ============================================================================
# Custom Exceptions
# ============================================================================

class TrendAnalysisError(Exception):
    """Base exception for trend analysis errors."""
    exit_code = EXIT_GENERAL_TREND_ERROR


class InsufficientHistoryError(TrendAnalysisError):
    """Not enough historical snapshots for analysis."""
    exit_code = EXIT_INSUFFICIENT_HISTORY


class InvalidSnapshotError(TrendAnalysisError):
    """Snapshot is corrupted or malformed."""
    exit_code = EXIT_INVALID_SNAPSHOT


class ComputationError(TrendAnalysisError):
    """Time series computation failed."""
    exit_code = EXIT_COMPUTATION_ERROR


class PredictionError(TrendAnalysisError):
    """Prediction model failed."""
    exit_code = EXIT_PREDICTION_ERROR


class HistoryStoreError(TrendAnalysisError):
    """History store read/write operation failed."""
    exit_code = EXIT_HISTORY_STORE_ERROR


class ChartGenerationError(TrendAnalysisError):
    """Chart generation failed."""
    exit_code = EXIT_CHART_GENERATION_ERROR


# ============================================================================
# Enums
# ============================================================================

class TrendDirection(Enum):
    """Direction of trend movement."""
    IMPROVING = "improving"
    STABLE = "stable"
    DECLINING = "declining"


class WarningLevel(Enum):
    """Severity level for early warnings."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AnomalyType(Enum):
    """Types of detected anomalies."""
    SUDDEN_SCORE_DROP = "sudden_score_drop"
    SUDDEN_SCORE_SPIKE = "sudden_score_spike"
    ISSUE_SPIKE = "issue_spike"
    UNUSUAL_VOLATILITY = "unusual_volatility"
    VERSION_DEGRADATION_CLUSTER = "version_degradation_cluster"


# ============================================================================
# Data Classes - Input/Configuration
# ============================================================================

@dataclass
class TrendConfig:
    """Configuration for trend analysis."""
    history_dir: Path
    output_path: Optional[Path] = None

    min_snapshots: int = 3
    max_snapshots: int = 100

    ma_windows: List[int] = field(default_factory=lambda: [3, 7, 14])

    zscore_threshold: float = 2.0
    score_drop_threshold: float = 15.0
    issue_spike_threshold: int = 5

    prediction_horizon: int = 3
    confidence_level: float = 0.95

    warning_score_threshold: float = 60.0
    critical_score_threshold: float = 40.0
    volatility_threshold: float = 10.0

    generate_charts: bool = False
    chart_output_dir: Optional[Path] = None

    verbose: bool = False


@dataclass
class SnapshotMetadata:
    """Metadata for a single dashboard snapshot."""
    snapshot_id: str
    timestamp: datetime
    version: Optional[str]
    file_path: Path
    file_size: int

    repository_score: float
    overall_health: str
    total_issues: int
    critical_issues: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "snapshot_id": self.snapshot_id,
            "timestamp": self.timestamp.isoformat(),
            "version": self.version,
            "file_path": str(self.file_path),
            "file_size": self.file_size,
            "repository_score": self.repository_score,
            "overall_health": self.overall_health,
            "total_issues": self.total_issues,
            "critical_issues": self.critical_issues
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SnapshotMetadata":
        """Create from dictionary."""
        return cls(
            snapshot_id=data["snapshot_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            version=data.get("version"),
            file_path=Path(data["file_path"]),
            file_size=data.get("file_size", 0),
            repository_score=data.get("repository_score", 0.0),
            overall_health=data.get("overall_health", "unknown"),
            total_issues=data.get("total_issues", 0),
            critical_issues=data.get("critical_issues", 0)
        )


@dataclass
class DashboardSnapshot:
    """Complete dashboard snapshot with all data."""
    metadata: SnapshotMetadata
    data: Dict[str, Any]
    versions_health: List[Dict[str, Any]] = field(default_factory=list)


# ============================================================================
# Data Classes - Analysis Results
# ============================================================================

@dataclass
class RegressionResult:
    """Result of linear regression analysis."""
    slope: float
    intercept: float
    r_squared: float
    standard_error: float
    p_value: float

    @property
    def is_significant(self) -> bool:
        """Check if trend is statistically significant (p < 0.05)."""
        return self.p_value < 0.05

    def predict(self, x: float) -> float:
        """Predict y value for given x."""
        return self.slope * x + self.intercept

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "slope": self.slope,
            "intercept": self.intercept,
            "r_squared": self.r_squared,
            "standard_error": self.standard_error,
            "p_value": self.p_value,
            "is_significant": self.is_significant
        }


@dataclass
class MovingAverageResult:
    """Result of moving average calculation."""
    window_size: int
    values: List[Optional[float]]
    current_value: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "window_size": self.window_size,
            "values": self.values,
            "current_value": self.current_value
        }


@dataclass
class VolatilityResult:
    """Result of volatility analysis."""
    standard_deviation: float
    variance: float
    coefficient_of_variation: float
    recent_volatility: float
    volatility_trend: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "standard_deviation": self.standard_deviation,
            "variance": self.variance,
            "coefficient_of_variation": self.coefficient_of_variation,
            "recent_volatility": self.recent_volatility,
            "volatility_trend": self.volatility_trend
        }


@dataclass
class Anomaly:
    """Detected anomaly in time series."""
    anomaly_id: str
    anomaly_type: str
    timestamp: datetime
    snapshot_id: str
    severity: str

    actual_value: float
    expected_value: float
    deviation: float
    zscore: float

    description: str
    affected_metric: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "anomaly_id": self.anomaly_id,
            "anomaly_type": self.anomaly_type,
            "timestamp": self.timestamp.isoformat(),
            "snapshot_id": self.snapshot_id,
            "severity": self.severity,
            "actual_value": self.actual_value,
            "expected_value": self.expected_value,
            "deviation": self.deviation,
            "zscore": self.zscore,
            "description": self.description,
            "affected_metric": self.affected_metric
        }


@dataclass
class EarlyWarning:
    """Early warning indicator."""
    warning_id: str
    warning_type: str
    level: str
    timestamp: datetime

    title: str
    message: str
    affected_metric: str
    current_value: float
    threshold_value: float

    trend_direction: str
    estimated_time_to_threshold: Optional[int]

    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "warning_id": self.warning_id,
            "warning_type": self.warning_type,
            "level": self.level,
            "timestamp": self.timestamp.isoformat(),
            "title": self.title,
            "message": self.message,
            "affected_metric": self.affected_metric,
            "current_value": self.current_value,
            "threshold_value": self.threshold_value,
            "trend_direction": self.trend_direction,
            "estimated_time_to_threshold": self.estimated_time_to_threshold,
            "recommendations": self.recommendations
        }


@dataclass
class PredictionResult:
    """Result of predictive analysis."""
    predicted_score: float
    confidence_interval: Tuple[float, float]
    confidence_level: float

    predictions: List[Dict[str, Any]] = field(default_factory=list)

    probability_yellow: float = 0.0
    probability_red: float = 0.0

    model_confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "predicted_score": self.predicted_score,
            "confidence_interval": list(self.confidence_interval),
            "confidence_level": self.confidence_level,
            "predictions": self.predictions,
            "probability_yellow": self.probability_yellow,
            "probability_red": self.probability_red,
            "model_confidence": self.model_confidence
        }


@dataclass
class VersionTrend:
    """Trend analysis for a specific version."""
    version: str
    first_seen: datetime
    last_seen: datetime
    snapshot_count: int

    health_history: List[str]
    current_health: str
    degradation_count: int
    recovery_count: int

    stability_score: float
    is_consistently_degrading: bool

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "version": self.version,
            "first_seen": self.first_seen.isoformat(),
            "last_seen": self.last_seen.isoformat(),
            "snapshot_count": self.snapshot_count,
            "health_history": self.health_history,
            "current_health": self.current_health,
            "degradation_count": self.degradation_count,
            "recovery_count": self.recovery_count,
            "stability_score": self.stability_score,
            "is_consistently_degrading": self.is_consistently_degrading
        }


@dataclass
class IssueTrend:
    """Trend analysis for issues."""
    total_issues_trend: RegressionResult
    critical_issues_trend: RegressionResult

    issues_per_snapshot: float
    critical_per_snapshot: float

    recent_total_issues: int
    recent_critical_issues: int

    direction: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_issues_trend": self.total_issues_trend.to_dict(),
            "critical_issues_trend": self.critical_issues_trend.to_dict(),
            "issues_per_snapshot": self.issues_per_snapshot,
            "critical_per_snapshot": self.critical_per_snapshot,
            "recent_total_issues": self.recent_total_issues,
            "recent_critical_issues": self.recent_critical_issues,
            "direction": self.direction
        }


@dataclass
class ScoreTrend:
    """Comprehensive score trend analysis."""
    regression: RegressionResult
    moving_averages: Dict[int, MovingAverageResult]
    volatility: VolatilityResult

    direction: str
    gradient: float

    score_history: List[float]
    timestamps: List[datetime]

    current_score: float
    previous_score: float
    score_change: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "regression": self.regression.to_dict(),
            "moving_averages": {str(k): v.to_dict() for k, v in self.moving_averages.items()},
            "volatility": self.volatility.to_dict(),
            "direction": self.direction,
            "gradient": self.gradient,
            "score_history": self.score_history,
            "timestamps": [t.isoformat() for t in self.timestamps],
            "current_score": self.current_score,
            "previous_score": self.previous_score,
            "score_change": self.score_change
        }


@dataclass
class TrendGraphData:
    """Data for rendering trend graphs."""
    timestamps: List[str]
    scores: List[float]
    issues: List[int]
    critical_issues: List[int]

    regression_line: List[float]
    moving_average_3: List[Optional[float]]
    moving_average_7: List[Optional[float]]
    moving_average_14: List[Optional[float]]

    prediction_timestamps: List[str]
    prediction_values: List[float]
    prediction_lower: List[float]
    prediction_upper: List[float]

    anomaly_points: List[Dict[str, Any]]
    warning_points: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


# ============================================================================
# Main Output Data Class
# ============================================================================

@dataclass
class TrendReport:
    """Complete trend analysis report."""
    report_id: str
    generated_at: str
    history_dir: str

    snapshots_analyzed: int
    first_snapshot: str
    last_snapshot: str
    analysis_window_days: int

    overall_trend: str
    trend_confidence: float

    score_trend: Optional[ScoreTrend] = None

    current_score: float = 0.0
    current_health: str = "unknown"

    predicted_next_score: float = 0.0
    confidence_interval: Tuple[float, float] = field(default_factory=lambda: (0.0, 0.0))
    prediction: Optional[PredictionResult] = None

    ma_3: float = 0.0
    ma_7: float = 0.0
    ma_14: float = 0.0

    score_volatility: float = 0.0
    volatility_trend: str = "stable"

    issue_trend: Optional[IssueTrend] = None

    anomalies: List[Anomaly] = field(default_factory=list)
    total_anomalies: int = 0

    early_warnings: List[EarlyWarning] = field(default_factory=list)
    total_warnings: int = 0

    version_trends: List[VersionTrend] = field(default_factory=list)
    degrading_versions: List[str] = field(default_factory=list)

    trend_graph_data: Optional[TrendGraphData] = None

    regression_slope: float = 0.0
    regression_intercept: float = 0.0
    regression_r_squared: float = 0.0

    analysis_duration_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "report_id": self.report_id,
            "generated_at": self.generated_at,
            "history_dir": self.history_dir,
            "snapshots_analyzed": self.snapshots_analyzed,
            "first_snapshot": self.first_snapshot,
            "last_snapshot": self.last_snapshot,
            "analysis_window_days": self.analysis_window_days,
            "overall_trend": self.overall_trend,
            "trend_confidence": self.trend_confidence,
            "current_score": self.current_score,
            "current_health": self.current_health,
            "predicted_next_score": self.predicted_next_score,
            "confidence_interval": list(self.confidence_interval),
            "ma_3": self.ma_3,
            "ma_7": self.ma_7,
            "ma_14": self.ma_14,
            "score_volatility": self.score_volatility,
            "volatility_trend": self.volatility_trend,
            "total_anomalies": self.total_anomalies,
            "anomalies": [a.to_dict() for a in self.anomalies],
            "total_warnings": self.total_warnings,
            "early_warnings": [w.to_dict() for w in self.early_warnings],
            "degrading_versions": self.degrading_versions,
            "version_trends": [v.to_dict() for v in self.version_trends],
            "regression_slope": self.regression_slope,
            "regression_intercept": self.regression_intercept,
            "regression_r_squared": self.regression_r_squared,
            "analysis_duration_ms": self.analysis_duration_ms
        }

        if self.score_trend:
            result["score_trend"] = self.score_trend.to_dict()
        if self.prediction:
            result["prediction"] = self.prediction.to_dict()
        if self.issue_trend:
            result["issue_trend"] = self.issue_trend.to_dict()
        if self.trend_graph_data:
            result["trend_graph_data"] = self.trend_graph_data.to_dict()

        return result


# ============================================================================
# Statistical Utilities
# ============================================================================

class StatisticsCalculator:
    """
    Statistical computation utilities.

    Implements pure-Python statistical methods to avoid numpy dependency.
    """

    @staticmethod
    def mean(values: List[float]) -> float:
        """Calculate arithmetic mean."""
        if not values:
            return 0.0
        return sum(values) / len(values)

    @staticmethod
    def standard_deviation(values: List[float], sample: bool = True) -> float:
        """Calculate standard deviation."""
        if len(values) < 2:
            return 0.0

        mean_val = StatisticsCalculator.mean(values)
        variance = sum((x - mean_val) ** 2 for x in values)

        divisor = len(values) - 1 if sample else len(values)
        if divisor == 0:
            return 0.0

        return math.sqrt(variance / divisor)

    @staticmethod
    def variance(values: List[float], sample: bool = True) -> float:
        """Calculate variance."""
        if len(values) < 2:
            return 0.0

        mean_val = StatisticsCalculator.mean(values)
        variance = sum((x - mean_val) ** 2 for x in values)

        divisor = len(values) - 1 if sample else len(values)
        if divisor == 0:
            return 0.0

        return variance / divisor

    @staticmethod
    def linear_regression(x: List[float], y: List[float]) -> RegressionResult:
        """Perform ordinary least squares linear regression."""
        n = len(x)
        if n < 2 or len(y) != n:
            return RegressionResult(
                slope=0.0,
                intercept=0.0,
                r_squared=0.0,
                standard_error=0.0,
                p_value=1.0
            )

        mean_x = StatisticsCalculator.mean(x)
        mean_y = StatisticsCalculator.mean(y)

        # Calculate slope and intercept
        numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
        denominator = sum((x[i] - mean_x) ** 2 for i in range(n))

        if denominator == 0:
            return RegressionResult(
                slope=0.0,
                intercept=mean_y,
                r_squared=0.0,
                standard_error=0.0,
                p_value=1.0
            )

        slope = numerator / denominator
        intercept = mean_y - slope * mean_x

        # Calculate R-squared
        ss_res = sum((y[i] - (slope * x[i] + intercept)) ** 2 for i in range(n))
        ss_tot = sum((y[i] - mean_y) ** 2 for i in range(n))

        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
        r_squared = max(0.0, min(1.0, r_squared))  # Clamp to [0, 1]

        # Calculate standard error of slope
        if n > 2:
            mse = ss_res / (n - 2)
            se_slope = math.sqrt(mse / denominator) if denominator > 0 else 0.0
        else:
            se_slope = 0.0

        # Calculate p-value using t-statistic approximation
        if se_slope > 0:
            t_stat = abs(slope) / se_slope
            # Simplified p-value approximation
            df = n - 2
            p_value = StatisticsCalculator._t_to_p(t_stat, df)
        else:
            p_value = 1.0 if slope == 0 else 0.0

        return RegressionResult(
            slope=slope,
            intercept=intercept,
            r_squared=r_squared,
            standard_error=se_slope,
            p_value=p_value
        )

    @staticmethod
    def _t_to_p(t_stat: float, df: int) -> float:
        """Approximate p-value from t-statistic (two-tailed)."""
        # Simplified approximation without scipy
        # Using normal approximation for large df
        if df <= 0:
            return 1.0

        if df > 100:
            # Use normal approximation for large df
            z = t_stat
            # Approximation: P(Z > z) for standard normal
            p_one_tail = 0.5 * math.erfc(z / math.sqrt(2))
            return 2 * p_one_tail
        else:
            # Simple approximation based on typical t-distribution behavior
            # This is a rough approximation
            if t_stat < 1:
                return 0.5
            elif t_stat < 2:
                return 0.1
            elif t_stat < 3:
                return 0.02
            elif t_stat < 4:
                return 0.001
            else:
                return 0.0001

    @staticmethod
    def moving_average(values: List[float], window: int) -> List[Optional[float]]:
        """Calculate simple moving average."""
        if not values or window <= 0:
            return []

        result: List[Optional[float]] = []
        for i in range(len(values)):
            if i < window - 1:
                result.append(None)
            else:
                window_values = values[i - window + 1:i + 1]
                result.append(StatisticsCalculator.mean(window_values))

        return result

    @staticmethod
    def zscore(value: float, mean: float, std: float) -> float:
        """Calculate z-score."""
        if std == 0:
            return 0.0
        return (value - mean) / std

    @staticmethod
    def t_distribution_ppf(confidence: float, df: int) -> float:
        """Approximate t-distribution percent point function."""
        # Approximation for confidence intervals
        if df <= 0:
            return 0.0

        # Common t-values for 95% confidence
        t_values = {
            1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
            6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
            15: 2.131, 20: 2.086, 25: 2.060, 30: 2.042, 60: 2.000,
            120: 1.980
        }

        alpha = 1 - confidence
        if confidence == 0.95:
            # Use table lookup
            if df in t_values:
                return t_values[df]
            elif df < 30:
                # Interpolate
                lower = max(k for k in t_values.keys() if k <= df)
                upper = min(k for k in t_values.keys() if k >= df)
                if lower == upper:
                    return t_values[lower]
                frac = (df - lower) / (upper - lower)
                return t_values[lower] + frac * (t_values[upper] - t_values[lower])
            else:
                return 1.96  # Normal approximation
        else:
            # Rough approximation for other confidence levels
            z = 1.96 * (confidence / 0.95)
            return z

    @staticmethod
    def confidence_interval(
        mean: float,
        std_error: float,
        confidence: float,
        n: int
    ) -> Tuple[float, float]:
        """Calculate confidence interval."""
        if n < 2:
            return (mean, mean)

        df = n - 1
        t_value = StatisticsCalculator.t_distribution_ppf(confidence, df)
        margin = t_value * std_error

        return (mean - margin, mean + margin)


# ============================================================================
# Health History Store
# ============================================================================

class HealthHistoryStore:
    """
    Stores and retrieves historical dashboard snapshots.

    Directory Structure:
        history_dir/
            index.json
            snapshots/
                2024-01-01T00-00-00_v1.0.0.json
                ...
    """

    INDEX_FILENAME = "index.json"
    SNAPSHOTS_DIR = "snapshots"

    def __init__(self, history_dir: Path):
        """Initialize the history store."""
        self.history_dir = Path(history_dir)
        self.snapshots_dir = self.history_dir / self.SNAPSHOTS_DIR
        self.index_path = self.history_dir / self.INDEX_FILENAME

        self._index: Dict[str, SnapshotMetadata] = {}
        self._index_loaded = False

    def initialize(self) -> None:
        """Initialize the history store directory structure."""
        try:
            self.history_dir.mkdir(parents=True, exist_ok=True)
            self.snapshots_dir.mkdir(parents=True, exist_ok=True)

            if self.index_path.exists():
                self._load_index()
            else:
                self._index = {}
                self._save_index()

            self._index_loaded = True

        except Exception as e:
            raise HistoryStoreError(f"Failed to initialize history store: {e}")

    def add_snapshot(
        self,
        dashboard_path: Path,
        version: Optional[str] = None,
        timestamp: Optional[datetime] = None
    ) -> SnapshotMetadata:
        """Add a dashboard snapshot to the history store."""
        if not self._index_loaded:
            self.initialize()

        try:
            # Load and validate dashboard
            with open(dashboard_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Validate required fields
            if "repository_score" not in data and "overall_health" not in data:
                raise InvalidSnapshotError(
                    "Dashboard missing required fields: repository_score or overall_health"
                )

            # Extract version if not provided
            if version is None:
                version = data.get("version") or data.get("latest_version")

            # Use provided or current timestamp
            if timestamp is None:
                # Try to extract from dashboard
                ts_str = data.get("scan_timestamp") or data.get("generated_at")
                if ts_str:
                    try:
                        timestamp = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
                    except (ValueError, AttributeError):
                        timestamp = datetime.utcnow()
                else:
                    timestamp = datetime.utcnow()

            # Generate snapshot ID
            snapshot_id = self._generate_snapshot_id(timestamp, version)

            # Copy file to snapshots directory
            dest_path = self.snapshots_dir / f"{snapshot_id}.json"
            shutil.copy2(dashboard_path, dest_path)

            # Create metadata
            metadata = self._extract_snapshot_metadata(dest_path, data, snapshot_id, timestamp, version)

            # Update index
            self._index[snapshot_id] = metadata
            self._save_index()

            logger.info(f"Added snapshot: {snapshot_id}")
            return metadata

        except InvalidSnapshotError:
            raise
        except Exception as e:
            raise HistoryStoreError(f"Failed to add snapshot: {e}")

    def load_snapshot(self, snapshot_id: str) -> DashboardSnapshot:
        """Load a specific snapshot by ID."""
        if not self._index_loaded:
            self.initialize()

        if snapshot_id not in self._index:
            raise InvalidSnapshotError(f"Snapshot not found: {snapshot_id}")

        metadata = self._index[snapshot_id]

        try:
            with open(metadata.file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            versions_health = data.get("versions_health", [])

            return DashboardSnapshot(
                metadata=metadata,
                data=data,
                versions_health=versions_health
            )

        except Exception as e:
            raise InvalidSnapshotError(f"Failed to load snapshot {snapshot_id}: {e}")

    def get_all_snapshots(self) -> List[SnapshotMetadata]:
        """Get metadata for all snapshots, sorted by timestamp."""
        if not self._index_loaded:
            self.initialize()

        snapshots = list(self._index.values())
        snapshots.sort(key=lambda s: s.timestamp)
        return snapshots

    def get_last_n_snapshots(self, n: int) -> List[DashboardSnapshot]:
        """Load the last N snapshots."""
        if not self._index_loaded:
            self.initialize()

        all_metadata = self.get_all_snapshots()

        if len(all_metadata) < n:
            n = len(all_metadata)

        if n == 0:
            raise InsufficientHistoryError("No snapshots available")

        # Get last n snapshot IDs
        last_n = all_metadata[-n:]

        # Load each snapshot
        snapshots = []
        for metadata in last_n:
            snapshot = self.load_snapshot(metadata.snapshot_id)
            snapshots.append(snapshot)

        return snapshots

    def get_snapshots_in_range(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> List[DashboardSnapshot]:
        """Load snapshots within a date range."""
        if not self._index_loaded:
            self.initialize()

        all_metadata = self.get_all_snapshots()

        # Filter by date range
        filtered = [
            m for m in all_metadata
            if start_date <= m.timestamp <= end_date
        ]

        # Load each snapshot
        snapshots = []
        for metadata in filtered:
            snapshot = self.load_snapshot(metadata.snapshot_id)
            snapshots.append(snapshot)

        return snapshots

    def get_snapshot_count(self) -> int:
        """Get the total number of snapshots in the store."""
        if not self._index_loaded:
            self.initialize()
        return len(self._index)

    def validate_index(self) -> Tuple[bool, List[str]]:
        """Validate index integrity."""
        if not self._index_loaded:
            self.initialize()

        issues = []

        for snapshot_id, metadata in self._index.items():
            if not metadata.file_path.exists():
                issues.append(f"Missing file: {snapshot_id}")
            else:
                # Validate JSON
                try:
                    with open(metadata.file_path, 'r', encoding='utf-8') as f:
                        json.load(f)
                except json.JSONDecodeError:
                    issues.append(f"Invalid JSON: {snapshot_id}")

        return (len(issues) == 0, issues)

    def rebuild_index(self) -> int:
        """Rebuild the index from snapshot files."""
        self._index = {}

        if not self.snapshots_dir.exists():
            self.snapshots_dir.mkdir(parents=True, exist_ok=True)
            self._save_index()
            return 0

        count = 0
        for file_path in self.snapshots_dir.glob("*.json"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Extract snapshot ID from filename
                snapshot_id = file_path.stem

                # Try to parse timestamp from ID
                try:
                    ts_str = snapshot_id.split('_')[0].replace('-', ':')[:19]
                    timestamp = datetime.fromisoformat(ts_str.replace(':', '-', 2))
                except (ValueError, IndexError):
                    timestamp = datetime.fromtimestamp(file_path.stat().st_mtime)

                # Extract version
                version = None
                if '_v' in snapshot_id:
                    version = snapshot_id.split('_v')[-1]

                metadata = self._extract_snapshot_metadata(
                    file_path, data, snapshot_id, timestamp, version
                )
                self._index[snapshot_id] = metadata
                count += 1

            except Exception as e:
                logger.warning(f"Failed to index {file_path}: {e}")

        self._save_index()
        self._index_loaded = True
        return count

    def _load_index(self) -> None:
        """Load the index from disk."""
        try:
            with open(self.index_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self._index = {}
            for snapshot_id, metadata_dict in data.get("snapshots", {}).items():
                self._index[snapshot_id] = SnapshotMetadata.from_dict(metadata_dict)

            self._index_loaded = True

        except Exception as e:
            logger.warning(f"Failed to load index, rebuilding: {e}")
            self.rebuild_index()

    def _save_index(self) -> None:
        """Save the index to disk."""
        try:
            data = {
                "version": "1.0.0",
                "updated_at": datetime.utcnow().isoformat(),
                "snapshot_count": len(self._index),
                "snapshots": {
                    sid: metadata.to_dict()
                    for sid, metadata in self._index.items()
                }
            }

            # Atomic write
            temp_path = self.index_path.with_suffix('.tmp')
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)

            temp_path.replace(self.index_path)

        except Exception as e:
            raise HistoryStoreError(f"Failed to save index: {e}")

    def _generate_snapshot_id(
        self,
        timestamp: datetime,
        version: Optional[str]
    ) -> str:
        """Generate a unique snapshot ID."""
        ts_str = timestamp.strftime("%Y-%m-%dT%H-%M-%S")
        if version:
            # Sanitize version for filename
            safe_version = version.replace('/', '-').replace('\\', '-')
            return f"{ts_str}_v{safe_version}"
        else:
            return f"{ts_str}_unknown"

    def _extract_snapshot_metadata(
        self,
        file_path: Path,
        data: Dict[str, Any],
        snapshot_id: str,
        timestamp: datetime,
        version: Optional[str]
    ) -> SnapshotMetadata:
        """Extract metadata from a dashboard JSON."""
        return SnapshotMetadata(
            snapshot_id=snapshot_id,
            timestamp=timestamp,
            version=version,
            file_path=file_path,
            file_size=file_path.stat().st_size,
            repository_score=data.get("repository_score", 0.0),
            overall_health=data.get("overall_health", "unknown"),
            total_issues=data.get("total_issues", 0),
            critical_issues=data.get("critical_issues", 0)
        )


# ============================================================================
# Trend Analyzer
# ============================================================================

class TrendAnalyzer:
    """
    Core time-series analysis engine.

    Computes:
    - Score trends via linear regression
    - Moving averages (3, 7, 14 snapshot windows)
    - Volatility metrics
    - Issue trends
    - Version stability trends
    - Anomaly detection
    - Predictive health scores
    - Early warning indicators
    """

    def __init__(self, config: TrendConfig):
        """Initialize the trend analyzer."""
        self.config = config
        self.stats = StatisticsCalculator()

        self.snapshots: List[DashboardSnapshot] = []

        self._score_trend: Optional[ScoreTrend] = None
        self._issue_trend: Optional[IssueTrend] = None
        self._anomalies: List[Anomaly] = []
        self._early_warnings: List[EarlyWarning] = []
        self._version_trends: Dict[str, VersionTrend] = {}
        self._prediction: Optional[PredictionResult] = None

    def load_snapshots(self, snapshots: List[DashboardSnapshot]) -> None:
        """Load snapshots for analysis."""
        if len(snapshots) < self.config.min_snapshots:
            raise InsufficientHistoryError(
                f"Need at least {self.config.min_snapshots} snapshots, "
                f"got {len(snapshots)}"
            )

        self.snapshots = snapshots

        # Reset computed results
        self._score_trend = None
        self._issue_trend = None
        self._anomalies = []
        self._early_warnings = []
        self._version_trends = {}
        self._prediction = None

    def analyze(self) -> TrendReport:
        """Perform complete trend analysis."""
        if not self.snapshots:
            raise InsufficientHistoryError("No snapshots loaded")

        start_time = datetime.utcnow()

        try:
            # Compute all trends
            score_trend = self.compute_score_trend()
            issue_trend = self.compute_issue_trend()
            version_trends = self.compute_version_trends()
            anomalies = self.detect_anomalies()
            prediction = self.generate_prediction()
            early_warnings = self.generate_early_warnings()
            graph_data = self.generate_graph_data()

            # Build report
            first_snapshot = self.snapshots[0]
            last_snapshot = self.snapshots[-1]

            days_diff = (last_snapshot.metadata.timestamp - first_snapshot.metadata.timestamp).days

            # Get MA values
            ma_3 = score_trend.moving_averages.get(3, MovingAverageResult(3, [], 0.0)).current_value
            ma_7 = score_trend.moving_averages.get(7, MovingAverageResult(7, [], 0.0)).current_value
            ma_14 = score_trend.moving_averages.get(14, MovingAverageResult(14, [], 0.0)).current_value

            # Determine degrading versions
            degrading = [vt.version for vt in version_trends if vt.is_consistently_degrading]

            report = TrendReport(
                report_id=f"trend_report_{datetime.utcnow().isoformat()}",
                generated_at=datetime.utcnow().isoformat(),
                history_dir=str(self.config.history_dir),
                snapshots_analyzed=len(self.snapshots),
                first_snapshot=first_snapshot.metadata.timestamp.isoformat(),
                last_snapshot=last_snapshot.metadata.timestamp.isoformat(),
                analysis_window_days=days_diff,
                overall_trend=score_trend.direction,
                trend_confidence=score_trend.regression.r_squared,
                score_trend=score_trend,
                current_score=score_trend.current_score,
                current_health=last_snapshot.metadata.overall_health,
                predicted_next_score=prediction.predicted_score,
                confidence_interval=prediction.confidence_interval,
                prediction=prediction,
                ma_3=ma_3,
                ma_7=ma_7,
                ma_14=ma_14,
                score_volatility=score_trend.volatility.standard_deviation,
                volatility_trend=score_trend.volatility.volatility_trend,
                issue_trend=issue_trend,
                anomalies=anomalies,
                total_anomalies=len(anomalies),
                early_warnings=early_warnings,
                total_warnings=len(early_warnings),
                version_trends=version_trends,
                degrading_versions=degrading,
                trend_graph_data=graph_data,
                regression_slope=score_trend.regression.slope,
                regression_intercept=score_trend.regression.intercept,
                regression_r_squared=score_trend.regression.r_squared
            )

            duration = (datetime.utcnow() - start_time).total_seconds() * 1000
            report.analysis_duration_ms = duration

            return report

        except Exception as e:
            raise ComputationError(f"Trend analysis failed: {e}")

    def compute_score_trend(self) -> ScoreTrend:
        """Compute comprehensive score trend analysis."""
        # Extract score history
        scores = [s.metadata.repository_score for s in self.snapshots]
        timestamps = [s.metadata.timestamp for s in self.snapshots]

        # Create x values (0, 1, 2, ...)
        x_values = list(range(len(scores)))

        # Perform linear regression
        regression = StatisticsCalculator.linear_regression(x_values, scores)

        # Calculate moving averages
        moving_averages = self._compute_moving_averages(scores)

        # Calculate volatility
        volatility = self._compute_volatility(scores)

        # Determine trend direction
        direction = self._determine_trend_direction(regression, volatility)

        # Current and previous scores
        current_score = scores[-1]
        previous_score = scores[-2] if len(scores) > 1 else current_score
        score_change = current_score - previous_score

        self._score_trend = ScoreTrend(
            regression=regression,
            moving_averages=moving_averages,
            volatility=volatility,
            direction=direction.value,
            gradient=regression.slope,
            score_history=scores,
            timestamps=timestamps,
            current_score=current_score,
            previous_score=previous_score,
            score_change=score_change
        )

        return self._score_trend

    def _compute_moving_averages(self, values: List[float]) -> Dict[int, MovingAverageResult]:
        """Compute moving averages for configured windows."""
        result = {}

        for window in self.config.ma_windows:
            ma_values = StatisticsCalculator.moving_average(values, window)

            # Get current (last) MA value
            current_value = 0.0
            for v in reversed(ma_values):
                if v is not None:
                    current_value = v
                    break

            result[window] = MovingAverageResult(
                window_size=window,
                values=ma_values,
                current_value=current_value
            )

        return result

    def _compute_volatility(self, values: List[float]) -> VolatilityResult:
        """Compute volatility metrics."""
        if len(values) < 2:
            return VolatilityResult(
                standard_deviation=0.0,
                variance=0.0,
                coefficient_of_variation=0.0,
                recent_volatility=0.0,
                volatility_trend="stable"
            )

        # Overall metrics
        std_dev = StatisticsCalculator.standard_deviation(values)
        variance = StatisticsCalculator.variance(values)
        mean = StatisticsCalculator.mean(values)
        cv = std_dev / mean if mean != 0 else 0.0

        # Recent volatility (last 5 or half, whichever is smaller)
        recent_n = min(5, len(values) // 2)
        if recent_n >= 2:
            recent_std = StatisticsCalculator.standard_deviation(values[-recent_n:])
        else:
            recent_std = std_dev

        # Volatility trend (comparing first half to second half)
        if len(values) >= 4:
            mid = len(values) // 2
            first_half_std = StatisticsCalculator.standard_deviation(values[:mid])
            second_half_std = StatisticsCalculator.standard_deviation(values[mid:])

            if second_half_std > first_half_std * 1.2:
                vol_trend = "increasing"
            elif second_half_std < first_half_std * 0.8:
                vol_trend = "decreasing"
            else:
                vol_trend = "stable"
        else:
            vol_trend = "stable"

        return VolatilityResult(
            standard_deviation=std_dev,
            variance=variance,
            coefficient_of_variation=cv,
            recent_volatility=recent_std,
            volatility_trend=vol_trend
        )

    def _determine_trend_direction(
        self,
        regression: RegressionResult,
        volatility: VolatilityResult
    ) -> TrendDirection:
        """Determine overall trend direction."""
        slope = regression.slope

        # Consider slope relative to volatility
        threshold = volatility.standard_deviation * 0.1 if volatility.standard_deviation > 0 else 1.0

        if slope > threshold:
            return TrendDirection.IMPROVING
        elif slope < -threshold:
            return TrendDirection.DECLINING
        else:
            return TrendDirection.STABLE

    def compute_issue_trend(self) -> IssueTrend:
        """Compute issue trend analysis."""
        total_issues = [s.metadata.total_issues for s in self.snapshots]
        critical_issues = [s.metadata.critical_issues for s in self.snapshots]

        x_values = list(range(len(total_issues)))

        # Regression for total and critical issues
        total_regression = StatisticsCalculator.linear_regression(x_values, total_issues)
        critical_regression = StatisticsCalculator.linear_regression(x_values, critical_issues)

        # Rate of change
        issues_per_snapshot = total_regression.slope
        critical_per_snapshot = critical_regression.slope

        # Recent issues (last 3 snapshots)
        recent_n = min(3, len(self.snapshots))
        recent_total = sum(total_issues[-recent_n:])
        recent_critical = sum(critical_issues[-recent_n:])

        # Direction
        if issues_per_snapshot > 0.5:
            direction = TrendDirection.DECLINING.value  # More issues = declining health
        elif issues_per_snapshot < -0.5:
            direction = TrendDirection.IMPROVING.value
        else:
            direction = TrendDirection.STABLE.value

        self._issue_trend = IssueTrend(
            total_issues_trend=total_regression,
            critical_issues_trend=critical_regression,
            issues_per_snapshot=issues_per_snapshot,
            critical_per_snapshot=critical_per_snapshot,
            recent_total_issues=recent_total,
            recent_critical_issues=recent_critical,
            direction=direction
        )

        return self._issue_trend

    def compute_version_trends(self) -> List[VersionTrend]:
        """Compute stability trends for each version."""
        version_data: Dict[str, Dict[str, Any]] = {}

        # Collect version health across snapshots
        for snapshot in self.snapshots:
            for vh in snapshot.versions_health:
                version = vh.get("version", "unknown")
                health = vh.get("health_status", "unknown")

                if version not in version_data:
                    version_data[version] = {
                        "first_seen": snapshot.metadata.timestamp,
                        "last_seen": snapshot.metadata.timestamp,
                        "health_history": [],
                        "snapshot_count": 0
                    }

                version_data[version]["last_seen"] = snapshot.metadata.timestamp
                version_data[version]["health_history"].append(health)
                version_data[version]["snapshot_count"] += 1

        # Analyze each version
        trends = []
        for version, data in version_data.items():
            health_history = data["health_history"]

            # Count degradations and recoveries
            degradation_count = 0
            recovery_count = 0
            health_order = {"green": 0, "yellow": 1, "red": 2}

            for i in range(1, len(health_history)):
                prev = health_order.get(health_history[i-1], 0)
                curr = health_order.get(health_history[i], 0)

                if curr > prev:
                    degradation_count += 1
                elif curr < prev:
                    recovery_count += 1

            # Compute stability score
            stability_score, is_degrading = self._compute_version_stability(health_history)

            trends.append(VersionTrend(
                version=version,
                first_seen=data["first_seen"],
                last_seen=data["last_seen"],
                snapshot_count=data["snapshot_count"],
                health_history=health_history,
                current_health=health_history[-1] if health_history else "unknown",
                degradation_count=degradation_count,
                recovery_count=recovery_count,
                stability_score=stability_score,
                is_consistently_degrading=is_degrading
            ))

        self._version_trends = {vt.version: vt for vt in trends}
        return trends

    def _compute_version_stability(
        self,
        health_history: List[str]
    ) -> Tuple[float, bool]:
        """Compute stability score for a version."""
        if not health_history:
            return 100.0, False

        # Stability based on health status distribution
        green_count = health_history.count("green")
        yellow_count = health_history.count("yellow")
        red_count = health_history.count("red")
        total = len(health_history)

        # Weighted score (green=100, yellow=50, red=0)
        stability = (green_count * 100 + yellow_count * 50) / total if total > 0 else 0

        # Check for consistent degradation (3+ consecutive degradations)
        is_degrading = False
        if len(health_history) >= 3:
            health_order = {"green": 0, "yellow": 1, "red": 2}
            consecutive = 0
            for i in range(1, len(health_history)):
                prev = health_order.get(health_history[i-1], 0)
                curr = health_order.get(health_history[i], 0)
                if curr > prev:
                    consecutive += 1
                    if consecutive >= 2:
                        is_degrading = True
                        break
                else:
                    consecutive = 0

        return stability, is_degrading

    def detect_anomalies(self) -> List[Anomaly]:
        """Detect anomalies in the time series."""
        anomalies = []

        # Score anomalies
        anomalies.extend(self._detect_score_anomalies())

        # Issue anomalies
        anomalies.extend(self._detect_issue_anomalies())

        self._anomalies = anomalies
        return anomalies

    def _detect_score_anomalies(self) -> List[Anomaly]:
        """Detect anomalies in score time series."""
        anomalies = []

        scores = [s.metadata.repository_score for s in self.snapshots]
        mean = StatisticsCalculator.mean(scores)
        std = StatisticsCalculator.standard_deviation(scores)

        for i, snapshot in enumerate(self.snapshots):
            score = snapshot.metadata.repository_score
            zscore = StatisticsCalculator.zscore(score, mean, std)

            # Check for z-score anomaly
            if abs(zscore) > self.config.zscore_threshold:
                anomaly_type = (
                    AnomalyType.SUDDEN_SCORE_DROP if zscore < 0
                    else AnomalyType.SUDDEN_SCORE_SPIKE
                )

                severity = WarningLevel.HIGH if abs(zscore) > 3 else WarningLevel.MEDIUM

                anomalies.append(Anomaly(
                    anomaly_id=f"anomaly_score_{snapshot.metadata.snapshot_id}",
                    anomaly_type=anomaly_type.value,
                    timestamp=snapshot.metadata.timestamp,
                    snapshot_id=snapshot.metadata.snapshot_id,
                    severity=severity.value,
                    actual_value=score,
                    expected_value=mean,
                    deviation=abs(score - mean),
                    zscore=zscore,
                    description=f"Score {score:.1f} deviates significantly from mean {mean:.1f}",
                    affected_metric="repository_score"
                ))

            # Check for sudden drop
            if i > 0:
                prev_score = scores[i-1]
                drop = prev_score - score

                if drop >= self.config.score_drop_threshold:
                    anomalies.append(Anomaly(
                        anomaly_id=f"anomaly_drop_{snapshot.metadata.snapshot_id}",
                        anomaly_type=AnomalyType.SUDDEN_SCORE_DROP.value,
                        timestamp=snapshot.metadata.timestamp,
                        snapshot_id=snapshot.metadata.snapshot_id,
                        severity=WarningLevel.HIGH.value,
                        actual_value=score,
                        expected_value=prev_score,
                        deviation=drop,
                        zscore=StatisticsCalculator.zscore(score, prev_score, std) if std > 0 else 0,
                        description=f"Score dropped {drop:.1f} points from previous snapshot",
                        affected_metric="repository_score"
                    ))

        return anomalies

    def _detect_issue_anomalies(self) -> List[Anomaly]:
        """Detect anomalies in issue counts."""
        anomalies = []

        issues = [s.metadata.total_issues for s in self.snapshots]

        for i in range(1, len(self.snapshots)):
            snapshot = self.snapshots[i]
            current = issues[i]
            previous = issues[i-1]
            increase = current - previous

            if increase >= self.config.issue_spike_threshold:
                mean = StatisticsCalculator.mean(issues[:i]) if i > 0 else 0
                std = StatisticsCalculator.standard_deviation(issues[:i]) if i > 1 else 1
                zscore = StatisticsCalculator.zscore(current, mean, std) if std > 0 else 0

                anomalies.append(Anomaly(
                    anomaly_id=f"anomaly_issues_{snapshot.metadata.snapshot_id}",
                    anomaly_type=AnomalyType.ISSUE_SPIKE.value,
                    timestamp=snapshot.metadata.timestamp,
                    snapshot_id=snapshot.metadata.snapshot_id,
                    severity=WarningLevel.HIGH.value if increase >= 10 else WarningLevel.MEDIUM.value,
                    actual_value=float(current),
                    expected_value=float(previous),
                    deviation=float(increase),
                    zscore=zscore,
                    description=f"Issues increased by {increase} (from {previous} to {current})",
                    affected_metric="total_issues"
                ))

        return anomalies

    def generate_prediction(self) -> PredictionResult:
        """Generate predictive health scores."""
        if self._score_trend is None:
            self.compute_score_trend()

        regression = self._score_trend.regression
        n = len(self.snapshots)

        predictions = []
        horizon = self.config.prediction_horizon

        # Calculate standard error for predictions
        scores = [s.metadata.repository_score for s in self.snapshots]
        residuals = [
            scores[i] - regression.predict(float(i))
            for i in range(n)
        ]
        residual_std = StatisticsCalculator.standard_deviation(residuals) if len(residuals) > 1 else 0

        for step in range(1, horizon + 1):
            x = n + step - 1
            predicted = regression.predict(float(x))

            # Confidence interval (widens with distance)
            se = residual_std * math.sqrt(1 + 1/n + ((x - n/2)**2) / n)
            t_val = StatisticsCalculator.t_distribution_ppf(self.config.confidence_level, n - 2)

            lower = max(0, predicted - t_val * se)
            upper = min(100, predicted + t_val * se)

            predictions.append({
                "step": step,
                "predicted_score": round(predicted, 2),
                "confidence_interval": (round(lower, 2), round(upper, 2))
            })

        # First prediction
        next_score = predictions[0]["predicted_score"] if predictions else self._score_trend.current_score
        next_ci = predictions[0]["confidence_interval"] if predictions else (next_score, next_score)

        # Estimate probabilities
        prob_yellow = self._estimate_threshold_probability(
            self._score_trend.current_score,
            regression,
            self.config.warning_score_threshold,
            horizon
        )

        prob_red = self._estimate_threshold_probability(
            self._score_trend.current_score,
            regression,
            self.config.critical_score_threshold,
            horizon
        )

        # Model confidence based on R-squared
        model_confidence = regression.r_squared

        self._prediction = PredictionResult(
            predicted_score=next_score,
            confidence_interval=next_ci,
            confidence_level=self.config.confidence_level,
            predictions=predictions,
            probability_yellow=prob_yellow,
            probability_red=prob_red,
            model_confidence=model_confidence
        )

        return self._prediction

    def _estimate_threshold_probability(
        self,
        current_score: float,
        regression: RegressionResult,
        threshold: float,
        horizon: int
    ) -> float:
        """Estimate probability of score crossing threshold."""
        if current_score <= threshold:
            return 1.0  # Already at or below threshold

        if regression.slope >= 0:
            # Improving or stable trend - lower probability
            return 0.1

        # Calculate steps to threshold at current rate
        if regression.slope != 0:
            steps_to_threshold = (current_score - threshold) / abs(regression.slope)

            if steps_to_threshold <= horizon:
                # Use simple linear probability
                return min(1.0, horizon / steps_to_threshold * 0.8)
            else:
                return 0.2

        return 0.1

    def generate_early_warnings(self) -> List[EarlyWarning]:
        """Generate early warning indicators."""
        warnings = []

        # Check degradation
        degradation = self._check_degradation_warning()
        if degradation:
            warnings.append(degradation)

        # Check volatility
        volatility = self._check_volatility_warning()
        if volatility:
            warnings.append(volatility)

        # Check threshold proximity
        warnings.extend(self._check_threshold_proximity())

        self._early_warnings = warnings
        return warnings

    def _check_degradation_warning(self) -> Optional[EarlyWarning]:
        """Check for slow degradation trend."""
        if self._score_trend is None:
            return None

        regression = self._score_trend.regression

        # Only warn if declining trend is significant
        if regression.slope < -1 and regression.is_significant:
            # Estimate time to threshold
            current = self._score_trend.current_score
            threshold = self.config.warning_score_threshold

            if current > threshold and regression.slope != 0:
                steps = int((current - threshold) / abs(regression.slope))
            else:
                steps = None

            return EarlyWarning(
                warning_id=f"warning_degradation_{datetime.utcnow().isoformat()}",
                warning_type="slow_degradation",
                level=WarningLevel.MEDIUM.value,
                timestamp=datetime.utcnow(),
                title="Slow Health Degradation Detected",
                message=f"Repository health is declining at {abs(regression.slope):.1f} points per snapshot",
                affected_metric="repository_score",
                current_value=current,
                threshold_value=threshold,
                trend_direction=TrendDirection.DECLINING.value,
                estimated_time_to_threshold=steps,
                recommendations=[
                    "Investigate root cause of degradation",
                    "Review recent changes and publications",
                    "Consider preventive maintenance"
                ]
            )

        return None

    def _check_volatility_warning(self) -> Optional[EarlyWarning]:
        """Check for increasing volatility."""
        if self._score_trend is None:
            return None

        volatility = self._score_trend.volatility

        if volatility.standard_deviation > self.config.volatility_threshold:
            return EarlyWarning(
                warning_id=f"warning_volatility_{datetime.utcnow().isoformat()}",
                warning_type="high_volatility",
                level=WarningLevel.MEDIUM.value,
                timestamp=datetime.utcnow(),
                title="High Score Volatility",
                message=f"Repository score volatility ({volatility.standard_deviation:.1f}) exceeds threshold",
                affected_metric="repository_score",
                current_value=volatility.standard_deviation,
                threshold_value=self.config.volatility_threshold,
                trend_direction=volatility.volatility_trend,
                estimated_time_to_threshold=None,
                recommendations=[
                    "Investigate sources of instability",
                    "Review intermittent issues",
                    "Consider more frequent monitoring"
                ]
            )

        return None

    def _check_threshold_proximity(self) -> List[EarlyWarning]:
        """Check if score is approaching warning/critical thresholds."""
        warnings = []

        if self._score_trend is None:
            return warnings

        current = self._score_trend.current_score

        # Warning threshold proximity (within 10 points)
        if current <= self.config.warning_score_threshold + 10 and current > self.config.warning_score_threshold:
            warnings.append(EarlyWarning(
                warning_id=f"warning_threshold_{datetime.utcnow().isoformat()}",
                warning_type="approaching_threshold",
                level=WarningLevel.MEDIUM.value,
                timestamp=datetime.utcnow(),
                title="Approaching Warning Threshold",
                message=f"Score {current:.1f} is within 10 points of warning threshold ({self.config.warning_score_threshold})",
                affected_metric="repository_score",
                current_value=current,
                threshold_value=self.config.warning_score_threshold,
                trend_direction=self._score_trend.direction,
                estimated_time_to_threshold=None,
                recommendations=[
                    "Take preventive action to maintain health",
                    "Address any pending issues"
                ]
            ))

        # Critical threshold proximity
        if current <= self.config.critical_score_threshold + 15 and current > self.config.critical_score_threshold:
            warnings.append(EarlyWarning(
                warning_id=f"warning_critical_{datetime.utcnow().isoformat()}",
                warning_type="approaching_critical",
                level=WarningLevel.HIGH.value,
                timestamp=datetime.utcnow(),
                title="Approaching Critical Threshold",
                message=f"Score {current:.1f} is within 15 points of critical threshold ({self.config.critical_score_threshold})",
                affected_metric="repository_score",
                current_value=current,
                threshold_value=self.config.critical_score_threshold,
                trend_direction=self._score_trend.direction,
                estimated_time_to_threshold=None,
                recommendations=[
                    "Immediate action required",
                    "Address all critical and error issues",
                    "Consider emergency maintenance"
                ]
            ))

        return warnings

    def generate_graph_data(self) -> TrendGraphData:
        """Generate data for trend visualization."""
        timestamps = [s.metadata.timestamp.isoformat() for s in self.snapshots]
        scores = [s.metadata.repository_score for s in self.snapshots]
        issues = [s.metadata.total_issues for s in self.snapshots]
        critical = [s.metadata.critical_issues for s in self.snapshots]

        # Regression line
        regression_line = []
        if self._score_trend:
            for i in range(len(scores)):
                regression_line.append(self._score_trend.regression.predict(float(i)))

        # Moving averages
        ma_3 = StatisticsCalculator.moving_average(scores, 3)
        ma_7 = StatisticsCalculator.moving_average(scores, 7)
        ma_14 = StatisticsCalculator.moving_average(scores, 14)

        # Prediction extension
        pred_timestamps = []
        pred_values = []
        pred_lower = []
        pred_upper = []

        if self._prediction:
            last_ts = self.snapshots[-1].metadata.timestamp
            for pred in self._prediction.predictions:
                step = pred["step"]
                # Assume daily snapshots for timestamp
                pred_ts = last_ts + timedelta(days=step)
                pred_timestamps.append(pred_ts.isoformat())
                pred_values.append(pred["predicted_score"])
                pred_lower.append(pred["confidence_interval"][0])
                pred_upper.append(pred["confidence_interval"][1])

        # Anomaly points
        anomaly_points = [
            {
                "timestamp": a.timestamp.isoformat(),
                "value": a.actual_value,
                "type": a.anomaly_type
            }
            for a in self._anomalies
        ]

        # Warning points
        warning_points = [
            {
                "timestamp": w.timestamp.isoformat(),
                "value": w.current_value,
                "message": w.title
            }
            for w in self._early_warnings
        ]

        return TrendGraphData(
            timestamps=timestamps,
            scores=scores,
            issues=issues,
            critical_issues=critical,
            regression_line=regression_line,
            moving_average_3=ma_3,
            moving_average_7=ma_7,
            moving_average_14=ma_14,
            prediction_timestamps=pred_timestamps,
            prediction_values=pred_values,
            prediction_lower=pred_lower,
            prediction_upper=pred_upper,
            anomaly_points=anomaly_points,
            warning_points=warning_points
        )


# ============================================================================
# Chart Generator
# ============================================================================

class TrendChartGenerator:
    """Generates trend visualization charts."""

    def __init__(self, output_dir: Path):
        """Initialize chart generator."""
        self.output_dir = Path(output_dir)
        self._matplotlib_available = self._check_matplotlib()

    def _check_matplotlib(self) -> bool:
        """Check if matplotlib is available."""
        try:
            import matplotlib
            return True
        except ImportError:
            return False

    def generate_charts(
        self,
        graph_data: TrendGraphData,
        report: TrendReport
    ) -> List[Path]:
        """Generate all trend charts."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        generated = []

        try:
            if self._matplotlib_available:
                generated.append(self._generate_score_chart(graph_data))
                generated.append(self._generate_issue_chart(graph_data))
            else:
                # ASCII fallback
                ascii_chart = self._generate_ascii_chart(
                    graph_data.scores,
                    "Repository Health Score Trend"
                )
                chart_path = self.output_dir / "score_trend.txt"
                with open(chart_path, 'w', encoding='utf-8') as f:
                    f.write(ascii_chart)
                generated.append(chart_path)

            return generated

        except Exception as e:
            raise ChartGenerationError(f"Failed to generate charts: {e}")

    def _generate_score_chart(self, graph_data: TrendGraphData) -> Path:
        """Generate score trend chart using matplotlib."""
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        from datetime import datetime

        fig, ax = plt.subplots(figsize=(12, 6))

        # Parse timestamps
        dates = [datetime.fromisoformat(ts) for ts in graph_data.timestamps]

        # Plot score history
        ax.plot(dates, graph_data.scores, 'b-', linewidth=2, label='Score')

        # Plot regression line
        if graph_data.regression_line:
            ax.plot(dates, graph_data.regression_line, 'r--', alpha=0.7, label='Trend')

        # Plot moving averages
        if any(v is not None for v in graph_data.moving_average_7):
            ma7_clean = [(d, v) for d, v in zip(dates, graph_data.moving_average_7) if v is not None]
            if ma7_clean:
                ma7_dates, ma7_vals = zip(*ma7_clean)
                ax.plot(ma7_dates, ma7_vals, 'g-', alpha=0.5, label='7-day MA')

        # Plot predictions
        if graph_data.prediction_timestamps:
            pred_dates = [datetime.fromisoformat(ts) for ts in graph_data.prediction_timestamps]
            ax.plot(pred_dates, graph_data.prediction_values, 'b:', linewidth=2, label='Predicted')
            ax.fill_between(
                pred_dates,
                graph_data.prediction_lower,
                graph_data.prediction_upper,
                alpha=0.2,
                color='blue',
                label='95% CI'
            )

        # Mark anomalies
        for anomaly in graph_data.anomaly_points:
            anom_date = datetime.fromisoformat(anomaly["timestamp"])
            ax.scatter([anom_date], [anomaly["value"]], color='red', s=100, zorder=5, marker='x')

        ax.set_xlabel('Date')
        ax.set_ylabel('Health Score')
        ax.set_title('Repository Health Score Trend')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        fig.autofmt_xdate()

        # Set y-axis limits
        ax.set_ylim(0, 100)

        chart_path = self.output_dir / "score_trend.png"
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        return chart_path

    def _generate_issue_chart(self, graph_data: TrendGraphData) -> Path:
        """Generate issue trend chart."""
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        from datetime import datetime

        fig, ax = plt.subplots(figsize=(12, 6))

        dates = [datetime.fromisoformat(ts) for ts in graph_data.timestamps]

        ax.bar(dates, graph_data.issues, width=0.8, alpha=0.7, label='Total Issues', color='orange')
        ax.bar(dates, graph_data.critical_issues, width=0.8, alpha=0.9, label='Critical', color='red')

        ax.set_xlabel('Date')
        ax.set_ylabel('Issue Count')
        ax.set_title('Repository Issues Trend')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3, axis='y')

        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        fig.autofmt_xdate()

        chart_path = self.output_dir / "issue_trend.png"
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        return chart_path

    def _generate_ascii_chart(
        self,
        values: List[float],
        title: str,
        width: int = 80,
        height: int = 20
    ) -> str:
        """Generate ASCII chart as fallback."""
        if not values:
            return f"{title}\n(No data)"

        min_val = min(values)
        max_val = max(values)
        value_range = max_val - min_val or 1

        lines = [title, "=" * len(title), ""]

        # Y-axis labels
        y_labels = [
            f"{max_val:6.1f} |",
            f"{(max_val + min_val) / 2:6.1f} |",
            f"{min_val:6.1f} |"
        ]

        # Create chart grid
        chart_width = width - 10
        chart_height = height - 5

        # Scale values to chart height
        scaled = [
            int((v - min_val) / value_range * (chart_height - 1))
            for v in values
        ]

        # Sample values if too many
        if len(scaled) > chart_width:
            step = len(scaled) / chart_width
            scaled = [scaled[int(i * step)] for i in range(chart_width)]

        # Build chart rows
        for row in range(chart_height - 1, -1, -1):
            line = "       |"
            for col in range(len(scaled)):
                if scaled[col] >= row:
                    line += "*"
                else:
                    line += " "
            lines.append(line)

        # X-axis
        lines.append("       +" + "-" * len(scaled))
        lines.append(f"        Start {'.' * (len(scaled) - 10)} End")

        # Stats
        lines.append("")
        lines.append(f"Points: {len(values)}")
        lines.append(f"Min: {min_val:.1f}")
        lines.append(f"Max: {max_val:.1f}")
        lines.append(f"Latest: {values[-1]:.1f}")

        return "\n".join(lines)


# ============================================================================
# Main Trend Engine
# ============================================================================

class TrendEngine:
    """Main orchestrator for trend analysis."""

    def __init__(self, config: TrendConfig):
        """Initialize the trend engine."""
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

        self.history_store = HealthHistoryStore(config.history_dir)
        self.analyzer = TrendAnalyzer(config)

        if config.generate_charts and config.chart_output_dir:
            self.chart_generator = TrendChartGenerator(config.chart_output_dir)
        else:
            self.chart_generator = None

    def run(self) -> Tuple[TrendReport, int]:
        """Run the complete trend analysis pipeline."""
        start_time = datetime.utcnow()

        try:
            logger.info("=" * 80)
            logger.info("REPOSITORY HEALTH TREND ANALYZER")
            logger.info("=" * 80)

            # Step 1: Initialize history store
            logger.info("Step 1: Loading history store...")
            self.history_store.initialize()
            snapshot_count = self.history_store.get_snapshot_count()
            logger.info(f"  Found {snapshot_count} snapshot(s)")

            if snapshot_count < self.config.min_snapshots:
                raise InsufficientHistoryError(
                    f"Need at least {self.config.min_snapshots} snapshots, "
                    f"found {snapshot_count}"
                )

            # Step 2: Load snapshots
            logger.info("\nStep 2: Loading snapshots...")
            snapshots = self.history_store.get_last_n_snapshots(
                min(snapshot_count, self.config.max_snapshots)
            )
            logger.info(f"  Loaded {len(snapshots)} snapshot(s)")

            # Step 3: Run analysis
            logger.info("\nStep 3: Analyzing trends...")
            self.analyzer.load_snapshots(snapshots)
            report = self.analyzer.analyze()

            duration = (datetime.utcnow() - start_time).total_seconds() * 1000
            report.analysis_duration_ms = duration

            logger.info(f"  Overall trend: {report.overall_trend}")
            logger.info(f"  Current score: {report.current_score:.1f}")
            logger.info(f"  Predicted next: {report.predicted_next_score:.1f}")
            logger.info(f"  Anomalies: {report.total_anomalies}")
            logger.info(f"  Warnings: {report.total_warnings}")

            # Step 4: Generate charts (optional)
            if self.chart_generator and report.trend_graph_data:
                logger.info("\nStep 4: Generating charts...")
                try:
                    chart_paths = self.chart_generator.generate_charts(
                        report.trend_graph_data,
                        report
                    )
                    logger.info(f"  Generated {len(chart_paths)} chart(s)")
                except ChartGenerationError as e:
                    logger.warning(f"  Chart generation failed: {e}")

            # Step 5: Write output
            if self.config.output_path:
                logger.info("\nStep 5: Writing report...")
                self._write_report(report)
                logger.info(f"  Wrote report to {self.config.output_path}")

            exit_code = EXIT_TREND_SUCCESS

            logger.info("\n" + "=" * 80)
            logger.info("TREND ANALYSIS COMPLETE")
            logger.info("=" * 80)
            logger.info(f"Analysis Duration: {duration:.1f}ms")
            logger.info(f"Exit Code: {exit_code}")
            logger.info("=" * 80)

            return report, exit_code

        except InsufficientHistoryError as e:
            logger.error(f"Insufficient history: {e}")
            return self._error_report(str(e)), EXIT_INSUFFICIENT_HISTORY
        except InvalidSnapshotError as e:
            logger.error(f"Invalid snapshot: {e}")
            return self._error_report(str(e)), EXIT_INVALID_SNAPSHOT
        except ComputationError as e:
            logger.error(f"Computation error: {e}")
            return self._error_report(str(e)), EXIT_COMPUTATION_ERROR
        except PredictionError as e:
            logger.error(f"Prediction error: {e}")
            return self._error_report(str(e)), EXIT_PREDICTION_ERROR
        except HistoryStoreError as e:
            logger.error(f"History store error: {e}")
            return self._error_report(str(e)), EXIT_HISTORY_STORE_ERROR
        except ChartGenerationError as e:
            logger.error(f"Chart generation error: {e}")
            return self._error_report(str(e)), EXIT_CHART_GENERATION_ERROR
        except Exception as e:
            logger.error(f"Trend analysis error: {e}")
            return self._error_report(str(e)), EXIT_GENERAL_TREND_ERROR

    def _write_report(self, report: TrendReport) -> None:
        """Write trend report to JSON file."""
        try:
            output_path = Path(self.config.output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report.to_dict(), f, indent=2, default=str)

        except Exception as e:
            raise HistoryStoreError(f"Failed to write report: {e}")

    def _error_report(self, error_message: str) -> TrendReport:
        """Generate an error report."""
        return TrendReport(
            report_id=f"trend_report_error_{datetime.utcnow().isoformat()}",
            generated_at=datetime.utcnow().isoformat(),
            history_dir=str(self.config.history_dir),
            snapshots_analyzed=0,
            first_snapshot="N/A",
            last_snapshot="N/A",
            analysis_window_days=0,
            overall_trend="unknown",
            trend_confidence=0.0,
            early_warnings=[
                EarlyWarning(
                    warning_id=f"error_{datetime.utcnow().isoformat()}",
                    warning_type="analysis_error",
                    level=WarningLevel.CRITICAL.value,
                    timestamp=datetime.utcnow(),
                    title="Analysis Error",
                    message=error_message,
                    affected_metric="all",
                    current_value=0.0,
                    threshold_value=0.0,
                    trend_direction="unknown",
                    estimated_time_to_threshold=None
                )
            ]
        )


# ============================================================================
# Utility Functions
# ============================================================================

def add_snapshot_to_history(
    history_dir: Path,
    dashboard_path: Path,
    version: Optional[str] = None
) -> SnapshotMetadata:
    """Convenience function to add a snapshot to history."""
    store = HealthHistoryStore(history_dir)
    store.initialize()
    return store.add_snapshot(dashboard_path, version)


def get_trend_summary(history_dir: Path) -> Dict[str, Any]:
    """Get a quick trend summary without full analysis."""
    store = HealthHistoryStore(history_dir)
    store.initialize()

    snapshots = store.get_all_snapshots()

    if len(snapshots) < 2:
        return {
            "status": "insufficient_data",
            "snapshots": len(snapshots),
            "message": "Need at least 2 snapshots for trend summary"
        }

    first = snapshots[0]
    last = snapshots[-1]

    score_change = last.repository_score - first.repository_score

    if score_change > 5:
        trend = "improving"
    elif score_change < -5:
        trend = "declining"
    else:
        trend = "stable"

    return {
        "status": "ok",
        "snapshots": len(snapshots),
        "trend": trend,
        "first_score": first.repository_score,
        "last_score": last.repository_score,
        "score_change": score_change,
        "first_date": first.timestamp.isoformat(),
        "last_date": last.timestamp.isoformat()
    }
