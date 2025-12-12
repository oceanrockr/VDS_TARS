#!/usr/bin/env python3
"""
Regression Analyzer - Post-GA Weekly Regression Detection

Analyzes regression from multiple baselines:
- GA Day baseline (ga_kpi_summary.json)
- 7-Day KPIs (stability data)
- Staging metrics (pre-GA)
- v1.0.0 baseline (historical stub)

Detects performance, resource, and availability regressions with severity classification.

Usage:
    python regression_analyzer.py --ga-baseline ga_kpis/ga_kpi_summary.json --7day-data stability/
    python regression_analyzer.py --ga-baseline ga_kpis/ga_kpi_summary.json --7day-data stability/ --staging-baseline staging_baseline.json

Author: T.A.R.S. Platform Team
Phase: 14.6 - Post-GA 7-Day Stabilization & Retrospective
"""

import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Enterprise imports (Phase 14.6)
try:
    from enterprise_config import load_config
    from compliance.enforcer import ComplianceEnforcer
    from security.encryption import AESEncryption
    from security.signing import ReportSigner
    from telemetry import get_logger
    ENTERPRISE_AVAILABLE = True
except ImportError:
    ENTERPRISE_AVAILABLE = False
    print("Warning: Enterprise features not available. Running in legacy mode.")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class BaselineMetrics:
    """
    Baseline metrics from a specific source.
    """
    source: str  # "ga_day", "staging", "v1.0.0", "7day_avg"
    timestamp: str

    # Core Metrics
    availability: float
    error_rate: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float

    # Resource Metrics
    avg_cpu_percent: float
    avg_memory_percent: float
    peak_cpu_percent: float
    peak_memory_percent: float

    # Infrastructure
    db_p95_latency_ms: float
    redis_hit_rate: float
    cluster_cpu_utilization: float
    cluster_memory_utilization: float

    # Cost
    estimated_cost_per_hour: Optional[float] = None


@dataclass
class RegressionEvent:
    """
    A detected regression event.
    """
    metric_name: str
    baseline_source: str  # Which baseline was violated
    baseline_value: float
    current_value: float
    regression_percent: float
    severity: str  # "critical", "high", "medium", "low"
    category: str  # "performance", "resource", "availability", "cost"
    impact: str  # Human-readable impact description
    first_detected: str  # Timestamp
    mitigation_priority: str  # "P0", "P1", "P2"
    recommended_actions: List[str] = field(default_factory=list)


@dataclass
class RegressionSummary:
    """
    Overall regression analysis summary.
    """
    analysis_timestamp: str
    ga_day_timestamp: str
    seven_day_end_timestamp: str

    # Baseline Sources
    baselines_analyzed: List[str] = field(default_factory=list)

    # Regression Counts
    total_regressions: int = 0
    critical_regressions: int = 0
    high_regressions: int = 0
    medium_regressions: int = 0
    low_regressions: int = 0

    # Category Breakdown
    performance_regressions: int = 0
    resource_regressions: int = 0
    availability_regressions: int = 0
    cost_regressions: int = 0

    # Detailed Regressions
    regressions: List[RegressionEvent] = field(default_factory=list)

    # Comparison Summary
    vs_ga_day: Dict[str, float] = field(default_factory=dict)  # % changes
    vs_staging: Dict[str, float] = field(default_factory=dict)
    vs_v1_0_0: Dict[str, float] = field(default_factory=dict)

    # Recommendations
    overall_assessment: str  # "stable", "minor_regression", "major_regression", "critical"
    recommendations: List[str] = field(default_factory=list)
    rollback_recommended: bool = False
    rollback_reason: Optional[str] = None


class RegressionDetector:
    """
    Regression detection engine with configurable rules.
    """

    # Regression thresholds by metric type
    THRESHOLDS = {
        "availability": {"critical": -1.0, "high": -0.5, "medium": -0.2},  # % decrease
        "error_rate": {"critical": 100.0, "high": 50.0, "medium": 20.0},  # % increase
        "latency": {"critical": 50.0, "high": 30.0, "medium": 15.0},  # % increase
        "cpu": {"critical": 50.0, "high": 30.0, "medium": 20.0},  # % increase
        "memory": {"critical": 50.0, "high": 30.0, "medium": 20.0},  # % increase
        "cost": {"critical": 30.0, "high": 20.0, "medium": 10.0},  # % increase
    }

    def __init__(self):
        """
        Initialize regression detector.
        """
        logger.info("RegressionDetector initialized with standard thresholds")

    def detect_regression(
        self,
        metric_name: str,
        baseline_value: float,
        current_value: float,
        baseline_source: str,
        metric_type: str
    ) -> Optional[RegressionEvent]:
        """
        Detect if a regression occurred for a specific metric.

        Args:
            metric_name: Name of the metric
            baseline_value: Baseline value
            current_value: Current value
            baseline_source: Source of baseline
            metric_type: Type of metric (for threshold lookup)

        Returns:
            RegressionEvent if regression detected, None otherwise
        """
        # Calculate regression percentage
        regression_percent = self.calculate_regression_percent(
            baseline_value, current_value, metric_type
        )

        # Classify severity
        severity = self.classify_severity(regression_percent, metric_type)

        # Only report if there's a meaningful regression
        if severity == "none" or abs(regression_percent) < 1.0:
            return None

        # Determine category
        category = "performance"
        if "cpu" in metric_name.lower() or "memory" in metric_name.lower():
            category = "resource"
        elif "availability" in metric_name.lower():
            category = "availability"
        elif "cost" in metric_name.lower():
            category = "cost"

        # Generate impact description
        direction = "increased" if regression_percent > 0 else "decreased"
        if metric_type == "availability":
            direction = "decreased" if regression_percent > 0 else "increased"

        impact = f"{metric_name} {direction} by {abs(regression_percent):.1f}% from {baseline_value:.2f} to {current_value:.2f}"

        # Determine mitigation priority
        priority_map = {"critical": "P0", "high": "P1", "medium": "P2", "low": "P2"}
        mitigation_priority = priority_map.get(severity, "P2")

        # Get mitigation actions
        recommended_actions = self.suggest_mitigation_actions(
            metric_name, severity, category
        )

        return RegressionEvent(
            metric_name=metric_name,
            baseline_source=baseline_source,
            baseline_value=baseline_value,
            current_value=current_value,
            regression_percent=regression_percent,
            severity=severity,
            category=category,
            impact=impact,
            first_detected=datetime.now(timezone.utc).isoformat(),
            mitigation_priority=mitigation_priority,
            recommended_actions=recommended_actions
        )

    def calculate_regression_percent(
        self,
        baseline: float,
        current: float,
        metric_type: str
    ) -> float:
        """
        Calculate regression percentage.

        For "lower is better" metrics (error_rate, latency, cpu, memory, cost):
            positive % = regression (increase)
        For "higher is better" metrics (availability, hit_rate):
            negative % = regression (decrease)

        Args:
            baseline: Baseline value
            current: Current value
            metric_type: Type of metric

        Returns:
            Regression percentage (positive = regression)
        """
        if baseline == 0:
            # Handle division by zero
            if current == 0:
                return 0.0
            # If baseline is 0 but current is not, this is a significant change
            return 100.0 if current > 0 else -100.0

        # Calculate percentage change
        percent_change = ((current - baseline) / baseline) * 100.0

        # For "higher is better" metrics, invert the sign
        # so that decrease = positive regression
        if metric_type in ["availability"]:
            percent_change = -percent_change

        return percent_change

    def classify_severity(self, regression_percent: float, metric_type: str) -> str:
        """
        Classify regression severity.

        Args:
            regression_percent: Regression percentage
            metric_type: Type of metric

        Returns:
            Severity: "critical", "high", "medium", "low", "none"
        """
        # Get thresholds for this metric type
        thresholds = self.THRESHOLDS.get(metric_type, self.THRESHOLDS["latency"])

        # Take absolute value for comparison
        abs_regression = abs(regression_percent)

        # Classify based on thresholds (in descending order)
        if abs_regression >= thresholds["critical"]:
            return "critical"
        elif abs_regression >= thresholds["high"]:
            return "high"
        elif abs_regression >= thresholds["medium"]:
            return "medium"
        elif abs_regression >= 1.0:  # Any regression > 1%
            return "low"
        else:
            return "none"

    def suggest_mitigation_actions(
        self,
        metric_name: str,
        severity: str,
        category: str
    ) -> List[str]:
        """
        Suggest mitigation actions for a regression.

        Args:
            metric_name: Name of metric
            severity: Regression severity
            category: Regression category

        Returns:
            List of recommended actions
        """
        actions = []

        # Category-specific actions
        if category == "performance":
            if "latency" in metric_name.lower():
                actions.extend([
                    "Analyze slow query logs and optimize database queries",
                    "Review application profiling data for hot paths",
                    "Consider implementing or tuning response caching",
                    "Check for network latency issues between services"
                ])
            if "error" in metric_name.lower():
                actions.extend([
                    "Review application logs for error patterns",
                    "Check for newly introduced bugs in recent deployments",
                    "Verify external service dependencies are healthy"
                ])

        elif category == "resource":
            if "cpu" in metric_name.lower():
                actions.extend([
                    "Analyze CPU profiling data to identify hotspots",
                    "Review recent code changes for inefficient algorithms",
                    "Consider horizontal pod autoscaling (HPA) adjustments",
                    "Optimize CPU-intensive operations or move to background jobs"
                ])
            if "memory" in metric_name.lower():
                actions.extend([
                    "Check for memory leaks using heap dumps",
                    "Review object lifecycle and garbage collection logs",
                    "Optimize data structures and reduce memory footprint",
                    "Consider increasing pod memory limits if sustained"
                ])

        elif category == "availability":
            actions.extend([
                "Review incident reports for root causes",
                "Check for deployment-related issues",
                "Verify health check configurations",
                "Review recent infrastructure changes",
                "Consider implementing circuit breakers or retry logic"
            ])

        elif category == "cost":
            actions.extend([
                "Review resource allocation and right-size pods",
                "Analyze autoscaling policies for over-provisioning",
                "Optimize database and cache usage patterns",
                "Consider reserved instances or committed use discounts"
            ])

        # Severity-specific urgency
        if severity == "critical":
            actions.insert(0, "URGENT: Escalate to on-call engineer immediately")
            actions.insert(1, "Consider rollback to previous stable version")
        elif severity == "high":
            actions.insert(0, "High priority: Address within 24 hours")

        return actions[:5]  # Return top 5 most relevant actions


class RegressionAnalyzer:
    """
    Main regression analysis engine.
    """

    def __init__(
        self,
        ga_baseline_file: str,
        seven_day_data_dir: str,
        staging_baseline_file: Optional[str] = None,
        v1_0_0_baseline_file: Optional[str] = None,
        output_dir: str = ".",
        # Enterprise features
        compliance_enforcer: Optional[ComplianceEnforcer] = None,
        encryptor: Optional[AESEncryption] = None,
        signer: Optional[ReportSigner] = None,
    ):
        """
        Initialize regression analyzer.

        Args:
            ga_baseline_file: Path to GA Day baseline JSON
            seven_day_data_dir: Path to 7-day stability data directory
            staging_baseline_file: (Optional) Path to staging baseline JSON
            v1_0_0_baseline_file: (Optional) Path to v1.0.0 baseline JSON
            output_dir: Output directory for analysis results
            compliance_enforcer: Optional compliance enforcer (enterprise)
            encryptor: Optional AES encryptor (enterprise)
            signer: Optional RSA signer (enterprise)
        """
        self.ga_baseline_file = ga_baseline_file
        self.seven_day_data_dir = Path(seven_day_data_dir)
        self.staging_baseline_file = staging_baseline_file
        self.v1_0_0_baseline_file = v1_0_0_baseline_file
        self.output_dir = Path(output_dir)

        self.detector = RegressionDetector()

        self.baselines: Dict[str, BaselineMetrics] = {}
        self.seven_day_metrics: Optional[BaselineMetrics] = None

        # Enterprise features (Phase 14.6)
        self.compliance_enforcer = compliance_enforcer
        self.encryptor = encryptor
        self.signer = signer

        logger.info("RegressionAnalyzer initialized")

    def load_ga_baseline(self) -> BaselineMetrics:
        """
        Load GA Day baseline metrics.

        Returns:
            BaselineMetrics from GA Day
        """
        logger.info(f"Loading GA baseline from {self.ga_baseline_file}")

        with open(self.ga_baseline_file, 'r') as f:
            data = json.load(f)

        # Parse GA KPI summary format
        baseline = BaselineMetrics(
            source="ga_day",
            timestamp=data.get("certification_timestamp", data.get("ga_end", "")),
            availability=data.get("overall_availability", 0.0),
            error_rate=data.get("overall_error_rate", 0.0),
            p50_latency_ms=data.get("avg_p50_latency_ms", 0.0),
            p95_latency_ms=data.get("avg_p95_latency_ms", 0.0),
            p99_latency_ms=data.get("avg_p99_latency_ms", 0.0),
            avg_cpu_percent=data.get("avg_cpu_percent", 0.0),
            avg_memory_percent=data.get("avg_memory_percent", 0.0),
            peak_cpu_percent=data.get("peak_cpu_percent", 0.0),
            peak_memory_percent=data.get("peak_memory_percent", 0.0),
            db_p95_latency_ms=data.get("avg_db_latency_ms", 0.0),
            redis_hit_rate=data.get("avg_redis_hit_rate", 0.0),
            cluster_cpu_utilization=data.get("avg_cpu_percent", 0.0),
            cluster_memory_utilization=data.get("avg_memory_percent", 0.0),
            estimated_cost_per_hour=data.get("estimated_cost_per_hour")
        )

        logger.info(f"GA baseline loaded: availability={baseline.availability}%, error_rate={baseline.error_rate}%")
        self.baselines["ga_day"] = baseline
        return baseline

    def load_staging_baseline(self) -> Optional[BaselineMetrics]:
        """
        Load staging baseline metrics (if available).

        Returns:
            BaselineMetrics from staging, or None if not available
        """
        if not self.staging_baseline_file:
            logger.info("No staging baseline file provided")
            return None

        if not Path(self.staging_baseline_file).exists():
            logger.warning(f"Staging baseline file not found: {self.staging_baseline_file}")
            return None

        logger.info(f"Loading staging baseline from {self.staging_baseline_file}")

        with open(self.staging_baseline_file, 'r') as f:
            data = json.load(f)

        # Parse staging baseline (assumed same format as GA baseline)
        baseline = BaselineMetrics(
            source="staging",
            timestamp=data.get("certification_timestamp", data.get("ga_end", "")),
            availability=data.get("overall_availability", 0.0),
            error_rate=data.get("overall_error_rate", 0.0),
            p50_latency_ms=data.get("avg_p50_latency_ms", 0.0),
            p95_latency_ms=data.get("avg_p95_latency_ms", 0.0),
            p99_latency_ms=data.get("avg_p99_latency_ms", 0.0),
            avg_cpu_percent=data.get("avg_cpu_percent", 0.0),
            avg_memory_percent=data.get("avg_memory_percent", 0.0),
            peak_cpu_percent=data.get("peak_cpu_percent", 0.0),
            peak_memory_percent=data.get("peak_memory_percent", 0.0),
            db_p95_latency_ms=data.get("avg_db_latency_ms", 0.0),
            redis_hit_rate=data.get("avg_redis_hit_rate", 0.0),
            cluster_cpu_utilization=data.get("avg_cpu_percent", 0.0),
            cluster_memory_utilization=data.get("avg_memory_percent", 0.0),
            estimated_cost_per_hour=data.get("estimated_cost_per_hour")
        )

        logger.info(f"Staging baseline loaded: availability={baseline.availability}%, error_rate={baseline.error_rate}%")
        self.baselines["staging"] = baseline
        return baseline

    def load_v1_0_0_baseline(self) -> Optional[BaselineMetrics]:
        """
        Load v1.0.0 baseline metrics (historical stub).

        Returns:
            BaselineMetrics from v1.0.0, or None if not available
        """
        if not self.v1_0_0_baseline_file:
            logger.info("No v1.0.0 baseline file provided (stub)")
            return None

        if not Path(self.v1_0_0_baseline_file).exists():
            logger.warning(f"v1.0.0 baseline file not found: {self.v1_0_0_baseline_file}")
            return None

        logger.info(f"Loading v1.0.0 baseline from {self.v1_0_0_baseline_file}")

        with open(self.v1_0_0_baseline_file, 'r') as f:
            data = json.load(f)

        # Parse v1.0.0 baseline (assumed same format as GA baseline)
        baseline = BaselineMetrics(
            source="v1.0.0",
            timestamp=data.get("certification_timestamp", data.get("ga_end", "")),
            availability=data.get("overall_availability", 0.0),
            error_rate=data.get("overall_error_rate", 0.0),
            p50_latency_ms=data.get("avg_p50_latency_ms", 0.0),
            p95_latency_ms=data.get("avg_p95_latency_ms", 0.0),
            p99_latency_ms=data.get("avg_p99_latency_ms", 0.0),
            avg_cpu_percent=data.get("avg_cpu_percent", 0.0),
            avg_memory_percent=data.get("avg_memory_percent", 0.0),
            peak_cpu_percent=data.get("peak_cpu_percent", 0.0),
            peak_memory_percent=data.get("peak_memory_percent", 0.0),
            db_p95_latency_ms=data.get("avg_db_latency_ms", 0.0),
            redis_hit_rate=data.get("avg_redis_hit_rate", 0.0),
            cluster_cpu_utilization=data.get("avg_cpu_percent", 0.0),
            cluster_memory_utilization=data.get("avg_memory_percent", 0.0),
            estimated_cost_per_hour=data.get("estimated_cost_per_hour")
        )

        logger.info(f"v1.0.0 baseline loaded: availability={baseline.availability}%, error_rate={baseline.error_rate}%")
        self.baselines["v1.0.0"] = baseline
        return baseline

    def calculate_7day_average(self) -> BaselineMetrics:
        """
        Calculate average metrics from 7-day stability data.

        Returns:
            BaselineMetrics with 7-day averages
        """
        logger.info(f"Calculating 7-day average from {self.seven_day_data_dir}")

        # Load all daily summary files
        daily_summaries = []
        for day_num in range(1, 8):
            summary_file = self.seven_day_data_dir / f"day_{day_num:02d}_summary.json"
            if summary_file.exists():
                with open(summary_file, 'r') as f:
                    daily_summaries.append(json.load(f))
            else:
                logger.warning(f"Daily summary not found: {summary_file}")

        if not daily_summaries:
            raise ValueError(f"No daily summaries found in {self.seven_day_data_dir}")

        logger.info(f"Loaded {len(daily_summaries)} daily summaries")

        # Calculate averages
        def avg(field: str) -> float:
            values = [d.get(field, 0.0) for d in daily_summaries if field in d]
            return sum(values) / len(values) if values else 0.0

        def max_val(field: str) -> float:
            values = [d.get(field, 0.0) for d in daily_summaries if field in d]
            return max(values) if values else 0.0

        baseline = BaselineMetrics(
            source="7day_avg",
            timestamp=datetime.now(timezone.utc).isoformat(),
            availability=avg("avg_availability"),
            error_rate=avg("avg_error_rate"),
            p50_latency_ms=avg("avg_p50_latency_ms"),
            p95_latency_ms=avg("avg_p95_latency_ms"),
            p99_latency_ms=avg("avg_p99_latency_ms"),
            avg_cpu_percent=avg("avg_cpu_percent"),
            avg_memory_percent=avg("avg_memory_percent"),
            peak_cpu_percent=max_val("peak_cpu_percent"),
            peak_memory_percent=max_val("peak_memory_percent"),
            db_p95_latency_ms=avg("avg_db_p95_latency_ms"),
            redis_hit_rate=avg("avg_redis_hit_rate"),
            cluster_cpu_utilization=avg("avg_cluster_cpu_utilization"),
            cluster_memory_utilization=avg("avg_cluster_memory_utilization"),
            estimated_cost_per_hour=avg("avg_cost_per_hour") if any("avg_cost_per_hour" in d for d in daily_summaries) else None
        )

        logger.info(f"7-day average calculated: availability={baseline.availability:.2f}%, error_rate={baseline.error_rate:.4f}%")
        self.seven_day_metrics = baseline
        return baseline

    def compare_to_baseline(
        self,
        baseline: BaselineMetrics,
        current: BaselineMetrics
    ) -> List[RegressionEvent]:
        """
        Compare current metrics to a baseline.

        Args:
            baseline: Baseline metrics
            current: Current metrics

        Returns:
            List of detected regressions
        """
        regressions = []

        # Define metric comparisons: (metric_name, baseline_val, current_val, metric_type)
        comparisons = [
            ("availability", baseline.availability, current.availability, "availability"),
            ("error_rate", baseline.error_rate, current.error_rate, "error_rate"),
            ("p50_latency_ms", baseline.p50_latency_ms, current.p50_latency_ms, "latency"),
            ("p95_latency_ms", baseline.p95_latency_ms, current.p95_latency_ms, "latency"),
            ("p99_latency_ms", baseline.p99_latency_ms, current.p99_latency_ms, "latency"),
            ("avg_cpu_percent", baseline.avg_cpu_percent, current.avg_cpu_percent, "cpu"),
            ("avg_memory_percent", baseline.avg_memory_percent, current.avg_memory_percent, "memory"),
            ("peak_cpu_percent", baseline.peak_cpu_percent, current.peak_cpu_percent, "cpu"),
            ("peak_memory_percent", baseline.peak_memory_percent, current.peak_memory_percent, "memory"),
            ("db_p95_latency_ms", baseline.db_p95_latency_ms, current.db_p95_latency_ms, "latency"),
            ("redis_hit_rate", baseline.redis_hit_rate, current.redis_hit_rate, "availability"),
            ("cluster_cpu_utilization", baseline.cluster_cpu_utilization, current.cluster_cpu_utilization, "cpu"),
            ("cluster_memory_utilization", baseline.cluster_memory_utilization, current.cluster_memory_utilization, "memory"),
        ]

        # Add cost comparison if available
        if baseline.estimated_cost_per_hour and current.estimated_cost_per_hour:
            comparisons.append((
                "estimated_cost_per_hour",
                baseline.estimated_cost_per_hour,
                current.estimated_cost_per_hour,
                "cost"
            ))

        # Detect regressions for each metric
        for metric_name, baseline_val, current_val, metric_type in comparisons:
            regression = self.detector.detect_regression(
                metric_name=metric_name,
                baseline_value=baseline_val,
                current_value=current_val,
                baseline_source=baseline.source,
                metric_type=metric_type
            )
            if regression:
                regressions.append(regression)

        logger.info(f"Comparison vs {baseline.source}: {len(regressions)} regressions detected")
        return regressions

    def analyze_all_baselines(self) -> List[RegressionEvent]:
        """
        Analyze regressions against all available baselines.

        Returns:
            Combined list of all regressions detected
        """
        all_regressions = []

        # Load GA baseline (required)
        ga_baseline = self.load_ga_baseline()

        # Calculate 7-day average (required)
        seven_day_avg = self.calculate_7day_average()

        # Compare 7-day average to GA baseline
        ga_regressions = self.compare_to_baseline(ga_baseline, seven_day_avg)
        all_regressions.extend(ga_regressions)

        # Load and compare staging baseline (optional)
        staging_baseline = self.load_staging_baseline()
        if staging_baseline:
            staging_regressions = self.compare_to_baseline(staging_baseline, seven_day_avg)
            all_regressions.extend(staging_regressions)

        # Load and compare v1.0.0 baseline (optional)
        v1_0_0_baseline = self.load_v1_0_0_baseline()
        if v1_0_0_baseline:
            v1_0_0_regressions = self.compare_to_baseline(v1_0_0_baseline, seven_day_avg)
            all_regressions.extend(v1_0_0_regressions)

        # Sort by severity (critical first)
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        all_regressions.sort(key=lambda r: (severity_order.get(r.severity, 4), abs(r.regression_percent)), reverse=True)

        logger.info(f"Total regressions detected across all baselines: {len(all_regressions)}")
        return all_regressions

    def determine_overall_assessment(
        self,
        regressions: List[RegressionEvent]
    ) -> str:
        """
        Determine overall regression assessment.

        Args:
            regressions: List of all regressions

        Returns:
            Assessment: "stable", "minor_regression", "major_regression", "critical"
        """
        if not regressions:
            return "stable"

        # Count by severity
        critical_count = sum(1 for r in regressions if r.severity == "critical")
        high_count = sum(1 for r in regressions if r.severity == "high")
        medium_count = sum(1 for r in regressions if r.severity == "medium")

        # Determine overall assessment based on counts
        if critical_count >= 3:
            return "critical"
        elif critical_count >= 1:
            return "major_regression"
        elif high_count >= 5:
            return "major_regression"
        elif high_count >= 2:
            return "minor_regression"
        elif medium_count >= 5:
            return "minor_regression"
        else:
            return "stable"

    def should_rollback(self, regressions: List[RegressionEvent]) -> tuple[bool, Optional[str]]:
        """
        Determine if rollback is recommended.

        Rollback criteria:
        - Any critical availability regression
        - 3+ critical regressions
        - 5+ high regressions
        - Any P50 latency regression >100%
        - Any availability regression > 10%
        - Any latency regression > 25%

        Args:
            regressions: List of regressions

        Returns:
            Tuple of (should_rollback, reason)
        """
        if not regressions:
            return False, None

        # Check for critical availability regressions
        for r in regressions:
            if r.severity == "critical" and "availability" in r.metric_name.lower():
                return True, f"Critical availability regression: {r.impact}"

        # Check for severe latency regressions
        for r in regressions:
            if "p50" in r.metric_name.lower() and abs(r.regression_percent) > 100:
                return True, f"P50 latency doubled: {r.impact}"
            if "latency" in r.metric_name.lower() and abs(r.regression_percent) > 25:
                return True, f"Severe latency regression: {r.impact}"

        # Check for severe availability degradation
        for r in regressions:
            if "availability" in r.metric_name.lower() and abs(r.regression_percent) > 10:
                return True, f"Severe availability degradation: {r.impact}"

        # Count by severity
        critical_count = sum(1 for r in regressions if r.severity == "critical")
        high_count = sum(1 for r in regressions if r.severity == "high")

        if critical_count >= 3:
            return True, f"Too many critical regressions ({critical_count})"

        if high_count >= 5:
            return True, f"Too many high severity regressions ({high_count})"

        return False, None

    def generate_recommendations(
        self,
        regressions: List[RegressionEvent],
        overall_assessment: str
    ) -> List[str]:
        """
        Generate high-level recommendations.

        Args:
            regressions: List of regressions
            overall_assessment: Overall assessment

        Returns:
            List of recommendations
        """
        recommendations = []

        # Assessment-based recommendations
        if overall_assessment == "critical":
            recommendations.append("URGENT: Immediate action required - consider emergency rollback")
            recommendations.append("Escalate to incident commander for assessment")
        elif overall_assessment == "major_regression":
            recommendations.append("High priority: Schedule immediate investigation and remediation")
            recommendations.append("Prepare rollback plan as contingency")
        elif overall_assessment == "minor_regression":
            recommendations.append("Monitor closely and plan remediation in next sprint")
        else:
            recommendations.append("System is stable - continue monitoring")

        # Category-specific patterns
        perf_regs = [r for r in regressions if r.category == "performance"]
        resource_regs = [r for r in regressions if r.category == "resource"]
        avail_regs = [r for r in regressions if r.category == "availability"]

        if len(perf_regs) >= 3:
            recommendations.append("Performance degradation detected across multiple metrics - review recent code changes")

        if len(resource_regs) >= 3:
            recommendations.append("Resource utilization trending upward - consider capacity planning")

        if avail_regs:
            recommendations.append("Availability impacted - review incident logs and implement reliability improvements")

        # P0/P1 action items
        p0_count = sum(1 for r in regressions if r.mitigation_priority == "P0")
        p1_count = sum(1 for r in regressions if r.mitigation_priority == "P1")

        if p0_count > 0:
            recommendations.append(f"{p0_count} P0 action items require immediate attention")
        if p1_count > 0:
            recommendations.append(f"{p1_count} P1 action items should be addressed within 24 hours")

        return recommendations[:8]  # Return top 8 recommendations

    def analyze(self) -> RegressionSummary:
        """
        Run complete regression analysis.

        Returns:
            RegressionSummary with all analysis results
        """
        logger.info("Starting regression analysis")

        # 1. Detect regressions against all baselines
        regressions = self.analyze_all_baselines()

        # 2. Count by severity
        critical_count = sum(1 for r in regressions if r.severity == "critical")
        high_count = sum(1 for r in regressions if r.severity == "high")
        medium_count = sum(1 for r in regressions if r.severity == "medium")
        low_count = sum(1 for r in regressions if r.severity == "low")

        # 3. Count by category
        perf_count = sum(1 for r in regressions if r.category == "performance")
        resource_count = sum(1 for r in regressions if r.category == "resource")
        avail_count = sum(1 for r in regressions if r.category == "availability")
        cost_count = sum(1 for r in regressions if r.category == "cost")

        # 4. Build comparison summaries (% changes by baseline)
        vs_ga_day = {}
        vs_staging = {}
        vs_v1_0_0 = {}

        for r in regressions:
            if r.baseline_source == "ga_day":
                vs_ga_day[r.metric_name] = r.regression_percent
            elif r.baseline_source == "staging":
                vs_staging[r.metric_name] = r.regression_percent
            elif r.baseline_source == "v1.0.0":
                vs_v1_0_0[r.metric_name] = r.regression_percent

        # 5. Determine overall assessment
        overall_assessment = self.determine_overall_assessment(regressions)

        # 6. Check rollback recommendation
        should_rollback, rollback_reason = self.should_rollback(regressions)

        # 7. Generate recommendations
        recommendations = self.generate_recommendations(regressions, overall_assessment)

        # 8. Construct summary
        summary = RegressionSummary(
            analysis_timestamp=datetime.now(timezone.utc).isoformat(),
            ga_day_timestamp=self.baselines.get("ga_day").timestamp if "ga_day" in self.baselines else "",
            seven_day_end_timestamp=self.seven_day_metrics.timestamp if self.seven_day_metrics else "",
            baselines_analyzed=[b for b in self.baselines.keys()],
            total_regressions=len(regressions),
            critical_regressions=critical_count,
            high_regressions=high_count,
            medium_regressions=medium_count,
            low_regressions=low_count,
            performance_regressions=perf_count,
            resource_regressions=resource_count,
            availability_regressions=avail_count,
            cost_regressions=cost_count,
            regressions=regressions,
            vs_ga_day=vs_ga_day,
            vs_staging=vs_staging,
            vs_v1_0_0=vs_v1_0_0,
            overall_assessment=overall_assessment,
            recommendations=recommendations,
            rollback_recommended=should_rollback,
            rollback_reason=rollback_reason
        )

        logger.info(f"Analysis complete: {overall_assessment}, {len(regressions)} regressions, rollback={should_rollback}")
        return summary

    def save_summary_json(self, summary: RegressionSummary) -> str:
        """
        Save regression summary as JSON.

        Args:
            summary: Regression summary

        Returns:
            Path to saved JSON file
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        json_file = self.output_dir / "regression_summary.json"

        # Convert to dict with proper serialization
        summary_dict = asdict(summary)

        with open(json_file, 'w') as f:
            json.dump(summary_dict, f, indent=2)

        # Enterprise: Encrypt if enabled
        if self.encryptor:
            encrypted_file = json_file.with_suffix(".json.enc")
            self.encryptor.encrypt_file(json_file, encrypted_file)
            logger.info(f"Encrypted regression summary: {encrypted_file}")

        # Enterprise: Sign if enabled
        if self.signer:
            signature = self.signer.sign_file(json_file)
            sig_file = json_file.with_suffix(".json.sig")
            with open(sig_file, "w") as f:
                f.write(f"RSA-PSS-SHA256\n{signature}\n")
            logger.info(f"Signed regression summary: {sig_file}")

        logger.info(f"JSON summary saved: {json_file}")
        return str(json_file)

    def save_summary_markdown(self, summary: RegressionSummary) -> str:
        """
        Save regression summary as Markdown.

        Args:
            summary: Regression summary

        Returns:
            Path to saved Markdown file
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        md_file = self.output_dir / "regression_summary.md"

        with open(md_file, 'w') as f:
            # Header
            f.write("# T.A.R.S. v1.0.1 - 7-Day Regression Analysis\n\n")
            f.write(f"**Analysis Timestamp:** {summary.analysis_timestamp}\n\n")
            f.write(f"**GA Day:** {summary.ga_day_timestamp}\n\n")
            f.write(f"**7-Day Period End:** {summary.seven_day_end_timestamp}\n\n")
            f.write(f"**Baselines Analyzed:** {', '.join(summary.baselines_analyzed)}\n\n")
            f.write("---\n\n")

            # Executive Summary
            f.write("## Executive Summary\n\n")

            status_emoji = {
                "stable": "✅",
                "minor_regression": "⚠️",
                "major_regression": "❌",
                "critical": "🚨"
            }
            emoji = status_emoji.get(summary.overall_assessment, "❓")

            f.write(f"**Overall Assessment:** {emoji} {summary.overall_assessment.upper().replace('_', ' ')}\n\n")
            f.write(f"**Total Regressions Detected:** {summary.total_regressions}\n\n")
            f.write(f"**Rollback Recommended:** {'🚨 YES' if summary.rollback_recommended else '✅ NO'}\n\n")

            if summary.rollback_reason:
                f.write(f"**Rollback Reason:** {summary.rollback_reason}\n\n")

            # Regression Summary Table
            f.write("### Regression Severity Breakdown\n\n")
            f.write("| Severity | Count | Percentage |\n")
            f.write("|----------|-------|------------|\n")

            total = summary.total_regressions if summary.total_regressions > 0 else 1
            f.write(f"| 🚨 Critical | {summary.critical_regressions} | {summary.critical_regressions/total*100:.1f}% |\n")
            f.write(f"| ❌ High | {summary.high_regressions} | {summary.high_regressions/total*100:.1f}% |\n")
            f.write(f"| ⚠️ Medium | {summary.medium_regressions} | {summary.medium_regressions/total*100:.1f}% |\n")
            f.write(f"| ℹ️ Low | {summary.low_regressions} | {summary.low_regressions/total*100:.1f}% |\n")
            f.write("\n")

            # Category Breakdown
            f.write("### Regression Category Breakdown\n\n")
            f.write("| Category | Count |\n")
            f.write("|----------|-------|\n")
            f.write(f"| Performance | {summary.performance_regressions} |\n")
            f.write(f"| Resource | {summary.resource_regressions} |\n")
            f.write(f"| Availability | {summary.availability_regressions} |\n")
            f.write(f"| Cost | {summary.cost_regressions} |\n")
            f.write("\n")

            # Baseline Comparisons
            if summary.vs_ga_day:
                f.write("## Comparison vs GA Day Baseline\n\n")
                f.write("| Metric | Change (%) |\n")
                f.write("|--------|------------|\n")
                for metric, change in sorted(summary.vs_ga_day.items(), key=lambda x: abs(x[1]), reverse=True)[:10]:
                    arrow = "📈" if change > 0 else "📉"
                    f.write(f"| {metric} | {arrow} {change:+.2f}% |\n")
                f.write("\n")

            if summary.vs_staging:
                f.write("## Comparison vs Staging Baseline\n\n")
                f.write("| Metric | Change (%) |\n")
                f.write("|--------|------------|\n")
                for metric, change in sorted(summary.vs_staging.items(), key=lambda x: abs(x[1]), reverse=True)[:10]:
                    arrow = "📈" if change > 0 else "📉"
                    f.write(f"| {metric} | {arrow} {change:+.2f}% |\n")
                f.write("\n")

            # Detailed Regressions
            if summary.regressions:
                f.write("## Detailed Regression Events\n\n")

                # Group by severity
                for severity_level in ["critical", "high", "medium", "low"]:
                    regs = [r for r in summary.regressions if r.severity == severity_level]
                    if not regs:
                        continue

                    severity_emoji_map = {"critical": "🚨", "high": "❌", "medium": "⚠️", "low": "ℹ️"}
                    f.write(f"### {severity_emoji_map.get(severity_level, '')} {severity_level.upper()} Severity\n\n")

                    for reg in regs:
                        f.write(f"#### {reg.metric_name}\n\n")
                        f.write(f"- **Impact:** {reg.impact}\n")
                        f.write(f"- **Baseline Source:** {reg.baseline_source}\n")
                        f.write(f"- **Category:** {reg.category}\n")
                        f.write(f"- **Priority:** {reg.mitigation_priority}\n")
                        f.write(f"- **Detected:** {reg.first_detected}\n")
                        f.write("\n**Recommended Actions:**\n\n")
                        for i, action in enumerate(reg.recommended_actions, 1):
                            f.write(f"{i}. {action}\n")
                        f.write("\n")
            else:
                f.write("## No Regressions Detected\n\n")
                f.write("All metrics are within acceptable ranges compared to baseline.\n\n")

            # Recommendations
            f.write("## Recommendations\n\n")
            for i, rec in enumerate(summary.recommendations, 1):
                f.write(f"{i}. {rec}\n")
            f.write("\n")

            # Action Items Checklist
            f.write("## Action Items\n\n")

            p0_items = [r for r in summary.regressions if r.mitigation_priority == "P0"]
            p1_items = [r for r in summary.regressions if r.mitigation_priority == "P1"]
            p2_items = [r for r in summary.regressions if r.mitigation_priority == "P2"]

            if p0_items:
                f.write("### P0 (Immediate)\n\n")
                for r in p0_items:
                    f.write(f"- [ ] {r.metric_name}: {r.impact}\n")
                f.write("\n")

            if p1_items:
                f.write("### P1 (24 Hours)\n\n")
                for r in p1_items:
                    f.write(f"- [ ] {r.metric_name}: {r.impact}\n")
                f.write("\n")

            if p2_items:
                f.write("### P2 (Next Sprint)\n\n")
                for r in p2_items[:10]:  # Limit to top 10 P2 items
                    f.write(f"- [ ] {r.metric_name}: {r.impact}\n")
                f.write("\n")

            # Footer
            f.write("---\n\n")
            f.write(f"*Generated by T.A.R.S. Regression Analyzer - {summary.analysis_timestamp}*\n")

        # Enterprise: Encrypt if enabled
        if self.encryptor:
            encrypted_file = md_file.with_suffix(".md.enc")
            self.encryptor.encrypt_file(md_file, encrypted_file)
            logger.info(f"Encrypted markdown summary: {encrypted_file}")

        # Enterprise: Sign if enabled
        if self.signer:
            signature = self.signer.sign_file(md_file)
            sig_file = md_file.with_suffix(".md.sig")
            with open(sig_file, "w") as f:
                f.write(f"RSA-PSS-SHA256\n{signature}\n")
            logger.info(f"Signed markdown summary: {sig_file}")

        logger.info(f"Markdown summary saved: {md_file}")
        return str(md_file)


def main():
    """
    CLI entry point.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="T.A.R.S. Regression Analyzer - Detect regressions from GA baseline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis with GA baseline
  python regression_analyzer.py --ga-baseline ga_kpis/ga_kpi_summary.json --7day-data stability/

  # With staging baseline
  python regression_analyzer.py --ga-baseline ga_kpis/ga_kpi_summary.json --7day-data stability/ --staging-baseline staging_baseline.json

  # Custom output directory
  python regression_analyzer.py --ga-baseline ga_kpis/ga_kpi_summary.json --7day-data stability/ --output reports/

  # Test mode (only load baselines)
  python regression_analyzer.py --ga-baseline ga_kpis/ga_kpi_summary.json --7day-data stability/ --only-baseline
        """
    )
    parser.add_argument(
        "--ga-baseline",
        required=True,
        help="Path to GA Day baseline JSON (ga_kpi_summary.json)"
    )
    parser.add_argument(
        "--7day-data",
        required=True,
        help="Path to 7-day stability data directory"
    )
    parser.add_argument(
        "--staging-baseline",
        help="Path to staging baseline JSON (optional)"
    )
    parser.add_argument(
        "--v1-0-0-baseline",
        help="Path to v1.0.0 baseline JSON (optional, stub)"
    )
    parser.add_argument(
        "--output",
        default=".",
        help="Output directory for analysis results (default: current directory)"
    )
    parser.add_argument(
        "--only-baseline",
        action="store_true",
        help="Only load and print baseline metrics (test mode)"
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Run in test mode (same as --only-baseline)"
    )

    # Enterprise configuration arguments (Phase 14.6)
    if ENTERPRISE_AVAILABLE:
        parser.add_argument("--profile", type=str, default="local", help="Enterprise config profile (local, dev, staging, prod)")
        parser.add_argument("--config", type=str, help="Path to enterprise config file")
        parser.add_argument("--encrypt", action="store_true", help="Encrypt output files (requires AES key)")
        parser.add_argument("--sign", action="store_true", help="Sign output files (requires RSA key)")
        parser.add_argument("--no-compliance", action="store_true", help="Disable compliance enforcement")

    args = parser.parse_args()

    # Initialize enterprise features
    compliance_enforcer = None
    encryptor = None
    signer = None
    config = None

    if ENTERPRISE_AVAILABLE:
        try:
            # Load enterprise configuration
            config_file = Path(args.config) if hasattr(args, 'config') and args.config else None
            config = load_config(
                config_file=config_file,
                environment=args.profile if hasattr(args, 'profile') else "local",
            )

            logger.info(f"✓ Enterprise config loaded (profile: {config.environment.value})")

            # Override with CLI args if provided
            if args.output == "." and config.observability.output_dir != "output":
                args.output = config.observability.output_dir

            # Initialize compliance enforcer
            if not args.no_compliance and config.compliance.enabled_standards:
                from pathlib import Path as P
                compliance_enforcer = ComplianceEnforcer(
                    enabled_standards=config.compliance.enabled_standards,
                    controls_dir=P("compliance/policies"),
                    audit_log_path=P(config.observability.output_dir) / "audit.log" if config.compliance.enable_audit_trail else None,
                    strict_mode=False,
                )
                logger.info(f"✓ Compliance enforcer initialized (standards: {', '.join(config.compliance.enabled_standards)})")

            # Initialize encryption
            if args.encrypt or config.security.enable_encryption:
                key_path = Path(config.security.aes_key_path) if config.security.aes_key_path else None
                if key_path and key_path.exists():
                    encryptor = AESEncryption(key_path=key_path)
                    logger.info(f"✓ AES encryption initialized")
                else:
                    logger.warning("⚠ Encryption requested but no valid AES key found")

            # Initialize signing
            if args.sign or config.security.enable_signing:
                private_key_path = Path(config.security.rsa_private_key_path) if config.security.rsa_private_key_path else None
                if private_key_path and private_key_path.exists():
                    signer = ReportSigner(private_key_path=private_key_path)
                    logger.info(f"✓ RSA signing initialized")
                else:
                    logger.warning("⚠ Signing requested but no valid RSA key found")

        except Exception as e:
            logger.warning(f"⚠ Failed to load enterprise config: {e}")
            logger.info("Falling back to legacy CLI configuration")
            config = None

    try:
        # Instantiate analyzer
        analyzer = RegressionAnalyzer(
            ga_baseline_file=args.ga_baseline,
            seven_day_data_dir=args.__dict__["7day-data"],
            staging_baseline_file=args.staging_baseline,
            v1_0_0_baseline_file=args.v1_0_0_baseline,
            output_dir=args.output,
            compliance_enforcer=compliance_enforcer,
            encryptor=encryptor,
            signer=signer,
        )

        # Test mode: only load baselines
        if args.only_baseline or args.test_mode:
            logger.info("Test mode: Loading baselines only")
            ga_baseline = analyzer.load_ga_baseline()
            seven_day_avg = analyzer.calculate_7day_average()

            print("\n" + "="*60)
            print("BASELINE SUMMARY")
            print("="*60)
            print(f"\nGA Day Baseline:")
            print(f"  Availability: {ga_baseline.availability:.2f}%")
            print(f"  Error Rate: {ga_baseline.error_rate:.4f}%")
            print(f"  P95 Latency: {ga_baseline.p95_latency_ms:.2f}ms")
            print(f"  P99 Latency: {ga_baseline.p99_latency_ms:.2f}ms")
            print(f"  CPU: {ga_baseline.avg_cpu_percent:.2f}%")
            print(f"  Memory: {ga_baseline.avg_memory_percent:.2f}%")

            print(f"\n7-Day Average:")
            print(f"  Availability: {seven_day_avg.availability:.2f}%")
            print(f"  Error Rate: {seven_day_avg.error_rate:.4f}%")
            print(f"  P95 Latency: {seven_day_avg.p95_latency_ms:.2f}ms")
            print(f"  P99 Latency: {seven_day_avg.p99_latency_ms:.2f}ms")
            print(f"  CPU: {seven_day_avg.avg_cpu_percent:.2f}%")
            print(f"  Memory: {seven_day_avg.avg_memory_percent:.2f}%")

            if args.staging_baseline:
                staging = analyzer.load_staging_baseline()
                if staging:
                    print(f"\nStaging Baseline:")
                    print(f"  Availability: {staging.availability:.2f}%")
                    print(f"  Error Rate: {staging.error_rate:.4f}%")
                    print(f"  P95 Latency: {staging.p95_latency_ms:.2f}ms")

            print("\n" + "="*60)
            logger.info("Test mode complete")
            return

        # Run full analysis
        logger.info("Running full regression analysis")
        summary = analyzer.analyze()

        # Save reports
        json_path = analyzer.save_summary_json(summary)
        md_path = analyzer.save_summary_markdown(summary)

        # Print key findings
        print("\n" + "="*60)
        print("REGRESSION ANALYSIS SUMMARY")
        print("="*60)
        print(f"\nOverall Assessment: {summary.overall_assessment.upper().replace('_', ' ')}")
        print(f"Total Regressions: {summary.total_regressions}")
        print(f"  - Critical: {summary.critical_regressions}")
        print(f"  - High: {summary.high_regressions}")
        print(f"  - Medium: {summary.medium_regressions}")
        print(f"  - Low: {summary.low_regressions}")
        print(f"\nRollback Recommended: {'YES' if summary.rollback_recommended else 'NO'}")
        if summary.rollback_reason:
            print(f"Rollback Reason: {summary.rollback_reason}")

        print("\nTop Recommendations:")
        for i, rec in enumerate(summary.recommendations[:5], 1):
            print(f"  {i}. {rec}")

        print(f"\nReports saved:")
        print(f"  - JSON: {json_path}")
        print(f"  - Markdown: {md_path}")
        print("="*60 + "\n")

        logger.info("Regression analysis complete")

    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
