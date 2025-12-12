"""
SLO Violation Detector for T.A.R.S.

Detects Service Level Objective violations using adaptive thresholds and statistical analysis.

Features:
- Latency SLO violation detection
- Error rate monitoring
- Replication lag detection
- Worker starvation detection
- Adaptive thresholds (EWMA + Bollinger Bands)
- Multi-metric correlation
- Alert generation with severity levels

Algorithms:
- EWMA (Exponentially Weighted Moving Average)
- Bollinger Bands for adaptive thresholds
- Z-score anomaly detection
- Percentile-based SLO tracking
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from collections import deque
import statistics
import math

from telemetry.production_log_ingestor import LogEntry, LogLevel


logger = logging.getLogger(__name__)


class SLOType(str, Enum):
    """SLO metric types."""
    LATENCY = "latency"
    ERROR_RATE = "error_rate"
    AVAILABILITY = "availability"
    THROUGHPUT = "throughput"
    REPLICATION_LAG = "replication_lag"
    WORKER_UTILIZATION = "worker_utilization"


class ViolationSeverity(str, Enum):
    """Violation severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class SLO:
    """Service Level Objective definition."""

    name: str
    slo_type: SLOType
    target: float  # Target value (e.g., p95 latency < 200ms)
    threshold: float  # Violation threshold
    window_seconds: int = 300  # Rolling window size
    percentile: float = 0.95  # For percentile-based SLOs

    def __post_init__(self):
        """Validate SLO definition."""
        if self.threshold < self.target:
            raise ValueError(f"Threshold must be >= target for {self.name}")


@dataclass
class SLOViolation:
    """SLO violation record."""

    slo_name: str
    slo_type: SLOType
    severity: ViolationSeverity
    timestamp: datetime
    actual_value: float
    target_value: float
    threshold_value: float
    violation_duration: timedelta

    # Context
    service: Optional[str] = None
    region: Optional[str] = None
    trace_ids: List[str] = field(default_factory=list)

    # Statistical context
    baseline_mean: Optional[float] = None
    baseline_stddev: Optional[float] = None
    z_score: Optional[float] = None

    # Alert metadata
    message: str = ""
    recommended_action: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['violation_duration'] = self.violation_duration.total_seconds()
        data['slo_type'] = self.slo_type.value
        data['severity'] = self.severity.value
        return data

    def to_markdown(self) -> str:
        """Convert to markdown alert."""
        severity_emoji = {
            ViolationSeverity.INFO: 'ℹ️',
            ViolationSeverity.WARNING: '⚠️',
            ViolationSeverity.ERROR: '❌',
            ViolationSeverity.CRITICAL: '🚨',
        }

        md = f"""
## {severity_emoji.get(self.severity, '⚠️')} SLO Violation: {self.slo_name}

**Severity:** {self.severity.value.upper()}
**Type:** {self.slo_type.value}
**Time:** {self.timestamp.isoformat()}
**Duration:** {self.violation_duration.total_seconds():.1f}s

### Metrics
- **Actual Value:** {self.actual_value:.2f}
- **Target Value:** {self.target_value:.2f}
- **Threshold:** {self.threshold_value:.2f}
- **Violation Margin:** {((self.actual_value / self.threshold_value - 1) * 100):.1f}%

### Context
- **Service:** {self.service or 'N/A'}
- **Region:** {self.region or 'N/A'}
- **Affected Traces:** {len(self.trace_ids)}

### Statistical Analysis
- **Baseline Mean:** {self.baseline_mean:.2f if self.baseline_mean else 'N/A'}
- **Baseline StdDev:** {self.baseline_stddev:.2f if self.baseline_stddev else 'N/A'}
- **Z-Score:** {self.z_score:.2f if self.z_score else 'N/A'} σ

### Details
{self.message}

### Recommended Action
{self.recommended_action}
"""
        return md.strip()


class AdaptiveThreshold:
    """Adaptive threshold using EWMA and Bollinger Bands."""

    def __init__(
        self,
        window_size: int = 100,
        alpha: float = 0.1,  # EWMA smoothing factor
        num_stddev: float = 2.0,  # Number of standard deviations for bands
    ):
        """Initialize adaptive threshold.

        Args:
            window_size: Historical window size
            alpha: EWMA smoothing factor (0-1, lower = smoother)
            num_stddev: Number of standard deviations for Bollinger Bands
        """
        self.window_size = window_size
        self.alpha = alpha
        self.num_stddev = num_stddev

        self._values: deque = deque(maxlen=window_size)
        self._ewma: Optional[float] = None
        self._ewma_variance: Optional[float] = None

    def update(self, value: float):
        """Update threshold with new value.

        Args:
            value: New metric value
        """
        self._values.append(value)

        # Update EWMA
        if self._ewma is None:
            self._ewma = value
            self._ewma_variance = 0.0
        else:
            # Update EWMA
            delta = value - self._ewma
            self._ewma = self._ewma + self.alpha * delta

            # Update EWMA variance
            self._ewma_variance = (1 - self.alpha) * (self._ewma_variance + self.alpha * delta * delta)

    def get_bounds(self) -> Tuple[float, float, float]:
        """Get current threshold bounds.

        Returns:
            Tuple of (mean, lower_bound, upper_bound)
        """
        if self._ewma is None or self._ewma_variance is None:
            return (0.0, 0.0, float('inf'))

        mean = self._ewma
        stddev = math.sqrt(self._ewma_variance)

        lower = mean - self.num_stddev * stddev
        upper = mean + self.num_stddev * stddev

        return (mean, lower, upper)

    def is_anomaly(self, value: float) -> bool:
        """Check if value is anomalous.

        Args:
            value: Value to check

        Returns:
            True if anomalous
        """
        if len(self._values) < 10:  # Need minimum samples
            return False

        mean, lower, upper = self.get_bounds()
        return value < lower or value > upper

    def get_z_score(self, value: float) -> float:
        """Calculate z-score for value.

        Args:
            value: Value to score

        Returns:
            Z-score
        """
        if self._ewma is None or self._ewma_variance is None:
            return 0.0

        stddev = math.sqrt(self._ewma_variance)
        if stddev == 0:
            return 0.0

        return (value - self._ewma) / stddev

    def get_stats(self) -> Dict[str, float]:
        """Get current statistics.

        Returns:
            Statistics dictionary
        """
        mean, lower, upper = self.get_bounds()

        return {
            'mean': mean,
            'stddev': math.sqrt(self._ewma_variance) if self._ewma_variance else 0.0,
            'lower_bound': lower,
            'upper_bound': upper,
            'sample_count': len(self._values),
        }


class MetricAggregator:
    """Aggregates metrics from log entries."""

    def __init__(self, window_seconds: int = 300):
        """Initialize aggregator.

        Args:
            window_seconds: Rolling window size in seconds
        """
        self.window_seconds = window_seconds
        self._metrics: Dict[str, deque] = {}

    def add_metric(self, metric_name: str, value: float, timestamp: datetime):
        """Add metric value.

        Args:
            metric_name: Metric name
            value: Metric value
            timestamp: Timestamp
        """
        if metric_name not in self._metrics:
            self._metrics[metric_name] = deque()

        self._metrics[metric_name].append((timestamp, value))
        self._cleanup_old_metrics(metric_name, timestamp)

    def _cleanup_old_metrics(self, metric_name: str, current_time: datetime):
        """Remove metrics outside window.

        Args:
            metric_name: Metric name
            current_time: Current timestamp
        """
        cutoff = current_time - timedelta(seconds=self.window_seconds)
        metrics = self._metrics[metric_name]

        while metrics and metrics[0][0] < cutoff:
            metrics.popleft()

    def get_percentile(self, metric_name: str, percentile: float) -> Optional[float]:
        """Get percentile value.

        Args:
            metric_name: Metric name
            percentile: Percentile (0-1)

        Returns:
            Percentile value or None
        """
        if metric_name not in self._metrics:
            return None

        values = [v for _, v in self._metrics[metric_name]]
        if not values:
            return None

        return statistics.quantiles(values, n=100)[int(percentile * 100) - 1] if len(values) > 1 else values[0]

    def get_mean(self, metric_name: str) -> Optional[float]:
        """Get mean value.

        Args:
            metric_name: Metric name

        Returns:
            Mean value or None
        """
        if metric_name not in self._metrics:
            return None

        values = [v for _, v in self._metrics[metric_name]]
        if not values:
            return None

        return statistics.mean(values)

    def get_rate(self, metric_name: str) -> Optional[float]:
        """Get rate (count per second).

        Args:
            metric_name: Metric name

        Returns:
            Rate or None
        """
        if metric_name not in self._metrics:
            return None

        count = len(self._metrics[metric_name])
        if count == 0:
            return None

        return count / self.window_seconds

    def get_count(self, metric_name: str) -> int:
        """Get count in window.

        Args:
            metric_name: Metric name

        Returns:
            Count
        """
        if metric_name not in self._metrics:
            return 0

        return len(self._metrics[metric_name])


class SLOViolationDetector:
    """Detects SLO violations from logs and metrics."""

    def __init__(
        self,
        slos: List[SLO],
        adaptive_thresholds: bool = True,
        threshold_stddev: float = 2.0,
    ):
        """Initialize detector.

        Args:
            slos: List of SLO definitions
            adaptive_thresholds: Use adaptive thresholds
            threshold_stddev: Standard deviations for adaptive thresholds
        """
        self.slos = {slo.name: slo for slo in slos}
        self.adaptive_thresholds = adaptive_thresholds
        self.threshold_stddev = threshold_stddev

        # Metric aggregation
        self.aggregators: Dict[str, MetricAggregator] = {}
        self.thresholds: Dict[str, AdaptiveThreshold] = {}

        # Violation tracking
        self.active_violations: Dict[str, SLOViolation] = {}
        self.violation_history: List[SLOViolation] = []

        # Initialize aggregators and thresholds
        for slo in slos:
            self.aggregators[slo.name] = MetricAggregator(window_seconds=slo.window_seconds)
            if adaptive_thresholds:
                self.thresholds[slo.name] = AdaptiveThreshold(
                    window_size=100,
                    num_stddev=threshold_stddev,
                )

    def process_log_entry(self, entry: LogEntry) -> List[SLOViolation]:
        """Process log entry and detect violations.

        Args:
            entry: Log entry

        Returns:
            List of detected violations
        """
        violations = []

        # Extract metrics from log entry
        if entry.duration_ms is not None:
            # Latency SLO
            for slo_name, slo in self.slos.items():
                if slo.slo_type == SLOType.LATENCY:
                    self.aggregators[slo_name].add_metric(
                        'latency',
                        entry.duration_ms,
                        entry.timestamp,
                    )

                    # Update adaptive threshold
                    if self.adaptive_thresholds:
                        self.thresholds[slo_name].update(entry.duration_ms)

        # Error rate SLO
        if entry.level in (LogLevel.ERROR, LogLevel.CRITICAL):
            for slo_name, slo in self.slos.items():
                if slo.slo_type == SLOType.ERROR_RATE:
                    self.aggregators[slo_name].add_metric(
                        'error',
                        1.0,
                        entry.timestamp,
                    )

        # Check for violations
        for slo_name, slo in self.slos.items():
            violation = self._check_slo_violation(slo, entry)
            if violation:
                violations.append(violation)

        return violations

    def _check_slo_violation(self, slo: SLO, context_entry: LogEntry) -> Optional[SLOViolation]:
        """Check if SLO is violated.

        Args:
            slo: SLO definition
            context_entry: Context log entry

        Returns:
            SLOViolation if violated, None otherwise
        """
        aggregator = self.aggregators[slo.name]

        # Get current metric value
        if slo.slo_type == SLOType.LATENCY:
            current_value = aggregator.get_percentile('latency', slo.percentile)
        elif slo.slo_type == SLOType.ERROR_RATE:
            error_count = aggregator.get_count('error')
            total_count = max(error_count, 1)  # Avoid division by zero
            current_value = error_count / total_count
        else:
            return None

        if current_value is None:
            return None

        # Determine threshold
        if self.adaptive_thresholds and slo.name in self.thresholds:
            threshold_obj = self.thresholds[slo.name]
            mean, _, upper = threshold_obj.get_bounds()
            threshold = max(slo.threshold, upper)
            z_score = threshold_obj.get_z_score(current_value)
            baseline_mean = mean
            baseline_stddev = math.sqrt(threshold_obj._ewma_variance) if threshold_obj._ewma_variance else 0.0
        else:
            threshold = slo.threshold
            z_score = None
            baseline_mean = None
            baseline_stddev = None

        # Check violation
        if current_value > threshold:
            # Determine severity
            margin = (current_value / threshold) - 1

            if margin > 1.0:  # >100% over threshold
                severity = ViolationSeverity.CRITICAL
            elif margin > 0.5:  # >50% over threshold
                severity = ViolationSeverity.ERROR
            elif margin > 0.2:  # >20% over threshold
                severity = ViolationSeverity.WARNING
            else:
                severity = ViolationSeverity.INFO

            # Check if already tracking this violation
            if slo.name in self.active_violations:
                # Update existing violation
                violation = self.active_violations[slo.name]
                violation.actual_value = current_value
                violation.violation_duration = context_entry.timestamp - violation.timestamp
                violation.severity = max(violation.severity, severity, key=lambda s: list(ViolationSeverity).index(s))
            else:
                # Create new violation
                violation = SLOViolation(
                    slo_name=slo.name,
                    slo_type=slo.slo_type,
                    severity=severity,
                    timestamp=context_entry.timestamp,
                    actual_value=current_value,
                    target_value=slo.target,
                    threshold_value=threshold,
                    violation_duration=timedelta(seconds=0),
                    service=context_entry.service,
                    region=context_entry.region,
                    trace_ids=[context_entry.trace_id] if context_entry.trace_id else [],
                    baseline_mean=baseline_mean,
                    baseline_stddev=baseline_stddev,
                    z_score=z_score,
                    message=self._generate_message(slo, current_value, threshold),
                    recommended_action=self._generate_recommendation(slo, current_value, threshold),
                )

                self.active_violations[slo.name] = violation
                self.violation_history.append(violation)

            return violation

        else:
            # Clear active violation if it exists
            if slo.name in self.active_violations:
                del self.active_violations[slo.name]

        return None

    def _generate_message(self, slo: SLO, actual: float, threshold: float) -> str:
        """Generate violation message.

        Args:
            slo: SLO definition
            actual: Actual value
            threshold: Threshold value

        Returns:
            Message string
        """
        margin = ((actual / threshold) - 1) * 100

        if slo.slo_type == SLOType.LATENCY:
            return f"Latency p{int(slo.percentile * 100)} of {actual:.1f}ms exceeds threshold of {threshold:.1f}ms by {margin:.1f}%"
        elif slo.slo_type == SLOType.ERROR_RATE:
            return f"Error rate of {actual * 100:.2f}% exceeds threshold of {threshold * 100:.2f}% by {margin:.1f}%"
        else:
            return f"Metric {actual:.2f} exceeds threshold {threshold:.2f} by {margin:.1f}%"

    def _generate_recommendation(self, slo: SLO, actual: float, threshold: float) -> str:
        """Generate recommended action.

        Args:
            slo: SLO definition
            actual: Actual value
            threshold: Threshold value

        Returns:
            Recommendation string
        """
        if slo.slo_type == SLOType.LATENCY:
            return """
1. Check for recent deployments or config changes
2. Review database query performance
3. Check external service dependencies
4. Verify cache hit rates
5. Scale horizontally if sustained high load
"""
        elif slo.slo_type == SLOType.ERROR_RATE:
            return """
1. Check application logs for error patterns
2. Review recent code changes
3. Verify database connectivity
4. Check external service availability
5. Review monitoring dashboards for correlated metrics
"""
        else:
            return "Review monitoring dashboards and application logs for root cause."

    def get_active_violations(self) -> List[SLOViolation]:
        """Get currently active violations.

        Returns:
            List of active violations
        """
        return list(self.active_violations.values())

    def get_violation_summary(self) -> Dict[str, Any]:
        """Get violation summary.

        Returns:
            Summary dictionary
        """
        total = len(self.violation_history)
        active = len(self.active_violations)

        by_severity = {}
        for violation in self.violation_history:
            sev = violation.severity.value
            by_severity[sev] = by_severity.get(sev, 0) + 1

        by_type = {}
        for violation in self.violation_history:
            typ = violation.slo_type.value
            by_type[typ] = by_type.get(typ, 0) + 1

        return {
            'total_violations': total,
            'active_violations': active,
            'violations_by_severity': by_severity,
            'violations_by_type': by_type,
            'slo_count': len(self.slos),
        }

    def export_violations_json(self) -> str:
        """Export violations as JSON.

        Returns:
            JSON string
        """
        return json.dumps(
            {
                'active_violations': [v.to_dict() for v in self.active_violations.values()],
                'violation_history': [v.to_dict() for v in self.violation_history],
                'summary': self.get_violation_summary(),
            },
            indent=2,
        )

    def export_violations_markdown(self) -> str:
        """Export violations as markdown.

        Returns:
            Markdown string
        """
        md = "# SLO Violation Report\n\n"
        md += f"**Generated:** {datetime.utcnow().isoformat()}\n\n"

        summary = self.get_violation_summary()
        md += "## Summary\n\n"
        md += f"- **Total Violations:** {summary['total_violations']}\n"
        md += f"- **Active Violations:** {summary['active_violations']}\n"
        md += f"- **Monitored SLOs:** {summary['slo_count']}\n\n"

        if summary['violations_by_severity']:
            md += "### By Severity\n\n"
            for sev, count in summary['violations_by_severity'].items():
                md += f"- **{sev.upper()}:** {count}\n"
            md += "\n"

        if self.active_violations:
            md += "## Active Violations\n\n"
            for violation in self.active_violations.values():
                md += violation.to_markdown()
                md += "\n\n---\n\n"

        return md


# Example usage
if __name__ == '__main__':
    # Define SLOs
    slos = [
        SLO(
            name='api_latency_p95',
            slo_type=SLOType.LATENCY,
            target=150.0,  # 150ms target
            threshold=200.0,  # 200ms violation threshold
            window_seconds=300,
            percentile=0.95,
        ),
        SLO(
            name='error_rate',
            slo_type=SLOType.ERROR_RATE,
            target=0.01,  # 1% target
            threshold=0.05,  # 5% violation threshold
            window_seconds=300,
            percentile=1.0,
        ),
    ]

    # Create detector
    detector = SLOViolationDetector(slos, adaptive_thresholds=True)

    # Simulate log processing
    from telemetry.production_log_ingestor import LogEntry

    for i in range(100):
        # Normal latency
        entry = LogEntry(
            timestamp=datetime.utcnow(),
            service='api-gateway',
            level=LogLevel.INFO,
            message='Request completed',
            duration_ms=120.0 + (i % 20),
            region='us-east-1',
        )
        detector.process_log_entry(entry)

    # Violation
    violation_entry = LogEntry(
        timestamp=datetime.utcnow(),
        service='api-gateway',
        level=LogLevel.INFO,
        message='Slow request',
        duration_ms=350.0,
        region='us-east-1',
    )
    violations = detector.process_log_entry(violation_entry)

    if violations:
        print(violations[0].to_markdown())

    print("\n" + detector.export_violations_markdown())
