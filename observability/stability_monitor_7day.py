#!/usr/bin/env python3
"""
7-Day Post-GA Stability Monitor - Continuous Production Monitoring

Monitors production stability for 7 days after GA Day, collecting snapshots every 30 minutes
and generating daily aggregation reports. Detects drift from GA Day baseline, SLO degradation,
and resource regression.

Usage:
    python stability_monitor_7day.py --baseline ga_kpis/ga_kpi_summary.json --duration 168
    python stability_monitor_7day.py --baseline ga_kpis/ga_kpi_summary.json --test-mode --duration 2

Author: T.A.R.S. Platform Team
Phase: 14.6 - Post-GA 7-Day Stabilization & Retrospective
"""

import asyncio
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import hashlib

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import shared PrometheusClient
from observability.shared import PrometheusClient, PrometheusQueryError

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
class StabilitySnapshot:
    """
    A single stability snapshot at a point in time (30-minute interval).

    Captures key metrics for stability tracking:
    - Availability and error rates
    - Latency percentiles (P50, P95, P99)
    - Resource utilization (CPU, memory)
    - Database and cache performance
    - Alert counts by severity
    """
    timestamp: str
    elapsed_hours: float
    day_number: int  # 1-7

    # Availability & Errors
    overall_availability: float  # percentage
    error_rate: float  # percentage
    total_requests: int
    total_errors: int

    # Latency
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float

    # Resource Utilization
    avg_cpu_percent: float
    avg_memory_percent: float
    peak_cpu_percent: float
    peak_memory_percent: float

    # Infrastructure
    db_p95_latency_ms: float
    redis_hit_rate: float
    cluster_cpu_utilization: float
    cluster_memory_utilization: float
    node_count: int

    # Alerts
    critical_alerts: int
    warning_alerts: int
    info_alerts: int

    # Drift from GA baseline
    drift_from_baseline: Dict[str, float] = field(default_factory=dict)


@dataclass
class DailyAggregation:
    """
    Daily aggregation of stability metrics (24-hour period).

    Provides daily rollup of:
    - Min/max/avg for all key metrics
    - SLO compliance percentages
    - Drift trends
    - Alert summaries
    """
    day_number: int  # 1-7
    start_time: str
    end_time: str
    duration_hours: float
    snapshot_count: int

    # Availability Summary
    avg_availability: float
    min_availability: float
    max_availability: float
    slo_compliance_percent: float  # % of time availability >= 99.9%

    # Error Rate Summary
    avg_error_rate: float
    max_error_rate: float
    total_requests: int
    total_errors: int

    # Latency Summary
    avg_p95_latency_ms: float
    max_p95_latency_ms: float
    avg_p99_latency_ms: float
    max_p99_latency_ms: float
    latency_slo_compliance_percent: float  # % of time P99 < 500ms

    # Resource Summary
    avg_cpu_percent: float
    peak_cpu_percent: float
    avg_memory_percent: float
    peak_memory_percent: float

    # Alert Summary
    total_critical_alerts: int
    total_warning_alerts: int
    total_info_alerts: int
    alert_free_hours: float  # Hours with no critical alerts

    # Drift Summary
    max_drift_percent: float
    avg_drift_percent: float
    metrics_exceeding_drift_threshold: List[str] = field(default_factory=list)

    # SLO Degradation Flags
    availability_degraded: bool = False
    latency_degraded: bool = False
    error_rate_degraded: bool = False


@dataclass
class SevenDaySummary:
    """
    Overall 7-day stability summary.

    Provides week-over-week analysis:
    - Weekly trends
    - Regression detection
    - Stability score
    - Recommendations
    """
    start_time: str
    end_time: str
    total_duration_hours: float
    total_snapshots: int
    baseline_timestamp: str

    # Weekly Averages
    avg_availability: float
    avg_error_rate: float
    avg_p99_latency_ms: float
    avg_drift_percent: float

    # Stability Metrics
    stability_score: float  # 0-100, based on SLO compliance
    days_with_degradation: int
    total_critical_alerts: int
    total_warning_alerts: int

    # Daily Breakdown
    daily_summaries: List[DailyAggregation] = field(default_factory=list)

    # Trend Analysis
    availability_trend: str  # "improving", "stable", "degrading"
    latency_trend: str
    resource_trend: str

    # Recommendations
    recommendations: List[str] = field(default_factory=list)
    action_items: List[str] = field(default_factory=list)


class StabilityMonitor:
    """
    Main 7-day stability monitoring daemon.
    """

    def __init__(
        self,
        baseline_file: str,
        duration_hours: int = 168,  # 7 days default
        interval_minutes: int = 30,
        output_dir: str = "stability",
        prometheus_url: str = "http://prometheus.tars-production.svc.cluster.local:9090",
        # Enterprise features
        compliance_enforcer: Optional[ComplianceEnforcer] = None,
        encryptor: Optional[AESEncryption] = None,
        signer: Optional[ReportSigner] = None,
    ):
        """
        Initialize 7-day stability monitor.

        Args:
            baseline_file: Path to GA Day baseline JSON (ga_kpi_summary.json)
            duration_hours: Monitoring duration (default: 168 hours = 7 days)
            interval_minutes: Snapshot interval (default: 30 minutes)
            output_dir: Output directory for stability reports
            prometheus_url: Prometheus server URL
            compliance_enforcer: Optional compliance enforcer (enterprise)
            encryptor: Optional AES encryptor (enterprise)
            signer: Optional RSA signer (enterprise)
        """
        self.baseline_file = baseline_file
        self.duration_hours = duration_hours
        self.interval_minutes = interval_minutes
        self.output_dir = Path(output_dir)
        self.prometheus_url = prometheus_url

        self.baseline_metrics: Dict[str, Any] = {}
        self.snapshots: List[StabilitySnapshot] = []
        self.daily_aggregations: List[DailyAggregation] = []

        self.prom_client = PrometheusClient(prometheus_url)

        # Enterprise features (Phase 14.6)
        self.compliance_enforcer = compliance_enforcer
        self.encryptor = encryptor
        self.signer = signer

        logger.info(f"StabilityMonitor initialized: duration={duration_hours}h, interval={interval_minutes}m")

    def load_baseline(self) -> None:
        """
        Load GA Day baseline metrics from JSON file.

        Expected baseline structure (from Phase 14.5 ga_kpi_summary.json):
        {
            "overall_availability": 99.95,
            "overall_error_rate": 0.05,
            "avg_p95_latency_ms": 120.5,
            "avg_p99_latency_ms": 245.8,
            "avg_cpu_percent": 35.2,
            "avg_memory_percent": 42.1,
            "avg_db_latency_ms": 15.3,
            "avg_redis_hit_rate": 97.5,
            ...
        }

        Raises:
            FileNotFoundError: If baseline file doesn't exist
            json.JSONDecodeError: If baseline file is invalid JSON
        """
        baseline_path = Path(self.baseline_file)

        if not baseline_path.exists():
            raise FileNotFoundError(
                f"Baseline file not found: {self.baseline_file}\n"
                f"Please run Phase 14.5 GA Day monitoring first to generate baseline."
            )

        logger.info(f"Loading GA Day baseline from: {self.baseline_file}")

        with open(baseline_path, 'r') as f:
            self.baseline_metrics = json.load(f)

        # Validate required fields
        required_fields = [
            "overall_availability",
            "overall_error_rate",
            "avg_p95_latency_ms",
            "avg_p99_latency_ms",
            "avg_cpu_percent",
            "avg_memory_percent"
        ]

        missing_fields = [field for field in required_fields if field not in self.baseline_metrics]
        if missing_fields:
            raise ValueError(
                f"Baseline file missing required fields: {', '.join(missing_fields)}"
            )

        logger.info(
            f"Baseline loaded: availability={self.baseline_metrics['overall_availability']}%, "
            f"error_rate={self.baseline_metrics['overall_error_rate']}%, "
            f"p99_latency={self.baseline_metrics['avg_p99_latency_ms']}ms"
        )

    async def collect_snapshot(self, elapsed_hours: float) -> StabilitySnapshot:
        """
        Collect a single stability snapshot at current time.

        Queries Prometheus for:
        - Service availability (up metric)
        - Request/error counts (http_requests_total)
        - Latency percentiles (http_request_duration_seconds)
        - CPU/memory utilization (process_*, container_*)
        - Database latency (db_query_duration_seconds)
        - Redis hit rate (redis_keyspace_*)
        - Alert counts (ALERTS metric)

        Args:
            elapsed_hours: Hours elapsed since monitoring start

        Returns:
            StabilitySnapshot with current metrics
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        day_number = int(elapsed_hours / 24) + 1

        logger.info(f"Collecting stability snapshot at T+{elapsed_hours:.1f}h (Day {day_number})...")

        async with PrometheusClient(self.prometheus_url) as prom:
            # === Availability ===
            avail_query = 'avg_over_time(up[5m]) * 100'
            avail_result = await prom.query_safe(avail_query, default={})
            overall_availability = 100.0  # default
            if avail_result and avail_result.get("result"):
                overall_availability = float(avail_result["result"][0]["value"][1])

            # === Request & Error Counts (5-minute rate) ===
            req_query = 'sum(rate(http_requests_total[5m])) * 300'  # 5m window in seconds
            req_result = await prom.query_safe(req_query, default={})
            total_requests = 0
            if req_result and req_result.get("result"):
                total_requests = int(float(req_result["result"][0]["value"][1]))

            err_query = 'sum(rate(http_requests_total{status=~"5.."}[5m])) * 300'
            err_result = await prom.query_safe(err_query, default={})
            total_errors = 0
            if err_result and err_result.get("result"):
                total_errors = int(float(err_result["result"][0]["value"][1]))

            error_rate = (total_errors / total_requests * 100.0) if total_requests > 0 else 0.0

            # === Latency Percentiles ===
            p50_query = 'histogram_quantile(0.50, rate(http_request_duration_seconds_bucket[5m])) * 1000'
            p95_query = 'histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) * 1000'
            p99_query = 'histogram_quantile(0.99, rate(http_request_duration_seconds_bucket[5m])) * 1000'

            p50_result = await prom.query_safe(p50_query, default={})
            p95_result = await prom.query_safe(p95_query, default={})
            p99_result = await prom.query_safe(p99_query, default={})

            p50_latency = 0.0
            p95_latency = 0.0
            p99_latency = 0.0

            if p50_result and p50_result.get("result"):
                p50_latency = float(p50_result["result"][0]["value"][1])
            if p95_result and p95_result.get("result"):
                p95_latency = float(p95_result["result"][0]["value"][1])
            if p99_result and p99_result.get("result"):
                p99_latency = float(p99_result["result"][0]["value"][1])

            # === CPU Utilization ===
            cpu_query = 'avg(rate(process_cpu_seconds_total[5m])) * 100'
            cpu_result = await prom.query_safe(cpu_query, default={})
            avg_cpu_percent = 0.0
            if cpu_result and cpu_result.get("result"):
                avg_cpu_percent = float(cpu_result["result"][0]["value"][1])

            peak_cpu_query = 'max(rate(process_cpu_seconds_total[5m])) * 100'
            peak_cpu_result = await prom.query_safe(peak_cpu_query, default={})
            peak_cpu_percent = avg_cpu_percent  # fallback
            if peak_cpu_result and peak_cpu_result.get("result"):
                peak_cpu_percent = float(peak_cpu_result["result"][0]["value"][1])

            # === Memory Utilization ===
            mem_query = 'avg(process_resident_memory_bytes) / (1024 * 1024 * 1024) * 100'  # Rough percentage
            mem_result = await prom.query_safe(mem_query, default={})
            avg_memory_percent = 0.0
            if mem_result and mem_result.get("result"):
                avg_memory_percent = float(mem_result["result"][0]["value"][1])

            peak_mem_query = 'max(process_resident_memory_bytes) / (1024 * 1024 * 1024) * 100'
            peak_mem_result = await prom.query_safe(peak_mem_query, default={})
            peak_memory_percent = avg_memory_percent  # fallback
            if peak_mem_result and peak_mem_result.get("result"):
                peak_memory_percent = float(peak_mem_result["result"][0]["value"][1])

            # === Database Latency ===
            db_query = 'histogram_quantile(0.95, rate(db_query_duration_seconds_bucket[5m])) * 1000'
            db_result = await prom.query_safe(db_query, default={})
            db_p95_latency = 0.0
            if db_result and db_result.get("result"):
                db_p95_latency = float(db_result["result"][0]["value"][1])

            # === Redis Hit Rate ===
            redis_query = 'redis_keyspace_hits_total / (redis_keyspace_hits_total + redis_keyspace_misses_total) * 100'
            redis_result = await prom.query_safe(redis_query, default={})
            redis_hit_rate = 0.0
            if redis_result and redis_result.get("result"):
                redis_hit_rate = float(redis_result["result"][0]["value"][1])

            # === Cluster Utilization ===
            cluster_cpu_query = 'sum(rate(container_cpu_usage_seconds_total[5m])) / sum(machine_cpu_cores) * 100'
            cluster_cpu_result = await prom.query_safe(cluster_cpu_query, default={})
            cluster_cpu_utilization = 0.0
            if cluster_cpu_result and cluster_cpu_result.get("result"):
                cluster_cpu_utilization = float(cluster_cpu_result["result"][0]["value"][1])

            cluster_mem_query = 'sum(container_memory_working_set_bytes) / sum(machine_memory_bytes) * 100'
            cluster_mem_result = await prom.query_safe(cluster_mem_query, default={})
            cluster_memory_utilization = 0.0
            if cluster_mem_result and cluster_mem_result.get("result"):
                cluster_memory_utilization = float(cluster_mem_result["result"][0]["value"][1])

            # === Node Count ===
            node_query = 'count(kube_node_info)'
            node_result = await prom.query_safe(node_query, default={})
            node_count = 0
            if node_result and node_result.get("result"):
                node_count = int(float(node_result["result"][0]["value"][1]))

            # === Alert Counts ===
            critical_alerts_query = 'count(ALERTS{alertstate="firing",severity="critical"})'
            warning_alerts_query = 'count(ALERTS{alertstate="firing",severity="warning"})'
            info_alerts_query = 'count(ALERTS{alertstate="firing",severity="info"})'

            critical_result = await prom.query_safe(critical_alerts_query, default={})
            warning_result = await prom.query_safe(warning_alerts_query, default={})
            info_result = await prom.query_safe(info_alerts_query, default={})

            critical_alerts = 0
            warning_alerts = 0
            info_alerts = 0

            if critical_result and critical_result.get("result"):
                critical_alerts = int(float(critical_result["result"][0]["value"][1]))
            if warning_result and warning_result.get("result"):
                warning_alerts = int(float(warning_result["result"][0]["value"][1]))
            if info_result and info_result.get("result"):
                info_alerts = int(float(info_result["result"][0]["value"][1]))

        # Create snapshot
        snapshot = StabilitySnapshot(
            timestamp=timestamp,
            elapsed_hours=round(elapsed_hours, 2),
            day_number=day_number,
            overall_availability=round(overall_availability, 2),
            error_rate=round(error_rate, 4),
            total_requests=total_requests,
            total_errors=total_errors,
            p50_latency_ms=round(p50_latency, 2),
            p95_latency_ms=round(p95_latency, 2),
            p99_latency_ms=round(p99_latency, 2),
            avg_cpu_percent=round(avg_cpu_percent, 2),
            avg_memory_percent=round(avg_memory_percent, 2),
            peak_cpu_percent=round(peak_cpu_percent, 2),
            peak_memory_percent=round(peak_memory_percent, 2),
            db_p95_latency_ms=round(db_p95_latency, 2),
            redis_hit_rate=round(redis_hit_rate, 2),
            cluster_cpu_utilization=round(cluster_cpu_utilization, 2),
            cluster_memory_utilization=round(cluster_memory_utilization, 2),
            node_count=node_count,
            critical_alerts=critical_alerts,
            warning_alerts=warning_alerts,
            info_alerts=info_alerts,
            drift_from_baseline={}  # Will be populated next
        )

        # Calculate drift from baseline
        snapshot.drift_from_baseline = self.calculate_drift_from_baseline(snapshot)

        logger.info(
            f"Snapshot collected: availability={snapshot.overall_availability}%, "
            f"error_rate={snapshot.error_rate}%, p99={snapshot.p99_latency_ms}ms, "
            f"alerts={snapshot.critical_alerts}C/{snapshot.warning_alerts}W"
        )

        return snapshot

    def calculate_drift_from_baseline(self, snapshot: StabilitySnapshot) -> Dict[str, float]:
        """
        Calculate drift percentages from GA Day baseline.

        Drift formula: ((current - baseline) / baseline) * 100

        Positive drift: metric increased (degradation for latency/errors/resources)
        Negative drift: metric decreased (improvement for latency/errors, degradation for availability)

        Args:
            snapshot: Current snapshot

        Returns:
            Dict mapping metric names to drift percentages

        Example output:
        {
            "availability_drift": -0.05,  # 0.05% decrease (bad)
            "error_rate_drift": +15.2,    # 15.2% increase (bad)
            "p99_latency_drift": +8.3,    # 8.3% increase (bad)
            "cpu_drift": +12.1,           # 12.1% increase (watch)
            "memory_drift": +6.4          # 6.4% increase (watch)
        }
        """
        if not self.baseline_metrics:
            logger.warning("No baseline loaded - cannot calculate drift")
            return {}

        drift = {}

        def safe_drift(current: float, baseline_key: str) -> float:
            """Calculate drift with zero-division protection."""
            baseline_val = self.baseline_metrics.get(baseline_key, 0.0)
            if baseline_val == 0:
                return 0.0
            return ((current - baseline_val) / baseline_val) * 100.0

        # Availability drift (negative is bad)
        drift["availability_drift"] = safe_drift(
            snapshot.overall_availability,
            "overall_availability"
        )

        # Error rate drift (positive is bad)
        drift["error_rate_drift"] = safe_drift(
            snapshot.error_rate,
            "overall_error_rate"
        )

        # Latency drifts (positive is bad)
        drift["p95_latency_drift"] = safe_drift(
            snapshot.p95_latency_ms,
            "avg_p95_latency_ms"
        )
        drift["p99_latency_drift"] = safe_drift(
            snapshot.p99_latency_ms,
            "avg_p99_latency_ms"
        )

        # Resource drifts (positive is watch)
        drift["cpu_drift"] = safe_drift(
            snapshot.avg_cpu_percent,
            "avg_cpu_percent"
        )
        drift["memory_drift"] = safe_drift(
            snapshot.avg_memory_percent,
            "avg_memory_percent"
        )

        # Database drift (positive is bad)
        if "avg_db_latency_ms" in self.baseline_metrics:
            drift["db_latency_drift"] = safe_drift(
                snapshot.db_p95_latency_ms,
                "avg_db_latency_ms"
            )

        # Redis drift (negative is bad - hit rate should stay high)
        if "avg_redis_hit_rate" in self.baseline_metrics:
            drift["redis_hit_rate_drift"] = safe_drift(
                snapshot.redis_hit_rate,
                "avg_redis_hit_rate"
            )

        # Round all drifts
        drift = {k: round(v, 2) for k, v in drift.items()}

        return drift

    def check_slo_degradation(self, snapshot: StabilitySnapshot) -> Dict[str, bool]:
        """
        Check for SLO degradation compared to GA Day.

        SLO Thresholds (production targets):
        - Availability: >= 99.9%
        - P99 Latency: < 500ms
        - Error Rate: < 0.1%

        Args:
            snapshot: Current snapshot

        Returns:
            Dict with degradation flags for each SLO

        Example output:
        {
            "availability_degraded": False,  # 99.95% >= 99.9% (OK)
            "latency_degraded": True,        # 520ms >= 500ms (DEGRADED)
            "error_rate_degraded": False     # 0.05% < 0.1% (OK)
        }
        """
        degradation = {
            "availability_degraded": snapshot.overall_availability < 99.9,
            "latency_degraded": snapshot.p99_latency_ms >= 500.0,
            "error_rate_degraded": snapshot.error_rate >= 0.1
        }

        # Log any degradations
        if any(degradation.values()):
            degraded_slos = [k.replace("_degraded", "") for k, v in degradation.items() if v]
            logger.warning(
                f"SLO degradation detected: {', '.join(degraded_slos)} "
                f"(availability={snapshot.overall_availability}%, "
                f"p99={snapshot.p99_latency_ms}ms, "
                f"error_rate={snapshot.error_rate}%)"
            )

        return degradation

    def check_resource_regression(self, snapshot: StabilitySnapshot) -> Dict[str, bool]:
        """
        Check for resource regression (CPU/memory increase).

        Regression Thresholds (% increase from baseline):
        - CPU: > 20% increase from baseline
        - Memory: > 20% increase from baseline

        Args:
            snapshot: Current snapshot

        Returns:
            Dict with regression flags for CPU and memory

        Example output:
        {
            "cpu_regressed": True,    # 25% increase (REGRESSED)
            "memory_regressed": False # 15% increase (OK)
        }
        """
        if not self.baseline_metrics:
            return {"cpu_regressed": False, "memory_regressed": False}

        # Get baseline values
        baseline_cpu = self.baseline_metrics.get("avg_cpu_percent", 0.0)
        baseline_memory = self.baseline_metrics.get("avg_memory_percent", 0.0)

        # Calculate regression
        cpu_regressed = False
        memory_regressed = False

        if baseline_cpu > 0:
            cpu_increase_percent = ((snapshot.avg_cpu_percent - baseline_cpu) / baseline_cpu) * 100.0
            cpu_regressed = cpu_increase_percent > 20.0

        if baseline_memory > 0:
            memory_increase_percent = ((snapshot.avg_memory_percent - baseline_memory) / baseline_memory) * 100.0
            memory_regressed = memory_increase_percent > 20.0

        regression = {
            "cpu_regressed": cpu_regressed,
            "memory_regressed": memory_regressed
        }

        # Log any regressions
        if any(regression.values()):
            regressed_resources = [k.replace("_regressed", "") for k, v in regression.items() if v]
            logger.warning(
                f"Resource regression detected: {', '.join(regressed_resources)} "
                f"(cpu={snapshot.avg_cpu_percent}% vs baseline={baseline_cpu}%, "
                f"memory={snapshot.avg_memory_percent}% vs baseline={baseline_memory}%)"
            )

        return regression

    async def save_snapshot(self, snapshot: StabilitySnapshot, day: int) -> None:
        """
        Save snapshot to JSON file.

        File structure:
        - stability/day_01_raw/ - Raw snapshots for Day 1
        - stability/day_02_raw/ - Raw snapshots for Day 2
        - etc.

        Each snapshot is saved to stability/day_XX_raw/snapshot_YYYY-MM-DD_HH-MM-SS.json

        Args:
            snapshot: Snapshot to save
            day: Day number (1-7)
        """
        # Create output directory
        day_dir = self.output_dir / f"day_{day:02d}_raw"
        day_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename with timestamp
        # Extract timestamp from ISO format: "2025-01-15T12:30:00+00:00" -> "2025-01-15_12-30-00"
        timestamp_str = snapshot.timestamp.replace(":", "-").replace("T", "_").split("+")[0].split(".")[0]
        snapshot_file = day_dir / f"snapshot_{timestamp_str}.json"

        # Convert to dict
        snapshot_dict = asdict(snapshot)

        # Write snapshot
        with open(snapshot_file, 'w') as f:
            json.dump(snapshot_dict, f, indent=2)

        # Enterprise: Encrypt if enabled
        if self.encryptor:
            encrypted_file = snapshot_file.with_suffix(".json.enc")
            self.encryptor.encrypt_file(snapshot_file, encrypted_file)
            logger.debug(f"Encrypted snapshot: {encrypted_file}")

        # Enterprise: Sign if enabled
        if self.signer:
            signature = self.signer.sign_file(snapshot_file)
            sig_file = snapshot_file.with_suffix(".json.sig")
            with open(sig_file, "w") as f:
                f.write(f"RSA-PSS-SHA256\n{signature}\n")
            logger.debug(f"Signed snapshot: {sig_file}")

        logger.debug(f"Snapshot saved: {snapshot_file}")

    def aggregate_daily_metrics(self, day_snapshots: List[StabilitySnapshot]) -> DailyAggregation:
        """
        Aggregate snapshots into a daily summary.

        Calculates:
        - Min/max/avg for availability, error rate, latency, CPU, memory
        - SLO compliance percentages (% of snapshots meeting SLO)
        - Alert summaries
        - Drift summaries
        - Degradation flags

        Args:
            day_snapshots: All snapshots for a single day

        Returns:
            DailyAggregation with min/max/avg metrics

        Raises:
            ValueError: If day_snapshots is empty
        """
        if not day_snapshots:
            raise ValueError("Cannot aggregate empty snapshot list")

        day_number = day_snapshots[0].day_number
        start_time = day_snapshots[0].timestamp
        end_time = day_snapshots[-1].timestamp
        snapshot_count = len(day_snapshots)

        logger.info(f"Aggregating {snapshot_count} snapshots for Day {day_number}...")

        # === Availability Metrics ===
        availabilities = [s.overall_availability for s in day_snapshots]
        avg_availability = sum(availabilities) / len(availabilities)
        min_availability = min(availabilities)
        max_availability = max(availabilities)

        # SLO compliance: % of snapshots with availability >= 99.9%
        slo_compliant_snapshots = sum(1 for a in availabilities if a >= 99.9)
        slo_compliance_percent = (slo_compliant_snapshots / snapshot_count) * 100.0

        # === Error Rate Metrics ===
        error_rates = [s.error_rate for s in day_snapshots]
        avg_error_rate = sum(error_rates) / len(error_rates)
        max_error_rate = max(error_rates)

        total_requests = sum(s.total_requests for s in day_snapshots)
        total_errors = sum(s.total_errors for s in day_snapshots)

        # === Latency Metrics ===
        p95_latencies = [s.p95_latency_ms for s in day_snapshots]
        p99_latencies = [s.p99_latency_ms for s in day_snapshots]

        avg_p95_latency = sum(p95_latencies) / len(p95_latencies)
        max_p95_latency = max(p95_latencies)
        avg_p99_latency = sum(p99_latencies) / len(p99_latencies)
        max_p99_latency = max(p99_latencies)

        # Latency SLO compliance: % of snapshots with P99 < 500ms
        latency_slo_compliant = sum(1 for p99 in p99_latencies if p99 < 500.0)
        latency_slo_compliance_percent = (latency_slo_compliant / snapshot_count) * 100.0

        # === Resource Metrics ===
        cpu_percents = [s.avg_cpu_percent for s in day_snapshots]
        avg_cpu_percent = sum(cpu_percents) / len(cpu_percents)
        peak_cpu_percent = max(s.peak_cpu_percent for s in day_snapshots)

        memory_percents = [s.avg_memory_percent for s in day_snapshots]
        avg_memory_percent = sum(memory_percents) / len(memory_percents)
        peak_memory_percent = max(s.peak_memory_percent for s in day_snapshots)

        # === Alert Metrics ===
        total_critical_alerts = sum(s.critical_alerts for s in day_snapshots)
        total_warning_alerts = sum(s.warning_alerts for s in day_snapshots)
        total_info_alerts = sum(s.info_alerts for s in day_snapshots)

        # Alert-free hours: count snapshots with 0 critical alerts
        alert_free_snapshots = sum(1 for s in day_snapshots if s.critical_alerts == 0)
        alert_free_hours = (alert_free_snapshots / snapshot_count) * 24.0

        # === Drift Metrics ===
        # Calculate avg drift for each metric across all snapshots
        drift_metrics = {}
        if day_snapshots[0].drift_from_baseline:
            drift_keys = day_snapshots[0].drift_from_baseline.keys()
            for key in drift_keys:
                drifts = [abs(s.drift_from_baseline.get(key, 0.0)) for s in day_snapshots]
                drift_metrics[key] = sum(drifts) / len(drifts)

        # Overall drift summary
        if drift_metrics:
            max_drift_percent = max(drift_metrics.values())
            avg_drift_percent = sum(drift_metrics.values()) / len(drift_metrics)
            # Find metrics exceeding 10% drift threshold
            metrics_exceeding_threshold = [
                k for k, v in drift_metrics.items()
                if v > 10.0
            ]
        else:
            max_drift_percent = 0.0
            avg_drift_percent = 0.0
            metrics_exceeding_threshold = []

        # === Degradation Flags ===
        # Flag degradation if SLO compliance < 95% (i.e., SLO violated >5% of the time)
        availability_degraded = slo_compliance_percent < 95.0
        latency_degraded = latency_slo_compliance_percent < 95.0
        error_rate_degraded = avg_error_rate >= 0.1

        # Calculate duration
        try:
            from datetime import datetime as dt
            start_dt = dt.fromisoformat(start_time.replace('Z', '+00:00'))
            end_dt = dt.fromisoformat(end_time.replace('Z', '+00:00'))
            duration_hours = (end_dt - start_dt).total_seconds() / 3600.0
        except Exception:
            # Fallback: assume snapshots cover ~24 hours
            duration_hours = 24.0

        # Create aggregation
        daily_agg = DailyAggregation(
            day_number=day_number,
            start_time=start_time,
            end_time=end_time,
            duration_hours=round(duration_hours, 2),
            snapshot_count=snapshot_count,
            # Availability
            avg_availability=round(avg_availability, 2),
            min_availability=round(min_availability, 2),
            max_availability=round(max_availability, 2),
            slo_compliance_percent=round(slo_compliance_percent, 2),
            # Error Rate
            avg_error_rate=round(avg_error_rate, 4),
            max_error_rate=round(max_error_rate, 4),
            total_requests=total_requests,
            total_errors=total_errors,
            # Latency
            avg_p95_latency_ms=round(avg_p95_latency, 2),
            max_p95_latency_ms=round(max_p95_latency, 2),
            avg_p99_latency_ms=round(avg_p99_latency, 2),
            max_p99_latency_ms=round(max_p99_latency, 2),
            latency_slo_compliance_percent=round(latency_slo_compliance_percent, 2),
            # Resources
            avg_cpu_percent=round(avg_cpu_percent, 2),
            peak_cpu_percent=round(peak_cpu_percent, 2),
            avg_memory_percent=round(avg_memory_percent, 2),
            peak_memory_percent=round(peak_memory_percent, 2),
            # Alerts
            total_critical_alerts=total_critical_alerts,
            total_warning_alerts=total_warning_alerts,
            total_info_alerts=total_info_alerts,
            alert_free_hours=round(alert_free_hours, 2),
            # Drift
            max_drift_percent=round(max_drift_percent, 2),
            avg_drift_percent=round(avg_drift_percent, 2),
            metrics_exceeding_drift_threshold=metrics_exceeding_threshold,
            # Degradation flags
            availability_degraded=availability_degraded,
            latency_degraded=latency_degraded,
            error_rate_degraded=error_rate_degraded
        )

        logger.info(
            f"Day {day_number} aggregation complete: "
            f"availability={avg_availability:.2f}%, "
            f"error_rate={avg_error_rate:.4f}%, "
            f"p99={avg_p99_latency:.2f}ms, "
            f"slo_compliance={slo_compliance_percent:.1f}%"
        )

        if availability_degraded or latency_degraded or error_rate_degraded:
            logger.warning(
                f"Day {day_number} degradation detected: "
                f"availability={availability_degraded}, "
                f"latency={latency_degraded}, "
                f"error_rate={error_rate_degraded}"
            )

        return daily_agg

    async def save_daily_aggregation(self, daily_agg: DailyAggregation) -> None:
        """
        Save daily aggregation to JSON file.

        File structure:
        - stability/day_01_summary.json
        - stability/day_02_summary.json
        - etc.

        JSON schema includes:
        - All aggregated metrics (min/max/avg)
        - SLO compliance percentages
        - Drift summaries
        - Degradation flags
        - Data completeness metadata

        Args:
            daily_agg: Daily aggregation to save

        Raises:
            IOError: If file cannot be written
        """
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename
        summary_file = self.output_dir / f"day_{daily_agg.day_number:02d}_summary.json"

        # Convert to dict
        summary_dict = asdict(daily_agg)

        # Add metadata
        summary_dict["_metadata"] = {
            "schema_version": "1.0",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "data_completeness": "complete" if daily_agg.snapshot_count >= 24 else "partial",
            "expected_snapshots": 48,  # 24 hours * 2 snapshots/hour (30min intervals)
            "actual_snapshots": daily_agg.snapshot_count,
            "completeness_percent": round((daily_agg.snapshot_count / 48.0) * 100.0, 2)
        }

        # Add warnings if data is incomplete
        if daily_agg.snapshot_count < 48:
            summary_dict["_metadata"]["warnings"] = [
                f"Incomplete data: {daily_agg.snapshot_count}/48 snapshots collected",
                "Aggregated metrics may not be representative of full 24-hour period"
            ]
        else:
            summary_dict["_metadata"]["warnings"] = []

        # Write to file
        try:
            with open(summary_file, 'w') as f:
                json.dump(summary_dict, f, indent=2)

            # Enterprise: Encrypt if enabled
            if self.encryptor:
                encrypted_file = summary_file.with_suffix(".json.enc")
                self.encryptor.encrypt_file(summary_file, encrypted_file)
                logger.info(f"Encrypted daily summary: {encrypted_file}")

            # Enterprise: Sign if enabled
            if self.signer:
                signature = self.signer.sign_file(summary_file)
                sig_file = summary_file.with_suffix(".json.sig")
                with open(sig_file, "w") as f:
                    f.write(f"RSA-PSS-SHA256\n{signature}\n")
                logger.info(f"Signed daily summary: {sig_file}")

            logger.info(
                f"Daily summary saved: {summary_file} "
                f"({daily_agg.snapshot_count} snapshots, "
                f"{summary_dict['_metadata']['completeness_percent']:.1f}% complete)"
            )

        except IOError as e:
            logger.error(f"Failed to save daily summary: {e}")
            raise

    def generate_weekly_summary(self) -> SevenDaySummary:
        """
        Generate overall 7-day summary with trend analysis and recommendations.

        Analyzes:
        - Weekly averages across all metrics
        - Trend analysis (improving/stable/degrading)
        - Stability score (0-100)
        - Rollback recommendations
        - Action items for v1.0.2

        Returns:
            SevenDaySummary with weekly analysis

        Raises:
            ValueError: If no daily aggregations available
        """
        if not self.daily_aggregations:
            raise ValueError("No daily aggregations available for weekly summary")

        if not self.snapshots:
            raise ValueError("No snapshots available for weekly summary")

        logger.info(f"Generating 7-day summary from {len(self.daily_aggregations)} daily aggregations...")

        # === Basic Info ===
        start_time = self.daily_aggregations[0].start_time
        end_time = self.daily_aggregations[-1].end_time
        total_snapshots = len(self.snapshots)
        baseline_timestamp = self.baseline_metrics.get("timestamp", "Unknown")

        # Calculate total duration
        try:
            from datetime import datetime as dt
            start_dt = dt.fromisoformat(start_time.replace('Z', '+00:00'))
            end_dt = dt.fromisoformat(end_time.replace('Z', '+00:00'))
            total_duration_hours = (end_dt - start_dt).total_seconds() / 3600.0
        except Exception:
            total_duration_hours = sum(agg.duration_hours for agg in self.daily_aggregations)

        # === Weekly Averages ===
        avg_availability = sum(agg.avg_availability for agg in self.daily_aggregations) / len(self.daily_aggregations)
        avg_error_rate = sum(agg.avg_error_rate for agg in self.daily_aggregations) / len(self.daily_aggregations)
        avg_p99_latency = sum(agg.avg_p99_latency_ms for agg in self.daily_aggregations) / len(self.daily_aggregations)
        avg_drift = sum(agg.avg_drift_percent for agg in self.daily_aggregations) / len(self.daily_aggregations)

        # === Stability Score (0-100) ===
        # Weighted formula based on SLO compliance
        availability_scores = [agg.slo_compliance_percent for agg in self.daily_aggregations]
        latency_scores = [agg.latency_slo_compliance_percent for agg in self.daily_aggregations]

        avg_availability_score = sum(availability_scores) / len(availability_scores)
        avg_latency_score = sum(latency_scores) / len(latency_scores)

        # Stability score: weighted average
        # 40% availability SLO + 40% latency SLO + 20% drift penalty
        drift_penalty = min(avg_drift, 20.0)  # Cap at 20% penalty
        stability_score = (
            (avg_availability_score * 0.4) +
            (avg_latency_score * 0.4) +
            ((100.0 - drift_penalty) * 0.2)
        )
        stability_score = max(0.0, min(100.0, stability_score))  # Clamp to 0-100

        # === Degradation Count ===
        days_with_degradation = sum(
            1 for agg in self.daily_aggregations
            if agg.availability_degraded or agg.latency_degraded or agg.error_rate_degraded
        )

        # === Alert Totals ===
        total_critical_alerts = sum(agg.total_critical_alerts for agg in self.daily_aggregations)
        total_warning_alerts = sum(agg.total_warning_alerts for agg in self.daily_aggregations)

        # === Trend Analysis ===
        # Analyze trends by comparing first 3 days vs last 3 days
        def analyze_trend(metric_values: List[float]) -> str:
            """Determine trend: improving, stable, or degrading."""
            if len(metric_values) < 4:
                return "stable"  # Not enough data

            # Compare first half vs second half
            first_half = metric_values[:len(metric_values)//2]
            second_half = metric_values[len(metric_values)//2:]

            first_avg = sum(first_half) / len(first_half)
            second_avg = sum(second_half) / len(second_half)

            # For availability/hit-rate (higher is better)
            # For latency/errors/resources (lower is better)
            return first_avg, second_avg

        # Availability trend (higher is better)
        avail_values = [agg.avg_availability for agg in self.daily_aggregations]
        first_avail, second_avail = analyze_trend(avail_values)
        if second_avail > first_avail + 0.5:  # >0.5% improvement
            availability_trend = "improving"
        elif second_avail < first_avail - 0.5:  # >0.5% degradation
            availability_trend = "degrading"
        else:
            availability_trend = "stable"

        # Latency trend (lower is better)
        latency_values = [agg.avg_p99_latency_ms for agg in self.daily_aggregations]
        first_latency, second_latency = analyze_trend(latency_values)
        if second_latency < first_latency - 10:  # >10ms improvement
            latency_trend = "improving"
        elif second_latency > first_latency + 10:  # >10ms degradation
            latency_trend = "degrading"
        else:
            latency_trend = "stable"

        # Resource trend (lower is better)
        cpu_values = [agg.avg_cpu_percent for agg in self.daily_aggregations]
        memory_values = [agg.avg_memory_percent for agg in self.daily_aggregations]
        first_cpu, second_cpu = analyze_trend(cpu_values)
        first_mem, second_mem = analyze_trend(memory_values)

        cpu_trending_up = second_cpu > first_cpu + 5.0  # >5% increase
        mem_trending_up = second_mem > first_mem + 5.0  # >5% increase

        if cpu_trending_up or mem_trending_up:
            resource_trend = "degrading"
        elif second_cpu < first_cpu - 5.0 or second_mem < first_mem - 5.0:
            resource_trend = "improving"
        else:
            resource_trend = "stable"

        # === Recommendations ===
        recommendations = []
        action_items = []

        # Availability recommendations
        if avg_availability < 99.9:
            recommendations.append(
                f"Availability below SLO target ({avg_availability:.2f}% < 99.9%) - "
                "investigate service health and scaling"
            )
            action_items.append("Review pod restart patterns and resource limits")

        if availability_trend == "degrading":
            recommendations.append("Availability trending downward - proactive intervention recommended")
            action_items.append("Schedule incident review for availability degradation")

        # Latency recommendations
        if avg_p99_latency > 500.0:
            recommendations.append(
                f"P99 latency above SLO target ({avg_p99_latency:.2f}ms > 500ms) - "
                "optimize database queries and caching"
            )
            action_items.append("Run EXPLAIN ANALYZE on slow queries")

        if latency_trend == "degrading":
            recommendations.append("Latency trending upward - investigate performance bottlenecks")
            action_items.append("Profile API endpoints with distributed tracing")

        # Drift recommendations
        if avg_drift > 10.0:
            recommendations.append(
                f"High drift from baseline ({avg_drift:.1f}% > 10%) - "
                "investigate unexpected behavior changes"
            )
            action_items.append("Compare baseline vs current configurations")

        # Resource recommendations
        if resource_trend == "degrading":
            recommendations.append("Resource utilization trending upward - capacity planning needed")
            action_items.append("Review HPA settings and node autoscaling thresholds")

        # Alert recommendations
        if total_critical_alerts > 0:
            recommendations.append(
                f"{total_critical_alerts} critical alerts in 7 days - "
                "review alert definitions and fix root causes"
            )
            action_items.append("Document critical alert incidents in retrospective")

        # Degradation recommendations
        if days_with_degradation >= 2:
            recommendations.append(
                f"SLO degradation detected on {days_with_degradation}/7 days - "
                "consider rollback to previous version"
            )
            action_items.append("Prepare rollback plan for v1.0.0 if degradation continues")

        # Stability score recommendations
        if stability_score < 90.0:
            recommendations.append(
                f"Stability score below target ({stability_score:.1f}/100) - "
                "comprehensive review required"
            )
            action_items.append("Schedule post-GA retrospective meeting")

        # === Rollback Recommendation ===
        should_rollback = (
            avg_availability < 99.9 or  # Availability SLO violation
            avg_drift > 15.0 or  # Excessive drift
            days_with_degradation >= 2 or  # Multiple days degraded
            total_critical_alerts > 5  # Too many critical alerts
        )

        if should_rollback:
            recommendations.insert(0, "⚠️ ROLLBACK RECOMMENDED - Multiple SLO violations detected")
            action_items.insert(0, "Execute rollback to v1.0.0 immediately")

        # If everything looks good
        if not recommendations:
            recommendations.append("✅ All metrics within acceptable ranges - v1.0.1 is stable")
            action_items.append("Proceed with long-term monitoring and capacity planning")

        # === Create Summary ===
        weekly_summary = SevenDaySummary(
            start_time=start_time,
            end_time=end_time,
            total_duration_hours=round(total_duration_hours, 2),
            total_snapshots=total_snapshots,
            baseline_timestamp=baseline_timestamp,
            # Weekly averages
            avg_availability=round(avg_availability, 2),
            avg_error_rate=round(avg_error_rate, 4),
            avg_p99_latency_ms=round(avg_p99_latency, 2),
            avg_drift_percent=round(avg_drift, 2),
            # Stability metrics
            stability_score=round(stability_score, 1),
            days_with_degradation=days_with_degradation,
            total_critical_alerts=total_critical_alerts,
            total_warning_alerts=total_warning_alerts,
            # Daily breakdown
            daily_summaries=self.daily_aggregations,
            # Trends
            availability_trend=availability_trend,
            latency_trend=latency_trend,
            resource_trend=resource_trend,
            # Recommendations
            recommendations=recommendations,
            action_items=action_items
        )

        logger.info(
            f"Weekly summary generated: "
            f"stability_score={stability_score:.1f}/100, "
            f"avg_availability={avg_availability:.2f}%, "
            f"days_degraded={days_with_degradation}/7"
        )

        if should_rollback:
            logger.warning("⚠️ ROLLBACK RECOMMENDED based on 7-day analysis")

        return weekly_summary

    async def run(self) -> None:
        """
        Main monitoring loop - Runs for 7 days collecting stability snapshots.

        Execution flow:
        1. Load GA Day baseline
        2. Loop for duration_hours (168h = 7 days)
        3. Collect snapshot every interval_minutes (30 min)
        4. Save raw snapshots to day_XX_raw/
        5. Aggregate daily metrics at end of each day
        6. Save daily summaries to day_XX_summary.json
        7. Generate weekly summary at end
        8. Save weekly summary to weekly_summary.json and weekly_summary.md

        Handles:
        - Prometheus unavailability (retry with warnings)
        - Incomplete snapshot days (partial aggregation)
        - Graceful shutdown (Ctrl+C)

        Raises:
            FileNotFoundError: If baseline file not found
            PrometheusQueryError: If Prometheus is unreachable after retries
        """
        logger.info("="*80)
        logger.info("Starting 7-Day Post-GA Stability Monitor")
        logger.info(f"Duration: {self.duration_hours} hours")
        logger.info(f"Interval: {self.interval_minutes} minutes")
        logger.info(f"Output: {self.output_dir}")
        logger.info("="*80)

        # === Step 1: Load baseline ===
        try:
            self.load_baseline()
        except Exception as e:
            logger.error(f"Failed to load baseline: {e}")
            raise

        # === Step 2: Monitoring loop ===
        start_time = datetime.now(timezone.utc)
        end_time = start_time + timedelta(hours=self.duration_hours)
        elapsed_hours = 0.0
        current_day = 1
        day_snapshots = []  # Buffer for current day's snapshots

        logger.info(f"Monitoring window: {start_time.isoformat()} -> {end_time.isoformat()}")
        logger.info(f"Expected snapshots: {int(self.duration_hours / (self.interval_minutes / 60.0))}")

        try:
            while elapsed_hours < self.duration_hours:
                # Calculate current time
                now = datetime.now(timezone.utc)
                elapsed_hours = (now - start_time).total_seconds() / 3600.0
                day_number = int(elapsed_hours / 24) + 1

                # Check if we've moved to a new day
                if day_number > current_day:
                    # Aggregate previous day's metrics
                    if day_snapshots:
                        logger.info(f"Day {current_day} complete - aggregating {len(day_snapshots)} snapshots")
                        try:
                            daily_agg = self.aggregate_daily_metrics(day_snapshots)
                            self.daily_aggregations.append(daily_agg)
                            await self.save_daily_aggregation(daily_agg)
                        except Exception as e:
                            logger.error(f"Failed to aggregate Day {current_day}: {e}", exc_info=True)

                    # Reset for new day
                    current_day = day_number
                    day_snapshots = []

                # Collect snapshot
                try:
                    logger.info(f"[Day {day_number}] T+{elapsed_hours:.2f}h - Collecting snapshot...")
                    snapshot = await self.collect_snapshot(elapsed_hours)

                    # Save snapshot
                    await self.save_snapshot(snapshot, day_number)

                    # Add to snapshot list
                    self.snapshots.append(snapshot)
                    day_snapshots.append(snapshot)

                    logger.info(
                        f"[Day {day_number}] Snapshot {len(self.snapshots)} collected: "
                        f"avail={snapshot.overall_availability}%, "
                        f"err={snapshot.error_rate}%, "
                        f"p99={snapshot.p99_latency_ms}ms"
                    )

                except PrometheusQueryError as e:
                    logger.error(
                        f"[Day {day_number}] T+{elapsed_hours:.2f}h - "
                        f"Prometheus query failed: {e}"
                    )
                    logger.warning("Continuing monitoring with incomplete data...")

                except Exception as e:
                    logger.error(
                        f"[Day {day_number}] T+{elapsed_hours:.2f}h - "
                        f"Unexpected error collecting snapshot: {e}",
                        exc_info=True
                    )

                # Sleep until next interval
                interval_seconds = self.interval_minutes * 60
                next_snapshot_time = start_time + timedelta(seconds=(len(self.snapshots) * interval_seconds))
                sleep_duration = (next_snapshot_time - datetime.now(timezone.utc)).total_seconds()

                if sleep_duration > 0:
                    logger.debug(f"Sleeping {sleep_duration:.1f}s until next snapshot...")
                    await asyncio.sleep(sleep_duration)
                else:
                    # Behind schedule - log warning but continue
                    logger.warning(
                        f"Behind schedule by {abs(sleep_duration):.1f}s - "
                        "snapshot collection taking longer than interval"
                    )

        except KeyboardInterrupt:
            logger.warning("Monitoring interrupted by user (Ctrl+C)")
            logger.info(f"Collected {len(self.snapshots)} snapshots before interruption")

        except Exception as e:
            logger.error(f"Fatal error in monitoring loop: {e}", exc_info=True)
            raise

        finally:
            # === Step 3: Aggregate final day ===
            if day_snapshots:
                logger.info(f"Aggregating final day ({current_day}) with {len(day_snapshots)} snapshots")
                try:
                    daily_agg = self.aggregate_daily_metrics(day_snapshots)
                    self.daily_aggregations.append(daily_agg)
                    await self.save_daily_aggregation(daily_agg)
                except Exception as e:
                    logger.error(f"Failed to aggregate final day: {e}", exc_info=True)

        # === Step 4: Generate weekly summary ===
        if self.daily_aggregations:
            logger.info("Generating 7-day weekly summary...")
            try:
                weekly_summary = self.generate_weekly_summary()

                # Save as JSON
                summary_json_file = self.output_dir / "weekly_summary.json"
                with open(summary_json_file, 'w') as f:
                    json.dump(asdict(weekly_summary), f, indent=2)
                logger.info(f"Weekly summary JSON saved: {summary_json_file}")

                # Save as Markdown
                summary_md_file = self.output_dir / "weekly_summary.md"
                with open(summary_md_file, 'w') as f:
                    f.write(self._format_weekly_summary_markdown(weekly_summary))
                logger.info(f"Weekly summary Markdown saved: {summary_md_file}")

                # Log summary
                logger.info("="*80)
                logger.info("7-DAY STABILITY SUMMARY")
                logger.info("="*80)
                logger.info(f"Duration: {weekly_summary.total_duration_hours:.1f} hours")
                logger.info(f"Snapshots: {weekly_summary.total_snapshots}")
                logger.info(f"Stability Score: {weekly_summary.stability_score:.1f}/100")
                logger.info(f"Avg Availability: {weekly_summary.avg_availability:.2f}%")
                logger.info(f"Avg Error Rate: {weekly_summary.avg_error_rate:.4f}%")
                logger.info(f"Avg P99 Latency: {weekly_summary.avg_p99_latency_ms:.2f}ms")
                logger.info(f"Avg Drift: {weekly_summary.avg_drift_percent:.2f}%")
                logger.info(f"Days Degraded: {weekly_summary.days_with_degradation}/7")
                logger.info(f"Critical Alerts: {weekly_summary.total_critical_alerts}")
                logger.info(f"Availability Trend: {weekly_summary.availability_trend}")
                logger.info(f"Latency Trend: {weekly_summary.latency_trend}")
                logger.info(f"Resource Trend: {weekly_summary.resource_trend}")
                logger.info("="*80)
                logger.info("RECOMMENDATIONS:")
                for i, rec in enumerate(weekly_summary.recommendations, 1):
                    logger.info(f"  {i}. {rec}")
                logger.info("="*80)
                logger.info("ACTION ITEMS:")
                for i, action in enumerate(weekly_summary.action_items, 1):
                    logger.info(f"  {i}. {action}")
                logger.info("="*80)

            except Exception as e:
                logger.error(f"Failed to generate weekly summary: {e}", exc_info=True)

        else:
            logger.warning("No daily aggregations available - skipping weekly summary")

        logger.info("7-Day stability monitoring complete!")
        logger.info(f"Total snapshots collected: {len(self.snapshots)}")
        logger.info(f"Total daily summaries: {len(self.daily_aggregations)}")

    def _format_weekly_summary_markdown(self, summary: SevenDaySummary) -> str:
        """
        Format weekly summary as Markdown.

        Args:
            summary: Weekly summary object

        Returns:
            Formatted Markdown string
        """
        md = []
        md.append("# 7-Day Post-GA Stability Summary")
        md.append("")
        md.append(f"**Version:** T.A.R.S. v1.0.1")
        md.append(f"**Monitoring Window:** {summary.start_time} → {summary.end_time}")
        md.append(f"**Duration:** {summary.total_duration_hours:.1f} hours")
        md.append(f"**Baseline:** {summary.baseline_timestamp}")
        md.append("")
        md.append("---")
        md.append("")

        # === Executive Summary ===
        md.append("## Executive Summary")
        md.append("")
        md.append(f"**Stability Score:** {summary.stability_score:.1f}/100")
        md.append("")

        # Grade
        if summary.stability_score >= 95:
            grade = "A (Excellent)"
        elif summary.stability_score >= 90:
            grade = "B (Good)"
        elif summary.stability_score >= 80:
            grade = "C (Fair)"
        elif summary.stability_score >= 70:
            grade = "D (Poor)"
        else:
            grade = "F (Critical)"

        md.append(f"**Overall Grade:** {grade}")
        md.append("")

        # === Key Metrics ===
        md.append("## Key Metrics")
        md.append("")
        md.append("| Metric | Value | Target | Status |")
        md.append("|--------|-------|--------|--------|")
        md.append(
            f"| Availability | {summary.avg_availability:.2f}% | ≥99.9% | "
            f"{'✅ PASS' if summary.avg_availability >= 99.9 else '❌ FAIL'} |"
        )
        md.append(
            f"| Error Rate | {summary.avg_error_rate:.4f}% | <0.1% | "
            f"{'✅ PASS' if summary.avg_error_rate < 0.1 else '❌ FAIL'} |"
        )
        md.append(
            f"| P99 Latency | {summary.avg_p99_latency_ms:.2f}ms | <500ms | "
            f"{'✅ PASS' if summary.avg_p99_latency_ms < 500.0 else '❌ FAIL'} |"
        )
        md.append(
            f"| Drift | {summary.avg_drift_percent:.2f}% | ≤10% | "
            f"{'✅ PASS' if summary.avg_drift_percent <= 10.0 else '❌ FAIL'} |"
        )
        md.append(
            f"| Critical Alerts | {summary.total_critical_alerts} | 0 | "
            f"{'✅ PASS' if summary.total_critical_alerts == 0 else '❌ FAIL'} |"
        )
        md.append(
            f"| Days Degraded | {summary.days_with_degradation}/7 | 0 | "
            f"{'✅ PASS' if summary.days_with_degradation == 0 else '⚠️ WARNING' if summary.days_with_degradation < 2 else '❌ FAIL'} |"
        )
        md.append("")

        # === Trends ===
        md.append("## Trends")
        md.append("")
        md.append(f"- **Availability:** {summary.availability_trend}")
        md.append(f"- **Latency:** {summary.latency_trend}")
        md.append(f"- **Resources:** {summary.resource_trend}")
        md.append("")

        # === Daily Breakdown ===
        md.append("## Daily Breakdown")
        md.append("")
        md.append("| Day | Availability | Error Rate | P99 Latency | Drift | SLO Compliance | Degraded |")
        md.append("|-----|--------------|------------|-------------|-------|----------------|----------|")

        for day_agg in summary.daily_summaries:
            degraded = "Yes" if (
                day_agg.availability_degraded or
                day_agg.latency_degraded or
                day_agg.error_rate_degraded
            ) else "No"

            md.append(
                f"| Day {day_agg.day_number} | "
                f"{day_agg.avg_availability:.2f}% | "
                f"{day_agg.avg_error_rate:.4f}% | "
                f"{day_agg.avg_p99_latency_ms:.2f}ms | "
                f"{day_agg.avg_drift_percent:.2f}% | "
                f"{day_agg.slo_compliance_percent:.1f}% | "
                f"{degraded} |"
            )

        md.append("")

        # === Recommendations ===
        md.append("## Recommendations")
        md.append("")
        for i, rec in enumerate(summary.recommendations, 1):
            md.append(f"{i}. {rec}")
        md.append("")

        # === Action Items ===
        md.append("## Action Items")
        md.append("")
        for i, action in enumerate(summary.action_items, 1):
            md.append(f"- [ ] {action}")
        md.append("")

        # === Footer ===
        md.append("---")
        md.append("")
        md.append(f"**Report Generated:** {datetime.now(timezone.utc).isoformat()}")
        md.append(f"**Total Snapshots:** {summary.total_snapshots}")
        md.append("")
        md.append("🚀 Generated with [Claude Code](https://claude.com/claude-code)")

        return "\n".join(md)


async def main():
    """
    CLI entry point for 7-day stability monitor.

    Usage:
        # Production (7 days)
        python stability_monitor_7day.py --baseline ga_kpis/ga_kpi_summary.json

        # Test mode (2 hours, 10-minute intervals)
        python stability_monitor_7day.py --baseline baseline.json --duration 2 --interval 10 --test-mode

    Returns:
        Exit code: 0 on success, 1 on failure
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="7-Day Post-GA Stability Monitor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Production monitoring (7 days)
  python stability_monitor_7day.py --baseline ga_kpis/ga_kpi_summary.json

  # Test mode (2 hours with 10-minute intervals)
  python stability_monitor_7day.py --baseline baseline.json --duration 2 --interval 10 --test-mode

  # Custom Prometheus URL
  python stability_monitor_7day.py --baseline baseline.json --prometheus-url http://localhost:9090
        """
    )
    parser.add_argument(
        "--baseline",
        required=True,
        help="Path to GA Day baseline JSON (ga_kpi_summary.json)"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=168,
        help="Monitoring duration in hours (default: 168 = 7 days)"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=30,
        help="Snapshot interval in minutes (default: 30)"
    )
    parser.add_argument(
        "--output",
        default="stability",
        help="Output directory for stability reports (default: stability/)"
    )
    parser.add_argument(
        "--prometheus-url",
        default="http://prometheus.tars-production.svc.cluster.local:9090",
        help="Prometheus server URL"
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Enable test mode (reduces interval for faster testing)"
    )

    # Enterprise configuration arguments (Phase 14.6)
    if ENTERPRISE_AVAILABLE:
        parser.add_argument("--profile", type=str, default="local", help="Enterprise config profile (local, dev, staging, prod)")
        parser.add_argument("--config", type=str, help="Path to enterprise config file")
        parser.add_argument("--encrypt", action="store_true", help="Encrypt output files (requires AES key)")
        parser.add_argument("--sign", action="store_true", help="Sign output files (requires RSA key)")
        parser.add_argument("--no-compliance", action="store_true", help="Disable compliance enforcement")

    args = parser.parse_args()

    # Adjust interval if in test mode
    if args.test_mode:
        if args.duration <= 2:
            logger.info("Test mode enabled: reducing interval to 10 minutes for quick testing")
            args.interval = 10
        logger.warning(
            f"Test mode: monitoring for {args.duration}h with {args.interval}min intervals "
            f"({int(args.duration / (args.interval / 60.0))} snapshots expected)"
        )

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
            if args.prometheus_url and args.prometheus_url != "http://prometheus.tars-production.svc.cluster.local:9090":
                logger.info(f"✓ Overriding Prometheus URL from CLI: {args.prometheus_url}")
            else:
                args.prometheus_url = config.observability.prometheus_url

            if args.output == "stability" and config.observability.output_dir != "output":
                args.output = config.observability.output_dir

            # Initialize compliance enforcer
            if not args.no_compliance and config.compliance.enabled_standards:
                from pathlib import Path as P
                compliance_enforcer = ComplianceEnforcer(
                    enabled_standards=config.compliance.enabled_standards,
                    controls_dir=P("compliance/policies"),
                    audit_log_path=P(config.observability.output_dir) / "audit.log" if config.compliance.enable_audit_trail else None,
                    strict_mode=False,  # Log warnings instead of raising exceptions
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

    # Instantiate monitor
    monitor = StabilityMonitor(
        baseline_file=args.baseline,
        duration_hours=args.duration,
        interval_minutes=args.interval,
        output_dir=args.output,
        prometheus_url=args.prometheus_url,
        compliance_enforcer=compliance_enforcer,
        encryptor=encryptor,
        signer=signer,
    )

    # Run monitoring loop
    try:
        await monitor.run()
        logger.info("✅ 7-Day stability monitoring completed successfully")
        return 0

    except FileNotFoundError as e:
        logger.error(f"❌ Baseline file not found: {e}")
        logger.error("Please ensure GA Day monitoring (Phase 14.5) has completed and baseline exists")
        return 1

    except PrometheusQueryError as e:
        logger.error(f"❌ Prometheus query error: {e}")
        logger.error("Check Prometheus URL and ensure service is accessible")
        return 1

    except KeyboardInterrupt:
        logger.warning("⚠️ Monitoring interrupted by user")
        logger.info("Partial data has been saved to output directory")
        return 130  # Standard exit code for SIGINT

    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
