#!/usr/bin/env python3
"""
Production Drift Detector - Phase 14.4/14.5

Detects drift from baseline metrics in production environment.
Monitors configuration, resource utilization, performance, and behavior drift.

Usage:
    python drift_detector.py --baseline baseline_metrics.json --duration 24 --output drift_analysis.json
    python drift_detector.py --baseline-file staging_snapshot.json --check-interval 300

Author: T.A.R.S. Platform Team
Phase: 14.4/14.5 - GA Day Monitoring & Drift Detection
"""

import asyncio
import hashlib
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiohttp

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class DriftMetric:
    """Individual drift metric"""
    name: str
    category: str  # config, resource, performance, behavior
    baseline_value: float
    current_value: float
    drift_percent: float
    threshold_percent: float
    status: str  # ok, warning, critical
    timestamp: str


@dataclass
class DriftCheck:
    """A single drift detection check"""
    timestamp: str
    elapsed_hours: float
    total_checks: int
    drifts_detected: int
    critical_drifts: int
    warning_drifts: int
    metrics: List[DriftMetric] = field(default_factory=list)


@dataclass
class DriftSummary:
    """Overall drift analysis summary"""
    baseline_timestamp: str
    start_time: str
    end_time: str
    duration_hours: float
    total_checks: int
    total_drifts: int
    critical_drifts: int
    warning_drifts: int
    ok_checks: int
    drift_categories: Dict[str, int] = field(default_factory=dict)
    top_drifts: List[DriftMetric] = field(default_factory=list)
    mitigation_actions: List[str] = field(default_factory=list)


class PrometheusClient:
    """Async Prometheus client for drift detection"""

    def __init__(self, base_url: str = "http://localhost:9090"):
        self.base_url = base_url.rstrip("/")
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def query(self, query: str) -> Optional[Dict[str, Any]]:
        """Execute a PromQL query"""
        if not self.session:
            raise RuntimeError("Client not initialized")

        try:
            url = f"{self.base_url}/api/v1/query"
            params = {"query": query}

            async with self.session.get(url, params=params, timeout=10) as resp:
                if resp.status != 200:
                    logger.error(f"Prometheus query failed: {resp.status}")
                    return None

                data = await resp.json()
                if data.get("status") != "success":
                    logger.error(f"Prometheus query error: {data}")
                    return None

                return data.get("data", {})

        except asyncio.TimeoutError:
            logger.error(f"Prometheus query timeout: {query}")
            return None
        except Exception as e:
            logger.error(f"Prometheus query exception: {e}")
            return None


class DriftDetector:
    """Detects drift from baseline metrics"""

    def __init__(
        self,
        baseline_file: Path,
        prometheus_url: str = "http://localhost:9090",
        output_file: Path = Path("drift_analysis.json"),
        check_interval_seconds: int = 300,
        # Drift thresholds
        warning_threshold: float = 5.0,  # 5% drift
        critical_threshold: float = 10.0,  # 10% drift
    ):
        self.baseline_file = baseline_file
        self.prometheus_url = prometheus_url
        self.output_file = output_file
        self.check_interval_seconds = check_interval_seconds
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold

        self.baseline: Dict[str, Any] = {}
        self.checks: List[DriftCheck] = []
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None

        # Drift detection rules
        self.drift_rules = {
            # Resource drift (10% threshold)
            "cpu_usage": {"category": "resource", "threshold": 10.0},
            "memory_usage": {"category": "resource", "threshold": 10.0},
            "disk_usage": {"category": "resource", "threshold": 10.0},
            "network_throughput": {"category": "resource", "threshold": 15.0},

            # Performance drift (5% threshold)
            "api_latency_p95": {"category": "performance", "threshold": 5.0},
            "api_latency_p99": {"category": "performance", "threshold": 5.0},
            "db_latency_p95": {"category": "performance", "threshold": 5.0},
            "error_rate": {"category": "performance", "threshold": 50.0},  # 50% change allowed

            # Configuration drift (0% threshold - strict)
            "replica_count": {"category": "config", "threshold": 0.0},
            "resource_limits": {"category": "config", "threshold": 0.0},

            # Behavior drift (10% threshold)
            "request_rate": {"category": "behavior", "threshold": 10.0},
            "cache_hit_rate": {"category": "behavior", "threshold": 10.0},
        }

    def load_baseline(self):
        """Load baseline metrics from file"""
        logger.info(f"Loading baseline from {self.baseline_file}")

        if not self.baseline_file.exists():
            raise FileNotFoundError(f"Baseline file not found: {self.baseline_file}")

        with open(self.baseline_file, "r") as f:
            self.baseline = json.load(f)

        logger.info(f"Baseline loaded: {len(self.baseline)} metrics")

    async def check_drift(self) -> DriftCheck:
        """Perform drift detection check"""
        logger.info("Checking for drift...")

        timestamp = datetime.now(timezone.utc).isoformat()
        elapsed_hours = 0.0
        if self.start_time:
            elapsed = datetime.now(timezone.utc) - self.start_time
            elapsed_hours = elapsed.total_seconds() / 3600.0

        async with PrometheusClient(self.prometheus_url) as prom:
            drift_metrics = []

            # Check CPU drift
            cpu_current = await self._query_metric(prom, "avg(rate(container_cpu_usage_seconds_total[5m])) * 100")
            if cpu_current is not None and "cpu_usage" in self.baseline:
                drift_metric = self._calculate_drift(
                    "cpu_usage",
                    self.baseline["cpu_usage"],
                    cpu_current,
                    self.drift_rules["cpu_usage"]["threshold"]
                )
                drift_metrics.append(drift_metric)

            # Check Memory drift
            mem_current = await self._query_metric(prom, "sum(container_memory_working_set_bytes) / sum(machine_memory_bytes) * 100")
            if mem_current is not None and "memory_usage" in self.baseline:
                drift_metric = self._calculate_drift(
                    "memory_usage",
                    self.baseline["memory_usage"],
                    mem_current,
                    self.drift_rules["memory_usage"]["threshold"]
                )
                drift_metrics.append(drift_metric)

            # Check API latency drift
            api_p95_current = await self._query_metric(prom, "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) * 1000")
            if api_p95_current is not None and "api_latency_p95" in self.baseline:
                drift_metric = self._calculate_drift(
                    "api_latency_p95",
                    self.baseline["api_latency_p95"],
                    api_p95_current,
                    self.drift_rules["api_latency_p95"]["threshold"]
                )
                drift_metrics.append(drift_metric)

            # Check error rate drift
            error_rate_current = await self._query_metric(prom, "rate(http_requests_total{status=~\"5..\" }[5m]) * 100")
            if error_rate_current is not None and "error_rate" in self.baseline:
                drift_metric = self._calculate_drift(
                    "error_rate",
                    self.baseline["error_rate"],
                    error_rate_current if error_rate_current > 0 else 0.01,  # Avoid div by zero
                    self.drift_rules["error_rate"]["threshold"]
                )
                drift_metrics.append(drift_metric)

            # Check request rate drift
            req_rate_current = await self._query_metric(prom, "sum(rate(http_requests_total[5m]))")
            if req_rate_current is not None and "request_rate" in self.baseline:
                drift_metric = self._calculate_drift(
                    "request_rate",
                    self.baseline["request_rate"],
                    req_rate_current,
                    self.drift_rules["request_rate"]["threshold"]
                )
                drift_metrics.append(drift_metric)

            # Check cache hit rate drift
            cache_hit_current = await self._query_metric(prom, "redis_keyspace_hits_total / (redis_keyspace_hits_total + redis_keyspace_misses_total) * 100")
            if cache_hit_current is not None and "cache_hit_rate" in self.baseline:
                drift_metric = self._calculate_drift(
                    "cache_hit_rate",
                    self.baseline["cache_hit_rate"],
                    cache_hit_current,
                    self.drift_rules["cache_hit_rate"]["threshold"]
                )
                drift_metrics.append(drift_metric)

        # Count drifts
        critical_count = sum(1 for m in drift_metrics if m.status == "critical")
        warning_count = sum(1 for m in drift_metrics if m.status == "warning")
        total_drifts = critical_count + warning_count

        check = DriftCheck(
            timestamp=timestamp,
            elapsed_hours=round(elapsed_hours, 2),
            total_checks=len(drift_metrics),
            drifts_detected=total_drifts,
            critical_drifts=critical_count,
            warning_drifts=warning_count,
            metrics=drift_metrics
        )

        self.checks.append(check)

        logger.info(f"Drift check complete: {total_drifts} drifts ({critical_count} critical, {warning_count} warning)")

        return check

    async def _query_metric(self, prom: PrometheusClient, query: str) -> Optional[float]:
        """Query a single metric from Prometheus"""
        result = await prom.query(query)

        if not result or not result.get("result"):
            return None

        try:
            value = float(result["result"][0]["value"][1])
            return value
        except (IndexError, KeyError, ValueError):
            return None

    def _calculate_drift(
        self,
        name: str,
        baseline: float,
        current: float,
        threshold: float
    ) -> DriftMetric:
        """Calculate drift for a metric"""
        if baseline == 0:
            drift_percent = 100.0 if current != 0 else 0.0
        else:
            drift_percent = abs((current - baseline) / baseline * 100.0)

        # Determine status
        if drift_percent >= self.critical_threshold or drift_percent >= threshold:
            status = "critical"
        elif drift_percent >= self.warning_threshold:
            status = "warning"
        else:
            status = "ok"

        category = self.drift_rules.get(name, {}).get("category", "unknown")

        return DriftMetric(
            name=name,
            category=category,
            baseline_value=round(baseline, 2),
            current_value=round(current, 2),
            drift_percent=round(drift_percent, 2),
            threshold_percent=threshold,
            status=status,
            timestamp=datetime.now(timezone.utc).isoformat()
        )

    def calculate_summary(self) -> DriftSummary:
        """Calculate overall drift summary"""
        if not self.checks or not self.start_time or not self.end_time:
            raise ValueError("No drift checks completed")

        duration_hours = (self.end_time - self.start_time).total_seconds() / 3600.0

        total_checks = sum(c.total_checks for c in self.checks)
        total_drifts = sum(c.drifts_detected for c in self.checks)
        critical_drifts = sum(c.critical_drifts for c in self.checks)
        warning_drifts = sum(c.warning_drifts for c in self.checks)
        ok_checks = total_checks - total_drifts

        # Count drifts by category
        drift_categories: Dict[str, int] = {}
        all_drift_metrics: List[DriftMetric] = []

        for check in self.checks:
            for metric in check.metrics:
                if metric.status != "ok":
                    all_drift_metrics.append(metric)
                    drift_categories[metric.category] = drift_categories.get(metric.category, 0) + 1

        # Get top 10 drifts
        top_drifts = sorted(all_drift_metrics, key=lambda m: m.drift_percent, reverse=True)[:10]

        # Generate mitigation actions
        mitigation_actions = []
        if critical_drifts > 0:
            mitigation_actions.append("Review and address critical drift alerts immediately")
        if any(d.category == "performance" and d.status == "critical" for d in all_drift_metrics):
            mitigation_actions.append("Performance regression detected - investigate P95/P99 latency increases")
        if any(d.name == "error_rate" and d.status == "critical" for d in all_drift_metrics):
            mitigation_actions.append("Error rate spike detected - check application logs and alerts")
        if any(d.category == "config" and d.status != "ok" for d in all_drift_metrics):
            mitigation_actions.append("Configuration drift detected - verify deployment configuration matches baseline")
        if not mitigation_actions:
            mitigation_actions.append("All metrics within acceptable drift thresholds - no action required")

        return DriftSummary(
            baseline_timestamp=self.baseline.get("timestamp", "Unknown"),
            start_time=self.start_time.isoformat(),
            end_time=self.end_time.isoformat(),
            duration_hours=round(duration_hours, 2),
            total_checks=total_checks,
            total_drifts=total_drifts,
            critical_drifts=critical_drifts,
            warning_drifts=warning_drifts,
            ok_checks=ok_checks,
            drift_categories=drift_categories,
            top_drifts=top_drifts,
            mitigation_actions=mitigation_actions
        )

    def save_check(self, check: DriftCheck):
        """Save drift check to file"""
        self.output_file.parent.mkdir(parents=True, exist_ok=True)

        # Save incremental check
        check_file = self.output_file.parent / f"drift_check_{len(self.checks):04d}.json"
        with open(check_file, "w") as f:
            json.dump({
                "timestamp": check.timestamp,
                "elapsed_hours": check.elapsed_hours,
                "total_checks": check.total_checks,
                "drifts_detected": check.drifts_detected,
                "critical_drifts": check.critical_drifts,
                "warning_drifts": check.warning_drifts,
                "metrics": [asdict(m) for m in check.metrics]
            }, f, indent=2)

        logger.info(f"Drift check saved: {check_file}")

    def save_summary(self, summary: DriftSummary):
        """Save final drift summary"""
        self.output_file.parent.mkdir(parents=True, exist_ok=True)

        summary_dict = asdict(summary)

        with open(self.output_file, "w") as f:
            json.dump(summary_dict, f, indent=2)

        logger.info(f"Drift summary saved: {self.output_file}")

        # Generate SHA256 hash
        with open(self.output_file, "rb") as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()

        hash_file = self.output_file.parent / f"{self.output_file.name}.sha256"
        with open(hash_file, "w") as f:
            f.write(f"{file_hash}  {self.output_file.name}\n")

        logger.info(f"Hash saved: {hash_file}")

    async def run_detection(
        self,
        duration_hours: float = 24.0,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ):
        """Run drift detection for specified duration"""
        self.load_baseline()

        self.start_time = start_time or datetime.now(timezone.utc)
        self.end_time = end_time or (self.start_time + timedelta(hours=duration_hours))

        logger.info(f"Starting drift detection from {self.start_time} to {self.end_time}")
        logger.info(f"Check interval: {self.check_interval_seconds} seconds")

        check_num = 1

        while datetime.now(timezone.utc) < self.end_time:
            try:
                # Perform drift check
                check = await self.check_drift()

                # Save check
                self.save_check(check)

                # Log critical drifts
                if check.critical_drifts > 0:
                    logger.warning(f"⚠️  CRITICAL: {check.critical_drifts} critical drifts detected!")

                check_num += 1

                # Wait for next interval
                await asyncio.sleep(self.check_interval_seconds)

            except KeyboardInterrupt:
                logger.warning("Drift detection interrupted by user")
                break
            except Exception as e:
                logger.error(f"Error during drift detection: {e}", exc_info=True)
                await asyncio.sleep(60)  # Wait 1 minute before retrying

        logger.info("Drift detection complete")

        # Generate summary
        logger.info("Generating summary...")
        summary = self.calculate_summary()
        self.save_summary(summary)

        logger.info(f"Drift detection complete. Total drifts: {summary.total_drifts}")
        logger.info(f"  Critical: {summary.critical_drifts}")
        logger.info(f"  Warning: {summary.warning_drifts}")


async def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="T.A.R.S. Production Drift Detector")
    parser.add_argument("--baseline-file", type=str, required=True, help="Baseline metrics file (JSON)")
    parser.add_argument("--duration", type=float, default=24.0, help="Detection duration in hours (default: 24)")
    parser.add_argument("--check-interval", type=int, default=300, help="Check interval in seconds (default: 300)")
    parser.add_argument("--output", type=str, default="drift_analysis.json", help="Output file (default: drift_analysis.json)")
    parser.add_argument("--prometheus-url", type=str, default="http://localhost:9090", help="Prometheus URL")
    parser.add_argument("--warning-threshold", type=float, default=5.0, help="Warning threshold percentage (default: 5.0)")
    parser.add_argument("--critical-threshold", type=float, default=10.0, help="Critical threshold percentage (default: 10.0)")

    args = parser.parse_args()

    # Create detector
    detector = DriftDetector(
        baseline_file=Path(args.baseline_file),
        prometheus_url=args.prometheus_url,
        output_file=Path(args.output),
        check_interval_seconds=args.check_interval,
        warning_threshold=args.warning_threshold,
        critical_threshold=args.critical_threshold
    )

    # Run detection
    await detector.run_detection(duration_hours=args.duration)


if __name__ == "__main__":
    asyncio.run(main())
