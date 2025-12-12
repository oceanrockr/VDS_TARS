#!/usr/bin/env python3
"""
Real-Time SLO Monitor for T.A.R.S. GA Deployment

Continuously monitors production SLOs during GA deployment and alerts on violations.
Designed to run during canary stages to catch issues immediately.

Usage:
    python monitor_realtime_slos.py --environment=production --deployment=canary
    python monitor_realtime_slos.py --environment=production --duration=30m
    python monitor_realtime_slos.py --help

Features:
    - 5-second polling interval for near-real-time monitoring
    - Configurable SLO thresholds
    - Slack/webhook alerts on violations
    - Rolling JSON log output
    - Graceful shutdown (SIGINT/SIGTERM)
    - Async implementation for efficiency

Author: T.A.R.S. Release Team
Version: 1.0.0
"""

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List
from urllib.parse import urljoin

import aiohttp


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class SLOThresholds:
    """SLO threshold configuration"""
    availability_pct: float = 99.9
    latency_p95_ms: float = 100.0
    latency_p99_ms: float = 200.0
    error_rate_pct: float = 1.0
    memory_mb_per_pod: float = 2500.0
    cpu_cores_per_pod: float = 2.0


@dataclass
class MonitorConfig:
    """Monitor configuration"""
    environment: str
    deployment: Optional[str]
    prometheus_url: str
    poll_interval_sec: int = 5
    duration_minutes: Optional[int] = None
    thresholds: SLOThresholds = None
    alert_webhook_url: Optional[str] = None
    output_file: str = "slo_monitor.json"
    namespace: str = "tars-production"

    def __post_init__(self):
        if self.thresholds is None:
            self.thresholds = SLOThresholds()


@dataclass
class SLOSnapshot:
    """Single point-in-time SLO measurement"""
    timestamp: str
    deployment: Optional[str]
    availability_pct: float
    latency_p95_ms: float
    latency_p99_ms: float
    error_rate_pct: float
    memory_mb_avg: float
    cpu_cores_avg: float
    violations: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================================
# Logging Setup
# ============================================================================

def setup_logging(verbose: bool = False) -> logging.Logger:
    """Configure structured logging"""
    level = logging.DEBUG if verbose else logging.INFO

    logging.basicConfig(
        level=level,
        format='{"timestamp":"%(asctime)s","level":"%(levelname)s","message":"%(message)s"}',
        datefmt='%Y-%m-%dT%H:%M:%SZ'
    )

    return logging.getLogger(__name__)


logger = setup_logging()


# ============================================================================
# Prometheus Queries
# ============================================================================

class PrometheusClient:
    """Async Prometheus query client"""

    def __init__(self, base_url: str, timeout: int = 10):
        self.base_url = base_url
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(timeout=self.timeout)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def query(self, promql: str) -> Optional[Dict[str, Any]]:
        """Execute PromQL query"""
        if not self.session:
            raise RuntimeError("PrometheusClient used outside context manager")

        url = urljoin(self.base_url, "/api/v1/query")
        params = {"query": promql}

        try:
            async with self.session.get(url, params=params) as response:
                if response.status != 200:
                    logger.error(f"Prometheus query failed: {response.status}")
                    return None

                data = await response.json()

                if data.get("status") != "success":
                    logger.error(f"Prometheus returned error: {data.get('error')}")
                    return None

                return data.get("data", {})

        except asyncio.TimeoutError:
            logger.error("Prometheus query timeout")
            return None
        except Exception as e:
            logger.error(f"Prometheus query error: {e}")
            return None

    async def query_scalar(self, promql: str) -> Optional[float]:
        """Execute PromQL query and return scalar result"""
        result = await self.query(promql)

        if not result:
            return None

        result_type = result.get("resultType")
        result_data = result.get("result", [])

        if result_type == "vector" and len(result_data) > 0:
            value = result_data[0].get("value", [None, None])
            if len(value) > 1:
                try:
                    return float(value[1])
                except (ValueError, TypeError):
                    return None

        elif result_type == "scalar":
            try:
                return float(result_data[1])
            except (ValueError, TypeError, IndexError):
                return None

        return None


# ============================================================================
# SLO Metrics Collection
# ============================================================================

class SLOCollector:
    """Collects SLO metrics from Prometheus"""

    def __init__(self, config: MonitorConfig, prom_client: PrometheusClient):
        self.config = config
        self.prom = prom_client

    def _build_selector(self) -> str:
        """Build Prometheus label selector"""
        selectors = [f'namespace="{self.config.namespace}"']

        if self.config.deployment:
            selectors.append(f'deployment="{self.config.deployment}"')

        return ",".join(selectors)

    async def get_availability(self) -> Optional[float]:
        """Calculate availability percentage (5-minute window)"""
        selector = self._build_selector()

        # Availability = avg(up) over 5 minutes * 100
        query = f'avg_over_time(up{{{selector}}}[5m]) * 100'

        return await self.prom.query_scalar(query)

    async def get_latency_p95(self) -> Optional[float]:
        """Calculate p95 latency in milliseconds (5-minute window)"""
        selector = self._build_selector()

        # P95 latency from histogram
        query = f'''
        histogram_quantile(0.95,
            rate(http_request_duration_seconds_bucket{{{selector}}}[5m])
        ) * 1000
        '''

        return await self.prom.query_scalar(query)

    async def get_latency_p99(self) -> Optional[float]:
        """Calculate p99 latency in milliseconds (5-minute window)"""
        selector = self._build_selector()

        query = f'''
        histogram_quantile(0.99,
            rate(http_request_duration_seconds_bucket{{{selector}}}[5m])
        ) * 1000
        '''

        return await self.prom.query_scalar(query)

    async def get_error_rate(self) -> Optional[float]:
        """Calculate error rate percentage (5-minute window)"""
        selector = self._build_selector()

        # Error rate = rate(5xx) / rate(total) * 100
        query = f'''
        (
            rate(http_requests_total{{{selector},status=~"5.."}}[5m])
            /
            rate(http_requests_total{{{selector}}}[5m])
        ) * 100
        '''

        result = await self.prom.query_scalar(query)

        # Return 0 if no errors (query returns NaN)
        return result if result is not None else 0.0

    async def get_memory_usage(self) -> Optional[float]:
        """Calculate average memory usage in MB"""
        selector = self._build_selector()

        # Average container memory usage
        query = f'avg(container_memory_usage_bytes{{{selector}}}) / 1024 / 1024'

        return await self.prom.query_scalar(query)

    async def get_cpu_usage(self) -> Optional[float]:
        """Calculate average CPU usage in cores"""
        selector = self._build_selector()

        # Average CPU usage rate (5-minute window)
        query = f'avg(rate(container_cpu_usage_seconds_total{{{selector}}}[5m]))'

        return await self.prom.query_scalar(query)

    async def collect_snapshot(self) -> Optional[SLOSnapshot]:
        """Collect all SLO metrics as a snapshot"""
        logger.debug("Collecting SLO snapshot...")

        # Collect all metrics concurrently
        availability, latency_p95, latency_p99, error_rate, memory, cpu = await asyncio.gather(
            self.get_availability(),
            self.get_latency_p95(),
            self.get_latency_p99(),
            self.get_error_rate(),
            self.get_memory_usage(),
            self.get_cpu_usage(),
            return_exceptions=False
        )

        # Check for missing metrics
        if availability is None or latency_p95 is None:
            logger.warning("Critical metrics missing, skipping snapshot")
            return None

        # Default values for optional metrics
        latency_p99 = latency_p99 if latency_p99 is not None else latency_p95 * 1.5
        error_rate = error_rate if error_rate is not None else 0.0
        memory = memory if memory is not None else 0.0
        cpu = cpu if cpu is not None else 0.0

        # Check for violations
        violations = []
        thresholds = self.config.thresholds

        if availability < thresholds.availability_pct:
            violations.append(f"Availability {availability:.2f}% < {thresholds.availability_pct}%")

        if latency_p95 > thresholds.latency_p95_ms:
            violations.append(f"P95 latency {latency_p95:.1f}ms > {thresholds.latency_p95_ms}ms")

        if latency_p99 > thresholds.latency_p99_ms:
            violations.append(f"P99 latency {latency_p99:.1f}ms > {thresholds.latency_p99_ms}ms")

        if error_rate > thresholds.error_rate_pct:
            violations.append(f"Error rate {error_rate:.2f}% > {thresholds.error_rate_pct}%")

        if memory > thresholds.memory_mb_per_pod:
            violations.append(f"Memory {memory:.0f}MB > {thresholds.memory_mb_per_pod}MB")

        if cpu > thresholds.cpu_cores_per_pod:
            violations.append(f"CPU {cpu:.2f} cores > {thresholds.cpu_cores_per_pod} cores")

        snapshot = SLOSnapshot(
            timestamp=datetime.utcnow().isoformat() + "Z",
            deployment=self.config.deployment,
            availability_pct=round(availability, 3),
            latency_p95_ms=round(latency_p95, 1),
            latency_p99_ms=round(latency_p99, 1),
            error_rate_pct=round(error_rate, 3),
            memory_mb_avg=round(memory, 0),
            cpu_cores_avg=round(cpu, 2),
            violations=violations
        )

        logger.info(f"SLO snapshot: availability={availability:.2f}%, p95={latency_p95:.1f}ms, errors={error_rate:.2f}%")

        if violations:
            logger.warning(f"SLO violations: {', '.join(violations)}")

        return snapshot


# ============================================================================
# Alerting
# ============================================================================

class AlertManager:
    """Manages alerting via webhooks"""

    def __init__(self, webhook_url: Optional[str]):
        self.webhook_url = webhook_url
        self.session: Optional[aiohttp.ClientSession] = None
        self.last_alert_time: Optional[datetime] = None
        self.min_alert_interval = timedelta(minutes=5)  # Rate limit alerts

    async def __aenter__(self):
        if self.webhook_url:
            self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def send_alert(self, snapshot: SLOSnapshot) -> bool:
        """Send alert for SLO violations"""
        if not self.webhook_url or not self.session:
            return False

        if not snapshot.violations:
            return False

        # Rate limit alerts
        now = datetime.utcnow()
        if self.last_alert_time and (now - self.last_alert_time) < self.min_alert_interval:
            logger.debug("Alert rate limited")
            return False

        # Build alert message
        deployment_str = f" ({snapshot.deployment})" if snapshot.deployment else ""
        violations_str = "\n".join(f"• {v}" for v in snapshot.violations)

        message = {
            "text": f"🚨 *SLO Violations Detected{deployment_str}*",
            "blocks": [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*SLO Violations{deployment_str}*\n\n{violations_str}"
                    }
                },
                {
                    "type": "section",
                    "fields": [
                        {"type": "mrkdwn", "text": f"*Availability:*\n{snapshot.availability_pct:.2f}%"},
                        {"type": "mrkdwn", "text": f"*P95 Latency:*\n{snapshot.latency_p95_ms:.1f}ms"},
                        {"type": "mrkdwn", "text": f"*Error Rate:*\n{snapshot.error_rate_pct:.2f}%"},
                        {"type": "mrkdwn", "text": f"*Time:*\n{snapshot.timestamp}"}
                    ]
                }
            ]
        }

        try:
            async with self.session.post(self.webhook_url, json=message) as response:
                if response.status == 200:
                    logger.info("Alert sent successfully")
                    self.last_alert_time = now
                    return True
                else:
                    logger.error(f"Alert failed: {response.status}")
                    return False

        except Exception as e:
            logger.error(f"Alert error: {e}")
            return False


# ============================================================================
# Main Monitor
# ============================================================================

class RealTimeSLOMonitor:
    """Main monitoring orchestrator"""

    def __init__(self, config: MonitorConfig):
        self.config = config
        self.snapshots: List[SLOSnapshot] = []
        self.running = True
        self.start_time = datetime.utcnow()

    def _setup_signal_handlers(self):
        """Setup graceful shutdown on SIGINT/SIGTERM"""
        def handler(signum, frame):
            logger.info(f"Received signal {signum}, shutting down...")
            self.running = False

        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)

    def _should_continue(self) -> bool:
        """Check if monitoring should continue"""
        if not self.running:
            return False

        if self.config.duration_minutes is None:
            return True

        elapsed = (datetime.utcnow() - self.start_time).total_seconds() / 60
        return elapsed < self.config.duration_minutes

    def _save_snapshots(self):
        """Save snapshots to JSON file"""
        output_path = Path(self.config.output_file)

        data = {
            "config": {
                "environment": self.config.environment,
                "deployment": self.config.deployment,
                "start_time": self.start_time.isoformat() + "Z",
                "end_time": datetime.utcnow().isoformat() + "Z",
                "thresholds": asdict(self.config.thresholds)
            },
            "snapshots": [s.to_dict() for s in self.snapshots],
            "summary": self._generate_summary()
        }

        try:
            with output_path.open("w") as f:
                json.dump(data, f, indent=2)

            logger.info(f"Saved {len(self.snapshots)} snapshots to {output_path}")

        except Exception as e:
            logger.error(f"Failed to save snapshots: {e}")

    def _generate_summary(self) -> Dict[str, Any]:
        """Generate summary statistics"""
        if not self.snapshots:
            return {}

        total_violations = sum(1 for s in self.snapshots if s.violations)
        violation_rate = total_violations / len(self.snapshots) * 100

        avg_availability = sum(s.availability_pct for s in self.snapshots) / len(self.snapshots)
        avg_latency_p95 = sum(s.latency_p95_ms for s in self.snapshots) / len(self.snapshots)
        avg_error_rate = sum(s.error_rate_pct for s in self.snapshots) / len(self.snapshots)

        max_latency_p95 = max(s.latency_p95_ms for s in self.snapshots)
        max_error_rate = max(s.error_rate_pct for s in self.snapshots)
        min_availability = min(s.availability_pct for s in self.snapshots)

        return {
            "total_snapshots": len(self.snapshots),
            "snapshots_with_violations": total_violations,
            "violation_rate_pct": round(violation_rate, 2),
            "avg_availability_pct": round(avg_availability, 3),
            "avg_latency_p95_ms": round(avg_latency_p95, 1),
            "avg_error_rate_pct": round(avg_error_rate, 3),
            "min_availability_pct": round(min_availability, 3),
            "max_latency_p95_ms": round(max_latency_p95, 1),
            "max_error_rate_pct": round(max_error_rate, 3)
        }

    async def run(self):
        """Main monitoring loop"""
        self._setup_signal_handlers()

        logger.info(f"Starting real-time SLO monitoring for {self.config.environment}")
        logger.info(f"Deployment: {self.config.deployment or 'all'}")
        logger.info(f"Poll interval: {self.config.poll_interval_sec}s")
        logger.info(f"Duration: {self.config.duration_minutes or 'unlimited'} minutes")

        async with PrometheusClient(self.config.prometheus_url) as prom_client:
            collector = SLOCollector(self.config, prom_client)

            async with AlertManager(self.config.alert_webhook_url) as alert_mgr:
                while self._should_continue():
                    try:
                        # Collect snapshot
                        snapshot = await collector.collect_snapshot()

                        if snapshot:
                            self.snapshots.append(snapshot)

                            # Send alert if violations detected
                            if snapshot.violations:
                                await alert_mgr.send_alert(snapshot)

                        # Wait for next poll
                        await asyncio.sleep(self.config.poll_interval_sec)

                    except Exception as e:
                        logger.error(f"Monitoring error: {e}", exc_info=True)
                        await asyncio.sleep(self.config.poll_interval_sec)

        # Save results
        self._save_snapshots()

        # Print summary
        summary = self._generate_summary()
        logger.info("=" * 60)
        logger.info("MONITORING SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total snapshots: {summary.get('total_snapshots', 0)}")
        logger.info(f"Violations: {summary.get('snapshots_with_violations', 0)} ({summary.get('violation_rate_pct', 0):.1f}%)")
        logger.info(f"Avg availability: {summary.get('avg_availability_pct', 0):.2f}%")
        logger.info(f"Avg p95 latency: {summary.get('avg_latency_p95_ms', 0):.1f}ms")
        logger.info(f"Avg error rate: {summary.get('avg_error_rate_pct', 0):.2f}%")
        logger.info("=" * 60)


# ============================================================================
# CLI
# ============================================================================

def parse_duration(duration_str: str) -> int:
    """Parse duration string (e.g., '30m', '2h') to minutes"""
    if duration_str.endswith('m'):
        return int(duration_str[:-1])
    elif duration_str.endswith('h'):
        return int(duration_str[:-1]) * 60
    elif duration_str.endswith('s'):
        return int(duration_str[:-1]) // 60
    else:
        raise ValueError(f"Invalid duration format: {duration_str}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="Real-time SLO monitor for T.A.R.S. GA deployment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Monitor canary deployment for 30 minutes
  python monitor_realtime_slos.py --environment=production --deployment=canary --duration=30m

  # Monitor all deployments continuously
  python monitor_realtime_slos.py --environment=production

  # Monitor with custom thresholds and alerting
  python monitor_realtime_slos.py \\
    --environment=production \\
    --deployment=canary \\
    --threshold-availability=99.5 \\
    --threshold-latency-p95=120 \\
    --alert-webhook=https://hooks.slack.com/services/YOUR/WEBHOOK/URL
        """
    )

    parser.add_argument(
        "--environment",
        required=True,
        choices=["production", "staging"],
        help="Environment to monitor"
    )

    parser.add_argument(
        "--deployment",
        help="Specific deployment to monitor (e.g., 'canary', 'stable')"
    )

    parser.add_argument(
        "--prometheus-url",
        default=os.getenv("PROMETHEUS_URL", "http://localhost:9090"),
        help="Prometheus server URL (default: $PROMETHEUS_URL or http://localhost:9090)"
    )

    parser.add_argument(
        "--namespace",
        default="tars-production",
        help="Kubernetes namespace (default: tars-production)"
    )

    parser.add_argument(
        "--poll-interval",
        type=int,
        default=5,
        help="Poll interval in seconds (default: 5)"
    )

    parser.add_argument(
        "--duration",
        help="Monitoring duration (e.g., '30m', '2h', '120s'). Run indefinitely if not specified."
    )

    parser.add_argument(
        "--threshold-availability",
        type=float,
        default=99.9,
        help="Availability threshold percentage (default: 99.9)"
    )

    parser.add_argument(
        "--threshold-latency-p95",
        type=float,
        default=100.0,
        help="P95 latency threshold in ms (default: 100)"
    )

    parser.add_argument(
        "--threshold-latency-p99",
        type=float,
        default=200.0,
        help="P99 latency threshold in ms (default: 200)"
    )

    parser.add_argument(
        "--threshold-error-rate",
        type=float,
        default=1.0,
        help="Error rate threshold percentage (default: 1.0)"
    )

    parser.add_argument(
        "--threshold-memory",
        type=float,
        default=2500.0,
        help="Memory threshold in MB per pod (default: 2500)"
    )

    parser.add_argument(
        "--threshold-cpu",
        type=float,
        default=2.0,
        help="CPU threshold in cores per pod (default: 2.0)"
    )

    parser.add_argument(
        "--alert-webhook",
        default=os.getenv("ALERT_WEBHOOK_URL"),
        help="Webhook URL for alerts (default: $ALERT_WEBHOOK_URL)"
    )

    parser.add_argument(
        "--output",
        default="slo_monitor.json",
        help="Output JSON file (default: slo_monitor.json)"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_args()

    # Setup logging
    global logger
    logger = setup_logging(args.verbose)

    # Parse duration
    duration_minutes = None
    if args.duration:
        try:
            duration_minutes = parse_duration(args.duration)
        except ValueError as e:
            logger.error(str(e))
            sys.exit(1)

    # Build configuration
    thresholds = SLOThresholds(
        availability_pct=args.threshold_availability,
        latency_p95_ms=args.threshold_latency_p95,
        latency_p99_ms=args.threshold_latency_p99,
        error_rate_pct=args.threshold_error_rate,
        memory_mb_per_pod=args.threshold_memory,
        cpu_cores_per_pod=args.threshold_cpu
    )

    config = MonitorConfig(
        environment=args.environment,
        deployment=args.deployment,
        prometheus_url=args.prometheus_url,
        namespace=args.namespace,
        poll_interval_sec=args.poll_interval,
        duration_minutes=duration_minutes,
        thresholds=thresholds,
        alert_webhook_url=args.alert_webhook,
        output_file=args.output
    )

    # Run monitor
    monitor = RealTimeSLOMonitor(config)

    try:
        asyncio.run(monitor.run())
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
