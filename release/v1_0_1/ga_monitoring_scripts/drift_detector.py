#!/usr/bin/env python3
"""
Production Drift Detector for T.A.R.S. GA Deployment

Compares staging vs production (or canary vs stable) to detect drift in:
- API latency
- Error rates
- Resource utilization
- Database query performance
- Distributed tracing coverage

Usage:
    python drift_detector.py --environment=production --compare-to=staging
    python drift_detector.py --canary-deployment=canary --stable-deployment=stable
    python drift_detector.py --help

Features:
    - Side-by-side metric comparison
    - Configurable drift thresholds
    - Automatic alerting on significant drift
    - JSON + Markdown output
    - Async implementation

Author: T.A.R.S. Release Team
Version: 1.0.0
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from urllib.parse import urljoin

import aiohttp


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class DriftThresholds:
    """Drift detection thresholds (percentage difference)"""
    latency_pct: float = 10.0  # Alert if latency differs by >10%
    error_rate_pct: float = 5.0  # Alert if error rate differs by >5%
    memory_pct: float = 20.0  # Alert if memory differs by >20%
    cpu_pct: float = 15.0  # Alert if CPU differs by >15%
    trace_coverage_pct: float = 5.0  # Alert if coverage differs by >5%


@dataclass
class DriftConfig:
    """Drift detector configuration"""
    environment: str
    compare_to: Optional[str] = None  # staging, production
    canary_deployment: Optional[str] = None
    stable_deployment: Optional[str] = None
    prometheus_url: str = "http://localhost:9090"
    postgres_url: Optional[str] = None
    namespace: str = "tars-production"
    compare_namespace: Optional[str] = None
    duration_minutes: int = 10
    interval_minutes: int = 15
    thresholds: DriftThresholds = field(default_factory=DriftThresholds)
    output_file: str = "drift_report.json"
    alert_webhook_url: Optional[str] = None


@dataclass
class MetricSnapshot:
    """Snapshot of metrics from a deployment"""
    name: str
    availability_pct: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    error_rate_pct: float
    memory_mb_avg: float
    cpu_cores_avg: float
    db_query_time_ms: float
    trace_coverage_pct: float


@dataclass
class DriftAnalysis:
    """Drift analysis result"""
    metric: str
    baseline_value: float
    comparison_value: float
    diff_pct: float
    threshold_pct: float
    is_drift: bool
    severity: str  # "info", "warning", "critical"

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
# Prometheus Client
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

    async def query_scalar(self, promql: str) -> Optional[float]:
        """Execute PromQL query and return scalar result"""
        if not self.session:
            raise RuntimeError("PrometheusClient used outside context manager")

        url = urljoin(self.base_url, "/api/v1/query")
        params = {"query": promql}

        try:
            async with self.session.get(url, params=params) as response:
                if response.status != 200:
                    logger.debug(f"Prometheus query failed: {response.status}")
                    return None

                data = await response.json()

                if data.get("status") != "success":
                    return None

                result = data.get("data", {})
                result_type = result.get("resultType")
                result_data = result.get("result", [])

                if result_type == "vector" and len(result_data) > 0:
                    value = result_data[0].get("value", [None, None])
                    if len(value) > 1:
                        try:
                            return float(value[1])
                        except (ValueError, TypeError):
                            return None

                return None

        except Exception as e:
            logger.debug(f"Prometheus query error: {e}")
            return None


# ============================================================================
# Metrics Collector
# ============================================================================

class MetricsCollector:
    """Collects metrics for drift comparison"""

    def __init__(self, prom_client: PrometheusClient, config: DriftConfig):
        self.prom = prom_client
        self.config = config

    def _build_selector(self, deployment: Optional[str] = None, namespace: Optional[str] = None) -> str:
        """Build Prometheus label selector"""
        ns = namespace or self.config.namespace
        selectors = [f'namespace="{ns}"']

        if deployment:
            selectors.append(f'deployment="{deployment}"')

        return ",".join(selectors)

    async def _get_availability(self, selector: str) -> float:
        """Get availability percentage"""
        query = f'avg_over_time(up{{{selector}}}[5m]) * 100'
        result = await self.prom.query_scalar(query)
        return result if result is not None else 0.0

    async def _get_latency_percentile(self, selector: str, percentile: float) -> float:
        """Get latency percentile in milliseconds"""
        query = f'''
        histogram_quantile({percentile},
            rate(http_request_duration_seconds_bucket{{{selector}}}[5m])
        ) * 1000
        '''
        result = await self.prom.query_scalar(query)
        return result if result is not None else 0.0

    async def _get_error_rate(self, selector: str) -> float:
        """Get error rate percentage"""
        query = f'''
        (
            rate(http_requests_total{{{selector},status=~"5.."}}[5m])
            /
            rate(http_requests_total{{{selector}}}[5m])
        ) * 100
        '''
        result = await self.prom.query_scalar(query)
        return result if result is not None else 0.0

    async def _get_memory(self, selector: str) -> float:
        """Get average memory usage in MB"""
        query = f'avg(container_memory_usage_bytes{{{selector}}}) / 1024 / 1024'
        result = await self.prom.query_scalar(query)
        return result if result is not None else 0.0

    async def _get_cpu(self, selector: str) -> float:
        """Get average CPU usage in cores"""
        query = f'avg(rate(container_cpu_usage_seconds_total{{{selector}}}[5m]))'
        result = await self.prom.query_scalar(query)
        return result if result is not None else 0.0

    async def _get_db_query_time(self, selector: str) -> float:
        """Get average database query time in milliseconds"""
        query = f'avg(rate(db_query_duration_seconds_sum{{{selector}}}[5m]) / rate(db_query_duration_seconds_count{{{selector}}}[5m])) * 1000'
        result = await self.prom.query_scalar(query)
        return result if result is not None else 0.0

    async def _get_trace_coverage(self, selector: str) -> float:
        """Get distributed tracing coverage percentage"""
        query = f'''
        (
            count(traces_total{{{selector},has_trace_id="true"}})
            /
            count(traces_total{{{selector}}})
        ) * 100
        '''
        result = await self.prom.query_scalar(query)
        return result if result is not None else 0.0

    async def collect_snapshot(self, name: str, deployment: Optional[str] = None, namespace: Optional[str] = None) -> MetricSnapshot:
        """Collect all metrics as a snapshot"""
        selector = self._build_selector(deployment, namespace)

        logger.debug(f"Collecting snapshot for {name} (selector: {selector})")

        # Collect all metrics concurrently
        results = await asyncio.gather(
            self._get_availability(selector),
            self._get_latency_percentile(selector, 0.5),
            self._get_latency_percentile(selector, 0.95),
            self._get_latency_percentile(selector, 0.99),
            self._get_error_rate(selector),
            self._get_memory(selector),
            self._get_cpu(selector),
            self._get_db_query_time(selector),
            self._get_trace_coverage(selector),
            return_exceptions=False
        )

        snapshot = MetricSnapshot(
            name=name,
            availability_pct=round(results[0], 3),
            latency_p50_ms=round(results[1], 1),
            latency_p95_ms=round(results[2], 1),
            latency_p99_ms=round(results[3], 1),
            error_rate_pct=round(results[4], 3),
            memory_mb_avg=round(results[5], 0),
            cpu_cores_avg=round(results[6], 2),
            db_query_time_ms=round(results[7], 1),
            trace_coverage_pct=round(results[8], 1)
        )

        logger.info(f"Snapshot {name}: availability={snapshot.availability_pct}%, p95={snapshot.latency_p95_ms}ms, errors={snapshot.error_rate_pct}%")

        return snapshot


# ============================================================================
# Drift Analyzer
# ============================================================================

class DriftAnalyzer:
    """Analyzes drift between two metric snapshots"""

    def __init__(self, config: DriftConfig):
        self.config = config

    @staticmethod
    def _calculate_drift_pct(baseline: float, comparison: float) -> float:
        """Calculate percentage difference"""
        if baseline == 0:
            return 100.0 if comparison != 0 else 0.0

        return abs((comparison - baseline) / baseline * 100)

    @staticmethod
    def _determine_severity(diff_pct: float, threshold_pct: float) -> str:
        """Determine drift severity"""
        if diff_pct < threshold_pct:
            return "info"
        elif diff_pct < threshold_pct * 2:
            return "warning"
        else:
            return "critical"

    def analyze_drift(self, baseline: MetricSnapshot, comparison: MetricSnapshot) -> List[DriftAnalysis]:
        """Analyze drift between two snapshots"""
        logger.info(f"Analyzing drift: {baseline.name} vs {comparison.name}")

        analyses = []

        # Latency P95
        latency_diff = self._calculate_drift_pct(baseline.latency_p95_ms, comparison.latency_p95_ms)
        analyses.append(DriftAnalysis(
            metric="latency_p95_ms",
            baseline_value=baseline.latency_p95_ms,
            comparison_value=comparison.latency_p95_ms,
            diff_pct=round(latency_diff, 2),
            threshold_pct=self.config.thresholds.latency_pct,
            is_drift=latency_diff > self.config.thresholds.latency_pct,
            severity=self._determine_severity(latency_diff, self.config.thresholds.latency_pct)
        ))

        # Latency P99
        latency_p99_diff = self._calculate_drift_pct(baseline.latency_p99_ms, comparison.latency_p99_ms)
        analyses.append(DriftAnalysis(
            metric="latency_p99_ms",
            baseline_value=baseline.latency_p99_ms,
            comparison_value=comparison.latency_p99_ms,
            diff_pct=round(latency_p99_diff, 2),
            threshold_pct=self.config.thresholds.latency_pct,
            is_drift=latency_p99_diff > self.config.thresholds.latency_pct,
            severity=self._determine_severity(latency_p99_diff, self.config.thresholds.latency_pct)
        ))

        # Error Rate
        error_diff = self._calculate_drift_pct(baseline.error_rate_pct, comparison.error_rate_pct)
        analyses.append(DriftAnalysis(
            metric="error_rate_pct",
            baseline_value=baseline.error_rate_pct,
            comparison_value=comparison.error_rate_pct,
            diff_pct=round(error_diff, 2),
            threshold_pct=self.config.thresholds.error_rate_pct,
            is_drift=error_diff > self.config.thresholds.error_rate_pct,
            severity=self._determine_severity(error_diff, self.config.thresholds.error_rate_pct)
        ))

        # Memory
        memory_diff = self._calculate_drift_pct(baseline.memory_mb_avg, comparison.memory_mb_avg)
        analyses.append(DriftAnalysis(
            metric="memory_mb_avg",
            baseline_value=baseline.memory_mb_avg,
            comparison_value=comparison.memory_mb_avg,
            diff_pct=round(memory_diff, 2),
            threshold_pct=self.config.thresholds.memory_pct,
            is_drift=memory_diff > self.config.thresholds.memory_pct,
            severity=self._determine_severity(memory_diff, self.config.thresholds.memory_pct)
        ))

        # CPU
        cpu_diff = self._calculate_drift_pct(baseline.cpu_cores_avg, comparison.cpu_cores_avg)
        analyses.append(DriftAnalysis(
            metric="cpu_cores_avg",
            baseline_value=baseline.cpu_cores_avg,
            comparison_value=comparison.cpu_cores_avg,
            diff_pct=round(cpu_diff, 2),
            threshold_pct=self.config.thresholds.cpu_pct,
            is_drift=cpu_diff > self.config.thresholds.cpu_pct,
            severity=self._determine_severity(cpu_diff, self.config.thresholds.cpu_pct)
        ))

        # Database Query Time
        db_diff = self._calculate_drift_pct(baseline.db_query_time_ms, comparison.db_query_time_ms)
        analyses.append(DriftAnalysis(
            metric="db_query_time_ms",
            baseline_value=baseline.db_query_time_ms,
            comparison_value=comparison.db_query_time_ms,
            diff_pct=round(db_diff, 2),
            threshold_pct=self.config.thresholds.latency_pct,
            is_drift=db_diff > self.config.thresholds.latency_pct,
            severity=self._determine_severity(db_diff, self.config.thresholds.latency_pct)
        ))

        # Trace Coverage
        trace_diff = self._calculate_drift_pct(baseline.trace_coverage_pct, comparison.trace_coverage_pct)
        analyses.append(DriftAnalysis(
            metric="trace_coverage_pct",
            baseline_value=baseline.trace_coverage_pct,
            comparison_value=comparison.trace_coverage_pct,
            diff_pct=round(trace_diff, 2),
            threshold_pct=self.config.thresholds.trace_coverage_pct,
            is_drift=trace_diff > self.config.thresholds.trace_coverage_pct,
            severity=self._determine_severity(trace_diff, self.config.thresholds.trace_coverage_pct)
        ))

        # Log significant drift
        drift_detected = [a for a in analyses if a.is_drift]
        if drift_detected:
            logger.warning(f"Detected {len(drift_detected)} metrics with significant drift")
            for analysis in drift_detected:
                logger.warning(f"  {analysis.metric}: {analysis.diff_pct:.1f}% drift ({analysis.severity})")

        return analyses


# ============================================================================
# Alert Manager
# ============================================================================

class AlertManager:
    """Manages drift alerts"""

    def __init__(self, webhook_url: Optional[str]):
        self.webhook_url = webhook_url
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        if self.webhook_url:
            self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def send_drift_alert(self, baseline_name: str, comparison_name: str, analyses: List[DriftAnalysis]) -> bool:
        """Send alert for drift detection"""
        if not self.webhook_url or not self.session:
            return False

        drift_detected = [a for a in analyses if a.is_drift]
        if not drift_detected:
            return False

        # Group by severity
        critical = [a for a in drift_detected if a.severity == "critical"]
        warning = [a for a in drift_detected if a.severity == "warning"]

        # Build alert message
        drift_lines = []
        for analysis in critical:
            drift_lines.append(f"🔴 *{analysis.metric}*: {analysis.diff_pct:.1f}% drift (baseline: {analysis.baseline_value}, current: {analysis.comparison_value})")

        for analysis in warning:
            drift_lines.append(f"🟡 *{analysis.metric}*: {analysis.diff_pct:.1f}% drift (baseline: {analysis.baseline_value}, current: {analysis.comparison_value})")

        drift_str = "\n".join(drift_lines)

        message = {
            "text": f"⚠️ *Drift Detected: {baseline_name} vs {comparison_name}*",
            "blocks": [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Drift Detected*\n\nBaseline: {baseline_name}\nComparison: {comparison_name}\n\n{drift_str}"
                    }
                },
                {
                    "type": "section",
                    "fields": [
                        {"type": "mrkdwn", "text": f"*Critical:*\n{len(critical)}"},
                        {"type": "mrkdwn", "text": f"*Warning:*\n{len(warning)}"},
                        {"type": "mrkdwn", "text": f"*Total:*\n{len(drift_detected)}"},
                        {"type": "mrkdwn", "text": f"*Time:*\n{datetime.utcnow().isoformat()}Z"}
                    ]
                }
            ]
        }

        try:
            async with self.session.post(self.webhook_url, json=message) as response:
                if response.status == 200:
                    logger.info("Drift alert sent successfully")
                    return True
                else:
                    logger.error(f"Drift alert failed: {response.status}")
                    return False

        except Exception as e:
            logger.error(f"Drift alert error: {e}")
            return False


# ============================================================================
# Main Drift Detector
# ============================================================================

class DriftDetector:
    """Main drift detection orchestrator"""

    def __init__(self, config: DriftConfig):
        self.config = config
        self.snapshots: List[Tuple[MetricSnapshot, MetricSnapshot, List[DriftAnalysis]]] = []

    def _generate_markdown_report(self) -> str:
        """Generate markdown drift report"""
        if not self.snapshots:
            return "No drift data collected."

        # Use latest snapshot for report
        baseline, comparison, analyses = self.snapshots[-1]

        report = f"""# Drift Analysis Report

**Baseline:** {baseline.name}
**Comparison:** {comparison.name}
**Timestamp:** {datetime.utcnow().isoformat()}Z

---

## Summary

"""

        drift_detected = [a for a in analyses if a.is_drift]
        critical = [a for a in drift_detected if a.severity == "critical"]
        warning = [a for a in drift_detected if a.severity == "warning"]

        report += f"- **Total Metrics Analyzed:** {len(analyses)}\n"
        report += f"- **Metrics with Drift:** {len(drift_detected)}\n"
        report += f"- **Critical:** {len(critical)}\n"
        report += f"- **Warning:** {len(warning)}\n\n"

        report += "---\n\n## Detailed Comparison\n\n"
        report += "| Metric | Baseline | Comparison | Diff % | Threshold % | Status |\n"
        report += "|--------|----------|------------|--------|-------------|--------|\n"

        for analysis in analyses:
            status_icon = "🔴" if analysis.severity == "critical" else "🟡" if analysis.severity == "warning" else "✅"
            report += f"| {analysis.metric} | {analysis.baseline_value:.2f} | {analysis.comparison_value:.2f} | {analysis.diff_pct:.1f}% | {analysis.threshold_pct:.1f}% | {status_icon} {analysis.severity} |\n"

        report += "\n---\n\n## Drift Details\n\n"

        if critical:
            report += "### 🔴 Critical Drift\n\n"
            for analysis in critical:
                report += f"**{analysis.metric}**\n"
                report += f"- Baseline: {analysis.baseline_value:.2f}\n"
                report += f"- Comparison: {analysis.comparison_value:.2f}\n"
                report += f"- Drift: {analysis.diff_pct:.1f}% (threshold: {analysis.threshold_pct:.1f}%)\n\n"

        if warning:
            report += "### 🟡 Warning Drift\n\n"
            for analysis in warning:
                report += f"**{analysis.metric}**\n"
                report += f"- Baseline: {analysis.baseline_value:.2f}\n"
                report += f"- Comparison: {analysis.comparison_value:.2f}\n"
                report += f"- Drift: {analysis.diff_pct:.1f}% (threshold: {analysis.threshold_pct:.1f}%)\n\n"

        if not drift_detected:
            report += "✅ No significant drift detected.\n\n"

        report += "---\n\n"
        report += "*Generated by T.A.R.S. Drift Detector*\n"

        return report

    def _save_report(self):
        """Save drift report to JSON and Markdown"""
        output_path = Path(self.config.output_file)

        # JSON report
        data = {
            "config": {
                "environment": self.config.environment,
                "compare_to": self.config.compare_to,
                "canary_deployment": self.config.canary_deployment,
                "stable_deployment": self.config.stable_deployment,
                "timestamp": datetime.utcnow().isoformat() + "Z"
            },
            "snapshots": [
                {
                    "baseline": asdict(baseline),
                    "comparison": asdict(comparison),
                    "analyses": [a.to_dict() for a in analyses]
                }
                for baseline, comparison, analyses in self.snapshots
            ]
        }

        try:
            with output_path.open("w") as f:
                json.dump(data, f, indent=2)

            logger.info(f"Saved drift report to {output_path}")

            # Markdown report
            md_path = output_path.with_suffix(".md")
            markdown = self._generate_markdown_report()

            with md_path.open("w") as f:
                f.write(markdown)

            logger.info(f"Saved markdown report to {md_path}")

        except Exception as e:
            logger.error(f"Failed to save report: {e}")

    async def run_once(self) -> Optional[List[DriftAnalysis]]:
        """Run drift detection once"""
        async with PrometheusClient(self.config.prometheus_url) as prom_client:
            collector = MetricsCollector(prom_client, self.config)

            # Determine what to compare
            if self.config.compare_to:
                # Environment comparison (e.g., staging vs production)
                baseline_ns = self.config.compare_namespace or f"tars-{self.config.compare_to}"
                comparison_ns = self.config.namespace

                logger.info(f"Comparing {self.config.compare_to} vs {self.config.environment}")

                baseline = await collector.collect_snapshot(
                    self.config.compare_to,
                    namespace=baseline_ns
                )
                comparison = await collector.collect_snapshot(
                    self.config.environment,
                    namespace=comparison_ns
                )

            elif self.config.canary_deployment and self.config.stable_deployment:
                # Deployment comparison (e.g., canary vs stable)
                logger.info(f"Comparing {self.config.stable_deployment} vs {self.config.canary_deployment}")

                baseline = await collector.collect_snapshot(
                    self.config.stable_deployment,
                    deployment=self.config.stable_deployment
                )
                comparison = await collector.collect_snapshot(
                    self.config.canary_deployment,
                    deployment=self.config.canary_deployment
                )

            else:
                logger.error("Must specify either --compare-to or both --canary-deployment and --stable-deployment")
                return None

        # Analyze drift
        analyzer = DriftAnalyzer(self.config)
        analyses = analyzer.analyze_drift(baseline, comparison)

        # Store snapshot
        self.snapshots.append((baseline, comparison, analyses))

        # Send alert if drift detected
        async with AlertManager(self.config.alert_webhook_url) as alert_mgr:
            await alert_mgr.send_drift_alert(baseline.name, comparison.name, analyses)

        return analyses

    async def run(self):
        """Main drift detection loop"""
        logger.info("Starting drift detection")

        if self.config.duration_minutes:
            # Run for specified duration
            start_time = datetime.utcnow()
            end_time = start_time + timedelta(minutes=self.config.duration_minutes)

            while datetime.utcnow() < end_time:
                await self.run_once()

                # Wait for next interval
                if datetime.utcnow() < end_time:
                    logger.info(f"Waiting {self.config.interval_minutes} minutes until next check...")
                    await asyncio.sleep(self.config.interval_minutes * 60)

        else:
            # Run once
            await self.run_once()

        # Save report
        self._save_report()

        # Print summary
        if self.snapshots:
            _, _, latest_analyses = self.snapshots[-1]
            drift_count = sum(1 for a in latest_analyses if a.is_drift)

            logger.info("=" * 60)
            logger.info("DRIFT DETECTION SUMMARY")
            logger.info("=" * 60)
            logger.info(f"Total metrics analyzed: {len(latest_analyses)}")
            logger.info(f"Metrics with drift: {drift_count}")

            if drift_count > 0:
                critical = sum(1 for a in latest_analyses if a.is_drift and a.severity == "critical")
                warning = sum(1 for a in latest_analyses if a.is_drift and a.severity == "warning")
                logger.warning(f"Critical: {critical}, Warning: {warning}")
            else:
                logger.info("✅ No significant drift detected")

            logger.info("=" * 60)


# ============================================================================
# CLI
# ============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="Drift detector for T.A.R.S. GA deployment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare staging vs production
  python drift_detector.py --environment=production --compare-to=staging

  # Compare canary vs stable deployment
  python drift_detector.py --canary-deployment=canary --stable-deployment=stable

  # Continuous monitoring for 24 hours
  python drift_detector.py --environment=production --compare-to=staging --duration=1440 --interval=15
        """
    )

    parser.add_argument(
        "--environment",
        required=True,
        help="Primary environment to monitor (e.g., production)"
    )

    parser.add_argument(
        "--compare-to",
        help="Environment to compare against (e.g., staging)"
    )

    parser.add_argument(
        "--canary-deployment",
        help="Canary deployment name (alternative to --compare-to)"
    )

    parser.add_argument(
        "--stable-deployment",
        help="Stable deployment name (required with --canary-deployment)"
    )

    parser.add_argument(
        "--prometheus-url",
        default=os.getenv("PROMETHEUS_URL", "http://localhost:9090"),
        help="Prometheus server URL (default: $PROMETHEUS_URL or http://localhost:9090)"
    )

    parser.add_argument(
        "--namespace",
        default="tars-production",
        help="Kubernetes namespace for primary environment (default: tars-production)"
    )

    parser.add_argument(
        "--compare-namespace",
        help="Kubernetes namespace for comparison environment (default: tars-{compare-to})"
    )

    parser.add_argument(
        "--duration",
        type=int,
        help="Monitoring duration in minutes (run once if not specified)"
    )

    parser.add_argument(
        "--interval",
        type=int,
        default=15,
        help="Check interval in minutes (default: 15)"
    )

    parser.add_argument(
        "--threshold-latency",
        type=float,
        default=10.0,
        help="Latency drift threshold percentage (default: 10.0)"
    )

    parser.add_argument(
        "--threshold-error-rate",
        type=float,
        default=5.0,
        help="Error rate drift threshold percentage (default: 5.0)"
    )

    parser.add_argument(
        "--threshold-memory",
        type=float,
        default=20.0,
        help="Memory drift threshold percentage (default: 20.0)"
    )

    parser.add_argument(
        "--threshold-cpu",
        type=float,
        default=15.0,
        help="CPU drift threshold percentage (default: 15.0)"
    )

    parser.add_argument(
        "--alert-webhook",
        default=os.getenv("ALERT_WEBHOOK_URL"),
        help="Webhook URL for alerts (default: $ALERT_WEBHOOK_URL)"
    )

    parser.add_argument(
        "--output",
        default="drift_report.json",
        help="Output JSON file (default: drift_report.json)"
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

    # Validate arguments
    if not args.compare_to and not (args.canary_deployment and args.stable_deployment):
        logger.error("Must specify either --compare-to or both --canary-deployment and --stable-deployment")
        sys.exit(1)

    if args.canary_deployment and not args.stable_deployment:
        logger.error("--stable-deployment required when using --canary-deployment")
        sys.exit(1)

    # Build configuration
    thresholds = DriftThresholds(
        latency_pct=args.threshold_latency,
        error_rate_pct=args.threshold_error_rate,
        memory_pct=args.threshold_memory,
        cpu_pct=args.threshold_cpu
    )

    config = DriftConfig(
        environment=args.environment,
        compare_to=args.compare_to,
        canary_deployment=args.canary_deployment,
        stable_deployment=args.stable_deployment,
        prometheus_url=args.prometheus_url,
        namespace=args.namespace,
        compare_namespace=args.compare_namespace,
        duration_minutes=args.duration,
        interval_minutes=args.interval,
        thresholds=thresholds,
        output_file=args.output,
        alert_webhook_url=args.alert_webhook
    )

    # Run detector
    detector = DriftDetector(config)

    try:
        asyncio.run(detector.run())
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
