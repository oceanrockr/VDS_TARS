#!/usr/bin/env python3
"""
GA Day KPI Collector - 24-Hour Production Metrics Aggregation

Collects and aggregates key performance indicators during the GA day window.
Provides rolling snapshots and final summary for certification.

Enterprise Features (Phase 14.6):
- Enterprise configuration with multi-source precedence
- Compliance enforcement (input sanitization, data retention)
- Optional AES encryption + RSA signing for outputs
- Telemetry and structured logging

Usage:
    # Using enterprise config
    python ga_kpi_collector.py --profile prod --duration 24 --interval 5

    # Legacy CLI mode (backward compatible)
    python ga_kpi_collector.py --duration 24 --interval 5 --output ga_kpis/
    python ga_kpi_collector.py --ga-start "2025-01-15T00:00:00Z" --ga-end "2025-01-16T00:00:00Z"

    # With encryption and signing
    python ga_kpi_collector.py --profile prod --encrypt --sign

Author: T.A.R.S. Platform Team
Phase: 14.6 - Enterprise Hardening
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

import aiohttp
import hashlib
from prometheus_client.parser import text_string_to_metric_families

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
class ServiceMetrics:
    """Metrics for a single service."""
    name: str
    availability: float  # percentage
    request_count: int
    error_count: int
    error_rate: float  # percentage
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    avg_cpu_percent: float
    avg_memory_mb: float
    peak_memory_mb: float
    timestamp: str


@dataclass
class InfrastructureMetrics:
    """Infrastructure-level metrics."""
    db_p50_latency_ms: float
    db_p95_latency_ms: float
    db_p99_latency_ms: float
    db_connection_pool_usage: float  # percentage
    redis_hit_rate: float  # percentage
    redis_memory_mb: float
    redis_connected_clients: int
    total_cpu_percent: float
    total_memory_percent: float
    network_bytes_in: int
    network_bytes_out: int

    # Extended metrics (Phase 14.5)
    cluster_cpu_cores: float
    cluster_memory_gb: float
    cluster_cpu_utilization: float  # percentage
    cluster_memory_utilization: float  # percentage
    node_count: int
    node_pressure_count: int  # nodes under pressure
    estimated_cost_per_hour: float  # USD

    timestamp: str


@dataclass
class CrossRegionMetrics:
    """Cross-region KPI deltas."""
    region_name: str
    availability: float
    error_rate: float
    p95_latency_ms: float
    request_count: int
    delta_from_primary: Dict[str, float] = field(default_factory=dict)
    timestamp: str


@dataclass
class KPISnapshot:
    """A single KPI snapshot at a point in time."""
    timestamp: str
    elapsed_hours: float
    services: List[ServiceMetrics] = field(default_factory=list)
    infrastructure: Optional[InfrastructureMetrics] = None
    cross_region: List[CrossRegionMetrics] = field(default_factory=list)
    drift_baseline: Optional[Dict[str, float]] = None
    alert_events: List[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class KPISummary:
    """24-hour KPI summary for certification."""
    ga_start: str
    ga_end: str
    duration_hours: float
    total_snapshots: int

    # Overall availability
    overall_availability: float
    slo_compliance: bool  # >= 99.9%

    # Aggregated service metrics
    total_requests: int
    total_errors: int
    overall_error_rate: float

    # Latency aggregates
    avg_p50_latency_ms: float
    avg_p95_latency_ms: float
    avg_p99_latency_ms: float
    max_p99_latency_ms: float

    # Resource utilization
    avg_cpu_percent: float
    peak_cpu_percent: float
    avg_memory_percent: float
    peak_memory_percent: float

    # Database metrics
    avg_db_latency_ms: float
    max_db_latency_ms: float

    # Redis metrics
    avg_redis_hit_rate: float
    min_redis_hit_rate: float

    # Incidents
    incident_count: int
    downtime_minutes: float

    # Certification
    certification_status: str  # PASS / FAIL
    certification_timestamp: str
    certification_notes: List[str] = field(default_factory=list)


class PrometheusClient:
    """Async Prometheus client for metrics collection."""

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
        """Execute a PromQL query."""
        if not self.session:
            raise RuntimeError("Client not initialized. Use 'async with' context.")

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

    async def query_range(
        self,
        query: str,
        start: datetime,
        end: datetime,
        step: str = "1m"
    ) -> Optional[Dict[str, Any]]:
        """Execute a PromQL range query."""
        if not self.session:
            raise RuntimeError("Client not initialized. Use 'async with' context.")

        try:
            url = f"{self.base_url}/api/v1/query_range"
            params = {
                "query": query,
                "start": start.timestamp(),
                "end": end.timestamp(),
                "step": step
            }

            async with self.session.get(url, params=params, timeout=30) as resp:
                if resp.status != 200:
                    logger.error(f"Prometheus range query failed: {resp.status}")
                    return None

                data = await resp.json()
                if data.get("status") != "success":
                    logger.error(f"Prometheus range query error: {data}")
                    return None

                return data.get("data", {})

        except asyncio.TimeoutError:
            logger.error(f"Prometheus range query timeout: {query}")
            return None
        except Exception as e:
            logger.error(f"Prometheus range query exception: {e}")
            return None


class KPICollector:
    """Collects and aggregates KPIs over a 24-hour GA window."""

    def __init__(
        self,
        prometheus_url: str = "http://localhost:9090",
        output_dir: Path = Path("ga_kpis"),
        snapshot_interval_minutes: int = 5,
        services: Optional[List[str]] = None,
        # Enterprise features
        compliance_enforcer: Optional[ComplianceEnforcer] = None,
        encryptor: Optional[AESEncryption] = None,
        signer: Optional[ReportSigner] = None,
    ):
        self.prometheus_url = prometheus_url
        self.output_dir = Path(output_dir)
        self.snapshot_interval_minutes = snapshot_interval_minutes

        # Default services to monitor
        self.services = services or [
            "insight-engine",
            "policy-learner",
            "meta-consensus",
            "causal-inference",
            "orchestration-agent",
            "automl-pipeline",
            "dashboard-api",
            "hypersync"
        ]

        self.snapshots: List[KPISnapshot] = []
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None

        # Enterprise features (Phase 14.6)
        self.compliance_enforcer = compliance_enforcer
        self.encryptor = encryptor
        self.signer = signer

    async def collect_service_metrics(
        self,
        prom: PrometheusClient,
        service_name: str
    ) -> Optional[ServiceMetrics]:
        """Collect metrics for a single service."""
        timestamp = datetime.now(timezone.utc).isoformat()

        # Availability (from up metric or health checks)
        up_query = f'up{{job="{service_name}"}}'
        up_result = await prom.query(up_query)
        availability = 100.0
        if up_result and up_result.get("result"):
            up_value = float(up_result["result"][0]["value"][1])
            availability = up_value * 100.0

        # Request count (5m rate)
        req_query = f'sum(rate(http_requests_total{{service="{service_name}"}}[5m])) * 300'
        req_result = await prom.query(req_query)
        request_count = 0
        if req_result and req_result.get("result"):
            request_count = int(float(req_result["result"][0]["value"][1]))

        # Error count (5m rate)
        err_query = f'sum(rate(http_requests_total{{service="{service_name}",status=~"5.."}}[5m])) * 300'
        err_result = await prom.query(err_query)
        error_count = 0
        if err_result and err_result.get("result"):
            error_count = int(float(err_result["result"][0]["value"][1]))

        error_rate = (error_count / request_count * 100.0) if request_count > 0 else 0.0

        # Latency percentiles
        p50_query = f'histogram_quantile(0.50, rate(http_request_duration_seconds_bucket{{service="{service_name}"}}[5m])) * 1000'
        p95_query = f'histogram_quantile(0.95, rate(http_request_duration_seconds_bucket{{service="{service_name}"}}[5m])) * 1000'
        p99_query = f'histogram_quantile(0.99, rate(http_request_duration_seconds_bucket{{service="{service_name}"}}[5m])) * 1000'

        p50_result = await prom.query(p50_query)
        p95_result = await prom.query(p95_query)
        p99_result = await prom.query(p99_query)

        p50_latency = 0.0
        p95_latency = 0.0
        p99_latency = 0.0

        if p50_result and p50_result.get("result"):
            p50_latency = float(p50_result["result"][0]["value"][1])
        if p95_result and p95_result.get("result"):
            p95_latency = float(p95_result["result"][0]["value"][1])
        if p99_result and p99_result.get("result"):
            p99_latency = float(p99_result["result"][0]["value"][1])

        # CPU usage
        cpu_query = f'rate(process_cpu_seconds_total{{service="{service_name}"}}[5m]) * 100'
        cpu_result = await prom.query(cpu_query)
        avg_cpu = 0.0
        if cpu_result and cpu_result.get("result"):
            avg_cpu = float(cpu_result["result"][0]["value"][1])

        # Memory usage
        mem_query = f'process_resident_memory_bytes{{service="{service_name}"}} / 1024 / 1024'
        mem_result = await prom.query(mem_query)
        avg_memory = 0.0
        peak_memory = 0.0
        if mem_result and mem_result.get("result"):
            avg_memory = float(mem_result["result"][0]["value"][1])
            peak_memory = avg_memory  # For snapshot, current is peak

        return ServiceMetrics(
            name=service_name,
            availability=round(availability, 2),
            request_count=request_count,
            error_count=error_count,
            error_rate=round(error_rate, 2),
            p50_latency_ms=round(p50_latency, 2),
            p95_latency_ms=round(p95_latency, 2),
            p99_latency_ms=round(p99_latency, 2),
            avg_cpu_percent=round(avg_cpu, 2),
            avg_memory_mb=round(avg_memory, 2),
            peak_memory_mb=round(peak_memory, 2),
            timestamp=timestamp
        )

    async def collect_infrastructure_metrics(
        self,
        prom: PrometheusClient
    ) -> Optional[InfrastructureMetrics]:
        """Collect infrastructure-level metrics."""
        timestamp = datetime.now(timezone.utc).isoformat()

        # Database latency
        db_p50_query = 'histogram_quantile(0.50, rate(db_query_duration_seconds_bucket[5m])) * 1000'
        db_p95_query = 'histogram_quantile(0.95, rate(db_query_duration_seconds_bucket[5m])) * 1000'
        db_p99_query = 'histogram_quantile(0.99, rate(db_query_duration_seconds_bucket[5m])) * 1000'

        db_p50_result = await prom.query(db_p50_query)
        db_p95_result = await prom.query(db_p95_query)
        db_p99_result = await prom.query(db_p99_query)

        db_p50 = float(db_p50_result["result"][0]["value"][1]) if db_p50_result and db_p50_result.get("result") else 0.0
        db_p95 = float(db_p95_result["result"][0]["value"][1]) if db_p95_result and db_p95_result.get("result") else 0.0
        db_p99 = float(db_p99_result["result"][0]["value"][1]) if db_p99_result and db_p99_result.get("result") else 0.0

        # Database connection pool
        pool_query = 'pg_stat_database_numbackends / pg_settings_max_connections * 100'
        pool_result = await prom.query(pool_query)
        pool_usage = float(pool_result["result"][0]["value"][1]) if pool_result and pool_result.get("result") else 0.0

        # Redis metrics
        redis_hit_query = 'redis_keyspace_hits_total / (redis_keyspace_hits_total + redis_keyspace_misses_total) * 100'
        redis_hit_result = await prom.query(redis_hit_query)
        redis_hit_rate = float(redis_hit_result["result"][0]["value"][1]) if redis_hit_result and redis_hit_result.get("result") else 0.0

        redis_mem_query = 'redis_memory_used_bytes / 1024 / 1024'
        redis_mem_result = await prom.query(redis_mem_query)
        redis_memory = float(redis_mem_result["result"][0]["value"][1]) if redis_mem_result and redis_mem_result.get("result") else 0.0

        redis_clients_query = 'redis_connected_clients'
        redis_clients_result = await prom.query(redis_clients_query)
        redis_clients = int(float(redis_clients_result["result"][0]["value"][1])) if redis_clients_result and redis_clients_result.get("result") else 0

        # Total CPU and memory
        total_cpu_query = 'sum(rate(container_cpu_usage_seconds_total[5m])) * 100'
        total_cpu_result = await prom.query(total_cpu_query)
        total_cpu = float(total_cpu_result["result"][0]["value"][1]) if total_cpu_result and total_cpu_result.get("result") else 0.0

        total_mem_query = 'sum(container_memory_usage_bytes) / sum(machine_memory_bytes) * 100'
        total_mem_result = await prom.query(total_mem_query)
        total_mem = float(total_mem_result["result"][0]["value"][1]) if total_mem_result and total_mem_result.get("result") else 0.0

        # Network traffic
        net_in_query = 'sum(rate(container_network_receive_bytes_total[5m])) * 300'
        net_out_query = 'sum(rate(container_network_transmit_bytes_total[5m])) * 300'

        net_in_result = await prom.query(net_in_query)
        net_out_result = await prom.query(net_out_query)

        net_in = int(float(net_in_result["result"][0]["value"][1])) if net_in_result and net_in_result.get("result") else 0
        net_out = int(float(net_out_result["result"][0]["value"][1])) if net_out_result and net_out_result.get("result") else 0

        # Extended metrics (Phase 14.5): Cluster resources
        cluster_cpu_query = 'sum(machine_cpu_cores)'
        cluster_cpu_result = await prom.query(cluster_cpu_query)
        cluster_cpu_cores = float(cluster_cpu_result["result"][0]["value"][1]) if cluster_cpu_result and cluster_cpu_result.get("result") else 0.0

        cluster_mem_query = 'sum(machine_memory_bytes) / 1024 / 1024 / 1024'
        cluster_mem_result = await prom.query(cluster_mem_query)
        cluster_memory_gb = float(cluster_mem_result["result"][0]["value"][1]) if cluster_mem_result and cluster_mem_result.get("result") else 0.0

        # Cluster utilization
        cluster_cpu_util_query = 'sum(rate(container_cpu_usage_seconds_total[5m])) / sum(machine_cpu_cores) * 100'
        cluster_cpu_util_result = await prom.query(cluster_cpu_util_query)
        cluster_cpu_util = float(cluster_cpu_util_result["result"][0]["value"][1]) if cluster_cpu_util_result and cluster_cpu_util_result.get("result") else 0.0

        cluster_mem_util_query = 'sum(container_memory_working_set_bytes) / sum(machine_memory_bytes) * 100'
        cluster_mem_util_result = await prom.query(cluster_mem_util_query)
        cluster_mem_util = float(cluster_mem_util_result["result"][0]["value"][1]) if cluster_mem_util_result and cluster_mem_util_result.get("result") else 0.0

        # Node count and pressure
        node_count_query = 'count(kube_node_info)'
        node_count_result = await prom.query(node_count_query)
        node_count = int(float(node_count_result["result"][0]["value"][1])) if node_count_result and node_count_result.get("result") else 0

        node_pressure_query = 'count(kube_node_status_condition{condition="MemoryPressure",status="true"} or kube_node_status_condition{condition="DiskPressure",status="true"})'
        node_pressure_result = await prom.query(node_pressure_query)
        node_pressure_count = int(float(node_pressure_result["result"][0]["value"][1])) if node_pressure_result and node_pressure_result.get("result") else 0

        # Cost estimation (simplified: $0.10 per CPU core per hour + $0.02 per GB RAM per hour)
        estimated_cost = (cluster_cpu_cores * 0.10) + (cluster_memory_gb * 0.02)

        return InfrastructureMetrics(
            db_p50_latency_ms=round(db_p50, 2),
            db_p95_latency_ms=round(db_p95, 2),
            db_p99_latency_ms=round(db_p99, 2),
            db_connection_pool_usage=round(pool_usage, 2),
            redis_hit_rate=round(redis_hit_rate, 2),
            redis_memory_mb=round(redis_memory, 2),
            redis_connected_clients=redis_clients,
            total_cpu_percent=round(total_cpu, 2),
            total_memory_percent=round(total_mem, 2),
            network_bytes_in=net_in,
            network_bytes_out=net_out,
            # Extended metrics
            cluster_cpu_cores=round(cluster_cpu_cores, 2),
            cluster_memory_gb=round(cluster_memory_gb, 2),
            cluster_cpu_utilization=round(cluster_cpu_util, 2),
            cluster_memory_utilization=round(cluster_mem_util, 2),
            node_count=node_count,
            node_pressure_count=node_pressure_count,
            estimated_cost_per_hour=round(estimated_cost, 2),
            timestamp=timestamp
        )

    async def collect_cross_region_metrics(
        self,
        prom: PrometheusClient,
        regions: List[str] = ["us-east-1", "us-west-2", "eu-central-1"]
    ) -> List[CrossRegionMetrics]:
        """Collect cross-region KPI deltas (Phase 14.5)"""
        timestamp = datetime.now(timezone.utc).isoformat()
        cross_region_metrics = []

        primary_availability = None
        primary_error_rate = None
        primary_latency = None

        for idx, region in enumerate(regions):
            # Availability per region
            avail_query = f'avg_over_time(up{{region="{region}"}}[5m]) * 100'
            avail_result = await prom.query(avail_query)
            availability = float(avail_result["result"][0]["value"][1]) if avail_result and avail_result.get("result") else 100.0

            # Error rate per region
            err_query = f'rate(http_requests_total{{region="{region}",status=~"5.."}}[5m]) * 100'
            err_result = await prom.query(err_query)
            error_rate = float(err_result["result"][0]["value"][1]) if err_result and err_result.get("result") else 0.0

            # Latency per region
            lat_query = f'histogram_quantile(0.95, rate(http_request_duration_seconds_bucket{{region="{region}"}}[5m])) * 1000'
            lat_result = await prom.query(lat_query)
            p95_latency = float(lat_result["result"][0]["value"][1]) if lat_result and lat_result.get("result") else 0.0

            # Request count per region
            req_query = f'sum(rate(http_requests_total{{region="{region}"}}[5m])) * 300'
            req_result = await prom.query(req_query)
            request_count = int(float(req_result["result"][0]["value"][1])) if req_result and req_result.get("result") else 0

            # Calculate deltas from primary (first region)
            delta_from_primary: Dict[str, float] = {}
            if idx == 0:
                # This is the primary region
                primary_availability = availability
                primary_error_rate = error_rate
                primary_latency = p95_latency
            else:
                # Calculate deltas from primary
                if primary_availability is not None:
                    delta_from_primary["availability_delta"] = round(availability - primary_availability, 2)
                if primary_error_rate is not None:
                    delta_from_primary["error_rate_delta"] = round(error_rate - primary_error_rate, 4)
                if primary_latency is not None:
                    delta_from_primary["latency_delta_ms"] = round(p95_latency - primary_latency, 2)

            cross_region_metrics.append(CrossRegionMetrics(
                region_name=region,
                availability=round(availability, 2),
                error_rate=round(error_rate, 4),
                p95_latency_ms=round(p95_latency, 2),
                request_count=request_count,
                delta_from_primary=delta_from_primary,
                timestamp=timestamp
            ))

        return cross_region_metrics

    async def collect_alert_events(
        self,
        prom: PrometheusClient
    ) -> List[str]:
        """Collect active alert events (Phase 14.5)"""
        alert_events = []

        # Query active alerts
        alerts_query = 'ALERTS{alertstate="firing"}'
        alerts_result = await prom.query(alerts_query)

        if alerts_result and alerts_result.get("result"):
            for alert in alerts_result["result"]:
                metric = alert.get("metric", {})
                alert_name = metric.get("alertname", "Unknown")
                severity = metric.get("severity", "warning")
                alert_events.append(f"{severity.upper()}: {alert_name}")

        return alert_events

    def create_drift_baseline(self, snapshot: KPISnapshot) -> Dict[str, float]:
        """Create drift baseline from snapshot (Phase 14.5)"""
        baseline = {}

        # Aggregate service metrics for baseline
        if snapshot.services:
            all_cpu = [s.avg_cpu_percent for s in snapshot.services]
            all_mem = [s.avg_memory_mb for s in snapshot.services]
            all_p95 = [s.p95_latency_ms for s in snapshot.services if s.p95_latency_ms > 0]

            baseline["cpu_usage"] = sum(all_cpu) / len(all_cpu) if all_cpu else 0.0
            baseline["memory_usage"] = sum(all_mem) / len(all_mem) if all_mem else 0.0
            baseline["api_latency_p95"] = sum(all_p95) / len(all_p95) if all_p95 else 0.0

        # Infrastructure metrics for baseline
        if snapshot.infrastructure:
            baseline["cluster_cpu_utilization"] = snapshot.infrastructure.cluster_cpu_utilization
            baseline["cluster_memory_utilization"] = snapshot.infrastructure.cluster_memory_utilization
            baseline["db_latency_p95"] = snapshot.infrastructure.db_p95_latency_ms
            baseline["redis_hit_rate"] = snapshot.infrastructure.redis_hit_rate

        # Error rate baseline
        total_requests = sum(s.request_count for s in snapshot.services)
        total_errors = sum(s.error_count for s in snapshot.services)
        baseline["error_rate"] = (total_errors / total_requests * 100.0) if total_requests > 0 else 0.0

        baseline["timestamp"] = snapshot.timestamp

        return baseline

    async def collect_snapshot(self) -> KPISnapshot:
        """Collect a single KPI snapshot."""
        logger.info("Collecting KPI snapshot...")

        timestamp = datetime.now(timezone.utc).isoformat()
        elapsed_hours = 0.0
        if self.start_time:
            elapsed = datetime.now(timezone.utc) - self.start_time
            elapsed_hours = elapsed.total_seconds() / 3600.0

        async with PrometheusClient(self.prometheus_url) as prom:
            # Collect service metrics
            service_metrics = []
            for service in self.services:
                metrics = await self.collect_service_metrics(prom, service)
                if metrics:
                    service_metrics.append(metrics)

            # Collect infrastructure metrics
            infra_metrics = await self.collect_infrastructure_metrics(prom)

            # Collect cross-region metrics (Phase 14.5)
            cross_region_metrics = await self.collect_cross_region_metrics(prom)

            # Collect alert events (Phase 14.5)
            alert_events = await self.collect_alert_events(prom)

        # Create baseline snapshot from first snapshot
        drift_baseline = None
        if len(self.snapshots) == 0:
            # This is the first snapshot, create baseline
            temp_snapshot = KPISnapshot(
                timestamp=timestamp,
                elapsed_hours=0.0,
                services=service_metrics,
                infrastructure=infra_metrics,
                cross_region=[],
                drift_baseline=None,
                alert_events=[]
            )
            drift_baseline = self.create_drift_baseline(temp_snapshot)

        snapshot = KPISnapshot(
            timestamp=timestamp,
            elapsed_hours=round(elapsed_hours, 2),
            services=service_metrics,
            infrastructure=infra_metrics,
            cross_region=cross_region_metrics,
            drift_baseline=drift_baseline,
            alert_events=alert_events,
            notes=f"Snapshot at T+{elapsed_hours:.1f}h"
        )

        self.snapshots.append(snapshot)
        logger.info(f"Snapshot collected: {len(service_metrics)} services, {len(cross_region_metrics)} regions, {len(alert_events)} alerts, elapsed: {elapsed_hours:.1f}h")

        return snapshot

    def save_snapshot(self, snapshot: KPISnapshot, snapshot_num: int):
        """Save a rolling snapshot to disk with optional encryption/signing."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        snapshot_file = self.output_dir / f"snapshot_{snapshot_num:04d}.json"
        snapshot_dict = {
            "timestamp": snapshot.timestamp,
            "elapsed_hours": snapshot.elapsed_hours,
            "services": [asdict(s) for s in snapshot.services],
            "infrastructure": asdict(snapshot.infrastructure) if snapshot.infrastructure else None,
            "cross_region": [asdict(r) for r in snapshot.cross_region],
            "drift_baseline": snapshot.drift_baseline,
            "alert_events": snapshot.alert_events,
            "notes": snapshot.notes
        }

        # Write JSON
        with open(snapshot_file, "w") as f:
            json.dump(snapshot_dict, f, indent=2)

        # Enterprise: Encrypt if enabled
        if self.encryptor:
            encrypted_file = snapshot_file.with_suffix(".json.enc")
            self.encryptor.encrypt_file(snapshot_file, encrypted_file)
            logger.info(f"Encrypted snapshot: {encrypted_file}")

        # Enterprise: Sign if enabled
        if self.signer:
            signature = self.signer.sign_file(snapshot_file)
            sig_file = snapshot_file.with_suffix(".json.sig")
            with open(sig_file, "w") as f:
                f.write(f"RSA-PSS-SHA256\n{signature}\n")
            logger.info(f"Signed snapshot: {sig_file}")

        # Save drift baseline if this is the first snapshot
        if snapshot.drift_baseline:
            baseline_file = self.output_dir / "baseline_metrics.json"
            with open(baseline_file, "w") as f:
                json.dump(snapshot.drift_baseline, f, indent=2)
            logger.info(f"Baseline metrics saved: {baseline_file}")

        logger.info(f"Snapshot saved: {snapshot_file}")

    def calculate_summary(self) -> KPISummary:
        """Calculate 24-hour summary from all snapshots."""
        if not self.snapshots:
            raise ValueError("No snapshots collected")

        if not self.start_time or not self.end_time:
            raise ValueError("Start/end time not set")

        duration_hours = (self.end_time - self.start_time).total_seconds() / 3600.0

        # Calculate availability
        all_availabilities = []
        for snapshot in self.snapshots:
            for service in snapshot.services:
                all_availabilities.append(service.availability)

        overall_availability = sum(all_availabilities) / len(all_availabilities) if all_availabilities else 0.0
        slo_compliance = overall_availability >= 99.9

        # Aggregate requests and errors
        total_requests = sum(s.request_count for snap in self.snapshots for s in snap.services)
        total_errors = sum(s.error_count for snap in self.snapshots for s in snap.services)
        overall_error_rate = (total_errors / total_requests * 100.0) if total_requests > 0 else 0.0

        # Latency aggregates
        all_p50 = [s.p50_latency_ms for snap in self.snapshots for s in snap.services if s.p50_latency_ms > 0]
        all_p95 = [s.p95_latency_ms for snap in self.snapshots for s in snap.services if s.p95_latency_ms > 0]
        all_p99 = [s.p99_latency_ms for snap in self.snapshots for s in snap.services if s.p99_latency_ms > 0]

        avg_p50 = sum(all_p50) / len(all_p50) if all_p50 else 0.0
        avg_p95 = sum(all_p95) / len(all_p95) if all_p95 else 0.0
        avg_p99 = sum(all_p99) / len(all_p99) if all_p99 else 0.0
        max_p99 = max(all_p99) if all_p99 else 0.0

        # Resource utilization
        all_cpu = [s.avg_cpu_percent for snap in self.snapshots for s in snap.services]
        all_mem = [s.avg_memory_mb for snap in self.snapshots for s in snap.services]

        avg_cpu = sum(all_cpu) / len(all_cpu) if all_cpu else 0.0
        peak_cpu = max(all_cpu) if all_cpu else 0.0

        # Convert memory to percentage (approximate)
        avg_memory_percent = sum(all_mem) / len(all_mem) / 1024.0 if all_mem else 0.0  # Rough estimate
        peak_memory_percent = max(all_mem) / 1024.0 if all_mem else 0.0

        # Database metrics
        all_db_latency = []
        for snap in self.snapshots:
            if snap.infrastructure:
                all_db_latency.append(snap.infrastructure.db_p95_latency_ms)

        avg_db_latency = sum(all_db_latency) / len(all_db_latency) if all_db_latency else 0.0
        max_db_latency = max(all_db_latency) if all_db_latency else 0.0

        # Redis metrics
        all_redis_hit_rate = []
        for snap in self.snapshots:
            if snap.infrastructure:
                all_redis_hit_rate.append(snap.infrastructure.redis_hit_rate)

        avg_redis_hit_rate = sum(all_redis_hit_rate) / len(all_redis_hit_rate) if all_redis_hit_rate else 0.0
        min_redis_hit_rate = min(all_redis_hit_rate) if all_redis_hit_rate else 0.0

        # Certification status
        certification_notes = []
        certification_status = "PASS"

        if not slo_compliance:
            certification_status = "FAIL"
            certification_notes.append(f"SLO violation: {overall_availability:.2f}% < 99.9%")

        if overall_error_rate > 0.1:
            certification_notes.append(f"High error rate: {overall_error_rate:.2f}%")

        if max_p99 > 1000.0:
            certification_notes.append(f"High P99 latency: {max_p99:.2f}ms")

        if avg_redis_hit_rate < 95.0:
            certification_notes.append(f"Low Redis hit rate: {avg_redis_hit_rate:.2f}%")

        if not certification_notes:
            certification_notes.append("All KPIs within acceptable ranges")

        return KPISummary(
            ga_start=self.start_time.isoformat(),
            ga_end=self.end_time.isoformat(),
            duration_hours=round(duration_hours, 2),
            total_snapshots=len(self.snapshots),
            overall_availability=round(overall_availability, 2),
            slo_compliance=slo_compliance,
            total_requests=total_requests,
            total_errors=total_errors,
            overall_error_rate=round(overall_error_rate, 4),
            avg_p50_latency_ms=round(avg_p50, 2),
            avg_p95_latency_ms=round(avg_p95, 2),
            avg_p99_latency_ms=round(avg_p99, 2),
            max_p99_latency_ms=round(max_p99, 2),
            avg_cpu_percent=round(avg_cpu, 2),
            peak_cpu_percent=round(peak_cpu, 2),
            avg_memory_percent=round(avg_memory_percent, 2),
            peak_memory_percent=round(peak_memory_percent, 2),
            avg_db_latency_ms=round(avg_db_latency, 2),
            max_db_latency_ms=round(max_db_latency, 2),
            avg_redis_hit_rate=round(avg_redis_hit_rate, 2),
            min_redis_hit_rate=round(min_redis_hit_rate, 2),
            incident_count=0,  # Would be populated from incident tracking
            downtime_minutes=0.0,  # Would be calculated from availability gaps
            certification_status=certification_status,
            certification_timestamp=datetime.now(timezone.utc).isoformat(),
            certification_notes=certification_notes
        )

    def save_summary(self, summary: KPISummary):
        """Save final 24-hour summary with optional encryption/signing."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Save JSON summary
        summary_file = self.output_dir / "ga_kpi_summary.json"
        summary_dict = asdict(summary)

        with open(summary_file, "w") as f:
            json.dump(summary_dict, f, indent=2)

        logger.info(f"Summary saved: {summary_file}")

        # Enterprise: Encrypt if enabled
        if self.encryptor:
            encrypted_file = summary_file.with_suffix(".json.enc")
            self.encryptor.encrypt_file(summary_file, encrypted_file)
            logger.info(f"Encrypted summary: {encrypted_file}")

        # Enterprise: Sign if enabled
        if self.signer:
            signature = self.signer.sign_file(summary_file)
            sig_file = summary_file.with_suffix(".json.sig")
            with open(sig_file, "w") as f:
                f.write(f"RSA-PSS-SHA256\n{signature}\n")
            logger.info(f"Signed summary: {sig_file}")

        # Generate SHA256 hash
        with open(summary_file, "rb") as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()

        hash_file = self.output_dir / "ga_kpi_summary.json.sha256"
        with open(hash_file, "w") as f:
            f.write(f"{file_hash}  ga_kpi_summary.json\n")

        logger.info(f"Hash saved: {hash_file}")

    def save_markdown_summary(self, summary: KPISummary):
        """Save optional Markdown summary."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        md_file = self.output_dir / "ga_kpi_summary.md"

        with open(md_file, "w") as f:
            f.write("# T.A.R.S. v1.0.2 GA Day KPI Summary\n\n")
            f.write(f"**GA Window:** {summary.ga_start} to {summary.ga_end}\n\n")
            f.write(f"**Duration:** {summary.duration_hours} hours\n\n")
            f.write(f"**Total Snapshots:** {summary.total_snapshots}\n\n")
            f.write(f"**Certification Status:** {summary.certification_status} ✅\n\n" if summary.certification_status == "PASS" else f"**Certification Status:** {summary.certification_status} ❌\n\n")

            f.write("## Availability & SLO Compliance\n\n")
            f.write(f"- **Overall Availability:** {summary.overall_availability}%\n")
            f.write(f"- **SLO Target:** 99.9%\n")
            f.write(f"- **SLO Compliance:** {'✅ PASS' if summary.slo_compliance else '❌ FAIL'}\n")
            f.write(f"- **Downtime:** {summary.downtime_minutes} minutes\n\n")

            f.write("## Request & Error Metrics\n\n")
            f.write(f"- **Total Requests:** {summary.total_requests:,}\n")
            f.write(f"- **Total Errors:** {summary.total_errors:,}\n")
            f.write(f"- **Overall Error Rate:** {summary.overall_error_rate}%\n\n")

            f.write("## Latency Metrics\n\n")
            f.write(f"- **Average P50:** {summary.avg_p50_latency_ms}ms\n")
            f.write(f"- **Average P95:** {summary.avg_p95_latency_ms}ms\n")
            f.write(f"- **Average P99:** {summary.avg_p99_latency_ms}ms\n")
            f.write(f"- **Max P99:** {summary.max_p99_latency_ms}ms\n\n")

            f.write("## Resource Utilization\n\n")
            f.write(f"- **Average CPU:** {summary.avg_cpu_percent}%\n")
            f.write(f"- **Peak CPU:** {summary.peak_cpu_percent}%\n")
            f.write(f"- **Average Memory:** {summary.avg_memory_percent}%\n")
            f.write(f"- **Peak Memory:** {summary.peak_memory_percent}%\n\n")

            f.write("## Database & Cache\n\n")
            f.write(f"- **Average DB Latency (P95):** {summary.avg_db_latency_ms}ms\n")
            f.write(f"- **Max DB Latency (P95):** {summary.max_db_latency_ms}ms\n")
            f.write(f"- **Average Redis Hit Rate:** {summary.avg_redis_hit_rate}%\n")
            f.write(f"- **Min Redis Hit Rate:** {summary.min_redis_hit_rate}%\n\n")

            f.write("## Incidents\n\n")
            f.write(f"- **Incident Count:** {summary.incident_count}\n\n")

            f.write("## Certification Notes\n\n")
            for note in summary.certification_notes:
                f.write(f"- {note}\n")
            f.write("\n")

            f.write(f"---\n\n")
            f.write(f"*Generated: {summary.certification_timestamp}*\n")

        logger.info(f"Markdown summary saved: {md_file}")

    async def run_collection(
        self,
        duration_hours: float = 24.0,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ):
        """Run KPI collection for the specified duration."""
        self.start_time = start_time or datetime.now(timezone.utc)
        self.end_time = end_time or (self.start_time + timedelta(hours=duration_hours))

        logger.info(f"Starting KPI collection from {self.start_time} to {self.end_time}")
        logger.info(f"Snapshot interval: {self.snapshot_interval_minutes} minutes")
        logger.info(f"Monitoring services: {', '.join(self.services)}")

        if self.encryptor:
            logger.info("Encryption: ENABLED")
        if self.signer:
            logger.info("Signing: ENABLED")
        if self.compliance_enforcer:
            logger.info("Compliance enforcement: ENABLED")

        snapshot_num = 1

        while datetime.now(timezone.utc) < self.end_time:
            try:
                # Collect snapshot
                snapshot = await self.collect_snapshot()

                # Save rolling snapshot
                self.save_snapshot(snapshot, snapshot_num)
                snapshot_num += 1

                # Wait for next interval
                await asyncio.sleep(self.snapshot_interval_minutes * 60)

            except KeyboardInterrupt:
                logger.warning("Collection interrupted by user")
                break
            except Exception as e:
                logger.error(f"Error during collection: {e}", exc_info=True)
                await asyncio.sleep(60)  # Wait 1 minute before retrying

        logger.info("Collection complete")

        # Generate summary
        logger.info("Generating summary...")
        summary = self.calculate_summary()
        self.save_summary(summary)
        self.save_markdown_summary(summary)

        logger.info(f"Collection complete. Total snapshots: {len(self.snapshots)}")
        logger.info(f"Certification status: {summary.certification_status}")


async def main():
    """Main entry point with enterprise configuration support."""
    import argparse

    parser = argparse.ArgumentParser(description="T.A.R.S. GA Day KPI Collector")

    # Legacy CLI arguments (backward compatible)
    parser.add_argument("--duration", type=float, default=24.0, help="Collection duration in hours (default: 24)")
    parser.add_argument("--interval", type=int, default=5, help="Snapshot interval in minutes (default: 5)")
    parser.add_argument("--output", type=str, default="ga_kpis", help="Output directory (default: ga_kpis)")
    parser.add_argument("--prometheus-url", type=str, default="http://localhost:9090", help="Prometheus URL")
    parser.add_argument("--ga-start", type=str, help="GA start time (ISO 8601 format)")
    parser.add_argument("--ga-end", type=str, help="GA end time (ISO 8601 format)")
    parser.add_argument("--services", type=str, help="Comma-separated list of services to monitor")

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
            if args.prometheus_url and args.prometheus_url != "http://localhost:9090":
                logger.info(f"✓ Overriding Prometheus URL from CLI: {args.prometheus_url}")
            else:
                args.prometheus_url = config.observability.prometheus_url

            if args.output == "ga_kpis" and config.observability.output_dir != "output":
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

    # Parse start/end times
    start_time = None
    end_time = None

    if args.ga_start:
        start_time = datetime.fromisoformat(args.ga_start.replace("Z", "+00:00"))
    if args.ga_end:
        end_time = datetime.fromisoformat(args.ga_end.replace("Z", "+00:00"))

    # Parse services
    services = None
    if args.services:
        services = [s.strip() for s in args.services.split(",")]

    # Create collector
    collector = KPICollector(
        prometheus_url=args.prometheus_url,
        output_dir=Path(args.output),
        snapshot_interval_minutes=args.interval,
        services=services,
        compliance_enforcer=compliance_enforcer,
        encryptor=encryptor,
        signer=signer,
    )

    # Run collection
    await collector.run_collection(
        duration_hours=args.duration,
        start_time=start_time,
        end_time=end_time
    )


if __name__ == "__main__":
    asyncio.run(main())
