#!/usr/bin/env python3
"""
GA 7-Day Retrospective Generator - Automated Post-GA Analysis

Auto-generates comprehensive retrospective document combining:
- GA Day metrics
- 7-Day stability monitoring
- Regression analysis
- Anomaly detection results
- Cost analysis
- SLO burn-down
- Recommendations for v1.0.2

Output: docs/final/GA_7DAY_RETROSPECTIVE.md

Usage:
    python generate_retrospective.py --ga-data ga_kpis/ --7day-data stability/ --regression regression_summary.json
    python generate_retrospective.py --auto  # Auto-detect all data files

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
class SuccessMetric:
    """
    A success metric from the GA + 7-day period.
    """
    category: str  # "availability", "performance", "stability", "cost"
    description: str
    metric_name: str
    target_value: str
    actual_value: str
    achievement_percent: float  # % of target achieved


@dataclass
class DegradationEvent:
    """
    A degradation or regression event.
    """
    day_occurred: int  # 0 = GA day, 1-7 = post-GA days
    category: str  # "performance", "resource", "availability"
    description: str
    severity: str  # "critical", "high", "medium", "low"
    impact: str
    resolution_status: str  # "resolved", "mitigated", "open"
    resolution_details: Optional[str] = None


@dataclass
class UnexpectedDrift:
    """
    An unexpected drift event.
    """
    metric_name: str
    baseline_value: float
    final_value: float
    drift_percent: float
    trend: str  # "increasing", "decreasing", "volatile"
    potential_causes: List[str] = field(default_factory=list)
    investigation_needed: bool = False


@dataclass
class CostAnalysis:
    """
    Cost analysis for the 7-day period.
    """
    ga_day_cost: float  # USD
    seven_day_total_cost: float  # USD
    daily_average_cost: float  # USD
    cost_trend: str  # "increasing", "stable", "decreasing"
    cost_breakdown: Dict[str, float] = field(default_factory=dict)  # By resource type
    cost_optimization_recommendations: List[str] = field(default_factory=list)


@dataclass
class SLOBurnDown:
    """
    SLO burn-down analysis (JSON only, not in Markdown).
    """
    slo_name: str
    target: float
    budget: float  # Error budget
    budget_consumed_percent: float
    days_to_exhaustion: Optional[float]  # Days until budget exhausted (if trending)
    compliance_by_day: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class RetrospectiveData:
    """
    Complete retrospective data structure.
    """
    generation_timestamp: str
    ga_day_timestamp: str
    seven_day_end_timestamp: str

    # Successes
    successes: List[SuccessMetric] = field(default_factory=list)

    # Degradations
    degradations: List[DegradationEvent] = field(default_factory=list)

    # Unexpected Drifts
    unexpected_drifts: List[UnexpectedDrift] = field(default_factory=list)

    # Cost Analysis
    cost_analysis: Optional[CostAnalysis] = None

    # SLO Burn-Down (JSON only)
    slo_burn_downs: List[SLOBurnDown] = field(default_factory=list)

    # Recommendations for v1.0.2
    recommendations_v1_0_2: List[str] = field(default_factory=list)

    # Process Improvements
    process_improvements: List[str] = field(default_factory=list)

    # Action Items
    action_items: List[Dict[str, str]] = field(default_factory=list)


class DataLoader:
    """
    Loads all required data for retrospective generation.
    """

    def __init__(
        self,
        ga_data_dir: str,
        seven_day_data_dir: str,
        regression_file: str,
        anomaly_file: str = "anomaly_events.json"
    ):
        """
        Initialize data loader.

        Args:
            ga_data_dir: Path to GA Day data directory
            seven_day_data_dir: Path to 7-day stability data directory
            regression_file: Path to regression analysis JSON
            anomaly_file: Path to anomaly events JSON
        """
        self.ga_data_dir = Path(ga_data_dir)
        self.seven_day_data_dir = Path(seven_day_data_dir)
        self.regression_file = regression_file
        self.anomaly_file = anomaly_file

        logger.info("DataLoader initialized")

    def load_ga_kpi_summary(self) -> Dict[str, Any]:
        """
        Load GA Day KPI summary.

        Returns:
            GA KPI summary dict
        """
        ga_kpi_file = self.ga_data_dir / "ga_kpi_summary.json"
        if not ga_kpi_file.exists():
            logger.error(f"GA KPI summary file not found: {ga_kpi_file}")
            raise FileNotFoundError(f"GA KPI summary file not found: {ga_kpi_file}")

        with open(ga_kpi_file, 'r') as f:
            data = json.load(f)

        logger.info(f"Loaded GA KPI summary from {ga_kpi_file}")
        return data

    def load_seven_day_summaries(self) -> List[Dict[str, Any]]:
        """
        Load all 7-day daily summaries.

        Returns:
            List of daily summary dicts (day 1-7)
        """
        summaries = []
        for day in range(1, 8):
            day_file = self.seven_day_data_dir / f"day_{day:02d}_summary.json"
            if not day_file.exists():
                logger.warning(f"Daily summary file not found: {day_file}")
                continue

            with open(day_file, 'r') as f:
                data = json.load(f)
                data['day_number'] = day  # Add day number for reference
                summaries.append(data)

        logger.info(f"Loaded {len(summaries)} daily summaries")
        return summaries

    def load_regression_analysis(self) -> Dict[str, Any]:
        """
        Load regression analysis summary.

        Returns:
            Regression summary dict
        """
        regression_path = Path(self.regression_file)
        if not regression_path.exists():
            logger.warning(f"Regression analysis file not found: {regression_path}")
            return {}

        with open(regression_path, 'r') as f:
            data = json.load(f)

        logger.info(f"Loaded regression analysis from {regression_path}")
        return data

    def load_anomaly_events(self) -> List[Dict[str, Any]]:
        """
        Load anomaly events.

        Returns:
            List of anomaly event dicts
        """
        anomaly_path = Path(self.anomaly_file)
        if not anomaly_path.exists():
            logger.warning(f"Anomaly events file not found: {anomaly_path}")
            return []

        with open(anomaly_path, 'r') as f:
            data = json.load(f)

        # Handle both array and object formats
        if isinstance(data, dict) and 'anomalies' in data:
            events = data['anomalies']
        elif isinstance(data, list):
            events = data
        else:
            events = []

        logger.info(f"Loaded {len(events)} anomaly events from {anomaly_path}")
        return events

    def load_health_reports(self) -> List[Dict[str, Any]]:
        """
        Load health reports from the 7-day period.

        Returns:
            List of health report dicts
        """
        reports = []
        for day in range(1, 8):
            report_file = self.seven_day_data_dir / f"day_{day:02d}_health_report.json"
            if not report_file.exists():
                logger.debug(f"Health report not found: {report_file}")
                continue

            with open(report_file, 'r') as f:
                data = json.load(f)
                data['day_number'] = day
                reports.append(data)

        logger.info(f"Loaded {len(reports)} health reports")
        return reports


class SuccessAnalyzer:
    """
    Analyzes and extracts success metrics.
    """

    def extract_successes(
        self,
        ga_kpi: Dict[str, Any],
        seven_day_summaries: List[Dict[str, Any]]
    ) -> List[SuccessMetric]:
        """
        Extract success metrics from GA + 7-day data.

        Successes include:
        - Availability >= 99.9%
        - P99 latency < 500ms
        - Error rate < 0.1%
        - Zero critical incidents
        - Stable resource utilization

        Args:
            ga_kpi: GA Day KPI summary
            seven_day_summaries: 7-day daily summaries

        Returns:
            List of SuccessMetric objects
        """
        successes = []

        # Calculate 7-day averages
        if not seven_day_summaries:
            logger.warning("No 7-day summaries available for success extraction")
            return successes

        # Availability success
        avg_availability = sum(s.get('overall_availability', 0) for s in seven_day_summaries) / len(seven_day_summaries)
        if avg_availability >= 99.9:
            successes.append(SuccessMetric(
                category="availability",
                description="Maintained high availability throughout 7-day period",
                metric_name="overall_availability",
                target_value="99.9%",
                actual_value=f"{avg_availability:.3f}%",
                achievement_percent=(avg_availability / 99.9) * 100
            ))

        # Error rate success
        avg_error_rate = sum(s.get('overall_error_rate', 0) for s in seven_day_summaries) / len(seven_day_summaries)
        if avg_error_rate < 0.1:
            successes.append(SuccessMetric(
                category="performance",
                description="Error rate remained below SLO target",
                metric_name="overall_error_rate",
                target_value="< 0.1%",
                actual_value=f"{avg_error_rate:.4f}%",
                achievement_percent=((0.1 - avg_error_rate) / 0.1) * 100
            ))

        # P99 latency success
        avg_p99 = sum(s.get('avg_p99_latency_ms', 0) for s in seven_day_summaries) / len(seven_day_summaries)
        if avg_p99 < 500:
            successes.append(SuccessMetric(
                category="performance",
                description="P99 latency consistently below 500ms target",
                metric_name="avg_p99_latency_ms",
                target_value="< 500ms",
                actual_value=f"{avg_p99:.2f}ms",
                achievement_percent=((500 - avg_p99) / 500) * 100
            ))

        # CPU stability
        avg_cpu = sum(s.get('avg_cpu_percent', 0) for s in seven_day_summaries) / len(seven_day_summaries)
        if avg_cpu < 60:
            successes.append(SuccessMetric(
                category="stability",
                description="CPU utilization remained stable and within acceptable range",
                metric_name="avg_cpu_percent",
                target_value="< 60%",
                actual_value=f"{avg_cpu:.2f}%",
                achievement_percent=((60 - avg_cpu) / 60) * 100
            ))

        # Memory stability
        avg_memory = sum(s.get('avg_memory_percent', 0) for s in seven_day_summaries) / len(seven_day_summaries)
        if avg_memory < 70:
            successes.append(SuccessMetric(
                category="stability",
                description="Memory utilization remained stable and within acceptable range",
                metric_name="avg_memory_percent",
                target_value="< 70%",
                actual_value=f"{avg_memory:.2f}%",
                achievement_percent=((70 - avg_memory) / 70) * 100
            ))

        # Redis hit rate success
        avg_redis_hit_rate = sum(s.get('avg_redis_hit_rate', 0) for s in seven_day_summaries) / len(seven_day_summaries)
        if avg_redis_hit_rate > 90:
            successes.append(SuccessMetric(
                category="performance",
                description="Redis cache maintained high hit rate",
                metric_name="avg_redis_hit_rate",
                target_value="> 90%",
                actual_value=f"{avg_redis_hit_rate:.2f}%",
                achievement_percent=(avg_redis_hit_rate / 90) * 100
            ))

        # Check for zero critical incidents
        critical_count = sum(s.get('critical_incident_count', 0) for s in seven_day_summaries)
        if critical_count == 0:
            successes.append(SuccessMetric(
                category="stability",
                description="Zero critical incidents during 7-day period",
                metric_name="critical_incident_count",
                target_value="0",
                actual_value="0",
                achievement_percent=100.0
            ))

        # Database performance
        avg_db_p95 = sum(s.get('avg_db_p95_latency_ms', 0) for s in seven_day_summaries) / len(seven_day_summaries)
        if avg_db_p95 < 100:
            successes.append(SuccessMetric(
                category="performance",
                description="Database P95 latency remained optimal",
                metric_name="avg_db_p95_latency_ms",
                target_value="< 100ms",
                actual_value=f"{avg_db_p95:.2f}ms",
                achievement_percent=((100 - avg_db_p95) / 100) * 100
            ))

        logger.info(f"Extracted {len(successes)} success metrics")
        return successes


class DegradationAnalyzer:
    """
    Analyzes degradations and regressions.
    """

    def extract_degradations(
        self,
        regression_data: Dict[str, Any],
        seven_day_summaries: List[Dict[str, Any]]
    ) -> List[DegradationEvent]:
        """
        Extract degradation events from regression analysis and daily summaries.

        Args:
            regression_data: Regression analysis summary
            seven_day_summaries: 7-day daily summaries

        Returns:
            List of DegradationEvent objects
        """
        degradations = []

        # Extract from regression analysis
        if regression_data and 'regressions' in regression_data:
            for regression in regression_data['regressions']:
                # Only include medium severity and above
                if regression.get('severity') in ['critical', 'high', 'medium']:
                    degradations.append(DegradationEvent(
                        day_occurred=0,  # From regression analysis (GA comparison)
                        category=regression.get('category', 'unknown'),
                        description=regression.get('impact', ''),
                        severity=regression.get('severity', 'unknown'),
                        impact=regression.get('impact', ''),
                        resolution_status="open",
                        resolution_details=None
                    ))

        # Extract from daily summaries
        for summary in seven_day_summaries:
            day = summary.get('day_number', 0)

            # Check for availability drops
            availability = summary.get('overall_availability', 100)
            if availability < 99.9:
                degradations.append(DegradationEvent(
                    day_occurred=day,
                    category="availability",
                    description=f"Availability dropped to {availability:.3f}% on Day {day}",
                    severity="high" if availability < 99.0 else "medium",
                    impact=f"SLO violation: availability {availability:.3f}% < 99.9% target",
                    resolution_status="resolved" if day < 7 else "open",
                    resolution_details=f"Recovered by Day {day + 1}" if day < 7 else None
                ))

            # Check for error rate spikes
            error_rate = summary.get('overall_error_rate', 0)
            if error_rate > 0.1:
                degradations.append(DegradationEvent(
                    day_occurred=day,
                    category="performance",
                    description=f"Error rate spike to {error_rate:.4f}% on Day {day}",
                    severity="high" if error_rate > 1.0 else "medium",
                    impact=f"SLO violation: error rate {error_rate:.4f}% > 0.1% target",
                    resolution_status="resolved" if day < 7 else "open",
                    resolution_details=None
                ))

            # Check for latency spikes
            p99_latency = summary.get('avg_p99_latency_ms', 0)
            if p99_latency > 500:
                degradations.append(DegradationEvent(
                    day_occurred=day,
                    category="performance",
                    description=f"P99 latency spike to {p99_latency:.2f}ms on Day {day}",
                    severity="high" if p99_latency > 1000 else "medium",
                    impact=f"Performance degradation: P99 latency {p99_latency:.2f}ms > 500ms target",
                    resolution_status="resolved" if day < 7 else "open",
                    resolution_details=None
                ))

            # Check for resource exhaustion
            peak_cpu = summary.get('peak_cpu_percent', 0)
            if peak_cpu > 80:
                degradations.append(DegradationEvent(
                    day_occurred=day,
                    category="resource",
                    description=f"CPU utilization peaked at {peak_cpu:.2f}% on Day {day}",
                    severity="critical" if peak_cpu > 95 else "high" if peak_cpu > 90 else "medium",
                    impact=f"Resource exhaustion risk: CPU {peak_cpu:.2f}%",
                    resolution_status="mitigated" if day < 7 else "open",
                    resolution_details="Auto-scaled" if day < 7 else None
                ))

            peak_memory = summary.get('peak_memory_percent', 0)
            if peak_memory > 80:
                degradations.append(DegradationEvent(
                    day_occurred=day,
                    category="resource",
                    description=f"Memory utilization peaked at {peak_memory:.2f}% on Day {day}",
                    severity="critical" if peak_memory > 95 else "high" if peak_memory > 90 else "medium",
                    impact=f"Resource exhaustion risk: memory {peak_memory:.2f}%",
                    resolution_status="mitigated" if day < 7 else "open",
                    resolution_details="Auto-scaled" if day < 7 else None
                ))

        logger.info(f"Extracted {len(degradations)} degradation events")
        return degradations


class DriftAnalyzer:
    """
    Analyzes unexpected drift patterns.
    """

    def extract_unexpected_drifts(
        self,
        regression_data: Dict[str, Any],
        anomaly_events: List[Dict[str, Any]],
        ga_kpi: Dict[str, Any],
        seven_day_summaries: List[Dict[str, Any]]
    ) -> List[UnexpectedDrift]:
        """
        Extract unexpected drift events.

        Unexpected drifts are:
        - Metrics that drifted >10% but weren't regressions
        - Volatile metrics with high variance
        - Trending metrics (gradual increase/decrease)

        Args:
            regression_data: Regression analysis summary
            anomaly_events: Anomaly events
            ga_kpi: GA Day KPI summary
            seven_day_summaries: 7-day daily summaries

        Returns:
            List of UnexpectedDrift objects
        """
        drifts = []

        if not seven_day_summaries or not ga_kpi:
            logger.warning("Insufficient data for drift analysis")
            return drifts

        # Get regression metric names for exclusion
        regression_metrics = set()
        if regression_data and 'regressions' in regression_data:
            regression_metrics = {r.get('metric_name', '') for r in regression_data['regressions']}

        # Calculate 7-day average for comparison
        metrics_to_check = [
            'avg_cpu_percent', 'avg_memory_percent', 'avg_redis_hit_rate',
            'avg_db_p95_latency_ms', 'cluster_cpu_utilization', 'cluster_memory_utilization'
        ]

        for metric in metrics_to_check:
            if metric in regression_metrics:
                continue  # Skip if already flagged as regression

            ga_value = ga_kpi.get(metric, 0)
            if ga_value == 0:
                continue

            # Calculate 7-day average
            values = [s.get(metric, 0) for s in seven_day_summaries]
            if not values:
                continue

            avg_value = sum(values) / len(values)
            drift_percent = ((avg_value - ga_value) / ga_value) * 100

            # Check for significant drift (>10% but not regression)
            if abs(drift_percent) > 10 and abs(drift_percent) < 30:  # Not regression-level
                # Determine trend
                if len(values) >= 3:
                    first_half_avg = sum(values[:len(values)//2]) / (len(values)//2)
                    second_half_avg = sum(values[len(values)//2:]) / (len(values) - len(values)//2)
                    if second_half_avg > first_half_avg * 1.1:
                        trend = "increasing"
                    elif second_half_avg < first_half_avg * 0.9:
                        trend = "decreasing"
                    else:
                        trend = "volatile"
                else:
                    trend = "increasing" if drift_percent > 0 else "decreasing"

                # Suggest potential causes
                causes = self._suggest_drift_causes(metric, drift_percent, trend, anomaly_events)

                drifts.append(UnexpectedDrift(
                    metric_name=metric,
                    baseline_value=ga_value,
                    final_value=avg_value,
                    drift_percent=drift_percent,
                    trend=trend,
                    potential_causes=causes,
                    investigation_needed=abs(drift_percent) > 15
                ))

        logger.info(f"Extracted {len(drifts)} unexpected drift events")
        return drifts

    def _suggest_drift_causes(
        self,
        metric: str,
        drift_percent: float,
        trend: str,
        anomaly_events: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Suggest potential causes for drift.

        Args:
            metric: Metric name
            drift_percent: Drift percentage
            trend: Trend direction
            anomaly_events: Anomaly events

        Returns:
            List of potential causes
        """
        causes = []

        # Check for correlated anomalies
        related_anomalies = [a for a in anomaly_events if metric in a.get('metric_name', '')]
        if related_anomalies:
            causes.append(f"Correlated with {len(related_anomalies)} anomaly events")

        # Metric-specific causes
        if 'cpu' in metric.lower():
            if drift_percent > 0:
                causes.extend([
                    "Increased workload or request volume",
                    "Less efficient code paths being executed",
                    "Background tasks or batch jobs"
                ])
            else:
                causes.extend([
                    "Workload optimization improvements",
                    "Reduced request volume",
                    "More efficient algorithms deployed"
                ])
        elif 'memory' in metric.lower():
            if drift_percent > 0:
                causes.extend([
                    "Memory leak or cache growth",
                    "Increased data volume in memory",
                    "Object pooling changes"
                ])
            else:
                causes.extend([
                    "Memory optimization or garbage collection tuning",
                    "Reduced cache size",
                    "More efficient data structures"
                ])
        elif 'redis' in metric.lower():
            if drift_percent > 0:
                causes.extend([
                    "Cache warming effectiveness improved",
                    "More predictable access patterns"
                ])
            else:
                causes.extend([
                    "Cache key churn or eviction",
                    "Access pattern changes",
                    "Cold cache scenarios"
                ])
        elif 'latency' in metric.lower():
            if drift_percent > 0:
                causes.extend([
                    "Database query performance degradation",
                    "Network latency increase",
                    "Resource contention"
                ])
            else:
                causes.extend([
                    "Query optimization improvements",
                    "Better caching strategies",
                    "Reduced lock contention"
                ])

        # Trend-specific causes
        if trend == "volatile":
            causes.append("High variance suggests external factors or traffic patterns")

        return causes[:3]  # Return top 3 causes


class CostAnalyzer:
    """
    Analyzes cost trends and optimization opportunities.
    """

    def analyze_costs(
        self,
        ga_kpi: Dict[str, Any],
        seven_day_summaries: List[Dict[str, Any]]
    ) -> CostAnalysis:
        """
        Analyze costs for GA Day + 7-day period.

        Args:
            ga_kpi: GA Day KPI summary
            seven_day_summaries: 7-day daily summaries

        Returns:
            CostAnalysis object
        """
        # Extract GA Day cost (if available)
        ga_day_cost = ga_kpi.get('estimated_cost_per_hour', 10.0)  # Default estimate

        # Calculate 7-day costs
        daily_costs = []
        for summary in seven_day_summaries:
            cost = summary.get('estimated_cost_per_hour', ga_day_cost)
            daily_costs.append(cost)

        seven_day_total = sum(daily_costs) * 24  # Convert hourly to daily estimate
        daily_average = sum(daily_costs) / len(daily_costs) if daily_costs else ga_day_cost

        # Determine cost trend
        if len(daily_costs) >= 3:
            first_half_avg = sum(daily_costs[:len(daily_costs)//2]) / (len(daily_costs)//2)
            second_half_avg = sum(daily_costs[len(daily_costs)//2:]) / (len(daily_costs) - len(daily_costs)//2)

            if second_half_avg > first_half_avg * 1.1:
                cost_trend = "increasing"
            elif second_half_avg < first_half_avg * 0.9:
                cost_trend = "decreasing"
            else:
                cost_trend = "stable"
        else:
            cost_trend = "stable"

        # Generate cost breakdown (estimated)
        cost_breakdown = {
            "compute": daily_average * 0.6,  # 60% compute
            "storage": daily_average * 0.2,  # 20% storage
            "network": daily_average * 0.15, # 15% network
            "other": daily_average * 0.05    # 5% other
        }

        # Generate optimization recommendations
        recommendations = []

        if cost_trend == "increasing":
            recommendations.append("Investigate increasing cost trend - review resource utilization patterns")
            recommendations.append("Consider implementing auto-scaling policies to reduce idle capacity")

        # Check if CPU/memory utilization is low
        avg_cpu = sum(s.get('avg_cpu_percent', 0) for s in seven_day_summaries) / len(seven_day_summaries)
        if avg_cpu < 40:
            recommendations.append(f"CPU utilization is low ({avg_cpu:.1f}%) - consider right-sizing instances")

        avg_memory = sum(s.get('avg_memory_percent', 0) for s in seven_day_summaries) / len(seven_day_summaries)
        if avg_memory < 50:
            recommendations.append(f"Memory utilization is low ({avg_memory:.1f}%) - consider smaller instance types")

        if daily_average > ga_day_cost * 1.2:
            recommendations.append("Cost increased >20% vs GA Day - review recent changes and workload patterns")

        if not recommendations:
            recommendations.append("Cost efficiency is within acceptable range - continue monitoring")

        return CostAnalysis(
            ga_day_cost=ga_day_cost,
            seven_day_total_cost=seven_day_total,
            daily_average_cost=daily_average,
            cost_trend=cost_trend,
            cost_breakdown=cost_breakdown,
            cost_optimization_recommendations=recommendations
        )


class SLOAnalyzer:
    """
    Analyzes SLO burn-down over the 7-day period.
    """

    # SLO definitions
    SLOS = [
        {"name": "Availability", "target": 99.9, "budget": 0.1},
        {"name": "P99 Latency", "target": 500.0, "budget": None},  # Hard limit
        {"name": "Error Rate", "target": 0.1, "budget": 0.9},
    ]

    def analyze_slo_burn_down(
        self,
        ga_kpi: Dict[str, Any],
        seven_day_summaries: List[Dict[str, Any]]
    ) -> List[SLOBurnDown]:
        """
        Analyze SLO burn-down for each SLO.

        Args:
            ga_kpi: GA Day KPI summary
            seven_day_summaries: 7-day daily summaries

        Returns:
            List of SLOBurnDown objects
        """
        slo_burn_downs = []

        for slo_def in self.SLOS:
            slo_name = slo_def['name']
            target = slo_def['target']
            budget = slo_def['budget']

            compliance_by_day = []

            # Add GA Day
            ga_value = self._get_slo_value(ga_kpi, slo_name)
            if ga_value is not None:
                compliance_by_day.append({
                    'day': 0,
                    'value': ga_value,
                    'compliant': self._check_compliance(ga_value, target, slo_name),
                    'budget_consumed': 0.0
                })

            # Add 7-day data
            for summary in seven_day_summaries:
                day = summary.get('day_number', 0)
                value = self._get_slo_value(summary, slo_name)

                if value is not None:
                    compliant = self._check_compliance(value, target, slo_name)
                    compliance_by_day.append({
                        'day': day,
                        'value': value,
                        'compliant': compliant,
                        'budget_consumed': 0.0  # Will calculate below
                    })

            # Calculate budget consumed
            if budget is not None:
                total_violations = sum(1 for c in compliance_by_day if not c['compliant'])
                total_days = len(compliance_by_day)

                if total_days > 0:
                    violation_rate = (total_violations / total_days) * 100
                    budget_consumed = min((violation_rate / budget) * 100, 100.0)
                else:
                    budget_consumed = 0.0

                # Calculate per-day budget consumed
                for i, compliance in enumerate(compliance_by_day):
                    violations_so_far = sum(1 for c in compliance_by_day[:i+1] if not c['compliant'])
                    days_so_far = i + 1
                    if days_so_far > 0:
                        compliance['budget_consumed'] = min(
                            ((violations_so_far / days_so_far) / (budget / 100)) * 100,
                            100.0
                        )
            else:
                budget_consumed = 0.0  # Hard limit, no error budget

            # Project days to exhaustion
            days_to_exhaustion = None
            if budget is not None and budget_consumed > 0:
                # Simple linear projection
                days_elapsed = len(compliance_by_day)
                if budget_consumed < 100:
                    burn_rate = budget_consumed / days_elapsed
                    remaining_budget = 100 - budget_consumed
                    days_to_exhaustion = remaining_budget / burn_rate if burn_rate > 0 else None

            slo_burn_downs.append(SLOBurnDown(
                slo_name=slo_name,
                target=target,
                budget=budget if budget is not None else 0.0,
                budget_consumed_percent=budget_consumed,
                days_to_exhaustion=days_to_exhaustion,
                compliance_by_day=compliance_by_day
            ))

        logger.info(f"Analyzed {len(slo_burn_downs)} SLO burn-downs")
        return slo_burn_downs

    def _get_slo_value(self, data: Dict[str, Any], slo_name: str) -> Optional[float]:
        """Get SLO value from data dict."""
        if slo_name == "Availability":
            return data.get('overall_availability')
        elif slo_name == "P99 Latency":
            return data.get('avg_p99_latency_ms')
        elif slo_name == "Error Rate":
            return data.get('overall_error_rate')
        return None

    def _check_compliance(self, value: float, target: float, slo_name: str) -> bool:
        """Check if value meets SLO target."""
        if slo_name == "Availability":
            return value >= target
        elif slo_name == "P99 Latency":
            return value <= target
        elif slo_name == "Error Rate":
            return value <= target
        return True


class RecommendationGenerator:
    """
    Generates recommendations for v1.0.2.
    """

    def generate_recommendations(
        self,
        successes: List[SuccessMetric],
        degradations: List[DegradationEvent],
        drifts: List[UnexpectedDrift],
        cost_analysis: CostAnalysis,
        slo_burn_downs: List[SLOBurnDown]
    ) -> List[str]:
        """
        Generate recommendations for v1.0.2 based on retrospective data.

        Args:
            successes: Success metrics
            degradations: Degradation events
            drifts: Unexpected drifts
            cost_analysis: Cost analysis
            slo_burn_downs: SLO burn-down data

        Returns:
            List of recommendation strings
        """
        recommendations = []

        # Critical degradations (P0)
        critical_degradations = [d for d in degradations if d.severity == "critical"]
        if critical_degradations:
            recommendations.append(
                f"[P0] Address {len(critical_degradations)} critical degradation(s) immediately - "
                f"these pose significant risk to system stability"
            )
            for deg in critical_degradations[:3]:  # Top 3
                recommendations.append(f"  - {deg.description}")

        # High severity degradations (P1)
        high_degradations = [d for d in degradations if d.severity == "high"]
        if high_degradations:
            recommendations.append(
                f"[P1] Investigate and resolve {len(high_degradations)} high-severity degradation(s) "
                f"within next sprint"
            )

        # SLO violations
        violated_slos = [slo for slo in slo_burn_downs if slo.budget_consumed_percent > 50]
        if violated_slos:
            recommendations.append(
                f"[P0] SLO budget consumption critical for {len(violated_slos)} SLO(s) - "
                f"implement immediate corrective actions"
            )
            for slo in violated_slos:
                if slo.days_to_exhaustion and slo.days_to_exhaustion < 14:
                    recommendations.append(
                        f"  - {slo.slo_name}: {slo.budget_consumed_percent:.1f}% consumed, "
                        f"~{slo.days_to_exhaustion:.1f} days to exhaustion"
                    )

        # Unexpected drifts requiring investigation
        investigation_drifts = [d for d in drifts if d.investigation_needed]
        if investigation_drifts:
            recommendations.append(
                f"[P2] Investigate {len(investigation_drifts)} unexpected metric drift(s) - "
                f"may indicate emerging issues"
            )
            for drift in investigation_drifts[:2]:  # Top 2
                recommendations.append(
                    f"  - {drift.metric_name}: {drift.drift_percent:+.1f}% drift ({drift.trend})"
                )

        # Cost optimization
        if cost_analysis.cost_trend == "increasing":
            recommendations.append(
                "[P2] Cost trend is increasing - implement cost optimization measures from analysis"
            )

        if cost_analysis.cost_optimization_recommendations:
            recommendations.append("[P3] Cost optimization opportunities:")
            for rec in cost_analysis.cost_optimization_recommendations[:2]:
                recommendations.append(f"  - {rec}")

        # Performance improvements based on successes
        performance_successes = [s for s in successes if s.category == "performance"]
        if len(performance_successes) >= 3:
            recommendations.append(
                "[P3] Performance metrics are strong - document best practices for knowledge sharing"
            )

        # Stability recommendations
        if not critical_degradations and not high_degradations:
            recommendations.append(
                "[P3] System stability is good - focus on proactive improvements and optimization"
            )
        else:
            recommendations.append(
                "[P1] Implement enhanced monitoring for degraded metrics to detect future issues earlier"
            )

        # Resource efficiency
        medium_degradations = [d for d in degradations if d.severity == "medium" and d.category == "resource"]
        if medium_degradations:
            recommendations.append(
                f"[P2] {len(medium_degradations)} resource efficiency issues detected - "
                f"review autoscaling policies"
            )

        # Limit to top 10 recommendations
        return recommendations[:10]

    def generate_process_improvements(
        self,
        degradations: List[DegradationEvent],
        anomaly_events: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Generate process improvement recommendations.

        Args:
            degradations: Degradation events
            anomaly_events: Anomaly events

        Returns:
            List of process improvement strings
        """
        improvements = []

        # Monitoring enhancements
        if len(anomaly_events) > 10:
            improvements.append(
                "Enhance anomaly detection thresholds - high volume of anomalies may indicate "
                "threshold tuning needed or genuine system instability"
            )

        unresolved_degradations = [d for d in degradations if d.resolution_status == "open"]
        if unresolved_degradations:
            improvements.append(
                f"{len(unresolved_degradations)} degradations remain unresolved - "
                f"implement incident response SLAs and tracking"
            )

        # Testing improvements
        performance_degradations = [d for d in degradations if d.category == "performance"]
        if len(performance_degradations) > 3:
            improvements.append(
                "Expand performance testing in pre-production - multiple performance degradations "
                "suggest gaps in load testing coverage"
            )

        # Deployment process
        day_0_issues = [d for d in degradations if d.day_occurred == 0]
        if day_0_issues:
            improvements.append(
                f"{len(day_0_issues)} issues detected on GA Day - strengthen pre-launch "
                f"validation and go/no-go criteria"
            )

        # Observability
        improvements.append(
            "Continue 7-day post-GA monitoring practice - valuable for detecting "
            "delayed issues and trends"
        )

        # Documentation
        improvements.append(
            "Document runbook procedures for degradation patterns observed during 7-day period"
        )

        return improvements[:5]  # Top 5


class RetrospectiveGenerator:
    """
    Main retrospective generator.
    """

    def __init__(
        self,
        ga_data_dir: str,
        seven_day_data_dir: str,
        regression_file: str,
        anomaly_file: str = "anomaly_events.json",
        output_file: str = "docs/final/GA_7DAY_RETROSPECTIVE.md",
        # Enterprise features
        compliance_enforcer: Optional[ComplianceEnforcer] = None,
        encryptor: Optional[AESEncryption] = None,
        signer: Optional[ReportSigner] = None,
    ):
        """
        Initialize retrospective generator.

        Args:
            ga_data_dir: Path to GA Day data directory
            seven_day_data_dir: Path to 7-day stability data directory
            regression_file: Path to regression analysis JSON
            anomaly_file: Path to anomaly events JSON
            output_file: Output file path
            compliance_enforcer: Optional compliance enforcer (enterprise)
            encryptor: Optional AES encryptor (enterprise)
            signer: Optional RSA signer (enterprise)
        """
        self.ga_data_dir = ga_data_dir
        self.seven_day_data_dir = seven_day_data_dir
        self.regression_file = regression_file
        self.anomaly_file = anomaly_file
        self.output_file = Path(output_file)

        self.loader = DataLoader(ga_data_dir, seven_day_data_dir, regression_file, anomaly_file)

        # Enterprise features (Phase 14.6)
        self.compliance_enforcer = compliance_enforcer
        self.encryptor = encryptor
        self.signer = signer

        logger.info("RetrospectiveGenerator initialized")

    def generate(self) -> RetrospectiveData:
        """
        Generate complete retrospective data.

        Returns:
            RetrospectiveData object
        """
        logger.info("Starting retrospective generation...")

        # 1. Load all data sources
        logger.info("Loading data sources...")
        ga_kpi = self.loader.load_ga_kpi_summary()
        seven_day_summaries = self.loader.load_seven_day_summaries()
        regression_data = self.loader.load_regression_analysis()
        anomaly_events = self.loader.load_anomaly_events()

        # 2. Extract successes
        logger.info("Extracting successes...")
        success_analyzer = SuccessAnalyzer()
        successes = success_analyzer.extract_successes(ga_kpi, seven_day_summaries)

        # 3. Extract degradations
        logger.info("Extracting degradations...")
        degradation_analyzer = DegradationAnalyzer()
        degradations = degradation_analyzer.extract_degradations(regression_data, seven_day_summaries)

        # 4. Extract unexpected drifts
        logger.info("Extracting unexpected drifts...")
        drift_analyzer = DriftAnalyzer()
        drifts = drift_analyzer.extract_unexpected_drifts(
            regression_data, anomaly_events, ga_kpi, seven_day_summaries
        )

        # 5. Analyze costs
        logger.info("Analyzing costs...")
        cost_analyzer = CostAnalyzer()
        cost_analysis = cost_analyzer.analyze_costs(ga_kpi, seven_day_summaries)

        # 6. Analyze SLO burn-down
        logger.info("Analyzing SLO burn-down...")
        slo_analyzer = SLOAnalyzer()
        slo_burn_downs = slo_analyzer.analyze_slo_burn_down(ga_kpi, seven_day_summaries)

        # 7. Generate recommendations
        logger.info("Generating recommendations...")
        rec_generator = RecommendationGenerator()
        recommendations = rec_generator.generate_recommendations(
            successes, degradations, drifts, cost_analysis, slo_burn_downs
        )

        # 8. Generate process improvements
        logger.info("Generating process improvements...")
        process_improvements = rec_generator.generate_process_improvements(degradations, anomaly_events)

        # 9. Generate action items
        logger.info("Generating action items...")
        action_items = self._generate_action_items(degradations, drifts, recommendations)

        # Create retrospective data
        retro_data = RetrospectiveData(
            generation_timestamp=datetime.now(timezone.utc).isoformat(),
            ga_day_timestamp=ga_kpi.get('timestamp', ''),
            seven_day_end_timestamp=seven_day_summaries[-1].get('timestamp', '') if seven_day_summaries else '',
            successes=successes,
            degradations=degradations,
            unexpected_drifts=drifts,
            cost_analysis=cost_analysis,
            slo_burn_downs=slo_burn_downs,
            recommendations_v1_0_2=recommendations,
            process_improvements=process_improvements,
            action_items=action_items
        )

        logger.info("Retrospective generation complete")
        return retro_data

    def _generate_action_items(
        self,
        degradations: List[DegradationEvent],
        drifts: List[UnexpectedDrift],
        recommendations: List[str]
    ) -> List[Dict[str, str]]:
        """Generate action items checklist."""
        action_items = []

        # P0 items from recommendations
        p0_recs = [r for r in recommendations if '[P0]' in r]
        for rec in p0_recs:
            action_items.append({
                'priority': 'P0',
                'description': rec.replace('[P0] ', ''),
                'status': 'open'
            })

        # P1 items from recommendations
        p1_recs = [r for r in recommendations if '[P1]' in r]
        for rec in p1_recs:
            action_items.append({
                'priority': 'P1',
                'description': rec.replace('[P1] ', ''),
                'status': 'open'
            })

        # P2 items from recommendations
        p2_recs = [r for r in recommendations if '[P2]' in r]
        for rec in p2_recs[:3]:  # Limit P2 items
            action_items.append({
                'priority': 'P2',
                'description': rec.replace('[P2] ', ''),
                'status': 'open'
            })

        return action_items

    def save_markdown(self, data: RetrospectiveData) -> str:
        """
        Save retrospective as Markdown.

        Args:
            data: Retrospective data

        Returns:
            Path to saved Markdown file
        """
        markdown_content = self.format_markdown(data)

        # Ensure output directory exists
        self.output_file.parent.mkdir(parents=True, exist_ok=True)

        # Write to file with UTF-8 encoding (for emoji support on Windows)
        with open(self.output_file, 'w', encoding='utf-8') as f:
            f.write(markdown_content)

        # Enterprise: Encrypt if enabled
        if self.encryptor:
            encrypted_file = self.output_file.with_suffix(".md.enc")
            self.encryptor.encrypt_file(self.output_file, encrypted_file)
            logger.info(f"Encrypted retrospective: {encrypted_file}")

        # Enterprise: Sign if enabled
        if self.signer:
            signature = self.signer.sign_file(self.output_file)
            sig_file = self.output_file.with_suffix(".md.sig")
            with open(sig_file, "w") as f:
                f.write(f"RSA-PSS-SHA256\n{signature}\n")
            logger.info(f"Signed retrospective: {sig_file}")

        logger.info(f"Saved Markdown retrospective to {self.output_file}")
        return str(self.output_file)

    def save_json(self, data: RetrospectiveData) -> str:
        """
        Save retrospective as JSON (includes SLO burn-down).

        Args:
            data: Retrospective data

        Returns:
            Path to saved JSON file
        """
        json_file = self.output_file.with_suffix('.json')

        # Ensure output directory exists
        json_file.parent.mkdir(parents=True, exist_ok=True)

        # Convert to dict
        data_dict = asdict(data)

        # Write to file
        with open(json_file, 'w') as f:
            json.dump(data_dict, f, indent=2)

        # Enterprise: Encrypt if enabled
        if self.encryptor:
            encrypted_file = json_file.with_suffix(".json.enc")
            self.encryptor.encrypt_file(json_file, encrypted_file)
            logger.info(f"Encrypted JSON retrospective: {encrypted_file}")

        # Enterprise: Sign if enabled
        if self.signer:
            signature = self.signer.sign_file(json_file)
            sig_file = json_file.with_suffix(".json.sig")
            with open(sig_file, "w") as f:
                f.write(f"RSA-PSS-SHA256\n{signature}\n")
            logger.info(f"Signed JSON retrospective: {sig_file}")

        logger.info(f"Saved JSON retrospective to {json_file}")
        return str(json_file)

    def format_markdown(self, data: RetrospectiveData) -> str:
        """
        Format retrospective data as Markdown string.

        Args:
            data: Retrospective data

        Returns:
            Formatted Markdown string
        """
        md = []

        # Header
        md.append("# T.A.R.S. v1.0.1 - GA 7-Day Retrospective\n")
        md.append(f"**Generated:** {data.generation_timestamp}\n")
        md.append(f"**GA Day:** {data.ga_day_timestamp}\n")
        md.append(f"**7-Day Period End:** {data.seven_day_end_timestamp}\n")
        md.append("\n---\n")

        # Executive Summary
        md.append("## Executive Summary\n")
        md.append(f"- **Total Successes:** {len(data.successes)}\n")
        md.append(f"- **Total Degradations:** {len(data.degradations)}\n")

        critical_count = sum(1 for d in data.degradations if d.severity == "critical")
        high_count = sum(1 for d in data.degradations if d.severity == "high")
        md.append(f"  - Critical: {critical_count}, High: {high_count}\n")

        md.append(f"- **Unexpected Drifts:** {len(data.unexpected_drifts)}\n")
        md.append(f"- **Cost Trend:** {data.cost_analysis.cost_trend if data.cost_analysis else 'N/A'}\n")

        # Overall status
        if critical_count > 0:
            status = "CRITICAL - Immediate action required"
        elif high_count > 2:
            status = "NEEDS ATTENTION - High-priority issues detected"
        elif len(data.degradations) > 0:
            status = "STABLE WITH MINOR ISSUES"
        else:
            status = "EXCELLENT - All metrics within targets"
        md.append(f"- **Overall Status:** {status}\n")
        md.append("\n---\n")

        # What Went Well
        md.append("## What Went Well ✅\n")
        if data.successes:
            md.append("| Category | Metric | Target | Actual | Achievement |\n")
            md.append("|----------|--------|--------|--------|-------------|\n")
            for success in data.successes:
                md.append(
                    f"| {success.category} | {success.metric_name} | "
                    f"{success.target_value} | {success.actual_value} | "
                    f"{success.achievement_percent:.1f}% |\n"
                )
            md.append("\n")
            md.append("### Success Highlights\n")
            for success in data.successes[:5]:  # Top 5
                md.append(f"- {success.description}\n")
        else:
            md.append("No significant successes identified during the 7-day period.\n")
        md.append("\n---\n")

        # What Could Be Improved
        md.append("## What Could Be Improved ⚠️\n")
        if data.degradations:
            # Group by severity
            critical_degs = [d for d in data.degradations if d.severity == "critical"]
            high_degs = [d for d in data.degradations if d.severity == "high"]
            medium_degs = [d for d in data.degradations if d.severity == "medium"]

            if critical_degs:
                md.append("### 🚨 Critical Degradations\n")
                for deg in critical_degs:
                    md.append(f"- **Day {deg.day_occurred}**: {deg.description}\n")
                    md.append(f"  - **Impact:** {deg.impact}\n")
                    md.append(f"  - **Status:** {deg.resolution_status}\n")
                    if deg.resolution_details:
                        md.append(f"  - **Resolution:** {deg.resolution_details}\n")
                md.append("\n")

            if high_degs:
                md.append("### ❌ High Severity Degradations\n")
                for deg in high_degs[:5]:  # Top 5
                    md.append(f"- **Day {deg.day_occurred}**: {deg.description}\n")
                    md.append(f"  - **Impact:** {deg.impact}\n")
                md.append("\n")

            if medium_degs:
                md.append(f"### ⚠️ Medium Severity Issues ({len(medium_degs)} total)\n")
                for deg in medium_degs[:3]:  # Top 3
                    md.append(f"- **Day {deg.day_occurred}**: {deg.description}\n")
                md.append("\n")
        else:
            md.append("No significant degradations detected during the 7-day period.\n")
        md.append("\n---\n")

        # Unexpected Drifts
        md.append("## Unexpected Drifts 📊\n")
        if data.unexpected_drifts:
            md.append("Metrics that drifted significantly but did not trigger regression thresholds:\n\n")
            md.append("| Metric | Baseline | Final | Drift % | Trend | Investigation |\n")
            md.append("|--------|----------|-------|---------|-------|---------------|\n")
            for drift in data.unexpected_drifts:
                investigation = "⚠️ Yes" if drift.investigation_needed else "ℹ️ Monitor"
                md.append(
                    f"| {drift.metric_name} | {drift.baseline_value:.2f} | "
                    f"{drift.final_value:.2f} | {drift.drift_percent:+.1f}% | "
                    f"{drift.trend} | {investigation} |\n"
                )
            md.append("\n")
            md.append("### Potential Causes\n")
            for drift in data.unexpected_drifts:
                if drift.potential_causes:
                    md.append(f"**{drift.metric_name}:**\n")
                    for cause in drift.potential_causes:
                        md.append(f"- {cause}\n")
                    md.append("\n")
        else:
            md.append("No unexpected metric drifts detected.\n")
        md.append("\n---\n")

        # Cost Analysis
        md.append("## Cost Analysis 💰\n")
        if data.cost_analysis:
            ca = data.cost_analysis
            md.append(f"- **GA Day Cost:** ${ca.ga_day_cost:.2f}/hour\n")
            md.append(f"- **7-Day Total Cost:** ${ca.seven_day_total_cost:.2f}\n")
            md.append(f"- **Daily Average Cost:** ${ca.daily_average_cost:.2f}/hour\n")
            md.append(f"- **Cost Trend:** {ca.cost_trend}\n")
            md.append("\n### Cost Breakdown\n")
            for resource, cost in ca.cost_breakdown.items():
                md.append(f"- **{resource.capitalize()}:** ${cost:.2f}/hour\n")
            md.append("\n### Optimization Recommendations\n")
            for rec in ca.cost_optimization_recommendations:
                md.append(f"- {rec}\n")
        else:
            md.append("Cost analysis data not available.\n")
        md.append("\n---\n")

        # SLO Burn-Down Summary
        md.append("## SLO Compliance Summary\n")
        md.append("*Full SLO burn-down data available in JSON report.*\n\n")
        if data.slo_burn_downs:
            md.append("| SLO | Target | Budget Consumed | Status |\n")
            md.append("|-----|--------|----------------|--------|\n")
            for slo in data.slo_burn_downs:
                if slo.budget_consumed_percent > 75:
                    status = "🚨 Critical"
                elif slo.budget_consumed_percent > 50:
                    status = "⚠️ At Risk"
                elif slo.budget_consumed_percent > 25:
                    status = "⚡ Moderate"
                else:
                    status = "✅ Healthy"
                md.append(
                    f"| {slo.slo_name} | {slo.target} | "
                    f"{slo.budget_consumed_percent:.1f}% | {status} |\n"
                )
            md.append("\n")
        md.append("\n---\n")

        # Recommendations for v1.0.2
        md.append("## Recommendations for v1.0.2 🚀\n")
        if data.recommendations_v1_0_2:
            for i, rec in enumerate(data.recommendations_v1_0_2, 1):
                md.append(f"{i}. {rec}\n")
        else:
            md.append("No recommendations generated.\n")
        md.append("\n---\n")

        # Process Improvements
        md.append("## Process Improvements 🔧\n")
        if data.process_improvements:
            for i, improvement in enumerate(data.process_improvements, 1):
                md.append(f"{i}. {improvement}\n")
        else:
            md.append("No process improvements identified.\n")
        md.append("\n---\n")

        # Action Items
        md.append("## Action Items\n")
        if data.action_items:
            p0_items = [a for a in data.action_items if a['priority'] == 'P0']
            p1_items = [a for a in data.action_items if a['priority'] == 'P1']
            p2_items = [a for a in data.action_items if a['priority'] == 'P2']

            if p0_items:
                md.append("### 🚨 P0 (Critical - Immediate Action)\n")
                for item in p0_items:
                    md.append(f"- [ ] {item['description']}\n")
                md.append("\n")

            if p1_items:
                md.append("### ❌ P1 (High Priority - Within 24-48h)\n")
                for item in p1_items:
                    md.append(f"- [ ] {item['description']}\n")
                md.append("\n")

            if p2_items:
                md.append("### ⚠️ P2 (Medium Priority - Within Sprint)\n")
                for item in p2_items:
                    md.append(f"- [ ] {item['description']}\n")
                md.append("\n")
        else:
            md.append("No action items generated.\n")
        md.append("\n---\n")

        # Footer
        md.append("## Appendix\n")
        md.append("- **Full SLO Burn-Down Data:** See `GA_7DAY_RETROSPECTIVE.json`\n")
        md.append("- **Regression Analysis:** See `regression_summary.json`\n")
        md.append("- **Anomaly Events:** See `anomaly_events.json`\n")
        md.append("\n---\n")
        md.append(f"\n**Generated by T.A.R.S. Retrospective Generator**\n")
        md.append(f"**Timestamp:** {data.generation_timestamp}\n")

        return "".join(md)


def main():
    """
    CLI entry point.
    """
    import argparse

    parser = argparse.ArgumentParser(description="GA 7-Day Retrospective Generator")
    parser.add_argument(
        "--ga-data",
        help="Path to GA Day data directory (default: ga_kpis/)"
    )
    parser.add_argument(
        "--7day-data",
        help="Path to 7-day stability data directory (default: stability/)"
    )
    parser.add_argument(
        "--regression",
        help="Path to regression analysis JSON (default: regression_summary.json)"
    )
    parser.add_argument(
        "--anomalies",
        default="anomaly_events.json",
        help="Path to anomaly events JSON (default: anomaly_events.json)"
    )
    parser.add_argument(
        "--output",
        default="docs/final/GA_7DAY_RETROSPECTIVE.md",
        help="Output file path (default: docs/final/GA_7DAY_RETROSPECTIVE.md)"
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Auto-detect all data files from standard locations"
    )

    # Enterprise configuration arguments (Phase 14.6)
    if ENTERPRISE_AVAILABLE:
        parser.add_argument("--profile", type=str, default="local", help="Enterprise config profile (local, dev, staging, prod)")
        parser.add_argument("--config", type=str, help="Path to enterprise config file")
        parser.add_argument("--encrypt", action="store_true", help="Encrypt output files (requires AES key)")
        parser.add_argument("--sign", action="store_true", help="Sign output files (requires RSA key)")
        parser.add_argument("--no-compliance", action="store_true", help="Disable compliance enforcement")

    args = parser.parse_args()

    # Auto-detect files if --auto flag set
    if args.auto:
        ga_data_dir = "ga_kpis"
        seven_day_data_dir = "stability"
        regression_file = "regression_summary.json"
        anomaly_file = "anomaly_events.json"
        output_file = "docs/final/GA_7DAY_RETROSPECTIVE.md"
        logger.info("Auto-detecting data files from standard locations")
    else:
        ga_data_dir = args.ga_data or "ga_kpis"
        seven_day_data_dir = getattr(args, '7day_data', None) or "stability"
        regression_file = args.regression or "regression_summary.json"
        anomaly_file = args.anomalies
        output_file = args.output

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
            if output_file == "docs/final/GA_7DAY_RETROSPECTIVE.md" and config.observability.output_dir != "output":
                output_file = str(Path(config.observability.output_dir) / "GA_7DAY_RETROSPECTIVE.md")

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
        # Instantiate RetrospectiveGenerator
        logger.info("Initializing Retrospective Generator...")
        generator = RetrospectiveGenerator(
            ga_data_dir=ga_data_dir,
            seven_day_data_dir=seven_day_data_dir,
            regression_file=regression_file,
            anomaly_file=anomaly_file,
            output_file=output_file,
            compliance_enforcer=compliance_enforcer,
            encryptor=encryptor,
            signer=signer,
        )

        # Generate retrospective
        logger.info("Generating retrospective data...")
        retro_data = generator.generate()

        # Save Markdown and JSON files
        logger.info("Saving retrospective reports...")
        md_path = generator.save_markdown(retro_data)
        json_path = generator.save_json(retro_data)

        # Print summary to console
        print("\n" + "=" * 60)
        print("RETROSPECTIVE GENERATION COMPLETE")
        print("=" * 60)
        print(f"\nSuccesses: {len(retro_data.successes)}")
        print(f"Degradations: {len(retro_data.degradations)}")
        critical_count = sum(1 for d in retro_data.degradations if d.severity == "critical")
        high_count = sum(1 for d in retro_data.degradations if d.severity == "high")
        print(f"  - Critical: {critical_count}")
        print(f"  - High: {high_count}")
        print(f"Unexpected Drifts: {len(retro_data.unexpected_drifts)}")
        print(f"Cost Trend: {retro_data.cost_analysis.cost_trend if retro_data.cost_analysis else 'N/A'}")
        print(f"\nRecommendations: {len(retro_data.recommendations_v1_0_2)}")
        print(f"Process Improvements: {len(retro_data.process_improvements)}")
        print(f"Action Items: {len(retro_data.action_items)}")
        p0_count = sum(1 for a in retro_data.action_items if a['priority'] == 'P0')
        p1_count = sum(1 for a in retro_data.action_items if a['priority'] == 'P1')
        print(f"  - P0: {p0_count}, P1: {p1_count}")
        print(f"\nReports saved:")
        print(f"  - Markdown: {md_path}")
        print(f"  - JSON: {json_path}")
        print("=" * 60)

        logger.info("Retrospective generation complete")
        return 0

    except FileNotFoundError as e:
        logger.error(f"Required data file not found: {e}")
        print(f"\nERROR: Required data file not found: {e}")
        print("Please ensure all data files exist or use --auto for auto-detection.")
        return 1
    except Exception as e:
        logger.error(f"Error generating retrospective: {e}", exc_info=True)
        print(f"\nERROR: Failed to generate retrospective: {e}")
        return 1


if __name__ == "__main__":
    main()
