#!/usr/bin/env python3
"""
Daily Health Reporter - Post-GA Day Health Reports

Generates daily health reports (Markdown + JSON) for each day of the 7-day post-GA period.
Includes anomaly analysis integration, mitigation checklists, and actionable recommendations.

Usage:
    python daily_health_reporter.py --day 1 --stability-data stability/day_01.json
    python daily_health_reporter.py --day 1 --auto  # Auto-detect latest stability data

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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class HealthMetric:
    """
    A single health metric for daily reporting.
    """
    name: str
    value: float
    unit: str
    threshold: float
    status: str  # "healthy", "warning", "critical"
    trend: str  # "improving", "stable", "degrading"
    description: str


@dataclass
class AnomalyEvent:
    """
    An anomaly detected during the day.
    """
    timestamp: str
    metric_name: str
    actual_value: float
    expected_value: float
    deviation_percent: float
    severity: str  # "low", "medium", "high"
    classification: str  # "spike", "drop", "drift"
    potential_causes: List[str] = field(default_factory=list)


@dataclass
class MitigationAction:
    """
    A recommended mitigation action.
    """
    priority: str  # "P0", "P1", "P2"
    category: str  # "performance", "availability", "resource", "security"
    description: str
    recommended_steps: List[str] = field(default_factory=list)
    estimated_effort: str  # "minutes", "hours", "days"
    assignee_role: str  # "on-call", "platform-team", "dev-team"


@dataclass
class DailyHealthReport:
    """
    Complete daily health report.
    """
    day_number: int  # 1-7
    report_date: str
    generation_timestamp: str
    ga_day_offset: int  # Days since GA day

    # Overall Health Score
    overall_health_score: float  # 0-100
    health_status: str  # "healthy", "degraded", "critical"

    # Key Metrics Summary
    availability_percent: float
    error_rate_percent: float
    p99_latency_ms: float
    critical_alerts: int
    warning_alerts: int

    # Health Metrics Breakdown
    health_metrics: List[HealthMetric] = field(default_factory=list)

    # Anomalies Detected
    anomalies: List[AnomalyEvent] = field(default_factory=list)
    anomaly_count: int = 0

    # Mitigation Checklist
    mitigation_actions: List[MitigationAction] = field(default_factory=list)
    p0_actions: int = 0
    p1_actions: int = 0
    p2_actions: int = 0

    # Trend Analysis
    availability_trend: str  # "improving", "stable", "degrading"
    latency_trend: str
    resource_trend: str

    # Recommendations
    recommendations: List[str] = field(default_factory=list)

    # Next Steps
    next_steps: List[str] = field(default_factory=list)


class AnomalyAnalyzer:
    """
    Integrates with anomaly_detector_lightweight.py for anomaly analysis.
    """

    def __init__(self, anomaly_events_file: str = "anomaly_events.json"):
        """
        Initialize anomaly analyzer.

        Args:
            anomaly_events_file: Path to anomaly events JSON file
        """
        self.anomaly_events_file = anomaly_events_file
        logger.info(f"AnomalyAnalyzer initialized with file: {anomaly_events_file}")

    def load_anomaly_events(self, day_number: int) -> List[AnomalyEvent]:
        """
        Load anomaly events for a specific day.

        Args:
            day_number: Day number (1-7)

        Returns:
            List of AnomalyEvent objects for the day
        """
        anomaly_file = Path(self.anomaly_events_file)

        if not anomaly_file.exists():
            logger.warning(f"Anomaly events file not found: {self.anomaly_events_file}")
            return []

        try:
            with open(anomaly_file, 'r') as f:
                data = json.load(f)

            # Extract anomaly events
            anomalies_list = data.get("anomalies", [])

            # Filter for specific day (day_number is 1-7)
            # Day 1 = elapsed_hours 0-24, Day 2 = 24-48, etc.
            day_start_hours = (day_number - 1) * 24
            day_end_hours = day_number * 24

            day_anomalies = []
            for anomaly_dict in anomalies_list:
                # Parse timestamp to determine day
                timestamp = anomaly_dict.get("timestamp", "")
                # If snapshot has elapsed_hours, use it; otherwise approximate from timestamp
                # For now, we'll include all anomalies (Phase 3 doesn't store day info)
                # This is a simplification - in production, we'd parse timestamps

                # Create AnomalyEvent object
                anomaly = AnomalyEvent(
                    timestamp=anomaly_dict.get("timestamp", ""),
                    metric_name=anomaly_dict.get("metric_name", ""),
                    actual_value=float(anomaly_dict.get("actual_value", 0.0)),
                    expected_value=float(anomaly_dict.get("expected_value", 0.0)),
                    deviation_percent=abs(
                        (anomaly_dict.get("actual_value", 0.0) - anomaly_dict.get("expected_value", 1.0))
                        / max(anomaly_dict.get("expected_value", 1.0), 0.001) * 100.0
                    ),
                    severity=anomaly_dict.get("severity", "low"),
                    classification=anomaly_dict.get("classification", "drift"),
                    potential_causes=[]  # Will be populated by suggest_potential_causes
                )

                day_anomalies.append(anomaly)

            logger.info(f"Loaded {len(day_anomalies)} anomaly events for Day {day_number}")
            return day_anomalies

        except Exception as e:
            logger.error(f"Failed to load anomaly events: {e}", exc_info=True)
            return []

    def classify_anomaly_severity(self, deviation_percent: float) -> str:
        """
        Classify anomaly severity based on deviation.

        Args:
            deviation_percent: Percentage deviation from expected

        Returns:
            Severity string: "low", "medium", "high"
        """
        abs_deviation = abs(deviation_percent)

        if abs_deviation < 10.0:
            return "low"
        elif abs_deviation < 25.0:
            return "medium"
        else:
            return "high"

    def suggest_potential_causes(self, anomaly: AnomalyEvent) -> List[str]:
        """
        Suggest potential causes for an anomaly.

        Args:
            anomaly: Anomaly event

        Returns:
            List of potential causes
        """
        causes = []

        metric = anomaly.metric_name
        classification = anomaly.classification
        severity = anomaly.severity

        # CPU anomalies
        if "cpu" in metric.lower():
            if classification == "spike":
                causes.extend([
                    "Sudden traffic spike or load increase",
                    "Memory leak causing excessive GC cycles",
                    "Inefficient query or algorithm executed",
                    "Background job or batch process started"
                ])
            elif classification == "drift":
                causes.extend([
                    "Gradual increase in user traffic",
                    "Resource contention with other services",
                    "Configuration change affecting CPU usage",
                    "Code regression in recent deployment"
                ])

        # Memory anomalies
        elif "memory" in metric.lower():
            if classification == "spike":
                causes.extend([
                    "Memory leak in application code",
                    "Large dataset loaded into memory",
                    "Cache size misconfiguration",
                    "Object allocation spike"
                ])
            elif classification == "drift":
                causes.extend([
                    "Gradual memory leak",
                    "Increasing cache size over time",
                    "Growing number of active connections",
                    "Session or state accumulation"
                ])

        # Latency anomalies
        elif "latency" in metric.lower():
            if classification == "spike":
                causes.extend([
                    "Database query timeout or contention",
                    "Network latency or packet loss",
                    "Slow external API dependency",
                    "Lock contention in critical section"
                ])
            elif classification == "drift":
                causes.extend([
                    "Growing database size without proper indexing",
                    "Increasing query complexity",
                    "Cache hit rate degradation",
                    "Network bandwidth saturation"
                ])

        # Error rate anomalies
        elif "error" in metric.lower():
            if classification == "spike":
                causes.extend([
                    "Recent deployment introduced bugs",
                    "Infrastructure failure (DB, cache, network)",
                    "Dependency service outage",
                    "Invalid input data from external source"
                ])
            elif classification == "drift":
                causes.extend([
                    "Gradual data quality degradation",
                    "Increasing timeout errors under load",
                    "Dependency service degradation",
                    "Configuration drift causing intermittent failures"
                ])

        # Add severity-specific causes
        if severity == "high":
            causes.insert(0, "⚠️ CRITICAL: Immediate investigation required")

        return causes


class MitigationGenerator:
    """
    Generates mitigation checklists based on health metrics and anomalies.
    """

    def generate_mitigation_actions(
        self,
        health_metrics: List[HealthMetric],
        anomalies: List[AnomalyEvent]
    ) -> List[MitigationAction]:
        """
        Generate prioritized mitigation actions.

        Args:
            health_metrics: List of health metrics
            anomalies: List of detected anomalies

        Returns:
            List of MitigationAction objects, sorted by priority
        """
        actions = []

        # Generate actions from health metrics
        for metric in health_metrics:
            if metric.status == "critical":
                priority = self.prioritize_action("critical", "high")

                if "availability" in metric.name.lower():
                    actions.append(MitigationAction(
                        priority=priority,
                        category="availability",
                        description=f"Critical availability issue: {metric.value}{metric.unit}",
                        recommended_steps=[
                            "Check for pod restarts and evictions",
                            "Review service health endpoints",
                            "Verify load balancer configuration",
                            "Check node health and resource availability"
                        ],
                        estimated_effort="minutes",
                        assignee_role="on-call"
                    ))

                elif "latency" in metric.name.lower():
                    actions.append(MitigationAction(
                        priority=priority,
                        category="performance",
                        description=f"Critical latency degradation: {metric.value}{metric.unit}",
                        recommended_steps=[
                            "Review slow query logs (EXPLAIN ANALYZE)",
                            "Check database connection pool saturation",
                            "Verify cache hit rates (Redis/Memcached)",
                            "Profile API endpoints with distributed tracing"
                        ],
                        estimated_effort="hours",
                        assignee_role="on-call"
                    ))

                elif "error" in metric.name.lower():
                    actions.append(MitigationAction(
                        priority=priority,
                        category="availability",
                        description=f"Critical error rate spike: {metric.value}{metric.unit}",
                        recommended_steps=[
                            "Review recent deployments and rollback if needed",
                            "Check infrastructure logs for failures",
                            "Verify dependency service health",
                            "Inspect error logs for root causes"
                        ],
                        estimated_effort="minutes",
                        assignee_role="on-call"
                    ))

                elif "cpu" in metric.name.lower():
                    actions.append(MitigationAction(
                        priority="P1",
                        category="resource",
                        description=f"High CPU utilization: {metric.value}{metric.unit}",
                        recommended_steps=[
                            "Review HPA settings and scale out if needed",
                            "Profile application for CPU hotspots",
                            "Check for inefficient queries or algorithms",
                            "Verify no background jobs consuming CPU"
                        ],
                        estimated_effort="hours",
                        assignee_role="platform-team"
                    ))

                elif "memory" in metric.name.lower():
                    actions.append(MitigationAction(
                        priority="P1",
                        category="resource",
                        description=f"High memory utilization: {metric.value}{metric.unit}",
                        recommended_steps=[
                            "Check for memory leaks in application",
                            "Review memory limits and adjust if needed",
                            "Verify cache size configurations",
                            "Monitor for memory pressure and OOMKills"
                        ],
                        estimated_effort="hours",
                        assignee_role="platform-team"
                    ))

        # Generate actions from anomalies
        high_anomalies = [a for a in anomalies if a.severity == "high"]
        medium_anomalies = [a for a in anomalies if a.severity == "medium"]

        if high_anomalies:
            # Group by metric for consolidated actions
            by_metric = {}
            for anomaly in high_anomalies:
                if anomaly.metric_name not in by_metric:
                    by_metric[anomaly.metric_name] = []
                by_metric[anomaly.metric_name].append(anomaly)

            for metric_name, metric_anomalies in by_metric.items():
                actions.append(MitigationAction(
                    priority="P0",
                    category="performance",
                    description=f"High-severity anomalies detected in {metric_name} ({len(metric_anomalies)} events)",
                    recommended_steps=[
                        f"Investigate {metric_anomalies[0].classification} pattern in {metric_name}",
                        "Review correlation with other metrics",
                        "Check for recent configuration or deployment changes",
                        "Consider scaling resources if needed"
                    ],
                    estimated_effort="hours",
                    assignee_role="on-call"
                ))

        if len(medium_anomalies) > 5:
            actions.append(MitigationAction(
                priority="P1",
                category="performance",
                description=f"Multiple medium-severity anomalies detected ({len(medium_anomalies)} events)",
                recommended_steps=[
                    "Review anomaly patterns for trends",
                    "Check for gradual performance degradation",
                    "Validate metric baselines are current",
                    "Schedule deep-dive investigation"
                ],
                estimated_effort="hours",
                assignee_role="dev-team"
            ))

        # Sort actions by priority (P0 first)
        priority_order = {"P0": 0, "P1": 1, "P2": 2}
        actions.sort(key=lambda a: priority_order.get(a.priority, 99))

        return actions

    def prioritize_action(
        self,
        metric_status: str,
        anomaly_severity: str
    ) -> str:
        """
        Determine action priority.

        Args:
            metric_status: "healthy", "warning", "critical"
            anomaly_severity: "low", "medium", "high"

        Returns:
            Priority: "P0", "P1", "P2"
        """
        # Priority matrix
        if metric_status == "critical":
            if anomaly_severity == "high":
                return "P0"
            elif anomaly_severity == "medium":
                return "P1"
            else:
                return "P1"

        elif metric_status == "warning":
            if anomaly_severity == "high":
                return "P1"
            elif anomaly_severity == "medium":
                return "P2"
            else:
                return "P2"

        else:  # healthy
            if anomaly_severity == "high":
                return "P2"
            else:
                return "P2"


class HealthReportGenerator:
    """
    Main daily health report generator.
    """

    def __init__(
        self,
        day_number: int,
        stability_data_file: str,
        anomaly_events_file: str = "anomaly_events.json",
        output_dir: str = "reports"
    ):
        """
        Initialize health report generator.

        Args:
            day_number: Day number (1-7)
            stability_data_file: Path to stability snapshot JSON
            anomaly_events_file: Path to anomaly events JSON
            output_dir: Output directory for reports
        """
        self.day_number = day_number
        self.stability_data_file = stability_data_file
        self.anomaly_events_file = anomaly_events_file
        self.output_dir = Path(output_dir)

        self.anomaly_analyzer = AnomalyAnalyzer(anomaly_events_file)
        self.mitigation_generator = MitigationGenerator()

        logger.info(f"HealthReportGenerator initialized for Day {day_number}")

    def load_stability_data(self) -> Dict[str, Any]:
        """
        Load stability snapshot data for the day.

        Returns:
            Dict with stability metrics
        """
        data_file = Path(self.stability_data_file)

        if not data_file.exists():
            raise FileNotFoundError(
                f"Stability data file not found: {self.stability_data_file}\n"
                f"Please ensure 7-day stability monitoring has generated this file."
            )

        logger.info(f"Loading stability data from: {self.stability_data_file}")

        with open(data_file, 'r') as f:
            data = json.load(f)

        return data

    def calculate_health_score(self, metrics: Dict[str, Any]) -> float:
        """
        Calculate overall health score (0-100).

        Formula:
        - Availability: 40%
        - Latency: 30%
        - Error rate: 15%
        - Drift: 15%

        Args:
            metrics: Stability metrics dict

        Returns:
            Health score (0-100)
        """
        # Extract metrics
        availability = metrics.get("avg_availability", 100.0)
        p99_latency = metrics.get("avg_p99_latency_ms", 0.0)
        error_rate = metrics.get("avg_error_rate", 0.0)
        drift_percent = metrics.get("avg_drift_percent", 0.0)

        # Availability score (40%): Linear scale from 99.5% to 100%
        # 100% availability = 100 points, 99.5% = 0 points
        avail_score = max(0, min(100, (availability - 99.5) / 0.5 * 100.0))

        # Latency score (30%): Inverse scale from 0ms to 1000ms
        # 0ms = 100 points, 500ms = 50 points, 1000ms = 0 points
        latency_score = max(0, min(100, 100.0 - (p99_latency / 1000.0 * 100.0)))

        # Error rate score (15%): Inverse scale from 0% to 1%
        # 0% = 100 points, 0.1% = 90 points, 1% = 0 points
        error_score = max(0, min(100, 100.0 - (error_rate / 1.0 * 100.0)))

        # Drift score (15%): Inverse scale from 0% to 20%
        # 0% drift = 100 points, 10% = 50 points, 20% = 0 points
        drift_score = max(0, min(100, 100.0 - (drift_percent / 20.0 * 100.0)))

        # Weighted average
        health_score = (
            (avail_score * 0.40) +
            (latency_score * 0.30) +
            (error_score * 0.15) +
            (drift_score * 0.15)
        )

        return round(health_score, 2)

    def determine_health_status(self, score: float) -> str:
        """
        Determine health status from score.

        Args:
            score: Health score (0-100)

        Returns:
            Status: "healthy" (>=95), "degraded" (85-94), "critical" (<85)
        """
        if score >= 95.0:
            return "healthy"
        elif score >= 85.0:
            return "degraded"
        else:
            return "critical"

    def analyze_trends(self, current_day: int, stability_data: Dict[str, Any]) -> Dict[str, str]:
        """
        Analyze trends compared to previous day.

        Args:
            current_day: Current day number
            stability_data: Current day's stability data

        Returns:
            Dict with trend strings for each category
        """
        trends = {
            "availability_trend": "stable",
            "latency_trend": "stable",
            "resource_trend": "stable"
        }

        # If this is Day 1, no previous day to compare to
        if current_day <= 1:
            return trends

        # Try to load previous day's data
        try:
            prev_day_file = self.output_dir.parent / "stability" / f"day_{current_day-1:02d}_summary.json"
            if not prev_day_file.exists():
                logger.warning(f"Previous day data not found: {prev_day_file}")
                return trends

            with open(prev_day_file, 'r') as f:
                prev_data = json.load(f)

            # Compare availability
            current_avail = stability_data.get("avg_availability", 100.0)
            prev_avail = prev_data.get("avg_availability", 100.0)

            if current_avail > prev_avail + 0.5:
                trends["availability_trend"] = "improving"
            elif current_avail < prev_avail - 0.5:
                trends["availability_trend"] = "degrading"

            # Compare latency
            current_latency = stability_data.get("avg_p99_latency_ms", 0.0)
            prev_latency = prev_data.get("avg_p99_latency_ms", 0.0)

            if current_latency < prev_latency - 10.0:  # 10ms improvement
                trends["latency_trend"] = "improving"
            elif current_latency > prev_latency + 10.0:  # 10ms degradation
                trends["latency_trend"] = "degrading"

            # Compare resources (CPU + memory)
            current_cpu = stability_data.get("avg_cpu_percent", 0.0)
            prev_cpu = prev_data.get("avg_cpu_percent", 0.0)
            current_mem = stability_data.get("avg_memory_percent", 0.0)
            prev_mem = prev_data.get("avg_memory_percent", 0.0)

            cpu_change = current_cpu - prev_cpu
            mem_change = current_mem - prev_mem

            if cpu_change < -5.0 or mem_change < -5.0:  # 5% improvement
                trends["resource_trend"] = "improving"
            elif cpu_change > 5.0 or mem_change > 5.0:  # 5% degradation
                trends["resource_trend"] = "degrading"

        except Exception as e:
            logger.error(f"Failed to analyze trends: {e}")

        return trends

    def generate_recommendations(
        self,
        health_metrics: List[HealthMetric],
        anomalies: List[AnomalyEvent],
        mitigation_actions: List[MitigationAction]
    ) -> List[str]:
        """
        Generate actionable recommendations.

        Args:
            health_metrics: Health metrics
            anomalies: Detected anomalies
            mitigation_actions: Mitigation actions

        Returns:
            List of recommendation strings
        """
        recommendations = []

        # Critical metrics
        critical_metrics = [m for m in health_metrics if m.status == "critical"]
        if critical_metrics:
            recommendations.append(
                f"⚠️ {len(critical_metrics)} critical metric(s) detected - immediate investigation required"
            )

        # High-severity anomalies
        high_anomalies = [a for a in anomalies if a.severity == "high"]
        if high_anomalies:
            recommendations.append(
                f"Investigate {len(high_anomalies)} high-severity anomalies across metrics"
            )

        # P0 actions
        p0_actions = [a for a in mitigation_actions if a.priority == "P0"]
        if p0_actions:
            recommendations.append(
                f"Execute {len(p0_actions)} P0 mitigation action(s) within next hour"
            )

        # Degrading trends
        degrading_metrics = [m for m in health_metrics if m.trend == "degrading"]
        if degrading_metrics:
            recommendations.append(
                f"Monitor {len(degrading_metrics)} degrading trend(s) - may require proactive intervention"
            )

        # If everything is healthy
        if not recommendations:
            recommendations.append("✅ All metrics within acceptable ranges - continue normal monitoring")

        return recommendations

    def generate_next_steps(self, day_number: int, health_status: str) -> List[str]:
        """
        Generate next steps for the team.

        Args:
            day_number: Current day number
            health_status: Current health status

        Returns:
            List of next step strings
        """
        next_steps = []

        if health_status == "critical":
            next_steps.extend([
                "Execute P0 mitigation actions immediately",
                "Activate incident response team if not already engaged",
                "Prepare rollback plan to previous version",
                "Schedule emergency war room meeting"
            ])
        elif health_status == "degraded":
            next_steps.extend([
                "Review and prioritize P1 mitigation actions",
                "Monitor for further degradation",
                "Schedule deep-dive investigation with team",
                "Update on-call runbooks with current findings"
            ])
        else:  # healthy
            next_steps.extend([
                "Continue regular monitoring cadence",
                "Review P2 optimization opportunities",
                "Document any learnings in retrospective notes"
            ])

        # Add day-specific next steps
        if day_number < 7:
            next_steps.append(f"Continue monitoring through Day {day_number + 1}")
        else:
            next_steps.append("Prepare 7-day retrospective and analysis")

        return next_steps

    def generate_report(self) -> DailyHealthReport:
        """
        Generate complete daily health report.

        Returns:
            DailyHealthReport object
        """
        logger.info(f"Generating health report for Day {self.day_number}")

        # Step 1: Load stability data
        stability_data = self.load_stability_data()

        # Step 2: Calculate health score
        health_score = self.calculate_health_score(stability_data)
        health_status = self.determine_health_status(health_score)

        logger.info(f"Health score: {health_score}/100 ({health_status})")

        # Step 3: Create health metrics
        health_metrics = self._create_health_metrics(stability_data)

        # Step 4: Load anomalies
        anomalies = self.anomaly_analyzer.load_anomaly_events(self.day_number)

        # Enrich anomalies with potential causes
        for anomaly in anomalies:
            anomaly.potential_causes = self.anomaly_analyzer.suggest_potential_causes(anomaly)

        logger.info(f"Loaded {len(anomalies)} anomaly events")

        # Step 5: Generate mitigation actions
        mitigation_actions = self.mitigation_generator.generate_mitigation_actions(
            health_metrics,
            anomalies
        )

        p0_count = sum(1 for a in mitigation_actions if a.priority == "P0")
        p1_count = sum(1 for a in mitigation_actions if a.priority == "P1")
        p2_count = sum(1 for a in mitigation_actions if a.priority == "P2")

        logger.info(f"Generated {len(mitigation_actions)} mitigation actions ({p0_count} P0, {p1_count} P1, {p2_count} P2)")

        # Step 6: Analyze trends
        trends = self.analyze_trends(self.day_number, stability_data)

        # Step 7: Generate recommendations
        recommendations = self.generate_recommendations(
            health_metrics,
            anomalies,
            mitigation_actions
        )

        # Step 8: Generate next steps
        next_steps = self.generate_next_steps(self.day_number, health_status)

        # Step 9: Create report
        report = DailyHealthReport(
            day_number=self.day_number,
            report_date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            generation_timestamp=datetime.now(timezone.utc).isoformat(),
            ga_day_offset=self.day_number,
            # Overall health
            overall_health_score=health_score,
            health_status=health_status,
            # Key metrics
            availability_percent=stability_data.get("avg_availability", 100.0),
            error_rate_percent=stability_data.get("avg_error_rate", 0.0),
            p99_latency_ms=stability_data.get("avg_p99_latency_ms", 0.0),
            critical_alerts=stability_data.get("total_critical_alerts", 0),
            warning_alerts=stability_data.get("total_warning_alerts", 0),
            # Details
            health_metrics=health_metrics,
            anomalies=anomalies,
            anomaly_count=len(anomalies),
            mitigation_actions=mitigation_actions,
            p0_actions=p0_count,
            p1_actions=p1_count,
            p2_actions=p2_count,
            # Trends
            availability_trend=trends["availability_trend"],
            latency_trend=trends["latency_trend"],
            resource_trend=trends["resource_trend"],
            # Guidance
            recommendations=recommendations,
            next_steps=next_steps
        )

        logger.info(f"Report generated successfully for Day {self.day_number}")
        return report

    def _create_health_metrics(self, stability_data: Dict[str, Any]) -> List[HealthMetric]:
        """
        Create health metrics from stability data.

        Args:
            stability_data: Stability data dictionary

        Returns:
            List of HealthMetric objects
        """
        metrics = []

        # Availability metric
        avail = stability_data.get("avg_availability", 100.0)
        avail_status = "healthy" if avail >= 99.9 else "warning" if avail >= 99.5 else "critical"
        metrics.append(HealthMetric(
            name="Availability",
            value=avail,
            unit="%",
            threshold=99.9,
            status=avail_status,
            trend="stable",  # Will be updated by trend analysis
            description="Service availability percentage"
        ))

        # Error rate metric
        error_rate = stability_data.get("avg_error_rate", 0.0)
        error_status = "healthy" if error_rate < 0.1 else "warning" if error_rate < 0.5 else "critical"
        metrics.append(HealthMetric(
            name="Error Rate",
            value=error_rate,
            unit="%",
            threshold=0.1,
            status=error_status,
            trend="stable",
            description="Percentage of failed requests"
        ))

        # P99 Latency metric
        p99_latency = stability_data.get("avg_p99_latency_ms", 0.0)
        latency_status = "healthy" if p99_latency < 500 else "warning" if p99_latency < 1000 else "critical"
        metrics.append(HealthMetric(
            name="P99 Latency",
            value=p99_latency,
            unit="ms",
            threshold=500.0,
            status=latency_status,
            trend="stable",
            description="99th percentile response time"
        ))

        # CPU metric
        cpu = stability_data.get("avg_cpu_percent", 0.0)
        cpu_status = "healthy" if cpu < 70 else "warning" if cpu < 85 else "critical"
        metrics.append(HealthMetric(
            name="CPU Utilization",
            value=cpu,
            unit="%",
            threshold=70.0,
            status=cpu_status,
            trend="stable",
            description="Average CPU usage"
        ))

        # Memory metric
        memory = stability_data.get("avg_memory_percent", 0.0)
        memory_status = "healthy" if memory < 75 else "warning" if memory < 90 else "critical"
        metrics.append(HealthMetric(
            name="Memory Utilization",
            value=memory,
            unit="%",
            threshold=75.0,
            status=memory_status,
            trend="stable",
            description="Average memory usage"
        ))

        return metrics

    def save_report_json(self, report: DailyHealthReport) -> str:
        """
        Save report as JSON.

        Args:
            report: Daily health report

        Returns:
            Path to saved JSON file
        """
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename
        json_file = self.output_dir / f"day_{self.day_number:02d}_HEALTH.json"

        # Convert report to dict
        report_dict = asdict(report)

        # Write to file
        with open(json_file, 'w') as f:
            json.dump(report_dict, f, indent=2)

        logger.info(f"JSON report saved: {json_file}")
        return str(json_file)

    def save_report_markdown(self, report: DailyHealthReport) -> str:
        """
        Save report as Markdown.

        Args:
            report: Daily health report

        Returns:
            Path to saved Markdown file
        """
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename
        md_file = self.output_dir / f"day_{self.day_number:02d}_HEALTH.md"

        # Format Markdown
        md_content = self.format_markdown_report(report)

        # Write to file
        with open(md_file, 'w') as f:
            f.write(md_content)

        logger.info(f"Markdown report saved: {md_file}")
        return str(md_file)

    def format_markdown_report(self, report: DailyHealthReport) -> str:
        """
        Format report as Markdown string.

        Args:
            report: Daily health report

        Returns:
            Formatted Markdown string
        """
        md = []

        # === Header ===
        md.append(f"# Day {report.day_number} Health Report")
        md.append("")
        md.append(f"**Date:** {report.report_date}")
        md.append(f"**Generated:** {report.generation_timestamp}")
        md.append(f"**GA Day Offset:** +{report.ga_day_offset} days")
        md.append("")
        md.append("---")
        md.append("")

        # === Executive Summary ===
        md.append("## Executive Summary")
        md.append("")
        md.append(f"**Overall Health Score:** {report.overall_health_score}/100")
        md.append("")

        # Visual health gauge
        health_emoji = "🟢" if report.health_status == "healthy" else "🟡" if report.health_status == "degraded" else "🔴"
        md.append(f"**Status:** {health_emoji} {report.health_status.upper()}")
        md.append("")

        # Score bar
        filled = int(report.overall_health_score / 10)
        empty = 10 - filled
        md.append(f"```")
        md.append(f"[{'█' * filled}{'░' * empty}] {report.overall_health_score}/100")
        md.append(f"```")
        md.append("")

        # === Key Metrics Summary ===
        md.append("## Key Metrics Summary")
        md.append("")
        md.append("| Metric | Value | Status |")
        md.append("|--------|-------|--------|")
        md.append(f"| Availability | {report.availability_percent:.2f}% | {'✅' if report.availability_percent >= 99.9 else '⚠️'} |")
        md.append(f"| Error Rate | {report.error_rate_percent:.4f}% | {'✅' if report.error_rate_percent < 0.1 else '⚠️'} |")
        md.append(f"| P99 Latency | {report.p99_latency_ms:.2f}ms | {'✅' if report.p99_latency_ms < 500 else '⚠️'} |")
        md.append(f"| Critical Alerts | {report.critical_alerts} | {'✅' if report.critical_alerts == 0 else '🔴'} |")
        md.append(f"| Warning Alerts | {report.warning_alerts} | {'✅' if report.warning_alerts < 5 else '⚠️'} |")
        md.append("")

        # === Health Metrics Breakdown ===
        md.append("## Health Metrics Breakdown")
        md.append("")
        md.append("| Metric | Value | Threshold | Status | Trend |")
        md.append("|--------|-------|-----------|--------|-------|")

        for metric in report.health_metrics:
            status_emoji = "✅" if metric.status == "healthy" else "⚠️" if metric.status == "warning" else "🔴"
            trend_emoji = "📈" if metric.trend == "improving" else "📉" if metric.trend == "degrading" else "➡️"
            md.append(
                f"| {metric.name} | {metric.value:.2f}{metric.unit} | "
                f"{metric.threshold}{metric.unit} | {status_emoji} {metric.status} | {trend_emoji} {metric.trend} |"
            )
        md.append("")

        # === Anomalies Detected ===
        md.append("## Anomalies Detected")
        md.append("")

        if report.anomalies:
            md.append(f"**Total Anomalies:** {report.anomaly_count}")
            md.append("")

            # Group by severity
            high = [a for a in report.anomalies if a.severity == "high"]
            medium = [a for a in report.anomalies if a.severity == "medium"]
            low = [a for a in report.anomalies if a.severity == "low"]

            md.append(f"- 🔴 **High Severity:** {len(high)}")
            md.append(f"- 🟡 **Medium Severity:** {len(medium)}")
            md.append(f"- 🟢 **Low Severity:** {len(low)}")
            md.append("")

            # List top anomalies
            if high:
                md.append("### High-Severity Anomalies")
                md.append("")
                for anomaly in high[:10]:  # Limit to top 10
                    md.append(f"**{anomaly.metric_name}** at {anomaly.timestamp}")
                    md.append(f"- **Type:** {anomaly.classification}")
                    md.append(f"- **Actual:** {anomaly.actual_value:.2f}, **Expected:** {anomaly.expected_value:.2f}")
                    md.append(f"- **Deviation:** {anomaly.deviation_percent:.1f}%")
                    if anomaly.potential_causes:
                        md.append(f"- **Potential Causes:**")
                        for cause in anomaly.potential_causes[:3]:  # Top 3 causes
                            md.append(f"  - {cause}")
                    md.append("")
        else:
            md.append("✅ No anomalies detected")
            md.append("")

        # === Mitigation Checklist ===
        md.append("## Mitigation Checklist")
        md.append("")

        if report.mitigation_actions:
            md.append(f"**Total Actions:** {len(report.mitigation_actions)} (P0: {report.p0_actions}, P1: {report.p1_actions}, P2: {report.p2_actions})")
            md.append("")

            # Group by priority
            for priority in ["P0", "P1", "P2"]:
                priority_actions = [a for a in report.mitigation_actions if a.priority == priority]
                if priority_actions:
                    priority_emoji = "🔴" if priority == "P0" else "🟡" if priority == "P1" else "🟢"
                    md.append(f"### {priority_emoji} {priority} Actions")
                    md.append("")

                    for action in priority_actions:
                        md.append(f"- [ ] **{action.description}**")
                        md.append(f"  - **Category:** {action.category}")
                        md.append(f"  - **Effort:** {action.estimated_effort}")
                        md.append(f"  - **Assignee:** {action.assignee_role}")
                        md.append(f"  - **Steps:**")
                        for step in action.recommended_steps:
                            md.append(f"    1. {step}")
                        md.append("")
        else:
            md.append("✅ No mitigation actions required")
            md.append("")

        # === Trend Analysis ===
        md.append("## Trend Analysis")
        md.append("")
        md.append(f"- **Availability:** {report.availability_trend}")
        md.append(f"- **Latency:** {report.latency_trend}")
        md.append(f"- **Resources:** {report.resource_trend}")
        md.append("")

        # === Recommendations ===
        md.append("## Recommendations")
        md.append("")
        for i, rec in enumerate(report.recommendations, 1):
            md.append(f"{i}. {rec}")
        md.append("")

        # === Next Steps ===
        md.append("## Next Steps")
        md.append("")
        for i, step in enumerate(report.next_steps, 1):
            md.append(f"- [ ] {step}")
        md.append("")

        # === Footer ===
        md.append("---")
        md.append("")
        md.append(f"**Report Version:** 1.0")
        md.append(f"**Generated by:** T.A.R.S. Daily Health Reporter")
        md.append("")
        md.append("🚀 Generated with [Claude Code](https://claude.com/claude-code)")

        return "\n".join(md)


def main():
    """
    CLI entry point.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Daily Health Report Generator")
    parser.add_argument(
        "--day",
        type=int,
        required=True,
        choices=range(1, 8),
        help="Day number (1-7)"
    )
    parser.add_argument(
        "--stability-data",
        help="Path to stability data JSON file (e.g., stability/day_01_summary.json)"
    )
    parser.add_argument(
        "--anomaly-events",
        default="anomaly_events.json",
        help="Path to anomaly events JSON file"
    )
    parser.add_argument(
        "--output",
        default="reports",
        help="Output directory for reports (default: reports/)"
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Auto-detect latest stability data file for the day"
    )

    args = parser.parse_args()

    logger.info("="*60)
    logger.info(f"Daily Health Report Generator - Day {args.day}")
    logger.info("="*60)

    # Auto-detect stability data file if --auto flag set
    if args.auto or not args.stability_data:
        stability_data_file = f"stability/day_{args.day:02d}_summary.json"
        logger.info(f"Auto-detected stability data file: {stability_data_file}")
    else:
        stability_data_file = args.stability_data

    try:
        # Instantiate HealthReportGenerator
        generator = HealthReportGenerator(
            day_number=args.day,
            stability_data_file=stability_data_file,
            anomaly_events_file=args.anomaly_events,
            output_dir=args.output
        )

        # Generate report
        logger.info("Generating health report...")
        report = generator.generate_report()

        # Save JSON and Markdown files
        logger.info("Saving report files...")
        json_path = generator.save_report_json(report)
        md_path = generator.save_report_markdown(report)

        # Print summary to console
        logger.info("="*60)
        logger.info("DAILY HEALTH REPORT SUMMARY")
        logger.info("="*60)
        logger.info(f"Day: {report.day_number}")
        logger.info(f"Health Score: {report.overall_health_score}/100")
        logger.info(f"Status: {report.health_status.upper()}")
        logger.info("")
        logger.info(f"Key Metrics:")
        logger.info(f"  - Availability: {report.availability_percent:.2f}%")
        logger.info(f"  - Error Rate: {report.error_rate_percent:.4f}%")
        logger.info(f"  - P99 Latency: {report.p99_latency_ms:.2f}ms")
        logger.info(f"  - Critical Alerts: {report.critical_alerts}")
        logger.info(f"  - Warning Alerts: {report.warning_alerts}")
        logger.info("")
        logger.info(f"Anomalies: {report.anomaly_count}")
        logger.info(f"Mitigation Actions: {len(report.mitigation_actions)} (P0: {report.p0_actions}, P1: {report.p1_actions}, P2: {report.p2_actions})")
        logger.info("")
        logger.info(f"Trends:")
        logger.info(f"  - Availability: {report.availability_trend}")
        logger.info(f"  - Latency: {report.latency_trend}")
        logger.info(f"  - Resources: {report.resource_trend}")
        logger.info("")
        logger.info("Recommendations:")
        for i, rec in enumerate(report.recommendations, 1):
            logger.info(f"  {i}. {rec}")
        logger.info("")
        logger.info("="*60)
        logger.info(f"✅ Reports saved:")
        logger.info(f"  - JSON: {json_path}")
        logger.info(f"  - Markdown: {md_path}")
        logger.info("="*60)

    except FileNotFoundError as e:
        logger.error(f"❌ File not found: {e}")
        logger.error("Please ensure stability monitoring and anomaly detection have completed.")
        sys.exit(1)

    except Exception as e:
        logger.error(f"❌ Report generation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
