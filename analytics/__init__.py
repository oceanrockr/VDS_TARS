"""
Analytics Module - Repository Health Dashboard & Alerting Engine

This module provides comprehensive repository health monitoring by aggregating
data from all release pipeline components (Tasks 3-7):
- Verification reports (Task 3)
- Validation reports (Task 4)
- Publication reports (Task 5)
- Rollback reports (Task 6)
- Integrity scanner reports (Task 7)

Main Components:
- ReportAggregator: Loads and normalizes reports from multiple sources
- RepositoryHealthDashboard: Orchestrates health analysis and scoring
- HTMLRenderer: Generates human-readable HTML dashboards
- AlertingEngine: Evaluates rules and generates alerts (Task 9)
- AlertDispatcher: Dispatches alerts to multiple channels

Exit Codes (60-69) - Dashboard:
- 60: Health OK (Green)
- 61: Health Warning (Yellow)
- 62: Health Critical (Red)
- 63: Aggregation Failure
- 64: Missing Reports
- 65: Malformed Report
- 66: HTML Render Failure
- 67: Dashboard Write Failure
- 68: Health Threshold Violation
- 69: General Dashboard Error

Exit Codes (70-79) - Alerting:
- 70: No alerts triggered
- 71: Alerts triggered (non-critical)
- 72: Critical alerts triggered
- 73: Invalid dashboard input
- 74: Channel dispatch failure
- 75: Alert rule evaluation failure
- 76: Alerts JSON write failure
- 79: General alerting error

Version: 1.1.0
Phase: 14.7 Task 9
"""

from typing import Dict, List, Any

__version__ = "1.1.0"
__all__ = [
    # Dashboard components (Task 8)
    "ReportAggregator",
    "RepositoryHealthDashboard",
    "HTMLRenderer",
    "HealthStatus",
    "HealthReport",
    "DashboardConfig",
    "DashboardError",
    "AggregationError",
    "MalformedReportError",
    "MissingReportsError",
    "HTMLRenderError",
    "DashboardWriteError",
    "HealthThresholdError",
    # Alerting components (Task 9)
    "AlertingEngine",
    "AlertDispatcher",
    "AlertRulesEngine",
    "Alert",
    "AlertReport",
    "AlertRule",
    "AlertSeverity",
    "AlertType",
    "AlertingConfig",
    "ChannelConfig",
    "ChannelType",
    "AlertingError",
    "InvalidDashboardError",
    "ChannelDispatchError",
    "RuleEvaluationError",
    "AlertWriteError",
]

# Lazy imports to avoid circular dependencies
def __getattr__(name: str) -> Any:
    """Lazy import of module components."""
    # Dashboard components (Task 8)
    if name == "ReportAggregator":
        from analytics.report_aggregator import ReportAggregator
        return ReportAggregator
    elif name == "RepositoryHealthDashboard":
        from analytics.repository_health_dashboard import RepositoryHealthDashboard
        return RepositoryHealthDashboard
    elif name == "HTMLRenderer":
        from analytics.html_renderer import HTMLRenderer
        return HTMLRenderer
    elif name == "HealthStatus":
        from analytics.repository_health_dashboard import HealthStatus
        return HealthStatus
    elif name == "HealthReport":
        from analytics.repository_health_dashboard import HealthReport
        return HealthReport
    elif name == "DashboardConfig":
        from analytics.repository_health_dashboard import DashboardConfig
        return DashboardConfig
    elif name in [
        "DashboardError",
        "AggregationError",
        "MalformedReportError",
        "MissingReportsError",
        "HTMLRenderError",
        "DashboardWriteError",
        "HealthThresholdError",
    ]:
        from analytics.repository_health_dashboard import (
            DashboardError,
            AggregationError,
            MalformedReportError,
            MissingReportsError,
            HTMLRenderError,
            DashboardWriteError,
            HealthThresholdError,
        )
        return locals()[name]

    # Alerting components (Task 9)
    elif name == "AlertingEngine":
        from analytics.alerting_engine import AlertingEngine
        return AlertingEngine
    elif name == "AlertDispatcher":
        from analytics.alerting_engine import AlertDispatcher
        return AlertDispatcher
    elif name == "AlertRulesEngine":
        from analytics.alerting_engine import AlertRulesEngine
        return AlertRulesEngine
    elif name == "Alert":
        from analytics.alerting_engine import Alert
        return Alert
    elif name == "AlertReport":
        from analytics.alerting_engine import AlertReport
        return AlertReport
    elif name == "AlertRule":
        from analytics.alerting_engine import AlertRule
        return AlertRule
    elif name == "AlertSeverity":
        from analytics.alerting_engine import AlertSeverity
        return AlertSeverity
    elif name == "AlertType":
        from analytics.alerting_engine import AlertType
        return AlertType
    elif name == "AlertingConfig":
        from analytics.alerting_engine import AlertingConfig
        return AlertingConfig
    elif name == "ChannelConfig":
        from analytics.alerting_engine import ChannelConfig
        return ChannelConfig
    elif name == "ChannelType":
        from analytics.alerting_engine import ChannelType
        return ChannelType
    elif name in [
        "AlertingError",
        "InvalidDashboardError",
        "ChannelDispatchError",
        "RuleEvaluationError",
        "AlertWriteError",
    ]:
        from analytics.alerting_engine import (
            AlertingError,
            InvalidDashboardError,
            ChannelDispatchError,
            RuleEvaluationError,
            AlertWriteError,
        )
        return locals()[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
