"""
Repository Health Alerting Engine - Alert Generation and Dispatch

This module implements a complete alerting engine for the T.A.R.S. release pipeline:
1. Evaluates alert rules against dashboard JSON data
2. Generates typed alerts with severity levels
3. Dispatches alerts to multiple channels (console, file, email, webhook)
4. Supports trend-based alerts by comparing current vs. previous dashboards
5. Provides policy-based exit codes (70-79)

Alert Types:
- RepositoryStatusAlert: Health status is RED or YELLOW
- CriticalIssueAlert: Issues with severity = CRITICAL
- MissingArtifactAlert: Missing artifacts detected
- CorruptedArtifactAlert: Corrupted artifacts detected
- MetadataMissingAlert: SBOM/SLSA metadata missing
- RapidRegressionAlert: Multiple new issues since last run
- VersionHealthAlert: Version health degradation
- RepositoryScoreDropAlert: Significant score drop

Exit Codes (70-79):
- 70: No alerts triggered (normal)
- 71: Alerts triggered (non-critical)
- 72: Critical alerts triggered
- 73: Invalid dashboard input
- 74: Channel dispatch failure
- 75: Alert rule evaluation failure
- 76: Alerts JSON write failure
- 79: General alerting engine error

Version: 1.0.0
Phase: 14.7 Task 9
"""

import json
import logging
import smtplib
import urllib.request
import urllib.error
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum, auto
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Set, Type

logger = logging.getLogger(__name__)


# ============================================================================
# Exit Codes (70-79)
# ============================================================================

EXIT_NO_ALERTS = 70
EXIT_ALERTS_TRIGGERED = 71
EXIT_CRITICAL_ALERTS = 72
EXIT_INVALID_DASHBOARD = 73
EXIT_CHANNEL_DISPATCH_FAILURE = 74
EXIT_RULE_EVALUATION_FAILURE = 75
EXIT_ALERTS_WRITE_FAILURE = 76
EXIT_GENERAL_ALERTING_ERROR = 79


# ============================================================================
# Custom Exceptions
# ============================================================================

class AlertingError(Exception):
    """Base exception for alerting engine errors."""
    exit_code = EXIT_GENERAL_ALERTING_ERROR


class InvalidDashboardError(AlertingError):
    """Dashboard JSON is invalid or malformed."""
    exit_code = EXIT_INVALID_DASHBOARD


class ChannelDispatchError(AlertingError):
    """Failed to dispatch alerts to channel."""
    exit_code = EXIT_CHANNEL_DISPATCH_FAILURE


class RuleEvaluationError(AlertingError):
    """Failed to evaluate alert rule."""
    exit_code = EXIT_RULE_EVALUATION_FAILURE


class AlertWriteError(AlertingError):
    """Failed to write alerts JSON."""
    exit_code = EXIT_ALERTS_WRITE_FAILURE


# ============================================================================
# Enums
# ============================================================================

class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

    @classmethod
    def from_string(cls, value: str) -> "AlertSeverity":
        """Convert string to AlertSeverity."""
        try:
            return cls[value.upper()]
        except KeyError:
            return cls.INFO

    def __lt__(self, other: "AlertSeverity") -> bool:
        """Compare severity levels."""
        order = {self.INFO: 0, self.WARNING: 1, self.ERROR: 2, self.CRITICAL: 3}
        return order[self] < order[other]

    def __le__(self, other: "AlertSeverity") -> bool:
        order = {self.INFO: 0, self.WARNING: 1, self.ERROR: 2, self.CRITICAL: 3}
        return order[self] <= order[other]


class AlertType(Enum):
    """Types of alerts that can be triggered."""
    REPOSITORY_STATUS = "repository_status"
    CRITICAL_ISSUE = "critical_issue"
    MISSING_ARTIFACT = "missing_artifact"
    CORRUPTED_ARTIFACT = "corrupted_artifact"
    METADATA_MISSING = "metadata_missing"
    RAPID_REGRESSION = "rapid_regression"
    VERSION_HEALTH = "version_health"
    REPOSITORY_SCORE_DROP = "repository_score_drop"
    ORPHANED_ARTIFACT = "orphaned_artifact"
    ROLLBACK_FAILURE = "rollback_failure"
    CUSTOM = "custom"


class ChannelType(Enum):
    """Types of alert dispatch channels."""
    CONSOLE = "console"
    FILE = "file"
    EMAIL = "email"
    WEBHOOK = "webhook"


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class Alert:
    """Represents a single alert."""
    alert_id: str
    alert_type: str  # AlertType value
    severity: str    # AlertSeverity value
    title: str
    message: str
    timestamp: str
    source: str = "alerting_engine"

    # Additional context
    version: Optional[str] = None
    artifact: Optional[str] = None
    issue_count: int = 0
    score_change: Optional[float] = None
    previous_status: Optional[str] = None
    current_status: Optional[str] = None

    # Metadata
    rule_name: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        return asdict(self)


@dataclass
class AlertRule:
    """Defines a rule for generating alerts."""
    name: str
    alert_type: AlertType
    severity: AlertSeverity
    description: str
    enabled: bool = True

    # Rule configuration
    threshold: Optional[float] = None
    condition: Optional[str] = None

    def __post_init__(self):
        """Validate rule configuration."""
        if not self.name:
            raise ValueError("Rule name is required")


@dataclass
class AlertReport:
    """Complete alert report structure."""
    report_id: str
    generated_at: str
    dashboard_path: str
    previous_dashboard_path: Optional[str]

    # Alert summary
    total_alerts: int = 0
    critical_alerts: int = 0
    error_alerts: int = 0
    warning_alerts: int = 0
    info_alerts: int = 0

    # Alert list
    alerts: List[Dict[str, Any]] = field(default_factory=list)

    # Dashboard data snapshot
    repository_score: float = 0.0
    health_status: str = "unknown"
    total_issues: int = 0

    # Trend data
    previous_score: Optional[float] = None
    score_change: Optional[float] = None
    new_issues_count: int = 0

    # Dispatch status
    channels_dispatched: List[str] = field(default_factory=list)
    dispatch_errors: List[Dict[str, str]] = field(default_factory=list)

    # Metadata
    evaluation_duration_ms: float = 0.0
    rules_evaluated: int = 0
    rules_triggered: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return asdict(self)


@dataclass
class ChannelConfig:
    """Configuration for an alert channel."""
    channel_type: ChannelType
    enabled: bool = True

    # File channel config
    output_path: Optional[Path] = None

    # Email channel config
    email_to: Optional[str] = None
    email_from: Optional[str] = None
    smtp_host: Optional[str] = None
    smtp_port: int = 587

    # Webhook channel config
    webhook_url: Optional[str] = None
    webhook_headers: Dict[str, str] = field(default_factory=dict)

    # Filtering
    min_severity: AlertSeverity = AlertSeverity.INFO


@dataclass
class AlertingConfig:
    """Configuration for the alerting engine."""
    # Input
    current_dashboard_path: Path
    previous_dashboard_path: Optional[Path] = None

    # Output
    output_path: Optional[Path] = None

    # Channels
    channels: List[ChannelConfig] = field(default_factory=list)

    # Thresholds
    severity_threshold: AlertSeverity = AlertSeverity.INFO
    score_drop_threshold: float = 10.0  # Points drop to trigger alert
    rapid_regression_threshold: int = 3  # New issues to trigger regression alert

    # Behavior
    fail_on_critical: bool = True
    fail_on_any_alert: bool = False
    verbose: bool = False


# ============================================================================
# Alert Channels (Abstract Base + Implementations)
# ============================================================================

class AlertChannel(ABC):
    """Abstract base class for alert channels."""

    def __init__(self, config: ChannelConfig):
        """Initialize channel with config."""
        self.config = config

    @abstractmethod
    def dispatch(self, alerts: List[Alert], report: AlertReport) -> bool:
        """
        Dispatch alerts through this channel.

        Args:
            alerts: List of alerts to dispatch
            report: Complete alert report

        Returns:
            True if dispatch succeeded, False otherwise
        """
        pass

    def filter_alerts(self, alerts: List[Alert]) -> List[Alert]:
        """Filter alerts by minimum severity."""
        min_severity = self.config.min_severity
        return [
            alert for alert in alerts
            if AlertSeverity.from_string(alert.severity) >= min_severity
        ]


class ConsoleChannel(AlertChannel):
    """Dispatches alerts to console output."""

    def dispatch(self, alerts: List[Alert], report: AlertReport) -> bool:
        """Print alerts to console."""
        filtered = self.filter_alerts(alerts)

        if not filtered:
            print("\n[Alerting Engine] No alerts above threshold to display.")
            return True

        print("\n" + "=" * 80)
        print("REPOSITORY HEALTH ALERTS")
        print("=" * 80)
        print(f"Dashboard: {report.dashboard_path}")
        print(f"Generated: {report.generated_at}")
        print(f"Health Status: {report.health_status.upper()}")
        print(f"Repository Score: {report.repository_score:.1f}/100")
        print("-" * 80)

        # Group by severity
        by_severity: Dict[str, List[Alert]] = {}
        for alert in filtered:
            sev = alert.severity
            if sev not in by_severity:
                by_severity[sev] = []
            by_severity[sev].append(alert)

        # Print in severity order
        severity_order = ["CRITICAL", "ERROR", "WARNING", "INFO"]
        icons = {
            "CRITICAL": "",
            "ERROR": "",
            "WARNING": "",
            "INFO": ""
        }

        for severity in severity_order:
            if severity not in by_severity:
                continue

            alerts_list = by_severity[severity]
            icon = icons.get(severity, "")
            print(f"\n{icon} {severity} ({len(alerts_list)}):")
            print("-" * 40)

            for alert in alerts_list:
                print(f"  [{alert.alert_type}] {alert.title}")
                print(f"    {alert.message}")
                if alert.recommendations:
                    for rec in alert.recommendations[:2]:  # Limit to 2
                        print(f"      -> {rec}")
                print()

        print("=" * 80)
        print(f"Total Alerts: {len(filtered)}")
        print("=" * 80)

        return True


class FileChannel(AlertChannel):
    """Dispatches alerts to a file."""

    def dispatch(self, alerts: List[Alert], report: AlertReport) -> bool:
        """Write alerts to file."""
        filtered = self.filter_alerts(alerts)

        if not self.config.output_path:
            logger.error("FileChannel: No output path configured")
            return False

        try:
            output_path = Path(self.config.output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Write as text report
            lines = []
            lines.append("=" * 80)
            lines.append("REPOSITORY HEALTH ALERTS")
            lines.append("=" * 80)
            lines.append(f"Dashboard: {report.dashboard_path}")
            lines.append(f"Generated: {report.generated_at}")
            lines.append(f"Health Status: {report.health_status.upper()}")
            lines.append(f"Repository Score: {report.repository_score:.1f}/100")
            lines.append("")
            lines.append(f"Total Alerts: {len(filtered)}")
            lines.append(f"  Critical: {report.critical_alerts}")
            lines.append(f"  Error: {report.error_alerts}")
            lines.append(f"  Warning: {report.warning_alerts}")
            lines.append(f"  Info: {report.info_alerts}")
            lines.append("-" * 80)
            lines.append("")

            for alert in filtered:
                lines.append(f"[{alert.severity}] {alert.alert_type}: {alert.title}")
                lines.append(f"  Message: {alert.message}")
                lines.append(f"  Timestamp: {alert.timestamp}")
                if alert.version:
                    lines.append(f"  Version: {alert.version}")
                if alert.artifact:
                    lines.append(f"  Artifact: {alert.artifact}")
                if alert.recommendations:
                    lines.append("  Recommendations:")
                    for rec in alert.recommendations:
                        lines.append(f"    - {rec}")
                lines.append("")

            lines.append("=" * 80)

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))

            logger.info(f"FileChannel: Wrote {len(filtered)} alerts to {output_path}")
            return True

        except Exception as e:
            logger.error(f"FileChannel: Failed to write alerts: {e}")
            return False


class EmailChannel(AlertChannel):
    """Dispatches alerts via email (mock implementation)."""

    def dispatch(self, alerts: List[Alert], report: AlertReport) -> bool:
        """Generate email content (mock - does not actually send)."""
        filtered = self.filter_alerts(alerts)

        if not self.config.email_to:
            logger.error("EmailChannel: No recipient configured")
            return False

        # Generate email content
        email_body = self._generate_email_body(filtered, report)
        email_subject = self._generate_subject(report)

        # Log the generated email (mock)
        logger.info(f"EmailChannel: Generated email to {self.config.email_to}")
        logger.debug(f"  Subject: {email_subject}")
        logger.debug(f"  Body length: {len(email_body)} chars")

        # Store generated content for testing/verification
        self._last_email = {
            "to": self.config.email_to,
            "from": self.config.email_from or "alerts@tars.local",
            "subject": email_subject,
            "body": email_body,
            "html": self._generate_html_email(filtered, report)
        }

        return True

    def _generate_subject(self, report: AlertReport) -> str:
        """Generate email subject line."""
        status = report.health_status.upper()
        score = report.repository_score

        if report.critical_alerts > 0:
            return f"[CRITICAL] T.A.R.S. Repository Health Alert - {report.critical_alerts} Critical Issue(s)"
        elif report.error_alerts > 0:
            return f"[ERROR] T.A.R.S. Repository Health Alert - Score: {score:.0f}/100"
        elif report.warning_alerts > 0:
            return f"[WARNING] T.A.R.S. Repository Health Alert - Status: {status}"
        else:
            return f"[INFO] T.A.R.S. Repository Health Report - Status: {status}"

    def _generate_email_body(self, alerts: List[Alert], report: AlertReport) -> str:
        """Generate plain text email body."""
        lines = []
        lines.append("T.A.R.S. Repository Health Alert Report")
        lines.append("=" * 50)
        lines.append("")
        lines.append(f"Generated: {report.generated_at}")
        lines.append(f"Health Status: {report.health_status.upper()}")
        lines.append(f"Repository Score: {report.repository_score:.1f}/100")
        lines.append("")

        if report.score_change is not None:
            direction = "increased" if report.score_change > 0 else "decreased"
            lines.append(f"Score Change: {direction} by {abs(report.score_change):.1f} points")
            lines.append("")

        lines.append("Alert Summary:")
        lines.append(f"  Total: {len(alerts)}")
        lines.append(f"  Critical: {report.critical_alerts}")
        lines.append(f"  Error: {report.error_alerts}")
        lines.append(f"  Warning: {report.warning_alerts}")
        lines.append(f"  Info: {report.info_alerts}")
        lines.append("")
        lines.append("-" * 50)
        lines.append("")

        for alert in alerts:
            lines.append(f"[{alert.severity}] {alert.title}")
            lines.append(f"  {alert.message}")
            lines.append("")

        lines.append("-" * 50)
        lines.append("")
        lines.append("This is an automated alert from T.A.R.S. Release Pipeline.")
        lines.append("Please review the repository health dashboard for details.")

        return '\n'.join(lines)

    def _generate_html_email(self, alerts: List[Alert], report: AlertReport) -> str:
        """Generate HTML email body."""
        severity_colors = {
            "CRITICAL": "#dc3545",
            "ERROR": "#fd7e14",
            "WARNING": "#ffc107",
            "INFO": "#17a2b8"
        }

        status_colors = {
            "green": "#28a745",
            "yellow": "#ffc107",
            "red": "#dc3545"
        }

        html = f"""<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }}
        .header {{ background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); color: white; padding: 20px; }}
        .status-badge {{ display: inline-block; padding: 5px 15px; border-radius: 20px; font-weight: bold; }}
        .alert-card {{ border-left: 4px solid; padding: 10px; margin: 10px 0; background: #f8f9fa; }}
        .summary {{ display: flex; gap: 20px; margin: 20px 0; }}
        .stat {{ text-align: center; padding: 10px; background: #f8f9fa; border-radius: 8px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>T.A.R.S. Repository Health Alert</h1>
        <p>Generated: {report.generated_at}</p>
    </div>

    <div style="padding: 20px;">
        <h2>Health Status:
            <span class="status-badge" style="background-color: {status_colors.get(report.health_status, '#6c757d')};">
                {report.health_status.upper()}
            </span>
        </h2>

        <div class="summary">
            <div class="stat">
                <div style="font-size: 24px; font-weight: bold;">{report.repository_score:.0f}</div>
                <div>Score</div>
            </div>
            <div class="stat">
                <div style="font-size: 24px; font-weight: bold;">{len(alerts)}</div>
                <div>Alerts</div>
            </div>
            <div class="stat">
                <div style="font-size: 24px; font-weight: bold;">{report.total_issues}</div>
                <div>Issues</div>
            </div>
        </div>

        <h3>Alerts</h3>
"""

        for alert in alerts:
            color = severity_colors.get(alert.severity, "#6c757d")
            html += f"""
        <div class="alert-card" style="border-color: {color};">
            <strong>[{alert.severity}]</strong> {alert.title}
            <p>{alert.message}</p>
        </div>
"""

        html += """
        <hr>
        <p style="color: #6c757d; font-size: 12px;">
            This is an automated alert from T.A.R.S. Release Pipeline.
        </p>
    </div>
</body>
</html>
"""
        return html


class WebhookChannel(AlertChannel):
    """Dispatches alerts via webhook (mock implementation)."""

    def dispatch(self, alerts: List[Alert], report: AlertReport) -> bool:
        """Generate webhook payload (mock - does not actually send)."""
        filtered = self.filter_alerts(alerts)

        if not self.config.webhook_url:
            logger.error("WebhookChannel: No webhook URL configured")
            return False

        # Generate webhook payload
        payload = self._generate_payload(filtered, report)

        # Log the generated payload (mock)
        logger.info(f"WebhookChannel: Generated payload for {self.config.webhook_url}")
        logger.debug(f"  Payload size: {len(json.dumps(payload))} bytes")

        # Store generated payload for testing/verification
        self._last_payload = {
            "url": self.config.webhook_url,
            "method": "POST",
            "headers": {
                "Content-Type": "application/json",
                **self.config.webhook_headers
            },
            "payload": payload
        }

        return True

    def _generate_payload(self, alerts: List[Alert], report: AlertReport) -> Dict[str, Any]:
        """Generate webhook JSON payload."""
        return {
            "event_type": "repository_health_alert",
            "timestamp": report.generated_at,
            "source": "tars_alerting_engine",
            "dashboard": {
                "path": report.dashboard_path,
                "health_status": report.health_status,
                "repository_score": report.repository_score,
                "total_issues": report.total_issues
            },
            "trend": {
                "previous_score": report.previous_score,
                "score_change": report.score_change,
                "new_issues": report.new_issues_count
            },
            "summary": {
                "total_alerts": len(alerts),
                "critical": report.critical_alerts,
                "error": report.error_alerts,
                "warning": report.warning_alerts,
                "info": report.info_alerts
            },
            "alerts": [alert.to_dict() for alert in alerts],
            "metadata": {
                "rules_evaluated": report.rules_evaluated,
                "rules_triggered": report.rules_triggered,
                "evaluation_duration_ms": report.evaluation_duration_ms
            }
        }


# ============================================================================
# Alert Rules Engine
# ============================================================================

class AlertRulesEngine:
    """
    Evaluates alert rules against dashboard data and generates alerts.

    This engine:
    1. Loads and validates dashboard JSON
    2. Optionally compares with previous dashboard for trends
    3. Evaluates a configurable set of rules
    4. Generates typed Alert objects
    5. Produces a structured AlertReport
    """

    def __init__(self, config: AlertingConfig):
        """Initialize the rules engine."""
        self.config = config

        if config.verbose:
            logger.setLevel(logging.DEBUG)

        # Initialize default rules
        self.rules: List[AlertRule] = self._create_default_rules()

        # Data storage
        self.current_dashboard: Optional[Dict[str, Any]] = None
        self.previous_dashboard: Optional[Dict[str, Any]] = None

    def _create_default_rules(self) -> List[AlertRule]:
        """Create the default set of alert rules."""
        return [
            AlertRule(
                name="repository_status_red",
                alert_type=AlertType.REPOSITORY_STATUS,
                severity=AlertSeverity.CRITICAL,
                description="Repository health status is RED",
                condition="health_status == 'red'"
            ),
            AlertRule(
                name="repository_status_yellow",
                alert_type=AlertType.REPOSITORY_STATUS,
                severity=AlertSeverity.WARNING,
                description="Repository health status is YELLOW",
                condition="health_status == 'yellow'"
            ),
            AlertRule(
                name="critical_issues",
                alert_type=AlertType.CRITICAL_ISSUE,
                severity=AlertSeverity.CRITICAL,
                description="Critical issues detected in repository",
                condition="critical_issues > 0"
            ),
            AlertRule(
                name="missing_artifacts",
                alert_type=AlertType.MISSING_ARTIFACT,
                severity=AlertSeverity.ERROR,
                description="Missing artifacts detected",
                condition="missing_artifacts > 0"
            ),
            AlertRule(
                name="corrupted_artifacts",
                alert_type=AlertType.CORRUPTED_ARTIFACT,
                severity=AlertSeverity.CRITICAL,
                description="Corrupted artifacts detected",
                condition="corrupted_artifacts > 0"
            ),
            AlertRule(
                name="orphaned_artifacts",
                alert_type=AlertType.ORPHANED_ARTIFACT,
                severity=AlertSeverity.WARNING,
                description="Orphaned artifacts detected",
                condition="orphaned_artifacts > 0"
            ),
            AlertRule(
                name="missing_sbom",
                alert_type=AlertType.METADATA_MISSING,
                severity=AlertSeverity.WARNING,
                description="Versions missing SBOM metadata",
                condition="versions_missing_sbom > 0"
            ),
            AlertRule(
                name="missing_slsa",
                alert_type=AlertType.METADATA_MISSING,
                severity=AlertSeverity.WARNING,
                description="Versions missing SLSA provenance",
                condition="versions_missing_slsa > 0"
            ),
            AlertRule(
                name="score_drop",
                alert_type=AlertType.REPOSITORY_SCORE_DROP,
                severity=AlertSeverity.WARNING,
                description="Significant repository score drop detected",
                threshold=10.0  # Points
            ),
            AlertRule(
                name="rapid_regression",
                alert_type=AlertType.RAPID_REGRESSION,
                severity=AlertSeverity.ERROR,
                description="Rapid regression - multiple new issues",
                threshold=3.0  # New issues
            ),
            AlertRule(
                name="version_health_degradation",
                alert_type=AlertType.VERSION_HEALTH,
                severity=AlertSeverity.WARNING,
                description="Version health has degraded"
            ),
            AlertRule(
                name="rollback_failure",
                alert_type=AlertType.ROLLBACK_FAILURE,
                severity=AlertSeverity.CRITICAL,
                description="Rollback operation failed"
            )
        ]

    def load_dashboard(self, path: Path) -> Dict[str, Any]:
        """Load and validate a dashboard JSON file."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Validate required fields
            required_fields = ["overall_health", "repository_score"]
            for field in required_fields:
                if field not in data:
                    raise InvalidDashboardError(f"Missing required field: {field}")

            return data

        except json.JSONDecodeError as e:
            raise InvalidDashboardError(f"Invalid JSON in dashboard: {e}")
        except FileNotFoundError:
            raise InvalidDashboardError(f"Dashboard file not found: {path}")
        except Exception as e:
            raise InvalidDashboardError(f"Failed to load dashboard: {e}")

    def evaluate_rules(self) -> List[Alert]:
        """
        Evaluate all enabled rules and generate alerts.

        Returns:
            List of Alert objects for triggered rules
        """
        if not self.current_dashboard:
            raise RuleEvaluationError("No dashboard data loaded")

        alerts: List[Alert] = []

        for rule in self.rules:
            if not rule.enabled:
                continue

            try:
                triggered_alerts = self._evaluate_rule(rule)
                alerts.extend(triggered_alerts)
            except Exception as e:
                logger.warning(f"Failed to evaluate rule '{rule.name}': {e}")

        return alerts

    def _evaluate_rule(self, rule: AlertRule) -> List[Alert]:
        """Evaluate a single rule and return any triggered alerts."""
        alerts: List[Alert] = []
        dashboard = self.current_dashboard

        timestamp = datetime.utcnow().isoformat()

        # Repository Status Rules
        if rule.alert_type == AlertType.REPOSITORY_STATUS:
            health_status = dashboard.get("overall_health", "").lower()

            if rule.name == "repository_status_red" and health_status == "red":
                alerts.append(Alert(
                    alert_id=f"alert_{rule.name}_{timestamp}",
                    alert_type=rule.alert_type.value,
                    severity=rule.severity.value,
                    title="Repository Health CRITICAL",
                    message=f"Repository health status is RED with score {dashboard.get('repository_score', 0):.1f}/100",
                    timestamp=timestamp,
                    rule_name=rule.name,
                    current_status="red",
                    recommendations=[
                        "Review critical issues immediately",
                        "Consider rollback if integrity is compromised",
                        "Run full integrity scan with repair enabled"
                    ]
                ))

            elif rule.name == "repository_status_yellow" and health_status == "yellow":
                alerts.append(Alert(
                    alert_id=f"alert_{rule.name}_{timestamp}",
                    alert_type=rule.alert_type.value,
                    severity=rule.severity.value,
                    title="Repository Health WARNING",
                    message=f"Repository health status is YELLOW with score {dashboard.get('repository_score', 0):.1f}/100",
                    timestamp=timestamp,
                    rule_name=rule.name,
                    current_status="yellow",
                    recommendations=[
                        "Address warning-level issues before they escalate",
                        "Review issues table in health dashboard"
                    ]
                ))

        # Critical Issues Rule
        elif rule.alert_type == AlertType.CRITICAL_ISSUE:
            critical_count = dashboard.get("critical_issues", 0)
            if critical_count > 0:
                alerts.append(Alert(
                    alert_id=f"alert_{rule.name}_{timestamp}",
                    alert_type=rule.alert_type.value,
                    severity=rule.severity.value,
                    title=f"{critical_count} Critical Issue(s) Detected",
                    message=f"Repository has {critical_count} critical-severity issue(s) requiring immediate attention",
                    timestamp=timestamp,
                    rule_name=rule.name,
                    issue_count=critical_count,
                    recommendations=[
                        "Address critical issues immediately",
                        "Check integrity scan results for details",
                        "Consider pausing releases until resolved"
                    ]
                ))

        # Missing Artifacts Rule
        elif rule.alert_type == AlertType.MISSING_ARTIFACT:
            missing_count = dashboard.get("missing_artifacts", 0)
            if missing_count > 0:
                alerts.append(Alert(
                    alert_id=f"alert_{rule.name}_{timestamp}",
                    alert_type=rule.alert_type.value,
                    severity=rule.severity.value,
                    title=f"{missing_count} Missing Artifact(s)",
                    message=f"Repository is missing {missing_count} expected artifact(s)",
                    timestamp=timestamp,
                    rule_name=rule.name,
                    issue_count=missing_count,
                    recommendations=[
                        "Review integrity scan report for missing artifacts",
                        "Re-publish affected versions or restore from backup",
                        "Consider rollback if critical versions affected"
                    ]
                ))

        # Corrupted Artifacts Rule
        elif rule.alert_type == AlertType.CORRUPTED_ARTIFACT:
            corrupted_count = dashboard.get("corrupted_artifacts", 0)
            if corrupted_count > 0:
                alerts.append(Alert(
                    alert_id=f"alert_{rule.name}_{timestamp}",
                    alert_type=rule.alert_type.value,
                    severity=rule.severity.value,
                    title=f"{corrupted_count} Corrupted Artifact(s)",
                    message=f"Repository has {corrupted_count} artifact(s) with checksum mismatches",
                    timestamp=timestamp,
                    rule_name=rule.name,
                    issue_count=corrupted_count,
                    recommendations=[
                        "DO NOT use corrupted artifacts",
                        "Run integrity scan with --repair to fix",
                        "Investigate source of corruption"
                    ]
                ))

        # Orphaned Artifacts Rule
        elif rule.alert_type == AlertType.ORPHANED_ARTIFACT:
            orphaned_count = dashboard.get("orphaned_artifacts", 0)
            if orphaned_count > 0:
                alerts.append(Alert(
                    alert_id=f"alert_{rule.name}_{timestamp}",
                    alert_type=rule.alert_type.value,
                    severity=rule.severity.value,
                    title=f"{orphaned_count} Orphaned Artifact(s)",
                    message=f"Repository has {orphaned_count} artifact(s) not referenced by any manifest",
                    timestamp=timestamp,
                    rule_name=rule.name,
                    issue_count=orphaned_count,
                    recommendations=[
                        "Review orphaned artifacts for cleanup",
                        "Run integrity scan with --repair-orphans",
                        "Check for incomplete publications"
                    ]
                ))

        # Metadata Missing Rules
        elif rule.alert_type == AlertType.METADATA_MISSING:
            versions_health = dashboard.get("versions_health", [])

            if rule.name == "missing_sbom":
                missing_sbom = sum(1 for v in versions_health if not v.get("sbom_present", True))
                if missing_sbom > 0:
                    alerts.append(Alert(
                        alert_id=f"alert_{rule.name}_{timestamp}",
                        alert_type=rule.alert_type.value,
                        severity=rule.severity.value,
                        title=f"{missing_sbom} Version(s) Missing SBOM",
                        message=f"{missing_sbom} version(s) are missing Software Bill of Materials",
                        timestamp=timestamp,
                        rule_name=rule.name,
                        issue_count=missing_sbom,
                        recommendations=[
                            "Re-publish affected versions with SBOM generation enabled",
                            "SBOM is required for supply chain security compliance"
                        ]
                    ))

            elif rule.name == "missing_slsa":
                missing_slsa = sum(1 for v in versions_health if not v.get("slsa_present", True))
                if missing_slsa > 0:
                    alerts.append(Alert(
                        alert_id=f"alert_{rule.name}_{timestamp}",
                        alert_type=rule.alert_type.value,
                        severity=rule.severity.value,
                        title=f"{missing_slsa} Version(s) Missing SLSA Provenance",
                        message=f"{missing_slsa} version(s) are missing SLSA provenance attestation",
                        timestamp=timestamp,
                        rule_name=rule.name,
                        issue_count=missing_slsa,
                        recommendations=[
                            "Re-publish affected versions with SLSA generation enabled",
                            "SLSA provenance provides build integrity guarantees"
                        ]
                    ))

        # Repository Score Drop Rule (trend-based)
        elif rule.alert_type == AlertType.REPOSITORY_SCORE_DROP:
            if self.previous_dashboard:
                current_score = dashboard.get("repository_score", 0)
                previous_score = self.previous_dashboard.get("repository_score", 0)
                score_drop = previous_score - current_score

                threshold = rule.threshold or self.config.score_drop_threshold

                if score_drop >= threshold:
                    alerts.append(Alert(
                        alert_id=f"alert_{rule.name}_{timestamp}",
                        alert_type=rule.alert_type.value,
                        severity=rule.severity.value,
                        title=f"Repository Score Dropped by {score_drop:.1f} Points",
                        message=f"Score dropped from {previous_score:.1f} to {current_score:.1f} (threshold: {threshold:.1f})",
                        timestamp=timestamp,
                        rule_name=rule.name,
                        score_change=-score_drop,
                        previous_status=self.previous_dashboard.get("overall_health"),
                        current_status=dashboard.get("overall_health"),
                        recommendations=[
                            "Investigate recent changes causing score drop",
                            "Review new issues in dashboard",
                            "Compare with previous dashboard for changes"
                        ]
                    ))

        # Rapid Regression Rule (trend-based)
        elif rule.alert_type == AlertType.RAPID_REGRESSION:
            if self.previous_dashboard:
                current_issues = dashboard.get("total_issues", 0)
                previous_issues = self.previous_dashboard.get("total_issues", 0)
                new_issues = current_issues - previous_issues

                threshold = int(rule.threshold or self.config.rapid_regression_threshold)

                if new_issues >= threshold:
                    alerts.append(Alert(
                        alert_id=f"alert_{rule.name}_{timestamp}",
                        alert_type=rule.alert_type.value,
                        severity=rule.severity.value,
                        title=f"Rapid Regression: {new_issues} New Issues",
                        message=f"Issues increased from {previous_issues} to {current_issues} ({new_issues} new)",
                        timestamp=timestamp,
                        rule_name=rule.name,
                        issue_count=new_issues,
                        recommendations=[
                            "Investigate root cause of rapid issue increase",
                            "Review recent publications and changes",
                            "Consider pausing releases until stabilized"
                        ]
                    ))

        # Version Health Degradation Rule (trend-based)
        elif rule.alert_type == AlertType.VERSION_HEALTH:
            if self.previous_dashboard:
                current_versions = {v["version"]: v for v in dashboard.get("versions_health", [])}
                previous_versions = {v["version"]: v for v in self.previous_dashboard.get("versions_health", [])}

                for version, current in current_versions.items():
                    if version in previous_versions:
                        previous = previous_versions[version]

                        # Check for degradation
                        health_order = {"green": 0, "yellow": 1, "red": 2}
                        current_health = current.get("health_status", "green")
                        previous_health = previous.get("health_status", "green")

                        if health_order.get(current_health, 0) > health_order.get(previous_health, 0):
                            alerts.append(Alert(
                                alert_id=f"alert_{rule.name}_{version}_{timestamp}",
                                alert_type=rule.alert_type.value,
                                severity=rule.severity.value,
                                title=f"Version {version} Health Degraded",
                                message=f"Version {version} health changed from {previous_health.upper()} to {current_health.upper()}",
                                timestamp=timestamp,
                                rule_name=rule.name,
                                version=version,
                                previous_status=previous_health,
                                current_status=current_health,
                                recommendations=[
                                    f"Review issues for version {version}",
                                    "Consider rollback if critical functionality affected"
                                ]
                            ))

        # Rollback Failure Rule
        elif rule.alert_type == AlertType.ROLLBACK_FAILURE:
            rollback_history = dashboard.get("rollback_history", [])
            failed_rollbacks = [r for r in rollback_history if r.get("status") == "failed"]

            if failed_rollbacks:
                for rollback in failed_rollbacks:
                    alerts.append(Alert(
                        alert_id=f"alert_{rule.name}_{rollback.get('from_version')}_{timestamp}",
                        alert_type=rule.alert_type.value,
                        severity=rule.severity.value,
                        title=f"Rollback Failed: {rollback.get('from_version')} -> {rollback.get('to_version')}",
                        message=f"Rollback from version {rollback.get('from_version')} to {rollback.get('to_version')} failed",
                        timestamp=timestamp,
                        rule_name=rule.name,
                        version=rollback.get("from_version"),
                        recommendations=[
                            "Investigate rollback failure cause",
                            "Check rollback report for details",
                            "Manual intervention may be required"
                        ]
                    ))

        return alerts

    def generate_report(self, alerts: List[Alert]) -> AlertReport:
        """Generate a complete alert report."""
        timestamp = datetime.utcnow().isoformat()

        # Calculate summary
        critical = sum(1 for a in alerts if a.severity == "CRITICAL")
        error = sum(1 for a in alerts if a.severity == "ERROR")
        warning = sum(1 for a in alerts if a.severity == "WARNING")
        info = sum(1 for a in alerts if a.severity == "INFO")

        # Trend data
        previous_score = None
        score_change = None
        new_issues = 0

        if self.previous_dashboard:
            previous_score = self.previous_dashboard.get("repository_score")
            if previous_score is not None:
                current_score = self.current_dashboard.get("repository_score", 0)
                score_change = current_score - previous_score

            current_issues = self.current_dashboard.get("total_issues", 0)
            prev_issues = self.previous_dashboard.get("total_issues", 0)
            new_issues = max(0, current_issues - prev_issues)

        return AlertReport(
            report_id=f"alert_report_{timestamp}",
            generated_at=timestamp,
            dashboard_path=str(self.config.current_dashboard_path),
            previous_dashboard_path=str(self.config.previous_dashboard_path) if self.config.previous_dashboard_path else None,
            total_alerts=len(alerts),
            critical_alerts=critical,
            error_alerts=error,
            warning_alerts=warning,
            info_alerts=info,
            alerts=[a.to_dict() for a in alerts],
            repository_score=self.current_dashboard.get("repository_score", 0),
            health_status=self.current_dashboard.get("overall_health", "unknown"),
            total_issues=self.current_dashboard.get("total_issues", 0),
            previous_score=previous_score,
            score_change=score_change,
            new_issues_count=new_issues,
            rules_evaluated=len([r for r in self.rules if r.enabled]),
            rules_triggered=len(set(a.rule_name for a in alerts if a.rule_name))
        )


# ============================================================================
# Alert Dispatcher
# ============================================================================

class AlertDispatcher:
    """
    Central dispatcher for sending alerts to configured channels.

    Handles:
    - Channel initialization
    - Alert filtering by severity
    - Dispatch to multiple channels
    - Error handling and fallback
    """

    def __init__(self, config: AlertingConfig):
        """Initialize dispatcher with configuration."""
        self.config = config
        self.channels: List[AlertChannel] = []

        # Initialize configured channels
        for channel_config in config.channels:
            if channel_config.enabled:
                channel = self._create_channel(channel_config)
                if channel:
                    self.channels.append(channel)

    def _create_channel(self, config: ChannelConfig) -> Optional[AlertChannel]:
        """Create a channel instance from config."""
        try:
            if config.channel_type == ChannelType.CONSOLE:
                return ConsoleChannel(config)
            elif config.channel_type == ChannelType.FILE:
                return FileChannel(config)
            elif config.channel_type == ChannelType.EMAIL:
                return EmailChannel(config)
            elif config.channel_type == ChannelType.WEBHOOK:
                return WebhookChannel(config)
            else:
                logger.warning(f"Unknown channel type: {config.channel_type}")
                return None
        except Exception as e:
            logger.error(f"Failed to create channel {config.channel_type}: {e}")
            return None

    def dispatch(self, alerts: List[Alert], report: AlertReport) -> AlertReport:
        """
        Dispatch alerts to all configured channels.

        Args:
            alerts: List of alerts to dispatch
            report: Alert report for context

        Returns:
            Updated report with dispatch status
        """
        # Filter by severity threshold
        threshold = self.config.severity_threshold
        filtered_alerts = [
            a for a in alerts
            if AlertSeverity.from_string(a.severity) >= threshold
        ]

        if not filtered_alerts:
            logger.info("No alerts above severity threshold to dispatch")

        dispatched = []
        errors = []

        for channel in self.channels:
            channel_name = channel.config.channel_type.value
            try:
                success = channel.dispatch(filtered_alerts, report)
                if success:
                    dispatched.append(channel_name)
                    logger.info(f"Successfully dispatched to {channel_name}")
                else:
                    errors.append({
                        "channel": channel_name,
                        "error": "Dispatch returned failure"
                    })
                    logger.warning(f"Dispatch to {channel_name} returned failure")
            except Exception as e:
                errors.append({
                    "channel": channel_name,
                    "error": str(e)
                })
                logger.error(f"Failed to dispatch to {channel_name}: {e}")

        # Update report
        report.channels_dispatched = dispatched
        report.dispatch_errors = errors

        return report


# ============================================================================
# Main Alerting Engine
# ============================================================================

class AlertingEngine:
    """
    Main orchestrator for the Repository Health Alerting Engine.

    This class:
    1. Loads dashboard data
    2. Evaluates alert rules
    3. Generates alerts
    4. Dispatches to channels
    5. Writes alert report
    6. Returns appropriate exit code
    """

    def __init__(self, config: AlertingConfig):
        """Initialize the alerting engine."""
        self.config = config

        if config.verbose:
            logging.basicConfig(
                level=logging.DEBUG,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            logger.setLevel(logging.DEBUG)
        else:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s'
            )

        # Initialize components
        self.rules_engine = AlertRulesEngine(config)
        self.dispatcher = AlertDispatcher(config)

    def run(self) -> tuple[AlertReport, int]:
        """
        Run the complete alerting pipeline.

        Returns:
            Tuple of (AlertReport, exit_code)
        """
        start_time = datetime.utcnow()

        try:
            logger.info("=" * 80)
            logger.info("REPOSITORY HEALTH ALERTING ENGINE")
            logger.info("=" * 80)

            # Step 1: Load dashboards
            logger.info("Step 1: Loading dashboard data...")
            self.rules_engine.current_dashboard = self.rules_engine.load_dashboard(
                self.config.current_dashboard_path
            )
            logger.info(f"  Loaded current dashboard: {self.config.current_dashboard_path}")

            if self.config.previous_dashboard_path and self.config.previous_dashboard_path.exists():
                self.rules_engine.previous_dashboard = self.rules_engine.load_dashboard(
                    self.config.previous_dashboard_path
                )
                logger.info(f"  Loaded previous dashboard: {self.config.previous_dashboard_path}")
            else:
                logger.info("  No previous dashboard loaded (trend alerts disabled)")

            # Step 2: Evaluate rules
            logger.info("\nStep 2: Evaluating alert rules...")
            alerts = self.rules_engine.evaluate_rules()
            logger.info(f"  Evaluated {len(self.rules_engine.rules)} rules")
            logger.info(f"  Generated {len(alerts)} alert(s)")

            # Step 3: Generate report
            logger.info("\nStep 3: Generating alert report...")
            report = self.rules_engine.generate_report(alerts)

            # Calculate duration
            duration = (datetime.utcnow() - start_time).total_seconds() * 1000
            report.evaluation_duration_ms = duration

            logger.info(f"  Critical: {report.critical_alerts}")
            logger.info(f"  Error: {report.error_alerts}")
            logger.info(f"  Warning: {report.warning_alerts}")
            logger.info(f"  Info: {report.info_alerts}")

            # Step 4: Dispatch alerts
            logger.info("\nStep 4: Dispatching alerts...")
            report = self.dispatcher.dispatch(alerts, report)
            logger.info(f"  Dispatched to {len(report.channels_dispatched)} channel(s)")

            # Step 5: Write output
            if self.config.output_path:
                logger.info("\nStep 5: Writing alert report...")
                self._write_report(report)
                logger.info(f"  Wrote report to {self.config.output_path}")

            # Determine exit code
            exit_code = self._determine_exit_code(report)

            logger.info("\n" + "=" * 80)
            logger.info("ALERTING COMPLETE")
            logger.info("=" * 80)
            logger.info(f"Total Alerts: {report.total_alerts}")
            logger.info(f"Exit Code: {exit_code}")
            logger.info("=" * 80)

            return report, exit_code

        except InvalidDashboardError as e:
            logger.error(f"Invalid dashboard: {e}")
            return self._error_report(str(e)), EXIT_INVALID_DASHBOARD
        except RuleEvaluationError as e:
            logger.error(f"Rule evaluation failed: {e}")
            return self._error_report(str(e)), EXIT_RULE_EVALUATION_FAILURE
        except Exception as e:
            logger.error(f"Alerting engine error: {e}")
            return self._error_report(str(e)), EXIT_GENERAL_ALERTING_ERROR

    def _write_report(self, report: AlertReport) -> None:
        """Write alert report to JSON file."""
        try:
            output_path = Path(self.config.output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report.to_dict(), f, indent=2, default=str)

        except Exception as e:
            raise AlertWriteError(f"Failed to write alert report: {e}")

    def _determine_exit_code(self, report: AlertReport) -> int:
        """Determine appropriate exit code based on report and config."""
        # Check for dispatch errors
        if report.dispatch_errors and len(report.dispatch_errors) == len(self.dispatcher.channels):
            return EXIT_CHANNEL_DISPATCH_FAILURE

        # Check for critical alerts
        if report.critical_alerts > 0 and self.config.fail_on_critical:
            return EXIT_CRITICAL_ALERTS

        # Check for any alerts
        if report.total_alerts > 0 and self.config.fail_on_any_alert:
            return EXIT_ALERTS_TRIGGERED

        # Check for non-critical alerts
        if report.total_alerts > 0:
            return EXIT_ALERTS_TRIGGERED

        return EXIT_NO_ALERTS

    def _error_report(self, error_message: str) -> AlertReport:
        """Generate an error report."""
        return AlertReport(
            report_id=f"alert_report_error_{datetime.utcnow().isoformat()}",
            generated_at=datetime.utcnow().isoformat(),
            dashboard_path=str(self.config.current_dashboard_path),
            previous_dashboard_path=None,
            dispatch_errors=[{"channel": "engine", "error": error_message}]
        )


# ============================================================================
# Utility Functions
# ============================================================================

def create_default_channels(
    output_dir: Optional[Path] = None,
    email_to: Optional[str] = None,
    webhook_url: Optional[str] = None
) -> List[ChannelConfig]:
    """Create default channel configurations."""
    channels = [
        ChannelConfig(channel_type=ChannelType.CONSOLE, enabled=True)
    ]

    if output_dir:
        channels.append(ChannelConfig(
            channel_type=ChannelType.FILE,
            enabled=True,
            output_path=output_dir / "alerts.txt"
        ))

    if email_to:
        channels.append(ChannelConfig(
            channel_type=ChannelType.EMAIL,
            enabled=True,
            email_to=email_to
        ))

    if webhook_url:
        channels.append(ChannelConfig(
            channel_type=ChannelType.WEBHOOK,
            enabled=True,
            webhook_url=webhook_url
        ))

    return channels


def parse_channels(channel_str: str) -> List[ChannelType]:
    """Parse comma-separated channel string."""
    channels = []
    for name in channel_str.split(','):
        name = name.strip().lower()
        try:
            channels.append(ChannelType(name))
        except ValueError:
            logger.warning(f"Unknown channel type: {name}")
    return channels
