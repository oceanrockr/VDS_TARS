"""
Organization-Level Alerting & Escalation Engine

This module implements an org-level alerting layer that:
1. Generates alerts from SLO violations, risk tiers, trend degradation, and integrity issues
2. Routes alerts to multiple channels (console, JSON, stdout, email, slack/webhook stubs)
3. Supports configurable escalation rules with action dispatch
4. Integrates with CI/CD pipelines via exit codes (100-109)

Alert Sources:
- SLO Violations: From org-health-report.json SLO evaluations
- High-Risk Repositories: Repos at HIGH or CRITICAL risk tier
- Org-Wide Trend Signals: Declining trends, low green %, score drops
- Integrity Issues: Load errors, missing/corrupted artifacts

Exit Codes (100-109):
- 100: Success, no alerts
- 101: Alerts present (non-critical)
- 102: Critical alerts present
- 103: Config error
- 104: Unable to parse org-health-report.json
- 105: Routing failure
- 199: General alerting error

Version: 1.0.0
Phase: 14.8 Task 2
"""

import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set, Callable, Union

logger = logging.getLogger(__name__)


# ============================================================================
# Exit Codes (100-109)
# ============================================================================

EXIT_ORG_ALERT_SUCCESS = 100
EXIT_ALERTS_PRESENT = 101
EXIT_CRITICAL_ALERTS = 102
EXIT_ALERTING_CONFIG_ERROR = 103
EXIT_ORG_REPORT_PARSE_ERROR = 104
EXIT_ROUTING_FAILURE = 105
EXIT_GENERAL_ALERTING_ERROR = 199


# ============================================================================
# Custom Exceptions
# ============================================================================

class OrgAlertingError(Exception):
    """Base exception for org alerting engine errors."""
    exit_code = EXIT_GENERAL_ALERTING_ERROR


class OrgAlertConfigError(OrgAlertingError):
    """Configuration error in alerting engine."""
    exit_code = EXIT_ALERTING_CONFIG_ERROR


class OrgReportParseError(OrgAlertingError):
    """Failed to parse org health report."""
    exit_code = EXIT_ORG_REPORT_PARSE_ERROR


class OrgRoutingError(OrgAlertingError):
    """Failed to route alerts to channels."""
    exit_code = EXIT_ROUTING_FAILURE


# ============================================================================
# Enums
# ============================================================================

class OrgAlertCategory(Enum):
    """Categories of org-level alerts."""
    SLO = "slo"                   # SLO/SLA policy violations
    RISK = "risk"                 # High-risk repository alerts
    TREND = "trend"               # Org-wide trend signals
    INTEGRITY = "integrity"       # Missing/corrupted artifacts, load errors
    CONFIG = "config"             # Configuration warnings
    UNKNOWN = "unknown"


class AlertSeverity(Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    def __lt__(self, other: "AlertSeverity") -> bool:
        """Compare severity levels."""
        order = {self.LOW: 0, self.MEDIUM: 1, self.HIGH: 2, self.CRITICAL: 3}
        return order[self] < order[other]

    def __le__(self, other: "AlertSeverity") -> bool:
        order = {self.LOW: 0, self.MEDIUM: 1, self.HIGH: 2, self.CRITICAL: 3}
        return order[self] <= order[other]

    def __gt__(self, other: "AlertSeverity") -> bool:
        order = {self.LOW: 0, self.MEDIUM: 1, self.HIGH: 2, self.CRITICAL: 3}
        return order[self] > order[other]

    def __ge__(self, other: "AlertSeverity") -> bool:
        order = {self.LOW: 0, self.MEDIUM: 1, self.HIGH: 2, self.CRITICAL: 3}
        return order[self] >= order[other]

    @classmethod
    def from_string(cls, value: str) -> "AlertSeverity":
        """Convert string to AlertSeverity."""
        value_lower = value.lower() if value else "low"
        try:
            return cls(value_lower)
        except ValueError:
            return cls.LOW


class OrgAlertChannelType(Enum):
    """Types of org alert dispatch channels."""
    CONSOLE = "console"
    JSON_FILE = "json_file"
    STDOUT = "stdout"
    EMAIL = "email"           # Placeholder stub
    SLACK = "slack"           # Placeholder stub
    WEBHOOK = "webhook"       # Placeholder stub


class EscalationActionType(Enum):
    """Types of escalation actions."""
    ESCALATE_TO = "escalate_to"     # Escalate to on-call/team
    NOTIFY = "notify"               # Send notification
    LOG = "log"                     # Log to system
    SUPPRESS = "suppress"           # Suppress alert
    CUSTOM = "custom"               # Custom action


# ============================================================================
# Data Classes - Alert
# ============================================================================

@dataclass
class OrgAlert:
    """
    Represents a single organization-level alert.

    Contains all information about the alert including:
    - Unique identifier and timestamp
    - Category and severity
    - Affected repositories and context
    - Recommendations for resolution
    """
    alert_id: str
    category: OrgAlertCategory
    severity: AlertSeverity
    title: str
    message: str
    timestamp: str

    # Source information
    source: str = "org_alerting_engine"
    source_type: str = ""  # "slo", "risk", "trend", "integrity"

    # SLO-specific fields
    slo_id: Optional[str] = None
    slo_description: Optional[str] = None
    current_value: Optional[float] = None
    target_value: Optional[float] = None

    # Risk-specific fields
    risk_tier: Optional[str] = None
    risk_score: Optional[float] = None
    reason_codes: List[str] = field(default_factory=list)

    # Trend-specific fields
    metric_name: Optional[str] = None
    metric_value: Optional[float] = None
    threshold: Optional[float] = None

    # Affected repositories
    affected_repos: List[str] = field(default_factory=list)
    violating_repos: List[str] = field(default_factory=list)

    # Additional context
    details: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)

    # Escalation tracking
    escalated: bool = False
    escalation_actions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            "alert_id": self.alert_id,
            "category": self.category.value,
            "severity": self.severity.value,
            "title": self.title,
            "message": self.message,
            "timestamp": self.timestamp,
            "source": self.source,
            "source_type": self.source_type,
            "slo_id": self.slo_id,
            "slo_description": self.slo_description,
            "current_value": self.current_value,
            "target_value": self.target_value,
            "risk_tier": self.risk_tier,
            "risk_score": self.risk_score,
            "reason_codes": self.reason_codes,
            "metric_name": self.metric_name,
            "metric_value": self.metric_value,
            "threshold": self.threshold,
            "affected_repos": self.affected_repos,
            "violating_repos": self.violating_repos,
            "details": self.details,
            "recommendations": self.recommendations,
            "escalated": self.escalated,
            "escalation_actions": self.escalation_actions
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OrgAlert":
        """Create OrgAlert from dictionary."""
        return cls(
            alert_id=data.get("alert_id", ""),
            category=OrgAlertCategory(data.get("category", "unknown")),
            severity=AlertSeverity.from_string(data.get("severity", "low")),
            title=data.get("title", ""),
            message=data.get("message", ""),
            timestamp=data.get("timestamp", ""),
            source=data.get("source", "org_alerting_engine"),
            source_type=data.get("source_type", ""),
            slo_id=data.get("slo_id"),
            slo_description=data.get("slo_description"),
            current_value=data.get("current_value"),
            target_value=data.get("target_value"),
            risk_tier=data.get("risk_tier"),
            risk_score=data.get("risk_score"),
            reason_codes=data.get("reason_codes", []),
            metric_name=data.get("metric_name"),
            metric_value=data.get("metric_value"),
            threshold=data.get("threshold"),
            affected_repos=data.get("affected_repos", []),
            violating_repos=data.get("violating_repos", []),
            details=data.get("details", {}),
            recommendations=data.get("recommendations", []),
            escalated=data.get("escalated", False),
            escalation_actions=data.get("escalation_actions", [])
        )


# ============================================================================
# Data Classes - Escalation Rules
# ============================================================================

@dataclass
class EscalationCondition:
    """
    Defines when an escalation rule should trigger.

    Supports:
    - Category matching (SLO, RISK, TREND, INTEGRITY)
    - Severity matching (LOW, MEDIUM, HIGH, CRITICAL)
    - Metric-based conditions (metric >= threshold)
    """
    alert_category: Optional[OrgAlertCategory] = None
    severity: Optional[AlertSeverity] = None
    metric: Optional[str] = None
    operator: Optional[str] = None  # "==", "!=", "<", "<=", ">", ">="
    value: Optional[float] = None

    def matches(self, alert: OrgAlert, org_metrics: Dict[str, Any] = None) -> bool:
        """
        Check if an alert matches this condition.

        Args:
            alert: Alert to check
            org_metrics: Optional org-level metrics for metric conditions

        Returns:
            True if condition matches
        """
        # Check category match
        if self.alert_category is not None:
            if alert.category != self.alert_category:
                return False

        # Check severity match
        if self.severity is not None:
            if alert.severity != self.severity:
                return False

        # Check metric condition
        if self.metric and self.operator and self.value is not None:
            if org_metrics is None:
                return False

            metric_value = org_metrics.get(self.metric)
            if metric_value is None:
                return False

            return self._check_operator(metric_value, self.value)

        return True

    def _check_operator(self, current: float, target: float) -> bool:
        """Check operator condition."""
        if self.operator == "==":
            return current == target
        elif self.operator == "!=":
            return current != target
        elif self.operator == "<":
            return current < target
        elif self.operator == "<=":
            return current <= target
        elif self.operator == ">":
            return current > target
        elif self.operator == ">=":
            return current >= target
        return False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "alert_category": self.alert_category.value if self.alert_category else None,
            "severity": self.severity.value if self.severity else None,
            "metric": self.metric,
            "operator": self.operator,
            "value": self.value
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EscalationCondition":
        """Create from dictionary."""
        category = None
        if data.get("alert_category"):
            category = OrgAlertCategory(data["alert_category"])

        severity = None
        if data.get("severity"):
            severity = AlertSeverity.from_string(data["severity"])

        return cls(
            alert_category=category,
            severity=severity,
            metric=data.get("metric"),
            operator=data.get("operator"),
            value=data.get("value")
        )


@dataclass
class EscalationAction:
    """
    Defines an action to take when escalation rule matches.

    Actions:
    - escalate_to:oncall - Escalate to on-call team
    - notify:slack:channel - Send Slack notification
    - notify:email:recipient - Send email notification
    - log - Log the escalation
    """
    action_type: EscalationActionType
    target: Optional[str] = None       # Target for escalation (e.g., "oncall", "leadership")
    channel: Optional[str] = None      # Channel for notification (e.g., "slack", "email")
    recipient: Optional[str] = None    # Specific recipient (e.g., "org-slo-critical", "team@email")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "action_type": self.action_type.value,
            "target": self.target,
            "channel": self.channel,
            "recipient": self.recipient
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EscalationAction":
        """Create from dictionary."""
        return cls(
            action_type=EscalationActionType(data.get("action_type", "log")),
            target=data.get("target"),
            channel=data.get("channel"),
            recipient=data.get("recipient")
        )

    @classmethod
    def from_string(cls, action_str: str) -> "EscalationAction":
        """
        Parse action from string format.

        Formats:
        - "escalate_to:oncall"
        - "notify:slack:org-slo-critical"
        - "notify:email:leadership"
        - "log"
        """
        parts = action_str.split(":")

        if parts[0] == "escalate_to":
            return cls(
                action_type=EscalationActionType.ESCALATE_TO,
                target=parts[1] if len(parts) > 1 else None
            )
        elif parts[0] == "notify":
            return cls(
                action_type=EscalationActionType.NOTIFY,
                channel=parts[1] if len(parts) > 1 else None,
                recipient=parts[2] if len(parts) > 2 else None
            )
        elif parts[0] == "log":
            return cls(action_type=EscalationActionType.LOG)
        elif parts[0] == "suppress":
            return cls(action_type=EscalationActionType.SUPPRESS)
        else:
            return cls(action_type=EscalationActionType.CUSTOM, target=action_str)


@dataclass
class EscalationRule:
    """
    Defines an escalation rule with conditions and actions.

    Example:
        rule = EscalationRule(
            id="slo-critical",
            description="Escalate critical SLO violations",
            condition=EscalationCondition(
                alert_category=OrgAlertCategory.SLO,
                severity=AlertSeverity.CRITICAL
            ),
            actions=[
                EscalationAction.from_string("escalate_to:oncall"),
                EscalationAction.from_string("notify:slack:org-slo-critical")
            ]
        )
    """
    id: str
    description: str
    condition: EscalationCondition
    actions: List[EscalationAction]
    enabled: bool = True
    priority: int = 0  # Higher priority rules evaluated first

    def matches(self, alert: OrgAlert, org_metrics: Dict[str, Any] = None) -> bool:
        """Check if rule matches an alert."""
        if not self.enabled:
            return False
        return self.condition.matches(alert, org_metrics)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "description": self.description,
            "condition": self.condition.to_dict(),
            "actions": [a.to_dict() for a in self.actions],
            "enabled": self.enabled,
            "priority": self.priority
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EscalationRule":
        """Create from dictionary."""
        # Parse condition from "when" block
        when_data = data.get("when", data.get("condition", {}))
        condition = EscalationCondition.from_dict(when_data)

        # Parse actions
        actions_data = data.get("actions", [])
        actions = []
        for action_data in actions_data:
            if isinstance(action_data, str):
                actions.append(EscalationAction.from_string(action_data))
            else:
                actions.append(EscalationAction.from_dict(action_data))

        return cls(
            id=data.get("id", ""),
            description=data.get("description", ""),
            condition=condition,
            actions=actions,
            enabled=data.get("enabled", True),
            priority=data.get("priority", 0)
        )


# ============================================================================
# Data Classes - Routing Configuration
# ============================================================================

@dataclass
class OrgAlertChannelConfig:
    """
    Configuration for an alert routing channel.
    """
    channel_type: OrgAlertChannelType
    enabled: bool = True

    # File channel config
    output_path: Optional[Path] = None

    # Email channel config (stub)
    email_to: Optional[str] = None
    email_from: Optional[str] = None
    smtp_host: Optional[str] = None
    smtp_port: int = 587

    # Slack channel config (stub)
    slack_webhook_url: Optional[str] = None
    slack_channel: Optional[str] = None

    # Webhook channel config (stub)
    webhook_url: Optional[str] = None
    webhook_headers: Dict[str, str] = field(default_factory=dict)

    # Filtering
    min_severity: AlertSeverity = AlertSeverity.LOW
    categories: Optional[List[OrgAlertCategory]] = None  # None = all categories

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "channel_type": self.channel_type.value,
            "enabled": self.enabled,
            "output_path": str(self.output_path) if self.output_path else None,
            "email_to": self.email_to,
            "email_from": self.email_from,
            "smtp_host": self.smtp_host,
            "smtp_port": self.smtp_port,
            "slack_webhook_url": self.slack_webhook_url,
            "slack_channel": self.slack_channel,
            "webhook_url": self.webhook_url,
            "webhook_headers": self.webhook_headers,
            "min_severity": self.min_severity.value,
            "categories": [c.value for c in self.categories] if self.categories else None
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OrgAlertChannelConfig":
        """Create from dictionary."""
        categories = None
        if data.get("categories"):
            categories = [OrgAlertCategory(c) for c in data["categories"]]

        return cls(
            channel_type=OrgAlertChannelType(data.get("channel_type", "console")),
            enabled=data.get("enabled", True),
            output_path=Path(data["output_path"]) if data.get("output_path") else None,
            email_to=data.get("email_to"),
            email_from=data.get("email_from"),
            smtp_host=data.get("smtp_host"),
            smtp_port=data.get("smtp_port", 587),
            slack_webhook_url=data.get("slack_webhook_url"),
            slack_channel=data.get("slack_channel"),
            webhook_url=data.get("webhook_url"),
            webhook_headers=data.get("webhook_headers", {}),
            min_severity=AlertSeverity.from_string(data.get("min_severity", "low")),
            categories=categories
        )


# ============================================================================
# Data Classes - Configuration
# ============================================================================

@dataclass
class OrgAlertThresholds:
    """
    Configurable thresholds for trend-based alert generation.
    """
    # Declining trend thresholds
    percent_declining_warning: float = 0.20     # >= 20% declining triggers warning
    percent_declining_critical: float = 0.40    # >= 40% declining triggers critical

    # Green percentage thresholds
    percent_green_warning: float = 0.60         # < 60% green triggers warning
    percent_green_critical: float = 0.40        # < 40% green triggers critical

    # Average score thresholds
    avg_score_warning: float = 70.0             # < 70 avg triggers warning
    avg_score_critical: float = 50.0            # < 50 avg triggers critical

    # Volatility thresholds (future use)
    high_volatility_threshold: float = 15.0     # Score volatility > 15 triggers alert

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OrgAlertThresholds":
        """Create from dictionary."""
        return cls(
            percent_declining_warning=data.get("percent_declining_warning", 0.20),
            percent_declining_critical=data.get("percent_declining_critical", 0.40),
            percent_green_warning=data.get("percent_green_warning", 0.60),
            percent_green_critical=data.get("percent_green_critical", 0.40),
            avg_score_warning=data.get("avg_score_warning", 70.0),
            avg_score_critical=data.get("avg_score_critical", 50.0),
            high_volatility_threshold=data.get("high_volatility_threshold", 15.0)
        )


@dataclass
class OrgAlertConfig:
    """
    Configuration for the Org-Level Alerting & Escalation Engine.
    """
    # Input
    org_report_path: Path

    # Output
    output_path: Optional[Path] = None

    # Routing channels
    channels: List[OrgAlertChannelConfig] = field(default_factory=list)

    # Escalation rules
    escalation_rules: List[EscalationRule] = field(default_factory=list)

    # Thresholds for trend-based alerts
    thresholds: OrgAlertThresholds = field(default_factory=OrgAlertThresholds)

    # Behavior flags
    fail_on_critical: bool = False
    fail_on_any_alerts: bool = False
    verbose: bool = False

    # Alert generation flags
    generate_slo_alerts: bool = True
    generate_risk_alerts: bool = True
    generate_trend_alerts: bool = True
    generate_integrity_alerts: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "org_report_path": str(self.org_report_path),
            "output_path": str(self.output_path) if self.output_path else None,
            "channels": [c.to_dict() for c in self.channels],
            "escalation_rules": [r.to_dict() for r in self.escalation_rules],
            "thresholds": self.thresholds.to_dict(),
            "fail_on_critical": self.fail_on_critical,
            "fail_on_any_alerts": self.fail_on_any_alerts,
            "verbose": self.verbose,
            "generate_slo_alerts": self.generate_slo_alerts,
            "generate_risk_alerts": self.generate_risk_alerts,
            "generate_trend_alerts": self.generate_trend_alerts,
            "generate_integrity_alerts": self.generate_integrity_alerts
        }

    @classmethod
    def from_yaml_file(cls, yaml_path: Path, org_report_path: Path) -> "OrgAlertConfig":
        """Load configuration from YAML file."""
        try:
            import yaml
        except ImportError:
            raise OrgAlertConfigError("PyYAML not installed. Install with: pip install pyyaml")

        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)

            return cls._from_config_dict(data, org_report_path)
        except Exception as e:
            raise OrgAlertConfigError(f"Failed to load config from {yaml_path}: {e}")

    @classmethod
    def from_json_file(cls, json_path: Path, org_report_path: Path) -> "OrgAlertConfig":
        """Load configuration from JSON file."""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            return cls._from_config_dict(data, org_report_path)
        except Exception as e:
            raise OrgAlertConfigError(f"Failed to load config from {json_path}: {e}")

    @classmethod
    def _from_config_dict(cls, data: Dict[str, Any], org_report_path: Path) -> "OrgAlertConfig":
        """Create config from dictionary."""
        # Parse channels
        channels = []
        for channel_data in data.get("channels", []):
            channels.append(OrgAlertChannelConfig.from_dict(channel_data))

        # Parse escalation rules
        escalation_rules = []
        for rule_data in data.get("escalation_rules", []):
            escalation_rules.append(EscalationRule.from_dict(rule_data))

        # Parse thresholds
        thresholds = OrgAlertThresholds.from_dict(data.get("thresholds", {}))

        return cls(
            org_report_path=org_report_path,
            output_path=Path(data["output_path"]) if data.get("output_path") else None,
            channels=channels,
            escalation_rules=escalation_rules,
            thresholds=thresholds,
            fail_on_critical=data.get("fail_on_critical", False),
            fail_on_any_alerts=data.get("fail_on_any_alerts", False),
            verbose=data.get("verbose", False),
            generate_slo_alerts=data.get("generate_slo_alerts", True),
            generate_risk_alerts=data.get("generate_risk_alerts", True),
            generate_trend_alerts=data.get("generate_trend_alerts", True),
            generate_integrity_alerts=data.get("generate_integrity_alerts", True)
        )


# ============================================================================
# Data Classes - Report
# ============================================================================

@dataclass
class OrgAlertReport:
    """
    Complete organization alert report.

    Contains all generated alerts, routing status, and summary statistics.
    """
    report_id: str
    generated_at: str
    org_report_path: str

    # Summary statistics
    total_alerts: int = 0
    critical_alerts: int = 0
    high_alerts: int = 0
    medium_alerts: int = 0
    low_alerts: int = 0

    # Category breakdown
    slo_alerts: int = 0
    risk_alerts: int = 0
    trend_alerts: int = 0
    integrity_alerts: int = 0

    # Alert list
    alerts: List[Dict[str, Any]] = field(default_factory=list)

    # Escalation summary
    escalations_triggered: int = 0
    escalation_actions: List[Dict[str, Any]] = field(default_factory=list)

    # Routing status
    channels_dispatched: List[str] = field(default_factory=list)
    dispatch_errors: List[Dict[str, str]] = field(default_factory=list)

    # Source org health summary
    org_health_status: str = "unknown"
    org_health_score: float = 0.0
    org_risk_tier: str = "unknown"
    slos_violated: int = 0
    total_repos: int = 0

    # Metadata
    evaluation_duration_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OrgAlertReport":
        """Create from dictionary."""
        return cls(
            report_id=data.get("report_id", ""),
            generated_at=data.get("generated_at", ""),
            org_report_path=data.get("org_report_path", ""),
            total_alerts=data.get("total_alerts", 0),
            critical_alerts=data.get("critical_alerts", 0),
            high_alerts=data.get("high_alerts", 0),
            medium_alerts=data.get("medium_alerts", 0),
            low_alerts=data.get("low_alerts", 0),
            slo_alerts=data.get("slo_alerts", 0),
            risk_alerts=data.get("risk_alerts", 0),
            trend_alerts=data.get("trend_alerts", 0),
            integrity_alerts=data.get("integrity_alerts", 0),
            alerts=data.get("alerts", []),
            escalations_triggered=data.get("escalations_triggered", 0),
            escalation_actions=data.get("escalation_actions", []),
            channels_dispatched=data.get("channels_dispatched", []),
            dispatch_errors=data.get("dispatch_errors", []),
            org_health_status=data.get("org_health_status", "unknown"),
            org_health_score=data.get("org_health_score", 0.0),
            org_risk_tier=data.get("org_risk_tier", "unknown"),
            slos_violated=data.get("slos_violated", 0),
            total_repos=data.get("total_repos", 0),
            evaluation_duration_ms=data.get("evaluation_duration_ms", 0.0)
        )


# ============================================================================
# Alert Channels (Abstract Base + Implementations)
# ============================================================================

class OrgAlertChannel(ABC):
    """Abstract base class for org alert channels."""

    def __init__(self, config: OrgAlertChannelConfig):
        """Initialize channel with config."""
        self.config = config

    @abstractmethod
    def dispatch(self, alerts: List[OrgAlert], report: OrgAlertReport) -> bool:
        """
        Dispatch alerts through this channel.

        Args:
            alerts: List of alerts to dispatch
            report: Complete alert report

        Returns:
            True if dispatch succeeded, False otherwise
        """
        pass

    def filter_alerts(self, alerts: List[OrgAlert]) -> List[OrgAlert]:
        """Filter alerts by minimum severity and categories."""
        filtered = []
        for alert in alerts:
            # Check severity
            if alert.severity < self.config.min_severity:
                continue

            # Check categories
            if self.config.categories is not None:
                if alert.category not in self.config.categories:
                    continue

            filtered.append(alert)

        return filtered


class ConsoleOrgAlertChannel(OrgAlertChannel):
    """Dispatches org alerts to console output."""

    def dispatch(self, alerts: List[OrgAlert], report: OrgAlertReport) -> bool:
        """Print alerts to console."""
        filtered = self.filter_alerts(alerts)

        if not filtered:
            print("\n[Org Alerting Engine] No alerts above threshold to display.")
            return True

        print("\n" + "=" * 80)
        print("ORGANIZATION-LEVEL HEALTH ALERTS")
        print("=" * 80)
        print(f"Org Health: {report.org_health_status.upper()} | Score: {report.org_health_score:.1f}")
        print(f"Risk Tier: {report.org_risk_tier.upper()} | Repos: {report.total_repos}")
        print(f"Generated: {report.generated_at}")
        print("-" * 80)

        # Group by severity
        by_severity: Dict[str, List[OrgAlert]] = {}
        for alert in filtered:
            sev = alert.severity.value.upper()
            if sev not in by_severity:
                by_severity[sev] = []
            by_severity[sev].append(alert)

        # Print in severity order
        severity_order = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
        icons = {
            "CRITICAL": "[CRIT]",
            "HIGH": "[HIGH]",
            "MEDIUM": "[MED]",
            "LOW": "[LOW]"
        }

        for severity in severity_order:
            if severity not in by_severity:
                continue

            alerts_list = by_severity[severity]
            icon = icons.get(severity, "")
            print(f"\n{icon} {severity} ({len(alerts_list)}):")
            print("-" * 40)

            for alert in alerts_list:
                print(f"  [{alert.category.value.upper()}] {alert.title}")
                print(f"    {alert.message}")
                if alert.affected_repos:
                    repos_str = ", ".join(alert.affected_repos[:3])
                    if len(alert.affected_repos) > 3:
                        repos_str += f" (+{len(alert.affected_repos) - 3} more)"
                    print(f"    Repos: {repos_str}")
                if alert.recommendations:
                    print(f"    -> {alert.recommendations[0]}")
                print()

        print("=" * 80)
        print(f"Total Alerts: {len(filtered)} | Escalations: {report.escalations_triggered}")
        print("=" * 80)

        return True


class JsonFileOrgAlertChannel(OrgAlertChannel):
    """Dispatches org alerts to a JSON file."""

    def dispatch(self, alerts: List[OrgAlert], report: OrgAlertReport) -> bool:
        """Write alerts to JSON file."""
        if not self.config.output_path:
            logger.error("JsonFileChannel: No output path configured")
            return False

        try:
            output_path = Path(self.config.output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report.to_dict(), f, indent=2, default=str)

            logger.info(f"JsonFileChannel: Wrote {len(alerts)} alerts to {output_path}")
            return True

        except Exception as e:
            logger.error(f"JsonFileChannel: Failed to write alerts: {e}")
            return False


class StdoutOrgAlertChannel(OrgAlertChannel):
    """Dispatches org alerts to stdout as JSON."""

    def dispatch(self, alerts: List[OrgAlert], report: OrgAlertReport) -> bool:
        """Print alert report as JSON to stdout."""
        try:
            print(json.dumps(report.to_dict(), indent=2, default=str))
            return True
        except Exception as e:
            logger.error(f"StdoutChannel: Failed to output JSON: {e}")
            return False


class EmailOrgAlertChannel(OrgAlertChannel):
    """Placeholder stub for email alerts."""

    def dispatch(self, alerts: List[OrgAlert], report: OrgAlertReport) -> bool:
        """Generate email content (stub - does not actually send)."""
        filtered = self.filter_alerts(alerts)

        if not self.config.email_to:
            logger.warning("EmailChannel: No recipient configured (stub)")
            return True

        # Log what would be sent
        subject = self._generate_subject(report)
        logger.info(f"EmailChannel [STUB]: Would send email to {self.config.email_to}")
        logger.info(f"  Subject: {subject}")
        logger.info(f"  Alerts: {len(filtered)}")

        # Store for testing
        self._last_email = {
            "to": self.config.email_to,
            "from": self.config.email_from or "alerts@tars.local",
            "subject": subject,
            "alert_count": len(filtered)
        }

        return True

    def _generate_subject(self, report: OrgAlertReport) -> str:
        """Generate email subject line."""
        if report.critical_alerts > 0:
            return f"[CRITICAL] T.A.R.S. Org Health Alert - {report.critical_alerts} Critical"
        elif report.high_alerts > 0:
            return f"[HIGH] T.A.R.S. Org Health Alert - {report.high_alerts} High Priority"
        else:
            return f"[INFO] T.A.R.S. Org Health Report - {report.total_alerts} Alert(s)"


class SlackOrgAlertChannel(OrgAlertChannel):
    """Placeholder stub for Slack alerts."""

    def dispatch(self, alerts: List[OrgAlert], report: OrgAlertReport) -> bool:
        """Generate Slack payload (stub - does not actually send)."""
        filtered = self.filter_alerts(alerts)

        if not self.config.slack_webhook_url:
            logger.warning("SlackChannel: No webhook URL configured (stub)")
            return True

        # Log what would be sent
        logger.info(f"SlackChannel [STUB]: Would post to {self.config.slack_channel or 'default'}")
        logger.info(f"  Alerts: {len(filtered)}")

        # Store for testing
        self._last_payload = {
            "webhook_url": self.config.slack_webhook_url,
            "channel": self.config.slack_channel,
            "alert_count": len(filtered)
        }

        return True


class WebhookOrgAlertChannel(OrgAlertChannel):
    """Placeholder stub for webhook alerts."""

    def dispatch(self, alerts: List[OrgAlert], report: OrgAlertReport) -> bool:
        """Generate webhook payload (stub - does not actually send)."""
        if not self.config.webhook_url:
            logger.warning("WebhookChannel: No URL configured (stub)")
            return True

        # Log what would be sent
        logger.info(f"WebhookChannel [STUB]: Would POST to {self.config.webhook_url}")
        logger.info(f"  Alerts: {report.total_alerts}")

        # Store for testing
        self._last_payload = {
            "url": self.config.webhook_url,
            "method": "POST",
            "headers": self.config.webhook_headers,
            "payload": report.to_dict()
        }

        return True


# ============================================================================
# Alert Generation Engine
# ============================================================================

class OrgAlertGenerator:
    """
    Generates organization-level alerts from org health report.

    Alert sources:
    1. SLO Violations - Each violated SLO becomes an alert
    2. High-Risk Repos - HIGH/CRITICAL risk repos generate alerts
    3. Trend Signals - Declining trends, low green %, etc.
    4. Integrity Issues - Load errors, missing artifacts
    """

    def __init__(self, config: OrgAlertConfig):
        """Initialize alert generator."""
        self.config = config
        self._alert_counter = 0

    def generate_all_alerts(self, org_report: Dict[str, Any]) -> List[OrgAlert]:
        """
        Generate all alerts from org health report.

        Args:
            org_report: Parsed org-health-report.json

        Returns:
            List of OrgAlert objects
        """
        alerts: List[OrgAlert] = []

        # Generate alerts from each source
        if self.config.generate_slo_alerts:
            alerts.extend(self._generate_slo_alerts(org_report))

        if self.config.generate_risk_alerts:
            alerts.extend(self._generate_risk_alerts(org_report))

        if self.config.generate_trend_alerts:
            alerts.extend(self._generate_trend_alerts(org_report))

        if self.config.generate_integrity_alerts:
            alerts.extend(self._generate_integrity_alerts(org_report))

        return alerts

    def _generate_alert_id(self) -> str:
        """Generate unique alert ID."""
        self._alert_counter += 1
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        return f"org_alert_{timestamp}_{self._alert_counter:04d}"

    def _generate_slo_alerts(self, org_report: Dict[str, Any]) -> List[OrgAlert]:
        """Generate alerts from SLO violations."""
        alerts = []
        timestamp = datetime.utcnow().isoformat()

        slo_results = org_report.get("slo_results", [])

        for slo in slo_results:
            if slo.get("satisfied", True):
                continue

            # Determine severity from SLO violation
            violation_severity = slo.get("violation_severity", "medium")
            severity = AlertSeverity.from_string(violation_severity)

            violating_repos = slo.get("violating_repos", [])

            alert = OrgAlert(
                alert_id=self._generate_alert_id(),
                category=OrgAlertCategory.SLO,
                severity=severity,
                title=f"SLO Violated: {slo.get('slo_id', 'unknown')}",
                message=f"{slo.get('slo_description', 'SLO policy violated')}. "
                       f"Current: {slo.get('current_value', 0):.2f}, "
                       f"Target: {slo.get('target_value', 0)} "
                       f"({slo.get('operator', '??')})",
                timestamp=timestamp,
                source_type="slo",
                slo_id=slo.get("slo_id"),
                slo_description=slo.get("slo_description"),
                current_value=slo.get("current_value"),
                target_value=slo.get("target_value"),
                violating_repos=violating_repos,
                affected_repos=violating_repos,
                recommendations=[
                    f"Review and address issues in violating repositories",
                    f"Evaluate SLO target appropriateness",
                    f"Consider phased remediation plan"
                ]
            )
            alerts.append(alert)

        return alerts

    def _generate_risk_alerts(self, org_report: Dict[str, Any]) -> List[OrgAlert]:
        """Generate alerts from high-risk repositories."""
        alerts = []
        timestamp = datetime.utcnow().isoformat()

        top_risk_repos = org_report.get("top_risk_repos", [])

        for repo in top_risk_repos:
            risk_tier = repo.get("risk_tier", "low").lower()

            # Only alert on HIGH and CRITICAL
            if risk_tier not in ("high", "critical"):
                continue

            severity = AlertSeverity.CRITICAL if risk_tier == "critical" else AlertSeverity.HIGH

            alert = OrgAlert(
                alert_id=self._generate_alert_id(),
                category=OrgAlertCategory.RISK,
                severity=severity,
                title=f"{risk_tier.upper()} Risk: {repo.get('repo_id', 'unknown')}",
                message=f"Repository '{repo.get('repo_name', repo.get('repo_id'))}' is at "
                       f"{risk_tier.upper()} risk (score: {repo.get('risk_score', 0):.1f}). "
                       f"Health: {repo.get('health_status', 'unknown').upper()}, "
                       f"Critical Issues: {repo.get('critical_issues', 0)}",
                timestamp=timestamp,
                source_type="risk",
                risk_tier=risk_tier,
                risk_score=repo.get("risk_score"),
                reason_codes=repo.get("reason_codes", []),
                affected_repos=[repo.get("repo_id")],
                recommendations=[
                    f"Immediately review {repo.get('repo_id')} health dashboard",
                    "Address critical issues first",
                    "Consider rollback if integrity compromised"
                ]
            )
            alerts.append(alert)

        return alerts

    def _generate_trend_alerts(self, org_report: Dict[str, Any]) -> List[OrgAlert]:
        """Generate alerts from org-wide trend signals."""
        alerts = []
        timestamp = datetime.utcnow().isoformat()

        metrics = org_report.get("metrics", {})
        thresholds = self.config.thresholds

        total_repos = metrics.get("total_repos", 0)
        if total_repos == 0:
            return alerts

        # Check percent declining
        percent_declining = metrics.get("repos_declining", 0) / total_repos
        if percent_declining >= thresholds.percent_declining_critical:
            alerts.append(OrgAlert(
                alert_id=self._generate_alert_id(),
                category=OrgAlertCategory.TREND,
                severity=AlertSeverity.CRITICAL,
                title="Critical: High Percentage of Declining Repos",
                message=f"{percent_declining*100:.1f}% of repositories have declining health trends "
                       f"(threshold: {thresholds.percent_declining_critical*100}%)",
                timestamp=timestamp,
                source_type="trend",
                metric_name="percent_declining",
                metric_value=percent_declining,
                threshold=thresholds.percent_declining_critical,
                recommendations=[
                    "Investigate root cause of widespread decline",
                    "Consider org-wide health improvement initiative",
                    "Review recent changes affecting multiple repos"
                ]
            ))
        elif percent_declining >= thresholds.percent_declining_warning:
            alerts.append(OrgAlert(
                alert_id=self._generate_alert_id(),
                category=OrgAlertCategory.TREND,
                severity=AlertSeverity.MEDIUM,
                title="Warning: Elevated Percentage of Declining Repos",
                message=f"{percent_declining*100:.1f}% of repositories have declining health trends "
                       f"(threshold: {thresholds.percent_declining_warning*100}%)",
                timestamp=timestamp,
                source_type="trend",
                metric_name="percent_declining",
                metric_value=percent_declining,
                threshold=thresholds.percent_declining_warning,
                recommendations=[
                    "Monitor declining repos closely",
                    "Prioritize stabilization efforts"
                ]
            ))

        # Check percent green
        percent_green = metrics.get("percent_green", 0) / 100  # Convert from percentage
        if percent_green < thresholds.percent_green_critical:
            alerts.append(OrgAlert(
                alert_id=self._generate_alert_id(),
                category=OrgAlertCategory.TREND,
                severity=AlertSeverity.CRITICAL,
                title="Critical: Low Percentage of Healthy Repos",
                message=f"Only {percent_green*100:.1f}% of repositories are GREEN "
                       f"(threshold: {thresholds.percent_green_critical*100}%)",
                timestamp=timestamp,
                source_type="trend",
                metric_name="percent_green",
                metric_value=percent_green,
                threshold=thresholds.percent_green_critical,
                recommendations=[
                    "Immediate action required on unhealthy repos",
                    "Consider pausing releases until health improves",
                    "Escalate to leadership"
                ]
            ))
        elif percent_green < thresholds.percent_green_warning:
            alerts.append(OrgAlert(
                alert_id=self._generate_alert_id(),
                category=OrgAlertCategory.TREND,
                severity=AlertSeverity.MEDIUM,
                title="Warning: Below Target Green Percentage",
                message=f"Only {percent_green*100:.1f}% of repositories are GREEN "
                       f"(target: {thresholds.percent_green_warning*100}%)",
                timestamp=timestamp,
                source_type="trend",
                metric_name="percent_green",
                metric_value=percent_green,
                threshold=thresholds.percent_green_warning,
                recommendations=[
                    "Review yellow and red repos for improvement opportunities",
                    "Track weekly progress toward green targets"
                ]
            ))

        # Check average score
        avg_score = metrics.get("avg_score", 0)
        if avg_score < thresholds.avg_score_critical:
            alerts.append(OrgAlert(
                alert_id=self._generate_alert_id(),
                category=OrgAlertCategory.TREND,
                severity=AlertSeverity.CRITICAL,
                title="Critical: Low Organization Average Score",
                message=f"Organization average score is {avg_score:.1f} "
                       f"(threshold: {thresholds.avg_score_critical})",
                timestamp=timestamp,
                source_type="trend",
                metric_name="avg_score",
                metric_value=avg_score,
                threshold=thresholds.avg_score_critical,
                recommendations=[
                    "Focus on lowest-scoring repos first",
                    "Consider org-wide quality initiative"
                ]
            ))
        elif avg_score < thresholds.avg_score_warning:
            alerts.append(OrgAlert(
                alert_id=self._generate_alert_id(),
                category=OrgAlertCategory.TREND,
                severity=AlertSeverity.MEDIUM,
                title="Warning: Below Target Average Score",
                message=f"Organization average score is {avg_score:.1f} "
                       f"(target: {thresholds.avg_score_warning})",
                timestamp=timestamp,
                source_type="trend",
                metric_name="avg_score",
                metric_value=avg_score,
                threshold=thresholds.avg_score_warning,
                recommendations=[
                    "Set improvement targets per repo",
                    "Track weekly score trends"
                ]
            ))

        return alerts

    def _generate_integrity_alerts(self, org_report: Dict[str, Any]) -> List[OrgAlert]:
        """Generate alerts from integrity issues (load errors)."""
        alerts = []
        timestamp = datetime.utcnow().isoformat()

        load_errors = org_report.get("load_errors", [])

        if not load_errors:
            return alerts

        # Group errors by type
        error_repos = [e.get("repo_id", "unknown") for e in load_errors]

        severity = AlertSeverity.HIGH if len(load_errors) > 2 else AlertSeverity.MEDIUM

        alert = OrgAlert(
            alert_id=self._generate_alert_id(),
            category=OrgAlertCategory.INTEGRITY,
            severity=severity,
            title=f"Data Integrity: {len(load_errors)} Repository Load Error(s)",
            message=f"Failed to load health data for {len(load_errors)} repository(ies). "
                   f"These repos may have missing or corrupted artifacts.",
            timestamp=timestamp,
            source_type="integrity",
            affected_repos=error_repos,
            details={"load_errors": load_errors},
            recommendations=[
                "Check artifact generation pipeline for affected repos",
                "Verify dashboard/alerts/trends files exist",
                "Run integrity scan on affected repos"
            ]
        )
        alerts.append(alert)

        return alerts


# ============================================================================
# Escalation Engine
# ============================================================================

class EscalationEngine:
    """
    Evaluates escalation rules and executes actions.

    The escalation engine:
    1. Matches alerts against configured rules
    2. Executes matching actions (as stubs)
    3. Tracks escalation history
    """

    def __init__(self, rules: List[EscalationRule]):
        """Initialize escalation engine."""
        self.rules = sorted(rules, key=lambda r: r.priority, reverse=True)
        self._escalation_log: List[Dict[str, Any]] = []

    def process_alerts(
        self,
        alerts: List[OrgAlert],
        org_metrics: Dict[str, Any] = None
    ) -> Tuple[List[OrgAlert], List[Dict[str, Any]]]:
        """
        Process alerts through escalation rules.

        Args:
            alerts: List of alerts to process
            org_metrics: Optional org-level metrics for rule conditions

        Returns:
            Tuple of (updated alerts, escalation actions taken)
        """
        actions_taken = []

        for alert in alerts:
            for rule in self.rules:
                if rule.matches(alert, org_metrics):
                    # Execute actions for this rule
                    for action in rule.actions:
                        action_result = self._execute_action(action, alert, rule)
                        if action_result:
                            actions_taken.append(action_result)
                            alert.escalated = True
                            alert.escalation_actions.append(
                                f"{action.action_type.value}:{action.target or action.channel or 'default'}"
                            )

        return alerts, actions_taken

    def _execute_action(
        self,
        action: EscalationAction,
        alert: OrgAlert,
        rule: EscalationRule
    ) -> Optional[Dict[str, Any]]:
        """
        Execute an escalation action (stub implementation).

        Args:
            action: Action to execute
            alert: Alert that triggered the action
            rule: Rule that matched

        Returns:
            Action result dictionary or None
        """
        result = {
            "action_type": action.action_type.value,
            "rule_id": rule.id,
            "alert_id": alert.alert_id,
            "alert_severity": alert.severity.value,
            "timestamp": datetime.utcnow().isoformat()
        }

        if action.action_type == EscalationActionType.ESCALATE_TO:
            logger.info(f"[ESCALATION STUB] Escalating alert {alert.alert_id} to: {action.target}")
            result["target"] = action.target
            result["message"] = f"Would escalate to {action.target}"

        elif action.action_type == EscalationActionType.NOTIFY:
            logger.info(f"[ESCALATION STUB] Notifying via {action.channel}: {action.recipient}")
            result["channel"] = action.channel
            result["recipient"] = action.recipient
            result["message"] = f"Would notify {action.recipient} via {action.channel}"

        elif action.action_type == EscalationActionType.LOG:
            logger.info(f"[ESCALATION] Logged alert: {alert.alert_id} - {alert.title}")
            result["message"] = "Alert logged"

        elif action.action_type == EscalationActionType.SUPPRESS:
            logger.info(f"[ESCALATION] Suppressing alert: {alert.alert_id}")
            result["message"] = "Alert suppressed"
            return None  # Don't log suppressed actions

        else:
            result["message"] = f"Custom action: {action.target}"

        self._escalation_log.append(result)
        return result

    def get_escalation_log(self) -> List[Dict[str, Any]]:
        """Get the escalation action log."""
        return self._escalation_log


# ============================================================================
# Alert Router/Dispatcher
# ============================================================================

class OrgAlertDispatcher:
    """
    Routes alerts to configured channels.
    """

    def __init__(self, config: OrgAlertConfig):
        """Initialize dispatcher with configuration."""
        self.config = config
        self.channels: List[OrgAlertChannel] = []

        # Initialize configured channels
        for channel_config in config.channels:
            if channel_config.enabled:
                channel = self._create_channel(channel_config)
                if channel:
                    self.channels.append(channel)

    def _create_channel(self, config: OrgAlertChannelConfig) -> Optional[OrgAlertChannel]:
        """Create a channel instance from config."""
        try:
            if config.channel_type == OrgAlertChannelType.CONSOLE:
                return ConsoleOrgAlertChannel(config)
            elif config.channel_type == OrgAlertChannelType.JSON_FILE:
                return JsonFileOrgAlertChannel(config)
            elif config.channel_type == OrgAlertChannelType.STDOUT:
                return StdoutOrgAlertChannel(config)
            elif config.channel_type == OrgAlertChannelType.EMAIL:
                return EmailOrgAlertChannel(config)
            elif config.channel_type == OrgAlertChannelType.SLACK:
                return SlackOrgAlertChannel(config)
            elif config.channel_type == OrgAlertChannelType.WEBHOOK:
                return WebhookOrgAlertChannel(config)
            else:
                logger.warning(f"Unknown channel type: {config.channel_type}")
                return None
        except Exception as e:
            logger.error(f"Failed to create channel {config.channel_type}: {e}")
            return None

    def dispatch(self, alerts: List[OrgAlert], report: OrgAlertReport) -> OrgAlertReport:
        """
        Dispatch alerts to all configured channels.

        Args:
            alerts: List of alerts to dispatch
            report: Alert report for context

        Returns:
            Updated report with dispatch status
        """
        dispatched = []
        errors = []

        for channel in self.channels:
            channel_name = channel.config.channel_type.value
            try:
                success = channel.dispatch(alerts, report)
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
# Main Engine
# ============================================================================

class OrgAlertingEngine:
    """
    Main orchestrator for the Organization-Level Alerting & Escalation Engine.

    This class:
    1. Loads org health report
    2. Generates alerts from all sources
    3. Processes escalation rules
    4. Dispatches to channels
    5. Writes alert report
    6. Returns appropriate exit code
    """

    def __init__(self, config: OrgAlertConfig):
        """Initialize the org alerting engine."""
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
        self.generator = OrgAlertGenerator(config)
        self.escalation_engine = EscalationEngine(config.escalation_rules)
        self.dispatcher = OrgAlertDispatcher(config)

        # Data storage
        self.org_report: Optional[Dict[str, Any]] = None

    def run(self) -> Tuple[OrgAlertReport, int]:
        """
        Run the complete org alerting pipeline.

        Returns:
            Tuple of (OrgAlertReport, exit_code)
        """
        start_time = datetime.utcnow()

        try:
            logger.info("=" * 80)
            logger.info("ORGANIZATION-LEVEL ALERTING & ESCALATION ENGINE")
            logger.info("=" * 80)

            # Step 1: Load org health report
            logger.info("\nStep 1: Loading org health report...")
            self.org_report = self._load_org_report()
            logger.info(f"  Loaded report: {self.config.org_report_path}")
            logger.info(f"  Org Health: {self.org_report.get('org_health_status', 'unknown').upper()}")
            logger.info(f"  Repos: {self.org_report.get('repos_loaded', 0)}")

            # Step 2: Generate alerts
            logger.info("\nStep 2: Generating alerts...")
            alerts = self.generator.generate_all_alerts(self.org_report)
            logger.info(f"  Generated {len(alerts)} alert(s)")

            # Step 3: Process escalations
            logger.info("\nStep 3: Processing escalation rules...")
            alerts, escalation_actions = self.escalation_engine.process_alerts(
                alerts,
                self.org_report.get("metrics", {})
            )
            logger.info(f"  Triggered {len(escalation_actions)} escalation action(s)")

            # Step 4: Generate report
            logger.info("\nStep 4: Generating alert report...")
            report = self._generate_report(alerts, escalation_actions)

            # Calculate duration
            duration = (datetime.utcnow() - start_time).total_seconds() * 1000
            report.evaluation_duration_ms = duration

            logger.info(f"  Critical: {report.critical_alerts}")
            logger.info(f"  High: {report.high_alerts}")
            logger.info(f"  Medium: {report.medium_alerts}")
            logger.info(f"  Low: {report.low_alerts}")

            # Step 5: Dispatch alerts
            logger.info("\nStep 5: Dispatching alerts...")
            report = self.dispatcher.dispatch(alerts, report)
            logger.info(f"  Dispatched to {len(report.channels_dispatched)} channel(s)")

            # Step 6: Write output
            if self.config.output_path:
                logger.info("\nStep 6: Writing alert report...")
                self._write_report(report)
                logger.info(f"  Wrote report to {self.config.output_path}")

            # Determine exit code
            exit_code = self._determine_exit_code(report)

            logger.info("\n" + "=" * 80)
            logger.info("ORG ALERTING COMPLETE")
            logger.info("=" * 80)
            logger.info(f"Total Alerts: {report.total_alerts}")
            logger.info(f"Escalations: {report.escalations_triggered}")
            logger.info(f"Exit Code: {exit_code}")
            logger.info("=" * 80)

            return report, exit_code

        except OrgReportParseError as e:
            logger.error(f"Failed to parse org health report: {e}")
            return self._error_report(str(e)), EXIT_ORG_REPORT_PARSE_ERROR
        except OrgAlertConfigError as e:
            logger.error(f"Configuration error: {e}")
            return self._error_report(str(e)), EXIT_ALERTING_CONFIG_ERROR
        except OrgRoutingError as e:
            logger.error(f"Routing error: {e}")
            return self._error_report(str(e)), EXIT_ROUTING_FAILURE
        except Exception as e:
            logger.error(f"Org alerting error: {e}")
            return self._error_report(str(e)), EXIT_GENERAL_ALERTING_ERROR

    def _load_org_report(self) -> Dict[str, Any]:
        """Load and validate org health report."""
        report_path = self.config.org_report_path

        if not report_path.exists():
            raise OrgReportParseError(f"Org health report not found: {report_path}")

        try:
            with open(report_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Validate required fields
            required_fields = ["org_health_status", "repos_loaded"]
            for field_name in required_fields:
                if field_name not in data:
                    raise OrgReportParseError(f"Missing required field: {field_name}")

            return data

        except json.JSONDecodeError as e:
            raise OrgReportParseError(f"Invalid JSON in org health report: {e}")
        except Exception as e:
            raise OrgReportParseError(f"Failed to load org health report: {e}")

    def _generate_report(
        self,
        alerts: List[OrgAlert],
        escalation_actions: List[Dict[str, Any]]
    ) -> OrgAlertReport:
        """Generate alert report from alerts."""
        timestamp = datetime.utcnow().isoformat()

        # Count by severity
        critical = sum(1 for a in alerts if a.severity == AlertSeverity.CRITICAL)
        high = sum(1 for a in alerts if a.severity == AlertSeverity.HIGH)
        medium = sum(1 for a in alerts if a.severity == AlertSeverity.MEDIUM)
        low = sum(1 for a in alerts if a.severity == AlertSeverity.LOW)

        # Count by category
        slo_count = sum(1 for a in alerts if a.category == OrgAlertCategory.SLO)
        risk_count = sum(1 for a in alerts if a.category == OrgAlertCategory.RISK)
        trend_count = sum(1 for a in alerts if a.category == OrgAlertCategory.TREND)
        integrity_count = sum(1 for a in alerts if a.category == OrgAlertCategory.INTEGRITY)

        return OrgAlertReport(
            report_id=f"org_alert_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            generated_at=timestamp,
            org_report_path=str(self.config.org_report_path),
            total_alerts=len(alerts),
            critical_alerts=critical,
            high_alerts=high,
            medium_alerts=medium,
            low_alerts=low,
            slo_alerts=slo_count,
            risk_alerts=risk_count,
            trend_alerts=trend_count,
            integrity_alerts=integrity_count,
            alerts=[a.to_dict() for a in alerts],
            escalations_triggered=len(escalation_actions),
            escalation_actions=escalation_actions,
            org_health_status=self.org_report.get("org_health_status", "unknown"),
            org_health_score=self.org_report.get("org_health_score", 0.0),
            org_risk_tier=self.org_report.get("org_risk_tier", "unknown"),
            slos_violated=self.org_report.get("slos_violated", 0),
            total_repos=self.org_report.get("repos_loaded", 0)
        )

    def _write_report(self, report: OrgAlertReport) -> None:
        """Write alert report to JSON file."""
        try:
            output_path = Path(self.config.output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report.to_dict(), f, indent=2, default=str)

        except Exception as e:
            raise OrgRoutingError(f"Failed to write alert report: {e}")

    def _determine_exit_code(self, report: OrgAlertReport) -> int:
        """Determine appropriate exit code based on report and config."""
        # Check for routing failures
        if (report.dispatch_errors and
            len(report.dispatch_errors) == len(self.dispatcher.channels) and
            len(self.dispatcher.channels) > 0):
            return EXIT_ROUTING_FAILURE

        # Check for critical alerts
        if report.critical_alerts > 0 and self.config.fail_on_critical:
            return EXIT_CRITICAL_ALERTS

        # Check for any alerts
        if report.total_alerts > 0 and self.config.fail_on_any_alerts:
            return EXIT_ALERTS_PRESENT

        # Check for any alerts (non-failing)
        if report.total_alerts > 0:
            return EXIT_ALERTS_PRESENT

        return EXIT_ORG_ALERT_SUCCESS

    def _error_report(self, error_message: str) -> OrgAlertReport:
        """Generate an error report."""
        return OrgAlertReport(
            report_id=f"org_alert_error_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            generated_at=datetime.utcnow().isoformat(),
            org_report_path=str(self.config.org_report_path),
            dispatch_errors=[{"channel": "engine", "error": error_message}]
        )


# ============================================================================
# Utility Functions
# ============================================================================

def create_default_escalation_rules() -> List[EscalationRule]:
    """
    Create a default set of escalation rules.

    Returns:
        List of default EscalationRule objects
    """
    return [
        EscalationRule(
            id="slo-critical",
            description="Escalate critical SLO violations",
            condition=EscalationCondition(
                alert_category=OrgAlertCategory.SLO,
                severity=AlertSeverity.CRITICAL
            ),
            actions=[
                EscalationAction.from_string("escalate_to:oncall"),
                EscalationAction.from_string("notify:slack:org-slo-critical"),
                EscalationAction.from_string("notify:email:leadership")
            ],
            priority=100
        ),
        EscalationRule(
            id="high-risk-repo",
            description="Notify on high-risk repositories",
            condition=EscalationCondition(
                alert_category=OrgAlertCategory.RISK,
                severity=AlertSeverity.HIGH
            ),
            actions=[
                EscalationAction.from_string("notify:slack:repo-high-risk"),
                EscalationAction.from_string("log")
            ],
            priority=80
        ),
        EscalationRule(
            id="critical-risk-repo",
            description="Escalate critical-risk repositories",
            condition=EscalationCondition(
                alert_category=OrgAlertCategory.RISK,
                severity=AlertSeverity.CRITICAL
            ),
            actions=[
                EscalationAction.from_string("escalate_to:oncall"),
                EscalationAction.from_string("notify:slack:org-critical"),
                EscalationAction.from_string("log")
            ],
            priority=90
        ),
        EscalationRule(
            id="org-declining",
            description="Alert on org-wide declining trend",
            condition=EscalationCondition(
                metric="percent_declining",
                operator=">=",
                value=0.30
            ),
            actions=[
                EscalationAction.from_string("notify:email:org-devops"),
                EscalationAction.from_string("log")
            ],
            priority=50
        ),
        EscalationRule(
            id="integrity-issues",
            description="Notify on integrity issues",
            condition=EscalationCondition(
                alert_category=OrgAlertCategory.INTEGRITY
            ),
            actions=[
                EscalationAction.from_string("notify:slack:infrastructure"),
                EscalationAction.from_string("log")
            ],
            priority=60
        )
    ]


def create_default_channels(
    output_dir: Optional[Path] = None,
    json_output: bool = False
) -> List[OrgAlertChannelConfig]:
    """
    Create default channel configurations.

    Args:
        output_dir: Optional output directory for JSON file
        json_output: If True, add stdout JSON channel

    Returns:
        List of OrgAlertChannelConfig objects
    """
    channels = [
        OrgAlertChannelConfig(
            channel_type=OrgAlertChannelType.CONSOLE,
            enabled=True
        )
    ]

    if output_dir:
        channels.append(OrgAlertChannelConfig(
            channel_type=OrgAlertChannelType.JSON_FILE,
            enabled=True,
            output_path=output_dir / "org-alerts.json"
        ))

    if json_output:
        channels.append(OrgAlertChannelConfig(
            channel_type=OrgAlertChannelType.STDOUT,
            enabled=True
        ))

    return channels


def load_escalation_config(config_path: Path) -> List[EscalationRule]:
    """
    Load escalation rules from a configuration file.

    Args:
        config_path: Path to YAML or JSON config file

    Returns:
        List of EscalationRule objects
    """
    if config_path.suffix in ('.yaml', '.yml'):
        try:
            import yaml
            with open(config_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
        except ImportError:
            raise OrgAlertConfigError("PyYAML not installed. Install with: pip install pyyaml")
    else:
        with open(config_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

    rules = []
    for rule_data in data.get("escalation_rules", []):
        rules.append(EscalationRule.from_dict(rule_data))

    return rules
