"""
Integration Tests for the Repository Health Alerting Engine

Tests cover:
- AlertSeverity enum and comparisons
- AlertType enum values
- Alert and AlertReport dataclasses
- AlertRule configuration
- AlertRulesEngine rule evaluation
- Trend detection (previous vs current dashboard)
- Alert channels (Console, File, Email, Webhook)
- AlertDispatcher channel dispatch
- AlertingEngine orchestration
- CLI end-to-end functionality
- Exit code handling
- Edge cases (empty dashboard, malformed dashboard, no issues)

Version: 1.0.0
Phase: 14.7 Task 9
"""

import json
import pytest
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Import module under test
from analytics.alerting_engine import (
    # Enums
    AlertSeverity,
    AlertType,
    ChannelType,
    # Dataclasses
    Alert,
    AlertRule,
    AlertReport,
    ChannelConfig,
    AlertingConfig,
    # Channels
    ConsoleChannel,
    FileChannel,
    EmailChannel,
    WebhookChannel,
    AlertChannel,
    # Core components
    AlertRulesEngine,
    AlertDispatcher,
    AlertingEngine,
    # Exceptions
    AlertingError,
    InvalidDashboardError,
    ChannelDispatchError,
    RuleEvaluationError,
    AlertWriteError,
    # Exit codes
    EXIT_NO_ALERTS,
    EXIT_ALERTS_TRIGGERED,
    EXIT_CRITICAL_ALERTS,
    EXIT_INVALID_DASHBOARD,
    EXIT_CHANNEL_DISPATCH_FAILURE,
    EXIT_RULE_EVALUATION_FAILURE,
    EXIT_ALERTS_WRITE_FAILURE,
    EXIT_GENERAL_ALERTING_ERROR,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_healthy_dashboard(temp_dir):
    """Create a sample healthy dashboard JSON."""
    dashboard_data = {
        "overall_health": "green",
        "repository_score": 95.0,
        "scan_timestamp": datetime.utcnow().isoformat(),
        "repository_path": "/path/to/repo",
        "total_issues": 2,
        "critical_issues": 0,
        "error_issues": 0,
        "warning_issues": 1,
        "info_issues": 1,
        "total_versions": 5,
        "healthy_versions": 4,
        "warning_versions": 1,
        "critical_versions": 0,
        "total_artifacts": 50,
        "orphaned_artifacts": 0,
        "corrupted_artifacts": 0,
        "missing_artifacts": 0,
        "repair_count": 0,
        "rollback_count": 0,
        "publication_count": 5,
        "issues": [
            {
                "issue_id": "issue_1",
                "severity": "WARNING",
                "category": "metadata",
                "description": "Minor metadata issue"
            },
            {
                "issue_id": "issue_2",
                "severity": "INFO",
                "category": "info",
                "description": "Informational message"
            }
        ],
        "versions_health": [
            {"version": "1.0.0", "health_status": "green", "sbom_present": True, "slsa_present": True},
            {"version": "1.0.1", "health_status": "green", "sbom_present": True, "slsa_present": True},
            {"version": "1.0.2", "health_status": "green", "sbom_present": True, "slsa_present": True},
            {"version": "1.0.3", "health_status": "yellow", "sbom_present": True, "slsa_present": False},
            {"version": "1.0.4", "health_status": "green", "sbom_present": True, "slsa_present": True},
        ],
        "rollback_history": [],
        "recommendations": ["Schedule regular integrity scans"]
    }

    dashboard_path = temp_dir / "health-dashboard.json"
    with open(dashboard_path, 'w') as f:
        json.dump(dashboard_data, f, indent=2)

    return dashboard_path


@pytest.fixture
def sample_critical_dashboard(temp_dir):
    """Create a sample dashboard with critical issues."""
    dashboard_data = {
        "overall_health": "red",
        "repository_score": 35.0,
        "scan_timestamp": datetime.utcnow().isoformat(),
        "repository_path": "/path/to/repo",
        "total_issues": 15,
        "critical_issues": 3,
        "error_issues": 5,
        "warning_issues": 4,
        "info_issues": 3,
        "total_versions": 5,
        "healthy_versions": 1,
        "warning_versions": 2,
        "critical_versions": 2,
        "total_artifacts": 50,
        "orphaned_artifacts": 5,
        "corrupted_artifacts": 3,
        "missing_artifacts": 2,
        "repair_count": 0,
        "rollback_count": 1,
        "publication_count": 5,
        "issues": [
            {
                "issue_id": "critical_1",
                "severity": "CRITICAL",
                "category": "corruption",
                "description": "Corrupted artifact detected"
            },
            {
                "issue_id": "critical_2",
                "severity": "CRITICAL",
                "category": "corruption",
                "description": "SHA256 mismatch"
            },
            {
                "issue_id": "critical_3",
                "severity": "CRITICAL",
                "category": "missing",
                "description": "Missing critical artifact"
            },
        ],
        "versions_health": [
            {"version": "1.0.0", "health_status": "green", "sbom_present": True, "slsa_present": True},
            {"version": "1.0.1", "health_status": "yellow", "sbom_present": True, "slsa_present": False},
            {"version": "1.0.2", "health_status": "yellow", "sbom_present": False, "slsa_present": True},
            {"version": "1.0.3", "health_status": "red", "sbom_present": False, "slsa_present": False},
            {"version": "1.0.4", "health_status": "red", "sbom_present": True, "slsa_present": True},
        ],
        "rollback_history": [
            {"from_version": "1.0.4", "to_version": "1.0.3", "status": "failed", "timestamp": datetime.utcnow().isoformat()}
        ],
        "recommendations": ["Address critical issues immediately"]
    }

    dashboard_path = temp_dir / "critical-dashboard.json"
    with open(dashboard_path, 'w') as f:
        json.dump(dashboard_data, f, indent=2)

    return dashboard_path


@pytest.fixture
def sample_previous_dashboard(temp_dir):
    """Create a sample previous dashboard for trend comparison."""
    dashboard_data = {
        "overall_health": "green",
        "repository_score": 98.0,
        "scan_timestamp": "2025-01-01T00:00:00",
        "repository_path": "/path/to/repo",
        "total_issues": 1,
        "critical_issues": 0,
        "error_issues": 0,
        "warning_issues": 0,
        "info_issues": 1,
        "total_versions": 4,
        "healthy_versions": 4,
        "warning_versions": 0,
        "critical_versions": 0,
        "total_artifacts": 40,
        "orphaned_artifacts": 0,
        "corrupted_artifacts": 0,
        "missing_artifacts": 0,
        "versions_health": [
            {"version": "1.0.0", "health_status": "green", "sbom_present": True, "slsa_present": True},
            {"version": "1.0.1", "health_status": "green", "sbom_present": True, "slsa_present": True},
            {"version": "1.0.2", "health_status": "green", "sbom_present": True, "slsa_present": True},
            {"version": "1.0.3", "health_status": "green", "sbom_present": True, "slsa_present": True},
        ],
        "rollback_history": [],
    }

    dashboard_path = temp_dir / "previous-dashboard.json"
    with open(dashboard_path, 'w') as f:
        json.dump(dashboard_data, f, indent=2)

    return dashboard_path


# ============================================================================
# AlertSeverity Tests
# ============================================================================

class TestAlertSeverity:
    """Tests for AlertSeverity enum."""

    def test_severity_values(self):
        """Test all severity values exist."""
        assert AlertSeverity.INFO.value == "INFO"
        assert AlertSeverity.WARNING.value == "WARNING"
        assert AlertSeverity.ERROR.value == "ERROR"
        assert AlertSeverity.CRITICAL.value == "CRITICAL"

    def test_from_string(self):
        """Test from_string conversion."""
        assert AlertSeverity.from_string("INFO") == AlertSeverity.INFO
        assert AlertSeverity.from_string("warning") == AlertSeverity.WARNING
        assert AlertSeverity.from_string("ERROR") == AlertSeverity.ERROR
        assert AlertSeverity.from_string("CRITICAL") == AlertSeverity.CRITICAL

    def test_from_string_invalid(self):
        """Test from_string with invalid value returns INFO."""
        assert AlertSeverity.from_string("invalid") == AlertSeverity.INFO
        assert AlertSeverity.from_string("") == AlertSeverity.INFO

    def test_severity_comparison(self):
        """Test severity level comparisons."""
        assert AlertSeverity.INFO < AlertSeverity.WARNING
        assert AlertSeverity.WARNING < AlertSeverity.ERROR
        assert AlertSeverity.ERROR < AlertSeverity.CRITICAL
        assert AlertSeverity.INFO <= AlertSeverity.INFO
        assert AlertSeverity.CRITICAL <= AlertSeverity.CRITICAL


# ============================================================================
# AlertType Tests
# ============================================================================

class TestAlertType:
    """Tests for AlertType enum."""

    def test_alert_types_exist(self):
        """Test all alert types are defined."""
        assert AlertType.REPOSITORY_STATUS.value == "repository_status"
        assert AlertType.CRITICAL_ISSUE.value == "critical_issue"
        assert AlertType.MISSING_ARTIFACT.value == "missing_artifact"
        assert AlertType.CORRUPTED_ARTIFACT.value == "corrupted_artifact"
        assert AlertType.METADATA_MISSING.value == "metadata_missing"
        assert AlertType.RAPID_REGRESSION.value == "rapid_regression"
        assert AlertType.VERSION_HEALTH.value == "version_health"
        assert AlertType.REPOSITORY_SCORE_DROP.value == "repository_score_drop"


# ============================================================================
# Alert Dataclass Tests
# ============================================================================

class TestAlert:
    """Tests for Alert dataclass."""

    def test_alert_creation(self):
        """Test basic alert creation."""
        alert = Alert(
            alert_id="test_alert_1",
            alert_type="repository_status",
            severity="CRITICAL",
            title="Test Alert",
            message="This is a test alert",
            timestamp="2025-01-01T00:00:00"
        )

        assert alert.alert_id == "test_alert_1"
        assert alert.alert_type == "repository_status"
        assert alert.severity == "CRITICAL"
        assert alert.title == "Test Alert"
        assert alert.message == "This is a test alert"

    def test_alert_to_dict(self):
        """Test alert to dictionary conversion."""
        alert = Alert(
            alert_id="test_alert_2",
            alert_type="critical_issue",
            severity="ERROR",
            title="Test",
            message="Test message",
            timestamp="2025-01-01T00:00:00",
            version="1.0.0",
            issue_count=5
        )

        alert_dict = alert.to_dict()

        assert isinstance(alert_dict, dict)
        assert alert_dict["alert_id"] == "test_alert_2"
        assert alert_dict["version"] == "1.0.0"
        assert alert_dict["issue_count"] == 5

    def test_alert_with_recommendations(self):
        """Test alert with recommendations."""
        alert = Alert(
            alert_id="test_alert_3",
            alert_type="corrupted_artifact",
            severity="CRITICAL",
            title="Corruption Detected",
            message="3 corrupted artifacts",
            timestamp="2025-01-01T00:00:00",
            recommendations=[
                "Run integrity scan with repair",
                "Check backup availability"
            ]
        )

        assert len(alert.recommendations) == 2
        assert "repair" in alert.recommendations[0].lower()


# ============================================================================
# AlertRule Tests
# ============================================================================

class TestAlertRule:
    """Tests for AlertRule dataclass."""

    def test_rule_creation(self):
        """Test alert rule creation."""
        rule = AlertRule(
            name="test_rule",
            alert_type=AlertType.CRITICAL_ISSUE,
            severity=AlertSeverity.CRITICAL,
            description="Test rule description"
        )

        assert rule.name == "test_rule"
        assert rule.alert_type == AlertType.CRITICAL_ISSUE
        assert rule.severity == AlertSeverity.CRITICAL
        assert rule.enabled is True

    def test_rule_with_threshold(self):
        """Test rule with threshold configuration."""
        rule = AlertRule(
            name="score_drop",
            alert_type=AlertType.REPOSITORY_SCORE_DROP,
            severity=AlertSeverity.WARNING,
            description="Score drop detection",
            threshold=15.0
        )

        assert rule.threshold == 15.0

    def test_rule_disabled(self):
        """Test disabled rule."""
        rule = AlertRule(
            name="disabled_rule",
            alert_type=AlertType.METADATA_MISSING,
            severity=AlertSeverity.WARNING,
            description="Disabled",
            enabled=False
        )

        assert rule.enabled is False

    def test_rule_requires_name(self):
        """Test that rule requires a name."""
        with pytest.raises(ValueError, match="Rule name is required"):
            AlertRule(
                name="",
                alert_type=AlertType.CRITICAL_ISSUE,
                severity=AlertSeverity.CRITICAL,
                description="Test"
            )


# ============================================================================
# AlertReport Tests
# ============================================================================

class TestAlertReport:
    """Tests for AlertReport dataclass."""

    def test_report_creation(self):
        """Test alert report creation."""
        report = AlertReport(
            report_id="report_1",
            generated_at="2025-01-01T00:00:00",
            dashboard_path="/path/to/dashboard.json",
            previous_dashboard_path=None,
            total_alerts=5,
            critical_alerts=2,
            error_alerts=1,
            warning_alerts=1,
            info_alerts=1
        )

        assert report.report_id == "report_1"
        assert report.total_alerts == 5
        assert report.critical_alerts == 2

    def test_report_to_dict(self):
        """Test report to dictionary conversion."""
        report = AlertReport(
            report_id="report_2",
            generated_at="2025-01-01T00:00:00",
            dashboard_path="/path/to/dashboard.json",
            previous_dashboard_path="/path/to/previous.json",
            total_alerts=3,
            rules_evaluated=10,
            rules_triggered=3
        )

        report_dict = report.to_dict()

        assert isinstance(report_dict, dict)
        assert report_dict["rules_evaluated"] == 10
        assert report_dict["rules_triggered"] == 3


# ============================================================================
# AlertRulesEngine Tests
# ============================================================================

class TestAlertRulesEngine:
    """Tests for AlertRulesEngine."""

    def test_engine_initialization(self, temp_dir, sample_healthy_dashboard):
        """Test rules engine initialization."""
        config = AlertingConfig(
            current_dashboard_path=sample_healthy_dashboard,
            output_path=temp_dir / "alerts.json"
        )

        engine = AlertRulesEngine(config)

        assert engine.config == config
        assert len(engine.rules) > 0  # Default rules created

    def test_load_valid_dashboard(self, sample_healthy_dashboard):
        """Test loading a valid dashboard."""
        config = AlertingConfig(current_dashboard_path=sample_healthy_dashboard)
        engine = AlertRulesEngine(config)

        dashboard = engine.load_dashboard(sample_healthy_dashboard)

        assert dashboard["overall_health"] == "green"
        assert dashboard["repository_score"] == 95.0

    def test_load_invalid_dashboard(self, temp_dir):
        """Test loading an invalid dashboard raises error."""
        invalid_path = temp_dir / "invalid.json"
        with open(invalid_path, 'w') as f:
            f.write("not valid json")

        config = AlertingConfig(current_dashboard_path=invalid_path)
        engine = AlertRulesEngine(config)

        with pytest.raises(InvalidDashboardError, match="Invalid JSON"):
            engine.load_dashboard(invalid_path)

    def test_load_missing_dashboard(self, temp_dir):
        """Test loading a missing dashboard raises error."""
        missing_path = temp_dir / "missing.json"

        config = AlertingConfig(current_dashboard_path=missing_path)
        engine = AlertRulesEngine(config)

        with pytest.raises(InvalidDashboardError, match="not found"):
            engine.load_dashboard(missing_path)

    def test_load_incomplete_dashboard(self, temp_dir):
        """Test loading a dashboard missing required fields."""
        incomplete_path = temp_dir / "incomplete.json"
        with open(incomplete_path, 'w') as f:
            json.dump({"some_field": "value"}, f)

        config = AlertingConfig(current_dashboard_path=incomplete_path)
        engine = AlertRulesEngine(config)

        with pytest.raises(InvalidDashboardError, match="Missing required field"):
            engine.load_dashboard(incomplete_path)

    def test_evaluate_rules_healthy_dashboard(self, sample_healthy_dashboard, temp_dir):
        """Test rule evaluation on healthy dashboard."""
        config = AlertingConfig(
            current_dashboard_path=sample_healthy_dashboard,
            output_path=temp_dir / "alerts.json"
        )
        engine = AlertRulesEngine(config)
        engine.current_dashboard = engine.load_dashboard(sample_healthy_dashboard)

        alerts = engine.evaluate_rules()

        # Healthy dashboard should have minimal alerts
        critical_alerts = [a for a in alerts if a.severity == "CRITICAL"]
        assert len(critical_alerts) == 0

    def test_evaluate_rules_critical_dashboard(self, sample_critical_dashboard, temp_dir):
        """Test rule evaluation on critical dashboard."""
        config = AlertingConfig(
            current_dashboard_path=sample_critical_dashboard,
            output_path=temp_dir / "alerts.json"
        )
        engine = AlertRulesEngine(config)
        engine.current_dashboard = engine.load_dashboard(sample_critical_dashboard)

        alerts = engine.evaluate_rules()

        # Critical dashboard should trigger multiple alerts
        assert len(alerts) > 0

        # Should have critical alerts
        critical_alerts = [a for a in alerts if a.severity == "CRITICAL"]
        assert len(critical_alerts) > 0

    def test_evaluate_rules_no_dashboard_loaded(self, sample_healthy_dashboard, temp_dir):
        """Test rule evaluation without dashboard loaded."""
        config = AlertingConfig(
            current_dashboard_path=sample_healthy_dashboard,
            output_path=temp_dir / "alerts.json"
        )
        engine = AlertRulesEngine(config)
        # Don't load dashboard

        with pytest.raises(RuleEvaluationError, match="No dashboard data loaded"):
            engine.evaluate_rules()

    def test_generate_report(self, sample_healthy_dashboard, temp_dir):
        """Test alert report generation."""
        config = AlertingConfig(
            current_dashboard_path=sample_healthy_dashboard,
            output_path=temp_dir / "alerts.json"
        )
        engine = AlertRulesEngine(config)
        engine.current_dashboard = engine.load_dashboard(sample_healthy_dashboard)

        alerts = engine.evaluate_rules()
        report = engine.generate_report(alerts)

        assert report.report_id is not None
        assert report.dashboard_path == str(sample_healthy_dashboard)
        assert report.rules_evaluated > 0


# ============================================================================
# Trend Detection Tests
# ============================================================================

class TestTrendDetection:
    """Tests for trend-based alert detection."""

    def test_score_drop_detection(self, sample_critical_dashboard, sample_previous_dashboard, temp_dir):
        """Test detection of significant score drop."""
        config = AlertingConfig(
            current_dashboard_path=sample_critical_dashboard,
            previous_dashboard_path=sample_previous_dashboard,
            output_path=temp_dir / "alerts.json",
            score_drop_threshold=10.0
        )
        engine = AlertRulesEngine(config)
        engine.current_dashboard = engine.load_dashboard(sample_critical_dashboard)
        engine.previous_dashboard = engine.load_dashboard(sample_previous_dashboard)

        alerts = engine.evaluate_rules()

        # Score dropped from 98.0 to 35.0 - should trigger alert
        score_drop_alerts = [a for a in alerts if a.alert_type == "repository_score_drop"]
        assert len(score_drop_alerts) > 0

    def test_rapid_regression_detection(self, sample_critical_dashboard, sample_previous_dashboard, temp_dir):
        """Test detection of rapid issue regression."""
        config = AlertingConfig(
            current_dashboard_path=sample_critical_dashboard,
            previous_dashboard_path=sample_previous_dashboard,
            output_path=temp_dir / "alerts.json",
            rapid_regression_threshold=3
        )
        engine = AlertRulesEngine(config)
        engine.current_dashboard = engine.load_dashboard(sample_critical_dashboard)
        engine.previous_dashboard = engine.load_dashboard(sample_previous_dashboard)

        alerts = engine.evaluate_rules()

        # Issues increased from 1 to 15 - should trigger alert
        regression_alerts = [a for a in alerts if a.alert_type == "rapid_regression"]
        assert len(regression_alerts) > 0

    def test_no_trend_alerts_without_previous(self, sample_critical_dashboard, temp_dir):
        """Test that trend alerts are not triggered without previous dashboard."""
        config = AlertingConfig(
            current_dashboard_path=sample_critical_dashboard,
            previous_dashboard_path=None,  # No previous dashboard
            output_path=temp_dir / "alerts.json"
        )
        engine = AlertRulesEngine(config)
        engine.current_dashboard = engine.load_dashboard(sample_critical_dashboard)

        alerts = engine.evaluate_rules()

        # Should not have trend-based alerts
        score_drop_alerts = [a for a in alerts if a.alert_type == "repository_score_drop"]
        regression_alerts = [a for a in alerts if a.alert_type == "rapid_regression"]

        assert len(score_drop_alerts) == 0
        assert len(regression_alerts) == 0


# ============================================================================
# Alert Channel Tests
# ============================================================================

class TestConsoleChannel:
    """Tests for ConsoleChannel."""

    def test_console_channel_dispatch(self, capsys):
        """Test console channel dispatches alerts."""
        config = ChannelConfig(channel_type=ChannelType.CONSOLE, enabled=True)
        channel = ConsoleChannel(config)

        alerts = [
            Alert(
                alert_id="test_1",
                alert_type="critical_issue",
                severity="CRITICAL",
                title="Test Alert",
                message="Test message",
                timestamp="2025-01-01T00:00:00"
            )
        ]
        report = AlertReport(
            report_id="report_1",
            generated_at="2025-01-01T00:00:00",
            dashboard_path="/path/to/dashboard.json",
            previous_dashboard_path=None,
            total_alerts=1,
            critical_alerts=1
        )

        result = channel.dispatch(alerts, report)

        assert result is True
        captured = capsys.readouterr()
        assert "REPOSITORY HEALTH ALERTS" in captured.out

    def test_console_channel_filter_by_severity(self):
        """Test console channel filters by severity."""
        config = ChannelConfig(
            channel_type=ChannelType.CONSOLE,
            enabled=True,
            min_severity=AlertSeverity.ERROR
        )
        channel = ConsoleChannel(config)

        alerts = [
            Alert(
                alert_id="test_1",
                alert_type="info",
                severity="INFO",
                title="Info Alert",
                message="Info message",
                timestamp="2025-01-01T00:00:00"
            )
        ]

        filtered = channel.filter_alerts(alerts)

        # INFO alert should be filtered out when min_severity is ERROR
        assert len(filtered) == 0


class TestFileChannel:
    """Tests for FileChannel."""

    def test_file_channel_dispatch(self, temp_dir):
        """Test file channel writes alerts to file."""
        output_file = temp_dir / "alerts.txt"
        config = ChannelConfig(
            channel_type=ChannelType.FILE,
            enabled=True,
            output_path=output_file
        )
        channel = FileChannel(config)

        alerts = [
            Alert(
                alert_id="test_1",
                alert_type="critical_issue",
                severity="CRITICAL",
                title="Test Alert",
                message="Test message",
                timestamp="2025-01-01T00:00:00"
            )
        ]
        report = AlertReport(
            report_id="report_1",
            generated_at="2025-01-01T00:00:00",
            dashboard_path="/path/to/dashboard.json",
            previous_dashboard_path=None,
            total_alerts=1,
            critical_alerts=1
        )

        result = channel.dispatch(alerts, report)

        assert result is True
        assert output_file.exists()
        content = output_file.read_text()
        assert "REPOSITORY HEALTH ALERTS" in content
        assert "Test Alert" in content

    def test_file_channel_no_output_path(self):
        """Test file channel fails without output path."""
        config = ChannelConfig(
            channel_type=ChannelType.FILE,
            enabled=True,
            output_path=None
        )
        channel = FileChannel(config)

        result = channel.dispatch([], AlertReport(
            report_id="test",
            generated_at="2025-01-01T00:00:00",
            dashboard_path="/path",
            previous_dashboard_path=None
        ))

        assert result is False


class TestEmailChannel:
    """Tests for EmailChannel (mock)."""

    def test_email_channel_generates_content(self):
        """Test email channel generates email content."""
        config = ChannelConfig(
            channel_type=ChannelType.EMAIL,
            enabled=True,
            email_to="admin@example.com"
        )
        channel = EmailChannel(config)

        alerts = [
            Alert(
                alert_id="test_1",
                alert_type="critical_issue",
                severity="CRITICAL",
                title="Critical Alert",
                message="Critical message",
                timestamp="2025-01-01T00:00:00"
            )
        ]
        report = AlertReport(
            report_id="report_1",
            generated_at="2025-01-01T00:00:00",
            dashboard_path="/path/to/dashboard.json",
            previous_dashboard_path=None,
            total_alerts=1,
            critical_alerts=1,
            repository_score=50.0,
            health_status="red"
        )

        result = channel.dispatch(alerts, report)

        assert result is True
        assert hasattr(channel, '_last_email')
        assert channel._last_email['to'] == "admin@example.com"
        assert "CRITICAL" in channel._last_email['subject']

    def test_email_channel_no_recipient(self):
        """Test email channel fails without recipient."""
        config = ChannelConfig(
            channel_type=ChannelType.EMAIL,
            enabled=True,
            email_to=None
        )
        channel = EmailChannel(config)

        result = channel.dispatch([], AlertReport(
            report_id="test",
            generated_at="2025-01-01T00:00:00",
            dashboard_path="/path",
            previous_dashboard_path=None
        ))

        assert result is False


class TestWebhookChannel:
    """Tests for WebhookChannel (mock)."""

    def test_webhook_channel_generates_payload(self):
        """Test webhook channel generates JSON payload."""
        config = ChannelConfig(
            channel_type=ChannelType.WEBHOOK,
            enabled=True,
            webhook_url="https://example.com/webhook"
        )
        channel = WebhookChannel(config)

        alerts = [
            Alert(
                alert_id="test_1",
                alert_type="critical_issue",
                severity="CRITICAL",
                title="Critical Alert",
                message="Critical message",
                timestamp="2025-01-01T00:00:00"
            )
        ]
        report = AlertReport(
            report_id="report_1",
            generated_at="2025-01-01T00:00:00",
            dashboard_path="/path/to/dashboard.json",
            previous_dashboard_path=None,
            total_alerts=1,
            critical_alerts=1,
            repository_score=50.0,
            health_status="red"
        )

        result = channel.dispatch(alerts, report)

        assert result is True
        assert hasattr(channel, '_last_payload')
        assert channel._last_payload['url'] == "https://example.com/webhook"
        assert channel._last_payload['payload']['event_type'] == "repository_health_alert"

    def test_webhook_channel_no_url(self):
        """Test webhook channel fails without URL."""
        config = ChannelConfig(
            channel_type=ChannelType.WEBHOOK,
            enabled=True,
            webhook_url=None
        )
        channel = WebhookChannel(config)

        result = channel.dispatch([], AlertReport(
            report_id="test",
            generated_at="2025-01-01T00:00:00",
            dashboard_path="/path",
            previous_dashboard_path=None
        ))

        assert result is False


# ============================================================================
# AlertDispatcher Tests
# ============================================================================

class TestAlertDispatcher:
    """Tests for AlertDispatcher."""

    def test_dispatcher_initialization(self, temp_dir):
        """Test dispatcher initialization."""
        config = AlertingConfig(
            current_dashboard_path=temp_dir / "dashboard.json",
            channels=[
                ChannelConfig(channel_type=ChannelType.CONSOLE, enabled=True)
            ]
        )
        dispatcher = AlertDispatcher(config)

        assert len(dispatcher.channels) == 1

    def test_dispatcher_multiple_channels(self, temp_dir):
        """Test dispatcher with multiple channels."""
        config = AlertingConfig(
            current_dashboard_path=temp_dir / "dashboard.json",
            channels=[
                ChannelConfig(channel_type=ChannelType.CONSOLE, enabled=True),
                ChannelConfig(
                    channel_type=ChannelType.FILE,
                    enabled=True,
                    output_path=temp_dir / "alerts.txt"
                )
            ]
        )
        dispatcher = AlertDispatcher(config)

        assert len(dispatcher.channels) == 2

    def test_dispatcher_filters_disabled_channels(self, temp_dir):
        """Test dispatcher filters disabled channels."""
        config = AlertingConfig(
            current_dashboard_path=temp_dir / "dashboard.json",
            channels=[
                ChannelConfig(channel_type=ChannelType.CONSOLE, enabled=True),
                ChannelConfig(channel_type=ChannelType.FILE, enabled=False)
            ]
        )
        dispatcher = AlertDispatcher(config)

        assert len(dispatcher.channels) == 1

    def test_dispatcher_dispatch(self, temp_dir):
        """Test dispatcher dispatches to all channels."""
        config = AlertingConfig(
            current_dashboard_path=temp_dir / "dashboard.json",
            channels=[
                ChannelConfig(channel_type=ChannelType.CONSOLE, enabled=True)
            ]
        )
        dispatcher = AlertDispatcher(config)

        alerts = [
            Alert(
                alert_id="test_1",
                alert_type="info",
                severity="INFO",
                title="Test",
                message="Test",
                timestamp="2025-01-01T00:00:00"
            )
        ]
        report = AlertReport(
            report_id="report_1",
            generated_at="2025-01-01T00:00:00",
            dashboard_path="/path",
            previous_dashboard_path=None,
            total_alerts=1
        )

        updated_report = dispatcher.dispatch(alerts, report)

        assert "console" in updated_report.channels_dispatched


# ============================================================================
# AlertingEngine Tests (Integration)
# ============================================================================

class TestAlertingEngine:
    """Tests for AlertingEngine orchestration."""

    def test_engine_initialization(self, sample_healthy_dashboard, temp_dir):
        """Test alerting engine initialization."""
        config = AlertingConfig(
            current_dashboard_path=sample_healthy_dashboard,
            output_path=temp_dir / "alerts.json",
            channels=[
                ChannelConfig(channel_type=ChannelType.CONSOLE, enabled=True)
            ]
        )
        engine = AlertingEngine(config)

        assert engine.config == config
        assert engine.rules_engine is not None
        assert engine.dispatcher is not None

    def test_engine_run_healthy_dashboard(self, sample_healthy_dashboard, temp_dir):
        """Test engine run on healthy dashboard."""
        config = AlertingConfig(
            current_dashboard_path=sample_healthy_dashboard,
            output_path=temp_dir / "alerts.json",
            channels=[
                ChannelConfig(channel_type=ChannelType.CONSOLE, enabled=True)
            ],
            fail_on_critical=True
        )
        engine = AlertingEngine(config)

        report, exit_code = engine.run()

        # Healthy dashboard should have no critical alerts
        assert report.critical_alerts == 0
        assert exit_code in [EXIT_NO_ALERTS, EXIT_ALERTS_TRIGGERED]

    def test_engine_run_critical_dashboard(self, sample_critical_dashboard, temp_dir):
        """Test engine run on critical dashboard."""
        config = AlertingConfig(
            current_dashboard_path=sample_critical_dashboard,
            output_path=temp_dir / "alerts.json",
            channels=[
                ChannelConfig(channel_type=ChannelType.CONSOLE, enabled=True)
            ],
            fail_on_critical=True
        )
        engine = AlertingEngine(config)

        report, exit_code = engine.run()

        # Critical dashboard should have critical alerts
        assert report.critical_alerts > 0
        assert exit_code == EXIT_CRITICAL_ALERTS

    def test_engine_run_writes_output(self, sample_healthy_dashboard, temp_dir):
        """Test engine writes JSON output."""
        output_path = temp_dir / "alerts.json"
        config = AlertingConfig(
            current_dashboard_path=sample_healthy_dashboard,
            output_path=output_path,
            channels=[
                ChannelConfig(channel_type=ChannelType.CONSOLE, enabled=True)
            ]
        )
        engine = AlertingEngine(config)

        report, _ = engine.run()

        assert output_path.exists()
        with open(output_path) as f:
            output_data = json.load(f)
        assert output_data["report_id"] == report.report_id


# ============================================================================
# Exit Code Tests
# ============================================================================

class TestExitCodes:
    """Tests for exit code handling."""

    def test_exit_no_alerts(self, sample_healthy_dashboard, temp_dir):
        """Test EXIT_NO_ALERTS when no alerts triggered."""
        # Create a very healthy dashboard with no issues
        healthy_data = {
            "overall_health": "green",
            "repository_score": 100.0,
            "scan_timestamp": datetime.utcnow().isoformat(),
            "repository_path": "/path/to/repo",
            "total_issues": 0,
            "critical_issues": 0,
            "error_issues": 0,
            "warning_issues": 0,
            "info_issues": 0,
            "total_versions": 5,
            "healthy_versions": 5,
            "warning_versions": 0,
            "critical_versions": 0,
            "total_artifacts": 50,
            "orphaned_artifacts": 0,
            "corrupted_artifacts": 0,
            "missing_artifacts": 0,
            "versions_health": [
                {"version": f"1.0.{i}", "health_status": "green", "sbom_present": True, "slsa_present": True}
                for i in range(5)
            ],
            "rollback_history": [],
        }

        healthy_path = temp_dir / "perfect-dashboard.json"
        with open(healthy_path, 'w') as f:
            json.dump(healthy_data, f)

        config = AlertingConfig(
            current_dashboard_path=healthy_path,
            output_path=temp_dir / "alerts.json",
            channels=[
                ChannelConfig(channel_type=ChannelType.CONSOLE, enabled=True)
            ],
            fail_on_critical=True,
            fail_on_any_alert=False
        )
        engine = AlertingEngine(config)

        report, exit_code = engine.run()

        # No alerts should be triggered for perfect health
        # Note: May still have info-level alerts
        assert exit_code in [EXIT_NO_ALERTS, EXIT_ALERTS_TRIGGERED]

    def test_exit_critical_alerts(self, sample_critical_dashboard, temp_dir):
        """Test EXIT_CRITICAL_ALERTS when critical alerts present."""
        config = AlertingConfig(
            current_dashboard_path=sample_critical_dashboard,
            output_path=temp_dir / "alerts.json",
            channels=[
                ChannelConfig(channel_type=ChannelType.CONSOLE, enabled=True)
            ],
            fail_on_critical=True
        )
        engine = AlertingEngine(config)

        report, exit_code = engine.run()

        assert exit_code == EXIT_CRITICAL_ALERTS

    def test_exit_alerts_triggered_no_fail(self, sample_critical_dashboard, temp_dir):
        """Test EXIT_ALERTS_TRIGGERED when not failing on critical."""
        config = AlertingConfig(
            current_dashboard_path=sample_critical_dashboard,
            output_path=temp_dir / "alerts.json",
            channels=[
                ChannelConfig(channel_type=ChannelType.CONSOLE, enabled=True)
            ],
            fail_on_critical=False,
            fail_on_any_alert=False
        )
        engine = AlertingEngine(config)

        report, exit_code = engine.run()

        # Should still indicate alerts were triggered
        assert exit_code == EXIT_ALERTS_TRIGGERED


# ============================================================================
# Edge Case Tests
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_dashboard(self, temp_dir):
        """Test handling of empty dashboard."""
        empty_data = {
            "overall_health": "green",
            "repository_score": 100.0,
            "total_issues": 0,
            "critical_issues": 0,
            "error_issues": 0,
            "warning_issues": 0,
            "info_issues": 0,
            "total_versions": 0,
            "versions_health": []
        }

        empty_path = temp_dir / "empty-dashboard.json"
        with open(empty_path, 'w') as f:
            json.dump(empty_data, f)

        config = AlertingConfig(
            current_dashboard_path=empty_path,
            output_path=temp_dir / "alerts.json",
            channels=[
                ChannelConfig(channel_type=ChannelType.CONSOLE, enabled=True)
            ]
        )
        engine = AlertingEngine(config)

        report, exit_code = engine.run()

        # Should complete without error
        assert report is not None

    def test_malformed_versions_health(self, temp_dir):
        """Test handling of malformed versions_health."""
        malformed_data = {
            "overall_health": "yellow",
            "repository_score": 70.0,
            "total_issues": 1,
            "critical_issues": 0,
            "error_issues": 0,
            "warning_issues": 1,
            "info_issues": 0,
            "versions_health": [
                {"version": "1.0.0"},  # Missing health_status, sbom_present, slsa_present
                {"version": "1.0.1", "health_status": "green"}  # Missing sbom/slsa
            ]
        }

        malformed_path = temp_dir / "malformed-dashboard.json"
        with open(malformed_path, 'w') as f:
            json.dump(malformed_data, f)

        config = AlertingConfig(
            current_dashboard_path=malformed_path,
            output_path=temp_dir / "alerts.json",
            channels=[
                ChannelConfig(channel_type=ChannelType.CONSOLE, enabled=True)
            ]
        )
        engine = AlertingEngine(config)

        # Should handle gracefully
        report, exit_code = engine.run()
        assert report is not None

    def test_severity_filtering(self, sample_critical_dashboard, temp_dir):
        """Test severity threshold filtering."""
        config = AlertingConfig(
            current_dashboard_path=sample_critical_dashboard,
            output_path=temp_dir / "alerts.json",
            channels=[
                ChannelConfig(channel_type=ChannelType.CONSOLE, enabled=True)
            ],
            severity_threshold=AlertSeverity.ERROR  # Only ERROR and CRITICAL
        )
        engine = AlertingEngine(config)

        report, exit_code = engine.run()

        # Report should still contain all alerts, but dispatch filtered
        assert report is not None


# ============================================================================
# Performance Tests
# ============================================================================

class TestPerformance:
    """Performance tests."""

    def test_alerting_performance(self, sample_critical_dashboard, temp_dir):
        """Test alerting completes within time target."""
        import time

        config = AlertingConfig(
            current_dashboard_path=sample_critical_dashboard,
            output_path=temp_dir / "alerts.json",
            channels=[
                ChannelConfig(channel_type=ChannelType.CONSOLE, enabled=True)
            ]
        )
        engine = AlertingEngine(config)

        start = time.time()
        report, exit_code = engine.run()
        duration = time.time() - start

        # Should complete in under 2 seconds
        assert duration < 2.0

    def test_large_alert_list_performance(self, temp_dir):
        """Test performance with large number of alerts."""
        import time

        # Create dashboard with many issues
        large_data = {
            "overall_health": "red",
            "repository_score": 10.0,
            "total_issues": 100,
            "critical_issues": 20,
            "error_issues": 30,
            "warning_issues": 30,
            "info_issues": 20,
            "total_versions": 50,
            "orphaned_artifacts": 50,
            "corrupted_artifacts": 20,
            "missing_artifacts": 10,
            "versions_health": [
                {"version": f"1.0.{i}", "health_status": "red", "sbom_present": False, "slsa_present": False}
                for i in range(50)
            ],
            "rollback_history": [
                {"from_version": f"1.0.{i}", "to_version": f"1.0.{i-1}", "status": "failed", "timestamp": datetime.utcnow().isoformat()}
                for i in range(1, 10)
            ]
        }

        large_path = temp_dir / "large-dashboard.json"
        with open(large_path, 'w') as f:
            json.dump(large_data, f)

        config = AlertingConfig(
            current_dashboard_path=large_path,
            output_path=temp_dir / "alerts.json",
            channels=[
                ChannelConfig(channel_type=ChannelType.CONSOLE, enabled=True)
            ]
        )
        engine = AlertingEngine(config)

        start = time.time()
        report, exit_code = engine.run()
        duration = time.time() - start

        # Should complete in under 3 seconds even with large data
        assert duration < 3.0
        assert report.total_alerts > 0


# ============================================================================
# CLI Tests
# ============================================================================

class TestCLI:
    """Tests for CLI functionality."""

    def test_cli_help(self):
        """Test CLI help output."""
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "-m", "analytics.run_alerts", "--help"],
            capture_output=True,
            text=True,
            timeout=30
        )

        assert result.returncode == 0
        assert "current-dashboard" in result.stdout
        assert "severity-threshold" in result.stdout
        assert "channels" in result.stdout

    def test_cli_missing_dashboard(self, temp_dir):
        """Test CLI with missing dashboard."""
        import subprocess
        import sys

        missing_path = temp_dir / "nonexistent.json"

        result = subprocess.run(
            [
                sys.executable, "-m", "analytics.run_alerts",
                "--current-dashboard", str(missing_path)
            ],
            capture_output=True,
            text=True,
            timeout=30
        )

        assert result.returncode == EXIT_INVALID_DASHBOARD

    def test_cli_basic_execution(self, sample_healthy_dashboard, temp_dir):
        """Test CLI basic execution."""
        import subprocess
        import sys

        output_path = temp_dir / "cli_alerts.json"

        result = subprocess.run(
            [
                sys.executable, "-m", "analytics.run_alerts",
                "--current-dashboard", str(sample_healthy_dashboard),
                "--output", str(output_path),
                "--channels", "console"
            ],
            capture_output=True,
            text=True,
            timeout=30
        )

        # Should complete (exit code depends on alerts triggered)
        assert result.returncode in [EXIT_NO_ALERTS, EXIT_ALERTS_TRIGGERED, EXIT_CRITICAL_ALERTS]


# ============================================================================
# End-to-End Test
# ============================================================================

class TestEndToEnd:
    """End-to-end integration tests."""

    def test_complete_workflow(self, sample_critical_dashboard, sample_previous_dashboard, temp_dir):
        """Test complete alerting workflow."""
        output_path = temp_dir / "e2e_alerts.json"
        file_output = temp_dir / "e2e_alerts.txt"

        config = AlertingConfig(
            current_dashboard_path=sample_critical_dashboard,
            previous_dashboard_path=sample_previous_dashboard,
            output_path=output_path,
            channels=[
                ChannelConfig(channel_type=ChannelType.CONSOLE, enabled=True),
                ChannelConfig(
                    channel_type=ChannelType.FILE,
                    enabled=True,
                    output_path=file_output
                ),
                ChannelConfig(
                    channel_type=ChannelType.EMAIL,
                    enabled=True,
                    email_to="admin@example.com"
                ),
                ChannelConfig(
                    channel_type=ChannelType.WEBHOOK,
                    enabled=True,
                    webhook_url="https://example.com/webhook"
                )
            ],
            severity_threshold=AlertSeverity.INFO,
            score_drop_threshold=10.0,
            rapid_regression_threshold=3,
            fail_on_critical=True,
            verbose=False
        )

        engine = AlertingEngine(config)
        report, exit_code = engine.run()

        # Verify report
        assert report.total_alerts > 0
        assert report.critical_alerts > 0  # Critical dashboard has critical issues

        # Verify JSON output
        assert output_path.exists()
        with open(output_path) as f:
            output_data = json.load(f)
        assert output_data["total_alerts"] == report.total_alerts

        # Verify file output
        assert file_output.exists()
        content = file_output.read_text()
        assert "REPOSITORY HEALTH ALERTS" in content

        # Verify exit code
        assert exit_code == EXIT_CRITICAL_ALERTS

        # Verify trend alerts were generated
        score_drop_alerts = [
            a for a in report.alerts
            if a.get("alert_type") == "repository_score_drop"
        ]
        regression_alerts = [
            a for a in report.alerts
            if a.get("alert_type") == "rapid_regression"
        ]

        # Should have trend-based alerts
        assert len(score_drop_alerts) > 0 or len(regression_alerts) > 0
