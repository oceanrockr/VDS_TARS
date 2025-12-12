"""
Integration Tests for Organization-Level Alerting & Escalation Engine

Tests the org-level alerting, escalation rules, and routing functionality.

Test Coverage:
- Alert generation from SLO violations
- Alert generation from high-risk repos
- Alert generation from trend signals
- Alert generation from integrity issues
- Alert severity scoring
- Escalation rule matching
- Escalation action execution (stubs)
- Routing to channels
- CLI behavior and exit codes
- Edge cases (no alerts, all critical, mixed)

Version: 1.0.0
Phase: 14.8 Task 2
"""

import json
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import patch, MagicMock

import pytest

# Import the modules under test
from analytics.org_alerting_engine import (
    OrgAlertConfig,
    OrgAlertingEngine,
    OrgAlertGenerator,
    OrgAlertDispatcher,
    EscalationEngine,
    OrgAlertReport,
    OrgAlert,
    OrgAlertCategory,
    AlertSeverity,
    OrgAlertChannelConfig,
    OrgAlertChannelType,
    OrgAlertThresholds,
    EscalationRule,
    EscalationCondition,
    EscalationAction,
    EscalationActionType,
    ConsoleOrgAlertChannel,
    JsonFileOrgAlertChannel,
    StdoutOrgAlertChannel,
    EmailOrgAlertChannel,
    SlackOrgAlertChannel,
    WebhookOrgAlertChannel,
    create_default_escalation_rules,
    create_default_channels,
    load_escalation_config,
    EXIT_ORG_ALERT_SUCCESS,
    EXIT_ALERTS_PRESENT,
    EXIT_CRITICAL_ALERTS,
    EXIT_ALERTING_CONFIG_ERROR,
    EXIT_ORG_REPORT_PARSE_ERROR,
    EXIT_ROUTING_FAILURE,
    EXIT_GENERAL_ALERTING_ERROR,
    OrgAlertConfigError,
    OrgReportParseError,
    OrgRoutingError,
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test artifacts."""
    temp = tempfile.mkdtemp(prefix="org_alert_test_")
    yield Path(temp)
    shutil.rmtree(temp, ignore_errors=True)


@pytest.fixture
def sample_org_report() -> Dict[str, Any]:
    """Sample org-health-report.json data."""
    return {
        "report_id": "org_health_test",
        "generated_at": datetime.utcnow().isoformat(),
        "root_dir": "/test/org-health",
        "org_health_status": "yellow",
        "org_health_score": 75.0,
        "org_risk_tier": "medium",
        "metrics": {
            "total_repos": 10,
            "repos_green": 6,
            "repos_yellow": 3,
            "repos_red": 1,
            "repos_low_risk": 5,
            "repos_medium_risk": 3,
            "repos_high_risk": 1,
            "repos_critical_risk": 1,
            "repos_improving": 3,
            "repos_stable": 5,
            "repos_declining": 2,
            "avg_score": 75.0,
            "min_score": 45.0,
            "max_score": 95.0,
            "total_issues": 50,
            "total_critical_issues": 5,
            "percent_green": 60.0,
            "percent_improving": 30.0
        },
        "total_slos": 3,
        "slos_satisfied": 2,
        "slos_violated": 1,
        "slo_results": [
            {
                "slo_id": "org-percent-green",
                "slo_description": "At least 80% of repos should be GREEN",
                "satisfied": False,
                "current_value": 0.60,
                "target_value": 0.80,
                "operator": ">=",
                "repos_evaluated": 10,
                "violating_repos": ["repo-c", "repo-d", "repo-e", "repo-f"],
                "violation_severity": "medium"
            },
            {
                "slo_id": "min-score",
                "slo_description": "Minimum score should be >= 60",
                "satisfied": True,
                "current_value": 65.0,
                "target_value": 60.0,
                "operator": ">=",
                "repos_evaluated": 10,
                "violating_repos": []
            }
        ],
        "top_risk_repos": [
            {
                "repo_id": "repo-x",
                "repo_name": "Legacy Service",
                "risk_tier": "critical",
                "risk_score": 85.0,
                "health_status": "red",
                "repository_score": 35.0,
                "trend_direction": "declining",
                "critical_issues": 8,
                "critical_alerts": 2,
                "reason_codes": ["health_red", "critical_issues:8", "declining_trend"]
            },
            {
                "repo_id": "repo-y",
                "repo_name": "Auth Service",
                "risk_tier": "high",
                "risk_score": 65.0,
                "health_status": "yellow",
                "repository_score": 55.0,
                "trend_direction": "stable",
                "critical_issues": 3,
                "critical_alerts": 0,
                "reason_codes": ["critical_issues:3", "low_score:55"]
            }
        ],
        "repositories": [],
        "recommendations": [],
        "repos_discovered": 10,
        "repos_loaded": 10,
        "repos_failed": 0,
        "load_errors": []
    }


@pytest.fixture
def sample_org_report_healthy() -> Dict[str, Any]:
    """Sample org report with no issues."""
    return {
        "report_id": "org_health_healthy",
        "generated_at": datetime.utcnow().isoformat(),
        "root_dir": "/test/org-health",
        "org_health_status": "green",
        "org_health_score": 92.0,
        "org_risk_tier": "low",
        "metrics": {
            "total_repos": 5,
            "repos_green": 5,
            "repos_yellow": 0,
            "repos_red": 0,
            "repos_low_risk": 5,
            "repos_medium_risk": 0,
            "repos_high_risk": 0,
            "repos_critical_risk": 0,
            "repos_improving": 2,
            "repos_stable": 3,
            "repos_declining": 0,
            "avg_score": 92.0,
            "min_score": 85.0,
            "max_score": 98.0,
            "total_issues": 5,
            "total_critical_issues": 0,
            "percent_green": 100.0,
            "percent_improving": 40.0
        },
        "total_slos": 2,
        "slos_satisfied": 2,
        "slos_violated": 0,
        "slo_results": [],
        "top_risk_repos": [],
        "repositories": [],
        "recommendations": [],
        "repos_discovered": 5,
        "repos_loaded": 5,
        "repos_failed": 0,
        "load_errors": []
    }


@pytest.fixture
def sample_org_report_critical() -> Dict[str, Any]:
    """Sample org report with critical issues."""
    return {
        "report_id": "org_health_critical",
        "generated_at": datetime.utcnow().isoformat(),
        "root_dir": "/test/org-health",
        "org_health_status": "red",
        "org_health_score": 35.0,
        "org_risk_tier": "critical",
        "metrics": {
            "total_repos": 5,
            "repos_green": 0,
            "repos_yellow": 1,
            "repos_red": 4,
            "repos_low_risk": 0,
            "repos_medium_risk": 1,
            "repos_high_risk": 2,
            "repos_critical_risk": 2,
            "repos_improving": 0,
            "repos_stable": 1,
            "repos_declining": 4,
            "avg_score": 35.0,
            "min_score": 15.0,
            "max_score": 55.0,
            "total_issues": 100,
            "total_critical_issues": 25,
            "percent_green": 0.0,
            "percent_improving": 0.0
        },
        "total_slos": 3,
        "slos_satisfied": 0,
        "slos_violated": 3,
        "slo_results": [
            {
                "slo_id": "all-green",
                "slo_description": "All repos should be GREEN",
                "satisfied": False,
                "current_value": 0.0,
                "target_value": 1.0,
                "operator": "==",
                "repos_evaluated": 5,
                "violating_repos": ["repo-a", "repo-b", "repo-c", "repo-d", "repo-e"],
                "violation_severity": "critical"
            }
        ],
        "top_risk_repos": [
            {
                "repo_id": "repo-critical-1",
                "repo_name": "Critical Service 1",
                "risk_tier": "critical",
                "risk_score": 95.0,
                "health_status": "red",
                "repository_score": 15.0,
                "trend_direction": "declining",
                "critical_issues": 15,
                "critical_alerts": 5,
                "reason_codes": ["health_red", "critical_issues:15"]
            },
            {
                "repo_id": "repo-critical-2",
                "repo_name": "Critical Service 2",
                "risk_tier": "critical",
                "risk_score": 90.0,
                "health_status": "red",
                "repository_score": 20.0,
                "trend_direction": "declining",
                "critical_issues": 10,
                "critical_alerts": 3,
                "reason_codes": ["health_red", "critical_issues:10"]
            }
        ],
        "repositories": [],
        "recommendations": [],
        "repos_discovered": 5,
        "repos_loaded": 5,
        "repos_failed": 0,
        "load_errors": []
    }


@pytest.fixture
def sample_org_report_with_errors() -> Dict[str, Any]:
    """Sample org report with load errors."""
    return {
        "report_id": "org_health_errors",
        "generated_at": datetime.utcnow().isoformat(),
        "root_dir": "/test/org-health",
        "org_health_status": "yellow",
        "org_health_score": 70.0,
        "org_risk_tier": "medium",
        "metrics": {
            "total_repos": 3,
            "repos_green": 2,
            "repos_yellow": 1,
            "repos_red": 0,
            "repos_low_risk": 2,
            "repos_medium_risk": 1,
            "repos_high_risk": 0,
            "repos_critical_risk": 0,
            "repos_improving": 1,
            "repos_stable": 2,
            "repos_declining": 0,
            "avg_score": 70.0,
            "min_score": 60.0,
            "max_score": 80.0,
            "total_issues": 10,
            "total_critical_issues": 1,
            "percent_green": 66.7,
            "percent_improving": 33.3
        },
        "total_slos": 1,
        "slos_satisfied": 1,
        "slos_violated": 0,
        "slo_results": [],
        "top_risk_repos": [],
        "repositories": [],
        "recommendations": [],
        "repos_discovered": 5,
        "repos_loaded": 3,
        "repos_failed": 2,
        "load_errors": [
            {"repo_id": "repo-fail-1", "error": "Invalid JSON in dashboard"},
            {"repo_id": "repo-fail-2", "error": "Missing health-dashboard.json"}
        ]
    }


def create_org_report_file(temp_dir: Path, report_data: Dict[str, Any]) -> Path:
    """Create an org health report JSON file."""
    report_path = temp_dir / "org-health-report.json"
    with open(report_path, 'w') as f:
        json.dump(report_data, f)
    return report_path


# ============================================================================
# Test: Alert Generation - SLO Violations
# ============================================================================

class TestSloAlertGeneration:
    """Tests for SLO violation alert generation."""

    def test_generates_alert_for_violated_slo(self, temp_dir, sample_org_report):
        """Test that violated SLO generates an alert."""
        report_path = create_org_report_file(temp_dir, sample_org_report)
        config = OrgAlertConfig(
            org_report_path=report_path,
            generate_slo_alerts=True,
            generate_risk_alerts=False,
            generate_trend_alerts=False,
            generate_integrity_alerts=False
        )

        generator = OrgAlertGenerator(config)
        alerts = generator.generate_all_alerts(sample_org_report)

        # Should have at least one SLO alert
        slo_alerts = [a for a in alerts if a.category == OrgAlertCategory.SLO]
        assert len(slo_alerts) >= 1

        # Check alert content
        alert = slo_alerts[0]
        assert alert.slo_id == "org-percent-green"
        assert alert.current_value == 0.60
        assert alert.target_value == 0.80
        assert len(alert.violating_repos) == 4

    def test_no_slo_alerts_when_all_satisfied(self, temp_dir, sample_org_report_healthy):
        """Test that no SLO alerts when all SLOs satisfied."""
        report_path = create_org_report_file(temp_dir, sample_org_report_healthy)
        config = OrgAlertConfig(
            org_report_path=report_path,
            generate_slo_alerts=True,
            generate_risk_alerts=False,
            generate_trend_alerts=False,
            generate_integrity_alerts=False
        )

        generator = OrgAlertGenerator(config)
        alerts = generator.generate_all_alerts(sample_org_report_healthy)

        slo_alerts = [a for a in alerts if a.category == OrgAlertCategory.SLO]
        assert len(slo_alerts) == 0

    def test_slo_alert_severity_from_config(self, temp_dir, sample_org_report_critical):
        """Test that SLO alert severity comes from SLO violation_severity."""
        report_path = create_org_report_file(temp_dir, sample_org_report_critical)
        config = OrgAlertConfig(
            org_report_path=report_path,
            generate_slo_alerts=True,
            generate_risk_alerts=False,
            generate_trend_alerts=False,
            generate_integrity_alerts=False
        )

        generator = OrgAlertGenerator(config)
        alerts = generator.generate_all_alerts(sample_org_report_critical)

        slo_alerts = [a for a in alerts if a.category == OrgAlertCategory.SLO]
        assert len(slo_alerts) >= 1
        # The critical SLO should have CRITICAL severity
        critical_slo_alerts = [a for a in slo_alerts if a.severity == AlertSeverity.CRITICAL]
        assert len(critical_slo_alerts) >= 1

    def test_disabled_slo_alerts(self, temp_dir, sample_org_report):
        """Test that SLO alerts can be disabled."""
        report_path = create_org_report_file(temp_dir, sample_org_report)
        config = OrgAlertConfig(
            org_report_path=report_path,
            generate_slo_alerts=False
        )

        generator = OrgAlertGenerator(config)
        alerts = generator.generate_all_alerts(sample_org_report)

        slo_alerts = [a for a in alerts if a.category == OrgAlertCategory.SLO]
        assert len(slo_alerts) == 0


# ============================================================================
# Test: Alert Generation - Risk Alerts
# ============================================================================

class TestRiskAlertGeneration:
    """Tests for high-risk repository alert generation."""

    def test_generates_alert_for_critical_risk_repo(self, temp_dir, sample_org_report):
        """Test that critical risk repo generates alert."""
        report_path = create_org_report_file(temp_dir, sample_org_report)
        config = OrgAlertConfig(
            org_report_path=report_path,
            generate_slo_alerts=False,
            generate_risk_alerts=True,
            generate_trend_alerts=False,
            generate_integrity_alerts=False
        )

        generator = OrgAlertGenerator(config)
        alerts = generator.generate_all_alerts(sample_org_report)

        risk_alerts = [a for a in alerts if a.category == OrgAlertCategory.RISK]
        assert len(risk_alerts) >= 1

        # Check for critical risk repo alert
        critical_alerts = [a for a in risk_alerts if a.severity == AlertSeverity.CRITICAL]
        assert len(critical_alerts) >= 1
        assert "repo-x" in critical_alerts[0].affected_repos

    def test_generates_alert_for_high_risk_repo(self, temp_dir, sample_org_report):
        """Test that high risk repo generates alert."""
        report_path = create_org_report_file(temp_dir, sample_org_report)
        config = OrgAlertConfig(
            org_report_path=report_path,
            generate_slo_alerts=False,
            generate_risk_alerts=True,
            generate_trend_alerts=False,
            generate_integrity_alerts=False
        )

        generator = OrgAlertGenerator(config)
        alerts = generator.generate_all_alerts(sample_org_report)

        risk_alerts = [a for a in alerts if a.category == OrgAlertCategory.RISK]
        high_alerts = [a for a in risk_alerts if a.severity == AlertSeverity.HIGH]
        # Should have at least one high risk alert
        assert len(high_alerts) >= 1 or any(a.risk_tier == "high" for a in risk_alerts)

    def test_no_risk_alerts_for_low_risk_repos(self, temp_dir, sample_org_report_healthy):
        """Test that no risk alerts for healthy repos."""
        report_path = create_org_report_file(temp_dir, sample_org_report_healthy)
        config = OrgAlertConfig(
            org_report_path=report_path,
            generate_slo_alerts=False,
            generate_risk_alerts=True,
            generate_trend_alerts=False,
            generate_integrity_alerts=False
        )

        generator = OrgAlertGenerator(config)
        alerts = generator.generate_all_alerts(sample_org_report_healthy)

        risk_alerts = [a for a in alerts if a.category == OrgAlertCategory.RISK]
        assert len(risk_alerts) == 0


# ============================================================================
# Test: Alert Generation - Trend Alerts
# ============================================================================

class TestTrendAlertGeneration:
    """Tests for trend-based alert generation."""

    def test_generates_alert_for_high_declining(self, temp_dir, sample_org_report_critical):
        """Test that high declining percentage generates alert."""
        report_path = create_org_report_file(temp_dir, sample_org_report_critical)
        config = OrgAlertConfig(
            org_report_path=report_path,
            generate_slo_alerts=False,
            generate_risk_alerts=False,
            generate_trend_alerts=True,
            generate_integrity_alerts=False,
            thresholds=OrgAlertThresholds(
                percent_declining_warning=0.20,
                percent_declining_critical=0.40
            )
        )

        generator = OrgAlertGenerator(config)
        alerts = generator.generate_all_alerts(sample_org_report_critical)

        trend_alerts = [a for a in alerts if a.category == OrgAlertCategory.TREND]
        # 80% declining (4/5) should trigger critical
        assert len(trend_alerts) >= 1

    def test_generates_alert_for_low_green_percentage(self, temp_dir, sample_org_report_critical):
        """Test that low green percentage generates alert."""
        report_path = create_org_report_file(temp_dir, sample_org_report_critical)
        config = OrgAlertConfig(
            org_report_path=report_path,
            generate_slo_alerts=False,
            generate_risk_alerts=False,
            generate_trend_alerts=True,
            generate_integrity_alerts=False,
            thresholds=OrgAlertThresholds(
                percent_green_warning=0.60,
                percent_green_critical=0.40
            )
        )

        generator = OrgAlertGenerator(config)
        alerts = generator.generate_all_alerts(sample_org_report_critical)

        trend_alerts = [a for a in alerts if a.category == OrgAlertCategory.TREND]
        # 0% green should trigger critical
        low_green_alerts = [a for a in trend_alerts if "healthy" in a.title.lower() or "green" in a.title.lower()]
        assert len(low_green_alerts) >= 1

    def test_generates_alert_for_low_avg_score(self, temp_dir, sample_org_report_critical):
        """Test that low average score generates alert."""
        report_path = create_org_report_file(temp_dir, sample_org_report_critical)
        config = OrgAlertConfig(
            org_report_path=report_path,
            generate_slo_alerts=False,
            generate_risk_alerts=False,
            generate_trend_alerts=True,
            generate_integrity_alerts=False,
            thresholds=OrgAlertThresholds(
                avg_score_warning=70.0,
                avg_score_critical=50.0
            )
        )

        generator = OrgAlertGenerator(config)
        alerts = generator.generate_all_alerts(sample_org_report_critical)

        trend_alerts = [a for a in alerts if a.category == OrgAlertCategory.TREND]
        # Score of 35 should trigger critical
        score_alerts = [a for a in trend_alerts if "score" in a.title.lower()]
        assert len(score_alerts) >= 1

    def test_no_trend_alerts_for_healthy_org(self, temp_dir, sample_org_report_healthy):
        """Test that no trend alerts for healthy org."""
        report_path = create_org_report_file(temp_dir, sample_org_report_healthy)
        config = OrgAlertConfig(
            org_report_path=report_path,
            generate_slo_alerts=False,
            generate_risk_alerts=False,
            generate_trend_alerts=True,
            generate_integrity_alerts=False
        )

        generator = OrgAlertGenerator(config)
        alerts = generator.generate_all_alerts(sample_org_report_healthy)

        trend_alerts = [a for a in alerts if a.category == OrgAlertCategory.TREND]
        assert len(trend_alerts) == 0


# ============================================================================
# Test: Alert Generation - Integrity Alerts
# ============================================================================

class TestIntegrityAlertGeneration:
    """Tests for integrity issue alert generation."""

    def test_generates_alert_for_load_errors(self, temp_dir, sample_org_report_with_errors):
        """Test that load errors generate integrity alert."""
        report_path = create_org_report_file(temp_dir, sample_org_report_with_errors)
        config = OrgAlertConfig(
            org_report_path=report_path,
            generate_slo_alerts=False,
            generate_risk_alerts=False,
            generate_trend_alerts=False,
            generate_integrity_alerts=True
        )

        generator = OrgAlertGenerator(config)
        alerts = generator.generate_all_alerts(sample_org_report_with_errors)

        integrity_alerts = [a for a in alerts if a.category == OrgAlertCategory.INTEGRITY]
        assert len(integrity_alerts) >= 1

        # Check affected repos
        alert = integrity_alerts[0]
        assert "repo-fail-1" in alert.affected_repos or "repo-fail-2" in alert.affected_repos

    def test_no_integrity_alerts_without_errors(self, temp_dir, sample_org_report_healthy):
        """Test that no integrity alerts when no errors."""
        report_path = create_org_report_file(temp_dir, sample_org_report_healthy)
        config = OrgAlertConfig(
            org_report_path=report_path,
            generate_slo_alerts=False,
            generate_risk_alerts=False,
            generate_trend_alerts=False,
            generate_integrity_alerts=True
        )

        generator = OrgAlertGenerator(config)
        alerts = generator.generate_all_alerts(sample_org_report_healthy)

        integrity_alerts = [a for a in alerts if a.category == OrgAlertCategory.INTEGRITY]
        assert len(integrity_alerts) == 0


# ============================================================================
# Test: Escalation Rules
# ============================================================================

class TestEscalationRules:
    """Tests for escalation rule matching and execution."""

    def test_escalation_rule_matches_category(self):
        """Test that escalation rule matches by category."""
        rule = EscalationRule(
            id="test-rule",
            description="Test rule",
            condition=EscalationCondition(
                alert_category=OrgAlertCategory.SLO
            ),
            actions=[EscalationAction.from_string("log")]
        )

        alert_slo = OrgAlert(
            alert_id="test-1",
            category=OrgAlertCategory.SLO,
            severity=AlertSeverity.HIGH,
            title="Test",
            message="Test",
            timestamp=datetime.utcnow().isoformat()
        )

        alert_risk = OrgAlert(
            alert_id="test-2",
            category=OrgAlertCategory.RISK,
            severity=AlertSeverity.HIGH,
            title="Test",
            message="Test",
            timestamp=datetime.utcnow().isoformat()
        )

        assert rule.matches(alert_slo) is True
        assert rule.matches(alert_risk) is False

    def test_escalation_rule_matches_severity(self):
        """Test that escalation rule matches by severity."""
        rule = EscalationRule(
            id="test-rule",
            description="Test rule",
            condition=EscalationCondition(
                severity=AlertSeverity.CRITICAL
            ),
            actions=[EscalationAction.from_string("log")]
        )

        alert_critical = OrgAlert(
            alert_id="test-1",
            category=OrgAlertCategory.SLO,
            severity=AlertSeverity.CRITICAL,
            title="Test",
            message="Test",
            timestamp=datetime.utcnow().isoformat()
        )

        alert_high = OrgAlert(
            alert_id="test-2",
            category=OrgAlertCategory.SLO,
            severity=AlertSeverity.HIGH,
            title="Test",
            message="Test",
            timestamp=datetime.utcnow().isoformat()
        )

        assert rule.matches(alert_critical) is True
        assert rule.matches(alert_high) is False

    def test_escalation_rule_matches_combined(self):
        """Test that escalation rule matches combined conditions."""
        rule = EscalationRule(
            id="test-rule",
            description="Test rule",
            condition=EscalationCondition(
                alert_category=OrgAlertCategory.SLO,
                severity=AlertSeverity.CRITICAL
            ),
            actions=[EscalationAction.from_string("log")]
        )

        alert_match = OrgAlert(
            alert_id="test-1",
            category=OrgAlertCategory.SLO,
            severity=AlertSeverity.CRITICAL,
            title="Test",
            message="Test",
            timestamp=datetime.utcnow().isoformat()
        )

        alert_no_match = OrgAlert(
            alert_id="test-2",
            category=OrgAlertCategory.SLO,
            severity=AlertSeverity.HIGH,
            title="Test",
            message="Test",
            timestamp=datetime.utcnow().isoformat()
        )

        assert rule.matches(alert_match) is True
        assert rule.matches(alert_no_match) is False

    def test_escalation_engine_executes_actions(self):
        """Test that escalation engine executes matching actions."""
        rules = [
            EscalationRule(
                id="slo-critical",
                description="Escalate critical SLO",
                condition=EscalationCondition(
                    alert_category=OrgAlertCategory.SLO,
                    severity=AlertSeverity.CRITICAL
                ),
                actions=[
                    EscalationAction.from_string("escalate_to:oncall"),
                    EscalationAction.from_string("notify:slack:org-alerts")
                ]
            )
        ]

        engine = EscalationEngine(rules)

        alerts = [
            OrgAlert(
                alert_id="test-1",
                category=OrgAlertCategory.SLO,
                severity=AlertSeverity.CRITICAL,
                title="Test",
                message="Test",
                timestamp=datetime.utcnow().isoformat()
            )
        ]

        processed_alerts, actions = engine.process_alerts(alerts)

        assert len(actions) == 2
        assert actions[0]["action_type"] == "escalate_to"
        assert actions[1]["action_type"] == "notify"
        assert processed_alerts[0].escalated is True

    def test_escalation_engine_respects_priority(self):
        """Test that higher priority rules are evaluated first."""
        rules = [
            EscalationRule(
                id="low-priority",
                description="Low priority rule",
                condition=EscalationCondition(alert_category=OrgAlertCategory.SLO),
                actions=[EscalationAction.from_string("log")],
                priority=10
            ),
            EscalationRule(
                id="high-priority",
                description="High priority rule",
                condition=EscalationCondition(alert_category=OrgAlertCategory.SLO),
                actions=[EscalationAction.from_string("escalate_to:oncall")],
                priority=100
            )
        ]

        engine = EscalationEngine(rules)

        # Verify rules are sorted by priority
        assert engine.rules[0].id == "high-priority"
        assert engine.rules[1].id == "low-priority"


# ============================================================================
# Test: Routing Channels
# ============================================================================

class TestRoutingChannels:
    """Tests for alert routing channels."""

    def test_console_channel_dispatch(self, capsys):
        """Test that console channel dispatches alerts."""
        config = OrgAlertChannelConfig(
            channel_type=OrgAlertChannelType.CONSOLE,
            enabled=True
        )
        channel = ConsoleOrgAlertChannel(config)

        alerts = [
            OrgAlert(
                alert_id="test-1",
                category=OrgAlertCategory.SLO,
                severity=AlertSeverity.HIGH,
                title="Test Alert",
                message="Test message",
                timestamp=datetime.utcnow().isoformat()
            )
        ]

        report = OrgAlertReport(
            report_id="test",
            generated_at=datetime.utcnow().isoformat(),
            org_report_path="/test",
            org_health_status="yellow",
            org_health_score=75.0,
            org_risk_tier="medium",
            total_repos=10
        )

        result = channel.dispatch(alerts, report)
        assert result is True

        captured = capsys.readouterr()
        assert "Test Alert" in captured.out

    def test_json_file_channel_dispatch(self, temp_dir):
        """Test that JSON file channel writes output."""
        output_path = temp_dir / "alerts.json"
        config = OrgAlertChannelConfig(
            channel_type=OrgAlertChannelType.JSON_FILE,
            enabled=True,
            output_path=output_path
        )
        channel = JsonFileOrgAlertChannel(config)

        alerts = [
            OrgAlert(
                alert_id="test-1",
                category=OrgAlertCategory.SLO,
                severity=AlertSeverity.HIGH,
                title="Test Alert",
                message="Test message",
                timestamp=datetime.utcnow().isoformat()
            )
        ]

        report = OrgAlertReport(
            report_id="test",
            generated_at=datetime.utcnow().isoformat(),
            org_report_path="/test",
            total_alerts=1
        )

        result = channel.dispatch(alerts, report)
        assert result is True
        assert output_path.exists()

        with open(output_path) as f:
            saved = json.load(f)
        assert saved["total_alerts"] == 1

    def test_stdout_channel_dispatch(self, capsys):
        """Test that stdout channel outputs JSON."""
        config = OrgAlertChannelConfig(
            channel_type=OrgAlertChannelType.STDOUT,
            enabled=True
        )
        channel = StdoutOrgAlertChannel(config)

        report = OrgAlertReport(
            report_id="test",
            generated_at=datetime.utcnow().isoformat(),
            org_report_path="/test",
            total_alerts=5
        )

        result = channel.dispatch([], report)
        assert result is True

        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert parsed["total_alerts"] == 5

    def test_email_channel_stub(self):
        """Test that email channel stub works."""
        config = OrgAlertChannelConfig(
            channel_type=OrgAlertChannelType.EMAIL,
            enabled=True,
            email_to="test@example.com"
        )
        channel = EmailOrgAlertChannel(config)

        report = OrgAlertReport(
            report_id="test",
            generated_at=datetime.utcnow().isoformat(),
            org_report_path="/test"
        )

        result = channel.dispatch([], report)
        assert result is True
        assert channel._last_email["to"] == "test@example.com"

    def test_slack_channel_stub(self):
        """Test that Slack channel stub works."""
        config = OrgAlertChannelConfig(
            channel_type=OrgAlertChannelType.SLACK,
            enabled=True,
            slack_webhook_url="https://hooks.slack.com/test",
            slack_channel="#alerts"
        )
        channel = SlackOrgAlertChannel(config)

        report = OrgAlertReport(
            report_id="test",
            generated_at=datetime.utcnow().isoformat(),
            org_report_path="/test"
        )

        result = channel.dispatch([], report)
        assert result is True
        assert channel._last_payload["channel"] == "#alerts"

    def test_webhook_channel_stub(self):
        """Test that webhook channel stub works."""
        config = OrgAlertChannelConfig(
            channel_type=OrgAlertChannelType.WEBHOOK,
            enabled=True,
            webhook_url="https://example.com/webhook"
        )
        channel = WebhookOrgAlertChannel(config)

        report = OrgAlertReport(
            report_id="test",
            generated_at=datetime.utcnow().isoformat(),
            org_report_path="/test"
        )

        result = channel.dispatch([], report)
        assert result is True
        assert channel._last_payload["url"] == "https://example.com/webhook"

    def test_channel_severity_filter(self):
        """Test that channel filters by severity."""
        config = OrgAlertChannelConfig(
            channel_type=OrgAlertChannelType.CONSOLE,
            enabled=True,
            min_severity=AlertSeverity.HIGH
        )
        channel = ConsoleOrgAlertChannel(config)

        alerts = [
            OrgAlert(
                alert_id="low-1",
                category=OrgAlertCategory.SLO,
                severity=AlertSeverity.LOW,
                title="Low Alert",
                message="Test",
                timestamp=datetime.utcnow().isoformat()
            ),
            OrgAlert(
                alert_id="high-1",
                category=OrgAlertCategory.SLO,
                severity=AlertSeverity.HIGH,
                title="High Alert",
                message="Test",
                timestamp=datetime.utcnow().isoformat()
            )
        ]

        filtered = channel.filter_alerts(alerts)
        assert len(filtered) == 1
        assert filtered[0].alert_id == "high-1"


# ============================================================================
# Test: Engine Integration
# ============================================================================

class TestEngineIntegration:
    """Tests for the full alerting engine integration."""

    def test_engine_success_no_alerts(self, temp_dir, sample_org_report_healthy):
        """Test successful run with no alerts returns EXIT_ORG_ALERT_SUCCESS."""
        report_path = create_org_report_file(temp_dir, sample_org_report_healthy)
        config = OrgAlertConfig(
            org_report_path=report_path,
            channels=[],
            fail_on_critical=False,
            fail_on_any_alerts=False
        )

        engine = OrgAlertingEngine(config)
        report, exit_code = engine.run()

        assert exit_code == EXIT_ORG_ALERT_SUCCESS
        assert report.total_alerts == 0

    def test_engine_alerts_present_exit_code(self, temp_dir, sample_org_report):
        """Test that alerts present returns EXIT_ALERTS_PRESENT."""
        report_path = create_org_report_file(temp_dir, sample_org_report)
        config = OrgAlertConfig(
            org_report_path=report_path,
            channels=[],
            fail_on_critical=False,
            fail_on_any_alerts=False
        )

        engine = OrgAlertingEngine(config)
        report, exit_code = engine.run()

        assert exit_code == EXIT_ALERTS_PRESENT
        assert report.total_alerts > 0

    def test_engine_critical_alerts_exit_code(self, temp_dir, sample_org_report_critical):
        """Test that critical alerts with fail_on_critical returns EXIT_CRITICAL_ALERTS."""
        report_path = create_org_report_file(temp_dir, sample_org_report_critical)
        config = OrgAlertConfig(
            org_report_path=report_path,
            channels=[],
            fail_on_critical=True
        )

        engine = OrgAlertingEngine(config)
        report, exit_code = engine.run()

        assert exit_code == EXIT_CRITICAL_ALERTS
        assert report.critical_alerts > 0

    def test_engine_missing_report_exit_code(self, temp_dir):
        """Test that missing report returns EXIT_ORG_REPORT_PARSE_ERROR."""
        config = OrgAlertConfig(
            org_report_path=temp_dir / "nonexistent.json",
            channels=[]
        )

        engine = OrgAlertingEngine(config)
        report, exit_code = engine.run()

        assert exit_code == EXIT_ORG_REPORT_PARSE_ERROR

    def test_engine_invalid_json_exit_code(self, temp_dir):
        """Test that invalid JSON returns EXIT_ORG_REPORT_PARSE_ERROR."""
        report_path = temp_dir / "invalid.json"
        with open(report_path, 'w') as f:
            f.write("{ invalid json }")

        config = OrgAlertConfig(
            org_report_path=report_path,
            channels=[]
        )

        engine = OrgAlertingEngine(config)
        report, exit_code = engine.run()

        assert exit_code == EXIT_ORG_REPORT_PARSE_ERROR

    def test_engine_writes_output_file(self, temp_dir, sample_org_report):
        """Test that engine writes output file when configured."""
        report_path = create_org_report_file(temp_dir, sample_org_report)
        output_path = temp_dir / "output" / "alerts.json"

        config = OrgAlertConfig(
            org_report_path=report_path,
            output_path=output_path,
            channels=[]
        )

        engine = OrgAlertingEngine(config)
        report, exit_code = engine.run()

        assert output_path.exists()
        with open(output_path) as f:
            saved = json.load(f)
        assert saved["total_alerts"] == report.total_alerts


# ============================================================================
# Test: Data Classes & Serialization
# ============================================================================

class TestDataClasses:
    """Tests for dataclass serialization and deserialization."""

    def test_org_alert_to_dict(self):
        """Test OrgAlert serialization."""
        alert = OrgAlert(
            alert_id="test-1",
            category=OrgAlertCategory.SLO,
            severity=AlertSeverity.HIGH,
            title="Test Alert",
            message="Test message",
            timestamp="2025-01-07T12:00:00",
            slo_id="test-slo",
            affected_repos=["repo-a", "repo-b"]
        )

        data = alert.to_dict()
        assert data["alert_id"] == "test-1"
        assert data["category"] == "slo"
        assert data["severity"] == "high"
        assert data["slo_id"] == "test-slo"
        assert len(data["affected_repos"]) == 2

    def test_org_alert_from_dict(self):
        """Test OrgAlert deserialization."""
        data = {
            "alert_id": "test-1",
            "category": "slo",
            "severity": "critical",
            "title": "Test",
            "message": "Test",
            "timestamp": "2025-01-07T12:00:00",
            "slo_id": "test-slo"
        }

        alert = OrgAlert.from_dict(data)
        assert alert.alert_id == "test-1"
        assert alert.category == OrgAlertCategory.SLO
        assert alert.severity == AlertSeverity.CRITICAL

    def test_escalation_action_from_string(self):
        """Test EscalationAction parsing from string."""
        action1 = EscalationAction.from_string("escalate_to:oncall")
        assert action1.action_type == EscalationActionType.ESCALATE_TO
        assert action1.target == "oncall"

        action2 = EscalationAction.from_string("notify:slack:alerts")
        assert action2.action_type == EscalationActionType.NOTIFY
        assert action2.channel == "slack"
        assert action2.recipient == "alerts"

        action3 = EscalationAction.from_string("log")
        assert action3.action_type == EscalationActionType.LOG

    def test_escalation_rule_from_dict(self):
        """Test EscalationRule parsing from dictionary."""
        data = {
            "id": "test-rule",
            "description": "Test",
            "when": {
                "alert_category": "slo",
                "severity": "critical"
            },
            "actions": [
                "escalate_to:oncall",
                "notify:slack:alerts"
            ]
        }

        rule = EscalationRule.from_dict(data)
        assert rule.id == "test-rule"
        assert rule.condition.alert_category == OrgAlertCategory.SLO
        assert rule.condition.severity == AlertSeverity.CRITICAL
        assert len(rule.actions) == 2

    def test_org_alert_report_serialization(self):
        """Test OrgAlertReport round-trip serialization."""
        report = OrgAlertReport(
            report_id="test-report",
            generated_at="2025-01-07T12:00:00",
            org_report_path="/test/path",
            total_alerts=5,
            critical_alerts=1,
            high_alerts=2,
            medium_alerts=1,
            low_alerts=1,
            slo_alerts=3,
            risk_alerts=2
        )

        data = report.to_dict()
        restored = OrgAlertReport.from_dict(data)

        assert restored.report_id == report.report_id
        assert restored.total_alerts == report.total_alerts
        assert restored.critical_alerts == report.critical_alerts


# ============================================================================
# Test: Configuration
# ============================================================================

class TestConfiguration:
    """Tests for configuration loading and validation."""

    def test_default_escalation_rules(self):
        """Test that default escalation rules are created."""
        rules = create_default_escalation_rules()
        assert len(rules) >= 3

        # Check for essential rules
        rule_ids = [r.id for r in rules]
        assert "slo-critical" in rule_ids

    def test_default_channels(self):
        """Test that default channels are created."""
        channels = create_default_channels()
        assert len(channels) >= 1

        # Should include console
        channel_types = [c.channel_type for c in channels]
        assert OrgAlertChannelType.CONSOLE in channel_types

    def test_load_json_config(self, temp_dir):
        """Test loading configuration from JSON file."""
        config_data = {
            "escalation_rules": [
                {
                    "id": "test-rule",
                    "description": "Test",
                    "when": {"alert_category": "slo"},
                    "actions": ["log"]
                }
            ]
        }

        config_path = temp_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config_data, f)

        rules = load_escalation_config(config_path)
        assert len(rules) == 1
        assert rules[0].id == "test-rule"

    def test_alert_thresholds(self):
        """Test alert threshold configuration."""
        thresholds = OrgAlertThresholds(
            percent_declining_warning=0.15,
            percent_declining_critical=0.30,
            percent_green_warning=0.70,
            percent_green_critical=0.50
        )

        data = thresholds.to_dict()
        restored = OrgAlertThresholds.from_dict(data)

        assert restored.percent_declining_warning == 0.15
        assert restored.percent_declining_critical == 0.30


# ============================================================================
# Test: Edge Cases
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_org_report(self, temp_dir):
        """Test handling of org report with minimal data."""
        report_data = {
            "report_id": "empty",
            "generated_at": datetime.utcnow().isoformat(),
            "org_health_status": "unknown",
            "repos_loaded": 0,
            "metrics": {
                "total_repos": 0,
                "repos_green": 0,
                "repos_yellow": 0,
                "repos_red": 0,
                "repos_declining": 0,
                "percent_green": 0,
                "avg_score": 0
            },
            "slo_results": [],
            "top_risk_repos": [],
            "load_errors": []
        }

        report_path = create_org_report_file(temp_dir, report_data)
        config = OrgAlertConfig(
            org_report_path=report_path,
            channels=[]
        )

        engine = OrgAlertingEngine(config)
        report, exit_code = engine.run()

        # Should complete without error
        assert exit_code in (EXIT_ORG_ALERT_SUCCESS, EXIT_ALERTS_PRESENT)

    def test_all_alerts_are_informational(self, temp_dir, sample_org_report_with_errors):
        """Test when all alerts are low severity."""
        report_path = create_org_report_file(temp_dir, sample_org_report_with_errors)
        config = OrgAlertConfig(
            org_report_path=report_path,
            channels=[],
            generate_slo_alerts=False,
            generate_risk_alerts=False,
            generate_trend_alerts=False,
            generate_integrity_alerts=True,
            fail_on_critical=True
        )

        engine = OrgAlertingEngine(config)
        report, exit_code = engine.run()

        # Should not return CRITICAL exit code if no critical alerts
        if report.critical_alerts == 0:
            assert exit_code != EXIT_CRITICAL_ALERTS

    def test_alert_id_uniqueness(self, temp_dir, sample_org_report):
        """Test that generated alert IDs are unique."""
        report_path = create_org_report_file(temp_dir, sample_org_report)
        config = OrgAlertConfig(
            org_report_path=report_path
        )

        generator = OrgAlertGenerator(config)
        alerts = generator.generate_all_alerts(sample_org_report)

        alert_ids = [a.alert_id for a in alerts]
        assert len(alert_ids) == len(set(alert_ids)), "Alert IDs should be unique"

    def test_severity_comparison(self):
        """Test severity level comparisons."""
        assert AlertSeverity.LOW < AlertSeverity.MEDIUM
        assert AlertSeverity.MEDIUM < AlertSeverity.HIGH
        assert AlertSeverity.HIGH < AlertSeverity.CRITICAL
        assert AlertSeverity.CRITICAL >= AlertSeverity.CRITICAL
        assert AlertSeverity.LOW <= AlertSeverity.LOW


# ============================================================================
# Test: CLI Behavior (via module import)
# ============================================================================

class TestCLIBehavior:
    """Tests for CLI argument parsing and behavior."""

    def test_cli_module_importable(self):
        """Test that CLI module can be imported."""
        from analytics import run_org_alerts
        assert hasattr(run_org_alerts, 'main')
        assert hasattr(run_org_alerts, 'create_parser')

    def test_cli_parser_required_args(self):
        """Test that parser requires org-report."""
        from analytics.run_org_alerts import create_parser
        parser = create_parser()

        # Should fail without required args
        import sys
        from io import StringIO
        old_stderr = sys.stderr
        sys.stderr = StringIO()

        try:
            parser.parse_args([])
            assert False, "Should have raised SystemExit"
        except SystemExit:
            pass
        finally:
            sys.stderr = old_stderr

    def test_cli_parser_accepts_all_options(self):
        """Test that parser accepts all documented options."""
        from analytics.run_org_alerts import create_parser
        parser = create_parser()

        args = parser.parse_args([
            "--org-report", "/test/report.json",
            "--output", "/test/output.json",
            "--fail-on-critical",
            "--fail-on-any-alerts",
            "--use-default-escalations",
            "--no-slo-alerts",
            "--no-risk-alerts",
            "--no-trend-alerts",
            "--no-integrity-alerts",
            "--declining-warning", "0.25",
            "--declining-critical", "0.45",
            "--green-warning", "0.65",
            "--green-critical", "0.45",
            "--score-warning", "65.0",
            "--score-critical", "45.0",
            "--summary-only",
            "--verbose"
        ])

        assert args.org_report == Path("/test/report.json")
        assert args.fail_on_critical is True
        assert args.no_slo_alerts is True
        assert args.declining_warning == 0.25


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
