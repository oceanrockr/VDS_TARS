#!/usr/bin/env python3
"""
Test Suite for SLA Reporting & Executive Readiness Dashboard Engine

This module provides comprehensive tests for:
- SLA policy parsing (JSON and YAML)
- Compliance logic per SLA type
- Breach attribution correctness
- Executive readiness scoring
- Multi-window evaluation
- CI/CD exit code behavior

Version: 1.0.0
Phase: 14.8 Task 5
"""

import json
import os
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from analytics.org_sla_intelligence import (
    # Main classes
    SLAIntelligenceEngine,
    SLAPolicyLoader,
    SLAComplianceEngine,
    SLABreachAttributionEngine,
    ExecutiveReadinessEngine,

    # Data classes
    SLAPolicy,
    SLATarget,
    SLAWindowResult,
    SLAComplianceResult,
    SLABreach,
    SLARootCause,
    ExecutiveReadinessScore,
    SLAScorecard,
    SLAIntelligenceSummary,
    SLAIntelligenceReport,
    RiskNarrative,

    # Config classes
    SLAIntelligenceConfig,
    SLAThresholds,

    # Enums
    SLAType,
    SLAStatus,
    SLASeverity,
    ReadinessTier,
    RiskOutlook,

    # Exit codes
    EXIT_SLA_SUCCESS,
    EXIT_SLA_AT_RISK,
    EXIT_SLA_BREACH,
    EXIT_SLA_CONFIG_ERROR,
    EXIT_SLA_PARSE_ERROR,
    EXIT_GENERAL_SLA_ERROR,

    # Utility functions
    create_default_thresholds,
    create_default_policies
)


class TestSLAEnums(unittest.TestCase):
    """Test enum definitions and comparisons."""

    def test_sla_type_values(self):
        """Test SLAType enum values."""
        self.assertEqual(SLAType.AVAILABILITY.value, "availability")
        self.assertEqual(SLAType.RELIABILITY.value, "reliability")
        self.assertEqual(SLAType.INCIDENT_RESPONSE.value, "incident_response")
        self.assertEqual(SLAType.CHANGE_FAILURE_RATE.value, "change_failure_rate")

    def test_sla_status_comparison(self):
        """Test SLAStatus comparison."""
        self.assertTrue(SLAStatus.COMPLIANT < SLAStatus.AT_RISK)
        self.assertTrue(SLAStatus.AT_RISK < SLAStatus.BREACHED)
        self.assertTrue(SLAStatus.UNKNOWN < SLAStatus.COMPLIANT)

    def test_sla_severity_comparison(self):
        """Test SLASeverity comparison."""
        self.assertTrue(SLASeverity.LOW < SLASeverity.MEDIUM)
        self.assertTrue(SLASeverity.MEDIUM < SLASeverity.HIGH)
        self.assertTrue(SLASeverity.HIGH < SLASeverity.CRITICAL)

    def test_readiness_tier_values(self):
        """Test ReadinessTier enum values."""
        self.assertEqual(ReadinessTier.GREEN.value, "green")
        self.assertEqual(ReadinessTier.YELLOW.value, "yellow")
        self.assertEqual(ReadinessTier.RED.value, "red")

    def test_risk_outlook_values(self):
        """Test RiskOutlook enum values."""
        self.assertEqual(RiskOutlook.IMPROVING.value, "improving")
        self.assertEqual(RiskOutlook.STABLE.value, "stable")
        self.assertEqual(RiskOutlook.DEGRADING.value, "degrading")
        self.assertEqual(RiskOutlook.CRITICAL.value, "critical")


class TestSLATarget(unittest.TestCase):
    """Test SLATarget data class."""

    def test_target_creation(self):
        """Test creating an SLA target."""
        target = SLATarget(
            metric_name="availability",
            target_value=99.9,
            warning_threshold=99.0,
            breach_threshold=95.0,
            unit="%",
            higher_is_better=True
        )

        self.assertEqual(target.metric_name, "availability")
        self.assertEqual(target.target_value, 99.9)
        self.assertEqual(target.warning_threshold, 99.0)
        self.assertEqual(target.breach_threshold, 95.0)
        self.assertTrue(target.higher_is_better)

    def test_target_to_dict(self):
        """Test converting target to dictionary."""
        target = SLATarget(
            metric_name="response_time",
            target_value=100.0,
            warning_threshold=200.0,
            breach_threshold=500.0,
            unit="ms",
            higher_is_better=False
        )

        result = target.to_dict()
        self.assertEqual(result["metric_name"], "response_time")
        self.assertEqual(result["target_value"], 100.0)
        self.assertFalse(result["higher_is_better"])


class TestSLAPolicy(unittest.TestCase):
    """Test SLAPolicy data class."""

    def test_policy_creation(self):
        """Test creating an SLA policy."""
        policy = SLAPolicy(
            policy_id="sla_001",
            policy_name="Test SLA",
            sla_type=SLAType.AVAILABILITY,
            description="Test description",
            priority=1
        )

        self.assertEqual(policy.policy_id, "sla_001")
        self.assertEqual(policy.sla_type, SLAType.AVAILABILITY)
        self.assertEqual(policy.priority, 1)

    def test_policy_from_dict(self):
        """Test creating policy from dictionary."""
        data = {
            "policy_id": "sla_availability",
            "policy_name": "Availability SLA",
            "sla_type": "availability",
            "description": "Service availability target",
            "targets": [
                {
                    "metric_name": "availability_score",
                    "target_value": 99.0,
                    "warning_threshold": 95.0,
                    "breach_threshold": 90.0
                }
            ],
            "priority": 1,
            "severity_on_breach": "critical"
        }

        policy = SLAPolicy.from_dict(data)

        self.assertEqual(policy.policy_id, "sla_availability")
        self.assertEqual(policy.sla_type, SLAType.AVAILABILITY)
        self.assertEqual(len(policy.targets), 1)
        self.assertEqual(policy.severity_on_breach, SLASeverity.CRITICAL)

    def test_policy_to_dict(self):
        """Test converting policy to dictionary."""
        policy = SLAPolicy(
            policy_id="sla_001",
            policy_name="Test SLA",
            sla_type=SLAType.RELIABILITY,
            targets=[
                SLATarget(
                    metric_name="health_score",
                    target_value=80.0,
                    warning_threshold=70.0,
                    breach_threshold=60.0
                )
            ]
        )

        result = policy.to_dict()
        self.assertEqual(result["policy_id"], "sla_001")
        self.assertEqual(result["sla_type"], "reliability")
        self.assertEqual(len(result["targets"]), 1)


class TestSLAThresholds(unittest.TestCase):
    """Test SLAThresholds configuration."""

    def test_default_thresholds(self):
        """Test default threshold values."""
        thresholds = SLAThresholds()

        self.assertEqual(thresholds.at_risk_percentage, 90.0)
        self.assertEqual(thresholds.breach_percentage, 80.0)
        self.assertEqual(thresholds.green_threshold, 80.0)
        self.assertEqual(thresholds.yellow_threshold, 60.0)

    def test_thresholds_from_dict(self):
        """Test creating thresholds from dictionary."""
        data = {
            "at_risk_percentage": 85.0,
            "breach_percentage": 70.0,
            "green_threshold": 90.0
        }

        thresholds = SLAThresholds.from_dict(data)

        self.assertEqual(thresholds.at_risk_percentage, 85.0)
        self.assertEqual(thresholds.breach_percentage, 70.0)
        self.assertEqual(thresholds.green_threshold, 90.0)

    def test_create_default_thresholds(self):
        """Test utility function for default thresholds."""
        thresholds = create_default_thresholds()
        self.assertIsInstance(thresholds, SLAThresholds)


class TestSLAPolicyLoader(unittest.TestCase):
    """Test SLA policy loading functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.org_report_path = Path(self.temp_dir) / "org-health-report.json"

        # Create minimal org report
        org_report = {"repositories": [], "org_health_score": 80.0}
        with open(self.org_report_path, 'w') as f:
            json.dump(org_report, f)

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_load_default_policies(self):
        """Test loading default policies when no file provided."""
        config = SLAIntelligenceConfig(org_report_path=self.org_report_path)
        loader = SLAPolicyLoader(config)

        policies = loader.load_policies()

        self.assertGreater(len(policies), 0)
        self.assertTrue(all(isinstance(p, SLAPolicy) for p in policies))

    def test_load_json_policies(self):
        """Test loading policies from JSON file."""
        policy_data = {
            "policies": [
                {
                    "policy_id": "custom_sla",
                    "policy_name": "Custom SLA",
                    "sla_type": "availability",
                    "targets": [
                        {
                            "metric_name": "custom_metric",
                            "target_value": 99.5,
                            "warning_threshold": 98.0,
                            "breach_threshold": 95.0
                        }
                    ]
                }
            ]
        }

        policy_path = Path(self.temp_dir) / "policies.json"
        with open(policy_path, 'w') as f:
            json.dump(policy_data, f)

        config = SLAIntelligenceConfig(
            org_report_path=self.org_report_path,
            sla_policy_path=policy_path
        )
        loader = SLAPolicyLoader(config)

        policies = loader.load_policies()

        self.assertEqual(len(policies), 1)
        self.assertEqual(policies[0].policy_id, "custom_sla")

    def test_create_default_policies_utility(self):
        """Test utility function for default policies."""
        policies = create_default_policies()

        self.assertGreater(len(policies), 0)
        # Should have availability, reliability, incident_response, change_failure_rate
        policy_types = [p.sla_type for p in policies]
        self.assertIn(SLAType.AVAILABILITY, policy_types)
        self.assertIn(SLAType.RELIABILITY, policy_types)


class TestSLAComplianceEngine(unittest.TestCase):
    """Test SLA compliance evaluation."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.org_report_path = Path(self.temp_dir) / "org-health-report.json"

        # Create org report with test data
        self.org_report = {
            "org_health_score": 85.0,
            "org_health_status": "yellow",
            "repositories": [
                {
                    "repo_id": "repo-1",
                    "repository_score": 90.0,
                    "risk_tier": "low",
                    "trends": {"overall_trend": "stable"}
                },
                {
                    "repo_id": "repo-2",
                    "repository_score": 60.0,
                    "risk_tier": "high",
                    "trends": {"overall_trend": "declining"}
                }
            ]
        }

        with open(self.org_report_path, 'w') as f:
            json.dump(self.org_report, f)

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_evaluate_compliance_compliant(self):
        """Test compliance evaluation for compliant SLA."""
        config = SLAIntelligenceConfig(org_report_path=self.org_report_path)
        engine = SLAComplianceEngine(config)

        # Create a policy that should be compliant (target 80, actual 85)
        policy = SLAPolicy(
            policy_id="test_sla",
            policy_name="Test SLA",
            sla_type=SLAType.AVAILABILITY,
            targets=[
                SLATarget(
                    metric_name="availability_score",
                    target_value=80.0,
                    warning_threshold=70.0,
                    breach_threshold=60.0
                )
            ]
        )

        results = engine.evaluate_compliance([policy], self.org_report)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].overall_status, SLAStatus.COMPLIANT)

    def test_evaluate_compliance_at_risk(self):
        """Test compliance evaluation for at-risk SLA."""
        config = SLAIntelligenceConfig(org_report_path=self.org_report_path)
        engine = SLAComplianceEngine(config)

        # Create a policy that should be at-risk (target 90, actual 85)
        policy = SLAPolicy(
            policy_id="test_sla",
            policy_name="Test SLA",
            sla_type=SLAType.AVAILABILITY,
            targets=[
                SLATarget(
                    metric_name="availability_score",
                    target_value=90.0,
                    warning_threshold=80.0,
                    breach_threshold=70.0
                )
            ]
        )

        results = engine.evaluate_compliance([policy], self.org_report)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].overall_status, SLAStatus.AT_RISK)

    def test_evaluate_compliance_breached(self):
        """Test compliance evaluation for breached SLA."""
        config = SLAIntelligenceConfig(org_report_path=self.org_report_path)
        engine = SLAComplianceEngine(config)

        # Create a policy that should be breached (target 99, actual 85)
        policy = SLAPolicy(
            policy_id="test_sla",
            policy_name="Test SLA",
            sla_type=SLAType.AVAILABILITY,
            targets=[
                SLATarget(
                    metric_name="availability_score",
                    target_value=99.0,
                    warning_threshold=95.0,
                    breach_threshold=90.0
                )
            ]
        )

        results = engine.evaluate_compliance([policy], self.org_report)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].overall_status, SLAStatus.BREACHED)

    def test_evaluate_multiple_windows(self):
        """Test evaluation across multiple time windows."""
        config = SLAIntelligenceConfig(
            org_report_path=self.org_report_path,
            evaluation_windows=[7, 30, 90]
        )
        engine = SLAComplianceEngine(config)

        policy = SLAPolicy(
            policy_id="test_sla",
            policy_name="Test SLA",
            sla_type=SLAType.AVAILABILITY,
            targets=[
                SLATarget(
                    metric_name="availability_score",
                    target_value=80.0,
                    warning_threshold=70.0,
                    breach_threshold=60.0
                )
            ],
            evaluation_windows=[7, 30, 90]
        )

        results = engine.evaluate_compliance([policy], self.org_report)

        self.assertEqual(len(results[0].window_results), 3)

    def test_evaluate_lower_is_better(self):
        """Test evaluation for metrics where lower is better."""
        config = SLAIntelligenceConfig(org_report_path=self.org_report_path)
        engine = SLAComplianceEngine(config)

        # For response time, lower is better
        policy = SLAPolicy(
            policy_id="response_sla",
            policy_name="Response Time SLA",
            sla_type=SLAType.INCIDENT_RESPONSE,
            targets=[
                SLATarget(
                    metric_name="response_time_hours",
                    target_value=4.0,
                    warning_threshold=8.0,
                    breach_threshold=24.0,
                    higher_is_better=False
                )
            ]
        )

        results = engine.evaluate_compliance([policy], self.org_report)

        # Should be compliant since no critical alerts
        self.assertEqual(len(results), 1)


class TestSLABreachAttributionEngine(unittest.TestCase):
    """Test breach attribution functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.org_report_path = Path(self.temp_dir) / "org-health-report.json"

        self.org_report = {
            "org_health_score": 55.0,
            "repositories": [
                {
                    "repo_id": "failing-repo",
                    "repository_score": 45.0,
                    "risk_tier": "critical",
                    "trends": {"overall_trend": "declining"}
                }
            ]
        }

        with open(self.org_report_path, 'w') as f:
            json.dump(self.org_report, f)

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_attribute_breach_to_repos(self):
        """Test breach attribution to repositories."""
        config = SLAIntelligenceConfig(org_report_path=self.org_report_path)
        engine = SLABreachAttributionEngine(config)

        # Create a breached compliance result
        compliance_result = SLAComplianceResult(
            policy_id="test_sla",
            policy_name="Test SLA",
            sla_type=SLAType.AVAILABILITY,
            overall_status=SLAStatus.BREACHED,
            affected_repos=["failing-repo"],
            window_results=[
                SLAWindowResult(
                    window_size=7,
                    window_label="7-interval",
                    actual_value=55.0,
                    target_value=99.0,
                    status=SLAStatus.BREACHED
                )
            ]
        )

        policy = SLAPolicy(
            policy_id="test_sla",
            policy_name="Test SLA",
            sla_type=SLAType.AVAILABILITY,
            targets=[SLATarget("availability", 99.0, 95.0, 90.0)],
            severity_on_breach=SLASeverity.CRITICAL
        )

        breaches = engine.attribute_breaches(
            [compliance_result],
            [policy],
            self.org_report
        )

        self.assertEqual(len(breaches), 1)
        self.assertEqual(breaches[0].severity, SLASeverity.CRITICAL)
        self.assertTrue(len(breaches[0].root_causes) > 0)

    def test_attribute_with_temporal_data(self):
        """Test breach attribution with temporal intelligence data."""
        config = SLAIntelligenceConfig(org_report_path=self.org_report_path)
        engine = SLABreachAttributionEngine(config)

        compliance_result = SLAComplianceResult(
            policy_id="test_sla",
            policy_name="Test SLA",
            sla_type=SLAType.AVAILABILITY,
            overall_status=SLAStatus.BREACHED,
            affected_repos=["repo-a"],
            window_results=[
                SLAWindowResult(
                    window_size=7,
                    window_label="7-interval",
                    actual_value=55.0,
                    target_value=99.0,
                    status=SLAStatus.BREACHED
                )
            ]
        )

        policy = SLAPolicy(
            policy_id="test_sla",
            policy_name="Test SLA",
            sla_type=SLAType.AVAILABILITY,
            targets=[SLATarget("availability", 99.0, 95.0, 90.0)]
        )

        temporal_report = {
            "propagation_paths": [
                {
                    "path_id": "path_001",
                    "repo_sequence": ["repo-a", "repo-b"],
                    "total_lag": 2,
                    "path_confidence": 0.8
                }
            ],
            "anomalies": [
                {
                    "anomaly_id": "anomaly_001",
                    "severity": "high",
                    "title": "Test Anomaly",
                    "message": "Test message",
                    "evidence": ["evidence1"],
                    "affected_repos": ["repo-a"]
                }
            ]
        }

        breaches = engine.attribute_breaches(
            [compliance_result],
            [policy],
            self.org_report,
            temporal_report=temporal_report
        )

        self.assertEqual(len(breaches), 1)
        # Should have root causes from both repo and temporal data
        cause_types = [c.cause_type for c in breaches[0].root_causes]
        self.assertTrue(any(t in cause_types for t in ["repo_degradation", "propagation_path", "temporal_anomaly"]))


class TestExecutiveReadinessEngine(unittest.TestCase):
    """Test executive readiness scoring."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.org_report_path = Path(self.temp_dir) / "org-health-report.json"

        self.org_report = {
            "org_health_score": 85.0,
            "org_health_status": "green",
            "repositories": [
                {
                    "repo_id": "repo-1",
                    "repository_score": 90.0,
                    "risk_tier": "low",
                    "trends": {"overall_trend": "improving"}
                },
                {
                    "repo_id": "repo-2",
                    "repository_score": 80.0,
                    "risk_tier": "medium",
                    "trends": {"overall_trend": "stable"}
                }
            ]
        }

        with open(self.org_report_path, 'w') as f:
            json.dump(self.org_report, f)

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_calculate_readiness_green(self):
        """Test readiness calculation for healthy org."""
        config = SLAIntelligenceConfig(org_report_path=self.org_report_path)
        engine = ExecutiveReadinessEngine(config)

        # All compliant
        compliance_results = [
            SLAComplianceResult(
                policy_id="sla_1",
                policy_name="SLA 1",
                sla_type=SLAType.AVAILABILITY,
                overall_status=SLAStatus.COMPLIANT,
                trend_direction="stable"
            ),
            SLAComplianceResult(
                policy_id="sla_2",
                policy_name="SLA 2",
                sla_type=SLAType.RELIABILITY,
                overall_status=SLAStatus.COMPLIANT,
                trend_direction="improving"
            )
        ]

        readiness = engine.calculate_readiness(
            compliance_results,
            [],
            self.org_report
        )

        self.assertGreaterEqual(readiness.readiness_score, 80.0)
        self.assertEqual(readiness.readiness_tier, ReadinessTier.GREEN)
        self.assertEqual(readiness.compliant_slas, 2)

    def test_calculate_readiness_yellow(self):
        """Test readiness calculation with at-risk SLAs."""
        config = SLAIntelligenceConfig(org_report_path=self.org_report_path)
        engine = ExecutiveReadinessEngine(config)

        compliance_results = [
            SLAComplianceResult(
                policy_id="sla_1",
                policy_name="SLA 1",
                sla_type=SLAType.AVAILABILITY,
                overall_status=SLAStatus.COMPLIANT
            ),
            SLAComplianceResult(
                policy_id="sla_2",
                policy_name="SLA 2",
                sla_type=SLAType.RELIABILITY,
                overall_status=SLAStatus.AT_RISK
            )
        ]

        readiness = engine.calculate_readiness(
            compliance_results,
            [],
            self.org_report
        )

        self.assertEqual(readiness.compliant_slas, 1)
        self.assertEqual(readiness.at_risk_slas, 1)

    def test_calculate_readiness_red(self):
        """Test readiness calculation with breached SLAs."""
        config = SLAIntelligenceConfig(org_report_path=self.org_report_path)
        engine = ExecutiveReadinessEngine(config)

        compliance_results = [
            SLAComplianceResult(
                policy_id="sla_1",
                policy_name="SLA 1",
                sla_type=SLAType.AVAILABILITY,
                overall_status=SLAStatus.BREACHED
            ),
            SLAComplianceResult(
                policy_id="sla_2",
                policy_name="SLA 2",
                sla_type=SLAType.RELIABILITY,
                overall_status=SLAStatus.BREACHED
            )
        ]

        readiness = engine.calculate_readiness(
            compliance_results,
            [],
            self.org_report
        )

        self.assertLess(readiness.readiness_score, 60.0)
        self.assertEqual(readiness.readiness_tier, ReadinessTier.RED)
        self.assertEqual(readiness.breached_slas, 2)

    def test_generate_scorecards(self):
        """Test scorecard generation."""
        config = SLAIntelligenceConfig(org_report_path=self.org_report_path)
        engine = ExecutiveReadinessEngine(config)

        compliance_results = [
            SLAComplianceResult(
                policy_id="sla_1",
                policy_name="Availability SLA",
                sla_type=SLAType.AVAILABILITY,
                overall_status=SLAStatus.COMPLIANT,
                trend_direction="improving",
                window_results=[
                    SLAWindowResult(
                        window_size=7,
                        window_label="7-interval",
                        actual_value=99.5,
                        target_value=99.0,
                        status=SLAStatus.COMPLIANT
                    )
                ]
            )
        ]

        policies = [
            SLAPolicy(
                policy_id="sla_1",
                policy_name="Availability SLA",
                sla_type=SLAType.AVAILABILITY,
                targets=[SLATarget("availability", 99.0, 95.0, 90.0)]
            )
        ]

        scorecards = engine.generate_scorecards(compliance_results, policies)

        self.assertEqual(len(scorecards), 1)
        self.assertEqual(scorecards[0].policy_name, "Availability SLA")
        self.assertEqual(scorecards[0].status, SLAStatus.COMPLIANT)

    def test_generate_risk_narrative(self):
        """Test risk narrative generation."""
        config = SLAIntelligenceConfig(org_report_path=self.org_report_path)
        engine = ExecutiveReadinessEngine(config)

        readiness = ExecutiveReadinessScore(
            readiness_score=85.0,
            readiness_tier=ReadinessTier.GREEN,
            compliant_slas=3,
            at_risk_slas=0,
            breached_slas=0,
            total_slas=3,
            risk_outlook=RiskOutlook.STABLE,
            executive_summary="All systems operational",
            key_concerns=[],
            positive_highlights=["100% SLA compliance"]
        )

        narrative = engine.generate_risk_narrative(readiness, [], [])

        self.assertIsInstance(narrative, RiskNarrative)
        self.assertTrue(len(narrative.headline) > 0)
        self.assertTrue(len(narrative.summary_paragraph) > 0)


class TestSLAIntelligenceEngine(unittest.TestCase):
    """Test main SLA Intelligence Engine."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.org_report_path = Path(self.temp_dir) / "org-health-report.json"
        self.output_path = Path(self.temp_dir) / "sla-intelligence-report.json"

        self.org_report = {
            "org_health_score": 85.0,
            "org_health_status": "green",
            "repositories": [
                {
                    "repo_id": "repo-1",
                    "repository_score": 90.0,
                    "risk_tier": "low",
                    "trends": {"overall_trend": "stable"}
                },
                {
                    "repo_id": "repo-2",
                    "repository_score": 85.0,
                    "risk_tier": "low",
                    "trends": {"overall_trend": "improving"}
                }
            ]
        }

        with open(self.org_report_path, 'w') as f:
            json.dump(self.org_report, f)

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_full_pipeline_compliant(self):
        """Test full pipeline with compliant SLAs."""
        config = SLAIntelligenceConfig(
            org_report_path=self.org_report_path,
            output_path=self.output_path
        )

        engine = SLAIntelligenceEngine(config)
        report, exit_code = engine.run()

        self.assertIsInstance(report, SLAIntelligenceReport)
        self.assertTrue(self.output_path.exists())
        # With default policies and healthy org, should be mostly compliant
        self.assertIn(exit_code, [EXIT_SLA_SUCCESS, EXIT_SLA_AT_RISK])

    def test_exit_code_compliant(self):
        """Test exit code for fully compliant state."""
        # Create a very healthy org report
        healthy_org = {
            "org_health_score": 99.0,
            "org_health_status": "green",
            "repositories": [
                {
                    "repo_id": "repo-1",
                    "repository_score": 99.0,
                    "risk_tier": "low",
                    "trends": {"overall_trend": "stable"}
                }
            ]
        }

        healthy_path = Path(self.temp_dir) / "healthy-report.json"
        with open(healthy_path, 'w') as f:
            json.dump(healthy_org, f)

        config = SLAIntelligenceConfig(org_report_path=healthy_path)
        engine = SLAIntelligenceEngine(config)
        report, exit_code = engine.run()

        # Should be compliant
        self.assertEqual(exit_code, EXIT_SLA_SUCCESS)

    def test_exit_code_at_risk(self):
        """Test exit code for at-risk state."""
        at_risk_org = {
            "org_health_score": 75.0,
            "org_health_status": "yellow",
            "repositories": [
                {
                    "repo_id": "repo-1",
                    "repository_score": 75.0,
                    "risk_tier": "medium",
                    "trends": {"overall_trend": "stable"}
                }
            ]
        }

        at_risk_path = Path(self.temp_dir) / "at-risk-report.json"
        with open(at_risk_path, 'w') as f:
            json.dump(at_risk_org, f)

        config = SLAIntelligenceConfig(org_report_path=at_risk_path)
        engine = SLAIntelligenceEngine(config)
        report, exit_code = engine.run()

        # Should be at risk
        self.assertEqual(exit_code, EXIT_SLA_AT_RISK)

    def test_exit_code_breached(self):
        """Test exit code for breached state."""
        breached_org = {
            "org_health_score": 50.0,
            "org_health_status": "red",
            "repositories": [
                {
                    "repo_id": "repo-1",
                    "repository_score": 50.0,
                    "risk_tier": "critical",
                    "trends": {"overall_trend": "declining"}
                }
            ]
        }

        breached_path = Path(self.temp_dir) / "breached-report.json"
        with open(breached_path, 'w') as f:
            json.dump(breached_org, f)

        config = SLAIntelligenceConfig(org_report_path=breached_path)
        engine = SLAIntelligenceEngine(config)
        report, exit_code = engine.run()

        # Should be breached
        self.assertEqual(exit_code, EXIT_SLA_BREACH)

    def test_report_structure(self):
        """Test generated report structure."""
        config = SLAIntelligenceConfig(
            org_report_path=self.org_report_path,
            output_path=self.output_path
        )

        engine = SLAIntelligenceEngine(config)
        report, _ = engine.run()

        # Verify report structure
        self.assertTrue(len(report.report_id) > 0)
        self.assertTrue(len(report.generated_at) > 0)
        self.assertIsInstance(report.summary, SLAIntelligenceSummary)
        self.assertIsInstance(report.executive_readiness, ExecutiveReadinessScore)
        self.assertIsInstance(report.scorecards, list)
        self.assertIsInstance(report.compliance_results, list)
        self.assertIsInstance(report.risk_narrative, RiskNarrative)

    def test_report_to_dict(self):
        """Test report serialization."""
        config = SLAIntelligenceConfig(org_report_path=self.org_report_path)
        engine = SLAIntelligenceEngine(config)
        report, _ = engine.run()

        report_dict = report.to_dict()

        self.assertIn("report_id", report_dict)
        self.assertIn("summary", report_dict)
        self.assertIn("executive_readiness", report_dict)
        self.assertIn("scorecards", report_dict)
        self.assertIn("breaches", report_dict)
        self.assertIn("risk_narrative", report_dict)


class TestCLIInterface(unittest.TestCase):
    """Test CLI interface."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.org_report_path = Path(self.temp_dir) / "org-health-report.json"

        org_report = {
            "org_health_score": 85.0,
            "org_health_status": "green",
            "repositories": []
        }

        with open(self.org_report_path, 'w') as f:
            json.dump(org_report, f)

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_cli_basic_run(self):
        """Test basic CLI run."""
        from analytics.run_org_sla_intelligence import main

        exit_code = main([
            "--org-report", str(self.org_report_path)
        ])

        self.assertIn(exit_code, [EXIT_SLA_SUCCESS, EXIT_SLA_AT_RISK, EXIT_SLA_BREACH])

    def test_cli_with_output(self):
        """Test CLI with output file."""
        from analytics.run_org_sla_intelligence import main

        output_path = Path(self.temp_dir) / "output.json"

        exit_code = main([
            "--org-report", str(self.org_report_path),
            "--output", str(output_path)
        ])

        self.assertTrue(output_path.exists())

    def test_cli_json_output(self):
        """Test CLI JSON output mode."""
        from analytics.run_org_sla_intelligence import main
        import io
        from contextlib import redirect_stdout

        f = io.StringIO()
        with redirect_stdout(f):
            main([
                "--org-report", str(self.org_report_path),
                "--json"
            ])

        output = f.getvalue()
        # Should be valid JSON
        report = json.loads(output)
        self.assertIn("report_id", report)

    def test_cli_missing_file(self):
        """Test CLI with missing input file."""
        from analytics.run_org_sla_intelligence import main

        exit_code = main([
            "--org-report", "/nonexistent/path.json"
        ])

        self.assertEqual(exit_code, EXIT_SLA_PARSE_ERROR)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_empty_org_report(self):
        """Test handling of empty org report."""
        org_report_path = Path(self.temp_dir) / "empty.json"
        with open(org_report_path, 'w') as f:
            json.dump({"repositories": []}, f)

        config = SLAIntelligenceConfig(org_report_path=org_report_path)
        engine = SLAIntelligenceEngine(config)
        report, exit_code = engine.run()

        self.assertIsInstance(report, SLAIntelligenceReport)

    def test_invalid_json(self):
        """Test handling of invalid JSON."""
        invalid_path = Path(self.temp_dir) / "invalid.json"
        with open(invalid_path, 'w') as f:
            f.write("not valid json{")

        config = SLAIntelligenceConfig(org_report_path=invalid_path)
        engine = SLAIntelligenceEngine(config)
        report, exit_code = engine.run()

        self.assertEqual(exit_code, EXIT_SLA_PARSE_ERROR)

    def test_missing_fields_graceful(self):
        """Test graceful handling of missing fields."""
        minimal_report_path = Path(self.temp_dir) / "minimal.json"
        with open(minimal_report_path, 'w') as f:
            json.dump({"repositories": [{"repo_id": "test"}]}, f)

        config = SLAIntelligenceConfig(org_report_path=minimal_report_path)
        engine = SLAIntelligenceEngine(config)
        report, exit_code = engine.run()

        # Should not crash
        self.assertIsInstance(report, SLAIntelligenceReport)

    def test_large_number_of_repos(self):
        """Test handling of large number of repositories."""
        large_org_path = Path(self.temp_dir) / "large.json"

        repos = [
            {
                "repo_id": f"repo-{i}",
                "repository_score": 50 + (i % 50),
                "risk_tier": "low" if i % 3 == 0 else "medium",
                "trends": {"overall_trend": "stable"}
            }
            for i in range(100)
        ]

        with open(large_org_path, 'w') as f:
            json.dump({
                "org_health_score": 75.0,
                "repositories": repos
            }, f)

        config = SLAIntelligenceConfig(org_report_path=large_org_path)
        engine = SLAIntelligenceEngine(config)
        report, exit_code = engine.run()

        self.assertEqual(report.total_repos, 100)


class TestDataClassSerialization(unittest.TestCase):
    """Test data class serialization/deserialization."""

    def test_window_result_to_dict(self):
        """Test SLAWindowResult serialization."""
        result = SLAWindowResult(
            window_size=7,
            window_label="7-day",
            actual_value=95.5,
            target_value=99.0,
            variance=-3.5,
            status=SLAStatus.AT_RISK
        )

        d = result.to_dict()

        self.assertEqual(d["window_size"], 7)
        self.assertEqual(d["actual_value"], 95.5)
        self.assertEqual(d["status"], "at_risk")

    def test_breach_to_dict(self):
        """Test SLABreach serialization."""
        breach = SLABreach(
            breach_id="breach_001",
            policy_id="sla_001",
            policy_name="Test SLA",
            sla_type=SLAType.AVAILABILITY,
            severity=SLASeverity.HIGH,
            breach_timestamp="2025-01-08T12:00:00",
            breach_window=7,
            root_causes=[
                SLARootCause(
                    cause_id="cause_001",
                    cause_type="repo_degradation",
                    title="Test Cause",
                    description="Test description"
                )
            ]
        )

        d = breach.to_dict()

        self.assertEqual(d["breach_id"], "breach_001")
        self.assertEqual(d["severity"], "high")
        self.assertEqual(len(d["root_causes"]), 1)

    def test_readiness_to_dict(self):
        """Test ExecutiveReadinessScore serialization."""
        readiness = ExecutiveReadinessScore(
            readiness_score=85.0,
            readiness_tier=ReadinessTier.GREEN,
            risk_outlook=RiskOutlook.STABLE
        )

        d = readiness.to_dict()

        self.assertEqual(d["readiness_score"], 85.0)
        self.assertEqual(d["readiness_tier"], "green")
        self.assertEqual(d["risk_outlook"], "stable")


if __name__ == "__main__":
    unittest.main(verbosity=2)
