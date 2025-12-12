"""
Integration Tests for Organization Health Governance Engine

Tests the org-level health aggregation, SLO evaluation, and reporting functionality.

Test Coverage:
- Repository discovery from directory structure
- Loading per-repo health data (dashboard, alerts, trends)
- Handling missing/partial artifacts
- SLO policy evaluation (satisfied and violated)
- Tag-based repository selection
- Org risk scoring and tiering
- CLI behavior and exit codes
- Edge cases (zero repos, all failing, mixed states)

Version: 1.0.0
Phase: 14.8 Task 1
"""

import json
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import patch

import pytest

# Import the modules under test
from analytics.org_health_aggregator import (
    OrgHealthConfig,
    OrgHealthEngine,
    OrgHealthAggregator,
    OrgHealthReport,
    OrgMetrics,
    RepositoryHealthSnapshot,
    SloPolicy,
    SloEvaluationResult,
    RepoSelector,
    SloOperator,
    RiskTier,
    HealthStatus,
    TrendDirection,
    AlertSummary,
    TrendSummary,
    RiskTierThresholds,
    Recommendation,
    load_slo_config,
    create_default_slo_policies,
    EXIT_ORG_SUCCESS,
    EXIT_SLO_VIOLATIONS,
    EXIT_HIGH_ORG_RISK,
    EXIT_NO_REPOS_DISCOVERED,
    EXIT_CONFIG_ERROR,
    EXIT_AGGREGATION_ERROR,
    EXIT_GENERAL_ORG_ERROR,
    NoReposDiscoveredError,
    ConfigError,
    AggregationError,
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def temp_org_dir():
    """Create a temporary directory for org health artifacts."""
    temp_dir = tempfile.mkdtemp(prefix="org_health_test_")
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_dashboard_data() -> Dict[str, Any]:
    """Sample dashboard JSON data."""
    return {
        "scan_timestamp": datetime.utcnow().isoformat(),
        "repository_name": "test-repo",
        "overall_health": "green",
        "repository_score": 85.0,
        "total_issues": 5,
        "critical_issues": 0,
        "missing_artifacts": 0,
        "corrupted_artifacts": 0,
        "latest_version": "1.0.0",
        "versions_health": [
            {"version": "1.0.0", "health_status": "green"},
            {"version": "0.9.0", "health_status": "green"}
        ]
    }


@pytest.fixture
def sample_alerts_data() -> Dict[str, Any]:
    """Sample alerts JSON data."""
    return {
        "generated_at": datetime.utcnow().isoformat(),
        "total_alerts": 2,
        "critical_alerts": 0,
        "error_alerts": 1,
        "warning_alerts": 1,
        "info_alerts": 0,
        "alerts": []
    }


@pytest.fixture
def sample_trends_data() -> Dict[str, Any]:
    """Sample trends JSON data."""
    return {
        "generated_at": datetime.utcnow().isoformat(),
        "overall_trend": "improving",
        "trend_confidence": 0.85,
        "predicted_next_score": 87.0,
        "score_volatility": 3.5,
        "total_anomalies": 0,
        "total_warnings": 1
    }


def create_repo_artifacts(
    org_dir: Path,
    repo_id: str,
    dashboard_data: Dict[str, Any] = None,
    alerts_data: Dict[str, Any] = None,
    trends_data: Dict[str, Any] = None
) -> Path:
    """Create artifacts for a single repository."""
    repo_dir = org_dir / repo_id

    if dashboard_data:
        dashboard_dir = repo_dir / "dashboard"
        dashboard_dir.mkdir(parents=True, exist_ok=True)
        with open(dashboard_dir / "health-dashboard.json", 'w') as f:
            json.dump(dashboard_data, f)

    if alerts_data:
        alerts_dir = repo_dir / "alerts"
        alerts_dir.mkdir(parents=True, exist_ok=True)
        with open(alerts_dir / "alerts.json", 'w') as f:
            json.dump(alerts_data, f)

    if trends_data:
        trends_dir = repo_dir / "trends"
        trends_dir.mkdir(parents=True, exist_ok=True)
        with open(trends_dir / "trend-report.json", 'w') as f:
            json.dump(trends_data, f)

    return repo_dir


# ============================================================================
# Test: Repository Discovery
# ============================================================================

class TestRepositoryDiscovery:
    """Tests for repository discovery functionality."""

    def test_discover_single_repo(self, temp_org_dir, sample_dashboard_data):
        """Test discovering a single repository."""
        create_repo_artifacts(temp_org_dir, "repo-a", dashboard_data=sample_dashboard_data)

        config = OrgHealthConfig(root_dir=temp_org_dir)
        aggregator = OrgHealthAggregator(config)
        repos = aggregator.discover_repositories()

        assert len(repos) == 1
        assert "repo-a" in repos

    def test_discover_multiple_repos(self, temp_org_dir, sample_dashboard_data):
        """Test discovering multiple repositories."""
        for repo_id in ["repo-a", "repo-b", "repo-c"]:
            create_repo_artifacts(temp_org_dir, repo_id, dashboard_data=sample_dashboard_data)

        config = OrgHealthConfig(root_dir=temp_org_dir)
        aggregator = OrgHealthAggregator(config)
        repos = aggregator.discover_repositories()

        assert len(repos) == 3
        assert set(repos) == {"repo-a", "repo-b", "repo-c"}

    def test_discover_with_filter(self, temp_org_dir, sample_dashboard_data):
        """Test repository discovery with filter."""
        for repo_id in ["repo-a", "repo-b", "repo-c"]:
            create_repo_artifacts(temp_org_dir, repo_id, dashboard_data=sample_dashboard_data)

        config = OrgHealthConfig(
            root_dir=temp_org_dir,
            repo_filter=["repo-a", "repo-c"]
        )
        aggregator = OrgHealthAggregator(config)
        repos = aggregator.discover_repositories()

        assert len(repos) == 2
        assert set(repos) == {"repo-a", "repo-c"}

    def test_discover_ignores_hidden_dirs(self, temp_org_dir, sample_dashboard_data):
        """Test that hidden directories are ignored."""
        create_repo_artifacts(temp_org_dir, "repo-a", dashboard_data=sample_dashboard_data)
        create_repo_artifacts(temp_org_dir, ".hidden-repo", dashboard_data=sample_dashboard_data)

        config = OrgHealthConfig(root_dir=temp_org_dir)
        aggregator = OrgHealthAggregator(config)
        repos = aggregator.discover_repositories()

        assert len(repos) == 1
        assert "repo-a" in repos
        assert ".hidden-repo" not in repos

    def test_discover_no_repos_raises_error(self, temp_org_dir):
        """Test that empty directory raises NoReposDiscoveredError."""
        config = OrgHealthConfig(root_dir=temp_org_dir)
        aggregator = OrgHealthAggregator(config)

        with pytest.raises(NoReposDiscoveredError):
            aggregator.discover_repositories()

    def test_discover_nonexistent_dir_raises_error(self):
        """Test that nonexistent directory raises NoReposDiscoveredError."""
        config = OrgHealthConfig(root_dir=Path("/nonexistent/path"))
        aggregator = OrgHealthAggregator(config)

        with pytest.raises(NoReposDiscoveredError):
            aggregator.discover_repositories()


# ============================================================================
# Test: Loading Repository Health Data
# ============================================================================

class TestLoadingRepositoryHealth:
    """Tests for loading per-repo health data."""

    def test_load_complete_repo(
        self, temp_org_dir, sample_dashboard_data, sample_alerts_data, sample_trends_data
    ):
        """Test loading a repo with all artifacts."""
        create_repo_artifacts(
            temp_org_dir, "repo-a",
            dashboard_data=sample_dashboard_data,
            alerts_data=sample_alerts_data,
            trends_data=sample_trends_data
        )

        config = OrgHealthConfig(root_dir=temp_org_dir)
        aggregator = OrgHealthAggregator(config)
        aggregator.discover_repositories()
        snapshot = aggregator.load_repository_health("repo-a")

        assert snapshot is not None
        assert snapshot.repo_id == "repo-a"
        assert snapshot.has_dashboard is True
        assert snapshot.has_alerts is True
        assert snapshot.has_trends is True
        assert snapshot.repository_score == 85.0
        assert snapshot.health_status == HealthStatus.GREEN
        assert snapshot.alerts.total_alerts == 2
        assert snapshot.trends.overall_trend == TrendDirection.IMPROVING

    def test_load_repo_dashboard_only(self, temp_org_dir, sample_dashboard_data):
        """Test loading a repo with only dashboard."""
        create_repo_artifacts(temp_org_dir, "repo-a", dashboard_data=sample_dashboard_data)

        config = OrgHealthConfig(root_dir=temp_org_dir)
        aggregator = OrgHealthAggregator(config)
        aggregator.discover_repositories()
        snapshot = aggregator.load_repository_health("repo-a")

        assert snapshot is not None
        assert snapshot.has_dashboard is True
        assert snapshot.has_alerts is False
        assert snapshot.has_trends is False

    def test_load_repo_alerts_only(self, temp_org_dir, sample_alerts_data):
        """Test loading a repo with only alerts."""
        create_repo_artifacts(temp_org_dir, "repo-a", alerts_data=sample_alerts_data)

        config = OrgHealthConfig(root_dir=temp_org_dir)
        aggregator = OrgHealthAggregator(config)
        aggregator.discover_repositories()
        snapshot = aggregator.load_repository_health("repo-a")

        assert snapshot is not None
        assert snapshot.has_dashboard is False
        assert snapshot.has_alerts is True

    def test_load_multiple_repos(
        self, temp_org_dir, sample_dashboard_data, sample_alerts_data
    ):
        """Test loading multiple repositories."""
        for repo_id in ["repo-a", "repo-b", "repo-c"]:
            dashboard = sample_dashboard_data.copy()
            dashboard["repository_name"] = repo_id
            create_repo_artifacts(
                temp_org_dir, repo_id,
                dashboard_data=dashboard,
                alerts_data=sample_alerts_data
            )

        config = OrgHealthConfig(root_dir=temp_org_dir)
        aggregator = OrgHealthAggregator(config)
        count = aggregator.load_all_repositories()

        assert count == 3

    def test_load_repo_with_malformed_json(self, temp_org_dir):
        """Test handling of malformed JSON files."""
        repo_dir = temp_org_dir / "repo-a" / "dashboard"
        repo_dir.mkdir(parents=True)
        with open(repo_dir / "health-dashboard.json", 'w') as f:
            f.write("{ invalid json }")

        config = OrgHealthConfig(root_dir=temp_org_dir)
        aggregator = OrgHealthAggregator(config)
        aggregator.discover_repositories()
        snapshot = aggregator.load_repository_health("repo-a")

        # Should return snapshot but with has_dashboard=False due to error
        assert snapshot is not None
        assert snapshot.has_dashboard is False
        assert len(aggregator._load_errors) > 0


# ============================================================================
# Test: Risk Scoring
# ============================================================================

class TestRiskScoring:
    """Tests for risk score and tier computation."""

    def test_low_risk_green_repo(self, temp_org_dir, sample_dashboard_data):
        """Test that healthy repo gets LOW risk."""
        sample_dashboard_data["repository_score"] = 95.0
        sample_dashboard_data["critical_issues"] = 0
        create_repo_artifacts(temp_org_dir, "repo-a", dashboard_data=sample_dashboard_data)

        config = OrgHealthConfig(root_dir=temp_org_dir)
        aggregator = OrgHealthAggregator(config)
        aggregator.load_all_repositories()
        snapshot = aggregator._repositories.get("repo-a")

        assert snapshot.risk_tier == RiskTier.LOW

    def test_medium_risk_yellow_repo(self, temp_org_dir, sample_dashboard_data):
        """Test that yellow repo gets at least MEDIUM risk."""
        sample_dashboard_data["overall_health"] = "yellow"
        sample_dashboard_data["repository_score"] = 70.0
        sample_dashboard_data["critical_issues"] = 1
        create_repo_artifacts(temp_org_dir, "repo-a", dashboard_data=sample_dashboard_data)

        config = OrgHealthConfig(root_dir=temp_org_dir)
        aggregator = OrgHealthAggregator(config)
        aggregator.load_all_repositories()
        snapshot = aggregator._repositories.get("repo-a")

        assert snapshot.risk_tier in (RiskTier.MEDIUM, RiskTier.HIGH)

    def test_high_risk_low_score(self, temp_org_dir, sample_dashboard_data):
        """Test that low score repo gets HIGH risk."""
        sample_dashboard_data["overall_health"] = "red"
        sample_dashboard_data["repository_score"] = 45.0
        sample_dashboard_data["critical_issues"] = 3
        create_repo_artifacts(temp_org_dir, "repo-a", dashboard_data=sample_dashboard_data)

        config = OrgHealthConfig(root_dir=temp_org_dir)
        aggregator = OrgHealthAggregator(config)
        aggregator.load_all_repositories()
        snapshot = aggregator._repositories.get("repo-a")

        assert snapshot.risk_tier in (RiskTier.HIGH, RiskTier.CRITICAL)

    def test_critical_risk_many_critical_issues(self, temp_org_dir, sample_dashboard_data):
        """Test that many critical issues triggers CRITICAL risk."""
        sample_dashboard_data["overall_health"] = "red"
        sample_dashboard_data["repository_score"] = 35.0
        sample_dashboard_data["critical_issues"] = 10
        create_repo_artifacts(temp_org_dir, "repo-a", dashboard_data=sample_dashboard_data)

        config = OrgHealthConfig(root_dir=temp_org_dir)
        aggregator = OrgHealthAggregator(config)
        aggregator.load_all_repositories()
        snapshot = aggregator._repositories.get("repo-a")

        assert snapshot.risk_tier == RiskTier.CRITICAL


# ============================================================================
# Test: SLO Evaluation
# ============================================================================

class TestSloEvaluation:
    """Tests for SLO policy evaluation."""

    def test_slo_satisfied_all_green(self, temp_org_dir, sample_dashboard_data):
        """Test SLO satisfied when all repos are green."""
        for repo_id in ["repo-a", "repo-b"]:
            dashboard = sample_dashboard_data.copy()
            dashboard["overall_health"] = "green"
            create_repo_artifacts(temp_org_dir, repo_id, dashboard_data=dashboard)

        slo = SloPolicy(
            id="all-green",
            description="All repos must be green",
            repo_selector=RepoSelector(all=True),
            metric="percent_green",
            target=1.0,
            operator=SloOperator.EQUALS
        )

        config = OrgHealthConfig(root_dir=temp_org_dir, slo_policies=[slo])
        aggregator = OrgHealthAggregator(config)
        aggregator.load_all_repositories()
        results = aggregator.evaluate_slos()

        assert len(results) == 1
        assert results[0].satisfied is True
        assert results[0].current_value == 1.0

    def test_slo_violated_not_all_green(self, temp_org_dir, sample_dashboard_data):
        """Test SLO violated when not all repos are green."""
        dashboard_a = sample_dashboard_data.copy()
        dashboard_a["overall_health"] = "green"
        dashboard_b = sample_dashboard_data.copy()
        dashboard_b["overall_health"] = "yellow"

        create_repo_artifacts(temp_org_dir, "repo-a", dashboard_data=dashboard_a)
        create_repo_artifacts(temp_org_dir, "repo-b", dashboard_data=dashboard_b)

        slo = SloPolicy(
            id="all-green",
            description="All repos must be green",
            repo_selector=RepoSelector(all=True),
            metric="percent_green",
            target=1.0,
            operator=SloOperator.EQUALS
        )

        config = OrgHealthConfig(root_dir=temp_org_dir, slo_policies=[slo])
        aggregator = OrgHealthAggregator(config)
        aggregator.load_all_repositories()
        results = aggregator.evaluate_slos()

        assert len(results) == 1
        assert results[0].satisfied is False
        assert results[0].current_value == 0.5
        assert "repo-b" in results[0].violating_repos

    def test_slo_critical_issues_threshold(self, temp_org_dir, sample_dashboard_data):
        """Test SLO for maximum critical issues."""
        dashboard_a = sample_dashboard_data.copy()
        dashboard_a["critical_issues"] = 2
        dashboard_b = sample_dashboard_data.copy()
        dashboard_b["critical_issues"] = 8  # Violates threshold

        create_repo_artifacts(temp_org_dir, "repo-a", dashboard_data=dashboard_a)
        create_repo_artifacts(temp_org_dir, "repo-b", dashboard_data=dashboard_b)

        slo = SloPolicy(
            id="max-critical",
            description="No more than 5 critical issues",
            repo_selector=RepoSelector(all=True),
            metric="critical_issues",
            target=5,
            operator=SloOperator.LESS_THAN_OR_EQUALS
        )

        config = OrgHealthConfig(root_dir=temp_org_dir, slo_policies=[slo])
        aggregator = OrgHealthAggregator(config)
        aggregator.load_all_repositories()
        results = aggregator.evaluate_slos()

        assert len(results) == 1
        assert results[0].satisfied is False
        assert results[0].current_value == 8
        assert "repo-b" in results[0].violating_repos

    def test_slo_tag_based_selection(self, temp_org_dir, sample_dashboard_data):
        """Test SLO with tag-based repo selection."""
        dashboard_a = sample_dashboard_data.copy()
        dashboard_a["overall_health"] = "green"
        dashboard_b = sample_dashboard_data.copy()
        dashboard_b["overall_health"] = "red"  # This is not a core repo

        create_repo_artifacts(temp_org_dir, "repo-a", dashboard_data=dashboard_a)
        create_repo_artifacts(temp_org_dir, "repo-b", dashboard_data=dashboard_b)

        slo = SloPolicy(
            id="core-green",
            description="Core repos must be green",
            repo_selector=RepoSelector(tags=["core"]),
            metric="percent_green",
            target=1.0,
            operator=SloOperator.EQUALS
        )

        config = OrgHealthConfig(
            root_dir=temp_org_dir,
            slo_policies=[slo],
            repo_tags={"repo-a": ["core"]}  # Only repo-a is core
        )
        aggregator = OrgHealthAggregator(config)
        aggregator.load_all_repositories()
        results = aggregator.evaluate_slos()

        assert len(results) == 1
        # Only repo-a is evaluated (core tag), and it's green
        assert results[0].satisfied is True
        assert results[0].repos_evaluated == 1

    def test_multiple_slo_policies(self, temp_org_dir, sample_dashboard_data):
        """Test evaluation of multiple SLO policies."""
        for repo_id in ["repo-a", "repo-b"]:
            create_repo_artifacts(temp_org_dir, repo_id, dashboard_data=sample_dashboard_data)

        slos = [
            SloPolicy(
                id="percent-green",
                description="80% green",
                repo_selector=RepoSelector(all=True),
                metric="percent_green",
                target=0.8,
                operator=SloOperator.GREATER_THAN_OR_EQUALS
            ),
            SloPolicy(
                id="min-score",
                description="Min score 70",
                repo_selector=RepoSelector(all=True),
                metric="repository_score",
                target=70.0,
                operator=SloOperator.GREATER_THAN_OR_EQUALS,
                aggregation="min"
            )
        ]

        config = OrgHealthConfig(root_dir=temp_org_dir, slo_policies=slos)
        aggregator = OrgHealthAggregator(config)
        aggregator.load_all_repositories()
        results = aggregator.evaluate_slos()

        assert len(results) == 2


# ============================================================================
# Test: Org-Level Metrics
# ============================================================================

class TestOrgMetrics:
    """Tests for org-level metrics computation."""

    def test_compute_metrics_all_green(self, temp_org_dir, sample_dashboard_data):
        """Test metrics when all repos are green."""
        for repo_id in ["repo-a", "repo-b", "repo-c"]:
            dashboard = sample_dashboard_data.copy()
            dashboard["overall_health"] = "green"
            dashboard["repository_score"] = 90.0
            create_repo_artifacts(temp_org_dir, repo_id, dashboard_data=dashboard)

        config = OrgHealthConfig(root_dir=temp_org_dir)
        aggregator = OrgHealthAggregator(config)
        aggregator.load_all_repositories()
        metrics = aggregator.compute_org_metrics()

        assert metrics.total_repos == 3
        assert metrics.repos_green == 3
        assert metrics.repos_yellow == 0
        assert metrics.repos_red == 0
        assert metrics.percent_green == 100.0
        assert metrics.avg_score == 90.0

    def test_compute_metrics_mixed_health(self, temp_org_dir, sample_dashboard_data):
        """Test metrics with mixed health statuses."""
        statuses = [("green", 90.0), ("yellow", 70.0), ("red", 40.0)]
        for i, (status, score) in enumerate(statuses):
            dashboard = sample_dashboard_data.copy()
            dashboard["overall_health"] = status
            dashboard["repository_score"] = score
            create_repo_artifacts(temp_org_dir, f"repo-{i}", dashboard_data=dashboard)

        config = OrgHealthConfig(root_dir=temp_org_dir)
        aggregator = OrgHealthAggregator(config)
        aggregator.load_all_repositories()
        metrics = aggregator.compute_org_metrics()

        assert metrics.total_repos == 3
        assert metrics.repos_green == 1
        assert metrics.repos_yellow == 1
        assert metrics.repos_red == 1
        assert metrics.percent_green == pytest.approx(33.33, rel=0.1)
        assert metrics.min_score == 40.0
        assert metrics.max_score == 90.0

    def test_compute_metrics_issue_totals(self, temp_org_dir, sample_dashboard_data):
        """Test issue totals aggregation."""
        issues = [(10, 2), (5, 0), (15, 5)]
        for i, (total, critical) in enumerate(issues):
            dashboard = sample_dashboard_data.copy()
            dashboard["total_issues"] = total
            dashboard["critical_issues"] = critical
            create_repo_artifacts(temp_org_dir, f"repo-{i}", dashboard_data=dashboard)

        config = OrgHealthConfig(root_dir=temp_org_dir)
        aggregator = OrgHealthAggregator(config)
        aggregator.load_all_repositories()
        metrics = aggregator.compute_org_metrics()

        assert metrics.total_issues == 30
        assert metrics.total_critical_issues == 7
        assert metrics.max_issues_per_repo == 15
        assert metrics.max_critical_per_repo == 5


# ============================================================================
# Test: Recommendations
# ============================================================================

class TestRecommendations:
    """Tests for recommendation generation."""

    def test_recommendation_for_critical_repos(self, temp_org_dir, sample_dashboard_data):
        """Test that critical risk repos trigger recommendations."""
        dashboard = sample_dashboard_data.copy()
        dashboard["overall_health"] = "red"
        dashboard["repository_score"] = 30.0
        dashboard["critical_issues"] = 10
        create_repo_artifacts(temp_org_dir, "repo-a", dashboard_data=dashboard)

        config = OrgHealthConfig(root_dir=temp_org_dir)
        aggregator = OrgHealthAggregator(config)
        aggregator.load_all_repositories()
        aggregator.compute_org_metrics()
        recommendations = aggregator.generate_recommendations()

        # Should have critical risk recommendation
        critical_recs = [r for r in recommendations if r.priority == "critical"]
        assert len(critical_recs) > 0
        assert "repo-a" in critical_recs[0].affected_repos

    def test_recommendation_for_slo_violations(self, temp_org_dir, sample_dashboard_data):
        """Test that SLO violations trigger recommendations."""
        dashboard = sample_dashboard_data.copy()
        dashboard["overall_health"] = "yellow"  # Not green
        create_repo_artifacts(temp_org_dir, "repo-a", dashboard_data=dashboard)

        slo = SloPolicy(
            id="all-green",
            description="All repos must be green",
            repo_selector=RepoSelector(all=True),
            metric="percent_green",
            target=1.0,
            operator=SloOperator.EQUALS
        )

        config = OrgHealthConfig(root_dir=temp_org_dir, slo_policies=[slo])
        aggregator = OrgHealthAggregator(config)
        aggregator.load_all_repositories()
        aggregator.compute_org_metrics()
        aggregator.evaluate_slos()
        recommendations = aggregator.generate_recommendations()

        # Should have SLO violation recommendation
        slo_recs = [r for r in recommendations if "SLO" in r.title]
        assert len(slo_recs) > 0


# ============================================================================
# Test: Report Generation
# ============================================================================

class TestReportGeneration:
    """Tests for org health report generation."""

    def test_generate_complete_report(
        self, temp_org_dir, sample_dashboard_data, sample_alerts_data, sample_trends_data
    ):
        """Test generating a complete org health report."""
        for repo_id in ["repo-a", "repo-b"]:
            create_repo_artifacts(
                temp_org_dir, repo_id,
                dashboard_data=sample_dashboard_data,
                alerts_data=sample_alerts_data,
                trends_data=sample_trends_data
            )

        config = OrgHealthConfig(
            root_dir=temp_org_dir,
            slo_policies=create_default_slo_policies()
        )
        aggregator = OrgHealthAggregator(config)
        aggregator.load_all_repositories()
        report = aggregator.generate_org_health_report()

        assert report.report_id is not None
        assert report.generated_at is not None
        assert report.repos_discovered == 2
        assert report.repos_loaded == 2
        assert report.metrics.total_repos == 2
        assert len(report.repositories) == 2
        assert len(report.slo_results) > 0

    def test_report_serialization(self, temp_org_dir, sample_dashboard_data):
        """Test that report can be serialized to JSON."""
        create_repo_artifacts(temp_org_dir, "repo-a", dashboard_data=sample_dashboard_data)

        config = OrgHealthConfig(root_dir=temp_org_dir)
        aggregator = OrgHealthAggregator(config)
        aggregator.load_all_repositories()
        report = aggregator.generate_org_health_report()

        # Should serialize without errors
        report_dict = report.to_dict()
        json_str = json.dumps(report_dict, default=str)
        assert len(json_str) > 0

        # Should deserialize back
        parsed = json.loads(json_str)
        assert parsed["repos_loaded"] == 1


# ============================================================================
# Test: Engine & CLI Behavior
# ============================================================================

class TestEngineAndCLI:
    """Tests for the main engine and CLI behavior."""

    def test_engine_success_exit_code(self, temp_org_dir, sample_dashboard_data):
        """Test successful run returns EXIT_ORG_SUCCESS."""
        for repo_id in ["repo-a", "repo-b"]:
            create_repo_artifacts(temp_org_dir, repo_id, dashboard_data=sample_dashboard_data)

        config = OrgHealthConfig(
            root_dir=temp_org_dir,
            fail_on_slo_violation=False,
            fail_on_critical_org_risk=False
        )
        engine = OrgHealthEngine(config)
        report, exit_code = engine.run()

        assert exit_code == EXIT_ORG_SUCCESS
        assert report.repos_loaded == 2

    def test_engine_slo_violation_exit_code(self, temp_org_dir, sample_dashboard_data):
        """Test SLO violation returns EXIT_SLO_VIOLATIONS when configured."""
        dashboard = sample_dashboard_data.copy()
        dashboard["overall_health"] = "red"
        create_repo_artifacts(temp_org_dir, "repo-a", dashboard_data=dashboard)

        slo = SloPolicy(
            id="all-green",
            description="All repos must be green",
            repo_selector=RepoSelector(all=True),
            metric="percent_green",
            target=1.0,
            operator=SloOperator.EQUALS
        )

        config = OrgHealthConfig(
            root_dir=temp_org_dir,
            slo_policies=[slo],
            fail_on_slo_violation=True
        )
        engine = OrgHealthEngine(config)
        report, exit_code = engine.run()

        assert exit_code == EXIT_SLO_VIOLATIONS

    def test_engine_critical_risk_exit_code(self, temp_org_dir, sample_dashboard_data):
        """Test critical org risk returns EXIT_HIGH_ORG_RISK when configured."""
        dashboard = sample_dashboard_data.copy()
        dashboard["overall_health"] = "red"
        dashboard["repository_score"] = 20.0
        dashboard["critical_issues"] = 15
        create_repo_artifacts(temp_org_dir, "repo-a", dashboard_data=dashboard)

        config = OrgHealthConfig(
            root_dir=temp_org_dir,
            fail_on_critical_org_risk=True
        )
        engine = OrgHealthEngine(config)
        report, exit_code = engine.run()

        assert exit_code == EXIT_HIGH_ORG_RISK

    def test_engine_no_repos_exit_code(self, temp_org_dir):
        """Test empty org returns EXIT_NO_REPOS_DISCOVERED."""
        config = OrgHealthConfig(root_dir=temp_org_dir)
        engine = OrgHealthEngine(config)
        report, exit_code = engine.run()

        assert exit_code == EXIT_NO_REPOS_DISCOVERED

    def test_engine_writes_output_file(self, temp_org_dir, sample_dashboard_data):
        """Test that engine writes output file when configured."""
        create_repo_artifacts(temp_org_dir, "repo-a", dashboard_data=sample_dashboard_data)

        output_path = temp_org_dir / "output" / "report.json"
        config = OrgHealthConfig(
            root_dir=temp_org_dir,
            output_path=output_path
        )
        engine = OrgHealthEngine(config)
        report, exit_code = engine.run()

        assert output_path.exists()
        with open(output_path) as f:
            saved_report = json.load(f)
        assert saved_report["repos_loaded"] == 1


# ============================================================================
# Test: Edge Cases
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_zero_repos_loaded(self, temp_org_dir):
        """Test handling when no repos can be loaded."""
        # Create a repo dir but no valid artifacts
        (temp_org_dir / "repo-a" / "invalid").mkdir(parents=True)

        config = OrgHealthConfig(root_dir=temp_org_dir)
        aggregator = OrgHealthAggregator(config)

        with pytest.raises(NoReposDiscoveredError):
            aggregator.discover_repositories()

    def test_all_repos_failing_to_load(self, temp_org_dir):
        """Test when all repos fail to load due to bad data."""
        # Create repos with invalid JSON
        for repo_id in ["repo-a", "repo-b"]:
            repo_dir = temp_org_dir / repo_id / "dashboard"
            repo_dir.mkdir(parents=True)
            with open(repo_dir / "health-dashboard.json", 'w') as f:
                f.write("invalid json")

        config = OrgHealthConfig(root_dir=temp_org_dir)
        aggregator = OrgHealthAggregator(config)
        aggregator.discover_repositories()
        count = aggregator.load_all_repositories()

        # Repos discovered but all failed to load
        assert len(aggregator._discovered_repos) == 2
        assert count == 0 or len(aggregator._load_errors) == 2

    def test_mixed_success_failure_loading(self, temp_org_dir, sample_dashboard_data):
        """Test when some repos load successfully and others fail."""
        # Valid repo
        create_repo_artifacts(temp_org_dir, "repo-a", dashboard_data=sample_dashboard_data)

        # Invalid repo
        repo_dir = temp_org_dir / "repo-b" / "dashboard"
        repo_dir.mkdir(parents=True)
        with open(repo_dir / "health-dashboard.json", 'w') as f:
            f.write("invalid json")

        config = OrgHealthConfig(root_dir=temp_org_dir)
        aggregator = OrgHealthAggregator(config)
        aggregator.discover_repositories()
        count = aggregator.load_all_repositories()

        assert len(aggregator._discovered_repos) == 2
        # At least repo-a should load
        assert "repo-a" in aggregator._repositories

    def test_slo_with_no_matching_repos(self, temp_org_dir, sample_dashboard_data):
        """Test SLO evaluation when selector matches no repos."""
        create_repo_artifacts(temp_org_dir, "repo-a", dashboard_data=sample_dashboard_data)

        slo = SloPolicy(
            id="nonexistent-tag",
            description="SLO for nonexistent tag",
            repo_selector=RepoSelector(tags=["nonexistent"]),
            metric="percent_green",
            target=1.0,
            operator=SloOperator.EQUALS
        )

        config = OrgHealthConfig(root_dir=temp_org_dir, slo_policies=[slo])
        aggregator = OrgHealthAggregator(config)
        aggregator.load_all_repositories()
        results = aggregator.evaluate_slos()

        assert len(results) == 1
        # Vacuously true when no repos match
        assert results[0].satisfied is True
        assert results[0].repos_evaluated == 0


# ============================================================================
# Test: SLO Policy Configuration
# ============================================================================

class TestSloConfiguration:
    """Tests for SLO policy configuration parsing."""

    def test_repo_selector_all_match(self):
        """Test RepoSelector with all=True."""
        selector = RepoSelector(all=True)
        assert selector.matches("any-repo", []) is True
        assert selector.matches("another-repo", ["tag1"]) is True

    def test_repo_selector_tag_match(self):
        """Test RepoSelector with tags."""
        selector = RepoSelector(tags=["core", "critical"])
        assert selector.matches("repo-a", ["core"]) is True
        assert selector.matches("repo-b", ["critical"]) is True
        assert selector.matches("repo-c", ["other"]) is False
        assert selector.matches("repo-d", []) is False

    def test_repo_selector_pattern_match(self):
        """Test RepoSelector with id_pattern."""
        selector = RepoSelector(id_pattern=r"^api-.*")
        assert selector.matches("api-gateway", []) is True
        assert selector.matches("api-service", []) is True
        assert selector.matches("web-frontend", []) is False

    def test_slo_operator_comparisons(self):
        """Test all SLO operator comparisons."""
        from analytics.org_health_aggregator import OrgHealthAggregator

        # Create a minimal aggregator just for testing the method
        config = OrgHealthConfig(root_dir=Path("/tmp"))
        aggregator = OrgHealthAggregator(config)

        # Test all operators
        assert aggregator._check_slo_condition(5, 5, SloOperator.EQUALS) is True
        assert aggregator._check_slo_condition(4, 5, SloOperator.EQUALS) is False

        assert aggregator._check_slo_condition(4, 5, SloOperator.NOT_EQUALS) is True
        assert aggregator._check_slo_condition(5, 5, SloOperator.NOT_EQUALS) is False

        assert aggregator._check_slo_condition(4, 5, SloOperator.LESS_THAN) is True
        assert aggregator._check_slo_condition(5, 5, SloOperator.LESS_THAN) is False

        assert aggregator._check_slo_condition(5, 5, SloOperator.LESS_THAN_OR_EQUALS) is True
        assert aggregator._check_slo_condition(6, 5, SloOperator.LESS_THAN_OR_EQUALS) is False

        assert aggregator._check_slo_condition(6, 5, SloOperator.GREATER_THAN) is True
        assert aggregator._check_slo_condition(5, 5, SloOperator.GREATER_THAN) is False

        assert aggregator._check_slo_condition(5, 5, SloOperator.GREATER_THAN_OR_EQUALS) is True
        assert aggregator._check_slo_condition(4, 5, SloOperator.GREATER_THAN_OR_EQUALS) is False

    def test_default_slo_policies(self):
        """Test that default SLO policies are created correctly."""
        policies = create_default_slo_policies()
        assert len(policies) >= 3

        # Check that essential policies exist
        policy_ids = [p.id for p in policies]
        assert any("green" in pid for pid in policy_ids)
        assert any("critical" in pid for pid in policy_ids)


# ============================================================================
# Test: Config Loading
# ============================================================================

class TestConfigLoading:
    """Tests for configuration file loading."""

    def test_load_json_config(self, temp_org_dir):
        """Test loading configuration from JSON file."""
        config_data = {
            "slo_policies": [
                {
                    "id": "test-slo",
                    "description": "Test SLO",
                    "repo_selector": {"all": True},
                    "metric": "percent_green",
                    "target": 0.9,
                    "operator": ">="
                }
            ]
        }

        config_path = temp_org_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config_data, f)

        policies = load_slo_config(config_path)
        assert len(policies) == 1
        assert policies[0].id == "test-slo"
        assert policies[0].target == 0.9

    def test_load_yaml_config_without_pyyaml(self, temp_org_dir):
        """Test that YAML loading raises error without PyYAML."""
        config_path = temp_org_dir / "config.yaml"
        config_path.write_text("slo_policies: []")

        # This may or may not raise depending on if PyYAML is installed
        try:
            policies = load_slo_config(config_path)
            # If it succeeds, PyYAML is installed
            assert isinstance(policies, list)
        except ConfigError as e:
            # PyYAML not installed
            assert "pyyaml" in str(e).lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
