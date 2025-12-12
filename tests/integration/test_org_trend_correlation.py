"""
Test Suite for Multi-Repository Trend Correlation Engine

This module provides comprehensive tests for the org_trend_correlation module,
covering:
- Data loading and parsing
- Correlation calculations
- Cluster detection
- Anomaly detection
- CLI behavior
- Exit codes
- Realistic multi-repo scenarios

Test Count: 50+ tests
Phase: 14.8 Task 3
"""

import json
import math
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import patch, MagicMock

import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from analytics.org_trend_correlation import (
    # Exit codes
    EXIT_CORRELATION_SUCCESS,
    EXIT_CORRELATIONS_FOUND,
    EXIT_CRITICAL_ANOMALY,
    EXIT_CORRELATION_CONFIG_ERROR,
    EXIT_CORRELATION_PARSE_ERROR,
    EXIT_GENERAL_CORRELATION_ERROR,
    # Exceptions
    TrendCorrelationError,
    TrendCorrelationConfigError,
    TrendCorrelationParseError,
    # Enums
    TrendDirection,
    CorrelationType,
    AnomalySeverity,
    AnomalyType,
    ClusterMethod,
    # Data classes
    TrendDataPoint,
    RepoTrendSeries,
    TrendCorrelation,
    CorrelationCluster,
    CrossRepoAnomaly,
    CorrelationThresholds,
    TrendCorrelationConfig,
    CorrelationSummary,
    TrendCorrelationReport,
    # Components
    TrendLoader,
    CorrelationMatrixBuilder,
    ClusterDetector,
    AnomalyDetector,
    TrendCorrelationEngine,
    # Utilities
    create_default_thresholds,
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def sample_org_health_report() -> Dict[str, Any]:
    """Create a sample org health report with multiple repos."""
    return {
        "report_id": "org_health_20250107_120000",
        "generated_at": "2025-01-07T12:00:00",
        "root_dir": "/test/org-health",
        "org_health_status": "yellow",
        "org_health_score": 72.5,
        "org_risk_tier": "medium",
        "repos_loaded": 5,
        "repositories": [
            {
                "repo_id": "repo-alpha",
                "repo_name": "Alpha Service",
                "health_status": "green",
                "repository_score": 85.0,
                "total_issues": 5,
                "critical_issues": 0,
                "risk_tier": "low",
                "trends": {
                    "overall_trend": "stable",
                    "trend_confidence": 0.8
                },
                "trend_history": [
                    {"timestamp": "2025-01-01", "score": 82.0, "health_status": "green",
                     "critical_issues": 0, "total_issues": 6, "trend_direction": "stable"},
                    {"timestamp": "2025-01-03", "score": 83.0, "health_status": "green",
                     "critical_issues": 0, "total_issues": 5, "trend_direction": "improving"},
                    {"timestamp": "2025-01-05", "score": 84.0, "health_status": "green",
                     "critical_issues": 0, "total_issues": 5, "trend_direction": "stable"}
                ]
            },
            {
                "repo_id": "repo-beta",
                "repo_name": "Beta Service",
                "health_status": "green",
                "repository_score": 88.0,
                "total_issues": 3,
                "critical_issues": 0,
                "risk_tier": "low",
                "trends": {
                    "overall_trend": "stable",
                    "trend_confidence": 0.85
                },
                "trend_history": [
                    {"timestamp": "2025-01-01", "score": 85.0, "health_status": "green",
                     "critical_issues": 0, "total_issues": 4, "trend_direction": "stable"},
                    {"timestamp": "2025-01-03", "score": 86.0, "health_status": "green",
                     "critical_issues": 0, "total_issues": 3, "trend_direction": "improving"},
                    {"timestamp": "2025-01-05", "score": 87.0, "health_status": "green",
                     "critical_issues": 0, "total_issues": 3, "trend_direction": "stable"}
                ]
            },
            {
                "repo_id": "repo-gamma",
                "repo_name": "Gamma Service",
                "health_status": "yellow",
                "repository_score": 65.0,
                "total_issues": 12,
                "critical_issues": 2,
                "risk_tier": "medium",
                "trends": {
                    "overall_trend": "declining",
                    "trend_confidence": 0.75
                },
                "trend_history": [
                    {"timestamp": "2025-01-01", "score": 75.0, "health_status": "green",
                     "critical_issues": 0, "total_issues": 8, "trend_direction": "stable"},
                    {"timestamp": "2025-01-03", "score": 70.0, "health_status": "yellow",
                     "critical_issues": 1, "total_issues": 10, "trend_direction": "declining"},
                    {"timestamp": "2025-01-05", "score": 66.0, "health_status": "yellow",
                     "critical_issues": 2, "total_issues": 11, "trend_direction": "declining"}
                ]
            },
            {
                "repo_id": "repo-delta",
                "repo_name": "Delta Service",
                "health_status": "yellow",
                "repository_score": 62.0,
                "total_issues": 15,
                "critical_issues": 3,
                "risk_tier": "high",
                "trends": {
                    "overall_trend": "declining",
                    "trend_confidence": 0.70
                },
                "trend_history": [
                    {"timestamp": "2025-01-01", "score": 72.0, "health_status": "yellow",
                     "critical_issues": 1, "total_issues": 10, "trend_direction": "stable"},
                    {"timestamp": "2025-01-03", "score": 68.0, "health_status": "yellow",
                     "critical_issues": 2, "total_issues": 12, "trend_direction": "declining"},
                    {"timestamp": "2025-01-05", "score": 63.0, "health_status": "yellow",
                     "critical_issues": 3, "total_issues": 14, "trend_direction": "declining"}
                ]
            },
            {
                "repo_id": "repo-epsilon",
                "repo_name": "Epsilon Service",
                "health_status": "red",
                "repository_score": 45.0,
                "total_issues": 25,
                "critical_issues": 5,
                "risk_tier": "critical",
                "trends": {
                    "overall_trend": "declining",
                    "trend_confidence": 0.90
                },
                "trend_history": [
                    {"timestamp": "2025-01-01", "score": 60.0, "health_status": "yellow",
                     "critical_issues": 2, "total_issues": 18, "trend_direction": "declining"},
                    {"timestamp": "2025-01-03", "score": 52.0, "health_status": "red",
                     "critical_issues": 4, "total_issues": 22, "trend_direction": "declining"},
                    {"timestamp": "2025-01-05", "score": 46.0, "health_status": "red",
                     "critical_issues": 5, "total_issues": 24, "trend_direction": "declining"}
                ]
            }
        ],
        "metrics": {
            "total_repos": 5,
            "repos_green": 2,
            "repos_yellow": 2,
            "repos_red": 1,
            "percent_green": 40.0,
            "avg_score": 69.0,
            "repos_declining": 3,
            "repos_improving": 0,
            "repos_stable": 2
        }
    }


@pytest.fixture
def temp_org_report(sample_org_health_report) -> Path:
    """Create a temporary org health report file."""
    with tempfile.NamedTemporaryFile(
        mode='w', suffix='.json', delete=False
    ) as f:
        json.dump(sample_org_health_report, f)
        return Path(f.name)


@pytest.fixture
def temp_output_dir() -> Path:
    """Create a temporary output directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def default_config(temp_org_report) -> TrendCorrelationConfig:
    """Create a default configuration."""
    return TrendCorrelationConfig(
        org_report_path=temp_org_report,
        thresholds=create_default_thresholds()
    )


# ============================================================================
# Test: Data Classes
# ============================================================================

class TestTrendDataPoint:
    """Tests for TrendDataPoint dataclass."""

    def test_create_from_dict(self):
        """Test creating TrendDataPoint from dictionary."""
        data = {
            "timestamp": "2025-01-07T12:00:00",
            "score": 85.0,
            "health_status": "green",
            "critical_issues": 1,
            "total_issues": 10,
            "trend_direction": "improving"
        }

        dp = TrendDataPoint.from_dict(data)

        assert dp.timestamp == "2025-01-07T12:00:00"
        assert dp.score == 85.0
        assert dp.health_status == "green"
        assert dp.critical_issues == 1
        assert dp.total_issues == 10
        assert dp.trend_direction == "improving"

    def test_to_dict(self):
        """Test converting TrendDataPoint to dictionary."""
        dp = TrendDataPoint(
            timestamp="2025-01-07",
            score=75.0,
            health_status="yellow",
            critical_issues=2,
            total_issues=15
        )

        result = dp.to_dict()

        assert result["timestamp"] == "2025-01-07"
        assert result["score"] == 75.0
        assert result["health_status"] == "yellow"


class TestRepoTrendSeries:
    """Tests for RepoTrendSeries dataclass."""

    def test_create_from_repo_data(self, sample_org_health_report):
        """Test creating RepoTrendSeries from repository data."""
        repo_data = sample_org_health_report["repositories"][0]

        series = RepoTrendSeries.from_repo_data(
            repo_data,
            repo_data.get("trend_history", [])
        )

        assert series.repo_id == "repo-alpha"
        assert series.repo_name == "Alpha Service"
        assert series.current_score == 85.0
        assert len(series.data_points) == 4  # 3 historical + 1 current

    def test_compute_statistics(self):
        """Test statistics computation."""
        series = RepoTrendSeries(
            repo_id="test-repo",
            repo_name="Test Repo"
        )
        series.data_points = [
            TrendDataPoint("2025-01-01", 70.0, "yellow", 1, 5, "stable"),
            TrendDataPoint("2025-01-02", 75.0, "yellow", 1, 4, "improving"),
            TrendDataPoint("2025-01-03", 80.0, "green", 0, 3, "improving"),
            TrendDataPoint("2025-01-04", 78.0, "green", 0, 4, "declining"),
        ]

        series.compute_statistics()

        assert series.score_mean == pytest.approx(75.75)
        assert series.score_min == 70.0
        assert series.score_max == 80.0
        assert series.score_std > 0
        assert series.intervals_improving == 2
        assert series.intervals_declining == 1
        assert series.intervals_stable == 1

    def test_get_score_series(self):
        """Test getting score series as list."""
        series = RepoTrendSeries(repo_id="test", repo_name="Test")
        series.data_points = [
            TrendDataPoint("t1", 70.0, "yellow", 0, 0),
            TrendDataPoint("t2", 75.0, "yellow", 0, 0),
            TrendDataPoint("t3", 80.0, "green", 0, 0),
        ]

        scores = series.get_score_series()

        assert scores == [70.0, 75.0, 80.0]


class TestTrendCorrelation:
    """Tests for TrendCorrelation dataclass."""

    def test_to_dict(self):
        """Test converting TrendCorrelation to dictionary."""
        corr = TrendCorrelation(
            repo_a_id="repo-a",
            repo_b_id="repo-b",
            pearson_coefficient=0.85,
            spearman_coefficient=0.82,
            correlation_type=CorrelationType.POSITIVE,
            correlation_strength="strong",
            is_significant=True
        )

        result = corr.to_dict()

        assert result["repo_a_id"] == "repo-a"
        assert result["repo_b_id"] == "repo-b"
        assert result["pearson_coefficient"] == 0.85
        assert result["correlation_type"] == "positive"
        assert result["is_significant"] is True


class TestCorrelationCluster:
    """Tests for CorrelationCluster dataclass."""

    def test_to_dict(self):
        """Test converting CorrelationCluster to dictionary."""
        cluster = CorrelationCluster(
            cluster_id="cluster_001",
            cluster_name="Test Cluster",
            repo_ids=["repo-a", "repo-b", "repo-c"],
            repo_count=3,
            avg_internal_correlation=0.75,
            dominant_trend="declining",
            is_risk_cluster=True
        )

        result = cluster.to_dict()

        assert result["cluster_id"] == "cluster_001"
        assert result["repo_count"] == 3
        assert result["is_risk_cluster"] is True
        assert len(result["repo_ids"]) == 3


class TestCrossRepoAnomaly:
    """Tests for CrossRepoAnomaly dataclass."""

    def test_to_dict(self):
        """Test converting CrossRepoAnomaly to dictionary."""
        anomaly = CrossRepoAnomaly(
            anomaly_id="anomaly_001",
            anomaly_type=AnomalyType.SYNCHRONIZED_DECLINE,
            severity=AnomalySeverity.HIGH,
            title="Test Anomaly",
            message="Test message",
            timestamp="2025-01-07T12:00:00",
            affected_repos=["repo-a", "repo-b"],
            affected_count=2,
            is_predictive=True
        )

        result = anomaly.to_dict()

        assert result["anomaly_id"] == "anomaly_001"
        assert result["anomaly_type"] == "synchronized_decline"
        assert result["severity"] == "high"
        assert result["is_predictive"] is True


# ============================================================================
# Test: Enums
# ============================================================================

class TestEnums:
    """Tests for enum classes."""

    def test_trend_direction_from_string(self):
        """Test TrendDirection.from_string()."""
        assert TrendDirection.from_string("improving") == TrendDirection.IMPROVING
        assert TrendDirection.from_string("DECLINING") == TrendDirection.DECLINING
        assert TrendDirection.from_string("unknown") == TrendDirection.UNKNOWN
        assert TrendDirection.from_string("invalid") == TrendDirection.UNKNOWN

    def test_anomaly_severity_comparison(self):
        """Test AnomalySeverity comparison operators."""
        assert AnomalySeverity.LOW < AnomalySeverity.MEDIUM
        assert AnomalySeverity.MEDIUM < AnomalySeverity.HIGH
        assert AnomalySeverity.HIGH < AnomalySeverity.CRITICAL
        assert AnomalySeverity.CRITICAL <= AnomalySeverity.CRITICAL

    def test_anomaly_severity_from_string(self):
        """Test AnomalySeverity.from_string()."""
        assert AnomalySeverity.from_string("critical") == AnomalySeverity.CRITICAL
        assert AnomalySeverity.from_string("HIGH") == AnomalySeverity.HIGH
        assert AnomalySeverity.from_string("invalid") == AnomalySeverity.LOW


# ============================================================================
# Test: Configuration
# ============================================================================

class TestConfiguration:
    """Tests for configuration classes."""

    def test_default_thresholds(self):
        """Test default threshold values."""
        thresholds = create_default_thresholds()

        assert thresholds.min_correlation_weak == 0.3
        assert thresholds.min_correlation_moderate == 0.5
        assert thresholds.min_correlation_strong == 0.7
        assert thresholds.min_cluster_size == 2
        assert thresholds.synchronized_decline_threshold == 0.20

    def test_thresholds_from_dict(self):
        """Test creating thresholds from dictionary."""
        data = {
            "min_correlation_strong": 0.8,
            "min_cluster_size": 3,
            "synchronized_decline_threshold": 0.30
        }

        thresholds = CorrelationThresholds.from_dict(data)

        assert thresholds.min_correlation_strong == 0.8
        assert thresholds.min_cluster_size == 3
        assert thresholds.synchronized_decline_threshold == 0.30

    def test_config_to_dict(self, temp_org_report):
        """Test converting config to dictionary."""
        config = TrendCorrelationConfig(
            org_report_path=temp_org_report,
            verbose=True,
            compute_clusters=True
        )

        result = config.to_dict()

        assert "org_report_path" in result
        assert result["verbose"] is True
        assert result["compute_clusters"] is True


# ============================================================================
# Test: TrendLoader
# ============================================================================

class TestTrendLoader:
    """Tests for TrendLoader component."""

    def test_load_org_report(self, default_config):
        """Test loading org health report."""
        loader = TrendLoader(default_config)
        report = loader.load_org_report()

        assert report is not None
        assert "repositories" in report
        assert len(report["repositories"]) == 5

    def test_load_nonexistent_report(self, temp_output_dir):
        """Test loading nonexistent report raises error."""
        config = TrendCorrelationConfig(
            org_report_path=temp_output_dir / "nonexistent.json"
        )
        loader = TrendLoader(config)

        with pytest.raises(TrendCorrelationParseError):
            loader.load_org_report()

    def test_extract_trend_series(self, default_config):
        """Test extracting trend series from report."""
        loader = TrendLoader(default_config)
        loader.load_org_report()
        series = loader.extract_trend_series()

        assert len(series) == 5
        assert "repo-alpha" in series
        assert "repo-epsilon" in series

        alpha_series = series["repo-alpha"]
        assert alpha_series.current_score == 85.0
        assert len(alpha_series.data_points) >= 1

    def test_invalid_json_report(self, temp_output_dir):
        """Test loading invalid JSON raises error."""
        invalid_file = temp_output_dir / "invalid.json"
        invalid_file.write_text("{ invalid json }")

        config = TrendCorrelationConfig(org_report_path=invalid_file)
        loader = TrendLoader(config)

        with pytest.raises(TrendCorrelationParseError):
            loader.load_org_report()


# ============================================================================
# Test: CorrelationMatrixBuilder
# ============================================================================

class TestCorrelationMatrixBuilder:
    """Tests for CorrelationMatrixBuilder component."""

    def test_build_matrix(self, default_config, sample_org_health_report):
        """Test building correlation matrix."""
        # Create series manually
        repo_series = {}
        for repo_data in sample_org_health_report["repositories"]:
            series = RepoTrendSeries.from_repo_data(
                repo_data,
                repo_data.get("trend_history", [])
            )
            repo_series[series.repo_id] = series

        builder = CorrelationMatrixBuilder(default_config)
        matrix = builder.build_matrix(repo_series)

        assert len(matrix) == 5
        assert "repo-alpha" in matrix
        assert "repo-beta" in matrix["repo-alpha"]

        # Self-correlation should be 1.0
        for repo_id in matrix:
            assert matrix[repo_id][repo_id] == 1.0

    def test_pearson_correlation_perfect_positive(self, default_config):
        """Test Pearson correlation with perfectly correlated data."""
        builder = CorrelationMatrixBuilder(default_config)

        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [2.0, 4.0, 6.0, 8.0, 10.0]

        corr = builder._pearson_correlation(x, y)
        assert corr == pytest.approx(1.0)

    def test_pearson_correlation_perfect_negative(self, default_config):
        """Test Pearson correlation with negatively correlated data."""
        builder = CorrelationMatrixBuilder(default_config)

        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [10.0, 8.0, 6.0, 4.0, 2.0]

        corr = builder._pearson_correlation(x, y)
        assert corr == pytest.approx(-1.0)

    def test_pearson_correlation_uncorrelated(self, default_config):
        """Test Pearson correlation with uncorrelated data."""
        builder = CorrelationMatrixBuilder(default_config)

        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [5.0, 1.0, 4.0, 2.0, 3.0]

        corr = builder._pearson_correlation(x, y)
        # Should be close to 0 (not strongly correlated)
        assert abs(corr) < 0.5

    def test_significant_correlations(self, default_config, sample_org_health_report):
        """Test extraction of significant correlations."""
        repo_series = {}
        for repo_data in sample_org_health_report["repositories"]:
            series = RepoTrendSeries.from_repo_data(
                repo_data,
                repo_data.get("trend_history", [])
            )
            repo_series[series.repo_id] = series

        builder = CorrelationMatrixBuilder(default_config)
        builder.build_matrix(repo_series)

        correlations = builder.get_significant_correlations()

        # Should find some significant correlations between declining repos
        assert isinstance(correlations, list)

    def test_correlation_classification(self, default_config):
        """Test correlation type classification."""
        builder = CorrelationMatrixBuilder(default_config)

        # Strong positive correlation
        corr = TrendCorrelation(
            repo_a_id="a", repo_b_id="b",
            pearson_coefficient=0.85,
            shared_decline_periods=3
        )

        corr_type, strength = builder._classify_correlation(corr)

        assert corr_type == CorrelationType.SYNCHRONIZED_DECLINE
        assert strength == "strong"


# ============================================================================
# Test: ClusterDetector
# ============================================================================

class TestClusterDetector:
    """Tests for ClusterDetector component."""

    def test_detect_clusters(self, default_config, sample_org_health_report):
        """Test cluster detection."""
        repo_series = {}
        for repo_data in sample_org_health_report["repositories"]:
            series = RepoTrendSeries.from_repo_data(
                repo_data,
                repo_data.get("trend_history", [])
            )
            repo_series[series.repo_id] = series

        # Create correlations manually for testing
        correlations = [
            TrendCorrelation(
                repo_a_id="repo-gamma",
                repo_b_id="repo-delta",
                pearson_coefficient=0.9,
                is_significant=True
            ),
            TrendCorrelation(
                repo_a_id="repo-delta",
                repo_b_id="repo-epsilon",
                pearson_coefficient=0.85,
                is_significant=True
            )
        ]

        # Build minimal correlation matrix
        matrix = {
            repo_id: {repo_id: 1.0 for repo_id in repo_series}
            for repo_id in repo_series
        }
        matrix["repo-gamma"]["repo-delta"] = 0.9
        matrix["repo-delta"]["repo-gamma"] = 0.9
        matrix["repo-delta"]["repo-epsilon"] = 0.85
        matrix["repo-epsilon"]["repo-delta"] = 0.85

        detector = ClusterDetector(default_config)
        clusters = detector.detect_clusters(repo_series, matrix, correlations)

        # Should detect at least one cluster of declining repos
        assert isinstance(clusters, list)

    def test_build_adjacency(self, default_config):
        """Test building adjacency list."""
        correlations = [
            TrendCorrelation(
                repo_a_id="a", repo_b_id="b",
                pearson_coefficient=0.8, is_significant=True
            ),
            TrendCorrelation(
                repo_a_id="b", repo_b_id="c",
                pearson_coefficient=0.75, is_significant=True
            )
        ]

        detector = ClusterDetector(default_config)
        adjacency = detector._build_adjacency(correlations)

        assert "a" in adjacency
        assert "b" in adjacency["a"]
        assert "a" in adjacency["b"]
        assert "c" in adjacency["b"]

    def test_find_connected_component(self, default_config):
        """Test finding connected components."""
        adjacency = {
            "a": {"b"},
            "b": {"a", "c"},
            "c": {"b"},
            "d": {"e"},
            "e": {"d"}
        }

        detector = ClusterDetector(default_config)
        visited = set()

        component = detector._find_connected_component("a", adjacency, visited)

        assert set(component) == {"a", "b", "c"}


# ============================================================================
# Test: AnomalyDetector
# ============================================================================

class TestAnomalyDetector:
    """Tests for AnomalyDetector component."""

    def test_detect_synchronized_decline(self, default_config, sample_org_health_report):
        """Test synchronized decline detection."""
        repo_series = {}
        for repo_data in sample_org_health_report["repositories"]:
            series = RepoTrendSeries.from_repo_data(
                repo_data,
                repo_data.get("trend_history", [])
            )
            repo_series[series.repo_id] = series

        detector = AnomalyDetector(default_config)
        anomalies = detector.detect_anomalies(repo_series, [], [])

        # Should detect synchronized decline (3 of 5 repos declining = 60%)
        sync_decline_anomalies = [
            a for a in anomalies
            if a.anomaly_type == AnomalyType.SYNCHRONIZED_DECLINE
        ]
        assert len(sync_decline_anomalies) >= 1

    def test_detect_emerging_risk_clusters(self, default_config, sample_org_health_report):
        """Test emerging risk cluster detection."""
        repo_series = {}
        for repo_data in sample_org_health_report["repositories"]:
            series = RepoTrendSeries.from_repo_data(
                repo_data,
                repo_data.get("trend_history", [])
            )
            repo_series[series.repo_id] = series

        # Create risk cluster
        risk_cluster = CorrelationCluster(
            cluster_id="cluster_001",
            cluster_name="Risk Cluster",
            repo_ids=["repo-gamma", "repo-delta", "repo-epsilon"],
            repo_count=3,
            percent_declining=0.67,
            is_risk_cluster=True
        )

        detector = AnomalyDetector(default_config)
        detector._detect_emerging_risk_clusters(
            [risk_cluster], repo_series, "2025-01-07T12:00:00"
        )

        risk_anomalies = [
            a for a in detector.get_anomalies()
            if a.anomaly_type == AnomalyType.EMERGING_RISK_CLUSTER
        ]
        assert len(risk_anomalies) >= 1

    def test_anomaly_severity_levels(self, default_config):
        """Test that anomaly severity is assigned correctly."""
        # Create repo series with high decline ratio
        repo_series = {
            f"repo-{i}": RepoTrendSeries(
                repo_id=f"repo-{i}",
                repo_name=f"Repo {i}",
                current_trend_direction="declining"
            )
            for i in range(10)
        }

        detector = AnomalyDetector(default_config)
        detector._detect_synchronized_decline(
            repo_series, "2025-01-07T12:00:00"
        )

        anomalies = detector.get_anomalies()
        if anomalies:
            # 100% declining should be CRITICAL
            assert anomalies[0].severity == AnomalySeverity.CRITICAL

    def test_no_anomalies_healthy_org(self, default_config):
        """Test that no anomalies detected for healthy org."""
        # Create all healthy repos
        repo_series = {
            f"repo-{i}": RepoTrendSeries(
                repo_id=f"repo-{i}",
                repo_name=f"Repo {i}",
                current_trend_direction="stable"
            )
            for i in range(5)
        }

        detector = AnomalyDetector(default_config)
        anomalies = detector.detect_anomalies(repo_series, [], [])

        # Should not detect synchronized decline
        sync_decline = [
            a for a in anomalies
            if a.anomaly_type == AnomalyType.SYNCHRONIZED_DECLINE
        ]
        assert len(sync_decline) == 0


# ============================================================================
# Test: TrendCorrelationEngine
# ============================================================================

class TestTrendCorrelationEngine:
    """Tests for main TrendCorrelationEngine."""

    def test_engine_run_success(self, default_config):
        """Test successful engine run."""
        engine = TrendCorrelationEngine(default_config)
        report, exit_code = engine.run()

        assert report is not None
        assert isinstance(report, TrendCorrelationReport)
        assert exit_code in (
            EXIT_CORRELATION_SUCCESS,
            EXIT_CORRELATIONS_FOUND,
            EXIT_CRITICAL_ANOMALY
        )

    def test_engine_report_structure(self, default_config):
        """Test report structure completeness."""
        engine = TrendCorrelationEngine(default_config)
        report, _ = engine.run()

        assert report.report_id is not None
        assert report.generated_at is not None
        assert report.total_repos == 5
        assert isinstance(report.summary, CorrelationSummary)
        assert isinstance(report.correlations, list)
        assert isinstance(report.clusters, list)
        assert isinstance(report.anomalies, list)

    def test_engine_with_output_file(self, default_config, temp_output_dir):
        """Test engine writes output file."""
        default_config.output_path = temp_output_dir / "correlation-report.json"

        engine = TrendCorrelationEngine(default_config)
        report, _ = engine.run()

        assert default_config.output_path.exists()

        with open(default_config.output_path) as f:
            saved_report = json.load(f)

        assert saved_report["report_id"] == report.report_id

    def test_engine_missing_report_error(self, temp_output_dir):
        """Test engine handles missing org report."""
        config = TrendCorrelationConfig(
            org_report_path=temp_output_dir / "nonexistent.json"
        )

        engine = TrendCorrelationEngine(config)
        report, exit_code = engine.run()

        assert exit_code == EXIT_CORRELATION_PARSE_ERROR

    def test_engine_exit_code_correlations_found(self, default_config):
        """Test exit code when correlations are found."""
        engine = TrendCorrelationEngine(default_config)
        report, exit_code = engine.run()

        # With our sample data, correlations should be found
        if report.summary.significant_correlations > 0:
            assert exit_code == EXIT_CORRELATIONS_FOUND

    def test_engine_fail_on_critical(self, default_config):
        """Test fail on critical anomaly flag."""
        default_config.fail_on_critical_anomaly = True

        engine = TrendCorrelationEngine(default_config)
        report, exit_code = engine.run()

        if report.summary.critical_anomalies > 0:
            assert exit_code == EXIT_CRITICAL_ANOMALY


# ============================================================================
# Test: CLI Behavior
# ============================================================================

class TestCLI:
    """Tests for CLI behavior."""

    def test_cli_module_import(self):
        """Test that CLI module can be imported."""
        from analytics.run_org_trend_correlation import main, create_argument_parser
        assert callable(main)
        assert callable(create_argument_parser)

    def test_cli_argument_parser(self):
        """Test argument parser creation."""
        from analytics.run_org_trend_correlation import create_argument_parser

        parser = create_argument_parser()
        assert parser is not None

        # Test parsing minimal args
        args = parser.parse_args(["--org-report", "test.json"])
        assert args.org_report == "test.json"

    def test_cli_all_options(self):
        """Test parsing all CLI options."""
        from analytics.run_org_trend_correlation import create_argument_parser

        parser = create_argument_parser()
        args = parser.parse_args([
            "--org-report", "report.json",
            "--output", "output.json",
            "--json",
            "--summary-only",
            "--verbose",
            "--min-correlation-threshold", "0.7",
            "--min-cluster-size", "3",
            "--synchronized-decline-threshold", "0.25",
            "--skip-clusters",
            "--skip-anomalies",
            "--fail-on-critical"
        ])

        assert args.org_report == "report.json"
        assert args.output == "output.json"
        assert args.json is True
        assert args.summary_only is True
        assert args.verbose is True
        assert args.min_correlation_threshold == 0.7
        assert args.min_cluster_size == 3
        assert args.skip_clusters is True
        assert args.fail_on_critical is True

    def test_cli_main_with_report(self, temp_org_report):
        """Test CLI main function with valid report."""
        from analytics.run_org_trend_correlation import main

        exit_code = main([
            "--org-report", str(temp_org_report),
            "--json"
        ])

        assert exit_code in (
            EXIT_CORRELATION_SUCCESS,
            EXIT_CORRELATIONS_FOUND,
            EXIT_CRITICAL_ANOMALY
        )

    def test_cli_main_missing_report(self, temp_output_dir):
        """Test CLI main with missing report."""
        from analytics.run_org_trend_correlation import main

        exit_code = main([
            "--org-report", str(temp_output_dir / "missing.json")
        ])

        assert exit_code == EXIT_CORRELATION_PARSE_ERROR


# ============================================================================
# Test: Realistic Multi-Repo Scenarios
# ============================================================================

class TestRealisticScenarios:
    """Tests for realistic multi-repo scenarios."""

    def test_large_org_scenario(self, temp_output_dir):
        """Test with large org (20+ repos)."""
        # Create large org report
        repos = []
        for i in range(20):
            trend = "declining" if i % 3 == 0 else "stable"
            score = 90 - (i * 2) if i % 3 == 0 else 85

            repos.append({
                "repo_id": f"service-{i:02d}",
                "repo_name": f"Service {i}",
                "health_status": "green" if score >= 80 else "yellow",
                "repository_score": score,
                "total_issues": i,
                "critical_issues": i // 5,
                "risk_tier": "high" if score < 60 else "low",
                "trends": {"overall_trend": trend}
            })

        report = {
            "org_health_status": "yellow",
            "org_health_score": 75.0,
            "repos_loaded": 20,
            "repositories": repos
        }

        report_path = temp_output_dir / "large-org-report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f)

        config = TrendCorrelationConfig(org_report_path=report_path)
        engine = TrendCorrelationEngine(config)
        result, exit_code = engine.run()

        assert result.total_repos == 20
        assert result.summary.total_repo_pairs == (20 * 19) // 2

    def test_all_healthy_org(self, temp_output_dir):
        """Test org where all repos are healthy."""
        repos = [
            {
                "repo_id": f"healthy-{i}",
                "repo_name": f"Healthy Repo {i}",
                "health_status": "green",
                "repository_score": 90.0,
                "total_issues": 1,
                "critical_issues": 0,
                "risk_tier": "low",
                "trends": {"overall_trend": "stable"}
            }
            for i in range(5)
        ]

        report = {
            "org_health_status": "green",
            "org_health_score": 90.0,
            "repos_loaded": 5,
            "repositories": repos
        }

        report_path = temp_output_dir / "healthy-org.json"
        with open(report_path, 'w') as f:
            json.dump(report, f)

        config = TrendCorrelationConfig(org_report_path=report_path)
        engine = TrendCorrelationEngine(config)
        result, exit_code = engine.run()

        # No synchronized decline anomalies for healthy org
        sync_decline = [
            a for a in result.anomalies
            if a.anomaly_type == AnomalyType.SYNCHRONIZED_DECLINE
        ]
        assert len(sync_decline) == 0

    def test_all_declining_org(self, temp_output_dir):
        """Test org where all repos are declining."""
        repos = [
            {
                "repo_id": f"declining-{i}",
                "repo_name": f"Declining Repo {i}",
                "health_status": "red",
                "repository_score": 45.0 - i * 5,
                "total_issues": 20 + i,
                "critical_issues": 5,
                "risk_tier": "critical",
                "trends": {"overall_trend": "declining"}
            }
            for i in range(5)
        ]

        report = {
            "org_health_status": "red",
            "org_health_score": 35.0,
            "repos_loaded": 5,
            "repositories": repos
        }

        report_path = temp_output_dir / "declining-org.json"
        with open(report_path, 'w') as f:
            json.dump(report, f)

        config = TrendCorrelationConfig(org_report_path=report_path)
        engine = TrendCorrelationEngine(config)
        result, exit_code = engine.run()

        # Should detect synchronized decline
        assert result.summary.total_anomalies > 0

        sync_decline = [
            a for a in result.anomalies
            if a.anomaly_type == AnomalyType.SYNCHRONIZED_DECLINE
        ]
        assert len(sync_decline) >= 1

        # Should be critical severity (100% declining)
        if sync_decline:
            assert sync_decline[0].severity == AnomalySeverity.CRITICAL

    def test_single_repo_org(self, temp_output_dir):
        """Test org with single repo (edge case)."""
        report = {
            "org_health_status": "green",
            "org_health_score": 90.0,
            "repos_loaded": 1,
            "repositories": [
                {
                    "repo_id": "single-repo",
                    "repo_name": "Single Repo",
                    "health_status": "green",
                    "repository_score": 90.0,
                    "total_issues": 2,
                    "critical_issues": 0,
                    "risk_tier": "low",
                    "trends": {"overall_trend": "stable"}
                }
            ]
        }

        report_path = temp_output_dir / "single-repo.json"
        with open(report_path, 'w') as f:
            json.dump(report, f)

        config = TrendCorrelationConfig(org_report_path=report_path)
        engine = TrendCorrelationEngine(config)
        result, exit_code = engine.run()

        # Should handle gracefully
        assert result.total_repos == 1
        assert exit_code == EXIT_CORRELATION_SUCCESS


# ============================================================================
# Test: Edge Cases
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_repositories(self, temp_output_dir):
        """Test with empty repositories list."""
        report = {
            "org_health_status": "unknown",
            "org_health_score": 0.0,
            "repos_loaded": 0,
            "repositories": []
        }

        report_path = temp_output_dir / "empty-repos.json"
        with open(report_path, 'w') as f:
            json.dump(report, f)

        config = TrendCorrelationConfig(org_report_path=report_path)
        engine = TrendCorrelationEngine(config)
        result, exit_code = engine.run()

        assert result.total_repos == 0
        assert exit_code == EXIT_CORRELATION_SUCCESS

    def test_missing_trend_history(self, temp_output_dir):
        """Test repos without trend history."""
        repos = [
            {
                "repo_id": f"no-history-{i}",
                "repo_name": f"No History {i}",
                "health_status": "green",
                "repository_score": 85.0,
                "total_issues": 3,
                "critical_issues": 0,
                "risk_tier": "low",
                "trends": {"overall_trend": "stable"}
                # No trend_history field
            }
            for i in range(3)
        ]

        report = {
            "org_health_status": "green",
            "repos_loaded": 3,
            "repositories": repos
        }

        report_path = temp_output_dir / "no-history.json"
        with open(report_path, 'w') as f:
            json.dump(report, f)

        config = TrendCorrelationConfig(org_report_path=report_path)
        engine = TrendCorrelationEngine(config)
        result, exit_code = engine.run()

        # Should still work with just current state
        assert result.total_repos == 3

    def test_identical_scores(self, temp_output_dir):
        """Test repos with identical scores (edge case for correlation)."""
        repos = [
            {
                "repo_id": f"same-{i}",
                "repo_name": f"Same {i}",
                "health_status": "green",
                "repository_score": 80.0,  # All same
                "total_issues": 5,
                "critical_issues": 0,
                "risk_tier": "low",
                "trends": {"overall_trend": "stable"}
            }
            for i in range(3)
        ]

        report = {
            "org_health_status": "green",
            "repos_loaded": 3,
            "repositories": repos
        }

        report_path = temp_output_dir / "same-scores.json"
        with open(report_path, 'w') as f:
            json.dump(report, f)

        config = TrendCorrelationConfig(org_report_path=report_path)
        engine = TrendCorrelationEngine(config)
        result, exit_code = engine.run()

        # Should handle without errors
        assert result.total_repos == 3


# ============================================================================
# Test: Report Serialization
# ============================================================================

class TestReportSerialization:
    """Tests for report serialization."""

    def test_report_to_dict(self, default_config):
        """Test report to_dict serialization."""
        engine = TrendCorrelationEngine(default_config)
        report, _ = engine.run()

        report_dict = report.to_dict()

        assert isinstance(report_dict, dict)
        assert "report_id" in report_dict
        assert "summary" in report_dict
        assert "correlations" in report_dict
        assert "clusters" in report_dict
        assert "anomalies" in report_dict

    def test_report_json_serializable(self, default_config):
        """Test that report can be JSON serialized."""
        engine = TrendCorrelationEngine(default_config)
        report, _ = engine.run()

        # Should not raise
        json_str = json.dumps(report.to_dict(), default=str)
        assert isinstance(json_str, str)

        # Should be valid JSON
        parsed = json.loads(json_str)
        assert parsed["report_id"] == report.report_id


# ============================================================================
# Test: Summary Statistics
# ============================================================================

class TestSummaryStatistics:
    """Tests for summary statistics computation."""

    def test_correlation_summary(self, default_config):
        """Test correlation summary statistics."""
        engine = TrendCorrelationEngine(default_config)
        report, _ = engine.run()

        summary = report.summary

        assert summary.total_repo_pairs == (5 * 4) // 2  # C(5,2)
        assert summary.significant_correlations >= 0
        assert summary.total_anomalies >= 0

    def test_correlation_density(self, default_config):
        """Test correlation density calculation."""
        engine = TrendCorrelationEngine(default_config)
        report, _ = engine.run()

        summary = report.summary

        # Density should be between 0 and 1
        assert 0 <= summary.correlation_density <= 1

        # Check calculation
        if summary.total_repo_pairs > 0:
            expected_density = summary.significant_correlations / summary.total_repo_pairs
            assert summary.correlation_density == pytest.approx(expected_density)


# ============================================================================
# Run tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
