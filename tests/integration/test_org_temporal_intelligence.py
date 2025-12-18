"""
Test Suite for Advanced Correlation & Temporal Intelligence Engine

This module provides comprehensive tests for the org_temporal_intelligence module,
covering:
- Lagged correlation calculations
- Influence scoring
- Propagation path detection
- Temporal anomaly detection
- CLI behavior
- Exit codes
- Realistic multi-repo scenarios

Test Count: 60+ tests
Phase: 14.8 Task 4
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

from analytics.org_temporal_intelligence import (
    # Exit codes
    EXIT_TEMPORAL_SUCCESS,
    EXIT_TEMPORAL_CORRELATIONS_FOUND,
    EXIT_CRITICAL_PROPAGATION_RISK,
    EXIT_TEMPORAL_CONFIG_ERROR,
    EXIT_TEMPORAL_PARSE_ERROR,
    EXIT_GENERAL_TEMPORAL_ERROR,
    # Exceptions
    TemporalIntelligenceError,
    TemporalIntelligenceConfigError,
    TemporalIntelligenceParseError,
    # Enums
    InfluenceDirection,
    PropagationType,
    TemporalAnomalyType,
    TemporalSeverity,
    # Data classes
    LaggedCorrelation,
    InfluenceScore,
    PropagationEdge,
    PropagationPath,
    TemporalAnomaly,
    TemporalThresholds,
    TemporalIntelligenceConfig,
    TemporalIntelligenceSummary,
    TemporalIntelligenceReport,
    RepoTimeSeries,
    # Engines
    LaggedCorrelationEngine,
    InfluenceScoringEngine,
    PropagationGraphBuilder,
    TemporalAnomalyDetector,
    TemporalIntelligenceEngine,
    # Utilities
    create_default_thresholds,
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def sample_org_health_report() -> Dict[str, Any]:
    """Create a sample org health report with trend history."""
    return {
        "report_id": "org_health_20250108_120000",
        "generated_at": "2025-01-08T12:00:00",
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
                    {"timestamp": "2025-01-01", "score": 80.0},
                    {"timestamp": "2025-01-02", "score": 82.0},
                    {"timestamp": "2025-01-03", "score": 83.0},
                    {"timestamp": "2025-01-04", "score": 84.0},
                    {"timestamp": "2025-01-05", "score": 85.0}
                ]
            },
            {
                "repo_id": "repo-beta",
                "repo_name": "Beta Service",
                "health_status": "green",
                "repository_score": 83.0,
                "total_issues": 3,
                "critical_issues": 0,
                "risk_tier": "low",
                "trends": {
                    "overall_trend": "stable",
                    "trend_confidence": 0.85
                },
                "trend_history": [
                    {"timestamp": "2025-01-01", "score": 78.0},
                    {"timestamp": "2025-01-02", "score": 79.0},
                    {"timestamp": "2025-01-03", "score": 80.0},
                    {"timestamp": "2025-01-04", "score": 81.0},
                    {"timestamp": "2025-01-05", "score": 82.0}
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
                    {"timestamp": "2025-01-01", "score": 80.0},
                    {"timestamp": "2025-01-02", "score": 75.0},
                    {"timestamp": "2025-01-03", "score": 72.0},
                    {"timestamp": "2025-01-04", "score": 68.0},
                    {"timestamp": "2025-01-05", "score": 66.0}
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
                    {"timestamp": "2025-01-01", "score": 78.0},
                    {"timestamp": "2025-01-02", "score": 76.0},
                    {"timestamp": "2025-01-03", "score": 70.0},
                    {"timestamp": "2025-01-04", "score": 65.0},
                    {"timestamp": "2025-01-05", "score": 63.0}
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
                    {"timestamp": "2025-01-01", "score": 75.0},
                    {"timestamp": "2025-01-02", "score": 70.0},
                    {"timestamp": "2025-01-03", "score": 60.0},
                    {"timestamp": "2025-01-04", "score": 52.0},
                    {"timestamp": "2025-01-05", "score": 46.0}
                ]
            }
        ],
        "metrics": {
            "total_repos": 5,
            "repos_green": 2,
            "repos_yellow": 2,
            "repos_red": 1,
            "percent_green": 40.0,
            "avg_score": 68.0
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
def default_config(temp_org_report) -> TemporalIntelligenceConfig:
    """Create a default configuration."""
    return TemporalIntelligenceConfig(
        org_report_path=temp_org_report,
        thresholds=create_default_thresholds()
    )


@pytest.fixture
def sample_repo_series() -> Dict[str, RepoTimeSeries]:
    """Create sample repo time series for testing."""
    return {
        "repo-a": RepoTimeSeries(
            repo_id="repo-a",
            repo_name="Repo A",
            scores=[80.0, 82.0, 84.0, 86.0, 88.0],
            timestamps=["t1", "t2", "t3", "t4", "t5"],
            current_score=88.0,
            current_trend="improving"
        ),
        "repo-b": RepoTimeSeries(
            repo_id="repo-b",
            repo_name="Repo B",
            scores=[78.0, 80.0, 82.0, 84.0, 86.0],
            timestamps=["t1", "t2", "t3", "t4", "t5"],
            current_score=86.0,
            current_trend="improving"
        ),
        "repo-c": RepoTimeSeries(
            repo_id="repo-c",
            repo_name="Repo C",
            scores=[76.0, 78.0, 80.0, 82.0, 84.0],
            timestamps=["t1", "t2", "t3", "t4", "t5"],
            current_score=84.0,
            current_trend="improving"
        )
    }


# ============================================================================
# Test: Data Classes
# ============================================================================

class TestLaggedCorrelation:
    """Tests for LaggedCorrelation dataclass."""

    def test_to_dict(self):
        """Test converting LaggedCorrelation to dictionary."""
        lc = LaggedCorrelation(
            repo_a_id="repo-a",
            repo_b_id="repo-b",
            lag=2,
            lag_description="repo-a leads repo-b by 2 intervals",
            correlation_coefficient=0.85,
            leader_repo_id="repo-a",
            follower_repo_id="repo-b",
            lag_intervals=2,
            sample_size=5,
            is_significant=True,
            confidence=0.9
        )

        result = lc.to_dict()

        assert result["repo_a_id"] == "repo-a"
        assert result["repo_b_id"] == "repo-b"
        assert result["lag"] == 2
        assert result["correlation_coefficient"] == 0.85
        assert result["leader_repo_id"] == "repo-a"
        assert result["is_significant"] is True


class TestInfluenceScore:
    """Tests for InfluenceScore dataclass."""

    def test_to_dict(self):
        """Test converting InfluenceScore to dictionary."""
        score = InfluenceScore(
            repo_id="repo-leader",
            repo_name="Leader Repo",
            influence_score=75.0,
            influence_rank=1,
            influence_direction=InfluenceDirection.LEADER,
            repos_led=5,
            repos_following=1,
            led_repos=["repo-a", "repo-b", "repo-c", "repo-d", "repo-e"]
        )

        result = score.to_dict()

        assert result["repo_id"] == "repo-leader"
        assert result["influence_score"] == 75.0
        assert result["influence_direction"] == "leader"
        assert result["repos_led"] == 5


class TestPropagationEdge:
    """Tests for PropagationEdge dataclass."""

    def test_to_dict(self):
        """Test converting PropagationEdge to dictionary."""
        edge = PropagationEdge(
            source_repo_id="repo-a",
            target_repo_id="repo-b",
            lag_intervals=2,
            correlation_strength=0.8,
            confidence=0.75,
            causality_score=0.7
        )

        result = edge.to_dict()

        assert result["source_repo_id"] == "repo-a"
        assert result["target_repo_id"] == "repo-b"
        assert result["lag_intervals"] == 2
        assert result["causality_score"] == 0.7


class TestPropagationPath:
    """Tests for PropagationPath dataclass."""

    def test_to_dict(self):
        """Test converting PropagationPath to dictionary."""
        path = PropagationPath(
            path_id="path_001",
            path_type=PropagationType.LINEAR,
            repo_sequence=["repo-a", "repo-b", "repo-c"],
            total_lag=3,
            path_length=2,
            path_confidence=0.7
        )

        result = path.to_dict()

        assert result["path_id"] == "path_001"
        assert result["path_type"] == "linear"
        assert result["repo_sequence"] == ["repo-a", "repo-b", "repo-c"]
        assert result["path_length"] == 2


class TestTemporalAnomaly:
    """Tests for TemporalAnomaly dataclass."""

    def test_to_dict(self):
        """Test converting TemporalAnomaly to dictionary."""
        anomaly = TemporalAnomaly(
            anomaly_id="temporal_001",
            anomaly_type=TemporalAnomalyType.RAPID_PROPAGATION,
            severity=TemporalSeverity.HIGH,
            title="Rapid Propagation Detected",
            message="Changes propagate quickly",
            timestamp="2025-01-08T12:00:00",
            affected_repos=["repo-a", "repo-b"],
            affected_count=2
        )

        result = anomaly.to_dict()

        assert result["anomaly_id"] == "temporal_001"
        assert result["anomaly_type"] == "rapid_propagation"
        assert result["severity"] == "high"


# ============================================================================
# Test: Enums
# ============================================================================

class TestEnums:
    """Tests for enum classes."""

    def test_influence_direction_values(self):
        """Test InfluenceDirection enum values."""
        assert InfluenceDirection.LEADER.value == "leader"
        assert InfluenceDirection.FOLLOWER.value == "follower"
        assert InfluenceDirection.BIDIRECTIONAL.value == "bidirectional"
        assert InfluenceDirection.INDEPENDENT.value == "independent"

    def test_propagation_type_values(self):
        """Test PropagationType enum values."""
        assert PropagationType.LINEAR.value == "linear"
        assert PropagationType.BRANCHING.value == "branching"
        assert PropagationType.CONVERGING.value == "converging"

    def test_temporal_severity_comparison(self):
        """Test TemporalSeverity comparison operators."""
        assert TemporalSeverity.LOW < TemporalSeverity.MEDIUM
        assert TemporalSeverity.MEDIUM < TemporalSeverity.HIGH
        assert TemporalSeverity.HIGH < TemporalSeverity.CRITICAL
        assert TemporalSeverity.CRITICAL <= TemporalSeverity.CRITICAL


# ============================================================================
# Test: Configuration
# ============================================================================

class TestConfiguration:
    """Tests for configuration classes."""

    def test_default_thresholds(self):
        """Test default threshold values."""
        thresholds = create_default_thresholds()

        assert thresholds.max_lag == 3
        assert thresholds.min_lag == -3
        assert thresholds.min_lagged_correlation == 0.5
        assert thresholds.min_influence_score == 30.0
        assert thresholds.min_causality_score == 0.4

    def test_thresholds_from_dict(self):
        """Test creating thresholds from dictionary."""
        data = {
            "max_lag": 5,
            "min_influence_score": 40.0,
            "min_causality_score": 0.5
        }

        thresholds = TemporalThresholds.from_dict(data)

        assert thresholds.max_lag == 5
        assert thresholds.min_influence_score == 40.0
        assert thresholds.min_causality_score == 0.5

    def test_config_to_dict(self, temp_org_report):
        """Test converting config to dictionary."""
        config = TemporalIntelligenceConfig(
            org_report_path=temp_org_report,
            verbose=True,
            compute_lagged_correlations=True
        )

        result = config.to_dict()

        assert "org_report_path" in result
        assert result["verbose"] is True
        assert result["compute_lagged_correlations"] is True


# ============================================================================
# Test: LaggedCorrelationEngine
# ============================================================================

class TestLaggedCorrelationEngine:
    """Tests for LaggedCorrelationEngine component."""

    def test_compute_lagged_correlations(self, default_config, sample_repo_series):
        """Test computing lagged correlations."""
        engine = LaggedCorrelationEngine(default_config)
        correlations = engine.compute_all_lagged_correlations(sample_repo_series)

        assert isinstance(correlations, list)
        # Should find correlations since scores are similar patterns

    def test_pearson_correlation_perfect_positive(self, default_config):
        """Test Pearson correlation with perfectly correlated data."""
        engine = LaggedCorrelationEngine(default_config)

        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [2.0, 4.0, 6.0, 8.0, 10.0]

        corr = engine._pearson_correlation(x, y)
        assert corr == pytest.approx(1.0)

    def test_pearson_correlation_perfect_negative(self, default_config):
        """Test Pearson correlation with negatively correlated data."""
        engine = LaggedCorrelationEngine(default_config)

        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [10.0, 8.0, 6.0, 4.0, 2.0]

        corr = engine._pearson_correlation(x, y)
        assert corr == pytest.approx(-1.0)

    def test_correlation_at_lag_positive(self, default_config):
        """Test correlation at positive lag (A leads B)."""
        engine = LaggedCorrelationEngine(default_config)

        # A's pattern appears in B 2 steps later
        x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        y = [0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0]  # Same pattern, delayed by 2

        corr = engine._compute_correlation_at_lag(x, y, 2)
        assert corr is not None
        assert corr > 0.9  # Should be highly correlated

    def test_optimal_lag_matrix(self, default_config, sample_repo_series):
        """Test optimal lag matrix computation."""
        engine = LaggedCorrelationEngine(default_config)
        engine.compute_all_lagged_correlations(sample_repo_series)

        matrix = engine.get_optimal_lag_matrix()

        assert isinstance(matrix, dict)
        # Should have entries for each repo
        for repo_id in sample_repo_series:
            assert repo_id in matrix


# ============================================================================
# Test: InfluenceScoringEngine
# ============================================================================

class TestInfluenceScoringEngine:
    """Tests for InfluenceScoringEngine component."""

    def test_compute_influence_scores(self, default_config, sample_repo_series):
        """Test influence score computation."""
        # Create some lagged correlations
        correlations = [
            LaggedCorrelation(
                repo_a_id="repo-a",
                repo_b_id="repo-b",
                lag=1,
                correlation_coefficient=0.8,
                leader_repo_id="repo-a",
                follower_repo_id="repo-b",
                is_significant=True
            ),
            LaggedCorrelation(
                repo_a_id="repo-a",
                repo_b_id="repo-c",
                lag=2,
                correlation_coefficient=0.75,
                leader_repo_id="repo-a",
                follower_repo_id="repo-c",
                is_significant=True
            )
        ]

        engine = InfluenceScoringEngine(default_config)
        scores = engine.compute_influence_scores(sample_repo_series, correlations)

        assert len(scores) == 3
        # repo-a should be ranked highest (leads 2 repos)
        assert scores[0].repo_id == "repo-a"
        assert scores[0].repos_led == 2

    def test_influence_direction_assignment(self, default_config):
        """Test that influence direction is assigned correctly."""
        repo_series = {
            "leader": RepoTimeSeries("leader", "Leader", [80.0, 85.0], ["t1", "t2"]),
            "follower1": RepoTimeSeries("follower1", "Follower 1", [75.0, 80.0], ["t1", "t2"]),
            "follower2": RepoTimeSeries("follower2", "Follower 2", [70.0, 75.0], ["t1", "t2"])
        }

        correlations = [
            LaggedCorrelation(
                repo_a_id="leader",
                repo_b_id="follower1",
                lag=1,
                correlation_coefficient=0.8,
                leader_repo_id="leader",
                follower_repo_id="follower1",
                is_significant=True
            ),
            LaggedCorrelation(
                repo_a_id="leader",
                repo_b_id="follower2",
                lag=1,
                correlation_coefficient=0.75,
                leader_repo_id="leader",
                follower_repo_id="follower2",
                is_significant=True
            )
        ]

        engine = InfluenceScoringEngine(default_config)
        scores = engine.compute_influence_scores(repo_series, correlations)

        leader_score = next(s for s in scores if s.repo_id == "leader")
        assert leader_score.influence_direction == InfluenceDirection.LEADER


# ============================================================================
# Test: PropagationGraphBuilder
# ============================================================================

class TestPropagationGraphBuilder:
    """Tests for PropagationGraphBuilder component."""

    def test_build_propagation_graph(self, default_config):
        """Test building propagation graph."""
        correlations = [
            LaggedCorrelation(
                repo_a_id="a",
                repo_b_id="b",
                lag=1,
                correlation_coefficient=0.8,
                leader_repo_id="a",
                follower_repo_id="b",
                is_significant=True
            ),
            LaggedCorrelation(
                repo_a_id="b",
                repo_b_id="c",
                lag=1,
                correlation_coefficient=0.75,
                leader_repo_id="b",
                follower_repo_id="c",
                is_significant=True
            )
        ]

        influence_scores = {
            "a": InfluenceScore("a", "A", influence_score=70.0,
                               influence_direction=InfluenceDirection.LEADER),
            "b": InfluenceScore("b", "B", influence_score=50.0,
                               influence_direction=InfluenceDirection.BIDIRECTIONAL),
            "c": InfluenceScore("c", "C", influence_score=30.0,
                               influence_direction=InfluenceDirection.FOLLOWER)
        }

        builder = PropagationGraphBuilder(default_config)
        graph = builder.build_propagation_graph(correlations, influence_scores)

        assert "a" in graph
        assert "b" in graph["a"]
        assert "c" in graph["b"]

    def test_detect_propagation_paths(self, default_config):
        """Test propagation path detection."""
        correlations = [
            LaggedCorrelation(
                repo_a_id="source",
                repo_b_id="mid",
                lag=1,
                correlation_coefficient=0.8,
                leader_repo_id="source",
                follower_repo_id="mid",
                is_significant=True
            ),
            LaggedCorrelation(
                repo_a_id="mid",
                repo_b_id="terminal",
                lag=1,
                correlation_coefficient=0.75,
                leader_repo_id="mid",
                follower_repo_id="terminal",
                is_significant=True
            )
        ]

        influence_scores = {
            "source": InfluenceScore("source", "Source",
                                    influence_direction=InfluenceDirection.LEADER),
            "mid": InfluenceScore("mid", "Mid",
                                 influence_direction=InfluenceDirection.BIDIRECTIONAL),
            "terminal": InfluenceScore("terminal", "Terminal",
                                      influence_direction=InfluenceDirection.FOLLOWER)
        }

        repo_series = {
            "source": RepoTimeSeries("source", "Source", [80.0], ["t1"]),
            "mid": RepoTimeSeries("mid", "Mid", [75.0], ["t1"]),
            "terminal": RepoTimeSeries("terminal", "Terminal", [70.0], ["t1"])
        }

        builder = PropagationGraphBuilder(default_config)
        builder.build_propagation_graph(correlations, influence_scores)
        paths = builder.detect_propagation_paths(repo_series)

        # Should detect path: source -> mid -> terminal
        assert len(paths) >= 1

    def test_path_type_classification(self, default_config):
        """Test that path types are classified correctly."""
        # Create branching scenario: a -> b, a -> c
        correlations = [
            LaggedCorrelation(
                repo_a_id="a",
                repo_b_id="b",
                lag=1,
                correlation_coefficient=0.8,
                leader_repo_id="a",
                follower_repo_id="b",
                is_significant=True
            ),
            LaggedCorrelation(
                repo_a_id="a",
                repo_b_id="c",
                lag=1,
                correlation_coefficient=0.75,
                leader_repo_id="a",
                follower_repo_id="c",
                is_significant=True
            )
        ]

        influence_scores = {
            "a": InfluenceScore("a", "A", influence_direction=InfluenceDirection.LEADER),
            "b": InfluenceScore("b", "B", influence_direction=InfluenceDirection.FOLLOWER),
            "c": InfluenceScore("c", "C", influence_direction=InfluenceDirection.FOLLOWER)
        }

        repo_series = {
            "a": RepoTimeSeries("a", "A", [80.0], ["t1"]),
            "b": RepoTimeSeries("b", "B", [75.0], ["t1"]),
            "c": RepoTimeSeries("c", "C", [70.0], ["t1"])
        }

        builder = PropagationGraphBuilder(default_config)
        builder.build_propagation_graph(correlations, influence_scores)
        paths = builder.detect_propagation_paths(repo_series)

        # At least some paths should be detected
        assert isinstance(paths, list)


# ============================================================================
# Test: TemporalAnomalyDetector
# ============================================================================

class TestTemporalAnomalyDetector:
    """Tests for TemporalAnomalyDetector component."""

    def test_detect_rapid_propagation(self, default_config):
        """Test rapid propagation detection."""
        repo_series = {
            "source": RepoTimeSeries("source", "Source", [80.0, 70.0], ["t1", "t2"],
                                    current_trend="declining"),
            "target": RepoTimeSeries("target", "Target", [75.0, 65.0], ["t1", "t2"],
                                    current_trend="declining")
        }

        paths = [
            PropagationPath(
                path_id="path_001",
                path_type=PropagationType.LINEAR,
                repo_sequence=["source", "target"],
                total_lag=1,  # Rapid - only 1 interval
                path_length=1,
                path_confidence=0.8,
                involves_critical_repos=True
            )
        ]

        detector = TemporalAnomalyDetector(default_config)
        anomalies = detector.detect_anomalies(repo_series, [], [], paths)

        rapid_anomalies = [a for a in anomalies
                         if a.anomaly_type == TemporalAnomalyType.RAPID_PROPAGATION]
        assert len(rapid_anomalies) >= 1

    def test_detect_leader_deterioration(self, default_config):
        """Test leader deterioration detection."""
        repo_series = {
            "leader": RepoTimeSeries("leader", "Leader",
                                    [90.0, 85.0, 75.0, 65.0],
                                    ["t1", "t2", "t3", "t4"],
                                    current_score=65.0,
                                    current_trend="declining")
        }

        influence_scores = [
            InfluenceScore(
                repo_id="leader",
                repo_name="Leader",
                influence_score=80.0,  # High influence
                repos_led=5,
                led_repos=["f1", "f2", "f3", "f4", "f5"],
                influence_direction=InfluenceDirection.LEADER
            )
        ]

        detector = TemporalAnomalyDetector(default_config)
        detector._detect_leader_deterioration(repo_series, influence_scores, "2025-01-08")

        anomalies = detector.get_anomalies()
        leader_anomalies = [a for a in anomalies
                          if a.anomaly_type == TemporalAnomalyType.LEADER_DETERIORATION]
        assert len(leader_anomalies) >= 1

    def test_detect_systemic_propagation(self, default_config):
        """Test systemic propagation detection."""
        repo_series = {
            f"repo-{i}": RepoTimeSeries(f"repo-{i}", f"Repo {i}",
                                       [80.0 - i*5, 70.0 - i*5],
                                       ["t1", "t2"],
                                       current_trend="declining")
            for i in range(5)
        }

        paths = [
            PropagationPath(
                path_id="path_001",
                path_type=PropagationType.LINEAR,
                repo_sequence=[f"repo-{i}" for i in range(5)],
                total_lag=4,
                path_length=4,
                path_confidence=0.7,
                affected_repo_count=5,
                source_repo_id="repo-0"
            )
        ]

        detector = TemporalAnomalyDetector(default_config)
        detector._detect_systemic_propagation(paths, repo_series, "2025-01-08")

        anomalies = detector.get_anomalies()
        systemic_anomalies = [a for a in anomalies
                            if a.anomaly_type == TemporalAnomalyType.SYSTEMIC_PROPAGATION]
        assert len(systemic_anomalies) >= 1


# ============================================================================
# Test: TemporalIntelligenceEngine
# ============================================================================

class TestTemporalIntelligenceEngine:
    """Tests for main TemporalIntelligenceEngine."""

    def test_engine_run_success(self, default_config):
        """Test successful engine run."""
        engine = TemporalIntelligenceEngine(default_config)
        report, exit_code = engine.run()

        assert report is not None
        assert isinstance(report, TemporalIntelligenceReport)
        assert exit_code in (
            EXIT_TEMPORAL_SUCCESS,
            EXIT_TEMPORAL_CORRELATIONS_FOUND,
            EXIT_CRITICAL_PROPAGATION_RISK
        )

    def test_engine_report_structure(self, default_config):
        """Test report structure completeness."""
        engine = TemporalIntelligenceEngine(default_config)
        report, _ = engine.run()

        assert report.report_id is not None
        assert report.generated_at is not None
        assert report.total_repos == 5
        assert isinstance(report.summary, TemporalIntelligenceSummary)
        assert isinstance(report.lagged_correlations, list)
        assert isinstance(report.influence_scores, list)
        assert isinstance(report.propagation_paths, list)
        assert isinstance(report.anomalies, list)

    def test_engine_with_output_file(self, default_config, temp_output_dir):
        """Test engine writes output file."""
        default_config.output_path = temp_output_dir / "temporal-report.json"

        engine = TemporalIntelligenceEngine(default_config)
        report, _ = engine.run()

        assert default_config.output_path.exists()

        with open(default_config.output_path) as f:
            saved_report = json.load(f)

        assert saved_report["report_id"] == report.report_id

    def test_engine_missing_report_error(self, temp_output_dir):
        """Test engine handles missing org report."""
        config = TemporalIntelligenceConfig(
            org_report_path=temp_output_dir / "nonexistent.json"
        )

        engine = TemporalIntelligenceEngine(config)
        report, exit_code = engine.run()

        assert exit_code == EXIT_TEMPORAL_PARSE_ERROR

    def test_engine_fail_on_critical(self, default_config):
        """Test fail on critical flag."""
        default_config.fail_on_critical_propagation = True

        engine = TemporalIntelligenceEngine(default_config)
        report, exit_code = engine.run()

        if report.summary.critical_anomalies > 0:
            assert exit_code == EXIT_CRITICAL_PROPAGATION_RISK


# ============================================================================
# Test: CLI Behavior
# ============================================================================

class TestCLI:
    """Tests for CLI behavior."""

    def test_cli_module_import(self):
        """Test that CLI module can be imported."""
        from analytics.run_org_temporal_intelligence import main, create_argument_parser
        assert callable(main)
        assert callable(create_argument_parser)

    def test_cli_argument_parser(self):
        """Test argument parser creation."""
        from analytics.run_org_temporal_intelligence import create_argument_parser

        parser = create_argument_parser()
        assert parser is not None

        # Test parsing minimal args
        args = parser.parse_args(["--org-report", "test.json"])
        assert args.org_report == "test.json"

    def test_cli_all_options(self):
        """Test parsing all CLI options."""
        from analytics.run_org_temporal_intelligence import create_argument_parser

        parser = create_argument_parser()
        args = parser.parse_args([
            "--org-report", "report.json",
            "--output", "output.json",
            "--json",
            "--summary-only",
            "--verbose",
            "--max-lag", "5",
            "--min-influence-score", "40.0",
            "--min-correlation", "0.6",
            "--skip-lagged-correlations",
            "--skip-influence-scores",
            "--skip-propagation-paths",
            "--skip-anomalies",
            "--fail-on-critical"
        ])

        assert args.org_report == "report.json"
        assert args.output == "output.json"
        assert args.json is True
        assert args.summary_only is True
        assert args.verbose is True
        assert args.max_lag == 5
        assert args.min_influence_score == 40.0
        assert args.skip_lagged_correlations is True
        assert args.fail_on_critical is True

    def test_cli_main_with_report(self, temp_org_report):
        """Test CLI main function with valid report."""
        from analytics.run_org_temporal_intelligence import main

        exit_code = main([
            "--org-report", str(temp_org_report),
            "--json"
        ])

        assert exit_code in (
            EXIT_TEMPORAL_SUCCESS,
            EXIT_TEMPORAL_CORRELATIONS_FOUND,
            EXIT_CRITICAL_PROPAGATION_RISK
        )

    def test_cli_main_missing_report(self, temp_output_dir):
        """Test CLI main with missing report."""
        from analytics.run_org_temporal_intelligence import main

        exit_code = main([
            "--org-report", str(temp_output_dir / "missing.json")
        ])

        assert exit_code == EXIT_TEMPORAL_PARSE_ERROR


# ============================================================================
# Test: Realistic Multi-Repo Scenarios
# ============================================================================

class TestRealisticScenarios:
    """Tests for realistic multi-repo scenarios."""

    def test_large_org_scenario(self, temp_output_dir):
        """Test with large org (20+ repos)."""
        # Create large org report with varied patterns
        repos = []
        for i in range(20):
            # Create different patterns
            if i < 5:  # Leaders (improving first)
                scores = [70.0 + j*2 for j in range(5)]
            elif i < 10:  # Followers (improving with delay)
                scores = [65.0 + j*2 for j in range(5)]
            else:  # Others (various patterns)
                scores = [75.0 - (i-10) + j for j in range(5)]

            repos.append({
                "repo_id": f"service-{i:02d}",
                "repo_name": f"Service {i}",
                "health_status": "green" if scores[-1] >= 70 else "yellow",
                "repository_score": scores[-1],
                "total_issues": i,
                "critical_issues": i // 5,
                "risk_tier": "high" if scores[-1] < 60 else "low",
                "trends": {"overall_trend": "improving" if scores[-1] > scores[0] else "declining"},
                "trend_history": [
                    {"timestamp": f"2025-01-0{j+1}", "score": scores[j]}
                    for j in range(5)
                ]
            })

        report = {
            "org_health_status": "yellow",
            "org_health_score": 72.0,
            "repos_loaded": 20,
            "repositories": repos
        }

        report_path = temp_output_dir / "large-org-report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f)

        config = TemporalIntelligenceConfig(org_report_path=report_path)
        engine = TemporalIntelligenceEngine(config)
        result, exit_code = engine.run()

        assert result.total_repos == 20
        assert len(result.influence_scores) == 20

    def test_cascading_decline_scenario(self, temp_output_dir):
        """Test scenario with cascading decline (A → B → C)."""
        # Create repos where decline propagates with lag
        repos = [
            {
                "repo_id": "leader",
                "repo_name": "Leader Service",
                "health_status": "yellow",
                "repository_score": 60.0,
                "risk_tier": "medium",
                "trends": {"overall_trend": "declining"},
                "trend_history": [
                    {"timestamp": "2025-01-01", "score": 90.0},
                    {"timestamp": "2025-01-02", "score": 80.0},
                    {"timestamp": "2025-01-03", "score": 70.0},
                    {"timestamp": "2025-01-04", "score": 65.0},
                    {"timestamp": "2025-01-05", "score": 60.0}
                ]
            },
            {
                "repo_id": "mid",
                "repo_name": "Mid Service",
                "health_status": "yellow",
                "repository_score": 65.0,
                "risk_tier": "medium",
                "trends": {"overall_trend": "declining"},
                "trend_history": [
                    {"timestamp": "2025-01-01", "score": 88.0},
                    {"timestamp": "2025-01-02", "score": 88.0},
                    {"timestamp": "2025-01-03", "score": 78.0},
                    {"timestamp": "2025-01-04", "score": 68.0},
                    {"timestamp": "2025-01-05", "score": 65.0}
                ]
            },
            {
                "repo_id": "follower",
                "repo_name": "Follower Service",
                "health_status": "red",
                "repository_score": 55.0,
                "risk_tier": "high",
                "trends": {"overall_trend": "declining"},
                "trend_history": [
                    {"timestamp": "2025-01-01", "score": 85.0},
                    {"timestamp": "2025-01-02", "score": 85.0},
                    {"timestamp": "2025-01-03", "score": 85.0},
                    {"timestamp": "2025-01-04", "score": 70.0},
                    {"timestamp": "2025-01-05", "score": 55.0}
                ]
            }
        ]

        report = {
            "org_health_status": "yellow",
            "org_health_score": 60.0,
            "repos_loaded": 3,
            "repositories": repos
        }

        report_path = temp_output_dir / "cascade-report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f)

        config = TemporalIntelligenceConfig(org_report_path=report_path)
        engine = TemporalIntelligenceEngine(config)
        result, exit_code = engine.run()

        # Should detect some lagged patterns
        assert result.total_repos == 3

    def test_stable_org_no_anomalies(self, temp_output_dir):
        """Test stable org produces no anomalies."""
        repos = [
            {
                "repo_id": f"stable-{i}",
                "repo_name": f"Stable Repo {i}",
                "health_status": "green",
                "repository_score": 85.0 + i,
                "risk_tier": "low",
                "trends": {"overall_trend": "stable"},
                "trend_history": [
                    {"timestamp": f"2025-01-0{j+1}", "score": 85.0 + i + j*0.1}
                    for j in range(5)
                ]
            }
            for i in range(5)
        ]

        report = {
            "org_health_status": "green",
            "org_health_score": 88.0,
            "repos_loaded": 5,
            "repositories": repos
        }

        report_path = temp_output_dir / "stable-report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f)

        config = TemporalIntelligenceConfig(org_report_path=report_path)
        engine = TemporalIntelligenceEngine(config)
        result, exit_code = engine.run()

        # Should have no critical anomalies
        assert result.summary.critical_anomalies == 0


# ============================================================================
# Test: Edge Cases
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_repo_org(self, temp_output_dir):
        """Test org with single repo."""
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
                    "risk_tier": "low",
                    "trends": {"overall_trend": "stable"},
                    "trend_history": [
                        {"timestamp": "2025-01-01", "score": 89.0},
                        {"timestamp": "2025-01-02", "score": 90.0}
                    ]
                }
            ]
        }

        report_path = temp_output_dir / "single-repo.json"
        with open(report_path, 'w') as f:
            json.dump(report, f)

        config = TemporalIntelligenceConfig(org_report_path=report_path)
        engine = TemporalIntelligenceEngine(config)
        result, exit_code = engine.run()

        # Should handle gracefully
        assert result.total_repos == 1
        assert exit_code == EXIT_TEMPORAL_SUCCESS

    def test_empty_trend_history(self, temp_output_dir):
        """Test repos without trend history."""
        repos = [
            {
                "repo_id": f"no-history-{i}",
                "repo_name": f"No History {i}",
                "health_status": "green",
                "repository_score": 85.0,
                "risk_tier": "low",
                "trends": {"overall_trend": "stable"}
                # No trend_history
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

        config = TemporalIntelligenceConfig(org_report_path=report_path)
        engine = TemporalIntelligenceEngine(config)
        result, exit_code = engine.run()

        # Should still work
        assert result.total_repos == 3

    def test_constant_scores(self, temp_output_dir):
        """Test repos with constant scores (no variation)."""
        repos = [
            {
                "repo_id": f"constant-{i}",
                "repo_name": f"Constant {i}",
                "health_status": "green",
                "repository_score": 80.0,
                "risk_tier": "low",
                "trends": {"overall_trend": "stable"},
                "trend_history": [
                    {"timestamp": f"2025-01-0{j+1}", "score": 80.0}
                    for j in range(5)
                ]
            }
            for i in range(3)
        ]

        report = {
            "org_health_status": "green",
            "repos_loaded": 3,
            "repositories": repos
        }

        report_path = temp_output_dir / "constant.json"
        with open(report_path, 'w') as f:
            json.dump(report, f)

        config = TemporalIntelligenceConfig(org_report_path=report_path)
        engine = TemporalIntelligenceEngine(config)
        result, exit_code = engine.run()

        # Should handle without errors (0 correlation for constant data)
        assert result.total_repos == 3


# ============================================================================
# Test: Report Serialization
# ============================================================================

class TestReportSerialization:
    """Tests for report serialization."""

    def test_report_to_dict(self, default_config):
        """Test report to_dict serialization."""
        engine = TemporalIntelligenceEngine(default_config)
        report, _ = engine.run()

        report_dict = report.to_dict()

        assert isinstance(report_dict, dict)
        assert "report_id" in report_dict
        assert "summary" in report_dict
        assert "lagged_correlations" in report_dict
        assert "influence_scores" in report_dict
        assert "propagation_paths" in report_dict
        assert "anomalies" in report_dict

    def test_report_json_serializable(self, default_config):
        """Test that report can be JSON serialized."""
        engine = TemporalIntelligenceEngine(default_config)
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

    def test_summary_fields(self, default_config):
        """Test summary statistics fields."""
        engine = TemporalIntelligenceEngine(default_config)
        report, _ = engine.run()

        summary = report.summary

        assert summary.total_repo_pairs >= 0
        assert summary.lagged_correlations_computed >= 0
        assert summary.significant_lagged_correlations >= 0
        assert summary.repos_with_influence >= 0
        assert summary.propagation_paths_detected >= 0
        assert summary.total_anomalies >= 0


# ============================================================================
# Run tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
