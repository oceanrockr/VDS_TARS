"""
Integration Tests for Repository Health Trend Analyzer

Phase 14.7 Task 10 - Comprehensive test suite for the trend analysis engine.

Test Coverage:
- HealthHistoryStore operations (loading, saving, indexing)
- StatisticsCalculator methods
- TrendAnalyzer computations
- Anomaly detection
- Early warning generation
- Predictive analysis
- CLI functionality
- Error handling

Target: 30+ tests, 95%+ branch coverage
"""

import json
import os
import shutil
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any
from unittest.mock import patch, MagicMock

import pytest

from analytics.trend_analyzer import (
    # Exit codes
    EXIT_TREND_SUCCESS,
    EXIT_INSUFFICIENT_HISTORY,
    EXIT_INVALID_SNAPSHOT,
    EXIT_COMPUTATION_ERROR,
    EXIT_PREDICTION_ERROR,
    EXIT_HISTORY_STORE_ERROR,
    EXIT_GENERAL_TREND_ERROR,
    # Exceptions
    TrendAnalysisError,
    InsufficientHistoryError,
    InvalidSnapshotError,
    ComputationError,
    PredictionError,
    HistoryStoreError,
    # Enums
    TrendDirection,
    WarningLevel,
    AnomalyType,
    # Data classes
    TrendConfig,
    SnapshotMetadata,
    DashboardSnapshot,
    RegressionResult,
    MovingAverageResult,
    VolatilityResult,
    Anomaly,
    EarlyWarning,
    PredictionResult,
    VersionTrend,
    IssueTrend,
    ScoreTrend,
    TrendGraphData,
    TrendReport,
    # Classes
    HealthHistoryStore,
    StatisticsCalculator,
    TrendAnalyzer,
    TrendChartGenerator,
    TrendEngine,
    # Utility functions
    add_snapshot_to_history,
    get_trend_summary,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    tmp = tempfile.mkdtemp()
    yield Path(tmp)
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture
def sample_dashboard_data() -> Dict[str, Any]:
    """Create sample dashboard data."""
    return {
        "repository_score": 85.0,
        "overall_health": "green",
        "total_issues": 5,
        "critical_issues": 0,
        "error_issues": 1,
        "warning_issues": 4,
        "scan_timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "versions_health": [
            {"version": "1.0.0", "health_status": "green"},
            {"version": "0.9.0", "health_status": "yellow"},
        ]
    }


@pytest.fixture
def sample_dashboard_path(temp_dir, sample_dashboard_data) -> Path:
    """Create a sample dashboard JSON file."""
    dashboard_path = temp_dir / "dashboard.json"
    with open(dashboard_path, 'w') as f:
        json.dump(sample_dashboard_data, f)
    return dashboard_path


@pytest.fixture
def history_dir(temp_dir) -> Path:
    """Create a history directory."""
    hist_dir = temp_dir / "history"
    hist_dir.mkdir()
    return hist_dir


@pytest.fixture
def populated_history(history_dir) -> Path:
    """Create a history store with multiple snapshots."""
    store = HealthHistoryStore(history_dir)
    store.initialize()

    # Create 5 snapshots with declining scores
    base_time = datetime.utcnow() - timedelta(days=5)
    scores = [95.0, 90.0, 85.0, 80.0, 75.0]

    for i, score in enumerate(scores):
        snapshot_time = base_time + timedelta(days=i)
        dashboard_data = {
            "repository_score": score,
            "overall_health": "green" if score >= 80 else "yellow",
            "total_issues": 10 - i,
            "critical_issues": i if i < 3 else 2,
            "scan_timestamp": snapshot_time.isoformat(),
            "version": f"1.0.{i}",
            "versions_health": [
                {"version": f"1.0.{i}", "health_status": "green" if score >= 80 else "yellow"}
            ]
        }

        dashboard_path = history_dir / f"temp_dashboard_{i}.json"
        with open(dashboard_path, 'w') as f:
            json.dump(dashboard_data, f)

        store.add_snapshot(dashboard_path, f"1.0.{i}", snapshot_time)
        dashboard_path.unlink()  # Clean up temp file

    return history_dir


@pytest.fixture
def trend_config(history_dir) -> TrendConfig:
    """Create a basic trend configuration."""
    return TrendConfig(
        history_dir=history_dir,
        min_snapshots=3,
        max_snapshots=100,
        zscore_threshold=2.0,
        score_drop_threshold=15.0,
        issue_spike_threshold=5,
        prediction_horizon=3,
        confidence_level=0.95,
        warning_score_threshold=60.0,
        critical_score_threshold=40.0,
        volatility_threshold=10.0,
        verbose=False
    )


# ============================================================================
# StatisticsCalculator Tests
# ============================================================================

class TestStatisticsCalculator:
    """Tests for StatisticsCalculator."""

    def test_mean_basic(self):
        """Test mean calculation."""
        assert StatisticsCalculator.mean([1, 2, 3, 4, 5]) == 3.0

    def test_mean_empty(self):
        """Test mean with empty list."""
        assert StatisticsCalculator.mean([]) == 0.0

    def test_mean_single(self):
        """Test mean with single value."""
        assert StatisticsCalculator.mean([42.0]) == 42.0

    def test_standard_deviation_basic(self):
        """Test standard deviation calculation."""
        values = [2, 4, 4, 4, 5, 5, 7, 9]
        std = StatisticsCalculator.standard_deviation(values)
        assert 1.9 < std < 2.1  # Approximately 2

    def test_standard_deviation_empty(self):
        """Test std dev with insufficient values."""
        assert StatisticsCalculator.standard_deviation([]) == 0.0
        assert StatisticsCalculator.standard_deviation([5.0]) == 0.0

    def test_variance_basic(self):
        """Test variance calculation."""
        values = [2, 4, 4, 4, 5, 5, 7, 9]
        var = StatisticsCalculator.variance(values)
        assert 3.5 < var < 4.5

    def test_linear_regression_basic(self):
        """Test linear regression."""
        x = [0, 1, 2, 3, 4]
        y = [0, 2, 4, 6, 8]  # Perfect line y = 2x
        result = StatisticsCalculator.linear_regression(x, y)

        assert abs(result.slope - 2.0) < 0.01
        assert abs(result.intercept) < 0.01
        assert result.r_squared > 0.99

    def test_linear_regression_with_noise(self):
        """Test linear regression with noisy data."""
        x = [0, 1, 2, 3, 4, 5]
        y = [1.1, 2.9, 5.2, 6.8, 9.1, 10.9]  # Roughly y = 2x + 1
        result = StatisticsCalculator.linear_regression(x, y)

        assert 1.8 < result.slope < 2.2
        assert 0.8 < result.intercept < 1.5
        assert result.r_squared > 0.95

    def test_linear_regression_insufficient_data(self):
        """Test linear regression with insufficient data."""
        result = StatisticsCalculator.linear_regression([1], [1])
        assert result.slope == 0.0
        assert result.p_value == 1.0

    def test_moving_average_basic(self):
        """Test moving average calculation."""
        values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        ma = StatisticsCalculator.moving_average(values, 3)

        # First two values should be None
        assert ma[0] is None
        assert ma[1] is None

        # Check remaining values
        assert ma[2] == 2.0  # (1+2+3)/3
        assert ma[3] == 3.0  # (2+3+4)/3
        assert ma[9] == 9.0  # (8+9+10)/3

    def test_moving_average_window_larger_than_values(self):
        """Test MA when window is larger than values."""
        values = [1, 2, 3]
        ma = StatisticsCalculator.moving_average(values, 5)
        assert all(v is None for v in ma)

    def test_zscore_basic(self):
        """Test z-score calculation."""
        zscore = StatisticsCalculator.zscore(100, 80, 10)
        assert zscore == 2.0

    def test_zscore_zero_std(self):
        """Test z-score with zero standard deviation."""
        zscore = StatisticsCalculator.zscore(100, 80, 0)
        assert zscore == 0.0

    def test_confidence_interval(self):
        """Test confidence interval calculation."""
        ci = StatisticsCalculator.confidence_interval(100, 5, 0.95, 30)
        lower, upper = ci
        assert lower < 100 < upper
        assert 89 < lower < 95  # Rough bounds
        assert 105 < upper < 111


# ============================================================================
# HealthHistoryStore Tests
# ============================================================================

class TestHealthHistoryStore:
    """Tests for HealthHistoryStore."""

    def test_initialize_creates_directory(self, temp_dir):
        """Test that initialize creates necessary directories."""
        history_dir = temp_dir / "new_history"
        store = HealthHistoryStore(history_dir)
        store.initialize()

        assert history_dir.exists()
        assert (history_dir / "snapshots").exists()
        assert (history_dir / "index.json").exists()

    def test_add_snapshot_basic(self, history_dir, sample_dashboard_path):
        """Test adding a snapshot."""
        store = HealthHistoryStore(history_dir)
        store.initialize()

        metadata = store.add_snapshot(sample_dashboard_path, "1.0.0")

        assert metadata.snapshot_id is not None
        assert metadata.version == "1.0.0"
        assert metadata.repository_score == 85.0
        assert metadata.overall_health == "green"

    def test_add_snapshot_extracts_version(self, history_dir, temp_dir):
        """Test that version is extracted from dashboard."""
        dashboard = temp_dir / "dashboard.json"
        with open(dashboard, 'w') as f:
            json.dump({
                "repository_score": 90.0,
                "overall_health": "green",
                "total_issues": 0,
                "critical_issues": 0,
                "version": "2.0.0",
            }, f)

        store = HealthHistoryStore(history_dir)
        store.initialize()

        metadata = store.add_snapshot(dashboard)
        assert metadata.version == "2.0.0"

    def test_get_snapshot_count(self, populated_history):
        """Test getting snapshot count."""
        store = HealthHistoryStore(populated_history)
        store.initialize()

        assert store.get_snapshot_count() == 5

    def test_get_all_snapshots_sorted(self, populated_history):
        """Test that snapshots are sorted by timestamp."""
        store = HealthHistoryStore(populated_history)
        store.initialize()

        snapshots = store.get_all_snapshots()
        timestamps = [s.timestamp for s in snapshots]

        assert timestamps == sorted(timestamps)

    def test_get_last_n_snapshots(self, populated_history):
        """Test getting last N snapshots."""
        store = HealthHistoryStore(populated_history)
        store.initialize()

        snapshots = store.get_last_n_snapshots(3)
        assert len(snapshots) == 3

        # Should be most recent 3
        scores = [s.metadata.repository_score for s in snapshots]
        assert scores == [85.0, 80.0, 75.0]

    def test_get_last_n_more_than_available(self, populated_history):
        """Test requesting more snapshots than available."""
        store = HealthHistoryStore(populated_history)
        store.initialize()

        snapshots = store.get_last_n_snapshots(10)
        assert len(snapshots) == 5

    def test_load_snapshot(self, populated_history):
        """Test loading a specific snapshot."""
        store = HealthHistoryStore(populated_history)
        store.initialize()

        all_snapshots = store.get_all_snapshots()
        snapshot_id = all_snapshots[0].snapshot_id

        snapshot = store.load_snapshot(snapshot_id)
        assert snapshot.metadata.snapshot_id == snapshot_id
        assert "repository_score" in snapshot.data

    def test_load_snapshot_not_found(self, history_dir):
        """Test loading non-existent snapshot."""
        store = HealthHistoryStore(history_dir)
        store.initialize()

        with pytest.raises(InvalidSnapshotError):
            store.load_snapshot("nonexistent_id")

    def test_validate_index(self, populated_history):
        """Test index validation."""
        store = HealthHistoryStore(populated_history)
        store.initialize()

        is_valid, issues = store.validate_index()
        assert is_valid
        assert len(issues) == 0

    def test_rebuild_index(self, populated_history):
        """Test rebuilding index."""
        store = HealthHistoryStore(populated_history)
        store.initialize()

        # Delete index
        (populated_history / "index.json").unlink()

        # Rebuild
        count = store.rebuild_index()
        assert count == 5

    def test_add_invalid_snapshot(self, history_dir, temp_dir):
        """Test adding an invalid snapshot."""
        # Create malformed dashboard
        invalid_dashboard = temp_dir / "invalid.json"
        with open(invalid_dashboard, 'w') as f:
            json.dump({"not_a_dashboard": True}, f)

        store = HealthHistoryStore(history_dir)
        store.initialize()

        with pytest.raises(InvalidSnapshotError):
            store.add_snapshot(invalid_dashboard)


# ============================================================================
# TrendAnalyzer Tests
# ============================================================================

class TestTrendAnalyzer:
    """Tests for TrendAnalyzer."""

    def test_load_snapshots(self, trend_config, populated_history):
        """Test loading snapshots."""
        store = HealthHistoryStore(populated_history)
        store.initialize()
        snapshots = store.get_last_n_snapshots(5)

        trend_config.history_dir = populated_history
        analyzer = TrendAnalyzer(trend_config)
        analyzer.load_snapshots(snapshots)

        assert len(analyzer.snapshots) == 5

    def test_load_snapshots_insufficient(self, trend_config):
        """Test loading insufficient snapshots."""
        analyzer = TrendAnalyzer(trend_config)

        with pytest.raises(InsufficientHistoryError):
            analyzer.load_snapshots([])

    def test_compute_score_trend(self, trend_config, populated_history):
        """Test score trend computation."""
        store = HealthHistoryStore(populated_history)
        store.initialize()
        snapshots = store.get_last_n_snapshots(5)

        trend_config.history_dir = populated_history
        analyzer = TrendAnalyzer(trend_config)
        analyzer.load_snapshots(snapshots)

        score_trend = analyzer.compute_score_trend()

        assert score_trend.direction == TrendDirection.DECLINING.value
        assert score_trend.regression.slope < 0
        assert len(score_trend.score_history) == 5
        assert score_trend.current_score == 75.0

    def test_compute_moving_averages(self, trend_config, populated_history):
        """Test moving average computation."""
        store = HealthHistoryStore(populated_history)
        store.initialize()
        snapshots = store.get_last_n_snapshots(5)

        trend_config.history_dir = populated_history
        analyzer = TrendAnalyzer(trend_config)
        analyzer.load_snapshots(snapshots)

        score_trend = analyzer.compute_score_trend()

        assert 3 in score_trend.moving_averages
        assert score_trend.moving_averages[3].window_size == 3
        assert score_trend.moving_averages[3].current_value > 0

    def test_compute_volatility(self, trend_config, populated_history):
        """Test volatility computation."""
        store = HealthHistoryStore(populated_history)
        store.initialize()
        snapshots = store.get_last_n_snapshots(5)

        trend_config.history_dir = populated_history
        analyzer = TrendAnalyzer(trend_config)
        analyzer.load_snapshots(snapshots)

        score_trend = analyzer.compute_score_trend()

        assert score_trend.volatility.standard_deviation > 0
        assert score_trend.volatility.variance > 0
        assert score_trend.volatility.volatility_trend in ["increasing", "stable", "decreasing"]

    def test_compute_issue_trend(self, trend_config, populated_history):
        """Test issue trend computation."""
        store = HealthHistoryStore(populated_history)
        store.initialize()
        snapshots = store.get_last_n_snapshots(5)

        trend_config.history_dir = populated_history
        analyzer = TrendAnalyzer(trend_config)
        analyzer.load_snapshots(snapshots)

        issue_trend = analyzer.compute_issue_trend()

        assert issue_trend.total_issues_trend is not None
        assert issue_trend.critical_issues_trend is not None
        assert issue_trend.direction in [d.value for d in TrendDirection]

    def test_detect_anomalies(self, trend_config, populated_history):
        """Test anomaly detection."""
        store = HealthHistoryStore(populated_history)
        store.initialize()
        snapshots = store.get_last_n_snapshots(5)

        trend_config.history_dir = populated_history
        trend_config.zscore_threshold = 1.0  # Lower threshold to detect more
        analyzer = TrendAnalyzer(trend_config)
        analyzer.load_snapshots(snapshots)

        # First compute score trend
        analyzer.compute_score_trend()

        anomalies = analyzer.detect_anomalies()
        # May or may not have anomalies depending on data
        assert isinstance(anomalies, list)

    def test_generate_prediction(self, trend_config, populated_history):
        """Test prediction generation."""
        store = HealthHistoryStore(populated_history)
        store.initialize()
        snapshots = store.get_last_n_snapshots(5)

        trend_config.history_dir = populated_history
        analyzer = TrendAnalyzer(trend_config)
        analyzer.load_snapshots(snapshots)
        analyzer.compute_score_trend()

        prediction = analyzer.generate_prediction()

        assert prediction.predicted_score is not None
        assert len(prediction.predictions) == trend_config.prediction_horizon
        assert 0 <= prediction.probability_yellow <= 1
        assert 0 <= prediction.probability_red <= 1

    def test_generate_early_warnings(self, trend_config, populated_history):
        """Test early warning generation."""
        store = HealthHistoryStore(populated_history)
        store.initialize()
        snapshots = store.get_last_n_snapshots(5)

        trend_config.history_dir = populated_history
        analyzer = TrendAnalyzer(trend_config)
        analyzer.load_snapshots(snapshots)
        analyzer.compute_score_trend()

        warnings = analyzer.generate_early_warnings()
        assert isinstance(warnings, list)

        # Should have degradation warning due to declining trend
        if warnings:
            warning_types = [w.warning_type for w in warnings]
            # May have "slow_degradation" if trend is significant
            assert all(isinstance(w, EarlyWarning) for w in warnings)

    def test_analyze_complete(self, trend_config, populated_history):
        """Test complete analysis."""
        store = HealthHistoryStore(populated_history)
        store.initialize()
        snapshots = store.get_last_n_snapshots(5)

        trend_config.history_dir = populated_history
        analyzer = TrendAnalyzer(trend_config)
        analyzer.load_snapshots(snapshots)

        report = analyzer.analyze()

        assert isinstance(report, TrendReport)
        assert report.snapshots_analyzed == 5
        assert report.overall_trend in [d.value for d in TrendDirection]
        assert report.current_score == 75.0
        assert report.score_trend is not None
        assert report.issue_trend is not None

    def test_generate_graph_data(self, trend_config, populated_history):
        """Test graph data generation."""
        store = HealthHistoryStore(populated_history)
        store.initialize()
        snapshots = store.get_last_n_snapshots(5)

        trend_config.history_dir = populated_history
        analyzer = TrendAnalyzer(trend_config)
        analyzer.load_snapshots(snapshots)
        analyzer.compute_score_trend()
        analyzer.generate_prediction()
        analyzer.detect_anomalies()
        analyzer.generate_early_warnings()

        graph_data = analyzer.generate_graph_data()

        assert len(graph_data.timestamps) == 5
        assert len(graph_data.scores) == 5
        assert len(graph_data.regression_line) == 5


# ============================================================================
# TrendEngine Tests
# ============================================================================

class TestTrendEngine:
    """Tests for TrendEngine."""

    def test_run_success(self, populated_history, temp_dir):
        """Test successful trend analysis run."""
        output_path = temp_dir / "trend_report.json"

        config = TrendConfig(
            history_dir=populated_history,
            output_path=output_path,
            min_snapshots=3,
            verbose=False
        )

        engine = TrendEngine(config)
        report, exit_code = engine.run()

        assert exit_code == EXIT_TREND_SUCCESS
        assert report.snapshots_analyzed == 5
        assert output_path.exists()

    def test_run_insufficient_history(self, history_dir, temp_dir):
        """Test run with insufficient history."""
        # Add only 1 snapshot
        store = HealthHistoryStore(history_dir)
        store.initialize()

        dashboard = temp_dir / "dashboard.json"
        with open(dashboard, 'w') as f:
            json.dump({
                "repository_score": 90.0,
                "overall_health": "green",
                "total_issues": 0,
                "critical_issues": 0,
            }, f)
        store.add_snapshot(dashboard)

        config = TrendConfig(
            history_dir=history_dir,
            min_snapshots=3,
        )

        engine = TrendEngine(config)
        report, exit_code = engine.run()

        assert exit_code == EXIT_INSUFFICIENT_HISTORY

    def test_run_with_charts(self, populated_history, temp_dir):
        """Test run with chart generation."""
        chart_dir = temp_dir / "charts"

        config = TrendConfig(
            history_dir=populated_history,
            output_path=temp_dir / "trend_report.json",
            generate_charts=True,
            chart_output_dir=chart_dir,
            min_snapshots=3,
        )

        engine = TrendEngine(config)
        report, exit_code = engine.run()

        assert exit_code == EXIT_TREND_SUCCESS
        # Chart dir should be created even if matplotlib not available
        assert chart_dir.exists()


# ============================================================================
# Utility Functions Tests
# ============================================================================

class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_add_snapshot_to_history(self, history_dir, sample_dashboard_path):
        """Test add_snapshot_to_history function."""
        metadata = add_snapshot_to_history(
            history_dir,
            sample_dashboard_path,
            "1.0.0"
        )

        assert metadata.version == "1.0.0"
        assert metadata.repository_score == 85.0

    def test_get_trend_summary_insufficient(self, history_dir, sample_dashboard_path):
        """Test get_trend_summary with insufficient data."""
        # Add only 1 snapshot
        add_snapshot_to_history(history_dir, sample_dashboard_path)

        summary = get_trend_summary(history_dir)
        assert summary["status"] == "insufficient_data"

    def test_get_trend_summary_success(self, populated_history):
        """Test get_trend_summary with sufficient data."""
        summary = get_trend_summary(populated_history)

        assert summary["status"] == "ok"
        assert summary["snapshots"] == 5
        assert summary["trend"] in ["improving", "stable", "declining"]
        assert "score_change" in summary


# ============================================================================
# Data Class Tests
# ============================================================================

class TestDataClasses:
    """Tests for data classes."""

    def test_snapshot_metadata_to_dict(self):
        """Test SnapshotMetadata serialization."""
        metadata = SnapshotMetadata(
            snapshot_id="test_id",
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            version="1.0.0",
            file_path=Path("/test/path.json"),
            file_size=1000,
            repository_score=85.0,
            overall_health="green",
            total_issues=5,
            critical_issues=0
        )

        d = metadata.to_dict()
        assert d["snapshot_id"] == "test_id"
        assert d["version"] == "1.0.0"
        assert d["repository_score"] == 85.0

    def test_snapshot_metadata_from_dict(self):
        """Test SnapshotMetadata deserialization."""
        data = {
            "snapshot_id": "test_id",
            "timestamp": "2024-01-01T12:00:00",
            "version": "1.0.0",
            "file_path": "/test/path.json",
            "file_size": 1000,
            "repository_score": 85.0,
            "overall_health": "green",
            "total_issues": 5,
            "critical_issues": 0
        }

        metadata = SnapshotMetadata.from_dict(data)
        assert metadata.snapshot_id == "test_id"
        assert metadata.version == "1.0.0"

    def test_regression_result_predict(self):
        """Test RegressionResult prediction."""
        result = RegressionResult(
            slope=2.0,
            intercept=1.0,
            r_squared=0.99,
            standard_error=0.1,
            p_value=0.001
        )

        assert result.predict(0) == 1.0
        assert result.predict(5) == 11.0
        assert result.is_significant

    def test_trend_report_to_dict(self, populated_history):
        """Test TrendReport serialization."""
        store = HealthHistoryStore(populated_history)
        store.initialize()
        snapshots = store.get_last_n_snapshots(5)

        config = TrendConfig(history_dir=populated_history)
        analyzer = TrendAnalyzer(config)
        analyzer.load_snapshots(snapshots)

        report = analyzer.analyze()
        d = report.to_dict()

        assert "report_id" in d
        assert "overall_trend" in d
        assert "current_score" in d
        assert "predicted_next_score" in d


# ============================================================================
# Chart Generator Tests
# ============================================================================

class TestTrendChartGenerator:
    """Tests for TrendChartGenerator."""

    def test_ascii_chart_generation(self, temp_dir):
        """Test ASCII chart generation."""
        generator = TrendChartGenerator(temp_dir)

        values = [85.0, 80.0, 75.0, 70.0, 65.0]
        chart = generator._generate_ascii_chart(values, "Test Chart")

        assert "Test Chart" in chart
        assert "Start" in chart
        assert "End" in chart

    def test_ascii_chart_empty_values(self, temp_dir):
        """Test ASCII chart with empty values."""
        generator = TrendChartGenerator(temp_dir)

        chart = generator._generate_ascii_chart([], "Empty Chart")
        assert "No data" in chart


# ============================================================================
# Exception Tests
# ============================================================================

class TestExceptions:
    """Tests for custom exceptions."""

    def test_insufficient_history_error(self):
        """Test InsufficientHistoryError."""
        error = InsufficientHistoryError("Not enough data")
        assert error.exit_code == EXIT_INSUFFICIENT_HISTORY

    def test_invalid_snapshot_error(self):
        """Test InvalidSnapshotError."""
        error = InvalidSnapshotError("Corrupted snapshot")
        assert error.exit_code == EXIT_INVALID_SNAPSHOT

    def test_computation_error(self):
        """Test ComputationError."""
        error = ComputationError("Math failed")
        assert error.exit_code == EXIT_COMPUTATION_ERROR


# ============================================================================
# CLI Tests
# ============================================================================

class TestCLI:
    """Tests for CLI functionality."""

    def test_cli_import(self):
        """Test that CLI module imports correctly."""
        from analytics.run_trends import main, create_parser
        assert callable(main)
        assert callable(create_parser)

    def test_cli_parser_creation(self):
        """Test CLI parser creation."""
        from analytics.run_trends import create_parser
        parser = create_parser()
        assert parser is not None

    def test_cli_help(self):
        """Test CLI help output."""
        from analytics.run_trends import create_parser
        parser = create_parser()
        # Should not raise
        help_text = parser.format_help()
        assert "--history-dir" in help_text
        assert "--output" in help_text
        assert "--generate-charts" in help_text


# ============================================================================
# Edge Case Tests
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_snapshot_regression(self, trend_config):
        """Test regression with single snapshot."""
        values = [85.0]
        result = StatisticsCalculator.linear_regression([0], values)
        assert result.slope == 0.0

    def test_constant_score_trend(self, history_dir, temp_dir, trend_config):
        """Test trend with constant scores."""
        store = HealthHistoryStore(history_dir)
        store.initialize()

        # Add 5 snapshots with same score
        for i in range(5):
            dashboard = temp_dir / f"dashboard_{i}.json"
            with open(dashboard, 'w') as f:
                json.dump({
                    "repository_score": 80.0,
                    "overall_health": "green",
                    "total_issues": 5,
                    "critical_issues": 0,
                    "version": f"1.0.{i}"
                }, f)
            store.add_snapshot(dashboard, f"1.0.{i}", datetime.utcnow() + timedelta(days=i))
            dashboard.unlink()

        snapshots = store.get_last_n_snapshots(5)
        trend_config.history_dir = history_dir
        analyzer = TrendAnalyzer(trend_config)
        analyzer.load_snapshots(snapshots)

        score_trend = analyzer.compute_score_trend()
        assert score_trend.direction == TrendDirection.STABLE.value

    def test_extreme_score_drop(self, history_dir, temp_dir, trend_config):
        """Test handling of extreme score drop."""
        store = HealthHistoryStore(history_dir)
        store.initialize()

        scores = [95.0, 90.0, 85.0, 30.0, 25.0]  # Extreme drop

        for i, score in enumerate(scores):
            dashboard = temp_dir / f"dashboard_{i}.json"
            with open(dashboard, 'w') as f:
                json.dump({
                    "repository_score": score,
                    "overall_health": "red" if score < 40 else "green",
                    "total_issues": 20 - i,
                    "critical_issues": 5 if score < 40 else 0,
                }, f)
            store.add_snapshot(dashboard, f"1.0.{i}", datetime.utcnow() + timedelta(days=i))
            dashboard.unlink()

        snapshots = store.get_last_n_snapshots(5)
        trend_config.history_dir = history_dir
        trend_config.score_drop_threshold = 10.0
        analyzer = TrendAnalyzer(trend_config)
        analyzer.load_snapshots(snapshots)
        analyzer.compute_score_trend()

        anomalies = analyzer.detect_anomalies()
        # Should detect the extreme drop
        drop_anomalies = [a for a in anomalies if a.anomaly_type == AnomalyType.SUDDEN_SCORE_DROP.value]
        assert len(drop_anomalies) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
