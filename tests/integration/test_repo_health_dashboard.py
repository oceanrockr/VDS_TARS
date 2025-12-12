"""
Integration Tests - Repository Health Dashboard

Test suite for Phase 14.7 Task 8 - Repository Health Dashboard & Aggregated Analytics Engine.

This test suite covers:
- ReportAggregator: Loading, validating, and normalizing reports
- HealthScoreCalculator: Computing health scores and statuses
- RecommendationGenerator: Generating actionable recommendations
- HTMLRenderer: Rendering HTML dashboards
- RepositoryHealthDashboard: End-to-end orchestration
- CLI integration: Command-line interface
- Exit codes: 60-69 validation

Version: 1.0.0
Phase: 14.7 Task 8
"""

import json
import pytest
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

# Import components under test
from analytics.report_aggregator import (
    ReportAggregator,
    ReportType,
    NormalizedIssue,
    NormalizedVersion,
    AggregatedData
)
from analytics.html_renderer import HTMLRenderer
from analytics.repository_health_dashboard import (
    RepositoryHealthDashboard,
    DashboardConfig,
    DashboardFormat,
    HealthStatus,
    HealthThresholds,
    HealthScoreCalculator,
    RecommendationGenerator,
    EXIT_HEALTH_OK,
    EXIT_HEALTH_WARNING,
    EXIT_HEALTH_CRITICAL,
    EXIT_AGGREGATION_FAILURE,
    EXIT_MISSING_REPORTS,
    EXIT_MALFORMED_REPORT,
    EXIT_HTML_RENDER_FAILURE,
    EXIT_DASHBOARD_WRITE_FAILURE,
    DashboardError,
    AggregationError,
    MalformedReportError,
    HTMLRenderError,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_dir():
    """Create temporary directory for test artifacts."""
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


@pytest.fixture
def sample_repository(temp_dir: Path) -> Path:
    """Create sample repository with index.json."""
    repo_dir = temp_dir / "repository"
    repo_dir.mkdir(parents=True)

    # Create index.json
    index_data = {
        "versions": [
            {
                "version": "v1.0.0",
                "artifacts": ["file1.tar.gz", "file2.tar.gz"],
                "published_at": "2025-11-28T10:00:00Z",
                "sbom_present": True,
                "slsa_present": True,
                "manifest_valid": True
            },
            {
                "version": "v1.0.1",
                "artifacts": ["file3.tar.gz"],
                "published_at": "2025-11-28T11:00:00Z",
                "sbom_present": False,
                "slsa_present": True,
                "manifest_valid": True
            }
        ]
    }

    with open(repo_dir / "index.json", 'w') as f:
        json.dump(index_data, f, indent=2)

    return repo_dir


@pytest.fixture
def sample_scan_report(temp_dir: Path) -> Path:
    """Create sample integrity scan report."""
    scan_dir = temp_dir / "scan-reports"
    scan_dir.mkdir(parents=True)

    scan_data = {
        "scan_timestamp": "2025-11-28T12:00:00Z",
        "repository_path": str(temp_dir / "repository"),
        "scan_status": "SUCCESS_WITH_WARNINGS",
        "total_issues": 3,
        "all_issues": [
            {
                "issue_type": "ARTIFACT_ORPHANED",
                "severity": "WARNING",
                "description": "Orphaned artifact detected",
                "artifact": "old-file.tar.gz",
                "version": None,
                "detected_at": "2025-11-28T12:00:00Z"
            },
            {
                "issue_type": "SBOM_MISSING",
                "severity": "ERROR",
                "description": "SBOM file missing for version",
                "artifact": None,
                "version": "v1.0.1",
                "detected_at": "2025-11-28T12:00:00Z"
            },
            {
                "issue_type": "MANIFEST_VALID",
                "severity": "INFO",
                "description": "Manifest validation passed",
                "artifact": None,
                "version": "v1.0.0",
                "detected_at": "2025-11-28T12:00:00Z"
            }
        ]
    }

    report_path = scan_dir / "integrity-scan-report.json"
    with open(report_path, 'w') as f:
        json.dump(scan_data, f, indent=2)

    return scan_dir


@pytest.fixture
def sample_rollback_report(temp_dir: Path) -> Path:
    """Create sample rollback report."""
    rollback_dir = temp_dir / "rollback-reports"
    rollback_dir.mkdir(parents=True)

    rollback_data = {
        "rollback_timestamp": "2025-11-28T13:00:00Z",
        "from_version": "v1.0.1",
        "to_version": "v1.0.0",
        "status": "success",
        "artifacts_rolled_back": 1,
        "errors": []
    }

    report_path = rollback_dir / "rollback-report.json"
    with open(report_path, 'w') as f:
        json.dump(rollback_data, f, indent=2)

    return rollback_dir


@pytest.fixture
def sample_publisher_report(temp_dir: Path) -> Path:
    """Create sample publisher report."""
    publisher_dir = temp_dir / "publisher-reports"
    publisher_dir.mkdir(parents=True)

    publisher_data = {
        "version": "v1.0.0",
        "published_at": "2025-11-28T10:00:00Z",
        "artifacts": ["file1.tar.gz", "file2.tar.gz"],
        "status": "success"
    }

    report_path = publisher_dir / "publication-report.json"
    with open(report_path, 'w') as f:
        json.dump(publisher_data, f, indent=2)

    return publisher_dir


# ============================================================================
# ReportAggregator Tests
# ============================================================================

class TestReportAggregator:
    """Test ReportAggregator functionality."""

    def test_aggregator_initialization(self, sample_repository: Path):
        """Test aggregator initialization."""
        aggregator = ReportAggregator(
            repository_path=sample_repository,
            verbose=True
        )

        assert aggregator.repository_path == sample_repository
        assert aggregator.verbose is True

    def test_discover_reports(
        self,
        sample_repository: Path,
        sample_scan_report: Path,
        sample_rollback_report: Path
    ):
        """Test report discovery."""
        aggregator = ReportAggregator(
            repository_path=sample_repository,
            scan_output_dir=sample_scan_report,
            rollback_output_dir=sample_rollback_report
        )

        discovered = aggregator.discover_reports()

        assert ReportType.INDEX in discovered
        assert len(discovered[ReportType.INDEX]) == 1

        assert ReportType.INTEGRITY_SCAN in discovered
        assert len(discovered[ReportType.INTEGRITY_SCAN]) == 1

        assert ReportType.ROLLBACK in discovered
        assert len(discovered[ReportType.ROLLBACK]) == 1

    def test_load_json_report(
        self,
        sample_repository: Path,
        sample_scan_report: Path
    ):
        """Test JSON report loading."""
        aggregator = ReportAggregator(repository_path=sample_repository)

        report_path = sample_scan_report / "integrity-scan-report.json"
        data = aggregator.load_json_report(report_path, ReportType.INTEGRITY_SCAN)

        assert data is not None
        assert "scan_timestamp" in data
        assert "all_issues" in data

    def test_validate_integrity_scan_report(self, sample_repository: Path):
        """Test integrity scan report validation."""
        aggregator = ReportAggregator(repository_path=sample_repository)

        valid_data = {
            "scan_timestamp": "2025-11-28T12:00:00Z",
            "repository_path": "/path/to/repo",
            "scan_status": "SUCCESS",
            "all_issues": []
        }

        errors = aggregator.validate_integrity_scan_report(valid_data)
        assert len(errors) == 0

        invalid_data = {"scan_timestamp": "2025-11-28T12:00:00Z"}
        errors = aggregator.validate_integrity_scan_report(invalid_data)
        assert len(errors) > 0

    def test_normalize_integrity_scan_issues(self, sample_repository: Path):
        """Test issue normalization from scan reports."""
        aggregator = ReportAggregator(repository_path=sample_repository)

        scan_data = {
            "all_issues": [
                {
                    "issue_type": "ARTIFACT_CORRUPTED",
                    "severity": "CRITICAL",
                    "description": "File corrupted",
                    "artifact": "file.tar.gz",
                    "version": "v1.0.0"
                }
            ]
        }

        issues = aggregator.normalize_integrity_scan_issues(scan_data)

        assert len(issues) == 1
        assert issues[0].severity == "CRITICAL"
        assert issues[0].category == "ARTIFACT_CORRUPTED"
        assert issues[0].source == "integrity_scan"

    def test_aggregate_all_reports(
        self,
        sample_repository: Path,
        sample_scan_report: Path,
        sample_rollback_report: Path,
        sample_publisher_report: Path
    ):
        """Test complete report aggregation."""
        aggregator = ReportAggregator(
            repository_path=sample_repository,
            scan_output_dir=sample_scan_report,
            rollback_output_dir=sample_rollback_report,
            publisher_output_dir=sample_publisher_report
        )

        aggregated = aggregator.aggregate_all_reports()

        assert isinstance(aggregated, AggregatedData)
        assert aggregated.total_versions == 2
        assert aggregated.total_artifacts == 3
        assert len(aggregated.all_issues) > 0
        assert len(aggregated.versions) == 2


# ============================================================================
# HealthScoreCalculator Tests
# ============================================================================

class TestHealthScoreCalculator:
    """Test HealthScoreCalculator functionality."""

    def test_calculator_initialization(self):
        """Test calculator initialization."""
        thresholds = HealthThresholds(green_min=80.0, yellow_min=50.0)
        calculator = HealthScoreCalculator(thresholds)

        assert calculator.thresholds.green_min == 80.0
        assert calculator.thresholds.yellow_min == 50.0

    def test_calculate_score_perfect_repository(self):
        """Test score calculation for perfect repository."""
        calculator = HealthScoreCalculator(HealthThresholds())

        # Perfect repository: no issues, all metadata present
        aggregated_data = AggregatedData(
            repository_path="/test",
            scan_timestamp=datetime.utcnow(),
            total_versions=2,
            total_artifacts=5,
            critical_issues=0,
            error_issues=0,
            warning_issues=0,
            info_issues=0,
            versions=[
                NormalizedVersion(
                    version="v1.0.0",
                    artifact_count=3,
                    health_status="green",
                    sbom_present=True,
                    slsa_present=True,
                    manifest_valid=True
                ),
                NormalizedVersion(
                    version="v1.0.1",
                    artifact_count=2,
                    health_status="green",
                    sbom_present=True,
                    slsa_present=True,
                    manifest_valid=True
                )
            ]
        )

        score = calculator.calculate_score(aggregated_data)
        assert score > 100  # Will be capped at 100
        # After capping
        score = min(100.0, score)
        assert score == 100.0

    def test_calculate_score_with_issues(self):
        """Test score calculation with various issues."""
        calculator = HealthScoreCalculator(HealthThresholds())

        aggregated_data = AggregatedData(
            repository_path="/test",
            scan_timestamp=datetime.utcnow(),
            total_versions=1,
            total_artifacts=2,
            critical_issues=1,  # -10 points
            error_issues=2,     # -10 points (2 * 5)
            warning_issues=3,   # -6 points (3 * 2)
            info_issues=5,      # -2.5 points (5 * 0.5)
            versions=[
                NormalizedVersion(
                    version="v1.0.0",
                    artifact_count=2,
                    health_status="red",
                    sbom_present=False,  # -5 points
                    slsa_present=False,  # -5 points
                    manifest_valid=False  # -3 points
                )
            ]
        )

        score = calculator.calculate_score(aggregated_data)
        expected = 100 - 10 - 10 - 6 - 2.5 - 5 - 5 - 3
        assert abs(score - expected) < 0.1

    def test_determine_status_green(self):
        """Test green status determination."""
        calculator = HealthScoreCalculator(HealthThresholds())

        aggregated_data = AggregatedData(
            repository_path="/test",
            scan_timestamp=datetime.utcnow(),
            critical_issues=0,
            error_issues=0
        )

        status = calculator.determine_status(85.0, aggregated_data)
        assert status == HealthStatus.GREEN

    def test_determine_status_yellow(self):
        """Test yellow status determination."""
        calculator = HealthScoreCalculator(HealthThresholds())

        aggregated_data = AggregatedData(
            repository_path="/test",
            scan_timestamp=datetime.utcnow(),
            critical_issues=0,
            error_issues=1
        )

        status = calculator.determine_status(65.0, aggregated_data)
        assert status == HealthStatus.YELLOW

    def test_determine_status_red_critical_issues(self):
        """Test red status with critical issues."""
        calculator = HealthScoreCalculator(HealthThresholds())

        aggregated_data = AggregatedData(
            repository_path="/test",
            scan_timestamp=datetime.utcnow(),
            critical_issues=1,
            error_issues=0
        )

        # Even with high score, critical issues = red
        status = calculator.determine_status(95.0, aggregated_data)
        assert status == HealthStatus.RED

    def test_determine_status_red_low_score(self):
        """Test red status with low score."""
        calculator = HealthScoreCalculator(HealthThresholds())

        aggregated_data = AggregatedData(
            repository_path="/test",
            scan_timestamp=datetime.utcnow(),
            critical_issues=0
        )

        status = calculator.determine_status(40.0, aggregated_data)
        assert status == HealthStatus.RED


# ============================================================================
# RecommendationGenerator Tests
# ============================================================================

class TestRecommendationGenerator:
    """Test RecommendationGenerator functionality."""

    def test_generate_recommendations_healthy(self):
        """Test recommendations for healthy repository."""
        aggregated_data = AggregatedData(
            repository_path="/test",
            scan_timestamp=datetime.utcnow(),
            critical_issues=0,
            error_issues=0,
            warning_issues=0,
            corrupted_artifacts=0,
            orphaned_artifacts=0,
            missing_artifacts=0
        )

        recommendations = RecommendationGenerator.generate_recommendations(
            aggregated_data,
            95.0,
            HealthStatus.GREEN
        )

        assert len(recommendations) > 0
        assert any("healthy" in rec.lower() for rec in recommendations)

    def test_generate_recommendations_critical_issues(self):
        """Test recommendations for critical issues."""
        aggregated_data = AggregatedData(
            repository_path="/test",
            scan_timestamp=datetime.utcnow(),
            critical_issues=3,
            corrupted_artifacts=2,
            missing_artifacts=1
        )

        recommendations = RecommendationGenerator.generate_recommendations(
            aggregated_data,
            45.0,
            HealthStatus.RED
        )

        assert len(recommendations) > 0
        assert any("critical" in rec.lower() or "urgent" in rec.lower() for rec in recommendations)
        assert any("corrupted" in rec.lower() for rec in recommendations)
        assert any("missing" in rec.lower() for rec in recommendations)

    def test_generate_recommendations_missing_metadata(self):
        """Test recommendations for missing SBOM/SLSA."""
        aggregated_data = AggregatedData(
            repository_path="/test",
            scan_timestamp=datetime.utcnow(),
            versions=[
                NormalizedVersion(
                    version="v1.0.0",
                    artifact_count=2,
                    health_status="yellow",
                    sbom_present=False,
                    slsa_present=False
                )
            ]
        )

        recommendations = RecommendationGenerator.generate_recommendations(
            aggregated_data,
            70.0,
            HealthStatus.YELLOW
        )

        assert any("sbom" in rec.lower() for rec in recommendations)
        assert any("slsa" in rec.lower() for rec in recommendations)


# ============================================================================
# HTMLRenderer Tests
# ============================================================================

class TestHTMLRenderer:
    """Test HTMLRenderer functionality."""

    def test_renderer_initialization(self):
        """Test renderer initialization."""
        renderer = HTMLRenderer(verbose=True)
        assert renderer.verbose is True

    def test_render_dashboard(self, temp_dir: Path):
        """Test HTML dashboard rendering."""
        renderer = HTMLRenderer()

        aggregated_data = AggregatedData(
            repository_path="/test",
            scan_timestamp=datetime.utcnow(),
            total_versions=2,
            total_artifacts=5,
            critical_issues=0,
            error_issues=1,
            warning_issues=2,
            info_issues=3,
            versions=[
                NormalizedVersion(
                    version="v1.0.0",
                    artifact_count=3,
                    health_status="green",
                    sbom_present=True,
                    slsa_present=True
                )
            ]
        )

        recommendations = ["Recommendation 1", "Recommendation 2"]
        output_path = temp_dir / "test-dashboard.html"

        success = renderer.render_dashboard(
            aggregated_data,
            85.5,
            "green",
            recommendations,
            output_path
        )

        assert success is True
        assert output_path.exists()

        # Verify HTML content
        content = output_path.read_text()
        assert "T.A.R.S. Repository Health Dashboard" in content
        assert "85.5/100" in content
        assert "GREEN" in content
        assert "v1.0.0" in content

    def test_render_dashboard_with_issues(self, temp_dir: Path):
        """Test HTML rendering with issues."""
        renderer = HTMLRenderer()

        issue = NormalizedIssue(
            issue_id="test-1",
            source="integrity_scan",
            severity="CRITICAL",
            category="ARTIFACT_CORRUPTED",
            description="Test issue",
            artifact="file.tar.gz",
            version="v1.0.0"
        )

        aggregated_data = AggregatedData(
            repository_path="/test",
            scan_timestamp=datetime.utcnow(),
            all_issues=[issue],
            critical_issues=1
        )

        output_path = temp_dir / "test-dashboard-issues.html"
        success = renderer.render_dashboard(
            aggregated_data,
            45.0,
            "red",
            ["Fix critical issues"],
            output_path
        )

        assert success is True
        content = output_path.read_text()
        assert "CRITICAL" in content
        assert "Test issue" in content


# ============================================================================
# RepositoryHealthDashboard Tests
# ============================================================================

class TestRepositoryHealthDashboard:
    """Test RepositoryHealthDashboard orchestration."""

    def test_dashboard_initialization(self, sample_repository: Path, temp_dir: Path):
        """Test dashboard initialization."""
        config = DashboardConfig(
            repository_path=sample_repository,
            output_dir=temp_dir / "dashboard",
            format=DashboardFormat.BOTH
        )

        dashboard = RepositoryHealthDashboard(config)
        assert dashboard.config == config

    def test_generate_dashboard_json_only(
        self,
        sample_repository: Path,
        sample_scan_report: Path,
        temp_dir: Path
    ):
        """Test dashboard generation with JSON format."""
        config = DashboardConfig(
            repository_path=sample_repository,
            output_dir=temp_dir / "dashboard",
            format=DashboardFormat.JSON,
            scan_output_dir=sample_scan_report
        )

        dashboard = RepositoryHealthDashboard(config)
        health_report = dashboard.generate_dashboard()

        assert health_report is not None
        assert health_report.overall_health in ["green", "yellow", "red"]
        assert 0 <= health_report.repository_score <= 100

        # Check JSON file
        json_path = temp_dir / "dashboard" / "health-dashboard.json"
        assert json_path.exists()

        # HTML should not exist
        html_path = temp_dir / "dashboard" / "health-dashboard.html"
        assert not html_path.exists()

    def test_generate_dashboard_html_only(
        self,
        sample_repository: Path,
        sample_scan_report: Path,
        temp_dir: Path
    ):
        """Test dashboard generation with HTML format."""
        config = DashboardConfig(
            repository_path=sample_repository,
            output_dir=temp_dir / "dashboard",
            format=DashboardFormat.HTML,
            scan_output_dir=sample_scan_report
        )

        dashboard = RepositoryHealthDashboard(config)
        health_report = dashboard.generate_dashboard()

        assert health_report is not None

        # Check HTML file
        html_path = temp_dir / "dashboard" / "health-dashboard.html"
        assert html_path.exists()

        # JSON should not exist
        json_path = temp_dir / "dashboard" / "health-dashboard.json"
        assert not json_path.exists()

    def test_generate_dashboard_both_formats(
        self,
        sample_repository: Path,
        sample_scan_report: Path,
        temp_dir: Path
    ):
        """Test dashboard generation with both formats."""
        config = DashboardConfig(
            repository_path=sample_repository,
            output_dir=temp_dir / "dashboard",
            format=DashboardFormat.BOTH,
            scan_output_dir=sample_scan_report
        )

        dashboard = RepositoryHealthDashboard(config)
        health_report = dashboard.generate_dashboard()

        assert health_report is not None

        # Both files should exist
        json_path = temp_dir / "dashboard" / "health-dashboard.json"
        html_path = temp_dir / "dashboard" / "health-dashboard.html"
        assert json_path.exists()
        assert html_path.exists()

    def test_determine_exit_code_green(self, sample_repository: Path, temp_dir: Path):
        """Test exit code for green status."""
        config = DashboardConfig(
            repository_path=sample_repository,
            output_dir=temp_dir / "dashboard"
        )

        dashboard = RepositoryHealthDashboard(config)

        # Mock health report with green status
        from analytics.repository_health_dashboard import HealthReport
        health_report = HealthReport(
            overall_health="green",
            repository_score=95.0,
            scan_timestamp=datetime.utcnow().isoformat(),
            repository_path=str(sample_repository)
        )

        exit_code = dashboard.determine_exit_code(health_report)
        assert exit_code == EXIT_HEALTH_OK

    def test_determine_exit_code_yellow_no_fail(
        self,
        sample_repository: Path,
        temp_dir: Path
    ):
        """Test exit code for yellow status without fail flag."""
        config = DashboardConfig(
            repository_path=sample_repository,
            output_dir=temp_dir / "dashboard",
            fail_on_yellow=False
        )

        dashboard = RepositoryHealthDashboard(config)

        from analytics.repository_health_dashboard import HealthReport
        health_report = HealthReport(
            overall_health="yellow",
            repository_score=65.0,
            scan_timestamp=datetime.utcnow().isoformat(),
            repository_path=str(sample_repository)
        )

        exit_code = dashboard.determine_exit_code(health_report)
        assert exit_code == EXIT_HEALTH_OK

    def test_determine_exit_code_yellow_with_fail(
        self,
        sample_repository: Path,
        temp_dir: Path
    ):
        """Test exit code for yellow status with fail flag."""
        config = DashboardConfig(
            repository_path=sample_repository,
            output_dir=temp_dir / "dashboard",
            fail_on_yellow=True
        )

        dashboard = RepositoryHealthDashboard(config)

        from analytics.repository_health_dashboard import HealthReport
        health_report = HealthReport(
            overall_health="yellow",
            repository_score=65.0,
            scan_timestamp=datetime.utcnow().isoformat(),
            repository_path=str(sample_repository)
        )

        exit_code = dashboard.determine_exit_code(health_report)
        assert exit_code == EXIT_HEALTH_WARNING

    def test_determine_exit_code_red(self, sample_repository: Path, temp_dir: Path):
        """Test exit code for red status."""
        config = DashboardConfig(
            repository_path=sample_repository,
            output_dir=temp_dir / "dashboard",
            fail_on_red=True
        )

        dashboard = RepositoryHealthDashboard(config)

        from analytics.repository_health_dashboard import HealthReport
        health_report = HealthReport(
            overall_health="red",
            repository_score=35.0,
            scan_timestamp=datetime.utcnow().isoformat(),
            repository_path=str(sample_repository)
        )

        exit_code = dashboard.determine_exit_code(health_report)
        assert exit_code == EXIT_HEALTH_CRITICAL


# ============================================================================
# CLI Integration Tests
# ============================================================================

class TestCLIIntegration:
    """Test command-line interface."""

    def test_cli_help(self):
        """Test CLI help message."""
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "-m", "analytics.repository_health_dashboard", "--help"],
            capture_output=True,
            text=True
        )

        assert result.returncode == 0
        assert "T.A.R.S. Repository Health Dashboard" in result.stdout
        assert "--repository-path" in result.stdout
        assert "--output-dir" in result.stdout

    def test_cli_basic_execution(
        self,
        sample_repository: Path,
        sample_scan_report: Path,
        temp_dir: Path
    ):
        """Test basic CLI execution."""
        import subprocess
        import sys

        output_dir = temp_dir / "cli-dashboard"

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "analytics.repository_health_dashboard",
                "--repository-path", str(sample_repository),
                "--output-dir", str(output_dir),
                "--scan-reports", str(sample_scan_report),
                "--format", "json"
            ],
            capture_output=True,
            text=True,
            timeout=30
        )

        # Should succeed or return dashboard-specific exit code
        assert result.returncode in [EXIT_HEALTH_OK, EXIT_HEALTH_WARNING, EXIT_HEALTH_CRITICAL]

        # Check output file
        json_path = output_dir / "health-dashboard.json"
        assert json_path.exists()


# ============================================================================
# Performance Tests
# ============================================================================

class TestPerformance:
    """Test performance characteristics."""

    def test_dashboard_generation_performance(
        self,
        sample_repository: Path,
        sample_scan_report: Path,
        temp_dir: Path
    ):
        """Test dashboard generation completes within 3 seconds."""
        import time

        config = DashboardConfig(
            repository_path=sample_repository,
            output_dir=temp_dir / "dashboard",
            scan_output_dir=sample_scan_report
        )

        dashboard = RepositoryHealthDashboard(config)

        start_time = time.time()
        health_report = dashboard.generate_dashboard()
        elapsed_time = time.time() - start_time

        assert health_report is not None
        assert elapsed_time < 3.0  # Should complete in under 3 seconds

    def test_large_issue_list_performance(self, sample_repository: Path, temp_dir: Path):
        """Test performance with large number of issues."""
        import time

        # Create aggregated data with many issues
        issues = [
            NormalizedIssue(
                issue_id=f"issue-{i}",
                source="test",
                severity="WARNING",
                category="test",
                description=f"Test issue {i}",
                artifact=f"file{i}.tar.gz"
            )
            for i in range(100)
        ]

        aggregated_data = AggregatedData(
            repository_path=str(sample_repository),
            scan_timestamp=datetime.utcnow(),
            all_issues=issues,
            warning_issues=100
        )

        renderer = HTMLRenderer()
        output_path = temp_dir / "large-dashboard.html"

        start_time = time.time()
        success = renderer.render_dashboard(
            aggregated_data,
            50.0,
            "yellow",
            ["Fix issues"],
            output_path
        )
        elapsed_time = time.time() - start_time

        assert success is True
        assert elapsed_time < 2.0  # Should handle 100 issues quickly


# ============================================================================
# End-to-End Tests
# ============================================================================

class TestEndToEnd:
    """End-to-end integration tests."""

    def test_complete_dashboard_workflow(
        self,
        sample_repository: Path,
        sample_scan_report: Path,
        sample_rollback_report: Path,
        sample_publisher_report: Path,
        temp_dir: Path
    ):
        """Test complete dashboard generation workflow."""
        config = DashboardConfig(
            repository_path=sample_repository,
            output_dir=temp_dir / "e2e-dashboard",
            format=DashboardFormat.BOTH,
            scan_output_dir=sample_scan_report,
            rollback_output_dir=sample_rollback_report,
            publisher_output_dir=sample_publisher_report,
            verbose=True
        )

        dashboard = RepositoryHealthDashboard(config)
        health_report = dashboard.generate_dashboard()

        # Verify health report
        assert health_report is not None
        assert health_report.overall_health in ["green", "yellow", "red"]
        assert 0 <= health_report.repository_score <= 100
        assert health_report.total_versions == 2
        assert health_report.total_issues >= 0
        assert len(health_report.recommendations) > 0

        # Verify output files
        json_path = temp_dir / "e2e-dashboard" / "health-dashboard.json"
        html_path = temp_dir / "e2e-dashboard" / "health-dashboard.html"
        assert json_path.exists()
        assert html_path.exists()

        # Verify JSON content
        with open(json_path) as f:
            json_data = json.load(f)
            assert "overall_health" in json_data
            assert "repository_score" in json_data
            assert "recommendations" in json_data

        # Verify HTML content
        html_content = html_path.read_text()
        assert "<!DOCTYPE html>" in html_content
        assert "T.A.R.S." in html_content
        assert str(health_report.repository_score) in html_content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
