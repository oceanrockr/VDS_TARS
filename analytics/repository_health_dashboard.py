"""
Repository Health Dashboard - Core Orchestrator

This module orchestrates the complete dashboard generation process:
1. Loads and aggregates all available reports
2. Computes repository health score (0-100)
3. Determines overall health status (green/yellow/red)
4. Generates actionable recommendations
5. Produces JSON + HTML dashboard outputs

Exit Codes (60-69):
- 60: Health OK (Green)
- 61: Health Warning (Yellow)
- 62: Health Critical (Red)
- 63: Aggregation Failure
- 64: Missing Reports
- 65: Malformed Report
- 66: HTML Render Failure
- 67: Dashboard Write Failure
- 68: Health Threshold Violation
- 69: General Dashboard Error

Version: 1.0.0
Phase: 14.7 Task 8
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any

from analytics.report_aggregator import ReportAggregator, AggregatedData, NormalizedIssue
from analytics.html_renderer import HTMLRenderer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Exit Codes (60-69)
# ============================================================================

EXIT_HEALTH_OK = 60
EXIT_HEALTH_WARNING = 61
EXIT_HEALTH_CRITICAL = 62
EXIT_AGGREGATION_FAILURE = 63
EXIT_MISSING_REPORTS = 64
EXIT_MALFORMED_REPORT = 65
EXIT_HTML_RENDER_FAILURE = 66
EXIT_DASHBOARD_WRITE_FAILURE = 67
EXIT_HEALTH_THRESHOLD_VIOLATION = 68
EXIT_GENERAL_DASHBOARD_ERROR = 69


# ============================================================================
# Custom Exceptions
# ============================================================================

class DashboardError(Exception):
    """Base exception for dashboard errors."""
    exit_code = EXIT_GENERAL_DASHBOARD_ERROR


class AggregationError(DashboardError):
    """Failed to aggregate reports."""
    exit_code = EXIT_AGGREGATION_FAILURE


class MalformedReportError(DashboardError):
    """Report has invalid schema."""
    exit_code = EXIT_MALFORMED_REPORT


class MissingReportsError(DashboardError):
    """Required reports are missing."""
    exit_code = EXIT_MISSING_REPORTS


class HTMLRenderError(DashboardError):
    """Failed to render HTML dashboard."""
    exit_code = EXIT_HTML_RENDER_FAILURE


class DashboardWriteError(DashboardError):
    """Failed to write dashboard files."""
    exit_code = EXIT_DASHBOARD_WRITE_FAILURE


class HealthThresholdError(DashboardError):
    """Health score below configured threshold."""
    exit_code = EXIT_HEALTH_THRESHOLD_VIOLATION


# ============================================================================
# Enums
# ============================================================================

class HealthStatus(Enum):
    """Overall repository health status."""
    GREEN = "green"      # Score >= 80, no critical issues
    YELLOW = "yellow"    # Score 50-79, or has error/warning issues
    RED = "red"          # Score < 50, or has critical issues


class DashboardFormat(Enum):
    """Dashboard output formats."""
    JSON = "json"
    HTML = "html"
    BOTH = "both"


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class HealthThresholds:
    """Configurable health score thresholds."""
    green_min: float = 80.0   # Minimum score for green status
    yellow_min: float = 50.0  # Minimum score for yellow status
    # Below yellow_min = red


@dataclass
class DashboardConfig:
    """Configuration for dashboard generation."""
    repository_path: Path
    output_dir: Path
    format: DashboardFormat = DashboardFormat.BOTH

    # Report directories
    scan_output_dir: Optional[Path] = None
    rollback_output_dir: Optional[Path] = None
    publisher_output_dir: Optional[Path] = None
    validator_output_dir: Optional[Path] = None

    # Health thresholds
    thresholds: HealthThresholds = field(default_factory=HealthThresholds)

    # Flags
    fail_on_yellow: bool = False  # Exit with error on yellow status
    fail_on_red: bool = True      # Exit with error on red status
    verbose: bool = False


@dataclass
class HealthReport:
    """Complete repository health report."""
    # Overall health
    overall_health: str  # green, yellow, red
    repository_score: float  # 0-100
    scan_timestamp: str
    repository_path: str

    # Issues summary
    total_issues: int = 0
    critical_issues: int = 0
    error_issues: int = 0
    warning_issues: int = 0
    info_issues: int = 0

    # Version health
    total_versions: int = 0
    healthy_versions: int = 0  # green
    warning_versions: int = 0  # yellow
    critical_versions: int = 0  # red

    # Artifact health
    total_artifacts: int = 0
    orphaned_artifacts: int = 0
    corrupted_artifacts: int = 0
    missing_artifacts: int = 0

    # Operation history
    repair_count: int = 0
    rollback_count: int = 0
    publication_count: int = 0

    # Detailed data
    issues: List[Dict[str, Any]] = field(default_factory=list)
    versions_health: List[Dict[str, Any]] = field(default_factory=list)
    repair_history: List[Dict[str, Any]] = field(default_factory=list)
    rollback_history: List[Dict[str, Any]] = field(default_factory=list)
    publication_history: List[Dict[str, Any]] = field(default_factory=list)

    # Recommendations
    recommendations: List[str] = field(default_factory=list)

    # Metadata
    reports_aggregated: int = 0
    generation_duration_ms: float = 0.0


# ============================================================================
# Health Score Calculator
# ============================================================================

class HealthScoreCalculator:
    """
    Calculates repository health score (0-100).

    Scoring Algorithm:
    - Start with base score of 100
    - Deduct points for issues based on severity:
      * CRITICAL: -10 points each
      * ERROR: -5 points each
      * WARNING: -2 points each
      * INFO: -0.5 points each
    - Deduct points for missing metadata:
      * Missing SBOM: -5 points per version
      * Missing SLSA: -5 points per version
      * Invalid manifest: -3 points per version
    - Bonus points for clean history:
      * No issues in last 5 versions: +10 points
      * All versions have SBOM/SLSA: +5 points
    - Floor at 0, cap at 100
    """

    def __init__(self, thresholds: HealthThresholds):
        """Initialize calculator with thresholds."""
        self.thresholds = thresholds

    def calculate_score(self, aggregated_data: AggregatedData) -> float:
        """Calculate health score from aggregated data."""
        score = 100.0

        # Deduct for issues
        score -= aggregated_data.critical_issues * 10.0
        score -= aggregated_data.error_issues * 5.0
        score -= aggregated_data.warning_issues * 2.0
        score -= aggregated_data.info_issues * 0.5

        # Deduct for missing metadata
        for version in aggregated_data.versions:
            if not version.sbom_present:
                score -= 5.0
            if not version.slsa_present:
                score -= 5.0
            if not version.manifest_valid:
                score -= 3.0

        # Bonus for clean recent history (last 5 versions)
        recent_versions = sorted(
            aggregated_data.versions,
            key=lambda v: v.version,
            reverse=True
        )[:5]

        if recent_versions:
            recent_has_issues = any(len(v.issues) > 0 for v in recent_versions)
            if not recent_has_issues:
                score += 10.0

        # Bonus for complete metadata on all versions
        if aggregated_data.versions:
            all_have_sbom = all(v.sbom_present for v in aggregated_data.versions)
            all_have_slsa = all(v.slsa_present for v in aggregated_data.versions)
            if all_have_sbom and all_have_slsa:
                score += 5.0

        # Floor and cap
        score = max(0.0, min(100.0, score))

        return score

    def determine_status(
        self,
        score: float,
        aggregated_data: AggregatedData
    ) -> HealthStatus:
        """Determine health status from score and data."""
        # Critical issues always result in RED
        if aggregated_data.critical_issues > 0:
            return HealthStatus.RED

        # Score-based determination
        if score >= self.thresholds.green_min:
            return HealthStatus.GREEN
        elif score >= self.thresholds.yellow_min:
            return HealthStatus.YELLOW
        else:
            return HealthStatus.RED


# ============================================================================
# Recommendation Generator
# ============================================================================

class RecommendationGenerator:
    """Generates actionable recommendations based on health data."""

    @staticmethod
    def generate_recommendations(
        aggregated_data: AggregatedData,
        health_score: float,
        health_status: HealthStatus
    ) -> List[str]:
        """Generate recommendations from aggregated data."""
        recommendations = []

        # Critical issues
        if aggregated_data.critical_issues > 0:
            recommendations.append(
                f"⚠️  URGENT: Address {aggregated_data.critical_issues} critical "
                f"issue{'s' if aggregated_data.critical_issues != 1 else ''} immediately"
            )

        # Corrupted artifacts
        if aggregated_data.corrupted_artifacts > 0:
            recommendations.append(
                f"🔴 {aggregated_data.corrupted_artifacts} corrupted artifact{'s' if aggregated_data.corrupted_artifacts != 1 else ''} "
                f"detected - run integrity scan with repair enabled"
            )

        # Orphaned artifacts
        if aggregated_data.orphaned_artifacts > 0:
            recommendations.append(
                f"🟡 {aggregated_data.orphaned_artifacts} orphaned artifact{'s' if aggregated_data.orphaned_artifacts != 1 else ''} "
                f"found - consider running cleanup with --repair-orphans"
            )

        # Missing artifacts
        if aggregated_data.missing_artifacts > 0:
            recommendations.append(
                f"🔴 {aggregated_data.missing_artifacts} missing artifact{'s' if aggregated_data.missing_artifacts != 1 else ''} "
                f"- may require rollback or re-publication"
            )

        # Missing SBOM/SLSA
        missing_sbom = sum(1 for v in aggregated_data.versions if not v.sbom_present)
        missing_slsa = sum(1 for v in aggregated_data.versions if not v.slsa_present)

        if missing_sbom > 0:
            recommendations.append(
                f"📋 {missing_sbom} version{'s' if missing_sbom != 1 else ''} missing SBOM "
                f"- re-publish with SBOM generation enabled"
            )

        if missing_slsa > 0:
            recommendations.append(
                f"📋 {missing_slsa} version{'s' if missing_slsa != 1 else ''} missing SLSA provenance "
                f"- re-publish with SLSA generation enabled"
            )

        # Error issues
        if aggregated_data.error_issues > 0:
            recommendations.append(
                f"🟠 {aggregated_data.error_issues} error-level issue{'s' if aggregated_data.error_issues != 1 else ''} "
                f"require attention - review detailed issues table"
            )

        # Health score
        if health_score < 50:
            recommendations.append(
                f"❌ Health score critically low ({health_score:.1f}/100) - "
                f"immediate remediation required"
            )
        elif health_score < 80:
            recommendations.append(
                f"⚠️  Health score below optimal ({health_score:.1f}/100) - "
                f"address issues to improve repository health"
            )

        # No issues - positive recommendation
        if len(aggregated_data.all_issues) == 0:
            recommendations.append(
                "✅ Repository is healthy - no issues detected. Continue regular scans."
            )

        # Regular scanning
        if not recommendations or health_score >= 80:
            recommendations.append(
                "🔄 Schedule regular integrity scans (daily recommended) to maintain health"
            )

        return recommendations


# ============================================================================
# Repository Health Dashboard (Main Orchestrator)
# ============================================================================

class RepositoryHealthDashboard:
    """
    Main orchestrator for repository health dashboard generation.

    This class:
    1. Initializes report aggregator and renderers
    2. Loads and aggregates all available reports
    3. Computes health score and status
    4. Generates recommendations
    5. Produces JSON and/or HTML outputs
    6. Returns appropriate exit codes
    """

    def __init__(self, config: DashboardConfig):
        """
        Initialize dashboard with configuration.

        Args:
            config: Dashboard configuration
        """
        self.config = config

        if config.verbose:
            logger.setLevel(logging.DEBUG)

        # Initialize components
        self.aggregator = ReportAggregator(
            repository_path=config.repository_path,
            scan_output_dir=config.scan_output_dir,
            rollback_output_dir=config.rollback_output_dir,
            publisher_output_dir=config.publisher_output_dir,
            validator_output_dir=config.validator_output_dir,
            verbose=config.verbose
        )

        self.calculator = HealthScoreCalculator(config.thresholds)
        self.html_renderer = HTMLRenderer(verbose=config.verbose)

    def generate_dashboard(self) -> HealthReport:
        """
        Generate complete health dashboard.

        Returns:
            HealthReport containing all health data

        Raises:
            AggregationError: If report aggregation fails
            HTMLRenderError: If HTML rendering fails
            DashboardWriteError: If writing outputs fails
        """
        start_time = datetime.utcnow()

        try:
            logger.info("=" * 80)
            logger.info("REPOSITORY HEALTH DASHBOARD")
            logger.info("=" * 80)
            logger.info(f"Repository: {self.config.repository_path}")
            logger.info(f"Output: {self.config.output_dir}")
            logger.info("")

            # Step 1: Aggregate reports
            logger.info("Step 1: Aggregating reports...")
            aggregated_data = self.aggregator.aggregate_all_reports()

            # Check for minimum data
            if aggregated_data.total_versions == 0:
                logger.warning("No version data found in repository")

            reports_loaded = len(aggregated_data.report_metadata)
            logger.info(f"  Loaded {reports_loaded} report{'s' if reports_loaded != 1 else ''}")
            logger.info(f"  Found {aggregated_data.total_versions} version{'s' if aggregated_data.total_versions != 1 else ''}, "
                       f"{aggregated_data.total_artifacts} artifact{'s' if aggregated_data.total_artifacts != 1 else ''}")
            logger.info(f"  Detected {len(aggregated_data.all_issues)} issue{'s' if len(aggregated_data.all_issues) != 1 else ''}")
            logger.info("")

            # Step 2: Calculate health score
            logger.info("Step 2: Computing health score...")
            health_score = self.calculator.calculate_score(aggregated_data)
            health_status = self.calculator.determine_status(health_score, aggregated_data)

            logger.info(f"  Health Score: {health_score:.1f}/100")
            logger.info(f"  Health Status: {health_status.value.upper()}")
            logger.info("")

            # Step 3: Generate recommendations
            logger.info("Step 3: Generating recommendations...")
            recommendations = RecommendationGenerator.generate_recommendations(
                aggregated_data,
                health_score,
                health_status
            )
            logger.info(f"  Generated {len(recommendations)} recommendation{'s' if len(recommendations) != 1 else ''}")
            logger.info("")

            # Step 4: Build health report
            logger.info("Step 4: Building health report...")
            health_report = self._build_health_report(
                aggregated_data,
                health_score,
                health_status,
                recommendations,
                start_time
            )
            logger.info("  Health report built successfully")
            logger.info("")

            # Step 5: Write outputs
            logger.info("Step 5: Writing outputs...")
            self._write_outputs(health_report, aggregated_data, recommendations)
            logger.info("")

            logger.info("=" * 80)
            logger.info("DASHBOARD GENERATION COMPLETE")
            logger.info("=" * 80)
            logger.info(f"Status: {health_status.value.upper()}")
            logger.info(f"Score: {health_score:.1f}/100")
            logger.info(f"Issues: {len(aggregated_data.all_issues)}")
            logger.info(f"Output: {self.config.output_dir}")
            logger.info("=" * 80)

            return health_report

        except Exception as e:
            logger.error(f"Dashboard generation failed: {e}")
            raise DashboardError(f"Dashboard generation failed: {e}")

    def _build_health_report(
        self,
        aggregated_data: AggregatedData,
        health_score: float,
        health_status: HealthStatus,
        recommendations: List[str],
        start_time: datetime
    ) -> HealthReport:
        """Build complete health report from aggregated data."""
        # Count version health statuses
        healthy_versions = sum(1 for v in aggregated_data.versions if v.health_status == "green")
        warning_versions = sum(1 for v in aggregated_data.versions if v.health_status == "yellow")
        critical_versions = sum(1 for v in aggregated_data.versions if v.health_status == "red")

        # Convert issues to dict format
        issues_dict = [
            {
                "issue_id": issue.issue_id,
                "source": issue.source,
                "severity": issue.severity,
                "category": issue.category,
                "description": issue.description,
                "artifact": issue.artifact,
                "version": issue.version,
                "detected_at": issue.detected_at.isoformat() if issue.detected_at else None,
                "repair_action": issue.repair_action
            }
            for issue in aggregated_data.all_issues
        ]

        # Convert versions to dict format
        versions_dict = [
            {
                "version": v.version,
                "artifact_count": v.artifact_count,
                "health_status": v.health_status,
                "issue_count": len(v.issues),
                "sbom_present": v.sbom_present,
                "slsa_present": v.slsa_present,
                "manifest_valid": v.manifest_valid,
                "published_at": v.published_at.isoformat() if v.published_at else None
            }
            for v in aggregated_data.versions
        ]

        # Calculate duration
        duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

        return HealthReport(
            overall_health=health_status.value,
            repository_score=health_score,
            scan_timestamp=aggregated_data.scan_timestamp.isoformat(),
            repository_path=aggregated_data.repository_path,
            total_issues=len(aggregated_data.all_issues),
            critical_issues=aggregated_data.critical_issues,
            error_issues=aggregated_data.error_issues,
            warning_issues=aggregated_data.warning_issues,
            info_issues=aggregated_data.info_issues,
            total_versions=aggregated_data.total_versions,
            healthy_versions=healthy_versions,
            warning_versions=warning_versions,
            critical_versions=critical_versions,
            total_artifacts=aggregated_data.total_artifacts,
            orphaned_artifacts=aggregated_data.orphaned_artifacts,
            corrupted_artifacts=aggregated_data.corrupted_artifacts,
            missing_artifacts=aggregated_data.missing_artifacts,
            repair_count=len(aggregated_data.repair_history),
            rollback_count=len(aggregated_data.rollback_history),
            publication_count=len(aggregated_data.publication_history),
            issues=issues_dict,
            versions_health=versions_dict,
            repair_history=aggregated_data.repair_history,
            rollback_history=aggregated_data.rollback_history,
            publication_history=aggregated_data.publication_history,
            recommendations=recommendations,
            reports_aggregated=len(aggregated_data.report_metadata),
            generation_duration_ms=duration_ms
        )

    def _write_outputs(
        self,
        health_report: HealthReport,
        aggregated_data: AggregatedData,
        recommendations: List[str]
    ) -> None:
        """Write dashboard outputs (JSON and/or HTML)."""
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        # Write JSON report
        if self.config.format in [DashboardFormat.JSON, DashboardFormat.BOTH]:
            json_path = self.config.output_dir / "health-dashboard.json"
            try:
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(asdict(health_report), f, indent=2, default=str)
                logger.info(f"  ✓ JSON report: {json_path}")
            except Exception as e:
                raise DashboardWriteError(f"Failed to write JSON report: {e}")

        # Write HTML dashboard
        if self.config.format in [DashboardFormat.HTML, DashboardFormat.BOTH]:
            html_path = self.config.output_dir / "health-dashboard.html"
            try:
                success = self.html_renderer.render_dashboard(
                    aggregated_data,
                    health_report.repository_score,
                    health_report.overall_health,
                    recommendations,
                    html_path
                )
                if not success:
                    raise HTMLRenderError("HTML renderer returned failure")
                logger.info(f"  ✓ HTML dashboard: {html_path}")
            except Exception as e:
                raise HTMLRenderError(f"Failed to render HTML dashboard: {e}")

    def determine_exit_code(self, health_report: HealthReport) -> int:
        """
        Determine exit code based on health report and config.

        Args:
            health_report: Generated health report

        Returns:
            Exit code (60-69)
        """
        health_status = HealthStatus(health_report.overall_health)

        if health_status == HealthStatus.GREEN:
            return EXIT_HEALTH_OK
        elif health_status == HealthStatus.YELLOW:
            if self.config.fail_on_yellow:
                return EXIT_HEALTH_WARNING
            return EXIT_HEALTH_OK
        else:  # RED
            if self.config.fail_on_red:
                return EXIT_HEALTH_CRITICAL
            return EXIT_HEALTH_OK


# ============================================================================
# CLI Entry Point
# ============================================================================

def main() -> int:
    """
    Main CLI entry point.

    Returns:
        Exit code (60-69)
    """
    parser = argparse.ArgumentParser(
        description="T.A.R.S. Repository Health Dashboard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate dashboard with default settings
  python -m analytics.repository_health_dashboard \\
    --repository-path ./artifact-repository \\
    --output-dir ./dashboard

  # Generate with all report sources
  python -m analytics.repository_health_dashboard \\
    --repository-path ./artifact-repository \\
    --scan-reports ./integrity-scan \\
    --rollback-reports ./rollback \\
    --publisher-reports ./publish \\
    --output-dir ./dashboard \\
    --format both

  # Fail on yellow status
  python -m analytics.repository_health_dashboard \\
    --repository-path ./artifact-repository \\
    --output-dir ./dashboard \\
    --fail-on-yellow \\
    --verbose

Exit Codes:
  60 - Health OK (Green)
  61 - Health Warning (Yellow)
  62 - Health Critical (Red)
  63 - Aggregation Failure
  64 - Missing Reports
  65 - Malformed Report
  66 - HTML Render Failure
  67 - Dashboard Write Failure
  68 - Health Threshold Violation
  69 - General Dashboard Error
        """
    )

    # Repository configuration
    parser.add_argument(
        "--repository-path",
        type=Path,
        required=True,
        help="Path to artifact repository"
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for dashboard files"
    )

    # Report source directories
    parser.add_argument(
        "--scan-reports",
        type=Path,
        help="Directory containing integrity scan reports"
    )

    parser.add_argument(
        "--rollback-reports",
        type=Path,
        help="Directory containing rollback reports"
    )

    parser.add_argument(
        "--publisher-reports",
        type=Path,
        help="Directory containing publisher reports"
    )

    parser.add_argument(
        "--validator-reports",
        type=Path,
        help="Directory containing validation reports"
    )

    # Output format
    parser.add_argument(
        "--format",
        type=str,
        choices=["json", "html", "both"],
        default="both",
        help="Dashboard output format (default: both)"
    )

    # Health thresholds
    parser.add_argument(
        "--green-threshold",
        type=float,
        default=80.0,
        help="Minimum score for green status (default: 80.0)"
    )

    parser.add_argument(
        "--yellow-threshold",
        type=float,
        default=50.0,
        help="Minimum score for yellow status (default: 50.0)"
    )

    # Failure modes
    parser.add_argument(
        "--fail-on-yellow",
        action="store_true",
        help="Exit with error code on yellow status"
    )

    parser.add_argument(
        "--no-fail-on-red",
        action="store_true",
        help="Don't exit with error code on red status (exit 60 instead)"
    )

    # Verbosity
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )

    args = parser.parse_args()

    try:
        # Build configuration
        config = DashboardConfig(
            repository_path=args.repository_path,
            output_dir=args.output_dir,
            format=DashboardFormat(args.format),
            scan_output_dir=args.scan_reports,
            rollback_output_dir=args.rollback_reports,
            publisher_output_dir=args.publisher_reports,
            validator_output_dir=args.validator_reports,
            thresholds=HealthThresholds(
                green_min=args.green_threshold,
                yellow_min=args.yellow_threshold
            ),
            fail_on_yellow=args.fail_on_yellow,
            fail_on_red=not args.no_fail_on_red,
            verbose=args.verbose
        )

        # Generate dashboard
        dashboard = RepositoryHealthDashboard(config)
        health_report = dashboard.generate_dashboard()

        # Determine exit code
        exit_code = dashboard.determine_exit_code(health_report)

        return exit_code

    except DashboardError as e:
        logger.error(f"Dashboard error: {e}")
        return e.exit_code
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return EXIT_GENERAL_DASHBOARD_ERROR


if __name__ == "__main__":
    sys.exit(main())
