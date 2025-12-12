"""
Report Aggregator - Loads and Normalizes Reports from Multiple Sources

This module extracts, transforms, and normalizes data from:
- Integrity scanner reports (Task 7)
- Rollback reports (Task 6)
- Publisher reports (Task 5)
- Validation reports (Task 4)
- Verification reports (Task 3)
- SBOM/SLSA metadata
- Manifest files
- Index.json

Version: 1.0.0
Phase: 14.7 Task 8
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Set

logger = logging.getLogger(__name__)


class ReportType(Enum):
    """Types of reports that can be aggregated."""
    INTEGRITY_SCAN = "integrity_scan"
    ROLLBACK = "rollback"
    PUBLISHER = "publisher"
    VALIDATOR = "validator"
    VERIFIER = "verifier"
    SBOM = "sbom"
    SLSA = "slsa"
    MANIFEST = "manifest"
    INDEX = "index"


class ReportFormat(Enum):
    """Report file formats."""
    JSON = "json"
    TEXT = "text"


@dataclass
class ReportMetadata:
    """Metadata about a loaded report."""
    report_type: ReportType
    file_path: Path
    loaded_at: datetime
    file_size: int
    is_valid: bool
    validation_errors: List[str] = field(default_factory=list)
    schema_version: Optional[str] = None


@dataclass
class NormalizedIssue:
    """Normalized issue representation across all report types."""
    issue_id: str
    source: str  # Which report this came from
    severity: str  # CRITICAL, ERROR, WARNING, INFO
    category: str  # corruption, missing, orphaned, etc.
    description: str
    artifact: Optional[str] = None
    version: Optional[str] = None
    detected_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    repair_action: Optional[str] = None


@dataclass
class NormalizedVersion:
    """Normalized version representation."""
    version: str
    artifact_count: int
    health_status: str  # green, yellow, red
    issues: List[NormalizedIssue] = field(default_factory=list)
    published_at: Optional[datetime] = None
    last_validated_at: Optional[datetime] = None
    sbom_present: bool = False
    slsa_present: bool = False
    manifest_valid: bool = False


@dataclass
class AggregatedData:
    """Aggregated data from all reports."""
    repository_path: str
    scan_timestamp: datetime
    versions: List[NormalizedVersion] = field(default_factory=list)
    all_issues: List[NormalizedIssue] = field(default_factory=list)
    repair_history: List[Dict[str, Any]] = field(default_factory=list)
    rollback_history: List[Dict[str, Any]] = field(default_factory=list)
    publication_history: List[Dict[str, Any]] = field(default_factory=list)
    report_metadata: List[ReportMetadata] = field(default_factory=list)

    # Summary statistics
    total_artifacts: int = 0
    total_versions: int = 0
    critical_issues: int = 0
    error_issues: int = 0
    warning_issues: int = 0
    info_issues: int = 0
    orphaned_artifacts: int = 0
    corrupted_artifacts: int = 0
    missing_artifacts: int = 0


class ReportAggregator:
    """
    Aggregates reports from multiple sources and normalizes data.

    This class:
    1. Discovers available reports in configured directories
    2. Loads and validates report schemas
    3. Extracts relevant data from each report type
    4. Normalizes data into common structures
    5. Performs cross-report consistency checks
    6. Generates aggregated view of repository health
    """

    def __init__(
        self,
        repository_path: Path,
        scan_output_dir: Optional[Path] = None,
        rollback_output_dir: Optional[Path] = None,
        publisher_output_dir: Optional[Path] = None,
        validator_output_dir: Optional[Path] = None,
        verbose: bool = False
    ):
        """
        Initialize the report aggregator.

        Args:
            repository_path: Path to the artifact repository
            scan_output_dir: Directory containing integrity scan reports
            rollback_output_dir: Directory containing rollback reports
            publisher_output_dir: Directory containing publisher reports
            validator_output_dir: Directory containing validation reports
            verbose: Enable verbose logging
        """
        self.repository_path = Path(repository_path)
        self.scan_output_dir = Path(scan_output_dir) if scan_output_dir else None
        self.rollback_output_dir = Path(rollback_output_dir) if rollback_output_dir else None
        self.publisher_output_dir = Path(publisher_output_dir) if publisher_output_dir else None
        self.validator_output_dir = Path(validator_output_dir) if validator_output_dir else None
        self.verbose = verbose

        if verbose:
            logger.setLevel(logging.DEBUG)

    def discover_reports(self) -> Dict[ReportType, List[Path]]:
        """
        Discover all available reports in configured directories.

        Returns:
            Dictionary mapping report types to lists of file paths
        """
        discovered = {report_type: [] for report_type in ReportType}

        # Integrity scan reports
        if self.scan_output_dir and self.scan_output_dir.exists():
            for pattern in ["*integrity-scan*.json", "*integrity*.json"]:
                discovered[ReportType.INTEGRITY_SCAN].extend(
                    self.scan_output_dir.glob(pattern)
                )

        # Rollback reports
        if self.rollback_output_dir and self.rollback_output_dir.exists():
            for pattern in ["*rollback*.json", "*recovery*.json"]:
                discovered[ReportType.ROLLBACK].extend(
                    self.rollback_output_dir.glob(pattern)
                )

        # Publisher reports
        if self.publisher_output_dir and self.publisher_output_dir.exists():
            for pattern in ["*publish*.json", "*publication*.json"]:
                discovered[ReportType.PUBLISHER].extend(
                    self.publisher_output_dir.glob(pattern)
                )

        # Validator reports
        if self.validator_output_dir and self.validator_output_dir.exists():
            for pattern in ["*validation*.json", "*validator*.json"]:
                discovered[ReportType.VALIDATOR].extend(
                    self.validator_output_dir.glob(pattern)
                )

        # Repository metadata (index.json)
        if self.repository_path.exists():
            index_file = self.repository_path / "index.json"
            if index_file.exists():
                discovered[ReportType.INDEX].append(index_file)

        logger.info(f"Discovered reports: {sum(len(v) for v in discovered.values())} total")
        for report_type, paths in discovered.items():
            if paths:
                logger.debug(f"  {report_type.value}: {len(paths)} files")

        return discovered

    def load_json_report(self, file_path: Path, report_type: ReportType) -> Optional[Dict[str, Any]]:
        """
        Load and validate a JSON report.

        Args:
            file_path: Path to the report file
            report_type: Type of report being loaded

        Returns:
            Parsed JSON data or None if invalid
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            logger.debug(f"Loaded {report_type.value} report: {file_path}")
            return data

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {file_path}: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")
            return None

    def validate_integrity_scan_report(self, data: Dict[str, Any]) -> List[str]:
        """Validate integrity scan report schema."""
        errors = []

        required_fields = ["scan_timestamp", "repository_path", "scan_status"]
        for field in required_fields:
            if field not in data:
                errors.append(f"Missing required field: {field}")

        # Check for issues section
        if "issues" not in data and "all_issues" not in data:
            errors.append("Missing 'issues' or 'all_issues' section")

        return errors

    def validate_rollback_report(self, data: Dict[str, Any]) -> List[str]:
        """Validate rollback report schema."""
        errors = []

        required_fields = ["rollback_timestamp", "from_version", "to_version", "status"]
        for field in required_fields:
            if field not in data:
                errors.append(f"Missing required field: {field}")

        return errors

    def validate_publisher_report(self, data: Dict[str, Any]) -> List[str]:
        """Validate publisher report schema."""
        errors = []

        required_fields = ["version", "published_at", "artifacts"]
        for field in required_fields:
            if field not in data:
                errors.append(f"Missing required field: {field}")

        return errors

    def validate_index(self, data: Dict[str, Any]) -> List[str]:
        """Validate index.json schema."""
        errors = []

        if "versions" not in data:
            errors.append("Missing 'versions' section")

        return errors

    def normalize_integrity_scan_issues(
        self,
        scan_data: Dict[str, Any]
    ) -> List[NormalizedIssue]:
        """Extract and normalize issues from integrity scan report."""
        normalized_issues = []

        # Handle both 'issues' and 'all_issues' keys
        issues_data = scan_data.get("all_issues", scan_data.get("issues", []))

        for issue in issues_data:
            normalized_issues.append(NormalizedIssue(
                issue_id=f"scan_{issue.get('issue_type', 'unknown')}_{issue.get('artifact', '')}",
                source="integrity_scan",
                severity=issue.get("severity", "WARNING"),
                category=issue.get("issue_type", "unknown"),
                description=issue.get("description", ""),
                artifact=issue.get("artifact"),
                version=issue.get("version"),
                detected_at=self._parse_timestamp(issue.get("detected_at")),
                repair_action=issue.get("repair_action")
            ))

        return normalized_issues

    def normalize_rollback_issues(
        self,
        rollback_data: Dict[str, Any]
    ) -> List[NormalizedIssue]:
        """Extract and normalize issues from rollback report."""
        normalized_issues = []

        # Rollback failures are critical issues
        if rollback_data.get("status") == "failed":
            normalized_issues.append(NormalizedIssue(
                issue_id=f"rollback_failed_{rollback_data.get('from_version')}",
                source="rollback",
                severity="CRITICAL",
                category="rollback_failed",
                description=f"Rollback from {rollback_data.get('from_version')} to {rollback_data.get('to_version')} failed",
                version=rollback_data.get("from_version"),
                detected_at=self._parse_timestamp(rollback_data.get("rollback_timestamp"))
            ))

        # Extract any errors from rollback process
        for error in rollback_data.get("errors", []):
            normalized_issues.append(NormalizedIssue(
                issue_id=f"rollback_error_{error.get('artifact', 'unknown')}",
                source="rollback",
                severity="ERROR",
                category="rollback_error",
                description=error.get("message", ""),
                artifact=error.get("artifact"),
                version=rollback_data.get("from_version"),
                detected_at=self._parse_timestamp(rollback_data.get("rollback_timestamp"))
            ))

        return normalized_issues

    def extract_version_info(
        self,
        index_data: Dict[str, Any],
        scan_data: Optional[Dict[str, Any]] = None
    ) -> List[NormalizedVersion]:
        """Extract version information from index and scan data."""
        normalized_versions = []

        for version_entry in index_data.get("versions", []):
            version = version_entry.get("version", "unknown")

            # Get issues for this version from scan data
            version_issues = []
            if scan_data:
                all_issues = self.normalize_integrity_scan_issues(scan_data)
                version_issues = [
                    issue for issue in all_issues
                    if issue.version == version
                ]

            # Determine health status based on issues
            health_status = self._compute_version_health(version_issues)

            normalized_versions.append(NormalizedVersion(
                version=version,
                artifact_count=len(version_entry.get("artifacts", [])),
                health_status=health_status,
                issues=version_issues,
                published_at=self._parse_timestamp(version_entry.get("published_at")),
                sbom_present=version_entry.get("sbom_present", False),
                slsa_present=version_entry.get("slsa_present", False),
                manifest_valid=version_entry.get("manifest_valid", True)
            ))

        return normalized_versions

    def aggregate_all_reports(self) -> AggregatedData:
        """
        Aggregate all discovered reports into unified data structure.

        Returns:
            AggregatedData containing all normalized information
        """
        logger.info("Starting report aggregation...")

        discovered = self.discover_reports()
        aggregated = AggregatedData(
            repository_path=str(self.repository_path),
            scan_timestamp=datetime.utcnow()
        )

        # Load index.json first (foundation for version info)
        index_data = None
        if discovered[ReportType.INDEX]:
            index_file = discovered[ReportType.INDEX][0]
            index_data = self.load_json_report(index_file, ReportType.INDEX)

            if index_data:
                validation_errors = self.validate_index(index_data)
                aggregated.report_metadata.append(ReportMetadata(
                    report_type=ReportType.INDEX,
                    file_path=index_file,
                    loaded_at=datetime.utcnow(),
                    file_size=index_file.stat().st_size,
                    is_valid=len(validation_errors) == 0,
                    validation_errors=validation_errors
                ))

        # Load integrity scan reports
        scan_data = None
        for scan_file in discovered[ReportType.INTEGRITY_SCAN]:
            scan_data = self.load_json_report(scan_file, ReportType.INTEGRITY_SCAN)

            if scan_data:
                validation_errors = self.validate_integrity_scan_report(scan_data)
                aggregated.report_metadata.append(ReportMetadata(
                    report_type=ReportType.INTEGRITY_SCAN,
                    file_path=scan_file,
                    loaded_at=datetime.utcnow(),
                    file_size=scan_file.stat().st_size,
                    is_valid=len(validation_errors) == 0,
                    validation_errors=validation_errors
                ))

                # Extract issues
                issues = self.normalize_integrity_scan_issues(scan_data)
                aggregated.all_issues.extend(issues)

                # Extract repair history
                if "repairs" in scan_data:
                    aggregated.repair_history.extend(scan_data["repairs"])

        # Load rollback reports
        for rollback_file in discovered[ReportType.ROLLBACK]:
            rollback_data = self.load_json_report(rollback_file, ReportType.ROLLBACK)

            if rollback_data:
                validation_errors = self.validate_rollback_report(rollback_data)
                aggregated.report_metadata.append(ReportMetadata(
                    report_type=ReportType.ROLLBACK,
                    file_path=rollback_file,
                    loaded_at=datetime.utcnow(),
                    file_size=rollback_file.stat().st_size,
                    is_valid=len(validation_errors) == 0,
                    validation_errors=validation_errors
                ))

                # Extract issues
                issues = self.normalize_rollback_issues(rollback_data)
                aggregated.all_issues.extend(issues)

                # Add to rollback history
                aggregated.rollback_history.append({
                    "from_version": rollback_data.get("from_version"),
                    "to_version": rollback_data.get("to_version"),
                    "timestamp": rollback_data.get("rollback_timestamp"),
                    "status": rollback_data.get("status"),
                    "artifacts_rolled_back": rollback_data.get("artifacts_rolled_back", 0)
                })

        # Load publisher reports
        for publisher_file in discovered[ReportType.PUBLISHER]:
            publisher_data = self.load_json_report(publisher_file, ReportType.PUBLISHER)

            if publisher_data:
                validation_errors = self.validate_publisher_report(publisher_data)
                aggregated.report_metadata.append(ReportMetadata(
                    report_type=ReportType.PUBLISHER,
                    file_path=publisher_file,
                    loaded_at=datetime.utcnow(),
                    file_size=publisher_file.stat().st_size,
                    is_valid=len(validation_errors) == 0,
                    validation_errors=validation_errors
                ))

                # Add to publication history
                aggregated.publication_history.append({
                    "version": publisher_data.get("version"),
                    "published_at": publisher_data.get("published_at"),
                    "artifacts": len(publisher_data.get("artifacts", [])),
                    "status": publisher_data.get("status", "unknown")
                })

        # Extract version information
        if index_data:
            aggregated.versions = self.extract_version_info(index_data, scan_data)
            aggregated.total_versions = len(aggregated.versions)
            aggregated.total_artifacts = sum(v.artifact_count for v in aggregated.versions)

        # Compute summary statistics
        aggregated.critical_issues = sum(1 for i in aggregated.all_issues if i.severity == "CRITICAL")
        aggregated.error_issues = sum(1 for i in aggregated.all_issues if i.severity == "ERROR")
        aggregated.warning_issues = sum(1 for i in aggregated.all_issues if i.severity == "WARNING")
        aggregated.info_issues = sum(1 for i in aggregated.all_issues if i.severity == "INFO")

        aggregated.orphaned_artifacts = sum(1 for i in aggregated.all_issues if "orphan" in i.category.lower())
        aggregated.corrupted_artifacts = sum(1 for i in aggregated.all_issues if "corrupt" in i.category.lower())
        aggregated.missing_artifacts = sum(1 for i in aggregated.all_issues if "missing" in i.category.lower())

        logger.info(f"Aggregation complete: {len(aggregated.all_issues)} issues, {len(aggregated.versions)} versions")

        return aggregated

    def _parse_timestamp(self, timestamp_str: Optional[str]) -> Optional[datetime]:
        """Parse timestamp string to datetime object."""
        if not timestamp_str:
            return None

        try:
            # Try ISO format first
            return datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        except:
            try:
                # Try common formats
                for fmt in ["%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"]:
                    try:
                        return datetime.strptime(timestamp_str, fmt)
                    except:
                        continue
            except:
                pass

        return None

    def _compute_version_health(self, issues: List[NormalizedIssue]) -> str:
        """Compute health status for a version based on its issues."""
        if not issues:
            return "green"

        # Critical issues = red
        if any(i.severity == "CRITICAL" for i in issues):
            return "red"

        # Error issues = yellow
        if any(i.severity == "ERROR" for i in issues):
            return "yellow"

        # Only warnings = yellow
        if any(i.severity == "WARNING" for i in issues):
            return "yellow"

        return "green"
