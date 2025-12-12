#!/usr/bin/env python3
"""
Repository Integrity Scanner - Phase 14.7 Task 7

Production-grade repository integrity validation and consistency verification system.
Provides automated detection of corruption, inconsistencies, and automated repair capabilities.

Author: T.A.R.S. Development Team
Version: 1.0.0
Date: 2025-11-28
"""

import json
import hashlib
import shutil
import uuid
import platform
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from enum import Enum
import logging


# ============================================================================
# LOGGING SETUP
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# CUSTOM EXCEPTIONS (Exit Codes 50-59)
# ============================================================================

class IntegrityError(Exception):
    """Base exception for all integrity errors."""
    exit_code = 59

    def __init__(self, message: str, exit_code: Optional[int] = None):
        super().__init__(message)
        if exit_code is not None:
            self.exit_code = exit_code


class IntegrityArtifactCorruptedError(IntegrityError):
    """Artifact integrity validation failed."""
    exit_code = 52


class IntegrityManifestMismatchError(IntegrityError):
    """Manifest data does not match actual artifacts."""
    exit_code = 53


class IntegrityIndexInconsistentError(IntegrityError):
    """Index.json inconsistent with repository state."""
    exit_code = 54


class IntegritySBOMSLSAError(IntegrityError):
    """SBOM/SLSA integrity validation failed."""
    exit_code = 55


class IntegrityOrphanDetectedError(IntegrityError):
    """Orphan artifacts detected in repository."""
    exit_code = 56


class IntegritySignatureError(IntegrityError):
    """Signature validation failed."""
    exit_code = 57


class IntegrityRepairError(IntegrityError):
    """Repair operation failed or required but not performed."""
    exit_code = 58


class IntegrityScanError(IntegrityError):
    """General integrity scan error."""
    exit_code = 59


# ============================================================================
# ENUMS
# ============================================================================

class IntegrityScanPolicy(Enum):
    """Integrity scan policy modes."""
    STRICT = "strict"
    LENIENT = "lenient"
    AUDIT_ONLY = "audit_only"


class IntegrityIssueType(Enum):
    """Types of integrity issues."""
    ARTIFACT_CORRUPTED = "artifact_corrupted"
    ARTIFACT_MISSING = "artifact_missing"
    ARTIFACT_ORPHANED = "artifact_orphaned"
    MANIFEST_MISMATCH = "manifest_mismatch"
    MANIFEST_MISSING = "manifest_missing"
    MANIFEST_MALFORMED = "manifest_malformed"
    INDEX_INCONSISTENT = "index_inconsistent"
    INDEX_MISSING = "index_missing"
    INDEX_MALFORMED = "index_malformed"
    INDEX_VERSION_MISSING = "index_version_missing"
    INDEX_VERSION_EXTRA = "index_version_extra"
    SBOM_MISSING = "sbom_missing"
    SBOM_MALFORMED = "sbom_malformed"
    SBOM_HASH_MISMATCH = "sbom_hash_mismatch"
    SLSA_MISSING = "slsa_missing"
    SLSA_MALFORMED = "slsa_malformed"
    SLSA_HASH_MISMATCH = "slsa_hash_mismatch"
    SIGNATURE_MISSING = "signature_missing"
    SIGNATURE_INVALID = "signature_invalid"
    VERSION_DUPLICATE = "version_duplicate"
    VERSION_ORDERING = "version_ordering"


class IntegrityIssueSeverity(Enum):
    """Severity levels for integrity issues."""
    CRITICAL = "critical"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class IntegrityRepairAction(Enum):
    """Types of repair actions."""
    REMOVE_ORPHAN = "remove_orphan"
    REBUILD_INDEX_JSON = "rebuild_index_json"
    REBUILD_INDEX_MD = "rebuild_index_md"
    FIX_INDEX_ENTRY = "fix_index_entry"
    RESTORE_MANIFEST = "restore_manifest"
    NO_ACTION = "no_action"


class IntegrityScanStatus(Enum):
    """Integrity scan status."""
    SUCCESS = "success"
    SUCCESS_WITH_WARNINGS = "success_with_warnings"
    FAILED = "failed"
    REPAIR_REQUIRED = "repair_required"


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class IntegrityIssue:
    """Represents a single integrity issue."""
    issue_type: str
    severity: str
    version: Optional[str]
    artifact: Optional[str]
    description: str
    expected_value: Optional[str] = None
    actual_value: Optional[str] = None
    repair_action: Optional[str] = None
    can_auto_repair: bool = False


@dataclass
class IntegrityArtifactValidation:
    """Validation result for a single artifact."""
    artifact_path: str
    exists: bool
    size_bytes: int = 0
    sha256_computed: Optional[str] = None
    sha256_expected: Optional[str] = None
    sha256_matches: bool = False
    signature_valid: Optional[bool] = None
    issues: List[IntegrityIssue] = field(default_factory=list)


@dataclass
class IntegrityVersionValidation:
    """Validation result for a complete version."""
    version: str
    in_index: bool
    in_repository: bool
    manifest_valid: bool
    sbom_valid: bool
    slsa_valid: bool
    total_artifacts: int = 0
    corrupted_artifacts: int = 0
    missing_artifacts: int = 0
    total_size_bytes: int = 0
    artifacts: List[IntegrityArtifactValidation] = field(default_factory=list)
    issues: List[IntegrityIssue] = field(default_factory=list)


@dataclass
class IntegrityScanReport:
    """Comprehensive integrity scan report."""
    scan_id: str
    timestamp: str
    repository_type: str
    repository_location: str
    policy_mode: str
    scan_status: str

    # Repository health
    total_versions: int = 0
    total_artifacts: int = 0
    total_size_bytes: int = 0
    orphan_artifacts: int = 0

    # Issues summary
    total_issues: int = 0
    critical_issues: int = 0
    error_issues: int = 0
    warning_issues: int = 0
    info_issues: int = 0

    # Issue breakdown
    corrupted_artifacts: int = 0
    missing_artifacts: int = 0
    manifest_issues: int = 0
    index_issues: int = 0
    sbom_slsa_issues: int = 0
    signature_issues: int = 0

    # Validation results
    versions: List[IntegrityVersionValidation] = field(default_factory=list)
    all_issues: List[IntegrityIssue] = field(default_factory=list)

    # Repair
    repair_enabled: bool = False
    repairs_applied: int = 0
    repairs_failed: int = 0

    # Performance
    scan_duration_seconds: float = 0.0

    # Exit
    exit_code: int = 50
    summary: str = ""


@dataclass
class IntegrityRepairResult:
    """Result of a repair operation."""
    action: str
    success: bool
    version: Optional[str] = None
    artifact: Optional[str] = None
    description: str = ""
    error: Optional[str] = None


# ============================================================================
# INTEGRITY REPOSITORY ADAPTER
# ============================================================================

class IntegrityRepositoryAdapter:
    """
    Repository adapter for integrity operations.
    Wraps publisher's AbstractRepository with integrity-specific methods.
    """

    def __init__(self, repository):
        """
        Initialize adapter.

        Args:
            repository: AbstractRepository instance (from publisher)
        """
        self.repository = repository
        logger.info(f"IntegrityRepositoryAdapter initialized for {repository.repo_type}")

    def list_all_versions(self) -> List[str]:
        """
        List all versions in repository.

        Returns:
            List of version strings
        """
        try:
            return self.repository.list_versions()
        except Exception as e:
            logger.error(f"Failed to list versions: {e}")
            return []

    def get_index(self) -> Optional[Dict[str, Any]]:
        """
        Get repository index.

        Returns:
            Index data or None
        """
        try:
            return self.repository.get_index()
        except Exception as e:
            logger.error(f"Failed to get index: {e}")
            return None

    def get_version_artifacts(self, version: str) -> List[str]:
        """
        List all artifacts for a version.

        Args:
            version: Version string

        Returns:
            List of artifact paths
        """
        artifacts = []
        try:
            # List artifacts in version directory
            if hasattr(self.repository, 'base_path'):
                version_dir = self.repository.base_path / version
                if version_dir.exists() and version_dir.is_dir():
                    for item in version_dir.rglob('*'):
                        if item.is_file():
                            rel_path = str(item.relative_to(self.repository.base_path))
                            artifacts.append(rel_path)
        except Exception as e:
            logger.error(f"Failed to list artifacts for {version}: {e}")

        return artifacts

    def get_artifact_content(self, artifact_path: str) -> Optional[bytes]:
        """
        Read artifact content.

        Args:
            artifact_path: Relative artifact path

        Returns:
            Artifact bytes or None
        """
        try:
            if hasattr(self.repository, 'base_path'):
                full_path = self.repository.base_path / artifact_path
                if full_path.exists():
                    return full_path.read_bytes()
        except Exception as e:
            logger.error(f"Failed to read {artifact_path}: {e}")

        return None

    def artifact_exists(self, artifact_path: str) -> bool:
        """
        Check if artifact exists.

        Args:
            artifact_path: Relative artifact path

        Returns:
            True if exists
        """
        return self.repository.exists(artifact_path)

    def get_artifact_size(self, artifact_path: str) -> int:
        """
        Get artifact size in bytes.

        Args:
            artifact_path: Relative artifact path

        Returns:
            Size in bytes or 0
        """
        try:
            if hasattr(self.repository, 'base_path'):
                full_path = self.repository.base_path / artifact_path
                if full_path.exists():
                    return full_path.stat().st_size
        except Exception as e:
            logger.error(f"Failed to get size of {artifact_path}: {e}")

        return 0

    def compute_sha256(self, artifact_path: str) -> Optional[str]:
        """
        Compute SHA256 hash of artifact.

        Args:
            artifact_path: Relative artifact path

        Returns:
            SHA256 hex digest or None
        """
        content = self.get_artifact_content(artifact_path)
        if content is not None:
            return hashlib.sha256(content).hexdigest()
        return None

    def get_manifest(self, version: str) -> Optional[Dict[str, Any]]:
        """
        Get version's manifest.json.

        Args:
            version: Version string

        Returns:
            Manifest data or None
        """
        manifest_path = f"{version}/manifest.json"
        content = self.get_artifact_content(manifest_path)
        if content:
            try:
                return json.loads(content.decode('utf-8'))
            except Exception as e:
                logger.error(f"Failed to parse manifest for {version}: {e}")
        return None

    def get_sbom(self, version: str) -> Optional[Dict[str, Any]]:
        """
        Get version's SBOM.

        Args:
            version: Version string

        Returns:
            SBOM data or None
        """
        sbom_path = f"{version}/sbom.json"
        content = self.get_artifact_content(sbom_path)
        if content:
            try:
                return json.loads(content.decode('utf-8'))
            except Exception as e:
                logger.error(f"Failed to parse SBOM for {version}: {e}")
        return None

    def get_slsa(self, version: str) -> Optional[Dict[str, Any]]:
        """
        Get version's SLSA provenance.

        Args:
            version: Version string

        Returns:
            SLSA data or None
        """
        slsa_path = f"{version}/slsa-provenance.json"
        content = self.get_artifact_content(slsa_path)
        if content:
            try:
                return json.loads(content.decode('utf-8'))
            except Exception as e:
                logger.error(f"Failed to parse SLSA for {version}: {e}")
        return None

    def delete_artifact(self, artifact_path: str) -> bool:
        """
        Delete artifact.

        Args:
            artifact_path: Relative artifact path

        Returns:
            True if successful
        """
        try:
            return self.repository.delete(artifact_path)
        except Exception as e:
            logger.error(f"Failed to delete {artifact_path}: {e}")
            return False

    def update_index(self, index_data: Dict[str, Any]) -> bool:
        """
        Update repository index.

        Args:
            index_data: New index data

        Returns:
            True if successful
        """
        try:
            return self.repository.update_index(index_data)
        except Exception as e:
            logger.error(f"Failed to update index: {e}")
            return False

    def list_all_artifacts(self) -> List[str]:
        """
        List ALL artifacts in repository (for orphan detection).

        Returns:
            List of all artifact paths
        """
        artifacts = []
        try:
            if hasattr(self.repository, 'base_path'):
                base = self.repository.base_path
                for item in base.rglob('*'):
                    if item.is_file():
                        # Exclude index files
                        rel = str(item.relative_to(base))
                        if rel not in ['index.json', 'index.md']:
                            artifacts.append(rel)
        except Exception as e:
            logger.error(f"Failed to list all artifacts: {e}")

        return artifacts


# ============================================================================
# INTEGRITY SCAN POLICY ENGINE
# ============================================================================

class IntegrityScanPolicyEngine:
    """
    Enforces integrity scan policies.
    """

    def __init__(self, policy_mode: IntegrityScanPolicy):
        """
        Initialize policy engine.

        Args:
            policy_mode: Policy mode
        """
        self.policy_mode = policy_mode
        logger.info(f"IntegrityScanPolicyEngine initialized with policy: {policy_mode.value}")

    def should_fail_on_issue(self, severity: IntegrityIssueSeverity) -> bool:
        """
        Determine if scan should fail on issue.

        Args:
            severity: Issue severity

        Returns:
            True if should fail
        """
        if self.policy_mode == IntegrityScanPolicy.AUDIT_ONLY:
            return False

        if self.policy_mode == IntegrityScanPolicy.STRICT:
            return severity in [IntegrityIssueSeverity.CRITICAL, IntegrityIssueSeverity.ERROR]

        if self.policy_mode == IntegrityScanPolicy.LENIENT:
            return severity == IntegrityIssueSeverity.CRITICAL

        return False

    def categorize_severity(self, issue_type: IntegrityIssueType) -> IntegrityIssueSeverity:
        """
        Categorize issue severity.

        Args:
            issue_type: Issue type

        Returns:
            Severity level
        """
        critical_issues = [
            IntegrityIssueType.ARTIFACT_CORRUPTED,
            IntegrityIssueType.MANIFEST_MISMATCH,
            IntegrityIssueType.INDEX_INCONSISTENT,
        ]

        error_issues = [
            IntegrityIssueType.ARTIFACT_MISSING,
            IntegrityIssueType.MANIFEST_MISSING,
            IntegrityIssueType.INDEX_MISSING,
            IntegrityIssueType.SBOM_MISSING,
            IntegrityIssueType.SLSA_MISSING,
            IntegrityIssueType.SIGNATURE_INVALID,
        ]

        warning_issues = [
            IntegrityIssueType.ARTIFACT_ORPHANED,
            IntegrityIssueType.SBOM_MALFORMED,
            IntegrityIssueType.SLSA_MALFORMED,
            IntegrityIssueType.SIGNATURE_MISSING,
            IntegrityIssueType.VERSION_ORDERING,
        ]

        if issue_type in critical_issues:
            return IntegrityIssueSeverity.CRITICAL
        elif issue_type in error_issues:
            return IntegrityIssueSeverity.ERROR
        elif issue_type in warning_issues:
            return IntegrityIssueSeverity.WARNING
        else:
            return IntegrityIssueSeverity.INFO

    def determine_exit_code(self, issues: List[IntegrityIssue]) -> int:
        """
        Determine appropriate exit code.

        Args:
            issues: List of issues

        Returns:
            Exit code (50-59)
        """
        if not issues:
            return 50  # Success

        # Count by type
        has_corrupted = any(i.issue_type == IntegrityIssueType.ARTIFACT_CORRUPTED.value for i in issues)
        has_manifest = any('manifest' in i.issue_type for i in issues)
        has_index = any('index' in i.issue_type for i in issues)
        has_sbom_slsa = any(i.issue_type in [
            IntegrityIssueType.SBOM_MISSING.value,
            IntegrityIssueType.SBOM_MALFORMED.value,
            IntegrityIssueType.SLSA_MISSING.value,
            IntegrityIssueType.SLSA_MALFORMED.value
        ] for i in issues)
        has_orphan = any(i.issue_type == IntegrityIssueType.ARTIFACT_ORPHANED.value for i in issues)
        has_signature = any('signature' in i.issue_type for i in issues)

        # Prioritize exit codes
        if has_corrupted:
            return 52
        if has_manifest:
            return 53
        if has_index:
            return 54
        if has_sbom_slsa:
            return 55
        if has_orphan:
            return 56
        if has_signature:
            return 57

        # Only warnings
        return 51


# ============================================================================
# INTEGRITY SCANNER
# ============================================================================

class IntegrityScanner:
    """
    Core integrity scanner implementation.
    """

    def __init__(self, adapter: IntegrityRepositoryAdapter, policy_engine: IntegrityScanPolicyEngine):
        """
        Initialize scanner.

        Args:
            adapter: Repository adapter
            policy_engine: Policy engine
        """
        self.adapter = adapter
        self.policy_engine = policy_engine
        logger.info("IntegrityScanner initialized")

    def scan_artifact(
        self,
        artifact_path: str,
        expected_hash: Optional[str] = None
    ) -> IntegrityArtifactValidation:
        """
        Scan single artifact.

        Args:
            artifact_path: Artifact path
            expected_hash: Expected SHA256 hash

        Returns:
            Artifact validation result
        """
        validation = IntegrityArtifactValidation(
            artifact_path=artifact_path,
            exists=self.adapter.artifact_exists(artifact_path)
        )

        if not validation.exists:
            issue = IntegrityIssue(
                issue_type=IntegrityIssueType.ARTIFACT_MISSING.value,
                severity=IntegrityIssueSeverity.ERROR.value,
                version=artifact_path.split('/')[0] if '/' in artifact_path else None,
                artifact=artifact_path,
                description=f"Artifact {artifact_path} is missing from repository",
                can_auto_repair=False
            )
            validation.issues.append(issue)
            return validation

        # Get size
        validation.size_bytes = self.adapter.get_artifact_size(artifact_path)

        # Compute hash
        validation.sha256_computed = self.adapter.compute_sha256(artifact_path)

        if expected_hash:
            validation.sha256_expected = expected_hash
            validation.sha256_matches = (validation.sha256_computed == expected_hash)

            if not validation.sha256_matches:
                issue = IntegrityIssue(
                    issue_type=IntegrityIssueType.ARTIFACT_CORRUPTED.value,
                    severity=IntegrityIssueSeverity.CRITICAL.value,
                    version=artifact_path.split('/')[0] if '/' in artifact_path else None,
                    artifact=artifact_path,
                    description=f"Artifact {artifact_path} hash mismatch",
                    expected_value=expected_hash,
                    actual_value=validation.sha256_computed,
                    can_auto_repair=False
                )
                validation.issues.append(issue)

        return validation

    def scan_manifest(self, version: str) -> Tuple[bool, List[IntegrityIssue], Optional[Dict[str, Any]]]:
        """
        Scan version manifest.

        Args:
            version: Version string

        Returns:
            (valid, issues, manifest_data)
        """
        issues = []
        manifest = self.adapter.get_manifest(version)

        if not manifest:
            issues.append(IntegrityIssue(
                issue_type=IntegrityIssueType.MANIFEST_MISSING.value,
                severity=IntegrityIssueSeverity.ERROR.value,
                version=version,
                artifact=f"{version}/manifest.json",
                description=f"Manifest missing for version {version}",
                can_auto_repair=False
            ))
            return False, issues, None

        # Validate manifest structure
        required_fields = ['version', 'timestamp', 'artifacts']
        for field in required_fields:
            if field not in manifest:
                issues.append(IntegrityIssue(
                    issue_type=IntegrityIssueType.MANIFEST_MALFORMED.value,
                    severity=IntegrityIssueSeverity.ERROR.value,
                    version=version,
                    artifact=f"{version}/manifest.json",
                    description=f"Manifest missing required field: {field}",
                    can_auto_repair=False
                ))

        return len(issues) == 0, issues, manifest

    def scan_sbom_slsa(self, version: str) -> Tuple[bool, bool, List[IntegrityIssue]]:
        """
        Scan SBOM and SLSA provenance.

        Args:
            version: Version string

        Returns:
            (sbom_valid, slsa_valid, issues)
        """
        issues = []
        sbom_valid = True
        slsa_valid = True

        # Check SBOM
        sbom = self.adapter.get_sbom(version)
        if not sbom:
            issues.append(IntegrityIssue(
                issue_type=IntegrityIssueType.SBOM_MISSING.value,
                severity=IntegrityIssueSeverity.ERROR.value,
                version=version,
                artifact=f"{version}/sbom.json",
                description=f"SBOM missing for version {version}",
                can_auto_repair=False
            ))
            sbom_valid = False
        else:
            # Validate SBOM structure
            if 'bomFormat' not in sbom or 'specVersion' not in sbom:
                issues.append(IntegrityIssue(
                    issue_type=IntegrityIssueType.SBOM_MALFORMED.value,
                    severity=IntegrityIssueSeverity.WARNING.value,
                    version=version,
                    artifact=f"{version}/sbom.json",
                    description=f"SBOM malformed for version {version}",
                    can_auto_repair=False
                ))
                sbom_valid = False

        # Check SLSA
        slsa = self.adapter.get_slsa(version)
        if not slsa:
            issues.append(IntegrityIssue(
                issue_type=IntegrityIssueType.SLSA_MISSING.value,
                severity=IntegrityIssueSeverity.ERROR.value,
                version=version,
                artifact=f"{version}/slsa-provenance.json",
                description=f"SLSA provenance missing for version {version}",
                can_auto_repair=False
            ))
            slsa_valid = False
        else:
            # Validate SLSA structure
            if 'predicate' not in slsa or 'subject' not in slsa:
                issues.append(IntegrityIssue(
                    issue_type=IntegrityIssueType.SLSA_MALFORMED.value,
                    severity=IntegrityIssueSeverity.WARNING.value,
                    version=version,
                    artifact=f"{version}/slsa-provenance.json",
                    description=f"SLSA provenance malformed for version {version}",
                    can_auto_repair=False
                ))
                slsa_valid = False

        return sbom_valid, slsa_valid, issues

    def scan_version(self, version: str) -> IntegrityVersionValidation:
        """
        Scan complete version.

        Args:
            version: Version string

        Returns:
            Version validation result
        """
        logger.info(f"Scanning version: {version}")

        validation = IntegrityVersionValidation(
            version=version,
            in_index=False,
            in_repository=True,
            manifest_valid=False,
            sbom_valid=False,
            slsa_valid=False
        )

        # Scan manifest
        manifest_valid, manifest_issues, manifest = self.scan_manifest(version)
        validation.manifest_valid = manifest_valid
        validation.issues.extend(manifest_issues)

        # Scan SBOM/SLSA
        sbom_valid, slsa_valid, sbom_slsa_issues = self.scan_sbom_slsa(version)
        validation.sbom_valid = sbom_valid
        validation.slsa_valid = slsa_valid
        validation.issues.extend(sbom_slsa_issues)

        # Get actual artifacts
        actual_artifacts = self.adapter.get_version_artifacts(version)
        validation.total_artifacts = len(actual_artifacts)

        # Cross-validate with manifest
        if manifest and 'artifacts' in manifest:
            manifest_artifacts = manifest['artifacts']

            # Check each manifest artifact exists and matches hash
            for artifact_entry in manifest_artifacts:
                if isinstance(artifact_entry, dict):
                    artifact_path = artifact_entry.get('path', '')
                    expected_hash = artifact_entry.get('sha256', '')
                else:
                    artifact_path = str(artifact_entry)
                    expected_hash = None

                artifact_validation = self.scan_artifact(artifact_path, expected_hash)
                validation.artifacts.append(artifact_validation)

                if not artifact_validation.exists:
                    validation.missing_artifacts += 1
                elif not artifact_validation.sha256_matches and expected_hash:
                    validation.corrupted_artifacts += 1

                validation.total_size_bytes += artifact_validation.size_bytes
                validation.issues.extend(artifact_validation.issues)

        return validation

    def scan_index_consistency(
        self,
        index_data: Optional[Dict[str, Any]],
        repo_versions: List[str]
    ) -> List[IntegrityIssue]:
        """
        Scan index consistency.

        Args:
            index_data: Index data
            repo_versions: Versions in repository

        Returns:
            List of issues
        """
        issues = []

        if not index_data:
            issues.append(IntegrityIssue(
                issue_type=IntegrityIssueType.INDEX_MISSING.value,
                severity=IntegrityIssueSeverity.ERROR.value,
                version=None,
                artifact="index.json",
                description="Repository index.json is missing",
                repair_action=IntegrityRepairAction.REBUILD_INDEX_JSON.value,
                can_auto_repair=True
            ))
            return issues

        # Validate index structure
        if 'releases' not in index_data:
            issues.append(IntegrityIssue(
                issue_type=IntegrityIssueType.INDEX_MALFORMED.value,
                severity=IntegrityIssueSeverity.ERROR.value,
                version=None,
                artifact="index.json",
                description="Index.json missing 'releases' field",
                repair_action=IntegrityRepairAction.REBUILD_INDEX_JSON.value,
                can_auto_repair=True
            ))
            return issues

        index_versions = set(index_data['releases'].keys())
        repo_versions_set = set(repo_versions)

        # Versions in index but not in repo
        for version in index_versions - repo_versions_set:
            issues.append(IntegrityIssue(
                issue_type=IntegrityIssueType.INDEX_VERSION_EXTRA.value,
                severity=IntegrityIssueSeverity.ERROR.value,
                version=version,
                artifact="index.json",
                description=f"Version {version} in index but missing from repository",
                repair_action=IntegrityRepairAction.FIX_INDEX_ENTRY.value,
                can_auto_repair=True
            ))

        # Versions in repo but not in index
        for version in repo_versions_set - index_versions:
            issues.append(IntegrityIssue(
                issue_type=IntegrityIssueType.INDEX_VERSION_MISSING.value,
                severity=IntegrityIssueSeverity.ERROR.value,
                version=version,
                artifact="index.json",
                description=f"Version {version} in repository but missing from index",
                repair_action=IntegrityRepairAction.FIX_INDEX_ENTRY.value,
                can_auto_repair=True
            ))

        return issues

    def detect_orphans(
        self,
        all_artifacts: List[str],
        known_artifacts: Set[str]
    ) -> List[IntegrityIssue]:
        """
        Detect orphaned artifacts.

        Args:
            all_artifacts: All artifacts in repository
            known_artifacts: Known artifacts from versions

        Returns:
            List of orphan issues
        """
        issues = []

        for artifact in all_artifacts:
            if artifact not in known_artifacts:
                issues.append(IntegrityIssue(
                    issue_type=IntegrityIssueType.ARTIFACT_ORPHANED.value,
                    severity=IntegrityIssueSeverity.WARNING.value,
                    version=None,
                    artifact=artifact,
                    description=f"Orphan artifact detected: {artifact}",
                    repair_action=IntegrityRepairAction.REMOVE_ORPHAN.value,
                    can_auto_repair=True
                ))

        return issues


# ============================================================================
# INTEGRITY REPAIR ENGINE
# ============================================================================

class IntegrityRepairEngine:
    """
    Safe-mode repair engine.
    """

    def __init__(self, adapter: IntegrityRepositoryAdapter):
        """
        Initialize repair engine.

        Args:
            adapter: Repository adapter
        """
        self.adapter = adapter
        logger.info("IntegrityRepairEngine initialized")

    def repair_remove_orphan(self, artifact_path: str) -> IntegrityRepairResult:
        """
        Remove orphan artifact.

        Args:
            artifact_path: Artifact to remove

        Returns:
            Repair result
        """
        try:
            success = self.adapter.delete_artifact(artifact_path)
            return IntegrityRepairResult(
                action=IntegrityRepairAction.REMOVE_ORPHAN.value,
                success=success,
                artifact=artifact_path,
                description=f"Removed orphan artifact: {artifact_path}" if success else "Failed to remove orphan"
            )
        except Exception as e:
            return IntegrityRepairResult(
                action=IntegrityRepairAction.REMOVE_ORPHAN.value,
                success=False,
                artifact=artifact_path,
                description=f"Failed to remove orphan: {artifact_path}",
                error=str(e)
            )

    def repair_rebuild_index_json(
        self,
        versions: List[IntegrityVersionValidation]
    ) -> IntegrityRepairResult:
        """
        Rebuild index.json from repository state.

        Args:
            versions: Version validations

        Returns:
            Repair result
        """
        try:
            releases = {}

            for version_validation in versions:
                if version_validation.in_repository:
                    releases[version_validation.version] = {
                        "version": version_validation.version,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "artifacts": version_validation.total_artifacts,
                        "size_bytes": version_validation.total_size_bytes,
                        "manifest_valid": version_validation.manifest_valid,
                        "sbom_valid": version_validation.sbom_valid,
                        "slsa_valid": version_validation.slsa_valid
                    }

            index_data = {
                "format_version": "1.0",
                "repository": "T.A.R.S. Release Repository",
                "generated": datetime.now(timezone.utc).isoformat(),
                "total_releases": len(releases),
                "releases": releases
            }

            success = self.adapter.update_index(index_data)

            return IntegrityRepairResult(
                action=IntegrityRepairAction.REBUILD_INDEX_JSON.value,
                success=success,
                description=f"Rebuilt index.json with {len(releases)} releases" if success else "Failed to rebuild index"
            )
        except Exception as e:
            return IntegrityRepairResult(
                action=IntegrityRepairAction.REBUILD_INDEX_JSON.value,
                success=False,
                description="Failed to rebuild index.json",
                error=str(e)
            )

    def repair_fix_index_entry(
        self,
        version: str,
        action: str
    ) -> IntegrityRepairResult:
        """
        Fix single index entry.

        Args:
            version: Version to fix
            action: 'add' or 'remove'

        Returns:
            Repair result
        """
        try:
            index_data = self.adapter.get_index()
            if not index_data:
                return IntegrityRepairResult(
                    action=IntegrityRepairAction.FIX_INDEX_ENTRY.value,
                    success=False,
                    version=version,
                    description="Cannot fix index entry: index.json missing"
                )

            if action == 'remove':
                if 'releases' in index_data and version in index_data['releases']:
                    del index_data['releases'][version]
                    index_data['total_releases'] = len(index_data['releases'])

            elif action == 'add':
                manifest = self.adapter.get_manifest(version)
                if manifest:
                    if 'releases' not in index_data:
                        index_data['releases'] = {}

                    index_data['releases'][version] = {
                        "version": version,
                        "timestamp": manifest.get('timestamp', datetime.now(timezone.utc).isoformat()),
                        "artifacts": len(manifest.get('artifacts', [])),
                    }
                    index_data['total_releases'] = len(index_data['releases'])

            success = self.adapter.update_index(index_data)

            return IntegrityRepairResult(
                action=IntegrityRepairAction.FIX_INDEX_ENTRY.value,
                success=success,
                version=version,
                description=f"Fixed index entry for {version} ({action})" if success else f"Failed to {action} index entry"
            )
        except Exception as e:
            return IntegrityRepairResult(
                action=IntegrityRepairAction.FIX_INDEX_ENTRY.value,
                success=False,
                version=version,
                description=f"Failed to fix index entry for {version}",
                error=str(e)
            )


# ============================================================================
# INTEGRITY REPORT BUILDER
# ============================================================================

class IntegrityReportBuilder:
    """
    Builds comprehensive integrity reports.
    """

    @staticmethod
    def build_json_report(report: IntegrityScanReport) -> str:
        """
        Build JSON report.

        Args:
            report: Scan report

        Returns:
            JSON string
        """
        return json.dumps(asdict(report), indent=2)

    @staticmethod
    def build_text_report(report: IntegrityScanReport) -> str:
        """
        Build human-readable text report.

        Args:
            report: Scan report

        Returns:
            Text report
        """
        lines = []
        lines.append("=" * 80)
        lines.append("T.A.R.S. REPOSITORY INTEGRITY SCAN REPORT")
        lines.append("=" * 80)
        lines.append(f"Scan ID: {report.scan_id}")
        lines.append(f"Timestamp: {report.timestamp}")
        lines.append(f"Repository: {report.repository_type} ({report.repository_location})")
        lines.append(f"Policy: {report.policy_mode}")
        lines.append(f"Status: {report.scan_status}")
        lines.append("")

        lines.append("REPOSITORY HEALTH")
        lines.append("-" * 80)
        lines.append(f"Total Versions: {report.total_versions}")
        lines.append(f"Total Artifacts: {report.total_artifacts}")
        lines.append(f"Total Size: {report.total_size_bytes:,} bytes ({report.total_size_bytes / 1024 / 1024:.2f} MB)")
        lines.append(f"Orphan Artifacts: {report.orphan_artifacts}")
        lines.append("")

        lines.append("ISSUES SUMMARY")
        lines.append("-" * 80)
        lines.append(f"Total Issues: {report.total_issues}")
        lines.append(f"  Critical: {report.critical_issues}")
        lines.append(f"  Errors: {report.error_issues}")
        lines.append(f"  Warnings: {report.warning_issues}")
        lines.append(f"  Info: {report.info_issues}")
        lines.append("")

        lines.append("ISSUE BREAKDOWN")
        lines.append("-" * 80)
        lines.append(f"Corrupted Artifacts: {report.corrupted_artifacts}")
        lines.append(f"Missing Artifacts: {report.missing_artifacts}")
        lines.append(f"Manifest Issues: {report.manifest_issues}")
        lines.append(f"Index Issues: {report.index_issues}")
        lines.append(f"SBOM/SLSA Issues: {report.sbom_slsa_issues}")
        lines.append(f"Signature Issues: {report.signature_issues}")
        lines.append("")

        if report.all_issues:
            lines.append("DETAILED ISSUES")
            lines.append("-" * 80)
            for i, issue in enumerate(report.all_issues[:50], 1):  # Limit to first 50
                lines.append(f"{i}. [{issue.severity.upper()}] {issue.description}")
                if issue.artifact:
                    lines.append(f"   Artifact: {issue.artifact}")
                if issue.expected_value and issue.actual_value:
                    lines.append(f"   Expected: {issue.expected_value[:64]}")
                    lines.append(f"   Actual: {issue.actual_value[:64]}")
                if issue.can_auto_repair and issue.repair_action:
                    lines.append(f"   Repair: {issue.repair_action}")
                lines.append("")

            if len(report.all_issues) > 50:
                lines.append(f"... and {len(report.all_issues) - 50} more issues (see JSON report)")
                lines.append("")

        if report.repair_enabled:
            lines.append("REPAIRS")
            lines.append("-" * 80)
            lines.append(f"Repairs Applied: {report.repairs_applied}")
            lines.append(f"Repairs Failed: {report.repairs_failed}")
            lines.append("")

        lines.append("PERFORMANCE")
        lines.append("-" * 80)
        lines.append(f"Scan Duration: {report.scan_duration_seconds:.2f} seconds")
        lines.append("")

        lines.append("SUMMARY")
        lines.append("-" * 80)
        lines.append(report.summary)
        lines.append("")
        lines.append("=" * 80)

        return "\n".join(lines)


# ============================================================================
# INTEGRITY SCAN ORCHESTRATOR
# ============================================================================

class IntegrityScanOrchestrator:
    """
    Top-level orchestrator for integrity scans.
    """

    def __init__(
        self,
        repository,
        policy_mode: IntegrityScanPolicy = IntegrityScanPolicy.STRICT,
        repair_enabled: bool = False,
        repair_orphans: bool = False,
        repair_index: bool = False
    ):
        """
        Initialize orchestrator.

        Args:
            repository: AbstractRepository instance
            policy_mode: Policy mode
            repair_enabled: Enable repairs
            repair_orphans: Enable orphan removal
            repair_index: Enable index rebuilding
        """
        self.repository = repository
        self.adapter = IntegrityRepositoryAdapter(repository)
        self.policy_engine = IntegrityScanPolicyEngine(policy_mode)
        self.scanner = IntegrityScanner(self.adapter, self.policy_engine)
        self.repair_engine = IntegrityRepairEngine(self.adapter)

        self.policy_mode = policy_mode
        self.repair_enabled = repair_enabled
        self.repair_orphans = repair_orphans
        self.repair_index = repair_index

        logger.info(f"IntegrityScanOrchestrator initialized (policy={policy_mode.value}, repair={repair_enabled})")

    def scan_repository(
        self,
        output_dir: Path,
        json_report_path: Optional[Path] = None,
        text_report_path: Optional[Path] = None
    ) -> IntegrityScanReport:
        """
        Scan entire repository.

        Args:
            output_dir: Output directory
            json_report_path: Optional JSON report path
            text_report_path: Optional text report path

        Returns:
            Scan report
        """
        start_time = time.time()

        logger.info("Starting repository integrity scan")

        # Create report
        report = IntegrityScanReport(
            scan_id=str(uuid.uuid4())[:8],
            timestamp=datetime.now(timezone.utc).isoformat(),
            repository_type=self.repository.repo_type,
            repository_location=str(getattr(self.repository, 'base_path', 'unknown')),
            policy_mode=self.policy_mode.value,
            scan_status=IntegrityScanStatus.SUCCESS.value,
            repair_enabled=self.repair_enabled
        )

        # Get index
        index_data = self.adapter.get_index()

        # Get all versions
        repo_versions = self.adapter.list_all_versions()
        report.total_versions = len(repo_versions)

        logger.info(f"Found {len(repo_versions)} versions in repository")

        # Scan each version
        all_known_artifacts = set()

        for version in repo_versions:
            version_validation = self.scanner.scan_version(version)

            # Mark if in index
            if index_data and 'releases' in index_data:
                version_validation.in_index = version in index_data['releases']

            report.versions.append(version_validation)
            report.total_artifacts += version_validation.total_artifacts
            report.total_size_bytes += version_validation.total_size_bytes
            report.corrupted_artifacts += version_validation.corrupted_artifacts
            report.missing_artifacts += version_validation.missing_artifacts
            report.all_issues.extend(version_validation.issues)

            # Track known artifacts
            for artifact in self.adapter.get_version_artifacts(version):
                all_known_artifacts.add(artifact)

        # Scan index consistency
        index_issues = self.scanner.scan_index_consistency(index_data, repo_versions)
        report.all_issues.extend(index_issues)
        report.index_issues = len(index_issues)

        # Detect orphans
        all_artifacts = self.adapter.list_all_artifacts()
        orphan_issues = self.scanner.detect_orphans(all_artifacts, all_known_artifacts)
        report.all_issues.extend(orphan_issues)
        report.orphan_artifacts = len(orphan_issues)

        # Count issues by severity
        for issue in report.all_issues:
            severity = IntegrityIssueSeverity(issue.severity)
            if severity == IntegrityIssueSeverity.CRITICAL:
                report.critical_issues += 1
            elif severity == IntegrityIssueSeverity.ERROR:
                report.error_issues += 1
            elif severity == IntegrityIssueSeverity.WARNING:
                report.warning_issues += 1
            else:
                report.info_issues += 1

            # Count by type
            if 'manifest' in issue.issue_type:
                report.manifest_issues += 1
            if issue.issue_type in [IntegrityIssueType.SBOM_MISSING.value, IntegrityIssueType.SLSA_MISSING.value]:
                report.sbom_slsa_issues += 1
            if 'signature' in issue.issue_type:
                report.signature_issues += 1

        report.total_issues = len(report.all_issues)

        # Apply repairs if enabled
        if self.repair_enabled and report.total_issues > 0:
            logger.info("Applying repairs...")
            repair_results = self._apply_repairs(report)

            for result in repair_results:
                if result.success:
                    report.repairs_applied += 1
                else:
                    report.repairs_failed += 1

        # Determine status and exit code
        if report.total_issues == 0:
            report.scan_status = IntegrityScanStatus.SUCCESS.value
            report.exit_code = 50
            report.summary = "Repository integrity scan passed with no issues"
        elif report.critical_issues > 0 or report.error_issues > 0:
            if self.policy_engine.should_fail_on_issue(IntegrityIssueSeverity.ERROR):
                report.scan_status = IntegrityScanStatus.FAILED.value
            else:
                report.scan_status = IntegrityScanStatus.SUCCESS_WITH_WARNINGS.value
            report.exit_code = self.policy_engine.determine_exit_code(report.all_issues)
            report.summary = f"Repository integrity scan found {report.total_issues} issues ({report.critical_issues} critical, {report.error_issues} errors)"
        else:
            report.scan_status = IntegrityScanStatus.SUCCESS_WITH_WARNINGS.value
            report.exit_code = 51
            report.summary = f"Repository integrity scan completed with {report.warning_issues} warnings"

        # Performance
        report.scan_duration_seconds = time.time() - start_time

        # Write reports
        output_dir.mkdir(parents=True, exist_ok=True)

        if json_report_path:
            json_content = IntegrityReportBuilder.build_json_report(report)
            json_report_path.write_text(json_content)
            logger.info(f"JSON report written to {json_report_path}")

        if text_report_path:
            text_content = IntegrityReportBuilder.build_text_report(report)
            text_report_path.write_text(text_content)
            logger.info(f"Text report written to {text_report_path}")

        logger.info(f"Integrity scan complete: {report.scan_status} (exit code: {report.exit_code})")

        return report

    def _apply_repairs(self, report: IntegrityScanReport) -> List[IntegrityRepairResult]:
        """
        Apply repairs based on issues.

        Args:
            report: Scan report

        Returns:
            List of repair results
        """
        results = []

        # Repair orphans if enabled
        if self.repair_orphans:
            orphan_issues = [i for i in report.all_issues if i.issue_type == IntegrityIssueType.ARTIFACT_ORPHANED.value]
            for issue in orphan_issues:
                if issue.artifact:
                    result = self.repair_engine.repair_remove_orphan(issue.artifact)
                    results.append(result)

        # Repair index if enabled
        if self.repair_index:
            index_issues = [i for i in report.all_issues if 'index' in i.issue_type]

            # Rebuild entire index if missing or malformed
            rebuild_full = any(i.issue_type in [
                IntegrityIssueType.INDEX_MISSING.value,
                IntegrityIssueType.INDEX_MALFORMED.value
            ] for i in index_issues)

            if rebuild_full:
                result = self.repair_engine.repair_rebuild_index_json(report.versions)
                results.append(result)
            else:
                # Fix individual entries
                for issue in index_issues:
                    if issue.issue_type == IntegrityIssueType.INDEX_VERSION_EXTRA.value and issue.version:
                        result = self.repair_engine.repair_fix_index_entry(issue.version, 'remove')
                        results.append(result)
                    elif issue.issue_type == IntegrityIssueType.INDEX_VERSION_MISSING.value and issue.version:
                        result = self.repair_engine.repair_fix_index_entry(issue.version, 'add')
                        results.append(result)

        return results


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

def main():
    """CLI entry point for standalone integrity scanner."""
    import argparse
    import sys

    # Import publisher for repository factory
    try:
        from publisher.release_publisher import RepositoryFactory
    except ImportError:
        logger.error("Failed to import RepositoryFactory from publisher")
        sys.exit(59)

    parser = argparse.ArgumentParser(
        description="T.A.R.S. Repository Integrity Scanner",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Repository configuration
    parser.add_argument('--repository-type', choices=['local', 's3', 'gcs'], default='local',
                        help='Repository type')
    parser.add_argument('--repository-path', type=str, default='./repository',
                        help='Local repository path')
    parser.add_argument('--repository-bucket', type=str, default='default-bucket',
                        help='S3/GCS bucket name')
    parser.add_argument('--repository-prefix', type=str, default='',
                        help='S3/GCS prefix')

    # Scan configuration
    parser.add_argument('--policy', choices=['strict', 'lenient', 'audit_only'], default='strict',
                        help='Scan policy mode')
    parser.add_argument('--repair', action='store_true',
                        help='Enable repair mode')
    parser.add_argument('--repair-orphans', action='store_true',
                        help='Enable orphan artifact removal')
    parser.add_argument('--repair-index', action='store_true',
                        help='Enable index rebuilding')

    # Output configuration
    parser.add_argument('--output-dir', type=str, default='./integrity-scan',
                        help='Output directory for reports')
    parser.add_argument('--json-report', type=str,
                        help='JSON report path')
    parser.add_argument('--text-report', type=str,
                        help='Text report path')

    # Other
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose output')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create repository
    repo_config = {
        "type": args.repository_type,
        "path": args.repository_path,
        "bucket": args.repository_bucket,
        "prefix": args.repository_prefix
    }

    try:
        repository = RepositoryFactory.create(args.repository_type, repo_config)
    except Exception as e:
        logger.error(f"Failed to create repository: {e}")
        sys.exit(59)

    # Create orchestrator
    policy_mode = IntegrityScanPolicy(args.policy)
    orchestrator = IntegrityScanOrchestrator(
        repository=repository,
        policy_mode=policy_mode,
        repair_enabled=args.repair,
        repair_orphans=args.repair_orphans,
        repair_index=args.repair_index
    )

    # Run scan
    output_dir = Path(args.output_dir)
    json_report_path = Path(args.json_report) if args.json_report else output_dir / "integrity-scan-report.json"
    text_report_path = Path(args.text_report) if args.text_report else output_dir / "integrity-scan-report.txt"

    try:
        report = orchestrator.scan_repository(
            output_dir=output_dir,
            json_report_path=json_report_path,
            text_report_path=text_report_path
        )

        # Print summary
        print("\n" + IntegrityReportBuilder.build_text_report(report))

        sys.exit(report.exit_code)

    except Exception as e:
        logger.error(f"Integrity scan failed: {e}", exc_info=True)
        sys.exit(59)


if __name__ == '__main__':
    main()
