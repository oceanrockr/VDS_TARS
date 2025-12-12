#!/usr/bin/env python3
"""
Release Publisher - Phase 14.7 Task 5

Automated release publication system with repository abstraction, policy enforcement,
atomic publishing, versioning, audit logging, and comprehensive reporting.

Author: T.A.R.S. Development Team
Version: 1.0.0
Date: 2025-11-28
"""

import json
import hashlib
import shutil
import uuid
import platform
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
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
# CUSTOM EXCEPTIONS (Exit Codes 30-39)
# ============================================================================

class PublicationError(Exception):
    """Base exception for all publication errors."""
    exit_code = 39

    def __init__(self, message: str, exit_code: Optional[int] = None):
        super().__init__(message)
        if exit_code is not None:
            self.exit_code = exit_code


class VerificationRequiredError(PublicationError):
    """Task 3 verification required but not passed."""
    exit_code = 30


class ValidationRequiredError(PublicationError):
    """Task 4 validation required but not passed."""
    exit_code = 31


class DuplicateVersionError(PublicationError):
    """Attempt to republish existing version."""
    exit_code = 32


class MetadataMissingError(PublicationError):
    """Required metadata files missing."""
    exit_code = 33


class SignatureRequiredError(PublicationError):
    """Signed artifacts required but not present."""
    exit_code = 34


class EncryptionRequiredError(PublicationError):
    """Encrypted artifacts required but not present."""
    exit_code = 35


class RepositoryError(PublicationError):
    """Repository operation failed."""
    exit_code = 36


class AtomicPublishError(PublicationError):
    """Atomic publish operation failed."""
    exit_code = 37


class PolicyViolationError(PublicationError):
    """Publication policy violation."""
    exit_code = 38


# ============================================================================
# ENUMS
# ============================================================================

class PublicationPolicy(Enum):
    """Publication policy modes."""
    STRICT = "strict"
    LENIENT = "lenient"


class RepositoryType(Enum):
    """Supported repository types."""
    LOCAL = "local"
    S3 = "s3"
    GCS = "gcs"


class PublicationStatus(Enum):
    """Publication operation status."""
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class VerificationResult:
    """Task 3 verification result."""
    passed: bool
    timestamp: str
    hash_verified: bool = False
    signature_verified: bool = False
    sbom_validated: bool = False
    slsa_validated: bool = False
    policy_passed: bool = False
    exit_code: int = 0


@dataclass
class ValidationResult:
    """Task 4 validation result."""
    passed: bool
    timestamp: str
    sbom_delta_passed: bool = False
    slsa_delta_passed: bool = False
    api_compat_passed: bool = False
    performance_passed: bool = False
    security_passed: bool = False
    behavioral_passed: bool = False
    exit_code: int = 0


@dataclass
class ReleaseMetadata:
    """Release metadata for indexing."""
    version: str
    timestamp: str
    sha256: str
    size_bytes: int
    verification_result: Optional[Dict[str, Any]] = None
    validation_result: Optional[Dict[str, Any]] = None
    sbom_id: Optional[str] = None
    slsa_id: Optional[str] = None
    provenance_hash: Optional[str] = None
    signed: bool = False
    encrypted: bool = False
    artifacts: List[str] = field(default_factory=list)


@dataclass
class PublicationReport:
    """Comprehensive publication report."""
    version: str
    status: str
    timestamp: str
    repository_type: str
    repository_location: str
    policy_mode: str

    # Pre-flight checks
    verification_passed: bool = False
    validation_passed: bool = False
    metadata_complete: bool = False
    signatures_present: bool = False

    # Publication details
    artifacts_published: List[str] = field(default_factory=list)
    total_size_bytes: int = 0
    publication_duration_seconds: float = 0.0

    # Post-publication
    index_updated: bool = False
    audit_log_created: bool = False
    audit_log_signed: bool = False

    # Issues/warnings
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    # Exit
    exit_code: int = 0
    summary: str = ""


# ============================================================================
# REPOSITORY ABSTRACTION LAYER
# ============================================================================

class AbstractRepository(ABC):
    """
    Abstract base class for repository adapters.
    Defines interface for artifact storage and retrieval.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.repo_type = config.get("type", "unknown")
        logger.info(f"Initializing {self.__class__.__name__} with config: {config}")

    @abstractmethod
    def exists(self, path: str) -> bool:
        """Check if artifact exists at path."""
        pass

    @abstractmethod
    def upload(self, local_path: Path, remote_path: str) -> bool:
        """Upload artifact from local path to remote path."""
        pass

    @abstractmethod
    def download(self, remote_path: str, local_path: Path) -> bool:
        """Download artifact from remote path to local path."""
        pass

    @abstractmethod
    def list_versions(self) -> List[str]:
        """List all published versions."""
        pass

    @abstractmethod
    def get_index(self) -> Optional[Dict[str, Any]]:
        """Retrieve release index."""
        pass

    @abstractmethod
    def update_index(self, index_data: Dict[str, Any]) -> bool:
        """Update release index."""
        pass

    @abstractmethod
    def delete(self, path: str) -> bool:
        """Delete artifact (for rollback/cleanup)."""
        pass

    def validate_config(self) -> bool:
        """Validate repository configuration."""
        return True


class LocalRepository(AbstractRepository):
    """
    Local filesystem repository adapter.
    Stores artifacts in a local directory structure.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.base_path = Path(config.get("path", "./repository"))
        self.base_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"LocalRepository initialized at {self.base_path}")

    def exists(self, path: str) -> bool:
        """Check if artifact exists."""
        full_path = self.base_path / path
        exists = full_path.exists()
        logger.debug(f"Checking existence of {path}: {exists}")
        return exists

    def upload(self, local_path: Path, remote_path: str) -> bool:
        """Upload (copy) file to repository."""
        try:
            target = self.base_path / remote_path
            target.parent.mkdir(parents=True, exist_ok=True)

            if target.exists():
                logger.warning(f"Target {remote_path} already exists, skipping")
                return False

            shutil.copy2(local_path, target)
            logger.info(f"Uploaded {local_path} → {remote_path}")
            return True
        except Exception as e:
            logger.error(f"Upload failed: {e}")
            return False

    def download(self, remote_path: str, local_path: Path) -> bool:
        """Download (copy) file from repository."""
        try:
            source = self.base_path / remote_path
            if not source.exists():
                logger.error(f"Source {remote_path} not found")
                return False

            local_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, local_path)
            logger.info(f"Downloaded {remote_path} → {local_path}")
            return True
        except Exception as e:
            logger.error(f"Download failed: {e}")
            return False

    def list_versions(self) -> List[str]:
        """List all version directories."""
        try:
            versions = [d.name for d in self.base_path.iterdir() if d.is_dir() and d.name.startswith("v")]
            versions.sort(reverse=True)
            logger.debug(f"Found {len(versions)} versions")
            return versions
        except Exception as e:
            logger.error(f"Failed to list versions: {e}")
            return []

    def get_index(self) -> Optional[Dict[str, Any]]:
        """Retrieve index.json."""
        try:
            index_file = self.base_path / "index.json"
            if not index_file.exists():
                logger.warning("index.json not found, returning empty index")
                return {"releases": [], "last_updated": datetime.now(timezone.utc).isoformat()}

            with open(index_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to read index: {e}")
            return None

    def update_index(self, index_data: Dict[str, Any]) -> bool:
        """Update index.json."""
        try:
            index_file = self.base_path / "index.json"
            with open(index_file, 'w', encoding='utf-8') as f:
                json.dump(index_data, f, indent=2, sort_keys=True)
            logger.info("Index updated successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to update index: {e}")
            return False

    def delete(self, path: str) -> bool:
        """Delete artifact."""
        try:
            target = self.base_path / path
            if target.is_file():
                target.unlink()
            elif target.is_dir():
                shutil.rmtree(target)
            logger.info(f"Deleted {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete {path}: {e}")
            return False


class S3StyleRepository(AbstractRepository):
    """
    S3-style repository adapter (simulated, no network calls).
    Implements S3-like semantics using local storage.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.bucket = config.get("bucket", "default-bucket")
        self.prefix = config.get("prefix", "")
        # Simulate S3 with local directory
        self.local_base = Path(config.get("local_base", "./s3-simulation")) / self.bucket
        self.local_base.mkdir(parents=True, exist_ok=True)
        logger.info(f"S3StyleRepository initialized: bucket={self.bucket}, prefix={self.prefix}")

    def _get_key(self, path: str) -> str:
        """Construct S3 key."""
        if self.prefix:
            return f"{self.prefix}/{path}"
        return path

    def exists(self, path: str) -> bool:
        """Check if object exists."""
        key = self._get_key(path)
        full_path = self.local_base / key
        exists = full_path.exists()
        logger.debug(f"S3: Checking s3://{self.bucket}/{key}: {exists}")
        return exists

    def upload(self, local_path: Path, remote_path: str) -> bool:
        """Upload object to S3 (simulated)."""
        try:
            key = self._get_key(remote_path)
            target = self.local_base / key
            target.parent.mkdir(parents=True, exist_ok=True)

            if target.exists():
                logger.warning(f"S3: Object s3://{self.bucket}/{key} already exists")
                return False

            shutil.copy2(local_path, target)
            logger.info(f"S3: Uploaded {local_path} → s3://{self.bucket}/{key}")
            return True
        except Exception as e:
            logger.error(f"S3: Upload failed: {e}")
            return False

    def download(self, remote_path: str, local_path: Path) -> bool:
        """Download object from S3 (simulated)."""
        try:
            key = self._get_key(remote_path)
            source = self.local_base / key
            if not source.exists():
                logger.error(f"S3: Object s3://{self.bucket}/{key} not found")
                return False

            local_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, local_path)
            logger.info(f"S3: Downloaded s3://{self.bucket}/{key} → {local_path}")
            return True
        except Exception as e:
            logger.error(f"S3: Download failed: {e}")
            return False

    def list_versions(self) -> List[str]:
        """List version prefixes."""
        try:
            prefix_path = self.local_base / self.prefix if self.prefix else self.local_base
            versions = [d.name for d in prefix_path.iterdir() if d.is_dir() and d.name.startswith("v")]
            versions.sort(reverse=True)
            logger.debug(f"S3: Found {len(versions)} versions")
            return versions
        except Exception as e:
            logger.error(f"S3: Failed to list versions: {e}")
            return []

    def get_index(self) -> Optional[Dict[str, Any]]:
        """Retrieve index.json from S3."""
        try:
            key = self._get_key("index.json")
            index_file = self.local_base / key
            if not index_file.exists():
                logger.warning("S3: index.json not found")
                return {"releases": [], "last_updated": datetime.now(timezone.utc).isoformat()}

            with open(index_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"S3: Failed to read index: {e}")
            return None

    def update_index(self, index_data: Dict[str, Any]) -> bool:
        """Update index.json in S3."""
        try:
            key = self._get_key("index.json")
            index_file = self.local_base / key
            index_file.parent.mkdir(parents=True, exist_ok=True)

            with open(index_file, 'w', encoding='utf-8') as f:
                json.dump(index_data, f, indent=2, sort_keys=True)
            logger.info("S3: Index updated successfully")
            return True
        except Exception as e:
            logger.error(f"S3: Failed to update index: {e}")
            return False

    def delete(self, path: str) -> bool:
        """Delete object from S3."""
        try:
            key = self._get_key(path)
            target = self.local_base / key
            if target.is_file():
                target.unlink()
            elif target.is_dir():
                shutil.rmtree(target)
            logger.info(f"S3: Deleted s3://{self.bucket}/{key}")
            return True
        except Exception as e:
            logger.error(f"S3: Failed to delete {key}: {e}")
            return False


class GCSStyleRepository(AbstractRepository):
    """
    GCS-style repository adapter (simulated, no network calls).
    Implements Google Cloud Storage-like semantics using local storage.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.bucket = config.get("bucket", "default-bucket")
        self.prefix = config.get("prefix", "")
        # Simulate GCS with local directory
        self.local_base = Path(config.get("local_base", "./gcs-simulation")) / self.bucket
        self.local_base.mkdir(parents=True, exist_ok=True)
        logger.info(f"GCSStyleRepository initialized: bucket={self.bucket}, prefix={self.prefix}")

    def _get_blob_name(self, path: str) -> str:
        """Construct GCS blob name."""
        if self.prefix:
            return f"{self.prefix}/{path}"
        return path

    def exists(self, path: str) -> bool:
        """Check if blob exists."""
        blob_name = self._get_blob_name(path)
        full_path = self.local_base / blob_name
        exists = full_path.exists()
        logger.debug(f"GCS: Checking gs://{self.bucket}/{blob_name}: {exists}")
        return exists

    def upload(self, local_path: Path, remote_path: str) -> bool:
        """Upload blob to GCS (simulated)."""
        try:
            blob_name = self._get_blob_name(remote_path)
            target = self.local_base / blob_name
            target.parent.mkdir(parents=True, exist_ok=True)

            if target.exists():
                logger.warning(f"GCS: Blob gs://{self.bucket}/{blob_name} already exists")
                return False

            shutil.copy2(local_path, target)
            logger.info(f"GCS: Uploaded {local_path} → gs://{self.bucket}/{blob_name}")
            return True
        except Exception as e:
            logger.error(f"GCS: Upload failed: {e}")
            return False

    def download(self, remote_path: str, local_path: Path) -> bool:
        """Download blob from GCS (simulated)."""
        try:
            blob_name = self._get_blob_name(remote_path)
            source = self.local_base / blob_name
            if not source.exists():
                logger.error(f"GCS: Blob gs://{self.bucket}/{blob_name} not found")
                return False

            local_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, local_path)
            logger.info(f"GCS: Downloaded gs://{self.bucket}/{blob_name} → {local_path}")
            return True
        except Exception as e:
            logger.error(f"GCS: Download failed: {e}")
            return False

    def list_versions(self) -> List[str]:
        """List version prefixes."""
        try:
            prefix_path = self.local_base / self.prefix if self.prefix else self.local_base
            versions = [d.name for d in prefix_path.iterdir() if d.is_dir() and d.name.startswith("v")]
            versions.sort(reverse=True)
            logger.debug(f"GCS: Found {len(versions)} versions")
            return versions
        except Exception as e:
            logger.error(f"GCS: Failed to list versions: {e}")
            return []

    def get_index(self) -> Optional[Dict[str, Any]]:
        """Retrieve index.json from GCS."""
        try:
            blob_name = self._get_blob_name("index.json")
            index_file = self.local_base / blob_name
            if not index_file.exists():
                logger.warning("GCS: index.json not found")
                return {"releases": [], "last_updated": datetime.now(timezone.utc).isoformat()}

            with open(index_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"GCS: Failed to read index: {e}")
            return None

    def update_index(self, index_data: Dict[str, Any]) -> bool:
        """Update index.json in GCS."""
        try:
            blob_name = self._get_blob_name("index.json")
            index_file = self.local_base / blob_name
            index_file.parent.mkdir(parents=True, exist_ok=True)

            with open(index_file, 'w', encoding='utf-8') as f:
                json.dump(index_data, f, indent=2, sort_keys=True)
            logger.info("GCS: Index updated successfully")
            return True
        except Exception as e:
            logger.error(f"GCS: Failed to update index: {e}")
            return False

    def delete(self, path: str) -> bool:
        """Delete blob from GCS."""
        try:
            blob_name = self._get_blob_name(path)
            target = self.local_base / blob_name
            if target.is_file():
                target.unlink()
            elif target.is_dir():
                shutil.rmtree(target)
            logger.info(f"GCS: Deleted gs://{self.bucket}/{blob_name}")
            return True
        except Exception as e:
            logger.error(f"GCS: Failed to delete {blob_name}: {e}")
            return False


# ============================================================================
# REPOSITORY FACTORY
# ============================================================================

class RepositoryFactory:
    """Factory for creating repository adapters."""

    @staticmethod
    def create(repo_type: str, config: Dict[str, Any]) -> AbstractRepository:
        """Create repository adapter based on type."""
        repo_type = repo_type.lower()

        if repo_type == "local":
            return LocalRepository(config)
        elif repo_type == "s3":
            return S3StyleRepository(config)
        elif repo_type == "gcs":
            return GCSStyleRepository(config)
        else:
            raise ValueError(f"Unsupported repository type: {repo_type}")


# ============================================================================
# PUBLICATION POLICY ENGINE
# ============================================================================

class PublicationPolicyEngine:
    """
    Enforces publication policies before release.
    Validates Task 3/4 results, metadata, signatures, encryption.
    """

    def __init__(self, mode: PublicationPolicy = PublicationPolicy.STRICT):
        self.mode = mode
        logger.info(f"PolicyEngine initialized in {mode.value} mode")

    def enforce(
        self,
        release_dir: Path,
        verification_result: Optional[VerificationResult] = None,
        validation_result: Optional[ValidationResult] = None,
        require_signatures: bool = True,
        require_encryption: bool = False
    ) -> Tuple[bool, List[str], List[str]]:
        """
        Enforce publication policy.

        Returns:
            (passed, warnings, errors)
        """
        warnings = []
        errors = []

        # Check Task 3 verification
        if verification_result:
            if not verification_result.passed:
                msg = f"Task 3 verification failed (exit code {verification_result.exit_code})"
                if self.mode == PublicationPolicy.STRICT:
                    errors.append(msg)
                else:
                    warnings.append(msg)
        else:
            msg = "Task 3 verification result not provided"
            if self.mode == PublicationPolicy.STRICT:
                errors.append(msg)
            else:
                warnings.append(msg)

        # Check Task 4 validation
        if validation_result:
            if not validation_result.passed:
                msg = f"Task 4 validation failed (exit code {validation_result.exit_code})"
                if self.mode == PublicationPolicy.STRICT:
                    errors.append(msg)
                else:
                    warnings.append(msg)
        else:
            msg = "Task 4 validation result not provided"
            if self.mode == PublicationPolicy.STRICT:
                errors.append(msg)
            else:
                warnings.append(msg)

        # Check for manifest
        manifest_file = release_dir / "manifest.json"
        if not manifest_file.exists():
            msg = "manifest.json not found"
            if self.mode == PublicationPolicy.STRICT:
                errors.append(msg)
            else:
                warnings.append(msg)

        # Check for signatures
        if require_signatures:
            sig_files = list(release_dir.rglob("*.sig"))
            if not sig_files:
                msg = "No signature files (.sig) found"
                if self.mode == PublicationPolicy.STRICT:
                    errors.append(msg)
                else:
                    warnings.append(msg)

        # Check for encryption
        if require_encryption:
            enc_files = list(release_dir.rglob("*.enc"))
            if not enc_files:
                msg = "No encrypted files (.enc) found"
                if self.mode == PublicationPolicy.STRICT:
                    errors.append(msg)
                else:
                    warnings.append(msg)

        # Check for SBOM
        sbom_files = list(release_dir.glob("sbom/*.json"))
        if not sbom_files:
            msg = "No SBOM files found in sbom/"
            warnings.append(msg)

        # Check for SLSA provenance
        slsa_files = list(release_dir.glob("slsa/*.json"))
        if not slsa_files:
            msg = "No SLSA provenance files found in slsa/"
            warnings.append(msg)

        passed = len(errors) == 0
        return passed, warnings, errors


# ============================================================================
# ATOMIC PUBLISHER ENGINE
# ============================================================================

class AtomicPublisher:
    """
    Implements atomic release publishing with staging and promotion.
    Ensures no partial publishes via write-staging-verify-promote cycle.
    """

    def __init__(self, repository: AbstractRepository):
        self.repository = repository
        self.staging_id = str(uuid.uuid4())[:8]
        logger.info(f"AtomicPublisher initialized with staging ID: {self.staging_id}")

    def publish(
        self,
        version: str,
        release_dir: Path,
        metadata: ReleaseMetadata
    ) -> Tuple[bool, List[str], int]:
        """
        Atomically publish release.

        Returns:
            (success, published_artifacts, total_bytes)
        """
        published = []
        total_bytes = 0
        staging_prefix = f".staging-{self.staging_id}/{version}"

        try:
            # Step 1: Check for duplicate version
            if self.repository.exists(version):
                raise DuplicateVersionError(
                    f"Version {version} already published (immutability violation)"
                )

            # Step 2: Upload to staging area
            logger.info(f"Stage 1: Uploading to staging area {staging_prefix}")
            artifacts = list(release_dir.rglob("*"))
            artifacts = [a for a in artifacts if a.is_file()]

            for artifact in artifacts:
                rel_path = artifact.relative_to(release_dir)
                staging_path = f"{staging_prefix}/{rel_path}"

                success = self.repository.upload(artifact, staging_path)
                if not success:
                    raise AtomicPublishError(f"Failed to upload {artifact} to staging")

                published.append(str(rel_path))
                total_bytes += artifact.stat().st_size

            logger.info(f"Stage 2: Uploaded {len(published)} artifacts ({total_bytes} bytes)")

            # Step 3: Verify staging area
            logger.info("Stage 3: Verifying staged artifacts")
            for artifact_path in published:
                staging_path = f"{staging_prefix}/{artifact_path}"
                if not self.repository.exists(staging_path):
                    raise AtomicPublishError(f"Verification failed: {staging_path} missing")

            logger.info("Stage 4: All artifacts verified in staging")

            # Step 4: Promote staging to production
            logger.info(f"Stage 5: Promoting {staging_prefix} → {version}")
            promoted = []
            for artifact_path in published:
                staging_path = f"{staging_prefix}/{artifact_path}"
                prod_path = f"{version}/{artifact_path}"

                # Download from staging
                temp_file = Path(f"/tmp/publish-{self.staging_id}-{artifact_path.replace('/', '-')}")
                temp_file.parent.mkdir(parents=True, exist_ok=True)

                if not self.repository.download(staging_path, temp_file):
                    raise AtomicPublishError(f"Failed to download {staging_path} for promotion")

                # Upload to production
                if not self.repository.upload(temp_file, prod_path):
                    raise AtomicPublishError(f"Failed to promote {artifact_path} to production")

                promoted.append(prod_path)
                temp_file.unlink()

            logger.info(f"Stage 6: Promoted {len(promoted)} artifacts to production")

            # Step 5: Cleanup staging
            logger.info("Stage 7: Cleaning up staging area")
            for artifact_path in published:
                staging_path = f"{staging_prefix}/{artifact_path}"
                self.repository.delete(staging_path)

            # Delete staging directory
            self.repository.delete(staging_prefix)

            logger.info(f"✓ Atomic publish complete: {version}")
            return True, [f"{version}/{p}" for p in published], total_bytes

        except Exception as e:
            logger.error(f"Atomic publish failed, rolling back: {e}")

            # Rollback: delete staging area
            try:
                for artifact_path in published:
                    staging_path = f"{staging_prefix}/{artifact_path}"
                    self.repository.delete(staging_path)
                self.repository.delete(staging_prefix)
            except Exception as rollback_err:
                logger.error(f"Rollback failed: {rollback_err}")

            raise AtomicPublishError(f"Atomic publish failed: {e}")


# ============================================================================
# INDEX BUILDER
# ============================================================================

class IndexBuilder:
    """
    Manages release index (index.json + index.md).
    Maintains historical metadata for all published releases.
    """

    def __init__(self, repository: AbstractRepository):
        self.repository = repository

    def update(
        self,
        version: str,
        metadata: ReleaseMetadata,
        verification_result: Optional[VerificationResult],
        validation_result: Optional[ValidationResult]
    ) -> bool:
        """
        Update index with new release metadata.
        """
        try:
            # Get current index
            index = self.repository.get_index()
            if index is None:
                index = {
                    "releases": [],
                    "last_updated": datetime.now(timezone.utc).isoformat(),
                    "total_releases": 0
                }

            # Check if version already exists
            existing_versions = [r["version"] for r in index.get("releases", [])]
            if version in existing_versions:
                logger.warning(f"Version {version} already in index, skipping")
                return False

            # Build release entry
            release_entry = {
                "version": version,
                "timestamp": metadata.timestamp,
                "sha256": metadata.sha256,
                "size_bytes": metadata.size_bytes,
                "artifacts": metadata.artifacts,
                "sbom_id": metadata.sbom_id,
                "slsa_id": metadata.slsa_id,
                "provenance_hash": metadata.provenance_hash,
                "signed": metadata.signed,
                "encrypted": metadata.encrypted,
            }

            # Add verification result
            if verification_result:
                release_entry["verification"] = {
                    "passed": verification_result.passed,
                    "timestamp": verification_result.timestamp,
                    "hash_verified": verification_result.hash_verified,
                    "signature_verified": verification_result.signature_verified,
                    "sbom_validated": verification_result.sbom_validated,
                    "slsa_validated": verification_result.slsa_validated,
                }

            # Add validation result
            if validation_result:
                release_entry["validation"] = {
                    "passed": validation_result.passed,
                    "timestamp": validation_result.timestamp,
                    "sbom_delta_passed": validation_result.sbom_delta_passed,
                    "slsa_delta_passed": validation_result.slsa_delta_passed,
                    "api_compat_passed": validation_result.api_compat_passed,
                    "performance_passed": validation_result.performance_passed,
                    "security_passed": validation_result.security_passed,
                    "behavioral_passed": validation_result.behavioral_passed,
                }

            # Add to index
            index["releases"].insert(0, release_entry)  # Most recent first
            index["total_releases"] = len(index["releases"])
            index["last_updated"] = datetime.now(timezone.utc).isoformat()

            # Update repository index
            success = self.repository.update_index(index)
            if success:
                logger.info(f"Index updated with {version}")

                # Generate markdown summary
                self._generate_markdown_index(index)

            return success

        except Exception as e:
            logger.error(f"Failed to update index: {e}")
            return False

    def _generate_markdown_index(self, index: Dict[str, Any]) -> bool:
        """Generate index.md summary."""
        try:
            lines = [
                "# T.A.R.S. Release Index\n",
                f"**Last Updated:** {index['last_updated']}\n",
                f"**Total Releases:** {index['total_releases']}\n",
                "\n---\n",
                "\n## Published Releases\n"
            ]

            for release in index["releases"]:
                lines.append(f"\n### {release['version']}\n")
                lines.append(f"- **Timestamp:** {release['timestamp']}\n")
                lines.append(f"- **SHA256:** {release['sha256'][:16]}...\n")
                lines.append(f"- **Size:** {release['size_bytes']:,} bytes\n")
                lines.append(f"- **Artifacts:** {len(release.get('artifacts', []))}\n")
                lines.append(f"- **Signed:** {'✓' if release.get('signed') else '✗'}\n")

                if release.get("verification"):
                    ver = release["verification"]
                    status = "✓ PASSED" if ver["passed"] else "✗ FAILED"
                    lines.append(f"- **Verification:** {status}\n")

                if release.get("validation"):
                    val = release["validation"]
                    status = "✓ PASSED" if val["passed"] else "✗ FAILED"
                    lines.append(f"- **Validation:** {status}\n")

            md_content = "".join(lines)

            # Write to temporary file, then upload
            temp_md = Path(f"/tmp/index-{uuid.uuid4()}.md")
            with open(temp_md, 'w', encoding='utf-8') as f:
                f.write(md_content)

            self.repository.upload(temp_md, "index.md")
            temp_md.unlink()

            logger.info("Generated index.md")
            return True

        except Exception as e:
            logger.error(f"Failed to generate markdown index: {e}")
            return False


# ============================================================================
# AUDIT LOG BUILDER
# ============================================================================

class AuditLogBuilder:
    """
    Generates audit logs with optional RSA-PSS signatures.
    Records all publication metadata and operations.
    """

    def __init__(self, sign_logs: bool = False):
        self.sign_logs = sign_logs

    def generate(
        self,
        version: str,
        metadata: ReleaseMetadata,
        verification_result: Optional[VerificationResult],
        validation_result: Optional[ValidationResult],
        publication_details: Dict[str, Any],
        output_dir: Path
    ) -> Tuple[bool, bool]:
        """
        Generate audit log.

        Returns:
            (log_created, log_signed)
        """
        try:
            timestamp = datetime.now(timezone.utc).isoformat()
            machine_id = platform.node()

            audit_data = {
                "audit_id": str(uuid.uuid4()),
                "version": version,
                "timestamp": timestamp,
                "machine_id": machine_id,
                "metadata": asdict(metadata),
                "verification": asdict(verification_result) if verification_result else None,
                "validation": asdict(validation_result) if validation_result else None,
                "publication": publication_details,
            }

            # Write audit log
            audit_file = output_dir / f"{version}.audit.json"
            audit_file.parent.mkdir(parents=True, exist_ok=True)

            with open(audit_file, 'w', encoding='utf-8') as f:
                json.dump(audit_data, f, indent=2, sort_keys=True)

            logger.info(f"Audit log created: {audit_file}")

            # Sign if requested
            if self.sign_logs:
                sig_created = self._sign_audit_log(audit_file)
                return True, sig_created

            return True, False

        except Exception as e:
            logger.error(f"Failed to generate audit log: {e}")
            return False, False

    def _sign_audit_log(self, audit_file: Path) -> bool:
        """
        Sign audit log with RSA-PSS (simulated).
        In production, would use actual private key.
        """
        try:
            # Read audit log
            with open(audit_file, 'rb') as f:
                audit_bytes = f.read()

            # Compute SHA256 hash (signature simulation)
            audit_hash = hashlib.sha256(audit_bytes).hexdigest()

            # Create signature file (simulated)
            sig_file = audit_file.with_suffix('.audit.sig')
            sig_data = {
                "algorithm": "RSA-PSS-SHA256",
                "hash": audit_hash,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "note": "Simulated signature - replace with actual RSA-PSS in production"
            }

            with open(sig_file, 'w', encoding='utf-8') as f:
                json.dump(sig_data, f, indent=2)

            logger.info(f"Audit log signed: {sig_file}")
            return True

        except Exception as e:
            logger.error(f"Failed to sign audit log: {e}")
            return False


# ============================================================================
# PUBLICATION ORCHESTRATOR
# ============================================================================

class PublisherOrchestrator:
    """
    Top-level orchestrator for release publication.
    Coordinates policy enforcement, atomic publishing, indexing, and audit logging.
    """

    def __init__(
        self,
        repository: AbstractRepository,
        policy_mode: PublicationPolicy = PublicationPolicy.STRICT,
        sign_audit_logs: bool = False,
        require_signatures: bool = True,
        require_encryption: bool = False
    ):
        self.repository = repository
        self.policy_engine = PublicationPolicyEngine(policy_mode)
        self.atomic_publisher = AtomicPublisher(repository)
        self.index_builder = IndexBuilder(repository)
        self.audit_builder = AuditLogBuilder(sign_audit_logs)

        self.policy_mode = policy_mode
        self.require_signatures = require_signatures
        self.require_encryption = require_encryption

        logger.info(f"PublisherOrchestrator initialized (policy={policy_mode.value})")

    def publish_release(
        self,
        version: str,
        release_dir: Path,
        verification_result: Optional[VerificationResult] = None,
        validation_result: Optional[ValidationResult] = None,
        audit_output_dir: Optional[Path] = None
    ) -> PublicationReport:
        """
        Execute complete publication workflow.

        Returns:
            PublicationReport with comprehensive details
        """
        start_time = datetime.now(timezone.utc)
        timestamp = start_time.isoformat()

        report = PublicationReport(
            version=version,
            status=PublicationStatus.FAILED.value,
            timestamp=timestamp,
            repository_type=self.repository.repo_type,
            repository_location=str(getattr(self.repository, 'base_path', 'unknown')),
            policy_mode=self.policy_mode.value,
        )

        try:
            # Step 1: Policy enforcement
            logger.info(f"Step 1: Enforcing publication policy for {version}")
            policy_passed, warnings, errors = self.policy_engine.enforce(
                release_dir,
                verification_result,
                validation_result,
                self.require_signatures,
                self.require_encryption
            )

            report.warnings.extend(warnings)
            report.errors.extend(errors)
            report.verification_passed = verification_result.passed if verification_result else False
            report.validation_passed = validation_result.passed if validation_result else False

            if not policy_passed:
                report.exit_code = PolicyViolationError.exit_code
                report.summary = f"Publication policy violations: {len(errors)} errors"
                raise PolicyViolationError(report.summary)

            logger.info("✓ Policy checks passed")

            # Step 2: Extract metadata
            logger.info("Step 2: Extracting release metadata")
            metadata = self._extract_metadata(release_dir, verification_result, validation_result)
            report.metadata_complete = True

            # Check signatures
            sig_files = list(release_dir.rglob("*.sig"))
            report.signatures_present = len(sig_files) > 0
            metadata.signed = report.signatures_present

            # Check encryption
            enc_files = list(release_dir.rglob("*.enc"))
            metadata.encrypted = len(enc_files) > 0

            logger.info(f"✓ Metadata extracted (signed={metadata.signed}, encrypted={metadata.encrypted})")

            # Step 3: Atomic publish
            logger.info("Step 3: Atomically publishing release")
            success, published_artifacts, total_bytes = self.atomic_publisher.publish(
                version, release_dir, metadata
            )

            if not success:
                report.exit_code = AtomicPublishError.exit_code
                report.summary = "Atomic publish failed"
                raise AtomicPublishError(report.summary)

            report.artifacts_published = published_artifacts
            report.total_size_bytes = total_bytes
            metadata.artifacts = published_artifacts
            metadata.size_bytes = total_bytes

            logger.info(f"✓ Published {len(published_artifacts)} artifacts ({total_bytes:,} bytes)")

            # Step 4: Update index
            logger.info("Step 4: Updating release index")
            index_updated = self.index_builder.update(
                version, metadata, verification_result, validation_result
            )
            report.index_updated = index_updated

            if index_updated:
                logger.info("✓ Index updated")
            else:
                report.warnings.append("Index update failed (non-fatal)")

            # Step 5: Generate audit log
            logger.info("Step 5: Generating audit log")
            if audit_output_dir is None:
                audit_output_dir = release_dir / "audit"

            publication_details = {
                "artifacts_published": len(published_artifacts),
                "total_bytes": total_bytes,
                "repository_type": self.repository.repo_type,
                "policy_mode": self.policy_mode.value,
            }

            audit_created, audit_signed = self.audit_builder.generate(
                version, metadata, verification_result, validation_result,
                publication_details, audit_output_dir
            )

            report.audit_log_created = audit_created
            report.audit_log_signed = audit_signed

            if audit_created:
                logger.info(f"✓ Audit log created (signed={audit_signed})")
            else:
                report.warnings.append("Audit log generation failed (non-fatal)")

            # Success
            end_time = datetime.now(timezone.utc)
            report.publication_duration_seconds = (end_time - start_time).total_seconds()
            report.status = PublicationStatus.SUCCESS.value
            report.exit_code = 0
            report.summary = f"Successfully published {version} ({len(published_artifacts)} artifacts, {total_bytes:,} bytes)"

            logger.info(f"✓ Publication complete: {version} ({report.publication_duration_seconds:.2f}s)")
            return report

        except DuplicateVersionError as e:
            report.exit_code = e.exit_code
            report.errors.append(str(e))
            report.summary = str(e)
            logger.error(f"✗ Publication failed: {e}")
            return report

        except PolicyViolationError as e:
            report.exit_code = e.exit_code
            report.summary = str(e)
            logger.error(f"✗ Publication failed: {e}")
            return report

        except AtomicPublishError as e:
            report.exit_code = e.exit_code
            report.errors.append(str(e))
            report.summary = str(e)
            logger.error(f"✗ Publication failed: {e}")
            return report

        except Exception as e:
            report.exit_code = PublicationError.exit_code
            report.errors.append(f"Unexpected error: {e}")
            report.summary = f"Publication failed: {e}"
            logger.error(f"✗ Publication failed: {e}", exc_info=True)
            return report

    def _extract_metadata(
        self,
        release_dir: Path,
        verification_result: Optional[VerificationResult],
        validation_result: Optional[ValidationResult]
    ) -> ReleaseMetadata:
        """Extract release metadata from directory."""
        # Compute SHA256 of entire release
        hasher = hashlib.sha256()
        total_size = 0

        for file_path in sorted(release_dir.rglob("*")):
            if file_path.is_file():
                with open(file_path, 'rb') as f:
                    hasher.update(f.read())
                total_size += file_path.stat().st_size

        release_hash = hasher.hexdigest()

        # Extract SBOM ID
        sbom_id = None
        sbom_files = list(release_dir.glob("sbom/*.json"))
        if sbom_files:
            try:
                with open(sbom_files[0], 'r', encoding='utf-8') as f:
                    sbom_data = json.load(f)
                    sbom_id = sbom_data.get("serialNumber") or sbom_data.get("SPDXID")
            except Exception:
                pass

        # Extract SLSA ID
        slsa_id = None
        provenance_hash = None
        slsa_files = list(release_dir.glob("slsa/*.json"))
        if slsa_files:
            try:
                with open(slsa_files[0], 'r', encoding='utf-8') as f:
                    slsa_data = json.load(f)
                    # Compute hash of provenance
                    provenance_bytes = json.dumps(slsa_data, sort_keys=True).encode()
                    provenance_hash = hashlib.sha256(provenance_bytes).hexdigest()
                    slsa_id = slsa_data.get("_type", "https://in-toto.io/Statement/v1")
            except Exception:
                pass

        return ReleaseMetadata(
            version=release_dir.name,
            timestamp=datetime.now(timezone.utc).isoformat(),
            sha256=release_hash,
            size_bytes=total_size,
            verification_result=asdict(verification_result) if verification_result else None,
            validation_result=asdict(validation_result) if validation_result else None,
            sbom_id=sbom_id,
            slsa_id=slsa_id,
            provenance_hash=provenance_hash,
        )

    def generate_json_report(self, report: PublicationReport, output_path: Path) -> bool:
        """Generate JSON publication report."""
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(asdict(report), f, indent=2, sort_keys=True)
            logger.info(f"JSON report written: {output_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to write JSON report: {e}")
            return False

    def generate_text_report(self, report: PublicationReport, output_path: Path) -> bool:
        """Generate text publication report."""
        try:
            lines = [
                "=" * 80,
                "T.A.R.S. RELEASE PUBLICATION REPORT",
                "=" * 80,
                f"Version:           {report.version}",
                f"Status:            {report.status.upper()}",
                f"Timestamp:         {report.timestamp}",
                f"Repository Type:   {report.repository_type}",
                f"Repository Path:   {report.repository_location}",
                f"Policy Mode:       {report.policy_mode}",
                f"Duration:          {report.publication_duration_seconds:.2f}s",
                "",
                "-" * 80,
                "PRE-FLIGHT CHECKS",
                "-" * 80,
                f"Verification (Task 3):  {'PASSED' if report.verification_passed else 'FAILED'}",
                f"Validation (Task 4):    {'PASSED' if report.validation_passed else 'FAILED'}",
                f"Metadata Complete:      {'YES' if report.metadata_complete else 'NO'}",
                f"Signatures Present:     {'YES' if report.signatures_present else 'NO'}",
                "",
                "-" * 80,
                "PUBLICATION DETAILS",
                "-" * 80,
                f"Artifacts Published:    {len(report.artifacts_published)}",
                f"Total Size:             {report.total_size_bytes:,} bytes",
                f"Index Updated:          {'YES' if report.index_updated else 'NO'}",
                f"Audit Log Created:      {'YES' if report.audit_log_created else 'NO'}",
                f"Audit Log Signed:       {'YES' if report.audit_log_signed else 'NO'}",
                "",
            ]

            if report.artifacts_published:
                lines.append("-" * 80)
                lines.append("PUBLISHED ARTIFACTS")
                lines.append("-" * 80)
                for artifact in report.artifacts_published[:20]:  # First 20
                    lines.append(f"  - {artifact}")
                if len(report.artifacts_published) > 20:
                    lines.append(f"  ... and {len(report.artifacts_published) - 20} more")
                lines.append("")

            if report.warnings:
                lines.append("-" * 80)
                lines.append(f"WARNINGS ({len(report.warnings)})")
                lines.append("-" * 80)
                for warning in report.warnings:
                    lines.append(f"  ⚠ {warning}")
                lines.append("")

            if report.errors:
                lines.append("-" * 80)
                lines.append(f"ERRORS ({len(report.errors)})")
                lines.append("-" * 80)
                for error in report.errors:
                    lines.append(f"  ✗ {error}")
                lines.append("")

            lines.extend([
                "-" * 80,
                "SUMMARY",
                "-" * 80,
                report.summary,
                "",
                f"Exit Code: {report.exit_code}",
                "=" * 80,
            ])

            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(lines))

            logger.info(f"Text report written: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to write text report: {e}")
            return False


# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    """CLI entry point for release publisher."""
    import argparse

    parser = argparse.ArgumentParser(description="T.A.R.S. Release Publisher")
    parser.add_argument("--version", required=True, help="Release version")
    parser.add_argument("--release-dir", required=True, type=Path, help="Release directory")
    parser.add_argument("--repository-type", choices=["local", "s3", "gcs"], default="local")
    parser.add_argument("--repository-path", type=Path, help="Local repository path")
    parser.add_argument("--repository-bucket", help="S3/GCS bucket name")
    parser.add_argument("--repository-prefix", default="", help="S3/GCS prefix")
    parser.add_argument("--policy", choices=["strict", "lenient"], default="strict")
    parser.add_argument("--require-signatures", action="store_true", default=True)
    parser.add_argument("--require-encryption", action="store_true", default=False)
    parser.add_argument("--sign-audit-log", action="store_true", default=False)
    parser.add_argument("--verification-result", type=Path, help="Task 3 verification result JSON")
    parser.add_argument("--validation-result", type=Path, help="Task 4 validation result JSON")
    parser.add_argument("--json-report", type=Path, help="JSON report output path")
    parser.add_argument("--text-report", type=Path, help="Text report output path")
    parser.add_argument("--audit-output-dir", type=Path, help="Audit log output directory")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Build repository config
    repo_config = {"type": args.repository_type}
    if args.repository_type == "local":
        repo_config["path"] = args.repository_path or "./repository"
    else:
        repo_config["bucket"] = args.repository_bucket or "default-bucket"
        repo_config["prefix"] = args.repository_prefix

    # Create repository
    repository = RepositoryFactory.create(args.repository_type, repo_config)

    # Load verification result
    verification_result = None
    if args.verification_result and args.verification_result.exists():
        with open(args.verification_result, 'r') as f:
            ver_data = json.load(f)
            verification_result = VerificationResult(**ver_data)

    # Load validation result
    validation_result = None
    if args.validation_result and args.validation_result.exists():
        with open(args.validation_result, 'r') as f:
            val_data = json.load(f)
            validation_result = ValidationResult(**val_data)

    # Create orchestrator
    policy_mode = PublicationPolicy.STRICT if args.policy == "strict" else PublicationPolicy.LENIENT
    orchestrator = PublisherOrchestrator(
        repository=repository,
        policy_mode=policy_mode,
        sign_audit_logs=args.sign_audit_log,
        require_signatures=args.require_signatures,
        require_encryption=args.require_encryption,
    )

    # Publish release
    report = orchestrator.publish_release(
        version=args.version,
        release_dir=args.release_dir,
        verification_result=verification_result,
        validation_result=validation_result,
        audit_output_dir=args.audit_output_dir,
    )

    # Generate reports
    if args.json_report:
        orchestrator.generate_json_report(report, args.json_report)

    if args.text_report:
        orchestrator.generate_text_report(report, args.text_report)

    # Print summary
    print(f"\n{'='*80}")
    print(f"Publication Status: {report.status.upper()}")
    print(f"{'='*80}")
    print(report.summary)
    print(f"Exit Code: {report.exit_code}")
    print(f"{'='*80}\n")

    return report.exit_code


if __name__ == "__main__":
    exit(main())
