#!/usr/bin/env python3
"""
Test Suite for Release Publisher - Phase 14.7 Task 5

Comprehensive tests covering repository adapters, atomic publishing,
policy enforcement, indexing, audit logging, and orchestration.

Author: T.A.R.S. Development Team
Version: 1.0.0
Date: 2025-11-28
"""

import pytest
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timezone
from publisher.release_publisher import (
    # Repositories
    RepositoryFactory,
    LocalRepository,
    S3StyleRepository,
    GCSStyleRepository,
    # Engines
    PublicationPolicyEngine,
    AtomicPublisher,
    IndexBuilder,
    AuditLogBuilder,
    PublisherOrchestrator,
    # Data classes
    VerificationResult,
    ValidationResult,
    ReleaseMetadata,
    PublicationReport,
    PublicationPolicy,
    # Exceptions
    DuplicateVersionError,
    PolicyViolationError,
    AtomicPublishError,
    MetadataMissingError,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def temp_dir():
    """Create temporary directory."""
    temp = tempfile.mkdtemp()
    yield Path(temp)
    shutil.rmtree(temp, ignore_errors=True)


@pytest.fixture
def mock_release_dir(temp_dir):
    """Create mock release directory with artifacts."""
    release_dir = temp_dir / "v1.0.2"
    release_dir.mkdir(parents=True)

    # Create manifest
    manifest = {
        "version": "1.0.2",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "artifacts": []
    }
    with open(release_dir / "manifest.json", 'w') as f:
        json.dump(manifest, f)

    # Create SBOM
    sbom_dir = release_dir / "sbom"
    sbom_dir.mkdir()
    sbom = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "serialNumber": "urn:uuid:test-sbom-123",
        "version": 1,
        "components": []
    }
    with open(sbom_dir / "sbom.json", 'w') as f:
        json.dump(sbom, f)

    # Create SLSA provenance
    slsa_dir = release_dir / "slsa"
    slsa_dir.mkdir()
    slsa = {
        "_type": "https://in-toto.io/Statement/v1",
        "subject": [],
        "predicateType": "https://slsa.dev/provenance/v1",
        "predicate": {}
    }
    with open(slsa_dir / "provenance.json", 'w') as f:
        json.dump(slsa, f)

    # Create signature file
    with open(release_dir / "manifest.json.sig", 'w') as f:
        f.write("mock_signature")

    # Create some artifacts
    artifacts_dir = release_dir / "artifacts"
    artifacts_dir.mkdir()
    for i in range(3):
        with open(artifacts_dir / f"artifact{i}.bin", 'wb') as f:
            f.write(b"test_data" * 100)

    return release_dir


@pytest.fixture
def verification_result_passed():
    """Passed verification result."""
    return VerificationResult(
        passed=True,
        timestamp=datetime.now(timezone.utc).isoformat(),
        hash_verified=True,
        signature_verified=True,
        sbom_validated=True,
        slsa_validated=True,
        policy_passed=True,
        exit_code=0
    )


@pytest.fixture
def verification_result_failed():
    """Failed verification result."""
    return VerificationResult(
        passed=False,
        timestamp=datetime.now(timezone.utc).isoformat(),
        hash_verified=False,
        exit_code=10
    )


@pytest.fixture
def validation_result_passed():
    """Passed validation result."""
    return ValidationResult(
        passed=True,
        timestamp=datetime.now(timezone.utc).isoformat(),
        sbom_delta_passed=True,
        slsa_delta_passed=True,
        api_compat_passed=True,
        performance_passed=True,
        security_passed=True,
        behavioral_passed=True,
        exit_code=0
    )


@pytest.fixture
def validation_result_failed():
    """Failed validation result."""
    return ValidationResult(
        passed=False,
        timestamp=datetime.now(timezone.utc).isoformat(),
        performance_passed=False,
        exit_code=24
    )


@pytest.fixture
def local_repo(temp_dir):
    """Create local repository."""
    repo_path = temp_dir / "repository"
    config = {"type": "local", "path": str(repo_path)}
    return LocalRepository(config)


@pytest.fixture
def s3_repo(temp_dir):
    """Create S3-style repository."""
    config = {
        "type": "s3",
        "bucket": "test-bucket",
        "prefix": "releases",
        "local_base": str(temp_dir / "s3-sim")
    }
    return S3StyleRepository(config)


@pytest.fixture
def gcs_repo(temp_dir):
    """Create GCS-style repository."""
    config = {
        "type": "gcs",
        "bucket": "test-bucket",
        "prefix": "releases",
        "local_base": str(temp_dir / "gcs-sim")
    }
    return GCSStyleRepository(config)


# ============================================================================
# TEST: REPOSITORY FACTORY
# ============================================================================

class TestRepositoryFactory:
    """Test repository factory."""

    def test_create_local_repository(self, temp_dir):
        """Test creating local repository."""
        config = {"type": "local", "path": str(temp_dir / "repo")}
        repo = RepositoryFactory.create("local", config)
        assert isinstance(repo, LocalRepository)
        assert repo.base_path == temp_dir / "repo"

    def test_create_s3_repository(self, temp_dir):
        """Test creating S3 repository."""
        config = {
            "type": "s3",
            "bucket": "my-bucket",
            "prefix": "releases",
            "local_base": str(temp_dir)
        }
        repo = RepositoryFactory.create("s3", config)
        assert isinstance(repo, S3StyleRepository)
        assert repo.bucket == "my-bucket"
        assert repo.prefix == "releases"

    def test_create_gcs_repository(self, temp_dir):
        """Test creating GCS repository."""
        config = {
            "type": "gcs",
            "bucket": "my-bucket",
            "prefix": "releases",
            "local_base": str(temp_dir)
        }
        repo = RepositoryFactory.create("gcs", config)
        assert isinstance(repo, GCSStyleRepository)
        assert repo.bucket == "my-bucket"

    def test_unsupported_repository_type(self):
        """Test unsupported repository type."""
        with pytest.raises(ValueError, match="Unsupported repository type"):
            RepositoryFactory.create("azure", {})


# ============================================================================
# TEST: LOCAL REPOSITORY
# ============================================================================

class TestLocalRepository:
    """Test local repository adapter."""

    def test_upload_file(self, local_repo, temp_dir):
        """Test uploading file."""
        # Create test file
        test_file = temp_dir / "test.txt"
        test_file.write_text("hello world")

        # Upload
        success = local_repo.upload(test_file, "v1.0.0/test.txt")
        assert success

        # Verify
        assert local_repo.exists("v1.0.0/test.txt")
        uploaded = local_repo.base_path / "v1.0.0/test.txt"
        assert uploaded.read_text() == "hello world"

    def test_upload_duplicate_rejected(self, local_repo, temp_dir):
        """Test duplicate upload is rejected."""
        test_file = temp_dir / "test.txt"
        test_file.write_text("content")

        # First upload succeeds
        assert local_repo.upload(test_file, "v1.0.0/test.txt")

        # Second upload fails
        assert not local_repo.upload(test_file, "v1.0.0/test.txt")

    def test_download_file(self, local_repo, temp_dir):
        """Test downloading file."""
        # Upload first
        test_file = temp_dir / "upload.txt"
        test_file.write_text("download test")
        local_repo.upload(test_file, "v1.0.0/file.txt")

        # Download
        download_path = temp_dir / "download.txt"
        success = local_repo.download("v1.0.0/file.txt", download_path)
        assert success
        assert download_path.read_text() == "download test"

    def test_download_nonexistent_file(self, local_repo, temp_dir):
        """Test downloading nonexistent file fails."""
        download_path = temp_dir / "missing.txt"
        success = local_repo.download("v1.0.0/missing.txt", download_path)
        assert not success

    def test_list_versions(self, local_repo, temp_dir):
        """Test listing versions."""
        # Create version directories
        for version in ["v1.0.0", "v1.0.1", "v1.0.2"]:
            (local_repo.base_path / version).mkdir(parents=True)

        versions = local_repo.list_versions()
        assert len(versions) == 3
        assert "v1.0.2" in versions
        assert "v1.0.1" in versions
        assert "v1.0.0" in versions

    def test_get_index_empty(self, local_repo):
        """Test getting index when it doesn't exist."""
        index = local_repo.get_index()
        assert index is not None
        assert "releases" in index
        assert len(index["releases"]) == 0

    def test_update_index(self, local_repo):
        """Test updating index."""
        index_data = {
            "releases": [
                {"version": "v1.0.0", "timestamp": "2025-01-01T00:00:00Z"}
            ],
            "last_updated": datetime.now(timezone.utc).isoformat()
        }

        success = local_repo.update_index(index_data)
        assert success

        # Verify
        retrieved = local_repo.get_index()
        assert len(retrieved["releases"]) == 1
        assert retrieved["releases"][0]["version"] == "v1.0.0"

    def test_delete_file(self, local_repo, temp_dir):
        """Test deleting file."""
        # Create and upload file
        test_file = temp_dir / "delete_me.txt"
        test_file.write_text("delete")
        local_repo.upload(test_file, "v1.0.0/delete_me.txt")

        # Delete
        success = local_repo.delete("v1.0.0/delete_me.txt")
        assert success
        assert not local_repo.exists("v1.0.0/delete_me.txt")


# ============================================================================
# TEST: S3-STYLE REPOSITORY
# ============================================================================

class TestS3StyleRepository:
    """Test S3-style repository adapter."""

    def test_upload_with_prefix(self, s3_repo, temp_dir):
        """Test uploading with prefix."""
        test_file = temp_dir / "test.txt"
        test_file.write_text("s3 test")

        success = s3_repo.upload(test_file, "v1.0.0/test.txt")
        assert success

        # Check correct path (prefix applied)
        expected_path = s3_repo.local_base / "releases/v1.0.0/test.txt"
        assert expected_path.exists()
        assert expected_path.read_text() == "s3 test"

    def test_list_versions(self, s3_repo):
        """Test listing versions in S3."""
        # Create version directories
        for version in ["v1.0.0", "v1.0.1"]:
            (s3_repo.local_base / "releases" / version).mkdir(parents=True)

        versions = s3_repo.list_versions()
        assert len(versions) == 2
        assert "v1.0.1" in versions

    def test_get_index(self, s3_repo, temp_dir):
        """Test getting index from S3."""
        # Create index
        index_data = {
            "releases": [{"version": "v1.0.0"}],
            "last_updated": datetime.now(timezone.utc).isoformat()
        }

        # Upload index
        index_file = temp_dir / "index.json"
        with open(index_file, 'w') as f:
            json.dump(index_data, f)

        s3_repo.upload(index_file, "index.json")

        # Retrieve
        retrieved = s3_repo.get_index()
        assert len(retrieved["releases"]) == 1
        assert retrieved["releases"][0]["version"] == "v1.0.0"


# ============================================================================
# TEST: GCS-STYLE REPOSITORY
# ============================================================================

class TestGCSStyleRepository:
    """Test GCS-style repository adapter."""

    def test_upload_blob(self, gcs_repo, temp_dir):
        """Test uploading blob."""
        test_file = temp_dir / "test.txt"
        test_file.write_text("gcs test")

        success = gcs_repo.upload(test_file, "v1.0.0/test.txt")
        assert success

        # Verify path
        expected_path = gcs_repo.local_base / "releases/v1.0.0/test.txt"
        assert expected_path.exists()

    def test_download_blob(self, gcs_repo, temp_dir):
        """Test downloading blob."""
        # Upload first
        test_file = temp_dir / "upload.txt"
        test_file.write_text("gcs download")
        gcs_repo.upload(test_file, "v1.0.0/file.txt")

        # Download
        download_path = temp_dir / "download.txt"
        success = gcs_repo.download("v1.0.0/file.txt", download_path)
        assert success
        assert download_path.read_text() == "gcs download"


# ============================================================================
# TEST: PUBLICATION POLICY ENGINE
# ============================================================================

class TestPublicationPolicyEngine:
    """Test publication policy engine."""

    def test_strict_mode_all_passed(
        self,
        mock_release_dir,
        verification_result_passed,
        validation_result_passed
    ):
        """Test strict mode with all checks passed."""
        engine = PublicationPolicyEngine(PublicationPolicy.STRICT)

        passed, warnings, errors = engine.enforce(
            mock_release_dir,
            verification_result_passed,
            validation_result_passed,
            require_signatures=True,
            require_encryption=False
        )

        assert passed
        assert len(errors) == 0

    def test_strict_mode_verification_failed(
        self,
        mock_release_dir,
        verification_result_failed,
        validation_result_passed
    ):
        """Test strict mode with verification failure."""
        engine = PublicationPolicyEngine(PublicationPolicy.STRICT)

        passed, warnings, errors = engine.enforce(
            mock_release_dir,
            verification_result_failed,
            validation_result_passed
        )

        assert not passed
        assert len(errors) > 0
        assert any("verification failed" in e.lower() for e in errors)

    def test_strict_mode_validation_failed(
        self,
        mock_release_dir,
        verification_result_passed,
        validation_result_failed
    ):
        """Test strict mode with validation failure."""
        engine = PublicationPolicyEngine(PublicationPolicy.STRICT)

        passed, warnings, errors = engine.enforce(
            mock_release_dir,
            verification_result_passed,
            validation_result_failed
        )

        assert not passed
        assert len(errors) > 0
        assert any("validation failed" in e.lower() for e in errors)

    def test_strict_mode_missing_manifest(
        self,
        temp_dir,
        verification_result_passed,
        validation_result_passed
    ):
        """Test strict mode with missing manifest."""
        release_dir = temp_dir / "v1.0.2"
        release_dir.mkdir()

        engine = PublicationPolicyEngine(PublicationPolicy.STRICT)

        passed, warnings, errors = engine.enforce(
            release_dir,
            verification_result_passed,
            validation_result_passed
        )

        assert not passed
        assert any("manifest.json not found" in e for e in errors)

    def test_lenient_mode_failures_become_warnings(
        self,
        mock_release_dir,
        verification_result_failed,
        validation_result_failed
    ):
        """Test lenient mode converts failures to warnings."""
        engine = PublicationPolicyEngine(PublicationPolicy.LENIENT)

        passed, warnings, errors = engine.enforce(
            mock_release_dir,
            verification_result_failed,
            validation_result_failed
        )

        assert passed  # Lenient mode passes
        assert len(warnings) > 0
        assert len(errors) == 0

    def test_missing_results_strict_mode(self, mock_release_dir):
        """Test strict mode with missing results."""
        engine = PublicationPolicyEngine(PublicationPolicy.STRICT)

        passed, warnings, errors = engine.enforce(
            mock_release_dir,
            verification_result=None,
            validation_result=None
        )

        assert not passed
        assert len(errors) >= 2  # Missing verification + validation


# ============================================================================
# TEST: ATOMIC PUBLISHER
# ============================================================================

class TestAtomicPublisher:
    """Test atomic publisher engine."""

    def test_atomic_publish_success(self, local_repo, mock_release_dir):
        """Test successful atomic publish."""
        publisher = AtomicPublisher(local_repo)
        metadata = ReleaseMetadata(
            version="v1.0.2",
            timestamp=datetime.now(timezone.utc).isoformat(),
            sha256="abc123",
            size_bytes=1024
        )

        success, published, total_bytes = publisher.publish(
            "v1.0.2", mock_release_dir, metadata
        )

        assert success
        assert len(published) > 0
        assert total_bytes > 0

        # Verify artifacts in production location
        assert local_repo.exists("v1.0.2/manifest.json")
        assert local_repo.exists("v1.0.2/sbom/sbom.json")

    def test_atomic_publish_duplicate_version(self, local_repo, mock_release_dir):
        """Test atomic publish rejects duplicate version."""
        publisher = AtomicPublisher(local_repo)
        metadata = ReleaseMetadata(
            version="v1.0.2",
            timestamp=datetime.now(timezone.utc).isoformat(),
            sha256="abc123",
            size_bytes=1024
        )

        # First publish succeeds
        success1, _, _ = publisher.publish("v1.0.2", mock_release_dir, metadata)
        assert success1

        # Second publish fails
        with pytest.raises(DuplicateVersionError):
            publisher.publish("v1.0.2", mock_release_dir, metadata)

    def test_atomic_publish_staging_cleanup(self, local_repo, mock_release_dir):
        """Test staging area is cleaned up after publish."""
        publisher = AtomicPublisher(local_repo)
        metadata = ReleaseMetadata(
            version="v1.0.2",
            timestamp=datetime.now(timezone.utc).isoformat(),
            sha256="abc123",
            size_bytes=1024
        )

        publisher.publish("v1.0.2", mock_release_dir, metadata)

        # Check no staging directories remain
        staging_dirs = list(local_repo.base_path.glob(".staging-*"))
        assert len(staging_dirs) == 0


# ============================================================================
# TEST: INDEX BUILDER
# ============================================================================

class TestIndexBuilder:
    """Test index builder."""

    def test_update_index_new_release(
        self,
        local_repo,
        verification_result_passed,
        validation_result_passed
    ):
        """Test updating index with new release."""
        builder = IndexBuilder(local_repo)
        metadata = ReleaseMetadata(
            version="v1.0.2",
            timestamp=datetime.now(timezone.utc).isoformat(),
            sha256="abc123",
            size_bytes=1024,
            sbom_id="urn:uuid:sbom-123",
            slsa_id="https://in-toto.io/Statement/v1",
            signed=True
        )

        success = builder.update(
            "v1.0.2",
            metadata,
            verification_result_passed,
            validation_result_passed
        )

        assert success

        # Verify index
        index = local_repo.get_index()
        assert len(index["releases"]) == 1
        assert index["releases"][0]["version"] == "v1.0.2"
        assert index["releases"][0]["signed"] is True
        assert "verification" in index["releases"][0]
        assert "validation" in index["releases"][0]

    def test_update_index_multiple_releases(self, local_repo):
        """Test index with multiple releases."""
        builder = IndexBuilder(local_repo)

        # Add v1.0.0
        metadata1 = ReleaseMetadata(
            version="v1.0.0",
            timestamp="2025-01-01T00:00:00Z",
            sha256="hash1",
            size_bytes=1000
        )
        builder.update("v1.0.0", metadata1, None, None)

        # Add v1.0.1
        metadata2 = ReleaseMetadata(
            version="v1.0.1",
            timestamp="2025-01-02T00:00:00Z",
            sha256="hash2",
            size_bytes=2000
        )
        builder.update("v1.0.1", metadata2, None, None)

        # Verify
        index = local_repo.get_index()
        assert len(index["releases"]) == 2
        # Most recent first
        assert index["releases"][0]["version"] == "v1.0.1"
        assert index["releases"][1]["version"] == "v1.0.0"

    def test_update_index_duplicate_rejected(self, local_repo):
        """Test duplicate version in index is rejected."""
        builder = IndexBuilder(local_repo)
        metadata = ReleaseMetadata(
            version="v1.0.0",
            timestamp=datetime.now(timezone.utc).isoformat(),
            sha256="hash",
            size_bytes=1000
        )

        # First update succeeds
        assert builder.update("v1.0.0", metadata, None, None)

        # Second update fails
        assert not builder.update("v1.0.0", metadata, None, None)


# ============================================================================
# TEST: AUDIT LOG BUILDER
# ============================================================================

class TestAuditLogBuilder:
    """Test audit log builder."""

    def test_generate_audit_log_unsigned(
        self,
        temp_dir,
        verification_result_passed,
        validation_result_passed
    ):
        """Test generating unsigned audit log."""
        builder = AuditLogBuilder(sign_logs=False)
        metadata = ReleaseMetadata(
            version="v1.0.2",
            timestamp=datetime.now(timezone.utc).isoformat(),
            sha256="abc123",
            size_bytes=1024
        )

        publication_details = {
            "artifacts_published": 5,
            "total_bytes": 1024,
            "repository_type": "local"
        }

        log_created, log_signed = builder.generate(
            "v1.0.2",
            metadata,
            verification_result_passed,
            validation_result_passed,
            publication_details,
            temp_dir
        )

        assert log_created
        assert not log_signed

        # Verify audit log file
        audit_file = temp_dir / "v1.0.2.audit.json"
        assert audit_file.exists()

        with open(audit_file, 'r') as f:
            audit_data = json.load(f)
            assert audit_data["version"] == "v1.0.2"
            assert "audit_id" in audit_data
            assert "timestamp" in audit_data
            assert "metadata" in audit_data

    def test_generate_audit_log_signed(self, temp_dir):
        """Test generating signed audit log."""
        builder = AuditLogBuilder(sign_logs=True)
        metadata = ReleaseMetadata(
            version="v1.0.2",
            timestamp=datetime.now(timezone.utc).isoformat(),
            sha256="abc123",
            size_bytes=1024
        )

        log_created, log_signed = builder.generate(
            "v1.0.2",
            metadata,
            None,
            None,
            {},
            temp_dir
        )

        assert log_created
        assert log_signed

        # Verify signature file
        sig_file = temp_dir / "v1.0.2.audit.sig"
        assert sig_file.exists()

        with open(sig_file, 'r') as f:
            sig_data = json.load(f)
            assert sig_data["algorithm"] == "RSA-PSS-SHA256"
            assert "hash" in sig_data


# ============================================================================
# TEST: PUBLISHER ORCHESTRATOR
# ============================================================================

class TestPublisherOrchestrator:
    """Test publisher orchestrator."""

    def test_publish_release_success(
        self,
        local_repo,
        mock_release_dir,
        verification_result_passed,
        validation_result_passed,
        temp_dir
    ):
        """Test successful release publication."""
        orchestrator = PublisherOrchestrator(
            repository=local_repo,
            policy_mode=PublicationPolicy.STRICT,
            sign_audit_logs=False
        )

        report = orchestrator.publish_release(
            version="v1.0.2",
            release_dir=mock_release_dir,
            verification_result=verification_result_passed,
            validation_result=validation_result_passed,
            audit_output_dir=temp_dir
        )

        assert report.status == "success"
        assert report.exit_code == 0
        assert report.verification_passed
        assert report.validation_passed
        assert len(report.artifacts_published) > 0
        assert report.index_updated
        assert report.audit_log_created

    def test_publish_release_policy_violation(
        self,
        local_repo,
        mock_release_dir,
        verification_result_failed,
        validation_result_passed
    ):
        """Test publication fails on policy violation."""
        orchestrator = PublisherOrchestrator(
            repository=local_repo,
            policy_mode=PublicationPolicy.STRICT
        )

        report = orchestrator.publish_release(
            version="v1.0.2",
            release_dir=mock_release_dir,
            verification_result=verification_result_failed,
            validation_result=validation_result_passed
        )

        assert report.status == "failed"
        assert report.exit_code == 38  # PolicyViolationError
        assert len(report.errors) > 0

    def test_publish_release_lenient_mode(
        self,
        local_repo,
        mock_release_dir,
        verification_result_failed,
        validation_result_failed,
        temp_dir
    ):
        """Test lenient mode allows publication with warnings."""
        orchestrator = PublisherOrchestrator(
            repository=local_repo,
            policy_mode=PublicationPolicy.LENIENT,
            require_signatures=False
        )

        report = orchestrator.publish_release(
            version="v1.0.2",
            release_dir=mock_release_dir,
            verification_result=verification_result_failed,
            validation_result=validation_result_failed,
            audit_output_dir=temp_dir
        )

        assert report.status == "success"  # Lenient allows
        assert len(report.warnings) > 0

    def test_publish_release_duplicate_version(
        self,
        local_repo,
        mock_release_dir,
        verification_result_passed,
        validation_result_passed
    ):
        """Test duplicate version is rejected."""
        orchestrator = PublisherOrchestrator(
            repository=local_repo,
            policy_mode=PublicationPolicy.STRICT
        )

        # First publication succeeds
        report1 = orchestrator.publish_release(
            version="v1.0.2",
            release_dir=mock_release_dir,
            verification_result=verification_result_passed,
            validation_result=validation_result_passed
        )
        assert report1.status == "success"

        # Second publication fails
        report2 = orchestrator.publish_release(
            version="v1.0.2",
            release_dir=mock_release_dir,
            verification_result=verification_result_passed,
            validation_result=validation_result_passed
        )
        assert report2.status == "failed"
        assert report2.exit_code == 32  # DuplicateVersionError

    def test_generate_json_report(self, local_repo, temp_dir):
        """Test JSON report generation."""
        orchestrator = PublisherOrchestrator(repository=local_repo)
        report = PublicationReport(
            version="v1.0.2",
            status="success",
            timestamp=datetime.now(timezone.utc).isoformat(),
            repository_type="local",
            repository_location="/repo",
            policy_mode="strict",
            exit_code=0,
            summary="Test report"
        )

        output_file = temp_dir / "report.json"
        success = orchestrator.generate_json_report(report, output_file)

        assert success
        assert output_file.exists()

        with open(output_file, 'r') as f:
            data = json.load(f)
            assert data["version"] == "v1.0.2"
            assert data["status"] == "success"

    def test_generate_text_report(self, local_repo, temp_dir):
        """Test text report generation."""
        orchestrator = PublisherOrchestrator(repository=local_repo)
        report = PublicationReport(
            version="v1.0.2",
            status="success",
            timestamp=datetime.now(timezone.utc).isoformat(),
            repository_type="local",
            repository_location="/repo",
            policy_mode="strict",
            exit_code=0,
            summary="Test completed successfully"
        )

        output_file = temp_dir / "report.txt"
        success = orchestrator.generate_text_report(report, output_file)

        assert success
        assert output_file.exists()

        content = output_file.read_text()
        assert "v1.0.2" in content
        assert "SUCCESS" in content.upper()


# ============================================================================
# TEST: CLI INTEGRATION
# ============================================================================

class TestCLIIntegration:
    """Test CLI integration."""

    def test_cli_basic_invocation(
        self,
        mock_release_dir,
        temp_dir,
        verification_result_passed,
        validation_result_passed
    ):
        """Test basic CLI invocation."""
        # Save verification/validation results
        ver_file = temp_dir / "verification.json"
        with open(ver_file, 'w') as f:
            json.dump({
                "passed": True,
                "timestamp": verification_result_passed.timestamp,
                "hash_verified": True,
                "signature_verified": True,
                "sbom_validated": True,
                "slsa_validated": True,
                "policy_passed": True,
                "exit_code": 0
            }, f)

        val_file = temp_dir / "validation.json"
        with open(val_file, 'w') as f:
            json.dump({
                "passed": True,
                "timestamp": validation_result_passed.timestamp,
                "sbom_delta_passed": True,
                "slsa_delta_passed": True,
                "api_compat_passed": True,
                "performance_passed": True,
                "security_passed": True,
                "behavioral_passed": True,
                "exit_code": 0
            }, f)

        # Import and run main
        from publisher.release_publisher import main
        import sys

        repo_path = temp_dir / "repository"
        json_report = temp_dir / "publication.json"
        text_report = temp_dir / "publication.txt"

        sys.argv = [
            "release_publisher.py",
            "--version", "v1.0.2",
            "--release-dir", str(mock_release_dir),
            "--repository-type", "local",
            "--repository-path", str(repo_path),
            "--policy", "strict",
            "--verification-result", str(ver_file),
            "--validation-result", str(val_file),
            "--json-report", str(json_report),
            "--text-report", str(text_report),
        ]

        exit_code = main()

        assert exit_code == 0
        assert json_report.exists()
        assert text_report.exists()

    def test_cli_with_s3_repository(self, mock_release_dir, temp_dir):
        """Test CLI with S3 repository."""
        from publisher.release_publisher import main
        import sys

        json_report = temp_dir / "publication.json"

        sys.argv = [
            "release_publisher.py",
            "--version", "v1.0.2",
            "--release-dir", str(mock_release_dir),
            "--repository-type", "s3",
            "--repository-bucket", "test-bucket",
            "--repository-prefix", "releases",
            "--policy", "lenient",
            "--json-report", str(json_report),
        ]

        # Lenient mode should succeed even without verification/validation
        exit_code = main()

        # Should complete (may pass or fail based on policy)
        assert json_report.exists()


# ============================================================================
# TEST: END-TO-END WORKFLOW
# ============================================================================

class TestEndToEndWorkflow:
    """Test complete end-to-end publication workflow."""

    def test_full_publication_workflow(
        self,
        temp_dir,
        mock_release_dir,
        verification_result_passed,
        validation_result_passed
    ):
        """Test complete publication workflow."""
        # 1. Create repository
        repo_config = {"type": "local", "path": str(temp_dir / "repository")}
        repository = RepositoryFactory.create("local", repo_config)

        # 2. Create orchestrator
        orchestrator = PublisherOrchestrator(
            repository=repository,
            policy_mode=PublicationPolicy.STRICT,
            sign_audit_logs=True,
            require_signatures=True,
            require_encryption=False
        )

        # 3. Publish release
        report = orchestrator.publish_release(
            version="v1.0.2",
            release_dir=mock_release_dir,
            verification_result=verification_result_passed,
            validation_result=validation_result_passed,
            audit_output_dir=temp_dir / "audit"
        )

        # 4. Verify publication report
        assert report.status == "success"
        assert report.exit_code == 0
        assert report.verification_passed
        assert report.validation_passed
        assert report.metadata_complete
        assert report.signatures_present
        assert len(report.artifacts_published) > 0
        assert report.total_size_bytes > 0
        assert report.index_updated
        assert report.audit_log_created
        assert report.audit_log_signed
        assert report.publication_duration_seconds > 0
        assert report.publication_duration_seconds < 5.0  # Performance target

        # 5. Verify artifacts in repository
        assert repository.exists("v1.0.2/manifest.json")
        assert repository.exists("v1.0.2/sbom/sbom.json")
        assert repository.exists("v1.0.2/slsa/provenance.json")

        # 6. Verify index
        index = repository.get_index()
        assert len(index["releases"]) == 1
        assert index["releases"][0]["version"] == "v1.0.2"
        assert index["releases"][0]["signed"] is True

        # 7. Verify audit log
        audit_file = temp_dir / "audit/v1.0.2.audit.json"
        assert audit_file.exists()
        audit_sig = temp_dir / "audit/v1.0.2.audit.sig"
        assert audit_sig.exists()

        # 8. Generate reports
        json_report = temp_dir / "publication.json"
        text_report = temp_dir / "publication.txt"

        assert orchestrator.generate_json_report(report, json_report)
        assert orchestrator.generate_text_report(report, text_report)

        assert json_report.exists()
        assert text_report.exists()

        # 9. Verify report contents
        with open(json_report, 'r') as f:
            report_data = json.load(f)
            assert report_data["version"] == "v1.0.2"
            assert report_data["status"] == "success"
            assert report_data["exit_code"] == 0

        report_text = text_report.read_text()
        assert "v1.0.2" in report_text
        assert "SUCCESS" in report_text.upper()


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

class TestPerformance:
    """Test performance requirements."""

    def test_publication_completes_under_5_seconds(
        self,
        local_repo,
        mock_release_dir,
        verification_result_passed,
        validation_result_passed
    ):
        """Test publication completes in < 5 seconds."""
        import time

        orchestrator = PublisherOrchestrator(
            repository=local_repo,
            policy_mode=PublicationPolicy.STRICT
        )

        start_time = time.time()

        report = orchestrator.publish_release(
            version="v1.0.2",
            release_dir=mock_release_dir,
            verification_result=verification_result_passed,
            validation_result=validation_result_passed
        )

        elapsed = time.time() - start_time

        assert report.status == "success"
        assert elapsed < 5.0  # Must complete in < 5 seconds
        assert report.publication_duration_seconds < 5.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
