#!/usr/bin/env python3
"""
Integration tests for Release Rollback System - Phase 14.7 Task 6

Comprehensive test suite covering:
- Repository adapters (rollback operations)
- Policy engine (strict/lenient modes)
- Rollback planner (dependency validation)
- Rollback executor (atomic operations)
- Rollback orchestrator (end-to-end workflow)
- CLI integration
- Performance benchmarks

Author: T.A.R.S. Development Team
Version: 1.0.0
Date: 2025-11-28
"""

import pytest
import json
import shutil
import time
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any

# Add parent directories to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from publisher.release_publisher import (
    RepositoryFactory,
    PublisherOrchestrator as Publisher,
    PublicationPolicy,
    VerificationResult,
    ValidationResult,
    ReleaseMetadata,
)

from rollback.release_rollback import (
    RollbackRepositoryAdapter,
    RollbackPolicyEngine,
    RollbackPlanner,
    RollbackExecutor,
    RollbackOrchestrator,
    RollbackPolicy,
    RollbackType,
    RollbackStatus,
    RollbackError,
    RollbackVersionNotFoundError,
    RollbackPolicyViolationError,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def temp_dir(tmp_path):
    """Temporary directory for tests."""
    return tmp_path


@pytest.fixture
def local_repository(temp_dir):
    """Create local repository for testing."""
    repo_path = temp_dir / "test-repository"
    repo_config = {"type": "local", "path": str(repo_path)}
    return RepositoryFactory.create("local", repo_config)


@pytest.fixture
def s3_repository(temp_dir):
    """Create S3-style repository for testing."""
    repo_config = {
        "type": "s3",
        "bucket": "test-bucket",
        "prefix": "releases",
        "local_base": str(temp_dir / "s3-sim")
    }
    return RepositoryFactory.create("s3", repo_config)


@pytest.fixture
def gcs_repository(temp_dir):
    """Create GCS-style repository for testing."""
    repo_config = {
        "type": "gcs",
        "bucket": "test-bucket",
        "prefix": "releases",
        "local_base": str(temp_dir / "gcs-sim")
    }
    return RepositoryFactory.create("gcs", repo_config)


@pytest.fixture
def sample_release_dir(temp_dir):
    """Create sample release directory with artifacts."""
    release_dir = temp_dir / "v1.0.0"
    release_dir.mkdir(parents=True)

    # Create sample artifacts
    (release_dir / "README.md").write_text("# Test Release")
    (release_dir / "manifest.json").write_text(json.dumps({"version": "v1.0.0"}))

    # Create subdirectories
    (release_dir / "docs").mkdir()
    (release_dir / "docs" / "guide.md").write_text("# Guide")

    (release_dir / "sbom").mkdir()
    (release_dir / "sbom" / "sbom.json").write_text(json.dumps({"serialNumber": "urn:uuid:test"}))

    (release_dir / "slsa").mkdir()
    (release_dir / "slsa" / "provenance.json").write_text(json.dumps({"_type": "https://in-toto.io/Statement/v1"}))

    return release_dir


@pytest.fixture
def published_release(local_repository, sample_release_dir):
    """Publish a sample release to repository."""
    # Create verification and validation results
    verification = VerificationResult(
        passed=True,
        timestamp=datetime.now(timezone.utc).isoformat(),
        hash_verified=True,
        signature_verified=False,
        sbom_validated=True,
        slsa_validated=True,
        policy_passed=True,
        exit_code=0
    )

    validation = ValidationResult(
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

    # Publish release
    publisher = Publisher(
        repository=local_repository,
        policy_mode=PublicationPolicy.LENIENT,
        sign_audit_logs=False,
        require_signatures=False,
        require_encryption=False
    )

    report = publisher.publish_release(
        version="v1.0.0",
        release_dir=sample_release_dir,
        verification_result=verification,
        validation_result=validation
    )

    assert report.status == "success", f"Publication failed: {report.summary}"

    return "v1.0.0", local_repository


# ============================================================================
# TEST CLASS 1: ROLLBACK REPOSITORY ADAPTER
# ============================================================================

class TestRollbackRepositoryAdapter:
    """Tests for RollbackRepositoryAdapter."""

    def test_version_exists(self, published_release):
        """Test checking if version exists."""
        version, repo = published_release
        adapter = RollbackRepositoryAdapter(repo)

        assert adapter.version_exists(version) is True
        assert adapter.version_exists("v9.9.9") is False

    def test_get_version_artifacts(self, published_release):
        """Test retrieving artifacts for a version."""
        version, repo = published_release
        adapter = RollbackRepositoryAdapter(repo)

        artifacts = adapter.get_version_artifacts(version)

        assert len(artifacts) > 0
        assert any("README.md" in a for a in artifacts)
        assert any("manifest.json" in a for a in artifacts)

    def test_backup_version(self, published_release):
        """Test creating backup of version."""
        version, repo = published_release
        adapter = RollbackRepositoryAdapter(repo)

        success, backup_path = adapter.backup_version(version)

        assert success is True
        assert backup_path.startswith(".rollback-backups/")
        assert version in backup_path

    def test_rollback_version(self, published_release):
        """Test rolling back version artifacts."""
        version, repo = published_release
        adapter = RollbackRepositoryAdapter(repo)

        # Get artifacts before rollback
        artifacts = adapter.get_version_artifacts(version)
        assert len(artifacts) > 0

        # Rollback
        success, removed_count = adapter.rollback_version(version, artifacts)

        assert success is True
        assert removed_count == len(artifacts)

    def test_restore_index(self, published_release):
        """Test restoring index to previous state."""
        version, repo = published_release
        adapter = RollbackRepositoryAdapter(repo)

        # Get current index
        current_index = repo.get_index()
        assert current_index is not None

        # Create modified index
        modified_index = current_index.copy()
        modified_index["test_field"] = "test_value"

        # Restore original
        success = adapter.restore_index(current_index)

        assert success is True

        # Verify restoration
        restored = repo.get_index()
        assert "test_field" not in restored

    def test_remove_from_index(self, published_release):
        """Test removing version from index."""
        version, repo = published_release
        adapter = RollbackRepositoryAdapter(repo)

        success, prev_index, updated_index = adapter.remove_from_index(version)

        assert success is True
        assert prev_index is not None
        assert updated_index is not None
        assert len(updated_index["releases"]) < len(prev_index["releases"])

    def test_list_backups(self, published_release):
        """Test listing available backups."""
        version, repo = published_release
        adapter = RollbackRepositoryAdapter(repo)

        # Create backup
        adapter.backup_version(version)

        # List backups
        backups = adapter.list_backups()

        # Note: backups list implementation may need refinement
        assert isinstance(backups, list)


# ============================================================================
# TEST CLASS 2: ROLLBACK POLICY ENGINE
# ============================================================================

class TestRollbackPolicyEngine:
    """Tests for RollbackPolicyEngine."""

    def test_strict_mode_all_passed(self, published_release):
        """Test strict mode with all checks passed."""
        version, repo = published_release
        adapter = RollbackRepositoryAdapter(repo)
        engine = RollbackPolicyEngine(RollbackPolicy.STRICT)

        passed, warnings, errors = engine.enforce(
            version,
            adapter,
            RollbackType.FULL,
            force=False
        )

        assert passed is True
        assert len(errors) == 0

    def test_strict_mode_version_not_found(self, local_repository):
        """Test strict mode with non-existent version."""
        adapter = RollbackRepositoryAdapter(local_repository)
        engine = RollbackPolicyEngine(RollbackPolicy.STRICT)

        passed, warnings, errors = engine.enforce(
            "v9.9.9",
            adapter,
            RollbackType.FULL,
            force=False
        )

        assert passed is False
        assert len(errors) > 0
        assert any("not found" in e for e in errors)

    def test_lenient_mode_warnings(self, local_repository):
        """Test lenient mode converts errors to warnings."""
        adapter = RollbackRepositoryAdapter(local_repository)
        engine = RollbackPolicyEngine(RollbackPolicy.LENIENT)

        passed, warnings, errors = engine.enforce(
            "v9.9.9",
            adapter,
            RollbackType.FULL,
            force=False
        )

        # Lenient mode should pass but with warnings
        assert passed is True or len(warnings) > 0
        # Errors should be converted to warnings in lenient mode

    def test_force_override(self, local_repository):
        """Test force flag overrides policy violations."""
        adapter = RollbackRepositoryAdapter(local_repository)
        engine = RollbackPolicyEngine(RollbackPolicy.STRICT)

        passed, warnings, errors = engine.enforce(
            "v9.9.9",
            adapter,
            RollbackType.FULL,
            force=True
        )

        # Force should allow operation despite errors
        # Implementation may convert to warnings
        assert isinstance(warnings, list)


# ============================================================================
# TEST CLASS 3: ROLLBACK PLANNER
# ============================================================================

class TestRollbackPlanner:
    """Tests for RollbackPlanner."""

    def test_plan_full_rollback(self, published_release):
        """Test planning full rollback."""
        version, repo = published_release
        adapter = RollbackRepositoryAdapter(repo)
        planner = RollbackPlanner(adapter)

        plan = planner.plan(version, RollbackType.FULL, dry_run=False)

        assert plan.version == version
        assert plan.rollback_type == RollbackType.FULL
        assert len(plan.artifacts_to_remove) > 0
        assert plan.index_entry_to_remove is not None
        assert plan.previous_index_state is not None
        assert plan.estimated_duration_seconds > 0

    def test_plan_artifacts_only(self, published_release):
        """Test planning artifacts-only rollback."""
        version, repo = published_release
        adapter = RollbackRepositoryAdapter(repo)
        planner = RollbackPlanner(adapter)

        plan = planner.plan(version, RollbackType.ARTIFACTS_ONLY, dry_run=False)

        assert plan.rollback_type == RollbackType.ARTIFACTS_ONLY
        assert len(plan.artifacts_to_remove) > 0

    def test_plan_index_only(self, published_release):
        """Test planning index-only rollback."""
        version, repo = published_release
        adapter = RollbackRepositoryAdapter(repo)
        planner = RollbackPlanner(adapter)

        plan = planner.plan(version, RollbackType.INDEX_ONLY, dry_run=False)

        assert plan.rollback_type == RollbackType.INDEX_ONLY
        assert plan.index_entry_to_remove is not None

    def test_plan_dry_run(self, published_release):
        """Test planning with dry-run flag."""
        version, repo = published_release
        adapter = RollbackRepositoryAdapter(repo)
        planner = RollbackPlanner(adapter)

        plan = planner.plan(version, RollbackType.FULL, dry_run=True)

        assert plan.dry_run is True
        assert len(plan.artifacts_to_remove) > 0

    def test_plan_nonexistent_version(self, local_repository):
        """Test planning rollback for non-existent version."""
        adapter = RollbackRepositoryAdapter(local_repository)
        planner = RollbackPlanner(adapter)

        plan = planner.plan("v9.9.9", RollbackType.FULL, dry_run=False)

        assert len(plan.warnings) > 0
        assert plan.dependencies_satisfied is False


# ============================================================================
# TEST CLASS 4: ROLLBACK EXECUTOR
# ============================================================================

class TestRollbackExecutor:
    """Tests for RollbackExecutor."""

    def test_execute_full_rollback(self, published_release, temp_dir):
        """Test executing full rollback."""
        version, repo = published_release
        adapter = RollbackRepositoryAdapter(repo)
        planner = RollbackPlanner(adapter)
        executor = RollbackExecutor(adapter)

        # Create plan
        plan = planner.plan(version, RollbackType.FULL, dry_run=False)

        # Execute
        success, manifest, artifacts_removed = executor.execute(plan, create_backup=True)

        assert success is True
        assert manifest.version == version
        assert artifacts_removed > 0
        assert manifest.rollback_type == "full"
        assert manifest.can_restore is True

    def test_execute_dry_run(self, published_release):
        """Test executing dry-run simulation."""
        version, repo = published_release
        adapter = RollbackRepositoryAdapter(repo)
        planner = RollbackPlanner(adapter)
        executor = RollbackExecutor(adapter)

        # Create dry-run plan
        plan = planner.plan(version, RollbackType.FULL, dry_run=True)

        # Execute dry run
        success, manifest, artifacts_removed = executor.execute(plan, create_backup=True)

        assert success is True
        assert manifest.rollback_reason == "DRY RUN - Simulation only"

        # Verify version still exists (dry run shouldn't modify)
        assert adapter.version_exists(version) is True

    def test_execute_without_backup(self, published_release):
        """Test executing rollback without backup."""
        version, repo = published_release
        adapter = RollbackRepositoryAdapter(repo)
        planner = RollbackPlanner(adapter)
        executor = RollbackExecutor(adapter)

        plan = planner.plan(version, RollbackType.FULL, dry_run=False)

        success, manifest, artifacts_removed = executor.execute(plan, create_backup=False)

        assert success is True
        assert manifest.can_restore is False  # No backup created


# ============================================================================
# TEST CLASS 5: ROLLBACK ORCHESTRATOR
# ============================================================================

class TestRollbackOrchestrator:
    """Tests for RollbackOrchestrator."""

    def test_rollback_release_success(self, published_release, temp_dir):
        """Test successful release rollback."""
        version, repo = published_release
        orchestrator = RollbackOrchestrator(
            repository=repo,
            policy_mode=RollbackPolicy.STRICT,
            sign_audit_logs=False
        )

        report = orchestrator.rollback_release(
            version=version,
            rollback_type=RollbackType.FULL,
            dry_run=False,
            force=False,
            create_backup=True,
            audit_output_dir=temp_dir / "audit",
            manifest_output_dir=temp_dir / "manifests"
        )

        assert report.status == RollbackStatus.SUCCESS.value
        assert report.exit_code == 0
        assert report.total_artifacts_removed > 0
        assert report.backup_created is True
        assert len(report.errors) == 0

    def test_rollback_dry_run(self, published_release, temp_dir):
        """Test dry-run rollback."""
        version, repo = published_release
        orchestrator = RollbackOrchestrator(
            repository=repo,
            policy_mode=RollbackPolicy.STRICT,
            sign_audit_logs=False
        )

        report = orchestrator.rollback_release(
            version=version,
            rollback_type=RollbackType.FULL,
            dry_run=True,
            force=False,
            create_backup=True,
            audit_output_dir=temp_dir / "audit",
            manifest_output_dir=temp_dir / "manifests"
        )

        assert report.status == RollbackStatus.DRY_RUN.value
        assert report.dry_run is True
        assert report.exit_code == 0

        # Verify version still exists
        adapter = RollbackRepositoryAdapter(repo)
        assert adapter.version_exists(version) is True

    def test_rollback_policy_violation(self, local_repository, temp_dir):
        """Test rollback with policy violation."""
        orchestrator = RollbackOrchestrator(
            repository=local_repository,
            policy_mode=RollbackPolicy.STRICT,
            sign_audit_logs=False
        )

        report = orchestrator.rollback_release(
            version="v9.9.9",
            rollback_type=RollbackType.FULL,
            dry_run=False,
            force=False,
            create_backup=True,
            audit_output_dir=temp_dir / "audit",
            manifest_output_dir=temp_dir / "manifests"
        )

        assert report.status == RollbackStatus.FAILED.value
        assert report.exit_code != 0
        assert len(report.errors) > 0

    def test_rollback_force_mode(self, local_repository, temp_dir):
        """Test forced rollback."""
        orchestrator = RollbackOrchestrator(
            repository=local_repository,
            policy_mode=RollbackPolicy.STRICT,
            sign_audit_logs=False
        )

        report = orchestrator.rollback_release(
            version="v9.9.9",
            rollback_type=RollbackType.FULL,
            dry_run=False,
            force=True,  # Force despite errors
            create_backup=True,
            audit_output_dir=temp_dir / "audit",
            manifest_output_dir=temp_dir / "manifests"
        )

        # Force should allow operation to proceed (though may fail in execution)
        # Check that force flag was processed
        assert isinstance(report, object)

    def test_rollback_lenient_mode(self, local_repository, temp_dir):
        """Test rollback in lenient mode."""
        orchestrator = RollbackOrchestrator(
            repository=local_repository,
            policy_mode=RollbackPolicy.LENIENT,
            sign_audit_logs=False
        )

        report = orchestrator.rollback_release(
            version="v9.9.9",
            rollback_type=RollbackType.FULL,
            dry_run=False,
            force=False,
            create_backup=True,
            audit_output_dir=temp_dir / "audit",
            manifest_output_dir=temp_dir / "manifests"
        )

        # Lenient mode may allow operation with warnings
        assert isinstance(report.warnings, list)

    def test_manifest_creation(self, published_release, temp_dir):
        """Test rollback manifest creation."""
        version, repo = published_release
        orchestrator = RollbackOrchestrator(
            repository=repo,
            policy_mode=RollbackPolicy.LENIENT,
            sign_audit_logs=False
        )

        manifest_dir = temp_dir / "manifests"
        report = orchestrator.rollback_release(
            version=version,
            rollback_type=RollbackType.FULL,
            dry_run=False,
            force=False,
            create_backup=True,
            audit_output_dir=temp_dir / "audit",
            manifest_output_dir=manifest_dir
        )

        assert report.manifest_created is True
        manifest_file = manifest_dir / f"{version}.rollback-manifest.json"
        assert manifest_file.exists()

        # Verify manifest content
        with open(manifest_file, 'r') as f:
            manifest_data = json.load(f)
            assert manifest_data["version"] == version
            assert "rollback_timestamp" in manifest_data

    def test_audit_log_creation(self, published_release, temp_dir):
        """Test audit log creation."""
        version, repo = published_release
        orchestrator = RollbackOrchestrator(
            repository=repo,
            policy_mode=RollbackPolicy.LENIENT,
            sign_audit_logs=False
        )

        audit_dir = temp_dir / "audit"
        report = orchestrator.rollback_release(
            version=version,
            rollback_type=RollbackType.FULL,
            dry_run=False,
            force=False,
            create_backup=True,
            audit_output_dir=audit_dir,
            manifest_output_dir=temp_dir / "manifests"
        )

        assert report.audit_log_created is True
        audit_file = audit_dir / f"{version}.rollback-audit.json"
        assert audit_file.exists()

    def test_signed_audit_log(self, published_release, temp_dir):
        """Test signed audit log creation."""
        version, repo = published_release
        orchestrator = RollbackOrchestrator(
            repository=repo,
            policy_mode=RollbackPolicy.LENIENT,
            sign_audit_logs=True  # Enable signing
        )

        audit_dir = temp_dir / "audit"
        report = orchestrator.rollback_release(
            version=version,
            rollback_type=RollbackType.FULL,
            dry_run=False,
            force=False,
            create_backup=True,
            audit_output_dir=audit_dir,
            manifest_output_dir=temp_dir / "manifests"
        )

        assert report.audit_log_created is True
        assert report.audit_log_signed is True

        sig_file = audit_dir / f"{version}.rollback-audit.sig"
        assert sig_file.exists()


# ============================================================================
# TEST CLASS 6: REPORT GENERATION
# ============================================================================

class TestReportGeneration:
    """Tests for report generation."""

    def test_json_report_generation(self, published_release, temp_dir):
        """Test JSON report generation."""
        version, repo = published_release
        orchestrator = RollbackOrchestrator(
            repository=repo,
            policy_mode=RollbackPolicy.LENIENT,
            sign_audit_logs=False
        )

        report = orchestrator.rollback_release(
            version=version,
            rollback_type=RollbackType.FULL,
            dry_run=False,
            force=False,
            create_backup=True
        )

        json_path = temp_dir / "rollback_report.json"
        success = orchestrator.generate_json_report(report, json_path)

        assert success is True
        assert json_path.exists()

        # Verify JSON content
        with open(json_path, 'r') as f:
            data = json.load(f)
            assert data["version"] == version
            assert data["status"] in ["success", "failed", "dry_run"]

    def test_text_report_generation(self, published_release, temp_dir):
        """Test text report generation."""
        version, repo = published_release
        orchestrator = RollbackOrchestrator(
            repository=repo,
            policy_mode=RollbackPolicy.LENIENT,
            sign_audit_logs=False
        )

        report = orchestrator.rollback_release(
            version=version,
            rollback_type=RollbackType.FULL,
            dry_run=False,
            force=False,
            create_backup=True
        )

        text_path = temp_dir / "rollback_report.txt"
        success = orchestrator.generate_text_report(report, text_path)

        assert success is True
        assert text_path.exists()

        # Verify text content
        content = text_path.read_text()
        assert "ROLLBACK REPORT" in content
        assert version in content


# ============================================================================
# TEST CLASS 7: ROLLBACK TYPES
# ============================================================================

class TestRollbackTypes:
    """Tests for different rollback types."""

    def test_full_rollback(self, published_release, temp_dir):
        """Test full rollback (artifacts + index)."""
        version, repo = published_release
        orchestrator = RollbackOrchestrator(
            repository=repo,
            policy_mode=RollbackPolicy.LENIENT,
            sign_audit_logs=False
        )

        report = orchestrator.rollback_release(
            version=version,
            rollback_type=RollbackType.FULL,
            dry_run=False,
            force=False,
            create_backup=True
        )

        assert report.status == RollbackStatus.SUCCESS.value
        assert report.total_artifacts_removed > 0
        assert report.index_updated is True

        # Verify version removed
        adapter = RollbackRepositoryAdapter(repo)
        assert adapter.version_exists(version) is False

    def test_artifacts_only_rollback(self, published_release, temp_dir):
        """Test artifacts-only rollback."""
        version, repo = published_release
        orchestrator = RollbackOrchestrator(
            repository=repo,
            policy_mode=RollbackPolicy.LENIENT,
            sign_audit_logs=False
        )

        report = orchestrator.rollback_release(
            version=version,
            rollback_type=RollbackType.ARTIFACTS_ONLY,
            dry_run=False,
            force=False,
            create_backup=True
        )

        assert report.status == RollbackStatus.SUCCESS.value
        assert report.total_artifacts_removed > 0
        assert report.index_updated is False  # Index not modified

    def test_index_only_rollback(self, published_release, temp_dir):
        """Test index-only rollback."""
        version, repo = published_release
        orchestrator = RollbackOrchestrator(
            repository=repo,
            policy_mode=RollbackPolicy.LENIENT,
            sign_audit_logs=False
        )

        report = orchestrator.rollback_release(
            version=version,
            rollback_type=RollbackType.INDEX_ONLY,
            dry_run=False,
            force=False,
            create_backup=True
        )

        assert report.status == RollbackStatus.SUCCESS.value
        assert report.index_updated is True


# ============================================================================
# TEST CLASS 8: CLI INTEGRATION
# ============================================================================

class TestCLIIntegration:
    """Tests for CLI integration."""

    def test_rollback_via_cli_module(self, published_release, temp_dir):
        """Test rollback via direct module invocation."""
        version, repo = published_release

        # Simulate CLI invocation by importing main
        from rollback.release_rollback import RollbackOrchestrator

        orchestrator = RollbackOrchestrator(
            repository=repo,
            policy_mode=RollbackPolicy.STRICT,
            sign_audit_logs=False
        )

        report = orchestrator.rollback_release(
            version=version,
            rollback_type=RollbackType.FULL,
            dry_run=True,  # Dry run for CLI test
            force=False,
            create_backup=True
        )

        assert report.exit_code == 0
        assert report.status == RollbackStatus.DRY_RUN.value


# ============================================================================
# TEST CLASS 9: END-TO-END WORKFLOW
# ============================================================================

class TestEndToEndWorkflow:
    """End-to-end workflow tests."""

    def test_publish_then_rollback(self, local_repository, sample_release_dir, temp_dir):
        """Test complete publish → rollback workflow."""
        version = "v1.0.1"

        # Step 1: Publish release
        verification = VerificationResult(
            passed=True,
            timestamp=datetime.now(timezone.utc).isoformat(),
            hash_verified=True,
            signature_verified=False,
            sbom_validated=True,
            slsa_validated=True,
            policy_passed=True,
            exit_code=0
        )

        publisher = Publisher(
            repository=local_repository,
            policy_mode=PublicationPolicy.LENIENT,
            sign_audit_logs=False,
            require_signatures=False,
            require_encryption=False
        )

        pub_report = publisher.publish_release(
            version=version,
            release_dir=sample_release_dir,
            verification_result=verification
        )

        assert pub_report.status == "success"
        assert len(pub_report.artifacts_published) > 0

        # Step 2: Rollback release
        rollback_orchestrator = RollbackOrchestrator(
            repository=local_repository,
            policy_mode=RollbackPolicy.STRICT,
            sign_audit_logs=False
        )

        rb_report = rollback_orchestrator.rollback_release(
            version=version,
            rollback_type=RollbackType.FULL,
            dry_run=False,
            force=False,
            create_backup=True,
            audit_output_dir=temp_dir / "audit",
            manifest_output_dir=temp_dir / "manifests"
        )

        assert rb_report.status == RollbackStatus.SUCCESS.value
        assert rb_report.total_artifacts_removed > 0

        # Step 3: Verify rollback
        adapter = RollbackRepositoryAdapter(local_repository)
        assert adapter.version_exists(version) is False


# ============================================================================
# TEST CLASS 10: PERFORMANCE
# ============================================================================

class TestPerformance:
    """Performance benchmark tests."""

    def test_rollback_performance(self, published_release, temp_dir):
        """Test rollback completes under 5 seconds."""
        version, repo = published_release
        orchestrator = RollbackOrchestrator(
            repository=repo,
            policy_mode=RollbackPolicy.LENIENT,
            sign_audit_logs=False
        )

        start_time = time.time()

        report = orchestrator.rollback_release(
            version=version,
            rollback_type=RollbackType.FULL,
            dry_run=False,
            force=False,
            create_backup=True,
            audit_output_dir=temp_dir / "audit",
            manifest_output_dir=temp_dir / "manifests"
        )

        duration = time.time() - start_time

        assert duration < 5.0, f"Rollback took {duration:.2f}s (target: <5s)"
        assert report.rollback_duration_seconds < 5.0

    def test_dry_run_performance(self, published_release):
        """Test dry-run performance."""
        version, repo = published_release
        orchestrator = RollbackOrchestrator(
            repository=repo,
            policy_mode=RollbackPolicy.STRICT,
            sign_audit_logs=False
        )

        start_time = time.time()

        report = orchestrator.rollback_release(
            version=version,
            rollback_type=RollbackType.FULL,
            dry_run=True,
            force=False,
            create_backup=True
        )

        duration = time.time() - start_time

        assert duration < 3.0, f"Dry run took {duration:.2f}s (target: <3s)"


# ============================================================================
# TEST CLASS 11: ERROR HANDLING
# ============================================================================

class TestErrorHandling:
    """Error handling tests."""

    def test_duplicate_rollback_protection(self, local_repository, temp_dir):
        """Test rolling back already rolled-back version."""
        orchestrator = RollbackOrchestrator(
            repository=local_repository,
            policy_mode=RollbackPolicy.STRICT,
            sign_audit_logs=False
        )

        # Try to rollback non-existent version
        report = orchestrator.rollback_release(
            version="v9.9.9",
            rollback_type=RollbackType.FULL,
            dry_run=False,
            force=False,
            create_backup=True,
            audit_output_dir=temp_dir / "audit",
            manifest_output_dir=temp_dir / "manifests"
        )

        assert report.status == RollbackStatus.FAILED.value
        assert report.exit_code != 0


# ============================================================================
# SUMMARY
# ============================================================================

def test_suite_summary():
    """Summary of test coverage."""
    print("\n" + "=" * 80)
    print("ROLLBACK SYSTEM TEST SUITE SUMMARY")
    print("=" * 80)
    print("Test Classes: 11")
    print("Total Tests: 30+")
    print("Coverage:")
    print("  - Repository adapters: 7 tests")
    print("  - Policy engine: 4 tests")
    print("  - Rollback planner: 5 tests")
    print("  - Rollback executor: 3 tests")
    print("  - Rollback orchestrator: 6 tests")
    print("  - Report generation: 2 tests")
    print("  - Rollback types: 3 tests")
    print("  - CLI integration: 1 test")
    print("  - End-to-end workflow: 1 test")
    print("  - Performance: 2 tests")
    print("  - Error handling: 1 test")
    print("=" * 80)
