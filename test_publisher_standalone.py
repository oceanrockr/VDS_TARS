#!/usr/bin/env python3
"""
Standalone test runner for release publisher.
Validates core functionality without pytest dependencies.
"""

import sys
import tempfile
import shutil
import json
from pathlib import Path
from datetime import datetime, timezone

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from publisher.release_publisher import (
    RepositoryFactory,
    LocalRepository,
    S3StyleRepository,
    GCSStyleRepository,
    PublicationPolicyEngine,
    PublicationPolicy,
    AtomicPublisher,
    IndexBuilder,
    AuditLogBuilder,
    PublisherOrchestrator,
    VerificationResult,
    ValidationResult,
    ReleaseMetadata,
    DuplicateVersionError,
    PolicyViolationError,
)


def create_mock_release(temp_dir: Path, version: str = "v1.0.2"):
    """Create mock release directory."""
    release_dir = temp_dir / version
    release_dir.mkdir(parents=True)

    # Create manifest
    manifest = {
        "version": version,
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

    # Create signature
    with open(release_dir / "manifest.json.sig", 'w') as f:
        f.write("mock_signature")

    # Create artifacts
    artifacts_dir = release_dir / "artifacts"
    artifacts_dir.mkdir()
    for i in range(3):
        with open(artifacts_dir / f"artifact{i}.bin", 'wb') as f:
            f.write(b"test_data" * 100)

    return release_dir


def test_repository_factory():
    """Test repository factory."""
    print("Testing RepositoryFactory...")

    temp_dir = tempfile.mkdtemp()
    try:
        # Local repository
        config = {"type": "local", "path": temp_dir}
        repo = RepositoryFactory.create("local", config)
        assert isinstance(repo, LocalRepository)

        # S3 repository
        config = {"type": "s3", "bucket": "test", "local_base": temp_dir}
        repo = RepositoryFactory.create("s3", config)
        assert isinstance(repo, S3StyleRepository)

        # GCS repository
        config = {"type": "gcs", "bucket": "test", "local_base": temp_dir}
        repo = RepositoryFactory.create("gcs", config)
        assert isinstance(repo, GCSStyleRepository)

        print("[PASS] RepositoryFactory tests passed")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_local_repository():
    """Test local repository operations."""
    print("Testing LocalRepository...")

    temp_dir = tempfile.mkdtemp()
    try:
        config = {"type": "local", "path": temp_dir}
        repo = LocalRepository(config)

        # Test upload
        test_file = Path(temp_dir) / "test.txt"
        test_file.write_text("hello world")
        assert repo.upload(test_file, "v1.0.0/test.txt")

        # Test exists
        assert repo.exists("v1.0.0/test.txt")
        assert not repo.exists("v1.0.0/missing.txt")

        # Test duplicate rejection
        assert not repo.upload(test_file, "v1.0.0/test.txt")

        # Test download
        download_path = Path(temp_dir) / "download.txt"
        assert repo.download("v1.0.0/test.txt", download_path)
        assert download_path.read_text() == "hello world"

        # Test list versions
        (Path(temp_dir) / "v1.0.1").mkdir()
        (Path(temp_dir) / "v1.0.2").mkdir()
        versions = repo.list_versions()
        assert len(versions) >= 1

        # Test index
        index = repo.get_index()
        assert "releases" in index

        print("[PASS] LocalRepository tests passed")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_publication_policy():
    """Test publication policy engine."""
    print("Testing PublicationPolicyEngine...")

    temp_dir = tempfile.mkdtemp()
    try:
        release_dir = create_mock_release(Path(temp_dir))

        # Strict mode with passing results
        engine = PublicationPolicyEngine(PublicationPolicy.STRICT)
        verification = VerificationResult(
            passed=True,
            timestamp=datetime.now(timezone.utc).isoformat(),
            hash_verified=True,
            signature_verified=True,
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

        passed, warnings, errors = engine.enforce(
            release_dir, verification, validation,
            require_signatures=True, require_encryption=False
        )
        assert passed
        assert len(errors) == 0

        # Strict mode with failing verification
        verification_failed = VerificationResult(
            passed=False,
            timestamp=datetime.now(timezone.utc).isoformat(),
            exit_code=10
        )
        passed, warnings, errors = engine.enforce(
            release_dir, verification_failed, validation
        )
        assert not passed
        assert len(errors) > 0

        # Lenient mode
        engine_lenient = PublicationPolicyEngine(PublicationPolicy.LENIENT)
        passed, warnings, errors = engine_lenient.enforce(
            release_dir, verification_failed, validation
        )
        assert passed  # Lenient allows with warnings
        assert len(warnings) > 0

        print("[PASS] PublicationPolicyEngine tests passed")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_atomic_publisher():
    """Test atomic publisher."""
    print("Testing AtomicPublisher...")

    import uuid
    temp_base = tempfile.gettempdir()
    test_id = str(uuid.uuid4())[:8]
    temp_dir1 = Path(temp_base) / f"test-repo-{test_id}"
    temp_dir2 = Path(temp_base) / f"test-release-{test_id}"

    try:
        temp_dir1.mkdir(exist_ok=True)
        temp_dir2.mkdir(exist_ok=True)

        repo_path = temp_dir1 / "repo"
        config = {"type": "local", "path": str(repo_path)}
        repo = LocalRepository(config)

        version_id = f"v{test_id}"
        release_dir = create_mock_release(temp_dir2, version_id)
        publisher = AtomicPublisher(repo)

        metadata = ReleaseMetadata(
            version=version_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            sha256="abc123",
            size_bytes=1024
        )

        # Test successful publish
        success, published, total_bytes = publisher.publish(
            version_id, release_dir, metadata
        )
        assert success
        assert len(published) > 0
        assert total_bytes > 0

        # Test duplicate rejection
        try:
            publisher.publish(version_id, release_dir, metadata)
            assert False, "Should have raised DuplicateVersionError"
        except DuplicateVersionError:
            pass  # Expected

        print("[PASS] AtomicPublisher tests passed")
    finally:
        shutil.rmtree(temp_dir1, ignore_errors=True)
        shutil.rmtree(temp_dir2, ignore_errors=True)


def test_index_builder():
    """Test index builder."""
    print("Testing IndexBuilder...")

    temp_dir = tempfile.mkdtemp()
    try:
        config = {"type": "local", "path": temp_dir}
        repo = LocalRepository(config)
        builder = IndexBuilder(repo)

        metadata = ReleaseMetadata(
            version="v1.0.2",
            timestamp=datetime.now(timezone.utc).isoformat(),
            sha256="abc123",
            size_bytes=1024,
            sbom_id="urn:uuid:sbom-123",
            signed=True
        )

        verification = VerificationResult(
            passed=True,
            timestamp=datetime.now(timezone.utc).isoformat(),
            exit_code=0
        )

        # Update index
        success = builder.update("v1.0.2", metadata, verification, None)
        assert success

        # Verify index
        index = repo.get_index()
        assert len(index["releases"]) == 1
        assert index["releases"][0]["version"] == "v1.0.2"

        # Test duplicate rejection
        success = builder.update("v1.0.2", metadata, verification, None)
        assert not success

        print("[PASS] IndexBuilder tests passed")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_audit_log_builder():
    """Test audit log builder."""
    print("Testing AuditLogBuilder...")

    temp_dir = tempfile.mkdtemp()
    try:
        builder = AuditLogBuilder(sign_logs=True)

        metadata = ReleaseMetadata(
            version="v1.0.2",
            timestamp=datetime.now(timezone.utc).isoformat(),
            sha256="abc123",
            size_bytes=1024
        )

        log_created, log_signed = builder.generate(
            "v1.0.2", metadata, None, None, {},
            Path(temp_dir)
        )

        assert log_created
        assert log_signed

        # Verify files
        audit_file = Path(temp_dir) / "v1.0.2.audit.json"
        assert audit_file.exists(), f"Audit file not found: {audit_file}"

        # Check for signature file (with correct extension)
        sig_files = list(Path(temp_dir).glob("v1.0.2.*.sig"))
        assert len(sig_files) > 0, f"No signature files found in {temp_dir}, files: {list(Path(temp_dir).iterdir())}"

        print("[PASS] AuditLogBuilder tests passed")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_publisher_orchestrator():
    """Test publisher orchestrator (end-to-end)."""
    print("Testing PublisherOrchestrator...")

    temp_dir = tempfile.mkdtemp()
    try:
        repo_path = Path(temp_dir) / "repo"
        config = {"type": "local", "path": str(repo_path)}
        repo = LocalRepository(config)

        release_dir = create_mock_release(Path(temp_dir))

        verification = VerificationResult(
            passed=True,
            timestamp=datetime.now(timezone.utc).isoformat(),
            hash_verified=True,
            signature_verified=True,
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

        orchestrator = PublisherOrchestrator(
            repository=repo,
            policy_mode=PublicationPolicy.STRICT,
            sign_audit_logs=True
        )

        # Publish release
        report = orchestrator.publish_release(
            version="v1.0.2",
            release_dir=release_dir,
            verification_result=verification,
            validation_result=validation,
            audit_output_dir=Path(temp_dir) / "audit"
        )

        # Verify report
        assert report.status == "success"
        assert report.exit_code == 0
        assert report.verification_passed
        assert report.validation_passed
        assert len(report.artifacts_published) > 0
        assert report.index_updated
        assert report.audit_log_created
        assert report.publication_duration_seconds < 5.0

        # Generate reports
        json_report = Path(temp_dir) / "report.json"
        text_report = Path(temp_dir) / "report.txt"

        assert orchestrator.generate_json_report(report, json_report)
        assert orchestrator.generate_text_report(report, text_report)

        assert json_report.exists()
        assert text_report.exists()

        print("[PASS] PublisherOrchestrator tests passed")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    """Run all tests."""
    print("=" * 80)
    print("RELEASE PUBLISHER STANDALONE TEST SUITE")
    print("=" * 80)
    print()

    tests = [
        test_repository_factory,
        test_local_repository,
        test_publication_policy,
        test_atomic_publisher,
        test_index_builder,
        test_audit_log_builder,
        test_publisher_orchestrator,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"[FAIL] {test.__name__} failed: {e}")
            failed += 1
        except Exception as e:
            print(f"[ERROR] {test.__name__} error: {e}")
            failed += 1

    print()
    print("=" * 80)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 80)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
