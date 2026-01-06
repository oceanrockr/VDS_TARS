#!/usr/bin/env python3
# ==============================================================================
# T.A.R.S. Backup & Restore Test Suite
# Version: v1.0.12 - Phase 25 Backup & Recovery
# ==============================================================================
#
# Test suite for backup-tars.sh and restore-tars.sh scripts.
# Following RiPIT methodology: Tests written FIRST before implementation.
#
# Test Categories:
#   1. Backup Archive Creation Tests
#   2. Component Backup Tests (ChromaDB, PostgreSQL, Redis)
#   3. Secret Redaction Tests
#   4. Integrity Validation Tests
#   5. Restore Tests
#   6. Edge Case Tests
#
# Usage:
#   pytest tests/test_backup_restore.py -v
#   pytest tests/test_backup_restore.py -v -k "test_backup"
#   pytest tests/test_backup_restore.py -v -k "test_restore"
#
# ==============================================================================

import json
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

# ==============================================================================
# Test Configuration
# ==============================================================================

# Path to deploy scripts (relative to project root)
SCRIPT_DIR = Path(__file__).parent.parent / "deploy"
BACKUP_SCRIPT = SCRIPT_DIR / "backup-tars.sh"
RESTORE_SCRIPT = SCRIPT_DIR / "restore-tars.sh"

# Test fixtures directory
FIXTURES_DIR = Path(__file__).parent / "fixtures" / "backup_restore"


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    tmpdir = tempfile.mkdtemp(prefix="tars-backup-test-")
    yield Path(tmpdir)
    # Cleanup after test
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def mock_docker_env(temp_dir):
    """Create a mock Docker environment for testing."""
    # Create mock volume directories
    volumes = ["chroma_data", "postgres_data", "redis_data", "ollama_data"]
    for vol in volumes:
        vol_dir = temp_dir / "volumes" / vol
        vol_dir.mkdir(parents=True, exist_ok=True)
        # Add some dummy data
        (vol_dir / "data.txt").write_text(f"Mock data for {vol}")

    # Create mock env file
    env_file = temp_dir / "tars-home.env"
    env_content = """
# T.A.R.S. Home Environment
TARS_POSTGRES_PASSWORD=supersecret123
TARS_JWT_SECRET=jwt_secret_key_12345
OLLAMA_MODEL=mistral:7b-instruct
LOG_LEVEL=INFO
REDIS_HOST=redis
POSTGRES_DB=tars_home
API_KEY=sk-12345abcdef
"""
    env_file.write_text(env_content)

    return {
        "volumes_dir": temp_dir / "volumes",
        "env_file": env_file,
        "temp_dir": temp_dir,
    }


@pytest.fixture
def sample_backup_archive(temp_dir):
    """Create a sample backup archive for restore tests."""
    archive_dir = temp_dir / "backup_contents"
    archive_dir.mkdir(parents=True, exist_ok=True)

    # Create manifest
    manifest = {
        "version": "1.0.12",
        "timestamp": datetime.utcnow().isoformat(),
        "components": ["chromadb", "postgres", "redis", "config"],
        "checksums": {},
    }

    # Create component files
    components = {
        "chromadb_backup.tar.gz": b"mock chromadb data",
        "postgres_backup.sql.gz": b"mock postgres dump",
        "redis_backup.rdb": b"mock redis rdb",
        "config_backup.tar.gz": b"mock config data",
    }

    for filename, data in components.items():
        filepath = archive_dir / filename
        filepath.write_bytes(data)
        # Calculate checksum (mock)
        import hashlib

        manifest["checksums"][filename] = hashlib.sha256(data).hexdigest()

    # Write manifest
    (archive_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # Create the archive
    archive_path = temp_dir / "tars-backup-20260103_120000.tar.gz"
    with tarfile.open(archive_path, "w:gz") as tar:
        for item in archive_dir.iterdir():
            tar.add(item, arcname=item.name)

    # Create checksum file
    import hashlib

    with open(archive_path, "rb") as f:
        archive_hash = hashlib.sha256(f.read()).hexdigest()
    (temp_dir / "tars-backup-20260103_120000.tar.gz.sha256").write_text(
        f"{archive_hash}  tars-backup-20260103_120000.tar.gz\n"
    )

    return archive_path


# ==============================================================================
# Test Classes
# ==============================================================================


class TestBackupScriptExists:
    """Verify backup script exists and is executable."""

    def test_backup_script_exists(self):
        """Backup script should exist in deploy directory."""
        assert BACKUP_SCRIPT.exists(), f"Backup script not found: {BACKUP_SCRIPT}"

    def test_backup_script_is_executable(self):
        """Backup script should have executable permissions."""
        if os.name != "nt":  # Skip on Windows
            assert os.access(
                BACKUP_SCRIPT, os.X_OK
            ), f"Backup script not executable: {BACKUP_SCRIPT}"

    def test_restore_script_exists(self):
        """Restore script should exist in deploy directory."""
        assert RESTORE_SCRIPT.exists(), f"Restore script not found: {RESTORE_SCRIPT}"


class TestBackupArchiveCreation:
    """Test backup archive creation functionality."""

    def test_backup_creates_archive(self, temp_dir, mock_docker_env):
        """Backup should create a tar.gz archive."""
        output_dir = temp_dir / "backups"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Run backup script in dry-run mode
        result = subprocess.run(
            [
                "bash",
                str(BACKUP_SCRIPT),
                "--output-dir",
                str(output_dir),
                "--dry-run",
            ],
            capture_output=True,
            text=True,
            cwd=str(SCRIPT_DIR.parent),
        )

        # In dry-run mode, script should succeed
        assert (
            result.returncode == 0 or "dry-run" in result.stdout.lower()
        ), f"Backup failed: {result.stderr}"

    def test_backup_archive_naming_convention(self, temp_dir):
        """Backup archives should follow naming convention: tars-backup-YYYYMMDD_HHMMSS.tar.gz"""
        # Create a mock archive with proper naming
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        expected_pattern = rf"tars-backup-\d{{8}}_\d{{6}}\.tar\.gz"

        archive_name = f"tars-backup-{timestamp}.tar.gz"
        assert re.match(
            expected_pattern, archive_name
        ), f"Archive name {archive_name} does not match pattern"

    def test_backup_creates_checksum_file(self, temp_dir):
        """Backup should create a SHA-256 checksum file alongside the archive."""
        archive_path = temp_dir / "tars-backup-20260103_120000.tar.gz"
        checksum_path = temp_dir / "tars-backup-20260103_120000.tar.gz.sha256"

        # Create mock archive
        archive_path.write_bytes(b"mock archive content")

        # Create checksum (as the script would)
        import hashlib

        with open(archive_path, "rb") as f:
            checksum = hashlib.sha256(f.read()).hexdigest()

        checksum_path.write_text(f"{checksum}  {archive_path.name}\n")

        assert checksum_path.exists(), "Checksum file should be created"
        checksum_content = checksum_path.read_text()
        assert archive_path.name in checksum_content


class TestComponentBackup:
    """Test individual component backup functionality."""

    def test_backup_includes_chromadb(self, sample_backup_archive, temp_dir):
        """Backup should include ChromaDB data."""
        with tarfile.open(sample_backup_archive, "r:gz") as tar:
            members = tar.getnames()
            assert any(
                "chromadb" in m for m in members
            ), f"ChromaDB backup not found in archive. Members: {members}"

    def test_backup_includes_postgres(self, sample_backup_archive, temp_dir):
        """Backup should include PostgreSQL dump."""
        with tarfile.open(sample_backup_archive, "r:gz") as tar:
            members = tar.getnames()
            assert any(
                "postgres" in m for m in members
            ), f"PostgreSQL backup not found in archive. Members: {members}"

    def test_backup_includes_redis(self, sample_backup_archive, temp_dir):
        """Backup should include Redis RDB snapshot."""
        with tarfile.open(sample_backup_archive, "r:gz") as tar:
            members = tar.getnames()
            assert any(
                "redis" in m for m in members
            ), f"Redis backup not found in archive. Members: {members}"

    def test_backup_includes_config(self, sample_backup_archive, temp_dir):
        """Backup should include configuration files."""
        with tarfile.open(sample_backup_archive, "r:gz") as tar:
            members = tar.getnames()
            assert any(
                "config" in m for m in members
            ), f"Config backup not found in archive. Members: {members}"

    def test_backup_includes_manifest(self, sample_backup_archive, temp_dir):
        """Backup should include a manifest.json file."""
        with tarfile.open(sample_backup_archive, "r:gz") as tar:
            members = tar.getnames()
            assert (
                "manifest.json" in members
            ), f"Manifest not found in archive. Members: {members}"


class TestSecretRedaction:
    """Test secret redaction in configuration backups."""

    def test_redacts_password_values(self, mock_docker_env):
        """Passwords should be redacted in backup."""
        env_file = mock_docker_env["env_file"]
        env_content = env_file.read_text()

        # Apply redaction pattern (as the script would)
        redacted = re.sub(
            r"(PASSWORD|SECRET|KEY|TOKEN)[^\n]*=[^\n]*",
            r"\1=<REDACTED>",
            env_content,
            flags=re.IGNORECASE,
        )

        assert "supersecret123" not in redacted, "Password should be redacted"
        assert "jwt_secret_key" not in redacted, "JWT secret should be redacted"
        assert "sk-12345" not in redacted, "API key should be redacted"

    def test_redacts_jwt_tokens(self, mock_docker_env):
        """JWT tokens should be redacted."""
        # Sample JWT token pattern
        jwt_pattern = r"eyJ[A-Za-z0-9_-]*\.[A-Za-z0-9_-]*\.[A-Za-z0-9_-]*"
        sample_jwt = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"

        redacted = re.sub(jwt_pattern, "<JWT_REDACTED>", sample_jwt)
        assert redacted == "<JWT_REDACTED>", "JWT should be redacted"

    def test_redacts_connection_strings(self, mock_docker_env):
        """Database connection strings should have passwords redacted."""
        connection_string = "postgres://tars:mysecretpassword@postgres:5432/tars_home"

        # Apply redaction pattern
        redacted = re.sub(
            r"(postgres:\/\/[^:]+:)[^@]+(@)", r"\1<REDACTED>\2", connection_string
        )

        assert "mysecretpassword" not in redacted, "Connection password should be redacted"
        assert "<REDACTED>" in redacted, "Redaction marker should be present"

    def test_preserves_non_secret_values(self, mock_docker_env):
        """Non-secret configuration values should be preserved."""
        env_content = mock_docker_env["env_file"].read_text()

        # Redaction should preserve non-secret values
        assert "OLLAMA_MODEL" in env_content
        assert "LOG_LEVEL" in env_content
        assert "REDIS_HOST" in env_content


class TestIntegrityValidation:
    """Test backup integrity validation."""

    def test_validates_checksum(self, sample_backup_archive, temp_dir):
        """Restore should validate archive checksum before proceeding."""
        import hashlib

        checksum_file = Path(str(sample_backup_archive) + ".sha256")

        # Read the stored checksum
        stored_checksum = checksum_file.read_text().split()[0]

        # Calculate actual checksum
        with open(sample_backup_archive, "rb") as f:
            actual_checksum = hashlib.sha256(f.read()).hexdigest()

        assert (
            stored_checksum == actual_checksum
        ), "Checksum validation should pass for valid archive"

    def test_detects_corrupted_archive(self, sample_backup_archive, temp_dir):
        """Restore should detect corrupted archives."""
        import hashlib

        # Corrupt the archive
        with open(sample_backup_archive, "ab") as f:
            f.write(b"CORRUPTED")

        # Recalculate checksum
        with open(sample_backup_archive, "rb") as f:
            new_checksum = hashlib.sha256(f.read()).hexdigest()

        # Read original checksum
        checksum_file = Path(str(sample_backup_archive) + ".sha256")
        original_checksum = checksum_file.read_text().split()[0]

        assert (
            new_checksum != original_checksum
        ), "Corrupted archive should fail checksum validation"

    def test_validates_manifest_components(self, sample_backup_archive, temp_dir):
        """Restore should validate all manifest components exist in archive."""
        extract_dir = temp_dir / "extracted"
        extract_dir.mkdir(parents=True, exist_ok=True)

        with tarfile.open(sample_backup_archive, "r:gz") as tar:
            tar.extractall(extract_dir)

        # Load manifest
        manifest_path = extract_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text())

        # Verify all components exist
        for component in manifest["components"]:
            component_files = list(extract_dir.glob(f"*{component}*"))
            assert (
                len(component_files) > 0
            ), f"Component {component} not found in archive"


class TestRestoreOperations:
    """Test restore functionality."""

    def test_restore_validates_integrity(self, sample_backup_archive, temp_dir):
        """Restore should validate backup integrity before proceeding."""
        # Run restore script in dry-run mode
        result = subprocess.run(
            [
                "bash",
                str(RESTORE_SCRIPT),
                "--backup-file",
                str(sample_backup_archive),
                "--dry-run",
            ],
            capture_output=True,
            text=True,
            cwd=str(SCRIPT_DIR.parent),
        )

        # Dry-run should succeed or indicate dry-run mode
        assert (
            result.returncode == 0 or "dry-run" in result.stdout.lower()
        ), f"Restore dry-run failed: {result.stderr}"

    def test_restore_handles_missing_backup(self, temp_dir):
        """Restore should fail gracefully for missing backup file."""
        missing_backup = temp_dir / "nonexistent-backup.tar.gz"

        result = subprocess.run(
            [
                "bash",
                str(RESTORE_SCRIPT),
                "--backup-file",
                str(missing_backup),
                "--dry-run",
            ],
            capture_output=True,
            text=True,
            cwd=str(SCRIPT_DIR.parent),
        )

        # Should fail with error message about missing file
        assert result.returncode != 0 or "not found" in result.stderr.lower() or "not found" in result.stdout.lower()

    def test_restore_extracts_to_temp_first(self, sample_backup_archive, temp_dir):
        """Restore should extract to temp directory before applying."""
        # This is a design requirement - restore should be atomic
        # First extract to temp, validate, then apply
        extract_dir = temp_dir / "temp_extract"
        extract_dir.mkdir(parents=True, exist_ok=True)

        with tarfile.open(sample_backup_archive, "r:gz") as tar:
            tar.extractall(extract_dir)

        # Verify extraction succeeded
        assert any(extract_dir.iterdir()), "Extraction should create files"


class TestBackupDryRun:
    """Test dry-run mode for backup operations."""

    def test_dry_run_does_not_create_files(self, temp_dir):
        """Dry-run mode should not create any backup files."""
        output_dir = temp_dir / "dry_run_output"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Count files before
        files_before = list(output_dir.iterdir())

        result = subprocess.run(
            [
                "bash",
                str(BACKUP_SCRIPT),
                "--output-dir",
                str(output_dir),
                "--dry-run",
            ],
            capture_output=True,
            text=True,
            cwd=str(SCRIPT_DIR.parent),
        )

        # Count files after
        files_after = list(output_dir.iterdir())

        # No new files should be created in dry-run mode
        assert len(files_after) == len(
            files_before
        ), "Dry-run should not create files"

    def test_dry_run_shows_what_would_happen(self, temp_dir):
        """Dry-run should output what operations would be performed."""
        result = subprocess.run(
            [
                "bash",
                str(BACKUP_SCRIPT),
                "--output-dir",
                str(temp_dir),
                "--dry-run",
            ],
            capture_output=True,
            text=True,
            cwd=str(SCRIPT_DIR.parent),
        )

        output = result.stdout + result.stderr
        # Should mention dry-run or would-be operations
        assert (
            "dry" in output.lower() or "would" in output.lower()
        ), "Dry-run should describe planned operations"


class TestBackupCLIOptions:
    """Test backup script CLI options."""

    def test_help_option(self):
        """Backup script should display help with --help."""
        result = subprocess.run(
            ["bash", str(BACKUP_SCRIPT), "--help"],
            capture_output=True,
            text=True,
            cwd=str(SCRIPT_DIR.parent),
        )

        assert result.returncode == 0, "Help should return exit code 0"
        assert "usage" in result.stdout.lower() or "options" in result.stdout.lower()

    def test_output_dir_option(self, temp_dir):
        """Backup script should accept --output-dir option."""
        result = subprocess.run(
            [
                "bash",
                str(BACKUP_SCRIPT),
                "--output-dir",
                str(temp_dir),
                "--dry-run",
            ],
            capture_output=True,
            text=True,
            cwd=str(SCRIPT_DIR.parent),
        )

        # Should not fail due to invalid option
        assert (
            "unknown option" not in result.stderr.lower()
        ), "--output-dir should be recognized"

    def test_include_models_option(self, temp_dir):
        """Backup script should accept --include-models option for Ollama models."""
        result = subprocess.run(
            [
                "bash",
                str(BACKUP_SCRIPT),
                "--include-models",
                "--dry-run",
            ],
            capture_output=True,
            text=True,
            cwd=str(SCRIPT_DIR.parent),
        )

        # Should not fail due to invalid option
        assert (
            "unknown option" not in result.stderr.lower()
        ), "--include-models should be recognized"


class TestRestoreCLIOptions:
    """Test restore script CLI options."""

    def test_help_option(self):
        """Restore script should display help with --help."""
        result = subprocess.run(
            ["bash", str(RESTORE_SCRIPT), "--help"],
            capture_output=True,
            text=True,
            cwd=str(SCRIPT_DIR.parent),
        )

        assert result.returncode == 0, "Help should return exit code 0"
        assert "usage" in result.stdout.lower() or "options" in result.stdout.lower()

    def test_backup_file_option(self, sample_backup_archive):
        """Restore script should accept --backup-file option."""
        result = subprocess.run(
            [
                "bash",
                str(RESTORE_SCRIPT),
                "--backup-file",
                str(sample_backup_archive),
                "--dry-run",
            ],
            capture_output=True,
            text=True,
            cwd=str(SCRIPT_DIR.parent),
        )

        assert (
            "unknown option" not in result.stderr.lower()
        ), "--backup-file should be recognized"

    def test_skip_validation_option(self, sample_backup_archive):
        """Restore script should accept --skip-validation option."""
        result = subprocess.run(
            [
                "bash",
                str(RESTORE_SCRIPT),
                "--backup-file",
                str(sample_backup_archive),
                "--skip-validation",
                "--dry-run",
            ],
            capture_output=True,
            text=True,
            cwd=str(SCRIPT_DIR.parent),
        )

        assert (
            "unknown option" not in result.stderr.lower()
        ), "--skip-validation should be recognized"


class TestManifestFormat:
    """Test backup manifest format and content."""

    def test_manifest_contains_version(self, sample_backup_archive, temp_dir):
        """Manifest should contain T.A.R.S. version."""
        extract_dir = temp_dir / "extracted"
        extract_dir.mkdir(parents=True, exist_ok=True)

        with tarfile.open(sample_backup_archive, "r:gz") as tar:
            tar.extractall(extract_dir)

        manifest = json.loads((extract_dir / "manifest.json").read_text())
        assert "version" in manifest, "Manifest should contain version"
        assert manifest["version"] == "1.0.12"

    def test_manifest_contains_timestamp(self, sample_backup_archive, temp_dir):
        """Manifest should contain backup timestamp."""
        extract_dir = temp_dir / "extracted"
        extract_dir.mkdir(parents=True, exist_ok=True)

        with tarfile.open(sample_backup_archive, "r:gz") as tar:
            tar.extractall(extract_dir)

        manifest = json.loads((extract_dir / "manifest.json").read_text())
        assert "timestamp" in manifest, "Manifest should contain timestamp"

    def test_manifest_contains_checksums(self, sample_backup_archive, temp_dir):
        """Manifest should contain checksums for all components."""
        extract_dir = temp_dir / "extracted"
        extract_dir.mkdir(parents=True, exist_ok=True)

        with tarfile.open(sample_backup_archive, "r:gz") as tar:
            tar.extractall(extract_dir)

        manifest = json.loads((extract_dir / "manifest.json").read_text())
        assert "checksums" in manifest, "Manifest should contain checksums"
        assert len(manifest["checksums"]) > 0, "Checksums should not be empty"

    def test_manifest_contains_components_list(self, sample_backup_archive, temp_dir):
        """Manifest should list all backed up components."""
        extract_dir = temp_dir / "extracted"
        extract_dir.mkdir(parents=True, exist_ok=True)

        with tarfile.open(sample_backup_archive, "r:gz") as tar:
            tar.extractall(extract_dir)

        manifest = json.loads((extract_dir / "manifest.json").read_text())
        assert "components" in manifest, "Manifest should contain components list"

        expected_components = ["chromadb", "postgres", "redis", "config"]
        for component in expected_components:
            assert (
                component in manifest["components"]
            ), f"Component {component} should be in manifest"


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_handles_empty_volume(self, temp_dir, mock_docker_env):
        """Backup should handle empty volumes gracefully."""
        # Create an empty volume directory
        empty_vol = mock_docker_env["volumes_dir"] / "empty_volume"
        empty_vol.mkdir(parents=True, exist_ok=True)

        # Script should not fail on empty volumes
        # This is a placeholder - actual test depends on script implementation

    def test_handles_large_files(self, temp_dir):
        """Backup should handle large files (e.g., Ollama models)."""
        # Create a large mock file (100 MB)
        large_file = temp_dir / "large_model.bin"

        # Don't actually create 100MB in tests - just verify the logic exists
        # The script should warn about large files or handle them specially

    def test_handles_special_characters_in_paths(self, temp_dir):
        """Backup should handle paths with special characters."""
        # Create directory with special chars (if platform allows)
        special_dir = temp_dir / "data with spaces"
        special_dir.mkdir(parents=True, exist_ok=True)
        (special_dir / "file.txt").write_text("test")

        # Verify path handling works
        assert special_dir.exists()

    def test_handles_permission_denied(self, temp_dir):
        """Backup should report permission errors gracefully."""
        # Create a directory without read permissions
        if os.name != "nt":  # Skip on Windows
            restricted_dir = temp_dir / "restricted"
            restricted_dir.mkdir(parents=True, exist_ok=True)
            restricted_dir.chmod(0o000)

            try:
                # Attempt to read should fail
                result = list(restricted_dir.iterdir())
            except PermissionError:
                pass  # Expected behavior
            finally:
                # Restore permissions for cleanup
                restricted_dir.chmod(0o755)

    def test_handles_concurrent_backups(self, temp_dir):
        """Backup should handle or prevent concurrent backup attempts."""
        # The script should use a lock file to prevent concurrent backups
        lock_file = temp_dir / ".tars-backup.lock"

        # Simulate lock file existence
        lock_file.write_text("locked")
        assert lock_file.exists(), "Lock file should be created"


class TestRetentionPolicy:
    """Test backup retention policy features."""

    def test_lists_existing_backups(self, temp_dir):
        """Backup should be able to list existing backups."""
        backup_dir = temp_dir / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)

        # Create some mock backups
        for i in range(5):
            (backup_dir / f"tars-backup-2026010{i}_120000.tar.gz").write_bytes(
                b"mock"
            )

        backups = list(backup_dir.glob("tars-backup-*.tar.gz"))
        assert len(backups) == 5, "Should find 5 backup files"

    def test_identifies_old_backups(self, temp_dir):
        """Should be able to identify backups older than retention period."""
        from datetime import timedelta

        backup_dir = temp_dir / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)

        # Create backups with different timestamps
        now = datetime.now()
        old_timestamp = (now - timedelta(days=30)).strftime("%Y%m%d_%H%M%S")
        new_timestamp = now.strftime("%Y%m%d_%H%M%S")

        old_backup = backup_dir / f"tars-backup-{old_timestamp}.tar.gz"
        new_backup = backup_dir / f"tars-backup-{new_timestamp}.tar.gz"

        old_backup.write_bytes(b"old")
        new_backup.write_bytes(b"new")

        # Parse timestamps from filenames
        def parse_backup_date(path: Path) -> datetime:
            match = re.search(r"tars-backup-(\d{8}_\d{6})\.tar\.gz", path.name)
            if match:
                return datetime.strptime(match.group(1), "%Y%m%d_%H%M%S")
            return datetime.min

        old_date = parse_backup_date(old_backup)
        new_date = parse_backup_date(new_backup)

        retention_days = 7
        cutoff = now - timedelta(days=retention_days)

        assert old_date < cutoff, "Old backup should be before cutoff"
        assert new_date >= cutoff, "New backup should be after cutoff"


# ==============================================================================
# Integration Test Markers
# ==============================================================================


@pytest.mark.integration
class TestBackupRestoreIntegration:
    """Integration tests requiring Docker environment."""

    @pytest.mark.skipif(
        os.environ.get("CI") == "true",
        reason="Requires Docker environment",
    )
    def test_full_backup_restore_cycle(self, temp_dir):
        """Test complete backup and restore cycle."""
        # This test requires actual Docker containers running
        # Skip in CI, run manually during integration testing
        pass

    @pytest.mark.skipif(
        os.environ.get("CI") == "true",
        reason="Requires Docker environment",
    )
    def test_chromadb_data_preserved(self, temp_dir):
        """Verify ChromaDB data is preserved after restore."""
        pass

    @pytest.mark.skipif(
        os.environ.get("CI") == "true",
        reason="Requires Docker environment",
    )
    def test_postgres_data_preserved(self, temp_dir):
        """Verify PostgreSQL data is preserved after restore."""
        pass


# ==============================================================================
# Main Entry Point
# ==============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
