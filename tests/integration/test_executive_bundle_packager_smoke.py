#!/usr/bin/env python3
"""
Smoke Test Suite for Executive Bundle Packager

This module provides smoke tests for the bundle packager:
- Synthetic run directory packaging
- Manifest generation
- Checksum creation
- Archive creation (ZIP and tar.gz)
- Exit code behavior

Version: 1.0.0
Phase: 16 - Ops Automation Hardening
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
import zipfile
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestPackagerWithSyntheticData(unittest.TestCase):
    """Test packager with synthetic run directory."""

    def setUp(self):
        """Set up synthetic test data."""
        self.test_dir = tempfile.mkdtemp()

        # Create synthetic run directory structure
        self.run_dir = Path(self.test_dir) / "tars-run-20251222-140000"
        self.run_dir.mkdir()

        # Create placeholder JSON files
        self.create_synthetic_files()

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def create_synthetic_files(self):
        """Create synthetic report files."""
        # org-health-report.json
        org_health = {
            "version": "1.0.0",
            "generated_at": "2025-12-22T14:00:00Z",
            "organization_health": {
                "aggregate_score": 85.5,
                "repository_count": 5
            }
        }
        with open(self.run_dir / "org-health-report.json", "w") as f:
            json.dump(org_health, f)

        # org-alerts.json
        alerts = {
            "version": "1.0.0",
            "alerts": [],
            "summary": {"total": 0, "critical": 0}
        }
        with open(self.run_dir / "org-alerts.json", "w") as f:
            json.dump(alerts, f)

        # sla-intelligence-report.json
        sla = {
            "version": "1.0.0",
            "executive_readiness": {
                "readiness_score": 92.5,
                "tier": "GREEN"
            }
        }
        with open(self.run_dir / "sla-intelligence-report.json", "w") as f:
            json.dump(sla, f)

        # executive-summary.md
        with open(self.run_dir / "executive-summary.md", "w") as f:
            f.write("# Executive Summary\n\nAll systems healthy.\n")

        # bundle-manifest.json (from orchestrator)
        manifest = {
            "manifest_version": "2.0",
            "generated_at": "2025-12-22T14:00:00Z",
            "steps": [
                {"name": "Org Health Report", "exit_code": 90},
                {"name": "SLA Intelligence", "exit_code": 140}
            ]
        }
        with open(self.run_dir / "bundle-manifest.json", "w") as f:
            json.dump(manifest, f)

    def test_packager_creates_zip(self):
        """Test that packager creates ZIP archive."""
        output_dir = Path(self.test_dir) / "output"

        result = subprocess.run(
            [
                sys.executable,
                "scripts/package_executive_bundle.py",
                "--run-dir", str(self.run_dir),
                "--output-dir", str(output_dir),
                "--bundle-name", "test-bundle"
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent
        )

        self.assertEqual(result.returncode, 0, f"Packager failed: {result.stderr}")

        # Check ZIP exists
        zip_path = output_dir / "test-bundle.zip"
        self.assertTrue(zip_path.exists(), f"ZIP not found at {zip_path}")

    def test_packager_creates_manifest(self):
        """Test that packager creates manifest file."""
        output_dir = Path(self.test_dir) / "output"

        result = subprocess.run(
            [
                sys.executable,
                "scripts/package_executive_bundle.py",
                "--run-dir", str(self.run_dir),
                "--output-dir", str(output_dir),
                "--bundle-name", "test-bundle"
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent
        )

        self.assertEqual(result.returncode, 0)

        # Check manifest exists
        manifest_path = output_dir / "test-bundle-manifest.json"
        self.assertTrue(manifest_path.exists())

        # Validate manifest content
        with open(manifest_path) as f:
            manifest = json.load(f)

        self.assertIn("manifest_version", manifest)
        self.assertIn("bundle_name", manifest)
        self.assertIn("included_files", manifest)

    def test_packager_creates_checksums(self):
        """Test that packager creates checksums file."""
        output_dir = Path(self.test_dir) / "output"

        result = subprocess.run(
            [
                sys.executable,
                "scripts/package_executive_bundle.py",
                "--run-dir", str(self.run_dir),
                "--output-dir", str(output_dir),
                "--bundle-name", "test-bundle"
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent
        )

        self.assertEqual(result.returncode, 0)

        # Check checksums file exists
        checksums_path = output_dir / "test-bundle-checksums.sha256"
        self.assertTrue(checksums_path.exists())

        # Validate checksums content
        content = checksums_path.read_text()
        self.assertIn("SHA-256", content)

    def test_packager_creates_tar(self):
        """Test that packager creates tar.gz when --tar is specified."""
        output_dir = Path(self.test_dir) / "output"

        result = subprocess.run(
            [
                sys.executable,
                "scripts/package_executive_bundle.py",
                "--run-dir", str(self.run_dir),
                "--output-dir", str(output_dir),
                "--bundle-name", "test-bundle",
                "--tar"
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent
        )

        self.assertEqual(result.returncode, 0)

        # Check tar.gz exists
        tar_path = output_dir / "test-bundle.tar.gz"
        self.assertTrue(tar_path.exists())

    def test_packager_no_checksums_option(self):
        """Test that --no-checksums skips checksum generation."""
        output_dir = Path(self.test_dir) / "output"

        result = subprocess.run(
            [
                sys.executable,
                "scripts/package_executive_bundle.py",
                "--run-dir", str(self.run_dir),
                "--output-dir", str(output_dir),
                "--bundle-name", "test-bundle",
                "--no-checksums"
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent
        )

        self.assertEqual(result.returncode, 0)

        # Checksums file should not exist
        checksums_path = output_dir / "test-bundle-checksums.sha256"
        self.assertFalse(checksums_path.exists())


class TestPackagerHelpAndArgs(unittest.TestCase):
    """Test help and argument handling."""

    def test_help_displays_correctly(self):
        """Test that --help displays usage information."""
        result = subprocess.run(
            [
                sys.executable,
                "scripts/package_executive_bundle.py",
                "--help"
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent
        )

        self.assertEqual(result.returncode, 0)
        self.assertIn("--run-dir", result.stdout)
        self.assertIn("--output-dir", result.stdout)
        self.assertIn("--bundle-name", result.stdout)
        self.assertIn("--zip", result.stdout)
        self.assertIn("--tar", result.stdout)
        self.assertIn("--checksums", result.stdout)

    def test_missing_run_dir_exits_with_error(self):
        """Test that missing --run-dir exits with error."""
        result = subprocess.run(
            [
                sys.executable,
                "scripts/package_executive_bundle.py"
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent
        )

        # argparse exits with code 2 for missing required argument
        self.assertEqual(result.returncode, 2)
        self.assertIn("--run-dir", result.stderr)


class TestPackagerExitCodes(unittest.TestCase):
    """Test exit code behavior."""

    def test_nonexistent_run_dir_exits_with_code_1(self):
        """Test that nonexistent run directory exits with code 1."""
        result = subprocess.run(
            [
                sys.executable,
                "scripts/package_executive_bundle.py",
                "--run-dir", "/nonexistent/path/12345"
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent
        )

        self.assertEqual(result.returncode, 1)

    def test_empty_run_dir_exits_with_code_2(self):
        """Test that empty run directory exits with appropriate code."""
        test_dir = tempfile.mkdtemp()
        empty_run_dir = Path(test_dir) / "empty-run"
        empty_run_dir.mkdir()

        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/package_executive_bundle.py",
                    "--run-dir", str(empty_run_dir)
                ],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent.parent
            )

            # Empty directory should return exit code 1 (invalid) or 2 (no files)
            self.assertIn(result.returncode, [1, 2])
        finally:
            import shutil
            shutil.rmtree(test_dir, ignore_errors=True)


class TestPackagerZipContents(unittest.TestCase):
    """Test ZIP archive contents."""

    def setUp(self):
        """Set up synthetic test data."""
        self.test_dir = tempfile.mkdtemp()
        self.run_dir = Path(self.test_dir) / "tars-run-20251222-140000"
        self.run_dir.mkdir()

        # Create a few test files
        (self.run_dir / "test1.json").write_text('{"key": "value1"}')
        (self.run_dir / "test2.json").write_text('{"key": "value2"}')
        (self.run_dir / "summary.md").write_text("# Summary\n\nTest content.")

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_zip_contains_all_files(self):
        """Test that ZIP contains all run directory files."""
        output_dir = Path(self.test_dir) / "output"

        result = subprocess.run(
            [
                sys.executable,
                "scripts/package_executive_bundle.py",
                "--run-dir", str(self.run_dir),
                "--output-dir", str(output_dir),
                "--bundle-name", "test-bundle"
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent
        )

        self.assertEqual(result.returncode, 0)

        zip_path = output_dir / "test-bundle.zip"
        with zipfile.ZipFile(zip_path, 'r') as zf:
            names = zf.namelist()

            self.assertIn("test1.json", names)
            self.assertIn("test2.json", names)
            self.assertIn("summary.md", names)


class TestPackagerModule(unittest.TestCase):
    """Test packager module imports."""

    def test_import_packager(self):
        """Test that packager can be imported."""
        try:
            from scripts.package_executive_bundle import (
                ExecutiveBundlePackager,
                generate_timestamp,
                compute_sha256,
                EXIT_SUCCESS,
                EXIT_INVALID_RUN_DIR
            )
        except ImportError as e:
            self.fail(f"Failed to import packager: {e}")

    def test_exit_codes_defined(self):
        """Test that exit codes are properly defined."""
        from scripts.package_executive_bundle import (
            EXIT_SUCCESS,
            EXIT_INVALID_RUN_DIR,
            EXIT_NO_FILES,
            EXIT_ARCHIVE_FAILED,
            EXIT_GENERAL_ERROR
        )

        self.assertEqual(EXIT_SUCCESS, 0)
        self.assertEqual(EXIT_INVALID_RUN_DIR, 1)
        self.assertEqual(EXIT_NO_FILES, 2)
        self.assertEqual(EXIT_ARCHIVE_FAILED, 3)
        self.assertEqual(EXIT_GENERAL_ERROR, 199)

    def test_compute_sha256(self):
        """Test SHA-256 computation."""
        from scripts.package_executive_bundle import compute_sha256

        test_dir = tempfile.mkdtemp()
        test_file = Path(test_dir) / "test.txt"
        test_file.write_text("Hello, World!")

        try:
            checksum = compute_sha256(test_file)

            # Should be a 64-character hex string
            self.assertIsInstance(checksum, str)
            self.assertEqual(len(checksum), 64)

            # Known SHA-256 of "Hello, World!"
            expected = "dffd6021bb2bd5b0af676290809ec3a53191dd81c7f70a4b28688a362182986f"
            self.assertEqual(checksum, expected)
        finally:
            import shutil
            shutil.rmtree(test_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
