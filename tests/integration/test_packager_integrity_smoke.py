#!/usr/bin/env python3
"""
Smoke tests for T.A.R.S. Executive Bundle Packager Integrity Features

Tests the enhanced integrity and signing features in package_executive_bundle.py.

Version: 1.0.0
Phase: 18 - Ops Integrations
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.package_executive_bundle import (
    ExecutiveBundlePackager,
    check_gpg_available,
    compute_sha256,
    get_tars_version,
    get_git_commit,
    collect_files,
    EXIT_SUCCESS,
    EXIT_INVALID_RUN_DIR,
    EXIT_NO_FILES,
)


class TestCheckGPGAvailable(unittest.TestCase):
    """Test GPG availability check."""

    def test_gpg_check_returns_bool(self):
        """Test that GPG check returns boolean."""
        result = check_gpg_available()
        self.assertIsInstance(result, bool)


class TestSHA256Computation(unittest.TestCase):
    """Test SHA-256 checksum computation."""

    def test_compute_sha256_file(self):
        """Test SHA-256 computation for a file."""
        temp_dir = tempfile.mkdtemp()
        try:
            test_file = Path(temp_dir) / "test.txt"
            test_file.write_text("Hello, World!")

            checksum = compute_sha256(test_file)

            # SHA-256 should be 64 hex characters
            self.assertEqual(len(checksum), 64)
            self.assertTrue(all(c in "0123456789abcdef" for c in checksum))
        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_compute_sha256_consistent(self):
        """Test SHA-256 computation is consistent."""
        temp_dir = tempfile.mkdtemp()
        try:
            test_file = Path(temp_dir) / "test.txt"
            test_file.write_text("Test content")

            checksum1 = compute_sha256(test_file)
            checksum2 = compute_sha256(test_file)

            self.assertEqual(checksum1, checksum2)
        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)


class TestCollectFiles(unittest.TestCase):
    """Test file collection from run directory."""

    def test_collect_files_flat(self):
        """Test collecting files from flat directory."""
        temp_dir = tempfile.mkdtemp()
        try:
            run_dir = Path(temp_dir)
            (run_dir / "file1.json").write_text("{}")
            (run_dir / "file2.md").write_text("# Test")

            files = collect_files(run_dir)

            self.assertEqual(len(files), 2)
        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_collect_files_nested(self):
        """Test collecting files from nested directories."""
        temp_dir = tempfile.mkdtemp()
        try:
            run_dir = Path(temp_dir)
            (run_dir / "file1.json").write_text("{}")
            subdir = run_dir / "subdir"
            subdir.mkdir()
            (subdir / "file2.json").write_text("{}")

            files = collect_files(run_dir)

            self.assertEqual(len(files), 2)
        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)


class TestExecutiveBundlePackagerIntegrity(unittest.TestCase):
    """Test ExecutiveBundlePackager integrity features."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.run_dir = Path(self.temp_dir) / "tars-run-20251225-080000"
        self.run_dir.mkdir()
        self.output_dir = Path(self.temp_dir) / "output"
        self.output_dir.mkdir()

        # Create test files
        (self.run_dir / "org-health-report.json").write_text('{"status": "ok"}')
        (self.run_dir / "executive-summary.md").write_text("# Summary")

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_packager_creates_checksums(self):
        """Test packager creates checksums file."""
        packager = ExecutiveBundlePackager(
            run_dir=str(self.run_dir),
            output_dir=str(self.output_dir),
            create_checksums=True
        )
        result = packager.package()

        self.assertEqual(result, EXIT_SUCCESS)

        # Find checksums file
        checksums_files = list(self.output_dir.glob("*-checksums.sha256"))
        self.assertEqual(len(checksums_files), 1)

        # Verify content
        content = checksums_files[0].read_text()
        self.assertIn("SHA-256", content)

    def test_packager_creates_integrity_doc(self):
        """Test packager creates integrity documentation."""
        packager = ExecutiveBundlePackager(
            run_dir=str(self.run_dir),
            output_dir=str(self.output_dir),
            create_checksums=True
        )
        result = packager.package()

        self.assertEqual(result, EXIT_SUCCESS)

        # Find integrity doc
        integrity_files = list(self.output_dir.glob("*-integrity.md"))
        self.assertEqual(len(integrity_files), 1)

        # Verify content
        content = integrity_files[0].read_text()
        self.assertIn("Integrity Verification", content)
        self.assertIn("SHA-256", content)
        self.assertIn("Verify", content)

    def test_packager_integrity_doc_content(self):
        """Test integrity doc has proper verification instructions."""
        packager = ExecutiveBundlePackager(
            run_dir=str(self.run_dir),
            output_dir=str(self.output_dir),
            create_checksums=True
        )
        result = packager.package()

        self.assertEqual(result, EXIT_SUCCESS)

        # Read integrity doc
        integrity_files = list(self.output_dir.glob("*-integrity.md"))
        content = integrity_files[0].read_text()

        # Should have verification commands
        self.assertIn("sha256sum", content)
        self.assertIn("PowerShell", content)
        self.assertIn("Included Files", content)

    def test_packager_sign_flag_without_gpg(self):
        """Test packager handles sign flag when GPG not available."""
        # Mock GPG as unavailable
        with patch("scripts.package_executive_bundle.check_gpg_available", return_value=False):
            packager = ExecutiveBundlePackager(
                run_dir=str(self.run_dir),
                output_dir=str(self.output_dir),
                sign=True
            )
            result = packager.package()

            # Should still succeed
            self.assertEqual(result, EXIT_SUCCESS)

            # Should not create .sig files
            sig_files = list(self.output_dir.glob("*.sig"))
            self.assertEqual(len(sig_files), 0)

    @patch("scripts.package_executive_bundle.check_gpg_available")
    @patch("subprocess.run")
    def test_packager_creates_signature(self, mock_run, mock_gpg_check):
        """Test packager creates GPG signature when available."""
        mock_gpg_check.return_value = True
        mock_run.return_value = MagicMock(returncode=0)

        packager = ExecutiveBundlePackager(
            run_dir=str(self.run_dir),
            output_dir=str(self.output_dir),
            sign=True,
            gpg_key_id="TEST_KEY"
        )
        packager.gpg_available = True  # Override
        result = packager.package()

        self.assertEqual(result, EXIT_SUCCESS)

        # GPG should have been called
        self.assertTrue(mock_run.called)

    def test_packager_manifest_has_checksum_info(self):
        """Test packager manifest includes checksum info."""
        packager = ExecutiveBundlePackager(
            run_dir=str(self.run_dir),
            output_dir=str(self.output_dir),
            create_checksums=True,
            create_manifest=True
        )
        result = packager.package()

        self.assertEqual(result, EXIT_SUCCESS)

        # Read manifest
        manifest_files = list(self.output_dir.glob("*-manifest.json"))
        self.assertEqual(len(manifest_files), 1)

        with open(manifest_files[0], "r") as f:
            manifest = json.load(f)

        self.assertIn("checksums", manifest)

    def test_generate_bundle_integrity_doc_content(self):
        """Test integrity doc generation method."""
        packager = ExecutiveBundlePackager(
            run_dir=str(self.run_dir),
            output_dir=str(self.output_dir),
            create_checksums=True
        )
        packager.validate_run_dir()
        packager.checksums = packager.compute_checksums()

        content = packager.generate_bundle_integrity_doc()

        # Check required sections
        self.assertIn("T.A.R.S. Executive Bundle Integrity Verification", content)
        self.assertIn("How to Verify This Bundle", content)
        self.assertIn("Verify SHA-256 Checksums", content)
        self.assertIn("Linux/macOS", content)
        self.assertIn("Windows", content)
        self.assertIn("Included Files", content)


class TestPackagerWithConfig(unittest.TestCase):
    """Test packager config file support."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.run_dir = Path(self.temp_dir) / "tars-run-20251225-080000"
        self.run_dir.mkdir()
        (self.run_dir / "test.json").write_text("{}")

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_packager_config_support_exists(self):
        """Test that packager has config support in CLI."""
        from scripts.package_executive_bundle import create_argument_parser

        parser = create_argument_parser()
        args = parser.parse_args([
            "--run-dir", str(self.run_dir),
            "--config", "tars.yml"
        ])

        self.assertEqual(args.config, "tars.yml")

    def test_packager_sign_flag_in_cli(self):
        """Test that packager has sign flag in CLI."""
        from scripts.package_executive_bundle import create_argument_parser

        parser = create_argument_parser()
        args = parser.parse_args([
            "--run-dir", str(self.run_dir),
            "--sign",
            "--gpg-key-id", "ABC123"
        ])

        self.assertTrue(args.sign)
        self.assertEqual(args.gpg_key_id, "ABC123")


class TestVersionAndGitHelpers(unittest.TestCase):
    """Test version and git helper functions."""

    def test_get_tars_version_returns_string(self):
        """Test version helper returns string."""
        version = get_tars_version()
        self.assertIsInstance(version, str)
        self.assertTrue(len(version) > 0)

    def test_get_git_commit_returns_string_or_none(self):
        """Test git commit helper returns string or None."""
        commit = get_git_commit()
        self.assertTrue(commit is None or isinstance(commit, str))


if __name__ == "__main__":
    unittest.main(verbosity=2)
