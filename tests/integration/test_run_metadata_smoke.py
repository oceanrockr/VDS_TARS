#!/usr/bin/env python3
"""
Smoke Tests for Run Metadata Generator

Tests the generate_run_metadata.py script functionality including:
- Metadata file creation
- Git information capture
- Host environment capture
- Artifact listing
- CLI interface

Phase: 17 - Post-GA Observability
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path


class TestRunMetadataSmokeTests(unittest.TestCase):
    """Smoke tests for Run Metadata Generator."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.project_root = Path(__file__).parent.parent.parent
        cls.script_path = cls.project_root / "scripts" / "generate_run_metadata.py"

    def setUp(self):
        """Create temporary directories for each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.run_dir = Path(self.temp_dir) / "tars-run-20251224-120000"
        self.run_dir.mkdir(parents=True)

    def tearDown(self):
        """Clean up temporary directories."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_sample_files(self):
        """Create sample files in the run directory."""
        # Create sample reports
        (self.run_dir / "org-health-report.json").write_text('{"status": "healthy"}')
        (self.run_dir / "sla-intelligence-report.json").write_text('{"compliance": "ok"}')
        (self.run_dir / "bundle-manifest.json").write_text(json.dumps({
            "manifest_version": "2.1",
            "pipeline": {
                "root_dir": "/test",
                "output_dir": str(self.run_dir),
                "duration_seconds": 10.5
            },
            "steps": [
                {"name": "Org Health Report", "exit_code": 0},
                {"name": "SLA Intelligence", "exit_code": 140}
            ]
        }))

    def test_script_exists(self):
        """Test that the metadata generator script exists."""
        self.assertTrue(
            self.script_path.exists(),
            f"Script not found at {self.script_path}"
        )

    def test_help_command(self):
        """Test that --help command works."""
        result = subprocess.run(
            [sys.executable, str(self.script_path), "--help"],
            capture_output=True,
            text=True,
            timeout=30
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("generate_run_metadata", result.stdout)
        self.assertIn("--run-dir", result.stdout)

    def test_missing_run_dir_returns_error(self):
        """Test that missing run directory returns exit code 1."""
        nonexistent_dir = Path(self.temp_dir) / "nonexistent"
        result = subprocess.run(
            [sys.executable, str(self.script_path), "--run-dir", str(nonexistent_dir)],
            capture_output=True,
            text=True,
            timeout=30
        )
        self.assertEqual(result.returncode, 1)

    def test_metadata_file_created(self):
        """Test that metadata file is created."""
        self._create_sample_files()

        result = subprocess.run(
            [sys.executable, str(self.script_path), "--run-dir", str(self.run_dir)],
            capture_output=True,
            text=True,
            timeout=30
        )

        self.assertEqual(result.returncode, 0)
        metadata_path = self.run_dir / "run-metadata.json"
        self.assertTrue(metadata_path.exists())

    def test_metadata_contains_required_fields(self):
        """Test that metadata contains all required fields."""
        self._create_sample_files()

        subprocess.run(
            [sys.executable, str(self.script_path), "--run-dir", str(self.run_dir)],
            capture_output=True,
            text=True,
            timeout=30
        )

        metadata_path = self.run_dir / "run-metadata.json"
        with open(metadata_path) as f:
            metadata = json.load(f)

        # Check required top-level fields
        self.assertIn("metadata_version", metadata)
        self.assertIn("metadata_schema", metadata)
        self.assertIn("generated_at", metadata)
        self.assertIn("generator", metadata)
        self.assertIn("tars", metadata)
        self.assertIn("environment", metadata)
        self.assertIn("run", metadata)
        self.assertIn("exit_codes", metadata)
        self.assertIn("artifacts", metadata)
        self.assertIn("provenance", metadata)

    def test_metadata_captures_tars_version(self):
        """Test that T.A.R.S. version is captured."""
        self._create_sample_files()

        subprocess.run(
            [sys.executable, str(self.script_path), "--run-dir", str(self.run_dir)],
            capture_output=True,
            text=True,
            timeout=30
        )

        metadata_path = self.run_dir / "run-metadata.json"
        with open(metadata_path) as f:
            metadata = json.load(f)

        self.assertIn("tars", metadata)
        self.assertIn("version", metadata["tars"])
        # Version should be a non-empty string
        self.assertTrue(metadata["tars"]["version"])

    def test_metadata_captures_environment(self):
        """Test that host environment is captured."""
        self._create_sample_files()

        subprocess.run(
            [sys.executable, str(self.script_path), "--run-dir", str(self.run_dir)],
            capture_output=True,
            text=True,
            timeout=30
        )

        metadata_path = self.run_dir / "run-metadata.json"
        with open(metadata_path) as f:
            metadata = json.load(f)

        env = metadata["environment"]
        self.assertIn("os", env)
        self.assertIn("python_version", env)
        self.assertIn("hostname", env)

    def test_metadata_extracts_exit_codes(self):
        """Test that exit codes are extracted from manifest."""
        self._create_sample_files()

        subprocess.run(
            [sys.executable, str(self.script_path), "--run-dir", str(self.run_dir)],
            capture_output=True,
            text=True,
            timeout=30
        )

        metadata_path = self.run_dir / "run-metadata.json"
        with open(metadata_path) as f:
            metadata = json.load(f)

        exit_codes = metadata["exit_codes"]
        self.assertIn("Org Health Report", exit_codes)
        self.assertEqual(exit_codes["Org Health Report"], 0)
        self.assertIn("SLA Intelligence", exit_codes)
        self.assertEqual(exit_codes["SLA Intelligence"], 140)

    def test_metadata_lists_artifacts(self):
        """Test that artifacts are listed."""
        self._create_sample_files()

        subprocess.run(
            [sys.executable, str(self.script_path), "--run-dir", str(self.run_dir)],
            capture_output=True,
            text=True,
            timeout=30
        )

        metadata_path = self.run_dir / "run-metadata.json"
        with open(metadata_path) as f:
            metadata = json.load(f)

        artifacts = metadata["artifacts"]
        # Should have at least the files we created plus run-metadata.json
        self.assertGreaterEqual(len(artifacts), 3)

        # Check artifact structure
        for artifact in artifacts:
            self.assertIn("path", artifact)
            self.assertIn("size_bytes", artifact)

    def test_custom_output_path(self):
        """Test that custom output path works."""
        self._create_sample_files()
        custom_output = Path(self.temp_dir) / "custom" / "metadata.json"

        result = subprocess.run(
            [
                sys.executable, str(self.script_path),
                "--run-dir", str(self.run_dir),
                "--output", str(custom_output)
            ],
            capture_output=True,
            text=True,
            timeout=30
        )

        self.assertEqual(result.returncode, 0)
        self.assertTrue(custom_output.exists())

    def test_cli_flags_captured(self):
        """Test that CLI flags are captured when provided."""
        self._create_sample_files()

        result = subprocess.run(
            [
                sys.executable, str(self.script_path),
                "--run-dir", str(self.run_dir),
                "--cli-flags", "--root ./org-health --fail-on-breach"
            ],
            capture_output=True,
            text=True,
            timeout=30
        )

        self.assertEqual(result.returncode, 0)

        metadata_path = self.run_dir / "run-metadata.json"
        with open(metadata_path) as f:
            metadata = json.load(f)

        self.assertEqual(
            metadata["run"]["cli_flags"],
            "--root ./org-health --fail-on-breach"
        )

    def test_empty_run_dir_still_generates_metadata(self):
        """Test that empty run directory still generates metadata."""
        # Run directory exists but is empty
        result = subprocess.run(
            [sys.executable, str(self.script_path), "--run-dir", str(self.run_dir)],
            capture_output=True,
            text=True,
            timeout=30
        )

        self.assertEqual(result.returncode, 0)
        metadata_path = self.run_dir / "run-metadata.json"
        self.assertTrue(metadata_path.exists())

    def test_verbose_mode(self):
        """Test verbose mode produces debug output."""
        self._create_sample_files()

        result = subprocess.run(
            [
                sys.executable, str(self.script_path),
                "--run-dir", str(self.run_dir),
                "-v"
            ],
            capture_output=True,
            text=True,
            timeout=30
        )

        self.assertEqual(result.returncode, 0)
        # Verbose mode should produce more output
        self.assertIn("Metadata Generation Complete", result.stderr)

    def test_provenance_structure(self):
        """Test that provenance section has correct structure."""
        self._create_sample_files()

        subprocess.run(
            [sys.executable, str(self.script_path), "--run-dir", str(self.run_dir)],
            capture_output=True,
            text=True,
            timeout=30
        )

        metadata_path = self.run_dir / "run-metadata.json"
        with open(metadata_path) as f:
            metadata = json.load(f)

        provenance = metadata["provenance"]
        self.assertIn("build_type", provenance)
        self.assertIn("reproducible", provenance)
        self.assertIn("attestation", provenance)
        self.assertEqual(provenance["attestation"]["type"], "tars-provenance-v1")


class TestMetadataIntegrationFunction(unittest.TestCase):
    """Tests for the integration function."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.project_root = Path(__file__).parent.parent.parent
        # Add project root to path for imports
        sys.path.insert(0, str(cls.project_root))

    def setUp(self):
        """Create temporary directory for each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.run_dir = Path(self.temp_dir) / "tars-run-test"
        self.run_dir.mkdir(parents=True)

    def tearDown(self):
        """Clean up temporary directory."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_integration_function_exists(self):
        """Test that integration function can be imported."""
        try:
            from scripts.generate_run_metadata import integrate_with_orchestrator
            self.assertTrue(callable(integrate_with_orchestrator))
        except ImportError:
            # Expected if running standalone
            self.skipTest("Cannot import module in standalone mode")

    def test_integration_function_returns_bool(self):
        """Test that integration function returns boolean."""
        try:
            from scripts.generate_run_metadata import integrate_with_orchestrator
            result = integrate_with_orchestrator(self.run_dir)
            self.assertIsInstance(result, bool)
        except ImportError:
            self.skipTest("Cannot import module in standalone mode")


if __name__ == "__main__":
    unittest.main()
