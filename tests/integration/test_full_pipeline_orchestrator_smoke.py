#!/usr/bin/env python3
"""
Smoke Test Suite for Full Pipeline Orchestrator

This module provides smoke tests for the pipeline orchestrator:
- Dry-run mode command plan validation
- Timestamp generation
- Output directory structure
- Argument parsing
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
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestOrchestratorDryRun(unittest.TestCase):
    """Test dry-run mode command plan."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.org_health_dir = Path(self.test_dir) / "org-health"
        self.org_health_dir.mkdir()

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_dry_run_does_not_crash(self):
        """Test that --dry-run executes without error."""
        result = subprocess.run(
            [
                sys.executable,
                "scripts/run_full_org_governance_pipeline.py",
                "--root", str(self.org_health_dir),
                "--dry-run"
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent
        )

        # Should succeed (exit code 0 or 1 depending on module availability)
        self.assertIn(result.returncode, [0, 1])
        self.assertIn("DRY RUN", result.stderr + result.stdout)

    def test_dry_run_prints_commands(self):
        """Test that --dry-run prints expected commands."""
        result = subprocess.run(
            [
                sys.executable,
                "scripts/run_full_org_governance_pipeline.py",
                "--root", str(self.org_health_dir),
                "--dry-run"
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent
        )

        output = result.stderr + result.stdout
        # Should mention module execution
        self.assertTrue(
            "Would execute" in output or "DRY RUN" in output,
            f"Expected dry-run output, got: {output[:500]}"
        )

    def test_help_displays_correctly(self):
        """Test that --help displays usage information."""
        result = subprocess.run(
            [
                sys.executable,
                "scripts/run_full_org_governance_pipeline.py",
                "--help"
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent
        )

        self.assertEqual(result.returncode, 0)
        self.assertIn("--root", result.stdout)
        self.assertIn("--timestamp", result.stdout)
        self.assertIn("--format", result.stdout)
        self.assertIn("--print-paths", result.stdout)
        self.assertIn("--dry-run", result.stdout)


class TestOrchestratorTimestamp(unittest.TestCase):
    """Test timestamp handling."""

    def test_generate_timestamp_format(self):
        """Test that timestamp format is correct."""
        from scripts.run_full_org_governance_pipeline import generate_timestamp

        timestamp = generate_timestamp()

        # Should be YYYYMMDD-HHMMSS format
        self.assertRegex(timestamp, r"^\d{8}-\d{6}$")

    def test_custom_timestamp_accepted(self):
        """Test that custom timestamp is accepted."""
        test_dir = tempfile.mkdtemp()
        org_health_dir = Path(test_dir) / "org-health"
        org_health_dir.mkdir()

        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/run_full_org_governance_pipeline.py",
                    "--root", str(org_health_dir),
                    "--timestamp", "20251222-140000",
                    "--dry-run"
                ],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent.parent
            )

            output = result.stderr + result.stdout
            self.assertIn("20251222-140000", output)
        finally:
            import shutil
            shutil.rmtree(test_dir, ignore_errors=True)


class TestOrchestratorOutputFormat(unittest.TestCase):
    """Test output format options."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.org_health_dir = Path(self.test_dir) / "org-health"
        self.org_health_dir.mkdir()

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_flat_format_option(self):
        """Test --format flat option is accepted."""
        result = subprocess.run(
            [
                sys.executable,
                "scripts/run_full_org_governance_pipeline.py",
                "--root", str(self.org_health_dir),
                "--format", "flat",
                "--dry-run"
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent
        )

        self.assertIn(result.returncode, [0, 1])

    def test_structured_format_option(self):
        """Test --format structured option is accepted."""
        result = subprocess.run(
            [
                sys.executable,
                "scripts/run_full_org_governance_pipeline.py",
                "--root", str(self.org_health_dir),
                "--format", "structured",
                "--dry-run"
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent
        )

        self.assertIn(result.returncode, [0, 1])


class TestOrchestratorPrintPaths(unittest.TestCase):
    """Test --print-paths functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.org_health_dir = Path(self.test_dir) / "org-health"
        self.org_health_dir.mkdir()

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_print_paths_shows_output(self):
        """Test that --print-paths shows artifact paths."""
        result = subprocess.run(
            [
                sys.executable,
                "scripts/run_full_org_governance_pipeline.py",
                "--root", str(self.org_health_dir),
                "--print-paths",
                "--dry-run"
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent
        )

        output = result.stderr + result.stdout
        # Should print ARTIFACT PATHS section
        self.assertIn("ARTIFACT PATHS", output)


class TestOrchestratorExitCodes(unittest.TestCase):
    """Test exit code behavior."""

    def test_missing_root_exits_with_error(self):
        """Test that missing --root argument exits with error."""
        result = subprocess.run(
            [
                sys.executable,
                "scripts/run_full_org_governance_pipeline.py"
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent
        )

        # argparse exits with code 2 for missing required argument
        self.assertEqual(result.returncode, 2)
        self.assertIn("--root", result.stderr)

    def test_nonexistent_root_handled(self):
        """Test that nonexistent root directory is handled."""
        result = subprocess.run(
            [
                sys.executable,
                "scripts/run_full_org_governance_pipeline.py",
                "--root", "/nonexistent/path/12345",
                "--dry-run"
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent
        )

        # Should handle gracefully (dry run mode)
        self.assertIn(result.returncode, [0, 1, 199])


class TestOrchestratorModuleCheck(unittest.TestCase):
    """Test module availability checking."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.org_health_dir = Path(self.test_dir) / "org-health"
        self.org_health_dir.mkdir()

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_module_check_output(self):
        """Test that module availability is reported."""
        result = subprocess.run(
            [
                sys.executable,
                "scripts/run_full_org_governance_pipeline.py",
                "--root", str(self.org_health_dir),
                "--dry-run"
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent
        )

        output = result.stderr + result.stdout
        # Should mention "Available Modules" or similar
        self.assertTrue(
            "Available Modules" in output or "Module" in output or "module" in output,
            f"Expected module check output, got: {output[:500]}"
        )


class TestOrchestratorClass(unittest.TestCase):
    """Test PipelineOrchestrator class directly."""

    def test_import_orchestrator(self):
        """Test that orchestrator can be imported."""
        try:
            from scripts.run_full_org_governance_pipeline import (
                PipelineOrchestrator,
                generate_timestamp,
                EXIT_SUCCESS,
                EXIT_PIPELINE_ERROR
            )
        except ImportError as e:
            self.fail(f"Failed to import orchestrator: {e}")

    def test_generate_timestamp_not_empty(self):
        """Test that generate_timestamp returns non-empty string."""
        from scripts.run_full_org_governance_pipeline import generate_timestamp

        timestamp = generate_timestamp()
        self.assertIsInstance(timestamp, str)
        self.assertGreater(len(timestamp), 0)

    def test_exit_codes_defined(self):
        """Test that exit codes are properly defined."""
        from scripts.run_full_org_governance_pipeline import (
            EXIT_SUCCESS,
            EXIT_PIPELINE_ERROR,
            EXIT_SLA_AT_RISK,
            EXIT_SLA_BREACH,
            EXIT_GENERAL_ERROR
        )

        self.assertEqual(EXIT_SUCCESS, 0)
        self.assertEqual(EXIT_PIPELINE_ERROR, 1)
        self.assertEqual(EXIT_SLA_AT_RISK, 140)
        self.assertEqual(EXIT_SLA_BREACH, 142)
        self.assertEqual(EXIT_GENERAL_ERROR, 199)


if __name__ == "__main__":
    unittest.main()
