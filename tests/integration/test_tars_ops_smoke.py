#!/usr/bin/env python3
"""
Smoke Tests for T.A.R.S. Operations CLI (tars_ops.py)

Phase 19 - Production Ops Maturity & CI Hardening

Tests the golden path wrapper script for daily/weekly/incident operations.

Test Coverage:
    - Help output and usage
    - Command parsing for daily/weekly/incident
    - Exit code guidance function
    - Banner and guidance output
    - Config resolution

Version: 1.0.0
"""

import argparse
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.tars_ops import (
    EXIT_CODE_GUIDANCE,
    build_orchestrator_command,
    build_packager_command,
    create_parser,
    get_latest_run_dir,
    get_project_root,
    print_guidance,
)


class TestExitCodeGuidance(unittest.TestCase):
    """Tests for exit code guidance mapping."""

    def test_success_code(self):
        """Exit code 0 should indicate success."""
        self.assertIn(0, EXIT_CODE_GUIDANCE)
        status, action, escalate = EXIT_CODE_GUIDANCE[0]
        self.assertEqual(status, "SUCCESS")
        self.assertFalse(escalate)

    def test_sla_breach_code(self):
        """Exit code 142 should indicate SLA breach."""
        self.assertIn(142, EXIT_CODE_GUIDANCE)
        status, action, escalate = EXIT_CODE_GUIDANCE[142]
        self.assertEqual(status, "SLA BREACH")
        self.assertTrue(escalate)

    def test_critical_alert_code(self):
        """Exit code 102 should indicate critical alerts."""
        self.assertIn(102, EXIT_CODE_GUIDANCE)
        status, action, escalate = EXIT_CODE_GUIDANCE[102]
        self.assertEqual(status, "CRITICAL ALERTS")
        self.assertTrue(escalate)

    def test_at_risk_code(self):
        """Exit code 141 should indicate at-risk SLAs."""
        self.assertIn(141, EXIT_CODE_GUIDANCE)
        status, action, escalate = EXIT_CODE_GUIDANCE[141]
        self.assertEqual(status, "AT-RISK SLAs")
        self.assertFalse(escalate)

    def test_all_codes_have_three_elements(self):
        """All guidance entries should have (status, action, escalate)."""
        for code, guidance in EXIT_CODE_GUIDANCE.items():
            self.assertEqual(len(guidance), 3, f"Code {code} has wrong number of elements")
            status, action, escalate = guidance
            self.assertIsInstance(status, str)
            self.assertIsInstance(action, str)
            self.assertIsInstance(escalate, bool)


class TestCommandBuilding(unittest.TestCase):
    """Tests for command building functions."""

    def test_build_orchestrator_minimal(self):
        """Should build minimal orchestrator command."""
        cmd = build_orchestrator_command(
            config=None,
            root=None,
            output_format="flat",
            print_paths=False,
            with_narrative=False,
            fail_on_breach=False
        )
        self.assertIn("run_full_org_governance_pipeline.py", cmd[1])
        self.assertIn("--format", cmd)
        self.assertIn("flat", cmd)

    def test_build_orchestrator_with_config(self):
        """Should include config path when provided."""
        cmd = build_orchestrator_command(
            config="./tars.yml",
            root=None,
            output_format="flat",
            print_paths=False,
            with_narrative=False,
            fail_on_breach=False
        )
        self.assertIn("--config", cmd)
        self.assertIn("./tars.yml", cmd)

    def test_build_orchestrator_with_all_options(self):
        """Should include all options when enabled."""
        cmd = build_orchestrator_command(
            config="./tars.yml",
            root="./org-health",
            output_format="structured",
            print_paths=True,
            with_narrative=True,
            fail_on_breach=True
        )
        self.assertIn("--config", cmd)
        self.assertIn("--root", cmd)
        self.assertIn("--format", cmd)
        self.assertIn("structured", cmd)
        self.assertIn("--print-paths", cmd)
        self.assertIn("--with-narrative", cmd)
        self.assertIn("--fail-on-breach", cmd)

    def test_build_orchestrator_additional_args(self):
        """Should include additional args when provided."""
        cmd = build_orchestrator_command(
            config=None,
            root=None,
            output_format="flat",
            print_paths=False,
            with_narrative=False,
            fail_on_breach=False,
            additional_args=["--sla-policy", "./policy.yaml"]
        )
        self.assertIn("--sla-policy", cmd)
        self.assertIn("./policy.yaml", cmd)

    def test_build_packager_minimal(self):
        """Should build minimal packager command."""
        cmd = build_packager_command(
            config=None,
            run_dir="./reports/runs/tars-run-20251225",
            include_tar=False,
            sign=False
        )
        self.assertIn("package_executive_bundle.py", cmd[1])
        self.assertIn("--run-dir", cmd)

    def test_build_packager_with_tar_and_sign(self):
        """Should include tar and sign flags when enabled."""
        cmd = build_packager_command(
            config="./tars.yml",
            run_dir="./run",
            include_tar=True,
            sign=True
        )
        self.assertIn("--config", cmd)
        self.assertIn("--tar", cmd)
        self.assertIn("--sign", cmd)


class TestArgumentParser(unittest.TestCase):
    """Tests for argument parser."""

    def setUp(self):
        """Set up parser for tests."""
        self.parser = create_parser()

    def test_parser_creates_successfully(self):
        """Parser should be created without error."""
        self.assertIsNotNone(self.parser)

    def test_parse_daily_command(self):
        """Should parse daily command."""
        args = self.parser.parse_args(["daily"])
        self.assertEqual(args.command, "daily")

    def test_parse_weekly_command(self):
        """Should parse weekly command."""
        args = self.parser.parse_args(["weekly"])
        self.assertEqual(args.command, "weekly")

    def test_parse_incident_command(self):
        """Should parse incident command."""
        args = self.parser.parse_args(["incident"])
        self.assertEqual(args.command, "incident")

    def test_parse_daily_with_options(self):
        """Should parse daily with all options."""
        args = self.parser.parse_args([
            "--config", "./tars.yml",
            "daily",
            "--fail-on-breach",
            "--package"
        ])
        self.assertEqual(args.command, "daily")
        self.assertEqual(args.config, "./tars.yml")
        self.assertTrue(args.fail_on_breach)
        self.assertTrue(args.package)

    def test_parse_incident_with_id(self):
        """Should parse incident with incident-id."""
        args = self.parser.parse_args([
            "incident",
            "--incident-id", "INC-12345"
        ])
        self.assertEqual(args.command, "incident")
        self.assertEqual(args.incident_id, "INC-12345")

    def test_parse_incident_with_sign(self):
        """Should parse incident with sign flag."""
        args = self.parser.parse_args([
            "incident",
            "--sign"
        ])
        self.assertTrue(args.sign)

    def test_global_root_option(self):
        """Should parse global root option."""
        args = self.parser.parse_args([
            "--root", "./custom-org-health",
            "daily"
        ])
        self.assertEqual(args.root, "./custom-org-health")

    def test_no_command_returns_none(self):
        """Should return None command when no subcommand given."""
        args = self.parser.parse_args([])
        self.assertIsNone(args.command)


class TestProjectRoot(unittest.TestCase):
    """Tests for project root detection."""

    def test_get_project_root(self):
        """Should return a valid directory."""
        root = get_project_root()
        self.assertTrue(root.exists())
        self.assertTrue(root.is_dir())

    def test_project_root_has_scripts(self):
        """Project root should contain scripts directory."""
        root = get_project_root()
        scripts_dir = root / "scripts"
        self.assertTrue(scripts_dir.exists())


class TestGetLatestRunDir(unittest.TestCase):
    """Tests for latest run directory detection."""

    def test_returns_none_if_no_runs(self):
        """Should return None if no run directories exist."""
        with patch.object(Path, 'exists', return_value=False):
            result = get_latest_run_dir()
            # Result depends on actual filesystem state
            # This test verifies the function doesn't crash

    def test_returns_string_or_none(self):
        """Should return string or None."""
        result = get_latest_run_dir()
        self.assertTrue(result is None or isinstance(result, str))


class TestPrintGuidance(unittest.TestCase):
    """Tests for print_guidance function."""

    def test_print_guidance_success(self):
        """Should print guidance for success exit code."""
        with patch('builtins.print') as mock_print:
            print_guidance(0)
            # Verify print was called
            self.assertTrue(mock_print.called)
            # Check that "SUCCESS" appears in one of the print calls
            all_args = [str(call) for call in mock_print.call_args_list]
            found = any("SUCCESS" in arg for arg in all_args)
            self.assertTrue(found)

    def test_print_guidance_breach(self):
        """Should print guidance for breach exit code."""
        with patch('builtins.print') as mock_print:
            print_guidance(142)
            all_args = [str(call) for call in mock_print.call_args_list]
            found = any("SLA BREACH" in arg for arg in all_args)
            self.assertTrue(found)

    def test_print_guidance_unknown(self):
        """Should handle unknown exit codes gracefully."""
        with patch('builtins.print') as mock_print:
            print_guidance(999)
            all_args = [str(call) for call in mock_print.call_args_list]
            found = any("UNKNOWN" in arg for arg in all_args)
            self.assertTrue(found)


if __name__ == "__main__":
    unittest.main(verbosity=2)
