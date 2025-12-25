#!/usr/bin/env python3
"""
Smoke Tests for Executive Narrative Generator

Tests the generate_executive_narrative.py script functionality including:
- Narrative file creation
- Health tier determination
- SLA summary formatting
- Risk identification
- Graceful handling of missing reports

Phase: 17 - Post-GA Observability
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestExecutiveNarrativeSmokeTests(unittest.TestCase):
    """Smoke tests for Executive Narrative Generator."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.project_root = Path(__file__).parent.parent.parent
        cls.script_path = cls.project_root / "scripts" / "generate_executive_narrative.py"

    def setUp(self):
        """Create temporary directories for each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.run_dir = Path(self.temp_dir) / "tars-run-20251224-120000"
        self.run_dir.mkdir(parents=True)

    def tearDown(self):
        """Clean up temporary directories."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_minimal_reports(self):
        """Create minimal reports for testing."""
        # Create SLA intelligence report
        sla_report = {
            "executive_readiness": {
                "readiness_score": 85,
                "tier": "GREEN"
            },
            "summary": {
                "total_slas": 5,
                "compliant_count": 4,
                "at_risk_count": 1,
                "breach_count": 0
            },
            "compliance_results": [
                {"sla_id": "availability", "status": "COMPLIANT"},
                {"sla_id": "latency", "status": "AT_RISK"}
            ],
            "breaches": []
        }
        (self.run_dir / "sla-intelligence-report.json").write_text(json.dumps(sla_report))

    def _create_full_reports(self):
        """Create full set of reports for comprehensive testing."""
        self._create_minimal_reports()

        # Create org health report
        org_health = {
            "risk_tier": "LOW",
            "aggregate_score": 92,
            "repository_reports": [
                {"repo": "repo-a", "health_score": 95},
                {"repo": "repo-b", "health_score": 88}
            ]
        }
        (self.run_dir / "org-health-report.json").write_text(json.dumps(org_health))

        # Create alerts report
        alerts_report = {
            "alerts": [
                {"severity": "WARNING", "message": "Test warning"}
            ]
        }
        (self.run_dir / "org-alerts.json").write_text(json.dumps(alerts_report))

        # Create temporal intelligence report
        temporal_report = {
            "influence_scores": [
                {"repository": "repo-a", "influence_score": 75, "classification": "LEADER"}
            ],
            "propagation_paths": [
                {"path": ["repo-a", "repo-b", "repo-c"], "impact_score": 0.8}
            ],
            "temporal_anomalies": []
        }
        (self.run_dir / "temporal-intelligence-report.json").write_text(json.dumps(temporal_report))

        # Create trend correlation report
        trend_report = {
            "clusters": [
                {
                    "repositories": ["repo-a", "repo-b"],
                    "correlation_strength": 0.85,
                    "is_anomalous": False
                }
            ]
        }
        (self.run_dir / "trend-correlation-report.json").write_text(json.dumps(trend_report))

        # Create bundle manifest
        manifest = {
            "timestamp": "20251224-120000",
            "tars_version": "1.0.7"
        }
        (self.run_dir / "bundle-manifest.json").write_text(json.dumps(manifest))

    def test_script_exists(self):
        """Test that the narrative generator script exists."""
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
        self.assertIn("generate_executive_narrative", result.stdout)
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

    def test_narrative_file_created(self):
        """Test that narrative file is created."""
        self._create_minimal_reports()

        result = subprocess.run(
            [sys.executable, str(self.script_path), "--run-dir", str(self.run_dir)],
            capture_output=True,
            text=True,
            timeout=30
        )

        self.assertEqual(result.returncode, 0)
        narrative_path = self.run_dir / "executive-narrative.md"
        self.assertTrue(narrative_path.exists())

    def test_narrative_contains_required_sections(self):
        """Test that narrative contains all required sections."""
        self._create_minimal_reports()

        subprocess.run(
            [sys.executable, str(self.script_path), "--run-dir", str(self.run_dir)],
            capture_output=True,
            text=True,
            timeout=30
        )

        narrative_path = self.run_dir / "executive-narrative.md"
        content = narrative_path.read_text()

        # Check required sections
        self.assertIn("# T.A.R.S. Executive Narrative Report", content)
        self.assertIn("## Overall Health Status", content)
        self.assertIn("## Executive Summary", content)
        self.assertIn("## SLA Status Summary", content)
        self.assertIn("## Key Risks Identified", content)
        self.assertIn("## Notable Trends & Propagation Signals", content)
        self.assertIn("## Recommended Next Actions", content)

    def test_narrative_shows_health_tier(self):
        """Test that narrative shows health tier."""
        self._create_minimal_reports()

        subprocess.run(
            [sys.executable, str(self.script_path), "--run-dir", str(self.run_dir)],
            capture_output=True,
            text=True,
            timeout=30
        )

        narrative_path = self.run_dir / "executive-narrative.md"
        content = narrative_path.read_text()

        # Should show GREEN tier
        self.assertIn("GREEN", content)

    def test_narrative_shows_sla_summary(self):
        """Test that narrative shows SLA summary."""
        self._create_minimal_reports()

        subprocess.run(
            [sys.executable, str(self.script_path), "--run-dir", str(self.run_dir)],
            capture_output=True,
            text=True,
            timeout=30
        )

        narrative_path = self.run_dir / "executive-narrative.md"
        content = narrative_path.read_text()

        # Should include SLA metrics
        self.assertIn("Executive Readiness Score", content)
        self.assertIn("85", content)  # Readiness score
        self.assertIn("Total SLAs Evaluated", content)

    def test_narrative_with_full_reports(self):
        """Test narrative generation with full set of reports."""
        self._create_full_reports()

        result = subprocess.run(
            [sys.executable, str(self.script_path), "--run-dir", str(self.run_dir)],
            capture_output=True,
            text=True,
            timeout=30
        )

        self.assertEqual(result.returncode, 0)

        narrative_path = self.run_dir / "executive-narrative.md"
        content = narrative_path.read_text()

        # Should show leader repositories from temporal report
        self.assertIn("Leader Repositories", content)
        self.assertIn("repo-a", content)

    def test_narrative_with_no_reports(self):
        """Test that narrative generates gracefully with no reports."""
        # Run directory exists but has no reports
        result = subprocess.run(
            [sys.executable, str(self.script_path), "--run-dir", str(self.run_dir)],
            capture_output=True,
            text=True,
            timeout=30
        )

        # Should still succeed
        self.assertEqual(result.returncode, 0)

        narrative_path = self.run_dir / "executive-narrative.md"
        self.assertTrue(narrative_path.exists())

        content = narrative_path.read_text()
        # Should indicate limited data
        self.assertIn("AMBER", content)  # Default tier when no data

    def test_custom_output_path(self):
        """Test that custom output path works."""
        self._create_minimal_reports()
        custom_output = Path(self.temp_dir) / "custom" / "narrative.md"

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

    def test_narrative_no_raw_json(self):
        """Test that narrative does not contain raw JSON dumps."""
        self._create_full_reports()

        subprocess.run(
            [sys.executable, str(self.script_path), "--run-dir", str(self.run_dir)],
            capture_output=True,
            text=True,
            timeout=30
        )

        narrative_path = self.run_dir / "executive-narrative.md"
        content = narrative_path.read_text()

        # Should not contain JSON-like patterns
        self.assertNotIn('{"', content)
        self.assertNotIn("'status':", content)

    def test_verbose_mode(self):
        """Test verbose mode produces debug output."""
        self._create_minimal_reports()

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
        self.assertIn("Narrative Generation Complete", result.stderr)

    def test_red_tier_with_breach(self):
        """Test that narrative shows RED tier when SLA breach exists."""
        # Create report with breach
        sla_report = {
            "executive_readiness": {
                "readiness_score": 45,
                "tier": "RED"
            },
            "summary": {
                "total_slas": 5,
                "compliant_count": 3,
                "at_risk_count": 1,
                "breach_count": 1
            },
            "compliance_results": [
                {"sla_id": "availability", "status": "BREACHED"}
            ],
            "breaches": [
                {
                    "sla_id": "availability",
                    "root_causes": [{"cause": "Infrastructure failure", "confidence": 0.9}]
                }
            ]
        }
        (self.run_dir / "sla-intelligence-report.json").write_text(json.dumps(sla_report))

        subprocess.run(
            [sys.executable, str(self.script_path), "--run-dir", str(self.run_dir)],
            capture_output=True,
            text=True,
            timeout=30
        )

        narrative_path = self.run_dir / "executive-narrative.md"
        content = narrative_path.read_text()

        # Should show RED tier and breach info
        self.assertIn("RED", content)
        self.assertIn("CRITICAL", content)
        self.assertIn("availability", content)

    def test_narrative_footer(self):
        """Test that narrative includes proper footer."""
        self._create_minimal_reports()

        subprocess.run(
            [sys.executable, str(self.script_path), "--run-dir", str(self.run_dir)],
            capture_output=True,
            text=True,
            timeout=30
        )

        narrative_path = self.run_dir / "executive-narrative.md"
        content = narrative_path.read_text()

        self.assertIn("automatically generated", content.lower())
        self.assertIn("T.A.R.S.", content)


class TestNarrativeWithPartialData(unittest.TestCase):
    """Tests for narrative generation with partial/missing data."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.project_root = Path(__file__).parent.parent.parent
        cls.script_path = cls.project_root / "scripts" / "generate_executive_narrative.py"

    def setUp(self):
        """Create temporary directory for each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.run_dir = Path(self.temp_dir) / "tars-run-test"
        self.run_dir.mkdir(parents=True)

    def tearDown(self):
        """Clean up temporary directory."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_missing_sla_report(self):
        """Test narrative generates without SLA report."""
        # Only create org health report
        (self.run_dir / "org-health-report.json").write_text('{"risk_tier": "LOW"}')

        result = subprocess.run(
            [sys.executable, str(self.script_path), "--run-dir", str(self.run_dir)],
            capture_output=True,
            text=True,
            timeout=30
        )

        self.assertEqual(result.returncode, 0)
        narrative_path = self.run_dir / "executive-narrative.md"
        self.assertTrue(narrative_path.exists())

    def test_missing_temporal_report(self):
        """Test narrative generates without temporal report."""
        # Create minimal SLA report
        sla_report = {
            "executive_readiness": {"readiness_score": 75, "tier": "YELLOW"},
            "summary": {"total_slas": 3, "compliant_count": 2, "at_risk_count": 1, "breach_count": 0}
        }
        (self.run_dir / "sla-intelligence-report.json").write_text(json.dumps(sla_report))

        result = subprocess.run(
            [sys.executable, str(self.script_path), "--run-dir", str(self.run_dir)],
            capture_output=True,
            text=True,
            timeout=30
        )

        self.assertEqual(result.returncode, 0)
        narrative_path = self.run_dir / "executive-narrative.md"
        content = narrative_path.read_text()

        # Should indicate no significant trends
        self.assertIn("Notable Trends", content)

    def test_malformed_json_handled_gracefully(self):
        """Test that malformed JSON is handled gracefully."""
        # Create malformed JSON
        (self.run_dir / "sla-intelligence-report.json").write_text("not valid json {")

        result = subprocess.run(
            [sys.executable, str(self.script_path), "--run-dir", str(self.run_dir)],
            capture_output=True,
            text=True,
            timeout=30
        )

        # Should still succeed, just with limited data
        self.assertEqual(result.returncode, 0)

    def test_empty_json_reports_handled(self):
        """Test that empty JSON objects are handled gracefully."""
        (self.run_dir / "sla-intelligence-report.json").write_text("{}")
        (self.run_dir / "org-health-report.json").write_text("{}")

        result = subprocess.run(
            [sys.executable, str(self.script_path), "--run-dir", str(self.run_dir)],
            capture_output=True,
            text=True,
            timeout=30
        )

        self.assertEqual(result.returncode, 0)
        narrative_path = self.run_dir / "executive-narrative.md"
        self.assertTrue(narrative_path.exists())


if __name__ == "__main__":
    unittest.main()
