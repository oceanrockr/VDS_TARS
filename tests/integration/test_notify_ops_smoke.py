#!/usr/bin/env python3
"""
Smoke tests for T.A.R.S. Notification Hook Module

Tests the notify_ops.py notification functionality.

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

from scripts.notify_ops import (
    EXIT_CODE_SEVERITY_MAP,
    EXIT_CODE_ACTION_MAP,
    EXIT_CODE_TITLE_MAP,
    build_notification_payload,
    get_tars_version,
    load_run_metadata,
    load_executive_readiness,
)


class TestExitCodeMappings(unittest.TestCase):
    """Test exit code mapping constants."""

    def test_severity_map_has_critical_codes(self):
        """Test severity map includes critical exit codes."""
        critical_codes = [92, 102, 122, 132, 142]
        for code in critical_codes:
            self.assertIn(code, EXIT_CODE_SEVERITY_MAP)

    def test_severity_map_values(self):
        """Test severity values are valid."""
        valid_severities = {"SEV-1", "SEV-2", "SEV-3", "SEV-4", "INFO"}
        for severity in EXIT_CODE_SEVERITY_MAP.values():
            self.assertIn(severity, valid_severities)

    def test_action_map_has_recommendations(self):
        """Test action map has non-empty recommendations."""
        for code, action in EXIT_CODE_ACTION_MAP.items():
            self.assertTrue(len(action) > 0)

    def test_title_map_has_titles(self):
        """Test title map has non-empty titles."""
        for code, title in EXIT_CODE_TITLE_MAP.items():
            self.assertTrue(len(title) > 0)

    def test_maps_have_same_keys(self):
        """Test all maps have consistent keys."""
        severity_keys = set(EXIT_CODE_SEVERITY_MAP.keys())
        action_keys = set(EXIT_CODE_ACTION_MAP.keys())
        title_keys = set(EXIT_CODE_TITLE_MAP.keys())

        # All should have the same keys
        self.assertEqual(severity_keys, action_keys)
        self.assertEqual(severity_keys, title_keys)


class TestBuildNotificationPayload(unittest.TestCase):
    """Test build_notification_payload function."""

    def test_basic_payload_structure(self):
        """Test basic payload has required fields."""
        payload = build_notification_payload(
            title="Test Title",
            message="Test message",
            severity="SEV-2",
            exit_code=None,
            run_dir=None
        )

        self.assertEqual(payload["title"], "Test Title")
        self.assertEqual(payload["message"], "Test message")
        self.assertEqual(payload["severity"], "SEV-2")
        self.assertEqual(payload["source"], "T.A.R.S.")
        self.assertIn("timestamp", payload)
        self.assertIn("version", payload)

    def test_payload_with_exit_code(self):
        """Test payload includes exit code details."""
        payload = build_notification_payload(
            title="Test",
            message="Test",
            severity="SEV-1",
            exit_code=142,
            run_dir=None
        )

        self.assertEqual(payload["exit_code"], 142)
        self.assertIn("recommended_action", payload)
        self.assertTrue(len(payload["recommended_action"]) > 0)

    def test_payload_with_unknown_exit_code(self):
        """Test payload handles unknown exit code."""
        payload = build_notification_payload(
            title="Test",
            message="Test",
            severity="SEV-3",
            exit_code=999,
            run_dir=None
        )

        self.assertEqual(payload["exit_code"], 999)
        self.assertIn("recommended_action", payload)
        self.assertIn("OPS_RUNBOOK", payload["recommended_action"])

    def test_payload_with_run_dir(self):
        """Test payload includes run directory."""
        temp_dir = tempfile.mkdtemp()
        try:
            run_dir = Path(temp_dir)
            payload = build_notification_payload(
                title="Test",
                message="Test",
                severity="INFO",
                exit_code=0,
                run_dir=run_dir
            )

            self.assertIn("run_dir", payload)
            self.assertEqual(payload["run_dir"], str(run_dir))
        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_payload_with_metadata(self):
        """Test payload extracts metadata from run directory."""
        temp_dir = tempfile.mkdtemp()
        try:
            run_dir = Path(temp_dir)

            # Create run-metadata.json
            metadata = {
                "tars_version": "1.0.8",
                "git_commit": "abc123",
                "duration_seconds": 45.5
            }
            with open(run_dir / "run-metadata.json", "w") as f:
                json.dump(metadata, f)

            payload = build_notification_payload(
                title="Test",
                message="Test",
                severity="INFO",
                exit_code=0,
                run_dir=run_dir
            )

            self.assertIn("metadata", payload)
            self.assertEqual(payload["metadata"]["tars_version"], "1.0.8")
            self.assertEqual(payload["metadata"]["git_commit"], "abc123")
        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_payload_with_sla_report(self):
        """Test payload extracts executive readiness from SLA report."""
        temp_dir = tempfile.mkdtemp()
        try:
            run_dir = Path(temp_dir)

            # Create sla-intelligence-report.json
            sla_report = {
                "executive_readiness": {
                    "tier": "YELLOW",
                    "readiness_score": 72
                }
            }
            with open(run_dir / "sla-intelligence-report.json", "w") as f:
                json.dump(sla_report, f)

            payload = build_notification_payload(
                title="Test",
                message="Test",
                severity="SEV-3",
                exit_code=141,
                run_dir=run_dir
            )

            self.assertIn("executive_readiness_tier", payload)
            self.assertEqual(payload["executive_readiness_tier"], "YELLOW")
        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_payload_with_extra_metadata(self):
        """Test payload includes extra metadata."""
        payload = build_notification_payload(
            title="Test",
            message="Test",
            severity="INFO",
            exit_code=0,
            run_dir=None,
            extra_metadata={"custom_field": "custom_value"}
        )

        self.assertIn("extra", payload)
        self.assertEqual(payload["extra"]["custom_field"], "custom_value")


class TestLoadFunctions(unittest.TestCase):
    """Test load helper functions."""

    def test_load_run_metadata_success(self):
        """Test loading run metadata from file."""
        temp_dir = tempfile.mkdtemp()
        try:
            run_dir = Path(temp_dir)
            metadata = {"tars_version": "1.0.8", "test": True}
            with open(run_dir / "run-metadata.json", "w") as f:
                json.dump(metadata, f)

            result = load_run_metadata(run_dir)
            self.assertEqual(result["tars_version"], "1.0.8")
            self.assertTrue(result["test"])
        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_load_run_metadata_missing(self):
        """Test loading metadata when file is missing."""
        temp_dir = tempfile.mkdtemp()
        try:
            run_dir = Path(temp_dir)
            result = load_run_metadata(run_dir)
            self.assertEqual(result, {})
        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_load_run_metadata_invalid_json(self):
        """Test loading metadata with invalid JSON."""
        temp_dir = tempfile.mkdtemp()
        try:
            run_dir = Path(temp_dir)
            with open(run_dir / "run-metadata.json", "w") as f:
                f.write("{ invalid json }")

            result = load_run_metadata(run_dir)
            self.assertEqual(result, {})
        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_load_executive_readiness_success(self):
        """Test loading executive readiness tier."""
        temp_dir = tempfile.mkdtemp()
        try:
            run_dir = Path(temp_dir)
            sla_report = {"executive_readiness": {"tier": "GREEN"}}
            with open(run_dir / "sla-intelligence-report.json", "w") as f:
                json.dump(sla_report, f)

            result = load_executive_readiness(run_dir)
            self.assertEqual(result, "GREEN")
        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_load_executive_readiness_missing(self):
        """Test loading readiness when file is missing."""
        temp_dir = tempfile.mkdtemp()
        try:
            run_dir = Path(temp_dir)
            result = load_executive_readiness(run_dir)
            self.assertIsNone(result)
        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_get_tars_version(self):
        """Test get_tars_version returns a string."""
        version = get_tars_version()
        self.assertIsInstance(version, str)
        self.assertTrue(len(version) > 0)


class TestWebhookSending(unittest.TestCase):
    """Test webhook sending functions."""

    @patch("scripts.notify_ops.urllib.request.urlopen")
    def test_send_webhook_success(self, mock_urlopen):
        """Test successful webhook send."""
        from scripts.notify_ops import send_webhook

        mock_response = MagicMock()
        mock_response.getcode.return_value = 200
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        payload = {"test": "data"}
        result = send_webhook("https://example.com/webhook", payload)

        self.assertTrue(result)
        mock_urlopen.assert_called_once()

    @patch("scripts.notify_ops.urllib.request.urlopen")
    def test_send_webhook_failure(self, mock_urlopen):
        """Test webhook send failure."""
        from scripts.notify_ops import send_webhook
        import urllib.error

        mock_urlopen.side_effect = urllib.error.URLError("Connection failed")

        payload = {"test": "data"}
        result = send_webhook("https://example.com/webhook", payload)

        self.assertFalse(result)

    @patch("scripts.notify_ops.urllib.request.urlopen")
    def test_send_slack_webhook_format(self, mock_urlopen):
        """Test Slack webhook formats payload correctly."""
        from scripts.notify_ops import send_slack_webhook

        mock_response = MagicMock()
        mock_response.getcode.return_value = 200
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        payload = {
            "severity": "SEV-1",
            "title": "Test Alert",
            "message": "Test message",
            "exit_code": 142,
            "version": "1.0.8"
        }
        result = send_slack_webhook("https://hooks.slack.com/test", payload)

        self.assertTrue(result)
        mock_urlopen.assert_called_once()

        # Verify Slack-formatted payload was sent
        call_args = mock_urlopen.call_args
        request = call_args[0][0]
        sent_data = json.loads(request.data.decode("utf-8"))

        self.assertIn("attachments", sent_data)
        self.assertTrue(len(sent_data["attachments"]) > 0)
        self.assertIn("color", sent_data["attachments"][0])
        self.assertIn("title", sent_data["attachments"][0])


class TestMainFunction(unittest.TestCase):
    """Test main function and CLI."""

    def test_main_no_destination(self):
        """Test main returns 0 when no destination configured."""
        from scripts.notify_ops import main
        import sys
        from unittest.mock import patch

        with patch.object(sys, "argv", ["notify_ops.py", "--title", "Test"]):
            result = main()
            # Should return 0 (never fail)
            self.assertEqual(result, 0)

    def test_main_dry_run(self):
        """Test main with dry-run mode."""
        from scripts.notify_ops import main
        import sys
        from io import StringIO

        with patch.object(sys, "argv", [
            "notify_ops.py",
            "--title", "Test",
            "--message", "Test message",
            "--severity", "SEV-2",
            "--dry-run"
        ]):
            with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
                result = main()
                output = mock_stdout.getvalue()

                self.assertEqual(result, 0)
                # Dry run prints payload
                self.assertIn("{", output)

    def test_main_exit_code_auto_severity(self):
        """Test that exit code auto-determines severity."""
        from scripts.notify_ops import main
        import sys
        from io import StringIO

        with patch.object(sys, "argv", [
            "notify_ops.py",
            "--exit-code", "142",
            "--dry-run"
        ]):
            with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
                result = main()
                output = mock_stdout.getvalue()

                self.assertEqual(result, 0)
                # Should have SEV-1 for exit code 142
                self.assertIn("SEV-1", output)


if __name__ == "__main__":
    unittest.main(verbosity=2)
