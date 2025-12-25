#!/usr/bin/env python3
"""
Smoke tests for T.A.R.S. Retention Manager Module

Tests the retention_manage.py functionality.

Version: 1.0.0
Phase: 18 - Ops Integrations
"""

import os
import sys
import tempfile
import unittest
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import patch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.retention_manage import (
    RetentionManager,
    parse_timestamp_from_dirname,
    get_dir_age_days,
    get_dir_size,
    format_size,
    EXIT_SUCCESS,
    EXIT_CONFIG_ERROR,
)


class TestTimestampParsing(unittest.TestCase):
    """Test timestamp parsing from directory names."""

    def test_valid_dirname(self):
        """Test parsing valid directory name."""
        result = parse_timestamp_from_dirname("tars-run-20251225-080000")
        self.assertIsNotNone(result)
        self.assertEqual(result.year, 2025)
        self.assertEqual(result.month, 12)
        self.assertEqual(result.day, 25)
        self.assertEqual(result.hour, 8)
        self.assertEqual(result.minute, 0)
        self.assertEqual(result.second, 0)

    def test_invalid_dirname(self):
        """Test parsing invalid directory name."""
        result = parse_timestamp_from_dirname("invalid-name")
        self.assertIsNone(result)

    def test_partial_dirname(self):
        """Test parsing partial directory name."""
        result = parse_timestamp_from_dirname("tars-run")
        self.assertIsNone(result)

    def test_wrong_prefix(self):
        """Test parsing wrong prefix."""
        result = parse_timestamp_from_dirname("other-run-20251225-080000")
        self.assertIsNone(result)


class TestDirAgeCalculation(unittest.TestCase):
    """Test directory age calculation."""

    def test_age_from_dirname(self):
        """Test age calculation from directory name."""
        temp_dir = tempfile.mkdtemp()
        try:
            # Create directory with timestamp in name
            run_dir = Path(temp_dir) / "tars-run-20241225-080000"
            run_dir.mkdir()

            now = datetime(2025, 12, 25, 8, 0, 0, tzinfo=timezone.utc)
            age = get_dir_age_days(run_dir, now)

            # Should be approximately 365 days
            self.assertAlmostEqual(age, 365, delta=1)
        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_age_from_mtime(self):
        """Test age calculation from modification time."""
        temp_dir = tempfile.mkdtemp()
        try:
            # Create directory without timestamp in name
            run_dir = Path(temp_dir) / "other-directory"
            run_dir.mkdir()

            now = datetime.now(timezone.utc)
            age = get_dir_age_days(run_dir, now)

            # Should be close to 0 (just created)
            self.assertLess(age, 1)
        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)


class TestDirSizeCalculation(unittest.TestCase):
    """Test directory size calculation."""

    def test_empty_dir(self):
        """Test size of empty directory."""
        temp_dir = tempfile.mkdtemp()
        try:
            size = get_dir_size(Path(temp_dir))
            self.assertEqual(size, 0)
        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_dir_with_files(self):
        """Test size of directory with files."""
        temp_dir = tempfile.mkdtemp()
        try:
            # Create some files
            (Path(temp_dir) / "file1.txt").write_text("Hello")
            (Path(temp_dir) / "file2.txt").write_text("World!")

            size = get_dir_size(Path(temp_dir))
            self.assertGreater(size, 0)
        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_dir_with_subdirs(self):
        """Test size of directory with subdirectories."""
        temp_dir = tempfile.mkdtemp()
        try:
            subdir = Path(temp_dir) / "subdir"
            subdir.mkdir()
            (subdir / "file.txt").write_text("Content")

            size = get_dir_size(Path(temp_dir))
            self.assertGreater(size, 0)
        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)


class TestFormatSize(unittest.TestCase):
    """Test size formatting."""

    def test_bytes(self):
        """Test byte formatting."""
        self.assertIn("B", format_size(100))

    def test_kilobytes(self):
        """Test kilobyte formatting."""
        self.assertIn("KB", format_size(1024))

    def test_megabytes(self):
        """Test megabyte formatting."""
        self.assertIn("MB", format_size(1024 * 1024))

    def test_gigabytes(self):
        """Test gigabyte formatting."""
        self.assertIn("GB", format_size(1024 * 1024 * 1024))


class TestRetentionManager(unittest.TestCase):
    """Test RetentionManager class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.runs_dir = Path(self.temp_dir) / "runs"
        self.archive_dir = Path(self.temp_dir) / "archive"
        self.runs_dir.mkdir()
        self.archive_dir.mkdir()

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def create_run_dir(self, name: str, files: list = None):
        """Create a test run directory."""
        run_dir = self.runs_dir / name
        run_dir.mkdir()
        if files:
            for filename, content in files:
                (run_dir / filename).write_text(content)
        return run_dir

    def test_discover_runs_empty(self):
        """Test discovery in empty directory."""
        manager = RetentionManager(
            runs_dir=str(self.runs_dir),
            dry_run=True
        )
        runs = manager.discover_runs()
        self.assertEqual(len(runs), 0)

    def test_discover_runs_with_directories(self):
        """Test discovery finds run directories."""
        self.create_run_dir("tars-run-20251220-080000")
        self.create_run_dir("tars-run-20251221-080000")
        self.create_run_dir("other-directory")  # Should be ignored

        manager = RetentionManager(
            runs_dir=str(self.runs_dir),
            dry_run=True
        )
        runs = manager.discover_runs()

        # Should find 2 tars-run-* directories
        self.assertEqual(len(runs), 2)
        self.assertEqual(manager.stats["scanned"], 2)

    def test_dry_run_no_changes(self):
        """Test dry run doesn't make changes."""
        run_dir = self.create_run_dir(
            "tars-run-20241101-080000",
            [("test.txt", "content")]
        )

        manager = RetentionManager(
            runs_dir=str(self.runs_dir),
            compress_after=30,  # Should compress
            dry_run=True
        )
        result = manager.apply_retention()

        # Should succeed
        self.assertEqual(result, EXIT_SUCCESS)
        # Directory should still exist
        self.assertTrue(run_dir.exists())
        # No actual compression
        self.assertEqual(manager.stats["compressed"], 0)

    def test_compress_old_directory(self):
        """Test compressing old directory."""
        run_dir = self.create_run_dir(
            "tars-run-20241101-080000",
            [("test.txt", "content")]
        )

        manager = RetentionManager(
            runs_dir=str(self.runs_dir),
            compress_after=30,
            dry_run=False  # Actually apply
        )
        result = manager.apply_retention()

        self.assertEqual(result, EXIT_SUCCESS)
        # Original directory should be removed
        self.assertFalse(run_dir.exists())
        # Tar file should exist
        tar_file = self.runs_dir / "tars-run-20241101-080000.tar.gz"
        self.assertTrue(tar_file.exists())
        self.assertEqual(manager.stats["compressed"], 1)

    def test_move_to_archive(self):
        """Test moving to archive directory."""
        run_dir = self.create_run_dir(
            "tars-run-20241101-080000",
            [("test.txt", "content")]
        )

        manager = RetentionManager(
            runs_dir=str(self.runs_dir),
            archive_dir=str(self.archive_dir),
            days_hot=1,  # Move after 1 day
            compress_after=999,  # Don't compress
            dry_run=False
        )
        result = manager.apply_retention()

        self.assertEqual(result, EXIT_SUCCESS)
        # Original should be moved
        self.assertFalse(run_dir.exists())
        self.assertEqual(manager.stats["moved"], 1)

    def test_delete_old_directory(self):
        """Test deleting old directory."""
        run_dir = self.create_run_dir(
            "tars-run-20231101-080000",
            [("test.txt", "content")]
        )

        manager = RetentionManager(
            runs_dir=str(self.runs_dir),
            delete_after=365,  # Delete after 1 year
            dry_run=False
        )
        result = manager.apply_retention()

        self.assertEqual(result, EXIT_SUCCESS)
        self.assertFalse(run_dir.exists())
        self.assertEqual(manager.stats["deleted"], 1)

    def test_keep_recent_directory(self):
        """Test recent directories are kept."""
        # Use current timestamp
        now = datetime.now(timezone.utc)
        timestamp = now.strftime("%Y%m%d-%H%M%S")
        run_dir = self.create_run_dir(
            f"tars-run-{timestamp}",
            [("test.txt", "content")]
        )

        manager = RetentionManager(
            runs_dir=str(self.runs_dir),
            days_hot=30,
            compress_after=30,
            dry_run=False
        )
        result = manager.apply_retention()

        self.assertEqual(result, EXIT_SUCCESS)
        # Directory should still exist
        self.assertTrue(run_dir.exists())
        self.assertEqual(manager.stats["compressed"], 0)
        self.assertEqual(manager.stats["moved"], 0)
        self.assertEqual(manager.stats["deleted"], 0)

    def test_nonexistent_runs_dir(self):
        """Test handling of nonexistent runs directory."""
        manager = RetentionManager(
            runs_dir="/nonexistent/path",
            dry_run=True
        )
        runs = manager.discover_runs()
        self.assertEqual(len(runs), 0)


class TestMainFunction(unittest.TestCase):
    """Test main function and CLI."""

    def test_main_dry_run_default(self):
        """Test that dry run is default."""
        from scripts.retention_manage import main
        import sys

        temp_dir = tempfile.mkdtemp()
        try:
            runs_dir = Path(temp_dir) / "runs"
            runs_dir.mkdir()

            with patch.object(sys, "argv", [
                "retention_manage.py",
                "--runs-dir", str(runs_dir)
            ]):
                result = main()
                self.assertEqual(result, EXIT_SUCCESS)
        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_main_config_error(self):
        """Test main with nonexistent directory."""
        from scripts.retention_manage import main
        import sys

        with patch.object(sys, "argv", [
            "retention_manage.py",
            "--runs-dir", "/nonexistent/path"
        ]):
            result = main()
            self.assertEqual(result, EXIT_CONFIG_ERROR)


class TestStatisticsTracking(unittest.TestCase):
    """Test statistics tracking."""

    def test_initial_stats(self):
        """Test initial statistics are zero."""
        manager = RetentionManager(
            runs_dir=tempfile.mkdtemp(),
            dry_run=True
        )
        self.assertEqual(manager.stats["scanned"], 0)
        self.assertEqual(manager.stats["compressed"], 0)
        self.assertEqual(manager.stats["moved"], 0)
        self.assertEqual(manager.stats["deleted"], 0)
        self.assertEqual(manager.stats["errors"], 0)

    def test_stats_update_on_scan(self):
        """Test statistics update on scan."""
        temp_dir = tempfile.mkdtemp()
        try:
            runs_dir = Path(temp_dir) / "runs"
            runs_dir.mkdir()
            (runs_dir / "tars-run-20251220-080000").mkdir()
            (runs_dir / "tars-run-20251221-080000").mkdir()

            manager = RetentionManager(
                runs_dir=str(runs_dir),
                dry_run=True
            )
            manager.discover_runs()

            self.assertEqual(manager.stats["scanned"], 2)
        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
