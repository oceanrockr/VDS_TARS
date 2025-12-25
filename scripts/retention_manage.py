#!/usr/bin/env python3
"""
T.A.R.S. Retention Management Helper

Platform-aware script for managing artifact retention (hot/warm/archive tiers).
Safely moves, compresses, and cleans up old pipeline runs.

SAFETY FIRST:
- Default mode is --dry-run (preview only)
- Explicit --apply required for actual changes
- Logs every action before execution
- Never deletes without explicit --delete flag

Usage:
    # Preview what would be cleaned up
    python scripts/retention_manage.py --runs-dir ./reports/runs --dry-run

    # Apply retention policy
    python scripts/retention_manage.py --runs-dir ./reports/runs --apply

    # Custom retention periods
    python scripts/retention_manage.py --runs-dir ./reports/runs --days-hot 14 --compress-after 14 --apply

    # Move old runs to archive
    python scripts/retention_manage.py --runs-dir ./reports/runs --archive-dir ./reports/archive --apply

Exit Codes:
    0:   Success
    1:   Configuration error
    199: General error

Version: 1.0.0
Phase: 18 - Ops Integrations
"""

import argparse
import json
import logging
import os
import platform
import shutil
import sys
import tarfile
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Exit codes
EXIT_SUCCESS = 0
EXIT_CONFIG_ERROR = 1
EXIT_GENERAL_ERROR = 199

# Import config loader if available
try:
    from scripts.tars_config import TarsConfigLoader
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    TarsConfigLoader = None


def parse_timestamp_from_dirname(dirname: str) -> Optional[datetime]:
    """
    Parse timestamp from directory name like 'tars-run-20251225-080000'.

    Returns None if parsing fails.
    """
    try:
        # Extract timestamp part (expected: YYYYMMDD-HHMMSS)
        parts = dirname.split("-")
        if len(parts) >= 4 and parts[0] == "tars" and parts[1] == "run":
            timestamp_str = f"{parts[2]}-{parts[3]}"
            return datetime.strptime(timestamp_str, "%Y%m%d-%H%M%S").replace(tzinfo=timezone.utc)
    except (ValueError, IndexError):
        pass

    # Try alternative: use modification time
    return None


def get_dir_age_days(dir_path: Path, now: datetime) -> float:
    """Get the age of a directory in days."""
    # Try to parse from directory name first
    dirname = dir_path.name
    parsed_time = parse_timestamp_from_dirname(dirname)

    if parsed_time:
        delta = now - parsed_time
        return delta.total_seconds() / 86400

    # Fall back to modification time
    try:
        mtime = datetime.fromtimestamp(dir_path.stat().st_mtime, tz=timezone.utc)
        delta = now - mtime
        return delta.total_seconds() / 86400
    except OSError:
        return 0


def get_dir_size(dir_path: Path) -> int:
    """Get total size of a directory in bytes."""
    total = 0
    try:
        for item in dir_path.rglob("*"):
            if item.is_file():
                total += item.stat().st_size
    except OSError:
        pass
    return total


def format_size(size_bytes: int) -> str:
    """Format size in human-readable format."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(size_bytes) < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


class RetentionManager:
    """Manages artifact retention for T.A.R.S. pipeline runs."""

    def __init__(
        self,
        runs_dir: str,
        archive_dir: Optional[str] = None,
        days_hot: int = 30,
        days_warm: int = 90,
        days_archive: int = 365,
        compress_after: int = 30,
        delete_after: Optional[int] = None,
        dry_run: bool = True
    ):
        self.runs_dir = Path(runs_dir).resolve()
        self.archive_dir = Path(archive_dir).resolve() if archive_dir else None
        self.days_hot = days_hot
        self.days_warm = days_warm
        self.days_archive = days_archive
        self.compress_after = compress_after
        self.delete_after = delete_after
        self.dry_run = dry_run
        self.now = datetime.now(timezone.utc)

        # Statistics
        self.stats = {
            "scanned": 0,
            "compressed": 0,
            "moved": 0,
            "deleted": 0,
            "errors": 0,
            "bytes_before": 0,
            "bytes_after": 0,
        }

    def discover_runs(self) -> List[Tuple[Path, float, int]]:
        """
        Discover all run directories.

        Returns list of (path, age_days, size_bytes) tuples.
        """
        runs = []

        if not self.runs_dir.exists():
            logger.warning(f"Runs directory does not exist: {self.runs_dir}")
            return runs

        for item in self.runs_dir.iterdir():
            if item.is_dir() and item.name.startswith("tars-run-"):
                age = get_dir_age_days(item, self.now)
                size = get_dir_size(item)
                runs.append((item, age, size))
                self.stats["bytes_before"] += size
                self.stats["scanned"] += 1

        # Sort by age (oldest first)
        runs.sort(key=lambda x: x[1], reverse=True)
        return runs

    def compress_directory(self, dir_path: Path) -> Optional[Path]:
        """Compress a directory to tar.gz."""
        archive_path = dir_path.with_suffix(".tar.gz")

        if self.dry_run:
            logger.info(f"[DRY RUN] Would compress: {dir_path} -> {archive_path.name}")
            return archive_path

        try:
            logger.info(f"Compressing: {dir_path} -> {archive_path.name}")
            with tarfile.open(archive_path, "w:gz") as tf:
                for item in dir_path.rglob("*"):
                    if item.is_file():
                        arcname = item.relative_to(dir_path.parent)
                        tf.add(item, arcname=arcname)

            # Remove original directory
            shutil.rmtree(dir_path)
            self.stats["compressed"] += 1
            self.stats["bytes_after"] += archive_path.stat().st_size
            return archive_path

        except Exception as e:
            logger.error(f"Compression failed for {dir_path}: {e}")
            self.stats["errors"] += 1
            return None

    def move_to_archive(self, path: Path) -> Optional[Path]:
        """Move a file or directory to the archive directory."""
        if not self.archive_dir:
            logger.warning("No archive directory configured")
            return None

        # Create year/month subdirectory structure
        year_month = self.now.strftime("%Y/%m")
        target_dir = self.archive_dir / year_month
        target_path = target_dir / path.name

        if self.dry_run:
            logger.info(f"[DRY RUN] Would move: {path} -> {target_path}")
            return target_path

        try:
            target_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Moving: {path} -> {target_path}")
            shutil.move(str(path), str(target_path))
            self.stats["moved"] += 1
            return target_path

        except Exception as e:
            logger.error(f"Move failed for {path}: {e}")
            self.stats["errors"] += 1
            return None

    def delete_path(self, path: Path) -> bool:
        """Delete a file or directory."""
        if self.dry_run:
            logger.info(f"[DRY RUN] Would delete: {path}")
            return True

        try:
            logger.info(f"Deleting: {path}")
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            self.stats["deleted"] += 1
            return True

        except Exception as e:
            logger.error(f"Delete failed for {path}: {e}")
            self.stats["errors"] += 1
            return False

    def apply_retention(self) -> int:
        """Apply the retention policy to all runs."""
        logger.info("=" * 60)
        logger.info("T.A.R.S. Retention Manager v1.0")
        logger.info("=" * 60)
        logger.info(f"Runs Directory: {self.runs_dir}")
        if self.archive_dir:
            logger.info(f"Archive Directory: {self.archive_dir}")
        logger.info(f"Mode: {'DRY RUN' if self.dry_run else 'APPLY'}")
        logger.info(f"Retention: hot={self.days_hot}d, compress={self.compress_after}d")
        if self.delete_after:
            logger.info(f"Delete after: {self.delete_after} days")
        logger.info("")

        # Discover runs
        runs = self.discover_runs()
        logger.info(f"Found {len(runs)} run directories")
        logger.info("")

        if not runs:
            logger.info("Nothing to process")
            return EXIT_SUCCESS

        # Process each run
        for path, age, size in runs:
            age_str = f"{age:.1f} days"
            size_str = format_size(size)
            logger.debug(f"Processing: {path.name} (age: {age_str}, size: {size_str})")

            # Determine action based on age
            if self.delete_after and age > self.delete_after:
                logger.info(f"[DELETE] {path.name} (age: {age_str})")
                self.delete_path(path)

            elif age > self.compress_after:
                # Compress and optionally move to archive
                logger.info(f"[COMPRESS] {path.name} (age: {age_str})")
                compressed = self.compress_directory(path)

                if compressed and self.archive_dir and age > self.days_warm:
                    logger.info(f"[ARCHIVE] {compressed.name}")
                    self.move_to_archive(compressed)

            elif self.archive_dir and age > self.days_hot:
                # Move to archive without compression
                logger.info(f"[ARCHIVE] {path.name} (age: {age_str})")
                self.move_to_archive(path)

            else:
                logger.debug(f"[KEEP] {path.name} (age: {age_str})")

        # Summary
        logger.info("")
        logger.info("=" * 60)
        logger.info("Retention Summary")
        logger.info("=" * 60)
        logger.info(f"Directories scanned: {self.stats['scanned']}")
        logger.info(f"Compressed: {self.stats['compressed']}")
        logger.info(f"Moved to archive: {self.stats['moved']}")
        logger.info(f"Deleted: {self.stats['deleted']}")
        logger.info(f"Errors: {self.stats['errors']}")

        if self.dry_run:
            logger.info("")
            logger.info("This was a DRY RUN. Use --apply to execute changes.")

        return EXIT_SUCCESS


def create_argument_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        prog="retention_manage",
        description="Manage T.A.R.S. pipeline run artifact retention",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exit Codes:
  0      Success
  1      Configuration error
  199    General error

Examples:
  # Preview retention actions
  python scripts/retention_manage.py --runs-dir ./reports/runs --dry-run

  # Apply retention with archiving
  python scripts/retention_manage.py --runs-dir ./reports/runs --archive-dir ./reports/archive --apply

  # Custom retention periods
  python scripts/retention_manage.py --runs-dir ./reports/runs --days-hot 14 --compress-after 14 --apply

  # Delete runs older than 1 year
  python scripts/retention_manage.py --runs-dir ./reports/runs --delete-after 365 --apply
"""
    )

    # Config file option
    parser.add_argument(
        "--config",
        default=None,
        help="Path to T.A.R.S. config file (YAML or JSON)"
    )

    # Directory options
    parser.add_argument(
        "--runs-dir",
        default="./reports/runs",
        help="Directory containing pipeline runs (default: ./reports/runs)"
    )

    parser.add_argument(
        "--archive-dir",
        default=None,
        help="Directory for archived runs (optional)"
    )

    # Retention periods
    parser.add_argument(
        "--days-hot",
        type=int,
        default=30,
        help="Days to keep in hot storage before archiving (default: 30)"
    )

    parser.add_argument(
        "--days-warm",
        type=int,
        default=90,
        help="Days to keep in warm storage (default: 90)"
    )

    parser.add_argument(
        "--days-archive",
        type=int,
        default=365,
        help="Days to keep in archive before deletion (default: 365)"
    )

    parser.add_argument(
        "--compress-after",
        type=int,
        default=30,
        help="Days before compressing runs to tar.gz (default: 30)"
    )

    parser.add_argument(
        "--delete-after",
        type=int,
        default=None,
        help="Days after which to delete runs (optional, requires explicit use)"
    )

    # Action mode
    action_group = parser.add_mutually_exclusive_group()
    action_group.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Preview actions without executing (default)"
    )

    action_group.add_argument(
        "--apply",
        action="store_true",
        help="Actually execute retention actions"
    )

    # Verbosity
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )

    return parser


def main() -> int:
    """Main entry point."""
    parser = create_argument_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load config file if available
    ret_config = {}
    if CONFIG_AVAILABLE and args.config:
        loader = TarsConfigLoader(config_path=args.config)
        config = loader.load()
        ret_config = config.get("retention", {})

    # Merge CLI args with config
    runs_dir = args.runs_dir
    archive_dir = args.archive_dir or ret_config.get("archive_dir")
    days_hot = args.days_hot if args.days_hot != 30 else ret_config.get("days_hot", 30)
    days_warm = args.days_warm if args.days_warm != 90 else ret_config.get("days_warm", 90)
    days_archive = args.days_archive if args.days_archive != 365 else ret_config.get("days_archive", 365)
    compress_after = args.compress_after if args.compress_after != 30 else ret_config.get("compress_after", 30)
    delete_after = args.delete_after

    # Determine mode
    dry_run = not args.apply

    # Validate runs directory
    if not Path(runs_dir).exists():
        logger.error(f"Runs directory does not exist: {runs_dir}")
        return EXIT_CONFIG_ERROR

    try:
        manager = RetentionManager(
            runs_dir=runs_dir,
            archive_dir=archive_dir,
            days_hot=days_hot,
            days_warm=days_warm,
            days_archive=days_archive,
            compress_after=compress_after,
            delete_after=delete_after,
            dry_run=dry_run
        )

        return manager.apply_retention()

    except KeyboardInterrupt:
        logger.info("Retention management interrupted")
        return EXIT_CONFIG_ERROR
    except Exception as e:
        logger.error(f"Retention error: {e}")
        return EXIT_GENERAL_ERROR


if __name__ == "__main__":
    sys.exit(main())
