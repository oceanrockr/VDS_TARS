#!/usr/bin/env python3
"""
Run Metadata & Provenance Generator

Generates a machine-readable run provenance artifact (run-metadata.json)
that captures comprehensive information about a T.A.R.S. pipeline execution.

Features:
    - T.A.R.S. version capture
    - Git commit hash (if available)
    - UTC timestamp
    - Host OS and Python version
    - CLI flags used
    - Exit codes per engine (if available from bundle-manifest.json)
    - Duration per stage (best-effort)

Usage:
    # Generate metadata for a completed run
    python scripts/generate_run_metadata.py --run-dir ./reports/runs/tars-run-20251222-140000

    # Generate metadata with explicit CLI flags recorded
    python scripts/generate_run_metadata.py --run-dir ./reports/runs/tars-run-20251222-140000 \
        --cli-flags "--root ./org-health --fail-on-breach"

    # Output to custom location
    python scripts/generate_run_metadata.py --run-dir ./reports/runs/tars-run-20251222-140000 \
        --output ./custom/run-metadata.json

Exit Codes:
    0:   Success, metadata generated
    1:   Run directory not found
    2:   Failed to generate metadata (warning only - non-fatal)
    199: General error

Version: 1.0.0
Phase: 17 - Post-GA Observability
"""

import argparse
import json
import logging
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Exit codes
EXIT_SUCCESS = 0
EXIT_RUN_DIR_NOT_FOUND = 1
EXIT_METADATA_FAILED = 2
EXIT_GENERAL_ERROR = 199

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_tars_version() -> str:
    """Get T.A.R.S. version from VERSION file."""
    version_file = Path(__file__).parent.parent / "VERSION"
    if version_file.exists():
        try:
            return version_file.read_text().strip()
        except Exception:
            pass
    return "unknown"


def get_git_commit() -> Optional[str]:
    """Get current git commit hash (short form)."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=Path(__file__).parent.parent
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def get_git_branch() -> Optional[str]:
    """Get current git branch name."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=Path(__file__).parent.parent
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def get_git_dirty() -> bool:
    """Check if git working directory has uncommitted changes."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=Path(__file__).parent.parent
        )
        if result.returncode == 0:
            return bool(result.stdout.strip())
    except Exception:
        pass
    return False


def get_host_info() -> Dict[str, Any]:
    """Get host system information."""
    return {
        "os": platform.system(),
        "os_version": platform.version(),
        "os_release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "hostname": platform.node(),
    }


def extract_exit_codes_from_manifest(run_dir: Path) -> Dict[str, int]:
    """Extract exit codes from bundle-manifest.json if present."""
    manifest_path = run_dir / "bundle-manifest.json"
    exit_codes = {}

    if manifest_path.exists():
        try:
            with open(manifest_path, "r") as f:
                manifest = json.load(f)

            if "steps" in manifest:
                for step in manifest["steps"]:
                    if step.get("exit_code") is not None:
                        step_name = step.get("name", "unknown")
                        exit_codes[step_name] = step["exit_code"]
        except Exception as e:
            logger.warning(f"Could not extract exit codes from manifest: {e}")

    return exit_codes


def extract_stage_durations(run_dir: Path) -> Dict[str, Optional[float]]:
    """Extract stage durations from bundle-manifest.json if present."""
    manifest_path = run_dir / "bundle-manifest.json"
    durations = {}

    if manifest_path.exists():
        try:
            with open(manifest_path, "r") as f:
                manifest = json.load(f)

            # Get overall duration
            if "pipeline" in manifest:
                pipeline = manifest["pipeline"]
                if pipeline.get("duration_seconds") is not None:
                    durations["total_pipeline"] = pipeline["duration_seconds"]

            # Get per-step durations if available
            if "steps" in manifest:
                for step in manifest["steps"]:
                    step_name = step.get("name", "unknown")
                    # Duration per step not currently tracked, but placeholder for future
                    durations[step_name] = None

        except Exception as e:
            logger.warning(f"Could not extract durations from manifest: {e}")

    return durations


def extract_pipeline_config(run_dir: Path) -> Dict[str, Any]:
    """Extract pipeline configuration from bundle-manifest.json if present."""
    manifest_path = run_dir / "bundle-manifest.json"
    config = {}

    if manifest_path.exists():
        try:
            with open(manifest_path, "r") as f:
                manifest = json.load(f)

            if "pipeline" in manifest:
                pipeline = manifest["pipeline"]
                config = {
                    "root_dir": pipeline.get("root_dir"),
                    "output_dir": pipeline.get("output_dir"),
                    "output_format": pipeline.get("output_format"),
                    "sla_policy": pipeline.get("sla_policy"),
                    "windows": pipeline.get("windows"),
                    "dry_run": pipeline.get("dry_run"),
                }

        except Exception as e:
            logger.warning(f"Could not extract pipeline config: {e}")

    return config


def list_artifacts(run_dir: Path) -> List[Dict[str, Any]]:
    """List all artifacts in the run directory with metadata."""
    artifacts = []

    for item in run_dir.rglob("*"):
        if item.is_file():
            rel_path = item.relative_to(run_dir)
            try:
                stat = item.stat()
                artifacts.append({
                    "path": str(rel_path),
                    "size_bytes": stat.st_size,
                    "modified_at": datetime.fromtimestamp(
                        stat.st_mtime, tz=timezone.utc
                    ).isoformat(),
                })
            except Exception:
                artifacts.append({
                    "path": str(rel_path),
                    "size_bytes": None,
                    "modified_at": None,
                })

    return sorted(artifacts, key=lambda x: x["path"])


class RunMetadataGenerator:
    """Generates run metadata and provenance artifacts."""

    def __init__(
        self,
        run_dir: str,
        output_path: Optional[str] = None,
        cli_flags: Optional[str] = None
    ):
        self.run_dir = Path(run_dir).resolve()
        self.cli_flags = cli_flags

        # Output path defaults to run-metadata.json inside run directory
        if output_path:
            self.output_path = Path(output_path).resolve()
        else:
            self.output_path = self.run_dir / "run-metadata.json"

        # Metadata collection
        self.metadata: Dict[str, Any] = {}

    def validate_run_dir(self) -> bool:
        """Validate that the run directory exists."""
        if not self.run_dir.exists():
            logger.error(f"Run directory does not exist: {self.run_dir}")
            return False

        if not self.run_dir.is_dir():
            logger.error(f"Run path is not a directory: {self.run_dir}")
            return False

        return True

    def collect_metadata(self) -> Dict[str, Any]:
        """Collect all metadata for the run."""
        generation_time = datetime.now(timezone.utc)

        metadata = {
            "metadata_version": "1.0.0",
            "metadata_schema": "tars-run-metadata-v1",
            "generated_at": generation_time.isoformat(),
            "generator": {
                "name": "T.A.R.S. Run Metadata Generator",
                "version": "1.0.0",
                "phase": "17"
            },

            # T.A.R.S. version info
            "tars": {
                "version": get_tars_version(),
                "git_commit": get_git_commit(),
                "git_branch": get_git_branch(),
                "git_dirty": get_git_dirty(),
            },

            # Host environment
            "environment": get_host_info(),

            # Run directory info
            "run": {
                "directory": str(self.run_dir),
                "directory_name": self.run_dir.name,
                "cli_flags": self.cli_flags,
            },

            # Pipeline configuration (from manifest if available)
            "pipeline_config": extract_pipeline_config(self.run_dir),

            # Exit codes per engine
            "exit_codes": extract_exit_codes_from_manifest(self.run_dir),

            # Stage durations (best-effort)
            "durations": extract_stage_durations(self.run_dir),

            # Artifacts list
            "artifacts": list_artifacts(self.run_dir),

            # Provenance
            "provenance": {
                "build_type": "automated",
                "reproducible": True,
                "attestation": {
                    "type": "tars-provenance-v1",
                    "generated_at": generation_time.isoformat(),
                    "generator_identity": "T.A.R.S. Pipeline"
                }
            }
        }

        self.metadata = metadata
        return metadata

    def write_metadata(self) -> bool:
        """Write metadata to output file."""
        try:
            # Ensure parent directory exists
            self.output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(self.output_path, "w") as f:
                json.dump(self.metadata, f, indent=2, default=str)

            logger.info(f"Written run metadata: {self.output_path}")
            return True

        except Exception as e:
            logger.warning(f"Failed to write metadata: {e}")
            return False

    def generate(self) -> int:
        """Generate run metadata."""
        logger.info("=" * 60)
        logger.info("T.A.R.S. Run Metadata Generator v1.0")
        logger.info("=" * 60)
        logger.info(f"Run Directory: {self.run_dir}")
        logger.info(f"Output Path: {self.output_path}")
        logger.info("")

        # Validate run directory
        if not self.validate_run_dir():
            return EXIT_RUN_DIR_NOT_FOUND

        # Collect metadata
        logger.info("Collecting metadata...")
        self.collect_metadata()

        # Write metadata
        if not self.write_metadata():
            logger.warning("Metadata generation failed (non-fatal)")
            return EXIT_METADATA_FAILED

        # Summary
        logger.info("")
        logger.info("=" * 60)
        logger.info("Metadata Generation Complete")
        logger.info("=" * 60)
        logger.info(f"T.A.R.S. Version: {self.metadata.get('tars', {}).get('version', 'unknown')}")
        logger.info(f"Git Commit: {self.metadata.get('tars', {}).get('git_commit', 'unknown')}")
        logger.info(f"Artifacts Found: {len(self.metadata.get('artifacts', []))}")
        logger.info(f"Exit Codes Captured: {len(self.metadata.get('exit_codes', {}))}")
        logger.info(f"Output: {self.output_path}")

        return EXIT_SUCCESS


def integrate_with_orchestrator(run_dir: Path, cli_flags: Optional[str] = None) -> bool:
    """
    Integration function to be called from the pipeline orchestrator.

    Returns True on success, False on failure.
    Silent failure allowed (logs warning only).
    """
    try:
        generator = RunMetadataGenerator(
            run_dir=str(run_dir),
            cli_flags=cli_flags
        )
        result = generator.generate()
        return result == EXIT_SUCCESS
    except Exception as e:
        logger.warning(f"Metadata generation failed (non-fatal): {e}")
        return False


def create_argument_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        prog="generate_run_metadata",
        description="Generate run metadata and provenance artifacts for T.A.R.S. pipeline runs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exit Codes:
  0      Success, metadata generated
  1      Run directory not found
  2      Failed to generate metadata (warning only)
  199    General error

Examples:
  # Generate metadata for a completed run
  python scripts/generate_run_metadata.py --run-dir ./reports/runs/tars-run-20251222-140000

  # Generate with CLI flags recorded
  python scripts/generate_run_metadata.py --run-dir ./reports/runs/tars-run-20251222-140000 \\
      --cli-flags "--root ./org-health --fail-on-breach"

  # Custom output location
  python scripts/generate_run_metadata.py --run-dir ./reports/runs/tars-run-20251222-140000 \\
      --output ./custom/run-metadata.json
"""
    )

    # Required arguments
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Path to a completed orchestrator run directory"
    )

    # Optional arguments
    parser.add_argument(
        "--output",
        default=None,
        help="Custom output path for run-metadata.json (default: <run-dir>/run-metadata.json)"
    )

    parser.add_argument(
        "--cli-flags",
        default=None,
        help="CLI flags used when running the pipeline (for provenance tracking)"
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

    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        generator = RunMetadataGenerator(
            run_dir=args.run_dir,
            output_path=args.output,
            cli_flags=args.cli_flags
        )

        return generator.generate()

    except KeyboardInterrupt:
        logger.info("Metadata generation interrupted by user")
        return EXIT_GENERAL_ERROR
    except Exception as e:
        logger.error(f"Metadata generation error: {e}")
        return EXIT_GENERAL_ERROR


if __name__ == "__main__":
    sys.exit(main())
