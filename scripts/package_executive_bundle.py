#!/usr/bin/env python3
"""
Executive Bundle Packager for T.A.R.S. Organization Governance Reports

Creates a single, deployable archive containing all governance reports from
a completed pipeline run, with manifest and checksum verification.

Usage:
    # Basic packaging (zip archive)
    python scripts/package_executive_bundle.py --run-dir ./reports/runs/tars-run-20251222-140000

    # Custom output directory and bundle name
    python scripts/package_executive_bundle.py --run-dir ./reports/runs/tars-run-20251222-140000 \
        --output-dir ./release/executive \
        --bundle-name executive-report-Q4

    # Include tar.gz archive
    python scripts/package_executive_bundle.py --run-dir ./reports/runs/tars-run-20251222-140000 --tar

    # Skip checksums
    python scripts/package_executive_bundle.py --run-dir ./reports/runs/tars-run-20251222-140000 --no-checksums

Exit Codes:
    0:   Success, bundle created
    1:   Run directory not found or invalid
    2:   No files to package
    3:   Archive creation failed
    199: General error

Version: 1.1.0
Phase: 17 - Post-GA Observability
"""

import argparse
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import tarfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any

# Exit codes
EXIT_SUCCESS = 0
EXIT_INVALID_RUN_DIR = 1
EXIT_NO_FILES = 2
EXIT_ARCHIVE_FAILED = 3
EXIT_GENERAL_ERROR = 199

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def generate_timestamp() -> str:
    """Generate a UTC timestamp in YYYYMMDD-HHMMSS format."""
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def compute_sha256(file_path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            sha256_hash.update(block)
    return sha256_hash.hexdigest()


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
    """Get current git commit hash."""
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


def collect_files(run_dir: Path) -> List[Path]:
    """Collect all files from the run directory."""
    files = []
    for item in run_dir.rglob("*"):
        if item.is_file():
            files.append(item)
    return sorted(files)


def extract_exit_codes(run_dir: Path) -> Dict[str, int]:
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


# Compliance mapping for SOC-2 and ISO 27001
COMPLIANCE_MAPPING = {
    "org-health-report.json": {
        "soc2": ["CC6.1", "CC7.1"],  # System Operations, Risk Assessment
        "iso27001": ["A.12.1", "A.12.6"],  # Operations Security, Technical Vulnerability Management
        "incident_evidence": True,
        "description": "Organization health status and aggregate metrics"
    },
    "org-alerts.json": {
        "soc2": ["CC7.2", "CC7.3"],  # Anomaly Detection, Incident Response
        "iso27001": ["A.16.1"],  # Information Security Incident Management
        "incident_evidence": True,
        "description": "Active alerts and escalation status"
    },
    "trend-correlation-report.json": {
        "soc2": ["CC7.1"],  # Risk Assessment
        "iso27001": ["A.12.6"],  # Technical Vulnerability Management
        "incident_evidence": False,
        "description": "Cross-repository trend correlation analysis"
    },
    "temporal-intelligence-report.json": {
        "soc2": ["CC7.1", "CC7.2"],  # Risk Assessment, Anomaly Detection
        "iso27001": ["A.12.6"],  # Technical Vulnerability Management
        "incident_evidence": True,
        "description": "Time-lagged correlation and propagation analysis"
    },
    "sla-intelligence-report.json": {
        "soc2": ["CC6.1", "CC7.1"],  # System Operations, Risk Assessment
        "iso27001": ["A.12.1", "A.18.2"],  # Operations Security, Compliance Reviews
        "incident_evidence": True,
        "description": "SLA compliance, executive readiness, and breach attribution"
    },
    "executive-summary.md": {
        "soc2": ["CC7.3"],  # Incident Response
        "iso27001": ["A.18.2"],  # Compliance Reviews
        "incident_evidence": True,
        "description": "Executive summary for leadership review"
    },
    "executive-narrative.md": {
        "soc2": ["CC7.3"],  # Incident Response
        "iso27001": ["A.18.2"],  # Compliance Reviews
        "incident_evidence": True,
        "description": "Plain-English executive narrative"
    },
    "run-metadata.json": {
        "soc2": ["CC6.1"],  # System Operations
        "iso27001": ["A.12.4"],  # Logging and Monitoring
        "incident_evidence": True,
        "description": "Run provenance and metadata"
    },
    "bundle-manifest.json": {
        "soc2": ["CC6.1"],  # System Operations
        "iso27001": ["A.12.4"],  # Logging and Monitoring
        "incident_evidence": False,
        "description": "Pipeline execution manifest"
    },
}


class ExecutiveBundlePackager:
    """Packages T.A.R.S. governance reports into a single executive bundle."""

    def __init__(
        self,
        run_dir: str,
        output_dir: str,
        bundle_name: Optional[str] = None,
        create_zip: bool = True,
        create_tar: bool = False,
        create_checksums: bool = True,
        create_manifest: bool = True,
        create_compliance_index: bool = True
    ):
        self.run_dir = Path(run_dir).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.create_zip = create_zip
        self.create_tar = create_tar
        self.create_checksums = create_checksums
        self.create_manifest = create_manifest
        self.create_compliance_index = create_compliance_index

        # Generate bundle name if not provided
        self.version = get_tars_version()
        self.timestamp = generate_timestamp()
        self.git_commit = get_git_commit()

        if bundle_name:
            self.bundle_name = bundle_name
        else:
            self.bundle_name = f"tars-exec-bundle-{self.version}-{self.timestamp}"

        # State
        self.files: List[Path] = []
        self.checksums: Dict[str, str] = {}
        self.archive_paths: List[Path] = []

    def validate_run_dir(self) -> bool:
        """Validate that the run directory exists and contains files."""
        if not self.run_dir.exists():
            logger.error(f"Run directory does not exist: {self.run_dir}")
            return False

        if not self.run_dir.is_dir():
            logger.error(f"Run path is not a directory: {self.run_dir}")
            return False

        self.files = collect_files(self.run_dir)
        if not self.files:
            logger.error(f"No files found in run directory: {self.run_dir}")
            return False

        logger.info(f"Found {len(self.files)} files in run directory")
        return True

    def compute_checksums(self) -> Dict[str, str]:
        """Compute SHA-256 checksums for all files."""
        checksums = {}
        for file_path in self.files:
            relative_path = file_path.relative_to(self.run_dir)
            checksum = compute_sha256(file_path)
            checksums[str(relative_path)] = checksum
            logger.debug(f"Checksum: {relative_path} = {checksum[:16]}...")
        return checksums

    def generate_manifest(self) -> Dict[str, Any]:
        """Generate manifest for the executive bundle."""
        exit_codes = extract_exit_codes(self.run_dir)

        return {
            "manifest_version": "1.0",
            "bundle_name": self.bundle_name,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "generator": "T.A.R.S. Executive Bundle Packager",
            "packager_version": "1.0.0",
            "tars_version": self.version,
            "git_commit": self.git_commit,
            "timestamp": self.timestamp,
            "source": {
                "run_dir": str(self.run_dir),
                "file_count": len(self.files)
            },
            "included_files": [
                str(f.relative_to(self.run_dir)) for f in self.files
            ],
            "exit_codes": exit_codes,
            "checksums": self.checksums if self.create_checksums else None,
            "archives": {
                "zip": self.create_zip,
                "tar_gz": self.create_tar
            }
        }

    def write_checksums_file(self, output_path: Path) -> None:
        """Write checksums to a SHA-256 checksum file."""
        with open(output_path, "w") as f:
            f.write(f"# SHA-256 checksums for {self.bundle_name}\n")
            f.write(f"# Generated: {datetime.now(timezone.utc).isoformat()}\n")
            f.write(f"# T.A.R.S. Version: {self.version}\n")
            f.write("#\n")
            for filename, checksum in sorted(self.checksums.items()):
                f.write(f"{checksum}  {filename}\n")
        logger.info(f"Written checksums file: {output_path}")

    def generate_compliance_index(self) -> str:
        """Generate compliance index markdown."""
        lines = []
        generation_time = datetime.now(timezone.utc)

        # Header
        lines.append("# T.A.R.S. Executive Bundle Compliance Index")
        lines.append("")
        lines.append(f"**Generated:** {generation_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        lines.append(f"**Bundle Name:** {self.bundle_name}")
        lines.append(f"**T.A.R.S. Version:** {self.version}")
        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append("## Disclaimer")
        lines.append("")
        lines.append("*This compliance index is informational only and does not constitute legal or audit advice.*")
        lines.append("*Mappings are high-level references to assist auditors in locating relevant evidence.*")
        lines.append("")
        lines.append("---")
        lines.append("")

        # Included Artifacts
        lines.append("## Included Artifacts")
        lines.append("")
        lines.append("| File | Description | SHA-256 Hash (first 16 chars) |")
        lines.append("|------|-------------|-------------------------------|")

        for file_path in self.files:
            rel_path = file_path.relative_to(self.run_dir)
            filename = str(rel_path)
            checksum = self.checksums.get(filename, "N/A")[:16] + "..." if filename in self.checksums else "N/A"

            # Get description from mapping or default
            mapping = COMPLIANCE_MAPPING.get(filename, {})
            description = mapping.get("description", "Supporting artifact")

            lines.append(f"| {filename} | {description} | {checksum} |")

        lines.append("")
        lines.append("---")
        lines.append("")

        # SOC-2 Control Mapping
        lines.append("## SOC-2 Type II Control Mapping")
        lines.append("")
        lines.append("| Control | Description | Evidence Files |")
        lines.append("|---------|-------------|----------------|")

        soc2_controls = {
            "CC6.1": "System Operations - Logical and Physical Access",
            "CC7.1": "Risk Assessment - Risk Identification and Analysis",
            "CC7.2": "Anomaly Detection - Security Event Monitoring",
            "CC7.3": "Incident Response - Security Incident Management",
        }

        for control_id, description in soc2_controls.items():
            evidence_files = []
            for file_path in self.files:
                rel_path = str(file_path.relative_to(self.run_dir))
                mapping = COMPLIANCE_MAPPING.get(rel_path, {})
                if control_id in mapping.get("soc2", []):
                    evidence_files.append(rel_path)
            if evidence_files:
                lines.append(f"| {control_id} | {description} | {', '.join(evidence_files)} |")
            else:
                lines.append(f"| {control_id} | {description} | - |")

        lines.append("")
        lines.append("---")
        lines.append("")

        # ISO 27001 Control Mapping
        lines.append("## ISO 27001 Control Mapping")
        lines.append("")
        lines.append("| Control | Description | Evidence Files |")
        lines.append("|---------|-------------|----------------|")

        iso_controls = {
            "A.12.1": "Operations Security - Operational Procedures",
            "A.12.4": "Logging and Monitoring",
            "A.12.6": "Technical Vulnerability Management",
            "A.16.1": "Information Security Incident Management",
            "A.18.2": "Information Security Reviews",
        }

        for control_id, description in iso_controls.items():
            evidence_files = []
            for file_path in self.files:
                rel_path = str(file_path.relative_to(self.run_dir))
                mapping = COMPLIANCE_MAPPING.get(rel_path, {})
                if control_id in mapping.get("iso27001", []):
                    evidence_files.append(rel_path)
            if evidence_files:
                lines.append(f"| {control_id} | {description} | {', '.join(evidence_files)} |")
            else:
                lines.append(f"| {control_id} | {description} | - |")

        lines.append("")
        lines.append("---")
        lines.append("")

        # Incident Response Evidence Markers
        lines.append("## Incident Response Evidence")
        lines.append("")
        lines.append("The following artifacts are marked as suitable for incident response evidence collection:")
        lines.append("")

        incident_files = []
        for file_path in self.files:
            rel_path = str(file_path.relative_to(self.run_dir))
            mapping = COMPLIANCE_MAPPING.get(rel_path, {})
            if mapping.get("incident_evidence", False):
                incident_files.append(rel_path)

        if incident_files:
            for f in incident_files:
                lines.append(f"- {f}")
        else:
            lines.append("- No incident evidence markers found")

        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append("*End of Compliance Index*")
        lines.append("")

        return "\n".join(lines)

    def write_compliance_index(self, output_path: Path) -> None:
        """Write compliance index to a markdown file."""
        content = self.generate_compliance_index()
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)
        logger.info(f"Written compliance index: {output_path}")

    def create_zip_archive(self, output_path: Path) -> bool:
        """Create a ZIP archive of all files."""
        try:
            with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for file_path in self.files:
                    relative_path = file_path.relative_to(self.run_dir)
                    zf.write(file_path, relative_path)

            logger.info(f"Created ZIP archive: {output_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to create ZIP archive: {e}")
            return False

    def create_tar_archive(self, output_path: Path) -> bool:
        """Create a tar.gz archive of all files."""
        try:
            with tarfile.open(output_path, "w:gz") as tf:
                for file_path in self.files:
                    relative_path = file_path.relative_to(self.run_dir)
                    tf.add(file_path, arcname=relative_path)

            logger.info(f"Created tar.gz archive: {output_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to create tar.gz archive: {e}")
            return False

    def compute_archive_checksum(self, archive_path: Path) -> str:
        """Compute checksum of the archive itself."""
        checksum = compute_sha256(archive_path)
        checksum_path = archive_path.with_suffix(archive_path.suffix + ".sha256")
        with open(checksum_path, "w") as f:
            f.write(f"{checksum}  {archive_path.name}\n")
        logger.info(f"Archive checksum: {checksum_path}")
        return checksum

    def package(self) -> int:
        """Create the executive bundle package."""
        logger.info("=" * 60)
        logger.info("T.A.R.S. Executive Bundle Packager v1.0")
        logger.info("=" * 60)
        logger.info(f"Run Directory: {self.run_dir}")
        logger.info(f"Output Directory: {self.output_dir}")
        logger.info(f"Bundle Name: {self.bundle_name}")
        logger.info("")

        # Validate run directory
        if not self.validate_run_dir():
            return EXIT_INVALID_RUN_DIR

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Compute checksums
        if self.create_checksums:
            logger.info("Computing checksums...")
            self.checksums = self.compute_checksums()

        # Generate manifest
        if self.create_manifest:
            logger.info("Generating manifest...")
            manifest = self.generate_manifest()
            manifest_path = self.output_dir / f"{self.bundle_name}-manifest.json"
            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=2)
            logger.info(f"Generated manifest: {manifest_path}")

        # Write checksums file
        if self.create_checksums:
            checksums_path = self.output_dir / f"{self.bundle_name}-checksums.sha256"
            self.write_checksums_file(checksums_path)

        # Generate compliance index (Phase 17)
        if self.create_compliance_index:
            logger.info("Generating compliance index...")
            compliance_path = self.output_dir / f"{self.bundle_name}-compliance-index.md"
            self.write_compliance_index(compliance_path)

        # Create archives
        archive_success = True

        if self.create_zip:
            zip_path = self.output_dir / f"{self.bundle_name}.zip"
            if self.create_zip_archive(zip_path):
                self.archive_paths.append(zip_path)
                if self.create_checksums:
                    self.compute_archive_checksum(zip_path)
            else:
                archive_success = False

        if self.create_tar:
            tar_path = self.output_dir / f"{self.bundle_name}.tar.gz"
            if self.create_tar_archive(tar_path):
                self.archive_paths.append(tar_path)
                if self.create_checksums:
                    self.compute_archive_checksum(tar_path)
            else:
                archive_success = False

        if not archive_success:
            return EXIT_ARCHIVE_FAILED

        # Final summary
        logger.info("")
        logger.info("=" * 60)
        logger.info("Bundle Complete")
        logger.info("=" * 60)
        logger.info(f"Bundle Name: {self.bundle_name}")
        logger.info(f"Files Packaged: {len(self.files)}")
        logger.info(f"Output Directory: {self.output_dir}")
        logger.info("")
        logger.info("Generated Files:")
        if self.create_manifest:
            logger.info(f"  - {self.bundle_name}-manifest.json")
        if self.create_checksums:
            logger.info(f"  - {self.bundle_name}-checksums.sha256")
        for archive_path in self.archive_paths:
            logger.info(f"  - {archive_path.name}")
            if self.create_checksums:
                logger.info(f"  - {archive_path.name}.sha256")

        return EXIT_SUCCESS


def create_argument_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        prog="package_executive_bundle",
        description="Package T.A.R.S. governance reports into a single executive bundle",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exit Codes:
  0      Success, bundle created
  1      Run directory not found or invalid
  2      No files to package
  3      Archive creation failed
  199    General error

Examples:
  # Basic packaging (zip archive)
  python scripts/package_executive_bundle.py --run-dir ./reports/runs/tars-run-20251222-140000

  # Custom output directory and bundle name
  python scripts/package_executive_bundle.py --run-dir ./reports/runs/tars-run-20251222-140000 \\
      --output-dir ./release/executive \\
      --bundle-name executive-report-Q4

  # Include tar.gz archive
  python scripts/package_executive_bundle.py --run-dir ./reports/runs/tars-run-20251222-140000 --tar

  # Skip checksums
  python scripts/package_executive_bundle.py --run-dir ./reports/runs/tars-run-20251222-140000 --no-checksums
"""
    )

    # Required arguments
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Path to a completed orchestrator run directory"
    )

    # Output options
    parser.add_argument(
        "--output-dir",
        default="./release/executive",
        help="Output directory for the bundle (default: ./release/executive)"
    )

    parser.add_argument(
        "--bundle-name",
        default=None,
        help="Bundle name (default: tars-exec-bundle-<version>-<timestamp>)"
    )

    # Archive options
    parser.add_argument(
        "--zip",
        dest="create_zip",
        action="store_true",
        default=True,
        help="Create ZIP archive (default: true)"
    )

    parser.add_argument(
        "--no-zip",
        dest="create_zip",
        action="store_false",
        help="Do not create ZIP archive"
    )

    parser.add_argument(
        "--tar",
        dest="create_tar",
        action="store_true",
        default=False,
        help="Create tar.gz archive (default: false)"
    )

    # Checksum options
    parser.add_argument(
        "--checksums",
        dest="create_checksums",
        action="store_true",
        default=True,
        help="Create SHA-256 checksums (default: true)"
    )

    parser.add_argument(
        "--no-checksums",
        dest="create_checksums",
        action="store_false",
        help="Do not create checksums"
    )

    # Manifest options
    parser.add_argument(
        "--manifest",
        dest="create_manifest",
        action="store_true",
        default=True,
        help="Create manifest file (default: true)"
    )

    parser.add_argument(
        "--no-manifest",
        dest="create_manifest",
        action="store_false",
        help="Do not create manifest"
    )

    # Compliance index options (Phase 17)
    parser.add_argument(
        "--compliance-index",
        dest="create_compliance_index",
        action="store_true",
        default=True,
        help="Create compliance index markdown (default: true)"
    )

    parser.add_argument(
        "--no-compliance-index",
        dest="create_compliance_index",
        action="store_false",
        help="Do not create compliance index"
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
        packager = ExecutiveBundlePackager(
            run_dir=args.run_dir,
            output_dir=args.output_dir,
            bundle_name=args.bundle_name,
            create_zip=args.create_zip,
            create_tar=args.create_tar,
            create_checksums=args.create_checksums,
            create_manifest=args.create_manifest,
            create_compliance_index=args.create_compliance_index
        )

        return packager.package()

    except KeyboardInterrupt:
        logger.info("Packaging interrupted by user")
        return EXIT_GENERAL_ERROR
    except Exception as e:
        logger.error(f"Packaging error: {e}")
        return EXIT_GENERAL_ERROR


if __name__ == "__main__":
    sys.exit(main())
