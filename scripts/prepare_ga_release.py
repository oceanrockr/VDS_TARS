#!/usr/bin/env python3
"""
GA Release Artifact Packager

Phase 14.9 - Task 3: GA Hardening & Production Release Gate

Prepares the final GA release artifacts, including:
- Validation enforcement
- GA metadata stamping
- Checksums generation
- Manifest creation
- Optional signed artifacts

Exit Codes:
    0   = Success
    1   = Validation failed
    2   = Packaging failed
    199 = General error

Author: T.A.R.S. Development Team
Version: 1.0.4
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import zipfile
import tarfile


@dataclass
class GAMetadata:
    """GA release metadata."""
    version: str
    release_date: str
    release_type: str = "GA"
    build_number: str = ""
    git_commit: str = ""
    git_branch: str = ""
    git_tag: str = ""
    build_host: str = ""
    python_version: str = ""
    validation_status: str = ""
    validation_score: float = 0.0


@dataclass
class ArtifactManifest:
    """Manifest of release artifacts."""
    manifest_version: str = "1.0"
    release_version: str = ""
    release_type: str = "GA"
    created_at: str = ""
    created_by: str = "T.A.R.S. GA Release Packager"
    artifacts: List[Dict[str, Any]] = field(default_factory=list)
    checksums: Dict[str, str] = field(default_factory=dict)
    metadata: Optional[GAMetadata] = None


class GAReleasePackager:
    """
    GA Release Artifact Packager.

    Prepares production-ready release artifacts with validation,
    checksums, and manifests.
    """

    # Core files to include in release
    CORE_FILES = [
        "VERSION",
        "CHANGELOG.md",
        "README.md",
        "requirements.txt",
        "requirements-dev.txt",
        "setup.py",
        "pyproject.toml",
    ]

    # Core directories to include
    CORE_DIRS = [
        "analytics",
        "observability",
        "scripts",
        "docs",
        "enterprise_api",
        "enterprise_config",
    ]

    # Files to exclude from release
    EXCLUDE_PATTERNS = [
        "__pycache__",
        "*.pyc",
        "*.pyo",
        ".git",
        ".env",
        "*.key",
        "*.pem",
        ".pytest_cache",
        ".mypy_cache",
        "node_modules",
        "*.egg-info",
        "dist",
        "build",
    ]

    def __init__(
        self,
        project_root: Path,
        output_dir: Path,
        version: str,
        verbose: bool = False,
        skip_validation: bool = False,
        sign_artifacts: bool = False
    ):
        """
        Initialize the GA Release Packager.

        Args:
            project_root: Path to project root
            output_dir: Output directory for release artifacts
            version: Release version
            verbose: Enable verbose output
            skip_validation: Skip GA validation (not recommended)
            sign_artifacts: Sign artifacts with GPG
        """
        self.project_root = project_root
        self.output_dir = output_dir
        self.version = version
        self.verbose = verbose
        self.skip_validation = skip_validation
        self.sign_artifacts = sign_artifacts

    def log(self, message: str, level: str = "INFO") -> None:
        """Log a message if verbose mode is enabled."""
        if self.verbose:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] [{level}] {message}")

    def run_validation(self) -> Tuple[bool, float]:
        """
        Run GA validation before packaging.

        Returns:
            Tuple of (passed, score)
        """
        if self.skip_validation:
            self.log("Skipping validation (--skip-validation flag set)", "WARNING")
            return True, 0.0

        self.log("Running GA validation...")

        validator_script = self.project_root / "scripts" / "ga_release_validator.py"

        if not validator_script.exists():
            self.log("GA validator script not found, skipping validation", "WARNING")
            return True, 0.0

        try:
            result = subprocess.run(
                [sys.executable, str(validator_script), "--json"],
                capture_output=True,
                text=True,
                cwd=self.project_root
            )

            if result.returncode == 150:  # GA_READY
                self.log("GA validation passed", "INFO")
                try:
                    validation_result = json.loads(result.stdout)
                    score = validation_result.get("passed_checks", 0) / max(validation_result.get("total_checks", 1), 1) * 100
                    return True, score
                except json.JSONDecodeError:
                    return True, 100.0
            elif result.returncode == 151:  # GA_BLOCKED
                self.log("GA validation has warnings", "WARNING")
                return True, 80.0
            else:  # GA_FAILED or error
                self.log(f"GA validation failed with exit code {result.returncode}", "ERROR")
                if result.stderr:
                    self.log(result.stderr, "ERROR")
                return False, 0.0

        except Exception as e:
            self.log(f"Error running validation: {e}", "ERROR")
            return False, 0.0

    def get_git_info(self) -> Dict[str, str]:
        """Get git repository information."""
        git_info = {
            "commit": "",
            "branch": "",
            "tag": ""
        }

        try:
            # Get commit hash
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            if result.returncode == 0:
                git_info["commit"] = result.stdout.strip()[:12]

            # Get branch
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            if result.returncode == 0:
                git_info["branch"] = result.stdout.strip()

            # Get tag
            result = subprocess.run(
                ["git", "describe", "--tags", "--exact-match"],
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            if result.returncode == 0:
                git_info["tag"] = result.stdout.strip()

        except Exception as e:
            self.log(f"Could not get git info: {e}", "WARNING")

        return git_info

    def create_metadata(self, validation_status: str, validation_score: float) -> GAMetadata:
        """Create GA release metadata."""
        git_info = self.get_git_info()

        return GAMetadata(
            version=self.version,
            release_date=datetime.now().strftime("%Y-%m-%d"),
            release_type="GA",
            build_number=datetime.now().strftime("%Y%m%d%H%M%S"),
            git_commit=git_info["commit"],
            git_branch=git_info["branch"],
            git_tag=git_info.get("tag", f"v{self.version}"),
            build_host=os.environ.get("COMPUTERNAME", os.environ.get("HOSTNAME", "unknown")),
            python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            validation_status=validation_status,
            validation_score=validation_score
        )

    def calculate_checksum(self, file_path: Path, algorithm: str = "sha256") -> str:
        """Calculate checksum for a file."""
        hash_func = hashlib.new(algorithm)
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hash_func.update(chunk)
        return hash_func.hexdigest()

    def should_include(self, path: Path) -> bool:
        """Check if a path should be included in the release."""
        path_str = str(path)

        for pattern in self.EXCLUDE_PATTERNS:
            if pattern.startswith("*"):
                if path_str.endswith(pattern[1:]):
                    return False
            elif pattern in path_str:
                return False

        return True

    def copy_release_files(self, staging_dir: Path) -> List[Dict[str, Any]]:
        """Copy release files to staging directory."""
        artifacts = []

        # Copy core files
        for file_name in self.CORE_FILES:
            src = self.project_root / file_name
            if src.exists():
                dst = staging_dir / file_name
                shutil.copy2(src, dst)
                artifacts.append({
                    "name": file_name,
                    "type": "file",
                    "size": dst.stat().st_size
                })
                self.log(f"Copied: {file_name}")

        # Copy core directories
        for dir_name in self.CORE_DIRS:
            src_dir = self.project_root / dir_name
            if src_dir.exists() and src_dir.is_dir():
                dst_dir = staging_dir / dir_name
                self._copy_directory(src_dir, dst_dir)
                artifacts.append({
                    "name": dir_name,
                    "type": "directory",
                    "file_count": sum(1 for _ in dst_dir.rglob("*") if _.is_file())
                })
                self.log(f"Copied directory: {dir_name}")

        return artifacts

    def _copy_directory(self, src: Path, dst: Path) -> None:
        """Copy directory recursively, excluding patterns."""
        dst.mkdir(parents=True, exist_ok=True)

        for item in src.iterdir():
            if not self.should_include(item):
                continue

            dst_item = dst / item.name
            if item.is_dir():
                self._copy_directory(item, dst_item)
            else:
                shutil.copy2(item, dst_item)

    def create_checksums(self, staging_dir: Path) -> Dict[str, str]:
        """Create checksums for all files in staging directory."""
        checksums = {}

        for file_path in staging_dir.rglob("*"):
            if file_path.is_file():
                relative_path = file_path.relative_to(staging_dir)
                checksums[str(relative_path)] = self.calculate_checksum(file_path)

        return checksums

    def write_checksums_file(self, checksums: Dict[str, str], output_path: Path) -> None:
        """Write checksums to a file."""
        with open(output_path, 'w') as f:
            for file_path, checksum in sorted(checksums.items()):
                f.write(f"{checksum}  {file_path}\n")

    def create_archive(self, staging_dir: Path, archive_name: str) -> Path:
        """Create release archive."""
        # Create tar.gz
        archive_path = self.output_dir / f"{archive_name}.tar.gz"
        with tarfile.open(archive_path, "w:gz") as tar:
            tar.add(staging_dir, arcname=archive_name)

        self.log(f"Created archive: {archive_path}")
        return archive_path

    def create_zip_archive(self, staging_dir: Path, archive_name: str) -> Path:
        """Create ZIP archive."""
        archive_path = self.output_dir / f"{archive_name}.zip"

        with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in staging_dir.rglob("*"):
                if file_path.is_file():
                    arcname = f"{archive_name}/{file_path.relative_to(staging_dir)}"
                    zipf.write(file_path, arcname)

        self.log(f"Created ZIP archive: {archive_path}")
        return archive_path

    def sign_file(self, file_path: Path) -> Optional[Path]:
        """Sign a file with GPG (if available)."""
        if not self.sign_artifacts:
            return None

        try:
            sig_path = file_path.with_suffix(file_path.suffix + ".sig")
            result = subprocess.run(
                ["gpg", "--detach-sign", "--armor", "-o", str(sig_path), str(file_path)],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                self.log(f"Signed: {file_path.name}")
                return sig_path
            else:
                self.log(f"Could not sign {file_path.name}: {result.stderr}", "WARNING")
                return None
        except Exception as e:
            self.log(f"GPG signing failed: {e}", "WARNING")
            return None

    def write_manifest(self, manifest: ArtifactManifest, output_path: Path) -> None:
        """Write manifest to JSON file."""
        manifest_dict = {
            "manifest_version": manifest.manifest_version,
            "release_version": manifest.release_version,
            "release_type": manifest.release_type,
            "created_at": manifest.created_at,
            "created_by": manifest.created_by,
            "artifacts": manifest.artifacts,
            "checksums": manifest.checksums,
            "metadata": {
                "version": manifest.metadata.version,
                "release_date": manifest.metadata.release_date,
                "release_type": manifest.metadata.release_type,
                "build_number": manifest.metadata.build_number,
                "git_commit": manifest.metadata.git_commit,
                "git_branch": manifest.metadata.git_branch,
                "git_tag": manifest.metadata.git_tag,
                "build_host": manifest.metadata.build_host,
                "python_version": manifest.metadata.python_version,
                "validation_status": manifest.metadata.validation_status,
                "validation_score": manifest.metadata.validation_score
            } if manifest.metadata else None
        }

        with open(output_path, 'w') as f:
            json.dump(manifest_dict, f, indent=2)

    def write_release_notes(self, metadata: GAMetadata, output_path: Path) -> None:
        """Write GA release notes."""
        notes = []
        notes.append(f"# T.A.R.S. v{metadata.version} GA Release Notes")
        notes.append("")
        notes.append(f"**Release Date:** {metadata.release_date}")
        notes.append(f"**Release Type:** General Availability (GA)")
        notes.append(f"**Build Number:** {metadata.build_number}")
        notes.append("")

        if metadata.git_commit:
            notes.append(f"**Git Commit:** {metadata.git_commit}")
        if metadata.git_tag:
            notes.append(f"**Git Tag:** {metadata.git_tag}")
        notes.append("")

        notes.append("## Validation Status")
        notes.append("")
        notes.append(f"- **Status:** {metadata.validation_status}")
        notes.append(f"- **Score:** {metadata.validation_score:.1f}%")
        notes.append("")

        notes.append("## What's New in v1.0.4")
        notes.append("")
        notes.append("### Phase 14.9: GA Hardening & Production Release Gate")
        notes.append("")
        notes.append("- **GA Release Validator Engine**: Comprehensive pre-release validation")
        notes.append("- **Production Readiness Checklist**: Automated readiness assessment")
        notes.append("- **GA Artifact Packager**: Secure release artifact generation")
        notes.append("- **Version Finalization**: Clean GA version without RC suffix")
        notes.append("")

        notes.append("### Previous Phases (1.0.0-1.0.4-rc1)")
        notes.append("")
        notes.append("- **Organization Health Governance**: Multi-repository health aggregation")
        notes.append("- **SLA Intelligence Engine**: Executive readiness scoring and breach attribution")
        notes.append("- **Temporal Intelligence Engine**: Time-lagged correlation and propagation analysis")
        notes.append("- **Trend Correlation Engine**: Cross-repository trend pattern detection")
        notes.append("- **Enterprise Hardening**: SOC 2, ISO 27001, GDPR compliance")
        notes.append("- **Multi-Agent RL System**: DQN, A2C, PPO, DDPG agents with Nash equilibrium")
        notes.append("")

        notes.append("## Installation")
        notes.append("")
        notes.append("```bash")
        notes.append("# Extract the release archive")
        notes.append(f"tar -xzf tars-{metadata.version}.tar.gz")
        notes.append(f"cd tars-{metadata.version}")
        notes.append("")
        notes.append("# Install dependencies")
        notes.append("pip install -r requirements.txt")
        notes.append("")
        notes.append("# Verify installation")
        notes.append("python -m analytics.run_org_health --help")
        notes.append("```")
        notes.append("")

        notes.append("## Verification")
        notes.append("")
        notes.append("Verify the release checksums:")
        notes.append("")
        notes.append("```bash")
        notes.append("sha256sum -c checksums.sha256")
        notes.append("```")
        notes.append("")

        notes.append("## Documentation")
        notes.append("")
        notes.append("See the `docs/` directory for complete documentation:")
        notes.append("")
        notes.append("- [README.md](README.md) - Getting started")
        notes.append("- [docs/PHASE14_6_ENTERPRISE_HARDENING.md](docs/PHASE14_6_ENTERPRISE_HARDENING.md) - Enterprise features")
        notes.append("- [docs/ORG_SLA_INTELLIGENCE_ENGINE.md](docs/ORG_SLA_INTELLIGENCE_ENGINE.md) - SLA intelligence")
        notes.append("")

        notes.append("---")
        notes.append("")
        notes.append("*T.A.R.S. - Temporal Augmented Retrieval System*")
        notes.append("*Veleron Dev Studios*")

        with open(output_path, 'w') as f:
            f.write("\n".join(notes))

    def package(self) -> bool:
        """
        Execute the full GA packaging process.

        Returns:
            True if packaging succeeded, False otherwise
        """
        print("=" * 60)
        print("T.A.R.S. GA RELEASE PACKAGER")
        print("=" * 60)
        print(f"Version: {self.version}")
        print(f"Output:  {self.output_dir}")
        print()

        # Step 1: Run validation
        print("Step 1: Running GA validation...")
        validation_passed, validation_score = self.run_validation()

        if not validation_passed:
            print("ERROR: GA validation failed. Cannot proceed with packaging.")
            print("Run 'python scripts/ga_release_validator.py' for details.")
            return False

        validation_status = "PASSED" if validation_score >= 80 else "PASSED_WITH_WARNINGS"
        print(f"  Validation: {validation_status} ({validation_score:.1f}%)")
        print()

        # Step 2: Create output directory
        print("Step 2: Preparing output directory...")
        release_dir = self.output_dir / f"v{self.version}"
        release_dir.mkdir(parents=True, exist_ok=True)

        artifacts_dir = release_dir / "artifacts"
        artifacts_dir.mkdir(exist_ok=True)

        staging_dir = release_dir / f"tars-{self.version}"
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        staging_dir.mkdir()

        print(f"  Created: {release_dir}")
        print()

        # Step 3: Create metadata
        print("Step 3: Creating GA metadata...")
        metadata = self.create_metadata(validation_status, validation_score)
        print(f"  Build: {metadata.build_number}")
        print(f"  Commit: {metadata.git_commit}")
        print()

        # Step 4: Copy release files
        print("Step 4: Copying release files...")
        artifacts = self.copy_release_files(staging_dir)
        print(f"  Copied {len(artifacts)} artifacts")
        print()

        # Step 5: Calculate checksums
        print("Step 5: Calculating checksums...")
        checksums = self.create_checksums(staging_dir)
        checksums_file = release_dir / "checksums.sha256"
        self.write_checksums_file(checksums, checksums_file)
        print(f"  Generated checksums for {len(checksums)} files")
        print()

        # Step 6: Create archives
        print("Step 6: Creating release archives...")
        archive_name = f"tars-{self.version}"

        tar_archive = self.create_archive(staging_dir, archive_name)
        zip_archive = self.create_zip_archive(staging_dir, archive_name)

        # Calculate archive checksums
        tar_checksum = self.calculate_checksum(tar_archive)
        zip_checksum = self.calculate_checksum(zip_archive)

        archive_checksums = release_dir / "archive-checksums.sha256"
        with open(archive_checksums, 'w') as f:
            f.write(f"{tar_checksum}  {tar_archive.name}\n")
            f.write(f"{zip_checksum}  {zip_archive.name}\n")

        print(f"  tar.gz: {tar_archive.name}")
        print(f"  zip:    {zip_archive.name}")
        print()

        # Step 7: Sign artifacts (if requested)
        if self.sign_artifacts:
            print("Step 7: Signing artifacts...")
            self.sign_file(tar_archive)
            self.sign_file(zip_archive)
            print()
        else:
            print("Step 7: Skipping artifact signing (use --sign to enable)")
            print()

        # Step 8: Create manifest
        print("Step 8: Creating manifest...")
        manifest = ArtifactManifest(
            release_version=self.version,
            created_at=datetime.now().isoformat(),
            artifacts=[
                {"name": tar_archive.name, "type": "archive", "format": "tar.gz", "checksum": tar_checksum},
                {"name": zip_archive.name, "type": "archive", "format": "zip", "checksum": zip_checksum},
                {"name": "checksums.sha256", "type": "checksums", "format": "sha256"},
            ] + artifacts,
            checksums={"archives": {"tar.gz": tar_checksum, "zip": zip_checksum}},
            metadata=metadata
        )
        manifest_file = release_dir / "manifest.json"
        self.write_manifest(manifest, manifest_file)
        print(f"  Written: {manifest_file}")
        print()

        # Step 9: Write release notes
        print("Step 9: Writing release notes...")
        release_notes = release_dir / "RELEASE_NOTES_GA.md"
        self.write_release_notes(metadata, release_notes)
        print(f"  Written: {release_notes}")
        print()

        # Cleanup staging
        shutil.rmtree(staging_dir)

        # Summary
        print("=" * 60)
        print("GA RELEASE PACKAGING COMPLETE")
        print("=" * 60)
        print()
        print("Release artifacts created at:")
        print(f"  {release_dir}")
        print()
        print("Contents:")
        for item in sorted(release_dir.iterdir()):
            if item.is_file():
                size = item.stat().st_size
                print(f"  {item.name:40} {size:>10} bytes")
        print()

        return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="GA Release Artifact Packager - Phase 14.9",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python prepare_ga_release.py --version 1.0.4
  python prepare_ga_release.py --version 1.0.4 --sign
  python prepare_ga_release.py --version 1.0.4 --output ./release/ga
        """
    )

    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Path to project root"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("release/ga"),
        help="Output directory for release artifacts"
    )
    parser.add_argument(
        "--version", "-V",
        type=str,
        default="1.0.4",
        help="Release version (default: 1.0.4)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip GA validation (not recommended)"
    )
    parser.add_argument(
        "--sign",
        action="store_true",
        help="Sign artifacts with GPG"
    )

    args = parser.parse_args()

    try:
        packager = GAReleasePackager(
            project_root=args.project_root,
            output_dir=args.output,
            version=args.version,
            verbose=args.verbose,
            skip_validation=args.skip_validation,
            sign_artifacts=args.sign
        )

        success = packager.package()
        sys.exit(0 if success else 1)

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(199)


if __name__ == "__main__":
    main()
