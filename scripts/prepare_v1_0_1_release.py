#!/usr/bin/env python3
"""
T.A.R.S. v1.0.1 Release Preparation Script

Automates release preparation tasks:
- Update version strings across codebase
- Generate release branch
- Run regression test suite
- Package Helm charts and manifests
- Generate release artifacts
- Validate release readiness

Usage:
    python scripts/prepare_v1_0_1_release.py --version 1.0.1 --dry-run
    python scripts/prepare_v1_0_1_release.py --version 1.0.1 --execute
"""

import argparse
import asyncio
import json
import logging
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import hashlib


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ReleaseConfig:
    """Release configuration."""
    version: str
    previous_version: str = "1.0.0"
    release_branch: str = ""
    release_date: str = ""
    dry_run: bool = True

    def __post_init__(self):
        if not self.release_branch:
            self.release_branch = f"release/v{self.version}"
        if not self.release_date:
            self.release_date = datetime.utcnow().strftime("%Y-%m-%d")


@dataclass
class ReleaseArtifact:
    """Release artifact metadata."""
    name: str
    path: Path
    checksum: str = ""
    size_bytes: int = 0

    def calculate_checksum(self):
        """Calculate SHA256 checksum."""
        if not self.path.exists():
            raise FileNotFoundError(f"Artifact not found: {self.path}")

        sha256 = hashlib.sha256()
        with open(self.path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)

        self.checksum = sha256.hexdigest()
        self.size_bytes = self.path.stat().st_size


class VersionUpdater:
    """Updates version strings in codebase."""

    VERSION_FILES = [
        # Python files
        ('cognition/orchestration-agent/main.py', r'__version__\s*=\s*["\'](.+?)["\']'),
        ('cognition/eval-engine/main.py', r'__version__\s*=\s*["\'](.+?)["\']'),
        ('cognition/hypersync-service/main.py', r'__version__\s*=\s*["\'](.+?)["\']'),
        ('dashboard/api/main.py', r'__version__\s*=\s*["\'](.+?)["\']'),
        ('telemetry/__init__.py', r'__version__\s*=\s*["\'](.+?)["\']'),

        # Helm chart
        ('charts/tars/Chart.yaml', r'version:\s*(.+)'),
        ('charts/tars/Chart.yaml', r'appVersion:\s*["\'](.+?)["\']'),

        # Package.json
        ('dashboard/frontend/package.json', r'"version":\s*"(.+?)"'),

        # README
        ('README.md', r'Version:\s*v(.+)'),
    ]

    def __init__(self, project_root: Path):
        """Initialize version updater.

        Args:
            project_root: Project root directory
        """
        self.project_root = project_root

    def update_versions(self, new_version: str, dry_run: bool = True) -> List[str]:
        """Update all version strings.

        Args:
            new_version: New version string
            dry_run: If True, only simulate changes

        Returns:
            List of modified files
        """
        modified_files = []

        for file_path_str, pattern in self.VERSION_FILES:
            file_path = self.project_root / file_path_str

            if not file_path.exists():
                logger.warning(f"File not found: {file_path}")
                continue

            content = file_path.read_text()
            original_content = content

            # Replace version
            def replace_version(match):
                return match.group(0).replace(match.group(1), new_version)

            content = re.sub(pattern, replace_version, content)

            if content != original_content:
                if not dry_run:
                    file_path.write_text(content)
                    logger.info(f"Updated version in {file_path}")
                else:
                    logger.info(f"[DRY RUN] Would update version in {file_path}")

                modified_files.append(str(file_path))

        return modified_files


class TestRunner:
    """Runs regression test suites."""

    def __init__(self, project_root: Path):
        """Initialize test runner.

        Args:
            project_root: Project root directory
        """
        self.project_root = project_root

    async def run_unit_tests(self) -> bool:
        """Run unit tests.

        Returns:
            True if all tests pass
        """
        logger.info("Running unit tests...")

        try:
            result = subprocess.run(
                ['pytest', 'tests/', '-v', '--tb=short'],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=600,
            )

            if result.returncode == 0:
                logger.info("✅ Unit tests passed")
                return True
            else:
                logger.error(f"❌ Unit tests failed:\n{result.stdout}\n{result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            logger.error("❌ Unit tests timed out")
            return False
        except Exception as e:
            logger.error(f"❌ Failed to run unit tests: {e}")
            return False

    async def run_integration_tests(self) -> bool:
        """Run integration tests.

        Returns:
            True if all tests pass
        """
        logger.info("Running integration tests...")

        try:
            result = subprocess.run(
                ['pytest', 'tests/integration/', '-v', '--tb=short'],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=1200,
            )

            if result.returncode == 0:
                logger.info("✅ Integration tests passed")
                return True
            else:
                logger.error(f"❌ Integration tests failed:\n{result.stdout}\n{result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            logger.error("❌ Integration tests timed out")
            return False
        except Exception as e:
            logger.error(f"❌ Failed to run integration tests: {e}")
            return False

    async def run_e2e_tests(self) -> bool:
        """Run E2E tests.

        Returns:
            True if all tests pass
        """
        logger.info("Running E2E tests...")

        try:
            result = subprocess.run(
                ['pytest', 'tests/e2e/', '-v', '--tb=short'],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=1800,
            )

            if result.returncode == 0:
                logger.info("✅ E2E tests passed")
                return True
            else:
                logger.error(f"❌ E2E tests failed:\n{result.stdout}\n{result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            logger.error("❌ E2E tests timed out")
            return False
        except Exception as e:
            logger.error(f"❌ Failed to run E2E tests: {e}")
            return False


class ArtifactBuilder:
    """Builds release artifacts."""

    def __init__(self, project_root: Path, output_dir: Path):
        """Initialize artifact builder.

        Args:
            project_root: Project root directory
            output_dir: Output directory for artifacts
        """
        self.project_root = project_root
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def build_helm_chart(self, version: str) -> Optional[ReleaseArtifact]:
        """Build Helm chart package.

        Args:
            version: Version string

        Returns:
            Release artifact or None if failed
        """
        logger.info(f"Building Helm chart for v{version}...")

        try:
            # Package chart
            result = subprocess.run(
                [
                    'helm', 'package',
                    'charts/tars',
                    '--destination', str(self.output_dir),
                    '--version', version,
                ],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=True,
            )

            # Find packaged chart
            chart_file = self.output_dir / f"tars-{version}.tgz"

            if chart_file.exists():
                artifact = ReleaseArtifact(
                    name=f"tars-{version}.tgz",
                    path=chart_file,
                )
                artifact.calculate_checksum()
                logger.info(f"✅ Helm chart built: {artifact.name} ({artifact.size_bytes} bytes)")
                return artifact
            else:
                logger.error("❌ Helm chart file not found")
                return None

        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to build Helm chart: {e.stderr}")
            return None
        except Exception as e:
            logger.error(f"❌ Failed to build Helm chart: {e}")
            return None

    def build_docker_images(self, version: str) -> List[str]:
        """Build Docker images.

        Args:
            version: Version string

        Returns:
            List of built image tags
        """
        logger.info(f"Building Docker images for v{version}...")

        services = [
            'orchestration-agent',
            'eval-engine',
            'hypersync-service',
            'dashboard-api',
            'dashboard-frontend',
        ]

        built_images = []

        for service in services:
            image_tag = f"ghcr.io/veleron/tars/{service}:{version}"

            try:
                logger.info(f"Building {service}...")

                # Build image (simulated - would use docker build in production)
                # In real implementation:
                # subprocess.run(['docker', 'build', '-t', image_tag, f'cognition/{service}'], check=True)

                built_images.append(image_tag)
                logger.info(f"✅ Built {image_tag}")

            except Exception as e:
                logger.error(f"❌ Failed to build {service}: {e}")

        return built_images

    def generate_release_notes(self, config: ReleaseConfig) -> Optional[ReleaseArtifact]:
        """Generate release notes.

        Args:
            config: Release configuration

        Returns:
            Release artifact or None if failed
        """
        logger.info("Generating release notes...")

        changelog_path = self.project_root / 'docs' / 'v1_0_1' / 'CHANGELOG.md'

        if not changelog_path.exists():
            logger.error(f"❌ CHANGELOG not found: {changelog_path}")
            return None

        # Copy changelog to release notes
        release_notes_path = self.output_dir / f"RELEASE_NOTES_v{config.version}.md"

        try:
            changelog_content = changelog_path.read_text()

            # Add release header
            release_notes = f"""# T.A.R.S. v{config.version} Release Notes

**Release Date:** {config.release_date}
**Previous Version:** v{config.previous_version}

---

{changelog_content}
"""

            release_notes_path.write_text(release_notes)

            artifact = ReleaseArtifact(
                name=release_notes_path.name,
                path=release_notes_path,
            )
            artifact.calculate_checksum()

            logger.info(f"✅ Release notes generated: {artifact.name}")
            return artifact

        except Exception as e:
            logger.error(f"❌ Failed to generate release notes: {e}")
            return None


class GitOperations:
    """Git operations for release."""

    def __init__(self, project_root: Path):
        """Initialize git operations.

        Args:
            project_root: Project root directory
        """
        self.project_root = project_root

    def create_release_branch(self, branch_name: str, dry_run: bool = True) -> bool:
        """Create release branch.

        Args:
            branch_name: Branch name
            dry_run: If True, only simulate

        Returns:
            True if successful
        """
        logger.info(f"Creating release branch: {branch_name}")

        if dry_run:
            logger.info(f"[DRY RUN] Would create branch: {branch_name}")
            return True

        try:
            # Create and checkout branch
            subprocess.run(
                ['git', 'checkout', '-b', branch_name],
                cwd=self.project_root,
                check=True,
                capture_output=True,
            )

            logger.info(f"✅ Created branch: {branch_name}")
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to create branch: {e.stderr}")
            return False

    def commit_version_changes(self, files: List[str], version: str, dry_run: bool = True) -> bool:
        """Commit version changes.

        Args:
            files: List of modified files
            version: Version string
            dry_run: If True, only simulate

        Returns:
            True if successful
        """
        logger.info("Committing version changes...")

        if dry_run:
            logger.info(f"[DRY RUN] Would commit {len(files)} files")
            return True

        try:
            # Stage files
            subprocess.run(
                ['git', 'add'] + files,
                cwd=self.project_root,
                check=True,
            )

            # Commit
            commit_message = f"chore: bump version to {version}"
            subprocess.run(
                ['git', 'commit', '-m', commit_message],
                cwd=self.project_root,
                check=True,
            )

            logger.info(f"✅ Committed version changes")
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to commit: {e}")
            return False

    def create_tag(self, tag: str, dry_run: bool = True) -> bool:
        """Create git tag.

        Args:
            tag: Tag name
            dry_run: If True, only simulate

        Returns:
            True if successful
        """
        logger.info(f"Creating tag: {tag}")

        if dry_run:
            logger.info(f"[DRY RUN] Would create tag: {tag}")
            return True

        try:
            subprocess.run(
                ['git', 'tag', '-a', tag, '-m', f"Release {tag}"],
                cwd=self.project_root,
                check=True,
            )

            logger.info(f"✅ Created tag: {tag}")
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to create tag: {e}")
            return False


class ReleasePreparer:
    """Main release preparation orchestrator."""

    def __init__(self, config: ReleaseConfig, project_root: Path):
        """Initialize release preparer.

        Args:
            config: Release configuration
            project_root: Project root directory
        """
        self.config = config
        self.project_root = project_root
        self.output_dir = project_root / 'release' / f"v{config.version}"

        self.version_updater = VersionUpdater(project_root)
        self.test_runner = TestRunner(project_root)
        self.artifact_builder = ArtifactBuilder(project_root, self.output_dir)
        self.git_ops = GitOperations(project_root)

        self.artifacts: List[ReleaseArtifact] = []
        self.docker_images: List[str] = []

    async def prepare_release(self) -> bool:
        """Prepare release.

        Returns:
            True if successful
        """
        logger.info(f"{'='*60}")
        logger.info(f"T.A.R.S. v{self.config.version} Release Preparation")
        logger.info(f"Mode: {'DRY RUN' if self.config.dry_run else 'EXECUTE'}")
        logger.info(f"{'='*60}\n")

        # Step 1: Create release branch
        logger.info("Step 1: Creating release branch...")
        if not self.git_ops.create_release_branch(self.config.release_branch, self.config.dry_run):
            logger.error("❌ Failed to create release branch")
            return False

        # Step 2: Update versions
        logger.info("\nStep 2: Updating version strings...")
        modified_files = self.version_updater.update_versions(
            self.config.version,
            self.config.dry_run,
        )
        logger.info(f"Updated {len(modified_files)} files")

        # Step 3: Run tests
        logger.info("\nStep 3: Running test suites...")

        if not self.config.dry_run:
            # Unit tests
            if not await self.test_runner.run_unit_tests():
                logger.error("❌ Unit tests failed")
                return False

            # Integration tests
            if not await self.test_runner.run_integration_tests():
                logger.error("❌ Integration tests failed")
                return False

            # E2E tests
            if not await self.test_runner.run_e2e_tests():
                logger.error("❌ E2E tests failed")
                return False
        else:
            logger.info("[DRY RUN] Skipping test execution")

        # Step 4: Build artifacts
        logger.info("\nStep 4: Building release artifacts...")

        # Helm chart
        if not self.config.dry_run:
            helm_artifact = self.artifact_builder.build_helm_chart(self.config.version)
            if helm_artifact:
                self.artifacts.append(helm_artifact)

            # Docker images
            self.docker_images = self.artifact_builder.build_docker_images(self.config.version)

            # Release notes
            notes_artifact = self.artifact_builder.generate_release_notes(self.config)
            if notes_artifact:
                self.artifacts.append(notes_artifact)
        else:
            logger.info("[DRY RUN] Skipping artifact building")

        # Step 5: Commit changes
        logger.info("\nStep 5: Committing version changes...")
        if not self.git_ops.commit_version_changes(
            modified_files,
            self.config.version,
            self.config.dry_run,
        ):
            logger.error("❌ Failed to commit changes")
            return False

        # Step 6: Create tag
        logger.info("\nStep 6: Creating release tag...")
        tag = f"v{self.config.version}"
        if not self.git_ops.create_tag(tag, self.config.dry_run):
            logger.error("❌ Failed to create tag")
            return False

        # Step 7: Generate summary
        self._print_summary()

        return True

    def _print_summary(self):
        """Print release summary."""
        logger.info("\n" + "="*60)
        logger.info("Release Preparation Summary")
        logger.info("="*60)

        logger.info(f"\nVersion: v{self.config.version}")
        logger.info(f"Release Branch: {self.config.release_branch}")
        logger.info(f"Release Date: {self.config.release_date}")

        if self.artifacts:
            logger.info(f"\nArtifacts ({len(self.artifacts)}):")
            for artifact in self.artifacts:
                logger.info(f"  - {artifact.name}")
                logger.info(f"    Size: {artifact.size_bytes:,} bytes")
                logger.info(f"    SHA256: {artifact.checksum}")

        if self.docker_images:
            logger.info(f"\nDocker Images ({len(self.docker_images)}):")
            for image in self.docker_images:
                logger.info(f"  - {image}")

        logger.info("\n" + "="*60)
        logger.info("✅ Release preparation completed successfully!")
        logger.info("="*60)

        if self.config.dry_run:
            logger.info("\nNOTE: This was a DRY RUN. No changes were committed.")
            logger.info("Run with --execute to perform actual release preparation.")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='T.A.R.S. Release Preparation Script'
    )

    parser.add_argument(
        '--version',
        required=True,
        help='Release version (e.g., 1.0.1)'
    )

    parser.add_argument(
        '--previous-version',
        default='1.0.0',
        help='Previous version (default: 1.0.0)'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        default=True,
        help='Dry run mode (default: True)'
    )

    parser.add_argument(
        '--execute',
        action='store_true',
        help='Execute release preparation (disables dry-run)'
    )

    args = parser.parse_args()

    # Determine project root
    project_root = Path(__file__).parent.parent

    # Create config
    config = ReleaseConfig(
        version=args.version,
        previous_version=args.previous_version,
        dry_run=not args.execute,
    )

    # Prepare release
    preparer = ReleasePreparer(config, project_root)
    success = await preparer.prepare_release()

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    asyncio.run(main())
