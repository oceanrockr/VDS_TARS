#!/usr/bin/env python3
"""
scripts/prepare_release_artifacts.py

Automatically generate all release artifacts for T.A.R.S. v1.0.2-rc1.

This script:
- Collects all required documentation and code artifacts
- Validates file existence and integrity
- Optionally signs and encrypts artifacts (enterprise mode)
- Generates SBOM and SLSA provenance (optional)
- Creates a release manifest with SHA256 hashes
- Packages everything into release/ directory

Compatible with Phase 14.6 (v1.0.2-RC1)

Exit Codes:
  0 - Success
  1 - Required artifact missing
  2 - Enterprise modules missing when required
  3 - Unexpected error
"""

import argparse
import hashlib
import json
import logging
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ============================================================================
# CONFIGURATION
# ============================================================================

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent

# Required artifacts (relative to project root)
REQUIRED_ARTIFACTS = [
    "VERSION",
    "README.md",
    "CHANGELOG.md",
    "RELEASE_NOTES_v1.0.2-RC1.md",
    "docs/PHASE14_6_ENTERPRISE_HARDENING.md",
    "docs/PHASE14_6_API_GUIDE.md",
    "scripts/test_phase9_end_to_end.py",
    "examples/api_client.py",
    "examples/compliance_check.sh",
]

# Optional artifacts (won't fail if missing)
OPTIONAL_ARTIFACTS = [
    "docs/PHASE14_6_PRODUCTION_RUNBOOK.md",
    "docs/PHASE14_6_DOCKER.md",
    "docs/PHASE14_6_QUICKSTART.md",
    "requirements-dev.txt",
    "pyproject.toml",
    "pytest.ini",
]

# Optional glob patterns for observability data
OBSERVABILITY_PATTERNS = [
    "observability/**/*.json",
    "observability/**/*.md",
    "observability/**/*.enc",
    "observability/**/*.sig",
]

# ============================================================================
# LOGGING SETUP
# ============================================================================

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    """Configure logging based on verbosity level."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(levelname)s: %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def compute_sha256(file_path: Path) -> str:
    """
    Compute SHA256 hash of a file.

    Args:
        file_path: Path to file

    Returns:
        Hexadecimal SHA256 hash
    """
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest()


def read_version(version_file: Path) -> str:
    """
    Read version from VERSION file.

    Args:
        version_file: Path to VERSION file

    Returns:
        Version string

    Raises:
        FileNotFoundError: If VERSION file doesn't exist
        ValueError: If VERSION file is empty or invalid
    """
    if not version_file.exists():
        raise FileNotFoundError(f"VERSION file not found: {version_file}")

    version = version_file.read_text().strip()

    if not version:
        raise ValueError("VERSION file is empty")

    logger.info(f"Version: {version}")
    return version


def validate_artifact(artifact_path: Path) -> bool:
    """
    Validate that an artifact exists and is readable.

    Args:
        artifact_path: Path to artifact

    Returns:
        True if valid, False otherwise
    """
    if not artifact_path.exists():
        logger.error(f"Missing artifact: {artifact_path}")
        return False

    if not artifact_path.is_file():
        logger.error(f"Artifact is not a file: {artifact_path}")
        return False

    if artifact_path.stat().st_size == 0:
        logger.warning(f"Artifact is empty: {artifact_path}")

    return True


# ============================================================================
# ENTERPRISE INTEGRATION
# ============================================================================

def import_enterprise_modules() -> Tuple[bool, Optional[object], Optional[object]]:
    """
    Attempt to import enterprise modules.

    Returns:
        Tuple of (success, SecurityManager class, load_config function)
    """
    try:
        sys.path.insert(0, str(PROJECT_ROOT))
        from enterprise_config.config_loader import load_config
        from security.security_manager import SecurityManager

        logger.debug("Enterprise modules imported successfully")
        return True, SecurityManager, load_config

    except ImportError as e:
        logger.debug(f"Enterprise modules not available: {e}")
        return False, None, None


def sign_artifacts(
    artifacts: List[Path],
    security_manager: object,
    output_dir: Path,
    dry_run: bool
) -> List[Path]:
    """
    Sign artifacts using RSA-PSS.

    Args:
        artifacts: List of artifact paths
        security_manager: SecurityManager instance
        output_dir: Output directory
        dry_run: If True, don't write files

    Returns:
        List of signature file paths
    """
    signature_files = []

    for artifact in artifacts:
        sig_file = output_dir / f"{artifact.name}.sig"

        if dry_run:
            logger.info(f"[DRY RUN] Would sign: {artifact.name} → {sig_file.name}")
            signature_files.append(sig_file)
            continue

        logger.info(f"Signing: {artifact.name}")

        with open(artifact, 'rb') as f:
            data = f.read()

        signature = security_manager.sign(data)

        with open(sig_file, 'wb') as f:
            f.write(signature)

        logger.debug(f"Created signature: {sig_file}")
        signature_files.append(sig_file)

    return signature_files


def encrypt_artifacts(
    artifacts: List[Path],
    security_manager: object,
    output_dir: Path,
    dry_run: bool
) -> List[Path]:
    """
    Encrypt artifacts using AES-256-GCM.

    Args:
        artifacts: List of artifact paths
        security_manager: SecurityManager instance
        output_dir: Output directory
        dry_run: If True, don't write files

    Returns:
        List of encrypted file paths
    """
    encrypted_files = []

    for artifact in artifacts:
        # Only encrypt JSON and MD files
        if artifact.suffix not in ['.json', '.md']:
            continue

        enc_file = output_dir / f"{artifact.name}.enc"

        if dry_run:
            logger.info(f"[DRY RUN] Would encrypt: {artifact.name} → {enc_file.name}")
            encrypted_files.append(enc_file)
            continue

        logger.info(f"Encrypting: {artifact.name}")

        with open(artifact, 'rb') as f:
            data = f.read()

        encrypted = security_manager.encrypt(data)

        with open(enc_file, 'wb') as f:
            f.write(encrypted)

        logger.debug(f"Created encrypted file: {enc_file}")
        encrypted_files.append(enc_file)

    return encrypted_files


# ============================================================================
# ARTIFACT COLLECTION
# ============================================================================

def collect_artifacts(
    project_root: Path,
    output_dir: Path,
    dry_run: bool
) -> List[Path]:
    """
    Collect all required and optional artifacts.

    Args:
        project_root: Project root directory
        output_dir: Output directory
        dry_run: If True, don't copy files

    Returns:
        List of collected artifact paths

    Raises:
        FileNotFoundError: If required artifact is missing
    """
    collected = []

    # Collect required artifacts
    logger.info("Collecting required artifacts...")
    for artifact_rel in REQUIRED_ARTIFACTS:
        artifact_path = project_root / artifact_rel

        if not validate_artifact(artifact_path):
            raise FileNotFoundError(f"Required artifact missing: {artifact_rel}")

        dest_path = output_dir / artifact_path.name

        if dry_run:
            logger.info(f"[DRY RUN] Would copy: {artifact_rel} -> {dest_path.name}")
        else:
            shutil.copy2(artifact_path, dest_path)
            logger.debug(f"Copied: {artifact_rel}")

        collected.append(dest_path if not dry_run else artifact_path)

    # Collect optional artifacts
    logger.info("Collecting optional artifacts...")
    for artifact_rel in OPTIONAL_ARTIFACTS:
        artifact_path = project_root / artifact_rel

        if not artifact_path.exists():
            logger.debug(f"Optional artifact not found (skipping): {artifact_rel}")
            continue

        dest_path = output_dir / artifact_path.name

        if dry_run:
            logger.info(f"[DRY RUN] Would copy: {artifact_rel} -> {dest_path.name}")
        else:
            shutil.copy2(artifact_path, dest_path)
            logger.debug(f"Copied: {artifact_rel}")

        collected.append(dest_path if not dry_run else artifact_path)

    # Collect observability data (if exists)
    logger.info("Collecting observability data...")
    observability_dir = project_root / "observability"

    if observability_dir.exists():
        for pattern in OBSERVABILITY_PATTERNS:
            for file_path in project_root.glob(pattern):
                if file_path.is_file():
                    dest_path = output_dir / file_path.name

                    if dry_run:
                        logger.debug(f"[DRY RUN] Would copy: {file_path.relative_to(project_root)}")
                    else:
                        shutil.copy2(file_path, dest_path)
                        logger.debug(f"Copied observability data: {file_path.name}")

                    collected.append(dest_path if not dry_run else file_path)

    logger.info(f"Collected {len(collected)} artifacts")
    return collected


# ============================================================================
# MANIFEST GENERATION
# ============================================================================

def generate_manifest(
    version: str,
    profile: str,
    artifacts: List[Path],
    output_dir: Path,
    signed: bool,
    encrypted: bool,
    sbom: bool,
    slsa: bool,
    perf_tests: bool,
    security_audit: bool,
    dry_run: bool
) -> Path:
    """
    Generate release manifest with artifact metadata.

    Args:
        version: Release version
        profile: Configuration profile
        artifacts: List of artifact paths
        output_dir: Output directory
        signed: Whether artifacts were signed
        encrypted: Whether artifacts were encrypted
        sbom: Whether SBOM was generated
        slsa: Whether SLSA provenance was generated
        perf_tests: Whether performance tests were run
        security_audit: Whether security audit was run
        dry_run: If True, don't write file

    Returns:
        Path to manifest file
    """
    logger.info("Generating release manifest...")

    manifest = {
        "version": version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "profile": profile,
        "artifacts": [],
        "enterprise": {
            "signed": signed,
            "encrypted": encrypted,
            "sbom": sbom,
            "slsa": slsa,
        },
        "validation": {
            "performance_tests": perf_tests,
            "security_audit": security_audit,
        }
    }

    # Add artifact metadata
    for artifact in sorted(artifacts):
        if artifact.exists():
            artifact_meta = {
                "name": artifact.name,
                "size": artifact.stat().st_size,
                "sha256": compute_sha256(artifact),
            }
            manifest["artifacts"].append(artifact_meta)

    manifest_path = output_dir / "manifest.json"

    if dry_run:
        logger.info(f"[DRY RUN] Would write manifest: {manifest_path}")
        logger.debug(json.dumps(manifest, indent=2))
    else:
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        logger.info(f"Created manifest: {manifest_path}")

    return manifest_path


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main() -> int:
    """
    Main entry point.

    Returns:
        Exit code
    """
    parser = argparse.ArgumentParser(
        description="Prepare T.A.R.S. v1.0.2-rc1 release artifacts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic release (no signing/encryption)
  python scripts/prepare_release_artifacts.py

  # Full enterprise release (signed, encrypted, SBOM, SLSA)
  python scripts/prepare_release_artifacts.py --profile prod --sign --encrypt --include-sbom --include-slsa

  # Dry run to see what would be done
  python scripts/prepare_release_artifacts.py --dry-run --verbose

  # Custom output directory
  python scripts/prepare_release_artifacts.py --output-dir /tmp/tars-release
        """
    )

    parser.add_argument(
        '--version-file',
        type=Path,
        default=PROJECT_ROOT / 'VERSION',
        help='Path to VERSION file (default: VERSION)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=PROJECT_ROOT / 'release',
        help='Output directory (default: release/)'
    )
    parser.add_argument(
        '--profile',
        default='prod',
        help='Configuration profile (default: prod)'
    )
    parser.add_argument(
        '--config',
        type=Path,
        help='Path to config file (optional)'
    )
    parser.add_argument(
        '--sign',
        action='store_true',
        help='Sign artifacts with RSA-PSS'
    )
    parser.add_argument(
        '--encrypt',
        action='store_true',
        help='Encrypt artifacts with AES-256-GCM'
    )
    parser.add_argument(
        '--include-sbom',
        action='store_true',
        help='Generate SBOM (Software Bill of Materials)'
    )
    parser.add_argument(
        '--include-slsa',
        action='store_true',
        help='Generate SLSA provenance'
    )
    parser.add_argument(
        '--run-performance-tests',
        action='store_true',
        help='Run performance tests and include results'
    )
    parser.add_argument(
        '--run-security-audit',
        action='store_true',
        help='Run security audit and include report'
    )
    parser.add_argument(
        '--api-url',
        help='API URL for performance and security tests (e.g., http://localhost:3001)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without making changes'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    parser.add_argument(
        '--verify-release',
        action='store_true',
        help='Run release verification after artifact generation (Phase 14.7 Task 3)'
    )
    parser.add_argument(
        '--verification-policy',
        choices=['strict', 'lenient'],
        default='strict',
        help='Release verification policy mode (default: strict)'
    )
    parser.add_argument(
        '--public-key',
        type=Path,
        help='Path to RSA public key for release verification'
    )
    parser.add_argument(
        '--post-release-validation',
        action='store_true',
        help='Run post-release validation suite (Phase 14.7 Task 4)'
    )
    parser.add_argument(
        '--validation-policy',
        choices=['strict', 'lenient'],
        default='strict',
        help='Post-release validation policy mode (default: strict)'
    )
    parser.add_argument(
        '--baseline-release',
        help='Baseline release version for delta analysis'
    )
    parser.add_argument(
        '--baseline-sbom',
        type=Path,
        help='Baseline SBOM path for delta analysis'
    )
    parser.add_argument(
        '--baseline-slsa',
        type=Path,
        help='Baseline SLSA provenance path for delta analysis'
    )
    parser.add_argument(
        '--api-schema',
        type=Path,
        help='API schema file (OpenAPI/Swagger format) for compatibility check'
    )
    # Phase 14.7 Task 5: Release Publisher arguments
    parser.add_argument(
        '--publish-release',
        action='store_true',
        help='Publish release to artifact repository (Phase 14.7 Task 5)'
    )
    parser.add_argument(
        '--repository-type',
        choices=['local', 's3', 'gcs'],
        default='local',
        help='Repository type for publication (default: local)'
    )
    parser.add_argument(
        '--repository-path',
        type=Path,
        help='Local repository path (for local repository type)'
    )
    parser.add_argument(
        '--repository-bucket',
        help='S3/GCS bucket name (for s3/gcs repository types)'
    )
    parser.add_argument(
        '--repository-prefix',
        default='',
        help='S3/GCS prefix (optional, default: empty)'
    )
    parser.add_argument(
        '--publication-policy',
        choices=['strict', 'lenient'],
        default='strict',
        help='Publication policy mode (default: strict)'
    )
    parser.add_argument(
        '--sign-audit-log',
        action='store_true',
        help='Sign audit logs with RSA-PSS'
    )
    parser.add_argument(
        '--performance-baseline',
        type=Path,
        help='Performance baseline JSON for drift detection'
    )
    parser.add_argument(
        '--security-baseline',
        type=Path,
        help='Security baseline JSON for regression detection'
    )
    parser.add_argument(
        '--behavior-baseline',
        type=Path,
        help='Behavioral test baseline JSON for regression detection'
    )
    # Phase 14.7 Task 6: Release Rollback arguments
    parser.add_argument(
        '--rollback-release',
        help='Rollback a specific release version (Phase 14.7 Task 6)'
    )
    parser.add_argument(
        '--rollback-type',
        choices=['full', 'artifacts_only', 'index_only'],
        default='full',
        help='Rollback type: full (default), artifacts_only, or index_only'
    )
    parser.add_argument(
        '--rollback-policy',
        choices=['strict', 'lenient'],
        default='strict',
        help='Rollback policy mode (default: strict)'
    )
    parser.add_argument(
        '--rollback-dry-run',
        action='store_true',
        help='Simulate rollback without making changes'
    )
    parser.add_argument(
        '--rollback-force',
        action='store_true',
        help='Force rollback despite warnings (use with caution)'
    )
    parser.add_argument(
        '--rollback-no-backup',
        action='store_true',
        help='Skip backup creation before rollback (not recommended)'
    )
    parser.add_argument(
        '--rollback-output-dir',
        type=Path,
        help='Rollback manifest and audit output directory'
    )
    # Phase 14.7 Task 7: Repository Integrity Scanner arguments
    parser.add_argument(
        '--scan-repository',
        action='store_true',
        help='Run repository integrity scan (Phase 14.7 Task 7)'
    )
    parser.add_argument(
        '--scan-policy',
        choices=['strict', 'lenient', 'audit_only'],
        default='strict',
        help='Integrity scan policy mode (default: strict)'
    )
    parser.add_argument(
        '--scan-repair',
        action='store_true',
        help='Enable automatic repairs during scan'
    )
    parser.add_argument(
        '--scan-repair-orphans',
        action='store_true',
        help='Remove orphaned artifacts during scan'
    )
    parser.add_argument(
        '--scan-repair-index',
        action='store_true',
        help='Rebuild repository index during scan'
    )
    parser.add_argument(
        '--scan-output-dir',
        type=Path,
        help='Integrity scan output directory'
    )
    # Phase 14.7 Task 8: Repository Health Dashboard arguments
    parser.add_argument(
        '--generate-dashboard',
        action='store_true',
        help='Generate repository health dashboard (Phase 14.7 Task 8)'
    )
    parser.add_argument(
        '--dashboard-output-dir',
        type=Path,
        help='Dashboard output directory (default: output-dir/dashboard)'
    )
    parser.add_argument(
        '--dashboard-format',
        choices=['json', 'html', 'both'],
        default='both',
        help='Dashboard output format (default: both)'
    )
    parser.add_argument(
        '--dashboard-fail-on-yellow',
        action='store_true',
        help='Exit with error code on yellow health status'
    )
    parser.add_argument(
        '--dashboard-no-fail-on-red',
        action='store_true',
        help='Do not exit with error code on red health status'
    )
    parser.add_argument(
        '--dashboard-green-threshold',
        type=float,
        default=80.0,
        help='Minimum health score for green status (default: 80.0)'
    )
    parser.add_argument(
        '--dashboard-yellow-threshold',
        type=float,
        default=50.0,
        help='Minimum health score for yellow status (default: 50.0)'
    )
    # Phase 14.7 Task 9: Alerting Engine arguments
    parser.add_argument(
        '--run-alerts',
        action='store_true',
        help='Run alerting engine after dashboard generation (Phase 14.7 Task 9)'
    )
    parser.add_argument(
        '--alert-threshold',
        choices=['INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO',
        help='Minimum alert severity to report (default: INFO)'
    )
    parser.add_argument(
        '--alert-channels',
        type=str,
        default='console',
        help='Comma-separated alert channels: console,file,email,webhook (default: console)'
    )
    parser.add_argument(
        '--alert-output-dir',
        type=Path,
        help='Alert output directory (default: output-dir/alerts)'
    )
    parser.add_argument(
        '--alert-email-to',
        type=str,
        help='Email recipient for alert notifications'
    )
    parser.add_argument(
        '--alert-webhook-url',
        type=str,
        help='Webhook URL for alert notifications'
    )
    parser.add_argument(
        '--alert-fail-on-critical',
        action='store_true',
        default=True,
        help='Exit with error code on critical alerts (default: true)'
    )
    parser.add_argument(
        '--alert-no-fail-on-critical',
        action='store_true',
        help='Do not exit with error code on critical alerts'
    )
    parser.add_argument(
        '--alert-fail-on-any',
        action='store_true',
        help='Exit with error code on any alert'
    )
    parser.add_argument(
        '--previous-dashboard',
        type=Path,
        help='Previous dashboard JSON for trend-based alerts'
    )
    # Phase 14.7 Task 10: Trend Analyzer arguments
    parser.add_argument(
        '--update-history',
        action='store_true',
        help='Add current dashboard to history store (Phase 14.7 Task 10)'
    )
    parser.add_argument(
        '--run-trends',
        action='store_true',
        help='Run trend analysis after dashboard generation (Phase 14.7 Task 10)'
    )
    parser.add_argument(
        '--trend-history-dir',
        type=Path,
        help='Directory for dashboard history snapshots (default: output-dir/dashboard-history)'
    )
    parser.add_argument(
        '--trend-output',
        type=Path,
        help='Output path for trend report JSON'
    )
    parser.add_argument(
        '--trend-charts',
        action='store_true',
        help='Generate trend visualization charts'
    )
    parser.add_argument(
        '--trend-min-snapshots',
        type=int,
        default=3,
        help='Minimum snapshots required for trend analysis (default: 3)'
    )
    parser.add_argument(
        '--trend-prediction-horizon',
        type=int,
        default=3,
        help='Number of snapshots to predict ahead (default: 3)'
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(verbose=args.verbose)

    logger.info("=" * 80)
    logger.info("T.A.R.S. v1.0.2-RC1 Release Artifact Preparation")
    logger.info("=" * 80)

    try:
        # Read version
        version = read_version(args.version_file)

        # Check for enterprise features
        enterprise_available, SecurityManager, load_config = import_enterprise_modules()

        if (args.sign or args.encrypt) and not enterprise_available:
            logger.error("Enterprise modules required for signing/encryption but not available")
            logger.error("Install with: pip install -r requirements-dev.txt")
            return 2

        # Create output directory
        if not args.dry_run:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Output directory: {args.output_dir}")
        else:
            logger.info(f"[DRY RUN] Output directory: {args.output_dir}")

        # Collect artifacts
        artifacts = collect_artifacts(PROJECT_ROOT, args.output_dir, args.dry_run)

        # Enterprise features
        signed = False
        encrypted = False
        sbom_generated = False
        slsa_generated = False

        if enterprise_available and (args.sign or args.encrypt):
            logger.info("Initializing enterprise security...")

            # Load config
            config = load_config(profile=args.profile, config_path=args.config)

            # Initialize security manager
            sec_manager = SecurityManager(
                encryption_key_path=config.security.encryption.key_path if args.encrypt else None,
                signing_key_path=config.security.signing.key_path if args.sign else None,
            )

            # Sign artifacts
            if args.sign:
                logger.info("Signing artifacts...")
                signature_files = sign_artifacts(artifacts, sec_manager, args.output_dir, args.dry_run)
                artifacts.extend(signature_files)
                signed = True

            # Encrypt artifacts
            if args.encrypt:
                logger.info("Encrypting artifacts...")
                encrypted_files = encrypt_artifacts(artifacts, sec_manager, args.output_dir, args.dry_run)
                artifacts.extend(encrypted_files)
                encrypted = True

        # SBOM generation (production implementation)
        if args.include_sbom:
            logger.info("Generating SBOM (Software Bill of Materials)...")
            sbom_dir = args.output_dir / "sbom"

            if args.dry_run:
                logger.info(f"[DRY RUN] Would generate SBOM in: {sbom_dir}")
                sbom_generated = True
            else:
                try:
                    # Import SBOM generator
                    from security.sbom_generator import generate_sbom_for_tars

                    # Create SBOM directory
                    sbom_dir.mkdir(parents=True, exist_ok=True)

                    # Determine signing key path
                    signing_key = None
                    if args.sign and enterprise_available:
                        signing_key = config.security.signing.key_path

                    # Generate both CycloneDX and SPDX formats
                    generate_sbom_for_tars(
                        output_dir=sbom_dir,
                        formats=["cyclonedx", "spdx"],
                        sign=args.sign and signing_key is not None,
                        signing_key_path=signing_key,
                        project_version=version
                    )

                    # Add SBOM files to artifacts
                    for sbom_file in sbom_dir.glob("*.json"):
                        artifacts.append(sbom_file)
                        logger.info(f"Generated SBOM: {sbom_file.name}")

                    # Add signatures if generated
                    if args.sign:
                        for sig_file in sbom_dir.glob("*.sig"):
                            artifacts.append(sig_file)
                            logger.debug(f"Generated signature: {sig_file.name}")

                    sbom_generated = True
                    logger.info("✓ SBOM generation complete")

                except ImportError as e:
                    logger.error(f"Failed to import SBOM generator: {e}")
                    logger.error("Ensure security module is available")
                    sbom_generated = False
                except Exception as e:
                    logger.error(f"SBOM generation failed: {e}")
                    if args.verbose:
                        import traceback
                        traceback.print_exc()
                    sbom_generated = False

        # SLSA provenance (production implementation)
        if args.include_slsa:
            logger.info("Generating SLSA provenance...")
            slsa_dir = args.output_dir / "slsa"

            if args.dry_run:
                logger.info(f"[DRY RUN] Would generate SLSA provenance in: {slsa_dir}")
                slsa_generated = True
            else:
                try:
                    # Import SLSA generator
                    from security.slsa_generator import generate_slsa_provenance_for_tars

                    # Create SLSA directory
                    slsa_dir.mkdir(parents=True, exist_ok=True)

                    # Collect all artifact paths for provenance
                    artifact_paths = [a for a in artifacts if a.suffix in ['.json', '.md', '.txt', '.py']]

                    if not artifact_paths:
                        logger.warning("No suitable artifacts found for SLSA provenance")
                        slsa_generated = False
                    else:
                        # Determine signing key path
                        signing_key = None
                        if args.sign and enterprise_available:
                            signing_key = config.security.signing.key_path

                        # Generate provenance
                        slsa_file = slsa_dir / f"tars-v{version}.provenance.json"
                        generate_slsa_provenance_for_tars(
                            artifact_paths=artifact_paths,
                            output_path=slsa_file,
                            build_type="https://slsa.dev/build-types/python/package/v1",
                            sign=args.sign and signing_key is not None,
                            signing_key_path=signing_key,
                            project_version=version
                        )

                        # Add provenance to artifacts
                        artifacts.append(slsa_file)
                        logger.info(f"Generated SLSA provenance: {slsa_file.name}")

                        # Add signature if generated
                        if args.sign:
                            sig_file = Path(str(slsa_file) + '.sig')
                            if sig_file.exists():
                                artifacts.append(sig_file)
                                logger.debug(f"Generated signature: {sig_file.name}")

                        slsa_generated = True
                        logger.info("✓ SLSA provenance generation complete")

                except ImportError as e:
                    logger.error(f"Failed to import SLSA generator: {e}")
                    logger.error("Ensure security module is available")
                    slsa_generated = False
                except Exception as e:
                    logger.error(f"SLSA provenance generation failed: {e}")
                    if args.verbose:
                        import traceback
                        traceback.print_exc()
                    slsa_generated = False

        # Performance testing (Phase 14.7 Task 2)
        perf_test_completed = False
        if args.run_performance_tests:
            logger.info("Running performance tests...")

            if not args.api_url:
                logger.warning("No API URL provided, skipping performance tests")
                logger.warning("Use --api-url to specify target API endpoint")
            elif args.dry_run:
                logger.info(f"[DRY RUN] Would run performance tests against: {args.api_url}")
                perf_test_completed = True
            else:
                try:
                    import subprocess
                    perf_dir = args.output_dir / "performance"
                    perf_dir.mkdir(parents=True, exist_ok=True)

                    perf_json = perf_dir / f"tars-v{version}-performance.json"
                    perf_md = perf_dir / f"tars-v{version}-performance.md"

                    # Run performance tests
                    logger.info(f"Testing API endpoint: {args.api_url}")
                    result = subprocess.run(
                        [
                            sys.executable,
                            str(PROJECT_ROOT / "performance" / "run_performance_tests.py"),
                            "--url", args.api_url,
                            "--duration", "60",  # 1 minute for release validation
                            "--concurrency", "20",
                            "--output-json", str(perf_json),
                            "--output-md", str(perf_md),
                            "--profile", "standard"
                        ] + (["--verbose"] if args.verbose else []),
                        capture_output=True,
                        text=True,
                        timeout=300
                    )

                    if result.returncode == 0:
                        artifacts.append(perf_json)
                        artifacts.append(perf_md)
                        logger.info(f"✓ Performance tests complete: {perf_json.name}, {perf_md.name}")
                        perf_test_completed = True
                    elif result.returncode == 2:
                        logger.warning("Performance regressions detected (see report)")
                        artifacts.append(perf_json)
                        artifacts.append(perf_md)
                        perf_test_completed = True
                    else:
                        logger.error(f"Performance tests failed with exit code {result.returncode}")
                        if args.verbose and result.stderr:
                            logger.error(f"Error output: {result.stderr}")
                        perf_test_completed = False

                except subprocess.TimeoutExpired:
                    logger.error("Performance tests timed out (>5 minutes)")
                    perf_test_completed = False
                except Exception as e:
                    logger.error(f"Performance testing failed: {e}")
                    if args.verbose:
                        import traceback
                        traceback.print_exc()
                    perf_test_completed = False

        # Security audit (Phase 14.7 Task 2)
        security_audit_completed = False
        if args.run_security_audit:
            logger.info("Running security audit...")

            if args.dry_run:
                logger.info(f"[DRY RUN] Would run security audit")
                security_audit_completed = True
            else:
                try:
                    import subprocess
                    audit_dir = args.output_dir / "security_audit"
                    audit_dir.mkdir(parents=True, exist_ok=True)

                    audit_json = audit_dir / f"tars-v{version}-security-audit.json"

                    # Build audit command
                    audit_cmd = [
                        sys.executable,
                        str(PROJECT_ROOT / "security" / "security_audit.py"),
                        "--deep",
                        "--json", str(audit_json)
                    ]

                    # Add SBOM scan if available
                    if sbom_generated:
                        sbom_file = sbom_dir / f"tars-v{version}-cyclonedx.json"
                        if sbom_file.exists():
                            audit_cmd.extend(["--scan-sbom", str(sbom_file)])

                    # Add API check if URL provided
                    if args.api_url:
                        audit_cmd.extend(["--check-api", args.api_url])

                    # Add config check
                    if args.config and args.config.exists():
                        audit_cmd.extend(["--check-config", str(args.config)])

                    if args.verbose:
                        audit_cmd.append("--verbose")

                    # Run security audit
                    logger.info("Performing security audit...")
                    result = subprocess.run(
                        audit_cmd,
                        capture_output=True,
                        text=True,
                        timeout=300
                    )

                    # Always add audit report, even if findings detected
                    if audit_json.exists():
                        artifacts.append(audit_json)
                        logger.info(f"Security audit report: {audit_json.name}")
                        security_audit_completed = True

                    if result.returncode == 0:
                        logger.info("✓ Security audit complete: No critical issues")
                    elif result.returncode == 1:
                        logger.warning("Security audit found HIGH severity issues (see report)")
                    elif result.returncode == 2:
                        logger.warning("Security audit found CRITICAL issues (see report)")
                    else:
                        logger.error(f"Security audit failed with exit code {result.returncode}")
                        if args.verbose and result.stderr:
                            logger.error(f"Error output: {result.stderr}")

                except subprocess.TimeoutExpired:
                    logger.error("Security audit timed out (>5 minutes)")
                    security_audit_completed = False
                except Exception as e:
                    logger.error(f"Security audit failed: {e}")
                    if args.verbose:
                        import traceback
                        traceback.print_exc()
                    security_audit_completed = False

        # Generate manifest
        manifest_path = generate_manifest(
            version=version,
            profile=args.profile,
            artifacts=artifacts,
            output_dir=args.output_dir,
            signed=signed,
            encrypted=encrypted,
            sbom=sbom_generated,
            slsa=slsa_generated,
            perf_tests=perf_test_completed,
            security_audit=security_audit_completed,
            dry_run=args.dry_run,
        )

        # Release Verification (Phase 14.7 Task 3)
        verification_passed = False
        if args.verify_release:
            logger.info("")
            logger.info("=" * 80)
            logger.info("RELEASE VERIFICATION (Phase 14.7 Task 3)")
            logger.info("=" * 80)
            logger.info("")

            if args.dry_run:
                logger.info(f"[DRY RUN] Would verify release artifacts")
                logger.info(f"  Policy: {args.verification_policy}")
                logger.info(f"  Manifest: {manifest_path}")
                if sbom_generated:
                    logger.info(f"  SBOM: Yes")
                if slsa_generated:
                    logger.info(f"  SLSA: Yes")
                verification_passed = True
            else:
                try:
                    # Import release verifier
                    from security.release_verifier import ReleaseVerifier

                    # Determine primary artifact
                    primary_artifact = args.output_dir / f"tars-v{version}.tar.gz"
                    if not primary_artifact.exists():
                        # Use first artifact as primary
                        artifact_files = [a for a in artifacts if a.suffix in ['.tar.gz', '.zip', '.json']]
                        primary_artifact = artifact_files[0] if artifact_files else artifacts[0]

                    # Determine SBOM and SLSA paths
                    sbom_file = None
                    slsa_file = None

                    if sbom_generated:
                        sbom_candidates = list((args.output_dir / "sbom").glob(f"tars-v{version}-cyclonedx.json"))
                        if not sbom_candidates:
                            sbom_candidates = list((args.output_dir / "sbom").glob("*cyclonedx.json"))
                        sbom_file = sbom_candidates[0] if sbom_candidates else None

                    if slsa_generated:
                        slsa_candidates = list((args.output_dir / "slsa").glob(f"tars-v{version}.provenance.json"))
                        if not slsa_candidates:
                            slsa_candidates = list((args.output_dir / "slsa").glob("*.provenance.json"))
                        slsa_file = slsa_candidates[0] if slsa_candidates else None

                    # Initialize verifier
                    public_key = args.public_key
                    if not public_key and enterprise_available and args.sign:
                        # Use public key from config
                        public_key = config.security.signing.key_path.with_suffix('.pub')
                        if not public_key.exists():
                            public_key = None

                    logger.info(f"Initializing release verifier (mode: {args.verification_policy})")
                    verifier = ReleaseVerifier(
                        mode=args.verification_policy,
                        public_key_path=public_key
                    )

                    # Run verification
                    logger.info(f"Verifying release artifacts...")
                    logger.info(f"  Artifact: {primary_artifact.name}")
                    if sbom_file:
                        logger.info(f"  SBOM: {sbom_file.name}")
                    if slsa_file:
                        logger.info(f"  SLSA: {slsa_file.name}")
                    logger.info(f"  Manifest: {manifest_path.name}")
                    logger.info("")

                    verification_report = verifier.verify_release(
                        artifact_path=primary_artifact,
                        sbom_path=sbom_file,
                        slsa_path=slsa_file,
                        manifest_path=manifest_path,
                        version=version
                    )

                    # Save verification reports
                    verify_dir = args.output_dir / "verification"
                    verify_dir.mkdir(parents=True, exist_ok=True)

                    verify_json = verify_dir / f"tars-v{version}-verification.json"
                    verify_text = verify_dir / f"tars-v{version}-verification.txt"

                    verifier.save_report(verification_report, verify_json, format='json')
                    verifier.save_report(verification_report, verify_text, format='text')

                    artifacts.append(verify_json)
                    artifacts.append(verify_text)

                    # Check verification status
                    if verification_report.overall_status == "passed":
                        logger.info("✓ Release verification PASSED")
                        verification_passed = True
                    elif verification_report.overall_status == "warning":
                        if args.verification_policy == 'lenient':
                            logger.warning("⚠ Release verification completed with WARNINGS (lenient mode)")
                            verification_passed = True
                        else:
                            logger.error("✗ Release verification FAILED (warnings in strict mode)")
                            verification_passed = False
                    else:
                        logger.error("✗ Release verification FAILED")
                        logger.error(f"  Failed checks: {verification_report.failed_checks}/{verification_report.total_checks}")
                        if verification_report.errors:
                            for error in verification_report.errors[:5]:
                                logger.error(f"  - {error}")
                        verification_passed = False

                    logger.info("")
                    logger.info(f"Verification report: {verify_json}")
                    logger.info(f"Summary report: {verify_text}")
                    logger.info("")

                    # In strict mode, fail release if verification failed
                    if not verification_passed and args.verification_policy == 'strict':
                        logger.error("=" * 80)
                        logger.error("RELEASE VERIFICATION FAILED - ABORTING")
                        logger.error("=" * 80)
                        logger.error("Fix verification errors and retry, or use --verification-policy lenient")
                        return 7  # Policy gate failure

                except ImportError as e:
                    logger.error(f"Failed to import release verifier: {e}")
                    logger.error("Ensure security module is available")
                    if args.verification_policy == 'strict':
                        return 7
                    else:
                        logger.warning("Continuing without verification (lenient mode)")
                        verification_passed = False
                except Exception as e:
                    logger.error(f"Release verification failed: {e}")
                    if args.verbose:
                        import traceback
                        traceback.print_exc()
                    if args.verification_policy == 'strict':
                        return 7
                    else:
                        logger.warning("Continuing without verification (lenient mode)")
                        verification_passed = False

        # Post-Release Validation (Phase 14.7 Task 4)
        post_validation_passed = False
        if args.post_release_validation:
            logger.info("")
            logger.info("=" * 80)
            logger.info("POST-RELEASE VALIDATION SUITE (Phase 14.7 Task 4)")
            logger.info("=" * 80)
            logger.info("")

            if args.dry_run:
                logger.info(f"[DRY RUN] Would run post-release validation")
                logger.info(f"  Policy: {args.validation_policy}")
                logger.info(f"  Baseline version: {args.baseline_release or 'N/A'}")
                post_validation_passed = True
            else:
                try:
                    # Import post-release validation orchestrator
                    from validation.post_release_validation import ValidationOrchestrator

                    if not args.baseline_release:
                        logger.warning("No baseline release version specified")
                        logger.warning("Skipping post-release validation (use --baseline-release)")
                        logger.warning("")
                        post_validation_passed = None  # Skipped
                    else:
                        # Determine current SBOM and SLSA paths
                        current_sbom = None
                        current_slsa = None
                        current_perf = None
                        current_security = None
                        current_behavior = None

                        if sbom_generated:
                            sbom_candidates = list((args.output_dir / "sbom").glob(f"tars-v{version}-cyclonedx.json"))
                            current_sbom = sbom_candidates[0] if sbom_candidates else None

                        if slsa_generated:
                            slsa_candidates = list((args.output_dir / "slsa").glob(f"tars-v{version}.provenance.json"))
                            current_slsa = slsa_candidates[0] if slsa_candidates else None

                        if perf_test_completed:
                            perf_dir = args.output_dir / "performance"
                            perf_candidates = list(perf_dir.glob(f"tars-v{version}-performance.json"))
                            current_perf = perf_candidates[0] if perf_candidates else None

                        if security_audit_completed:
                            audit_dir = args.output_dir / "security_audit"
                            audit_candidates = list(audit_dir.glob(f"tars-v{version}-security-audit.json"))
                            current_security = audit_candidates[0] if audit_candidates else None

                        # Initialize orchestrator
                        logger.info(f"Initializing validation orchestrator (mode: {args.validation_policy})")
                        orchestrator = ValidationOrchestrator(mode=args.validation_policy)

                        # Run validation
                        logger.info(f"Running post-release validation...")
                        logger.info(f"  Current version: {version}")
                        logger.info(f"  Baseline version: {args.baseline_release}")
                        if args.baseline_sbom and current_sbom:
                            logger.info(f"  SBOM delta: Yes")
                        if args.baseline_slsa and current_slsa:
                            logger.info(f"  SLSA delta: Yes")
                        if args.api_schema:
                            logger.info(f"  API compatibility: Yes")
                        if args.performance_baseline and current_perf:
                            logger.info(f"  Performance drift: Yes")
                        if args.security_baseline and current_security:
                            logger.info(f"  Security regression: Yes")
                        if args.behavior_baseline:
                            logger.info(f"  Behavioral regression: Yes")
                        logger.info("")

                        validation_report = orchestrator.validate_release(
                            version=version,
                            baseline_version=args.baseline_release,
                            baseline_sbom_path=args.baseline_sbom,
                            current_sbom_path=current_sbom,
                            baseline_slsa_path=args.baseline_slsa,
                            current_slsa_path=current_slsa,
                            baseline_api_schema_path=args.api_schema,
                            current_api_schema_path=args.api_schema,  # Use same schema for now
                            baseline_perf_path=args.performance_baseline,
                            current_perf_path=current_perf,
                            baseline_security_path=args.security_baseline,
                            current_security_path=current_security,
                            baseline_behavior_path=args.behavior_baseline,
                            current_behavior_path=current_behavior
                        )

                        # Save validation reports
                        validation_dir = args.output_dir / "post_validation"
                        validation_dir.mkdir(parents=True, exist_ok=True)

                        validation_json = validation_dir / f"tars-v{version}-post-validation.json"
                        validation_text = validation_dir / f"tars-v{version}-post-validation.txt"

                        orchestrator.generate_json_report(validation_report, validation_json)
                        orchestrator.generate_text_report(validation_report, validation_text)

                        artifacts.append(validation_json)
                        artifacts.append(validation_text)

                        # Check validation status
                        if validation_report.overall_status == "passed":
                            logger.info("✓ Post-release validation PASSED")
                            post_validation_passed = True
                        elif validation_report.overall_status == "warning":
                            if args.validation_policy == 'lenient':
                                logger.warning("⚠ Post-release validation completed with WARNINGS (lenient mode)")
                                post_validation_passed = True
                            else:
                                logger.error("✗ Post-release validation FAILED (warnings in strict mode)")
                                post_validation_passed = False
                        else:
                            logger.error("✗ Post-release validation FAILED")
                            logger.error(f"  Failed checks: {validation_report.failed_checks}/{validation_report.total_checks}")
                            logger.error(f"  Summary: {validation_report.summary}")
                            post_validation_passed = False

                        logger.info("")
                        logger.info(f"Validation report: {validation_json}")
                        logger.info(f"Summary report: {validation_text}")
                        logger.info(f"Execution time: {validation_report.execution_time_seconds:.2f}s")
                        logger.info("")

                        # In strict mode, fail release if validation failed
                        if not post_validation_passed and args.validation_policy == 'strict':
                            logger.error("=" * 80)
                            logger.error("POST-RELEASE VALIDATION FAILED - ABORTING")
                            logger.error("=" * 80)
                            logger.error("Fix validation errors and retry, or use --validation-policy lenient")
                            return validation_report.exit_code  # Return specific exit code

                except ImportError as e:
                    logger.error(f"Failed to import post-release validation module: {e}")
                    logger.error("Ensure validation module is available")
                    if args.validation_policy == 'strict':
                        return 28  # Validation orchestration error
                    else:
                        logger.warning("Continuing without post-release validation (lenient mode)")
                        post_validation_passed = False
                except Exception as e:
                    logger.error(f"Post-release validation failed: {e}")
                    if args.verbose:
                        import traceback
                        traceback.print_exc()
                    if args.validation_policy == 'strict':
                        return 29  # General validation error
                    else:
                        logger.warning("Continuing without post-release validation (lenient mode)")
                        post_validation_passed = False

        # ====================================================================
        # PHASE 14.7 TASK 5: RELEASE PUBLICATION
        # ====================================================================

        publication_report = None
        publication_successful = False

        if args.publish_release:
            logger.info("")
            logger.info("=" * 80)
            logger.info("PHASE 14.7 TASK 5: RELEASE PUBLICATION")
            logger.info("=" * 80)

            try:
                from publisher.release_publisher import (
                    RepositoryFactory,
                    PublisherOrchestrator,
                    PublicationPolicy,
                    VerificationResult,
                    ValidationResult,
                )

                # Build repository configuration
                repo_config = {"type": args.repository_type}
                if args.repository_type == "local":
                    repo_path = args.repository_path or (PROJECT_ROOT / "artifact-repository")
                    repo_config["path"] = str(repo_path)
                    logger.info(f"Repository: Local ({repo_path})")
                elif args.repository_type == "s3":
                    repo_config["bucket"] = args.repository_bucket or "tars-releases"
                    repo_config["prefix"] = args.repository_prefix
                    repo_config["local_base"] = str(PROJECT_ROOT / "s3-simulation")
                    logger.info(f"Repository: S3 (bucket={repo_config['bucket']}, prefix={repo_config['prefix']})")
                elif args.repository_type == "gcs":
                    repo_config["bucket"] = args.repository_bucket or "tars-releases"
                    repo_config["prefix"] = args.repository_prefix
                    repo_config["local_base"] = str(PROJECT_ROOT / "gcs-simulation")
                    logger.info(f"Repository: GCS (bucket={repo_config['bucket']}, prefix={repo_config['prefix']})")

                # Create repository
                repository = RepositoryFactory.create(args.repository_type, repo_config)

                # Convert verification result
                verification_result_obj = None
                if verification_passed:
                    verification_result_obj = VerificationResult(
                        passed=True,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        hash_verified=True,
                        signature_verified=signed,
                        sbom_validated=sbom_generated,
                        slsa_validated=slsa_generated,
                        policy_passed=True,
                        exit_code=0
                    )

                # Convert validation result
                validation_result_obj = None
                if args.post_release_validation and post_validation_passed is not None:
                    validation_result_obj = ValidationResult(
                        passed=post_validation_passed,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        sbom_delta_passed=post_validation_passed,
                        slsa_delta_passed=post_validation_passed,
                        api_compat_passed=post_validation_passed,
                        performance_passed=post_validation_passed,
                        security_passed=post_validation_passed,
                        behavioral_passed=post_validation_passed,
                        exit_code=0 if post_validation_passed else 20
                    )

                # Create orchestrator
                policy_mode = PublicationPolicy.STRICT if args.publication_policy == 'strict' else PublicationPolicy.LENIENT
                orchestrator = PublisherOrchestrator(
                    repository=repository,
                    policy_mode=policy_mode,
                    sign_audit_logs=args.sign_audit_log,
                    require_signatures=signed,
                    require_encryption=encrypted
                )

                # Publish release
                logger.info("")
                logger.info("Publishing release artifacts...")
                logger.info(f"  Version: {version}")
                logger.info(f"  Policy: {args.publication_policy}")
                logger.info(f"  Verification: {'PASSED' if verification_passed else 'N/A'}")
                logger.info(f"  Validation: {'PASSED' if post_validation_passed else 'N/A'}")
                logger.info("")

                publication_report = orchestrator.publish_release(
                    version=version,
                    release_dir=release_version_dir,
                    verification_result=verification_result_obj,
                    validation_result=validation_result_obj,
                    audit_output_dir=release_version_dir / "audit"
                )

                # Generate publication reports
                publication_json = release_version_dir / "publication_report.json"
                publication_text = release_version_dir / "publication_report.txt"

                orchestrator.generate_json_report(publication_report, publication_json)
                orchestrator.generate_text_report(publication_report, publication_text)

                # Check result
                if publication_report.status == "success":
                    logger.info("=" * 80)
                    logger.info("✓ RELEASE PUBLICATION SUCCESSFUL")
                    logger.info("=" * 80)
                    logger.info(f"  Status: {publication_report.status.upper()}")
                    logger.info(f"  Artifacts: {len(publication_report.artifacts_published)}")
                    logger.info(f"  Total Size: {publication_report.total_size_bytes:,} bytes")
                    logger.info(f"  Index Updated: {'YES' if publication_report.index_updated else 'NO'}")
                    logger.info(f"  Audit Log: {'YES' if publication_report.audit_log_created else 'NO'}")
                    logger.info(f"  Duration: {publication_report.publication_duration_seconds:.2f}s")
                    logger.info("")
                    logger.info(f"Summary: {publication_report.summary}")
                    logger.info("")
                    logger.info(f"Publication report: {publication_json}")
                    logger.info(f"Summary report: {publication_text}")
                    publication_successful = True
                else:
                    logger.error("=" * 80)
                    logger.error("✗ RELEASE PUBLICATION FAILED")
                    logger.error("=" * 80)
                    logger.error(f"  Status: {publication_report.status.upper()}")
                    logger.error(f"  Exit Code: {publication_report.exit_code}")
                    logger.error(f"  Errors: {len(publication_report.errors)}")
                    for error in publication_report.errors:
                        logger.error(f"    - {error}")
                    logger.error("")
                    logger.error(f"Summary: {publication_report.summary}")
                    publication_successful = False

                    # In strict mode, abort on publication failure
                    if args.publication_policy == 'strict':
                        logger.error("")
                        logger.error("RELEASE PUBLICATION FAILED - ABORTING")
                        logger.error("Fix publication errors and retry, or use --publication-policy lenient")
                        return publication_report.exit_code

            except ImportError as e:
                logger.error(f"Failed to import release publisher module: {e}")
                logger.error("Ensure publisher module is available")
                if args.publication_policy == 'strict':
                    return 37  # Atomic publish error
                else:
                    logger.warning("Continuing without release publication (lenient mode)")
                    publication_successful = False
            except Exception as e:
                logger.error(f"Release publication failed: {e}")
                if args.verbose:
                    import traceback
                    traceback.print_exc()
                if args.publication_policy == 'strict':
                    return 39  # General publication error
                else:
                    logger.warning("Continuing without release publication (lenient mode)")
                    publication_successful = False

        # ====================================================================
        # PHASE 14.7 TASK 6: RELEASE ROLLBACK
        # ====================================================================

        rollback_report = None
        rollback_successful = False

        if args.rollback_release:
            logger.info("")
            logger.info("=" * 80)
            logger.info("PHASE 14.7 TASK 6: RELEASE ROLLBACK")
            logger.info("=" * 80)

            try:
                from publisher.release_publisher import RepositoryFactory
                from rollback.release_rollback import (
                    RollbackOrchestrator,
                    RollbackPolicy,
                    RollbackType,
                )

                # Build repository configuration (reuse from publication args)
                repo_config = {"type": args.repository_type}
                if args.repository_type == "local":
                    repo_path = args.repository_path or (PROJECT_ROOT / "artifact-repository")
                    repo_config["path"] = str(repo_path)
                    logger.info(f"Repository: Local ({repo_path})")
                elif args.repository_type == "s3":
                    repo_config["bucket"] = args.repository_bucket or "tars-releases"
                    repo_config["prefix"] = args.repository_prefix
                    repo_config["local_base"] = str(PROJECT_ROOT / "s3-simulation")
                    logger.info(f"Repository: S3 (bucket={repo_config['bucket']}, prefix={repo_config['prefix']})")
                elif args.repository_type == "gcs":
                    repo_config["bucket"] = args.repository_bucket or "tars-releases"
                    repo_config["prefix"] = args.repository_prefix
                    repo_config["local_base"] = str(PROJECT_ROOT / "gcs-simulation")
                    logger.info(f"Repository: GCS (bucket={repo_config['bucket']}, prefix={repo_config['prefix']})")

                # Create repository
                repository = RepositoryFactory.create(args.repository_type, repo_config)

                # Map rollback type
                rollback_type_map = {
                    "full": RollbackType.FULL,
                    "artifacts_only": RollbackType.ARTIFACTS_ONLY,
                    "index_only": RollbackType.INDEX_ONLY,
                }
                rollback_type = rollback_type_map[args.rollback_type]

                # Create orchestrator
                policy_mode = RollbackPolicy.STRICT if args.rollback_policy == 'strict' else RollbackPolicy.LENIENT
                orchestrator = RollbackOrchestrator(
                    repository=repository,
                    policy_mode=policy_mode,
                    sign_audit_logs=args.sign_audit_log,
                )

                # Execute rollback
                logger.info("")
                logger.info("Rolling back release...")
                logger.info(f"  Version: {args.rollback_release}")
                logger.info(f"  Type: {args.rollback_type}")
                logger.info(f"  Policy: {args.rollback_policy}")
                logger.info(f"  Dry Run: {'YES' if args.rollback_dry_run else 'NO'}")
                logger.info(f"  Backup: {'NO' if args.rollback_no_backup else 'YES'}")
                logger.info("")

                # Determine output directories
                rollback_manifest_dir = args.rollback_output_dir or (PROJECT_ROOT / "rollback-manifests")
                rollback_audit_dir = args.rollback_output_dir or (PROJECT_ROOT / "rollback-audit")

                rollback_report = orchestrator.rollback_release(
                    version=args.rollback_release,
                    rollback_type=rollback_type,
                    dry_run=args.rollback_dry_run,
                    force=args.rollback_force,
                    create_backup=not args.rollback_no_backup,
                    audit_output_dir=rollback_audit_dir,
                    manifest_output_dir=rollback_manifest_dir,
                )

                # Generate rollback reports
                rollback_json = rollback_manifest_dir / f"{args.rollback_release}.rollback-report.json"
                rollback_text = rollback_manifest_dir / f"{args.rollback_release}.rollback-report.txt"

                orchestrator.generate_json_report(rollback_report, rollback_json)
                orchestrator.generate_text_report(rollback_report, rollback_text)

                # Check result
                if rollback_report.status in ["success", "dry_run"]:
                    logger.info("=" * 80)
                    if args.rollback_dry_run:
                        logger.info("✓ ROLLBACK DRY RUN COMPLETE")
                    else:
                        logger.info("✓ RELEASE ROLLBACK SUCCESSFUL")
                    logger.info("=" * 80)
                    logger.info(f"  Status: {rollback_report.status.upper()}")
                    logger.info(f"  Artifacts Removed: {rollback_report.total_artifacts_removed}")
                    logger.info(f"  Bytes Freed: {rollback_report.total_bytes_freed:,} bytes")
                    logger.info(f"  Index Updated: {'YES' if rollback_report.index_updated else 'NO'}")
                    logger.info(f"  Backup Created: {'YES' if rollback_report.backup_created else 'NO'}")
                    logger.info(f"  Manifest Created: {'YES' if rollback_report.manifest_created else 'NO'}")
                    logger.info(f"  Duration: {rollback_report.rollback_duration_seconds:.2f}s")
                    logger.info("")
                    logger.info(f"Summary: {rollback_report.summary}")
                    logger.info("")
                    if not args.rollback_dry_run:
                        logger.info(f"Rollback report: {rollback_json}")
                        logger.info(f"Summary report: {rollback_text}")
                    rollback_successful = True
                else:
                    logger.error("=" * 80)
                    logger.error("✗ RELEASE ROLLBACK FAILED")
                    logger.error("=" * 80)
                    logger.error(f"  Status: {rollback_report.status.upper()}")
                    logger.error(f"  Exit Code: {rollback_report.exit_code}")
                    logger.error(f"  Errors: {len(rollback_report.errors)}")
                    for error in rollback_report.errors:
                        logger.error(f"    - {error}")
                    logger.error("")
                    logger.error(f"Summary: {rollback_report.summary}")
                    rollback_successful = False

                    # In strict mode, abort on rollback failure
                    if args.rollback_policy == 'strict':
                        logger.error("")
                        logger.error("ROLLBACK FAILED - ABORTING")
                        logger.error("Fix rollback errors and retry, or use --rollback-policy lenient")
                        return rollback_report.exit_code

            except ImportError as e:
                logger.error(f"Failed to import rollback module: {e}")
                logger.error("Ensure rollback module is available")
                if args.rollback_policy == 'strict':
                    return 43  # Atomic rollback error
                else:
                    logger.warning("Continuing without rollback (lenient mode)")
                    rollback_successful = False
            except Exception as e:
                logger.error(f"Rollback failed: {e}")
                if args.verbose:
                    import traceback
                    traceback.print_exc()
                if args.rollback_policy == 'strict':
                    return 49  # General rollback error
                else:
                    logger.warning("Continuing without rollback (lenient mode)")
                    rollback_successful = False

        # ====================================================================
        # PHASE 14.7 TASK 7: REPOSITORY INTEGRITY SCANNER
        # ====================================================================

        scan_report = None
        scan_successful = False

        if args.scan_repository:
            logger.info("")
            logger.info("=" * 80)
            logger.info("PHASE 14.7 TASK 7: REPOSITORY INTEGRITY SCAN")
            logger.info("=" * 80)

            try:
                from publisher.release_publisher import RepositoryFactory
                from integrity.repository_integrity_scanner import (
                    IntegrityScanOrchestrator,
                    IntegrityScanPolicy,
                    IntegrityReportBuilder,
                    IntegrityScanStatus,
                )

                # Build repository configuration (reuse from publication args)
                repo_config = {"type": args.repository_type}
                if args.repository_type == "local":
                    repo_path = args.repository_path or (PROJECT_ROOT / "artifact-repository")
                    repo_config["path"] = str(repo_path)
                    logger.info(f"Repository: Local ({repo_path})")
                elif args.repository_type == "s3":
                    repo_config["bucket"] = args.repository_bucket or "tars-releases"
                    repo_config["prefix"] = args.repository_prefix
                    repo_config["local_base"] = str(PROJECT_ROOT / "s3-simulation")
                    logger.info(f"Repository: S3 (bucket={repo_config['bucket']}, prefix={repo_config['prefix']})")
                elif args.repository_type == "gcs":
                    repo_config["bucket"] = args.repository_bucket or "tars-releases"
                    repo_config["prefix"] = args.repository_prefix
                    repo_config["local_base"] = str(PROJECT_ROOT / "gcs-simulation")
                    logger.info(f"Repository: GCS (bucket={repo_config['bucket']}, prefix={repo_config['prefix']})")

                # Create repository
                repository = RepositoryFactory.create(args.repository_type, repo_config)

                # Map policy mode
                policy_map = {
                    "strict": IntegrityScanPolicy.STRICT,
                    "lenient": IntegrityScanPolicy.LENIENT,
                    "audit_only": IntegrityScanPolicy.AUDIT_ONLY,
                }
                policy_mode = policy_map[args.scan_policy]

                # Create orchestrator
                orchestrator = IntegrityScanOrchestrator(
                    repository=repository,
                    policy_mode=policy_mode,
                    repair_enabled=args.scan_repair,
                    repair_orphans=args.scan_repair_orphans,
                    repair_index=args.scan_repair_index,
                )

                # Execute scan
                logger.info("")
                logger.info("Scanning repository integrity...")
                logger.info(f"  Policy: {args.scan_policy}")
                logger.info(f"  Repair Enabled: {'YES' if args.scan_repair else 'NO'}")
                logger.info(f"  Repair Orphans: {'YES' if args.scan_repair_orphans else 'NO'}")
                logger.info(f"  Repair Index: {'YES' if args.scan_repair_index else 'NO'}")
                logger.info("")

                # Determine output directory
                scan_output_dir = args.scan_output_dir or (PROJECT_ROOT / "integrity-scan")

                scan_json = scan_output_dir / "integrity-scan-report.json"
                scan_text = scan_output_dir / "integrity-scan-report.txt"

                scan_report = orchestrator.scan_repository(
                    output_dir=scan_output_dir,
                    json_report_path=scan_json,
                    text_report_path=scan_text,
                )

                # Check result
                if scan_report.scan_status in [IntegrityScanStatus.SUCCESS.value, IntegrityScanStatus.SUCCESS_WITH_WARNINGS.value]:
                    logger.info("=" * 80)
                    logger.info("✓ REPOSITORY INTEGRITY SCAN COMPLETE")
                    logger.info("=" * 80)
                    logger.info(f"  Status: {scan_report.scan_status.upper()}")
                    logger.info(f"  Total Versions: {scan_report.total_versions}")
                    logger.info(f"  Total Artifacts: {scan_report.total_artifacts}")
                    logger.info(f"  Total Size: {scan_report.total_size_bytes:,} bytes ({scan_report.total_size_bytes / 1024 / 1024:.2f} MB)")
                    logger.info(f"  Total Issues: {scan_report.total_issues}")
                    logger.info(f"    Critical: {scan_report.critical_issues}")
                    logger.info(f"    Errors: {scan_report.error_issues}")
                    logger.info(f"    Warnings: {scan_report.warning_issues}")
                    logger.info(f"  Corrupted Artifacts: {scan_report.corrupted_artifacts}")
                    logger.info(f"  Missing Artifacts: {scan_report.missing_artifacts}")
                    logger.info(f"  Orphan Artifacts: {scan_report.orphan_artifacts}")
                    logger.info(f"  Index Issues: {scan_report.index_issues}")
                    logger.info(f"  SBOM/SLSA Issues: {scan_report.sbom_slsa_issues}")
                    if scan_report.repair_enabled:
                        logger.info(f"  Repairs Applied: {scan_report.repairs_applied}")
                        logger.info(f"  Repairs Failed: {scan_report.repairs_failed}")
                    logger.info(f"  Duration: {scan_report.scan_duration_seconds:.2f}s")
                    logger.info("")
                    logger.info(f"Summary: {scan_report.summary}")
                    logger.info("")
                    logger.info(f"JSON report: {scan_json}")
                    logger.info(f"Text report: {scan_text}")
                    logger.info("")

                    scan_successful = True

                    # In strict mode, fail if critical or error issues found
                    if args.scan_policy == 'strict' and (scan_report.critical_issues > 0 or scan_report.error_issues > 0):
                        logger.error("INTEGRITY SCAN DETECTED CRITICAL/ERROR ISSUES - ABORTING")
                        logger.error("Fix integrity issues and retry, or use --scan-policy lenient")
                        return scan_report.exit_code

                else:
                    logger.error("=" * 80)
                    logger.error("✗ REPOSITORY INTEGRITY SCAN FAILED")
                    logger.error("=" * 80)
                    logger.error(f"  Status: {scan_report.scan_status.upper()}")
                    logger.error(f"  Exit Code: {scan_report.exit_code}")
                    logger.error(f"  Total Issues: {scan_report.total_issues}")
                    logger.error(f"    Critical: {scan_report.critical_issues}")
                    logger.error(f"    Errors: {scan_report.error_issues}")
                    logger.error("")
                    logger.error(f"Summary: {scan_report.summary}")
                    logger.error("")
                    logger.error("Top Issues:")
                    for i, issue in enumerate(scan_report.all_issues[:10], 1):
                        logger.error(f"  {i}. [{issue.severity.upper()}] {issue.description}")
                    logger.error("")
                    scan_successful = False

                    # In strict mode, abort on scan failure
                    if args.scan_policy == 'strict':
                        logger.error("")
                        logger.error("INTEGRITY SCAN FAILED - ABORTING")
                        logger.error("Fix integrity issues and retry, or use --scan-policy lenient")
                        return scan_report.exit_code

            except ImportError as e:
                logger.error(f"Failed to import integrity scanner module: {e}")
                logger.error("Ensure integrity module is available")
                if args.scan_policy == 'strict':
                    return 59  # General integrity error
                else:
                    logger.warning("Continuing without integrity scan (lenient mode)")
                    scan_successful = False
            except Exception as e:
                logger.error(f"Integrity scan failed: {e}")
                if args.verbose:
                    import traceback
                    traceback.print_exc()
                if args.scan_policy == 'strict':
                    return 59  # General integrity error
                else:
                    logger.warning("Continuing without integrity scan (lenient mode)")
                    scan_successful = False

        # Repository Health Dashboard (Phase 14.7 Task 8)
        dashboard_generated = False
        dashboard_health_score = None
        dashboard_health_status = None
        if args.generate_dashboard:
            logger.info("")
            logger.info("=" * 80)
            logger.info("REPOSITORY HEALTH DASHBOARD (Phase 14.7 Task 8)")
            logger.info("=" * 80)
            logger.info("")

            try:
                # Import dashboard components
                from analytics.repository_health_dashboard import (
                    RepositoryHealthDashboard,
                    DashboardConfig,
                    DashboardFormat,
                    HealthThresholds,
                    DashboardError
                )

                # Determine output directory
                dashboard_output_dir = args.dashboard_output_dir or (args.output_dir / "dashboard")

                # Determine repository path (use same as publication/rollback if available)
                repository_path = args.repository_path or (PROJECT_ROOT / "artifact-repository")

                # Determine report source directories
                scan_reports_dir = args.scan_output_dir or (args.output_dir / "integrity-scan")
                rollback_reports_dir = args.rollback_output_dir or (args.output_dir / "rollback")
                publisher_reports_dir = args.output_dir / "publish"
                validator_reports_dir = args.output_dir / "validation"

                logger.info(f"Repository: {repository_path}")
                logger.info(f"Dashboard Output: {dashboard_output_dir}")
                logger.info(f"Format: {args.dashboard_format}")
                logger.info("")

                # Build dashboard configuration
                dashboard_config = DashboardConfig(
                    repository_path=repository_path,
                    output_dir=dashboard_output_dir,
                    format=DashboardFormat(args.dashboard_format),
                    scan_output_dir=scan_reports_dir if scan_reports_dir.exists() else None,
                    rollback_output_dir=rollback_reports_dir if rollback_reports_dir.exists() else None,
                    publisher_output_dir=publisher_reports_dir if publisher_reports_dir.exists() else None,
                    validator_output_dir=validator_reports_dir if validator_reports_dir.exists() else None,
                    thresholds=HealthThresholds(
                        green_min=args.dashboard_green_threshold,
                        yellow_min=args.dashboard_yellow_threshold
                    ),
                    fail_on_yellow=args.dashboard_fail_on_yellow,
                    fail_on_red=not args.dashboard_no_fail_on_red,
                    verbose=args.verbose
                )

                # Generate dashboard
                logger.info("Generating health dashboard...")
                dashboard = RepositoryHealthDashboard(dashboard_config)
                health_report = dashboard.generate_dashboard()

                dashboard_health_score = health_report.repository_score
                dashboard_health_status = health_report.overall_health

                logger.info("")
                logger.info("✓ DASHBOARD GENERATION COMPLETE")
                logger.info(f"  Health Status: {health_report.overall_health.upper()}")
                logger.info(f"  Health Score: {health_report.repository_score:.1f}/100")
                logger.info(f"  Total Issues: {health_report.total_issues}")
                logger.info(f"  Critical: {health_report.critical_issues}, Errors: {health_report.error_issues}, Warnings: {health_report.warning_issues}")
                logger.info(f"  Versions: {health_report.total_versions} ({health_report.healthy_versions} healthy, {health_report.warning_versions} warnings, {health_report.critical_versions} critical)")
                logger.info("")

                # Output files
                if args.dashboard_format in ['json', 'both']:
                    json_path = dashboard_output_dir / "health-dashboard.json"
                    logger.info(f"  JSON Report: {json_path}")

                if args.dashboard_format in ['html', 'both']:
                    html_path = dashboard_output_dir / "health-dashboard.html"
                    logger.info(f"  HTML Dashboard: {html_path}")

                logger.info("")

                # Display recommendations
                if health_report.recommendations:
                    logger.info("Recommendations:")
                    for rec in health_report.recommendations[:5]:  # Show first 5
                        logger.info(f"  • {rec}")
                    if len(health_report.recommendations) > 5:
                        logger.info(f"  ... and {len(health_report.recommendations) - 5} more (see dashboard)")
                    logger.info("")

                dashboard_generated = True

                # Check exit code from dashboard
                dashboard_exit_code = dashboard.determine_exit_code(health_report)
                if dashboard_exit_code != 60:  # EXIT_HEALTH_OK
                    logger.warning(f"Dashboard health check returned exit code {dashboard_exit_code}")
                    if dashboard_config.fail_on_red and health_report.overall_health == "red":
                        logger.error("Repository health is CRITICAL (red) - failing build")
                        return dashboard_exit_code
                    elif dashboard_config.fail_on_yellow and health_report.overall_health == "yellow":
                        logger.warning("Repository health has WARNINGS (yellow) - failing build")
                        return dashboard_exit_code

            except DashboardError as e:
                logger.error(f"Dashboard generation failed: {e}")
                if args.verbose:
                    import traceback
                    traceback.print_exc()
                dashboard_generated = False
                # Don't fail the entire release process for dashboard errors
                logger.warning("Continuing without dashboard (non-critical)")

            except Exception as e:
                logger.error(f"Unexpected dashboard error: {e}")
                if args.verbose:
                    import traceback
                    traceback.print_exc()
                dashboard_generated = False
                logger.warning("Continuing without dashboard (non-critical)")

        # ====================================================================
        # PHASE 14.7 TASK 9: ALERTING ENGINE
        # ====================================================================

        alerts_generated = False
        alert_report = None
        alert_exit_code = 70  # EXIT_NO_ALERTS

        if args.run_alerts:
            logger.info("")
            logger.info("=" * 80)
            logger.info("ALERTING ENGINE (Phase 14.7 Task 9)")
            logger.info("=" * 80)
            logger.info("")

            try:
                from analytics.alerting_engine import (
                    AlertingEngine,
                    AlertingConfig,
                    ChannelConfig,
                    ChannelType,
                    AlertSeverity,
                    EXIT_NO_ALERTS,
                    EXIT_ALERTS_TRIGGERED,
                    EXIT_CRITICAL_ALERTS,
                )

                # Determine dashboard path
                dashboard_output_dir = args.dashboard_output_dir or (args.output_dir / "dashboard")
                current_dashboard_path = dashboard_output_dir / "health-dashboard.json"

                if not current_dashboard_path.exists():
                    logger.warning(f"Dashboard JSON not found at {current_dashboard_path}")
                    logger.warning("Alerting requires a health dashboard - run with --generate-dashboard first")
                    alerts_generated = False
                else:
                    # Determine alert output directory
                    alert_output_dir = args.alert_output_dir or (args.output_dir / "alerts")
                    alert_output_path = alert_output_dir / "alerts.json"

                    # Build channel configurations
                    channels = []
                    channel_names = [c.strip().lower() for c in args.alert_channels.split(',')]

                    if 'console' in channel_names:
                        channels.append(ChannelConfig(
                            channel_type=ChannelType.CONSOLE,
                            enabled=True
                        ))

                    if 'file' in channel_names:
                        channels.append(ChannelConfig(
                            channel_type=ChannelType.FILE,
                            enabled=True,
                            output_path=alert_output_dir / "alerts.txt"
                        ))

                    if 'email' in channel_names and args.alert_email_to:
                        channels.append(ChannelConfig(
                            channel_type=ChannelType.EMAIL,
                            enabled=True,
                            email_to=args.alert_email_to
                        ))
                    elif 'email' in channel_names:
                        logger.warning("Email channel requested but --alert-email-to not provided")

                    if 'webhook' in channel_names and args.alert_webhook_url:
                        channels.append(ChannelConfig(
                            channel_type=ChannelType.WEBHOOK,
                            enabled=True,
                            webhook_url=args.alert_webhook_url
                        ))
                    elif 'webhook' in channel_names:
                        logger.warning("Webhook channel requested but --alert-webhook-url not provided")

                    # Default to console if no channels configured
                    if not channels:
                        channels.append(ChannelConfig(
                            channel_type=ChannelType.CONSOLE,
                            enabled=True
                        ))

                    # Build alerting configuration
                    alerting_config = AlertingConfig(
                        current_dashboard_path=current_dashboard_path,
                        previous_dashboard_path=args.previous_dashboard,
                        output_path=alert_output_path,
                        channels=channels,
                        severity_threshold=AlertSeverity[args.alert_threshold],
                        fail_on_critical=args.alert_fail_on_critical and not args.alert_no_fail_on_critical,
                        fail_on_any_alert=args.alert_fail_on_any,
                        verbose=args.verbose
                    )

                    logger.info(f"Current Dashboard: {current_dashboard_path}")
                    if args.previous_dashboard:
                        logger.info(f"Previous Dashboard: {args.previous_dashboard}")
                    logger.info(f"Severity Threshold: {args.alert_threshold}")
                    logger.info(f"Channels: {args.alert_channels}")
                    logger.info(f"Output: {alert_output_path}")
                    logger.info("")

                    # Run alerting engine
                    logger.info("Evaluating alert rules...")
                    engine = AlertingEngine(alerting_config)
                    alert_report, alert_exit_code = engine.run()

                    alerts_generated = True

                    # Display summary
                    logger.info("")
                    logger.info("✓ ALERTING ENGINE COMPLETE")
                    logger.info(f"  Total Alerts: {alert_report.total_alerts}")
                    logger.info(f"    Critical: {alert_report.critical_alerts}")
                    logger.info(f"    Error: {alert_report.error_alerts}")
                    logger.info(f"    Warning: {alert_report.warning_alerts}")
                    logger.info(f"    Info: {alert_report.info_alerts}")
                    logger.info(f"  Rules Evaluated: {alert_report.rules_evaluated}")
                    logger.info(f"  Rules Triggered: {alert_report.rules_triggered}")
                    logger.info(f"  Channels Dispatched: {', '.join(alert_report.channels_dispatched) or 'None'}")
                    if alert_report.dispatch_errors:
                        logger.warning(f"  Dispatch Errors: {len(alert_report.dispatch_errors)}")
                    logger.info(f"  Duration: {alert_report.evaluation_duration_ms:.0f}ms")
                    logger.info("")

                    if alert_report.total_alerts > 0:
                        logger.info(f"Alert Report: {alert_output_path}")
                        if 'file' in channel_names:
                            logger.info(f"Alert Text: {alert_output_dir / 'alerts.txt'}")
                    logger.info("")

                    # Check if we should fail based on alerts
                    if alert_exit_code == EXIT_CRITICAL_ALERTS and alerting_config.fail_on_critical:
                        logger.error("CRITICAL ALERTS DETECTED - failing build")
                        logger.error("Use --alert-no-fail-on-critical to continue despite critical alerts")
                        return alert_exit_code
                    elif alert_exit_code == EXIT_ALERTS_TRIGGERED and alerting_config.fail_on_any_alert:
                        logger.warning("ALERTS TRIGGERED - failing build (--alert-fail-on-any specified)")
                        return alert_exit_code

            except ImportError as e:
                logger.error(f"Failed to import alerting engine module: {e}")
                logger.error("Ensure analytics.alerting_engine module is available")
                alerts_generated = False
                # Don't fail for alerting errors unless critical
                logger.warning("Continuing without alerting (non-critical)")

            except Exception as e:
                logger.error(f"Alerting engine error: {e}")
                if args.verbose:
                    import traceback
                    traceback.print_exc()
                alerts_generated = False
                logger.warning("Continuing without alerting (non-critical)")

        # ====================================================================
        # PHASE 14.7 TASK 10: TREND ANALYZER
        # ====================================================================

        trend_history_updated = False
        trend_analysis_completed = False
        trend_report = None

        if args.update_history or args.run_trends:
            logger.info("")
            logger.info("=" * 80)
            logger.info("TREND ANALYZER (Phase 14.7 Task 10)")
            logger.info("=" * 80)
            logger.info("")

            try:
                from analytics.trend_analyzer import (
                    TrendConfig,
                    TrendEngine,
                    HealthHistoryStore,
                    add_snapshot_to_history,
                    EXIT_TREND_SUCCESS,
                    EXIT_INSUFFICIENT_HISTORY,
                )

                # Determine paths
                dashboard_output_dir = args.dashboard_output_dir or (args.output_dir / "dashboard")
                current_dashboard_path = dashboard_output_dir / "health-dashboard.json"
                trend_history_dir = args.trend_history_dir or (args.output_dir / "dashboard-history")
                trend_output_path = args.trend_output or (args.output_dir / "trend-report.json")
                chart_output_dir = trend_history_dir / "charts" if args.trend_charts else None

                # Step 1: Update history (if requested)
                if args.update_history:
                    if not current_dashboard_path.exists():
                        logger.warning(f"Dashboard JSON not found at {current_dashboard_path}")
                        logger.warning("History update requires a dashboard - run with --generate-dashboard first")
                    else:
                        logger.info("Updating dashboard history...")
                        logger.info(f"  History Dir: {trend_history_dir}")
                        logger.info(f"  Dashboard: {current_dashboard_path}")

                        try:
                            snapshot_metadata = add_snapshot_to_history(
                                trend_history_dir,
                                current_dashboard_path,
                                version
                            )
                            logger.info(f"  Added snapshot: {snapshot_metadata.snapshot_id}")
                            logger.info(f"  Score: {snapshot_metadata.repository_score:.1f}")
                            logger.info(f"  Health: {snapshot_metadata.overall_health}")
                            logger.info("")
                            trend_history_updated = True
                        except Exception as e:
                            logger.error(f"Failed to add snapshot to history: {e}")
                            if args.verbose:
                                import traceback
                                traceback.print_exc()

                # Step 2: Run trend analysis (if requested)
                if args.run_trends:
                    logger.info("Running trend analysis...")
                    logger.info(f"  History Dir: {trend_history_dir}")
                    logger.info(f"  Output: {trend_output_path}")
                    logger.info(f"  Min Snapshots: {args.trend_min_snapshots}")
                    logger.info(f"  Prediction Horizon: {args.trend_prediction_horizon}")
                    if args.trend_charts:
                        logger.info(f"  Charts: {chart_output_dir}")
                    logger.info("")

                    # Build configuration
                    trend_config = TrendConfig(
                        history_dir=trend_history_dir,
                        output_path=trend_output_path,
                        min_snapshots=args.trend_min_snapshots,
                        prediction_horizon=args.trend_prediction_horizon,
                        generate_charts=args.trend_charts,
                        chart_output_dir=chart_output_dir,
                        verbose=args.verbose
                    )

                    # Run analysis
                    engine = TrendEngine(trend_config)
                    trend_report, trend_exit_code = engine.run()

                    if trend_exit_code == EXIT_TREND_SUCCESS:
                        trend_analysis_completed = True
                        logger.info("")
                        logger.info("✓ TREND ANALYSIS COMPLETE")
                        logger.info(f"  Snapshots Analyzed: {trend_report.snapshots_analyzed}")
                        logger.info(f"  Overall Trend: {trend_report.overall_trend.upper()}")
                        logger.info(f"  Current Score: {trend_report.current_score:.1f}")
                        logger.info(f"  Predicted Next: {trend_report.predicted_next_score:.1f}")
                        logger.info(f"  Confidence: [{trend_report.confidence_interval[0]:.1f}, {trend_report.confidence_interval[1]:.1f}]")
                        logger.info(f"  Anomalies: {trend_report.total_anomalies}")
                        logger.info(f"  Warnings: {trend_report.total_warnings}")
                        logger.info(f"  Degrading Versions: {len(trend_report.degrading_versions)}")
                        logger.info("")
                        logger.info(f"  Trend Report: {trend_output_path}")
                        if args.trend_charts and chart_output_dir:
                            logger.info(f"  Charts: {chart_output_dir}")
                        logger.info("")

                        # Display early warnings
                        if trend_report.early_warnings:
                            logger.info("Early Warnings:")
                            for warning in trend_report.early_warnings[:3]:
                                logger.info(f"  [{warning.level.upper()}] {warning.title}")
                                logger.info(f"    {warning.message}")
                            if len(trend_report.early_warnings) > 3:
                                logger.info(f"  ... and {len(trend_report.early_warnings) - 3} more")
                            logger.info("")

                    elif trend_exit_code == EXIT_INSUFFICIENT_HISTORY:
                        logger.warning("Not enough history for trend analysis")
                        logger.warning(f"Need at least {args.trend_min_snapshots} snapshots")
                        logger.warning("Run with --update-history to add more snapshots over time")
                        trend_analysis_completed = False
                    else:
                        logger.warning(f"Trend analysis returned exit code {trend_exit_code}")
                        trend_analysis_completed = False

            except ImportError as e:
                logger.error(f"Failed to import trend analyzer module: {e}")
                logger.error("Ensure analytics.trend_analyzer module is available")
                # Don't fail for trend errors
                logger.warning("Continuing without trend analysis (non-critical)")

            except Exception as e:
                logger.error(f"Trend analyzer error: {e}")
                if args.verbose:
                    import traceback
                    traceback.print_exc()
                logger.warning("Continuing without trend analysis (non-critical)")

        # Success summary
        logger.info("")
        logger.info("=" * 80)
        logger.info("RELEASE PREPARATION COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Version:    {version}")
        logger.info(f"Profile:    {args.profile}")
        logger.info(f"Artifacts:  {len(artifacts)}")
        logger.info(f"Output Dir: {args.output_dir}")
        logger.info("")
        logger.info("Enterprise Features:")
        logger.info(f"  Signed:        {'Yes' if signed else 'No'}")
        logger.info(f"  Encrypted:     {'Yes' if encrypted else 'No'}")
        logger.info(f"  SBOM:          {'Yes' if sbom_generated else 'No'}")
        logger.info(f"  SLSA:          {'Yes' if slsa_generated else 'No'}")
        logger.info("")
        logger.info("Validation:")
        logger.info(f"  Performance:       {'Yes' if perf_test_completed else 'No'}")
        logger.info(f"  Security:          {'Yes' if security_audit_completed else 'No'}")
        logger.info(f"  Verification:      {'Passed' if verification_passed else ('Skipped' if not args.verify_release else 'Failed')}")
        if args.post_release_validation:
            status = 'Passed' if post_validation_passed is True else ('Skipped' if post_validation_passed is None else 'Failed')
            logger.info(f"  Post-Validation:   {status}")
        if args.publish_release:
            pub_status = 'Published' if publication_successful else 'Failed'
            logger.info(f"  Publication:       {pub_status}")
        if args.rollback_release:
            rb_status = 'Success' if rollback_successful else 'Failed'
            if args.rollback_dry_run:
                rb_status = 'Dry Run'
            logger.info(f"  Rollback:          {rb_status}")
        if args.scan_repository:
            scan_status = 'Passed' if scan_successful else 'Failed'
            logger.info(f"  Integrity Scan:    {scan_status}")
            if scan_report:
                logger.info(f"    Issues: {scan_report.total_issues} ({scan_report.critical_issues} critical, {scan_report.error_issues} errors)")
        if args.generate_dashboard:
            dashboard_status = 'Generated' if dashboard_generated else 'Failed'
            logger.info(f"  Health Dashboard:  {dashboard_status}")
            if dashboard_generated and dashboard_health_status:
                logger.info(f"    Status: {dashboard_health_status.upper()}, Score: {dashboard_health_score:.1f}/100")
        if args.run_alerts:
            alerts_status = 'Generated' if alerts_generated else 'Failed'
            logger.info(f"  Alerting Engine:   {alerts_status}")
            if alerts_generated and alert_report:
                logger.info(f"    Alerts: {alert_report.total_alerts} ({alert_report.critical_alerts} critical, {alert_report.error_alerts} errors)")
        if args.update_history or args.run_trends:
            if args.update_history:
                history_status = 'Updated' if trend_history_updated else 'Failed'
                logger.info(f"  History Update:    {history_status}")
            if args.run_trends:
                trend_status = 'Completed' if trend_analysis_completed else ('Insufficient History' if not trend_analysis_completed else 'Failed')
                logger.info(f"  Trend Analysis:    {trend_status}")
                if trend_analysis_completed and trend_report:
                    logger.info(f"    Trend: {trend_report.overall_trend.upper()}, Score: {trend_report.current_score:.1f} -> {trend_report.predicted_next_score:.1f}")
        logger.info("")

        if args.dry_run:
            logger.info("[DRY RUN] No files were created")
        else:
            logger.info(f"Release artifacts ready in: {args.output_dir}")
            logger.info(f"Manifest: {manifest_path}")
            if args.publish_release and publication_successful:
                logger.info(f"Published to repository: {args.repository_type}")
                if publication_report:
                    logger.info(f"  Artifacts: {len(publication_report.artifacts_published)}")
                    logger.info(f"  Size: {publication_report.total_size_bytes:,} bytes")

        return 0

    except FileNotFoundError as e:
        logger.error(f"Missing artifact: {e}")
        return 1

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 3


if __name__ == '__main__':
    sys.exit(main())
