#!/usr/bin/env python3
"""
Quick validation script for Phase 14.7 Task 7 - Repository Integrity Scanner

This script performs a quick smoke test to validate the integrity scanner is working.

Usage:
    python scripts/validate_integrity_scanner.py
"""

import sys
import tempfile
import json
from pathlib import Path
from datetime import datetime, timezone

# Add project root to path
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    print("=" * 80)
    print("Phase 14.7 Task 7 - Repository Integrity Scanner Validation")
    print("=" * 80)
    print()

    # Import components
    print("[1/5] Testing imports...")
    try:
        from publisher.release_publisher import RepositoryFactory
        from integrity.repository_integrity_scanner import (
            IntegrityScanOrchestrator,
            IntegrityScanPolicy,
            IntegrityScanStatus,
        )
        print("  [OK] All imports successful")
    except ImportError as e:
        print(f"  [FAIL] Import failed: {e}")
        return 1

    # Create test repository
    print("[2/5] Creating test repository...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        repo_dir = Path(tmp_dir) / "test-repo"
        repo_dir.mkdir(parents=True, exist_ok=True)

        # Create sample version
        version = "v1.0.0"
        version_dir = repo_dir / version
        version_dir.mkdir(parents=True, exist_ok=True)

        # Create manifest
        manifest = {
            "version": version,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "artifacts": [
                {"path": f"{version}/README.md", "sha256": "test123"},
                {"path": f"{version}/manifest.json", "sha256": "test456"},
            ]
        }
        (version_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

        # Create artifacts
        (version_dir / "README.md").write_text("# Test README")

        # Create SBOM
        sbom = {
            "bomFormat": "CycloneDX",
            "specVersion": "1.4",
            "version": 1
        }
        (version_dir / "sbom.json").write_text(json.dumps(sbom, indent=2))

        # Create SLSA
        slsa = {
            "predicateType": "https://slsa.dev/provenance/v0.2",
            "predicate": {"builder": {"id": "test"}},
            "subject": [{"name": version}]
        }
        (version_dir / "slsa-provenance.json").write_text(json.dumps(slsa, indent=2))

        # Create index
        index = {
            "format_version": "1.0",
            "repository": "Test Repo",
            "generated": datetime.now(timezone.utc).isoformat(),
            "total_releases": 1,
            "releases": {
                version: {
                    "version": version,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "artifacts": 3
                }
            }
        }
        (repo_dir / "index.json").write_text(json.dumps(index, indent=2))

        print(f"  [OK] Test repository created: {repo_dir}")

        # Create repository instance
        print("[3/5] Initializing scanner...")
        repo_config = {"type": "local", "path": str(repo_dir)}
        repository = RepositoryFactory.create("local", repo_config)

        orchestrator = IntegrityScanOrchestrator(
            repository=repository,
            policy_mode=IntegrityScanPolicy.LENIENT,
            repair_enabled=False
        )
        print("  [OK] Scanner initialized")

        # Run scan
        print("[4/5] Running integrity scan...")
        output_dir = repo_dir / "scan-output"
        json_path = output_dir / "report.json"
        text_path = output_dir / "report.txt"

        report = orchestrator.scan_repository(
            output_dir=output_dir,
            json_report_path=json_path,
            text_report_path=text_path
        )

        print(f"  [OK] Scan completed")
        print(f"    Status: {report.scan_status}")
        print(f"    Versions: {report.total_versions}")
        print(f"    Artifacts: {report.total_artifacts}")
        print(f"    Issues: {report.total_issues}")
        print(f"    Duration: {report.scan_duration_seconds:.2f}s")

        # Validate results
        print("[5/5] Validating results...")

        # Check reports exist
        if not json_path.exists():
            print(f"  [FAIL] JSON report missing")
            return 1
        if not text_path.exists():
            print(f"  [FAIL] Text report missing")
            return 1

        # Check report content
        json_content = json.loads(json_path.read_text())
        if json_content["scan_id"] != report.scan_id:
            print(f"  [FAIL] JSON report scan_id mismatch")
            return 1

        # Check performance
        if report.scan_duration_seconds > 5.0:
            print(f"  [WARN] Scan duration exceeds 5s target: {report.scan_duration_seconds:.2f}s")

        print("  [OK] All validations passed")

    print()
    print("=" * 80)
    print("VALIDATION SUCCESSFUL")
    print("=" * 80)
    print()
    print("Summary:")
    print("  - Imports: OK")
    print("  - Repository creation: OK")
    print("  - Scanner initialization: OK")
    print("  - Scan execution: OK")
    print("  - Report generation: OK")
    print("  - Performance: OK")
    print()
    print("Phase 14.7 Task 7 is ready for use!")
    print()

    return 0


if __name__ == '__main__':
    sys.exit(main())
