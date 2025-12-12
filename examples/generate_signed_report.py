#!/usr/bin/env python3
"""
T.A.R.S. Signed Report Generation Example

Demonstrates complete secure report generation workflow:
1. Load enterprise configuration
2. Run retrospective generator
3. Sign report with RSA-PSS
4. Encrypt report with AES-256 (optional)
5. Generate SBOM (CycloneDX)
6. Generate SLSA provenance
7. Verify signatures

This script is intended for auditors, DevOps engineers, and compliance teams
to generate tamper-proof, verifiable reports.

Usage:
    # Generate signed report
    python examples/generate_signed_report.py

    # Generate signed + encrypted report
    python examples/generate_signed_report.py --encrypt

    # Generate with custom period
    python examples/generate_signed_report.py --days 30

    # Generate with full SBOM + SLSA provenance
    python examples/generate_signed_report.py --full-provenance
"""

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from enterprise_config import load_enterprise_config
from security import SecurityManager, generate_sbom, generate_slsa_provenance
from compliance import ComplianceEnforcer
from metrics import get_logger

logger = get_logger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate signed and optionally encrypted T.A.R.S. retrospective report"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Number of days to analyze (default: 7)"
    )
    parser.add_argument(
        "--profile",
        type=str,
        default="prod",
        choices=["local", "dev", "staging", "prod"],
        help="Configuration profile (default: prod)"
    )
    parser.add_argument(
        "--encrypt",
        action="store_true",
        help="Encrypt the report with AES-256"
    )
    parser.add_argument(
        "--full-provenance",
        action="store_true",
        help="Generate SBOM and SLSA provenance"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports"),
        help="Output directory for reports (default: ./reports)"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify signatures after generation"
    )
    return parser.parse_args()


def generate_retrospective_data(days: int) -> dict:
    """
    Generate retrospective report data.

    In production, this would call the actual retrospective generator.
    For this example, we generate mock data.
    """
    logger.info(f"Generating retrospective report for last {days} days")

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    report = {
        "period": {
            "start": start_date.isoformat(),
            "end": end_date.isoformat(),
            "days": days
        },
        "overall_metrics": {
            "availability": 99.95,
            "total_requests": 875000,
            "error_rate": 0.03,
            "avg_latency_ms": 46.2
        },
        "highlights": [
            f"Achieved 99.95% availability over {days} days",
            "Handled 875K requests with 0.03% error rate",
            "Zero critical incidents",
            "All SLOs met or exceeded"
        ],
        "concerns": [
            "1 medium-severity regression detected in orchestration service",
            "Latency p99 increased by 15ms compared to previous period"
        ],
        "anomalies_summary": {
            "total": 5,
            "high_severity": 1,
            "medium_severity": 3,
            "low_severity": 1,
            "resolved": 4,
            "active": 1
        },
        "regressions_summary": {
            "total": 1,
            "mitigated": 0,
            "under_investigation": 1
        },
        "recommendations": [
            "Investigate orchestration service database connection pooling",
            "Consider scaling API gateway for peak traffic periods",
            "Review alerting thresholds for latency metrics",
            "Implement automated mitigation for common failure scenarios"
        ],
        "compliance_status": {
            "soc2_compliant": True,
            "iso27001_compliant": True,
            "gdpr_compliant": True,
            "total_violations": 0
        },
        "generated_at": datetime.now().isoformat(),
        "generated_by": "automated_report_generator",
        "report_version": "1.0.2"
    }

    logger.info("Retrospective report generated successfully")
    return report


def save_report(data: dict, output_path: Path, pretty: bool = True):
    """Save report to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        if pretty:
            json.dump(data, f, indent=2)
        else:
            json.dump(data, f)

    logger.info(f"Report saved to: {output_path}")


def main():
    """Main entry point."""
    args = parse_args()

    print("=" * 80)
    print("T.A.R.S. Signed Report Generation")
    print("=" * 80)
    print()

    # Step 1: Load enterprise configuration
    print(f"[1/7] Loading enterprise configuration (profile: {args.profile})...")
    try:
        config = load_enterprise_config(profile=args.profile)
        print(f"      ✓ Configuration loaded")
        print(f"        - Encryption: {config.security.encryption_enabled}")
        print(f"        - Signing: {config.security.signing_enabled}")
        print(f"        - Compliance: {', '.join(config.compliance.enabled_standards)}")
    except Exception as e:
        print(f"      ✗ Failed to load configuration: {e}")
        sys.exit(1)

    print()

    # Step 2: Initialize security manager
    print("[2/7] Initializing security manager...")
    try:
        security = SecurityManager(
            encryption_enabled=config.security.encryption_enabled,
            encryption_key_path=config.security.encryption_key_path,
            signing_enabled=config.security.signing_enabled,
            signing_key_path=config.security.signing_key_path,
            public_key_path=config.security.public_key_path
        )
        print(f"      ✓ Security manager initialized")
    except Exception as e:
        print(f"      ✗ Failed to initialize security manager: {e}")
        print(f"      Note: Ensure keys are generated and configured")
        sys.exit(1)

    print()

    # Step 3: Generate retrospective report
    print(f"[3/7] Generating retrospective report ({args.days} days)...")
    try:
        report_data = generate_retrospective_data(args.days)
        print(f"      ✓ Report generated")
        print(f"        - Availability: {report_data['overall_metrics']['availability']}%")
        print(f"        - Total requests: {report_data['overall_metrics']['total_requests']:,}")
        print(f"        - Anomalies: {report_data['anomalies_summary']['total']}")
        print(f"        - Regressions: {report_data['regressions_summary']['total']}")
    except Exception as e:
        print(f"      ✗ Failed to generate report: {e}")
        sys.exit(1)

    print()

    # Step 4: Sign the report
    print("[4/7] Signing report with RSA-PSS...")
    try:
        signed_report = security.sign_json(report_data)
        print(f"      ✓ Report signed successfully")
        print(f"        - Algorithm: RSA-PSS")
        print(f"        - Signature: {signed_report['signature'][:50]}...")
        print(f"        - Signed at: {signed_report['signed_at']}")
    except Exception as e:
        print(f"      ✗ Failed to sign report: {e}")
        sys.exit(1)

    print()

    # Step 5: Optionally encrypt the report
    encrypted_report = None
    if args.encrypt:
        print("[5/7] Encrypting report with AES-256...")
        try:
            # Serialize signed report to JSON
            report_json = json.dumps(signed_report)

            # Encrypt
            encrypted_data = security.encrypt_data(report_json)
            encrypted_report = {
                "encrypted_data": encrypted_data,
                "algorithm": "AES-256-GCM",
                "encrypted_at": datetime.now().isoformat()
            }
            print(f"      ✓ Report encrypted successfully")
            print(f"        - Algorithm: AES-256-GCM")
            print(f"        - Encrypted size: {len(encrypted_data)} bytes")
        except Exception as e:
            print(f"      ✗ Failed to encrypt report: {e}")
            sys.exit(1)
    else:
        print("[5/7] Skipping encryption (use --encrypt to enable)")

    print()

    # Step 6: Save reports
    print("[6/7] Saving reports...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    try:
        # Save signed report
        signed_path = args.output_dir / f"retrospective_signed_{timestamp}.json"
        save_report(signed_report, signed_path)
        print(f"      ✓ Signed report: {signed_path}")

        # Save encrypted report if requested
        if encrypted_report:
            encrypted_path = args.output_dir / f"retrospective_encrypted_{timestamp}.json"
            save_report(encrypted_report, encrypted_path)
            print(f"      ✓ Encrypted report: {encrypted_path}")

        # Save original data for reference
        original_path = args.output_dir / f"retrospective_original_{timestamp}.json"
        save_report(report_data, original_path)
        print(f"      ✓ Original report: {original_path}")

    except Exception as e:
        print(f"      ✗ Failed to save reports: {e}")
        sys.exit(1)

    print()

    # Step 7: Generate SBOM and SLSA provenance if requested
    if args.full_provenance:
        print("[7/7] Generating SBOM and SLSA provenance...")
        try:
            # Generate SBOM
            sbom_path = args.output_dir / f"sbom_{timestamp}.json"
            sbom = generate_sbom(
                format="cyclonedx",
                output_path=str(sbom_path)
            )
            print(f"      ✓ SBOM generated: {sbom_path}")

            # Generate SLSA provenance for signed report
            provenance_path = args.output_dir / f"provenance_{timestamp}.json"
            provenance = generate_slsa_provenance(
                artifact_path=str(signed_path),
                builder="T.A.R.S. Report Generator",
                build_type="automated_retrospective",
                output_path=str(provenance_path)
            )
            print(f"      ✓ SLSA provenance: {provenance_path}")

        except Exception as e:
            print(f"      ✗ Failed to generate provenance: {e}")
            # Don't exit, provenance is optional
    else:
        print("[7/7] Skipping SBOM/SLSA (use --full-provenance to enable)")

    print()

    # Optional: Verify signatures
    if args.verify:
        print("[VERIFY] Verifying signatures...")
        try:
            is_valid = security.verify_signature(signed_report)
            if is_valid:
                print(f"         ✓ Signature verification PASSED")
                print(f"           Report integrity confirmed")
            else:
                print(f"         ✗ Signature verification FAILED")
                print(f"           Report may be tampered")
                sys.exit(1)
        except Exception as e:
            print(f"         ✗ Verification error: {e}")
            sys.exit(1)

        print()

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print(f"Report period:       {args.days} days")
    print(f"Overall availability: {report_data['overall_metrics']['availability']}%")
    print(f"Total requests:      {report_data['overall_metrics']['total_requests']:,}")
    print(f"Anomalies detected:  {report_data['anomalies_summary']['total']}")
    print(f"Regressions:         {report_data['regressions_summary']['total']}")
    print()
    print("Generated files:")
    print(f"  - Signed report:     {signed_path}")
    if encrypted_report:
        print(f"  - Encrypted report:  {encrypted_path}")
    if args.full_provenance:
        print(f"  - SBOM:              {sbom_path}")
        print(f"  - SLSA provenance:   {provenance_path}")
    print()
    print("Security features:")
    print(f"  - RSA-PSS signature: ✓")
    print(f"  - AES-256 encryption: {'✓' if args.encrypt else '✗ (disabled)'}")
    print(f"  - SBOM generated:     {'✓' if args.full_provenance else '✗ (disabled)'}")
    print(f"  - SLSA provenance:    {'✓' if args.full_provenance else '✗ (disabled)'}")
    print()

    if args.verify:
        print("Verification: ✓ PASSED")
        print()

    print("✓ Report generation complete!")
    print()

    # Compliance note
    if config.compliance.enabled_standards:
        print("Compliance:")
        for standard in config.compliance.enabled_standards:
            status = report_data["compliance_status"].get(f"{standard}_compliant", False)
            print(f"  - {standard.upper()}: {'✓ COMPLIANT' if status else '✗ NON-COMPLIANT'}")
        print()

    print("=" * 80)


if __name__ == "__main__":
    main()
