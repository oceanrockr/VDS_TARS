#!/usr/bin/env python3
"""
T.A.R.S. Operations CLI - Golden Path Wrapper

Phase 19 - Production Ops Maturity & CI Hardening

Provides a single entry-point for common operational workflows:
    - daily:    Quick health check (flat output, minimal processing)
    - weekly:   Trend analysis (structured output, executive bundle, retention summary)
    - incident: Incident response mode (structured + narrative + signing if configured)

This wrapper delegates to the underlying orchestrator and packager scripts,
respecting the same config precedence and CLI override rules.

Usage:
    # Daily health check (quick)
    python scripts/tars_ops.py daily

    # Weekly trend report with executive bundle
    python scripts/tars_ops.py weekly

    # Incident response mode with full artifacts
    python scripts/tars_ops.py incident --incident-id INC-12345

    # All commands support config override
    python scripts/tars_ops.py daily --config ./tars.prod.yml

Exit Codes:
    Returns the exit code from the underlying orchestrator.
    See docs/OPS_RUNBOOK.md for exit code guidance.

Version: 1.0.0
Phase: 19 - Production Ops Maturity
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Exit code to guidance mapping
EXIT_CODE_GUIDANCE: Dict[int, Tuple[str, str, bool]] = {
    # (status, action, escalate)
    0: ("SUCCESS", "All health checks passed. No action required.", False),
    1: ("PIPELINE ERROR", "One or more pipeline steps failed. Check logs.", False),
    92: ("HIGH ORG RISK", "Review org health report immediately.", True),
    102: ("CRITICAL ALERTS", "Follow Incident Playbook SEV-1 procedure.", True),
    122: ("CRITICAL ANOMALY", "Investigate correlation clusters.", True),
    132: ("PROPAGATION RISK", "Isolate leader repos, freeze deployments.", True),
    140: ("SLA SUCCESS", "All SLAs compliant. No action required.", False),
    141: ("AT-RISK SLAs", "Increase monitoring frequency.", False),
    142: ("SLA BREACH", "Initiate incident response. Notify stakeholders.", True),
    199: ("GENERAL ERROR", "Check logs, verify configuration.", False),
}


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.resolve()


def print_banner(mode: str) -> None:
    """Print operation mode banner."""
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print()
    print("=" * 60)
    print(f"T.A.R.S. Operations - {mode.upper()} MODE")
    print(f"Started: {timestamp}")
    print("=" * 60)
    print()


def print_guidance(exit_code: int) -> None:
    """Print next action guidance based on exit code."""
    print()
    print("-" * 60)
    print("NEXT ACTION GUIDANCE")
    print("-" * 60)

    if exit_code in EXIT_CODE_GUIDANCE:
        status, action, escalate = EXIT_CODE_GUIDANCE[exit_code]
        print(f"Exit Code: {exit_code}")
        print(f"Status:    {status}")
        print(f"Action:    {action}")
        if escalate:
            print(f"Escalate:  YES - Follow incident playbook immediately")
    else:
        print(f"Exit Code: {exit_code}")
        print(f"Status:    UNKNOWN")
        print(f"Action:    Review pipeline logs for details")

    print("-" * 60)
    print()


def build_orchestrator_command(
    config: Optional[str],
    root: Optional[str],
    output_format: str,
    print_paths: bool,
    with_narrative: bool,
    fail_on_breach: bool,
    additional_args: Optional[List[str]] = None
) -> List[str]:
    """Build the orchestrator command."""
    cmd = [sys.executable, "scripts/run_full_org_governance_pipeline.py"]

    if config:
        cmd.extend(["--config", config])

    if root:
        cmd.extend(["--root", root])

    cmd.extend(["--format", output_format])

    if print_paths:
        cmd.append("--print-paths")

    if with_narrative:
        cmd.append("--with-narrative")

    if fail_on_breach:
        cmd.append("--fail-on-breach")

    if additional_args:
        cmd.extend(additional_args)

    return cmd


def build_packager_command(
    config: Optional[str],
    run_dir: str,
    include_tar: bool,
    sign: bool
) -> List[str]:
    """Build the packager command."""
    cmd = [sys.executable, "scripts/package_executive_bundle.py"]

    if config:
        cmd.extend(["--config", config])

    cmd.extend(["--run-dir", run_dir])

    if include_tar:
        cmd.append("--tar")

    if sign:
        cmd.append("--sign")

    return cmd


def get_latest_run_dir() -> Optional[str]:
    """Get the most recent run directory."""
    runs_dir = get_project_root() / "reports" / "runs"
    if not runs_dir.exists():
        return None

    run_dirs = sorted(
        [d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith("tars-run-")],
        key=lambda x: x.stat().st_mtime,
        reverse=True
    )

    return str(run_dirs[0]) if run_dirs else None


def run_daily(args: argparse.Namespace) -> int:
    """Run daily health check."""
    print_banner("daily")

    os.chdir(get_project_root())

    cmd = build_orchestrator_command(
        config=args.config,
        root=args.root,
        output_format="flat",  # Fast, simple output for daily
        print_paths=True,
        with_narrative=False,  # Skip narrative for speed
        fail_on_breach=args.fail_on_breach
    )

    print(f"Command: {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd)
    exit_code = result.returncode

    # Optionally package if requested
    if args.package and exit_code in [0, 140, 141]:
        run_dir = get_latest_run_dir()
        if run_dir:
            print()
            print("Packaging executive bundle...")
            pkg_cmd = build_packager_command(
                config=args.config,
                run_dir=run_dir,
                include_tar=False,  # ZIP only for daily
                sign=False
            )
            subprocess.run(pkg_cmd)

    print_guidance(exit_code)
    return exit_code


def run_weekly(args: argparse.Namespace) -> int:
    """Run weekly trend analysis with executive bundle."""
    print_banner("weekly")

    os.chdir(get_project_root())

    cmd = build_orchestrator_command(
        config=args.config,
        root=args.root,
        output_format="structured",  # Full structured output for weekly
        print_paths=True,
        with_narrative=True,  # Include executive narrative
        fail_on_breach=args.fail_on_breach
    )

    print(f"Command: {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd)
    exit_code = result.returncode

    # Package executive bundle
    run_dir = get_latest_run_dir()
    if run_dir:
        print()
        print("Packaging executive bundle (ZIP + tar.gz)...")
        pkg_cmd = build_packager_command(
            config=args.config,
            run_dir=run_dir,
            include_tar=True,  # Full bundle with tar
            sign=False
        )
        subprocess.run(pkg_cmd)

    # Show retention dry-run summary
    print()
    print("Retention summary (dry-run):")
    retention_cmd = [
        sys.executable, "scripts/retention_manage.py",
        "--root", str(get_project_root() / "reports" / "runs"),
        "--days-hot", "30",
        "--days-warm", "90",
        "--summary-only"
    ]
    subprocess.run(retention_cmd, capture_output=False)

    print_guidance(exit_code)
    return exit_code


def run_incident(args: argparse.Namespace) -> int:
    """Run incident response mode with full artifacts."""
    print_banner("incident")

    incident_id = args.incident_id or f"INC-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
    print(f"Incident ID: {incident_id}")
    print()

    os.chdir(get_project_root())

    # Use strict SLA policy if available
    sla_policy = None
    strict_policy = get_project_root() / "policies" / "examples" / "internal_platform_strict.yaml"
    if strict_policy.exists():
        sla_policy = str(strict_policy)
        print(f"Using strict SLA policy: {sla_policy}")

    additional_args = []
    if sla_policy:
        additional_args.extend(["--sla-policy", sla_policy])

    cmd = build_orchestrator_command(
        config=args.config,
        root=args.root,
        output_format="structured",  # Full structured for incident
        print_paths=True,
        with_narrative=True,  # Include executive narrative
        fail_on_breach=False,  # Don't fail - we need all data
        additional_args=additional_args
    )

    print(f"Command: {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd)
    exit_code = result.returncode

    # Package with integrity (signing if configured)
    run_dir = get_latest_run_dir()
    if run_dir:
        print()
        print("Packaging incident evidence bundle (with integrity verification)...")
        pkg_cmd = build_packager_command(
            config=args.config,
            run_dir=run_dir,
            include_tar=True,
            sign=args.sign  # Sign if requested
        )
        subprocess.run(pkg_cmd)

    # Save incident ID to run directory
    if run_dir:
        incident_file = Path(run_dir) / "incident-id.txt"
        incident_file.write_text(f"Incident ID: {incident_id}\n")
        print(f"Incident ID saved to: {incident_file}")

    print_guidance(exit_code)

    # Additional incident-specific guidance
    if exit_code in [102, 132, 142]:
        print()
        print("=" * 60)
        print("INCIDENT RESPONSE REQUIRED")
        print("=" * 60)
        print()
        print("1. Review the executive narrative in the run directory")
        print("2. Check SLA breach details in sla-intelligence-report.json")
        print("3. Follow docs/INCIDENT_PLAYBOOK.md for SEV-1 procedures")
        print("4. Escalate to on-call and leadership as required")
        print()

    return exit_code


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        prog="tars_ops",
        description="T.A.R.S. Operations CLI - Golden Path Wrapper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  daily     Quick health check (flat output, minimal processing)
  weekly    Trend analysis (structured output, executive bundle)
  incident  Incident response (structured + narrative + signing)

Examples:
  # Quick daily check
  python scripts/tars_ops.py daily

  # Weekly with custom config
  python scripts/tars_ops.py weekly --config ./tars.prod.yml

  # Incident response
  python scripts/tars_ops.py incident --incident-id INC-12345

See docs/OPS_RUNBOOK.md for complete operational guidance.
"""
    )

    # Global options
    parser.add_argument(
        "--config",
        default=None,
        help="Path to T.A.R.S. config file (YAML or JSON)"
    )
    parser.add_argument(
        "--root",
        default=None,
        help="Root directory for org health data"
    )

    subparsers = parser.add_subparsers(dest="command", help="Operation mode")

    # Daily subcommand
    daily_parser = subparsers.add_parser("daily", help="Run daily health check")
    daily_parser.add_argument(
        "--fail-on-breach",
        action="store_true",
        help="Exit with code 142 if SLA breach detected"
    )
    daily_parser.add_argument(
        "--package",
        action="store_true",
        help="Package executive bundle after successful run"
    )

    # Weekly subcommand
    weekly_parser = subparsers.add_parser("weekly", help="Run weekly trend analysis")
    weekly_parser.add_argument(
        "--fail-on-breach",
        action="store_true",
        help="Exit with code 142 if SLA breach detected"
    )

    # Incident subcommand
    incident_parser = subparsers.add_parser("incident", help="Run incident response mode")
    incident_parser.add_argument(
        "--incident-id",
        default=None,
        help="Incident ID for tracking (e.g., INC-12345)"
    )
    incident_parser.add_argument(
        "--sign",
        action="store_true",
        help="GPG sign the evidence bundle (requires GPG configured)"
    )

    return parser


def main() -> int:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    # Check TARS_CONFIG environment variable if no config specified
    if not args.config and os.environ.get("TARS_CONFIG"):
        args.config = os.environ["TARS_CONFIG"]

    if args.command == "daily":
        return run_daily(args)
    elif args.command == "weekly":
        return run_weekly(args)
    elif args.command == "incident":
        return run_incident(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
