#!/usr/bin/env python3
"""
Full Organization Governance Pipeline Orchestrator

Runs the complete T.A.R.S. organization health governance pipeline and
generates a consolidated executive bundle with all reports.

Pipeline Steps:
    1. Org Health Report (Task 1 engine)
    2. Org Alerts (Task 2 engine) - if available
    3. Trend Correlation (Task 3 engine) - if available
    4. Temporal Intelligence (Task 4 engine) - if available
    5. SLA Intelligence (Task 5 engine)

Usage:
    # Basic run with org health directory
    python scripts/run_full_org_governance_pipeline.py --root ./org-health

    # With custom output directory and timestamp
    python scripts/run_full_org_governance_pipeline.py --root ./org-health --outdir ./reports/runs --timestamp 20251222-140000

    # With structured output format (daily/weekly/executive subdirs)
    python scripts/run_full_org_governance_pipeline.py --root ./org-health --format structured

    # Print artifact paths for CI/CD logging
    python scripts/run_full_org_governance_pipeline.py --root ./org-health --print-paths

    # Dry run mode (shows commands without executing)
    python scripts/run_full_org_governance_pipeline.py --root ./org-health --dry-run

    # CI/CD mode with failure on breach
    python scripts/run_full_org_governance_pipeline.py --root ./org-health --fail-on-breach

Exit Codes:
    0:   Success, all steps completed
    1:   Pipeline error (one or more steps failed)
    140: Success, but SLA at-risk detected
    142: Success, but SLA breach detected
    199: General orchestrator error

Version: 2.1.0
Phase: 17 - Post-GA Observability
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple

# Import metadata generator integration function
try:
    from scripts.generate_run_metadata import integrate_with_orchestrator as generate_metadata
    METADATA_AVAILABLE = True
except ImportError:
    METADATA_AVAILABLE = False
    generate_metadata = None

# Exit codes
EXIT_SUCCESS = 0
EXIT_PIPELINE_ERROR = 1
EXIT_SLA_AT_RISK = 140
EXIT_SLA_BREACH = 142
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


class PipelineStep:
    """Represents a single step in the pipeline."""

    def __init__(
        self,
        name: str,
        module: str,
        required_inputs: List[str],
        output_file: str,
        optional: bool = False
    ):
        self.name = name
        self.module = module
        self.required_inputs = required_inputs
        self.output_file = output_file
        self.optional = optional
        self.exit_code: Optional[int] = None
        self.executed = False
        self.skipped = False
        self.error_message: Optional[str] = None


class PipelineOrchestrator:
    """Orchestrates the full organization governance pipeline."""

    def __init__(
        self,
        root_dir: str,
        output_dir: str,
        timestamp: str,
        output_format: str = "flat",
        sla_policy: Optional[str] = None,
        windows: Optional[List[int]] = None,
        fail_on_breach: bool = False,
        fail_on_critical: bool = False,
        json_output: bool = False,
        summary_only: bool = False,
        dry_run: bool = False,
        print_paths: bool = False
    ):
        self.root_dir = Path(root_dir).resolve()
        self.timestamp = timestamp
        self.output_format = output_format
        self.sla_policy = sla_policy
        self.windows = windows or []
        self.fail_on_breach = fail_on_breach
        self.fail_on_critical = fail_on_critical
        self.json_output = json_output
        self.summary_only = summary_only
        self.dry_run = dry_run
        self.print_paths = print_paths
        self.with_narrative = False  # Will be set by args
        self.cli_flags_str: Optional[str] = None  # For metadata provenance

        # Build output directory with timestamp subdirectory
        base_output = Path(output_dir).resolve()
        self.output_dir = base_output / f"tars-run-{timestamp}"

        # Pipeline state
        self.steps: List[PipelineStep] = []
        self.outputs: Dict[str, Path] = {}
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None

        # Check available modules
        self.available_modules = self._check_available_modules()

        # Initialize pipeline steps
        self._initialize_steps()

    def _check_available_modules(self) -> Dict[str, bool]:
        """Check which analytics modules are available."""
        modules = {
            "org_health": False,
            "org_alerts": False,
            "org_trend_correlation": False,
            "org_temporal_intelligence": False,
            "org_sla_intelligence": False
        }

        try:
            from analytics import org_health_aggregator
            modules["org_health"] = True
        except ImportError:
            pass

        try:
            from analytics import org_alerting_engine
            modules["org_alerts"] = True
        except ImportError:
            pass

        try:
            from analytics import org_trend_correlation
            modules["org_trend_correlation"] = True
        except ImportError:
            pass

        try:
            from analytics import org_temporal_intelligence
            modules["org_temporal_intelligence"] = True
        except ImportError:
            pass

        try:
            from analytics import org_sla_intelligence
            modules["org_sla_intelligence"] = True
        except ImportError:
            pass

        return modules

    def _initialize_steps(self) -> None:
        """Initialize pipeline steps based on available modules."""
        # Step 1: Org Health (required)
        self.steps.append(PipelineStep(
            name="Org Health Report",
            module="analytics.run_org_health",
            required_inputs=[],
            output_file="org-health-report.json",
            optional=False
        ))

        # Step 2: Org Alerts (optional)
        self.steps.append(PipelineStep(
            name="Org Alerts",
            module="analytics.run_org_alerts",
            required_inputs=["org-health-report.json"],
            output_file="org-alerts.json",
            optional=True
        ))

        # Step 3: Trend Correlation (optional)
        self.steps.append(PipelineStep(
            name="Trend Correlation",
            module="analytics.run_org_trend_correlation",
            required_inputs=["org-health-report.json"],
            output_file="trend-correlation-report.json",
            optional=True
        ))

        # Step 4: Temporal Intelligence (optional)
        self.steps.append(PipelineStep(
            name="Temporal Intelligence",
            module="analytics.run_org_temporal_intelligence",
            required_inputs=["org-health-report.json"],
            output_file="temporal-intelligence-report.json",
            optional=True
        ))

        # Step 5: SLA Intelligence (required)
        self.steps.append(PipelineStep(
            name="SLA Intelligence",
            module="analytics.run_org_sla_intelligence",
            required_inputs=["org-health-report.json"],
            output_file="sla-intelligence-report.json",
            optional=False
        ))

    def _get_output_path(self, filename: str) -> Path:
        """Get the output path for a file based on output format."""
        if self.output_format == "structured":
            # Organize files into subdirectories
            if filename in ["org-health-report.json", "org-alerts.json"]:
                subdir = "daily"
            elif filename in ["trend-correlation-report.json", "temporal-intelligence-report.json"]:
                subdir = "weekly"
            else:
                subdir = "executive"
            return self.output_dir / subdir / filename
        else:
            # Flat structure - all files in one directory
            return self.output_dir / filename

    def _build_command(self, step: PipelineStep) -> List[str]:
        """Build the command for a pipeline step."""
        cmd = [sys.executable, "-m", step.module]

        # Add inputs based on step
        if step.name == "Org Health Report":
            cmd.extend(["--root-dir", str(self.root_dir)])
        elif step.name == "Org Alerts":
            cmd.extend(["--org-report", str(self.outputs["org-health-report.json"])])
        elif step.name == "Trend Correlation":
            cmd.extend(["--org-report", str(self.outputs["org-health-report.json"])])
        elif step.name == "Temporal Intelligence":
            cmd.extend(["--org-report", str(self.outputs["org-health-report.json"])])
            if "trend-correlation-report.json" in self.outputs:
                cmd.extend(["--trend-correlation-report", str(self.outputs["trend-correlation-report.json"])])
        elif step.name == "SLA Intelligence":
            cmd.extend(["--org-report", str(self.outputs["org-health-report.json"])])
            if "org-alerts.json" in self.outputs:
                cmd.extend(["--alerts-report", str(self.outputs["org-alerts.json"])])
            if "trend-correlation-report.json" in self.outputs:
                cmd.extend(["--trend-correlation-report", str(self.outputs["trend-correlation-report.json"])])
            if "temporal-intelligence-report.json" in self.outputs:
                cmd.extend(["--temporal-intelligence-report", str(self.outputs["temporal-intelligence-report.json"])])
            if self.sla_policy:
                cmd.extend(["--sla-policy", self.sla_policy])
            for window in self.windows:
                cmd.extend(["--window", str(window)])

        # Add output path
        output_path = self._get_output_path(step.output_file)
        cmd.extend(["--output", str(output_path)])

        # Add common flags
        if self.json_output:
            cmd.append("--json")

        return cmd

    def _run_step(self, step: PipelineStep) -> bool:
        """Run a single pipeline step."""
        # Check if module is available
        module_key = step.module.replace("analytics.run_", "")
        if not self.available_modules.get(module_key, False):
            if step.optional:
                step.skipped = True
                logger.info(f"Skipping {step.name}: module not available")
                return True
            else:
                step.error_message = f"Required module {step.module} not available"
                logger.error(step.error_message)
                return False

        # Check required inputs
        for input_file in step.required_inputs:
            if input_file not in self.outputs:
                if step.optional:
                    step.skipped = True
                    logger.info(f"Skipping {step.name}: missing input {input_file}")
                    return True
                else:
                    step.error_message = f"Missing required input: {input_file}"
                    logger.error(step.error_message)
                    return False

        # Build command
        cmd = self._build_command(step)

        # Dry run mode
        if self.dry_run:
            logger.info(f"[DRY RUN] Would execute: {' '.join(cmd)}")
            step.executed = True
            step.exit_code = 0
            self.outputs[step.output_file] = self._get_output_path(step.output_file)
            return True

        # Ensure output directory exists
        output_path = self._get_output_path(step.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Execute command
        logger.info(f"Running: {step.name}")
        logger.debug(f"Command: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )

            step.exit_code = result.returncode
            step.executed = True

            # Log output
            if result.stdout:
                logger.debug(f"stdout: {result.stdout[:500]}")
            if result.stderr:
                logger.debug(f"stderr: {result.stderr[:500]}")

            # Check for success
            if output_path.exists():
                self.outputs[step.output_file] = output_path
                logger.info(f"Completed: {step.name} (exit code: {step.exit_code})")
                return True
            elif step.optional:
                step.skipped = True
                logger.info(f"Skipping {step.name}: output not generated")
                return True
            else:
                step.error_message = f"Output file not generated: {output_path}"
                logger.error(step.error_message)
                return False

        except subprocess.TimeoutExpired:
            step.error_message = "Execution timed out"
            logger.error(f"{step.name}: {step.error_message}")
            return False
        except Exception as e:
            step.error_message = str(e)
            logger.error(f"{step.name} failed: {e}")
            return False

    def _generate_executive_summary(self) -> str:
        """Generate executive summary markdown."""
        lines = [
            "# T.A.R.S. Organization Governance Report",
            "",
            f"**Generated:** {datetime.now(timezone.utc).isoformat()}",
            f"**Pipeline Version:** 2.0.0",
            f"**Run Timestamp:** {self.timestamp}",
            "",
            "---",
            "",
            "## Executive Summary",
            ""
        ]

        # Read SLA intelligence report if available
        if "sla-intelligence-report.json" in self.outputs:
            try:
                with open(self.outputs["sla-intelligence-report.json"], "r") as f:
                    sla_report = json.load(f)

                # Extract key metrics
                if "executive_readiness" in sla_report:
                    readiness = sla_report["executive_readiness"]
                    lines.append(f"**Executive Readiness Score:** {readiness.get('readiness_score', 'N/A')}/100")
                    lines.append(f"**Readiness Tier:** {readiness.get('tier', 'N/A')}")
                    lines.append("")

                if "summary" in sla_report:
                    summary = sla_report["summary"]
                    lines.append(f"- **Total SLAs Evaluated:** {summary.get('total_slas', 0)}")
                    lines.append(f"- **Compliant:** {summary.get('compliant_count', 0)}")
                    lines.append(f"- **At Risk:** {summary.get('at_risk_count', 0)}")
                    lines.append(f"- **Breached:** {summary.get('breach_count', 0)}")
                    lines.append("")

            except Exception as e:
                lines.append(f"*Unable to parse SLA report: {e}*")
                lines.append("")

        # Pipeline execution summary
        lines.extend([
            "## Pipeline Execution",
            "",
            "| Step | Status | Exit Code |",
            "|------|--------|-----------|"
        ])

        for step in self.steps:
            if step.skipped:
                status = "Skipped"
                code = "-"
            elif step.executed:
                status = "Completed" if step.exit_code is not None else "Unknown"
                code = str(step.exit_code) if step.exit_code is not None else "-"
            else:
                status = "Not Run"
                code = "-"

            lines.append(f"| {step.name} | {status} | {code} |")

        lines.extend([
            "",
            "## Generated Reports",
            ""
        ])

        for filename, path in self.outputs.items():
            lines.append(f"- [{filename}](./{filename})")

        lines.extend([
            "",
            "---",
            "",
            f"*Generated by T.A.R.S. Pipeline Orchestrator v2.0.0*"
        ])

        return "\n".join(lines)

    def _generate_run_metadata(self) -> bool:
        """Generate run metadata and provenance artifact."""
        if not METADATA_AVAILABLE:
            logger.debug("Metadata generator not available, skipping")
            return False

        if self.dry_run:
            logger.info("[DRY RUN] Would generate run-metadata.json")
            return True

        try:
            # Build CLI flags string for provenance
            cli_flags = self.cli_flags_str or ""

            # Call metadata generator integration function
            success = generate_metadata(self.output_dir, cli_flags)
            if success:
                self.outputs["run-metadata.json"] = self.output_dir / "run-metadata.json"
                logger.info("Generated: run-metadata.json")
            else:
                logger.warning("Metadata generation failed (non-fatal)")
            return success
        except Exception as e:
            logger.warning(f"Metadata generation error (non-fatal): {e}")
            return False

    def _generate_executive_narrative(self) -> bool:
        """Generate executive narrative markdown."""
        if self.dry_run:
            logger.info("[DRY RUN] Would generate executive-narrative.md")
            return True

        try:
            # Import narrative generator
            from scripts.generate_executive_narrative import ExecutiveNarrativeGenerator

            generator = ExecutiveNarrativeGenerator(
                run_dir=str(self.output_dir)
            )
            result = generator.generate()
            if result == 0:
                self.outputs["executive-narrative.md"] = self.output_dir / "executive-narrative.md"
                logger.info("Generated: executive-narrative.md")
                return True
            else:
                logger.warning("Narrative generation returned non-zero (non-fatal)")
                return False
        except ImportError:
            logger.debug("Executive narrative generator not available, skipping")
            return False
        except Exception as e:
            logger.warning(f"Narrative generation error (non-fatal): {e}")
            return False

    def _generate_bundle_manifest(self) -> Dict[str, Any]:
        """Generate bundle manifest."""
        # Try to read version from VERSION file
        version = "unknown"
        version_file = Path(__file__).parent.parent / "VERSION"
        if version_file.exists():
            try:
                version = version_file.read_text().strip()
            except Exception:
                pass

        # Try to get git commit hash
        git_commit = None
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
                cwd=Path(__file__).parent.parent
            )
            if result.returncode == 0:
                git_commit = result.stdout.strip()
        except Exception:
            pass

        return {
            "manifest_version": "2.1",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "generator": "T.A.R.S. Pipeline Orchestrator",
            "orchestrator_version": "2.1.0",
            "tars_version": version,
            "git_commit": git_commit,
            "timestamp": self.timestamp,
            "phase": "17",
            "pipeline": {
                "root_dir": str(self.root_dir),
                "output_dir": str(self.output_dir),
                "output_format": self.output_format,
                "sla_policy": self.sla_policy,
                "windows": self.windows,
                "dry_run": self.dry_run,
                "start_time": self.start_time.isoformat() if self.start_time else None,
                "end_time": self.end_time.isoformat() if self.end_time else None,
                "duration_seconds": (
                    (self.end_time - self.start_time).total_seconds()
                    if self.start_time and self.end_time else None
                )
            },
            "available_modules": self.available_modules,
            "steps": [
                {
                    "name": step.name,
                    "module": step.module,
                    "executed": step.executed,
                    "skipped": step.skipped,
                    "exit_code": step.exit_code,
                    "output_file": step.output_file if step.output_file in self.outputs else None,
                    "error": step.error_message
                }
                for step in self.steps
            ],
            "outputs": {
                filename: str(path)
                for filename, path in self.outputs.items()
            }
        }

    def _print_artifact_paths(self) -> None:
        """Print artifact paths for CI/CD logging."""
        print("\n" + "=" * 60)
        print("ARTIFACT PATHS")
        print("=" * 60)
        print(f"Run Directory: {self.output_dir}")
        print(f"Timestamp: {self.timestamp}")
        print("")
        print("Generated Files:")
        for filename, path in sorted(self.outputs.items()):
            print(f"  {filename}: {path}")
        print("=" * 60 + "\n")

    def run(self) -> int:
        """Run the full pipeline."""
        self.start_time = datetime.now(timezone.utc)

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories for structured format
        if self.output_format == "structured":
            (self.output_dir / "daily").mkdir(exist_ok=True)
            (self.output_dir / "weekly").mkdir(exist_ok=True)
            (self.output_dir / "executive").mkdir(exist_ok=True)

        logger.info("=" * 60)
        logger.info("T.A.R.S. Organization Governance Pipeline v2.0")
        logger.info("=" * 60)
        logger.info(f"Root Directory: {self.root_dir}")
        logger.info(f"Output Directory: {self.output_dir}")
        logger.info(f"Timestamp: {self.timestamp}")
        logger.info(f"Output Format: {self.output_format}")
        if self.dry_run:
            logger.info("Mode: DRY RUN (no commands executed)")
        logger.info("")

        # Check available modules
        logger.info("Available Modules:")
        for module, available in self.available_modules.items():
            status = "OK" if available else "NOT FOUND"
            logger.info(f"  - {module}: {status}")
        logger.info("")

        # Run pipeline steps
        pipeline_success = True
        sla_exit_code = EXIT_SUCCESS

        for step in self.steps:
            success = self._run_step(step)
            if not success and not step.optional:
                pipeline_success = False
                break

            # Track SLA Intelligence exit code for final status
            if step.name == "SLA Intelligence" and step.exit_code is not None:
                sla_exit_code = step.exit_code

        self.end_time = datetime.now(timezone.utc)
        duration = (self.end_time - self.start_time).total_seconds()

        # Generate executive summary
        if not self.dry_run:
            try:
                summary_path = self._get_output_path("executive-summary.md")
                summary_path.parent.mkdir(parents=True, exist_ok=True)
                with open(summary_path, "w") as f:
                    f.write(self._generate_executive_summary())
                self.outputs["executive-summary.md"] = summary_path
                logger.info(f"Generated: executive-summary.md")
            except Exception as e:
                logger.warning(f"Failed to generate executive summary: {e}")

        # Generate bundle manifest
        if not self.dry_run:
            try:
                manifest_path = self._get_output_path("bundle-manifest.json")
                manifest_path.parent.mkdir(parents=True, exist_ok=True)
                with open(manifest_path, "w") as f:
                    json.dump(self._generate_bundle_manifest(), f, indent=2)
                self.outputs["bundle-manifest.json"] = manifest_path
                logger.info(f"Generated: bundle-manifest.json")
            except Exception as e:
                logger.warning(f"Failed to generate bundle manifest: {e}")

        # Generate run metadata (Phase 17)
        self._generate_run_metadata()

        # Generate executive narrative if requested (Phase 17)
        if self.with_narrative:
            self._generate_executive_narrative()

        # Final summary
        logger.info("")
        logger.info("=" * 60)
        logger.info("Pipeline Complete")
        logger.info("=" * 60)
        logger.info(f"Duration: {duration:.2f} seconds")
        logger.info(f"Reports Generated: {len(self.outputs)}")
        logger.info(f"Output Directory: {self.output_dir}")

        # Print artifact paths if requested
        if self.print_paths:
            self._print_artifact_paths()

        # Determine exit code
        if not pipeline_success:
            logger.error("Pipeline failed")
            return EXIT_PIPELINE_ERROR

        # Check for SLA conditions
        if self.fail_on_breach and sla_exit_code == 142:
            logger.warning("SLA breach detected (--fail-on-breach enabled)")
            return EXIT_SLA_BREACH

        if self.fail_on_critical and sla_exit_code in [132, 142]:
            logger.warning("Critical condition detected (--fail-on-critical enabled)")
            return sla_exit_code

        # Return SLA status for informational purposes
        if sla_exit_code == 141:
            logger.info("Note: At-risk SLAs detected")
            return EXIT_SUCCESS  # Don't fail unless --fail-on-breach

        logger.info("Pipeline completed successfully")
        return EXIT_SUCCESS


def create_argument_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        prog="run_full_org_governance_pipeline",
        description="Run the full T.A.R.S. organization governance pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exit Codes:
  0      Success, all steps completed
  1      Pipeline error (one or more steps failed)
  140    SLA at-risk detected (with --fail-on-breach)
  142    SLA breach detected (with --fail-on-breach)
  199    General orchestrator error

Examples:
  # Basic run
  python scripts/run_full_org_governance_pipeline.py --root ./org-health

  # With custom output directory and timestamp
  python scripts/run_full_org_governance_pipeline.py --root ./org-health --outdir ./reports/runs --timestamp 20251222-140000

  # With structured output format (daily/weekly/executive subdirs)
  python scripts/run_full_org_governance_pipeline.py --root ./org-health --format structured

  # Print artifact paths for CI/CD
  python scripts/run_full_org_governance_pipeline.py --root ./org-health --print-paths

  # Dry run mode
  python scripts/run_full_org_governance_pipeline.py --root ./org-health --dry-run

  # CI/CD mode
  python scripts/run_full_org_governance_pipeline.py --root ./org-health --fail-on-breach
"""
    )

    # Required arguments
    parser.add_argument(
        "--root",
        required=True,
        help="Root directory containing org health data (repo health dashboards)"
    )

    # Output options
    parser.add_argument(
        "--outdir",
        default="./reports/runs",
        help="Base output directory for generated reports (default: ./reports/runs)"
    )

    parser.add_argument(
        "--timestamp",
        default=None,
        help="Timestamp for run directory (format: YYYYMMDD-HHMMSS). Default: current UTC time"
    )

    parser.add_argument(
        "--format",
        dest="output_format",
        choices=["flat", "structured"],
        default="flat",
        help="Output format: 'flat' (all files in one dir) or 'structured' (daily/weekly/executive subdirs)"
    )

    parser.add_argument(
        "--print-paths",
        action="store_true",
        help="Print final artifact paths (useful for CI/CD logging)"
    )

    # SLA options
    parser.add_argument(
        "--sla-policy",
        help="Path to SLA policy YAML file"
    )

    parser.add_argument(
        "--window",
        dest="windows",
        type=int,
        action="append",
        help="Evaluation window in days (can be specified multiple times)"
    )

    # Output format options
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output in JSON format"
    )

    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Generate summary outputs only"
    )

    # CI/CD options
    parser.add_argument(
        "--fail-on-breach",
        action="store_true",
        help="Exit with code 142 if SLA breach is detected"
    )

    parser.add_argument(
        "--fail-on-critical",
        action="store_true",
        help="Exit with non-zero code if any critical condition is detected"
    )

    # Phase 17 options
    parser.add_argument(
        "--with-narrative",
        action="store_true",
        help="Generate executive narrative markdown (Phase 17, default OFF)"
    )

    # Dry run
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show commands without executing them"
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

    # Generate timestamp if not provided
    timestamp = args.timestamp if args.timestamp else generate_timestamp()

    try:
        # Build CLI flags string for metadata provenance
        cli_flags_str = " ".join(sys.argv[1:])

        orchestrator = PipelineOrchestrator(
            root_dir=args.root,
            output_dir=args.outdir,
            timestamp=timestamp,
            output_format=args.output_format,
            sla_policy=args.sla_policy,
            windows=args.windows,
            fail_on_breach=args.fail_on_breach,
            fail_on_critical=args.fail_on_critical,
            json_output=args.json,
            summary_only=args.summary_only,
            dry_run=args.dry_run,
            print_paths=args.print_paths
        )

        # Set Phase 17 options
        orchestrator.with_narrative = args.with_narrative
        orchestrator.cli_flags_str = cli_flags_str

        return orchestrator.run()

    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        return EXIT_PIPELINE_ERROR
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        return EXIT_GENERAL_ERROR


if __name__ == "__main__":
    sys.exit(main())
