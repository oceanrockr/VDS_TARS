#!/usr/bin/env python3
"""
CLI Tool for SLA Reporting & Executive Readiness Dashboard Engine

This module provides the command-line interface for evaluating SLA compliance,
attributing breaches, and generating executive readiness reports.

Usage:
    python -m analytics.run_org_sla_intelligence --org-report ./org-health-report.json

    # With all optional inputs
    python -m analytics.run_org_sla_intelligence \\
        --org-report ./org-health-report.json \\
        --alerts-report ./org-alerts.json \\
        --trend-correlation-report ./trend-correlation-report.json \\
        --temporal-intelligence-report ./temporal-intelligence-report.json \\
        --sla-policy ./sla-policies.yaml \\
        --output ./sla-intelligence-report.json

    # CI/CD mode with failure on breach
    python -m analytics.run_org_sla_intelligence --org-report ./report.json \\
        --fail-on-breach

Exit Codes:
    140: Success, all SLAs compliant
    141: At-risk SLAs detected
    142: SLA breach detected
    143: Config error
    144: Parsing error
    199: General SLA intelligence error

Version: 1.0.0
Phase: 14.8 Task 5
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

from analytics.org_sla_intelligence import (
    SLAIntelligenceEngine,
    SLAIntelligenceConfig,
    SLAThresholds,
    SLAIntelligenceReport,
    SLAStatus,
    SLASeverity,
    ReadinessTier,
    EXIT_SLA_SUCCESS,
    EXIT_SLA_AT_RISK,
    EXIT_SLA_BREACH,
    EXIT_SLA_CONFIG_ERROR,
    EXIT_SLA_PARSE_ERROR,
    EXIT_GENERAL_SLA_ERROR
)

logger = logging.getLogger(__name__)


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        prog="run_org_sla_intelligence",
        description="SLA Reporting & Executive Readiness Dashboard Engine - "
                   "Evaluates SLA compliance, attributes breaches, and generates executive reports",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exit Codes:
  140    Success, all SLAs compliant
  141    At-risk SLAs detected
  142    SLA breach detected
  143    Configuration error
  144    Parsing error (invalid input report)
  199    General SLA intelligence error

Examples:
  # Basic analysis
  python -m analytics.run_org_sla_intelligence --org-report ./org-health-report.json

  # Full analysis with all inputs
  python -m analytics.run_org_sla_intelligence \\
      --org-report ./org-health-report.json \\
      --alerts-report ./org-alerts.json \\
      --trend-correlation-report ./trend-correlation-report.json \\
      --temporal-intelligence-report ./temporal-intelligence-report.json

  # With custom SLA policy
  python -m analytics.run_org_sla_intelligence \\
      --org-report ./org-health-report.json \\
      --sla-policy ./sla-policies.yaml

  # Save output to file
  python -m analytics.run_org_sla_intelligence --org-report ./report.json \\
      --output ./sla-intelligence-report.json

  # JSON output to stdout
  python -m analytics.run_org_sla_intelligence --org-report ./report.json --json

  # Custom evaluation windows
  python -m analytics.run_org_sla_intelligence --org-report ./report.json \\
      --window 7 --window 30

  # CI/CD mode - fail on any breach
  python -m analytics.run_org_sla_intelligence --org-report ./report.json \\
      --fail-on-breach

  # Summary only (minimal output)
  python -m analytics.run_org_sla_intelligence --org-report ./report.json \\
      --summary-only --json
"""
    )

    # Required arguments
    parser.add_argument(
        "--org-report",
        type=str,
        required=True,
        help="Path to org-health-report.json from Phase 14.8 Task 1"
    )

    # Optional inputs
    input_group = parser.add_argument_group("Optional Input Reports")
    input_group.add_argument(
        "--alerts-report",
        type=str,
        default=None,
        help="Path to org-alerts.json from Phase 14.8 Task 2"
    )
    input_group.add_argument(
        "--trend-correlation-report",
        type=str,
        default=None,
        help="Path to trend-correlation-report.json from Phase 14.8 Task 3"
    )
    input_group.add_argument(
        "--temporal-intelligence-report",
        type=str,
        default=None,
        help="Path to temporal-intelligence-report.json from Phase 14.8 Task 4"
    )
    input_group.add_argument(
        "--sla-policy",
        type=str,
        default=None,
        help="Path to SLA policy file (YAML or JSON)"
    )

    # Output options
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Path to write sla-intelligence-report.json"
    )
    output_group.add_argument(
        "--json",
        action="store_true",
        help="Output full report as JSON to stdout"
    )
    output_group.add_argument(
        "--summary-only",
        action="store_true",
        help="Only output summary statistics (no detailed data)"
    )
    output_group.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )

    # Evaluation options
    eval_group = parser.add_argument_group("Evaluation Options")
    eval_group.add_argument(
        "--window",
        type=int,
        action="append",
        default=None,
        dest="windows",
        help="Evaluation window size (can specify multiple, e.g., --window 7 --window 30)"
    )

    # CI/CD options
    ci_group = parser.add_argument_group("CI/CD Options")
    ci_group.add_argument(
        "--fail-on-breach",
        action="store_true",
        help="Exit with code 142 if any SLA is breached"
    )
    ci_group.add_argument(
        "--fail-on-at-risk",
        action="store_true",
        help="Exit with code 141 if any SLA is at risk"
    )

    return parser


def validate_org_report_path(path_str: str) -> Path:
    """Validate that the org report path exists."""
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Org health report not found: {path}")
    if not path.is_file():
        raise ValueError(f"Org health report path is not a file: {path}")
    return path


def validate_optional_path(path_str: Optional[str]) -> Optional[Path]:
    """Validate optional path if provided."""
    if not path_str:
        return None
    path = Path(path_str)
    if path.exists() and path.is_file():
        return path
    logger.warning(f"Optional file not found: {path}")
    return None


def build_config_from_args(args: argparse.Namespace) -> SLAIntelligenceConfig:
    """Build SLAIntelligenceConfig from parsed arguments."""
    # Validate org report path
    org_report_path = validate_org_report_path(args.org_report)

    # Validate optional paths
    alerts_report_path = validate_optional_path(args.alerts_report)
    correlation_report_path = validate_optional_path(args.trend_correlation_report)
    temporal_report_path = validate_optional_path(args.temporal_intelligence_report)
    sla_policy_path = validate_optional_path(args.sla_policy)

    # Build thresholds (use defaults)
    thresholds = SLAThresholds()

    # Determine output path
    output_path = None
    if args.output:
        output_path = Path(args.output)

    # Evaluation windows
    windows = args.windows if args.windows else [7, 30, 90]

    config = SLAIntelligenceConfig(
        org_report_path=org_report_path,
        alerts_report_path=alerts_report_path,
        correlation_report_path=correlation_report_path,
        temporal_report_path=temporal_report_path,
        sla_policy_path=sla_policy_path,
        output_path=output_path,
        thresholds=thresholds,
        evaluation_windows=windows,
        verbose=args.verbose,
        summary_only=args.summary_only,
        fail_on_breach=args.fail_on_breach,
        fail_on_at_risk=args.fail_on_at_risk
    )

    return config


def print_summary(report: SLAIntelligenceReport) -> None:
    """Print human-readable summary to console."""
    print("\n" + "=" * 80)
    print("SLA REPORTING & EXECUTIVE READINESS DASHBOARD")
    print("=" * 80)

    # Executive Readiness Banner
    readiness = report.executive_readiness
    tier_colors = {
        "green": "[GREEN]",
        "yellow": "[YELLOW]",
        "red": "[RED]",
        "unknown": "[UNKNOWN]"
    }
    tier_display = tier_colors.get(readiness.readiness_tier, "[?]")

    print(f"\n{tier_display} EXECUTIVE READINESS: {readiness.readiness_score:.0f}/100")
    print("-" * 40)
    print(f"Risk Outlook: {readiness.risk_outlook}")

    # SLA Summary
    print("\n" + "-" * 40)
    print("SLA COMPLIANCE SUMMARY")
    print("-" * 40)
    summary = report.summary
    print(f"Total SLAs Evaluated: {summary.total_slas_evaluated}")
    print(f"  - Compliant: {summary.compliant_slas}")
    print(f"  - At Risk: {summary.at_risk_slas}")
    print(f"  - Breached: {summary.breached_slas}")
    print(f"Overall Compliance Rate: {summary.overall_compliance_rate:.1f}%")

    # Scorecards
    if report.scorecards:
        print("\n" + "-" * 40)
        print("SLA SCORECARDS")
        print("-" * 40)

        for card in report.scorecards:
            status_icons = {
                "compliant": "[OK]",
                "at_risk": "[!]",
                "breached": "[X]",
                "unknown": "[?]"
            }
            icon = status_icons.get(card.status.value, "[?]")
            trend = {"arrow_up": "^", "arrow_down": "v", "arrow_right": "-"}.get(
                card.trend_indicator, "-"
            )

            print(f"\n  {icon} {card.policy_name}")
            print(f"      Current: {card.current_value:.1f}{card.unit} "
                  f"| Target: {card.target_value:.1f}{card.unit}")
            print(f"      Trend: {trend} ({card.trend_description})")
            print(f"      Status: {card.plain_english_status}")

    # Breaches
    if report.breaches:
        print("\n" + "-" * 40)
        print("BREACHES & AT-RISK")
        print("-" * 40)
        print(f"Total Issues: {summary.total_breaches}")

        severity_order = ["critical", "high", "medium", "low"]
        for severity in severity_order:
            breaches = [b for b in report.breaches if b.severity.value == severity]
            if not breaches:
                continue

            icon = {"critical": "[CRIT]", "high": "[HIGH]",
                    "medium": "[MED]", "low": "[LOW]"}[severity]

            print(f"\n  {icon} {severity.upper()} ({len(breaches)}):")
            for breach in breaches[:2]:
                print(f"    - {breach.policy_name}")
                print(f"      Actual: {breach.actual_value:.1f} | Target: {breach.target_value:.1f}")
                if breach.primary_cause:
                    print(f"      Primary Cause: {breach.primary_cause}")

    # Risk Narrative
    if report.risk_narrative:
        print("\n" + "-" * 40)
        print("EXECUTIVE SUMMARY")
        print("-" * 40)
        print(f"\n  {report.risk_narrative.headline}")
        print(f"\n  {report.risk_narrative.summary_paragraph}")

        if report.risk_narrative.key_risks:
            print("\n  Key Risks:")
            for risk in report.risk_narrative.key_risks[:3]:
                print(f"    - {risk}")

        if report.risk_narrative.recommended_focus_areas:
            print("\n  Recommended Focus Areas:")
            for area in report.risk_narrative.recommended_focus_areas[:3]:
                print(f"    - {area}")

    # Recommendations
    if report.recommendations:
        print("\n" + "-" * 40)
        print("RECOMMENDATIONS")
        print("-" * 40)
        for rec in report.recommendations[:3]:
            priority = rec.get("priority", "medium").upper()
            print(f"\n  [{priority}] {rec.get('title', 'Recommendation')}")
            print(f"    {rec.get('message', '')}")

    # Footer
    print("\n" + "=" * 80)
    print(f"Analysis Time: {report.analysis_duration_ms:.1f}ms")
    print(f"Generated: {report.generated_at}")
    print("=" * 80)


def print_json_output(report: SLAIntelligenceReport, summary_only: bool = False) -> None:
    """Print JSON output to stdout."""
    output = report.to_dict()

    if summary_only:
        # Only include summary fields
        output = {
            "report_id": output["report_id"],
            "generated_at": output["generated_at"],
            "org_report_path": output["org_report_path"],
            "summary": output["summary"],
            "executive_readiness": {
                "readiness_score": output["executive_readiness"]["readiness_score"],
                "readiness_tier": output["executive_readiness"]["readiness_tier"],
                "risk_outlook": output["executive_readiness"]["risk_outlook"],
                "compliant_slas": output["executive_readiness"]["compliant_slas"],
                "at_risk_slas": output["executive_readiness"]["at_risk_slas"],
                "breached_slas": output["executive_readiness"]["breached_slas"]
            },
            "breach_summary": output["breach_summary"],
            "risk_narrative": {
                "headline": output["risk_narrative"]["headline"],
                "summary_paragraph": output["risk_narrative"]["summary_paragraph"]
            },
            "org_health_status": output["org_health_status"],
            "org_health_score": output["org_health_score"],
            "analysis_duration_ms": output["analysis_duration_ms"]
        }

    print(json.dumps(output, indent=2, default=str))


def main(args: Optional[list] = None) -> int:
    """
    Main entry point for the CLI.

    Args:
        args: Optional list of command-line arguments (for testing)

    Returns:
        Exit code
    """
    parser = create_argument_parser()
    parsed_args = parser.parse_args(args)

    try:
        # Build configuration
        config = build_config_from_args(parsed_args)

        # Create and run engine
        engine = SLAIntelligenceEngine(config)
        report, exit_code = engine.run()

        # Handle output
        if parsed_args.json:
            print_json_output(report, parsed_args.summary_only)
        else:
            print_summary(report)

        return exit_code

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        print(f"Error: {e}", file=sys.stderr)
        return EXIT_SLA_PARSE_ERROR

    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        print(f"Error: {e}", file=sys.stderr)
        return EXIT_SLA_CONFIG_ERROR

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"Error: {e}", file=sys.stderr)
        return EXIT_GENERAL_SLA_ERROR


if __name__ == "__main__":
    sys.exit(main())
