#!/usr/bin/env python3
"""
Organization Health Governance CLI

Command-line interface for the org-level health governance engine.

Usage:
    python -m analytics.run_org_health --root-dir ./org-health --config ./org-health-config.yaml

Exit Codes (90-99):
    90: Success, no SLO violations
    91: SLO violations detected
    92: Org risk >= HIGH tier threshold
    93: No repos discovered / loaded
    94: Config error
    95: Data aggregation error
    99: General org-health error

Version: 1.0.0
Phase: 14.8 Task 1
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

from analytics.org_health_aggregator import (
    OrgHealthConfig,
    OrgHealthEngine,
    OrgHealthReport,
    SloPolicy,
    RepoSelector,
    SloOperator,
    RiskTierThresholds,
    load_slo_config,
    create_default_slo_policies,
    EXIT_ORG_SUCCESS,
    EXIT_SLO_VIOLATIONS,
    EXIT_HIGH_ORG_RISK,
    EXIT_NO_REPOS_DISCOVERED,
    EXIT_CONFIG_ERROR,
    EXIT_AGGREGATION_ERROR,
    EXIT_GENERAL_ORG_ERROR,
)

logger = logging.getLogger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        prog="run_org_health",
        description="Organization Health Governance Engine - Multi-Repo Aggregation & SLO Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exit Codes:
  90  Success, no SLO violations
  91  SLO violations detected
  92  Org risk >= HIGH tier threshold
  93  No repos discovered / loaded
  94  Config error
  95  Data aggregation error
  99  General org-health error

Examples:
  # Basic org health analysis with default SLOs
  python -m analytics.run_org_health --root-dir ./org-health

  # With custom SLO configuration
  python -m analytics.run_org_health --root-dir ./org-health --config ./org-health-config.yaml

  # Output to JSON file
  python -m analytics.run_org_health --root-dir ./org-health --output ./org-health-report.json

  # Filter specific repos
  python -m analytics.run_org_health --root-dir ./org-health --repos repo-a,repo-b,repo-c

  # Summary only mode
  python -m analytics.run_org_health --root-dir ./org-health --summary-only

  # Fail on SLO violations (for CI/CD)
  python -m analytics.run_org_health --root-dir ./org-health --fail-on-slo-violation

Directory Structure Expected:
  org-health/
    repo-a/
      dashboard/health-dashboard.json
      alerts/alerts.json
      trends/trend-report.json
    repo-b/
      ...
"""
    )

    # Required arguments
    parser.add_argument(
        "--root-dir",
        type=Path,
        required=True,
        help="Root directory containing per-repo health artifacts"
    )

    # Configuration
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to YAML or JSON configuration file with SLO policies"
    )

    # Output
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for org health report JSON"
    )

    # Filtering
    parser.add_argument(
        "--repos",
        type=str,
        default=None,
        help="Comma-separated list of repo IDs to analyze (default: all discovered)"
    )

    # Artifact paths
    parser.add_argument(
        "--dashboard-subdir",
        type=str,
        default="dashboard",
        help="Subdirectory name for dashboard artifacts (default: dashboard)"
    )

    parser.add_argument(
        "--alerts-subdir",
        type=str,
        default="alerts",
        help="Subdirectory name for alert artifacts (default: alerts)"
    )

    parser.add_argument(
        "--trends-subdir",
        type=str,
        default="trends",
        help="Subdirectory name for trend artifacts (default: trends)"
    )

    parser.add_argument(
        "--dashboard-filename",
        type=str,
        default="health-dashboard.json",
        help="Dashboard filename (default: health-dashboard.json)"
    )

    parser.add_argument(
        "--alerts-filename",
        type=str,
        default="alerts.json",
        help="Alerts filename (default: alerts.json)"
    )

    parser.add_argument(
        "--trends-filename",
        type=str,
        default="trend-report.json",
        help="Trends filename (default: trend-report.json)"
    )

    # Behavior flags
    parser.add_argument(
        "--fail-on-slo-violation",
        action="store_true",
        help="Exit with code 91 if any SLO is violated"
    )

    parser.add_argument(
        "--fail-on-critical-risk",
        action="store_true",
        help="Exit with code 92 if org risk is HIGH or CRITICAL"
    )

    parser.add_argument(
        "--use-default-slos",
        action="store_true",
        help="Use default SLO policies if no config provided"
    )

    # Output options
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only print a quick summary (no full analysis output)"
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress non-essential output"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON to stdout"
    )

    return parser


def print_summary(report: OrgHealthReport) -> None:
    """Print a formatted summary of the org health report."""
    print("\n" + "=" * 80)
    print("ORGANIZATION HEALTH SUMMARY")
    print("=" * 80)

    # Org-level status
    status_icon = {
        "green": "[OK]",
        "yellow": "[WARN]",
        "red": "[CRITICAL]"
    }.get(report.org_health_status, "[?]")

    print(f"\n{status_icon} Overall Health: {report.org_health_status.upper()}")
    print(f"    Score: {report.org_health_score:.1f}/100")
    print(f"    Risk Tier: {report.org_risk_tier.upper()}")

    # Repository counts
    metrics = report.metrics
    print("\n" + "-" * 40)
    print("REPOSITORIES")
    print("-" * 40)
    print(f"  Total: {metrics.total_repos}")
    print(f"  GREEN: {metrics.repos_green} | YELLOW: {metrics.repos_yellow} | RED: {metrics.repos_red}")

    if metrics.repos_improving or metrics.repos_declining:
        print(f"  Improving: {metrics.repos_improving} | Stable: {metrics.repos_stable} | Declining: {metrics.repos_declining}")

    # Risk distribution
    print("\n" + "-" * 40)
    print("RISK TIERS")
    print("-" * 40)
    print(f"  LOW: {metrics.repos_low_risk}")
    print(f"  MEDIUM: {metrics.repos_medium_risk}")
    print(f"  HIGH: {metrics.repos_high_risk}")
    print(f"  CRITICAL: {metrics.repos_critical_risk}")

    # SLO status
    print("\n" + "-" * 40)
    print("SLO STATUS")
    print("-" * 40)
    print(f"  Total: {report.total_slos}")
    print(f"  Satisfied: {report.slos_satisfied}")
    print(f"  Violated: {report.slos_violated}")

    if report.slos_violated > 0:
        print("\n  VIOLATED SLOs:")
        for result in report.slo_results:
            if not result.satisfied:
                print(f"    - [{result.slo_id}] {result.slo_description}")
                print(f"      Current: {result.current_value:.2f} | Target: {result.operator} {result.target_value}")
                if result.violating_repos:
                    print(f"      Violating repos: {', '.join(result.violating_repos[:5])}")
                    if len(result.violating_repos) > 5:
                        print(f"        ... and {len(result.violating_repos) - 5} more")

    # Top risk repos
    if report.top_risk_repos:
        print("\n" + "-" * 40)
        print("TOP RISK REPOSITORIES")
        print("-" * 40)
        for i, repo in enumerate(report.top_risk_repos[:5], 1):
            print(f"  {i}. {repo.repo_name} ({repo.risk_tier.upper()})")
            print(f"     Score: {repo.repository_score:.1f} | Trend: {repo.trend_direction}")
            if repo.reason_codes:
                print(f"     Reasons: {', '.join(repo.reason_codes[:3])}")

    # Recommendations
    if report.recommendations:
        print("\n" + "-" * 40)
        print("RECOMMENDATIONS")
        print("-" * 40)
        for rec in report.recommendations[:3]:
            priority_icon = {
                "critical": "[!!!]",
                "high": "[!!]",
                "medium": "[!]",
                "low": "[.]"
            }.get(rec.priority, "[?]")
            print(f"  {priority_icon} {rec.title}")
            print(f"      {rec.message}")
            if rec.affected_repos:
                print(f"      Affected: {', '.join(rec.affected_repos[:3])}")
                if len(rec.affected_repos) > 3:
                    print(f"        ... and {len(rec.affected_repos) - 3} more")

    # Metadata
    print("\n" + "=" * 80)
    print(f"Generated: {report.generated_at}")
    print(f"Repos Discovered: {report.repos_discovered} | Loaded: {report.repos_loaded} | Failed: {report.repos_failed}")
    print(f"Analysis Duration: {report.analysis_duration_ms:.1f}ms")
    print("=" * 80)


def print_detailed_report(report: OrgHealthReport) -> None:
    """Print detailed report including all repositories."""
    print_summary(report)

    # All repositories
    print("\n" + "=" * 80)
    print("ALL REPOSITORIES")
    print("=" * 80)

    for repo_data in report.repositories:
        status_icon = {
            "green": "[G]",
            "yellow": "[Y]",
            "red": "[R]",
            "unknown": "[?]"
        }.get(repo_data["health_status"], "[?]")

        risk_icon = {
            "low": "L",
            "medium": "M",
            "high": "H",
            "critical": "C"
        }.get(repo_data["risk_tier"], "?")

        print(f"\n{status_icon}[{risk_icon}] {repo_data['repo_name']}")
        print(f"    Score: {repo_data['repository_score']:.1f}")
        print(f"    Issues: {repo_data['total_issues']} (critical: {repo_data['critical_issues']})")

        if repo_data.get("has_trends"):
            print(f"    Trend: {repo_data['trends']['overall_trend']}")

        if repo_data.get("has_alerts") and repo_data['alerts']['total_alerts'] > 0:
            alerts = repo_data['alerts']
            print(f"    Alerts: {alerts['total_alerts']} (critical: {alerts['critical_alerts']})")


def main() -> int:
    """Main entry point for the CLI."""
    parser = create_parser()
    args = parser.parse_args()

    # Configure logging
    if args.quiet:
        log_level = logging.WARNING
    elif args.verbose:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO

    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    try:
        # Build configuration
        root_dir = args.root_dir

        if not root_dir.exists():
            logger.error(f"Root directory does not exist: {root_dir}")
            return EXIT_NO_REPOS_DISCOVERED

        # Load SLO policies
        slo_policies = []
        if args.config:
            if not args.config.exists():
                logger.error(f"Config file does not exist: {args.config}")
                return EXIT_CONFIG_ERROR
            try:
                slo_policies = load_slo_config(args.config)
                logger.info(f"Loaded {len(slo_policies)} SLO policies from {args.config}")
            except Exception as e:
                logger.error(f"Failed to load config: {e}")
                return EXIT_CONFIG_ERROR
        elif args.use_default_slos:
            slo_policies = create_default_slo_policies()
            logger.info(f"Using {len(slo_policies)} default SLO policies")

        # Parse repo filter
        repo_filter = None
        if args.repos:
            repo_filter = [r.strip() for r in args.repos.split(',')]
            logger.info(f"Filtering to repos: {repo_filter}")

        # Build config
        config = OrgHealthConfig(
            root_dir=root_dir,
            dashboard_subdir=args.dashboard_subdir,
            alerts_subdir=args.alerts_subdir,
            trends_subdir=args.trends_subdir,
            dashboard_filename=args.dashboard_filename,
            alerts_filename=args.alerts_filename,
            trends_filename=args.trends_filename,
            output_path=args.output,
            slo_policies=slo_policies,
            fail_on_slo_violation=args.fail_on_slo_violation,
            fail_on_critical_org_risk=args.fail_on_critical_risk,
            verbose=args.verbose,
            repo_filter=repo_filter
        )

        # Run analysis
        engine = OrgHealthEngine(config)
        report, exit_code = engine.run()

        # Output results
        if args.json:
            print(json.dumps(report.to_dict(), indent=2, default=str))
        elif not args.quiet:
            if args.summary_only:
                print_summary(report)
            else:
                print_detailed_report(report)

        return exit_code

    except KeyboardInterrupt:
        print("\nAnalysis cancelled by user")
        return EXIT_GENERAL_ORG_ERROR
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return EXIT_GENERAL_ORG_ERROR


if __name__ == "__main__":
    sys.exit(main())
