#!/usr/bin/env python3
"""
CLI Tool for Multi-Repository Trend Correlation Engine

This module provides the command-line interface for running cross-repository
trend correlation analysis.

Usage:
    python -m analytics.run_org_trend_correlation --org-report ./org-health-report.json

    # With output file
    python -m analytics.run_org_trend_correlation --org-report ./report.json \
        --output ./trend-correlation-report.json

    # Custom thresholds
    python -m analytics.run_org_trend_correlation --org-report ./report.json \
        --min-correlation-threshold 0.6 \
        --min-cluster-size 3

    # CI/CD mode with failure on critical
    python -m analytics.run_org_trend_correlation --org-report ./report.json \
        --fail-on-critical

Exit Codes:
    120: Success, no concerning correlations
    121: Correlations found (non-critical)
    122: Critical cross-repo anomaly detected
    123: Config error
    124: Parsing error
    199: General correlation error

Version: 1.0.0
Phase: 14.8 Task 3
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

from analytics.org_trend_correlation import (
    TrendCorrelationEngine,
    TrendCorrelationConfig,
    CorrelationThresholds,
    TrendCorrelationReport,
    EXIT_CORRELATION_SUCCESS,
    EXIT_CORRELATIONS_FOUND,
    EXIT_CRITICAL_ANOMALY,
    EXIT_CORRELATION_CONFIG_ERROR,
    EXIT_CORRELATION_PARSE_ERROR,
    EXIT_GENERAL_CORRELATION_ERROR
)

logger = logging.getLogger(__name__)


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        prog="run_org_trend_correlation",
        description="Multi-Repository Trend Correlation Engine - "
                   "Analyzes cross-repository trend patterns and detects correlated anomalies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exit Codes:
  120    Success, no concerning correlations
  121    Correlations found (non-critical)
  122    Critical cross-repo anomaly detected
  123    Configuration error
  124    Parsing error (invalid org report)
  199    General correlation error

Examples:
  # Basic analysis
  python -m analytics.run_org_trend_correlation --org-report ./org-health-report.json

  # Save output to file
  python -m analytics.run_org_trend_correlation --org-report ./report.json \\
      --output ./trend-correlation-report.json

  # JSON output to stdout
  python -m analytics.run_org_trend_correlation --org-report ./report.json --json

  # Custom correlation threshold
  python -m analytics.run_org_trend_correlation --org-report ./report.json \\
      --min-correlation-threshold 0.7

  # CI/CD mode - fail on critical anomalies
  python -m analytics.run_org_trend_correlation --org-report ./report.json \\
      --fail-on-critical

  # Summary only (no detailed correlations)
  python -m analytics.run_org_trend_correlation --org-report ./report.json \\
      --summary-only
"""
    )

    # Required arguments
    parser.add_argument(
        "--org-report",
        type=str,
        required=True,
        help="Path to org-health-report.json from Phase 14.8 Task 1"
    )

    # Output options
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Path to write trend-correlation-report.json"
    )
    output_group.add_argument(
        "--json",
        action="store_true",
        help="Output full report as JSON to stdout"
    )
    output_group.add_argument(
        "--summary-only",
        action="store_true",
        help="Only output summary statistics (no detailed correlations)"
    )
    output_group.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )

    # Threshold options
    threshold_group = parser.add_argument_group("Threshold Options")
    threshold_group.add_argument(
        "--min-correlation-threshold",
        type=float,
        default=0.5,
        help="Minimum correlation coefficient to consider significant (default: 0.5)"
    )
    threshold_group.add_argument(
        "--min-cluster-size",
        type=int,
        default=2,
        help="Minimum number of repos to form a cluster (default: 2)"
    )
    threshold_group.add_argument(
        "--synchronized-decline-threshold",
        type=float,
        default=0.20,
        help="Minimum ratio of repos declining together to trigger anomaly (default: 0.20)"
    )

    # Analysis options
    analysis_group = parser.add_argument_group("Analysis Options")
    analysis_group.add_argument(
        "--skip-clusters",
        action="store_true",
        help="Skip cluster detection"
    )
    analysis_group.add_argument(
        "--skip-anomalies",
        action="store_true",
        help="Skip anomaly detection"
    )
    analysis_group.add_argument(
        "--skip-leading-indicators",
        action="store_true",
        help="Skip leading indicator detection"
    )

    # CI/CD options
    ci_group = parser.add_argument_group("CI/CD Options")
    ci_group.add_argument(
        "--fail-on-critical",
        action="store_true",
        help="Exit with code 122 if critical anomalies are detected"
    )
    ci_group.add_argument(
        "--fail-on-any-correlations",
        action="store_true",
        help="Exit with code 121 if any significant correlations are found"
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


def build_config_from_args(args: argparse.Namespace) -> TrendCorrelationConfig:
    """Build TrendCorrelationConfig from parsed arguments."""
    # Validate org report path
    org_report_path = validate_org_report_path(args.org_report)

    # Build thresholds
    thresholds = CorrelationThresholds(
        significance_threshold=args.min_correlation_threshold,
        min_cluster_correlation=args.min_correlation_threshold,
        min_cluster_size=args.min_cluster_size,
        synchronized_decline_threshold=args.synchronized_decline_threshold
    )

    # Determine output path
    output_path = None
    if args.output:
        output_path = Path(args.output)

    config = TrendCorrelationConfig(
        org_report_path=org_report_path,
        output_path=output_path,
        thresholds=thresholds,
        compute_clusters=not args.skip_clusters,
        detect_anomalies=not args.skip_anomalies,
        compute_leading_indicators=not args.skip_leading_indicators,
        verbose=args.verbose,
        summary_only=args.summary_only,
        fail_on_critical_anomaly=args.fail_on_critical,
        fail_on_any_correlations=args.fail_on_any_correlations
    )

    return config


def print_summary(report: TrendCorrelationReport) -> None:
    """Print human-readable summary to console."""
    print("\n" + "=" * 80)
    print("MULTI-REPOSITORY TREND CORRELATION ANALYSIS")
    print("=" * 80)

    print(f"\nOrg Health: {report.org_health_status.upper()} "
          f"| Score: {report.org_health_score:.1f}")
    print(f"Repos Analyzed: {report.total_repos}")
    print(f"Analysis Time: {report.analysis_duration_ms:.1f}ms")

    print("\n" + "-" * 40)
    print("CORRELATION SUMMARY")
    print("-" * 40)

    summary = report.summary
    print(f"Total Repo Pairs: {summary.total_repo_pairs}")
    print(f"Significant Correlations: {summary.significant_correlations}")
    print(f"  - Positive: {summary.positive_correlations}")
    print(f"  - Negative: {summary.negative_correlations}")
    print(f"  - Synchronized Decline: {summary.synchronized_decline_pairs}")
    print(f"  - Synchronized Improvement: {summary.synchronized_improvement_pairs}")
    print(f"Correlation Density: {summary.correlation_density*100:.1f}%")

    if summary.total_clusters > 0:
        print("\n" + "-" * 40)
        print("CLUSTERS")
        print("-" * 40)
        print(f"Total Clusters: {summary.total_clusters}")
        print(f"Largest Cluster: {summary.largest_cluster_size} repos")
        print(f"Risk Clusters: {summary.risk_clusters}")

        # Show top clusters
        for cluster in report.clusters[:3]:
            print(f"\n  [{cluster.cluster_id}] {cluster.cluster_name}")
            print(f"    Repos: {cluster.repo_count} | "
                  f"Avg Correlation: {cluster.avg_internal_correlation:.2f}")
            print(f"    Trend: {cluster.dominant_trend.upper()} | "
                  f"Risk: {cluster.cluster_risk_tier.upper()}")
            if cluster.repo_ids[:3]:
                print(f"    Members: {', '.join(cluster.repo_ids[:3])}"
                      + (f" (+{len(cluster.repo_ids)-3} more)" if len(cluster.repo_ids) > 3 else ""))

    if summary.total_anomalies > 0:
        print("\n" + "-" * 40)
        print("ANOMALIES")
        print("-" * 40)
        print(f"Total Anomalies: {summary.total_anomalies}")
        print(f"  - Critical: {summary.critical_anomalies}")
        print(f"  - High: {summary.high_anomalies}")
        print(f"Predictive Indicators: {summary.predictive_indicators}")

        # Show anomalies by severity
        severity_order = ["critical", "high", "medium", "low"]
        for severity in severity_order:
            anomalies = [a for a in report.anomalies if a.severity.value == severity]
            if not anomalies:
                continue

            icon = {"critical": "[CRIT]", "high": "[HIGH]",
                    "medium": "[MED]", "low": "[LOW]"}[severity]

            print(f"\n  {icon} {severity.upper()} ({len(anomalies)}):")
            for anomaly in anomalies[:2]:
                print(f"    - {anomaly.title}")
                print(f"      {anomaly.message}")
                if anomaly.is_predictive:
                    print(f"      [PREDICTIVE] Confidence: {anomaly.confidence*100:.0f}%")

    if report.recommendations:
        print("\n" + "-" * 40)
        print("RECOMMENDATIONS")
        print("-" * 40)
        for rec in report.recommendations[:3]:
            priority = rec.get("priority", "medium").upper()
            print(f"\n  [{priority}] {rec.get('title', 'Recommendation')}")
            print(f"    {rec.get('message', '')}")

    print("\n" + "=" * 80)


def print_json_output(report: TrendCorrelationReport, summary_only: bool = False) -> None:
    """Print JSON output to stdout."""
    output = report.to_dict()

    if summary_only:
        # Only include summary fields
        output = {
            "report_id": output["report_id"],
            "generated_at": output["generated_at"],
            "org_report_path": output["org_report_path"],
            "summary": output["summary"],
            "org_health_status": output["org_health_status"],
            "org_health_score": output["org_health_score"],
            "total_repos": output["total_repos"],
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
        engine = TrendCorrelationEngine(config)
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
        return EXIT_CORRELATION_PARSE_ERROR

    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        print(f"Error: {e}", file=sys.stderr)
        return EXIT_CORRELATION_CONFIG_ERROR

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"Error: {e}", file=sys.stderr)
        return EXIT_GENERAL_CORRELATION_ERROR


if __name__ == "__main__":
    sys.exit(main())
