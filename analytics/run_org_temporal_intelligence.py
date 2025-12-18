#!/usr/bin/env python3
"""
CLI Tool for Advanced Correlation & Temporal Intelligence Engine

This module provides the command-line interface for running time-lagged
correlation analysis, influence scoring, and propagation path detection.

Usage:
    python -m analytics.run_org_temporal_intelligence --org-report ./org-health-report.json

    # With correlation report
    python -m analytics.run_org_temporal_intelligence --org-report ./report.json \\
        --trend-correlation-report ./trend-correlation-report.json

    # Custom thresholds
    python -m analytics.run_org_temporal_intelligence --org-report ./report.json \\
        --max-lag 5 \\
        --min-influence-score 40.0

    # CI/CD mode with failure on critical
    python -m analytics.run_org_temporal_intelligence --org-report ./report.json \\
        --fail-on-critical

Exit Codes:
    130: Success, no temporal risks
    131: Temporal correlations found (non-critical)
    132: Critical propagation risk detected
    133: Config error
    134: Parsing error
    199: General temporal intelligence error

Version: 1.0.0
Phase: 14.8 Task 4
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

from analytics.org_temporal_intelligence import (
    TemporalIntelligenceEngine,
    TemporalIntelligenceConfig,
    TemporalThresholds,
    TemporalIntelligenceReport,
    InfluenceDirection,
    TemporalSeverity,
    EXIT_TEMPORAL_SUCCESS,
    EXIT_TEMPORAL_CORRELATIONS_FOUND,
    EXIT_CRITICAL_PROPAGATION_RISK,
    EXIT_TEMPORAL_CONFIG_ERROR,
    EXIT_TEMPORAL_PARSE_ERROR,
    EXIT_GENERAL_TEMPORAL_ERROR
)

logger = logging.getLogger(__name__)


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        prog="run_org_temporal_intelligence",
        description="Advanced Correlation & Temporal Intelligence Engine - "
                   "Analyzes time-lagged correlations, influence patterns, and propagation paths",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exit Codes:
  130    Success, no temporal risks
  131    Temporal correlations found (non-critical)
  132    Critical propagation risk detected
  133    Configuration error
  134    Parsing error (invalid input report)
  199    General temporal intelligence error

Examples:
  # Basic analysis
  python -m analytics.run_org_temporal_intelligence --org-report ./org-health-report.json

  # Save output to file
  python -m analytics.run_org_temporal_intelligence --org-report ./report.json \\
      --output ./temporal-intelligence-report.json

  # JSON output to stdout
  python -m analytics.run_org_temporal_intelligence --org-report ./report.json --json

  # Custom lag window
  python -m analytics.run_org_temporal_intelligence --org-report ./report.json \\
      --max-lag 5

  # Custom influence threshold
  python -m analytics.run_org_temporal_intelligence --org-report ./report.json \\
      --min-influence-score 40.0

  # CI/CD mode - fail on critical propagation risks
  python -m analytics.run_org_temporal_intelligence --org-report ./report.json \\
      --fail-on-critical

  # Summary only (no detailed correlations)
  python -m analytics.run_org_temporal_intelligence --org-report ./report.json \\
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

    # Optional input
    parser.add_argument(
        "--trend-correlation-report",
        type=str,
        default=None,
        help="Path to trend-correlation-report.json from Phase 14.8 Task 3 (optional)"
    )

    # Output options
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Path to write temporal-intelligence-report.json"
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

    # Threshold options
    threshold_group = parser.add_argument_group("Threshold Options")
    threshold_group.add_argument(
        "--max-lag",
        type=int,
        default=3,
        help="Maximum lag to analyze (±max_lag intervals, default: 3)"
    )
    threshold_group.add_argument(
        "--min-influence-score",
        type=float,
        default=30.0,
        help="Minimum influence score to consider significant (default: 30.0)"
    )
    threshold_group.add_argument(
        "--min-correlation",
        type=float,
        default=0.5,
        help="Minimum lagged correlation coefficient (default: 0.5)"
    )
    threshold_group.add_argument(
        "--min-causality-score",
        type=float,
        default=0.4,
        help="Minimum causality score for propagation edges (default: 0.4)"
    )

    # Analysis options
    analysis_group = parser.add_argument_group("Analysis Options")
    analysis_group.add_argument(
        "--skip-lagged-correlations",
        action="store_true",
        help="Skip lagged correlation computation"
    )
    analysis_group.add_argument(
        "--skip-influence-scores",
        action="store_true",
        help="Skip influence score computation"
    )
    analysis_group.add_argument(
        "--skip-propagation-paths",
        action="store_true",
        help="Skip propagation path detection"
    )
    analysis_group.add_argument(
        "--skip-anomalies",
        action="store_true",
        help="Skip temporal anomaly detection"
    )

    # CI/CD options
    ci_group = parser.add_argument_group("CI/CD Options")
    ci_group.add_argument(
        "--fail-on-critical",
        action="store_true",
        help="Exit with code 132 if critical propagation risks detected"
    )
    ci_group.add_argument(
        "--fail-on-any-temporal-patterns",
        action="store_true",
        help="Exit with code 131 if any temporal patterns detected"
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


def build_config_from_args(args: argparse.Namespace) -> TemporalIntelligenceConfig:
    """Build TemporalIntelligenceConfig from parsed arguments."""
    # Validate org report path
    org_report_path = validate_org_report_path(args.org_report)

    # Validate optional correlation report path
    correlation_report_path = None
    if args.trend_correlation_report:
        corr_path = Path(args.trend_correlation_report)
        if corr_path.exists():
            correlation_report_path = corr_path

    # Build thresholds
    thresholds = TemporalThresholds(
        max_lag=args.max_lag,
        min_lag=-args.max_lag,
        min_lagged_correlation=args.min_correlation,
        min_significant_correlation=args.min_correlation,
        min_influence_score=args.min_influence_score,
        min_causality_score=args.min_causality_score
    )

    # Determine output path
    output_path = None
    if args.output:
        output_path = Path(args.output)

    config = TemporalIntelligenceConfig(
        org_report_path=org_report_path,
        correlation_report_path=correlation_report_path,
        output_path=output_path,
        thresholds=thresholds,
        compute_lagged_correlations=not args.skip_lagged_correlations,
        compute_influence_scores=not args.skip_influence_scores,
        compute_propagation_paths=not args.skip_propagation_paths,
        detect_temporal_anomalies=not args.skip_anomalies,
        verbose=args.verbose,
        summary_only=args.summary_only,
        fail_on_critical_propagation=args.fail_on_critical,
        fail_on_any_temporal_patterns=args.fail_on_any_temporal_patterns
    )

    return config


def print_summary(report: TemporalIntelligenceReport) -> None:
    """Print human-readable summary to console."""
    print("\n" + "=" * 80)
    print("ADVANCED CORRELATION & TEMPORAL INTELLIGENCE ANALYSIS")
    print("=" * 80)

    print(f"\nOrg Health: {report.org_health_status.upper()} "
          f"| Score: {report.org_health_score:.1f}")
    print(f"Repos Analyzed: {report.total_repos}")
    print(f"Analysis Time: {report.analysis_duration_ms:.1f}ms")

    print("\n" + "-" * 40)
    print("LAGGED CORRELATION SUMMARY")
    print("-" * 40)

    summary = report.summary
    print(f"Total Repo Pairs: {summary.total_repo_pairs}")
    print(f"Lagged Correlations Computed: {summary.lagged_correlations_computed}")
    print(f"Significant Correlations: {summary.significant_lagged_correlations}")
    print(f"Leader-Follower Pairs: {summary.leader_follower_pairs}")

    print("\n" + "-" * 40)
    print("INFLUENCE SCORING")
    print("-" * 40)
    print(f"Repos with Influence: {summary.repos_with_influence}")
    print(f"High-Influence Repos: {summary.high_influence_repos}")
    print(f"Leader Repos: {summary.leader_repos}")
    print(f"Follower Repos: {summary.follower_repos}")

    # Show top leaders
    if report.influence_scores:
        print("\n  Top Influencers:")
        for score in report.influence_scores[:5]:
            direction_icon = {
                InfluenceDirection.LEADER: "[LEADER]",
                InfluenceDirection.FOLLOWER: "[FOLLOWER]",
                InfluenceDirection.BIDIRECTIONAL: "[BIDIRECTIONAL]",
                InfluenceDirection.INDEPENDENT: "[INDEPENDENT]"
            }.get(score.influence_direction, "[?]")

            print(f"    #{score.influence_rank} {score.repo_id}: "
                  f"Score {score.influence_score:.1f} {direction_icon}")
            if score.repos_led > 0:
                print(f"        Leads {score.repos_led} repo(s): "
                      f"{', '.join(score.led_repos[:3])}"
                      + (f" (+{len(score.led_repos)-3})" if len(score.led_repos) > 3 else ""))

    if summary.propagation_paths_detected > 0:
        print("\n" + "-" * 40)
        print("PROPAGATION PATHS")
        print("-" * 40)
        print(f"Paths Detected: {summary.propagation_paths_detected}")
        print(f"  - Linear Paths: {summary.linear_paths}")
        print(f"  - Branching Paths: {summary.branching_paths}")
        print(f"Longest Path: {summary.longest_path_length} hop(s)")
        print(f"Repos in Paths: {summary.repos_in_paths}")
        print(f"Avg Propagation Lag: {summary.avg_propagation_lag:.1f} interval(s)")

        # Show top paths
        if report.propagation_paths:
            print("\n  Notable Paths:")
            for path in report.propagation_paths[:3]:
                print(f"    [{path.path_id}] {' → '.join(path.repo_sequence)}")
                print(f"        Lag: {path.total_lag} interval(s) | "
                      f"Confidence: {path.path_confidence:.2f}")

    if summary.total_anomalies > 0:
        print("\n" + "-" * 40)
        print("TEMPORAL ANOMALIES")
        print("-" * 40)
        print(f"Total Anomalies: {summary.total_anomalies}")
        print(f"  - Critical: {summary.critical_anomalies}")
        print(f"  - High: {summary.high_anomalies}")
        print(f"Systemic Risks: {summary.systemic_risks}")

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
                if anomaly.lag_pattern:
                    print(f"      Lag Pattern: {anomaly.lag_pattern}")

    if report.recommendations:
        print("\n" + "-" * 40)
        print("RECOMMENDATIONS")
        print("-" * 40)
        for rec in report.recommendations[:3]:
            priority = rec.get("priority", "medium").upper()
            print(f"\n  [{priority}] {rec.get('title', 'Recommendation')}")
            print(f"    {rec.get('message', '')}")

    if report.monitoring_priorities:
        print("\n" + "-" * 40)
        print("MONITORING PRIORITIES")
        print("-" * 40)
        for i, repo_id in enumerate(report.monitoring_priorities[:5], 1):
            print(f"  {i}. {repo_id}")

    print("\n" + "=" * 80)


def print_json_output(report: TemporalIntelligenceReport, summary_only: bool = False) -> None:
    """Print JSON output to stdout."""
    output = report.to_dict()

    if summary_only:
        # Only include summary fields
        output = {
            "report_id": output["report_id"],
            "generated_at": output["generated_at"],
            "org_report_path": output["org_report_path"],
            "summary": output["summary"],
            "leader_ranking": output["leader_ranking"],
            "monitoring_priorities": output["monitoring_priorities"],
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
        engine = TemporalIntelligenceEngine(config)
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
        return EXIT_TEMPORAL_PARSE_ERROR

    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        print(f"Error: {e}", file=sys.stderr)
        return EXIT_TEMPORAL_CONFIG_ERROR

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"Error: {e}", file=sys.stderr)
        return EXIT_GENERAL_TEMPORAL_ERROR


if __name__ == "__main__":
    sys.exit(main())
