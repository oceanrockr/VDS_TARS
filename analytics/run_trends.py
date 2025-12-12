#!/usr/bin/env python3
"""
Repository Health Trend Analyzer CLI

Command-line interface for the trend analysis engine.

Usage:
    python -m analytics.run_trends --history-dir ./dashboard-history --output ./trend-report.json

Exit Codes (80-89):
    80: Trend analysis successful
    81: Not enough history (minimum snapshots not met)
    82: Invalid snapshot (corrupted or malformed)
    83: Time series computation error
    84: Prediction error
    85: History store read/write error
    86: Chart generation error
    89: General trend analysis failure

Version: 1.0.0
Phase: 14.7 Task 10
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from analytics.trend_analyzer import (
    TrendConfig,
    TrendEngine,
    TrendReport,
    HealthHistoryStore,
    add_snapshot_to_history,
    get_trend_summary,
    EXIT_TREND_SUCCESS,
    EXIT_INSUFFICIENT_HISTORY,
    EXIT_INVALID_SNAPSHOT,
    EXIT_COMPUTATION_ERROR,
    EXIT_PREDICTION_ERROR,
    EXIT_HISTORY_STORE_ERROR,
    EXIT_CHART_GENERATION_ERROR,
    EXIT_GENERAL_TREND_ERROR,
)

logger = logging.getLogger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        prog="run_trends",
        description="Repository Health Trend Analyzer - Time-Series Analysis Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exit Codes:
  80  Trend analysis successful
  81  Not enough history (minimum snapshots not met)
  82  Invalid snapshot (corrupted or malformed)
  83  Time series computation error
  84  Prediction error
  85  History store read/write error
  86  Chart generation error
  89  General trend analysis failure

Examples:
  # Basic trend analysis
  python -m analytics.run_trends --history-dir ./dashboard-history

  # With output file
  python -m analytics.run_trends --history-dir ./dashboard-history --output ./trend-report.json

  # With chart generation
  python -m analytics.run_trends --history-dir ./dashboard-history --generate-charts --chart-dir ./charts

  # Add a dashboard to history first
  python -m analytics.run_trends --add-snapshot ./dashboard.json --history-dir ./dashboard-history

  # Quick summary (no full analysis)
  python -m analytics.run_trends --history-dir ./dashboard-history --summary-only
"""
    )

    # Required arguments
    parser.add_argument(
        "--history-dir",
        type=Path,
        required=True,
        help="Directory containing historical dashboard snapshots"
    )

    # Optional arguments
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for trend report JSON"
    )

    parser.add_argument(
        "--generate-charts",
        action="store_true",
        help="Generate trend visualization charts"
    )

    parser.add_argument(
        "--chart-dir",
        type=Path,
        default=None,
        help="Directory for chart output (default: history-dir/charts)"
    )

    # Analysis parameters
    parser.add_argument(
        "--min-snapshots",
        type=int,
        default=3,
        help="Minimum snapshots required for analysis (default: 3)"
    )

    parser.add_argument(
        "--max-snapshots",
        type=int,
        default=100,
        help="Maximum snapshots to analyze (default: 100)"
    )

    parser.add_argument(
        "--zscore-threshold",
        type=float,
        default=2.0,
        help="Z-score threshold for anomaly detection (default: 2.0)"
    )

    parser.add_argument(
        "--score-drop-threshold",
        type=float,
        default=15.0,
        help="Score drop threshold for anomaly detection (default: 15.0)"
    )

    parser.add_argument(
        "--issue-spike-threshold",
        type=int,
        default=5,
        help="Issue spike threshold for anomaly detection (default: 5)"
    )

    parser.add_argument(
        "--prediction-horizon",
        type=int,
        default=3,
        help="Number of snapshots to predict ahead (default: 3)"
    )

    parser.add_argument(
        "--confidence-level",
        type=float,
        default=0.95,
        help="Confidence level for predictions (default: 0.95)"
    )

    parser.add_argument(
        "--warning-threshold",
        type=float,
        default=60.0,
        help="Warning score threshold (default: 60.0)"
    )

    parser.add_argument(
        "--critical-threshold",
        type=float,
        default=40.0,
        help="Critical score threshold (default: 40.0)"
    )

    parser.add_argument(
        "--volatility-threshold",
        type=float,
        default=10.0,
        help="High volatility threshold (default: 10.0)"
    )

    # Snapshot management
    parser.add_argument(
        "--add-snapshot",
        type=Path,
        default=None,
        help="Add a dashboard snapshot to history (runs before analysis)"
    )

    parser.add_argument(
        "--snapshot-version",
        type=str,
        default=None,
        help="Version string for added snapshot"
    )

    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Rebuild the history index from snapshot files"
    )

    parser.add_argument(
        "--validate-index",
        action="store_true",
        help="Validate history index integrity"
    )

    # Output options
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only print a quick trend summary (no full analysis)"
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


def print_summary(report: TrendReport) -> None:
    """Print a formatted summary of the trend report."""
    print("\n" + "=" * 80)
    print("REPOSITORY HEALTH TREND ANALYSIS")
    print("=" * 80)

    print(f"\nAnalysis Window: {report.first_snapshot} to {report.last_snapshot}")
    print(f"Snapshots Analyzed: {report.snapshots_analyzed}")
    print(f"Window Duration: {report.analysis_window_days} days")

    print("\n" + "-" * 40)
    print("CURRENT STATE")
    print("-" * 40)
    print(f"  Current Score: {report.current_score:.1f}/100")
    print(f"  Current Health: {report.current_health.upper()}")

    print("\n" + "-" * 40)
    print("TREND ANALYSIS")
    print("-" * 40)
    print(f"  Overall Trend: {report.overall_trend.upper()}")
    print(f"  Trend Confidence: {report.trend_confidence:.1%}")
    print(f"  Regression Slope: {report.regression_slope:+.2f} points/snapshot")

    if report.score_trend:
        print(f"  Score Change: {report.score_trend.score_change:+.1f}")

    print("\n" + "-" * 40)
    print("MOVING AVERAGES")
    print("-" * 40)
    if report.ma_3 > 0:
        print(f"  3-Snapshot MA: {report.ma_3:.1f}")
    if report.ma_7 > 0:
        print(f"  7-Snapshot MA: {report.ma_7:.1f}")
    if report.ma_14 > 0:
        print(f"  14-Snapshot MA: {report.ma_14:.1f}")

    print("\n" + "-" * 40)
    print("VOLATILITY")
    print("-" * 40)
    print(f"  Standard Deviation: {report.score_volatility:.2f}")
    print(f"  Volatility Trend: {report.volatility_trend}")

    print("\n" + "-" * 40)
    print("PREDICTIONS")
    print("-" * 40)
    print(f"  Predicted Next Score: {report.predicted_next_score:.1f}")
    print(f"  95% Confidence Interval: [{report.confidence_interval[0]:.1f}, {report.confidence_interval[1]:.1f}]")

    if report.prediction:
        print(f"  Probability of YELLOW (< 60): {report.prediction.probability_yellow:.0%}")
        print(f"  Probability of RED (< 40): {report.prediction.probability_red:.0%}")

    if report.total_anomalies > 0:
        print("\n" + "-" * 40)
        print(f"ANOMALIES ({report.total_anomalies})")
        print("-" * 40)
        for anomaly in report.anomalies[:5]:  # Show first 5
            print(f"  [{anomaly.severity.upper()}] {anomaly.anomaly_type}")
            print(f"    {anomaly.description}")

    if report.total_warnings > 0:
        print("\n" + "-" * 40)
        print(f"EARLY WARNINGS ({report.total_warnings})")
        print("-" * 40)
        for warning in report.early_warnings:
            print(f"  [{warning.level.upper()}] {warning.title}")
            print(f"    {warning.message}")
            if warning.recommendations:
                print(f"    -> {warning.recommendations[0]}")

    if report.degrading_versions:
        print("\n" + "-" * 40)
        print("DEGRADING VERSIONS")
        print("-" * 40)
        for version in report.degrading_versions:
            print(f"  - {version}")

    print("\n" + "=" * 80)
    print(f"Analysis Duration: {report.analysis_duration_ms:.1f}ms")
    print("=" * 80)


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
        history_dir = args.history_dir

        # Handle rebuild index
        if args.rebuild_index:
            logger.info(f"Rebuilding index for {history_dir}")
            store = HealthHistoryStore(history_dir)
            store.initialize()
            count = store.rebuild_index()
            print(f"Rebuilt index: {count} snapshots indexed")
            return EXIT_TREND_SUCCESS

        # Handle validate index
        if args.validate_index:
            logger.info(f"Validating index for {history_dir}")
            store = HealthHistoryStore(history_dir)
            store.initialize()
            is_valid, issues = store.validate_index()

            if is_valid:
                print("Index is valid")
                return EXIT_TREND_SUCCESS
            else:
                print(f"Index has {len(issues)} issue(s):")
                for issue in issues:
                    print(f"  - {issue}")
                return EXIT_INVALID_SNAPSHOT

        # Handle add snapshot
        if args.add_snapshot:
            logger.info(f"Adding snapshot: {args.add_snapshot}")
            metadata = add_snapshot_to_history(
                history_dir,
                args.add_snapshot,
                args.snapshot_version
            )
            print(f"Added snapshot: {metadata.snapshot_id}")
            print(f"  Score: {metadata.repository_score:.1f}")
            print(f"  Health: {metadata.overall_health}")
            print(f"  Issues: {metadata.total_issues}")

            if args.summary_only:
                return EXIT_TREND_SUCCESS

        # Handle summary only
        if args.summary_only:
            summary = get_trend_summary(history_dir)

            if args.json:
                import json
                print(json.dumps(summary, indent=2))
            else:
                print("\n" + "=" * 60)
                print("QUICK TREND SUMMARY")
                print("=" * 60)
                print(f"Status: {summary.get('status', 'unknown')}")
                print(f"Snapshots: {summary.get('snapshots', 0)}")

                if summary.get("status") == "ok":
                    print(f"Trend: {summary.get('trend', 'unknown').upper()}")
                    print(f"First Score: {summary.get('first_score', 0):.1f}")
                    print(f"Last Score: {summary.get('last_score', 0):.1f}")
                    print(f"Score Change: {summary.get('score_change', 0):+.1f}")
                else:
                    print(f"Message: {summary.get('message', 'Unknown error')}")

                print("=" * 60)

            return EXIT_TREND_SUCCESS

        # Build configuration
        chart_dir = args.chart_dir or (history_dir / "charts")

        config = TrendConfig(
            history_dir=history_dir,
            output_path=args.output,
            min_snapshots=args.min_snapshots,
            max_snapshots=args.max_snapshots,
            zscore_threshold=args.zscore_threshold,
            score_drop_threshold=args.score_drop_threshold,
            issue_spike_threshold=args.issue_spike_threshold,
            prediction_horizon=args.prediction_horizon,
            confidence_level=args.confidence_level,
            warning_score_threshold=args.warning_threshold,
            critical_score_threshold=args.critical_threshold,
            volatility_threshold=args.volatility_threshold,
            generate_charts=args.generate_charts,
            chart_output_dir=chart_dir if args.generate_charts else None,
            verbose=args.verbose
        )

        # Run analysis
        engine = TrendEngine(config)
        report, exit_code = engine.run()

        # Output results
        if args.json:
            import json
            print(json.dumps(report.to_dict(), indent=2, default=str))
        elif not args.quiet:
            print_summary(report)

        return exit_code

    except KeyboardInterrupt:
        print("\nAnalysis cancelled by user")
        return EXIT_GENERAL_TREND_ERROR
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return EXIT_GENERAL_TREND_ERROR


if __name__ == "__main__":
    sys.exit(main())
