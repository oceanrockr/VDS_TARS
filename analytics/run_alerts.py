"""
T.A.R.S. Repository Health Alerting CLI

Command-line interface for the Repository Health Alerting Engine.
Evaluates alert rules against dashboard JSON and dispatches alerts.

Usage:
    python -m analytics.run_alerts \\
      --current-dashboard path.json \\
      --previous-dashboard old.json \\
      --output alerts.json \\
      --channels console,file,email \\
      --severity-threshold WARNING \\
      --email-to admin@example.com

Exit Codes (70-79):
    70 - No alerts triggered (normal)
    71 - Alerts triggered (non-critical)
    72 - Critical alerts triggered
    73 - Invalid dashboard input
    74 - Channel dispatch failure
    75 - Alert rule evaluation failure
    76 - Alerts JSON write failure
    79 - General alerting engine error

Version: 1.0.0
Phase: 14.7 Task 9
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from analytics.alerting_engine import (
    AlertingEngine,
    AlertingConfig,
    ChannelConfig,
    ChannelType,
    AlertSeverity,
    EXIT_NO_ALERTS,
    EXIT_ALERTS_TRIGGERED,
    EXIT_CRITICAL_ALERTS,
    EXIT_INVALID_DASHBOARD,
    EXIT_CHANNEL_DISPATCH_FAILURE,
    EXIT_RULE_EVALUATION_FAILURE,
    EXIT_ALERTS_WRITE_FAILURE,
    EXIT_GENERAL_ALERTING_ERROR,
)


def parse_channels(channel_str: str, output_dir: Optional[Path] = None,
                   email_to: Optional[str] = None,
                   webhook_url: Optional[str] = None) -> List[ChannelConfig]:
    """Parse channel string and create configurations."""
    configs = []

    for name in channel_str.split(','):
        name = name.strip().lower()

        if name == 'console':
            configs.append(ChannelConfig(
                channel_type=ChannelType.CONSOLE,
                enabled=True
            ))

        elif name == 'file':
            if output_dir:
                configs.append(ChannelConfig(
                    channel_type=ChannelType.FILE,
                    enabled=True,
                    output_path=output_dir / "alerts.txt"
                ))
            else:
                print(f"Warning: File channel requires --output directory")

        elif name == 'email':
            if email_to:
                configs.append(ChannelConfig(
                    channel_type=ChannelType.EMAIL,
                    enabled=True,
                    email_to=email_to
                ))
            else:
                print(f"Warning: Email channel requires --email-to")

        elif name == 'webhook':
            if webhook_url:
                configs.append(ChannelConfig(
                    channel_type=ChannelType.WEBHOOK,
                    enabled=True,
                    webhook_url=webhook_url
                ))
            else:
                print(f"Warning: Webhook channel requires --webhook-url")

        else:
            print(f"Warning: Unknown channel type: {name}")

    return configs


def discover_previous_dashboard(current_path: Path) -> Optional[Path]:
    """
    Auto-discover previous dashboard in same directory.

    Looks for patterns like:
    - health-dashboard.previous.json
    - health-dashboard.backup.json
    - previous-health-dashboard.json
    """
    directory = current_path.parent
    patterns = [
        "health-dashboard.previous.json",
        "health-dashboard.backup.json",
        "previous-health-dashboard.json",
        "health-dashboard-previous.json",
    ]

    for pattern in patterns:
        candidate = directory / pattern
        if candidate.exists():
            return candidate

    # Try finding any other dashboard file (excluding current)
    for json_file in directory.glob("*dashboard*.json"):
        if json_file != current_path and json_file.exists():
            return json_file

    return None


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="run_alerts",
        description="T.A.R.S. Repository Health Alerting Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage - evaluate alerts and print to console
  python -m analytics.run_alerts \\
    --current-dashboard ./dashboard/health-dashboard.json

  # With previous dashboard for trend-based alerts
  python -m analytics.run_alerts \\
    --current-dashboard ./dashboard/health-dashboard.json \\
    --previous-dashboard ./dashboard/health-dashboard.previous.json \\
    --output ./alerts/alerts.json

  # Multiple channels with email
  python -m analytics.run_alerts \\
    --current-dashboard ./dashboard/health-dashboard.json \\
    --channels console,file,email \\
    --email-to admin@example.com \\
    --output ./alerts

  # Fail on critical alerts only
  python -m analytics.run_alerts \\
    --current-dashboard ./dashboard/health-dashboard.json \\
    --severity-threshold WARNING \\
    --fail-on-critical

  # CI/CD mode - fail on any alert
  python -m analytics.run_alerts \\
    --current-dashboard ./dashboard/health-dashboard.json \\
    --fail-on-any-alert \\
    --severity-threshold ERROR

Exit Codes:
  70 - No alerts triggered (normal)
  71 - Alerts triggered (non-critical)
  72 - Critical alerts triggered
  73 - Invalid dashboard input
  74 - Channel dispatch failure
  75 - Alert rule evaluation failure
  76 - Alerts JSON write failure
  79 - General alerting engine error
        """
    )

    # Required arguments
    parser.add_argument(
        "--current-dashboard",
        type=Path,
        required=True,
        help="Path to current health dashboard JSON"
    )

    # Optional arguments
    parser.add_argument(
        "--previous-dashboard",
        type=Path,
        help="Path to previous dashboard JSON for trend alerts"
    )

    parser.add_argument(
        "--auto-discover-previous",
        action="store_true",
        help="Auto-discover previous dashboard in same directory"
    )

    parser.add_argument(
        "--output",
        type=Path,
        help="Output path for alerts JSON report"
    )

    # Channel configuration
    parser.add_argument(
        "--channels",
        type=str,
        default="console",
        help="Comma-separated list of channels: console,file,email,webhook (default: console)"
    )

    parser.add_argument(
        "--email-to",
        type=str,
        help="Email recipient for email channel"
    )

    parser.add_argument(
        "--webhook-url",
        type=str,
        help="Webhook URL for webhook channel"
    )

    # Thresholds
    parser.add_argument(
        "--severity-threshold",
        type=str,
        choices=["INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Minimum severity level to report (default: INFO)"
    )

    parser.add_argument(
        "--score-drop-threshold",
        type=float,
        default=10.0,
        help="Score drop (points) to trigger alert (default: 10.0)"
    )

    parser.add_argument(
        "--regression-threshold",
        type=int,
        default=3,
        help="New issues count to trigger regression alert (default: 3)"
    )

    # Failure modes
    parser.add_argument(
        "--fail-on-critical",
        action="store_true",
        default=True,
        help="Exit with code 72 on critical alerts (default: true)"
    )

    parser.add_argument(
        "--no-fail-on-critical",
        action="store_true",
        help="Don't fail on critical alerts"
    )

    parser.add_argument(
        "--fail-on-any-alert",
        action="store_true",
        help="Exit with code 71 on any alert"
    )

    # Output control
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress non-essential output"
    )

    args = parser.parse_args()

    # Validate current dashboard exists
    if not args.current_dashboard.exists():
        print(f"Error: Dashboard file not found: {args.current_dashboard}")
        return EXIT_INVALID_DASHBOARD

    # Handle previous dashboard
    previous_dashboard = args.previous_dashboard
    if not previous_dashboard and args.auto_discover_previous:
        previous_dashboard = discover_previous_dashboard(args.current_dashboard)
        if previous_dashboard:
            print(f"Auto-discovered previous dashboard: {previous_dashboard}")

    if previous_dashboard and not previous_dashboard.exists():
        print(f"Warning: Previous dashboard not found: {previous_dashboard}")
        previous_dashboard = None

    # Parse channels
    output_dir = args.output.parent if args.output else None
    channels = parse_channels(
        args.channels,
        output_dir=output_dir,
        email_to=args.email_to,
        webhook_url=args.webhook_url
    )

    if not channels:
        # Default to console if no valid channels
        channels = [ChannelConfig(channel_type=ChannelType.CONSOLE, enabled=True)]

    # Build configuration
    config = AlertingConfig(
        current_dashboard_path=args.current_dashboard,
        previous_dashboard_path=previous_dashboard,
        output_path=args.output,
        channels=channels,
        severity_threshold=AlertSeverity[args.severity_threshold],
        score_drop_threshold=args.score_drop_threshold,
        rapid_regression_threshold=args.regression_threshold,
        fail_on_critical=args.fail_on_critical and not args.no_fail_on_critical,
        fail_on_any_alert=args.fail_on_any_alert,
        verbose=args.verbose and not args.quiet
    )

    # Run alerting engine
    try:
        engine = AlertingEngine(config)
        report, exit_code = engine.run()

        # Print summary if not quiet
        if not args.quiet:
            print(f"\nAlert Summary:")
            print(f"  Total: {report.total_alerts}")
            print(f"  Critical: {report.critical_alerts}")
            print(f"  Error: {report.error_alerts}")
            print(f"  Warning: {report.warning_alerts}")
            print(f"  Info: {report.info_alerts}")
            print(f"\nExit Code: {exit_code}")

        return exit_code

    except KeyboardInterrupt:
        print("\nAborted by user")
        return EXIT_GENERAL_ALERTING_ERROR
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return EXIT_GENERAL_ALERTING_ERROR


if __name__ == "__main__":
    sys.exit(main())
