#!/usr/bin/env python3
"""
Organization-Level Alerting & Escalation CLI

Command-line interface for the org-level alerting and escalation engine.

Usage:
    python -m analytics.run_org_alerts --org-report ./org-health-report.json

Exit Codes (100-109):
    100: Success, no alerts
    101: Alerts present (non-critical)
    102: Critical alerts present
    103: Config error
    104: Unable to parse org-health-report.json
    105: Routing failure
    199: General alerting error

Version: 1.0.0
Phase: 14.8 Task 2
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional, List

from analytics.org_alerting_engine import (
    OrgAlertConfig,
    OrgAlertingEngine,
    OrgAlertReport,
    OrgAlert,
    OrgAlertChannelConfig,
    OrgAlertChannelType,
    EscalationRule,
    AlertSeverity,
    OrgAlertThresholds,
    create_default_escalation_rules,
    create_default_channels,
    load_escalation_config,
    EXIT_ORG_ALERT_SUCCESS,
    EXIT_ALERTS_PRESENT,
    EXIT_CRITICAL_ALERTS,
    EXIT_ALERTING_CONFIG_ERROR,
    EXIT_ORG_REPORT_PARSE_ERROR,
    EXIT_ROUTING_FAILURE,
    EXIT_GENERAL_ALERTING_ERROR,
)

logger = logging.getLogger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        prog="run_org_alerts",
        description="Organization-Level Alerting & Escalation Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exit Codes:
  100  Success, no alerts
  101  Alerts present (non-critical)
  102  Critical alerts present
  103  Config error
  104  Unable to parse org-health-report.json
  105  Routing failure
  199  General alerting error

Examples:
  # Basic alerting from org health report
  python -m analytics.run_org_alerts --org-report ./org-health-report.json

  # With custom configuration
  python -m analytics.run_org_alerts --org-report ./org-health-report.json \\
      --config ./org-alerting-config.yaml

  # Output to JSON file
  python -m analytics.run_org_alerts --org-report ./org-health-report.json \\
      --output ./org-alerts.json

  # CI/CD mode - fail on critical alerts
  python -m analytics.run_org_alerts --org-report ./org-health-report.json \\
      --fail-on-critical

  # CI/CD mode - fail on any alerts
  python -m analytics.run_org_alerts --org-report ./org-health-report.json \\
      --fail-on-any-alerts

  # JSON output to stdout
  python -m analytics.run_org_alerts --org-report ./org-health-report.json \\
      --json

  # Quick summary
  python -m analytics.run_org_alerts --org-report ./org-health-report.json \\
      --summary-only

Alert Categories:
  SLO       - SLO/SLA policy violations
  RISK      - High-risk repository alerts
  TREND     - Org-wide trend signals (declining, low green %)
  INTEGRITY - Data integrity issues (load errors)

Escalation Actions:
  escalate_to:oncall     - Escalate to on-call team
  notify:slack:channel   - Send Slack notification
  notify:email:recipient - Send email notification
  log                    - Log the escalation
"""
    )

    # Required arguments
    parser.add_argument(
        "--org-report",
        type=Path,
        required=True,
        help="Path to org-health-report.json from Task 1"
    )

    # Configuration
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to YAML or JSON alerting configuration file"
    )

    # Output
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for org alerts report JSON"
    )

    # Behavior flags
    parser.add_argument(
        "--fail-on-critical",
        action="store_true",
        help="Exit with code 102 if any critical alerts are generated"
    )

    parser.add_argument(
        "--fail-on-any-alerts",
        action="store_true",
        help="Exit with code 101 if any alerts are generated"
    )

    parser.add_argument(
        "--use-default-escalations",
        action="store_true",
        help="Use default escalation rules if no config provided"
    )

    # Alert generation control
    parser.add_argument(
        "--no-slo-alerts",
        action="store_true",
        help="Disable SLO violation alerts"
    )

    parser.add_argument(
        "--no-risk-alerts",
        action="store_true",
        help="Disable high-risk repository alerts"
    )

    parser.add_argument(
        "--no-trend-alerts",
        action="store_true",
        help="Disable org-wide trend alerts"
    )

    parser.add_argument(
        "--no-integrity-alerts",
        action="store_true",
        help="Disable data integrity alerts"
    )

    # Threshold overrides
    parser.add_argument(
        "--declining-warning",
        type=float,
        default=0.20,
        help="Percent declining repos for warning (default: 0.20)"
    )

    parser.add_argument(
        "--declining-critical",
        type=float,
        default=0.40,
        help="Percent declining repos for critical (default: 0.40)"
    )

    parser.add_argument(
        "--green-warning",
        type=float,
        default=0.60,
        help="Percent green repos below which is warning (default: 0.60)"
    )

    parser.add_argument(
        "--green-critical",
        type=float,
        default=0.40,
        help="Percent green repos below which is critical (default: 0.40)"
    )

    parser.add_argument(
        "--score-warning",
        type=float,
        default=70.0,
        help="Average score below which is warning (default: 70.0)"
    )

    parser.add_argument(
        "--score-critical",
        type=float,
        default=50.0,
        help="Average score below which is critical (default: 50.0)"
    )

    # Output options
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only print a quick summary (no detailed alert output)"
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

    # Channel options
    parser.add_argument(
        "--no-console",
        action="store_true",
        help="Disable console output channel"
    )

    return parser


def print_summary(report: OrgAlertReport) -> None:
    """Print a formatted summary of the org alert report."""
    print("\n" + "=" * 80)
    print("ORGANIZATION ALERT SUMMARY")
    print("=" * 80)

    # Source org health info
    status_icon = {
        "green": "[OK]",
        "yellow": "[WARN]",
        "red": "[CRITICAL]"
    }.get(report.org_health_status, "[?]")

    print(f"\n{status_icon} Source Org Health: {report.org_health_status.upper()}")
    print(f"    Score: {report.org_health_score:.1f}/100")
    print(f"    Risk Tier: {report.org_risk_tier.upper()}")
    print(f"    Repos: {report.total_repos}")

    # Alert summary
    print("\n" + "-" * 40)
    print("ALERT SUMMARY")
    print("-" * 40)
    print(f"  Total Alerts: {report.total_alerts}")
    print(f"  Critical: {report.critical_alerts}")
    print(f"  High: {report.high_alerts}")
    print(f"  Medium: {report.medium_alerts}")
    print(f"  Low: {report.low_alerts}")

    # By category
    print("\n  By Category:")
    print(f"    SLO Violations: {report.slo_alerts}")
    print(f"    Risk Alerts: {report.risk_alerts}")
    print(f"    Trend Alerts: {report.trend_alerts}")
    print(f"    Integrity Alerts: {report.integrity_alerts}")

    # Escalations
    print("\n" + "-" * 40)
    print("ESCALATIONS")
    print("-" * 40)
    print(f"  Triggered: {report.escalations_triggered}")
    if report.escalation_actions:
        print("\n  Actions Taken:")
        for action in report.escalation_actions[:5]:
            print(f"    - [{action.get('action_type', 'unknown')}] {action.get('message', '')}")
        if len(report.escalation_actions) > 5:
            print(f"    ... and {len(report.escalation_actions) - 5} more")

    # Routing status
    if report.channels_dispatched:
        print("\n" + "-" * 40)
        print("ROUTING STATUS")
        print("-" * 40)
        print(f"  Dispatched to: {', '.join(report.channels_dispatched)}")
        if report.dispatch_errors:
            print(f"  Errors: {len(report.dispatch_errors)}")
            for error in report.dispatch_errors[:3]:
                print(f"    - {error.get('channel', 'unknown')}: {error.get('error', 'unknown')}")

    # Metadata
    print("\n" + "=" * 80)
    print(f"Generated: {report.generated_at}")
    print(f"Duration: {report.evaluation_duration_ms:.1f}ms")
    print("=" * 80)


def print_detailed_report(report: OrgAlertReport) -> None:
    """Print detailed report including all alerts."""
    print_summary(report)

    if not report.alerts:
        print("\nNo alerts generated.")
        return

    # All alerts by severity
    print("\n" + "=" * 80)
    print("ALERT DETAILS")
    print("=" * 80)

    severity_order = ["critical", "high", "medium", "low"]
    severity_icons = {
        "critical": "[CRIT]",
        "high": "[HIGH]",
        "medium": "[MED]",
        "low": "[LOW]"
    }

    for severity in severity_order:
        alerts = [a for a in report.alerts if a.get("severity") == severity]
        if not alerts:
            continue

        icon = severity_icons.get(severity, "[?]")
        print(f"\n{icon} {severity.upper()} ({len(alerts)})")
        print("-" * 60)

        for alert in alerts:
            category = alert.get("category", "unknown").upper()
            print(f"\n  [{category}] {alert.get('title', 'Unknown')}")
            print(f"    {alert.get('message', '')}")

            # Show affected repos if any
            affected = alert.get("affected_repos", [])
            if affected:
                repos_str = ", ".join(affected[:3])
                if len(affected) > 3:
                    repos_str += f" (+{len(affected) - 3} more)"
                print(f"    Repos: {repos_str}")

            # Show recommendations if any
            recommendations = alert.get("recommendations", [])
            if recommendations:
                print(f"    -> {recommendations[0]}")

            # Show escalation status
            if alert.get("escalated"):
                actions = alert.get("escalation_actions", [])
                print(f"    Escalated: {', '.join(actions[:2])}")


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
        # Validate org report path
        org_report_path = args.org_report
        if not org_report_path.exists():
            logger.error(f"Org health report not found: {org_report_path}")
            return EXIT_ORG_REPORT_PARSE_ERROR

        # Build channels
        channels: List[OrgAlertChannelConfig] = []

        if not args.no_console and not args.json:
            channels.append(OrgAlertChannelConfig(
                channel_type=OrgAlertChannelType.CONSOLE,
                enabled=True
            ))

        if args.output:
            channels.append(OrgAlertChannelConfig(
                channel_type=OrgAlertChannelType.JSON_FILE,
                enabled=True,
                output_path=args.output
            ))

        if args.json:
            channels.append(OrgAlertChannelConfig(
                channel_type=OrgAlertChannelType.STDOUT,
                enabled=True
            ))

        # Build escalation rules
        escalation_rules: List[EscalationRule] = []
        if args.config:
            if not args.config.exists():
                logger.error(f"Config file does not exist: {args.config}")
                return EXIT_ALERTING_CONFIG_ERROR
            try:
                escalation_rules = load_escalation_config(args.config)
                logger.info(f"Loaded {len(escalation_rules)} escalation rules from {args.config}")
            except Exception as e:
                logger.error(f"Failed to load config: {e}")
                return EXIT_ALERTING_CONFIG_ERROR
        elif args.use_default_escalations:
            escalation_rules = create_default_escalation_rules()
            logger.info(f"Using {len(escalation_rules)} default escalation rules")

        # Build thresholds
        thresholds = OrgAlertThresholds(
            percent_declining_warning=args.declining_warning,
            percent_declining_critical=args.declining_critical,
            percent_green_warning=args.green_warning,
            percent_green_critical=args.green_critical,
            avg_score_warning=args.score_warning,
            avg_score_critical=args.score_critical
        )

        # Build config
        config = OrgAlertConfig(
            org_report_path=org_report_path,
            output_path=args.output,
            channels=channels,
            escalation_rules=escalation_rules,
            thresholds=thresholds,
            fail_on_critical=args.fail_on_critical,
            fail_on_any_alerts=args.fail_on_any_alerts,
            verbose=args.verbose,
            generate_slo_alerts=not args.no_slo_alerts,
            generate_risk_alerts=not args.no_risk_alerts,
            generate_trend_alerts=not args.no_trend_alerts,
            generate_integrity_alerts=not args.no_integrity_alerts
        )

        # Run alerting engine
        engine = OrgAlertingEngine(config)
        report, exit_code = engine.run()

        # Output results
        if args.json:
            # JSON already output via stdout channel
            pass
        elif not args.quiet:
            if args.summary_only:
                print_summary(report)
            else:
                print_detailed_report(report)

        return exit_code

    except KeyboardInterrupt:
        print("\nAlerting cancelled by user")
        return EXIT_GENERAL_ALERTING_ERROR
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return EXIT_GENERAL_ALERTING_ERROR


if __name__ == "__main__":
    sys.exit(main())
