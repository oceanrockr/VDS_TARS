#!/usr/bin/env python3
"""
T.A.R.S. Operations Notification Hook

Provides opt-in notification capabilities for T.A.R.S. governance pipeline runs.
Supports webhook, Slack, and PagerDuty (stub) destinations.

IMPORTANT: This script NEVER fails the pipeline. All errors are logged as warnings
and the script exits 0 to avoid breaking automated workflows.

Usage:
    # Notify via webhook
    python scripts/notify_ops.py \
        --run-dir ./reports/runs/tars-run-20251225-080000 \
        --webhook-url https://example.com/webhook \
        --severity SEV-2 \
        --title "SLA Breach Detected" \
        --message "Repository X breached availability SLA"

    # Notify via Slack
    python scripts/notify_ops.py \
        --run-dir ./reports/runs/tars-run-20251225-080000 \
        --slack-webhook-url https://hooks.slack.com/services/... \
        --severity SEV-1 \
        --title "Critical Alert"

    # Notify from orchestrator exit code
    python scripts/notify_ops.py \
        --exit-code 142 \
        --run-dir ./reports/runs/tars-run-20251225-080000 \
        --webhook-url https://example.com/webhook

Exit Codes:
    0:   Success (or graceful failure - never breaks pipeline)

Version: 1.0.0
Phase: 18 - Ops Integrations
"""

import argparse
import json
import logging
import os
import sys
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Exit code to severity mapping
EXIT_CODE_SEVERITY_MAP = {
    0: "INFO",
    92: "SEV-2",    # High org risk
    102: "SEV-1",   # Critical alerts
    122: "SEV-2",   # Critical anomaly
    132: "SEV-1",   # Propagation risk
    141: "SEV-3",   # At-risk SLAs
    142: "SEV-1",   # SLA breach
    199: "SEV-3",   # General error
}

# Exit code to recommended action mapping
EXIT_CODE_ACTION_MAP = {
    0: "No action required - pipeline completed successfully.",
    92: "Review org health report and identify failing repositories.",
    102: "Follow Incident Playbook SEV-1 procedures immediately.",
    122: "Investigate correlation clusters for synchronized decline.",
    132: "Isolate leader repositories and consider freezing deployments.",
    141: "Increase monitoring frequency for at-risk SLAs.",
    142: "Initiate incident response and notify stakeholders.",
    199: "Check logs, verify configuration, and re-run if needed.",
}

# Exit code to title mapping
EXIT_CODE_TITLE_MAP = {
    0: "T.A.R.S. Pipeline Success",
    92: "High Organization Risk Detected",
    102: "Critical Alerts Generated",
    122: "Critical Correlation Anomaly",
    132: "Propagation Risk Detected",
    141: "SLAs At Risk",
    142: "SLA Breach Detected",
    199: "Pipeline Error",
}


def get_tars_version() -> str:
    """Get T.A.R.S. version from VERSION file."""
    version_file = Path(__file__).parent.parent / "VERSION"
    if version_file.exists():
        try:
            return version_file.read_text().strip()
        except Exception:
            pass
    return "unknown"


def load_run_metadata(run_dir: Path) -> Dict[str, Any]:
    """Load run metadata from run-metadata.json if available."""
    metadata_path = run_dir / "run-metadata.json"
    if metadata_path.exists():
        try:
            with open(metadata_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load run metadata: {e}")
    return {}


def load_executive_readiness(run_dir: Path) -> Optional[str]:
    """Extract executive readiness tier from SLA report."""
    sla_path = run_dir / "sla-intelligence-report.json"
    if sla_path.exists():
        try:
            with open(sla_path, "r") as f:
                sla_report = json.load(f)
            readiness = sla_report.get("executive_readiness", {})
            return readiness.get("tier", "UNKNOWN")
        except Exception:
            pass
    return None


def build_notification_payload(
    title: str,
    message: str,
    severity: str,
    exit_code: Optional[int],
    run_dir: Optional[Path],
    extra_metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Build the notification payload."""
    timestamp = datetime.now(timezone.utc).isoformat()
    version = get_tars_version()

    payload = {
        "source": "T.A.R.S.",
        "version": version,
        "timestamp": timestamp,
        "severity": severity,
        "title": title,
        "message": message,
    }

    if exit_code is not None:
        payload["exit_code"] = exit_code
        payload["recommended_action"] = EXIT_CODE_ACTION_MAP.get(
            exit_code, "Review exit code in OPS_RUNBOOK.md"
        )

    if run_dir and run_dir.exists():
        payload["run_dir"] = str(run_dir)

        # Try to load metadata
        metadata = load_run_metadata(run_dir)
        if metadata:
            payload["metadata"] = {
                "tars_version": metadata.get("tars_version"),
                "git_commit": metadata.get("git_commit"),
                "duration_seconds": metadata.get("duration_seconds"),
            }

        # Try to get executive readiness tier
        tier = load_executive_readiness(run_dir)
        if tier:
            payload["executive_readiness_tier"] = tier

    if extra_metadata:
        payload["extra"] = extra_metadata

    return payload


def send_webhook(url: str, payload: Dict[str, Any], timeout: int = 30) -> bool:
    """Send notification via generic webhook (POST JSON)."""
    try:
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "User-Agent": f"T.A.R.S/{get_tars_version()}",
            },
            method="POST"
        )

        with urllib.request.urlopen(request, timeout=timeout) as response:
            status_code = response.getcode()
            if 200 <= status_code < 300:
                logger.info(f"Webhook notification sent successfully (status: {status_code})")
                return True
            else:
                logger.warning(f"Webhook returned non-2xx status: {status_code}")
                return False

    except urllib.error.URLError as e:
        logger.warning(f"Webhook request failed: {e}")
        return False
    except Exception as e:
        logger.warning(f"Webhook error: {e}")
        return False


def send_slack_webhook(url: str, payload: Dict[str, Any], timeout: int = 30) -> bool:
    """Send notification via Slack webhook."""
    # Transform payload to Slack format
    severity = payload.get("severity", "INFO")
    title = payload.get("title", "T.A.R.S. Notification")
    message = payload.get("message", "")

    # Color based on severity
    color_map = {
        "SEV-1": "#FF0000",  # Red
        "SEV-2": "#FFA500",  # Orange
        "SEV-3": "#FFFF00",  # Yellow
        "SEV-4": "#00FF00",  # Green
        "INFO": "#0000FF",   # Blue
    }
    color = color_map.get(severity, "#808080")

    # Build Slack message
    fields = []

    if payload.get("exit_code") is not None:
        fields.append({
            "title": "Exit Code",
            "value": str(payload["exit_code"]),
            "short": True
        })

    if payload.get("executive_readiness_tier"):
        fields.append({
            "title": "Readiness Tier",
            "value": payload["executive_readiness_tier"],
            "short": True
        })

    if payload.get("recommended_action"):
        fields.append({
            "title": "Recommended Action",
            "value": payload["recommended_action"],
            "short": False
        })

    if payload.get("run_dir"):
        fields.append({
            "title": "Run Directory",
            "value": payload["run_dir"],
            "short": False
        })

    slack_payload = {
        "attachments": [
            {
                "color": color,
                "title": f"[{severity}] {title}",
                "text": message,
                "fields": fields,
                "footer": f"T.A.R.S. v{payload.get('version', 'unknown')}",
                "ts": int(datetime.now(timezone.utc).timestamp())
            }
        ]
    }

    return send_webhook(url, slack_payload, timeout)


def send_pagerduty(routing_key: str, payload: Dict[str, Any], timeout: int = 30) -> bool:
    """Send notification via PagerDuty Events API v2 (stub implementation)."""
    logger.warning("PagerDuty integration is a placeholder - not fully implemented")

    # PagerDuty severity mapping
    pd_severity_map = {
        "SEV-1": "critical",
        "SEV-2": "error",
        "SEV-3": "warning",
        "SEV-4": "info",
        "INFO": "info",
    }

    severity = pd_severity_map.get(payload.get("severity", "INFO"), "info")

    pd_payload = {
        "routing_key": routing_key,
        "event_action": "trigger",
        "payload": {
            "summary": f"{payload.get('title', 'T.A.R.S. Alert')}: {payload.get('message', '')}",
            "severity": severity,
            "source": "T.A.R.S.",
            "custom_details": payload
        }
    }

    url = "https://events.pagerduty.com/v2/enqueue"

    try:
        data = json.dumps(pd_payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "User-Agent": f"T.A.R.S/{get_tars_version()}",
            },
            method="POST"
        )

        with urllib.request.urlopen(request, timeout=timeout) as response:
            status_code = response.getcode()
            if 200 <= status_code < 300:
                logger.info(f"PagerDuty notification sent (status: {status_code})")
                return True
            else:
                logger.warning(f"PagerDuty returned non-2xx: {status_code}")
                return False

    except Exception as e:
        logger.warning(f"PagerDuty error: {e}")
        return False


def create_argument_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        prog="notify_ops",
        description="Send T.A.R.S. pipeline notifications to external systems",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exit Codes:
  0      Always (notifications never fail the pipeline)

Severity Levels:
  SEV-1  Critical - immediate action required
  SEV-2  High - action required within 1 hour
  SEV-3  Medium - action required within 24 hours
  SEV-4  Low - informational

Examples:
  # Notify via webhook
  python scripts/notify_ops.py \\
      --webhook-url https://example.com/webhook \\
      --severity SEV-2 \\
      --title "Alert" \\
      --message "Something happened"

  # Auto-notify based on exit code
  python scripts/notify_ops.py \\
      --exit-code 142 \\
      --run-dir ./reports/runs/tars-run-20251225-080000 \\
      --webhook-url https://example.com/webhook
"""
    )

    # Destination options
    parser.add_argument(
        "--webhook-url",
        help="Generic webhook URL (POST JSON)"
    )

    parser.add_argument(
        "--slack-webhook-url",
        help="Slack webhook URL"
    )

    parser.add_argument(
        "--pagerduty-routing-key",
        help="PagerDuty routing key (stub implementation)"
    )

    # Content options
    parser.add_argument(
        "--title",
        help="Notification title (auto-generated from exit code if not provided)"
    )

    parser.add_argument(
        "--message",
        default="",
        help="Notification message body"
    )

    parser.add_argument(
        "--severity",
        choices=["SEV-1", "SEV-2", "SEV-3", "SEV-4", "INFO"],
        help="Severity level (auto-determined from exit code if not provided)"
    )

    parser.add_argument(
        "--exit-code",
        type=int,
        help="Pipeline exit code (for auto-generating severity and title)"
    )

    # Context options
    parser.add_argument(
        "--run-dir",
        help="Path to pipeline run directory (for metadata extraction)"
    )

    parser.add_argument(
        "--metadata-path",
        help="Path to run-metadata.json (alternative to --run-dir)"
    )

    # Behavior options
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print payload without sending"
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Request timeout in seconds (default: 30)"
    )

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

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Determine severity and title from exit code if not provided
    exit_code = args.exit_code
    severity = args.severity
    title = args.title

    if exit_code is not None:
        if not severity:
            severity = EXIT_CODE_SEVERITY_MAP.get(exit_code, "SEV-3")
        if not title:
            title = EXIT_CODE_TITLE_MAP.get(exit_code, f"T.A.R.S. Exit Code {exit_code}")
    else:
        if not severity:
            severity = "INFO"
        if not title:
            title = "T.A.R.S. Notification"

    # Parse run directory
    run_dir = Path(args.run_dir) if args.run_dir else None

    # Build payload
    payload = build_notification_payload(
        title=title,
        message=args.message,
        severity=severity,
        exit_code=exit_code,
        run_dir=run_dir
    )

    # Dry run mode
    if args.dry_run:
        logger.info("Dry run - payload would be:")
        print(json.dumps(payload, indent=2))
        return 0

    # Check if any destination is configured
    has_destination = any([
        args.webhook_url,
        args.slack_webhook_url,
        args.pagerduty_routing_key
    ])

    if not has_destination:
        logger.warning("No notification destination configured. Use --webhook-url, --slack-webhook-url, or --pagerduty-routing-key.")
        return 0  # Never fail

    # Send notifications
    success_count = 0
    total_count = 0

    if args.webhook_url:
        total_count += 1
        if send_webhook(args.webhook_url, payload, args.timeout):
            success_count += 1

    if args.slack_webhook_url:
        total_count += 1
        if send_slack_webhook(args.slack_webhook_url, payload, args.timeout):
            success_count += 1

    if args.pagerduty_routing_key:
        total_count += 1
        if send_pagerduty(args.pagerduty_routing_key, payload, args.timeout):
            success_count += 1

    logger.info(f"Notifications sent: {success_count}/{total_count}")

    # Always return 0 - never break pipeline
    return 0


if __name__ == "__main__":
    sys.exit(main())
