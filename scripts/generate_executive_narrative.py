#!/usr/bin/env python3
"""
Executive Narrative Generator

Creates a plain-English executive narrative summarizing a T.A.R.S. pipeline run.
Designed for human consumption by executives, stakeholders, and auditors.

Features:
    - Executive Summary (1-2 paragraphs)
    - Overall Health Status (GREEN / AMBER / RED)
    - SLA Status Summary
    - Key Risks Identified
    - Notable Trends / Propagation Signals
    - Recommended Next Actions

Usage:
    # Generate narrative for a completed run
    python scripts/generate_executive_narrative.py --run-dir ./reports/runs/tars-run-20251222-140000

    # Custom output path
    python scripts/generate_executive_narrative.py --run-dir ./reports/runs/tars-run-20251222-140000 \
        --output ./reports/executive-narrative.md

Exit Codes:
    0:   Success, narrative generated
    1:   Run directory not found
    2:   Critical reports missing (narrative incomplete)
    199: General error

Version: 1.0.0
Phase: 17 - Post-GA Observability
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Exit codes
EXIT_SUCCESS = 0
EXIT_RUN_DIR_NOT_FOUND = 1
EXIT_REPORTS_MISSING = 2
EXIT_GENERAL_ERROR = 199

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Health tier mappings
HEALTH_TIERS = {
    "GREEN": {"emoji": "GREEN", "description": "Healthy - All systems operating normally"},
    "AMBER": {"emoji": "AMBER", "description": "Caution - Some issues require attention"},
    "RED": {"emoji": "RED", "description": "Critical - Immediate action required"},
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


def load_json_report(run_dir: Path, filename: str) -> Optional[Dict[str, Any]]:
    """Load a JSON report from the run directory."""
    # Try direct path first
    report_path = run_dir / filename

    # Try alternative paths (structured format)
    alternative_paths = [
        run_dir / "executive" / filename,
        run_dir / "daily" / filename,
        run_dir / "weekly" / filename,
    ]

    for path in [report_path] + alternative_paths:
        if path.exists():
            try:
                with open(path, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to parse {path}: {e}")

    return None


def determine_overall_health(
    sla_report: Optional[Dict],
    org_health: Optional[Dict],
    alerts_report: Optional[Dict]
) -> Tuple[str, str]:
    """
    Determine overall health status based on available reports.

    Returns: (tier, explanation)
    """
    # Default to GREEN with uncertainty if no data
    if not sla_report and not org_health and not alerts_report:
        return "AMBER", "Insufficient data to determine health status"

    # Check SLA report for executive readiness
    if sla_report:
        exec_readiness = sla_report.get("executive_readiness", {})
        tier = exec_readiness.get("tier", "").upper()

        if tier in HEALTH_TIERS:
            return tier, f"Based on executive readiness score of {exec_readiness.get('readiness_score', 'N/A')}/100"

        # Check for breaches
        summary = sla_report.get("summary", {})
        if summary.get("breach_count", 0) > 0:
            return "RED", f"{summary.get('breach_count')} SLA breaches detected"
        if summary.get("at_risk_count", 0) > 0:
            return "AMBER", f"{summary.get('at_risk_count')} SLAs at risk"

    # Check alerts
    if alerts_report:
        alerts = alerts_report.get("alerts", [])
        critical_count = sum(1 for a in alerts if a.get("severity", "").lower() == "critical")
        if critical_count > 0:
            return "RED", f"{critical_count} critical alerts active"
        if len(alerts) > 0:
            return "AMBER", f"{len(alerts)} alerts require attention"

    # Check org health
    if org_health:
        risk_tier = org_health.get("risk_tier", "").upper()
        if risk_tier == "HIGH":
            return "RED", "High organization risk detected"
        if risk_tier == "MEDIUM":
            return "AMBER", "Medium organization risk detected"

    return "GREEN", "All systems operating within normal parameters"


def format_sla_summary(sla_report: Optional[Dict]) -> List[str]:
    """Format SLA status summary."""
    lines = []

    if not sla_report:
        lines.append("- SLA Intelligence report not available")
        return lines

    summary = sla_report.get("summary", {})
    exec_readiness = sla_report.get("executive_readiness", {})

    # Executive readiness
    if exec_readiness:
        score = exec_readiness.get("readiness_score", "N/A")
        tier = exec_readiness.get("tier", "N/A")
        lines.append(f"- **Executive Readiness Score:** {score}/100 ({tier})")

    # SLA counts
    total = summary.get("total_slas", 0)
    compliant = summary.get("compliant_count", 0)
    at_risk = summary.get("at_risk_count", 0)
    breached = summary.get("breach_count", 0)

    if total > 0:
        lines.append(f"- **Total SLAs Evaluated:** {total}")
        lines.append(f"  - Compliant: {compliant} ({compliant*100//total}%)" if total else f"  - Compliant: {compliant}")
        lines.append(f"  - At Risk: {at_risk}")
        lines.append(f"  - Breached: {breached}")

    # Compliance results summary
    compliance_results = sla_report.get("compliance_results", [])
    if compliance_results:
        breached_slas = [r for r in compliance_results if r.get("status") == "BREACHED"]
        if breached_slas:
            lines.append("")
            lines.append("**Breached SLAs:**")
            for sla in breached_slas[:5]:  # Limit to top 5
                sla_id = sla.get("sla_id", "Unknown")
                lines.append(f"  - {sla_id}")

    return lines


def format_key_risks(
    sla_report: Optional[Dict],
    temporal_report: Optional[Dict],
    trend_report: Optional[Dict]
) -> List[str]:
    """Format key risks identified."""
    lines = []
    risks_found = False

    # SLA breaches
    if sla_report:
        breaches = sla_report.get("breaches", [])
        if breaches:
            risks_found = True
            lines.append("**SLA Breach Risks:**")
            for breach in breaches[:3]:
                sla_id = breach.get("sla_id", "Unknown")
                root_causes = breach.get("root_causes", [])
                if root_causes:
                    cause = root_causes[0].get("cause", "Unknown cause")
                    lines.append(f"  - {sla_id}: {cause}")
                else:
                    lines.append(f"  - {sla_id}")
            lines.append("")

    # Temporal anomalies
    if temporal_report:
        anomalies = temporal_report.get("temporal_anomalies", [])
        if anomalies:
            risks_found = True
            lines.append("**Temporal Risk Signals:**")
            for anomaly in anomalies[:3]:
                anomaly_type = anomaly.get("type", "Unknown")
                description = anomaly.get("description", "")
                lines.append(f"  - {anomaly_type}: {description[:100]}")
            lines.append("")

    # Trend anomalies
    if trend_report:
        clusters = trend_report.get("clusters", [])
        anomalous_clusters = [c for c in clusters if c.get("is_anomalous")]
        if anomalous_clusters:
            risks_found = True
            lines.append("**Trend Anomaly Clusters:**")
            for cluster in anomalous_clusters[:3]:
                repos = cluster.get("repositories", [])
                lines.append(f"  - {len(repos)} repositories showing correlated decline")
            lines.append("")

    if not risks_found:
        lines.append("No significant risks identified in this analysis period.")

    return lines


def format_trends_and_propagation(
    temporal_report: Optional[Dict],
    trend_report: Optional[Dict]
) -> List[str]:
    """Format notable trends and propagation signals."""
    lines = []
    signals_found = False

    # Leader/follower relationships
    if temporal_report:
        influence_scores = temporal_report.get("influence_scores", [])
        leaders = [s for s in influence_scores if s.get("classification") == "LEADER"]

        if leaders:
            signals_found = True
            lines.append("**Leader Repositories (High Influence):**")
            for leader in leaders[:5]:
                repo = leader.get("repository", "Unknown")
                score = leader.get("influence_score", 0)
                lines.append(f"  - {repo} (influence score: {score})")
            lines.append("")

        # Propagation paths
        propagation_paths = temporal_report.get("propagation_paths", [])
        if propagation_paths:
            signals_found = True
            lines.append("**Active Propagation Paths:**")
            for path in propagation_paths[:3]:
                path_nodes = path.get("path", [])
                if len(path_nodes) >= 2:
                    lines.append(f"  - {' -> '.join(path_nodes[:5])}")
            lines.append("")

    # Correlation clusters
    if trend_report:
        clusters = trend_report.get("clusters", [])
        if clusters:
            signals_found = True
            lines.append("**Correlated Repository Groups:**")
            for i, cluster in enumerate(clusters[:3], 1):
                repos = cluster.get("repositories", [])
                strength = cluster.get("correlation_strength", "N/A")
                lines.append(f"  - Group {i}: {len(repos)} repositories (correlation: {strength})")
            lines.append("")

    if not signals_found:
        lines.append("No significant trends or propagation patterns detected.")

    return lines


def format_recommended_actions(
    health_tier: str,
    sla_report: Optional[Dict],
    temporal_report: Optional[Dict],
    alerts_report: Optional[Dict]
) -> List[str]:
    """Format recommended next actions based on findings."""
    lines = []
    actions = []

    # Tier-based recommendations
    if health_tier == "RED":
        actions.append("**IMMEDIATE:** Initiate incident response per playbook")
        actions.append("**IMMEDIATE:** Notify stakeholders of critical status")

    if health_tier == "AMBER":
        actions.append("Increase monitoring frequency for at-risk components")
        actions.append("Review and prioritize remediation backlog")

    # SLA-based recommendations
    if sla_report:
        summary = sla_report.get("summary", {})
        if summary.get("breach_count", 0) > 0:
            actions.append("Investigate root causes of SLA breaches")
            actions.append("Update SLA policy thresholds if appropriate")
        if summary.get("at_risk_count", 0) > 0:
            actions.append("Monitor at-risk SLAs closely in next 24-48 hours")

    # Temporal-based recommendations
    if temporal_report:
        anomalies = temporal_report.get("temporal_anomalies", [])
        if anomalies:
            actions.append("Investigate temporal anomalies for early warning signals")
        leaders = temporal_report.get("influence_scores", [])
        leaders = [s for s in leaders if s.get("classification") == "LEADER"]
        if leaders:
            actions.append("Prioritize health of high-influence leader repositories")

    # Alert-based recommendations
    if alerts_report:
        alerts = alerts_report.get("alerts", [])
        critical = [a for a in alerts if a.get("severity", "").lower() == "critical"]
        if critical:
            actions.append("Address all critical alerts before next business day")

    # Default actions
    if health_tier == "GREEN" and not actions:
        actions.append("Continue regular monitoring cadence")
        actions.append("Archive this report for trend analysis")

    for action in actions:
        lines.append(f"- {action}")

    return lines


def generate_executive_summary(
    health_tier: str,
    health_explanation: str,
    sla_report: Optional[Dict],
    run_timestamp: str
) -> List[str]:
    """Generate 1-2 paragraph executive summary."""
    lines = []

    # Build summary based on health tier
    tier_info = HEALTH_TIERS.get(health_tier, HEALTH_TIERS["AMBER"])

    if health_tier == "GREEN":
        lines.append(
            f"The T.A.R.S. organization health assessment completed on {run_timestamp} "
            f"indicates **healthy** operational status across monitored repositories and services. "
            f"{health_explanation}."
        )
    elif health_tier == "AMBER":
        lines.append(
            f"The T.A.R.S. organization health assessment completed on {run_timestamp} "
            f"indicates **caution** status requiring attention. {health_explanation}. "
            "While core services remain operational, proactive intervention is recommended "
            "to prevent escalation."
        )
    else:  # RED
        lines.append(
            f"**CRITICAL:** The T.A.R.S. organization health assessment completed on {run_timestamp} "
            f"indicates **critical** status requiring immediate attention. {health_explanation}. "
            "Incident response procedures should be initiated immediately."
        )

    lines.append("")

    # Add SLA context if available
    if sla_report:
        summary = sla_report.get("summary", {})
        total = summary.get("total_slas", 0)
        compliant = summary.get("compliant_count", 0)

        if total > 0:
            compliance_rate = (compliant * 100) // total
            lines.append(
                f"SLA compliance stands at {compliance_rate}% ({compliant}/{total} SLAs compliant). "
                f"Executive readiness score is "
                f"{sla_report.get('executive_readiness', {}).get('readiness_score', 'N/A')}/100, "
                f"placing the organization in the "
                f"**{sla_report.get('executive_readiness', {}).get('tier', 'UNKNOWN')}** tier "
                "for board reporting purposes."
            )

    return lines


class ExecutiveNarrativeGenerator:
    """Generates executive narratives from T.A.R.S. pipeline runs."""

    def __init__(
        self,
        run_dir: str,
        output_path: Optional[str] = None
    ):
        self.run_dir = Path(run_dir).resolve()

        # Output path defaults to executive-narrative.md inside run directory
        if output_path:
            self.output_path = Path(output_path).resolve()
        else:
            self.output_path = self.run_dir / "executive-narrative.md"

        # Reports
        self.sla_report: Optional[Dict] = None
        self.org_health: Optional[Dict] = None
        self.alerts_report: Optional[Dict] = None
        self.temporal_report: Optional[Dict] = None
        self.trend_report: Optional[Dict] = None
        self.manifest: Optional[Dict] = None

    def validate_run_dir(self) -> bool:
        """Validate that the run directory exists."""
        if not self.run_dir.exists():
            logger.error(f"Run directory does not exist: {self.run_dir}")
            return False

        if not self.run_dir.is_dir():
            logger.error(f"Run path is not a directory: {self.run_dir}")
            return False

        return True

    def load_reports(self) -> bool:
        """Load all available reports from the run directory."""
        # Load SLA Intelligence report
        self.sla_report = load_json_report(self.run_dir, "sla-intelligence-report.json")
        if self.sla_report:
            logger.info("Loaded SLA Intelligence report")

        # Load Org Health report
        self.org_health = load_json_report(self.run_dir, "org-health-report.json")
        if self.org_health:
            logger.info("Loaded Org Health report")

        # Load Alerts report
        self.alerts_report = load_json_report(self.run_dir, "org-alerts.json")
        if self.alerts_report:
            logger.info("Loaded Org Alerts report")

        # Load Temporal Intelligence report
        self.temporal_report = load_json_report(self.run_dir, "temporal-intelligence-report.json")
        if self.temporal_report:
            logger.info("Loaded Temporal Intelligence report")

        # Load Trend Correlation report
        self.trend_report = load_json_report(self.run_dir, "trend-correlation-report.json")
        if self.trend_report:
            logger.info("Loaded Trend Correlation report")

        # Load bundle manifest
        self.manifest = load_json_report(self.run_dir, "bundle-manifest.json")
        if self.manifest:
            logger.info("Loaded bundle manifest")

        # At least one report should be available
        reports_available = any([
            self.sla_report,
            self.org_health,
            self.alerts_report,
            self.temporal_report,
            self.trend_report
        ])

        if not reports_available:
            logger.warning("No reports found in run directory - narrative will be limited")

        return True

    def generate_narrative(self) -> str:
        """Generate the executive narrative markdown."""
        generation_time = datetime.now(timezone.utc)

        # Extract run timestamp from directory name or manifest
        run_timestamp = self.run_dir.name.replace("tars-run-", "")
        if self.manifest:
            run_timestamp = self.manifest.get("timestamp", run_timestamp)

        # Determine overall health
        health_tier, health_explanation = determine_overall_health(
            self.sla_report,
            self.org_health,
            self.alerts_report
        )

        # Build narrative
        lines = []

        # Header
        lines.append("# T.A.R.S. Executive Narrative Report")
        lines.append("")
        lines.append(f"**Generated:** {generation_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        lines.append(f"**Run Timestamp:** {run_timestamp}")
        lines.append(f"**T.A.R.S. Version:** {get_tars_version()}")
        lines.append(f"**Report Type:** Executive Summary for Leadership Review")
        lines.append("")
        lines.append("---")
        lines.append("")

        # Overall Health Status
        tier_info = HEALTH_TIERS.get(health_tier, HEALTH_TIERS["AMBER"])
        lines.append("## Overall Health Status")
        lines.append("")
        lines.append(f"**Status: {tier_info['emoji']}**")
        lines.append("")
        lines.append(f"*{tier_info['description']}*")
        lines.append("")
        lines.append("---")
        lines.append("")

        # Executive Summary
        lines.append("## Executive Summary")
        lines.append("")
        lines.extend(generate_executive_summary(
            health_tier,
            health_explanation,
            self.sla_report,
            run_timestamp
        ))
        lines.append("")
        lines.append("---")
        lines.append("")

        # SLA Status Summary
        lines.append("## SLA Status Summary")
        lines.append("")
        lines.extend(format_sla_summary(self.sla_report))
        lines.append("")
        lines.append("---")
        lines.append("")

        # Key Risks Identified
        lines.append("## Key Risks Identified")
        lines.append("")
        lines.extend(format_key_risks(
            self.sla_report,
            self.temporal_report,
            self.trend_report
        ))
        lines.append("")
        lines.append("---")
        lines.append("")

        # Notable Trends / Propagation Signals
        lines.append("## Notable Trends & Propagation Signals")
        lines.append("")
        lines.extend(format_trends_and_propagation(
            self.temporal_report,
            self.trend_report
        ))
        lines.append("")
        lines.append("---")
        lines.append("")

        # Recommended Next Actions
        lines.append("## Recommended Next Actions")
        lines.append("")
        lines.extend(format_recommended_actions(
            health_tier,
            self.sla_report,
            self.temporal_report,
            self.alerts_report
        ))
        lines.append("")
        lines.append("---")
        lines.append("")

        # Footer
        lines.append("*This narrative was automatically generated by T.A.R.S. Executive Narrative Generator v1.0.*")
        lines.append("")
        lines.append("*For detailed technical information, refer to the full JSON reports in the run directory.*")
        lines.append("")

        return "\n".join(lines)

    def write_narrative(self, narrative: str) -> bool:
        """Write narrative to output file."""
        try:
            # Ensure parent directory exists
            self.output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(self.output_path, "w", encoding="utf-8") as f:
                f.write(narrative)

            logger.info(f"Written executive narrative: {self.output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to write narrative: {e}")
            return False

    def generate(self) -> int:
        """Generate executive narrative."""
        logger.info("=" * 60)
        logger.info("T.A.R.S. Executive Narrative Generator v1.0")
        logger.info("=" * 60)
        logger.info(f"Run Directory: {self.run_dir}")
        logger.info(f"Output Path: {self.output_path}")
        logger.info("")

        # Validate run directory
        if not self.validate_run_dir():
            return EXIT_RUN_DIR_NOT_FOUND

        # Load reports
        self.load_reports()

        # Generate narrative
        logger.info("Generating executive narrative...")
        narrative = self.generate_narrative()

        # Write narrative
        if not self.write_narrative(narrative):
            return EXIT_REPORTS_MISSING

        # Summary
        logger.info("")
        logger.info("=" * 60)
        logger.info("Narrative Generation Complete")
        logger.info("=" * 60)
        logger.info(f"Output: {self.output_path}")
        logger.info(f"Reports Loaded: SLA={bool(self.sla_report)}, "
                    f"OrgHealth={bool(self.org_health)}, "
                    f"Alerts={bool(self.alerts_report)}, "
                    f"Temporal={bool(self.temporal_report)}, "
                    f"Trends={bool(self.trend_report)}")

        return EXIT_SUCCESS


def create_argument_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        prog="generate_executive_narrative",
        description="Generate executive narrative summaries from T.A.R.S. pipeline runs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exit Codes:
  0      Success, narrative generated
  1      Run directory not found
  2      Critical reports missing
  199    General error

Examples:
  # Generate narrative for a completed run
  python scripts/generate_executive_narrative.py --run-dir ./reports/runs/tars-run-20251222-140000

  # Custom output location
  python scripts/generate_executive_narrative.py --run-dir ./reports/runs/tars-run-20251222-140000 \\
      --output ./reports/executive-narrative.md
"""
    )

    # Required arguments
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Path to a completed orchestrator run directory"
    )

    # Optional arguments
    parser.add_argument(
        "--output",
        default=None,
        help="Custom output path for executive-narrative.md (default: <run-dir>/executive-narrative.md)"
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

    try:
        generator = ExecutiveNarrativeGenerator(
            run_dir=args.run_dir,
            output_path=args.output
        )

        return generator.generate()

    except KeyboardInterrupt:
        logger.info("Narrative generation interrupted by user")
        return EXIT_GENERAL_ERROR
    except Exception as e:
        logger.error(f"Narrative generation error: {e}")
        return EXIT_GENERAL_ERROR


if __name__ == "__main__":
    sys.exit(main())
