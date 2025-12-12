"""
HTML Renderer - Generates Human-Readable HTML Dashboards

This module creates visually appealing HTML dashboards with:
- Status badges (green/yellow/red)
- Summary statistics tables
- Issue severity breakdowns
- Version-by-version health cards
- Timeline of repairs, rollbacks, and scans
- Recommendations and action items

Version: 1.0.0
Phase: 14.7 Task 8
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from analytics.report_aggregator import AggregatedData, NormalizedIssue, NormalizedVersion

logger = logging.getLogger(__name__)


class HTMLRenderer:
    """
    Generates HTML dashboards from aggregated repository data.

    This class:
    1. Formats aggregated data into HTML components
    2. Generates status badges and severity indicators
    3. Creates tables, charts, and timelines
    4. Applies CSS styling for readability
    5. Outputs self-contained HTML files
    """

    def __init__(self, verbose: bool = False):
        """
        Initialize the HTML renderer.

        Args:
            verbose: Enable verbose logging
        """
        self.verbose = verbose
        if verbose:
            logger.setLevel(logging.DEBUG)

    def render_dashboard(
        self,
        aggregated_data: AggregatedData,
        health_score: float,
        overall_health: str,
        recommendations: List[str],
        output_path: Path
    ) -> bool:
        """
        Render complete HTML dashboard.

        Args:
            aggregated_data: Aggregated repository data
            health_score: Overall repository health score (0-100)
            overall_health: Overall health status (green/yellow/red)
            recommendations: List of recommended actions
            output_path: Path to write HTML file

        Returns:
            True if successful, False otherwise
        """
        try:
            html_content = self._generate_html(
                aggregated_data,
                health_score,
                overall_health,
                recommendations
            )

            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)

            logger.info(f"HTML dashboard written to: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to render HTML dashboard: {e}")
            return False

    def _generate_html(
        self,
        data: AggregatedData,
        health_score: float,
        overall_health: str,
        recommendations: List[str]
    ) -> str:
        """Generate complete HTML document."""
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>T.A.R.S. Repository Health Dashboard</title>
    <style>
        {self._get_css_styles()}
    </style>
</head>
<body>
    <div class="container">
        {self._render_header(data, health_score, overall_health)}
        {self._render_summary_stats(data)}
        {self._render_health_breakdown(data)}
        {self._render_version_cards(data)}
        {self._render_issues_table(data)}
        {self._render_timeline(data)}
        {self._render_recommendations(recommendations)}
        {self._render_footer(data)}
    </div>
</body>
</html>"""

    def _get_css_styles(self) -> str:
        """Get CSS styles for the dashboard."""
        return """
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: #f5f7fa;
            color: #333;
            line-height: 1.6;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }

        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            border-radius: 12px;
            margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }

        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            font-weight: 700;
        }

        .header .subtitle {
            font-size: 1.1em;
            opacity: 0.9;
        }

        .health-badge {
            display: inline-block;
            padding: 12px 24px;
            border-radius: 8px;
            font-size: 1.2em;
            font-weight: 600;
            margin-top: 20px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        .health-green {
            background: #10b981;
            color: white;
        }

        .health-yellow {
            background: #f59e0b;
            color: white;
        }

        .health-red {
            background: #ef4444;
            color: white;
        }

        .health-score {
            font-size: 3em;
            font-weight: 700;
            margin: 20px 0;
        }

        .card {
            background: white;
            border-radius: 12px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.07);
        }

        .card h2 {
            font-size: 1.8em;
            margin-bottom: 20px;
            color: #1f2937;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
        }

        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }

        .stat-box {
            background: linear-gradient(135deg, #f0f4f8 0%, #e2e8f0 100%);
            padding: 25px;
            border-radius: 10px;
            text-align: center;
            border-left: 5px solid #667eea;
        }

        .stat-value {
            font-size: 2.5em;
            font-weight: 700;
            color: #667eea;
            margin-bottom: 5px;
        }

        .stat-label {
            font-size: 0.9em;
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .severity-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }

        .severity-box {
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            color: white;
            font-weight: 600;
        }

        .severity-critical {
            background: #ef4444;
        }

        .severity-error {
            background: #f97316;
        }

        .severity-warning {
            background: #f59e0b;
        }

        .severity-info {
            background: #3b82f6;
        }

        .version-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }

        .version-card {
            background: #f9fafb;
            border-radius: 10px;
            padding: 20px;
            border-left: 5px solid #667eea;
        }

        .version-card.health-green {
            border-left-color: #10b981;
        }

        .version-card.health-yellow {
            border-left-color: #f59e0b;
        }

        .version-card.health-red {
            border-left-color: #ef4444;
        }

        .version-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }

        .version-name {
            font-size: 1.3em;
            font-weight: 600;
            color: #1f2937;
        }

        .version-status {
            padding: 5px 12px;
            border-radius: 5px;
            font-size: 0.8em;
            font-weight: 600;
            text-transform: uppercase;
        }

        .version-meta {
            font-size: 0.9em;
            color: #64748b;
            margin-bottom: 10px;
        }

        .version-badges {
            display: flex;
            gap: 8px;
            margin-top: 10px;
        }

        .badge {
            padding: 4px 10px;
            border-radius: 4px;
            font-size: 0.75em;
            font-weight: 600;
            text-transform: uppercase;
        }

        .badge-success {
            background: #d1fae5;
            color: #065f46;
        }

        .badge-danger {
            background: #fee2e2;
            color: #991b1b;
        }

        .table-wrapper {
            overflow-x: auto;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }

        th {
            background: #f1f5f9;
            padding: 15px;
            text-align: left;
            font-weight: 600;
            color: #475569;
            border-bottom: 2px solid #e2e8f0;
        }

        td {
            padding: 15px;
            border-bottom: 1px solid #e2e8f0;
        }

        tr:hover {
            background: #f8fafc;
        }

        .timeline {
            position: relative;
            padding-left: 30px;
            margin-top: 20px;
        }

        .timeline::before {
            content: '';
            position: absolute;
            left: 10px;
            top: 0;
            bottom: 0;
            width: 2px;
            background: #e2e8f0;
        }

        .timeline-item {
            position: relative;
            padding: 15px 0 15px 20px;
            margin-bottom: 20px;
        }

        .timeline-item::before {
            content: '';
            position: absolute;
            left: -24px;
            top: 20px;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: #667eea;
            border: 3px solid white;
            box-shadow: 0 0 0 2px #667eea;
        }

        .timeline-content {
            background: #f9fafb;
            padding: 15px;
            border-radius: 8px;
        }

        .timeline-title {
            font-weight: 600;
            color: #1f2937;
            margin-bottom: 5px;
        }

        .timeline-meta {
            font-size: 0.85em;
            color: #64748b;
        }

        .recommendations {
            background: #eff6ff;
            border-left: 5px solid #3b82f6;
            padding: 20px;
            border-radius: 8px;
        }

        .recommendations ul {
            list-style: none;
            margin-top: 15px;
        }

        .recommendations li {
            padding: 10px 0;
            padding-left: 25px;
            position: relative;
        }

        .recommendations li::before {
            content: '→';
            position: absolute;
            left: 0;
            color: #3b82f6;
            font-weight: 700;
        }

        .footer {
            text-align: center;
            padding: 30px;
            color: #64748b;
            font-size: 0.9em;
            margin-top: 50px;
        }

        .no-data {
            text-align: center;
            padding: 40px;
            color: #94a3b8;
            font-style: italic;
        }
        """

    def _render_header(
        self,
        data: AggregatedData,
        health_score: float,
        overall_health: str
    ) -> str:
        """Render dashboard header with health status."""
        health_class = f"health-{overall_health}"
        health_text = overall_health.upper()

        return f"""
        <div class="header">
            <h1>T.A.R.S. Repository Health Dashboard</h1>
            <div class="subtitle">
                Repository: {data.repository_path}<br>
                Scan Time: {data.scan_timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}
            </div>
            <div class="health-score">{health_score:.1f}/100</div>
            <div class="health-badge {health_class}">{health_text}</div>
        </div>
        """

    def _render_summary_stats(self, data: AggregatedData) -> str:
        """Render summary statistics grid."""
        return f"""
        <div class="card">
            <h2>Summary Statistics</h2>
            <div class="stats-grid">
                <div class="stat-box">
                    <div class="stat-value">{data.total_versions}</div>
                    <div class="stat-label">Total Versions</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{data.total_artifacts}</div>
                    <div class="stat-label">Total Artifacts</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{len(data.all_issues)}</div>
                    <div class="stat-label">Total Issues</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{len(data.repair_history)}</div>
                    <div class="stat-label">Repairs Applied</div>
                </div>
            </div>
        </div>
        """

    def _render_health_breakdown(self, data: AggregatedData) -> str:
        """Render issue severity breakdown."""
        return f"""
        <div class="card">
            <h2>Issue Severity Breakdown</h2>
            <div class="severity-grid">
                <div class="severity-box severity-critical">
                    <div style="font-size: 2em; margin-bottom: 5px;">{data.critical_issues}</div>
                    <div>CRITICAL</div>
                </div>
                <div class="severity-box severity-error">
                    <div style="font-size: 2em; margin-bottom: 5px;">{data.error_issues}</div>
                    <div>ERROR</div>
                </div>
                <div class="severity-box severity-warning">
                    <div style="font-size: 2em; margin-bottom: 5px;">{data.warning_issues}</div>
                    <div>WARNING</div>
                </div>
                <div class="severity-box severity-info">
                    <div style="font-size: 2em; margin-bottom: 5px;">{data.info_issues}</div>
                    <div>INFO</div>
                </div>
            </div>
        </div>
        """

    def _render_version_cards(self, data: AggregatedData) -> str:
        """Render version health cards."""
        if not data.versions:
            return f"""
            <div class="card">
                <h2>Version Health</h2>
                <div class="no-data">No version data available</div>
            </div>
            """

        version_cards = ""
        for version in sorted(data.versions, key=lambda v: v.version, reverse=True):
            health_class = f"health-{version.health_status}"
            status_text = version.health_status.upper()

            sbom_badge = f'<span class="badge badge-success">SBOM ✓</span>' if version.sbom_present else f'<span class="badge badge-danger">SBOM ✗</span>'
            slsa_badge = f'<span class="badge badge-success">SLSA ✓</span>' if version.slsa_present else f'<span class="badge badge-danger">SLSA ✗</span>'

            issue_count = len(version.issues)
            issue_text = f"{issue_count} issue{'s' if issue_count != 1 else ''}"

            version_cards += f"""
            <div class="version-card {health_class}">
                <div class="version-header">
                    <div class="version-name">{version.version}</div>
                    <div class="version-status {health_class}">{status_text}</div>
                </div>
                <div class="version-meta">
                    {version.artifact_count} artifacts • {issue_text}
                </div>
                <div class="version-badges">
                    {sbom_badge}
                    {slsa_badge}
                </div>
            </div>
            """

        return f"""
        <div class="card">
            <h2>Version Health</h2>
            <div class="version-grid">
                {version_cards}
            </div>
        </div>
        """

    def _render_issues_table(self, data: AggregatedData) -> str:
        """Render detailed issues table."""
        if not data.all_issues:
            return f"""
            <div class="card">
                <h2>Issues</h2>
                <div class="no-data">No issues detected - repository is healthy!</div>
            </div>
            """

        # Sort issues by severity (CRITICAL > ERROR > WARNING > INFO)
        severity_order = {"CRITICAL": 0, "ERROR": 1, "WARNING": 2, "INFO": 3}
        sorted_issues = sorted(
            data.all_issues,
            key=lambda i: (severity_order.get(i.severity, 4), i.issue_id)
        )

        # Limit to first 50 issues for display
        display_issues = sorted_issues[:50]

        rows = ""
        for issue in display_issues:
            severity_class = f"severity-{issue.severity.lower()}"
            rows += f"""
            <tr>
                <td><span class="badge {severity_class}" style="color: white;">{issue.severity}</span></td>
                <td>{issue.category}</td>
                <td>{issue.description[:100]}{'...' if len(issue.description) > 100 else ''}</td>
                <td>{issue.version or 'N/A'}</td>
                <td>{issue.artifact or 'N/A'}</td>
                <td>{issue.source}</td>
            </tr>
            """

        showing_text = ""
        if len(sorted_issues) > 50:
            showing_text = f"<p style='margin-top: 15px; color: #64748b; font-style: italic;'>Showing 50 of {len(sorted_issues)} total issues</p>"

        return f"""
        <div class="card">
            <h2>Issues</h2>
            <div class="table-wrapper">
                <table>
                    <thead>
                        <tr>
                            <th>Severity</th>
                            <th>Category</th>
                            <th>Description</th>
                            <th>Version</th>
                            <th>Artifact</th>
                            <th>Source</th>
                        </tr>
                    </thead>
                    <tbody>
                        {rows}
                    </tbody>
                </table>
            </div>
            {showing_text}
        </div>
        """

    def _render_timeline(self, data: AggregatedData) -> str:
        """Render timeline of operations."""
        # Combine all operations into timeline
        timeline_items = []

        # Add publications
        for pub in data.publication_history:
            timeline_items.append({
                'timestamp': self._parse_timestamp(pub.get('published_at')),
                'title': f"Published {pub.get('version')}",
                'meta': f"{pub.get('artifacts')} artifacts • Status: {pub.get('status')}"
            })

        # Add rollbacks
        for rollback in data.rollback_history:
            timeline_items.append({
                'timestamp': self._parse_timestamp(rollback.get('timestamp')),
                'title': f"Rollback: {rollback.get('from_version')} → {rollback.get('to_version')}",
                'meta': f"{rollback.get('artifacts_rolled_back')} artifacts • Status: {rollback.get('status')}"
            })

        # Add repairs
        for repair in data.repair_history:
            timeline_items.append({
                'timestamp': self._parse_timestamp(repair.get('timestamp')),
                'title': f"Repair: {repair.get('action', 'Unknown')}",
                'meta': repair.get('description', 'No description')
            })

        # Sort by timestamp (most recent first)
        timeline_items = sorted(
            [item for item in timeline_items if item['timestamp']],
            key=lambda x: x['timestamp'],
            reverse=True
        )[:20]  # Limit to 20 most recent

        if not timeline_items:
            return f"""
            <div class="card">
                <h2>Operation Timeline</h2>
                <div class="no-data">No operations recorded</div>
            </div>
            """

        timeline_html = ""
        for item in timeline_items:
            timestamp_str = item['timestamp'].strftime('%Y-%m-%d %H:%M:%S') if item['timestamp'] else 'Unknown'
            timeline_html += f"""
            <div class="timeline-item">
                <div class="timeline-content">
                    <div class="timeline-title">{item['title']}</div>
                    <div class="timeline-meta">{item['meta']}<br>{timestamp_str}</div>
                </div>
            </div>
            """

        return f"""
        <div class="card">
            <h2>Operation Timeline</h2>
            <div class="timeline">
                {timeline_html}
            </div>
        </div>
        """

    def _render_recommendations(self, recommendations: List[str]) -> str:
        """Render recommendations section."""
        if not recommendations:
            return ""

        rec_items = ""
        for rec in recommendations:
            rec_items += f"<li>{rec}</li>"

        return f"""
        <div class="card">
            <h2>Recommendations</h2>
            <div class="recommendations">
                <strong>Action Items:</strong>
                <ul>
                    {rec_items}
                </ul>
            </div>
        </div>
        """

    def _render_footer(self, data: AggregatedData) -> str:
        """Render dashboard footer."""
        report_count = len(data.report_metadata)
        return f"""
        <div class="footer">
            Generated by T.A.R.S. Repository Health Dashboard v1.0.0<br>
            Aggregated from {report_count} report{'s' if report_count != 1 else ''}<br>
            {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}
        </div>
        """

    def _parse_timestamp(self, timestamp_str: Optional[str]) -> Optional[datetime]:
        """Parse timestamp string to datetime object."""
        if not timestamp_str:
            return None

        try:
            # Try ISO format first
            return datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        except:
            try:
                # Try common formats
                for fmt in ["%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"]:
                    try:
                        return datetime.strptime(timestamp_str, fmt)
                    except:
                        continue
            except:
                pass

        return None
