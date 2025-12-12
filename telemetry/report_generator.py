"""
Operational Report Generator for T.A.R.S.

Generates daily and weekly operational reports from production telemetry data.

Features:
- Daily and weekly operational reports
- SLO compliance summaries
- Latency histograms and percentiles
- Error taxonomy and trending
- Region drift analysis
- Regression likelihood scoring
- Multi-format export (PDF, Markdown, JSON)
- Async implementation for performance

Report Sections:
- Executive summary
- SLO compliance
- Performance metrics
- Error analysis
- Regional analysis
- Regression detection
- Recommendations
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any
import statistics
from collections import Counter, defaultdict

try:
    import aiofiles
    AIOFILES_AVAILABLE = True
except ImportError:
    AIOFILES_AVAILABLE = False
    aiofiles = None

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

from telemetry.production_log_ingestor import LogEntry, LogLevel
from telemetry.slo_violation_detector import SLOViolation, ViolationSeverity
from telemetry.regression_classifier import RegressionPrediction, RegressionType


logger = logging.getLogger(__name__)


class ReportPeriod(str, Enum):
    """Report time period."""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


class ReportFormat(str, Enum):
    """Report output format."""
    MARKDOWN = "markdown"
    JSON = "json"
    PDF = "pdf"


@dataclass
class PerformanceMetrics:
    """Performance metrics summary."""

    # Latency metrics
    latency_p50: float = 0.0
    latency_p95: float = 0.0
    latency_p99: float = 0.0
    latency_max: float = 0.0
    latency_mean: float = 0.0

    # Throughput metrics
    total_requests: int = 0
    requests_per_second: float = 0.0

    # Error metrics
    total_errors: int = 0
    error_rate: float = 0.0

    # Success metrics
    success_rate: float = 0.0


@dataclass
class ErrorTaxonomy:
    """Error classification and trending."""

    # Error counts by type
    errors_by_type: Dict[str, int] = field(default_factory=dict)

    # Error counts by service
    errors_by_service: Dict[str, int] = field(default_factory=dict)

    # Error counts by region
    errors_by_region: Dict[str, int] = field(default_factory=dict)

    # Error trending
    error_trend: str = "stable"  # increasing, decreasing, stable

    # Top error messages
    top_errors: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class RegionalAnalysis:
    """Multi-region analysis."""

    # Latency by region
    latency_by_region: Dict[str, float] = field(default_factory=dict)

    # Request distribution
    requests_by_region: Dict[str, int] = field(default_factory=dict)

    # Error rates by region
    error_rate_by_region: Dict[str, float] = field(default_factory=dict)

    # Region drift score (max latency diff)
    drift_score: float = 0.0


@dataclass
class SLOSummary:
    """SLO compliance summary."""

    # SLO compliance rate
    compliance_rate: float = 0.0

    # Violations by severity
    violations_by_severity: Dict[str, int] = field(default_factory=dict)

    # Violations by type
    violations_by_type: Dict[str, int] = field(default_factory=dict)

    # Total violation duration
    total_violation_duration_seconds: float = 0.0

    # SLO status
    slo_status: Dict[str, str] = field(default_factory=dict)  # name -> status


@dataclass
class OperationalReport:
    """Complete operational report."""

    # Metadata
    report_id: str
    period: ReportPeriod
    start_time: datetime
    end_time: datetime
    generated_at: datetime = field(default_factory=datetime.utcnow)

    # Metrics
    performance: PerformanceMetrics = field(default_factory=PerformanceMetrics)
    slo_summary: SLOSummary = field(default_factory=SLOSummary)
    error_taxonomy: ErrorTaxonomy = field(default_factory=ErrorTaxonomy)
    regional_analysis: RegionalAnalysis = field(default_factory=RegionalAnalysis)

    # Violations and regressions
    violations: List[SLOViolation] = field(default_factory=list)
    regressions: List[RegressionPrediction] = field(default_factory=list)

    # Recommendations
    recommendations: List[str] = field(default_factory=list)

    # Health score
    overall_health_score: float = 100.0  # 0-100

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            'report_id': self.report_id,
            'period': self.period.value,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat(),
            'generated_at': self.generated_at.isoformat(),
            'performance': asdict(self.performance),
            'slo_summary': asdict(self.slo_summary),
            'error_taxonomy': asdict(self.error_taxonomy),
            'regional_analysis': asdict(self.regional_analysis),
            'violations': [v.to_dict() for v in self.violations],
            'regressions': [r.to_dict() for r in self.regressions],
            'recommendations': self.recommendations,
            'overall_health_score': round(self.overall_health_score, 2),
        }


class ReportGenerator:
    """Generates operational reports from telemetry data."""

    def __init__(self, output_dir: Path):
        """Initialize report generator.

        Args:
            output_dir: Output directory for reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def generate_report(
        self,
        logs: List[LogEntry],
        violations: List[SLOViolation],
        regressions: List[RegressionPrediction],
        period: ReportPeriod = ReportPeriod.DAILY,
        formats: List[ReportFormat] = None,
    ) -> OperationalReport:
        """Generate operational report.

        Args:
            logs: Production log entries
            violations: SLO violations
            regressions: Regression predictions
            period: Report period
            formats: Output formats

        Returns:
            Generated report
        """
        if formats is None:
            formats = [ReportFormat.MARKDOWN, ReportFormat.JSON]

        # Determine time range
        if logs:
            start_time = min(log.timestamp for log in logs)
            end_time = max(log.timestamp for log in logs)
        else:
            end_time = datetime.utcnow()
            if period == ReportPeriod.DAILY:
                start_time = end_time - timedelta(days=1)
            elif period == ReportPeriod.WEEKLY:
                start_time = end_time - timedelta(weeks=1)
            else:
                start_time = end_time - timedelta(days=30)

        # Generate report ID
        report_id = f"TARS-{period.value.upper()}-{end_time.strftime('%Y%m%d-%H%M%S')}"

        # Create report
        report = OperationalReport(
            report_id=report_id,
            period=period,
            start_time=start_time,
            end_time=end_time,
        )

        # Analyze logs
        await self._analyze_performance(logs, report)
        await self._analyze_errors(logs, report)
        await self._analyze_regional(logs, report)

        # Analyze violations
        await self._analyze_slo(violations, report)

        # Add regressions
        report.regressions = regressions

        # Calculate health score
        self._calculate_health_score(report)

        # Generate recommendations
        self._generate_recommendations(report)

        # Export in requested formats
        for fmt in formats:
            await self._export_report(report, fmt)

        return report

    async def _analyze_performance(self, logs: List[LogEntry], report: OperationalReport):
        """Analyze performance metrics.

        Args:
            logs: Log entries
            report: Report to update
        """
        latencies = [log.duration_ms for log in logs if log.duration_ms is not None]

        if latencies:
            latencies_sorted = sorted(latencies)
            n = len(latencies_sorted)

            report.performance.latency_p50 = latencies_sorted[int(n * 0.50)]
            report.performance.latency_p95 = latencies_sorted[int(n * 0.95)]
            report.performance.latency_p99 = latencies_sorted[int(n * 0.99)]
            report.performance.latency_max = max(latencies)
            report.performance.latency_mean = statistics.mean(latencies)

        report.performance.total_requests = len(logs)

        duration = (report.end_time - report.start_time).total_seconds()
        if duration > 0:
            report.performance.requests_per_second = len(logs) / duration

        errors = sum(1 for log in logs if log.level in (LogLevel.ERROR, LogLevel.CRITICAL))
        report.performance.total_errors = errors
        report.performance.error_rate = errors / max(len(logs), 1)
        report.performance.success_rate = 1.0 - report.performance.error_rate

    async def _analyze_errors(self, logs: List[LogEntry], report: OperationalReport):
        """Analyze error taxonomy.

        Args:
            logs: Log entries
            report: Report to update
        """
        error_logs = [log for log in logs if log.error is not None]

        # Count by type
        error_types = Counter(log.error_type or 'Unknown' for log in error_logs)
        report.error_taxonomy.errors_by_type = dict(error_types)

        # Count by service
        errors_by_service = Counter(log.service for log in error_logs)
        report.error_taxonomy.errors_by_service = dict(errors_by_service)

        # Count by region
        errors_by_region = Counter(log.region or 'unknown' for log in error_logs)
        report.error_taxonomy.errors_by_region = dict(errors_by_region)

        # Top errors
        error_messages = Counter(log.error for log in error_logs)
        report.error_taxonomy.top_errors = [
            {'message': msg, 'count': count}
            for msg, count in error_messages.most_common(10)
        ]

        # Simple trending (would need historical data for real trending)
        report.error_taxonomy.error_trend = "stable"

    async def _analyze_regional(self, logs: List[LogEntry], report: OperationalReport):
        """Analyze regional metrics.

        Args:
            logs: Log entries
            report: Report to update
        """
        # Group by region
        by_region = defaultdict(list)
        for log in logs:
            region = log.region or 'unknown'
            by_region[region].append(log)

        # Calculate per-region metrics
        for region, region_logs in by_region.items():
            # Latency
            latencies = [log.duration_ms for log in region_logs if log.duration_ms is not None]
            if latencies:
                report.regional_analysis.latency_by_region[region] = statistics.mean(latencies)

            # Request count
            report.regional_analysis.requests_by_region[region] = len(region_logs)

            # Error rate
            errors = sum(1 for log in region_logs if log.level in (LogLevel.ERROR, LogLevel.CRITICAL))
            report.regional_analysis.error_rate_by_region[region] = errors / max(len(region_logs), 1)

        # Calculate drift score
        if len(report.regional_analysis.latency_by_region) > 1:
            latencies = list(report.regional_analysis.latency_by_region.values())
            report.regional_analysis.drift_score = max(latencies) - min(latencies)

    async def _analyze_slo(self, violations: List[SLOViolation], report: OperationalReport):
        """Analyze SLO compliance.

        Args:
            violations: SLO violations
            report: Report to update
        """
        report.violations = violations

        if not violations:
            report.slo_summary.compliance_rate = 1.0
            return

        # Count by severity
        by_severity = Counter(v.severity.value for v in violations)
        report.slo_summary.violations_by_severity = dict(by_severity)

        # Count by type
        by_type = Counter(v.slo_type.value for v in violations)
        report.slo_summary.violations_by_type = dict(by_type)

        # Total violation duration
        total_duration = sum(v.violation_duration.total_seconds() for v in violations)
        report.slo_summary.total_violation_duration_seconds = total_duration

        # SLO status
        for violation in violations:
            report.slo_summary.slo_status[violation.slo_name] = "violated"

        # Calculate compliance rate (simplified)
        duration = (report.end_time - report.start_time).total_seconds()
        if duration > 0:
            report.slo_summary.compliance_rate = max(0.0, 1.0 - (total_duration / duration))

    def _calculate_health_score(self, report: OperationalReport):
        """Calculate overall health score.

        Args:
            report: Report to update
        """
        score = 100.0

        # Deduct for SLO violations
        score -= len(report.violations) * 2.0

        # Deduct for error rate
        score -= report.performance.error_rate * 20.0

        # Deduct for regressions
        for regression in report.regressions:
            if regression.regression_type != RegressionType.BENIGN:
                score -= regression.confidence * 10.0

        # Deduct for regional drift
        if report.regional_analysis.drift_score > 100:  # >100ms drift
            score -= 5.0

        report.overall_health_score = max(0.0, min(100.0, score))

    def _generate_recommendations(self, report: OperationalReport):
        """Generate actionable recommendations.

        Args:
            report: Report to update
        """
        recommendations = []

        # SLO-based recommendations
        if report.slo_summary.compliance_rate < 0.95:
            recommendations.append(
                f"SLO compliance at {report.slo_summary.compliance_rate * 100:.1f}% - "
                "investigate and resolve active violations"
            )

        # Performance recommendations
        if report.performance.latency_p95 > 200:
            recommendations.append(
                f"P95 latency at {report.performance.latency_p95:.1f}ms - "
                "consider performance optimization"
            )

        if report.performance.error_rate > 0.05:
            recommendations.append(
                f"Error rate at {report.performance.error_rate * 100:.2f}% - "
                "review error logs and address top failures"
            )

        # Regional recommendations
        if report.regional_analysis.drift_score > 100:
            recommendations.append(
                f"Regional latency drift at {report.regional_analysis.drift_score:.1f}ms - "
                "investigate cross-region performance"
            )

        # Regression recommendations
        for regression in report.regressions:
            if regression.regression_type != RegressionType.BENIGN and regression.confidence > 0.7:
                recommendations.append(
                    f"Potential {regression.regression_type.value} detected with "
                    f"{regression.confidence * 100:.1f}% confidence - investigate immediately"
                )

        # Health score recommendations
        if report.overall_health_score < 80:
            recommendations.append(
                f"Overall health score at {report.overall_health_score:.1f}/100 - "
                "urgent attention required"
            )

        report.recommendations = recommendations

    async def _export_report(self, report: OperationalReport, fmt: ReportFormat):
        """Export report in specified format.

        Args:
            report: Report to export
            fmt: Output format
        """
        if fmt == ReportFormat.MARKDOWN:
            await self._export_markdown(report)
        elif fmt == ReportFormat.JSON:
            await self._export_json(report)
        elif fmt == ReportFormat.PDF:
            await self._export_pdf(report)

    async def _export_markdown(self, report: OperationalReport):
        """Export report as Markdown.

        Args:
            report: Report to export
        """
        md = f"""# T.A.R.S. Operational Report

**Report ID:** {report.report_id}
**Period:** {report.period.value.title()}
**Time Range:** {report.start_time.isoformat()} to {report.end_time.isoformat()}
**Generated:** {report.generated_at.isoformat()}

---

## Executive Summary

**Overall Health Score:** {report.overall_health_score:.1f}/100

### Key Metrics
- **Total Requests:** {report.performance.total_requests:,}
- **Requests/Second:** {report.performance.requests_per_second:.2f}
- **Success Rate:** {report.performance.success_rate * 100:.2f}%
- **Error Rate:** {report.performance.error_rate * 100:.2f}%
- **SLO Compliance:** {report.slo_summary.compliance_rate * 100:.2f}%

---

## Performance Metrics

### Latency Distribution
- **P50:** {report.performance.latency_p50:.2f}ms
- **P95:** {report.performance.latency_p95:.2f}ms
- **P99:** {report.performance.latency_p99:.2f}ms
- **Max:** {report.performance.latency_max:.2f}ms
- **Mean:** {report.performance.latency_mean:.2f}ms

### Throughput
- **Total Requests:** {report.performance.total_requests:,}
- **Requests/Second:** {report.performance.requests_per_second:.2f}

---

## SLO Compliance

**Compliance Rate:** {report.slo_summary.compliance_rate * 100:.2f}%
**Total Violations:** {len(report.violations)}
**Violation Duration:** {report.slo_summary.total_violation_duration_seconds:.1f}s

### Violations by Severity
"""
        for severity, count in report.slo_summary.violations_by_severity.items():
            md += f"- **{severity.upper()}:** {count}\n"

        md += "\n### Violations by Type\n"
        for slo_type, count in report.slo_summary.violations_by_type.items():
            md += f"- **{slo_type}:** {count}\n"

        md += f"""
---

## Error Analysis

**Total Errors:** {report.performance.total_errors}
**Error Rate:** {report.performance.error_rate * 100:.2f}%
**Trend:** {report.error_taxonomy.error_trend.title()}

### Top Errors
"""
        for i, error in enumerate(report.error_taxonomy.top_errors[:5], 1):
            md += f"{i}. **{error['message'][:100]}...** ({error['count']} occurrences)\n"

        md += "\n### Errors by Service\n"
        for service, count in sorted(report.error_taxonomy.errors_by_service.items(), key=lambda x: x[1], reverse=True)[:5]:
            md += f"- **{service}:** {count}\n"

        md += f"""
---

## Regional Analysis

**Drift Score:** {report.regional_analysis.drift_score:.2f}ms

### Latency by Region
"""
        for region, latency in sorted(report.regional_analysis.latency_by_region.items()):
            md += f"- **{region}:** {latency:.2f}ms\n"

        md += "\n### Request Distribution\n"
        total_requests = sum(report.regional_analysis.requests_by_region.values())
        for region, count in sorted(report.regional_analysis.requests_by_region.items(), key=lambda x: x[1], reverse=True):
            pct = (count / total_requests * 100) if total_requests > 0 else 0
            md += f"- **{region}:** {count:,} ({pct:.1f}%)\n"

        md += f"""
---

## Regression Detection

**Total Predictions:** {len(report.regressions)}
"""

        regression_by_type = Counter(r.regression_type.value for r in report.regressions)
        for reg_type, count in regression_by_type.items():
            md += f"- **{reg_type.replace('_', ' ').title()}:** {count}\n"

        if report.recommendations:
            md += "\n---\n\n## Recommendations\n\n"
            for i, rec in enumerate(report.recommendations, 1):
                md += f"{i}. {rec}\n"

        md += "\n---\n\n**End of Report**\n"

        # Write file
        output_path = self.output_dir / f"{report.report_id}.md"

        if AIOFILES_AVAILABLE:
            async with aiofiles.open(output_path, 'w') as f:
                await f.write(md)
        else:
            output_path.write_text(md)

        logger.info(f"Markdown report exported to {output_path}")

    async def _export_json(self, report: OperationalReport):
        """Export report as JSON.

        Args:
            report: Report to export
        """
        output_path = self.output_dir / f"{report.report_id}.json"

        json_data = json.dumps(report.to_dict(), indent=2)

        if AIOFILES_AVAILABLE:
            async with aiofiles.open(output_path, 'w') as f:
                await f.write(json_data)
        else:
            output_path.write_text(json_data)

        logger.info(f"JSON report exported to {output_path}")

    async def _export_pdf(self, report: OperationalReport):
        """Export report as PDF.

        Args:
            report: Report to export
        """
        if not REPORTLAB_AVAILABLE:
            logger.warning("ReportLab not available, skipping PDF export")
            return

        output_path = self.output_dir / f"{report.report_id}.pdf"

        # Create PDF
        doc = SimpleDocTemplate(str(output_path), pagesize=letter)
        story = []
        styles = getSampleStyleSheet()

        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1a1a1a'),
            alignment=TA_CENTER,
        )
        story.append(Paragraph("T.A.R.S. Operational Report", title_style))
        story.append(Spacer(1, 0.3 * inch))

        # Metadata
        story.append(Paragraph(f"<b>Report ID:</b> {report.report_id}", styles['Normal']))
        story.append(Paragraph(f"<b>Period:</b> {report.period.value.title()}", styles['Normal']))
        story.append(Paragraph(f"<b>Generated:</b> {report.generated_at.isoformat()}", styles['Normal']))
        story.append(Spacer(1, 0.3 * inch))

        # Health Score
        health_color = colors.green if report.overall_health_score >= 90 else (
            colors.yellow if report.overall_health_score >= 70 else colors.red
        )
        story.append(Paragraph(
            f"<b>Overall Health Score:</b> <font color='{health_color.hexval()}'>{report.overall_health_score:.1f}/100</font>",
            styles['Heading2']
        ))
        story.append(Spacer(1, 0.2 * inch))

        # Performance table
        story.append(Paragraph("Performance Metrics", styles['Heading2']))
        perf_data = [
            ['Metric', 'Value'],
            ['Total Requests', f"{report.performance.total_requests:,}"],
            ['Requests/Second', f"{report.performance.requests_per_second:.2f}"],
            ['Success Rate', f"{report.performance.success_rate * 100:.2f}%"],
            ['P50 Latency', f"{report.performance.latency_p50:.2f}ms"],
            ['P95 Latency', f"{report.performance.latency_p95:.2f}ms"],
            ['P99 Latency', f"{report.performance.latency_p99:.2f}ms"],
        ]
        perf_table = Table(perf_data)
        perf_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        story.append(perf_table)

        doc.build(story)
        logger.info(f"PDF report exported to {output_path}")


# Example usage
if __name__ == '__main__':
    async def example():
        from telemetry.production_log_ingestor import LogEntry, LogLevel

        # Create sample logs
        logs = [
            LogEntry(
                timestamp=datetime.utcnow(),
                service='api-gateway',
                level=LogLevel.INFO,
                message='Request completed',
                duration_ms=120.0,
                region='us-east-1',
            )
            for _ in range(1000)
        ]

        # Generate report
        generator = ReportGenerator(output_dir=Path('reports'))
        report = await generator.generate_report(
            logs=logs,
            violations=[],
            regressions=[],
            period=ReportPeriod.DAILY,
            formats=[ReportFormat.MARKDOWN, ReportFormat.JSON],
        )

        print(f"Report generated: {report.report_id}")
        print(f"Health score: {report.overall_health_score:.1f}/100")

    asyncio.run(example())
