"""
SLA Reporting & Executive Readiness Dashboard Engine

This module implements an organization-level SLA intelligence layer that:
1. Translates technical signals into executive-grade SLA insights
2. Aggregates risk, breaches, and trends across repos
3. Produces board-ready summaries, not raw metrics
4. Is fully CI/CD-compatible and audit-friendly

This task answers:
"Are we meeting our commitments - and if not, where, why, and what is the business impact?"

Architecture:
- SLAPolicy: Definition of an SLA with targets and thresholds
- SLAWindowResult: Compliance result for a specific time window
- SLAComplianceResult: Overall compliance status for an SLA
- SLABreach: Record of an SLA breach with attribution
- SLARootCause: Root cause mapping for breaches
- ExecutiveReadinessScore: Organization-wide readiness assessment
- SLAIntelligenceSummary: Summary statistics
- SLAIntelligenceReport: Complete output with all analytics

Exit Codes (140-149):
- 140: Success, all SLAs compliant
- 141: At-risk SLAs detected
- 142: SLA breach detected
- 143: Config error
- 144: Parsing error
- 199: General SLA intelligence error

Input:
    org-health-report.json from Phase 14.8 Task 1
    org-alerts.json from Phase 14.8 Task 2
    trend-correlation-report.json from Phase 14.8 Task 3 (optional)
    temporal-intelligence-report.json from Phase 14.8 Task 4 (optional)
    SLA policy file (YAML/JSON)

Output:
    sla-intelligence-report.json

Version: 1.0.0
Phase: 14.8 Task 5
"""

import json
import logging
import math
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set

logger = logging.getLogger(__name__)


# ============================================================================
# Exit Codes (140-149)
# ============================================================================

EXIT_SLA_SUCCESS = 140
EXIT_SLA_AT_RISK = 141
EXIT_SLA_BREACH = 142
EXIT_SLA_CONFIG_ERROR = 143
EXIT_SLA_PARSE_ERROR = 144
EXIT_GENERAL_SLA_ERROR = 199


# ============================================================================
# Custom Exceptions
# ============================================================================

class SLAIntelligenceError(Exception):
    """Base exception for SLA intelligence engine errors."""
    exit_code = EXIT_GENERAL_SLA_ERROR


class SLAIntelligenceConfigError(SLAIntelligenceError):
    """Configuration error in SLA intelligence engine."""
    exit_code = EXIT_SLA_CONFIG_ERROR


class SLAIntelligenceParseError(SLAIntelligenceError):
    """Failed to parse input reports."""
    exit_code = EXIT_SLA_PARSE_ERROR


# ============================================================================
# Enums
# ============================================================================

class SLAType(Enum):
    """Types of SLAs supported."""
    AVAILABILITY = "availability"
    RELIABILITY = "reliability"
    INCIDENT_RESPONSE = "incident_response"
    CHANGE_FAILURE_RATE = "change_failure_rate"
    MEAN_TIME_TO_RECOVERY = "mttr"
    DEPLOYMENT_FREQUENCY = "deployment_frequency"
    LEAD_TIME = "lead_time"
    CUSTOM = "custom"


class SLAStatus(Enum):
    """Status of SLA compliance."""
    COMPLIANT = "compliant"
    AT_RISK = "at_risk"
    BREACHED = "breached"
    UNKNOWN = "unknown"

    def __lt__(self, other: "SLAStatus") -> bool:
        """Compare severity: BREACHED > AT_RISK > COMPLIANT > UNKNOWN."""
        order = {self.UNKNOWN: 0, self.COMPLIANT: 1, self.AT_RISK: 2, self.BREACHED: 3}
        return order[self] < order[other]


class SLASeverity(Enum):
    """Severity level for SLA breaches."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    def __lt__(self, other: "SLASeverity") -> bool:
        order = {self.LOW: 0, self.MEDIUM: 1, self.HIGH: 2, self.CRITICAL: 3}
        return order[self] < order[other]


class ReadinessTier(Enum):
    """Executive readiness tier classification."""
    GREEN = "green"      # All systems go - meeting/exceeding SLAs
    YELLOW = "yellow"    # Caution - some SLAs at risk
    RED = "red"          # Critical - SLAs breached, action required
    UNKNOWN = "unknown"


class RiskOutlook(Enum):
    """Forward-looking risk assessment."""
    IMPROVING = "improving"
    STABLE = "stable"
    DEGRADING = "degrading"
    CRITICAL = "critical"


# ============================================================================
# Data Classes - SLA Policy
# ============================================================================

@dataclass
class SLATarget:
    """Target definition for an SLA metric."""
    metric_name: str
    target_value: float
    warning_threshold: float  # At-risk threshold
    breach_threshold: float   # Breach threshold
    unit: str = "%"
    higher_is_better: bool = True  # True if higher values are better

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metric_name": self.metric_name,
            "target_value": self.target_value,
            "warning_threshold": self.warning_threshold,
            "breach_threshold": self.breach_threshold,
            "unit": self.unit,
            "higher_is_better": self.higher_is_better
        }


@dataclass
class SLAPolicy:
    """
    Definition of a Service Level Agreement policy.

    Supports:
    - Multiple SLA types (availability, reliability, etc.)
    - Per-repo and org-wide targets
    - Rolling window evaluation
    """
    policy_id: str
    policy_name: str
    sla_type: SLAType
    description: str = ""

    # Targets
    targets: List[SLATarget] = field(default_factory=list)

    # Scope
    applies_to_repos: List[str] = field(default_factory=list)  # Empty = all repos
    applies_to_org: bool = True

    # Priority
    priority: int = 1  # 1 = highest
    severity_on_breach: SLASeverity = SLASeverity.HIGH

    # Business context
    business_impact: str = ""
    stakeholders: List[str] = field(default_factory=list)

    # Compliance windows (in intervals)
    evaluation_windows: List[int] = field(default_factory=lambda: [7, 30, 90])

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "policy_id": self.policy_id,
            "policy_name": self.policy_name,
            "sla_type": self.sla_type.value,
            "description": self.description,
            "targets": [t.to_dict() for t in self.targets],
            "applies_to_repos": self.applies_to_repos,
            "applies_to_org": self.applies_to_org,
            "priority": self.priority,
            "severity_on_breach": self.severity_on_breach.value,
            "business_impact": self.business_impact,
            "stakeholders": self.stakeholders,
            "evaluation_windows": self.evaluation_windows
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SLAPolicy":
        """Create from dictionary."""
        targets = [
            SLATarget(
                metric_name=t.get("metric_name", ""),
                target_value=t.get("target_value", 0.0),
                warning_threshold=t.get("warning_threshold", 0.0),
                breach_threshold=t.get("breach_threshold", 0.0),
                unit=t.get("unit", "%"),
                higher_is_better=t.get("higher_is_better", True)
            )
            for t in data.get("targets", [])
        ]

        return cls(
            policy_id=data.get("policy_id", ""),
            policy_name=data.get("policy_name", ""),
            sla_type=SLAType(data.get("sla_type", "custom")),
            description=data.get("description", ""),
            targets=targets,
            applies_to_repos=data.get("applies_to_repos", []),
            applies_to_org=data.get("applies_to_org", True),
            priority=data.get("priority", 1),
            severity_on_breach=SLASeverity(data.get("severity_on_breach", "high")),
            business_impact=data.get("business_impact", ""),
            stakeholders=data.get("stakeholders", []),
            evaluation_windows=data.get("evaluation_windows", [7, 30, 90])
        )


# ============================================================================
# Data Classes - SLA Results
# ============================================================================

@dataclass
class SLAWindowResult:
    """Compliance result for a specific time window."""
    window_size: int  # In intervals (e.g., 7, 30, 90)
    window_label: str

    # Metrics
    actual_value: float = 0.0
    target_value: float = 0.0
    variance: float = 0.0  # Actual - Target (positive = exceeding)
    variance_percentage: float = 0.0

    # Status
    status: SLAStatus = SLAStatus.UNKNOWN

    # Time range
    start_timestamp: str = ""
    end_timestamp: str = ""
    data_points: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "window_size": self.window_size,
            "window_label": self.window_label,
            "actual_value": self.actual_value,
            "target_value": self.target_value,
            "variance": self.variance,
            "variance_percentage": self.variance_percentage,
            "status": self.status.value,
            "start_timestamp": self.start_timestamp,
            "end_timestamp": self.end_timestamp,
            "data_points": self.data_points
        }


@dataclass
class SLAComplianceResult:
    """Overall compliance result for an SLA."""
    policy_id: str
    policy_name: str
    sla_type: SLAType

    # Overall status (worst across windows)
    overall_status: SLAStatus = SLAStatus.UNKNOWN
    compliance_percentage: float = 0.0  # 0-100

    # Per-window results
    window_results: List[SLAWindowResult] = field(default_factory=list)

    # Trend
    trend_direction: str = "stable"  # improving, stable, degrading
    trend_confidence: float = 0.0

    # Affected entities
    affected_repos: List[str] = field(default_factory=list)
    affected_count: int = 0

    # Risk metrics
    days_until_breach: Optional[int] = None  # Projected days until breach (if at_risk)
    breach_probability: float = 0.0  # 0-1

    # Timestamps
    evaluated_at: str = ""
    last_compliant_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "policy_id": self.policy_id,
            "policy_name": self.policy_name,
            "sla_type": self.sla_type.value,
            "overall_status": self.overall_status.value,
            "compliance_percentage": self.compliance_percentage,
            "window_results": [w.to_dict() for w in self.window_results],
            "trend_direction": self.trend_direction,
            "trend_confidence": self.trend_confidence,
            "affected_repos": self.affected_repos,
            "affected_count": self.affected_count,
            "days_until_breach": self.days_until_breach,
            "breach_probability": self.breach_probability,
            "evaluated_at": self.evaluated_at,
            "last_compliant_at": self.last_compliant_at
        }


# ============================================================================
# Data Classes - Breach Attribution
# ============================================================================

@dataclass
class SLARootCause:
    """Root cause mapping for an SLA breach."""
    cause_id: str
    cause_type: str  # repo_degradation, correlation_cluster, propagation_path, alert_pattern

    # Description
    title: str
    description: str

    # Attribution
    confidence_score: float = 0.0  # 0-1
    contribution_percentage: float = 0.0  # Estimated % contribution to breach

    # Evidence
    evidence: List[str] = field(default_factory=list)
    related_repos: List[str] = field(default_factory=list)
    related_alerts: List[str] = field(default_factory=list)
    related_correlations: List[str] = field(default_factory=list)
    related_paths: List[str] = field(default_factory=list)

    # Metadata
    detected_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "cause_id": self.cause_id,
            "cause_type": self.cause_type,
            "title": self.title,
            "description": self.description,
            "confidence_score": self.confidence_score,
            "contribution_percentage": self.contribution_percentage,
            "evidence": self.evidence,
            "related_repos": self.related_repos,
            "related_alerts": self.related_alerts,
            "related_correlations": self.related_correlations,
            "related_paths": self.related_paths,
            "detected_at": self.detected_at
        }


@dataclass
class SLABreach:
    """Record of an SLA breach with full attribution."""
    breach_id: str
    policy_id: str
    policy_name: str
    sla_type: SLAType
    severity: SLASeverity

    # Breach details
    breach_timestamp: str
    breach_window: int  # Which window was breached
    actual_value: float = 0.0
    target_value: float = 0.0
    breach_magnitude: float = 0.0  # How far below target

    # Status
    status: str = "active"  # active, acknowledged, resolved
    duration_hours: float = 0.0

    # Root causes
    root_causes: List[SLARootCause] = field(default_factory=list)
    primary_cause: str = ""

    # Impact
    business_impact: str = ""
    affected_repos: List[str] = field(default_factory=list)
    affected_services: List[str] = field(default_factory=list)
    estimated_revenue_impact: float = 0.0

    # Actions
    recommended_actions: List[str] = field(default_factory=list)
    escalation_contacts: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "breach_id": self.breach_id,
            "policy_id": self.policy_id,
            "policy_name": self.policy_name,
            "sla_type": self.sla_type.value,
            "severity": self.severity.value,
            "breach_timestamp": self.breach_timestamp,
            "breach_window": self.breach_window,
            "actual_value": self.actual_value,
            "target_value": self.target_value,
            "breach_magnitude": self.breach_magnitude,
            "status": self.status,
            "duration_hours": self.duration_hours,
            "root_causes": [rc.to_dict() for rc in self.root_causes],
            "primary_cause": self.primary_cause,
            "business_impact": self.business_impact,
            "affected_repos": self.affected_repos,
            "affected_services": self.affected_services,
            "estimated_revenue_impact": self.estimated_revenue_impact,
            "recommended_actions": self.recommended_actions,
            "escalation_contacts": self.escalation_contacts
        }


# ============================================================================
# Data Classes - Executive Readiness
# ============================================================================

@dataclass
class ExecutiveReadinessScore:
    """Organization-wide executive readiness assessment."""
    # Overall score
    readiness_score: float = 0.0  # 0-100
    readiness_tier: ReadinessTier = ReadinessTier.UNKNOWN

    # Component scores (all 0-100)
    sla_compliance_score: float = 0.0
    trend_health_score: float = 0.0
    temporal_risk_score: float = 0.0
    propagation_exposure_score: float = 0.0

    # Weights used
    weights: Dict[str, float] = field(default_factory=lambda: {
        "sla_compliance": 0.40,
        "trend_health": 0.25,
        "temporal_risk": 0.20,
        "propagation_exposure": 0.15
    })

    # Risk outlook
    risk_outlook: RiskOutlook = RiskOutlook.STABLE
    outlook_confidence: float = 0.0
    forecast_windows: int = 2  # How many windows ahead

    # Key metrics
    compliant_slas: int = 0
    at_risk_slas: int = 0
    breached_slas: int = 0
    total_slas: int = 0

    # Health indicators
    high_risk_repos: List[str] = field(default_factory=list)
    improving_repos: List[str] = field(default_factory=list)
    declining_repos: List[str] = field(default_factory=list)

    # Narrative
    executive_summary: str = ""
    key_concerns: List[str] = field(default_factory=list)
    positive_highlights: List[str] = field(default_factory=list)

    # Timestamp
    calculated_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "readiness_score": self.readiness_score,
            "readiness_tier": self.readiness_tier.value,
            "sla_compliance_score": self.sla_compliance_score,
            "trend_health_score": self.trend_health_score,
            "temporal_risk_score": self.temporal_risk_score,
            "propagation_exposure_score": self.propagation_exposure_score,
            "weights": self.weights,
            "risk_outlook": self.risk_outlook.value,
            "outlook_confidence": self.outlook_confidence,
            "forecast_windows": self.forecast_windows,
            "compliant_slas": self.compliant_slas,
            "at_risk_slas": self.at_risk_slas,
            "breached_slas": self.breached_slas,
            "total_slas": self.total_slas,
            "high_risk_repos": self.high_risk_repos,
            "improving_repos": self.improving_repos,
            "declining_repos": self.declining_repos,
            "executive_summary": self.executive_summary,
            "key_concerns": self.key_concerns,
            "positive_highlights": self.positive_highlights,
            "calculated_at": self.calculated_at
        }


# ============================================================================
# Data Classes - Summary and Report
# ============================================================================

@dataclass
class SLAScorecard:
    """Board-ready SLA scorecard for a single policy."""
    policy_id: str
    policy_name: str
    sla_type: str

    # Traffic light status
    status: SLAStatus = SLAStatus.UNKNOWN
    status_icon: str = ""  # For display

    # Current performance
    current_value: float = 0.0
    target_value: float = 0.0
    unit: str = "%"

    # Trend
    trend_indicator: str = ""  # up, down, stable
    trend_description: str = ""

    # Risk
    risk_level: str = "low"
    days_at_current_status: int = 0

    # Plain English
    plain_english_status: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "policy_id": self.policy_id,
            "policy_name": self.policy_name,
            "sla_type": self.sla_type,
            "status": self.status.value,
            "status_icon": self.status_icon,
            "current_value": self.current_value,
            "target_value": self.target_value,
            "unit": self.unit,
            "trend_indicator": self.trend_indicator,
            "trend_description": self.trend_description,
            "risk_level": self.risk_level,
            "days_at_current_status": self.days_at_current_status,
            "plain_english_status": self.plain_english_status
        }


@dataclass
class SLAIntelligenceSummary:
    """Summary statistics for SLA intelligence analysis."""
    # SLA overview
    total_slas_evaluated: int = 0
    compliant_slas: int = 0
    at_risk_slas: int = 0
    breached_slas: int = 0

    # Breach summary
    total_breaches: int = 0
    critical_breaches: int = 0
    high_breaches: int = 0
    active_breaches: int = 0

    # Compliance metrics
    overall_compliance_rate: float = 0.0  # 0-100
    avg_sla_health: float = 0.0

    # Trend summary
    improving_slas: int = 0
    stable_slas: int = 0
    degrading_slas: int = 0

    # Repository impact
    repos_with_sla_impact: int = 0
    total_repos_evaluated: int = 0

    # Executive readiness
    executive_readiness_score: float = 0.0
    readiness_tier: str = "unknown"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class RiskNarrative:
    """Plain English risk narrative for executives."""
    headline: str = ""
    summary_paragraph: str = ""

    # Structured sections
    current_status: str = ""
    key_risks: List[str] = field(default_factory=list)
    mitigating_factors: List[str] = field(default_factory=list)
    recommended_focus_areas: List[str] = field(default_factory=list)

    # Time context
    reporting_period: str = ""
    next_review_date: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "headline": self.headline,
            "summary_paragraph": self.summary_paragraph,
            "current_status": self.current_status,
            "key_risks": self.key_risks,
            "mitigating_factors": self.mitigating_factors,
            "recommended_focus_areas": self.recommended_focus_areas,
            "reporting_period": self.reporting_period,
            "next_review_date": self.next_review_date
        }


@dataclass
class SLAIntelligenceReport:
    """
    Complete SLA Intelligence Report.

    Contains all SLA analysis results including:
    - SLA scorecards
    - Compliance results
    - Breach list with attribution
    - Executive readiness score
    - Risk outlook
    - Board-level recommendations
    """
    # Report metadata
    report_id: str
    generated_at: str
    report_version: str = "1.0.0"

    # Input paths
    org_report_path: str = ""
    alerts_report_path: str = ""
    correlation_report_path: str = ""
    temporal_report_path: str = ""
    sla_policy_path: str = ""

    # Summary
    summary: SLAIntelligenceSummary = field(default_factory=SLAIntelligenceSummary)

    # Executive readiness
    executive_readiness: ExecutiveReadinessScore = field(default_factory=ExecutiveReadinessScore)

    # SLA scorecards (board-ready)
    scorecards: List[SLAScorecard] = field(default_factory=list)

    # Compliance results (detailed)
    compliance_results: List[SLAComplianceResult] = field(default_factory=list)

    # Breaches
    breaches: List[SLABreach] = field(default_factory=list)
    breach_summary: Dict[str, Any] = field(default_factory=dict)

    # Risk narrative
    risk_narrative: RiskNarrative = field(default_factory=RiskNarrative)

    # Recommendations (prioritized)
    recommendations: List[Dict[str, Any]] = field(default_factory=list)

    # Policies evaluated
    policies_evaluated: List[str] = field(default_factory=list)

    # Source report metadata
    org_health_status: str = "unknown"
    org_health_score: float = 0.0
    total_repos: int = 0

    # Performance
    analysis_duration_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "report_id": self.report_id,
            "generated_at": self.generated_at,
            "report_version": self.report_version,
            "org_report_path": self.org_report_path,
            "alerts_report_path": self.alerts_report_path,
            "correlation_report_path": self.correlation_report_path,
            "temporal_report_path": self.temporal_report_path,
            "sla_policy_path": self.sla_policy_path,
            "summary": self.summary.to_dict(),
            "executive_readiness": self.executive_readiness.to_dict(),
            "scorecards": [s.to_dict() for s in self.scorecards],
            "compliance_results": [c.to_dict() for c in self.compliance_results],
            "breaches": [b.to_dict() for b in self.breaches],
            "breach_summary": self.breach_summary,
            "risk_narrative": self.risk_narrative.to_dict(),
            "recommendations": self.recommendations,
            "policies_evaluated": self.policies_evaluated,
            "org_health_status": self.org_health_status,
            "org_health_score": self.org_health_score,
            "total_repos": self.total_repos,
            "analysis_duration_ms": self.analysis_duration_ms
        }


# ============================================================================
# Data Classes - Configuration
# ============================================================================

@dataclass
class SLAThresholds:
    """Configurable thresholds for SLA intelligence engine."""
    # Compliance thresholds
    at_risk_percentage: float = 90.0  # Below target but above this = at_risk
    breach_percentage: float = 80.0   # Below this = breached

    # Attribution thresholds
    min_confidence_score: float = 0.3
    min_contribution_percentage: float = 10.0

    # Readiness scoring weights
    sla_compliance_weight: float = 0.40
    trend_health_weight: float = 0.25
    temporal_risk_weight: float = 0.20
    propagation_exposure_weight: float = 0.15

    # Readiness tier thresholds
    green_threshold: float = 80.0   # Score >= 80 = GREEN
    yellow_threshold: float = 60.0  # Score >= 60 = YELLOW, else RED

    # Outlook thresholds
    improving_threshold: float = 5.0   # Trend > 5% improvement
    degrading_threshold: float = -5.0  # Trend < -5% = degrading

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SLAThresholds":
        """Create from dictionary."""
        return cls(
            at_risk_percentage=data.get("at_risk_percentage", 90.0),
            breach_percentage=data.get("breach_percentage", 80.0),
            min_confidence_score=data.get("min_confidence_score", 0.3),
            min_contribution_percentage=data.get("min_contribution_percentage", 10.0),
            sla_compliance_weight=data.get("sla_compliance_weight", 0.40),
            trend_health_weight=data.get("trend_health_weight", 0.25),
            temporal_risk_weight=data.get("temporal_risk_weight", 0.20),
            propagation_exposure_weight=data.get("propagation_exposure_weight", 0.15),
            green_threshold=data.get("green_threshold", 80.0),
            yellow_threshold=data.get("yellow_threshold", 60.0),
            improving_threshold=data.get("improving_threshold", 5.0),
            degrading_threshold=data.get("degrading_threshold", -5.0)
        )


@dataclass
class SLAIntelligenceConfig:
    """Configuration for the SLA Intelligence Engine."""
    # Input paths
    org_report_path: Path
    alerts_report_path: Optional[Path] = None
    correlation_report_path: Optional[Path] = None
    temporal_report_path: Optional[Path] = None
    sla_policy_path: Optional[Path] = None

    # Output
    output_path: Optional[Path] = None

    # Thresholds
    thresholds: SLAThresholds = field(default_factory=SLAThresholds)

    # Evaluation windows (in intervals)
    evaluation_windows: List[int] = field(default_factory=lambda: [7, 30, 90])

    # Behavior flags
    verbose: bool = False
    summary_only: bool = False

    # CI/CD flags
    fail_on_breach: bool = False
    fail_on_at_risk: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "org_report_path": str(self.org_report_path),
            "alerts_report_path": str(self.alerts_report_path) if self.alerts_report_path else None,
            "correlation_report_path": str(self.correlation_report_path) if self.correlation_report_path else None,
            "temporal_report_path": str(self.temporal_report_path) if self.temporal_report_path else None,
            "sla_policy_path": str(self.sla_policy_path) if self.sla_policy_path else None,
            "output_path": str(self.output_path) if self.output_path else None,
            "thresholds": self.thresholds.to_dict(),
            "evaluation_windows": self.evaluation_windows,
            "verbose": self.verbose,
            "summary_only": self.summary_only,
            "fail_on_breach": self.fail_on_breach,
            "fail_on_at_risk": self.fail_on_at_risk
        }


# ============================================================================
# SLA Policy Loader
# ============================================================================

class SLAPolicyLoader:
    """
    Loads and validates SLA policies from configuration files.

    Supports:
    - JSON format
    - YAML format (if PyYAML installed)
    - Default policies if no file provided
    """

    def __init__(self, config: SLAIntelligenceConfig):
        """Initialize policy loader."""
        self.config = config
        self._policies: List[SLAPolicy] = []

    def load_policies(self) -> List[SLAPolicy]:
        """
        Load SLA policies from configuration.

        Returns:
            List of SLAPolicy objects
        """
        if self.config.sla_policy_path and self.config.sla_policy_path.exists():
            self._policies = self._load_from_file(self.config.sla_policy_path)
        else:
            logger.info("No SLA policy file provided, using default policies")
            self._policies = self._create_default_policies()

        logger.info(f"Loaded {len(self._policies)} SLA policies")
        return self._policies

    def _load_from_file(self, path: Path) -> List[SLAPolicy]:
        """Load policies from file."""
        try:
            if path.suffix in ('.yaml', '.yml'):
                return self._load_yaml(path)
            else:
                return self._load_json(path)
        except Exception as e:
            raise SLAIntelligenceParseError(f"Failed to load SLA policy file: {e}")

    def _load_json(self, path: Path) -> List[SLAPolicy]:
        """Load policies from JSON file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        policies_data = data.get("policies", data.get("sla_policies", []))
        if isinstance(policies_data, dict):
            policies_data = [policies_data]

        return [SLAPolicy.from_dict(p) for p in policies_data]

    def _load_yaml(self, path: Path) -> List[SLAPolicy]:
        """Load policies from YAML file."""
        try:
            import yaml
            with open(path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)

            policies_data = data.get("policies", data.get("sla_policies", []))
            if isinstance(policies_data, dict):
                policies_data = [policies_data]

            return [SLAPolicy.from_dict(p) for p in policies_data]
        except ImportError:
            raise SLAIntelligenceConfigError(
                "PyYAML not installed. Install with: pip install pyyaml"
            )

    def _create_default_policies(self) -> List[SLAPolicy]:
        """Create default SLA policies."""
        return [
            SLAPolicy(
                policy_id="sla_availability",
                policy_name="Service Availability SLA",
                sla_type=SLAType.AVAILABILITY,
                description="Measures overall service availability across repositories",
                targets=[
                    SLATarget(
                        metric_name="availability_score",
                        target_value=99.0,
                        warning_threshold=95.0,
                        breach_threshold=90.0,
                        unit="%",
                        higher_is_better=True
                    )
                ],
                priority=1,
                severity_on_breach=SLASeverity.CRITICAL,
                business_impact="Direct impact on customer experience and revenue",
                evaluation_windows=[7, 30, 90]
            ),
            SLAPolicy(
                policy_id="sla_reliability",
                policy_name="System Reliability SLA",
                sla_type=SLAType.RELIABILITY,
                description="Measures system reliability based on health scores",
                targets=[
                    SLATarget(
                        metric_name="health_score",
                        target_value=80.0,
                        warning_threshold=70.0,
                        breach_threshold=60.0,
                        unit="score",
                        higher_is_better=True
                    )
                ],
                priority=2,
                severity_on_breach=SLASeverity.HIGH,
                business_impact="Affects system stability and maintenance costs",
                evaluation_windows=[7, 30, 90]
            ),
            SLAPolicy(
                policy_id="sla_incident_response",
                policy_name="Incident Response SLA",
                sla_type=SLAType.INCIDENT_RESPONSE,
                description="Measures time to respond to critical alerts",
                targets=[
                    SLATarget(
                        metric_name="response_time_hours",
                        target_value=4.0,
                        warning_threshold=8.0,
                        breach_threshold=24.0,
                        unit="hours",
                        higher_is_better=False
                    )
                ],
                priority=1,
                severity_on_breach=SLASeverity.HIGH,
                business_impact="Extended outages increase customer impact",
                evaluation_windows=[7, 30]
            ),
            SLAPolicy(
                policy_id="sla_change_failure",
                policy_name="Change Failure Rate SLA",
                sla_type=SLAType.CHANGE_FAILURE_RATE,
                description="Percentage of deployments causing failures",
                targets=[
                    SLATarget(
                        metric_name="change_failure_rate",
                        target_value=10.0,
                        warning_threshold=15.0,
                        breach_threshold=25.0,
                        unit="%",
                        higher_is_better=False
                    )
                ],
                priority=3,
                severity_on_breach=SLASeverity.MEDIUM,
                business_impact="High failure rates increase technical debt and risk",
                evaluation_windows=[30, 90]
            )
        ]

    def get_policies(self) -> List[SLAPolicy]:
        """Get loaded policies."""
        return self._policies


# ============================================================================
# SLA Compliance Engine
# ============================================================================

class SLAComplianceEngine:
    """
    Evaluates SLA compliance using org health metrics, alerts, and correlations.

    Features:
    - Multi-window evaluation (7, 30, 90 intervals)
    - Per-repo and org-wide assessment
    - Trend analysis for compliance
    - Projected breach detection
    """

    def __init__(self, config: SLAIntelligenceConfig):
        """Initialize compliance engine."""
        self.config = config
        self.thresholds = config.thresholds
        self._compliance_results: List[SLAComplianceResult] = []

    def evaluate_compliance(
        self,
        policies: List[SLAPolicy],
        org_report: Dict[str, Any],
        alerts_report: Optional[Dict[str, Any]] = None,
        correlation_report: Optional[Dict[str, Any]] = None,
        temporal_report: Optional[Dict[str, Any]] = None
    ) -> List[SLAComplianceResult]:
        """
        Evaluate SLA compliance for all policies.

        Args:
            policies: List of SLA policies to evaluate
            org_report: Organization health report
            alerts_report: Optional alerts report
            correlation_report: Optional correlation report
            temporal_report: Optional temporal intelligence report

        Returns:
            List of SLAComplianceResult
        """
        self._compliance_results = []
        timestamp = datetime.utcnow().isoformat()

        # Extract metrics from reports
        org_metrics = self._extract_org_metrics(org_report)
        alert_metrics = self._extract_alert_metrics(alerts_report) if alerts_report else {}

        for policy in policies:
            result = self._evaluate_policy(
                policy, org_metrics, alert_metrics, timestamp
            )
            self._compliance_results.append(result)

        logger.info(f"Evaluated compliance for {len(self._compliance_results)} policies")
        return self._compliance_results

    def _extract_org_metrics(self, org_report: Dict[str, Any]) -> Dict[str, Any]:
        """Extract relevant metrics from org health report."""
        metrics = {
            "org_health_score": org_report.get("org_health_score", 0.0),
            "org_health_status": org_report.get("org_health_status", "unknown"),
            "total_repos": len(org_report.get("repositories", [])),
            "repositories": {},
            "trend_history": []
        }

        # Extract per-repo metrics
        for repo in org_report.get("repositories", []):
            repo_id = repo.get("repo_id", "")
            metrics["repositories"][repo_id] = {
                "health_score": repo.get("repository_score", 0.0),
                "risk_tier": repo.get("risk_tier", "low"),
                "trends": repo.get("trends", {}),
                "trend_history": repo.get("trend_history", [])
            }

        # Extract org-level trend history if available
        metrics["trend_history"] = org_report.get("trend_history", [])

        return metrics

    def _extract_alert_metrics(self, alerts_report: Dict[str, Any]) -> Dict[str, Any]:
        """Extract relevant metrics from alerts report."""
        alerts = alerts_report.get("alerts", [])

        return {
            "total_alerts": len(alerts),
            "critical_alerts": len([a for a in alerts if a.get("severity") == "critical"]),
            "high_alerts": len([a for a in alerts if a.get("severity") == "high"]),
            "open_alerts": len([a for a in alerts if a.get("status") == "open"]),
            "alerts_by_repo": self._group_alerts_by_repo(alerts)
        }

    def _group_alerts_by_repo(self, alerts: List[Dict]) -> Dict[str, int]:
        """Group alert counts by repository."""
        by_repo = {}
        for alert in alerts:
            for repo in alert.get("affected_repos", []):
                by_repo[repo] = by_repo.get(repo, 0) + 1
        return by_repo

    def _evaluate_policy(
        self,
        policy: SLAPolicy,
        org_metrics: Dict[str, Any],
        alert_metrics: Dict[str, Any],
        timestamp: str
    ) -> SLAComplianceResult:
        """Evaluate compliance for a single policy."""
        result = SLAComplianceResult(
            policy_id=policy.policy_id,
            policy_name=policy.policy_name,
            sla_type=policy.sla_type,
            evaluated_at=timestamp
        )

        # Get the primary target
        if not policy.targets:
            result.overall_status = SLAStatus.UNKNOWN
            return result

        primary_target = policy.targets[0]

        # Evaluate each window
        for window in policy.evaluation_windows:
            window_result = self._evaluate_window(
                policy, primary_target, window, org_metrics, alert_metrics
            )
            result.window_results.append(window_result)

        # Determine overall status (worst across windows)
        statuses = [w.status for w in result.window_results]
        if SLAStatus.BREACHED in statuses:
            result.overall_status = SLAStatus.BREACHED
        elif SLAStatus.AT_RISK in statuses:
            result.overall_status = SLAStatus.AT_RISK
        elif all(s == SLAStatus.COMPLIANT for s in statuses):
            result.overall_status = SLAStatus.COMPLIANT
        else:
            result.overall_status = SLAStatus.UNKNOWN

        # Calculate compliance percentage
        compliant_windows = len([w for w in result.window_results
                                if w.status == SLAStatus.COMPLIANT])
        result.compliance_percentage = (compliant_windows / len(result.window_results) * 100
                                       if result.window_results else 0.0)

        # Determine trend
        result.trend_direction, result.trend_confidence = self._calculate_trend(
            result.window_results, primary_target
        )

        # Identify affected repos
        result.affected_repos, result.affected_count = self._identify_affected_repos(
            policy, org_metrics
        )

        # Calculate breach probability and days until breach
        if result.overall_status == SLAStatus.AT_RISK:
            result.breach_probability = self._estimate_breach_probability(
                result.window_results, primary_target
            )
            result.days_until_breach = self._estimate_days_until_breach(
                result.window_results, primary_target
            )

        return result

    def _evaluate_window(
        self,
        policy: SLAPolicy,
        target: SLATarget,
        window: int,
        org_metrics: Dict[str, Any],
        alert_metrics: Dict[str, Any]
    ) -> SLAWindowResult:
        """Evaluate compliance for a specific time window."""
        result = SLAWindowResult(
            window_size=window,
            window_label=f"{window}-interval",
            target_value=target.target_value,
            end_timestamp=datetime.utcnow().isoformat()
        )

        # Get actual value based on SLA type and metric
        actual_value = self._get_metric_value(
            policy.sla_type, target.metric_name, org_metrics, alert_metrics, window
        )
        result.actual_value = actual_value

        # Calculate variance
        result.variance = actual_value - target.target_value
        if target.target_value > 0:
            result.variance_percentage = (result.variance / target.target_value) * 100

        # Determine status based on thresholds
        if target.higher_is_better:
            # Higher is better (e.g., availability)
            if actual_value >= target.target_value:
                result.status = SLAStatus.COMPLIANT
            elif actual_value >= target.warning_threshold:
                result.status = SLAStatus.AT_RISK
            else:
                result.status = SLAStatus.BREACHED
        else:
            # Lower is better (e.g., response time)
            if actual_value <= target.target_value:
                result.status = SLAStatus.COMPLIANT
            elif actual_value <= target.warning_threshold:
                result.status = SLAStatus.AT_RISK
            else:
                result.status = SLAStatus.BREACHED

        return result

    def _get_metric_value(
        self,
        sla_type: SLAType,
        metric_name: str,
        org_metrics: Dict[str, Any],
        alert_metrics: Dict[str, Any],
        window: int
    ) -> float:
        """Get the actual metric value for evaluation."""
        # Map SLA type to appropriate metric source
        if sla_type == SLAType.AVAILABILITY:
            # Use org health score as proxy for availability
            return org_metrics.get("org_health_score", 0.0)

        elif sla_type == SLAType.RELIABILITY:
            # Use average repo health score
            repos = org_metrics.get("repositories", {})
            if repos:
                scores = [r.get("health_score", 0.0) for r in repos.values()]
                return sum(scores) / len(scores)
            return 0.0

        elif sla_type == SLAType.INCIDENT_RESPONSE:
            # Estimate response time from alert data
            if alert_metrics:
                # If we have critical alerts, assume degraded response time
                critical_count = alert_metrics.get("critical_alerts", 0)
                if critical_count > 5:
                    return 24.0  # 24 hours (breach)
                elif critical_count > 2:
                    return 8.0   # 8 hours (at-risk)
                else:
                    return 2.0   # 2 hours (compliant)
            return 2.0  # Default to compliant if no data

        elif sla_type == SLAType.CHANGE_FAILURE_RATE:
            # Estimate from repo health trends
            repos = org_metrics.get("repositories", {})
            declining_count = 0
            for repo_data in repos.values():
                trend = repo_data.get("trends", {}).get("overall_trend", "")
                if trend.lower() == "declining":
                    declining_count += 1

            if repos:
                return (declining_count / len(repos)) * 100
            return 0.0

        else:
            # Custom - try to find metric in org report
            return org_metrics.get(metric_name, 50.0)

    def _calculate_trend(
        self,
        window_results: List[SLAWindowResult],
        target: SLATarget
    ) -> Tuple[str, float]:
        """Calculate trend direction from window results."""
        if len(window_results) < 2:
            return "stable", 0.5

        # Compare shorter window to longer window
        short_window = window_results[0]
        long_window = window_results[-1]

        if target.higher_is_better:
            if short_window.actual_value > long_window.actual_value * 1.05:
                return "improving", 0.8
            elif short_window.actual_value < long_window.actual_value * 0.95:
                return "degrading", 0.8
        else:
            if short_window.actual_value < long_window.actual_value * 0.95:
                return "improving", 0.8
            elif short_window.actual_value > long_window.actual_value * 1.05:
                return "degrading", 0.8

        return "stable", 0.6

    def _identify_affected_repos(
        self,
        policy: SLAPolicy,
        org_metrics: Dict[str, Any]
    ) -> Tuple[List[str], int]:
        """Identify repositories affected by SLA compliance."""
        affected = []
        repos = org_metrics.get("repositories", {})

        for repo_id, repo_data in repos.items():
            # Check if repo is in scope
            if policy.applies_to_repos and repo_id not in policy.applies_to_repos:
                continue

            # Check if repo is below threshold
            health_score = repo_data.get("health_score", 100.0)
            if health_score < 70.0:  # Below threshold
                affected.append(repo_id)

        return affected, len(affected)

    def _estimate_breach_probability(
        self,
        window_results: List[SLAWindowResult],
        target: SLATarget
    ) -> float:
        """Estimate probability of breach based on current trajectory."""
        if not window_results:
            return 0.0

        # Use most recent window
        recent = window_results[0]

        # Calculate distance to breach threshold
        if target.higher_is_better:
            distance_to_breach = recent.actual_value - target.breach_threshold
            distance_to_target = target.target_value - target.breach_threshold
        else:
            distance_to_breach = target.breach_threshold - recent.actual_value
            distance_to_target = target.breach_threshold - target.target_value

        if distance_to_target == 0:
            return 0.5

        # Simple linear probability estimate
        prob = 1.0 - (distance_to_breach / distance_to_target)
        return max(0.0, min(1.0, prob))

    def _estimate_days_until_breach(
        self,
        window_results: List[SLAWindowResult],
        target: SLATarget
    ) -> Optional[int]:
        """Estimate days until breach if current trend continues."""
        if len(window_results) < 2:
            return None

        # Calculate rate of change
        short_window = window_results[0]
        long_window = window_results[-1]

        rate = (short_window.actual_value - long_window.actual_value) / max(
            long_window.window_size - short_window.window_size, 1
        )

        if rate == 0:
            return None

        # Calculate days until breach
        if target.higher_is_better:
            if rate >= 0:  # Improving or stable
                return None
            distance = short_window.actual_value - target.breach_threshold
        else:
            if rate <= 0:  # Improving or stable
                return None
            distance = target.breach_threshold - short_window.actual_value

        days = int(abs(distance / rate))
        return max(1, min(days, 365))  # Cap at 1 year

    def get_compliance_results(self) -> List[SLAComplianceResult]:
        """Get computed compliance results."""
        return self._compliance_results


# ============================================================================
# SLA Breach Attribution Engine
# ============================================================================

class SLABreachAttributionEngine:
    """
    Attributes SLA breaches to root causes using correlation and temporal data.

    Attribution sources:
    - Specific repositories with declining health
    - Correlated clusters from Task 3
    - Temporal propagation paths from Task 4
    - Alert patterns from Task 2
    """

    def __init__(self, config: SLAIntelligenceConfig):
        """Initialize breach attribution engine."""
        self.config = config
        self.thresholds = config.thresholds
        self._breaches: List[SLABreach] = []
        self._breach_counter = 0

    def attribute_breaches(
        self,
        compliance_results: List[SLAComplianceResult],
        policies: List[SLAPolicy],
        org_report: Dict[str, Any],
        alerts_report: Optional[Dict[str, Any]] = None,
        correlation_report: Optional[Dict[str, Any]] = None,
        temporal_report: Optional[Dict[str, Any]] = None
    ) -> List[SLABreach]:
        """
        Attribute root causes to SLA breaches.

        Args:
            compliance_results: Results from compliance evaluation
            policies: SLA policies
            org_report: Organization health report
            alerts_report: Optional alerts report
            correlation_report: Optional correlation report
            temporal_report: Optional temporal intelligence report

        Returns:
            List of SLABreach with attribution
        """
        self._breaches = []
        timestamp = datetime.utcnow().isoformat()

        # Find breached or at-risk SLAs
        for result in compliance_results:
            if result.overall_status in (SLAStatus.BREACHED, SLAStatus.AT_RISK):
                policy = next((p for p in policies if p.policy_id == result.policy_id), None)
                if policy:
                    breach = self._create_breach(
                        result, policy, org_report, alerts_report,
                        correlation_report, temporal_report, timestamp
                    )
                    self._breaches.append(breach)

        logger.info(f"Attributed {len(self._breaches)} breaches/at-risk SLAs")
        return self._breaches

    def _create_breach(
        self,
        result: SLAComplianceResult,
        policy: SLAPolicy,
        org_report: Dict[str, Any],
        alerts_report: Optional[Dict[str, Any]],
        correlation_report: Optional[Dict[str, Any]],
        temporal_report: Optional[Dict[str, Any]],
        timestamp: str
    ) -> SLABreach:
        """Create a breach record with attribution."""
        self._breach_counter += 1

        # Get worst window
        worst_window = min(
            result.window_results,
            key=lambda w: w.actual_value if policy.targets[0].higher_is_better
                         else -w.actual_value
        ) if result.window_results else None

        breach = SLABreach(
            breach_id=f"breach_{datetime.utcnow().strftime('%Y%m%d')}_{self._breach_counter:04d}",
            policy_id=policy.policy_id,
            policy_name=policy.policy_name,
            sla_type=policy.sla_type,
            severity=policy.severity_on_breach if result.overall_status == SLAStatus.BREACHED
                    else SLASeverity.MEDIUM,
            breach_timestamp=timestamp,
            breach_window=worst_window.window_size if worst_window else 0,
            actual_value=worst_window.actual_value if worst_window else 0.0,
            target_value=worst_window.target_value if worst_window else 0.0,
            breach_magnitude=abs(worst_window.variance) if worst_window else 0.0,
            status="active" if result.overall_status == SLAStatus.BREACHED else "at_risk",
            business_impact=policy.business_impact,
            affected_repos=result.affected_repos[:10],
            escalation_contacts=policy.stakeholders
        )

        # Attribute root causes
        root_causes = []

        # 1. Repository degradation causes
        repo_causes = self._attribute_to_repos(org_report, result)
        root_causes.extend(repo_causes)

        # 2. Correlation cluster causes
        if correlation_report:
            cluster_causes = self._attribute_to_correlations(correlation_report, result)
            root_causes.extend(cluster_causes)

        # 3. Temporal propagation causes
        if temporal_report:
            temporal_causes = self._attribute_to_temporal(temporal_report, result)
            root_causes.extend(temporal_causes)

        # 4. Alert pattern causes
        if alerts_report:
            alert_causes = self._attribute_to_alerts(alerts_report, result)
            root_causes.extend(alert_causes)

        # Sort by confidence and contribution
        root_causes.sort(key=lambda c: c.confidence_score, reverse=True)

        breach.root_causes = root_causes[:5]  # Top 5 causes
        if root_causes:
            breach.primary_cause = root_causes[0].title

        # Generate recommendations
        breach.recommended_actions = self._generate_recommendations(breach, root_causes)

        return breach

    def _attribute_to_repos(
        self,
        org_report: Dict[str, Any],
        result: SLAComplianceResult
    ) -> List[SLARootCause]:
        """Attribute breach to specific repository issues."""
        causes = []
        repositories = org_report.get("repositories", [])

        # Find repositories with poor health
        for repo in repositories:
            repo_id = repo.get("repo_id", "")
            health_score = repo.get("repository_score", 100.0)
            risk_tier = repo.get("risk_tier", "low")

            if health_score < 60.0 or risk_tier in ("high", "critical"):
                cause = SLARootCause(
                    cause_id=f"repo_cause_{repo_id}",
                    cause_type="repo_degradation",
                    title=f"Repository Degradation: {repo_id}",
                    description=f"{repo_id} has a health score of {health_score:.1f} "
                               f"(risk tier: {risk_tier})",
                    confidence_score=min(1.0, (100 - health_score) / 50),
                    contribution_percentage=min(100, (100 - health_score)),
                    evidence=[
                        f"Health score: {health_score:.1f}",
                        f"Risk tier: {risk_tier}"
                    ],
                    related_repos=[repo_id],
                    detected_at=datetime.utcnow().isoformat()
                )
                causes.append(cause)

        return causes[:3]  # Top 3 repo causes

    def _attribute_to_correlations(
        self,
        correlation_report: Dict[str, Any],
        result: SLAComplianceResult
    ) -> List[SLARootCause]:
        """Attribute breach to correlation clusters."""
        causes = []
        clusters = correlation_report.get("clusters", [])

        for cluster in clusters:
            cluster_id = cluster.get("cluster_id", "")
            repos = cluster.get("repos", [])
            correlation_strength = cluster.get("avg_correlation", 0.0)

            # Check if cluster overlaps with affected repos
            overlap = set(repos) & set(result.affected_repos)
            if overlap and correlation_strength > 0.5:
                cause = SLARootCause(
                    cause_id=f"cluster_cause_{cluster_id}",
                    cause_type="correlation_cluster",
                    title=f"Correlated Cluster: {cluster_id}",
                    description=f"Cluster of {len(repos)} repos with avg correlation "
                               f"{correlation_strength:.2f} affecting SLA compliance",
                    confidence_score=min(1.0, correlation_strength),
                    contribution_percentage=min(100, len(overlap) / max(len(result.affected_repos), 1) * 100),
                    evidence=[
                        f"Cluster size: {len(repos)} repos",
                        f"Correlation strength: {correlation_strength:.2f}",
                        f"Overlap with affected repos: {len(overlap)}"
                    ],
                    related_repos=list(overlap),
                    related_correlations=[cluster_id],
                    detected_at=datetime.utcnow().isoformat()
                )
                causes.append(cause)

        return causes[:2]  # Top 2 cluster causes

    def _attribute_to_temporal(
        self,
        temporal_report: Dict[str, Any],
        result: SLAComplianceResult
    ) -> List[SLARootCause]:
        """Attribute breach to temporal propagation patterns."""
        causes = []
        paths = temporal_report.get("propagation_paths", [])
        anomalies = temporal_report.get("anomalies", [])

        for path in paths:
            path_id = path.get("path_id", "")
            repo_sequence = path.get("repo_sequence", [])

            # Check if path involves affected repos
            overlap = set(repo_sequence) & set(result.affected_repos)
            if overlap:
                cause = SLARootCause(
                    cause_id=f"temporal_cause_{path_id}",
                    cause_type="propagation_path",
                    title=f"Propagation Path: {path_id}",
                    description=f"Temporal propagation path affecting "
                               f"{len(repo_sequence)} repos with total lag "
                               f"{path.get('total_lag', 0)} intervals",
                    confidence_score=path.get("path_confidence", 0.5),
                    contribution_percentage=min(100, len(overlap) / max(len(result.affected_repos), 1) * 100),
                    evidence=[
                        f"Path: {' -> '.join(repo_sequence[:3])}...",
                        f"Total lag: {path.get('total_lag', 0)} intervals"
                    ],
                    related_repos=list(overlap),
                    related_paths=[path_id],
                    detected_at=datetime.utcnow().isoformat()
                )
                causes.append(cause)

        # Also check temporal anomalies
        for anomaly in anomalies:
            if anomaly.get("severity") in ("high", "critical"):
                cause = SLARootCause(
                    cause_id=f"anomaly_cause_{anomaly.get('anomaly_id', '')}",
                    cause_type="temporal_anomaly",
                    title=anomaly.get("title", "Temporal Anomaly"),
                    description=anomaly.get("message", ""),
                    confidence_score=0.7,
                    contribution_percentage=30.0,
                    evidence=anomaly.get("evidence", [])[:3],
                    related_repos=anomaly.get("affected_repos", [])[:5],
                    detected_at=datetime.utcnow().isoformat()
                )
                causes.append(cause)

        return causes[:2]  # Top 2 temporal causes

    def _attribute_to_alerts(
        self,
        alerts_report: Dict[str, Any],
        result: SLAComplianceResult
    ) -> List[SLARootCause]:
        """Attribute breach to alert patterns."""
        causes = []
        alerts = alerts_report.get("alerts", [])

        # Group critical/high alerts by type
        alert_types = {}
        for alert in alerts:
            if alert.get("severity") in ("critical", "high"):
                alert_type = alert.get("alert_type", "unknown")
                if alert_type not in alert_types:
                    alert_types[alert_type] = []
                alert_types[alert_type].append(alert)

        for alert_type, type_alerts in alert_types.items():
            if len(type_alerts) >= 2:  # Pattern threshold
                affected_repos = set()
                for alert in type_alerts:
                    affected_repos.update(alert.get("affected_repos", []))

                cause = SLARootCause(
                    cause_id=f"alert_cause_{alert_type}",
                    cause_type="alert_pattern",
                    title=f"Alert Pattern: {alert_type}",
                    description=f"{len(type_alerts)} {alert_type} alerts detected "
                               f"affecting {len(affected_repos)} repos",
                    confidence_score=min(1.0, len(type_alerts) / 5),
                    contribution_percentage=min(100, len(type_alerts) * 10),
                    evidence=[
                        f"Alert count: {len(type_alerts)}",
                        f"Affected repos: {len(affected_repos)}"
                    ],
                    related_repos=list(affected_repos)[:5],
                    related_alerts=[a.get("alert_id", "") for a in type_alerts[:5]],
                    detected_at=datetime.utcnow().isoformat()
                )
                causes.append(cause)

        return causes[:2]  # Top 2 alert causes

    def _generate_recommendations(
        self,
        breach: SLABreach,
        root_causes: List[SLARootCause]
    ) -> List[str]:
        """Generate recommendations based on breach and root causes."""
        recommendations = []

        # Based on severity
        if breach.severity == SLASeverity.CRITICAL:
            recommendations.append("IMMEDIATE: Escalate to leadership and initiate incident response")

        # Based on SLA type
        if breach.sla_type == SLAType.AVAILABILITY:
            recommendations.append("Review system redundancy and failover mechanisms")
        elif breach.sla_type == SLAType.RELIABILITY:
            recommendations.append("Conduct reliability review of affected components")
        elif breach.sla_type == SLAType.INCIDENT_RESPONSE:
            recommendations.append("Review on-call procedures and escalation paths")

        # Based on root causes
        for cause in root_causes[:2]:
            if cause.cause_type == "repo_degradation":
                recommendations.append(f"Prioritize remediation of {cause.related_repos[0] if cause.related_repos else 'degraded repos'}")
            elif cause.cause_type == "propagation_path":
                recommendations.append("Implement circuit breakers to prevent cascade failures")
            elif cause.cause_type == "alert_pattern":
                recommendations.append("Review and address recurring alert patterns")

        return recommendations[:5]

    def get_breaches(self) -> List[SLABreach]:
        """Get attributed breaches."""
        return self._breaches


# ============================================================================
# Executive Readiness Engine
# ============================================================================

class ExecutiveReadinessEngine:
    """
    Produces executive-grade readiness scoring and board-ready outputs.

    Features:
    - Single Executive Readiness Score (0-100)
    - Readiness tier classification (GREEN/YELLOW/RED)
    - Forward-looking risk outlook
    - Plain English narratives
    """

    def __init__(self, config: SLAIntelligenceConfig):
        """Initialize executive readiness engine."""
        self.config = config
        self.thresholds = config.thresholds

    def calculate_readiness(
        self,
        compliance_results: List[SLAComplianceResult],
        breaches: List[SLABreach],
        org_report: Dict[str, Any],
        temporal_report: Optional[Dict[str, Any]] = None
    ) -> ExecutiveReadinessScore:
        """
        Calculate executive readiness score.

        Args:
            compliance_results: SLA compliance results
            breaches: List of breaches
            org_report: Organization health report
            temporal_report: Optional temporal report

        Returns:
            ExecutiveReadinessScore
        """
        timestamp = datetime.utcnow().isoformat()

        readiness = ExecutiveReadinessScore(calculated_at=timestamp)

        # Calculate component scores
        readiness.sla_compliance_score = self._calculate_sla_compliance_score(
            compliance_results
        )
        readiness.trend_health_score = self._calculate_trend_health_score(
            org_report
        )
        readiness.temporal_risk_score = self._calculate_temporal_risk_score(
            temporal_report
        )
        readiness.propagation_exposure_score = self._calculate_propagation_score(
            temporal_report
        )

        # Calculate weighted readiness score
        weights = readiness.weights
        readiness.readiness_score = (
            readiness.sla_compliance_score * weights["sla_compliance"] +
            readiness.trend_health_score * weights["trend_health"] +
            readiness.temporal_risk_score * weights["temporal_risk"] +
            readiness.propagation_exposure_score * weights["propagation_exposure"]
        )

        # Determine tier
        if readiness.readiness_score >= self.thresholds.green_threshold:
            readiness.readiness_tier = ReadinessTier.GREEN
        elif readiness.readiness_score >= self.thresholds.yellow_threshold:
            readiness.readiness_tier = ReadinessTier.YELLOW
        else:
            readiness.readiness_tier = ReadinessTier.RED

        # SLA counts
        readiness.total_slas = len(compliance_results)
        readiness.compliant_slas = len([r for r in compliance_results
                                        if r.overall_status == SLAStatus.COMPLIANT])
        readiness.at_risk_slas = len([r for r in compliance_results
                                     if r.overall_status == SLAStatus.AT_RISK])
        readiness.breached_slas = len([r for r in compliance_results
                                      if r.overall_status == SLAStatus.BREACHED])

        # Risk outlook
        readiness.risk_outlook, readiness.outlook_confidence = self._calculate_outlook(
            compliance_results, org_report
        )

        # Identify key repos
        readiness.high_risk_repos, readiness.improving_repos, readiness.declining_repos = \
            self._categorize_repos(org_report)

        # Generate narrative
        readiness.executive_summary = self._generate_executive_summary(readiness)
        readiness.key_concerns = self._identify_concerns(compliance_results, breaches)
        readiness.positive_highlights = self._identify_highlights(compliance_results, org_report)

        logger.info(f"Calculated executive readiness: {readiness.readiness_score:.1f} "
                   f"({readiness.readiness_tier.value})")

        return readiness

    def _calculate_sla_compliance_score(
        self,
        compliance_results: List[SLAComplianceResult]
    ) -> float:
        """Calculate SLA compliance component score."""
        if not compliance_results:
            return 100.0

        compliant = len([r for r in compliance_results
                        if r.overall_status == SLAStatus.COMPLIANT])
        at_risk = len([r for r in compliance_results
                      if r.overall_status == SLAStatus.AT_RISK])

        # Weight: compliant = 100%, at_risk = 50%, breached = 0%
        score = (compliant * 100 + at_risk * 50) / len(compliance_results)
        return min(100.0, max(0.0, score))

    def _calculate_trend_health_score(
        self,
        org_report: Dict[str, Any]
    ) -> float:
        """Calculate trend health component score."""
        repositories = org_report.get("repositories", [])
        if not repositories:
            return 50.0

        improving = 0
        stable = 0
        declining = 0

        for repo in repositories:
            trends = repo.get("trends", {})
            overall_trend = trends.get("overall_trend", "").lower() if isinstance(trends, dict) else ""

            if overall_trend == "improving":
                improving += 1
            elif overall_trend == "declining":
                declining += 1
            else:
                stable += 1

        total = len(repositories)
        # Weight: improving = 100, stable = 70, declining = 30
        score = (improving * 100 + stable * 70 + declining * 30) / total
        return min(100.0, max(0.0, score))

    def _calculate_temporal_risk_score(
        self,
        temporal_report: Optional[Dict[str, Any]]
    ) -> float:
        """Calculate temporal risk component score (inverted - higher = less risk)."""
        if not temporal_report:
            return 80.0  # Default to low risk if no data

        anomalies = temporal_report.get("anomalies", [])
        critical = len([a for a in anomalies if a.get("severity") == "critical"])
        high = len([a for a in anomalies if a.get("severity") == "high"])

        # More anomalies = lower score
        risk_factor = critical * 20 + high * 10
        score = max(0.0, 100.0 - risk_factor)
        return score

    def _calculate_propagation_score(
        self,
        temporal_report: Optional[Dict[str, Any]]
    ) -> float:
        """Calculate propagation exposure score (inverted - higher = less exposure)."""
        if not temporal_report:
            return 85.0  # Default to low exposure if no data

        summary = temporal_report.get("summary", {})
        paths_detected = summary.get("propagation_paths_detected", 0)
        interconnectedness = summary.get("org_interconnectedness", 0.0)

        # More paths and higher interconnectedness = higher risk = lower score
        exposure_factor = paths_detected * 5 + interconnectedness * 20
        score = max(0.0, 100.0 - exposure_factor)
        return score

    def _calculate_outlook(
        self,
        compliance_results: List[SLAComplianceResult],
        org_report: Dict[str, Any]
    ) -> Tuple[RiskOutlook, float]:
        """Calculate forward-looking risk outlook."""
        # Count trend directions
        improving = 0
        degrading = 0

        for result in compliance_results:
            if result.trend_direction == "improving":
                improving += 1
            elif result.trend_direction == "degrading":
                degrading += 1

        # Also consider org trend
        org_status = org_report.get("org_health_status", "").lower()

        if degrading > improving and degrading >= 2:
            return RiskOutlook.DEGRADING, 0.7
        elif improving > degrading and improving >= 2:
            return RiskOutlook.IMPROVING, 0.7
        elif org_status in ("red", "critical"):
            return RiskOutlook.CRITICAL, 0.8
        else:
            return RiskOutlook.STABLE, 0.6

    def _categorize_repos(
        self,
        org_report: Dict[str, Any]
    ) -> Tuple[List[str], List[str], List[str]]:
        """Categorize repositories by risk level and trend."""
        high_risk = []
        improving = []
        declining = []

        for repo in org_report.get("repositories", []):
            repo_id = repo.get("repo_id", "")
            risk_tier = repo.get("risk_tier", "low")
            trends = repo.get("trends", {})
            overall_trend = trends.get("overall_trend", "").lower() if isinstance(trends, dict) else ""

            if risk_tier in ("high", "critical"):
                high_risk.append(repo_id)

            if overall_trend == "improving":
                improving.append(repo_id)
            elif overall_trend == "declining":
                declining.append(repo_id)

        return high_risk[:10], improving[:10], declining[:10]

    def _generate_executive_summary(self, readiness: ExecutiveReadinessScore) -> str:
        """Generate plain English executive summary."""
        tier_status = {
            ReadinessTier.GREEN: "The organization is meeting its service level commitments",
            ReadinessTier.YELLOW: "Some service level commitments require attention",
            ReadinessTier.RED: "Service level commitments are at risk and require immediate action"
        }

        summary = tier_status.get(readiness.readiness_tier, "Service level status is being evaluated")

        # Add SLA status
        if readiness.breached_slas > 0:
            summary += f". {readiness.breached_slas} SLA(s) are currently breached"
        elif readiness.at_risk_slas > 0:
            summary += f". {readiness.at_risk_slas} SLA(s) are at risk of breach"
        else:
            summary += f". All {readiness.compliant_slas} SLAs are compliant"

        # Add outlook
        outlook_text = {
            RiskOutlook.IMPROVING: "The overall trend is improving",
            RiskOutlook.STABLE: "The situation is stable",
            RiskOutlook.DEGRADING: "Conditions are degrading and require monitoring",
            RiskOutlook.CRITICAL: "Critical intervention is needed"
        }
        summary += f". {outlook_text.get(readiness.risk_outlook, '')}"

        return summary

    def _identify_concerns(
        self,
        compliance_results: List[SLAComplianceResult],
        breaches: List[SLABreach]
    ) -> List[str]:
        """Identify key concerns for executives."""
        concerns = []

        # Breached SLAs
        for result in compliance_results:
            if result.overall_status == SLAStatus.BREACHED:
                concerns.append(f"{result.policy_name} is breached")

        # At-risk SLAs with high breach probability
        for result in compliance_results:
            if result.overall_status == SLAStatus.AT_RISK and result.breach_probability > 0.5:
                concerns.append(f"{result.policy_name} has {result.breach_probability*100:.0f}% "
                              f"breach probability")

        # Critical breaches
        for breach in breaches:
            if breach.severity == SLASeverity.CRITICAL:
                concerns.append(f"Critical: {breach.policy_name} breach requires immediate attention")

        return concerns[:5]

    def _identify_highlights(
        self,
        compliance_results: List[SLAComplianceResult],
        org_report: Dict[str, Any]
    ) -> List[str]:
        """Identify positive highlights for executives."""
        highlights = []

        # Improving SLAs
        improving = [r for r in compliance_results if r.trend_direction == "improving"]
        if improving:
            highlights.append(f"{len(improving)} SLA(s) showing improvement trend")

        # High compliance rate
        compliant = len([r for r in compliance_results
                        if r.overall_status == SLAStatus.COMPLIANT])
        if compliant == len(compliance_results) and compliance_results:
            highlights.append("100% SLA compliance achieved")
        elif compliant / max(len(compliance_results), 1) >= 0.8:
            highlights.append(f"{compliant}/{len(compliance_results)} SLAs fully compliant")

        # Healthy org score
        org_score = org_report.get("org_health_score", 0)
        if org_score >= 80:
            highlights.append(f"Organization health score at {org_score:.1f}")

        return highlights[:3]

    def generate_scorecards(
        self,
        compliance_results: List[SLAComplianceResult],
        policies: List[SLAPolicy]
    ) -> List[SLAScorecard]:
        """Generate board-ready scorecards for each SLA."""
        scorecards = []

        for result in compliance_results:
            policy = next((p for p in policies if p.policy_id == result.policy_id), None)

            # Get most recent window result
            recent_window = result.window_results[0] if result.window_results else None

            # Status icons
            status_icons = {
                SLAStatus.COMPLIANT: "green_circle",
                SLAStatus.AT_RISK: "yellow_circle",
                SLAStatus.BREACHED: "red_circle",
                SLAStatus.UNKNOWN: "white_circle"
            }

            # Trend indicators
            trend_icons = {
                "improving": "arrow_up",
                "stable": "arrow_right",
                "degrading": "arrow_down"
            }

            # Plain English status
            plain_status = {
                SLAStatus.COMPLIANT: "Meeting target",
                SLAStatus.AT_RISK: "Below target - attention needed",
                SLAStatus.BREACHED: "Target missed - action required",
                SLAStatus.UNKNOWN: "Insufficient data"
            }

            scorecard = SLAScorecard(
                policy_id=result.policy_id,
                policy_name=result.policy_name,
                sla_type=result.sla_type.value,
                status=result.overall_status,
                status_icon=status_icons.get(result.overall_status, "white_circle"),
                current_value=recent_window.actual_value if recent_window else 0.0,
                target_value=recent_window.target_value if recent_window else 0.0,
                unit=policy.targets[0].unit if policy and policy.targets else "%",
                trend_indicator=trend_icons.get(result.trend_direction, "arrow_right"),
                trend_description=f"Trend: {result.trend_direction}",
                risk_level="high" if result.overall_status == SLAStatus.BREACHED else
                          "medium" if result.overall_status == SLAStatus.AT_RISK else "low",
                plain_english_status=plain_status.get(result.overall_status, "Status unknown")
            )
            scorecards.append(scorecard)

        return scorecards

    def generate_risk_narrative(
        self,
        readiness: ExecutiveReadinessScore,
        compliance_results: List[SLAComplianceResult],
        breaches: List[SLABreach]
    ) -> RiskNarrative:
        """Generate plain English risk narrative for executives."""
        # Headline based on tier
        headlines = {
            ReadinessTier.GREEN: "Service Levels: All Systems Operational",
            ReadinessTier.YELLOW: "Service Levels: Attention Required",
            ReadinessTier.RED: "Service Levels: Critical Action Needed"
        }

        narrative = RiskNarrative(
            headline=headlines.get(readiness.readiness_tier, "Service Level Report"),
            reporting_period=f"As of {datetime.utcnow().strftime('%B %d, %Y')}",
            next_review_date=(datetime.utcnow().strftime('%B %d, %Y'))  # Today for now
        )

        # Summary paragraph
        narrative.summary_paragraph = readiness.executive_summary

        # Current status
        narrative.current_status = f"Readiness Score: {readiness.readiness_score:.0f}/100 ({readiness.readiness_tier.value.upper()})"

        # Key risks
        narrative.key_risks = readiness.key_concerns

        # Mitigating factors
        if readiness.positive_highlights:
            narrative.mitigating_factors = readiness.positive_highlights
        else:
            narrative.mitigating_factors = ["No significant mitigating factors identified"]

        # Recommended focus areas
        focus_areas = []
        if readiness.breached_slas > 0:
            focus_areas.append("Restore breached SLAs to compliance")
        if readiness.at_risk_slas > 0:
            focus_areas.append("Prevent at-risk SLAs from breaching")
        if readiness.declining_repos:
            focus_areas.append(f"Address declining health in {len(readiness.declining_repos)} repositories")
        if readiness.high_risk_repos:
            focus_areas.append(f"Prioritize {len(readiness.high_risk_repos)} high-risk repositories")

        narrative.recommended_focus_areas = focus_areas if focus_areas else ["Maintain current compliance levels"]

        return narrative


# ============================================================================
# Main Engine
# ============================================================================

class SLAIntelligenceEngine:
    """
    Main orchestrator for the SLA Reporting & Executive Readiness Dashboard Engine.

    This class:
    1. Loads SLA policies from configuration
    2. Evaluates SLA compliance using org health and alert data
    3. Attributes breaches to root causes
    4. Calculates executive readiness score
    5. Generates board-ready outputs
    6. Returns appropriate exit code
    """

    def __init__(self, config: SLAIntelligenceConfig):
        """Initialize the SLA intelligence engine."""
        self.config = config

        if config.verbose:
            logging.basicConfig(
                level=logging.DEBUG,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            logger.setLevel(logging.DEBUG)
        else:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s'
            )

        # Initialize components
        self.policy_loader = SLAPolicyLoader(config)
        self.compliance_engine = SLAComplianceEngine(config)
        self.attribution_engine = SLABreachAttributionEngine(config)
        self.readiness_engine = ExecutiveReadinessEngine(config)

        # Results storage
        self._policies: List[SLAPolicy] = []
        self._compliance_results: List[SLAComplianceResult] = []
        self._breaches: List[SLABreach] = []
        self._readiness: Optional[ExecutiveReadinessScore] = None

        # Input reports
        self._org_report: Optional[Dict[str, Any]] = None
        self._alerts_report: Optional[Dict[str, Any]] = None
        self._correlation_report: Optional[Dict[str, Any]] = None
        self._temporal_report: Optional[Dict[str, Any]] = None

    def run(self) -> Tuple[SLAIntelligenceReport, int]:
        """
        Run the complete SLA intelligence pipeline.

        Returns:
            Tuple of (SLAIntelligenceReport, exit_code)
        """
        start_time = datetime.utcnow()

        try:
            logger.info("=" * 80)
            logger.info("SLA REPORTING & EXECUTIVE READINESS DASHBOARD ENGINE")
            logger.info("=" * 80)

            # Step 1: Load input reports
            logger.info("\nStep 1: Loading input reports...")
            self._load_reports()

            # Step 2: Load SLA policies
            logger.info("\nStep 2: Loading SLA policies...")
            self._policies = self.policy_loader.load_policies()
            logger.info(f"  Loaded {len(self._policies)} policies")

            # Step 3: Evaluate compliance
            logger.info("\nStep 3: Evaluating SLA compliance...")
            self._compliance_results = self.compliance_engine.evaluate_compliance(
                self._policies,
                self._org_report,
                self._alerts_report,
                self._correlation_report,
                self._temporal_report
            )
            logger.info(f"  Evaluated {len(self._compliance_results)} SLAs")

            # Step 4: Attribute breaches
            logger.info("\nStep 4: Attributing breaches...")
            self._breaches = self.attribution_engine.attribute_breaches(
                self._compliance_results,
                self._policies,
                self._org_report,
                self._alerts_report,
                self._correlation_report,
                self._temporal_report
            )
            logger.info(f"  Attributed {len(self._breaches)} breaches/at-risk")

            # Step 5: Calculate executive readiness
            logger.info("\nStep 5: Calculating executive readiness...")
            self._readiness = self.readiness_engine.calculate_readiness(
                self._compliance_results,
                self._breaches,
                self._org_report,
                self._temporal_report
            )
            logger.info(f"  Readiness: {self._readiness.readiness_score:.1f} "
                       f"({self._readiness.readiness_tier.value})")

            # Step 6: Generate report
            logger.info("\nStep 6: Generating SLA intelligence report...")
            report = self._generate_report()

            # Calculate duration
            duration = (datetime.utcnow() - start_time).total_seconds() * 1000
            report.analysis_duration_ms = duration

            # Step 7: Write output
            if self.config.output_path:
                logger.info("\nStep 7: Writing report...")
                self._write_report(report)
                logger.info(f"  Wrote report to {self.config.output_path}")

            # Determine exit code
            exit_code = self._determine_exit_code()

            logger.info("\n" + "=" * 80)
            logger.info("SLA INTELLIGENCE ANALYSIS COMPLETE")
            logger.info("=" * 80)
            logger.info(f"Executive Readiness: {self._readiness.readiness_score:.1f} "
                       f"({self._readiness.readiness_tier.value})")
            logger.info(f"SLAs Evaluated: {len(self._compliance_results)}")
            logger.info(f"  - Compliant: {self._readiness.compliant_slas}")
            logger.info(f"  - At Risk: {self._readiness.at_risk_slas}")
            logger.info(f"  - Breached: {self._readiness.breached_slas}")
            logger.info(f"Exit Code: {exit_code}")
            logger.info("=" * 80)

            return report, exit_code

        except SLAIntelligenceParseError as e:
            logger.error(f"Failed to parse input report: {e}")
            return self._error_report(str(e)), EXIT_SLA_PARSE_ERROR
        except SLAIntelligenceConfigError as e:
            logger.error(f"Configuration error: {e}")
            return self._error_report(str(e)), EXIT_SLA_CONFIG_ERROR
        except Exception as e:
            logger.error(f"SLA intelligence error: {e}")
            import traceback
            traceback.print_exc()
            return self._error_report(str(e)), EXIT_GENERAL_SLA_ERROR

    def _load_reports(self) -> None:
        """Load all input reports."""
        # Load org report (required)
        if not self.config.org_report_path.exists():
            raise SLAIntelligenceParseError(
                f"Org health report not found: {self.config.org_report_path}"
            )

        try:
            with open(self.config.org_report_path, 'r', encoding='utf-8') as f:
                self._org_report = json.load(f)
            logger.info(f"  Loaded org report: {self.config.org_report_path}")
        except json.JSONDecodeError as e:
            raise SLAIntelligenceParseError(f"Invalid JSON in org report: {e}")

        # Load alerts report (optional)
        if self.config.alerts_report_path and self.config.alerts_report_path.exists():
            try:
                with open(self.config.alerts_report_path, 'r', encoding='utf-8') as f:
                    self._alerts_report = json.load(f)
                logger.info(f"  Loaded alerts report: {self.config.alerts_report_path}")
            except Exception as e:
                logger.warning(f"  Could not load alerts report: {e}")

        # Load correlation report (optional)
        if self.config.correlation_report_path and self.config.correlation_report_path.exists():
            try:
                with open(self.config.correlation_report_path, 'r', encoding='utf-8') as f:
                    self._correlation_report = json.load(f)
                logger.info(f"  Loaded correlation report: {self.config.correlation_report_path}")
            except Exception as e:
                logger.warning(f"  Could not load correlation report: {e}")

        # Load temporal report (optional)
        if self.config.temporal_report_path and self.config.temporal_report_path.exists():
            try:
                with open(self.config.temporal_report_path, 'r', encoding='utf-8') as f:
                    self._temporal_report = json.load(f)
                logger.info(f"  Loaded temporal report: {self.config.temporal_report_path}")
            except Exception as e:
                logger.warning(f"  Could not load temporal report: {e}")

    def _generate_report(self) -> SLAIntelligenceReport:
        """Generate the SLA intelligence report."""
        timestamp = datetime.utcnow().isoformat()

        # Generate scorecards
        scorecards = self.readiness_engine.generate_scorecards(
            self._compliance_results, self._policies
        )

        # Generate risk narrative
        risk_narrative = self.readiness_engine.generate_risk_narrative(
            self._readiness, self._compliance_results, self._breaches
        )

        # Build summary
        summary = SLAIntelligenceSummary(
            total_slas_evaluated=len(self._compliance_results),
            compliant_slas=self._readiness.compliant_slas,
            at_risk_slas=self._readiness.at_risk_slas,
            breached_slas=self._readiness.breached_slas,
            total_breaches=len(self._breaches),
            critical_breaches=len([b for b in self._breaches
                                  if b.severity == SLASeverity.CRITICAL]),
            high_breaches=len([b for b in self._breaches
                             if b.severity == SLASeverity.HIGH]),
            active_breaches=len([b for b in self._breaches if b.status == "active"]),
            overall_compliance_rate=self._readiness.compliant_slas /
                                   max(self._readiness.total_slas, 1) * 100,
            avg_sla_health=self._readiness.sla_compliance_score,
            improving_slas=len([r for r in self._compliance_results
                               if r.trend_direction == "improving"]),
            stable_slas=len([r for r in self._compliance_results
                           if r.trend_direction == "stable"]),
            degrading_slas=len([r for r in self._compliance_results
                              if r.trend_direction == "degrading"]),
            repos_with_sla_impact=len(set(r for result in self._compliance_results
                                         for r in result.affected_repos)),
            total_repos_evaluated=len(self._org_report.get("repositories", [])),
            executive_readiness_score=self._readiness.readiness_score,
            readiness_tier=self._readiness.readiness_tier.value
        )

        # Generate recommendations
        recommendations = self._generate_recommendations()

        # Breach summary
        breach_summary = {
            "total": len(self._breaches),
            "by_severity": {
                "critical": len([b for b in self._breaches if b.severity == SLASeverity.CRITICAL]),
                "high": len([b for b in self._breaches if b.severity == SLASeverity.HIGH]),
                "medium": len([b for b in self._breaches if b.severity == SLASeverity.MEDIUM]),
                "low": len([b for b in self._breaches if b.severity == SLASeverity.LOW])
            },
            "by_status": {
                "active": len([b for b in self._breaches if b.status == "active"]),
                "at_risk": len([b for b in self._breaches if b.status == "at_risk"])
            }
        }

        report = SLAIntelligenceReport(
            report_id=f"sla_intelligence_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            generated_at=timestamp,
            org_report_path=str(self.config.org_report_path),
            alerts_report_path=str(self.config.alerts_report_path) if self.config.alerts_report_path else "",
            correlation_report_path=str(self.config.correlation_report_path) if self.config.correlation_report_path else "",
            temporal_report_path=str(self.config.temporal_report_path) if self.config.temporal_report_path else "",
            sla_policy_path=str(self.config.sla_policy_path) if self.config.sla_policy_path else "",
            summary=summary,
            executive_readiness=self._readiness,
            scorecards=scorecards,
            compliance_results=self._compliance_results,
            breaches=self._breaches,
            breach_summary=breach_summary,
            risk_narrative=risk_narrative,
            recommendations=recommendations,
            policies_evaluated=[p.policy_id for p in self._policies],
            org_health_status=self._org_report.get("org_health_status", "unknown"),
            org_health_score=self._org_report.get("org_health_score", 0.0),
            total_repos=len(self._org_report.get("repositories", []))
        )

        return report

    def _generate_recommendations(self) -> List[Dict[str, Any]]:
        """Generate prioritized recommendations."""
        recommendations = []
        rec_id = 0

        # Critical breaches
        critical_breaches = [b for b in self._breaches if b.severity == SLASeverity.CRITICAL]
        if critical_breaches:
            rec_id += 1
            recommendations.append({
                "id": f"rec_{rec_id:03d}",
                "priority": "critical",
                "title": "Address Critical SLA Breaches",
                "message": f"{len(critical_breaches)} critical SLA breach(es) require immediate attention",
                "actions": critical_breaches[0].recommended_actions[:3] if critical_breaches[0].recommended_actions else [],
                "affected_slas": [b.policy_name for b in critical_breaches]
            })

        # At-risk SLAs with high breach probability
        high_risk = [r for r in self._compliance_results
                    if r.overall_status == SLAStatus.AT_RISK and r.breach_probability > 0.5]
        if high_risk:
            rec_id += 1
            recommendations.append({
                "id": f"rec_{rec_id:03d}",
                "priority": "high",
                "title": "Prevent Imminent SLA Breaches",
                "message": f"{len(high_risk)} SLA(s) have >50% breach probability",
                "actions": [
                    "Prioritize remediation of at-risk SLAs",
                    "Review resource allocation",
                    "Consider temporary service adjustments"
                ],
                "affected_slas": [r.policy_name for r in high_risk]
            })

        # Declining repos
        if self._readiness and self._readiness.declining_repos:
            rec_id += 1
            recommendations.append({
                "id": f"rec_{rec_id:03d}",
                "priority": "medium",
                "title": "Address Declining Repositories",
                "message": f"{len(self._readiness.declining_repos)} repositories showing declining health",
                "actions": [
                    "Investigate root causes of decline",
                    "Allocate engineering resources for remediation",
                    "Implement monitoring for early warning"
                ],
                "affected_repos": self._readiness.declining_repos[:5]
            })

        # General maintenance if no critical issues
        if not recommendations:
            rec_id += 1
            recommendations.append({
                "id": f"rec_{rec_id:03d}",
                "priority": "low",
                "title": "Maintain Current Compliance",
                "message": "All SLAs are currently compliant - maintain vigilance",
                "actions": [
                    "Continue regular monitoring",
                    "Review SLA targets for potential optimization",
                    "Document current success patterns"
                ]
            })

        return recommendations

    def _write_report(self, report: SLAIntelligenceReport) -> None:
        """Write report to JSON file."""
        try:
            output_path = Path(self.config.output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report.to_dict(), f, indent=2, default=str)

        except Exception as e:
            raise SLAIntelligenceError(f"Failed to write report: {e}")

    def _determine_exit_code(self) -> int:
        """Determine appropriate exit code."""
        if self._readiness.breached_slas > 0:
            if self.config.fail_on_breach:
                return EXIT_SLA_BREACH
            return EXIT_SLA_BREACH

        if self._readiness.at_risk_slas > 0:
            if self.config.fail_on_at_risk:
                return EXIT_SLA_AT_RISK
            return EXIT_SLA_AT_RISK

        return EXIT_SLA_SUCCESS

    def _error_report(self, error_message: str) -> SLAIntelligenceReport:
        """Generate error report."""
        return SLAIntelligenceReport(
            report_id=f"sla_intelligence_error_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            generated_at=datetime.utcnow().isoformat(),
            org_report_path=str(self.config.org_report_path),
            recommendations=[{
                "id": "error",
                "priority": "critical",
                "title": "Analysis Error",
                "message": error_message,
                "actions": ["Review error and retry"]
            }]
        )


# ============================================================================
# Utility Functions
# ============================================================================

def create_default_thresholds() -> SLAThresholds:
    """Create default SLA thresholds."""
    return SLAThresholds()


def create_default_policies() -> List[SLAPolicy]:
    """Create default SLA policies."""
    loader = SLAPolicyLoader(SLAIntelligenceConfig(org_report_path=Path(".")))
    return loader._create_default_policies()
