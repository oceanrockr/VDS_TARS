"""
Pydantic models for Cognitive Analytics Core
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum


class InsightType(str, Enum):
    POLICY_OPTIMIZATION = "policy_optimization"
    CONSENSUS_TUNING = "consensus_tuning"
    ETHICAL_THRESHOLD = "ethical_threshold"
    ANOMALY_CORRELATION = "anomaly_correlation"
    RESOURCE_EFFICIENCY = "resource_efficiency"


class InsightPriority(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class AdaptiveInsight(BaseModel):
    """Cognitive recommendation generated from analytics"""
    id: str = Field(..., description="Unique insight identifier")
    type: InsightType
    priority: InsightPriority
    title: str
    description: str
    recommendation: Dict[str, Any] = Field(..., description="Actionable recommendation")
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    evidence: Dict[str, Any] = Field(default_factory=dict, description="Supporting data")
    impact_estimate: Dict[str, float] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    status: str = Field(default="pending", description="pending/approved/rejected/applied")

    class Config:
        json_schema_extra = {
            "example": {
                "id": "insight-001",
                "type": "policy_optimization",
                "priority": "high",
                "title": "Reduce scaling cooldown period",
                "description": "Analysis shows 85% of scaling decisions succeed within 30s cooldown",
                "recommendation": {
                    "policy": "operational/scaling",
                    "parameter": "cooldown_seconds",
                    "current_value": 60,
                    "suggested_value": 30,
                    "change_type": "reduce"
                },
                "confidence_score": 0.87,
                "evidence": {
                    "sample_size": 1450,
                    "success_rate": 0.85,
                    "mean_decision_time": 28.3,
                    "p95_decision_time": 42.1
                },
                "impact_estimate": {
                    "latency_reduction_percent": 20.5,
                    "violation_risk_increase": 2.1
                },
                "status": "pending"
            }
        }


class PolicyMetrics(BaseModel):
    """Metrics derived from policy audit trail"""
    policy_id: str
    policy_type: str
    total_evaluations: int
    allow_count: int
    deny_count: int
    warn_count: int
    avg_evaluation_latency_ms: float
    violation_rate: float
    false_positive_rate: Optional[float] = None
    window_start: datetime
    window_end: datetime


class ConsensusMetrics(BaseModel):
    """Metrics from consensus operations"""
    algorithm: str  # raft/pbft
    total_votes: int
    avg_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    success_rate: float
    quorum_failures: int
    window_start: datetime
    window_end: datetime


class AnomalyCorrelation(BaseModel):
    """Correlation between anomalies and policy decisions"""
    anomaly_type: str
    policy_id: str
    correlation_coefficient: float = Field(..., ge=-1.0, le=1.0)
    sample_size: int
    avg_mttr_seconds: float  # Mean Time To Remediation
    policy_effectiveness_score: float = Field(..., ge=0.0, le=1.0)


class EthicalMetrics(BaseModel):
    """Metrics for ethical policy evaluation"""
    policy_id: str
    fairness_score: float = Field(..., ge=0.0, le=1.0)
    demographic_balance: Dict[str, float]
    bias_detected: bool
    underrepresented_groups: List[str] = Field(default_factory=list)
    total_decisions: int
    deny_rate_by_group: Dict[str, float] = Field(default_factory=dict)
    window_start: datetime
    window_end: datetime


class CognitiveState(BaseModel):
    """Overall cognitive system state"""
    last_analysis_timestamp: datetime
    total_insights_generated: int
    insights_applied: int
    insights_rejected: int
    avg_confidence_score: float
    policy_adaptation_success_rate: float
    consensus_improvement_percent: float
    ethical_fairness_delta: float
    active_insights: List[str] = Field(default_factory=list)


class InsightRecommendationRequest(BaseModel):
    """Request for insight recommendations"""
    insight_types: Optional[List[InsightType]] = None
    min_confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    priority: Optional[InsightPriority] = None
    limit: int = Field(default=10, ge=1, le=100)


class InsightRecommendationResponse(BaseModel):
    """Response with cognitive recommendations"""
    insights: List[AdaptiveInsight]
    total_count: int
    avg_confidence: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)
