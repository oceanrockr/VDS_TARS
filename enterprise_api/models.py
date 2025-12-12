"""
API Models for T.A.R.S. Enterprise API

Pydantic models for request/response validation.
"""

from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field
from datetime import datetime


# Authentication Models

class LoginRequest(BaseModel):
    """Login request for JWT authentication."""
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=8)


class TokenResponse(BaseModel):
    """JWT token response."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int  # seconds


class APIKeyRequest(BaseModel):
    """API key creation request."""
    name: str = Field(..., description="API key name")
    role: Literal["readonly", "sre", "admin"] = Field(..., description="API key role")


class APIKeyResponse(BaseModel):
    """API key creation response."""
    api_key: str
    name: str
    role: str
    created_at: datetime


# Data Models

class HealthResponse(BaseModel):
    """Health check response."""
    status: Literal["healthy", "degraded", "unhealthy"]
    version: str
    timestamp: datetime
    checks: Dict[str, bool] = Field(default_factory=dict)


class GAKPIResponse(BaseModel):
    """GA Day KPI response."""
    ga_timestamp: str
    overall_availability: float
    error_rate: float
    p99_latency_ms: float
    avg_cpu_percent: float
    avg_memory_percent: float
    cost_estimate: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DailySummaryResponse(BaseModel):
    """Daily stability summary response."""
    day_number: int
    timestamp: str
    metrics: Dict[str, Any]
    drift_vs_ga: Dict[str, float] = Field(default_factory=dict)
    status: str
    rollback_recommended: bool = False


class AnomalyEventResponse(BaseModel):
    """Anomaly event response."""
    timestamp: str
    metric_name: str
    value: float
    baseline_value: float
    z_score: float
    threshold: float
    severity: Literal["low", "medium", "high", "critical"]


class RegressionSummaryResponse(BaseModel):
    """Regression analysis summary response."""
    total_regressions: int
    critical_regressions: int
    high_regressions: int
    medium_regressions: int
    rollback_recommended: bool
    regressions: List[Dict[str, Any]]


class RetrospectiveResponse(BaseModel):
    """Retrospective report response (metadata only)."""
    generation_timestamp: str
    ga_day_timestamp: str
    seven_day_end_timestamp: str
    overall_status: str
    success_count: int
    degradation_count: int
    drift_count: int
    recommendation_count: int
    available_formats: List[str] = Field(default=["markdown", "json"])


# Error Models

class ErrorResponse(BaseModel):
    """Error response."""
    error: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# List Responses

class GAKPIListResponse(BaseModel):
    """List of GA KPI files."""
    files: List[str]
    count: int


class DailySummaryListResponse(BaseModel):
    """List of daily summaries."""
    summaries: List[int]  # Day numbers
    count: int


class AnomalyListResponse(BaseModel):
    """List of anomaly events."""
    anomalies: List[AnomalyEventResponse]
    count: int
    total_critical: int
    total_high: int
