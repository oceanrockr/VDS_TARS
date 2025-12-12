"""
Enterprise Observability API for T.A.R.S.

FastAPI application providing REST access to observability data.
"""

from typing import Optional, List, Annotated
from pathlib import Path
from datetime import datetime, timedelta
import json

from fastapi import FastAPI, Depends, HTTPException, status, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from .models import (
    HealthResponse,
    GAKPIResponse,
    DailySummaryResponse,
    AnomalyEventResponse,
    AnomalyListResponse,
    RegressionSummaryResponse,
    RetrospectiveResponse,
    LoginRequest,
    TokenResponse,
    ErrorResponse,
    GAKPIListResponse,
    DailySummaryListResponse,
)
from .security import (
    SecurityManager,
    get_security_manager,
    get_current_user_api_key,
    require_role,
    User,
)
from enterprise_config.schema import RBACRole


# Initialize app
app = FastAPI(
    title="T.A.R.S. Enterprise Observability API",
    description="REST API for accessing T.A.R.S. observability data",
    version="1.0.2-dev",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Rate limiting
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure from enterprise_config in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data directories (configure from enterprise_config in production)
DATA_DIR = Path("output")
GA_KPI_DIR = DATA_DIR / "ga_kpis"
STABILITY_DIR = DATA_DIR / "stability"
ANOMALY_DIR = DATA_DIR / "anomalies"
REGRESSION_DIR = DATA_DIR / "regression"
RETROSPECTIVE_DIR = Path("docs/final")


# Health Check Endpoints

@app.get(
    "/healthz",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Health check endpoint",
)
@limiter.limit("30/minute")
async def health_check(
    request: Response,
) -> HealthResponse:
    """
    Health check endpoint (no authentication required).

    Returns service health status.
    """
    return HealthResponse(
        status="healthy",
        version="1.0.2-dev",
        timestamp=datetime.utcnow(),
        checks={
            "api": True,
            "data_dir": DATA_DIR.exists(),
        }
    )


# Authentication Endpoints

@app.post(
    "/auth/login",
    response_model=TokenResponse,
    tags=["Authentication"],
    summary="Login with username/password",
)
@limiter.limit("10/minute")
async def login(
    request: Response,
    credentials: LoginRequest,
    security_manager: Annotated[SecurityManager, Depends(get_security_manager)],
) -> TokenResponse:
    """
    Authenticate with username and password to obtain JWT token.

    **Note:** Only available if auth_mode=jwt in configuration.
    """
    user = security_manager.authenticate_user(
        credentials.username,
        credentials.password
    )

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token = security_manager.create_access_token(user)

    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
        expires_in=security_manager.token_expiration_minutes * 60,
    )


# GA KPI Endpoints

@app.get(
    "/ga",
    response_model=GAKPIResponse,
    tags=["GA KPI"],
    summary="Get GA Day KPI summary",
)
@limiter.limit("100/minute")
async def get_ga_kpi(
    request: Response,
    current_user: Annotated[User, Depends(get_current_user_api_key)],
) -> GAKPIResponse:
    """
    Get GA Day KPI summary.

    **Required role:** readonly
    """
    kpi_file = GA_KPI_DIR / "ga_kpi_summary.json"

    if not kpi_file.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="GA KPI summary not found"
        )

    with open(kpi_file, "r") as f:
        data = json.load(f)

    return GAKPIResponse(
        ga_timestamp=data.get("ga_timestamp", ""),
        overall_availability=data.get("overall_availability", 0.0),
        error_rate=data.get("error_rate", 0.0),
        p99_latency_ms=data.get("p99_latency_ms", 0.0),
        avg_cpu_percent=data.get("avg_cpu_percent", 0.0),
        avg_memory_percent=data.get("avg_memory_percent", 0.0),
        cost_estimate=data.get("cost_estimate", 0.0),
        metadata=data,
    )


@app.get(
    "/ga/files",
    response_model=GAKPIListResponse,
    tags=["GA KPI"],
    summary="List available GA KPI files",
)
@limiter.limit("100/minute")
async def list_ga_kpi_files(
    request: Response,
    current_user: Annotated[User, Depends(get_current_user_api_key)],
) -> GAKPIListResponse:
    """
    List all available GA KPI files.

    **Required role:** readonly
    """
    if not GA_KPI_DIR.exists():
        return GAKPIListResponse(files=[], count=0)

    files = [f.name for f in GA_KPI_DIR.glob("*.json")]

    return GAKPIListResponse(files=files, count=len(files))


# Daily Summary Endpoints

@app.get(
    "/day/{day_number}",
    response_model=DailySummaryResponse,
    tags=["Daily Summaries"],
    summary="Get daily stability summary",
)
@limiter.limit("100/minute")
async def get_daily_summary(
    request: Response,
    day_number: int,
    current_user: Annotated[User, Depends(get_current_user_api_key)],
) -> DailySummaryResponse:
    """
    Get stability summary for a specific day.

    **Required role:** readonly

    **Parameters:**
    - day_number: Day number (0-7, where 0 is GA Day)
    """
    if day_number < 0 or day_number > 7:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Day number must be between 0 and 7"
        )

    summary_file = STABILITY_DIR / f"day_{day_number:02d}_summary.json"

    if not summary_file.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Summary for day {day_number} not found"
        )

    with open(summary_file, "r") as f:
        data = json.load(f)

    return DailySummaryResponse(
        day_number=day_number,
        timestamp=data.get("timestamp", ""),
        metrics=data.get("metrics", {}),
        drift_vs_ga=data.get("drift_vs_ga", {}),
        status=data.get("status", "unknown"),
        rollback_recommended=data.get("rollback_recommended", False),
    )


@app.get(
    "/day",
    response_model=DailySummaryListResponse,
    tags=["Daily Summaries"],
    summary="List available daily summaries",
)
@limiter.limit("100/minute")
async def list_daily_summaries(
    request: Response,
    current_user: Annotated[User, Depends(get_current_user_api_key)],
) -> DailySummaryListResponse:
    """
    List all available daily summaries.

    **Required role:** readonly
    """
    if not STABILITY_DIR.exists():
        return DailySummaryListResponse(summaries=[], count=0)

    files = list(STABILITY_DIR.glob("day_*_summary.json"))
    day_numbers = []

    for f in files:
        # Extract day number from filename
        import re
        match = re.search(r"day_(\d+)_summary\.json", f.name)
        if match:
            day_numbers.append(int(match.group(1)))

    day_numbers.sort()

    return DailySummaryListResponse(summaries=day_numbers, count=len(day_numbers))


# Anomaly Endpoints

@app.get(
    "/anomalies",
    response_model=AnomalyListResponse,
    tags=["Anomalies"],
    summary="Get all anomaly events",
)
@limiter.limit("100/minute")
async def get_anomalies(
    request: Response,
    current_user: Annotated[User, Depends(get_current_user_api_key)],
    severity: Optional[str] = None,
) -> AnomalyListResponse:
    """
    Get all detected anomaly events.

    **Required role:** readonly

    **Query parameters:**
    - severity: Filter by severity (low, medium, high, critical)
    """
    anomaly_file = ANOMALY_DIR / "anomaly_events.json"

    if not anomaly_file.exists():
        return AnomalyListResponse(
            anomalies=[],
            count=0,
            total_critical=0,
            total_high=0,
        )

    with open(anomaly_file, "r") as f:
        data = json.load(f)

    anomalies = []
    for event in data.get("events", []):
        anomaly = AnomalyEventResponse(
            timestamp=event.get("timestamp", ""),
            metric_name=event.get("metric_name", ""),
            value=event.get("value", 0.0),
            baseline_value=event.get("baseline_value", 0.0),
            z_score=event.get("z_score", 0.0),
            threshold=event.get("threshold", 3.0),
            severity=event.get("severity", "medium"),
        )

        # Filter by severity if specified
        if severity is None or anomaly.severity == severity:
            anomalies.append(anomaly)

    # Count by severity
    total_critical = sum(1 for a in anomalies if a.severity == "critical")
    total_high = sum(1 for a in anomalies if a.severity == "high")

    return AnomalyListResponse(
        anomalies=anomalies,
        count=len(anomalies),
        total_critical=total_critical,
        total_high=total_high,
    )


# Regression Endpoints

@app.get(
    "/regressions",
    response_model=RegressionSummaryResponse,
    tags=["Regressions"],
    summary="Get regression analysis summary",
)
@limiter.limit("100/minute")
async def get_regressions(
    request: Response,
    current_user: Annotated[User, Depends(get_current_user_api_key)],
) -> RegressionSummaryResponse:
    """
    Get regression analysis summary.

    **Required role:** readonly
    """
    regression_file = REGRESSION_DIR / "regression_summary.json"

    if not regression_file.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Regression summary not found"
        )

    with open(regression_file, "r") as f:
        data = json.load(f)

    regressions = data.get("regressions", [])

    # Count by severity
    critical = sum(1 for r in regressions if r.get("severity") == "critical")
    high = sum(1 for r in regressions if r.get("severity") == "high")
    medium = sum(1 for r in regressions if r.get("severity") == "medium")

    return RegressionSummaryResponse(
        total_regressions=len(regressions),
        critical_regressions=critical,
        high_regressions=high,
        medium_regressions=medium,
        rollback_recommended=data.get("rollback_recommended", False),
        regressions=regressions,
    )


# Retrospective Endpoints

@app.get(
    "/retrospective",
    response_model=RetrospectiveResponse,
    tags=["Retrospective"],
    summary="Get retrospective metadata",
)
@limiter.limit("100/minute")
async def get_retrospective_metadata(
    request: Response,
    current_user: Annotated[User, Depends(get_current_user_api_key)],
) -> RetrospectiveResponse:
    """
    Get retrospective report metadata (not full report).

    **Required role:** readonly

    Use `/retrospective/markdown` or `/retrospective/json` to download full report.
    """
    retro_json = RETROSPECTIVE_DIR / "GA_7DAY_RETROSPECTIVE.json"

    if not retro_json.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Retrospective report not found"
        )

    with open(retro_json, "r") as f:
        data = json.load(f)

    return RetrospectiveResponse(
        generation_timestamp=data.get("generation_timestamp", ""),
        ga_day_timestamp=data.get("ga_day_timestamp", ""),
        seven_day_end_timestamp=data.get("seven_day_end_timestamp", ""),
        overall_status=data.get("overall_status", "unknown"),
        success_count=len(data.get("successes", [])),
        degradation_count=len(data.get("degradations", [])),
        drift_count=len(data.get("unexpected_drifts", [])),
        recommendation_count=len(data.get("recommendations_v1_0_2", [])),
        available_formats=["markdown", "json"],
    )


@app.get(
    "/retrospective/markdown",
    tags=["Retrospective"],
    summary="Download retrospective (Markdown)",
)
@limiter.limit("50/minute")
async def get_retrospective_markdown(
    request: Response,
    current_user: Annotated[User, Depends(get_current_user_api_key)],
):
    """
    Download full retrospective report in Markdown format.

    **Required role:** readonly
    """
    retro_md = RETROSPECTIVE_DIR / "GA_7DAY_RETROSPECTIVE.md"

    if not retro_md.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Retrospective Markdown report not found"
        )

    return FileResponse(
        path=retro_md,
        media_type="text/markdown",
        filename="retrospective.md",
    )


@app.get(
    "/retrospective/json",
    tags=["Retrospective"],
    summary="Download retrospective (JSON)",
)
@limiter.limit("50/minute")
async def get_retrospective_json(
    request: Response,
    current_user: Annotated[User, Depends(get_current_user_api_key)],
):
    """
    Download full retrospective report in JSON format.

    **Required role:** readonly
    """
    retro_json = RETROSPECTIVE_DIR / "GA_7DAY_RETROSPECTIVE.json"

    if not retro_json.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Retrospective JSON report not found"
        )

    return FileResponse(
        path=retro_json,
        media_type="application/json",
        filename="retrospective.json",
    )


# Error handlers

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    """Custom HTTP exception handler."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            detail=str(exc.detail),
        ).dict(),
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    """General exception handler."""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc),
        ).dict(),
    )


# Startup/Shutdown

@app.on_event("startup")
async def startup_event():
    """Startup event handler."""
    print("T.A.R.S. Enterprise Observability API starting...")
    print(f"Version: 1.0.2-dev")
    print(f"Data directory: {DATA_DIR}")


@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event handler."""
    print("T.A.R.S. Enterprise Observability API shutting down...")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8100)
