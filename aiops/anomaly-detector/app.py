"""
T.A.R.S. AI-Powered Anomaly Detection Service
Ingests metrics/logs/traces and detects anomalies using ML models.
"""
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from prometheus_client import make_asgi_app, Gauge, Counter
import uvicorn
import logging
from typing import Dict, List, Optional
from pydantic import BaseModel
from datetime import datetime, timedelta

from models import AnomalyDetector
from prom_client import PrometheusClient, LokiClient, JaegerClient
from webhooks import AlertmanagerWebhook

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="T.A.R.S. Anomaly Detector",
    description="ML-based anomaly detection for metrics, logs, and traces",
    version="1.0.0"
)

# Prometheus metrics
anomaly_score_gauge = Gauge(
    'tars_anomaly_score',
    'Current anomaly score for a signal',
    ['signal', 'service']
)
anomalies_detected_counter = Counter(
    'tars_anomalies_detected_total',
    'Total number of anomalies detected',
    ['signal', 'severity']
)
predictions_counter = Counter(
    'tars_predictions_total',
    'Total number of predictions made',
    ['status']
)

# Configuration (from env vars in production)
config = {
    "prometheus_url": "http://prometheus-kube-prometheus-prometheus:9090",
    "loki_url": "http://loki:3100",
    "jaeger_url": "http://jaeger-query:16686",
    "window_hours": 24,
    "refresh_seconds": 30,
    "score_threshold": 0.8,
    "pushgateway_url": "http://prometheus-pushgateway:9091"
}

# Initialize clients
prom_client = PrometheusClient(config["prometheus_url"])
loki_client = LokiClient(config["loki_url"])
jaeger_client = JaegerClient(config["jaeger_url"])

# Initialize anomaly detector
detector = AnomalyDetector(
    window_hours=config["window_hours"],
    score_threshold=config["score_threshold"]
)

# Alertmanager webhook handler
alertmanager_handler = AlertmanagerWebhook(detector, prom_client)


class PredictionRequest(BaseModel):
    """Request model for on-demand predictions"""
    signal: str
    service: str
    metric_query: str
    lookback_hours: Optional[int] = 24


class PredictionResponse(BaseModel):
    """Response model for predictions"""
    signal: str
    service: str
    is_anomaly: bool
    score: float
    confidence: float
    severity: str
    reason: str
    timestamp: datetime
    recommendations: List[str]


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: datetime
    services: Dict[str, str]


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "service": "T.A.R.S. Anomaly Detector",
        "version": "1.0.0",
        "status": "operational"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    services_status = {
        "prometheus": "unknown",
        "loki": "unknown",
        "jaeger": "unknown"
    }

    # Check Prometheus
    try:
        prom_client.query("up")
        services_status["prometheus"] = "healthy"
    except Exception as e:
        logger.error(f"Prometheus health check failed: {e}")
        services_status["prometheus"] = "unhealthy"

    # Check Loki
    try:
        loki_client.query_range(
            '{namespace="tars"}',
            start=datetime.now() - timedelta(minutes=5),
            end=datetime.now(),
            limit=1
        )
        services_status["loki"] = "healthy"
    except Exception as e:
        logger.error(f"Loki health check failed: {e}")
        services_status["loki"] = "unhealthy"

    # Overall status
    overall_status = "healthy" if all(
        s == "healthy" for s in services_status.values()
    ) else "degraded"

    return HealthResponse(
        status=overall_status,
        timestamp=datetime.now(),
        services=services_status
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    On-demand anomaly prediction for a specific metric.

    Args:
        request: Prediction request with signal, service, and metric query

    Returns:
        Prediction response with anomaly verdict and details
    """
    try:
        logger.info(f"Prediction request for {request.service}/{request.signal}")
        predictions_counter.labels(status="requested").inc()

        # Query metrics
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=request.lookback_hours)

        data = prom_client.query_range(
            request.metric_query,
            start=start_time,
            end=end_time,
            step="30s"
        )

        if not data:
            raise HTTPException(
                status_code=404,
                detail=f"No data found for query: {request.metric_query}"
            )

        # Run anomaly detection
        result = detector.detect(
            signal_name=request.signal,
            service_name=request.service,
            data=data
        )

        # Update metrics
        anomaly_score_gauge.labels(
            signal=request.signal,
            service=request.service
        ).set(result["score"])

        if result["is_anomaly"]:
            anomalies_detected_counter.labels(
                signal=request.signal,
                severity=result["severity"]
            ).inc()
            predictions_counter.labels(status="anomaly_detected").inc()
        else:
            predictions_counter.labels(status="normal").inc()

        # Push to Pushgateway
        try:
            from prometheus_client import CollectorRegistry, push_to_gateway
            registry = CollectorRegistry()
            g = Gauge(
                'tars_anomaly_score',
                'Anomaly score',
                ['signal', 'service'],
                registry=registry
            )
            g.labels(signal=request.signal, service=request.service).set(result["score"])
            push_to_gateway(
                config["pushgateway_url"],
                job='anomaly-detector',
                registry=registry
            )
        except Exception as e:
            logger.warning(f"Failed to push to Pushgateway: {e}")

        return PredictionResponse(
            signal=request.signal,
            service=request.service,
            is_anomaly=result["is_anomaly"],
            score=result["score"],
            confidence=result["confidence"],
            severity=result["severity"],
            reason=result["reason"],
            timestamp=datetime.now(),
            recommendations=result.get("recommendations", [])
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        predictions_counter.labels(status="error").inc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/webhook/alertmanager")
async def alertmanager_webhook(request: Request):
    """
    Webhook receiver for Alertmanager alerts.
    Correlates alerts with anomaly detection and generates incident IDs.
    """
    try:
        payload = await request.json()
        logger.info(f"Received Alertmanager webhook with {len(payload.get('alerts', []))} alerts")

        # Process through webhook handler
        response = await alertmanager_handler.process(payload)

        return JSONResponse(content=response)

    except Exception as e:
        logger.error(f"Alertmanager webhook processing failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/anomalies/recent")
async def get_recent_anomalies(hours: int = 24, limit: int = 100):
    """
    Get recent anomalies detected in the last N hours.

    Args:
        hours: Lookback window in hours
        limit: Maximum number of anomalies to return

    Returns:
        List of recent anomalies with details
    """
    try:
        anomalies = detector.get_recent_anomalies(hours=hours, limit=limit)
        return {
            "count": len(anomalies),
            "anomalies": anomalies,
            "timestamp": datetime.now()
        }
    except Exception as e:
        logger.error(f"Failed to fetch recent anomalies: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models/status")
async def get_model_status():
    """
    Get status and performance metrics of anomaly detection models.

    Returns:
        Model statistics and performance metrics
    """
    try:
        status = detector.get_model_status()
        return {
            "status": status,
            "timestamp": datetime.now()
        }
    except Exception as e:
        logger.error(f"Failed to get model status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Mount Prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


if __name__ == "__main__":
    import os

    # Load config from environment
    config["prometheus_url"] = os.getenv("PROMETHEUS_URL", config["prometheus_url"])
    config["loki_url"] = os.getenv("LOKI_URL", config["loki_url"])
    config["jaeger_url"] = os.getenv("JAEGER_URL", config["jaeger_url"])
    config["score_threshold"] = float(os.getenv("SCORE_THRESHOLD", config["score_threshold"]))

    # Run server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8080,
        log_level="info"
    )
