"""
Cognitive Analytics Core - FastAPI Server
Insight Engine for T.A.R.S. Phase 10
"""
import os
import logging
import asyncio
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

from models import (
    InsightRecommendationRequest, InsightRecommendationResponse,
    CognitiveState, InsightType, InsightPriority
)
from stream_processor import StreamProcessor
from recommender import CognitiveRecommender

# Configuration
DB_URL = os.getenv(
    "POSTGRES_URL",
    "postgresql://tars:tars@postgres:5432/tars"
)
INSIGHT_REFRESH_INTERVAL = int(os.getenv("INSIGHT_REFRESH_INTERVAL", "60"))
ANALYSIS_WINDOW_MINUTES = int(os.getenv("ANALYSIS_WINDOW_MINUTES", "60"))

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Prometheus Metrics
RECOMMENDATION_SCORE = Gauge(
    'tars_cognitive_recommendation_score',
    'Cognitive recommendation score for insights',
    ['insight_id', 'type', 'priority']
)

INSIGHTS_GENERATED = Counter(
    'tars_cognitive_insights_generated_total',
    'Total cognitive insights generated',
    ['type', 'priority']
)

INSIGHTS_APPLIED = Counter(
    'tars_cognitive_insights_applied_total',
    'Total cognitive insights successfully applied',
    ['type']
)

INSIGHT_PROCESSING_TIME = Histogram(
    'tars_cognitive_insight_processing_seconds',
    'Time to generate insights',
    buckets=[0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0]
)

RECOMMENDATION_REQUESTS = Counter(
    'tars_cognitive_recommendation_requests_total',
    'Total recommendation API requests'
)

# Global services
stream_processor: Optional[StreamProcessor] = None
recommender: Optional[CognitiveRecommender] = None
processor_task: Optional[asyncio.Task] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle management for FastAPI app"""
    global stream_processor, recommender, processor_task

    logger.info("Starting Cognitive Analytics Core")

    # Initialize services
    stream_processor = StreamProcessor(
        db_url=DB_URL,
        analysis_window_minutes=ANALYSIS_WINDOW_MINUTES,
        insight_refresh_interval=INSIGHT_REFRESH_INTERVAL
    )
    recommender = CognitiveRecommender(db_url=DB_URL)

    await stream_processor.initialize()
    await recommender.initialize()

    # Create database tables
    await create_tables()

    # Start background processor
    processor_task = asyncio.create_task(stream_processor.run_continuous())

    logger.info("Cognitive Analytics Core ready")

    yield

    # Shutdown
    logger.info("Shutting down Cognitive Analytics Core")
    if processor_task:
        processor_task.cancel()
        try:
            await processor_task
        except asyncio.CancelledError:
            pass

    await stream_processor.shutdown()
    await recommender.shutdown()


app = FastAPI(
    title="T.A.R.S. Cognitive Analytics Core",
    description="Insight Engine for adaptive policy learning and consensus optimization",
    version="0.8.0-alpha",
    lifespan=lifespan
)


async def create_tables():
    """Create required database tables"""
    import asyncpg

    conn = await asyncpg.connect(DB_URL)

    try:
        # Cognitive insights table
        await conn.execute("""
        CREATE TABLE IF NOT EXISTS cognitive_insights (
            id VARCHAR(255) PRIMARY KEY,
            type VARCHAR(50) NOT NULL,
            priority VARCHAR(20) NOT NULL,
            title TEXT NOT NULL,
            description TEXT NOT NULL,
            recommendation JSONB NOT NULL,
            confidence_score FLOAT NOT NULL,
            evidence JSONB,
            impact_estimate JSONB,
            status VARCHAR(20) DEFAULT 'pending',
            timestamp TIMESTAMP DEFAULT NOW(),
            expires_at TIMESTAMP
        )
        """)

        # Create indexes
        await conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_insights_status ON cognitive_insights(status);
        CREATE INDEX IF NOT EXISTS idx_insights_type ON cognitive_insights(type);
        CREATE INDEX IF NOT EXISTS idx_insights_timestamp ON cognitive_insights(timestamp DESC);
        CREATE INDEX IF NOT EXISTS idx_insights_confidence ON cognitive_insights(confidence_score DESC);
        """)

        logger.info("Database tables initialized")

    finally:
        await conn.close()


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "cognitive-analytics-core",
        "version": "0.8.0-alpha"
    }


@app.post("/api/v1/insights/recommendations", response_model=InsightRecommendationResponse)
async def get_recommendations(request: InsightRecommendationRequest):
    """Get cognitive recommendations"""
    RECOMMENDATION_REQUESTS.inc()

    try:
        response = await recommender.get_recommendations(request)

        # Update Prometheus metrics
        for insight in response.insights:
            score = await recommender.get_recommendation_score(insight.id)
            RECOMMENDATION_SCORE.labels(
                insight_id=insight.id,
                type=insight.type.value,
                priority=insight.priority.value
            ).set(score)

        return response

    except Exception as e:
        logger.error(f"Error fetching recommendations: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/insights/{insight_id}")
async def get_insight(insight_id: str):
    """Get specific insight by ID"""
    import asyncpg

    conn = await asyncpg.connect(DB_URL)
    try:
        row = await conn.fetchrow(
            "SELECT * FROM cognitive_insights WHERE id = $1",
            insight_id
        )

        if not row:
            raise HTTPException(status_code=404, detail="Insight not found")

        return dict(row)

    finally:
        await conn.close()


@app.post("/api/v1/insights/{insight_id}/status")
async def update_insight_status(insight_id: str, status: str):
    """Update insight status"""
    valid_statuses = ["pending", "approved", "rejected", "applied"]
    if status not in valid_statuses:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid status. Must be one of: {valid_statuses}"
        )

    success = await recommender.update_insight_status(insight_id, status)

    if not success:
        raise HTTPException(status_code=404, detail="Insight not found")

    # Update metrics
    if status == "applied":
        import asyncpg
        conn = await asyncpg.connect(DB_URL)
        try:
            row = await conn.fetchrow(
                "SELECT type FROM cognitive_insights WHERE id = $1",
                insight_id
            )
            if row:
                INSIGHTS_APPLIED.labels(type=row['type']).inc()
        finally:
            await conn.close()

    return {"insight_id": insight_id, "status": status, "updated": True}


@app.get("/api/v1/state", response_model=CognitiveState)
async def get_cognitive_state():
    """Get overall cognitive system state"""
    try:
        state = await recommender.get_cognitive_state()
        return state
    except Exception as e:
        logger.error(f"Error fetching cognitive state: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/insights/trigger")
async def trigger_insight_generation():
    """Manually trigger insight generation"""
    try:
        with INSIGHT_PROCESSING_TIME.time():
            insights = await stream_processor.generate_insights()
            await stream_processor.save_insights(insights)

        # Update metrics
        for insight in insights:
            INSIGHTS_GENERATED.labels(
                type=insight.type.value,
                priority=insight.priority.value
            ).inc()

        return {
            "triggered": True,
            "insights_generated": len(insights),
            "timestamp": insights[0].timestamp.isoformat() if insights else None
        }

    except Exception as e:
        logger.error(f"Error triggering insights: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8090, log_level="info")
