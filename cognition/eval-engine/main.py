"""
Evaluation Engine - Main FastAPI Application
Port: 8099

Real RL agent evaluation with regression detection and baseline management.
"""
import sys
import os
import logging
from contextlib import asynccontextmanager
import asyncpg
import redis.asyncio as aioredis
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Histogram, Gauge

# Add shared to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'shared'))
from auth_routes import router as auth_router
from rate_limiter import rate_limit_middleware

from config import EvalEngineConfig
from routes import eval_router, baseline_router, health_router
from environment_manager import EnvironmentCache
from metrics_calculator import MetricsCalculator
from regression_detector import RegressionDetector
from nash_scorer import NashScorer
from baseline_manager import BaselineManager
from workers import AgentEvaluationWorker
from instrumentation import setup_tracing

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ============================================================================
# Prometheus Metrics
# ============================================================================

EVALUATIONS_TOTAL = Counter(
    "tars_eval_evaluations_total",
    "Total agent evaluations",
    ["agent_type", "environment", "status"]
)

REGRESSION_DETECTED_TOTAL = Counter(
    "tars_eval_regression_detected_total",
    "Total regressions detected",
    ["agent_type", "environment"]
)

EVALUATION_DURATION = Histogram(
    "tars_eval_duration_seconds",
    "Evaluation duration in seconds",
    ["agent_type", "num_episodes"],
    buckets=[10, 30, 60, 120, 300, 600]
)

EPISODES_TOTAL = Counter(
    "tars_eval_episodes_total",
    "Total episodes executed",
    ["agent_type", "environment"]
)

ENV_CACHE_SIZE = Gauge(
    "tars_eval_env_cache_size",
    "Number of cached environments"
)


# Global state
db_pool: asyncpg.Pool = None
redis_client: aioredis.Redis = None
config: EvalEngineConfig = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan management.
    Initialize DB connections on startup, cleanup on shutdown.
    """
    global db_pool, redis_client, config

    # TODO: Implement lifespan startup
    # 1. config = EvalEngineConfig.from_env()
    # 2. logger.info("Starting Evaluation Engine on port %d", config.port)
    #
    # 3. # Initialize distributed tracing
    #    setup_tracing(
    #        app,
    #        service_name="tars-eval-engine",
    #        service_version="v1.0.0-rc2",
    #        jaeger_endpoint=os.getenv("JAEGER_ENDPOINT", "http://jaeger:4317"),
    #        sample_rate=float(os.getenv("TRACE_SAMPLE_RATE", "0.1"))  # 10% sampling in prod
    #    )
    #    logger.info("Distributed tracing initialized")
    #
    # 4. # Initialize PostgreSQL pool
    #    db_pool = await asyncpg.create_pool(
    #        config.postgres_url,
    #        min_size=5,
    #        max_size=20,
    #        command_timeout=60
    #    )
    #    logger.info("PostgreSQL pool created")
    #
    # 5. # Initialize Redis client
    #    redis_client = await aioredis.from_url(
    #        config.redis_url,
    #        encoding="utf-8",
    #        decode_responses=True
    #    )
    #    logger.info("Redis client created")
    #
    # 6. # Initialize components
    #    env_manager = EnvironmentCache(max_size=config.env_cache_size)
    #    metrics_calc = MetricsCalculator()
    #    regression_detector = RegressionDetector(config.thresholds)
    #    baseline_manager = BaselineManager(db_pool)
    #
    #    # Store in app.state
    #    app.state.db_pool = db_pool
    #    app.state.redis_client = redis_client
    #    app.state.config = config
    #    app.state.env_manager = env_manager
    #    app.state.metrics_calc = metrics_calc
    #    app.state.regression_detector = regression_detector
    #    app.state.baseline_manager = baseline_manager
    #    app.state.worker = AgentEvaluationWorker(env_manager, metrics_calc, regression_detector)
    #
    #    logger.info("Evaluation Engine started successfully")

    yield

    # TODO: Implement lifespan shutdown
    # 1. logger.info("Shutting down Evaluation Engine")
    # 2. await app.state.env_manager.close_all()
    # 3. await db_pool.close()
    # 4. await redis_client.close()
    # 5. logger.info("Evaluation Engine shut down successfully")


app = FastAPI(
    title="T.A.R.S. Evaluation Engine",
    description="Real RL agent evaluation with regression detection and baseline management",
    version="v1.0.0-rc2",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add rate limiting middleware
app.middleware("http")(rate_limit_middleware)

# Include routers
app.include_router(auth_router)
app.include_router(health_router)
app.include_router(eval_router)
app.include_router(baseline_router)


if __name__ == "__main__":
    import uvicorn
    # TODO: Load config and run
    # config = EvalEngineConfig.from_env()
    # uvicorn.run(app, host="0.0.0.0", port=config.port, log_level="info")
    uvicorn.run(app, host="0.0.0.0", port=8099, log_level="info")
