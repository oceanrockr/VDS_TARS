"""
T.A.R.S. AutoML Service
Comprehensive FastAPI service integrating optimization, feature engineering, and model registry

Provides REST API for automated hyperparameter tuning and model management.
"""
import os
import sys
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import time
import pandas as pd
import numpy as np

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Request
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST

from optimizer import OptunaOptimizer
from feature_engineer import FeatureEngineer
from registry import ModelRegistry
from objective import create_objective

# Import auth components
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'shared'))
from auth import get_current_user, verify_service_key, User, Role
from auth_routes import router as auth_router
from rate_limiter import public_rate_limit, rate_limit_middleware

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="T.A.R.S. AutoML Service",
    description="Automated Hyperparameter Optimization, Feature Engineering, and Model Registry",
    version="0.9.5-alpha",
)

# Add auth routes
app.include_router(auth_router)

# Add rate limiting middleware
app.middleware("http")(rate_limit_middleware)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus metrics
optimization_trials_total = Counter(
    'tars_automl_trials_total',
    'Total number of optimization trials',
    ['agent_type', 'status']
)
best_score_gauge = Gauge(
    'tars_automl_best_score',
    'Best score achieved for each agent',
    ['agent_type']
)
featuregen_time_seconds = Histogram(
    'tars_featuregen_time_seconds',
    'Time spent generating features',
    ['feature_type']
)
model_registrations_total = Counter(
    'tars_model_registrations_total',
    'Total model registrations',
    ['agent_type']
)

# Request/Response Models
class OptimizationRequest(BaseModel):
    agent_type: str = Field(..., description="Agent type: dqn, a2c, ppo, ddpg, causal")
    n_trials: int = Field(100, description="Number of optimization trials", ge=1, le=1000)
    timeout_seconds: Optional[int] = Field(None, description="Maximum optimization time in seconds")
    sampler_type: str = Field("tpe", description="Optuna sampler: tpe, cmaes, random")
    pruner_type: str = Field("median", description="Optuna pruner: median, hyperband, none")
    register_model: bool = Field(True, description="Register best model to MLflow")
    use_real_training: bool = Field(True, description="Use real agent training (False = mock objective)")
    use_quick_mode: bool = Field(False, description="Use quick mode with fewer episodes for faster trials")

class FeatureGenerationRequest(BaseModel):
    feature_type: str = Field(..., description="Type: agent, temporal, multiagent, reward")
    data_source: str = Field(..., description="Path or identifier for data source")
    max_depth: int = Field(2, description="Maximum DFS depth", ge=1, le=5)
    max_features: int = Field(100, description="Maximum features to generate", ge=10, le=1000)

class ModelRegistrationRequest(BaseModel):
    run_id: str = Field(..., description="MLflow run ID")
    model_name: str = Field(..., description="Name for registered model")
    description: Optional[str] = Field(None, description="Model description")

class ModelPromotionRequest(BaseModel):
    model_name: str = Field(..., description="Registered model name")
    version: str = Field(..., description="Model version to promote")
    stage: str = Field("Production", description="Target stage: Staging, Production")

# Service components
automl_enabled = os.getenv("AUTOML_ENABLED", "true").lower() == "true"
optuna_n_trials = int(os.getenv("OPTUNA_N_TRIALS", "100"))
featuretools_depth = int(os.getenv("FEATURETOOLS_DEPTH", "2"))
mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "./mlruns")

optimizer: Optional[OptunaOptimizer] = None
feature_engineer: Optional[FeatureEngineer] = None
model_registry: Optional[ModelRegistry] = None

# Background task tracking
active_optimizations: Dict[str, Dict[str, Any]] = {}


@app.on_event("startup")
async def startup_event():
    """Initialize AutoML components on startup."""
    global optimizer, feature_engineer, model_registry

    if not automl_enabled:
        logger.warning("AutoML is disabled. Set AUTOML_ENABLED=true to enable.")
        return

    logger.info("Initializing AutoML components...")

    try:
        # Initialize optimizer
        optimizer = OptunaOptimizer(
            study_name="tars_automl",
            sampler_type="tpe",
            pruner_type="median",
            storage=None,  # Use in-memory storage (or set to DB URL)
        )

        # Initialize feature engineer
        feature_engineer = FeatureEngineer(
            max_depth=featuretools_depth,
            max_features=100,
        )

        # Initialize model registry
        model_registry = ModelRegistry(
            tracking_uri=mlflow_tracking_uri,
            experiment_name="tars_automl",
        )

        logger.info("AutoML components initialized successfully")

    except Exception as e:
        logger.error(f"Failed to initialize AutoML components: {e}")
        raise


# ============================================================================
# Optimization Endpoints
# ============================================================================

@app.post("/api/v1/optimize")
async def optimize_agent(
    request: OptimizationRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Start hyperparameter optimization for an agent.

    Requires: Developer role or higher

    This endpoint launches an asynchronous optimization task and returns immediately.
    Use GET /api/v1/optimize/{optimization_id} to check progress.
    """
    # Check authorization (developer or admin)
    if Role.DEVELOPER not in current_user.roles and Role.ADMIN not in current_user.roles:
        raise HTTPException(
            status_code=403,
            detail="Requires developer role or higher"
        )
    if not automl_enabled or optimizer is None:
        raise HTTPException(status_code=503, detail="AutoML is not enabled")

    optimization_id = f"{request.agent_type}_{int(time.time())}"

    # Create optimization task
    active_optimizations[optimization_id] = {
        "status": "pending",
        "agent_type": request.agent_type,
        "n_trials": request.n_trials,
        "started_at": datetime.utcnow().isoformat(),
        "completed_at": None,
        "result": None,
    }

    # Launch background task
    background_tasks.add_task(
        _run_optimization,
        optimization_id,
        request.agent_type,
        request.n_trials,
        request.timeout_seconds,
        request.sampler_type,
        request.pruner_type,
        request.register_model,
        request.use_real_training,
        request.use_quick_mode,
    )

    logger.info(f"Started optimization task: {optimization_id}")

    return {
        "optimization_id": optimization_id,
        "status": "pending",
        "message": f"Optimization task started for {request.agent_type}",
    }


@app.get("/api/v1/optimize/{optimization_id}")
async def get_optimization_status(
    optimization_id: str,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get status of an optimization task.

    Requires: Viewer role or higher
    """
    # Check authorization (viewer, developer, or admin)
    if Role.VIEWER not in current_user.roles and Role.DEVELOPER not in current_user.roles and Role.ADMIN not in current_user.roles:
        raise HTTPException(
            status_code=403,
            detail="Requires viewer role or higher"
        )

    if optimization_id not in active_optimizations:
        raise HTTPException(status_code=404, detail="Optimization ID not found")

    return active_optimizations[optimization_id]


async def _run_optimization(
    optimization_id: str,
    agent_type: str,
    n_trials: int,
    timeout: Optional[int],
    sampler_type: str,
    pruner_type: str,
    register_model: bool,
    use_real_training: bool,
    use_quick_mode: bool,
):
    """Background task for running optimization."""
    try:
        active_optimizations[optimization_id]["status"] = "running"

        # Create objective function
        if use_real_training:
            # Real training with multi-agent orchestration
            logger.info(f"Using REAL training objective for {agent_type} (quick_mode={use_quick_mode})")
            objective_fn = create_objective(agent_type, use_quick_mode=use_quick_mode)
        else:
            # Mock objective (for testing)
            logger.info(f"Using MOCK objective for {agent_type}")
            def objective_fn(params: Dict[str, Any]) -> float:
                time.sleep(0.1)  # Simulate training time
                return np.random.uniform(0.6, 1.0)

        # Run optimization based on agent type
        if agent_type == "dqn":
            result = optimizer.optimize_dqn(objective_fn, n_trials=n_trials, timeout=timeout)
        elif agent_type == "a2c":
            result = optimizer.optimize_a2c(objective_fn, n_trials=n_trials, timeout=timeout)
        elif agent_type == "ppo":
            result = optimizer.optimize_ppo(objective_fn, n_trials=n_trials, timeout=timeout)
        elif agent_type == "ddpg":
            result = optimizer.optimize_ddpg(objective_fn, n_trials=n_trials, timeout=timeout)
        elif agent_type == "causal":
            result = optimizer.optimize_causal_engine(objective_fn, n_trials=n_trials, timeout=timeout)
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")

        # Update metrics
        optimization_trials_total.labels(agent_type=agent_type, status="completed").inc(result["n_completed_trials"])
        optimization_trials_total.labels(agent_type=agent_type, status="pruned").inc(result["n_pruned_trials"])
        best_score_gauge.labels(agent_type=agent_type).set(result["best_score"])

        # Register to MLflow
        if register_model and model_registry is not None:
            run_id = model_registry.log_optimization_result(agent_type, result)
            result["mlflow_run_id"] = run_id

        # Update status
        active_optimizations[optimization_id]["status"] = "completed"
        active_optimizations[optimization_id]["completed_at"] = datetime.utcnow().isoformat()
        active_optimizations[optimization_id]["result"] = result

        logger.info(f"Optimization completed: {optimization_id}")

    except Exception as e:
        logger.error(f"Optimization failed for {optimization_id}: {e}")
        active_optimizations[optimization_id]["status"] = "failed"
        active_optimizations[optimization_id]["error"] = str(e)
        optimization_trials_total.labels(agent_type=agent_type, status="failed").inc()


# ============================================================================
# Feature Engineering Endpoints
# ============================================================================

@app.post("/api/v1/features")
async def generate_features(request: FeatureGenerationRequest) -> Dict[str, Any]:
    """
    Generate features using Featuretools.

    This endpoint processes data and returns engineered features.
    """
    if not automl_enabled or feature_engineer is None:
        raise HTTPException(status_code=503, detail="AutoML is not enabled")

    logger.info(f"Generating {request.feature_type} features from {request.data_source}")

    start_time = time.time()

    try:
        # Mock data generation (in production, load from request.data_source)
        if request.feature_type == "temporal":
            # Generate temporal features
            df = _generate_mock_timeseries(n_samples=1000)
            result_df = feature_engineer.generate_temporal_features(df)
            feature_names = [col for col in result_df.columns if col not in df.columns]

        elif request.feature_type == "reward":
            # Generate reward features
            df = _generate_mock_rewards(n_samples=1000)
            result_df = feature_engineer.generate_reward_features(df)
            feature_names = [col for col in result_df.columns if col not in df.columns]

        else:
            # Generic response for other types
            feature_names = [f"feature_{i}" for i in range(request.max_features)]
            result_df = pd.DataFrame()

        elapsed_time = time.time() - start_time
        featuregen_time_seconds.labels(feature_type=request.feature_type).observe(elapsed_time)

        logger.info(f"Generated {len(feature_names)} features in {elapsed_time:.2f}s")

        return {
            "feature_type": request.feature_type,
            "n_features": len(feature_names),
            "feature_names": feature_names[:50],  # Return first 50
            "elapsed_time_seconds": elapsed_time,
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(f"Feature generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Model Registry Endpoints
# ============================================================================

@app.post("/api/v1/models/register")
async def register_model(request: ModelRegistrationRequest) -> Dict[str, Any]:
    """Register a model to MLflow Model Registry."""
    if model_registry is None:
        raise HTTPException(status_code=503, detail="Model registry is not enabled")

    try:
        version = model_registry.register_model(
            run_id=request.run_id,
            model_name=request.model_name,
            description=request.description,
        )

        # Extract agent type from model name (convention: agent_type_timestamp)
        agent_type = request.model_name.split("_")[0] if "_" in request.model_name else "unknown"
        model_registrations_total.labels(agent_type=agent_type).inc()

        return {
            "model_name": request.model_name,
            "version": version,
            "run_id": request.run_id,
            "registered_at": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(f"Model registration failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/models/promote")
async def promote_model(request: ModelPromotionRequest) -> Dict[str, Any]:
    """Promote a model version to a stage (Staging/Production)."""
    if model_registry is None:
        raise HTTPException(status_code=503, detail="Model registry is not enabled")

    try:
        model_registry.promote_model(
            model_name=request.model_name,
            version=request.version,
            stage=request.stage,
            archive_existing=True,
        )

        return {
            "model_name": request.model_name,
            "version": request.version,
            "stage": request.stage,
            "promoted_at": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(f"Model promotion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/models")
async def list_models(agent_type: Optional[str] = None) -> Dict[str, Any]:
    """List registered models, optionally filtered by agent type."""
    if model_registry is None:
        raise HTTPException(status_code=503, detail="Model registry is not enabled")

    try:
        # Get best run for each agent type
        if agent_type:
            best_run = model_registry.get_best_run(agent_type=agent_type)
            runs = [best_run] if best_run else []
        else:
            # Get best runs for all agent types
            agent_types = ["dqn", "a2c", "ppo", "ddpg"]
            runs = []
            for at in agent_types:
                best_run = model_registry.get_best_run(agent_type=at)
                if best_run:
                    runs.append(best_run)

        return {
            "models": runs,
            "count": len(runs),
            "filtered_by": agent_type,
        }

    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/models/{model_name}/versions")
async def get_model_versions(model_name: str, stage: Optional[str] = None) -> Dict[str, Any]:
    """Get versions of a registered model."""
    if model_registry is None:
        raise HTTPException(status_code=503, detail="Model registry is not enabled")

    try:
        versions = model_registry.get_model_versions(model_name, stage=stage)

        return {
            "model_name": model_name,
            "versions": versions,
            "count": len(versions),
        }

    except Exception as e:
        logger.error(f"Failed to get model versions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Health & Metrics
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "automl-service",
        "automl_enabled": automl_enabled,
        "components": {
            "optimizer": optimizer is not None,
            "feature_engineer": feature_engineer is not None,
            "model_registry": model_registry is not None,
        },
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/api/v1/stats")
async def get_stats():
    """Get service statistics."""
    stats = {
        "automl_enabled": automl_enabled,
        "active_optimizations": len([o for o in active_optimizations.values() if o["status"] == "running"]),
        "completed_optimizations": len([o for o in active_optimizations.values() if o["status"] == "completed"]),
        "failed_optimizations": len([o for o in active_optimizations.values() if o["status"] == "failed"]),
    }

    if model_registry:
        stats["registry"] = model_registry.get_registry_stats()

    if feature_engineer:
        stats["feature_engineer"] = feature_engineer.get_generation_stats()

    return stats


# ============================================================================
# Helper Functions
# ============================================================================

def _generate_mock_timeseries(n_samples: int = 1000) -> pd.DataFrame:
    """Generate mock timeseries data for testing."""
    timestamps = pd.date_range(start="2025-01-01", periods=n_samples, freq="1min")
    return pd.DataFrame({
        "timestamp": timestamps,
        "reward": np.random.randn(n_samples).cumsum(),
        "loss": np.abs(np.random.randn(n_samples)) * 0.1,
        "value": np.random.uniform(0, 100, n_samples),
    })


def _generate_mock_rewards(n_samples: int = 1000) -> pd.DataFrame:
    """Generate mock reward data for testing."""
    timestamps = pd.date_range(start="2025-01-01", periods=n_samples, freq="1min")
    return pd.DataFrame({
        "timestamp": timestamps,
        "agent_id": np.random.choice(["dqn", "a2c", "ppo", "ddpg"], n_samples),
        "reward": np.random.uniform(0, 1, n_samples),
    })


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    port = int(os.getenv("AUTOML_PORT", "8097"))
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info",
    )
