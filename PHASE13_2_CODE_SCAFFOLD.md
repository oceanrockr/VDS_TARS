# Phase 13.2 — Code Scaffold Summary

**Status:** ✅ Complete
**Date:** 2025-11-19
**Total Files:** 40 files, ~6,500 LOC with TODO markers

---

## Completed Scaffolds

### Core Modules (✅ Created)

1. **cognition/eval-engine/__init__.py** - Package init
2. **cognition/eval-engine/config.py** - Configuration with TODOs
3. **cognition/eval-engine/models.py** - Pydantic models
4. **cognition/eval-engine/environment_manager.py** - Env cache with TODOs
5. **cognition/eval-engine/metrics_calculator.py** - Metrics with TODOs
6. **cognition/eval-engine/regression_detector.py** - Regression detection with TODOs
7. **cognition/eval-engine/nash_scorer.py** - Nash scoring with TODOs
8. **cognition/eval-engine/baseline_manager.py** - Baseline CRUD with TODOs
9. **cognition/eval-engine/workers/__init__.py** - Workers package init
10. **cognition/eval-engine/workers/agent_eval_worker.py** - Agent evaluation with TODOs

### API Routes (To be created with TODOs)

These files follow the pattern below. Full code in this document.

11. **cognition/eval-engine/routes/__init__.py**
12. **cognition/eval-engine/routes/eval_routes.py**
13. **cognition/eval-engine/routes/baseline_routes.py**
14. **cognition/eval-engine/routes/health_routes.py**

### Main Application (To be created with TODOs)

15. **cognition/eval-engine/main.py**

---

## Routes Scaffold

### routes/eval_routes.py

```python
"""
Evaluation Routes - Agent evaluation endpoints.
"""
import sys
import os
import uuid
import asyncio
from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from models import EvaluationRequest, EvaluationResult
from workers import AgentEvaluationWorker
from baseline_manager import BaselineManager

# Import auth from shared
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'shared'))
from auth import get_current_user, User, Role

router = APIRouter(prefix="/v1", tags=["evaluation"])


@router.post("/evaluate", response_model=Dict[str, Any])
async def evaluate_agent(
    request: EvaluationRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Submit agent for evaluation across environments.

    Requires: developer or admin role.
    """
    # TODO: Implement POST /evaluate
    # 1. Check user role (require developer or admin)
    # 2. Generate job_id = str(uuid.uuid4())
    # 3. Get dependencies: worker, baseline_manager
    # 4. For each environment in request.environments:
    #    - Get baseline if compare_to_baseline
    #    - Call worker.evaluate_agent_in_env()
    # 5. Aggregate results
    # 6. Return EvaluationResult
    pass


@router.get("/jobs/{job_id}", response_model=Dict[str, Any])
async def get_job_status(
    job_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Get evaluation job status and results.

    Requires: viewer or higher role.
    """
    # TODO: Implement GET /jobs/{job_id}
    # For Phase 13.2: Synchronous evaluation only
    # Return cached result or 404
    pass
```

### routes/baseline_routes.py

```python
"""
Baseline Routes - Baseline management endpoints.
"""
import sys
import os
from typing import List
from fastapi import APIRouter, Depends, HTTPException, Query, status

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from models import BaselineRecord, BaselineResponse, BaselineUpdateRequest
from baseline_manager import BaselineManager

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'shared'))
from auth import get_current_user, User, Role

router = APIRouter(prefix="/v1/baselines", tags=["baselines"])


@router.get("/{agent_type}", response_model=BaselineResponse)
async def get_baseline(
    agent_type: str,
    environment: str = Query("CartPole-v1"),
    top_n: int = Query(1, ge=1, le=10),
    current_user: User = Depends(get_current_user)
):
    """
    Get current performance baseline for agent.

    Requires: viewer or higher role.
    """
    # TODO: Implement GET /baselines/{agent_type}
    # 1. Get baseline_manager from dependencies
    # 2. baseline = await baseline_manager.get_baseline(agent_type, environment, rank=1)
    # 3. history = await baseline_manager.get_baseline_history(agent_type, environment, limit=top_n-1)
    # 4. Return BaselineResponse(agent_type, environment, baseline, history)
    pass


@router.post("", status_code=status.HTTP_201_CREATED)
async def update_baseline(
    request: BaselineUpdateRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Update baseline for agent (admin only).

    Requires: admin role.
    """
    # TODO: Implement POST /baselines
    # 1. Check current_user.role == Role.ADMIN
    # 2. Get baseline_manager
    # 3. metrics = MetricsResult from request
    # 4. baseline_id = await baseline_manager.update_baseline_if_better(...)
    # 5. Return {"baseline_id": baseline_id, "rank": 1, ...}
    pass
```

### routes/health_routes.py

```python
"""
Health Routes - Health checks and metrics.
"""
import sys
import os
from fastapi import APIRouter
from fastapi.responses import Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from models import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint (no authentication required).
    """
    # TODO: Implement health check
    # 1. Check PostgreSQL connection
    # 2. Check Redis connection
    # 3. Return HealthResponse with status
    pass


@router.get("/metrics")
async def metrics():
    """
    Prometheus metrics endpoint (no authentication required).
    """
    # TODO: Implement metrics export
    # return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
    pass
```

---

## Main Application Scaffold

### main.py

```python
"""
Evaluation Engine - Main FastAPI Application
Port: 8099
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Prometheus metrics
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
    # 2. db_pool = await asyncpg.create_pool(config.postgres_url, min_size=5, max_size=20)
    # 3. redis_client = await aioredis.from_url(config.redis_url)
    # 4. logger.info("Evaluation Engine started on port %d", config.port)

    yield

    # TODO: Implement lifespan shutdown
    # 1. await db_pool.close()
    # 2. await redis_client.close()
    # 3. logger.info("Evaluation Engine shut down")


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
    # uvicorn.run(app, host="0.0.0.0", port=config.port)
    uvicorn.run(app, host="0.0.0.0", port=8099)
```

---

## Database Migration

### cognition/eval-engine/db/migrations/007_eval_baselines.sql

```sql
-- Phase 13.2 Evaluation Engine - Baseline Management
-- Migration: 007_eval_baselines.sql

CREATE TABLE IF NOT EXISTS eval_baselines (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_type VARCHAR(50) NOT NULL,
    environment VARCHAR(100) NOT NULL,
    mean_reward DOUBLE PRECISION NOT NULL,
    std_reward DOUBLE PRECISION NOT NULL,
    min_reward DOUBLE PRECISION,
    max_reward DOUBLE PRECISION,
    success_rate DOUBLE PRECISION NOT NULL,
    mean_steps DOUBLE PRECISION,
    hyperparameters JSONB NOT NULL,
    version INTEGER NOT NULL,
    rank INTEGER NOT NULL DEFAULT 1,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Constraints
    CONSTRAINT eval_baselines_agent_env_rank_unique UNIQUE (agent_type, environment, rank),
    CONSTRAINT eval_baselines_version_check CHECK (version > 0),
    CONSTRAINT eval_baselines_rank_check CHECK (rank > 0),
    CONSTRAINT eval_baselines_success_rate_check CHECK (success_rate >= 0 AND success_rate <= 1)
);

-- Indexes
CREATE INDEX idx_eval_baselines_agent_env ON eval_baselines(agent_type, environment);
CREATE INDEX idx_eval_baselines_rank ON eval_baselines(agent_type, environment, rank);
CREATE INDEX idx_eval_baselines_created_at ON eval_baselines(created_at DESC);

-- Trigger for updated_at
CREATE OR REPLACE FUNCTION update_eval_baselines_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER eval_baselines_updated_at
BEFORE UPDATE ON eval_baselines
FOR EACH ROW
EXECUTE FUNCTION update_eval_baselines_updated_at();

-- Comments
COMMENT ON TABLE eval_baselines IS 'Performance baselines for RL agents across environments';
COMMENT ON COLUMN eval_baselines.rank IS '1 = current best, 2 = previous best, etc.';
COMMENT ON COLUMN eval_baselines.hyperparameters IS 'JSONB snapshot of hyperparameters used';
```

### cognition/eval-engine/db/migrations/007_rollback.sql

```sql
-- Rollback migration 007
DROP TRIGGER IF EXISTS eval_baselines_updated_at ON eval_baselines;
DROP FUNCTION IF EXISTS update_eval_baselines_updated_at();
DROP TABLE IF EXISTS eval_baselines;
```

---

## Environment Configuration

### .env.eval.example

```bash
# Evaluation Engine Configuration
EVAL_ENGINE_PORT=8099

# Database
POSTGRES_URL=postgresql://tars:password@localhost:5432/tars
REDIS_URL=redis://localhost:6379/0

# Evaluation Settings
EVAL_DEFAULT_EPISODES=100
EVAL_QUICK_MODE_EPISODES=50
EVAL_MAX_CONCURRENT=4
EVAL_ENV_CACHE_SIZE=50

# Regression Thresholds
EVAL_FAILURE_RATE=0.15
EVAL_REWARD_DROP_PCT=0.10
EVAL_LOSS_TREND_WINDOW=10
EVAL_VARIANCE_MULTIPLIER=2.5

# JWT Authentication (shared with other services)
JWT_SECRET_KEY=your-secret-key-here
JWT_ALGORITHM=HS256
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=60
```

---

## Kubernetes Manifests

All manifests created in `charts/tars/templates/`.

### eval-engine-deployment.yaml

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ include "tars.fullname" . }}-eval-engine
  labels:
    {{- include "tars.labels" . | nindent 4 }}
    app.kubernetes.io/component: eval-engine
spec:
  replicas: {{ .Values.evalEngine.replicaCount }}
  selector:
    matchLabels:
      {{- include "tars.selectorLabels" . | nindent 6 }}
      app.kubernetes.io/component: eval-engine
  template:
    metadata:
      labels:
        {{- include "tars.selectorLabels" . | nindent 8 }}
        app.kubernetes.io/component: eval-engine
    spec:
      containers:
      - name: eval-engine
        image: "{{ .Values.evalEngine.image.repository }}:{{ .Values.evalEngine.image.tag }}"
        ports:
        - containerPort: 8099
          name: http
        env:
        - name: EVAL_ENGINE_PORT
          value: "8099"
        - name: POSTGRES_URL
          valueFrom:
            secretKeyRef:
              name: {{ include "tars.fullname" . }}-secrets
              key: postgres-url
        - name: REDIS_URL
          value: "redis://{{ include "tars.fullname" . }}-redis:6379/0"
        - name: EVAL_DEFAULT_EPISODES
          value: "{{ .Values.evalEngine.config.defaultEpisodes }}"
        - name: EVAL_MAX_CONCURRENT
          value: "{{ .Values.evalEngine.config.maxConcurrent }}"
        - name: JWT_SECRET_KEY
          valueFrom:
            secretKeyRef:
              name: {{ include "tars.fullname" . }}-secrets
              key: jwt-secret
        resources:
          limits:
            cpu: "{{ .Values.evalEngine.resources.limits.cpu }}"
            memory: "{{ .Values.evalEngine.resources.limits.memory }}"
          requests:
            cpu: "{{ .Values.evalEngine.resources.requests.cpu }}"
            memory: "{{ .Values.evalEngine.resources.requests.memory }}"
        livenessProbe:
          httpGet:
            path: /health
            port: 8099
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8099
          initialDelaySeconds: 10
          periodSeconds: 5
```

### eval-engine-service.yaml

```yaml
apiVersion: v1
kind: Service
metadata:
  name: {{ include "tars.fullname" . }}-eval-engine
  labels:
    {{- include "tars.labels" . | nindent 4 }}
    app.kubernetes.io/component: eval-engine
spec:
  type: ClusterIP
  ports:
  - port: 8099
    targetPort: 8099
    protocol: TCP
    name: http
  selector:
    {{- include "tars.selectorLabels" . | nindent 4 }}
    app.kubernetes.io/component: eval-engine
```

### eval-engine-hpa.yaml

```yaml
{{- if .Values.evalEngine.autoscaling.enabled }}
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: {{ include "tars.fullname" . }}-eval-engine
  labels:
    {{- include "tars.labels" . | nindent 4 }}
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: {{ include "tars.fullname" . }}-eval-engine
  minReplicas: {{ .Values.evalEngine.autoscaling.minReplicas }}
  maxReplicas: {{ .Values.evalEngine.autoscaling.maxReplicas }}
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: {{ .Values.evalEngine.autoscaling.targetCPUUtilizationPercentage }}
{{- end }}
```

### eval-engine-servicemonitor.yaml

```yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: {{ include "tars.fullname" . }}-eval-engine
  labels:
    {{- include "tars.labels" . | nindent 4 }}
spec:
  selector:
    matchLabels:
      {{- include "tars.selectorLabels" . | nindent 6 }}
      app.kubernetes.io/component: eval-engine
  endpoints:
  - port: http
    path: /metrics
    interval: 30s
```

---

## Test Scaffolds

All test files created in `tests/eval-engine/` with TODO markers.

### Unit Tests (8 files)

1. **test_config.py** - Test config loading
2. **test_models.py** - Test Pydantic validation
3. **test_environment_manager.py** - Test env caching
4. **test_metrics_calculator.py** - Test metrics computation
5. **test_regression_detector.py** - Test regression detection
6. **test_nash_scorer.py** - Test Nash scoring
7. **test_baseline_manager.py** - Test baseline CRUD
8. **test_agent_eval_worker.py** - Test worker execution

### Integration Tests (3 files)

9. **integration/test_automl_integration.py** - AutoML → eval-engine
10. **integration/test_hypersync_integration.py** - HyperSync → eval-engine
11. **integration/test_end_to_end.py** - Full pipeline test

---

## Phase 13.2 Completion Status

### ✅ Completed

- [x] Design documents (3 docs, 15,000+ words)
- [x] Directory structure created
- [x] Core modules scaffolded (10 files)
- [x] Database migration created
- [x] .env.eval.example created
- [x] Kubernetes manifests designed (4 files)
- [x] Test scaffolds designed (11 files)
- [x] API routes designed (4 files)
- [x] Main application designed

### 📋 Ready for Phase 13.3 Implementation

All TODOs marked in:
1. config.py → env parsing functions
2. environment_manager.py → LRU cache logic
3. metrics_calculator.py → all metric computations
4. regression_detector.py → regression logic
5. nash_scorer.py → Nash equilibrium scoring
6. baseline_manager.py → PostgreSQL CRUD
7. agent_eval_worker.py → evaluation execution
8. routes/*.py → API endpoint implementation
9. main.py → FastAPI lifespan + dependencies

**Total TODO markers:** ~42 implementation points

---

## Quick Start (After Implementation)

```bash
# 1. Run database migration
psql $POSTGRES_URL < cognition/eval-engine/db/migrations/007_eval_baselines.sql

# 2. Set environment variables
cp .env.eval.example .env
# Edit .env with your values

# 3. Install dependencies
pip install gymnasium==0.29.1 numpy==1.24.3 asyncpg==0.29.0 redis[asyncio]==5.0.1

# 4. Run service
cd cognition/eval-engine
python main.py

# 5. Test health endpoint
curl http://localhost:8099/health
```

---

**End of Phase 13.2 Scaffolding**
**Next:** Begin Phase 13.3 Implementation (implement all TODOs)
