# Phase 13.2 — Evaluation Engine Design Document

**Version:** 1.0.0
**Status:** Design Complete
**Author:** T.A.R.S. Architecture Team
**Date:** 2025-11-19
**Target Release:** v1.0.0-RC2
**Prerequisites:** Phase 13.1 Unified Pipeline Design

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Architecture](#2-system-architecture)
3. [Component Specifications](#3-component-specifications)
4. [Data Models](#4-data-models)
5. [API Endpoints](#5-api-endpoints)
6. [Worker Architecture](#6-worker-architecture)
7. [Metrics & Regression Detection](#7-metrics--regression-detection)
8. [Nash Equilibrium Integration](#8-nash-equilibrium-integration)
9. [Baseline Management](#9-baseline-management)
10. [Environment Management](#10-environment-management)
11. [Database Schema](#11-database-schema)
12. [Deployment Architecture](#12-deployment-architecture)
13. [Performance Requirements](#13-performance-requirements)
14. [Security Considerations](#14-security-considerations)

---

## 1. Executive Summary

### 1.1 Purpose

The **Evaluation Engine** is the central component that replaces mock training with **real RL agent evaluation** in the T.A.R.S. pipeline. It provides:

- **Multi-environment agent evaluation** (CartPole, LunarLander, MountainCar)
- **Comprehensive metrics calculation** (reward, loss, stability, entropy)
- **Regression detection** with automatic rollback triggers
- **Baseline management** for performance tracking
- **Nash equilibrium scoring** for multi-agent coordination
- **Production-grade evaluation API** with RBAC and rate limiting

### 1.2 Key Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Evaluation Latency** | <30s (quick mode) | 50 episodes, single env |
| **Full Evaluation** | <120s | 100 episodes, 3 envs |
| **Regression Detection** | <100ms | Real-time during evaluation |
| **API Response Time** | <50ms | GET/POST endpoints |
| **Baseline Lookup** | <10ms | PostgreSQL query |
| **Nash Scoring** | <200ms | 4-agent payoff matrix |
| **Environment Cache Hit Rate** | >90% | LRU cache with 50 env limit |

### 1.3 Integration Points

```
┌─────────────────────────────────────────────────────────────┐
│                     Evaluation Engine (Port 8099)           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   │
│  │ REST API     │   │ Worker Pool  │   │ Metrics      │   │
│  │ (FastAPI)    │───│ (asyncio)    │───│ Calculator   │   │
│  └──────┬───────┘   └──────┬───────┘   └──────────────┘   │
│         │                  │                               │
│         │                  ▼                               │
│         │         ┌──────────────┐                         │
│         │         │ Environment  │                         │
│         │         │ Manager      │                         │
│         │         │ (LRU Cache)  │                         │
│         │         └──────────────┘                         │
│         │                                                   │
│         ▼                                                   │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   │
│  │ Regression   │   │ Nash Scorer  │   │ Baseline     │   │
│  │ Detector     │   │              │   │ Manager      │   │
│  └──────────────┘   └──────────────┘   └──────────────┘   │
│         │                  │                  │            │
└─────────┼──────────────────┼──────────────────┼────────────┘
          │                  │                  │
          ▼                  ▼                  ▼
    ┌──────────┐       ┌──────────┐      ┌──────────┐
    │ AutoML   │       │ HyperSync│      │ Postgres │
    │ (8096)   │       │ (8098)   │      │ (5432)   │
    └──────────┘       └──────────┘      └──────────┘
```

---

## 2. System Architecture

### 2.1 Service Overview

**Port:** 8099
**Framework:** FastAPI + asyncio
**Language:** Python 3.11+
**Dependencies:**
- `gymnasium` - RL environments
- `numpy` - Numerical computations
- `asyncpg` - PostgreSQL async driver
- `redis.asyncio` - Redis async client
- `fastapi` - REST API framework
- `prometheus_client` - Metrics export

### 2.2 Component Breakdown

#### 2.2.1 Core Modules (cognition/eval-engine/)

```
eval-engine/
├── __init__.py
├── main.py                    # FastAPI app + lifespan management
├── config.py                  # Environment-based configuration
├── models.py                  # Pydantic request/response models
├── routes/
│   ├── __init__.py
│   ├── eval_routes.py         # POST /v1/evaluate, GET /v1/jobs/{id}
│   ├── baseline_routes.py     # GET/POST /v1/baselines
│   └── health_routes.py       # GET /health, /metrics
├── workers/
│   ├── __init__.py
│   └── agent_eval_worker.py   # Core evaluation executor
├── environment_manager.py     # Gym environment cache + wrappers
├── metrics_calculator.py      # Reward, loss, stability, entropy
├── regression_detector.py     # Performance regression detection
├── nash_scorer.py             # Nash equilibrium scoring
├── baseline_manager.py        # Baseline CRUD operations
└── db/
    └── migrations/
        └── 007_eval_baselines.sql
```

#### 2.2.2 Worker Architecture

```python
# Agent Evaluation Flow
┌─────────────────┐
│ POST /evaluate  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Enqueue Job     │ (Redis Stream or in-memory)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Worker Pool     │ (asyncio.gather, concurrency=4)
│ - Get Env       │
│ - Load Agent    │
│ - Run Episodes  │
│ - Calc Metrics  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Regression Check│
│ - Compare       │
│ - Detect Drop   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Nash Scoring    │ (if multi-agent mode)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Return Results  │
└─────────────────┘
```

---

## 3. Component Specifications

### 3.1 REST API Layer (main.py)

**Responsibilities:**
- Serve FastAPI application
- JWT authentication integration
- Rate limiting middleware
- Prometheus metrics export
- Health checks
- Lifespan management (DB pool init/cleanup)

**Endpoints:**
- `POST /v1/evaluate` - Submit evaluation job
- `GET /v1/jobs/{job_id}` - Get job status/results
- `GET /v1/baselines/{agent_type}` - Get current baseline
- `POST /v1/baselines` - Update baseline
- `GET /health` - Health check
- `GET /metrics` - Prometheus metrics

### 3.2 Configuration Manager (config.py)

**Responsibilities:**
- Load environment variables
- Validate configuration
- Provide typed config objects
- Default values for regression thresholds

**Key Classes:**
```python
@dataclass
class RegressionThresholds:
    failure_rate: float = 0.15          # Max 15% episode failures
    reward_drop_pct: float = 0.10       # Max 10% reward drop
    loss_trend_window: int = 10         # Check last 10 episodes
    variance_multiplier: float = 2.5    # Max 2.5x variance increase

    @classmethod
    def from_env(cls) -> "RegressionThresholds": ...

@dataclass
class EvalEngineConfig:
    port: int
    postgres_url: str
    redis_url: str
    default_episodes: int = 100
    quick_mode_episodes: int = 50
    max_concurrent_evals: int = 4
    env_cache_size: int = 50
    thresholds: RegressionThresholds

    @classmethod
    def from_env(cls) -> "EvalEngineConfig": ...
```

**Environment Variables:**
```bash
EVAL_ENGINE_PORT=8099
POSTGRES_URL=postgresql://user:pass@localhost:5432/tars
REDIS_URL=redis://localhost:6379/0
EVAL_DEFAULT_EPISODES=100
EVAL_QUICK_MODE_EPISODES=50
EVAL_MAX_CONCURRENT=4
EVAL_ENV_CACHE_SIZE=50

# Regression Thresholds
EVAL_FAILURE_RATE=0.15
EVAL_REWARD_DROP_PCT=0.10
EVAL_LOSS_TREND_WINDOW=10
EVAL_VARIANCE_MULTIPLIER=2.5
```

### 3.3 Data Models (models.py)

**Pydantic Models:**
```python
class EvaluationRequest(BaseModel):
    agent_type: Literal["DQN", "A2C", "PPO", "DDPG"]
    agent_state: Dict[str, Any]  # Serialized agent weights
    hyperparameters: Dict[str, float]
    environments: List[str] = ["CartPole-v1"]
    num_episodes: int = 100
    quick_mode: bool = False
    compare_to_baseline: bool = True

class EpisodeResult(BaseModel):
    episode_num: int
    total_reward: float
    steps: int
    success: bool
    loss: Optional[float]

class MetricsResult(BaseModel):
    mean_reward: float
    std_reward: float
    min_reward: float
    max_reward: float
    success_rate: float
    mean_steps: float
    mean_loss: Optional[float]
    loss_trend: Optional[str]  # "increasing" | "decreasing" | "stable"
    action_entropy: float
    reward_variance: float

class RegressionResult(BaseModel):
    detected: bool
    regression_score: float  # 0.0-1.0
    reason: Optional[str]
    should_rollback: bool
    failed_checks: List[str]

class EvaluationResult(BaseModel):
    job_id: str
    agent_type: str
    environment: str
    metrics: MetricsResult
    regression: RegressionResult
    episodes: List[EpisodeResult]
    baseline_comparison: Optional[Dict[str, float]]
    nash_score: Optional[float]
    duration_seconds: float
    completed_at: str
```

---

## 4. Data Models

### 4.1 Request/Response Schemas

See [Section 3.3](#33-data-models-modelspy) for complete Pydantic models.

### 4.2 Internal Data Structures

**Baseline Record:**
```python
@dataclass
class Baseline:
    agent_type: str
    environment: str
    mean_reward: float
    std_reward: float
    success_rate: float
    hyperparameters: Dict[str, Any]
    created_at: datetime
    version: int
    rank: int  # 1 = current best, 2 = previous, etc.
```

**Environment Cache Entry:**
```python
@dataclass
class CachedEnvironment:
    env_id: str
    env: gym.Env
    last_accessed: datetime
    access_count: int
```

---

## 5. API Endpoints

### 5.1 POST /v1/evaluate

**Purpose:** Submit agent for evaluation across environments.

**Request:**
```json
{
  "agent_type": "DQN",
  "agent_state": {
    "network_weights": "base64_encoded_state_dict",
    "version": 13
  },
  "hyperparameters": {
    "learning_rate": 0.001,
    "gamma": 0.99,
    "epsilon": 0.1
  },
  "environments": ["CartPole-v1", "LunarLander-v2"],
  "num_episodes": 100,
  "quick_mode": false,
  "compare_to_baseline": true
}
```

**Response (202 Accepted):**
```json
{
  "job_id": "eval-550e8400-e29b-41d4-a716-446655440000",
  "status": "running",
  "submitted_at": "2025-11-19T10:30:00Z",
  "estimated_duration_sec": 120
}
```

**Response (200 OK - synchronous mode):**
```json
{
  "job_id": "eval-550e8400...",
  "agent_type": "DQN",
  "results": [
    {
      "environment": "CartPole-v1",
      "metrics": {
        "mean_reward": 485.2,
        "std_reward": 18.3,
        "success_rate": 0.98,
        "action_entropy": 0.65
      },
      "regression": {
        "detected": false,
        "regression_score": 0.02,
        "should_rollback": false
      },
      "baseline_comparison": {
        "reward_improvement": 0.08,
        "vs_baseline_mean": 450.0
      },
      "duration_seconds": 58.2
    }
  ],
  "nash_score": 0.92,
  "completed_at": "2025-11-19T10:32:05Z"
}
```

**Authentication:** JWT required (Role: viewer+)
**Rate Limit:** 10 requests/minute (developer), 5/min (viewer)

### 5.2 GET /v1/jobs/{job_id}

**Purpose:** Retrieve evaluation job status and results.

**Response (200 OK):**
```json
{
  "job_id": "eval-550e8400...",
  "status": "completed",  // queued | running | completed | failed
  "submitted_at": "2025-11-19T10:30:00Z",
  "completed_at": "2025-11-19T10:32:05Z",
  "results": { /* Same as POST /evaluate response */ }
}
```

**Authentication:** JWT required (Role: viewer+)
**Rate Limit:** 30 requests/minute

### 5.3 GET /v1/baselines/{agent_type}

**Purpose:** Get current performance baseline for agent.

**Query Parameters:**
- `environment` (optional): Filter by environment
- `top_n` (default: 1): Return top N baselines

**Response (200 OK):**
```json
{
  "agent_type": "DQN",
  "environment": "CartPole-v1",
  "baseline": {
    "mean_reward": 450.0,
    "std_reward": 22.5,
    "success_rate": 0.95,
    "version": 12,
    "created_at": "2025-11-18T14:30:00Z"
  },
  "history": [
    {
      "version": 11,
      "mean_reward": 425.0,
      "created_at": "2025-11-17T10:00:00Z"
    }
  ]
}
```

**Authentication:** JWT required (Role: viewer+)
**Rate Limit:** 30 requests/minute

### 5.4 POST /v1/baselines

**Purpose:** Update baseline for agent (admin only).

**Request:**
```json
{
  "agent_type": "DQN",
  "environment": "CartPole-v1",
  "mean_reward": 485.2,
  "std_reward": 18.3,
  "success_rate": 0.98,
  "hyperparameters": {...},
  "version": 13
}
```

**Response (201 Created):**
```json
{
  "baseline_id": "baseline-uuid",
  "rank": 1,
  "replaced_version": 12,
  "improvement_pct": 7.8
}
```

**Authentication:** JWT required (Role: admin)
**Rate Limit:** 10 requests/minute

---

## 6. Worker Architecture

### 6.1 Agent Evaluation Worker (agent_eval_worker.py)

**Purpose:** Execute agent evaluation in isolated async tasks.

**Key Methods:**
```python
class AgentEvaluationWorker:
    def __init__(
        self,
        env_manager: EnvironmentManager,
        metrics_calc: MetricsCalculator,
        regression_detector: RegressionDetector
    ): ...

    async def evaluate_agent_in_env(
        self,
        agent_type: str,
        agent_state: Dict[str, Any],
        hyperparameters: Dict[str, float],
        environment: str,
        num_episodes: int,
        baseline: Optional[Baseline]
    ) -> Dict[str, Any]:
        """
        Execute full agent evaluation.

        Steps:
        1. Load agent model from state dict
        2. Get/create environment from cache
        3. Run N episodes, collect results
        4. Calculate metrics
        5. Detect regression vs baseline
        6. Return comprehensive results

        Returns:
            {
                "environment": str,
                "episodes": List[EpisodeResult],
                "metrics": MetricsResult,
                "regression": RegressionResult,
                "duration_seconds": float
            }
        """
        ...

    def _load_agent_model(
        self,
        agent_type: str,
        state_dict: Dict[str, Any],
        hyperparameters: Dict[str, float]
    ) -> Any:
        """Load agent (DQN, A2C, PPO, DDPG) from state dict."""
        ...

    async def _run_episode(
        self,
        agent: Any,
        env: gym.Env,
        episode_num: int
    ) -> EpisodeResult:
        """Run single episode, return reward/steps/loss."""
        ...
```

**Concurrency:**
- Use `asyncio.gather()` to run multiple environment evals in parallel
- Max concurrency controlled by `EVAL_MAX_CONCURRENT` (default: 4)
- Timeout enforcement: 5 minutes per environment eval

### 6.2 Worker Pool Management

```python
class EvaluationService:
    def __init__(self, config: EvalEngineConfig):
        self.worker = AgentEvaluationWorker(...)
        self.semaphore = asyncio.Semaphore(config.max_concurrent_evals)

    async def submit_evaluation(
        self,
        request: EvaluationRequest
    ) -> EvaluationResult:
        async with self.semaphore:
            # Run evaluations for all environments
            tasks = [
                self.worker.evaluate_agent_in_env(
                    request.agent_type,
                    request.agent_state,
                    request.hyperparameters,
                    env,
                    request.num_episodes,
                    baseline=await self._get_baseline(request.agent_type, env)
                )
                for env in request.environments
            ]
            results = await asyncio.gather(*tasks)
            return self._aggregate_results(results)
```

---

## 7. Metrics & Regression Detection

### 7.1 Metrics Calculator (metrics_calculator.py)

**Responsibilities:**
- Compute reward statistics (mean, std, min, max)
- Calculate success rate
- Compute loss trend (increasing/decreasing/stable)
- Calculate action entropy
- Compute reward variance

**Key Methods:**
```python
class MetricsCalculator:
    def compute_reward_metrics(
        self,
        episodes: List[EpisodeResult]
    ) -> Dict[str, float]:
        """
        Returns:
            {
                "mean_reward": float,
                "std_reward": float,
                "min_reward": float,
                "max_reward": float,
                "success_rate": float,
                "mean_steps": float
            }
        """
        rewards = [ep.total_reward for ep in episodes]
        return {
            "mean_reward": np.mean(rewards),
            "std_reward": np.std(rewards),
            "min_reward": np.min(rewards),
            "max_reward": np.max(rewards),
            "success_rate": sum(ep.success for ep in episodes) / len(episodes),
            "mean_steps": np.mean([ep.steps for ep in episodes])
        }

    def compute_loss_metrics(
        self,
        episodes: List[EpisodeResult]
    ) -> Dict[str, Any]:
        """
        Returns:
            {
                "mean_loss": float,
                "loss_trend": "increasing" | "decreasing" | "stable"
            }
        """
        losses = [ep.loss for ep in episodes if ep.loss is not None]
        if not losses:
            return {"mean_loss": None, "loss_trend": None}

        trend = self.detect_loss_trend(losses)
        return {
            "mean_loss": np.mean(losses),
            "loss_trend": trend
        }

    def detect_loss_trend(
        self,
        losses: List[float],
        window: int = 10
    ) -> str:
        """
        Detect if loss is increasing/decreasing/stable using linear regression.
        """
        if len(losses) < window:
            return "stable"

        recent = losses[-window:]
        x = np.arange(len(recent))
        slope, _ = np.polyfit(x, recent, 1)

        if slope > 0.05:
            return "increasing"
        elif slope < -0.05:
            return "decreasing"
        else:
            return "stable"

    def compute_stability_metrics(
        self,
        episodes: List[EpisodeResult]
    ) -> Dict[str, float]:
        """
        Compute reward variance and coefficient of variation.
        """
        rewards = [ep.total_reward for ep in episodes]
        mean_reward = np.mean(rewards)
        variance = np.var(rewards)
        cv = np.std(rewards) / mean_reward if mean_reward != 0 else 0

        return {
            "reward_variance": variance,
            "coefficient_of_variation": cv
        }

    def compute_action_entropy(
        self,
        action_distribution: np.ndarray
    ) -> float:
        """
        Compute Shannon entropy of action distribution.
        """
        probs = action_distribution / action_distribution.sum()
        probs = probs[probs > 0]  # Remove zeros
        return -np.sum(probs * np.log(probs))
```

### 7.2 Regression Detector (regression_detector.py)

**Responsibilities:**
- Compare current metrics to baseline
- Detect performance regressions
- Generate rollback recommendations
- Calculate regression score (0.0-1.0)

**Key Methods:**
```python
class RegressionDetector:
    def __init__(self, thresholds: RegressionThresholds):
        self.thresholds = thresholds

    def should_trigger_rollback(
        self,
        current_metrics: MetricsResult,
        baseline: Baseline
    ) -> RegressionResult:
        """
        Check if current performance represents a regression.

        Regression Criteria:
        1. Failure rate > threshold (default 15%)
        2. Reward drop > threshold (default 10%)
        3. Loss trend is "increasing" for >10 episodes
        4. Variance increased by >2.5x

        Returns:
            RegressionResult with detected flag and reason
        """
        failed_checks = []

        # Check 1: Failure rate
        if current_metrics.success_rate < (1 - self.thresholds.failure_rate):
            failed_checks.append(
                f"Failure rate {1 - current_metrics.success_rate:.1%} "
                f"exceeds threshold {self.thresholds.failure_rate:.1%}"
            )

        # Check 2: Reward drop
        reward_drop = (baseline.mean_reward - current_metrics.mean_reward) / baseline.mean_reward
        if reward_drop > self.thresholds.reward_drop_pct:
            failed_checks.append(
                f"Reward dropped {reward_drop:.1%} "
                f"(threshold: {self.thresholds.reward_drop_pct:.1%})"
            )

        # Check 3: Loss trend
        if current_metrics.loss_trend == "increasing":
            failed_checks.append("Loss trend is increasing")

        # Check 4: Variance increase
        variance_ratio = current_metrics.reward_variance / baseline.std_reward**2
        if variance_ratio > self.thresholds.variance_multiplier:
            failed_checks.append(
                f"Variance increased {variance_ratio:.1f}x "
                f"(threshold: {self.thresholds.variance_multiplier:.1f}x)"
            )

        detected = len(failed_checks) > 0
        score = self.compute_regression_score(current_metrics, baseline)

        return RegressionResult(
            detected=detected,
            regression_score=score,
            reason=self.generate_rollback_reason(failed_checks),
            should_rollback=detected,
            failed_checks=failed_checks
        )

    def compute_regression_score(
        self,
        current: MetricsResult,
        baseline: Baseline
    ) -> float:
        """
        Compute regression score 0.0-1.0 (1.0 = severe regression).

        Formula:
            score = weighted_average([
                reward_drop_ratio,
                failure_rate_ratio,
                variance_ratio,
                loss_trend_penalty
            ])
        """
        # TODO: Implement weighted scoring
        ...

    def generate_rollback_reason(
        self,
        failed_checks: List[str]
    ) -> Optional[str]:
        """Generate human-readable rollback reason."""
        if not failed_checks:
            return None

        return f"Performance regression detected: {'; '.join(failed_checks)}"
```

---

## 8. Nash Equilibrium Integration

### 8.1 Nash Scorer (nash_scorer.py)

**Responsibilities:**
- Compute Nash equilibrium scores for multi-agent evaluation
- Build payoff matrix from agent rewards
- Integrate with existing `NashEquilibriumSolver` from Phase 11
- Detect agent conflicts

**Key Methods:**
```python
class NashScorer:
    def __init__(self, nash_solver: NashEquilibriumSolver):
        self.solver = nash_solver

    async def compute_agent_nash_scores(
        self,
        agent_results: Dict[str, MetricsResult]
    ) -> Dict[str, float]:
        """
        Compute Nash scores for each agent.

        Args:
            agent_results: Dict mapping agent_type -> MetricsResult

        Returns:
            Dict mapping agent_type -> nash_score (0.0-1.0)
        """
        payoff_matrix = self.build_payoff_matrix(agent_results)
        equilibrium = self.solver.solve(payoff_matrix)

        return {
            agent: equilibrium.scores[i]
            for i, agent in enumerate(agent_results.keys())
        }

    def build_payoff_matrix(
        self,
        agent_results: Dict[str, MetricsResult]
    ) -> np.ndarray:
        """
        Build N×N payoff matrix from agent rewards.

        Matrix[i][j] = reward of agent i when interacting with agent j
        For single-agent envs, use mean_reward on diagonal.
        """
        agents = list(agent_results.keys())
        n = len(agents)
        matrix = np.zeros((n, n))

        for i, agent_i in enumerate(agents):
            for j, agent_j in enumerate(agents):
                if i == j:
                    matrix[i][j] = agent_results[agent_i].mean_reward
                else:
                    # For multi-agent envs, use actual interaction rewards
                    # For single-agent, use normalized rewards
                    matrix[i][j] = self.normalize_rewards(
                        agent_results[agent_i].mean_reward,
                        agent_results[agent_j].mean_reward
                    )

        return matrix

    def normalize_rewards(
        self,
        reward_i: float,
        reward_j: float
    ) -> float:
        """Normalize rewards for payoff matrix."""
        return (reward_i + reward_j) / 2.0

    def compute_conflict_score(
        self,
        agent_results: Dict[str, MetricsResult]
    ) -> float:
        """
        Compute conflict score (0.0-1.0) based on reward variance.
        High conflict = high variance across agents.
        """
        rewards = [result.mean_reward for result in agent_results.values()]
        return np.std(rewards) / np.mean(rewards) if np.mean(rewards) != 0 else 0.0
```

---

## 9. Baseline Management

### 9.1 Baseline Manager (baseline_manager.py)

**Responsibilities:**
- CRUD operations for performance baselines
- PostgreSQL integration
- Baseline ranking (top N performers)
- Historical baseline tracking

**Key Methods:**
```python
class BaselineManager:
    def __init__(self, db_pool: asyncpg.Pool):
        self.db = db_pool

    async def get_baseline(
        self,
        agent_type: str,
        environment: str,
        rank: int = 1
    ) -> Optional[Baseline]:
        """
        Get baseline for agent+environment.

        Args:
            rank: 1 = best, 2 = second best, etc.
        """
        query = """
            SELECT * FROM eval_baselines
            WHERE agent_type = $1 AND environment = $2 AND rank = $3
        """
        row = await self.db.fetchrow(query, agent_type, environment, rank)
        return Baseline.from_db_row(row) if row else None

    async def insert_baseline(
        self,
        baseline: Baseline
    ) -> str:
        """Insert new baseline, return ID."""
        query = """
            INSERT INTO eval_baselines (
                agent_type, environment, mean_reward, std_reward,
                success_rate, hyperparameters, version, rank
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            RETURNING id
        """
        baseline_id = await self.db.fetchval(
            query,
            baseline.agent_type,
            baseline.environment,
            baseline.mean_reward,
            baseline.std_reward,
            baseline.success_rate,
            json.dumps(baseline.hyperparameters),
            baseline.version,
            baseline.rank
        )
        return str(baseline_id)

    async def update_baseline_if_better(
        self,
        agent_type: str,
        environment: str,
        new_metrics: MetricsResult,
        hyperparameters: Dict[str, Any],
        version: int
    ) -> Optional[str]:
        """
        Update baseline if new metrics are better.

        Returns:
            baseline_id if updated, None otherwise
        """
        current = await self.get_baseline(agent_type, environment)

        if current is None or new_metrics.mean_reward > current.mean_reward:
            # Demote current baseline
            if current:
                await self.rerank_baselines(agent_type, environment)

            # Insert new baseline as rank 1
            new_baseline = Baseline(
                agent_type=agent_type,
                environment=environment,
                mean_reward=new_metrics.mean_reward,
                std_reward=new_metrics.std_reward,
                success_rate=new_metrics.success_rate,
                hyperparameters=hyperparameters,
                created_at=datetime.utcnow(),
                version=version,
                rank=1
            )
            return await self.insert_baseline(new_baseline)

        return None

    async def rerank_baselines(
        self,
        agent_type: str,
        environment: str
    ):
        """Increment rank of all existing baselines."""
        query = """
            UPDATE eval_baselines
            SET rank = rank + 1
            WHERE agent_type = $1 AND environment = $2
        """
        await self.db.execute(query, agent_type, environment)

    async def get_baseline_history(
        self,
        agent_type: str,
        environment: str,
        limit: int = 10
    ) -> List[Baseline]:
        """Get historical baselines ordered by rank."""
        query = """
            SELECT * FROM eval_baselines
            WHERE agent_type = $1 AND environment = $2
            ORDER BY rank ASC
            LIMIT $3
        """
        rows = await self.db.fetch(query, agent_type, environment, limit)
        return [Baseline.from_db_row(row) for row in rows]
```

---

## 10. Environment Management

### 10.1 Environment Manager (environment_manager.py)

**Responsibilities:**
- Gymnasium environment lifecycle
- LRU caching (50 envs default)
- Environment wrappers (noise injection, reward delay)
- Thread-safe environment access

**Key Methods:**
```python
class EnvironmentCache:
    def __init__(self, max_size: int = 50):
        self.cache: OrderedDict[str, CachedEnvironment] = OrderedDict()
        self.max_size = max_size
        self.lock = asyncio.Lock()

    async def get_or_create_env(
        self,
        env_id: str,
        apply_wrappers: bool = True
    ) -> gym.Env:
        """
        Get environment from cache or create new one.

        LRU eviction policy: Remove least recently used if cache full.
        """
        async with self.lock:
            if env_id in self.cache:
                # Move to end (most recently used)
                cached = self.cache.pop(env_id)
                cached.last_accessed = datetime.utcnow()
                cached.access_count += 1
                self.cache[env_id] = cached
                return cached.env

            # Create new environment
            env = await self._create_env(env_id, apply_wrappers)

            # Evict LRU if cache full
            if len(self.cache) >= self.max_size:
                lru_key, lru_cached = self.cache.popitem(last=False)
                lru_cached.env.close()

            # Add to cache
            self.cache[env_id] = CachedEnvironment(
                env_id=env_id,
                env=env,
                last_accessed=datetime.utcnow(),
                access_count=1
            )

            return env

    async def _create_env(
        self,
        env_id: str,
        apply_wrappers: bool
    ) -> gym.Env:
        """
        Create Gymnasium environment with optional wrappers.

        Wrappers:
        - NoiseWrapper: Add Gaussian noise to observations (sigma=0.01)
        - RewardDelayWrapper: Delay reward by 1-3 steps
        """
        env = gym.make(env_id)

        if apply_wrappers:
            env = NoiseWrapper(env, noise_sigma=0.01)
            env = RewardDelayWrapper(env, delay_range=(1, 3))

        return env

    async def close_all(self):
        """Close all cached environments."""
        async with self.lock:
            for cached in self.cache.values():
                cached.env.close()
            self.cache.clear()


class NoiseWrapper(gym.ObservationWrapper):
    """Add Gaussian noise to observations."""
    def __init__(self, env, noise_sigma: float = 0.01):
        super().__init__(env)
        self.noise_sigma = noise_sigma

    def observation(self, obs):
        noise = np.random.normal(0, self.noise_sigma, size=obs.shape)
        return obs + noise


class RewardDelayWrapper(gym.Wrapper):
    """Delay reward by N steps."""
    def __init__(self, env, delay_range: Tuple[int, int] = (1, 3)):
        super().__init__(env)
        self.delay_range = delay_range
        self.reward_buffer = deque()

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)

        # Add current reward to buffer
        delay = np.random.randint(*self.delay_range)
        self.reward_buffer.append((reward, delay))

        # Release delayed rewards
        released_reward = 0.0
        for i, (r, d) in enumerate(self.reward_buffer):
            if d <= 0:
                released_reward += r
                self.reward_buffer.popleft()
            else:
                self.reward_buffer[i] = (r, d - 1)

        return obs, released_reward, done, truncated, info
```

---

## 11. Database Schema

### 11.1 PostgreSQL Migration (007_eval_baselines.sql)

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

---

## 12. Deployment Architecture

### 12.1 Kubernetes Deployment

**Helm Chart:** `charts/tars/templates/eval-engine-deployment.yaml`

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
          value: "{{ .Values.evalEngine.defaultEpisodes }}"
        - name: EVAL_MAX_CONCURRENT
          value: "{{ .Values.evalEngine.maxConcurrent }}"
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

**Helm Values:**
```yaml
evalEngine:
  enabled: true
  replicaCount: 2
  image:
    repository: tars/eval-engine
    tag: v1.0.0-rc2
  resources:
    limits:
      cpu: "2000m"
      memory: "4Gi"
    requests:
      cpu: "1000m"
      memory: "2Gi"
  defaultEpisodes: 100
  maxConcurrent: 4
  autoscaling:
    enabled: true
    minReplicas: 2
    maxReplicas: 10
    targetCPUUtilizationPercentage: 70
```

### 12.2 Service Definition

```yaml
apiVersion: v1
kind: Service
metadata:
  name: {{ include "tars.fullname" . }}-eval-engine
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

---

## 13. Performance Requirements

### 13.1 SLOs

| Metric | Target | P50 | P95 | P99 |
|--------|--------|-----|-----|-----|
| **Quick Mode Eval (50 eps)** | <30s | 25s | 35s | 45s |
| **Full Eval (100 eps, 1 env)** | <60s | 50s | 70s | 90s |
| **Multi-Env Eval (3 envs)** | <120s | 110s | 140s | 180s |
| **API Response Time** | <50ms | 30ms | 60ms | 100ms |
| **Regression Detection** | <100ms | 50ms | 120ms | 200ms |
| **Baseline Lookup** | <10ms | 5ms | 15ms | 30ms |
| **Nash Scoring (4 agents)** | <200ms | 150ms | 250ms | 400ms |

### 13.2 Resource Limits

- **CPU:** 2 cores per replica (4 cores max during burst)
- **Memory:** 4GB per replica (8GB max)
- **Concurrent Evaluations:** 4 per replica
- **Environment Cache:** 50 environments per replica (~500MB)

### 13.3 Scaling Strategy

- **HPA Trigger:** CPU >70% for 2 minutes
- **Min Replicas:** 2 (HA)
- **Max Replicas:** 10
- **Scale-Up Rate:** 2 pods/minute
- **Scale-Down Rate:** 1 pod/5 minutes (slow drain)

---

## 14. Security Considerations

### 14.1 Authentication & Authorization

**JWT Authentication:**
- All endpoints (except `/health`) require valid JWT
- Roles: `viewer` (read-only), `developer` (submit evals), `admin` (manage baselines)

**RBAC Matrix:**

| Endpoint | Viewer | Developer | Admin |
|----------|--------|-----------|-------|
| `POST /v1/evaluate` | ❌ | ✅ | ✅ |
| `GET /v1/jobs/{id}` | ✅ | ✅ | ✅ |
| `GET /v1/baselines` | ✅ | ✅ | ✅ |
| `POST /v1/baselines` | ❌ | ❌ | ✅ |

### 14.2 Rate Limiting

**Public Endpoints:**
- `GET /health`: Unlimited
- `GET /metrics`: Unlimited (Prometheus scraping)

**Authenticated Endpoints:**
- Viewer: 30 requests/minute
- Developer: 10 requests/minute (POST /evaluate)
- Admin: 20 requests/minute

### 14.3 Input Validation

- **Agent State:** Max size 10MB (prevent OOM)
- **Hyperparameters:** Schema validation with Pydantic
- **Environments:** Whitelist only (`CartPole-v1`, `LunarLander-v2`, `MountainCar-v0`)
- **Episodes:** Max 500 (prevent resource exhaustion)

### 14.4 Secrets Management

- **Postgres Credentials:** Kubernetes Secret
- **JWT Secret:** Shared ConfigMap with other services
- **Redis Password:** Optional, via Secret

---

## 15. Phase 13.2 Task Breakdown

### 15.1 Task List (14 Tasks)

| Task ID | Description | Owner | Est. LOC | Priority |
|---------|-------------|-------|----------|----------|
| **PH13.2-T01** | Create directory structure | DevOps | N/A | P0 |
| **PH13.2-T02** | Implement config.py + models.py | Backend | 450 | P0 |
| **PH13.2-T03** | Implement environment_manager.py | Backend | 380 | P0 |
| **PH13.2-T04** | Implement metrics_calculator.py | Backend | 520 | P0 |
| **PH13.2-T05** | Implement baseline_manager.py | Backend | 420 | P0 |
| **PH13.2-T06** | Implement regression_detector.py | Backend | 380 | P0 |
| **PH13.2-T07** | Implement nash_scorer.py | Backend | 320 | P1 |
| **PH13.2-T08** | Implement agent_eval_worker.py | Backend | 650 | P0 |
| **PH13.2-T09** | Implement routes (eval, baseline, health) | Backend | 550 | P0 |
| **PH13.2-T10** | Implement main.py (FastAPI app) | Backend | 380 | P0 |
| **PH13.2-T11** | Create PostgreSQL migration 007 | Database | 80 | P0 |
| **PH13.2-T12** | Create Kubernetes manifests | DevOps | 350 | P1 |
| **PH13.2-T13** | Write unit tests | QA | 1200 | P0 |
| **PH13.2-T14** | Integration tests with AutoML | QA | 800 | P1 |

**Total Estimated LOC:** ~6,480 lines

### 15.2 Dependencies

```
T01 (Directory) → T02-T10 (All implementation tasks)
T02 (Config/Models) → T03-T08 (All core components)
T03 (Environment) → T08 (Worker needs env manager)
T04 (Metrics) → T08 (Worker needs metrics calc)
T05 (Baseline) → T06, T09 (Regression + routes need baseline)
T06 (Regression) → T08 (Worker needs regression detection)
T07 (Nash) → T08 (Worker uses Nash scoring)
T08 (Worker) → T09 (Routes call worker)
T09 (Routes) → T10 (Main includes routes)
T11 (Migration) → T05 (Baseline manager needs schema)
T10 (Main) → T13 (Tests need running service)
```

---

## 16. Appendices

### 16.1 Glossary

- **Baseline:** Historical best performance for an agent+environment
- **Regression:** Performance degradation compared to baseline
- **Nash Score:** Equilibrium score from multi-agent game theory
- **Quick Mode:** Evaluation with 50 episodes for faster feedback
- **LRU Cache:** Least Recently Used cache eviction policy

### 16.2 References

- [Phase 13.1 Unified Pipeline Design](PHASE13_1_UNIFIED_PIPELINE_DESIGN.md)
- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [Nash Equilibrium Solver](cognition/orchestration-agent/nash/solver.py)
- [T.A.R.S. Phase 11 Architecture](PHASE11_ARCHITECTURE.md)

---

**End of Phase 13.2 Design Document**
**Next:** Phase 13.2 Scaffolding → Phase 13.3 Implementation
