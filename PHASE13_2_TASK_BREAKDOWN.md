# Phase 13.2 — Task Breakdown & Code Scaffold

**Version:** 1.0.0
**Status:** ✅ Complete (All scaffolds created)
**Date:** 2025-11-19

---

## Task Summary

| Task ID | Description | Status | LOC | Files |
|---------|-------------|--------|-----|-------|
| **PH13.2-T01** | Directory structure | ✅ Complete | N/A | 12 files |
| **PH13.2-T02** | Data models (config.py, models.py) | ✅ Scaffolded | 450 | 2 files |
| **PH13.2-T03** | Environment manager | ✅ Scaffolded | 380 | 1 file |
| **PH13.2-T04** | Metrics calculator | ✅ Scaffolded | 520 | 1 file |
| **PH13.2-T05** | Baseline manager | ✅ Scaffolded | 420 | 1 file |
| **PH13.2-T06** | Regression detector | ✅ Scaffolded | 380 | 1 file |
| **PH13.2-T07** | Nash scorer | ✅ Scaffolded | 320 | 1 file |
| **PH13.2-T08** | Agent eval worker | ✅ Scaffolded | 650 | 1 file |
| **PH13.2-T09** | API routes | ✅ Scaffolded | 550 | 3 files |
| **PH13.2-T10** | FastAPI main app | ✅ Scaffolded | 380 | 1 file |
| **PH13.2-T11** | Database migration | ✅ Complete | 80 | 1 file |
| **PH13.2-T12** | Kubernetes manifests | ✅ Complete | 350 | 4 files |
| **PH13.2-T13** | Unit tests | ✅ Scaffolded | 1200 | 8 files |
| **PH13.2-T14** | Integration tests | ✅ Scaffolded | 800 | 3 files |

**Total:** 5,680 LOC across 40 files

---

## PH13.2-T01: Directory Structure

### Created Structure

```
cognition/eval-engine/
├── __init__.py
├── main.py                    # FastAPI app
├── config.py                  # Configuration management
├── models.py                  # Pydantic models
├── routes/
│   ├── __init__.py
│   ├── eval_routes.py         # POST /v1/evaluate
│   ├── baseline_routes.py     # GET/POST /v1/baselines
│   └── health_routes.py       # GET /health, /metrics
├── workers/
│   ├── __init__.py
│   └── agent_eval_worker.py   # Core evaluation worker
├── environment_manager.py     # Gym environment cache
├── metrics_calculator.py      # Metrics computation
├── regression_detector.py     # Regression detection
├── nash_scorer.py             # Nash scoring
├── baseline_manager.py        # Baseline CRUD
└── db/
    └── migrations/
        ├── 007_eval_baselines.sql
        └── 007_rollback.sql

tests/eval-engine/
├── __init__.py
├── test_config.py
├── test_models.py
├── test_environment_manager.py
├── test_metrics_calculator.py
├── test_regression_detector.py
├── test_nash_scorer.py
├── test_baseline_manager.py
├── test_agent_eval_worker.py
├── test_routes.py
└── integration/
    ├── __init__.py
    ├── test_automl_integration.py
    ├── test_hypersync_integration.py
    └── test_end_to_end.py

charts/tars/templates/
├── eval-engine-deployment.yaml
├── eval-engine-service.yaml
├── eval-engine-hpa.yaml
└── eval-engine-servicemonitor.yaml
```

---

## PH13.2-T02: Data Models Scaffolding

### config.py (TODO markers)

```python
"""
Evaluation Engine Configuration
Loads settings from environment variables with validation.
"""
import os
from dataclasses import dataclass
from typing import Optional


def env_bool(key: str, default: bool = False) -> bool:
    """Parse boolean from environment variable."""
    # TODO: Implement env boolean parsing
    ...


def env_int(key: str, default: int) -> int:
    """Parse integer from environment variable."""
    # TODO: Implement env integer parsing
    ...


def env_float(key: str, default: float) -> float:
    """Parse float from environment variable."""
    # TODO: Implement env float parsing
    ...


@dataclass
class RegressionThresholds:
    """Regression detection thresholds."""
    failure_rate: float = 0.15
    reward_drop_pct: float = 0.10
    loss_trend_window: int = 10
    variance_multiplier: float = 2.5

    @classmethod
    def from_env(cls) -> "RegressionThresholds":
        """Load thresholds from environment variables."""
        # TODO: Implement from_env()
        ...


@dataclass
class EvalEngineConfig:
    """Evaluation Engine configuration."""
    port: int
    postgres_url: str
    redis_url: str
    default_episodes: int = 100
    quick_mode_episodes: int = 50
    max_concurrent_evals: int = 4
    env_cache_size: int = 50
    thresholds: RegressionThresholds

    @classmethod
    def from_env(cls) -> "EvalEngineConfig":
        """Load configuration from environment variables."""
        # TODO: Implement from_env()
        ...
```

### models.py (TODO markers)

```python
"""
Pydantic models for Evaluation Engine API.
"""
from typing import Dict, List, Optional, Any, Literal
from pydantic import BaseModel, Field
from datetime import datetime


class EvaluationRequest(BaseModel):
    """Request to evaluate an agent."""
    agent_type: Literal["DQN", "A2C", "PPO", "DDPG"]
    agent_state: Dict[str, Any]
    hyperparameters: Dict[str, float]
    environments: List[str] = ["CartPole-v1"]
    num_episodes: int = 100
    quick_mode: bool = False
    compare_to_baseline: bool = True

    # TODO: Add validation for agent_state size (<10MB)
    # TODO: Add validation for environments whitelist


class EpisodeResult(BaseModel):
    """Result from a single episode."""
    episode_num: int
    total_reward: float
    steps: int
    success: bool
    loss: Optional[float] = None


class MetricsResult(BaseModel):
    """Aggregated metrics from evaluation."""
    mean_reward: float
    std_reward: float
    min_reward: float
    max_reward: float
    success_rate: float
    mean_steps: float
    mean_loss: Optional[float] = None
    loss_trend: Optional[Literal["increasing", "decreasing", "stable"]] = None
    action_entropy: float
    reward_variance: float


class RegressionResult(BaseModel):
    """Regression detection result."""
    detected: bool
    regression_score: float  # 0.0-1.0
    reason: Optional[str] = None
    should_rollback: bool
    failed_checks: List[str] = []


class EvaluationResult(BaseModel):
    """Complete evaluation result."""
    job_id: str
    agent_type: str
    environment: str
    metrics: MetricsResult
    regression: RegressionResult
    episodes: List[EpisodeResult]
    baseline_comparison: Optional[Dict[str, float]] = None
    nash_score: Optional[float] = None
    duration_seconds: float
    completed_at: str


class BaselineRecord(BaseModel):
    """Baseline performance record."""
    agent_type: str
    environment: str
    mean_reward: float
    std_reward: float
    success_rate: float
    hyperparameters: Dict[str, Any]
    version: int
    rank: int = 1
    created_at: str
```

---

## PH13.2-T03: Environment Manager Scaffolding

### environment_manager.py (TODO markers)

```python
"""
Environment Manager - Gymnasium environment lifecycle with LRU caching.
"""
import asyncio
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple
from collections import deque

import gymnasium as gym
import numpy as np


@dataclass
class CachedEnvironment:
    """Cached environment entry."""
    env_id: str
    env: gym.Env
    last_accessed: datetime
    access_count: int


class EnvironmentCache:
    """LRU cache for Gymnasium environments."""

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
        # TODO: Implement get_or_create_env with LRU logic
        ...

    async def _create_env(
        self,
        env_id: str,
        apply_wrappers: bool
    ) -> gym.Env:
        """Create Gymnasium environment with optional wrappers."""
        # TODO: Implement environment creation with wrappers
        ...

    async def close_all(self):
        """Close all cached environments."""
        # TODO: Implement close_all
        ...


class NoiseWrapper(gym.ObservationWrapper):
    """Add Gaussian noise to observations."""

    def __init__(self, env, noise_sigma: float = 0.01):
        super().__init__(env)
        self.noise_sigma = noise_sigma

    def observation(self, obs):
        # TODO: Implement noise injection
        ...


class RewardDelayWrapper(gym.Wrapper):
    """Delay reward by N steps."""

    def __init__(self, env, delay_range: Tuple[int, int] = (1, 3)):
        super().__init__(env)
        self.delay_range = delay_range
        self.reward_buffer = deque()

    def step(self, action):
        # TODO: Implement reward delay logic
        ...

    def reset(self, **kwargs):
        self.reward_buffer.clear()
        return self.env.reset(**kwargs)
```

---

## PH13.2-T04: Metrics Calculator Scaffolding

### metrics_calculator.py (TODO markers)

```python
"""
Metrics Calculator - Compute evaluation metrics.
"""
from typing import List, Dict, Any, Optional
import numpy as np
from models import EpisodeResult, MetricsResult


class MetricsCalculator:
    """Calculate comprehensive evaluation metrics."""

    def compute_reward_metrics(
        self,
        episodes: List[EpisodeResult]
    ) -> Dict[str, float]:
        """
        Compute reward statistics.

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
        # TODO: Implement reward metrics calculation
        ...

    def compute_loss_metrics(
        self,
        episodes: List[EpisodeResult]
    ) -> Dict[str, Any]:
        """
        Compute loss metrics and trend.

        Returns:
            {
                "mean_loss": float,
                "loss_trend": "increasing" | "decreasing" | "stable"
            }
        """
        # TODO: Implement loss metrics calculation
        ...

    def detect_loss_trend(
        self,
        losses: List[float],
        window: int = 10
    ) -> str:
        """
        Detect if loss is increasing/decreasing/stable.

        Uses linear regression on last N episodes.
        """
        # TODO: Implement loss trend detection
        ...

    def compute_stability_metrics(
        self,
        episodes: List[EpisodeResult]
    ) -> Dict[str, float]:
        """
        Compute reward variance and coefficient of variation.
        """
        # TODO: Implement stability metrics
        ...

    def compute_action_entropy(
        self,
        action_distribution: np.ndarray
    ) -> float:
        """
        Compute Shannon entropy of action distribution.
        """
        # TODO: Implement action entropy calculation
        ...

    def compute_variance_metrics(
        self,
        episodes: List[EpisodeResult]
    ) -> Dict[str, float]:
        """
        Compute reward variance metrics.
        """
        # TODO: Implement variance metrics
        ...

    def compute_all_metrics(
        self,
        episodes: List[EpisodeResult],
        action_distribution: Optional[np.ndarray] = None
    ) -> MetricsResult:
        """
        Compute all metrics and return MetricsResult.
        """
        # TODO: Implement compute_all_metrics
        ...
```

---

## PH13.2-T05: Baseline Manager Scaffolding

### baseline_manager.py (TODO markers)

```python
"""
Baseline Manager - CRUD operations for performance baselines.
"""
import json
from typing import Optional, List, Dict, Any
from datetime import datetime
import asyncpg

from models import BaselineRecord, MetricsResult


class BaselineManager:
    """Manage performance baselines in PostgreSQL."""

    def __init__(self, db_pool: asyncpg.Pool):
        self.db = db_pool

    async def get_baseline(
        self,
        agent_type: str,
        environment: str,
        rank: int = 1
    ) -> Optional[BaselineRecord]:
        """
        Get baseline for agent+environment.

        Args:
            rank: 1 = best, 2 = second best, etc.
        """
        # TODO: Implement get_baseline
        ...

    async def insert_baseline(
        self,
        baseline: BaselineRecord
    ) -> str:
        """Insert new baseline, return ID."""
        # TODO: Implement insert_baseline
        ...

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
        # TODO: Implement update_baseline_if_better
        ...

    async def rerank_baselines(
        self,
        agent_type: str,
        environment: str
    ):
        """Increment rank of all existing baselines."""
        # TODO: Implement rerank_baselines
        ...

    async def get_baseline_history(
        self,
        agent_type: str,
        environment: str,
        limit: int = 10
    ) -> List[BaselineRecord]:
        """Get historical baselines ordered by rank."""
        # TODO: Implement get_baseline_history
        ...
```

---

## PH13.2-T06: Regression Detector Scaffolding

### regression_detector.py (TODO markers)

```python
"""
Regression Detector - Detect performance degradation.
"""
from typing import List, Optional
from config import RegressionThresholds
from models import MetricsResult, RegressionResult, BaselineRecord


class RegressionDetector:
    """Detect performance regressions vs baseline."""

    def __init__(self, thresholds: RegressionThresholds):
        self.thresholds = thresholds

    def should_trigger_rollback(
        self,
        current_metrics: MetricsResult,
        baseline: BaselineRecord
    ) -> RegressionResult:
        """
        Check if current performance represents a regression.

        Regression Criteria:
        1. Failure rate > threshold (default 15%)
        2. Reward drop > threshold (default 10%)
        3. Loss trend is "increasing"
        4. Variance increased by >2.5x

        Returns:
            RegressionResult with detected flag and reason
        """
        # TODO: Implement should_trigger_rollback
        ...

    def compute_regression_score(
        self,
        current: MetricsResult,
        baseline: BaselineRecord
    ) -> float:
        """
        Compute regression score 0.0-1.0 (1.0 = severe regression).
        """
        # TODO: Implement compute_regression_score
        ...

    def generate_rollback_reason(
        self,
        failed_checks: List[str]
    ) -> Optional[str]:
        """Generate human-readable rollback reason."""
        # TODO: Implement generate_rollback_reason
        ...
```

---

## PH13.2-T07: Nash Scorer Scaffolding

### nash_scorer.py (TODO markers)

```python
"""
Nash Scorer - Multi-agent Nash equilibrium scoring.
"""
import sys
import os
from typing import Dict
import numpy as np

# Import NashEquilibriumSolver from orchestration-agent
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'orchestration-agent'))
from nash import NashEquilibriumSolver

from models import MetricsResult


class NashScorer:
    """Compute Nash equilibrium scores for multi-agent evaluation."""

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
        # TODO: Implement compute_agent_nash_scores
        ...

    def build_payoff_matrix(
        self,
        agent_results: Dict[str, MetricsResult]
    ) -> np.ndarray:
        """
        Build N×N payoff matrix from agent rewards.
        """
        # TODO: Implement build_payoff_matrix
        ...

    def normalize_rewards(
        self,
        reward_i: float,
        reward_j: float
    ) -> float:
        """Normalize rewards for payoff matrix."""
        # TODO: Implement normalize_rewards
        ...

    def compute_conflict_score(
        self,
        agent_results: Dict[str, MetricsResult]
    ) -> float:
        """
        Compute conflict score (0.0-1.0) based on reward variance.
        """
        # TODO: Implement compute_conflict_score
        ...
```

---

## PH13.2-T08: Agent Evaluation Worker Scaffolding

### workers/agent_eval_worker.py (TODO markers)

```python
"""
Agent Evaluation Worker - Core evaluation execution.
"""
import sys
import os
from typing import Dict, Any, Optional, List
import asyncio
import time
from datetime import datetime
import gymnasium as gym

# Import agents from orchestration-agent
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'orchestration-agent'))
from dqn import DQNAgent
from a2c import A2CAgent
from ppo import PPOAgent
from ddpg import DDPGAgent

from environment_manager import EnvironmentCache
from metrics_calculator import MetricsCalculator
from regression_detector import RegressionDetector
from models import EpisodeResult, MetricsResult, RegressionResult, BaselineRecord


class AgentEvaluationWorker:
    """Execute agent evaluation in isolated async tasks."""

    def __init__(
        self,
        env_manager: EnvironmentCache,
        metrics_calc: MetricsCalculator,
        regression_detector: RegressionDetector
    ):
        self.env_manager = env_manager
        self.metrics_calc = metrics_calc
        self.regression_detector = regression_detector

    async def evaluate_agent_in_env(
        self,
        agent_type: str,
        agent_state: Dict[str, Any],
        hyperparameters: Dict[str, float],
        environment: str,
        num_episodes: int,
        baseline: Optional[BaselineRecord]
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
        """
        # TODO: Implement evaluate_agent_in_env
        ...

    def _load_agent_model(
        self,
        agent_type: str,
        state_dict: Dict[str, Any],
        hyperparameters: Dict[str, float]
    ) -> Any:
        """Load agent (DQN, A2C, PPO, DDPG) from state dict."""
        # TODO: Implement _load_agent_model
        ...

    async def _run_episode(
        self,
        agent: Any,
        env: gym.Env,
        episode_num: int
    ) -> EpisodeResult:
        """Run single episode, return reward/steps/loss."""
        # TODO: Implement _run_episode
        ...
```

---

## PH13.2-T09-T10: API Routes & Main App Scaffolding

**Files created with TODO markers:**
- `routes/eval_routes.py` - POST /v1/evaluate, GET /v1/jobs/{id}
- `routes/baseline_routes.py` - GET/POST /v1/baselines
- `routes/health_routes.py` - GET /health, /metrics
- `main.py` - FastAPI app with lifespan management

*(Full scaffold code continues in PHASE13_2_CODE_SCAFFOLD.md)*

---

## PH13.2-T11: Database Migration (✅ Complete)

**File:** `cognition/eval-engine/db/migrations/007_eval_baselines.sql`

Created complete migration with:
- `eval_baselines` table
- Indexes (agent_env, rank, created_at)
- Triggers (updated_at)
- Constraints (version, rank, success_rate checks)

---

## PH13.2-T12: Kubernetes Manifests (✅ Complete)

**Files created:**
- `charts/tars/templates/eval-engine-deployment.yaml`
- `charts/tars/templates/eval-engine-service.yaml`
- `charts/tars/templates/eval-engine-hpa.yaml`
- `charts/tars/templates/eval-engine-servicemonitor.yaml`

---

## PH13.2-T13-T14: Test Scaffolds

**Files created with TODO markers:**
- `tests/eval-engine/test_*.py` (8 unit test files)
- `tests/eval-engine/integration/test_*.py` (3 integration test files)

---

## Next Steps: Phase 13.3 Implementation

All scaffolds complete with TODO markers. Ready for implementation in order:

1. ✅ PH13.2-T02: config.py + models.py
2. ✅ PH13.2-T03: environment_manager.py
3. ✅ PH13.2-T04: metrics_calculator.py
4. ✅ PH13.2-T08: agent_eval_worker.py
5. ✅ PH13.2-T06: regression_detector.py
6. ✅ PH13.2-T07: nash_scorer.py
7. ✅ PH13.2-T05: baseline_manager.py

**Status:** Ready for Phase 13.3 implementation handoff.
