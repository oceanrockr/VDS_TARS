# C4 Level 4: Code Diagram

## T.A.R.S. Evaluation Engine - Code-Level Architecture

This diagram shows the class-level structure and key code patterns for critical components.

## 1. AgentEvaluationWorker - Class Diagram

```mermaid
classDiagram
    class AgentEvaluationWorker {
        -EnvironmentCache env_cache
        -MetricsCalculator metrics_calculator
        -RegressionDetector regression_detector
        -NashScorer nash_scorer
        +evaluate_agent_in_env(agent_type, hyperparameters, environment, num_episodes, baseline) EnvironmentResult
        -_load_agent(agent_type, hyperparameters) Agent
        -_run_episodes(agent, env, num_episodes) Tuple~List~float~, List~int~~
        -_handle_worker_error(error) None
    }

    class EnvironmentCache {
        -OrderedDict~str, gym.Env~ _cache
        -int _max_size
        -Redis _redis
        -int _hits
        -int _misses
        +get_or_create(env_name) gym.Env
        +clear_cache() None
        +get_stats() CacheStats
        -_add_to_cache(env_name, env) None
        -_evict_lru() None
    }

    class MetricsCalculator {
        +calculate(episode_rewards, episode_steps, success_threshold) MetricsResult$
        -_compute_percentiles(rewards) Dict~str, float~$
    }

    class RegressionDetector {
        -float failure_rate_threshold
        -float reward_drop_pct_threshold
        -float variance_multiplier
        +detect(current_metrics, baseline) RegressionReport
        -_check_reward_drop(current, baseline) Tuple~bool, float~
        -_check_failure_rate(metrics) Tuple~bool, float~
        -_check_variance_explosion(current, baseline) Tuple~bool, float~
        -_aggregate_severity(signals) str
    }

    class NashScorer {
        -float conflict_weight
        -float deviation_weight
        -float stability_threshold
        +score(agent_type, current_metrics, baseline, all_agents_metrics) NashScores
        -_compute_conflict(all_agents_metrics) float
        -_compute_deviation(current, baseline) float
        -_compute_stability(conflict, deviation) float
        -_recommend_action(stability_score) str
    }

    class BaselineManager {
        -Pool db_pool
        +get_baseline(agent_type, environment, rank) BaselineRecord
        +get_baseline_history(agent_type, environment, limit) List~BaselineRecord~
        +update_baseline_if_better(agent_type, environment, metrics, hyperparameters) str
        +demote_ranks(agent_type, environment) None
        -_insert_new_baseline(record) str
        -_compare_metrics(current, baseline) bool
    }

    AgentEvaluationWorker --> EnvironmentCache : uses
    AgentEvaluationWorker --> MetricsCalculator : uses
    AgentEvaluationWorker --> RegressionDetector : uses
    AgentEvaluationWorker --> NashScorer : uses
```

## 2. Evaluation Flow - Sequence Diagram

```mermaid
sequenceDiagram
    participant Route as eval_routes
    participant Worker as AgentEvaluationWorker
    participant Cache as EnvironmentCache
    participant Metrics as MetricsCalculator
    participant Regression as RegressionDetector
    participant Nash as NashScorer
    participant DB as BaselineManager

    Route->>DB: get_baseline(agent_type, environment)
    DB-->>Route: baseline

    Route->>Worker: evaluate_agent_in_env(...)
    activate Worker

    Worker->>Worker: _load_agent(agent_type, hyperparameters)
    Worker->>Cache: get_or_create(environment)
    activate Cache
    Cache->>Cache: Check L1 cache
    alt Cache Hit
        Cache-->>Worker: cached_env
    else Cache Miss
        Cache->>Cache: gym.make(environment)
        Cache->>Cache: _add_to_cache(env_name, env)
        Cache-->>Worker: new_env
    end
    deactivate Cache

    loop num_episodes times
        Worker->>Worker: _run_episodes(agent, env, num_episodes)
        Worker->>Worker: Collect episode_rewards, episode_steps
    end

    Worker->>Metrics: calculate(episode_rewards, episode_steps)
    activate Metrics
    Metrics->>Metrics: np.mean(), np.std()
    Metrics-->>Worker: MetricsResult
    deactivate Metrics

    alt baseline provided
        Worker->>Regression: detect(current_metrics, baseline)
        activate Regression
        Regression->>Regression: _check_reward_drop()
        Regression->>Regression: _check_failure_rate()
        Regression->>Regression: _check_variance_explosion()
        Regression->>Regression: _aggregate_severity(signals)
        Regression-->>Worker: RegressionReport
        deactivate Regression
    end

    Worker->>Nash: score(agent_type, metrics, baseline, all_agents)
    activate Nash
    Nash->>Nash: _compute_conflict()
    Nash->>Nash: _compute_deviation()
    Nash->>Nash: _compute_stability()
    Nash->>Nash: _recommend_action()
    Nash-->>Worker: NashScores
    deactivate Nash

    Worker-->>Route: EnvironmentResult(metrics, regression, nash)
    deactivate Worker

    Route-->>Route: Aggregate results
    Route-->>Route: Return EvaluationResult
```

## 3. Baseline Management - State Diagram

```mermaid
stateDiagram-v2
    [*] --> CheckingBaseline: update_baseline_if_better()

    CheckingBaseline --> NoBaseline: No rank=1 baseline exists
    CheckingBaseline --> ComparingMetrics: Baseline exists

    NoBaseline --> InsertingNew: Create rank=1 baseline
    InsertingNew --> [*]: Return baseline_id

    ComparingMetrics --> IsBetter: current.mean_reward > baseline.mean_reward
    ComparingMetrics --> IsWorse: current.mean_reward <= baseline.mean_reward

    IsBetter --> DemotingRanks: Shift ranks (1→2, 2→3, ...)
    DemotingRanks --> InsertingNew: Insert as rank=1
    InsertingNew --> [*]: Return baseline_id, is_new_best=true

    IsWorse --> FindingRank: Find appropriate rank by mean_reward
    FindingRank --> InsertingNew: Insert at rank=N
    InsertingNew --> [*]: Return baseline_id, is_new_best=false
```

**SQL for Demoting Ranks:**
```sql
-- Shift all ranks down by 1
UPDATE eval_baselines
SET rank = rank + 1
WHERE agent_type = $1 AND environment = $2;

-- Insert new baseline at rank=1
INSERT INTO eval_baselines (
    agent_type, environment, mean_reward, std_reward,
    success_rate, hyperparameters, rank, version
) VALUES (
    $1, $2, $3, $4, $5, $6, 1, (SELECT COALESCE(MAX(version), 0) + 1 FROM eval_baselines WHERE agent_type=$1 AND environment=$2)
) RETURNING id;
```

## 4. Regression Detection Algorithm

```mermaid
flowchart TD
    A[Start: detect current_metrics, baseline] --> B{Reward Drop Check}

    B -->|drop > 10%| C[Signal: reward_drop, severity=high]
    B -->|drop <= 10%| D{Failure Rate Check}

    D -->|rate > 15%| E[Signal: high_failure_rate, severity=medium]
    D -->|rate <= 15%| F{Variance Check}

    F -->|std > 2.5x baseline| G[Signal: variance_explosion, severity=low]
    F -->|std <= 2.5x| H{Any Signals?}

    C --> I[Aggregate Signals]
    E --> I
    G --> I

    I --> J[severity = max signal severity]
    J --> K[confidence = 0.80 + 0.05 * num_signals]
    K --> L[is_regression = True]
    L --> M[Return RegressionReport]

    H -->|No| N[is_regression = False, severity = none]
    N --> M

    M --> O[End]
```

**Code Implementation:**
```python
def detect(self, current_metrics: MetricsResult, baseline: BaselineRecord) -> RegressionReport:
    signals = []

    # Signal 1: Reward drop
    reward_drop_pct = (baseline.mean_reward - current_metrics.mean_reward) / baseline.mean_reward
    if reward_drop_pct > 0.10:  # 10% threshold
        signals.append(("reward_drop", reward_drop_pct, "high"))

    # Signal 2: Failure rate
    failure_rate = 1 - current_metrics.success_rate
    if failure_rate > 0.15:  # 15% threshold
        signals.append(("high_failure_rate", failure_rate, "medium"))

    # Signal 3: Variance explosion
    if current_metrics.std_reward > baseline.std_reward * 2.5:
        signals.append(("variance_explosion", current_metrics.std_reward / baseline.std_reward, "low"))

    if not signals:
        return RegressionReport(is_regression=False, confidence=0.98, severity="none", details="Stable")

    severity = max(s[2] for s in signals)  # "high" > "medium" > "low"
    confidence = min(0.95, 0.80 + 0.05 * len(signals))
    details = "; ".join([f"{s[0]}: {s[1]:.2%}" for s in signals])

    return RegressionReport(
        is_regression=True,
        confidence=confidence,
        severity=severity,
        details=details,
        metrics={s[0]: s[1] for s in signals}
    )
```

## 5. Environment Cache - LRU Implementation

```mermaid
flowchart TD
    A[get_or_create env_name] --> B{In L1 Cache?}

    B -->|Yes| C[Move to end LRU order]
    C --> D[Increment hits counter]
    D --> E[Return cached environment]

    B -->|No| F{In Redis L2?}
    F -->|Yes| G[Deserialize from Redis]
    G --> H[Add to L1 cache]
    H --> E

    F -->|No| I[Increment misses counter]
    I --> J[gym.make env_name]
    J --> K{Cache Full?}

    K -->|Yes| L[Evict oldest LRU entry]
    L --> M[Close evicted environment]
    M --> N[Add new environment to cache]

    K -->|No| N
    N --> O[Optional: Serialize to Redis]
    O --> E

    E --> P[End]
```

**Code Implementation:**
```python
from collections import OrderedDict

class EnvironmentCache:
    def __init__(self, max_size: int = 50):
        self._cache: OrderedDict[str, gym.Env] = OrderedDict()
        self._max_size = max_size
        self._hits = 0
        self._misses = 0

    async def get_or_create(self, env_name: str) -> gym.Env:
        # L1 cache hit
        if env_name in self._cache:
            self._hits += 1
            self._cache.move_to_end(env_name)  # Mark as recently used
            ENV_CACHE_SIZE.set(len(self._cache))
            return self._cache[env_name]

        # Cache miss
        self._misses += 1
        env = gym.make(env_name)
        self._add_to_cache(env_name, env)
        return env

    def _add_to_cache(self, env_name: str, env: gym.Env):
        # LRU eviction if full
        if len(self._cache) >= self._max_size:
            evicted_name, evicted_env = self._cache.popitem(last=False)
            evicted_env.close()
            logger.info(f"Evicted environment: {evicted_name}")

        self._cache[env_name] = env
        ENV_CACHE_SIZE.set(len(self._cache))
```

## 6. Pydantic Models - Data Flow

```mermaid
classDiagram
    class EvaluationRequest {
        +str agent_type
        +Dict~str, Any~ hyperparameters
        +List~str~ environments
        +int num_episodes
        +bool compare_to_baseline
        +bool detect_regressions
    }

    class EvaluationResult {
        +str job_id
        +str status
        +str agent_type
        +List~str~ environments
        +Dict~str, EnvironmentResult~ results
        +datetime started_at
        +datetime completed_at
        +float duration_seconds
        +Optional~str~ error
    }

    class EnvironmentResult {
        +MetricsResult metrics
        +Optional~RegressionReport~ regression
        +Optional~NashScores~ nash_scores
    }

    class MetricsResult {
        +float mean_reward
        +float std_reward
        +float min_reward
        +float max_reward
        +float success_rate
        +float mean_steps
    }

    class RegressionReport {
        +bool is_regression
        +float confidence
        +str severity
        +str details
        +Optional~Dict~ metrics
    }

    class NashScores {
        +float conflict_score
        +float deviation_from_equilibrium
        +float stability_score
        +str recommendation
        +Optional~str~ details
    }

    class BaselineRecord {
        +str id
        +str agent_type
        +str environment
        +float mean_reward
        +float std_reward
        +float success_rate
        +Dict hyperparameters
        +int rank
        +int version
        +datetime created_at
    }

    EvaluationRequest --> EvaluationResult : produces
    EvaluationResult --> EnvironmentResult : contains
    EnvironmentResult --> MetricsResult : contains
    EnvironmentResult --> RegressionReport : contains
    EnvironmentResult --> NashScores : contains
    RegressionReport --> BaselineRecord : compares with
```

## 7. FastAPI Dependency Injection Graph

```mermaid
graph TB
    subgraph "HTTP Request"
        A[POST /v1/evaluate]
    end

    subgraph "Middleware"
        B[Auth Middleware]
        C[Rate Limiter]
        D[Tracing Middleware]
    end

    subgraph "Route Handler"
        E[evaluate_agent]
    end

    subgraph "Dependencies"
        F[get_current_user]
        G[get_worker]
        H[get_baseline_manager]
    end

    subgraph "Global State"
        I[_worker: AgentEvaluationWorker]
        J[_baseline_manager: BaselineManager]
        K[_env_cache: EnvironmentCache]
        L[db_pool: asyncpg.Pool]
        M[redis_client: aioredis.Redis]
    end

    A --> B
    B --> C
    C --> D
    D --> E

    E --> F
    E --> G
    E --> H

    F --> B
    G --> I
    H --> J

    I --> K
    J --> L
    K --> M
```

**Dependency Registration (main.py):**
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    global db_pool, redis_client, _env_cache, _worker, _baseline_manager

    # Initialize connections
    db_pool = await asyncpg.create_pool(config.postgres_url)
    redis_client = await aioredis.from_url(config.redis_url)

    # Initialize components
    _env_cache = EnvironmentCache(max_size=config.env_cache_size)
    _metrics_calculator = MetricsCalculator()
    _regression_detector = RegressionDetector(
        failure_rate_threshold=config.failure_rate,
        reward_drop_pct_threshold=config.reward_drop_pct
    )
    _nash_scorer = NashScorer()
    _baseline_manager = BaselineManager(db_pool)

    # Initialize worker
    _worker = AgentEvaluationWorker(
        env_cache=_env_cache,
        metrics_calculator=_metrics_calculator,
        regression_detector=_regression_detector,
        nash_scorer=_nash_scorer
    )

    yield

    # Cleanup
    await db_pool.close()
    await redis_client.close()
```

## 8. Error Handling Patterns

### Custom Exception Hierarchy
```python
class EvalEngineException(Exception):
    """Base exception for eval-engine."""
    pass

class EnvironmentNotFoundError(EvalEngineException):
    """Raised when Gymnasium environment doesn't exist."""
    pass

class AgentLoadError(EvalEngineException):
    """Raised when agent fails to load."""
    pass

class EvaluationTimeoutError(EvalEngineException):
    """Raised when evaluation exceeds timeout."""
    pass

class BaselineNotFoundError(EvalEngineException):
    """Raised when baseline doesn't exist."""
    pass
```

### Exception Handling in Worker
```python
async def evaluate_agent_in_env(self, ...) -> EnvironmentResult:
    try:
        agent = self._load_agent(agent_type, hyperparameters)
    except Exception as e:
        raise AgentLoadError(f"Failed to load agent: {e}")

    try:
        env = await self.env_cache.get_or_create(environment)
    except gym.error.UnregisteredEnv:
        raise EnvironmentNotFoundError(f"Environment '{environment}' not found")

    try:
        async with asyncio.timeout(EVAL_WORKER_TIMEOUT):
            episode_rewards, episode_steps = await self._run_episodes(agent, env, num_episodes)
    except asyncio.TimeoutError:
        raise EvaluationTimeoutError(f"Evaluation exceeded {EVAL_WORKER_TIMEOUT}s timeout")

    # Rest of evaluation...
```

### HTTP Exception Mapping
```python
# In main.py
@app.exception_handler(EvalEngineException)
async def eval_engine_exception_handler(request: Request, exc: EvalEngineException):
    status_code = 500
    if isinstance(exc, EnvironmentNotFoundError):
        status_code = 400
    elif isinstance(exc, BaselineNotFoundError):
        status_code = 404
    elif isinstance(exc, EvaluationTimeoutError):
        status_code = 504

    return JSONResponse(
        status_code=status_code,
        content={
            "error": exc.__class__.__name__,
            "message": str(exc),
            "trace_id": request.state.trace_id if hasattr(request.state, "trace_id") else None
        }
    )
```

## 9. Prometheus Metrics Implementation

```python
from prometheus_client import Counter, Histogram, Gauge

# Counters
EVALUATIONS_TOTAL = Counter(
    "tars_eval_evaluations_total",
    "Total agent evaluations",
    ["agent_type", "environment", "status"]
)

REGRESSION_DETECTED_TOTAL = Counter(
    "tars_eval_regression_detected_total",
    "Total regressions detected",
    ["agent_type", "environment", "severity"]
)

# Histograms
EVALUATION_DURATION = Histogram(
    "tars_eval_duration_seconds",
    "Evaluation duration in seconds",
    ["agent_type", "num_episodes"],
    buckets=[10, 30, 60, 120, 300, 600]
)

# Gauges
ENV_CACHE_SIZE = Gauge(
    "tars_eval_env_cache_size",
    "Number of cached environments"
)

# Usage in worker
async def evaluate_agent_in_env(self, ...):
    start_time = time.time()
    try:
        # Evaluation logic...
        EVALUATIONS_TOTAL.labels(agent_type=agent_type, environment=environment, status="success").inc()
        if regression_report.is_regression:
            REGRESSION_DETECTED_TOTAL.labels(
                agent_type=agent_type,
                environment=environment,
                severity=regression_report.severity
            ).inc()
    except Exception:
        EVALUATIONS_TOTAL.labels(agent_type=agent_type, environment=environment, status="failed").inc()
        raise
    finally:
        duration = time.time() - start_time
        EVALUATION_DURATION.labels(agent_type=agent_type, num_episodes=str(num_episodes)).observe(duration)
```

---

**Previous Levels:**
- [C4 Level 1 - System Context](C4_Level1_SystemContext.md)
- [C4 Level 2 - Container Diagram](C4_Level2_Container.md)
- [C4 Level 3 - Component Diagram](C4_Level3_Component.md)
