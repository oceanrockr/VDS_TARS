# C4 Level 1: System Context Diagram

## T.A.R.S. Evaluation Engine - System Context

This diagram shows how the Evaluation Engine fits within the broader T.A.R.S. ecosystem and its interactions with external actors and systems.

```mermaid
C4Context
    title System Context - T.A.R.S. Evaluation Engine

    Person(developer, "ML Developer", "Develops and optimizes RL agents")
    Person(sre, "SRE/Operator", "Monitors and operates the platform")
    Person(admin, "Administrator", "Manages baselines and system configuration")

    System_Boundary(tars, "T.A.R.S. Platform") {
        System(eval_engine, "Evaluation Engine", "Real RL agent evaluation with regression detection and baseline management")
        System(automl, "AutoML Pipeline", "Hyperparameter optimization with Optuna + MLflow")
        System(hypersync, "HyperSync Service", "Hot-reload hyperparameters with approval workflow")
        System(orchestration, "Multi-Agent Orchestration", "Coordinates 4 RL agents (DQN, A2C, PPO, DDPG)")
        System(dashboard, "Dashboard UI", "React frontend for metrics and control")
    }

    System_Ext(prometheus, "Prometheus", "Metrics collection and alerting")
    System_Ext(grafana, "Grafana", "Metrics visualization")
    System_Ext(jaeger, "Jaeger", "Distributed tracing backend")
    SystemDb_Ext(postgres, "PostgreSQL", "Persistent storage for baselines and results")
    SystemDb_Ext(redis, "Redis", "Caching, streams, and rate limiting")

    Rel(developer, eval_engine, "Submits agents for evaluation", "HTTPS/JSON")
    Rel(admin, eval_engine, "Manages baselines", "HTTPS/JSON")
    Rel(sre, grafana, "Monitors dashboards", "HTTPS")
    Rel(sre, prometheus, "Configures alerts", "HTTPS")

    Rel(automl, eval_engine, "Requests evaluation of optimized hyperparameters", "HTTP/JSON")
    Rel(eval_engine, hypersync, "Triggers hot-reload on regression", "HTTP/JSON")
    Rel(eval_engine, orchestration, "Evaluates agent performance", "HTTP/JSON")
    Rel(dashboard, eval_engine, "Fetches evaluation results", "HTTPS/JSON")

    Rel(eval_engine, postgres, "Stores baselines and history", "SQL")
    Rel(eval_engine, redis, "Caches environments, enforces rate limits", "Redis Protocol")
    Rel(eval_engine, prometheus, "Exports metrics", "HTTP/Prometheus")
    Rel(eval_engine, jaeger, "Sends traces", "OTLP/gRPC")

    Rel(prometheus, grafana, "Provides metrics data", "PromQL")
    Rel(grafana, sre, "Displays dashboards and alerts", "HTTPS")

    UpdateLayoutConfig($c4ShapeInRow="3", $c4BoundaryInRow="1")
```

## Key Interactions

### 1. Developer Workflow
```
Developer → Evaluation Engine:
  POST /v1/evaluate
    {
      "agent_type": "dqn",
      "hyperparameters": {...},
      "environments": ["CartPole-v1"],
      "num_episodes": 100,
      "detect_regressions": true
    }

Evaluation Engine → Developer:
  200 OK
    {
      "job_id": "uuid",
      "results": {
        "CartPole-v1": {
          "metrics": {...},
          "regression": {...},
          "nash_scores": {...}
        }
      }
    }
```

### 2. AutoML Integration
```
AutoML Pipeline → Evaluation Engine:
  POST /v1/evaluate (with optimized hyperparameters)

Evaluation Engine → AutoML Pipeline:
  Returns metrics for Optuna objective function

AutoML Pipeline → MLflow:
  Logs metrics + hyperparameters
```

### 3. Regression-Triggered Hot-Reload
```
Evaluation Engine → (Detects Regression):
  RegressionDetector.detect() → is_regression=True, severity=high

Evaluation Engine → HyperSync:
  POST /v1/rollback
    {
      "agent_type": "dqn",
      "reason": "Regression detected: -15% reward drop"
    }

HyperSync → Orchestration:
  Reverts to previous stable hyperparameters
```

### 4. Observability Flow
```
Evaluation Engine → Prometheus:
  Exports:
    - tars_eval_evaluations_total
    - tars_eval_duration_seconds
    - tars_eval_regression_detected_total

Prometheus → Grafana:
  Queries metrics via PromQL

Grafana → SRE:
  Displays dashboards, fires alerts
```

## External Dependencies

| System | Purpose | Protocol | Criticality |
|--------|---------|----------|-------------|
| PostgreSQL | Baseline storage | SQL | **Critical** |
| Redis | Environment cache, rate limiting | Redis | **High** |
| Prometheus | Metrics collection | HTTP | Medium |
| Jaeger | Distributed tracing | OTLP/gRPC | Low |
| AutoML | Upstream optimizer | HTTP/JSON | High |
| HyperSync | Downstream hot-reload | HTTP/JSON | High |

## Security Boundaries

```mermaid
graph TB
    subgraph "Public Zone"
        A[Developer]
        B[SRE]
    end

    subgraph "DMZ - API Gateway"
        C[Ingress + TLS Termination]
        D[Rate Limiter]
    end

    subgraph "Application Zone"
        E[Evaluation Engine]
        F[AutoML Pipeline]
        G[HyperSync]
    end

    subgraph "Data Zone"
        H[PostgreSQL]
        I[Redis]
    end

    A -->|HTTPS + JWT| C
    B -->|HTTPS + JWT| C
    C --> D
    D -->|JWT Validation| E
    E -->|TLS + mTLS| F
    E -->|TLS + mTLS| G
    E -->|Encrypted| H
    E -->|Encrypted| I
```

## Deployment Context

- **Kubernetes Cluster**: All services run as pods
- **Namespace**: `tars` (production) or `tars-staging`
- **Ingress**: TLS termination with cert-manager
- **Service Mesh**: Optional Istio for mTLS
- **Multi-Region**: Active-active with cross-region replication (Phase 6-9)

## Failure Modes

1. **PostgreSQL Down**: Evaluation fails, returns 503
2. **Redis Down**: Environment cache disabled, rate limiting degraded
3. **Prometheus Down**: Metrics lost, no alerts
4. **Jaeger Down**: Traces lost, no impact on functionality
5. **AutoML Down**: Eval engine continues serving direct requests
6. **HyperSync Down**: Regression detection works, rollback unavailable

---

**Next Level**: [C4 Level 2 - Container Diagram](C4_Level2_Container.md)
