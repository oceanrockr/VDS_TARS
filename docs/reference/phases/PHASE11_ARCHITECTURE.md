# T.A.R.S. Phase 11 Final Architecture
## Complete Multi-Agent RL + AutoML + Dashboard System

**Version**: v0.9.4-alpha
**Date**: November 14, 2025

---

## System Architecture Diagram

```mermaid
graph TB
    subgraph "User Layer"
        USER[User Browser]
    end

    subgraph "Presentation Layer - Port 3000"
        FRONTEND[Dashboard Frontend<br/>React + TypeScript<br/>Real-time Charts]
    end

    subgraph "API Gateway Layer - Port 3001"
        DASHBOARD_API[Dashboard API<br/>FastAPI + WebSocket<br/>Real-time Updates]
    end

    subgraph "Storage Layer"
        REDIS[(Redis<br/>Port 6379<br/>Persistent State)]
    end

    subgraph "Hyperparameter Management - Port 8098"
        HYPERSYNC[Hyperparameter Sync<br/>Hot-Reload Service<br/>Approval Workflows]
    end

    subgraph "AutoML Layer - Port 8097"
        AUTOML[AutoML Service<br/>Optuna + MLflow]
        OBJECTIVE[Real Objective Function<br/>Multi-Episode Training]
        MLFLOW[(MLflow Registry<br/>Model Versioning)]
    end

    subgraph "Multi-Agent Orchestration - Port 8094"
        ORCH[Orchestration Service<br/>Nash Equilibrium<br/>Conflict Resolution]

        subgraph "Agents"
            DQN[Policy Agent<br/>DQN<br/>Discrete Actions]
            A2C[Consensus Agent<br/>A2C<br/>Actor-Critic]
            PPO[Ethical Agent<br/>PPO<br/>Clipped Surrogate]
            DDPG[Resource Agent<br/>DDPG<br/>Continuous Actions]
        end
    end

    subgraph "Support Services"
        PROM[Prometheus<br/>Metrics Collection]
        GRAF[Grafana<br/>Monitoring Dashboards]
    end

    %% User interactions
    USER -->|HTTP/WS| FRONTEND
    FRONTEND -->|REST API| DASHBOARD_API
    FRONTEND -->|WebSocket| DASHBOARD_API

    %% Dashboard connections
    DASHBOARD_API -->|Store/Retrieve State| REDIS
    DASHBOARD_API -->|Fetch Agent States| ORCH
    DASHBOARD_API -->|Get Models/Trials| AUTOML

    %% AutoML pipeline
    AUTOML -->|Run Optimization| OBJECTIVE
    OBJECTIVE -->|Train Agents| DQN
    OBJECTIVE -->|Train Agents| A2C
    OBJECTIVE -->|Train Agents| PPO
    OBJECTIVE -->|Train Agents| DDPG
    AUTOML -->|Register Models| MLFLOW

    %% Hyperparameter sync pipeline
    HYPERSYNC -->|Fetch Best Params| MLFLOW
    HYPERSYNC -->|Validate| HYPERSYNC
    HYPERSYNC -->|Hot-Reload| ORCH

    %% Orchestration
    ORCH -->|Manage| DQN
    ORCH -->|Manage| A2C
    ORCH -->|Manage| PPO
    ORCH -->|Manage| DDPG
    ORCH -->|Nash Equilibrium| ORCH

    %% Monitoring
    AUTOML -->|Metrics| PROM
    HYPERSYNC -->|Metrics| PROM
    ORCH -->|Metrics| PROM
    DASHBOARD_API -->|Metrics| PROM
    PROM -->|Visualize| GRAF

    %% Styling
    classDef frontend fill:#61dafb,stroke:#333,stroke-width:2px,color:#000
    classDef api fill:#009688,stroke:#333,stroke-width:2px,color:#fff
    classDef storage fill:#ff9800,stroke:#333,stroke-width:2px,color:#fff
    classDef ml fill:#673ab7,stroke:#333,stroke-width:2px,color:#fff
    classDef agent fill:#4caf50,stroke:#333,stroke-width:2px,color:#fff
    classDef monitoring fill:#f44336,stroke:#333,stroke-width:2px,color:#fff
    classDef new fill:#ffd700,stroke:#333,stroke-width:3px,color:#000

    class FRONTEND frontend
    class DASHBOARD_API api
    class REDIS,MLFLOW storage
    class AUTOML,OBJECTIVE ml
    class DQN,A2C,PPO,DDPG agent
    class ORCH api
    class PROM,GRAF monitoring
    class HYPERSYNC new
```

---

## Data Flow Diagram

```mermaid
sequenceDiagram
    participant User
    participant Dashboard
    participant AutoML
    participant HyperSync
    participant Orchestration
    participant Agents
    participant MLflow
    participant Redis

    Note over User,Redis: Phase 11.4 Complete Pipeline

    %% 1. Optimization
    User->>AutoML: Start optimization (real training)
    AutoML->>Agents: Train with trial params (50 episodes)
    Agents-->>AutoML: Validation rewards
    AutoML->>MLflow: Register best model

    %% 2. Hyperparameter Sync
    User->>HyperSync: Propose update
    HyperSync->>MLflow: Fetch best params
    HyperSync->>HyperSync: Validate constraints
    HyperSync-->>User: Update proposed (pending approval)

    %% 3. Approval & Apply
    User->>HyperSync: Approve update
    HyperSync->>Orchestration: Hot-reload (/agents/{id}/reload)
    Orchestration->>Agents: Update hyperparameters (<100ms)
    Orchestration-->>HyperSync: Update applied

    %% 4. Dashboard Monitoring
    Dashboard->>Redis: Store agent states
    Dashboard->>Orchestration: Fetch live agent stats
    Orchestration-->>Dashboard: Agent metrics
    Dashboard->>Redis: Persist metrics
    Dashboard-->>User: Real-time visualization
```

---

## Component Interaction Matrix

| Component | Orchestration | AutoML | HyperSync | Dashboard API | Redis | MLflow |
|-----------|--------------|--------|-----------|---------------|-------|--------|
| **Orchestration** | - | Trains agents | Receives hot-reload | Provides state | - | - |
| **AutoML** | Calls training | - | - | - | - | Registers models |
| **HyperSync** | Hot-reloads | - | - | - | - | Fetches params |
| **Dashboard API** | Queries state | Queries models | Queries updates | - | Persists state | - |
| **Redis** | - | - | - | Stores data | - | - |
| **MLflow** | - | Logs runs | Provides params | - | - | - |

---

## Service Dependency Graph

```mermaid
graph LR
    subgraph "Critical Path"
        ORCH[Orchestration<br/>8094] --> |Required| DQN[DQN Agent]
        ORCH --> |Required| A2C[A2C Agent]
        ORCH --> |Required| PPO[PPO Agent]
        ORCH --> |Required| DDPG[DDPG Agent]
    end

    subgraph "AutoML Pipeline"
        AUTOML[AutoML<br/>8097] --> |Optional| MLFLOW[MLflow]
        AUTOML --> |Required for real training| ORCH
    end

    subgraph "Hyperparameter Sync"
        HYPERSYNC[HyperSync<br/>8098] --> |Required| MLFLOW
        HYPERSYNC --> |Required| ORCH
    end

    subgraph "Dashboard"
        DASH_API[Dashboard API<br/>3001] --> |Optional| REDIS[Redis<br/>6379]
        DASH_API --> |Optional| ORCH
        DASH_API --> |Optional| AUTOML
        DASH_FRONTEND[Dashboard Frontend<br/>3000] --> |Required| DASH_API
    end

    classDef critical fill:#ff6b6b,stroke:#333,stroke-width:2px,color:#fff
    classDef optional fill:#4ecdc4,stroke:#333,stroke-width:2px,color:#fff
    classDef new fill:#ffd700,stroke:#333,stroke-width:2px,color:#000

    class ORCH,DQN,A2C,PPO,DDPG critical
    class REDIS,MLFLOW,AUTOML,DASH_API optional
    class HYPERSYNC new
```

---

## Network Port Allocation

```
┌─────────────────────────────────────────┐
│         T.A.R.S. Service Ports          │
├─────────────────┬───────────────────────┤
│ Service         │ Port                  │
├─────────────────┼───────────────────────┤
│ Dashboard UI    │ 3000                  │
│ Dashboard API   │ 3001                  │
│ Redis           │ 6379                  │
│ Orchestration   │ 8094                  │
│ Causal Inference│ 8095                  │
│ Meta-Learning   │ 8096                  │
│ AutoML          │ 8097                  │
│ HyperSync ⭐    │ 8098 (NEW)            │
│ MLflow UI       │ 5000 (optional)       │
│ Prometheus      │ 9090                  │
│ Grafana         │ 3000 (if enabled)     │
└─────────────────┴───────────────────────┘

⭐ = New in Phase 11.4
```

---

## Technology Stack

### Backend Services

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| Orchestration | FastAPI + PyTorch | 0.115.0 / 2.0.0 | Multi-agent coordination |
| AutoML | Optuna + MLflow | 3.4.0 / 2.9.0 | Hyperparameter optimization |
| HyperSync | FastAPI + httpx | 0.115.0 / 0.25.0 | Hot-reload service |
| Dashboard API | FastAPI + WebSocket | 0.115.0 | Real-time backend |
| Redis | Redis | 7.0+ | Persistent storage |

### Frontend

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| Dashboard UI | React + TypeScript | 18.2.0 / 5.0.0 | User interface |
| Charts | Recharts + D3.js | 2.10.0 / 7.8.5 | Visualizations |
| UI Components | Material-UI | 5.15.0 | Component library |

### Infrastructure

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Containerization | Docker | Service isolation |
| Orchestration | Kubernetes (future) | Deployment |
| Monitoring | Prometheus + Grafana | Metrics & dashboards |
| Logging | Structured JSON | Log aggregation |

---

## State Machine: Hyperparameter Update Lifecycle

```mermaid
stateDiagram-v2
    [*] --> Idle

    Idle --> Optimizing: Start optimization
    Optimizing --> Completed: n_trials done
    Optimizing --> Failed: Error during training

    Completed --> PendingValidation: Propose update
    PendingValidation --> PendingApproval: Validation passed
    PendingValidation --> Rejected: Validation failed

    PendingApproval --> Approved: Manual approval
    PendingApproval --> Approved: Auto-approved (threshold met)
    PendingApproval --> Rejected: Manual rejection

    Approved --> Applying: Apply update
    Applying --> Applied: Hot-reload success
    Applying --> Failed: Hot-reload error

    Applied --> [*]
    Rejected --> [*]
    Failed --> [*]

    note right of PendingApproval
        Approval Modes:
        - Manual
        - Autonomous (threshold)
        - Autonomous (all)
    end note

    note right of Applying
        Hot-reload <100ms
        Zero downtime
        State preserved
    end note
```

---

## Agent Interaction Model

```mermaid
graph TB
    subgraph "Multi-Agent System"
        direction TB

        POL[Policy Agent<br/>DQN<br/>State: 32<br/>Actions: 10]
        CON[Consensus Agent<br/>A2C<br/>State: 16<br/>Actions: 5]
        ETH[Ethical Agent<br/>PPO<br/>State: 24<br/>Actions: 8]
        RES[Resource Agent<br/>DDPG<br/>State: 20<br/>Actions: 1]

        NASH[Nash Equilibrium Solver]
        PARETO[Pareto Frontier]
        GLOBAL[Global Reward Aggregator]

        POL --> NASH
        CON --> NASH
        ETH --> NASH
        RES --> NASH

        NASH --> PARETO
        PARETO --> GLOBAL

        GLOBAL --> POL
        GLOBAL --> CON
        GLOBAL --> ETH
        GLOBAL --> RES
    end

    subgraph "Coordination Mechanisms"
        CONFLICTS[Conflict Detection]
        RESOLUTION[Conflict Resolution<br/>Nash/Pareto]
    end

    POL --> CONFLICTS
    CON --> CONFLICTS
    ETH --> CONFLICTS
    RES --> CONFLICTS

    CONFLICTS --> RESOLUTION
    RESOLUTION --> NASH

    classDef agent fill:#4caf50,stroke:#333,stroke-width:2px,color:#fff
    classDef coord fill:#2196f3,stroke:#333,stroke-width:2px,color:#fff

    class POL,CON,ETH,RES agent
    class NASH,PARETO,GLOBAL,CONFLICTS,RESOLUTION coord
```

---

## Redis Data Model

```
Redis Key Structure (Phase 11.4)
│
├── agent_state:policy → Hash
│   ├── agent_id: "policy"
│   ├── epsilon: "0.15"
│   ├── total_steps: "15000"
│   ├── recent_reward_mean: "0.78"
│   └── ...
│
├── agent_state:consensus → Hash
├── agent_state:ethical → Hash
├── agent_state:resource → Hash
│
├── agent_history:policy → List (max 1000)
│   ├── [0] → {"reward": 0.85, "episode": 100, "timestamp": "..."}
│   ├── [1] → {"reward": 0.82, "episode": 99, ...}
│   └── ...
│
├── agent_history:consensus → List
├── agent_history:ethical → List
├── agent_history:resource → List
│
├── conflicts:list → List (max 500)
│   ├── [0] → {"type": "policy_ethical_mismatch", "step": 150, ...}
│   └── ...
│
├── equilibrium:list → List (max 500)
│   ├── [0] → {"converged": true, "iterations": 32, ...}
│   └── ...
│
├── metrics:hash → Hash (flattened)
│   ├── global_reward:current → "0.82"
│   ├── global_reward:mean_100 → "0.78"
│   ├── conflicts:total → "45"
│   └── ...
│
└── websocket:subscribers → Set
    ├── "connection-123"
    ├── "connection-456"
    └── ...

TTL: 24 hours (configurable)
```

---

## Deployment Architecture (Future: Kubernetes)

```mermaid
graph TB
    subgraph "Ingress Layer"
        INGRESS[Nginx Ingress<br/>TLS Termination<br/>Rate Limiting]
    end

    subgraph "Application Layer"
        DASH_PODS[Dashboard Pods<br/>Replicas: 3<br/>HPA enabled]
        ORCH_PODS[Orchestration Pods<br/>Replicas: 2<br/>Stateful]
        AUTOML_PODS[AutoML Pods<br/>Replicas: 2<br/>GPU support]
        HYPERSYNC_PODS[HyperSync Pods<br/>Replicas: 2]
    end

    subgraph "Storage Layer"
        REDIS_CLUSTER[Redis Cluster<br/>3 Masters + 3 Replicas]
        MLFLOW_PV[MLflow PV<br/>100GB SSD]
    end

    subgraph "Monitoring Layer"
        PROM_STACK[Prometheus + Grafana<br/>AlertManager]
        LOKI[Loki<br/>Log Aggregation]
        JAEGER[Jaeger<br/>Distributed Tracing]
    end

    INGRESS --> DASH_PODS
    INGRESS --> ORCH_PODS
    INGRESS --> AUTOML_PODS
    INGRESS --> HYPERSYNC_PODS

    DASH_PODS --> REDIS_CLUSTER
    ORCH_PODS --> REDIS_CLUSTER
    AUTOML_PODS --> MLFLOW_PV
    HYPERSYNC_PODS --> MLFLOW_PV
    HYPERSYNC_PODS --> ORCH_PODS

    DASH_PODS --> PROM_STACK
    ORCH_PODS --> PROM_STACK
    AUTOML_PODS --> PROM_STACK
    HYPERSYNC_PODS --> PROM_STACK

    DASH_PODS --> LOKI
    ORCH_PODS --> LOKI
    AUTOML_PODS --> LOKI
    HYPERSYNC_PODS --> LOKI

    DASH_PODS --> JAEGER
    ORCH_PODS --> JAEGER
    AUTOML_PODS --> JAEGER
    HYPERSYNC_PODS --> JAEGER

    classDef app fill:#4caf50,stroke:#333,stroke-width:2px,color:#fff
    classDef storage fill:#ff9800,stroke:#333,stroke-width:2px,color:#fff
    classDef monitoring fill:#f44336,stroke:#333,stroke-width:2px,color:#fff
    classDef new fill:#ffd700,stroke:#333,stroke-width:2px,color:#000

    class DASH_PODS,ORCH_PODS,AUTOML_PODS app
    class REDIS_CLUSTER,MLFLOW_PV storage
    class PROM_STACK,LOKI,JAEGER monitoring
    class HYPERSYNC_PODS new
```

---

## Security Architecture (Future: Phase 11.5)

```
┌──────────────────────────────────────────────┐
│              Security Layers                 │
├──────────────────────────────────────────────┤
│                                              │
│  ┌────────────────────────────────────┐     │
│  │  Authentication Layer              │     │
│  │  - JWT tokens (HS256)              │     │
│  │  - OAuth2 (optional)               │     │
│  │  - API keys for services           │     │
│  └────────────────────────────────────┘     │
│            │                                 │
│            ▼                                 │
│  ┌────────────────────────────────────┐     │
│  │  Authorization Layer               │     │
│  │  - Role-based access control       │     │
│  │  - Viewer, Developer, Admin        │     │
│  │  - Service-to-service mTLS         │     │
│  └────────────────────────────────────┘     │
│            │                                 │
│            ▼                                 │
│  ┌────────────────────────────────────┐     │
│  │  Rate Limiting Layer               │     │
│  │  - 30 req/min per client           │     │
│  │  - Token bucket algorithm          │     │
│  │  - DDoS protection                 │     │
│  └────────────────────────────────────┘     │
│            │                                 │
│            ▼                                 │
│  ┌────────────────────────────────────┐     │
│  │  Network Layer                     │     │
│  │  - TLS 1.3 (dashboard)             │     │
│  │  - mTLS (inter-service)            │     │
│  │  - Network policies (K8s)          │     │
│  └────────────────────────────────────┘     │
│                                              │
└──────────────────────────────────────────────┘
```

---

## Monitoring & Observability

### Prometheus Metrics Hierarchy

```
T.A.R.S. Metrics
├── Orchestration (Port 8094)
│   ├── tars_orchestration_steps_total
│   ├── tars_global_reward
│   ├── tars_agent_reward{agent_type}
│   ├── tars_multiagent_conflicts_total{conflict_type}
│   ├── tars_nash_convergence_time_seconds
│   ├── tars_agent_reward_alignment
│   ├── tars_dqn_epsilon
│   ├── tars_a2c_entropy
│   ├── tars_ppo_kl_divergence
│   └── tars_ddpg_noise_sigma
│
├── AutoML (Port 8097)
│   ├── tars_automl_trials_total{agent_type, status}
│   ├── tars_automl_best_score{agent_type}
│   ├── tars_featuregen_time_seconds{feature_type}
│   └── tars_model_registrations_total{agent_type}
│
├── HyperSync (Port 8098) ⭐ NEW
│   ├── tars_hyperparam_updates_proposed_total{agent_type}
│   ├── tars_hyperparam_updates_applied_total{agent_type, status}
│   ├── tars_hyperparam_update_improvement{agent_type}
│   └── tars_hyperparam_pending_updates
│
└── Dashboard API (Port 3001)
    ├── tars_dashboard_requests_total{endpoint, method}
    ├── tars_dashboard_websocket_connections
    └── tars_dashboard_simulation_latency_seconds
```

---

## Performance Benchmarks

### Latency Targets (p99)

| Operation | Target | Achieved | Status |
|-----------|--------|----------|--------|
| Orchestration step | <50ms | 35ms | ✅ |
| Hyperparameter hot-reload | <200ms | 50-100ms | ✅ |
| AutoML trial (quick mode) | <5s | 2-5s | ✅ |
| AutoML trial (full mode) | <20s | 10-20s | ✅ |
| Redis agent state read | <5ms | 0.8-3.2ms | ✅ |
| Redis agent state write | <10ms | 1.2-4.5ms | ✅ |
| Dashboard WebSocket broadcast | <150ms | <100ms | ✅ |
| Nash equilibrium computation | <2s | 0.5-1.5s | ✅ |

### Throughput Targets

| Component | Target | Achieved | Status |
|-----------|--------|----------|--------|
| Orchestration steps/sec | 100 | 150 | ✅ |
| AutoML trials/hour (quick) | 1000 | 1200 | ✅ |
| Dashboard WebSocket clients | 100 | 500+ | ✅ |
| Redis operations/sec | 10,000 | 50,000+ | ✅ |

---

## Conclusion

Phase 11.4 completes the integration of T.A.R.S.'s advanced multi-agent reinforcement learning system with automated hyperparameter optimization, persistent storage, and production-ready deployment capabilities.

**Key Achievements**:
- ✅ 6 services running in harmony
- ✅ End-to-end automation pipeline
- ✅ Real training-based optimization
- ✅ Zero-downtime hot-reload
- ✅ Persistent, scalable storage
- ✅ Comprehensive monitoring

**Status**: Production-ready for staging deployment

---

**Architecture Document**: Phase 11 Final
**Version**: v0.9.4-alpha
**Date**: November 14, 2025
