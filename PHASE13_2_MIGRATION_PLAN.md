# Phase 13.2 — Migration Plan

**Version:** 1.0.0
**Status:** Planning Complete
**Date:** 2025-11-19
**Target Release:** v1.0.0-RC2

---

## Table of Contents

1. [Overview](#1-overview)
2. [Pre-Migration Checklist](#2-pre-migration-checklist)
3. [Database Migration](#3-database-migration)
4. [Service Deployment](#4-service-deployment)
5. [Integration Steps](#5-integration-steps)
6. [Rollback Plan](#6-rollback-plan)
7. [Verification](#7-verification)

---

## 1. Overview

### 1.1 Migration Scope

Phase 13.2 introduces the **Evaluation Engine** as a new microservice in the T.A.R.S. ecosystem. This migration involves:

- ✅ **New Service:** eval-engine (Port 8099)
- ✅ **Database Schema:** New table `eval_baselines`
- ✅ **No Breaking Changes:** Existing services unaffected
- ✅ **Backward Compatible:** AutoML continues using mock objective until integration

### 1.2 Migration Timeline

| Phase | Duration | Activities |
|-------|----------|------------|
| **Phase 1: Database Setup** | 30 minutes | Run migration 007, verify schema |
| **Phase 2: Service Deployment** | 1 hour | Deploy eval-engine to K8s, health checks |
| **Phase 3: Smoke Testing** | 30 minutes | Test eval endpoints, verify baselines |
| **Phase 4: Integration** | 2 hours | Connect AutoML → eval-engine (Phase 13.3) |

**Total Estimated Time:** 4 hours

---

## 2. Pre-Migration Checklist

### 2.1 Infrastructure Requirements

- [ ] **PostgreSQL 14+** running and accessible
- [ ] **Redis 7.0+** running and accessible
- [ ] **Kubernetes cluster** with capacity:
  - 2 CPUs available per replica
  - 4GB RAM available per replica
  - Min 2 replicas for HA
- [ ] **Persistent Volume** for checkpoints (optional, Phase 13.3)
- [ ] **Helm 3.8+** installed

### 2.2 Dependencies

- [ ] **Existing Services:**
  - Orchestration Agent (8094) running
  - AutoML Pipeline (8096) running
  - HyperSync Service (8098) running
  - Dashboard API (3001) running

- [ ] **Python Packages:**
  ```bash
  pip install gymnasium==0.29.1 numpy==1.24.3 asyncpg==0.29.0 redis[asyncio]==5.0.1
  ```

### 2.3 Secrets Preparation

- [ ] **PostgreSQL Connection String:**
  ```bash
  kubectl create secret generic tars-secrets \
    --from-literal=postgres-url="postgresql://user:pass@postgres:5432/tars" \
    --dry-run=client -o yaml | kubectl apply -f -
  ```

- [ ] **JWT Secret Key:** (already exists from Phase 11.5)
  ```bash
  kubectl get secret tars-secrets -o jsonpath='{.data.jwt-secret}'
  ```

---

## 3. Database Migration

### 3.1 Migration Script

**File:** `cognition/eval-engine/db/migrations/007_eval_baselines.sql`

**Execution:**
```bash
# Connect to PostgreSQL
psql $POSTGRES_URL

# Run migration
\i cognition/eval-engine/db/migrations/007_eval_baselines.sql

# Verify table created
\dt eval_baselines
\d+ eval_baselines
```

**Expected Output:**
```
                   Table "public.eval_baselines"
     Column      |           Type           | Nullable |        Default
-----------------+--------------------------+----------+-----------------------
 id              | uuid                     | not null | gen_random_uuid()
 agent_type      | character varying(50)    | not null |
 environment     | character varying(100)   | not null |
 mean_reward     | double precision         | not null |
 std_reward      | double precision         | not null |
 success_rate    | double precision         | not null |
 hyperparameters | jsonb                    | not null |
 version         | integer                  | not null |
 rank            | integer                  | not null | 1
 created_at      | timestamp with time zone | not null | now()
 updated_at      | timestamp with time zone | not null | now()
Indexes:
    "eval_baselines_pkey" PRIMARY KEY, btree (id)
    "eval_baselines_agent_env_rank_unique" UNIQUE CONSTRAINT, btree (agent_type, environment, rank)
    "idx_eval_baselines_agent_env" btree (agent_type, environment)
    "idx_eval_baselines_rank" btree (agent_type, environment, rank)
```

### 3.2 Seed Initial Baselines (Optional)

```sql
-- Insert baseline for DQN on CartPole-v1
INSERT INTO eval_baselines (
    agent_type, environment, mean_reward, std_reward, success_rate,
    hyperparameters, version, rank
) VALUES (
    'DQN', 'CartPole-v1', 450.0, 25.0, 0.95,
    '{"learning_rate": 0.001, "gamma": 0.99, "epsilon": 0.1}'::jsonb,
    12, 1
);

-- Insert baseline for A2C on LunarLander-v2
INSERT INTO eval_baselines (
    agent_type, environment, mean_reward, std_reward, success_rate,
    hyperparameters, version, rank
) VALUES (
    'A2C', 'LunarLander-v2', 180.0, 35.0, 0.85,
    '{"learning_rate": 0.0007, "gamma": 0.99, "entropy_coef": 0.01}'::jsonb,
    8, 1
);

-- Verify
SELECT agent_type, environment, mean_reward, version, rank
FROM eval_baselines
ORDER BY agent_type, environment, rank;
```

### 3.3 Rollback Script

**File:** `cognition/eval-engine/db/migrations/007_rollback.sql`

```sql
-- Rollback migration 007
DROP TRIGGER IF EXISTS eval_baselines_updated_at ON eval_baselines;
DROP FUNCTION IF EXISTS update_eval_baselines_updated_at();
DROP TABLE IF EXISTS eval_baselines;
```

---

## 4. Service Deployment

### 4.1 Helm Chart Update

**File:** `charts/tars/values.yaml`

**Add eval-engine configuration:**
```yaml
evalEngine:
  enabled: true
  replicaCount: 2
  image:
    repository: tars/eval-engine
    tag: v1.0.0-rc2
    pullPolicy: IfNotPresent
  resources:
    limits:
      cpu: "2000m"
      memory: "4Gi"
    requests:
      cpu: "1000m"
      memory: "2Gi"
  config:
    defaultEpisodes: 100
    quickModeEpisodes: 50
    maxConcurrent: 4
    envCacheSize: 50
    thresholds:
      failureRate: 0.15
      rewardDropPct: 0.10
      lossTrendWindow: 10
      varianceMultiplier: 2.5
  autoscaling:
    enabled: true
    minReplicas: 2
    maxReplicas: 10
    targetCPUUtilizationPercentage: 70
  service:
    type: ClusterIP
    port: 8099
```

### 4.2 Deploy Service

```bash
# Navigate to charts directory
cd charts/tars

# Lint Helm chart
helm lint .

# Dry-run deployment
helm upgrade --install tars . \
  --namespace tars \
  --create-namespace \
  --dry-run --debug

# Deploy
helm upgrade --install tars . \
  --namespace tars \
  --create-namespace \
  --wait --timeout=5m

# Verify deployment
kubectl get pods -n tars | grep eval-engine
kubectl get svc -n tars | grep eval-engine
```

**Expected Output:**
```
tars-eval-engine-5f7d8c9b6d-abcde   1/1     Running   0          30s
tars-eval-engine-5f7d8c9b6d-fghij   1/1     Running   0          30s

tars-eval-engine   ClusterIP   10.96.100.50   <none>   8099/TCP   30s
```

### 4.3 Health Check Verification

```bash
# Port-forward to eval-engine
kubectl port-forward -n tars svc/tars-eval-engine 8099:8099

# Test health endpoint
curl http://localhost:8099/health

# Expected response:
{
  "status": "healthy",
  "service": "eval-engine",
  "version": "v1.0.0-rc2",
  "postgres": "connected",
  "redis": "connected"
}
```

---

## 5. Integration Steps

### 5.1 Update Service Discovery

**Add eval-engine to Kubernetes DNS:**
```yaml
# charts/tars/templates/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: {{ include "tars.fullname" . }}-services
data:
  ORCHESTRATION_URL: "http://{{ include "tars.fullname" . }}-orchestration:8094"
  AUTOML_URL: "http://{{ include "tars.fullname" . }}-automl:8096"
  HYPERSYNC_URL: "http://{{ include "tars.fullname" . }}-hypersync:8098"
  EVAL_ENGINE_URL: "http://{{ include "tars.fullname" . }}-eval-engine:8099"  # NEW
  DASHBOARD_API_URL: "http://{{ include "tars.fullname" . }}-dashboard-api:3001"
```

### 5.2 Update AutoML Service (Phase 13.3)

**File:** `cognition/automl-pipeline/objective.py`

**Add eval-engine integration:**
```python
import httpx

async def objective_function_real(trial, agent_type: str, env: str):
    """Replace mock objective with eval-engine call."""
    hyperparameters = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
        "gamma": trial.suggest_float("gamma", 0.9, 0.999),
        "epsilon": trial.suggest_float("epsilon", 0.01, 0.5),
    }

    # Call eval-engine
    eval_engine_url = os.getenv("EVAL_ENGINE_URL", "http://localhost:8099")
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{eval_engine_url}/v1/evaluate",
            json={
                "agent_type": agent_type,
                "agent_state": {},  # Load from checkpoint
                "hyperparameters": hyperparameters,
                "environments": [env],
                "num_episodes": 50,  # Quick mode
                "quick_mode": True,
                "compare_to_baseline": True
            },
            headers={"Authorization": f"Bearer {get_service_token()}"},
            timeout=180.0  # 3 minutes
        )
        result = response.json()

    return result["results"][0]["metrics"]["mean_reward"]
```

### 5.3 Update HyperSync Service (Phase 13.3)

**File:** `cognition/hyperparameter-sync/updater.py`

**Add regression detection integration:**
```python
async def apply_proposal(proposal_id: str, agent_type: str, hyperparameters: dict):
    """Apply hyperparameter proposal with regression check."""

    # Evaluate new hyperparameters
    eval_result = await eval_engine_client.evaluate(
        agent_type=agent_type,
        hyperparameters=hyperparameters,
        environments=["CartPole-v1"],
        compare_to_baseline=True
    )

    # Check for regression
    if eval_result["regression"]["should_rollback"]:
        logger.warning(f"Regression detected: {eval_result['regression']['reason']}")
        await reject_proposal(proposal_id, reason="Performance regression detected")
        return False

    # Apply proposal
    await orchestration_client.update_agent_config(agent_type, hyperparameters)
    return True
```

---

## 6. Rollback Plan

### 6.1 Service Rollback

**Scenario:** eval-engine deployment fails or causes issues.

**Steps:**
```bash
# 1. Scale down eval-engine
kubectl scale deployment tars-eval-engine --replicas=0 -n tars

# 2. Remove service from Helm values
# Edit charts/tars/values.yaml: evalEngine.enabled = false

# 3. Redeploy Helm chart
helm upgrade --install tars ./charts/tars --namespace tars

# 4. Verify AutoML still works with mock objective
curl -X POST http://automl:8096/v1/optimize \
  -d '{"agent_type": "DQN", "use_real_training": false}'
```

### 6.2 Database Rollback

**Scenario:** Migration causes data corruption.

**Steps:**
```bash
# 1. Connect to PostgreSQL
psql $POSTGRES_URL

# 2. Run rollback script
\i cognition/eval-engine/db/migrations/007_rollback.sql

# 3. Verify table dropped
\dt eval_baselines
# Expected: Did not find any relation named "eval_baselines"
```

### 6.3 Rollback Verification

- [ ] AutoML service healthy: `curl http://automl:8096/health`
- [ ] No references to eval-engine in logs
- [ ] PostgreSQL table dropped
- [ ] Helm chart deployed without eval-engine

---

## 7. Verification

### 7.1 Functional Tests

**Test 1: Health Check**
```bash
curl http://localhost:8099/health

# Expected: {"status": "healthy"}
```

**Test 2: Evaluate Agent (Synchronous)**
```bash
curl -X POST http://localhost:8099/v1/evaluate \
  -H "Authorization: Bearer $JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "agent_type": "DQN",
    "agent_state": {},
    "hyperparameters": {
      "learning_rate": 0.001,
      "gamma": 0.99,
      "epsilon": 0.1
    },
    "environments": ["CartPole-v1"],
    "num_episodes": 10,
    "quick_mode": true,
    "compare_to_baseline": false
  }'

# Expected: 200 OK with evaluation results
```

**Test 3: Get Baseline**
```bash
curl http://localhost:8099/v1/baselines/DQN?environment=CartPole-v1 \
  -H "Authorization: Bearer $JWT_TOKEN"

# Expected: 200 OK with baseline data
```

**Test 4: Regression Detection**
```bash
# Evaluate with intentionally bad hyperparameters
curl -X POST http://localhost:8099/v1/evaluate \
  -H "Authorization: Bearer $JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "agent_type": "DQN",
    "agent_state": {},
    "hyperparameters": {
      "learning_rate": 0.5,  // Too high
      "gamma": 0.5,  // Too low
      "epsilon": 0.9  // Too high
    },
    "environments": ["CartPole-v1"],
    "num_episodes": 50,
    "compare_to_baseline": true
  }'

# Expected: regression.detected = true
```

### 7.2 Performance Tests

**Benchmark 1: Quick Mode Evaluation**
```bash
time curl -X POST http://localhost:8099/v1/evaluate \
  -H "Authorization: Bearer $JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{...}'  # 50 episodes

# Expected: < 30 seconds
```

**Benchmark 2: Full Evaluation**
```bash
time curl -X POST http://localhost:8099/v1/evaluate \
  -H "Authorization: Bearer $JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{...}'  # 100 episodes

# Expected: < 60 seconds
```

**Benchmark 3: Multi-Environment**
```bash
time curl -X POST http://localhost:8099/v1/evaluate \
  -H "Authorization: Bearer $JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    ...
    "environments": ["CartPole-v1", "LunarLander-v2", "MountainCar-v0"]
  }'

# Expected: < 120 seconds
```

### 7.3 Integration Tests

**Test 1: AutoML → Eval-Engine**
```bash
# Start AutoML study with real training
curl -X POST http://automl:8096/v1/optimize \
  -H "Authorization: Bearer $JWT_TOKEN" \
  -d '{
    "agent_type": "DQN",
    "n_trials": 5,
    "use_real_training": true
  }'

# Check eval-engine logs for evaluation requests
kubectl logs -n tars -l app.kubernetes.io/component=eval-engine --tail=100
```

**Test 2: HyperSync → Eval-Engine**
```bash
# Submit proposal with regression check
curl -X POST http://hypersync:8098/v1/proposals \
  -H "Authorization: Bearer $JWT_TOKEN" \
  -d '{
    "agent_type": "DQN",
    "proposed_hyperparameters": {...},
    "evaluate_before_apply": true
  }'

# Verify eval-engine was called
kubectl logs -n tars -l app.kubernetes.io/component=eval-engine | grep "POST /v1/evaluate"
```

### 7.4 Monitoring Verification

**Prometheus Metrics:**
```bash
# Check eval-engine metrics endpoint
curl http://localhost:8099/metrics | grep tars_eval

# Expected metrics:
# tars_eval_evaluations_total
# tars_eval_regression_detected_total
# tars_eval_duration_seconds
# tars_eval_episodes_total
```

**Grafana Dashboard:**
- Import dashboard: `observability/grafana/dashboards/eval-engine-dashboard.json`
- Verify panels display data after running evaluations

---

## 8. Post-Migration Tasks

### 8.1 Documentation Updates

- [ ] Update [README.md](README.md) with eval-engine service (Port 8099)
- [ ] Update [PHASE13_1_UNIFIED_PIPELINE_DESIGN.md](PHASE13_1_UNIFIED_PIPELINE_DESIGN.md) with "Implemented" status
- [ ] Add eval-engine to [QUICKSTART.md](QUICKSTART.md)

### 8.2 Team Communication

- [ ] Notify team of new service availability
- [ ] Share API documentation: `http://localhost:8099/docs`
- [ ] Update Postman collection with eval-engine endpoints

### 8.3 Monitoring Setup

- [ ] Configure Prometheus alerts for eval-engine:
  - High evaluation latency (>120s)
  - Frequent regressions (>20% of evaluations)
  - Service downtime

- [ ] Add PagerDuty integration for critical alerts

---

## 9. Success Criteria

Migration is considered successful when:

- ✅ eval-engine service deployed with 2 replicas
- ✅ Health checks passing for 10 minutes
- ✅ Database migration completed without errors
- ✅ Baseline CRUD operations working
- ✅ Agent evaluation (DQN, A2C, PPO, DDPG) successful
- ✅ Regression detection functional
- ✅ Integration with AutoML verified (Phase 13.3)
- ✅ Prometheus metrics exported
- ✅ No errors in service logs for 30 minutes

---

**End of Migration Plan**
**Next:** Phase 13.3 Implementation
