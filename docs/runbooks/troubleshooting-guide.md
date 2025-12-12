# Troubleshooting Guide - T.A.R.S. Evaluation Engine

**Version:** v1.0.0-rc2
**Last Updated:** 2025-11-19

---

## Table of Contents

1. [High Evaluation Latency](#high-evaluation-latency)
2. [Evaluation Failures](#evaluation-failures)
3. [PostgreSQL Issues](#postgresql-issues)
4. [Redis Issues](#redis-issues)
5. [Worker Hangs](#worker-hangs)
6. [HPA Not Scaling](#hpa-not-scaling)
7. [Network Partition](#network-partition)
8. [JWT Rotation Issues](#jwt-rotation-issues)
9. [Slow Episodes](#slow-episodes)
10. [Missing Metrics in Grafana](#missing-metrics-in-grafana)

---

## High Evaluation Latency

**Symptom:** p95 latency > 300s

**Alert:** `HighEvaluationLatency`

### Diagnosis

```bash
# Check current latency
kubectl logs -n tars deployment/tars-eval-engine | grep "evaluation_duration"

# Check CPU usage
kubectl top pods -n tars | grep eval-engine

# Check worker pool size
curl http://localhost:8099/metrics | grep tars_eval_worker_pool_size
```

### Common Causes

#### 1. CPU Contention
**Symptoms:**
- Pod CPU > 90% of limit
- Slow episode execution
- High worker queue depth

**Solution:**
```bash
# Increase CPU limits
kubectl edit deployment tars-eval-engine -n tars

# Update resources:
spec:
  template:
    spec:
      containers:
      - name: eval-engine
        resources:
          requests:
            cpu: "4"
          limits:
            cpu: "8"
```

#### 2. Too Many Episodes
**Symptoms:**
- Long-running evaluations (>10 minutes)
- `num_episodes` > 100

**Solution:**
- Reduce `num_episodes` to 50 for quick mode
- Use `EVAL_QUICK_MODE_EPISODES=50` env var

#### 3. Slow Environments
**Symptoms:**
- Certain environments take >1s per episode
- Environments: MuJoCo, Atari

**Solution:**
- Cache compiled environments in Redis
- Pre-warm environment cache:
```python
async def prewarm_cache(env_names):
    for env in env_names:
        await env_cache.get_or_create(env)
```

---

## Evaluation Failures

**Symptom:** Failure rate > 5%

**Alert:** `EvaluationFailureRateHigh`

### Diagnosis

```bash
# Check failure rate
curl http://localhost:8099/metrics | grep 'tars_eval_evaluations_total{status="failed"}'

# Get recent errors
kubectl logs -n tars deployment/tars-eval-engine --tail=100 | grep ERROR
```

### Common Causes

#### 1. Invalid Hyperparameters
**Error:** `400 Bad Request - Validation error`

**Solution:**
```bash
# Check hyperparameters against schema
# Example: learning_rate must be > 0
{
  "hyperparameters": {
    "learning_rate": 0.001,  # Must be positive
    "gamma": 0.99,           # Must be in (0, 1)
    "epsilon": 0.1           # Must be in [0, 1]
  }
}
```

#### 2. Environment Not Found
**Error:** `EnvironmentNotFoundError: 'UnknownEnv-v1' not found`

**Solution:**
```bash
# List available environments
python -c "import gymnasium as gym; print(gym.envs.registry.keys())"

# Common typos:
# - "CartPole-v1" (correct) vs "CartPole-V1" (wrong)
# - "Acrobot-v1" vs "Acrobat-v1"
```

#### 3. Worker Timeout
**Error:** `EvaluationTimeoutError: Evaluation exceeded 600s timeout`

**Solution:**
```bash
# Increase timeout
export EVAL_WORKER_TIMEOUT=1200  # 20 minutes

# Or reduce episodes
export EVAL_DEFAULT_EPISODES=50
```

---

## PostgreSQL Issues

### Connection Pool Exhausted

**Alert:** `PostgreSQLConnectionPoolExhausted`

**Diagnosis:**
```bash
# Check active connections
psql $POSTGRES_URL -c "SELECT count(*) FROM pg_stat_activity WHERE datname='tars';"

# Check max connections
psql $POSTGRES_URL -c "SHOW max_connections;"
```

**Solution:**
```python
# Increase pool size in main.py
db_pool = await asyncpg.create_pool(
    config.postgres_url,
    min_size=10,    # Increased from 5
    max_size=40,    # Increased from 20
    command_timeout=60
)
```

### Slow Queries

**Alert:** `PostgreSQLSlowQueries`

**Diagnosis:**
```sql
-- Find slow queries
SELECT query, mean_exec_time, calls
FROM pg_stat_statements
WHERE mean_exec_time > 500  -- > 500ms
ORDER BY mean_exec_time DESC
LIMIT 10;

-- Explain slow query
EXPLAIN ANALYZE
SELECT * FROM eval_baselines
WHERE agent_type = 'dqn' AND environment = 'CartPole-v1' AND rank = 1;
```

**Solution:**
```sql
-- Add missing index
CREATE INDEX idx_eval_baselines_agent_env_rank
ON eval_baselines(agent_type, environment, rank);

-- Vacuum analyze
VACUUM ANALYZE eval_baselines;
```

---

## Redis Issues

### Redis Down

**Alert:** `RedisConnectionFailures`

**Impact:**
- Environment cache disabled
- Rate limiting degraded
- Increased latency

**Diagnosis:**
```bash
# Check Redis health
kubectl get pods -n tars | grep redis

# Check Redis logs
kubectl logs -n tars redis-0

# Test connection
redis-cli -h redis.tars.svc.cluster.local ping
```

**Solution:**
```bash
# Restart Redis (if pod crashed)
kubectl delete pod redis-0 -n tars

# Check Redis configuration
kubectl describe statefulset redis -n tars

# Increase memory (if OOM)
kubectl edit statefulset redis -n tars
# Update: resources.limits.memory: "2Gi"
```

### Redis Memory High

**Alert:** `RedisMemoryHigh`

**Diagnosis:**
```bash
# Check memory usage
redis-cli INFO memory

# Check cache size
redis-cli DBSIZE

# Check TTLs
redis-cli --scan --pattern "env:*" | xargs redis-cli TTL
```

**Solution:**
```bash
# Set eviction policy
redis-cli CONFIG SET maxmemory-policy allkeys-lru

# Reduce cache TTL
redis-cli CONFIG SET maxmemory 2gb

# Clear cache (last resort)
redis-cli FLUSHDB
```

---

## Worker Hangs

**Symptom:** Evaluation stuck, no progress for >5 minutes

**Diagnosis:**
```bash
# Check worker status
kubectl logs -n tars deployment/tars-eval-engine | grep "worker"

# Check for deadlocks
kubectl exec -it deployment/tars-eval-engine -n tars -- python -c "
import psutil
for proc in psutil.process_iter(['pid', 'name', 'status']):
    if proc.info['status'] == 'zombie':
        print(f'Zombie process: {proc.info}')
"
```

**Solution:**
```bash
# Kill stuck evaluation (if job_id known)
curl -X DELETE http://localhost:8099/v1/jobs/{job_id} \
  -H "Authorization: Bearer $ACCESS_TOKEN"

# Restart pod
kubectl delete pod -n tars -l app=tars-eval-engine

# Increase worker timeout
export EVAL_WORKER_TIMEOUT=1200
```

---

## HPA Not Scaling

**Symptom:** HPA stuck at min replicas despite high CPU

**Diagnosis:**
```bash
# Check HPA status
kubectl get hpa -n tars tars-eval-engine

# Check current CPU
kubectl top pods -n tars | grep eval-engine

# Check HPA events
kubectl describe hpa tars-eval-engine -n tars
```

**Common Issues:**

#### 1. Metrics Server Not Running
```bash
kubectl get deployment metrics-server -n kube-system

# If missing, install:
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
```

#### 2. Resource Requests Not Set
```yaml
# HPA requires resource requests
spec:
  containers:
  - name: eval-engine
    resources:
      requests:
        cpu: "2"      # Required!
        memory: "4Gi"
```

#### 3. HPA Misconfigured
```bash
# Check HPA config
kubectl get hpa tars-eval-engine -n tars -o yaml

# Update target CPU
kubectl patch hpa tars-eval-engine -n tars -p '{"spec":{"targetCPUUtilizationPercentage":70}}'
```

---

## Network Partition

**Symptom:** Service unreachable from other pods

**Diagnosis:**
```bash
# Check service endpoints
kubectl get endpoints -n tars tars-eval-engine

# Test connectivity
kubectl run -it --rm debug --image=curlimages/curl --restart=Never -- \
  curl http://tars-eval-engine.tars.svc.cluster.local:8099/health

# Check network policies
kubectl get networkpolicies -n tars
```

**Solution:**
```bash
# Check DNS resolution
kubectl run -it --rm debug --image=busybox --restart=Never -- \
  nslookup tars-eval-engine.tars.svc.cluster.local

# Check firewall rules (GKE)
gcloud compute firewall-rules list --filter="name~tars"

# Verify service selector
kubectl get svc tars-eval-engine -n tars -o yaml
kubectl get pods -n tars --show-labels | grep eval-engine
```

---

## JWT Rotation Issues

**Symptom:** `401 Unauthorized` after JWT secret rotation

**Diagnosis:**
```bash
# Check JWT secret
kubectl get secret tars-secrets -n tars -o jsonpath='{.data.jwt-secret}' | base64 -d

# Check token expiry
echo $ACCESS_TOKEN | cut -d. -f2 | base64 -d | jq '.exp'
```

**Solution:**
```bash
# Refresh token
curl -X POST http://localhost:8099/auth/refresh \
  -H "Content-Type: application/json" \
  -d '{"refresh_token": "'$REFRESH_TOKEN'"}'

# Update secret (if rotated)
kubectl create secret generic tars-secrets \
  --from-literal=jwt-secret="new-secret-key" \
  --dry-run=client -o yaml | kubectl apply -f -

# Restart pods to pick up new secret
kubectl rollout restart deployment tars-eval-engine -n tars
```

---

## Slow Episodes

**Symptom:** Episodes take >1s each

**Diagnosis:**
```python
# Add timing to worker
import time
start = time.time()
for ep in range(num_episodes):
    # Run episode
    print(f"Episode {ep} took {time.time() - start:.2f}s")
    start = time.time()
```

**Common Causes:**

#### 1. Render Enabled
```python
# Disable rendering
env = gym.make("CartPole-v1", render_mode=None)  # Not "human"
```

#### 2. Slow Agent Inference
```python
# Profile agent.predict()
import cProfile
cProfile.runctx('agent.predict(obs)', globals(), locals())
```

#### 3. Environment Overhead
```python
# Use vectorized environments
from gymnasium.vector import make_vec_env
env = make_vec_env("CartPole-v1", n_envs=4)
```

---

## Missing Metrics in Grafana

**Symptom:** Grafana panels show "No data"

**Diagnosis:**
```bash
# Check Prometheus targets
curl http://prometheus:9090/api/v1/targets | jq '.data.activeTargets[] | select(.labels.job=="eval-engine")'

# Check metrics endpoint
curl http://localhost:8099/metrics | grep tars_eval

# Check Prometheus scrape config
kubectl get configmap prometheus-server -n monitoring -o yaml
```

**Solution:**

#### 1. ServiceMonitor Not Applied
```bash
kubectl apply -f charts/tars/templates/eval-engine-servicemonitor.yaml
```

#### 2. Wrong Prometheus Namespace
```yaml
# Ensure ServiceMonitor has correct label
metadata:
  labels:
    prometheus: kube-prometheus  # Must match Prometheus selector
```

#### 3. Metrics Not Exported
```python
# Ensure metrics are incremented
EVALUATIONS_TOTAL.labels(agent_type="dqn", environment="CartPole-v1", status="success").inc()
```

---

## Quick Reference Commands

```bash
# Health check
curl http://localhost:8099/health

# Metrics
curl http://localhost:8099/metrics

# Logs
kubectl logs -n tars deployment/tars-eval-engine -f

# Restart
kubectl rollout restart deployment tars-eval-engine -n tars

# Scale manually
kubectl scale deployment tars-eval-engine -n tars --replicas=5

# Port forward
kubectl port-forward -n tars svc/tars-eval-engine 8099:8099
```

---

## Related Documents

- [Evaluation Pipeline Runbook](evaluation-pipeline-runbook.md)
- [On-Call Playbook](oncall-playbook.md)
