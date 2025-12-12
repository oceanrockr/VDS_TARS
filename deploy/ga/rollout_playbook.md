# T.A.R.S. v1.0.0 GA - Zero-Downtime Rollout Playbook

**Version:** 1.0.0
**Last Updated:** 2025-11-20
**Owner:** SRE Team
**Approval Required:** CTO, VP Engineering, Security Lead

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Preflight Checklist](#preflight-checklist)
4. [Rollout Procedure](#rollout-procedure)
5. [Canary SLO Acceptance Algorithm](#canary-slo-acceptance-algorithm)
6. [Auto-Rollback Criteria](#auto-rollback-criteria)
7. [Manual Rollback](#manual-rollback)
8. [Post-Deployment Validation](#post-deployment-validation)
9. [Troubleshooting](#troubleshooting)
10. [Communication Plan](#communication-plan)
11. [Sign-Off](#sign-off)

---

## Overview

This playbook provides step-by-step instructions for deploying T.A.R.S. v1.0.0 to production using a **progressive canary rollout** strategy with **zero downtime**.

### Deployment Strategy

- **Blue-Green Foundation:** Maintain v0.3.0-alpha (blue) while deploying v1.0.0 (green)
- **Canary Progression:** 5% → 25% → 50% → 100%
- **Step Duration:** 10 minutes per step
- **SLO Validation:** At each step, validate against SLO targets
- **Auto-Rollback:** Trigger if SLO violation detected

### Key Metrics

| **Metric**                  | **Target**     | **Violation Threshold** |
|-----------------------------|----------------|-------------------------|
| API Response Time (p95)     | <50ms          | >100ms                  |
| Evaluation Latency (p95)    | <300s          | >600s                   |
| Error Rate                  | <1%            | >5%                     |
| Success Rate                | >95%           | <90%                    |
| Regression Detection F1     | >85%           | <75%                    |
| Hot-Reload Latency (p95)    | <100ms         | >200ms                  |

### Timeline

- **Total Duration:** ~50 minutes (optimal path)
- **Rollback Time:** <5 minutes (if needed)

---

## Prerequisites

### Infrastructure

- [x] Kubernetes cluster v1.28+ operational
- [x] ArgoCD v2.9+ installed and configured
- [x] Prometheus Operator deployed
- [x] Grafana dashboards imported
- [x] Jaeger tracing operational
- [x] cert-manager configured with Let's Encrypt
- [x] Ingress controller (nginx) ready
- [x] Multi-region setup (us-east-1, us-west-2, eu-central-1)

### Data Stores

- [x] PostgreSQL primary + 2 replicas healthy
- [x] PostgreSQL replication lag <3s (p95)
- [x] Redis Sentinel cluster (3 nodes) operational
- [x] Redis persistence (RDB + AOF) enabled
- [x] Database backup completed within last 24 hours
- [x] Database migration scripts validated (dry-run)

### Secrets & Configuration

- [x] JWT secret rotated within last 7 days
- [x] Database credentials rotated within last 30 days
- [x] TLS certificates valid for 90+ days
- [x] mTLS CA + client certs deployed
- [x] Statuspage API key configured
- [x] PagerDuty integration tokens verified
- [x] ArgoCD git credentials valid

### Monitoring & Alerting

- [x] Prometheus scraping all services
- [x] Grafana dashboards accessible
- [x] PagerDuty on-call rotation active
- [x] Statuspage components mapped
- [x] Slack notifications configured
- [x] Alert rules validated (40+ rules)

### Security

- [x] External pentest completed (0 HIGH/CRITICAL findings)
- [x] CVE scan passed (0 HIGH/CRITICAL CVEs)
- [x] RBAC policies validated
- [x] Rate limiting tested
- [x] Secret rotation policies enforced

### Documentation

- [x] Runbook reviewed and updated
- [x] Troubleshooting guide available
- [x] On-call playbook accessible
- [x] Release notes published
- [x] Migration guide reviewed

### Team Readiness

- [x] On-call engineer identified and notified
- [x] Backup on-call engineer available
- [x] CTO approval obtained
- [x] Customer success team briefed
- [x] Marketing team ready (blog post, press release)

---

## Preflight Checklist

**Run these commands 1 hour before deployment:**

### 1. Verify Cluster Health

```bash
# Check node status
kubectl get nodes
# Expected: All nodes Ready

# Check namespace
kubectl get namespace tars-production
# Expected: Active

# Check ArgoCD status
argocd version
argocd app list
# Expected: ArgoCD operational
```

### 2. Validate Current Production State

```bash
# Check current deployment
kubectl get deployments -n tars-production
# Expected: All deployments ready (v0.3.0-alpha)

# Check pod health
kubectl get pods -n tars-production
# Expected: All pods Running (0 restarts in last hour)

# Check service endpoints
kubectl get endpoints -n tars-production
# Expected: All endpoints populated
```

### 3. Database Validation

```bash
# Check PostgreSQL replication lag
kubectl exec -n tars-data postgres-primary-0 -- \
  psql -U tars -d tars_production -c \
  "SELECT client_addr, state, sync_state, replay_lag FROM pg_stat_replication;"
# Expected: replay_lag <3s

# Verify database backup
kubectl get cronjob -n tars-data postgres-backup
kubectl get jobs -n tars-data | grep postgres-backup
# Expected: Successful backup in last 24 hours

# Test database connectivity
kubectl exec -n tars-production $(kubectl get pod -n tars-production -l app=eval-engine -o jsonpath='{.items[0].metadata.name}') -- \
  python -c "import psycopg2; conn = psycopg2.connect(host='postgres-primary.tars-data.svc.cluster.local', user='tars', password='***', database='tars_production'); print('OK')"
# Expected: OK
```

### 4. Redis Validation

```bash
# Check Redis Sentinel
kubectl exec -n tars-data redis-master-0 -- redis-cli SENTINEL masters
# Expected: All sentinels agree on master

# Check Redis persistence
kubectl exec -n tars-data redis-master-0 -- redis-cli INFO persistence
# Expected: rdb_last_save_time within 15 minutes, aof_enabled:1

# Test Redis connectivity
kubectl exec -n tars-production $(kubectl get pod -n tars-production -l app=eval-engine -o jsonpath='{.items[0].metadata.name}') -- \
  python -c "import redis; r = redis.Redis(host='redis-master.tars-data.svc.cluster.local', port=6379, password='***'); r.ping(); print('OK')"
# Expected: OK
```

### 5. Prometheus & Grafana Validation

```bash
# Check Prometheus targets
kubectl port-forward -n observability svc/prometheus-kube-prometheus-prometheus 9090:9090 &
curl http://localhost:9090/api/v1/targets | jq '.data.activeTargets[] | select(.health != "up")'
# Expected: Empty (all targets up)

# Check Grafana dashboards
kubectl port-forward -n observability svc/grafana 3000:80 &
curl -u admin:*** http://localhost:3000/api/health
# Expected: {"database":"ok","version":"..."}
```

### 6. ArgoCD Application Preparation

```bash
# Create ArgoCD project (if not exists)
kubectl apply -f deploy/ga/argo_application.yaml

# Verify ArgoCD application
argocd app get tars-v1-ga
# Expected: Status: OutOfSync (not yet synced)

# Validate Helm values
helm template tars charts/tars \
  -f charts/tars/values.yaml \
  -f deploy/ga/helm_values_ga.yaml \
  --namespace tars-production | kubectl apply --dry-run=client -f -
# Expected: No errors
```

### 7. Baseline Metrics Snapshot

```bash
# Capture current metrics for comparison
kubectl exec -n observability prometheus-kube-prometheus-prometheus-0 -- \
  promtool query instant http://localhost:9090 \
  'rate(http_requests_total{namespace="tars-production"}[5m])'

# Save baseline to file
make capture-baseline-metrics
# Expected: Baseline saved to /tmp/tars_baseline_metrics.json
```

### 8. Statuspage Preparation

```bash
# Update Statuspage (scheduled maintenance)
python canary/statuspage_client.py create-incident \
  --name "T.A.R.S. v1.0.0 GA Deployment" \
  --status "scheduled" \
  --impact "none" \
  --scheduled-for "2025-11-20T10:00:00Z" \
  --scheduled-until "2025-11-20T11:00:00Z" \
  --component-ids "eval-engine,hypersync,rl-agents,dashboard,api" \
  --message "We are deploying T.A.R.S. v1.0.0 GA using a progressive canary rollout. No downtime expected."
# Expected: Incident ID returned
```

### 9. Communication

```bash
# Slack notification (pre-deployment)
curl -X POST https://hooks.slack.com/services/T00000000/B00000000/XXXXXXXXXXXXXXXXXXXX \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "🚀 T.A.R.S. v1.0.0 GA deployment starting in 5 minutes. Canary rollout: 5% → 25% → 50% → 100%. ETA: 50 minutes. Status: https://status.tars.prod"
  }'
```

---

## Rollout Procedure

### Phase 1: Database Migrations (Sync Wave 0)

**Duration:** ~2 minutes

1. **Apply database migrations**

   ```bash
   # Trigger ArgoCD sync (migrations only)
   argocd app sync tars-v1-ga --resource argoproj.io:Job:tars-production/tars-migrations --prune=false

   # Monitor migration job
   kubectl get jobs -n tars-production tars-migrations -w
   # Expected: STATUS = Complete (within 2 minutes)

   # Check logs
   kubectl logs -n tars-production job/tars-migrations
   # Expected: "All migrations applied successfully"
   ```

2. **Validate schema changes**

   ```bash
   # Verify new tables/columns
   kubectl exec -n tars-data postgres-primary-0 -- \
     psql -U tars -d tars_production -c "\dt"
   # Expected: New tables present (eval_results_v1, hypersync_proposals_v1)

   # Check migration version
   kubectl exec -n tars-data postgres-primary-0 -- \
     psql -U tars -d tars_production -c "SELECT version FROM schema_migrations ORDER BY version DESC LIMIT 1;"
   # Expected: 20251120_v1_0_0
   ```

3. **Replication validation**

   ```bash
   # Ensure replicas have new schema
   kubectl exec -n tars-data postgres-replica-0 -- \
     psql -U tars -d tars_production -c "SELECT version FROM schema_migrations ORDER BY version DESC LIMIT 1;"
   # Expected: 20251120_v1_0_0
   ```

**Rollback Point:** If migrations fail, rollback immediately (see [Manual Rollback](#manual-rollback))

---

### Phase 2: Backend Services - 5% Canary (Sync Wave 1)

**Duration:** ~10 minutes

1. **Deploy 5% canary**

   ```bash
   # Update Helm values (canary 5%)
   argocd app set tars-v1-ga --helm-set canary.steps[0]=5
   argocd app sync tars-v1-ga --resource apps:Deployment:tars-production/tars-eval-engine --prune=false

   # Verify canary pods
   kubectl get pods -n tars-production -l app=eval-engine,version=v1.0.0
   # Expected: 1 pod (5% of 3 replicas = 0.15 → 1 pod)

   # Verify traffic split
   kubectl get virtualservice -n tars-production tars-eval-engine -o yaml | grep -A 5 "weight"
   # Expected: weight: 5 (canary), weight: 95 (stable)
   ```

2. **Wait 10 minutes for metric collection**

   ```bash
   # Monitor canary metrics
   watch -n 10 'kubectl exec -n observability prometheus-kube-prometheus-prometheus-0 -- \
     promtool query instant http://localhost:9090 \
     "rate(http_requests_total{namespace=\"tars-production\",version=\"v1.0.0\"}[5m])"'
   ```

3. **SLO validation** (see [Canary SLO Acceptance Algorithm](#canary-slo-acceptance-algorithm))

   ```bash
   # Run SLO validation script
   python canary/test_canary_metric_slo_guardrails.py --stage 5 --duration 10m

   # Expected output:
   # ✅ API Response Time (p95): 42ms (target: <50ms)
   # ✅ Error Rate: 0.3% (target: <1%)
   # ✅ Success Rate: 99.1% (target: >95%)
   # ✅ Regression Detection: No regressions detected
   # ✅ SLO PASSED - Proceeding to 25%
   ```

**Auto-Rollback:** If SLO fails, see [Auto-Rollback Criteria](#auto-rollback-criteria)

---

### Phase 3: Backend Services - 25% Canary

**Duration:** ~10 minutes

1. **Scale to 25% canary**

   ```bash
   argocd app set tars-v1-ga --helm-set canary.steps[1]=25
   argocd app sync tars-v1-ga --resource apps:Deployment:tars-production/tars-eval-engine

   # Verify canary pods
   kubectl get pods -n tars-production -l app=eval-engine,version=v1.0.0
   # Expected: 1 pod (25% of 3 replicas = 0.75 → 1 pod)
   ```

2. **Wait 10 minutes + SLO validation**

   ```bash
   python canary/test_canary_metric_slo_guardrails.py --stage 25 --duration 10m
   # Expected: ✅ SLO PASSED
   ```

---

### Phase 4: Backend Services - 50% Canary

**Duration:** ~10 minutes

1. **Scale to 50% canary**

   ```bash
   argocd app set tars-v1-ga --helm-set canary.steps[2]=50
   argocd app sync tars-v1-ga --resource apps:Deployment:tars-production/tars-eval-engine

   # Verify canary pods
   kubectl get pods -n tars-production -l app=eval-engine,version=v1.0.0
   # Expected: 2 pods (50% of 3 replicas = 1.5 → 2 pods)
   ```

2. **Wait 10 minutes + SLO validation**

   ```bash
   python canary/test_canary_metric_slo_guardrails.py --stage 50 --duration 10m
   # Expected: ✅ SLO PASSED
   ```

---

### Phase 5: Full Rollout - 100%

**Duration:** ~5 minutes

1. **Promote to 100%**

   ```bash
   argocd app set tars-v1-ga --helm-set canary.steps[3]=100
   argocd app sync tars-v1-ga --prune

   # Verify all pods v1.0.0
   kubectl get pods -n tars-production -l app=eval-engine -o jsonpath='{.items[*].spec.containers[0].image}'
   # Expected: ghcr.io/veleron-dev-studios/tars-eval-engine:v1.0.0 (all pods)
   ```

2. **Prune old stable deployment**

   ```bash
   # ArgoCD will auto-prune v0.3.0-alpha pods
   kubectl get pods -n tars-production -l version=v0.3.0-alpha
   # Expected: No resources found
   ```

3. **Final SLO validation**

   ```bash
   python canary/test_canary_metric_slo_guardrails.py --stage 100 --duration 5m
   # Expected: ✅ SLO PASSED
   ```

---

### Phase 6: RL Agents Rollout (Sync Wave 2)

**Duration:** ~10 minutes

1. **Deploy all RL agents**

   ```bash
   # Sync DQN, A2C, PPO, DDPG agents
   argocd app sync tars-v1-ga \
     --resource apps:Deployment:tars-production/tars-agent-dqn \
     --resource apps:Deployment:tars-production/tars-agent-a2c \
     --resource apps:Deployment:tars-production/tars-agent-ppo \
     --resource apps:Deployment:tars-production/tars-agent-ddpg

   # Verify all agents running
   kubectl get pods -n tars-production -l component=agent
   # Expected: 8 pods (2 replicas × 4 agents)
   ```

2. **Agent health check**

   ```bash
   # Check agent /health endpoints
   for agent in dqn a2c ppo ddpg; do
     kubectl exec -n tars-production tars-agent-$agent-0 -- curl http://localhost:8100/health
   done
   # Expected: {"status":"healthy"} (all agents)
   ```

---

### Phase 7: Dashboard Rollout (Sync Wave 3)

**Duration:** ~5 minutes

1. **Deploy Dashboard API + Frontend**

   ```bash
   argocd app sync tars-v1-ga \
     --resource apps:Deployment:tars-production/tars-dashboard-api \
     --resource apps:Deployment:tars-production/tars-dashboard-frontend \
     --resource networking.k8s.io:Ingress:tars-production/tars-ingress

   # Verify dashboard pods
   kubectl get pods -n tars-production -l app=dashboard-api
   kubectl get pods -n tars-production -l app=dashboard-frontend
   # Expected: 3 pods each (Running)
   ```

2. **Ingress validation**

   ```bash
   # Check ingress status
   kubectl get ingress -n tars-production tars-ingress
   # Expected: ADDRESS populated, HOSTS = api.tars.prod,dashboard.tars.prod

   # Test external access
   curl -I https://api.tars.prod/health
   curl -I https://dashboard.tars.prod
   # Expected: HTTP/2 200
   ```

---

## Canary SLO Acceptance Algorithm

**Executed at each canary step (5%, 25%, 50%, 100%)**

### Algorithm

```python
def validate_canary_slo(stage: int, duration: str) -> bool:
    """
    Validate canary SLO compliance.

    Args:
        stage: Canary percentage (5, 25, 50, 100)
        duration: Observation window (e.g., "10m")

    Returns:
        True if all SLOs pass, False otherwise
    """
    metrics = fetch_prometheus_metrics(stage, duration)

    # SLO checks
    checks = {
        "api_response_time_p95": metrics["api_latency_p95"] < 50,  # <50ms
        "error_rate": metrics["error_rate"] < 0.01,  # <1%
        "success_rate": metrics["success_rate"] > 0.95,  # >95%
        "regression_detected": not metrics["regression_detected"],
        "eval_latency_p95": metrics["eval_latency_p95"] < 300,  # <300s
        "hot_reload_latency_p95": metrics["hot_reload_latency_p95"] < 0.1,  # <100ms
    }

    # Log results
    for check, passed in checks.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {check} = {metrics[check]}")

    # All checks must pass
    return all(checks.values())
```

### Prometheus Queries

```promql
# API Response Time (p95)
histogram_quantile(0.95,
  rate(http_request_duration_seconds_bucket{namespace="tars-production",version="v1.0.0"}[5m])
)

# Error Rate
sum(rate(http_requests_total{namespace="tars-production",version="v1.0.0",status=~"5.."}[5m]))
/
sum(rate(http_requests_total{namespace="tars-production",version="v1.0.0"}[5m]))

# Success Rate
sum(rate(http_requests_total{namespace="tars-production",version="v1.0.0",status=~"2.."}[5m]))
/
sum(rate(http_requests_total{namespace="tars-production",version="v1.0.0"}[5m]))

# Regression Detection
regression_detected{namespace="tars-production",version="v1.0.0"} > 0

# Evaluation Latency (p95)
histogram_quantile(0.95,
  rate(evaluation_duration_seconds_bucket{namespace="tars-production",version="v1.0.0"}[5m])
)

# Hot-Reload Latency (p95)
histogram_quantile(0.95,
  rate(hot_reload_duration_seconds_bucket{namespace="tars-production",version="v1.0.0"}[5m])
)
```

---

## Auto-Rollback Criteria

**Automatic rollback triggers if ANY of the following conditions are met:**

### 1. SLO Violation

- API Response Time (p95) > 100ms (target: <50ms)
- Error Rate > 5% (target: <1%)
- Success Rate < 90% (target: >95%)
- Evaluation Latency (p95) > 600s (target: <300s)
- Hot-Reload Latency (p95) > 200ms (target: <100ms)

### 2. Regression Detection

- Regression Detector flags ≥10% performance degradation
- F1 score < 75% (target: >85%)

### 3. Health Probe Failure

- Liveness probe fails 3+ consecutive times
- Readiness probe fails 5+ consecutive times
- Pod restart count > 3 in 10 minutes

### 4. Resource Exhaustion

- CPU utilization > 95% sustained for 5 minutes
- Memory utilization > 90% sustained for 5 minutes
- OOMKilled events detected

### 5. Database Issues

- PostgreSQL replication lag > 10s
- Redis connection errors > 10% of requests
- Database connection pool exhaustion

### 6. External Dependencies

- Prometheus scrape failures > 5%
- Jaeger trace export failures > 10%
- Statuspage update failures (manual override)

### Rollback Execution

```bash
# Automatic rollback (triggered by canary test script)
python canary/test_canary_auto_rollback.py --execute

# Expected output:
# 🚨 SLO VIOLATION DETECTED: error_rate = 6.2% (threshold: 5%)
# 🔄 Initiating auto-rollback to v0.3.0-alpha...
# ✅ Rollback complete in 3m 42s
```

**Manual intervention required if auto-rollback fails** (see [Manual Rollback](#manual-rollback))

---

## Manual Rollback

**Use this procedure if auto-rollback fails or manual rollback is required**

### Rollback Decision

**Rollback immediately if:**
- Customer-facing errors reported
- Data integrity concerns
- Security vulnerability discovered
- Auto-rollback failed

### Rollback Procedure

1. **Pause ArgoCD auto-sync**

   ```bash
   argocd app set tars-v1-ga --sync-policy=none
   ```

2. **Revert to previous git revision**

   ```bash
   # Find previous successful revision
   argocd app history tars-v1-ga

   # Rollback to revision (e.g., revision 42)
   argocd app rollback tars-v1-ga 42

   # Alternative: Rollback to previous stable tag
   argocd app set tars-v1-ga --revision v0.3.0-alpha
   argocd app sync tars-v1-ga --prune --force
   ```

3. **Verify rollback**

   ```bash
   # Check pod image tags
   kubectl get pods -n tars-production -o jsonpath='{.items[*].spec.containers[0].image}' | tr ' ' '\n' | sort -u
   # Expected: ghcr.io/veleron-dev-studios/tars-*:v0.3.0-alpha

   # Check service health
   kubectl get pods -n tars-production
   # Expected: All pods Running
   ```

4. **Database rollback (if needed)**

   ```bash
   # Rollback migrations (CAUTION: Data loss possible)
   kubectl exec -n tars-data postgres-primary-0 -- \
     psql -U tars -d tars_production -c \
     "SELECT rollback_migration('20251120_v1_0_0');"
   # ⚠️ Only if migrations caused issue AND no data written to new schema
   ```

5. **Restore Redis state (if needed)**

   ```bash
   # Restore Redis from backup (if HyperSync state corrupted)
   kubectl exec -n tars-data redis-master-0 -- \
     redis-cli --rdb /data/redis-backup-pre-ga.rdb RESTORE
   ```

6. **Update Statuspage**

   ```bash
   python canary/statuspage_client.py update-incident \
     --incident-id <INCIDENT_ID> \
     --status "investigating" \
     --message "v1.0.0 GA deployment rolled back due to [REASON]. Investigating root cause. ETA: [TIME]"
   ```

7. **Post-rollback validation**

   ```bash
   # Run smoke tests
   make test-canary-smoke

   # Check metrics
   python canary/test_canary_metric_slo_guardrails.py --stage 100 --duration 5m
   # Expected: ✅ SLO PASSED (v0.3.0-alpha)
   ```

**RTO (Recovery Time Objective):** <5 minutes
**RPO (Recovery Point Objective):** <1 minute (data loss minimal)

---

## Post-Deployment Validation

**Run these checks after 100% rollout complete**

### 1. Service Health

```bash
# All pods running
kubectl get pods -n tars-production
# Expected: All Running, 0 restarts

# All endpoints healthy
for svc in eval-engine hypersync orchestration agent-dqn agent-a2c agent-ppo agent-ddpg dashboard-api dashboard-frontend; do
  kubectl exec -n tars-production $(kubectl get pod -n tars-production -l app=$svc -o jsonpath='{.items[0].metadata.name}') -- curl -f http://localhost:8080/health || echo "$svc FAILED"
done
# Expected: No FAILED
```

### 2. End-to-End Smoke Test

```bash
# Submit evaluation request via API
curl -X POST https://api.tars.prod/v1/evaluations \
  -H "Authorization: Bearer $JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "agent_type": "dqn",
    "environment": "CartPole-v1",
    "episodes": 10,
    "hyperparameters": {"learning_rate": 0.001}
  }'
# Expected: HTTP 202 Accepted, evaluation_id returned

# Poll evaluation status
eval_id="<EVALUATION_ID>"
curl https://api.tars.prod/v1/evaluations/$eval_id \
  -H "Authorization: Bearer $JWT_TOKEN"
# Expected: status = "completed", reward_mean > 150
```

### 3. Dashboard UI Validation

```bash
# Test dashboard loading
curl -I https://dashboard.tars.prod
# Expected: HTTP/2 200

# Test dashboard API
curl https://api.tars.prod/v1/agents \
  -H "Authorization: Bearer $JWT_TOKEN"
# Expected: 4 agents (DQN, A2C, PPO, DDPG)
```

### 4. Multi-Region Validation

```bash
# Check cross-region replication lag
kubectl exec -n tars-data postgres-primary-0 -- \
  psql -U tars -d tars_production -c \
  "SELECT client_addr, state, replay_lag FROM pg_stat_replication WHERE application_name LIKE 'us-west-2%' OR application_name LIKE 'eu-central-1%';"
# Expected: replay_lag <3s (all regions)

# Check HyperSync cross-region sync
curl https://api.tars.prod/v1/hypersync/status \
  -H "Authorization: Bearer $JWT_TOKEN"
# Expected: all_regions_synced = true, max_lag_ms < 5000
```

### 5. Observability Stack

```bash
# Prometheus targets
kubectl port-forward -n observability svc/prometheus-kube-prometheus-prometheus 9090:9090 &
curl http://localhost:9090/api/v1/targets | jq '.data.activeTargets[] | select(.health != "up")'
# Expected: Empty (all up)

# Grafana dashboards
curl -u admin:*** http://localhost:3000/api/dashboards/uid/tars-overview
# Expected: HTTP 200

# Jaeger traces
kubectl port-forward -n observability svc/jaeger-query 16686:16686 &
curl http://localhost:16686/api/services
# Expected: ["eval-engine", "hypersync", "orchestration", ...]
```

### 6. Security Validation

```bash
# Test JWT authentication
curl https://api.tars.prod/v1/agents
# Expected: HTTP 401 Unauthorized

curl https://api.tars.prod/v1/agents \
  -H "Authorization: Bearer invalid_token"
# Expected: HTTP 401 Unauthorized

# Test rate limiting
for i in {1..60}; do
  curl -w "%{http_code}\n" -o /dev/null https://api.tars.prod/health
done
# Expected: First 50 requests = 200, remaining = 429 (rate limited)

# Test RBAC
curl https://api.tars.prod/v1/admin/users \
  -H "Authorization: Bearer $VIEWER_TOKEN"
# Expected: HTTP 403 Forbidden
```

### 7. Performance Benchmarks

```bash
# Run latency benchmark
make bench-latency
# Expected: p95 <300s

# Run throughput benchmark
make bench-throughput
# Expected: >40 RPS sustained

# Run regression detection benchmark
make bench-regression
# Expected: F1 >85%
```

### 8. Statuspage Update

```bash
# Mark deployment complete
python canary/statuspage_client.py update-incident \
  --incident-id <INCIDENT_ID> \
  --status "resolved" \
  --message "T.A.R.S. v1.0.0 GA deployment complete. All systems operational."

# Update component statuses
python canary/update_status_workflow.py --all-operational
```

---

## Troubleshooting

### Issue: Pods stuck in `Pending` state

**Symptoms:**
```bash
kubectl get pods -n tars-production
# NAME                           READY   STATUS    RESTARTS   AGE
# tars-eval-engine-v1-0-0-xxx    0/1     Pending   0          5m
```

**Diagnosis:**
```bash
kubectl describe pod tars-eval-engine-v1-0-0-xxx -n tars-production
# Events: 0/3 nodes are available: 3 Insufficient cpu.
```

**Resolution:**
- Scale down non-critical workloads
- Add more nodes to cluster
- Reduce resource requests in Helm values

---

### Issue: High API latency (>100ms p95)

**Symptoms:**
```bash
# SLO validation fails
# ❌ FAIL: api_response_time_p95 = 142ms
```

**Diagnosis:**
```bash
# Check database connection pool
kubectl exec -n tars-production tars-eval-engine-0 -- \
  curl http://localhost:9090/metrics | grep db_pool_active_connections
# db_pool_active_connections 95 (saturated)

# Check Redis latency
kubectl exec -n tars-data redis-master-0 -- redis-cli --latency
# min: 0, max: 42, avg: 3.14 (elevated)
```

**Resolution:**
- Increase database connection pool size
- Scale Redis to more replicas
- Enable connection pooling (PgBouncer)

---

### Issue: ArgoCD sync fails

**Symptoms:**
```bash
argocd app sync tars-v1-ga
# FATA[0002] rpc error: code = Unknown desc = Manifest generation error
```

**Diagnosis:**
```bash
# Check ArgoCD logs
kubectl logs -n argocd deployment/argocd-repo-server
# Error: failed to get chart from helm repo
```

**Resolution:**
```bash
# Refresh git repo
argocd app get tars-v1-ga --refresh

# Force sync
argocd app sync tars-v1-ga --force --prune
```

---

### Issue: Database migration fails

**Symptoms:**
```bash
kubectl logs -n tars-production job/tars-migrations
# ERROR: column "new_column" already exists
```

**Diagnosis:**
- Migration script not idempotent
- Previous migration partially applied

**Resolution:**
```bash
# Manually fix schema
kubectl exec -n tars-data postgres-primary-0 -- \
  psql -U tars -d tars_production -c "ALTER TABLE ..."

# Re-run migration
kubectl delete job -n tars-production tars-migrations
argocd app sync tars-v1-ga --resource argoproj.io:Job:tars-production/tars-migrations
```

---

## Communication Plan

### Pre-Deployment (T-24 hours)

- [x] Email to all stakeholders (CTO, VP Eng, Product, Customer Success)
- [x] Slack announcement in #engineering, #product, #customer-success
- [x] Statuspage scheduled maintenance created
- [x] Customer-facing notification (for enterprise customers)

### During Deployment

**Update frequency:** Every 10 minutes (per canary step)

**Slack template:**
```
🚀 T.A.R.S. v1.0.0 GA Deployment - Update

Status: [5% Canary | 25% Canary | 50% Canary | 100% Complete]
Progress: [X/Y] steps complete
SLO Status: ✅ All SLOs passing
ETA: [TIME] minutes remaining

Live Status: https://status.tars.prod
Grafana: https://grafana.tars.prod/d/tars-overview
```

### Post-Deployment (T+1 hour)

- [x] Slack announcement (deployment complete)
- [x] Email to stakeholders (success summary + metrics)
- [x] Statuspage incident resolved
- [x] Blog post published (marketing)
- [x] Press release (if applicable)

### Rollback Communication

**If rollback occurs:**
```
🚨 T.A.R.S. v1.0.0 GA Deployment - ROLLED BACK

Reason: [SLO violation | Health probe failure | ...]
Impact: None (rollback completed in <5 minutes)
Status: v0.3.0-alpha restored, all systems operational
Next Steps: Root cause analysis, rescheduled deployment

Incident Report: [LINK]
On-Call: [NAME]
```

---

## Sign-Off

**Deployment approved by:**

- [ ] **CTO** - _Name:_ ______________ _Date:_ ________ _Signature:_ ______________
- [ ] **VP Engineering** - _Name:_ ______________ _Date:_ ________ _Signature:_ ______________
- [ ] **SRE Lead** - _Name:_ ______________ _Date:_ ________ _Signature:_ ______________
- [ ] **Security Lead** - _Name:_ ______________ _Date:_ ________ _Signature:_ ______________
- [ ] **Product Lead** - _Name:_ ______________ _Date:_ ________ _Signature:_ ______________

**Deployment executed by:**

- **On-Call Engineer:** _Name:_ ______________ _Date:_ ________ _Start Time:_ ________ _End Time:_ ________

**Rollback authority:** CTO, VP Engineering, SRE Lead (any of these can trigger rollback)

---

**End of Rollout Playbook**

**Total:** ~1,400 LOC

**Next:** [Post-Deployment Validation](#post-deployment-validation) → [GA Launch Checklist](../docs/final/GA_LAUNCH_CHECKLIST.md)
