# On-Call Playbook - T.A.R.S. Evaluation Engine

**Version:** v1.0.0-rc2
**Last Updated:** 2025-11-19
**Audience:** SRE, On-Call Engineers

---

## Table of Contents

1. [Severity Definitions](#severity-definitions)
2. [Alert Quick Actions](#alert-quick-actions)
3. [High CPU Checklist](#high-cpu-checklist)
4. [Pod Crash Loop Checklist](#pod-crash-loop-checklist)
5. [Service Down Checklist](#service-down-checklist)
6. [Rollback Failures](#rollback-failures)
7. [Critical Regression](#critical-regression)
8. [Escalation Procedures](#escalation-procedures)

---

## Severity Definitions

| Severity | Impact | Response Time | Examples |
|----------|--------|---------------|----------|
| **P0 - Critical** | Service down or critical regression | < 15 min | Service unreachable, data loss, critical regression |
| **P1 - High** | Degraded performance | < 1 hour | High latency, partial outages, high error rate |
| **P2 - Medium** | Minor degradation | < 4 hours | Non-critical errors, slow queries |
| **P3 - Low** | Monitoring concern | < 24 hours | Cache misses, baseline drift |

---

## Alert Quick Actions

### 1. EvalEngineDown (P0)

**Alert:** `EvalEngineDown` - Evaluation Engine is down

**Immediate Actions (< 5 min):**

```bash
# 1. Check pod status
kubectl get pods -n tars | grep eval-engine

# 2. Check recent events
kubectl get events -n tars --sort-by='.lastTimestamp' | tail -20

# 3. Check logs for crash reason
kubectl logs -n tars deployment/tars-eval-engine --tail=100
```

**Common Causes:**

| Symptom | Cause | Fix |
|---------|-------|-----|
| `CrashLoopBackOff` | App crash on startup | Check logs → fix code → redeploy |
| `ImagePullBackOff` | Image not found | Check image tag → update deployment |
| `OOMKilled` | Out of memory | Increase memory limit → `kubectl edit deployment` |
| `Pending` | No resources | Scale down other pods or add nodes |

**Recovery Steps:**

```bash
# If OOMKilled - increase memory
kubectl patch deployment tars-eval-engine -n tars -p '
{
  "spec": {
    "template": {
      "spec": {
        "containers": [{
          "name": "eval-engine",
          "resources": {
            "limits": {"memory": "8Gi"}
          }
        }]
      }
    }
  }
}'

# If CrashLoopBackOff - check dependencies
kubectl logs -n tars deployment/tars-eval-engine | grep -i error

# Restart deployment
kubectl rollout restart deployment tars-eval-engine -n tars

# Monitor recovery
kubectl rollout status deployment tars-eval-engine -n tars
```

**Escalation:** If not resolved in 15 min → Page ML team lead

---

### 2. CriticalRegressionDetected (P0)

**Alert:** `CriticalRegressionDetected` - Critical performance regression

**Immediate Actions:**

```bash
# 1. Get regression details
kubectl logs -n tars deployment/tars-eval-engine | grep "critical regression" | tail -5

# 2. Identify affected agent
# Example log: "Critical regression detected: agent=dqn, env=CartPole-v1, reward_drop=25%"
export AGENT_TYPE="dqn"

# 3. Trigger immediate rollback
curl -X POST http://hypersync:8098/v1/rollback \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "agent_type": "'$AGENT_TYPE'",
    "reason": "Critical regression detected - on-call engineer rollback"
  }'

# 4. Verify rollback succeeded
kubectl logs -n tars deployment/tars-orchestration | grep rollback
```

**Verification:**

```bash
# Re-evaluate with previous baseline
curl -X POST http://localhost:8099/v1/evaluate \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -d '{
    "agent_type": "'$AGENT_TYPE'",
    "hyperparameters": { /* previous baseline hyperparameters */ },
    "environments": ["CartPole-v1"],
    "num_episodes": 50
  }'

# Expected: mean_reward should return to baseline levels
```

**Notification:**

```bash
# Post to Slack #tars-critical
"🚨 Critical regression detected and rolled back
Agent: $AGENT_TYPE
Regression: 25% reward drop
Action: Rolled back to previous baseline
On-call: @your-name"
```

**Escalation:** Notify ML team lead immediately

---

### 3. HighEvaluationLatency (P1)

**Alert:** `HighEvaluationLatency` - p95 latency > 300s

**Actions:**

```bash
# 1. Check current load
kubectl top pods -n tars | grep eval-engine

# 2. Check HPA status
kubectl get hpa -n tars tars-eval-engine

# 3. Manual scale if HPA not responding
kubectl scale deployment tars-eval-engine -n tars --replicas=10

# 4. Check slow evaluations
kubectl logs -n tars deployment/tars-eval-engine | grep "evaluation_duration" | tail -20
```

**If CPU > 90%:**
- HPA should auto-scale
- If not, check [HPA Not Scaling](troubleshooting-guide.md#hpa-not-scaling)

**If CPU < 50%:**
- Likely slow episodes or slow environments
- Check [Slow Episodes](troubleshooting-guide.md#slow-episodes)

---

### 4. PostgreSQLConnectionPoolExhausted (P1)

**Alert:** `PostgreSQLConnectionPoolExhausted`

**Actions:**

```bash
# 1. Check current connections
psql $POSTGRES_URL -c "
SELECT count(*), state
FROM pg_stat_activity
WHERE datname='tars'
GROUP BY state;"

# 2. Kill idle connections (if many idle)
psql $POSTGRES_URL -c "
SELECT pg_terminate_backend(pid)
FROM pg_stat_activity
WHERE datname='tars'
AND state='idle'
AND state_change < NOW() - INTERVAL '5 minutes';"

# 3. Restart eval-engine pods (forces new connections)
kubectl rollout restart deployment tars-eval-engine -n tars

# 4. Monitor connection count
watch -n 5 "psql $POSTGRES_URL -c \"SELECT count(*) FROM pg_stat_activity WHERE datname='tars';\""
```

**Long-term fix:**
- Increase pool size in [main.py](../../cognition/eval-engine/main.py)
- Add connection pooling metrics

---

### 5. EvaluationFailureRateHigh (P1)

**Alert:** `EvaluationFailureRateHigh` - >5% failures

**Actions:**

```bash
# 1. Get recent errors
kubectl logs -n tars deployment/tars-eval-engine --tail=200 | grep ERROR > errors.log

# 2. Count error types
cat errors.log | awk '{print $5}' | sort | uniq -c | sort -rn

# 3. Check most common error
# Example output:
# 45 EnvironmentNotFoundError
# 12 ValidationError
# 3 TimeoutError

# 4. Fix based on error type
```

**Common Fixes:**

| Error | Fix |
|-------|-----|
| `EnvironmentNotFoundError` | Check environment spelling, update AutoML constraints |
| `ValidationError` | Check hyperparameter ranges, update schema |
| `TimeoutError` | Increase `EVAL_WORKER_TIMEOUT` env var |
| `DatabaseError` | Check PostgreSQL health |

---

## High CPU Checklist

**Alert:** `EvalEnginePodCPUHigh`

- [ ] Check current CPU usage: `kubectl top pods -n tars`
- [ ] Check HPA status: `kubectl get hpa -n tars`
- [ ] Check for stuck evaluations: `kubectl logs -n tars deployment/tars-eval-engine | grep "stuck"`
- [ ] Check worker pool size: Configured for max concurrent evaluations
- [ ] Scale up if needed: `kubectl scale deployment tars-eval-engine -n tars --replicas=8`
- [ ] Increase CPU limits if chronically high: Update deployment YAML
- [ ] Monitor for 10 minutes: Verify scaling helps

**If CPU remains high after scaling:**
- Review recent AutoML trials (may be requesting many concurrent evaluations)
- Check for infinite loops in worker code
- Profile with `py-spy`: `kubectl exec -it <pod> -- py-spy top --pid 1`

---

## Pod Crash Loop Checklist

**Alert:** `EvalEngineHighRestartRate`

- [ ] Get crash logs: `kubectl logs -n tars <pod-name> --previous`
- [ ] Check exit code: `kubectl describe pod -n tars <pod-name> | grep "Exit Code"`
- [ ] Common exit codes:
  - `137` = OOMKilled (increase memory)
  - `1` = App exception (check logs)
  - `143` = SIGTERM (normal shutdown)
- [ ] Check resource limits: `kubectl describe pod -n tars <pod-name> | grep -A 5 Limits`
- [ ] Check liveness/readiness probes: May be too aggressive
- [ ] Check dependencies (PostgreSQL, Redis): Must be healthy before app starts

**Fix OOMKilled:**
```bash
kubectl patch deployment tars-eval-engine -n tars -p '
{
  "spec": {
    "template": {
      "spec": {
        "containers": [{
          "name": "eval-engine",
          "resources": {
            "limits": {"memory": "16Gi"},
            "requests": {"memory": "8Gi"}
          }
        }]
      }
    }
  }
}'
```

---

## Service Down Checklist

**Alert:** `EvalEngineDown`

- [ ] Check pod status: `kubectl get pods -n tars | grep eval-engine`
- [ ] Check service endpoints: `kubectl get endpoints -n tars tars-eval-engine`
- [ ] Check service definition: `kubectl get svc tars-eval-engine -n tars -o yaml`
- [ ] Test internal connectivity: `kubectl run -it --rm debug --image=curlimages/curl -- curl http://tars-eval-engine:8099/health`
- [ ] Check ingress: `kubectl get ingress -n tars`
- [ ] Check certificate: `kubectl get certificate -n tars`
- [ ] Check DNS: `nslookup tars-eval-engine.tars.svc.cluster.local`

**Recovery:**
```bash
# Restart deployment
kubectl rollout restart deployment tars-eval-engine -n tars

# If service missing
kubectl apply -f charts/tars/templates/eval-engine-service.yaml

# If ingress broken
kubectl delete ingress tars-ingress -n tars
kubectl apply -f charts/tars/templates/ingress.yaml
```

---

## Rollback Failures

**Alert:** `HyperSyncRollbackFailures`

**Impact:** Critical - regressions cannot be mitigated

**Actions:**

```bash
# 1. Check HyperSync service
kubectl get pods -n tars | grep hypersync

# 2. Test HyperSync health
curl http://hypersync:8098/health

# 3. Manual rollback via database
psql $POSTGRES_URL -c "
BEGIN;
UPDATE eval_baselines SET rank = 99 WHERE agent_type = '$AGENT_TYPE' AND rank = 1;
UPDATE eval_baselines SET rank = 1 WHERE agent_type = '$AGENT_TYPE' AND rank = 2;
UPDATE eval_baselines SET rank = 2 WHERE agent_type = '$AGENT_TYPE' AND rank = 99;
COMMIT;
"

# 4. Notify orchestration to reload baselines
kubectl rollout restart deployment tars-orchestration -n tars
```

**Escalation:** Page platform team + ML team lead

---

## Critical Regression

**Alert:** `CriticalRegressionDetected`

**Definition:** Reward drop >20% or failure rate >25%

**Response Protocol:**

1. **Immediate (0-5 min):**
   - Acknowledge alert
   - Trigger rollback (see above)
   - Post to #tars-critical Slack channel

2. **Short-term (5-30 min):**
   - Verify rollback succeeded
   - Re-evaluate with previous baseline
   - Document regression details

3. **Follow-up (30-120 min):**
   - Root cause analysis
   - Review AutoML trial that caused regression
   - Update regression thresholds if needed
   - Post mortem document

**Regression Report Template:**

```markdown
## Critical Regression Incident

**Time:** 2025-11-19 10:30 UTC
**Agent:** dqn
**Environment:** CartPole-v1
**Severity:** Critical

### Metrics
- Baseline mean_reward: 200.0
- Current mean_reward: 155.0
- Reward drop: 22.5%
- Failure rate: 30%

### Root Cause
AutoML trial #1234 with learning_rate=0.0001 (too low)

### Actions Taken
1. Rolled back to previous baseline (trial #1233)
2. Updated AutoML search space: learning_rate ∈ [0.0005, 0.01]
3. Restarted orchestration

### Prevention
- Added lower bound constraint on learning_rate
- Increased regression detection sensitivity for dqn
```

---

## Escalation Procedures

### On-Call Tiers

| Tier | Role | Contact |
|------|------|---------|
| **Tier 1** | On-call SRE | PagerDuty |
| **Tier 2** | ML Team Lead | Phone + Slack |
| **Tier 3** | Platform Architect | Email |

### When to Escalate

**Escalate to Tier 2 (ML Team Lead) if:**
- Cannot resolve P0 in 15 minutes
- Critical regression detected
- Data corruption suspected
- Multiple services affected

**Escalate to Tier 3 (Platform Architect) if:**
- Infrastructure-wide failure
- Security incident
- Data loss

### Escalation Contacts

```bash
# Slack channels
#tars-critical     # P0 incidents
#tars-alerts       # All alerts
#tars-oncall       # On-call coordination

# PagerDuty
eval-engine-oncall    # Auto-pages Tier 1
ml-team-oncall        # Auto-pages Tier 2
```

---

## Post-Incident Checklist

After resolving P0/P1 incident:

- [ ] Update incident ticket with resolution
- [ ] Post timeline to #tars-critical
- [ ] Schedule post-mortem (if P0)
- [ ] Update runbooks with new learnings
- [ ] File bugs for long-term fixes
- [ ] Update alert thresholds if needed

---

## Quick Reference

```bash
# Emergency rollback
curl -X POST http://hypersync:8098/v1/rollback -d '{"agent_type":"dqn"}'

# Force restart
kubectl delete pod -n tars -l app=tars-eval-engine

# Scale up
kubectl scale deployment tars-eval-engine -n tars --replicas=10

# Health check
curl http://localhost:8099/health

# Recent errors
kubectl logs -n tars deployment/tars-eval-engine --tail=100 | grep ERROR
```

---

## Related Documents

- [Evaluation Pipeline Runbook](evaluation-pipeline-runbook.md)
- [Troubleshooting Guide](troubleshooting-guide.md)
- [Prometheus Alerts](../../observability/alerts/prometheus-alerts.yaml)
