# T.A.R.S. v1.0.1 GA Day Runbook

**Version:** 1.0.1
**GA Date:** [TO BE SCHEDULED]
**Document Owner:** Release Manager
**Last Updated:** 2025-11-20
**Status:** Production Ready

---

## 🎯 Executive Summary

This runbook provides minute-by-minute instructions for executing the General Availability (GA) deployment of T.A.R.S. v1.0.1 to production. This deployment includes 5 critical hotfixes, performance improvements, and production-readiness enhancements.

**Deployment Window:** [START_TIME] - [END_TIME] (4-hour block)
**Expected Duration:** 90-120 minutes (including canary stages)
**Rollback Time:** < 5 minutes
**Zero-Downtime:** Yes (canary deployment)

---

## 📋 Pre-GA Checklist (T-24 Hours)

### Stakeholder Readiness

- [ ] **Release Manager:** Final go/no-go decision authority
- [ ] **SRE Lead:** On-call, monitoring dashboards ready
- [ ] **QA Lead:** Production validation suite verified
- [ ] **Engineering Lead:** Hotfix validation confirmed
- [ ] **Product Manager:** Customer communication draft approved
- [ ] **Support Team:** Briefed on changes, escalation path confirmed

### Technical Readiness

- [ ] **Staging Deployment:** v1.0.1 validated (100% test pass)
- [ ] **Production Validation Suite:** All 200+ tests passing in staging
- [ ] **Monitoring:** Grafana dashboards, Prometheus alerts, Jaeger tracing operational
- [ ] **PagerDuty:** Integration tested, escalation policy confirmed
- [ ] **Rollback Plan:** Helm rollback tested in staging
- [ ] **Database Migrations:** Validated in staging, rollback scripts ready
- [ ] **Feature Flags:** Configured, tested in staging
- [ ] **Canary Configuration:** values-canary.yaml reviewed

### Communication Readiness

- [ ] **Slack Channels:**
  - #tars-deployments (technical updates)
  - #tars-support (customer issues)
  - #tars-leadership (executive summary)
- [ ] **Status Page:** https://status.tars.ai prepared with "Scheduled Maintenance" notice
- [ ] **Customer Email:** Draft prepared for post-deployment announcement
- [ ] **Internal Wiki:** Updated with GA timeline

### Artifact Verification

```bash
# Verify Helm chart package
cd release/v1_0_1/
sha256sum -c SHA256SUMS
# Expected: tars-1.0.1.tgz: OK

# Verify manifest
cat manifest.production.json | jq '.version, .hotfixes | length'
# Expected: "1.0.1", 5

# Verify Docker images
docker pull ghcr.io/YOUR_ORG/tars-orchestration-agent:v1.0.1
docker pull ghcr.io/YOUR_ORG/tars-insight-engine:v1.0.1
docker pull ghcr.io/YOUR_ORG/tars-dashboard-api:v1.0.1
# All should succeed

# Verify database migrations
ls -l migrations/v1_0_1_*.sql
# Expected: v1_0_1_add_indexes.sql, v1_0_1_add_trace_columns.sql
```

---

## ⏰ GA Day Timeline

### T-60 Minutes: Final Preparation

**Time:** [START_TIME - 60 min]
**Duration:** 30 minutes
**Participants:** Release Manager, SRE Lead, QA Lead

**Actions:**

1. **War Room Setup**
   ```bash
   # Join video call
   # Open monitoring dashboards in separate tabs:
   # - Grafana: https://tars.ai/grafana/d/tars-overview
   # - Prometheus: https://tars.ai/prometheus/alerts
   # - Jaeger: https://tars.ai/jaeger
   # - Kubernetes Dashboard: kubectl proxy & open http://localhost:8001/api/v1/namespaces/kubernetes-dashboard/services/https:kubernetes-dashboard:/proxy/
   ```

2. **Status Page Update**
   ```bash
   # Update status page
   curl -X POST https://api.statuspage.io/v1/pages/YOUR_PAGE_ID/incidents \
     -H "Authorization: OAuth YOUR_TOKEN" \
     -d '{
       "incident": {
         "name": "T.A.R.S. v1.0.1 Production Deployment",
         "status": "scheduled",
         "scheduled_for": "[START_TIME]",
         "scheduled_until": "[END_TIME]",
         "message": "We are deploying T.A.R.S. v1.0.1 with performance improvements and bug fixes. No downtime expected."
       }
     }'
   ```

3. **Baseline Metrics Capture**
   ```bash
   # Capture current production metrics
   cd release/v1_0_1/ga_monitoring_scripts/

   # Baseline SLOs (save for comparison)
   python ga_kpi_collector.py \
     --environment=production \
     --duration=5m \
     --output=baseline_slos.json

   # Verify current state
   kubectl get deployments -n tars-production
   kubectl top pods -n tars-production
   ```

4. **Go/No-Go Poll**
   - **Release Manager:** "All artifacts verified?" → GO/NO-GO
   - **SRE Lead:** "Monitoring ready?" → GO/NO-GO
   - **QA Lead:** "Validation suite ready?" → GO/NO-GO
   - **Engineering Lead:** "Rollback plan confirmed?" → GO/NO-GO

**Go/No-Go Criteria:**

| Criterion | Threshold | Current | Status |
|-----------|-----------|---------|--------|
| Production availability | ≥ 99.9% (last 24h) | [CHECK] | [ ] |
| API p95 latency | < 100ms | [CHECK] | [ ] |
| Error rate | < 1% | [CHECK] | [ ] |
| Active incidents | 0 | [CHECK] | [ ] |
| Staging validation | 100% pass | [CHECK] | [ ] |

**Decision:** [ ] GO / [ ] NO-GO / [ ] POSTPONE

---

### T-30 Minutes: Pre-Deployment Checks

**Time:** [START_TIME - 30 min]
**Duration:** 20 minutes
**Participants:** SRE Lead

**Actions:**

1. **Kubernetes Health Check**
   ```bash
   # Verify cluster health
   kubectl get nodes
   # All nodes should be Ready

   # Check resource availability
   kubectl describe nodes | grep -A 5 "Allocated resources"
   # Ensure < 80% CPU/Memory on all nodes

   # Verify namespace
   kubectl get ns tars-production
   kubectl get resourcequota -n tars-production
   ```

2. **Database Health Check**
   ```bash
   # Check PostgreSQL
   kubectl exec -n tars-production deploy/tars-postgres -- \
     psql -U tars -d tars -c "SELECT version();"

   # Check connections
   kubectl exec -n tars-production deploy/tars-postgres -- \
     psql -U tars -d tars -c "SELECT count(*) FROM pg_stat_activity;"
   # Should be < max_connections (typically 100)

   # Verify replication lag (if applicable)
   kubectl exec -n tars-production deploy/tars-postgres -- \
     psql -U tars -d tars -c "SELECT pg_is_in_recovery(), pg_last_wal_receive_lsn();"
   ```

3. **Redis Health Check**
   ```bash
   # Check Redis
   kubectl exec -n tars-production deploy/tars-redis -- redis-cli ping
   # Expected: PONG

   # Check memory usage
   kubectl exec -n tars-production deploy/tars-redis -- \
     redis-cli INFO memory | grep used_memory_human
   # Should be < 80% of limit
   ```

4. **Backup Verification**
   ```bash
   # Verify recent database backup
   kubectl get cronjob -n tars-production tars-postgres-backup

   # Check last backup
   kubectl logs -n tars-production \
     $(kubectl get pods -n tars-production -l job-name --sort-by=.metadata.creationTimestamp -o name | tail -1)
   # Should show successful backup within last 24h
   ```

---

### T-0: Deployment Execution

**Time:** [START_TIME]
**Duration:** 60-90 minutes
**Participants:** Release Manager, SRE Lead, QA Lead

#### Stage 0: Pre-Deployment Database Migrations (T+0 to T+5)

**Duration:** 5 minutes

```bash
# Apply pre-deployment migrations (non-breaking)
kubectl create job tars-db-migrate-v1-0-1-pre \
  --from=cronjob/tars-db-maintenance \
  -n tars-production

# Apply migration
kubectl exec -n tars-production \
  $(kubectl get pod -n tars-production -l job-name=tars-db-migrate-v1-0-1-pre -o name) -- \
  psql -U tars -d tars -f /migrations/v1_0_1_add_indexes.sql

# Verify indexes created
kubectl exec -n tars-production deploy/tars-postgres -- \
  psql -U tars -d tars -c "
    SELECT indexname, tablename
    FROM pg_indexes
    WHERE schemaname='public'
    AND indexname IN ('idx_missions_status_created', 'idx_evaluations_mission_timestamp', 'idx_traces_span_id');
  "
# Expected: 3 rows

# Monitor migration impact
kubectl top pod -n tars-production -l app=postgres
# CPU/Memory should not spike significantly
```

**Validation:**
- [ ] All 3 indexes created successfully
- [ ] Database responsive (query time < 100ms)
- [ ] No connection pool exhaustion

**Rollback (if needed):**
```bash
# Drop indexes if causing issues
kubectl exec -n tars-production deploy/tars-postgres -- \
  psql -U tars -d tars -c "
    DROP INDEX IF EXISTS idx_missions_status_created;
    DROP INDEX IF EXISTS idx_evaluations_mission_timestamp;
    DROP INDEX IF EXISTS idx_traces_span_id;
  "
```

---

#### Stage 1: Canary 1% (T+5 to T+15)

**Duration:** 10 minutes
**Traffic:** 1% of production load

```bash
# Trigger GitHub Actions workflow
gh workflow run production_deploy_pipeline.yaml \
  --ref release/v1.0.1 \
  --field deployment_strategy=canary \
  --field canary_stages="1,10,25,50,100" \
  --field stage_duration_minutes=10 \
  --field enable_feature_flags=true \
  --field notify_pagerduty=true

# Monitor workflow
WORKFLOW_RUN_ID=$(gh run list --workflow=production_deploy_pipeline.yaml --limit 1 --json databaseId -q '.[0].databaseId')
gh run watch $WORKFLOW_RUN_ID

# Monitor canary deployment
watch -n 5 'kubectl get pods -n tars-production -l deployment=canary'
```

**Real-Time Monitoring:**
```bash
# Start real-time SLO monitor
cd release/v1_0_1/ga_monitoring_scripts/
python monitor_realtime_slos.py \
  --environment=production \
  --deployment=canary \
  --threshold-availability=99.0 \
  --threshold-latency-p95=120 \
  --threshold-error-rate=2.0 \
  --alert-webhook=$SLACK_WEBHOOK_URL
```

**Canary Gate Criteria (1%):**

| Metric | Threshold | Check |
|--------|-----------|-------|
| Canary pod status | All Running | `kubectl get pods -n tars-production -l deployment=canary` |
| Canary availability | ≥ 99.0% | Prometheus: `avg_over_time(up{deployment="canary"}[5m])*100` |
| Canary error rate | < 2% | Prometheus: `rate(http_requests_total{deployment="canary",status=~"5.."}[5m])/rate(http_requests_total{deployment="canary"}[5m])*100` |
| Canary p95 latency | < 120ms | Prometheus: `histogram_quantile(0.95, rate(http_request_duration_seconds_bucket{deployment="canary"}[5m]))*1000` |
| Canary memory | < 2.5GB per pod | `kubectl top pod -n tars-production -l deployment=canary` |
| No crash loops | 0 restarts | `kubectl get pods -n tars-production -l deployment=canary -o json \| jq '.items[].status.containerStatuses[].restartCount'` |

**Manual Validation:**
```bash
# Test canary endpoints directly
CANARY_POD=$(kubectl get pod -n tars-production -l deployment=canary -o name | head -1)

# Health check
kubectl exec -n tars-production $CANARY_POD -- curl -f http://localhost:8080/health
# Expected: {"status": "healthy"}

# Test TARS-1001 fix (WebSocket reconnection)
kubectl exec -n tars-production $CANARY_POD -- curl -f http://localhost:8080/api/v1/websocket/stats
# Expected: {"reconnection_enabled": true, "max_retries": 5}

# Test TARS-1002 fix (Grafana performance)
time curl -f -H "Host: tars.ai" http://CANARY_INGRESS_IP/grafana/d/tars-evaluation
# Expected: < 3 seconds
```

**Decision Point:**
- [ ] **PROCEED** to Canary 10% (all gates passed)
- [ ] **HOLD** (investigate anomalies)
- [ ] **ROLLBACK** (critical failure)

**Rollback Command (if needed):**
```bash
# Automatic rollback is built into pipeline
# Manual rollback:
kubectl delete deployment tars-canary -n tars-production
helm rollback tars -n tars-production
```

---

#### Stage 2: Canary 10% (T+15 to T+25)

**Duration:** 10 minutes
**Traffic:** 10% of production load

**Pipeline will automatically promote if Stage 1 passed.**

**Monitoring:**
```bash
# Watch canary scaling
watch -n 5 'kubectl get pods -n tars-production -l deployment=canary -o wide'

# Monitor traffic distribution
kubectl exec -n tars-production prometheus-0 -- \
  wget -qO- "http://localhost:9090/api/v1/query?query=sum(rate(http_requests_total[1m]))by(deployment)"
# Expected: canary ~10%, stable ~90%
```

**Canary Gate Criteria (10%):**

| Metric | Threshold | Check |
|--------|-----------|-------|
| Canary availability | ≥ 99.5% | Prometheus query (as above) |
| Canary error rate | < 1.5% | Prometheus query (as above) |
| Canary p95 latency | < 110ms | Prometheus query (as above) |
| Comparison to stable | Within 10% | `python drift_detector.py --deployment=canary --baseline=stable` |
| TARS-1001 validation | 0 reconnect failures | `python monitor_websocket_health.py --deployment=canary --duration=5m` |

**Hotfix Validation (Quick Spot Checks):**
```bash
# TARS-1003: Database query optimization
kubectl exec -n tars-production $CANARY_POD -- \
  curl -f http://localhost:8080/api/v1/missions?limit=100&status=active
# Check response time in logs (should be < 50ms)

# TARS-1004: Distributed tracing coverage
kubectl exec -n tars-production $CANARY_POD -- \
  curl -f http://localhost:8080/api/v1/traces/recent?limit=10
# Expected: All traces have span_id, trace_id

# TARS-1005: PPO memory optimization
kubectl top pod -n tars-production -l deployment=canary,agent_type=ppo
# Expected: < 1.5GB per pod (down from 3.7GB)
```

**Decision Point:**
- [ ] **PROCEED** to Canary 25%
- [ ] **HOLD**
- [ ] **ROLLBACK**

---

#### Stage 3: Canary 25% (T+25 to T+35)

**Duration:** 10 minutes
**Traffic:** 25% of production load

**Monitoring:**
```bash
# Start comprehensive drift detection
python drift_detector.py \
  --environment=production \
  --canary-deployment=canary \
  --stable-deployment=stable \
  --duration=10m \
  --output=drift_report_25pct.json
```

**Canary Gate Criteria (25%):**

| Metric | Threshold | Check |
|--------|-----------|-------|
| Canary availability | ≥ 99.9% | Prometheus |
| Canary error rate | < 1% | Prometheus |
| Canary p95 latency | < 100ms | Prometheus |
| Drift from stable | < 5% | drift_detector.py |
| All 5 hotfixes validated | 100% | Manual spot checks |

**Comprehensive Hotfix Validation:**
```bash
# Run production validation suite against canary
pytest release/v1_0_1/production_validation_suite.py \
  --environment=production \
  --deployment=canary \
  --ga_mode \
  -v \
  -m "canary"

# Expected: 100% pass rate (50+ canary-specific tests)
```

**Decision Point:**
- [ ] **PROCEED** to Canary 50%
- [ ] **HOLD**
- [ ] **ROLLBACK**

---

#### Stage 4: Canary 50% (T+35 to T+45)

**Duration:** 10 minutes
**Traffic:** 50% of production load

**Monitoring:**
```bash
# Full KPI collection
python ga_kpi_collector.py \
  --environment=production \
  --deployment=canary \
  --duration=10m \
  --output=canary_50pct_kpis.json

# Watch for any resource contention
kubectl top nodes
kubectl top pods -n tars-production --sort-by=memory
```

**Canary Gate Criteria (50%):**

| Metric | Threshold | Check |
|--------|-----------|-------|
| Canary availability | ≥ 99.9% | Prometheus |
| Canary error rate | < 0.5% | Prometheus |
| Canary p95 latency | < 100ms | Prometheus |
| Overall cluster health | No resource exhaustion | `kubectl top nodes` |
| Customer-reported issues | 0 | Check #tars-support Slack |

**Load Testing (Optional):**
```bash
# If traffic is low, run synthetic load test
kubectl create job tars-load-test \
  --image=grafana/k6 \
  -n tars-production \
  -- run /scripts/load_test.js --vus 100 --duration 5m
```

**Decision Point:**
- [ ] **PROCEED** to Canary 100% (Full Rollout)
- [ ] **HOLD**
- [ ] **ROLLBACK**

---

#### Stage 5: Canary 100% (T+45 to T+55)

**Duration:** 10 minutes
**Traffic:** 100% of production load (canary becomes primary)

```bash
# Pipeline will automatically promote canary to stable
# and terminate old pods gracefully

# Watch rollout
kubectl rollout status deployment/tars-orchestration-agent -n tars-production
kubectl rollout status deployment/tars-insight-engine -n tars-production
kubectl rollout status deployment/tars-dashboard-api -n tars-production
# All should complete successfully
```

**Canary Gate Criteria (100%):**

| Metric | Threshold | Check |
|--------|-----------|-------|
| All pods Running | 0 CrashLoopBackOff | `kubectl get pods -n tars-production` |
| Availability | ≥ 99.9% | Prometheus |
| Error rate | < 0.5% | Prometheus |
| p95 latency | < 100ms | Prometheus |
| All replicas ready | Desired = Current = Available | `kubectl get deployments -n tars-production` |

**Final Validation:**
```bash
# Run full production validation suite
pytest release/v1_0_1/production_validation_suite.py \
  --environment=production \
  --namespace=tars-production \
  --ga_mode \
  -v \
  --html=ga_validation_report.html \
  --self-contained-html

# Expected: 200+ tests, 100% pass rate
```

**Decision Point:**
- [ ] **GA DEPLOYMENT SUCCESSFUL** (proceed to post-deployment)
- [ ] **ROLLBACK**

---

#### Stage 6: Post-Deployment Migrations (T+55 to T+60)

**Duration:** 5 minutes

```bash
# Apply post-deployment migrations (if any)
# NOTE: v1.0.1 has no post-deployment migrations

# Add distributed tracing columns (already in pre-deployment)
kubectl exec -n tars-production deploy/tars-postgres -- \
  psql -U tars -d tars -c "
    SELECT column_name
    FROM information_schema.columns
    WHERE table_name='missions' AND column_name IN ('trace_id', 'span_id');
  "
# Expected: trace_id, span_id
```

---

### T+60 Minutes: Post-Deployment Validation

**Time:** [START_TIME + 60 min]
**Duration:** 30 minutes
**Participants:** Release Manager, SRE Lead, QA Lead

#### Immediate Validation (T+60 to T+70)

```bash
# 1. Deployment Health
kubectl get deployments -n tars-production
kubectl get pods -n tars-production -o wide
# All pods should be Running, 0 restarts

# 2. Service Health
for service in tars-orchestration-agent tars-insight-engine tars-dashboard-api; do
  kubectl exec -n tars-production deploy/$service -- curl -f http://localhost:8080/health
done
# All should return {"status": "healthy"}

# 3. External Endpoints
curl -f https://tars.ai/health
curl -f https://api.tars.ai/health
# Both should succeed

# 4. Database Health
kubectl exec -n tars-production deploy/tars-postgres -- \
  psql -U tars -d tars -c "SELECT count(*) FROM pg_stat_activity WHERE state='active';"
# Should be normal (< 50)

# 5. Monitoring Stack
curl -f https://tars.ai/grafana/api/health
curl -f https://tars.ai/prometheus/-/healthy
# Both should return OK
```

#### SLO Validation (T+70 to T+80)

```bash
# Generate SLO report for first 30 minutes post-deployment
python ga_kpi_collector.py \
  --environment=production \
  --duration=30m \
  --output=ga_first_30min_kpis.json \
  --compare-baseline=baseline_slos.json

# Check SLO compliance
python -c "
import json
with open('ga_first_30min_kpis.json') as f:
    kpis = json.load(f)
    print(f'Availability: {kpis[\"availability_pct\"]:.2f}% (target: ≥99.9%)')
    print(f'p95 Latency: {kpis[\"latency_p95_ms\"]:.1f}ms (target: <100ms)')
    print(f'Error Rate: {kpis[\"error_rate_pct\"]:.2f}% (target: <1%)')
"
```

**SLO Targets:**

| SLO | Target | Actual | Status |
|-----|--------|--------|--------|
| Availability | ≥ 99.9% | [CHECK] | [ ] |
| API p95 latency | < 100ms | [CHECK] | [ ] |
| Error rate | < 1% | [CHECK] | [ ] |
| WebSocket uptime | ≥ 99.9% | [CHECK] | [ ] |
| Grafana load time | < 3s | [CHECK] | [ ] |

#### Hotfix Validation (T+80 to T+90)

```bash
# Run hotfix-specific validation suite
pytest release/v1_0_1/production_validation_suite.py \
  --environment=production \
  --ga_mode \
  -v \
  -m "hotfix"

# Expected: 25+ hotfix tests, 100% pass
```

**Manual Hotfix Checks:**

```bash
# TARS-1001: WebSocket reconnection
python monitor_websocket_health.py \
  --environment=production \
  --duration=10m \
  --output=websocket_health_ga.json

# Expected: reconnection_success_rate ≥ 99.9%

# TARS-1002: Grafana performance
for dashboard in tars-overview tars-evaluation tars-agents; do
  time curl -f https://tars.ai/grafana/d/$dashboard > /dev/null
done
# All should complete < 3 seconds

# TARS-1003: Database query performance
kubectl exec -n tars-production deploy/tars-postgres -- \
  psql -U tars -d tars -c "EXPLAIN ANALYZE SELECT * FROM missions WHERE status='active' ORDER BY created_at DESC LIMIT 100;"
# Should use idx_missions_status_created, execution time < 50ms

# TARS-1004: Distributed tracing coverage
curl -f https://api.tars.ai/api/v1/traces/stats
# Expected: {"coverage_pct": 100.0, "total_traces": ...}

# TARS-1005: PPO memory usage
kubectl top pod -n tars-production -l agent_type=ppo
# Expected: < 1.5GB per pod (was 3.7GB)
```

**Validation Checklist:**

- [ ] TARS-1001: WebSocket reconnect success rate ≥ 99.9%
- [ ] TARS-1002: Grafana dashboards load < 3s
- [ ] TARS-1003: Database queries use indexes, < 50ms
- [ ] TARS-1004: Distributed tracing 100% coverage
- [ ] TARS-1005: PPO agent memory < 1.5GB

---

### T+90 Minutes: Communication & Cleanup

**Time:** [START_TIME + 90 min]
**Duration:** 10 minutes

#### Update Status Page

```bash
# Mark deployment as complete
curl -X PATCH https://api.statuspage.io/v1/pages/YOUR_PAGE_ID/incidents/INCIDENT_ID \
  -H "Authorization: OAuth YOUR_TOKEN" \
  -d '{
    "incident": {
      "status": "resolved",
      "message": "T.A.R.S. v1.0.1 deployment completed successfully. All systems operational."
    }
  }'
```

#### Slack Notifications

```bash
# Send success notification
curl -X POST $SLACK_WEBHOOK_URL \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "✅ *T.A.R.S. v1.0.1 GA Deployment Complete*",
    "blocks": [
      {
        "type": "section",
        "text": {
          "type": "mrkdwn",
          "text": "*T.A.R.S. v1.0.1 GA Deployment Complete*\n\n• Duration: [X] minutes\n• Strategy: Canary (1→10→25→50→100)\n• Status: ✅ SUCCESS\n• SLOs: All targets met\n• Hotfixes: 5/5 validated\n\nMonitoring: https://tars.ai/grafana"
        }
      }
    ]
  }'
```

#### Cleanup Canary Resources

```bash
# Pipeline should auto-cleanup, but verify
kubectl get pods -n tars-production -l deployment=canary
# Should be empty (all replaced with stable)

# Remove canary config map (if exists)
kubectl delete configmap tars-canary-config -n tars-production --ignore-not-found=true
```

---

## 📊 Post-GA Monitoring (First 24 Hours)

### Continuous Monitoring

```bash
# Run 24-hour KPI collector
nohup python ga_kpi_collector.py \
  --environment=production \
  --duration=24h \
  --interval=5m \
  --output=ga_24h_kpis.json \
  > ga_kpi_collector.log 2>&1 &

# Run drift detector (compare to staging)
nohup python drift_detector.py \
  --environment=production \
  --compare-to=staging \
  --duration=24h \
  --interval=15m \
  --output=ga_24h_drift.json \
  > drift_detector.log 2>&1 &
```

### Scheduled Checks

| Time Post-GA | Action | Responsible |
|--------------|--------|-------------|
| +2 hours | Review first 2h metrics, check for anomalies | SRE On-Call |
| +4 hours | Customer feedback review, support ticket check | Support Lead |
| +8 hours | Performance analysis, resource optimization | Engineering Lead |
| +12 hours | Mid-point SLO report | Release Manager |
| +24 hours | Full 24h GA Day Report, retrospective scheduling | Release Manager |

### Alert Thresholds (First 24 Hours)

**Stricter than normal to catch regressions early:**

| Alert | Normal Threshold | GA Day Threshold | Action |
|-------|------------------|------------------|--------|
| Availability | < 99.9% | < 99.95% | Immediate investigation |
| p95 Latency | > 100ms | > 90ms | Monitor closely |
| Error Rate | > 1% | > 0.5% | Immediate investigation |
| Pod Restarts | > 3/hour | > 1/hour | Investigate root cause |
| Memory Growth | > 10%/hour | > 5%/hour | Check for memory leaks |

---

## 🚨 Incident Response

### Severity Levels

**P0 (Critical) - Immediate Rollback Required**
- Availability < 99%
- Error rate > 5%
- Data corruption detected
- Security vulnerability exploited

**P1 (High) - Investigate, Prepare Rollback**
- Availability 99-99.5%
- Error rate 2-5%
- Performance degradation > 50%
- Critical feature broken

**P2 (Medium) - Monitor, Fix in Hotfix**
- Availability 99.5-99.9%
- Error rate 1-2%
- Performance degradation 20-50%
- Non-critical feature broken

**P3 (Low) - Track, Fix in Next Release**
- Minor UI issues
- Performance degradation < 20%
- Documentation errors

### Rollback Decision Tree

```
Is availability < 99%?
├─ YES → IMMEDIATE ROLLBACK
└─ NO
    │
    Is error rate > 5%?
    ├─ YES → IMMEDIATE ROLLBACK
    └─ NO
        │
        Is performance degraded > 50%?
        ├─ YES → PREPARE ROLLBACK, INVESTIGATE (15 min deadline)
        └─ NO
            │
            Is critical feature broken?
            ├─ YES → PREPARE ROLLBACK, INVESTIGATE (30 min deadline)
            └─ NO → MONITOR, FIX IN HOTFIX
```

### Rollback Procedure

```bash
# 1. Announce rollback decision
curl -X POST $SLACK_WEBHOOK_URL \
  -d '{"text": "🚨 *ROLLBACK INITIATED* - T.A.R.S. v1.0.1\nReason: [REASON]\nETA: 5 minutes"}'

# 2. Execute Helm rollback
helm rollback tars -n tars-production --wait --timeout=3m

# 3. Verify rollback
kubectl rollout status deployment -n tars-production --timeout=3m
kubectl get pods -n tars-production

# 4. Rollback database migrations (if needed)
kubectl exec -n tars-production deploy/tars-postgres -- \
  psql -U tars -d tars -f /migrations/rollback/v1_0_1_rollback.sql

# 5. Verify health
curl -f https://tars.ai/health
pytest release/v1_0_1/production_validation_suite.py --ga_mode -v

# 6. Update status page
curl -X POST https://api.statuspage.io/v1/pages/YOUR_PAGE_ID/incidents \
  -d '{"incident": {"name": "T.A.R.S. v1.0.1 Rollback", "status": "identified", "message": "We have rolled back to v1.0.0 due to [REASON]. Investigating."}}'

# 7. Notify stakeholders
echo "Rollback completed at $(date). Reason: [REASON]" | \
  mail -s "T.A.R.S. v1.0.1 Rollback Notification" team@tars.ai

# 8. Create incident post-mortem
# Template: docs/runbooks/incident_postmortem_template.md
```

**Expected Rollback Time:** < 5 minutes

---

## 📈 Success Metrics

### Deployment Success

- [ ] Zero-downtime deployment achieved
- [ ] All canary gates passed (5/5 stages)
- [ ] Manual approvals obtained (Release Manager + SRE)
- [ ] No rollback triggered
- [ ] Total deployment time < 120 minutes

### Technical Success

- [ ] All 200+ production validation tests passed
- [ ] All 5 hotfixes validated in production
- [ ] SLOs met: Availability ≥99.9%, Latency <100ms, Error Rate <1%
- [ ] Database migrations successful (3 indexes added)
- [ ] Distributed tracing 100% coverage
- [ ] PPO memory reduced 60% (3.7GB → 1.5GB)

### Business Success

- [ ] Zero customer-reported incidents in first 24h
- [ ] Support ticket volume unchanged or decreased
- [ ] Grafana dashboard load time improved 5x (15s → 3s)
- [ ] WebSocket reconnection uptime ≥99.9%
- [ ] No unplanned downtime

---

## 📞 Escalation Path

### Primary Contacts

| Role | Name | Phone | Slack | PagerDuty |
|------|------|-------|-------|-----------|
| Release Manager | [NAME] | [PHONE] | @release-manager | [SCHEDULE] |
| SRE Lead | [NAME] | [PHONE] | @sre-lead | [SCHEDULE] |
| Engineering Lead | [NAME] | [PHONE] | @eng-lead | [SCHEDULE] |
| QA Lead | [NAME] | [PHONE] | @qa-lead | [SCHEDULE] |
| Product Manager | [NAME] | [PHONE] | @pm | N/A |

### Escalation Levels

1. **L1 (SRE On-Call):** Initial response, basic troubleshooting
2. **L2 (SRE Lead):** Complex issues, rollback decisions
3. **L3 (Engineering Lead):** Critical bugs, architecture issues
4. **L4 (Release Manager):** Go/No-Go decisions, stakeholder communication
5. **L5 (Executive Team):** Customer-impacting outages, PR issues

### PagerDuty Integration

```bash
# Test PagerDuty integration
curl -X POST https://events.pagerduty.com/v2/enqueue \
  -H 'Content-Type: application/json' \
  -d '{
    "routing_key": "YOUR_INTEGRATION_KEY",
    "event_action": "trigger",
    "payload": {
      "summary": "T.A.R.S. v1.0.1 GA Deployment Test Alert",
      "severity": "info",
      "source": "ga-runbook-test"
    }
  }'
```

---

## 📋 Checklists

### Pre-GA Checklist (Complete 24h before GA)

**Technical:**
- [ ] Helm chart package verified (SHA256 checksum)
- [ ] Docker images pulled and verified
- [ ] Database migrations tested in staging
- [ ] Feature flags configured and tested
- [ ] Monitoring dashboards operational
- [ ] PagerDuty integration tested
- [ ] Rollback plan tested in staging
- [ ] Production validation suite passing 100%

**Organizational:**
- [ ] All stakeholders notified of GA window
- [ ] War room scheduled (video call + chat)
- [ ] On-call schedule confirmed
- [ ] Status page update scheduled
- [ ] Customer communication drafted
- [ ] Escalation path confirmed

**Go/No-Go:**
- [ ] Staging deployment successful (100% validation pass)
- [ ] No active P0/P1 incidents
- [ ] Production baseline healthy (99.9%+ availability)
- [ ] All approvals obtained

### During GA Checklist

**Stage 0: Pre-Deployment Migrations**
- [ ] Database migrations applied
- [ ] Indexes created and verified
- [ ] No performance impact

**Stage 1: Canary 1%**
- [ ] Canary pods Running
- [ ] Availability ≥99%
- [ ] Error rate <2%
- [ ] Latency p95 <120ms
- [ ] No crash loops

**Stage 2: Canary 10%**
- [ ] Availability ≥99.5%
- [ ] Error rate <1.5%
- [ ] Latency p95 <110ms
- [ ] Drift from stable <10%
- [ ] WebSocket reconnection working

**Stage 3: Canary 25%**
- [ ] Availability ≥99.9%
- [ ] Error rate <1%
- [ ] Latency p95 <100ms
- [ ] Drift from stable <5%
- [ ] All hotfixes validated

**Stage 4: Canary 50%**
- [ ] Availability ≥99.9%
- [ ] Error rate <0.5%
- [ ] Latency p95 <100ms
- [ ] No resource contention
- [ ] No customer-reported issues

**Stage 5: Canary 100%**
- [ ] All pods Running
- [ ] All replicas ready
- [ ] Full validation suite passed (200+ tests)
- [ ] SLOs met

**Stage 6: Post-Deployment**
- [ ] Post-deployment migrations (if any)
- [ ] Status page updated
- [ ] Stakeholders notified
- [ ] 24h monitoring started

### Post-GA Checklist (First 24 Hours)

**+2 Hours:**
- [ ] First 2h metrics reviewed
- [ ] No anomalies detected
- [ ] SLOs on track

**+4 Hours:**
- [ ] Customer feedback reviewed
- [ ] Support tickets checked
- [ ] No widespread issues

**+8 Hours:**
- [ ] Performance analysis complete
- [ ] Resource utilization optimal
- [ ] No memory leaks detected

**+12 Hours:**
- [ ] Mid-point SLO report generated
- [ ] All targets met
- [ ] Trend analysis positive

**+24 Hours:**
- [ ] Full 24h GA Day Report generated
- [ ] Retrospective scheduled
- [ ] Lessons learned documented
- [ ] Next steps planned

---

## 🔗 Reference Links

### Documentation
- **Production Pipeline:** [production_deploy_pipeline.yaml](production_deploy_pipeline.yaml)
- **Validation Suite:** [production_validation_suite.py](production_validation_suite.py)
- **Release Notes:** [PRODUCTION_RELEASE_NOTES.md](PRODUCTION_RELEASE_NOTES.md)
- **Monitoring Plan:** [production_monitoring_plan.md](production_monitoring_plan.md)
- **Quick Start:** [PHASE14_3_QUICKSTART.md](../PHASE14_3_QUICKSTART.md)

### Dashboards
- **Grafana Overview:** https://tars.ai/grafana/d/tars-overview
- **Prometheus Alerts:** https://tars.ai/prometheus/alerts
- **Jaeger Tracing:** https://tars.ai/jaeger
- **Kubernetes Dashboard:** kubectl proxy → http://localhost:8001/...

### Tools
- **GitHub Actions:** https://github.com/YOUR_ORG/tars/actions
- **PagerDuty:** https://YOUR_ORG.pagerduty.com
- **Status Page:** https://status.tars.ai
- **Slack:** #tars-deployments, #tars-support

---

## 📝 Post-GA Actions

### Immediate (0-2 Hours)

1. **Generate GA Day Report**
   ```bash
   python scripts/generate_ga_certification_package.py \
     --ga-start-time="[START_TIME]" \
     --ga-end-time="[END_TIME]" \
     --output-dir=release/v1_0_1/GA_CERTIFICATION/
   ```

2. **Review Initial Metrics**
   - Check Grafana dashboards
   - Review PagerDuty alerts (should be 0)
   - Monitor #tars-support for issues

3. **Stakeholder Update**
   - Send success notification (see Communication section)
   - Update status page to "All Systems Operational"

### Short-Term (2-24 Hours)

1. **Continuous Monitoring**
   - Run 24h KPI collector
   - Monitor drift from staging
   - Track customer feedback

2. **Performance Analysis**
   - API latency trends
   - Resource utilization
   - Error rate analysis

3. **Customer Communication**
   - Send release announcement email
   - Update documentation site
   - Monitor social media/forums

### Long-Term (1-7 Days)

1. **Post-Deployment Review**
   - Schedule retrospective meeting
   - Document lessons learned
   - Update runbooks based on experience

2. **Stability Assessment**
   - 7-day SLO compliance report
   - Capacity planning update
   - Cost analysis

3. **Future Planning**
   - Plan v1.1.0 features
   - Address technical debt identified during GA
   - Update CI/CD pipeline based on learnings

---

## 🎓 Lessons Learned Template

**Fill out within 48h of GA:**

### What Went Well
1.
2.
3.

### What Could Be Improved
1.
2.
3.

### Unexpected Issues
1.
2.
3.

### Action Items for Next Release
- [ ] Action 1 - Owner: [NAME] - Due: [DATE]
- [ ] Action 2 - Owner: [NAME] - Due: [DATE]
- [ ] Action 3 - Owner: [NAME] - Due: [DATE]

---

**Document Version:** 1.0
**Last Updated:** 2025-11-20
**Next Review:** Post-GA Retrospective

🚀 Generated with [Claude Code](https://claude.com/claude-code)
