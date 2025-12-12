# T.A.R.S. v1.0.1 Upgrade Playbook

**Version:** v1.0.1
**Release Date:** 2025-11-20
**Target Environment:** Production Multi-Region Kubernetes Cluster
**Downtime:** Zero (rolling update with canary deployment)
**Rollback Time:** <5 minutes

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Pre-Upgrade Checklist](#pre-upgrade-checklist)
3. [Rollout Strategy](#rollout-strategy)
4. [Detailed Upgrade Procedures](#detailed-upgrade-procedures)
5. [Canary Deployment](#canary-deployment)
6. [Post-Upgrade Validation](#post-upgrade-validation)
7. [Rollback Procedures](#rollback-procedures)
8. [Incident Response](#incident-response)
9. [Monitoring & Alerts](#monitoring--alerts)
10. [Sign-Off Checklist](#sign-off-checklist)

---

## Executive Summary

### What's Included in v1.0.1

This release addresses five critical hotfixes:

| Issue ID | Description | Impact | Risk Level |
|----------|-------------|--------|------------|
| **TARS-1001** | WebSocket reconnection fix | Manual refresh rate: 15% → <1% | Low ✅ |
| **TARS-1002** | Grafana query optimization | Dashboard load: 15s → 4.5s | Low ✅ |
| **TARS-1004** | Database index optimization | API p95: 500ms → <100ms | Medium ⚠️ |
| **TARS-1003** | Jaeger trace context fix | Trace continuity: ~60% → 100% | Medium ⚠️ |
| **TARS-1005** | PPO memory leak fix | Memory (24h): 4GB+ → <1GB | High 🔴 |

### Performance Improvements

- **Dashboard Performance:** 70% faster (15s → 4.5s)
- **Query Execution:** 97% faster (5000ms → 150ms)
- **API Latency:** 80% faster (500ms → <100ms p95)
- **PPO Memory:** 80% reduction (4GB → <1GB @ 24h)
- **WebSocket Reconnection:** Automated (<5s avg vs manual)
- **Trace Continuity:** 67% improvement (~60% → 100%)

### Deployment Window

- **Duration:** 45-60 minutes (full multi-region deployment)
- **Recommended Time:** Low-traffic window (02:00-04:00 UTC)
- **Team Availability:** SRE + Engineering on-call required

### Prerequisites

- Kubernetes 1.24+
- Helm 3.10+
- PostgreSQL 14+ with pg_stat_statements enabled
- Prometheus with recording rules support
- Redis 6.2+ for rate limiting
- Jaeger 1.35+ for distributed tracing

---

## Pre-Upgrade Checklist

### 1. Environment Validation (T-24 hours)

#### Infrastructure Health

```bash
# 1. Check cluster health
kubectl get nodes -o wide
kubectl top nodes

# Expected: All nodes Ready, CPU <70%, Memory <80%

# 2. Verify all services running
kubectl get pods -n tars-production -o wide

# Expected: All pods Running, no CrashLoopBackOff

# 3. Check resource capacity
kubectl describe nodes | grep -A5 "Allocated resources"

# Expected: >30% CPU and memory headroom for rolling updates

# 4. Verify storage availability
kubectl get pv,pvc -n tars-production

# Expected: All PVCs Bound, >20% free space
```

#### Database Readiness

```bash
# 1. Check PostgreSQL version and health
psql -h $DB_HOST -U $DB_USER -d tars -c "SELECT version();"
psql -h $DB_HOST -U $DB_USER -d tars -c "SELECT pg_database_size('tars') / (1024*1024*1024.0) AS size_gb;"

# Expected: PostgreSQL 14+, <80% disk usage

# 2. Verify pg_stat_statements extension
psql -h $DB_HOST -U $DB_USER -d tars -c "SELECT * FROM pg_extension WHERE extname = 'pg_stat_statements';"

# Expected: Extension present

# 3. Check database connections
psql -h $DB_HOST -U $DB_USER -d tars -c "SELECT count(*) FROM pg_stat_activity WHERE datname = 'tars';"

# Expected: <80% of max_connections

# 4. Run ANALYZE for query planner
psql -h $DB_HOST -U $DB_USER -d tars -c "ANALYZE;"

# Expected: Completes without errors
```

#### Prometheus & Grafana

```bash
# 1. Validate Prometheus config
kubectl exec -n monitoring prometheus-0 -- promtool check config /etc/prometheus/prometheus.yml

# Expected: SUCCESS

# 2. Check recording rules support
kubectl exec -n monitoring prometheus-0 -- promtool check rules /etc/prometheus/rules/*.yaml

# Expected: No errors

# 3. Verify Grafana datasource
curl -H "Authorization: Bearer $GRAFANA_API_KEY" http://grafana:3000/api/datasources

# Expected: Prometheus datasource healthy

# 4. Check disk space for metrics
kubectl exec -n monitoring prometheus-0 -- df -h /prometheus

# Expected: >30% free space
```

#### Redis Health

```bash
# 1. Check Redis connectivity
kubectl exec -n tars-production redis-0 -- redis-cli ping

# Expected: PONG

# 2. Verify memory usage
kubectl exec -n tars-production redis-0 -- redis-cli info memory | grep used_memory_human

# Expected: <70% of maxmemory

# 3. Check persistence
kubectl exec -n tars-production redis-0 -- redis-cli lastsave

# Expected: Recent timestamp

# 4. Test rate limiter keys
kubectl exec -n tars-production redis-0 -- redis-cli keys "rate_limit:*" | wc -l

# Expected: Keys present (rate limiting active)
```

### 2. Backup & Recovery Preparation (T-2 hours)

#### Database Backup

```bash
# 1. Create full database backup
export BACKUP_FILE="tars_v1_0_0_backup_$(date +%Y%m%d_%H%M%S).sql"
pg_dump -h $DB_HOST -U $DB_USER -d tars -F c -f /backups/$BACKUP_FILE

# 2. Verify backup integrity
pg_restore --list /backups/$BACKUP_FILE | head -n 20

# 3. Upload to S3 (or backup storage)
aws s3 cp /backups/$BACKUP_FILE s3://tars-backups/pre-upgrade/

# 4. Record backup details
echo "Backup: $BACKUP_FILE, Size: $(du -h /backups/$BACKUP_FILE)" >> upgrade_log.txt
```

#### Kubernetes State Backup

```bash
# 1. Export all Kubernetes resources
kubectl get all -n tars-production -o yaml > k8s_state_pre_upgrade.yaml

# 2. Export ConfigMaps and Secrets (encrypted)
kubectl get configmap -n tars-production -o yaml > configmaps_pre_upgrade.yaml
kubectl get secret -n tars-production -o yaml > secrets_pre_upgrade.yaml.enc

# 3. Export PVCs
kubectl get pvc -n tars-production -o yaml > pvcs_pre_upgrade.yaml

# 4. Backup Helm release values
helm get values tars -n tars-production > helm_values_v1_0_0.yaml
```

#### Redis Snapshot

```bash
# 1. Trigger RDB snapshot
kubectl exec -n tars-production redis-0 -- redis-cli BGSAVE

# 2. Wait for completion
kubectl exec -n tars-production redis-0 -- redis-cli LASTSAVE

# 3. Copy snapshot to backup location
kubectl cp tars-production/redis-0:/data/dump.rdb ./redis_backup_$(date +%Y%m%d_%H%M%S).rdb
```

#### Prometheus Data Snapshot

```bash
# 1. Create Prometheus snapshot
kubectl exec -n monitoring prometheus-0 -- curl -XPOST http://localhost:9090/api/v1/admin/tsdb/snapshot

# 2. Extract snapshot path from response
# Example: {"status":"success","data":{"name":"20251120T120000Z-abcd1234"}}

# 3. Archive snapshot
kubectl exec -n monitoring prometheus-0 -- tar -czf /tmp/prometheus_snapshot.tar.gz /prometheus/snapshots/<snapshot-name>
kubectl cp monitoring/prometheus-0:/tmp/prometheus_snapshot.tar.gz ./prometheus_backup_$(date +%Y%m%d_%H%M%S).tar.gz
```

### 3. Communication & Coordination (T-1 hour)

#### Stakeholder Notification

```bash
# 1. Post maintenance notice (if using Statuspage)
curl -X POST https://api.statuspage.io/v1/pages/$PAGE_ID/incidents \
  -H "Authorization: OAuth $STATUSPAGE_TOKEN" \
  -d '{
    "incident": {
      "name": "T.A.R.S. v1.0.1 Upgrade (Zero Downtime)",
      "status": "scheduled",
      "impact_override": "maintenance",
      "scheduled_for": "2025-11-20T02:00:00Z",
      "scheduled_until": "2025-11-20T03:00:00Z",
      "body": "We are upgrading T.A.R.S. to v1.0.1 with performance improvements. No downtime expected."
    }
  }'

# 2. Notify on-call team
# Send PagerDuty alert or Slack message
curl -X POST https://hooks.slack.com/services/$SLACK_WEBHOOK \
  -d '{
    "text": "🚀 T.A.R.S. v1.0.1 upgrade starting at 02:00 UTC. Team: SRE + Engineering on-call. Playbook: https://docs.tars.io/upgrades/v1.0.1"
  }'
```

#### Team Coordination

- **Upgrade Lead:** Coordinates all steps, owns final go/no-go decision
- **SRE On-Call:** Monitors infrastructure, handles rollback if needed
- **Engineering On-Call:** Troubleshoots application issues
- **Database Admin:** Manages index migrations, monitors query performance
- **Observability Lead:** Watches Grafana dashboards, validates traces

#### Rollback Criteria (Pre-Agreed)

Automatic rollback triggers:
- **Error rate >5%** for 5 consecutive minutes
- **API p95 latency >200ms** for 10 minutes
- **Database CPU >90%** for 5 minutes
- **Pod crash loop** in any service
- **Memory usage >4GB** in PPO agent after 30 minutes

Manual rollback triggers:
- Critical functionality broken
- Data corruption detected
- Upgrade Lead decision

---

## Rollout Strategy

### Multi-Region Rollout Order

T.A.R.S. uses an active-active multi-region architecture. Upgrade order:

1. **Staging Environment** (all regions) - Full validation
2. **Production: us-west-2** (10% canary → 50% → 100%)
3. **Production: us-east-1** (after us-west-2 stable for 30 minutes)
4. **Production: eu-west-1** (after us-east-1 stable for 30 minutes)
5. **Production: ap-southeast-1** (after eu-west-1 stable for 30 minutes)

### Canary Deployment Strategy

For each region:

```
Phase 1: Database Migration (10 minutes)
  ├── Apply indexes with CONCURRENTLY
  ├── Validate query performance
  └── NO application restarts yet

Phase 2: Canary (10% traffic, 15 minutes)
  ├── Deploy 1 pod per service with v1.0.1
  ├── Route 10% traffic to canary pods
  ├── Monitor error rate, latency, memory
  └── GO/NO-GO decision

Phase 3: Gradual Rollout (50% traffic, 15 minutes)
  ├── Scale canary to 50% of pods
  ├── Monitor for 15 minutes
  └── GO/NO-GO decision

Phase 4: Full Rollout (100% traffic, 10 minutes)
  ├── Update all remaining pods
  ├── Drain old pods gracefully
  └── Final validation

Phase 5: Post-Deployment (30 minutes)
  ├── Run regression suite
  ├── Validate SLOs
  ├── Update Statuspage
  └── Sign-off
```

### Traffic Shifting Mechanism

Using Istio VirtualService for canary routing:

```yaml
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: dashboard-api-canary
  namespace: tars-production
spec:
  hosts:
    - dashboard-api.tars-production.svc.cluster.local
  http:
    - match:
        - headers:
            x-canary:
              exact: "true"
      route:
        - destination:
            host: dashboard-api.tars-production.svc.cluster.local
            subset: v1-0-1
          weight: 100
    - route:
        - destination:
            host: dashboard-api.tars-production.svc.cluster.local
            subset: v1-0-1
          weight: 10  # Start at 10%, increase to 50%, then 100%
        - destination:
            host: dashboard-api.tars-production.svc.cluster.local
            subset: v1-0-0
          weight: 90
```

---

## Detailed Upgrade Procedures

### Phase 1: Database Index Migration (Zero Downtime)

**Duration:** 10-15 minutes per region
**Risk Level:** Medium ⚠️
**Rollback:** Automatic (indexes can be dropped without affecting queries)

#### Step 1.1: Validate Migration Script

```bash
# 1. Review migration SQL
cat fixes/fix_database_indexes/v1_0_1_add_indexes.sql

# 2. Dry-run validation (syntax check)
psql -h $DB_HOST -U $DB_USER -d tars --dry-run -f fixes/fix_database_indexes/v1_0_1_add_indexes.sql

# Expected: No syntax errors
```

#### Step 1.2: Apply Indexes with CONCURRENTLY

```bash
# 1. Connect to primary database
psql -h $DB_PRIMARY_HOST -U $DB_USER -d tars

# 2. Apply migration (CONCURRENTLY = no table locks)
\i fixes/fix_database_indexes/v1_0_1_add_indexes.sql

# Expected output:
# CREATE INDEX CONCURRENTLY idx_evaluations_agent_region_time
# CREATE INDEX CONCURRENTLY idx_training_steps_composite
# CREATE INDEX CONCURRENTLY idx_api_keys_user_active

# 3. Verify indexes created
SELECT
    schemaname,
    tablename,
    indexname,
    indexdef
FROM pg_indexes
WHERE indexname IN (
    'idx_evaluations_agent_region_time',
    'idx_training_steps_composite',
    'idx_api_keys_user_active'
);

# Expected: 3 rows returned with correct definitions

# 4. Check index validity (CONCURRENTLY can fail silently)
SELECT
    schemaname,
    tablename,
    indexname,
    pg_index.indisvalid
FROM pg_indexes
JOIN pg_class ON pg_indexes.indexname = pg_class.relname
JOIN pg_index ON pg_class.oid = pg_index.indexrelid
WHERE indexname IN (
    'idx_evaluations_agent_region_time',
    'idx_training_steps_composite',
    'idx_api_keys_user_active'
);

# Expected: All indisvalid = true
```

#### Step 1.3: Validate Query Performance

```bash
# 1. Run benchmark tests
cd fixes/fix_database_indexes
pytest db_index_tests.py::TestDatabaseIndexes::test_composite_index_performance -v

# Expected: p95 latency <100ms (vs 500ms baseline)

# 2. Check EXPLAIN plans
psql -h $DB_HOST -U $DB_USER -d tars -c "
EXPLAIN ANALYZE
SELECT agent_id, AVG(reward) as avg_reward
FROM evaluations
WHERE region = 'us-west-2'
  AND created_at > NOW() - INTERVAL '1 hour'
GROUP BY agent_id;
"

# Expected: Index Scan using idx_evaluations_agent_region_time

# 3. Monitor database load during benchmark
psql -h $DB_HOST -U $DB_USER -d tars -c "
SELECT
    state,
    count(*) as connections
FROM pg_stat_activity
WHERE datname = 'tars'
GROUP BY state;
"

# Expected: No significant increase in active connections
```

#### Step 1.4: Validate Replication to Read Replicas

```bash
# 1. Check replication lag
psql -h $DB_REPLICA_HOST -U $DB_USER -d tars -c "
SELECT
    EXTRACT(EPOCH FROM (NOW() - pg_last_xact_replay_timestamp())) AS replication_lag_seconds;
"

# Expected: <5 seconds

# 2. Verify indexes replicated
psql -h $DB_REPLICA_HOST -U $DB_USER -d tars -c "
SELECT indexname
FROM pg_indexes
WHERE indexname IN (
    'idx_evaluations_agent_region_time',
    'idx_training_steps_composite',
    'idx_api_keys_user_active'
);
"

# Expected: 3 indexes present

# 3. Test query on replica
psql -h $DB_REPLICA_HOST -U $DB_USER -d tars -c "
EXPLAIN (ANALYZE, BUFFERS)
SELECT * FROM evaluations WHERE agent_id = 'ppo' AND region = 'us-west-2' LIMIT 100;
"

# Expected: Uses new index, execution time <100ms
```

### Phase 2: Deploy Prometheus Recording Rules

**Duration:** 5 minutes
**Risk Level:** Low ✅
**Impact:** Improves query performance, no breaking changes

#### Step 2.1: Deploy Recording Rules

```bash
# 1. Validate recording rules syntax
promtool check rules fixes/fix_grafana_query_timeout/recording_rules.yaml

# Expected: SUCCESS

# 2. Deploy to Prometheus via ConfigMap
kubectl create configmap prometheus-recording-rules-v1-0-1 \
  --from-file=v1_0_1_recording_rules.yaml=fixes/fix_grafana_query_timeout/recording_rules.yaml \
  -n monitoring \
  --dry-run=client -o yaml | kubectl apply -f -

# 3. Update Prometheus ConfigMap to include new rules
kubectl patch configmap prometheus-config -n monitoring --type merge -p '{
  "data": {
    "prometheus.yml": "... include new recording rules file ..."
  }
}'

# 4. Reload Prometheus configuration (graceful, no restart)
kubectl exec -n monitoring prometheus-0 -- kill -HUP 1

# 5. Verify reload successful
kubectl logs -n monitoring prometheus-0 --tail=50 | grep "Completed loading of configuration file"

# Expected: "Completed loading of configuration file" with no errors
```

#### Step 2.2: Validate Recording Rules Evaluation

```bash
# 1. Check rule evaluation status
kubectl exec -n monitoring prometheus-0 -- promtool check config /etc/prometheus/prometheus.yml

# 2. Query for new recording rule metrics (wait 30s for first evaluation)
sleep 30
kubectl exec -n monitoring prometheus-0 -- curl -s 'http://localhost:9090/api/v1/query?query=tars:evaluation_latency:p95:1m' | jq .

# Expected: {"status":"success","data":{"resultType":"vector","result":[...]}}

# 3. Verify all 60+ recording rules present
kubectl exec -n monitoring prometheus-0 -- curl -s 'http://localhost:9090/api/v1/rules' | jq '[.data.groups[].rules[] | select(.name | startswith("tars:"))] | length'

# Expected: 60+ rules

# 4. Check rule evaluation duration
kubectl exec -n monitoring prometheus-0 -- curl -s 'http://localhost:9090/api/v1/query?query=prometheus_rule_evaluation_duration_seconds{rule_group=~"tars_.*"}' | jq .

# Expected: <100ms per rule group
```

#### Step 2.3: Deploy Grafana Dashboard Patch

```bash
# 1. Backup current dashboard
curl -H "Authorization: Bearer $GRAFANA_API_KEY" \
  "http://grafana:3000/api/dashboards/uid/tars-overview" > dashboard_backup_v1_0_0.json

# 2. Apply dashboard patch
curl -X POST -H "Authorization: Bearer $GRAFANA_API_KEY" \
  -H "Content-Type: application/json" \
  -d @fixes/fix_grafana_query_timeout/grafana_dashboard_patch.json \
  "http://grafana:3000/api/dashboards/db"

# Expected: {"status":"success","slug":"tars-overview","version":2}

# 3. Test dashboard load time
time curl -H "Authorization: Bearer $GRAFANA_API_KEY" \
  "http://grafana:3000/api/dashboards/uid/tars-overview" > /dev/null

# Expected: <1s response time (dashboard metadata)

# 4. Validate panel queries use recording rules
curl -H "Authorization: Bearer $GRAFANA_API_KEY" \
  "http://grafana:3000/api/dashboards/uid/tars-overview" | jq '.dashboard.panels[].targets[].expr' | grep "tars:" | wc -l

# Expected: 60+ panels using recording rules
```

### Phase 3: Deploy Application Updates (Rolling Update)

**Duration:** 20-30 minutes (per region)
**Risk Level:** Medium ⚠️ (TARS-1005 PPO memory fix is critical)
**Strategy:** Canary deployment with traffic shifting

#### Step 3.1: Update Helm Chart Values

```bash
# 1. Create v1.0.1 values overlay
cat > helm_values_v1_0_1.yaml <<EOF
# T.A.R.S. v1.0.1 Helm Values Overlay

global:
  image:
    tag: v1.0.1
    pullPolicy: IfNotPresent

# WebSocket reconnection settings (TARS-1001)
dashboard:
  api:
    websocket:
      heartbeat_interval: 30
      heartbeat_timeout: 10
      max_reconnect_attempts: 10
      reconnect_backoff_max: 30

# Jaeger trace context propagation (TARS-1003)
orchestration:
  tracing:
    jaeger_enabled: true
    trace_context_propagation: true
    redis_streams_tracing: true

# PPO memory leak fix (TARS-1005)
agents:
  ppo:
    memory:
      buffer_max_size: 10000
      clear_frequency: 1000
      tensorflow_graph_cleanup: true
    resources:
      limits:
        memory: 2Gi  # Reduced from 4Gi
      requests:
        memory: 512Mi

# Canary deployment settings
canary:
  enabled: true
  traffic_weight: 10  # Start at 10%
  analysis:
    interval: 5m
    threshold:
      error_rate: 0.05
      latency_p95: 200
      memory_usage: 2048

# HPA adjustments for v1.0.1
hpa:
  minReplicas: 3  # Increased for canary
  maxReplicas: 12

# Monitoring annotations
podAnnotations:
  prometheus.io/scrape: "true"
  version: "v1.0.1"
  hotfixes: "TARS-1001,TARS-1002,TARS-1003,TARS-1004,TARS-1005"
EOF

# 2. Validate Helm chart
helm lint charts/tars -f helm_values_v1_0_1.yaml

# Expected: No errors or warnings

# 3. Dry-run upgrade
helm upgrade tars charts/tars \
  -n tars-production \
  -f helm_values_v1_0_1.yaml \
  --dry-run --debug > helm_dry_run_output.yaml

# 4. Review diff
helm diff upgrade tars charts/tars \
  -n tars-production \
  -f helm_values_v1_0_1.yaml

# Expected: Review image tags, resource limits, canary settings
```

#### Step 3.2: Deploy Canary (10% Traffic)

```bash
# 1. Deploy canary release
helm upgrade tars charts/tars \
  -n tars-production \
  -f helm_values_v1_0_1.yaml \
  --set canary.enabled=true \
  --set canary.traffic_weight=10 \
  --wait --timeout=10m

# Expected: Releases deployed, 10% traffic to v1.0.1 pods

# 2. Verify canary pods running
kubectl get pods -n tars-production -l version=v1.0.1

# Expected: 1 pod per service (dashboard-api, orchestration, ppo-agent, etc.)

# 3. Check canary pod logs for errors
for pod in $(kubectl get pods -n tars-production -l version=v1.0.1 -o name); do
  echo "=== $pod ==="
  kubectl logs -n tars-production $pod --tail=50 | grep -i error
done

# Expected: No ERROR logs (WARNINGs acceptable if minor)

# 4. Validate traffic distribution
kubectl exec -n tars-production $(kubectl get pod -n tars-production -l app=istio-ingressgateway -o jsonpath='{.items[0].metadata.name}') -- \
  curl -s localhost:15000/stats/prometheus | grep "cluster.outbound.*tars.*version.*v1_0_1.*cx_active"

# Expected: ~10% of connections to v1.0.1 pods
```

#### Step 3.3: Canary Health Monitoring (15 minutes)

```bash
# 1. Monitor error rate
kubectl exec -n monitoring prometheus-0 -- curl -s 'http://localhost:9090/api/v1/query?query=sum(rate(http_requests_total{code=~"5..",version="v1.0.1"}[5m])) / sum(rate(http_requests_total{version="v1.0.1"}[5m]))' | jq -r '.data.result[0].value[1]'

# Expected: <0.05 (5% error rate threshold)

# 2. Monitor API latency
kubectl exec -n monitoring prometheus-0 -- curl -s 'http://localhost:9090/api/v1/query?query=histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket{version="v1.0.1"}[5m])) by (le))' | jq -r '.data.result[0].value[1]'

# Expected: <0.2 (200ms p95 threshold)

# 3. Monitor PPO agent memory
kubectl exec -n monitoring prometheus-0 -- curl -s 'http://localhost:9090/api/v1/query?query=container_memory_usage_bytes{pod=~"ppo-agent.*",version="v1.0.1"} / (1024*1024*1024)' | jq -r '.data.result[0].value[1]'

# Expected: <1.5 GB (vs 4GB+ baseline after 30 min)

# 4. Monitor WebSocket reconnections
kubectl exec -n monitoring prometheus-0 -- curl -s 'http://localhost:9090/api/v1/query?query=rate(websocket_reconnections_total{version="v1.0.1"}[5m])' | jq -r '.data.result[0].value[1]'

# Expected: <0.1 reconnections/sec (low reconnection rate)

# 5. Monitor Jaeger trace continuity
kubectl exec -n monitoring prometheus-0 -- curl -s 'http://localhost:9090/api/v1/query?query=sum(jaeger_traces_with_parent{version="v1.0.1"}) / sum(jaeger_traces_total{version="v1.0.1"})' | jq -r '.data.result[0].value[1]'

# Expected: >0.99 (99%+ traces have parent spans)
```

#### Step 3.4: Run Canary Validation Tests

```bash
# 1. Run targeted tests against canary pods
cd release/v1_0_1
pytest regression_suite_v1_0_1.py::TestCanaryValidation -v \
  --target-version=v1.0.1 \
  --canary-endpoint=http://dashboard-api-canary.tars-production.svc.cluster.local:3001

# Expected: All tests pass

# 2. Test WebSocket reconnection (TARS-1001)
pytest regression_suite_v1_0_1.py::TestWebSocketFix::test_reconnection_e2e -v

# Expected: PASSED - Reconnection <5s

# 3. Test Grafana query performance (TARS-1002)
pytest regression_suite_v1_0_1.py::TestGrafanaFix::test_dashboard_load_time -v

# Expected: PASSED - Dashboard loads <5s @ 5000+ evals

# 4. Test database performance (TARS-1004)
pytest regression_suite_v1_0_1.py::TestDatabaseFix::test_api_latency_p95 -v

# Expected: PASSED - p95 <100ms

# 5. Test PPO memory stability (TARS-1005) - 30-minute soak
pytest regression_suite_v1_0_1.py::TestPPOFix::test_memory_stability_accelerated -v

# Expected: PASSED - Memory <1GB after 30 min
```

#### Step 3.5: GO/NO-GO Decision (Canary Phase)

**Decision Maker:** Upgrade Lead
**Decision Criteria:**

| Metric | Threshold | Actual | Status |
|--------|-----------|--------|--------|
| Error rate | <5% | ___% | ☐ PASS ☐ FAIL |
| API p95 latency | <200ms | ___ms | ☐ PASS ☐ FAIL |
| PPO memory (30 min) | <2GB | ___GB | ☐ PASS ☐ FAIL |
| WebSocket reconnects | <0.1/sec | ___/sec | ☐ PASS ☐ FAIL |
| Trace continuity | >99% | ___% | ☐ PASS ☐ FAIL |
| Test suite pass rate | 100% | ___% | ☐ PASS ☐ FAIL |
| Pod stability | No crashes | ___ crashes | ☐ PASS ☐ FAIL |

**Decision:**
- ☐ **GO** - Proceed to 50% traffic
- ☐ **NO-GO** - Rollback to v1.0.0

**Signature:** _______________ **Time:** _______

#### Step 3.6: Scale to 50% Traffic

```bash
# 1. Increase canary traffic to 50%
helm upgrade tars charts/tars \
  -n tars-production \
  -f helm_values_v1_0_1.yaml \
  --set canary.traffic_weight=50 \
  --wait --timeout=10m

# 2. Verify traffic distribution
kubectl exec -n tars-production $(kubectl get pod -n tars-production -l app=istio-ingressgateway -o jsonpath='{.items[0].metadata.name}') -- \
  curl -s localhost:15000/stats/prometheus | grep "cluster.outbound.*tars.*version.*cx_active"

# Expected: ~50/50 split between v1.0.0 and v1.0.1

# 3. Monitor for 15 minutes (same metrics as Step 3.3)
# Use automated monitoring script
./scripts/monitor_canary.sh --duration=15m --threshold-error-rate=0.05

# Expected: All metrics within thresholds
```

#### Step 3.7: GO/NO-GO Decision (50% Traffic)

**Decision Maker:** Upgrade Lead
**Decision Criteria:** Same as Step 3.5

**Decision:**
- ☐ **GO** - Proceed to 100% traffic
- ☐ **NO-GO** - Rollback to v1.0.0

**Signature:** _______________ **Time:** _______

#### Step 3.8: Complete Rollout (100% Traffic)

```bash
# 1. Scale to 100% v1.0.1
helm upgrade tars charts/tars \
  -n tars-production \
  -f helm_values_v1_0_1.yaml \
  --set canary.enabled=false \
  --wait --timeout=10m

# 2. Verify all pods updated
kubectl get pods -n tars-production -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.spec.containers[0].image}{"\n"}{end}' | grep tars

# Expected: All pods using v1.0.1 images

# 3. Drain old v1.0.0 pods gracefully
kubectl delete pods -n tars-production -l version=v1.0.0 --grace-period=60

# 4. Verify no v1.0.0 pods remain
kubectl get pods -n tars-production -l version=v1.0.0

# Expected: No resources found
```

---

## Canary Deployment

### Automated Canary Analysis

T.A.R.S. uses Flagger for automated canary analysis:

```yaml
apiVersion: flagger.app/v1beta1
kind: Canary
metadata:
  name: dashboard-api
  namespace: tars-production
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: dashboard-api
  service:
    port: 3001
  analysis:
    interval: 5m
    threshold: 3  # Number of failed checks before rollback
    maxWeight: 50
    stepWeight: 10
    metrics:
      - name: request-success-rate
        thresholdRange:
          min: 95  # 95% success rate required
        interval: 1m
      - name: request-duration-p95
        thresholdRange:
          max: 200  # 200ms p95 latency max
        interval: 1m
      - name: memory-usage
        thresholdRange:
          max: 2048  # 2GB max memory
        interval: 1m
    webhooks:
      - name: load-test
        url: http://flagger-loadtester/
        timeout: 5s
        metadata:
          cmd: "hey -z 1m -q 10 -c 2 http://dashboard-api-canary.tars-production:3001/health"
      - name: acceptance-test
        url: http://flagger-loadtester/
        timeout: 30s
        metadata:
          type: bash
          cmd: "pytest /tests/regression_suite_v1_0_1.py::TestCanaryValidation -v"
```

### Manual Canary Validation Checklist

If automated canary analysis is not available, use this checklist:

#### Health Checks (Every 5 Minutes)

```bash
# 1. Check pod health
kubectl get pods -n tars-production -l version=v1.0.1 -o wide

# Expected: All Running, READY 1/1

# 2. Check service endpoints
kubectl get endpoints -n tars-production dashboard-api -o yaml | grep -A5 v1.0.1

# Expected: v1.0.1 pods in endpoint list

# 3. Test health endpoint
for pod in $(kubectl get pods -n tars-production -l version=v1.0.1,app=dashboard-api -o name); do
  kubectl exec -n tars-production $pod -- curl -s http://localhost:3001/health | jq .
done

# Expected: {"status":"ok","version":"v1.0.1"}
```

#### Performance Checks (Every 5 Minutes)

```bash
# 1. Error rate
kubectl exec -n monitoring prometheus-0 -- curl -s 'http://localhost:9090/api/v1/query?query=sum(rate(http_requests_total{code=~"5..",version="v1.0.1"}[5m])) / sum(rate(http_requests_total{version="v1.0.1"}[5m])) * 100' | jq -r '.data.result[0].value[1]'

# 2. Latency p50, p95, p99
kubectl exec -n monitoring prometheus-0 -- curl -s 'http://localhost:9090/api/v1/query?query=histogram_quantile(0.50, sum(rate(http_request_duration_seconds_bucket{version="v1.0.1"}[5m])) by (le)) * 1000' | jq -r '.data.result[0].value[1]'
kubectl exec -n monitoring prometheus-0 -- curl -s 'http://localhost:9090/api/v1/query?query=histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket{version="v1.0.1"}[5m])) by (le)) * 1000' | jq -r '.data.result[0].value[1]'
kubectl exec -n monitoring prometheus-0 -- curl -s 'http://localhost:9090/api/v1/query?query=histogram_quantile(0.99, sum(rate(http_request_duration_seconds_bucket{version="v1.0.1"}[5m])) by (le)) * 1000' | jq -r '.data.result[0].value[1]'

# 3. Memory usage
kubectl top pods -n tars-production -l version=v1.0.1

# 4. CPU usage
kubectl top pods -n tars-production -l version=v1.0.1 --sort-by=cpu
```

---

## Post-Upgrade Validation

### Phase 4: Comprehensive Testing (30 minutes)

#### Step 4.1: Run Full Regression Suite

```bash
# 1. Run all v1.0.1 regression tests
cd release/v1_0_1
pytest regression_suite_v1_0_1.py -v --html=report.html --self-contained-html

# Expected: All tests pass (100% pass rate required)

# 2. Generate test report
cat report.html | grep "tests passed"

# Expected: "205 tests passed"

# 3. Upload test results
aws s3 cp report.html s3://tars-artifacts/v1.0.1/regression_report_$(date +%Y%m%d_%H%M%S).html
```

#### Step 4.2: Validate Performance Improvements

```bash
# 1. Test WebSocket reconnection (TARS-1001)
pytest regression_suite_v1_0_1.py::TestWebSocketFix::test_reconnection_benchmark -v

# Expected: Average reconnection time <5s

# 2. Test Grafana query performance (TARS-1002)
pytest regression_suite_v1_0_1.py::TestGrafanaFix::test_query_execution_time -v

# Expected: Query execution <150ms (vs 5000ms baseline)

# 3. Test dashboard load time (TARS-1002)
pytest regression_suite_v1_0_1.py::TestGrafanaFix::test_dashboard_load_time -v

# Expected: Dashboard loads <5s @ 5000+ evaluations

# 4. Test database API latency (TARS-1004)
pytest regression_suite_v1_0_1.py::TestDatabaseFix::test_api_latency_p95 -v

# Expected: p95 latency <100ms (vs 500ms baseline)

# 5. Test API key authentication (TARS-1004)
pytest regression_suite_v1_0_1.py::TestDatabaseFix::test_api_key_auth_performance -v

# Expected: Auth latency <5ms (vs 150ms baseline)

# 6. Test Jaeger trace continuity (TARS-1003)
pytest regression_suite_v1_0_1.py::TestJaegerFix::test_trace_continuity -v

# Expected: 100% parent-child span linking (vs ~60% baseline)

# 7. Test multi-region trace propagation (TARS-1003)
pytest regression_suite_v1_0_1.py::TestJaegerFix::test_multi_region_traces -v

# Expected: Traces span all regions with correct context

# 8. Test PPO memory stability (TARS-1005) - Full 48-hour test
# This runs in background, monitor via Grafana
pytest regression_suite_v1_0_1.py::TestPPOFix::test_memory_stability_full --run-async

# Expected: Memory <1GB @ 48 hours (vs 4GB+ baseline)
```

#### Step 4.3: Validate SLOs

```bash
# 1. Check API latency SLO (<150ms p95)
kubectl exec -n monitoring prometheus-0 -- curl -s 'http://localhost:9090/api/v1/query?query=tars:slo_compliance:api_latency' | jq -r '.data.result[0].value[1]'

# Expected: 1.0 (100% compliance)

# 2. Check error rate SLO (<1%)
kubectl exec -n monitoring prometheus-0 -- curl -s 'http://localhost:9090/api/v1/query?query=tars:slo_compliance:error_rate' | jq -r '.data.result[0].value[1]'

# Expected: 1.0 (100% compliance)

# 3. Check evaluation success SLO (>99%)
kubectl exec -n monitoring prometheus-0 -- curl -s 'http://localhost:9090/api/v1/query?query=tars:slo_compliance:evaluation_success' | jq -r '.data.result[0].value[1]'

# Expected: 1.0 (100% compliance)

# 4. Generate SLO report
kubectl exec -n monitoring prometheus-0 -- curl -s 'http://localhost:9090/api/v1/query?query={__name__=~"tars:slo_compliance:.*"}' | jq -r '.data.result[] | "\(.metric.__name__): \(.value[1])"'

# Expected: All SLOs at 1.0 (100%)
```

#### Step 4.4: Validate End-to-End Workflows

```bash
# 1. Test full evaluation pipeline
pytest regression_suite_v1_0_1.py::TestEndToEnd::test_evaluation_pipeline -v

# Expected: PASSED - Full pipeline from agent training → evaluation → metrics → dashboard

# 2. Test multi-region replication
pytest regression_suite_v1_0_1.py::TestEndToEnd::test_multi_region_replication -v

# Expected: PASSED - Data replicates across all regions <5s

# 3. Test API authentication flow
pytest regression_suite_v1_0_1.py::TestEndToEnd::test_auth_flow -v

# Expected: PASSED - JWT auth, rate limiting, RBAC work correctly

# 4. Test WebSocket real-time updates
pytest regression_suite_v1_0_1.py::TestEndToEnd::test_websocket_realtime -v

# Expected: PASSED - Dashboard receives real-time updates via WebSocket
```

#### Step 4.5: Smoke Test Critical Paths

```bash
# 1. Test Dashboard UI
curl -I http://dashboard.tars.production/

# Expected: 200 OK, response time <1s

# 2. Test API health endpoints
for service in dashboard-api orchestration insight-engine adaptive-policy meta-consensus; do
  echo "Testing $service..."
  curl -s http://$service.tars-production.svc.cluster.local:8080/health | jq .
done

# Expected: All services return {"status":"ok","version":"v1.0.1"}

# 3. Test agent training
curl -X POST http://orchestration.tars-production.svc.cluster.local:8094/api/v1/train \
  -H "Content-Type: application/json" \
  -d '{"agent_id":"dqn","episodes":10}' | jq .

# Expected: 200 OK, training job started

# 4. Test evaluation submission
curl -X POST http://dashboard-api.tars-production.svc.cluster.local:3001/api/v1/evaluations \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TEST_JWT" \
  -d '{"agent_id":"ppo","environment":"CartPole-v1","reward":150}' | jq .

# Expected: 201 Created, evaluation stored
```

### Phase 5: Observability Validation

#### Step 5.1: Verify Prometheus Metrics

```bash
# 1. Check all services exporting metrics
kubectl exec -n monitoring prometheus-0 -- curl -s 'http://localhost:9090/api/v1/targets' | jq '[.data.activeTargets[] | select(.labels.version=="v1.0.1")] | length'

# Expected: 9 targets (one per service)

# 2. Verify recording rules evaluating
kubectl exec -n monitoring prometheus-0 -- curl -s 'http://localhost:9090/api/v1/rules?type=record' | jq '[.data.groups[].rules[] | select(.name | startswith("tars:"))] | length'

# Expected: 60+ recording rules

# 3. Check metric cardinality
kubectl exec -n monitoring prometheus-0 -- curl -s 'http://localhost:9090/api/v1/label/__name__/values' | jq '[.data[] | select(startswith("tars:"))] | length'

# Expected: <200 unique metric names (controlled cardinality)

# 4. Verify no stale metrics
kubectl exec -n monitoring prometheus-0 -- curl -s 'http://localhost:9090/api/v1/query?query=up{version="v1.0.0"}' | jq .

# Expected: No results (all v1.0.0 pods terminated)
```

#### Step 5.2: Verify Grafana Dashboards

```bash
# 1. Test dashboard load time
time curl -H "Authorization: Bearer $GRAFANA_API_KEY" \
  "http://grafana:3000/d/tars-overview/tars-overview?orgId=1&refresh=10s" > /dev/null

# Expected: <5s (vs 15s baseline)

# 2. Verify all panels loading
curl -H "Authorization: Bearer $GRAFANA_API_KEY" \
  "http://grafana:3000/api/dashboards/uid/tars-overview" | jq '.dashboard.panels | length'

# Expected: 60+ panels

# 3. Test panel query execution
curl -X POST -H "Authorization: Bearer $GRAFANA_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"queries":[{"expr":"tars:evaluation_latency:p95:1m","refId":"A"}],"from":"now-1h","to":"now"}' \
  "http://grafana:3000/api/ds/query" | jq '.results.A.frames[0].data.values'

# Expected: Data returned, query execution <200ms

# 4. Verify alerting rules
curl -H "Authorization: Bearer $GRAFANA_API_KEY" \
  "http://grafana:3000/api/v1/provisioning/alert-rules" | jq '[.[] | select(.labels.version=="v1.0.1")] | length'

# Expected: 10+ alert rules for v1.0.1
```

#### Step 5.3: Verify Jaeger Traces

```bash
# 1. Query recent traces
curl "http://jaeger-query:16686/api/traces?service=orchestration&limit=100" | jq '.data | length'

# Expected: 100 traces returned

# 2. Verify trace continuity (parent-child spans)
curl "http://jaeger-query:16686/api/traces?service=orchestration&limit=100" | \
  jq '[.data[] | .spans[] | select(.references | length > 0)] | length'

# Expected: >99% of spans have parent references

# 3. Test multi-region trace propagation
curl "http://jaeger-query:16686/api/traces?service=orchestration&tags=%7B%22region%22%3A%22us-west-2%22%7D&limit=10" | \
  jq '[.data[] | .spans[] | .process.tags[] | select(.key=="region")] | unique | length'

# Expected: Multiple regions in trace (us-west-2, us-east-1, etc.)

# 4. Verify Redis Streams trace context
curl "http://jaeger-query:16686/api/traces?service=redis-consumer&limit=10" | \
  jq '[.data[] | .spans[] | select(.operationName=="redis.stream.consume")] | length'

# Expected: >0 (Redis Streams operations traced)
```

---

## Rollback Procedures

### Automated Rollback (Triggered by Canary Failure)

If canary analysis detects a failure, Flagger automatically rolls back:

```bash
# Monitor Flagger canary status
kubectl get canary -n tars-production dashboard-api -w

# If status = "Failed", rollback is automatic:
# 1. Traffic shifted back to v1.0.0 (100%)
# 2. v1.0.1 pods scaled down to 0
# 3. Alert sent to PagerDuty/Slack

# Verify rollback completed
kubectl get pods -n tars-production -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.spec.containers[0].image}{"\n"}{end}' | grep tars

# Expected: All pods using v1.0.0 images
```

### Manual Rollback

If manual intervention is required:

#### Step 1: Immediate Traffic Shift (< 1 minute)

```bash
# 1. Shift 100% traffic back to v1.0.0
helm upgrade tars charts/tars \
  -n tars-production \
  -f helm_values_v1_0_0.yaml \
  --set canary.traffic_weight=0 \
  --wait --timeout=2m

# 2. Verify traffic shifted
kubectl exec -n tars-production $(kubectl get pod -n tars-production -l app=istio-ingressgateway -o jsonpath='{.items[0].metadata.name}') -- \
  curl -s localhost:15000/stats/prometheus | grep "cluster.outbound.*tars.*version.*cx_active"

# Expected: 100% traffic to v1.0.0 pods
```

#### Step 2: Rollback Application Pods (< 5 minutes)

```bash
# 1. Rollback Helm release
helm rollback tars -n tars-production

# Expected: Rolled back to revision X (v1.0.0)

# 2. Force pod recreation
kubectl delete pods -n tars-production -l version=v1.0.1 --grace-period=0 --force

# 3. Verify all pods running v1.0.0
kubectl get pods -n tars-production -o wide

# Expected: All pods Running with v1.0.0 images
```

#### Step 3: Rollback Database Indexes (Optional)

⚠️ **WARNING:** Only rollback indexes if they cause performance degradation.
Indexes are **read-only** and do not affect data integrity.

```bash
# 1. Drop indexes (if needed)
psql -h $DB_HOST -U $DB_USER -d tars <<EOF
DROP INDEX CONCURRENTLY IF EXISTS idx_evaluations_agent_region_time;
DROP INDEX CONCURRENTLY IF EXISTS idx_training_steps_composite;
DROP INDEX CONCURRENTLY IF EXISTS idx_api_keys_user_active;
EOF

# 2. Verify indexes dropped
psql -h $DB_HOST -U $DB_USER -d tars -c "SELECT indexname FROM pg_indexes WHERE indexname LIKE 'idx_%';"

# Expected: Indexes not listed
```

#### Step 4: Rollback Prometheus Recording Rules

```bash
# 1. Remove recording rules ConfigMap
kubectl delete configmap prometheus-recording-rules-v1-0-1 -n monitoring

# 2. Reload Prometheus config
kubectl exec -n monitoring prometheus-0 -- kill -HUP 1

# 3. Verify old metrics still available
kubectl exec -n monitoring prometheus-0 -- curl -s 'http://localhost:9090/api/v1/query?query=up' | jq .

# Expected: Prometheus healthy, old metrics queryable
```

#### Step 5: Rollback Grafana Dashboard

```bash
# 1. Restore dashboard from backup
curl -X POST -H "Authorization: Bearer $GRAFANA_API_KEY" \
  -H "Content-Type: application/json" \
  -d @dashboard_backup_v1_0_0.json \
  "http://grafana:3000/api/dashboards/db"

# Expected: Dashboard reverted to v1.0.0

# 2. Verify dashboard loads
curl -H "Authorization: Bearer $GRAFANA_API_KEY" \
  "http://grafana:3000/api/dashboards/uid/tars-overview" | jq '.dashboard.version'

# Expected: Version 1 (v1.0.0)
```

#### Step 6: Rollback Validation

```bash
# 1. Test system health
pytest release/v1_0_1/regression_suite_v1_0_1.py::TestRollback -v

# Expected: All tests pass (system restored to v1.0.0)

# 2. Verify SLOs met
kubectl exec -n monitoring prometheus-0 -- curl -s 'http://localhost:9090/api/v1/query?query={__name__=~"tars:slo_compliance:.*"}' | jq -r '.data.result[] | "\(.metric.__name__): \(.value[1])"'

# Expected: All SLOs at 1.0

# 3. Check error rate
kubectl exec -n monitoring prometheus-0 -- curl -s 'http://localhost:9090/api/v1/query?query=sum(rate(http_requests_total{code=~"5.."}[5m])) / sum(rate(http_requests_total[5m])) * 100' | jq -r '.data.result[0].value[1]'

# Expected: <1% error rate
```

### Rollback Communication

```bash
# 1. Update Statuspage
curl -X PATCH https://api.statuspage.io/v1/pages/$PAGE_ID/incidents/$INCIDENT_ID \
  -H "Authorization: OAuth $STATUSPAGE_TOKEN" \
  -d '{
    "incident": {
      "status": "resolved",
      "body": "v1.0.1 upgrade rolled back to v1.0.0. System fully operational. Root cause analysis in progress."
    }
  }'

# 2. Notify team
curl -X POST https://hooks.slack.com/services/$SLACK_WEBHOOK \
  -d '{
    "text": "⚠️ T.A.R.S. v1.0.1 upgrade rolled back to v1.0.0. System stable. RCA doc: https://docs.tars.io/incidents/v1.0.1-rollback"
  }'
```

---

## Incident Response

### Incident Severity Levels

| Level | Definition | Response Time | Escalation |
|-------|------------|---------------|------------|
| **SEV1** 🔴 | System down, data loss, >50% error rate | <5 min | Page all on-call |
| **SEV2** 🟠 | Degraded performance, <50% error rate | <15 min | Page primary on-call |
| **SEV3** 🟡 | Minor issues, SLO breach | <30 min | Slack notification |
| **SEV4** 🟢 | Informational, no user impact | <2 hours | Email notification |

### Common Issues & Resolutions

#### Issue 1: Canary Pods CrashLoopBackOff

**Symptoms:**
- Canary pods restart repeatedly
- Error logs show "Connection refused" or "Port already in use"

**Resolution:**
```bash
# 1. Check pod logs
kubectl logs -n tars-production <pod-name> --tail=100

# 2. Common causes:
#    - Port conflict (check if old pods still running)
#    - Config issue (check ConfigMap/Secret)
#    - Resource limits too low (check OOMKilled)

# 3. Fix resource limits (if OOMKilled)
kubectl edit deployment -n tars-production dashboard-api
# Increase memory limits: 2Gi → 4Gi

# 4. If unfixable, rollback
helm rollback tars -n tars-production
```

#### Issue 2: Database Index Creation Timeout

**Symptoms:**
- `CREATE INDEX CONCURRENTLY` hangs for >30 minutes
- High database CPU usage

**Resolution:**
```bash
# 1. Check index creation progress
psql -h $DB_HOST -U $DB_USER -d tars -c "
SELECT
    schemaname,
    tablename,
    indexname,
    pg_stat_progress_create_index.*
FROM pg_stat_progress_create_index
JOIN pg_stat_activity ON pg_stat_progress_create_index.pid = pg_stat_activity.pid
JOIN pg_indexes ON pg_stat_progress_create_index.relid::regclass::text = pg_indexes.tablename;
"

# 2. If stalled, cancel index creation
psql -h $DB_HOST -U $DB_USER -d tars -c "
SELECT pg_cancel_backend(pid)
FROM pg_stat_activity
WHERE query LIKE '%CREATE INDEX CONCURRENTLY%';
"

# 3. Drop invalid index
psql -h $DB_HOST -U $DB_USER -d tars -c "
DROP INDEX CONCURRENTLY IF EXISTS idx_evaluations_agent_region_time;
"

# 4. Retry with lower maintenance_work_mem
psql -h $DB_HOST -U $DB_USER -d tars -c "SET maintenance_work_mem = '256MB';"
# Then retry CREATE INDEX CONCURRENTLY
```

#### Issue 3: Prometheus Recording Rules Not Evaluating

**Symptoms:**
- Dashboard shows "No data" for recording rule metrics
- Recording rule metrics missing in Prometheus

**Resolution:**
```bash
# 1. Check Prometheus logs
kubectl logs -n monitoring prometheus-0 --tail=100 | grep -i error

# 2. Verify recording rules syntax
promtool check rules fixes/fix_grafana_query_timeout/recording_rules.yaml

# 3. Check rule evaluation failures
kubectl exec -n monitoring prometheus-0 -- curl -s 'http://localhost:9090/api/v1/rules?type=record' | jq '[.data.groups[].rules[] | select(.health=="bad")]'

# 4. If syntax error, fix and reload
kubectl delete configmap prometheus-recording-rules-v1-0-1 -n monitoring
kubectl create configmap prometheus-recording-rules-v1-0-1 --from-file=recording_rules.yaml=fixes/fix_grafana_query_timeout/recording_rules.yaml -n monitoring
kubectl exec -n monitoring prometheus-0 -- kill -HUP 1
```

#### Issue 4: PPO Agent Memory Still High (TARS-1005)

**Symptoms:**
- PPO agent memory >2GB after 30 minutes
- Memory leak fix not working

**Resolution:**
```bash
# 1. Check if patch applied
kubectl exec -n tars-production ppo-agent-0 -- python -c "
import sys
sys.path.insert(0, '/app')
from agents.ppo_agent import PPOAgent
print(PPOAgent.BUFFER_CLEAR_FREQUENCY)
"

# Expected: 1000 (vs default 10000)

# 2. Check TensorFlow graph cleanup
kubectl logs -n tars-production ppo-agent-0 | grep "Clearing TensorFlow graph"

# Expected: Log entries every 1000 training steps

# 3. If not applied, verify image
kubectl get pod -n tars-production ppo-agent-0 -o jsonpath='{.spec.containers[0].image}'

# Expected: tars/ppo-agent:v1.0.1

# 4. Force pod recreation
kubectl delete pod -n tars-production ppo-agent-0

# 5. Monitor memory after restart
watch kubectl top pod -n tars-production ppo-agent-0
```

#### Issue 5: WebSocket Connections Not Reconnecting (TARS-1001)

**Symptoms:**
- Dashboard shows "Disconnected" status
- WebSocket connections drop but don't reconnect

**Resolution:**
```bash
# 1. Check dashboard-api logs
kubectl logs -n tars-production dashboard-api-0 | grep -i websocket

# 2. Test WebSocket endpoint
wscat -c ws://dashboard-api.tars-production.svc.cluster.local:3001/ws

# Expected: Connection established, heartbeat pings every 30s

# 3. Check if client using new reconnection logic
curl http://dashboard.tars.production/static/js/main.js | grep "ReconnectingWebSocketClient"

# Expected: ReconnectingWebSocketClient present

# 4. Clear browser cache (client-side fix)
# Instruct users to hard refresh: Ctrl+Shift+R (Windows) / Cmd+Shift+R (Mac)
```

---

## Monitoring & Alerts

### Key Dashboards

1. **T.A.R.S. Overview Dashboard**
   - URL: http://grafana:3000/d/tars-overview
   - Metrics: Overall system health, request rates, latencies, error rates
   - Refresh: 10s

2. **T.A.R.S. Agent Performance Dashboard**
   - URL: http://grafana:3000/d/tars-agents
   - Metrics: Agent training progress, reward trends, memory usage
   - Refresh: 30s

3. **T.A.R.S. Database Dashboard**
   - URL: http://grafana:3000/d/tars-database
   - Metrics: Query performance, connection pool, index usage
   - Refresh: 15s

4. **T.A.R.S. Canary Dashboard** (during upgrade)
   - URL: http://grafana:3000/d/tars-canary
   - Metrics: Canary vs stable metrics side-by-side
   - Refresh: 5s

### Critical Alerts

| Alert | Condition | Severity | Action |
|-------|-----------|----------|--------|
| **High Error Rate** | >5% for 5min | SEV2 | Rollback canary |
| **API Latency High** | p95 >200ms for 10min | SEV2 | Rollback canary |
| **PPO Memory High** | >2GB for 30min | SEV2 | Investigate memory leak |
| **Database CPU High** | >90% for 5min | SEV2 | Check index creation |
| **Pod CrashLoop** | Any pod restarting >3x | SEV1 | Immediate rollback |
| **Trace Continuity Low** | <99% for 10min | SEV3 | Check Jaeger config |
| **Recording Rules Failing** | >5 rules failing | SEV3 | Check Prometheus config |

### Alert Notification Channels

- **PagerDuty:** SEV1, SEV2 alerts
- **Slack (#tars-alerts):** All severity levels
- **Email (eng-team@):** SEV3, SEV4 alerts
- **Statuspage:** SEV1, SEV2 (public incidents)

---

## Sign-Off Checklist

### Pre-Upgrade Sign-Off

- [ ] **SRE Lead:** Infrastructure health validated
- [ ] **Database Admin:** Database ready for migration
- [ ] **Engineering Lead:** Code review completed
- [ ] **QA Lead:** Staging tests passed
- [ ] **Product Owner:** Release approved

**Signature:** _______________ **Date:** ________ **Time:** ______

### Post-Upgrade Sign-Off

- [ ] **SRE Lead:** All regions upgraded successfully
- [ ] **Database Admin:** Indexes created, performance validated
- [ ] **Engineering Lead:** Regression suite passed 100%
- [ ] **QA Lead:** SLOs met, no critical issues
- [ ] **Observability Lead:** Dashboards and traces validated

**Signature:** _______________ **Date:** ________ **Time:** ______

### Final Release Sign-Off

- [ ] **Release Manager:** All deliverables complete
- [ ] **Security Lead:** No vulnerabilities introduced
- [ ] **Documentation Lead:** Release notes published
- [ ] **Support Lead:** Runbooks updated

**Signature:** _______________ **Date:** ________ **Time:** ______

---

## Appendix

### A. Environment Variables

```bash
# Database
export DB_HOST="postgres.tars-production.svc.cluster.local"
export DB_PRIMARY_HOST="postgres-primary.tars-production.svc.cluster.local"
export DB_REPLICA_HOST="postgres-replica.tars-production.svc.cluster.local"
export DB_USER="tars_admin"
export DB_NAME="tars"

# Redis
export REDIS_HOST="redis.tars-production.svc.cluster.local"
export REDIS_PORT="6379"

# Prometheus
export PROMETHEUS_URL="http://prometheus.monitoring.svc.cluster.local:9090"

# Grafana
export GRAFANA_URL="http://grafana.monitoring.svc.cluster.local:3000"
export GRAFANA_API_KEY="<from-secret>"

# Jaeger
export JAEGER_QUERY_URL="http://jaeger-query.tracing.svc.cluster.local:16686"

# Statuspage
export STATUSPAGE_PAGE_ID="<your-page-id>"
export STATUSPAGE_TOKEN="<from-secret>"

# Slack
export SLACK_WEBHOOK="<from-secret>"

# PagerDuty
export PAGERDUTY_INTEGRATION_KEY="<from-secret>"
```

### B. Useful Commands

```bash
# Watch pod status during rollout
watch kubectl get pods -n tars-production -o wide

# Stream logs from all v1.0.1 pods
kubectl logs -n tars-production -l version=v1.0.1 --tail=10 -f

# Check Helm release history
helm history tars -n tars-production

# Get current canary traffic weight
kubectl get virtualservice dashboard-api-canary -n tars-production -o jsonpath='{.spec.http[0].route[0].weight}'

# Check database connections
psql -h $DB_HOST -U $DB_USER -d tars -c "SELECT state, count(*) FROM pg_stat_activity WHERE datname='tars' GROUP BY state;"

# Check Redis memory
kubectl exec -n tars-production redis-0 -- redis-cli info memory | grep used_memory_human

# Test API endpoint
curl -H "Authorization: Bearer $TEST_JWT" http://dashboard-api.tars-production.svc.cluster.local:3001/api/v1/evaluations | jq .
```

### C. Troubleshooting Resources

- **Runbook:** https://docs.tars.io/runbooks/v1.0.1-upgrade
- **Architecture Diagram:** https://docs.tars.io/architecture/phase-11-5
- **Slack Channel:** #tars-ops
- **On-Call Rotation:** https://pagerduty.com/schedules/tars-oncall

---

**Playbook Version:** 1.0
**Last Updated:** 2025-11-20
**Maintained By:** T.A.R.S. SRE Team
**Review Frequency:** After each production deployment

---

**End of Upgrade Playbook**
