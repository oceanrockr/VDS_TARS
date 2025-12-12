# Production Diagnostics Runbook - T.A.R.S.

**Version:** 1.0.0
**Last Updated:** 2025-11-20
**Owner:** SRE Team

---

## Overview

This runbook provides step-by-step diagnostic procedures for common production issues in T.A.R.S. Use this guide for troubleshooting incidents, performance degradation, and system anomalies.

**Prerequisites:**
- `kubectl` access to production cluster
- Grafana dashboard access
- Jaeger trace viewer access
- PostgreSQL client (`psql`)
- Redis client (`redis-cli`)

---

## Table of Contents

1. [Latency Spikes Diagnosis](#1-latency-spikes-diagnosis)
2. [Worker Pool Starvation](#2-worker-pool-starvation)
3. [Redis Cache Churn](#3-redis-cache-churn)
4. [Region Failover Verification](#4-region-failover-verification)
5. [OpenTelemetry Trace Continuity Debugging](#5-opentelemetry-trace-continuity-debugging)
6. [Prometheus Query Optimizations](#6-prometheus-query-optimizations)

---

## 1. Latency Spikes Diagnosis

### Symptoms
- API response times >200ms (p95)
- Evaluation pipeline taking >300s
- Dashboard slow to load
- Prometheus alert: `APILatencyHigh`

### Quick Checks

**1.1 Check Current Latency:**
```bash
# P95 latency for last 5 minutes
kubectl exec -it prometheus-0 -n monitoring -- \
  promtool query instant 'http://localhost:9090' \
  'histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))'
```

**Expected:** <200ms
**Action if exceeded:** Continue to detailed diagnosis

**1.2 Check Service Health:**
```bash
# Check all service pods
kubectl get pods -n tars -l app.kubernetes.io/name=tars

# Check recent restarts
kubectl get pods -n tars -o custom-columns=\
  NAME:.metadata.name,\
  RESTARTS:.status.containerStatuses[0].restartCount
```

**Expected:** All `Running`, restarts <3
**Action if unhealthy:** Investigate pod logs

### Detailed Diagnosis

**1.3 Identify Slow Endpoints:**
```promql
# Grafana query: Top 10 slowest endpoints
topk(10,
  histogram_quantile(0.95,
    sum(rate(http_request_duration_seconds_bucket[5m])) by (route, le)
  )
)
```

**1.4 Check Database Query Performance:**
```bash
# Connect to PostgreSQL
kubectl exec -it postgres-0 -n tars -- psql -U postgres -d tars

# Find slow queries
SELECT pid, now() - query_start AS duration, query
FROM pg_stat_activity
WHERE state = 'active' AND now() - query_start > interval '1 second'
ORDER BY duration DESC;

# Check lock contention
SELECT * FROM pg_locks WHERE NOT granted;
```

**Common Causes:**
- Missing indexes (see [Phase 13 DB optimizations](../v1_0_1/CHANGELOG.md))
- Lock contention on `evaluations` table
- Connection pool exhaustion

**Mitigation:**
```sql
-- Add missing index (if needed)
CREATE INDEX CONCURRENTLY idx_evaluations_agent_timestamp
  ON evaluations(agent_id, created_at DESC);

-- Kill long-running query (if stuck)
SELECT pg_terminate_backend(<pid>);
```

**1.5 Check External Dependencies:**
```bash
# Check external API latency
kubectl exec -it orchestration-agent-0 -n tars -- \
  curl -w "@curl-format.txt" -o /dev/null -s \
  https://api.external-service.com/health

# curl-format.txt content:
# time_total:  %{time_total}s\n
```

**Threshold:** <100ms for external APIs

**1.6 Review OpenTelemetry Traces:**
1. Open Jaeger UI
2. Search for recent traces with high duration
3. Look for trace ID in logs:
   ```bash
   kubectl logs -n tars deployment/eval-engine --tail=1000 | grep <trace_id>
   ```

### Resolution

**Common Fixes:**
- Scale up worker pods: `kubectl scale deployment/eval-engine --replicas=5 -n tars`
- Clear Redis cache if stale: `redis-cli FLUSHDB`
- Restart slow service: `kubectl rollout restart deployment/orchestration-agent -n tars`
- Add database indexes (see v1.0.1 changelog)

**Verification:**
```bash
# Monitor latency for 5 minutes
watch -n 10 'kubectl exec -it prometheus-0 -n monitoring -- \
  promtool query instant http://localhost:9090 \
  "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))"'
```

---

## 2. Worker Pool Starvation

### Symptoms
- Evaluation queue depth >100
- Workers at 100% CPU
- Evaluation completion time increasing
- Prometheus alert: `EvalWorkerStarvation`

### Quick Checks

**2.1 Check Queue Depth:**
```bash
# Redis queue depth
kubectl exec -it redis-0 -n tars -- redis-cli XLEN eval-queue

# Prometheus metric
kubectl exec -it prometheus-0 -n monitoring -- \
  promtool query instant http://localhost:9090 \
  'tars_eval_queue_depth'
```

**Expected:** <50
**Critical:** >100

**2.2 Check Worker Utilization:**
```bash
# Worker CPU usage
kubectl top pods -n tars -l app=eval-engine

# Worker count
kubectl get pods -n tars -l app=eval-engine --no-headers | wc -l
```

**Expected:** CPU <80%, Worker count ≥3

### Detailed Diagnosis

**2.3 Analyze Queue Backlog:**
```bash
# Get oldest evaluation request
kubectl exec -it redis-0 -n tars -- redis-cli \
  XRANGE eval-queue - + COUNT 1

# Get queue age
kubectl exec -it redis-0 -n tars -- redis-cli \
  XINFO STREAM eval-queue
```

**2.4 Check for Stuck Workers:**
```bash
# Check worker pod status
kubectl get pods -n tars -l app=eval-engine -o wide

# Check worker logs for errors
kubectl logs -n tars deployment/eval-engine --tail=100 | grep -i error

# Check for pending episodes
kubectl exec -it eval-engine-0 -n tars -- \
  python -c "
from cognition.eval_engine.worker import get_active_episodes
print(f'Active episodes: {len(get_active_episodes())}')
"
```

**2.5 Profile Worker Performance:**
```bash
# Check episode duration distribution
kubectl exec -it prometheus-0 -n monitoring -- \
  promtool query instant http://localhost:9090 \
  'histogram_quantile(0.95, rate(tars_eval_episode_duration_seconds_bucket[5m]))'

# Expected: <60s per episode
```

### Resolution

**Quick Fixes:**

**2.6 Scale Up Workers:**
```bash
# Horizontal scaling
kubectl scale deployment/eval-engine --replicas=10 -n tars

# Verify scaling
kubectl get hpa eval-engine -n tars
```

**2.7 Clear Stuck Evaluations:**
```bash
# Identify failed evaluations
kubectl exec -it postgres-0 -n tars -- psql -U postgres -d tars -c \
  "SELECT id, agent_id, status, created_at
   FROM evaluations
   WHERE status = 'running' AND created_at < NOW() - INTERVAL '30 minutes';"

# Mark as failed (if truly stuck)
kubectl exec -it postgres-0 -n tars -- psql -U postgres -d tars -c \
  "UPDATE evaluations SET status = 'failed', error = 'Timeout'
   WHERE status = 'running' AND created_at < NOW() - INTERVAL '30 minutes';"
```

**2.8 Restart Workers (Last Resort):**
```bash
kubectl rollout restart deployment/eval-engine -n tars

# Monitor restart
kubectl rollout status deployment/eval-engine -n tars
```

**Verification:**
```bash
# Monitor queue depth recovery
watch -n 5 'kubectl exec -it redis-0 -n tars -- redis-cli XLEN eval-queue'

# Should decrease steadily
```

---

## 3. Redis Cache Churn

### Symptoms
- Cache hit rate <50%
- Frequent cache misses in logs
- Increased database query load
- Prometheus alert: `RedisCacheChurnHigh`

### Quick Checks

**3.1 Check Cache Hit Rate:**
```bash
# Redis INFO stats
kubectl exec -it redis-0 -n tars -- redis-cli INFO stats | grep keyspace

# Prometheus metric
kubectl exec -it prometheus-0 -n monitoring -- \
  promtool query instant http://localhost:9090 \
  'rate(tars_cache_hits_total[5m]) / (rate(tars_cache_hits_total[5m]) + rate(tars_cache_misses_total[5m]))'
```

**Expected:** >90%
**Action if below:** Investigate cache eviction

**3.2 Check Redis Memory:**
```bash
# Memory usage
kubectl exec -it redis-0 -n tars -- redis-cli INFO memory

# Key evictions
kubectl exec -it redis-0 -n tars -- redis-cli INFO stats | grep evicted_keys
```

**Expected:** Evictions = 0

### Detailed Diagnosis

**3.3 Analyze Cache Keys:**
```bash
# Count keys by pattern
kubectl exec -it redis-0 -n tars -- redis-cli --scan --pattern 'eval:*' | wc -l
kubectl exec -it redis-0 -n tars -- redis-cli --scan --pattern 'agent:*' | wc -l

# Check key TTLs
kubectl exec -it redis-0 -n tars -- redis-cli --scan --pattern 'eval:*' | \
  xargs -I {} sh -c 'echo "TTL {}" && kubectl exec -it redis-0 -n tars -- redis-cli TTL {}'
```

**3.4 Check Eviction Policy:**
```bash
kubectl exec -it redis-0 -n tars -- redis-cli CONFIG GET maxmemory-policy

# Should be: allkeys-lru or volatile-lru
```

### Resolution

**3.5 Increase Redis Memory:**
```yaml
# Edit Redis statefulset
kubectl edit statefulset redis -n tars

# Update:
resources:
  limits:
    memory: 4Gi  # Increase from 2Gi
  requests:
    memory: 4Gi
```

**3.6 Optimize Cache TTLs:**
```python
# Review cache TTL settings in code
# cognition/shared/cache.py

# Recommended TTLs:
# - Evaluation results: 3600s (1 hour)
# - Agent configs: 1800s (30 minutes)
# - Temporary data: 300s (5 minutes)
```

**3.7 Enable Cache Warming:**
```bash
# Pre-populate cache with hot data
kubectl exec -it dashboard-api-0 -n tars -- \
  python -m dashboard.scripts.warm_cache --agents all --evaluations 100
```

---

## 4. Region Failover Verification

### Symptoms
- Multi-region replication lag >5s
- Failed leader election
- Inconsistent data across regions
- Prometheus alert: `MultiRegionReplicationLagHigh`

### Quick Checks

**4.1 Check Region Health:**
```bash
# Check all region pods
kubectl get pods -n tars --context us-east-1
kubectl get pods -n tars --context us-west-2
kubectl get pods -n tars --context eu-central-1

# Check Raft cluster status
kubectl exec -it orchestration-agent-0 -n tars -- \
  curl localhost:8094/health | jq '.raft_status'
```

**4.2 Check Replication Lag:**
```bash
# PostgreSQL replication lag
kubectl exec -it postgres-0 -n tars -- psql -U postgres -d tars -c \
  "SELECT client_addr, state, sync_state,
          pg_wal_lsn_diff(pg_current_wal_lsn(), replay_lsn) AS lag_bytes
   FROM pg_stat_replication;"

# Expected: lag_bytes <10MB
```

**4.3 Check Redis Streams Lag:**
```bash
# Check consumer group lag
kubectl exec -it redis-0 -n tars -- redis-cli \
  XINFO GROUPS hypersync-proposals

# Expected: lag <100 messages
```

### Detailed Diagnosis

**4.4 Test Cross-Region Connectivity:**
```bash
# Ping from us-east-1 to us-west-2
kubectl exec -it orchestration-agent-0 -n tars --context us-east-1 -- \
  curl -w "@curl-format.txt" \
  http://orchestration-agent.tars.svc.cluster.local:8094/health \
  --resolve orchestration-agent.tars.svc.cluster.local:8094:<us-west-2-ip>

# Expected: <100ms
```

**4.5 Verify Trace Propagation:**
```bash
# Trigger cross-region operation
curl -X POST https://api.tars.com/v1/hot-reload \
  -H "Content-Type: application/json" \
  -d '{"agent_id": "test", "hyperparameters": {"lr": 0.001}}'

# Get trace ID from response headers
trace_id="<from X-Trace-ID header>"

# Search Jaeger for trace
# Verify all regions have spans
```

### Resolution

**4.6 Manual Failover Test:**
```bash
# Stop leader in us-east-1
kubectl scale deployment/orchestration-agent --replicas=0 -n tars --context us-east-1

# Wait for election (should be <5s)
sleep 10

# Check new leader
kubectl exec -it orchestration-agent-0 -n tars --context us-west-2 -- \
  curl localhost:8094/health | jq '.raft_status.role'

# Should be: "leader"

# Restore us-east-1
kubectl scale deployment/orchestration-agent --replicas=3 -n tars --context us-east-1
```

**4.7 Force Replication Sync:**
```bash
# PostgreSQL: Manually sync replica
kubectl exec -it postgres-0 -n tars --context us-west-2 -- \
  pg_receivewal -D /var/lib/postgresql/data -h postgres.us-east-1 -U replicator

# Redis: Re-subscribe to stream
kubectl exec -it redis-0 -n tars --context us-west-2 -- redis-cli \
  XGROUP SETID hypersync-proposals us-west-2 0
```

---

## 5. OpenTelemetry Trace Continuity Debugging

### Symptoms
- Orphaned spans in Jaeger
- Missing parent span context
- Trace breaks at service boundaries
- Prometheus metric: `tars_trace_orphaned_spans_total` >0

### Quick Checks

**5.1 Check Trace Sampling:**
```bash
# Check sampling rate
kubectl exec -it orchestration-agent-0 -n tars -- \
  grep OTEL_TRACES_SAMPLER /proc/1/environ

# Should be: parentbased_traceidratio or always_on
```

**5.2 Search for Orphaned Spans:**
```bash
# Jaeger query
curl -G 'http://jaeger-query:16686/api/traces' \
  --data-urlencode 'service=eval-engine' \
  --data-urlencode 'tags={"orphaned":"true"}' \
  --data-urlencode 'limit=20'
```

### Detailed Diagnosis

**5.3 Verify Trace Context Propagation:**
```bash
# Check HTTP headers
kubectl exec -it dashboard-api-0 -n tars -- \
  tcpdump -i any -A -s 0 'tcp port 8000 and (((ip[2:2] - ((ip[0]&0xf)<<2)) - ((tcp[12]&0xf0)>>2)) != 0)' | \
  grep -i traceparent

# Should see: traceparent: 00-<trace-id>-<span-id>-01
```

**5.4 Check Redis Message Trace Context:**
```bash
# Inspect Redis stream message
kubectl exec -it redis-0 -n tars -- redis-cli \
  XREAD COUNT 1 STREAMS hypersync-proposals 0-0 | grep trace_context

# Should contain: "_trace_context" field
```

### Resolution

**5.5 Fix Missing Trace Context (Code):**
```python
# In cognition/shared/tracing.py
from opentelemetry import trace
from opentelemetry.propagate import inject

def publish_to_redis(message: dict):
    # Inject trace context
    carrier = {}
    inject(carrier)  # Injects traceparent header
    message['_trace_context'] = carrier

    # Publish
    redis_client.xadd('stream', message)
```

**5.6 Increase Sampling for Debugging:**
```bash
# Temporarily set 100% sampling
kubectl set env deployment/orchestration-agent \
  OTEL_TRACES_SAMPLER=always_on \
  -n tars

# Restart deployment
kubectl rollout restart deployment/orchestration-agent -n tars
```

**Revert after debugging:**
```bash
kubectl set env deployment/orchestration-agent \
  OTEL_TRACES_SAMPLER=parentbased_traceidratio \
  OTEL_TRACES_SAMPLER_ARG=0.1 \
  -n tars
```

---

## 6. Prometheus Query Optimizations

### Symptoms
- Grafana dashboards timeout
- Prometheus queries >30s
- High Prometheus CPU usage
- Alert: `PrometheusQueryOverload`

### Quick Checks

**6.1 Check Prometheus Performance:**
```bash
# Query duration
kubectl exec -it prometheus-0 -n monitoring -- \
  curl localhost:9090/api/v1/query?query=prometheus_engine_query_duration_seconds

# Scrape duration
kubectl top pod prometheus-0 -n monitoring
```

**Expected:** Query duration <5s, CPU <70%

### Optimization Strategies

**6.2 Use Recording Rules:**
```yaml
# observability/prometheus/recording_rules.yaml
groups:
  - name: tars_aggregations
    interval: 60s
    rules:
      # Pre-aggregate evaluation metrics
      - record: tars:eval_reward:p95
        expr: histogram_quantile(0.95, sum(rate(tars_evaluation_reward_bucket[5m])) by (le, agent_id))

      # Pre-aggregate API latency
      - record: tars:api_latency:p95
        expr: histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket[5m])) by (le, route))
```

**Apply rules:**
```bash
kubectl apply -f observability/prometheus/recording_rules.yaml
kubectl rollout restart statefulset/prometheus -n monitoring
```

**6.3 Optimize PromQL Queries:**

**Before (slow):**
```promql
sum(rate(tars_evaluation_reward[5m])) by (agent_id)
```

**After (fast):**
```promql
# Use recording rule
tars:eval_reward:p95

# Or limit cardinality
topk(20, sum(rate(tars_evaluation_reward[5m])) by (agent_id))
```

**6.4 Add Query Timeout:**
```bash
# Edit Grafana datasource
kubectl edit configmap grafana-datasources -n monitoring

# Add:
jsonData:
  timeoutSeconds: 30
  queryTimeout: 30s
```

---

## Escalation Paths

### Level 1: On-Call Engineer
- Initial triage (this runbook)
- Quick fixes (restart, scale)
- Monitoring

### Level 2: SRE Lead
- Complex diagnosis
- Cross-region issues
- Database tuning

### Level 3: Engineering Lead
- Code fixes required
- Architecture changes
- Hotfix deployment

### Level 4: CTO
- Major incident
- Data loss risk
- Customer communication

---

## Post-Incident Checklist

- [ ] Incident documented in incident tracker
- [ ] Root cause identified
- [ ] Metrics returned to normal
- [ ] Monitoring alerts verified
- [ ] Post-mortem scheduled (for SEV-1/SEV-2)
- [ ] Runbook updated (if new procedures)
- [ ] Customer communication (if user-impacting)

---

## References

- [Production Readiness Checklist](../final/PRODUCTION_READINESS_CHECKLIST.md)
- [Phase 13.8 Benchmark Results](../../benchmarks/)
- [Grafana Dashboards](http://grafana.tars.internal)
- [Jaeger Tracing](http://jaeger.tars.internal)
- [PagerDuty Runbooks](https://pagerduty.com/runbooks)

---

**End of Production Diagnostics Runbook**
