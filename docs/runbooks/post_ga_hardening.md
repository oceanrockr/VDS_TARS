# Post-GA Hardening Runbook - T.A.R.S.

**Version:** 1.0.0
**Last Updated:** 2025-11-20
**Owner:** SRE Team
**Timeline:** Weeks 1-4 Post-GA

---

## Overview

This runbook provides a structured checklist for hardening T.A.R.S. in the first 4 weeks following General Availability deployment. Focus on observability, performance tuning, cost optimization, and reliability improvements based on real production usage.

**Goals:**
- Ensure production stability and SLO compliance
- Optimize resource utilization and costs
- Identify and fix production-specific issues
- Build operational muscle memory

---

## Table of Contents

1. [Week 1: Stabilization & Monitoring](#week-1-stabilization--monitoring)
2. [Week 2: Performance Optimization](#week-2-performance-optimization)
3. [Week 3: Cost Optimization](#week-3-cost-optimization)
4. [Week 4: Reliability Hardening](#week-4-reliability-hardening)
5. [Ongoing Activities](#ongoing-activities)

---

## Week 1: Stabilization & Monitoring

**Focus:** Ensure system stability, validate SLOs, establish baseline metrics

### Day 1-2: Initial Health Validation

#### 1.1 Validate All Services Running

```bash
# Check all deployments
kubectl get deployments -n tars

# Check pod health
kubectl get pods -n tars -o wide

# Check HPA status
kubectl get hpa -n tars

# Verify multi-region deployment
kubectl get pods -n tars --context us-east-1
kubectl get pods -n tars --context us-west-2
kubectl get pods -n tars --context eu-central-1
```

**Success Criteria:**
- [ ] All deployments at desired replica count
- [ ] Zero `CrashLoopBackOff` pods
- [ ] HPA configured and active
- [ ] All 3 regions healthy

#### 1.2 Validate SLO Compliance

**Check Grafana Dashboards:**
1. Open "T.A.R.S. SLO Dashboard"
2. Set time range to "Last 24 hours"
3. Verify all SLOs:

| SLO | Target | Threshold | Status |
|-----|--------|-----------|--------|
| API Latency (p95) | <150ms | <200ms | ✅ |
| Error Rate | <1% | <5% | ✅ |
| Availability | >99.5% | >99% | ✅ |
| Evaluation Latency (p95) | <120s | <300s | ✅ |

**Action Items:**
- [ ] Document baseline metrics
- [ ] Set up SLO alerting if not already configured
- [ ] Create daily SLO compliance report

#### 1.3 Review Error Logs

```bash
# Check error rate by service
kubectl logs -n tars deployment/orchestration-agent --tail=1000 | grep -i error | wc -l
kubectl logs -n tars deployment/eval-engine --tail=1000 | grep -i error | wc -l
kubectl logs -n tars deployment/dashboard-api --tail=1000 | grep -i error | wc -l

# Check for repeated errors
kubectl logs -n tars deployment/orchestration-agent --tail=5000 | \
  grep -i error | sort | uniq -c | sort -rn | head -20
```

**Action Items:**
- [ ] Document top 5 error types
- [ ] Create tickets for unexpected errors
- [ ] Set up error aggregation dashboard

### Day 3-4: Establish Baseline Metrics

#### 1.4 Resource Utilization Baseline

```bash
# CPU usage by pod
kubectl top pods -n tars --sort-by=cpu

# Memory usage by pod
kubectl top pods -n tars --sort-by=memory

# Node utilization
kubectl top nodes
```

**Create Baseline Report:**
```bash
# Export to CSV
kubectl top pods -n tars --sort-by=cpu --no-headers | \
  awk '{print $1","$2","$3}' > baseline_cpu.csv

kubectl top pods -n tars --sort-by=memory --no-headers | \
  awk '{print $1","$2","$3}' > baseline_memory.csv
```

**Action Items:**
- [ ] Document CPU/memory baseline per service
- [ ] Set up resource usage alerts (>80% sustained)
- [ ] Identify over-provisioned pods

#### 1.5 Traffic Pattern Analysis

```bash
# Requests per minute
kubectl exec -it prometheus-0 -n monitoring -- \
  promtool query instant http://localhost:9090 \
  'sum(rate(http_requests_total[5m])) * 60'

# Peak traffic hours
kubectl exec -it prometheus-0 -n monitoring -- \
  promtool query range http://localhost:9090 \
  'sum(rate(http_requests_total[1h]))' \
  --start $(date -u -d '7 days ago' +%s) \
  --end $(date -u +%s) \
  --step 3600
```

**Action Items:**
- [ ] Document traffic patterns (peak/off-peak)
- [ ] Identify daily/weekly patterns
- [ ] Plan autoscaling adjustments

### Day 5-7: Alerting & Monitoring Refinement

#### 1.6 Review Alert Firing Frequency

```bash
# Check PagerDuty incidents (if integrated)
curl -H "Authorization: Token token=<token>" \
  https://api.pagerduty.com/incidents?since=$(date -u -d '7 days ago' +%Y-%m-%d)

# Prometheus alert history
kubectl exec -it prometheus-0 -n monitoring -- \
  curl localhost:9090/api/v1/alerts
```

**Action Items:**
- [ ] Identify noisy alerts (>5 fires/day)
- [ ] Tune alert thresholds
- [ ] Add missing alerts

#### 1.7 Validate Runbook Coverage

**Test Runbooks:**
1. Simulate latency spike (use load testing)
2. Follow [production_diagnostics.md](production_diagnostics.md) steps
3. Document gaps or unclear steps

**Action Items:**
- [ ] Update runbooks based on findings
- [ ] Add missing diagnostic procedures
- [ ] Create quick-reference cards

---

## Week 2: Performance Optimization

**Focus:** Optimize latency, throughput, and resource efficiency

### Day 8-9: Latency Optimization

#### 2.1 Identify Slow Endpoints

```promql
# Grafana query: Slowest endpoints
topk(10,
  histogram_quantile(0.95,
    sum(rate(http_request_duration_seconds_bucket[24h])) by (route, le)
  )
) > 0.2  # >200ms
```

**Action Items per Endpoint:**
1. Review OpenTelemetry traces
2. Identify bottleneck (DB, external API, CPU)
3. Implement optimization:
   - Add caching
   - Add database index
   - Optimize query
   - Reduce payload size

**Example: Optimize `/api/v1/evaluations` endpoint**
```bash
# Before optimization
curl -w "@curl-format.txt" https://api.tars.com/v1/evaluations?limit=100
# time_total: 0.450s

# After adding pagination + caching
curl -w "@curl-format.txt" https://api.tars.com/v1/evaluations?limit=100
# time_total: 0.095s  (79% improvement)
```

#### 2.2 Database Query Optimization

```sql
-- Find slowest queries (PostgreSQL)
SELECT query, mean_exec_time, calls
FROM pg_stat_statements
ORDER BY mean_exec_time DESC
LIMIT 20;

-- Check missing indexes
SELECT schemaname, tablename, attname, n_distinct, correlation
FROM pg_stats
WHERE schemaname = 'public'
  AND correlation < 0.5
ORDER BY n_distinct DESC;
```

**Action Items:**
- [ ] Add indexes for top 10 slow queries
- [ ] Review query execution plans (EXPLAIN ANALYZE)
- [ ] Implement query result caching

#### 2.3 Cold Start Optimization

```bash
# Measure cold start time
kubectl delete pod eval-engine-0 -n tars
time kubectl wait --for=condition=ready pod -l app=eval-engine -n tars --timeout=300s

# Expected: <60s
```

**Optimization Strategies:**
- [ ] Reduce container image size
- [ ] Implement readiness probe optimization
- [ ] Pre-warm caches on startup

### Day 10-11: Throughput Optimization

#### 2.4 Validate Autoscaling at Peak Load

**Load Test:**
```bash
# Run throughput benchmark
cd benchmarks/
python throughput_bench.py --target-rps 100 --duration 600

# Monitor HPA during test
watch -n 5 'kubectl get hpa -n tars'
```

**Expected Behavior:**
- HPA scales up when CPU >70%
- Scales up within 60s
- Scales down after 5 minutes of low load

**Action Items:**
- [ ] Tune HPA min/max replicas
- [ ] Adjust target CPU utilization
- [ ] Add custom metrics for HPA (queue depth)

#### 2.5 Worker Pool Tuning

```python
# Analyze worker utilization
import prometheus_api_client

prom = PrometheusConnect(url="http://prometheus:9090")
result = prom.custom_query('tars_eval_worker_utilization')

# Calculate optimal worker count
# Target: 70-80% utilization during peak hours
optimal_workers = peak_queue_depth / (episodes_per_second_per_worker * 60)
```

**Action Items:**
- [ ] Set optimal worker pool size
- [ ] Implement dynamic worker scaling
- [ ] Add worker starvation alerts

### Day 12-14: Caching Strategy

#### 2.6 Analyze Cache Hit Rates

```bash
# Redis cache hit rate
kubectl exec -it redis-0 -n tars -- redis-cli INFO stats | grep keyspace_hits

# Calculate hit rate
hits=$(kubectl exec -it redis-0 -n tars -- redis-cli INFO stats | grep keyspace_hits | cut -d: -f2)
misses=$(kubectl exec -it redis-0 -n tars -- redis-cli INFO stats | grep keyspace_misses | cut -d: -f2)
hit_rate=$(echo "scale=2; $hits / ($hits + $misses) * 100" | bc)

echo "Cache hit rate: ${hit_rate}%"
# Target: >90%
```

**Action Items:**
- [ ] Identify frequently accessed data
- [ ] Implement cache warming for hot data
- [ ] Optimize cache TTLs (see benchmarks)

#### 2.7 Implement Query Result Caching

```python
# Example: Cache evaluation results
from functools import lru_cache
import redis

redis_client = redis.Redis(host='redis', port=6379)

def get_evaluations(agent_id: str, limit: int = 100):
    cache_key = f"eval:{agent_id}:{limit}"

    # Check cache
    cached = redis_client.get(cache_key)
    if cached:
        return json.loads(cached)

    # Query database
    results = db.query(Evaluation).filter_by(agent_id=agent_id).limit(limit).all()

    # Cache results (60s TTL)
    redis_client.setex(cache_key, 60, json.dumps(results))

    return results
```

---

## Week 3: Cost Optimization

**Focus:** Reduce cloud costs without impacting performance

### Day 15-16: Resource Right-Sizing

#### 3.1 Analyze Resource Requests vs Actual Usage

```bash
# Resource requests
kubectl describe deployment orchestration-agent -n tars | grep -A 5 "Requests"

# Actual usage (p95 over 7 days)
kubectl exec -it prometheus-0 -n monitoring -- \
  promtool query instant http://localhost:9090 \
  'quantile_over_time(0.95, container_cpu_usage_seconds_total{namespace="tars"}[7d])'
```

**Right-Sizing Guide:**

| Service | Current Requests | p95 Usage | Recommended |
|---------|-----------------|-----------|-------------|
| orchestration-agent | 2 CPU, 4Gi | 1.2 CPU, 2.5Gi | 1.5 CPU, 3Gi |
| eval-engine | 4 CPU, 8Gi | 2.8 CPU, 5Gi | 3 CPU, 6Gi |
| dashboard-api | 1 CPU, 2Gi | 0.4 CPU, 1Gi | 0.5 CPU, 1.5Gi |

**Action Items:**
- [ ] Reduce over-provisioned resources by 20-30%
- [ ] Monitor for OOMKilled or CPU throttling
- [ ] Document new resource baselines

#### 3.2 Database Connection Pool Optimization

```sql
-- Check active connections
SELECT count(*) FROM pg_stat_activity WHERE state = 'active';

-- Check idle connections
SELECT count(*) FROM pg_stat_activity WHERE state = 'idle';

-- Recommended pool size = (CPU cores * 2) + effective_spindle_count
```

**Action Items:**
- [ ] Reduce connection pool from 100 to 20 per service
- [ ] Implement connection pooling (PgBouncer)
- [ ] Monitor connection usage

### Day 17-18: Storage Optimization

#### 3.3 Analyze Storage Usage

```bash
# PVC usage
kubectl get pvc -n tars

# Actual usage
kubectl exec -it postgres-0 -n tars -- df -h /var/lib/postgresql/data

# Identify largest tables
kubectl exec -it postgres-0 -n tars -- psql -U postgres -d tars -c \
  "SELECT relname, pg_size_pretty(pg_total_relation_size(relid))
   FROM pg_catalog.pg_statio_user_tables
   ORDER BY pg_total_relation_size(relid) DESC LIMIT 10;"
```

**Action Items:**
- [ ] Implement data retention policies (delete evals >30 days)
- [ ] Archive historical data to object storage (S3)
- [ ] Enable compression for large tables

#### 3.4 Cost Analysis Dashboard

**Create Cost Tracking Dashboard:**
1. Track pod resource costs
2. Track storage costs
3. Track data transfer costs
4. Set up budget alerts

**Estimated Monthly Costs (Baseline):**
- Compute: $2,500
- Storage: $500
- Data Transfer: $300
- **Total:** ~$3,300/month

**Target after optimization:** <$2,500/month (24% reduction)

### Day 19-21: Autoscaling Optimization

#### 3.5 Reduce Minimum Replicas During Off-Peak

```yaml
# Update HPA for off-peak hours
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: eval-engine
spec:
  minReplicas: 2  # Reduce from 3 during 22:00-06:00 UTC
  maxReplicas: 10
  # Use cron-based autoscaling or external metrics
```

**Action Items:**
- [ ] Implement time-based autoscaling
- [ ] Reduce min replicas by 30% during off-peak
- [ ] Monitor queue depth during off-peak scale-down

---

## Week 4: Reliability Hardening

**Focus:** Chaos testing, disaster recovery validation, multi-region resilience

### Day 22-23: Chaos Engineering

#### 4.1 Run Weekly Chaos Tests

**Pod Failure Test:**
```bash
# Install chaos-mesh (if not already)
kubectl apply -f https://mirrors.chaos-mesh.org/v2.5.0/chaos-mesh.yaml

# Kill random pod
kubectl apply -f - <<EOF
apiVersion: chaos-mesh.org/v1alpha1
kind: PodChaos
metadata:
  name: pod-kill-test
  namespace: tars
spec:
  action: pod-kill
  mode: one
  selector:
    namespaces:
      - tars
    labelSelectors:
      app: orchestration-agent
  scheduler:
    cron: "@every 1h"
EOF

# Monitor recovery time
watch -n 1 'kubectl get pods -n tars -l app=orchestration-agent'
```

**Expected:** New pod ready within 60s, zero request failures

**Network Latency Injection:**
```yaml
apiVersion: chaos-mesh.org/v1alpha1
kind: NetworkChaos
metadata:
  name: network-delay-test
spec:
  action: delay
  mode: one
  selector:
    namespaces:
      - tars
  delay:
    latency: "100ms"
    correlation: "0"
  duration: "5m"
```

**Action Items:**
- [ ] Verify graceful degradation
- [ ] Validate retry logic
- [ ] Check timeout handling

#### 4.2 Multi-Region Failover Drill

**Test Leader Election:**
```bash
# Simulate us-east-1 region failure
kubectl scale deployment/orchestration-agent --replicas=0 -n tars --context us-east-1

# Wait for leader election (should be <5s)
sleep 10

# Verify new leader in us-west-2
kubectl exec -it orchestration-agent-0 -n tars --context us-west-2 -- \
  curl localhost:8094/health | jq '.raft_status.role'

# Measure failover time
echo "Failover completed in: <time>s"

# Restore us-east-1
kubectl scale deployment/orchestration-agent --replicas=3 -n tars --context us-east-1
```

**Success Criteria:**
- [ ] Leader elected in <5s
- [ ] Zero data loss
- [ ] Zero request failures during failover

### Day 24-25: Disaster Recovery Validation

#### 4.3 Test Database Backup & Restore

**Create Backup:**
```bash
# PostgreSQL backup
kubectl exec -it postgres-0 -n tars -- \
  pg_dump -U postgres -d tars -F custom -f /tmp/tars_backup_$(date +%Y%m%d).dump

# Copy backup to S3
kubectl cp tars/postgres-0:/tmp/tars_backup_*.dump ./backup.dump
aws s3 cp backup.dump s3://tars-backups/$(date +%Y%m%d)/
```

**Test Restore (in staging):**
```bash
# Restore from backup
kubectl exec -it postgres-0 -n tars-staging -- \
  pg_restore -U postgres -d tars -c /tmp/backup.dump

# Verify data integrity
kubectl exec -it postgres-0 -n tars-staging -- psql -U postgres -d tars -c \
  "SELECT count(*) FROM evaluations;"
```

**Action Items:**
- [ ] Validate RTO <4 hours
- [ ] Validate RPO <24 hours
- [ ] Document restore procedure
- [ ] Test restore monthly

#### 4.4 Production Profiling

**Enable Profiling (Temporarily):**
```python
# In orchestration-agent
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# ... run workload ...

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)
```

**Action Items:**
- [ ] Profile PPO agent training (memory usage)
- [ ] Identify CPU hotspots
- [ ] Optimize top 5 slowest functions

### Day 26-28: Region-Level Drift Monitoring

#### 4.5 Track Cross-Region Metrics

```promql
# Latency drift between regions
abs(
  avg(tars_api_latency_p95{region="us-east-1"}) -
  avg(tars_api_latency_p95{region="us-west-2"})
)

# Alert if drift >100ms
```

**Create Region Drift Dashboard:**
1. Latency by region
2. Error rate by region
3. Request distribution by region
4. Replication lag by region

**Action Items:**
- [ ] Set up drift alerts (>100ms)
- [ ] Investigate if drift >50ms sustained
- [ ] Balance load across regions

---

## Ongoing Activities

### Daily

- [ ] Check SLO compliance dashboard
- [ ] Review error rates
- [ ] Check for new Prometheus alerts
- [ ] Monitor on-call tickets

### Weekly

- [ ] Run chaos engineering tests
- [ ] Review cost metrics
- [ ] Update runbooks
- [ ] Team retrospective

### Monthly

- [ ] Test disaster recovery procedure
- [ ] Validate backup integrity
- [ ] Review security patches
- [ ] Quarterly planning

---

## Success Metrics (Post-4 Weeks)

| Metric | Baseline (Week 1) | Target (Week 4) | Actual |
|--------|-------------------|-----------------|--------|
| SLO Compliance | 95% | >99% | ___ |
| P95 API Latency | 150ms | <120ms | ___ |
| Error Rate | 2% | <1% | ___ |
| Monthly Cost | $3,300 | <$2,500 | ___ |
| Mean Time to Recovery | 15min | <10min | ___ |
| Chaos Test Pass Rate | 80% | >95% | ___ |

---

## Graduation Criteria

**Ready to exit hardening phase when:**
- [✅] All SLOs met for 2 consecutive weeks
- [✅] Cost reduced by ≥20%
- [✅] Zero SEV-1 incidents in past 2 weeks
- [✅] Chaos tests pass 95%
- [✅] Disaster recovery validated
- [✅] Runbooks tested and updated
- [✅] Team comfortable with on-call rotation

---

## References

- [Production Diagnostics Runbook](production_diagnostics.md)
- [Phase 13.9 GA Checklist](../final/GA_LAUNCH_CHECKLIST.md)
- [v1.0.1 Hotfix Plan](../v1_0_1/HOTFIX_PLAN.md)
- [Grafana Dashboards](http://grafana.tars.internal)
- [PagerDuty Runbooks](https://pagerduty.com)

---

**End of Post-GA Hardening Runbook**
