# T.A.R.S. v1.0.1 Hotfix Plan

**Version:** 1.0.1
**Target Release Date:** December 15, 2025 (2 weeks from GA)
**Release Type:** Patch / Hotfix
**Priority:** High

---

## Executive Summary

v1.0.1 is a focused hotfix release addressing 5 critical issues identified during GA deployment and early production usage. This release maintains API compatibility and requires no schema migrations.

**Timeline:** 2 weeks
**Risk Level:** Low (backward compatible)
**Downtime Required:** Zero (rolling deployment)

---

## Identified Issues

### 1. WebSocket Reconnection Issue ⚠️ **HIGH PRIORITY**

**Issue ID:** TARS-1001
**Component:** Dashboard Frontend
**Severity:** High
**Impact:** Users must manually refresh page after WebSocket disconnection

**Symptoms:**
- WebSocket connection drops after 5-10 minutes of inactivity
- Dashboard stops receiving real-time updates
- No automatic reconnection attempt
- Console shows: `WebSocket closed with code 1006`

**Root Cause:**
- Missing WebSocket heartbeat/ping mechanism
- No exponential backoff retry logic
- Browser idle timeout not handled

**Proposed Fix:**
```typescript
// dashboard/frontend/src/hooks/useWebSocket.ts
- Implement ping/pong heartbeat every 30s
- Add exponential backoff reconnection (1s, 2s, 4s, 8s, max 30s)
- Handle browser visibility API for idle detection
- Add connection state indicator to UI
```

**Files to Modify:**
- `dashboard/frontend/src/hooks/useWebSocket.ts` (~50 LOC changes)
- `dashboard/frontend/src/components/ConnectionStatus.tsx` (new, ~80 LOC)
- `dashboard/api/websocket_handler.py` (~30 LOC changes)

**Testing:**
- Unit tests for reconnection logic
- E2E test: Simulate network interruption
- Manual test: Leave dashboard idle for 15 minutes

**ETA:** 3 days

---

### 2. Grafana Query Timeout (>1000 Evaluations) ⚠️ **MEDIUM PRIORITY**

**Issue ID:** TARS-1002
**Component:** Observability
**Severity:** Medium
**Impact:** Grafana dashboards fail to load metrics when >1000 evaluations exist

**Symptoms:**
- Grafana queries timeout after 30s
- Error: `context deadline exceeded`
- Affects "Evaluation History" and "Reward Trends" panels
- Occurs when querying >1000 evaluation results

**Root Cause:**
- Prometheus query too broad (no time range limit)
- Missing metric cardinality limits
- Inefficient PromQL query structure

**Proposed Fix:**
```promql
# Before (inefficient):
sum(rate(tars_evaluation_reward[5m])) by (agent_id)

# After (optimized):
topk(100, sum(rate(tars_evaluation_reward[1h])) by (agent_id))
```

**Changes:**
- Add time range selector to Grafana dashboards (default: last 24h)
- Implement metric aggregation for historical data
- Add query caching with 5-minute TTL
- Update PromQL queries with `topk()` and time bounds

**Files to Modify:**
- `observability/dashboards/grafana_eval_engine.json` (~15 queries)
- `observability/prometheus/recording_rules.yaml` (new aggregations)
- `cognition/eval-engine/metrics.py` (add cardinality limits)

**Testing:**
- Load test with 5000+ evaluations
- Query performance benchmarks
- Grafana dashboard rendering time

**ETA:** 4 days

---

### 3. Jaeger 5% Sampling Edge Case 🔧 **LOW PRIORITY**

**Issue ID:** TARS-1003
**Component:** Distributed Tracing
**Severity:** Low
**Impact:** 5% of traces missing parent span context in multi-region scenarios

**Symptoms:**
- Trace continuity breaks at region boundaries
- Orphaned spans in Jaeger UI
- `parent_span_id` is null for ~5% of cross-region traces
- Only affects multi-region hot-reload operations

**Root Cause:**
- Race condition in trace context propagation
- Missing `traceparent` header in Redis Streams messages
- Async span finishing before child span starts

**Proposed Fix:**
```python
# cognition/shared/tracing.py
def propagate_trace_context(message: dict, span: Span) -> dict:
    """Inject trace context into message."""
    carrier = {}
    tracer.inject(span.context, Format.HTTP_HEADERS, carrier)
    message['_trace_context'] = carrier
    return message
```

**Changes:**
- Add trace context injection to all Redis Streams publishers
- Ensure span finish happens in `finally` block
- Increase sampling rate for hot-reload operations to 100%

**Files to Modify:**
- `cognition/shared/tracing.py` (~40 LOC)
- `cognition/hypersync-service/main.py` (~20 LOC)
- `cognition/orchestration-agent/main.py` (~15 LOC)

**Testing:**
- Multi-region trace continuity test
- 1000 cross-region hot-reload operations
- Verify 100% parent span linkage

**ETA:** 2 days

---

### 4. Database Index Tuning 🚀 **PERFORMANCE**

**Issue ID:** TARS-1004
**Component:** Database
**Severity:** Medium
**Impact:** Slow queries on evaluation history API (>500ms p95)

**Symptoms:**
- `/api/v1/evaluations` endpoint slow with large datasets
- PostgreSQL query planner not using indexes efficiently
- Seq scan on `evaluations` table for filtered queries

**Current Schema:**
```sql
-- Existing indexes
CREATE INDEX idx_evaluations_agent_id ON evaluations(agent_id);
CREATE INDEX idx_evaluations_timestamp ON evaluations(created_at);
```

**Proposed Indexes:**
```sql
-- Add composite indexes
CREATE INDEX idx_evaluations_agent_timestamp
  ON evaluations(agent_id, created_at DESC);

CREATE INDEX idx_evaluations_status_timestamp
  ON evaluations(status, created_at DESC)
  WHERE status IN ('completed', 'failed');

-- Add partial index for recent evaluations
CREATE INDEX idx_evaluations_recent
  ON evaluations(created_at DESC, agent_id)
  WHERE created_at > NOW() - INTERVAL '7 days';
```

**Query Optimization:**
```python
# Before:
evaluations = db.query(Evaluation).filter(
    Evaluation.agent_id == agent_id
).order_by(Evaluation.created_at.desc()).all()

# After:
evaluations = db.query(Evaluation).filter(
    Evaluation.agent_id == agent_id
).order_by(Evaluation.created_at.desc()).limit(100)
```

**Changes:**
- Add 3 new composite indexes
- Add query result pagination (limit 100 per page)
- Add query result caching (Redis, 1-minute TTL)
- Update EXPLAIN ANALYZE in monitoring

**Files to Modify:**
- `infra/db/migrations/v1_0_1_add_indexes.sql` (new)
- `dashboard/api/evaluation_routes.py` (~30 LOC)
- `dashboard/api/db_config.py` (add connection pooling config)

**Expected Improvement:**
- Query time: 500ms → <100ms (p95)
- Reduced database CPU usage by ~30%

**ETA:** 3 days

---

### 5. RL Agent Memory Leak (PPO) 🐛 **CRITICAL**

**Issue ID:** TARS-1005
**Component:** RL Agents (PPO)
**Severity:** Critical
**Impact:** PPO agent memory grows unbounded during long training sessions

**Symptoms:**
- Memory usage grows from 500MB to 4GB+ over 24 hours
- OOMKilled after ~30 hours of continuous training
- Only affects PPO agent (not DQN, A2C, DDPG)
- Memory not released after episode completion

**Root Cause:**
- Rollout buffer not cleared properly
- TensorFlow graph accumulation
- Missing `del` for large numpy arrays

**Proposed Fix:**
```python
# cognition/orchestration-agent/agents/ppo_agent.py

class PPOAgent:
    def _collect_rollouts(self):
        # ... collect rollouts ...

        # FIX: Explicitly clear buffer after training
        self.rollout_buffer.reset()

        # FIX: Clear TensorFlow graph
        tf.keras.backend.clear_session()

        # FIX: Delete large arrays
        del observations, actions, rewards
        gc.collect()
```

**Changes:**
- Add explicit buffer clearing after each training iteration
- Add TensorFlow graph clearing
- Add periodic garbage collection (every 10 episodes)
- Add memory usage monitoring metrics

**Files to Modify:**
- `cognition/orchestration-agent/agents/ppo_agent.py` (~25 LOC)
- `cognition/orchestration-agent/agents/base_agent.py` (~15 LOC, add memory tracking)

**Testing:**
- 48-hour soak test with memory profiling
- Memory growth rate must be <5MB/hour
- No OOMKilled events

**ETA:** 3 days

---

## Implementation Timeline

### Week 1

**Days 1-2:** Issue triage and prioritization
- Code review of all 5 issues
- Write unit tests for each fix
- Create feature branches

**Days 3-4:** High-priority fixes
- TARS-1001: WebSocket reconnection (3 days)
- TARS-1005: PPO memory leak (3 days, parallel)

**Day 5:** Testing and validation
- Unit tests
- Integration tests
- Memory profiling

### Week 2

**Days 6-8:** Medium/low priority fixes
- TARS-1002: Grafana query optimization (4 days)
- TARS-1004: Database indexes (3 days, parallel)
- TARS-1003: Jaeger tracing (2 days)

**Days 9-10:** Final testing
- Full regression test suite
- Load testing
- Multi-region failover tests

**Days 11-12:** Release preparation
- Update CHANGELOG.md
- Generate release notes
- Build and tag v1.0.1

**Days 13-14:** Deployment
- Deploy to staging (day 13)
- Production canary rollout (day 14)

---

## Testing Strategy

### Unit Tests
- WebSocket reconnection logic
- Database query optimization
- Memory management in PPO agent

### Integration Tests
- End-to-end dashboard workflow
- Multi-region trace propagation
- Database query performance

### Load Tests
- 10,000 concurrent dashboard users
- 5,000+ evaluation history queries
- 48-hour PPO training soak test

### Regression Tests
- Full Phase 13.8 benchmark suite
- Multi-region failover tests
- All E2E pipeline tests

---

## Rollback Plan

**Trigger Conditions:**
- Critical bug in v1.0.1
- >5% error rate increase
- >50% latency degradation
- Memory leak not resolved

**Rollback Procedure:**
1. Stop v1.0.1 deployment (ArgoCD pause)
2. Revert Kubernetes manifests to v1.0.0 tags
3. Database migration rollback (if needed)
4. Verify v1.0.0 health checks
5. Post-mortem and re-plan fixes

**Rollback Time:** <10 minutes (Helm rollback)

---

## Migration Guide

### For Dashboard Users

**No action required.** WebSocket reconnection is automatic.

### For API Clients

**No breaking changes.** All v1 APIs remain compatible.

Optional improvement: Use pagination for `/api/v1/evaluations`:
```bash
# Before:
curl /api/v1/evaluations?agent_id=abc123

# After (recommended):
curl /api/v1/evaluations?agent_id=abc123&limit=100&offset=0
```

### For Operators

**Database Migration (Zero Downtime):**
```bash
# Run migration (indexes created online)
kubectl exec -it postgres-0 -- psql -f /migrations/v1_0_1_add_indexes.sql

# Verify indexes
kubectl exec -it postgres-0 -- psql -c '\d+ evaluations'
```

**Grafana Dashboards:**
```bash
# Update dashboards via ArgoCD
argocd app sync tars-observability --resource grafana:ConfigMap:grafana-dashboards
```

---

## Success Criteria

### Functional
- ✅ WebSocket reconnects automatically within 30s
- ✅ Grafana dashboards load in <5s with 5000+ evaluations
- ✅ 100% trace continuity in multi-region operations
- ✅ PPO agent memory stable over 48 hours

### Performance
- ✅ API query time: <100ms (p95)
- ✅ Dashboard load time: <2s
- ✅ Memory growth rate: <5MB/hour

### Reliability
- ✅ Zero production incidents during rollout
- ✅ 100% test pass rate
- ✅ No regressions in existing functionality

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Database migration causes downtime | Low | High | Online index creation, test in staging |
| WebSocket fix breaks existing clients | Low | Medium | Backward compatible, feature flag |
| PPO memory leak not fully resolved | Medium | High | 48-hour soak test, monitoring alerts |
| Grafana query optimization ineffective | Low | Low | Benchmark before/after, rollback plan |

---

## Post-Release Monitoring

### Week 1 After Release
- Monitor WebSocket reconnection rate (target: <1% manual refresh)
- Monitor Grafana query performance (target: <5s load time)
- Monitor PPO agent memory (target: <5MB/hour growth)
- Monitor database query latency (target: <100ms p95)

### Week 2 After Release
- Collect user feedback on dashboard experience
- Analyze Jaeger trace continuity (target: 100%)
- Review production metrics for anomalies

### Week 4 After Release
- Retrospective meeting
- Document lessons learned
- Plan v1.1.0 roadmap

---

## Stakeholder Communication

### Internal
- Daily standup updates during implementation
- Slack notifications for release milestones
- Post-deployment health report

### External (if applicable)
- Release notes published to docs site
- Email notification to GA customers
- Status page update during deployment

---

## Approval Checklist

- [ ] Engineering Lead approval
- [ ] QA Lead sign-off on test plan
- [ ] SRE Lead approval of rollout plan
- [ ] Security review (if needed)
- [ ] CTO approval for production deployment

---

## References

- [Phase 13.9 Completion Report](../final/PHASE13_9_COMPLETION_REPORT.md)
- [Production Readiness Checklist](../final/PRODUCTION_READINESS_CHECKLIST.md)
- [Release Notes v1.0.0](../final/RELEASE_NOTES_V1_0.md)
- [GitHub Issues Tracker](https://github.com/veleron/tars/issues)

---

**End of v1.0.1 Hotfix Plan**
