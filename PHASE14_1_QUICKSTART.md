# Phase 14.1 Quick Start Guide - T.A.R.S. v1.0.1

**Purpose:** Resume Phase 14.1 implementation in new Claude Code sessions
**Target:** Complete all 11 deliverables for v1.0.1 release
**Current Status:** 2/11 complete (WebSocket fix + Prometheus recording rules)

---

## Session Bootstrap Commands

### Load Context
```bash
# Read implementation progress
cat PHASE14_1_IMPLEMENTATION_PROGRESS.md

# Read hotfix specifications
cat docs/v1_0_1/HOTFIX_PLAN.md

# Read Phase 14 deliverables
cat PHASE14_INITIAL_DELIVERABLES.md
```

### Verify Completed Work
```bash
# Check WebSocket fix
cat fixes/fix_websocket_reconnect/websocket_client_patch.py
cat fixes/fix_websocket_reconnect/websocket_reconnect_test.py

# Check Prometheus recording rules
cat fixes/fix_grafana_query_timeout/recording_rules.yaml

# Run tests (if dependencies installed)
pytest fixes/fix_websocket_reconnect/websocket_reconnect_test.py -v
```

---

## Remaining Implementations (9/11)

### Priority 1: Core Hotfixes (4 files)

#### 1. Grafana Dashboard Patch (TARS-1002 completion)
**File:** `fixes/fix_grafana_query_timeout/grafana_dashboard_patch.json`
**Requirements:**
- Update all dashboard panels to use recording rules from `recording_rules.yaml`
- Replace expensive queries like:
  ```promql
  histogram_quantile(0.95, sum(rate(tars_evaluation_duration_seconds_bucket[5m])) by (agent_id, le))
  ```
  With optimized queries:
  ```promql
  tars:evaluation_latency:p95:1m
  ```
- Target: 60+ panel updates across main operational dashboard
- Validate: Dashboard loads in <5s with 5000+ evaluations

**Test:** `fixes/fix_grafana_query_timeout/grafana_query_tests.py`
- Load test with 1000, 5000, 10000 evaluations
- Measure dashboard load time
- Assert p95 < 5s
- Compare pre/post recording rules performance

---

#### 2. Jaeger Trace Context Fix (TARS-1003)
**File:** `fixes/fix_jaeger_trace_context/trace_context_patch.py`
**Problem:** 5% sampling causes missing parent spans in multi-region traces
**Solution:**
- Ensure `trace_id` + `span_id` propagation via Redis Streams
- Force parent context injection for cross-region messages
- Implement trace context serialization helper
- Add sampling override for critical paths

**Key Implementation:**
```python
def inject_trace_context(redis_message: dict) -> dict:
    """
    Inject OpenTelemetry trace context into Redis Stream messages

    Ensures parent/child span continuity across regions even with
    head-based sampling (5% sample rate).
    """
    from opentelemetry import trace
    from opentelemetry.propagate import inject

    carrier = {}
    inject(carrier)  # Injects traceparent, tracestate headers

    redis_message['_trace_context'] = carrier
    return redis_message

def extract_trace_context(redis_message: dict) -> None:
    """
    Extract and restore trace context from Redis Stream message
    """
    from opentelemetry.propagate import extract

    carrier = redis_message.get('_trace_context', {})
    ctx = extract(carrier)

    # Attach context to current span
    token = context.attach(ctx)
    return token  # Must be detached later
```

**Test:** `fixes/fix_jaeger_trace_context/jaeger_trace_tests.py`
- Simulate multi-region evaluation (us-east-1 → eu-west-1)
- Force 5% sampling
- Query Jaeger API for trace completeness
- Assert 100% parent-child continuity

---

#### 3. Database Index Optimization (TARS-1004)
**File:** `fixes/fix_database_indexes/v1_0_1_add_indexes.sql`
**Problem:** API queries slow (p95: 500ms)
**Solution:** 3 composite indexes + pagination support

**SQL Migration:**
```sql
-- Index 1: Evaluation queries by agent + timestamp
CREATE INDEX CONCURRENTLY idx_evaluations_agent_created
ON evaluations(agent_id, created_at DESC, status)
WHERE deleted_at IS NULL;

-- Index 2: Evaluation queries by region + status
CREATE INDEX CONCURRENTLY idx_evaluations_region_status
ON evaluations(region, status, created_at DESC)
WHERE deleted_at IS NULL;

-- Index 3: Agent queries by type + active status
CREATE INDEX CONCURRENTLY idx_agents_type_active
ON agents(agent_type, is_active, last_updated DESC);

-- Analyze tables
ANALYZE evaluations;
ANALYZE agents;
```

**Expected Improvement:** 500ms → <100ms (80% reduction)

**Test:** `fixes/fix_database_indexes/db_index_tests.py`
- Benchmark queries before/after indexes
- Load 10,000+ evaluation records
- Run typical API queries (filter by agent, region, date range)
- Assert p95 < 100ms
- Verify `EXPLAIN ANALYZE` uses indexes

---

#### 4. PPO Memory Leak Fix (TARS-1005) 🔴 CRITICAL
**File:** `fixes/fix_ppo_memory_leak/ppo_memory_patch.py`
**Problem:** PPO agent memory grows 500MB → 4GB+ over 24 hours
**Root Cause:** TensorFlow graph accumulation + experience buffer not cleared

**Solution:**
```python
class PPOAgent:
    def train_step(self, experiences):
        # BEFORE FIX: Memory accumulates
        # self.buffer.extend(experiences)
        # loss = self.model.train_on_batch(...)

        # AFTER FIX: Explicit clearing
        with tf.GradientTape() as tape:
            loss = self._compute_loss(experiences)
            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        # Clear experience buffer after training
        self.buffer.clear()

        # Force TensorFlow graph cleanup
        tf.keras.backend.clear_session()
        gc.collect()

        return loss
```

**Test:** `fixes/fix_ppo_memory_leak/ppo_memory_tests.py`
- 48-hour soak test (accelerated to 30min via 100x speed)
- Monitor memory every 10 seconds
- Assert memory stable at <1GB
- Validate no performance degradation

---

### Priority 2: Release Engineering (3 files)

#### 5. Upgrade Playbook
**File:** `release/v1_0_1/upgrade_playbook.md`
**Sections:**
- Pre-upgrade checks (backup, health validation)
- Zero-downtime upgrade steps (Helm rolling update)
- Database migration (index creation with `CONCURRENTLY`)
- Configuration changes (recording rules deployment)
- Post-upgrade validation (smoke tests)
- Rollback procedures (Helm rollback)

**Key Procedure:**
```bash
# 1. Backup database
kubectl exec postgres-0 -- pg_dump tars > backup_v1.0.0.sql

# 2. Apply recording rules
kubectl apply -f fixes/fix_grafana_query_timeout/recording_rules.yaml
kubectl exec prometheus-0 -- kill -HUP 1

# 3. Upgrade Helm chart (rolling update)
helm upgrade tars charts/tars --version 1.0.1 --set image.tag=v1.0.1

# 4. Apply database indexes (non-blocking)
kubectl exec postgres-0 -- psql -f fixes/fix_database_indexes/v1_0_1_add_indexes.sql

# 5. Validate
curl https://api.tars.dev/health
```

---

#### 6. Regression Test Suite
**File:** `release/v1_0_1/regression_suite_v1_0_1.py`
**Coverage:**
- All 5 hotfixes validated end-to-end
- Integration with existing test suites (Phases 11-13)
- Performance benchmarks (latency, throughput, memory)
- Multi-region scenarios
- Failure injection tests

**Test Structure:**
```python
class V1_0_1_RegressionSuite:
    def test_tars_1001_websocket_reconnection(self):
        """Validate WebSocket auto-reconnect"""

    def test_tars_1002_grafana_performance(self):
        """Validate dashboard load <5s with 5k+ evals"""

    def test_tars_1003_jaeger_trace_continuity(self):
        """Validate 100% trace parent-child links"""

    def test_tars_1004_database_performance(self):
        """Validate API p95 <100ms"""

    def test_tars_1005_ppo_memory_stability(self):
        """Validate PPO memory stable over 48h"""

    def test_integration_all_fixes(self):
        """End-to-end validation of all fixes"""
```

---

#### 7. Build & Packaging Script
**File:** `release/v1_0_1/build_v1_0_1_package.py`
**Functions:**
- Update version strings across codebase (15+ files)
- Run regression test suite
- Build Helm chart package (`tars-1.0.1.tgz`)
- Build Docker images (5 services: dashboard-api, dashboard-frontend, eval-engine, orchestration, automl)
- Generate checksums (SHA256)
- Create release notes
- Tag git repository
- Dry-run mode for validation

**Usage:**
```bash
# Dry run
python release/v1_0_1/build_v1_0_1_package.py --version 1.0.1 --dry-run

# Execute build
python release/v1_0_1/build_v1_0_1_package.py --version 1.0.1 --execute

# Output artifacts:
# - charts/tars-1.0.1.tgz
# - docker images: tars/*:v1.0.1
# - checksums.txt
# - RELEASE_NOTES.md
```

---

### Priority 3: Observability (3 files)

#### 8. Real-Time SLO Monitor
**File:** `telemetry/real_time_slo_monitor.py`
**Purpose:** Streaming SLO violation detection with auto-escalation

**Architecture:**
```python
class RealTimeSLOMonitor:
    """
    Real-time SLO monitoring with Prometheus integration

    Wraps SLOViolationDetector from Phase 14 with:
    - Continuous Prometheus polling (15s interval)
    - Streaming violation detection
    - Auto-escalation to PagerDuty/Slack
    - Time-window analysis (1m, 5m, 1h)
    """

    async def start_monitoring(self):
        """Start real-time monitoring loop"""
        while True:
            metrics = await self.prometheus_client.query(...)
            violations = self.detector.process_metrics(metrics)

            for v in violations:
                if v.severity >= ViolationSeverity.ERROR:
                    await self.escalate(v)

            await asyncio.sleep(15)
```

**Integration:**
- Prometheus API client
- SLOViolationDetector from Phase 14
- PagerDuty incident creation
- Statuspage integration

---

#### 9. Live Regression Monitor
**File:** `telemetry/live_regression_monitor.py`
**Purpose:** Real-time regression prediction from streaming logs

**Architecture:**
```python
class LiveRegressionMonitor:
    """
    Real-time regression detection with ML classifier

    Wraps RegressionClassifier from Phase 14 with:
    - Streaming log ingestion (CloudWatch/Stackdriver)
    - Feature extraction pipeline
    - Real-time ML prediction
    - Auto-incident creation
    - Back-pressure management
    """

    async def monitor_stream(self):
        """Monitor log stream for regressions"""
        async for log_batch in self.log_ingestor.stream():
            features = self.extract_features(log_batch)
            predictions = await self.classifier.predict_batch(features)

            for pred in predictions:
                if pred.classification == RegressionType.PERFORMANCE:
                    await self.create_incident(pred)
```

**Integration:**
- ProductionLogIngestor from Phase 14
- RegressionClassifier from Phase 14
- Statuspage incident API
- Rate limiting for incident creation

---

#### 10. Validation Checklist Runbook
**File:** `docs/runbooks/v1_0_1_validation_checklist.md`
**Sections:**
- Pre-release validation (40+ items)
  - All tests passing
  - Performance benchmarks met
  - Security scans clean
  - Documentation updated
  - Release notes finalized
- Post-deployment validation (20+ items)
  - Health checks green
  - SLO compliance >99%
  - No error spikes
  - Monitoring operational
  - Rollback tested
- Integration with Phase 14 diagnostics
  - Reference [production_diagnostics.md](docs/runbooks/production_diagnostics.md)
  - Reference [post_ga_hardening.md](docs/runbooks/post_ga_hardening.md)

**Checklist Format:**
```markdown
### Pre-Release Validation

#### Code Quality
- [ ] All unit tests passing (pytest)
- [ ] Integration tests passing
- [ ] Regression suite passing (v1.0.1)
- [ ] Code coverage >85%
- [ ] No critical security vulnerabilities (Snyk scan)

#### Performance Benchmarks
- [ ] WebSocket reconnection <30s
- [ ] Grafana dashboard load <5s (5k+ evals)
- [ ] API p95 latency <100ms
- [ ] PPO memory stable <1GB (48h test)
- [ ] Jaeger trace continuity 100%

#### Documentation
- [ ] CHANGELOG.md updated
- [ ] HOTFIX_PLAN.md marked complete
- [ ] Upgrade playbook validated
- [ ] API documentation current
- [ ] Runbooks updated

### Post-Deployment Validation

#### Service Health
- [ ] All pods running (kubectl get pods -n tars)
- [ ] Health endpoints responding
- [ ] Prometheus targets UP
- [ ] Grafana dashboards loading
- [ ] Jaeger traces visible

#### SLO Compliance (first 24 hours)
- [ ] API latency p95 <150ms
- [ ] Error rate <1%
- [ ] Evaluation success rate >99%
- [ ] WebSocket stability >99%
- [ ] No memory leaks detected
```

---

## Implementation Priority Order

### Day 1: Core Fixes
1. ✅ TARS-1001: WebSocket reconnection (DONE)
2. ✅ TARS-1002: Prometheus recording rules (DONE)
3. 🔄 TARS-1002: Grafana dashboard patch + tests
4. 🔄 TARS-1004: Database indexes + tests (High impact, low risk)
5. 🔄 TARS-1003: Jaeger trace context + tests

### Day 2: Critical Fix + Release Prep
6. 🔴 TARS-1005: PPO memory leak + 48h test (Critical)
7. 🔄 Upgrade playbook (Needed for deployment)
8. 🔄 Regression suite (Gate for release)

### Day 3: Build + Observability
9. 🔄 Build & packaging script
10. 🔄 Real-time SLO monitor
11. 🔄 Live regression monitor
12. 🔄 Validation checklist

---

## Testing Strategy

### Unit Tests
- Each fix has dedicated test file
- Mock external dependencies (WebSocket server, Prometheus, Jaeger)
- Target: >85% code coverage

### Integration Tests
- Multi-component scenarios
- Real service dependencies (PostgreSQL, Redis, Prometheus)
- Docker Compose test environment

### Performance Tests
- Load testing with realistic data volumes
- Benchmarking before/after fixes
- Soak testing for memory leak fix (48h accelerated)

### Regression Tests
- Comprehensive v1.0.1 suite
- Validates all 5 hotfixes
- Must pass 100% before release

---

## Success Metrics

### Performance Targets
| Fix | Metric | Target | Status |
|-----|--------|--------|--------|
| TARS-1001 | WebSocket reconnect | <30s | ✅ <5s avg |
| TARS-1002 | Grafana load (5k evals) | <5s | 🔄 Projected 4.5s |
| TARS-1003 | Trace continuity | 100% | ⏳ Pending |
| TARS-1004 | API p95 latency | <100ms | ⏳ Pending |
| TARS-1005 | PPO memory (48h) | <1GB | ⏳ Pending |

### Release Readiness
- [ ] All 5 hotfixes implemented
- [ ] All tests passing (100%)
- [ ] Performance benchmarks validated
- [ ] Helm chart packaged
- [ ] Docker images built
- [ ] Release notes finalized
- [ ] Upgrade playbook validated

---

## Common Pitfalls & Tips

### Database Indexes
- ⚠️ Always use `CREATE INDEX CONCURRENTLY` to avoid table locks
- ⚠️ Run `ANALYZE` after index creation
- ⚠️ Test indexes with `EXPLAIN ANALYZE`
- ⚠️ Monitor index size growth

### PPO Memory Leak
- ⚠️ TensorFlow requires explicit `clear_session()` calls
- ⚠️ Use `torch.no_grad()` for inference
- ⚠️ Clear buffers after each training step
- ⚠️ Profile with `memory_profiler` and `py-spy`

### Jaeger Tracing
- ⚠️ Trace context must be serialized for Redis Streams
- ⚠️ Use `inject()` and `extract()` from `opentelemetry.propagate`
- ⚠️ Test with actual 5% sampling configuration
- ⚠️ Query Jaeger API to validate trace continuity

### Grafana Recording Rules
- ⚠️ Validate recording rules: `promtool check rules recording_rules.yaml`
- ⚠️ Monitor rule evaluation time in Prometheus
- ⚠️ Update dashboard JSON carefully (test in Grafana UI first)
- ⚠️ Use recording rules in alerts for consistent metrics

---

## File Locations Reference

```
VDS_TARS/
├── fixes/
│   ├── fix_websocket_reconnect/
│   │   ├── websocket_client_patch.py ✅ (680 LOC)
│   │   └── websocket_reconnect_test.py ✅ (850 LOC)
│   ├── fix_grafana_query_timeout/
│   │   ├── recording_rules.yaml ✅ (450 LOC)
│   │   ├── grafana_dashboard_patch.json ⏳
│   │   └── grafana_query_tests.py ⏳
│   ├── fix_jaeger_trace_context/
│   │   ├── trace_context_patch.py ⏳
│   │   └── jaeger_trace_tests.py ⏳
│   ├── fix_database_indexes/
│   │   ├── v1_0_1_add_indexes.sql ⏳
│   │   └── db_index_tests.py ⏳
│   └── fix_ppo_memory_leak/
│       ├── ppo_memory_patch.py ⏳
│       └── ppo_memory_tests.py ⏳
├── release/
│   └── v1_0_1/
│       ├── upgrade_playbook.md ⏳
│       ├── regression_suite_v1_0_1.py ⏳
│       └── build_v1_0_1_package.py ⏳
├── telemetry/
│   ├── real_time_slo_monitor.py ⏳
│   └── live_regression_monitor.py ⏳
├── docs/
│   └── runbooks/
│       └── v1_0_1_validation_checklist.md ⏳
└── PHASE14_1_IMPLEMENTATION_PROGRESS.md ✅
```

---

## Next Session Start Command

```bash
# Load this guide
cat PHASE14_1_QUICKSTART.md

# Check progress
cat PHASE14_1_IMPLEMENTATION_PROGRESS.md

# Continue with next priority file (Grafana dashboard patch)
# Or specify which fix to implement next
```

---

## Questions for User (if needed)

Before continuing implementation, clarify:
1. **PPO Memory Leak Test Duration**: Run full 48h test or 30-min accelerated test?
2. **Database Migration Strategy**: Apply indexes during Helm upgrade or as separate step?
3. **Deployment Target**: Which environment for v1.0.1 release (staging, production, both)?
4. **Monitoring Integration**: Deploy real-time monitors with v1.0.1 or as separate v1.0.2 feature?

---

**Status:** Ready for Phase 14.1 continuation
**Completion:** 2/11 deliverables (18%)
**Estimated Remaining:** 8-12 hours of focused implementation

**End of Quick Start Guide**
