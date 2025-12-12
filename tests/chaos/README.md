# T.A.R.S. Chaos Testing Harness

Comprehensive chaos testing suite to validate T.A.R.S. system resilience under adverse conditions.

## Overview

This testing harness validates:
- **Load handling**: Sustained and spike traffic patterns
- **JWT resilience**: Token issuance/verification under stress
- **Service resilience**: Graceful recovery from pod failures
- **Redis outage**: Fallback behavior when Redis is unavailable

## Test Categories

### 1. k6 Load Tests

Performance and load tests using k6.

| Test | Duration | Target | Thresholds |
|------|----------|--------|------------|
| `sustained-load.js` | 10 min | 100 RPS sustained | P95 <500ms, <5% errors |
| `spike-load.js` | 6 min | 50→500→50 RPS spikes | P95 <1s, <15% errors |
| `jwt-stress.js` | 6 min | 200 concurrent users | P95 <500ms, <5% errors |

### 2. Resilience Tests

Service disruption and recovery tests using Python.

| Test | Target | Expected Behavior |
|------|--------|-------------------|
| `redis-outage.py` | Redis pod | Graceful degradation, fallback to legacy mode |
| `pod-kill-test.py` | Service pods | Recovery <60s, minimal downtime |

## Prerequisites

### Software Requirements

- **k6** (>= v0.45.0): [Installation guide](https://k6.io/docs/getting-started/installation/)
- **Python 3.8+**: With `requests` library
- **kubectl**: For Kubernetes pod manipulation
- **curl**: For manual API testing

```bash
# Install k6 (Linux/macOS)
curl -L https://github.com/grafana/k6/releases/download/v0.45.0/k6-v0.45.0-linux-amd64.tar.gz | tar xvz
sudo mv k6 /usr/local/bin/

# Install Python dependencies
pip install requests

# Verify kubectl access
kubectl get pods -n tars
```

### System Requirements

- T.A.R.S. system deployed and running
- Admin credentials configured
- Kubernetes cluster access (for resilience tests)
- Sufficient resources (8+ vCPUs, 16GB+ RAM recommended)

## Quick Start

### Run All Tests

```bash
cd tests/chaos
chmod +x run-tests.sh
./run-tests.sh
```

This will:
1. Run all k6 load tests (sustained, spike, JWT stress)
2. Prompt for confirmation before resilience tests
3. Run Redis outage and pod kill tests
4. Generate summary report

### Run Individual Tests

#### k6 Load Tests

```bash
# Sustained load (10 min, 100 RPS)
k6 run k6/sustained-load.js

# Spike load (6 min, 50→500 RPS)
k6 run k6/spike-load.js

# JWT stress (6 min, 200 concurrent users)
k6 run k6/jwt-stress.js
```

#### Resilience Tests

```bash
# Redis outage simulation
python3 resilience/redis-outage.py

# Pod kill test (Orchestration, AutoML, HyperSync)
python3 resilience/pod-kill-test.py
```

### Skip Specific Test Categories

```bash
# Skip k6 tests
./run-tests.sh --skip-k6

# Skip resilience tests
./run-tests.sh --skip-resilience
```

## Configuration

### Environment Variables

Set these before running tests:

```bash
# API endpoint
export BASE_URL="http://localhost:3001"

# Admin credentials
export ADMIN_USERNAME="admin"
export ADMIN_PASSWORD="admin123"

# Kubernetes namespace
export NAMESPACE="tars"
```

### k6 Test Parameters

Modify test files to adjust:
- **Duration**: `duration: '10m'`
- **RPS target**: `rate: 100`
- **VU count**: `maxVUs: 200`
- **Thresholds**: `http_req_duration: ['p(95)<500']`

### Resilience Test Parameters

Edit Python scripts:
- `TEST_DURATION`: Test duration in seconds
- `MAX_WAIT_SEC`: Max recovery wait time
- `REDIS_POD_LABEL`: Redis pod selector

## Test Scenarios

### Scenario 1: Sustained Load

**Objective**: Validate system stability under continuous load.

**Test**: `k6/sustained-load.js`

**Load Profile**:
- 100 RPS constant for 10 minutes
- Mixed endpoint distribution (JWT status, agents, API keys, health)

**Success Criteria**:
- ✅ P95 latency <500ms
- ✅ P99 latency <1000ms
- ✅ Error rate <5%
- ✅ No memory leaks or OOM errors

### Scenario 2: Spike Load

**Objective**: Validate system elasticity under traffic spikes.

**Test**: `k6/spike-load.js`

**Load Profile**:
- 50 RPS baseline
- Spike to 500 RPS for 30s
- Return to 50 RPS
- Repeat spike

**Success Criteria**:
- ✅ P95 latency <1s
- ✅ Error rate <15% (during spike)
- ✅ System recovers to baseline performance
- ✅ No crashes or restarts

### Scenario 3: JWT Stress

**Objective**: Validate JWT issuance/verification at scale.

**Test**: `k6/jwt-stress.js`

**Load Profile**:
- Ramp from 10 to 200 concurrent users
- Each user: login → 5 authenticated requests → refresh token
- Sustain 200 users for 3 minutes

**Success Criteria**:
- ✅ Login P95 <200ms
- ✅ Token verification P95 <50ms
- ✅ Error rate <5%
- ✅ No token corruption or security issues

### Scenario 4: Redis Outage

**Objective**: Validate graceful degradation when Redis fails.

**Test**: `resilience/redis-outage.py`

**Disruption**:
1. Baseline: 30s normal operation
2. Kill Redis pod
3. Continue load for 60s (degraded mode)
4. Wait for Redis recovery
5. Verify full recovery

**Success Criteria**:
- ✅ System continues operating (degraded)
- ✅ JWT fallback to legacy mode
- ✅ API key fallback to in-memory store
- ✅ Success rate >50% during outage
- ✅ Full recovery after Redis restart

### Scenario 5: Pod Kill

**Objective**: Validate graceful recovery from pod failures.

**Test**: `resilience/pod-kill-test.py`

**Disruption**:
For each service (Orchestration, AutoML, HyperSync):
1. Verify service healthy
2. Kill pod
3. Measure downtime
4. Wait for recovery
5. Verify full recovery

**Success Criteria**:
- ✅ Recovery time <30s (PASS)
- ✅ Recovery time <60s (WARN)
- ✅ Max downtime <10s
- ✅ No data loss or corruption

## Results

### Output Files

Results are saved to `results/YYYYMMDD_HHMMSS/`:

```
results/20251115_143000/
├── sustained-load.json          # k6 raw metrics
├── sustained-load-summary.json  # k6 summary
├── sustained-load.log           # Test output
├── spike-load.json
├── spike-load-summary.json
├── spike-load.log
├── jwt-stress.json
├── jwt-stress-summary.json
├── jwt-stress.log
├── redis-outage.log             # Resilience test output
└── pod-kill-test.log
```

### Interpreting Results

#### k6 Metrics

- **http_req_duration**: Request latency (P95, P99)
- **http_req_failed**: % of failed requests
- **errors**: Custom error rate
- **\*_duration**: Custom timing metrics

#### Resilience Metrics

- **Recovery Time**: Time to restore service (seconds)
- **Max Downtime**: Longest continuous outage (seconds)
- **Availability**: % successful requests during test
- **Success Rate**: % requests that succeeded

### Example Output

```
=== JWT Stress Test Summary ===
Total Requests: 15234
Successful Logins: 2456
Failed Logins: 12
Error Rate: 0.48%
Avg Login Duration: 45.23ms
Avg Verification Duration: 3.12ms
P95 Request Duration: 287.45ms

✅ PASS: All thresholds met
```

## Troubleshooting

### k6 Test Failures

**Issue**: High error rate (>5%)

**Diagnosis**:
```bash
# Check service logs
kubectl logs -l app.kubernetes.io/name=tars -n tars --tail=100

# Check resource usage
kubectl top pods -n tars
```

**Solutions**:
- Increase service replicas
- Increase resource limits
- Scale database/Redis

**Issue**: High latency (P95 >500ms)

**Solutions**:
- Enable connection pooling
- Increase worker threads
- Optimize database queries

### Resilience Test Failures

**Issue**: Redis outage test shows 100% failure

**Diagnosis**:
```bash
# Check if fallback is working
kubectl logs -l app.kubernetes.io/component=orchestration -n tars | grep "fallback"
```

**Solutions**:
- Verify JWT fallback logic
- Check in-memory API key store
- Review error handling

**Issue**: Pod kill test shows slow recovery (>60s)

**Diagnosis**:
```bash
# Check pod restart time
kubectl describe pod <pod-name> -n tars

# Check readiness probe
kubectl get pods -n tars -w
```

**Solutions**:
- Adjust readiness probe timings
- Optimize startup time
- Increase pod resources

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Chaos Tests

on:
  schedule:
    - cron: '0 2 * * *'  # Daily at 2 AM

jobs:
  chaos-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Install k6
        run: |
          curl -L https://github.com/grafana/k6/releases/download/v0.45.0/k6-v0.45.0-linux-amd64.tar.gz | tar xvz
          sudo mv k6 /usr/local/bin/

      - name: Run k6 tests
        run: |
          cd tests/chaos
          ./run-tests.sh --skip-resilience

      - name: Upload results
        uses: actions/upload-artifact@v3
        with:
          name: chaos-test-results
          path: tests/chaos/results/
```

## Best Practices

### Before Running Tests

1. **Backup data**: Ensure recent backups exist
2. **Non-production**: Run in staging/test environment
3. **Notify team**: Inform team of planned disruption
4. **Monitor**: Set up monitoring dashboards
5. **Resources**: Ensure sufficient cluster resources

### During Tests

1. **Observe metrics**: Watch Prometheus/Grafana dashboards
2. **Check logs**: Monitor service logs for errors
3. **Network**: Ensure stable network connection
4. **Don't interrupt**: Let tests complete fully

### After Tests

1. **Review results**: Analyze all metrics and logs
2. **Document issues**: Create tickets for failures
3. **Cleanup**: Remove test data if needed
4. **Iterate**: Re-run after fixes

## Known Limitations

- **Rate limiting**: May affect test results if enabled
- **Resource contention**: Tests may compete for resources
- **Network latency**: External factors may skew results
- **Kubernetes scheduling**: Pod restart times vary

## Support

For issues or questions:
- Review logs in `results/` directory
- Check [PHASE12_TROUBLESHOOTING.md](../../docs/phase12/PHASE12_TROUBLESHOOTING.md)
- Open GitHub issue with test results attached

---

**Last Updated**: 2025-11-15
**Version**: 1.0.0
**Author**: T.A.R.S. DevOps Team
