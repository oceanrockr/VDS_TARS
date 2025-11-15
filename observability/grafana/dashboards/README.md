# T.A.R.S. Grafana Dashboards

This directory contains production-ready Grafana dashboards for monitoring T.A.R.S. multi-agent system.

## Dashboard Overview

| Dashboard | UID | Purpose | Metrics |
|-----------|-----|---------|---------|
| **Authentication Metrics** | `tars-auth-metrics` | Monitor authentication, authorization, and rate limiting | Login attempts/failures, JWT validation, rate limits, API key activity |
| **Agent Training** | `tars-agent-training` | Track agent learning performance | Reward curves, loss curves, entropy, Nash convergence, hyperparameters |
| **AutoML Optimization** | `tars-automl-optimization` | Monitor hyperparameter search | Trial scores, search progress, Pareto frontiers, parameter importance |
| **HyperSync Flow** | `tars-hypersync-flow` | Track hyperparameter synchronization | Proposals, approvals, drift, consistency scores |
| **System Health** | `tars-system-health` | Overall system monitoring | CPU, memory, latency, errors, restarts, Redis ops, GC |

## Features

All dashboards include:

✅ **Template Variables**: Namespace and service/agent filtering
✅ **Auto-refresh**: 10-second refresh interval
✅ **Prometheus Integration**: All metrics from Prometheus datasource
✅ **Dark Theme**: Production-ready styling
✅ **Alerting Thresholds**: Visual indicators for critical values
✅ **Time Range Control**: Default 1-hour view, customizable
✅ **Export-Ready**: JSON format for import/export

## Installation

### Option 1: Manual Import (Development)

1. Access Grafana UI (default: http://localhost:3000)
2. Navigate to **Dashboards** → **Import**
3. Upload JSON file or paste contents
4. Select Prometheus datasource
5. Click **Import**

### Option 2: Helm Deployment (Production)

The dashboards are automatically deployed via Helm chart:

```bash
# Install with Grafana dashboards
helm install tars ./charts/tars \
  -f charts/tars/values-security.yaml \
  --set grafana.enabled=true \
  --set grafana.dashboards.enabled=true \
  -n tars
```

The Helm chart creates a ConfigMap containing all dashboards and mounts them into Grafana.

### Option 3: Kubernetes ConfigMap (Manual)

```bash
# Create ConfigMap from dashboard files
kubectl create configmap grafana-dashboards \
  --from-file=observability/grafana/dashboards/ \
  -n tars

# Mount to Grafana pod (add to Grafana deployment spec)
volumeMounts:
  - name: dashboards
    mountPath: /etc/grafana/provisioning/dashboards

volumes:
  - name: dashboards
    configMap:
      name: grafana-dashboards
```

## Dashboard Details

### 1. Authentication Metrics (`auth_metrics.json`)

**Purpose**: Monitor security and access control

**Panels**:
- Login failures (5m window)
- JWT validation failures
- Rate limit hits (429 responses)
- Active JWT tokens
- Login rate over time
- JWT validation rate
- JWT validation latency (p50/p95/p99)
- Rate limiting activity
- API key activity (validations, creation, revocation)
- Top users by login attempts (table)

**Key Metrics**:
- `auth_login_attempts_total`
- `auth_login_failures_total`
- `auth_jwt_validations_total`
- `auth_jwt_validation_failures_total`
- `auth_jwt_validation_duration_seconds`
- `rate_limit_requests_total`
- `rate_limit_exceeded_total`
- `auth_api_key_validations_total`

**Alerting Thresholds**:
- Login failures > 10/5m → Red
- JWT validation failures > 100/5m → Red
- Rate limit hits > 50/5m → Red

---

### 2. Agent Training (`agent_training.json`)

**Purpose**: Monitor reinforcement learning agent performance

**Panels**:
- Active agents count
- Max training steps
- Average reward
- Training steps/min
- Reward curves by agent
- Loss curves by agent
- Policy entropy (exploration)
- Nash equilibrium convergence
- Hyperparameter evolution (LR, gamma, epsilon)
- Agent performance summary (table)

**Key Metrics**:
- `agent_training_step`
- `agent_training_reward`
- `agent_training_loss`
- `agent_exploration_entropy`
- `nash_convergence_distance`
- `agent_hyperparameter_value`

**Alerting Thresholds**:
- Average reward < 0.3 → Red
- Average reward 0.3-0.6 → Yellow
- Average reward > 0.6 → Green

---

### 3. AutoML Optimization (`automl_optimization.json`)

**Purpose**: Track automated hyperparameter optimization

**Panels**:
- Active searches count
- Trials completed (1h)
- Best trial score
- Search progress (%)
- Best trial score over time
- Trial completion rate
- Hyperparameter importance (bar chart)
- Pareto frontier (multi-objective)
- Trial duration (p50/p95/p99)
- Top 10 trials by score (table)

**Key Metrics**:
- `automl_active_searches`
- `automl_trials_completed_total`
- `automl_best_trial_score`
- `automl_search_progress`
- `automl_hyperparam_importance`
- `automl_pareto_frontier_score`
- `automl_trial_duration_seconds`

**Alerting Thresholds**:
- Best trial score < 0.5 → Red
- Best trial score 0.5-0.7 → Yellow
- Best trial score > 0.7 → Green

---

### 4. HyperSync Flow (`hypersync_flow.json`)

**Purpose**: Monitor cross-agent hyperparameter synchronization

**Panels**:
- Proposals (1h)
- Approvals (1h)
- Approval rate (%)
- Max drift magnitude
- Proposal activity (proposals/approvals/rejections)
- Hyperparameter drift by agent
- Global consistency score
- Sync operation duration (p50/p95/p99)
- Sync operation rate
- Recent proposals (table)

**Key Metrics**:
- `hypersync_proposals_total`
- `hypersync_approvals_total`
- `hypersync_rejections_total`
- `hypersync_drift_magnitude`
- `hypersync_consistency_score`
- `hypersync_sync_duration_seconds`
- `hypersync_sync_operations_total`

**Alerting Thresholds**:
- Approval rate < 50% → Red
- Approval rate 50-80% → Yellow
- Approval rate > 80% → Green
- Drift magnitude > 0.1 → Red

---

### 5. System Health (`system_health.json`)

**Purpose**: Overall system resource monitoring and health

**Panels**:
- Service status (Orchestration, AutoML, HyperSync, Dashboard API)
- CPU usage by service
- Memory usage by service
- Request latency (p95/p99)
- Request rate by service
- 5xx error rate by service
- Pod restarts
- Redis operations (GET, SET, total)
- Python GC collections by generation

**Key Metrics**:
- `up`
- `process_cpu_seconds_total`
- `process_resident_memory_bytes`
- `http_request_duration_seconds`
- `http_requests_total`
- `kube_pod_container_status_restarts_total`
- `redis_commands_processed_total`
- `python_gc_collections_total`

**Alerting Thresholds**:
- Service down (up = 0) → Red
- Request latency p95 > 1000ms → Red

---

## Template Variables

All dashboards support template variables for filtering:

| Variable | Type | Source | Purpose |
|----------|------|--------|---------|
| `namespace` | Query | `label_values(up, namespace)` | Filter by Kubernetes namespace |
| `agent_id` | Query | `label_values(agent_training_step, agent_id)` | Filter by specific agent(s) |
| `search_id` | Query | `label_values(automl_best_trial_score, search_id)` | Filter by AutoML search |
| `service` | Query | `label_values(up, job)` | Filter by service name |

Variables support:
- **Multi-select**: Select multiple agents/services
- **All option**: View all items
- **Regex filtering**: Use wildcards
- **Auto-refresh**: Variables update dynamically

## Customization

### Changing Refresh Interval

Edit dashboard JSON:

```json
"refresh": "10s"  // Change to "30s", "1m", "5m", etc.
```

### Adjusting Time Range

Edit dashboard JSON:

```json
"time": {
  "from": "now-1h",  // Change to "now-6h", "now-24h", etc.
  "to": "now"
}
```

### Adding Alerts

1. Click panel title → **Edit**
2. Go to **Alert** tab
3. Configure alert rule using Prometheus queries
4. Set notification channel

### Modifying Thresholds

Edit panel JSON:

```json
"thresholds": {
  "mode": "absolute",
  "steps": [
    { "color": "green", "value": null },
    { "color": "yellow", "value": 50 },
    { "color": "red", "value": 100 }
  ]
}
```

## Prometheus Datasource Configuration

Dashboards expect a Prometheus datasource named **"Prometheus"**.

Configure in Grafana:
1. **Configuration** → **Data Sources** → **Add data source**
2. Select **Prometheus**
3. Set URL: `http://prometheus:9090` (or your Prometheus endpoint)
4. Click **Save & Test**

For Helm deployment, this is configured automatically via `values-security.yaml`.

## Troubleshooting

### No Data in Panels

**Cause**: Prometheus not scraping metrics

**Solution**:
```bash
# Check Prometheus targets
kubectl port-forward -n tars svc/prometheus 9090:9090
# Visit http://localhost:9090/targets

# Check if metrics exist
curl http://localhost:9090/api/v1/query?query=up
```

### Dashboard Variables Not Populating

**Cause**: Metric labels missing or Prometheus datasource misconfigured

**Solution**:
1. Verify Prometheus datasource is selected
2. Check metrics have required labels (namespace, agent_id, etc.)
3. Run raw Prometheus query to verify label values exist

### Panels Show "No Data"

**Cause**: Metric name mismatch or no data in time range

**Solution**:
1. Expand time range (use "Last 24 hours")
2. Verify metric names match your Prometheus metrics
3. Check if services are emitting metrics

### Import Fails

**Cause**: JSON syntax error or missing datasource

**Solution**:
1. Validate JSON using `jq` or online validator
2. Ensure Prometheus datasource exists
3. Check Grafana version compatibility (v8.0+)

## Metrics Reference

See [`PHASE12_IMPLEMENTATION_REPORT.md`](../../../PHASE12_IMPLEMENTATION_REPORT.md) for complete metrics documentation.

## Contributing

When adding new dashboards:

1. Follow existing naming convention: `{category}_{description}.json`
2. Use UID format: `tars-{category}-{description}`
3. Include namespace template variable
4. Set refresh to "10s"
5. Use dark theme
6. Add tags: `["tars", "{category}", ...]`
7. Document in this README
8. Test import before committing

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2025-11-14 | Initial dashboard pack (5 dashboards) |

## License

Part of T.A.R.S. Phase 12 implementation.

## Support

For issues or questions:
- Check Grafana logs: `kubectl logs -n tars deployment/grafana`
- Verify Prometheus metrics: `kubectl port-forward -n tars svc/prometheus 9090:9090`
- Review Phase 12 documentation: [`PHASE12_QUICKSTART.md`](../../../PHASE12_QUICKSTART.md)
