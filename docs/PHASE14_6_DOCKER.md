# Phase 14.6 - Docker Deployment Guide

**T.A.R.S. v1.0.2-pre - Dockerized Observability Stack**

This guide covers deploying Phase 14.6 monitoring and retrospective tools using Docker and Kubernetes.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Docker Image](#docker-image)
3. [Docker Compose](#docker-compose)
4. [Kubernetes Deployment](#kubernetes-deployment)
5. [Configuration](#configuration)
6. [Data Persistence](#data-persistence)
7. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Build the Docker Image

```bash
# Build from source
docker build -t tars-observability:1.0.2-pre .

# Or pull from registry (when available)
docker pull ghcr.io/veleron-dev/tars/tars-observability:latest
```

### Run Retrospective Generator

```bash
# Run with test data
docker run -v $(pwd)/test_data:/test_data:ro \
  -v $(pwd)/output:/data/output \
  tars-observability:1.0.2-pre \
  tars-retro --auto --output-dir /data/output

# Run with production data
docker run -v $(pwd)/data:/data \
  tars-observability:1.0.2-pre \
  tars-retro --auto
```

---

## Docker Image

### Image Details

- **Base Image:** `python:3.11-slim`
- **Size:** ~250 MB (compressed)
- **User:** Non-root user `tars` (UID 1000)
- **Entry Point:** `tars-retro` (retrospective generator)
- **Default Command:** `--auto --output-dir /data/output`

### Included CLI Tools

All 6 Phase 14.6 CLI tools are available:

1. `tars-ga-kpi` - GA Day KPI collection
2. `tars-stability-monitor` - 7-day stability monitoring
3. `tars-anomaly-detector` - Anomaly detection
4. `tars-health-report` - Health reporting
5. `tars-regression-analyzer` - Regression analysis
6. `tars-retro` - Retrospective generation

### Running Individual Tools

```bash
# GA Day KPI Collection
docker run -v $(pwd)/data:/data \
  tars-observability:1.0.2-pre \
  tars-ga-kpi --prometheus-url http://prometheus:9090 \
              --output-dir /data/ga_kpis \
              --ga-timestamp "2025-11-18T00:00:00Z"

# Day 1 Stability Monitoring
docker run -v $(pwd)/data:/data \
  tars-observability:1.0.2-pre \
  tars-stability-monitor --prometheus-url http://prometheus:9090 \
                         --output-dir /data/stability \
                         --day-number 1 \
                         --ga-baseline /data/ga_kpis/ga_kpi_summary.json

# Anomaly Detection
docker run -v $(pwd)/data:/data \
  tars-observability:1.0.2-pre \
  tars-anomaly-detector --prometheus-url http://prometheus:9090 \
                        --output-file /data/anomalies/anomaly_events.json \
                        --baseline /data/ga_kpis/ga_kpi_summary.json \
                        --z-threshold 3.0

# Health Report (Day 1)
docker run -v $(pwd)/data:/data \
  tars-observability:1.0.2-pre \
  tars-health-report --stability-data /data/stability/day_01_summary.json \
                     --anomaly-data /data/anomalies/anomaly_events.json \
                     --output-file /data/health/day_01_HEALTH.json \
                     --day-number 1

# Regression Analysis (Day 7)
docker run -v $(pwd)/data:/data \
  tars-observability:1.0.2-pre \
  tars-regression-analyzer --ga-baseline /data/ga_kpis/ga_kpi_summary.json \
                           --7day-summaries /data/stability \
                           --output /data/regression/regression_summary.json

# Retrospective Generation (Day 7)
docker run -v $(pwd)/data:/data \
  tars-observability:1.0.2-pre \
  tars-retro --auto --output-dir /data/output
```

---

## Docker Compose

### Start Services

The `docker-compose.yaml` provides multiple service profiles:

```bash
# 1. Run GA Day KPI Collection
docker-compose --profile ga-day up ga-kpi-collector

# 2. Run Daily Monitoring (Day 1)
DAY_NUMBER=1 docker-compose --profile daily up stability-monitor health-reporter

# 3. Run Continuous Anomaly Detection
docker-compose --profile monitoring up -d anomaly-detector

# 4. Run Day 7 Full Analysis
docker-compose --profile day7 up regression-analyzer retrospective-generator

# 5. Run Retrospective Only (default)
docker-compose up retrospective-generator

# 6. Run All Services
docker-compose --profile ga-day --profile daily --profile monitoring --profile day7 up
```

### Environment Variables

Configure via `.env` file:

```bash
# .env
PROMETHEUS_URL=http://prometheus:9090
GA_TIMESTAMP=2025-11-18T00:00:00Z
DAY_NUMBER=1
Z_THRESHOLD=3.0
EWMA_ALPHA=0.3
TZ=UTC
```

### View Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f retrospective-generator

# Last 100 lines
docker-compose logs --tail=100 tars-observability
```

### Stop Services

```bash
# Stop all services
docker-compose down

# Stop and remove volumes
docker-compose down -v
```

---

## Kubernetes Deployment

### Option 1: Kubernetes CronJob (Recommended)

Deploy Phase 14.6 as scheduled CronJobs:

```yaml
# k8s/tars-observability-cronjob.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: tars-observability

---
apiVersion: v1
kind: ConfigMap
metadata:
  name: tars-config
  namespace: tars-observability
data:
  PROMETHEUS_URL: "http://prometheus.monitoring.svc.cluster.local:9090"
  GA_TIMESTAMP: "2025-11-18T00:00:00Z"
  Z_THRESHOLD: "3.0"

---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: tars-data-pvc
  namespace: tars-observability
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 10Gi
  storageClassName: standard

---
# GA Day KPI Collection (Run once on GA Day)
apiVersion: batch/v1
kind: Job
metadata:
  name: tars-ga-kpi
  namespace: tars-observability
spec:
  template:
    spec:
      containers:
      - name: ga-kpi
        image: ghcr.io/veleron-dev/tars/tars-observability:latest
        command: ["tars-ga-kpi"]
        args:
          - "--prometheus-url"
          - "$(PROMETHEUS_URL)"
          - "--output-dir"
          - "/data/ga_kpis"
          - "--ga-timestamp"
          - "$(GA_TIMESTAMP)"
        envFrom:
          - configMapRef:
              name: tars-config
        volumeMounts:
          - name: data
            mountPath: /data
      restartPolicy: OnFailure
      volumes:
        - name: data
          persistentVolumeClaim:
            claimName: tars-data-pvc

---
# Daily Stability Monitoring (Run daily at 11:59 PM)
apiVersion: batch/v1
kind: CronJob
metadata:
  name: tars-stability-monitor
  namespace: tars-observability
spec:
  schedule: "59 23 * * *"  # Daily at 11:59 PM
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: stability-monitor
            image: ghcr.io/veleron-dev/tars/tars-observability:latest
            command: ["tars-stability-monitor"]
            args:
              - "--prometheus-url"
              - "$(PROMETHEUS_URL)"
              - "--output-dir"
              - "/data/stability"
              - "--day-number"
              - "$(DAY_NUMBER)"
              - "--ga-baseline"
              - "/data/ga_kpis/ga_kpi_summary.json"
            envFrom:
              - configMapRef:
                  name: tars-config
            volumeMounts:
              - name: data
                mountPath: /data
          restartPolicy: OnFailure
          volumes:
            - name: data
              persistentVolumeClaim:
                claimName: tars-data-pvc

---
# Daily Health Reporting (Run daily at 12:05 AM)
apiVersion: batch/v1
kind: CronJob
metadata:
  name: tars-health-reporter
  namespace: tars-observability
spec:
  schedule: "5 0 * * *"  # Daily at 12:05 AM
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: health-reporter
            image: ghcr.io/veleron-dev/tars/tars-observability:latest
            command: ["tars-health-report"]
            args:
              - "--stability-data"
              - "/data/stability/day_$(DAY_NUMBER)_summary.json"
              - "--anomaly-data"
              - "/data/anomalies/anomaly_events.json"
              - "--output-file"
              - "/data/health/day_$(DAY_NUMBER)_HEALTH.json"
              - "--day-number"
              - "$(DAY_NUMBER)"
            envFrom:
              - configMapRef:
                  name: tars-config
            volumeMounts:
              - name: data
                mountPath: /data
          restartPolicy: OnFailure
          volumes:
            - name: data
              persistentVolumeClaim:
                claimName: tars-data-pvc

---
# Day 7 Retrospective (Run on Day 7)
apiVersion: batch/v1
kind: Job
metadata:
  name: tars-retrospective-day7
  namespace: tars-observability
spec:
  template:
    spec:
      containers:
      - name: retrospective
        image: ghcr.io/veleron-dev/tars/tars-observability:latest
        command: ["tars-retro"]
        args:
          - "--auto"
          - "--output-dir"
          - "/data/output"
        volumeMounts:
          - name: data
            mountPath: /data
      restartPolicy: OnFailure
      volumes:
        - name: data
          persistentVolumeClaim:
            claimName: tars-data-pvc
```

**Deploy:**

```bash
kubectl apply -f k8s/tars-observability-cronjob.yaml
```

**Monitor Jobs:**

```bash
# View CronJobs
kubectl get cronjobs -n tars-observability

# View Jobs
kubectl get jobs -n tars-observability

# View Pods
kubectl get pods -n tars-observability

# View logs
kubectl logs -n tars-observability job/tars-ga-kpi
kubectl logs -n tars-observability -l job-name=tars-stability-monitor
```

### Option 2: Kubernetes Deployment (Continuous Anomaly Detection)

For continuous anomaly detection:

```yaml
# k8s/tars-anomaly-detector-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: tars-anomaly-detector
  namespace: tars-observability
spec:
  replicas: 1
  selector:
    matchLabels:
      app: tars-anomaly-detector
  template:
    metadata:
      labels:
        app: tars-anomaly-detector
    spec:
      containers:
      - name: anomaly-detector
        image: ghcr.io/veleron-dev/tars/tars-observability:latest
        command: ["tars-anomaly-detector"]
        args:
          - "--prometheus-url"
          - "$(PROMETHEUS_URL)"
          - "--output-file"
          - "/data/anomalies/anomaly_events.json"
          - "--baseline"
          - "/data/ga_kpis/ga_kpi_summary.json"
          - "--z-threshold"
          - "$(Z_THRESHOLD)"
        envFrom:
          - configMapRef:
              name: tars-config
        volumeMounts:
          - name: data
            mountPath: /data
        resources:
          requests:
            memory: "256Mi"
            cpu: "100m"
          limits:
            memory: "512Mi"
            cpu: "500m"
      volumes:
        - name: data
          persistentVolumeClaim:
            claimName: tars-data-pvc
```

**Deploy:**

```bash
kubectl apply -f k8s/tars-anomaly-detector-deployment.yaml
```

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TARS_DATA_DIR` | `/data` | Base directory for data storage |
| `TARS_OUTPUT_DIR` | `/data/output` | Output directory for reports |
| `PROMETHEUS_URL` | `http://prometheus:9090` | Prometheus endpoint |
| `GA_TIMESTAMP` | `2025-11-18T00:00:00Z` | GA Day timestamp |
| `DAY_NUMBER` | `1` | Current day number (1-7) |
| `Z_THRESHOLD` | `3.0` | Z-score threshold for anomalies |
| `EWMA_ALPHA` | `0.3` | EWMA smoothing factor |
| `TZ` | `UTC` | Timezone |

### Volume Mounts

**Required volumes:**

- `/data` - Main data directory (read/write)
  - `/data/ga_kpis` - GA Day KPI data
  - `/data/stability` - 7-day stability summaries
  - `/data/anomalies` - Anomaly events
  - `/data/health` - Health reports
  - `/data/regression` - Regression analysis
  - `/data/output` - Final retrospective outputs

**Optional volumes:**

- `/test_data` - Test data (read-only)
- `/docs` - Documentation (read-only)

---

## Data Persistence

### Local Development

```bash
# Create data directories
mkdir -p data/{ga_kpis,stability,anomalies,health,regression,output}

# Mount data directory
docker run -v $(pwd)/data:/data tars-observability:1.0.2-pre
```

### Production (Kubernetes)

Use `PersistentVolumeClaim` for data persistence:

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: tars-data-pvc
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 10Gi
  storageClassName: standard  # Or your preferred storage class
```

### Backup Strategy

```bash
# Backup data directory
kubectl cp tars-observability/pod-name:/data ./backup/$(date +%Y%m%d)

# Or use persistent volume snapshots
kubectl apply -f k8s/tars-data-snapshot.yaml
```

---

## Troubleshooting

### Issue: "Permission denied" on `/data`

**Cause:** Container runs as non-root user (UID 1000).

**Solution:**
```bash
# Set correct permissions
sudo chown -R 1000:1000 ./data

# Or run with user override (not recommended)
docker run --user root -v $(pwd)/data:/data tars-observability:1.0.2-pre
```

### Issue: "FileNotFoundError: ga_kpi_summary.json"

**Cause:** GA Day KPI collection hasn't run yet.

**Solution:**
```bash
# Run GA Day KPI collection first
docker-compose --profile ga-day up ga-kpi-collector
```

### Issue: Docker build fails with "No module named..."

**Cause:** Dependencies not installed correctly.

**Solution:**
```bash
# Clear Docker cache and rebuild
docker build --no-cache -t tars-observability:1.0.2-pre .
```

### Issue: Kubernetes pod stuck in "CrashLoopBackOff"

**Cause:** Missing configuration or data.

**Solution:**
```bash
# Check logs
kubectl logs -n tars-observability pod-name

# Verify ConfigMap
kubectl get configmap -n tars-observability tars-config -o yaml

# Verify PVC is bound
kubectl get pvc -n tars-observability
```

---

## Best Practices

1. **Use Named Volumes:** For production, use named Docker volumes or Kubernetes PVCs
2. **Set Resource Limits:** Prevent resource exhaustion in Kubernetes
3. **Enable Health Checks:** Monitor container health
4. **Rotate Logs:** Use log rotation to prevent disk full
5. **Backup Data:** Regular backups of `/data` directory
6. **Security:** Run as non-root user (default: UID 1000)
7. **Network Isolation:** Use Docker networks or Kubernetes network policies

---

## Additional Resources

- [Phase 14.6 Quickstart](./PHASE14_6_QUICKSTART.md)
- [Production Runbook](./PHASE14_6_PRODUCTION_RUNBOOK.md)
- [GitHub Repository](https://github.com/veleron-dev/tars)
- [Docker Hub](https://hub.docker.com/r/veleron-dev/tars-observability)

---

**Generated:** 2025-11-26
**Version:** v1.0.2-pre
**Phase:** 14.6 - Docker Deployment Guide
