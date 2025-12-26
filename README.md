# T.A.R.S. - Temporal Augmented Retrieval System

**Version:** v1.0.9 (GA)
**Status:** General Availability - MVP Complete + Drop-In Operable
**Date:** December 26, 2025

---

## Overview

**T.A.R.S.** (Temporal Augmented Retrieval System) is an **enterprise-grade, production-ready multi-agent reinforcement learning platform** with comprehensive **compliance, security, and observability** features. Built for on-premises deployment, T.A.R.S. combines multi-agent RL orchestration, AutoML pipelines, cognitive analytics, and advanced RAG capabilities into a unified system for autonomous decision-making and continuous learning.

### What Makes T.A.R.S. Unique

- **Multi-Agent RL Orchestration:** Four specialized agents (DQN, A2C, PPO, DDPG) with Nash equilibrium coordination
- **AutoML Pipeline:** Optuna-powered hyperparameter optimization with real training loops and MLflow tracking
- **Enterprise Security:** AES-256-GCM encryption, RSA-PSS signing, JWT authentication, RBAC, and TLS support
- **Compliance Framework:** SOC 2, ISO 27001, GDPR compliance with runtime enforcement and audit trails
- **Observability Suite:** GA KPIs, anomaly detection, drift monitoring, regression analysis, and automated retrospectives
- **Organization Health Governance:** Multi-repository health aggregation, SLO/SLA policy evaluation, and org-level trend correlation
- **Temporal Intelligence:** Time-lagged correlation, influence scoring, propagation path detection, and causality analysis
- **SLA Intelligence:** Executive readiness dashboards, breach attribution, compliance tracking, and board-ready reporting
- **Advanced Analytics:** Repository health dashboards, alerting engines, trend analyzers, and predictive scoring
- **GA Release Tooling:** Comprehensive validation, production readiness checklists, and secure artifact packaging
- **Operator Enablement:** Production runbooks, incident playbooks, SLA policy templates, and governance policies
- **Cloud-Native Deployment:** Kubernetes Helm charts with HPA, PDB, and zero-downtime updates
- **Complete Privacy:** 100% on-premises with no cloud dependencies or external telemetry

---

## Installation

### Prerequisites

**For Local/Docker Deployment:**
- Python 3.9+
- Redis 6.0+
- 16 GB RAM minimum (32 GB recommended)
- Optional: NVIDIA GPU for LLM inference

**For Kubernetes Deployment:**
- Kubernetes 1.24+ cluster
- Helm 3.0+
- kubectl configured
- cert-manager (optional, for TLS)

### Quick Install

```bash
# Clone repository
git clone https://github.com/oceanrockr/VDS_TARS.git
cd VDS_TARS

# Install dependencies
pip install -r requirements-dev.txt

# Configure environment (use local profile for development)
export TARS_PROFILE=local

# Start Redis (required for API server)
docker run -d -p 6379:6379 redis:7-alpine

# Run the observability API server
python scripts/run_api_server.py --profile local --reload
```

**Access the API:**
- **Swagger UI:** http://localhost:8100/docs
- **Health Check:** http://localhost:8100/health
- **Metrics:** http://localhost:8100/metrics

### Docker Installation

```bash
# Build and run with Docker Compose
docker-compose -f docker-compose.yaml up -d

# Verify services
docker-compose ps
```

### CI/CD Integration

**GitHub Actions Example:**

```yaml
name: Daily Compliance Check
on:
  schedule:
    - cron: '0 8 * * *'  # Daily at 8 AM UTC

jobs:
  compliance:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Install dependencies
        run: pip install -r requirements-dev.txt
      - name: Run compliance check
        run: bash examples/compliance_check.sh
        env:
          TARS_PROFILE: prod
      - name: Upload report
        uses: actions/upload-artifact@v3
        with:
          name: compliance-report
          path: compliance_report.json
```

**Jenkins Pipeline Example:**

```groovy
pipeline {
    agent any
    triggers {
        cron('H 8 * * *')  // Daily at 8 AM
    }
    stages {
        stage('Compliance Check') {
            steps {
                sh 'pip install -r requirements-dev.txt'
                sh 'bash examples/compliance_check.sh'
            }
        }
        stage('Generate Retrospective') {
            steps {
                sh 'python scripts/generate_retrospective.py --profile prod --sign --encrypt'
            }
        }
    }
    post {
        always {
            archiveArtifacts artifacts: 'retrospective_*.md', fingerprint: true
        }
    }
}
```

---

## CLI Tools

T.A.R.S. provides **5 core observability CLI tools** for monitoring, analysis, and reporting. All tools support **enterprise mode** with encryption, signing, and compliance features.

### 1. GA KPI Collector

**Purpose:** Collect and track General Availability (GA) Key Performance Indicators for production readiness assessment.

**Basic Usage:**
```bash
# Collect GA KPIs (legacy mode, backward compatible)
python observability/ga_kpi_collector.py

# Enterprise mode with encryption and signing
python observability/ga_kpi_collector.py \
  --profile prod \
  --encrypt \
  --sign \
  --output ga_kpi_report.json.enc
```

**Output:** JSON report with 10+ GA KPIs (availability, performance, security, compliance)

### 2. Stability Monitor (7-Day)

**Purpose:** Monitor system stability over 7-day rolling windows with trend analysis.

**Basic Usage:**
```bash
# Monitor stability (legacy mode)
python observability/stability_monitor_7day.py

# Enterprise mode with compliance validation
python observability/stability_monitor_7day.py \
  --profile prod \
  --encrypt \
  --sign \
  --output stability_report.json.enc
```

**Output:** 7-day stability metrics with trend analysis and alerts

### 3. Anomaly Detector

**Purpose:** Detect anomalies in system metrics using statistical analysis and machine learning.

**Basic Usage:**
```bash
# Detect anomalies (legacy mode)
python observability/anomaly_detector_lightweight.py

# Enterprise mode with signed output
python observability/anomaly_detector_lightweight.py \
  --profile prod \
  --sign \
  --output anomalies.json.sig
```

**Output:** List of detected anomalies with severity, confidence, and recommendations

### 4. Regression Analyzer

**Purpose:** Analyze performance regressions across deployments and versions.

**Basic Usage:**
```bash
# Analyze regressions (legacy mode)
python observability/regression_analyzer.py

# Enterprise mode with encryption
python observability/regression_analyzer.py \
  --profile prod \
  --encrypt \
  --output regressions.json.enc
```

**Output:** Performance regression report with root cause analysis

### 5. Retrospective Generator

**Purpose:** Generate comprehensive retrospective reports with optional encryption and signing.

**Basic Usage:**
```bash
# Generate retrospective (legacy mode)
python scripts/generate_retrospective.py

# Enterprise mode with full SBOM and SLSA provenance
python scripts/generate_retrospective.py \
  --profile prod \
  --encrypt \
  --sign \
  --output retrospective_$(date +%Y%m%d).md.enc
```

**Output:** Markdown retrospective report with executive summary, achievements, metrics, and next steps

### Common CLI Flags

All enterprise-enabled CLI tools support the following flags:

| Flag | Description | Default |
|------|-------------|---------|
| `--profile` | Environment profile (local, dev, staging, prod) | `local` |
| `--config` | Path to custom config YAML | Auto-detected |
| `--encrypt` | Encrypt output with AES-256-GCM | `false` |
| `--sign` | Sign output with RSA-PSS | `false` |
| `--no-compliance` | Disable compliance enforcement | `false` |
| `--output` | Custom output file path | Auto-generated |

---

## Enterprise Mode

T.A.R.S. v1.0.2 introduces **enterprise-grade features** for production environments, including compliance, security, and advanced observability.

### Enterprise Configuration

**Multi-Source Configuration (Precedence: CLI > Env > File > Vault):**

```yaml
# enterprise_config/prod.yaml
app:
  name: "TARS-Production"
  environment: "prod"
  log_level: "INFO"
  log_format: "json"

compliance:
  enabled: true
  standards: ["soc2", "iso27001", "gdpr"]
  mode: "warn"  # log, warn, block

security:
  encryption:
    enabled: true
    algorithm: "AES-256-GCM"
    key_path: "/etc/tars/secrets/aes.key"
  signing:
    enabled: true
    algorithm: "RSA-PSS"
    key_path: "/etc/tars/secrets/rsa.key"

secrets:
  backend: "vault"  # vault, aws, gcp, file
  vault:
    url: "https://vault.example.com"
    token_env: "VAULT_TOKEN"
    path: "secret/tars"

telemetry:
  prometheus:
    enabled: true
    port: 9090
```

**Environment Profiles:** `local`, `dev`, `staging`, `prod`

### Encryption & Signing Examples

**1. Encrypt Sensitive Output:**

```bash
# Generate AES-256 key
python -c "import os; print(os.urandom(32).hex())" > /etc/tars/secrets/aes.key

# Encrypt GA KPI report
python observability/ga_kpi_collector.py \
  --profile prod \
  --encrypt \
  --output ga_kpi_report.json.enc

# Decrypt (using example script)
python examples/decrypt_report.py ga_kpi_report.json.enc
```

**2. Sign Output for Integrity:**

```bash
# Generate RSA-4096 signing key
openssl genrsa -out /etc/tars/secrets/rsa.key 4096
openssl rsa -in /etc/tars/secrets/rsa.key -pubout -out /etc/tars/secrets/rsa.pub

# Sign retrospective report
python scripts/generate_retrospective.py \
  --profile prod \
  --sign \
  --output retrospective.md.sig

# Verify signature (using example script)
python examples/verify_signature.py retrospective.md.sig
```

**3. Combined Encryption + Signing:**

```bash
# Generate signed and encrypted report with SBOM and SLSA provenance
python examples/generate_signed_report.py \
  --profile prod \
  --encrypt \
  --full-provenance \
  --verify

# Outputs:
# - retrospective_20251127.md.enc (encrypted report)
# - retrospective_20251127.md.sig (RSA-PSS signature)
# - sbom_cyclonedx.json (CycloneDX SBOM)
# - slsa_provenance.json (SLSA Level 3 provenance)
```

### Retrospective Generation

**Basic Retrospective:**

```bash
python scripts/generate_retrospective.py --profile prod
```

**Encrypted Retrospective:**

```bash
python scripts/generate_retrospective.py --profile prod --encrypt
```

**Signed Retrospective:**

```bash
python scripts/generate_retrospective.py --profile prod --sign
```

**Full Enterprise Retrospective (Encrypted + Signed + SBOM + SLSA):**

```bash
python scripts/generate_retrospective.py \
  --profile prod \
  --encrypt \
  --sign \
  --sbom cyclonedx \
  --slsa 3 \
  --output-dir /opt/tars/reports/
```

**Output Files:**
- `retrospective_YYYYMMDD.md.enc` - Encrypted retrospective report
- `retrospective_YYYYMMDD.md.sig` - RSA-PSS digital signature
- `sbom_cyclonedx.json` - Software Bill of Materials (CycloneDX format)
- `slsa_provenance.json` - SLSA Level 3 provenance metadata

**Verify Report Integrity:**

```bash
# Verify signature
python examples/verify_signature.py retrospective_20251127.md.sig

# Decrypt and verify
python examples/decrypt_report.py retrospective_20251127.md.enc --verify
```

### API Server Quickstart

**Start the Observability API Server:**

```bash
# Development mode (auto-reload)
python scripts/run_api_server.py --profile local --reload

# Production mode (with TLS)
python scripts/run_api_server.py --profile prod --port 8443
```

**API Authentication:**

```bash
# Login with JWT
curl -X POST http://localhost:8100/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "admin123"}'

# Response: {"access_token": "eyJ...", "token_type": "bearer"}

# Use API with JWT
curl http://localhost:8100/api/ga \
  -H "Authorization: Bearer eyJ..."

# Or use API key
curl http://localhost:8100/api/ga \
  -H "X-API-Key: tars_admin_default_key_change_in_prod"
```

**Available API Endpoints:**

| Endpoint | Method | Description | Auth Required |
|----------|--------|-------------|---------------|
| `/health` | GET | Health check | No |
| `/metrics` | GET | Prometheus metrics | No |
| `/auth/login` | POST | JWT login | No |
| `/auth/refresh` | POST | Refresh JWT | Yes |
| `/api/ga` | GET | GA KPIs | Yes |
| `/api/daily` | GET | Last 7 daily summaries | Yes |
| `/api/daily/{date}` | GET | Summary for specific date | Yes |
| `/api/anomalies` | GET | All anomalies | Yes |
| `/api/anomalies/{date}` | GET | Anomalies for date | Yes |
| `/api/regressions` | GET | All regressions | Yes |
| `/api/regressions/{date}` | GET | Regressions for date | Yes |
| `/api/retrospective` | POST | Generate retrospective | Yes (admin) |
| `/api/retrospective/download/{filename}` | GET | Download retrospective | Yes (admin) |

**See:** [docs/PHASE14_6_API_GUIDE.md](docs/PHASE14_6_API_GUIDE.md) for complete API documentation

### Compliance Framework

**Supported Standards:**
- **SOC 2 Type II:** 18 controls (access control, encryption, audit trails, retention)
- **ISO 27001:** 20 controls (information security, risk management, incident response)
- **GDPR:** Partial support (data minimization, right to erasure, PII redaction)

**Compliance Modes:**
- `log` - Log violations, continue execution
- `warn` - Warn about violations, continue execution
- `block` - Block execution on critical violations

**Example Compliance Check:**

```bash
# Run compliance check
bash examples/compliance_check.sh

# Exit code 0 = all controls pass
# Exit code 1 = at least one control failed

# Output: compliance_report.json with pass/fail status for each control
```

**Compliance Report Fields:**
- `control_id` - Unique control identifier (e.g., `SOC2-AC-001`)
- `standard` - Compliance standard (SOC2, ISO27001, GDPR)
- `status` - pass, fail, warning
- `message` - Description of finding
- `remediation` - Suggested fix

---

## Deployment

### Local Deployment

**1. Start Redis:**

```bash
docker run -d -p 6379:6379 redis:7-alpine
```

**2. Configure Environment:**

```bash
export TARS_PROFILE=local
export TARS_LOG_LEVEL=INFO
```

**3. Start API Server:**

```bash
python scripts/run_api_server.py --profile local --reload
```

**4. Run CLI Tools:**

```bash
# Collect GA KPIs
python observability/ga_kpi_collector.py --profile local

# Monitor stability
python observability/stability_monitor_7day.py --profile local

# Detect anomalies
python observability/anomaly_detector_lightweight.py --profile local

# Generate retrospective
python scripts/generate_retrospective.py --profile local
```

### Docker Deployment

**1. Build Image:**

```bash
docker build -t tars-observability:v1.0.2-rc1 .
```

**2. Run with Docker Compose:**

```bash
docker-compose -f docker-compose.yaml up -d
```

**Services Started:**
- `tars-api` - Observability API server (port 8100)
- `redis` - Redis state store (port 6379)

### Kubernetes Deployment

**1. Create Namespace and Secrets:**

```bash
kubectl create namespace tars

# Generate JWT secret
kubectl create secret generic tars-secrets \
  --from-literal=jwt-secret="$(openssl rand -base64 32)" \
  --from-literal=api-key="$(openssl rand -base64 32)" \
  -n tars

# Generate AES encryption key
kubectl create secret generic tars-encryption \
  --from-literal=aes-key="$(openssl rand -hex 32)" \
  -n tars

# Generate RSA signing key
openssl genrsa -out rsa.key 4096
kubectl create secret generic tars-signing \
  --from-file=rsa-key=rsa.key \
  -n tars
rm rsa.key
```

**2. Install with Helm:**

```bash
helm install tars ./charts/tars \
  -n tars \
  -f charts/tars/values.yaml
```

**3. Verify Deployment:**

```bash
kubectl get pods -n tars
kubectl get svc -n tars
kubectl get ingress -n tars
```

**4. Access API:**

```bash
# Port-forward for local access
kubectl port-forward -n tars svc/tars-api 8100:8100

# Or access via Ingress
# https://tars.example.com
```

### Production Deployment Checklist

**Before deploying to production:**

- [ ] Change default passwords and API keys
- [ ] Generate production AES and RSA keys
- [ ] Configure Vault or AWS Secrets Manager
- [ ] Enable TLS with valid certificates
- [ ] Configure rate limiting (Redis required)
- [ ] Set up Prometheus and Grafana
- [ ] Configure alert rules
- [ ] Test backup and restore procedures
- [ ] Review and apply security contexts
- [ ] Enable audit logging
- [ ] Configure network policies
- [ ] Test disaster recovery procedures

**See:** [docs/PHASE14_6_ENTERPRISE_HARDENING.md](docs/PHASE14_6_ENTERPRISE_HARDENING.md) for complete production hardening guide

---

## Features

### Multi-Agent Reinforcement Learning

- **Four Specialized Agents:** DQN, A2C, PPO, DDPG
- **Nash Equilibrium:** Multi-agent coordination and conflict resolution
- **Reward System:** Comprehensive reward shaping with temporal credit assignment
- **Real Training:** Actual multi-episode training for hyperparameter evaluation

### AutoML & Optimization

- **Optuna Integration:** Tree-structured Parzen Estimator (TPE) for hyperparameter search
- **MLflow Tracking:** Complete experiment tracking with model versioning
- **Hyperparameter Sync:** Hot-reload optimized parameters with <100ms latency
- **Approval Workflows:** Manual, threshold-based, and fully autonomous modes

### Enterprise Security

- **Encryption:** AES-256-GCM for data at rest
- **Signing:** RSA-PSS (4096-bit) for integrity verification
- **Authentication:** JWT (HS256) with 60-min access tokens and 7-day refresh tokens
- **Authorization:** RBAC with 3 roles (admin, sre, readonly)
- **Rate Limiting:** Redis-backed sliding window with configurable limits
- **TLS/HTTPS:** Full TLS 1.3 support with cert-manager integration
- **SBOM:** CycloneDX and SPDX software bill of materials
- **SLSA:** Level 3 provenance for supply chain security

### Compliance & Governance

- **SOC 2 Type II:** 18 controls covering access, encryption, audit trails
- **ISO 27001:** 20 controls for information security management
- **GDPR:** PII redaction, data minimization, right to erasure
- **Runtime Enforcement:** Log, warn, or block modes
- **Audit Trail:** Cryptographic chaining with tamper detection

### Observability Suite

- **GA KPI Tracking:** 10+ production readiness metrics
- **Stability Monitoring:** 7-day rolling window analysis
- **Anomaly Detection:** Statistical and ML-based anomaly detection
- **Regression Analysis:** Performance regression tracking across deployments
- **Automated Retrospectives:** Generate comprehensive reports with insights
- **Prometheus Metrics:** 7+ custom metrics for monitoring
- **Structured Logging:** JSON and text logging with log levels

### Organization Health Governance (Phase 14.7-14.8)

- **Repository Health Dashboard:** Comprehensive health scoring with issue tracking
- **Alerting Engine:** Configurable severity-based alerts with escalation rules
- **Trend Analyzer:** Time-series analysis with predictive scoring and early warnings
- **Org Health Aggregator:** Multi-repository health aggregation with SLO/SLA evaluation
- **Org Alerting Engine:** Organization-wide alerting with escalation and routing
- **Trend Correlation Engine:** Cross-repository trend pattern analysis and anomaly detection
- **Temporal Intelligence Engine:** Time-lagged correlations, influence scoring, propagation path detection, and causality heuristics
- **SLA Intelligence Engine:** Executive readiness scoring, SLA compliance tracking, breach attribution, and board-ready reporting

### Advanced RAG Pipeline

- **Cross-Encoder Reranking:** MS MARCO MiniLM model (+20-25% MRR improvement)
- **Semantic Chunking:** Dynamic chunk sizing based on content boundaries
- **Hybrid Search:** BM25 keyword + vector similarity fusion
- **Query Expansion:** LLM-based query reformulation
- **Local LLM Inference:** GPU-accelerated Ollama with Mistral 7B

---

## Documentation

### Comprehensive Guides
- [Enterprise Hardening Guide](docs/PHASE14_6_ENTERPRISE_HARDENING.md) - Complete enterprise features documentation (2,500+ LOC)
- [API Guide](docs/PHASE14_6_API_GUIDE.md) - Complete API reference with examples (1,600+ LOC)
- [Production Runbook](docs/PHASE14_6_PRODUCTION_RUNBOOK.md) - Operations guide for production deployments
- [Docker Guide](docs/PHASE14_6_DOCKER.md) - Docker and Docker Compose deployment
- [Quick Start](docs/PHASE14_6_QUICKSTART.md) - Get started in 5 minutes

### Organization Health & Analytics (Phase 14.7-14.8)
- [Repository Health Dashboard Guide](docs/REPOSITORY_HEALTH_DASHBOARD_GUIDE.md) - Health scoring and monitoring
- [Alerting Engine Guide](docs/ALERTING_ENGINE_GUIDE.md) - Alert configuration and escalation
- [Trend Analyzer Guide](docs/TREND_ANALYZER_GUIDE.md) - Time-series analysis and predictions
- [Org Health Governance Guide](docs/ORG_HEALTH_GOVERNANCE_GUIDE.md) - Multi-repo health aggregation
- [Org Alerting & Escalation Guide](docs/ORG_ALERTING_AND_ESCALATION_ENGINE.md) - Org-level alerting
- [Trend Correlation Engine Guide](docs/ORG_TREND_CORRELATION_ENGINE.md) - Cross-repo trend analysis
- [Temporal Intelligence Engine Guide](docs/ORG_TEMPORAL_INTELLIGENCE_ENGINE.md) - Time-lagged correlation and propagation analysis
- [SLA Intelligence Engine Guide](docs/ORG_SLA_INTELLIGENCE_ENGINE.md) - SLA compliance and executive readiness

### GA Release & Hardening (Phase 14.9)
- [Phase 14.9 GA Release Summary](docs/PHASE14_9_GA_RELEASE_SUMMARY.md) - GA hardening and release gate
- [GA Release Notes](RELEASE_NOTES_GA.md) - v1.0.4 GA release notes
- [MVP Progress Visualization](docs/MVP_PROGRESS_VISUALIZATION.md) - Complete project progress

### Post-GA Operations (Phase 15)
- [Operator Runbook](docs/OPS_RUNBOOK.md) - Daily and weekly operations guide
- [Incident Playbook](docs/INCIDENT_PLAYBOOK.md) - Incident response and troubleshooting
- [Post-GA Governance Policy](docs/POST_GA_GOVERNANCE.md) - Change management and release policy
- [SLA Policy Templates](policies/examples/) - Ready-to-use SLA policy examples

### Ops Automation Hardening (Phase 16)
- Cross-platform command support (Windows PowerShell + Bash parity)
- Executive bundle packaging with single deliverable
- GitHub Actions scheduled runner templates
- Enhanced pipeline orchestrator with timestamp handling

### Post-GA Observability (Phase 17)
- Run metadata and provenance artifacts for audit trails
- Executive narrative generator for plain-English summaries
- Compliance index with SOC-2/ISO-27001 control mappings
- Enhanced operator documentation with quick action tables
- 30-minute operator checklist for daily operations
- Golden incident path for SEV-1 SLA breaches

### Ops Integrations (Phase 18)
- [Configuration Guide](docs/CONFIGURATION_GUIDE.md) - Unified config file support for all tools
- Unified config file (`tars.yml`/`tars.json`) with precedence: CLI > --config > ENV > defaults
- Notification hook interface (webhook, Slack, PagerDuty stub) that never fails pipeline
- Evidence bundle GPG signing with integrity verification documentation
- Retention management helper for hot/warm/archive tier management
- 59 new smoke tests for config, notify, retention, and packager integrity

### Production Ops Maturity (Phase 19)
- [Adoption Guide](docs/ADOPTION_GUIDE.md) - Minimal rollout checklist and best practices
- Golden Path CLI (`scripts/tars_ops.py`) - Single entry-point for daily/weekly/incident operations
- Config-first GitHub Actions with exit code guidance in job summaries
- Safe `${VAR}` environment variable expansion in config files
- Examples pack with ready-to-use configs, workflows, and notification templates
- 54 new tests for env expansion and golden path CLI

### Implementation Reports
- [Phase 14.8 Task 5 Summary](docs/PHASE14_8_TASK5_COMPLETION_SUMMARY.md) - SLA Intelligence Engine
- [Phase 14.8 Task 4 Summary](docs/PHASE14_8_TASK4_COMPLETION_SUMMARY.md) - Temporal Intelligence Engine
- [Phase 14.8 Task 3 Summary](docs/PHASE14_8_TASK3_COMPLETION_SUMMARY.md) - Trend Correlation Engine
- [Phase 14.8 Task 2 Summary](docs/PHASE14_8_TASK2_COMPLETION_SUMMARY.md) - Org Alerting Engine
- [Phase 14.8 Task 1 Summary](docs/PHASE14_8_TASK1_COMPLETION_SUMMARY.md) - Org Health Governance
- [Phase 14.7 Task 10 Summary](docs/PHASE14_7_TASK10_COMPLETION_SUMMARY.md) - Trend Analyzer
- [Phase 14.6 Final Assembly](PHASE14_6_FINAL_ASSEMBLY_SUMMARY.md) - Enterprise hardening
- [Phase 11.5 Implementation Report](PHASE11_5_IMPLEMENTATION_REPORT.md) - Security & deployment
- [Phase 11 Architecture](PHASE11_ARCHITECTURE.md) - Complete system architecture

### API Documentation
- **Swagger UI:** http://localhost:8100/docs (when API server is running)
- **ReDoc:** http://localhost:8100/redoc

---

## Technology Stack

### Backend
- **Framework:** FastAPI 0.104.1 (Python 3.11+)
- **Authentication:** JWT (HS256) with python-jose
- **Encryption:** cryptography 41.0.7 (AES-256-GCM, RSA-PSS)
- **Compliance:** Custom framework (pydantic-based)
- **State Management:** Redis 7.0+ with persistence
- **RL Frameworks:** Stable-Baselines3 (DQN, A2C, PPO, DDPG)
- **AutoML:** Optuna 3.4.0 with TPE sampler
- **Experiment Tracking:** MLflow 2.8.0

### Infrastructure
- **Orchestration:** Kubernetes 1.24+ with Helm 3.0+
- **Containerization:** Docker 20.10+ & Docker Compose 2.0+
- **Monitoring:** Prometheus + Grafana
- **Secrets Management:** Vault, AWS Secrets Manager, GCP Secret Manager
- **Certificate Management:** cert-manager 1.13+

---

## Performance Metrics

### Security Performance
| Metric | Value | Details |
|--------|-------|---------|
| **JWT Generation** | 2-3ms | HS256 algorithm |
| **JWT Validation** | <5ms | >10,000 req/s throughput |
| **AES Encryption** | <10ms | 1MB payload |
| **RSA Signing** | <20ms | 4096-bit key |
| **Rate Limit Overhead** | <1ms | Redis-backed |

### Observability Performance
| Metric | Value | Details |
|--------|-------|---------|
| **GA KPI Collection** | <2s | 10+ metrics |
| **Anomaly Detection** | <5s | Statistical + ML |
| **Regression Analysis** | <10s | Multi-version comparison |
| **Retrospective Generation** | <30s | Full report with signing |

### Multi-Agent RL Performance
| Metric | Value | Details |
|--------|-------|---------|
| **Agent Training Time** | 2-5s (quick) / 10-20s (full) | 10/50 episodes per trial |
| **Hot-Reload Latency** | 50-100ms | Hyperparameter update |
| **Hyperparameter Improvement** | +5-12pp | Reward improvement |

---

## Project Statistics

### Code Metrics
- **Total Lines of Code:** ~202,050+ lines (cumulative across all phases)
- **Organization Health & Analytics:** ~26,000+ lines (Phase 14.7-14.9)
- **Post-GA Operations:** ~3,500+ lines (Phase 15)
- **Ops Automation Hardening:** ~1,500+ lines (Phase 16)
- **Post-GA Observability:** ~2,500+ lines (Phase 17)
- **Ops Integrations:** ~2,750+ lines (Phase 18)
- **Production Ops Maturity:** ~1,800+ lines (Phase 19)
- **Core Observability:** ~12,000+ lines (Phase 14.6)
- **Multi-Agent RL:** ~22,910 lines (Phase 11)
- **RAG Foundation:** ~9,920 lines (Phases 1-5)
- **Federation:** ~8,500 lines (Phases 6-9)

### System Composition
- **Core Services:** 9 production services
- **Observability Tools:** 5 CLI tools + 1 API server + 1 pipeline orchestrator + 1 bundle packager + 2 narrative/metadata generators
- **Ops Integration Tools:** 4 scripts (config loader, notify hook, retention manager, golden path CLI)
- **Analytics Engines:** 8 analytics modules (dashboard, alerting, trends, org-health, org-alerting, correlation, temporal-intelligence, sla-intelligence)
- **Release Tools:** 4 scripts (GA validator, readiness checker, GA packager, executive bundle packager)
- **Operations Docs:** 4 runbooks (operator, incident, governance, configuration)
- **Policy Templates:** 6 SLA policy templates
- **CI/CD Workflows:** 2 GitHub Actions workflows (daily, weekly)
- **API Endpoints:** 12 REST endpoints (observability API)
- **Test Coverage:** 543+ test cases across 23+ test suites
- **Example Templates:** 8 examples (configs, workflows, notifications, retention)

---

## Contributing

This project follows the VDS RiPIT Agent Coding Workflow v2.9 conventions. See [Reference Docs](Reference%20Docs/) for development guidelines.

### Development Setup
```bash
# Install dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/ -v --cov

# Format code
black . && isort .
```

---

## Security

**T.A.R.S. is designed with security and privacy as core principles:**

- **Data Sovereignty:** All data remains local - no cloud dependencies
- **Encryption:** AES-256-GCM for data at rest, TLS 1.3 for data in transit
- **Authentication:** JWT-based authentication with short-lived tokens
- **Privacy:** No telemetry or usage tracking - all data stored locally
- **Isolation:** Docker/Kubernetes containerization for service isolation
- **Dependencies:** Regular security audits and dependency scanning
- **Supply Chain:** SBOM and SLSA provenance for all releases

---

## License

[License information to be added]

---

## Contact

**Project maintained by Veleron Dev Studios**

**Repository:** https://github.com/oceanrockr/VDS_TARS

---

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history.

See [RELEASE_NOTES_GA.md](RELEASE_NOTES_GA.md) for the v1.0.4 GA release notes.

---

**Last Updated:** December 26, 2025
**Version:** v1.0.9 (General Availability)
**Status:** GA Release Complete + Drop-In Operable - MVP 100%

### Development Phases Completed
- **Phases 1-5:** RAG Foundation and Advanced Retrieval
- **Phases 6-9:** Multi-Region Federation
- **Phase 10:** Cognitive Analytics
- **Phase 11:** Multi-Agent RL System (11.1-11.5)
- **Phase 12:** QA Suite and Testing
- **Phase 13:** Eval Engine and Migration
- **Phase 14.1-14.5:** Core Infrastructure
- **Phase 14.6:** Enterprise Hardening (SOC 2, ISO 27001, GDPR)
- **Phase 14.7:** Repository Health Analytics (Tasks 1-10)
- **Phase 14.8:** Organization Health Governance (Tasks 1-5)
  - Task 1: Org Health Aggregator
  - Task 2: Org Alerting & Escalation Engine
  - Task 3: Multi-Repository Trend Correlation Engine
  - Task 4: Advanced Correlation & Temporal Intelligence Engine
  - Task 5: SLA Reporting & Executive Readiness Dashboard
- **Phase 14.9:** GA Hardening & Production Release Gate
  - Task 1: GA Release Validator Engine
  - Task 2: Production Readiness Checklist Generator
  - Task 3: GA Release Artifact Packager
  - Task 4: Version Finalization
  - Task 5: Documentation
- **Phase 15:** Post-GA Operations Enablement
  - Task 1: Operator Runbook
  - Task 2: Incident Response & Troubleshooting Playbook
  - Task 3: SLA Policy Template Pack (6 templates)
  - Task 4: Full Pipeline Orchestrator Script
  - Task 5: Post-GA Governance Policy
- **Phase 16:** Ops Automation Hardening + Executive Bundle Packaging
  - Task 1: Command Portability & Timestamp Handling
  - Task 2: Executive Bundle Packager Script
  - Task 3: Cross-Platform Docs Update
  - Task 4: GitHub Actions Workflow Templates
  - Task 5: Smoke Tests for Orchestrator + Packager
- **Phase 17:** Post-GA Observability, Compliance Evidence & Operator UX
  - Task 1: Run Metadata & Provenance Generator
  - Task 2: Executive Narrative Generator
  - Task 3: Compliance Index for Executive Bundles
  - Task 4: Operator Documentation UX Improvements
  - Task 5: Smoke Tests for Metadata + Narrative
- **Phase 18:** Ops Integrations, Config Management & Evidence Security
  - Task 1: Unified Config File Support (tars.yml/tars.json)
  - Task 2: Notification Hook Interface (Webhook, Slack, PagerDuty)
  - Task 3: Evidence Bundle Security Hardening (GPG signing)
  - Task 4: Retention Helper Script (hot/warm/archive tiers)
  - Task 5: Tests, Docs & Release Hygiene
- **Phase 19:** Production Ops Maturity & CI Hardening
  - Task 1: GitHub Actions Config-First Execution
  - Task 2: Environment Variable Expansion in Config
  - Task 3: Golden Path Wrapper Script (tars_ops.py)
  - Task 4: Examples Pack for Real-World Adoption
  - Task 5: Versioning, Documentation, and Verification
