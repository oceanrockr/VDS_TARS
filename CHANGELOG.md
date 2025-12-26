# Changelog

All notable changes to this project will be documented in this file.

The format follows the guidelines of
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added

### Changed

### Deprecated

### Removed

### Fixed

### Security

---

## [1.0.9] - 2025-12-26

### Added

- **Phase 19: Production Ops Maturity & CI Hardening**
  - GitHub Actions Config-First Execution (`.github/config/tars.ci.yml`)
    - Centralized CI configuration file with no secrets
    - Updated `tars_daily_ops.yml` and `tars_weekly_ops.yml` workflows
    - Job summaries with exit code guidance and operator recommendations
    - Artifact retention policies (30 days daily, 90 days weekly)
    - Opt-in notifications via environment variables
  - Environment Variable Expansion in Config (`scripts/tars_config.py` v1.1)
    - Safe `${VAR_NAME}` expansion from `os.environ`
    - Missing variables preserved as literals with warning
    - No recursive expansion or shell execution (security)
    - Applied to all string values in config
  - Golden Path Wrapper Script (`scripts/tars_ops.py`, ~400 LOC)
    - Single entry-point for common operations
    - `daily` command: Quick health check (flat output)
    - `weekly` command: Trend analysis + executive bundle + retention summary
    - `incident` command: Full output + narrative + optional signing
    - Exit code guidance printed after each run
    - Respects config precedence (CLI > --config > TARS_CONFIG env > defaults)
  - Examples Pack for Real-World Adoption (`examples/`)
    - `examples/configs/tars.dev.yml` - Development configuration
    - `examples/configs/tars.ci.yml` - CI/CD configuration template
    - `examples/configs/tars.incident.yml` - Incident response configuration
    - `examples/github-actions/minimal-workflow.yml` - Minimal workflow template
    - `examples/notifications/webhook-payload.json` - Sample webhook payload
    - `examples/notifications/slack-message.json` - Sample Slack message
    - `examples/retention/sample-config.yml` - Retention configuration
    - `examples/retention/dry-run-output.txt` - Sample retention output
  - Adoption Guide (`docs/ADOPTION_GUIDE.md`, ~400 LOC)
    - Minimal rollout checklist (5 phases)
    - Secrets management best practices
    - GPG signing posture recommendations
    - Retention tier configurations
    - Common deployment patterns
    - Troubleshooting guide
  - Unit Tests for Config Env Expansion (`tests/unit/test_config_env_expansion.py`, ~350 LOC)
    - 27 test cases covering pattern matching, expansion, and edge cases
  - Smoke Tests for Golden Path CLI (`tests/integration/test_tars_ops_smoke.py`, ~300 LOC)
    - 27 test cases for command parsing, guidance, and building

### Changed

- **Operator Runbook Updates** (`docs/OPS_RUNBOOK.md` v1.0.9)
  - Added Golden Path CLI section with `tars_ops.py` examples
  - Updated to Phase 19
- **Config Loader Updates** (`scripts/tars_config.py` v1.1)
  - Added `expand_env_vars_in_config()` function
  - Added `expand_env_vars_in_string()` function
  - Added `expand_env` parameter to `load()` method
  - Updated docstrings for environment variable expansion

### Documentation

- Added Adoption Guide (`docs/ADOPTION_GUIDE.md`)
- Updated Operator Runbook with Golden Path CLI
- Added examples directory with templates

---

## [1.0.8] - 2025-12-25

### Added

- **Phase 18: Ops Integrations, Config Management, and Evidence Security**
  - Unified Config File Support (`scripts/tars_config.py`, ~350 LOC)
    - Single-file configuration for all T.A.R.S. governance tools
    - YAML (PyYAML) and JSON format support
    - Config precedence: CLI > --config > TARS_CONFIG env > ./tars.yml > defaults
    - Deep merge with built-in defaults
    - Validation for format and retention values
    - Namespaced sections: orchestrator, packager, retention, notify
  - Notification Hook Interface (`scripts/notify_ops.py`, ~400 LOC)
    - Opt-in webhook, Slack, and PagerDuty (stub) notifications
    - Exit code to severity/action/title auto-mapping
    - Never fails pipeline (always returns 0)
    - Dry-run mode for payload preview
    - Run metadata extraction from run directories
    - Slack message formatting with color-coded severity
  - Evidence Bundle Security Hardening (`scripts/package_executive_bundle.py` v1.2)
    - GPG signing support (`--sign`, `--gpg-key-id` flags)
    - Integrity verification documentation (`{bundle}-integrity.md`)
    - SHA-256 checksums with cross-platform verification instructions
    - Graceful fallback when GPG is unavailable
    - Detached signature (.sig) file generation
  - Retention Helper Script (`scripts/retention_manage.py`, ~400 LOC)
    - Hot/warm/archive tier management
    - Platform-aware compression (tar.gz on Linux/macOS, zip on Windows)
    - Dry-run mode as default (safe operation)
    - Directory discovery by timestamp pattern
    - Statistics tracking (scanned/compressed/moved/deleted)
    - Force flag for actual modifications
  - Configuration Guide (`docs/CONFIGURATION_GUIDE.md`, ~400 LOC)
    - Complete configuration reference
    - Examples for minimal, CI/CD, and incident response configs
    - Multi-environment setup guide (dev/staging/prod)
    - Environment variable support documentation
    - Troubleshooting guide
  - Smoke Tests (`tests/integration/`, ~1,100 LOC)
    - `test_config_loader_smoke.py` (~300 LOC, 18 tests)
    - `test_notify_ops_smoke.py` (~250 LOC, 15 tests)
    - `test_retention_manage_smoke.py` (~300 LOC, 12 tests)
    - `test_packager_integrity_smoke.py` (~250 LOC, 14 tests)

### Changed

- **Pipeline Orchestrator Enhancements** (`scripts/run_full_org_governance_pipeline.py` v2.2)
  - Integrated config file support with `--config` flag
  - Added `--notify-on-exit-codes` for selective notification
  - Added `--notify-webhook-url` for webhook integration
  - Config values merged with CLI precedence
  - Notification hook invocation after pipeline completion
- **Executive Bundle Packager** (`scripts/package_executive_bundle.py` v1.2)
  - Added `--config` flag for config file support
  - Added `--sign` and `--gpg-key-id` flags for GPG signing
  - Added `check_gpg_available()` function for graceful detection
  - Added `generate_bundle_integrity_doc()` method
  - Generates integrity verification documentation

### Documentation

- Added Configuration Guide (`docs/CONFIGURATION_GUIDE.md`)
- Updated README.md with Phase 18 features
- Updated MVP Progress Visualization for Phase 18 completion

---

## [1.0.7] - 2025-12-24

### Added

- **Phase 17: Post-GA Observability, Compliance Evidence & Operator UX**
  - Run Metadata & Provenance Generator (`scripts/generate_run_metadata.py`, ~400 LOC)
    - Machine-readable run provenance artifact (run-metadata.json)
    - Captures T.A.R.S. version, git commit, host OS, Python version
    - CLI flags recording for audit trail
    - Exit codes extraction from bundle manifest
    - Duration per stage tracking (best-effort)
    - Artifact listing with file sizes and timestamps
    - Provenance attestation structure
  - Executive Narrative Generator (`scripts/generate_executive_narrative.py`, ~500 LOC)
    - Plain-English executive summary for leadership review
    - Health tier determination (GREEN/AMBER/RED)
    - SLA status summary with compliance metrics
    - Key risks identification from multiple report sources
    - Notable trends and propagation signal analysis
    - Recommended next actions based on findings
    - Graceful handling of missing/partial reports
  - Compliance Index for Executive Bundles (`scripts/package_executive_bundle.py` v1.1)
    - `compliance-index.md` artifact with audit mappings
    - SOC-2 Type II control mapping (CC6.1, CC7.1, CC7.2, CC7.3)
    - ISO 27001 control mapping (A.12.1, A.12.4, A.12.6, A.16.1, A.18.2)
    - Incident response evidence markers
    - SHA-256 hash references for each artifact
    - Informational disclaimer for auditors
  - Smoke Tests (`tests/integration/`)
    - `test_run_metadata_smoke.py` (~350 LOC) - 14 test cases
    - `test_executive_narrative_smoke.py` (~400 LOC) - 16 test cases
    - Coverage for missing data, malformed inputs, graceful degradation

### Changed

- **Pipeline Orchestrator Enhancements** (`scripts/run_full_org_governance_pipeline.py` v2.1)
  - Integrated run metadata generation (auto-generates run-metadata.json)
  - Added `--with-narrative` flag for optional executive narrative generation
  - CLI flags captured for provenance tracking
  - Updated manifest version to 2.1, phase to 17
- **Executive Bundle Packager** (`scripts/package_executive_bundle.py` v1.1)
  - Added `--compliance-index` / `--no-compliance-index` flags
  - Compliance index generation enabled by default
- **Documentation Updates**
  - `docs/OPS_RUNBOOK.md` v1.0.7
    - Added "If You See This Exit Code, Do This" quick action table
    - Added "30-Minute Operator Checklist" for daily operations
    - Decision flow diagram for exit code handling
  - `docs/INCIDENT_PLAYBOOK.md` v1.0.7
    - Added "Golden Incident Path (SEV-1 SLA Breach)" section
    - 15-minute containment procedure with fill-in-the-blank templates
    - Root cause to containment action mapping table

### Documentation

- Updated README.md with Phase 17 features
- Updated MVP Progress Visualization for Phase 17 completion

---

## [1.0.6] - 2025-12-22

### Added

- **Phase 16: Ops Automation Hardening + Executive Bundle Packaging**
  - Enhanced Pipeline Orchestrator (`scripts/run_full_org_governance_pipeline.py`, upgraded to v2.0)
    - Cross-platform timestamp handling (no shell dependencies)
    - `--timestamp` flag for deterministic output directory naming
    - `--format` flag: `flat` or `structured` (daily/weekly/executive subdirs)
    - `--print-paths` flag for CI/CD artifact path logging
    - `--outdir` creates timestamped subdirectories: `tars-run-<timestamp>/`
    - Robust path handling via `pathlib`
    - Git commit hash and version in bundle manifest
  - Executive Bundle Packager (`scripts/package_executive_bundle.py`, ~400 LOC)
    - Single archive creation (ZIP, optional tar.gz)
    - SHA-256 checksums for all files and archives
    - Manifest with version, git commit, timestamps, exit codes
    - `--run-dir`, `--output-dir`, `--bundle-name` options
    - `--no-checksums`, `--no-manifest` for minimal bundles
  - GitHub Actions Workflows (`.github/workflows/`)
    - `tars_daily_ops.yml` - Daily health check (08:00 UTC)
    - `tars_weekly_ops.yml` - Weekly trend report (Monday 10:00 UTC)
    - Manual trigger with configurable options
    - Artifact upload and summary generation
  - Smoke Tests (`tests/integration/`)
    - `test_full_pipeline_orchestrator_smoke.py` (~300 LOC)
    - `test_executive_bundle_packager_smoke.py` (~350 LOC)
    - Dry-run validation, timestamp format, exit codes
    - Synthetic data packaging, ZIP/manifest verification

### Changed

- **Documentation Updates**
  - `docs/OPS_RUNBOOK.md` updated to v1.0.6 with cross-platform commands
    - Added Platform Quick Start section (Bash, PowerShell, CI)
    - Replaced `$(date ...)` examples with orchestrator timestamp handling
    - Added CI/CD Integration section with workflow references
  - `docs/INCIDENT_PLAYBOOK.md` updated to v1.0.6
    - Added cross-platform evidence collection examples
    - Integrated orchestrator and packager commands

- **Version Updates**
  - VERSION file updated to 1.0.6
  - README.md updated with Phase 16 features
  - MVP Progress Visualization updated to reflect Phase 16 completion

---

## [1.0.5] - 2025-12-22

### Added

- **Phase 15: Post-GA Operations Enablement**
  - Operator Runbook (`docs/OPS_RUNBOOK.md`, ~400 LOC)
    - Daily and weekly operations procedures
    - Exit code reference table (90-199)
    - Golden path commands for operators
    - Artifact storage recommendations
    - Troubleshooting quick reference
  - Incident Response & Troubleshooting Playbook (`docs/INCIDENT_PLAYBOOK.md`, ~500 LOC)
    - Incident classification and severity levels
    - Decision tree for incident response
    - Triage procedures by incident type
    - Containment actions (operational, not code changes)
    - Evidence collection procedures
    - Escalation matrix and templates
    - Post-incident review process
  - SLA Policy Template Pack (`policies/examples/`, 6 templates)
    - `availability_default.yaml` - Standard availability SLA (99.5% uptime)
    - `incident_response_default.yaml` - Incident response time targets
    - `reliability_default.yaml` - Error rates and latency targets
    - `dora_metrics_default.yaml` - DORA metrics (deployment freq, lead time, MTTR, CFR)
    - `internal_platform_strict.yaml` - Strict thresholds for critical infrastructure
    - `startup_lenient.yaml` - Lenient thresholds for MVPs and beta services
    - `README.md` - Usage guide for policy templates
  - Full Pipeline Orchestrator Script (`scripts/run_full_org_governance_pipeline.py`, ~550 LOC)
    - Runs complete org health governance pipeline
    - Executes: org-health -> org-alerts -> trend-correlation -> temporal-intelligence -> sla-intelligence
    - Generates executive bundle with all reports
    - Supports `--dry-run` mode for command preview
    - CI/CD exit codes for automation
    - Graceful handling of optional modules
  - Post-GA Governance Policy (`docs/POST_GA_GOVERNANCE.md`, ~350 LOC)
    - Codebase classification (frozen, allowed, monitored areas)
    - Change proposal process (docs-only, scripts-only, core changes)
    - Versioning rules (patch, minor, major)
    - Pre-merge checklist requirements
    - Release process and emergency procedures

### Changed

- **Version Updates**
  - VERSION file updated to 1.0.5
  - README.md updated with Phase 15 documentation
  - MVP Progress Visualization updated to include Phase 15

- **Documentation Structure**
  - Added Post-GA Operations section to README
  - Added operator enablement to feature list
  - Updated project statistics with Phase 15 metrics

---

## [1.0.4] - 2025-12-21

### Added

- **Phase 14.9: GA Hardening & Production Release Gate**
  - GA Release Validator Engine (`scripts/ga_release_validator.py`, ~600 LOC)
    - Comprehensive pre-release validation with 8 check categories
    - Exit codes 150 (GA READY), 151 (GA BLOCKED), 152 (GA FAILED), 199 (ERROR)
    - Validates VERSION files, documentation, analytics modules, CLI tools
    - Checks for placeholder markers, security issues, test coverage
    - JSON and human-readable output formats
  - Production Readiness Checklist Generator (`scripts/generate_production_readiness_checklist.py`, ~700 LOC)
    - Machine-readable JSON and human-readable Markdown outputs
    - 8 category evaluation: code quality, test coverage, docs, CI/CD, operational safety, backward compatibility, security, performance
    - Automated scoring and readiness assessment
  - GA Release Artifact Packager (`scripts/prepare_ga_release.py`, ~600 LOC)
    - Enforces GA validation before packaging
    - Generates checksums (SHA-256) for all artifacts
    - Creates manifest.json with full release metadata
    - Produces tar.gz and zip archives
    - Optional GPG signing support
    - GA-stamped release notes generation

- **Phase 14.8: Organization Health Governance** (Complete)
  - Task 5: SLA Reporting & Executive Readiness Dashboard Engine (`analytics/org_sla_intelligence.py`, ~1,700 LOC)
    - Executive readiness scoring (0-100) with tier classification (GREEN/YELLOW/RED)
    - SLA compliance evaluation across multiple time windows (7, 30, 90)
    - Breach attribution with root cause mapping and confidence scores
    - SLA scorecards with plain-english status descriptions
    - Risk narrative generation for board-ready reporting
    - Exit codes 140-144, 199 for CI/CD integration
  - Task 4: Temporal Intelligence Engine (`analytics/org_temporal_intelligence.py`, ~1,600 LOC)
    - Time-lagged correlation analysis at multiple offsets (-3 to +3)
    - Leader/follower relationship identification
    - Directional influence scoring (0-100)
    - Propagation path detection via DFS traversal
    - Causality heuristics with temporal precedence
    - Four temporal anomaly types (rapid propagation, leader deterioration, systemic propagation, synchronized lag)
    - Exit codes 130-134, 199 for CI/CD integration
  - Task 3: Multi-Repository Trend Correlation Engine (`analytics/org_trend_correlation.py`, ~1,500 LOC)
    - Cross-repository Pearson/Spearman correlation analysis
    - Cluster detection for correlated repositories
    - Leading indicator pattern identification
    - Rule-based anomaly detection
    - Exit codes 120-124, 199 for CI/CD integration
  - Task 2: Org Alerting & Escalation Engine (`analytics/org_alerting_engine.py`, ~1,200 LOC)
    - Organization-wide alerting with routing rules
    - Escalation policies with multi-tier support
    - Alert aggregation and deduplication
    - Exit codes 110-114, 199 for CI/CD integration
  - Task 1: Org Health Aggregator (`analytics/org_health_aggregator.py`, ~1,100 LOC)
    - Multi-repository health aggregation
    - SLO/SLA policy evaluation
    - Organization-level health scoring
    - Exit codes 100-104, 199 for CI/CD integration

### Changed

- **Version Finalization**
  - VERSION file updated to 1.0.4 (GA, no RC suffix)
  - All RC language removed from documentation
  - Production-ready status confirmed

- **Documentation Updates**
  - README.md updated with Phase 14.8-14.9 features
  - MVP Progress Visualization updated to 100% completion
  - Added Phase 14.9 GA Release Summary documentation

### Fixed

- **Validation Improvements**
  - Enhanced exit code validation across all analytics modules
  - Improved error messages for CI/CD integration failures
  - Fixed placeholder detection patterns for edge cases

### Security

- **Release Security**
  - SHA-256 checksums for all release artifacts
  - Optional GPG signing for release packages
  - Manifest with cryptographic verification support

---

## [1.0.3-RC1] - 2025-01-08

### Added

- **Phase 14.8 Task 4: Temporal Intelligence Engine**
  - Time-lagged correlation analysis
  - Influence scoring and propagation path detection
  - Causality heuristics

---

## [1.0.2-RC1] - 2025-11-27

### Added

- **Enterprise Configuration System**
  - Multi-source configuration loading with precedence: CLI > Environment > File > Vault
  - Support for 4 environment profiles: `local`, `dev`, `staging`, `prod`
  - Secrets backend integration: HashiCorp Vault, AWS Secrets Manager, GCP Secret Manager, file-based
  - Deep merge configuration with environment variable interpolation (`${VAR}` syntax)
  - Pydantic-based schema validation for all configuration types

- **Compliance Framework**
  - SOC 2 Type II compliance engine with 18 controls (access control, encryption, audit trails, data retention)
  - ISO 27001 compliance engine with 20 controls (information security, risk management, incident response)
  - GDPR-lite features: PII redaction (email, IP, SSN), data minimization, right to erasure
  - Runtime enforcement modes: `log`, `warn`, `block`
  - Compliance scoring and summary reporting
  - Cryptographic audit trail chaining with tamper detection

- **Enterprise Security Framework**
  - AES-256-GCM encryption for all output artifacts with authenticated encryption
  - RSA-PSS (4096-bit) signing for integrity verification with SHA-256
  - PGP encryption support via python-gnupg (optional, requires GPG binary)
  - SBOM generation in CycloneDX 1.4+ and SPDX 2.3+ formats
  - SLSA Level 3 provenance generation with build attestation
  - Security manager module (`security/security_manager.py`, 850+ LOC)

- **Enterprise API Server**
  - FastAPI-based observability API server (`enterprise_api/main.py`, 1,200+ LOC)
  - 12 REST endpoints: health, metrics, auth (login/refresh), GA KPIs, daily summaries, anomalies, regressions, retrospective generation
  - JWT authentication (HS256) with 60-minute access tokens and 7-day refresh tokens
  - API key authentication with SHA-256 hashing
  - RBAC with 3 roles: `admin`, `sre`, `readonly`
  - Redis-backed rate limiting (30 req/min public, 100 req/min admin)
  - Prometheus metrics endpoint
  - API server launcher script (`scripts/run_api_server.py`, 280 LOC)

- **Enterprise-Enabled Observability Modules**
  - `observability/ga_kpi_collector.py` - Enterprise config, encryption, signing, compliance
  - `observability/stability_monitor_7day.py` - Enterprise config, encryption, signing, compliance
  - `observability/anomaly_detector_lightweight.py` - Enterprise config, encryption, signing, compliance
  - `observability/regression_analyzer.py` - Enterprise config, encryption, signing, compliance
  - `scripts/generate_retrospective.py` - Enterprise config, encryption, signing, SBOM, SLSA provenance
  - All modules support `--profile`, `--config`, `--encrypt`, `--sign`, `--no-compliance` flags
  - 100% backward compatibility with legacy CLI flags

- **Retrospective Generator v2**
  - Markdown and JSON output formats
  - Optional encryption (AES-256-GCM) and signing (RSA-PSS)
  - SBOM generation (CycloneDX, SPDX)
  - SLSA Level 3 provenance generation
  - Executive summary with key achievements and metrics
  - Complete example implementation (`examples/generate_signed_report.py`, 400 LOC)

- **Production-Grade CI/CD**
  - GitHub Actions workflow (`.github/workflows/release_phase14_6.yml`)
  - Multi-Python version testing (3.9, 3.10, 3.11)
  - Linting: flake8, pylint, black, isort
  - Type checking: mypy
  - Test suite execution with pytest
  - Wheel build and distribution
  - Multi-architecture Docker build (linux/amd64, linux/arm64)
  - PyPI trusted publishing support
  - Automated release artifact generation

- **Docker Compose Profiles**
  - `ga-day` - Run GA KPI collector
  - `daily` - Run daily health reporter
  - `monitoring` - Run anomaly detector
  - `day7` - Run 7-day stability monitor
  - `api` - Run enterprise API server
  - Multi-service orchestration with Redis backend

- **Kubernetes Templates**
  - CronJob templates for scheduled observability tasks
  - Deployment templates for API server
  - PersistentVolumeClaim templates for data storage
  - ConfigMap templates for enterprise configuration
  - Secret templates for encryption/signing keys
  - Service and Ingress templates for API exposure

- **Comprehensive Documentation**
  - Enterprise Hardening Guide (`docs/PHASE14_6_ENTERPRISE_HARDENING.md`, 2,500+ LOC)
  - Enterprise API Guide (`docs/PHASE14_6_API_GUIDE.md`, 1,600+ LOC)
  - Production Runbook (`docs/PHASE14_6_PRODUCTION_RUNBOOK.md`, 6,000+ words)
  - Docker Deployment Guide (`docs/PHASE14_6_DOCKER.md`, 4,000+ words)
  - Quick Start Guide (`docs/PHASE14_6_QUICKSTART.md`)
  - Production-grade README.md with 5 major sections

- **Internal Adoption Toolkit**
  - Onboarding scripts for new users (`internal_adoption/onboarding/`)
  - Example workflows for common tasks (`internal_adoption/examples/`)
  - Training materials and best practices
  - Integration examples (Slack, GitHub Actions, Grafana)

- **Test Scaffolding**
  - Enterprise config test suite (`tests/test_enterprise_config.py`, 400 LOC, 80%+ coverage target)
  - Compliance test suite (`tests/test_compliance.py`, 500 LOC, 70%+ coverage target)
  - Security test suite (`tests/test_security.py`, 400 LOC, 85%+ coverage target)
  - API test suite (`tests/test_api.py`, 600 LOC, 90%+ coverage target)
  - Total test LOC: 1,900+ lines

- **Dependencies**
  - `fastapi==0.104.1` - Enterprise API framework
  - `uvicorn[standard]==0.24.0` - ASGI server with auto-reload
  - `python-multipart==0.0.6` - Form data parsing
  - `slowapi==0.1.9` - Rate limiting for FastAPI
  - `python-jose[cryptography]==3.3.0` - JWT token generation/validation
  - `passlib[bcrypt]==1.7.4` - Password hashing
  - `cryptography==41.0.7` - AES encryption, RSA signing
  - `python-gnupg==0.5.2` - PGP encryption (optional)

### Changed

- **Observability Module Rewrites**
  - All 5 observability modules rewritten with enterprise hooks (config loading, encryption, signing, compliance enforcement)
  - Standardized CLI argument parsing across all modules
  - Unified error handling and logging patterns
  - Consistent output artifact structure with `.enc` (encrypted) and `.sig` (signature) file support
  - Telemetry wrappers for Prometheus metrics integration

- **Compliance Integration**
  - Compliance enforcer integrated into retrospective generation pipeline
  - Compliance enforcer integrated into regression analysis pipeline
  - Input sanitization (PII redaction) applied to all observability outputs
  - Data retention policies enforced at runtime

- **CLI Parsers**
  - Updated all CLI parsers with enterprise flags:
    - `--profile` - Environment profile (local, dev, staging, prod)
    - `--config` - Path to custom configuration file
    - `--encrypt` - Encrypt output with AES-256-GCM
    - `--sign` - Sign output with RSA-PSS
    - `--no-compliance` - Disable compliance enforcement
  - Backward compatibility maintained for legacy flags (`--api-url`, `--output`, `--test-mode`)

- **Output Artifact Structure**
  - JSON schemas updated for GA KPIs, daily summaries, regressions, and retrospectives
  - Support for encrypted artifacts (`.json.enc`, `.md.enc`)
  - Support for signature files (`.sig`)
  - Support for SBOM files (`sbom_cyclonedx.json`, `sbom_spdx.json`)
  - Support for SLSA provenance files (`slsa_provenance.json`)

- **Documentation Overhaul**
  - README.md completely rewritten for v1.0.2 with enterprise features
  - All Phase 14.6 implementation reports updated
  - API documentation expanded with authentication, RBAC, and rate limiting details
  - Production deployment guides added for Docker and Kubernetes
  - Security best practices documentation added

- **Configuration Management**
  - Migrated from flat environment variables to structured YAML configuration
  - Added configuration validation with detailed error messages
  - Added configuration precedence documentation
  - Added example configuration files for all profiles

### Deprecated

- None (v1.0.2-RC1 introduces enterprise features without breaking existing APIs or deprecating legacy functionality)

### Removed

- None (RC1 maintains full backward compatibility with v1.0.1)

### Fixed

- **GA KPI Collector**
  - Fixed baseline parsing edge cases for missing fields in GA Day data
  - Fixed NaN handling in metric calculations
  - Fixed JSON serialization for infinity and NaN values

- **Stability Monitor**
  - Fixed drift detection handling for edge cases (zero variance, missing data points)
  - Fixed 7-day window alignment issues
  - Fixed memory leak in long-running stability analysis

- **Retrospective Generator**
  - Fixed chronological ordering of events in retrospective reports
  - Fixed sorting edge cases when multiple events have identical timestamps
  - Fixed Markdown table formatting for complex metric data

- **Regression Analyzer**
  - Fixed error handling in baseline loader when data is missing
  - Fixed regression detection false positives for sub-1% changes
  - Fixed performance metric comparison for multi-version analysis

- **CLI Tools**
  - Fixed test-mode inconsistencies across all CLI tools
  - Fixed Windows emoji rendering in Markdown outputs (fallback to ASCII)
  - Fixed path handling on Windows PowerShell (forward slash normalization)

- **API Server**
  - Fixed CORS headers for cross-origin requests
  - Fixed rate limiting counter reset logic
  - Fixed JWT token expiration edge cases
  - Fixed error response formatting for 422 validation errors

### Security

- **Encryption**
  - Adopted AES-256-GCM as the default encryption algorithm (replacing AES-256-CBC)
  - Implemented authenticated encryption with associated data (AEAD)
  - Added IV (initialization vector) randomization for each encryption operation
  - Added encryption key rotation support via configuration

- **Signing**
  - Adopted RSA-PSS signatures with SHA-256 (replacing PKCS#1 v1.5)
  - Increased minimum RSA key size to 4096 bits
  - Added signature verification in all output artifact workflows
  - Added signature timestamp validation

- **Credential Management**
  - Added automatic credential and secret detection in outputs
  - Added safeguards against committing `.env` files and secret keys
  - Added secret redaction in logs and error messages
  - Added Vault/AWS Secrets Manager integration for production deployments

- **API Security**
  - Added TLS certificate validation in API server (configurable)
  - Enabled rate limiting by default for all API endpoints
  - Added RBAC enforcement for all protected endpoints
  - Added API key SHA-256 hashing (no plaintext storage)
  - Added JWT token blacklist support for logout

- **Compliance**
  - Added compliance enforcement runtime modes (`log`, `warn`, `block`)
  - Added PII redaction in all observability outputs
  - Added cryptographic audit trail chaining
  - Added tamper detection for audit logs
  - Added data retention policy enforcement

---

## [1.0.1] - 2025-10-15

### Added

- **Core Observability Tools**
  - GA Day KPI Collector v1 (`observability/ga_kpi_collector.py`)
  - 7-Day Stability Monitor v1 (`observability/stability_monitor_7day.py`)
  - Anomaly Detector with EWMA and Z-score hybrid detection (`observability/anomaly_detector_lightweight.py`)
  - Regression Analyzer v1 with automatic rollback detection (`observability/regression_analyzer.py`)
  - Retrospective Generator v1 with Markdown output (`scripts/generate_retrospective.py`)

- **Structured Output Formats**
  - JSON output for all observability tools
  - Daily summary JSON schema with standardized fields
  - GA KPI JSON schema with baseline comparison
  - Anomaly JSON schema with severity and confidence
  - Regression JSON schema with before/after metrics

- **CLI Infrastructure**
  - Argument parsing with `argparse` for all tools
  - Test mode support (`--test-mode`) for dry runs
  - Custom output path support (`--output`)
  - Verbose logging support (`--verbose`)

- **Documentation**
  - Initial implementation reports for all observability tools
  - CLI usage documentation
  - JSON schema documentation

### Changed

- **Performance Improvements**
  - Improved latency and availability scoring algorithms
  - Reduced memory usage in 7-day stability monitoring (30% reduction)
  - Optimized anomaly detection window size for faster processing

- **Documentation**
  - Expanded README with observability tool descriptions
  - Added example outputs for all tools
  - Improved error message clarity

### Fixed

- **Time Window Alignment**
  - Fixed off-by-one errors in 7-day stability window calculations
  - Fixed timezone handling for daily summary generation
  - Fixed edge cases in time range parsing

- **Drift Detection**
  - Fixed drift detection false positives for stable metrics
  - Fixed drift calculation for metrics with high variance
  - Fixed edge cases in baseline comparison

### Security

- **Initial Security Features**
  - Added basic TLS support for API calls (partial implementation)
  - Added input validation for all CLI arguments
  - Added file permission checks for output directories

---

## [1.0.0] - 2025-09-01

### Added

- **First Production-Ready Release**
  - Core observability pipeline with KPI baselines
  - Daily summary generation with key metrics
  - Drift detection for performance degradation
  - Basic anomaly pattern recognition

- **Initial Documentation**
  - Project README with overview and features
  - Installation instructions
  - Basic usage examples
  - Architecture documentation

- **Core Infrastructure**
  - Python 3.9+ support
  - JSON-based data storage
  - File-based configuration
  - Logging infrastructure

### Changed

- Stabilized API interfaces for all core modules
- Finalized JSON output schemas
- Improved error handling across all modules

### Fixed

- Various bug fixes from beta releases
- Performance optimizations
- Documentation corrections

---

## [0.9.0] - 2025-08-15

### Added

- **Beta Release**
  - Beta version of GA KPI collector
  - Beta version of stability monitor
  - Initial anomaly detection prototype
  - Basic retrospective generation

- **Testing Infrastructure**
  - Unit tests for core modules
  - Integration test framework
  - Test data generation utilities

### Changed

- Refactored data collection pipeline
- Improved metric calculation accuracy
- Updated output formats based on user feedback

### Fixed

- Data parsing edge cases
- Metric calculation overflow errors
- Output file handling issues

---

## [0.8.0] - 2025-07-01

### Added

- **Pre-GA / Alpha Release**
  - Initial prototype of observability pipeline
  - Basic KPI tracking
  - Simple daily summary generation
  - Prototype anomaly detection

- **Development Infrastructure**
  - Project structure and build system
  - Development environment setup
  - Initial CI/CD pipeline

### Changed

- Established coding standards
- Defined output format specifications
- Refined metric definitions

### Fixed

- Initial bug fixes from prototype testing
- Data loading edge cases
- Configuration parsing issues

---

[Unreleased]: https://github.com/oceanrockr/VDS_TARS/compare/v1.0.2-RC1...HEAD
[1.0.2-RC1]: https://github.com/oceanrockr/VDS_TARS/compare/v1.0.1...v1.0.2-RC1
[1.0.1]: https://github.com/oceanrockr/VDS_TARS/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/oceanrockr/VDS_TARS/compare/v0.9.0...v1.0.0
[0.9.0]: https://github.com/oceanrockr/VDS_TARS/compare/v0.8.0...v0.9.0
[0.8.0]: https://github.com/oceanrockr/VDS_TARS/tree/v0.8.0
