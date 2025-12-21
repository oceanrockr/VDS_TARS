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
