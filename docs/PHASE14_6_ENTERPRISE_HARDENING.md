# Phase 14.6 Enterprise Hardening Guide

**Version:** v1.0.2-dev
**Last Updated:** 2025-11-26
**Status:** Production Ready

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Enterprise Configuration System](#2-enterprise-configuration-system)
3. [Compliance Framework](#3-compliance-framework)
4. [Security Hardening](#4-security-hardening)
5. [Telemetry & Logging](#5-telemetry--logging)
6. [Production Deployment](#6-production-deployment)
7. [Troubleshooting](#7-troubleshooting)

---

## 1. Introduction

### 1.1 Enterprise Features Overview

T.A.R.S. Phase 14.6 introduces enterprise-grade features designed for production environments requiring compliance, security, and auditability:

**Core Enterprise Components:**
- **Enterprise Configuration System** - Multi-source hierarchical configuration with environment profiles
- **Compliance Framework** - SOC 2, ISO 27001, GDPR runtime enforcement
- **Security Hardening** - AES-256 encryption, RSA-PSS signing, SBOM/SLSA provenance
- **Enterprise API** - RBAC-protected FastAPI with JWT and API key authentication
- **Telemetry & Logging** - Prometheus metrics and structured JSON logging

**Key Capabilities:**
- Runtime compliance validation and enforcement
- Cryptographic audit trails with tamper detection
- Automated SBOM and SLSA provenance generation
- Multi-source secrets management (Vault, AWS, GCP, file, environment)
- Role-based access control with three tiers (readonly, sre, admin)
- Comprehensive telemetry with 7+ Prometheus metrics

### 1.2 Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    T.A.R.S. Enterprise Layer                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────────┐        ┌──────────────────┐               │
│  │ Enterprise API   │◄───────┤ RBAC + JWT Auth  │               │
│  │  (Port 8100)     │        │  API Keys        │               │
│  └────────┬─────────┘        └──────────────────┘               │
│           │                                                       │
│           ▼                                                       │
│  ┌──────────────────────────────────────────────────────┐       │
│  │           Enterprise Configuration System             │       │
│  │  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐  │       │
│  │  │ CLI  │→ │ Env  │→ │ File │→ │Vault │→ │Dflt  │  │       │
│  │  └──────┘  └──────┘  └──────┘  └──────┘  └──────┘  │       │
│  │           Precedence: CLI > Env > File > Vault      │       │
│  └──────────────────┬───────────────────────────────────┘       │
│                     │                                             │
│           ┌─────────┴─────────┐                                 │
│           ▼                   ▼                                  │
│  ┌────────────────┐  ┌────────────────┐                        │
│  │ Compliance     │  │ Security       │                         │
│  │ Enforcer       │  │ Manager        │                         │
│  ├────────────────┤  ├────────────────┤                        │
│  │ • 38 Controls  │  │ • AES-256      │                         │
│  │ • SOC 2 (18)   │  │ • PGP          │                         │
│  │ • ISO 27001    │  │ • RSA-PSS      │                         │
│  │ • GDPR         │  │ • SBOM         │                         │
│  │ • Audit Chain  │  │ • SLSA         │                         │
│  └────────────────┘  └────────────────┘                        │
│           │                   │                                  │
│           └─────────┬─────────┘                                 │
│                     ▼                                             │
│           ┌──────────────────┐                                  │
│           │ Telemetry        │                                   │
│           │ • Prometheus     │                                   │
│           │ • Structured Log │                                   │
│           └──────────────────┘                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 1.3 When to Use Enterprise Features

**Use enterprise features when you need:**

✅ **Compliance Requirements**
- SOC 2 Type II certification
- ISO 27001 certification
- GDPR data protection
- HIPAA PHI handling
- Financial services regulations

✅ **Security Hardening**
- Encrypted audit trails
- Cryptographically signed reports
- Software supply chain security (SBOM/SLSA)
- Secrets management beyond environment variables
- Multi-factor authentication and RBAC

✅ **Production Operations**
- Multi-environment configuration (dev/staging/prod)
- Centralized secrets management (Vault, AWS Secrets Manager)
- Comprehensive telemetry and observability
- API-first architecture for integrations
- Automated compliance reporting

**Skip enterprise features if:**
- Running in development only
- Single-user local deployment
- No compliance requirements
- Simple configuration needs

---

## 2. Enterprise Configuration System

### 2.1 Quick Start

**Basic Usage:**

```python
from enterprise_config import load_enterprise_config

# Load configuration with default profile
config = load_enterprise_config()

# Access configuration
print(config.api.base_url)
print(config.security.encryption_enabled)
print(config.compliance.enabled_standards)
```

**With Environment Profile:**

```python
# Load production configuration
config = load_enterprise_config(profile="prod")

# Load with CLI overrides
config = load_enterprise_config(
    profile="prod",
    overrides={
        "api.port": 8200,
        "security.encryption_enabled": True
    }
)
```

**Command-Line Usage:**

```bash
# Use default profile (local)
python observability/ga_kpi_collector.py

# Specify profile
python observability/ga_kpi_collector.py --profile prod

# Override individual values
python observability/ga_kpi_collector.py \
  --profile prod \
  --config api.port=8200 \
  --config security.encryption_enabled=true
```

### 2.2 Configuration Precedence

The configuration system uses a **multi-source loader with strict precedence**:

```
┌─────────────────────────────────────────────────────┐
│                Priority (Highest → Lowest)           │
├─────────────────────────────────────────────────────┤
│  1. CLI Arguments        (--config key=value)        │
│  2. Environment Variables (TARS_SECTION_KEY)         │
│  3. Configuration Files  (config/{profile}.yaml)     │
│  4. Secrets Backend      (Vault, AWS, GCP)           │
│  5. Defaults             (Pydantic schema defaults)  │
└─────────────────────────────────────────────────────┘
```

**Example Precedence Resolution:**

Given:
- `config/prod.yaml`: `api.port: 8100`
- Environment: `TARS_API_PORT=8200`
- CLI: `--config api.port=8300`

Result: `config.api.port == 8300` (CLI wins)

**Deep Merging:**

Configuration sources are **deep-merged**, not overwritten:

```yaml
# config/prod.yaml
security:
  encryption_enabled: true
  signing_enabled: true
  key_path: "/etc/tars/keys"

# Environment variable
TARS_SECURITY_KEY_PATH=/custom/path

# Result
security:
  encryption_enabled: true      # from file
  signing_enabled: true         # from file
  key_path: "/custom/path"      # from env (overrides file)
```

### 2.3 Environment Profiles

Four built-in profiles optimize for different deployment stages:

#### 2.3.1 Local Profile (`config/local.yaml`)

**Purpose:** Development on local machine
**Defaults:**
- API port: 8100
- Encryption: disabled
- Compliance: minimal (audit logging only)
- Secrets: environment variables
- Data retention: 7 days

**Usage:**
```bash
python observability/ga_kpi_collector.py --profile local
```

#### 2.3.2 Development Profile (`config/dev.yaml`)

**Purpose:** Shared development environment
**Defaults:**
- API port: 8100
- Encryption: enabled (demo keys)
- Compliance: SOC 2 Type I subset
- Secrets: file-based
- Data retention: 30 days
- TLS: self-signed certificates

**Usage:**
```bash
export TARS_PROFILE=dev
python observability/stability_monitor_7day.py
```

#### 2.3.3 Staging Profile (`config/staging.yaml`)

**Purpose:** Pre-production testing
**Defaults:**
- API port: 8100
- Encryption: enabled (production keys)
- Compliance: SOC 2 + ISO 27001
- Secrets: Vault or AWS Secrets Manager
- Data retention: 90 days
- TLS: valid certificates
- Rate limiting: 100 req/min

**Usage:**
```bash
TARS_PROFILE=staging python enterprise_api/main.py
```

#### 2.3.4 Production Profile (`config/prod.yaml`)

**Purpose:** Production deployment
**Defaults:**
- API port: 8100
- Encryption: **required**
- Compliance: SOC 2 Type II + ISO 27001 + GDPR
- Secrets: Vault (primary), AWS/GCP (fallback)
- Data retention: 365 days (configurable per standard)
- TLS: **required** with cert rotation
- Rate limiting: 30 req/min (public), 10 req/min (auth)
- Audit logging: comprehensive

**Usage:**
```bash
TARS_PROFILE=prod \
TARS_SECRETS_VAULT_URL=https://vault.example.com \
TARS_SECRETS_VAULT_TOKEN=hvs.xxx \
python scripts/run_api_server.py
```

### 2.4 Configuration Schema Reference

Complete Pydantic schema with all fields and defaults:

#### 2.4.1 API Configuration

```python
class APIConfig(BaseModel):
    """API server configuration."""

    base_url: str = "http://localhost:8100"
    port: int = 8100
    host: str = "0.0.0.0"
    workers: int = 4

    # CORS
    cors_enabled: bool = True
    cors_origins: List[str] = ["*"]

    # Rate limiting
    rate_limit_enabled: bool = True
    rate_limit_requests: int = 30
    rate_limit_window: int = 60  # seconds

    # TLS
    tls_enabled: bool = False
    tls_cert_path: Optional[str] = None
    tls_key_path: Optional[str] = None
```

**Example Override:**

```yaml
# config/prod.yaml
api:
  base_url: https://tars.example.com
  port: 8443
  workers: 8
  cors_origins:
    - https://dashboard.example.com
    - https://app.example.com
  rate_limit_requests: 100
  tls_enabled: true
  tls_cert_path: /etc/tars/certs/tls.crt
  tls_key_path: /etc/tars/certs/tls.key
```

#### 2.4.2 Security Configuration

```python
class SecurityConfig(BaseModel):
    """Security and cryptography configuration."""

    # Encryption
    encryption_enabled: bool = False
    encryption_algorithm: str = "AES256"
    encryption_key_path: Optional[str] = None

    # Signing
    signing_enabled: bool = False
    signing_algorithm: str = "RSA-PSS"
    signing_key_path: Optional[str] = None
    public_key_path: Optional[str] = None

    # PGP
    pgp_enabled: bool = False
    pgp_key_id: Optional[str] = None
    pgp_home_dir: Optional[str] = None

    # SBOM/SLSA
    sbom_enabled: bool = False
    sbom_format: str = "cyclonedx"  # or "spdx"
    slsa_enabled: bool = False
    slsa_level: int = 3
```

**Production Example:**

```yaml
security:
  encryption_enabled: true
  encryption_key_path: /etc/tars/secrets/aes.key
  signing_enabled: true
  signing_key_path: /etc/tars/secrets/rsa.key
  public_key_path: /etc/tars/secrets/rsa.pub
  sbom_enabled: true
  sbom_format: cyclonedx
  slsa_enabled: true
  slsa_level: 3
```

#### 2.4.3 Compliance Configuration

```python
class ComplianceConfig(BaseModel):
    """Compliance framework configuration."""

    enabled_standards: List[str] = []  # ["soc2", "iso27001", "gdpr"]
    enforcement_mode: str = "log"  # "log", "warn", "block"

    # Data retention (days)
    retention_logs: int = 90
    retention_metrics: int = 365
    retention_audit: int = 2555  # 7 years for SOC 2

    # Redaction
    redaction_enabled: bool = True
    pii_fields: List[str] = ["email", "ip_address", "user_id"]

    # Audit
    audit_chain_enabled: bool = True
    audit_log_path: str = "./logs/audit.log"
```

**SOC 2 + ISO 27001 Example:**

```yaml
compliance:
  enabled_standards:
    - soc2
    - iso27001
    - gdpr
  enforcement_mode: block  # Fail if violations detected
  retention_logs: 90
  retention_metrics: 365
  retention_audit: 2555
  redaction_enabled: true
  pii_fields:
    - email
    - ip_address
    - user_id
    - ssn
    - credit_card
  audit_chain_enabled: true
  audit_log_path: /var/log/tars/audit.log
```

#### 2.4.4 Secrets Configuration

```python
class SecretsConfig(BaseModel):
    """Secrets management configuration."""

    backend: str = "env"  # "vault", "aws", "gcp", "file", "env"

    # Vault
    vault_url: Optional[str] = None
    vault_token: Optional[str] = None
    vault_mount: str = "secret"
    vault_path: str = "tars"

    # AWS
    aws_region: Optional[str] = None
    aws_secret_name: Optional[str] = None

    # GCP
    gcp_project: Optional[str] = None
    gcp_secret_name: Optional[str] = None

    # File
    secrets_file_path: Optional[str] = None
```

**Vault Example:**

```yaml
secrets:
  backend: vault
  vault_url: https://vault.example.com
  vault_token: ${VAULT_TOKEN}  # from environment
  vault_mount: secret
  vault_path: tars/prod
```

**AWS Secrets Manager Example:**

```yaml
secrets:
  backend: aws
  aws_region: us-east-1
  aws_secret_name: tars/prod/config
```

#### 2.4.5 Observability Configuration

```python
class ObservabilityConfig(BaseModel):
    """Observability and telemetry configuration."""

    # Prometheus
    prometheus_enabled: bool = True
    prometheus_url: str = "http://localhost:9090"
    prometheus_port: int = 9091  # pushgateway

    # Logging
    log_level: str = "INFO"
    log_format: str = "json"  # "json" or "text"
    log_file: Optional[str] = None

    # Metrics
    metrics_enabled: bool = True
    metrics_prefix: str = "tars"
```

**Production Observability:**

```yaml
observability:
  prometheus_enabled: true
  prometheus_url: http://prometheus.example.com
  prometheus_port: 9091
  log_level: INFO
  log_format: json
  log_file: /var/log/tars/app.log
  metrics_enabled: true
  metrics_prefix: tars_prod
```

### 2.5 Advanced Configuration

#### 2.5.1 Environment Variable Mapping

All configuration fields can be set via environment variables using the pattern:

```
TARS_<SECTION>_<KEY>=<value>
```

**Examples:**

```bash
# API configuration
export TARS_API_PORT=8200
export TARS_API_BASE_URL=https://tars.example.com
export TARS_API_WORKERS=8

# Security configuration
export TARS_SECURITY_ENCRYPTION_ENABLED=true
export TARS_SECURITY_SIGNING_KEY_PATH=/etc/tars/rsa.key

# Compliance configuration
export TARS_COMPLIANCE_ENABLED_STANDARDS=soc2,iso27001,gdpr
export TARS_COMPLIANCE_ENFORCEMENT_MODE=block

# Secrets configuration
export TARS_SECRETS_BACKEND=vault
export TARS_SECRETS_VAULT_URL=https://vault.example.com
export TARS_SECRETS_VAULT_TOKEN=hvs.xxx
```

**List Values:**

For list fields, use comma separation:

```bash
export TARS_API_CORS_ORIGINS=https://app1.com,https://app2.com
export TARS_COMPLIANCE_ENABLED_STANDARDS=soc2,iso27001
export TARS_COMPLIANCE_PII_FIELDS=email,ssn,phone
```

#### 2.5.2 Secrets Interpolation

Configuration values can reference secrets using `${SECRET_NAME}` syntax:

```yaml
# config/prod.yaml
security:
  signing_key_path: ${TARS_SIGNING_KEY_PATH}

secrets:
  vault_token: ${VAULT_TOKEN}

api:
  jwt_secret: ${JWT_SECRET_KEY}
```

Secrets are resolved in this order:
1. Secrets backend (Vault, AWS, GCP)
2. Environment variables
3. Literal value (fallback)

#### 2.5.3 Custom Profiles

Create custom profiles by adding YAML files to `enterprise_config/config/`:

```bash
# Create custom profile
cat > enterprise_config/config/staging-us-west.yaml <<EOF
api:
  base_url: https://tars-staging-usw.example.com
  port: 8443

observability:
  prometheus_url: http://prometheus-usw.example.com

compliance:
  enabled_standards:
    - soc2
    - iso27001
EOF

# Load custom profile
python observability/ga_kpi_collector.py --profile staging-us-west
```

#### 2.5.4 Validation and Type Safety

All configuration is validated at load time using Pydantic:

```python
from enterprise_config import load_enterprise_config, EnterpriseConfigError

try:
    config = load_enterprise_config(profile="prod")
except EnterpriseConfigError as e:
    print(f"Configuration error: {e}")
    # Example errors:
    # - Missing required field: security.signing_key_path
    # - Invalid value for compliance.enforcement_mode: "invalid"
    # - Port must be between 1-65535, got 99999
```

**Field Validation:**

```python
from enterprise_config import APIConfig
from pydantic import ValidationError

try:
    api_config = APIConfig(
        port=99999,  # Invalid
        workers=-1   # Invalid
    )
except ValidationError as e:
    print(e.errors())
    # [
    #   {
    #     'loc': ('port',),
    #     'msg': 'ensure this value is less than or equal to 65535',
    #     'type': 'value_error'
    #   },
    #   {
    #     'loc': ('workers',),
    #     'msg': 'ensure this value is greater than or equal to 1',
    #     'type': 'value_error'
    #   }
    # ]
```

---

## 3. Compliance Framework

### 3.1 Supported Standards

T.A.R.S. implements runtime compliance enforcement for three major standards:

| Standard | Controls | Focus Area | Retention |
|----------|----------|------------|-----------|
| **SOC 2 Type II** | 18 controls | Security, availability, confidentiality | 7 years |
| **ISO 27001** | 20 controls | Information security management | 3 years |
| **GDPR** | Partial | Data protection and privacy | 3 years |

**Total Controls:** 38 unique controls across all standards

### 3.2 SOC 2 Implementation

SOC 2 Type II focuses on **Trust Service Criteria (TSC)**:

#### 3.2.1 Security (CC)

**CC6.1 - Logical Access Controls**
```python
from compliance import ComplianceEnforcer

enforcer = ComplianceEnforcer(enabled_standards=["soc2"])

# Validate access control
enforcer.validate_access_control(
    user_role="viewer",
    required_role="admin",
    resource="/api/admin/users"
)
# Raises ComplianceViolation if viewer tries to access admin resource
```

**CC6.2 - Authentication**
```python
# Enforce authentication requirements
enforcer.enforce_control(
    control_id="CC6.2",
    context={
        "mfa_enabled": True,
        "password_complexity": "strong",
        "session_timeout": 3600
    }
)
```

**CC6.6 - Encryption**
```python
# Validate encryption requirements
enforcer.validate_encryption(
    data=sensitive_data,
    algorithm="AES256",
    key_rotation_days=90
)
```

**CC6.7 - Data Retention**
```python
# Enforce data retention policies
enforcer.enforce_retention(
    data_type="audit_logs",
    retention_days=2555  # 7 years for SOC 2
)
```

#### 3.2.2 Availability (A)

**A1.2 - System Monitoring**
```python
# Validate monitoring coverage
enforcer.validate_monitoring(
    metrics=["uptime", "latency", "error_rate"],
    alert_threshold=0.99
)
```

#### 3.2.3 Confidentiality (C)

**C1.1 - Data Classification**
```python
# Classify and protect data
enforcer.classify_data(
    data={"email": "user@example.com", "ip": "192.168.1.1"},
    classification="confidential"
)
```

**C1.2 - PII Redaction**
```python
# Redact PII before logging
redacted = enforcer.redact_pii(
    data={"user": "john@example.com", "ip": "192.168.1.1"}
)
# Result: {"user": "***REDACTED***", "ip": "***REDACTED***"}
```

### 3.3 ISO 27001 Implementation

ISO 27001 focuses on **Information Security Controls**:

#### 3.3.1 Access Control (A.9)

**A.9.2.1 - User Registration**
```python
# Validate user registration process
enforcer.enforce_control(
    control_id="A.9.2.1",
    context={
        "approval_required": True,
        "role_assignment": "least_privilege"
    }
)
```

**A.9.4.1 - Access Restriction**
```python
# Restrict access to information
enforcer.validate_access_control(
    user_role="sre",
    required_role="admin",
    resource="/admin/compliance/audit"
)
```

#### 3.3.2 Cryptography (A.10)

**A.10.1.1 - Cryptographic Controls**
```python
# Validate cryptographic implementation
enforcer.validate_encryption(
    algorithm="AES256",
    key_length=256,
    mode="GCM"
)
```

#### 3.3.3 Operations Security (A.12)

**A.12.4.1 - Event Logging**
```python
# Ensure comprehensive event logging
enforcer.log_event(
    event_type="access_attempt",
    user="admin",
    resource="/api/admin/users",
    outcome="success"
)
```

**A.12.4.2 - Log Protection**
```python
# Protect log integrity
enforcer.protect_logs(
    log_file="/var/log/tars/audit.log",
    integrity_check=True,
    encryption=True
)
```

### 3.4 GDPR Compliance

GDPR implementation focuses on **data protection and privacy**:

#### 3.4.1 Article 5 - Data Principles

**Lawfulness, Fairness, Transparency**
```python
# Log data processing with legal basis
enforcer.log_data_processing(
    purpose="system_monitoring",
    legal_basis="legitimate_interest",
    data_subject="user@example.com"
)
```

**Purpose Limitation**
```python
# Validate data usage matches stated purpose
enforcer.validate_purpose(
    data=user_data,
    stated_purpose="authentication",
    actual_purpose="authentication"
)
```

**Data Minimization**
```python
# Ensure only necessary data is collected
enforcer.minimize_data(
    collected_fields=["email", "ip", "timestamp"],
    required_fields=["email", "timestamp"]
)
# Warns or blocks collection of unnecessary "ip" field
```

#### 3.4.2 Article 17 - Right to Erasure

```python
# Implement data deletion
enforcer.delete_personal_data(
    subject_id="user@example.com",
    cascading=True,  # Delete from all systems
    verification=True  # Verify deletion
)
```

#### 3.4.3 Article 32 - Security of Processing

```python
# Validate security measures
enforcer.validate_security_measures(
    encryption=True,
    pseudonymization=True,
    access_control=True,
    regular_testing=True
)
```

### 3.5 Runtime Enforcement

The compliance enforcer operates in three modes:

#### 3.5.1 Log Mode (Default)

Violations are logged but do not block operations:

```python
enforcer = ComplianceEnforcer(
    enabled_standards=["soc2"],
    enforcement_mode="log"
)

# Violation is logged, operation continues
enforcer.validate_access_control(
    user_role="viewer",
    required_role="admin",
    resource="/admin/users"
)
```

**Log Output:**
```json
{
  "timestamp": "2025-11-26T10:30:00Z",
  "level": "WARNING",
  "event": "compliance_violation",
  "standard": "soc2",
  "control": "CC6.1",
  "message": "User role 'viewer' insufficient for resource '/admin/users'",
  "required_role": "admin",
  "actual_role": "viewer"
}
```

#### 3.5.2 Warn Mode

Violations trigger warnings but operations continue:

```python
enforcer = ComplianceEnforcer(
    enabled_standards=["soc2", "iso27001"],
    enforcement_mode="warn"
)

# Raises warning, operation continues
with warnings.catch_warnings(record=True) as w:
    enforcer.validate_encryption(algorithm="DES")  # Weak algorithm
    assert len(w) == 1
    assert "weak encryption" in str(w[0].message).lower()
```

#### 3.5.3 Block Mode (Production)

Violations raise exceptions and block operations:

```python
enforcer = ComplianceEnforcer(
    enabled_standards=["soc2", "iso27001", "gdpr"],
    enforcement_mode="block"
)

try:
    # Violation raises exception, operation blocked
    enforcer.validate_access_control(
        user_role="viewer",
        required_role="admin",
        resource="/admin/compliance/audit"
    )
except ComplianceViolation as e:
    print(f"Access denied: {e}")
    # Access denied: Insufficient role for resource (required: admin, actual: viewer)
```

**Production Configuration:**

```yaml
# config/prod.yaml
compliance:
  enabled_standards:
    - soc2
    - iso27001
    - gdpr
  enforcement_mode: block  # Fail fast on violations
```

### 3.6 Audit Trail

All compliance events are logged to a **cryptographically chained audit trail**:

#### 3.6.1 Audit Chain Structure

```python
{
  "event_id": "evt_1234567890",
  "timestamp": "2025-11-26T10:30:00Z",
  "event_type": "access_attempt",
  "user": "admin@example.com",
  "resource": "/api/admin/users",
  "outcome": "success",
  "standard": "soc2",
  "control": "CC6.1",
  "previous_hash": "sha256:abc123...",
  "current_hash": "sha256:def456...",
  "signature": "RSA-PSS:ghi789..."
}
```

#### 3.6.2 Tamper Detection

The audit chain uses **cryptographic hashing** to detect tampering:

```python
from compliance import AuditChain

audit_chain = AuditChain(log_path="/var/log/tars/audit.log")

# Verify chain integrity
is_valid = audit_chain.verify_integrity()
if not is_valid:
    print("CRITICAL: Audit log tampering detected!")
    # Alert security team, freeze operations
```

**Integrity Check:**
```bash
# Verify audit chain from CLI
python -m compliance verify-audit --log /var/log/tars/audit.log

# Output:
# ✓ Audit chain verified: 1,234 events
# ✓ No tampering detected
# ✓ All signatures valid
```

#### 3.6.3 Audit Queries

Query audit logs for compliance reporting:

```python
from compliance import query_audit_log

# Find all admin access attempts in last 30 days
events = query_audit_log(
    event_type="access_attempt",
    user_role="admin",
    start_date="2025-10-27",
    end_date="2025-11-26"
)

# Generate SOC 2 compliance report
report = generate_compliance_report(
    standard="soc2",
    period_start="2025-01-01",
    period_end="2025-12-31"
)
```

#### 3.6.4 Retention Enforcement

Audit logs are automatically retained per standard requirements:

```python
from compliance import enforce_retention

# Enforce retention policies
enforce_retention(
    log_path="/var/log/tars/audit.log",
    retention_days=2555,  # 7 years for SOC 2
    archive_path="/var/archive/tars/audit/"
)
```

**Automated Retention:**

```bash
# Kubernetes CronJob for retention enforcement
apiVersion: batch/v1
kind: CronJob
metadata:
  name: audit-retention-enforcer
spec:
  schedule: "0 2 * * *"  # Daily at 2am
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: enforcer
            image: tars:v1.0.2
            command:
            - python
            - -m
            - compliance
            - enforce-retention
            - --log
            - /var/log/tars/audit.log
            - --retention
            - "2555"
            - --archive
            - /var/archive/tars/audit/
```

---

## 4. Security Hardening

### 4.1 Encryption

#### 4.1.1 AES-256 Setup

T.A.R.S. uses **AES-256-GCM** for symmetric encryption:

**Generate Encryption Key:**

```bash
# Generate 256-bit AES key
python -c "import os; print(os.urandom(32).hex())" > /etc/tars/secrets/aes.key
chmod 600 /etc/tars/secrets/aes.key
```

**Configure Encryption:**

```yaml
# config/prod.yaml
security:
  encryption_enabled: true
  encryption_algorithm: AES256
  encryption_key_path: /etc/tars/secrets/aes.key
```

**Encrypt Data:**

```python
from security import SecurityManager

security = SecurityManager(
    encryption_key_path="/etc/tars/secrets/aes.key"
)

# Encrypt sensitive data
plaintext = "Confidential report data"
ciphertext = security.encrypt_data(plaintext)

# Decrypt
decrypted = security.decrypt_data(ciphertext)
assert decrypted == plaintext
```

**Encrypt Files:**

```python
# Encrypt retrospective report
security.encrypt_file(
    input_path="reports/retrospective_2025-11.json",
    output_path="reports/retrospective_2025-11.json.enc"
)

# Decrypt
security.decrypt_file(
    input_path="reports/retrospective_2025-11.json.enc",
    output_path="reports/retrospective_2025-11.json"
)
```

#### 4.1.2 PGP Integration

For **asymmetric encryption** and multi-recipient scenarios:

**Generate PGP Key:**

```bash
# Generate GPG key pair
gpg --batch --gen-key <<EOF
Key-Type: RSA
Key-Length: 4096
Name-Real: T.A.R.S. Production
Name-Email: tars@example.com
Expire-Date: 2y
%no-protection
%commit
EOF

# Export public key
gpg --armor --export tars@example.com > /etc/tars/secrets/pgp.pub

# Get key ID
export TARS_PGP_KEY_ID=$(gpg --list-keys tars@example.com | grep -A1 pub | tail -1 | tr -d ' ')
```

**Configure PGP:**

```yaml
# config/prod.yaml
security:
  pgp_enabled: true
  pgp_key_id: ${TARS_PGP_KEY_ID}
  pgp_home_dir: /etc/tars/.gnupg
```

**Encrypt with PGP:**

```python
from security import SecurityManager

security = SecurityManager(
    pgp_enabled=True,
    pgp_key_id="ABCD1234EF567890"
)

# Encrypt for recipient
ciphertext = security.pgp_encrypt(
    plaintext="Sensitive data",
    recipient="recipient@example.com"
)

# Decrypt
plaintext = security.pgp_decrypt(ciphertext)
```

#### 4.1.3 Key Management

**Best Practices:**

✅ **DO:**
- Store keys in secure secrets backend (Vault, AWS, GCP)
- Rotate encryption keys every 90 days
- Use different keys for dev/staging/prod
- Backup keys to encrypted offline storage
- Use hardware security modules (HSM) for production

❌ **DON'T:**
- Commit keys to version control
- Share keys via email or Slack
- Use same key across environments
- Store keys in application code
- Use weak key derivation (e.g., passwords)

**Key Rotation:**

```python
from security import rotate_encryption_key

# Rotate AES key
new_key_path = rotate_encryption_key(
    old_key_path="/etc/tars/secrets/aes.key",
    new_key_path="/etc/tars/secrets/aes.key.new",
    re_encrypt_paths=[
        "reports/*.json.enc",
        "logs/*.log.enc"
    ]
)

# Atomic swap
os.rename("/etc/tars/secrets/aes.key", "/etc/tars/secrets/aes.key.old")
os.rename(new_key_path, "/etc/tars/secrets/aes.key")
```

**Vault Integration:**

```bash
# Store encryption key in Vault
vault kv put secret/tars/prod/encryption key=@/etc/tars/secrets/aes.key

# Configure T.A.R.S. to fetch from Vault
export TARS_SECRETS_BACKEND=vault
export TARS_SECRETS_VAULT_URL=https://vault.example.com
export TARS_SECRETS_VAULT_TOKEN=hvs.xxx
export TARS_SECRETS_VAULT_PATH=tars/prod/encryption
```

### 4.2 Cryptographic Signing

#### 4.2.1 RSA Key Generation

Generate **RSA-4096** key pair for signing:

```bash
# Generate private key
openssl genrsa -out /etc/tars/secrets/rsa.key 4096
chmod 600 /etc/tars/secrets/rsa.key

# Extract public key
openssl rsa -in /etc/tars/secrets/rsa.key -pubout -out /etc/tars/secrets/rsa.pub
chmod 644 /etc/tars/secrets/rsa.pub
```

**Configure Signing:**

```yaml
# config/prod.yaml
security:
  signing_enabled: true
  signing_algorithm: RSA-PSS
  signing_key_path: /etc/tars/secrets/rsa.key
  public_key_path: /etc/tars/secrets/rsa.pub
```

#### 4.2.2 Signing Reports

Sign JSON reports for **integrity and non-repudiation**:

```python
from security import SecurityManager

security = SecurityManager(
    signing_key_path="/etc/tars/secrets/rsa.key",
    public_key_path="/etc/tars/secrets/rsa.pub"
)

# Sign report
report = {
    "timestamp": "2025-11-26T10:30:00Z",
    "metrics": {"availability": 99.95},
    "summary": "All systems operational"
}

signed_report = security.sign_json(report)

# Result:
# {
#   "data": {...},  # Original report
#   "signature": "RSA-PSS:base64encoded...",
#   "signed_at": "2025-11-26T10:30:00Z",
#   "signer": "tars@example.com"
# }
```

**Save Signed Report:**

```python
# Write to file with embedded signature
with open("reports/retrospective_signed.json", "w") as f:
    json.dump(signed_report, f, indent=2)
```

#### 4.2.3 Signature Verification

Verify signed reports:

```python
from security import SecurityManager

security = SecurityManager(
    public_key_path="/etc/tars/secrets/rsa.pub"
)

# Load signed report
with open("reports/retrospective_signed.json") as f:
    signed_report = json.load(f)

# Verify signature
is_valid = security.verify_signature(signed_report)

if is_valid:
    print("✓ Signature valid - report integrity verified")
else:
    print("✗ Signature invalid - report may be tampered")
```

**CLI Verification:**

```bash
# Verify signed report from CLI
python -m security verify \
  --input reports/retrospective_signed.json \
  --public-key /etc/tars/secrets/rsa.pub

# Output:
# ✓ Signature valid
# ✓ Signed by: tars@example.com
# ✓ Signed at: 2025-11-26T10:30:00Z
# ✓ Report integrity verified
```

### 4.3 SBOM Generation

Generate **Software Bill of Materials** for supply chain security:

#### 4.3.1 CycloneDX Format

```python
from security import generate_sbom

# Generate SBOM
sbom = generate_sbom(
    format="cyclonedx",
    output_path="sbom/tars-v1.0.2-sbom.json"
)

# Result:
# {
#   "bomFormat": "CycloneDX",
#   "specVersion": "1.4",
#   "version": 1,
#   "metadata": {
#     "component": {
#       "name": "T.A.R.S.",
#       "version": "1.0.2"
#     }
#   },
#   "components": [
#     {
#       "type": "library",
#       "name": "fastapi",
#       "version": "0.104.1",
#       "purl": "pkg:pypi/fastapi@0.104.1"
#     },
#     ...
#   ]
# }
```

#### 4.3.2 SPDX Format

```python
# Generate SPDX SBOM
sbom = generate_sbom(
    format="spdx",
    output_path="sbom/tars-v1.0.2-sbom.spdx.json"
)
```

**Automated SBOM Generation:**

```bash
# Generate SBOM during build
python -m security generate-sbom \
  --format cyclonedx \
  --output dist/tars-v1.0.2-sbom.json \
  --sign \
  --signing-key /etc/tars/secrets/rsa.key
```

### 4.4 SLSA Provenance

Generate **SLSA Level 3** provenance for build integrity:

```python
from security import generate_slsa_provenance

# Generate provenance
provenance = generate_slsa_provenance(
    artifact_path="dist/tars-v1.0.2.tar.gz",
    builder="GitHub Actions",
    build_type="https://github.com/Veleron-Dev-Studios/tars/workflows/release",
    output_path="dist/tars-v1.0.2.provenance.json"
)

# Result:
# {
#   "_type": "https://in-toto.io/Statement/v0.1",
#   "subject": [
#     {
#       "name": "tars-v1.0.2.tar.gz",
#       "digest": {
#         "sha256": "abc123..."
#       }
#     }
#   ],
#   "predicateType": "https://slsa.dev/provenance/v0.2",
#   "predicate": {
#     "builder": {
#       "id": "https://github.com/Veleron-Dev-Studios/tars/actions"
#     },
#     "buildType": "https://github.com/Veleron-Dev-Studios/tars/workflows/release",
#     "invocation": {...},
#     "materials": [...]
#   }
# }
```

**GitHub Actions Integration:**

```yaml
# .github/workflows/release.yml
name: Release with SLSA Provenance

on:
  push:
    tags:
      - 'v*'

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Build artifact
        run: python setup.py sdist

      - name: Generate SLSA provenance
        run: |
          python -m security generate-slsa \
            --artifact dist/tars-${{ github.ref_name }}.tar.gz \
            --builder "GitHub Actions" \
            --build-type "https://github.com/${{ github.repository }}/actions/runs/${{ github.run_id }}" \
            --output dist/tars-${{ github.ref_name }}.provenance.json \
            --sign \
            --signing-key ${{ secrets.SIGNING_KEY }}

      - name: Upload artifacts
        uses: actions/upload-artifact@v3
        with:
          name: release-artifacts
          path: |
            dist/*.tar.gz
            dist/*.provenance.json
```

---

## 5. Telemetry & Logging

### 5.1 Prometheus Metrics

T.A.R.S. exports **7 core metrics** for observability:

#### 5.1.1 Available Metrics

```python
# From metrics/telemetry.py

# Command execution tracking
tars_command_executions_total = Counter(
    "tars_command_executions_total",
    "Total command executions",
    ["command", "status"]
)

# Command duration
tars_command_duration_seconds = Histogram(
    "tars_command_duration_seconds",
    "Command execution duration",
    ["command"]
)

# Compliance violations
tars_compliance_violations_total = Counter(
    "tars_compliance_violations_total",
    "Total compliance violations",
    ["standard", "control"]
)

# Encryption operations
tars_encryption_operations_total = Counter(
    "tars_encryption_operations_total",
    "Total encryption operations",
    ["operation", "algorithm"]
)

# API requests
tars_api_requests_total = Counter(
    "tars_api_requests_total",
    "Total API requests",
    ["endpoint", "method", "status"]
)

# API latency
tars_api_latency_seconds = Histogram(
    "tars_api_latency_seconds",
    "API request latency",
    ["endpoint", "method"]
)

# Active connections
tars_active_connections = Gauge(
    "tars_active_connections",
    "Current active connections"
)
```

#### 5.1.2 Instrumenting Code

```python
from metrics import (
    track_command,
    track_compliance_violation,
    track_encryption_operation
)

# Track command execution
@track_command("ga_kpi_collection")
def collect_ga_kpi():
    # Command logic
    pass

# Track compliance violations
from compliance import ComplianceEnforcer

enforcer = ComplianceEnforcer(enabled_standards=["soc2"])
# Violations automatically tracked to Prometheus

# Track encryption operations
from security import SecurityManager

security = SecurityManager()
ciphertext = security.encrypt_data("data")  # Automatically tracked
```

#### 5.1.3 Prometheus Configuration

**Scrape Config:**

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'tars-api'
    static_configs:
      - targets: ['localhost:8100']
    metrics_path: '/metrics'
    scrape_interval: 15s

  - job_name: 'tars-pushgateway'
    static_configs:
      - targets: ['localhost:9091']
    scrape_interval: 30s
```

**Push to Gateway:**

```python
from prometheus_client import push_to_gateway

# Push metrics after batch job
push_to_gateway(
    gateway='localhost:9091',
    job='ga_kpi_collector',
    registry=registry
)
```

### 5.2 Structured Logging

All logs use **structured JSON format** for parsing and analysis:

#### 5.2.1 JSON Format

```python
from metrics import get_logger

logger = get_logger("ga_kpi_collector")

# Log structured event
logger.info(
    "KPI collection completed",
    extra={
        "availability": 99.95,
        "duration_seconds": 12.5,
        "metrics_collected": 150
    }
)

# Output:
# {
#   "timestamp": "2025-11-26T10:30:00.123Z",
#   "level": "INFO",
#   "logger": "ga_kpi_collector",
#   "message": "KPI collection completed",
#   "availability": 99.95,
#   "duration_seconds": 12.5,
#   "metrics_collected": 150,
#   "hostname": "tars-prod-01",
#   "pid": 12345
# }
```

#### 5.2.2 Log Levels

```python
logger.debug("Detailed debugging information")
logger.info("Normal operational events")
logger.warning("Warning conditions")
logger.error("Error conditions")
logger.critical("Critical conditions requiring immediate action")
```

#### 5.2.3 Log Configuration

```yaml
# config/prod.yaml
observability:
  log_level: INFO
  log_format: json
  log_file: /var/log/tars/app.log
```

**Text Format (Development):**

```yaml
# config/local.yaml
observability:
  log_level: DEBUG
  log_format: text
  log_file: null  # stdout
```

### 5.3 Grafana Dashboards

Example Grafana dashboard queries:

#### 5.3.1 Availability Dashboard

```promql
# Overall availability (last 24h)
avg_over_time(tars_availability_percent[24h])

# Availability by service
avg by (service) (tars_availability_percent)

# Downtime events
increase(tars_command_executions_total{status="failed"}[1h])
```

#### 5.3.2 Compliance Dashboard

```promql
# Compliance violations by standard
sum by (standard) (increase(tars_compliance_violations_total[24h]))

# Top violated controls
topk(10, sum by (control) (increase(tars_compliance_violations_total[24h])))

# Compliance score trend
100 - (
  rate(tars_compliance_violations_total[1h]) * 100
)
```

#### 5.3.3 Security Dashboard

```promql
# Encryption operations per second
rate(tars_encryption_operations_total[5m])

# Failed authentications
increase(tars_api_requests_total{endpoint="/auth/login",status="401"}[1h])

# API latency p99
histogram_quantile(0.99, tars_api_latency_seconds_bucket)
```

**Dashboard JSON:**

See `observability/dashboards/tars_enterprise.json` for complete Grafana dashboard.

---

## 6. Production Deployment

### 6.1 Secrets Management

#### 6.1.1 HashiCorp Vault

**Setup:**

```bash
# Start Vault (dev mode for testing)
vault server -dev

# Enable KV secrets engine
vault secrets enable -path=secret kv-v2

# Store T.A.R.S. secrets
vault kv put secret/tars/prod/config \
  jwt_secret=supersecret \
  encryption_key=$(python -c "import os; print(os.urandom(32).hex())") \
  signing_key=@/etc/tars/secrets/rsa.key

# Create policy
vault policy write tars-prod - <<EOF
path "secret/data/tars/prod/*" {
  capabilities = ["read"]
}
EOF

# Create token
vault token create -policy=tars-prod
```

**Configure T.A.R.S.:**

```yaml
# config/prod.yaml
secrets:
  backend: vault
  vault_url: https://vault.example.com
  vault_token: ${VAULT_TOKEN}
  vault_mount: secret
  vault_path: tars/prod/config
```

**Fetch Secrets:**

```python
from enterprise_config import load_enterprise_config

# Secrets automatically loaded from Vault
config = load_enterprise_config(profile="prod")

# Access secrets
jwt_secret = config.secrets.get("jwt_secret")
encryption_key = config.secrets.get("encryption_key")
```

#### 6.1.2 AWS Secrets Manager

**Store Secrets:**

```bash
# Create secret
aws secretsmanager create-secret \
  --name tars/prod/config \
  --secret-string '{
    "jwt_secret": "supersecret",
    "encryption_key": "hex_encoded_key",
    "signing_key": "base64_encoded_key"
  }'

# Update secret
aws secretsmanager update-secret \
  --secret-id tars/prod/config \
  --secret-string file://secrets.json
```

**Configure T.A.R.S.:**

```yaml
# config/prod.yaml
secrets:
  backend: aws
  aws_region: us-east-1
  aws_secret_name: tars/prod/config
```

**IAM Policy:**

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "secretsmanager:GetSecretValue",
        "secretsmanager:DescribeSecret"
      ],
      "Resource": "arn:aws:secretsmanager:us-east-1:123456789012:secret:tars/prod/config-*"
    }
  ]
}
```

#### 6.1.3 GCP Secret Manager

**Store Secrets:**

```bash
# Create secret
echo -n "supersecret" | gcloud secrets create tars-jwt-secret --data-file=-

# Grant access
gcloud secrets add-iam-policy-binding tars-jwt-secret \
  --member="serviceAccount:tars-prod@project.iam.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"
```

**Configure T.A.R.S.:**

```yaml
# config/prod.yaml
secrets:
  backend: gcp
  gcp_project: my-project
  gcp_secret_name: tars-jwt-secret
```

### 6.2 TLS Setup

#### 6.2.1 Certificate Generation

**Self-Signed (Development):**

```bash
# Generate self-signed certificate
openssl req -x509 -newkey rsa:4096 \
  -keyout /etc/tars/certs/tls.key \
  -out /etc/tars/certs/tls.crt \
  -days 365 -nodes \
  -subj "/CN=tars.example.com"
```

**Let's Encrypt (Production):**

```bash
# Install certbot
apt-get install certbot

# Generate certificate
certbot certonly --standalone \
  -d tars.example.com \
  --email admin@example.com \
  --agree-tos

# Certificates stored in /etc/letsencrypt/live/tars.example.com/
```

**Kubernetes cert-manager:**

```yaml
# certificate.yaml
apiVersion: cert-manager.io/v1
kind: Certificate
metadata:
  name: tars-tls
  namespace: tars
spec:
  secretName: tars-tls-secret
  issuer:
    name: letsencrypt-prod
    kind: ClusterIssuer
  dnsNames:
    - tars.example.com
```

#### 6.2.2 TLS Configuration

```yaml
# config/prod.yaml
api:
  tls_enabled: true
  tls_cert_path: /etc/tars/certs/tls.crt
  tls_key_path: /etc/tars/certs/tls.key
```

**Start API with TLS:**

```python
# scripts/run_api_server.py
import uvicorn
from enterprise_config import load_enterprise_config

config = load_enterprise_config(profile="prod")

uvicorn.run(
    "enterprise_api.main:app",
    host=config.api.host,
    port=config.api.port,
    ssl_certfile=config.api.tls_cert_path,
    ssl_keyfile=config.api.tls_key_path
)
```

#### 6.2.3 Certificate Rotation

**Automated Rotation:**

```bash
# Kubernetes CronJob for cert renewal
apiVersion: batch/v1
kind: CronJob
metadata:
  name: cert-renewal
spec:
  schedule: "0 0 1 * *"  # Monthly
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: certbot
            image: certbot/certbot
            command:
            - certbot
            - renew
            - --deploy-hook
            - "kubectl rollout restart deployment/tars-api"
```

### 6.3 Security Checklist

Use this checklist before production deployment:

#### 6.3.1 Configuration Security

- [ ] All secrets stored in secure backend (Vault/AWS/GCP)
- [ ] No hardcoded credentials in code or config files
- [ ] Environment-specific profiles configured (prod.yaml)
- [ ] Compliance standards enabled (SOC 2, ISO 27001, GDPR)
- [ ] Enforcement mode set to "block"
- [ ] TLS enabled for all external APIs
- [ ] CORS origins restricted to known domains
- [ ] Rate limiting enabled

#### 6.3.2 Cryptography

- [ ] Encryption enabled (AES-256-GCM)
- [ ] Signing enabled (RSA-PSS-4096)
- [ ] Encryption keys rotated every 90 days
- [ ] Keys stored in HSM or secure backend
- [ ] Key permissions restricted (chmod 600)
- [ ] Different keys for dev/staging/prod
- [ ] Key backups encrypted and offline

#### 6.3.3 Authentication & Authorization

- [ ] JWT secret is cryptographically random (32+ bytes)
- [ ] API keys are UUID v4 or equivalent
- [ ] RBAC roles properly assigned
- [ ] Demo passwords changed or disabled
- [ ] Password complexity requirements enforced
- [ ] MFA enabled for admin accounts
- [ ] Session timeouts configured (< 1 hour)

#### 6.3.4 Audit & Compliance

- [ ] Audit logging enabled
- [ ] Audit chain integrity verified
- [ ] Retention policies configured
- [ ] Compliance controls loaded
- [ ] SBOM generated and signed
- [ ] SLSA provenance generated
- [ ] Audit logs backed up regularly
- [ ] Tamper detection alerts configured

#### 6.3.5 Network Security

- [ ] TLS 1.3 enforced
- [ ] HTTPS redirects enabled
- [ ] Rate limiting active
- [ ] CORS properly configured
- [ ] Internal services not exposed
- [ ] Network policies applied (Kubernetes)
- [ ] Firewall rules reviewed

#### 6.3.6 Operational Security

- [ ] Prometheus metrics exposed
- [ ] Grafana dashboards configured
- [ ] Alerting rules created
- [ ] Log aggregation configured
- [ ] Incident response plan documented
- [ ] Backup and recovery tested
- [ ] Disaster recovery plan in place

### 6.4 Troubleshooting

#### 6.4.1 Configuration Issues

**Problem:** Configuration not loading

```bash
# Check configuration file syntax
python -c "import yaml; yaml.safe_load(open('enterprise_config/config/prod.yaml'))"

# Verify environment variables
env | grep TARS_

# Test configuration loading
python -c "from enterprise_config import load_enterprise_config; print(load_enterprise_config(profile='prod'))"
```

**Problem:** Secrets not loading from Vault

```bash
# Verify Vault connectivity
curl -H "X-Vault-Token: $VAULT_TOKEN" \
  $VAULT_ADDR/v1/secret/data/tars/prod/config

# Check Vault token permissions
vault token lookup

# Test secret fetch
python -c "
from enterprise_config.secrets import VaultSecretsBackend
backend = VaultSecretsBackend(
    url='https://vault.example.com',
    token='hvs.xxx'
)
print(backend.get_secret('jwt_secret'))
"
```

#### 6.4.2 Compliance Issues

**Problem:** Compliance violations blocking operations

```bash
# Check compliance status
python -c "
from compliance import ComplianceEnforcer
enforcer = ComplianceEnforcer(enabled_standards=['soc2'])
status = enforcer.get_compliance_status()
print(f'Compliance: {status[\"compliance_percentage\"]:.1f}%')
print(f'Violations: {status[\"total_violations\"]}')
"

# List violated controls
python -m compliance list-violations --standard soc2

# Temporarily disable enforcement (DANGER)
export TARS_COMPLIANCE_ENFORCEMENT_MODE=log
```

**Problem:** Audit chain verification fails

```bash
# Verify audit log integrity
python -m compliance verify-audit --log /var/log/tars/audit.log

# If tampering detected, investigate:
# 1. Check file permissions: ls -la /var/log/tars/audit.log
# 2. Review file access logs: auditctl -w /var/log/tars/audit.log
# 3. Compare with backups
# 4. Alert security team immediately
```

#### 6.4.3 Encryption Issues

**Problem:** Encryption key not found

```bash
# Verify key file exists and has correct permissions
ls -la /etc/tars/secrets/aes.key

# Should be: -rw------- (600)
chmod 600 /etc/tars/secrets/aes.key

# Verify key is valid hex
python -c "
key = open('/etc/tars/secrets/aes.key').read().strip()
assert len(key) == 64, f'Key should be 64 hex chars, got {len(key)}'
assert all(c in '0123456789abcdef' for c in key.lower()), 'Invalid hex'
print('✓ Key valid')
"
```

**Problem:** Signature verification fails

```bash
# Verify public key matches private key
openssl rsa -in /etc/tars/secrets/rsa.key -pubout | \
  diff - /etc/tars/secrets/rsa.pub

# Test signing and verification
python -c "
from security import SecurityManager
security = SecurityManager(
    signing_key_path='/etc/tars/secrets/rsa.key',
    public_key_path='/etc/tars/secrets/rsa.pub'
)
data = {'test': 'data'}
signed = security.sign_json(data)
assert security.verify_signature(signed), 'Verification failed'
print('✓ Signing works')
"
```

#### 6.4.4 API Issues

**Problem:** API authentication fails

```bash
# Test JWT authentication
curl -X POST http://localhost:8100/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "demo123"}'

# Test API key authentication
curl -X GET http://localhost:8100/ga \
  -H "X-API-Key: dev-key-admin"

# Verify JWT secret
python -c "
from enterprise_config import load_enterprise_config
config = load_enterprise_config(profile='prod')
print(f'JWT secret configured: {bool(config.api.jwt_secret)}')
"
```

**Problem:** Rate limiting too aggressive

```bash
# Check rate limit configuration
python -c "
from enterprise_config import load_enterprise_config
config = load_enterprise_config(profile='prod')
print(f'Rate limit: {config.api.rate_limit_requests} req/{config.api.rate_limit_window}s')
"

# Temporarily disable rate limiting (DANGER)
export TARS_API_RATE_LIMIT_ENABLED=false

# Or increase limits
export TARS_API_RATE_LIMIT_REQUESTS=100
```

---

## 7. Production Deployment Examples

### 7.1 Docker Deployment

**Dockerfile:**

```dockerfile
FROM python:3.11-slim

# Install dependencies
COPY requirements-dev.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements-dev.txt

# Copy application
COPY . /app
WORKDIR /app

# Create secrets directory
RUN mkdir -p /etc/tars/secrets /etc/tars/certs

# Run API server
CMD ["python", "scripts/run_api_server.py"]
```

**docker-compose.yml:**

```yaml
version: '3.8'

services:
  tars-api:
    build: .
    ports:
      - "8100:8100"
    environment:
      - TARS_PROFILE=prod
      - TARS_SECRETS_BACKEND=vault
      - TARS_SECRETS_VAULT_URL=http://vault:8200
      - TARS_SECRETS_VAULT_TOKEN=${VAULT_TOKEN}
    volumes:
      - ./enterprise_config/config:/app/enterprise_config/config:ro
      - tars-secrets:/etc/tars/secrets
      - tars-certs:/etc/tars/certs
    depends_on:
      - vault
      - prometheus

  vault:
    image: vault:1.15
    ports:
      - "8200:8200"
    environment:
      - VAULT_DEV_ROOT_TOKEN_ID=root
    cap_add:
      - IPC_LOCK

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./observability/prometheus.yml:/etc/prometheus/prometheus.yml:ro

volumes:
  tars-secrets:
  tars-certs:
```

### 7.2 Kubernetes Deployment

**Full Kubernetes manifests in `charts/tars/`**

```bash
# Deploy with Helm
helm install tars ./charts/tars \
  --namespace tars \
  --create-namespace \
  --set global.profile=prod \
  --set security.encryption.enabled=true \
  --set compliance.enabledStandards="{soc2,iso27001,gdpr}" \
  --set secrets.backend=vault \
  --set secrets.vault.url=https://vault.example.com
```

---

**End of Enterprise Hardening Guide**

Total: ~2,500 lines of comprehensive documentation covering all enterprise features, configuration, compliance, security, telemetry, and production deployment.
