"""
Configuration Schema Definitions for T.A.R.S. Enterprise Configuration

Uses Pydantic for validation and type safety.
"""

from typing import Optional, List, Literal, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field, validator, SecretStr
from pathlib import Path


class Environment(str, Enum):
    """Deployment environment."""
    LOCAL = "local"
    DEV = "dev"
    STAGING = "staging"
    PROD = "prod"


class SecretsBackend(str, Enum):
    """Supported secrets management backends."""
    ENV = "env"
    FILE = "file"
    VAULT = "vault"
    AWS_SECRETS_MANAGER = "aws_secrets_manager"
    GCP_SECRET_MANAGER = "gcp_secret_manager"


class ComplianceStandard(str, Enum):
    """Supported compliance standards."""
    SOC2 = "soc2"
    ISO27001 = "iso27001"
    NIST_800_53 = "nist_800_53"
    GDPR = "gdpr"
    HIPAA = "hipaa"
    FAA_FCC = "faa_fcc"


class RBACRole(str, Enum):
    """RBAC roles for API access."""
    READONLY = "readonly"
    SRE = "sre"
    ADMIN = "admin"


class AuthMode(str, Enum):
    """Authentication modes for API."""
    TOKEN = "token"
    JWT = "jwt"
    API_KEY = "api_key"


class SecurityConfig(BaseModel):
    """Security configuration."""

    # Secrets management
    secrets_backend: SecretsBackend = Field(
        default=SecretsBackend.ENV,
        description="Backend for secrets management"
    )
    vault_url: Optional[str] = Field(
        default=None,
        description="HashiCorp Vault URL (if backend=vault)"
    )
    vault_token: Optional[SecretStr] = Field(
        default=None,
        description="Vault authentication token"
    )
    aws_region: Optional[str] = Field(
        default=None,
        description="AWS region for Secrets Manager"
    )
    gcp_project_id: Optional[str] = Field(
        default=None,
        description="GCP project ID for Secret Manager"
    )

    # Encryption
    enable_encryption: bool = Field(
        default=True,
        description="Enable AES-256 encryption for reports"
    )
    encryption_key_path: Optional[Path] = Field(
        default=None,
        description="Path to AES encryption key file"
    )
    enable_pgp: bool = Field(
        default=False,
        description="Enable PGP encryption"
    )
    pgp_public_key_path: Optional[Path] = Field(
        default=None,
        description="Path to PGP public key for encryption"
    )

    # Signing
    enable_signing: bool = Field(
        default=True,
        description="Enable cryptographic signing of reports"
    )
    signing_key_path: Optional[Path] = Field(
        default=None,
        description="Path to signing key (RSA private key)"
    )

    # Input sanitization
    enable_input_sanitization: bool = Field(
        default=True,
        description="Enable input sanitization on CLI parameters"
    )
    max_input_length: int = Field(
        default=10000,
        description="Maximum allowed input length for string parameters"
    )

    # Sensitive data redaction
    enable_redaction: bool = Field(
        default=True,
        description="Automatically redact sensitive fields in reports"
    )
    redaction_patterns: List[str] = Field(
        default_factory=lambda: [
            r"api[_-]?key",
            r"secret",
            r"password",
            r"token",
            r"credential",
            r"private[_-]?key",
        ],
        description="Regex patterns for sensitive field detection"
    )


class ComplianceConfig(BaseModel):
    """Compliance framework configuration."""

    enabled_standards: List[ComplianceStandard] = Field(
        default_factory=list,
        description="Enabled compliance standards"
    )

    # SOC 2
    soc2_controls_path: Optional[Path] = Field(
        default=Path("compliance/policies/standard_soc2.yaml"),
        description="Path to SOC 2 controls definition"
    )

    # ISO 27001
    iso27001_controls_path: Optional[Path] = Field(
        default=Path("compliance/policies/standard_iso27001.yaml"),
        description="Path to ISO 27001 controls definition"
    )

    # NIST 800-53
    nist_controls_path: Optional[Path] = Field(
        default=Path("compliance/policies/standard_nist_800_53.yaml"),
        description="Path to NIST 800-53 controls definition"
    )

    # GDPR
    gdpr_retention_days: int = Field(
        default=90,
        description="GDPR metric retention period (days)"
    )
    gdpr_enforce_deletion: bool = Field(
        default=False,
        description="Enforce automatic deletion after retention period"
    )

    # FAA/FCC logs immutability
    enable_log_immutability: bool = Field(
        default=False,
        description="Enable write-once log immutability (FAA/FCC mode)"
    )
    immutable_log_path: Optional[Path] = Field(
        default=None,
        description="Path to immutable log storage"
    )

    # Audit trail
    enable_audit_trail: bool = Field(
        default=True,
        description="Enable compliance audit trail"
    )
    audit_log_path: Path = Field(
        default=Path("logs/compliance_audit.jsonl"),
        description="Path to compliance audit log"
    )


class ObservabilityConfig(BaseModel):
    """Observability configuration (existing systems)."""

    prometheus_url: str = Field(
        default="http://localhost:9090",
        description="Prometheus metrics endpoint URL"
    )

    output_dir: Path = Field(
        default=Path("output"),
        description="Default output directory for reports"
    )

    ga_baseline_path: Optional[Path] = Field(
        default=None,
        description="Path to GA Day baseline KPI file"
    )

    # Metric collection
    collection_interval_seconds: int = Field(
        default=60,
        description="Metric collection interval (seconds)"
    )

    # Anomaly detection
    anomaly_z_threshold: float = Field(
        default=3.0,
        description="Z-score threshold for anomaly detection"
    )
    ewma_alpha: float = Field(
        default=0.3,
        description="EWMA smoothing factor (0-1)"
    )


class APIConfig(BaseModel):
    """Enterprise API configuration."""

    enabled: bool = Field(
        default=False,
        description="Enable Enterprise Observability API"
    )

    host: str = Field(
        default="0.0.0.0",
        description="API server bind address"
    )

    port: int = Field(
        default=8100,
        description="API server port"
    )

    # Authentication
    auth_mode: AuthMode = Field(
        default=AuthMode.API_KEY,
        description="Authentication mode"
    )

    api_keys: Dict[str, RBACRole] = Field(
        default_factory=dict,
        description="API keys mapped to roles (key -> role)"
    )

    jwt_secret: Optional[SecretStr] = Field(
        default=None,
        description="JWT signing secret (if auth_mode=jwt)"
    )
    jwt_algorithm: str = Field(
        default="HS256",
        description="JWT signing algorithm"
    )
    jwt_expiration_minutes: int = Field(
        default=60,
        description="JWT token expiration time (minutes)"
    )

    # TLS
    enable_tls: bool = Field(
        default=False,
        description="Enable TLS for API server"
    )
    tls_cert_path: Optional[Path] = Field(
        default=None,
        description="Path to TLS certificate"
    )
    tls_key_path: Optional[Path] = Field(
        default=None,
        description="Path to TLS private key"
    )

    # CORS
    enable_cors: bool = Field(
        default=True,
        description="Enable CORS"
    )
    cors_origins: List[str] = Field(
        default_factory=lambda: ["*"],
        description="Allowed CORS origins"
    )

    # Rate limiting
    enable_rate_limiting: bool = Field(
        default=True,
        description="Enable API rate limiting"
    )
    rate_limit_requests: int = Field(
        default=100,
        description="Maximum requests per window"
    )
    rate_limit_window_seconds: int = Field(
        default=60,
        description="Rate limit window (seconds)"
    )


class TelemetryConfig(BaseModel):
    """Internal telemetry configuration."""

    enabled: bool = Field(
        default=True,
        description="Enable internal telemetry"
    )

    # Prometheus metrics export
    enable_prometheus_metrics: bool = Field(
        default=True,
        description="Export Prometheus metrics for T.A.R.S. internals"
    )
    prometheus_port: int = Field(
        default=9101,
        description="Prometheus metrics exporter port"
    )

    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Logging level"
    )
    log_format: Literal["json", "text"] = Field(
        default="json",
        description="Log output format"
    )
    log_file: Optional[Path] = Field(
        default=Path("logs/tars_telemetry.log"),
        description="Log file path (None for stdout only)"
    )

    # Command tracking
    track_cli_commands: bool = Field(
        default=True,
        description="Track CLI command execution metrics"
    )

    # Error tracking
    enable_error_tracking: bool = Field(
        default=True,
        description="Enable error event tracking"
    )


class EnterpriseConfig(BaseModel):
    """Root enterprise configuration."""

    environment: Environment = Field(
        default=Environment.LOCAL,
        description="Deployment environment"
    )

    version: str = Field(
        default="1.0.2-dev",
        description="T.A.R.S. version"
    )

    # Sub-configurations
    security: SecurityConfig = Field(
        default_factory=SecurityConfig,
        description="Security configuration"
    )

    compliance: ComplianceConfig = Field(
        default_factory=ComplianceConfig,
        description="Compliance configuration"
    )

    observability: ObservabilityConfig = Field(
        default_factory=ObservabilityConfig,
        description="Observability configuration"
    )

    api: APIConfig = Field(
        default_factory=APIConfig,
        description="Enterprise API configuration"
    )

    telemetry: TelemetryConfig = Field(
        default_factory=TelemetryConfig,
        description="Internal telemetry configuration"
    )

    # SBOM & Provenance
    enable_sbom_generation: bool = Field(
        default=True,
        description="Generate SBOM (CycloneDX/SPDX) on build"
    )
    sbom_formats: List[Literal["cyclonedx", "spdx"]] = Field(
        default_factory=lambda: ["cyclonedx", "spdx"],
        description="SBOM output formats"
    )

    enable_slsa_provenance: bool = Field(
        default=True,
        description="Generate SLSA provenance metadata"
    )
    slsa_level: Literal[1, 2, 3] = Field(
        default=2,
        description="Target SLSA level (1-3)"
    )

    class Config:
        use_enum_values = True
        validate_assignment = True
        extra = "forbid"  # Reject unknown fields

    @validator("environment", pre=True)
    def normalize_environment(cls, v):
        """Normalize environment strings."""
        if isinstance(v, str):
            return v.lower()
        return v

    def to_dict(self) -> Dict[str, Any]:
        """Export configuration as dictionary."""
        return self.dict(exclude_none=True)

    def to_json(self) -> str:
        """Export configuration as JSON string."""
        return self.json(exclude_none=True, indent=2)
