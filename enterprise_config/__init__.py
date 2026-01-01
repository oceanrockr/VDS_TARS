"""
Enterprise Configuration System for T.A.R.S. v1.0.2

Provides unified configuration loading with:
- Environment variable mapping
- YAML & JSON config file support
- Precedence rules (CLI > env > file > defaults)
- Schema validation via Pydantic
- Default profiles (local/dev/staging/prod)
"""

from .loader import ConfigLoader, load_config
from .schema import (
    EnterpriseConfig,
    SecurityConfig,
    ComplianceConfig,
    ObservabilityConfig,
    APIConfig,
    TelemetryConfig,
)


class EnterpriseConfigError(Exception):
    """
    Exception raised for enterprise configuration errors.

    Includes:
    - Invalid configuration values
    - Missing required fields
    - Schema validation failures
    - File loading errors
    """

    def __init__(self, message: str, field: str = None, value=None):
        self.field = field
        self.value = value
        super().__init__(message)


# Alias for backward compatibility
SecretsConfig = SecurityConfig
load_enterprise_config = load_config


__all__ = [
    "ConfigLoader",
    "load_config",
    "load_enterprise_config",
    "EnterpriseConfig",
    "SecurityConfig",
    "SecretsConfig",
    "ComplianceConfig",
    "ObservabilityConfig",
    "APIConfig",
    "TelemetryConfig",
    "EnterpriseConfigError",
]

__version__ = "1.0.2-dev"
