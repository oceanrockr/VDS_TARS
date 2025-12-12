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

__all__ = [
    "ConfigLoader",
    "load_config",
    "EnterpriseConfig",
    "SecurityConfig",
    "ComplianceConfig",
    "ObservabilityConfig",
    "APIConfig",
    "TelemetryConfig",
]

__version__ = "1.0.2-dev"
