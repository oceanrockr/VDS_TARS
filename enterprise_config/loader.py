"""
Configuration Loader for T.A.R.S. Enterprise Configuration

Implements precedence: CLI args > Environment variables > Config files > Defaults

Supports:
- YAML and JSON configuration files
- Environment variable mapping (TARS_*)
- Default profile loading (local/dev/staging/prod)
- Schema validation via Pydantic
- Config merging with deep dictionary updates
"""

import os
import json
import yaml
from typing import Optional, Dict, Any, List
from pathlib import Path
from copy import deepcopy

from .schema import EnterpriseConfig, Environment


class ConfigLoader:
    """
    Enterprise configuration loader with multi-source precedence.

    Precedence order (highest to lowest):
    1. CLI arguments (passed as overrides)
    2. Environment variables (TARS_*)
    3. Config file (YAML/JSON)
    4. Default profile
    5. Schema defaults
    """

    ENV_PREFIX = "TARS_"

    def __init__(
        self,
        config_file: Optional[Path] = None,
        environment: Optional[Environment] = None,
        overrides: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize configuration loader.

        Args:
            config_file: Path to YAML/JSON config file
            environment: Target environment (local/dev/staging/prod)
            overrides: CLI argument overrides (highest precedence)
        """
        self.config_file = config_file
        self.environment = environment or self._detect_environment()
        self.overrides = overrides or {}

    def load(self) -> EnterpriseConfig:
        """
        Load configuration with full precedence chain.

        Returns:
            Validated EnterpriseConfig instance
        """
        # Start with default profile
        config_dict = self._load_default_profile(self.environment)

        # Merge config file (if provided)
        if self.config_file:
            file_config = self._load_config_file(self.config_file)
            config_dict = self._deep_merge(config_dict, file_config)

        # Merge environment variables
        env_config = self._load_from_environment()
        config_dict = self._deep_merge(config_dict, env_config)

        # Apply CLI overrides (highest precedence)
        config_dict = self._deep_merge(config_dict, self.overrides)

        # Set environment in config
        config_dict["environment"] = self.environment

        # Validate and return
        return EnterpriseConfig(**config_dict)

    def _detect_environment(self) -> Environment:
        """Detect environment from TARS_ENVIRONMENT or default to LOCAL."""
        env_str = os.getenv(f"{self.ENV_PREFIX}ENVIRONMENT", "local").lower()
        try:
            return Environment(env_str)
        except ValueError:
            return Environment.LOCAL

    def _load_default_profile(self, environment: Environment) -> Dict[str, Any]:
        """
        Load default configuration profile for environment.

        Args:
            environment: Target environment

        Returns:
            Default configuration dictionary
        """
        defaults_dir = Path(__file__).parent / "defaults"
        profile_file = defaults_dir / f"{environment.value}.yaml"

        if profile_file.exists():
            return self._load_yaml_file(profile_file)
        else:
            # Return minimal defaults if profile doesn't exist
            return self._get_minimal_defaults()

    def _get_minimal_defaults(self) -> Dict[str, Any]:
        """Return minimal default configuration."""
        return {
            "security": {
                "secrets_backend": "env",
                "enable_encryption": True,
                "enable_signing": True,
                "enable_input_sanitization": True,
                "enable_redaction": True,
            },
            "compliance": {
                "enabled_standards": [],
                "enable_audit_trail": True,
            },
            "observability": {
                "prometheus_url": "http://localhost:9090",
                "output_dir": "output",
                "collection_interval_seconds": 60,
                "anomaly_z_threshold": 3.0,
                "ewma_alpha": 0.3,
            },
            "api": {
                "enabled": False,
                "host": "0.0.0.0",
                "port": 8100,
                "auth_mode": "api_key",
                "enable_cors": True,
                "enable_rate_limiting": True,
            },
            "telemetry": {
                "enabled": True,
                "enable_prometheus_metrics": True,
                "log_level": "INFO",
                "log_format": "json",
                "track_cli_commands": True,
                "enable_error_tracking": True,
            },
            "enable_sbom_generation": True,
            "sbom_formats": ["cyclonedx", "spdx"],
            "enable_slsa_provenance": True,
            "slsa_level": 2,
        }

    def _load_config_file(self, config_file: Path) -> Dict[str, Any]:
        """
        Load configuration from YAML or JSON file.

        Args:
            config_file: Path to config file

        Returns:
            Configuration dictionary
        """
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")

        suffix = config_file.suffix.lower()

        if suffix in [".yaml", ".yml"]:
            return self._load_yaml_file(config_file)
        elif suffix == ".json":
            return self._load_json_file(config_file)
        else:
            raise ValueError(
                f"Unsupported config file format: {suffix}. "
                "Use .yaml, .yml, or .json"
            )

    def _load_yaml_file(self, file_path: Path) -> Dict[str, Any]:
        """Load YAML configuration file."""
        with open(file_path, "r") as f:
            return yaml.safe_load(f) or {}

    def _load_json_file(self, file_path: Path) -> Dict[str, Any]:
        """Load JSON configuration file."""
        with open(file_path, "r") as f:
            return json.load(f)

    def _load_from_environment(self) -> Dict[str, Any]:
        """
        Load configuration from environment variables.

        Environment variable mapping:
        - TARS_ENVIRONMENT -> environment
        - TARS_SECURITY_SECRETS_BACKEND -> security.secrets_backend
        - TARS_API_ENABLED -> api.enabled
        - TARS_API_PORT -> api.port
        - etc.

        Returns:
            Configuration dictionary from environment variables
        """
        config_dict: Dict[str, Any] = {}

        for key, value in os.environ.items():
            if not key.startswith(self.ENV_PREFIX):
                continue

            # Remove prefix and convert to lowercase
            key_parts = key[len(self.ENV_PREFIX):].lower().split("_")

            # Build nested dictionary
            current = config_dict
            for part in key_parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]

            # Set the value (convert types)
            final_key = key_parts[-1]
            current[final_key] = self._convert_env_value(value)

        return config_dict

    def _convert_env_value(self, value: str) -> Any:
        """
        Convert environment variable string to appropriate type.

        Args:
            value: Environment variable string value

        Returns:
            Converted value (bool, int, float, list, or str)
        """
        # Boolean
        if value.lower() in ["true", "false"]:
            return value.lower() == "true"

        # Integer
        try:
            return int(value)
        except ValueError:
            pass

        # Float
        try:
            return float(value)
        except ValueError:
            pass

        # JSON list/dict
        if value.startswith("[") or value.startswith("{"):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                pass

        # Comma-separated list
        if "," in value:
            return [v.strip() for v in value.split(",")]

        # String
        return value

    def _deep_merge(
        self,
        base: Dict[str, Any],
        override: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Deep merge two dictionaries.

        Args:
            base: Base dictionary
            override: Override dictionary (takes precedence)

        Returns:
            Merged dictionary
        """
        result = deepcopy(base)

        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = deepcopy(value)

        return result


def load_config(
    config_file: Optional[Path] = None,
    environment: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> EnterpriseConfig:
    """
    Convenience function to load enterprise configuration.

    Args:
        config_file: Path to YAML/JSON config file
        environment: Target environment (local/dev/staging/prod)
        overrides: CLI argument overrides

    Returns:
        Validated EnterpriseConfig instance

    Example:
        >>> config = load_config(
        ...     config_file=Path("config/prod.yaml"),
        ...     environment="prod",
        ...     overrides={"api": {"port": 8200}}
        ... )
    """
    env = Environment(environment) if environment else None
    loader = ConfigLoader(
        config_file=config_file,
        environment=env,
        overrides=overrides,
    )
    return loader.load()


def load_config_from_cli_args(
    config_file: Optional[str] = None,
    environment: Optional[str] = None,
    prometheus_url: Optional[str] = None,
    output_dir: Optional[str] = None,
    api_enabled: Optional[bool] = None,
    api_port: Optional[int] = None,
    **kwargs
) -> EnterpriseConfig:
    """
    Load configuration from common CLI arguments.

    This helper function maps common CLI arguments to configuration overrides.

    Args:
        config_file: Path to config file
        environment: Target environment
        prometheus_url: Prometheus URL override
        output_dir: Output directory override
        api_enabled: Enable API override
        api_port: API port override
        **kwargs: Additional overrides

    Returns:
        Validated EnterpriseConfig instance
    """
    overrides: Dict[str, Any] = {}

    # Map CLI args to config structure
    if prometheus_url:
        overrides.setdefault("observability", {})["prometheus_url"] = prometheus_url

    if output_dir:
        overrides.setdefault("observability", {})["output_dir"] = output_dir

    if api_enabled is not None:
        overrides.setdefault("api", {})["enabled"] = api_enabled

    if api_port is not None:
        overrides.setdefault("api", {})["port"] = api_port

    # Merge any additional kwargs
    overrides = {**overrides, **kwargs}

    # Load configuration
    config_path = Path(config_file) if config_file else None
    return load_config(
        config_file=config_path,
        environment=environment,
        overrides=overrides,
    )
