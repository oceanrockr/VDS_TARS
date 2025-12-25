#!/usr/bin/env python3
"""
T.A.R.S. Configuration Loader Module

Provides unified configuration file support for the T.A.R.S. organization
governance pipeline. Supports YAML (if PyYAML available) and JSON formats.

Configuration Precedence:
    1. CLI --config <path> (explicit path)
    2. Environment variable TARS_CONFIG
    3. Default ./tars.yml or ./tars.yaml (if present)
    4. Default ./tars.json (if present)
    5. Fallback to CLI defaults

Config Namespace Structure:
    orchestrator:
        root: "./org-health"
        outdir: "./reports/runs"
        format: "flat"  # or "structured"
        print_paths: true
        fail_on_breach: false
        fail_on_critical: false
        with_narrative: false
        sla_policy: null
        windows: []
    packager:
        output_dir: "./release/executive"
        bundle_name_template: "tars-exec-bundle-{version}-{timestamp}"
        tar: false
        zip: true
        checksums: true
        manifest: true
        compliance_index: true
        signing:
            enabled: false
            gpg_key_id: null
    retention:
        enabled: false
        days_hot: 30
        days_warm: 90
        days_archive: 365
        compress_after: 30
    notify:
        enabled: false
        exit_codes: [92, 102, 122, 132, 142]
        webhook_url: null
        slack_webhook_url: null
        pagerduty_routing_key: null

Usage:
    from scripts.tars_config import TarsConfigLoader

    # Load config from CLI arg or auto-discovery
    loader = TarsConfigLoader(config_path=args.config)
    config = loader.load()

    # Access config values with defaults
    root_dir = config.get("orchestrator", {}).get("root", "./org-health")

Exit Codes:
    N/A - This is a library module, not a standalone script.

Version: 1.0.0
Phase: 18 - Ops Integrations, Config Management
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Try to import PyYAML
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    yaml = None


# Default config values
DEFAULT_CONFIG: Dict[str, Any] = {
    "orchestrator": {
        "root": "./org-health",
        "outdir": "./reports/runs",
        "format": "flat",
        "print_paths": False,
        "fail_on_breach": False,
        "fail_on_critical": False,
        "with_narrative": False,
        "sla_policy": None,
        "windows": [],
    },
    "packager": {
        "output_dir": "./release/executive",
        "bundle_name_template": "tars-exec-bundle-{version}-{timestamp}",
        "tar": False,
        "zip": True,
        "checksums": True,
        "manifest": True,
        "compliance_index": True,
        "signing": {
            "enabled": False,
            "gpg_key_id": None,
        },
    },
    "retention": {
        "enabled": False,
        "days_hot": 30,
        "days_warm": 90,
        "days_archive": 365,
        "compress_after": 30,
    },
    "notify": {
        "enabled": False,
        "exit_codes": [92, 102, 122, 132, 142],
        "webhook_url": None,
        "slack_webhook_url": None,
        "pagerduty_routing_key": None,
    },
}


class TarsConfigError(Exception):
    """Exception raised for configuration errors."""
    pass


class TarsConfigLoader:
    """
    Unified configuration loader for T.A.R.S. tools.

    Supports YAML (if PyYAML installed) and JSON formats.
    Falls back gracefully if config files are missing or malformed.
    """

    # Default config file names to search for
    DEFAULT_CONFIG_FILES = [
        "tars.yml",
        "tars.yaml",
        "tars.json",
    ]

    def __init__(
        self,
        config_path: Optional[str] = None,
        base_dir: Optional[str] = None,
        silent: bool = False
    ):
        """
        Initialize the config loader.

        Args:
            config_path: Explicit path to config file (highest priority)
            base_dir: Base directory for config file search (default: cwd)
            silent: If True, suppress warning logs for missing configs
        """
        self.config_path = config_path
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()
        self.silent = silent
        self._config: Optional[Dict[str, Any]] = None
        self._resolved_path: Optional[Path] = None

    def _find_config_file(self) -> Optional[Path]:
        """
        Find the configuration file based on precedence rules.

        Returns:
            Path to config file if found, None otherwise.
        """
        # 1. Explicit CLI path
        if self.config_path:
            path = Path(self.config_path)
            if path.exists():
                return path.resolve()
            if not self.silent:
                logger.warning(f"Specified config file not found: {self.config_path}")
            return None

        # 2. Environment variable
        env_config = os.environ.get("TARS_CONFIG")
        if env_config:
            path = Path(env_config)
            if path.exists():
                return path.resolve()
            if not self.silent:
                logger.warning(f"TARS_CONFIG path not found: {env_config}")

        # 3. Default file search
        for filename in self.DEFAULT_CONFIG_FILES:
            path = self.base_dir / filename
            if path.exists():
                return path.resolve()

        return None

    def _parse_yaml(self, content: str, filepath: Path) -> Dict[str, Any]:
        """Parse YAML content."""
        if not YAML_AVAILABLE:
            raise TarsConfigError(
                f"YAML config file found ({filepath}) but PyYAML is not installed. "
                "Install with: pip install pyyaml"
            )
        try:
            return yaml.safe_load(content) or {}
        except yaml.YAMLError as e:
            raise TarsConfigError(f"YAML parse error in {filepath}: {e}")

    def _parse_json(self, content: str, filepath: Path) -> Dict[str, Any]:
        """Parse JSON content."""
        try:
            return json.loads(content) or {}
        except json.JSONDecodeError as e:
            raise TarsConfigError(f"JSON parse error in {filepath}: {e}")

    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deep merge two dictionaries.

        Override values take precedence. Nested dicts are merged recursively.
        """
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    def _validate_config(self, config: Dict[str, Any]) -> None:
        """
        Validate configuration structure.

        Performs light validation to catch obvious issues.
        """
        # Check for unknown top-level keys (warn but don't fail)
        known_keys = set(DEFAULT_CONFIG.keys())
        for key in config.keys():
            if key not in known_keys:
                if not self.silent:
                    logger.warning(f"Unknown config key (ignored): {key}")

        # Validate specific values
        orchestrator = config.get("orchestrator", {})
        if orchestrator.get("format") not in [None, "flat", "structured"]:
            raise TarsConfigError(
                f"Invalid orchestrator.format: {orchestrator.get('format')}. "
                "Must be 'flat' or 'structured'."
            )

        retention = config.get("retention", {})
        for key in ["days_hot", "days_warm", "days_archive", "compress_after"]:
            val = retention.get(key)
            if val is not None and (not isinstance(val, int) or val < 0):
                raise TarsConfigError(
                    f"retention.{key} must be a non-negative integer, got: {val}"
                )

    def load(self) -> Dict[str, Any]:
        """
        Load and return the configuration.

        Returns:
            Merged configuration dictionary (defaults + file config).

        Note:
            If config file is not found or parsing fails, returns defaults
            with a warning (does not raise exception).
        """
        if self._config is not None:
            return self._config

        # Start with defaults
        config = self._deep_merge({}, DEFAULT_CONFIG)

        # Find config file
        config_file = self._find_config_file()
        if config_file:
            self._resolved_path = config_file
            try:
                content = config_file.read_text(encoding="utf-8")

                # Parse based on extension
                if config_file.suffix in [".yml", ".yaml"]:
                    file_config = self._parse_yaml(content, config_file)
                elif config_file.suffix == ".json":
                    file_config = self._parse_json(content, config_file)
                else:
                    # Try YAML first, then JSON
                    try:
                        if YAML_AVAILABLE:
                            file_config = self._parse_yaml(content, config_file)
                        else:
                            file_config = self._parse_json(content, config_file)
                    except TarsConfigError:
                        file_config = self._parse_json(content, config_file)

                # Validate and merge
                self._validate_config(file_config)
                config = self._deep_merge(config, file_config)

                if not self.silent:
                    logger.info(f"Loaded config from: {config_file}")

            except TarsConfigError as e:
                if not self.silent:
                    logger.warning(f"Config parse failed, using defaults: {e}")
            except Exception as e:
                if not self.silent:
                    logger.warning(f"Config load failed, using defaults: {e}")
        else:
            if not self.silent:
                logger.debug("No config file found, using defaults")

        self._config = config
        return config

    def get(self, *keys: str, default: Any = None) -> Any:
        """
        Get a nested config value.

        Args:
            *keys: Sequence of keys to traverse (e.g., "orchestrator", "root")
            default: Default value if key not found

        Returns:
            Config value or default.
        """
        config = self.load()
        for key in keys:
            if isinstance(config, dict) and key in config:
                config = config[key]
            else:
                return default
        return config

    def get_orchestrator_config(self) -> Dict[str, Any]:
        """Get the orchestrator configuration section."""
        return self.load().get("orchestrator", {})

    def get_packager_config(self) -> Dict[str, Any]:
        """Get the packager configuration section."""
        return self.load().get("packager", {})

    def get_retention_config(self) -> Dict[str, Any]:
        """Get the retention configuration section."""
        return self.load().get("retention", {})

    def get_notify_config(self) -> Dict[str, Any]:
        """Get the notification configuration section."""
        return self.load().get("notify", {})

    @property
    def resolved_path(self) -> Optional[Path]:
        """Return the resolved config file path, if any."""
        return self._resolved_path

    @property
    def yaml_available(self) -> bool:
        """Return True if YAML parsing is available."""
        return YAML_AVAILABLE

    def __repr__(self) -> str:
        path_str = str(self._resolved_path) if self._resolved_path else "None"
        return f"TarsConfigLoader(path={path_str}, yaml={YAML_AVAILABLE})"


def get_config_from_args(
    config_arg: Optional[str] = None,
    silent: bool = False
) -> Dict[str, Any]:
    """
    Convenience function to load config from CLI args.

    Args:
        config_arg: Value of --config CLI argument (may be None)
        silent: Suppress warning logs

    Returns:
        Loaded configuration dictionary.
    """
    loader = TarsConfigLoader(config_path=config_arg, silent=silent)
    return loader.load()


def merge_cli_with_config(
    cli_args: Dict[str, Any],
    config: Dict[str, Any],
    namespace: str
) -> Dict[str, Any]:
    """
    Merge CLI arguments with config file values.

    CLI args take precedence over config file values.
    Only non-None CLI args override config values.

    Args:
        cli_args: Dictionary of CLI argument values
        config: Loaded config dictionary
        namespace: Config namespace to use (e.g., "orchestrator")

    Returns:
        Merged configuration for the namespace.
    """
    base_config = config.get(namespace, {})
    result = base_config.copy()

    for key, value in cli_args.items():
        if value is not None:
            result[key] = value

    return result


# Module-level convenience
_default_loader: Optional[TarsConfigLoader] = None


def get_default_loader() -> TarsConfigLoader:
    """Get or create the default config loader."""
    global _default_loader
    if _default_loader is None:
        _default_loader = TarsConfigLoader(silent=True)
    return _default_loader


def get_config() -> Dict[str, Any]:
    """Get configuration using the default loader."""
    return get_default_loader().load()


if __name__ == "__main__":
    # Simple test/demo when run directly
    import sys

    print("T.A.R.S. Configuration Loader")
    print("=" * 40)
    print(f"YAML available: {YAML_AVAILABLE}")
    print()

    loader = TarsConfigLoader()
    config = loader.load()

    print(f"Resolved path: {loader.resolved_path}")
    print()
    print("Loaded configuration:")
    print(json.dumps(config, indent=2, default=str))
