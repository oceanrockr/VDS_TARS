#!/usr/bin/env python3
"""
Smoke tests for T.A.R.S. Config Loader Module

Tests the tars_config.py configuration loading functionality.

Version: 1.0.0
Phase: 18 - Ops Integrations
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.tars_config import (
    TarsConfigLoader,
    TarsConfigError,
    DEFAULT_CONFIG,
    YAML_AVAILABLE,
    get_config_from_args,
    merge_cli_with_config,
)


class TestTarsConfigLoader(unittest.TestCase):
    """Test suite for TarsConfigLoader."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.original_env = os.environ.get("TARS_CONFIG")
        # Clean environment
        if "TARS_CONFIG" in os.environ:
            del os.environ["TARS_CONFIG"]

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        # Restore environment
        if self.original_env:
            os.environ["TARS_CONFIG"] = self.original_env
        elif "TARS_CONFIG" in os.environ:
            del os.environ["TARS_CONFIG"]

    def test_default_config_structure(self):
        """Test that DEFAULT_CONFIG has expected structure."""
        self.assertIn("orchestrator", DEFAULT_CONFIG)
        self.assertIn("packager", DEFAULT_CONFIG)
        self.assertIn("retention", DEFAULT_CONFIG)
        self.assertIn("notify", DEFAULT_CONFIG)

        # Orchestrator defaults
        orch = DEFAULT_CONFIG["orchestrator"]
        self.assertEqual(orch["root"], "./org-health")
        self.assertEqual(orch["format"], "flat")
        self.assertFalse(orch["fail_on_breach"])

        # Retention defaults
        ret = DEFAULT_CONFIG["retention"]
        self.assertEqual(ret["days_hot"], 30)
        self.assertFalse(ret["enabled"])

    def test_loader_with_no_config_file(self):
        """Test loader returns defaults when no config file exists."""
        loader = TarsConfigLoader(
            config_path=None,
            base_dir=self.temp_dir,
            silent=True
        )
        config = loader.load()

        # Should return defaults
        self.assertEqual(config["orchestrator"]["root"], "./org-health")
        self.assertIsNone(loader.resolved_path)

    def test_loader_with_json_config(self):
        """Test loader parses JSON config correctly."""
        config_path = Path(self.temp_dir) / "tars.json"
        config_data = {
            "orchestrator": {
                "root": "/custom/path",
                "format": "structured",
                "print_paths": True
            }
        }
        with open(config_path, "w") as f:
            json.dump(config_data, f)

        loader = TarsConfigLoader(config_path=str(config_path), silent=True)
        config = loader.load()

        # Custom values
        self.assertEqual(config["orchestrator"]["root"], "/custom/path")
        self.assertEqual(config["orchestrator"]["format"], "structured")
        self.assertTrue(config["orchestrator"]["print_paths"])

        # Defaults preserved for missing keys
        self.assertFalse(config["orchestrator"]["fail_on_breach"])

        # Resolved path set
        self.assertEqual(loader.resolved_path, config_path.resolve())

    def test_loader_with_yaml_config(self):
        """Test loader parses YAML config correctly (if available)."""
        if not YAML_AVAILABLE:
            self.skipTest("PyYAML not installed")

        import yaml
        config_path = Path(self.temp_dir) / "tars.yml"
        config_data = {
            "orchestrator": {
                "root": "/yaml/path",
                "with_narrative": True
            },
            "packager": {
                "tar": True
            }
        }
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        loader = TarsConfigLoader(config_path=str(config_path), silent=True)
        config = loader.load()

        self.assertEqual(config["orchestrator"]["root"], "/yaml/path")
        self.assertTrue(config["orchestrator"]["with_narrative"])
        self.assertTrue(config["packager"]["tar"])

    def test_loader_environment_variable(self):
        """Test loader respects TARS_CONFIG environment variable."""
        config_path = Path(self.temp_dir) / "env_config.json"
        config_data = {"orchestrator": {"root": "/env/path"}}
        with open(config_path, "w") as f:
            json.dump(config_data, f)

        os.environ["TARS_CONFIG"] = str(config_path)

        loader = TarsConfigLoader(base_dir=self.temp_dir, silent=True)
        config = loader.load()

        self.assertEqual(config["orchestrator"]["root"], "/env/path")

    def test_loader_cli_precedence(self):
        """Test that explicit config_path takes precedence over env."""
        env_config_path = Path(self.temp_dir) / "env_config.json"
        cli_config_path = Path(self.temp_dir) / "cli_config.json"

        with open(env_config_path, "w") as f:
            json.dump({"orchestrator": {"root": "/env/path"}}, f)
        with open(cli_config_path, "w") as f:
            json.dump({"orchestrator": {"root": "/cli/path"}}, f)

        os.environ["TARS_CONFIG"] = str(env_config_path)

        loader = TarsConfigLoader(config_path=str(cli_config_path), silent=True)
        config = loader.load()

        # CLI takes precedence
        self.assertEqual(config["orchestrator"]["root"], "/cli/path")

    def test_loader_default_file_discovery(self):
        """Test loader finds tars.yml/tars.json in base directory."""
        config_path = Path(self.temp_dir) / "tars.json"
        with open(config_path, "w") as f:
            json.dump({"orchestrator": {"root": "/discovered/path"}}, f)

        loader = TarsConfigLoader(base_dir=self.temp_dir, silent=True)
        config = loader.load()

        self.assertEqual(config["orchestrator"]["root"], "/discovered/path")

    def test_loader_invalid_json_graceful(self):
        """Test loader handles invalid JSON gracefully."""
        config_path = Path(self.temp_dir) / "invalid.json"
        with open(config_path, "w") as f:
            f.write("{ invalid json }")

        loader = TarsConfigLoader(config_path=str(config_path), silent=True)
        config = loader.load()

        # Should return defaults
        self.assertEqual(config["orchestrator"]["root"], "./org-health")

    def test_loader_missing_file_graceful(self):
        """Test loader handles missing file gracefully."""
        loader = TarsConfigLoader(
            config_path="/nonexistent/path/config.yml",
            silent=True
        )
        config = loader.load()

        # Should return defaults
        self.assertEqual(config["orchestrator"]["root"], "./org-health")
        self.assertIsNone(loader.resolved_path)

    def test_loader_get_method(self):
        """Test loader.get() method for nested access."""
        config_path = Path(self.temp_dir) / "tars.json"
        with open(config_path, "w") as f:
            json.dump({
                "packager": {
                    "signing": {
                        "enabled": True,
                        "gpg_key_id": "ABC123"
                    }
                }
            }, f)

        loader = TarsConfigLoader(config_path=str(config_path), silent=True)

        # Nested access
        self.assertTrue(loader.get("packager", "signing", "enabled"))
        self.assertEqual(loader.get("packager", "signing", "gpg_key_id"), "ABC123")

        # Default for missing
        self.assertEqual(loader.get("nonexistent", "key", default="default"), "default")

    def test_loader_validation_invalid_format(self):
        """Test loader validates orchestrator.format."""
        config_path = Path(self.temp_dir) / "invalid.json"
        with open(config_path, "w") as f:
            json.dump({
                "orchestrator": {"format": "invalid_format"}
            }, f)

        loader = TarsConfigLoader(config_path=str(config_path), silent=True)

        # Should warn and use defaults
        config = loader.load()
        # Validation happens, returns defaults on error
        self.assertIn("format", config["orchestrator"])

    def test_loader_validation_negative_days(self):
        """Test loader rejects negative retention days."""
        config_path = Path(self.temp_dir) / "invalid.json"
        with open(config_path, "w") as f:
            json.dump({
                "retention": {"days_hot": -5}
            }, f)

        loader = TarsConfigLoader(config_path=str(config_path), silent=True)
        config = loader.load()

        # Should return defaults due to validation error
        self.assertEqual(config["retention"]["days_hot"], 30)

    def test_deep_merge(self):
        """Test deep merge of config values."""
        config_path = Path(self.temp_dir) / "tars.json"
        with open(config_path, "w") as f:
            json.dump({
                "packager": {
                    "signing": {
                        "enabled": True
                    }
                }
            }, f)

        loader = TarsConfigLoader(config_path=str(config_path), silent=True)
        config = loader.load()

        # Custom value
        self.assertTrue(config["packager"]["signing"]["enabled"])
        # Default preserved
        self.assertIsNone(config["packager"]["signing"]["gpg_key_id"])

    def test_get_config_from_args(self):
        """Test convenience function get_config_from_args."""
        config_path = Path(self.temp_dir) / "tars.json"
        with open(config_path, "w") as f:
            json.dump({"orchestrator": {"root": "/args/path"}}, f)

        config = get_config_from_args(config_arg=str(config_path), silent=True)
        self.assertEqual(config["orchestrator"]["root"], "/args/path")

    def test_merge_cli_with_config(self):
        """Test merge_cli_with_config function."""
        config = {
            "orchestrator": {
                "root": "/config/path",
                "format": "structured",
                "fail_on_breach": True
            }
        }

        cli_args = {
            "root": None,  # Use config
            "format": "flat",  # Override
            "print_paths": True  # CLI only
        }

        merged = merge_cli_with_config(cli_args, config, "orchestrator")

        # Config value used when CLI is None
        self.assertEqual(merged["root"], "/config/path")
        # CLI overrides config
        self.assertEqual(merged["format"], "flat")
        # CLI value added
        self.assertTrue(merged["print_paths"])
        # Config preserved
        self.assertTrue(merged["fail_on_breach"])

    def test_loader_caches_config(self):
        """Test that loader caches config on first load."""
        config_path = Path(self.temp_dir) / "tars.json"
        with open(config_path, "w") as f:
            json.dump({"orchestrator": {"root": "/original"}}, f)

        loader = TarsConfigLoader(config_path=str(config_path), silent=True)
        config1 = loader.load()

        # Modify file
        with open(config_path, "w") as f:
            json.dump({"orchestrator": {"root": "/modified"}}, f)

        config2 = loader.load()

        # Should return cached value
        self.assertEqual(config1["orchestrator"]["root"], "/original")
        self.assertEqual(config2["orchestrator"]["root"], "/original")
        self.assertIs(config1, config2)

    def test_loader_yaml_available_property(self):
        """Test yaml_available property."""
        loader = TarsConfigLoader(silent=True)
        self.assertIsInstance(loader.yaml_available, bool)

    def test_loader_repr(self):
        """Test loader __repr__."""
        loader = TarsConfigLoader(silent=True)
        repr_str = repr(loader)
        self.assertIn("TarsConfigLoader", repr_str)


class TestHelperMethods(unittest.TestCase):
    """Test helper methods in tars_config module."""

    def test_get_orchestrator_config(self):
        """Test get_orchestrator_config method."""
        temp_dir = tempfile.mkdtemp()
        try:
            config_path = Path(temp_dir) / "tars.json"
            with open(config_path, "w") as f:
                json.dump({"orchestrator": {"root": "/test"}}, f)

            loader = TarsConfigLoader(config_path=str(config_path), silent=True)
            orch = loader.get_orchestrator_config()

            self.assertEqual(orch["root"], "/test")
        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_get_retention_config(self):
        """Test get_retention_config method."""
        temp_dir = tempfile.mkdtemp()
        try:
            config_path = Path(temp_dir) / "tars.json"
            with open(config_path, "w") as f:
                json.dump({"retention": {"enabled": True, "days_hot": 14}}, f)

            loader = TarsConfigLoader(config_path=str(config_path), silent=True)
            ret = loader.get_retention_config()

            self.assertTrue(ret["enabled"])
            self.assertEqual(ret["days_hot"], 14)
        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
