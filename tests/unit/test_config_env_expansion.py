#!/usr/bin/env python3
"""
Unit Tests for Configuration Environment Variable Expansion

Phase 19 - Production Ops Maturity & CI Hardening

Tests the safe environment variable expansion functionality in the
T.A.R.S. configuration loader.

Test Coverage:
    - Single variable expansion (present/missing)
    - Multiple variables in one string
    - Nested dict/list traversal
    - Non-string values preserved
    - Silent mode suppresses warnings
    - Edge cases (empty strings, special chars)

Version: 1.0.0
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.tars_config import (
    ENV_VAR_PATTERN,
    TarsConfigLoader,
    expand_env_vars_in_config,
    expand_env_vars_in_string,
)


class TestEnvVarPattern(unittest.TestCase):
    """Tests for the environment variable regex pattern."""

    def test_matches_simple_var(self):
        """Should match ${VAR_NAME}."""
        match = ENV_VAR_PATTERN.search("${MY_VAR}")
        self.assertIsNotNone(match)
        self.assertEqual(match.group(1), "MY_VAR")

    def test_matches_underscore_prefix(self):
        """Should match $_VAR (underscore prefix)."""
        match = ENV_VAR_PATTERN.search("${_PRIVATE}")
        self.assertIsNotNone(match)
        self.assertEqual(match.group(1), "_PRIVATE")

    def test_matches_with_numbers(self):
        """Should match ${VAR123}."""
        match = ENV_VAR_PATTERN.search("${VAR123}")
        self.assertIsNotNone(match)
        self.assertEqual(match.group(1), "VAR123")

    def test_no_match_without_braces(self):
        """Should not match $VAR without braces."""
        match = ENV_VAR_PATTERN.search("$MY_VAR")
        self.assertIsNone(match)

    def test_no_match_number_prefix(self):
        """Should not match ${123VAR} (number prefix)."""
        match = ENV_VAR_PATTERN.search("${123VAR}")
        self.assertIsNone(match)


class TestExpandEnvVarsInString(unittest.TestCase):
    """Tests for expand_env_vars_in_string function."""

    def test_expand_present_variable(self):
        """Should expand variable that is set."""
        with patch.dict(os.environ, {"TEST_VAR": "test_value"}):
            result = expand_env_vars_in_string("${TEST_VAR}")
            self.assertEqual(result, "test_value")

    def test_preserve_missing_variable(self):
        """Should preserve literal for missing variable."""
        # Ensure variable is not set
        os.environ.pop("MISSING_VAR", None)
        result = expand_env_vars_in_string("${MISSING_VAR}", silent=True)
        self.assertEqual(result, "${MISSING_VAR}")

    def test_multiple_variables(self):
        """Should expand multiple variables in one string."""
        with patch.dict(os.environ, {"VAR_A": "alpha", "VAR_B": "beta"}):
            result = expand_env_vars_in_string("${VAR_A}/${VAR_B}/path")
            self.assertEqual(result, "alpha/beta/path")

    def test_mixed_present_missing(self):
        """Should handle mix of present and missing variables."""
        os.environ.pop("MISSING_VAR", None)
        with patch.dict(os.environ, {"PRESENT_VAR": "here"}):
            result = expand_env_vars_in_string("${PRESENT_VAR}/${MISSING_VAR}", silent=True)
            self.assertEqual(result, "here/${MISSING_VAR}")

    def test_no_variables_unchanged(self):
        """Should return string unchanged if no variables."""
        result = expand_env_vars_in_string("no variables here")
        self.assertEqual(result, "no variables here")

    def test_empty_string(self):
        """Should handle empty string."""
        result = expand_env_vars_in_string("")
        self.assertEqual(result, "")

    def test_non_string_passthrough(self):
        """Should return non-string as-is."""
        result = expand_env_vars_in_string(123)  # type: ignore
        self.assertEqual(result, 123)

    def test_dollar_without_brace_unchanged(self):
        """Should not expand $VAR without braces."""
        result = expand_env_vars_in_string("$VAR is not expanded")
        self.assertEqual(result, "$VAR is not expanded")

    def test_variable_value_with_special_chars(self):
        """Should expand variable with special characters in value."""
        with patch.dict(os.environ, {"SPECIAL_VAR": "value/with:special@chars"}):
            result = expand_env_vars_in_string("prefix/${SPECIAL_VAR}/suffix")
            self.assertEqual(result, "prefix/value/with:special@chars/suffix")


class TestExpandEnvVarsInConfig(unittest.TestCase):
    """Tests for expand_env_vars_in_config function."""

    def test_expand_in_flat_dict(self):
        """Should expand variables in flat dict."""
        with patch.dict(os.environ, {"TEST_VAR": "expanded"}):
            config = {"key": "${TEST_VAR}"}
            result = expand_env_vars_in_config(config)
            self.assertEqual(result["key"], "expanded")

    def test_expand_in_nested_dict(self):
        """Should expand variables in nested dict."""
        with patch.dict(os.environ, {"NESTED_VAR": "deep_value"}):
            config = {
                "outer": {
                    "inner": {
                        "value": "${NESTED_VAR}"
                    }
                }
            }
            result = expand_env_vars_in_config(config)
            self.assertEqual(result["outer"]["inner"]["value"], "deep_value")

    def test_expand_in_list(self):
        """Should expand variables in list items."""
        with patch.dict(os.environ, {"LIST_VAR": "item"}):
            config = {"items": ["${LIST_VAR}", "static"]}
            result = expand_env_vars_in_config(config)
            self.assertEqual(result["items"], ["item", "static"])

    def test_expand_in_nested_list(self):
        """Should expand variables in nested lists."""
        with patch.dict(os.environ, {"DEEP_VAR": "found"}):
            config = {"data": [["${DEEP_VAR}"]]}
            result = expand_env_vars_in_config(config)
            self.assertEqual(result["data"][0][0], "found")

    def test_preserve_non_string_values(self):
        """Should preserve int, float, bool, None."""
        config = {
            "int_val": 42,
            "float_val": 3.14,
            "bool_val": True,
            "null_val": None
        }
        result = expand_env_vars_in_config(config)
        self.assertEqual(result["int_val"], 42)
        self.assertEqual(result["float_val"], 3.14)
        self.assertEqual(result["bool_val"], True)
        self.assertIsNone(result["null_val"])

    def test_mixed_config(self):
        """Should handle complex mixed config."""
        with patch.dict(os.environ, {"URL": "https://example.com", "PORT": "8080"}):
            config = {
                "server": {
                    "url": "${URL}",
                    "port": 8080,  # Not a string, should stay as-is
                    "enabled": True
                },
                "endpoints": [
                    {"path": "${URL}/api", "timeout": 30}
                ]
            }
            result = expand_env_vars_in_config(config)
            self.assertEqual(result["server"]["url"], "https://example.com")
            self.assertEqual(result["server"]["port"], 8080)
            self.assertEqual(result["endpoints"][0]["path"], "https://example.com/api")


class TestTarsConfigLoaderEnvExpansion(unittest.TestCase):
    """Tests for TarsConfigLoader with environment variable expansion."""

    def test_load_expands_env_vars(self):
        """Should expand env vars when loading config file."""
        config_content = {
            "orchestrator": {
                "root": "${TARS_ROOT}/org-health"
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "tars.json"
            config_path.write_text(json.dumps(config_content))

            with patch.dict(os.environ, {"TARS_ROOT": "/data"}):
                loader = TarsConfigLoader(config_path=str(config_path), silent=True)
                config = loader.load()
                self.assertEqual(config["orchestrator"]["root"], "/data/org-health")

    def test_load_with_expand_env_disabled(self):
        """Should not expand when expand_env=False."""
        config_content = {
            "orchestrator": {
                "root": "${TARS_ROOT}/org-health"
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "tars.json"
            config_path.write_text(json.dumps(config_content))

            with patch.dict(os.environ, {"TARS_ROOT": "/data"}):
                loader = TarsConfigLoader(config_path=str(config_path), silent=True)
                # First load with expansion disabled
                loader._config = None  # Reset cache
                # Need to call internal method or create new loader
                loader2 = TarsConfigLoader(config_path=str(config_path), silent=True)
                config = loader2.load(expand_env=False)
                self.assertEqual(config["orchestrator"]["root"], "${TARS_ROOT}/org-health")

    def test_load_preserves_missing_vars_with_warning(self):
        """Should preserve missing vars and log warning."""
        config_content = {
            "notify": {
                "webhook_url": "${MISSING_WEBHOOK}"
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "tars.json"
            config_path.write_text(json.dumps(config_content))

            # Ensure variable is not set
            os.environ.pop("MISSING_WEBHOOK", None)

            loader = TarsConfigLoader(config_path=str(config_path), silent=True)
            config = loader.load()
            self.assertEqual(config["notify"]["webhook_url"], "${MISSING_WEBHOOK}")


class TestEdgeCases(unittest.TestCase):
    """Edge case tests for environment variable expansion."""

    def test_adjacent_variables(self):
        """Should expand adjacent variables."""
        with patch.dict(os.environ, {"A": "1", "B": "2"}):
            result = expand_env_vars_in_string("${A}${B}")
            self.assertEqual(result, "12")

    def test_empty_variable_value(self):
        """Should expand variable with empty value."""
        with patch.dict(os.environ, {"EMPTY_VAR": ""}):
            result = expand_env_vars_in_string("prefix/${EMPTY_VAR}/suffix")
            self.assertEqual(result, "prefix//suffix")

    def test_variable_in_variable_name_no_recursion(self):
        """Should not recursively expand (security)."""
        with patch.dict(os.environ, {"OUTER": "${INNER}", "INNER": "value"}):
            result = expand_env_vars_in_string("${OUTER}")
            # Should return "${INNER}" not "value" (no recursion)
            self.assertEqual(result, "${INNER}")

    def test_malformed_braces_unchanged(self):
        """Should not expand malformed patterns."""
        result = expand_env_vars_in_string("${}")
        self.assertEqual(result, "${}")

        result = expand_env_vars_in_string("${")
        self.assertEqual(result, "${")

        result = expand_env_vars_in_string("${}abc")
        self.assertEqual(result, "${}abc")


if __name__ == "__main__":
    unittest.main(verbosity=2)
