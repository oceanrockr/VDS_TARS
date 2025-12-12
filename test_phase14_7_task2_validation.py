#!/usr/bin/env python3
"""
Phase 14.7 Task 2 Validation Test Suite

Validates:
1. Production Runbook completeness
2. Performance Testing Suite functionality
3. Security Audit Tool functionality
4. Integration with prepare_release_artifacts.py

Author: T.A.R.S. Team
Version: 1.0.2
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple

# ============================================================================
# Configuration
# ============================================================================

PROJECT_ROOT = Path(__file__).parent
REQUIRED_FILES = {
    "runbook": PROJECT_ROOT / "docs" / "PRODUCTION_RUNBOOK.md",
    "perf_test": PROJECT_ROOT / "performance" / "run_performance_tests.py",
    "security_audit": PROJECT_ROOT / "security" / "security_audit.py",
    "release_script": PROJECT_ROOT / "scripts" / "prepare_release_artifacts.py",
}

# ============================================================================
# Test Utilities
# ============================================================================

def print_test_header(test_name: str):
    """Print test header"""
    print(f"\n{'='*80}")
    print(f"TEST: {test_name}")
    print(f"{'='*80}")


def print_result(passed: bool, message: str):
    """Print test result"""
    status = "[PASS]" if passed else "[FAIL]"
    print(f"{status} {message}")
    return passed


# ============================================================================
# Test 1: File Existence and Structure
# ============================================================================

def test_file_existence() -> bool:
    """Test that all required files exist"""
    print_test_header("File Existence")

    all_pass = True

    for name, path in REQUIRED_FILES.items():
        exists = path.exists()
        all_pass &= print_result(exists, f"{name}: {path}")

        if exists and path.suffix == '.py':
            # Check if Python file is valid
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Check shebang
                    has_shebang = content.startswith('#!/usr/bin/env python')
                    all_pass &= print_result(has_shebang, f"  - Has shebang")

                    # Check docstring
                    has_docstring = '"""' in content[:500]
                    all_pass &= print_result(has_docstring, f"  - Has docstring")

                    # Check main function
                    has_main = 'def main()' in content or 'if __name__ ==' in content
                    all_pass &= print_result(has_main, f"  - Has main function")

            except Exception as e:
                all_pass &= print_result(False, f"  - Error reading file: {e}")

    return all_pass


# ============================================================================
# Test 2: Production Runbook Validation
# ============================================================================

def test_production_runbook() -> bool:
    """Validate Production Runbook completeness"""
    print_test_header("Production Runbook Validation")

    runbook_path = REQUIRED_FILES["runbook"]

    if not runbook_path.exists():
        return print_result(False, "Runbook file not found")

    try:
        with open(runbook_path, 'r', encoding='utf-8') as f:
            content = f.read()

        all_pass = True

        # Check required sections
        required_sections = [
            "# T.A.R.S. Production Runbook",
            "## Overview",
            "## System Architecture",
            "## Deployment Models",
            "## Operations Guide",
            "## Maintenance Procedures",
            "## Incident Response Playbooks",
            "## Disaster Recovery",
            "## Diagnostics & Troubleshooting",
            "## Performance Benchmarks",
            "## Appendix"
        ]

        for section in required_sections:
            has_section = section in content
            all_pass &= print_result(has_section, f"Section: {section}")

        # Check for specific deployment models
        deployment_models = [
            "### Model 1: Local Development",
            "### Model 2: Docker Compose",
            "### Model 3: Kubernetes (Production)",
            "### Model 4: Air-Gapped Deployment",
            "### Model 5: Enterprise with Vault"
        ]

        for model in deployment_models:
            has_model = model in content
            all_pass &= print_result(has_model, f"Deployment: {model}")

        # Check for incident playbooks
        playbooks = [
            "### Playbook 1: High API Latency",
            "### Playbook 2: Elevated Anomaly Rate",
            "### Playbook 3: GA KPI Regression",
            "### Playbook 4: Security Incident",
            "### Playbook 5: Compliance Control Violation"
        ]

        for playbook in playbooks:
            has_playbook = playbook in content
            all_pass &= print_result(has_playbook, f"Playbook: {playbook}")

        # Check word count (should be comprehensive)
        word_count = len(content.split())
        min_words = 15000  # ~1,000-2,200 LOC should be 15k+ words
        has_sufficient_content = word_count >= min_words
        all_pass &= print_result(
            has_sufficient_content,
            f"Word count: {word_count:,} (minimum {min_words:,})"
        )

        # Check for code examples
        has_code_blocks = content.count('```') >= 20
        all_pass &= print_result(has_code_blocks, f"Code blocks: {content.count('```') // 2}")

        return all_pass

    except Exception as e:
        return print_result(False, f"Error validating runbook: {e}")


# ============================================================================
# Test 3: Performance Testing Suite Validation
# ============================================================================

def test_performance_suite() -> bool:
    """Validate Performance Testing Suite"""
    print_test_header("Performance Testing Suite Validation")

    perf_test_path = REQUIRED_FILES["perf_test"]

    if not perf_test_path.exists():
        return print_result(False, "Performance test file not found")

    try:
        with open(perf_test_path, 'r', encoding='utf-8') as f:
            content = f.read()

        all_pass = True

        # Check required classes
        required_classes = [
            "class TestConfig",
            "class RequestResult",
            "class EndpointStats",
            "class SystemMetrics",
            "class PerformanceReport",
            "class PerformanceHTTPClient",
            "class SystemMonitor",
            "class TestScenario",
            "class PerformanceTester",
            "class ReportGenerator"
        ]

        for cls in required_classes:
            has_class = cls in content
            all_pass &= print_result(has_class, f"Class: {cls}")

        # Check required CLI arguments
        required_args = [
            "'--url'",
            "'--duration'",
            "'--concurrency'",
            "'--profile'",
            "'--baseline'",
            "'--output-json'",
            "'--output-md'",
            "'--verbose'"
        ]

        for arg in required_args:
            has_arg = arg in content
            all_pass &= print_result(has_arg, f"CLI arg: {arg}")

        # Check for regression detection
        has_regression = "regression" in content.lower()
        all_pass &= print_result(has_regression, "Regression detection")

        # Check for metrics calculation
        metrics = ["p95", "p99", "latency", "throughput"]
        for metric in metrics:
            has_metric = metric in content
            all_pass &= print_result(has_metric, f"Metric: {metric}")

        # Check line count
        line_count = len(content.split('\n'))
        min_lines = 400
        has_sufficient_lines = line_count >= min_lines
        all_pass &= print_result(
            has_sufficient_lines,
            f"Line count: {line_count} (minimum {min_lines})"
        )

        return all_pass

    except Exception as e:
        return print_result(False, f"Error validating performance suite: {e}")


# ============================================================================
# Test 4: Security Audit Tool Validation
# ============================================================================

def test_security_audit_tool() -> bool:
    """Validate Security Audit Tool"""
    print_test_header("Security Audit Tool Validation")

    audit_path = REQUIRED_FILES["security_audit"]

    if not audit_path.exists():
        return print_result(False, "Security audit file not found")

    try:
        with open(audit_path, 'r', encoding='utf-8') as f:
            content = f.read()

        all_pass = True

        # Check required classes
        required_classes = [
            "class Severity",
            "class SecurityFinding",
            "class AuditReport",
            "class FileIntegrityChecker",
            "class SignatureVerifier",
            "class EncryptedFileInspector",
            "class SBOMVulnerabilityScanner",
            "class APISecurityTester",
            "class ConfigurationAuditor",
            "class SecurityAuditor"
        ]

        for cls in required_classes:
            has_class = cls in content
            all_pass &= print_result(has_class, f"Class: {cls}")

        # Check required CLI arguments
        required_args = [
            "'--deep'",
            "'--scan-sbom'",
            "'--verify-signature'",
            "'--check-api'",
            "'--check-config'",
            "'--json'",
            "'--verbose'"
        ]

        for arg in required_args:
            has_arg = arg in content
            all_pass &= print_result(has_arg, f"CLI arg: {arg}")

        # Check for security features
        security_features = [
            "SHA-256",
            "RSA-PSS",
            "AES-256-GCM",
            "CVE",
            "vulnerability"
        ]

        for feature in security_features:
            has_feature = feature in content
            all_pass &= print_result(has_feature, f"Feature: {feature}")

        # Check line count
        line_count = len(content.split('\n'))
        min_lines = 350
        has_sufficient_lines = line_count >= min_lines
        all_pass &= print_result(
            has_sufficient_lines,
            f"Line count: {line_count} (minimum {min_lines})"
        )

        # Check for Grype/Trivy integration
        has_grype = "grype" in content.lower()
        has_trivy = "trivy" in content.lower()
        has_scanner = has_grype or has_trivy
        all_pass &= print_result(has_scanner, "Vulnerability scanner integration")

        return all_pass

    except Exception as e:
        return print_result(False, f"Error validating security audit: {e}")


# ============================================================================
# Test 5: Integration with prepare_release_artifacts.py
# ============================================================================

def test_release_script_integration() -> bool:
    """Validate integration with release script"""
    print_test_header("Release Script Integration")

    release_script_path = REQUIRED_FILES["release_script"]

    if not release_script_path.exists():
        return print_result(False, "Release script not found")

    try:
        with open(release_script_path, 'r', encoding='utf-8') as f:
            content = f.read()

        all_pass = True

        # Check for new CLI arguments
        new_args = [
            "'--run-performance-tests'",
            "'--run-security-audit'",
            "'--api-url'"
        ]

        for arg in new_args:
            has_arg = arg in content
            all_pass &= print_result(has_arg, f"CLI arg: {arg}")

        # Check for performance testing integration
        perf_markers = [
            "# Performance testing (Phase 14.7 Task 2)",
            "perf_test_completed",
            "run_performance_tests.py"
        ]

        for marker in perf_markers:
            has_marker = marker in content
            all_pass &= print_result(has_marker, f"Performance marker: {marker}")

        # Check for security audit integration
        audit_markers = [
            "# Security audit (Phase 14.7 Task 2)",
            "security_audit_completed",
            "security_audit.py"
        ]

        for marker in audit_markers:
            has_marker = marker in content
            all_pass &= print_result(has_marker, f"Security marker: {marker}")

        # Check manifest update
        manifest_checks = [
            "perf_tests=",
            "security_audit=",
            '"validation":'
        ]

        for check in manifest_checks:
            has_check = check in content
            all_pass &= print_result(has_check, f"Manifest update: {check}")

        return all_pass

    except Exception as e:
        return print_result(False, f"Error validating release script: {e}")


# ============================================================================
# Test 6: Documentation Cross-References
# ============================================================================

def test_documentation_cross_refs() -> bool:
    """Test documentation cross-references"""
    print_test_header("Documentation Cross-References")

    all_pass = True

    # Check that runbook references the new tools
    runbook_path = REQUIRED_FILES["runbook"]
    if runbook_path.exists():
        with open(runbook_path, 'r', encoding='utf-8') as f:
            runbook_content = f.read()

        # Should reference performance testing
        has_perf_ref = "performance" in runbook_content.lower()
        all_pass &= print_result(has_perf_ref, "Runbook references performance testing")

        # Should reference security audit
        has_security_ref = "security audit" in runbook_content.lower()
        all_pass &= print_result(has_security_ref, "Runbook references security audit")

        # Should have command examples
        has_commands = "```bash" in runbook_content or "```" in runbook_content
        all_pass &= print_result(has_commands, "Runbook has command examples")

    return all_pass


# ============================================================================
# Test 7: File Size and Quality Checks
# ============================================================================

def test_file_quality() -> bool:
    """Test file quality metrics"""
    print_test_header("File Quality Checks")

    all_pass = True

    quality_checks = {
        "runbook": {
            "path": REQUIRED_FILES["runbook"],
            "min_size": 100000,  # ~100KB for comprehensive runbook
            "file_type": "markdown"
        },
        "perf_test": {
            "path": REQUIRED_FILES["perf_test"],
            "min_size": 20000,  # ~20KB for 400-700 LOC
            "file_type": "python"
        },
        "security_audit": {
            "path": REQUIRED_FILES["security_audit"],
            "min_size": 15000,  # ~15KB for 350-600 LOC
            "file_type": "python"
        }
    }

    for name, checks in quality_checks.items():
        path = checks["path"]
        if not path.exists():
            all_pass &= print_result(False, f"{name}: File not found")
            continue

        file_size = path.stat().st_size
        min_size = checks["min_size"]
        size_ok = file_size >= min_size
        all_pass &= print_result(
            size_ok,
            f"{name}: Size {file_size:,} bytes (min {min_size:,})"
        )

        # Check encoding
        try:
            with open(path, 'r', encoding='utf-8') as f:
                f.read()
            all_pass &= print_result(True, f"{name}: UTF-8 encoding valid")
        except UnicodeDecodeError:
            all_pass &= print_result(False, f"{name}: UTF-8 encoding error")

    return all_pass


# ============================================================================
# Main Test Runner
# ============================================================================

def main() -> int:
    """Run all validation tests"""

    print("=" * 80)
    print("PHASE 14.7 TASK 2 VALIDATION TEST SUITE")
    print("=" * 80)
    print(f"Project Root: {PROJECT_ROOT}")
    print()

    tests = [
        ("File Existence", test_file_existence),
        ("Production Runbook", test_production_runbook),
        ("Performance Suite", test_performance_suite),
        ("Security Audit Tool", test_security_audit_tool),
        ("Release Script Integration", test_release_script_integration),
        ("Documentation Cross-References", test_documentation_cross_refs),
        ("File Quality", test_file_quality)
    ]

    results = []

    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed))
        except Exception as e:
            print(f"\n[ERROR] Test '{test_name}' raised exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    total = len(results)
    passed = sum(1 for _, p in results if p)
    failed = total - passed

    for test_name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status} {test_name}")

    print("=" * 80)
    print(f"Total tests: {total}")
    print(f"Passed:      {passed}")
    print(f"Failed:      {failed}")
    print("=" * 80)

    if failed == 0:
        print("\n[PASS] ALL TESTS PASSED - Phase 14.7 Task 2 implementations validated!")
        return 0
    else:
        print(f"\n[FAIL] {failed} test(s) failed - Please review implementation")
        return 1


if __name__ == '__main__':
    sys.exit(main())
