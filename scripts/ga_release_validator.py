#!/usr/bin/env python3
"""
GA Release Validator Engine

Phase 14.9 - Task 1: GA Hardening & Production Release Gate

This module provides a comprehensive validation engine to determine whether
the T.A.R.S. system is ready for General Availability (GA) release.

Exit Codes:
    150 = GA READY - All validations passed
    151 = GA BLOCKED - Warnings present but no hard errors
    152 = GA FAILED - Hard errors detected
    199 = GENERAL ERROR - Unexpected error during validation

Author: T.A.R.S. Development Team
Version: 1.0.4
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Exit codes
EXIT_GA_READY = 150
EXIT_GA_BLOCKED = 151
EXIT_GA_FAILED = 152
EXIT_GENERAL_ERROR = 199


class ValidationSeverity(Enum):
    """Severity levels for validation results."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ValidationCategory(Enum):
    """Categories for validation checks."""
    VERSION_FILES = "version_files"
    REQUIRED_DOCS = "required_docs"
    ANALYTICS_MODULES = "analytics_modules"
    CLI_TOOLS = "cli_tools"
    EXIT_CODES = "exit_codes"
    CODE_QUALITY = "code_quality"
    SCHEMA_CONSISTENCY = "schema_consistency"
    SECURITY = "security"
    TESTS = "tests"


@dataclass
class ValidationResult:
    """Result of a single validation check."""
    check_id: str
    check_name: str
    category: ValidationCategory
    passed: bool
    severity: ValidationSeverity
    message: str
    details: Optional[Dict[str, Any]] = None
    remediation: Optional[str] = None


@dataclass
class ValidationReport:
    """Complete validation report."""
    report_id: str
    generated_at: str
    version: str
    overall_status: str
    total_checks: int
    passed_checks: int
    warning_checks: int
    failed_checks: int
    critical_checks: int
    results: List[ValidationResult] = field(default_factory=list)
    ga_ready: bool = False
    exit_code: int = EXIT_GA_FAILED
    summary: str = ""
    blockers: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class GAValidator:
    """
    GA Release Validator Engine.

    Performs comprehensive validation checks to determine GA readiness.
    """

    # Required files for GA release
    REQUIRED_VERSION_FILES = [
        "VERSION",
        "CHANGELOG.md",
    ]

    # Required documentation files
    REQUIRED_DOCS = [
        "README.md",
        "docs/PHASE14_6_ENTERPRISE_HARDENING.md",
        "docs/PHASE14_6_API_GUIDE.md",
        "docs/PHASE14_6_PRODUCTION_RUNBOOK.md",
        "docs/ORG_SLA_INTELLIGENCE_ENGINE.md",
        "docs/ORG_TEMPORAL_INTELLIGENCE_ENGINE.md",
        "docs/ORG_TREND_CORRELATION_ENGINE.md",
        "docs/ORG_HEALTH_GOVERNANCE_GUIDE.md",
    ]

    # Required analytics modules
    REQUIRED_ANALYTICS_MODULES = [
        "analytics/org_sla_intelligence.py",
        "analytics/org_temporal_intelligence.py",
        "analytics/org_trend_correlation.py",
        "analytics/org_alerting_engine.py",
        "analytics/org_health_aggregator.py",
        "analytics/trend_analyzer.py",
        "analytics/alerting_engine.py",
        "analytics/repository_health_dashboard.py",
    ]

    # Required CLI runners
    REQUIRED_CLI_RUNNERS = [
        "analytics/run_org_sla_intelligence.py",
        "analytics/run_org_temporal_intelligence.py",
        "analytics/run_org_trend_correlation.py",
        "analytics/run_org_alerts.py",
        "analytics/run_org_health.py",
        "analytics/run_trends.py",
        "analytics/run_alerts.py",
    ]

    # Placeholder patterns to detect
    PLACEHOLDER_PATTERNS = [
        r'\bTODO\b',
        r'\bFIXME\b',
        r'\bXXX\b',
        r'\bHACK\b',
        r'PLACEHOLDER',
        r'NOT_IMPLEMENTED',
        r'COMING_SOON',
    ]

    # Expected exit code ranges by module
    EXPECTED_EXIT_CODES = {
        "org_sla_intelligence": {"success": [140], "warning": [141], "error": [142, 143, 144], "general": [199]},
        "org_temporal_intelligence": {"success": [130], "warning": [131], "error": [132, 133, 134], "general": [199]},
        "org_trend_correlation": {"success": [120], "warning": [121], "error": [122, 123, 124], "general": [199]},
        "org_alerting": {"success": [110], "warning": [111], "error": [112, 113, 114], "general": [199]},
        "org_health": {"success": [100], "warning": [101], "error": [102, 103, 104], "general": [199]},
    }

    def __init__(self, project_root: Path, verbose: bool = False, dry_run: bool = False):
        """
        Initialize the GA Validator.

        Args:
            project_root: Path to the project root directory
            verbose: Enable verbose output
            dry_run: Run in dry-run mode (no external commands)
        """
        self.project_root = project_root
        self.verbose = verbose
        self.dry_run = dry_run
        self.results: List[ValidationResult] = []

    def log(self, message: str, level: str = "INFO") -> None:
        """Log a message if verbose mode is enabled."""
        if self.verbose:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] [{level}] {message}")

    def add_result(self, result: ValidationResult) -> None:
        """Add a validation result."""
        self.results.append(result)
        status = "PASS" if result.passed else "FAIL"
        self.log(f"[{status}] {result.check_name}: {result.message}",
                 "INFO" if result.passed else "WARNING")

    def validate_version_files(self) -> List[ValidationResult]:
        """Validate required version files exist and have proper content."""
        results = []

        # Check VERSION file
        version_path = self.project_root / "VERSION"
        if version_path.exists():
            version_content = version_path.read_text().strip()

            # Check for RC suffix (should not be present in GA)
            if "-rc" in version_content.lower() or "-RC" in version_content:
                results.append(ValidationResult(
                    check_id="VER001",
                    check_name="VERSION file GA format",
                    category=ValidationCategory.VERSION_FILES,
                    passed=False,
                    severity=ValidationSeverity.CRITICAL,
                    message=f"VERSION file contains RC suffix: {version_content}",
                    details={"current_version": version_content},
                    remediation="Update VERSION file to remove -rc suffix for GA release"
                ))
            else:
                # Validate semantic versioning format
                semver_pattern = r'^\d+\.\d+\.\d+$'
                if re.match(semver_pattern, version_content):
                    results.append(ValidationResult(
                        check_id="VER001",
                        check_name="VERSION file GA format",
                        category=ValidationCategory.VERSION_FILES,
                        passed=True,
                        severity=ValidationSeverity.INFO,
                        message=f"VERSION file has valid GA format: {version_content}",
                        details={"version": version_content}
                    ))
                else:
                    results.append(ValidationResult(
                        check_id="VER001",
                        check_name="VERSION file GA format",
                        category=ValidationCategory.VERSION_FILES,
                        passed=False,
                        severity=ValidationSeverity.ERROR,
                        message=f"VERSION file has invalid format: {version_content}",
                        details={"current_version": version_content},
                        remediation="Use semantic versioning format: MAJOR.MINOR.PATCH"
                    ))
        else:
            results.append(ValidationResult(
                check_id="VER001",
                check_name="VERSION file existence",
                category=ValidationCategory.VERSION_FILES,
                passed=False,
                severity=ValidationSeverity.CRITICAL,
                message="VERSION file not found",
                remediation="Create VERSION file with GA version number"
            ))

        # Check CHANGELOG.md
        changelog_path = self.project_root / "CHANGELOG.md"
        if changelog_path.exists():
            changelog_content = changelog_path.read_text()

            # Check for GA entry
            if "[1.0.4]" in changelog_content or "## [1.0.4]" in changelog_content:
                results.append(ValidationResult(
                    check_id="VER002",
                    check_name="CHANGELOG GA entry",
                    category=ValidationCategory.VERSION_FILES,
                    passed=True,
                    severity=ValidationSeverity.INFO,
                    message="CHANGELOG.md contains GA version entry"
                ))
            else:
                results.append(ValidationResult(
                    check_id="VER002",
                    check_name="CHANGELOG GA entry",
                    category=ValidationCategory.VERSION_FILES,
                    passed=False,
                    severity=ValidationSeverity.WARNING,
                    message="CHANGELOG.md missing GA version entry",
                    remediation="Add [1.0.4] section to CHANGELOG.md"
                ))
        else:
            results.append(ValidationResult(
                check_id="VER002",
                check_name="CHANGELOG existence",
                category=ValidationCategory.VERSION_FILES,
                passed=False,
                severity=ValidationSeverity.CRITICAL,
                message="CHANGELOG.md not found",
                remediation="Create CHANGELOG.md with GA release notes"
            ))

        return results

    def validate_required_docs(self) -> List[ValidationResult]:
        """Validate required documentation files exist."""
        results = []

        for doc_path in self.REQUIRED_DOCS:
            full_path = self.project_root / doc_path
            if full_path.exists():
                # Check file is not empty
                content = full_path.read_text()
                if len(content.strip()) > 100:  # Minimum content threshold
                    results.append(ValidationResult(
                        check_id=f"DOC_{doc_path.replace('/', '_').upper()[:20]}",
                        check_name=f"Documentation: {doc_path}",
                        category=ValidationCategory.REQUIRED_DOCS,
                        passed=True,
                        severity=ValidationSeverity.INFO,
                        message=f"Documentation file exists with content: {doc_path}",
                        details={"path": doc_path, "size_bytes": len(content)}
                    ))
                else:
                    results.append(ValidationResult(
                        check_id=f"DOC_{doc_path.replace('/', '_').upper()[:20]}",
                        check_name=f"Documentation: {doc_path}",
                        category=ValidationCategory.REQUIRED_DOCS,
                        passed=False,
                        severity=ValidationSeverity.WARNING,
                        message=f"Documentation file is too short: {doc_path}",
                        details={"path": doc_path, "size_bytes": len(content)},
                        remediation=f"Add meaningful content to {doc_path}"
                    ))
            else:
                results.append(ValidationResult(
                    check_id=f"DOC_{doc_path.replace('/', '_').upper()[:20]}",
                    check_name=f"Documentation: {doc_path}",
                    category=ValidationCategory.REQUIRED_DOCS,
                    passed=False,
                    severity=ValidationSeverity.ERROR,
                    message=f"Required documentation missing: {doc_path}",
                    remediation=f"Create {doc_path} with appropriate content"
                ))

        return results

    def validate_analytics_modules(self) -> List[ValidationResult]:
        """Validate required analytics modules exist and are syntactically valid."""
        results = []

        for module_path in self.REQUIRED_ANALYTICS_MODULES:
            full_path = self.project_root / module_path
            if full_path.exists():
                # Check Python syntax
                try:
                    content = full_path.read_text()
                    compile(content, module_path, 'exec')
                    results.append(ValidationResult(
                        check_id=f"MOD_{module_path.split('/')[-1].upper()[:20]}",
                        check_name=f"Analytics Module: {module_path}",
                        category=ValidationCategory.ANALYTICS_MODULES,
                        passed=True,
                        severity=ValidationSeverity.INFO,
                        message=f"Module exists and has valid syntax: {module_path}",
                        details={"path": module_path, "lines": len(content.splitlines())}
                    ))
                except SyntaxError as e:
                    results.append(ValidationResult(
                        check_id=f"MOD_{module_path.split('/')[-1].upper()[:20]}",
                        check_name=f"Analytics Module: {module_path}",
                        category=ValidationCategory.ANALYTICS_MODULES,
                        passed=False,
                        severity=ValidationSeverity.CRITICAL,
                        message=f"Module has syntax error: {module_path}",
                        details={"path": module_path, "error": str(e)},
                        remediation=f"Fix syntax error in {module_path}: {e}"
                    ))
            else:
                results.append(ValidationResult(
                    check_id=f"MOD_{module_path.split('/')[-1].upper()[:20]}",
                    check_name=f"Analytics Module: {module_path}",
                    category=ValidationCategory.ANALYTICS_MODULES,
                    passed=False,
                    severity=ValidationSeverity.CRITICAL,
                    message=f"Required module missing: {module_path}",
                    remediation=f"Create {module_path} module"
                ))

        return results

    def validate_cli_runners(self) -> List[ValidationResult]:
        """Validate CLI runner scripts exist and have proper help output."""
        results = []

        for cli_path in self.REQUIRED_CLI_RUNNERS:
            full_path = self.project_root / cli_path
            if full_path.exists():
                # Check for argparse or help functionality
                content = full_path.read_text()
                has_argparse = "argparse" in content or "ArgumentParser" in content
                has_help = "--help" in content or "add_argument" in content

                if has_argparse and has_help:
                    results.append(ValidationResult(
                        check_id=f"CLI_{cli_path.split('/')[-1].upper()[:20]}",
                        check_name=f"CLI Runner: {cli_path}",
                        category=ValidationCategory.CLI_TOOLS,
                        passed=True,
                        severity=ValidationSeverity.INFO,
                        message=f"CLI runner has proper argument handling: {cli_path}"
                    ))
                else:
                    results.append(ValidationResult(
                        check_id=f"CLI_{cli_path.split('/')[-1].upper()[:20]}",
                        check_name=f"CLI Runner: {cli_path}",
                        category=ValidationCategory.CLI_TOOLS,
                        passed=False,
                        severity=ValidationSeverity.WARNING,
                        message=f"CLI runner missing argparse: {cli_path}",
                        remediation=f"Add proper CLI argument handling to {cli_path}"
                    ))
            else:
                results.append(ValidationResult(
                    check_id=f"CLI_{cli_path.split('/')[-1].upper()[:20]}",
                    check_name=f"CLI Runner: {cli_path}",
                    category=ValidationCategory.CLI_TOOLS,
                    passed=False,
                    severity=ValidationSeverity.ERROR,
                    message=f"Required CLI runner missing: {cli_path}",
                    remediation=f"Create {cli_path} CLI tool"
                ))

        return results

    def validate_exit_codes(self) -> List[ValidationResult]:
        """Validate exit code definitions in modules."""
        results = []

        exit_code_modules = [
            ("analytics/org_sla_intelligence.py", ["140", "141", "142", "143", "144", "199"]),
            ("analytics/org_temporal_intelligence.py", ["130", "131", "132", "133", "134", "199"]),
            ("analytics/org_trend_correlation.py", ["120", "121", "122", "123", "124", "199"]),
        ]

        for module_path, expected_codes in exit_code_modules:
            full_path = self.project_root / module_path
            if full_path.exists():
                content = full_path.read_text()
                missing_codes = []

                for code in expected_codes:
                    if code not in content:
                        missing_codes.append(code)

                if not missing_codes:
                    results.append(ValidationResult(
                        check_id=f"EXIT_{module_path.split('/')[-1].upper()[:15]}",
                        check_name=f"Exit Codes: {module_path}",
                        category=ValidationCategory.EXIT_CODES,
                        passed=True,
                        severity=ValidationSeverity.INFO,
                        message=f"All expected exit codes defined in {module_path}",
                        details={"expected_codes": expected_codes}
                    ))
                else:
                    results.append(ValidationResult(
                        check_id=f"EXIT_{module_path.split('/')[-1].upper()[:15]}",
                        check_name=f"Exit Codes: {module_path}",
                        category=ValidationCategory.EXIT_CODES,
                        passed=False,
                        severity=ValidationSeverity.WARNING,
                        message=f"Missing exit codes in {module_path}: {missing_codes}",
                        details={"missing_codes": missing_codes},
                        remediation=f"Define exit codes {missing_codes} in {module_path}"
                    ))

        return results

    def validate_code_quality(self) -> List[ValidationResult]:
        """Check for placeholder markers and code quality issues."""
        results = []

        # Files to scan for placeholders
        scan_paths = [
            "analytics/*.py",
            "scripts/*.py",
            "observability/*.py",
        ]

        placeholder_files = []

        for pattern in scan_paths:
            import glob
            for file_path in glob.glob(str(self.project_root / pattern)):
                path = Path(file_path)
                if path.exists() and path.is_file():
                    content = path.read_text()

                    for placeholder in self.PLACEHOLDER_PATTERNS:
                        matches = re.findall(placeholder, content, re.IGNORECASE)
                        if matches:
                            placeholder_files.append({
                                "file": str(path.relative_to(self.project_root)),
                                "pattern": placeholder,
                                "count": len(matches)
                            })

        if not placeholder_files:
            results.append(ValidationResult(
                check_id="QUAL001",
                check_name="Code Quality - No Placeholders",
                category=ValidationCategory.CODE_QUALITY,
                passed=True,
                severity=ValidationSeverity.INFO,
                message="No placeholder markers found in codebase"
            ))
        else:
            results.append(ValidationResult(
                check_id="QUAL001",
                check_name="Code Quality - Placeholders Found",
                category=ValidationCategory.CODE_QUALITY,
                passed=False,
                severity=ValidationSeverity.WARNING,
                message=f"Found placeholder markers in {len(placeholder_files)} locations",
                details={"files": placeholder_files},
                remediation="Remove or resolve TODO/FIXME/placeholder markers before GA"
            ))

        return results

    def validate_security(self) -> List[ValidationResult]:
        """Validate security-related files and configurations."""
        results = []

        # Check for security-sensitive files that shouldn't be committed
        sensitive_patterns = [
            ".env",
            "*.key",
            "*.pem",
            "secrets.yaml",
            "credentials.json",
        ]

        sensitive_found = []
        for pattern in sensitive_patterns:
            import glob
            for file_path in glob.glob(str(self.project_root / pattern)):
                # Exclude example files
                if "example" not in file_path.lower() and "template" not in file_path.lower():
                    sensitive_found.append(file_path)

        if not sensitive_found:
            results.append(ValidationResult(
                check_id="SEC001",
                check_name="Security - No Sensitive Files",
                category=ValidationCategory.SECURITY,
                passed=True,
                severity=ValidationSeverity.INFO,
                message="No sensitive files found in repository root"
            ))
        else:
            results.append(ValidationResult(
                check_id="SEC001",
                check_name="Security - Sensitive Files Found",
                category=ValidationCategory.SECURITY,
                passed=False,
                severity=ValidationSeverity.CRITICAL,
                message=f"Found potentially sensitive files: {sensitive_found}",
                details={"files": sensitive_found},
                remediation="Remove or add to .gitignore: " + ", ".join(sensitive_found)
            ))

        # Check for .gitignore
        gitignore_path = self.project_root / ".gitignore"
        if gitignore_path.exists():
            content = gitignore_path.read_text()
            required_ignores = [".env", "*.key", "*.pem", "__pycache__"]
            missing_ignores = [ig for ig in required_ignores if ig not in content]

            if not missing_ignores:
                results.append(ValidationResult(
                    check_id="SEC002",
                    check_name="Security - .gitignore Coverage",
                    category=ValidationCategory.SECURITY,
                    passed=True,
                    severity=ValidationSeverity.INFO,
                    message=".gitignore contains all required patterns"
                ))
            else:
                results.append(ValidationResult(
                    check_id="SEC002",
                    check_name="Security - .gitignore Coverage",
                    category=ValidationCategory.SECURITY,
                    passed=False,
                    severity=ValidationSeverity.WARNING,
                    message=f".gitignore missing patterns: {missing_ignores}",
                    remediation=f"Add to .gitignore: {', '.join(missing_ignores)}"
                ))

        return results

    def validate_tests(self) -> List[ValidationResult]:
        """Validate test suite existence and coverage."""
        results = []

        test_dirs = [
            "tests",
            "tests/integration",
            "tests/unit",
        ]

        test_files_found = 0
        for test_dir in test_dirs:
            test_path = self.project_root / test_dir
            if test_path.exists():
                import glob
                test_files = list(glob.glob(str(test_path / "test_*.py")))
                test_files_found += len(test_files)

        if test_files_found >= 10:  # Minimum threshold for GA
            results.append(ValidationResult(
                check_id="TEST001",
                check_name="Test Coverage - Minimum Tests",
                category=ValidationCategory.TESTS,
                passed=True,
                severity=ValidationSeverity.INFO,
                message=f"Found {test_files_found} test files (minimum: 10)",
                details={"test_count": test_files_found}
            ))
        else:
            results.append(ValidationResult(
                check_id="TEST001",
                check_name="Test Coverage - Minimum Tests",
                category=ValidationCategory.TESTS,
                passed=False,
                severity=ValidationSeverity.WARNING,
                message=f"Only {test_files_found} test files found (minimum: 10)",
                details={"test_count": test_files_found},
                remediation="Add more test coverage before GA release"
            ))

        return results

    def run_all_validations(self) -> ValidationReport:
        """Run all validation checks and generate report."""
        self.log("Starting GA validation...", "INFO")

        # Run all validation categories
        self.results.extend(self.validate_version_files())
        self.results.extend(self.validate_required_docs())
        self.results.extend(self.validate_analytics_modules())
        self.results.extend(self.validate_cli_runners())
        self.results.extend(self.validate_exit_codes())
        self.results.extend(self.validate_code_quality())
        self.results.extend(self.validate_security())
        self.results.extend(self.validate_tests())

        # Aggregate results
        passed = sum(1 for r in self.results if r.passed)
        warnings = sum(1 for r in self.results if not r.passed and r.severity == ValidationSeverity.WARNING)
        errors = sum(1 for r in self.results if not r.passed and r.severity == ValidationSeverity.ERROR)
        critical = sum(1 for r in self.results if not r.passed and r.severity == ValidationSeverity.CRITICAL)

        # Determine overall status
        if critical > 0:
            overall_status = "FAILED"
            exit_code = EXIT_GA_FAILED
            ga_ready = False
        elif errors > 0:
            overall_status = "FAILED"
            exit_code = EXIT_GA_FAILED
            ga_ready = False
        elif warnings > 0:
            overall_status = "BLOCKED"
            exit_code = EXIT_GA_BLOCKED
            ga_ready = False
        else:
            overall_status = "READY"
            exit_code = EXIT_GA_READY
            ga_ready = True

        # Collect blockers and warnings
        blockers = [r.message for r in self.results
                   if not r.passed and r.severity in [ValidationSeverity.CRITICAL, ValidationSeverity.ERROR]]
        warning_messages = [r.message for r in self.results
                          if not r.passed and r.severity == ValidationSeverity.WARNING]

        # Get version
        version_path = self.project_root / "VERSION"
        version = version_path.read_text().strip() if version_path.exists() else "unknown"

        # Generate summary
        summary = f"GA Validation {'PASSED' if ga_ready else 'FAILED'}: "
        summary += f"{passed}/{len(self.results)} checks passed, "
        summary += f"{critical} critical, {errors} errors, {warnings} warnings"

        report = ValidationReport(
            report_id=f"ga_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            generated_at=datetime.now().isoformat(),
            version=version,
            overall_status=overall_status,
            total_checks=len(self.results),
            passed_checks=passed,
            warning_checks=warnings,
            failed_checks=errors,
            critical_checks=critical,
            results=self.results,
            ga_ready=ga_ready,
            exit_code=exit_code,
            summary=summary,
            blockers=blockers,
            warnings=warning_messages
        )

        self.log(summary, "INFO" if ga_ready else "WARNING")
        return report


def report_to_dict(report: ValidationReport) -> Dict[str, Any]:
    """Convert ValidationReport to dictionary for JSON serialization."""
    return {
        "report_id": report.report_id,
        "generated_at": report.generated_at,
        "version": report.version,
        "overall_status": report.overall_status,
        "total_checks": report.total_checks,
        "passed_checks": report.passed_checks,
        "warning_checks": report.warning_checks,
        "failed_checks": report.failed_checks,
        "critical_checks": report.critical_checks,
        "ga_ready": report.ga_ready,
        "exit_code": report.exit_code,
        "summary": report.summary,
        "blockers": report.blockers,
        "warnings": report.warnings,
        "results": [
            {
                "check_id": r.check_id,
                "check_name": r.check_name,
                "category": r.category.value,
                "passed": r.passed,
                "severity": r.severity.value,
                "message": r.message,
                "details": r.details,
                "remediation": r.remediation
            }
            for r in report.results
        ]
    }


def print_human_readable(report: ValidationReport) -> None:
    """Print human-readable validation report."""
    print("\n" + "=" * 70)
    print("GA RELEASE VALIDATION REPORT")
    print("=" * 70)
    print(f"\nReport ID: {report.report_id}")
    print(f"Generated: {report.generated_at}")
    print(f"Version:   {report.version}")
    print()

    # Status banner
    status_banner = {
        "READY": "[GA READY] All validations passed - safe to release",
        "BLOCKED": "[GA BLOCKED] Warnings present - review before release",
        "FAILED": "[GA FAILED] Critical errors - cannot release"
    }
    print(status_banner.get(report.overall_status, "[UNKNOWN]"))
    print()

    # Summary stats
    print("-" * 70)
    print("SUMMARY")
    print("-" * 70)
    print(f"  Total Checks:    {report.total_checks}")
    print(f"  Passed:          {report.passed_checks}")
    print(f"  Warnings:        {report.warning_checks}")
    print(f"  Errors:          {report.failed_checks}")
    print(f"  Critical:        {report.critical_checks}")
    print()

    # Blockers
    if report.blockers:
        print("-" * 70)
        print("BLOCKERS (must fix before GA)")
        print("-" * 70)
        for i, blocker in enumerate(report.blockers, 1):
            print(f"  {i}. {blocker}")
        print()

    # Warnings
    if report.warnings:
        print("-" * 70)
        print("WARNINGS (recommended to fix)")
        print("-" * 70)
        for i, warning in enumerate(report.warnings, 1):
            print(f"  {i}. {warning}")
        print()

    # Results by category
    print("-" * 70)
    print("RESULTS BY CATEGORY")
    print("-" * 70)

    categories = {}
    for result in report.results:
        cat_name = result.category.value
        if cat_name not in categories:
            categories[cat_name] = {"passed": 0, "failed": 0}
        if result.passed:
            categories[cat_name]["passed"] += 1
        else:
            categories[cat_name]["failed"] += 1

    for cat, counts in sorted(categories.items()):
        total = counts["passed"] + counts["failed"]
        pct = (counts["passed"] / total * 100) if total > 0 else 0
        status_icon = "[OK]" if counts["failed"] == 0 else "[!!]"
        print(f"  {status_icon} {cat}: {counts['passed']}/{total} ({pct:.0f}%)")

    print()
    print("=" * 70)
    print(f"EXIT CODE: {report.exit_code}")
    print("=" * 70)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="GA Release Validator Engine - Phase 14.9",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exit Codes:
  150  GA READY    - All validations passed
  151  GA BLOCKED  - Warnings present but no hard errors
  152  GA FAILED   - Hard errors detected
  199  ERROR       - Unexpected error during validation

Examples:
  python ga_release_validator.py
  python ga_release_validator.py --output validation.json --json
  python ga_release_validator.py --verbose --dry-run
        """
    )

    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Path to project root (default: current directory)"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Output path for JSON report"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output JSON to stdout"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run in dry-run mode (no external commands)"
    )
    parser.add_argument(
        "--fail-on-warning",
        action="store_true",
        help="Exit with failure code on warnings"
    )

    args = parser.parse_args()

    try:
        # Initialize validator
        validator = GAValidator(
            project_root=args.project_root,
            verbose=args.verbose,
            dry_run=args.dry_run
        )

        # Run validations
        report = validator.run_all_validations()

        # Convert to dict for serialization
        report_dict = report_to_dict(report)

        # Output results
        if args.json:
            print(json.dumps(report_dict, indent=2))
        else:
            print_human_readable(report)

        # Write to file if requested
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, 'w') as f:
                json.dump(report_dict, f, indent=2)
            if args.verbose:
                print(f"\nReport written to: {args.output}")

        # Determine exit code
        if args.fail_on_warning and report.warning_checks > 0:
            sys.exit(EXIT_GA_BLOCKED)

        sys.exit(report.exit_code)

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(EXIT_GENERAL_ERROR)


if __name__ == "__main__":
    main()
