"""
Production-Grade Post-Release Validation Suite (PRVS) & Regression Guard for T.A.R.S.

Comprehensive post-release validation subsystem that performs:
- Behavioral regression checks on critical services
- SBOM delta-analysis vs. previous release
- SLSA provenance delta-analysis
- API surface compatibility checks (breaking vs non-breaking)
- Performance drift detection with threshold alerting
- Security profile regression detection
- Auto-generated validation report (JSON + Markdown)

This subsystem runs AFTER artifact verification (Phase 14.7 Task 3)
and BEFORE release publication.

Designed for:
- CI/CD integration (post-verification gate)
- Offline operation (air-gapped environments)
- Deterministic output
- Cross-platform compatibility (Windows, Linux, macOS)
- Non-interactive batch processing
- < 5 second runtime for typical releases

Exit Codes (20-29 range for PRVS):
  0  - All validations passed
  20 - Behavioral regression detected
  21 - SBOM delta analysis failed (critical changes)
  22 - SLSA provenance regression detected
  23 - Breaking API changes detected
  24 - Performance drift exceeded threshold
  25 - Security regression detected
  26 - Validation data missing (baseline not found)
  27 - Validation orchestration error
  28 - Policy gate failure (strict mode)
  29 - General validation error

Compatible with Phase 14.7 Task 4
"""

from typing import Dict, Any, List, Optional, Tuple, Literal, Set
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
import json
import logging
import sys
import re
from enum import Enum

# Configure logging
logger = logging.getLogger(__name__)


# ============================================================================
# CUSTOM EXCEPTIONS
# ============================================================================

class ValidationError(Exception):
    """Base exception for validation failures."""
    pass


class BehavioralRegressionError(ValidationError):
    """Raised when behavioral regression is detected."""
    pass


class SBOMDeltaError(ValidationError):
    """Raised when critical SBOM changes are detected."""
    pass


class SLSADeltaError(ValidationError):
    """Raised when SLSA provenance regression is detected."""
    pass


class APICompatibilityError(ValidationError):
    """Raised when breaking API changes are detected."""
    pass


class PerformanceDriftError(ValidationError):
    """Raised when performance drift exceeds threshold."""
    pass


class SecurityRegressionError(ValidationError):
    """Raised when security regression is detected."""
    pass


class BaselineMissingError(ValidationError):
    """Raised when baseline data is missing."""
    pass


class ValidationOrchestrationError(ValidationError):
    """Raised when orchestration fails."""
    pass


class PolicyGateError(ValidationError):
    """Raised when policy gate fails in strict mode."""
    pass


# ============================================================================
# ENUMS
# ============================================================================

class ChangeType(Enum):
    """Type of change detected in delta analysis."""
    ADDED = "added"
    REMOVED = "removed"
    MODIFIED = "modified"
    UNCHANGED = "unchanged"


class Severity(Enum):
    """Severity level for detected issues."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class APIChangeType(Enum):
    """Type of API change."""
    BREAKING = "breaking"
    NON_BREAKING = "non_breaking"
    ADDITION = "addition"
    DEPRECATION = "deprecation"


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class ComponentDelta:
    """Represents a change in SBOM component."""
    name: str
    change_type: ChangeType
    old_version: Optional[str] = None
    new_version: Optional[str] = None
    severity: Severity = Severity.INFO
    details: str = ""


@dataclass
class SBOMDeltaResult:
    """Result of SBOM delta analysis."""
    total_components_baseline: int
    total_components_current: int
    added: List[ComponentDelta]
    removed: List[ComponentDelta]
    modified: List[ComponentDelta]
    unchanged: int
    critical_changes: List[ComponentDelta]
    status: Literal["passed", "failed", "warning"]
    details: str = ""


@dataclass
class SLSADelta:
    """Represents a change in SLSA provenance."""
    field: str
    change_type: ChangeType
    old_value: Any
    new_value: Any
    severity: Severity = Severity.INFO


@dataclass
class SLSADeltaResult:
    """Result of SLSA delta analysis."""
    level_baseline: int
    level_current: int
    builder_changed: bool
    build_type_changed: bool
    materials_delta: List[SLSADelta]
    parameters_delta: List[SLSADelta]
    status: Literal["passed", "failed", "warning"]
    details: str = ""


@dataclass
class APIChange:
    """Represents an API change."""
    endpoint: str
    method: str
    change_type: APIChangeType
    old_signature: Optional[str] = None
    new_signature: Optional[str] = None
    severity: Severity = Severity.INFO
    details: str = ""


@dataclass
class APICompatibilityResult:
    """Result of API compatibility check."""
    total_endpoints_baseline: int
    total_endpoints_current: int
    breaking_changes: List[APIChange]
    non_breaking_changes: List[APIChange]
    additions: List[APIChange]
    deprecations: List[APIChange]
    status: Literal["passed", "failed", "warning"]
    details: str = ""


@dataclass
class PerformanceMetric:
    """Performance metric comparison."""
    metric_name: str
    baseline_value: float
    current_value: float
    drift_percent: float
    threshold_percent: float
    exceeded: bool
    severity: Severity = Severity.INFO


@dataclass
class PerformanceDriftResult:
    """Result of performance drift analysis."""
    metrics: List[PerformanceMetric]
    exceeded_count: int
    max_drift_percent: float
    status: Literal["passed", "failed", "warning"]
    details: str = ""


@dataclass
class SecurityFinding:
    """Security finding comparison."""
    finding_type: str
    severity: Severity
    baseline_count: int
    current_count: int
    delta: int
    regression: bool


@dataclass
class SecurityRegressionResult:
    """Result of security regression scan."""
    findings: List[SecurityFinding]
    regressions_count: int
    improvements_count: int
    status: Literal["passed", "failed", "warning"]
    details: str = ""


@dataclass
class BehavioralTest:
    """Behavioral test result."""
    test_name: str
    baseline_result: bool
    current_result: bool
    regression: bool
    details: str = ""


@dataclass
class BehavioralRegressionResult:
    """Result of behavioral regression check."""
    tests: List[BehavioralTest]
    total_tests: int
    regressions_count: int
    status: Literal["passed", "failed", "warning"]
    details: str = ""


@dataclass
class ValidationReport:
    """Complete validation report."""
    version: str
    baseline_version: str
    timestamp: str
    overall_status: Literal["passed", "failed", "warning"]
    behavioral_regression: Optional[BehavioralRegressionResult] = None
    sbom_delta: Optional[SBOMDeltaResult] = None
    slsa_delta: Optional[SLSADeltaResult] = None
    api_compatibility: Optional[APICompatibilityResult] = None
    performance_drift: Optional[PerformanceDriftResult] = None
    security_regression: Optional[SecurityRegressionResult] = None
    failed_checks: int = 0
    warning_checks: int = 0
    passed_checks: int = 0
    total_checks: int = 0
    execution_time_seconds: float = 0.0
    exit_code: int = 0
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with enum serialization."""
        def serialize(obj):
            if isinstance(obj, Enum):
                return obj.value
            elif isinstance(obj, (ComponentDelta, SBOMDeltaResult, SLSADelta, SLSADeltaResult,
                                  APIChange, APICompatibilityResult, PerformanceMetric,
                                  PerformanceDriftResult, SecurityFinding, SecurityRegressionResult,
                                  BehavioralTest, BehavioralRegressionResult)):
                return {k: serialize(v) for k, v in asdict(obj).items()}
            elif isinstance(obj, list):
                return [serialize(item) for item in obj]
            elif isinstance(obj, dict):
                return {k: serialize(v) for k, v in obj.items()}
            return obj

        return serialize(asdict(self))


# ============================================================================
# SBOM DELTA ANALYZER
# ============================================================================

class SBOMDeltaAnalyzer:
    """
    Analyzes changes between baseline and current SBOM.

    Detects:
    - Added components (new dependencies)
    - Removed components (dependency removal)
    - Modified components (version changes)
    - Critical changes (major version upgrades, security-sensitive packages)
    """

    CRITICAL_PACKAGES = {
        'cryptography', 'pycryptodome', 'jwt', 'pyjwt',
        'requests', 'urllib3', 'paramiko', 'fabric',
        'django', 'flask', 'fastapi', 'starlette',
        'sqlalchemy', 'psycopg2', 'pymongo'
    }

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.SBOMDeltaAnalyzer")

    def analyze(
        self,
        baseline_sbom_path: Path,
        current_sbom_path: Path
    ) -> SBOMDeltaResult:
        """
        Perform SBOM delta analysis.

        Args:
            baseline_sbom_path: Path to baseline SBOM
            current_sbom_path: Path to current SBOM

        Returns:
            SBOMDeltaResult
        """
        self.logger.info(f"Analyzing SBOM delta: {baseline_sbom_path.name} -> {current_sbom_path.name}")

        baseline_components = self._load_sbom_components(baseline_sbom_path)
        current_components = self._load_sbom_components(current_sbom_path)

        baseline_map = {c['name']: c for c in baseline_components}
        current_map = {c['name']: c for c in current_components}

        added = []
        removed = []
        modified = []
        unchanged_count = 0

        # Detect added components
        for name, comp in current_map.items():
            if name not in baseline_map:
                delta = ComponentDelta(
                    name=name,
                    change_type=ChangeType.ADDED,
                    new_version=comp.get('version', 'unknown'),
                    severity=self._assess_severity(name, None, comp.get('version')),
                    details=f"New dependency added: {name}@{comp.get('version', 'unknown')}"
                )
                added.append(delta)

        # Detect removed and modified components
        for name, comp in baseline_map.items():
            if name not in current_map:
                delta = ComponentDelta(
                    name=name,
                    change_type=ChangeType.REMOVED,
                    old_version=comp.get('version', 'unknown'),
                    severity=Severity.MEDIUM,
                    details=f"Dependency removed: {name}@{comp.get('version', 'unknown')}"
                )
                removed.append(delta)
            else:
                current_comp = current_map[name]
                old_version = comp.get('version', 'unknown')
                new_version = current_comp.get('version', 'unknown')

                if old_version != new_version:
                    delta = ComponentDelta(
                        name=name,
                        change_type=ChangeType.MODIFIED,
                        old_version=old_version,
                        new_version=new_version,
                        severity=self._assess_severity(name, old_version, new_version),
                        details=f"Version changed: {name} {old_version} -> {new_version}"
                    )
                    modified.append(delta)
                else:
                    unchanged_count += 1

        # Identify critical changes
        critical_changes = [
            delta for delta in (added + removed + modified)
            if delta.severity in (Severity.CRITICAL, Severity.HIGH)
        ]

        # Determine overall status
        if critical_changes:
            status = "failed"
            details = f"Found {len(critical_changes)} critical SBOM changes"
        elif removed or any(d.severity == Severity.MEDIUM for d in modified):
            status = "warning"
            details = f"Found {len(removed)} removals and moderate changes"
        else:
            status = "passed"
            details = "No critical SBOM changes detected"

        return SBOMDeltaResult(
            total_components_baseline=len(baseline_components),
            total_components_current=len(current_components),
            added=added,
            removed=removed,
            modified=modified,
            unchanged=unchanged_count,
            critical_changes=critical_changes,
            status=status,
            details=details
        )

    def _load_sbom_components(self, sbom_path: Path) -> List[Dict[str, Any]]:
        """Load components from SBOM file (CycloneDX or SPDX)."""
        if not sbom_path.exists():
            raise BaselineMissingError(f"SBOM not found: {sbom_path}")

        with open(sbom_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # CycloneDX format
        if 'components' in data:
            return data['components']

        # SPDX format
        elif 'packages' in data:
            return [
                {
                    'name': pkg.get('name', 'unknown'),
                    'version': pkg.get('versionInfo', 'unknown'),
                    'type': 'library'
                }
                for pkg in data['packages']
            ]

        else:
            raise SBOMDeltaError(f"Unsupported SBOM format: {sbom_path}")

    def _assess_severity(self, name: str, old_version: Optional[str], new_version: Optional[str]) -> Severity:
        """Assess severity of component change."""
        # Critical packages always get higher severity
        if name.lower() in self.CRITICAL_PACKAGES:
            if old_version and new_version:
                # Check for major version change
                if self._is_major_version_change(old_version, new_version):
                    return Severity.CRITICAL
                return Severity.HIGH
            return Severity.HIGH

        # Non-critical packages
        if old_version and new_version and self._is_major_version_change(old_version, new_version):
            return Severity.MEDIUM

        return Severity.LOW

    def _is_major_version_change(self, old_version: str, new_version: str) -> bool:
        """Check if version change is a major version bump."""
        try:
            old_major = int(old_version.split('.')[0])
            new_major = int(new_version.split('.')[0])
            return new_major > old_major
        except (ValueError, IndexError):
            return False


# ============================================================================
# SLSA DELTA ANALYZER
# ============================================================================

class SLSADeltaAnalyzer:
    """
    Analyzes changes in SLSA provenance between releases.

    Detects:
    - SLSA level changes (downgrade is critical)
    - Builder identity changes
    - Build type changes
    - Material changes (source dependencies)
    - Build parameter changes
    """

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.SLSADeltaAnalyzer")

    def analyze(
        self,
        baseline_slsa_path: Path,
        current_slsa_path: Path
    ) -> SLSADeltaResult:
        """
        Perform SLSA provenance delta analysis.

        Args:
            baseline_slsa_path: Path to baseline SLSA provenance
            current_slsa_path: Path to current SLSA provenance

        Returns:
            SLSADeltaResult
        """
        self.logger.info(f"Analyzing SLSA delta: {baseline_slsa_path.name} -> {current_slsa_path.name}")

        baseline_slsa = self._load_slsa(baseline_slsa_path)
        current_slsa = self._load_slsa(current_slsa_path)

        baseline_level = self._detect_slsa_level(baseline_slsa)
        current_level = self._detect_slsa_level(current_slsa)

        builder_changed = self._compare_builder(baseline_slsa, current_slsa)
        build_type_changed = self._compare_build_type(baseline_slsa, current_slsa)

        materials_delta = self._compare_materials(baseline_slsa, current_slsa)
        parameters_delta = self._compare_parameters(baseline_slsa, current_slsa)

        # Determine status
        if current_level < baseline_level:
            status = "failed"
            details = f"SLSA level regression: L{baseline_level} -> L{current_level}"
        elif builder_changed or build_type_changed:
            status = "warning"
            details = "Builder or build type changed"
        else:
            status = "passed"
            details = "No critical SLSA changes detected"

        return SLSADeltaResult(
            level_baseline=baseline_level,
            level_current=current_level,
            builder_changed=builder_changed,
            build_type_changed=build_type_changed,
            materials_delta=materials_delta,
            parameters_delta=parameters_delta,
            status=status,
            details=details
        )

    def _load_slsa(self, slsa_path: Path) -> Dict[str, Any]:
        """Load SLSA provenance file."""
        if not slsa_path.exists():
            raise BaselineMissingError(f"SLSA provenance not found: {slsa_path}")

        with open(slsa_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _detect_slsa_level(self, slsa_data: Dict[str, Any]) -> int:
        """Detect SLSA level from provenance."""
        predicate = slsa_data.get('predicate', {})
        build_definition = predicate.get('buildDefinition', {})

        # Level 3: Requires hermetic, reproducible build
        if build_definition.get('externalParameters', {}).get('hermetic', False):
            return 3

        # Level 2: Requires signed provenance
        if 'buildMetadata' in predicate:
            return 2

        # Level 1: Basic provenance exists
        return 1

    def _compare_builder(self, baseline: Dict[str, Any], current: Dict[str, Any]) -> bool:
        """Compare builder identity."""
        baseline_builder = baseline.get('predicate', {}).get('builder', {}).get('id', '')
        current_builder = current.get('predicate', {}).get('builder', {}).get('id', '')
        return baseline_builder != current_builder

    def _compare_build_type(self, baseline: Dict[str, Any], current: Dict[str, Any]) -> bool:
        """Compare build type."""
        baseline_type = baseline.get('predicate', {}).get('buildType', '')
        current_type = current.get('predicate', {}).get('buildType', '')
        return baseline_type != current_type

    def _compare_materials(self, baseline: Dict[str, Any], current: Dict[str, Any]) -> List[SLSADelta]:
        """Compare resolved dependencies (materials)."""
        baseline_materials = baseline.get('predicate', {}).get('buildDefinition', {}).get('resolvedDependencies', [])
        current_materials = current.get('predicate', {}).get('buildDefinition', {}).get('resolvedDependencies', [])

        deltas = []
        baseline_uris = {m.get('uri', '') for m in baseline_materials}
        current_uris = {m.get('uri', '') for m in current_materials}

        for uri in current_uris - baseline_uris:
            deltas.append(SLSADelta(
                field='material',
                change_type=ChangeType.ADDED,
                old_value=None,
                new_value=uri,
                severity=Severity.LOW
            ))

        for uri in baseline_uris - current_uris:
            deltas.append(SLSADelta(
                field='material',
                change_type=ChangeType.REMOVED,
                old_value=uri,
                new_value=None,
                severity=Severity.MEDIUM
            ))

        return deltas

    def _compare_parameters(self, baseline: Dict[str, Any], current: Dict[str, Any]) -> List[SLSADelta]:
        """Compare build parameters."""
        baseline_params = baseline.get('predicate', {}).get('buildDefinition', {}).get('externalParameters', {})
        current_params = current.get('predicate', {}).get('buildDefinition', {}).get('externalParameters', {})

        deltas = []
        all_keys = set(baseline_params.keys()) | set(current_params.keys())

        for key in all_keys:
            baseline_val = baseline_params.get(key)
            current_val = current_params.get(key)

            if baseline_val != current_val:
                change_type = ChangeType.MODIFIED if (baseline_val and current_val) else (
                    ChangeType.ADDED if current_val else ChangeType.REMOVED
                )
                deltas.append(SLSADelta(
                    field=f'parameter.{key}',
                    change_type=change_type,
                    old_value=baseline_val,
                    new_value=current_val,
                    severity=Severity.LOW
                ))

        return deltas


# ============================================================================
# API SURFACE COMPARATOR
# ============================================================================

class APISurfaceComparator:
    """
    Compares API surface between releases to detect breaking changes.

    Detects:
    - Removed endpoints (BREAKING)
    - Changed HTTP methods (BREAKING)
    - Changed request/response schemas (BREAKING)
    - Added endpoints (NON-BREAKING)
    - Deprecated endpoints (WARNING)
    """

    BREAKING_PATTERNS = [
        r'removed',
        r'deleted',
        r'method.*changed',
        r'required.*parameter.*added',
        r'response.*type.*changed'
    ]

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.APISurfaceComparator")

    def compare(
        self,
        baseline_schema_path: Path,
        current_schema_path: Path
    ) -> APICompatibilityResult:
        """
        Compare API schemas (OpenAPI/Swagger format).

        Args:
            baseline_schema_path: Path to baseline API schema
            current_schema_path: Path to current API schema

        Returns:
            APICompatibilityResult
        """
        self.logger.info(f"Comparing API schemas: {baseline_schema_path.name} -> {current_schema_path.name}")

        baseline_api = self._load_api_schema(baseline_schema_path)
        current_api = self._load_api_schema(current_schema_path)

        baseline_endpoints = self._extract_endpoints(baseline_api)
        current_endpoints = self._extract_endpoints(current_api)

        breaking_changes = []
        non_breaking_changes = []
        additions = []
        deprecations = []

        # Detect removed endpoints (BREAKING)
        for endpoint_key in baseline_endpoints:
            if endpoint_key not in current_endpoints:
                endpoint_path, method = endpoint_key.split('::')
                breaking_changes.append(APIChange(
                    endpoint=endpoint_path,
                    method=method,
                    change_type=APIChangeType.BREAKING,
                    old_signature=baseline_endpoints[endpoint_key].get('summary', ''),
                    severity=Severity.CRITICAL,
                    details=f"Endpoint removed: {method} {endpoint_path}"
                ))

        # Detect added endpoints (NON-BREAKING)
        for endpoint_key in current_endpoints:
            if endpoint_key not in baseline_endpoints:
                endpoint_path, method = endpoint_key.split('::')
                additions.append(APIChange(
                    endpoint=endpoint_path,
                    method=method,
                    change_type=APIChangeType.ADDITION,
                    new_signature=current_endpoints[endpoint_key].get('summary', ''),
                    severity=Severity.INFO,
                    details=f"New endpoint added: {method} {endpoint_path}"
                ))

        # Detect modified endpoints
        for endpoint_key in baseline_endpoints:
            if endpoint_key in current_endpoints:
                baseline_def = baseline_endpoints[endpoint_key]
                current_def = current_endpoints[endpoint_key]

                changes = self._compare_endpoint_definition(baseline_def, current_def)
                endpoint_path, method = endpoint_key.split('::')

                for change in changes:
                    if change['breaking']:
                        breaking_changes.append(APIChange(
                            endpoint=endpoint_path,
                            method=method,
                            change_type=APIChangeType.BREAKING,
                            old_signature=str(change.get('old_value')),
                            new_signature=str(change.get('new_value')),
                            severity=Severity.HIGH,
                            details=change['description']
                        ))
                    else:
                        non_breaking_changes.append(APIChange(
                            endpoint=endpoint_path,
                            method=method,
                            change_type=APIChangeType.NON_BREAKING,
                            old_signature=str(change.get('old_value')),
                            new_signature=str(change.get('new_value')),
                            severity=Severity.LOW,
                            details=change['description']
                        ))

                # Check for deprecation
                if current_def.get('deprecated', False) and not baseline_def.get('deprecated', False):
                    deprecations.append(APIChange(
                        endpoint=endpoint_path,
                        method=method,
                        change_type=APIChangeType.DEPRECATION,
                        severity=Severity.MEDIUM,
                        details=f"Endpoint deprecated: {method} {endpoint_path}"
                    ))

        # Determine overall status
        if breaking_changes:
            status = "failed"
            details = f"Found {len(breaking_changes)} breaking API changes"
        elif deprecations:
            status = "warning"
            details = f"Found {len(deprecations)} deprecations"
        else:
            status = "passed"
            details = "No breaking API changes detected"

        return APICompatibilityResult(
            total_endpoints_baseline=len(baseline_endpoints),
            total_endpoints_current=len(current_endpoints),
            breaking_changes=breaking_changes,
            non_breaking_changes=non_breaking_changes,
            additions=additions,
            deprecations=deprecations,
            status=status,
            details=details
        )

    def _load_api_schema(self, schema_path: Path) -> Dict[str, Any]:
        """Load API schema (OpenAPI/Swagger format)."""
        if not schema_path.exists():
            raise BaselineMissingError(f"API schema not found: {schema_path}")

        with open(schema_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _extract_endpoints(self, schema: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Extract endpoints from OpenAPI schema."""
        endpoints = {}
        paths = schema.get('paths', {})

        for path, methods in paths.items():
            for method, definition in methods.items():
                if method.lower() in ('get', 'post', 'put', 'delete', 'patch', 'options', 'head'):
                    key = f"{path}::{method.upper()}"
                    endpoints[key] = definition

        return endpoints

    def _compare_endpoint_definition(
        self,
        baseline: Dict[str, Any],
        current: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Compare endpoint definitions for breaking/non-breaking changes."""
        changes = []

        # Check for added required parameters (BREAKING)
        baseline_params = {p['name']: p for p in baseline.get('parameters', [])}
        current_params = {p['name']: p for p in current.get('parameters', [])}

        for param_name, param in current_params.items():
            if param_name not in baseline_params and param.get('required', False):
                changes.append({
                    'breaking': True,
                    'description': f"Required parameter added: {param_name}",
                    'old_value': None,
                    'new_value': param_name
                })

        # Check for removed parameters (BREAKING if required)
        for param_name, param in baseline_params.items():
            if param_name not in current_params:
                breaking = param.get('required', False)
                changes.append({
                    'breaking': breaking,
                    'description': f"Parameter removed: {param_name}",
                    'old_value': param_name,
                    'new_value': None
                })

        # Check response schema changes (BREAKING if structure changed)
        baseline_responses = baseline.get('responses', {})
        current_responses = current.get('responses', {})

        for status_code in baseline_responses:
            if status_code in current_responses:
                baseline_schema = baseline_responses[status_code].get('schema', {})
                current_schema = current_responses[status_code].get('schema', {})

                if baseline_schema != current_schema:
                    changes.append({
                        'breaking': True,
                        'description': f"Response schema changed for {status_code}",
                        'old_value': baseline_schema,
                        'new_value': current_schema
                    })

        return changes


# ============================================================================
# PERFORMANCE DRIFT ANALYZER
# ============================================================================

class PerformanceDriftAnalyzer:
    """
    Analyzes performance drift between releases.

    Compares:
    - Response time percentiles (p50, p95, p99)
    - Throughput (requests/second)
    - Error rates
    - Resource utilization (CPU, memory)
    """

    DEFAULT_THRESHOLDS = {
        'response_time_p50': 10.0,      # 10% drift allowed
        'response_time_p95': 15.0,      # 15% drift allowed
        'response_time_p99': 20.0,      # 20% drift allowed
        'throughput': 10.0,             # 10% reduction allowed
        'error_rate': 5.0,              # 5% increase allowed
        'cpu_usage': 15.0,              # 15% increase allowed
        'memory_usage': 15.0            # 15% increase allowed
    }

    def __init__(self, custom_thresholds: Optional[Dict[str, float]] = None):
        self.logger = logging.getLogger(f"{__name__}.PerformanceDriftAnalyzer")
        self.thresholds = {**self.DEFAULT_THRESHOLDS, **(custom_thresholds or {})}

    def analyze(
        self,
        baseline_perf_path: Path,
        current_perf_path: Path
    ) -> PerformanceDriftResult:
        """
        Analyze performance drift.

        Args:
            baseline_perf_path: Path to baseline performance data (JSON)
            current_perf_path: Path to current performance data (JSON)

        Returns:
            PerformanceDriftResult
        """
        self.logger.info(f"Analyzing performance drift: {baseline_perf_path.name} -> {current_perf_path.name}")

        baseline_perf = self._load_performance_data(baseline_perf_path)
        current_perf = self._load_performance_data(current_perf_path)

        metrics = []
        exceeded_count = 0
        max_drift = 0.0

        for metric_name in baseline_perf:
            if metric_name not in current_perf:
                continue

            baseline_value = baseline_perf[metric_name]
            current_value = current_perf[metric_name]

            # Calculate drift percentage
            if baseline_value == 0:
                drift_percent = 0.0 if current_value == 0 else 100.0
            else:
                drift_percent = ((current_value - baseline_value) / baseline_value) * 100.0

            threshold = self.thresholds.get(metric_name, 10.0)

            # For throughput, negative drift is bad (reduction)
            if 'throughput' in metric_name.lower():
                exceeded = drift_percent < -threshold
            # For error rates and resource usage, positive drift is bad
            elif any(x in metric_name.lower() for x in ['error', 'cpu', 'memory']):
                exceeded = drift_percent > threshold
            # For response times, positive drift is bad
            else:
                exceeded = drift_percent > threshold

            severity = Severity.CRITICAL if exceeded and abs(drift_percent) > 30 else (
                Severity.HIGH if exceeded else Severity.INFO
            )

            metrics.append(PerformanceMetric(
                metric_name=metric_name,
                baseline_value=baseline_value,
                current_value=current_value,
                drift_percent=drift_percent,
                threshold_percent=threshold,
                exceeded=exceeded,
                severity=severity
            ))

            if exceeded:
                exceeded_count += 1

            max_drift = max(max_drift, abs(drift_percent))

        # Determine overall status
        if exceeded_count > 0:
            critical_count = sum(1 for m in metrics if m.severity == Severity.CRITICAL)
            if critical_count > 0:
                status = "failed"
                details = f"Critical performance regression: {critical_count} metrics exceeded thresholds"
            else:
                status = "warning"
                details = f"{exceeded_count} metrics exceeded thresholds"
        else:
            status = "passed"
            details = "No performance drift detected"

        return PerformanceDriftResult(
            metrics=metrics,
            exceeded_count=exceeded_count,
            max_drift_percent=max_drift,
            status=status,
            details=details
        )

    def _load_performance_data(self, perf_path: Path) -> Dict[str, float]:
        """Load performance data from JSON file."""
        if not perf_path.exists():
            raise BaselineMissingError(f"Performance data not found: {perf_path}")

        with open(perf_path, 'r', encoding='utf-8') as f:
            return json.load(f)


# ============================================================================
# SECURITY REGRESSION SCANNER
# ============================================================================

class SecurityRegressionScanner:
    """
    Scans for security regressions between releases.

    Analyzes:
    - Known vulnerability counts (CVE)
    - Security test failures
    - Permission/role changes
    - Encryption/auth configuration changes
    """

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.SecurityRegressionScanner")

    def scan(
        self,
        baseline_security_path: Path,
        current_security_path: Path
    ) -> SecurityRegressionResult:
        """
        Scan for security regressions.

        Args:
            baseline_security_path: Path to baseline security report (JSON)
            current_security_path: Path to current security report (JSON)

        Returns:
            SecurityRegressionResult
        """
        self.logger.info(f"Scanning security regression: {baseline_security_path.name} -> {current_security_path.name}")

        baseline_security = self._load_security_report(baseline_security_path)
        current_security = self._load_security_report(current_security_path)

        findings = []
        regressions_count = 0
        improvements_count = 0

        # Compare vulnerability counts by severity
        for severity_level in ['critical', 'high', 'medium', 'low']:
            baseline_count = baseline_security.get(f'{severity_level}_vulns', 0)
            current_count = current_security.get(f'{severity_level}_vulns', 0)
            delta = current_count - baseline_count

            if delta > 0:
                regressions_count += 1
            elif delta < 0:
                improvements_count += 1

            findings.append(SecurityFinding(
                finding_type=f'{severity_level}_vulnerabilities',
                severity=Severity[severity_level.upper()],
                baseline_count=baseline_count,
                current_count=current_count,
                delta=delta,
                regression=(delta > 0)
            ))

        # Compare security test results
        baseline_tests = baseline_security.get('security_tests_passed', 0)
        current_tests = current_security.get('security_tests_passed', 0)
        test_delta = current_tests - baseline_tests

        if test_delta < 0:
            regressions_count += 1
            findings.append(SecurityFinding(
                finding_type='security_tests',
                severity=Severity.HIGH,
                baseline_count=baseline_tests,
                current_count=current_tests,
                delta=test_delta,
                regression=True
            ))

        # Determine overall status
        if regressions_count > 0:
            critical_regressions = sum(1 for f in findings if f.regression and f.severity == Severity.CRITICAL)
            if critical_regressions > 0:
                status = "failed"
                details = f"Critical security regression: {critical_regressions} critical vulnerabilities added"
            else:
                status = "warning"
                details = f"{regressions_count} security regressions detected"
        else:
            status = "passed"
            details = f"No security regressions ({improvements_count} improvements)"

        return SecurityRegressionResult(
            findings=findings,
            regressions_count=regressions_count,
            improvements_count=improvements_count,
            status=status,
            details=details
        )

    def _load_security_report(self, security_path: Path) -> Dict[str, Any]:
        """Load security report from JSON file."""
        if not security_path.exists():
            raise BaselineMissingError(f"Security report not found: {security_path}")

        with open(security_path, 'r', encoding='utf-8') as f:
            return json.load(f)


# ============================================================================
# BEHAVIORAL REGRESSION CHECKER
# ============================================================================

class BehavioralRegressionChecker:
    """
    Checks for behavioral regressions in critical services.

    Runs:
    - Smoke tests against critical endpoints
    - Integration test comparison
    - Contract test validation
    """

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.BehavioralRegressionChecker")

    def check(
        self,
        baseline_behavior_path: Path,
        current_behavior_path: Path
    ) -> BehavioralRegressionResult:
        """
        Check for behavioral regressions.

        Args:
            baseline_behavior_path: Path to baseline behavioral test results
            current_behavior_path: Path to current behavioral test results

        Returns:
            BehavioralRegressionResult
        """
        self.logger.info(f"Checking behavioral regression: {baseline_behavior_path.name} -> {current_behavior_path.name}")

        baseline_tests = self._load_test_results(baseline_behavior_path)
        current_tests = self._load_test_results(current_behavior_path)

        tests = []
        regressions_count = 0

        # Compare test results
        for test_name in baseline_tests:
            baseline_result = baseline_tests[test_name]
            current_result = current_tests.get(test_name, False)

            regression = baseline_result and not current_result

            if regression:
                regressions_count += 1

            tests.append(BehavioralTest(
                test_name=test_name,
                baseline_result=baseline_result,
                current_result=current_result,
                regression=regression,
                details=f"Test {'PASSED' if current_result else 'FAILED'} (was {'PASSED' if baseline_result else 'FAILED'})"
            ))

        # Check for new tests
        for test_name in current_tests:
            if test_name not in baseline_tests:
                tests.append(BehavioralTest(
                    test_name=test_name,
                    baseline_result=False,
                    current_result=current_tests[test_name],
                    regression=False,
                    details="New test added"
                ))

        # Determine overall status
        if regressions_count > 0:
            status = "failed"
            details = f"{regressions_count} behavioral regressions detected"
        else:
            status = "passed"
            details = "No behavioral regressions detected"

        return BehavioralRegressionResult(
            tests=tests,
            total_tests=len(tests),
            regressions_count=regressions_count,
            status=status,
            details=details
        )

    def _load_test_results(self, test_path: Path) -> Dict[str, bool]:
        """Load test results from JSON file."""
        if not test_path.exists():
            raise BaselineMissingError(f"Test results not found: {test_path}")

        with open(test_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Expect format: {"test_name": true/false, ...}
        return data


# ============================================================================
# VALIDATION ORCHESTRATOR
# ============================================================================

class ValidationOrchestrator:
    """
    Orchestrates all validation checks and generates comprehensive report.

    Workflow:
    1. Run all enabled validation checks
    2. Aggregate results
    3. Apply policy gating (strict mode)
    4. Generate JSON + Markdown reports
    5. Return appropriate exit code
    """

    def __init__(
        self,
        mode: Literal['strict', 'lenient'] = 'strict',
        performance_thresholds: Optional[Dict[str, float]] = None
    ):
        self.mode = mode
        self.logger = logging.getLogger(f"{__name__}.ValidationOrchestrator")

        # Initialize subsystems
        self.sbom_analyzer = SBOMDeltaAnalyzer()
        self.slsa_analyzer = SLSADeltaAnalyzer()
        self.api_comparator = APISurfaceComparator()
        self.perf_analyzer = PerformanceDriftAnalyzer(performance_thresholds)
        self.security_scanner = SecurityRegressionScanner()
        self.behavior_checker = BehavioralRegressionChecker()

    def validate_release(
        self,
        version: str,
        baseline_version: str,
        baseline_sbom_path: Optional[Path] = None,
        current_sbom_path: Optional[Path] = None,
        baseline_slsa_path: Optional[Path] = None,
        current_slsa_path: Optional[Path] = None,
        baseline_api_schema_path: Optional[Path] = None,
        current_api_schema_path: Optional[Path] = None,
        baseline_perf_path: Optional[Path] = None,
        current_perf_path: Optional[Path] = None,
        baseline_security_path: Optional[Path] = None,
        current_security_path: Optional[Path] = None,
        baseline_behavior_path: Optional[Path] = None,
        current_behavior_path: Optional[Path] = None
    ) -> ValidationReport:
        """
        Perform complete release validation.

        Args:
            version: Current release version
            baseline_version: Baseline release version
            *_path: Paths to various validation data files

        Returns:
            ValidationReport
        """
        self.logger.info(f"Starting post-release validation: {baseline_version} -> {version}")
        start_time = datetime.now(timezone.utc)

        report = ValidationReport(
            version=version,
            baseline_version=baseline_version,
            timestamp=start_time.isoformat(),
            overall_status="passed"
        )

        failed_checks = 0
        warning_checks = 0
        passed_checks = 0

        # Run SBOM delta analysis
        if baseline_sbom_path and current_sbom_path:
            try:
                self.logger.info("Running SBOM delta analysis...")
                report.sbom_delta = self.sbom_analyzer.analyze(baseline_sbom_path, current_sbom_path)
                if report.sbom_delta.status == "failed":
                    failed_checks += 1
                elif report.sbom_delta.status == "warning":
                    warning_checks += 1
                else:
                    passed_checks += 1
            except Exception as e:
                self.logger.error(f"SBOM delta analysis failed: {e}")
                report.sbom_delta = None
                failed_checks += 1

        # Run SLSA delta analysis
        if baseline_slsa_path and current_slsa_path:
            try:
                self.logger.info("Running SLSA delta analysis...")
                report.slsa_delta = self.slsa_analyzer.analyze(baseline_slsa_path, current_slsa_path)
                if report.slsa_delta.status == "failed":
                    failed_checks += 1
                elif report.slsa_delta.status == "warning":
                    warning_checks += 1
                else:
                    passed_checks += 1
            except Exception as e:
                self.logger.error(f"SLSA delta analysis failed: {e}")
                report.slsa_delta = None
                failed_checks += 1

        # Run API compatibility check
        if baseline_api_schema_path and current_api_schema_path:
            try:
                self.logger.info("Running API compatibility check...")
                report.api_compatibility = self.api_comparator.compare(baseline_api_schema_path, current_api_schema_path)
                if report.api_compatibility.status == "failed":
                    failed_checks += 1
                elif report.api_compatibility.status == "warning":
                    warning_checks += 1
                else:
                    passed_checks += 1
            except Exception as e:
                self.logger.error(f"API compatibility check failed: {e}")
                report.api_compatibility = None
                failed_checks += 1

        # Run performance drift analysis
        if baseline_perf_path and current_perf_path:
            try:
                self.logger.info("Running performance drift analysis...")
                report.performance_drift = self.perf_analyzer.analyze(baseline_perf_path, current_perf_path)
                if report.performance_drift.status == "failed":
                    failed_checks += 1
                elif report.performance_drift.status == "warning":
                    warning_checks += 1
                else:
                    passed_checks += 1
            except Exception as e:
                self.logger.error(f"Performance drift analysis failed: {e}")
                report.performance_drift = None
                failed_checks += 1

        # Run security regression scan
        if baseline_security_path and current_security_path:
            try:
                self.logger.info("Running security regression scan...")
                report.security_regression = self.security_scanner.scan(baseline_security_path, current_security_path)
                if report.security_regression.status == "failed":
                    failed_checks += 1
                elif report.security_regression.status == "warning":
                    warning_checks += 1
                else:
                    passed_checks += 1
            except Exception as e:
                self.logger.error(f"Security regression scan failed: {e}")
                report.security_regression = None
                failed_checks += 1

        # Run behavioral regression check
        if baseline_behavior_path and current_behavior_path:
            try:
                self.logger.info("Running behavioral regression check...")
                report.behavioral_regression = self.behavior_checker.check(baseline_behavior_path, current_behavior_path)
                if report.behavioral_regression.status == "failed":
                    failed_checks += 1
                elif report.behavioral_regression.status == "warning":
                    warning_checks += 1
                else:
                    passed_checks += 1
            except Exception as e:
                self.logger.error(f"Behavioral regression check failed: {e}")
                report.behavioral_regression = None
                failed_checks += 1

        # Finalize report
        report.failed_checks = failed_checks
        report.warning_checks = warning_checks
        report.passed_checks = passed_checks
        report.total_checks = failed_checks + warning_checks + passed_checks

        end_time = datetime.now(timezone.utc)
        report.execution_time_seconds = (end_time - start_time).total_seconds()

        # Determine overall status and exit code
        if failed_checks > 0:
            report.overall_status = "failed"

            # Map to specific exit codes
            if report.behavioral_regression and report.behavioral_regression.status == "failed":
                report.exit_code = 20
            elif report.sbom_delta and report.sbom_delta.status == "failed":
                report.exit_code = 21
            elif report.slsa_delta and report.slsa_delta.status == "failed":
                report.exit_code = 22
            elif report.api_compatibility and report.api_compatibility.status == "failed":
                report.exit_code = 23
            elif report.performance_drift and report.performance_drift.status == "failed":
                report.exit_code = 24
            elif report.security_regression and report.security_regression.status == "failed":
                report.exit_code = 25
            else:
                report.exit_code = 29  # General validation error

            report.summary = f"Validation FAILED: {failed_checks} failures, {warning_checks} warnings"

            if self.mode == 'strict':
                self.logger.error(f"Strict mode: Release validation failed (exit {report.exit_code})")
        elif warning_checks > 0:
            report.overall_status = "warning"
            report.exit_code = 0 if self.mode == 'lenient' else 28
            report.summary = f"Validation passed with warnings: {warning_checks} warnings"
        else:
            report.overall_status = "passed"
            report.exit_code = 0
            report.summary = f"Validation PASSED: All {passed_checks} checks passed"

        self.logger.info(f"Validation complete: {report.summary} (took {report.execution_time_seconds:.2f}s)")

        return report

    def generate_json_report(self, report: ValidationReport, output_path: Path) -> None:
        """Generate JSON validation report."""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report.to_dict(), f, indent=2)
        self.logger.info(f"JSON report written to {output_path}")

    def generate_text_report(self, report: ValidationReport, output_path: Path) -> None:
        """Generate human-readable text validation report."""
        lines = [
            "=" * 80,
            "POST-RELEASE VALIDATION REPORT",
            "=" * 80,
            f"Version:          {report.version}",
            f"Baseline:         {report.baseline_version}",
            f"Timestamp:        {report.timestamp}",
            f"Overall Status:   {report.overall_status.upper()}",
            f"Exit Code:        {report.exit_code}",
            f"Execution Time:   {report.execution_time_seconds:.2f}s",
            "",
            f"Summary: {report.summary}",
            "",
            f"Checks: {report.passed_checks} passed, {report.warning_checks} warnings, {report.failed_checks} failed",
            "=" * 80,
            ""
        ]

        # SBOM Delta
        if report.sbom_delta:
            lines.extend([
                "SBOM DELTA ANALYSIS",
                "-" * 80,
                f"Status: {report.sbom_delta.status.upper()}",
                f"Components: {report.sbom_delta.total_components_baseline} -> {report.sbom_delta.total_components_current}",
                f"Added: {len(report.sbom_delta.added)}, Removed: {len(report.sbom_delta.removed)}, Modified: {len(report.sbom_delta.modified)}",
                f"Critical Changes: {len(report.sbom_delta.critical_changes)}",
                f"Details: {report.sbom_delta.details}",
                ""
            ])

        # SLSA Delta
        if report.slsa_delta:
            lines.extend([
                "SLSA PROVENANCE DELTA",
                "-" * 80,
                f"Status: {report.slsa_delta.status.upper()}",
                f"SLSA Level: {report.slsa_delta.level_baseline} -> {report.slsa_delta.level_current}",
                f"Builder Changed: {report.slsa_delta.builder_changed}",
                f"Build Type Changed: {report.slsa_delta.build_type_changed}",
                f"Details: {report.slsa_delta.details}",
                ""
            ])

        # API Compatibility
        if report.api_compatibility:
            lines.extend([
                "API COMPATIBILITY CHECK",
                "-" * 80,
                f"Status: {report.api_compatibility.status.upper()}",
                f"Endpoints: {report.api_compatibility.total_endpoints_baseline} -> {report.api_compatibility.total_endpoints_current}",
                f"Breaking Changes: {len(report.api_compatibility.breaking_changes)}",
                f"Additions: {len(report.api_compatibility.additions)}",
                f"Deprecations: {len(report.api_compatibility.deprecations)}",
                f"Details: {report.api_compatibility.details}",
                ""
            ])

        # Performance Drift
        if report.performance_drift:
            lines.extend([
                "PERFORMANCE DRIFT ANALYSIS",
                "-" * 80,
                f"Status: {report.performance_drift.status.upper()}",
                f"Metrics Analyzed: {len(report.performance_drift.metrics)}",
                f"Exceeded Thresholds: {report.performance_drift.exceeded_count}",
                f"Max Drift: {report.performance_drift.max_drift_percent:.2f}%",
                f"Details: {report.performance_drift.details}",
                ""
            ])

        # Security Regression
        if report.security_regression:
            lines.extend([
                "SECURITY REGRESSION SCAN",
                "-" * 80,
                f"Status: {report.security_regression.status.upper()}",
                f"Regressions: {report.security_regression.regressions_count}",
                f"Improvements: {report.security_regression.improvements_count}",
                f"Details: {report.security_regression.details}",
                ""
            ])

        # Behavioral Regression
        if report.behavioral_regression:
            lines.extend([
                "BEHAVIORAL REGRESSION CHECK",
                "-" * 80,
                f"Status: {report.behavioral_regression.status.upper()}",
                f"Total Tests: {report.behavioral_regression.total_tests}",
                f"Regressions: {report.behavioral_regression.regressions_count}",
                f"Details: {report.behavioral_regression.details}",
                ""
            ])

        lines.append("=" * 80)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

        self.logger.info(f"Text report written to {output_path}")


# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    """CLI entry point for standalone validation."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Post-Release Validation Suite (PRVS) for T.A.R.S.',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--version', required=True, help='Current release version')
    parser.add_argument('--baseline-version', required=True, help='Baseline release version')

    parser.add_argument('--baseline-sbom', type=Path, help='Baseline SBOM path')
    parser.add_argument('--current-sbom', type=Path, help='Current SBOM path')

    parser.add_argument('--baseline-slsa', type=Path, help='Baseline SLSA provenance path')
    parser.add_argument('--current-slsa', type=Path, help='Current SLSA provenance path')

    parser.add_argument('--baseline-api-schema', type=Path, help='Baseline API schema path')
    parser.add_argument('--current-api-schema', type=Path, help='Current API schema path')

    parser.add_argument('--baseline-perf', type=Path, help='Baseline performance data path')
    parser.add_argument('--current-perf', type=Path, help='Current performance data path')

    parser.add_argument('--baseline-security', type=Path, help='Baseline security report path')
    parser.add_argument('--current-security', type=Path, help='Current security report path')

    parser.add_argument('--baseline-behavior', type=Path, help='Baseline behavioral test results path')
    parser.add_argument('--current-behavior', type=Path, help='Current behavioral test results path')

    parser.add_argument('--policy', choices=['strict', 'lenient'], default='strict',
                       help='Policy enforcement mode (default: strict)')

    parser.add_argument('--json', type=Path, help='Output JSON report path')
    parser.add_argument('--text', type=Path, help='Output text report path')

    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')

    args = parser.parse_args()

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(levelname)s: %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    # Create orchestrator
    orchestrator = ValidationOrchestrator(mode=args.policy)

    # Run validation
    try:
        report = orchestrator.validate_release(
            version=args.version,
            baseline_version=args.baseline_version,
            baseline_sbom_path=args.baseline_sbom,
            current_sbom_path=args.current_sbom,
            baseline_slsa_path=args.baseline_slsa,
            current_slsa_path=args.current_slsa,
            baseline_api_schema_path=args.baseline_api_schema,
            current_api_schema_path=args.current_api_schema,
            baseline_perf_path=args.baseline_perf,
            current_perf_path=args.current_perf,
            baseline_security_path=args.baseline_security,
            current_security_path=args.current_security,
            baseline_behavior_path=args.baseline_behavior,
            current_behavior_path=args.current_behavior
        )

        # Generate reports
        if args.json:
            orchestrator.generate_json_report(report, args.json)

        if args.text:
            orchestrator.generate_text_report(report, args.text)

        # Print summary to console
        print(f"\n{report.summary}")
        print(f"Exit Code: {report.exit_code}")

        sys.exit(report.exit_code)

    except Exception as e:
        logger.error(f"Validation failed with error: {e}")
        sys.exit(29)


if __name__ == '__main__':
    main()
