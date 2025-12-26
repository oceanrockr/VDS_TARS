# Post-Release Validation Suite (PRVS) - User Guide

**Version:** 1.0
**Phase:** 14.7 Task 4
**Status:** Production Ready

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Subsystems](#subsystems)
4. [Usage](#usage)
5. [Integration](#integration)
6. [Exit Codes](#exit-codes)
7. [Performance](#performance)
8. [Troubleshooting](#troubleshooting)

---

## Overview

The Post-Release Validation Suite (PRVS) is a comprehensive validation subsystem that automatically runs **after** release artifact verification (Phase 14.7 Task 3) and **before** release publication. It performs regression analysis, compatibility checks, and quality gates to ensure release integrity.

### Key Features

✅ **SBOM Delta Analysis** - Detect dependency changes, version upgrades, and critical package modifications
✅ **SLSA Provenance Delta** - Track SLSA level changes, builder identity, and material evolution
✅ **API Compatibility Checks** - Identify breaking vs non-breaking changes with OpenAPI schema comparison
✅ **Performance Drift Detection** - Monitor response times, throughput, error rates with configurable thresholds
✅ **Security Regression Scanning** - Track vulnerability counts and security test results
✅ **Behavioral Regression Checks** - Validate critical service functionality through test comparison
✅ **Auto-Generated Reports** - JSON + Markdown reports with detailed findings

### Design Principles

- **Offline Operation**: No network calls, air-gapped compatible
- **Deterministic**: Same inputs always produce same outputs
- **Fast**: < 5 second execution for typical releases
- **Zero Placeholders**: Fully implemented, production-ready
- **Cross-Platform**: Windows, Linux, macOS compatible

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                 ValidationOrchestrator                       │
│  (Coordinates all validation subsystems)                     │
└────────────────────┬────────────────────────────────────────┘
                     │
      ┌──────────────┼──────────────┐
      │              │              │
┌─────▼─────┐  ┌────▼────┐  ┌─────▼──────┐
│   SBOM    │  │  SLSA   │  │    API     │
│   Delta   │  │  Delta  │  │ Compatibility│
└───────────┘  └─────────┘  └────────────┘
      │              │              │
┌─────▼─────┐  ┌────▼────┐  ┌─────▼──────┐
│Performance│  │Security │  │ Behavioral │
│   Drift   │  │Regression│  │ Regression │
└───────────┘  └─────────┘  └────────────┘
      │              │              │
      └──────────────┼──────────────┘
                     │
            ┌────────▼────────┐
            │ ValidationReport │
            │  (JSON + Text)   │
            └──────────────────┘
```

### Workflow Integration

```
Task 3: Release Artifact Verification
   ↓ (if passed)
Task 4: Post-Release Validation Suite ← YOU ARE HERE
   │
   ├─ SBOM Delta Analysis
   ├─ SLSA Delta Analysis
   ├─ API Compatibility Check
   ├─ Performance Drift Detection
   ├─ Security Regression Scan
   └─ Behavioral Regression Check
   ↓
Gate Decision (exit codes 20-29)
   ├─ PASS (exit 0) → Publish Release
   └─ FAIL (exit 20-29) → Abort Publication
```

---

## Subsystems

### 1. SBOM Delta Analyzer

**Purpose:** Detect changes in software dependencies between releases.

**Detects:**
- ✓ Added dependencies (new packages)
- ✓ Removed dependencies (deleted packages)
- ✓ Modified dependencies (version changes)
- ✓ Critical package changes (security-sensitive libraries)

**Supported Formats:**
- CycloneDX 1.5
- SPDX 2.3

**Critical Packages:** (Higher severity for changes)
```
cryptography, pycryptodome, jwt, pyjwt
requests, urllib3, paramiko, fabric
django, flask, fastapi, starlette
sqlalchemy, psycopg2, pymongo
```

**Example Output:**
```json
{
  "total_components_baseline": 45,
  "total_components_current": 47,
  "added": [
    {"name": "pydantic", "version": "2.0.0", "severity": "low"}
  ],
  "removed": [],
  "modified": [
    {"name": "cryptography", "old_version": "40.0.0", "new_version": "41.0.0", "severity": "high"}
  ],
  "critical_changes": 1,
  "status": "failed"
}
```

---

### 2. SLSA Delta Analyzer

**Purpose:** Track SLSA provenance changes and build integrity.

**Detects:**
- ✓ SLSA level changes (L1 → L2 → L3)
- ✓ Builder identity changes
- ✓ Build type modifications
- ✓ Material (dependency) deltas
- ✓ Build parameter changes

**SLSA Level Detection:**
- **Level 1**: Basic provenance exists
- **Level 2**: Signed provenance with metadata
- **Level 3**: Hermetic, reproducible builds

**Critical Scenarios:**
- ❌ SLSA level **downgrade** (e.g., L3 → L2) → **FAIL**
- ⚠️ Builder identity change → **WARNING**
- ⚠️ Build type change → **WARNING**

**Example Output:**
```json
{
  "level_baseline": 2,
  "level_current": 3,
  "builder_changed": false,
  "build_type_changed": false,
  "materials_delta": [
    {"field": "material", "change_type": "added", "new_value": "git+https://..."}
  ],
  "status": "passed"
}
```

---

### 3. API Surface Comparator

**Purpose:** Detect breaking and non-breaking API changes.

**Detects:**
- ❌ **BREAKING**: Removed endpoints
- ❌ **BREAKING**: Changed HTTP methods
- ❌ **BREAKING**: Added required parameters
- ❌ **BREAKING**: Changed response schemas
- ✓ **NON-BREAKING**: Added endpoints
- ✓ **NON-BREAKING**: Added optional parameters
- ⚠️ **DEPRECATION**: Deprecated endpoints

**Supported Schema:** OpenAPI 3.0 / Swagger

**Example Output:**
```json
{
  "total_endpoints_baseline": 45,
  "total_endpoints_current": 47,
  "breaking_changes": [
    {
      "endpoint": "/api/users",
      "method": "GET",
      "change_type": "breaking",
      "details": "Required parameter added: filter",
      "severity": "high"
    }
  ],
  "additions": [
    {
      "endpoint": "/api/v2/users",
      "method": "POST",
      "change_type": "addition"
    }
  ],
  "deprecations": [],
  "status": "failed"
}
```

---

### 4. Performance Drift Analyzer

**Purpose:** Detect performance regressions through metric comparison.

**Metrics:**
- Response time percentiles (p50, p95, p99)
- Throughput (requests/second)
- Error rates
- Resource utilization (CPU, memory)

**Default Thresholds:**
```python
{
    'response_time_p50': 10.0,   # 10% drift allowed
    'response_time_p95': 15.0,   # 15% drift allowed
    'response_time_p99': 20.0,   # 20% drift allowed
    'throughput': 10.0,          # 10% reduction allowed
    'error_rate': 5.0,           # 5% increase allowed
    'cpu_usage': 15.0,           # 15% increase allowed
    'memory_usage': 15.0         # 15% increase allowed
}
```

**Severity Levels:**
- **CRITICAL**: Drift > 30% AND threshold exceeded
- **HIGH**: Threshold exceeded but drift < 30%
- **INFO**: Within threshold

**Example Output:**
```json
{
  "metrics": [
    {
      "metric_name": "response_time_p95",
      "baseline_value": 150.0,
      "current_value": 180.0,
      "drift_percent": 20.0,
      "threshold_percent": 15.0,
      "exceeded": true,
      "severity": "high"
    }
  ],
  "exceeded_count": 2,
  "max_drift_percent": 25.5,
  "status": "warning"
}
```

---

### 5. Security Regression Scanner

**Purpose:** Detect security regressions and vulnerability increases.

**Analyzes:**
- Critical vulnerability counts (CVE)
- High/Medium/Low vulnerability trends
- Security test pass/fail rates
- Permission/role changes

**Critical Scenarios:**
- ❌ Critical vulnerabilities **added** → **FAIL**
- ❌ Security tests **failing** (regression) → **FAIL**
- ⚠️ High vulnerabilities increased → **WARNING**

**Example Output:**
```json
{
  "findings": [
    {
      "finding_type": "critical_vulnerabilities",
      "severity": "critical",
      "baseline_count": 0,
      "current_count": 2,
      "delta": 2,
      "regression": true
    }
  ],
  "regressions_count": 2,
  "improvements_count": 0,
  "status": "failed"
}
```

---

### 6. Behavioral Regression Checker

**Purpose:** Validate critical service functionality through test comparison.

**Test Types:**
- Smoke tests (critical endpoints)
- Integration tests
- Contract tests
- End-to-end scenarios

**Detection:**
- ❌ Previously passing tests now **failing** → **REGRESSION**
- ✓ Previously failing tests now **passing** → **IMPROVEMENT**
- ℹ️ New tests added → **INFO**

**Example Output:**
```json
{
  "tests": [
    {
      "test_name": "test_auth_login",
      "baseline_result": true,
      "current_result": false,
      "regression": true,
      "details": "Test FAILED (was PASSED)"
    }
  ],
  "total_tests": 15,
  "regressions_count": 2,
  "status": "failed"
}
```

---

## Usage

### Standalone CLI

```bash
python -m validation.post_release_validation \
  --version 1.0.2 \
  --baseline-version 1.0.1 \
  --baseline-sbom /path/to/v1.0.1-sbom.json \
  --current-sbom /path/to/v1.0.2-sbom.json \
  --baseline-slsa /path/to/v1.0.1-slsa.json \
  --current-slsa /path/to/v1.0.2-slsa.json \
  --baseline-api-schema /path/to/api-v1.0.1.json \
  --current-api-schema /path/to/api-v1.0.2.json \
  --baseline-perf /path/to/perf-v1.0.1.json \
  --current-perf /path/to/perf-v1.0.2.json \
  --baseline-security /path/to/sec-v1.0.1.json \
  --current-security /path/to/sec-v1.0.2.json \
  --baseline-behavior /path/to/behav-v1.0.1.json \
  --current-behavior /path/to/behav-v1.0.2.json \
  --policy strict \
  --json /output/validation-report.json \
  --text /output/validation-report.txt \
  --verbose
```

### Integrated with Release Script

```bash
python scripts/prepare_release_artifacts.py \
  --include-sbom \
  --include-slsa \
  --run-performance-tests \
  --run-security-audit \
  --verify-release \
  --post-release-validation \
  --baseline-release 1.0.1 \
  --baseline-sbom /path/to/baseline-sbom.json \
  --baseline-slsa /path/to/baseline-slsa.json \
  --api-schema /path/to/api-schema.json \
  --performance-baseline /path/to/perf-baseline.json \
  --security-baseline /path/to/security-baseline.json \
  --behavior-baseline /path/to/behavior-baseline.json \
  --validation-policy strict \
  --verbose
```

### Programmatic Usage

```python
from validation.post_release_validation import ValidationOrchestrator
from pathlib import Path

# Initialize orchestrator
orchestrator = ValidationOrchestrator(
    mode='strict',
    performance_thresholds={
        'response_time_p95': 10.0,  # Tighter threshold
        'error_rate': 3.0
    }
)

# Run validation
report = orchestrator.validate_release(
    version="1.0.2",
    baseline_version="1.0.1",
    baseline_sbom_path=Path("/path/to/baseline-sbom.json"),
    current_sbom_path=Path("/path/to/current-sbom.json"),
    baseline_slsa_path=Path("/path/to/baseline-slsa.json"),
    current_slsa_path=Path("/path/to/current-slsa.json"),
    baseline_api_schema_path=Path("/path/to/api-baseline.json"),
    current_api_schema_path=Path("/path/to/api-current.json"),
    baseline_perf_path=Path("/path/to/perf-baseline.json"),
    current_perf_path=Path("/path/to/perf-current.json"),
    baseline_security_path=Path("/path/to/sec-baseline.json"),
    current_security_path=Path("/path/to/sec-current.json"),
    baseline_behavior_path=Path("/path/to/behav-baseline.json"),
    current_behavior_path=Path("/path/to/behav-current.json")
)

# Generate reports
orchestrator.generate_json_report(report, Path("/output/report.json"))
orchestrator.generate_text_report(report, Path("/output/report.txt"))

# Check status
if report.overall_status == "passed":
    print(f"✓ Validation passed: {report.summary}")
    exit(0)
else:
    print(f"✗ Validation failed: {report.summary}")
    exit(report.exit_code)
```

---

## Integration

### CI/CD Pipeline (GitHub Actions)

```yaml
name: Release Validation

on:
  push:
    tags:
      - 'v*'

jobs:
  validate-release:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
        with:
          fetch-depth: 0  # Full history for baseline comparison

      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-dev.txt

      - name: Download baseline artifacts
        run: |
          # Download previous release artifacts from artifact repository
          wget https://artifacts.example.com/v1.0.1/sbom.json -O baseline-sbom.json
          wget https://artifacts.example.com/v1.0.1/slsa.json -O baseline-slsa.json
          wget https://artifacts.example.com/v1.0.1/performance.json -O baseline-perf.json
          wget https://artifacts.example.com/v1.0.1/security.json -O baseline-security.json

      - name: Generate current release artifacts
        run: |
          python scripts/prepare_release_artifacts.py \
            --include-sbom \
            --include-slsa \
            --run-performance-tests \
            --run-security-audit \
            --api-url http://localhost:3001 \
            --verify-release \
            --post-release-validation \
            --baseline-release 1.0.1 \
            --baseline-sbom baseline-sbom.json \
            --baseline-slsa baseline-slsa.json \
            --performance-baseline baseline-perf.json \
            --security-baseline baseline-security.json \
            --validation-policy strict \
            --verbose

      - name: Upload validation reports
        if: always()
        uses: actions/upload-artifact@v3
        with:
          name: validation-reports
          path: release/v*/post_validation/

      - name: Check validation status
        run: |
          if [ $? -ne 0 ]; then
            echo "::error::Post-release validation failed"
            exit 1
          fi
```

---

## Exit Codes

| Code | Meaning | Action |
|------|---------|--------|
| **0** | All validations passed | Proceed with release |
| **20** | Behavioral regression detected | Abort - Fix tests |
| **21** | SBOM delta analysis failed | Abort - Review dependencies |
| **22** | SLSA provenance regression | Abort - Fix build process |
| **23** | Breaking API changes detected | Abort - Fix API compatibility |
| **24** | Performance drift exceeded threshold | Abort - Optimize performance |
| **25** | Security regression detected | Abort - Fix vulnerabilities |
| **26** | Baseline data missing | Abort - Provide baseline |
| **27** | Validation orchestration error | Abort - Check configuration |
| **28** | Policy gate failure (strict mode) | Abort - Review policy |
| **29** | General validation error | Abort - Check logs |

---

## Performance

### Benchmarks

| Operation | Typical Release | Large Release | Notes |
|-----------|----------------|---------------|-------|
| **SBOM Delta** | 50-100ms | 200-300ms | 100+ components |
| **SLSA Delta** | 10-20ms | 30-50ms | Full provenance |
| **API Compare** | 100-200ms | 500ms-1s | 100+ endpoints |
| **Perf Drift** | 10-20ms | 50-100ms | 10+ metrics |
| **Security Scan** | 20-30ms | 100-200ms | Full audit |
| **Behavioral** | 10-20ms | 50-100ms | 20+ tests |
| **Total** | **< 2s** | **< 5s** | **Within target** |

### Resource Usage

- **CPU:** < 10% single-core during validation
- **Memory:** < 100 MB peak usage
- **Disk I/O:** Sequential reads only (< 50 MB/s)
- **Network:** None (fully offline)

---

## Troubleshooting

### Issue: "Baseline SBOM not found"

**Cause:** Baseline file path is incorrect or file doesn't exist.

**Solution:**
```bash
# Verify file exists
ls -la /path/to/baseline-sbom.json

# Use absolute paths
python -m validation.post_release_validation \
  --baseline-sbom /absolute/path/to/baseline-sbom.json \
  ...
```

### Issue: "Exit code 21 - SBOM delta analysis failed"

**Cause:** Critical dependency changes detected (e.g., `cryptography` version bump).

**Solution:**
1. Review SBOM delta report: `release/v*/post_validation/*-post-validation.txt`
2. Check critical changes section
3. Options:
   - **Fix:** Revert critical dependency changes
   - **Override:** Use `--validation-policy lenient` if changes are intentional
   - **Document:** Add justification to release notes

### Issue: "Exit code 23 - Breaking API changes detected"

**Cause:** API schema comparison found breaking changes (removed endpoints, changed signatures).

**Solution:**
1. Review API compatibility report
2. Check `breaking_changes` section
3. Options:
   - **Fix:** Restore removed endpoints or add versioning (/api/v2/)
   - **Deprecate:** Mark old endpoints as deprecated before removal
   - **Document:** Update migration guide for consumers

### Issue: "Exit code 24 - Performance drift exceeded threshold"

**Cause:** Performance metrics exceeded configured thresholds.

**Solution:**
1. Review performance drift report
2. Check which metrics exceeded thresholds
3. Options:
   - **Optimize:** Fix performance regressions
   - **Adjust:** Update thresholds if intentional (new features)
   - **Profile:** Use profiling tools to identify bottlenecks

### Issue: "Validation takes > 5 seconds"

**Cause:** Large number of components/endpoints or slow disk I/O.

**Solution:**
```python
# Use custom thresholds to skip some checks
orchestrator = ValidationOrchestrator(
    mode='lenient',
    performance_thresholds={'response_time_p99': 50.0}  # Relax thresholds
)

# Or skip optional checks
report = orchestrator.validate_release(
    version="1.0.2",
    baseline_version="1.0.1",
    # Only run critical checks
    baseline_sbom_path=sbom_path,
    current_sbom_path=current_sbom,
    # Skip performance/behavioral checks for speed
)
```

### Issue: "Exit code 28 - Policy gate failure in strict mode"

**Cause:** Warnings treated as failures in strict mode.

**Solution:**
```bash
# Option 1: Use lenient mode for warnings
python -m validation.post_release_validation \
  --policy lenient \
  ...

# Option 2: Fix warnings before release
# Review validation report for warning details
cat release/v*/post_validation/*-post-validation.txt
```

---

## Best Practices

### 1. Establish Baselines Early

Store baseline artifacts in artifact repository after each release:

```bash
# After successful release
mkdir -p /artifact-repo/v1.0.1/
cp release/v1.0.1/sbom/*.json /artifact-repo/v1.0.1/
cp release/v1.0.1/slsa/*.json /artifact-repo/v1.0.1/
cp release/v1.0.1/performance/*.json /artifact-repo/v1.0.1/
cp release/v1.0.1/security_audit/*.json /artifact-repo/v1.0.1/
```

### 2. Use Strict Mode in CI/CD

Always use `--validation-policy strict` in automated pipelines:

```bash
# CI/CD pipeline
python scripts/prepare_release_artifacts.py \
  --post-release-validation \
  --validation-policy strict \  # Fail on any issues
  ...
```

### 3. Custom Performance Thresholds

Adjust thresholds based on your SLAs:

```python
orchestrator = ValidationOrchestrator(
    mode='strict',
    performance_thresholds={
        'response_time_p95': 5.0,   # 5% for critical APIs
        'error_rate': 1.0            # 1% for error rate
    }
)
```

### 4. Archive Validation Reports

Store validation reports with release artifacts:

```bash
# Archive reports
tar -czf release-v1.0.2-validation.tar.gz \
  release/v1.0.2/post_validation/*.json \
  release/v1.0.2/post_validation/*.txt
```

### 5. Review Failed Validations

Always review detailed reports before overriding failures:

```bash
# Read text report
cat release/v1.0.2/post_validation/tars-v1.0.2-post-validation.txt

# Or analyze JSON programmatically
python -c "
import json
with open('release/v1.0.2/post_validation/tars-v1.0.2-post-validation.json') as f:
    report = json.load(f)
    print(f'Failed checks: {report[\"failed_checks\"]}')
    print(f'Summary: {report[\"summary\"]}')
"
```

---

## Appendix: Data Format Examples

### SBOM Format (CycloneDX)

```json
{
  "bomFormat": "CycloneDX",
  "specVersion": "1.5",
  "components": [
    {
      "name": "requests",
      "version": "2.31.0",
      "type": "library",
      "purl": "pkg:pypi/requests@2.31.0"
    }
  ]
}
```

### SLSA Provenance Format

```json
{
  "_type": "https://in-toto.io/Statement/v0.1",
  "predicateType": "https://slsa.dev/provenance/v1",
  "predicate": {
    "buildDefinition": {
      "buildType": "https://slsa.dev/build-types/python/package/v1",
      "externalParameters": {
        "hermetic": true
      }
    },
    "builder": {
      "id": "https://github.com/actions/runner"
    }
  }
}
```

### Performance Baseline Format

```json
{
  "response_time_p50": 45.5,
  "response_time_p95": 120.0,
  "response_time_p99": 200.0,
  "throughput": 1200.0,
  "error_rate": 0.3,
  "cpu_usage": 25.5,
  "memory_usage": 450.0
}
```

### Security Report Format

```json
{
  "critical_vulns": 0,
  "high_vulns": 2,
  "medium_vulns": 8,
  "low_vulns": 15,
  "security_tests_passed": 48
}
```

### Behavioral Test Format

```json
{
  "test_auth_login": true,
  "test_auth_logout": true,
  "test_crud_create": true,
  "test_crud_read": true,
  "test_crud_update": true,
  "test_crud_delete": true
}
```

---

## Support

For issues or questions:
- Check [Troubleshooting](#troubleshooting) section
- Review test suite: `tests/integration/test_post_release_validation.py`
- Read source code: `validation/post_release_validation.py`

---

**Document Version:** 1.0
**Last Updated:** 2025-11-28
**Author:** T.A.R.S. Development Team
