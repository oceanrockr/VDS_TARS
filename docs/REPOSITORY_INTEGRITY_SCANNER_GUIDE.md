# T.A.R.S. Repository Integrity Scanner & Consistency Verifier - User Guide

**Phase:** 14.7 Task 7
**Version:** 1.0.0
**Date:** 2025-11-28
**Status:** Production-Ready

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Key Features](#key-features)
4. [Installation](#installation)
5. [Quick Start](#quick-start)
6. [Usage Examples](#usage-examples)
7. [Scan Policy Modes](#scan-policy-modes)
8. [Repair Actions](#repair-actions)
9. [CLI Reference](#cli-reference)
10. [Programmatic API](#programmatic-api)
11. [Exit Codes](#exit-codes)
12. [Troubleshooting](#troubleshooting)
13. [Best Practices](#best-practices)
14. [Performance](#performance)
15. [Security Considerations](#security-considerations)

---

## Overview

The T.A.R.S. Repository Integrity Scanner provides production-grade validation and consistency verification for release artifact repositories with:

- **Artifact Integrity Validation**: SHA256 hash verification for all artifacts
- **Manifest Cross-Validation**: Detect corrupted, mismatched, missing artifacts
- **Index Consistency**: Ensure index.json accurately reflects repository state
- **SBOM/SLSA Validation**: Verify presence and format of security metadata
- **Orphan Detection**: Identify artifacts not referenced by any version
- **Automated Repairs**: Safe-mode repairs with rollback protection
- **Policy Enforcement**: Strict, lenient, or audit-only modes
- **Comprehensive Reporting**: JSON and text reports with actionable insights

### When to Use the Scanner

Use the integrity scanner to:

- **Detect Corruption**: Find artifacts with hash mismatches or data corruption
- **Validate Repository**: Ensure repository consistency before/after operations
- **Audit Compliance**: Verify SBOM/SLSA metadata presence and validity
- **Clean Repository**: Identify and remove orphaned artifacts
- **Troubleshoot Issues**: Diagnose publication or rollback failures
- **Monitor Health**: Track repository health metrics over time

---

## Architecture

### Component Diagram

```
┌─────────────────────────────────────────────────────────────┐
│              IntegrityScanOrchestrator                       │
│         (End-to-end scan coordination)                       │
└──────────┬──────────────────────────────────────────────────┘
           │
           ├──────┬────────────┬──────────────┬───────────────┐
           │      │            │              │               │
           │      │            │              │               │
           ▼      ▼            ▼              ▼               ▼
    ┌──────────┐ ┌──────┐ ┌────────┐  ┌──────────┐  ┌──────────┐
    │ Policy   │ │Scanner│ │ Repair │  │  Report  │  │Repository│
    │ Engine   │ │       │ │ Engine │  │ Builder  │  │ Adapter  │
    └──────────┘ └───────┘ └────────┘  └──────────┘  └──────────┘
           │                    │              │              │
           └────────────────────┴──────────────┴──────────────┘
                                │
                                ▼
                   ┌────────────────────────┐
                   │ Repository Adapter     │
                   │ (Local/S3/GCS)        │
                   └────────────────────────┘
```

### Subsystems

#### 1. **IntegrityRepositoryAdapter**
- Wraps publisher's AbstractRepository with integrity-specific operations
- Methods:
  - `list_all_versions()` - List all versions
  - `get_version_artifacts()` - List artifacts for version
  - `compute_sha256()` - Compute artifact hash
  - `get_manifest()` - Get version manifest
  - `get_sbom()` / `get_slsa()` - Get security metadata
  - `artifact_exists()` - Check artifact existence
  - `delete_artifact()` - Delete artifact (for repairs)

#### 2. **IntegrityScanPolicyEngine**
- Enforces scan policies
- Validates:
  - Issue severity thresholds
  - Fail vs. warn behavior
  - Exit code determination
- Modes:
  - **Strict**: Fail on critical/error issues
  - **Lenient**: Fail only on critical issues
  - **Audit Only**: Never fail, only report

#### 3. **IntegrityScanner**
- Core validation logic
- Scans:
  - **Artifacts**: Hash verification, existence checks
  - **Manifests**: Structure validation, cross-validation
  - **SBOM/SLSA**: Presence, format, required fields
  - **Index**: Consistency with repository state
  - **Orphans**: Artifacts not referenced by any version
- Detects 18 types of integrity issues

#### 4. **IntegrityRepairEngine**
- Safe-mode repair operations
- Actions:
  - Remove orphan artifacts
  - Rebuild index.json from repository state
  - Fix individual index entries (add/remove)
- All repairs are:
  - Optional (disabled by default)
  - Non-destructive (backups recommended)
  - Atomic (all-or-nothing)

#### 5. **IntegrityReportBuilder**
- Generates comprehensive reports
- Formats:
  - **JSON**: Machine-readable, complete data
  - **Text**: Human-readable summary
- Includes:
  - Repository health metrics
  - Issue breakdown by type/severity
  - Repair results
  - Actionable recommendations

---

## Key Features

### 1. Comprehensive Validation

**Artifact Integrity**:
- SHA256 hash verification
- Size validation
- Existence checks
- Signature validation (when present)

**Manifest Validation**:
- Structure validation
- Required field checks
- Artifact list cross-validation
- Hash mismatch detection

**Index Consistency**:
- Versions in index vs. repository
- Missing/extra entries
- Malformed index detection
- Ordering validation

**SBOM/SLSA Validation**:
- Presence verification
- Format validation (CycloneDX, SPDX, SLSA)
- Required field checks
- Hash verification

### 2. Multi-Mode Policy Enforcement

**Strict Mode** (Default):
- Fail on critical or error issues
- Recommended for CI/CD pipelines
- Exit code reflects highest severity issue

**Lenient Mode**:
- Fail only on critical issues
- Convert errors to warnings
- Continue scan despite issues

**Audit Only Mode**:
- Never fail
- Report all issues
- Exit code 50 (success) regardless

### 3. Automated Repairs

**Orphan Removal**:
- Identify artifacts not in any version
- Safe deletion with confirmation
- Frees repository storage

**Index Rebuilding**:
- Reconstruct index.json from repository
- Preserves version metadata
- Atomic update

**Index Entry Fixes**:
- Add missing version entries
- Remove invalid entries
- Update metadata

### 4. Rich Reporting

**JSON Report**:
```json
{
  "scan_id": "abc123",
  "timestamp": "2025-11-28T12:00:00Z",
  "total_issues": 5,
  "critical_issues": 0,
  "error_issues": 2,
  "warning_issues": 3,
  "versions": [...],
  "all_issues": [...],
  "scan_duration_seconds": 1.23
}
```

**Text Report**:
```
T.A.R.S. REPOSITORY INTEGRITY SCAN REPORT
Status: SUCCESS_WITH_WARNINGS
Total Versions: 10
Total Artifacts: 420
Issues: 5 (0 critical, 2 errors, 3 warnings)
```

---

## Installation

### Prerequisites

- Python 3.9+
- T.A.R.S. v1.0.2+ with Phase 14.7 Task 5 (Publisher) installed

### Setup

No additional installation required. The integrity scanner is included with T.A.R.S. v1.0.2.

Verify installation:

```bash
python -m integrity.repository_integrity_scanner --help
```

---

## Quick Start

### Example 1: Basic Repository Scan

```bash
# Scan local repository
python -m integrity.repository_integrity_scanner \
  --repository-type local \
  --repository-path /var/tars/artifacts \
  --policy strict \
  --output-dir ./integrity-scan \
  --verbose
```

### Example 2: Scan with Repairs

```bash
# Scan and auto-repair issues
python -m integrity.repository_integrity_scanner \
  --repository-type local \
  --repository-path /var/tars/artifacts \
  --policy lenient \
  --repair \
  --repair-orphans \
  --repair-index \
  --output-dir ./integrity-scan \
  --verbose
```

### Example 3: Scan via Integrated Script

```bash
# Using prepare_release_artifacts.py
python scripts/prepare_release_artifacts.py \
  --scan-repository \
  --scan-policy strict \
  --repository-type local \
  --repository-path /var/tars/artifacts \
  --scan-output-dir ./integrity-scan \
  --verbose
```

---

## Usage Examples

### 1. Strict Mode Scan (Recommended for CI/CD)

Fail on any critical or error issues:

```bash
python -m integrity.repository_integrity_scanner \
  --repository-type local \
  --repository-path /var/tars/artifacts \
  --policy strict \
  --output-dir ./integrity-scan
```

**Result:**
- Exit code 50: No issues
- Exit code 52-59: Issues found (see [Exit Codes](#exit-codes))

### 2. Lenient Mode Scan

Fail only on critical issues:

```bash
python -m integrity.repository_integrity_scanner \
  --repository-type local \
  --repository-path /var/tars/artifacts \
  --policy lenient \
  --output-dir ./integrity-scan
```

**Use Case:** Development environments where some warnings are acceptable.

### 3. Audit-Only Mode

Never fail, only report:

```bash
python -m integrity.repository_integrity_scanner \
  --repository-type local \
  --repository-path /var/tars/artifacts \
  --policy audit_only \
  --output-dir ./integrity-scan
```

**Use Case:** Regular health monitoring without blocking operations.

### 4. Scan with Orphan Removal

Remove orphaned artifacts:

```bash
python -m integrity.repository_integrity_scanner \
  --repository-type local \
  --repository-path /var/tars/artifacts \
  --policy lenient \
  --repair \
  --repair-orphans \
  --output-dir ./integrity-scan
```

**Warning:** Orphans will be permanently deleted. Review scan report before enabling repair.

### 5. Scan with Index Rebuild

Rebuild index.json from repository state:

```bash
python -m integrity.repository_integrity_scanner \
  --repository-type local \
  --repository-path /var/tars/artifacts \
  --policy lenient \
  --repair \
  --repair-index \
  --output-dir ./integrity-scan
```

**Use Case:** Index.json corrupted or missing.

### 6. S3-Style Repository Scan

Scan S3 bucket (simulated):

```bash
python -m integrity.repository_integrity_scanner \
  --repository-type s3 \
  --repository-bucket tars-releases \
  --repository-prefix production \
  --policy strict \
  --output-dir ./integrity-scan
```

### 7. GCS-Style Repository Scan

Scan GCS bucket (simulated):

```bash
python -m integrity.repository_integrity_scanner \
  --repository-type gcs \
  --repository-bucket tars-releases \
  --repository-prefix production \
  --policy strict \
  --output-dir ./integrity-scan
```

### 8. Custom Report Paths

Specify custom report locations:

```bash
python -m integrity.repository_integrity_scanner \
  --repository-type local \
  --repository-path /var/tars/artifacts \
  --policy strict \
  --json-report /var/reports/integrity-$(date +%Y%m%d).json \
  --text-report /var/reports/integrity-$(date +%Y%m%d).txt
```

---

## Scan Policy Modes

### STRICT (Default)

**Behavior:** Fail on critical or error issues

**Enforced Checks:**
- ✓ All artifacts must exist
- ✓ All artifact hashes must match manifest
- ✓ All manifests must be valid
- ✓ All versions must be in index
- ✓ SBOM/SLSA must be present and valid

**Use Cases:**
- Production environments
- CI/CD pipelines
- Pre-deployment validation

**Command:**
```bash
--policy strict
```

**Exit Codes:**
- 50: Success (no issues)
- 52-59: Issues found (scan fails)

---

### LENIENT

**Behavior:** Fail only on critical issues, convert errors to warnings

**Behavior:**
- ⚠ Missing artifacts → Warning
- ⚠ SBOM/SLSA missing → Warning
- ✓ Corrupted artifacts → Still fail (critical)
- ✓ Index inconsistent → Still fail (critical)

**Use Cases:**
- Development environments
- Migration operations
- Gradual cleanup

**Command:**
```bash
--policy lenient
```

**Exit Codes:**
- 50: Success (no critical issues)
- 51: Success with warnings
- 52-53: Critical issues found (scan fails)

---

### AUDIT_ONLY

**Behavior:** Never fail, only report issues

**Behavior:**
- ℹ All issues reported
- ℹ Exit code always 50 (success)
- ℹ No impact on workflow

**Use Cases:**
- Regular health monitoring
- Trend analysis
- Non-blocking scans

**Command:**
```bash
--policy audit_only
```

**Exit Codes:**
- Always 50 (success)

---

## Repair Actions

### REMOVE_ORPHAN

**Description:** Delete artifact not referenced by any version

**Safety:**
- ✓ Non-destructive to valid releases
- ⚠ Orphan permanently deleted
- ℹ Frees repository storage

**Command:**
```bash
--repair --repair-orphans
```

**Example:**
```
Found orphan: old-file.txt (1024 bytes)
Action: Removed successfully
```

---

### REBUILD_INDEX_JSON

**Description:** Reconstruct index.json from repository state

**Safety:**
- ✓ Atomic update
- ⚠ Replaces existing index
- ℹ Preserves version metadata

**Command:**
```bash
--repair --repair-index
```

**Example:**
```
Rebuilt index.json with 10 releases
Added: v1.0.0, v1.0.1, v1.0.2, ...
```

---

### FIX_INDEX_ENTRY

**Description:** Add or remove individual index entries

**Safety:**
- ✓ Targeted fix
- ✓ Preserves other entries
- ℹ Updates single version

**Command:**
```bash
--repair --repair-index
```

**Example:**
```
Fixed index entry for v1.0.2 (add)
Fixed index entry for v0.9.0 (remove)
```

---

## CLI Reference

### Standalone Scanner CLI

```bash
python -m integrity.repository_integrity_scanner [OPTIONS]
```

#### Repository Configuration

| Argument | Description | Default |
|----------|-------------|---------|
| `--repository-type` | Repository type: `local`, `s3`, `gcs` | `local` |
| `--repository-path` | Local repository path | `./repository` |
| `--repository-bucket` | S3/GCS bucket name | `default-bucket` |
| `--repository-prefix` | S3/GCS prefix | (empty) |

#### Scan Configuration

| Argument | Description | Default |
|----------|-------------|---------|
| `--policy` | Policy mode: `strict`, `lenient`, `audit_only` | `strict` |
| `--repair` | Enable repair mode | (disabled) |
| `--repair-orphans` | Enable orphan removal | (disabled) |
| `--repair-index` | Enable index rebuilding | (disabled) |

#### Output Configuration

| Argument | Description | Default |
|----------|-------------|---------|
| `--output-dir` | Output directory for reports | `./integrity-scan` |
| `--json-report` | JSON report path | `{output-dir}/integrity-scan-report.json` |
| `--text-report` | Text report path | `{output-dir}/integrity-scan-report.txt` |
| `--verbose` | Verbose output | (disabled) |

### Integrated Script CLI

```bash
python scripts/prepare_release_artifacts.py [OPTIONS]
```

#### Scanner-Specific Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--scan-repository` | Run integrity scan | (disabled) |
| `--scan-policy` | Policy mode | `strict` |
| `--scan-repair` | Enable repair mode | (disabled) |
| `--scan-repair-orphans` | Enable orphan removal | (disabled) |
| `--scan-repair-index` | Enable index rebuilding | (disabled) |
| `--scan-output-dir` | Output directory | `./integrity-scan` |

**Note:** Repository configuration arguments (`--repository-type`, `--repository-path`, etc.) are shared with publication/rollback arguments.

---

## Programmatic API

### Basic Usage

```python
from publisher.release_publisher import RepositoryFactory
from integrity.repository_integrity_scanner import (
    IntegrityScanOrchestrator,
    IntegrityScanPolicy,
)

# Create repository
repo_config = {"type": "local", "path": "/var/tars/artifacts"}
repository = RepositoryFactory.create("local", repo_config)

# Create orchestrator
orchestrator = IntegrityScanOrchestrator(
    repository=repository,
    policy_mode=IntegrityScanPolicy.STRICT,
    repair_enabled=False
)

# Run scan
from pathlib import Path
output_dir = Path("./integrity-scan")
json_path = output_dir / "report.json"
text_path = output_dir / "report.txt"

report = orchestrator.scan_repository(
    output_dir=output_dir,
    json_report_path=json_path,
    text_report_path=text_path
)

# Check result
if report.scan_status == "success":
    print(f"✓ Repository healthy")
    print(f"  Versions: {report.total_versions}")
    print(f"  Artifacts: {report.total_artifacts}")
else:
    print(f"✗ Issues found: {report.total_issues}")
    print(f"  Critical: {report.critical_issues}")
    print(f"  Errors: {report.error_issues}")
    exit(report.exit_code)
```

### Advanced Usage

```python
from pathlib import Path
from integrity.repository_integrity_scanner import (
    IntegrityRepositoryAdapter,
    IntegrityScanPolicyEngine,
    IntegrityScanner,
    IntegrityRepairEngine,
    IntegrityScanPolicy,
)

# Create adapter
adapter = IntegrityRepositoryAdapter(repository)

# List versions
versions = adapter.list_all_versions()
print(f"Found {len(versions)} versions")

# Create scanner
policy_engine = IntegrityScanPolicyEngine(IntegrityScanPolicy.STRICT)
scanner = IntegrityScanner(adapter, policy_engine)

# Scan specific version
validation = scanner.scan_version("v1.0.2")
print(f"Version {validation.version}:")
print(f"  Manifest valid: {validation.manifest_valid}")
print(f"  SBOM valid: {validation.sbom_valid}")
print(f"  Issues: {len(validation.issues)}")

# Apply repairs if needed
if validation.issues:
    repair_engine = IntegrityRepairEngine(adapter)
    for issue in validation.issues:
        if issue.can_auto_repair:
            # Apply repair based on repair_action
            pass
```

---

## Exit Codes

### Integrity Scanner Exit Codes (50-59)

| Code | Meaning | Action |
|------|---------|--------|
| **50** | Success (no issues) | Repository is healthy |
| **51** | Success with warnings | Review warnings, no critical issues |
| **52** | Artifact integrity failure | Corrupted artifacts detected |
| **53** | Manifest mismatch | Manifest doesn't match artifacts |
| **54** | Index inconsistency | Index.json inconsistent with repository |
| **55** | SBOM/SLSA issues | Security metadata missing/invalid |
| **56** | Orphan artifacts detected | Unreferenced artifacts found |
| **57** | Signature validation failure | Signature verification failed |
| **58** | Repair required/not performed | Issues detected, repairs not applied |
| **59** | General integrity error | Scan failed due to unexpected error |

---

## Troubleshooting

### Issue 1: "Artifact corrupted (hash mismatch)"

**Exit Code:** 52

**Cause:** Artifact SHA256 doesn't match manifest

**Solutions:**
1. Verify artifact wasn't modified after publication
2. Check for disk corruption: `fsck` (Linux) or `chkdsk` (Windows)
3. Re-publish the release if corruption confirmed
4. Check backup/restore integrity if recently restored

### Issue 2: "Index inconsistent with repository"

**Exit Code:** 54

**Cause:** index.json doesn't match actual versions in repository

**Solutions:**
1. Run scan with `--repair --repair-index` to rebuild index
2. Manually verify versions in repository vs. index
3. Check for incomplete publication or rollback operations
4. Review recent repository operations

### Issue 3: "Orphan artifacts detected"

**Exit Code:** 56

**Cause:** Artifacts exist but not referenced by any version

**Solutions:**
1. Review orphan list in scan report
2. Determine if orphans are from failed operations
3. Run with `--repair --repair-orphans` to remove
4. Manually delete orphans if specific artifacts identified

### Issue 4: "SBOM/SLSA missing"

**Exit Code:** 55

**Cause:** Security metadata missing for one or more versions

**Solutions:**
1. Use `--policy lenient` if SBOM/SLSA not required
2. Re-publish releases with `--include-sbom --include-slsa`
3. Manually generate SBOM/SLSA for affected versions
4. Update index to mark versions as not requiring SBOM/SLSA

### Issue 5: "Manifest missing or malformed"

**Exit Code:** 53

**Cause:** manifest.json missing or has invalid structure

**Solutions:**
1. Check if version was fully published
2. Manually create/fix manifest.json
3. Re-publish the version
4. Rollback and re-publish if corruption severe

### Issue 6: "Scan timeout or performance issues"

**Cause:** Large repository or slow I/O

**Solutions:**
1. Use `--policy audit_only` for faster scans (no validation overhead)
2. Scan specific versions instead of entire repository (future feature)
3. Check disk I/O performance: `iostat` (Linux) or Task Manager (Windows)
4. Consider network latency for S3/GCS repositories

---

## Best Practices

### 1. Schedule Regular Scans

```bash
# Daily cron job (Linux)
0 2 * * * python -m integrity.repository_integrity_scanner \
  --repository-type local \
  --repository-path /var/tars/artifacts \
  --policy audit_only \
  --output-dir /var/logs/integrity-scans/$(date +\%Y\%m\%d)
```

### 2. Use Strict Mode in CI/CD

```yaml
# GitHub Actions example
- name: Scan Repository Integrity
  run: |
    python -m integrity.repository_integrity_scanner \
      --repository-type s3 \
      --repository-bucket ${{ secrets.RELEASE_BUCKET }} \
      --policy strict \
      --output-dir ./integrity-scan
  env:
    AWS_ACCESS_KEY_ID: ${{ secrets.AWS_KEY }}
    AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET }}
```

### 3. Review Scan Reports Before Repairs

```bash
# Step 1: Scan without repair
python -m integrity.repository_integrity_scanner \
  --repository-type local \
  --repository-path /var/tars/artifacts \
  --policy lenient \
  --output-dir ./integrity-scan

# Step 2: Review report
cat ./integrity-scan/integrity-scan-report.txt

# Step 3: Apply repairs if appropriate
python -m integrity.repository_integrity_scanner \
  --repository-type local \
  --repository-path /var/tars/artifacts \
  --policy lenient \
  --repair \
  --repair-orphans \
  --repair-index \
  --output-dir ./integrity-scan-repair
```

### 4. Archive Scan Reports

```bash
# Archive reports with timestamp
--output-dir /archive/integrity-scans/$(date +%Y-%m-%d-%H%M%S)
```

### 5. Use Lenient Mode for Gradual Cleanup

```bash
# Phase 1: Identify issues
--policy audit_only

# Phase 2: Fix critical issues
--policy lenient --repair

# Phase 3: Enforce strict compliance
--policy strict
```

### 6. Monitor Repository Health Metrics

Track over time:
- Total versions
- Total artifacts
- Total size
- Issue trends
- Repair frequency

---

## Performance

### Benchmarks

| Operation | Typical Repo (10 versions, 420 artifacts, 500 MB) | Large Repo (100 versions, 4200 files, 5 GB) | Target |
|-----------|--------------------------------------------------|---------------------------------------------|--------|
| **Repository Scan** | 1-2s | 8-10s | < 5s (typical) |
| **Artifact Hash Verification** | 500-800ms | 4-5s | < 3s |
| **Index Consistency Check** | 50-100ms | 200ms | < 500ms |
| **SBOM/SLSA Validation** | 100-200ms | 500ms | < 1s |
| **Orphan Detection** | 100-200ms | 500ms | < 1s |
| **Repair Operations** | 200-500ms | 1-2s | < 3s |
| **Full Scan (no repair)** | **< 2s** | **< 10s** | **< 5s** |

### Optimization Tips

1. **Use Audit-Only for Monitoring**
   - Skip validation overhead
   - 30-50% faster than strict mode

2. **Disable Repairs for Read-Only Scans**
   - No repair overhead
   - Pure validation only

3. **Target Specific Policy Modes**
   - Strict: Full validation
   - Lenient: Reduced validation
   - Audit: Minimal validation

---

## Security Considerations

### 1. Access Control

**Recommendation:** Restrict scanner permissions to read-only unless repairs enabled.

```bash
# Linux/macOS
chmod 700 /path/to/scanner/script
chown integrity-user:integrity-group /path/to/scanner/script
```

### 2. Repair Permissions

**Recommendation:** Require elevated permissions for repair operations.

```bash
# Only allow repairs with explicit approval
--repair --repair-orphans  # Requires approval
```

### 3. Audit Logging

**Recommendation:** Log all scan and repair operations.

```bash
# Example: Log to syslog
python -m integrity.repository_integrity_scanner \
  --repository-type local \
  --repository-path /var/tars/artifacts \
  --policy strict 2>&1 | logger -t integrity-scan
```

### 4. Repository Isolation

**Recommendation:** Use separate repositories for production and non-production.

```bash
# Production
--repository-path /prod/artifacts

# Staging
--repository-path /staging/artifacts
```

### 5. CI/CD Integration

**Recommendation:** Use service accounts with limited permissions.

```yaml
# GitHub Actions example
- name: Scan Repository
  run: |
    python -m integrity.repository_integrity_scanner \
      --repository-type s3 \
      --repository-bucket ${{ secrets.PROD_BUCKET }} \
      --policy strict
  env:
    AWS_ACCESS_KEY_ID: ${{ secrets.SCANNER_SERVICE_KEY }}
    AWS_SECRET_ACCESS_KEY: ${{ secrets.SCANNER_SERVICE_SECRET }}
```

---

## Related Documentation

- **Phase 14.7 Task 5:** [Release Publisher Guide](RELEASE_PUBLISHER_GUIDE.md)
- **Phase 14.7 Task 6:** [Release Rollback Guide](RELEASE_ROLLBACK_GUIDE.md)
- **Phase 14.7 Task 3:** [Release Verifier Guide](RELEASE_VERIFIER_GUIDE.md)
- **Phase 14.7 Task 4:** [Post-Release Validation Guide](POST_RELEASE_VALIDATION_GUIDE.md)
- **Task 7 Completion Summary:** [PHASE14_7_TASK7_COMPLETION_SUMMARY.md](../PHASE14_7_TASK7_COMPLETION_SUMMARY.md)

---

## Support

For issues or questions:

1. Check [Troubleshooting](#troubleshooting) section
2. Review test suite: `tests/integration/test_repository_integrity_scanner.py`
3. Examine scan logs with `--verbose` mode
4. Contact T.A.R.S. development team

---

**Document Version:** 1.0
**Last Updated:** 2025-11-28
**Author:** T.A.R.S. Development Team
**Classification:** Internal - Engineering Documentation
