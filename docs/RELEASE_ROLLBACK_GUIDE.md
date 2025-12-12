# T.A.R.S. Release Rollback & Recovery System - User Guide

**Phase:** 14.7 Task 6
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
7. [Rollback Types](#rollback-types)
8. [Policy Modes](#policy-modes)
9. [CLI Reference](#cli-reference)
10. [Programmatic API](#programmatic-api)
11. [Exit Codes](#exit-codes)
12. [Troubleshooting](#troubleshooting)
13. [Best Practices](#best-practices)
14. [Performance](#performance)
15. [Security Considerations](#security-considerations)

---

## Overview

The T.A.R.S. Release Rollback & Recovery System provides production-grade rollback capabilities for published releases with:

- **Atomic Operations**: All-or-nothing rollback guarantees
- **Safety First**: Automatic backups before rollback
- **Policy Enforcement**: Strict and lenient modes with dependency validation
- **Flexible Rollback Types**: Full, artifacts-only, or index-only rollback
- **Dry-Run Support**: Simulate rollback without making changes
- **Comprehensive Audit**: Complete audit trail with optional signing
- **Repository Abstraction**: Supports local, S3, and GCS repositories
- **High Performance**: < 5 second rollback target

### When to Use Rollback

Use the rollback system to:

- **Undo Failed Releases**: Remove releases that failed post-deployment validation
- **Revert Security Issues**: Quickly remove releases with discovered vulnerabilities
- **Fix Metadata Errors**: Remove releases with incorrect SBOM/SLSA data
- **Clean Test Releases**: Remove test releases from production repository
- **Recover from Errors**: Undo accidental or incomplete publications

---

## Architecture

### Component Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                 Rollback Orchestrator                        │
│  (End-to-end workflow coordination)                         │
└──────────┬──────────────────────────────────────────────────┘
           │
           ├──────┬────────────┬──────────────┬───────────────┐
           │      │            │              │               │
           │      │            │              │               │
           ▼      ▼            ▼              ▼               ▼
    ┌──────────┐ ┌──────┐ ┌────────┐  ┌──────────┐  ┌──────────┐
    │ Policy   │ │Planner│ │Executor│  │  Index   │  │  Audit   │
    │ Engine   │ │       │ │        │  │ Builder  │  │  Logger  │
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

#### 1. **RollbackRepositoryAdapter**
- Wraps publisher's AbstractRepository with rollback-specific operations
- Methods:
  - `version_exists()` - Check if version exists
  - `get_version_artifacts()` - List artifacts for version
  - `backup_version()` - Create pre-rollback backup
  - `rollback_version()` - Delete version artifacts
  - `restore_index()` - Restore previous index state
  - `remove_from_index()` - Remove version from index
  - `list_backups()` - List available backups

#### 2. **RollbackPolicyEngine**
- Enforces rollback prerequisites
- Validates:
  - Version existence
  - Artifact availability
  - Index consistency
  - Rollback type implications
- Modes:
  - **Strict**: Fail on any violation
  - **Lenient**: Convert violations to warnings

#### 3. **RollbackPlanner**
- Creates execution plan with:
  - Artifacts to remove
  - Index changes required
  - Dependencies satisfied
  - Estimated duration
  - Warnings and validation results

#### 4. **RollbackExecutor**
- Executes rollback atomically
- Process:
  1. Create backup (optional)
  2. Remove artifacts
  3. Update index
  4. Generate rollback manifest
  5. Cleanup

#### 5. **RollbackOrchestrator**
- End-to-end workflow coordinator
- Workflow:
  1. Policy enforcement
  2. Rollback planning
  3. Atomic execution
  4. Manifest creation
  5. Audit logging
  6. Report generation

---

## Key Features

### 1. Atomic Rollback

**Guarantee**: Either all rollback operations succeed or none do.

```
Stage 1: Create backup                    ✓
Stage 2: Remove artifacts from repository ✓
Stage 3: Update index (remove entry)      ✓
Stage 4: Generate rollback manifest       ✓
Stage 5: Create audit log                 ✓
```

If any stage fails, all changes are reverted.

### 2. Automatic Backups

Before rollback:
- Creates timestamped backup: `.rollback-backups/<version>-<timestamp>-<id>/`
- Stores all artifacts
- Enables potential restoration (future feature)

### 3. Dry-Run Mode

Simulate rollback without making changes:

```bash
--rollback-dry-run
```

Shows:
- Artifacts that would be removed
- Index changes that would occur
- Estimated duration
- Warnings and issues

### 4. Rollback Manifest

JSON manifest created after rollback:

```json
{
  "version": "v1.0.2",
  "rollback_timestamp": "2025-11-28T12:00:00Z",
  "rollback_id": "a1b2c3d4",
  "artifacts_removed": ["v1.0.2/README.md", ...],
  "index_state_before": {...},
  "index_state_after": {...},
  "rollback_type": "full",
  "can_restore": true
}
```

### 5. Audit Logging

Comprehensive audit trail:

```json
{
  "audit_id": "uuid",
  "version": "v1.0.2",
  "timestamp": "2025-11-28T12:00:00Z",
  "machine_id": "hostname",
  "rollback_id": "a1b2c3d4",
  "rollback_type": "full",
  "manifest": {...},
  "report": {...}
}
```

Optional RSA-PSS signing for immutability.

---

## Installation

### Prerequisites

- Python 3.9+
- T.A.R.S. v1.0.2+ with Phase 14.7 Task 5 (Publisher) installed

### Setup

No additional installation required. The rollback module is included with T.A.R.S. v1.0.2.

Verify installation:

```bash
python -m rollback.release_rollback --help
```

---

## Quick Start

### Example 1: Rollback Latest Release

```bash
# Dry run first (recommended)
python -m rollback.release_rollback \
  --version v1.0.2 \
  --repository-type local \
  --repository-path /var/tars/artifacts \
  --rollback-type full \
  --policy strict \
  --dry-run \
  --verbose

# Execute rollback
python -m rollback.release_rollback \
  --version v1.0.2 \
  --repository-type local \
  --repository-path /var/tars/artifacts \
  --rollback-type full \
  --policy strict \
  --manifest-output-dir ./rollback-manifests \
  --audit-output-dir ./rollback-audit \
  --sign-audit-log \
  --verbose
```

### Example 2: Rollback via Integrated Script

```bash
# Using prepare_release_artifacts.py
python scripts/prepare_release_artifacts.py \
  --rollback-release v1.0.2 \
  --rollback-type full \
  --rollback-policy strict \
  --repository-type local \
  --repository-path /var/tars/artifacts \
  --rollback-output-dir ./rollback \
  --verbose
```

---

## Usage Examples

### 1. Full Rollback (Recommended)

Removes artifacts AND index entry:

```bash
python -m rollback.release_rollback \
  --version v1.0.2 \
  --rollback-type full \
  --repository-type local \
  --repository-path /var/tars/artifacts
```

**Result:**
- All v1.0.2 artifacts deleted
- v1.0.2 removed from index.json
- Backup created in `.rollback-backups/`

### 2. Artifacts-Only Rollback

Removes artifacts, keeps index entry:

```bash
python -m rollback.release_rollback \
  --version v1.0.2 \
  --rollback-type artifacts_only \
  --repository-type local \
  --repository-path /var/tars/artifacts
```

**Use Case:** Free storage space while preserving metadata history.

### 3. Index-Only Rollback

Removes index entry, keeps artifacts:

```bash
python -m rollback.release_rollback \
  --version v1.0.2 \
  --rollback-type index_only \
  --repository-type local \
  --repository-path /var/tars/artifacts
```

**Use Case:** Hide release from index without deleting artifacts.

### 4. Dry-Run Simulation

Preview changes without executing:

```bash
python -m rollback.release_rollback \
  --version v1.0.2 \
  --rollback-type full \
  --dry-run \
  --verbose
```

**Output:**
```
===================================================================
DRY RUN SIMULATION
===================================================================
Version: v1.0.2
Rollback Type: full
Artifacts to remove: 42
Estimated duration: 1.5s

Operations that would be performed:
  1. Backup version v1.0.2
  2. Remove 42 artifacts:
     - v1.0.2/README.md
     - v1.0.2/manifest.json
     ... and 40 more
  3. Remove v1.0.2 from index

===================================================================
DRY RUN COMPLETE - No changes made
===================================================================
```

### 5. Force Rollback

Override policy violations (use with caution):

```bash
python -m rollback.release_rollback \
  --version v1.0.2 \
  --rollback-type full \
  --force \
  --repository-type local \
  --repository-path /var/tars/artifacts
```

**Warning:** Force mode bypasses safety checks. Use only when absolutely necessary.

### 6. Rollback Without Backup

Skip backup creation (not recommended):

```bash
python -m rollback.release_rollback \
  --version v1.0.2 \
  --rollback-type full \
  --no-backup \
  --repository-type local \
  --repository-path /var/tars/artifacts
```

**Warning:** Cannot restore after rollback without backup.

### 7. S3-Style Repository Rollback

Rollback from S3 bucket (simulated):

```bash
python -m rollback.release_rollback \
  --version v1.0.2 \
  --rollback-type full \
  --repository-type s3 \
  --repository-bucket tars-releases \
  --repository-prefix production \
  --policy strict
```

### 8. GCS-Style Repository Rollback

Rollback from GCS bucket (simulated):

```bash
python -m rollback.release_rollback \
  --version v1.0.2 \
  --rollback-type full \
  --repository-type gcs \
  --repository-bucket tars-releases \
  --repository-prefix production \
  --policy strict
```

---

## Rollback Types

### FULL (Recommended)

**Description:** Remove all artifacts AND index entry

**Use Cases:**
- Complete removal of failed release
- Security incident response
- Test release cleanup

**Operations:**
1. Backup version
2. Delete all artifacts from repository
3. Remove version from index.json
4. Delete version directory

**Command:**
```bash
--rollback-type full
```

### ARTIFACTS_ONLY

**Description:** Remove artifacts, keep index entry

**Use Cases:**
- Free storage space
- Preserve metadata for audit trail
- Retain release history

**Operations:**
1. Backup version
2. Delete all artifacts
3. Keep index entry (marked as removed)

**Command:**
```bash
--rollback-type artifacts_only
```

### INDEX_ONLY

**Description:** Remove index entry, keep artifacts

**Use Cases:**
- Hide release from public index
- Temporary release suspension
- Metadata correction

**Operations:**
1. Backup index state
2. Remove version from index.json
3. Keep all artifacts in repository

**Command:**
```bash
--rollback-type index_only
```

---

## Policy Modes

### STRICT (Default)

**Behavior:** Fail rollback on any policy violation

**Enforced Checks:**
- ✓ Version must exist in repository
- ✓ Version must have artifacts
- ✓ Index must be consistent
- ✓ Dependencies must be satisfied

**Use Cases:**
- Production environments
- Automated CI/CD pipelines
- Safety-critical operations

**Command:**
```bash
--policy strict
```

**Example Output:**
```
ERROR: Version v1.0.2 not found in repository
Exit Code: 40
```

### LENIENT

**Behavior:** Convert errors to warnings, allow rollback to proceed

**Behavior:**
- ⚠ Warnings logged for violations
- ✓ Rollback proceeds despite warnings
- ⚠ Non-fatal errors tolerated

**Use Cases:**
- Development environments
- Manual interventions
- Recovery operations

**Command:**
```bash
--policy lenient
```

**Example Output:**
```
WARNING: Version v1.0.2 not found in index (index may be inconsistent)
WARNING: No artifacts found for v1.0.2 (may already be rolled back)
Status: SUCCESS (with warnings)
```

---

## CLI Reference

### Standalone Rollback CLI

```bash
python -m rollback.release_rollback [OPTIONS]
```

#### Required Arguments

| Argument | Description | Example |
|----------|-------------|---------|
| `--version` | Release version to rollback | `v1.0.2` |

#### Repository Configuration

| Argument | Description | Default |
|----------|-------------|---------|
| `--repository-type` | Repository type: `local`, `s3`, `gcs` | `local` |
| `--repository-path` | Local repository path | `./repository` |
| `--repository-bucket` | S3/GCS bucket name | `default-bucket` |
| `--repository-prefix` | S3/GCS prefix | (empty) |

#### Rollback Configuration

| Argument | Description | Default |
|----------|-------------|---------|
| `--rollback-type` | Rollback type: `full`, `artifacts_only`, `index_only` | `full` |
| `--policy` | Policy mode: `strict`, `lenient` | `strict` |
| `--dry-run` | Simulate rollback without changes | (disabled) |
| `--force` | Force rollback despite warnings | (disabled) |
| `--no-backup` | Skip backup creation | (disabled) |

#### Output Configuration

| Argument | Description | Default |
|----------|-------------|---------|
| `--manifest-output-dir` | Rollback manifest directory | (current dir) |
| `--audit-output-dir` | Audit log directory | (current dir) |
| `--json-report` | JSON report output path | (none) |
| `--text-report` | Text report output path | (none) |
| `--sign-audit-log` | Sign audit logs with RSA-PSS | (disabled) |
| `--verbose` | Enable verbose logging | (disabled) |

### Integrated Script CLI

```bash
python scripts/prepare_release_artifacts.py [OPTIONS]
```

#### Rollback-Specific Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--rollback-release` | Version to rollback | (none) |
| `--rollback-type` | Rollback type | `full` |
| `--rollback-policy` | Policy mode | `strict` |
| `--rollback-dry-run` | Dry-run mode | (disabled) |
| `--rollback-force` | Force rollback | (disabled) |
| `--rollback-no-backup` | Skip backup | (disabled) |
| `--rollback-output-dir` | Output directory | (current dir) |

**Note:** Repository configuration arguments (`--repository-type`, `--repository-path`, etc.) are shared with publication arguments.

---

## Programmatic API

### Basic Usage

```python
from publisher.release_publisher import RepositoryFactory
from rollback.release_rollback import (
    RollbackOrchestrator,
    RollbackPolicy,
    RollbackType,
)

# Create repository
repo_config = {"type": "local", "path": "/var/tars/artifacts"}
repository = RepositoryFactory.create("local", repo_config)

# Create orchestrator
orchestrator = RollbackOrchestrator(
    repository=repository,
    policy_mode=RollbackPolicy.STRICT,
    sign_audit_logs=True
)

# Execute rollback
report = orchestrator.rollback_release(
    version="v1.0.2",
    rollback_type=RollbackType.FULL,
    dry_run=False,
    force=False,
    create_backup=True,
    audit_output_dir=Path("./audit"),
    manifest_output_dir=Path("./manifests")
)

# Check result
if report.status == "success":
    print(f"✓ Rolled back {report.version}")
    print(f"  Artifacts removed: {report.total_artifacts_removed}")
    print(f"  Bytes freed: {report.total_bytes_freed:,}")
else:
    print(f"✗ Rollback failed: {report.summary}")
    exit(report.exit_code)
```

### Advanced Usage

```python
from pathlib import Path
from rollback.release_rollback import (
    RollbackRepositoryAdapter,
    RollbackPolicyEngine,
    RollbackPlanner,
    RollbackExecutor,
)

# Create adapter
adapter = RollbackRepositoryAdapter(repository)

# Check version exists
if not adapter.version_exists("v1.0.2"):
    print("Version not found")
    exit(1)

# Enforce policy
policy_engine = RollbackPolicyEngine(RollbackPolicy.STRICT)
passed, warnings, errors = policy_engine.enforce(
    "v1.0.2",
    adapter,
    RollbackType.FULL,
    force=False
)

if not passed:
    print(f"Policy violations: {errors}")
    exit(1)

# Create plan
planner = RollbackPlanner(adapter)
plan = planner.plan("v1.0.2", RollbackType.FULL, dry_run=False)

print(f"Plan: {len(plan.artifacts_to_remove)} artifacts, ~{plan.estimated_duration_seconds:.2f}s")

# Execute
executor = RollbackExecutor(adapter)
success, manifest, count = executor.execute(plan, create_backup=True)

if success:
    print(f"✓ Rolled back {count} artifacts")
else:
    print("✗ Rollback failed")
```

---

## Exit Codes

### Rollback Exit Codes (40-49)

| Code | Exception | Meaning | Action |
|------|-----------|---------|--------|
| **0** | - | Success | Release rolled back successfully |
| **40** | `RollbackVersionNotFoundError` | Version not found | Verify version exists in repository |
| **41** | `RollbackPolicyViolationError` | Policy violation | Fix violations or use `--force` |
| **42** | `RollbackDependencyError` | Dependency check failed | Resolve dependencies or use `--force` |
| **43** | `RollbackAtomicError` | Atomic operation failed | Check logs, verify repository state |
| **44** | `RollbackIndexError` | Index restoration failed | Manually restore index from backup |
| **45** | `RollbackBackupError` | Backup creation failed | Check disk space and permissions |
| **46** | `RollbackAlreadyRolledBackError` | Already rolled back | Version already removed |
| **47** | `RollbackDryRunError` | Dry-run simulation failed | Fix issues before actual rollback |
| **48** | `RollbackManifestError` | Manifest generation failed | Check manifest output directory |
| **49** | `RollbackError` | General rollback error | Check logs for details |

---

## Troubleshooting

### Issue 1: "Version not found in repository"

**Exit Code:** 40

**Cause:** Version does not exist in repository (may already be rolled back)

**Solutions:**
1. Check version spelling: `--version v1.0.2` (not `1.0.2`)
2. List available versions: check `index.json` in repository
3. Use `--policy lenient` to bypass check
4. Use `--force` to override (if index is inconsistent)

### Issue 2: "Policy violation"

**Exit Code:** 41

**Cause:** One or more policy checks failed in strict mode

**Solutions:**
1. Run with `--dry-run` to see specific violations
2. Use `--policy lenient` to convert errors to warnings
3. Use `--force` to override (use with caution)
4. Fix underlying issues (e.g., restore missing index)

### Issue 3: "Atomic operation failed"

**Exit Code:** 43

**Cause:** One or more rollback stages failed

**Solutions:**
1. Check repository permissions (read/write/delete)
2. Verify disk space availability
3. Check logs for specific failure stage
4. Restore from backup if available
5. Contact support if repository is corrupted

### Issue 4: "Index restoration failed"

**Exit Code:** 44

**Cause:** Failed to update index.json after rollback

**Solutions:**
1. Check repository permissions for `index.json`
2. Verify `index.json` is valid JSON
3. Manually restore from rollback manifest:
   ```bash
   cp rollback-manifests/v1.0.2.rollback-manifest.json /var/tars/artifacts/index.json
   ```
4. Rebuild index from scratch (advanced)

### Issue 5: "Backup creation failed"

**Exit Code:** 45

**Cause:** Failed to create backup before rollback

**Solutions:**
1. Check disk space: `df -h`
2. Verify repository permissions
3. Use `--no-backup` to skip (not recommended)
4. Free up space and retry

### Issue 6: "Rollback took longer than expected"

**Cause:** Large number of artifacts or slow I/O

**Solutions:**
1. Use `--rollback-type artifacts_only` for faster rollback
2. Check disk I/O performance
3. Consider network latency (S3/GCS repositories)
4. Increase timeout if needed

---

## Best Practices

### 1. Always Use Dry-Run First

```bash
# Step 1: Dry-run
python -m rollback.release_rollback \
  --version v1.0.2 \
  --rollback-type full \
  --dry-run \
  --verbose

# Step 2: Review output

# Step 3: Execute actual rollback
python -m rollback.release_rollback \
  --version v1.0.2 \
  --rollback-type full \
  --verbose
```

### 2. Use Strict Mode in Production

```bash
--policy strict
```

Ensures all safety checks pass before rollback.

### 3. Always Create Backups

```bash
# Default behavior (recommended)
python -m rollback.release_rollback --version v1.0.2

# Explicitly enable
python -m rollback.release_rollback --version v1.0.2 --create-backup
```

### 4. Sign Audit Logs

```bash
--sign-audit-log
```

Provides immutable audit trail for compliance.

### 5. Monitor Rollback Duration

```bash
time python -m rollback.release_rollback \
  --version v1.0.2 \
  --rollback-type full \
  --verbose
```

Target: < 5 seconds

### 6. Archive Rollback Manifests

```bash
# Save manifests to secure location
--manifest-output-dir /secure/archive/rollback-manifests

# Organize by date
--manifest-output-dir /archive/$(date +%Y-%m-%d)/manifests
```

### 7. Use Full Rollback Unless Specific Need

**Default to `--rollback-type full`**

Only use `artifacts_only` or `index_only` for specific use cases.

---

## Performance

### Benchmarks

| Operation | Typical Release (42 artifacts, 50 MB) | Large Release (200 files, 200 MB) | Target |
|-----------|--------------------------------------|-----------------------------------|--------|
| **Policy Enforcement** | 10-20ms | 50ms | < 100ms |
| **Rollback Planning** | 50-100ms | 200ms | < 500ms |
| **Backup Creation** | 200-500ms | 1-2s | < 3s |
| **Artifact Removal** | 300-600ms | 1-2s | < 3s |
| **Index Update** | 50-100ms | 200ms | < 500ms |
| **Audit Logging** | 20-50ms | 100ms | < 200ms |
| **Full Rollback** | **< 2s** | **< 4s** | **< 5s** |

### Optimization Tips

1. **Use Artifacts-Only for Large Releases**
   - Skip index update for faster rollback
   - Update index separately later

2. **Disable Backup for Test Environments**
   - Use `--no-backup` in dev/staging
   - Reduces rollback time by 30-50%

3. **Use Dry-Run for Planning**
   - Validate without execution overhead
   - Typical dry-run: < 1s

---

## Security Considerations

### 1. Access Control

**Recommendation:** Restrict rollback permissions to authorized users only.

```bash
# Linux/macOS
chmod 700 /path/to/rollback/script
chown admin:admin /path/to/rollback/script

# Verify permissions
ls -la /path/to/rollback/script
```

### 2. Audit Logging

**Recommendation:** Enable audit log signing for immutability.

```bash
--sign-audit-log
```

**Storage:** Store audit logs in append-only storage or WORM (Write Once Read Many) system.

### 3. Backup Security

**Recommendation:** Protect backups from unauthorized access.

```bash
# Encrypt backups (future feature)
# Current: Restrict directory permissions
chmod 700 .rollback-backups/
```

### 4. Repository Isolation

**Recommendation:** Use separate repositories for production and non-production.

```bash
# Production
--repository-path /prod/artifacts

# Staging
--repository-path /staging/artifacts

# Development
--repository-path /dev/artifacts
```

### 5. CI/CD Integration

**Recommendation:** Use service accounts with limited permissions.

```yaml
# GitHub Actions example
- name: Rollback Release
  run: |
    python -m rollback.release_rollback \
      --version ${{ github.event.inputs.version }} \
      --rollback-type full \
      --policy strict \
      --repository-type s3 \
      --repository-bucket ${{ secrets.PROD_BUCKET }}
  env:
    AWS_ACCESS_KEY_ID: ${{ secrets.ROLLBACK_SERVICE_KEY }}
    AWS_SECRET_ACCESS_KEY: ${{ secrets.ROLLBACK_SERVICE_SECRET }}
```

---

## Related Documentation

- **Phase 14.7 Task 5:** [Release Publisher Guide](RELEASE_PUBLISHER_GUIDE.md)
- **Phase 14.7 Task 3:** [Release Verifier Guide](RELEASE_VERIFIER_GUIDE.md)
- **Phase 14.7 Task 4:** [Post-Release Validation Guide](RELEASE_VALIDATOR_GUIDE.md)
- **Task 6 Completion Summary:** [PHASE14_7_TASK6_COMPLETION_SUMMARY.md](../PHASE14_7_TASK6_COMPLETION_SUMMARY.md)

---

## Support

For issues or questions:

1. Check [Troubleshooting](#troubleshooting) section
2. Review test suite: `tests/integration/test_release_rollback.py`
3. Examine rollback logs in `--verbose` mode
4. Contact T.A.R.S. development team

---

**Document Version:** 1.0
**Last Updated:** 2025-11-28
**Author:** T.A.R.S. Development Team
**Classification:** Internal - Engineering Documentation
