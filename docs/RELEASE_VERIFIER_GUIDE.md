# T.A.R.S. Release Artifact Verifier - User Guide

**Version:** 1.0.2
**Last Updated:** 2025-11-28
**Audience:** DevOps Engineers, Release Managers, Security Teams

---

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Architecture](#architecture)
4. [Usage Patterns](#usage-patterns)
5. [Integration](#integration)
6. [Policy Configuration](#policy-configuration)
7. [Troubleshooting](#troubleshooting)
8. [API Reference](#api-reference)

---

## Overview

### Purpose

The T.A.R.S. Release Artifact Verifier is a comprehensive integrity gate system that validates all release artifacts before acceptance into the artifact repository. It ensures supply chain security by verifying:

- **SBOM Integrity:** CycloneDX and SPDX format validation
- **SLSA Provenance:** Supply-chain attestation verification
- **Cryptographic Signatures:** RSA-PSS signature validation
- **Hash Correctness:** SHA-256 and SHA-512 integrity checks
- **Manifest Synchronization:** Artifact inventory validation
- **Policy Compliance:** Configurable enforcement rules

### Key Features

- **Comprehensive Verification:** All aspects of release artifacts
- **Deterministic Output:** Same artifacts → same results
- **Offline Capable:** No network dependencies
- **Cross-Platform:** Windows, Linux, macOS
- **CI/CD Ready:** Exit codes for automation
- **Flexible Policies:** Strict and lenient modes

### Exit Codes

| Code | Status | CI/CD Action |
|------|--------|--------------|
| 0 | ✅ All checks passed | Continue pipeline |
| 1 | ❌ Artifact not found | Abort |
| 2 | ❌ SBOM validation failed | Abort |
| 3 | ❌ SLSA validation failed | Abort |
| 4 | ❌ Signature verification failed | Abort |
| 5 | ❌ Hash verification failed | Abort |
| 6 | ❌ Manifest verification failed | Abort |
| 7 | ❌ Policy gate failure | Abort |
| 8 | ❌ General error | Abort |

---

## Quick Start

### Prerequisites

```bash
# Python 3.9+ required
python --version

# Install dependencies
pip install -r requirements.txt

# For signature verification
pip install cryptography
```

### Basic Verification

```bash
# Verify single artifact
python security/release_verifier.py \
  --artifact dist/tars-v1.0.2.tar.gz \
  --version 1.0.2 \
  --policy lenient \
  --verbose
```

**Expected Output:**
```
INFO: Starting release verification for dist/tars-v1.0.2.tar.gz
INFO: Verification mode: lenient
INFO: Verification complete: passed
INFO: Checks: 1/1 passed

================================================================================
VERIFICATION SUMMARY
================================================================================
Status:  PASSED
Checks:  1/1 passed
Failed:  0
Warnings: 0
================================================================================
```

### Full Verification with All Artifacts

```bash
python security/release_verifier.py \
  --artifact dist/tars-v1.0.2.tar.gz \
  --sbom release/v1.0.2/sbom/tars-v1.0.2-cyclonedx.json \
  --slsa release/v1.0.2/slsa/tars-v1.0.2.provenance.json \
  --manifest release/v1.0.2/manifest.json \
  --public-key /run/secrets/rsa_public.pem \
  --policy strict \
  --json verification_report.json \
  --text verification_report.txt \
  --verbose
```

---

## Architecture

### System Components

```
┌─────────────────────────────────────────────────────────┐
│             Release Artifact Verifier                    │
└─────────────────────────────────────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        │               │               │
    ┌───▼───┐      ┌───▼───┐      ┌───▼───┐
    │ Hash  │      │ Sig   │      │ SBOM  │
    │Verify │      │Verify │      │Verify │
    └───┬───┘      └───┬───┘      └───┬───┘
        │               │               │
        └───────────────┼───────────────┘
                        │
        ┌───────────────┼───────────────┐
        │               │               │
    ┌───▼───┐      ┌───▼────┐     ┌───▼────┐
    │ SLSA  │      │Manifest│     │ Policy │
    │Verify │      │ Verify │     │Enforcer│
    └───────┘      └────────┘     └────────┘
                        │
                   ┌────▼────┐
                   │Reporting│
                   │(JSON/TXT)│
                   └─────────┘
```

### Verification Workflow

```
1. Artifact Discovery
   ├─ Locate primary artifact
   ├─ Find SBOM (if specified)
   ├─ Find SLSA provenance (if specified)
   ├─ Find manifest (if specified)
   └─ Locate signatures (.sig files)

2. Hash Verification
   ├─ Compute SHA-256/SHA-512
   ├─ Compare against manifest (if present)
   └─ Record results

3. Signature Verification (if public key provided)
   ├─ Load RSA public key
   ├─ Verify RSA-PSS signature
   └─ Record results

4. SBOM Verification (if SBOM provided)
   ├─ Parse JSON (CycloneDX or SPDX)
   ├─ Validate required fields
   ├─ Check component metadata
   └─ Record results

5. SLSA Provenance Verification (if SLSA provided)
   ├─ Parse in-toto attestation
   ├─ Validate predicate type
   ├─ Check builder identity
   ├─ Verify subject hashes
   └─ Record results

6. Manifest Verification (if manifest provided)
   ├─ Parse manifest.json
   ├─ Check all artifacts exist
   ├─ Verify hashes match
   └─ Record results

7. Policy Enforcement
   ├─ Check all artifacts signed
   ├─ Check SBOM present and valid
   ├─ Check SLSA present and valid
   ├─ Check hash integrity
   └─ Determine pass/fail

8. Report Generation
   ├─ Generate JSON report
   ├─ Generate text report
   └─ Return exit code
```

---

## Usage Patterns

### Pattern 1: Development Verification (Lenient Mode)

**Use Case:** Verify artifacts during development without strict enforcement.

```bash
python security/release_verifier.py \
  --artifact dist/tars-dev-build.tar.gz \
  --policy lenient \
  --verbose
```

**Behavior:**
- Warnings allowed
- Missing SBOM/SLSA tolerated
- Unsigned artifacts accepted
- Exit code 0 even with warnings

### Pattern 2: Pre-Release Validation (Strict Mode)

**Use Case:** Gate releases before publishing to artifact repository.

```bash
python security/release_verifier.py \
  --artifact dist/tars-v1.0.2.tar.gz \
  --sbom release/sbom/tars-v1.0.2-cyclonedx.json \
  --slsa release/slsa/tars-v1.0.2.provenance.json \
  --manifest release/manifest.json \
  --public-key keys/release_public.pem \
  --policy strict \
  --json verification.json \
  --verbose
```

**Behavior:**
- All checks must pass
- Missing SBOM/SLSA causes failure
- Unsigned artifacts rejected
- Exit code 7 on any policy violation

### Pattern 3: Continuous Integration

**Use Case:** Automated verification in CI/CD pipeline.

```yaml
# .github/workflows/verify-release.yml
- name: Verify Release Artifacts
  run: |
    python security/release_verifier.py \
      --artifact dist/tars-${{ github.ref_name }}.tar.gz \
      --sbom release/sbom/tars-${{ github.ref_name }}-cyclonedx.json \
      --slsa release/slsa/tars-${{ github.ref_name }}.provenance.json \
      --manifest release/manifest.json \
      --public-key ${{ secrets.RELEASE_PUBLIC_KEY_PATH }} \
      --policy strict \
      --json $GITHUB_STEP_SUMMARY/verification.json \
      --text $GITHUB_STEP_SUMMARY/verification.txt

- name: Upload Verification Report
  if: always()
  uses: actions/upload-artifact@v3
  with:
    name: verification-report
    path: $GITHUB_STEP_SUMMARY/verification.*
```

### Pattern 4: Air-Gapped Environment

**Use Case:** Verification in isolated networks without internet access.

```bash
# 1. Pre-generate verification bundle (internet-connected machine)
mkdir -p verification-bundle
cp dist/tars-v1.0.2.tar.gz verification-bundle/
cp release/sbom/*.json verification-bundle/
cp release/slsa/*.json verification-bundle/
cp release/manifest.json verification-bundle/
cp keys/rsa_public.pem verification-bundle/
tar -czf verification-bundle.tar.gz verification-bundle/

# 2. Transfer to air-gapped machine
scp verification-bundle.tar.gz user@airgap:/tmp/

# 3. Verify on air-gapped machine
cd /tmp
tar -xzf verification-bundle.tar.gz
cd verification-bundle
python /opt/tars/security/release_verifier.py \
  --artifact tars-v1.0.2.tar.gz \
  --sbom tars-v1.0.2-cyclonedx.json \
  --slsa tars-v1.0.2.provenance.json \
  --manifest manifest.json \
  --public-key rsa_public.pem \
  --policy strict
```

### Pattern 5: Programmatic Verification

**Use Case:** Embed verification in Python applications.

```python
#!/usr/bin/env python3
"""
Programmatic verification example
"""
from pathlib import Path
from security.release_verifier import ReleaseVerifier

def verify_release_artifacts(release_dir: Path) -> bool:
    """
    Verify all artifacts in release directory.

    Returns:
        True if verification passed, False otherwise
    """
    verifier = ReleaseVerifier(
        mode='strict',
        public_key_path=Path('/keys/rsa_public.pem')
    )

    report = verifier.verify_release(
        artifact_path=release_dir / 'artifact.tar.gz',
        sbom_path=release_dir / 'sbom' / 'cyclonedx.json',
        slsa_path=release_dir / 'slsa' / 'provenance.json',
        manifest_path=release_dir / 'manifest.json',
        version='1.0.2'
    )

    # Save reports
    verifier.save_report(
        report,
        release_dir / 'verification.json',
        format='json'
    )

    # Check status
    if report.overall_status == 'passed':
        print(f"✓ Verification passed ({report.passed_checks}/{report.total_checks} checks)")
        return True
    else:
        print(f"✗ Verification failed ({report.failed_checks} failures)")
        for error in report.errors:
            print(f"  - {error}")
        return False

if __name__ == '__main__':
    success = verify_release_artifacts(Path('./release/v1.0.2'))
    exit(0 if success else 1)
```

---

## Integration

### Integration with `prepare_release_artifacts.py`

The release verifier integrates directly with the release preparation script:

```bash
# Generate and verify release artifacts in one command
python scripts/prepare_release_artifacts.py \
  --include-sbom \
  --include-slsa \
  --sign \
  --verify-release \
  --verification-policy strict \
  --public-key /keys/rsa_public.pem \
  --output-dir release/v1.0.2 \
  --verbose
```

**Workflow:**
1. Generate artifacts
2. Generate SBOM (CycloneDX + SPDX)
3. Generate SLSA provenance
4. Sign artifacts with RSA-PSS
5. Create manifest with hashes
6. **[Phase 14.7 Task 3]** Verify all artifacts
7. Gate decision:
   - **Strict mode:** Abort if verification fails
   - **Lenient mode:** Continue with warnings

**Output Structure:**
```
release/v1.0.2/
├── artifact files...
├── sbom/
│   ├── tars-v1.0.2-cyclonedx.json
│   ├── tars-v1.0.2-cyclonedx.json.sig
│   ├── tars-v1.0.2-spdx.json
│   └── tars-v1.0.2-spdx.json.sig
├── slsa/
│   ├── tars-v1.0.2.provenance.json
│   └── tars-v1.0.2.provenance.json.sig
├── verification/          ← NEW (Phase 14.7 Task 3)
│   ├── tars-v1.0.2-verification.json
│   └── tars-v1.0.2-verification.txt
└── manifest.json
```

### CI/CD Pipeline Integration

#### GitHub Actions

```yaml
name: Release Verification

on:
  release:
    types: [created]

jobs:
  verify:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install cryptography

      - name: Download release artifacts
        uses: actions/download-artifact@v3
        with:
          name: release-artifacts
          path: release/

      - name: Verify Release
        run: |
          python security/release_verifier.py \
            --artifact release/tars-${{ github.event.release.tag_name }}.tar.gz \
            --sbom release/sbom/tars-${{ github.event.release.tag_name }}-cyclonedx.json \
            --slsa release/slsa/tars-${{ github.event.release.tag_name }}.provenance.json \
            --manifest release/manifest.json \
            --public-key ${{ secrets.RELEASE_PUBLIC_KEY }} \
            --policy strict \
            --json verification.json \
            --text verification.txt \
            --verbose

      - name: Upload Verification Report
        if: always()
        uses: actions/upload-artifact@v3
        with:
          name: verification-report
          path: |
            verification.json
            verification.txt

      - name: Comment on PR
        if: failure()
        uses: actions/github-script@v6
        with:
          script: |
            const fs = require('fs');
            const report = fs.readFileSync('verification.txt', 'utf8');
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: `## ❌ Release Verification Failed\n\n\`\`\`\n${report}\n\`\`\``
            })
```

#### GitLab CI

```yaml
verify-release:
  stage: verify
  image: python:3.9
  script:
    - pip install -r requirements.txt
    - pip install cryptography
    - python security/release_verifier.py
        --artifact dist/tars-${CI_COMMIT_TAG}.tar.gz
        --sbom release/sbom/tars-${CI_COMMIT_TAG}-cyclonedx.json
        --slsa release/slsa/tars-${CI_COMMIT_TAG}.provenance.json
        --manifest release/manifest.json
        --public-key ${RELEASE_PUBLIC_KEY}
        --policy strict
        --json verification.json
        --text verification.txt
        --verbose
  artifacts:
    when: always
    paths:
      - verification.json
      - verification.txt
    reports:
      junit: verification.json
  only:
    - tags
```

---

## Policy Configuration

### Strict Mode (Default for Production)

**Enforcement:**
- All artifacts must be signed
- SBOM must be present and valid
- SLSA provenance must be present and valid
- All hash verifications must pass
- No warnings tolerated

**Configuration:**
```bash
--policy strict
```

**Use Cases:**
- Production releases
- Public artifact repositories
- Compliance-regulated environments
- High-security deployments

### Lenient Mode (Development/Testing)

**Enforcement:**
- Missing SBOM/SLSA allowed (warning)
- Unsigned artifacts allowed (warning)
- Hash mismatches logged but not blocking
- Warnings do not fail verification

**Configuration:**
```bash
--policy lenient
```

**Use Cases:**
- Development builds
- Internal testing
- Rapid prototyping
- Pre-verification checks

### Custom Policy Rules (Programmatic)

```python
from security.release_verifier import PolicyEnforcer, VerificationReport

class CustomPolicyEnforcer(PolicyEnforcer):
    """Custom policy with additional rules."""

    def check_policies(self, report: VerificationReport):
        policies = super().check_policies(report)

        # Add custom policy: minimum 50 SBOM components
        if report.sbom_results:
            components = sum(r.component_count for r in report.sbom_results)
            if components < 50:
                policies.append(PolicyCheckResult(
                    policy_name="minimum_components",
                    passed=False,
                    severity="warning",
                    message=f"Only {components} components (minimum: 50)"
                ))

        return policies
```

---

## Troubleshooting

### Issue 1: "cryptography library not available"

**Error:**
```
RuntimeError: cryptography library required for signature verification
```

**Solution:**
```bash
pip install cryptography
```

### Issue 2: "Signature verification failed"

**Error:**
```
SignatureError: Signature verification failed: Signature did not match digest
```

**Diagnosis:**
1. Check public key matches private key used for signing
2. Verify artifact not modified after signing
3. Ensure signature file (.sig) is present

**Solution:**
```bash
# Verify public/private key match
openssl rsa -in private.pem -pubout | diff - public.pem

# Re-sign if needed
python security/signing.py sign artifact.tar.gz --key private.pem
```

### Issue 3: "SBOM validation failed: Invalid JSON"

**Error:**
```
SBOMError: Invalid JSON: Expecting property name enclosed in double quotes
```

**Solution:**
```bash
# Validate JSON syntax
python -m json.tool sbom.json > /dev/null

# Fix JSON formatting
python -m json.tool sbom.json > sbom-fixed.json
```

### Issue 4: "Hash mismatch"

**Error:**
```
IntegrityError: Hash verification failed (expected: abc123..., got: def456...)
```

**Diagnosis:**
1. Artifact modified after manifest generation
2. Incorrect hash in manifest
3. File corruption during transfer

**Solution:**
```bash
# Recalculate hash
python -c "import hashlib; print(hashlib.sha256(open('artifact.tar.gz','rb').read()).hexdigest())"

# Update manifest
# ... edit manifest.json with new hash
```

### Issue 5: "Policy gate failure (exit code 7)"

**Error:**
```
PolicyViolationError: Policy gate failure in strict mode
```

**Diagnosis:**
Check verification report for failed policies:
```bash
cat verification.txt | grep "✗"
```

**Solutions:**
1. **Fix violations:** Address each failed policy
2. **Use lenient mode:** `--policy lenient` (temporary)
3. **Override in emergency:** Document exception and approve manually

### Debugging Tips

**Enable verbose logging:**
```bash
python security/release_verifier.py --verbose ...
```

**Check Python version:**
```bash
python --version  # Must be 3.9+
```

**Verify file permissions:**
```bash
ls -la artifact.tar.gz
# Should be readable
```

**Test in isolation:**
```bash
# Test hash verification only
python -c "from security.release_verifier import HashVerifier; print(HashVerifier.compute_hash('artifact.tar.gz', 'sha256'))"

# Test signature verification only
python -c "from security.release_verifier import SignatureVerifier; v = SignatureVerifier('public.pem'); print(v.verify_signature('artifact.tar.gz'))"
```

---

## API Reference

### Classes

#### `ReleaseVerifier`

Main orchestrator for release verification.

```python
ReleaseVerifier(
    mode: Literal['strict', 'lenient'] = 'strict',
    public_key_path: Optional[Path] = None
)
```

**Methods:**

- `verify_release(...)` → `VerificationReport`
- `save_report(report, output_path, format='json')`

#### `HashVerifier`

Hash computation and verification.

```python
HashVerifier.compute_hash(file_path: Path, algorithm: str = 'sha256') → str
HashVerifier.verify_hash(file_path: Path, expected_hash: str, algorithm: str = 'sha256') → HashVerificationResult
```

#### `SignatureVerifier`

RSA-PSS signature verification.

```python
SignatureVerifier(public_key_path: Optional[Path] = None)
verify_signature(file_path: Path, signature_path: Optional[Path] = None) → SignatureVerificationResult
```

#### `SBOMVerifier`

SBOM validation for CycloneDX and SPDX.

```python
SBOMVerifier.verify_cyclonedx(sbom_path: Path) → SBOMVerificationResult
SBOMVerifier.verify_spdx(sbom_path: Path) → SBOMVerificationResult
```

#### `SLSAVerifier`

SLSA provenance validation.

```python
SLSAVerifier.verify_provenance(provenance_path: Path) → SLSAVerificationResult
```

#### `ManifestVerifier`

Manifest synchronization validation.

```python
ManifestVerifier.verify_manifest(manifest_path: Path, artifact_dir: Optional[Path] = None) → ManifestVerificationResult
```

#### `PolicyEnforcer`

Policy compliance enforcement.

```python
PolicyEnforcer(mode: Literal['strict', 'lenient'] = 'strict')
check_policies(report: VerificationReport) → List[PolicyCheckResult]
```

### Data Classes

All result classes are `@dataclass` with `.to_dict()` serialization:

- `HashVerificationResult`
- `SignatureVerificationResult`
- `SBOMVerificationResult`
- `SLSAVerificationResult`
- `ManifestVerificationResult`
- `PolicyCheckResult`
- `VerificationReport`

---

## Support

### Documentation

- [Phase 14.7 Task 3 Completion Summary](PHASE14_7_TASK3_COMPLETION_SUMMARY.md)
- [Production Runbook](PRODUCTION_RUNBOOK.md)
- [API Guide](PHASE14_6_API_GUIDE.md)

### Contact

- **Issues:** https://github.com/veleron-dev-studios/tars/issues
- **Security:** security@velerondevstudios.com
- **Support:** support@velerondevstudios.com

---

**Document Version:** 1.0
**Last Updated:** 2025-11-28
**Classification:** Internal - User Documentation
