# Phase 14.6 — Release Package Structure

**Version:** v1.0.2-rc1
**Date:** 2025-11-27
**Package Type:** Release Candidate 1

---

## Overview

This document describes the complete structure of the T.A.R.S. v1.0.2-rc1 release package, including all files, their purposes, and instructions for extraction, validation, and verification.

---

## Release Package Contents

### Complete Directory Structure

```
release/
├── VERSION                                    # Version file
├── README.md                                  # Production documentation
├── CHANGELOG.md                               # Version history
├── RELEASE_NOTES_v1.0.2-RC1.md              # Release notes
├── manifest.json                             # Release manifest with SHA256 hashes
│
├── docs/                                     # Documentation
│   ├── PHASE14_6_ENTERPRISE_HARDENING.md    # Enterprise features guide (2,500+ LOC)
│   ├── PHASE14_6_API_GUIDE.md               # API reference (1,600+ LOC)
│   ├── PHASE14_6_PRODUCTION_RUNBOOK.md      # Operations guide
│   ├── PHASE14_6_DOCKER.md                  # Docker deployment
│   ├── PHASE14_6_QUICKSTART.md              # Quick start guide
│   ├── PHASE14_6_RELEASE_VALIDATION_CHECKLIST.md  # Validation checklist
│   └── PHASE14_6_RELEASE_PACKAGE_STRUCTURE.md     # This document
│
├── scripts/                                  # Utility scripts
│   ├── test_phase9_end_to_end.py            # E2E integration test
│   ├── prepare_release_artifacts.py          # Release packager
│   ├── run_api_server.py                    # API server launcher
│   └── tag_v1_0_2_rc1.sh                    # Git tagging script
│
├── examples/                                 # Example implementations
│   ├── api_client.py                        # Python API client
│   ├── compliance_check.sh                  # Bash compliance checker
│   └── generate_signed_report.py            # Signed report example
│
├── config/                                   # Configuration files
│   ├── requirements-dev.txt                 # Python dependencies
│   ├── pyproject.toml                       # Project metadata
│   └── pytest.ini                           # Pytest configuration
│
└── signatures/                               # Digital signatures (if --sign enabled)
    ├── VERSION.sig
    ├── README.md.sig
    ├── CHANGELOG.md.sig
    └── RELEASE_NOTES_v1.0.2-RC1.md.sig
```

---

## File Roles

### Core Documentation

| File | Purpose | Size | Format |
|------|---------|------|--------|
| VERSION | Version identifier | 11 bytes | Plain text |
| README.md | Project overview and quick start | ~50 KB | Markdown |
| CHANGELOG.md | Version history (Keep a Changelog) | ~30 KB | Markdown |
| RELEASE_NOTES_v1.0.2-RC1.md | RC1 release notes | ~40 KB | Markdown |
| manifest.json | Release manifest with SHA256 hashes | ~5 KB | JSON |

### Enterprise Guides

| File | Purpose | Size | Audience |
|------|---------|------|----------|
| PHASE14_6_ENTERPRISE_HARDENING.md | Enterprise features comprehensive guide | ~120 KB | DevOps, Security |
| PHASE14_6_API_GUIDE.md | API reference with examples | ~80 KB | Developers, Integrators |
| PHASE14_6_PRODUCTION_RUNBOOK.md | Operations and incident response | ~60 KB | SRE, Operations |
| PHASE14_6_DOCKER.md | Docker and Docker Compose deployment | ~40 KB | DevOps |
| PHASE14_6_QUICKSTART.md | 5-minute getting started guide | ~20 KB | All users |

### Code Artifacts

| File | Purpose | LOC | Language |
|------|---------|-----|----------|
| test_phase9_end_to_end.py | E2E integration test | 700 | Python 3.9+ |
| prepare_release_artifacts.py | Release packaging tool | 445 | Python 3.9+ |
| run_api_server.py | API server launcher | 280 | Python 3.9+ |
| tag_v1_0_2_rc1.sh | Git tagging script | 250 | Bash |
| api_client.py | Python API client | 470 | Python 3.9+ |
| compliance_check.sh | Compliance checker | 180 | Bash |
| generate_signed_report.py | Signed report example | 345 | Python 3.9+ |

### Configuration Files

| File | Purpose | Format |
|------|---------|--------|
| requirements-dev.txt | Python dependencies | pip requirements |
| pyproject.toml | Project metadata (PEP 518) | TOML |
| pytest.ini | Pytest configuration | INI |

---

## Distribution Formats

### Standard Distribution

**Filename:** `tars-v1.0.2-rc1.tar.gz`
**Contents:** All files listed above
**Size:** ~1.5 MB (estimated)
**Checksum:** SHA256 (in manifest.json)

### Enterprise Distribution (with signatures)

**Filename:** `tars-v1.0.2-rc1-enterprise.tar.gz`
**Contents:** Standard distribution + digital signatures
**Size:** ~1.7 MB (estimated)
**Checksum:** SHA256 (in manifest.json)
**Signature:** Included in signatures/ directory

---

## Extraction Instructions

### Standard Extraction

```bash
# Extract tarball
tar -xzf tars-v1.0.2-rc1.tar.gz

# Verify contents
cd release/
ls -la

# Validate manifest
python scripts/prepare_release_artifacts.py --verify-manifest
```

### Enterprise Extraction (with verification)

```bash
# Extract tarball
tar -xzf tars-v1.0.2-rc1-enterprise.tar.gz

# Verify contents
cd release/

# Verify signatures
for file in *.md VERSION; do
    if [ -f "signatures/$file.sig" ]; then
        echo "Verifying: $file"
        python -c "
from security.security_manager import SecurityManager
sm = SecurityManager(signing_key_path='keys/rsa_public.pem')
with open('$file', 'rb') as f:
    data = f.read()
with open('signatures/$file.sig', 'rb') as f:
    sig = f.read()
if sm.verify_signature(data, sig):
    print('  ✓ Valid')
else:
    print('  ✗ Invalid')
    exit(1)
"
    fi
done
```

---

## Validation Procedures

### File Integrity Verification

**Using manifest.json:**

```bash
# Verify all files against manifest
python << 'EOF'
import json
import hashlib
from pathlib import Path

with open('manifest.json') as f:
    manifest = json.load(f)

print(f"Validating {len(manifest['artifacts'])} artifacts...")

for artifact in manifest['artifacts']:
    file_path = Path(artifact['name'])

    if not file_path.exists():
        print(f"✗ Missing: {artifact['name']}")
        continue

    # Compute SHA256
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)

    computed = sha256.hexdigest()
    expected = artifact['sha256']

    if computed == expected:
        print(f"✓ {artifact['name']}")
    else:
        print(f"✗ {artifact['name']} (hash mismatch)")
        print(f"  Expected: {expected}")
        print(f"  Computed: {computed}")

print("Validation complete")
EOF
```

### Signature Verification

**Verify individual file:**

```bash
python << 'EOF'
from security.security_manager import SecurityManager
from pathlib import Path

# Initialize security manager with public key
sm = SecurityManager(signing_key_path='keys/rsa_public.pem')

# Verify README.md
with open('README.md', 'rb') as f:
    data = f.read()

with open('signatures/README.md.sig', 'rb') as f:
    signature = f.read()

if sm.verify_signature(data, signature):
    print("✓ README.md signature valid")
else:
    print("✗ README.md signature invalid")
    exit(1)
EOF
```

### Completeness Check

**Verify all required files exist:**

```bash
#!/usr/bin/env bash

REQUIRED_FILES=(
    "VERSION"
    "README.md"
    "CHANGELOG.md"
    "RELEASE_NOTES_v1.0.2-RC1.md"
    "manifest.json"
    "docs/PHASE14_6_ENTERPRISE_HARDENING.md"
    "docs/PHASE14_6_API_GUIDE.md"
    "scripts/test_phase9_end_to_end.py"
    "scripts/prepare_release_artifacts.py"
    "examples/api_client.py"
    "examples/compliance_check.sh"
)

MISSING=0

for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "✓ $file"
    else
        echo "✗ $file (missing)"
        MISSING=$((MISSING + 1))
    fi
done

if [ $MISSING -eq 0 ]; then
    echo ""
    echo "All required files present"
    exit 0
else
    echo ""
    echo "$MISSING files missing"
    exit 1
fi
```

---

## Installation from Package

### Quick Install (Local Development)

```bash
# Extract package
tar -xzf tars-v1.0.2-rc1.tar.gz
cd release/

# Install dependencies
pip install -r config/requirements-dev.txt

# Run quick start
less docs/PHASE14_6_QUICKSTART.md
```

### Enterprise Install (Production)

```bash
# Extract package
tar -xzf tars-v1.0.2-rc1-enterprise.tar.gz
cd release/

# Verify signatures (see above)

# Install dependencies
pip install -r config/requirements-dev.txt

# Generate production keys
python -c "import os; print(os.urandom(32).hex())" > /etc/tars/secrets/aes.key
openssl genrsa -out /etc/tars/secrets/rsa.key 4096
openssl rsa -in /etc/tars/secrets/rsa.key -pubout -out /etc/tars/secrets/rsa.pub

# Create production config
mkdir -p /etc/tars/config
cp docs/PHASE14_6_ENTERPRISE_HARDENING.md /etc/tars/docs/
# (follow enterprise hardening guide for full setup)

# Start API server
python scripts/run_api_server.py --profile prod --port 8443
```

---

## Official Distribution Contents List

### Manifest Version 1.0.2-rc1

**Generated:** 2025-11-27
**Profile:** prod
**Format:** JSON
**Signature:** Included (if enterprise distribution)

### Artifact Categories

| Category | Files | Total Size | Signed | Encrypted |
|----------|-------|------------|--------|-----------|
| Core Documentation | 5 | ~125 KB | Yes | No |
| Enterprise Guides | 7 | ~320 KB | Yes | No |
| Code Artifacts | 7 | ~50 KB | Yes | No |
| Configuration | 3 | ~10 KB | No | No |
| Signatures | 15 | ~10 KB | N/A | No |

### Total Package

- **Files:** 37 (standard) or 52 (enterprise with signatures)
- **Size:** ~1.5 MB (standard) or ~1.7 MB (enterprise)
- **SHA256:** (computed during packaging)
- **Signature:** (included if enterprise distribution)

---

## Verification Commands Reference

### Quick Verification Suite

```bash
# 1. Extract package
tar -xzf tars-v1.0.2-rc1.tar.gz && cd release/

# 2. Verify completeness
bash scripts/verify_completeness.sh

# 3. Verify manifest
python scripts/prepare_release_artifacts.py --verify-manifest

# 4. Verify signatures (if enterprise)
bash scripts/verify_signatures.sh

# 5. Run E2E test
python scripts/test_phase9_end_to_end.py

# 6. Validate compliance
bash examples/compliance_check.sh --profile local --standards soc2,iso27001
```

### Individual File Verification

```bash
# Verify specific file hash
sha256sum README.md
# Compare with manifest.json

# Verify specific file signature
python -c "
from security.security_manager import SecurityManager
sm = SecurityManager(signing_key_path='keys/rsa_public.pem')
with open('README.md', 'rb') as f1, open('signatures/README.md.sig', 'rb') as f2:
    print('Valid' if sm.verify_signature(f1.read(), f2.read()) else 'Invalid')
"
```

---

## Support & Documentation

### Quick Links

- **README:** [README.md](../README.md)
- **Release Notes:** [RELEASE_NOTES_v1.0.2-RC1.md](../RELEASE_NOTES_v1.0.2-RC1.md)
- **Changelog:** [CHANGELOG.md](../CHANGELOG.md)
- **Enterprise Guide:** [PHASE14_6_ENTERPRISE_HARDENING.md](PHASE14_6_ENTERPRISE_HARDENING.md)
- **API Guide:** [PHASE14_6_API_GUIDE.md](PHASE14_6_API_GUIDE.md)
- **Quick Start:** [PHASE14_6_QUICKSTART.md](PHASE14_6_QUICKSTART.md)

### Getting Help

- **Issues:** https://github.com/oceanrockr/VDS_TARS/issues
- **Documentation:** All guides included in `docs/` directory
- **Examples:** Working examples in `examples/` directory

---

## Package Signing Information

### Signing Authority

**Organization:** Veleron Dev Studios
**Project:** T.A.R.S. (Temporal Augmented Retrieval System)
**Version:** v1.0.2-rc1
**Algorithm:** RSA-PSS with SHA-256
**Key Size:** 4096-bit

### Public Key

The RSA public key for signature verification is available:
- **In Package:** `keys/rsa_public.pem` (if enterprise distribution)
- **Online:** https://github.com/oceanrockr/VDS_TARS/blob/main/keys/rsa_public.pem
- **Fingerprint:** (to be computed during key generation)

### Signature Coverage

The following files are signed in enterprise distributions:
- VERSION
- README.md
- CHANGELOG.md
- RELEASE_NOTES_v1.0.2-RC1.md
- All documentation in `docs/` (*.md)
- All Python scripts in `scripts/` (*.py)
- All examples in `examples/` (*.py, *.sh)

---

## Appendix A: File Checksums

### Core Files (Standard Distribution)

| File | SHA256 | Size |
|------|--------|------|
| VERSION | (computed during packaging) | 11 B |
| README.md | (computed during packaging) | ~50 KB |
| CHANGELOG.md | (computed during packaging) | ~30 KB |
| RELEASE_NOTES_v1.0.2-RC1.md | (computed during packaging) | ~40 KB |
| manifest.json | (computed during packaging) | ~5 KB |

**Note:** Complete checksums for all files are available in `manifest.json` after packaging.

---

## Appendix B: License Information

**License:** (To be determined)
**Copyright:** (C) 2025 Veleron Dev Studios
**Project:** T.A.R.S. - Temporal Augmented Retrieval System

---

**Version:** v1.0.2-rc1
**Last Updated:** 2025-11-27
**Status:** Release Candidate 1
**Package Format:** TAR.GZ with optional signatures
