# T.A.R.S. Post-GA Governance Policy

**Version:** 1.0.5
**Phase:** 15 - Post-GA Operations Enablement
**Status:** Production
**Effective Date:** December 22, 2025

---

## Table of Contents

1. [Overview](#overview)
2. [Codebase Classification](#codebase-classification)
3. [Change Proposal Process](#change-proposal-process)
4. [Versioning Rules](#versioning-rules)
5. [Pre-Merge Checklist](#pre-merge-checklist)
6. [Release Process](#release-process)
7. [Emergency Procedures](#emergency-procedures)

---

## Overview

T.A.R.S. v1.0.4 has achieved General Availability (GA) status. This document defines the governance policy for post-GA changes, ensuring production stability while allowing controlled evolution.

### Guiding Principles

1. **Stability First:** Production stability takes precedence over new features
2. **Additive Preferred:** Add new capabilities; avoid modifying stable code
3. **Validated Changes:** All changes must pass validation before merge
4. **Documented Intent:** Every change must have clear documentation

---

## Codebase Classification

### Frozen Areas (No Changes Without Critical Justification)

The following areas are considered **frozen** and require exceptional justification for modification:

| Path | Description | Justification Required |
|------|-------------|------------------------|
| `analytics/org_*.py` | Core analytics engines (Tasks 14.8) | Security fix or critical bug only |
| `scripts/ga_release_validator.py` | GA validation engine | Security fix only |
| `scripts/prepare_ga_release.py` | Release packager | Security fix only |
| `scripts/generate_production_readiness_checklist.py` | Readiness checker | Security fix only |
| `cognition/shared/auth.py` | Authentication module | Security fix only |
| `cognition/shared/rate_limiter.py` | Rate limiting | Security fix only |

**To modify frozen code:**
1. Create an RFC document explaining the need
2. Get approval from 2+ maintainers
3. Provide comprehensive test coverage
4. Document rollback plan

### Allowed Areas (Standard Review Process)

| Path | Description | Review Requirements |
|------|-------------|---------------------|
| `docs/` | Documentation | 1 maintainer approval |
| `scripts/` (new files) | New operational scripts | 1 maintainer + tests |
| `policies/` | SLA policy templates | 1 maintainer approval |
| `examples/` | Example code and configs | 1 maintainer approval |
| `tests/` | Test files | 1 maintainer approval |

### Monitored Areas (Careful Review Required)

| Path | Description | Review Requirements |
|------|-------------|---------------------|
| `analytics/run_*.py` | CLI tools | 2 maintainer approval |
| `observability/` | Observability modules | 2 maintainer approval |
| `enterprise_api/` | API server | 2 maintainer + security review |
| `charts/` | Kubernetes charts | 2 maintainer + ops review |

---

## Change Proposal Process

### Documentation-Only Changes

For changes to `docs/`, `policies/examples/`, or `examples/`:

```
1. Create branch: docs/short-description
2. Make changes
3. Self-review for accuracy
4. Open PR with label: docs-only
5. Get 1 maintainer approval
6. Merge
```

**No version bump required** for docs-only changes.

### Scripts-Only Changes (New Files)

For new scripts in `scripts/`:

```
1. Create branch: feat/script-name
2. Implement script following existing patterns
3. Add --help documentation
4. Add --dry-run mode if applicable
5. Test locally
6. Open PR with label: scripts-only
7. Get 1 maintainer approval
8. Merge
```

**Patch version bump** (e.g., 1.0.5 -> 1.0.6) for new scripts.

### Core Changes (Requires RFC)

For changes to frozen or monitored areas:

```
1. Create RFC document in docs/rfcs/RFC-XXX-title.md
2. Describe:
   - Problem statement
   - Proposed solution
   - Alternative approaches considered
   - Risk assessment
   - Rollback plan
   - Testing strategy
3. Get RFC approval from 2+ maintainers
4. Create branch: fix/issue-number or feat/rfc-number
5. Implement with full test coverage
6. Run GA validator: python scripts/ga_release_validator.py
7. Open PR referencing RFC
8. Get 2 maintainer approval
9. Security review if touching auth/crypto
10. Merge
```

**Minor or Major version bump** depending on scope.

---

## Versioning Rules

T.A.R.S. follows [Semantic Versioning](https://semver.org/):

```
MAJOR.MINOR.PATCH
  |     |     |
  |     |     +-- Docs, scripts, bug fixes (backward compatible)
  |     +-------- New features (backward compatible)
  +-------------- Breaking changes
```

### When to Bump

| Change Type | Version Bump | Example |
|-------------|--------------|---------|
| Documentation only | None | README update |
| New policy template | Patch | 1.0.5 -> 1.0.6 |
| New operational script | Patch | 1.0.5 -> 1.0.6 |
| Bug fix to analytics | Patch | 1.0.5 -> 1.0.6 |
| New analytics feature | Minor | 1.0.5 -> 1.1.0 |
| New CLI tool | Minor | 1.0.5 -> 1.1.0 |
| API breaking change | Major | 1.0.5 -> 2.0.0 |
| Exit code change | Major | 1.0.5 -> 2.0.0 |

### Version Files to Update

1. `VERSION` - Single source of truth
2. `CHANGELOG.md` - Add entry under `[Unreleased]`
3. `README.md` - Update version badge if present

---

## Pre-Merge Checklist

All PRs must pass these checks before merge:

### Required Checks (All PRs)

- [ ] **Syntax Check:** No Python syntax errors
  ```bash
  python -m py_compile scripts/your_script.py
  ```

- [ ] **Import Check:** All imports resolve
  ```bash
  python -c "import scripts.your_script"
  ```

- [ ] **Help Text:** CLI tools have --help
  ```bash
  python scripts/your_script.py --help
  ```

### Required for Script/Code Changes

- [ ] **GA Validator Passes:**
  ```bash
  python scripts/ga_release_validator.py
  # Exit code must be 150 (GA READY) or 151 (GA BLOCKED with acceptable warnings)
  ```

- [ ] **Production Readiness Check:**
  ```bash
  python scripts/generate_production_readiness_checklist.py
  # Overall score must be >= 90%
  ```

- [ ] **Existing Tests Pass:**
  ```bash
  python -m pytest tests/ -v --tb=short
  ```

### Required for Analytics Engine Changes

- [ ] **Exit Codes Documented:** All exit codes in module docstring
- [ ] **Backward Compatible:** Existing CLI flags still work
- [ ] **Integration Test Added:** Test in `tests/integration/`
- [ ] **RFC Approved:** Link to approved RFC

### Required for Security Changes

- [ ] **Security Review:** Approved by security team member
- [ ] **No Credential Leaks:** `git secrets --scan`
- [ ] **Dependency Audit:** `pip-audit`

---

## Release Process

### Regular Releases (Patch/Minor)

```bash
# 1. Ensure main is up to date
git checkout main
git pull origin main

# 2. Run validation
python scripts/ga_release_validator.py
# Must exit with code 150

# 3. Update VERSION file
echo "1.0.6" > VERSION

# 4. Update CHANGELOG.md
# Move [Unreleased] items to [1.0.6] section

# 5. Commit version bump
git add VERSION CHANGELOG.md
git commit -m "chore: bump version to 1.0.6"

# 6. Create tag
git tag -a v1.0.6 -m "Release v1.0.6"

# 7. Push
git push origin main
git push origin v1.0.6

# 8. (Optional) Create release artifacts
python scripts/prepare_ga_release.py --version 1.0.6
```

### Major Releases

Major releases require:
1. RFC approval
2. Migration guide
3. Deprecation notices (minimum 1 minor version warning)
4. Extended testing period (2 weeks minimum)

---

## Emergency Procedures

### Hotfix Process

For critical security or data integrity issues:

```bash
# 1. Create hotfix branch from latest tag
git checkout -b hotfix/issue-description v1.0.5

# 2. Implement minimal fix
# Keep changes as small as possible

# 3. Run validation
python scripts/ga_release_validator.py

# 4. Get expedited review (1 maintainer for security, 2 for data)

# 5. Merge to main
git checkout main
git merge hotfix/issue-description

# 6. Tag immediately
git tag -a v1.0.6 -m "Hotfix: issue-description"

# 7. Push
git push origin main
git push origin v1.0.6

# 8. Notify stakeholders
# Send email to ops-alerts@company.com

# 9. Document in CHANGELOG as Security fix
```

### Rollback Process

If a release causes issues:

```bash
# 1. Identify last known good version
GOOD_VERSION=v1.0.5

# 2. Revert to that version
git checkout main
git revert --no-commit HEAD..${GOOD_VERSION}
git commit -m "revert: rollback to ${GOOD_VERSION} due to [reason]"

# 3. Tag as patch
git tag -a v1.0.7 -m "Rollback to ${GOOD_VERSION}"

# 4. Push
git push origin main
git push origin v1.0.7

# 5. Document incident
# Create postmortem in docs/incidents/
```

---

## Governance Contacts

| Role | Responsibility |
|------|----------------|
| **Maintainers** | Code review, merge approval |
| **Security Team** | Security-related changes |
| **Ops Team** | Production deployments |
| **Release Manager** | Version bumps, release coordination |

---

## Related Documentation

- [Operator Runbook](OPS_RUNBOOK.md) - Day-to-day operations
- [Incident Playbook](INCIDENT_PLAYBOOK.md) - Incident response
- [GA Release Summary](PHASE14_9_GA_RELEASE_SUMMARY.md) - GA validation details
- [CHANGELOG.md](../CHANGELOG.md) - Version history

---

## Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2025-12-22 | T.A.R.S. Team | Initial governance policy |

---

**Document Version:** 1.0.0
**Maintained By:** T.A.R.S. Governance Team
