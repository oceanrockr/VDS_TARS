# T.A.R.S. Development Handoff - Phase 25 Backup & Recovery

**Project:** T.A.R.S. (Temporal Augmented Retrieval System)
**Current Version:** v1.0.11 (GA)
**Target Version:** v1.0.12
**Last Completed:** Phase 24 - Field Testing & Codebase Hygiene
**In Progress:** Phase 25 - Backup & Recovery (Analysis Complete)
**Repository:** https://github.com/oceanrockr/VDS_TARS.git
**Branch:** main (up to date)
**Status:** RiPIT Analysis Complete, Implementation Ready
**Date:** January 3, 2026

---

## Quick Start - Copy This Into New Claude Code Session

```markdown
# T.A.R.S. RiPIT Continuation Prompt - Phase 25

## Project Context
- **Repository:** https://github.com/oceanrockr/VDS_TARS.git
- **Version:** v1.0.11 (GA) → v1.0.12
- **Last Phase:** 24 - Field Testing & Codebase Hygiene (COMPLETE)
- **Current Phase:** 25 - Backup & Recovery (ANALYSIS COMPLETE - Ready to Implement)

## RiPIT Status
- **Confidence Score:** 92.50% (APPROVED)
- **Action:** Implement with noted uncertainties
- **Tests First:** Required per RiPIT methodology

## Phase 25 Deliverables
1. `tests/test_backup_restore.py` - Write FIRST (~350 LOC)
2. `deploy/backup-tars.sh` - Automated backup (~500 LOC)
3. `deploy/restore-tars.sh` - Restore procedure (~400 LOC)
4. `docs/BACKUP_RECOVERY.md` - Operations runbook (~400 LOC)

## Implementation Order (TDD)
1. Write tests first
2. Implement backup script
3. Implement restore script
4. Create documentation
5. Update VERSION/CHANGELOG/README
6. Git commit and push

## Key Files Reference
- `deploy/generate-support-bundle.sh` - Pattern template for scripts
- `deploy/docker-compose.home.yml` - Volume definitions
- `docs/reference/dev/CONFIDENCE_DRIVEN_DEVELOPMENT.md` - RiPIT guide

## Docker MCP
docker mcp client connect claude-code --global

## Do Not Rebuild
Phases 20-24 are complete and committed. Do not refactor unless bug discovered.
```

---

## Phase 24 Summary (COMPLETE)

### What Was Accomplished (Previous Session)
1. Codebase cleanup - Committed 12 files, removed 9 redundant docs
2. Module exports updated - XSS, security headers, certificate monitoring
3. Test suites added - sanitize, security_headers, certificate_monitor
4. Documentation updated - README, MVP progress

### Commit Details
```
Commit: 6c75a71
Message: feat(security): Phase 24 - Field Testing & Codebase Hygiene
Files: 21 changed, +4,677 insertions, -38 deletions
```

---

## Phase 25 Analysis (COMPLETE)

### RiPIT Confidence Score: 92.50%

```
Scoring Breakdown:
- API Documentation:     95% × 0.30 = 28.50
- Similar Patterns:      95% × 0.25 = 23.75
- Data Flow:             90% × 0.20 = 18.00
- Complexity:            85% × 0.15 = 12.75
- Impact:                95% × 0.10 = 9.50
                         ─────────────────
                         TOTAL: 92.50%
```

### Noted Uncertainties
1. ChromaDB API for bulk export may need specific version handling
2. Large ollama model backups may need special handling (multi-GB)
3. Restore while services running needs careful sequencing

---

## Phase 25 Implementation Details

### Components to Backup
| Component | Volume | Method | Priority |
|-----------|--------|--------|----------|
| ChromaDB | chroma_data | Volume tar + API export | HIGH |
| PostgreSQL | postgres_data | pg_dump | HIGH |
| Redis | redis_data | RDB snapshot | MEDIUM |
| Configuration | tars-home.env | Copy with redaction | HIGH |
| Ollama Models | ollama_data | Volume tar (optional) | LOW |

### backup-tars.sh Features
- ChromaDB collection export via HTTP API
- PostgreSQL pg_dump with compression
- Redis BGSAVE + RDB copy
- Configuration backup (secrets redacted)
- Archive creation (tar.gz)
- SHA-256 checksums
- Manifest with metadata
- CLI flags: --output-dir, --include-models, --skip-postgres, --dry-run

### restore-tars.sh Features
- Integrity validation (checksums)
- Service stop before restore
- Volume restoration
- Service restart
- Post-restore health check
- CLI flags: --backup-file, --skip-validation, --dry-run

### Test Requirements
```python
class TestBackupRestore:
    def test_backup_creates_archive(self): pass
    def test_backup_includes_chromadb(self): pass
    def test_backup_includes_postgres(self): pass
    def test_backup_redacts_secrets(self): pass
    def test_restore_validates_integrity(self): pass
    def test_restore_recovers_data(self): pass
    def test_restore_handles_missing_backup(self): pass
```

---

## Key Patterns from Existing Scripts

### Script Template (from generate-support-bundle.sh)
```bash
#!/bin/bash
# Header with version and description
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Logging functions
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_step() { echo -e "\n${CYAN}>>> $1${NC}"; }

# CLI parsing
parse_args() { ... }
show_help() { ... }

# Main function
main() { ... }
main "$@"
```

### Secret Redaction Pattern
```bash
redact_secrets() {
    sed -E \
        -e 's/(PASSWORD|SECRET|KEY|TOKEN)[[:space:]]*[:=][[:space:]]*[^[:space:]]+/\1=<REDACTED>/gi' \
        -e 's/(Bearer|Basic)[[:space:]]+[A-Za-z0-9+/=_-]+/\1 <REDACTED>/gi' \
        -e 's/eyJ[A-Za-z0-9_-]*\.[A-Za-z0-9_-]*\.[A-Za-z0-9_-]*/<JWT_REDACTED>/g'
}
```

---

## Docker Volumes Reference

From docker-compose.home.yml:
```yaml
volumes:
  ollama_data:
    driver: local
  chroma_data:
    driver: local
  postgres_data:
    driver: local
  redis_data:
    driver: local
  backend_logs:
    driver: local
```

Container names:
- tars-home-ollama
- tars-home-chromadb
- tars-home-redis
- tars-home-postgres
- tars-home-backend

---

## Success Criteria

### Phase 25 Completion Checklist
- [ ] Tests written and passing
- [ ] Backup script creates valid archive
- [ ] ChromaDB data properly exported
- [ ] PostgreSQL dump included
- [ ] Redis RDB snapshot included
- [ ] Configuration files included (secrets redacted)
- [ ] Integrity checksum generated
- [ ] Restore script recovers data
- [ ] Restore validates integrity before applying
- [ ] Post-restore health check passes
- [ ] Cron scheduling documented
- [ ] Recovery runbook complete
- [ ] VERSION bumped to v1.0.12
- [ ] CHANGELOG updated
- [ ] README updated
- [ ] Git committed and pushed

---

## Version Bump Procedure

When completing Phase 25:

1. **Update VERSION file**
   ```bash
   echo "1.0.12" > VERSION
   ```

2. **Update CHANGELOG.md**
   - Add new `[1.0.12]` section
   - Document all Phase 25 changes

3. **Update MVP_PROGRESS_VISUALIZATION.md**
   - Add Phase 25 section
   - Update progress bars

4. **Commit and push**
   ```bash
   git add -A
   git commit -m "feat(backup): Phase 25 - Backup & Recovery (v1.0.12)"
   git push origin main
   ```

---

## Git Workflow

### Current Branch Status
```
Branch: main
Status: Up to date with origin/main
Last Commit: ea0e040 docs(dev): Add Phase 24 dev notes, handoff prompt, and update RiPIT guide

Remaining Untracked:
- s3-simulation/
- scripts/handoff/

Remaining Modified (IDE/generated):
- .claude/settings.local.json
- coverage.xml
```

### Commit Message Format
```
feat(backup): Phase 25 - Backup & Recovery

- Add backup-tars.sh for automated backups
- Add restore-tars.sh for disaster recovery
- Add test suite for backup/restore validation
- Add BACKUP_RECOVERY.md operations runbook
- Support ChromaDB, PostgreSQL, Redis backup
- Include integrity validation with checksums

Phase: 25 - Backup & Recovery
Version: v1.0.12

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
```

---

## Contact & Resources

**Repository:** https://github.com/oceanrockr/VDS_TARS.git
**RiPIT Methodology:** https://github.com/Veleron-Dev-Studios-LLC/VDS_RiPIT-Agent-Coding-Workflow
**Docker MCP:** `docker mcp client connect claude-code --global`

---

## Final Notes

- **Quality over speed** - A well-tested feature is worth more than a rushed implementation
- **Tests FIRST** - Write tests before any implementation code
- **Use existing patterns** - generate-support-bundle.sh is the template
- **Document everything** - Update docs as you build
- **Ask questions** - If uncertain, ask for clarification
- **Clean commits** - One logical change per commit
- **No breaking changes** - Maintain backward compatibility

---

**End of Handoff Document**

Generated: January 3, 2026
Author: Claude Opus 4.5 (Project Orchestrator)
For: T.A.R.S. Development Team
