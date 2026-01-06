# T.A.R.S. Backup & Recovery Operations Runbook

**Version:** v1.0.12
**Phase:** 25 - Backup & Recovery
**Last Updated:** January 3, 2026
**Status:** Production Ready

---

## Table of Contents

1. [Quick Reference](#quick-reference)
2. [Overview](#overview)
3. [Backup Operations](#backup-operations)
   - [Running a Backup](#running-a-backup)
   - [Backup Components](#backup-components)
   - [Backup Verification](#backup-verification)
   - [Scheduled Backups (Cron)](#scheduled-backups-cron)
4. [Restore Operations](#restore-operations)
   - [Pre-Restore Checklist](#pre-restore-checklist)
   - [Restore Procedure](#restore-procedure)
   - [Post-Restore Validation](#post-restore-validation)
   - [Partial Restore](#partial-restore)
5. [Troubleshooting](#troubleshooting)
   - [Common Issues](#common-issues)
   - [Recovery Procedures](#recovery-procedures)
6. [Best Practices](#best-practices)
   - [Backup Frequency](#backup-frequency)
   - [Retention Policy](#retention-policy)
   - [Offsite Storage](#offsite-storage)
7. [Appendix](#appendix)
   - [Exit Codes](#exit-codes)
   - [Manifest Format](#manifest-format)
   - [Volume Locations](#volume-locations)

---

## Quick Reference

### Command Quick Reference

| Command | Description |
|---------|-------------|
| `./backup-tars.sh` | Create full backup (ChromaDB, PostgreSQL, Redis, config) |
| `./backup-tars.sh --dry-run` | Preview backup operations without making changes |
| `./backup-tars.sh --include-models` | Create backup including Ollama models (large!) |
| `./backup-tars.sh --output-dir /mnt/nas/backups` | Backup to custom location |
| `./backup-tars.sh --skip-chromadb` | Backup without ChromaDB |
| `./backup-tars.sh --skip-postgres` | Backup without PostgreSQL |
| `./backup-tars.sh --skip-redis` | Backup without Redis |
| `./restore-tars.sh --backup-file <path>` | Restore from backup archive |
| `./restore-tars.sh --backup-file <path> --dry-run` | Preview restore operations |
| `./restore-tars.sh --backup-file <path> --component chromadb` | Restore only ChromaDB |

### Quick Health Check

```bash
# Verify backup archive integrity
sha256sum -c tars-backup-YYYYMMDD_HHMMSS.tar.gz.sha256

# List backup contents
tar -tzf tars-backup-YYYYMMDD_HHMMSS.tar.gz

# View backup manifest
tar -xzf tars-backup-YYYYMMDD_HHMMSS.tar.gz -O tars-backup-YYYYMMDD_HHMMSS/manifest.json | jq .
```

---

## Overview

### Purpose

The T.A.R.S. Backup & Recovery system provides comprehensive data protection for all critical components of your T.A.R.S. deployment. This includes:

- **Vector embeddings** stored in ChromaDB
- **Analytics and audit data** stored in PostgreSQL
- **Cache and session data** stored in Redis
- **Configuration files** (with automatic secret redaction)
- **Ollama LLM models** (optional, for complete disaster recovery)

### Architecture

```
+-------------------+     +-------------------+     +-------------------+
|    ChromaDB       |     |    PostgreSQL     |     |      Redis        |
|  (Vector Store)   |     | (Analytics/Audit) |     |     (Cache)       |
+--------+----------+     +--------+----------+     +--------+----------+
         |                         |                         |
         v                         v                         v
+--------+----------+     +--------+----------+     +--------+----------+
|  Volume Tar Dump  |     |     pg_dump       |     |  BGSAVE + RDB     |
+--------+----------+     +--------+----------+     +--------+----------+
         |                         |                         |
         +------------+------------+------------+------------+
                      |                         |
                      v                         v
              +-------+-------+         +-------+-------+
              |   manifest.json |       |  config/      |
              +-------+-------+         +-------+-------+
                      |                         |
                      +------------+------------+
                                   |
                                   v
                      +------------+------------+
                      | tars-backup-YYYYMMDD.tar.gz |
                      +------------+------------+
                                   |
                                   v
                      +------------+------------+
                      |   .tar.gz.sha256        |
                      +---------------------------+
```

### Backup Location

Default backup location: `<PROJECT_ROOT>/backups/`

Recommended production location: `/mnt/nas/backups/tars/`

---

## Backup Operations

### Running a Backup

#### Standard Backup (Recommended)

Creates a backup of ChromaDB, PostgreSQL, Redis, and configuration files:

```bash
cd /opt/tars/deploy
./backup-tars.sh
```

**Expected output:**
```
===============================================================
   T.A.R.S. Backup Script - v1.0.12 (Phase 25)
===============================================================

[INFO] Backup configuration:
[INFO]   Output directory: /opt/tars/backups
[INFO]   Backup name: tars-backup-20260103_140000
[INFO]   Include models: false

>>> Running pre-flight checks...
[OK] Docker is available and running
[OK] Disk space: 45.23 GB available
[OK] Docker Compose file found
[OK] 5 T.A.R.S. containers running
[OK] Pre-flight checks complete

>>> Backing up ChromaDB vector database...
[INFO] Backing up volume: deploy_chroma_data
[OK] ChromaDB backup complete (2.34 GB)

>>> Backing up PostgreSQL database...
[INFO] Running pg_dump via docker exec...
[OK] PostgreSQL backup complete (156.78 MB)

>>> Backing up Redis cache...
[INFO] Triggering Redis BGSAVE...
[INFO] Copying RDB snapshot...
[OK] Redis backup complete (23.45 MB)

>>> Backing up configuration files...
[INFO] Backing up environment file (secrets redacted)...
[OK]   tars-home.env backed up (redacted)
[INFO] Backing up Docker Compose file...
[OK]   docker-compose.home.yml backed up
[OK] Configuration backup complete

>>> Generating backup manifest...
[OK] Manifest generated: manifest.json

>>> Creating backup archive...
[INFO] Compressing backup directory...
[OK] Archive created: 1.87 GB
[INFO] Generating SHA-256 checksum...
[OK] Checksum file created

============================================================
   Backup Complete
============================================================

  Archive:   /opt/tars/backups/tars-backup-20260103_140000.tar.gz
  Checksum:  /opt/tars/backups/tars-backup-20260103_140000.tar.gz.sha256
  Size:      1.87 GB

  Components backed up:
    + chromadb
    + postgres
    + redis
    + configuration
```

#### Full Backup with Models

Include Ollama models for complete disaster recovery:

```bash
./backup-tars.sh --include-models --output-dir /mnt/nas/backups
```

**Warning:** This can produce archives of 10-50 GB depending on downloaded models.

#### Dry Run (Preview)

Preview what the backup will do without creating any files:

```bash
./backup-tars.sh --dry-run
```

**Output:**
```
===============================================================
   Dry Run Summary
===============================================================

The following operations would be performed:

  Output directory: /opt/tars/backups
  Backup name: tars-backup-20260103_140000

  Components to backup:
    + ChromaDB vector database (volume tar)
    + PostgreSQL database (pg_dump)
    + Redis cache (BGSAVE + RDB copy)
    + Configuration files (secrets redacted)
    - Ollama models (not included)

  Files to be created:
    - tars-backup-20260103_140000.tar.gz
    - tars-backup-20260103_140000.tar.gz.sha256

No files were created (dry run mode)
```

### Backup Components

| Component | Data Type | Size Estimate | Method | Frequency |
|-----------|-----------|---------------|--------|-----------|
| **ChromaDB** | Vector embeddings, collections | 1-10 GB | Docker volume tar | Daily |
| **PostgreSQL** | Analytics, audit logs, metrics | 100 MB - 1 GB | pg_dump (SQL) | Daily |
| **Redis** | Cache, sessions, rate limits | 10-100 MB | BGSAVE + RDB copy | Daily |
| **Configuration** | tars-home.env, compose files | < 1 MB | Copy (redacted) | On change |
| **Ollama Models** | LLM weights (Mistral, etc.) | 5-50 GB | Docker volume tar | Weekly |

#### Volume Details

| Volume Name | Container | Mount Point | Description |
|-------------|-----------|-------------|-------------|
| `chroma_data` | tars-home-chromadb | /chroma/chroma | Vector database persistence |
| `postgres_data` | tars-home-postgres | /var/lib/postgresql/data | PostgreSQL data directory |
| `redis_data` | tars-home-redis | /data | Redis RDB snapshots |
| `ollama_data` | tars-home-ollama | /root/.ollama | Model weights and config |
| `backend_logs` | tars-home-backend | /app/logs | Application logs |

### Backup Verification

#### Verify Archive Integrity

```bash
# Navigate to backup directory
cd /mnt/nas/backups

# Verify checksum
sha256sum -c tars-backup-20260103_140000.tar.gz.sha256

# Expected output:
# tars-backup-20260103_140000.tar.gz: OK
```

#### Inspect Backup Contents

```bash
# List archive contents
tar -tzf tars-backup-20260103_140000.tar.gz

# Expected structure:
# tars-backup-20260103_140000/
# tars-backup-20260103_140000/manifest.json
# tars-backup-20260103_140000/chromadb.tar.gz
# tars-backup-20260103_140000/postgres.sql.gz
# tars-backup-20260103_140000/redis-dump.rdb
# tars-backup-20260103_140000/config/
# tars-backup-20260103_140000/config/tars-home.env.redacted
# tars-backup-20260103_140000/config/docker-compose.home.yml
# tars-backup-20260103_140000/config/INVENTORY.txt
```

#### View Backup Manifest

```bash
# Extract and view manifest
tar -xzf tars-backup-20260103_140000.tar.gz -O tars-backup-20260103_140000/manifest.json | jq .

# Expected output:
{
  "version": "1.0.12",
  "timestamp": "2026-01-03T14:00:00+00:00",
  "backup_name": "tars-backup-20260103_140000",
  "components": ["chromadb", "postgres", "redis", "configuration"],
  "checksums": {
    "chromadb.tar.gz": "a1b2c3d4e5f6...",
    "postgres.sql.gz": "b2c3d4e5f6a1...",
    "redis-dump.rdb": "c3d4e5f6a1b2...",
    "config/": "d4e5f6a1b2c3..."
  },
  "host": "tars-home-server",
  "docker_volumes": [
    "deploy_chroma_data",
    "deploy_postgres_data",
    "deploy_redis_data"
  ],
  "options": {
    "include_models": false,
    "skip_postgres": false,
    "skip_chromadb": false,
    "skip_redis": false
  },
  "failed_components": []
}
```

### Scheduled Backups (Cron)

#### Daily Backup at 2 AM

```bash
# Edit crontab
crontab -e

# Add daily backup job
0 2 * * * /opt/tars/deploy/backup-tars.sh --output-dir /mnt/nas/backups/daily >> /var/log/tars-backup.log 2>&1
```

#### Weekly Full Backup with Models (Sunday 3 AM)

```bash
# Weekly backup including Ollama models
0 3 * * 0 /opt/tars/deploy/backup-tars.sh --include-models --output-dir /mnt/nas/backups/weekly >> /var/log/tars-backup-weekly.log 2>&1
```

#### Monthly Archive (1st of Month, 4 AM)

```bash
# Monthly backup to archive storage
0 4 1 * * /opt/tars/deploy/backup-tars.sh --include-models --output-dir /mnt/nas/backups/monthly >> /var/log/tars-backup-monthly.log 2>&1
```

#### Complete Crontab Example

```bash
# T.A.R.S. Backup Schedule
# ========================
# Daily backup at 2:00 AM (no models)
0 2 * * * /opt/tars/deploy/backup-tars.sh --output-dir /mnt/nas/backups/daily >> /var/log/tars-backup.log 2>&1

# Weekly full backup on Sunday at 3:00 AM (with models)
0 3 * * 0 /opt/tars/deploy/backup-tars.sh --include-models --output-dir /mnt/nas/backups/weekly >> /var/log/tars-backup-weekly.log 2>&1

# Monthly archive on 1st at 4:00 AM (with models)
0 4 1 * * /opt/tars/deploy/backup-tars.sh --include-models --output-dir /mnt/nas/backups/monthly >> /var/log/tars-backup-monthly.log 2>&1

# Cleanup old daily backups (keep 7 days)
0 5 * * * find /mnt/nas/backups/daily -name "tars-backup-*.tar.gz" -mtime +7 -delete

# Cleanup old weekly backups (keep 4 weeks)
0 5 * * 0 find /mnt/nas/backups/weekly -name "tars-backup-*.tar.gz" -mtime +28 -delete
```

#### Verify Cron Job

```bash
# List current cron jobs
crontab -l

# Check cron service status
systemctl status cron

# Monitor backup logs
tail -f /var/log/tars-backup.log
```

---

## Restore Operations

### Pre-Restore Checklist

Before performing a restore, complete this checklist:

| Check | Command | Expected |
|-------|---------|----------|
| Backup archive exists | `ls -la <backup-file>` | File present |
| Checksum verified | `sha256sum -c <backup>.sha256` | OK |
| Docker running | `docker info` | No errors |
| Sufficient disk space | `df -h` | > backup size + 50% |
| T.A.R.S. containers stopped | `docker compose ps` | All stopped (recommended) |
| Current data backed up | `./backup-tars.sh` | Exit code 0 |

#### Pre-Restore Commands

```bash
# 1. Verify backup integrity
cd /mnt/nas/backups
sha256sum -c tars-backup-20260103_140000.tar.gz.sha256

# 2. Check disk space (need 2x backup size minimum)
df -h /opt/tars

# 3. Create safety backup of current state
cd /opt/tars/deploy
./backup-tars.sh --output-dir /tmp/safety-backup

# 4. Stop T.A.R.S. services (recommended for full restore)
docker compose -f docker-compose.home.yml down

# 5. Verify containers stopped
docker ps --filter "name=tars-home"
```

### Restore Procedure

#### Full Restore

Restores all components from a backup archive:

```bash
cd /opt/tars/deploy
./restore-tars.sh --backup-file /mnt/nas/backups/tars-backup-20260103_140000.tar.gz
```

**Restore Process:**

```
===============================================================
   T.A.R.S. Restore Script - v1.0.12 (Phase 25)
===============================================================

[INFO] Restore configuration:
[INFO]   Backup file: /mnt/nas/backups/tars-backup-20260103_140000.tar.gz
[INFO]   Restore mode: full

>>> Verifying backup archive...
[INFO] Checking SHA-256 checksum...
[OK] Checksum verified
[INFO] Extracting manifest...
[OK] Manifest loaded: 4 components

>>> Pre-restore checks...
[INFO] Checking container states...
[WARN] Container tars-home-postgres is running - will be stopped during restore
[OK] Pre-restore checks complete

>>> Restoring ChromaDB...
[INFO] Stopping ChromaDB container...
[INFO] Clearing existing volume data...
[INFO] Restoring volume from backup...
[INFO] Starting ChromaDB container...
[OK] ChromaDB restored successfully

>>> Restoring PostgreSQL...
[INFO] Stopping PostgreSQL container...
[INFO] Starting fresh PostgreSQL container...
[INFO] Waiting for PostgreSQL to be ready...
[INFO] Restoring database from pg_dump...
[OK] PostgreSQL restored successfully

>>> Restoring Redis...
[INFO] Stopping Redis container...
[INFO] Copying RDB snapshot...
[INFO] Starting Redis container...
[OK] Redis restored successfully

>>> Starting T.A.R.S. services...
[INFO] Running docker compose up...
[OK] All services started

>>> Post-restore validation...
[INFO] Checking service health...
[OK] Ollama: healthy
[OK] ChromaDB: healthy
[OK] Redis: healthy
[OK] PostgreSQL: healthy
[OK] Backend: healthy

============================================================
   Restore Complete
============================================================

  Restored from: tars-backup-20260103_140000.tar.gz
  Components restored:
    + chromadb
    + postgres
    + redis

  Services are running. Verify with:
    curl http://localhost:8000/health
```

#### Dry Run Restore

Preview restore operations without making changes:

```bash
./restore-tars.sh --backup-file /mnt/nas/backups/tars-backup-20260103_140000.tar.gz --dry-run
```

### Post-Restore Validation

After a restore, validate system integrity:

```bash
# 1. Check all container health
docker ps --filter "name=tars-home" --format "table {{.Names}}\t{{.Status}}"

# 2. Verify backend health endpoint
curl -s http://localhost:8000/health | jq .

# 3. Verify ChromaDB collections
curl -s http://localhost:8001/api/v1/collections | jq .

# 4. Check PostgreSQL connectivity
docker exec tars-home-postgres psql -U tars -d tars_home -c "SELECT COUNT(*) FROM information_schema.tables;"

# 5. Verify Redis data
docker exec tars-home-redis redis-cli INFO keyspace

# 6. Run RAG validation (optional but recommended)
./validate-rag.sh

# 7. Check application logs for errors
docker logs tars-home-backend --tail 100 | grep -i error
```

#### Health Check Script

```bash
#!/bin/bash
# post-restore-health-check.sh

echo "=== T.A.R.S. Post-Restore Health Check ==="
echo ""

# Check containers
echo "Container Status:"
docker ps --filter "name=tars-home" --format "  {{.Names}}: {{.Status}}"
echo ""

# Check endpoints
echo "Endpoint Health:"
for endpoint in "http://localhost:8000/health" "http://localhost:8001/api/v1/heartbeat" "http://localhost:11434/api/tags"; do
    status=$(curl -s -o /dev/null -w "%{http_code}" "$endpoint" 2>/dev/null)
    if [[ "$status" == "200" ]]; then
        echo "  $endpoint: OK"
    else
        echo "  $endpoint: FAILED ($status)"
    fi
done
echo ""

# Check databases
echo "Database Checks:"
pg_count=$(docker exec tars-home-postgres psql -U tars -d tars_home -t -c "SELECT COUNT(*) FROM information_schema.tables;" 2>/dev/null | tr -d ' ')
echo "  PostgreSQL tables: $pg_count"

redis_keys=$(docker exec tars-home-redis redis-cli DBSIZE 2>/dev/null | awk '{print $2}')
echo "  Redis keys: $redis_keys"

chroma_collections=$(curl -s http://localhost:8001/api/v1/collections 2>/dev/null | jq 'length')
echo "  ChromaDB collections: $chroma_collections"
echo ""

echo "=== Health Check Complete ==="
```

### Partial Restore

Restore individual components when only specific data is corrupted:

#### Restore Only ChromaDB

```bash
./restore-tars.sh --backup-file /mnt/nas/backups/tars-backup-20260103_140000.tar.gz --component chromadb
```

#### Restore Only PostgreSQL

```bash
./restore-tars.sh --backup-file /mnt/nas/backups/tars-backup-20260103_140000.tar.gz --component postgres
```

#### Restore Only Redis

```bash
./restore-tars.sh --backup-file /mnt/nas/backups/tars-backup-20260103_140000.tar.gz --component redis
```

#### Manual Partial Restore

For fine-grained control, manually restore components:

```bash
# Extract backup archive
mkdir -p /tmp/tars-restore
tar -xzf /mnt/nas/backups/tars-backup-20260103_140000.tar.gz -C /tmp/tars-restore

# Navigate to extracted directory
cd /tmp/tars-restore/tars-backup-20260103_140000

# Restore ChromaDB manually
docker stop tars-home-chromadb
docker run --rm \
    -v deploy_chroma_data:/data \
    -v $(pwd):/backup:ro \
    alpine:latest \
    sh -c "rm -rf /data/* && tar -xzf /backup/chromadb.tar.gz -C /data"
docker start tars-home-chromadb

# Restore PostgreSQL manually
docker exec -i tars-home-postgres psql -U tars -d tars_home < <(gunzip -c postgres.sql.gz)

# Restore Redis manually
docker stop tars-home-redis
docker cp redis-dump.rdb tars-home-redis:/data/dump.rdb
docker start tars-home-redis

# Cleanup
rm -rf /tmp/tars-restore
```

---

## Troubleshooting

### Common Issues

#### 1. Backup Failed: Container Not Running

**Error:**
```
[WARN] PostgreSQL container 'tars-home-postgres' is not running
[ERROR] PostgreSQL backup failed
```

**Cause:** The container being backed up is not running.

**Solution:**
```bash
# Check container status
docker ps -a --filter "name=tars-home"

# Start the specific container
docker start tars-home-postgres

# Or start all T.A.R.S. services
cd /opt/tars/deploy
docker compose -f docker-compose.home.yml up -d

# Wait for health checks to pass
sleep 30

# Retry backup
./backup-tars.sh
```

#### 2. Restore Failed: Checksum Mismatch

**Error:**
```
[ERROR] Checksum verification failed
[ERROR] Expected: a1b2c3d4...
[ERROR] Actual:   x9y8z7w6...
```

**Cause:** Backup archive is corrupted or was modified after creation.

**Solution:**
```bash
# Re-download or re-copy the backup from source
rsync -avP backup-server:/backups/tars-backup-20260103_140000.tar.gz ./

# Verify again
sha256sum -c tars-backup-20260103_140000.tar.gz.sha256

# If still failing, use a different backup
ls -la /mnt/nas/backups/daily/

# Choose an older, verified backup
./restore-tars.sh --backup-file /mnt/nas/backups/daily/tars-backup-20260102_020000.tar.gz
```

#### 3. PostgreSQL Restore Error: Database In Use

**Error:**
```
ERROR: database "tars_home" is being accessed by other users
DETAIL: There is 1 other session using the database.
```

**Cause:** Active connections to PostgreSQL during restore.

**Solution:**
```bash
# Stop the backend to release connections
docker stop tars-home-backend

# Force disconnect all sessions
docker exec tars-home-postgres psql -U tars -d postgres -c "
SELECT pg_terminate_backend(pid)
FROM pg_stat_activity
WHERE datname = 'tars_home' AND pid <> pg_backend_pid();"

# Drop and recreate database
docker exec tars-home-postgres psql -U tars -d postgres -c "DROP DATABASE IF EXISTS tars_home;"
docker exec tars-home-postgres psql -U tars -d postgres -c "CREATE DATABASE tars_home OWNER tars;"

# Now restore
gunzip -c postgres.sql.gz | docker exec -i tars-home-postgres psql -U tars -d tars_home

# Restart backend
docker start tars-home-backend
```

#### 4. ChromaDB Restore: Permission Denied

**Error:**
```
tar: chroma: Cannot open: Permission denied
tar: Exiting with failure status due to previous errors
```

**Cause:** Volume permission mismatch between backup source and restore target.

**Solution:**
```bash
# Stop ChromaDB
docker stop tars-home-chromadb

# Fix permissions on volume
docker run --rm \
    -v deploy_chroma_data:/data \
    alpine:latest \
    chmod -R 777 /data

# Retry restore
docker run --rm \
    -v deploy_chroma_data:/data \
    -v /tmp/tars-restore/tars-backup-20260103_140000:/backup:ro \
    alpine:latest \
    sh -c "rm -rf /data/* && tar -xzf /backup/chromadb.tar.gz -C /data && chown -R 1000:1000 /data"

# Start ChromaDB
docker start tars-home-chromadb
```

#### 5. Insufficient Disk Space

**Error:**
```
[WARN] Low disk space: 2.34 GB available
[WARN] Recommended: 5.00 GB for backup
tar: chromadb.tar.gz: Wrote only 2048 of 10240 bytes
tar: Error is not recoverable: exiting now
```

**Cause:** Not enough disk space for backup or restore operation.

**Solution:**
```bash
# Check disk usage
df -h

# Clean up old backups
find /mnt/nas/backups -name "tars-backup-*.tar.gz" -mtime +30 -delete

# Clean up Docker resources
docker system prune -af --volumes

# If still insufficient, use external storage
./backup-tars.sh --output-dir /mnt/external-drive/tars-backups
```

#### 6. Lock File Prevents Backup

**Error:**
```
[ERROR] Another backup is already running (PID: 12345)
[ERROR] Lock file: /tmp/.tars-backup.lock
```

**Cause:** Previous backup crashed or is still running.

**Solution:**
```bash
# Check if backup is actually running
ps aux | grep backup-tars

# If not running, remove stale lock
rm -f /tmp/.tars-backup.lock

# Retry backup
./backup-tars.sh
```

### Recovery Procedures

#### Complete System Recovery

When all data is lost and you need to restore from scratch:

```bash
# 1. Install T.A.R.S. on fresh system
./install-tars-home.sh

# 2. Stop all services
docker compose -f docker-compose.home.yml down

# 3. Restore from most recent backup
./restore-tars.sh --backup-file /mnt/nas/backups/weekly/tars-backup-20260102_030000.tar.gz

# 4. If models were backed up, they'll be restored
# Otherwise, re-pull models
docker exec tars-home-ollama ollama pull mistral:7b-instruct

# 5. Verify system health
./validate-deployment.sh
```

#### Recover from Corrupted ChromaDB

```bash
# 1. Stop services
docker compose -f docker-compose.home.yml down

# 2. Remove corrupted volume
docker volume rm deploy_chroma_data

# 3. Recreate volume
docker volume create deploy_chroma_data

# 4. Restore from backup
./restore-tars.sh --backup-file /mnt/nas/backups/daily/tars-backup-20260103_020000.tar.gz --component chromadb

# 5. Start services
docker compose -f docker-compose.home.yml up -d

# 6. Validate
curl http://localhost:8001/api/v1/collections | jq .
```

---

## Best Practices

### Backup Frequency

| Backup Type | Frequency | Contents | Retention |
|-------------|-----------|----------|-----------|
| **Daily** | Every day at 2 AM | ChromaDB, PostgreSQL, Redis, Config | 7 days |
| **Weekly** | Every Sunday at 3 AM | Full backup with Ollama models | 4 weeks |
| **Monthly** | 1st of month at 4 AM | Full backup with models | 12 months |
| **Pre-Upgrade** | Before any upgrade | Full backup with models | Until upgrade verified |

#### Recommended Schedule

```
Daily (2 AM):     ChromaDB + PostgreSQL + Redis + Config  (~2 GB)
Weekly (Sun 3 AM): Full backup + Ollama models           (~15 GB)
Monthly (1st 4 AM): Full backup + models to cold storage (~15 GB)
```

### Retention Policy

| Tier | Retention | Storage Location | Example Path |
|------|-----------|------------------|--------------|
| **Hot** | 7 days | Local NAS | `/mnt/nas/backups/daily/` |
| **Warm** | 4 weeks | NAS archive | `/mnt/nas/backups/weekly/` |
| **Cold** | 12 months | Cloud/Offsite | S3, Backblaze B2, etc. |
| **Archive** | Indefinite | Tape/Glacier | Annual snapshots |

#### Cleanup Script

```bash
#!/bin/bash
# backup-cleanup.sh - Run weekly to enforce retention

BACKUP_ROOT="/mnt/nas/backups"

echo "=== T.A.R.S. Backup Cleanup ==="
echo "Date: $(date)"
echo ""

# Daily: keep 7 days
echo "Cleaning daily backups (older than 7 days)..."
find "$BACKUP_ROOT/daily" -name "tars-backup-*.tar.gz*" -mtime +7 -delete -print

# Weekly: keep 4 weeks
echo "Cleaning weekly backups (older than 28 days)..."
find "$BACKUP_ROOT/weekly" -name "tars-backup-*.tar.gz*" -mtime +28 -delete -print

# Monthly: keep 12 months
echo "Cleaning monthly backups (older than 365 days)..."
find "$BACKUP_ROOT/monthly" -name "tars-backup-*.tar.gz*" -mtime +365 -delete -print

echo ""
echo "=== Cleanup Complete ==="

# Report remaining backups
echo ""
echo "Remaining backups:"
echo "  Daily:   $(ls -1 $BACKUP_ROOT/daily/*.tar.gz 2>/dev/null | wc -l) files"
echo "  Weekly:  $(ls -1 $BACKUP_ROOT/weekly/*.tar.gz 2>/dev/null | wc -l) files"
echo "  Monthly: $(ls -1 $BACKUP_ROOT/monthly/*.tar.gz 2>/dev/null | wc -l) files"
```

### Offsite Storage

#### Sync to Cloud Storage (S3)

```bash
# Install AWS CLI if not present
# apt install awscli

# Sync monthly backups to S3
aws s3 sync /mnt/nas/backups/monthly/ s3://your-bucket/tars-backups/monthly/ \
    --storage-class STANDARD_IA \
    --exclude "*" \
    --include "tars-backup-*.tar.gz" \
    --include "tars-backup-*.tar.gz.sha256"
```

#### Sync to Backblaze B2

```bash
# Install b2 CLI
# pip install b2

# Sync to B2
b2 sync /mnt/nas/backups/monthly/ b2://tars-backups/monthly/
```

#### Rsync to Remote Server

```bash
# Sync to remote backup server
rsync -avz --progress \
    /mnt/nas/backups/weekly/ \
    backup-server:/backups/tars/weekly/
```

#### Cron Job for Offsite Sync

```bash
# Weekly offsite sync (Sunday 6 AM, after backup completes)
0 6 * * 0 rsync -avz /mnt/nas/backups/weekly/ backup-server:/backups/tars/ >> /var/log/tars-offsite-sync.log 2>&1
```

---

## Appendix

### Exit Codes

| Code | Meaning | Action Required |
|------|---------|-----------------|
| **0** | Success | None - all components backed up/restored successfully |
| **1** | Error | Investigate logs, no components were processed |
| **2** | Partial Success | Some components failed - check manifest for details |

#### Exit Code Decision Tree

```
Exit Code 0: All OK
    └── Verify checksum file exists
    └── Store backup according to retention policy

Exit Code 1: Complete Failure
    └── Check Docker is running: docker info
    └── Check disk space: df -h
    └── Check container status: docker ps
    └── Review logs: /var/log/tars-backup.log

Exit Code 2: Partial Success
    └── Extract manifest: tar -xzf backup.tar.gz -O */manifest.json | jq .
    └── Check failed_components array
    └── Retry individual component backup/restore
    └── Accept partial backup if acceptable
```

### Manifest Format

The `manifest.json` file contains metadata about each backup:

```json
{
    "version": "1.0.12",
    "timestamp": "2026-01-03T14:00:00+00:00",
    "backup_name": "tars-backup-20260103_140000",
    "components": [
        "chromadb",
        "postgres",
        "redis",
        "configuration"
    ],
    "checksums": {
        "chromadb.tar.gz": "sha256:a1b2c3d4e5f6789...",
        "postgres.sql.gz": "sha256:b2c3d4e5f6a1789...",
        "redis-dump.rdb": "sha256:c3d4e5f6a1b2789...",
        "config/": "sha256:d4e5f6a1b2c3789..."
    },
    "host": "tars-home-server",
    "docker_volumes": [
        "deploy_chroma_data",
        "deploy_postgres_data",
        "deploy_redis_data"
    ],
    "options": {
        "include_models": false,
        "skip_postgres": false,
        "skip_chromadb": false,
        "skip_redis": false
    },
    "failed_components": []
}
```

#### Manifest Fields

| Field | Type | Description |
|-------|------|-------------|
| `version` | string | T.A.R.S. version at backup time |
| `timestamp` | ISO 8601 | Backup creation timestamp |
| `backup_name` | string | Unique backup identifier |
| `components` | array | Successfully backed up components |
| `checksums` | object | SHA-256 hashes for each component |
| `host` | string | Hostname of source machine |
| `docker_volumes` | array | Docker volume names backed up |
| `options` | object | CLI flags used during backup |
| `failed_components` | array | Components that failed to backup |

### Volume Locations

#### Docker Volume Paths

| Volume Name | Container Path | Host Path (Docker default) |
|-------------|---------------|---------------------------|
| `deploy_chroma_data` | `/chroma/chroma` | `/var/lib/docker/volumes/deploy_chroma_data/_data` |
| `deploy_postgres_data` | `/var/lib/postgresql/data` | `/var/lib/docker/volumes/deploy_postgres_data/_data` |
| `deploy_redis_data` | `/data` | `/var/lib/docker/volumes/deploy_redis_data/_data` |
| `deploy_ollama_data` | `/root/.ollama` | `/var/lib/docker/volumes/deploy_ollama_data/_data` |
| `deploy_backend_logs` | `/app/logs` | `/var/lib/docker/volumes/deploy_backend_logs/_data` |

#### Backup Archive Structure

```
tars-backup-YYYYMMDD_HHMMSS/
├── manifest.json              # Backup metadata and checksums
├── chromadb.tar.gz            # ChromaDB volume (vector embeddings)
├── postgres.sql.gz            # PostgreSQL dump (analytics/audit)
├── redis-dump.rdb             # Redis RDB snapshot (cache)
├── ollama-models.tar.gz       # Ollama volume (optional, if --include-models)
└── config/
    ├── INVENTORY.txt          # Backup inventory and notes
    ├── tars-home.env.redacted # Environment file (secrets redacted)
    ├── docker-compose.home.yml # Docker Compose configuration
    └── tars-home.yml.redacted # TARS config (secrets redacted)
```

---

## Related Documentation

- [OPS_RUNBOOK.md](./OPS_RUNBOOK.md) - General operations runbook
- [INSTALL_HOME.md](./INSTALL_HOME.md) - Home deployment installation guide
- [DEPLOYMENT_VALIDATION.md](./DEPLOYMENT_VALIDATION.md) - Deployment validation procedures
- [CONFIG_DOCTOR.md](./CONFIG_DOCTOR.md) - Configuration troubleshooting
- [SUPPORT_BUNDLE.md](./SUPPORT_BUNDLE.md) - Generating support bundles

---

**Document Version:** 1.0.12
**Last Updated:** January 3, 2026
**Maintainer:** T.A.R.S. DevOps Team
