#!/bin/bash
# ==============================================================================
# T.A.R.S. Restore Script
# Version: v1.0.12 - Phase 25 Backup & Recovery
# Target: Ubuntu 22.04 LTS, Home Network Deployment
# ==============================================================================
#
# Restores T.A.R.S. from a backup archive including:
#   - ChromaDB vector database
#   - PostgreSQL analytics database
#   - Redis cache (RDB snapshot)
#   - Configuration files (optional)
#   - Ollama models (if included in backup)
#
# WARNING: This will STOP all T.A.R.S. services and OVERWRITE current data!
#
# Usage:
#   ./restore-tars.sh --backup-file ./backups/tars-backup-20260103_120000.tar.gz
#   ./restore-tars.sh --backup-file backup.tar.gz --dry-run
#   ./restore-tars.sh --backup-file backup.tar.gz --force  # Skip confirmation
#
# Exit Codes:
#   0 - Success (all components restored)
#   1 - Error (restore failed)
#   2 - Partial restore (some components failed)
#
# ==============================================================================

set -e

# ==============================================================================
# Configuration
# ==============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
COMPOSE_FILE="$SCRIPT_DIR/docker-compose.home.yml"
ENV_FILE="$SCRIPT_DIR/tars-home.env"

# Restore configuration
VERSION="1.0.12"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
TEMP_DIR="/tmp/tars-restore-$TIMESTAMP"
LOCK_FILE="/tmp/.tars-restore.lock"

# Container names
CONTAINER_OLLAMA="tars-home-ollama"
CONTAINER_CHROMADB="tars-home-chromadb"
CONTAINER_REDIS="tars-home-redis"
CONTAINER_POSTGRES="tars-home-postgres"
CONTAINER_BACKEND="tars-home-backend"

# Volume names (base names, will be prefixed with project name)
VOLUME_OLLAMA="ollama_data"
VOLUME_CHROMADB="chroma_data"
VOLUME_POSTGRES="postgres_data"
VOLUME_REDIS="redis_data"
VOLUME_BACKEND_LOGS="backend_logs"

# CLI Flags
BACKUP_FILE=""
SKIP_VALIDATION=false
RESTORE_CONFIG=false
DRY_RUN=false
FORCE=false

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Tracking
COMPONENTS_RESTORED=()
COMPONENTS_FAILED=()
SERVICES_STOPPED=false
ROLLBACK_NEEDED=false

# ==============================================================================
# Logging Functions
# ==============================================================================
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_step() { echo -e "\n${CYAN}>>> $1${NC}"; }
log_dry() { echo -e "${YELLOW}[DRY-RUN]${NC} Would: $1"; }

# ==============================================================================
# Parse CLI Arguments
# ==============================================================================
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --backup-file)
                BACKUP_FILE="$2"
                shift 2
                ;;
            --skip-validation)
                SKIP_VALIDATION=true
                shift
                ;;
            --restore-config)
                RESTORE_CONFIG=true
                shift
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --force)
                FORCE=true
                shift
                ;;
            --help|-h)
                show_help
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done

    # Validate required arguments
    if [[ -z "$BACKUP_FILE" ]]; then
        log_error "Missing required argument: --backup-file"
        echo ""
        show_help
        exit 1
    fi

    # Validate backup file exists
    if [[ ! -f "$BACKUP_FILE" ]]; then
        log_error "Backup file not found: $BACKUP_FILE"
        exit 1
    fi
}

show_help() {
    cat << 'EOF'
T.A.R.S. Restore Script - v1.0.12 (Phase 25)

Restores T.A.R.S. from a backup archive created by backup-tars.sh.

WARNING: This will STOP all T.A.R.S. services and OVERWRITE current data!

USAGE:
    ./restore-tars.sh --backup-file <path> [OPTIONS]

REQUIRED:
    --backup-file <path>   Path to the backup archive (.tar.gz)

OPTIONS:
    --skip-validation      Skip SHA-256 checksum verification
                           (NOT RECOMMENDED - use only if checksum file is missing)

    --restore-config       Also restore configuration files
                           (Environment files, docker-compose, etc.)

    --dry-run              Preview what would happen without making changes

    --force                Skip confirmation prompts
                           (Use with caution!)

    --help, -h             Show this help message

EXAMPLES:
    # Standard restore with prompts
    ./restore-tars.sh --backup-file ./backups/tars-backup-20260103_120000.tar.gz

    # Preview restore operations
    ./restore-tars.sh --backup-file backup.tar.gz --dry-run

    # Automated restore (no prompts)
    ./restore-tars.sh --backup-file backup.tar.gz --force

    # Restore including configuration files
    ./restore-tars.sh --backup-file backup.tar.gz --restore-config

    # Restore with missing checksum file
    ./restore-tars.sh --backup-file backup.tar.gz --skip-validation

RESTORE PROCESS:
    1. Validate backup archive integrity (SHA-256)
    2. Extract to temporary directory
    3. Parse manifest.json for backup contents
    4. Stop T.A.R.S. services (docker compose down)
    5. Restore components in order:
       - PostgreSQL first (database foundation)
       - ChromaDB (vector data)
       - Redis (cache)
       - Configuration (if --restore-config)
       - Ollama models (if included in backup)
    6. Start T.A.R.S. services (docker compose up -d)
    7. Run health checks (wait for services)
    8. Post-restore validation

EXIT CODES:
    0 - Success (all components restored)
    1 - Error (restore failed)
    2 - Partial restore (some components failed)

ROLLBACK:
    If restore fails, the script will attempt to restart services.
    Original data may be lost if volumes were modified.
    ALWAYS test backups before relying on them!

EOF
}

# ==============================================================================
# Lock File Management
# ==============================================================================
acquire_lock() {
    if [[ "$DRY_RUN" == true ]]; then
        log_dry "Acquire lock file: $LOCK_FILE"
        return 0
    fi

    if [[ -f "$LOCK_FILE" ]]; then
        local pid
        pid=$(cat "$LOCK_FILE" 2>/dev/null)
        if kill -0 "$pid" 2>/dev/null; then
            log_error "Another restore operation is already running (PID: $pid)"
            log_error "Lock file: $LOCK_FILE"
            log_error "If you believe this is stale, remove the lock file manually"
            exit 1
        else
            log_warn "Removing stale lock file (PID $pid no longer running)"
            rm -f "$LOCK_FILE"
        fi
    fi

    echo $$ > "$LOCK_FILE"
    log_info "Acquired restore lock (PID: $$)"
}

release_lock() {
    if [[ "$DRY_RUN" == true ]]; then
        return 0
    fi

    if [[ -f "$LOCK_FILE" ]]; then
        rm -f "$LOCK_FILE"
        log_info "Released restore lock"
    fi
}

# ==============================================================================
# Cleanup and Rollback
# ==============================================================================
cleanup() {
    local exit_code=$?
    release_lock

    # Clean up temporary directory
    if [[ -d "$TEMP_DIR" ]] && [[ "$DRY_RUN" != true ]]; then
        log_info "Cleaning up temporary files..."
        rm -rf "$TEMP_DIR"
    fi

    # If services were stopped and we're exiting with error, try to restart
    if [[ "$SERVICES_STOPPED" == true ]] && [[ $exit_code -ne 0 ]] && [[ "$DRY_RUN" != true ]]; then
        log_warn "Restore failed - attempting to restart services..."
        attempt_service_restart
    fi

    exit $exit_code
}

trap cleanup EXIT INT TERM

attempt_service_restart() {
    log_step "Attempting to restart T.A.R.S. services (rollback)..."

    if [[ -f "$COMPOSE_FILE" ]]; then
        if docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" up -d 2>/dev/null; then
            log_success "Services restarted (rollback successful)"
            log_warn "NOTE: Data state may be inconsistent - verify manually"
        else
            log_error "Failed to restart services"
            log_error "Manual intervention required!"
            log_error "Try: docker compose -f $COMPOSE_FILE --env-file $ENV_FILE up -d"
        fi
    else
        log_error "Cannot restart services - compose file not found"
    fi
}

# ==============================================================================
# Utility Functions
# ==============================================================================
check_docker() {
    if ! command -v docker &>/dev/null; then
        log_error "Docker is not installed or not in PATH"
        exit 1
    fi

    if ! docker info &>/dev/null; then
        log_error "Docker daemon is not running or permission denied"
        exit 1
    fi

    log_success "Docker is available and running"
}

check_container_running() {
    local container_name="$1"
    if docker ps --format '{{.Names}}' | grep -q "^${container_name}$"; then
        return 0
    else
        return 1
    fi
}

check_container_exists() {
    local container_name="$1"
    if docker ps -a --format '{{.Names}}' | grep -q "^${container_name}$"; then
        return 0
    else
        return 1
    fi
}

get_volume_name() {
    local base_name="$1"
    local project_prefix
    project_prefix=$(docker compose -f "$COMPOSE_FILE" config --format json 2>/dev/null | jq -r '.name // "deploy"' 2>/dev/null || echo "deploy")

    local full_volume_name="${project_prefix}_${base_name}"

    # Check if full name exists first
    if docker volume ls --format '{{.Name}}' | grep -q "^${full_volume_name}$"; then
        echo "$full_volume_name"
    elif docker volume ls --format '{{.Name}}' | grep -q "^${base_name}$"; then
        echo "$base_name"
    else
        echo "$full_volume_name"
    fi
}

format_size() {
    local size=$1
    if [[ $size -ge 1073741824 ]]; then
        echo "$(awk "BEGIN {printf \"%.2f\", $size/1073741824}") GB"
    elif [[ $size -ge 1048576 ]]; then
        echo "$(awk "BEGIN {printf \"%.2f\", $size/1048576}") MB"
    elif [[ $size -ge 1024 ]]; then
        echo "$(awk "BEGIN {printf \"%.2f\", $size/1024}") KB"
    else
        echo "$size bytes"
    fi
}

# ==============================================================================
# Validation Functions
# ==============================================================================
validate_backup_checksum() {
    log_step "Validating backup archive integrity..."

    if [[ "$SKIP_VALIDATION" == true ]]; then
        log_warn "Skipping checksum validation (--skip-validation)"
        log_warn "This is NOT RECOMMENDED - backup integrity is not verified"
        return 0
    fi

    local checksum_file="${BACKUP_FILE}.sha256"

    if [[ "$DRY_RUN" == true ]]; then
        log_dry "Validate SHA-256 checksum against $checksum_file"
        return 0
    fi

    if [[ ! -f "$checksum_file" ]]; then
        log_error "Checksum file not found: $checksum_file"
        log_error "Cannot verify backup integrity without checksum file"
        log_error "Use --skip-validation to proceed anyway (NOT RECOMMENDED)"
        exit 1
    fi

    log_info "Calculating SHA-256 checksum of backup archive..."
    local expected_checksum
    expected_checksum=$(cat "$checksum_file" | awk '{print $1}')
    local actual_checksum
    actual_checksum=$(sha256sum "$BACKUP_FILE" | awk '{print $1}')

    if [[ "$expected_checksum" == "$actual_checksum" ]]; then
        log_success "Checksum verification passed"
        log_info "  Expected: $expected_checksum"
        log_info "  Actual:   $actual_checksum"
    else
        log_error "Checksum verification FAILED!"
        log_error "  Expected: $expected_checksum"
        log_error "  Actual:   $actual_checksum"
        log_error "The backup file may be corrupted or tampered with"
        exit 1
    fi
}

extract_backup() {
    log_step "Extracting backup archive..."

    if [[ "$DRY_RUN" == true ]]; then
        log_dry "Extract backup to $TEMP_DIR"
        return 0
    fi

    # Create temporary directory
    mkdir -p "$TEMP_DIR"
    log_info "Extracting to: $TEMP_DIR"

    # Extract the archive
    if tar -xzf "$BACKUP_FILE" -C "$TEMP_DIR" 2>/dev/null; then
        # Find the extracted directory (should be tars-backup-TIMESTAMP)
        local extracted_dir
        extracted_dir=$(find "$TEMP_DIR" -mindepth 1 -maxdepth 1 -type d | head -1)

        if [[ -n "$extracted_dir" ]]; then
            # Move contents up one level for easier access
            mv "$extracted_dir"/* "$TEMP_DIR/" 2>/dev/null || true
            rmdir "$extracted_dir" 2>/dev/null || true
        fi

        log_success "Backup extracted successfully"
    else
        log_error "Failed to extract backup archive"
        exit 1
    fi

    # Verify manifest exists
    if [[ ! -f "$TEMP_DIR/manifest.json" ]]; then
        log_error "manifest.json not found in backup archive"
        log_error "This may not be a valid T.A.R.S. backup"
        exit 1
    fi
}

parse_manifest() {
    log_step "Parsing backup manifest..."

    if [[ "$DRY_RUN" == true ]]; then
        log_dry "Parse manifest.json from backup"
        return 0
    fi

    local manifest="$TEMP_DIR/manifest.json"

    if [[ ! -f "$manifest" ]]; then
        log_error "Manifest file not found"
        exit 1
    fi

    # Parse manifest fields
    BACKUP_VERSION=$(jq -r '.version // "unknown"' "$manifest")
    BACKUP_TIMESTAMP=$(jq -r '.timestamp // "unknown"' "$manifest")
    BACKUP_NAME_FROM_MANIFEST=$(jq -r '.backup_name // "unknown"' "$manifest")
    BACKUP_HOST=$(jq -r '.host // "unknown"' "$manifest")

    # Parse components
    BACKUP_COMPONENTS=$(jq -r '.components[]? // empty' "$manifest" 2>/dev/null | tr '\n' ' ')
    BACKUP_HAS_CHROMADB=$(jq -r '.components | if . then (. | index("chromadb") != null) else false end' "$manifest")
    BACKUP_HAS_POSTGRES=$(jq -r '.components | if . then (. | index("postgres") != null) else false end' "$manifest")
    BACKUP_HAS_REDIS=$(jq -r '.components | if . then (. | index("redis") != null) else false end' "$manifest")
    BACKUP_HAS_OLLAMA=$(jq -r '.components | if . then (. | index("ollama") != null) else false end' "$manifest")
    BACKUP_HAS_CONFIG=$(jq -r '.components | if . then (. | index("configuration") != null) else false end' "$manifest")

    # Display manifest info
    log_success "Manifest parsed successfully"
    echo ""
    echo "  Backup Information:"
    echo "    Version:    $BACKUP_VERSION"
    echo "    Timestamp:  $BACKUP_TIMESTAMP"
    echo "    Name:       $BACKUP_NAME_FROM_MANIFEST"
    echo "    Source:     $BACKUP_HOST"
    echo ""
    echo "  Components in backup:"
    [[ "$BACKUP_HAS_POSTGRES" == "true" ]] && echo -e "    ${GREEN}+${NC} PostgreSQL database"
    [[ "$BACKUP_HAS_CHROMADB" == "true" ]] && echo -e "    ${GREEN}+${NC} ChromaDB vector database"
    [[ "$BACKUP_HAS_REDIS" == "true" ]] && echo -e "    ${GREEN}+${NC} Redis cache"
    [[ "$BACKUP_HAS_CONFIG" == "true" ]] && echo -e "    ${GREEN}+${NC} Configuration files"
    [[ "$BACKUP_HAS_OLLAMA" == "true" ]] && echo -e "    ${GREEN}+${NC} Ollama models"
    echo ""
}

# ==============================================================================
# Confirmation Prompt
# ==============================================================================
confirm_restore() {
    if [[ "$FORCE" == true ]]; then
        log_info "Skipping confirmation (--force)"
        return 0
    fi

    if [[ "$DRY_RUN" == true ]]; then
        log_dry "Prompt for confirmation"
        return 0
    fi

    echo ""
    echo -e "${RED}============================================================${NC}"
    echo -e "${RED}   WARNING: DESTRUCTIVE OPERATION${NC}"
    echo -e "${RED}============================================================${NC}"
    echo ""
    echo "This restore operation will:"
    echo ""
    echo "  1. STOP all T.A.R.S. services"
    echo "  2. OVERWRITE existing data in Docker volumes"
    echo "  3. REPLACE database contents"
    echo ""
    echo "Current data will be LOST and replaced with backup data."
    echo ""
    echo -e "${YELLOW}Make sure you have a backup of current data if needed!${NC}"
    echo ""

    read -p "Are you sure you want to proceed? (yes/no): " -r response
    echo ""

    if [[ "$response" != "yes" ]]; then
        log_info "Restore cancelled by user"
        exit 0
    fi

    log_info "User confirmed restore operation"
}

# ==============================================================================
# Service Management
# ==============================================================================
stop_services() {
    log_step "Stopping T.A.R.S. services..."

    if [[ "$DRY_RUN" == true ]]; then
        log_dry "Stop all T.A.R.S. containers via docker compose down"
        return 0
    fi

    # Check if any containers are running
    local running_containers=0
    for container in "$CONTAINER_BACKEND" "$CONTAINER_OLLAMA" "$CONTAINER_CHROMADB" "$CONTAINER_REDIS" "$CONTAINER_POSTGRES"; do
        if check_container_running "$container"; then
            ((running_containers++))
        fi
    done

    if [[ $running_containers -eq 0 ]]; then
        log_info "No T.A.R.S. containers are running"
        return 0
    fi

    log_info "Stopping $running_containers container(s)..."

    # Use docker compose to stop services gracefully
    if [[ -f "$COMPOSE_FILE" ]]; then
        if docker compose -f "$COMPOSE_FILE" down --timeout 30 2>/dev/null; then
            log_success "All services stopped"
            SERVICES_STOPPED=true
        else
            log_warn "docker compose down failed, trying to stop containers individually..."
            for container in "$CONTAINER_BACKEND" "$CONTAINER_OLLAMA" "$CONTAINER_CHROMADB" "$CONTAINER_REDIS" "$CONTAINER_POSTGRES"; do
                if check_container_running "$container"; then
                    docker stop "$container" --time 30 2>/dev/null || true
                fi
            done
            SERVICES_STOPPED=true
            log_success "Containers stopped individually"
        fi
    else
        log_warn "Compose file not found, stopping containers individually..."
        for container in "$CONTAINER_BACKEND" "$CONTAINER_OLLAMA" "$CONTAINER_CHROMADB" "$CONTAINER_REDIS" "$CONTAINER_POSTGRES"; do
            if check_container_running "$container"; then
                docker stop "$container" --time 30 2>/dev/null || true
            fi
        done
        SERVICES_STOPPED=true
        log_success "Containers stopped"
    fi

    # Wait a moment for containers to fully stop
    sleep 2

    # Verify all containers are stopped
    for container in "$CONTAINER_BACKEND" "$CONTAINER_OLLAMA" "$CONTAINER_CHROMADB" "$CONTAINER_REDIS" "$CONTAINER_POSTGRES"; do
        if check_container_running "$container"; then
            log_error "Container $container is still running!"
            log_error "Please stop it manually before proceeding"
            exit 1
        fi
    done

    log_success "All T.A.R.S. containers stopped"
}

start_services() {
    log_step "Starting T.A.R.S. services..."

    if [[ "$DRY_RUN" == true ]]; then
        log_dry "Start all T.A.R.S. containers via docker compose up -d"
        return 0
    fi

    if [[ ! -f "$COMPOSE_FILE" ]]; then
        log_error "Docker Compose file not found: $COMPOSE_FILE"
        log_error "Cannot start services without compose file"
        return 1
    fi

    if [[ ! -f "$ENV_FILE" ]]; then
        log_warn "Environment file not found: $ENV_FILE"
        log_warn "Services may fail to start without proper configuration"
    fi

    log_info "Starting services via docker compose..."

    if docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" up -d 2>/dev/null; then
        log_success "Docker compose up completed"
        SERVICES_STOPPED=false
    else
        log_error "Failed to start services"
        log_error "Try manually: docker compose -f $COMPOSE_FILE --env-file $ENV_FILE up -d"
        return 1
    fi
}

# ==============================================================================
# Restore: PostgreSQL Database
# ==============================================================================
restore_postgres() {
    log_step "Restoring PostgreSQL database..."

    if [[ "$BACKUP_HAS_POSTGRES" != "true" ]]; then
        log_warn "PostgreSQL not included in backup - skipping"
        return 0
    fi

    local backup_file="$TEMP_DIR/postgres.sql.gz"

    if [[ "$DRY_RUN" == true ]]; then
        log_dry "Restore PostgreSQL from postgres.sql.gz"
        COMPONENTS_RESTORED+=("postgres")
        return 0
    fi

    if [[ ! -f "$backup_file" ]]; then
        log_error "PostgreSQL backup file not found: postgres.sql.gz"
        COMPONENTS_FAILED+=("postgres")
        return 1
    fi

    local volume_name
    volume_name=$(get_volume_name "$VOLUME_POSTGRES")

    # Clear existing volume data
    log_info "Clearing existing PostgreSQL volume..."
    if docker volume ls --format '{{.Name}}' | grep -qE "^${volume_name}$"; then
        # Remove and recreate the volume
        docker volume rm "$volume_name" 2>/dev/null || true
    fi
    docker volume create "$volume_name" >/dev/null 2>&1 || true

    # Start PostgreSQL container temporarily to restore data
    log_info "Starting temporary PostgreSQL container..."

    # Get credentials from environment or use defaults
    local pg_user="${TARS_POSTGRES_USER:-tars}"
    local pg_db="${TARS_POSTGRES_DB:-tars_home}"
    local pg_password="${TARS_POSTGRES_PASSWORD:-}"

    if [[ -z "$pg_password" ]] && [[ -f "$ENV_FILE" ]]; then
        pg_password=$(grep "^TARS_POSTGRES_PASSWORD=" "$ENV_FILE" 2>/dev/null | cut -d'=' -f2-)
    fi

    if [[ -z "$pg_password" ]]; then
        log_error "PostgreSQL password not found in environment"
        log_error "Set TARS_POSTGRES_PASSWORD or ensure $ENV_FILE exists"
        COMPONENTS_FAILED+=("postgres")
        return 1
    fi

    # Start temporary postgres container
    local temp_container="tars-restore-postgres-$$"
    docker run -d \
        --name "$temp_container" \
        -e POSTGRES_USER="$pg_user" \
        -e POSTGRES_DB="$pg_db" \
        -e POSTGRES_PASSWORD="$pg_password" \
        -v "${volume_name}:/var/lib/postgresql/data" \
        postgres:15-alpine >/dev/null 2>&1

    # Wait for PostgreSQL to be ready
    log_info "Waiting for PostgreSQL to initialize..."
    local wait_count=0
    local max_wait=60
    while [[ $wait_count -lt $max_wait ]]; do
        if docker exec "$temp_container" pg_isready -U "$pg_user" -d "$pg_db" &>/dev/null; then
            break
        fi
        sleep 1
        ((wait_count++))
    done

    if [[ $wait_count -ge $max_wait ]]; then
        log_error "PostgreSQL failed to start within ${max_wait}s"
        docker stop "$temp_container" &>/dev/null || true
        docker rm "$temp_container" &>/dev/null || true
        COMPONENTS_FAILED+=("postgres")
        return 1
    fi

    log_success "PostgreSQL container ready"

    # Restore the database
    log_info "Restoring database from backup..."
    if gunzip -c "$backup_file" | docker exec -i "$temp_container" psql -U "$pg_user" -d "$pg_db" &>/dev/null; then
        log_success "PostgreSQL database restored"
        COMPONENTS_RESTORED+=("postgres")
    else
        log_error "Failed to restore PostgreSQL database"
        COMPONENTS_FAILED+=("postgres")
    fi

    # Stop and remove temporary container
    log_info "Cleaning up temporary container..."
    docker stop "$temp_container" &>/dev/null || true
    docker rm "$temp_container" &>/dev/null || true
}

# ==============================================================================
# Restore: ChromaDB Vector Database
# ==============================================================================
restore_chromadb() {
    log_step "Restoring ChromaDB vector database..."

    if [[ "$BACKUP_HAS_CHROMADB" != "true" ]]; then
        log_warn "ChromaDB not included in backup - skipping"
        return 0
    fi

    local backup_file="$TEMP_DIR/chromadb.tar.gz"

    if [[ "$DRY_RUN" == true ]]; then
        log_dry "Restore ChromaDB volume from chromadb.tar.gz"
        COMPONENTS_RESTORED+=("chromadb")
        return 0
    fi

    if [[ ! -f "$backup_file" ]]; then
        log_error "ChromaDB backup file not found: chromadb.tar.gz"
        COMPONENTS_FAILED+=("chromadb")
        return 1
    fi

    local volume_name
    volume_name=$(get_volume_name "$VOLUME_CHROMADB")

    # Clear existing volume data
    log_info "Clearing existing ChromaDB volume..."
    if docker volume ls --format '{{.Name}}' | grep -qE "^${volume_name}$"; then
        docker volume rm "$volume_name" 2>/dev/null || true
    fi
    docker volume create "$volume_name" >/dev/null 2>&1 || true

    # Restore volume from tar
    log_info "Restoring ChromaDB volume from backup..."
    if docker run --rm \
        -v "${volume_name}:/data" \
        -v "$TEMP_DIR:/backup:ro" \
        alpine:latest \
        tar -xzf /backup/chromadb.tar.gz -C /data 2>/dev/null; then

        log_success "ChromaDB volume restored"
        COMPONENTS_RESTORED+=("chromadb")
    else
        log_error "Failed to restore ChromaDB volume"
        COMPONENTS_FAILED+=("chromadb")
        return 1
    fi
}

# ==============================================================================
# Restore: Redis Cache
# ==============================================================================
restore_redis() {
    log_step "Restoring Redis cache..."

    if [[ "$BACKUP_HAS_REDIS" != "true" ]]; then
        log_warn "Redis not included in backup - skipping"
        return 0
    fi

    local backup_file="$TEMP_DIR/redis-dump.rdb"

    if [[ "$DRY_RUN" == true ]]; then
        log_dry "Restore Redis RDB snapshot from redis-dump.rdb"
        COMPONENTS_RESTORED+=("redis")
        return 0
    fi

    if [[ ! -f "$backup_file" ]]; then
        log_error "Redis backup file not found: redis-dump.rdb"
        COMPONENTS_FAILED+=("redis")
        return 1
    fi

    local volume_name
    volume_name=$(get_volume_name "$VOLUME_REDIS")

    # Clear existing volume data
    log_info "Clearing existing Redis volume..."
    if docker volume ls --format '{{.Name}}' | grep -qE "^${volume_name}$"; then
        docker volume rm "$volume_name" 2>/dev/null || true
    fi
    docker volume create "$volume_name" >/dev/null 2>&1 || true

    # Copy RDB file to volume
    log_info "Restoring Redis RDB snapshot..."
    if docker run --rm \
        -v "${volume_name}:/data" \
        -v "$TEMP_DIR:/backup:ro" \
        alpine:latest \
        cp /backup/redis-dump.rdb /data/dump.rdb 2>/dev/null; then

        log_success "Redis cache restored"
        COMPONENTS_RESTORED+=("redis")
    else
        log_error "Failed to restore Redis cache"
        COMPONENTS_FAILED+=("redis")
        return 1
    fi
}

# ==============================================================================
# Restore: Ollama Models (if present)
# ==============================================================================
restore_ollama() {
    log_step "Restoring Ollama models..."

    if [[ "$BACKUP_HAS_OLLAMA" != "true" ]]; then
        log_info "Ollama models not included in backup - skipping"
        return 0
    fi

    local backup_file="$TEMP_DIR/ollama-models.tar.gz"

    if [[ "$DRY_RUN" == true ]]; then
        log_dry "Restore Ollama models from ollama-models.tar.gz"
        COMPONENTS_RESTORED+=("ollama")
        return 0
    fi

    if [[ ! -f "$backup_file" ]]; then
        log_warn "Ollama backup file not found but was listed in manifest"
        log_warn "Models will need to be re-downloaded"
        return 0
    fi

    local volume_name
    volume_name=$(get_volume_name "$VOLUME_OLLAMA")

    # Clear existing volume data
    log_info "Clearing existing Ollama volume..."
    if docker volume ls --format '{{.Name}}' | grep -qE "^${volume_name}$"; then
        docker volume rm "$volume_name" 2>/dev/null || true
    fi
    docker volume create "$volume_name" >/dev/null 2>&1 || true

    # Restore volume from tar
    log_info "Restoring Ollama volume from backup (this may take a while)..."
    if docker run --rm \
        -v "${volume_name}:/data" \
        -v "$TEMP_DIR:/backup:ro" \
        alpine:latest \
        tar -xzf /backup/ollama-models.tar.gz -C /data 2>/dev/null; then

        log_success "Ollama models restored"
        COMPONENTS_RESTORED+=("ollama")
    else
        log_error "Failed to restore Ollama models"
        COMPONENTS_FAILED+=("ollama")
        return 1
    fi
}

# ==============================================================================
# Restore: Configuration Files (Optional)
# ==============================================================================
restore_configuration() {
    log_step "Restoring configuration files..."

    if [[ "$RESTORE_CONFIG" != true ]]; then
        log_info "Skipping configuration restore (use --restore-config to include)"
        return 0
    fi

    if [[ "$BACKUP_HAS_CONFIG" != "true" ]]; then
        log_warn "Configuration not included in backup - skipping"
        return 0
    fi

    local config_dir="$TEMP_DIR/config"

    if [[ "$DRY_RUN" == true ]]; then
        log_dry "Restore configuration files from config/"
        log_dry "NOTE: Secrets are redacted in backup, manual re-entry required"
        COMPONENTS_RESTORED+=("configuration")
        return 0
    fi

    if [[ ! -d "$config_dir" ]]; then
        log_warn "Configuration directory not found in backup"
        return 0
    fi

    log_warn "Configuration files in backup have REDACTED secrets"
    log_warn "You will need to manually re-enter sensitive values"

    # Restore docker-compose file if present
    if [[ -f "$config_dir/docker-compose.home.yml" ]]; then
        log_info "Restoring docker-compose.home.yml..."
        cp "$config_dir/docker-compose.home.yml" "$COMPOSE_FILE"
        log_success "  docker-compose.home.yml restored"
    fi

    # Note about environment file
    if [[ -f "$config_dir/tars-home.env.redacted" ]]; then
        log_info "Found redacted environment file"
        log_warn "  tars-home.env contains REDACTED secrets"
        log_warn "  Copy as reference: $config_dir/tars-home.env.redacted"
        log_warn "  You must manually set secrets in: $ENV_FILE"

        # Copy as .reference if env file doesn't exist
        if [[ ! -f "$ENV_FILE" ]]; then
            cp "$config_dir/tars-home.env.redacted" "${ENV_FILE}.reference"
            log_info "  Created reference file: ${ENV_FILE}.reference"
        fi
    fi

    # Restore tars-home.yml if present
    if [[ -f "$config_dir/tars-home.yml.redacted" ]]; then
        log_info "Found redacted config file"
        log_warn "  tars-home.yml contains REDACTED secrets"
        if [[ ! -f "$SCRIPT_DIR/tars-home.yml" ]]; then
            cp "$config_dir/tars-home.yml.redacted" "$SCRIPT_DIR/tars-home.yml.reference"
            log_info "  Created reference file: $SCRIPT_DIR/tars-home.yml.reference"
        fi
    fi

    COMPONENTS_RESTORED+=("configuration")
    log_success "Configuration files restored (secrets need manual entry)"
}

# ==============================================================================
# Health Check Functions
# ==============================================================================
wait_for_healthy() {
    local container="$1"
    local endpoint="$2"
    local max_wait="${3:-120}"
    local wait_count=0

    while [[ $wait_count -lt $max_wait ]]; do
        if [[ -n "$endpoint" ]]; then
            if curl -sf --max-time 5 "$endpoint" &>/dev/null; then
                return 0
            fi
        else
            # Check container health status
            local health
            health=$(docker inspect "$container" --format '{{.State.Health.Status}}' 2>/dev/null || echo "unknown")
            if [[ "$health" == "healthy" ]]; then
                return 0
            fi
            # Also check if container is running (for containers without healthcheck)
            if docker ps --format '{{.Names}}' | grep -q "^${container}$"; then
                local status
                status=$(docker inspect "$container" --format '{{.State.Status}}' 2>/dev/null)
                if [[ "$status" == "running" ]]; then
                    # If no healthcheck defined, treat running as healthy
                    local has_healthcheck
                    has_healthcheck=$(docker inspect "$container" --format '{{if .State.Health}}yes{{end}}' 2>/dev/null)
                    if [[ -z "$has_healthcheck" ]]; then
                        return 0
                    fi
                fi
            fi
        fi
        sleep 2
        ((wait_count+=2))
    done

    return 1
}

run_health_checks() {
    log_step "Running post-restore health checks..."

    if [[ "$DRY_RUN" == true ]]; then
        log_dry "Wait for all services to become healthy"
        log_dry "Check health endpoints for each service"
        return 0
    fi

    local all_healthy=true

    # Check PostgreSQL
    log_info "Checking PostgreSQL..."
    if wait_for_healthy "$CONTAINER_POSTGRES" "" 60; then
        # Also verify we can connect
        if docker exec "$CONTAINER_POSTGRES" pg_isready -U tars -d tars_home &>/dev/null; then
            log_success "  PostgreSQL: healthy"
        else
            log_warn "  PostgreSQL: container running but pg_isready failed"
            all_healthy=false
        fi
    else
        log_error "  PostgreSQL: not healthy"
        all_healthy=false
    fi

    # Check Redis
    log_info "Checking Redis..."
    if wait_for_healthy "$CONTAINER_REDIS" "" 30; then
        if docker exec "$CONTAINER_REDIS" redis-cli ping 2>/dev/null | grep -q "PONG"; then
            log_success "  Redis: healthy"
        else
            log_warn "  Redis: container running but ping failed"
            all_healthy=false
        fi
    else
        log_error "  Redis: not healthy"
        all_healthy=false
    fi

    # Check ChromaDB
    log_info "Checking ChromaDB..."
    if wait_for_healthy "$CONTAINER_CHROMADB" "http://localhost:8001/api/v1/heartbeat" 60; then
        log_success "  ChromaDB: healthy"
    else
        log_error "  ChromaDB: not healthy"
        all_healthy=false
    fi

    # Check Ollama
    log_info "Checking Ollama..."
    if wait_for_healthy "$CONTAINER_OLLAMA" "http://localhost:11434/api/tags" 90; then
        log_success "  Ollama: healthy"
    else
        log_warn "  Ollama: not responding (may need longer to load models)"
        # Don't fail completely for Ollama
    fi

    # Check Backend (last, depends on others)
    log_info "Checking Backend API..."
    if wait_for_healthy "$CONTAINER_BACKEND" "http://localhost:8000/health" 120; then
        log_success "  Backend: healthy"
    else
        log_error "  Backend: not healthy"
        all_healthy=false
    fi

    if [[ "$all_healthy" == true ]]; then
        log_success "All services are healthy!"
        return 0
    else
        log_warn "Some services failed health checks"
        log_warn "Check container logs: docker compose -f $COMPOSE_FILE logs"
        return 1
    fi
}

# ==============================================================================
# Post-Restore Validation
# ==============================================================================
post_restore_validation() {
    log_step "Running post-restore validation..."

    if [[ "$DRY_RUN" == true ]]; then
        log_dry "Validate restored data integrity"
        return 0
    fi

    local validation_passed=true

    # Check ChromaDB has collections
    log_info "Checking ChromaDB collections..."
    local collections
    collections=$(curl -sf --max-time 10 "http://localhost:8001/api/v1/collections" 2>/dev/null | jq -r '. | length' 2>/dev/null || echo "0")
    if [[ "$collections" -gt 0 ]]; then
        log_success "  ChromaDB: $collections collection(s) found"
    else
        log_warn "  ChromaDB: No collections found (may be empty backup)"
    fi

    # Check PostgreSQL tables
    log_info "Checking PostgreSQL tables..."
    local tables
    tables=$(docker exec "$CONTAINER_POSTGRES" psql -U tars -d tars_home -t -c "SELECT count(*) FROM information_schema.tables WHERE table_schema = 'public';" 2>/dev/null | tr -d ' ' || echo "0")
    if [[ "$tables" -gt 0 ]]; then
        log_success "  PostgreSQL: $tables table(s) found"
    else
        log_warn "  PostgreSQL: No tables found (may be empty backup)"
    fi

    # Check Redis keys
    log_info "Checking Redis keys..."
    local redis_keys
    redis_keys=$(docker exec "$CONTAINER_REDIS" redis-cli DBSIZE 2>/dev/null | awk '{print $2}' || echo "0")
    log_info "  Redis: $redis_keys key(s) in database"

    # Basic API check
    log_info "Checking API readiness..."
    local api_ready
    api_ready=$(curl -sf --max-time 10 "http://localhost:8000/ready" 2>/dev/null | jq -r '.status // "unknown"' 2>/dev/null || echo "error")
    if [[ "$api_ready" == "ready" ]] || [[ "$api_ready" == "ok" ]]; then
        log_success "  API: Ready"
    else
        log_warn "  API: Status = $api_ready"
    fi

    if [[ "$validation_passed" == true ]]; then
        log_success "Post-restore validation complete"
    else
        log_warn "Some validations had warnings - verify manually"
    fi
}

# ==============================================================================
# Dry Run Summary
# ==============================================================================
show_dry_run_summary() {
    echo ""
    echo -e "${CYAN}============================================================${NC}"
    echo -e "${CYAN}   Dry Run Summary${NC}"
    echo -e "${CYAN}============================================================${NC}"
    echo ""
    echo "The following operations would be performed:"
    echo ""
    echo "  Backup file: $BACKUP_FILE"
    echo "  Temp directory: $TEMP_DIR"
    echo ""
    echo "  Steps:"
    echo "    1. Validate backup checksum (SHA-256)"
    echo "    2. Extract backup to temporary directory"
    echo "    3. Parse manifest.json"
    echo "    4. Stop all T.A.R.S. services"
    echo "    5. Restore components:"

    echo -e "       ${GREEN}+${NC} PostgreSQL database (primary)"
    echo -e "       ${GREEN}+${NC} ChromaDB vector database"
    echo -e "       ${GREEN}+${NC} Redis cache"
    [[ "$RESTORE_CONFIG" == true ]] && echo -e "       ${GREEN}+${NC} Configuration files"
    echo -e "       ${BLUE}?${NC} Ollama models (if in backup)"

    echo "    6. Start T.A.R.S. services"
    echo "    7. Run health checks"
    echo "    8. Validate restored data"
    echo ""
    echo -e "${YELLOW}No changes were made (dry run mode)${NC}"
    echo ""
}

# ==============================================================================
# Final Summary
# ==============================================================================
show_summary() {
    echo ""
    echo -e "${CYAN}============================================================${NC}"
    echo -e "${CYAN}   Restore Complete${NC}"
    echo -e "${CYAN}============================================================${NC}"
    echo ""
    echo "  Backup file: $BACKUP_FILE"
    echo "  Backup date: $BACKUP_TIMESTAMP"
    echo ""
    echo "  Components restored:"
    for component in "${COMPONENTS_RESTORED[@]}"; do
        echo -e "    ${GREEN}+${NC} $component"
    done

    if [[ ${#COMPONENTS_FAILED[@]} -gt 0 ]]; then
        echo ""
        echo "  Components failed:"
        for component in "${COMPONENTS_FAILED[@]}"; do
            echo -e "    ${RED}-${NC} $component"
        done
    fi
    echo ""
}

# ==============================================================================
# Pre-flight Checks
# ==============================================================================
preflight_checks() {
    log_step "Running pre-flight checks..."

    # Check Docker availability
    check_docker

    # Check if compose file exists
    if [[ -f "$COMPOSE_FILE" ]]; then
        log_success "Docker Compose file found"
    else
        log_warn "Docker Compose file not found: $COMPOSE_FILE"
        log_warn "Will not be able to start services automatically"
    fi

    # Check if env file exists
    if [[ -f "$ENV_FILE" ]]; then
        log_success "Environment file found"
    else
        log_warn "Environment file not found: $ENV_FILE"
        log_warn "Services may fail to start without configuration"
    fi

    # Check backup file size
    local backup_size
    backup_size=$(stat -c%s "$BACKUP_FILE" 2>/dev/null || echo "0")
    log_info "Backup file size: $(format_size "$backup_size")"

    # Check available disk space
    local available_space
    available_space=$(df --output=avail -B1 "/tmp" 2>/dev/null | tail -1)
    log_info "Available temp space: $(format_size "$available_space")"

    # Need at least 2x backup size for extraction
    local required_space=$((backup_size * 3))
    if [[ "$available_space" -lt "$required_space" ]]; then
        log_warn "Low disk space - may need $(format_size "$required_space") for restore"
    fi

    log_success "Pre-flight checks complete"
}

# ==============================================================================
# Main
# ==============================================================================
main() {
    echo -e "${CYAN}===============================================================${NC}"
    echo -e "${CYAN}   T.A.R.S. Restore Script - v${VERSION} (Phase 25)              ${NC}"
    echo -e "${CYAN}===============================================================${NC}"
    echo ""

    # Parse command line arguments
    parse_args "$@"

    # Show configuration
    log_info "Restore configuration:"
    log_info "  Backup file: $BACKUP_FILE"
    log_info "  Skip validation: $SKIP_VALIDATION"
    log_info "  Restore config: $RESTORE_CONFIG"
    log_info "  Force mode: $FORCE"
    if [[ "$DRY_RUN" == true ]]; then
        echo -e "  ${YELLOW}Mode: DRY RUN (no changes will be made)${NC}"
    fi
    echo ""

    # Acquire lock
    acquire_lock

    # Run pre-flight checks
    preflight_checks

    # Validate backup integrity
    validate_backup_checksum

    # Extract backup archive
    extract_backup

    # Parse manifest
    parse_manifest

    # Confirm with user (unless --force)
    confirm_restore

    # Stop services
    stop_services

    # Restore components in order
    # PostgreSQL first (foundation)
    restore_postgres || true

    # ChromaDB (vector data)
    restore_chromadb || true

    # Redis (cache)
    restore_redis || true

    # Ollama models (if present)
    restore_ollama || true

    # Configuration (optional)
    restore_configuration || true

    # Start services
    if ! start_services; then
        log_error "Failed to start services after restore"
        exit 1
    fi

    # Run health checks
    if [[ "$DRY_RUN" != true ]]; then
        run_health_checks || true
        post_restore_validation || true
    fi

    # Show summary
    if [[ "$DRY_RUN" == true ]]; then
        show_dry_run_summary
        exit 0
    else
        show_summary
    fi

    # Determine exit code
    local exit_code=0
    if [[ ${#COMPONENTS_FAILED[@]} -gt 0 ]]; then
        if [[ ${#COMPONENTS_RESTORED[@]} -gt 0 ]]; then
            log_warn "Restore completed with some failures"
            exit_code=2
        else
            log_error "Restore failed - no components were restored"
            exit_code=1
        fi
    else
        log_success "Restore completed successfully!"
    fi

    echo ""
    echo "Next steps:"
    echo "  1. Verify services are running: docker compose -f $COMPOSE_FILE ps"
    echo "  2. Check logs for any errors: docker compose -f $COMPOSE_FILE logs"
    echo "  3. Test application functionality"
    if [[ "$RESTORE_CONFIG" == true ]]; then
        echo "  4. Re-enter secrets in $ENV_FILE (they were redacted in backup)"
    fi
    echo ""

    exit $exit_code
}

main "$@"
