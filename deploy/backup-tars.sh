#!/bin/bash
# ==============================================================================
# T.A.R.S. Backup Script
# Version: v1.0.12 - Phase 25 Backup & Recovery
# Target: Ubuntu 22.04 LTS, Home Network Deployment
# ==============================================================================
#
# Creates a complete backup of T.A.R.S. data including:
#   - ChromaDB vector database
#   - PostgreSQL analytics database
#   - Redis cache (RDB snapshot)
#   - Configuration files (secrets redacted)
#   - Ollama models (optional, can be large)
#
# Usage:
#   ./backup-tars.sh                          # Standard backup
#   ./backup-tars.sh --include-models         # Include Ollama models
#   ./backup-tars.sh --output-dir /mnt/nas    # Custom output location
#   ./backup-tars.sh --dry-run                # Preview operations
#
# Exit Codes:
#   0 - Success (all components backed up)
#   1 - Error (backup failed)
#   2 - Partial success (some components failed)
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

# Backup configuration
VERSION="1.0.12"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_NAME="tars-backup-$TIMESTAMP"
OUTPUT_DIR="$PROJECT_ROOT/backups"
LOCK_FILE="/tmp/.tars-backup.lock"

# Container names
CONTAINER_OLLAMA="tars-home-ollama"
CONTAINER_CHROMADB="tars-home-chromadb"
CONTAINER_REDIS="tars-home-redis"
CONTAINER_POSTGRES="tars-home-postgres"
CONTAINER_BACKEND="tars-home-backend"

# Volume names
VOLUME_OLLAMA="ollama_data"
VOLUME_CHROMADB="chroma_data"
VOLUME_POSTGRES="postgres_data"
VOLUME_REDIS="redis_data"
VOLUME_BACKEND_LOGS="backend_logs"

# CLI Flags
INCLUDE_MODELS=false
SKIP_POSTGRES=false
SKIP_CHROMADB=false
SKIP_REDIS=false
DRY_RUN=false

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Tracking
COMPONENTS_BACKED_UP=()
COMPONENTS_FAILED=()
COMPONENT_CHECKSUMS=()

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
            --output-dir)
                OUTPUT_DIR="$2"
                shift 2
                ;;
            --include-models)
                INCLUDE_MODELS=true
                shift
                ;;
            --skip-postgres)
                SKIP_POSTGRES=true
                shift
                ;;
            --skip-chromadb)
                SKIP_CHROMADB=true
                shift
                ;;
            --skip-redis)
                SKIP_REDIS=true
                shift
                ;;
            --dry-run)
                DRY_RUN=true
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
}

show_help() {
    cat << 'EOF'
T.A.R.S. Backup Script - v1.0.12 (Phase 25)

Creates a complete backup of T.A.R.S. data including databases, cache,
configuration files, and optionally Ollama models.

USAGE:
    ./backup-tars.sh [OPTIONS]

OPTIONS:
    --output-dir <path>    Output directory for backup archive
                           (default: ./backups)

    --include-models       Include Ollama model backup (optional)
                           WARNING: This can be very large (10+ GB)

    --skip-postgres        Skip PostgreSQL database backup

    --skip-chromadb        Skip ChromaDB vector database backup

    --skip-redis           Skip Redis cache backup

    --dry-run              Preview what would happen without creating files

    --help, -h             Show this help message

EXAMPLES:
    # Standard backup (ChromaDB, PostgreSQL, Redis, config)
    ./backup-tars.sh

    # Include Ollama models (large!)
    ./backup-tars.sh --include-models

    # Backup to NAS mount point
    ./backup-tars.sh --output-dir /mnt/nas/backups

    # Preview operations without making changes
    ./backup-tars.sh --dry-run

    # Backup only PostgreSQL and Redis (skip ChromaDB)
    ./backup-tars.sh --skip-chromadb

OUTPUT:
    Creates a timestamped archive: tars-backup-YYYYMMDD_HHMMSS.tar.gz
    With SHA-256 checksum file: tars-backup-YYYYMMDD_HHMMSS.tar.gz.sha256

COMPONENTS:
    Component    | Volume         | Method
    -------------|----------------|---------------------------
    ChromaDB     | chroma_data    | Docker volume tar
    PostgreSQL   | postgres_data  | pg_dump via docker exec
    Redis        | redis_data     | BGSAVE + RDB copy
    Config       | tars-home.env  | Copy with secret redaction
    Ollama       | ollama_data    | Docker volume tar (optional)

EXIT CODES:
    0 - Success (all components backed up)
    1 - Error (backup failed)
    2 - Partial success (some components failed)

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
            log_error "Another backup is already running (PID: $pid)"
            log_error "Lock file: $LOCK_FILE"
            log_error "If you believe this is stale, remove the lock file manually"
            exit 1
        else
            log_warn "Removing stale lock file (PID $pid no longer running)"
            rm -f "$LOCK_FILE"
        fi
    fi

    echo $$ > "$LOCK_FILE"
    log_info "Acquired backup lock (PID: $$)"
}

release_lock() {
    if [[ "$DRY_RUN" == true ]]; then
        return 0
    fi

    if [[ -f "$LOCK_FILE" ]]; then
        rm -f "$LOCK_FILE"
        log_info "Released backup lock"
    fi
}

# Trap to ensure lock is released on exit
cleanup() {
    local exit_code=$?
    release_lock

    # Clean up temporary backup directory on failure
    if [[ $exit_code -ne 0 ]] && [[ -d "$BACKUP_DIR" ]] && [[ "$DRY_RUN" != true ]]; then
        log_warn "Cleaning up partial backup directory: $BACKUP_DIR"
        rm -rf "$BACKUP_DIR"
    fi

    exit $exit_code
}

trap cleanup EXIT INT TERM

# ==============================================================================
# Secret Redaction Functions
# ==============================================================================
redact_secrets() {
    # Redact common secret patterns
    sed -E \
        -e 's/(PASSWORD|SECRET|KEY|TOKEN|CREDENTIAL)[[:space:]]*[:=][[:space:]]*[^[:space:]]+/\1=<REDACTED>/gi' \
        -e 's/(Bearer|Basic)[[:space:]]+[A-Za-z0-9+/=_-]+/\1 <REDACTED>/gi' \
        -e 's/([a-f0-9]{32,}|[A-Za-z0-9+/=]{32,})/<HASH_REDACTED>/g' \
        -e 's/(postgres:\/\/[^:]+:)[^@]+(@)/\1<REDACTED>\2/g' \
        -e 's/(redis:\/\/[^:]+:)[^@]+(@)/\1<REDACTED>\2/g' \
        -e 's/eyJ[A-Za-z0-9_-]*\.[A-Za-z0-9_-]*\.[A-Za-z0-9_-]*/<JWT_REDACTED>/g'
}

redact_env_file() {
    # Show variable names but redact all values that might be sensitive
    while IFS= read -r line; do
        # Skip empty lines and comments
        if [[ -z "$line" ]] || [[ "$line" =~ ^[[:space:]]*# ]]; then
            echo "$line"
            continue
        fi

        # Check if it's a variable assignment
        if [[ "$line" =~ ^([A-Za-z_][A-Za-z0-9_]*)= ]]; then
            var_name="${BASH_REMATCH[1]}"

            # Redact sensitive variables
            if [[ "$var_name" =~ PASSWORD|SECRET|KEY|TOKEN|CREDENTIAL|JWT ]]; then
                echo "${var_name}=<REDACTED>"
            elif [[ "$var_name" =~ ^(NAS_|POSTGRES_|REDIS_|API_|TARS_) ]]; then
                # Show presence but not value for connection strings
                echo "${var_name}=<SET>"
            else
                # Safe variables can be shown
                echo "$line"
            fi
        else
            echo "$line"
        fi
    done
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

check_volume_exists() {
    local volume_name="$1"
    # Get the project prefix from docker compose
    local project_prefix
    project_prefix=$(docker compose -f "$COMPOSE_FILE" config --format json 2>/dev/null | jq -r '.name // "deploy"' 2>/dev/null || echo "deploy")

    local full_volume_name="${project_prefix}_${volume_name}"

    if docker volume ls --format '{{.Name}}' | grep -qE "^${full_volume_name}$|^${volume_name}$"; then
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

calculate_checksum() {
    local file="$1"
    if [[ -f "$file" ]]; then
        sha256sum "$file" | awk '{print $1}'
    else
        echo "file_not_found"
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
# Backup: ChromaDB Vector Database
# ==============================================================================
backup_chromadb() {
    log_step "Backing up ChromaDB vector database..."

    if [[ "$SKIP_CHROMADB" == true ]]; then
        log_warn "Skipping ChromaDB backup (--skip-chromadb)"
        return 0
    fi

    local volume_name
    volume_name=$(get_volume_name "$VOLUME_CHROMADB")
    local output_file="$BACKUP_DIR/chromadb.tar.gz"

    if [[ "$DRY_RUN" == true ]]; then
        log_dry "Backup ChromaDB volume '$volume_name' to chromadb.tar.gz"
        COMPONENTS_BACKED_UP+=("chromadb")
        return 0
    fi

    # Check if volume exists
    if ! docker volume ls --format '{{.Name}}' | grep -qE "^${volume_name}$"; then
        log_warn "ChromaDB volume '$volume_name' not found"
        COMPONENTS_FAILED+=("chromadb")
        return 1
    fi

    log_info "Backing up volume: $volume_name"

    # Create a temporary container to access the volume and create a tar backup
    if docker run --rm \
        -v "${volume_name}:/data:ro" \
        -v "$BACKUP_DIR:/backup" \
        alpine:latest \
        tar -czf /backup/chromadb.tar.gz -C /data . 2>/dev/null; then

        local size
        size=$(stat -c%s "$output_file" 2>/dev/null || echo "0")
        local checksum
        checksum=$(calculate_checksum "$output_file")

        log_success "ChromaDB backup complete ($(format_size "$size"))"
        COMPONENTS_BACKED_UP+=("chromadb")
        COMPONENT_CHECKSUMS+=("chromadb.tar.gz:$checksum")
    else
        log_error "ChromaDB backup failed"
        COMPONENTS_FAILED+=("chromadb")
        return 1
    fi
}

# ==============================================================================
# Backup: PostgreSQL Database
# ==============================================================================
backup_postgres() {
    log_step "Backing up PostgreSQL database..."

    if [[ "$SKIP_POSTGRES" == true ]]; then
        log_warn "Skipping PostgreSQL backup (--skip-postgres)"
        return 0
    fi

    local output_file="$BACKUP_DIR/postgres.sql.gz"

    if [[ "$DRY_RUN" == true ]]; then
        log_dry "Run pg_dump in container '$CONTAINER_POSTGRES' to postgres.sql.gz"
        COMPONENTS_BACKED_UP+=("postgres")
        return 0
    fi

    # Check if container is running
    if ! check_container_running "$CONTAINER_POSTGRES"; then
        log_warn "PostgreSQL container '$CONTAINER_POSTGRES' is not running"
        COMPONENTS_FAILED+=("postgres")
        return 1
    fi

    log_info "Running pg_dump via docker exec..."

    # Get database credentials from environment or use defaults
    local pg_user="${TARS_POSTGRES_USER:-tars}"
    local pg_db="${TARS_POSTGRES_DB:-tars_home}"

    # Run pg_dump inside the container and compress output
    if docker exec "$CONTAINER_POSTGRES" \
        pg_dump -U "$pg_user" -d "$pg_db" --clean --if-exists --no-owner --no-privileges 2>/dev/null \
        | gzip > "$output_file"; then

        # Check if file is not empty (pg_dump succeeded)
        if [[ -s "$output_file" ]]; then
            local size
            size=$(stat -c%s "$output_file" 2>/dev/null || echo "0")
            local checksum
            checksum=$(calculate_checksum "$output_file")

            log_success "PostgreSQL backup complete ($(format_size "$size"))"
            COMPONENTS_BACKED_UP+=("postgres")
            COMPONENT_CHECKSUMS+=("postgres.sql.gz:$checksum")
        else
            log_error "PostgreSQL backup produced empty file"
            rm -f "$output_file"
            COMPONENTS_FAILED+=("postgres")
            return 1
        fi
    else
        log_error "PostgreSQL backup failed"
        rm -f "$output_file"
        COMPONENTS_FAILED+=("postgres")
        return 1
    fi
}

# ==============================================================================
# Backup: Redis Cache
# ==============================================================================
backup_redis() {
    log_step "Backing up Redis cache..."

    if [[ "$SKIP_REDIS" == true ]]; then
        log_warn "Skipping Redis backup (--skip-redis)"
        return 0
    fi

    local output_file="$BACKUP_DIR/redis-dump.rdb"

    if [[ "$DRY_RUN" == true ]]; then
        log_dry "Trigger BGSAVE and copy RDB snapshot from container '$CONTAINER_REDIS'"
        COMPONENTS_BACKED_UP+=("redis")
        return 0
    fi

    # Check if container is running
    if ! check_container_running "$CONTAINER_REDIS"; then
        log_warn "Redis container '$CONTAINER_REDIS' is not running"
        COMPONENTS_FAILED+=("redis")
        return 1
    fi

    log_info "Triggering Redis BGSAVE..."

    # Trigger background save
    if ! docker exec "$CONTAINER_REDIS" redis-cli BGSAVE &>/dev/null; then
        log_error "Failed to trigger Redis BGSAVE"
        COMPONENTS_FAILED+=("redis")
        return 1
    fi

    # Wait for BGSAVE to complete (up to 60 seconds)
    local wait_count=0
    local max_wait=60
    while [[ $wait_count -lt $max_wait ]]; do
        local bgsave_status
        bgsave_status=$(docker exec "$CONTAINER_REDIS" redis-cli LASTSAVE 2>/dev/null)

        # Check if BGSAVE is still in progress
        local bgsave_in_progress
        bgsave_in_progress=$(docker exec "$CONTAINER_REDIS" redis-cli INFO persistence 2>/dev/null | grep -c "rdb_bgsave_in_progress:1" || echo "0")

        if [[ "$bgsave_in_progress" == "0" ]]; then
            break
        fi

        sleep 1
        ((wait_count++))
    done

    if [[ $wait_count -ge $max_wait ]]; then
        log_warn "Redis BGSAVE timeout after ${max_wait}s, proceeding with copy anyway"
    fi

    log_info "Copying RDB snapshot..."

    # Copy the dump.rdb file from the container
    if docker cp "$CONTAINER_REDIS:/data/dump.rdb" "$output_file" 2>/dev/null; then
        local size
        size=$(stat -c%s "$output_file" 2>/dev/null || echo "0")
        local checksum
        checksum=$(calculate_checksum "$output_file")

        log_success "Redis backup complete ($(format_size "$size"))"
        COMPONENTS_BACKED_UP+=("redis")
        COMPONENT_CHECKSUMS+=("redis-dump.rdb:$checksum")
    else
        log_error "Failed to copy Redis RDB file"
        COMPONENTS_FAILED+=("redis")
        return 1
    fi
}

# ==============================================================================
# Backup: Configuration Files
# ==============================================================================
backup_configuration() {
    log_step "Backing up configuration files..."

    local config_dir="$BACKUP_DIR/config"

    if [[ "$DRY_RUN" == true ]]; then
        log_dry "Copy and redact configuration files to config/"
        COMPONENTS_BACKED_UP+=("configuration")
        return 0
    fi

    mkdir -p "$config_dir"

    # Backup tars-home.env (redacted)
    if [[ -f "$ENV_FILE" ]]; then
        log_info "Backing up environment file (secrets redacted)..."
        cat "$ENV_FILE" | redact_env_file > "$config_dir/tars-home.env.redacted"
        log_success "  tars-home.env backed up (redacted)"
    else
        log_warn "  Environment file not found: $ENV_FILE"
    fi

    # Backup docker-compose file
    if [[ -f "$COMPOSE_FILE" ]]; then
        log_info "Backing up Docker Compose file..."
        cp "$COMPOSE_FILE" "$config_dir/docker-compose.home.yml"
        log_success "  docker-compose.home.yml backed up"
    else
        log_warn "  Docker Compose file not found: $COMPOSE_FILE"
    fi

    # Backup tars-home.yml if exists
    if [[ -f "$SCRIPT_DIR/tars-home.yml" ]]; then
        log_info "Backing up TARS config file (secrets redacted)..."
        cat "$SCRIPT_DIR/tars-home.yml" | redact_secrets > "$config_dir/tars-home.yml.redacted"
        log_success "  tars-home.yml backed up (redacted)"
    fi

    # Create config inventory
    {
        echo "T.A.R.S. Configuration Backup Inventory"
        echo "========================================"
        echo "Backup Date: $(date -Iseconds)"
        echo "Version: $VERSION"
        echo ""
        echo "Files Included:"
        ls -la "$config_dir" 2>/dev/null | tail -n +4
        echo ""
        echo "NOTE: Sensitive values (passwords, secrets, tokens, JWTs)"
        echo "have been redacted from all configuration files."
    } > "$config_dir/INVENTORY.txt"

    local checksum
    checksum=$(cd "$config_dir" && find . -type f -exec sha256sum {} \; | sha256sum | awk '{print $1}')

    COMPONENTS_BACKED_UP+=("configuration")
    COMPONENT_CHECKSUMS+=("config/:$checksum")
    log_success "Configuration backup complete"
}

# ==============================================================================
# Backup: Ollama Models (Optional)
# ==============================================================================
backup_ollama() {
    log_step "Backing up Ollama models..."

    if [[ "$INCLUDE_MODELS" != true ]]; then
        log_info "Skipping Ollama models (use --include-models to include)"
        return 0
    fi

    local volume_name
    volume_name=$(get_volume_name "$VOLUME_OLLAMA")
    local output_file="$BACKUP_DIR/ollama-models.tar.gz"

    if [[ "$DRY_RUN" == true ]]; then
        log_dry "Backup Ollama volume '$volume_name' to ollama-models.tar.gz"
        log_dry "WARNING: This may be very large (10+ GB)"
        COMPONENTS_BACKED_UP+=("ollama")
        return 0
    fi

    # Check if volume exists
    if ! docker volume ls --format '{{.Name}}' | grep -qE "^${volume_name}$"; then
        log_warn "Ollama volume '$volume_name' not found"
        COMPONENTS_FAILED+=("ollama")
        return 1
    fi

    # Warn about size
    log_warn "Backing up Ollama models - this may take a while and produce a large file..."

    # Get approximate size first
    local approx_size
    approx_size=$(docker run --rm -v "${volume_name}:/data:ro" alpine:latest du -sh /data 2>/dev/null | awk '{print $1}')
    log_info "Approximate Ollama data size: ${approx_size:-unknown}"

    # Create the backup
    if docker run --rm \
        -v "${volume_name}:/data:ro" \
        -v "$BACKUP_DIR:/backup" \
        alpine:latest \
        tar -czf /backup/ollama-models.tar.gz -C /data . 2>/dev/null; then

        local size
        size=$(stat -c%s "$output_file" 2>/dev/null || echo "0")
        local checksum
        checksum=$(calculate_checksum "$output_file")

        log_success "Ollama backup complete ($(format_size "$size"))"
        COMPONENTS_BACKED_UP+=("ollama")
        COMPONENT_CHECKSUMS+=("ollama-models.tar.gz:$checksum")
    else
        log_error "Ollama backup failed"
        COMPONENTS_FAILED+=("ollama")
        return 1
    fi
}

# ==============================================================================
# Generate Manifest
# ==============================================================================
generate_manifest() {
    log_step "Generating backup manifest..."

    local manifest_file="$BACKUP_DIR/manifest.json"
    local hostname_val
    hostname_val=$(hostname 2>/dev/null || echo "unknown")

    if [[ "$DRY_RUN" == true ]]; then
        log_dry "Generate manifest.json with backup metadata"
        return 0
    fi

    # Build checksums JSON
    local checksums_json="{"
    local first=true
    for entry in "${COMPONENT_CHECKSUMS[@]}"; do
        local file="${entry%%:*}"
        local hash="${entry##*:}"
        if [[ "$first" == true ]]; then
            first=false
        else
            checksums_json+=","
        fi
        checksums_json+="\"$file\":\"$hash\""
    done
    checksums_json+="}"

    # Build components JSON array
    local components_json="["
    first=true
    for component in "${COMPONENTS_BACKED_UP[@]}"; do
        if [[ "$first" == true ]]; then
            first=false
        else
            components_json+=","
        fi
        components_json+="\"$component\""
    done
    components_json+="]"

    # Build docker volumes JSON array
    local volumes_json="["
    first=true
    for vol in "$VOLUME_CHROMADB" "$VOLUME_POSTGRES" "$VOLUME_REDIS"; do
        local vol_name
        vol_name=$(get_volume_name "$vol")
        if [[ "$first" == true ]]; then
            first=false
        else
            volumes_json+=","
        fi
        volumes_json+="\"$vol_name\""
    done
    if [[ "$INCLUDE_MODELS" == true ]]; then
        local ollama_vol
        ollama_vol=$(get_volume_name "$VOLUME_OLLAMA")
        volumes_json+=",\"$ollama_vol\""
    fi
    volumes_json+="]"

    # Create manifest JSON
    cat > "$manifest_file" << EOF
{
    "version": "$VERSION",
    "timestamp": "$(date -Iseconds)",
    "backup_name": "$BACKUP_NAME",
    "components": $components_json,
    "checksums": $checksums_json,
    "host": "$hostname_val",
    "docker_volumes": $volumes_json,
    "options": {
        "include_models": $INCLUDE_MODELS,
        "skip_postgres": $SKIP_POSTGRES,
        "skip_chromadb": $SKIP_CHROMADB,
        "skip_redis": $SKIP_REDIS
    },
    "failed_components": $(printf '%s\n' "${COMPONENTS_FAILED[@]}" | jq -R . | jq -s . 2>/dev/null || echo "[]")
}
EOF

    log_success "Manifest generated: manifest.json"
}

# ==============================================================================
# Create Final Archive
# ==============================================================================
create_archive() {
    log_step "Creating backup archive..."

    local archive_file="$OUTPUT_DIR/${BACKUP_NAME}.tar.gz"
    local checksum_file="$OUTPUT_DIR/${BACKUP_NAME}.tar.gz.sha256"

    if [[ "$DRY_RUN" == true ]]; then
        log_dry "Create archive: ${BACKUP_NAME}.tar.gz"
        log_dry "Generate checksum: ${BACKUP_NAME}.tar.gz.sha256"
        return 0
    fi

    # Create the tar.gz archive
    log_info "Compressing backup directory..."
    if tar -czf "$archive_file" -C "$OUTPUT_DIR" "$BACKUP_NAME"; then
        local size
        size=$(stat -c%s "$archive_file" 2>/dev/null || echo "0")
        log_success "Archive created: $(format_size "$size")"
    else
        log_error "Failed to create archive"
        return 1
    fi

    # Generate SHA-256 checksum
    log_info "Generating SHA-256 checksum..."
    if sha256sum "$archive_file" > "$checksum_file"; then
        log_success "Checksum file created"
    else
        log_error "Failed to generate checksum"
        return 1
    fi

    # Clean up uncompressed backup directory
    log_info "Cleaning up temporary files..."
    rm -rf "$BACKUP_DIR"

    # Final summary
    echo ""
    echo -e "${CYAN}============================================================${NC}"
    echo -e "${CYAN}   Backup Complete${NC}"
    echo -e "${CYAN}============================================================${NC}"
    echo ""
    echo -e "  Archive:   ${GREEN}$archive_file${NC}"
    echo -e "  Checksum:  $checksum_file"
    echo -e "  Size:      $(format_size "$(stat -c%s "$archive_file" 2>/dev/null || echo "0")")"
    echo ""
    echo "  Components backed up:"
    for component in "${COMPONENTS_BACKED_UP[@]}"; do
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
    echo "  Output directory: $OUTPUT_DIR"
    echo "  Backup name: $BACKUP_NAME"
    echo ""
    echo "  Components to backup:"

    if [[ "$SKIP_CHROMADB" != true ]]; then
        echo -e "    ${GREEN}+${NC} ChromaDB vector database (volume tar)"
    else
        echo -e "    ${YELLOW}-${NC} ChromaDB (skipped)"
    fi

    if [[ "$SKIP_POSTGRES" != true ]]; then
        echo -e "    ${GREEN}+${NC} PostgreSQL database (pg_dump)"
    else
        echo -e "    ${YELLOW}-${NC} PostgreSQL (skipped)"
    fi

    if [[ "$SKIP_REDIS" != true ]]; then
        echo -e "    ${GREEN}+${NC} Redis cache (BGSAVE + RDB copy)"
    else
        echo -e "    ${YELLOW}-${NC} Redis (skipped)"
    fi

    echo -e "    ${GREEN}+${NC} Configuration files (secrets redacted)"

    if [[ "$INCLUDE_MODELS" == true ]]; then
        echo -e "    ${YELLOW}+${NC} Ollama models (WARNING: may be large)"
    else
        echo -e "    ${BLUE}-${NC} Ollama models (not included)"
    fi

    echo ""
    echo "  Files to be created:"
    echo "    - ${BACKUP_NAME}.tar.gz"
    echo "    - ${BACKUP_NAME}.tar.gz.sha256"
    echo ""
    echo -e "${YELLOW}No files were created (dry run mode)${NC}"
    echo ""
}

# ==============================================================================
# Pre-flight Checks
# ==============================================================================
preflight_checks() {
    log_step "Running pre-flight checks..."

    # Check Docker availability
    check_docker

    # Check output directory
    if [[ ! -d "$OUTPUT_DIR" ]]; then
        if [[ "$DRY_RUN" == true ]]; then
            log_dry "Create output directory: $OUTPUT_DIR"
        else
            log_info "Creating output directory: $OUTPUT_DIR"
            mkdir -p "$OUTPUT_DIR"
        fi
    fi

    # Check disk space (need at least 5 GB free, 20 GB if including models)
    local required_space=5368709120  # 5 GB in bytes
    if [[ "$INCLUDE_MODELS" == true ]]; then
        required_space=21474836480  # 20 GB in bytes
    fi

    if [[ "$DRY_RUN" != true ]]; then
        local available_space
        available_space=$(df --output=avail -B1 "$OUTPUT_DIR" 2>/dev/null | tail -1)

        if [[ -n "$available_space" ]] && [[ "$available_space" -lt "$required_space" ]]; then
            log_warn "Low disk space: $(format_size "$available_space") available"
            log_warn "Recommended: $(format_size "$required_space") for backup"
        else
            log_success "Disk space: $(format_size "$available_space") available"
        fi
    fi

    # Check if compose file exists
    if [[ -f "$COMPOSE_FILE" ]]; then
        log_success "Docker Compose file found"
    else
        log_warn "Docker Compose file not found: $COMPOSE_FILE"
    fi

    # Check running containers
    local running_count=0
    for container in "$CONTAINER_OLLAMA" "$CONTAINER_CHROMADB" "$CONTAINER_REDIS" "$CONTAINER_POSTGRES" "$CONTAINER_BACKEND"; do
        if check_container_running "$container"; then
            ((running_count++))
        fi
    done

    if [[ $running_count -eq 0 ]]; then
        log_warn "No T.A.R.S. containers are running"
        log_warn "Some backups may fail if containers are not running"
    else
        log_success "$running_count T.A.R.S. containers running"
    fi

    log_success "Pre-flight checks complete"
}

# ==============================================================================
# Main
# ==============================================================================
main() {
    echo -e "${CYAN}===============================================================${NC}"
    echo -e "${CYAN}   T.A.R.S. Backup Script - v${VERSION} (Phase 25)              ${NC}"
    echo -e "${CYAN}===============================================================${NC}"
    echo ""

    # Parse command line arguments
    parse_args "$@"

    # Set up backup directory path
    BACKUP_DIR="$OUTPUT_DIR/$BACKUP_NAME"

    # Acquire lock (prevents concurrent backups)
    acquire_lock

    # Show configuration
    log_info "Backup configuration:"
    log_info "  Output directory: $OUTPUT_DIR"
    log_info "  Backup name: $BACKUP_NAME"
    log_info "  Include models: $INCLUDE_MODELS"
    if [[ "$DRY_RUN" == true ]]; then
        echo -e "  ${YELLOW}Mode: DRY RUN (no changes will be made)${NC}"
    fi
    echo ""

    # Run pre-flight checks
    preflight_checks

    # Create backup directory
    if [[ "$DRY_RUN" != true ]]; then
        mkdir -p "$BACKUP_DIR"
        log_info "Created backup directory: $BACKUP_DIR"
    fi

    # Perform backups
    backup_chromadb || true
    backup_postgres || true
    backup_redis || true
    backup_configuration || true
    backup_ollama || true

    # Generate manifest
    generate_manifest

    # Create final archive or show dry run summary
    if [[ "$DRY_RUN" == true ]]; then
        show_dry_run_summary
        exit 0
    else
        create_archive
    fi

    # Determine exit code based on results
    local exit_code=0
    if [[ ${#COMPONENTS_FAILED[@]} -gt 0 ]]; then
        if [[ ${#COMPONENTS_BACKED_UP[@]} -gt 0 ]]; then
            log_warn "Backup completed with some failures"
            exit_code=2
        else
            log_error "Backup failed - no components were backed up"
            exit_code=1
        fi
    else
        log_success "Backup completed successfully!"
        exit_code=0
    fi

    echo ""
    echo "Next steps:"
    echo "  1. Verify the backup archive exists and has expected size"
    echo "  2. Optionally copy to offsite storage (NAS, cloud, etc.)"
    echo "  3. Test restore periodically with restore-tars.sh"
    echo ""

    exit $exit_code
}

main "$@"
