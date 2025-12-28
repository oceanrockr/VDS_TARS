#!/bin/bash
# ==============================================================================
# T.A.R.S. Support Bundle Generator
# Version: v1.0.11 (GA) - Phase 23 Field Debugging Support
# Target: Ubuntu 22.04 LTS, Home Network Deployment
# ==============================================================================
#
# Generates a support bundle for troubleshooting T.A.R.S. issues without
# leaking sensitive information like passwords, tokens, or API keys.
#
# Bundle Contents:
#   - Docker container status and health
#   - Container logs (redacted)
#   - Environment variable presence (not values)
#   - Configuration file (secrets redacted)
#   - System info (CPU, RAM, GPU, disk)
#   - Validation script outputs
#   - Mount status
#
# Security:
#   - NEVER exports raw secrets
#   - Redacts passwords, tokens, keys aggressively
#   - Safe to share with support
#
# Usage:
#   ./generate-support-bundle.sh                    # Generate bundle
#   ./generate-support-bundle.sh --output-dir /tmp  # Custom output location
#   ./generate-support-bundle.sh --include-all-logs # Include extended logs
#
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
COMPOSE_FILE="$SCRIPT_DIR/docker-compose.home.yml"
ENV_FILE="$SCRIPT_DIR/tars-home.env"

# Default output location
OUTPUT_DIR="$PROJECT_ROOT/support"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BUNDLE_NAME="tars-support-bundle-$TIMESTAMP"
BUNDLE_DIR="$OUTPUT_DIR/$BUNDLE_NAME"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# CLI Flags
INCLUDE_ALL_LOGS=false
LOG_LINES=200

# Logging
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_step() { echo -e "\n${CYAN}>>> $1${NC}"; }

# ==============================================================================
# Parse CLI Arguments
# ==============================================================================
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --output-dir)
                OUTPUT_DIR="$2"
                BUNDLE_DIR="$OUTPUT_DIR/$BUNDLE_NAME"
                shift 2
                ;;
            --include-all-logs)
                INCLUDE_ALL_LOGS=true
                LOG_LINES=1000
                shift
                ;;
            --help|-h)
                show_help
                exit 0
                ;;
            *)
                echo "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

show_help() {
    echo "T.A.R.S. Support Bundle Generator"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --output-dir <path>    Output directory (default: ./support)"
    echo "  --include-all-logs     Include extended logs (1000 lines vs 200)"
    echo "  --help, -h             Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                           # Generate bundle in ./support"
    echo "  $0 --output-dir /tmp         # Generate in /tmp"
    echo "  $0 --include-all-logs        # Include more log history"
}

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
            if [[ "$var_name" =~ PASSWORD|SECRET|KEY|TOKEN|CREDENTIAL ]]; then
                echo "${var_name}=<REDACTED>"
            elif [[ "$var_name" =~ ^(NAS_|POSTGRES_|REDIS_|JWT_|API_) ]]; then
                # Show presence but not value
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
# Collect: System Information
# ==============================================================================
collect_system_info() {
    log_step "Collecting system information..."

    local output="$BUNDLE_DIR/system-info.txt"

    {
        echo "T.A.R.S. Support Bundle - System Information"
        echo "Generated: $(date -Iseconds)"
        echo "=============================================="
        echo ""

        echo "=== Host Information ==="
        echo "Hostname: $(hostname)"
        echo "OS: $(cat /etc/os-release 2>/dev/null | grep PRETTY_NAME | cut -d'"' -f2 || echo 'Unknown')"
        echo "Kernel: $(uname -r)"
        echo "Architecture: $(uname -m)"
        echo ""

        echo "=== CPU Information ==="
        grep -E "model name|cpu cores" /proc/cpuinfo 2>/dev/null | head -4 || echo "Could not read CPU info"
        echo ""

        echo "=== Memory Information ==="
        free -h 2>/dev/null || echo "Could not read memory info"
        echo ""

        echo "=== Disk Usage ==="
        df -h 2>/dev/null | grep -E "^/dev|Filesystem" || echo "Could not read disk info"
        echo ""

        echo "=== Docker Disk Usage ==="
        docker system df 2>/dev/null || echo "Docker not running"
        echo ""

        echo "=== GPU Information ==="
        if command -v nvidia-smi &>/dev/null; then
            nvidia-smi --query-gpu=name,memory.total,memory.free,utilization.gpu,temperature.gpu \
                --format=csv,noheader 2>/dev/null || echo "GPU query failed"
        else
            echo "NVIDIA GPU not detected (nvidia-smi not available)"
        fi
        echo ""

        echo "=== Docker Version ==="
        docker version 2>/dev/null || echo "Docker not running"
        echo ""

        echo "=== Docker Compose Version ==="
        docker compose version 2>/dev/null || echo "Docker Compose not available"

    } > "$output"

    log_success "System info collected"
}

# ==============================================================================
# Collect: Container Status
# ==============================================================================
collect_container_status() {
    log_step "Collecting container status..."

    local output="$BUNDLE_DIR/container-status.txt"

    {
        echo "T.A.R.S. Container Status"
        echo "Generated: $(date -Iseconds)"
        echo "========================="
        echo ""

        echo "=== Docker PS (All T.A.R.S. containers) ==="
        docker ps -a --filter "name=tars-home" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" 2>/dev/null || echo "Could not list containers"
        echo ""

        echo "=== Container Health Status ==="
        for container in tars-home-ollama tars-home-chromadb tars-home-redis tars-home-postgres tars-home-backend; do
            if docker ps -a --format '{{.Names}}' 2>/dev/null | grep -q "^${container}$"; then
                status=$(docker inspect "$container" --format '{{.State.Status}}' 2>/dev/null || echo "unknown")
                health=$(docker inspect "$container" --format '{{.State.Health.Status}}' 2>/dev/null || echo "no-healthcheck")
                restarts=$(docker inspect "$container" --format '{{.RestartCount}}' 2>/dev/null || echo "unknown")
                echo "$container: Status=$status, Health=$health, Restarts=$restarts"
            else
                echo "$container: NOT FOUND"
            fi
        done
        echo ""

        echo "=== Container Resource Usage ==="
        docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}" 2>/dev/null | grep -E "^NAME|tars-home" || echo "Could not get stats"
        echo ""

        echo "=== Docker Compose Status ==="
        if [[ -f "$COMPOSE_FILE" ]]; then
            docker compose -f "$COMPOSE_FILE" ps 2>/dev/null || echo "Could not get compose status"
        else
            echo "Compose file not found: $COMPOSE_FILE"
        fi

    } > "$output"

    log_success "Container status collected"
}

# ==============================================================================
# Collect: Container Logs (Redacted)
# ==============================================================================
collect_container_logs() {
    log_step "Collecting container logs (redacted)..."

    mkdir -p "$BUNDLE_DIR/logs"

    local containers=(
        "tars-home-backend"
        "tars-home-ollama"
        "tars-home-chromadb"
        "tars-home-redis"
        "tars-home-postgres"
    )

    for container in "${containers[@]}"; do
        local log_file="$BUNDLE_DIR/logs/${container}.log"

        if docker ps -a --format '{{.Names}}' 2>/dev/null | grep -q "^${container}$"; then
            {
                echo "=== $container logs (last $LOG_LINES lines) ==="
                echo "Captured: $(date -Iseconds)"
                echo ""
                docker logs "$container" --tail "$LOG_LINES" 2>&1 | redact_secrets
            } > "$log_file"
            log_success "  $container logs collected"
        else
            echo "$container: Container not found" > "$log_file"
            log_warn "  $container not found"
        fi
    done
}

# ==============================================================================
# Collect: Configuration (Redacted)
# ==============================================================================
collect_configuration() {
    log_step "Collecting configuration (secrets redacted)..."

    mkdir -p "$BUNDLE_DIR/config"

    # Environment file (redacted)
    if [[ -f "$ENV_FILE" ]]; then
        cat "$ENV_FILE" | redact_env_file > "$BUNDLE_DIR/config/tars-home.env.redacted"
        log_success "  Environment file collected (redacted)"
    else
        echo "Environment file not found: $ENV_FILE" > "$BUNDLE_DIR/config/tars-home.env.redacted"
        log_warn "  Environment file not found"
    fi

    # Docker Compose file (no secrets, safe to include)
    if [[ -f "$COMPOSE_FILE" ]]; then
        cp "$COMPOSE_FILE" "$BUNDLE_DIR/config/"
        log_success "  Docker Compose file collected"
    fi

    # tars-home.yml if exists
    if [[ -f "$SCRIPT_DIR/tars-home.yml" ]]; then
        cat "$SCRIPT_DIR/tars-home.yml" | redact_secrets > "$BUNDLE_DIR/config/tars-home.yml.redacted"
        log_success "  Configuration file collected (redacted)"
    fi

    # List what env vars are set (names only)
    {
        echo "Environment Variables Set (names only, values redacted):"
        echo "=========================================================="
        if [[ -f "$ENV_FILE" ]]; then
            grep -E "^[A-Za-z_][A-Za-z0-9_]*=" "$ENV_FILE" 2>/dev/null | cut -d'=' -f1 | sort
        fi
    } > "$BUNDLE_DIR/config/env-vars-present.txt"
}

# ==============================================================================
# Collect: API Health
# ==============================================================================
collect_api_health() {
    log_step "Collecting API health status..."

    local output="$BUNDLE_DIR/api-health.txt"

    {
        echo "T.A.R.S. API Health Status"
        echo "Generated: $(date -Iseconds)"
        echo "=========================="
        echo ""

        echo "=== /health endpoint ==="
        curl -s --max-time 10 http://localhost:8000/health 2>/dev/null | jq . 2>/dev/null || echo "Could not reach /health"
        echo ""

        echo "=== /ready endpoint ==="
        curl -s --max-time 10 http://localhost:8000/ready 2>/dev/null | jq . 2>/dev/null || echo "Could not reach /ready"
        echo ""

        echo "=== /metrics endpoint ==="
        curl -s --max-time 10 http://localhost:8000/metrics 2>/dev/null | jq . 2>/dev/null || echo "Could not reach /metrics"
        echo ""

        echo "=== RAG Stats ==="
        curl -s --max-time 10 http://localhost:8000/rag/stats 2>/dev/null | jq . 2>/dev/null || echo "Could not reach /rag/stats"
        echo ""

        echo "=== Ollama API ==="
        curl -s --max-time 10 http://localhost:11434/api/tags 2>/dev/null | jq '.models[].name' 2>/dev/null || echo "Could not reach Ollama API"
        echo ""

        echo "=== ChromaDB Heartbeat ==="
        curl -s --max-time 10 http://localhost:8001/api/v1/heartbeat 2>/dev/null | jq . 2>/dev/null || echo "Could not reach ChromaDB"

    } > "$output"

    log_success "API health collected"
}

# ==============================================================================
# Collect: Validation Results
# ==============================================================================
collect_validation_results() {
    log_step "Running validation scripts..."

    mkdir -p "$BUNDLE_DIR/validation"

    # Deployment validation
    if [[ -x "$SCRIPT_DIR/validate-deployment.sh" ]]; then
        {
            echo "=== Deployment Validation ==="
            echo "Captured: $(date -Iseconds)"
            echo ""
            "$SCRIPT_DIR/validate-deployment.sh" 2>&1 || true
        } > "$BUNDLE_DIR/validation/deployment.txt"
        log_success "  Deployment validation collected"
    fi

    # Security validation
    if [[ -x "$SCRIPT_DIR/validate-security.sh" ]]; then
        {
            echo "=== Security Validation ==="
            echo "Captured: $(date -Iseconds)"
            echo ""
            "$SCRIPT_DIR/validate-security.sh" 2>&1 | redact_secrets || true
        } > "$BUNDLE_DIR/validation/security.txt"
        log_success "  Security validation collected"
    fi

    # RAG validation
    if [[ -x "$SCRIPT_DIR/validate-rag.sh" ]]; then
        {
            echo "=== RAG Validation ==="
            echo "Captured: $(date -Iseconds)"
            echo ""
            "$SCRIPT_DIR/validate-rag.sh" 2>&1 | redact_secrets || true
        } > "$BUNDLE_DIR/validation/rag.txt"
        log_success "  RAG validation collected"
    fi

    # Config doctor
    if [[ -x "$SCRIPT_DIR/config-doctor.sh" ]]; then
        {
            echo "=== Configuration Doctor ==="
            echo "Captured: $(date -Iseconds)"
            echo ""
            "$SCRIPT_DIR/config-doctor.sh" 2>&1 | redact_secrets || true
        } > "$BUNDLE_DIR/validation/config-doctor.txt"
        log_success "  Config doctor results collected"
    fi
}

# ==============================================================================
# Collect: Mount Status
# ==============================================================================
collect_mount_status() {
    log_step "Collecting mount status..."

    local output="$BUNDLE_DIR/mount-status.txt"

    {
        echo "T.A.R.S. Mount Status"
        echo "Generated: $(date -Iseconds)"
        echo "====================="
        echo ""

        echo "=== Mount Points ==="
        mount | grep -E "(nfs|cifs|smb|llm_docs|nas)" 2>/dev/null || echo "No NAS mounts detected"
        echo ""

        echo "=== /etc/fstab entries (redacted) ==="
        grep -v "^#" /etc/fstab 2>/dev/null | grep -E "(nfs|cifs|smb)" | redact_secrets || echo "No NAS entries in fstab"
        echo ""

        echo "=== NAS Mount Point ==="
        NAS_MOUNT="${NAS_MOUNT_POINT:-/mnt/llm_docs}"
        if [[ -d "$NAS_MOUNT" ]]; then
            echo "Mount point exists: $NAS_MOUNT"
            if mountpoint -q "$NAS_MOUNT" 2>/dev/null; then
                echo "Status: Mounted"
                echo "Space: $(df -h "$NAS_MOUNT" 2>/dev/null | tail -1 | awk '{print $3"/"$2" used"}')"
                echo "Document count: $(find "$NAS_MOUNT" -type f \( -name "*.pdf" -o -name "*.docx" -o -name "*.txt" -o -name "*.md" \) 2>/dev/null | wc -l)"
            else
                echo "Status: Not mounted"
            fi
        else
            echo "Mount point does not exist: $NAS_MOUNT"
        fi

    } > "$output"

    log_success "Mount status collected"
}

# ==============================================================================
# Collect: Network Status
# ==============================================================================
collect_network_status() {
    log_step "Collecting network status..."

    local output="$BUNDLE_DIR/network-status.txt"

    {
        echo "T.A.R.S. Network Status"
        echo "Generated: $(date -Iseconds)"
        echo "======================="
        echo ""

        echo "=== Listening Ports ==="
        ss -tlnp 2>/dev/null | grep -E "8000|8001|11434|6379|5432|3000" || echo "Could not get port info"
        echo ""

        echo "=== Docker Networks ==="
        docker network ls 2>/dev/null | grep -E "^NETWORK|tars" || echo "Could not list networks"
        echo ""

        echo "=== Container Network Details ==="
        for container in tars-home-backend tars-home-ollama tars-home-chromadb; do
            if docker ps --format '{{.Names}}' 2>/dev/null | grep -q "^${container}$"; then
                echo "--- $container ---"
                docker inspect "$container" --format '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' 2>/dev/null || echo "N/A"
            fi
        done
        echo ""

        echo "=== Firewall Status ==="
        if command -v ufw &>/dev/null; then
            sudo ufw status 2>/dev/null || echo "Could not get UFW status"
        elif command -v firewall-cmd &>/dev/null; then
            sudo firewall-cmd --list-all 2>/dev/null || echo "Could not get firewalld status"
        else
            echo "No firewall detected"
        fi

    } > "$output"

    log_success "Network status collected"
}

# ==============================================================================
# Create Bundle Archive
# ==============================================================================
create_archive() {
    log_step "Creating bundle archive..."

    # Create manifest
    {
        echo "T.A.R.S. Support Bundle Manifest"
        echo "================================"
        echo "Bundle ID: $BUNDLE_NAME"
        echo "Generated: $(date -Iseconds)"
        echo "Generator: generate-support-bundle.sh v1.0.11"
        echo ""
        echo "Contents:"
        find "$BUNDLE_DIR" -type f -printf "  %P\n" | sort
        echo ""
        echo "SECURITY NOTE:"
        echo "This bundle has been redacted to remove sensitive information."
        echo "All passwords, tokens, and secrets have been replaced with <REDACTED>."
        echo "This bundle is safe to share with support personnel."
    } > "$BUNDLE_DIR/MANIFEST.txt"

    # Create tar.gz
    local archive="$OUTPUT_DIR/${BUNDLE_NAME}.tar.gz"
    tar -czf "$archive" -C "$OUTPUT_DIR" "$BUNDLE_NAME"

    # Calculate checksum
    sha256sum "$archive" > "${archive}.sha256"

    # Cleanup uncompressed directory
    rm -rf "$BUNDLE_DIR"

    log_success "Bundle archive created: $archive"
    echo ""
    echo -e "${CYAN}Support bundle generated:${NC}"
    echo "  Archive: $archive"
    echo "  Checksum: ${archive}.sha256"
    echo "  Size: $(du -h "$archive" | cut -f1)"
}

# ==============================================================================
# Main
# ==============================================================================
main() {
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}   T.A.R.S. Support Bundle Generator - v1.0.11 (Phase 23)      ${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
    echo ""

    parse_args "$@"

    # Create bundle directory
    mkdir -p "$BUNDLE_DIR"
    log_info "Bundle directory: $BUNDLE_DIR"

    # Collect all information
    collect_system_info
    collect_container_status
    collect_container_logs
    collect_configuration
    collect_api_health
    collect_validation_results
    collect_mount_status
    collect_network_status

    # Create archive
    create_archive

    echo ""
    echo -e "${GREEN}Support bundle generation complete!${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Share the .tar.gz file with support"
    echo "  2. Include the .sha256 file for integrity verification"
    echo "  3. Describe your issue when submitting"
    echo ""
    echo -e "${YELLOW}SECURITY NOTE:${NC} This bundle has been redacted to remove sensitive"
    echo "information. It is safe to share with support personnel."
}

main "$@"
