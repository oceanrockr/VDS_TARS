#!/bin/bash
# ==============================================================================
# T.A.R.S. Build Harness - Deterministic Build Validation
# Implements RiPIT:build-harness task for reproducible builds
# ==============================================================================

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD_LOG_DIR="${PROJECT_ROOT}/build/logs"
BUILD_ARTIFACTS_DIR="${PROJECT_ROOT}/build/artifacts"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${BUILD_LOG_DIR}/build_harness_${TIMESTAMP}.log"
REPORT_FILE="${BUILD_LOG_DIR}/harness_report.json"

# Create directories
mkdir -p "${BUILD_LOG_DIR}" "${BUILD_ARTIFACTS_DIR}"

# Logging functions
log() {
    echo -e "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "${LOG_FILE}"
}

log_success() {
    echo -e "${GREEN}✓${NC} $1" | tee -a "${LOG_FILE}"
}

log_error() {
    echo -e "${RED}✗${NC} $1" | tee -a "${LOG_FILE}"
}

log_warning() {
    echo -e "${YELLOW}⚠${NC} $1" | tee -a "${LOG_FILE}"
}

log_info() {
    echo -e "${BLUE}ℹ${NC} $1" | tee -a "${LOG_FILE}"
}

# Environment validation
validate_environment() {
    log_info "Phase 1: Validating build environment"
    
    if command -v docker &> /dev/null; then
        DOCKER_VERSION=$(docker --version)
        log_success "Docker installed: ${DOCKER_VERSION}"
    else
        log_error "Docker not found"
        return 1
    fi
    
    if command -v docker-compose &> /dev/null; then
        COMPOSE_VERSION=$(docker-compose --version)
        log_success "Docker Compose installed: ${COMPOSE_VERSION}"
    else
        log_error "Docker Compose not found"
        return 1
    fi
    
    if command -v nvidia-smi &> /dev/null; then
        GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo "Unable to query GPU")
        log_success "NVIDIA GPU detected: ${GPU_INFO}"
    else
        log_warning "NVIDIA GPU not detected - CPU fallback will be used"
    fi
    
    return 0
}

# Dependency validation
validate_dependencies() {
    log_info "Phase 2: Validating dependencies"
    
    if [ -f "${PROJECT_ROOT}/.env" ]; then
        log_success ".env file exists"
    else
        log_warning ".env file not found - using defaults"
    fi
    
    if [ -f "${PROJECT_ROOT}/backend/requirements.txt" ]; then
        log_success "Backend requirements.txt exists"
    else
        log_error "Backend requirements.txt not found"
        return 1
    fi
    
    if [ -f "${PROJECT_ROOT}/docker-compose.yml" ]; then
        log_success "docker-compose.yml exists"
    else
        log_error "docker-compose.yml not found"
        return 1
    fi
    
    return 0
}

# Main execution
main() {
    log_info "T.A.R.S. Build Harness - Phase 1 Validation"
    echo ""
    
    validate_environment || exit 1
    echo ""
    validate_dependencies || exit 1
    echo ""
    
    log_success "Build harness validation complete"
    
    cat > "${REPORT_FILE}" << 'JSON_EOF'
{
  "timestamp": "$(date -Iseconds)",
  "project": "T.A.R.S.",
  "phase": "Phase 1",
  "validation": {
    "environment": true,
    "dependencies": true
  },
  "success": true
}
JSON_EOF
    
    log_success "Build report generated: ${REPORT_FILE}"
    
    exit 0
}

main "$@"
