#!/bin/bash
#
# Phase 14.6 - Package Verification Script
#
# Quick verification that all Phase 8 deliverables are in place
#
# Usage:
#   bash scripts/verify_phase14_6_package.sh

set -euo pipefail

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_header() {
    echo ""
    echo -e "${BLUE}============================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}============================================================${NC}"
    echo ""
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

check_file() {
    if [ -f "$1" ]; then
        print_success "$1 exists"
        return 0
    else
        print_error "$1 NOT FOUND"
        return 1
    fi
}

check_dir() {
    if [ -d "$1" ]; then
        print_success "$1/ exists"
        return 0
    else
        print_error "$1/ NOT FOUND"
        return 1
    fi
}

print_header "Phase 14.6 - Package Verification"

echo "Verifying all Phase 8 deliverables..."
echo ""

ERRORS=0

# A. Packaging
print_header "A. Packaging"
check_file "pyproject.toml" || ((ERRORS++))
check_dir "tars_observability" || ((ERRORS++))
check_file "tars_observability/__init__.py" || ((ERRORS++))
check_file "tars_observability/__version__.py" || ((ERRORS++))
check_file "tars_observability/py.typed" || ((ERRORS++))
check_dir "tars_observability/cli" || ((ERRORS++))
check_file "tars_observability/cli/ga_kpi.py" || ((ERRORS++))
check_file "tars_observability/cli/stability_monitor.py" || ((ERRORS++))
check_file "tars_observability/cli/anomaly_detector.py" || ((ERRORS++))
check_file "tars_observability/cli/health_reporter.py" || ((ERRORS++))
check_file "tars_observability/cli/regression_analyzer.py" || ((ERRORS++))
check_file "tars_observability/cli/retrospective.py" || ((ERRORS++))

# B. CI/CD
print_header "B. CI/CD Release Automation"
check_file ".github/workflows/release_phase14_6.yml" || ((ERRORS++))

# C. Dockerization
print_header "C. Dockerization"
check_file "Dockerfile" || ((ERRORS++))
check_file "docker-compose.yaml" || ((ERRORS++))

# D. Documentation
print_header "D. Documentation"
check_file "docs/PHASE14_6_PRODUCTION_RUNBOOK.md" || ((ERRORS++))
check_file "docs/PHASE14_6_DOCKER.md" || ((ERRORS++))
check_file "docs/PHASE14_6_QUICKSTART.md" || ((ERRORS++))

# E. Internal Adoption
print_header "E. Internal Adoption Toolkit"
check_dir "internal_adoption" || ((ERRORS++))
check_file "internal_adoption/README.md" || ((ERRORS++))
check_file "internal_adoption/onboard.sh" || ((ERRORS++))
check_file "internal_adoption/slack_integration.sh" || ((ERRORS++))
check_file "internal_adoption/github_issues_import.py" || ((ERRORS++))

# F. Changelog
print_header "F. Changelog & Release Notes"
check_file "CHANGELOG_PHASE14_6.md" || ((ERRORS++))
check_file "PHASE14_6_PHASE8_COMPLETION_SUMMARY.md" || ((ERRORS++))

# Summary
print_header "Summary"

if [ $ERRORS -eq 0 ]; then
    print_success "All deliverables verified! (0 errors)"
    echo ""
    echo "Phase 14.6 - Phase 8 is COMPLETE ✅"
    echo ""
    echo "Next steps:"
    echo "  1. Install package: pip install -e ."
    echo "  2. Run onboarding: bash internal_adoption/onboard.sh"
    echo "  3. Test Docker: docker build -t tars-observability:1.0.2-pre ."
    echo "  4. Review docs: cat PHASE14_6_PHASE8_COMPLETION_SUMMARY.md"
    exit 0
else
    print_error "Found $ERRORS missing deliverables"
    echo ""
    echo "Please ensure all Phase 8 tasks are complete."
    exit 1
fi
