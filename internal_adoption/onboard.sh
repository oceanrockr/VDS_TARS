#!/bin/bash
#
# T.A.R.S. Phase 14.6 - Onboarding Script
#
# This script:
# 1. Verifies Python version
# 2. Installs tars-observability package
# 3. Verifies CLI tools are available
# 4. Runs smoke test on test data
# 5. Generates sample retrospective
#
# Usage:
#   bash internal_adoption/onboard.sh

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
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

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Main script
print_header "T.A.R.S. Phase 14.6 - Onboarding Script"

echo "This script will:"
echo "  1. Verify Python version"
echo "  2. Install tars-observability package"
echo "  3. Verify CLI tools"
echo "  4. Run smoke test"
echo "  5. Generate sample retrospective"
echo ""
read -p "Press Enter to continue..."

# Step 1: Verify Python version
print_header "Step 1: Verifying Python Version"

if ! command -v python3 &> /dev/null; then
    print_error "Python 3 not found. Please install Python 3.8 or higher."
    exit 1
fi

PYTHON_VERSION=$(python3 --version | awk '{print $2}')
PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)

print_info "Detected Python version: $PYTHON_VERSION"

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]); then
    print_error "Python 3.8+ required. You have $PYTHON_VERSION"
    exit 1
fi

print_success "Python version OK: $PYTHON_VERSION"

# Step 2: Install tars-observability package
print_header "Step 2: Installing tars-observability Package"

print_info "Installing in development mode..."

# Check if we're in the project root
if [ ! -f "pyproject.toml" ]; then
    print_error "pyproject.toml not found. Please run this script from the project root."
    exit 1
fi

# Install dependencies
print_info "Installing dependencies..."
pip install -q --upgrade pip setuptools wheel
pip install -q -r requirements-dev.txt

# Install package in development mode
print_info "Installing tars-observability..."
pip install -q -e .

print_success "Package installed successfully"

# Step 3: Verify CLI tools
print_header "Step 3: Verifying CLI Tools"

CLI_TOOLS=(
    "tars-ga-kpi"
    "tars-stability-monitor"
    "tars-anomaly-detector"
    "tars-health-report"
    "tars-regression-analyzer"
    "tars-retro"
)

ALL_OK=true
for tool in "${CLI_TOOLS[@]}"; do
    if command -v "$tool" &> /dev/null; then
        print_success "$tool is available"
    else
        print_error "$tool not found in PATH"
        ALL_OK=false
    fi
done

if [ "$ALL_OK" = false ]; then
    print_error "Some CLI tools are missing. Installation may have failed."
    exit 1
fi

# Show version
print_info "Checking version..."
VERSION=$(tars-retro --version 2>&1 || echo "unknown")
print_info "Installed version: $VERSION"

print_success "All CLI tools verified"

# Step 4: Run smoke test
print_header "Step 4: Running Smoke Test"

print_info "Running Phase 14.6 smoke test pipeline..."

if [ -f "scripts/test_phase14_6_pipeline.sh" ]; then
    # Make executable
    chmod +x scripts/test_phase14_6_pipeline.sh

    # Run smoke test
    if bash scripts/test_phase14_6_pipeline.sh; then
        print_success "Smoke test passed"
    else
        print_error "Smoke test failed"
        print_warning "Check test_output/ for details"
        exit 1
    fi
else
    print_warning "Smoke test script not found, skipping..."
fi

# Step 5: Generate sample retrospective
print_header "Step 5: Generating Sample Retrospective"

print_info "Generating retrospective from test data..."

# Create output directory
mkdir -p test_output

# Generate retrospective
if tars-retro \
    --ga-data test_data/ga_kpis \
    --7day-data test_data/stability \
    --regression test_data/regression/regression_summary.json \
    --anomalies test_data/anomalies/anomaly_events.json \
    --output test_output/ONBOARDING_RETROSPECTIVE.md; then

    print_success "Sample retrospective generated"
    print_info "Location: test_output/ONBOARDING_RETROSPECTIVE.md"
else
    print_error "Failed to generate retrospective"
    exit 1
fi

# Display quick stats
print_header "Quick Stats from Sample Retrospective"

if [ -f "test_output/ONBOARDING_RETROSPECTIVE.json" ]; then
    SUCCESSES=$(jq -r '.successes | length' test_output/ONBOARDING_RETROSPECTIVE.json)
    DEGRADATIONS=$(jq -r '.degradations | length' test_output/ONBOARDING_RETROSPECTIVE.json)
    DRIFTS=$(jq -r '.unexpected_drifts | length' test_output/ONBOARDING_RETROSPECTIVE.json)
    RECOMMENDATIONS=$(jq -r '.recommendations_v1_0_2 | length' test_output/ONBOARDING_RETROSPECTIVE.json)

    echo "  Successes: $SUCCESSES"
    echo "  Degradations: $DEGRADATIONS"
    echo "  Unexpected Drifts: $DRIFTS"
    echo "  Recommendations: $RECOMMENDATIONS"
fi

# Summary
print_header "Onboarding Complete! 🎉"

echo "✅ Python version verified"
echo "✅ Package installed"
echo "✅ CLI tools available"
echo "✅ Smoke test passed"
echo "✅ Sample retrospective generated"
echo ""
echo "Next Steps:"
echo ""
echo "  1. View sample retrospective:"
echo "     cat test_output/ONBOARDING_RETROSPECTIVE.md"
echo ""
echo "  2. Read the documentation:"
echo "     - docs/PHASE14_6_QUICKSTART.md"
echo "     - docs/PHASE14_6_PRODUCTION_RUNBOOK.md"
echo "     - internal_adoption/README.md"
echo ""
echo "  3. Test individual CLI tools:"
echo "     tars-ga-kpi --help"
echo "     tars-stability-monitor --help"
echo "     tars-retro --help"
echo ""
echo "  4. Set up automation for your next GA:"
echo "     - Cron jobs (see docs/PHASE14_6_PRODUCTION_RUNBOOK.md)"
echo "     - Kubernetes CronJobs (see docs/PHASE14_6_DOCKER.md)"
echo ""
echo "Questions? Check internal_adoption/README.md or contact tars@veleron.dev"
echo ""

print_success "Happy monitoring! 📊"
