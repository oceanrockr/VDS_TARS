#!/bin/bash
##############################################################################
# T.A.R.S. Chaos Testing - Test Runner
#
# Orchestrates all chaos tests in sequence.
# Usage: ./run-tests.sh [--skip-k6] [--skip-resilience]
##############################################################################

set -e

# Configuration
RESULTS_DIR="./results/$(date +%Y%m%d_%H%M%S)"
SKIP_K6=false
SKIP_RESILIENCE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-k6)
            SKIP_K6=true
            shift
            ;;
        --skip-resilience)
            SKIP_RESILIENCE=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Create results directory
mkdir -p "$RESULTS_DIR"

echo "============================================================"
echo "T.A.R.S. Chaos Testing Suite"
echo "============================================================"
echo "Results directory: $RESULTS_DIR"
echo ""

# Function to run k6 test
run_k6_test() {
    local test_name=$1
    local script_path=$2

    echo ""
    echo "============================================================"
    echo "Running k6 Test: $test_name"
    echo "============================================================"

    if command -v k6 &> /dev/null; then
        k6 run "$script_path" \
            --out json="$RESULTS_DIR/${test_name}.json" \
            --summary-export="$RESULTS_DIR/${test_name}-summary.json" \
            | tee "$RESULTS_DIR/${test_name}.log"

        echo "✅ $test_name completed"
    else
        echo "❌ k6 not installed. Skipping $test_name"
        echo "Install: https://k6.io/docs/getting-started/installation/"
    fi
}

# Function to run Python resilience test
run_resilience_test() {
    local test_name=$1
    local script_path=$2

    echo ""
    echo "============================================================"
    echo "Running Resilience Test: $test_name"
    echo "============================================================"

    python3 "$script_path" 2>&1 | tee "$RESULTS_DIR/${test_name}.log"

    echo "✅ $test_name completed"
}

# k6 Load Tests
if [ "$SKIP_K6" = false ]; then
    echo ""
    echo "========================================"
    echo "PHASE 1: k6 Load Tests"
    echo "========================================"

    run_k6_test "sustained-load" "./k6/sustained-load.js"

    echo ""
    echo "Cooling down for 60s..."
    sleep 60

    run_k6_test "spike-load" "./k6/spike-load.js"

    echo ""
    echo "Cooling down for 60s..."
    sleep 60

    run_k6_test "jwt-stress" "./k6/jwt-stress.js"

    echo ""
    echo "✅ k6 tests completed"
else
    echo "⏭️  Skipping k6 tests"
fi

# Resilience Tests
if [ "$SKIP_RESILIENCE" = false ]; then
    echo ""
    echo "========================================"
    echo "PHASE 2: Resilience Tests"
    echo "========================================"

    echo ""
    echo "⚠️  WARNING: Resilience tests will disrupt services!"
    echo "Ensure you are running in a non-production environment."
    read -p "Continue? (y/N): " confirm

    if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
        echo "Resilience tests skipped by user"
    else
        # Check if kubectl is available
        if ! command -v kubectl &> /dev/null; then
            echo "❌ kubectl not installed. Skipping resilience tests"
        else
            run_resilience_test "redis-outage" "./resilience/redis-outage.py"

            echo ""
            echo "Cooling down for 120s..."
            sleep 120

            run_resilience_test "pod-kill-test" "./resilience/pod-kill-test.py"

            echo ""
            echo "✅ Resilience tests completed"
        fi
    fi
else
    echo "⏭️  Skipping resilience tests"
fi

# Generate final report
echo ""
echo "============================================================"
echo "Test Summary"
echo "============================================================"
echo "Results saved to: $RESULTS_DIR"
echo ""

if [ -d "$RESULTS_DIR" ]; then
    echo "Files generated:"
    ls -lh "$RESULTS_DIR"
fi

echo ""
echo "============================================================"
echo "✅ Chaos Testing Complete"
echo "============================================================"
echo ""
echo "Next steps:"
echo "1. Review test results in $RESULTS_DIR"
echo "2. Check for any failures or degraded performance"
echo "3. Investigate and fix any issues found"
echo "4. Re-run tests to verify fixes"
echo ""
