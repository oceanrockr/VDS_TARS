#!/bin/bash
# Phase 14.6 - End-to-End Smoke Test Script
# Tests the complete retrospective generation pipeline on test data

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TEST_DATA_DIR="$PROJECT_ROOT/test_data"
TEST_OUTPUT_DIR="$PROJECT_ROOT/test_output"

echo "============================================================"
echo "Phase 14.6 - End-to-End Pipeline Smoke Test"
echo "============================================================"
echo ""

# Cleanup previous test output
if [ -d "$TEST_OUTPUT_DIR" ]; then
    echo "Cleaning up previous test output..."
    rm -rf "$TEST_OUTPUT_DIR"
fi

mkdir -p "$TEST_OUTPUT_DIR"

# Step 1: Verify test data exists
echo "Step 1: Verifying test data..."
if [ ! -f "$TEST_DATA_DIR/ga_kpis/ga_kpi_summary.json" ]; then
    echo "ERROR: GA KPI summary not found at $TEST_DATA_DIR/ga_kpis/ga_kpi_summary.json"
    exit 1
fi

DAY_COUNT=0
for day in {1..7}; do
    if [ -f "$TEST_DATA_DIR/stability/day_$(printf '%02d' $day)_summary.json" ]; then
        ((DAY_COUNT++))
    fi
done

if [ $DAY_COUNT -ne 7 ]; then
    echo "ERROR: Expected 7 daily summaries, found $DAY_COUNT"
    exit 1
fi

echo "  ✓ GA KPI summary found"
echo "  ✓ All 7 daily summaries found"

if [ -f "$TEST_DATA_DIR/regression/regression_summary.json" ]; then
    echo "  ✓ Regression summary found"
fi

if [ -f "$TEST_DATA_DIR/anomalies/anomaly_events.json" ]; then
    echo "  ✓ Anomaly events found"
fi

echo ""

# Step 2: Run retrospective generator
echo "Step 2: Running retrospective generator..."
python "$PROJECT_ROOT/scripts/generate_retrospective.py" \
    --ga-data "$TEST_DATA_DIR/ga_kpis" \
    --7day-data "$TEST_DATA_DIR/stability" \
    --regression "$TEST_DATA_DIR/regression/regression_summary.json" \
    --anomalies "$TEST_DATA_DIR/anomalies/anomaly_events.json" \
    --output "$TEST_OUTPUT_DIR/GA_7DAY_RETROSPECTIVE.md"

echo ""

# Step 3: Verify outputs
echo "Step 3: Verifying outputs..."

if [ ! -f "$TEST_OUTPUT_DIR/GA_7DAY_RETROSPECTIVE.md" ]; then
    echo "ERROR: Markdown report not generated"
    exit 1
fi
echo "  ✓ Markdown report generated"

if [ ! -f "$TEST_OUTPUT_DIR/GA_7DAY_RETROSPECTIVE.json" ]; then
    echo "ERROR: JSON report not generated"
    exit 1
fi
echo "  ✓ JSON report generated"

# Validate JSON structure
python -c "
import json
import sys

with open('$TEST_OUTPUT_DIR/GA_7DAY_RETROSPECTIVE.json') as f:
    data = json.load(f)

required_keys = [
    'generation_timestamp',
    'ga_day_timestamp',
    'seven_day_end_timestamp',
    'successes',
    'degradations',
    'unexpected_drifts',
    'cost_analysis',
    'slo_burn_downs',
    'recommendations_v1_0_2',
    'process_improvements',
    'action_items'
]

missing = [k for k in required_keys if k not in data]
if missing:
    print(f'ERROR: Missing JSON keys: {missing}')
    sys.exit(1)

print('  ✓ JSON structure validated')
"

# Validate Markdown sections
echo ""
echo "Step 4: Validating Markdown sections..."

EXPECTED_SECTIONS=(
    "# T.A.R.S. v1.0.1 - GA 7-Day Retrospective"
    "## Executive Summary"
    "## What Went Well"
    "## What Could Be Improved"
    "## Unexpected Drifts"
    "## Cost Analysis"
    "## SLO Compliance Summary"
    "## Recommendations for v1.0.2"
    "## Process Improvements"
    "## Action Items"
)

for section in "${EXPECTED_SECTIONS[@]}"; do
    if ! grep -q "$section" "$TEST_OUTPUT_DIR/GA_7DAY_RETROSPECTIVE.md"; then
        echo "ERROR: Missing section: $section"
        exit 1
    fi
done

echo "  ✓ All expected Markdown sections found"

# Display summary
echo ""
echo "============================================================"
echo "SMOKE TEST PASSED"
echo "============================================================"
echo ""
echo "Reports generated:"
echo "  - Markdown: $TEST_OUTPUT_DIR/GA_7DAY_RETROSPECTIVE.md"
echo "  - JSON: $TEST_OUTPUT_DIR/GA_7DAY_RETROSPECTIVE.json"
echo ""

# Display file sizes
MD_SIZE=$(wc -c < "$TEST_OUTPUT_DIR/GA_7DAY_RETROSPECTIVE.md")
JSON_SIZE=$(wc -c < "$TEST_OUTPUT_DIR/GA_7DAY_RETROSPECTIVE.json")

echo "File sizes:"
echo "  - Markdown: $(($MD_SIZE / 1024)) KB"
echo "  - JSON: $(($JSON_SIZE / 1024)) KB"
echo ""

# Display quick stats from JSON
python -c "
import json

with open('$TEST_OUTPUT_DIR/GA_7DAY_RETROSPECTIVE.json') as f:
    data = json.load(f)

print('Quick Stats:')
print(f'  - Successes: {len(data[\"successes\"])}')
print(f'  - Degradations: {len(data[\"degradations\"])}')
print(f'  - Unexpected Drifts: {len(data[\"unexpected_drifts\"])}')
print(f'  - Cost Trend: {data[\"cost_analysis\"][\"cost_trend\"]}')
print(f'  - Recommendations: {len(data[\"recommendations_v1_0_2\"])}')
print(f'  - Process Improvements: {len(data[\"process_improvements\"])}')
print(f'  - Action Items: {len(data[\"action_items\"])}')
"

echo ""
echo "============================================================"
