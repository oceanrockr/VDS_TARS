#!/usr/bin/env bash
#
# examples/compliance_check.sh
#
# Quick compliance status checker for T.A.R.S. Phase 14.6 (v1.0.2-RC1).
#
# Uses the Python compliance framework (ComplianceEnforcer + enterprise_config)
# to compute SOC 2 / ISO 27001 / GDPR alignment and prints a human-readable
# summary plus optional JSON output for CI pipelines.
#
# Exit Codes:
#   0 - Compliance check successful
#   1 - Compliance check failed (errors during execution)
#   2 - Installation/configuration issue (modules missing, invalid config)
#
# Usage Examples:
#   ./examples/compliance_check.sh
#   ./examples/compliance_check.sh --profile prod --standards soc2,iso27001,gdpr
#   ./examples/compliance_check.sh --json
#   ./examples/compliance_check.sh --config /etc/tars/config.yaml --profile prod --verbose

set -euo pipefail

# ============================================================================
# CONFIGURATION
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Default values
PROFILE="local"
CONFIG_FILE=""
STANDARDS="soc2,iso27001"
OUTPUT_JSON=0
VERBOSE=0

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

print_usage() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

Quick compliance status checker for T.A.R.S. Phase 14.6.

Options:
  --profile <name>       Environment profile (local, dev, staging, prod)
                         Default: local
  --config <path>        Explicit config file path
  --standards <list>     Comma-separated standards (soc2,iso27001,gdpr)
                         Default: soc2,iso27001
  --json                 Output JSON summary for CI/CD pipelines
  --verbose              Print detailed control status
  --help                 Show this help message

Examples:
  # Basic check (local profile, SOC 2 + ISO 27001)
  ./examples/compliance_check.sh

  # Production profile with all standards
  ./examples/compliance_check.sh --profile prod --standards soc2,iso27001,gdpr

  # JSON output for CI pipeline
  ./examples/compliance_check.sh --json

  # Custom config with verbose output
  ./examples/compliance_check.sh --config /etc/tars/config.yaml --profile prod --verbose

Exit Codes:
  0 - Compliance check successful
  1 - Compliance check failed (errors during execution)
  2 - Installation/configuration issue (modules missing, invalid config)

EOF
}

print_error() {
    echo "ERROR: $*" >&2
}

print_info() {
    echo "INFO: $*"
}

# ============================================================================
# ARGUMENT PARSING
# ============================================================================

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --profile)
                PROFILE="$2"
                shift 2
                ;;
            --config)
                CONFIG_FILE="$2"
                shift 2
                ;;
            --standards)
                STANDARDS="$2"
                shift 2
                ;;
            --json)
                OUTPUT_JSON=1
                shift
                ;;
            --verbose)
                VERBOSE=1
                shift
                ;;
            --help|-h)
                print_usage
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                print_usage
                exit 2
                ;;
        esac
    done
}

# ============================================================================
# COMPLIANCE CHECK
# ============================================================================

run_compliance_check() {
    # Export environment variables for Python script
    export TARS_PROFILE="$PROFILE"
    export TARS_CONFIG_FILE="$CONFIG_FILE"
    export TARS_STANDARDS="$STANDARDS"
    export TARS_OUTPUT_JSON="$OUTPUT_JSON"
    export TARS_VERBOSE="$VERBOSE"
    export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

    # Run embedded Python script
    python3 - << 'EOF'
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any

# ============================================================================
# ENVIRONMENT VARIABLE PARSING
# ============================================================================

PROFILE = os.getenv('TARS_PROFILE', 'local')
CONFIG_FILE = os.getenv('TARS_CONFIG_FILE', '')
STANDARDS_STR = os.getenv('TARS_STANDARDS', 'soc2,iso27001')
OUTPUT_JSON = os.getenv('TARS_OUTPUT_JSON', '0') == '1'
VERBOSE = os.getenv('TARS_VERBOSE', '0') == '1'

# Parse standards list
STANDARDS = [s.strip().lower() for s in STANDARDS_STR.split(',') if s.strip()]

# ============================================================================
# MODULE IMPORTS
# ============================================================================

try:
    from enterprise_config.config_loader import load_config
    from compliance.compliance_enforcer import ComplianceEnforcer
except ImportError as e:
    print(f"ERROR: T.A.R.S. enterprise compliance modules not available.", file=sys.stderr)
    print(f"       Did you install Phase 14.6 dependencies?", file=sys.stderr)
    print(f"       Missing module: {e}", file=sys.stderr)
    print(f"", file=sys.stderr)
    print(f"       Install with: pip install -r requirements-dev.txt", file=sys.stderr)
    sys.exit(2)

# ============================================================================
# CONFIGURATION LOADING
# ============================================================================

try:
    # Load enterprise configuration
    if CONFIG_FILE:
        config = load_config(profile=PROFILE, config_path=Path(CONFIG_FILE))
    else:
        config = load_config(profile=PROFILE)

    if VERBOSE:
        print(f"✓ Loaded configuration for profile: {PROFILE}")
        if CONFIG_FILE:
            print(f"  Config file: {CONFIG_FILE}")

except Exception as e:
    print(f"ERROR: Failed to load enterprise configuration.", file=sys.stderr)
    print(f"       Profile: {PROFILE}", file=sys.stderr)
    if CONFIG_FILE:
        print(f"       Config file: {CONFIG_FILE}", file=sys.stderr)
    print(f"       Error: {e}", file=sys.stderr)
    sys.exit(2)

# ============================================================================
# COMPLIANCE ENFORCEMENT
# ============================================================================

try:
    # Initialize compliance enforcer
    enforcer = ComplianceEnforcer(config=config)

    if VERBOSE:
        print(f"✓ Initialized ComplianceEnforcer")
        print(f"  Enabled standards: {', '.join(STANDARDS)}")

    # Get compliance summary
    summary = enforcer.get_summary_report()

    if VERBOSE:
        print(f"✓ Generated compliance summary report")

except Exception as e:
    print(f"ERROR: Failed to compute compliance status.", file=sys.stderr)
    print(f"       Error: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# PARSE COMPLIANCE RESULTS
# ============================================================================

# Extract overall compliance
overall_compliance = summary.get('overall_compliance_percentage', 0.0)

# Extract by-standard breakdown
by_standard = {}
standard_details = {}

for standard in STANDARDS:
    std_upper = standard.upper()
    std_data = summary.get(f'{standard}_summary', {})

    if std_data:
        compliance_pct = std_data.get('compliance_percentage', 0.0)
        by_standard[standard] = compliance_pct
        standard_details[standard] = {
            'compliance_percentage': compliance_pct,
            'implemented': std_data.get('implemented', 0),
            'partial': std_data.get('partial', 0),
            'not_implemented': std_data.get('not_implemented', 0),
            'total_controls': std_data.get('total_controls', 0),
        }

# Extract overall control counts
total_implemented = summary.get('total_implemented', 0)
total_partial = summary.get('total_partial', 0)
total_not_implemented = summary.get('total_not_implemented', 0)

# Determine readiness
if overall_compliance >= 90:
    readiness = "✅ Ready for external audit"
elif overall_compliance >= 75:
    readiness = "⚠️  Ready for internal audit (minor gaps)"
elif overall_compliance >= 60:
    readiness = "🔶 Partial compliance (moderate gaps)"
else:
    readiness = "❌ Significant compliance gaps"

# ============================================================================
# HUMAN-READABLE OUTPUT
# ============================================================================

if not OUTPUT_JSON:
    print("")
    print("=" * 80)
    print("  T.A.R.S. Enterprise Compliance Status (Phase 14.6 / v1.0.2-RC1)")
    print("=" * 80)
    print("")
    print(f"Profile:    {PROFILE}")
    print(f"Standards:  {', '.join([s.upper() for s in STANDARDS])}")
    print("")
    print(f"Overall Compliance: {overall_compliance:.1f}%")
    print("")

    if by_standard:
        print("By Standard:")
        for standard in STANDARDS:
            if standard in standard_details:
                details = standard_details[standard]
                pct = details['compliance_percentage']
                impl = details['implemented']
                part = details['partial']
                total = details['total_controls']

                print(f"  - {standard.upper():10s}  {pct:5.1f}%  ({impl}/{total} controls implemented, {part} partial)")
        print("")

    print("Control Status:")
    print(f"  - Implemented:          {total_implemented}")
    print(f"  - Partially Implemented: {total_partial}")
    print(f"  - Not Implemented:       {total_not_implemented}")
    print("")

    print(f"Assessment: {readiness}")
    print("")

    # Verbose output: show failing controls
    if VERBOSE and total_not_implemented > 0:
        print("Controls Not Implemented:")

        for standard in STANDARDS:
            std_data = summary.get(f'{standard}_controls', [])
            if std_data:
                failing = [c for c in std_data if c.get('status') == 'not_implemented']
                if failing:
                    print(f"\n  {standard.upper()}:")
                    for control in failing[:5]:  # Limit to 5 per standard
                        control_id = control.get('control_id', 'Unknown')
                        description = control.get('description', 'No description')
                        print(f"    - {control_id}: {description}")

                    if len(failing) > 5:
                        print(f"    ... and {len(failing) - 5} more")
        print("")

# ============================================================================
# JSON OUTPUT
# ============================================================================

if OUTPUT_JSON:
    json_output = {
        "profile": PROFILE,
        "standards": STANDARDS,
        "overall_compliance": round(overall_compliance, 2),
        "by_standard": {k: round(v, 2) for k, v in by_standard.items()},
        "implemented": total_implemented,
        "partial": total_partial,
        "not_implemented": total_not_implemented,
        "ready_for_audit": overall_compliance >= 90,
    }

    print(json.dumps(json_output, separators=(",", ":")))

# ============================================================================
# EXIT CODE
# ============================================================================

# Success
sys.exit(0)
EOF

    # Capture Python exit code
    PYTHON_EXIT=$?

    return $PYTHON_EXIT
}

# ============================================================================
# MAIN
# ============================================================================

main() {
    # Parse command-line arguments
    parse_args "$@"

    # Run compliance check
    run_compliance_check

    # Capture exit code
    EXIT_CODE=$?

    return $EXIT_CODE
}

# Run main function
main "$@"
exit $?
