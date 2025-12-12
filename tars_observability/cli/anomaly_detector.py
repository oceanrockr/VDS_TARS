#!/usr/bin/env python3
"""
CLI entry point for tars-anomaly-detector command.

Wraps observability/anomaly_detector_lightweight.py for distribution as a Python package.
"""

import sys
from pathlib import Path


def main():
    """Entry point for tars-anomaly-detector CLI command."""
    # Add project root to sys.path to import observability modules
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))

    # Import and run the anomaly detector
    try:
        from observability import anomaly_detector_lightweight

        # Run the main function
        sys.exit(anomaly_detector_lightweight.main())
    except ImportError as e:
        print(f"Error: Could not import anomaly_detector_lightweight: {e}", file=sys.stderr)
        print("Make sure the observability module is available.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error running anomaly detector: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
