#!/usr/bin/env python3
"""
CLI entry point for tars-health-report command.

Wraps observability/daily_health_reporter.py for distribution as a Python package.
"""

import sys
from pathlib import Path


def main():
    """Entry point for tars-health-report CLI command."""
    # Add project root to sys.path to import observability modules
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))

    # Import and run the health reporter
    try:
        from observability import daily_health_reporter

        # Run the main function
        sys.exit(daily_health_reporter.main())
    except ImportError as e:
        print(f"Error: Could not import daily_health_reporter: {e}", file=sys.stderr)
        print("Make sure the observability module is available.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error running health reporter: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
