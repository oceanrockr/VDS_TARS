#!/usr/bin/env python3
"""
CLI entry point for tars-stability-monitor command.

Wraps observability/stability_monitor_7day.py for distribution as a Python package.
"""

import sys
from pathlib import Path


def main():
    """Entry point for tars-stability-monitor CLI command."""
    # Add project root to sys.path to import observability modules
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))

    # Import and run the stability monitor
    try:
        from observability import stability_monitor_7day

        # Run the main function
        sys.exit(stability_monitor_7day.main())
    except ImportError as e:
        print(f"Error: Could not import stability_monitor_7day: {e}", file=sys.stderr)
        print("Make sure the observability module is available.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error running stability monitor: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
