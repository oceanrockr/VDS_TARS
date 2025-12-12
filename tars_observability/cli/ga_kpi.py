#!/usr/bin/env python3
"""
CLI entry point for tars-ga-kpi command.

Wraps observability/ga_kpi_collector.py for distribution as a Python package.
"""

import sys
import os
from pathlib import Path


def main():
    """Entry point for tars-ga-kpi CLI command."""
    # Add project root to sys.path to import observability modules
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))

    # Import and run the GA KPI collector
    try:
        from observability import ga_kpi_collector

        # Run the main function
        sys.exit(ga_kpi_collector.main())
    except ImportError as e:
        print(f"Error: Could not import ga_kpi_collector: {e}", file=sys.stderr)
        print("Make sure the observability module is available.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error running GA KPI collector: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
