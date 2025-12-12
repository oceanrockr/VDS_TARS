#!/usr/bin/env python3
"""
CLI entry point for tars-regression-analyzer command.

Wraps observability/regression_analyzer.py for distribution as a Python package.
"""

import sys
from pathlib import Path


def main():
    """Entry point for tars-regression-analyzer CLI command."""
    # Add project root to sys.path to import observability modules
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))

    # Import and run the regression analyzer
    try:
        from observability import regression_analyzer

        # Run the main function
        sys.exit(regression_analyzer.main())
    except ImportError as e:
        print(f"Error: Could not import regression_analyzer: {e}", file=sys.stderr)
        print("Make sure the observability module is available.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error running regression analyzer: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
