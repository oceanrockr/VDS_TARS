#!/usr/bin/env python3
"""
CLI entry point for tars-retro command.

Wraps scripts/generate_retrospective.py for distribution as a Python package.
"""

import sys
from pathlib import Path


def main():
    """Entry point for tars-retro CLI command."""
    # Add project root to sys.path to import scripts modules
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))

    # Import and run the retrospective generator
    try:
        from scripts import generate_retrospective

        # Run the main function
        sys.exit(generate_retrospective.main())
    except ImportError as e:
        print(f"Error: Could not import generate_retrospective: {e}", file=sys.stderr)
        print("Make sure the scripts module is available.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error running retrospective generator: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
