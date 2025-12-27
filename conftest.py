"""
Root conftest.py for T.A.R.S. project.

This file ensures the project root is in sys.path before any tests are collected.
"""

import os
import sys

# Add project root to path - this must happen before any imports
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
