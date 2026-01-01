"""
Enterprise API main module - re-exports app from app.py.

This module provides backward compatibility for imports from enterprise_api.main.
"""

from .app import app

__all__ = ["app"]
