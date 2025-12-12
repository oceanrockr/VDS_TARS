"""
Shared observability utilities for T.A.R.S. monitoring.

This module provides shared components used across Phase 14.5 (GA Day)
and Phase 14.6 (7-Day Stability) monitoring pipelines.
"""

from .prometheus_client import PrometheusClient, PrometheusQueryError

__all__ = ["PrometheusClient", "PrometheusQueryError"]
