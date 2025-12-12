"""
Internal Telemetry and Metrics for T.A.R.S.

Tracks T.A.R.S. system performance and operations.
"""

from .telemetry import TelemetryCollector, track_command, track_error
from .logging_config import configure_logging

__all__ = [
    "TelemetryCollector",
    "track_command",
    "track_error",
    "configure_logging",
]

__version__ = "1.0.2-dev"
