"""
T.A.R.S. Observability Framework - Phase 14.6

Post-GA 7-Day Stabilization & Retrospective Framework for production monitoring,
anomaly detection, health reporting, and automated retrospective generation.

Modules:
    - core: Core monitoring and analysis components
    - cli: Command-line interface tools

CLI Tools:
    - tars-ga-kpi: GA Day KPI collection
    - tars-stability-monitor: 7-day stability monitoring
    - tars-anomaly-detector: EWMA-based anomaly detection
    - tars-health-report: Daily health scoring and mitigation
    - tars-regression-analyzer: Multi-baseline regression analysis
    - tars-retro: Comprehensive retrospective generation

Usage:
    Install the package:
        pip install tars-observability

    Run CLI tools:
        tars-ga-kpi --prometheus-url http://localhost:9090
        tars-stability-monitor --day-number 1
        tars-retro --auto

    Import as library:
        from tars_observability.core import collectors
        from tars_observability.core import analyzers
"""

from tars_observability.__version__ import (
    __version__,
    __version_info__,
    PHASE,
    PHASE_NAME,
    RELEASE_DATE,
    get_version_string,
    get_full_version_info,
)

__all__ = [
    "__version__",
    "__version_info__",
    "PHASE",
    "PHASE_NAME",
    "RELEASE_DATE",
    "get_version_string",
    "get_full_version_info",
]
