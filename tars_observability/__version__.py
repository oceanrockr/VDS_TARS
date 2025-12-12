"""Version information for T.A.R.S. Observability Framework."""

__version__ = "1.0.2-pre"
__version_info__ = tuple(
    int(part) if part.isdigit() else part
    for part in __version__.replace("-", ".").split(".")
)

# Phase information
PHASE = "14.6"
PHASE_NAME = "Post-GA 7-Day Stabilization & Retrospective"
RELEASE_DATE = "2025-11-26"
RELEASE_STATUS = "pre-release"

# Module metadata
MODULE_NAME = "tars-observability"
MODULE_AUTHOR = "Veleron Dev Studios"
MODULE_LICENSE = "MIT"

# Feature flags
FEATURES = {
    "ga_kpi_collection": True,
    "stability_monitoring": True,
    "anomaly_detection": True,
    "health_reporting": True,
    "regression_analysis": True,
    "retrospective_generation": True,
    "cli_tools": True,
    "docker_support": True,
    "kubernetes_support": True,
}

# Version display
def get_version_string() -> str:
    """Get formatted version string."""
    return f"T.A.R.S. Observability v{__version__} (Phase {PHASE})"


def get_full_version_info() -> dict:
    """Get complete version information."""
    return {
        "version": __version__,
        "version_info": __version_info__,
        "phase": PHASE,
        "phase_name": PHASE_NAME,
        "release_date": RELEASE_DATE,
        "release_status": RELEASE_STATUS,
        "module_name": MODULE_NAME,
        "features": FEATURES,
    }


if __name__ == "__main__":
    print(get_version_string())
    import json
    print(json.dumps(get_full_version_info(), indent=2))
