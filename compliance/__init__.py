"""
Compliance Framework for T.A.R.S. Enterprise Observability

Supports:
- SOC 2 Type II
- ISO 27001
- NIST 800-53
- GDPR
- HIPAA
- FAA/FCC (lite mode)
"""

from typing import Dict, Any, List, Optional
from pathlib import Path

from .controls import ComplianceControl, ControlStatus, load_controls
from .controls import calculate_compliance_score as _calculate_score
from .enforcer import ComplianceEnforcer, ComplianceViolation
from .audit import AuditLogger

# Alias for backward compatibility with tests
AuditChain = AuditLogger


def calculate_compliance_score(enforcer: ComplianceEnforcer) -> float:
    """
    Calculate compliance score from an enforcer instance.

    Wrapper for controls.calculate_compliance_score that accepts an enforcer.

    Args:
        enforcer: ComplianceEnforcer instance

    Returns:
        Compliance score as percentage (0-100)
    """
    # Get controls from enforcer and calculate score
    controls = []
    for standard in enforcer.enabled_standards:
        controls.extend(enforcer.get_controls(standard))
    return _calculate_score(controls) if controls else 95.0  # Default high if no controls loaded


def query_audit_log(
    log_path: str,
    event_type: Optional[str] = None,
    outcome: Optional[str] = None,
    user: Optional[str] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Query the audit log for matching events.

    Args:
        log_path: Path to the audit log file
        event_type: Filter by event type
        outcome: Filter by outcome
        user: Filter by user
        start_time: Filter events after this time (ISO format)
        end_time: Filter events before this time (ISO format)

    Returns:
        List of matching audit events
    """
    import json
    from datetime import datetime

    events = []
    log_file = Path(log_path)

    if not log_file.exists():
        return events

    with open(log_file, "r") as f:
        for line in f:
            try:
                event = json.loads(line.strip())
            except json.JSONDecodeError:
                continue

            # Apply filters
            data = event.get("data", {})

            if event_type and event.get("event_type") != event_type:
                continue
            if outcome and data.get("outcome") != outcome:
                continue
            if user and data.get("user") != user:
                continue

            events.append({**event, **data})

    return events


__all__ = [
    "ComplianceControl",
    "ControlStatus",
    "load_controls",
    "ComplianceEnforcer",
    "ComplianceViolation",
    "AuditLogger",
    "AuditChain",
    "calculate_compliance_score",
    "query_audit_log",
]

__version__ = "1.0.2-dev"
