"""
Compliance Controls Management

Defines and loads compliance controls for various standards.
"""

from typing import List, Dict, Optional, Any
from enum import Enum
from dataclasses import dataclass, field
from pathlib import Path
import yaml


class ControlStatus(str, Enum):
    """Status of compliance control implementation."""
    NOT_IMPLEMENTED = "not_implemented"
    PARTIALLY_IMPLEMENTED = "partially_implemented"
    IMPLEMENTED = "implemented"
    VERIFIED = "verified"
    NON_COMPLIANT = "non_compliant"


class ControlCategory(str, Enum):
    """Categories of compliance controls."""
    ACCESS_CONTROL = "access_control"
    AUDIT_LOGGING = "audit_logging"
    DATA_PROTECTION = "data_protection"
    ENCRYPTION = "encryption"
    INCIDENT_RESPONSE = "incident_response"
    MONITORING = "monitoring"
    RISK_MANAGEMENT = "risk_management"
    SECURE_DEVELOPMENT = "secure_development"
    SYSTEM_HARDENING = "system_hardening"
    VENDOR_MANAGEMENT = "vendor_management"


@dataclass
class ComplianceControl:
    """
    Represents a single compliance control.
    """

    control_id: str
    standard: str  # soc2, iso27001, nist_800_53, etc.
    title: str
    description: str
    category: ControlCategory
    status: ControlStatus = ControlStatus.NOT_IMPLEMENTED

    # Implementation details
    implementation_notes: Optional[str] = None
    evidence_paths: List[str] = field(default_factory=list)

    # Metadata
    severity: str = "medium"  # low, medium, high, critical
    automated: bool = False
    testing_procedure: Optional[str] = None

    # Mapping to T.A.R.S. components
    related_components: List[str] = field(default_factory=list)
    related_apis: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Export control as dictionary."""
        return {
            "control_id": self.control_id,
            "standard": self.standard,
            "title": self.title,
            "description": self.description,
            "category": self.category.value,
            "status": self.status.value,
            "implementation_notes": self.implementation_notes,
            "evidence_paths": self.evidence_paths,
            "severity": self.severity,
            "automated": self.automated,
            "testing_procedure": self.testing_procedure,
            "related_components": self.related_components,
            "related_apis": self.related_apis,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ComplianceControl":
        """Create control from dictionary."""
        return cls(
            control_id=data["control_id"],
            standard=data["standard"],
            title=data["title"],
            description=data["description"],
            category=ControlCategory(data["category"]),
            status=ControlStatus(data.get("status", "not_implemented")),
            implementation_notes=data.get("implementation_notes"),
            evidence_paths=data.get("evidence_paths", []),
            severity=data.get("severity", "medium"),
            automated=data.get("automated", False),
            testing_procedure=data.get("testing_procedure"),
            related_components=data.get("related_components", []),
            related_apis=data.get("related_apis", []),
        )


def load_controls(controls_file: Path) -> List[ComplianceControl]:
    """
    Load compliance controls from YAML file.

    Args:
        controls_file: Path to controls definition file

    Returns:
        List of ComplianceControl instances
    """
    if not controls_file.exists():
        raise FileNotFoundError(f"Controls file not found: {controls_file}")

    with open(controls_file, "r") as f:
        data = yaml.safe_load(f)

    controls = []
    for control_data in data.get("controls", []):
        controls.append(ComplianceControl.from_dict(control_data))

    return controls


def get_controls_by_standard(
    controls: List[ComplianceControl],
    standard: str
) -> List[ComplianceControl]:
    """Filter controls by standard."""
    return [c for c in controls if c.standard == standard]


def get_controls_by_status(
    controls: List[ComplianceControl],
    status: ControlStatus
) -> List[ComplianceControl]:
    """Filter controls by implementation status."""
    return [c for c in controls if c.status == status]


def get_controls_by_category(
    controls: List[ComplianceControl],
    category: ControlCategory
) -> List[ComplianceControl]:
    """Filter controls by category."""
    return [c for c in controls if c.category == category]


def calculate_compliance_score(controls: List[ComplianceControl]) -> float:
    """
    Calculate overall compliance score (0-100%).

    Score calculation:
    - verified: 100%
    - implemented: 80%
    - partially_implemented: 50%
    - not_implemented: 0%
    - non_compliant: 0%

    Args:
        controls: List of compliance controls

    Returns:
        Compliance score as percentage (0-100)
    """
    if not controls:
        return 0.0

    score_map = {
        ControlStatus.VERIFIED: 100,
        ControlStatus.IMPLEMENTED: 80,
        ControlStatus.PARTIALLY_IMPLEMENTED: 50,
        ControlStatus.NOT_IMPLEMENTED: 0,
        ControlStatus.NON_COMPLIANT: 0,
    }

    total_score = sum(score_map[c.status] for c in controls)
    max_score = len(controls) * 100

    return (total_score / max_score) * 100 if max_score > 0 else 0.0
