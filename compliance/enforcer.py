"""
Compliance Enforcer

Runtime enforcement of compliance controls for T.A.R.S. operations.
"""

from typing import List, Dict, Optional, Any
from pathlib import Path
from datetime import datetime, timedelta
import json
import re

from .controls import ComplianceControl, ControlStatus, load_controls, ControlCategory
from .audit import AuditLogger


class ComplianceViolation(Exception):
    """Raised when a compliance control is violated."""

    def __init__(self, control_id: str, message: str):
        self.control_id = control_id
        self.message = message
        super().__init__(f"Compliance violation [{control_id}]: {message}")


class ComplianceEnforcer:
    """
    Enforces compliance controls at runtime.

    Features:
    - Input sanitization
    - Data retention enforcement
    - Encryption validation
    - Access control checks
    - Audit logging
    """

    def __init__(
        self,
        enabled_standards: List[str],
        controls_dir: Path = Path("compliance/policies"),
        audit_log_path: Optional[Path] = None,
        strict_mode: bool = False,
    ):
        """
        Initialize compliance enforcer.

        Args:
            enabled_standards: List of enabled compliance standards
            controls_dir: Directory containing control definitions
            audit_log_path: Path to audit log file
            strict_mode: If True, raise exceptions on violations; if False, log warnings
        """
        self.enabled_standards = enabled_standards
        self.controls_dir = controls_dir
        self.strict_mode = strict_mode

        # Load controls
        self.controls: List[ComplianceControl] = []
        for standard in enabled_standards:
            controls_file = controls_dir / f"standard_{standard}.yaml"
            if controls_file.exists():
                self.controls.extend(load_controls(controls_file))

        # Initialize audit logger
        self.audit_logger = AuditLogger(audit_log_path) if audit_log_path else None

        # Cache for control lookups
        self._control_cache: Dict[str, ComplianceControl] = {
            c.control_id: c for c in self.controls
        }

    def enforce_input_sanitization(
        self,
        input_data: str,
        max_length: int = 10000,
        allowed_patterns: Optional[List[str]] = None,
    ) -> str:
        """
        Enforce input sanitization controls.

        Args:
            input_data: Input string to sanitize
            max_length: Maximum allowed length
            allowed_patterns: List of regex patterns (if provided, input must match one)

        Returns:
            Sanitized input

        Raises:
            ComplianceViolation: If input violates sanitization controls
        """
        # Check length
        if len(input_data) > max_length:
            self._log_violation(
                "INPUT_SANITIZATION",
                f"Input exceeds maximum length: {len(input_data)} > {max_length}"
            )
            if self.strict_mode:
                raise ComplianceViolation(
                    "INPUT_SANITIZATION",
                    f"Input exceeds maximum length: {len(input_data)} > {max_length}"
                )
            input_data = input_data[:max_length]

        # Check patterns
        if allowed_patterns:
            matches = any(re.match(pattern, input_data) for pattern in allowed_patterns)
            if not matches:
                self._log_violation(
                    "INPUT_SANITIZATION",
                    "Input does not match allowed patterns"
                )
                if self.strict_mode:
                    raise ComplianceViolation(
                        "INPUT_SANITIZATION",
                        "Input does not match allowed patterns"
                    )

        # Remove potentially dangerous characters
        sanitized = self._sanitize_string(input_data)

        self._log_audit("input_sanitization", {
            "original_length": len(input_data),
            "sanitized_length": len(sanitized),
            "modified": input_data != sanitized,
        })

        return sanitized

    def enforce_data_retention(
        self,
        file_path: Path,
        retention_days: int,
        enforce_deletion: bool = False,
    ) -> bool:
        """
        Enforce GDPR/compliance data retention policies.

        Args:
            file_path: Path to file to check
            retention_days: Maximum retention period (days)
            enforce_deletion: If True, delete files exceeding retention

        Returns:
            True if file is compliant, False otherwise
        """
        if not file_path.exists():
            return True

        # Check file age
        file_age = datetime.now() - datetime.fromtimestamp(file_path.stat().st_mtime)
        exceeds_retention = file_age > timedelta(days=retention_days)

        if exceeds_retention:
            self._log_audit("data_retention_check", {
                "file_path": str(file_path),
                "file_age_days": file_age.days,
                "retention_days": retention_days,
                "exceeds_retention": True,
            })

            if enforce_deletion:
                file_path.unlink()
                self._log_audit("data_retention_deletion", {
                    "file_path": str(file_path),
                    "reason": "Exceeded retention period",
                })
                return True

            self._log_violation(
                "DATA_RETENTION",
                f"File exceeds retention period: {file_path} ({file_age.days} days)"
            )
            return False

        return True

    def enforce_encryption(
        self,
        file_path: Path,
        require_encryption: bool = True,
    ) -> bool:
        """
        Validate that sensitive files are encrypted.

        Args:
            file_path: Path to file to check
            require_encryption: If True, raise violation if not encrypted

        Returns:
            True if file is encrypted (or encryption not required), False otherwise
        """
        if not file_path.exists():
            return True

        # Check if file is encrypted (simple heuristic: check for common encrypted file markers)
        is_encrypted = self._check_file_encrypted(file_path)

        if not is_encrypted and require_encryption:
            self._log_violation(
                "ENCRYPTION",
                f"File is not encrypted: {file_path}"
            )
            if self.strict_mode:
                raise ComplianceViolation(
                    "ENCRYPTION",
                    f"File is not encrypted: {file_path}"
                )
            return False

        self._log_audit("encryption_check", {
            "file_path": str(file_path),
            "is_encrypted": is_encrypted,
        })

        return is_encrypted

    def enforce_access_control(
        self,
        user_role: str,
        required_role: str,
        resource: str,
    ) -> bool:
        """
        Enforce RBAC access control.

        Args:
            user_role: User's role (readonly, sre, admin)
            required_role: Required role for access
            resource: Resource being accessed

        Returns:
            True if access is allowed, False otherwise

        Raises:
            ComplianceViolation: If access is denied and strict_mode is enabled
        """
        # Role hierarchy: admin > sre > readonly
        role_hierarchy = {
            "readonly": 1,
            "sre": 2,
            "admin": 3,
        }

        user_level = role_hierarchy.get(user_role, 0)
        required_level = role_hierarchy.get(required_role, 99)

        access_granted = user_level >= required_level

        self._log_audit("access_control", {
            "user_role": user_role,
            "required_role": required_role,
            "resource": resource,
            "access_granted": access_granted,
        })

        if not access_granted:
            self._log_violation(
                "ACCESS_CONTROL",
                f"Access denied: user role '{user_role}' insufficient for '{resource}' (requires '{required_role}')"
            )
            if self.strict_mode:
                raise ComplianceViolation(
                    "ACCESS_CONTROL",
                    f"Access denied: insufficient permissions"
                )

        return access_granted

    def redact_sensitive_data(
        self,
        data: Dict[str, Any],
        redaction_patterns: List[str],
    ) -> Dict[str, Any]:
        """
        Redact sensitive fields from data dictionary.

        Args:
            data: Data dictionary to redact
            redaction_patterns: List of regex patterns for sensitive fields

        Returns:
            Redacted data dictionary
        """
        redacted = {}

        for key, value in data.items():
            # Check if key matches any redaction pattern
            should_redact = any(
                re.search(pattern, key, re.IGNORECASE)
                for pattern in redaction_patterns
            )

            if should_redact:
                redacted[key] = "***REDACTED***"
            elif isinstance(value, dict):
                redacted[key] = self.redact_sensitive_data(value, redaction_patterns)
            elif isinstance(value, list):
                redacted[key] = [
                    self.redact_sensitive_data(item, redaction_patterns)
                    if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                redacted[key] = value

        return redacted

    def get_compliance_status(self) -> Dict[str, Any]:
        """
        Get current compliance status summary.

        Returns:
            Dictionary with compliance metrics
        """
        total_controls = len(self.controls)
        status_counts = {}

        for control in self.controls:
            status_counts[control.status.value] = status_counts.get(control.status.value, 0) + 1

        return {
            "enabled_standards": self.enabled_standards,
            "total_controls": total_controls,
            "status_counts": status_counts,
            "compliance_percentage": self._calculate_compliance_percentage(),
        }

    # Private helper methods

    def _sanitize_string(self, s: str) -> str:
        """Remove potentially dangerous characters from string."""
        # Remove null bytes
        s = s.replace("\x00", "")

        # Remove control characters (except newline, tab, carriage return)
        s = "".join(char for char in s if char in "\n\r\t" or ord(char) >= 32)

        return s

    def _check_file_encrypted(self, file_path: Path) -> bool:
        """
        Simple heuristic to check if file is encrypted.

        Note: This is a basic check. For production, use proper encryption markers.
        """
        try:
            with open(file_path, "rb") as f:
                header = f.read(16)

            # Check for common encrypted file markers
            # PGP: starts with "-----BEGIN PGP MESSAGE-----"
            # AES: might have custom header

            # For now, check if file is mostly non-ASCII (entropy-based heuristic)
            if b"-----BEGIN PGP" in header:
                return True

            # Check entropy (encrypted files have high entropy)
            non_ascii_count = sum(1 for byte in header if byte > 127)
            return non_ascii_count / len(header) > 0.5 if header else False

        except Exception:
            return False

    def _calculate_compliance_percentage(self) -> float:
        """Calculate overall compliance percentage."""
        if not self.controls:
            return 100.0

        score_map = {
            ControlStatus.VERIFIED: 100,
            ControlStatus.IMPLEMENTED: 80,
            ControlStatus.PARTIALLY_IMPLEMENTED: 50,
            ControlStatus.NOT_IMPLEMENTED: 0,
            ControlStatus.NON_COMPLIANT: 0,
        }

        total_score = sum(score_map[c.status] for c in self.controls)
        max_score = len(self.controls) * 100

        return (total_score / max_score) * 100 if max_score > 0 else 0.0

    def _log_violation(self, control_id: str, message: str):
        """Log compliance violation."""
        if self.audit_logger:
            self.audit_logger.log_event("compliance_violation", {
                "control_id": control_id,
                "message": message,
                "timestamp": datetime.utcnow().isoformat(),
            })

    def _log_audit(self, event_type: str, data: Dict[str, Any]):
        """Log audit event."""
        if self.audit_logger:
            self.audit_logger.log_event(event_type, data)
