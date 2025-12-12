"""
Audit Logger for Compliance Events

Provides immutable audit trail for compliance activities.
"""

from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import json
import hashlib


class AuditLogger:
    """
    Immutable audit logger for compliance events.

    Features:
    - JSONL format for append-only logging
    - Event chaining with cryptographic hashing
    - Tamper detection
    """

    def __init__(self, log_path: Optional[Path] = None):
        """
        Initialize audit logger.

        Args:
            log_path: Path to audit log file (JSONL format)
        """
        self.log_path = log_path or Path("logs/compliance_audit.jsonl")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize with previous hash for chain integrity
        self._previous_hash = self._get_last_hash()

    def log_event(self, event_type: str, data: Dict[str, Any]):
        """
        Log a compliance audit event.

        Args:
            event_type: Type of audit event
            data: Event data
        """
        event = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "event_type": event_type,
            "data": data,
            "previous_hash": self._previous_hash,
        }

        # Calculate hash for chain integrity
        event_hash = self._calculate_hash(event)
        event["event_hash"] = event_hash

        # Write to log file (append-only)
        with open(self.log_path, "a") as f:
            f.write(json.dumps(event) + "\n")

        # Update previous hash for next event
        self._previous_hash = event_hash

    def verify_integrity(self) -> bool:
        """
        Verify audit log chain integrity.

        Returns:
            True if log is intact, False if tampering detected
        """
        if not self.log_path.exists():
            return True  # Empty log is valid

        previous_hash = None

        with open(self.log_path, "r") as f:
            for line in f:
                event = json.loads(line.strip())

                # Check hash chain
                if event["previous_hash"] != previous_hash:
                    return False  # Chain broken

                # Verify event hash
                event_copy = {k: v for k, v in event.items() if k != "event_hash"}
                expected_hash = self._calculate_hash(event_copy)

                if event["event_hash"] != expected_hash:
                    return False  # Event tampered

                previous_hash = event["event_hash"]

        return True

    def _get_last_hash(self) -> Optional[str]:
        """Get hash of last event in log."""
        if not self.log_path.exists():
            return None

        try:
            with open(self.log_path, "r") as f:
                # Read last line
                lines = f.readlines()
                if not lines:
                    return None

                last_event = json.loads(lines[-1].strip())
                return last_event.get("event_hash")
        except Exception:
            return None

    def _calculate_hash(self, event: Dict[str, Any]) -> str:
        """Calculate SHA-256 hash of event."""
        event_json = json.dumps(event, sort_keys=True)
        return hashlib.sha256(event_json.encode()).hexdigest()
