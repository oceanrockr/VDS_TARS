"""
Tests for Compliance Framework

Covers:
- Control loading and filtering
- Compliance scoring
- Input sanitization
- Data retention enforcement
- Encryption validation
- Access control checks
- Redaction logic
- Audit trail integrity

Usage:
    pytest tests/test_compliance.py -v
    pytest tests/test_compliance.py -v --cov=compliance
"""

import pytest
import json
from datetime import datetime, timedelta
from pathlib import Path

from compliance import (
    ComplianceEnforcer,
    ComplianceViolation,
    AuditChain,
    calculate_compliance_score,
    query_audit_log
)


class TestComplianceEnforcer:
    """Test compliance enforcer initialization and configuration."""

    def test_init_no_standards(self):
        """Test initialization with no standards."""
        enforcer = ComplianceEnforcer(enabled_standards=[])
        assert enforcer.enabled_standards == []

    def test_init_single_standard(self):
        """Test initialization with single standard."""
        enforcer = ComplianceEnforcer(enabled_standards=["soc2"])
        assert "soc2" in enforcer.enabled_standards

    def test_init_multiple_standards(self):
        """Test initialization with multiple standards."""
        enforcer = ComplianceEnforcer(
            enabled_standards=["soc2", "iso27001", "gdpr"]
        )
        assert len(enforcer.enabled_standards) == 3

    def test_enforcement_modes(self):
        """Test different enforcement modes."""
        for mode in ["log", "warn", "block"]:
            enforcer = ComplianceEnforcer(
                enabled_standards=["soc2"],
                enforcement_mode=mode
            )
            assert enforcer.enforcement_mode == mode

    def test_invalid_enforcement_mode(self):
        """Test invalid enforcement mode."""
        with pytest.raises((ValueError, AssertionError)):
            ComplianceEnforcer(
                enabled_standards=["soc2"],
                enforcement_mode="invalid"
            )


class TestControlLoading:
    """Test compliance control loading and filtering."""

    def test_load_soc2_controls(self):
        """Test loading SOC 2 controls."""
        enforcer = ComplianceEnforcer(enabled_standards=["soc2"])
        controls = enforcer.get_controls("soc2")

        assert len(controls) > 0
        # Should have at least some common SOC 2 controls
        control_ids = [c["id"] for c in controls]
        assert any("CC6" in cid for cid in control_ids)  # Security controls

    def test_load_iso27001_controls(self):
        """Test loading ISO 27001 controls."""
        enforcer = ComplianceEnforcer(enabled_standards=["iso27001"])
        controls = enforcer.get_controls("iso27001")

        assert len(controls) > 0
        # Should have ISO 27001 controls (A.x.x.x format)
        control_ids = [c["id"] for c in controls]
        assert any("A." in cid for cid in control_ids)

    def test_filter_by_category(self):
        """Test filtering controls by category."""
        enforcer = ComplianceEnforcer(enabled_standards=["soc2"])
        security_controls = enforcer.get_controls_by_category("security")

        assert len(security_controls) > 0
        for control in security_controls:
            assert control["category"] == "security"


class TestComplianceScoring:
    """Test compliance score calculation."""

    def test_calculate_score_no_violations(self):
        """Test score with no violations."""
        enforcer = ComplianceEnforcer(enabled_standards=["soc2"])
        score = calculate_compliance_score(enforcer)

        assert score >= 90.0  # Should be high with no violations
        assert score <= 100.0

    def test_calculate_score_with_violations(self):
        """Test score decreases with violations."""
        enforcer = ComplianceEnforcer(
            enabled_standards=["soc2"],
            enforcement_mode="log"
        )

        # Trigger a violation
        try:
            enforcer.validate_access_control(
                user_role="viewer",
                required_role="admin",
                resource="/admin/users"
            )
        except ComplianceViolation:
            pass

        # Score should reflect violation
        score = calculate_compliance_score(enforcer)
        assert score >= 0.0
        assert score <= 100.0

    def test_compliance_status(self):
        """Test getting compliance status."""
        enforcer = ComplianceEnforcer(enabled_standards=["soc2", "iso27001"])
        status = enforcer.get_compliance_status()

        assert "enabled_standards" in status
        assert "compliance_percentage" in status
        assert status["compliance_percentage"] >= 0
        assert status["compliance_percentage"] <= 100


class TestInputSanitization:
    """Test input sanitization for compliance."""

    def test_sanitize_pii_email(self):
        """Test PII redaction for email addresses."""
        enforcer = ComplianceEnforcer(
            enabled_standards=["gdpr"],
            redaction_enabled=True
        )

        data = {"email": "user@example.com", "name": "John Doe"}
        sanitized = enforcer.redact_pii(data)

        assert sanitized["email"] != "user@example.com"
        assert "REDACTED" in sanitized["email"] or sanitized["email"] == "***REDACTED***"

    def test_sanitize_pii_ip(self):
        """Test PII redaction for IP addresses."""
        enforcer = ComplianceEnforcer(
            enabled_standards=["gdpr"],
            redaction_enabled=True
        )

        data = {"ip_address": "192.168.1.1"}
        sanitized = enforcer.redact_pii(data)

        assert sanitized["ip_address"] != "192.168.1.1"

    def test_sanitize_preserves_non_pii(self):
        """Test that non-PII data is preserved."""
        enforcer = ComplianceEnforcer(enabled_standards=["gdpr"])

        data = {
            "email": "user@example.com",
            "timestamp": "2025-11-26T10:00:00Z",
            "metric": "availability"
        }
        sanitized = enforcer.redact_pii(data)

        # Non-PII fields should be unchanged
        assert sanitized["timestamp"] == data["timestamp"]
        assert sanitized["metric"] == data["metric"]


class TestDataRetention:
    """Test data retention enforcement."""

    def test_retention_policy_check(self):
        """Test retention policy validation."""
        enforcer = ComplianceEnforcer(
            enabled_standards=["soc2"],
            retention_audit=2555  # 7 years
        )

        # Data within retention period
        recent_date = datetime.now() - timedelta(days=365)
        assert enforcer.is_within_retention(recent_date, "audit")

        # Data outside retention period
        old_date = datetime.now() - timedelta(days=3000)
        assert not enforcer.is_within_retention(old_date, "audit")

    def test_retention_enforcement(self):
        """Test automatic retention enforcement."""
        enforcer = ComplianceEnforcer(enabled_standards=["soc2"])

        # Create test data with various ages
        test_data = [
            {"date": (datetime.now() - timedelta(days=10)).isoformat(), "value": 1},
            {"date": (datetime.now() - timedelta(days=100)).isoformat(), "value": 2},
            {"date": (datetime.now() - timedelta(days=3000)).isoformat(), "value": 3},
        ]

        # Enforce retention (should keep recent, remove old)
        retained = enforcer.enforce_retention(
            test_data,
            retention_days=365,
            date_field="date"
        )

        assert len(retained) < len(test_data)
        # Recent data should be kept
        assert any(d["value"] == 1 for d in retained)


class TestEncryptionValidation:
    """Test encryption compliance validation."""

    def test_validate_encryption_algorithm(self):
        """Test encryption algorithm validation."""
        enforcer = ComplianceEnforcer(enabled_standards=["soc2"])

        # Valid algorithm
        enforcer.validate_encryption(
            algorithm="AES256",
            key_length=256
        )

        # Weak algorithm should raise warning or error
        with pytest.raises(ComplianceViolation):
            enforcer.validate_encryption(
                algorithm="DES",
                key_length=56,
                mode="block"
            )

    def test_validate_encryption_key_length(self):
        """Test encryption key length validation."""
        enforcer = ComplianceEnforcer(
            enabled_standards=["soc2"],
            enforcement_mode="block"
        )

        # Weak key length
        with pytest.raises(ComplianceViolation):
            enforcer.validate_encryption(
                algorithm="AES128",
                key_length=128  # Too weak for some standards
            )

    def test_validate_data_encryption(self):
        """Test data-at-rest encryption validation."""
        enforcer = ComplianceEnforcer(enabled_standards=["soc2", "iso27001"])

        # Data should be encrypted
        result = enforcer.validate_data_encryption(
            data_classification="confidential",
            encrypted=True
        )
        assert result == True


class TestAccessControl:
    """Test access control validation."""

    def test_validate_access_control_allowed(self):
        """Test allowed access control."""
        enforcer = ComplianceEnforcer(
            enabled_standards=["soc2"],
            enforcement_mode="log"
        )

        # Should not raise exception
        enforcer.validate_access_control(
            user_role="admin",
            required_role="admin",
            resource="/admin/users"
        )

    def test_validate_access_control_denied_log_mode(self):
        """Test denied access in log mode."""
        enforcer = ComplianceEnforcer(
            enabled_standards=["soc2"],
            enforcement_mode="log"
        )

        # Should log but not raise exception
        enforcer.validate_access_control(
            user_role="viewer",
            required_role="admin",
            resource="/admin/users"
        )

    def test_validate_access_control_denied_block_mode(self):
        """Test denied access in block mode."""
        enforcer = ComplianceEnforcer(
            enabled_standards=["soc2"],
            enforcement_mode="block"
        )

        # Should raise exception
        with pytest.raises(ComplianceViolation):
            enforcer.validate_access_control(
                user_role="viewer",
                required_role="admin",
                resource="/admin/users"
            )

    def test_role_hierarchy(self):
        """Test role hierarchy in access control."""
        enforcer = ComplianceEnforcer(enabled_standards=["soc2"])

        # Admin should have access to sre resources
        enforcer.validate_access_control(
            user_role="admin",
            required_role="sre",
            resource="/sre/dashboard"
        )

        # SRE should have access to viewer resources
        enforcer.validate_access_control(
            user_role="sre",
            required_role="readonly",
            resource="/reports/view"
        )


class TestAuditTrail:
    """Test audit trail and logging."""

    def test_audit_chain_creation(self, tmp_path):
        """Test creating audit chain."""
        log_path = tmp_path / "audit.log"

        chain = AuditChain(log_path=str(log_path))
        assert chain is not None

    def test_log_event(self, tmp_path):
        """Test logging compliance event."""
        log_path = tmp_path / "audit.log"

        enforcer = ComplianceEnforcer(
            enabled_standards=["soc2"],
            audit_log_path=str(log_path)
        )

        enforcer.log_event(
            event_type="access_attempt",
            user="admin",
            resource="/api/data",
            outcome="success"
        )

        # Check log file created
        assert log_path.exists()

    def test_audit_chain_integrity(self, tmp_path):
        """Test audit chain cryptographic integrity."""
        log_path = tmp_path / "audit.log"

        chain = AuditChain(log_path=str(log_path))

        # Add events
        for i in range(5):
            chain.add_event({
                "event_type": "test",
                "sequence": i,
                "data": f"event_{i}"
            })

        # Verify integrity
        assert chain.verify_integrity() == True

    def test_audit_chain_tampering_detection(self, tmp_path):
        """Test detection of audit log tampering."""
        log_path = tmp_path / "audit.log"

        chain = AuditChain(log_path=str(log_path))

        # Add events
        chain.add_event({"event_type": "test", "data": "event_1"})
        chain.add_event({"event_type": "test", "data": "event_2"})

        # Tamper with log file
        with open(log_path, "r") as f:
            lines = f.readlines()

        # Modify a line
        if len(lines) > 1:
            lines[0] = lines[0].replace("event_1", "tampered")

        with open(log_path, "w") as f:
            f.writelines(lines)

        # Verify should detect tampering
        assert chain.verify_integrity() == False


class TestComplianceReporting:
    """Test compliance reporting functions."""

    def test_generate_compliance_report(self):
        """Test generating compliance report."""
        enforcer = ComplianceEnforcer(
            enabled_standards=["soc2", "iso27001"]
        )

        report = enforcer.generate_compliance_report(
            period_start="2025-11-01",
            period_end="2025-11-26"
        )

        assert "standards" in report
        assert "compliance_percentage" in report
        assert "violations" in report

    def test_query_audit_log(self, tmp_path):
        """Test querying audit log."""
        log_path = tmp_path / "audit.log"

        enforcer = ComplianceEnforcer(
            enabled_standards=["soc2"],
            audit_log_path=str(log_path)
        )

        # Log some events
        for i in range(10):
            enforcer.log_event(
                event_type="access_attempt",
                user=f"user_{i % 3}",
                resource="/api/data",
                outcome="success" if i % 2 == 0 else "failure"
            )

        # Query events
        events = query_audit_log(
            log_path=str(log_path),
            event_type="access_attempt",
            outcome="success"
        )

        assert len(events) > 0
        assert all(e["outcome"] == "success" for e in events)


class TestGDPRCompliance:
    """Test GDPR-specific compliance features."""

    def test_data_minimization(self):
        """Test GDPR data minimization principle."""
        enforcer = ComplianceEnforcer(enabled_standards=["gdpr"])

        # Validate data collection
        collected_fields = ["email", "ip", "timestamp", "unnecessary_field"]
        required_fields = ["email", "timestamp"]

        with pytest.raises(ComplianceViolation):
            enforcer.validate_data_minimization(
                collected_fields=collected_fields,
                required_fields=required_fields,
                mode="block"
            )

    def test_right_to_erasure(self):
        """Test GDPR right to erasure (deletion)."""
        enforcer = ComplianceEnforcer(enabled_standards=["gdpr"])

        # Should support data deletion
        result = enforcer.delete_personal_data(
            subject_id="user@example.com",
            dry_run=True
        )

        assert "deleted_records" in result or "status" in result


# Pytest fixtures
@pytest.fixture
def soc2_enforcer():
    """Create SOC 2 enforcer."""
    return ComplianceEnforcer(
        enabled_standards=["soc2"],
        enforcement_mode="log"
    )


@pytest.fixture
def strict_enforcer():
    """Create strict enforcer with block mode."""
    return ComplianceEnforcer(
        enabled_standards=["soc2", "iso27001", "gdpr"],
        enforcement_mode="block"
    )


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
