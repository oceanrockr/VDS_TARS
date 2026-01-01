"""
Tests for Certificate Monitor

Covers:
- Certificate information extraction
- Remote certificate checking
- Local certificate file parsing
- Alert severity calculation
- Prometheus metrics generation
- Error handling

Usage:
    pytest tests/test_certificate_monitor.py -v
    pytest tests/test_certificate_monitor.py -v --cov=security.certificate_monitor
"""

import pytest
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from cryptography import x509
from cryptography.x509.oid import NameOID
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.backends import default_backend

from security.certificate_monitor import (
    CertificateMonitor,
    CertificateInfo,
    CertificateAlert,
    check_domain_certificate,
    check_certificate_file
)


# Test Fixtures
@pytest.fixture
def temp_dir():
    """Create temporary directory for test certificates."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_certificate():
    """Generate a sample X.509 certificate for testing."""
    # Generate private key
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
        backend=default_backend()
    )

    # Create certificate
    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
        x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "California"),
        x509.NameAttribute(NameOID.LOCALITY_NAME, "San Francisco"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Test Organization"),
        x509.NameAttribute(NameOID.COMMON_NAME, "test.example.com"),
    ])

    cert = x509.CertificateBuilder().subject_name(
        subject
    ).issuer_name(
        issuer
    ).public_key(
        private_key.public_key()
    ).serial_number(
        x509.random_serial_number()
    ).not_valid_before(
        datetime.now(timezone.utc) - timedelta(days=30)
    ).not_valid_after(
        datetime.now(timezone.utc) + timedelta(days=365)
    ).add_extension(
        x509.SubjectAlternativeName([
            x509.DNSName("test.example.com"),
        ]),
        critical=False,
    ).sign(private_key, hashes.SHA256(), default_backend())

    return cert


@pytest.fixture
def expired_certificate():
    """Generate an expired X.509 certificate for testing."""
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
        backend=default_backend()
    )

    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, "expired.example.com"),
    ])

    cert = x509.CertificateBuilder().subject_name(
        subject
    ).issuer_name(
        issuer
    ).public_key(
        private_key.public_key()
    ).serial_number(
        x509.random_serial_number()
    ).not_valid_before(
        datetime.now(timezone.utc) - timedelta(days=400)
    ).not_valid_after(
        datetime.now(timezone.utc) - timedelta(days=10)
    ).sign(private_key, hashes.SHA256(), default_backend())

    return cert


@pytest.fixture
def cert_file(temp_dir, sample_certificate):
    """Create a temporary certificate file."""
    cert_path = temp_dir / "test_cert.pem"
    pem_data = sample_certificate.public_bytes(serialization.Encoding.PEM)
    cert_path.write_bytes(pem_data)
    return cert_path


@pytest.fixture
def expired_cert_file(temp_dir, expired_certificate):
    """Create a temporary expired certificate file."""
    cert_path = temp_dir / "expired_cert.pem"
    pem_data = expired_certificate.public_bytes(serialization.Encoding.PEM)
    cert_path.write_bytes(pem_data)
    return cert_path


# Test CertificateInfo Dataclass
class TestCertificateInfo:
    """Test CertificateInfo dataclass."""

    def test_certificate_info_creation(self):
        """Test creating CertificateInfo object."""
        now = datetime.now(timezone.utc)
        expiry = now + timedelta(days=90)

        cert_info = CertificateInfo(
            domain="example.com",
            issuer="Let's Encrypt",
            subject="example.com",
            not_before=now - timedelta(days=30),
            not_after=expiry,
            serial_number="ABC123",
            days_until_expiry=90,
            is_expired=False
        )

        assert cert_info.domain == "example.com"
        assert cert_info.issuer == "Let's Encrypt"
        assert cert_info.days_until_expiry == 90
        assert not cert_info.is_expired


# Test CertificateAlert Dataclass
class TestCertificateAlert:
    """Test CertificateAlert dataclass."""

    def test_certificate_alert_creation(self):
        """Test creating CertificateAlert object."""
        expires_at = datetime.now(timezone.utc) + timedelta(days=5)

        alert = CertificateAlert(
            domain="example.com",
            severity="CRITICAL",
            message="Certificate expiring soon",
            days_remaining=5,
            expires_at=expires_at
        )

        assert alert.domain == "example.com"
        assert alert.severity == "CRITICAL"
        assert alert.days_remaining == 5


# Test CertificateMonitor
class TestCertificateMonitor:
    """Test CertificateMonitor class."""

    def test_monitor_initialization(self):
        """Test monitor initialization."""
        domains = ["example.com:443", "test.com:443"]
        files = [Path("/etc/ssl/cert1.pem"), Path("/etc/ssl/cert2.pem")]

        monitor = CertificateMonitor(
            monitored_domains=domains,
            monitored_files=files
        )

        assert monitor.monitored_domains == domains
        assert monitor.monitored_files == files
        assert len(monitor._certificate_cache) == 0

    def test_thresholds(self):
        """Test severity thresholds are correctly defined."""
        monitor = CertificateMonitor()

        assert monitor.THRESHOLDS["CRITICAL"] == 7
        assert monitor.THRESHOLDS["HIGH"] == 14
        assert monitor.THRESHOLDS["WARNING"] == 30

    def test_get_alert_severity(self):
        """Test alert severity calculation."""
        monitor = CertificateMonitor()

        assert monitor.get_alert_severity(5) == "CRITICAL"
        assert monitor.get_alert_severity(7) == "CRITICAL"
        assert monitor.get_alert_severity(10) == "HIGH"
        assert monitor.get_alert_severity(14) == "HIGH"
        assert monitor.get_alert_severity(20) == "WARNING"
        assert monitor.get_alert_severity(30) == "WARNING"
        assert monitor.get_alert_severity(45) == "INFO"
        assert monitor.get_alert_severity(365) == "INFO"

    def test_check_certificate_file_valid(self, cert_file, sample_certificate):
        """Test checking a valid certificate file."""
        monitor = CertificateMonitor()
        cert_info = monitor.check_certificate_file(cert_file)

        assert cert_info is not None
        assert cert_info.domain == str(cert_file)
        assert cert_info.subject == "test.example.com"
        assert cert_info.issuer == "test.example.com"
        assert not cert_info.is_expired
        assert cert_info.days_until_expiry > 0

    def test_check_certificate_file_expired(self, expired_cert_file):
        """Test checking an expired certificate file."""
        monitor = CertificateMonitor()
        cert_info = monitor.check_certificate_file(expired_cert_file)

        assert cert_info is not None
        assert cert_info.is_expired
        assert cert_info.days_until_expiry < 0

    def test_check_certificate_file_not_found(self, temp_dir):
        """Test checking non-existent certificate file."""
        monitor = CertificateMonitor()
        non_existent = temp_dir / "nonexistent.pem"

        cert_info = monitor.check_certificate_file(non_existent)
        assert cert_info is None

    def test_check_certificate_file_invalid_format(self, temp_dir):
        """Test checking invalid certificate file."""
        monitor = CertificateMonitor()
        invalid_file = temp_dir / "invalid.pem"
        invalid_file.write_text("This is not a valid certificate")

        cert_info = monitor.check_certificate_file(invalid_file)
        assert cert_info is None

    @patch('socket.create_connection')
    @patch('ssl.create_default_context')
    def test_check_certificate_success(self, mock_ssl_context, mock_socket, sample_certificate):
        """Test successful remote certificate check."""
        # Mock SSL socket
        mock_ssl_socket = MagicMock()
        mock_ssl_socket.getpeercert.return_value = sample_certificate.public_bytes(
            serialization.Encoding.DER
        )

        # Mock context and socket
        mock_context = MagicMock()
        mock_context.wrap_socket.return_value.__enter__.return_value = mock_ssl_socket
        mock_ssl_context.return_value = mock_context

        mock_sock = MagicMock()
        mock_socket.return_value.__enter__.return_value = mock_sock

        # Test
        monitor = CertificateMonitor()
        cert_info = monitor.check_certificate("example.com", 443)

        assert cert_info is not None
        assert cert_info.domain == "example.com:443"
        assert cert_info.subject == "test.example.com"

    @patch('socket.create_connection')
    def test_check_certificate_timeout(self, mock_socket):
        """Test certificate check with connection timeout."""
        import socket
        mock_socket.side_effect = socket.timeout("Connection timed out")

        monitor = CertificateMonitor()
        cert_info = monitor.check_certificate("timeout.example.com", 443)

        assert cert_info is None

    @patch('socket.create_connection')
    def test_check_certificate_dns_error(self, mock_socket):
        """Test certificate check with DNS resolution error."""
        import socket
        mock_socket.side_effect = socket.gaierror("Name or service not known")

        monitor = CertificateMonitor()
        cert_info = monitor.check_certificate("nonexistent.example.com", 443)

        assert cert_info is None

    def test_create_alert(self, sample_certificate):
        """Test alert creation from certificate info."""
        monitor = CertificateMonitor()

        now = datetime.now(timezone.utc)
        cert_info = CertificateInfo(
            domain="example.com:443",
            issuer="Test CA",
            subject="example.com",
            not_before=now - timedelta(days=30),
            not_after=now + timedelta(days=5),
            serial_number="ABC123",
            days_until_expiry=5,
            is_expired=False
        )

        alert = monitor._create_alert(cert_info)

        assert alert.domain == "example.com:443"
        assert alert.severity == "CRITICAL"
        assert alert.days_remaining == 5
        assert "expires in 5 days" in alert.message

    def test_create_alert_expired(self):
        """Test alert creation for expired certificate."""
        monitor = CertificateMonitor()

        now = datetime.now(timezone.utc)
        cert_info = CertificateInfo(
            domain="expired.example.com",
            issuer="Test CA",
            subject="expired.example.com",
            not_before=now - timedelta(days=400),
            not_after=now - timedelta(days=10),
            serial_number="XYZ789",
            days_until_expiry=-10,
            is_expired=True
        )

        alert = monitor._create_alert(cert_info)

        assert alert.severity == "CRITICAL"
        assert "EXPIRED" in alert.message
        assert "10 days ago" in alert.message

    def test_check_all_with_files(self, cert_file, expired_cert_file):
        """Test checking all monitored certificates."""
        monitor = CertificateMonitor(
            monitored_files=[cert_file, expired_cert_file]
        )

        alerts = monitor.check_all()

        assert len(alerts) == 2

        # Check that cache is populated
        assert len(monitor._certificate_cache) == 2

    def test_get_prometheus_metrics(self, cert_file):
        """Test Prometheus metrics generation."""
        monitor = CertificateMonitor(monitored_files=[cert_file])
        monitor.check_all()

        metrics = monitor.get_prometheus_metrics()

        assert "certificate_expiry_days" in metrics
        assert "certificate_expiry_timestamp" in metrics
        assert "certificate_expired" in metrics
        assert "certificate_severity" in metrics
        assert "# HELP" in metrics
        assert "# TYPE" in metrics

    def test_get_prometheus_metrics_empty(self):
        """Test Prometheus metrics with no certificates."""
        monitor = CertificateMonitor()
        metrics = monitor.get_prometheus_metrics()

        assert "# HELP certificate_expiry_days" in metrics
        assert "# TYPE certificate_expiry_days gauge" in metrics

    def test_get_certificate_info_from_cache(self, cert_file):
        """Test retrieving certificate info from cache."""
        monitor = CertificateMonitor(monitored_files=[cert_file])
        monitor.check_all()

        cached_info = monitor.get_certificate_info(str(cert_file))

        assert cached_info is not None
        assert cached_info.domain == str(cert_file)

    def test_clear_cache(self, cert_file):
        """Test clearing certificate cache."""
        monitor = CertificateMonitor(monitored_files=[cert_file])
        monitor.check_all()

        assert len(monitor._certificate_cache) > 0

        monitor.clear_cache()
        assert len(monitor._certificate_cache) == 0

    def test_extract_certificate_info(self, sample_certificate):
        """Test certificate information extraction."""
        monitor = CertificateMonitor()
        cert_info = monitor._extract_certificate_info(
            sample_certificate,
            "test.example.com"
        )

        assert cert_info.domain == "test.example.com"
        assert cert_info.subject == "test.example.com"
        assert cert_info.issuer == "test.example.com"
        assert cert_info.serial_number is not None
        assert isinstance(cert_info.not_before, datetime)
        assert isinstance(cert_info.not_after, datetime)


# Test Convenience Functions
class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_check_certificate_file_function(self, cert_file):
        """Test check_certificate_file convenience function."""
        cert_info = check_certificate_file(cert_file)

        assert cert_info is not None
        assert cert_info.domain == str(cert_file)

    @patch('socket.create_connection')
    @patch('ssl.create_default_context')
    def test_check_domain_certificate_function(self, mock_ssl_context, mock_socket, sample_certificate):
        """Test check_domain_certificate convenience function."""
        # Mock SSL socket
        mock_ssl_socket = MagicMock()
        mock_ssl_socket.getpeercert.return_value = sample_certificate.public_bytes(
            serialization.Encoding.DER
        )

        mock_context = MagicMock()
        mock_context.wrap_socket.return_value.__enter__.return_value = mock_ssl_socket
        mock_ssl_context.return_value = mock_context

        mock_sock = MagicMock()
        mock_socket.return_value.__enter__.return_value = mock_sock

        # Test
        cert_info = check_domain_certificate("example.com", 443)

        assert cert_info is not None
        assert cert_info.domain == "example.com:443"


# Integration Tests
@pytest.mark.integration
class TestCertificateMonitorIntegration:
    """Integration tests (require network access)."""

    def test_check_real_certificate(self):
        """Test checking a real certificate (google.com)."""
        monitor = CertificateMonitor()
        cert_info = monitor.check_certificate("google.com", 443)

        # Google should have a valid certificate
        if cert_info:  # Only assert if connection succeeded
            assert cert_info.domain == "google.com:443"
            assert not cert_info.is_expired
            assert cert_info.days_until_expiry > 0
            # Issuer should be non-empty (varies based on certificate rotation)
            assert len(cert_info.issuer) > 0

    def test_check_all_with_real_domains(self):
        """Test checking multiple real domains."""
        monitor = CertificateMonitor(
            monitored_domains=["google.com:443", "github.com:443"]
        )

        alerts = monitor.check_all()

        # Should get alerts for accessible domains
        # Note: This may fail in restricted network environments
        assert isinstance(alerts, list)
