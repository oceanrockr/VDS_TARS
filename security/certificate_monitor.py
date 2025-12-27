"""
TLS Certificate Expiration Monitoring Module for T.A.R.S. Security

Provides certificate monitoring capabilities:
- Remote TLS certificate validation via network connections
- Local certificate file inspection
- Expiration alerting with configurable thresholds
- Prometheus metrics integration

Features:
- Multi-level severity thresholds (CRITICAL, HIGH, WARNING)
- Comprehensive certificate information extraction
- Support for both network and file-based certificate checks
- Production-ready error handling and logging
"""

import ssl
import socket
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from cryptography import x509
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization


# Configure module logger
logger = logging.getLogger(__name__)


@dataclass
class CertificateInfo:
    """
    Certificate information container.

    Attributes:
        domain: Domain name or certificate path
        issuer: Certificate issuer (CA)
        subject: Certificate subject (owner)
        not_before: Certificate valid from date
        not_after: Certificate expiration date
        serial_number: Certificate serial number
        days_until_expiry: Days remaining until expiration (negative if expired)
        is_expired: Whether certificate is currently expired
    """
    domain: str
    issuer: str
    subject: str
    not_before: datetime
    not_after: datetime
    serial_number: str
    days_until_expiry: int
    is_expired: bool


@dataclass
class CertificateAlert:
    """
    Certificate expiration alert.

    Attributes:
        domain: Domain or certificate identifier
        severity: Alert severity level (CRITICAL, HIGH, WARNING, INFO)
        message: Human-readable alert message
        days_remaining: Days until certificate expires
        expires_at: Certificate expiration timestamp
    """
    domain: str
    severity: str
    message: str
    days_remaining: int
    expires_at: datetime


class CertificateMonitor:
    """
    TLS certificate expiration monitoring system.

    Monitors certificates from network endpoints and local files,
    providing alerting and metrics for expiration tracking.

    Severity Thresholds:
    - CRITICAL: <= 7 days
    - HIGH: <= 14 days
    - WARNING: <= 30 days
    - INFO: > 30 days
    """

    # Severity thresholds in days
    THRESHOLDS = {
        "CRITICAL": 7,
        "HIGH": 14,
        "WARNING": 30
    }

    def __init__(self, monitored_domains: Optional[List[str]] = None,
                 monitored_files: Optional[List[Path]] = None):
        """
        Initialize certificate monitor.

        Args:
            monitored_domains: List of domain names to monitor (format: "domain:port")
            monitored_files: List of certificate file paths to monitor
        """
        self.monitored_domains = monitored_domains or []
        self.monitored_files = monitored_files or []
        self._certificate_cache: Dict[str, CertificateInfo] = {}

        logger.info(
            f"CertificateMonitor initialized: {len(self.monitored_domains)} domains, "
            f"{len(self.monitored_files)} files"
        )

    def check_certificate(self, domain: str, port: int = 443) -> Optional[CertificateInfo]:
        """
        Check certificate from network endpoint.

        Connects to the specified domain and port via TLS to retrieve
        and analyze the server certificate.

        Args:
            domain: Domain name or IP address
            port: TLS port (default: 443)

        Returns:
            CertificateInfo object or None if check fails

        Example:
            >>> monitor = CertificateMonitor()
            >>> cert_info = monitor.check_certificate("example.com")
            >>> if cert_info and cert_info.days_until_expiry < 30:
            ...     print(f"Certificate expires in {cert_info.days_until_expiry} days")
        """
        try:
            logger.debug(f"Checking certificate for {domain}:{port}")

            # Create SSL context
            context = ssl.create_default_context()

            # Connect and retrieve certificate
            with socket.create_connection((domain, port), timeout=10) as sock:
                with context.wrap_socket(sock, server_hostname=domain) as ssock:
                    # Get DER-encoded certificate
                    der_cert = ssock.getpeercert(binary_form=True)

                    if not der_cert:
                        logger.error(f"No certificate received from {domain}:{port}")
                        return None

                    # Parse certificate using cryptography library
                    cert = x509.load_der_x509_certificate(der_cert, default_backend())

                    # Extract certificate information
                    cert_info = self._extract_certificate_info(cert, f"{domain}:{port}")

                    logger.info(
                        f"Certificate for {domain}:{port} expires in "
                        f"{cert_info.days_until_expiry} days"
                    )

                    return cert_info

        except socket.timeout:
            logger.error(f"Connection timeout for {domain}:{port}")
            return None
        except socket.gaierror as e:
            logger.error(f"DNS resolution failed for {domain}: {e}")
            return None
        except ssl.SSLError as e:
            logger.error(f"SSL error for {domain}:{port}: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to check certificate for {domain}:{port}: {e}")
            return None

    def check_certificate_file(self, cert_path: Path) -> Optional[CertificateInfo]:
        """
        Check certificate from local file.

        Reads and parses a PEM-encoded certificate file to extract
        certificate information and expiration status.

        Args:
            cert_path: Path to PEM-encoded certificate file

        Returns:
            CertificateInfo object or None if parsing fails

        Example:
            >>> monitor = CertificateMonitor()
            >>> cert_info = monitor.check_certificate_file(Path("/etc/ssl/certs/server.crt"))
            >>> if cert_info:
            ...     print(f"Certificate issuer: {cert_info.issuer}")
        """
        try:
            if not cert_path.exists():
                logger.error(f"Certificate file not found: {cert_path}")
                return None

            logger.debug(f"Checking certificate file: {cert_path}")

            # Read certificate file
            with open(cert_path, "rb") as f:
                cert_data = f.read()

            # Parse PEM certificate
            cert = x509.load_pem_x509_certificate(cert_data, default_backend())

            # Extract certificate information
            cert_info = self._extract_certificate_info(cert, str(cert_path))

            logger.info(
                f"Certificate file {cert_path.name} expires in "
                f"{cert_info.days_until_expiry} days"
            )

            return cert_info

        except ValueError as e:
            logger.error(f"Invalid certificate format in {cert_path}: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to check certificate file {cert_path}: {e}")
            return None

    def _extract_certificate_info(
        self,
        cert: x509.Certificate,
        identifier: str
    ) -> CertificateInfo:
        """
        Extract certificate information from x509 certificate object.

        Args:
            cert: x509.Certificate object
            identifier: Domain or file identifier

        Returns:
            CertificateInfo object
        """
        # Extract subject
        subject_attrs = cert.subject.get_attributes_for_oid(x509.oid.NameOID.COMMON_NAME)
        subject = subject_attrs[0].value if subject_attrs else "Unknown"

        # Extract issuer
        issuer_attrs = cert.issuer.get_attributes_for_oid(x509.oid.NameOID.COMMON_NAME)
        issuer = issuer_attrs[0].value if issuer_attrs else "Unknown"

        # Get validity dates (ensure timezone-aware)
        not_before = cert.not_valid_before_utc
        not_after = cert.not_valid_after_utc

        # Calculate days until expiry
        now = datetime.now(timezone.utc)
        days_remaining = (not_after - now).days
        is_expired = now > not_after

        # Get serial number
        serial_number = hex(cert.serial_number)[2:].upper()

        return CertificateInfo(
            domain=identifier,
            issuer=issuer,
            subject=subject,
            not_before=not_before,
            not_after=not_after,
            serial_number=serial_number,
            days_until_expiry=days_remaining,
            is_expired=is_expired
        )

    def get_alert_severity(self, days_remaining: int) -> str:
        """
        Determine alert severity based on days remaining.

        Args:
            days_remaining: Days until certificate expires

        Returns:
            Severity level: "CRITICAL", "HIGH", "WARNING", or "INFO"

        Example:
            >>> monitor = CertificateMonitor()
            >>> monitor.get_alert_severity(5)
            'CRITICAL'
            >>> monitor.get_alert_severity(20)
            'HIGH'
        """
        if days_remaining <= self.THRESHOLDS["CRITICAL"]:
            return "CRITICAL"
        elif days_remaining <= self.THRESHOLDS["HIGH"]:
            return "HIGH"
        elif days_remaining <= self.THRESHOLDS["WARNING"]:
            return "WARNING"
        else:
            return "INFO"

    def check_all(self) -> List[CertificateAlert]:
        """
        Check all monitored certificates and generate alerts.

        Checks all configured domain endpoints and certificate files,
        generating alerts for certificates approaching expiration.

        Returns:
            List of CertificateAlert objects

        Example:
            >>> monitor = CertificateMonitor(
            ...     monitored_domains=["example.com:443"],
            ...     monitored_files=[Path("/etc/ssl/server.crt")]
            ... )
            >>> alerts = monitor.check_all()
            >>> critical_alerts = [a for a in alerts if a.severity == "CRITICAL"]
        """
        alerts: List[CertificateAlert] = []

        # Check domain certificates
        for domain_spec in self.monitored_domains:
            try:
                # Parse domain:port
                if ":" in domain_spec:
                    domain, port_str = domain_spec.split(":", 1)
                    port = int(port_str)
                else:
                    domain = domain_spec
                    port = 443

                cert_info = self.check_certificate(domain, port)

                if cert_info:
                    alert = self._create_alert(cert_info)
                    alerts.append(alert)
                    self._certificate_cache[domain_spec] = cert_info

            except ValueError as e:
                logger.error(f"Invalid domain specification '{domain_spec}': {e}")
            except Exception as e:
                logger.error(f"Error checking domain {domain_spec}: {e}")

        # Check certificate files
        for cert_path in self.monitored_files:
            try:
                cert_info = self.check_certificate_file(cert_path)

                if cert_info:
                    alert = self._create_alert(cert_info)
                    alerts.append(alert)
                    self._certificate_cache[str(cert_path)] = cert_info

            except Exception as e:
                logger.error(f"Error checking certificate file {cert_path}: {e}")

        # Log summary
        severity_counts = {}
        for alert in alerts:
            severity_counts[alert.severity] = severity_counts.get(alert.severity, 0) + 1

        logger.info(
            f"Certificate check complete: {len(alerts)} certificates checked. "
            f"Severity breakdown: {severity_counts}"
        )

        return alerts

    def _create_alert(self, cert_info: CertificateInfo) -> CertificateAlert:
        """
        Create alert from certificate information.

        Args:
            cert_info: CertificateInfo object

        Returns:
            CertificateAlert object
        """
        severity = self.get_alert_severity(cert_info.days_until_expiry)

        # Generate message
        if cert_info.is_expired:
            message = (
                f"Certificate for {cert_info.domain} has EXPIRED "
                f"({abs(cert_info.days_until_expiry)} days ago)"
            )
        else:
            message = (
                f"Certificate for {cert_info.domain} expires in "
                f"{cert_info.days_until_expiry} days"
            )

        return CertificateAlert(
            domain=cert_info.domain,
            severity=severity,
            message=message,
            days_remaining=cert_info.days_until_expiry,
            expires_at=cert_info.not_after
        )

    def get_prometheus_metrics(self) -> str:
        """
        Generate Prometheus-formatted metrics for certificate monitoring.

        Returns metrics for:
        - certificate_expiry_days: Days until certificate expires
        - certificate_expiry_timestamp: Unix timestamp of expiration
        - certificate_expired: Boolean indicator (1 if expired, 0 otherwise)

        Returns:
            Prometheus metrics in text format

        Example:
            >>> monitor = CertificateMonitor()
            >>> monitor.check_certificate("example.com")
            >>> print(monitor.get_prometheus_metrics())
            # HELP certificate_expiry_days Days until certificate expires
            # TYPE certificate_expiry_days gauge
            certificate_expiry_days{domain="example.com:443"} 45
            ...
        """
        metrics = []

        # Metric definitions
        metrics.append("# HELP certificate_expiry_days Days until certificate expires")
        metrics.append("# TYPE certificate_expiry_days gauge")

        for identifier, cert_info in self._certificate_cache.items():
            # Escape domain for Prometheus labels
            safe_domain = identifier.replace('"', '\\"')
            metrics.append(
                f'certificate_expiry_days{{domain="{safe_domain}"}} '
                f'{cert_info.days_until_expiry}'
            )

        metrics.append("")
        metrics.append("# HELP certificate_expiry_timestamp Unix timestamp when certificate expires")
        metrics.append("# TYPE certificate_expiry_timestamp gauge")

        for identifier, cert_info in self._certificate_cache.items():
            safe_domain = identifier.replace('"', '\\"')
            expiry_timestamp = int(cert_info.not_after.timestamp())
            metrics.append(
                f'certificate_expiry_timestamp{{domain="{safe_domain}"}} '
                f'{expiry_timestamp}'
            )

        metrics.append("")
        metrics.append("# HELP certificate_expired Certificate expiration status (1 = expired, 0 = valid)")
        metrics.append("# TYPE certificate_expired gauge")

        for identifier, cert_info in self._certificate_cache.items():
            safe_domain = identifier.replace('"', '\\"')
            expired_value = 1 if cert_info.is_expired else 0
            metrics.append(
                f'certificate_expired{{domain="{safe_domain}"}} '
                f'{expired_value}'
            )

        metrics.append("")
        metrics.append("# HELP certificate_severity Alert severity level (0=INFO, 1=WARNING, 2=HIGH, 3=CRITICAL)")
        metrics.append("# TYPE certificate_severity gauge")

        severity_map = {"INFO": 0, "WARNING": 1, "HIGH": 2, "CRITICAL": 3}

        for identifier, cert_info in self._certificate_cache.items():
            safe_domain = identifier.replace('"', '\\"')
            severity = self.get_alert_severity(cert_info.days_until_expiry)
            severity_value = severity_map.get(severity, 0)
            metrics.append(
                f'certificate_severity{{domain="{safe_domain}",severity="{severity}"}} '
                f'{severity_value}'
            )

        return "\n".join(metrics)

    def get_certificate_info(self, identifier: str) -> Optional[CertificateInfo]:
        """
        Retrieve cached certificate information.

        Args:
            identifier: Domain spec (e.g., "example.com:443") or file path

        Returns:
            CertificateInfo object or None if not found
        """
        return self._certificate_cache.get(identifier)

    def clear_cache(self) -> None:
        """Clear the certificate information cache."""
        self._certificate_cache.clear()
        logger.debug("Certificate cache cleared")


# Convenience function for one-off certificate checks
def check_domain_certificate(domain: str, port: int = 443) -> Optional[CertificateInfo]:
    """
    Quick check of a single domain certificate.

    Args:
        domain: Domain name
        port: TLS port (default: 443)

    Returns:
        CertificateInfo object or None

    Example:
        >>> from security.certificate_monitor import check_domain_certificate
        >>> cert = check_domain_certificate("google.com")
        >>> if cert:
        ...     print(f"Expires in {cert.days_until_expiry} days")
    """
    monitor = CertificateMonitor()
    return monitor.check_certificate(domain, port)


def check_certificate_file(cert_path: Path) -> Optional[CertificateInfo]:
    """
    Quick check of a single certificate file.

    Args:
        cert_path: Path to certificate file

    Returns:
        CertificateInfo object or None

    Example:
        >>> from pathlib import Path
        >>> from security.certificate_monitor import check_certificate_file
        >>> cert = check_certificate_file(Path("/etc/ssl/server.crt"))
        >>> if cert:
        ...     print(f"Subject: {cert.subject}")
    """
    monitor = CertificateMonitor()
    return monitor.check_certificate_file(cert_path)
