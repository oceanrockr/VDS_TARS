#!/usr/bin/env python3
"""
Certificate Monitoring Demo for T.A.R.S. Security

Demonstrates usage of the CertificateMonitor module for:
- Checking remote TLS certificates
- Monitoring local certificate files
- Generating alerts for expiring certificates
- Exporting Prometheus metrics

Usage:
    python examples/certificate_monitoring_demo.py

    Or from project root:
    python -m examples.certificate_monitoring_demo
"""

import sys
import logging
from pathlib import Path

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from security import CertificateMonitor, check_domain_certificate

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def demo_single_domain_check():
    """Demonstrate checking a single domain certificate."""
    print("\n" + "=" * 60)
    print("Demo 1: Single Domain Certificate Check")
    print("=" * 60)

    # Check a well-known domain
    domain = "google.com"
    print(f"\nChecking certificate for {domain}...")

    cert_info = check_domain_certificate(domain)

    if cert_info:
        print(f"\n  Domain: {cert_info.domain}")
        print(f"  Subject: {cert_info.subject}")
        print(f"  Issuer: {cert_info.issuer}")
        print(f"  Valid From: {cert_info.not_before}")
        print(f"  Valid Until: {cert_info.not_after}")
        print(f"  Serial Number: {cert_info.serial_number}")
        print(f"  Days Until Expiry: {cert_info.days_until_expiry}")
        print(f"  Is Expired: {cert_info.is_expired}")
    else:
        print(f"  Failed to retrieve certificate for {domain}")


def demo_multiple_domain_monitoring():
    """Demonstrate monitoring multiple domains."""
    print("\n" + "=" * 60)
    print("Demo 2: Multiple Domain Monitoring with Alerts")
    print("=" * 60)

    # List of domains to monitor
    domains = [
        "google.com:443",
        "github.com:443",
        "python.org:443"
    ]

    print(f"\nMonitoring {len(domains)} domains:")
    for domain in domains:
        print(f"  - {domain}")

    # Create monitor
    monitor = CertificateMonitor(monitored_domains=domains)

    # Check all certificates and generate alerts
    alerts = monitor.check_all()

    print(f"\n{len(alerts)} certificates checked\n")

    # Display alerts by severity
    severity_groups = {}
    for alert in alerts:
        if alert.severity not in severity_groups:
            severity_groups[alert.severity] = []
        severity_groups[alert.severity].append(alert)

    for severity in ["CRITICAL", "HIGH", "WARNING", "INFO"]:
        if severity in severity_groups:
            print(f"\n{severity} Alerts ({len(severity_groups[severity])}):")
            for alert in severity_groups[severity]:
                print(f"  - {alert.message}")


def demo_severity_thresholds():
    """Demonstrate alert severity calculation."""
    print("\n" + "=" * 60)
    print("Demo 3: Alert Severity Thresholds")
    print("=" * 60)

    monitor = CertificateMonitor()

    test_cases = [
        (3, "CRITICAL"),
        (7, "CRITICAL"),
        (10, "HIGH"),
        (14, "HIGH"),
        (20, "WARNING"),
        (30, "WARNING"),
        (45, "INFO"),
        (365, "INFO")
    ]

    print("\nThresholds:")
    print(f"  CRITICAL: <= {monitor.THRESHOLDS['CRITICAL']} days")
    print(f"  HIGH:     <= {monitor.THRESHOLDS['HIGH']} days")
    print(f"  WARNING:  <= {monitor.THRESHOLDS['WARNING']} days")
    print(f"  INFO:     > {monitor.THRESHOLDS['WARNING']} days")

    print("\nSeverity Calculations:")
    for days, expected_severity in test_cases:
        severity = monitor.get_alert_severity(days)
        status = "[OK]" if severity == expected_severity else "[FAIL]"
        print(f"  {status} {days:3d} days remaining -> {severity}")


def demo_prometheus_metrics():
    """Demonstrate Prometheus metrics generation."""
    print("\n" + "=" * 60)
    print("Demo 4: Prometheus Metrics Export")
    print("=" * 60)

    # Monitor a few domains
    domains = ["google.com:443", "github.com:443"]

    monitor = CertificateMonitor(monitored_domains=domains)
    monitor.check_all()

    # Generate Prometheus metrics
    metrics = monitor.get_prometheus_metrics()

    print("\nGenerated Prometheus Metrics:")
    print("-" * 60)
    print(metrics)


def demo_certificate_file_check():
    """Demonstrate checking local certificate files."""
    print("\n" + "=" * 60)
    print("Demo 5: Local Certificate File Monitoring")
    print("=" * 60)

    print("\nNote: This demo requires certificate files to exist.")
    print("Common certificate locations:")
    print("  - /etc/ssl/certs/")
    print("  - /etc/pki/tls/certs/")
    print("  - C:\\ProgramData\\ssl\\certs\\ (Windows)")

    # Example certificate paths (may not exist on all systems)
    example_paths = [
        "/etc/ssl/certs/ca-certificates.crt",
        "/etc/ssl/cert.pem",
    ]

    print("\nAttempting to check example certificate files...")
    for cert_path in example_paths:
        path = Path(cert_path)
        if path.exists():
            monitor = CertificateMonitor()
            cert_info = monitor.check_certificate_file(path)
            if cert_info:
                print(f"\n  [OK] {cert_path}")
                print(f"    Days until expiry: {cert_info.days_until_expiry}")
        else:
            print(f"\n  - {cert_path} (not found)")


def demo_error_handling():
    """Demonstrate error handling capabilities."""
    print("\n" + "=" * 60)
    print("Demo 6: Error Handling")
    print("=" * 60)

    monitor = CertificateMonitor()

    test_cases = [
        ("nonexistent-domain-12345.com", 443, "Non-existent domain"),
        ("localhost", 9999, "Invalid port"),
    ]

    print("\nTesting error scenarios:")
    for domain, port, description in test_cases:
        print(f"\n  {description}: {domain}:{port}")
        cert_info = monitor.check_certificate(domain, port)
        if cert_info is None:
            print(f"    [OK] Handled gracefully (returned None)")
        else:
            print(f"    Unexpected success")


def demo_cache_functionality():
    """Demonstrate certificate cache functionality."""
    print("\n" + "=" * 60)
    print("Demo 7: Certificate Cache")
    print("=" * 60)

    domains = ["google.com:443"]
    monitor = CertificateMonitor(monitored_domains=domains)

    print("\nChecking certificates...")
    monitor.check_all()

    print(f"Cache size: {len(monitor._certificate_cache)} certificates")

    # Retrieve from cache
    cached_cert = monitor.get_certificate_info("google.com:443")
    if cached_cert:
        print(f"\nRetrieved from cache:")
        print(f"  Domain: {cached_cert.domain}")
        print(f"  Days until expiry: {cached_cert.days_until_expiry}")

    # Clear cache
    monitor.clear_cache()
    print(f"\nCache cleared. New size: {len(monitor._certificate_cache)}")


def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("T.A.R.S. Certificate Monitoring System - Demo")
    print("=" * 60)

    demos = [
        ("Single Domain Check", demo_single_domain_check),
        ("Multiple Domain Monitoring", demo_multiple_domain_monitoring),
        ("Severity Thresholds", demo_severity_thresholds),
        ("Prometheus Metrics", demo_prometheus_metrics),
        ("Certificate File Check", demo_certificate_file_check),
        ("Error Handling", demo_error_handling),
        ("Cache Functionality", demo_cache_functionality),
    ]

    for i, (name, demo_func) in enumerate(demos, 1):
        try:
            demo_func()
        except Exception as e:
            print(f"\nError in demo '{name}': {e}")
            logger.exception(f"Demo {i} failed")

    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)
    print("\nFor more information, see:")
    print("  - security/certificate_monitor.py")
    print("  - tests/test_certificate_monitor.py")
    print("\n")


if __name__ == "__main__":
    main()
