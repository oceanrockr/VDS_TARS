#!/usr/bin/env python3
"""
T.A.R.S. Certificate Generation Script
Generate self-signed certificates for TLS and mTLS
Phase 11.5
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

try:
    from cryptography import x509
    from cryptography.x509.oid import NameOID, ExtendedKeyUsageOID
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.hazmat.backends import default_backend
except ImportError:
    print("Error: cryptography package not found")
    print("Install with: pip install cryptography")
    sys.exit(1)


def generate_private_key():
    """Generate RSA private key"""
    return rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
        backend=default_backend()
    )


def generate_ca_certificate(
    common_name: str = "T.A.R.S. Root CA",
    validity_days: int = 3650
):
    """
    Generate a Certificate Authority (CA) certificate

    Args:
        common_name: Common name for the CA
        validity_days: Certificate validity in days

    Returns:
        Tuple of (private_key, certificate)
    """
    # Generate private key
    private_key = generate_private_key()

    # Build certificate
    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
        x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "California"),
        x509.NameAttribute(NameOID.LOCALITY_NAME, "San Francisco"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "T.A.R.S."),
        x509.NameAttribute(NameOID.COMMON_NAME, common_name),
    ])

    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(private_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.utcnow())
        .not_valid_after(datetime.utcnow() + timedelta(days=validity_days))
        .add_extension(
            x509.BasicConstraints(ca=True, path_length=None),
            critical=True,
        )
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                key_cert_sign=True,
                crl_sign=True,
                key_encipherment=False,
                content_commitment=False,
                data_encipherment=False,
                key_agreement=False,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .sign(private_key, hashes.SHA256(), default_backend())
    )

    return private_key, cert


def generate_server_certificate(
    common_name: str,
    ca_key,
    ca_cert,
    dns_names: list = None,
    validity_days: int = 365
):
    """
    Generate a server certificate signed by CA

    Args:
        common_name: Common name (e.g., "orchestration-service")
        ca_key: CA private key
        ca_cert: CA certificate
        dns_names: List of DNS names (SANs)
        validity_days: Certificate validity in days

    Returns:
        Tuple of (private_key, certificate)
    """
    # Generate private key
    private_key = generate_private_key()

    # Build subject
    subject = x509.Name([
        x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
        x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "California"),
        x509.NameAttribute(NameOID.LOCALITY_NAME, "San Francisco"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "T.A.R.S."),
        x509.NameAttribute(NameOID.COMMON_NAME, common_name),
    ])

    # Add Subject Alternative Names (SANs)
    san_list = [x509.DNSName(common_name)]
    if dns_names:
        san_list.extend([x509.DNSName(name) for name in dns_names])
    san_list.append(x509.DNSName("localhost"))
    san_list.append(x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")))

    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(ca_cert.subject)
        .public_key(private_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.utcnow())
        .not_valid_after(datetime.utcnow() + timedelta(days=validity_days))
        .add_extension(
            x509.BasicConstraints(ca=False, path_length=None),
            critical=True,
        )
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                key_encipherment=True,
                key_cert_sign=False,
                crl_sign=False,
                content_commitment=False,
                data_encipherment=False,
                key_agreement=False,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(
            x509.ExtendedKeyUsage([
                ExtendedKeyUsageOID.SERVER_AUTH,
                ExtendedKeyUsageOID.CLIENT_AUTH,
            ]),
            critical=True,
        )
        .add_extension(
            x509.SubjectAlternativeName(san_list),
            critical=False,
        )
        .sign(ca_key, hashes.SHA256(), default_backend())
    )

    return private_key, cert


def save_key_and_cert(private_key, cert, name: str, output_dir: Path):
    """Save private key and certificate to files"""
    # Save private key
    key_path = output_dir / f"{name}.key"
    with open(key_path, "wb") as f:
        f.write(
            private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL,
                encryption_algorithm=serialization.NoEncryption(),
            )
        )
    print(f"✓ Generated private key: {key_path}")

    # Save certificate
    cert_path = output_dir / f"{name}.crt"
    with open(cert_path, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))
    print(f"✓ Generated certificate: {cert_path}")


def generate_all_certificates(output_dir: str = "./certs"):
    """Generate all certificates for T.A.R.S. services"""
    import ipaddress

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("T.A.R.S. Certificate Generation")
    print("=" * 60)

    # Step 1: Generate CA
    print("\n[1/6] Generating Certificate Authority (CA)...")
    ca_key, ca_cert = generate_ca_certificate()
    save_key_and_cert(ca_key, ca_cert, "ca", output_path)

    # Step 2: Generate Orchestration Service Certificate
    print("\n[2/6] Generating Orchestration Service certificate...")
    orch_key, orch_cert = generate_server_certificate(
        common_name="orchestration-service",
        ca_key=ca_key,
        ca_cert=ca_cert,
        dns_names=[
            "orchestration-service.tars.svc.cluster.local",
            "orchestration-service",
            "orchestration"
        ]
    )
    save_key_and_cert(orch_key, orch_cert, "orchestration", output_path)

    # Step 3: Generate AutoML Service Certificate
    print("\n[3/6] Generating AutoML Service certificate...")
    automl_key, automl_cert = generate_server_certificate(
        common_name="automl-service",
        ca_key=ca_key,
        ca_cert=ca_cert,
        dns_names=[
            "automl-service.tars.svc.cluster.local",
            "automl-service",
            "automl"
        ]
    )
    save_key_and_cert(automl_key, automl_cert, "automl", output_path)

    # Step 4: Generate HyperSync Service Certificate
    print("\n[4/6] Generating HyperSync Service certificate...")
    hypersync_key, hypersync_cert = generate_server_certificate(
        common_name="hypersync-service",
        ca_key=ca_key,
        ca_cert=ca_cert,
        dns_names=[
            "hypersync-service.tars.svc.cluster.local",
            "hypersync-service",
            "hypersync"
        ]
    )
    save_key_and_cert(hypersync_key, hypersync_cert, "hypersync", output_path)

    # Step 5: Generate Dashboard API Certificate
    print("\n[5/6] Generating Dashboard API certificate...")
    dashboard_key, dashboard_cert = generate_server_certificate(
        common_name="dashboard-api",
        ca_key=ca_key,
        ca_cert=ca_cert,
        dns_names=[
            "dashboard-api.tars.svc.cluster.local",
            "dashboard-api",
            "dashboard"
        ]
    )
    save_key_and_cert(dashboard_key, dashboard_cert, "dashboard-api", output_path)

    # Step 6: Generate Ingress/Gateway Certificate
    print("\n[6/6] Generating Ingress/Gateway certificate...")
    ingress_key, ingress_cert = generate_server_certificate(
        common_name="tars-ingress",
        ca_key=ca_key,
        ca_cert=ca_cert,
        dns_names=[
            "tars.local",
            "*.tars.local",
            "tars.example.com"
        ]
    )
    save_key_and_cert(ingress_key, ingress_cert, "ingress", output_path)

    print("\n" + "=" * 60)
    print("✅ Certificate generation complete!")
    print("=" * 60)
    print(f"\nCertificates saved to: {output_path.absolute()}")
    print("\nFiles generated:")
    print("  - ca.key, ca.crt (Certificate Authority)")
    print("  - orchestration.key, orchestration.crt")
    print("  - automl.key, automl.crt")
    print("  - hypersync.key, hypersync.crt")
    print("  - dashboard-api.key, dashboard-api.crt")
    print("  - ingress.key, ingress.crt")
    print("\nTo use these certificates:")
    print("  1. For local testing: Copy certs to service directories")
    print("  2. For Kubernetes: Create secrets from cert files")
    print("     kubectl create secret tls <name>-tls --cert=<name>.crt --key=<name>.key")
    print("\nFor production, use cert-manager with Let's Encrypt!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate TLS certificates for T.A.R.S. services"
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="./certs",
        help="Output directory for certificates (default: ./certs)"
    )

    args = parser.parse_args()

    generate_all_certificates(args.output_dir)
