"""
Security Hardening Module for T.A.R.S. Enterprise

Production-grade security components for enterprise deployments:

Cryptography:
- AES-256-GCM encryption for data at rest
- RSA-PSS cryptographic signing for integrity
- PGP encryption support

Supply Chain Security:
- SBOM generation (CycloneDX 1.5, SPDX 2.3)
- SLSA provenance (Level 2-3)
- Cryptographic signing for artifacts

Features:
- Full CycloneDX and SPDX support with license detection
- Complete SLSA v1.0 provenance with material tracking
- SHA-256 hashing for all artifacts
- PURL and CPE identifier generation
- Deterministic output for reproducibility
"""
from typing import Optional, Any
from pathlib import Path

# Encryption modules
from .encryption import AESEncryption, PGPEncryption

# Signing module
from .signing import ReportSigner

# Supply chain security - production generators
from .sbom_generator import SBOMGenerator, generate_sbom_for_tars, Dependency
from .slsa_generator import SLSAProvenanceGenerator, generate_slsa_provenance_for_tars

# Certificate monitoring
from .certificate_monitor import (
    CertificateMonitor,
    CertificateInfo,
    CertificateAlert,
    check_domain_certificate,
    check_certificate_file
)


class SecurityError(Exception):
    """
    Base exception for security-related errors.

    Includes:
    - Encryption/decryption failures
    - Signing failures
    - Key management errors
    - SBOM generation errors
    """

    def __init__(self, message: str, code: str = None):
        self.code = code
        super().__init__(message)


class SecurityManager:
    """
    Unified security manager for encryption, signing, and SBOM generation.

    Provides a single interface for all security operations.
    """

    def __init__(
        self,
        aes_key_path: Optional[Path] = None,
        rsa_key_path: Optional[Path] = None,
        pgp_key_id: Optional[str] = None,
    ):
        """
        Initialize security manager.

        Args:
            aes_key_path: Path to AES key file
            rsa_key_path: Path to RSA private key for signing
            pgp_key_id: PGP key ID for encryption
        """
        self.aes_key_path = aes_key_path
        self.rsa_key_path = rsa_key_path
        self.pgp_key_id = pgp_key_id

        # Initialize components lazily
        self._aes_encryptor = None
        self._pgp_encryptor = None
        self._signer = None

    @property
    def aes_encryptor(self) -> AESEncryption:
        """Get or create AES encryptor."""
        if self._aes_encryptor is None:
            self._aes_encryptor = AESEncryption(key_path=self.aes_key_path)
        return self._aes_encryptor

    @property
    def signer(self) -> ReportSigner:
        """Get or create report signer."""
        if self._signer is None:
            self._signer = ReportSigner(private_key_path=self.rsa_key_path)
        return self._signer

    def encrypt_data(self, data: str) -> str:
        """
        Encrypt data using AES-256-GCM.

        Args:
            data: Plaintext data to encrypt

        Returns:
            Base64-encoded ciphertext
        """
        try:
            return self.aes_encryptor.encrypt(data)
        except Exception as e:
            raise SecurityError(f"Encryption failed: {e}", code="ENCRYPT_FAILED")

    def decrypt_data(self, ciphertext: str) -> str:
        """
        Decrypt data using AES-256-GCM.

        Args:
            ciphertext: Base64-encoded ciphertext

        Returns:
            Decrypted plaintext
        """
        try:
            return self.aes_encryptor.decrypt(ciphertext)
        except Exception as e:
            raise SecurityError(f"Decryption failed: {e}", code="DECRYPT_FAILED")

    def sign_data(self, data: str) -> str:
        """
        Sign data using RSA-PSS.

        Args:
            data: Data to sign

        Returns:
            Base64-encoded signature
        """
        try:
            return self.signer.sign(data)
        except Exception as e:
            raise SecurityError(f"Signing failed: {e}", code="SIGN_FAILED")

    def verify_signature(self, data: str, signature: str) -> bool:
        """
        Verify RSA-PSS signature.

        Args:
            data: Original data
            signature: Base64-encoded signature

        Returns:
            True if signature is valid
        """
        try:
            return self.signer.verify(data, signature)
        except Exception as e:
            raise SecurityError(f"Verification failed: {e}", code="VERIFY_FAILED")


# Convenience aliases for backward compatibility
def generate_sbom(project_root: Path = Path("."), **kwargs) -> dict:
    """Generate SBOM for the project."""
    return generate_sbom_for_tars(project_root, **kwargs)


def generate_slsa_provenance(project_root: Path = Path("."), **kwargs) -> dict:
    """Generate SLSA provenance for the project."""
    return generate_slsa_provenance_for_tars(project_root, **kwargs)


# Public API
__all__ = [
    # Encryption
    "AESEncryption",
    "PGPEncryption",

    # Signing
    "ReportSigner",

    # SBOM
    "SBOMGenerator",
    "generate_sbom_for_tars",
    "generate_sbom",
    "Dependency",

    # SLSA
    "SLSAProvenanceGenerator",
    "generate_slsa_provenance_for_tars",
    "generate_slsa_provenance",

    # Certificate Monitoring
    "CertificateMonitor",
    "CertificateInfo",
    "CertificateAlert",
    "check_domain_certificate",
    "check_certificate_file",

    # Manager
    "SecurityManager",
    "SecurityError",
]

__version__ = "1.0.2"
__author__ = "Veleron Dev Studios"
__license__ = "Proprietary"
