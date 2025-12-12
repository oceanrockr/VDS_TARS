"""
Cryptographic Signing for Report Integrity

Provides RSA-based signing for retrospective and regression reports.
"""

from typing import Optional
from pathlib import Path
import hashlib
import base64
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding as asym_padding
from cryptography.hazmat.backends import default_backend
import json


class ReportSigner:
    """
    RSA-based signing for report integrity verification.

    Uses RSA-2048 with SHA-256 for signatures.
    """

    def __init__(
        self,
        private_key_path: Optional[Path] = None,
        public_key_path: Optional[Path] = None,
    ):
        """
        Initialize report signer.

        Args:
            private_key_path: Path to RSA private key (PEM format)
            public_key_path: Path to RSA public key (PEM format)
        """
        self.private_key = None
        self.public_key = None

        # Load or generate private key
        if private_key_path and private_key_path.exists():
            self.private_key = self._load_private_key(private_key_path)
        elif private_key_path:
            self.private_key = self._generate_key_pair(
                private_key_path,
                public_key_path or private_key_path.with_suffix(".pub")
            )

        # Load public key
        if public_key_path and public_key_path.exists():
            self.public_key = self._load_public_key(public_key_path)
        elif self.private_key:
            self.public_key = self.private_key.public_key()

    def sign_file(self, file_path: Path) -> str:
        """
        Sign a file and return base64-encoded signature.

        Args:
            file_path: Path to file to sign

        Returns:
            Base64-encoded signature
        """
        if not self.private_key:
            raise RuntimeError("No private key available for signing")

        with open(file_path, "rb") as f:
            data = f.read()

        signature = self.sign_data(data)
        return base64.b64encode(signature).decode("ascii")

    def sign_data(self, data: bytes) -> bytes:
        """
        Sign data with RSA private key.

        Args:
            data: Data to sign

        Returns:
            Raw signature bytes
        """
        if not self.private_key:
            raise RuntimeError("No private key available for signing")

        signature = self.private_key.sign(
            data,
            asym_padding.PSS(
                mgf=asym_padding.MGF1(hashes.SHA256()),
                salt_length=asym_padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )

        return signature

    def verify_file(self, file_path: Path, signature_b64: str) -> bool:
        """
        Verify file signature.

        Args:
            file_path: Path to file
            signature_b64: Base64-encoded signature

        Returns:
            True if signature is valid, False otherwise
        """
        if not self.public_key:
            raise RuntimeError("No public key available for verification")

        with open(file_path, "rb") as f:
            data = f.read()

        signature = base64.b64decode(signature_b64)
        return self.verify_data(data, signature)

    def verify_data(self, data: bytes, signature: bytes) -> bool:
        """
        Verify data signature.

        Args:
            data: Original data
            signature: Signature bytes

        Returns:
            True if signature is valid, False otherwise
        """
        if not self.public_key:
            raise RuntimeError("No public key available for verification")

        try:
            self.public_key.verify(
                signature,
                data,
                asym_padding.PSS(
                    mgf=asym_padding.MGF1(hashes.SHA256()),
                    salt_length=asym_padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return True
        except Exception:
            return False

    def sign_json_report(self, report: dict) -> dict:
        """
        Sign a JSON report and add signature field.

        Args:
            report: Report dictionary

        Returns:
            Report with signature field added
        """
        # Serialize report (canonical JSON)
        report_json = json.dumps(report, sort_keys=True, separators=(',', ':'))
        report_bytes = report_json.encode("utf-8")

        # Generate signature
        signature = self.sign_data(report_bytes)
        signature_b64 = base64.b64encode(signature).decode("ascii")

        # Calculate SHA-256 hash
        report_hash = hashlib.sha256(report_bytes).hexdigest()

        # Add signature metadata
        report_with_sig = report.copy()
        report_with_sig["_signature"] = {
            "algorithm": "RSA-PSS-SHA256",
            "signature": signature_b64,
            "sha256": report_hash,
        }

        return report_with_sig

    def verify_json_report(self, report: dict) -> bool:
        """
        Verify signature of a JSON report.

        Args:
            report: Report dictionary with _signature field

        Returns:
            True if signature is valid, False otherwise
        """
        if "_signature" not in report:
            raise ValueError("Report does not contain signature")

        # Extract signature metadata
        sig_metadata = report["_signature"]
        signature_b64 = sig_metadata["signature"]
        expected_hash = sig_metadata["sha256"]

        # Remove signature field for verification
        report_copy = {k: v for k, v in report.items() if k != "_signature"}

        # Serialize report (canonical JSON)
        report_json = json.dumps(report_copy, sort_keys=True, separators=(',', ':'))
        report_bytes = report_json.encode("utf-8")

        # Verify hash
        actual_hash = hashlib.sha256(report_bytes).hexdigest()
        if actual_hash != expected_hash:
            return False

        # Verify signature
        signature = base64.b64decode(signature_b64)
        return self.verify_data(report_bytes, signature)

    def _load_private_key(self, key_path: Path):
        """Load RSA private key from PEM file."""
        with open(key_path, "rb") as f:
            private_key = serialization.load_pem_private_key(
                f.read(),
                password=None,
                backend=default_backend()
            )
        return private_key

    def _load_public_key(self, key_path: Path):
        """Load RSA public key from PEM file."""
        with open(key_path, "rb") as f:
            public_key = serialization.load_pem_public_key(
                f.read(),
                backend=default_backend()
            )
        return public_key

    def _generate_key_pair(
        self,
        private_key_path: Path,
        public_key_path: Path,
    ):
        """Generate RSA-2048 key pair and save to files."""
        # Generate private key
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )

        # Save private key
        private_key_path.parent.mkdir(parents=True, exist_ok=True)
        with open(private_key_path, "wb") as f:
            f.write(private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            ))

        # Set restrictive permissions
        import os
        if hasattr(os, 'chmod'):
            os.chmod(private_key_path, 0o400)

        # Save public key
        public_key = private_key.public_key()
        with open(public_key_path, "wb") as f:
            f.write(public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            ))

        return private_key


def sign_markdown_report(
    report_path: Path,
    signature_path: Path,
    signer: ReportSigner,
):
    """
    Sign a Markdown report and save signature to separate file.

    Args:
        report_path: Path to Markdown report
        signature_path: Path to save signature file
        signer: ReportSigner instance
    """
    signature = signer.sign_file(report_path)

    signature_path.parent.mkdir(parents=True, exist_ok=True)
    with open(signature_path, "w") as f:
        f.write(f"RSA-PSS-SHA256\n{signature}\n")


def verify_markdown_report(
    report_path: Path,
    signature_path: Path,
    signer: ReportSigner,
) -> bool:
    """
    Verify Markdown report signature.

    Args:
        report_path: Path to Markdown report
        signature_path: Path to signature file
        signer: ReportSigner instance

    Returns:
        True if signature is valid, False otherwise
    """
    with open(signature_path, "r") as f:
        lines = f.readlines()
        signature_b64 = lines[1].strip()

    return signer.verify_file(report_path, signature_b64)
