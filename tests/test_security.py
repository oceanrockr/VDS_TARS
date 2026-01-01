"""
Tests for Security Manager

Covers:
- AES encryption/decryption roundtrip
- PGP encryption (if gpg available)
- RSA signing and verification
- JSON report signing
- SBOM generation
- SLSA provenance generation

Usage:
    pytest tests/test_security.py -v
    pytest tests/test_security.py -v --cov=security
"""

import pytest
import json
import shutil
import tempfile
from pathlib import Path

from security import (
    SecurityManager,
    generate_sbom,
    generate_slsa_provenance,
    SecurityError
)


def _is_gpg_available() -> bool:
    """Check if GPG is available on the system."""
    return shutil.which("gpg") is not None


class TestAESEncryption:
    """Test AES-256 encryption and decryption."""

    def test_aes_encryption_roundtrip(self, security_manager_aes):
        """Test encrypt and decrypt with AES."""
        plaintext = "Sensitive data that needs encryption"

        # Encrypt
        ciphertext = security_manager_aes.encrypt_data(plaintext)
        assert ciphertext != plaintext
        assert len(ciphertext) > 0

        # Decrypt
        decrypted = security_manager_aes.decrypt_data(ciphertext)
        assert decrypted == plaintext

    def test_aes_encryption_different_outputs(self, security_manager_aes):
        """Test that same plaintext produces different ciphertexts (IV)."""
        plaintext = "Test data"

        ciphertext1 = security_manager_aes.encrypt_data(plaintext)
        ciphertext2 = security_manager_aes.encrypt_data(plaintext)

        # Should be different due to random IV
        assert ciphertext1 != ciphertext2

        # But both should decrypt to same plaintext
        assert security_manager_aes.decrypt_data(ciphertext1) == plaintext
        assert security_manager_aes.decrypt_data(ciphertext2) == plaintext

    def test_aes_encrypt_empty_string(self, security_manager_aes):
        """Test encrypting empty string."""
        plaintext = ""

        ciphertext = security_manager_aes.encrypt_data(plaintext)
        decrypted = security_manager_aes.decrypt_data(ciphertext)

        assert decrypted == plaintext

    def test_aes_encrypt_large_data(self, security_manager_aes):
        """Test encrypting large data."""
        plaintext = "x" * 1_000_000  # 1MB of data

        ciphertext = security_manager_aes.encrypt_data(plaintext)
        decrypted = security_manager_aes.decrypt_data(ciphertext)

        assert decrypted == plaintext

    def test_aes_decrypt_invalid_ciphertext(self, security_manager_aes):
        """Test decrypting invalid ciphertext."""
        with pytest.raises((SecurityError, Exception)):
            security_manager_aes.decrypt_data("invalid_ciphertext")


class TestAESFileEncryption:
    """Test AES file encryption."""

    def test_file_encryption_roundtrip(self, security_manager_aes, tmp_path):
        """Test encrypting and decrypting a file."""
        # Create test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("Confidential file content")

        encrypted_file = tmp_path / "test.txt.enc"
        decrypted_file = tmp_path / "test_decrypted.txt"

        # Encrypt
        security_manager_aes.encrypt_file(
            input_path=str(test_file),
            output_path=str(encrypted_file)
        )

        assert encrypted_file.exists()
        assert encrypted_file.read_bytes() != test_file.read_bytes()

        # Decrypt
        security_manager_aes.decrypt_file(
            input_path=str(encrypted_file),
            output_path=str(decrypted_file)
        )

        assert decrypted_file.exists()
        assert decrypted_file.read_text() == test_file.read_text()

    def test_file_encryption_large_file(self, security_manager_aes, tmp_path):
        """Test encrypting large file."""
        # Create 10MB test file
        test_file = tmp_path / "large.bin"
        test_file.write_bytes(b"x" * 10_000_000)

        encrypted_file = tmp_path / "large.bin.enc"

        security_manager_aes.encrypt_file(
            input_path=str(test_file),
            output_path=str(encrypted_file)
        )

        assert encrypted_file.exists()
        assert encrypted_file.stat().st_size > 0


class TestRSASigning:
    """Test RSA-PSS signing and verification."""

    def test_rsa_sign_and_verify(self, security_manager_rsa):
        """Test signing and verifying data."""
        data = {"message": "Important data", "timestamp": "2025-11-26"}

        # Sign
        signed = security_manager_rsa.sign_json(data)

        assert "signature" in signed
        assert "signed_at" in signed
        assert signed["data"] == data

        # Verify
        is_valid = security_manager_rsa.verify_signature(signed)
        assert is_valid == True

    def test_rsa_verify_tampered_data(self, security_manager_rsa):
        """Test verification fails for tampered data."""
        data = {"message": "Important data"}

        # Sign
        signed = security_manager_rsa.sign_json(data)

        # Tamper with data
        signed["data"]["message"] = "Tampered data"

        # Verify should fail
        is_valid = security_manager_rsa.verify_signature(signed)
        assert is_valid == False

    def test_rsa_verify_tampered_signature(self, security_manager_rsa):
        """Test verification fails for tampered signature."""
        data = {"message": "Important data"}

        # Sign
        signed = security_manager_rsa.sign_json(data)

        # Tamper with signature
        signed["signature"] = "tampered_signature"

        # Verify should fail
        with pytest.raises((SecurityError, Exception)):
            security_manager_rsa.verify_signature(signed)

    def test_rsa_sign_complex_json(self, security_manager_rsa):
        """Test signing complex JSON structure."""
        data = {
            "report": {
                "metrics": [
                    {"name": "availability", "value": 99.95},
                    {"name": "latency", "value": 45.3}
                ],
                "metadata": {
                    "generated_at": "2025-11-26T10:00:00Z",
                    "version": "1.0.2"
                }
            }
        }

        signed = security_manager_rsa.sign_json(data)
        is_valid = security_manager_rsa.verify_signature(signed)

        assert is_valid == True


class TestPGPEncryption:
    """Test PGP encryption (if available)."""

    @pytest.mark.skipif(
        not _is_gpg_available(),
        reason="GPG not available"
    )
    def test_pgp_encryption(self, security_manager_pgp):
        """Test PGP encryption."""
        plaintext = "PGP encrypted message"

        ciphertext = security_manager_pgp.pgp_encrypt(
            plaintext,
            recipient="test@example.com"
        )

        assert ciphertext != plaintext
        assert "BEGIN PGP MESSAGE" in ciphertext

    @pytest.mark.skipif(
        not _is_gpg_available(),
        reason="GPG not available"
    )
    def test_pgp_decryption(self, security_manager_pgp):
        """Test PGP decryption."""
        plaintext = "PGP encrypted message"

        ciphertext = security_manager_pgp.pgp_encrypt(plaintext)
        decrypted = security_manager_pgp.pgp_decrypt(ciphertext)

        assert decrypted == plaintext


class TestSBOMGeneration:
    """Test SBOM generation."""

    def test_generate_cyclonedx_sbom(self, tmp_path):
        """Test generating CycloneDX SBOM."""
        output_path = tmp_path / "sbom.json"

        sbom = generate_sbom(
            format="cyclonedx",
            output_path=str(output_path)
        )

        assert output_path.exists()

        # Validate SBOM structure
        sbom_data = json.loads(output_path.read_text())
        assert "bomFormat" in sbom_data
        assert sbom_data["bomFormat"] == "CycloneDX"
        assert "components" in sbom_data

    def test_generate_spdx_sbom(self, tmp_path):
        """Test generating SPDX SBOM."""
        output_path = tmp_path / "sbom.spdx.json"

        sbom = generate_sbom(
            format="spdx",
            output_path=str(output_path)
        )

        assert output_path.exists()

        # Validate SBOM structure
        sbom_data = json.loads(output_path.read_text())
        assert "spdxVersion" in sbom_data or "SPDXID" in sbom_data

    def test_sbom_includes_dependencies(self, tmp_path):
        """Test SBOM includes Python dependencies."""
        output_path = tmp_path / "sbom.json"

        generate_sbom(format="cyclonedx", output_path=str(output_path))

        sbom_data = json.loads(output_path.read_text())
        components = sbom_data.get("components", [])

        # Should include common dependencies
        component_names = [c["name"] for c in components]
        assert len(component_names) > 0
        # Check for common T.A.R.S. dependencies
        assert any("fastapi" in name.lower() for name in component_names) or \
               any("pydantic" in name.lower() for name in component_names)


class TestSLSAProvenance:
    """Test SLSA provenance generation."""

    def test_generate_slsa_provenance(self, tmp_path):
        """Test generating SLSA provenance."""
        # Create test artifact
        artifact_path = tmp_path / "artifact.tar.gz"
        artifact_path.write_bytes(b"test artifact content")

        output_path = tmp_path / "provenance.json"

        provenance = generate_slsa_provenance(
            artifact_path=str(artifact_path),
            builder="GitHub Actions",
            build_type="automated_build",
            output_path=str(output_path)
        )

        assert output_path.exists()

        # Validate provenance structure
        prov_data = json.loads(output_path.read_text())
        assert "_type" in prov_data
        assert "subject" in prov_data
        assert "predicate" in prov_data

    def test_slsa_provenance_includes_digest(self, tmp_path):
        """Test SLSA provenance includes artifact digest."""
        artifact_path = tmp_path / "artifact.tar.gz"
        artifact_path.write_bytes(b"test content")

        output_path = tmp_path / "provenance.json"

        generate_slsa_provenance(
            artifact_path=str(artifact_path),
            builder="test",
            build_type="test",
            output_path=str(output_path)
        )

        prov_data = json.loads(output_path.read_text())
        subject = prov_data["subject"][0]

        assert "digest" in subject
        assert "sha256" in subject["digest"]
        assert len(subject["digest"]["sha256"]) == 64  # SHA256 hex length

    def test_slsa_provenance_level_3(self, tmp_path):
        """Test SLSA Level 3 provenance generation."""
        artifact_path = tmp_path / "artifact.tar.gz"
        artifact_path.write_bytes(b"content")

        output_path = tmp_path / "provenance.json"

        provenance = generate_slsa_provenance(
            artifact_path=str(artifact_path),
            builder="GitHub Actions",
            build_type="workflow",
            output_path=str(output_path),
            slsa_level=3
        )

        prov_data = json.loads(output_path.read_text())

        # SLSA Level 3 should include builder info
        assert "predicate" in prov_data
        assert "builder" in prov_data["predicate"]


class TestKeyManagement:
    """Test key management functions."""

    def test_generate_encryption_key(self, tmp_path):
        """Test generating AES encryption key."""
        key_path = tmp_path / "aes.key"

        from security import generate_encryption_key

        generate_encryption_key(output_path=str(key_path))

        assert key_path.exists()
        key_content = key_path.read_text().strip()
        assert len(key_content) == 64  # 32 bytes in hex = 64 chars

    def test_generate_rsa_keypair(self, tmp_path):
        """Test generating RSA key pair."""
        private_key_path = tmp_path / "rsa.key"
        public_key_path = tmp_path / "rsa.pub"

        from security import generate_rsa_keypair

        generate_rsa_keypair(
            private_key_path=str(private_key_path),
            public_key_path=str(public_key_path)
        )

        assert private_key_path.exists()
        assert public_key_path.exists()

        # Verify key format
        private_key = private_key_path.read_text()
        public_key = public_key_path.read_text()

        assert "BEGIN RSA PRIVATE KEY" in private_key or \
               "BEGIN PRIVATE KEY" in private_key
        assert "BEGIN PUBLIC KEY" in public_key


# Helper functions
def _is_gpg_available():
    """Check if GPG is available."""
    import shutil
    return shutil.which("gpg") is not None


# Pytest fixtures
@pytest.fixture
def aes_key(tmp_path):
    """Generate temporary AES key."""
    key_path = tmp_path / "aes.key"
    import os
    key_path.write_text(os.urandom(32).hex())
    return str(key_path)


@pytest.fixture
def rsa_keypair(tmp_path):
    """Generate temporary RSA key pair."""
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.backends import default_backend

    # Generate private key
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
        backend=default_backend()
    )

    # Generate public key
    public_key = private_key.public_key()

    # Save private key
    private_path = tmp_path / "rsa.key"
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()
    )
    private_path.write_bytes(private_pem)

    # Save public key
    public_path = tmp_path / "rsa.pub"
    public_pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    )
    public_path.write_bytes(public_pem)

    return str(private_path), str(public_path)


@pytest.fixture
def security_manager_aes(aes_key):
    """Create SecurityManager with AES encryption."""
    return SecurityManager(
        encryption_enabled=True,
        encryption_key_path=aes_key
    )


@pytest.fixture
def security_manager_rsa(rsa_keypair):
    """Create SecurityManager with RSA signing."""
    private_path, public_path = rsa_keypair
    return SecurityManager(
        signing_enabled=True,
        signing_key_path=private_path,
        public_key_path=public_path
    )


@pytest.fixture
def security_manager_pgp():
    """Create SecurityManager with PGP (if available)."""
    if not _is_gpg_available():
        pytest.skip("GPG not available")

    return SecurityManager(
        pgp_enabled=True,
        pgp_key_id="test@example.com"
    )


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
