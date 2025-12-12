"""
Encryption Utilities for T.A.R.S. Enterprise

Supports:
- AES-256 encryption for data at rest
- PGP/GPG encryption for additional security
"""

from typing import Optional
from pathlib import Path
import os
import base64
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import padding
import gnupg


class AESEncryption:
    """
    AES-256 encryption for reports and sensitive data.

    Uses AES-256-CBC with PKCS7 padding.
    """

    def __init__(self, key_path: Optional[Path] = None):
        """
        Initialize AES encryption.

        Args:
            key_path: Path to 32-byte AES key file (if None, generates new key)
        """
        if key_path and key_path.exists():
            with open(key_path, "rb") as f:
                self.key = f.read()
        else:
            self.key = self._generate_key()
            if key_path:
                key_path.parent.mkdir(parents=True, exist_ok=True)
                with open(key_path, "wb") as f:
                    f.write(self.key)
                # Set restrictive permissions (owner read-only)
                if hasattr(os, 'chmod'):
                    os.chmod(key_path, 0o400)

        if len(self.key) != 32:
            raise ValueError("AES-256 requires a 32-byte key")

    def encrypt(self, plaintext: bytes) -> bytes:
        """
        Encrypt data with AES-256-CBC.

        Args:
            plaintext: Data to encrypt

        Returns:
            Encrypted data (IV + ciphertext)
        """
        # Generate random IV
        iv = os.urandom(16)

        # Pad plaintext to block size (16 bytes)
        padder = padding.PKCS7(128).padder()
        padded_data = padder.update(plaintext) + padder.finalize()

        # Encrypt
        cipher = Cipher(
            algorithms.AES(self.key),
            modes.CBC(iv),
            backend=default_backend()
        )
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(padded_data) + encryptor.finalize()

        # Return IV + ciphertext
        return iv + ciphertext

    def decrypt(self, ciphertext: bytes) -> bytes:
        """
        Decrypt AES-256-CBC data.

        Args:
            ciphertext: Encrypted data (IV + ciphertext)

        Returns:
            Decrypted plaintext
        """
        # Extract IV (first 16 bytes)
        iv = ciphertext[:16]
        ciphertext = ciphertext[16:]

        # Decrypt
        cipher = Cipher(
            algorithms.AES(self.key),
            modes.CBC(iv),
            backend=default_backend()
        )
        decryptor = cipher.decryptor()
        padded_plaintext = decryptor.update(ciphertext) + decryptor.finalize()

        # Unpad
        unpadder = padding.PKCS7(128).unpadder()
        plaintext = unpadder.update(padded_plaintext) + unpadder.finalize()

        return plaintext

    def encrypt_file(self, input_path: Path, output_path: Path):
        """
        Encrypt a file.

        Args:
            input_path: Path to plaintext file
            output_path: Path to encrypted output file
        """
        with open(input_path, "rb") as f:
            plaintext = f.read()

        ciphertext = self.encrypt(plaintext)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(ciphertext)

    def decrypt_file(self, input_path: Path, output_path: Path):
        """
        Decrypt a file.

        Args:
            input_path: Path to encrypted file
            output_path: Path to decrypted output file
        """
        with open(input_path, "rb") as f:
            ciphertext = f.read()

        plaintext = self.decrypt(ciphertext)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(plaintext)

    def encrypt_string(self, plaintext: str) -> str:
        """
        Encrypt a string and return base64-encoded ciphertext.

        Args:
            plaintext: String to encrypt

        Returns:
            Base64-encoded encrypted string
        """
        ciphertext = self.encrypt(plaintext.encode("utf-8"))
        return base64.b64encode(ciphertext).decode("ascii")

    def decrypt_string(self, ciphertext_b64: str) -> str:
        """
        Decrypt a base64-encoded ciphertext string.

        Args:
            ciphertext_b64: Base64-encoded encrypted string

        Returns:
            Decrypted plaintext string
        """
        ciphertext = base64.b64decode(ciphertext_b64)
        plaintext = self.decrypt(ciphertext)
        return plaintext.decode("utf-8")

    @staticmethod
    def _generate_key() -> bytes:
        """Generate a random 32-byte AES-256 key."""
        return os.urandom(32)


class PGPEncryption:
    """
    PGP/GPG encryption for additional security layer.

    Requires gpg binary to be installed.
    """

    def __init__(self, gpg_home: Optional[Path] = None):
        """
        Initialize PGP encryption.

        Args:
            gpg_home: GPG home directory (default: ~/.gnupg)
        """
        gpg_home_str = str(gpg_home) if gpg_home else None
        self.gpg = gnupg.GPG(gnupghome=gpg_home_str)

    def encrypt(
        self,
        data: bytes,
        recipient: str,
        armor: bool = True,
    ) -> bytes:
        """
        Encrypt data with PGP.

        Args:
            data: Data to encrypt
            recipient: PGP key ID or email
            armor: If True, return ASCII-armored output

        Returns:
            Encrypted data
        """
        encrypted = self.gpg.encrypt(
            data,
            recipient,
            armor=armor,
            always_trust=True,  # For automated systems
        )

        if not encrypted.ok:
            raise RuntimeError(f"PGP encryption failed: {encrypted.status}")

        return str(encrypted).encode("utf-8") if armor else bytes(encrypted)

    def decrypt(
        self,
        ciphertext: bytes,
        passphrase: Optional[str] = None,
    ) -> bytes:
        """
        Decrypt PGP-encrypted data.

        Args:
            ciphertext: Encrypted data
            passphrase: Private key passphrase (if required)

        Returns:
            Decrypted plaintext
        """
        decrypted = self.gpg.decrypt(
            ciphertext,
            passphrase=passphrase,
            always_trust=True,
        )

        if not decrypted.ok:
            raise RuntimeError(f"PGP decryption failed: {decrypted.status}")

        return bytes(decrypted)

    def encrypt_file(
        self,
        input_path: Path,
        output_path: Path,
        recipient: str,
    ):
        """
        Encrypt a file with PGP.

        Args:
            input_path: Path to plaintext file
            output_path: Path to encrypted output file
            recipient: PGP key ID or email
        """
        with open(input_path, "rb") as f:
            encrypted = self.gpg.encrypt_file(
                f,
                recipient,
                armor=True,
                always_trust=True,
            )

        if not encrypted.ok:
            raise RuntimeError(f"PGP file encryption failed: {encrypted.status}")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(str(encrypted))

    def import_key(self, key_path: Path) -> str:
        """
        Import a PGP public key.

        Args:
            key_path: Path to public key file

        Returns:
            Key ID
        """
        with open(key_path, "r") as f:
            key_data = f.read()

        import_result = self.gpg.import_keys(key_data)

        if not import_result.count:
            raise RuntimeError("Failed to import PGP key")

        return import_result.fingerprints[0]

    def list_keys(self) -> list:
        """List all imported PGP keys."""
        return self.gpg.list_keys()
