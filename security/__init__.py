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

# Encryption modules
from .encryption import AESEncryption, PGPEncryption

# Signing module
from .signing import ReportSigner

# Supply chain security - production generators
from .sbom_generator import SBOMGenerator, generate_sbom_for_tars, Dependency
from .slsa_generator import SLSAProvenanceGenerator, generate_slsa_provenance_for_tars

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
    "Dependency",

    # SLSA
    "SLSAProvenanceGenerator",
    "generate_slsa_provenance_for_tars",
]

__version__ = "1.0.2"
__author__ = "Veleron Dev Studios"
__license__ = "Proprietary"
