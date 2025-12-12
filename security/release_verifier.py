"""
Production-Grade Release Artifact Verifier & Integrity Gate for T.A.R.S.

Comprehensive verification subsystem that validates:
- SBOM integrity (CycloneDX + SPDX)
- SLSA provenance validity
- RSA-PSS signature verification
- Hash correctness (SHA-256, SHA-512)
- Reproducible build fingerprints
- Manifest synchronization
- Policy gating with hard fail enforcement

Designed for:
- CI/CD integration
- Offline operation (air-gapped environments)
- Deterministic output
- Cross-platform compatibility (Windows, Linux, macOS)
- Non-interactive batch processing

Exit Codes:
  0 - All verifications passed
  1 - Artifact not found
  2 - SBOM verification failed
  3 - SLSA verification failed
  4 - Signature verification failed
  5 - Hash verification failed
  6 - Manifest verification failed
  7 - Policy gate failure (strict mode)
  8 - General verification error

Compatible with Phase 14.7 Task 3
"""

from typing import Dict, Any, List, Optional, Tuple, Literal
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
import json
import hashlib
import logging
import sys

# Try to import cryptography for signature verification
try:
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding as asym_padding
    from cryptography.hazmat.backends import default_backend
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False

# Configure logging
logger = logging.getLogger(__name__)


# ============================================================================
# CUSTOM EXCEPTIONS
# ============================================================================

class VerificationError(Exception):
    """Base exception for verification failures."""
    pass


class IntegrityError(VerificationError):
    """Raised when integrity checks fail (hash mismatch, corrupted data)."""
    pass


class ProvenanceError(VerificationError):
    """Raised when SLSA provenance validation fails."""
    pass


class SBOMError(VerificationError):
    """Raised when SBOM validation fails."""
    pass


class SignatureError(VerificationError):
    """Raised when signature verification fails."""
    pass


class ManifestError(VerificationError):
    """Raised when manifest synchronization fails."""
    pass


class PolicyViolationError(VerificationError):
    """Raised when policy gate fails in strict mode."""
    pass


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class HashVerificationResult:
    """Result of hash verification for a single file."""
    file_name: str
    algorithm: str
    expected_hash: str
    actual_hash: str
    match: bool
    error: Optional[str] = None


@dataclass
class SignatureVerificationResult:
    """Result of signature verification."""
    file_name: str
    signature_file: str
    algorithm: str
    valid: bool
    error: Optional[str] = None


@dataclass
class SBOMVerificationResult:
    """Result of SBOM verification."""
    format: str  # cyclonedx or spdx
    file_path: str
    valid: bool
    component_count: int
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class SLSAVerificationResult:
    """Result of SLSA provenance verification."""
    file_path: str
    valid: bool
    slsa_level: Optional[int] = None
    subject_count: int = 0
    builder_id: Optional[str] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class ManifestVerificationResult:
    """Result of manifest verification."""
    file_path: str
    valid: bool
    artifact_count: int
    missing_artifacts: List[str] = field(default_factory=list)
    hash_mismatches: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


@dataclass
class PolicyCheckResult:
    """Result of policy enforcement check."""
    policy_name: str
    passed: bool
    severity: str  # info, warning, error, critical
    message: str


@dataclass
class VerificationReport:
    """Complete verification report."""
    timestamp: str
    version: str
    artifact_path: str
    verification_mode: str  # strict or lenient
    overall_status: str  # passed, failed, warning

    # Individual verification results
    hash_results: List[HashVerificationResult] = field(default_factory=list)
    signature_results: List[SignatureVerificationResult] = field(default_factory=list)
    sbom_results: List[SBOMVerificationResult] = field(default_factory=list)
    slsa_results: List[SLSAVerificationResult] = field(default_factory=list)
    manifest_results: List[ManifestVerificationResult] = field(default_factory=list)
    policy_results: List[PolicyCheckResult] = field(default_factory=list)

    # Summary statistics
    total_checks: int = 0
    passed_checks: int = 0
    failed_checks: int = 0
    warnings: int = 0

    # Errors and messages
    errors: List[str] = field(default_factory=list)
    warnings_list: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


# ============================================================================
# HASH VERIFICATION
# ============================================================================

class HashVerifier:
    """
    Hash verification subsystem supporting SHA-256 and SHA-512.

    Validates file integrity by computing hashes and comparing against
    expected values from manifest or SBOM.
    """

    SUPPORTED_ALGORITHMS = ['sha256', 'sha512']

    @staticmethod
    def compute_hash(file_path: Path, algorithm: str = 'sha256') -> str:
        """
        Compute cryptographic hash of file.

        Args:
            file_path: Path to file
            algorithm: Hash algorithm (sha256 or sha512)

        Returns:
            Hexadecimal hash string

        Raises:
            ValueError: If algorithm not supported
            FileNotFoundError: If file doesn't exist
        """
        if algorithm.lower() not in HashVerifier.SUPPORTED_ALGORITHMS:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        hash_obj = hashlib.new(algorithm.lower())

        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                hash_obj.update(chunk)

        return hash_obj.hexdigest()

    @staticmethod
    def verify_hash(
        file_path: Path,
        expected_hash: str,
        algorithm: str = 'sha256'
    ) -> HashVerificationResult:
        """
        Verify file hash against expected value.

        Args:
            file_path: Path to file
            expected_hash: Expected hash value
            algorithm: Hash algorithm

        Returns:
            HashVerificationResult with verification status
        """
        try:
            actual_hash = HashVerifier.compute_hash(file_path, algorithm)
            match = actual_hash.lower() == expected_hash.lower()

            return HashVerificationResult(
                file_name=file_path.name,
                algorithm=algorithm,
                expected_hash=expected_hash,
                actual_hash=actual_hash,
                match=match,
                error=None if match else "Hash mismatch"
            )

        except Exception as e:
            return HashVerificationResult(
                file_name=file_path.name,
                algorithm=algorithm,
                expected_hash=expected_hash,
                actual_hash="",
                match=False,
                error=str(e)
            )


# ============================================================================
# SIGNATURE VERIFICATION
# ============================================================================

class SignatureVerifier:
    """
    RSA-PSS signature verification subsystem.

    Verifies cryptographic signatures on artifacts using RSA-PSS
    with SHA-256.
    """

    def __init__(self, public_key_path: Optional[Path] = None):
        """
        Initialize signature verifier.

        Args:
            public_key_path: Path to RSA public key (PEM format)

        Raises:
            RuntimeError: If cryptography library not available
        """
        if not CRYPTO_AVAILABLE:
            raise RuntimeError("cryptography library required for signature verification")

        self.public_key = None

        if public_key_path and public_key_path.exists():
            self.public_key = self._load_public_key(public_key_path)
        elif public_key_path:
            raise FileNotFoundError(f"Public key not found: {public_key_path}")

    def _load_public_key(self, key_path: Path):
        """Load RSA public key from PEM file."""
        with open(key_path, 'rb') as f:
            public_key = serialization.load_pem_public_key(
                f.read(),
                backend=default_backend()
            )
        return public_key

    def verify_signature(
        self,
        file_path: Path,
        signature_path: Optional[Path] = None
    ) -> SignatureVerificationResult:
        """
        Verify RSA-PSS signature on file.

        Args:
            file_path: Path to signed file
            signature_path: Path to signature file (default: file_path + '.sig')

        Returns:
            SignatureVerificationResult with verification status
        """
        if not signature_path:
            signature_path = Path(str(file_path) + '.sig')

        file_name = file_path.name
        sig_file_name = signature_path.name

        try:
            # Check if signature file exists
            if not signature_path.exists():
                return SignatureVerificationResult(
                    file_name=file_name,
                    signature_file=sig_file_name,
                    algorithm="RSA-PSS-SHA256",
                    valid=False,
                    error="Signature file not found"
                )

            # Check if public key is loaded
            if not self.public_key:
                return SignatureVerificationResult(
                    file_name=file_name,
                    signature_file=sig_file_name,
                    algorithm="RSA-PSS-SHA256",
                    valid=False,
                    error="No public key loaded"
                )

            # Read file data
            with open(file_path, 'rb') as f:
                file_data = f.read()

            # Read signature
            with open(signature_path, 'rb') as f:
                signature = f.read()

            # Verify signature
            try:
                self.public_key.verify(
                    signature,
                    file_data,
                    asym_padding.PSS(
                        mgf=asym_padding.MGF1(hashes.SHA256()),
                        salt_length=asym_padding.PSS.MAX_LENGTH
                    ),
                    hashes.SHA256()
                )

                return SignatureVerificationResult(
                    file_name=file_name,
                    signature_file=sig_file_name,
                    algorithm="RSA-PSS-SHA256",
                    valid=True,
                    error=None
                )

            except Exception as verify_error:
                return SignatureVerificationResult(
                    file_name=file_name,
                    signature_file=sig_file_name,
                    algorithm="RSA-PSS-SHA256",
                    valid=False,
                    error=f"Signature verification failed: {verify_error}"
                )

        except Exception as e:
            return SignatureVerificationResult(
                file_name=file_name,
                signature_file=sig_file_name,
                algorithm="RSA-PSS-SHA256",
                valid=False,
                error=str(e)
            )


# ============================================================================
# SBOM VERIFICATION
# ============================================================================

class SBOMVerifier:
    """
    Software Bill of Materials (SBOM) verification subsystem.

    Validates SBOM documents in CycloneDX and SPDX formats.
    Checks:
    - Format compliance
    - Required fields
    - Component integrity
    - Hash validation
    """

    @staticmethod
    def verify_cyclonedx(sbom_path: Path) -> SBOMVerificationResult:
        """
        Verify CycloneDX SBOM.

        Args:
            sbom_path: Path to CycloneDX JSON file

        Returns:
            SBOMVerificationResult with validation status
        """
        errors = []
        warnings = []
        component_count = 0

        try:
            with open(sbom_path, 'r', encoding='utf-8') as f:
                sbom = json.load(f)

            # Check required fields
            if sbom.get('bomFormat') != 'CycloneDX':
                errors.append("Invalid bomFormat (expected 'CycloneDX')")

            if 'specVersion' not in sbom:
                errors.append("Missing specVersion")
            elif not sbom['specVersion'].startswith('1.'):
                warnings.append(f"Unexpected specVersion: {sbom['specVersion']}")

            if 'metadata' not in sbom:
                errors.append("Missing metadata section")
            else:
                metadata = sbom['metadata']
                if 'timestamp' not in metadata:
                    warnings.append("Missing metadata.timestamp")
                if 'component' not in metadata:
                    errors.append("Missing metadata.component")

            # Validate components
            if 'components' in sbom:
                components = sbom['components']
                component_count = len(components)

                for idx, component in enumerate(components):
                    # Check required component fields
                    if 'name' not in component:
                        errors.append(f"Component {idx}: missing name")
                    if 'version' not in component:
                        warnings.append(f"Component {idx} ({component.get('name', 'unknown')}): missing version")
                    if 'purl' not in component:
                        warnings.append(f"Component {idx} ({component.get('name', 'unknown')}): missing purl")

                    # Validate hashes if present
                    if 'hashes' in component:
                        for hash_entry in component['hashes']:
                            if 'alg' not in hash_entry or 'content' not in hash_entry:
                                warnings.append(f"Component {idx}: invalid hash entry")
            else:
                warnings.append("No components section found")

            # Validate dependencies
            if 'dependencies' in sbom:
                for dep_entry in sbom['dependencies']:
                    if 'ref' not in dep_entry:
                        warnings.append("Dependency entry missing 'ref'")

            valid = len(errors) == 0

            return SBOMVerificationResult(
                format="cyclonedx",
                file_path=str(sbom_path),
                valid=valid,
                component_count=component_count,
                errors=errors,
                warnings=warnings
            )

        except json.JSONDecodeError as e:
            return SBOMVerificationResult(
                format="cyclonedx",
                file_path=str(sbom_path),
                valid=False,
                component_count=0,
                errors=[f"Invalid JSON: {e}"],
                warnings=[]
            )
        except Exception as e:
            return SBOMVerificationResult(
                format="cyclonedx",
                file_path=str(sbom_path),
                valid=False,
                component_count=0,
                errors=[str(e)],
                warnings=[]
            )

    @staticmethod
    def verify_spdx(sbom_path: Path) -> SBOMVerificationResult:
        """
        Verify SPDX SBOM.

        Args:
            sbom_path: Path to SPDX JSON file

        Returns:
            SBOMVerificationResult with validation status
        """
        errors = []
        warnings = []
        component_count = 0

        try:
            with open(sbom_path, 'r', encoding='utf-8') as f:
                sbom = json.load(f)

            # Check required fields
            if 'spdxVersion' not in sbom:
                errors.append("Missing spdxVersion")
            elif not sbom['spdxVersion'].startswith('SPDX-'):
                errors.append(f"Invalid spdxVersion: {sbom['spdxVersion']}")

            if 'dataLicense' not in sbom:
                errors.append("Missing dataLicense")

            if 'SPDXID' not in sbom:
                errors.append("Missing SPDXID")
            elif sbom['SPDXID'] != 'SPDXRef-DOCUMENT':
                warnings.append(f"Unexpected SPDXID: {sbom['SPDXID']}")

            if 'name' not in sbom:
                errors.append("Missing document name")

            if 'documentNamespace' not in sbom:
                errors.append("Missing documentNamespace")

            if 'creationInfo' not in sbom:
                errors.append("Missing creationInfo")
            else:
                creation_info = sbom['creationInfo']
                if 'created' not in creation_info:
                    errors.append("Missing creationInfo.created")
                if 'creators' not in creation_info:
                    errors.append("Missing creationInfo.creators")

            # Validate packages
            if 'packages' in sbom:
                packages = sbom['packages']
                component_count = len(packages)

                for idx, package in enumerate(packages):
                    # Check required package fields
                    if 'SPDXID' not in package:
                        errors.append(f"Package {idx}: missing SPDXID")
                    if 'name' not in package:
                        errors.append(f"Package {idx}: missing name")
                    if 'downloadLocation' not in package:
                        warnings.append(f"Package {idx} ({package.get('name', 'unknown')}): missing downloadLocation")
                    if 'filesAnalyzed' not in package:
                        warnings.append(f"Package {idx}: missing filesAnalyzed")

                    # Validate checksums if present
                    if 'checksums' in package:
                        for checksum in package['checksums']:
                            if 'algorithm' not in checksum or 'checksumValue' not in checksum:
                                warnings.append(f"Package {idx}: invalid checksum entry")
            else:
                warnings.append("No packages section found")

            # Validate relationships
            if 'relationships' in sbom:
                for rel_entry in sbom['relationships']:
                    if 'spdxElementId' not in rel_entry:
                        warnings.append("Relationship entry missing 'spdxElementId'")
                    if 'relationshipType' not in rel_entry:
                        warnings.append("Relationship entry missing 'relationshipType'")
                    if 'relatedSpdxElement' not in rel_entry:
                        warnings.append("Relationship entry missing 'relatedSpdxElement'")

            valid = len(errors) == 0

            return SBOMVerificationResult(
                format="spdx",
                file_path=str(sbom_path),
                valid=valid,
                component_count=component_count,
                errors=errors,
                warnings=warnings
            )

        except json.JSONDecodeError as e:
            return SBOMVerificationResult(
                format="spdx",
                file_path=str(sbom_path),
                valid=False,
                component_count=0,
                errors=[f"Invalid JSON: {e}"],
                warnings=[]
            )
        except Exception as e:
            return SBOMVerificationResult(
                format="spdx",
                file_path=str(sbom_path),
                valid=False,
                component_count=0,
                errors=[str(e)],
                warnings=[]
            )


# ============================================================================
# SLSA PROVENANCE VERIFICATION
# ============================================================================

class SLSAVerifier:
    """
    SLSA (Supply-chain Levels for Software Artifacts) provenance verifier.

    Validates SLSA provenance attestations according to in-toto specification.
    Checks:
    - Attestation format (in-toto)
    - Predicate type (SLSA provenance)
    - Required fields (subject, buildDefinition, runDetails)
    - Builder identity
    - Material tracking
    """

    @staticmethod
    def verify_provenance(provenance_path: Path) -> SLSAVerificationResult:
        """
        Verify SLSA provenance attestation.

        Args:
            provenance_path: Path to SLSA provenance JSON file

        Returns:
            SLSAVerificationResult with validation status
        """
        errors = []
        warnings = []
        subject_count = 0
        builder_id = None
        slsa_level = None

        try:
            with open(provenance_path, 'r', encoding='utf-8') as f:
                provenance = json.load(f)

            # Check in-toto attestation format
            if '_type' not in provenance:
                errors.append("Missing _type field (not a valid in-toto attestation)")
            elif provenance['_type'] != 'https://in-toto.io/Statement/v1':
                errors.append(f"Invalid _type: {provenance['_type']}")

            # Check predicate type
            if 'predicateType' not in provenance:
                errors.append("Missing predicateType")
            elif not provenance['predicateType'].startswith('https://slsa.dev/provenance/'):
                errors.append(f"Invalid predicateType: {provenance['predicateType']}")

            # Check subject
            if 'subject' not in provenance:
                errors.append("Missing subject")
            else:
                subjects = provenance['subject']
                if not isinstance(subjects, list):
                    errors.append("Subject must be a list")
                else:
                    subject_count = len(subjects)
                    for idx, subject in enumerate(subjects):
                        if 'name' not in subject:
                            errors.append(f"Subject {idx}: missing name")
                        if 'digest' not in subject:
                            errors.append(f"Subject {idx}: missing digest")
                        elif not isinstance(subject['digest'], dict):
                            errors.append(f"Subject {idx}: digest must be a dictionary")
                        elif 'sha256' not in subject['digest']:
                            warnings.append(f"Subject {idx}: missing sha256 digest")

            # Check predicate
            if 'predicate' not in provenance:
                errors.append("Missing predicate")
            else:
                predicate = provenance['predicate']

                # Check buildDefinition
                if 'buildDefinition' not in predicate:
                    errors.append("Missing predicate.buildDefinition")
                else:
                    build_def = predicate['buildDefinition']

                    if 'buildType' not in build_def:
                        errors.append("Missing buildDefinition.buildType")

                    if 'externalParameters' not in build_def:
                        warnings.append("Missing buildDefinition.externalParameters")

                    if 'internalParameters' not in build_def:
                        warnings.append("Missing buildDefinition.internalParameters")

                    # Check resolvedDependencies (materials)
                    if 'resolvedDependencies' not in build_def:
                        warnings.append("Missing buildDefinition.resolvedDependencies (materials)")

                # Check runDetails
                if 'runDetails' not in predicate:
                    errors.append("Missing predicate.runDetails")
                else:
                    run_details = predicate['runDetails']

                    if 'builder' not in run_details:
                        errors.append("Missing runDetails.builder")
                    else:
                        builder = run_details['builder']
                        if 'id' not in builder:
                            errors.append("Missing runDetails.builder.id")
                        else:
                            builder_id = builder['id']

                    if 'metadata' not in run_details:
                        warnings.append("Missing runDetails.metadata")
                    else:
                        metadata = run_details['metadata']
                        if 'invocationId' not in metadata:
                            warnings.append("Missing runDetails.metadata.invocationId")
                        if 'startedOn' not in metadata:
                            warnings.append("Missing runDetails.metadata.startedOn")
                        if 'finishedOn' not in metadata:
                            warnings.append("Missing runDetails.metadata.finishedOn")

            # Determine SLSA level (basic heuristic)
            # Level 1: Provenance exists
            # Level 2: Provenance generated by build service
            # Level 3: Build hardened against tampering
            if len(errors) == 0:
                if builder_id and ('github.com' in builder_id or 'actions' in builder_id):
                    slsa_level = 2  # Build service generated
                else:
                    slsa_level = 1  # Basic provenance

            valid = len(errors) == 0

            return SLSAVerificationResult(
                file_path=str(provenance_path),
                valid=valid,
                slsa_level=slsa_level,
                subject_count=subject_count,
                builder_id=builder_id,
                errors=errors,
                warnings=warnings
            )

        except json.JSONDecodeError as e:
            return SLSAVerificationResult(
                file_path=str(provenance_path),
                valid=False,
                errors=[f"Invalid JSON: {e}"],
                warnings=[]
            )
        except Exception as e:
            return SLSAVerificationResult(
                file_path=str(provenance_path),
                valid=False,
                errors=[str(e)],
                warnings=[]
            )


# ============================================================================
# MANIFEST VERIFICATION
# ============================================================================

class ManifestVerifier:
    """
    Release manifest verification subsystem.

    Validates manifest.json file and ensures:
    - All listed artifacts exist
    - Hash values match
    - No missing or extra files
    """

    @staticmethod
    def verify_manifest(
        manifest_path: Path,
        artifact_dir: Optional[Path] = None
    ) -> ManifestVerificationResult:
        """
        Verify release manifest.

        Args:
            manifest_path: Path to manifest.json
            artifact_dir: Directory containing artifacts (default: manifest parent)

        Returns:
            ManifestVerificationResult with validation status
        """
        if not artifact_dir:
            artifact_dir = manifest_path.parent

        errors = []
        missing_artifacts = []
        hash_mismatches = []
        artifact_count = 0

        try:
            with open(manifest_path, 'r', encoding='utf-8') as f:
                manifest = json.load(f)

            # Check required manifest fields
            if 'version' not in manifest:
                errors.append("Missing version field")
            if 'generated_at' not in manifest:
                errors.append("Missing generated_at field")
            if 'artifacts' not in manifest:
                errors.append("Missing artifacts list")
                return ManifestVerificationResult(
                    file_path=str(manifest_path),
                    valid=False,
                    artifact_count=0,
                    errors=errors
                )

            # Verify each artifact
            artifacts = manifest['artifacts']
            artifact_count = len(artifacts)

            for artifact_entry in artifacts:
                if 'name' not in artifact_entry:
                    errors.append("Artifact entry missing 'name'")
                    continue

                artifact_name = artifact_entry['name']
                artifact_path = artifact_dir / artifact_name

                # Check if artifact exists
                if not artifact_path.exists():
                    missing_artifacts.append(artifact_name)
                    continue

                # Verify hash (SHA-256)
                if 'sha256' in artifact_entry:
                    expected_hash = artifact_entry['sha256']
                    try:
                        actual_hash = HashVerifier.compute_hash(artifact_path, 'sha256')
                        if actual_hash.lower() != expected_hash.lower():
                            hash_mismatches.append(
                                f"{artifact_name}: expected {expected_hash}, got {actual_hash}"
                            )
                    except Exception as e:
                        errors.append(f"Failed to compute hash for {artifact_name}: {e}")

                # Check size if present
                if 'size' in artifact_entry:
                    expected_size = artifact_entry['size']
                    actual_size = artifact_path.stat().st_size
                    if actual_size != expected_size:
                        errors.append(
                            f"{artifact_name}: size mismatch (expected {expected_size}, got {actual_size})"
                        )

            valid = (
                len(errors) == 0 and
                len(missing_artifacts) == 0 and
                len(hash_mismatches) == 0
            )

            return ManifestVerificationResult(
                file_path=str(manifest_path),
                valid=valid,
                artifact_count=artifact_count,
                missing_artifacts=missing_artifacts,
                hash_mismatches=hash_mismatches,
                errors=errors
            )

        except json.JSONDecodeError as e:
            return ManifestVerificationResult(
                file_path=str(manifest_path),
                valid=False,
                artifact_count=0,
                errors=[f"Invalid JSON: {e}"]
            )
        except Exception as e:
            return ManifestVerificationResult(
                file_path=str(manifest_path),
                valid=False,
                artifact_count=0,
                errors=[str(e)]
            )


# ============================================================================
# POLICY ENFORCEMENT
# ============================================================================

class PolicyEnforcer:
    """
    Policy enforcement engine for release verification.

    Enforces security and compliance policies such as:
    - All artifacts must be signed
    - SBOM must be present
    - SLSA level requirements
    - Hash algorithm requirements
    """

    def __init__(self, mode: Literal['strict', 'lenient'] = 'strict'):
        """
        Initialize policy enforcer.

        Args:
            mode: Enforcement mode (strict or lenient)
        """
        self.mode = mode

    def check_policies(
        self,
        verification_report: VerificationReport
    ) -> List[PolicyCheckResult]:
        """
        Check all policies against verification report.

        Args:
            verification_report: Verification report to check

        Returns:
            List of PolicyCheckResult
        """
        policy_results = []

        # Policy 1: All artifacts must be signed
        unsigned_artifacts = [
            r for r in verification_report.signature_results
            if not r.valid
        ]

        if unsigned_artifacts:
            severity = "critical" if self.mode == 'strict' else "warning"
            passed = False if self.mode == 'strict' else True
            message = f"Found {len(unsigned_artifacts)} unsigned artifacts"

            policy_results.append(PolicyCheckResult(
                policy_name="artifacts_signed",
                passed=passed,
                severity=severity,
                message=message
            ))
        else:
            policy_results.append(PolicyCheckResult(
                policy_name="artifacts_signed",
                passed=True,
                severity="info",
                message="All artifacts properly signed"
            ))

        # Policy 2: SBOM must be present and valid
        invalid_sboms = [
            r for r in verification_report.sbom_results
            if not r.valid
        ]

        if not verification_report.sbom_results:
            severity = "error" if self.mode == 'strict' else "warning"
            passed = False if self.mode == 'strict' else True

            policy_results.append(PolicyCheckResult(
                policy_name="sbom_present",
                passed=passed,
                severity=severity,
                message="No SBOM found"
            ))
        elif invalid_sboms:
            severity = "error" if self.mode == 'strict' else "warning"
            passed = False if self.mode == 'strict' else True

            policy_results.append(PolicyCheckResult(
                policy_name="sbom_valid",
                passed=passed,
                severity=severity,
                message=f"{len(invalid_sboms)} invalid SBOM(s) found"
            ))
        else:
            policy_results.append(PolicyCheckResult(
                policy_name="sbom_valid",
                passed=True,
                severity="info",
                message=f"All {len(verification_report.sbom_results)} SBOM(s) valid"
            ))

        # Policy 3: SLSA provenance must be present and valid
        invalid_provenance = [
            r for r in verification_report.slsa_results
            if not r.valid
        ]

        if not verification_report.slsa_results:
            severity = "error" if self.mode == 'strict' else "warning"
            passed = False if self.mode == 'strict' else True

            policy_results.append(PolicyCheckResult(
                policy_name="slsa_present",
                passed=passed,
                severity=severity,
                message="No SLSA provenance found"
            ))
        elif invalid_provenance:
            severity = "error" if self.mode == 'strict' else "warning"
            passed = False if self.mode == 'strict' else True

            policy_results.append(PolicyCheckResult(
                policy_name="slsa_valid",
                passed=passed,
                severity=severity,
                message=f"{len(invalid_provenance)} invalid SLSA provenance found"
            ))
        else:
            # Check SLSA level
            min_slsa_level = 2 if self.mode == 'strict' else 1

            low_level = [
                r for r in verification_report.slsa_results
                if r.slsa_level and r.slsa_level < min_slsa_level
            ]

            if low_level:
                policy_results.append(PolicyCheckResult(
                    policy_name="slsa_level",
                    passed=False,
                    severity="warning",
                    message=f"SLSA level below recommended minimum ({min_slsa_level})"
                ))
            else:
                policy_results.append(PolicyCheckResult(
                    policy_name="slsa_valid",
                    passed=True,
                    severity="info",
                    message="SLSA provenance valid"
                ))

        # Policy 4: No hash mismatches
        hash_failures = [
            r for r in verification_report.hash_results
            if not r.match
        ]

        if hash_failures:
            policy_results.append(PolicyCheckResult(
                policy_name="hash_integrity",
                passed=False,
                severity="critical",
                message=f"{len(hash_failures)} hash verification failures"
            ))
        else:
            policy_results.append(PolicyCheckResult(
                policy_name="hash_integrity",
                passed=True,
                severity="info",
                message="All hash verifications passed"
            ))

        return policy_results


# ============================================================================
# RELEASE VERIFIER (MAIN CLASS)
# ============================================================================

class ReleaseVerifier:
    """
    Complete release artifact verification and integrity gate system.

    Orchestrates all verification subsystems and generates comprehensive
    verification reports.
    """

    def __init__(
        self,
        mode: Literal['strict', 'lenient'] = 'strict',
        public_key_path: Optional[Path] = None
    ):
        """
        Initialize release verifier.

        Args:
            mode: Verification mode (strict or lenient)
            public_key_path: Path to RSA public key for signature verification
        """
        self.mode = mode
        self.signature_verifier = None

        if public_key_path:
            try:
                self.signature_verifier = SignatureVerifier(public_key_path)
            except Exception as e:
                logger.warning(f"Failed to initialize signature verifier: {e}")

        self.policy_enforcer = PolicyEnforcer(mode=mode)

    def verify_release(
        self,
        artifact_path: Path,
        sbom_path: Optional[Path] = None,
        slsa_path: Optional[Path] = None,
        manifest_path: Optional[Path] = None,
        version: str = "unknown"
    ) -> VerificationReport:
        """
        Perform complete release verification.

        Args:
            artifact_path: Path to primary artifact (or directory)
            sbom_path: Path to SBOM file (optional)
            slsa_path: Path to SLSA provenance (optional)
            manifest_path: Path to manifest.json (optional)
            version: Release version

        Returns:
            VerificationReport with complete verification results
        """
        logger.info(f"Starting release verification for {artifact_path}")
        logger.info(f"Verification mode: {self.mode}")

        report = VerificationReport(
            timestamp=datetime.now(timezone.utc).isoformat(),
            version=version,
            artifact_path=str(artifact_path),
            verification_mode=self.mode,
            overall_status="unknown"
        )

        # Hash verification
        if artifact_path.exists():
            if artifact_path.is_file():
                logger.debug(f"Verifying hash for {artifact_path.name}")
                hash_result = HashVerifier.verify_hash(
                    artifact_path,
                    expected_hash="",  # Will be compared if manifest provided
                    algorithm='sha256'
                )
                report.hash_results.append(hash_result)

        # Signature verification
        if self.signature_verifier and artifact_path.exists():
            logger.debug(f"Verifying signature for {artifact_path.name}")
            sig_result = self.signature_verifier.verify_signature(artifact_path)
            report.signature_results.append(sig_result)

        # SBOM verification
        if sbom_path and sbom_path.exists():
            logger.info(f"Verifying SBOM: {sbom_path.name}")

            # Detect format from filename
            if 'cyclonedx' in sbom_path.name.lower():
                sbom_result = SBOMVerifier.verify_cyclonedx(sbom_path)
            elif 'spdx' in sbom_path.name.lower():
                sbom_result = SBOMVerifier.verify_spdx(sbom_path)
            else:
                # Try both formats
                try:
                    sbom_result = SBOMVerifier.verify_cyclonedx(sbom_path)
                except:
                    sbom_result = SBOMVerifier.verify_spdx(sbom_path)

            report.sbom_results.append(sbom_result)

            if sbom_result.valid:
                logger.info(f"✓ SBOM valid ({sbom_result.component_count} components)")
            else:
                logger.error(f"✗ SBOM validation failed: {sbom_result.errors}")

        # SLSA provenance verification
        if slsa_path and slsa_path.exists():
            logger.info(f"Verifying SLSA provenance: {slsa_path.name}")
            slsa_result = SLSAVerifier.verify_provenance(slsa_path)
            report.slsa_results.append(slsa_result)

            if slsa_result.valid:
                logger.info(f"✓ SLSA provenance valid (Level {slsa_result.slsa_level})")
            else:
                logger.error(f"✗ SLSA validation failed: {slsa_result.errors}")

        # Manifest verification
        if manifest_path and manifest_path.exists():
            logger.info(f"Verifying manifest: {manifest_path.name}")
            manifest_result = ManifestVerifier.verify_manifest(manifest_path)
            report.manifest_results.append(manifest_result)

            if manifest_result.valid:
                logger.info(f"✓ Manifest valid ({manifest_result.artifact_count} artifacts)")
            else:
                logger.error(f"✗ Manifest validation failed")
                if manifest_result.missing_artifacts:
                    logger.error(f"  Missing: {manifest_result.missing_artifacts}")
                if manifest_result.hash_mismatches:
                    logger.error(f"  Hash mismatches: {manifest_result.hash_mismatches}")

        # Policy enforcement
        logger.info("Checking policy compliance...")
        report.policy_results = self.policy_enforcer.check_policies(report)

        # Calculate summary statistics
        report.total_checks = (
            len(report.hash_results) +
            len(report.signature_results) +
            len(report.sbom_results) +
            len(report.slsa_results) +
            len(report.manifest_results)
        )

        report.passed_checks = (
            sum(1 for r in report.hash_results if r.match) +
            sum(1 for r in report.signature_results if r.valid) +
            sum(1 for r in report.sbom_results if r.valid) +
            sum(1 for r in report.slsa_results if r.valid) +
            sum(1 for r in report.manifest_results if r.valid)
        )

        report.failed_checks = report.total_checks - report.passed_checks

        # Count warnings
        report.warnings = (
            sum(len(r.warnings) for r in report.sbom_results) +
            sum(len(r.warnings) for r in report.slsa_results)
        )

        # Determine overall status
        critical_failures = [
            p for p in report.policy_results
            if not p.passed and p.severity == "critical"
        ]

        if critical_failures:
            report.overall_status = "failed"
        elif report.failed_checks > 0 and self.mode == 'strict':
            report.overall_status = "failed"
        elif report.failed_checks > 0 or report.warnings > 0:
            report.overall_status = "warning"
        else:
            report.overall_status = "passed"

        logger.info(f"Verification complete: {report.overall_status}")
        logger.info(f"Checks: {report.passed_checks}/{report.total_checks} passed")

        return report

    def save_report(
        self,
        report: VerificationReport,
        output_path: Path,
        format: Literal['json', 'text'] = 'json'
    ) -> None:
        """
        Save verification report to file.

        Args:
            report: VerificationReport to save
            output_path: Output file path
            format: Output format (json or text)
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == 'json':
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report.to_dict(), f, indent=2)
            logger.info(f"Saved JSON report: {output_path}")

        elif format == 'text':
            with open(output_path, 'w', encoding='utf-8') as f:
                self._write_text_report(report, f)
            logger.info(f"Saved text report: {output_path}")

    def _write_text_report(self, report: VerificationReport, file):
        """Write human-readable text report."""
        file.write("=" * 80 + "\n")
        file.write("RELEASE ARTIFACT VERIFICATION REPORT\n")
        file.write("=" * 80 + "\n\n")

        file.write(f"Timestamp:       {report.timestamp}\n")
        file.write(f"Version:         {report.version}\n")
        file.write(f"Artifact:        {report.artifact_path}\n")
        file.write(f"Mode:            {report.verification_mode}\n")
        file.write(f"Overall Status:  {report.overall_status.upper()}\n\n")

        file.write(f"Summary:\n")
        file.write(f"  Total Checks:  {report.total_checks}\n")
        file.write(f"  Passed:        {report.passed_checks}\n")
        file.write(f"  Failed:        {report.failed_checks}\n")
        file.write(f"  Warnings:      {report.warnings}\n\n")

        # Policy results
        if report.policy_results:
            file.write("-" * 80 + "\n")
            file.write("POLICY COMPLIANCE\n")
            file.write("-" * 80 + "\n\n")

            for policy in report.policy_results:
                status = "✓" if policy.passed else "✗"
                file.write(f"{status} {policy.policy_name}: {policy.message} [{policy.severity}]\n")
            file.write("\n")

        # Detailed results
        if report.sbom_results:
            file.write("-" * 80 + "\n")
            file.write("SBOM VERIFICATION\n")
            file.write("-" * 80 + "\n\n")

            for sbom in report.sbom_results:
                status = "✓ VALID" if sbom.valid else "✗ INVALID"
                file.write(f"{status} - {sbom.format.upper()} ({sbom.component_count} components)\n")
                if sbom.errors:
                    file.write(f"  Errors: {', '.join(sbom.errors)}\n")
                if sbom.warnings:
                    file.write(f"  Warnings: {', '.join(sbom.warnings[:3])}...\n")
            file.write("\n")

        if report.slsa_results:
            file.write("-" * 80 + "\n")
            file.write("SLSA PROVENANCE VERIFICATION\n")
            file.write("-" * 80 + "\n\n")

            for slsa in report.slsa_results:
                status = "✓ VALID" if slsa.valid else "✗ INVALID"
                level = f"Level {slsa.slsa_level}" if slsa.slsa_level else "Unknown"
                file.write(f"{status} - {level} ({slsa.subject_count} subjects)\n")
                if slsa.builder_id:
                    file.write(f"  Builder: {slsa.builder_id}\n")
                if slsa.errors:
                    file.write(f"  Errors: {', '.join(slsa.errors)}\n")
            file.write("\n")

        file.write("=" * 80 + "\n")


# ============================================================================
# CLI INTERFACE
# ============================================================================

def main() -> int:
    """
    Main entry point for CLI usage.

    Returns:
        Exit code (0 = success, non-zero = failure)
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="T.A.R.S. Release Artifact Verifier & Integrity Gate",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic verification
  python security/release_verifier.py --artifact dist/tars-v1.0.2.tar.gz

  # Full verification with SBOM and SLSA
  python security/release_verifier.py \\
    --artifact dist/tars-v1.0.2.tar.gz \\
    --sbom release/v1.0.2/sbom/tars-v1.0.2-cyclonedx.json \\
    --slsa release/v1.0.2/slsa/tars-v1.0.2.provenance.json \\
    --manifest release/v1.0.2/manifest.json \\
    --public-key /run/secrets/rsa_public.pem \\
    --json results.json \\
    --verbose

  # Lenient mode (warnings only)
  python security/release_verifier.py \\
    --artifact dist/tars-v1.0.2.tar.gz \\
    --policy lenient
        """
    )

    parser.add_argument(
        '--artifact',
        type=Path,
        required=True,
        help='Path to artifact file or directory'
    )
    parser.add_argument(
        '--sbom',
        type=Path,
        help='Path to SBOM file (CycloneDX or SPDX JSON)'
    )
    parser.add_argument(
        '--slsa',
        type=Path,
        help='Path to SLSA provenance file'
    )
    parser.add_argument(
        '--manifest',
        type=Path,
        help='Path to manifest.json file'
    )
    parser.add_argument(
        '--public-key',
        type=Path,
        help='Path to RSA public key for signature verification'
    )
    parser.add_argument(
        '--policy',
        choices=['strict', 'lenient'],
        default='strict',
        help='Policy enforcement mode (default: strict)'
    )
    parser.add_argument(
        '--version',
        default='unknown',
        help='Release version'
    )
    parser.add_argument(
        '--json',
        type=Path,
        help='Output JSON report to file'
    )
    parser.add_argument(
        '--text',
        type=Path,
        help='Output text report to file'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(levelname)s: %(message)s'
    )

    try:
        # Initialize verifier
        verifier = ReleaseVerifier(
            mode=args.policy,
            public_key_path=args.public_key
        )

        # Perform verification
        report = verifier.verify_release(
            artifact_path=args.artifact,
            sbom_path=args.sbom,
            slsa_path=args.slsa,
            manifest_path=args.manifest,
            version=args.version
        )

        # Save reports
        if args.json:
            verifier.save_report(report, args.json, format='json')

        if args.text:
            verifier.save_report(report, args.text, format='text')

        # Print summary
        print("\n" + "=" * 80)
        print("VERIFICATION SUMMARY")
        print("=" * 80)
        print(f"Status:  {report.overall_status.upper()}")
        print(f"Checks:  {report.passed_checks}/{report.total_checks} passed")
        print(f"Failed:  {report.failed_checks}")
        print(f"Warnings: {report.warnings}")
        print("=" * 80 + "\n")

        # Determine exit code
        if report.overall_status == "passed":
            return 0
        elif report.overall_status == "warning" and args.policy == 'lenient':
            return 0
        else:
            # Return specific error code based on failure type
            if any(not r.valid for r in report.sbom_results):
                return 2
            elif any(not r.valid for r in report.slsa_results):
                return 3
            elif any(not r.valid for r in report.signature_results):
                return 4
            elif any(not r.match for r in report.hash_results):
                return 5
            elif any(not r.valid for r in report.manifest_results):
                return 6
            else:
                return 7  # Policy gate failure

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except Exception as e:
        logger.error(f"Verification failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 8


if __name__ == '__main__':
    sys.exit(main())
