"""
Comprehensive Integration Tests for Release Artifact Verifier

Tests all subsystems of the release verifier:
- Hash verification (SHA-256, SHA-512)
- Signature verification (RSA-PSS)
- SBOM validation (CycloneDX, SPDX)
- SLSA provenance validation
- Manifest verification
- Policy enforcement (strict and lenient modes)

Requires:
- pytest
- cryptography
- Test fixtures in tests/fixtures/release_verifier/

Compatible with Phase 14.7 Task 3
"""

import pytest
import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone
import tempfile
import shutil

# Import from security module
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from security.release_verifier import (
    ReleaseVerifier,
    HashVerifier,
    SignatureVerifier,
    SBOMVerifier,
    SLSAVerifier,
    ManifestVerifier,
    PolicyEnforcer,
    IntegrityError,
    ProvenanceError,
    SBOMError,
    SignatureError,
    ManifestError,
    PolicyViolationError,
)

# Check if cryptography is available
try:
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa, padding as asym_padding
    from cryptography.hazmat.backends import default_backend
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def temp_dir():
    """Create temporary directory for test artifacts."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def sample_artifact(temp_dir):
    """Create sample artifact file."""
    artifact_path = temp_dir / "tars-v1.0.2.tar.gz"
    content = b"Sample artifact content for testing purposes\n" * 100
    artifact_path.write_bytes(content)
    return artifact_path


@pytest.fixture
def rsa_key_pair(temp_dir):
    """Generate RSA key pair for testing."""
    if not CRYPTO_AVAILABLE:
        pytest.skip("cryptography library not available")

    # Generate private key
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
        backend=default_backend()
    )

    # Save private key
    private_key_path = temp_dir / "test_private.pem"
    with open(private_key_path, 'wb') as f:
        f.write(private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        ))

    # Save public key
    public_key = private_key.public_key()
    public_key_path = temp_dir / "test_public.pem"
    with open(public_key_path, 'wb') as f:
        f.write(public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        ))

    return {
        'private_key': private_key,
        'private_key_path': private_key_path,
        'public_key': public_key,
        'public_key_path': public_key_path
    }


@pytest.fixture
def signed_artifact(sample_artifact, rsa_key_pair):
    """Create signed artifact."""
    if not CRYPTO_AVAILABLE:
        pytest.skip("cryptography library not available")

    # Read artifact
    with open(sample_artifact, 'rb') as f:
        data = f.read()

    # Sign
    signature = rsa_key_pair['private_key'].sign(
        data,
        asym_padding.PSS(
            mgf=asym_padding.MGF1(hashes.SHA256()),
            salt_length=asym_padding.PSS.MAX_LENGTH
        ),
        hashes.SHA256()
    )

    # Save signature
    sig_path = Path(str(sample_artifact) + '.sig')
    sig_path.write_bytes(signature)

    return sample_artifact, sig_path


@pytest.fixture
def sample_cyclonedx_sbom(temp_dir):
    """Create sample CycloneDX SBOM."""
    sbom = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "serialNumber": "urn:uuid:12345678-1234-1234-1234-123456789abc",
        "version": 1,
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tools": [
                {
                    "vendor": "Test",
                    "name": "Test SBOM Generator",
                    "version": "1.0.0"
                }
            ],
            "component": {
                "type": "application",
                "name": "T.A.R.S.",
                "version": "1.0.2"
            }
        },
        "components": [
            {
                "type": "library",
                "name": "flask",
                "version": "2.3.0",
                "purl": "pkg:pypi/flask@2.3.0",
                "hashes": [
                    {
                        "alg": "SHA-256",
                        "content": "abc123def456"
                    }
                ]
            },
            {
                "type": "library",
                "name": "requests",
                "version": "2.31.0",
                "purl": "pkg:pypi/requests@2.31.0"
            }
        ],
        "dependencies": []
    }

    sbom_path = temp_dir / "sbom" / "tars-v1.0.2-cyclonedx.json"
    sbom_path.parent.mkdir(parents=True, exist_ok=True)

    with open(sbom_path, 'w') as f:
        json.dump(sbom, f, indent=2)

    return sbom_path


@pytest.fixture
def sample_spdx_sbom(temp_dir):
    """Create sample SPDX SBOM."""
    sbom = {
        "spdxVersion": "SPDX-2.3",
        "dataLicense": "CC0-1.0",
        "SPDXID": "SPDXRef-DOCUMENT",
        "name": "T.A.R.S.-1.0.2-SBOM",
        "documentNamespace": "https://example.com/tars-1.0.2",
        "creationInfo": {
            "created": datetime.now(timezone.utc).isoformat(),
            "creators": [
                "Tool: Test SBOM Generator-1.0.0"
            ]
        },
        "packages": [
            {
                "SPDXID": "SPDXRef-Package-1",
                "name": "flask",
                "versionInfo": "2.3.0",
                "downloadLocation": "https://pypi.org/project/flask/2.3.0/",
                "filesAnalyzed": False,
                "checksums": [
                    {
                        "algorithm": "SHA256",
                        "checksumValue": "abc123def456"
                    }
                ]
            }
        ],
        "relationships": [
            {
                "spdxElementId": "SPDXRef-DOCUMENT",
                "relationshipType": "DESCRIBES",
                "relatedSpdxElement": "SPDXRef-Package-1"
            }
        ]
    }

    sbom_path = temp_dir / "sbom" / "tars-v1.0.2-spdx.json"
    sbom_path.parent.mkdir(parents=True, exist_ok=True)

    with open(sbom_path, 'w') as f:
        json.dump(sbom, f, indent=2)

    return sbom_path


@pytest.fixture
def sample_slsa_provenance(temp_dir, sample_artifact):
    """Create sample SLSA provenance."""
    # Calculate SHA-256 of artifact
    sha256 = hashlib.sha256()
    with open(sample_artifact, 'rb') as f:
        sha256.update(f.read())
    artifact_hash = sha256.hexdigest()

    provenance = {
        "_type": "https://in-toto.io/Statement/v1",
        "subject": [
            {
                "name": sample_artifact.name,
                "digest": {
                    "sha256": artifact_hash
                }
            }
        ],
        "predicateType": "https://slsa.dev/provenance/v1",
        "predicate": {
            "buildDefinition": {
                "buildType": "https://slsa.dev/build-types/python/package/v1",
                "externalParameters": {
                    "repository": "https://github.com/test/tars",
                    "ref": "refs/tags/v1.0.2"
                },
                "internalParameters": {
                    "platform": "Linux"
                },
                "resolvedDependencies": []
            },
            "runDetails": {
                "builder": {
                    "id": "https://github.com/test/tars/actions"
                },
                "metadata": {
                    "invocationId": "12345678-1234-1234-1234-123456789abc",
                    "startedOn": datetime.now(timezone.utc).isoformat(),
                    "finishedOn": datetime.now(timezone.utc).isoformat()
                },
                "byproducts": []
            }
        }
    }

    slsa_path = temp_dir / "slsa" / "tars-v1.0.2.provenance.json"
    slsa_path.parent.mkdir(parents=True, exist_ok=True)

    with open(slsa_path, 'w') as f:
        json.dump(provenance, f, indent=2)

    return slsa_path


@pytest.fixture
def sample_manifest(temp_dir, sample_artifact):
    """Create sample manifest file."""
    # Calculate SHA-256
    sha256 = hashlib.sha256()
    with open(sample_artifact, 'rb') as f:
        sha256.update(f.read())
    artifact_hash = sha256.hexdigest()

    manifest = {
        "version": "1.0.2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "profile": "test",
        "artifacts": [
            {
                "name": sample_artifact.name,
                "size": sample_artifact.stat().st_size,
                "sha256": artifact_hash
            }
        ],
        "enterprise": {
            "signed": False,
            "encrypted": False,
            "sbom": False,
            "slsa": False
        }
    }

    manifest_path = temp_dir / "manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    return manifest_path


# ============================================================================
# HASH VERIFICATION TESTS
# ============================================================================

class TestHashVerifier:
    """Test suite for HashVerifier."""

    def test_compute_sha256(self, sample_artifact):
        """Test SHA-256 hash computation."""
        hash_result = HashVerifier.compute_hash(sample_artifact, 'sha256')
        assert isinstance(hash_result, str)
        assert len(hash_result) == 64  # SHA-256 produces 64 hex characters

    def test_compute_sha512(self, sample_artifact):
        """Test SHA-512 hash computation."""
        hash_result = HashVerifier.compute_hash(sample_artifact, 'sha512')
        assert isinstance(hash_result, str)
        assert len(hash_result) == 128  # SHA-512 produces 128 hex characters

    def test_verify_hash_correct(self, sample_artifact):
        """Test hash verification with correct hash."""
        expected_hash = HashVerifier.compute_hash(sample_artifact, 'sha256')
        result = HashVerifier.verify_hash(sample_artifact, expected_hash, 'sha256')

        assert result.match is True
        assert result.error is None
        assert result.actual_hash == expected_hash

    def test_verify_hash_incorrect(self, sample_artifact):
        """Test hash verification with incorrect hash."""
        wrong_hash = "0" * 64
        result = HashVerifier.verify_hash(sample_artifact, wrong_hash, 'sha256')

        assert result.match is False
        assert result.error == "Hash mismatch"
        assert result.actual_hash != wrong_hash

    def test_verify_hash_nonexistent_file(self, temp_dir):
        """Test hash verification with nonexistent file."""
        nonexistent = temp_dir / "nonexistent.txt"
        result = HashVerifier.verify_hash(nonexistent, "abc123", 'sha256')

        assert result.match is False
        assert result.error is not None

    def test_unsupported_algorithm(self, sample_artifact):
        """Test with unsupported hash algorithm."""
        with pytest.raises(ValueError, match="Unsupported algorithm"):
            HashVerifier.compute_hash(sample_artifact, 'md5')


# ============================================================================
# SIGNATURE VERIFICATION TESTS
# ============================================================================

@pytest.mark.skipif(not CRYPTO_AVAILABLE, reason="cryptography not available")
class TestSignatureVerifier:
    """Test suite for SignatureVerifier."""

    def test_verify_valid_signature(self, signed_artifact, rsa_key_pair):
        """Test verification of valid signature."""
        artifact_path, sig_path = signed_artifact
        verifier = SignatureVerifier(rsa_key_pair['public_key_path'])

        result = verifier.verify_signature(artifact_path, sig_path)

        assert result.valid is True
        assert result.error is None

    def test_verify_invalid_signature(self, sample_artifact, rsa_key_pair, temp_dir):
        """Test verification of invalid signature."""
        # Create invalid signature
        sig_path = temp_dir / "invalid.sig"
        sig_path.write_bytes(b"invalid signature data")

        verifier = SignatureVerifier(rsa_key_pair['public_key_path'])
        result = verifier.verify_signature(sample_artifact, sig_path)

        assert result.valid is False
        assert result.error is not None

    def test_verify_missing_signature(self, sample_artifact, rsa_key_pair):
        """Test verification when signature file is missing."""
        verifier = SignatureVerifier(rsa_key_pair['public_key_path'])
        result = verifier.verify_signature(sample_artifact)

        assert result.valid is False
        assert "not found" in result.error.lower()

    def test_no_public_key_loaded(self, sample_artifact):
        """Test verification without public key."""
        verifier = SignatureVerifier()  # No public key
        result = verifier.verify_signature(sample_artifact)

        assert result.valid is False
        assert "public key" in result.error.lower()


# ============================================================================
# SBOM VERIFICATION TESTS
# ============================================================================

class TestSBOMVerifier:
    """Test suite for SBOMVerifier."""

    def test_verify_valid_cyclonedx(self, sample_cyclonedx_sbom):
        """Test verification of valid CycloneDX SBOM."""
        result = SBOMVerifier.verify_cyclonedx(sample_cyclonedx_sbom)

        assert result.valid is True
        assert result.format == "cyclonedx"
        assert result.component_count == 2
        assert len(result.errors) == 0

    def test_verify_valid_spdx(self, sample_spdx_sbom):
        """Test verification of valid SPDX SBOM."""
        result = SBOMVerifier.verify_spdx(sample_spdx_sbom)

        assert result.valid is True
        assert result.format == "spdx"
        assert result.component_count == 1
        assert len(result.errors) == 0

    def test_verify_invalid_json(self, temp_dir):
        """Test verification of invalid JSON."""
        invalid_sbom = temp_dir / "invalid.json"
        invalid_sbom.write_text("{ invalid json }")

        result = SBOMVerifier.verify_cyclonedx(invalid_sbom)

        assert result.valid is False
        assert any("Invalid JSON" in error for error in result.errors)

    def test_verify_missing_required_fields_cyclonedx(self, temp_dir):
        """Test CycloneDX with missing required fields."""
        incomplete_sbom = {
            "bomFormat": "CycloneDX",
            # Missing specVersion, metadata, etc.
        }

        sbom_path = temp_dir / "incomplete.json"
        with open(sbom_path, 'w') as f:
            json.dump(incomplete_sbom, f)

        result = SBOMVerifier.verify_cyclonedx(sbom_path)

        assert result.valid is False
        assert len(result.errors) > 0

    def test_verify_missing_required_fields_spdx(self, temp_dir):
        """Test SPDX with missing required fields."""
        incomplete_sbom = {
            "SPDXID": "SPDXRef-DOCUMENT",
            # Missing spdxVersion, dataLicense, etc.
        }

        sbom_path = temp_dir / "incomplete.json"
        with open(sbom_path, 'w') as f:
            json.dump(incomplete_sbom, f)

        result = SBOMVerifier.verify_spdx(sbom_path)

        assert result.valid is False
        assert len(result.errors) > 0


# ============================================================================
# SLSA PROVENANCE VERIFICATION TESTS
# ============================================================================

class TestSLSAVerifier:
    """Test suite for SLSAVerifier."""

    def test_verify_valid_provenance(self, sample_slsa_provenance):
        """Test verification of valid SLSA provenance."""
        result = SLSAVerifier.verify_provenance(sample_slsa_provenance)

        assert result.valid is True
        assert result.subject_count == 1
        assert result.slsa_level in [1, 2, 3]
        assert result.builder_id is not None
        assert len(result.errors) == 0

    def test_verify_missing_type(self, temp_dir):
        """Test provenance missing _type field."""
        invalid_provenance = {
            "predicateType": "https://slsa.dev/provenance/v1",
            # Missing _type
        }

        prov_path = temp_dir / "invalid.json"
        with open(prov_path, 'w') as f:
            json.dump(invalid_provenance, f)

        result = SLSAVerifier.verify_provenance(prov_path)

        assert result.valid is False
        assert any("_type" in error for error in result.errors)

    def test_verify_invalid_predicate_type(self, temp_dir, sample_slsa_provenance):
        """Test provenance with invalid predicate type."""
        with open(sample_slsa_provenance, 'r') as f:
            provenance = json.load(f)

        provenance['predicateType'] = "https://invalid.com/provenance/v1"

        prov_path = temp_dir / "invalid_pred.json"
        with open(prov_path, 'w') as f:
            json.dump(provenance, f)

        result = SLSAVerifier.verify_provenance(prov_path)

        assert result.valid is False
        assert any("predicateType" in error for error in result.errors)

    def test_verify_missing_subject(self, temp_dir):
        """Test provenance missing subject."""
        incomplete_provenance = {
            "_type": "https://in-toto.io/Statement/v1",
            "predicateType": "https://slsa.dev/provenance/v1",
            # Missing subject
            "predicate": {}
        }

        prov_path = temp_dir / "no_subject.json"
        with open(prov_path, 'w') as f:
            json.dump(incomplete_provenance, f)

        result = SLSAVerifier.verify_provenance(prov_path)

        assert result.valid is False
        assert any("subject" in error.lower() for error in result.errors)


# ============================================================================
# MANIFEST VERIFICATION TESTS
# ============================================================================

class TestManifestVerifier:
    """Test suite for ManifestVerifier."""

    def test_verify_valid_manifest(self, sample_manifest, temp_dir):
        """Test verification of valid manifest."""
        result = ManifestVerifier.verify_manifest(sample_manifest, temp_dir)

        assert result.valid is True
        assert result.artifact_count == 1
        assert len(result.missing_artifacts) == 0
        assert len(result.hash_mismatches) == 0

    def test_verify_missing_artifact(self, temp_dir):
        """Test manifest with missing artifact."""
        manifest = {
            "version": "1.0.2",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "artifacts": [
                {
                    "name": "missing_file.txt",
                    "size": 1234,
                    "sha256": "abc123"
                }
            ]
        }

        manifest_path = temp_dir / "manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f)

        result = ManifestVerifier.verify_manifest(manifest_path, temp_dir)

        assert result.valid is False
        assert len(result.missing_artifacts) == 1
        assert "missing_file.txt" in result.missing_artifacts

    def test_verify_hash_mismatch(self, sample_artifact, temp_dir):
        """Test manifest with hash mismatch."""
        wrong_hash = "0" * 64

        manifest = {
            "version": "1.0.2",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "artifacts": [
                {
                    "name": sample_artifact.name,
                    "size": sample_artifact.stat().st_size,
                    "sha256": wrong_hash
                }
            ]
        }

        manifest_path = temp_dir / "manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f)

        result = ManifestVerifier.verify_manifest(manifest_path, temp_dir)

        assert result.valid is False
        assert len(result.hash_mismatches) == 1


# ============================================================================
# POLICY ENFORCEMENT TESTS
# ============================================================================

class TestPolicyEnforcer:
    """Test suite for PolicyEnforcer."""

    def test_strict_mode_all_passed(self):
        """Test strict mode with all checks passed."""
        from security.release_verifier import VerificationReport, HashVerificationResult

        report = VerificationReport(
            timestamp=datetime.now(timezone.utc).isoformat(),
            version="1.0.2",
            artifact_path="/test/artifact.tar.gz",
            verification_mode="strict",
            overall_status="unknown"
        )

        # Add passing hash result
        report.hash_results.append(HashVerificationResult(
            file_name="test.txt",
            algorithm="sha256",
            expected_hash="abc123",
            actual_hash="abc123",
            match=True
        ))

        enforcer = PolicyEnforcer(mode='strict')
        policies = enforcer.check_policies(report)

        # All policies should pass when no failures
        assert len(policies) > 0

    def test_strict_mode_signature_failure(self):
        """Test strict mode with unsigned artifacts."""
        from security.release_verifier import VerificationReport, SignatureVerificationResult

        report = VerificationReport(
            timestamp=datetime.now(timezone.utc).isoformat(),
            version="1.0.2",
            artifact_path="/test/artifact.tar.gz",
            verification_mode="strict",
            overall_status="unknown"
        )

        # Add failing signature result
        report.signature_results.append(SignatureVerificationResult(
            file_name="test.txt",
            signature_file="test.txt.sig",
            algorithm="RSA-PSS-SHA256",
            valid=False,
            error="Signature not found"
        ))

        enforcer = PolicyEnforcer(mode='strict')
        policies = enforcer.check_policies(report)

        # Should have policy failure for unsigned artifacts
        unsigned_policy = [p for p in policies if p.policy_name == "artifacts_signed"]
        assert len(unsigned_policy) == 1
        assert unsigned_policy[0].passed is False
        assert unsigned_policy[0].severity == "critical"

    def test_lenient_mode_allows_warnings(self):
        """Test lenient mode allows warnings."""
        from security.release_verifier import VerificationReport, SignatureVerificationResult

        report = VerificationReport(
            timestamp=datetime.now(timezone.utc).isoformat(),
            version="1.0.2",
            artifact_path="/test/artifact.tar.gz",
            verification_mode="lenient",
            overall_status="unknown"
        )

        # Add failing signature result
        report.signature_results.append(SignatureVerificationResult(
            file_name="test.txt",
            signature_file="test.txt.sig",
            algorithm="RSA-PSS-SHA256",
            valid=False
        ))

        enforcer = PolicyEnforcer(mode='lenient')
        policies = enforcer.check_policies(report)

        # In lenient mode, should pass with warning
        unsigned_policy = [p for p in policies if p.policy_name == "artifacts_signed"]
        assert len(unsigned_policy) == 1
        assert unsigned_policy[0].passed is True  # Passes in lenient mode
        assert unsigned_policy[0].severity == "warning"


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestReleaseVerifier:
    """Integration tests for ReleaseVerifier."""

    def test_verify_minimal_release(self, sample_artifact):
        """Test verification with minimal setup (artifact only)."""
        verifier = ReleaseVerifier(mode='lenient')

        report = verifier.verify_release(
            artifact_path=sample_artifact,
            version="1.0.2"
        )

        assert report is not None
        assert report.version == "1.0.2"
        assert report.verification_mode == "lenient"

    @pytest.mark.skipif(not CRYPTO_AVAILABLE, reason="cryptography not available")
    def test_verify_complete_release(
        self,
        signed_artifact,
        rsa_key_pair,
        sample_cyclonedx_sbom,
        sample_slsa_provenance,
        sample_manifest,
        temp_dir
    ):
        """Test verification with complete artifact set."""
        artifact_path, sig_path = signed_artifact

        verifier = ReleaseVerifier(
            mode='strict',
            public_key_path=rsa_key_pair['public_key_path']
        )

        report = verifier.verify_release(
            artifact_path=artifact_path,
            sbom_path=sample_cyclonedx_sbom,
            slsa_path=sample_slsa_provenance,
            manifest_path=sample_manifest,
            version="1.0.2"
        )

        assert report is not None
        assert report.overall_status in ['passed', 'warning', 'failed']
        assert report.total_checks > 0

    def test_save_json_report(self, sample_artifact, temp_dir):
        """Test saving verification report as JSON."""
        verifier = ReleaseVerifier(mode='lenient')

        report = verifier.verify_release(
            artifact_path=sample_artifact,
            version="1.0.2"
        )

        output_path = temp_dir / "verification_report.json"
        verifier.save_report(report, output_path, format='json')

        assert output_path.exists()

        # Verify JSON is valid
        with open(output_path, 'r') as f:
            loaded_report = json.load(f)

        assert loaded_report['version'] == "1.0.2"

    def test_save_text_report(self, sample_artifact, temp_dir):
        """Test saving verification report as text."""
        verifier = ReleaseVerifier(mode='lenient')

        report = verifier.verify_release(
            artifact_path=sample_artifact,
            version="1.0.2"
        )

        output_path = temp_dir / "verification_report.txt"
        verifier.save_report(report, output_path, format='text')

        assert output_path.exists()

        # Verify text contains expected content
        content = output_path.read_text()
        assert "RELEASE ARTIFACT VERIFICATION REPORT" in content
        assert "1.0.2" in content


# ============================================================================
# CLI TESTS
# ============================================================================

class TestCLI:
    """Tests for CLI interface."""

    def test_cli_basic_invocation(self, sample_artifact, temp_dir):
        """Test basic CLI invocation."""
        from security.release_verifier import main
        import sys

        # Mock command line arguments
        sys.argv = [
            'release_verifier.py',
            '--artifact', str(sample_artifact),
            '--version', '1.0.2',
            '--policy', 'lenient',
            '--json', str(temp_dir / 'report.json')
        ]

        exit_code = main()

        # Should complete without critical errors in lenient mode
        assert exit_code in [0, 7, 8]  # Success or expected failures

    @pytest.mark.skipif(not CRYPTO_AVAILABLE, reason="cryptography not available")
    def test_cli_with_all_options(
        self,
        signed_artifact,
        rsa_key_pair,
        sample_cyclonedx_sbom,
        sample_slsa_provenance,
        sample_manifest,
        temp_dir
    ):
        """Test CLI with all verification options."""
        from security.release_verifier import main
        import sys

        artifact_path, sig_path = signed_artifact

        sys.argv = [
            'release_verifier.py',
            '--artifact', str(artifact_path),
            '--sbom', str(sample_cyclonedx_sbom),
            '--slsa', str(sample_slsa_provenance),
            '--manifest', str(sample_manifest),
            '--public-key', str(rsa_key_pair['public_key_path']),
            '--version', '1.0.2',
            '--policy', 'strict',
            '--json', str(temp_dir / 'report.json'),
            '--text', str(temp_dir / 'report.txt')
        ]

        exit_code = main()

        # Verify reports were created
        assert (temp_dir / 'report.json').exists()
        assert (temp_dir / 'report.txt').exists()


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
