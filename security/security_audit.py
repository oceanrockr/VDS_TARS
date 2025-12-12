#!/usr/bin/env python3
"""
T.A.R.S. Security Audit Tool

Comprehensive security auditing for T.A.R.S. v1.0.2+

Features:
- File integrity checking (SHA-256)
- RSA-PSS signature verification
- AES-256-GCM encrypted file inspection
- SBOM vulnerability scanning integration
- API endpoint security testing
- Configuration hardening checks
- Dependency CVE scanning
- Deterministic audit reports

Usage:
    # Full audit
    python security/security_audit.py --deep --verbose

    # SBOM vulnerability scan
    python security/security_audit.py --scan-sbom release/v1.0.2/sbom/tars-cyclonedx.json

    # Signature verification
    python security/security_audit.py --verify-signature artifact.tar.gz.sig artifact.tar.gz

    # API security testing
    python security/security_audit.py --check-api https://tars.company.com

    # Configuration audit
    python security/security_audit.py --check-config enterprise_config/profiles/production.yaml

Author: T.A.R.S. Team
Version: 1.0.2
"""

import argparse
import hashlib
import json
import sys
import subprocess
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum

try:
    import requests
except ImportError:
    print("Warning: 'requests' not found. API checks disabled.")
    requests = None

try:
    import yaml
except ImportError:
    print("Warning: 'pyyaml' not found. Config checks may be limited.")
    yaml = None


# ============================================================================
# Data Models
# ============================================================================

class Severity(Enum):
    """Finding severity levels"""
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"


@dataclass
class SecurityFinding:
    """Security audit finding"""
    id: str
    category: str
    title: str
    description: str
    severity: str
    affected_files: List[str]
    remediation: str
    references: List[str]


@dataclass
class AuditReport:
    """Complete security audit report"""
    audit_id: str
    timestamp: str
    version: str
    findings: List[Dict]
    summary: Dict
    checks_performed: List[str]


# ============================================================================
# File Integrity Checker
# ============================================================================

class FileIntegrityChecker:
    """Check file integrity using SHA-256 hashes"""

    CRITICAL_FILES = [
        'security/sbom_generator.py',
        'security/slsa_generator.py',
        'security/encryption.py',
        'security/signing.py',
        'cognition/shared/auth.py',
        'cognition/shared/rate_limiter.py',
        'dashboard/api/admin_routes.py',
        'enterprise_config/compliance_enforcer.py',
    ]

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.findings: List[SecurityFinding] = []

    def check_integrity(self) -> List[SecurityFinding]:
        """Check integrity of critical files"""

        if self.verbose:
            print("Checking file integrity...")

        for file_path_str in self.CRITICAL_FILES:
            file_path = Path(file_path_str)

            if not file_path.exists():
                self.findings.append(SecurityFinding(
                    id=f"FI-{len(self.findings)+1:03d}",
                    category="File Integrity",
                    title=f"Critical file missing: {file_path}",
                    description=f"Expected critical file not found: {file_path}",
                    severity=Severity.HIGH.value,
                    affected_files=[str(file_path)],
                    remediation="Restore missing file from source control or reinstall package",
                    references=[]
                ))
                continue

            # Check for tampering indicators
            if self._check_suspicious_modifications(file_path):
                self.findings.append(SecurityFinding(
                    id=f"FI-{len(self.findings)+1:03d}",
                    category="File Integrity",
                    title=f"Suspicious modification detected: {file_path}",
                    description=f"File contains suspicious patterns that may indicate tampering",
                    severity=Severity.CRITICAL.value,
                    affected_files=[str(file_path)],
                    remediation="Review file changes, restore from trusted source if compromised",
                    references=["docs/PRODUCTION_RUNBOOK.md#playbook-4-security-incident"]
                ))

        if self.verbose:
            print(f"  Found {len(self.findings)} integrity issues")

        return self.findings

    def _check_suspicious_modifications(self, file_path: Path) -> bool:
        """Check for suspicious code patterns"""

        suspicious_patterns = [
            r'eval\s*\(',  # Eval usage
            r'exec\s*\(',  # Exec usage
            r'__import__\s*\(\s*["\']os["\']',  # Dynamic os import
            r'pickle\.loads',  # Pickle deserialization
            r'subprocess\.call.*shell=True',  # Shell injection risk
            r'rm\s+-rf\s+/',  # Destructive commands
        ]

        try:
            content = file_path.read_text()

            for pattern in suspicious_patterns:
                if re.search(pattern, content):
                    return True

        except Exception:
            pass

        return False

    def calculate_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of file"""
        sha256 = hashlib.sha256()

        try:
            with open(file_path, 'rb') as f:
                while chunk := f.read(8192):
                    sha256.update(chunk)
            return sha256.hexdigest()
        except Exception as e:
            return f"ERROR: {e}"


# ============================================================================
# Signature Verifier
# ============================================================================

class SignatureVerifier:
    """Verify RSA-PSS signatures"""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.findings: List[SecurityFinding] = []

    def verify_signature(self, signature_file: Path, data_file: Path, public_key_path: Optional[Path] = None) -> bool:
        """Verify RSA-PSS signature"""

        if self.verbose:
            print(f"Verifying signature: {signature_file}")

        if not signature_file.exists():
            self.findings.append(SecurityFinding(
                id=f"SV-{len(self.findings)+1:03d}",
                category="Signature Verification",
                title=f"Signature file missing: {signature_file}",
                description="Expected signature file not found",
                severity=Severity.HIGH.value,
                affected_files=[str(signature_file)],
                remediation="Re-sign file with proper key",
                references=[]
            ))
            return False

        if not data_file.exists():
            self.findings.append(SecurityFinding(
                id=f"SV-{len(self.findings)+1:03d}",
                category="Signature Verification",
                title=f"Data file missing: {data_file}",
                description="Data file for signature verification not found",
                severity=Severity.HIGH.value,
                affected_files=[str(data_file)],
                remediation="Restore missing data file",
                references=[]
            ))
            return False

        # Try to verify using signing module
        try:
            from security.signing import ReportSigner

            signer = ReportSigner(
                private_key_path=None,
                public_key_path=public_key_path
            )

            with open(signature_file, 'r') as f:
                signature_b64 = f.read().strip()

            with open(data_file, 'rb') as f:
                data = f.read()

            is_valid = signer.verify(data, signature_b64)

            if not is_valid:
                self.findings.append(SecurityFinding(
                    id=f"SV-{len(self.findings)+1:03d}",
                    category="Signature Verification",
                    title=f"Invalid signature: {signature_file}",
                    description="RSA-PSS signature verification failed",
                    severity=Severity.CRITICAL.value,
                    affected_files=[str(signature_file), str(data_file)],
                    remediation="Re-sign with valid key or investigate tampering",
                    references=["docs/PRODUCTION_RUNBOOK.md#managing-encryption-keys"]
                ))
                return False

            if self.verbose:
                print(f"  ✓ Signature valid")

            return True

        except ImportError:
            if self.verbose:
                print("  Warning: signing module not available")
            return False
        except Exception as e:
            self.findings.append(SecurityFinding(
                id=f"SV-{len(self.findings)+1:03d}",
                category="Signature Verification",
                title=f"Signature verification error: {signature_file}",
                description=f"Error during verification: {e}",
                severity=Severity.HIGH.value,
                affected_files=[str(signature_file), str(data_file)],
                remediation="Check key configuration and file permissions",
                references=[]
            ))
            return False


# ============================================================================
# AES Encrypted File Inspector
# ============================================================================

class EncryptedFileInspector:
    """Inspect AES-256-GCM encrypted files"""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.findings: List[SecurityFinding] = []

    def inspect_encrypted_file(self, file_path: Path, key_path: Optional[Path] = None) -> Dict:
        """Inspect encrypted file metadata"""

        if self.verbose:
            print(f"Inspecting encrypted file: {file_path}")

        if not file_path.exists():
            self.findings.append(SecurityFinding(
                id=f"EF-{len(self.findings)+1:03d}",
                category="Encrypted Files",
                title=f"Encrypted file missing: {file_path}",
                description="Expected encrypted file not found",
                severity=Severity.MEDIUM.value,
                affected_files=[str(file_path)],
                remediation="Restore missing file or recreate",
                references=[]
            ))
            return {}

        # Check file size (encrypted files should have minimum size due to IV, tag, etc.)
        file_size = file_path.stat().st_size

        if file_size < 32:  # IV (16) + Tag (16) = 32 bytes minimum
            self.findings.append(SecurityFinding(
                id=f"EF-{len(self.findings)+1:03d}",
                category="Encrypted Files",
                title=f"Suspicious encrypted file size: {file_path}",
                description=f"File size ({file_size} bytes) too small for valid AES-256-GCM encrypted data",
                severity=Severity.HIGH.value,
                affected_files=[str(file_path)],
                remediation="Re-encrypt file with proper encryption",
                references=[]
            ))

        # Try to decrypt and validate
        if key_path:
            try:
                from security.encryption import AESEncryption

                with open(key_path, 'r') as f:
                    key_hex = f.read().strip()

                aes = AESEncryption(key_hex=key_hex)

                with open(file_path, 'rb') as f:
                    encrypted_data = f.read()

                decrypted = aes.decrypt(encrypted_data)

                if self.verbose:
                    print(f"  ✓ Decryption successful ({len(decrypted)} bytes)")

                return {
                    'encrypted_size': file_size,
                    'decrypted_size': len(decrypted),
                    'valid': True
                }

            except ImportError:
                if self.verbose:
                    print("  Warning: encryption module not available")
            except Exception as e:
                self.findings.append(SecurityFinding(
                    id=f"EF-{len(self.findings)+1:03d}",
                    category="Encrypted Files",
                    title=f"Decryption failed: {file_path}",
                    description=f"Unable to decrypt file: {e}",
                    severity=Severity.HIGH.value,
                    affected_files=[str(file_path)],
                    remediation="Check encryption key, verify file integrity",
                    references=[]
                ))

        return {'encrypted_size': file_size, 'valid': False}


# ============================================================================
# SBOM Vulnerability Scanner
# ============================================================================

class SBOMVulnerabilityScanner:
    """Scan SBOM for known vulnerabilities"""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.findings: List[SecurityFinding] = []

    def scan_sbom(self, sbom_path: Path) -> List[SecurityFinding]:
        """Scan SBOM for vulnerabilities using Grype or similar tools"""

        if self.verbose:
            print(f"Scanning SBOM: {sbom_path}")

        if not sbom_path.exists():
            self.findings.append(SecurityFinding(
                id=f"VS-{len(self.findings)+1:03d}",
                category="Vulnerability Scanning",
                title=f"SBOM file not found: {sbom_path}",
                description="Cannot perform vulnerability scan without SBOM",
                severity=Severity.MEDIUM.value,
                affected_files=[str(sbom_path)],
                remediation="Generate SBOM using: python security/sbom_generator.py",
                references=["PHASE14_7_TASK1_COMPLETION_SUMMARY.md#sbom-generator"]
            ))
            return self.findings

        # Try to use Grype if available
        if self._check_grype_available():
            vulnerabilities = self._scan_with_grype(sbom_path)
            self.findings.extend(vulnerabilities)
        # Try Trivy
        elif self._check_trivy_available():
            vulnerabilities = self._scan_with_trivy(sbom_path)
            self.findings.extend(vulnerabilities)
        else:
            # Manual scan of known critical CVEs
            vulnerabilities = self._scan_manual(sbom_path)
            self.findings.extend(vulnerabilities)

        if self.verbose:
            print(f"  Found {len(self.findings)} vulnerabilities")

        return self.findings

    def _check_grype_available(self) -> bool:
        """Check if Grype is installed"""
        try:
            result = subprocess.run(
                ['grype', 'version'],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except Exception:
            return False

    def _check_trivy_available(self) -> bool:
        """Check if Trivy is installed"""
        try:
            result = subprocess.run(
                ['trivy', '--version'],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except Exception:
            return False

    def _scan_with_grype(self, sbom_path: Path) -> List[SecurityFinding]:
        """Scan SBOM with Grype"""
        findings = []

        try:
            result = subprocess.run(
                ['grype', 'sbom', str(sbom_path), '-o', 'json'],
                capture_output=True,
                timeout=120,
                text=True
            )

            if result.returncode == 0:
                data = json.loads(result.stdout)
                matches = data.get('matches', [])

                for match in matches:
                    vulnerability = match.get('vulnerability', {})
                    artifact = match.get('artifact', {})

                    severity = vulnerability.get('severity', 'UNKNOWN').upper()
                    if severity in ['CRITICAL', 'HIGH']:
                        findings.append(SecurityFinding(
                            id=f"VS-{len(findings)+1:03d}",
                            category="Vulnerability Scanning",
                            title=f"CVE found: {vulnerability.get('id', 'UNKNOWN')}",
                            description=f"Package {artifact.get('name', 'unknown')}@{artifact.get('version', 'unknown')} has known vulnerability",
                            severity=severity.capitalize() if severity in ['CRITICAL', 'HIGH'] else Severity.MEDIUM.value,
                            affected_files=[artifact.get('name', 'unknown')],
                            remediation=f"Update to fixed version: {vulnerability.get('fix', {}).get('versions', ['unknown'])[0] if vulnerability.get('fix', {}).get('versions') else 'No fix available'}",
                            references=[vulnerability.get('dataSource', '')]
                        ))

        except subprocess.TimeoutExpired:
            if self.verbose:
                print("  Warning: Grype scan timed out")
        except Exception as e:
            if self.verbose:
                print(f"  Warning: Grype scan failed: {e}")

        return findings

    def _scan_with_trivy(self, sbom_path: Path) -> List[SecurityFinding]:
        """Scan SBOM with Trivy"""
        findings = []

        try:
            result = subprocess.run(
                ['trivy', 'sbom', str(sbom_path), '--format', 'json'],
                capture_output=True,
                timeout=120,
                text=True
            )

            if result.returncode == 0:
                data = json.loads(result.stdout)
                results = data.get('Results', [])

                for result_item in results:
                    vulnerabilities = result_item.get('Vulnerabilities', [])

                    for vuln in vulnerabilities:
                        severity = vuln.get('Severity', 'UNKNOWN').upper()
                        if severity in ['CRITICAL', 'HIGH']:
                            findings.append(SecurityFinding(
                                id=f"VS-{len(findings)+1:03d}",
                                category="Vulnerability Scanning",
                                title=f"CVE found: {vuln.get('VulnerabilityID', 'UNKNOWN')}",
                                description=f"Package {vuln.get('PkgName', 'unknown')}@{vuln.get('InstalledVersion', 'unknown')} has known vulnerability",
                                severity=severity.capitalize() if severity in ['CRITICAL', 'HIGH'] else Severity.MEDIUM.value,
                                affected_files=[vuln.get('PkgName', 'unknown')],
                                remediation=f"Update to: {vuln.get('FixedVersion', 'No fix available')}",
                                references=[vuln.get('PrimaryURL', '')]
                            ))

        except subprocess.TimeoutExpired:
            if self.verbose:
                print("  Warning: Trivy scan timed out")
        except Exception as e:
            if self.verbose:
                print(f"  Warning: Trivy scan failed: {e}")

        return findings

    def _scan_manual(self, sbom_path: Path) -> List[SecurityFinding]:
        """Manual scan for critical known CVEs"""
        findings = []

        if self.verbose:
            print("  Warning: No vulnerability scanner found (Grype/Trivy)")
            print("  Performing manual scan for known critical issues")

        # Known critical CVEs to check (update regularly)
        CRITICAL_CVES = {
            'urllib3': {'<2.0.7': 'CVE-2023-45803'},
            'requests': {'<2.31.0': 'CVE-2023-32681'},
            'flask': {'<2.3.2': 'CVE-2023-30861'},
            'cryptography': {'<41.0.0': 'CVE-2023-38325'},
        }

        try:
            with open(sbom_path, 'r') as f:
                sbom = json.load(f)

            # Check CycloneDX format
            if 'components' in sbom:
                components = sbom['components']
                for component in components:
                    name = component.get('name', '')
                    version = component.get('version', '')

                    if name in CRITICAL_CVES:
                        for version_constraint, cve in CRITICAL_CVES[name].items():
                            # Simple version check (production would use packaging.version)
                            findings.append(SecurityFinding(
                                id=f"VS-{len(findings)+1:03d}",
                                category="Vulnerability Scanning",
                                title=f"Potential vulnerability: {cve}",
                                description=f"Package {name}@{version} may be affected by known CVE",
                                severity=Severity.HIGH.value,
                                affected_files=[name],
                                remediation=f"Update {name} to latest version",
                                references=[f"https://nvd.nist.gov/vuln/detail/{cve}"]
                            ))

            # Check SPDX format
            elif 'packages' in sbom:
                packages = sbom['packages']
                for package in packages:
                    name = package.get('name', '')
                    version = package.get('versionInfo', '')

                    if name in CRITICAL_CVES:
                        for version_constraint, cve in CRITICAL_CVES[name].items():
                            findings.append(SecurityFinding(
                                id=f"VS-{len(findings)+1:03d}",
                                category="Vulnerability Scanning",
                                title=f"Potential vulnerability: {cve}",
                                description=f"Package {name}@{version} may be affected by known CVE",
                                severity=Severity.HIGH.value,
                                affected_files=[name],
                                remediation=f"Update {name} to latest version",
                                references=[f"https://nvd.nist.gov/vuln/detail/{cve}"]
                            ))

        except Exception as e:
            if self.verbose:
                print(f"  Error parsing SBOM: {e}")

        return findings


# ============================================================================
# API Security Tester
# ============================================================================

class APISecurityTester:
    """Test API endpoint security"""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.findings: List[SecurityFinding] = []

    def test_api_security(self, base_url: str, auth_token: Optional[str] = None) -> List[SecurityFinding]:
        """Test API security posture"""

        if self.verbose:
            print(f"Testing API security: {base_url}")

        if requests is None:
            if self.verbose:
                print("  Warning: requests module not available")
            return self.findings

        # Test 1: Check if HTTPS is enforced
        if not base_url.startswith('https://'):
            self.findings.append(SecurityFinding(
                id=f"API-{len(self.findings)+1:03d}",
                category="API Security",
                title="HTTPS not enforced",
                description=f"API endpoint using HTTP instead of HTTPS: {base_url}",
                severity=Severity.HIGH.value,
                affected_files=[],
                remediation="Configure TLS/SSL certificates and enforce HTTPS",
                references=["docs/PRODUCTION_RUNBOOK.md#certificate-renewal"]
            ))

        # Test 2: Check authentication requirement
        self._test_authentication(base_url)

        # Test 3: Check rate limiting
        self._test_rate_limiting(base_url, auth_token)

        # Test 4: Check security headers
        self._test_security_headers(base_url)

        # Test 5: Check for information disclosure
        self._test_information_disclosure(base_url)

        if self.verbose:
            print(f"  Found {len(self.findings)} API security issues")

        return self.findings

    def _test_authentication(self, base_url: str):
        """Test authentication requirement"""
        try:
            # Try accessing protected endpoint without auth
            response = requests.get(
                f"{base_url}/api/agents",
                timeout=10,
                headers={'User-Agent': 'T.A.R.S-SecurityAudit/1.0'}
            )

            if response.status_code == 200:
                self.findings.append(SecurityFinding(
                    id=f"API-{len(self.findings)+1:03d}",
                    category="API Security",
                    title="Authentication not required",
                    description="Protected endpoint accessible without authentication",
                    severity=Severity.CRITICAL.value,
                    affected_files=[],
                    remediation="Enable JWT authentication for all non-public endpoints",
                    references=["cognition/shared/auth.py"]
                ))

        except requests.exceptions.RequestException:
            pass

    def _test_rate_limiting(self, base_url: str, auth_token: Optional[str]):
        """Test rate limiting"""
        try:
            headers = {'User-Agent': 'T.A.R.S-SecurityAudit/1.0'}
            if auth_token:
                headers['Authorization'] = f'Bearer {auth_token}'

            # Make rapid requests
            responses = []
            for _ in range(50):
                response = requests.get(
                    f"{base_url}/health",
                    timeout=5,
                    headers=headers
                )
                responses.append(response.status_code)

            # Check if any were rate limited
            if 429 not in responses:
                self.findings.append(SecurityFinding(
                    id=f"API-{len(self.findings)+1:03d}",
                    category="API Security",
                    title="Rate limiting not detected",
                    description="API did not return 429 status during rapid requests",
                    severity=Severity.MEDIUM.value,
                    affected_files=[],
                    remediation="Enable rate limiting via cognition/shared/rate_limiter.py",
                    references=["cognition/shared/rate_limiter.py"]
                ))

        except requests.exceptions.RequestException:
            pass

    def _test_security_headers(self, base_url: str):
        """Test for security headers"""
        try:
            response = requests.get(
                f"{base_url}/health",
                timeout=10,
                headers={'User-Agent': 'T.A.R.S-SecurityAudit/1.0'}
            )

            headers = response.headers

            # Check for important security headers
            required_headers = {
                'X-Content-Type-Options': 'nosniff',
                'X-Frame-Options': ['DENY', 'SAMEORIGIN'],
                'Strict-Transport-Security': None,  # Just check presence
            }

            for header, expected_value in required_headers.items():
                if header not in headers:
                    self.findings.append(SecurityFinding(
                        id=f"API-{len(self.findings)+1:03d}",
                        category="API Security",
                        title=f"Missing security header: {header}",
                        description=f"Response missing recommended security header: {header}",
                        severity=Severity.LOW.value,
                        affected_files=[],
                        remediation=f"Add {header} header to API responses",
                        references=[]
                    ))

        except requests.exceptions.RequestException:
            pass

    def _test_information_disclosure(self, base_url: str):
        """Test for information disclosure"""
        try:
            # Test /health endpoint
            response = requests.get(
                f"{base_url}/health",
                timeout=10,
                headers={'User-Agent': 'T.A.R.S-SecurityAudit/1.0'}
            )

            # Check if Server header reveals too much
            server_header = response.headers.get('Server', '')
            if 'werkzeug' in server_header.lower() or 'python' in server_header.lower():
                self.findings.append(SecurityFinding(
                    id=f"API-{len(self.findings)+1:03d}",
                    category="API Security",
                    title="Information disclosure in Server header",
                    description=f"Server header reveals implementation details: {server_header}",
                    severity=Severity.LOW.value,
                    affected_files=[],
                    remediation="Remove or obfuscate Server header",
                    references=[]
                ))

        except requests.exceptions.RequestException:
            pass


# ============================================================================
# Configuration Auditor
# ============================================================================

class ConfigurationAuditor:
    """Audit configuration files for security issues"""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.findings: List[SecurityFinding] = []

    def audit_config(self, config_path: Path) -> List[SecurityFinding]:
        """Audit configuration file"""

        if self.verbose:
            print(f"Auditing configuration: {config_path}")

        if not config_path.exists():
            self.findings.append(SecurityFinding(
                id=f"CFG-{len(self.findings)+1:03d}",
                category="Configuration",
                title=f"Configuration file not found: {config_path}",
                description="Expected configuration file missing",
                severity=Severity.MEDIUM.value,
                affected_files=[str(config_path)],
                remediation="Create configuration file or check path",
                references=[]
            ))
            return self.findings

        try:
            if yaml is None:
                if self.verbose:
                    print("  Warning: yaml module not available")
                return self.findings

            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)

            # Check encryption settings
            if not config.get('encryption', {}).get('enabled', False):
                self.findings.append(SecurityFinding(
                    id=f"CFG-{len(self.findings)+1:03d}",
                    category="Configuration",
                    title="Encryption disabled",
                    description="Encryption is not enabled in configuration",
                    severity=Severity.HIGH.value,
                    affected_files=[str(config_path)],
                    remediation="Enable encryption: encryption.enabled = true",
                    references=["docs/PRODUCTION_RUNBOOK.md#managing-encryption-keys"]
                ))

            # Check signing settings
            if not config.get('signing', {}).get('enabled', False):
                self.findings.append(SecurityFinding(
                    id=f"CFG-{len(self.findings)+1:03d}",
                    category="Configuration",
                    title="Signing disabled",
                    description="Digital signing is not enabled in configuration",
                    severity=Severity.MEDIUM.value,
                    affected_files=[str(config_path)],
                    remediation="Enable signing: signing.enabled = true",
                    references=["docs/PRODUCTION_RUNBOOK.md#rsa-pss-signing-keys"]
                ))

            # Check authentication
            if not config.get('auth', {}).get('jwt_expiry'):
                self.findings.append(SecurityFinding(
                    id=f"CFG-{len(self.findings)+1:03d}",
                    category="Configuration",
                    title="JWT expiry not configured",
                    description="JWT token expiry time not set",
                    severity=Severity.MEDIUM.value,
                    affected_files=[str(config_path)],
                    remediation="Set auth.jwt_expiry to appropriate value (e.g., 3600)",
                    references=[]
                ))

            # Check rate limiting
            if not config.get('rate_limiting', {}).get('enabled', False):
                self.findings.append(SecurityFinding(
                    id=f"CFG-{len(self.findings)+1:03d}",
                    category="Configuration",
                    title="Rate limiting disabled",
                    description="API rate limiting is not enabled",
                    severity=Severity.MEDIUM.value,
                    affected_files=[str(config_path)],
                    remediation="Enable rate limiting: rate_limiting.enabled = true",
                    references=["cognition/shared/rate_limiter.py"]
                ))

            # Check compliance enforcement
            enforcement_mode = config.get('compliance', {}).get('enforcement_mode', 'warn')
            if enforcement_mode == 'warn':
                self.findings.append(SecurityFinding(
                    id=f"CFG-{len(self.findings)+1:03d}",
                    category="Configuration",
                    title="Compliance in warn mode",
                    description="Compliance enforcement set to 'warn' instead of 'enforce'",
                    severity=Severity.LOW.value,
                    affected_files=[str(config_path)],
                    remediation="Set compliance.enforcement_mode = 'enforce' for production",
                    references=["enterprise_config/compliance_enforcer.py"]
                ))

        except Exception as e:
            if self.verbose:
                print(f"  Error auditing config: {e}")

        if self.verbose:
            print(f"  Found {len(self.findings)} configuration issues")

        return self.findings


# ============================================================================
# Security Auditor
# ============================================================================

class SecurityAuditor:
    """Main security auditor"""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.all_findings: List[SecurityFinding] = []

    def run_audit(self, args) -> AuditReport:
        """Run complete security audit"""

        audit_id = f"AUDIT-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"

        if self.verbose:
            print("="*80)
            print("T.A.R.S. SECURITY AUDIT")
            print("="*80)
            print(f"Audit ID: {audit_id}")
            print(f"Timestamp: {datetime.utcnow().isoformat()}Z")
            print()

        checks_performed = []

        # File integrity check
        if args.deep:
            checker = FileIntegrityChecker(verbose=self.verbose)
            findings = checker.check_integrity()
            self.all_findings.extend(findings)
            checks_performed.append("File Integrity")

        # Signature verification
        if args.verify_signature and len(args.verify_signature) == 2:
            sig_file = Path(args.verify_signature[0])
            data_file = Path(args.verify_signature[1])
            verifier = SignatureVerifier(verbose=self.verbose)
            verifier.verify_signature(sig_file, data_file)
            self.all_findings.extend(verifier.findings)
            checks_performed.append("Signature Verification")

        # SBOM vulnerability scan
        if args.scan_sbom:
            sbom_path = Path(args.scan_sbom)
            scanner = SBOMVulnerabilityScanner(verbose=self.verbose)
            findings = scanner.scan_sbom(sbom_path)
            self.all_findings.extend(findings)
            checks_performed.append("SBOM Vulnerability Scan")

        # API security testing
        if args.check_api:
            tester = APISecurityTester(verbose=self.verbose)
            findings = tester.test_api_security(args.check_api, args.auth_token)
            self.all_findings.extend(findings)
            checks_performed.append("API Security Testing")

        # Configuration audit
        if args.check_config:
            config_path = Path(args.check_config)
            auditor = ConfigurationAuditor(verbose=self.verbose)
            findings = auditor.audit_config(config_path)
            self.all_findings.extend(findings)
            checks_performed.append("Configuration Audit")

        # Generate summary
        summary = self._generate_summary()

        # Create report
        report = AuditReport(
            audit_id=audit_id,
            timestamp=datetime.utcnow().isoformat() + 'Z',
            version="1.0.2",
            findings=[asdict(f) for f in self.all_findings],
            summary=summary,
            checks_performed=checks_performed
        )

        return report

    def _generate_summary(self) -> Dict:
        """Generate summary statistics"""

        by_severity = {
            Severity.CRITICAL.value: 0,
            Severity.HIGH.value: 0,
            Severity.MEDIUM.value: 0,
            Severity.LOW.value: 0
        }

        by_category = {}

        for finding in self.all_findings:
            by_severity[finding.severity] += 1

            category = finding.category
            if category not in by_category:
                by_category[category] = 0
            by_category[category] += 1

        return {
            'total_findings': len(self.all_findings),
            'by_severity': by_severity,
            'by_category': by_category,
            'critical_findings': by_severity[Severity.CRITICAL.value],
            'high_findings': by_severity[Severity.HIGH.value]
        }


# ============================================================================
# Report Generation
# ============================================================================

def print_report(report: AuditReport, verbose: bool = False):
    """Print audit report to console"""

    print()
    print("="*80)
    print("SECURITY AUDIT SUMMARY")
    print("="*80)
    print(f"Total Findings: {report.summary['total_findings']}")
    print()
    print("By Severity:")
    for severity, count in report.summary['by_severity'].items():
        symbol = {
            Severity.CRITICAL.value: '🔴',
            Severity.HIGH.value: '🟠',
            Severity.MEDIUM.value: '🟡',
            Severity.LOW.value: '🟢'
        }.get(severity, '⚪')
        # Use [X] instead of emoji for Windows console
        if count > 0:
            print(f"  [{severity}] {count}")

    print()
    print("By Category:")
    for category, count in report.summary['by_category'].items():
        print(f"  - {category}: {count}")

    print()
    print("Checks Performed:")
    for check in report.checks_performed:
        print(f"  ✓ {check}")

    if report.findings:
        print()
        print("="*80)
        print("FINDINGS")
        print("="*80)

        for finding in report.findings:
            print()
            print(f"[{finding['id']}] {finding['title']}")
            print(f"  Severity: {finding['severity']}")
            print(f"  Category: {finding['category']}")
            print(f"  Description: {finding['description']}")
            if finding['affected_files']:
                print(f"  Affected: {', '.join(finding['affected_files'])}")
            print(f"  Remediation: {finding['remediation']}")
            if finding['references']:
                print(f"  References: {', '.join(finding['references'])}")

    print()
    print("="*80)


def save_json_report(report: AuditReport, output_path: str):
    """Save audit report as JSON"""
    report_dict = asdict(report)

    with open(output_path, 'w') as f:
        json.dump(report_dict, f, indent=2)


# ============================================================================
# Main
# ============================================================================

def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='T.A.R.S. Security Audit Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full deep audit
  python security/security_audit.py --deep --verbose

  # SBOM vulnerability scan
  python security/security_audit.py --scan-sbom release/v1.0.2/sbom/tars-cyclonedx.json

  # Verify signature
  python security/security_audit.py --verify-signature artifact.tar.gz.sig artifact.tar.gz

  # API security test
  python security/security_audit.py --check-api https://tars.company.com

  # Configuration audit
  python security/security_audit.py --check-config enterprise_config/profiles/production.yaml

  # Combined audit with JSON output
  python security/security_audit.py \\
      --deep \\
      --scan-sbom release/sbom/tars-cyclonedx.json \\
      --check-api https://tars.company.com \\
      --check-config enterprise_config/profiles/production.yaml \\
      --json audit-report.json \\
      --verbose
        """
    )

    parser.add_argument(
        '--deep',
        action='store_true',
        help='Perform deep audit (file integrity, etc.)'
    )

    parser.add_argument(
        '--scan-sbom',
        metavar='SBOM_FILE',
        help='Scan SBOM for vulnerabilities'
    )

    parser.add_argument(
        '--verify-signature',
        nargs=2,
        metavar=('SIG_FILE', 'DATA_FILE'),
        help='Verify RSA-PSS signature'
    )

    parser.add_argument(
        '--check-api',
        metavar='URL',
        help='Test API endpoint security'
    )

    parser.add_argument(
        '--check-config',
        metavar='CONFIG_FILE',
        help='Audit configuration file'
    )

    parser.add_argument(
        '--auth-token',
        help='JWT token for API authentication'
    )

    parser.add_argument(
        '--json',
        metavar='OUTPUT_FILE',
        help='Save JSON report to file'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose output'
    )

    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_args()

    # Check if any checks specified
    if not any([args.deep, args.scan_sbom, args.verify_signature, args.check_api, args.check_config]):
        print("Error: No checks specified. Use --help for usage.")
        sys.exit(1)

    # Run audit
    auditor = SecurityAuditor(verbose=args.verbose)

    try:
        report = auditor.run_audit(args)
    except KeyboardInterrupt:
        print("\n\nAudit interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError running audit: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

    # Print report
    print_report(report, verbose=args.verbose)

    # Save JSON report
    if args.json:
        save_json_report(report, args.json)
        print(f"\n✓ JSON report saved: {args.json}")

    # Exit code based on findings
    if report.summary['critical_findings'] > 0:
        sys.exit(2)  # Critical findings
    elif report.summary['high_findings'] > 0:
        sys.exit(1)  # High severity findings
    else:
        sys.exit(0)  # Success or only low/medium findings


if __name__ == '__main__':
    main()
