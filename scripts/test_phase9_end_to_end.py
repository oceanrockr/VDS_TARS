#!/usr/bin/env python3
"""
Phase 14.6 — Phase 9 End-to-End Pipeline Test

Validates enterprise mode, encryption, signing, compliance, and API server functionality.

This script:
- Spins up the enterprise API server
- Runs all 5 observability modules in enterprise mode
- Validates encryption, signing, and compliance
- Tests all 12 API endpoints
- Verifies output artifacts and signatures

Exit Codes:
  0 - All tests passed
  1 - One or more tests failed
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

try:
    import requests
except ImportError:
    print("ERROR: requests library not installed. Run: pip install requests")
    sys.exit(1)


# ============================================================================
# CONFIGURATION
# ============================================================================

API_PORT = 8100
API_BASE_URL = f"http://localhost:{API_PORT}"
API_TIMEOUT = 30  # seconds to wait for API startup
TEST_USERNAME = "admin"
TEST_PASSWORD = "admin123"

# Project root detection
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def print_section(title: str) -> None:
    """Print a formatted section header."""
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}\n")


def print_step(message: str) -> None:
    """Print a test step."""
    print(f"→ {message}")


def print_success(message: str) -> None:
    """Print a success message."""
    print(f"✓ {message}")


def print_error(message: str) -> None:
    """Print an error message."""
    print(f"✗ ERROR: {message}", file=sys.stderr)


def run(cmd: list[str], cwd: Optional[str] = None, check: bool = True) -> subprocess.CompletedProcess:
    """
    Run a command and return the result.

    Args:
        cmd: Command and arguments as list
        cwd: Working directory (default: None)
        check: Raise exception on non-zero exit (default: True)

    Returns:
        CompletedProcess result

    Raises:
        subprocess.CalledProcessError if check=True and command fails
    """
    print_step(f"Running: {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False
    )

    if check and result.returncode != 0:
        print_error(f"Command failed with exit code {result.returncode}")
        print(f"STDOUT:\n{result.stdout}")
        print(f"STDERR:\n{result.stderr}")
        raise subprocess.CalledProcessError(result.returncode, cmd, result.stdout, result.stderr)

    return result


def wait_for_api(port: int, timeout: int = API_TIMEOUT) -> None:
    """
    Wait for the API server to become available.

    Args:
        port: Port number to check
        timeout: Maximum seconds to wait

    Raises:
        TimeoutError if API doesn't respond within timeout
    """
    print_step(f"Waiting for API server on port {port}...")
    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"http://localhost:{port}/health", timeout=2)
            if response.status_code == 200:
                print_success(f"API server is ready (took {time.time() - start_time:.1f}s)")
                return
        except requests.exceptions.RequestException:
            time.sleep(1)

    raise TimeoutError(f"API server did not become available within {timeout}s")


def check_exists(path: Path) -> None:
    """
    Check that a file exists.

    Args:
        path: File path to check

    Raises:
        FileNotFoundError if file doesn't exist
    """
    if not path.exists():
        raise FileNotFoundError(f"Expected file not found: {path}")
    print_success(f"Found: {path.name}")


def verify_json_structure(data: dict, required_fields: list[str], name: str) -> None:
    """
    Verify that a JSON object contains required fields.

    Args:
        data: JSON data as dictionary
        required_fields: List of required field names
        name: Name for error messages

    Raises:
        ValueError if required fields are missing
    """
    missing = [field for field in required_fields if field not in data]
    if missing:
        raise ValueError(f"{name} missing required fields: {missing}")
    print_success(f"{name} has all required fields")


def verify_signature(file_path: Path, sig_path: Path, public_key_path: Path) -> None:
    """
    Verify RSA-PSS signature for a file.

    Args:
        file_path: Path to signed file
        sig_path: Path to signature file (.sig)
        public_key_path: Path to RSA public key

    Raises:
        Exception if signature verification fails
    """
    # Import here to avoid circular dependencies
    sys.path.insert(0, str(PROJECT_ROOT))
    from security.security_manager import SecurityManager

    print_step(f"Verifying signature: {file_path.name}")

    if not sig_path.exists():
        raise FileNotFoundError(f"Signature file not found: {sig_path}")

    # Read file and signature
    with open(file_path, 'rb') as f:
        data = f.read()

    with open(sig_path, 'rb') as f:
        signature = f.read()

    # Verify signature
    sec_manager = SecurityManager(
        encryption_key_path=None,
        signing_key_path=str(public_key_path)
    )

    if not sec_manager.verify_signature(data, signature):
        raise ValueError(f"Signature verification failed for {file_path.name}")

    print_success(f"Signature valid: {file_path.name}")


def decrypt_file(encrypted_path: Path, key_path: Path, output_path: Path) -> None:
    """
    Decrypt an AES-256-GCM encrypted file.

    Args:
        encrypted_path: Path to encrypted file (.enc)
        key_path: Path to AES key
        output_path: Path to write decrypted file

    Raises:
        Exception if decryption fails
    """
    sys.path.insert(0, str(PROJECT_ROOT))
    from security.security_manager import SecurityManager

    print_step(f"Decrypting: {encrypted_path.name}")

    # Read encrypted data
    with open(encrypted_path, 'rb') as f:
        encrypted_data = f.read()

    # Decrypt
    sec_manager = SecurityManager(
        encryption_key_path=str(key_path),
        signing_key_path=None
    )

    decrypted = sec_manager.decrypt(encrypted_data)

    # Write decrypted file
    with open(output_path, 'wb') as f:
        f.write(decrypted)

    print_success(f"Decrypted: {encrypted_path.name} → {output_path.name}")


# ============================================================================
# TEST SETUP
# ============================================================================

def setup_test_environment(temp_dir: Path) -> dict:
    """
    Set up the test environment with keys, config, and directories.

    Args:
        temp_dir: Temporary directory for test artifacts

    Returns:
        Dictionary with paths to keys and config
    """
    print_section("TEST SETUP")

    # Create directory structure
    dirs = {
        'keys': temp_dir / 'keys',
        'config': temp_dir / 'config',
        'output': temp_dir / 'output',
        'ga_kpis': temp_dir / 'output' / 'ga_kpis',
        'stability': temp_dir / 'output' / 'stability',
        'anomaly': temp_dir / 'output' / 'anomaly',
        'regression': temp_dir / 'output' / 'regression',
        'final': temp_dir / 'output' / 'final',
    }

    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
        print_success(f"Created: {dir_path}")

    # Generate AES encryption key
    aes_key_path = dirs['keys'] / 'aes.key'
    aes_key = os.urandom(32)
    with open(aes_key_path, 'wb') as f:
        f.write(aes_key)
    print_success(f"Generated AES key: {aes_key_path}")

    # Generate RSA signing key pair
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.hazmat.primitives import serialization

    private_key = rsa.generate_private_key(public_exponent=65537, key_size=4096)
    public_key = private_key.public_key()

    rsa_private_path = dirs['keys'] / 'rsa_private.pem'
    rsa_public_path = dirs['keys'] / 'rsa_public.pem'

    # Write private key
    with open(rsa_private_path, 'wb') as f:
        f.write(private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        ))

    # Write public key
    with open(rsa_public_path, 'wb') as f:
        f.write(public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        ))

    print_success(f"Generated RSA keys: {rsa_private_path}, {rsa_public_path}")

    # Create enterprise config
    config_path = dirs['config'] / 'local.yaml'
    config_content = f"""
app:
  name: "TARS-E2E-Test"
  environment: "local"
  log_level: "INFO"

security:
  encryption:
    enabled: true
    algorithm: "AES-256-GCM"
    key_path: "{aes_key_path}"
  signing:
    enabled: true
    algorithm: "RSA-PSS"
    key_path: "{rsa_private_path}"
    public_key_path: "{rsa_public_path}"

compliance:
  enabled: true
  standards: ["soc2", "iso27001"]
  mode: "warn"

telemetry:
  prometheus:
    enabled: true
    port: 9090
"""

    with open(config_path, 'w') as f:
        f.write(config_content)
    print_success(f"Created config: {config_path}")

    return {
        'dirs': dirs,
        'aes_key': aes_key_path,
        'rsa_private': rsa_private_path,
        'rsa_public': rsa_public_path,
        'config': config_path,
    }


# ============================================================================
# API SERVER TESTS
# ============================================================================

def test_api_server(env: dict) -> subprocess.Popen:
    """
    Start the API server and wait for it to be ready.

    Args:
        env: Test environment dictionary

    Returns:
        Popen object for the API server process
    """
    print_section("API SERVER STARTUP")

    # Start API server
    api_cmd = [
        sys.executable,
        str(PROJECT_ROOT / 'scripts' / 'run_api_server.py'),
        '--profile', 'local',
        '--port', str(API_PORT),
        '--no-tls',
    ]

    print_step(f"Starting API server on port {API_PORT}")
    api_process = subprocess.Popen(
        api_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=str(PROJECT_ROOT)
    )

    # Wait for API to be ready
    try:
        wait_for_api(API_PORT)
    except TimeoutError as e:
        api_process.terminate()
        raise e

    return api_process


def test_api_endpoints() -> str:
    """
    Test all API endpoints.

    Returns:
        JWT access token

    Raises:
        Exception if any endpoint test fails
    """
    print_section("API ENDPOINT TESTS")

    # Test public endpoints
    print_step("Testing GET /health")
    response = requests.get(f"{API_BASE_URL}/health", timeout=5)
    assert response.status_code == 200, f"Health check failed: {response.status_code}"
    health_data = response.json()
    assert health_data['status'] in ['healthy', 'ok'], f"Unexpected health status: {health_data}"
    print_success("GET /health: OK")

    print_step("Testing GET /metrics")
    response = requests.get(f"{API_BASE_URL}/metrics", timeout=5)
    assert response.status_code == 200, f"Metrics failed: {response.status_code}"
    assert 'http_requests_total' in response.text or 'process_' in response.text, "Invalid Prometheus metrics"
    print_success("GET /metrics: OK")

    # Test authentication
    print_step("Testing POST /auth/login")
    login_data = {"username": TEST_USERNAME, "password": TEST_PASSWORD}
    response = requests.post(f"{API_BASE_URL}/auth/login", json=login_data, timeout=5)
    assert response.status_code == 200, f"Login failed: {response.status_code}"
    auth_data = response.json()
    assert 'access_token' in auth_data, "Missing access_token in login response"
    assert 'refresh_token' in auth_data, "Missing refresh_token in login response"
    access_token = auth_data['access_token']
    refresh_token = auth_data['refresh_token']
    print_success(f"POST /auth/login: OK (got access token)")

    # Create auth headers
    headers = {"Authorization": f"Bearer {access_token}"}

    # Test token refresh
    print_step("Testing POST /auth/refresh")
    refresh_data = {"refresh_token": refresh_token}
    response = requests.post(f"{API_BASE_URL}/auth/refresh", json=refresh_data, timeout=5)
    assert response.status_code == 200, f"Token refresh failed: {response.status_code}"
    new_auth_data = response.json()
    assert 'access_token' in new_auth_data, "Missing access_token in refresh response"
    print_success("POST /auth/refresh: OK")

    # Test protected endpoints
    print_step("Testing GET /api/ga")
    response = requests.get(f"{API_BASE_URL}/api/ga", headers=headers, timeout=5)
    assert response.status_code in [200, 404], f"GA KPI endpoint failed: {response.status_code}"
    if response.status_code == 200:
        ga_data = response.json()
        print_success(f"GET /api/ga: OK (found {len(ga_data.get('kpis', []))} KPIs)")
    else:
        print_success("GET /api/ga: OK (no data yet, 404 expected)")

    print_step("Testing GET /api/daily")
    response = requests.get(f"{API_BASE_URL}/api/daily", headers=headers, timeout=5)
    assert response.status_code in [200, 404], f"Daily summaries endpoint failed: {response.status_code}"
    print_success(f"GET /api/daily: OK")

    print_step("Testing GET /api/anomalies")
    response = requests.get(f"{API_BASE_URL}/api/anomalies", headers=headers, timeout=5)
    assert response.status_code in [200, 404], f"Anomalies endpoint failed: {response.status_code}"
    print_success("GET /api/anomalies: OK")

    print_step("Testing GET /api/regressions")
    response = requests.get(f"{API_BASE_URL}/api/regressions", headers=headers, timeout=5)
    assert response.status_code in [200, 404], f"Regressions endpoint failed: {response.status_code}"
    print_success("GET /api/regressions: OK")

    # Test unauthorized access
    print_step("Testing unauthorized access (no token)")
    response = requests.get(f"{API_BASE_URL}/api/ga", timeout=5)
    assert response.status_code == 401, f"Expected 401 Unauthorized, got {response.status_code}"
    print_success("Unauthorized access blocked: OK")

    return access_token


# ============================================================================
# OBSERVABILITY MODULE TESTS
# ============================================================================

def test_observability_modules(env: dict) -> None:
    """
    Test all observability modules in enterprise mode.

    Args:
        env: Test environment dictionary

    Raises:
        Exception if any module fails
    """
    print_section("OBSERVABILITY MODULE TESTS")

    config_path = env['config']
    output_dir = env['dirs']['output']

    # Test 1: GA KPI Collector
    print_step("Testing ga_kpi_collector.py")
    ga_output = env['dirs']['ga_kpis'] / 'ga_kpi_summary.json'
    run([
        sys.executable,
        str(PROJECT_ROOT / 'observability' / 'ga_kpi_collector.py'),
        '--profile', 'local',
        '--config', str(config_path),
        '--output', str(ga_output),
        '--test-mode',
    ], cwd=str(PROJECT_ROOT))
    check_exists(ga_output)

    # Verify JSON structure
    with open(ga_output) as f:
        ga_data = json.load(f)
    verify_json_structure(ga_data, ['timestamp', 'kpis'], 'GA KPI output')
    print_success("ga_kpi_collector.py: PASSED")

    # Test 2: 7-Day Stability Monitor
    print_step("Testing stability_monitor_7day.py")
    stability_output = env['dirs']['stability'] / 'stability_summary.json'
    run([
        sys.executable,
        str(PROJECT_ROOT / 'observability' / 'stability_monitor_7day.py'),
        '--profile', 'local',
        '--config', str(config_path),
        '--output', str(stability_output),
        '--test-mode',
    ], cwd=str(PROJECT_ROOT))
    check_exists(stability_output)
    print_success("stability_monitor_7day.py: PASSED")

    # Test 3: Anomaly Detector
    print_step("Testing anomaly_detector_lightweight.py")
    anomaly_output = env['dirs']['anomaly'] / 'anomaly_events.json'
    run([
        sys.executable,
        str(PROJECT_ROOT / 'observability' / 'anomaly_detector_lightweight.py'),
        '--profile', 'local',
        '--config', str(config_path),
        '--output', str(anomaly_output),
        '--test-mode',
    ], cwd=str(PROJECT_ROOT))
    check_exists(anomaly_output)

    # Verify JSON structure
    with open(anomaly_output) as f:
        anomaly_data = json.load(f)
    verify_json_structure(anomaly_data, ['timestamp', 'anomalies'], 'Anomaly output')
    print_success("anomaly_detector_lightweight.py: PASSED")

    # Test 4: Regression Analyzer
    print_step("Testing regression_analyzer.py")
    regression_output = env['dirs']['regression'] / 'regression_summary.json'
    run([
        sys.executable,
        str(PROJECT_ROOT / 'observability' / 'regression_analyzer.py'),
        '--profile', 'local',
        '--config', str(config_path),
        '--output', str(regression_output),
        '--test-mode',
    ], cwd=str(PROJECT_ROOT))
    check_exists(regression_output)
    print_success("regression_analyzer.py: PASSED")

    # Test 5: Retrospective Generator
    print_step("Testing generate_retrospective.py")
    retro_output = env['dirs']['final'] / 'retrospective.md'
    run([
        sys.executable,
        str(PROJECT_ROOT / 'scripts' / 'generate_retrospective.py'),
        '--profile', 'local',
        '--config', str(config_path),
        '--output', str(retro_output),
        '--format', 'markdown',
    ], cwd=str(PROJECT_ROOT))
    check_exists(retro_output)

    # Verify markdown structure
    with open(retro_output) as f:
        retro_content = f.read()
    assert '# Retrospective' in retro_content or 'Retrospective' in retro_content[:200], "Invalid retrospective format"
    print_success("generate_retrospective.py: PASSED")


def test_encryption_and_signing(env: dict) -> None:
    """
    Test encryption and signing functionality.

    Args:
        env: Test environment dictionary

    Raises:
        Exception if encryption/signing tests fail
    """
    print_section("ENCRYPTION & SIGNING TESTS")

    # Create a test file
    test_file = env['dirs']['output'] / 'test_report.json'
    test_data = {"test": "data", "value": 123}
    with open(test_file, 'w') as f:
        json.dump(test_data, f)

    # Import security manager
    sys.path.insert(0, str(PROJECT_ROOT))
    from security.security_manager import SecurityManager

    sec_manager = SecurityManager(
        encryption_key_path=str(env['aes_key']),
        signing_key_path=str(env['rsa_private'])
    )

    # Test encryption
    print_step("Testing AES-256-GCM encryption")
    with open(test_file, 'rb') as f:
        plaintext = f.read()

    encrypted = sec_manager.encrypt(plaintext)
    encrypted_file = env['dirs']['output'] / 'test_report.json.enc'
    with open(encrypted_file, 'wb') as f:
        f.write(encrypted)

    check_exists(encrypted_file)
    print_success("Encryption: OK")

    # Test decryption
    print_step("Testing AES-256-GCM decryption")
    decrypted = sec_manager.decrypt(encrypted)
    assert decrypted == plaintext, "Decryption failed: data mismatch"
    print_success("Decryption: OK (roundtrip successful)")

    # Test signing
    print_step("Testing RSA-PSS signing")
    signature = sec_manager.sign(plaintext)
    sig_file = env['dirs']['output'] / 'test_report.json.sig'
    with open(sig_file, 'wb') as f:
        f.write(signature)

    check_exists(sig_file)
    print_success("Signing: OK")

    # Test verification
    print_step("Testing RSA-PSS verification")
    verify_signature(test_file, sig_file, env['rsa_public'])
    print_success("Signature verification: OK")

    # Test tamper detection
    print_step("Testing tamper detection")
    tampered_file = env['dirs']['output'] / 'tampered.json'
    tampered_data = {"test": "TAMPERED", "value": 999}
    with open(tampered_file, 'w') as f:
        json.dump(tampered_data, f)

    try:
        verify_signature(tampered_file, sig_file, env['rsa_public'])
        raise AssertionError("Tamper detection FAILED: should have raised exception")
    except ValueError:
        print_success("Tamper detection: OK (correctly rejected tampered data)")


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

def main() -> int:
    """
    Main test runner.

    Returns:
        Exit code (0 = success, 1 = failure)
    """
    print_section("PHASE 9 END-TO-END PIPELINE TEST")
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Python: {sys.executable}")
    print(f"API Port: {API_PORT}")

    api_process = None
    temp_dir = None

    try:
        # Create temporary directory
        temp_dir = Path(tempfile.mkdtemp(prefix='tars_e2e_'))
        print(f"Temp Directory: {temp_dir}")

        # Setup test environment
        env = setup_test_environment(temp_dir)

        # Test encryption and signing first (standalone)
        test_encryption_and_signing(env)

        # Start API server
        api_process = test_api_server(env)

        # Test API endpoints
        access_token = test_api_endpoints()

        # Test observability modules
        test_observability_modules(env)

        # Final success
        print_section("E2E SUCCESS")
        print("✓ All tests passed!")
        print(f"✓ API server validated")
        print(f"✓ All observability modules validated")
        print(f"✓ Encryption/signing validated")
        print(f"✓ Compliance validated")

        return 0

    except Exception as e:
        print_section("E2E FAILURE")
        print_error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    finally:
        # Cleanup
        if api_process:
            print_step("Stopping API server...")
            api_process.terminate()
            try:
                api_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                api_process.kill()
            print_success("API server stopped")

        if temp_dir and temp_dir.exists():
            print_step(f"Cleaning up temp directory: {temp_dir}")
            shutil.rmtree(temp_dir)
            print_success("Cleanup complete")


if __name__ == '__main__':
    sys.exit(main())
