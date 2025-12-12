#!/usr/bin/env python3
"""
Quick validation test for Phase 14.7 SBOM and SLSA implementations.

Tests:
1. SBOM generation (CycloneDX and SPDX formats)
2. SLSA provenance generation
3. JSON schema validation
4. Deterministic output verification

Exit code 0 = all tests passed
"""

import json
import sys
from pathlib import Path
import tempfile

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

def test_sbom_generation():
    """Test SBOM generation for both CycloneDX and SPDX."""
    print("=" * 80)
    print("TEST 1: SBOM Generation")
    print("=" * 80)

    # Import directly from module to avoid __init__.py dependencies
    import importlib.util
    spec = importlib.util.spec_from_file_location("sbom_generator", PROJECT_ROOT / "security" / "sbom_generator.py")
    sbom_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sbom_module)
    SBOMGenerator = sbom_module.SBOMGenerator
    Dependency = sbom_module.Dependency

    # Create generator
    generator = SBOMGenerator(
        project_name="TestProject",
        project_version="1.0.0",
        project_root=PROJECT_ROOT
    )

    # Mock dependencies (since many are not installed)
    generator.dependencies = [
        Dependency(
            name="pytest",
            version="7.4.3",
            package_type="pypi",
            license="MIT",
            homepage="https://pytest.org"
        ),
        Dependency(
            name="requests",
            version="2.31.0",
            package_type="pypi",
            license="Apache-2.0"
        )
    ]

    print(f"Testing with {len(generator.dependencies)} mock dependencies")

    # Test CycloneDX generation
    print("\n[1/2] Testing CycloneDX format...")
    cyclonedx_sbom = generator.generate_cyclonedx()

    assert cyclonedx_sbom["bomFormat"] == "CycloneDX", "Invalid BOM format"
    assert cyclonedx_sbom["specVersion"] == "1.5", "Invalid spec version"
    assert len(cyclonedx_sbom["components"]) == 2, "Missing components"
    assert cyclonedx_sbom["components"][0]["purl"].startswith("pkg:pypi/"), "Invalid PURL"
    print("[OK] CycloneDX format valid")

    # Test SPDX generation
    print("\n[2/2] Testing SPDX format...")
    spdx_sbom = generator.generate_spdx()

    assert spdx_sbom["spdxVersion"] == "SPDX-2.3", "Invalid SPDX version"
    assert spdx_sbom["dataLicense"] == "CC0-1.0", "Invalid data license"
    assert len(spdx_sbom["packages"]) == 3, "Missing packages (main + 2 deps)"  # main + 2 deps
    assert len(spdx_sbom["relationships"]) == 3, "Missing relationships"  # DESCRIBES + 2 DEPENDS_ON
    print("[OK] SPDX format valid")

    print("\n[PASS] SBOM generation tests passed")
    return True


def test_slsa_generation():
    """Test SLSA provenance generation."""
    print("\n" + "=" * 80)
    print("TEST 2: SLSA Provenance Generation")
    print("=" * 80)

    # Import directly from module to avoid __init__.py dependencies
    import importlib.util
    spec = importlib.util.spec_from_file_location("slsa_generator", PROJECT_ROOT / "security" / "slsa_generator.py")
    slsa_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(slsa_module)
    SLSAProvenanceGenerator = slsa_module.SLSAProvenanceGenerator

    # Create temporary test artifact
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        f.write("test artifact content")
        artifact_path = Path(f.name)

    try:
        # Create generator
        generator = SLSAProvenanceGenerator(
            project_name="TestProject",
            project_version="1.0.0"
        )

        print(f"Testing with artifact: {artifact_path.name}")

        # Generate provenance
        print("\n[1/3] Generating provenance...")
        provenance = generator.generate_provenance(
            artifact_paths=artifact_path,
            include_materials=False  # Skip materials for quick test
        )

        # Validate structure
        print("\n[2/3] Validating structure...")
        assert provenance["_type"] == "https://in-toto.io/Statement/v1", "Invalid statement type"
        assert provenance["predicateType"] == "https://slsa.dev/provenance/v1", "Invalid predicate type"
        assert len(provenance["subject"]) == 1, "Missing subject"
        assert "sha256" in provenance["subject"][0]["digest"], "Missing SHA-256 digest"
        assert "buildDefinition" in provenance["predicate"], "Missing buildDefinition"
        assert "runDetails" in provenance["predicate"], "Missing runDetails"
        print("[OK] Provenance structure valid")

        # Validate builder info
        print("\n[3/3] Validating builder metadata...")
        builder = provenance["predicate"]["runDetails"]["builder"]
        assert "id" in builder, "Missing builder ID"
        assert "invocationId" in provenance["predicate"]["runDetails"]["metadata"], "Missing invocation ID"
        print("[OK] Builder metadata valid")

        print("\n[PASS] SLSA provenance tests passed")
        return True

    finally:
        # Clean up
        if artifact_path.exists():
            artifact_path.unlink()


def test_deterministic_output():
    """Test that SBOM generation is deterministic."""
    print("\n" + "=" * 80)
    print("TEST 3: Deterministic Output")
    print("=" * 80)

    # Import directly from module to avoid __init__.py dependencies
    import importlib.util
    spec = importlib.util.spec_from_file_location("sbom_generator", PROJECT_ROOT / "security" / "sbom_generator.py")
    sbom_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sbom_module)
    SBOMGenerator = sbom_module.SBOMGenerator
    Dependency = sbom_module.Dependency

    # Create two generators with same data
    deps = [Dependency("pytest", "7.4.3", "pypi")]

    gen1 = SBOMGenerator("TestProject", "1.0.0")
    gen1.dependencies = deps

    gen2 = SBOMGenerator("TestProject", "1.0.0")
    gen2.dependencies = deps

    # Generate SBOMs
    print("\n[1/2] Generating SBOM twice...")
    sbom1 = gen1.generate_cyclonedx()
    sbom2 = gen2.generate_cyclonedx()

    # Compare UUIDs (should be deterministic)
    print("\n[2/2] Comparing outputs...")
    uuid1 = sbom1["serialNumber"]
    uuid2 = sbom2["serialNumber"]

    assert uuid1 == uuid2, f"UUIDs not deterministic: {uuid1} != {uuid2}"
    print(f"[OK] Serial numbers match: {uuid1}")

    # Compare component count
    assert len(sbom1["components"]) == len(sbom2["components"]), "Component count mismatch"
    print("[OK] Component counts match")

    print("\n[PASS] Deterministic output tests passed")
    return True


def test_json_validity():
    """Test that generated JSON is valid."""
    print("\n" + "=" * 80)
    print("TEST 4: JSON Validity")
    print("=" * 80)

    # Import directly from module to avoid __init__.py dependencies
    import importlib.util
    spec = importlib.util.spec_from_file_location("sbom_generator", PROJECT_ROOT / "security" / "sbom_generator.py")
    sbom_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sbom_module)
    SBOMGenerator = sbom_module.SBOMGenerator
    Dependency = sbom_module.Dependency

    generator = SBOMGenerator("TestProject", "1.0.0")
    generator.dependencies = [Dependency("pytest", "7.4.3", "pypi")]

    print("\n[1/2] Testing CycloneDX JSON serialization...")
    cyclonedx = generator.generate_cyclonedx()
    try:
        json_str = json.dumps(cyclonedx, indent=2, sort_keys=True)
        reloaded = json.loads(json_str)
        assert reloaded["bomFormat"] == "CycloneDX"
        print("[OK] CycloneDX JSON valid")
    except json.JSONDecodeError as e:
        print(f"✗ CycloneDX JSON invalid: {e}")
        return False

    print("\n[2/2] Testing SPDX JSON serialization...")
    spdx = generator.generate_spdx()
    try:
        json_str = json.dumps(spdx, indent=2, sort_keys=True)
        reloaded = json.loads(json_str)
        assert reloaded["spdxVersion"] == "SPDX-2.3"
        print("[OK] SPDX JSON valid")
    except json.JSONDecodeError as e:
        print(f"✗ SPDX JSON invalid: {e}")
        return False

    print("\n[PASS] JSON validity tests passed")
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("PHASE 14.7 - SBOM & SLSA Implementation Validation")
    print("=" * 80)
    print()

    tests = [
        ("SBOM Generation", test_sbom_generation),
        ("SLSA Provenance", test_slsa_generation),
        ("Deterministic Output", test_deterministic_output),
        ("JSON Validity", test_json_validity),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
                print(f"\n[FAIL] {name} FAILED")
        except Exception as e:
            failed += 1
            print(f"\n[FAIL] {name} FAILED with exception:")
            print(f"   {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Total tests: {len(tests)}")
    print(f"Passed:      {passed}")
    print(f"Failed:      {failed}")
    print("=" * 80)

    if failed == 0:
        print("\n[PASS] ALL TESTS PASSED - Phase 14.7 implementations validated!")
        return 0
    else:
        print(f"\n[FAIL] {failed} TEST(S) FAILED - Review implementation")
        return 1


if __name__ == "__main__":
    sys.exit(main())
