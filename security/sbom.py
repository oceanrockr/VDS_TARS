"""
Software Bill of Materials (SBOM) Generator

Generates SBOM in multiple formats:
- CycloneDX (JSON/XML)
- SPDX (JSON/RDF)
"""

from typing import List, Dict, Any, Optional, Literal
from pathlib import Path
from datetime import datetime
import json
import hashlib
import subprocess
import re


class Dependency:
    """Represents a software dependency."""

    def __init__(
        self,
        name: str,
        version: str,
        package_type: str = "pypi",
        license: Optional[str] = None,
        homepage: Optional[str] = None,
        description: Optional[str] = None,
    ):
        self.name = name
        self.version = version
        self.package_type = package_type
        self.license = license
        self.homepage = homepage
        self.description = description

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "package_type": self.package_type,
            "license": self.license,
            "homepage": self.homepage,
            "description": self.description,
        }


class SBOMGenerator:
    """
    Generate Software Bill of Materials (SBOM) for T.A.R.S.

    Supports CycloneDX and SPDX formats.
    """

    def __init__(
        self,
        project_name: str = "tars-observability",
        project_version: str = "1.0.2-dev",
        project_root: Optional[Path] = None,
    ):
        """
        Initialize SBOM generator.

        Args:
            project_name: Project name
            project_version: Project version
            project_root: Path to project root directory
        """
        self.project_name = project_name
        self.project_version = project_version
        self.project_root = project_root or Path.cwd()

        self.dependencies: List[Dependency] = []

    def scan_dependencies(self):
        """
        Scan project for dependencies.

        Parses requirements files and extracts dependency information.
        """
        self.dependencies = []

        # Scan requirements files
        req_files = [
            self.project_root / "requirements-dev.txt",
            self.project_root / "requirements.txt",
        ]

        for req_file in req_files:
            if req_file.exists():
                self._parse_requirements_file(req_file)

    def _parse_requirements_file(self, req_file: Path):
        """Parse requirements.txt and extract dependencies."""
        with open(req_file, "r") as f:
            for line in f:
                line = line.strip()

                # Skip comments and empty lines
                if not line or line.startswith("#"):
                    continue

                # Parse package==version
                match = re.match(r"^([a-zA-Z0-9_\-\[\]]+)==([0-9\.]+)", line)
                if match:
                    name, version = match.groups()
                    # Remove extras (e.g., "package[extra]" -> "package")
                    name = re.sub(r"\[.*\]", "", name)

                    dep = Dependency(
                        name=name,
                        version=version,
                        package_type="pypi",
                    )
                    self.dependencies.append(dep)

    def generate_cyclonedx(
        self,
        output_format: Literal["json", "xml"] = "json"
    ) -> Dict[str, Any]:
        """
        Generate CycloneDX SBOM.

        Args:
            output_format: Output format (json or xml)

        Returns:
            CycloneDX SBOM as dictionary
        """
        sbom = {
            "bomFormat": "CycloneDX",
            "specVersion": "1.5",
            "serialNumber": f"urn:uuid:{self._generate_uuid()}",
            "version": 1,
            "metadata": {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "tools": [
                    {
                        "vendor": "T.A.R.S.",
                        "name": "SBOM Generator",
                        "version": "1.0.2-dev",
                    }
                ],
                "component": {
                    "type": "application",
                    "name": self.project_name,
                    "version": self.project_version,
                    "description": "T.A.R.S. Enterprise Observability Platform",
                }
            },
            "components": []
        }

        # Add dependencies
        for dep in self.dependencies:
            component = {
                "type": "library",
                "name": dep.name,
                "version": dep.version,
                "purl": f"pkg:pypi/{dep.name}@{dep.version}",
            }

            if dep.license:
                component["licenses"] = [{"license": {"id": dep.license}}]

            if dep.homepage:
                component["externalReferences"] = [
                    {
                        "type": "website",
                        "url": dep.homepage,
                    }
                ]

            sbom["components"].append(component)

        return sbom

    def generate_spdx(self) -> Dict[str, Any]:
        """
        Generate SPDX SBOM.

        Returns:
            SPDX SBOM as dictionary
        """
        sbom = {
            "spdxVersion": "SPDX-2.3",
            "dataLicense": "CC0-1.0",
            "SPDXID": "SPDXRef-DOCUMENT",
            "name": f"{self.project_name}-{self.project_version}",
            "documentNamespace": f"https://tars.example.com/spdx/{self.project_name}-{self.project_version}",
            "creationInfo": {
                "created": datetime.utcnow().isoformat() + "Z",
                "creators": [
                    "Tool: T.A.R.S. SBOM Generator",
                ],
                "licenseListVersion": "3.21",
            },
            "packages": []
        }

        # Add main package
        main_package = {
            "SPDXID": "SPDXRef-Package-Main",
            "name": self.project_name,
            "versionInfo": self.project_version,
            "downloadLocation": "NOASSERTION",
            "filesAnalyzed": False,
            "supplier": "Organization: Veleron Dev Studios",
        }
        sbom["packages"].append(main_package)

        # Add dependencies
        for idx, dep in enumerate(self.dependencies):
            package = {
                "SPDXID": f"SPDXRef-Package-{idx + 1}",
                "name": dep.name,
                "versionInfo": dep.version,
                "downloadLocation": f"https://pypi.org/project/{dep.name}/{dep.version}/",
                "filesAnalyzed": False,
                "externalRefs": [
                    {
                        "referenceCategory": "PACKAGE-MANAGER",
                        "referenceType": "purl",
                        "referenceLocator": f"pkg:pypi/{dep.name}@{dep.version}",
                    }
                ]
            }

            if dep.license:
                package["licenseConcluded"] = dep.license
                package["licenseDeclared"] = dep.license

            if dep.homepage:
                package["homepage"] = dep.homepage

            sbom["packages"].append(package)

        # Add relationships
        sbom["relationships"] = [
            {
                "spdxElementId": "SPDXRef-DOCUMENT",
                "relationshipType": "DESCRIBES",
                "relatedSpdxElement": "SPDXRef-Package-Main",
            }
        ]

        for idx in range(len(self.dependencies)):
            sbom["relationships"].append({
                "spdxElementId": "SPDXRef-Package-Main",
                "relationshipType": "DEPENDS_ON",
                "relatedSpdxElement": f"SPDXRef-Package-{idx + 1}",
            })

        return sbom

    def save_sbom(
        self,
        output_path: Path,
        format: Literal["cyclonedx", "spdx"] = "cyclonedx",
    ):
        """
        Save SBOM to file.

        Args:
            output_path: Path to save SBOM
            format: SBOM format (cyclonedx or spdx)
        """
        if format == "cyclonedx":
            sbom = self.generate_cyclonedx()
        elif format == "spdx":
            sbom = self.generate_spdx()
        else:
            raise ValueError(f"Unsupported SBOM format: {format}")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(sbom, f, indent=2)

    def _generate_uuid(self) -> str:
        """Generate a deterministic UUID based on project name and version."""
        data = f"{self.project_name}:{self.project_version}".encode("utf-8")
        hash_bytes = hashlib.sha256(data).digest()[:16]

        # Format as UUID (version 4)
        hash_bytes = bytearray(hash_bytes)
        hash_bytes[6] = (hash_bytes[6] & 0x0F) | 0x40  # Version 4
        hash_bytes[8] = (hash_bytes[8] & 0x3F) | 0x80  # Variant

        uuid_str = "-".join([
            hash_bytes[:4].hex(),
            hash_bytes[4:6].hex(),
            hash_bytes[6:8].hex(),
            hash_bytes[8:10].hex(),
            hash_bytes[10:16].hex(),
        ])

        return uuid_str


def generate_sbom_for_tars(
    output_dir: Path,
    formats: List[Literal["cyclonedx", "spdx"]] = ["cyclonedx", "spdx"],
):
    """
    Generate SBOM for T.A.R.S. in multiple formats.

    Args:
        output_dir: Directory to save SBOM files
        formats: List of SBOM formats to generate
    """
    generator = SBOMGenerator(
        project_name="tars-observability",
        project_version="1.0.2-dev",
    )

    generator.scan_dependencies()

    for format in formats:
        output_file = output_dir / f"sbom.{format}.json"
        generator.save_sbom(output_file, format=format)
        print(f"Generated SBOM: {output_file}")
