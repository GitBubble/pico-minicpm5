"""Generate a deterministic SPDX 2.3 source SBOM without external tooling."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

from .. import __version__
from ..contract import sha256_file
from .source import source_files


def _spdx_id(relative: str) -> str:
    return "SPDXRef-File-" + hashlib.sha1(relative.encode("utf-8")).hexdigest()


def _sha1_file(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as stream:
        while block := stream.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def generate_spdx(*, project_root: Path, output: Path) -> dict:
    files = source_files(project_root)
    records = []
    verification = []
    relationships = [
        {
            "spdxElementId": "SPDXRef-DOCUMENT",
            "relationshipType": "DESCRIBES",
            "relatedSpdxElement": "SPDXRef-Package",
        }
    ]
    for path in files:
        relative = path.relative_to(project_root).as_posix()
        digest = sha256_file(path)
        identifier = _spdx_id(relative)
        verification.append(_sha1_file(path))
        records.append(
            {
                "fileName": f"./{relative}",
                "SPDXID": identifier,
                "checksums": [{"algorithm": "SHA256", "checksumValue": digest}],
                "licenseConcluded": "NOASSERTION",
                "licenseInfoInFiles": ["NOASSERTION"],
                "copyrightText": "NOASSERTION",
            }
        )
        relationships.append(
            {
                "spdxElementId": "SPDXRef-Package",
                "relationshipType": "CONTAINS",
                "relatedSpdxElement": identifier,
            }
        )
    verification_code = hashlib.sha1("".join(sorted(verification)).encode("ascii")).hexdigest()
    epoch = int(os.environ.get("SOURCE_DATE_EPOCH", "0"))
    created = datetime.fromtimestamp(epoch, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    namespace = (
        "https://github.com/GitBubble/pico-minicpm5/spdx/"
        f"pico-minicpm5/{__version__}/{verification_code}"
    )
    document = {
        "spdxVersion": "SPDX-2.3",
        "dataLicense": "CC0-1.0",
        "SPDXID": "SPDXRef-DOCUMENT",
        "name": f"pico-minicpm5-{__version__}-source",
        "documentNamespace": namespace,
        "creationInfo": {
            "created": created,
            "creators": ["Tool: pico-minicpm5"],
            "licenseListVersion": "3.25",
        },
        "packages": [
            {
                "name": "pico-minicpm5",
                "SPDXID": "SPDXRef-Package",
                "versionInfo": __version__,
                "downloadLocation": "NOASSERTION",
                "filesAnalyzed": True,
                "packageVerificationCode": {
                    "packageVerificationCodeValue": verification_code
                },
                "licenseConcluded": "Apache-2.0",
                "licenseDeclared": "Apache-2.0",
                "copyrightText": "NOASSERTION",
            }
        ],
        "files": records,
        "relationships": relationships,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "schema": "pico.minicpm5.sbom.v1",
        "format": "SPDX-2.3",
        "path": output.name,
        "files": len(records),
        "sha256": sha256_file(output),
        "status": "PASS",
    }
