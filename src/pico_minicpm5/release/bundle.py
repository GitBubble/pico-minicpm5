"""Assemble a local three-handle model bundle from explicitly supplied assets."""
from __future__ import annotations

import json
from importlib import metadata as importlib_metadata
import hashlib
import math
from pathlib import Path
import shutil
import tempfile

from .. import __version__
from ..assets import export_runtime_assets
from ..contract import (
    HF_REPO_ID,
    HF_REVISION,
    PUBLIC_COSINE_THRESHOLD_EXCLUSIVE,
    sha256_file,
)
from .manifest import artifact_record, load_json, write_checksums

MODEL_NAMES = ("decode.om", "prefill.om", "head_flat.om")
MODEL_QUALIFICATION_ROLES = {
    "decode.om": "decode",
    "prefill.om": "prefill",
    "head_flat.om": "head_flat",
}
FROZEN_ACCEPTED = {
    "decode.om": "59d6bfc9f66edbacc90896f723320041fc6c352d85306054e090b13ecaf774b1",
    "prefill.om": "fea09d294513839ccab846050193def475859f4352e2cb3fad7c2f8f1dfa4536",
    "head_flat.om": "fd6675f4aad9d7244471afc637f4964c79a7504b2c73a350985f2d807105330e",
}
REQUIRED_ARTIFACT_ROLES = {
    "models/decode.om": "decode-transformer",
    "models/prefill.om": "position-zero-transformer",
    "models/head_flat.om": "vocabulary-head",
    "assets/token_embedding.f16.bin": "token-embedding",
    "assets/tokenizer.json": "tokenizer",
    "assets/assets-manifest.json": "asset-manifest",
    "LICENSE": "software-license",
    "NOTICE": "software-notice",
    "MODEL_LICENSE_NOTICE.md": "model-license-notice",
}
ASSET_HASHES = {
    "assets/token_embedding.f16.bin": "5a93b589f0c5920021c95e04327c0771da2721d8eec2dd7ac1b283aa0d3b7df5",
    "assets/tokenizer.json": "3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81",
}
CONTEXT = 1024
ZERO_RESIDUAL = {
    "bytes": 1536 * 4,
    "sha256": hashlib.sha256(bytes(1536 * 4)).hexdigest(),
}


def _valid_digest(value) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _valid_hash_record(value) -> bool:
    return (
        isinstance(value, dict)
        and isinstance(value.get("bytes"), int)
        and not isinstance(value.get("bytes"), bool)
        and value["bytes"] > 0
        and _valid_digest(value.get("sha256"))
    )


def _real_number(value) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _valid_score_execution(
    record: dict,
    *,
    role: str,
    artifact: dict,
    compiler: dict,
    position: int,
    outputs: list,
    inputs: dict | None = None,
) -> bool:
    execution = record.get("execution")
    if not (
        isinstance(execution, dict)
        and execution.get("role") == role
        and execution.get("om") == artifact
        and execution.get("compiler") == compiler
    ):
        return False
    capture = execution.get("capture")
    if not isinstance(capture, dict):
        return False
    captured_inputs = capture.get("inputs")
    captured_outputs = capture.get("outputs")
    return (
        _valid_digest(capture.get("manifest_sha256"))
        and capture.get("runner") in {"libinstsim", "ss928-board"}
        and capture.get("status") == "PASS"
        and capture.get("position") == position
        and capture.get("context") == CONTEXT
        and isinstance(captured_inputs, dict)
        and all(_valid_hash_record(value) for value in captured_inputs.values())
        and captured_outputs == outputs
        and all(_valid_hash_record(value) for value in captured_outputs)
        and (inputs is None or captured_inputs == inputs)
    )


def _qualified(qualification: dict) -> bool:
    schema = qualification.get("schema")
    try:
        declared = float(
            qualification.get(
                "threshold_exclusive",
                qualification.get(
                    "cosine_threshold_exclusive", PUBLIC_COSINE_THRESHOLD_EXCLUSIVE
                ),
            )
        )
    except (TypeError, ValueError):
        return False
    if not math.isfinite(declared) or not 0.0 < declared < 1.0:
        return False
    threshold = max(PUBLIC_COSINE_THRESHOLD_EXCLUSIVE, declared)
    if schema == "pico.minicpm5.qualification.v1":
        families = qualification.get("families", {})
        artifacts = qualification.get("artifacts", {})
        compiler = qualification.get("compiler", {})
        build_digest = compiler.get("build_manifest_sha256") if isinstance(compiler, dict) else None
        if (
            not isinstance(families, dict)
            or not isinstance(artifacts, dict)
            or set(families) != {"prefill", "decode"}
            or set(artifacts) != set(MODEL_QUALIFICATION_ROLES.values())
            or not isinstance(compiler, dict)
            or compiler.get("backend") != "atc"
            or not _valid_digest(build_digest)
            or any(not _valid_hash_record(record) for record in artifacts.values())
            or qualification.get("model")
            != {"repository": HF_REPO_ID, "revision": HF_REVISION}
            or qualification.get("target")
            != {"soc": "Hi3403", "npu_arch": "V101", "context": 1024}
        ):
            return False
        expected_compiler = {"backend": "atc", "build_manifest_sha256": build_digest}
        numeric = True
        for role, record in families.items():
            if not isinstance(record, dict):
                return False
            position = record.get("position")
            cosine = record.get("cosine")
            minimum = record.get("minimum_public_output")
            outputs = record.get("outputs")
            logical = record.get("logical_outputs")
            if (
                (
                    role == "prefill"
                    and (position != 0 or isinstance(position, bool))
                )
                or (
                    role == "decode"
                    and (
                        not isinstance(position, int)
                        or isinstance(position, bool)
                        or position < 1
                    )
                )
                or record.get("context") != CONTEXT
                or not isinstance(cosine, dict)
                or set(cosine) != {"next_hidden", "k_cur_all", "v_cur_all"}
                or not isinstance(outputs, list)
                or len(outputs) != 3
                or any(not _valid_hash_record(output) for output in outputs)
                or not isinstance(logical, dict)
                or set(logical) != {"next_hidden"}
                or not _valid_hash_record(logical.get("next_hidden"))
                or record.get("output_order") not in {"direct", "swapped"}
                or not _valid_score_execution(
                    record,
                    role=role,
                    artifact=artifacts[role],
                    compiler=expected_compiler,
                    position=position,
                    outputs=outputs,
                )
            ):
                return False
            try:
                if (
                    not _real_number(minimum)
                    or any(not _real_number(value) for value in cosine.values())
                ):
                    return False
                values = [float(cosine[name]) for name in ("next_hidden", "k_cur_all", "v_cur_all")]
                minimum_value = float(minimum)
            except (TypeError, ValueError):
                return False
            if (
                any(not math.isfinite(value) or value <= threshold or value > 1.0 for value in values)
                or not math.isclose(minimum_value, min(values), rel_tol=0.0, abs_tol=1e-12)
            ):
                numeric = False
        head = qualification.get("head")
        if not isinstance(head, dict):
            return False
        try:
            if not _real_number(head.get("cosine")) or not _real_number(
                head.get("minimum_public_output")
            ):
                return False
            head_cosine = float(head.get("cosine"))
            head_minimum = float(head.get("minimum_public_output"))
        except (TypeError, ValueError):
            return False
        position = head.get("position")
        inputs = head.get("inputs")
        output = head.get("output")
        top1_output, top1_reference = head.get("top1_output"), head.get("top1_reference")
        if position == 0:
            source_role = "prefill"
        elif isinstance(position, int) and not isinstance(position, bool) and position >= 1:
            source_role = "decode"
        else:
            return False
        source = families[source_role]
        head_numeric = (
            math.isfinite(head_cosine)
            and threshold < head_cosine <= 1.0
            and math.isclose(head_cosine, head_minimum, rel_tol=0.0, abs_tol=1e-12)
            and head.get("top1_exact") is True
            and isinstance(top1_output, int)
            and not isinstance(top1_output, bool)
            and isinstance(top1_reference, int)
            and not isinstance(top1_reference, bool)
            and top1_output == top1_reference
            and head.get("context") == CONTEXT
            and head.get("source_transformer_role") == source_role
            and source.get("position") == position
            and isinstance(inputs, dict)
            and set(inputs) == {"hidden", "residual"}
            and inputs.get("residual") == ZERO_RESIDUAL
            and inputs.get("hidden") == source["logical_outputs"]["next_hidden"]
            and _valid_hash_record(output)
            and _valid_score_execution(
                head,
                role="head_flat",
                artifact=artifacts["head_flat"],
                compiler=expected_compiler,
                position=position,
                outputs=[output],
                inputs=inputs,
            )
        )
        overall_minimum = qualification.get("minimum_public_output")
        try:
            minimum_matches = math.isclose(
                float(overall_minimum),
                min(
                    head_minimum,
                    *(float(record["minimum_public_output"]) for record in families.values()),
                ),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        except (TypeError, ValueError):
            minimum_matches = False
        verdict = qualification.get("verdict", {})
        return (
            numeric
            and head_numeric
            and minimum_matches
            and verdict.get("public_output_numeric") == "PASS"
            and verdict.get("head_numeric") == "PASS"
            and verdict.get("overall") == "PASS"
        )
    if schema == "pico.minicpm5.merged3handle.qualification.v1":
        legacy_artifacts = qualification.get("artifacts", {})
        if (
            not isinstance(legacy_artifacts, dict)
            or set(legacy_artifacts) != set(MODEL_QUALIFICATION_ROLES.values())
            or any(
                not _valid_hash_record(legacy_artifacts.get(role))
                or legacy_artifacts[role].get("sha256") != FROZEN_ACCEPTED[name]
                for name, role in MODEL_QUALIFICATION_ROLES.items()
            )
        ):
            return False
        records = qualification.get("board_numeric", [])
        by_model = {
            record.get("model"): record for record in records if isinstance(record, dict)
        }
        try:
            numeric = set(by_model) >= {"prefill", "decode"} and all(
                _real_number(
                    by_model[role].get("cosine", {}).get("minimum_public_output")
                )
                and float(
                    by_model[role]["cosine"]["minimum_public_output"]
                ) > threshold
                for role in ("prefill", "decode")
            )
        except (TypeError, ValueError):
            return False
        verdict = qualification.get("verdict", {})
        required = ("board_load", "public_output_numeric", "greedy_exact", "eos", "overall")
        return numeric and all(verdict.get(key) == "PASS" for key in required)
    return False


def _qualification_matches_models(qualification: dict, artifacts: dict) -> bool:
    evidence = qualification.get("artifacts")
    if not isinstance(evidence, dict):
        return False
    for name, role in MODEL_QUALIFICATION_ROLES.items():
        expected = evidence.get(role)
        actual = artifacts.get(f"models/{name}")
        if not isinstance(expected, dict) or not isinstance(actual, dict):
            return False
        if expected.get("sha256") != actual.get("sha256"):
            return False
        if int(expected.get("bytes", -1)) != int(actual.get("bytes", -2)):
            return False
    return True


def _legal_source(name: str) -> Path:
    source_candidate = Path(__file__).resolve().parents[3] / name
    if source_candidate.is_file():
        return source_candidate
    try:
        installed = importlib_metadata.files("pico-minicpm5") or ()
    except importlib_metadata.PackageNotFoundError:
        installed = ()
    for entry in installed:
        parts = Path(str(entry)).parts
        if entry.name == name and "licenses" in parts:
            located = Path(entry.locate())
            if located.is_file():
                return located
    raise FileNotFoundError(f"installed project does not provide {name}")


def _portable(value):
    if isinstance(value, dict):
        return {key: _portable(item) for key, item in value.items() if key != "path"}
    if isinstance(value, list):
        return [_portable(item) for item in value]
    if isinstance(value, str) and Path(value).is_absolute():
        return Path(value).name
    return value


def _assemble_bundle_into(
    *,
    models: Path,
    model_dir: Path,
    output: Path,
    qualification_path: Path | None = None,
) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    model_output, asset_output = output / "models", output / "assets"
    model_output.mkdir()
    asset_output.mkdir()
    artifacts = {}
    exact_frozen = True
    for name in MODEL_NAMES:
        source = models / name
        magic = b""
        if source.is_file():
            with source.open("rb") as stream:
                magic = stream.read(4)
        if not source.is_file() or magic != b"PICO":
            raise ValueError(f"missing or invalid PICO model: {source}")
        target = model_output / name
        shutil.copyfile(source, target)
        role = {"decode.om": "decode-transformer", "prefill.om": "position-zero-transformer",
                "head_flat.om": "vocabulary-head"}[name]
        record = artifact_record(target, role=role, relationship="derived-model")
        artifacts[f"models/{name}"] = record
        exact_frozen &= record["sha256"] == FROZEN_ACCEPTED[name]

    qualification = None
    if qualification_path is not None:
        qualification = load_json(qualification_path)
        if not _qualified(qualification):
            raise ValueError("qualification does not satisfy the strict numeric gate")
        if not _qualification_matches_models(qualification, artifacts):
            raise ValueError("qualification OM hashes do not match the release models")
        qualification = _portable(qualification)
        qualification_output = output / "qualification.json"
        qualification_output.write_text(
            json.dumps(qualification, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        artifacts["qualification.json"] = artifact_record(
            qualification_output, role="numeric-qualification", relationship="release-metadata"
        )
    elif not exact_frozen:
        raise ValueError(
            "new OM hashes require an explicit passing qualification; "
            "only the frozen accepted hashes may reuse the recorded v0.1.0 verdict"
        )

    asset_report = export_runtime_assets(
        model_dir=model_dir, output=asset_output, full_hash=True
    )
    for name, record in asset_report["assets"].items():
        artifacts[f"assets/{name}"] = {
            "role": "token-embedding" if name.startswith("token_embedding") else "tokenizer",
            **record,
        }
    assets_manifest = asset_output / "assets-manifest.json"
    artifacts["assets/assets-manifest.json"] = artifact_record(
        assets_manifest, role="asset-manifest", relationship="release-metadata"
    )
    for name in ("LICENSE", "NOTICE"):
        target = output / name
        shutil.copyfile(_legal_source(name), target)
        artifacts[name] = artifact_record(
            target, role="software-license" if name == "LICENSE" else "software-notice",
            relationship="project-metadata",
        )
    model_notice = output / "MODEL_LICENSE_NOTICE.md"
    model_notice.write_text(
        "# Model license notice\n\n"
        f"The model card for `{HF_REPO_ID}` at revision `{HF_REVISION}` declares "
        "Apache-2.0. The OM files and token embedding in this bundle are derived "
        "from that separately obtained checkpoint. See `LICENSE` and the pinned "
        "model card; no ATC/DDK or runtime library is included.\n",
        encoding="utf-8",
    )
    artifacts[model_notice.name] = artifact_record(
        model_notice, role="model-license-notice", relationship="derived-model-metadata"
    )
    manifest = {
        "schema": "pico.minicpm5.local-model-release.v1",
        "source": {"package": "pico-minicpm5", "version": __version__},
        "model": {"repository": HF_REPO_ID, "revision": HF_REVISION},
        "target": {"soc": "Hi3403", "npu_arch": "V101", "context": 1024},
        "abi": {
            "handles": 3,
            "transformer_public_inputs": 5,
            "transformer_runtime_inputs": 7,
            "transformer_outputs": 3,
        },
        "qualification": (
            qualification if qualification is not None else
            {"source": "frozen-v0.1.0-hashes", "overall": "PASS"}
        ),
        "artifacts": artifacts,
        "external_runtime_required": True,
        "excluded": [
            "ATC/DDK/libinstsim",
            "custom-op shared objects",
            "board runtime libraries",
            "prebuilt executor binary",
        ],
    }
    manifest_path = output / "release-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    sums_path = output / "SHA256SUMS"
    files = [path for path in output.rglob("*") if path.is_file() and path != sums_path]
    write_checksums(output, files)
    return manifest


def assemble_bundle(
    *,
    models: Path,
    model_dir: Path,
    output: Path,
    qualification_path: Path | None = None,
) -> dict:
    """Assemble in a same-filesystem staging directory, then publish atomically."""
    if output.exists():
        raise FileExistsError(f"release output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent)
    )
    try:
        manifest = _assemble_bundle_into(
            models=models,
            model_dir=model_dir,
            output=staging,
            qualification_path=qualification_path,
        )
        staging.replace(output)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def verify_bundle(root: Path) -> dict:
    manifest_path, sums_path = root / "release-manifest.json", root / "SHA256SUMS"
    if not manifest_path.is_file() or not sums_path.is_file():
        raise FileNotFoundError("release-manifest.json or SHA256SUMS is missing")
    symlinks = [path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_symlink()]
    if symlinks:
        raise ValueError(f"release bundle must not contain symlinks: {symlinks[:4]}")
    manifest = load_json(manifest_path)
    if manifest.get("schema") != "pico.minicpm5.local-model-release.v1":
        raise ValueError("unsupported release manifest schema")
    if manifest.get("source") != {"package": "pico-minicpm5", "version": __version__}:
        raise ValueError("release source version drift")
    if manifest.get("model") != {"repository": HF_REPO_ID, "revision": HF_REVISION}:
        raise ValueError("release model provenance drift")
    target = manifest.get("target", {})
    if target != {"soc": "Hi3403", "npu_arch": "V101", "context": 1024}:
        raise ValueError("release target contract drift")
    abi = manifest.get("abi", {})
    required_abi = {
        "handles": 3,
        "transformer_public_inputs": 5,
        "transformer_runtime_inputs": 7,
        "transformer_outputs": 3,
    }
    if abi != required_abi:
        raise ValueError("release ABI contract drift")

    expected = {}
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        try:
            digest, relative = line.split("  ", 1)
        except ValueError as exc:
            raise ValueError(f"malformed checksum line: {line!r}") from exc
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts or relative in expected:
            raise ValueError(f"unsafe or duplicate checksum path: {relative!r}")
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError(f"invalid SHA256 in checksum line: {line!r}")
        expected[relative] = digest
    errors = []
    for relative, digest in expected.items():
        path = root / relative
        if path.is_symlink() or not path.is_file() or sha256_file(path) != digest:
            errors.append(relative)
    actual_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != sums_path
    }
    extras = sorted(actual_files - set(expected))
    missing_checksums = sorted(set(expected) - actual_files)
    if "release-manifest.json" not in expected:
        errors.append("release-manifest.json")
    if errors or extras:
        raise ValueError(
            "release checksum failure: "
            f"mismatches={errors + missing_checksums} extras={extras}"
        )

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("release manifest has no artifact table")
    if not set(REQUIRED_ARTIFACT_ROLES).issubset(artifacts):
        missing = sorted(set(REQUIRED_ARTIFACT_ROLES) - set(artifacts))
        raise ValueError(f"release manifest is missing required artifacts: {missing}")
    declared_files = set(artifacts)
    allowed_metadata = {"release-manifest.json"}
    if actual_files != declared_files | allowed_metadata:
        raise ValueError("manifest artifact table does not cover every release file")
    for relative, record in artifacts.items():
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"unsafe manifest artifact path: {relative}")
        path = root / relative_path
        if (
            not isinstance(record, dict)
            or not path.is_file()
            or record.get("sha256") != sha256_file(path)
            or int(record.get("bytes", -1)) != path.stat().st_size
        ):
            raise ValueError(f"manifest artifact mismatch: {relative}")
    for relative, role in REQUIRED_ARTIFACT_ROLES.items():
        if artifacts[relative].get("role") != role:
            raise ValueError(f"manifest artifact role mismatch: {relative}")
    for relative, digest in ASSET_HASHES.items():
        if artifacts[relative].get("sha256") != digest:
            raise ValueError(f"release asset hash drift: {relative}")

    asset_manifest = load_json(root / "assets" / "assets-manifest.json")
    if asset_manifest.get("schema") != "pico.minicpm5.runtime-assets.v1":
        raise ValueError("unsupported runtime asset manifest")
    asset_records = asset_manifest.get("assets")
    if not isinstance(asset_records, dict) or set(asset_records) != {
        "token_embedding.f16.bin", "tokenizer.json"
    }:
        raise ValueError("runtime asset manifest is incomplete")
    for name, record in asset_records.items():
        release_record = artifacts[f"assets/{name}"]
        if (
            not _valid_hash_record(record)
            or record.get("bytes") != release_record.get("bytes")
            or record.get("sha256") != release_record.get("sha256")
        ):
            raise ValueError(f"runtime asset manifest mismatch: {name}")
    for name in MODEL_NAMES:
        with (root / "models" / name).open("rb") as stream:
            if stream.read(4) != b"PICO":
                raise ValueError(f"invalid model magic: {name}")

    exact_frozen = all(
        artifacts[f"models/{name}"].get("sha256") == digest
        for name, digest in FROZEN_ACCEPTED.items()
    )
    qualification = manifest.get("qualification", {})
    qualification_file = root / "qualification.json"
    if qualification_file.is_file():
        if load_json(qualification_file) != qualification:
            raise ValueError("qualification file and manifest disagree")
        if not _qualified(qualification) or not _qualification_matches_models(
            qualification, artifacts
        ):
            raise ValueError("release qualification is invalid or bound to other OMs")
    elif not (
        exact_frozen
        and qualification == {"source": "frozen-v0.1.0-hashes", "overall": "PASS"}
    ):
        raise ValueError("non-frozen release is missing portable qualification")
    return {"schema": "pico.minicpm5.release-verification.v1", "files": len(expected), "status": "PASS"}
