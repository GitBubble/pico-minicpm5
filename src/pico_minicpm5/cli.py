from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import sys

from . import __version__
from .assets import export_runtime_assets
from .calibration import build_calibration_bundle
from .compiler.atc import AtcCompiler
from .compiler.fake import FakeCompiler
from .contract import HF_REVISION, verify_checkpoint
from .hub import fetch_checkpoint
from .pipeline import build_three_handle
from .qualification import build_qualification
from .reference import DEFAULT_PROMPT_TOKENS, capture_reference
from .release.bundle import assemble_bundle, verify_bundle
from .release.source import create_source_archive
from .release.sbom import generate_spdx
from .score import (
    build_runtime_capture,
    score_head_output,
    score_public_outputs,
    write_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_CONFIG_ROOT = PROJECT_ROOT / "configs"
PACKAGE_CONFIG_ROOT = Path(__file__).resolve().parent / "data"
CONFIG_ROOT = SOURCE_CONFIG_ROOT if SOURCE_CONFIG_ROOT.is_dir() else PACKAGE_CONFIG_ROOT


def _clip_preset(family: str) -> Path:
    name = f"{family}-clips.json"
    candidates = (CONFIG_ROOT / "calibration" / name, CONFIG_ROOT / name)
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"packaged {family} clip preset is missing")


def _emit(value) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def _model_commands(subparsers) -> None:
    group = subparsers.add_parser("model", help="fetch and verify the pinned checkpoint")
    commands = group.add_subparsers(dest="model_command", required=True)
    fetch = commands.add_parser("fetch")
    fetch.add_argument("--local-dir", type=Path, required=True)
    fetch.add_argument("--revision", default=HF_REVISION)
    fetch.add_argument("--max-workers", type=int, default=4)
    fetch.add_argument("--dry-run", action="store_true")
    verify = commands.add_parser("verify")
    verify.add_argument("--model-dir", type=Path, required=True)
    verify.add_argument("--full-hash", action="store_true")


def _reference_commands(subparsers) -> None:
    group = subparsers.add_parser("reference", help="capture float references and calibration")
    commands = group.add_subparsers(dest="reference_command", required=True)
    capture = commands.add_parser("capture")
    capture.add_argument("--model-dir", type=Path, required=True)
    capture.add_argument("--out", type=Path, required=True)
    capture.add_argument("--context", type=int, default=1024)
    capture.add_argument("--dtype", choices=("float64", "float32"), default="float64")
    capture.add_argument(
        "--prompt-tokens", default=",".join(str(value) for value in DEFAULT_PROMPT_TOKENS)
    )
    calibrate = commands.add_parser("calibrate")
    calibrate.add_argument("--reference", type=Path, required=True)
    calibrate.add_argument("--out", type=Path, required=True)
    calibrate.add_argument("--family", choices=("prefill", "decode"), required=True)
    calibrate.add_argument("--context", type=int, default=1024)


def _onnx_commands(subparsers) -> None:
    group = subparsers.add_parser("onnx", help="actual-weight ONNX export and composition")
    commands = group.add_subparsers(dest="onnx_command", required=True)
    one = commands.add_parser("export-layer")
    one.add_argument("--model-dir", type=Path, required=True)
    one.add_argument("--layer", type=int, required=True)
    one.add_argument("--family", choices=("prefill", "decode"), required=True)
    one.add_argument("--context", type=int, default=1024)
    one.add_argument("--preset", type=Path)
    one.add_argument("--out", type=Path, required=True)
    all_layers = commands.add_parser("export-layers")
    all_layers.add_argument("--model-dir", type=Path, required=True)
    all_layers.add_argument("--family", choices=("prefill", "decode", "both"), default="both")
    all_layers.add_argument("--context", type=int, default=1024)
    all_layers.add_argument("--out", type=Path, required=True)
    head = commands.add_parser("export-head")
    head.add_argument("--model-dir", type=Path, required=True)
    head.add_argument("--out", type=Path, required=True)
    head.add_argument("--inline-data", action="store_true")
    compose = commands.add_parser("compose")
    compose.add_argument("--layers-dir", type=Path, required=True)
    compose.add_argument("--family", choices=("prefill", "decode"), required=True)
    compose.add_argument("--context", type=int, default=1024)
    compose.add_argument("--out", type=Path, required=True)
    compose.add_argument("--pack-input-kv", action=argparse.BooleanOptionalAction, default=True)
    compose.add_argument("--pack-output-kv", action=argparse.BooleanOptionalAction, default=True)
    compose.add_argument("--external-data", action=argparse.BooleanOptionalAction, default=True)


def _build_command(subparsers) -> None:
    command = subparsers.add_parser("build", help="compile the three OM handles")
    command.add_argument("--decode-onnx", type=Path, required=True)
    command.add_argument("--prefill-onnx", type=Path, required=True)
    command.add_argument("--head-onnx", type=Path, required=True)
    command.add_argument("--calibration", type=Path, required=True)
    command.add_argument("--out", type=Path, required=True)
    command.add_argument("--context", type=int, default=1024)
    command.add_argument("--backend", choices=("atc", "fake"), default="atc")
    command.add_argument("--atc", type=Path)
    command.add_argument("--custom-ops-lib", type=Path)
    command.add_argument("--dry-run", action="store_true")


def _release_commands(subparsers) -> None:
    group = subparsers.add_parser("release", help="source/model release assembly")
    commands = group.add_subparsers(dest="release_command", required=True)
    assemble = commands.add_parser("assemble")
    assemble.add_argument("--models", type=Path, required=True)
    assemble.add_argument("--model-dir", type=Path, required=True)
    assemble.add_argument("--qualification", type=Path)
    assemble.add_argument("--out", type=Path, required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("path", type=Path)
    source = commands.add_parser("source")
    source.add_argument("--project-root", type=Path)
    source.add_argument("--out", type=Path, default=PROJECT_ROOT / "artifacts")
    source.add_argument("--check-only", action="store_true")
    sbom = commands.add_parser("sbom")
    sbom.add_argument("--project-root", type=Path)
    sbom.add_argument("--out", type=Path, required=True)


def _qualification_command(subparsers) -> None:
    command = subparsers.add_parser(
        "qualify", help="combine prefill/decode numeric evidence into a release gate"
    )
    command.add_argument("--prefill-score", type=Path, required=True)
    command.add_argument("--decode-score", type=Path, required=True)
    command.add_argument("--head-score", type=Path, required=True)
    command.add_argument("--models", type=Path, required=True)
    command.add_argument("--greedy-report", type=Path)
    command.add_argument("--out", type=Path, required=True)


def _capture_command(subparsers) -> None:
    command = subparsers.add_parser(
        "capture", help="record hashes from one completed simulator or board execution"
    )
    command.add_argument("--runner", choices=("libinstsim", "ss928-board"), required=True)
    command.add_argument("--role", choices=("prefill", "decode", "head_flat"), required=True)
    command.add_argument("--position", type=int, required=True)
    command.add_argument("--context", type=int, default=1024)
    command.add_argument("--om", type=Path, required=True)
    command.add_argument("--build-manifest", type=Path, required=True)
    command.add_argument("--input", action="append", default=[], metavar="NAME=PATH")
    command.add_argument("--output", type=Path, action="append", required=True)
    command.add_argument("--report", type=Path, required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pico-minicpm5")
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("doctor", help="inspect open and external dependencies")
    _model_commands(subparsers)
    _reference_commands(subparsers)
    _onnx_commands(subparsers)
    assets = subparsers.add_parser("assets", help="export embedding and tokenizer assets")
    assets.add_argument("--model-dir", type=Path, required=True)
    assets.add_argument("--out", type=Path, required=True)
    _build_command(subparsers)
    score = subparsers.add_parser("score", help="strictly score three raw public outputs")
    score.add_argument("--output", type=Path, action="append", required=True)
    score.add_argument("--reference", type=Path, required=True)
    score.add_argument("--position", type=int, required=True)
    score.add_argument("--om", type=Path, required=True)
    score.add_argument("--build-manifest", type=Path, required=True)
    score.add_argument("--capture-manifest", type=Path, required=True)
    score.add_argument("--context", type=int, default=1024)
    score.add_argument("--threshold-exclusive", type=float, default=0.98)
    score.add_argument("--report", type=Path)
    head_score = subparsers.add_parser("score-head", help="score and bind dense head logits")
    head_score.add_argument("--output", type=Path, required=True)
    head_score.add_argument("--reference", type=Path, required=True)
    head_score.add_argument("--position", type=int, required=True)
    head_score.add_argument("--om", type=Path, required=True)
    head_score.add_argument("--build-manifest", type=Path, required=True)
    head_score.add_argument("--capture-manifest", type=Path, required=True)
    head_score.add_argument("--hidden-input", type=Path, required=True)
    head_score.add_argument("--residual-input", type=Path, required=True)
    head_score.add_argument("--context", type=int, default=1024)
    head_score.add_argument("--threshold-exclusive", type=float, default=0.98)
    head_score.add_argument("--report", type=Path, required=True)
    _capture_command(subparsers)
    _qualification_command(subparsers)
    _release_commands(subparsers)
    return parser


def _doctor() -> dict:
    optional = {
        name: importlib.util.find_spec(name) is not None
        for name in ("numpy", "onnx", "safetensors", "torch", "transformers")
    }
    tools = {name: shutil.which(name) for name in ("hf", "atc", "docker", "git")}
    return {
        "schema": "pico.minicpm5.doctor.v1",
        "version": __version__,
        "python": sys.version.split()[0],
        "python_modules": optional,
        "external_tools": tools,
        "source_pipeline_ready": all(optional[name] for name in ("numpy", "onnx", "safetensors")),
        "reference_ready": all(optional[name] for name in ("torch", "transformers")),
        "atc_is_external": True,
    }


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "doctor":
        _emit(_doctor())
        return 0
    if args.command == "model":
        if args.model_command == "fetch":
            _emit(fetch_checkpoint(args.local_dir, revision=args.revision,
                                   max_workers=args.max_workers, dry_run=args.dry_run))
        else:
            _emit(verify_checkpoint(args.model_dir, full_hash=args.full_hash))
        return 0
    if args.command == "reference":
        if args.reference_command == "capture":
            tokens = tuple(int(value) for value in args.prompt_tokens.split(",") if value.strip())
            _emit(capture_reference(model_dir=args.model_dir, output=args.out,
                                    context=args.context, prompt_tokens=tokens,
                                    dtype_name=args.dtype))
        else:
            _emit(build_calibration_bundle(reference=args.reference, output=args.out,
                                           family=args.family, context=args.context))
        return 0
    if args.command == "onnx":
        from .onnx.compose import compose_files
        from .onnx.head import export_head
        from .onnx.layer import export_layer

        if args.onnx_command == "export-layer":
            preset = args.preset or _clip_preset(args.family)
            _emit(export_layer(model_dir=args.model_dir, layer=args.layer,
                               family=args.family, context=args.context,
                               preset=preset, output=args.out))
        elif args.onnx_command == "export-layers":
            families = ("prefill", "decode") if args.family == "both" else (args.family,)
            reports = []
            for family in families:
                preset = _clip_preset(family)
                for layer in range(24):
                    output = args.out / family / f"layer{layer:02d}.onnx"
                    reports.append(export_layer(model_dir=args.model_dir, layer=layer,
                                                family=family, context=args.context,
                                                preset=preset, output=output))
            _emit({"schema": "pico.minicpm5.layer-export-set.v1", "builds": reports})
        elif args.onnx_command == "export-head":
            _emit(export_head(model_dir=args.model_dir, output=args.out,
                              external_data=not args.inline_data))
        else:
            paths = [args.layers_dir / f"layer{layer:02d}.onnx" for layer in range(24)]
            missing = [str(path) for path in paths if not path.is_file()]
            if missing:
                raise FileNotFoundError(f"missing layer ONNX files: {missing[:4]}")
            _emit(compose_files(layer_paths=paths, output=args.out, family=args.family,
                                context=args.context, pack_input_kv=args.pack_input_kv,
                                pack_output_kv=args.pack_output_kv,
                                external_data=args.external_data))
        return 0
    if args.command == "assets":
        _emit(export_runtime_assets(model_dir=args.model_dir, output=args.out, full_hash=True))
        return 0
    if args.command == "build":
        if args.backend == "fake":
            compiler = FakeCompiler()
        else:
            if args.atc is None or args.custom_ops_lib is None:
                raise SystemExit("ATC backend requires --atc and --custom-ops-lib")
            compiler = AtcCompiler(atc=args.atc, custom_ops_lib=args.custom_ops_lib,
                                   dry_run=args.dry_run)
        _emit(build_three_handle(compiler=compiler, decode_onnx=args.decode_onnx,
                                 prefill_onnx=args.prefill_onnx, head_onnx=args.head_onnx,
                                 calibration_root=args.calibration, output=args.out,
                                 context=args.context))
        return 0
    if args.command == "capture":
        inputs = {}
        for item in args.input:
            if "=" not in item:
                raise SystemExit("capture --input must use NAME=PATH")
            name, raw_path = item.split("=", 1)
            if not name or name in inputs or not raw_path:
                raise SystemExit("capture --input names must be nonempty and unique")
            inputs[name] = Path(raw_path)
        _emit(
            build_runtime_capture(
                runner=args.runner,
                role=args.role,
                position=args.position,
                context=args.context,
                om=args.om,
                build_manifest=args.build_manifest,
                inputs=inputs,
                outputs=tuple(args.output),
                report=args.report,
            )
        )
        return 0
    if args.command == "score":
        if len(args.output) != 3:
            raise SystemExit("score requires exactly three --output files")
        report = score_public_outputs(output_files=tuple(args.output), reference=args.reference,
                                      position=args.position, om=args.om,
                                      build_manifest=args.build_manifest,
                                      capture_manifest=args.capture_manifest,
                                      context=args.context,
                                      threshold_exclusive=args.threshold_exclusive)
        if args.report:
            write_score(report, args.report)
        _emit(report)
        return 0 if report["status"] == "PASS" else 1
    if args.command == "score-head":
        report = score_head_output(
            output_file=args.output,
            reference=args.reference,
            position=args.position,
            om=args.om,
            build_manifest=args.build_manifest,
            capture_manifest=args.capture_manifest,
            hidden_input=args.hidden_input,
            residual_input=args.residual_input,
            context=args.context,
            threshold_exclusive=args.threshold_exclusive,
        )
        write_score(report, args.report)
        _emit(report)
        return 0 if report["status"] == "PASS" else 1
    if args.command == "qualify":
        report = build_qualification(
            prefill_score=args.prefill_score,
            decode_score=args.decode_score,
            head_score=args.head_score,
            models=args.models,
            output=args.out,
            greedy_report=args.greedy_report,
        )
        _emit(report)
        return 0 if report["verdict"]["overall"] == "PASS" else 1
    if args.command == "release":
        if args.release_command == "assemble":
            _emit(assemble_bundle(models=args.models, model_dir=args.model_dir,
                                  output=args.out, qualification_path=args.qualification))
        elif args.release_command == "verify":
            _emit(verify_bundle(args.path))
        elif args.release_command == "source":
            project_root = args.project_root or PROJECT_ROOT
            if not (project_root / "pyproject.toml").is_file():
                raise SystemExit(
                    "release source must run from a source checkout or receive --project-root"
                )
            _emit(create_source_archive(project_root=project_root, output_dir=args.out,
                                        check_only=args.check_only))
        else:
            project_root = args.project_root or PROJECT_ROOT
            if not (project_root / "pyproject.toml").is_file():
                raise SystemExit(
                    "release sbom must run from a source checkout or receive --project-root"
                )
            _emit(generate_spdx(project_root=project_root, output=args.out))
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
