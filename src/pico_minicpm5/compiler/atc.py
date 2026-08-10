"""External ATC backend. No SDK or vendor library is bundled here."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess

from ..contract import sha256_file
from .base import CompileRequest


class AtcCompiler:
    def __init__(
        self,
        *,
        atc: Path,
        custom_ops_lib: Path,
        compile_mode: int = 1,
        online_model_type: int = 1,
        dump_data: int = 4,
        dry_run: bool = False,
    ) -> None:
        self.atc = atc.resolve()
        self.custom_ops_lib = custom_ops_lib.resolve()
        self.compile_mode = compile_mode
        self.online_model_type = online_model_type
        self.dump_data = dump_data
        self.dry_run = dry_run
        if not dry_run:
            if not self.atc.is_file():
                raise FileNotFoundError(self.atc)
            if not self.custom_ops_lib.is_file():
                raise FileNotFoundError(self.custom_ops_lib)

    @staticmethod
    def _external_locations(model: Path) -> list[str]:
        try:
            import onnx
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("install pico-minicpm5[onnx]") from exc
        graph = onnx.load(str(model), load_external_data=False).graph
        locations = set()
        for initializer in graph.initializer:
            for entry in initializer.external_data:
                if entry.key == "location":
                    location = Path(entry.value)
                    if (
                        not entry.value
                        or location.is_absolute()
                        or ".." in location.parts
                    ):
                        raise ValueError(
                            f"unsafe ONNX external-data location: {entry.value!r}"
                        )
                    locations.add(location.as_posix())
        return sorted(locations)

    def command(self, request: CompileRequest) -> list[str]:
        return [
            str(self.atc),
            f"--model={request.model.resolve()}",
            "--framework=5",
            f"--npu_arch={request.npu_arch}",
            f"--input_shape={request.input_shape}",
            f"--input_type={request.input_type}",
            f"--image_list={request.image_list()}",
            f"--custom_ops_lib={self.custom_ops_lib}",
            f"--compile_mode={self.compile_mode}",
            f"--online_model_type={self.online_model_type}",
            f"--dump_data={self.dump_data}",
        ]

    def compile(self, request: CompileRequest) -> dict:
        command = self.command(request)
        if self.dry_run:
            return {"backend": "atc", "dry_run": True, "command": command}
        work = request.output.parent / f"atc_{request.role}"
        if work.exists():
            raise FileExistsError(f"ATC work directory already exists: {work}")
        work.mkdir(parents=True)
        for location in self._external_locations(request.model):
            source = request.model.parent / location
            if not source.is_file():
                raise FileNotFoundError(source)
            link = work / location
            link.parent.mkdir(parents=True, exist_ok=True)
            if link.exists() or link.is_symlink():
                raise FileExistsError(link)
            link.symlink_to(os.path.relpath(source, link.parent))
        completed = subprocess.run(command, cwd=work, text=True, capture_output=True)
        (work / "atc.stdout.log").write_text(completed.stdout, encoding="utf-8")
        (work / "atc.stderr.log").write_text(completed.stderr, encoding="utf-8")
        generated = work / "inst.om"
        magic = b""
        if generated.is_file():
            with generated.open("rb") as stream:
                magic = stream.read(4)
        if completed.returncode or magic != b"PICO":
            tail = (completed.stderr or completed.stdout)[-2000:]
            raise RuntimeError(f"ATC failed for {request.role}:\n{tail}")
        request.output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(generated, request.output)
        report = {
            "schema": "pico.minicpm5.atc-compile.v1",
            "role": request.role,
            "backend": "atc",
            "command": command,
            "model": str(request.model.resolve()),
            "model_sha256": sha256_file(request.model),
            "atc_sha256": sha256_file(self.atc),
            "custom_ops_lib_sha256": sha256_file(self.custom_ops_lib),
            "output": str(request.output.resolve()),
            "output_bytes": request.output.stat().st_size,
            "output_sha256": sha256_file(request.output),
        }
        (request.output.with_suffix(request.output.suffix + ".build.json")).write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return report
