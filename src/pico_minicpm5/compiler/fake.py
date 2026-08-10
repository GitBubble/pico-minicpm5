from __future__ import annotations

import hashlib
import json

from .base import CompileRequest
from ..contract import sha256_file


class FakeCompiler:
    """Deterministic CI backend that exercises contracts without an SDK."""

    def compile(self, request: CompileRequest) -> dict:
        calibration = []
        for name in request.inputs:
            path = request.calibration_dir / f"{request.calibration_stem}.{name}.image_list"
            if not path.is_file():
                raise FileNotFoundError(path)
            calibration.append((name, sha256_file(path)))
        payload = json.dumps(
            {
                "role": request.role,
                "model_sha256": sha256_file(request.model),
                "calibration": calibration,
                "inputs": request.inputs,
                "input_shape": request.input_shape,
                "input_type": request.input_type,
                "npu_arch": request.npu_arch,
            },
            sort_keys=True,
        ).encode("utf-8")
        request.output.parent.mkdir(parents=True, exist_ok=True)
        request.output.write_bytes(b"PICO" + hashlib.sha256(payload).digest() + payload)
        return {
            "schema": "pico.minicpm5.fake-compile.v1",
            "role": request.role,
            "output": str(request.output.resolve()),
            "bytes": request.output.stat().st_size,
            "backend": "fake",
        }
