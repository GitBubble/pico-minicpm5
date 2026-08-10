from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CompileRequest:
    role: str
    model: Path
    output: Path
    calibration_dir: Path
    calibration_stem: str
    inputs: tuple[str, ...]
    input_shape: str
    input_type: str
    npu_arch: str = "V101"

    def image_list(self) -> str:
        entries = []
        for name in self.inputs:
            path = self.calibration_dir / f"{self.calibration_stem}.{name}.image_list"
            if not path.is_file():
                raise FileNotFoundError(path)
            entries.append(f"{name}:{path.resolve()}")
        return ";".join(entries)
