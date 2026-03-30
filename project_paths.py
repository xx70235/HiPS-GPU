"""Common path helpers for the paper code bundle."""

from __future__ import annotations

import os
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
CORE_ROOT = PROJECT_ROOT / "core"
RUNTIME_ROOT = PROJECT_ROOT / "runtime"
VALIDATION_ROOT = PROJECT_ROOT / "validation"
CASE_STUDY_ROOT = PROJECT_ROOT / "case_studies"

DEFAULT_DATA_ROOT = PROJECT_ROOT.parent / "paper_figure_sources_v3" / "data"
DATA_ROOT = Path(os.environ.get("HIPS_GPU_PAPER_DATA_ROOT", DEFAULT_DATA_ROOT))

DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "generated_outputs"
OUTPUT_ROOT = Path(os.environ.get("HIPS_GPU_PAPER_OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT))

DEFAULT_DOCS_ROOT = OUTPUT_ROOT / "docs"
DOCS_ROOT = Path(os.environ.get("HIPS_GPU_PAPER_DOCS_ROOT", DEFAULT_DOCS_ROOT))


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def default_hipsgen_bin() -> str:
    env = os.environ.get("HIPSGEN_BIN")
    if env:
        return env

    candidates = [
        CORE_ROOT / "bin" / "hipsgen_cuda",
        PROJECT_ROOT / "bin" / "hipsgen_cuda",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    found = shutil.which("hipsgen_cuda")
    if found:
        return found

    return str(candidates[0])
