"""Public interfaces backed only by the bundled compiled ST-Tensor/L1 modules."""

from pathlib import Path as _Path
import platform as _platform
import sys as _sys
from importlib.machinery import EXTENSION_SUFFIXES as _EXTENSION_SUFFIXES

_directory = _Path(__file__).resolve().parent
_tag = f"cp{_sys.version_info.major}{_sys.version_info.minor}"
_machine = _platform.machine().lower()
_machine = {"amd64": "x86_64", "aarch64": "arm64"}.get(_machine, _machine)
_native = _directory / "_native" / f"{_tag}-{_sys.platform}-{_machine}"
if _native.is_dir():
    __path__.insert(0, str(_native))
elif not any((_directory / ("_kernels" + suffix)).is_file() for suffix in _EXTENSION_SUFFIXES):
    raise ImportError(
        "No compiled ST-Tensor/L1 module for "
        f"{_tag}/{_sys.platform}/{_machine}. This release supports CPython 3.12 "
        "on macOS Apple Silicon, Linux x86_64 and Windows x86_64. "
        "Download the matching complete v1.0.2 release or platform wheel. "
        "No source-code fallback or automatic download is used."
    )

from ._kernels import SolverConfig, adaptive_l1_trend_filter, adaptive_sttensor
from .api import RefinementResult, production_config, refine_core, refine_encoded_core

__all__ = [
    "SolverConfig",
    "adaptive_sttensor",
    "adaptive_l1_trend_filter",
    "RefinementResult",
    "production_config",
    "refine_core",
    "refine_encoded_core",
]

__version__ = "1.0.2"

