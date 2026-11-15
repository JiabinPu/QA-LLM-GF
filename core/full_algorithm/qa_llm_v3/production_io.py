"""Bounded raw-byte slab caching for production, independent of model geometry."""
from __future__ import annotations

import ctypes
import os
import time
from collections import OrderedDict
from pathlib import Path

import h5py
import numpy as np

from qg_dataset import read_token_arrays


def available_memory_bytes() -> int:
    try:
        if os.name == "nt":
            class MemoryStatus(ctypes.Structure):
                _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
                    (name, ctypes.c_ulonglong) for name in
                    ("total", "available", "page_total", "page_available",
                     "virtual_total", "virtual_available", "extended")
                ]
            status = MemoryStatus()
            status.length = ctypes.sizeof(status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.available)
        else:
            return int(os.sysconf("SC_AVPHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, OSError, ValueError):
        pass
    return 0


def cache_budget_mb(requested: int, smoke: bool = False, workers: int = 1) -> int:
    """0: auto; -1: legacy one-strip cache; positive: requested memory target."""
    if requested < -1:
        raise ValueError("input-cache-mb must be -1, 0, or positive")
    if requested == -1 or (requested == 0 and smoke):
        return 0
    free = available_memory_bytes()
    safe_mb = max(0, int(free // (4 * max(1, workers) * 1024**2))) if free else 512 // max(1, workers)
    target = min(3072, safe_mb) if requested == 0 else min(requested, safe_mb)
    return target if target >= 256 else 0


class TokenStripCache:
    """Keep a raw-byte slab per window year; return exactly the requested rows.

    Larger reads avoid repeatedly decompressing 600x600 HDF5 chunks. No
    normalized feature cache or on-disk duplicate is created. The memory target
    is divided across window years; the minimum unit remains one legacy strip.
    """
    def __init__(self, capacity: int, memory_mb: int = 0) -> None:
        self.capacity = max(1, int(capacity))
        self.memory_bytes = max(0, int(memory_mb)) * 1024**2
        self.items = OrderedDict()
        self.layouts = {}
        self.clear()

    def _layout(self, path: Path, handle: h5py.File) -> tuple[int, int, int]:
        key = str(path)
        if key not in self.layouts:
            ref = handle["/obs/LAI_DN"]
            sensors, steps, width, height = ref.shape
            names = ("/obs/LAI_DN", "/obs/FPAR_DN", "/score/LAI", "/score/FPAR",
                     "/flag/LAI", "/flag/FPAR", "/uncert/LAI_STD_DN",
                     "/uncert/FPAR_STD_DN", "/qa/SCF_QC")
            bytes_per_cell = sum(handle[name].dtype.itemsize if name in handle else 1 for name in names)
            self.layouts[key] = (int(height), int(width), int(sensors * steps * width * bytes_per_cell))
        return self.layouts[key]

    def get(self, path: Path, handle: h5py.File, r0: int, r1: int) -> dict:
        height, width, bytes_per_row = self._layout(path, handle)
        if not 0 <= r0 < r1 <= height:
            raise ValueError(f"Invalid raw cache row bounds {r0}:{r1}, height={height}")
        strip_rows = r1 - r0
        affordable_rows = self.memory_bytes // (self.capacity * bytes_per_row) if self.memory_bytes else 0
        slab_rows = min(height, max(strip_rows, (affordable_rows // strip_rows) * strip_rows))
        slab_start = (r0 // slab_rows) * slab_rows
        slab_end = min(height, slab_start + slab_rows)
        if r1 > slab_end:
            slab_start, slab_end = r0, r1
        key = (str(path), slab_start, slab_end)
        self.last_slab_rows = slab_end - slab_start
        if key in self.items:
            self.hits += 1
            self.items.move_to_end(key)
            token = self.items[key]
        else:
            # Release the preceding slab before allocating its replacement.
            while len(self.items) >= self.capacity:
                self.items.popitem(last=False)
            start = time.perf_counter()
            token = read_token_arrays(handle, slab_start, slab_end, 0, width)
            self.read_seconds += time.perf_counter() - start
            self.loads += 1
            self.loaded_bytes += sum(v.nbytes for v in token.values() if isinstance(v, np.ndarray))
            self.items[key] = token
        return {
            name: value[r0-slab_start:r1-slab_start]
            if isinstance(value, np.ndarray) and value.ndim == 4 else value
            for name, value in token.items()
        }

    def full_block_bytes(self, path: Path, handle: h5py.File) -> int:
        height, _width, bytes_per_row = self._layout(path, handle)
        return height * bytes_per_row

    def clear(self) -> None:
        self.items.clear()
        self.layouts.clear()
        self.hits = self.loads = self.loaded_bytes = self.last_slab_rows = 0
        self.read_seconds = 0.0
