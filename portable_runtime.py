"""Frozen FP32 production with optional GPU owners and recorded patch backends.

Scientific model, preprocessing, solver and DN encoding remain in core/. The
tensor transfer and inference order follow the validated Mac production runtime.
All HDF handles belong to one spawned process. No training occurs here.
"""
from __future__ import annotations

from collections import deque
from contextlib import contextmanager
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import queue
import re
import sys
import time
import warnings
from datetime import datetime, timezone

from release_common import ROOT, load_json, load_frozen_model, load_calibration, sha256, write_json

LAYERS = ("Lai", "Fpar", "Lai_Uncertainty", "Fpar_Uncertainty", "Lai_QA", "Fpar_QA")
YEARS = list(range(2001, 2025))
PATCH = 40


def transfer_preserving_layout(tensor, device):
    """Preserve the production NumPy tensor's strides, including singleton axes."""
    import torch
    out = tensor.to(device=device, memory_format=torch.preserve_format)
    if out.stride() == tensor.stride() and out.storage_offset() == tensor.storage_offset():
        return out
    shape, strides, offset = tuple(tensor.shape), tuple(tensor.stride()), tensor.storage_offset()
    extent = offset + 1 + sum((n - 1) * s for n, s in zip(shape, strides))
    storage = torch.empty(extent, dtype=tensor.dtype, device=device)
    restored = torch.as_strided(storage, shape, strides, offset)
    select = tuple(slice(0, 1) if s == 0 and n > 1 else slice(None)
                   for n, s in zip(shape, strides))
    restored[select].copy_(out[select])
    return restored


def synchronize(device):
    import torch
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def select_owners(mode, cpu_workers, gpu_workers, gpu_device="auto", available=None):
    """GPU is optional for auto/mixed mode. Explicit mps/cuda requests are strict."""
    if min(cpu_workers, gpu_workers) < 0 or max(cpu_workers, gpu_workers) > 10:
        raise ValueError("Worker counts must be between zero and ten")
    if cpu_workers + gpu_workers == 0:
        raise ValueError("At least one worker must be requested")
    if available is None:
        import torch
        available = {"cuda": torch.cuda.is_available(), "mps": torch.backends.mps.is_available()}
    gpu = next((d for d in ("cuda", "mps") if available.get(d)), None)
    if gpu_device != "auto":
        gpu = gpu_device if available.get(gpu_device) else None
    reason = None
    if mode == "cpu":
        if cpu_workers == 0:
            raise ValueError("CPU mode needs at least one CPU worker")
        owners = ["cpu"] * cpu_workers
    elif mode in {"mps", "cuda"}:
        if not available.get(mode):
            raise RuntimeError(f"Explicitly requested {mode} is unavailable")
        if gpu_workers == 0:
            raise ValueError("GPU mode needs at least one GPU worker")
        owners = [mode] * gpu_workers
    elif mode in {"auto", "cpu_gpu"}:
        owners = ["cpu"] * cpu_workers
        if gpu and gpu_workers:
            owners += [gpu] * gpu_workers
        elif gpu_workers:
            reason = "No requested GPU is available; continuing on CPU"
        if not owners:
            owners = ["cpu"]
            reason = "No GPU is available; using one fallback CPU worker"
    else:
        raise ValueError(f"Unknown execution mode: {mode}")
    return owners, reason


def fit_cpu_budget(owners, logical_cpus=None):
    """Keep defaults runnable on smaller CPU-only machines as well as this Mac."""
    budget = max(1, ((os.cpu_count() or 1) if logical_cpus is None else logical_cpus) - 2)
    if len(owners) <= budget:
        return owners, None
    selected = []
    # Retain both lanes where at least two owner slots are available.
    backends = sorted(set(owners), key=lambda b: b == "cpu")
    for backend in backends[:budget]:
        selected.append(backend)
    remaining = list(owners)
    for backend in selected:
        remaining.remove(backend)
    selected += remaining[:budget - len(selected)]
    return selected, f"Owner count reduced to {budget} for the detected CPU thread budget"


class InputPreparer:
    def __init__(self, source, kind, settings, cache_mb=768):
        from qa_llm_v3.product import H5Cache
        from qa_llm_v3.production_io import TokenStripCache
        self.source, self.kind, self.settings = Path(source), kind, settings
        self.h5, self.raw, self.lc = H5Cache(40), TokenStripCache(5, memory_mb=cache_mb), {}
        self.tile = self.source.name if kind == "tile" else load_json(self.source / "manifest.json")["tile"]

    def close(self):
        self.h5.close()
        self.raw.clear()
        self.lc.clear()

    def prepare(self, year, row, col):
        import h5py
        import numpy as np
        from qg_dataset import read_token_arrays, read_lc_grid_patch
        from qa_llm_v3.data import build_input
        from qa_llm_v3.product import year_window, token_columns
        from qa_llm_v3.utils import parse_ints
        years, center = year_window(YEARS, year, 5)
        raw, cover, masks = [], [], []
        for y in years:
            if self.kind == "region":
                handle = self.h5.get(self.source / f"{y}.h5")
                token = read_token_arrays(handle, row, row + 40, col, col + 40)
                lc = read_lc_grid_patch(handle["LC_Type_3"], row, row + 40, col, col + 40, handle)
                veg = np.isin(lc, range(1, 9))
            else:
                br, bc, rr, cc = row // 600 + 1, col // 600 + 1, row % 600, col % 600
                path = self.source / "Module01_Intermediate" / "CompactStacks" / self.tile / str(y) / f"CompactQA_{self.tile}_{y}_R{br:02d}_C{bc:02d}.h5"
                token = token_columns(self.raw.get(path, self.h5.get(path), rr, rr + 40), cc, cc + 40)
                key = (y, br, bc)
                if key not in self.lc:
                    path = self.source / "Module00_Landcover" / "Masks" / f"LC_Type_3_vegetation_mask_{self.tile}_{y}.h5"
                    with h5py.File(path, "r") as h:
                        bounds = ((br - 1) * 600, br * 600, (bc - 1) * 600, bc * 600)
                        block = read_lc_grid_patch(h["LC_Type_3"], *bounds, h)
                        block_veg = read_lc_grid_patch(h["vegetation_mask"], *bounds, h) > 0
                    if len(self.lc) >= 32:
                        self.lc.pop(next(iter(self.lc)))
                    self.lc[key] = block, block_veg
                block, block_veg = self.lc[key]
                lc, veg = block[rr:rr + 40, cc:cc + 40], block_veg[rr:rr + 40, cc:cc + 40]
            raw.append(token)
            cover.append(lc)
            masks.append(veg)
        vegetation, inputs = masks[center], {}
        if vegetation.any():
            for variable in ("LAI", "FPAR"):
                args = self.settings[variable]
                features = [build_input(token, y, parse_ints(args["sensor_start_years"]), lc,
                                        parse_ints(args["lc_codes"]), active_variable=variable)
                            for token, y, lc in zip(raw, years, cover)]
                inputs[variable] = np.stack(features)[None]
        return dict(year=year, row=row, col=col, center=center, vegetation=vegetation, inputs=inputs)


class ModelRuntime:
    def __init__(self, backend, threads=1):
        import torch
        torch.set_num_threads(threads)
        if torch.get_num_interop_threads() != 1:
            torch.set_num_interop_threads(1)
        self.backend, self.device = backend, torch.device(backend)
        if backend == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        self.models = {}
        for variable in ("LAI", "FPAR"):
            model, solver, meta = load_frozen_model(variable, self.device)
            self.models[variable] = model, solver, meta, load_calibration(variable)
        self.settings = {v: self.models[v][2]["budget_calibration_args"] for v in self.models}

    def infer(self, prepared):
        import numpy as np
        import torch
        from qa_llm_v3.pipeline import run_window
        from qa_llm_v3.product import compose_layers
        from qa_llm_v3.uncertainty_core import center_landcover
        from qa_llm_v3.utils import parse_ints
        from solver_diagnostics import capture_core_diagnostics, empty_core_diagnostics
        out = np.full((46, 40, 40, 6), 65535, np.uint16)
        vegetation = prepared["vegetation"]
        stats = dict(backend=self.backend, solver_core_diagnostics={})
        if not vegetation.any():
            stats["solver_core_diagnostics"] = {v: empty_core_diagnostics(v, prepared) for v in self.models}
            return out, stats
        for vi, variable in enumerate(("LAI", "FPAR")):
            model, solver, meta, calibration = self.models[variable]
            settings = meta["budget_calibration_args"]
            x_cpu = torch.as_tensor(prepared["inputs"][variable], dtype=torch.float32)
            x = transfer_preserving_layout(x_cpu, self.device)
            mask = torch.as_tensor(vegetation[None], dtype=torch.bool, device=self.device)
            synchronize(self.device)
            with torch.inference_mode(), warnings.catch_warnings(record=True) as messages:
                warnings.simplefilter("always")
                result = run_window(model, x, prepared["center"], solver, core_size=8,
                                    training_solver=False, route_mode="learned_budget",
                                    route_max_selected_fraction=float(settings["policy_max_selected_fraction"]),
                                    budget_levels=meta["budget_policy"]["levels"], domain_mask=mask,
                                    structural_cell_blend=True)
                pft = center_landcover(x, torch.full((1,), prepared["center"], device=self.device),
                                      parse_ints(settings["lc_codes"]))[0]
                layers = compose_layers(result, x[:, prepared["center"]], mask, 8,
                                        variable=variable, product_calibration=calibration, pft=pft)[0]
            synchronize(self.device)
            out[..., [vi, vi + 2, vi + 4]] = layers
            stats["solver_core_diagnostics"][variable] = capture_core_diagnostics(variable, prepared, result)
            stats.setdefault("backend_warnings", []).extend(str(m.message) for m in messages)
            del x, x_cpu, mask, result, layers
        stats["backend_warnings"] = sorted(set(stats.get("backend_warnings", [])))
        return out, stats


def environment():
    import platform
    import torch, numpy, h5py, transformers, safetensors
    return dict(python=platform.python_version(), platform=platform.platform(), machine=platform.machine(),
                torch=torch.__version__, numpy=numpy.__version__, h5py=h5py.__version__,
                transformers=transformers.__version__, safetensors=safetensors.__version__,
                mps_available=torch.backends.mps.is_available(), cuda_available=torch.cuda.is_available(),
                mps_fallback=os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"),
                mps_fast_math=os.environ.get("PYTORCH_MPS_FAST_MATH", "unset"),
                mps_prefer_metal=os.environ.get("PYTORCH_MPS_PREFER_METAL", "unset"))


def patch_key(year, row, col):
    return f"{year}/{row}/{col}"


def patch_path(output, year, row, col):
    return Path(output) / "Patches" / str(year) / f"patch_{year}_r{row:04d}_c{col:04d}.h5"


def _read_patch(path, identity, *, tile):
    import h5py
    from solver_diagnostics import validate_core_diagnostics
    with h5py.File(path, "r") as h:
        if (h.attrs.get("complete") != 1 or h.attrs.get("tile") != tile
                or tuple(h["layers"].shape) != (46, 40, 40, 6)
                or str(h["layers"].dtype) != "uint16"
                or tuple(int(h.attrs[k]) for k in ("year", "row", "col")) != tuple(identity)):
            raise RuntimeError(f"Incompatible or incomplete cached patch: {path}")
        record = json.loads(h.attrs["record"])
        if tuple(record.get(k) for k in ("year", "row", "col")) != tuple(identity):
            raise RuntimeError(f"Cached patch record has another identity: {path}")
        if record.get("tile", tile) != tile:
            raise RuntimeError(f"Cached patch record has another tile: {path}")
        if record.get("backend") not in {"cpu", "mps", "cuda"}:
            raise RuntimeError(f"Cached patch lacks its actual backend: {path}")
        if set(record.get("solver_core_diagnostics", {})) != {"LAI", "FPAR"}:
            raise RuntimeError(f"Cached patch lacks separate actual solver diagnostics: {path}")
        for variable, diagnostic in record["solver_core_diagnostics"].items():
            validate_core_diagnostics(diagnostic)
            if diagnostic["Variable"] != variable:
                raise RuntimeError("Cached diagnostic variable identity differs")
        if hashlib.sha256(h["layers"][...].tobytes()).hexdigest() != record.get("layer_sha256"):
            raise RuntimeError(f"Cached patch DN checksum mismatch: {path}")
        return record


def _worker(config, backend, tasks, results):
    preparer = None
    try:
        runtime = ModelRuntime(backend, config["threads"])
        preparer = InputPreparer(config["source"], config["kind"], runtime.settings, config["cache_mb"])
        results.put(("ready", os.getpid(), backend, None))
        while True:
            job = tasks.get()
            if job is None:
                break
            record = None
            for year, row, col in job:
                prepared = preparer.prepare(year, row, col)
                values, stats = runtime.infer(prepared)
                record = dict(tile=config["tile"], year=year, row=row, col=col,
                              layer_sha256=hashlib.sha256(values.tobytes()).hexdigest(), **stats)
                path = patch_path(config["output"], year, row, col)
                if path.exists():
                    raise RuntimeError(f"Refusing duplicate publication: {path}")
                path.parent.mkdir(parents=True, exist_ok=True)
                temp = path.with_suffix(".tmp.h5")
                import h5py
                with h5py.File(temp, "w") as h:
                    h.create_dataset("layers", data=values, chunks=(1, 40, 40, 6), compression="lzf")
                    h.attrs.update(complete=1, year=year, row=row, col=col,
                                   tile=config["tile"],
                                   dimension_order="time,row,col,layer", layer_names=json.dumps(LAYERS),
                                   record=json.dumps(record, allow_nan=False))
                temp.replace(path)
                results.put(("patch", os.getpid(), backend, record))
            results.put(("ready", os.getpid(), backend, None))
    except BaseException as exc:
        import traceback
        results.put(("error", os.getpid(), backend, f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"))
    finally:
        if preparer is not None:
            preparer.close()


@contextmanager
def output_lock(output):
    path = output / ".production.lock"
    try:
        with path.open("x") as f:
            f.write(str(os.getpid()))
    except FileExistsError as exc:
        raise RuntimeError(f"Output is locked: {path}. Check the recorded process before removing a stale lock.") from exc
    try:
        yield
    finally:
        path.unlink(missing_ok=True)


def _atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(".tmp.json")
    write_json(tmp, value)
    tmp.replace(path)


def _output_geometry(config):
    return {k: config[k] for k in (
        "height", "width", "row_offset", "col_offset",
        "origin_row_1based", "origin_col_1based",
    )}


def _run_scope(config):
    """Describe cache structure without identifying code, weights or inputs."""
    return dict(
        schema="qa-llm-gf-portable-production-scope-v2",
        tile=config["tile"], kind=config["kind"],
        years=sorted(config["years"]),
        coords=sorted([list(origin) for origin in config["coords"]]),
        output_geometry=_output_geometry(config),
        dtype="float32", native_patch=40, core_size=8, window_years=5,
        input_transfer="preserve_numpy_strides_and_offset",
        uncertainty_application_count=1,
    )


def _check_resume_scope(saved, config):
    """Ignore obsolete provenance while retaining required output identities."""
    expected = _run_scope(config)
    if not isinstance(saved, dict):
        raise RuntimeError("Existing run manifest does not describe an output scope")
    actual = {key: saved.get(key) for key in expected if key != "schema"}
    if isinstance(actual["years"], (list, tuple)):
        actual["years"] = sorted(actual["years"])
    if isinstance(actual["coords"], (list, tuple)):
        actual["coords"] = sorted([list(origin) for origin in actual["coords"]])
    if actual != {key: value for key, value in expected.items() if key != "schema"}:
        raise RuntimeError("Existing output has a different tile, target scope or output structure")


def _check_annual_file(path, config, year, doy):
    """Check geometry and layer schema independently of legacy provenance."""
    import h5py
    with h5py.File(path, "r") as h:
        attributes = dict(
            complete=1, tile=config["tile"], year=year, doy=doy,
            rows=config["height"], cols=config["width"],
            origin_row_1based=config["origin_row_1based"],
            origin_col_1based=config["origin_col_1based"],
            uncertainty_application_count=1,
        )
        if any(h.attrs.get(key) != value for key, value in attributes.items()):
            raise RuntimeError(f"Annual product identity or geometry differs: {path}")
        if any(name not in h or tuple(h[name].shape) != (config["width"], config["height"])
               or str(h[name].dtype) != "uint16" for name in LAYERS):
            raise RuntimeError(f"Annual product layer schema differs: {path}")


def assemble_year(config, year, coords, records):
    """Read every patch once and retain the six production DN layers unchanged."""
    import h5py
    import numpy as np
    output = Path(config["output"])
    target = output / "Products" / str(year)
    seal = target / "annual_complete.json"
    if seal.is_file():
        metadata = load_json(seal)
        expected_files = {f"QA_LLM_GF_{config['tile']}_{year}{doy:03d}.h5" for doy in range(1, 362, 8)}
        if (metadata.get("year") != year
                or metadata.get("patches") != len(coords) or metadata.get("six_layer_dn_verified") is not True
                or set(metadata.get("files", {})) != expected_files):
            raise RuntimeError(f"Incompatible existing annual product: {target}")
        if metadata.get("tile", config["tile"]) != config["tile"]:
            raise RuntimeError(f"Annual product belongs to another tile: {target}")
        if ("output_geometry" in metadata
                and metadata["output_geometry"] != _output_geometry(config)):
            raise RuntimeError(f"Annual product has another output geometry: {target}")
        if ("patch_origins" in metadata
                and sorted(metadata["patch_origins"]) != sorted([list(origin) for origin in coords])):
            raise RuntimeError(f"Annual product has another patch scope: {target}")
        for name, expected in metadata["files"].items():
            if sha256(target / name) != expected:
                raise RuntimeError(f"Annual product changed: {target / name}")
            doy = int(name.rsplit("_", 1)[1][4:7])
            _check_annual_file(target / name, config, year, doy)
        return
    if target.exists():
        raise RuntimeError(f"Unsealed annual directory exists: {target}")
    temp = output / "Products" / f".{year}.assembling"
    if temp.exists():
        import shutil
        shutil.rmtree(temp)  # Only this engine's incomplete assembly scratch.
    temp.mkdir(parents=True)
    height, width = config["height"], config["width"]
    handles = []
    try:
        for doy in range(1, 362, 8):
            path = temp / f"QA_LLM_GF_{config['tile']}_{year}{doy:03d}.h5"
            h = h5py.File(path, "w")
            handles.append(h)
            h.attrs.update(complete=0, tile=config["tile"], year=year, doy=doy,
                           rows=height, cols=width,
                           origin_row_1based=config["origin_row_1based"], origin_col_1based=config["origin_col_1based"],
                           matlab_dimension_order="row,col", python_h5py_dimension_order="col,row",
                           uncertainty_application_count=1, physical_scale_factor=.001,
                           physical_fill_value=65535, uncertainty_quantity="calibrated sigma")
            for name in LAYERS:
                ds = h.create_dataset(name, (width, height), dtype="uint16", fillvalue=65535,
                                      chunks=(40, 40), compression="gzip", compression_opts=1, shuffle=True)
                ds.attrs.update(fill_value=65535, matlab_dimension_order="row,col", python_h5py_dimension_order="col,row")
                if name.endswith("_QA"):
                    ds.attrs["bit_layout"] = "0:2 source;3:4 usable;5:6 strict;7:8 saturated;9:15 confidence"
                else:
                    ds.attrs["scale_factor"] = .001
        for row, col in coords:
            path = patch_path(output, year, row, col)
            record = _read_patch(path, (year, row, col), tile=config["tile"])
            with h5py.File(path, "r") as h:
                values = h["layers"][...]
            if hashlib.sha256(values.tobytes()).hexdigest() != record["layer_sha256"]:
                raise RuntimeError(f"Patch data checksum mismatch: {path}")
            rr, cc = row - config["row_offset"], col - config["col_offset"]
            if (rr < 0 or cc < 0 or rr + PATCH > height or cc + PATCH > width
                    or rr % PATCH or cc % PATCH):
                raise RuntimeError(f"Patch lies outside the configured output geometry: {path}")
            for ti, h in enumerate(handles):
                for vi, name in enumerate(LAYERS):
                    h[name][cc:cc + 40, rr:rr + 40] = values[ti, ..., vi].T
        for h in handles:
            h.attrs["complete"] = 1
            h.flush()
    finally:
        for h in handles:
            h.close()
    files = {p.name: sha256(p) for p in sorted(temp.glob("*.h5"))}
    if len(files) != 46:
        raise RuntimeError("An annual product requires all 46 dates")
    # Compare saved H5 values against every source patch, not only file counts.
    annual = [h5py.File(p, "r") for p in sorted(temp.glob("*.h5"))]
    try:
        for row, col in coords:
            with h5py.File(patch_path(output, year, row, col), "r") as h:
                values = h["layers"][...]
            rr, cc = row - config["row_offset"], col - config["col_offset"]
            for ti, h in enumerate(annual):
                for vi, name in enumerate(LAYERS):
                    if not np.array_equal(h[name][cc:cc + 40, rr:rr + 40].T, values[ti, ..., vi]):
                        raise RuntimeError("Assembled product DN differs from its patch")
    finally:
        for h in annual:
            h.close()
    write_json(temp / "annual_complete.json", dict(
        tile=config["tile"], year=year, files=files, patches=len(coords),
        output_geometry=_output_geometry(config),
        patch_origins=sorted([list(origin) for origin in coords]),
        six_layer_dn_verified=True,
    ))
    temp.replace(target)


def produce(config):
    """One parent publishes durable backend assignments before dispatching work."""
    from solver_diagnostics import write_solver_statistics
    owners, fallback = select_owners(config["device"], config["cpu_workers"], config["gpu_workers"], config["gpu_device"])
    owners, cpu_limit = fit_cpu_budget(owners)
    fallback = "; ".join(v for v in (fallback, cpu_limit) if v) or None
    output = Path(config["output"])
    jobs = [(y, r, c) for y in config["years"] for r, c in config["coords"]]
    if len(jobs) != len(set(jobs)):
        raise ValueError("Each target-year/patch identity must occur exactly once")
    expected = set(jobs)
    if config["threads"] < 1 or (len(owners) * config["threads"] > max(1, (os.cpu_count() or 1) - 2) and len(owners) > 1):
        raise ValueError("Workers times threads exceed the CPU budget with two reserved cores")
    contract = _run_scope(config)
    output.mkdir(parents=True, exist_ok=True)
    with output_lock(output):
        manifest_path = output / "run_manifest.json"
        if manifest_path.exists():
            _check_resume_scope(load_json(manifest_path), config)
        if not manifest_path.exists():
            write_json(manifest_path, contract)
        replay = load_json(config["replay_map"]) if config.get("replay_map") else {}
        if "patch_backends" in replay:
            replay = replay["patch_backends"]
        assignments = {}
        claims_path = output / "backend_assignments.jsonl"
        if claims_path.exists():
            for line in claims_path.read_text().splitlines():
                a = json.loads(line)
                for identity in a["patches"]:
                    key = patch_key(*identity)
                    if key in assignments and assignments[key] != a["backend"]:
                        raise RuntimeError("Conflicting persisted patch backend")
                    assignments[key] = a["backend"]
        if replay:
            for identity in jobs:
                key = patch_key(*identity)
                if key not in replay:
                    raise ValueError(f"Replay map lacks {key}")
                if replay[key] not in {"cpu", "mps", "cuda"}:
                    raise ValueError("Replay contains an unknown backend")
                if key in assignments and assignments[key] != replay[key]:
                    raise RuntimeError("Replay conflicts with a persisted backend assignment")
                assignments[key] = replay[key]
        records, pending = [], deque()
        for identity in jobs:
            path = patch_path(output, *identity)
            if path.exists():
                record = _read_patch(path, identity, tile=config["tile"])
                if (patch_key(*identity) in assignments
                        and assignments[patch_key(*identity)] != record["backend"]):
                    raise RuntimeError("Cached patch has another execution backend")
                assignments[patch_key(*identity)] = record["backend"]
                records.append(record)
            else:
                pending.append(identity)
        # Persisted/replayed GPU assignments are never silently changed to CPU.
        required = {assignments[patch_key(*x)] for x in pending if patch_key(*x) in assignments}
        if not required <= set(owners):
            raise RuntimeError(f"Resume/replay requires owners for: {sorted(required - set(owners))}")
        remaining_owners = sorted(owners, key=lambda b: b == "cpu")
        active_owners = sorted(required)
        for backend in active_owners:
            remaining_owners.remove(backend)
        active_owners += remaining_owners[:max(0, min(len(owners), len(pending)) - len(active_owners))]
        resource_info = dict(requested_device=config["device"], effective_owners=active_owners, fallback=fallback,
                             requested_cpu_workers=config["cpu_workers"], requested_gpu_workers=config["gpu_workers"],
                             physical_gpu_count=int(any(b != "cpu" for b in active_owners)), threads=config["threads"])
        print(json.dumps(dict(event="resources", **resource_info)), flush=True)
        write_json(output / "execution_resources.json", resource_info)
        ctx = mp.get_context("spawn")
        results = ctx.Queue()
        processes, channels, busy, inflight = {}, {}, set(), {}
        completed = {(r["year"], r["row"], r["col"]) for r in records}
        resumed_count = len(records)
        start = time.monotonic()
        claims = claims_path.open("a", encoding="utf-8")
        group_size = max(1, min(15, (len(pending) + max(1, len(active_owners)) - 1) // max(1, len(active_owners))))
        try:
            for backend in active_owners:
                q = ctx.Queue()
                p = ctx.Process(target=_worker, args=(config, backend, q, results))
                p.start()
                processes[p.pid], channels[p.pid] = p, q
                # Load models one process at a time to bound startup memory peaks.
                while True:
                    try:
                        event, pid, lane, payload = results.get(timeout=2)
                    except queue.Empty:
                        if not p.is_alive():
                            raise RuntimeError("Worker exited during model initialization")
                        continue
                    if event == "error":
                        raise RuntimeError(payload)
                    if event != "ready" or pid != p.pid:
                        raise RuntimeError("Unexpected startup event")
                    break
            available = deque(processes)
            while pending or busy:
                for _ in range(len(available)):
                    pid = available.popleft()
                    lane = active_owners[list(processes).index(pid)]
                    group = []
                    # Match any durable backend; fresh work goes to the next free owner.
                    for _ in range(len(pending)):
                        identity = pending.popleft()
                        backend = assignments.get(patch_key(*identity))
                        if backend in (None, lane) and len(group) < group_size:
                            group.append(identity)
                        else:
                            pending.append(identity)
                    if group:
                        claims.write(json.dumps(dict(backend=lane, patches=group)) + "\n")
                        claims.flush()
                        os.fsync(claims.fileno())
                        for identity in group:
                            assignments[patch_key(*identity)] = lane
                        channels[pid].put(group)
                        busy.add(pid)
                        inflight[pid] = set(group)
                    else:
                        available.append(pid)
                if not busy:
                    if pending:
                        raise RuntimeError("No active owner can execute the persisted/replayed backend")
                    break
                try:
                    event, pid, lane, payload = results.get(timeout=2)
                except queue.Empty:
                    if any(not processes[pid].is_alive() for pid in busy):
                        raise RuntimeError("A worker exited unexpectedly; completed patches remain resumable")
                    continue
                if event == "error":
                    raise RuntimeError(payload)
                if pid not in busy or lane != active_owners[list(processes).index(pid)]:
                    raise RuntimeError("Worker event has an unknown owner or backend")
                if event == "patch":
                    identity = tuple(payload[k] for k in ("year", "row", "col"))
                    if identity not in inflight[pid] or identity in completed or payload.get("backend") != lane:
                        raise RuntimeError("Duplicate, unassigned or wrong-backend patch event")
                    saved = _read_patch(patch_path(output, *identity), identity, tile=config["tile"])
                    if saved != payload:
                        raise RuntimeError("Worker event differs from the saved patch record")
                    inflight[pid].remove(identity)
                    completed.add(identity)
                    records.append(payload)
                    progress = dict(event="patch_completed", year=payload["year"], completed=len(records),
                                    total=len(jobs), backend=lane, elapsed_seconds=time.monotonic() - start,
                                    utc=datetime.now(timezone.utc).isoformat(), resumed=resumed_count)
                    rate = (len(records) - resumed_count) / progress["elapsed_seconds"]
                    progress.update(patches_per_second=rate, estimated_remaining_seconds=(len(jobs) - len(records)) / rate)
                    _atomic_json(output / "status.json", progress)
                    print(json.dumps(progress), flush=True)
                elif event == "ready":
                    if inflight[pid]:
                        raise RuntimeError("Worker returned before completing its assigned patch identities")
                    busy.discard(pid)
                    del inflight[pid]
                    available.append(pid)
                else:
                    raise RuntimeError("Unknown worker event")
            for q in channels.values():
                q.put(None)
            for p in processes.values():
                p.join(timeout=30)
                if p.is_alive() or p.exitcode != 0:
                    raise RuntimeError("A production worker did not exit cleanly")
        finally:
            claims.close()
            for p in processes.values():
                if p.is_alive():
                    p.terminate()
            for p in processes.values():
                p.join(timeout=5)
            for q in channels.values():
                q.close()
            results.close()
        if len(records) != len(jobs) or completed != expected:
            raise RuntimeError("The requested patch scope is incomplete")
        write_json(output / "execution_manifest.json", dict(
                   tile=config["tile"],
                   patch_backends={patch_key(a["year"], a["row"], a["col"]): a["backend"] for a in records},
                   layer_sha256={patch_key(a["year"], a["row"], a["col"]): a["layer_sha256"] for a in records}))
        write_solver_statistics(records, output / "Products", config["years"], tile=config["tile"])
        for year in config["years"]:
            print(json.dumps(dict(event="assembling", year=year)), flush=True)
            assemble_year(config, year, config["coords"], records)
        summary = dict(ok=True, scope_complete=True, tile=config["tile"], requested_years=config["years"],
                       patch_count=len(records), final_file_count=46 * len(config["years"]),
                       six_layer_assembly_dn_verified=True,
                       elapsed_seconds=time.monotonic() - start, finished_utc=datetime.now(timezone.utc).isoformat(),
                       resumed_patch_count=resumed_count, **resource_info)
        write_json(output / "product_summary.json", summary)
        _atomic_json(output / "status.json", dict(event="completed", **summary))
        print(json.dumps(dict(event="completed", output=str(output), **summary)), flush=True)
        return summary


def source_identity(source, kind):
    """Return input structure only; do not hash inputs or record file mtimes."""
    source = Path(source)
    if kind == "region":
        manifest = load_json(source / "manifest.json")
        return {key: manifest[key] for key in (
            "tile", "years", "height", "width", "native_patch_size", "core_size",
            "origin_row_1based", "origin_col_1based",
        )}
    files = [source / "Module01_Intermediate" / "PythonDataset" / "token_index_python.csv"]
    files += sorted((source / "Module01_Intermediate" / "CompactStacks" / source.name).glob("*/*.h5"))
    files += sorted((source / "Module00_Landcover" / "Masks").glob(f"*{source.name}_*.h5"))
    if len(files) < 409 or not files[0].is_file():
        raise ValueError("Native tile requires its full 24-year compact stacks, LC masks and token index")
    return dict(tile=source.name, years=YEARS, height=2400, width=2400,
                native_patch_size=40, core_size=8)
