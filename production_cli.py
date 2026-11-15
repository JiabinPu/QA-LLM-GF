"""Common, side-effect-free preflight for portable region and native-tile jobs."""
import argparse
from pathlib import Path
import re

from release_common import ROOT, output_directory, load_json
from release_contracts import production_contract, model_contract


def parser(kind):
    p = argparse.ArgumentParser(description="Frozen QA-LLM-GF production; GPU is optional")
    p.add_argument("--device", choices=("auto", "cpu", "mps", "cuda", "cpu_gpu"), default="auto")
    p.add_argument("--gpu-device", choices=("auto", "mps", "cuda"), default="auto")
    p.add_argument("--cpu-workers", type=int, default=5)
    p.add_argument("--gpu-workers", type=int, default=5)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--cache-mb", type=int, default=768)
    p.add_argument("--years", default="2001:2024")
    p.add_argument("--output", type=Path)
    p.add_argument("--replay-map", type=Path, help="Previously recorded execution_manifest.json; preserve every patch backend")
    p.add_argument("--smoke", action="store_true", help="2013, first native 40x40 patch only")
    p.add_argument("--check-only", action="store_true", help="Validate inputs/assets/resources without inference or output writes")
    if kind == "tile":
        p.add_argument("--data-root", type=Path, required=True, help="One tile directory containing Module00/Module01 inputs")
    return p


def configure(a, kind):
    from qa_llm_v3.utils import parse_ints
    from portable_runtime import source_identity, select_owners, fit_cpu_budget
    if a.threads != 1:
        raise ValueError("This production profile uses one numerical thread per worker")
    if not 0 <= a.cache_mb <= 768:
        raise ValueError("cache-mb must be between zero and 768")
    years = [2013] if a.smoke else parse_ints(a.years)
    if not years or not set(years) <= set(range(2001, 2025)):
        raise ValueError("Target years must be within 2001–2024")
    if kind == "region":
        source = ROOT / "data" / "production_region"
        manifest = production_contract(source)
        tile = manifest["tile"]
        origin_row, origin_col = manifest["origin_row_1based"], manifest["origin_col_1based"]
        size = 40 if a.smoke else 200
    else:
        source = a.data_root.expanduser().resolve()
        tile = source.name
        if not re.fullmatch(r"h\d{2}v\d{2}", tile):
            raise ValueError("data-root must identify a single hNNvNN tile directory")
        origin_row = origin_col = 1
        size = 40 if a.smoke else 2400
        import pandas as pd
        index = pd.read_csv(source / "Module01_Intermediate" / "PythonDataset" / "token_index_python.csv")
        if (set(index.Tile) != {tile} or set(index.PatchSize) != {40} or set(index.Stride) != {40}
                or set(index.Year) != set(range(2001, 2025))):
            raise ValueError("Expected one native tile, 40x40 patch/stride, complete 2001–2024 context")
        expected = {(r + 1, c + 1) for r in range(0, 2400, 40) for c in range(0, 2400, 40)}
        for year, group in index.groupby("Year"):
            pairs = list(zip(group.GlobalRowStart.astype(int), group.GlobalColStart.astype(int)))
            if len(pairs) != 3600 or set(pairs) != expected:
                raise ValueError(f"Incomplete or duplicate native grid in {year}")
    for variable in ("LAI", "FPAR"):
        model_contract(variable)
    identity = source_identity(source, kind)
    label = "production_smoke" if a.smoke else "production_200x200"
    if kind == "tile":
        label = f"production_{tile}" + ("_smoke" if a.smoke else "")
    default = f"{label}_{a.device}"
    output = output_directory(a.output or ROOT / "runs" / default)
    if output == source or source in output.parents or output in source.parents:
        raise ValueError("Output must be separate from the input data")
    owners, reason = select_owners(a.device, a.cpu_workers, a.gpu_workers, a.gpu_device)
    owners, limit = fit_cpu_budget(owners)
    reason = "; ".join(v for v in (reason, limit) if v) or None
    config = dict(kind=kind, source=str(source), source_identity=identity, tile=tile, years=years,
                  coords=[(r, c) for r in range(0, size, 40) for c in range(0, size, 40)],
                  height=size, width=size, row_offset=0, col_offset=0,
                  origin_row_1based=origin_row, origin_col_1based=origin_col, output=str(output),
                  device=a.device, gpu_device=a.gpu_device, cpu_workers=a.cpu_workers,
                  gpu_workers=a.gpu_workers, threads=a.threads, cache_mb=a.cache_mb,
                  replay_map=str(a.replay_map.expanduser().resolve()) if a.replay_map else None)
    if a.check_only:
        import json
        print(json.dumps(dict(check_only=True, tile=tile, target_years=years, height=size,
                              width=size, native_patch_count=len(config["coords"]) * len(years),
                              requested_device=a.device, available_owners=owners, fallback=reason,
                              output=str(output), inference_started=False), indent=2))
    return config
