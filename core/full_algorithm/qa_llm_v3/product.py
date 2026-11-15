from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from collections import OrderedDict
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

from qg_dataset import read_lc_grid_patch, read_token_arrays

from .data import build_input
from .model import ControllerConfig, QALLMAdaptiveController
from .pipeline import hierarchical_uncertainty_components, run_window
from .production_io import TokenStripCache, cache_budget_mb
from .product_calibration import ProductCalibration
from .uncertainty_core import center_landcover
from .quality import (
    quality_protocol_metadata,
    validate_checkpoint_protocol,
    validate_checkpoint_tile,
)
from .solver import SolverConfig
from .utils import parse_ints, resolve_device
from .tile_policy import PRODUCTION_TILES


LAYERS = ("LAI", "FPAR", "uncertainty_LAI", "uncertainty_FPAR", "QA_LAI", "QA_FPAR")
DATASETS = ("/Lai", "/Fpar", "/Lai_Uncertainty", "/Fpar_Uncertainty", "/Lai_QA", "/Fpar_QA")
FILL = np.uint16(65535)
SCALE = 0.001


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def production_config_label(args: argparse.Namespace) -> str:
    payload = {
        "schema": "qa-llm-gf-independent-full-product",
        "variable": args.variable,
        "calibration_sha256": args.calibration_sha256,
        "calibration_report_sha256": args.calibration_report_sha256,
        "input_inventory_sha256": args.input_inventory_sha256,
        "window_years": int(args.window_years),
        "sensor_start_years": parse_ints(args.sensor_start_years),
        "core_size": int(args.core_size),
        "route_solver": bool(args.route_solver),
        "route_mode": str(args.route_mode),
        "route_min_fraction": float(args.route_min_fraction),
        "route_max_fraction": float(args.route_max_fraction),
        "route_threshold_lai": float(args.route_threshold_lai),
        "route_threshold_fpar": float(args.route_threshold_fpar),
        "route_max_selected_fraction": float(args.route_max_selected_fraction),
        "budget_levels": parse_ints(args.budget_levels),
        "uncertainty_scale_lai": float(args.uncertainty_scale_lai),
        "uncertainty_scale_fpar": float(args.uncertainty_scale_fpar),
        "lc_codes": parse_ints(args.lc_codes),
        "tile_rows": int(args.tile_rows),
        "tile_cols": int(args.tile_cols),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate the frozen QA-LLM-GF product")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--variable", choices=("LAI", "FPAR"), required=True)
    p.add_argument("--calibration-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--years", default="2001:2024")
    p.add_argument("--block-rows", default="1:4")
    p.add_argument("--block-cols", default="1:4")
    p.add_argument("--window-years", type=int, default=5)
    p.add_argument("--sensor-start-years", default="2001,2002,2012")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--core-size", type=int, default=8)
    p.add_argument("--route-solver", type=int, default=1)
    p.add_argument(
        "--route-mode", choices=["learned_budget", "threshold", "fraction"],
        default="learned_budget"
    )
    p.add_argument("--route-min-fraction", type=float, default=0.20)
    p.add_argument("--route-max-fraction", type=float, default=0.60)
    p.add_argument("--route-threshold-lai", type=float, default=0.78)
    p.add_argument("--route-threshold-fpar", type=float, default=0.78)
    p.add_argument("--route-max-selected-fraction", type=float, default=0.80)
    p.add_argument("--budget-levels", default="0,3,5,8,13,24")
    p.add_argument("--uncertainty-scale-lai", type=float, default=1.0)
    p.add_argument("--uncertainty-scale-fpar", type=float, default=1.0)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--torch-threads", type=int, default=0)
    p.add_argument("--lc-mask-dir", type=Path, required=True)
    p.add_argument("--lc-codes", default="1:8")
    p.add_argument("--tile-rows", type=int, default=2400)
    p.add_argument("--tile-cols", type=int, default=2400)
    p.add_argument("--compression", choices=["lzf", "gzip", "none"], default="gzip")
    p.add_argument("--compression-level", type=int, default=1)
    p.add_argument("--overwrite", type=int, default=0)
    p.add_argument("--max-patches-per-block", type=int, default=0)
    p.add_argument("--print-every", type=int, default=10)
    p.add_argument("--summary-file", default="product_blocks.csv")
    p.add_argument("--strip-cache", type=int, default=1)
    p.add_argument("--share-window-frontends", type=int, choices=(0,1), default=0,
                   help="Reuse identical input-window frontends across center years")
    p.add_argument("--input-cache-mb", type=int, default=0,
                   help="Raw slab cache target: 0=auto, -1=legacy strips, positive=MiB")
    p.add_argument("--annual-feature-cache", type=int, choices=(0,1), default=0)
    p.add_argument("--year-group-size", type=int, choices=(1,2,3), default=3)
    p.add_argument("--fold-constant-heads", type=int, choices=(0,1), default=0)
    p.add_argument("--production-fastpath", type=int, choices=(0,1), default=0,
                   help="Inference-only lean outputs and zero-budget solver bypass")
    return p.parse_args()


def load_model(path: Path, device: torch.device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    validate_checkpoint_protocol(checkpoint, budget=True)
    model = QALLMAdaptiveController(ControllerConfig(**checkpoint["model_config"])).to(device)
    result = model.load_state_dict(
        checkpoint["model_state"], strict=not bool(checkpoint.get("trainable_state_only", False))
    )
    if result.unexpected_keys:
        raise RuntimeError(f"Unexpected checkpoint keys: {result.unexpected_keys}")
    parameter_map = dict(model.named_parameters())
    missing_trainable = [
        key for key in result.missing_keys
        if key in parameter_map and parameter_map[key].requires_grad
    ]
    if missing_trainable:
        raise RuntimeError(f"Checkpoint is missing trainable keys: {missing_trainable}")
    model.eval()
    return model, SolverConfig(**checkpoint["solver_config"]), checkpoint


def prepare_product_contract(args, checkpoint):
    """Shared by inference and the merger; avoid divergent reuse defaults."""
    validate_checkpoint_protocol(checkpoint, budget=True)
    validate_checkpoint_tile(checkpoint, args.root.parent / 'MultiTile')
    if checkpoint['model_config'].get('active_variable') != args.variable:
        raise ValueError('Product variable and budget model do not match')
    calibrated = checkpoint['budget_calibration_args']
    args.core_size = int(calibrated['core_size'])
    args.window_years = int(calibrated['window_years'])
    args.sensor_start_years = calibrated['sensor_start_years']
    args.lc_codes = calibrated['lc_codes']
    if parse_ints(args.lc_codes) != list(range(1, 9)):
        raise ValueError('Product calibration requires LC_Type_3 codes 1 through 8')
    args.route_max_selected_fraction = float(calibrated['policy_max_selected_fraction'])
    levels = checkpoint['budget_policy']['levels']
    if not levels or levels[0] != 0 or levels[-1] != checkpoint['solver_config']['max_iterations']:
        raise ValueError('Budget levels and solver iteration limit disagree')
    args.budget_levels = ','.join(map(str, levels))
    calibration = ProductCalibration(args.calibration_dir, args.variable, args.checkpoint)
    args.calibration_sha256 = calibration.table_hash
    args.calibration_report_sha256 = calibration.report_hash
    index = args.root/'Module01_Intermediate'/'PythonDataset'/'token_index_python.csv'
    inventory = [('index_sha256', file_sha256(index))]
    # Stat inventory avoids rereading terabytes. Inputs must remain immutable during a run.
    for folder in (args.root/'Module01_Intermediate'/'CompactStacks', args.lc_mask_dir):
        files = sorted(folder.rglob('*.h5'))
        if not files:
            raise FileNotFoundError(f'No intermediate H5 files in {folder}')
        inventory.extend((str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns) for p in files)
    args.input_inventory_sha256 = hashlib.sha256(json.dumps(inventory).encode()).hexdigest()
    return calibration


def resolve_source(root: Path, row: pd.Series) -> Path:
    source = Path(str(row["SourceFile"]))
    if source.exists():
        return source
    candidate = (
        root / "Module01_Intermediate" / "CompactStacks" /
        str(row["Tile"]) / f"{int(row['Year']):04d}" / source.name
    )
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Compact source not found: {source}\nTried: {candidate}")


class H5Cache:
    def __init__(self, capacity: int = 12) -> None:
        self.capacity = capacity
        self.files: OrderedDict[str, h5py.File] = OrderedDict()

    def get(self, path: Path) -> h5py.File:
        key = str(path)
        if key in self.files:
            self.files.move_to_end(key)
            return self.files[key]
        handle = h5py.File(path, "r")
        self.files[key] = handle
        while len(self.files) > self.capacity:
            _key, old = self.files.popitem(last=False)
            old.close()
        return handle

    def close(self) -> None:
        for handle in self.files.values():
            handle.close()
        self.files.clear()


def token_columns(
    token: dict[str, np.ndarray | str], c0: int, c1: int
) -> dict[str, np.ndarray | str]:
    """Slice cached spatial arrays while preserving DOY and protocol metadata."""
    return {
        name: (
            values[:, c0:c1]
            if isinstance(values, np.ndarray) and values.ndim == 4
            else values
        )
        for name, values in token.items()
    }


def lc_file(mask_dir: Path, tile: str, year: int) -> Path:
    candidates = (
        mask_dir / f"LC_Type_3_vegetation_mask_{tile}_{year:04d}.h5",
        mask_dir / f"vegetation_mask_{tile}_{year:04d}.h5",
        mask_dir / f"{tile}_{year:04d}_vegetation_mask.h5",
        mask_dir / f"{year:04d}.h5",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"LC_Type_3 file not found for {tile} {year}: {candidates}")


def year_window(all_years: list[int], center: int, size: int) -> tuple[list[int], int]:
    position = all_years.index(center)
    start = max(0, position - size // 2)
    start = min(start, max(len(all_years) - size, 0))
    selected = all_years[start : start + size]
    return selected, selected.index(center)


def product_path(output: Path, tile: str, year: int, doy: int) -> Path:
    return output / "Tiles" / f"{year:04d}" / f"QA_LLM_GF_{tile}_{year:04d}{doy:03d}.h5"


def compression_args(args: argparse.Namespace) -> dict:
    if args.compression == "none":
        return {}
    if args.compression == "lzf":
        return {"compression": "lzf", "shuffle": True}
    return {"compression": "gzip", "compression_opts": args.compression_level, "shuffle": True}


def routing_label(args: argparse.Namespace) -> str:
    if not args.route_solver:
        return "full"
    if args.route_mode == "learned_budget":
        return (
            "data-consistent-prior default; usable-gap cell mask, "
            "LLM solver-benefit gate, then "
            f"min(adaptive,llm_pft_cap), budgets={args.budget_levels}"
        )
    if args.route_mode == "threshold":
        return (
            f"LLM-threshold LAI={args.route_threshold_lai:.3f} "
            f"FPAR={args.route_threshold_fpar:.3f} cap={args.route_max_selected_fraction:.3f}"
        )
    return f"LLM-fraction {args.route_min_fraction:.3f}-{args.route_max_fraction:.3f}"


def initialize_products(
    args: argparse.Namespace,
    output: Path,
    tile: str,
    year: int,
    doys: list[int],
    checkpoint: Path,
    checkpoint_sha256: str,
    production_config: str,
) -> None:
    for doy in doys:
        path = product_path(output, tile, year, doy)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and not args.overwrite:
            with h5py.File(path, "r") as f:
                compatible = all(
                    name in f and f[name].shape == (args.tile_cols, args.tile_rows)
                    for name in DATASETS
                )
                if not compatible:
                    raise RuntimeError(f"Existing product has incompatible shape: {path}")
                if str(f.attrs.get("checkpoint", "")) != str(checkpoint):
                    raise RuntimeError(
                        f"Existing product uses another checkpoint: {path}. Use --overwrite 1."
                    )
                if str(f.attrs.get("checkpoint_sha256", "")) != checkpoint_sha256:
                    raise RuntimeError(
                        f"Existing product uses different checkpoint contents: {path}. "
                        "Use --overwrite 1 or another output directory."
                    )
                if str(f.attrs.get("production_config", "")) != production_config:
                    raise RuntimeError(
                        f"Existing product uses another production configuration: {path}. "
                        "Use --overwrite 1 or another output directory."
                    )
                expected_routing = routing_label(args)
                if str(f.attrs.get("solver_routing", "full")) != expected_routing:
                    raise RuntimeError(
                        f"Existing product uses another solver routing mode: {path}. "
                        "Use --overwrite 1 or another output directory."
                    )
            continue
        if path.exists():
            path.unlink()
        with h5py.File(path, "w") as f:
            for dataset_name, layer_name in zip(DATASETS, LAYERS):
                dataset = f.create_dataset(
                    dataset_name,
                    shape=(args.tile_cols, args.tile_rows),
                    dtype="uint16",
                    fillvalue=FILL,
                    chunks=(min(200, args.tile_cols), min(200, args.tile_rows)),
                    **compression_args(args),
                )
                dataset.attrs["layer_name"] = layer_name
                dataset.attrs["matlab_dimension_order"] = "row,col"
                dataset.attrs["python_h5py_dimension_order"] = "col,row"
                if layer_name.startswith("QA_"):
                    dataset.attrs["fill_value"] = int(FILL)
                    dataset.attrs["bit_layout"] = (
                        "bits0-2=source_mode,bits3-4=usable_sensor_count,"
                        "bits5-6=strict_candidate_count,"
                        "bits7-8=saturated_candidate_count,"
                        "bits9-15=confidence_0_to_100"
                    )
                else:
                    dataset.attrs["scale_factor"] = SCALE
                    dataset.attrs["fill_value"] = int(FILL)
            f.attrs["algorithm"] = "QA-LLM-GF-independent-full-controller-ST-L1"
            f.attrs["active_variable"] = args.variable
            f.attrs["uncertainty_calibration_sha256"] = args.calibration_sha256
            f.attrs["uncertainty_calibration_report_sha256"] = args.calibration_report_sha256
            f.attrs["uncertainty_application_count"] = 1
            f.attrs["uncertainty_quantity"] = "raw hierarchical sigma times Module03B ScaleFactor; not interval half-width"
            f.attrs["tile"] = tile
            f.attrs["year"] = year
            f.attrs["doy"] = doy
            f.attrs["format_version"] = (
                "independent-full-QA-MATLAB-native-three-dataset"
            )
            f.attrs["matlab_dimension_order"] = "row,col"
            f.attrs["python_h5py_dimension_order"] = "col,row"
            f.attrs["dimension_order"] = "row,col"
            f.attrs["dataset_order"] = json.dumps(dict(zip(DATASETS, LAYERS)))
            f.attrs["physical_scale_factor"] = SCALE
            f.attrs["physical_fill_value"] = int(FILL)
            f.attrs["checkpoint"] = str(checkpoint)
            f.attrs["checkpoint_sha256"] = checkpoint_sha256
            f.attrs["production_config"] = production_config
            f.attrs["solver"] = "LLM-routed adaptive log-det ST-Tensor and second-order L1 on PHENO-relative residuals"
            f.attrs["phenology_expert"] = (
                "LLM-adaptive QA/PFT-consistent same-DOY mean-to-peak candidate; "
                "support and interannual dispersion are separate routing evidence"
            )
            f.attrs["routing_definition"] = (
                "gamma=PHENO mean-to-peak; hard Local-versus-PHENO selection "
                "requires a 0.01 normalized PHENO risk advantage; independent "
                "Module2B benefit then conditional-budget routing accepts ST/L1 "
                "only on structural gaps; candidate-relative LAI peak protection "
                "rejects downward solver corrections after PHENO wins"
            )
            f.attrs["solver_routing_reference"] = (
                "QA_LLM_DATA_CONSISTENT_PRIOR; ST-Tensor and L1 run only for "
                "validation-supported positive-benefit cores and replace only "
                "sustained usable-observation gap cells"
            )
            f.attrs["quality_score"] = "state-aware semantic QA and StdDev"
            f.attrs["quality_protocol"] = json.dumps(quality_protocol_metadata())
            f.attrs["observation_constraint"] = (
                "final QA-weighted data-consistency projection; saturated candidates "
                "are soft; strict unsaturated candidates activate exact fused anchors"
            )
            f.attrs["uncertainty_model"] = (
                "quadrature of learned data uncertainty, selected LLM candidate risk, "
                "within-local-group spread, local-versus-PHENO group spread, and "
                "accepted solver adjustment"
            )
            f.attrs["qa_mode_legend"] = (
                "0=reserved,1=strict-only,2=saturated-only,3=mixed strict+saturated,"
                "4=usable soft observation without candidate,5=no-observation "
                "local-group dominant,6=no-observation PHENO-group dominant,"
                "7=cell-selective ST-Tensor+L1 reconstruction"
            )
            f.attrs["qa_bit_layout"] = (
                "bits0-2=source_mode;bits3-4=usable_sensor_count;"
                "bits5-6=strict_candidate_count;"
                "bits7-8=saturated_candidate_count;"
                "bits9-15=confidence_0_to_100"
            )
            f.attrs["solver_routing"] = routing_label(args)
            f.attrs["routing_score"] = (
                "usable-gap cell mask followed by LLM prior-relative benefit and "
                "PFT-specific conditional budget"
            )
            f.attrs["complete"] = 0
            f.attrs["completed_blocks"] = json.dumps([])


def year_product_state(output: Path, tile: str, year: int, doys: list[int]) -> str:
    paths = [product_path(output, tile, year, doy) for doy in doys]
    if not any(path.exists() for path in paths):
        return "missing"
    complete = []
    for path in paths:
        if not path.exists():
            complete.append(False)
            continue
        try:
            with h5py.File(path, "r") as f:
                complete.append(int(f.attrs.get("complete", 0)) == 1)
        except OSError:
            complete.append(False)
    if all(complete):
        return "complete"
    return "partial"


def read_completed_blocks(output: Path, tile: str, year: int, doys: list[int]) -> set[str]:
    path = product_path(output, tile, year, doys[0])
    if not path.exists():
        return set()
    with h5py.File(path, "r") as f:
        value = f.attrs.get("completed_blocks", "[]")
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    try:
        return {str(item) for item in json.loads(str(value))}
    except (TypeError, ValueError, json.JSONDecodeError):
        return set()


class YearProductWriter:
    """Keep all 46 yearly product files open across block writes."""

    def __init__(self, output: Path, tile: str, year: int, doys: list[int]) -> None:
        self.handles = [h5py.File(product_path(output, tile, year, doy), "r+") for doy in doys]

    def write_block(self, block: np.ndarray, r0: int, c0: int) -> None:
        height, width = block.shape[1:3]
        for ti, handle in enumerate(self.handles):
            for li, dataset_name in enumerate(DATASETS):
                handle[dataset_name][
                    c0 : c0 + width, r0 : r0 + height
                ] = block[ti, :, :, li].T

    def mark_block_complete(self, completed: set[str], block_name: str) -> None:
        # A resume marker is valid only after every DOY dataset is durable. A
        # hard stop between marking the first file and flushing the others would
        # otherwise make a partially written block look complete.
        for handle in self.handles:
            handle.flush()
        completed.add(block_name)
        encoded = json.dumps(sorted(completed))
        # Resume state is read from the first DOY file; final close mirrors it.
        self.handles[0].attrs["completed_blocks"] = encoded
        self.handles[0].flush()

    def mark_complete(self, elapsed: float) -> None:
        for handle in self.handles:
            handle.attrs["completed_blocks"] = self.handles[0].attrs["completed_blocks"]
            handle.attrs["complete"] = 1
            handle.attrs["elapsed_year_sec"] = elapsed

    def close(self) -> None:
        for handle in self.handles:
            handle.flush()
            handle.close()
        self.handles.clear()


def encode_physical(value: np.ndarray) -> np.ndarray:
    out = np.full(value.shape, FILL, dtype=np.uint16)
    valid = np.isfinite(value)
    out[valid] = np.clip(np.rint(value[valid] / SCALE), 0, int(FILL) - 1).astype(np.uint16)
    return out


def pack_qa(
    mode: torch.Tensor,
    usable: torch.Tensor,
    strict: torch.Tensor,
    saturated: torch.Tensor,
    confidence: torch.Tensor,
) -> torch.Tensor:
    mode_bits = mode.clamp(0, 7).to(torch.int32)
    return (
        mode_bits |
        (usable.clamp(0, 3).to(torch.int32) << 3) |
        (strict.clamp(0, 3).to(torch.int32) << 5) |
        (saturated.clamp(0, 3).to(torch.int32) << 7) |
        (confidence.clamp(0, 100).to(torch.int32) << 9)
    )


def compose_layers(
    result: dict,
    center_x: torch.Tensor,
    vegetation: torch.Tensor,
    core_size: int,
    uncertainty_scales: tuple[float, float] = (1.0, 1.0),
    *, variable=None, product_calibration=None, pft=None,
) -> np.ndarray:
    prediction = result["prediction"]
    b, variables, time_count, h, w = prediction.shape
    valid = torch.stack([center_x[:, 2] > 0.5, center_x[:, 3] > 0.5], dim=1)
    candidates = torch.stack([center_x[:, 6] > 0.5, center_x[:, 7] > 0.5], dim=1)
    saturated = (center_x[:, 18] > 0.5).unsqueeze(1).expand_as(candidates)
    strict_count = (candidates & ~saturated).sum(dim=-1)
    saturated_count = (candidates & saturated).sum(dim=-1)
    usable_count = valid.sum(dim=-1)
    difficulty = result["difficulty"][:, :, None, None, None]
    pheno_supported = result["expert_support"][:, :, 3] > 0.0
    pheno_dominant = pheno_supported & (
        result["phenology_group_gate"] >= 0.5
    )
    no_observation = torch.where(pheno_dominant, 6, 5)
    mode = torch.where(
        (strict_count > 0) & (saturated_count > 0),
        3,
        torch.where(
            saturated_count > 0,
            2,
            torch.where(
                strict_count > 0,
                1,
                torch.where(usable_count > 0, 4, no_observation),
            ),
        ),
    )
    structural_fill = result.get("solver_structural_fill_mask")
    if structural_fill is not None:
        mode = torch.where(structural_fill, torch.full_like(mode, 7), mode)

    scale = torch.tensor([7.0, 1.0], dtype=prediction.dtype, device=prediction.device).view(1, 2, 1, 1, 1)
    uncertainty = hierarchical_uncertainty_components(result)["total"]
    calibration = torch.as_tensor(
        uncertainty_scales, dtype=uncertainty.dtype, device=uncertainty.device
    ).view(1, 2, 1, 1, 1)
    uncertainty = uncertainty * calibration
    if product_calibration is not None:
        if uncertainty_scales != (1.0, 1.0) or variable not in ("LAI", "FPAR"):
            raise ValueError("Do not combine Module02C scalar scaling with Module03B calibration")
        vi = ("LAI", "FPAR").index(variable)
        uncertainty = uncertainty.clone()
        uncertainty[:, vi] = product_calibration.apply_raw(uncertainty[:, vi], mode[:, vi], pft)
    confidence = 100.0 * torch.exp(-uncertainty / scale.clamp_min(1e-6)) * (1.0 - 0.2 * difficulty)
    confidence = confidence.clamp(0.0, 100.0).round()

    vegetation = vegetation[:, None, None].expand(b, variables, time_count, h, w)
    vegetation = vegetation & torch.isfinite(prediction) & torch.isfinite(uncertainty) & (uncertainty >= 0)
    prediction = torch.where(vegetation, prediction, torch.full_like(prediction, float("nan")))
    uncertainty = torch.where(vegetation, uncertainty, torch.full_like(uncertainty, float("nan")))
    qa = pack_qa(
        mode, usable_count, strict_count, saturated_count, confidence
    )
    qa = torch.where(vegetation, qa, torch.full_like(qa, int(FILL)))

    layer = np.full((b, time_count, h, w, 6), FILL, dtype=np.uint16)
    pred_np = prediction.detach().cpu().numpy()
    unc_np = uncertainty.detach().cpu().numpy()
    qa_np = qa.detach().cpu().numpy().astype(np.uint16)
    for vi in range(2):
        layer[..., vi] = encode_physical(pred_np[:, vi])
        layer[..., vi + 2] = encode_physical(unc_np[:, vi])
        layer[..., vi + 4] = qa_np[:, vi]
    if variable is not None:
        vi = ("LAI", "FPAR").index(variable)
        return layer[..., [vi, vi + 2, vi + 4]]
    return layer


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def upsert_summary(rows: list[dict], row: dict) -> None:
    key = (int(row["Year"]), int(row["BlockRow"]), int(row["BlockCol"]))
    rows[:] = [
        item for item in rows
        if (int(item["Year"]), int(item["BlockRow"]), int(item["BlockCol"])) != key
    ]
    rows.append(row)


def validate_execution_release(args, status) -> None:
    if status.get("full_chain_released", False):
        return
    root = args.root.resolve()
    from .production_release import production_authorized
    if (production_authorized(status, [root.name])
            and args.variable in ("LAI", "FPAR")
            and args.output_dir is not None
            and args.output_dir.resolve() == root / "Module4_Product" / "Parts" / args.variable
            and parse_ints(args.years)
            and set(parse_ints(args.years)).issubset(range(2001, 2025))):
        return
    expected = root / "Module04A_Pilot" / args.variable
    output = (args.output_dir or expected).resolve()
    pilot = (
        status.get("pilot_chain_authorized", False)
        and args.variable in ("LAI", "FPAR")
        and root.name in PRODUCTION_TILES
        and parse_ints(args.years) == [2013]
        and output == expected.resolve()
    )
    if not pilot:
        raise RuntimeError("Product execution locked: only authorized 2013 Module04A pilot is permitted")


def main(*, raw_cache_factory=None) -> None:
    global LAYERS, DATASETS
    args = parse_args()
    status = json.loads((Path(__file__).resolve().parents[1] / "restoration_status.json").read_text())
    validate_execution_release(args, status)
    if any((args.share_window_frontends, args.annual_feature_cache, args.fold_constant_heads, args.production_fastpath)):
        raise ValueError("Shared/annual fastpaths are not yet certified for independent-variable inputs")
    if not args.route_solver or args.route_mode != "learned_budget":
        raise ValueError("Production requires calibrated learned-budget ST/L1 routing")
    if (args.uncertainty_scale_lai, args.uncertainty_scale_fpar) != (1.0, 1.0):
        raise ValueError("Use Module03B calibration only, without additional scalar scaling")
    vi = ("LAI", "FPAR").index(args.variable)
    LAYERS = tuple(("LAI", "FPAR", "uncertainty_LAI", "uncertainty_FPAR", "QA_LAI", "QA_FPAR")[i] for i in (vi, vi+2, vi+4))
    DATASETS = tuple(("/Lai", "/Fpar", "/Lai_Uncertainty", "/Fpar_Uncertainty", "/Lai_QA", "/Fpar_QA")[i] for i in (vi, vi+2, vi+4))
    args.root = args.root.resolve()
    args.lc_mask_dir = args.lc_mask_dir.resolve()
    output = (
        args.output_dir
        or args.root / "Module04A_Pilot" / args.variable
    ).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    device = resolve_device(args.device)
    checkpoint_path = args.checkpoint.resolve()
    model, solver, checkpoint = load_model(checkpoint_path, device)
    model._fold_constant_heads = bool(args.fold_constant_heads)
    checkpoint_digest = file_sha256(checkpoint_path)
    product_calibration = prepare_product_contract(args, checkpoint)
    production_config = production_config_label(args)
    input_channels = int(checkpoint["model_config"]["input_channels"])
    lc_codes = parse_ints(args.lc_codes) if input_channels > 19 else []
    sensor_start_years = parse_ints(args.sensor_start_years)
    budget_levels = parse_ints(args.budget_levels)
    requested_years = parse_ints(args.years)
    selected_block_rows = set(parse_ints(args.block_rows))
    selected_block_cols = set(parse_ints(args.block_cols))
    expected_block_rows = set(range(1, math.ceil(args.tile_rows / 600) + 1))
    expected_block_cols = set(range(1, math.ceil(args.tile_cols / 600) + 1))
    expected_blocks = {
        f"R{row:02d}C{col:02d}" for row in expected_block_rows for col in expected_block_cols
    }

    index_path = args.root / "Module01_Intermediate" / "PythonDataset" / "token_index_python.csv"
    index = pd.read_csv(index_path)
    if set(index['Tile'].astype(str)) != {args.root.name}:
        raise ValueError('Production index does not match the requested tile')
    index = index[
        index["BlockRow"].astype(int).isin(selected_block_rows) &
        index["BlockCol"].astype(int).isin(selected_block_cols)
    ].copy()
    available_years = sorted(set(index["Year"].astype(int)))
    if not set(requested_years).issubset(available_years):
        raise ValueError('Requested production years are missing from the intermediate index')
    tile = str(index.iloc[0]["Tile"])
    lookup = {}
    for row in index.to_dict("records"):
        key = (
            int(row["BlockRow"]), int(row["BlockCol"]),
            int(row["GlobalRowStart"]), int(row["GlobalColStart"]), int(row["Year"]),
        )
        if key in lookup:
            raise ValueError(f"Duplicate production token: {key}")
        lookup[key] = row
    source_cache, lc_cache = H5Cache(16), H5Cache(8)
    raw_cache_mb = cache_budget_mb(args.input_cache_mb, args.max_patches_per_block > 0) if args.strip_cache else 0
    cache_factory = raw_cache_factory or TokenStripCache
    strip_cache = cache_factory(args.window_years, raw_cache_mb)
    if hasattr(strip_cache, 'resident_bytes'):
        raw_cache_mb = int(math.ceil(strip_cache.resident_bytes / 1024**2))
    print(f"Production execution: device={device}, threads={torch.get_num_threads()}, "
          f"batch={args.batch_size}, raw_cache_target={raw_cache_mb} MiB "
          f"({'slabs' if raw_cache_mb else 'legacy strips/direct reads'}); model patches unchanged", flush=True)
    summary_path = Path(args.summary_file)
    if not summary_path.is_absolute():
        summary_path = output / "Summaries" / summary_path
    if summary_path.exists() and summary_path.stat().st_size > 0:
        summary = pd.read_csv(summary_path).to_dict("records")
    else:
        summary = []

    shared_done = set()
    annual_groups = {}
    if args.annual_feature_cache:
        from .annual_features import bounded_year_groups, cache_fitted_group_size
        group_size = args.year_group_size
        if isinstance(strip_cache,TokenStripCache) and raw_cache_mb>0:
            source = resolve_source(args.root,index.iloc[0])
            source_bytes = strip_cache.full_block_bytes(source,source_cache.get(source))
            group_size = cache_fitted_group_size(group_size,args.window_years,raw_cache_mb,source_bytes)
        print(f'Annual execution: group_size={group_size} (requested={args.year_group_size}); '
              f'constant_head_folding={bool(args.fold_constant_heads)}',flush=True)
        for group in bounded_year_groups([y for y in requested_years if y in available_years],
                                         available_years,group_size,args.window_years):
            for year in group:
                annual_groups[year] = group
    try:
        for center_year in requested_years:
            if center_year in shared_done:
                continue
            if center_year not in available_years:
                continue
            window, center_index = year_window(available_years, center_year, args.window_years)
            if args.annual_feature_cache:
                from .product_windows import process_shared_years
                group = annual_groups[center_year]
                peers = [y for y in group if y not in shared_done]
                process_shared_years(args,peers,output=output,tile=tile,window=group,
                                     index=index,lookup=lookup,model=model,solver=solver,device=device,
                                     source_cache=source_cache,lc_cache=lc_cache,strip_cache=strip_cache,
                                     summary=summary,summary_path=summary_path,checkpoint_path=checkpoint_path,
                                     checkpoint_digest=checkpoint_digest,production_config=production_config,
                                     expected_blocks=expected_blocks)
                shared_done.update(peers)
                continue
            if args.share_window_frontends:
                peers = [y for y in requested_years if y in available_years and y not in shared_done
                         and year_window(available_years,y,args.window_years)[0] == window]
                if len(peers)>1:
                    from .product_windows import process_shared_years
                    process_shared_years(args,peers,output=output,tile=tile,window=window,
                                         index=index,lookup=lookup,model=model,solver=solver,device=device,
                                         source_cache=source_cache,lc_cache=lc_cache,strip_cache=strip_cache,
                                         summary=summary,summary_path=summary_path,checkpoint_path=checkpoint_path,
                                         checkpoint_digest=checkpoint_digest,production_config=production_config,
                                         expected_blocks=expected_blocks)
                    shared_done.update(peers)
                    continue
            center_rows = index[index["Year"].astype(int) == center_year]
            sample_source = resolve_source(args.root, center_rows.iloc[0])
            sample_doy = np.asarray(source_cache.get(sample_source).attrs.get(
                "DOY", np.arange(1, 362, 8)), dtype=np.float32)
            doys = [int(round(x)) for x in sample_doy]
            initialize_products(
                args,
                output,
                tile,
                center_year,
                doys,
                checkpoint_path,
                checkpoint_digest,
                production_config,
            )
            state = year_product_state(output, tile, center_year, doys)
            if state == "complete" and not args.overwrite:
                print(f"year={center_year} already complete; skipped", flush=True)
                continue
            completed_blocks = read_completed_blocks(output, tile, center_year, doys)
            year_tic = time.time()
            writer = YearProductWriter(output, tile, center_year, doys)
            try:
                for (block_row, block_col), rows in center_rows.groupby(["BlockRow", "BlockCol"], sort=True):
                    block_name = f"R{int(block_row):02d}C{int(block_col):02d}"
                    if block_name in completed_blocks and not args.overwrite:
                        print(f"year={center_year} block={block_name} already complete; skipped", flush=True)
                        continue
                    rows = rows.sort_values(["GlobalRowStart", "GlobalColStart"]).reset_index(drop=True)
                    block_r0 = (int(block_row) - 1) * 600
                    block_c0 = (int(block_col) - 1) * 600
                    block_r1 = min(block_r0 + 600, int(args.tile_rows))
                    block_c1 = min(block_c0 + 600, int(args.tile_cols))
                    block = np.full((len(doys), 600, 600, len(DATASETS)), FILL, dtype=np.uint16)
                    strip_cache.clear()

                    # LC is reused by every 40x40 patch in this block. Reading
                    # one 600x600 array per window year avoids thousands of
                    # small HDF5 reads without changing any model input.
                    lc_blocks: dict[int, np.ndarray] = {}
                    center_vegetation_block: np.ndarray | None = None
                    for year in window:
                        lc_handle = lc_cache.get(lc_file(args.lc_mask_dir, tile, year))
                        lc_blocks[year] = read_lc_grid_patch(
                            lc_handle["/LC_Type_3"],
                            block_r0,
                            block_r1,
                            block_c0,
                            block_c1,
                            lc_handle,
                        )
                        if year == center_year:
                            center_vegetation_block = read_lc_grid_patch(
                                lc_handle["/vegetation_mask"],
                                block_r0,
                                block_r1,
                                block_c0,
                                block_c1,
                                lc_handle,
                            ) > 0
                    if center_vegetation_block is None:
                        raise RuntimeError(
                            f"Missing center-year LC block for {tile} {center_year} {block_name}"
                        )
                    patch_used = 0
                    route_sum = 0.0
                    route_count = 0
                    route_score_sum = 0.0
                    route_score_count = 0
                    budget_fraction_sum = 0.0
                    budget_fraction_count = 0
                    active_fraction_sum = 0.0
                    active_fraction_count = 0
                    st_iteration_sum = 0.0
                    st_iteration_count = 0
                    l1_iteration_sum = 0.0
                    l1_iteration_count = 0
                    block_tic = time.time()
                    input_seconds = 0.0
                    inference_seconds = 0.0
                    assembly_seconds = 0.0
                    next_progress = max(1, args.print_every)
                    for start in range(0, len(rows), args.batch_size):
                        if args.max_patches_per_block > 0 and patch_used >= args.max_patches_per_block:
                            break
                        batch_rows = rows.iloc[start : start + args.batch_size]
                        input_tic = time.perf_counter()
                        windows, centers, vegetation_masks, locations = [], [], [], []
                        for _idx, center_row in batch_rows.iterrows():
                            spatial = (
                                int(center_row["BlockRow"]), int(center_row["BlockCol"]),
                                int(center_row["GlobalRowStart"]), int(center_row["GlobalColStart"]),
                            )
                            center_gr0 = int(center_row["GlobalRowStart"]) - 1
                            center_gr1 = int(center_row["GlobalRowEnd"])
                            center_gc0 = int(center_row["GlobalColStart"]) - 1
                            center_gc1 = int(center_row["GlobalColEnd"])
                            local_r0 = center_gr0 - block_r0
                            local_r1 = center_gr1 - block_r0
                            local_c0 = center_gc0 - block_c0
                            local_c1 = center_gc1 - block_c0
                            vegetation = center_vegetation_block[
                                local_r0:local_r1, local_c0:local_c1
                            ]
                            vegetation_masks.append(vegetation)
                            locations.append((
                                center_gr0 - block_r0,
                                center_gr1 - block_r0,
                                center_gc0 - block_c0,
                                center_gc1 - block_c0,
                            ))
                            if not bool(vegetation.any()):
                                windows.append(None)
                                centers.append(None)
                                continue
                            year_inputs = []
                            for year in window:
                                row = lookup[spatial + (year,)]
                                rr0 = int(row["LocalRowStart"]) - 1
                                rr1 = int(row["LocalRowEnd"])
                                cc0 = int(row["LocalColStart"]) - 1
                                cc1 = int(row["LocalColEnd"])
                                gr0 = int(row["GlobalRowStart"]) - 1
                                gr1 = int(row["GlobalRowEnd"])
                                gc0 = int(row["GlobalColStart"]) - 1
                                gc1 = int(row["GlobalColEnd"])
                                source = resolve_source(args.root, row)
                                source_handle = source_cache.get(source)
                                if args.strip_cache:
                                    token = token_columns(
                                        strip_cache.get(source, source_handle, rr0, rr1), cc0, cc1
                                    )
                                else:
                                    token = read_token_arrays(source_handle, rr0, rr1, cc0, cc1)
                                lc = lc_blocks[year][
                                    gr0 - block_r0:gr1 - block_r0,
                                    gc0 - block_c0:gc1 - block_c0,
                                ]
                                year_inputs.append(build_input(
                                    token, year, sensor_start_years, lc, lc_codes,
                                    preallocate=bool(args.production_fastpath),
                                    active_variable=args.variable,
                                ))
                            windows.append(np.stack(year_inputs))
                            centers.append(year_inputs[center_index])
                        active_np = np.asarray(
                            [bool(mask.any()) for mask in vegetation_masks], dtype=bool
                        )
                        patch_height, patch_width = vegetation_masks[0].shape
                        layers = np.full(
                            (len(locations), len(doys), patch_height, patch_width, len(DATASETS)),
                            FILL,
                            dtype=np.uint16,
                        )
                        if bool(active_np.any()):
                            active_windows = np.stack(
                                [windows[index] for index in np.flatnonzero(active_np)]
                            )
                            active_vegetation = np.stack(
                                [vegetation_masks[index] for index in np.flatnonzero(active_np)]
                            )
                            x_window = torch.as_tensor(active_windows, dtype=torch.float32, device=device)
                            center_x = x_window[:, center_index]
                            vegetation_t = torch.as_tensor(
                                active_vegetation, dtype=torch.bool, device=device
                            )
                            if device.type == "cuda":
                                torch.cuda.synchronize(device)
                            input_seconds += time.perf_counter() - input_tic
                            inference_tic = time.perf_counter()
                            with torch.inference_mode():
                                route_kwargs = {}
                                if args.route_solver:
                                    route_kwargs = {
                                        "route_mode": args.route_mode,
                                        "route_min_fraction": args.route_min_fraction,
                                        "route_max_fraction": args.route_max_fraction,
                                        "route_thresholds": (
                                            args.route_threshold_lai,
                                            args.route_threshold_fpar,
                                        ),
                                        "route_max_selected_fraction": (
                                            args.route_max_selected_fraction
                                        ),
                                        "budget_levels": budget_levels,
                                        "domain_mask": vegetation_t,
                                        "structural_cell_blend": True,
                                    }
                                result = run_window(
                                    model, x_window, center_index, solver, args.core_size, False,
                                    production_only=bool(args.production_fastpath),
                                    **route_kwargs,
                                )
                                active_layers = compose_layers(
                                    result,
                                    center_x,
                                    vegetation_t,
                                    args.core_size,
                                    (args.uncertainty_scale_lai, args.uncertainty_scale_fpar),
                                    variable=args.variable,
                                    product_calibration=product_calibration,
                                    pft=center_landcover(x_window,
                                        torch.full((x_window.shape[0],), center_index, device=device), lc_codes)[0],
                                )
                            route_sum += float(result["solver_route_fraction"].sum().cpu())
                            route_count += int(result["solver_route_fraction"].numel())
                            route_score_sum += float(result["solver_route_score"].sum().cpu())
                            route_score_count += int(result["solver_route_score"].numel())
                            budget_fraction_sum += float(
                                result["solver_budget_fraction"].sum().cpu()
                            )
                            budget_fraction_count += int(result["solver_budget_fraction"].numel())
                            active_fraction_sum += float(
                                result["solver_active_fraction"].sum().cpu()
                            )
                            active_fraction_count += int(result["solver_active_fraction"].numel())
                            st_iterations = result["st_diagnostics"]["iterations"]
                            l1_iterations = result["l1_diagnostics"]["iterations"]
                            st_iteration_sum += float(st_iterations.sum().cpu())
                            st_iteration_count += int(st_iterations.numel())
                            l1_iteration_sum += float(l1_iterations.sum().cpu())
                            l1_iteration_count += int(l1_iterations.numel())
                            layers[active_np] = active_layers
                            inference_seconds += time.perf_counter() - inference_tic
                        else:
                            input_seconds += time.perf_counter() - input_tic
                        assembly_tic = time.perf_counter()
                        for bi, (r0, r1, c0, c1) in enumerate(locations):
                            block[:, r0:r1, c0:c1, :] = layers[bi]
                        patch_used += len(locations)
                        assembly_seconds += time.perf_counter() - assembly_tic
                        if args.print_every > 0 and (patch_used >= next_progress or patch_used == len(rows)):
                            next_progress = patch_used + args.print_every
                            elapsed_block = time.time() - block_tic
                            eta = elapsed_block * (len(rows) - patch_used) / max(patch_used, 1)
                            print(
                                f"year={center_year} block=R{int(block_row):02d}C{int(block_col):02d} "
                                f"patch={patch_used}/{len(rows)} ({100*patch_used/len(rows):.1f}%) "
                                f"elapsed={elapsed_block:.1f}s block_ETA={eta:.1f}s "
                                f"input={input_seconds:.1f}s infer={inference_seconds:.1f}s "
                                f"raw_read={strip_cache.read_seconds:.1f}s cache_loads={strip_cache.loads} "
                                f"slab_rows={strip_cache.last_slab_rows}",
                                flush=True,
                            )
                    write_tic = time.perf_counter()
                    writer.write_block(block, block_r0, block_c0)
                    if patch_used == len(rows):
                        writer.mark_block_complete(completed_blocks, block_name)
                    write_seconds = time.perf_counter() - write_tic
                    upsert_summary(summary, {
                        "Year": center_year,
                        "BlockRow": int(block_row),
                        "BlockCol": int(block_col),
                        "PatchCount": patch_used,
                        "MeanSolverRouteFraction": route_sum / route_count if route_count else 0.0,
                        "MeanSolverRouteScore": (
                            route_score_sum / route_score_count if route_score_count else 0.0
                        ),
                        "MeanSolverBudgetFraction": (
                            budget_fraction_sum / budget_fraction_count
                            if budget_fraction_count else 0.0
                        ),
                        "MeanActiveCoreFraction": (
                            active_fraction_sum / active_fraction_count
                            if active_fraction_count else 0.0
                        ),
                        "MeanSTIterations": (
                            st_iteration_sum / st_iteration_count if st_iteration_count else 0.0
                        ),
                        "MeanL1Iterations": (
                            l1_iteration_sum / l1_iteration_count if l1_iteration_count else 0.0
                        ),
                        "ElapsedSec": time.time() - block_tic,
                        "InputSec": input_seconds,
                        "InferenceAndPackingSec": inference_seconds,
                        "AssemblySec": assembly_seconds,
                        "WriteSec": write_seconds,
                        "Execution": "shared_l1_rhs_raw_slabs",
                        "RawCacheTargetMiB": raw_cache_mb,
                        "RawReadSec": strip_cache.read_seconds,
                        "RawCacheLoads": strip_cache.loads,
                        "RawCacheHits": strip_cache.hits,
                        "RawBytesLoaded": strip_cache.loaded_bytes,
                    })
                    write_csv(summary_path, summary)
                elapsed = time.time() - year_tic
                is_complete = expected_blocks.issubset(completed_blocks)
                if is_complete:
                    writer.mark_complete(elapsed)
            finally:
                writer.close()
            write_csv(summary_path, summary)
            status = "complete" if is_complete else "partial output (resumable by block)"
            print(f"year={center_year} {status} elapsed={elapsed/60:.1f} min", flush=True)
    finally:
        source_cache.close()
        lc_cache.close()
    print(f"Product generation complete: {output}", flush=True)


if __name__ == "__main__":
    main()
