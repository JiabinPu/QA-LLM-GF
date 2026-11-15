"""Calibrate the two-stage, peak-safe residual-solver router."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from qg_dataset import DatasetOptions, read_lc_grid_patch

from .data import WindowedQADataset
from .variable import variable_index
from .model import ControllerConfig, QALLMAdaptiveController
from .pipeline import (
    aggregate_withheld_target,
    extract_budget_context,
    extract_budget_pft,
    extract_structural_gap_mask,
    hierarchical_uncertainty_components,
    prepare_window_frontend,
    run_window,
)
from .quality import (
    BUDGET_CHECKPOINT_VERSION,
    quality_protocol_metadata,
    validate_checkpoint_protocol,
    validate_checkpoint_tile,
)
from .solver import SolverConfig
from .utils import (
    parse_ints,
    parse_items,
    resolve_device,
    saturation_aware_target_weights,
    set_seed,
    split_target_domains,
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def oracle_cache_signature(
    args: argparse.Namespace,
    split: str,
    sample_count: int,
    levels: torch.Tensor,
    sampling_policy: str,
) -> tuple[str, dict[str, object]]:
    manifest_dir = (
        args.root.resolve()
        / "Module01_Intermediate"
        / "PythonDataset"
    )
    manifests = {
        "token_index": manifest_dir / "token_index_python.csv",
        "mask_manifest": manifest_dir / f"mask_manifest_{split}.csv",
    }
    missing = [str(path) for path in manifests.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Oracle cache input is missing: " + ", ".join(missing))
    lc_inventory = []
    indexed_lc: set[Path] = set(args.lc_mask_dir.resolve().glob("*.h5"))
    token_index = manifests["token_index"]
    if token_index.is_file():
        indexed = pd.read_csv(token_index, usecols=lambda name: name == "LCSourceFile")
        if "LCSourceFile" in indexed.columns:
            indexed_lc.update(
                Path(value).resolve()
                for value in indexed["LCSourceFile"].dropna().astype(str).unique()
                if value.strip()
            )
    for path in sorted(indexed_lc, key=str):
        if not path.is_file():
            continue
        stat = path.stat()
        lc_inventory.append((str(path), int(stat.st_size), int(stat.st_mtime_ns)))
    payload: dict[str, object] = {
        "schema": "qa-llm-gf-independent-oracle-cache",
        "active_variable": args.variable,
        "sampling_policy": sampling_policy,
        "split": split,
        "sample_count": int(sample_count),
        "seed": int(args.seed),
        "base_checkpoint_sha256": file_sha256(args.base_checkpoint.resolve()),
        "manifest_sha256": {
            name: file_sha256(path) for name, path in manifests.items()
        },
        "lc_inventory": lc_inventory,
        "scenarios": parse_items(args.scenarios),
        "window_years": int(args.window_years),
        "sensor_start_years": parse_ints(args.sensor_start_years),
        "lc_codes": parse_ints(args.lc_codes),
        "core_size": int(args.core_size),
        "budget_levels": [int(value) for value in levels.detach().cpu().tolist()],
        "normalized_error_tolerance": float(args.normalized_error_tolerance),
        "minimum_solver_gain": float(args.minimum_solver_gain),
        "saturated_target_weight_floor": float(args.saturated_target_weight_floor),
        "pft_stratified_sampling": bool(args.pft_stratified_sampling),
        "structural_min_run_steps": int(args.structural_min_run_steps),
        "structural_max_patch_missing": float(args.structural_max_patch_missing),
        "structural_max_support_deficit": float(args.structural_max_support_deficit),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Calibrate the QA-LLM solver budget head")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--variable", choices=("LAI", "FPAR"), required=True)
    p.add_argument("--base-checkpoint", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--oracle-cache-dir", type=Path, default=None)
    p.add_argument("--validation-oracle-cache-dir", type=Path, default=None)
    p.add_argument("--scenarios", required=True)
    p.add_argument("--window-years", type=int, default=5)
    p.add_argument("--sensor-start-years", default="2001,2002,2012")
    p.add_argument("--lc-mask-dir", type=Path, required=True)
    p.add_argument("--lc-codes", default="1:8")
    p.add_argument("--core-size", type=int, default=8)
    p.add_argument("--budget-levels", default="0,3,5,8,13,24")
    p.add_argument("--normalized-error-tolerance", type=float, default=0.005)
    p.add_argument("--minimum-solver-gain", type=float, default=0.002)
    p.add_argument("--saturated-target-weight-floor", type=float, default=0.70)
    p.add_argument("--train-samples-per-scenario", type=int, default=20)
    p.add_argument("--val-samples-per-scenario", type=int, default=5)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--learning-rate", type=float, default=2e-3)
    p.add_argument("--context-learning-rate", type=float, default=5e-2)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--class-weight-power", type=float, default=0.5)
    p.add_argument("--class-weight-cap", type=float, default=4.0)
    p.add_argument("--budget-regression-weight", type=float, default=0.5)
    p.add_argument("--underbudget-penalty-weight", type=float, default=0.25)
    p.add_argument("--need-solver-loss-weight", type=float, default=1.0)
    p.add_argument("--zero-overbudget-penalty-weight", type=float, default=0.25)
    p.add_argument("--policy-grid-min-quantile", type=float, default=0.05)
    p.add_argument("--policy-min-relative-nrmse-gain", type=float, default=0.002)
    p.add_argument("--policy-max-relative-nrmse-increase", type=float, default=0.002)
    p.add_argument("--policy-max-scenario-relative-nrmse-increase", type=float, default=0.01)
    p.add_argument("--policy-max-selected-fraction", type=float, default=0.25)
    p.add_argument("--policy-max-high-lai-relative-rmse-increase", type=float, default=0.0)
    p.add_argument("--policy-max-high-lai-absolute-bias-increase", type=float, default=0.0)
    p.add_argument(
        "--policy-require-spatial-block-lai-improvement", type=int, default=0
    )
    p.add_argument("--reuse-trained-head", type=int, default=0)
    p.add_argument("--refresh-budget-contexts", type=int, default=0)
    p.add_argument("--min-pft-calibration-fraction", type=float, default=0.01)
    p.add_argument("--min-pft-calibration-cores", type=int, default=100)
    p.add_argument("--pft-stratified-sampling", type=int, default=1)
    p.add_argument("--structural-min-run-steps", type=int, default=3)
    p.add_argument("--structural-max-patch-missing", type=float, default=0.80)
    p.add_argument("--structural-max-support-deficit", type=float, default=0.95)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--torch-threads", type=int, default=4)
    p.add_argument("--seed", type=int, default=20260810)
    p.add_argument("--overwrite-oracle-cache", type=int, default=0)
    p.add_argument("--print-every", type=int, default=5)
    return p.parse_args()


def load_upgraded_model(path: Path, levels: list[int], device: torch.device, variable: str):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    validate_checkpoint_protocol(checkpoint, budget=False)
    if checkpoint['model_config'].get('active_variable') != variable:
        raise ValueError(f'Wrong controller variable: expected {variable}.')
    config = dict(checkpoint["model_config"])
    config.update(
        use_budget_head=True,
        use_two_stage_budget_head=True,
        num_budget_levels=len(levels),
        solver_peak_safe_routing=True,
        solver_peak_safe_margin_lai=0.0,
    )
    model = QALLMAdaptiveController(ControllerConfig(**config)).to(device)
    result = model.load_state_dict(checkpoint["model_state"], strict=False)
    unexpected = list(result.unexpected_keys)
    parameter_map = dict(model.named_parameters())
    missing = [
        name for name in result.missing_keys
        if not name.startswith("budget_head.")
        and not name.startswith("budget_benefit_head.")
        and name not in {"budget_context_weight", "budget_benefit_context_weight"}
        and name in parameter_map
        and parameter_map[name].requires_grad
    ]
    if unexpected or missing:
        raise RuntimeError(f"Incompatible base checkpoint; missing={missing}, unexpected={unexpected}")
    for parameter in model.parameters():
        parameter.requires_grad = False
    if model.budget_head is None:
        raise RuntimeError("Failed to create the solver budget head.")
    for parameter in model.budget_head.parameters():
        parameter.requires_grad = True
    if model.budget_benefit_head is None:
        raise RuntimeError("Failed to create the solver-benefit head.")
    for parameter in model.budget_benefit_head.parameters():
        parameter.requires_grad = True
    if model.budget_context_weight is None:
        raise RuntimeError("Failed to create the QA-gap budget context weights.")
    model.budget_context_weight.requires_grad = True
    if model.budget_benefit_context_weight is None:
        raise RuntimeError("Failed to create benefit-context weights.")
    model.budget_benefit_context_weight.requires_grad = True
    solver = SolverConfig(**checkpoint["solver_config"])
    if levels[-1] != int(solver.max_iterations):
        raise ValueError(
            f"Last budget level ({levels[-1]}) must equal solver max_iterations "
            f"({solver.max_iterations})."
        )
    return model, solver, checkpoint


def build_dataset(args: argparse.Namespace, split: str, sample_count: int) -> WindowedQADataset:
    stratified = bool(args.pft_stratified_sampling)
    dataset = WindowedQADataset(
        args.root,
        split,
        parse_items(args.scenarios),
        args.window_years,
        parse_ints(args.sensor_start_years),
        DatasetOptions(
            active_variable=args.variable,
            return_meta=False,
            min_anchor_ratio=0.0,
            h5_cache_size=8,
            token_cache_size=8,
            use_lc_context=True,
            lc_mask_dir=args.lc_mask_dir,
            lc_codes=parse_ints(args.lc_codes),
        ),
        max_samples_per_scenario=0 if stratified else sample_count,
        seed=args.seed + (0 if split == "train" else 1),
    )
    if stratified:
        dataset.base.mask_manifest = pft_stratified_manifest(
            dataset,
            args,
            sample_count,
            args.seed + 1,
        )
        dataset.target_samples_per_scenario = int(sample_count)
    return dataset


def calibrate_validation_uncertainty(
    args: argparse.Namespace,
    model: QALLMAdaptiveController,
    solver: SolverConfig,
    levels: torch.Tensor,
    device: torch.device,
) -> list[dict[str, object]]:
    """Calibrate final routed uncertainty on complete gaps from the val split."""
    dataset = build_dataset(args, "val", args.val_samples_per_scenario)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    weight_sum = torch.zeros(2, dtype=torch.float64)
    z_sq_sum = torch.zeros(2, dtype=torch.float64)
    coverage_68 = torch.zeros(2, dtype=torch.float64)
    coverage_95 = torch.zeros(2, dtype=torch.float64)
    pixel_count = torch.zeros(2, dtype=torch.int64)
    model.eval()
    try:
        for batch_index, batch in enumerate(loader, start=1):
            x_window = batch["x_window"].to(device)
            center_index = batch["center_index"].to(device)
            target_mask_sensor = batch["target_mask"].to(device)
            target_weight_sensor = saturation_aware_target_weights(
                x_window,
                center_index,
                target_mask_sensor,
                batch["target_weight"].to(device),
                args.saturated_target_weight_floor,
                batch["target_saturated"].to(device),
            )
            target, target_mask, target_weight = aggregate_withheld_target(
                batch["target"].to(device),
                target_mask_sensor,
                target_weight_sensor,
            )
            with torch.inference_mode():
                routed = run_window(
                    model,
                    x_window,
                    center_index,
                    solver,
                    args.core_size,
                    False,
                    route_mode="learned_budget",
                    route_max_selected_fraction=args.policy_max_selected_fraction,
                    budget_levels=levels,
                    structural_cell_blend=True,
                )
                sigma = hierarchical_uncertainty_components(routed)["total"].clamp_min(
                    1e-6
                )
                complete_gap, _partial_sensor = split_target_domains(
                    target_mask.to(torch.bool), routed["observation_weight"]
                )
                use = target_weight * complete_gap.to(target_weight.dtype)
                error = (routed["prediction"] - target).abs()
                for variable_index in range(2):
                    variable_use = use[:, variable_index]
                    mass = variable_use.sum()
                    if float(mass.detach().cpu()) <= 0.0:
                        continue
                    variable_sigma = sigma[:, variable_index]
                    variable_error = error[:, variable_index]
                    weight_sum[variable_index] += mass.detach().cpu().double()
                    z_sq_sum[variable_index] += (
                        (variable_error / variable_sigma).square() * variable_use
                    ).sum().detach().cpu().double()
                    coverage_68[variable_index] += (
                        (variable_error <= variable_sigma).to(variable_use.dtype)
                        * variable_use
                    ).sum().detach().cpu().double()
                    coverage_95[variable_index] += (
                        (variable_error <= 1.96 * variable_sigma).to(variable_use.dtype)
                        * variable_use
                    ).sum().detach().cpu().double()
                    pixel_count[variable_index] += complete_gap[
                        :, variable_index
                    ].sum().detach().cpu()
            if args.print_every > 0 and batch_index % args.print_every == 0:
                print(
                    f"uncertainty calibration val: {batch_index}/{len(loader)} "
                    f"({100.0 * batch_index / max(len(loader), 1):.1f}%)",
                    flush=True,
                )
    finally:
        dataset.close()

    rows: list[dict[str, object]] = []
    for variable_index, variable in enumerate(("LAI", "FPAR")):
        if variable != args.variable:
            continue
        mass = float(weight_sum[variable_index])
        if mass <= 0.0:
            raise RuntimeError(
                f"Validation uncertainty calibration has no complete-gap {variable} targets."
            )
        raw_scale = math.sqrt(float(z_sq_sum[variable_index]) / mass)
        applied_scale = min(max(raw_scale, 0.50), 1.50)
        rows.append({
            "Variable": variable,
            "RawScaleFactor": raw_scale,
            "AppliedScaleFactor": applied_scale,
            "SourceScenario": "ALL",
            "SourceSplit": "val",
            "TargetDomain": "complete_gap",
            "ApprovedForProduction": 1,
            "N_pixels": int(pixel_count[variable_index]),
            "N_weighted": mass,
            "RawCoverage68": float(coverage_68[variable_index]) / mass,
            "RawCoverage95": float(coverage_95[variable_index]) / mass,
            "Rule": "validation_standardized_rmse_clipped_0p5_1p5",
        })
    return rows


def load_tile_pft_abundance(
    lc_mask_dir: Path,
    pft_codes: list[int],
    dataset_root: Path | None = None,
) -> tuple[dict[int, int], dict[int, float], str]:
    """Aggregate LC_Type_3 vegetation-class abundance over all tile years."""
    summary_paths = {lc_mask_dir.parent / "Summaries" / "landcover_mask_summary.csv"}
    if dataset_root is not None:
        index_path = (
            dataset_root.resolve() / "Module01_Intermediate"
            / "PythonDataset" / "token_index_python.csv"
        )
        if index_path.is_file():
            index = pd.read_csv(index_path, usecols=lambda name: name == "LCSourceFile")
            if "LCSourceFile" in index.columns:
                for value in index["LCSourceFile"].dropna().astype(str).unique():
                    source = Path(value.strip())
                    if source.name:
                        summary_paths.add(
                            source.parent.parent / "Summaries" / "landcover_mask_summary.csv"
                        )
    summary_paths = {path.resolve() for path in summary_paths if path.is_file()}
    if not summary_paths:
        raise FileNotFoundError(
            "No land-cover summary was found for the indexed tile sources."
        )
    counts = {int(code): 0 for code in pft_codes}
    for summary_path in sorted(summary_paths, key=str):
        with summary_path.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                for code in pft_codes:
                    column = f"LC{int(code):03d}_Count"
                    if column not in row:
                        raise KeyError(f"Missing {column} in {summary_path}")
                    counts[int(code)] += int(float(row[column] or 0))
    total = sum(counts.values())
    if total <= 0:
        raise RuntimeError("No LC_Type_3 vegetation pixels found in the indexed summaries.")
    fractions = {code: count / total for code, count in counts.items()}
    return counts, fractions, "|".join(str(path) for path in sorted(summary_paths, key=str))


def token_dominant_pft(
    dataset: WindowedQADataset,
    token_rows: np.ndarray,
    pft_codes: list[int],
) -> dict[int, int]:
    """Map token rows to dominant LC_Type_3 classes using annual tile grids."""
    token_index = dataset.base.token_index.set_index("TokenRow", drop=False)
    selected = token_index.loc[np.asarray(token_rows, dtype=np.int64)]
    lookup: dict[int, int] = {}
    handles: dict[Path, h5py.File] = {}
    try:
        for _, row in selected.iterrows():
            src = dataset.base._resolve_lc_file(row).resolve()
            if src not in handles:
                handles[src] = h5py.File(src, "r")
            handle = handles[src]
            lc_dataset = handle["/LC_Type_3"]
            inline = str(row.get("LCInline", "0")).strip().lower() not in {
                "", "0", "false", "nan"
            }
            if inline:
                r0 = int(row["LocalRowStart"]) - 1
                c0 = int(row["LocalColStart"]) - 1
            else:
                r0 = int(row["GlobalRowStart"]) - 1
                c0 = int(row["GlobalColStart"]) - 1
            size = int(row["PatchSize"])
            patch = read_lc_grid_patch(
                lc_dataset, r0, r0 + size, c0, c0 + size, handle
            )
            counts = np.asarray(
                [np.count_nonzero(patch == int(code)) for code in pft_codes]
            )
            lookup[int(row["TokenRow"])] = (
                int(pft_codes[int(counts.argmax())]) if counts.max(initial=0) > 0 else 0
            )
    finally:
        for handle in handles.values():
            handle.close()
    return lookup


def rotating_pft_quota(
    eligible: list[int], sample_count: int, scenario_index: int
) -> dict[int, int]:
    if not eligible:
        raise ValueError("eligible PFT list must not be empty")
    quota = {code: int(sample_count) // len(eligible) for code in eligible}
    remainder = int(sample_count) % len(eligible)
    start = (int(scenario_index) * max(remainder, 1)) % len(eligible)
    for offset in range(remainder):
        quota[eligible[(start + offset) % len(eligible)]] += 1
    return quota


def balanced_tile_pft_sample(
    group: pd.DataFrame,
    sample_count: int,
    eligible_pfts: list[int],
    seed: int,
    scenario_index: int,
) -> pd.DataFrame:
    """Select equal tile quotas, then rotate represented PFTs within each tile."""
    if "Tile" not in group.columns:
        tiles = ["__single_tile__"]
        working = group.assign(_CalibrationTile="__single_tile__")
    else:
        tiles = sorted(group["Tile"].astype(str).unique())
        working = group.assign(_CalibrationTile=group["Tile"].astype(str))
    tile_quota = rotating_pft_quota(
        list(range(len(tiles))), sample_count, scenario_index
    )
    selected_parts: list[pd.DataFrame] = []
    selected_indices: set[int] = set()
    for tile_index, tile in enumerate(tiles):
        requested = tile_quota[tile_index]
        candidates = working[
            (working["_CalibrationTile"] == tile)
            & working["CalibrationPFT"].isin(eligible_pfts)
        ]
        represented = sorted(candidates["CalibrationPFT"].unique().astype(int))
        if requested <= 0 or not represented:
            continue
        queues = {}
        for code in represented:
            rows = candidates[candidates["CalibrationPFT"] == code]
            queues[code] = list(rows.sample(
                frac=1.0,
                random_state=int(seed + 1009 * scenario_index + 97 * tile_index + 37 * code),
            ).index)
        order = represented[scenario_index % len(represented):] + represented[:scenario_index % len(represented)]
        chosen: list[int] = []
        while len(chosen) < requested and any(queues.values()):
            for code in order:
                if queues[code] and len(chosen) < requested:
                    chosen.append(queues[code].pop())
        selected_indices.update(chosen)
        selected_parts.append(working.loc[chosen])
    selected = pd.concat(selected_parts) if selected_parts else working.iloc[0:0]
    remaining = sample_count - len(selected)
    if remaining > 0:
        candidates = working[
            working["CalibrationPFT"].isin(eligible_pfts)
            & ~working.index.isin(selected_indices)
        ]
        if len(candidates) < remaining:
            raise RuntimeError(
                f"Only {len(selected) + len(candidates)} eligible tile/PFT rows are "
                f"available, fewer than requested {sample_count}."
            )
        selected = pd.concat([
            selected,
            candidates.sample(n=remaining, random_state=int(seed + 7919 * (scenario_index + 1))),
        ])
    # Preserve tile quotas while ensuring every globally eligible PFT that is
    # represented in this scenario appears at least once when capacity allows.
    if sample_count >= len(eligible_pfts) and not selected.empty:
        represented = set(selected["CalibrationPFT"].astype(int))
        for code in [value for value in eligible_pfts if value not in represented]:
            candidates = working[
                (working["CalibrationPFT"] == code)
                & ~working.index.isin(selected.index)
            ]
            if candidates.empty:
                continue
            replacement = candidates.sample(
                n=1, random_state=int(seed + 1543 * (scenario_index + 1) + code)
            ).iloc[0]
            tile = replacement["_CalibrationTile"]
            same_tile = selected[
                selected["_CalibrationTile"] == tile
            ] if "_CalibrationTile" in selected.columns else selected
            counts = selected["CalibrationPFT"].value_counts()
            removable = same_tile[
                same_tile["CalibrationPFT"].map(counts).fillna(0) > 1
            ]
            if removable.empty:
                continue
            selected = selected.drop(index=removable.index[-1])
            selected = pd.concat([selected, replacement.to_frame().T])
            represented.add(code)
    return selected.drop(columns="_CalibrationTile", errors="ignore")


def pft_stratified_manifest(
    dataset: WindowedQADataset,
    args: argparse.Namespace,
    sample_count: int,
    seed: int,
) -> pd.DataFrame:
    manifest = dataset.base.mask_manifest.copy()
    pft_codes = parse_ints(args.lc_codes)
    _counts, fractions, _source = load_tile_pft_abundance(
        args.lc_mask_dir, pft_codes, args.root
    )
    eligible = [
        code for code in pft_codes
        if fractions.get(code, 0.0) >= args.min_pft_calibration_fraction
    ]
    token_rows = manifest["TokenRow"].astype(np.int64).unique()
    lookup = token_dominant_pft(dataset, token_rows, pft_codes)
    manifest["CalibrationPFT"] = manifest["TokenRow"].map(lookup).fillna(0).astype(int)
    available = set(manifest["CalibrationPFT"].unique().tolist())
    eligible = [code for code in eligible if code in available]
    if not eligible:
        raise RuntimeError("No eligible PFT is represented in the validation manifest.")
    selected_parts = []
    for scenario_index, (_scenario, group) in enumerate(
        manifest.groupby("Scenario", sort=False)
    ):
        chosen = balanced_tile_pft_sample(
            group, sample_count, eligible, seed, scenario_index
        )
        selected_parts.append(chosen)
    selected = pd.concat(selected_parts, ignore_index=True)
    print(
        f"PFT-stratified {str(selected['Split'].iloc[0])} patches: "
        + ", ".join(
            f"PFT{code}={int((selected['CalibrationPFT'] == code).sum())}"
            for code in eligible
        ),
        flush=True,
    )
    if "Tile" in selected.columns:
        print(
            "Tile-balanced patches: " + ", ".join(
                f"{tile}={int(count)}"
                for tile, count in selected["Tile"].astype(str).value_counts().sort_index().items()
            ),
            flush=True,
        )
    return selected.reset_index(drop=True)


def to_core_vectors(x: torch.Tensor, core: int) -> torch.Tensor:
    b, variables, time, height, width = x.shape
    nr, nc = height // core, width // core
    return x.reshape(b, variables, time, nr, core, nc, core).permute(
        0, 1, 3, 5, 2, 4, 6
    ).reshape(b, variables, nr * nc, -1)


def candidate_oracle_labels(
    predictions: dict[int, torch.Tensor],
    target: torch.Tensor,
    target_mask: torch.Tensor,
    target_weight: torch.Tensor,
    levels: torch.Tensor,
    tolerance: float,
    minimum_solver_gain: float,
    core: int,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor,
]:
    scale = torch.tensor([7.0, 1.0], dtype=target.dtype, device=target.device).view(1, 2, 1, 1, 1)
    use = to_core_vectors(target_weight * target_mask.to(target_weight.dtype), core)
    denom = use.sum(dim=-1)
    errors = []
    high_use_mask = target_mask.to(torch.bool).clone()
    high_use_mask[:, 0] &= target[:, 0] >= 5.0
    high_use_mask[:, 1] = False
    high_use = to_core_vectors(
        target_weight * high_use_mask.to(target_weight.dtype), core
    )
    high_denom = high_use.sum(dim=-1)
    high_errors = []
    high_biases = []
    for budget in levels.tolist():
        error = to_core_vectors((predictions[int(budget)] - target) / scale, core)
        errors.append(((error.square() * use).sum(dim=-1) / denom.clamp_min(1e-8)).sqrt())
        high_errors.append(
            ((error.square() * high_use).sum(dim=-1) / high_denom.clamp_min(1e-8)).sqrt()
        )
        high_biases.append(
            (error * high_use).sum(dim=-1) / high_denom.clamp_min(1e-8)
        )
    error_by_budget = torch.stack(errors, dim=-1)
    prior_error = error_by_budget[..., 0]
    positive_error = error_by_budget[..., 1:]
    best_positive_error = positive_error.min(dim=-1).values
    solver_gain = prior_error - best_positive_error

    # Budget zero is the QA-projected data-consistent prior. A positive solver
    # budget is justified only when it improves that prior by a meaningful
    # validation margin. Among equivalent positive candidates, use the
    # smallest budget to minimize computation and smoothing exposure.
    sufficient = (
        positive_error <= best_positive_error.unsqueeze(-1) + float(tolerance)
    ) & (
        positive_error
        < prior_error.unsqueeze(-1) - float(minimum_solver_gain)
    )
    positive_label = sufficient.to(torch.int64).argmax(dim=-1) + 1
    labels = torch.where(
        solver_gain > float(minimum_solver_gain),
        positive_label,
        torch.zeros_like(positive_label),
    )
    labels = torch.where(denom > 0, labels, torch.full_like(labels, -100))
    return (
        labels,
        error_by_budget,
        solver_gain,
        denom,
        torch.stack(high_errors, dim=-1),
        torch.stack(high_biases, dim=-1),
        high_denom,
    )


def build_oracle_cache(
    args: argparse.Namespace,
    model: QALLMAdaptiveController,
    solver: SolverConfig,
    levels: torch.Tensor,
    split: str,
    sample_count: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    cache_dir = (
        args.validation_oracle_cache_dir
        if split == "val" else args.oracle_cache_dir
    )
    cache_path = cache_dir / f"budget_oracle_candidates_{split}.pt"
    peak_safe_oracle = bool(
        getattr(model.config, "solver_peak_safe_routing", False)
    )
    if peak_safe_oracle:
        sampling_policy = (
            f"pft-stratified-{split}-complete-gap-cell-selective-prior-"
            "v10-candidate-relative-peak-safe"
            if bool(args.pft_stratified_sampling)
            else "scenario-random-complete-gap-cell-selective-prior-"
            "v10-candidate-relative-peak-safe"
        )
    else:
        # Keep the non-peak-safe cache identity distinct from the active
        # candidate-relative peak-safe oracle.
        sampling_policy = (
            f"pft-stratified-{split}-complete-gap-cell-selective-prior-"
            "v9-rotated-high-safe"
            if bool(args.pft_stratified_sampling)
            else "scenario-random-complete-gap-cell-selective-prior-v9-high-safe"
        )
    oracle_definition = (
        "complete-gap-structural-cell-best-positive-budget-v10-"
        "candidate-relative-peak-safe"
        if peak_safe_oracle else
        "complete-gap-structural-cell-best-positive-budget-v9-rotated-high-safe"
    )
    cache_signature, cache_signature_payload = oracle_cache_signature(
        args, split, sample_count, levels, sampling_policy
    )
    if cache_path.is_file() and not args.overwrite_oracle_cache:
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        cache_compatible = (
            cached.get("sampling_policy") == sampling_policy
            and cached.get("cache_signature") == cache_signature
            and abs(
                float(cached.get("minimum_solver_gain", float("nan")))
                - float(args.minimum_solver_gain)
            ) <= 1e-12
            and abs(
                float(cached.get("normalized_error_tolerance", float("nan")))
                - float(args.normalized_error_tolerance)
            ) <= 1e-12
            and "errors_by_budget" in cached
            and "best_solver_gain" in cached
            and "target_mass" in cached
            and "high_lai_errors_by_budget" in cached
            and "high_lai_bias_by_budget" in cached
            and "high_lai_target_mass" in cached
            and cached.get("oracle_definition") == oracle_definition
        )
        if cache_compatible:
            print(f"Reusing oracle cache: {cache_path}", flush=True)
            return cached
        print(
            f"Rebuilding stale {split} oracle cache for {sampling_policy}.",
            flush=True,
        )
    dataset = build_dataset(args, split, sample_count)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    feature_rows, label_rows, error_rows, gain_rows, target_mass_rows = [], [], [], [], []
    high_error_rows, high_bias_rows, high_mass_rows = [], [], []
    scenarios: list[str] = []
    skipped_solver_samples = 0
    model.eval()
    try:
        for batch_index, batch in enumerate(loader, start=1):
            x_window = batch["x_window"].to(device)
            center_index = batch["center_index"].to(device)
            target_mask_sensor = batch["target_mask"].to(device)
            target_weight_sensor = saturation_aware_target_weights(
                x_window,
                center_index,
                target_mask_sensor,
                batch["target_weight"].to(device),
                args.saturated_target_weight_floor,
                batch["target_saturated"].to(device),
            )
            target, target_mask, target_weight = aggregate_withheld_target(
                batch["target"].to(device),
                target_mask_sensor,
                target_weight_sensor,
            )
            with torch.inference_mode():
                prepared_frontend = prepare_window_frontend(
                    model, x_window, return_budget_features=True
                )
                structural_cells = extract_structural_gap_mask(
                    x_window,
                    center_index,
                    min_run_steps=args.structural_min_run_steps,
                    max_patch_missing=args.structural_max_patch_missing,
                    max_support_deficit=args.structural_max_support_deficit,
                )
                has_structural_target = bool(
                    (structural_cells & target_mask.to(torch.bool)).any()
                )
                structural_target_mask = target_mask.to(torch.bool) & structural_cells
                if not has_structural_target:
                    skipped_solver_samples += 1
                    full = run_window(
                        model,
                        x_window,
                        center_index,
                        solver,
                        args.core_size,
                        False,
                        route_mode="threshold",
                        route_thresholds=(2.0, 2.0),
                        structural_cell_blend=True,
                        budget_levels=levels,
                        return_budget_features=True,
                        prepared_frontend=prepared_frontend,
                    )
                    predictions = {
                        int(budget): full["data_consistent_prior"]
                        for budget in levels.tolist()
                    }
                else:
                    full = run_window(
                        model,
                        x_window,
                        center_index,
                        solver,
                        args.core_size,
                        False,
                        budget_levels=levels,
                        force_full_budget=True,
                        structural_cell_blend=True,
                        return_budget_features=True,
                        prepared_frontend=prepared_frontend,
                    )
                    predictions = {
                        int(levels[0]): full["data_consistent_prior"],
                        int(levels[-1]): full["prediction"],
                    }
                    for budget in levels[1:-1].tolist():
                        l1_budget = max(
                            1,
                            int(math.ceil(
                                float(budget) / max(float(solver.max_iterations), 1.0)
                                * float(solver.l1_max_iterations)
                            )),
                        )
                        candidate_solver = replace(
                            solver,
                            min_iterations=int(budget),
                            max_iterations=int(budget),
                            l1_min_iterations=l1_budget,
                            l1_max_iterations=l1_budget,
                        )
                        candidate = run_window(
                            model,
                            x_window,
                            center_index,
                            candidate_solver,
                            args.core_size,
                            False,
                            structural_cell_blend=True,
                            prepared_frontend=prepared_frontend,
                        )
                        predictions[int(budget)] = candidate["prediction"]
                complete_gap_mask, _partial_sensor_mask = split_target_domains(
                    target_mask.to(torch.bool), full["observation_weight"]
                )
                if bool((structural_target_mask & ~complete_gap_mask).any()):
                    raise RuntimeError(
                        "Structural solver targets must be a subset of complete gaps."
                    )
                structural_target_mask &= complete_gap_mask
                (
                    labels, candidate_errors, solver_gain, target_mass,
                    high_errors, high_biases, high_mass,
                ) = candidate_oracle_labels(
                    predictions,
                    target,
                    structural_target_mask,
                    target_weight,
                    levels,
                    args.normalized_error_tolerance,
                    args.minimum_solver_gain,
                    args.core_size,
                )
            features = full["budget_features_core"].detach().cpu()
            feature_rows.append(features.reshape(-1, features.shape[-1]))
            label_rows.append(labels.permute(0, 2, 1).reshape(-1, 2).detach().cpu())
            error_rows.append(
                candidate_errors.permute(0, 2, 1, 3)
                .reshape(-1, 2, levels.numel()).detach().cpu()
            )
            gain_rows.append(
                solver_gain.permute(0, 2, 1).reshape(-1, 2).detach().cpu()
            )
            target_mass_rows.append(
                target_mass.permute(0, 2, 1).reshape(-1, 2).detach().cpu()
            )
            high_error_rows.append(
                high_errors.permute(0, 2, 1, 3)
                .reshape(-1, 2, levels.numel()).detach().cpu()
            )
            high_bias_rows.append(
                high_biases.permute(0, 2, 1, 3)
                .reshape(-1, 2, levels.numel()).detach().cpu()
            )
            high_mass_rows.append(
                high_mass.permute(0, 2, 1).reshape(-1, 2).detach().cpu()
            )
            scenarios.extend([str(batch["scenario"][0])] * features.shape[1])
            if args.print_every > 0 and batch_index % args.print_every == 0:
                print(
                    f"budget oracle {split}: {batch_index}/{len(loader)} "
                    f"({100.0 * batch_index / max(len(loader), 1):.1f}%)",
                    flush=True,
                )
    finally:
        dataset.close()
    cache = {
        "features": torch.cat(feature_rows),
        "labels": torch.cat(label_rows),
        "errors_by_budget": torch.cat(error_rows),
        "best_solver_gain": torch.cat(gain_rows),
        "target_mass": torch.cat(target_mass_rows),
        "high_lai_errors_by_budget": torch.cat(high_error_rows),
        "high_lai_bias_by_budget": torch.cat(high_bias_rows),
        "high_lai_target_mass": torch.cat(high_mass_rows),
        "scenarios": scenarios,
        "levels": levels.cpu(),
        "normalized_error_tolerance": float(args.normalized_error_tolerance),
        "minimum_solver_gain": float(args.minimum_solver_gain),
        "oracle_definition": oracle_definition,
        "sampling_policy": sampling_policy,
        "cache_signature": cache_signature,
        "cache_signature_payload": cache_signature_payload,
        "total_samples": len(dataset),
        "solver_skipped_samples": skipped_solver_samples,
    }
    torch.save(cache, cache_path)
    print(
        f"budget oracle {split}: skipped positive ST/L1 candidates for "
        f"{skipped_solver_samples}/{len(dataset)} samples without sustained targets.",
        flush=True,
    )
    return cache


def refresh_budget_context_cache(
    args: argparse.Namespace,
    cache: dict,
    split: str,
    sample_count: int,
) -> dict:
    version = "qa-usable-gap-context-cell-selective-v4"
    if (
        cache.get("context_aggregation") == version
        and "core_pft" in cache
        and "structural_eligible" in cache
    ):
        print(f"Reusing refreshed {split} budget contexts.", flush=True)
        return cache
    dataset = build_dataset(args, split, sample_count)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    context_rows = []
    structural_rows = []
    pft_rows = []
    pft_codes = parse_ints(args.lc_codes)
    refreshed_scenarios: list[str] = []
    try:
        for batch_index, batch in enumerate(loader, start=1):
            x_window = batch["x_window"]
            center_index = batch["center_index"]
            contexts = extract_budget_context(
                x_window, center_index, args.core_size
            )
            structural_cells = extract_structural_gap_mask(
                x_window,
                center_index,
                min_run_steps=args.structural_min_run_steps,
                max_patch_missing=args.structural_max_patch_missing,
                max_support_deficit=args.structural_max_support_deficit,
            )
            b, variables, time, height, width = structural_cells.shape
            nr, nc = height // args.core_size, width // args.core_size
            structural = structural_cells.reshape(
                b, variables, time, nr, args.core_size, nc, args.core_size
            ).permute(0, 1, 3, 5, 2, 4, 6).reshape(
                b, variables, nr * nc, -1
            ).any(dim=-1)
            context_rows.append(
                contexts.permute(0, 2, 1, 3).reshape(-1, 2, contexts.shape[-1]).cpu()
            )
            structural_rows.append(
                structural.permute(0, 2, 1).reshape(-1, 2).cpu()
            )
            dominant_pft = extract_budget_pft(
                x_window, center_index, args.core_size, pft_codes
            )
            pft_rows.append(dominant_pft.reshape(-1).cpu())
            refreshed_scenarios.extend([str(batch["scenario"][0])] * contexts.shape[2])
            if args.print_every > 0 and batch_index % args.print_every == 0:
                print(
                    f"budget context refresh {split}: {batch_index}/{len(loader)} "
                    f"({100.0 * batch_index / max(len(loader), 1):.1f}%)",
                    flush=True,
                )
    finally:
        dataset.close()
    refreshed = torch.cat(context_rows)
    refreshed_structural = torch.cat(structural_rows)
    refreshed_pft = torch.cat(pft_rows)
    if refreshed.shape[0] != cache["labels"].shape[0]:
        raise RuntimeError(
            f"Refreshed {split} context count {refreshed.shape[0]} does not match "
            f"cached labels {cache['labels'].shape[0]}."
        )
    if list(cache.get("scenarios", [])) != refreshed_scenarios:
        raise RuntimeError(f"Refreshed {split} sample order does not match the oracle cache.")
    if refreshed_pft.shape[0] != cache["labels"].shape[0]:
        raise RuntimeError(
            f"Refreshed {split} PFT count {refreshed_pft.shape[0]} does not match "
            f"cached labels {cache['labels'].shape[0]}."
        )
    cache = dict(cache)
    cache["contexts"] = refreshed
    cache["structural_eligible"] = refreshed_structural
    cache["core_pft"] = refreshed_pft
    cache["pft_codebook"] = pft_codes
    cache["context_aggregation"] = version
    cache_dir = (
        args.validation_oracle_cache_dir
        if split == "val" else args.oracle_cache_dir
    )
    torch.save(cache, cache_dir / f"budget_oracle_candidates_{split}.pt")
    return cache


def class_weights(
    labels: torch.Tensor,
    classes: int,
    power: float,
    cap: float,
) -> torch.Tensor:
    valid = labels[labels >= 0]
    counts = torch.bincount(valid, minlength=classes).float().clamp_min(1.0)
    weights = (counts.sum() / counts).pow(float(power))
    if cap > 0:
        weights = weights.clamp_max(float(cap))
    return weights / weights.mean()


def decode_budget_logits(
    logits: torch.Tensor,
    decision_quantiles: torch.Tensor | None = None,
    need_thresholds: torch.Tensor | None = None,
) -> torch.Tensor:
    if decision_quantiles is None or need_thresholds is None:
        return logits.argmax(dim=-1)
    probabilities = torch.softmax(logits, dim=-1)
    quantiles = decision_quantiles.to(device=logits.device, dtype=probabilities.dtype)
    thresholds = need_thresholds.to(device=logits.device, dtype=probabilities.dtype)
    if quantiles.ndim == 1:
        quantiles = quantiles.view(1, 2, 1)
        thresholds = thresholds.view(1, 2)
    elif quantiles.ndim == 2 and quantiles.shape == logits.shape[:2]:
        quantiles = quantiles.unsqueeze(-1)
        if thresholds.shape != logits.shape[:2]:
            raise ValueError("need_thresholds must match the resolved quantile rows.")
    else:
        raise ValueError(
            "decision_quantiles must have shape [2] or [rows,2], got "
            f"{tuple(quantiles.shape)}."
        )
    need_probability = 1.0 - probabilities[..., 0]
    conditional = probabilities[..., 1:] / need_probability.unsqueeze(-1).clamp_min(1e-8)
    positive_class = (conditional.cumsum(dim=-1) < quantiles).sum(dim=-1)
    positive_class = positive_class.clamp_max(logits.shape[-1] - 2) + 1
    return torch.where(need_probability >= thresholds, positive_class, 0)


def calibrate_pft_quantiles(
    logits: torch.Tensor,
    labels: torch.Tensor,
    errors_by_budget: torch.Tensor,
    target_mass: torch.Tensor,
    high_lai_errors_by_budget: torch.Tensor,
    high_lai_bias_by_budget: torch.Tensor,
    high_lai_target_mass: torch.Tensor,
    structural_eligible: torch.Tensor,
    core_pft: torch.Tensor,
    pft_codes: list[int],
    levels: torch.Tensor,
    scenario_names: list[str],
    min_fraction: float,
    min_calibration_cores: int,
    tile_pft_counts: dict[int, int],
    tile_pft_fractions: dict[int, float],
    grid_min_quantile: float,
    min_relative_nrmse_gain: float,
    max_relative_nrmse_increase: float,
    max_scenario_relative_nrmse_increase: float,
    max_selected_fraction: float,
    require_spatial_block_lai_improvement: bool,
    max_high_lai_relative_rmse_increase: float,
    max_high_lai_absolute_bias_increase: float,
    active_variable: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, list[dict]]:
    """Choose PFT policies by validation error relative to the data-consistent prior."""
    probabilities = torch.softmax(logits, dim=-1)
    pft_quantiles = torch.full(
        (2, len(pft_codes)),
        float(grid_min_quantile),
        dtype=torch.float32,
        device=logits.device,
    )
    pft_need_thresholds = torch.full(
        (2, len(pft_codes)),
        1.01,
        dtype=torch.float32,
        device=logits.device,
    )
    rows: list[dict] = []
    scenario_tensor = list(scenario_names)
    tile_vegetation_pixels = sum(tile_pft_counts.values())

    def policy_metrics(
        prediction: torch.Tensor,
        truth: torch.Tensor,
        candidate_errors: torch.Tensor,
        candidate_mass: torch.Tensor,
        high_candidate_errors: torch.Tensor,
        high_candidate_biases: torch.Tensor,
        high_candidate_mass: torch.Tensor,
        valid_names: list[str],
    ) -> dict[str, float]:
        selected_error = candidate_errors.gather(-1, prediction[:, None]).squeeze(-1)
        prior_error = candidate_errors[:, 0]
        selected_high_error = high_candidate_errors.gather(
            -1, prediction[:, None]
        ).squeeze(-1)
        prior_high_error = high_candidate_errors[:, 0]
        selected_high_bias = high_candidate_biases.gather(
            -1, prediction[:, None]
        ).squeeze(-1)
        prior_high_bias = high_candidate_biases[:, 0]

        def aggregate_rmse(error: torch.Tensor, use: torch.Tensor | None = None) -> float:
            mass = candidate_mass if use is None else candidate_mass[use]
            values = error if use is None else error[use]
            mse = (values.square() * mass).sum() / mass.sum().clamp_min(1e-8)
            return float(torch.sqrt(mse).detach().cpu())

        mean_prior = aggregate_rmse(prior_error)
        mean_selected = aggregate_rmse(selected_error)
        has_high = bool((high_candidate_mass > 0).any())
        if has_high:
            high_mass_sum = high_candidate_mass.sum().clamp_min(1e-8)
            prior_high_rmse = float(torch.sqrt(
                (prior_high_error.square() * high_candidate_mass).sum() / high_mass_sum
            ).detach().cpu())
            selected_high_rmse = float(torch.sqrt(
                (selected_high_error.square() * high_candidate_mass).sum() / high_mass_sum
            ).detach().cpu())
            prior_high_bias_value = float(
                (prior_high_bias * high_candidate_mass).sum().detach().cpu()
                / high_mass_sum.detach().cpu()
            )
            selected_high_bias_value = float(
                (selected_high_bias * high_candidate_mass).sum().detach().cpu()
                / high_mass_sum.detach().cpu()
            )
            high_relative_change = (
                selected_high_rmse / max(prior_high_rmse, 1e-12) - 1.0
            )
            high_abs_bias_increase = (
                abs(selected_high_bias_value) - abs(prior_high_bias_value)
            )
        else:
            prior_high_rmse = selected_high_rmse = float("nan")
            prior_high_bias_value = selected_high_bias_value = float("nan")
            high_relative_change = high_abs_bias_increase = float("nan")
        scenario_changes: list[float] = []
        spatial_change = float("nan")
        for scenario in sorted(set(valid_names)):
            use = torch.tensor(
                [name == scenario for name in valid_names],
                dtype=torch.bool,
                device=truth.device,
            )
            if not bool(use.any()):
                continue
            scenario_prior = aggregate_rmse(prior_error, use)
            scenario_selected = aggregate_rmse(selected_error, use)
            change = scenario_selected / max(scenario_prior, 1e-12) - 1.0
            scenario_changes.append(change)
            if scenario == "spatial_block":
                spatial_change = change
        return {
            "MeanPriorNRMSE": mean_prior,
            "MeanSelectedNRMSE": mean_selected,
            "RelativeNRMSEChangeVsPrior": mean_selected / max(mean_prior, 1e-12) - 1.0,
            "MaxScenarioRelativeNRMSEChangeVsPrior": max(scenario_changes, default=0.0),
            "SpatialBlockRelativeNRMSEChangeVsPrior": spatial_change,
            "SelectedFraction": float((prediction > 0).float().mean().detach().cpu()),
            "MeanOracleIterations": float(
                (levels[truth].float() * candidate_mass).sum().detach().cpu()
                / candidate_mass.sum().clamp_min(1e-8).detach().cpu()
            ),
            "MeanPredictedIterations": float(
                (levels[prediction].float() * candidate_mass).sum().detach().cpu()
                / candidate_mass.sum().clamp_min(1e-8).detach().cpu()
            ),
            "NeedSolverMissRate": float(
                ((prediction == 0) & (truth > 0)).float().mean().detach().cpu()
            ),
            "HighLAIPriorNRMSE": prior_high_rmse,
            "HighLAISelectedNRMSE": selected_high_rmse,
            "HighLAIRelativeNRMSEChangeVsPrior": high_relative_change,
            "HighLAIPriorNormalizedBias": prior_high_bias_value,
            "HighLAISelectedNormalizedBias": selected_high_bias_value,
            "HighLAIAbsoluteBiasIncreaseVsPrior": high_abs_bias_increase,
        }

    for vi, variable in enumerate(("LAI", "FPAR")):
        if active_variable is not None and variable != active_variable:
            continue
        for pi, pft_code in enumerate(pft_codes):
            valid = (labels[:, vi] >= 0) & (core_pft == int(pft_code))
            count = int(valid.sum().item())
            structural_fraction = (
                float(structural_eligible[:, vi][valid].float().mean().item())
                if count > 0 else float("nan")
            )
            tile_fraction = float(tile_pft_fractions.get(int(pft_code), 0.0))
            valid_names = [
                scenario_tensor[index]
                for index, is_valid in enumerate(valid.tolist())
                if is_valid
            ]
            selected_q = float(grid_min_quantile)
            selected_need_threshold = 1.01
            decision_reason = "independently-calibrated"
            selected_metrics: dict[str, float] | None = None
            if tile_fraction < float(min_fraction):
                decision_reason = "rare-biome-prior-only"
            elif count < int(min_calibration_cores):
                decision_reason = "insufficient-calibration-cores-prior-only"
            else:
                truth = labels[:, vi][valid]
                probs = probabilities[:, vi][valid]
                candidate_errors = errors_by_budget[:, vi][valid]
                candidate_mass = target_mass[:, vi][valid]
                high_candidate_errors = high_lai_errors_by_budget[:, vi][valid]
                high_candidate_biases = high_lai_bias_by_budget[:, vi][valid]
                high_candidate_mass = high_lai_target_mass[:, vi][valid]
                eligible = structural_eligible[:, vi][valid]
                need_probability = 1.0 - probs[:, 0]
                conditional = probs[:, 1:] / need_probability[:, None].clamp_min(1e-8)
                quantile_grid = torch.arange(
                    float(grid_min_quantile), 1.001, 0.05, device=logits.device
                )
                threshold_grid = torch.cat((
                    torch.arange(0.05, 1.001, 0.05, device=logits.device),
                    torch.tensor([1.01], device=logits.device),
                ))
                candidates: list[tuple[float, float, dict[str, float]]] = []
                for need_threshold in threshold_grid:
                    for quantile in quantile_grid:
                        positive = (conditional.cumsum(dim=-1) < quantile).sum(dim=-1)
                        positive = positive.clamp_max(levels.numel() - 2) + 1
                        prediction = torch.where(
                            need_probability >= need_threshold, positive, 0
                        )
                        prediction = torch.where(
                            eligible, prediction, torch.zeros_like(prediction)
                        )
                        # Production applies a score-ranked top-k cap after the
                        # threshold decision. Calibration must evaluate the same
                        # operation; rejecting the uncapped candidate fraction
                        # can incorrectly collapse an otherwise useful PFT to
                        # prior-only routing.
                        max_selected = int(
                            math.floor(float(max_selected_fraction) * prediction.numel())
                        )
                        selected_indices = torch.nonzero(
                            prediction > 0, as_tuple=False
                        ).flatten()
                        if selected_indices.numel() > max_selected:
                            capped = torch.zeros_like(prediction)
                            if max_selected > 0:
                                chosen = selected_indices[
                                    torch.topk(
                                        need_probability[selected_indices],
                                        k=max_selected,
                                        largest=True,
                                    ).indices
                                ]
                                capped[chosen] = prediction[chosen]
                            prediction = capped
                        metrics = policy_metrics(
                            prediction=prediction,
                            truth=truth,
                            candidate_errors=candidate_errors,
                            candidate_mass=candidate_mass,
                            high_candidate_errors=high_candidate_errors,
                            high_candidate_biases=high_candidate_biases,
                            high_candidate_mass=high_candidate_mass,
                            valid_names=valid_names,
                        )
                        candidates.append((
                            float(need_threshold.detach().cpu()),
                            float(quantile.detach().cpu()),
                            metrics,
                        ))
                feasible = [
                    item for item in candidates
                    if item[2]["RelativeNRMSEChangeVsPrior"]
                    <= min(
                        float(max_relative_nrmse_increase),
                        -float(min_relative_nrmse_gain),
                    )
                    and item[2]["MaxScenarioRelativeNRMSEChangeVsPrior"]
                    <= float(max_scenario_relative_nrmse_increase)
                    and item[2]["SelectedFraction"] <= float(max_selected_fraction)
                    and (
                        variable != "LAI"
                        or not math.isfinite(item[2]["HighLAIRelativeNRMSEChangeVsPrior"])
                        or item[2]["HighLAIRelativeNRMSEChangeVsPrior"]
                        <= float(max_high_lai_relative_rmse_increase)
                    )
                    and (
                        variable != "LAI"
                        or not math.isfinite(item[2]["HighLAIAbsoluteBiasIncreaseVsPrior"])
                        or item[2]["HighLAIAbsoluteBiasIncreaseVsPrior"]
                        <= float(max_high_lai_absolute_bias_increase)
                    )
                    and (
                        not require_spatial_block_lai_improvement
                        or
                        variable != "LAI"
                        or not math.isfinite(
                            item[2]["SpatialBlockRelativeNRMSEChangeVsPrior"]
                        )
                        or item[2]["SpatialBlockRelativeNRMSEChangeVsPrior"] <= 0.0
                    )
                ]
                if feasible:
                    selected_need_threshold, selected_q, selected_metrics = min(
                        feasible,
                        key=lambda item: (
                            item[2]["MeanSelectedNRMSE"],
                            item[2]["MeanPredictedIterations"],
                            -item[0],
                            item[1],
                        ),
                    )
                else:
                    decision_reason = "no-feasible-policy-prior-only"
            if selected_metrics is None:
                if count > 0:
                    truth = labels[:, vi][valid]
                    candidate_errors = errors_by_budget[:, vi][valid]
                    candidate_mass = target_mass[:, vi][valid]
                    high_candidate_errors = high_lai_errors_by_budget[:, vi][valid]
                    high_candidate_biases = high_lai_bias_by_budget[:, vi][valid]
                    high_candidate_mass = high_lai_target_mass[:, vi][valid]
                    selected_metrics = policy_metrics(
                        prediction=torch.zeros_like(truth),
                        truth=truth,
                        candidate_errors=candidate_errors,
                        candidate_mass=candidate_mass,
                        high_candidate_errors=high_candidate_errors,
                        high_candidate_biases=high_candidate_biases,
                        high_candidate_mass=high_candidate_mass,
                        valid_names=valid_names,
                    )
                else:
                    selected_metrics = {
                        key: float("nan") for key in (
                            "MeanPriorNRMSE", "MeanSelectedNRMSE",
                            "RelativeNRMSEChangeVsPrior",
                            "MaxScenarioRelativeNRMSEChangeVsPrior",
                            "SpatialBlockRelativeNRMSEChangeVsPrior",
                            "SelectedFraction", "MeanOracleIterations",
                            "MeanPredictedIterations", "NeedSolverMissRate",
                            "HighLAIPriorNRMSE", "HighLAISelectedNRMSE",
                            "HighLAIRelativeNRMSEChangeVsPrior",
                            "HighLAIPriorNormalizedBias",
                            "HighLAISelectedNormalizedBias",
                            "HighLAIAbsoluteBiasIncreaseVsPrior",
                        )
                    }
            pft_quantiles[vi, pi] = selected_q
            pft_need_thresholds[vi, pi] = selected_need_threshold
            rows.append({
                "Variable": variable,
                "PFTCode": int(pft_code),
                "TileBiomePixelCount": int(tile_pft_counts.get(int(pft_code), 0)),
                "TileVegetationPixelCount": int(tile_vegetation_pixels),
                "TileBiomeFraction": tile_fraction,
                "MinimumTileBiomeFraction": float(min_fraction),
                "MinimumCalibrationCoreCount": int(min_calibration_cores),
                "MinimumRelativeNRMSEGain": float(min_relative_nrmse_gain),
                "ValidCalibrationCoreCount": count,
                "StructuralEligibleFraction": structural_fraction,
                "NeedSolverThreshold": selected_need_threshold,
                "ConditionalBudgetQuantile": selected_q,
                "Quantile": selected_q,
                "UsesPriorOnly": int(selected_need_threshold > 1.0),
                "UsesConservativeFullBudget": 0,
                "DecisionReason": decision_reason,
                **selected_metrics,
            })
    return pft_need_thresholds, pft_quantiles, rows


def resolve_pft_quantiles(
    unknown_pft_quantiles: torch.Tensor,
    pft_quantiles: torch.Tensor,
    core_pft: torch.Tensor,
    pft_codes: list[int],
) -> torch.Tensor:
    resolved = unknown_pft_quantiles.view(1, 2).expand(core_pft.numel(), 2).clone()
    for pi, pft_code in enumerate(pft_codes):
        use = core_pft == int(pft_code)
        resolved[use] = pft_quantiles[:, pi]
    return resolved


def resolve_pft_need_thresholds(
    unknown_thresholds: torch.Tensor,
    pft_thresholds: torch.Tensor,
    core_pft: torch.Tensor,
    pft_codes: list[int],
) -> torch.Tensor:
    resolved = unknown_thresholds.view(1, 2).expand(core_pft.numel(), 2).clone()
    for pi, pft_code in enumerate(pft_codes):
        use = core_pft == int(pft_code)
        resolved[use] = pft_thresholds[:, pi]
    return resolved


def head_loss(
    model: QALLMAdaptiveController,
    features: torch.Tensor,
    contexts: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
    levels: torch.Tensor,
    regression_weight: float,
    underbudget_weight: float,
    need_solver_weight: float,
    zero_overbudget_weight: float,
    decision_quantiles: torch.Tensor | None = None,
    need_thresholds: torch.Tensor | None = None,
    structural_eligible: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    logits = model.budget_logits_from_core_features(features, contexts)
    valid = labels >= 0
    if not bool(valid.any()):
        raise ValueError('No valid structural-gap oracle targets; budget loss cannot be fitted.')
    prediction = decode_budget_logits(logits, decision_quantiles, need_thresholds)
    if structural_eligible is not None:
        prediction = torch.where(
            structural_eligible.to(device=prediction.device, dtype=torch.bool),
            prediction,
            torch.zeros_like(prediction),
        )
    safe_labels = labels.clamp_min(0)
    probabilities = torch.softmax(logits, dim=-1)
    need_target = (safe_labels > 0).to(probabilities.dtype)
    need_probability = 1.0 - probabilities[..., 0]
    need_logit = torch.logsumexp(logits[..., 1:], dim=-1) - logits[..., 0]
    need_losses = []
    for vi in range(2):
        use = valid[:, vi]
        if bool(use.any()):
            positive_weight = (
                weights[vi, 1:].mean() / weights[vi, 0].clamp_min(1e-6)
            ).clamp(1.0, 4.0)
            need_losses.append(F.binary_cross_entropy_with_logits(
                need_logit[:, vi][use],
                need_target[:, vi][use],
                pos_weight=positive_weight,
            ))
    need_loss = (
        torch.stack(need_losses).mean()
        if need_losses else logits.sum() * 0.0
    )
    conditional_losses = []
    for vi in range(2):
        use = valid[:, vi] & (safe_labels[:, vi] > 0)
        if bool(use.any()):
            conditional_losses.append(F.cross_entropy(
                logits[:, vi, 1:][use],
                safe_labels[:, vi][use] - 1,
                weight=weights[vi, 1:],
            ))
    conditional_loss = (
        torch.stack(conditional_losses).mean()
        if conditional_losses else logits.sum() * 0.0
    )
    level_values = levels.to(logits.dtype).view(1, 1, -1)
    expected_budget = (probabilities * level_values).sum(dim=-1)
    true_budget = levels[safe_labels].to(logits.dtype)
    budget_scale = max(float(levels[-1]), 1.0)
    regression_loss = F.smooth_l1_loss(
        expected_budget[valid] / budget_scale,
        true_budget[valid] / budget_scale,
    )
    underbudget_loss = (
        torch.relu(true_budget[valid] - expected_budget[valid]) / budget_scale
    ).mean()
    no_solver = valid & (safe_labels == 0)
    zero_overbudget_loss = (
        (expected_budget[no_solver] / budget_scale).mean()
        if bool(no_solver.any()) else expected_budget.sum() * 0.0
    )
    loss = (
        float(need_solver_weight) * need_loss
        + conditional_loss
        + float(regression_weight) * regression_loss
        + float(underbudget_weight) * underbudget_loss
        + float(zero_overbudget_weight) * zero_overbudget_loss
    )
    iteration_error = (levels[prediction] - levels[safe_labels]).abs().to(torch.float32)
    stats = {
        "Accuracy": float((prediction[valid] == labels[valid]).float().mean().detach().cpu()),
        "ClassMAE": float((prediction[valid] - labels[valid]).abs().float().mean().detach().cpu()),
        "WithinOneAccuracy": float(
            ((prediction[valid] - labels[valid]).abs() <= 1).float().mean().detach().cpu()
        ),
        "IterationMAE": float(iteration_error[valid].mean().detach().cpu()),
        "ExpectedBudgetMAE": float(
            (expected_budget[valid] - true_budget[valid]).abs().mean().detach().cpu()
        ),
        "UnderbudgetLoss": float(underbudget_loss.detach().cpu()),
        "NeedSolverLoss": float(need_loss.detach().cpu()),
        "ConditionalBudgetLoss": float(conditional_loss.detach().cpu()),
        "ZeroOverbudgetLoss": float(zero_overbudget_loss.detach().cpu()),
        "NeedSolverAccuracy": float(
            (((prediction > 0) == (safe_labels > 0))[valid]).float().mean().detach().cpu()
        ),
        "MeanPredictedIterations": float(levels[prediction][valid].float().mean().detach().cpu()),
        "MeanOracleIterations": float(true_budget[valid].mean().detach().cpu()),
    }
    for vi, variable in enumerate(("LAI", "FPAR")):
        use = valid[:, vi]
        stats[f"{variable}Accuracy"] = float(
            (prediction[:, vi][use] == labels[:, vi][use]).float().mean().detach().cpu()
        )
        stats[f"{variable}ClassMAE"] = float(
            (prediction[:, vi][use] - labels[:, vi][use]).abs().float().mean().detach().cpu()
        )
        stats[f"{variable}WithinOneAccuracy"] = float(
            ((prediction[:, vi][use] - labels[:, vi][use]).abs() <= 1)
            .float().mean().detach().cpu()
        )
        stats[f"{variable}IterationMAE"] = float(
            iteration_error[:, vi][use].mean().detach().cpu()
        )
    return loss, stats


def write_distribution(path: Path, caches: dict[str, dict], levels: list[int]) -> None:
    rows = []
    for split, cache in caches.items():
        labels = cache["labels"]
        for vi, variable in enumerate(("LAI", "FPAR")):
            if not bool((labels[:, vi] >= 0).any()):
                continue
            counts = Counter(labels[:, vi][labels[:, vi] >= 0].tolist())
            for class_index, budget in enumerate(levels):
                rows.append({
                    "Split": split,
                    "Variable": variable,
                    "BudgetClass": class_index,
                    "STIterations": budget,
                    "CoreCount": counts.get(class_index, 0),
                })
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_pft_distribution(path: Path, caches: dict[str, dict], pft_codes: list[int]) -> None:
    rows = []
    for split, cache in caches.items():
        labels = cache["labels"]
        core_pft = cache["core_pft"]
        for pft_code in pft_codes:
            pft_use = core_pft == int(pft_code)
            for vi, variable in enumerate(("LAI", "FPAR")):
                if not bool((labels[:, vi] >= 0).any()):
                    continue
                rows.append({
                    "Split": split,
                    "Variable": variable,
                    "PFTCode": int(pft_code),
                    "TotalCoreCount": int(pft_use.sum().item()),
                    "ValidOracleCoreCount": int(
                        (pft_use & (labels[:, vi] >= 0)).sum().item()
                    ),
                })
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_oracle_gain_summary(
    path: Path,
    caches: dict[str, dict],
    levels: torch.Tensor,
) -> None:
    rows = []
    for split, cache in caches.items():
        labels = cache["labels"]
        gains = cache["best_solver_gain"]
        scenarios = list(cache["scenarios"])
        for scenario in ["ALL", *sorted(set(scenarios))]:
            scenario_use = torch.tensor(
                [scenario == "ALL" or name == scenario for name in scenarios],
                dtype=torch.bool,
            )
            for vi, variable in enumerate(("LAI", "FPAR")):
                use = scenario_use & (labels[:, vi] >= 0)
                if not bool(use.any()):
                    continue
                selected_gains = gains[:, vi][use].to(torch.float32)
                selected_labels = labels[:, vi][use]
                rows.append({
                    "Split": split,
                    "Scenario": scenario,
                    "Variable": variable,
                    "ValidCoreCount": int(use.sum().item()),
                    "BeneficialSolverCoreCount": int((selected_labels > 0).sum().item()),
                    "BeneficialSolverFraction": float(
                        (selected_labels > 0).float().mean().item()
                    ),
                    "MeanBestSolverGainNRMSE": float(selected_gains.mean().item()),
                    "MedianBestSolverGainNRMSE": float(selected_gains.median().item()),
                    "P90BestSolverGainNRMSE": float(
                        torch.quantile(selected_gains, 0.90).item()
                    ),
                    "MeanOracleIterations": float(
                        levels.cpu()[selected_labels].float().mean().item()
                    ),
                })
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_best_diagnostics(
    output: Path,
    model: QALLMAdaptiveController,
    caches: dict[str, dict],
    weights: torch.Tensor,
    levels: torch.Tensor,
    best_epoch: int,
    device: torch.device,
    decision_quantiles: torch.Tensor,
    decision_need_thresholds: torch.Tensor,
    pft_quantiles: torch.Tensor,
    pft_need_thresholds: torch.Tensor,
    pft_codes: list[int],
) -> None:
    metric_rows = []
    confusion_rows = []
    model.eval()
    with torch.no_grad():
        for split, cache in caches.items():
            features = cache["features"].to(device)
            contexts = cache["contexts"].to(device)
            labels = cache["labels"].to(device)
            resolved_quantiles = resolve_pft_quantiles(
                decision_quantiles,
                pft_quantiles,
                cache["core_pft"].to(device),
                pft_codes,
            )
            resolved_need_thresholds = resolve_pft_need_thresholds(
                decision_need_thresholds,
                pft_need_thresholds,
                cache["core_pft"].to(device),
                pft_codes,
            )
            loss, stats = head_loss(
                model, features, contexts, labels, weights, levels,
                0.0, 0.0, 0.0, 0.0,
                resolved_quantiles, resolved_need_thresholds,
                cache["structural_eligible"].to(device),
            )
            metric_rows.append({
                "BestEpoch": best_epoch,
                "Split": split,
                "Loss": float(loss.detach().cpu()),
                **stats,
            })
            prediction = decode_budget_logits(
                model.budget_logits_from_core_features(features, contexts),
                resolved_quantiles,
                resolved_need_thresholds,
            )
            prediction = torch.where(
                cache["structural_eligible"].to(device),
                prediction,
                torch.zeros_like(prediction),
            )
            for vi, variable in enumerate(("LAI", "FPAR")):
                valid = labels[:, vi] >= 0
                if not bool(valid.any()):
                    continue
                truth_cpu = labels[:, vi][valid].detach().cpu()
                pred_cpu = prediction[:, vi][valid].detach().cpu()
                counts = Counter(zip(truth_cpu.tolist(), pred_cpu.tolist()))
                for truth_class, truth_budget in enumerate(levels.tolist()):
                    for pred_class, pred_budget in enumerate(levels.tolist()):
                        confusion_rows.append({
                            "Split": split,
                            "Variable": variable,
                            "OracleClass": truth_class,
                            "OracleIterations": truth_budget,
                            "PredictedClass": pred_class,
                            "PredictedIterations": pred_budget,
                            "CoreCount": counts.get((truth_class, pred_class), 0),
                        })
    for path, rows in (
        (output / "budget_best_metrics.csv", metric_rows),
        (output / "budget_confusion_matrix.csv", confusion_rows),
    ):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.root = args.root.resolve()
    args.output_dir = args.output_dir.resolve()
    args.oracle_cache_dir = (
        args.output_dir
        if args.oracle_cache_dir is None
        else args.oracle_cache_dir.resolve()
    )
    args.validation_oracle_cache_dir = (
        args.output_dir
        if args.validation_oracle_cache_dir is None
        else args.validation_oracle_cache_dir.resolve()
    )
    args.lc_mask_dir = args.lc_mask_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.oracle_cache_dir.mkdir(parents=True, exist_ok=True)
    args.validation_oracle_cache_dir.mkdir(parents=True, exist_ok=True)
    if not 0.0 <= args.min_pft_calibration_fraction <= 1.0:
        raise ValueError("min_pft_calibration_fraction must be between 0 and 1.")
    if args.min_pft_calibration_cores < 1:
        raise ValueError("min_pft_calibration_cores must be at least one.")
    if not 0.0 < args.policy_grid_min_quantile <= 1.0:
        raise ValueError("policy_grid_min_quantile must be in (0, 1].")
    if args.policy_min_relative_nrmse_gain < 0.0:
        raise ValueError("policy_min_relative_nrmse_gain must be non-negative.")
    if args.minimum_solver_gain < 0.0:
        raise ValueError("minimum_solver_gain must be nonnegative.")
    if not 0.0 <= args.policy_max_selected_fraction <= 1.0:
        raise ValueError("policy_max_selected_fraction must be between 0 and 1.")
    if args.saturated_target_weight_floor < 0.0 or args.saturated_target_weight_floor > 1.0:
        raise ValueError("saturated_target_weight_floor must be between 0 and 1.")
    if args.policy_max_high_lai_relative_rmse_increase < 0.0:
        raise ValueError("policy_max_high_lai_relative_rmse_increase must be non-negative.")
    if args.policy_max_high_lai_absolute_bias_increase < 0.0:
        raise ValueError("policy_max_high_lai_absolute_bias_increase must be non-negative.")
    for name in (
        "structural_max_patch_missing",
        "structural_max_support_deficit",
    ):
        value = float(getattr(args, name))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1.")
    if args.structural_min_run_steps < 1:
        raise ValueError("structural_min_run_steps must be at least one.")
    set_seed(args.seed)
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    device = resolve_device(args.device)
    levels_list = parse_ints(args.budget_levels)
    if sorted(set(levels_list)) != levels_list or levels_list[0] != 0:
        raise ValueError("budget_levels must be unique, increasing, and start at zero.")
    levels = torch.tensor(levels_list, dtype=torch.long, device=device)
    model, solver, base_checkpoint = load_upgraded_model(
        args.base_checkpoint.resolve(), levels_list, device, args.variable
    )
    validate_checkpoint_tile(base_checkpoint, args.root)
    base_checkpoint_digest = file_sha256(args.base_checkpoint.resolve())
    caches = {
        "train": build_oracle_cache(
            args, model, solver, levels, "train", args.train_samples_per_scenario, device
        ),
        "val": build_oracle_cache(
            args, model, solver, levels, "val", args.val_samples_per_scenario, device
        ),
    }
    for split, cache in caches.items():
        active = variable_index(args.variable)
        if not bool((cache['labels'][:, active] >= 0).any()):
            raise ValueError(f'No valid {args.variable} structural-gap targets in {split}; inspect sampling before calibration.')
        if bool((cache['labels'][:, 1 - active] >= 0).any()):
            raise ValueError('Oracle cache contains the other variable; refusing joint calibration.')
        for vi, variable in enumerate(("LAI", "FPAR")):
            if variable != args.variable:
                continue
            valid = cache["labels"][:, vi] >= 0
            labels_i = cache["labels"][:, vi][valid]
            beneficial_fraction = float((labels_i > 0).float().mean().item())
            mean_iterations = float(
                cache["levels"][labels_i].float().mean().item()
            )
            print(
                f"data-consistent-prior oracle {split} {variable}: "
                f"prior_only={1.0 - beneficial_fraction:.3f} "
                f"solver_beneficial={beneficial_fraction:.3f} "
                f"mean_iterations={mean_iterations:.3f}",
                flush=True,
            )
    needs_pft_refresh = any("core_pft" not in cache for cache in caches.values())
    if args.refresh_budget_contexts or needs_pft_refresh:
        caches = {
            "train": refresh_budget_context_cache(
                args, caches["train"], "train", args.train_samples_per_scenario
            ),
            "val": refresh_budget_context_cache(
                args, caches["val"], "val", args.val_samples_per_scenario
            ),
        }
    train_features = caches["train"]["features"].to(device)
    train_contexts = caches["train"]["contexts"].to(device)
    train_labels = caches["train"]["labels"].to(device)
    val_features = caches["val"]["features"].to(device)
    val_contexts = caches["val"]["contexts"].to(device)
    val_labels = caches["val"]["labels"].to(device)
    weights = torch.stack([
        class_weights(
            train_labels[:, vi],
            len(levels_list),
            args.class_weight_power,
            args.class_weight_cap,
        ) for vi in range(2)
    ]).to(device)
    checkpoint_path = args.output_dir / "best_budget.pt"
    reuse_head = False
    existing = None
    if bool(args.reuse_trained_head) and checkpoint_path.is_file():
        existing = torch.load(checkpoint_path, map_location=device, weights_only=False)
        reuse_head = (
            str(existing.get("base_checkpoint_sha256", ""))
            == base_checkpoint_digest
            and existing.get("budget_policy", {}).get("oracle")
            == "complete-gap-two-stage-benefit-budget-v10-peak-safe"
        )
        if not reuse_head:
            print(
                "Existing budget head belongs to another or unverified base checkpoint; "
                "retraining the lightweight head.",
                flush=True,
            )
    log_rows = []
    if reuse_head:
        assert existing is not None
        head_state = {
            name.removeprefix("budget_head."): value
            for name, value in existing["model_state"].items()
            if name.startswith("budget_head.")
        }
        model.budget_head.load_state_dict(head_state)
        benefit_state = {
            name.removeprefix("budget_benefit_head."): value
            for name, value in existing["model_state"].items()
            if name.startswith("budget_benefit_head.")
        }
        if model.budget_benefit_head is None or not benefit_state:
            raise RuntimeError("Existing budget checkpoint has no benefit head.")
        model.budget_benefit_head.load_state_dict(benefit_state)
        if not {
            "budget_context_weight", "budget_benefit_context_weight"
        }.issubset(existing["model_state"]):
            raise RuntimeError("Existing budget checkpoint has incomplete QA-gap contexts.")
        model.budget_context_weight.data.copy_(existing["model_state"]["budget_context_weight"])
        model.budget_benefit_context_weight.data.copy_(
            existing["model_state"]["budget_benefit_context_weight"]
        )
        best_epoch = int(existing.get("budget_policy", {}).get("best_epoch", 0))
        best_loss = float(existing.get("budget_policy", {}).get("best_validation_loss", float("nan")))
        print("Reusing trained budget head; calibrating only the decision policy.", flush=True)
    else:
        optimizer = torch.optim.AdamW(
            [
                {"params": model.budget_head.parameters(), "lr": args.learning_rate},
                {"params": model.budget_benefit_head.parameters(), "lr": args.learning_rate},
                {
                    "params": [
                        model.budget_context_weight,
                        model.budget_benefit_context_weight,
                    ],
                    "lr": args.context_learning_rate,
                },
            ],
            weight_decay=args.weight_decay,
        )
        best_loss = float("inf")
        best_state = None
        best_epoch = 0
        for epoch in range(1, args.epochs + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            train_loss, train_stats = head_loss(
                model, train_features, train_contexts, train_labels, weights, levels,
                args.budget_regression_weight, args.underbudget_penalty_weight,
                args.need_solver_loss_weight, args.zero_overbudget_penalty_weight,
            )
            train_loss.backward()
            optimizer.step()
            model.eval()
            with torch.no_grad():
                val_loss, val_stats = head_loss(
                    model, val_features, val_contexts, val_labels, weights, levels,
                    args.budget_regression_weight, args.underbudget_penalty_weight,
                    args.need_solver_loss_weight, args.zero_overbudget_penalty_weight,
                )
            row = {
                "Epoch": epoch,
                "TrainLoss": float(train_loss.detach().cpu()),
                "ValLoss": float(val_loss.detach().cpu()),
                **{f"Train{key}": value for key, value in train_stats.items()},
                **{f"Val{key}": value for key, value in val_stats.items()},
            }
            log_rows.append(row)
            print(
                f"budget epoch={epoch}/{args.epochs} train_loss={row['TrainLoss']:.5f} "
                f"val_loss={row['ValLoss']:.5f} val_acc={val_stats['Accuracy']:.3f} "
                f"variable={args.variable} accuracy={val_stats[args.variable + 'Accuracy']:.3f}",
                flush=True,
            )
            if row["ValLoss"] < best_loss:
                best_loss = row["ValLoss"]
                best_epoch = epoch
                best_state = {
                    "head": {
                        k: v.detach().cpu() for k, v in model.budget_head.state_dict().items()
                    },
                    "benefit_head": {
                        k: v.detach().cpu()
                        for k, v in model.budget_benefit_head.state_dict().items()
                    },
                    "context": model.budget_context_weight.detach().cpu().clone(),
                    "benefit_context": (
                        model.budget_benefit_context_weight.detach().cpu().clone()
                    ),
                }
        if best_state is None:
            raise RuntimeError("Budget calibration produced no checkpoint.")
        model.budget_head.load_state_dict(best_state["head"])
        model.budget_benefit_head.load_state_dict(best_state["benefit_head"])
        model.budget_context_weight.data.copy_(best_state["context"].to(device))
        model.budget_benefit_context_weight.data.copy_(
            best_state["benefit_context"].to(device)
        )

    model.eval()
    with torch.no_grad():
        val_logits = model.budget_logits_from_core_features(val_features, val_contexts)
    # Unknown/unclassified PFT codes use the data-consistent prior. There is
    # no defensible evidence for assigning them an expensive solver budget.
    decision_quantiles = torch.full(
        (2,), float(args.policy_grid_min_quantile), dtype=torch.float32, device=device
    )
    decision_need_thresholds = torch.full(
        (2,), 1.01, dtype=torch.float32, device=device
    )
    model.config.budget_quantiles = tuple(float(value) for value in decision_quantiles.cpu())
    model.config.budget_need_thresholds = tuple(
        float(value) for value in decision_need_thresholds.cpu()
    )
    pft_codes = parse_ints(args.lc_codes)
    tile_pft_counts, tile_pft_fractions, tile_pft_source = load_tile_pft_abundance(
        args.lc_mask_dir, pft_codes, args.root
    )
    pft_need_thresholds, pft_quantiles, pft_rows = calibrate_pft_quantiles(
        val_logits,
        val_labels,
        caches["val"]["errors_by_budget"].to(device),
        caches["val"]["target_mass"].to(device),
        caches["val"]["high_lai_errors_by_budget"].to(device),
        caches["val"]["high_lai_bias_by_budget"].to(device),
        caches["val"]["high_lai_target_mass"].to(device),
        caches["val"]["structural_eligible"].to(device),
        caches["val"]["core_pft"].to(device),
        pft_codes,
        levels,
        list(caches["val"]["scenarios"]),
        args.min_pft_calibration_fraction,
        args.min_pft_calibration_cores,
        tile_pft_counts,
        tile_pft_fractions,
        args.policy_grid_min_quantile,
        args.policy_min_relative_nrmse_gain,
        args.policy_max_relative_nrmse_increase,
        args.policy_max_scenario_relative_nrmse_increase,
        args.policy_max_selected_fraction,
        bool(args.policy_require_spatial_block_lai_improvement),
        args.policy_max_high_lai_relative_rmse_increase,
        args.policy_max_high_lai_absolute_bias_increase,
        active_variable=args.variable,
    )
    model.config.budget_pft_codes = tuple(int(code) for code in pft_codes)
    model.config.budget_pft_quantiles = tuple(
        tuple(float(value) for value in row) for row in pft_quantiles.cpu()
    )
    model.config.budget_pft_need_thresholds = tuple(
        tuple(float(value) for value in row) for row in pft_need_thresholds.cpu()
    )
    model.config.budget_structural_min_run_steps = int(args.structural_min_run_steps)
    model.config.budget_structural_max_patch_missing = float(
        args.structural_max_patch_missing
    )
    model.config.budget_structural_max_support_deficit = float(
        args.structural_max_support_deficit
    )
    model.config.budget_structural_cell_selective = True
    print(
        "Biome-specific benefit/budget policies: "
        + "; ".join(
            f"{variable}=" + ",".join(
                f"PFT{code}:need>={model.config.budget_pft_need_thresholds[vi][pi]:.2f},"
                f"q={model.config.budget_pft_quantiles[vi][pi]:.2f}"
                for pi, code in enumerate(pft_codes)
            )
            for vi, variable in enumerate(("LAI", "FPAR"))
            if variable == args.variable
        ),
        flush=True,
    )
    uncertainty_rows = calibrate_validation_uncertainty(
        args, model, solver, levels, device
    )
    state = dict(base_checkpoint["model_state"])
    for name, value in model.state_dict().items():
        if (
            name.startswith("budget_head.")
            or name.startswith("budget_benefit_head.")
            or name in {"budget_context_weight", "budget_benefit_context_weight"}
        ):
            state[name] = value.detach().cpu()
    upgraded = dict(base_checkpoint)
    upgraded.update({
        "version": BUDGET_CHECKPOINT_VERSION,
        "base_checkpoint_sha256": base_checkpoint_digest,
        "quality_protocol": quality_protocol_metadata(),
        "model_state": state,
        "persisted_state_names": sorted(state),
        "trainable_parameter_names": sorted(
            name for name in state
            if name.startswith("budget_head.")
            or name.startswith("budget_benefit_head.")
            or name in {"budget_context_weight", "budget_benefit_context_weight"}
        ),
        "model_config": model.config_dict(),
        "budget_policy": {
            "type": "independent_llm_benefit_then_conditional_budget_with_peak_safe_acceptance",
            "target_domain": "complete_gap_structural_cells",
            "levels": levels_list,
            "normalized_error_tolerance": float(args.normalized_error_tolerance),
            "minimum_solver_gain": float(args.minimum_solver_gain),
            "oracle": (
                "complete-gap-two-stage-benefit-budget-v10-peak-safe"
            ),
            "best_epoch": best_epoch,
            "best_validation_loss": best_loss,
            "decision": (
                "usable-gap-cell-mask-then-PFT-policy-then-score-ranked-top-k-"
                "relative-to-data-consistent-prior"
            ),
            "structural_gap_gate": {
                "cell_selective": True,
                "min_run_steps": model.config.budget_structural_min_run_steps,
                "max_patch_missing": model.config.budget_structural_max_patch_missing,
                "max_support_deficit": model.config.budget_structural_max_support_deficit,
            },
            "unknown_pft_safety_quantiles": list(model.config.budget_quantiles),
            "unknown_pft_need_thresholds": list(model.config.budget_need_thresholds),
            "pft_specific": True,
            "pft_codes": list(model.config.budget_pft_codes),
            "pft_quantiles": [list(row) for row in model.config.budget_pft_quantiles],
            "pft_need_thresholds": [
                list(row) for row in model.config.budget_pft_need_thresholds
            ],
            "min_pft_calibration_fraction": float(
                args.min_pft_calibration_fraction
            ),
            "min_pft_calibration_cores": int(args.min_pft_calibration_cores),
            "tile_pft_fraction_source": str(tile_pft_source),
            "tile_pft_pixel_counts": tile_pft_counts,
            "tile_pft_fractions": tile_pft_fractions,
            "policy_max_relative_nrmse_increase": float(
                args.policy_max_relative_nrmse_increase
            ),
            "policy_min_relative_nrmse_gain": float(
                args.policy_min_relative_nrmse_gain
            ),
            "policy_max_scenario_relative_nrmse_increase": float(
                args.policy_max_scenario_relative_nrmse_increase
            ),
            "policy_max_selected_fraction": float(args.policy_max_selected_fraction),
            "uncertainty_calibration": (
                "validation_complete_gap_routed_standardized_rmse"
            ),
            "policy_require_spatial_block_lai_improvement": bool(
                args.policy_require_spatial_block_lai_improvement
            ),
        },
        "budget_calibration_args": vars(args),
    })
    torch.save(upgraded, checkpoint_path)
    checkpoint_digest = file_sha256(checkpoint_path)
    for row in uncertainty_rows:
        row["CheckpointSHA256"] = checkpoint_digest
    uncertainty_path = args.output_dir / "uncertainty_scale_recommendation.csv"
    with uncertainty_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(uncertainty_rows[0]))
        writer.writeheader()
        writer.writerows(uncertainty_rows)
    if log_rows:
        with (args.output_dir / "budget_training_log.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(log_rows[0]))
            writer.writeheader()
            writer.writerows(log_rows)
    obsolete_global_table = args.output_dir / "budget_decision_calibration.csv"
    if obsolete_global_table.is_file():
        obsolete_global_table.unlink()
    if pft_rows:
        with (args.output_dir / "budget_pft_quantiles.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(pft_rows[0]))
            writer.writeheader()
            writer.writerows(pft_rows)
    abundance_rows = [
        {
            "PFTCode": int(code),
            "TileBiomePixelCount": int(tile_pft_counts[code]),
            "TileVegetationPixelCount": int(sum(tile_pft_counts.values())),
            "TileBiomeFraction": float(tile_pft_fractions[code]),
            "Decision": (
                "independent-calibration"
                if tile_pft_fractions[code] >= args.min_pft_calibration_fraction
                else "prior-only"
            ),
            "Source": str(tile_pft_source),
        }
        for code in pft_codes
    ]
    with (args.output_dir / "budget_tile_biome_abundance.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(abundance_rows[0]))
        writer.writeheader()
        writer.writerows(abundance_rows)
    write_distribution(args.output_dir / "budget_oracle_distribution.csv", caches, levels_list)
    write_pft_distribution(
        args.output_dir / "budget_pft_distribution.csv", caches, pft_codes
    )
    write_oracle_gain_summary(
        args.output_dir / "solver_gain_vs_data_consistent_prior.csv", caches, levels
    )
    obsolete_gain_table = args.output_dir / "solver_gain_vs_phenology_prior.csv"
    if obsolete_gain_table.is_file():
        obsolete_gain_table.unlink()
    write_best_diagnostics(
        args.output_dir, model, caches, weights, levels, best_epoch, device,
        decision_quantiles, decision_need_thresholds,
        pft_quantiles, pft_need_thresholds, pft_codes,
    )
    print(f"Budget calibration complete: {args.output_dir / 'best_budget.pt'}", flush=True)


if __name__ == "__main__":
    main()
