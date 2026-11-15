"""Archived uncertainty statistics; inference and old paths deliberately excluded."""
from __future__ import annotations


import argparse


import csv


import hashlib


import json


import math


from collections import defaultdict


from dataclasses import dataclass, field


from pathlib import Path


import numpy as np


import torch


from .utils import parse_ints, parse_items, resolve_device, saturation_aware_target_weights, set_seed


VARIABLES = ("LAI", "FPAR")


MODES = tuple(range(1, 8))


MODE_NAMES = {
    1: "strict_observation",
    2: "saturated_observation",
    3: "mixed_strict_saturated",
    4: "usable_nonanchor_observation",
    5: "local_gap_fill",
    6: "phenology_gap_fill",
    7: "st_tensor_l1_gap_fill",
}


EPS = 1.0e-8


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def weighted_quantile(values: np.ndarray, weights: np.ndarray, probability: float) -> float:
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values, weights = values[valid], weights[valid]
    if values.size == 0:
        return float("nan")
    order = np.argsort(values, kind="stable")
    values, weights = values[order], weights[order]
    cumulative = np.cumsum(weights)
    cumulative /= cumulative[-1]
    return float(np.interp(float(probability), cumulative, values))


def domain_for_mode(mode: int) -> str:
    return "observed" if int(mode) <= 4 else "reconstructed"


def center_landcover(
    x_window: torch.Tensor,
    center_index: torch.Tensor,
    pft_codes: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    batch = x_window.shape[0]
    indices = torch.arange(batch, device=x_window.device)
    center = x_window[indices, center_index.to(torch.long)]
    start, stop = 19, 19 + len(pft_codes)
    if center.shape[1] < stop:
        raise RuntimeError(
            f"PFT context requires at least {stop} channels; got {center.shape[1]}."
        )
    one_hot = center[:, start:stop, 0, :, :, 0]
    support, pft_index = one_hot.max(dim=1)
    codebook = torch.as_tensor(pft_codes, dtype=torch.long, device=center.device)
    pft = codebook[pft_index]
    pft = torch.where(support > 0.0, pft, torch.zeros_like(pft))
    return pft, pft > 0


@dataclass
class Chunk:
    error: np.ndarray
    sigma: np.ndarray
    weight: np.ndarray


@dataclass
class SampleStore:
    # leaf=(variable, QA mode, PFT); cluster=(tile, scenario)
    groups: dict[tuple[str, int, int], dict[tuple[str, str], list[Chunk]]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(list))
    )

    def add(
        self,
        leaf: tuple[str, int, int],
        cluster: tuple[str, str],
        error: np.ndarray,
        sigma: np.ndarray,
        weight: np.ndarray,
    ) -> None:
        if error.size:
            self.groups[leaf][cluster].append(
                Chunk(
                    np.asarray(error, dtype=np.float32),
                    np.asarray(sigma, dtype=np.float32),
                    np.asarray(weight, dtype=np.float32),
                )
            )


def center_state(
    x_window: torch.Tensor,
    center_index: torch.Tensor,
    result: dict,
    pft_codes: list[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reproduce the product QA mode and pixel PFT from the masked center input."""
    batch = x_window.shape[0]
    indices = torch.arange(batch, device=x_window.device)
    center = x_window[indices, center_index.to(torch.long)]
    valid = torch.stack([center[:, 2] > 0.5, center[:, 3] > 0.5], dim=1)
    candidates = torch.stack([center[:, 6] > 0.5, center[:, 7] > 0.5], dim=1)
    saturated = (center[:, 18] > 0.5).unsqueeze(1).expand_as(candidates)
    strict_count = (candidates & ~saturated).sum(dim=-1)
    saturated_count = (candidates & saturated).sum(dim=-1)
    usable_count = valid.sum(dim=-1)
    pheno_supported = result["expert_support"][:, :, 3] > 0.0
    pheno_dominant = pheno_supported & (result["phenology_group_gate"] >= 0.5)
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
    structural = result.get("solver_structural_fill_mask")
    if structural is not None:
        mode = torch.where(structural, torch.full_like(mode, 7), mode)

    pft, vegetation = center_landcover(x_window, center_index, pft_codes)
    return mode, pft, vegetation


def matching_leaves(
    store: SampleStore,
    variable: str,
    mode: int | None = None,
    pft: int | None = None,
    domain: str | None = None,
) -> list[tuple[str, int, int]]:
    output = []
    for leaf in store.groups:
        if leaf[0] != variable:
            continue
        if mode is not None and leaf[1] != mode:
            continue
        if pft is not None and leaf[2] != pft:
            continue
        if domain is not None and domain_for_mode(leaf[1]) != domain:
            continue
        output.append(leaf)
    return output


def balanced_arrays(
    store: SampleStore,
    leaves: list[tuple[str, int, int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    by_cluster: dict[tuple[str, str], list[Chunk]] = defaultdict(list)
    for leaf in leaves:
        for cluster, chunks in store.groups.get(leaf, {}).items():
            by_cluster[cluster].extend(chunks)
    errors, sigmas, weights = [], [], []
    raw_weight = 0.0
    pixels = 0
    for chunks in by_cluster.values():
        error = np.concatenate([chunk.error for chunk in chunks])
        sigma = np.concatenate([chunk.sigma for chunk in chunks])
        weight = np.concatenate([chunk.weight for chunk in chunks]).astype(np.float64)
        mass = float(weight.sum())
        if mass <= 0:
            continue
        errors.append(error.astype(np.float64))
        sigmas.append(sigma.astype(np.float64))
        weights.append(weight / mass)
        raw_weight += mass
        pixels += int(error.size)
    if not errors:
        return np.array([]), np.array([]), np.array([]), {
            "NPixels": 0, "NWeighted": 0.0, "NClusters": 0, "NTiles": 0,
        }
    return (
        np.concatenate(errors),
        np.concatenate(sigmas),
        np.concatenate(weights),
        {
            "NPixels": pixels,
            "NWeighted": raw_weight,
            "NClusters": len(errors),
            "NTiles": len({cluster[0] for cluster in by_cluster}),
        },
    )


def fit_group(args: argparse.Namespace, store: SampleStore, leaves: list) -> dict:
    error, sigma, weight, support = balanced_arrays(store, leaves)
    if error.size == 0:
        return {
            **support,
            "RawScaleFactor": float("nan"),
            "ScaleFactor": float("nan"),
            "ScaleClipped": 0,
            "Quantile68": float("nan"),
            "Quantile95": float("nan"),
            "ValidationStandardizedRMSE": float("nan"),
            "ValidationGaussianCoverage68": float("nan"),
            "ValidationGaussianCoverage95": float("nan"),
        }
    z_raw = error / np.maximum(sigma, EPS)
    raw_scale = math.sqrt(float(np.sum(weight * z_raw ** 2) / np.sum(weight)))
    scale = min(max(raw_scale, args.scale_min), args.scale_max)
    z = z_raw / scale
    return {
        **support,
        "RawScaleFactor": raw_scale,
        "ScaleFactor": scale,
        "ScaleClipped": int(not math.isclose(scale, raw_scale)),
        "Quantile68": weighted_quantile(z, weight, 0.68),
        "Quantile95": weighted_quantile(z, weight, 0.95),
        "ValidationStandardizedRMSE": math.sqrt(
            float(np.sum(weight * z ** 2) / np.sum(weight))
        ),
        "ValidationGaussianCoverage68": float(np.sum(weight * (z <= 1.0)) / np.sum(weight)),
        "ValidationGaussianCoverage95": float(np.sum(weight * (z <= 1.96)) / np.sum(weight)),
    }


def supported(args: argparse.Namespace, row: dict) -> bool:
    return (
        row.get("NPixels", 0) >= args.min_stratum_pixels
        and row.get("NTiles", 0) >= args.min_stratum_tiles
        and np.isfinite(row.get("ScaleFactor", np.nan))
    )


def fit_calibration(args: argparse.Namespace, store: SampleStore) -> tuple[list[dict], list[dict]]:
    pft_codes = parse_ints(args.lc_codes)
    cache: dict[tuple, dict] = {}

    def fitted(key: tuple, leaves: list) -> dict:
        if key not in cache:
            cache[key] = fit_group(args, store, leaves)
        return cache[key]

    for variable in (args.variable,):
        fitted(("variable", variable), matching_leaves(store, variable))
        for domain in ("observed", "reconstructed"):
            fitted(
                ("domain", variable, domain),
                matching_leaves(store, variable, domain=domain),
            )
        for mode in MODES:
            fitted(("mode", variable, mode), matching_leaves(store, variable, mode=mode))
            for pft in pft_codes:
                fitted(
                    ("mode_pft", variable, mode, pft),
                    matching_leaves(store, variable, mode=mode, pft=pft),
                )

    rows: list[dict] = []
    for variable in (args.variable,):
        for mode in MODES:
            for pft in pft_codes:
                candidates = [
                    ("variable_mode_pft", cache[("mode_pft", variable, mode, pft)]),
                    ("variable_mode", cache[("mode", variable, mode)]),
                    ("variable_domain", cache[("domain", variable, domain_for_mode(mode))]),
                    ("variable", cache[("variable", variable)]),
                ]
                level, selected = next(
                    ((level, value) for level, value in candidates if supported(args, value)),
                    candidates[-1],
                )
                if not np.isfinite(selected.get("ScaleFactor", np.nan)):
                    raise RuntimeError(f"No calibration support for {variable}, mode={mode}, PFT={pft}.")
                rows.append({
                    "Variable": variable,
                    "QAMode": mode,
                    "QAModeName": MODE_NAMES[mode],
                    "PFTCode": pft,
                    "TargetDomain": domain_for_mode(mode),
                    "CalibrationLevel": level,
                    "RawScaleFactor": selected["RawScaleFactor"],
                    "ScaleFactor": selected["ScaleFactor"],
                    "ScaleClipped": selected["ScaleClipped"],
                    "Quantile68": selected["Quantile68"],
                    "Quantile95": selected["Quantile95"],
                    "NPixels": selected["NPixels"],
                    "NWeighted": selected["NWeighted"],
                    "NClusters": selected["NClusters"],
                    "NTiles": selected["NTiles"],
                    "SourceSplit": "val",
                    "GroundDataUsed": 0,
                    "Application": "multiply_raw_hierarchical_uncertainty",
                })

    support_rows = []
    for key, value in sorted(cache.items(), key=lambda item: str(item[0])):
        support_rows.append({
            "Level": key[0],
            "Variable": key[1],
            "Domain": key[2] if key[0] == "domain" else "",
            "QAMode": key[2] if key[0] in {"mode", "mode_pft"} else "",
            "PFTCode": key[3] if key[0] == "mode_pft" else "",
            **value,
            "MeetsMinimumSupport": int(supported(args, value)),
        })
    return rows, support_rows


@dataclass
class MetricSums:
    n: int = 0
    weight: float = 0.0
    z2: float = 0.0
    cover68_gaussian: float = 0.0
    cover95_gaussian: float = 0.0
    cover68_conformal: float = 0.0
    cover95_conformal: float = 0.0
    sigma: float = 0.0
    width68: float = 0.0
    width95: float = 0.0
    nll: float = 0.0
    raw_z2: float = 0.0
    raw_cover68: float = 0.0
    raw_cover95: float = 0.0
    raw_sigma: float = 0.0
    raw_nll: float = 0.0

    def add(
        self,
        error: np.ndarray,
        raw_sigma: np.ndarray,
        sigma: np.ndarray,
        weight: np.ndarray,
        q68: float,
        q95: float,
    ) -> None:
        raw_sigma = np.maximum(raw_sigma, EPS)
        sigma = np.maximum(sigma, EPS)
        raw_z = error / raw_sigma
        z = error / sigma
        self.n += int(error.size)
        self.weight += float(weight.sum())
        self.z2 += float(np.sum(weight * z ** 2))
        self.cover68_gaussian += float(np.sum(weight * (z <= 1.0)))
        self.cover95_gaussian += float(np.sum(weight * (z <= 1.96)))
        self.cover68_conformal += float(np.sum(weight * (z <= q68)))
        self.cover95_conformal += float(np.sum(weight * (z <= q95)))
        self.sigma += float(np.sum(weight * sigma))
        self.width68 += float(np.sum(weight * (2.0 * q68 * sigma)))
        self.width95 += float(np.sum(weight * (2.0 * q95 * sigma)))
        self.nll += float(np.sum(weight * (np.log(sigma) + 0.5 * z ** 2)))
        self.raw_z2 += float(np.sum(weight * raw_z ** 2))
        self.raw_cover68 += float(np.sum(weight * (raw_z <= 1.0)))
        self.raw_cover95 += float(np.sum(weight * (raw_z <= 1.96)))
        self.raw_sigma += float(np.sum(weight * raw_sigma))
        self.raw_nll += float(np.sum(weight * (np.log(raw_sigma) + 0.5 * raw_z ** 2)))

    def normalized(self) -> dict:
        mass = max(self.weight, EPS)
        return {
            "N": self.n,
            "NWeighted": self.weight,
            "StandardizedRMSE": math.sqrt(self.z2 / mass),
            "GaussianCoverage68": self.cover68_gaussian / mass,
            "GaussianCoverage95": self.cover95_gaussian / mass,
            "ConformalCoverage68": self.cover68_conformal / mass,
            "ConformalCoverage95": self.cover95_conformal / mass,
            "MeanSigma": self.sigma / mass,
            "MeanIntervalWidth68": self.width68 / mass,
            "MeanIntervalWidth95": self.width95 / mass,
            "GaussianNLL": self.nll / mass,
            "RawStandardizedRMSE": math.sqrt(self.raw_z2 / mass),
            "RawGaussianCoverage68": self.raw_cover68 / mass,
            "RawGaussianCoverage95": self.raw_cover95 / mass,
            "RawMeanSigma": self.raw_sigma / mass,
            "RawGaussianNLL": self.raw_nll / mass,
        }


def evaluate_test(store: SampleStore, calibration_rows: list[dict]) -> list[dict]:
    lookup = {
        (row["Variable"], int(row["QAMode"]), int(row["PFTCode"])): row
        for row in calibration_rows
    }
    clusters: dict[tuple, MetricSums] = defaultdict(MetricSums)
    for leaf, cluster_map in store.groups.items():
        calibration = lookup[leaf]
        scale = float(calibration["ScaleFactor"])
        q68 = float(calibration["Quantile68"])
        q95 = float(calibration["Quantile95"])
        variable, mode, pft = leaf
        for cluster, chunks in cluster_map.items():
            error = np.concatenate([chunk.error for chunk in chunks]).astype(np.float64)
            raw_sigma = np.concatenate([chunk.sigma for chunk in chunks]).astype(np.float64)
            sigma = raw_sigma * scale
            weight = np.concatenate([chunk.weight for chunk in chunks]).astype(np.float64)
            tile, scenario = cluster
            groups = (
                ("ALL", "ALL"),
                ("QA_MODE", str(mode)),
                ("PFT", str(pft)),
                ("TILE", tile),
                ("SCENARIO", scenario),
            )
            for grouping, level in groups:
                clusters[(grouping, level, variable, cluster)].add(
                    error, raw_sigma, sigma, weight, q68, q95
                )

    rows = []
    group_keys = sorted({key[:3] for key in clusters})
    for grouping, level, variable in group_keys:
        parts = [value for key, value in clusters.items() if key[:3] == (grouping, level, variable)]
        pooled = MetricSums()
        for part in parts:
            for name in MetricSums.__dataclass_fields__:
                setattr(pooled, name, getattr(pooled, name) + getattr(part, name))
        rows.append({
            "Grouping": grouping,
            "Level": level,
            "Variable": variable,
            "Weighting": "pixel_quality_weighted",
            "NClusters": len(parts),
            **pooled.normalized(),
        })
        valid_parts = [part for part in parts if part.weight > 0]
        normalized = [part.normalized() for part in valid_parts]
        cluster_row = {
            "Grouping": grouping,
            "Level": level,
            "Variable": variable,
            "Weighting": "tile_scenario_equal",
            "NClusters": len(normalized),
            "N": sum(int(value["N"]) for value in normalized),
            "NWeighted": sum(float(value["NWeighted"]) for value in normalized),
        }
        cluster_row["StandardizedRMSE"] = math.sqrt(float(np.mean([
            part.z2 / part.weight for part in valid_parts
        ])))
        cluster_row["RawStandardizedRMSE"] = math.sqrt(float(np.mean([
            part.raw_z2 / part.weight for part in valid_parts
        ])))
        for name in (
            "GaussianCoverage68", "GaussianCoverage95",
            "ConformalCoverage68", "ConformalCoverage95", "MeanSigma",
            "MeanIntervalWidth68", "MeanIntervalWidth95", "GaussianNLL",
            "RawGaussianCoverage68", "RawGaussianCoverage95", "RawMeanSigma",
            "RawGaussianNLL",
        ):
            cluster_row[name] = float(np.mean([value[name] for value in normalized]))
        rows.append(cluster_row)
    return rows


def reliability_table(
    args: argparse.Namespace,
    store: SampleStore,
    calibration_rows: list[dict],
) -> list[dict]:
    lookup = {
        (row["Variable"], int(row["QAMode"]), int(row["PFTCode"])): row
        for row in calibration_rows
    }
    rows = []
    for variable in (args.variable,):
        by_cluster: dict[tuple[str, str], list[tuple[np.ndarray, ...]]] = defaultdict(list)
        for leaf, cluster_map in store.groups.items():
            if leaf[0] != variable:
                continue
            calibration = lookup[leaf]
            for cluster, chunks in cluster_map.items():
                for chunk in chunks:
                    n = chunk.error.size
                    by_cluster[cluster].append((
                        chunk.error.astype(np.float64),
                        chunk.sigma.astype(np.float64) * float(calibration["ScaleFactor"]),
                        chunk.weight.astype(np.float64),
                        np.full(n, float(calibration["Quantile68"])),
                        np.full(n, float(calibration["Quantile95"])),
                    ))
        if not by_cluster:
            continue
        per_cluster = max(1, args.reliability_max_points // len(by_cluster))
        arrays = []
        for cluster, pieces in sorted(by_cluster.items()):
            joined = [np.concatenate([piece[index] for piece in pieces]) for index in range(5)]
            if joined[0].size > per_cluster:
                positions = np.linspace(0, joined[0].size - 1, per_cluster).round().astype(int)
                joined = [value[positions] for value in joined]
            joined[2] = joined[2] / max(float(joined[2].sum()), EPS)
            arrays.append(joined)
        error, sigma, weight, q68, q95 = [
            np.concatenate([array[index] for array in arrays]) for index in range(5)
        ]
        edges = [weighted_quantile(sigma, weight, p) for p in np.linspace(0, 1, args.reliability_bins + 1)]
        edges[0], edges[-1] = -np.inf, np.inf
        for index in range(args.reliability_bins):
            use = (sigma > edges[index]) & (sigma <= edges[index + 1])
            mass = float(weight[use].sum())
            if mass <= 0:
                continue
            rows.append({
                "Variable": variable,
                "Bin": index + 1,
                "N": int(use.sum()),
                "MeanSigma": float(np.sum(weight[use] * sigma[use]) / mass),
                "ObservedRMSE": math.sqrt(float(np.sum(weight[use] * error[use] ** 2) / mass)),
                "ObservedMAE": float(np.sum(weight[use] * error[use]) / mass),
                "ConformalCoverage68": float(np.sum(weight[use] * (error[use] <= q68[use] * sigma[use])) / mass),
                "ConformalCoverage95": float(np.sum(weight[use] * (error[use] <= q95[use] * sigma[use])) / mass),
                "Weighting": "tile_scenario_equal",
            })
    return rows


def build_test_decisions(primary: list[dict], args: argparse.Namespace) -> list[dict]:
    decisions = []
    for row in primary:
        coverage68 = float(row["ConformalCoverage68"])
        coverage95 = float(row["ConformalCoverage95"])
        exact68 = abs(coverage68 - 0.68) <= args.coverage68_tolerance
        exact95 = abs(coverage95 - 0.95) <= args.coverage95_tolerance
        deploy68 = coverage68 >= 0.68 - args.deployment_undercoverage68
        deploy95 = coverage95 >= 0.95 - args.deployment_undercoverage95
        decisions.append({
            "Variable": row["Variable"],
            "Coverage68": coverage68,
            "Coverage95": coverage95,
            "Pass68": exact68,
            "Pass95": exact95,
            "Pass": bool(exact68 and exact95),
            "DeploymentPass68": bool(deploy68),
            "DeploymentPass95": bool(deploy95),
            "DeploymentPass": bool(deploy68 and deploy95),
            "Coverage68Deviation": coverage68 - 0.68,
            "Coverage95Deviation": coverage95 - 0.95,
        })
    return decisions


def update_report_decision(report: dict, decisions: list[dict], args: argparse.Namespace) -> dict:
    exact_passed = bool(decisions) and all(row["Pass"] for row in decisions)
    deployment_allowed = bool(decisions) and all(row["DeploymentPass"] for row in decisions)
    report.update({
        "schema": "qa-llm-gf-eight-tile-product-uncertainty-v2",
        "coverage_tolerances": {
            "coverage68": args.coverage68_tolerance,
            "coverage95": args.coverage95_tolerance,
        },
        "deployment_coverage_limits": {
            "undercoverage68": args.deployment_undercoverage68,
            "undercoverage95": args.deployment_undercoverage95,
            "overcoverage_blocks_deployment": False,
        },
        "test_decisions": decisions,
        "calibration_quality_passed": exact_passed,
        "deployment_allowed": deployment_allowed,
        # Preserve the historical meaning of passed: exact symmetric calibration.
        "passed": exact_passed,
        "deployment_status": (
            "validated_candidate_pending_application"
            if exact_passed
            else "validated_conservative_candidate_pending_application"
            if deployment_allowed
            else "test_deployment_gate_failed_do_not_apply"
        ),
    })
    return report


