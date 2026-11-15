from __future__ import annotations

import argparse
import csv
import json
import os
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler

from qg_dataset import DatasetOptions

from .data import WindowedQADataset
from .variable import variable_index, assert_single_batch
from .model import ControllerConfig, QALLMAdaptiveController
from .pipeline import aggregate_withheld_target, run_reliability_window, run_window
from .experts import EXPERT_NAMES, observed_si
from .quality import (
    TRAINING_CHECKPOINT_VERSION,
    quality_protocol_metadata,
    validate_checkpoint_protocol,
    validate_warm_start_protocol,
)
from .solver import SolverConfig
from .utils import (
    atomic_json,
    parse_ints,
    parse_items,
    resolve_device,
    saturation_aware_target_weights,
    set_seed,
    split_target_domains,
    variable_scale,
)


DEFAULT_SCENARIOS = (
    "random_10",
    "random_30",
    "temporal_block_3",
    "temporal_block_6",
    "spatial_block",
    "sensor_denial_MOD",
    "sensor_denial_MYD",
    "sensor_denial_VNP",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train frozen QA-LLM-GF candidate-risk selection")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--variable", choices=("LAI", "FPAR"), required=True)
    p.add_argument(
        "--output-subdir",
        default="Module02_Training",
    )
    p.add_argument("--run-name", default="qa_llm_gf")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--pretrain-epochs", type=int, default=3)
    p.add_argument(
        "--phenology-calibration-epochs",
        type=int,
        default=3,
        help=(
            "Epochs that calibrate residual-consistent Local/PHENO-mean/"
            "PHENO-peak risks with ST-Tensor/L1 disabled."
        ),
    )
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--pretrain-batch-size", type=int, default=2)
    p.add_argument("--solver-batch-size", type=int, default=1)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=20260720)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--torch-threads", type=int, default=0)
    p.add_argument("--window-years", type=int, default=5)
    p.add_argument("--sensor-start-years", default="2001,2002,2012")
    p.add_argument("--scenarios", default=",".join(DEFAULT_SCENARIOS))
    p.add_argument("--max-train-samples-per-scenario", type=int, default=200)
    p.add_argument("--max-val-samples-per-scenario", type=int, default=40)
    p.add_argument("--max-solver-train-samples-per-scenario", type=int, default=40)
    p.add_argument("--max-solver-val-samples-per-scenario", type=int, default=10)
    p.add_argument("--sample-reserve-fraction", type=float, default=0.10)
    p.add_argument("--max-train-batches", type=int, default=0)
    p.add_argument("--max-val-batches", type=int, default=0)
    p.add_argument("--h5-cache-size", type=int, default=8)
    p.add_argument("--token-cache-size", type=int, default=8)
    p.add_argument("--use-lc-context", type=int, default=1)
    p.add_argument("--lc-mask-dir", type=Path, default=None)
    p.add_argument("--lc-codes", default="1:8")
    p.add_argument("--input-channels", type=int, default=27)
    p.add_argument("--backbone", choices=["gpt2", "tiny"], default="gpt2")
    p.add_argument("--gpt-model", default="gpt2")
    p.add_argument("--gpt-layers", type=int, default=1)
    p.add_argument("--context-dim", type=int, default=64)
    p.add_argument("--local-dim", type=int, default=16)
    p.add_argument("--spatial-grid", type=int, default=4)
    p.add_argument("--hierarchy-hidden-dim", type=int, default=32)
    p.add_argument("--phenology-phase-radius", type=int, default=1)
    p.add_argument("--fit-visible-phenology-template", type=int, default=1)
    p.add_argument("--phenology-amplitude-min", type=float, default=0.50)
    p.add_argument("--phenology-amplitude-max", type=float, default=2.00)
    p.add_argument("--phenology-minimum-fit-steps", type=int, default=3)
    p.add_argument("--phenology-selector-bias", type=float, default=0.0)
    p.add_argument("--candidate-risk-margin", type=float, default=0.01)
    p.add_argument("--candidate-risk-temperature", type=float, default=0.08)
    p.add_argument("--solver-acceptance-bias", type=float, default=0.0)
    p.add_argument("--residual-scale-lai", type=float, default=1.0)
    p.add_argument("--residual-scale-fpar", type=float, default=0.10)
    p.add_argument("--amplitude-log-gain-limit", type=float, default=0.50)
    p.add_argument("--memory-logit-bias", type=float, default=-1.50)
    p.add_argument("--memory-support-scale-init", type=float, default=1.00)
    p.add_argument("--memory-dispersion-scale-init", type=float, default=2.00)
    p.add_argument("--saturated-target-weight-floor", type=float, default=0.70)
    p.add_argument("--core-size", type=int, default=8)
    p.add_argument("--solver-training-iterations", type=int, default=2)
    p.add_argument("--solver-min-iterations", type=int, default=3)
    p.add_argument("--solver-max-iterations", type=int, default=24)
    p.add_argument("--solver-rank-cap", type=int, default=8)
    p.add_argument("--solver-tolerance", type=float, default=1e-3)
    p.add_argument("--direct-loss-weight", type=float, default=0.20)
    p.add_argument("--value-bin-loss-weight", type=float, default=0.20)
    p.add_argument("--high-value-selection-weight", type=float, default=0.15)
    p.add_argument("--max-high-lai-relative-rmse-increase", type=float, default=0.0)
    p.add_argument("--max-high-lai-absolute-bias-increase", type=float, default=0.0)
    p.add_argument("--uncertainty-loss-weight", type=float, default=0.03)
    p.add_argument("--control-loss-weight", type=float, default=0.02)
    p.add_argument("--gate-loss-weight", type=float, default=0.10)
    p.add_argument("--gate-temperature", type=float, default=0.05)
    p.add_argument("--support-loss-weight", type=float, default=0.01)
    p.add_argument("--dynamic-loss-weight", type=float, default=0.02)
    p.add_argument("--amplitude-loss-weight", type=float, default=0.05)
    p.add_argument("--memory-gate-loss-weight", type=float, default=0.05)
    p.add_argument("--memory-gate-temperature", type=float, default=0.05)
    p.add_argument("--phenology-selector-loss-weight", type=float, default=0.05)
    p.add_argument("--phenology-selector-temperature", type=float, default=0.05)
    p.add_argument("--phenology-group-loss-weight", type=float, default=0.05)
    p.add_argument("--candidate-error-loss-weight", type=float, default=0.20)
    p.add_argument("--candidate-ranking-loss-weight", type=float, default=0.50)
    p.add_argument("--solver-acceptance-loss-weight", type=float, default=0.05)
    p.add_argument("--data-consistency-loss-weight", type=float, default=0.15)
    p.add_argument("--data-fidelity-gain", type=float, default=12.0)
    p.add_argument("--l1-data-fidelity-gain", type=float, default=8.0)
    p.add_argument("--l1-peak-relaxation", type=float, default=0.70)
    p.add_argument("--residual-phenology-solver", type=int, default=1)
    p.add_argument("--early-stop-patience", type=int, default=2)
    p.add_argument("--early-stop-min-delta", type=float, default=1e-4)
    p.add_argument("--resume-checkpoint", type=Path, default=None)
    p.add_argument("--initial-checkpoint", type=Path, default=None)
    p.add_argument(
        "--trainable-scope",
        choices=("control_heads", "from_scratch"),
        default="control_heads",
        help=(
            "Use control_heads for a tile-specific frozen foundation, or from_scratch "
            "for one pooled model with newly initialized remote-sensing task layers."
        ),
    )
    p.add_argument(
        "--control-target-domain",
        choices=("all_targets", "complete_gaps"),
        default="all_targets",
        help="Target pixels used to supervise the continuous control heads.",
    )
    p.add_argument("--print-every", type=int, default=10)
    return p.parse_args()


def build_dataset(
    args: argparse.Namespace,
    split: str,
    window_years: int,
    max_samples_per_scenario: int,
) -> WindowedQADataset:
    return WindowedQADataset(
        root=args.root,
        split=split,
        scenarios=parse_items(args.scenarios),
        window_years=window_years,
        sensor_start_years=parse_ints(args.sensor_start_years),
        options=DatasetOptions(
            active_variable=args.variable,
            return_meta=False,
            min_anchor_ratio=0.0,
            h5_cache_size=args.h5_cache_size,
            token_cache_size=args.token_cache_size,
            use_lc_context=bool(args.use_lc_context),
            lc_mask_dir=args.lc_mask_dir,
            lc_codes=parse_ints(args.lc_codes),
        ),
        max_samples_per_scenario=max_samples_per_scenario,
        sample_reserve_fraction=args.sample_reserve_fraction,
        seed=args.seed + (0 if split == "train" else 1),
    )


class TokenGroupedSampler(Sampler[int]):
    """Shuffle TokenRow groups while keeping scenario variants cache-local."""

    def __init__(self, dataset: WindowedQADataset, seed: int, shuffle: bool) -> None:
        self.groups = [
            group.index.to_numpy(dtype=np.int64).tolist()
            for _, group in dataset.base.mask_manifest.groupby("TokenRow", sort=False)
        ]
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.epoch = 0

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        order = np.arange(len(self.groups))
        if self.shuffle:
            rng.shuffle(order)
        self.epoch += 1
        for group_index in order:
            group = list(self.groups[int(group_index)])
            if self.shuffle:
                rng.shuffle(group)
            yield from group

    def __len__(self) -> int:
        return sum(len(group) for group in self.groups)


def si_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    weight: torch.Tensor,
    log_variance: torch.Tensor | None,
    uncertainty_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    scale = variable_scale(prediction, prediction.ndim)
    error = (prediction - target) / scale
    losses = []
    stats: dict[str, float] = {}
    names = ("lai", "fpar")
    for vi, name in enumerate(names):
        use = mask[:, vi]
        w = weight[:, vi] * use.to(weight.dtype)
        if float(w.sum().detach().cpu()) <= 0:
            continue
        e = error[:, vi]
        mse = (e.square() * w).sum() / w.sum().clamp_min(1.0)
        losses.append(mse)
        stats[f"{name}_norm_rmse"] = float(torch.sqrt(mse.detach()).cpu())
        physical = prediction[:, vi] - target[:, vi]
        stats[f"{name}_rmse"] = float(torch.sqrt((physical.square() * w).sum() / w.sum()).detach().cpu())
        stats[f"{name}_mae"] = float(((physical.abs() * w).sum() / w.sum()).detach().cpu())
        stats[f"{name}_bias"] = float(((physical * w).sum() / w.sum()).detach().cpu())
    if not losses:
        return prediction.sum() * 0.0, stats
    reconstruction_loss = torch.stack(losses).mean()
    loss = reconstruction_loss
    stats["reconstruction_loss"] = float(reconstruction_loss.detach().cpu())
    nll_value = prediction.sum() * 0.0
    if log_variance is not None and uncertainty_weight > 0:
        nll_weight = weight * mask.to(weight.dtype)
        nll = 0.5 * (torch.exp(-log_variance) * error.square() + log_variance) * nll_weight
        nll_value = nll.sum() / nll_weight.sum().clamp_min(1.0)
        loss = loss + float(uncertainty_weight) * nll_value
    stats["nll_loss"] = float(nll_value.detach().cpu())
    variable_rmse = [
        value for value in (stats.get("lai_norm_rmse"), stats.get("fpar_norm_rmse"))
        if value is not None and np.isfinite(value)
    ]
    stats["norm_rmse_balanced"] = float(np.mean(variable_rmse)) if variable_rmse else float("nan")
    return loss, stats


def value_bin_balanced_reconstruction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """Give each populated physical-value stratum equal, symmetric influence."""
    scale = variable_scale(prediction, prediction.ndim)
    normalized_error = (prediction - target) / scale
    bins = (
        ((0.0, 1.0), (1.0, 3.0), (3.0, 5.0), (5.0, float("inf"))),
        ((0.0, 0.25), (0.25, 0.50), (0.50, 0.75), (0.75, float("inf"))),
    )
    losses = []
    for vi in range(2):
        for lower, upper in bins[vi]:
            use = weight[:, vi] * mask[:, vi].to(weight.dtype)
            use = use * (target[:, vi] >= lower).to(weight.dtype)
            if np.isfinite(upper):
                use = use * (target[:, vi] < upper).to(weight.dtype)
            mass = use.sum()
            if float(mass.detach().cpu()) > 0:
                losses.append(
                    (normalized_error[:, vi].square() * use).sum()
                    / mass.clamp_min(1e-8)
                )
    return torch.stack(losses).mean() if losses else prediction.sum() * 0.0


def gate_supervision_loss(
    experts: torch.Tensor,
    gates: torch.Tensor,
    support: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    weight: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    scale = variable_scale(target, target.ndim).unsqueeze(2)
    normalized_error = (experts - target.unsqueeze(2)).abs() / scale
    available = support > 0.0
    no_available = ~available.any(dim=2, keepdim=True)
    available = available.clone()
    available[:, :, :1] |= no_available
    oracle_logits = -normalized_error.detach() / max(float(temperature), 1e-4)
    oracle = torch.softmax(oracle_logits.masked_fill(~available, -1e4), dim=2)
    cross_entropy = -(oracle * torch.log(gates.clamp_min(1e-8))).sum(dim=2)
    losses = []
    for vi in range(2):
        use = weight[:, vi] * mask[:, vi].to(weight.dtype)
        if float(use.sum().detach().cpu()) > 0:
            losses.append((cross_entropy[:, vi] * use).sum() / use.sum().clamp_min(1.0))
    return torch.stack(losses).mean() if losses else gates.sum() * 0.0


def gate_diagnostics(
    experts: torch.Tensor,
    gates: torch.Tensor,
    support: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    weight: torch.Tensor,
) -> dict[str, float]:
    scale = variable_scale(target, target.ndim).unsqueeze(2)
    available = support > 0.0
    no_available = ~available.any(dim=2, keepdim=True)
    available = available.clone()
    available[:, :, :1] |= no_available
    normalized_error = (experts - target.unsqueeze(2)).abs() / scale
    best_expert = normalized_error.masked_fill(~available, float("inf")).argmin(dim=2)
    selected_expert = gates.argmax(dim=2)
    use = weight * mask.to(weight.dtype)
    denom = use.sum().clamp_min(1.0)
    entropy = -(gates * torch.log(gates.clamp_min(1e-8))).sum(dim=2)
    result = {
        "gate_entropy": float(((entropy * use).sum() / denom).detach().cpu()),
        "gate_top1_accuracy": float(
            (((selected_expert == best_expert).to(use.dtype) * use).sum() / denom).detach().cpu()
        ),
    }
    for ei, name in enumerate(("lin", "idw", "cst", "pheno")):
        result[f"gate_{name}_mean"] = float(
            ((gates[:, :, ei] * use).sum() / denom).detach().cpu()
        )
    return result


def expert_support_loss(
    gates: torch.Tensor,
    support: torch.Tensor,
    mask: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    penalty = (gates * (1.0 - support.clamp(0.0, 1.0))).sum(dim=2)
    use = weight * mask.to(weight.dtype)
    return (penalty * use).sum() / use.sum().clamp_min(1.0)


def continuous_gate_oracle_loss(
    start_candidate: torch.Tensor,
    end_candidate: torch.Tensor,
    gate: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    weight: torch.Tensor,
    evidence_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Fit a continuous analytic interpolation oracle with value-bin balance."""
    scale = variable_scale(target, target.ndim)
    delta = end_candidate.detach() - start_candidate.detach()
    normalized_separation = (delta.abs() / scale).clamp(0.0, 1.0)
    separated = normalized_separation > 1e-6
    safe_delta = torch.where(separated, delta, torch.ones_like(delta))
    oracle = ((target.detach() - start_candidate.detach()) / safe_delta).clamp(0.0, 1.0)
    # A gate error matters in proportion to the physical effect of choosing
    # one endpoint over the other. This suppresses unstable oracle labels when
    # the candidates are numerically indistinguishable.
    use = (
        weight
        * mask.to(weight.dtype)
        * separated.to(weight.dtype)
        * normalized_separation.detach().square()
    )
    if evidence_weight is not None:
        use = use * evidence_weight.detach().clamp(0.0, 1.0)
    element_loss = torch.nn.functional.smooth_l1_loss(
        gate, oracle, reduction="none", beta=0.10
    )
    bins = (
        (("lt1", 0.0, 1.0), ("1to3", 1.0, 3.0), ("3to5", 3.0, 5.0), ("ge5", 5.0, float("inf"))),
        (("lt0p25", 0.0, 0.25), ("0p25to0p50", 0.25, 0.50), ("0p50to0p75", 0.50, 0.75), ("ge0p75", 0.75, float("inf"))),
    )
    bin_losses = []
    stats: dict[str, float] = {}
    for vi, name in enumerate(("lai", "fpar")):
        variable_use = use[:, vi]
        variable_mass = variable_use.sum()
        if float(variable_mass.detach().cpu()) > 0:
            stats[f"{name}_oracle"] = float(
                ((oracle[:, vi] * variable_use).sum() / variable_mass).detach().cpu()
            )
            stats[f"{name}_mae"] = float(
                (((gate[:, vi] - oracle[:, vi]).abs() * variable_use).sum() / variable_mass)
                .detach().cpu()
            )
            stats[f"{name}_separation"] = float(
                (
                    (normalized_separation[:, vi] * variable_use).sum()
                    / variable_mass
                ).detach().cpu()
            )
        for bin_name, lower, upper in bins[vi]:
            bin_use = variable_use * (target[:, vi] >= lower).to(weight.dtype)
            if np.isfinite(upper):
                bin_use = bin_use * (target[:, vi] < upper).to(weight.dtype)
            mass = bin_use.sum()
            if float(mass.detach().cpu()) > 0:
                stats[f"{name}_{bin_name}_gate"] = float(
                    ((gate[:, vi] * bin_use).sum() / mass).detach().cpu()
                )
                stats[f"{name}_{bin_name}_oracle"] = float(
                    ((oracle[:, vi] * bin_use).sum() / mass).detach().cpu()
                )
                stats[f"{name}_{bin_name}_mae"] = float(
                    (
                        ((gate[:, vi] - oracle[:, vi]).abs() * bin_use).sum()
                        / mass
                    ).detach().cpu()
                )
                stats[f"{name}_{bin_name}_mass"] = float(mass.detach().cpu())
                bin_losses.append(
                    (element_loss[:, vi] * bin_use).sum() / mass.clamp_min(1e-8)
                )
    if not bin_losses:
        return gate.sum() * 0.0, stats
    return torch.stack(bin_losses).mean(), stats


def candidate_error_prediction_loss(
    local_candidate: torch.Tensor,
    phenology_candidate: torch.Tensor,
    predicted_error: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    weight: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Fit endpoint errors with value-bin balance and conflict-aware weighting."""
    if predicted_error.shape[:3] != (
        target.shape[0], target.shape[1], 2
    ):
        raise ValueError("predicted_error must have shape [B,2,2,T,H,W].")
    scale = variable_scale(target, target.ndim)
    target_error = torch.stack(
        (
            (local_candidate.detach() - target.detach()).abs() / scale,
            (phenology_candidate.detach() - target.detach()).abs() / scale,
        ),
        dim=2,
    ).clamp(0.0, 2.0)
    separation = (
        (phenology_candidate.detach() - local_candidate.detach()).abs() / scale
    ).clamp(0.0, 1.0)
    # Gate error changes the reconstruction in proportion to squared endpoint
    # separation. Equal per-value-bin reduction prevents abundant middle-range
    # cells from dominating sparse but consequential conflicts.
    use = weight * mask.to(weight.dtype) * separation.square()
    element = torch.nn.functional.smooth_l1_loss(
        predicted_error, target_error, reduction="none", beta=0.10
    ).mean(dim=2)
    bin_losses = []
    stats: dict[str, float] = {}
    bins = (
        (("lt1", 0.0, 1.0), ("1to3", 1.0, 3.0), ("3to5", 3.0, 5.0), ("ge5", 5.0, float("inf"))),
        (("lt0p25", 0.0, 0.25), ("0p25to0p50", 0.25, 0.50), ("0p50to0p75", 0.50, 0.75), ("ge0p75", 0.75, float("inf"))),
    )
    for vi, name in enumerate(("lai", "fpar")):
        mass = use[:, vi].sum()
        if float(mass.detach().cpu()) <= 0:
            continue
        for ci, candidate_name in enumerate(("local", "phenology")):
            stats[f"{name}_{candidate_name}_predicted_error"] = float(
                (
                    predicted_error[:, vi, ci] * use[:, vi]
                ).sum().div(mass.clamp_min(1e-8)).detach().cpu()
            )
            stats[f"{name}_{candidate_name}_target_error"] = float(
                (
                    target_error[:, vi, ci] * use[:, vi]
                ).sum().div(mass.clamp_min(1e-8)).detach().cpu()
            )
        for bin_name, lower, upper in bins[vi]:
            bin_use = use[:, vi] * (target[:, vi] >= lower).to(weight.dtype)
            if np.isfinite(upper):
                bin_use = bin_use * (target[:, vi] < upper).to(weight.dtype)
            bin_mass = bin_use.sum()
            if float(bin_mass.detach().cpu()) <= 0:
                continue
            bin_losses.append(
                (element[:, vi] * bin_use).sum() / bin_mass.clamp_min(1e-8)
            )
            predicted_gate = predicted_error[:, vi, 0] / predicted_error[:, vi].sum(
                dim=1
            ).clamp_min(1e-6)
            stats[f"{name}_{bin_name}_error_gate"] = float(
                ((predicted_gate * bin_use).sum() / bin_mass).detach().cpu()
            )
            for ci, candidate_name in enumerate(("local", "phenology")):
                stats[f"{name}_{bin_name}_{candidate_name}_predicted_error"] = float(
                    (
                        predicted_error[:, vi, ci] * bin_use
                    ).sum().div(bin_mass.clamp_min(1e-8)).detach().cpu()
                )
                stats[f"{name}_{bin_name}_{candidate_name}_target_error"] = float(
                    (
                        target_error[:, vi, ci] * bin_use
                    ).sum().div(bin_mass.clamp_min(1e-8)).detach().cpu()
                )
    if not bin_losses:
        return predicted_error.sum() * 0.0, stats
    return torch.stack(bin_losses).mean(), stats


def three_candidate_risk_loss(
    local_candidate: torch.Tensor,
    phenology_mean_candidate: torch.Tensor,
    phenology_peak_candidate: torch.Tensor,
    predicted_risk: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    weight: torch.Tensor,
    ranking_weight: float,
    temperature: float,
    pft_code: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Calibrate Local/PHENO-mean/PHENO-peak risks and their direct ranking."""
    if predicted_risk.shape[:3] != (target.shape[0], target.shape[1], 3):
        raise ValueError("predicted_risk must have shape [B,2,3,T,H,W].")
    scale = variable_scale(target, target.ndim)
    candidates = torch.stack(
        (local_candidate, phenology_mean_candidate, phenology_peak_candidate), dim=2
    )
    target_risk = ((candidates.detach() - target.detach().unsqueeze(2)).abs() /
                   scale.unsqueeze(2)).clamp(0.0, 2.0)
    best = target_risk.argmin(dim=2)
    spread = (target_risk.max(dim=2).values - target_risk.min(dim=2).values).clamp(0.0, 1.0)
    use = weight * mask.to(weight.dtype) * (0.05 + spread.detach())
    regression = torch.nn.functional.smooth_l1_loss(
        predicted_risk, target_risk, reduction="none", beta=0.10
    ).mean(dim=2)
    logits = -predicted_risk / max(float(temperature), 1e-4)
    ranking = torch.nn.functional.cross_entropy(
        logits.permute(0, 1, 3, 4, 5, 2).reshape(-1, 3),
        best.reshape(-1),
        reduction="none",
    ).reshape_as(best)
    selected = predicted_risk.argmin(dim=2)

    if pft_code is None:
        pft_code = torch.zeros_like(target[:, 0], dtype=torch.long)
    elif pft_code.ndim == target.ndim - 2:
        pft_code = pft_code.unsqueeze(1).expand(-1, target.shape[2], -1, -1)
    if pft_code.shape != target[:, 0].shape:
        raise ValueError("pft_code must have shape [B,T,H,W] or [B,H,W].")

    bins = (
        ((0.0, 1.0), (1.0, 3.0), (3.0, 5.0), (5.0, float("inf"))),
        ((0.0, 0.25), (0.25, 0.50), (0.50, 0.75), (0.75, float("inf"))),
    )
    group_losses = []
    stats: dict[str, float] = {}
    candidate_names = ("local", "pheno_mean", "pheno_peak")
    for vi, variable_name in enumerate(("lai", "fpar")):
        variable_use = use[:, vi]
        mass = variable_use.sum()
        if float(mass.detach().cpu()) > 0:
            stats[f"{variable_name}_top1_accuracy"] = float(
                (((selected[:, vi] == best[:, vi]).to(weight.dtype) * variable_use).sum()
                 / mass).detach().cpu()
            )
            for ci, candidate_name in enumerate(candidate_names):
                stats[f"{variable_name}_{candidate_name}_selected_fraction"] = float(
                    (((selected[:, vi] == ci).to(weight.dtype) * variable_use).sum()
                     / mass).detach().cpu()
                )
                stats[f"{variable_name}_{candidate_name}_predicted_risk"] = float(
                    ((predicted_risk[:, vi, ci] * variable_use).sum() / mass).detach().cpu()
                )
                stats[f"{variable_name}_{candidate_name}_target_risk"] = float(
                    ((target_risk[:, vi, ci] * variable_use).sum() / mass).detach().cpu()
                )
        for lower, upper in bins[vi]:
            value_group = target[:, vi] >= lower
            if np.isfinite(upper):
                value_group &= target[:, vi] < upper
            represented_pfts = torch.unique(pft_code[value_group & (variable_use > 0)])
            for code in represented_pfts.tolist():
                group_use = variable_use * value_group.to(weight.dtype)
                group_use = group_use * (pft_code == int(code)).to(weight.dtype)
                group_mass = group_use.sum()
                if float(group_mass.detach().cpu()) <= 0:
                    continue
                class_mass = torch.stack(
                    [
                        (group_use * (best[:, vi] == ci).to(weight.dtype)).sum()
                        for ci in range(3)
                    ]
                )
                represented = class_mass > 0
                class_weight = torch.zeros_like(class_mass)
                class_weight[represented] = (
                    group_mass
                    / represented.sum().clamp_min(1).to(group_mass.dtype)
                    / class_mass[represented].clamp_min(1e-8)
                ).clamp(max=4.0)
                ranking_use = group_use * class_weight[best[:, vi]]
                group_losses.append(
                    (regression[:, vi] * group_use).sum() / group_mass.clamp_min(1e-8)
                    + float(ranking_weight)
                    * (ranking[:, vi] * ranking_use).sum()
                    / ranking_use.sum().clamp_min(1e-8)
                )
    if not group_losses:
        return predicted_risk.sum() * 0.0, stats
    return torch.stack(group_losses).mean(), stats


def dynamic_consistency_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    scale = variable_scale(prediction, prediction.ndim)
    pred_n = prediction / scale
    target_n = target / scale
    pair_mask = mask[:, :, 1:] & mask[:, :, :-1]
    pair_weight = torch.minimum(weight[:, :, 1:], weight[:, :, :-1]) * pair_mask.to(weight.dtype)
    if float(pair_weight.sum().detach().cpu()) <= 0:
        return prediction.sum() * 0.0
    pred_diff = pred_n[:, :, 1:] - pred_n[:, :, :-1]
    target_diff = target_n[:, :, 1:] - target_n[:, :, :-1]
    return ((pred_diff - target_diff).abs() * pair_weight).sum() / pair_weight.sum().clamp_min(1.0)


def amplitude_consistency_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """Match observed seasonal amplitude without a one-sided high-value penalty."""
    scale = variable_scale(prediction, prediction.ndim)
    prediction = prediction / scale
    target = target / scale
    use = weight * mask.to(weight.dtype)
    time_weight = use.sum(dim=(-1, -2))
    valid_time = time_weight > 0
    safe_weight = time_weight.clamp_min(1e-8)
    prediction_time = (prediction * use).sum(dim=(-1, -2)) / safe_weight
    target_time = (target * use).sum(dim=(-1, -2)) / safe_weight
    normalized_weight = time_weight / time_weight.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    prediction_mean = (prediction_time * normalized_weight).sum(dim=-1, keepdim=True)
    target_mean = (target_time * normalized_weight).sum(dim=-1, keepdim=True)
    prediction_std = torch.sqrt(
        ((prediction_time - prediction_mean).square() * normalized_weight).sum(dim=-1)
        + 1e-8
    )
    target_std = torch.sqrt(
        ((target_time - target_mean).square() * normalized_weight).sum(dim=-1)
        + 1e-8
    )
    valid_series = valid_time.sum(dim=-1) >= 2
    if not bool(valid_series.any()):
        return prediction.sum() * 0.0
    return torch.nn.functional.smooth_l1_loss(
        prediction_std[valid_series], target_std[valid_series]
    )


def phenology_memory_gate_loss(
    base_prior: torch.Tensor,
    memory: torch.Tensor,
    gate: torch.Tensor,
    support: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    weight: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Teach the LLM gate to use same-DOY memory only when it reduces error."""
    scale = variable_scale(target, target.ndim)
    base_error = (base_prior - target).abs() / scale
    memory_error = (memory - target).abs() / scale
    oracle = torch.sigmoid(
        (base_error.detach() - memory_error.detach()) / max(float(temperature), 1e-4)
    )
    use = weight * mask.to(weight.dtype) * support.detach().clamp(0.0, 1.0)
    if float(use.sum().detach().cpu()) <= 0:
        return gate.sum() * 0.0
    cross_entropy = torch.nn.functional.binary_cross_entropy(
        gate.clamp(1e-6, 1.0 - 1e-6), oracle, reduction="none"
    )
    return (cross_entropy * use).sum() / use.sum().clamp_min(1.0)


def observation_consistency_loss(
    prediction: torch.Tensor,
    observed: torch.Tensor,
    observation_weight: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Symmetric fidelity loss on visible QA-qualified observations."""
    scale = variable_scale(prediction, prediction.ndim)
    error = (prediction - observed) / scale
    losses = []
    stats: dict[str, float] = {}
    for vi, name in enumerate(("lai", "fpar")):
        weight = observation_weight[:, vi].clamp(0.0, 1.0)
        if float(weight.sum().detach().cpu()) <= 0:
            continue
        mse = (error[:, vi].square() * weight).sum() / weight.sum().clamp_min(1.0)
        losses.append(mse)
        stats[f"{name}_observation_norm_rmse"] = float(torch.sqrt(mse.detach()).cpu())
    value = torch.stack(losses).mean() if losses else prediction.sum() * 0.0
    stats["observation_consistency_loss"] = float(value.detach().cpu())
    return value, stats


class EpochMetrics:
    """Pixel-weighted micro metrics and scenario-variable macro normalized RMSE."""

    def __init__(self) -> None:
        self.values = defaultdict(lambda: {"weight": 0.0, "sq": 0.0, "abs": 0.0, "err": 0.0})
        self.high_values = defaultdict(
            lambda: {"weight": 0.0, "sq": 0.0, "abs": 0.0, "err": 0.0}
        )

    def update(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        weight: torch.Tensor,
        scenarios: list[str],
    ) -> None:
        for bi, scenario in enumerate(scenarios):
            for vi, variable in enumerate(("lai", "fpar")):
                use = weight[bi, vi] * mask[bi, vi].to(weight.dtype)
                total = float(use.sum().detach().cpu())
                if total <= 0:
                    continue
                error = prediction[bi, vi] - target[bi, vi]
                for key in (("ALL", variable), (str(scenario), variable)):
                    row = self.values[key]
                    row["weight"] += total
                    row["sq"] += float((error.square() * use).sum().detach().cpu())
                    row["abs"] += float((error.abs() * use).sum().detach().cpu())
                    row["err"] += float((error * use).sum().detach().cpu())
                threshold = 5.0 if variable == "lai" else 0.90
                high_use = use * (target[bi, vi] >= threshold).to(use.dtype)
                high_total = float(high_use.sum().detach().cpu())
                if high_total > 0:
                    row = self.high_values[variable]
                    row["weight"] += high_total
                    row["sq"] += float((error.square() * high_use).sum().detach().cpu())
                    row["abs"] += float((error.abs() * high_use).sum().detach().cpu())
                    row["err"] += float((error * high_use).sum().detach().cpu())

    def result(self) -> dict[str, float]:
        out: dict[str, float] = {}
        normalized_macro = []
        normalized_micro = []
        for variable, scale in (("lai", 7.0), ("fpar", 1.0)):
            row = self.values.get(("ALL", variable))
            if row and row["weight"] > 0:
                rmse = (row["sq"] / row["weight"]) ** 0.5
                out[f"{variable}_rmse"] = rmse
                out[f"{variable}_norm_rmse"] = rmse / scale
                out[f"{variable}_mae"] = row["abs"] / row["weight"]
                out[f"{variable}_bias"] = row["err"] / row["weight"]
                normalized_micro.append(rmse / scale)
        for (scenario, variable), row in self.values.items():
            if scenario == "ALL" or row["weight"] <= 0:
                continue
            scale = 7.0 if variable == "lai" else 1.0
            normalized_macro.append((row["sq"] / row["weight"]) ** 0.5 / scale)
        out["norm_rmse_micro_balanced"] = (
            float(np.mean(normalized_micro)) if normalized_micro else float("nan")
        )
        out["norm_rmse_macro"] = float(np.mean(normalized_macro)) if normalized_macro else float("nan")
        out["norm_rmse_balanced"] = out["norm_rmse_macro"]
        high_normalized = []
        for variable, scale, label in (
            ("lai", 7.0, "ge5"),
            ("fpar", 1.0, "ge0p9"),
        ):
            row = self.high_values.get(variable)
            if row and row["weight"] > 0:
                rmse = (row["sq"] / row["weight"]) ** 0.5
                out[f"{variable}_{label}_rmse"] = rmse
                out[f"{variable}_{label}_norm_rmse"] = rmse / scale
                out[f"{variable}_{label}_mae"] = row["abs"] / row["weight"]
                out[f"{variable}_{label}_bias"] = row["err"] / row["weight"]
                out[f"{variable}_{label}_weight"] = row["weight"]
                high_normalized.append(rmse / scale)
        out["high_value_norm_rmse_balanced"] = (
            float(np.mean(high_normalized)) if high_normalized else float("nan")
        )
        return out


def prefixed_metrics(prefix: str, values: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in values.items()}


def add_model_selection_metrics(
    values: dict[str, float],
    args: argparse.Namespace,
    baseline_high_rmse: float,
    baseline_high_abs_bias: float,
) -> None:
    """Select on complete gaps while enforcing LAI>=5 no-degradation."""
    prefix = "complete_gap_" if args.control_target_domain == "complete_gaps" else ""
    macro = values.get(f"{prefix}norm_rmse_macro", float("nan"))
    high_key = 'lai_ge5' if args.variable == 'LAI' else 'fpar_ge0p9'
    high_rmse = values.get(f"{prefix}{high_key}_norm_rmse", float("nan"))
    high_bias = values.get(f"{prefix}{high_key}_bias", float("nan"))
    high_weight = values.get(f"{prefix}{high_key}_weight", 0.0)
    has_high_constraint = (
        np.isfinite(baseline_high_rmse)
        and np.isfinite(baseline_high_abs_bias)
        and np.isfinite(high_rmse)
        and np.isfinite(high_bias)
        and float(high_weight) > 0.0
    )
    eligible = True
    if has_high_constraint:
        eligible = (
            high_rmse
            <= baseline_high_rmse
            * (1.0 + float(args.max_high_lai_relative_rmse_increase))
            + 1e-12
            and abs(high_bias)
            <= baseline_high_abs_bias
            + float(args.max_high_lai_absolute_bias_increase)
            + 1e-12
        )
    if np.isfinite(high_rmse):
        weight = float(args.high_value_selection_weight)
        score = (1.0 - weight) * macro + weight * high_rmse
    else:
        score = macro
    values["model_selection_eligible"] = float(eligible)
    values["selection_baseline_lai_ge5_norm_rmse"] = baseline_high_rmse
    values["selection_baseline_lai_ge5_abs_bias"] = baseline_high_abs_bias
    values["model_selection_score"] = score if eligible else float("inf")


def run_epoch(
    model: QALLMAdaptiveController,
    loader: DataLoader,
    device: torch.device,
    solver_config: SolverConfig,
    args: argparse.Namespace,
    epoch: int,
    optimizer: torch.optim.Optimizer | None,
    scenario_quota: int,
    control_stage: str = "joint",
) -> tuple[dict[str, float], dict[str, int]]:
    training = optimizer is not None
    if control_stage not in {"joint", "reliability", "solver"}:
        raise ValueError(f"Unknown control stage: {control_stage}")
    if args.trainable_scope == "control_heads":
        # Keep the inherited frontend and parent controls deterministic. Only
        # the head declared active for this calibration stage may enter train
        # mode; this remains correct if dropout/normalization is added later.
        model.eval()
        parent_heads = ("group_gate_correction_head",)
        reliability_heads = (
            "phenology_selector_head", "candidate_error_head", "candidate_rank_head"
        )
        solver_heads = ("solver_acceptance_head", "post_solver_acceptance_head")
        active_heads = (
            reliability_heads
            if control_stage == "reliability"
            else solver_heads
            if control_stage == "solver"
            else reliability_heads + solver_heads
        )
        for head_name in parent_heads + reliability_heads + solver_heads:
            head = getattr(model, head_name, None)
            if head is not None:
                head.train(training and head_name in active_heads)
    else:
        model.train(training)
    phase = "train" if training else "val"
    max_batches = args.max_train_batches if training else args.max_val_batches
    sums: dict[str, float] = {}
    scenario_counts: dict[str, int] = defaultdict(int)
    metrics_accumulator = EpochMetrics()
    complete_gap_metrics = EpochMetrics()
    partial_sensor_metrics = EpochMetrics()
    used_samples = 0
    forward_batches = 0
    skipped_no_target = 0
    skipped_quota = 0
    tic = time.time()
    for batch_idx, batch in enumerate(loader, start=1):
        if max_batches > 0 and forward_batches >= max_batches:
            break
        assert_single_batch(batch, args.variable)
        x_window = batch["x_window"].to(device)
        center_index = batch["center_index"].to(device)
        target_sensor = batch["target"].to(device)
        target_mask_sensor = batch["target_mask"].to(device)
        target_weight_sensor = batch["target_weight"].to(device)
        batch_size = x_window.shape[0]
        target_weight_sensor = saturation_aware_target_weights(
            x_window,
            center_index,
            target_mask_sensor,
            target_weight_sensor,
            args.saturated_target_weight_floor,
            batch["target_saturated"].to(device),
        )
        target, target_mask, target_weight = aggregate_withheld_target(
            target_sensor, target_mask_sensor, target_weight_sensor
        )
        batch_scenarios = [str(value) for value in batch["scenario"]]
        keep = []
        for bi, scenario in enumerate(batch_scenarios):
            if not bool(target_mask[bi].any()):
                skipped_no_target += 1
                continue
            if scenario_quota > 0 and scenario_counts[scenario] >= scenario_quota:
                skipped_quota += 1
                continue
            keep.append(bi)
            scenario_counts[scenario] += 1
        if not keep:
            continue
        keep_index = torch.as_tensor(keep, dtype=torch.long, device=device)
        x_window = x_window.index_select(0, keep_index)
        center_index = center_index.index_select(0, keep_index)
        target = target.index_select(0, keep_index)
        target_mask = target_mask.index_select(0, keep_index)
        target_weight = target_weight.index_select(0, keep_index)
        scenarios = [batch_scenarios[index] for index in keep]

        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            b = x_window.shape[0]
            gather_index = center_index.view(b, 1, 1, 1, 1, 1, 1).expand(
                b, 1, *x_window.shape[2:]
            )
            center_x = x_window.gather(1, gather_index).squeeze(1)
            if center_x.shape[1] >= 27:
                pft_score = center_x[:, 19:27].mean(dim=(2, 5))
                pft_code = pft_score.argmax(dim=1) + 1
                pft_code = torch.where(
                    pft_score.max(dim=1).values > 0,
                    pft_code,
                    torch.zeros_like(pft_code),
                )
            else:
                pft_code = torch.zeros(
                    (b, center_x.shape[3], center_x.shape[4]),
                    dtype=torch.long,
                    device=center_x.device,
                )
            if args.trainable_scope == "from_scratch":
                # Train the complete current QA-LLM prior on genuine multi-year
                # Local/PHENO evidence. Numerical ST-Tensor/L1 remains disabled
                # here and is calibrated separately in Module 2B.
                out = run_reliability_window(
                    model,
                    x_window,
                    center_index,
                    solver_config.data_fidelity_gain,
                )
                prior = out["data_consistent_prior"]
                prediction = prior
                log_variance = out["log_variance"]
                visible_observed = out["observed"]
                visible_weight = out["observation_weight"]
                st_iterations = 0.0
            elif epoch <= args.pretrain_epochs:
                out = model(center_x)
                prediction = out["prior"]
                prior = prediction
                log_variance = out["log_variance"]
                visible_observed, visible_weight, _, _ = observed_si(
                    center_x, out["reliability"]
                )
                st_iterations = 0.0
            else:
                if control_stage == "reliability":
                    out = run_reliability_window(
                        model,
                        x_window,
                        center_index,
                        solver_config.data_fidelity_gain,
                    )
                else:
                    out = run_window(
                        model,
                        x_window,
                        center_index,
                        solver_config,
                        core_size=args.core_size,
                        training_solver=True,
                    )
                prior = out["data_consistent_prior"]
                prediction = prior if control_stage == "reliability" else out["prediction"]
                log_variance = out["log_variance"]
                visible_observed = out["observed"]
                visible_weight = out["observation_weight"]
                st_iterations = float(out["st_diagnostics"]["iterations"].float().mean().detach().cpu())
            complete_gap_mask, partial_sensor_mask = split_target_domains(
                target_mask, visible_weight
            )
            control_mask = (
                complete_gap_mask
                if args.control_target_domain == "complete_gaps"
                else target_mask
            )
            reconstruction_mask = (
                control_mask if args.trainable_scope == "control_heads" else target_mask
            )
            valid = torch.stack([center_x[:, 2] > 0.5, center_x[:, 3] > 0.5], dim=1)
            observed_fraction = valid.to(center_x.dtype).mean(dim=(2, 3, 4, 5))
            time_coverage = valid.any(dim=-1).to(center_x.dtype).mean(dim=(3, 4))
            empty_time_fraction = (time_coverage < 0.10).to(center_x.dtype).mean(dim=2)
            scale = variable_scale(target, target.ndim).unsqueeze(2)
            expert_disagreement = (out["experts"] / scale).std(dim=2, unbiased=False).mean(dim=(2, 3, 4))
            difficulty_target = (
                0.10
                + 0.35 * (1.0 - observed_fraction)
                + 0.25 * empty_time_fraction
                + 0.30 * expert_disagreement.clamp(0.0, 1.0)
            ).clamp(0.0, 1.0)
            continuation_target = (
                0.60 * difficulty_target + 0.40 * expert_disagreement.clamp(0.0, 1.0)
            ).clamp(0.0, 1.0)
            vi = variable_index(args.variable)
            control_loss = 0.5 * (
                torch.nn.functional.mse_loss(out["difficulty"][:, vi], difficulty_target[:, vi]) +
                torch.nn.functional.mse_loss(out["continuation"][:, vi], continuation_target[:, vi])
            )
            active_uncertainty_weight = (
                0.0
                if args.trainable_scope == "control_heads"
                else float(args.uncertainty_loss_weight)
            )
            loss, metrics = si_loss(
                prediction,
                target,
                reconstruction_mask,
                target_weight,
                log_variance,
                active_uncertainty_weight,
            )
            reconstruction_loss = loss
            value_bin_loss = value_bin_balanced_reconstruction_loss(
                prediction, target, reconstruction_mask, target_weight
            )
            prior_loss = prediction.sum() * 0.0
            if (
                epoch > args.pretrain_epochs
                and control_stage != "reliability"
                and args.direct_loss_weight > 0
            ):
                direct_loss, _ = si_loss(
                    prior, target, reconstruction_mask, target_weight, None, 0.0
                )
                prior_loss = direct_loss
            gate_loss = gate_supervision_loss(
                out["experts"][:, :, :3],
                out["local_gate_weights"],
                out["expert_support"][:, :, :3],
                target,
                reconstruction_mask,
                target_weight,
                args.gate_temperature,
            )
            gate_stats = gate_diagnostics(
                out["experts"], out["gate_weights"], out["expert_support"],
                target, target_mask, target_weight
            )
            support_loss = expert_support_loss(
                out["gate_weights"], out["expert_support"], reconstruction_mask, target_weight
            )
            pheno_available = out["expert_support"][:, :, 3] > 0.0
            pheno_control_mask = control_mask & pheno_available
            selector_loss, selector_stats = continuous_gate_oracle_loss(
                out["phenology_mean_candidate_endpoint"],
                out["phenology_peak_candidate_endpoint"],
                out["phenology_selector"],
                target,
                pheno_control_mask,
                target_weight,
            )
            group_loss, group_stats = continuous_gate_oracle_loss(
                out["local_candidate_endpoint"],
                out["phenology_candidate_endpoint"],
                out["phenology_group_gate"],
                target,
                pheno_control_mask,
                target_weight,
            )
            if bool(getattr(model.config, "use_direct_three_candidate_routing", False)):
                candidate_error_loss, candidate_error_stats = three_candidate_risk_loss(
                    out["local_candidate_endpoint"],
                    out["phenology_mean_candidate_endpoint"],
                    out["phenology_peak_candidate_endpoint"],
                    out["candidate_risk"],
                    target,
                    pheno_control_mask,
                    target_weight,
                    args.candidate_ranking_loss_weight,
                    args.candidate_risk_temperature,
                    pft_code,
                )
            else:
                candidate_error_loss, candidate_error_stats = candidate_error_prediction_loss(
                    out["local_candidate_endpoint"],
                    out["phenology_candidate_endpoint"],
                    out["candidate_error"],
                    target,
                    pheno_control_mask,
                    target_weight,
                )
            if epoch > args.pretrain_epochs and control_stage != "reliability":
                acceptance_loss, acceptance_stats = continuous_gate_oracle_loss(
                    out["data_consistent_prior"],
                    out["solver_candidate"],
                    out["solver_acceptance"],
                    target,
                    control_mask,
                    target_weight,
                )
            else:
                acceptance_loss = prediction.sum() * 0.0
                acceptance_stats = {}
            auxiliary_prediction = prior if epoch > args.pretrain_epochs else prediction
            dynamic_loss = dynamic_consistency_loss(
                auxiliary_prediction, target, reconstruction_mask, target_weight
            )
            amplitude_loss = amplitude_consistency_loss(
                auxiliary_prediction, target, reconstruction_mask, target_weight
            )
            if epoch > args.pretrain_epochs and args.memory_gate_loss_weight > 0.0:
                memory_gate_loss = phenology_memory_gate_loss(
                    out["base_prior"],
                    out["phenology_memory"],
                    out["phenology_memory_gate"],
                    out["phenology_memory_support"],
                    target,
                    reconstruction_mask,
                    target_weight,
                    args.memory_gate_temperature,
                )
            else:
                memory_gate_loss = prediction.sum() * 0.0
            data_consistency_loss, data_consistency_stats = observation_consistency_loss(
                auxiliary_prediction, visible_observed, visible_weight
            )
            gate_weight = float(args.gate_loss_weight) * (
                1.0 if epoch <= args.pretrain_epochs else 0.5
            )
            if args.trainable_scope == "control_heads" and control_stage == "reliability":
                # Legacy head-only calibration isolates normalized Local and
                # adaptive-PHENO risks from reconstruction gradients.
                loss = float(args.candidate_error_loss_weight) * candidate_error_loss
            elif args.trainable_scope == "control_heads" and control_stage == "solver":
                loss = (
                    loss
                    + float(args.value_bin_loss_weight) * value_bin_loss
                    + float(args.solver_acceptance_loss_weight) * acceptance_loss
                )
            else:
                loss = (
                    loss
                    + float(args.value_bin_loss_weight) * value_bin_loss
                    + float(args.direct_loss_weight) * prior_loss
                    + float(args.control_loss_weight) * control_loss
                    + gate_weight * gate_loss
                    + float(args.support_loss_weight) * support_loss
                    + float(args.phenology_selector_loss_weight) * selector_loss
                    + float(args.phenology_group_loss_weight) * group_loss
                    + float(args.candidate_error_loss_weight) * candidate_error_loss
                    + float(args.solver_acceptance_loss_weight) * acceptance_loss
                    + float(args.dynamic_loss_weight) * dynamic_loss
                    + float(args.amplitude_loss_weight) * amplitude_loss
                    + float(args.memory_gate_loss_weight) * memory_gate_loss
                    + float(args.data_consistency_loss_weight) * data_consistency_loss
                )
            if training:
                named_losses = {
                    "total": loss,
                    "reconstruction": reconstruction_loss,
                    "value_bin": value_bin_loss,
                    "prior": prior_loss,
                    "control": control_loss,
                    "gate": gate_loss,
                    "support": support_loss,
                    "phenology_selector": selector_loss,
                    "phenology_group": group_loss,
                    "candidate_error": candidate_error_loss,
                    "solver_acceptance": acceptance_loss,
                    "dynamic": dynamic_loss,
                    "amplitude": amplitude_loss,
                    "memory_gate": memory_gate_loss,
                    "data_consistency": data_consistency_loss,
                }
                if not bool(torch.isfinite(loss.detach()).all()):
                    invalid_losses = [
                        name for name, value in named_losses.items()
                        if not bool(torch.isfinite(value.detach()).all())
                    ]
                    raise FloatingPointError(
                        "Non-finite loss components before backward: "
                        + ", ".join(invalid_losses)
                    )
                if os.environ.get("QG_DETECT_ANOMALY", "0") == "1":
                    with torch.autograd.detect_anomaly():
                        loss.backward()
                else:
                    loss.backward()
                bad_gradients = [
                    name for name, parameter in model.named_parameters()
                    if parameter.grad is not None
                    and not bool(torch.isfinite(parameter.grad).all())
                ]
                if bad_gradients:
                    raise FloatingPointError(
                        "Non-finite gradients in: " + ", ".join(bad_gradients[:12])
                    )
                if args.trainable_scope == "control_heads" and control_stage in {
                    "reliability", "solver"
                }:
                    expected_prefixes = (
                        (
                            "phenology_selector_head.",
                            "candidate_error_head.",
                            "candidate_rank_head.",
                        )
                        if control_stage == "reliability"
                        else (
                            "solver_acceptance_head.",
                            "post_solver_acceptance_head.",
                        )
                    )
                    leaked_gradients = [
                        name
                        for name, parameter in model.named_parameters()
                        if parameter.grad is not None
                        and not name.startswith(expected_prefixes)
                    ]
                    if leaked_gradients:
                        raise RuntimeError(
                            f"{control_stage} stage leaked gradients into: "
                            + ", ".join(leaked_gradients[:12])
                        )
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

        metrics_accumulator.update(prediction, target, target_mask, target_weight, scenarios)
        complete_gap_metrics.update(
            prediction, target, complete_gap_mask, target_weight, scenarios
        )
        partial_sensor_metrics.update(
            prediction, target, partial_sensor_mask, target_weight, scenarios
        )
        sample_count = len(scenarios)
        row = {
            "loss": float(loss.detach().cpu()),
            "reconstruction_loss": metrics.get("reconstruction_loss", float("nan")),
            "value_bin_loss": float(value_bin_loss.detach().cpu()),
            "nll_loss": metrics.get("nll_loss", float("nan")),
            "prior_loss": float(prior_loss.detach().cpu()),
            "gate_loss": float(gate_loss.detach().cpu()),
            "support_loss": float(support_loss.detach().cpu()),
            "phenology_selector_loss": float(selector_loss.detach().cpu()),
            "phenology_selector_lai": float(
                out["phenology_selector"][:, 0].mean().detach().cpu()
            ),
            "phenology_selector_fpar": float(
                out["phenology_selector"][:, 1].mean().detach().cpu()
            ),
            "phenology_selector_oracle_lai": selector_stats.get("lai_oracle", float("nan")),
            "phenology_selector_oracle_fpar": selector_stats.get("fpar_oracle", float("nan")),
            "phenology_selector_mae_lai": selector_stats.get("lai_mae", float("nan")),
            "phenology_selector_mae_fpar": selector_stats.get("fpar_mae", float("nan")),
            "phenology_selector_separation_lai": selector_stats.get(
                "lai_separation", float("nan")
            ),
            "phenology_selector_separation_fpar": selector_stats.get(
                "fpar_separation", float("nan")
            ),
            "phenology_selector_lai_ge5": selector_stats.get("lai_ge5_gate", float("nan")),
            "phenology_selector_oracle_lai_ge5": selector_stats.get("lai_ge5_oracle", float("nan")),
            "phenology_selector_mae_lai_ge5": selector_stats.get("lai_ge5_mae", float("nan")),
            "phenology_selector_lai_ge5_mass": selector_stats.get("lai_ge5_mass", 0.0),
            "phenology_group_loss": float(group_loss.detach().cpu()),
            "phenology_group_lai": float(
                out["phenology_group_gate"][:, 0].mean().detach().cpu()
            ),
            "phenology_group_fpar": float(
                out["phenology_group_gate"][:, 1].mean().detach().cpu()
            ),
            "phenology_group_oracle_lai": group_stats.get("lai_oracle", float("nan")),
            "phenology_group_oracle_fpar": group_stats.get("fpar_oracle", float("nan")),
            "phenology_group_mae_lai": group_stats.get("lai_mae", float("nan")),
            "phenology_group_mae_fpar": group_stats.get("fpar_mae", float("nan")),
            "phenology_group_separation_lai": group_stats.get(
                "lai_separation", float("nan")
            ),
            "phenology_group_separation_fpar": group_stats.get(
                "fpar_separation", float("nan")
            ),
            "phenology_group_lai_ge5": group_stats.get("lai_ge5_gate", float("nan")),
            "phenology_group_oracle_lai_ge5": group_stats.get("lai_ge5_oracle", float("nan")),
            "phenology_group_mae_lai_ge5": group_stats.get("lai_ge5_mae", float("nan")),
            "phenology_group_lai_ge5_mass": group_stats.get("lai_ge5_mass", 0.0),
            "candidate_error_loss": float(candidate_error_loss.detach().cpu()),
            "candidate_error_gate_lai": float(
                out["candidate_error_gate"][:, 0].mean().detach().cpu()
            ),
            "candidate_error_gate_fpar": float(
                out["candidate_error_gate"][:, 1].mean().detach().cpu()
            ),
            "candidate_error_lai_local_predicted": candidate_error_stats.get(
                "lai_local_predicted_error", float("nan")
            ),
            "candidate_error_lai_local_target": candidate_error_stats.get(
                "lai_local_target_error", float("nan")
            ),
            "candidate_error_lai_pheno_predicted": candidate_error_stats.get(
                "lai_phenology_predicted_error", float("nan")
            ),
            "candidate_error_lai_pheno_target": candidate_error_stats.get(
                "lai_phenology_target_error", float("nan")
            ),
            "candidate_error_lai_ge5_gate": candidate_error_stats.get(
                "lai_ge5_error_gate", float("nan")
            ),
            "candidate_error_lai_ge5_local_predicted": candidate_error_stats.get(
                "lai_ge5_local_predicted_error", float("nan")
            ),
            "candidate_error_lai_ge5_local_target": candidate_error_stats.get(
                "lai_ge5_local_target_error", float("nan")
            ),
            "candidate_error_lai_ge5_pheno_predicted": candidate_error_stats.get(
                "lai_ge5_phenology_predicted_error", float("nan")
            ),
            "candidate_error_lai_ge5_pheno_target": candidate_error_stats.get(
                "lai_ge5_phenology_target_error", float("nan")
            ),
            "candidate_rank_top1_lai": candidate_error_stats.get(
                "lai_top1_accuracy", float("nan")
            ),
            "candidate_rank_top1_fpar": candidate_error_stats.get(
                "fpar_top1_accuracy", float("nan")
            ),
            "candidate_rank_lai_local_fraction": candidate_error_stats.get(
                "lai_local_selected_fraction", float("nan")
            ),
            "candidate_rank_lai_pheno_mean_fraction": candidate_error_stats.get(
                "lai_pheno_mean_selected_fraction", float("nan")
            ),
            "candidate_rank_lai_pheno_peak_fraction": candidate_error_stats.get(
                "lai_pheno_peak_selected_fraction", float("nan")
            ),
            "candidate_rank_fpar_local_fraction": candidate_error_stats.get(
                "fpar_local_selected_fraction", float("nan")
            ),
            "candidate_rank_fpar_pheno_mean_fraction": candidate_error_stats.get(
                "fpar_pheno_mean_selected_fraction", float("nan")
            ),
            "candidate_rank_fpar_pheno_peak_fraction": candidate_error_stats.get(
                "fpar_pheno_peak_selected_fraction", float("nan")
            ),
            "candidate_rank_lai_local_predicted_risk": candidate_error_stats.get(
                "lai_local_predicted_risk", float("nan")
            ),
            "candidate_rank_lai_local_target_risk": candidate_error_stats.get(
                "lai_local_target_risk", float("nan")
            ),
            "candidate_rank_lai_pheno_mean_predicted_risk": candidate_error_stats.get(
                "lai_pheno_mean_predicted_risk", float("nan")
            ),
            "candidate_rank_lai_pheno_mean_target_risk": candidate_error_stats.get(
                "lai_pheno_mean_target_risk", float("nan")
            ),
            "candidate_rank_lai_pheno_peak_predicted_risk": candidate_error_stats.get(
                "lai_pheno_peak_predicted_risk", float("nan")
            ),
            "candidate_rank_lai_pheno_peak_target_risk": candidate_error_stats.get(
                "lai_pheno_peak_target_risk", float("nan")
            ),
            "candidate_rank_fpar_local_predicted_risk": candidate_error_stats.get(
                "fpar_local_predicted_risk", float("nan")
            ),
            "candidate_rank_fpar_local_target_risk": candidate_error_stats.get(
                "fpar_local_target_risk", float("nan")
            ),
            "candidate_rank_fpar_pheno_mean_predicted_risk": candidate_error_stats.get(
                "fpar_pheno_mean_predicted_risk", float("nan")
            ),
            "candidate_rank_fpar_pheno_mean_target_risk": candidate_error_stats.get(
                "fpar_pheno_mean_target_risk", float("nan")
            ),
            "candidate_rank_fpar_pheno_peak_predicted_risk": candidate_error_stats.get(
                "fpar_pheno_peak_predicted_risk", float("nan")
            ),
            "candidate_rank_fpar_pheno_peak_target_risk": candidate_error_stats.get(
                "fpar_pheno_peak_target_risk", float("nan")
            ),
            "solver_acceptance_loss": float(acceptance_loss.detach().cpu()),
            "solver_acceptance_lai": float(
                out.get("solver_acceptance", torch.ones_like(prediction))[:, 0]
                .mean().detach().cpu()
            ),
            "solver_acceptance_fpar": float(
                out.get("solver_acceptance", torch.ones_like(prediction))[:, 1]
                .mean().detach().cpu()
            ),
            "solver_acceptance_oracle_lai": acceptance_stats.get("lai_oracle", float("nan")),
            "solver_acceptance_oracle_fpar": acceptance_stats.get("fpar_oracle", float("nan")),
            "solver_acceptance_mae_lai": acceptance_stats.get("lai_mae", float("nan")),
            "solver_acceptance_mae_fpar": acceptance_stats.get("fpar_mae", float("nan")),
            "solver_acceptance_separation_lai": acceptance_stats.get(
                "lai_separation", float("nan")
            ),
            "solver_acceptance_separation_fpar": acceptance_stats.get(
                "fpar_separation", float("nan")
            ),
            "dynamic_loss": float(dynamic_loss.detach().cpu()),
            "amplitude_loss": float(amplitude_loss.detach().cpu()),
            "memory_gate_loss": float(memory_gate_loss.detach().cpu()),
            "memory_gate_lai": float(
                out.get("phenology_memory_gate", torch.zeros_like(prediction))[:, 0]
                .mean().detach().cpu()
            ),
            "memory_gate_fpar": float(
                out.get("phenology_memory_gate", torch.zeros_like(prediction))[:, 1]
                .mean().detach().cpu()
            ),
            "memory_support_lai": float(
                out.get("phenology_memory_support", torch.zeros_like(prediction))[:, 0]
                .mean().detach().cpu()
            ),
            "memory_support_fpar": float(
                out.get("phenology_memory_support", torch.zeros_like(prediction))[:, 1]
                .mean().detach().cpu()
            ),
            **data_consistency_stats,
            "control_loss": float(control_loss.detach().cpu()),
            "complete_gap_target_fraction": float(
                complete_gap_mask.sum().detach().cpu()
                / target_mask.sum().detach().cpu().clamp_min(1)
            ),
            "st_iterations": st_iterations,
            **gate_stats,
        }
        for key, value in row.items():
            if np.isfinite(value):
                sums[key] = sums.get(key, 0.0) + float(value) * sample_count
        used_samples += sample_count
        forward_batches += 1
        if args.print_every > 0 and batch_idx % args.print_every == 0:
            total = max_batches if max_batches > 0 else len(loader)
            print(
                f"{phase} epoch={epoch} batch={batch_idx}/{total} progress={100*batch_idx/max(total,1):.1f}% "
                f"loss={row['loss']:.6f} rec={row['reconstruction_loss']:.6f} "
                f"pheno_diag={row['phenology_selector_loss']:.5f} "
                f"risk_loss={row['candidate_error_loss']:.5f} "
                f"variable={args.variable} norm_rmse={metrics.get(args.variable.lower() + '_norm_rmse', float('nan')):.5f} "
                f"solver_diag={row['solver_acceptance_loss']:.5f} st_iter={st_iterations:.1f} "
                f"samples={used_samples} elapsed={time.time()-tic:.1f}s",
                flush=True,
            )
        if scenario_quota > 0 and all(
            scenario_counts.get(scenario, 0) >= scenario_quota for scenario in parse_items(args.scenarios)
        ):
            break
    result = {key: value / max(used_samples, 1) for key, value in sums.items()}
    result.update(metrics_accumulator.result())
    result.update(prefixed_metrics("complete_gap", complete_gap_metrics.result()))
    result.update(prefixed_metrics("partial_sensor", partial_sensor_metrics.result()))
    result["batches"] = float(forward_batches)
    result["samples"] = float(used_samples)
    result["skipped_no_target"] = float(skipped_no_target)
    result["skipped_quota"] = float(skipped_quota)
    result["elapsed_sec"] = time.time() - tic
    return result, dict(scenario_counts)


def save_checkpoint(
    path: Path,
    model: QALLMAdaptiveController,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    args: argparse.Namespace,
    solver_config: SolverConfig,
    metrics: dict[str, float],
) -> None:
    trainable_names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    persisted_names = set(getattr(model, "_checkpoint_state_names", trainable_names))
    persisted_names.update(trainable_names)
    compact_state = {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
        if name in persisted_names
    }
    torch.save(
        {
            "version": TRAINING_CHECKPOINT_VERSION,
            "quality_protocol": quality_protocol_metadata(),
            "epoch": epoch,
            "model_state": compact_state,
            "trainable_state_only": True,
            "trainable_parameter_names": sorted(trainable_names),
            "persisted_state_names": sorted(compact_state),
            "model_config": model.config_dict(),
            "solver_config": asdict(solver_config),
            "optimizer_state": optimizer.state_dict(),
            "metrics": metrics,
            "args": vars(args),
        },
        path,
    )


def remap_expert_output_head(
    source_state: dict[str, torch.Tensor],
    source_config: dict[str, object],
    model: QALLMAdaptiveController,
) -> dict[str, torch.Tensor]:
    """Warm-start a four-expert decoder from a three-expert checkpoint."""
    state = dict(source_state)
    old_k = int(source_config.get("num_experts", 3))
    new_k = int(model.config.num_experts)
    if old_k == new_k:
        return state
    if (old_k, new_k) != (3, 4):
        raise RuntimeError(f"Unsupported expert-head remap: {old_k} -> {new_k}")

    target_state = model.state_dict()
    for suffix in ("weight", "bias"):
        key = f"pixel_head.3.{suffix}"
        if key not in state or key not in target_state:
            raise RuntimeError(f"Warm-start checkpoint is missing {key!r}.")
        old = state[key]
        new = target_state[key].detach().clone()
        if old.shape[0] != 2 * old_k + 10 or new.shape[0] != 2 * new_k + 10:
            raise RuntimeError(
                f"Unexpected decoder layout for {key}: {tuple(old.shape)} -> {tuple(new.shape)}"
            )
        for variable in range(2):
            old_start, new_start = variable * old_k, variable * new_k
            new[new_start : new_start + old_k].copy_(old[old_start : old_start + old_k])
            # Initialize PHENO from the mean local gate, with a small negative
            # bias so evidence support must justify its initial selection.
            pheno = old[old_start : old_start + old_k].mean(dim=0)
            if suffix == "bias":
                pheno = pheno - 0.5
            new[new_start + old_k].copy_(pheno)
        new[2 * new_k :].copy_(old[2 * old_k :])
        state[key] = new
    return state


def set_control_training_stage(
    model: QALLMAdaptiveController,
    trainable_scope: str,
    stage: str,
) -> None:
    """Freeze unrelated controls so each calibration stage has one objective."""
    if trainable_scope != "control_heads":
        return
    parent_phenology_prefixes = ("group_gate_correction_head.",)
    reliability_prefixes = (
        "phenology_selector_head.",
        "candidate_error_head.",
        "candidate_rank_head.",
    )
    solver_prefixes = (
        "solver_acceptance_head.",
        "post_solver_acceptance_head.",
    )
    if stage == "reliability":
        active = reliability_prefixes
    elif stage == "solver":
        active = solver_prefixes
    elif stage == "joint":
        active = reliability_prefixes + solver_prefixes
    else:
        raise ValueError(f"Unknown control training stage: {stage}")
    control_prefixes = parent_phenology_prefixes + reliability_prefixes + solver_prefixes
    for name, parameter in model.named_parameters():
        if name.startswith(control_prefixes):
            parameter.requires_grad = name.startswith(active)


def restore_checkpoint_model(
    model: QALLMAdaptiveController,
    path: Path,
    device: torch.device,
) -> dict:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    validate_checkpoint_protocol(checkpoint, budget=False)
    result = model.load_state_dict(checkpoint["model_state"], strict=False)
    control_prefixes = (
        "phenology_selector_head.",
        "group_gate_correction_head.",
        "candidate_error_head.",
        "candidate_rank_head.",
        "solver_acceptance_head.",
        "post_solver_acceptance_head.",
    )
    missing_controls = [
        name for name in result.missing_keys if name.startswith(control_prefixes)
    ]
    unexpected = list(result.unexpected_keys)
    if missing_controls or unexpected:
        raise RuntimeError(
            f"Incompatible staged checkpoint {path}: "
            f"missing_controls={missing_controls}, unexpected={unexpected}"
        )
    return checkpoint


def validate_resume_configuration(
    checkpoint: dict,
    model: QALLMAdaptiveController,
    solver_config: SolverConfig,
) -> None:
    """Reject semantic configuration changes hidden behind compatible shapes."""

    def normalized(value: object) -> object:
        if isinstance(value, (list, tuple)):
            return tuple(normalized(item) for item in value)
        if isinstance(value, dict):
            return {str(key): normalized(item) for key, item in value.items()}
        if isinstance(value, np.generic):
            return value.item()
        return value

    expected_groups = {
        "model_config": model.config_dict(),
        "solver_config": asdict(solver_config),
    }
    mismatches: list[str] = []
    for group_name, expected in expected_groups.items():
        saved = checkpoint.get(group_name)
        if not isinstance(saved, dict):
            mismatches.append(f"{group_name}=missing")
            continue
        for key, expected_value in expected.items():
            if key not in saved:
                mismatches.append(f"{group_name}.{key}=missing")
                continue
            actual_value = saved[key]
            if isinstance(expected_value, float):
                try:
                    equal = abs(float(actual_value) - expected_value) <= 1e-12
                except (TypeError, ValueError):
                    equal = False
            else:
                equal = normalized(actual_value) == normalized(expected_value)
            if not equal:
                mismatches.append(
                    f"{group_name}.{key}: checkpoint={actual_value!r}, "
                    f"current={expected_value!r}"
                )
    if mismatches:
        raise RuntimeError(
            "Resume checkpoint configuration differs from the current run: "
            + "; ".join(mismatches[:12])
        )


def main() -> None:
    args = parse_args()
    if args.trainable_scope != 'from_scratch':
        raise ValueError('Restored independent training only supports from_scratch.')
    args.root = args.root.resolve()
    if args.residual_scale_lai <= 0 or args.residual_scale_fpar <= 0:
        raise ValueError("Residual scales must be positive.")
    if args.amplitude_log_gain_limit <= 0:
        raise ValueError("amplitude_log_gain_limit must be positive.")
    if not 0.0 < args.phenology_amplitude_min <= args.phenology_amplitude_max:
        raise ValueError("Invalid phenology amplitude bounds.")
    if args.phenology_minimum_fit_steps < 2:
        raise ValueError("phenology_minimum_fit_steps must be at least 2.")
    if not 0.0 < args.candidate_risk_temperature <= 0.25:
        raise ValueError("candidate_risk_temperature must be in (0, 0.25].")
    if not 0.0 <= args.candidate_risk_margin <= 0.10:
        raise ValueError("candidate_risk_margin must be in [0, 0.10].")
    if not 0.0 <= args.l1_peak_relaxation < 1.0:
        raise ValueError("l1_peak_relaxation must be in [0, 1).")
    if not 0.0 <= args.saturated_target_weight_floor <= 1.0:
        raise ValueError("saturated_target_weight_floor must be between 0 and 1.")
    if args.value_bin_loss_weight < 0.0:
        raise ValueError("value_bin_loss_weight must be non-negative.")
    if not 0.0 <= args.high_value_selection_weight <= 1.0:
        raise ValueError("high_value_selection_weight must be between 0 and 1.")
    if args.max_high_lai_relative_rmse_increase < 0.0:
        raise ValueError("max_high_lai_relative_rmse_increase must be non-negative.")
    if args.max_high_lai_absolute_bias_increase < 0.0:
        raise ValueError("max_high_lai_absolute_bias_increase must be non-negative.")
    if args.trainable_scope == "control_heads":
        if args.initial_checkpoint is None and args.resume_checkpoint is None:
            raise ValueError(
                "control_heads training requires --initial-checkpoint or --resume-checkpoint."
            )
        if args.pretrain_epochs != 0:
            raise ValueError("control_heads training requires --pretrain-epochs 0.")
        if args.phenology_calibration_epochs != args.epochs:
            raise ValueError(
                "Head-only Module 2 requires phenology_calibration_epochs == epochs; "
                "solver budgeting is calibrated only in Module 2B."
            )
    if args.trainable_scope == "from_scratch":
        if args.backbone != "gpt2":
            raise ValueError(
                "The scientific from_scratch protocol requires the pretrained GPT-2 "
                "backbone. The tiny backbone is restricted to generic component tests."
            )
        if args.initial_checkpoint is not None:
            raise ValueError(
                "from_scratch training prohibits --initial-checkpoint; only the "
                "declared pretrained language-model backbone may be loaded."
            )
        if args.pretrain_epochs != 0:
            raise ValueError(
                "from_scratch training uses the multi-year reliability path for every "
                "epoch and therefore requires --pretrain-epochs 0."
            )
        if args.phenology_calibration_epochs != args.epochs:
            raise ValueError(
                "from_scratch Module 2 trains only the QA-LLM prior; set "
                "phenology_calibration_epochs equal to epochs and calibrate ST/L1 "
                "benefit and budget in Module 2B."
            )
    set_seed(args.seed)
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    device = resolve_device(args.device)
    output = args.root / args.output_subdir / args.variable
    output.mkdir(parents=True, exist_ok=True)

    model_config = ControllerConfig(
        active_variable=args.variable,
        input_channels=args.input_channels,
        backbone=args.backbone,
        gpt_model=args.gpt_model,
        gpt_layers=args.gpt_layers,
        context_dim=args.context_dim,
        local_dim=args.local_dim,
        spatial_grid=args.spatial_grid,
        hierarchy_hidden_dim=args.hierarchy_hidden_dim,
        phenology_phase_radius=args.phenology_phase_radius,
        fit_visible_phenology_template=bool(args.fit_visible_phenology_template),
        phenology_amplitude_min=args.phenology_amplitude_min,
        phenology_amplitude_max=args.phenology_amplitude_max,
        phenology_minimum_fit_steps=args.phenology_minimum_fit_steps,
        residual_scale_lai=args.residual_scale_lai,
        residual_scale_fpar=args.residual_scale_fpar,
        use_amplitude_restoration=False,
        amplitude_log_gain_limit=args.amplitude_log_gain_limit,
        num_experts=len(EXPERT_NAMES),
        use_phenology_memory=False,
        use_hierarchical_phenology_gate=True,
        use_candidate_error_routing=True,
        use_direct_three_candidate_routing=False,
        use_absolute_candidate_risk_routing=True,
        hard_candidate_risk_selection=True,
        candidate_risk_margin=args.candidate_risk_margin,
        candidate_risk_temperature=args.candidate_risk_temperature,
        use_solver_acceptance_gate=False,
        use_post_solver_acceptance_gate=False,
        phenology_selector_bias=args.phenology_selector_bias,
        solver_acceptance_bias=args.solver_acceptance_bias,
        memory_logit_bias=args.memory_logit_bias,
        memory_support_scale_init=args.memory_support_scale_init,
        memory_dispersion_scale_init=args.memory_dispersion_scale_init,
    )
    solver_config = SolverConfig(
        min_iterations=args.solver_min_iterations,
        max_iterations=args.solver_max_iterations,
        training_iterations=args.solver_training_iterations,
        tolerance=args.solver_tolerance,
        rank_cap=args.solver_rank_cap,
        data_fidelity_gain=args.data_fidelity_gain,
        l1_data_fidelity_gain=args.l1_data_fidelity_gain,
        l1_peak_relaxation=args.l1_peak_relaxation,
        residual_phenology_mode=bool(args.residual_phenology_solver),
    )
    model = QALLMAdaptiveController(model_config).to(device)
    inherited_state_names: set[str] = set()
    if args.resume_checkpoint is not None and args.initial_checkpoint is not None:
        raise ValueError("Use either --resume-checkpoint or --initial-checkpoint, not both.")
    if args.initial_checkpoint is not None:
        if not args.initial_checkpoint.is_file():
            raise FileNotFoundError(args.initial_checkpoint)
        initial = torch.load(args.initial_checkpoint, map_location=device, weights_only=False)
        validate_warm_start_protocol(initial)
        recorded_root_text = str(initial.get("args", {}).get("root", "")).strip()
        if not recorded_root_text:
            raise RuntimeError(
                "Stage-A foundation checkpoint does not record its training tile root."
            )
        recorded_root = Path(recorded_root_text).resolve()
        if recorded_root != args.root:
            raise RuntimeError(
                "Cross-tile Stage-A foundation checkpoint rejected: "
                f"checkpoint root={recorded_root}, requested root={args.root}."
            )
        initial_state = remap_expert_output_head(
            initial["model_state"], initial.get("model_config", {}), model
        )
        inherited_state_names = set(initial_state)
        result = model.load_state_dict(initial_state, strict=False)
        parameter_map = dict(model.named_parameters())
        missing_trainable = [
            key for key in result.missing_keys
            if key in parameter_map
            and parameter_map[key].requires_grad
            and not key.startswith("memory_head.")
            and not key.startswith("phenology_selector_head.")
            and not key.startswith("group_gate_correction_head.")
            and not key.startswith("candidate_error_head.")
            and not key.startswith("candidate_rank_head.")
            and not key.startswith("solver_acceptance_head.")
            and not key.startswith("post_solver_acceptance_head.")
            and key not in {"memory_support_scale_raw", "memory_dispersion_scale_raw"}
        ]
        unexpected = [
            key for key in result.unexpected_keys
            if not key.startswith((
                "amplitude_head.", "memory_head.", "phenology_selector_head.",
                "group_gate_correction_head.", "candidate_error_head.",
                "candidate_rank_head.",
                "solver_acceptance_head.",
                "post_solver_acceptance_head.",
            ))
            and key not in {"memory_support_scale_raw", "memory_dispersion_scale_raw"}
        ]
        if missing_trainable or unexpected:
            raise RuntimeError(
                f"Incompatible initial checkpoint; missing={missing_trainable}, "
                f"unexpected={unexpected}"
            )
        old_config = initial.get("model_config", {})
        old_scales = (
            float(old_config.get("residual_scale_lai", model.config.residual_scale_lai)),
            float(old_config.get("residual_scale_fpar", model.config.residual_scale_fpar)),
        )
        new_scales = (
            float(model.config.residual_scale_lai),
            float(model.config.residual_scale_fpar),
        )
        final_pixel_layer = model.pixel_head[-1]
        residual_offset = 2 * model.config.num_experts
        with torch.no_grad():
            for variable_index, (old_scale, new_scale) in enumerate(
                zip(old_scales, new_scales)
            ):
                # Keep the initial physical correction approximately unchanged
                # in the near-linear tanh range while exposing the new capacity.
                ratio = old_scale / new_scale
                output_index = residual_offset + variable_index
                final_pixel_layer.weight[output_index].mul_(ratio)
                final_pixel_layer.bias[output_index].mul_(ratio)
        print(
            "Warm-start residual-logit rescaling: "
            f"LAI={old_scales[0] / new_scales[0]:.4f}, "
            f"FPAR={old_scales[1] / new_scales[1]:.4f}",
            flush=True,
        )
        print(f"Warm-starting compatible weights from: {args.initial_checkpoint}", flush=True)
    if args.trainable_scope == "control_heads":
        control_prefixes = (
            "phenology_selector_head.",
            "group_gate_correction_head.",
            "candidate_error_head.",
            "candidate_rank_head.",
            "solver_acceptance_head.",
            "post_solver_acceptance_head.",
        )
        for name, parameter in model.named_parameters():
            parameter.requires_grad = name.startswith(control_prefixes)
        trainable_names = {
            name for name, parameter in model.named_parameters() if parameter.requires_grad
        }
        if not trainable_names:
            raise RuntimeError("No routing-head parameters were selected for calibration.")
        # The optimizer is head-only, but the child checkpoint must remain a
        # standalone model. Preserve every inherited V48 state entry as well as
        # the updated controls so downstream modules never depend on the parent.
        model._checkpoint_state_names = inherited_state_names | trainable_names
        print(
            "Legacy inherited candidate-risk calibration: "
            f"trainable={sum(p.numel() for p in model.parameters() if p.requires_grad):,} "
            f"persisted_tensors={len(model._checkpoint_state_names)}",
            flush=True,
        )
    elif args.trainable_scope == "from_scratch":
        if args.initial_checkpoint is not None:
            raise RuntimeError("from_scratch training cannot inherit a QA-LLM checkpoint.")
        print(
            f"From-scratch task training: pretrained {args.gpt_model} backbone loaded; all current "
            "trainable remote-sensing projections, Local/PHENO controls, reliability, "
            "risk, residual, and uncertainty layers start from the current architecture.",
            flush=True,
        )
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=args.weight_decay)

    pretrain_train_ds = (
        build_dataset(args, "train", 1, args.max_train_samples_per_scenario)
        if args.pretrain_epochs > 0 else None
    )
    pretrain_val_ds = (
        build_dataset(args, "val", 1, args.max_val_samples_per_scenario)
        if args.pretrain_epochs > 0 else None
    )
    phenology_train_ds = (
        build_dataset(args, "train", args.window_years, args.max_train_samples_per_scenario)
        if args.trainable_scope in {"control_heads", "from_scratch"} else None
    )
    phenology_val_ds = (
        build_dataset(args, "val", args.window_years, args.max_val_samples_per_scenario)
        if args.trainable_scope in {"control_heads", "from_scratch"} else None
    )
    run_solver_stage = (
        args.trainable_scope != "from_scratch"
        and args.phenology_calibration_epochs < args.epochs
    )
    solver_train_ds = (
        build_dataset(
            args, "train", args.window_years, args.max_solver_train_samples_per_scenario
        )
        if run_solver_stage else None
    )
    solver_val_ds = (
        build_dataset(
            args, "val", args.window_years, args.max_solver_val_samples_per_scenario
        )
        if run_solver_stage else None
    )
    loader_args = {"num_workers": args.num_workers, "pin_memory": device.type == "cuda"}
    if args.num_workers > 0:
        loader_args["persistent_workers"] = True
    pretrain_batch_size = max(1, int(args.pretrain_batch_size or args.batch_size))
    solver_batch_size = max(1, int(args.solver_batch_size or args.batch_size))

    def make_loader(dataset: WindowedQADataset, batch_size: int, training: bool) -> DataLoader:
        sampler = TokenGroupedSampler(dataset, args.seed + (0 if training else 1000), training)
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler, **loader_args)

    pretrain_train_loader = (
        make_loader(pretrain_train_ds, pretrain_batch_size, True)
        if pretrain_train_ds is not None else None
    )
    pretrain_val_loader = (
        make_loader(pretrain_val_ds, pretrain_batch_size, False)
        if pretrain_val_ds is not None else None
    )
    phenology_train_loader = (
        make_loader(phenology_train_ds, pretrain_batch_size, True)
        if phenology_train_ds is not None else None
    )
    phenology_val_loader = (
        make_loader(phenology_val_ds, pretrain_batch_size, False)
        if phenology_val_ds is not None else None
    )
    solver_train_loader = (
        make_loader(solver_train_ds, solver_batch_size, True)
        if solver_train_ds is not None else None
    )
    solver_val_loader = (
        make_loader(solver_val_ds, solver_batch_size, False)
        if solver_val_ds is not None else None
    )

    start_epoch = 1
    if args.resume_checkpoint is not None and args.resume_checkpoint.is_file():
        checkpoint = torch.load(args.resume_checkpoint, map_location=device, weights_only=False)
        validate_checkpoint_protocol(checkpoint, budget=False)
        validate_resume_configuration(checkpoint, model, solver_config)
        result = model.load_state_dict(checkpoint["model_state"], strict=False)
        parameter_map = dict(model.named_parameters())
        missing_trainable = [
            key for key in result.missing_keys
            if key in parameter_map and parameter_map[key].requires_grad
        ]
        if missing_trainable or result.unexpected_keys:
            raise RuntimeError(
                f"Incompatible resume checkpoint; missing={missing_trainable}, "
                f"unexpected={result.unexpected_keys}"
            )
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        if args.trainable_scope == "control_heads":
            model._checkpoint_state_names = set(
                checkpoint.get("persisted_state_names", checkpoint["model_state"].keys())
            )
        start_epoch = int(checkpoint["epoch"]) + 1
        print(f"Resuming from epoch {start_epoch}: {args.resume_checkpoint}", flush=True)

    atomic_json(output / "run_config.json", {
        "checkpoint_version": TRAINING_CHECKPOINT_VERSION,
        "quality_protocol": quality_protocol_metadata(),
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "model": asdict(model_config),
        "solver": asdict(solver_config),
        "trainable_parameters": sum(p.numel() for p in trainable),
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "planned_samples": {
            "pretrain_train_per_scenario": args.max_train_samples_per_scenario,
            "pretrain_val_per_scenario": args.max_val_samples_per_scenario,
            "phenology_train_per_scenario": args.max_train_samples_per_scenario,
            "phenology_val_per_scenario": args.max_val_samples_per_scenario,
            "solver_train_per_scenario": args.max_solver_train_samples_per_scenario,
            "solver_val_per_scenario": args.max_solver_val_samples_per_scenario,
        },
        "control_stages": {
            "reliability_epochs": args.phenology_calibration_epochs,
            "solver_acceptance_epochs": (
                args.epochs - args.phenology_calibration_epochs
            ),
            "stage_1_prediction": "residual_consistent_data_consistent_prior",
            "stage_1_solver_iterations": 0,
            "stage_2_prediction": "external_module2b_budget_calibration",
        },
    })
    fields = [
        "epoch", "stage", "phase", "loss", "reconstruction_loss", "value_bin_loss",
        "nll_loss",
        "prior_loss", "gate_loss", "support_loss", "phenology_selector_loss",
        "phenology_selector_lai", "phenology_selector_fpar",
        "phenology_selector_oracle_lai", "phenology_selector_oracle_fpar",
        "phenology_selector_mae_lai", "phenology_selector_mae_fpar",
        "phenology_selector_separation_lai", "phenology_selector_separation_fpar",
        "phenology_selector_lai_ge5", "phenology_selector_oracle_lai_ge5",
        "phenology_selector_mae_lai_ge5", "phenology_selector_lai_ge5_mass",
        "phenology_group_loss", "phenology_group_lai", "phenology_group_fpar",
        "phenology_group_oracle_lai", "phenology_group_oracle_fpar",
        "phenology_group_mae_lai", "phenology_group_mae_fpar",
        "phenology_group_separation_lai", "phenology_group_separation_fpar",
        "phenology_group_lai_ge5", "phenology_group_oracle_lai_ge5",
        "phenology_group_mae_lai_ge5", "phenology_group_lai_ge5_mass",
        "candidate_error_loss", "candidate_error_gate_lai", "candidate_error_gate_fpar",
        "candidate_error_lai_local_predicted", "candidate_error_lai_local_target",
        "candidate_error_lai_pheno_predicted", "candidate_error_lai_pheno_target",
        "candidate_error_lai_ge5_gate", "candidate_error_lai_ge5_local_predicted",
        "candidate_error_lai_ge5_local_target", "candidate_error_lai_ge5_pheno_predicted",
        "candidate_error_lai_ge5_pheno_target",
        "candidate_rank_top1_lai", "candidate_rank_top1_fpar",
        "candidate_rank_lai_local_fraction", "candidate_rank_lai_pheno_mean_fraction",
        "candidate_rank_lai_pheno_peak_fraction", "candidate_rank_fpar_local_fraction",
        "candidate_rank_fpar_pheno_mean_fraction", "candidate_rank_fpar_pheno_peak_fraction",
        "candidate_rank_lai_local_predicted_risk", "candidate_rank_lai_local_target_risk",
        "candidate_rank_lai_pheno_mean_predicted_risk", "candidate_rank_lai_pheno_mean_target_risk",
        "candidate_rank_lai_pheno_peak_predicted_risk", "candidate_rank_lai_pheno_peak_target_risk",
        "candidate_rank_fpar_local_predicted_risk", "candidate_rank_fpar_local_target_risk",
        "candidate_rank_fpar_pheno_mean_predicted_risk", "candidate_rank_fpar_pheno_mean_target_risk",
        "candidate_rank_fpar_pheno_peak_predicted_risk", "candidate_rank_fpar_pheno_peak_target_risk",
        "solver_acceptance_loss", "solver_acceptance_lai", "solver_acceptance_fpar",
        "solver_acceptance_oracle_lai", "solver_acceptance_oracle_fpar",
        "solver_acceptance_mae_lai", "solver_acceptance_mae_fpar",
        "solver_acceptance_separation_lai", "solver_acceptance_separation_fpar",
        "dynamic_loss", "amplitude_loss",
        "memory_gate_loss", "memory_gate_lai", "memory_gate_fpar",
        "memory_support_lai", "memory_support_fpar", "control_loss",
        "observation_consistency_loss", "lai_observation_norm_rmse", "fpar_observation_norm_rmse",
        "gate_entropy", "gate_top1_accuracy", "gate_lin_mean", "gate_idw_mean", "gate_cst_mean",
        "gate_pheno_mean", "complete_gap_target_fraction",
        "norm_rmse_balanced", "norm_rmse_macro", "norm_rmse_micro_balanced",
        "high_value_norm_rmse_balanced", "model_selection_eligible",
        "selection_baseline_lai_ge5_norm_rmse",
        "selection_baseline_lai_ge5_abs_bias", "model_selection_score",
        "lai_norm_rmse", "fpar_norm_rmse",
        "lai_rmse", "fpar_rmse", "lai_mae", "fpar_mae", "lai_bias", "fpar_bias",
        "lai_ge5_rmse", "lai_ge5_norm_rmse", "lai_ge5_mae", "lai_ge5_bias",
        "lai_ge5_weight", "fpar_ge0p9_rmse", "fpar_ge0p9_norm_rmse",
        "fpar_ge0p9_mae", "fpar_ge0p9_bias", "fpar_ge0p9_weight",
        "st_iterations", "samples", "batches", "skipped_no_target", "skipped_quota", "elapsed_sec",
    ]
    domain_metric_suffixes = [
        "norm_rmse_balanced", "norm_rmse_macro", "norm_rmse_micro_balanced",
        "high_value_norm_rmse_balanced", "lai_norm_rmse", "fpar_norm_rmse",
        "lai_rmse", "fpar_rmse", "lai_mae", "fpar_mae", "lai_bias", "fpar_bias",
        "lai_ge5_rmse", "lai_ge5_norm_rmse", "lai_ge5_mae", "lai_ge5_bias",
        "lai_ge5_weight", "fpar_ge0p9_rmse", "fpar_ge0p9_norm_rmse",
        "fpar_ge0p9_mae", "fpar_ge0p9_bias", "fpar_ge0p9_weight",
    ]
    fields.extend(
        f"{domain}_{suffix}"
        for domain in ("complete_gap", "partial_sensor")
        for suffix in domain_metric_suffixes
    )
    log_path = output / "training_log.csv"
    count_path = output / "epoch_scenario_counts.csv"
    best = float("inf")
    best_phenology = float("inf")
    best_path = output / "best.pt"
    best_phenology_path = output / "best_reliability.pt"
    if start_epoch > 1 and best_phenology_path.is_file():
        phenology_checkpoint = torch.load(
            best_phenology_path, map_location="cpu", weights_only=False
        )
        best_phenology = float(
            phenology_checkpoint.get("metrics", {}).get(
                "model_selection_score",
                phenology_checkpoint.get("metrics", {}).get(
                    "norm_rmse_macro", float("inf")
                ),
            )
        )
    if (
        start_epoch > 1
        and best_path.is_file()
        and (
            args.trainable_scope == "from_scratch"
            or start_epoch > args.phenology_calibration_epochs + 1
        )
    ):
        best_checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
        best = float(
            best_checkpoint.get("metrics", {}).get(
                "model_selection_score",
                best_checkpoint.get("metrics", {}).get("norm_rmse_macro", float("inf")),
            )
        )
    no_improvement = 0
    phenology_baseline_path = output / "reliability_selection_baseline.json"
    if args.trainable_scope == "from_scratch":
        # An untrained random task head is not a scientific baseline and must
        # never be exported. Epoch selection starts only after optimization.
        phenology_baseline_metrics: dict[str, float] = {}
        phenology_baseline_high_rmse = float("nan")
        phenology_baseline_high_abs_bias = float("nan")
    else:
        if start_epoch > 1 and phenology_baseline_path.is_file():
            baseline_payload = json.loads(
                phenology_baseline_path.read_text(encoding="utf-8")
            )
            phenology_baseline_metrics = dict(baseline_payload["metrics"])
        else:
            print(
                "Evaluating the inherited data-consistent prior as the reliability safety baseline.",
                flush=True,
            )
            with torch.no_grad():
                phenology_baseline_metrics, _ = run_epoch(
                    model,
                    phenology_val_loader,
                    device,
                    solver_config,
                    args,
                    max(args.pretrain_epochs + 1, 1),
                    None,
                    args.max_val_samples_per_scenario,
                    control_stage="reliability",
                )
        phenology_baseline_high_rmse = float(
            phenology_baseline_metrics.get(
                "complete_gap_lai_ge5_norm_rmse", float("nan")
            )
        )
        phenology_baseline_high_abs_bias = abs(float(
            phenology_baseline_metrics.get("complete_gap_lai_ge5_bias", float("nan"))
        ))
        add_model_selection_metrics(
            phenology_baseline_metrics,
            args,
            phenology_baseline_high_rmse,
            phenology_baseline_high_abs_bias,
        )
        atomic_json(
            phenology_baseline_path, {"metrics": phenology_baseline_metrics}
        )
        if start_epoch == 1:
            best_phenology = float(phenology_baseline_metrics["model_selection_score"])
            save_checkpoint(
                best_phenology_path,
                model,
                optimizer,
                0,
                args,
                solver_config,
                phenology_baseline_metrics,
            )
            save_checkpoint(
                best_path, model, optimizer, 0, args, solver_config, phenology_baseline_metrics
            )
            print(
                "Inherited Local safety baseline: "
                f"LAI>=5 NRMSE={phenology_baseline_high_rmse:.5f}, "
                f"|bias|={phenology_baseline_high_abs_bias:.5f}, "
                f"score={best_phenology:.5f}",
                flush=True,
            )
    solver_baseline_path = output / "solver_selection_baseline.json"
    solver_baseline_metrics: dict[str, float] | None = None
    solver_baseline_high_rmse = float("nan")
    solver_baseline_high_abs_bias = float("nan")
    if (
        start_epoch > args.phenology_calibration_epochs + 1
        and solver_baseline_path.is_file()
    ):
        payload = json.loads(solver_baseline_path.read_text(encoding="utf-8"))
        solver_baseline_metrics = dict(payload["metrics"])
        solver_baseline_high_rmse = float(
            solver_baseline_metrics.get(
                "complete_gap_lai_ge5_norm_rmse", float("nan")
            )
        )
        solver_baseline_high_abs_bias = abs(float(
            solver_baseline_metrics.get("complete_gap_lai_ge5_bias", float("nan"))
        ))
    append = start_epoch > 1 and log_path.is_file()
    with log_path.open("a" if append else "w", newline="", encoding="utf-8") as f, count_path.open(
        "a" if append and count_path.is_file() else "w", newline="", encoding="utf-8"
    ) as count_file:
        writer = csv.DictWriter(f, fieldnames=fields)
        count_writer = csv.DictWriter(
            count_file, fieldnames=["epoch", "stage", "phase", "scenario", "samples"]
        )
        if not append:
            writer.writeheader()
            count_writer.writeheader()
        for epoch in range(start_epoch, args.epochs + 1):
            staged_controls = args.trainable_scope == "control_heads"
            from_scratch_training = args.trainable_scope == "from_scratch"
            if from_scratch_training:
                stage = "from_scratch_multiyear_prior"
                control_stage = "joint"
            elif staged_controls and epoch <= args.phenology_calibration_epochs:
                stage = "reliability_calibration"
                control_stage = "reliability"
            elif staged_controls:
                stage = "solver_acceptance_calibration"
                control_stage = "solver"
            else:
                control_stage = "joint"
                stage = (
                    "prior_pretrain"
                    if epoch <= args.pretrain_epochs
                    else "solver_finetune"
                )

            entering_solver_stage = (
                control_stage == "solver"
                and epoch == args.phenology_calibration_epochs + 1
            )
            if entering_solver_stage:
                if not best_phenology_path.is_file():
                    raise RuntimeError("Stage 2 requires best_reliability.pt from stage 1.")
                phenology_checkpoint = restore_checkpoint_model(
                    model, best_phenology_path, device
                )
                accepted_phenology_epoch = int(
                    phenology_checkpoint.get("epoch", 0)
                )
                if accepted_phenology_epoch <= 0:
                    raise RuntimeError(
                        "Stage 1 produced no accepted reliability update: every "
                        "reliability-calibration epoch was worse than the inherited "
                        "parent or failed the LAI>=5 no-degradation gate. Training "
                        "is stopped instead of silently exporting the inherited "
                        "baseline. Inspect training_log.csv before changing the "
                        "model or data; do not relax the safety gate merely to make "
                        "the run complete."
                    )
                control_parameters = [
                    parameter
                    for name, parameter in model.named_parameters()
                    if name.startswith((
                        "phenology_selector_head.",
                        "group_gate_correction_head.",
                        "solver_acceptance_head.",
                        "post_solver_acceptance_head.",
                    ))
                ]
                optimizer = torch.optim.AdamW(
                    control_parameters,
                    lr=args.learning_rate,
                    weight_decay=args.weight_decay,
                )
                if solver_baseline_metrics is None:
                    print(
                        "Stage 1 accepted at epoch "
                        f"{accepted_phenology_epoch}. Restored best_reliability.pt and "
                        "froze candidate-error routing; "
                        "evaluating its data-consistent prior as the eta safety baseline.",
                        flush=True,
                    )
                    with torch.no_grad():
                        solver_baseline_metrics, _ = run_epoch(
                            model,
                            solver_val_loader,
                            device,
                            solver_config,
                            args,
                            epoch,
                            None,
                            args.max_solver_val_samples_per_scenario,
                            control_stage="reliability",
                        )
                    solver_baseline_high_rmse = float(
                        solver_baseline_metrics.get(
                            "complete_gap_lai_ge5_norm_rmse", float("nan")
                        )
                    )
                    solver_baseline_high_abs_bias = abs(float(
                        solver_baseline_metrics.get(
                            "complete_gap_lai_ge5_bias", float("nan")
                        )
                    ))
                    add_model_selection_metrics(
                        solver_baseline_metrics,
                        args,
                        solver_baseline_high_rmse,
                        solver_baseline_high_abs_bias,
                    )
                    atomic_json(
                        solver_baseline_path, {"metrics": solver_baseline_metrics}
                    )
                best = float(solver_baseline_metrics["model_selection_score"])
                save_checkpoint(
                    best_path,
                    model,
                    optimizer,
                    args.phenology_calibration_epochs,
                    args,
                    solver_config,
                    solver_baseline_metrics,
                )
                no_improvement = 0

            set_control_training_stage(model, args.trainable_scope, control_stage)
            pretraining = (
                not staged_controls
                and not from_scratch_training
                and epoch <= args.pretrain_epochs
            )
            if from_scratch_training:
                if phenology_train_loader is None or phenology_val_loader is None:
                    raise RuntimeError("From-scratch multi-year loaders were not initialized.")
                train_loader, val_loader = phenology_train_loader, phenology_val_loader
                train_quota = args.max_train_samples_per_scenario
                val_quota = args.max_val_samples_per_scenario
            elif pretraining:
                if pretrain_train_loader is None or pretrain_val_loader is None:
                    raise RuntimeError("Pretraining loaders were not initialized.")
                train_loader, val_loader = pretrain_train_loader, pretrain_val_loader
                train_quota = args.max_train_samples_per_scenario
                val_quota = args.max_val_samples_per_scenario
            elif control_stage == "reliability":
                if phenology_train_loader is None or phenology_val_loader is None:
                    raise RuntimeError("Phenology calibration loaders were not initialized.")
                train_loader, val_loader = phenology_train_loader, phenology_val_loader
                train_quota = args.max_train_samples_per_scenario
                val_quota = args.max_val_samples_per_scenario
            else:
                if solver_train_loader is None or solver_val_loader is None:
                    raise RuntimeError("Solver loaders were not initialized.")
                train_loader, val_loader = solver_train_loader, solver_val_loader
                train_quota = args.max_solver_train_samples_per_scenario
                val_quota = args.max_solver_val_samples_per_scenario
            train_metrics, train_counts = run_epoch(
                model,
                train_loader,
                device,
                solver_config,
                args,
                epoch,
                optimizer,
                train_quota,
                control_stage=control_stage,
            )
            with torch.no_grad():
                val_metrics, val_counts = run_epoch(
                    model,
                    val_loader,
                    device,
                    solver_config,
                    args,
                    epoch,
                    None,
                    val_quota,
                    control_stage=control_stage,
                )
            if from_scratch_training:
                selection_high_rmse = float("nan")
                selection_high_abs_bias = float("nan")
            elif control_stage == "reliability":
                selection_high_rmse = phenology_baseline_high_rmse
                selection_high_abs_bias = phenology_baseline_high_abs_bias
            else:
                selection_high_rmse = solver_baseline_high_rmse
                selection_high_abs_bias = solver_baseline_high_abs_bias
            for values in (train_metrics, val_metrics):
                add_model_selection_metrics(
                    values, args, selection_high_rmse, selection_high_abs_bias
                )
            for phase, values in (("train", train_metrics), ("val", val_metrics)):
                payload = {"epoch": epoch, "stage": stage, "phase": phase} | values
                writer.writerow({key: payload.get(key, "") for key in fields})
            for phase, counts in (("train", train_counts), ("val", val_counts)):
                for scenario in parse_items(args.scenarios):
                    count_writer.writerow({
                        "epoch": epoch,
                        "stage": stage,
                        "phase": phase,
                        "scenario": scenario,
                        "samples": counts.get(scenario, 0),
                    })
            f.flush()
            count_file.flush()
            score = val_metrics.get("model_selection_score", float("inf"))
            if not np.isfinite(score):
                selection_prefix = (
                    "complete_gap_"
                    if args.control_target_domain == "complete_gaps"
                    else ""
                )
                macro = val_metrics.get(
                    f"{selection_prefix}norm_rmse_macro", float("nan")
                )
                if (
                    np.isfinite(macro)
                    and val_metrics.get("model_selection_eligible", 0.0) < 0.5
                ):
                    print(
                        "Checkpoint rejected by the LAI>=5 no-degradation gate; "
                        "retaining the inherited/best safe checkpoint.",
                        flush=True,
                    )
                else:
                    raise RuntimeError(
                        "Validation produced no finite complete-gap model-selection metric. "
                        "Check complete_gap_target_fraction and the selected scenarios."
                    )
            save_checkpoint(output / "last.pt", model, optimizer, epoch, args, solver_config, val_metrics)
            save_checkpoint(
                output / f"last_{control_stage}.pt",
                model,
                optimizer,
                epoch,
                args,
                solver_config,
                val_metrics,
            )
            if from_scratch_training:
                if score < best - float(args.early_stop_min_delta):
                    best, no_improvement = score, 0
                    save_checkpoint(
                        best_path, model, optimizer, epoch, args, solver_config, val_metrics
                    )
                else:
                    no_improvement += 1
            elif control_stage == "reliability":
                if score < best_phenology - float(args.early_stop_min_delta):
                    best_phenology = score
                    save_checkpoint(
                        best_phenology_path,
                        model,
                        optimizer,
                        epoch,
                        args,
                        solver_config,
                        val_metrics,
                    )
                    save_checkpoint(
                        best_path,
                        model,
                        optimizer,
                        epoch,
                        args,
                        solver_config,
                        val_metrics,
                    )
            elif pretraining:
                save_checkpoint(
                    output / "last_pretrain.pt", model, optimizer, epoch, args, solver_config, val_metrics
                )
            elif score < best - float(args.early_stop_min_delta):
                best, no_improvement = score, 0
                save_checkpoint(best_path, model, optimizer, epoch, args, solver_config, val_metrics)
            else:
                no_improvement += 1
            if from_scratch_training:
                status = f"best_from_scratch={best:.5f}"
            elif control_stage == "reliability":
                status = f"best_reliability={best_phenology:.5f} best_solver=not_started"
            else:
                status = f"best_solver={best:.5f}"
            print(
                f"epoch={epoch} stage={stage} "
                f"train_macro={train_metrics.get('norm_rmse_macro', float('nan')):.5f} "
                f"val_macro={val_metrics.get('norm_rmse_macro', float('nan')):.5f} "
                f"val_complete_gap={val_metrics.get('complete_gap_norm_rmse_macro', float('nan')):.5f} "
                f"val_complete_high={val_metrics.get('complete_gap_high_value_norm_rmse_balanced', float('nan')):.5f} "
                f"selection={score:.5f} {status}",
                flush=True,
            )
            if (
                (control_stage == "solver" or from_scratch_training)
                and args.early_stop_patience > 0
                and no_improvement >= args.early_stop_patience
            ):
                print(
                    f"Early stopping after {no_improvement} epochs without improvement.",
                    flush=True,
                )
                break
    if args.trainable_scope == "from_scratch":
        if not best_path.is_file():
            raise RuntimeError(
                "From-scratch training produced no finite accepted checkpoint. "
                "Inspect training_log.csv; an untrained model will not be exported."
            )
        selected = torch.load(best_path, map_location="cpu", weights_only=False)
        selected_epoch = int(selected.get("epoch", 0))
        if selected_epoch <= 0:
            raise RuntimeError("From-scratch training cannot select epoch 0.")
        atomic_json(
            output / "training_outcome.json",
            {
                "status": "trained_from_scratch_current_architecture",
                "accepted_update": True,
                "selected_epoch": selected_epoch,
                "selected_checkpoint": str(best_path.resolve()),
                "selection_score": float(
                    selected.get("metrics", {}).get(
                        "model_selection_score", float("nan")
                    )
                ),
                "initial_qa_llm_checkpoint": None,
                "language_model_initialization": args.gpt_model,
                "message": (
                    "The current QA-LLM task architecture was trained directly on "
                    "this tile from the declared pretrained language-model backbone; "
                    "no earlier QA-LLM checkpoint was loaded."
                ),
            },
        )
        print(
            f"From-scratch current-architecture training selected epoch {selected_epoch}.",
            flush=True,
        )
    if args.trainable_scope == "control_heads" and not run_solver_stage:
        accepted = torch.load(best_phenology_path, map_location="cpu", weights_only=False)
        accepted_epoch = int(accepted.get("epoch", 0))
        accepted_update = accepted_epoch > 0
        outcome = {
            "status": (
                "accepted_calibrated_update"
                if accepted_update
                else "safe_baseline_retained"
            ),
            "accepted_update": accepted_update,
            "selected_epoch": accepted_epoch,
            "selected_checkpoint": str(best_path.resolve()),
            "selection_score": float(
                accepted.get("metrics", {}).get("model_selection_score", float("nan"))
            ),
            "message": (
                "A tile-specific candidate-risk update passed all selection gates."
                if accepted_update
                else (
                    "No tile-specific update improved the frozen selection objective "
                    "without violating its safety gates. The valid inherited baseline "
                    "was retained; thresholds were not relaxed."
                )
            ),
        }
        atomic_json(output / "training_outcome.json", outcome)
        if accepted_update:
            print(
                f"Candidate-risk calibration accepted epoch {accepted_epoch}.",
                flush=True,
            )
        else:
            print(
                "Candidate-risk calibration retained the inherited safety baseline. "
                "This is a valid conservative outcome; Module 2B may proceed from "
                "best.pt. Inspect training_log.csv to report the rejected updates.",
                flush=True,
            )
    for dataset in (
        pretrain_train_ds,
        pretrain_val_ds,
        phenology_train_ds,
        phenology_val_ds,
        solver_train_ds,
        solver_val_ds,
    ):
        if dataset is not None:
            dataset.close()
    print(f"Training complete: {output}", flush=True)


if __name__ == "__main__":
    main()
