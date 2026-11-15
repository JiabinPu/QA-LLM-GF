"""End-to-end QA-LLM prior, adaptive ST-Tensor, and L1 pipeline."""

from __future__ import annotations

import math
from contextlib import nullcontext

import torch

from .experts import EXPERT_NAMES, build_window_experts, observed_si
from .solver import SolverConfig, adaptive_l1_trend_filter, adaptive_sttensor
from st_tensor_l1 import refine_encoded_core
from .utils import variable_scale


def aggregate_withheld_target(
    target: torch.Tensor,
    target_mask: torch.Tensor,
    target_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert sensor-specific withheld anchors into one SI target."""
    weights = target_weight * target_mask.to(target_weight.dtype)
    denom = weights.sum(dim=-1)
    value = (target * weights).sum(dim=-1) / denom.clamp_min(1e-8)
    mask = denom > 0
    # Confidence follows the same independent-evidence rule as observed_si().
    # Dividing by the fixed three-sensor capacity incorrectly downweighted the
    # Terra-only early record by 3x and made training depend on mission era.
    confidence = 1.0 - torch.prod(1.0 - weights.clamp(0.0, 1.0), dim=-1)
    confidence = torch.where(mask, confidence, torch.zeros_like(confidence))
    return value, mask, confidence


def apply_observation_fidelity(
    prediction: torch.Tensor,
    observed: torch.Tensor,
    observation_weight: torch.Tensor,
    anchor_value: torch.Tensor,
    anchor_mask: torch.Tensor,
    data_fidelity_gain: float,
) -> torch.Tensor:
    """Project visible cells back toward QA-qualified observations.

    Unsaturated main-algorithm observations are exact fused anchors. Saturated
    main-algorithm observations remain soft evidence, with strength determined
    by their continuous QA weight. Cells without usable observations are left
    unchanged, so the operation cannot inflate genuinely missing peaks.
    """
    if not (
        prediction.shape
        == observed.shape
        == observation_weight.shape
        == anchor_value.shape
        == anchor_mask.shape
    ):
        shapes = {
            "prediction": tuple(prediction.shape),
            "observed": tuple(observed.shape),
            "observation_weight": tuple(observation_weight.shape),
            "anchor_value": tuple(anchor_value.shape),
            "anchor_mask": tuple(anchor_mask.shape),
        }
        raise ValueError(f"Observation-fidelity tensors must have identical shapes: {shapes}")
    weight = float(data_fidelity_gain) * observation_weight.clamp(0.0, 1.0)
    projected = (prediction + weight * observed) / (1.0 + weight).clamp_min(1e-6)
    projected = torch.where(observation_weight > 0.0, projected, prediction)
    return torch.where(anchor_mask, anchor_value, projected)


def apply_candidate_relative_peak_guard(
    solver_acceptance: torch.Tensor,
    phenology_group_gate: torch.Tensor,
    phenology_candidate: torch.Tensor,
    local_candidate: torch.Tensor,
    solver_candidate: torch.Tensor,
    data_consistent_prior: torch.Tensor,
    solver_budget_fraction: torch.Tensor,
    enabled: bool,
    lai_margin: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reject only solver corrections that undo an LLM-selected LAI peak."""
    guarded_acceptance = solver_acceptance.clone()
    guard = torch.zeros_like(solver_acceptance, dtype=torch.bool)
    if not enabled:
        return guarded_acceptance, guard
    lai_guard = (
        (phenology_group_gate[:, 0] >= 0.5)
        & (phenology_candidate[:, 0] > local_candidate[:, 0] + float(lai_margin))
        & (solver_candidate[:, 0] < data_consistent_prior[:, 0] - 1e-6)
        & (solver_budget_fraction[:, 0] > 0.0)
    )
    guard[:, 0] = lai_guard
    guarded_acceptance[:, 0] = torch.where(
        lai_guard,
        torch.zeros_like(guarded_acceptance[:, 0]),
        guarded_acceptance[:, 0],
    )
    return guarded_acceptance, guard


def _to_cores(x: torch.Tensor, core: int) -> tuple[torch.Tensor, tuple[int, ...]]:
    # [B,V,H,W,T,Y] -> [B*V*ncore, core^2,T,Y]
    b, v, h, w, t, y = x.shape
    if h % core or w % core:
        raise ValueError(f"Patch {h}x{w} must be divisible by ST core size {core}.")
    nr, nc = h // core, w // core
    out = x.reshape(b, v, nr, core, nc, core, t, y)
    out = out.permute(0, 1, 2, 4, 3, 5, 6, 7).reshape(b * v * nr * nc, core * core, t, y)
    return out, (b, v, h, w, t, y, nr, nc)


def _from_cores(x: torch.Tensor, shape: tuple[int, ...], core: int) -> torch.Tensor:
    b, v, h, w, t, y, nr, nc = shape
    out = x.reshape(b, v, nr, nc, core, core, t, y)
    return out.permute(0, 1, 2, 4, 3, 5, 6, 7).reshape(b, v, h, w, t, y)


def _phenology_curvature_protection(sequence: torch.Tensor) -> torch.Tensor:
    """Return scale-free second-difference support for peak-aware L1."""
    if sequence.shape[-1] < 3:
        return sequence[..., :0]
    curvature = torch.abs(
        sequence[..., :-2] - 2.0 * sequence[..., 1:-1] + sequence[..., 2:]
    )
    reference = curvature.mean(dim=-1, keepdim=True).detach().clamp_min(1e-4)
    return (curvature / (2.0 * reference)).clamp(0.0, 1.0)


def _phenology_memory(
    x_window: torch.Tensor,
    reliability: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a QA-weighted same-DOY memory from the other years.

    The memory uses only main-algorithm anchor candidates. Saturated retrievals
    remain valid soft evidence, while an available unsaturated retrieval always
    outranks a saturated retrieval from the same year and sensor composite.
    """
    b, years, _channels, time, height, width, sensors = x_window.shape
    values = torch.stack((x_window[:, :, 0], x_window[:, :, 1]), dim=2)
    valid = torch.stack((x_window[:, :, 2] > 0.5, x_window[:, :, 3] > 0.5), dim=2)
    qa = torch.stack((x_window[:, :, 4], x_window[:, :, 5]), dim=2).clamp(0.0, 1.0)
    candidate = torch.stack(
        (x_window[:, :, 6] > 0.5, x_window[:, :, 7] > 0.5), dim=2
    ) & valid
    saturated = (x_window[:, :, 18] > 0.5).unsqueeze(2).expand_as(candidate)
    strict = candidate & ~saturated
    # The QA composite is evidence, not a target for backpropagation. Detaching
    # it keeps gradients on the LLM memory gate and avoids unstable derivatives
    # through zero interannual variance.
    rel = 0.5 + 0.5 * reliability.detach().clamp(0.0, 1.0)
    sensor_weight = candidate.to(values.dtype) * qa.clamp_min(0.02) * rel

    inf = torch.full_like(sensor_weight, float("inf"))
    weakest_strict = torch.where(strict, sensor_weight, inf).amin(dim=-1, keepdim=True)
    has_strict = strict.any(dim=-1, keepdim=True)
    sensor_weight = torch.where(
        saturated & candidate & has_strict,
        torch.minimum(sensor_weight, weakest_strict * (1.0 - 1e-4)),
        sensor_weight,
    )
    year_denom = sensor_weight.sum(dim=-1)
    year_value = (values * sensor_weight).sum(dim=-1) / year_denom.clamp_min(1e-8)
    year_confidence = 1.0 - torch.prod(1.0 - sensor_weight.clamp(0.0, 1.0), dim=-1)

    memory = torch.zeros_like(year_value)
    support = torch.zeros_like(year_confidence)
    dispersion = torch.zeros_like(year_confidence)
    year_index = torch.arange(years, dtype=values.dtype, device=values.device)
    physical_scale = torch.tensor(
        (7.0, 1.0), dtype=values.dtype, device=values.device
    ).view(1, 2, 1, 1, 1)
    for target_year in range(years):
        distance = 1.0 / (1.0 + torch.abs(year_index - float(target_year)))
        distance[target_year] = 0.0
        distance = distance.view(1, years, 1, 1, 1, 1)
        source_weight = year_confidence * distance
        denominator = source_weight.sum(dim=1)
        estimate = (year_value * source_weight).sum(dim=1) / denominator.clamp_min(1e-8)
        variance = (
            (year_value - estimate[:, None]).square() * source_weight
        ).sum(dim=1) / denominator.clamp_min(1e-8)
        confidence = 1.0 - torch.prod(
            1.0 - (year_confidence * distance).clamp(0.0, 1.0), dim=1
        )
        memory[:, target_year] = estimate
        support[:, target_year] = torch.where(
            denominator > 0.0, confidence, torch.zeros_like(confidence)
        )
        dispersion[:, target_year] = torch.where(
            denominator > 0.0,
            torch.sqrt(variance.clamp_min(0.0)) / physical_scale,
            torch.zeros_like(variance),
        )
    return memory, support, dispersion.clamp(0.0, 1.0), year_confidence


def _route_solver_cores(
    anchor_mask_core: torch.Tensor,
    observation_weight_core: torch.Tensor,
    domain_core: torch.Tensor,
    solver_demand: torch.Tensor,
    center_idx: torch.Tensor,
    b: int,
    ncore: int,
    mode: str,
    min_fraction: float | None,
    max_fraction: float | None,
    thresholds: torch.Tensor | None,
    max_selected_fraction: float,
    score_weights: tuple[float, float, float],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select expanded-budget cores using fixed thresholds or a fraction ablation."""
    if mode not in {"threshold", "fraction"}:
        raise ValueError(f"Unsupported solver route mode: {mode!r}")
    if not 0.0 < max_selected_fraction <= 1.0:
        raise ValueError("max_selected_fraction must be in (0, 1].")
    weights = torch.as_tensor(
        score_weights, dtype=anchor_mask_core.dtype, device=anchor_mask_core.device
    )
    if weights.numel() != 3 or bool((weights < 0).any()) or float(weights.sum()) <= 0.0:
        raise ValueError("route score weights must contain three nonnegative values.")
    weights = weights / weights.sum()
    core_center = center_idx[:, None, None].expand(b, 2, ncore).reshape(-1)
    gather_index = core_center.view(-1, 1, 1, 1).expand(
        -1, anchor_mask_core.shape[1], anchor_mask_core.shape[2], 1
    )
    center_anchor = anchor_mask_core.gather(-1, gather_index).squeeze(-1)
    center_weight = observation_weight_core.gather(-1, gather_index).squeeze(-1)
    center_domain = domain_core.gather(-1, gather_index).squeeze(-1)
    unresolved = (~center_anchor) & center_domain
    domain_count = center_domain.flatten(1).sum(dim=1)
    missing_fraction = unresolved.flatten(1).sum(dim=1).to(anchor_mask_core.dtype)
    missing_fraction = missing_fraction / domain_count.clamp_min(1).to(anchor_mask_core.dtype)
    mean_quality = (
        (center_weight * center_domain.to(center_weight.dtype)).flatten(1).sum(dim=1)
        / domain_count.clamp_min(1).to(center_weight.dtype)
    ).clamp(0.0, 1.0)
    needs_solver = unresolved.flatten(1).any(dim=1)
    demand_per_core = solver_demand[:, :, None].expand(b, 2, ncore).reshape(-1)
    scores = (
        weights[0] * demand_per_core
        + weights[1] * missing_fraction
        + weights[2] * (1.0 - mean_quality)
    ).clamp(0.0, 1.0)

    selected = torch.zeros_like(needs_solver)
    demand_flat = solver_demand.reshape(b, 2)
    for bi in range(b):
        for vi in range(2):
            start = (bi * 2 + vi) * ncore
            indices = torch.arange(start, start + ncore, device=anchor_mask_core.device)
            eligible = indices[needs_solver[indices]]
            if eligible.numel() == 0:
                continue
            if mode == "threshold":
                if thresholds is None or thresholds.numel() != 2:
                    raise ValueError("Threshold routing requires LAI and FPAR thresholds.")
                chosen = eligible[scores[eligible] >= thresholds[vi]]
                if chosen.numel() == 0:
                    chosen = eligible[torch.topk(scores[eligible], 1, sorted=False).indices]
                cap = max(1, math.ceil(max_selected_fraction * ncore))
                if chosen.numel() > cap:
                    chosen = chosen[torch.topk(scores[chosen], cap, sorted=False).indices]
            else:
                if min_fraction is None or max_fraction is None:
                    raise ValueError("Fraction routing requires minimum and maximum fractions.")
                if not 0.0 <= min_fraction <= max_fraction <= 1.0:
                    raise ValueError("Solver route fractions must satisfy 0 <= min <= max <= 1.")
                demand = float(demand_flat[bi, vi].detach())
                fraction = min_fraction + demand * (max_fraction - min_fraction)
                count = max(1, math.ceil(fraction * ncore))
                count = min(count, int(eligible.numel()))
                chosen = eligible[torch.topk(scores[eligible], count, sorted=False).indices]
            selected[chosen] = True
    return selected, scores, needs_solver


def _core_needs_solver(
    anchor_mask_core: torch.Tensor,
    domain_core: torch.Tensor,
    center_idx: torch.Tensor,
    b: int,
    ncore: int,
) -> torch.Tensor:
    core_center = center_idx[:, None, None].expand(b, 2, ncore).reshape(-1)
    gather_index = core_center.view(-1, 1, 1, 1).expand(
        -1, anchor_mask_core.shape[1], anchor_mask_core.shape[2], 1
    )
    center_anchor = anchor_mask_core.gather(-1, gather_index).squeeze(-1)
    center_domain = domain_core.gather(-1, gather_index).squeeze(-1)
    return ((~center_anchor) & center_domain).flatten(1).any(dim=1)


def _aggregate_budget_logits(
    logits: torch.Tensor,
    b: int,
    years: int,
    core: int,
) -> torch.Tensor:
    # [B*Y,V,L,T,H,W] -> [B,V,ncore,L]
    _flat, variables, levels, _time, height, width = logits.shape
    nr, nc = height // core, width // core
    mean_map = logits.reshape(b, years, variables, levels, _time, height, width).mean(dim=(1, 4))
    return mean_map.reshape(b, variables, levels, nr, core, nc, core).mean(
        dim=(4, 6)
    ).permute(0, 1, 3, 4, 2).reshape(b, variables, nr * nc, levels)


def _aggregate_budget_features(
    features: torch.Tensor,
    b: int,
    years: int,
    core: int,
) -> torch.Tensor:
    # [B*Y,D,T,H,W] -> [B,ncore,D]
    _flat, channels, _time, height, width = features.shape
    nr, nc = height // core, width // core
    mean_map = features.reshape(b, years, channels, _time, height, width).mean(dim=(1, 3))
    return mean_map.reshape(b, channels, nr, core, nc, core).mean(
        dim=(3, 5)
    ).permute(0, 2, 3, 1).reshape(b, nr * nc, channels)


def _budget_gap_context(
    usable_mask: torch.Tensor,
    anchor_mask: torch.Tensor,
    domain_mask: torch.Tensor,
    center_idx: torch.Tensor,
    core_size: int,
) -> torch.Tensor:
    """Return [B,V,ncore,4] QA-gap descriptors in [0, 1]."""
    b, years, variables, time, height, width = anchor_mask.shape
    nr, nc = height // core_size, width // core_size
    batch_index = torch.arange(b, device=anchor_mask.device)
    center_usable = usable_mask[batch_index, center_idx]
    domain = domain_mask[:, None, None, :, :]
    missing = ((~center_usable) & domain).reshape(
        b, variables, time, nr, core_size, nc, core_size
    ).permute(0, 1, 3, 5, 2, 4, 6).reshape(b, variables, nr * nc, time, -1)
    domain_core = domain_mask.reshape(b, nr, core_size, nc, core_size)
    domain_core = domain_core.permute(0, 1, 3, 2, 4).reshape(b, nr * nc, -1)
    pixel_count = domain_core.sum(dim=-1).clamp_min(1).to(torch.float32)
    missing_by_time = missing.sum(dim=-1).to(torch.float32) / pixel_count[:, None, :, None]
    mean_missing = missing_by_time.mean(dim=-1)
    max_missing = missing_by_time.amax(dim=-1)
    hard_missing = missing_by_time >= 0.5
    current = torch.zeros_like(mean_missing)
    longest = torch.zeros_like(mean_missing)
    for ti in range(time):
        current = torch.where(hard_missing[..., ti], current + 1.0, torch.zeros_like(current))
        longest = torch.maximum(longest, current)
    longest = longest / max(float(time), 1.0)
    support = (anchor_mask & domain_mask[:, None, None, None, :, :]).reshape(
        b, years, variables, time, nr, core_size, nc, core_size
    ).permute(0, 2, 4, 6, 1, 3, 5, 7).reshape(b, variables, nr * nc, -1)
    support = support.to(torch.float32).mean(dim=-1)
    return torch.stack((mean_missing, max_missing, longest, 1.0 - support), dim=-1)


def _vegetation_domain_from_lc(
    x_window: torch.Tensor,
    center_idx: torch.Tensor,
) -> torch.Tensor:
    """Derive the LC_Type_3 vegetation domain from the eight one-hot channels."""
    b, _years, channels, _time, height, width, _sensors = x_window.shape
    if channels < 27:
        return torch.ones((b, height, width), dtype=torch.bool, device=x_window.device)
    batch_index = torch.arange(b, device=x_window.device)
    lc = x_window[batch_index, center_idx, 19:27, 0, :, :, 0]
    return lc.sum(dim=1) > 0.5


def extract_budget_context(
    x_window: torch.Tensor,
    center_index: int | torch.Tensor,
    core_size: int,
    domain_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Extract QA-gap context without running the LLM, ST-Tensor, or L1."""
    b, years, _channels, _time, height, width, _sensors = x_window.shape
    valid = torch.stack((x_window[:, :, 2] > 0.5, x_window[:, :, 3] > 0.5), dim=2)
    anchor = torch.stack((x_window[:, :, 6] > 0.5, x_window[:, :, 7] > 0.5), dim=2)
    anchor_mask = (anchor & valid).any(dim=-1)
    if torch.is_tensor(center_index):
        center_idx = center_index.to(device=x_window.device, dtype=torch.long).reshape(b)
    else:
        center_idx = torch.full((b,), int(center_index), dtype=torch.long, device=x_window.device)
    if domain_mask is None:
        domain = _vegetation_domain_from_lc(x_window, center_idx)
    else:
        domain = domain_mask.to(device=x_window.device, dtype=torch.bool)
    usable_mask = valid.any(dim=-1)
    return _budget_gap_context(usable_mask, anchor_mask, domain, center_idx, core_size)


def structural_gap_cell_mask(
    usable_mask: torch.Tensor,
    domain_mask: torch.Tensor,
    center_idx: torch.Tensor,
    min_run_steps: int = 3,
    max_patch_missing: float = 0.80,
    max_support_deficit: float = 0.95,
) -> torch.Tensor:
    """Return center-year cells belonging to sustained usable-observation gaps."""
    if usable_mask.ndim != 6:
        raise ValueError(
            "usable_mask must have shape [batch,year,variable,time,height,width]."
        )
    b, _years, _variables, time, height, width = usable_mask.shape
    if domain_mask.shape != (b, height, width):
        raise ValueError(f"Expected domain mask {(b, height, width)}.")
    batch_index = torch.arange(b, device=usable_mask.device)
    domain = domain_mask[:, None, None, :, :]
    center_usable = usable_mask[batch_index, center_idx]
    missing = (~center_usable) & domain
    min_steps = max(1, int(min_run_steps))
    sustained = torch.zeros_like(missing)
    if min_steps <= time:
        for start in range(time - min_steps + 1):
            qualifies = missing[:, :, start:start + min_steps].all(dim=2)
            sustained[:, :, start:start + min_steps] |= qualifies[:, :, None]
    domain_pixels = domain_mask.sum(dim=(1, 2)).clamp_min(1).to(torch.float32)
    patch_missing = missing.sum(dim=(2, 3, 4)).to(torch.float32) / (
        domain_pixels[:, None] * float(time)
    )
    supported = usable_mask & domain_mask[:, None, None, None, :, :]
    support = supported.to(torch.float32).mean(dim=(1, 3))
    support_ok = (1.0 - support) < float(max_support_deficit)
    return (
        missing
        & sustained
        & (patch_missing < float(max_patch_missing))[:, :, None, None, None]
        & support_ok[:, :, None, :, :]
    )


def extract_structural_gap_mask(
    x_window: torch.Tensor,
    center_index: int | torch.Tensor,
    domain_mask: torch.Tensor | None = None,
    min_run_steps: int = 3,
    max_patch_missing: float = 0.80,
    max_support_deficit: float = 0.95,
) -> torch.Tensor:
    """Extract sustained gap cells directly from standardized QA channels."""
    b, _years, _channels, _time, height, width, _sensors = x_window.shape
    usable = torch.stack((x_window[:, :, 2] > 0.5, x_window[:, :, 3] > 0.5), dim=2)
    usable = usable.any(dim=-1)
    if torch.is_tensor(center_index):
        center_idx = center_index.to(device=x_window.device, dtype=torch.long).reshape(b)
    else:
        center_idx = torch.full((b,), int(center_index), dtype=torch.long, device=x_window.device)
    domain = (
        _vegetation_domain_from_lc(x_window, center_idx)
        if domain_mask is None
        else domain_mask.to(device=x_window.device, dtype=torch.bool)
    )
    return structural_gap_cell_mask(
        usable,
        domain,
        center_idx,
        min_run_steps,
        max_patch_missing,
        max_support_deficit,
    )


def solver_structural_eligibility(
    contexts: torch.Tensor,
    min_core_missing: float = 0.50,
    min_run_fraction: float = 3.0 / 46.0,
    max_patch_missing: float = 0.80,
    max_support_deficit: float = 0.95,
) -> torch.Tensor:
    """Identify cores with contiguous, locally concentrated QA gaps.

    The final dimension is [mean missing, maximum missing, longest run,
    five-year support deficit]. The patch-wide limit prevents a completely
    unsupported target year from being mistaken for a spatial reconstruction
    problem.
    """
    if contexts.ndim != 4 or contexts.shape[-1] != 4:
        raise ValueError(
            "Structural eligibility expects [batch,variable,core,4] contexts, "
            f"got {tuple(contexts.shape)}."
        )
    mean_missing = contexts[..., 0]
    max_missing = contexts[..., 1]
    longest_run = contexts[..., 2]
    support_deficit = contexts[..., 3]
    patch_missing = mean_missing.mean(dim=-1, keepdim=True)
    return (
        (max_missing >= float(min_core_missing))
        & (longest_run >= float(min_run_fraction))
        & (patch_missing < float(max_patch_missing))
        & (support_deficit < float(max_support_deficit))
    )


def extract_budget_pft(
    x_window: torch.Tensor,
    center_index: int | torch.Tensor,
    core_size: int,
    pft_codes: list[int] | tuple[int, ...],
) -> torch.Tensor:
    """Return the dominant LC_Type_3 code for each [B,ncore] spatial core."""
    b, _years, channels, _time, height, width, _sensors = x_window.shape
    codes = torch.as_tensor(pft_codes, dtype=torch.long, device=x_window.device)
    if codes.numel() == 0:
        return torch.zeros(
            (b, (height // core_size) * (width // core_size)),
            dtype=torch.long,
            device=x_window.device,
        )
    start = 19
    stop = start + codes.numel()
    if channels < stop:
        raise ValueError(f"Expected at least {stop} channels for PFT context, got {channels}.")
    if torch.is_tensor(center_index):
        center_idx = center_index.to(device=x_window.device, dtype=torch.long).reshape(b)
    else:
        center_idx = torch.full((b,), int(center_index), dtype=torch.long, device=x_window.device)
    batch_index = torch.arange(b, device=x_window.device)
    one_hot = x_window[batch_index, center_idx, start:stop, 0, :, :, 0]
    nr, nc = height // core_size, width // core_size
    fractions = one_hot.reshape(b, codes.numel(), nr, core_size, nc, core_size)
    fractions = fractions.mean(dim=(3, 5)).permute(0, 2, 3, 1).reshape(b, nr * nc, -1)
    support, index = fractions.max(dim=-1)
    dominant = codes[index]
    return torch.where(support > 0.0, dominant, torch.zeros_like(dominant))


def prepare_window_frontend(
    model: torch.nn.Module,
    x_window: torch.Tensor,
    return_budget_features: bool = False,
    prepared_annual: dict | None = None,
    production_only: bool = False,
) -> dict[str, object]:
    """Compute the experts and LLM controller once for reuse by solver variants."""
    if x_window.ndim != 7:
        raise ValueError(f"Expected [B,Y,C,T,H,W,S], got {tuple(x_window.shape)}")
    b, years, c, t, h, w, sensors = x_window.shape
    flat_x = x_window.reshape(b * years, c, t, h, w, sensors)
    flat_experts, flat_expert_support, phenology_aux = build_window_experts(
        x_window,
        phenology_phase_radius=int(
            getattr(model.config, "phenology_phase_radius", 1)
        ),
        fit_visible_phenology_template=bool(
            getattr(model.config, "fit_visible_phenology_template", False)
        ),
        phenology_amplitude_min=float(
            getattr(model.config, "phenology_amplitude_min", 0.50)
        ),
        phenology_amplitude_max=float(
            getattr(model.config, "phenology_amplitude_max", 2.00)
        ),
        phenology_minimum_fit_steps=int(
            getattr(model.config, "phenology_minimum_fit_steps", 3)
        ),
        prepared_local=None if prepared_annual is None else prepared_annual["local_experts"],
    )
    phenology_dispersion = phenology_aux["phenology_dispersion"]
    flat_phenology_dispersion = phenology_dispersion.reshape(
        b * years, 2, t, h, w
    )
    model_out = model(
        flat_x,
        (
            flat_experts,
            flat_expert_support,
            phenology_aux["phenology_peak"],
            flat_phenology_dispersion,
        ),
        return_budget_features=return_budget_features,
        **({} if prepared_annual is None else {"prepared_annual": prepared_annual["model"]}),
    )
    solver_phenology_reference = (
        phenology_aux["solver_phenology_mean"]
        + model_out["phenology_selector"]
        * (
            phenology_aux["solver_phenology_peak"]
            - phenology_aux["solver_phenology_mean"]
        )
    )
    # The hierarchical controller replaces the fourth expert's static mean
    # with its learned mean-to-peak phenology candidate.
    flat_experts = model_out["experts"]
    flat_expert_support = model_out["expert_support"]
    required = {
        "prior",
        "residual",
        "fused",
        "amplitude_prior",
        "amplitude_gain",
        "seasonal_baseline",
        "memory_logits",
        "gate_weights",
        "local_gate_weights",
        "local_fused",
        "local_support",
        "phenology_mean",
        "phenology_peak",
        "phenology_selector",
        "phenology_group_gate",
        "parent_phenology_group_gate",
        "candidate_error",
        "candidate_error_gate",
        "candidate_risk",
        "candidate_index",
        "solver_acceptance",
        "reliability",
        "log_variance",
        "difficulty",
        "prior_strength",
        "l1_lambda",
        "continuation",
        "budget_logits",
        "budget_features",
    }
    flat_out = {key: value for key, value in model_out.items() if key in required}
    observations = observed_si(flat_x, model_out["reliability"]) if production_only else None
    return {
        "flat_observations": observations,
        "flat_experts": flat_experts,
        "flat_expert_support": flat_expert_support,
        "phenology_expert_dispersion": phenology_dispersion,
        "phenology_fit_amplitude": phenology_aux["phenology_fit_amplitude"],
        "phenology_fit_phase_shift": phenology_aux["phenology_fit_phase_shift"],
        "solver_phenology_reference": solver_phenology_reference.reshape(
            b, years, 2, t, h, w
        ),
        "solver_phenology_support": phenology_aux["solver_phenology_support"],
        "solver_phenology_dispersion": phenology_aux["solver_phenology_dispersion"],
        "flat_out": flat_out,
    }


def run_reliability_window(
    model: torch.nn.Module,
    x_window: torch.Tensor,
    center_index: int | torch.Tensor,
    data_fidelity_gain: float,
) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
    """Run the center-year fusion contract without invoking ST-Tensor/L1.

    Module 2 trains or evaluates the current QA-LLM prior without invoking the
    numerical solver. Running the controller for every year and then executing
    zero-budget solvers is numerically redundant. This path builds the same
    multi-year PHENO expert but evaluates the LLM/controller only for the
    requested center year.
    """
    if x_window.ndim != 7:
        raise ValueError(f"Expected [B,Y,C,T,H,W,S], got {tuple(x_window.shape)}")
    b, years, _c, t, h, w, _s = x_window.shape
    if torch.is_tensor(center_index):
        center_idx = center_index.to(device=x_window.device, dtype=torch.long).reshape(b)
    else:
        center_idx = torch.full(
            (b,), int(center_index), dtype=torch.long, device=x_window.device
        )
    if bool(((center_idx < 0) | (center_idx >= years)).any()):
        raise ValueError("center_index is outside the supplied year window.")
    batch_index = torch.arange(b, device=x_window.device)
    flat_index = batch_index * years + center_idx
    center_x = x_window[batch_index, center_idx]

    flat_experts, flat_support, phenology_aux = build_window_experts(
        x_window,
        phenology_phase_radius=int(getattr(model.config, "phenology_phase_radius", 1)),
        fit_visible_phenology_template=bool(
            getattr(model.config, "fit_visible_phenology_template", False)
        ),
        phenology_amplitude_min=float(
            getattr(model.config, "phenology_amplitude_min", 0.50)
        ),
        phenology_amplitude_max=float(
            getattr(model.config, "phenology_amplitude_max", 2.00)
        ),
        phenology_minimum_fit_steps=int(
            getattr(model.config, "phenology_minimum_fit_steps", 3)
        ),
    )
    center_experts = flat_experts.index_select(0, flat_index)
    center_support = flat_support.index_select(0, flat_index)
    center_peak = phenology_aux["phenology_peak"].index_select(0, flat_index)
    center_dispersion = phenology_aux["phenology_dispersion"][batch_index, center_idx]
    out = model(
        center_x,
        (center_experts, center_support, center_peak, center_dispersion),
    )
    observed, observation_weight, anchor, anchor_mask = observed_si(
        center_x, out["reliability"]
    )
    data_consistent_prior = apply_observation_fidelity(
        out["prior"],
        observed,
        observation_weight,
        anchor,
        anchor_mask,
        data_fidelity_gain,
    )

    residual = out["prior"] - out["amplitude_prior"]

    def prior_endpoint(candidate: torch.Tensor) -> torch.Tensor:
        seasonal = candidate.mean(dim=2, keepdim=True)
        amplitude = seasonal + out["amplitude_gain"] * (candidate - seasonal)
        endpoint = amplitude + residual
        return torch.stack(
            [endpoint[:, 0].clamp(0.0, 7.0), endpoint[:, 1].clamp(0.0, 1.0)],
            dim=1,
        )

    local_endpoint = prior_endpoint(out["local_fused"])
    pheno_mean_endpoint = prior_endpoint(out["phenology_mean"])
    pheno_peak_endpoint = prior_endpoint(out["phenology_peak"])
    pheno_endpoint = prior_endpoint(out["experts"][:, :, 3])
    zero_iterations = torch.zeros(
        (b * 2,), dtype=out["prior"].dtype, device=out["prior"].device
    )
    fit_amplitude = phenology_aux["phenology_fit_amplitude"][batch_index, center_idx]
    fit_phase = phenology_aux["phenology_fit_phase_shift"][batch_index, center_idx]
    solver_support = phenology_aux["solver_phenology_support"][batch_index, center_idx]
    solver_dispersion = phenology_aux["solver_phenology_dispersion"][batch_index, center_idx]
    return {
        "prediction": data_consistent_prior,
        "solver_candidate": data_consistent_prior,
        "solver_acceptance": out["solver_acceptance"],
        "pre_solver_acceptance": out["solver_acceptance"],
        "st_prediction": data_consistent_prior,
        "prior": out["prior"],
        "base_prior": out["prior"],
        "data_consistent_prior": data_consistent_prior,
        "fused": out["fused"],
        "amplitude_prior": out["amplitude_prior"],
        "amplitude_gain": out["amplitude_gain"],
        "residual": residual,
        "observed": observed,
        "observation_weight": observation_weight,
        "anchor_value": anchor,
        "anchor_mask": anchor_mask,
        "experts": out["experts"],
        "expert_support": out["expert_support"],
        "gate_weights": out["gate_weights"],
        "local_gate_weights": out["local_gate_weights"],
        "local_fused": out["local_fused"],
        "local_support": out["local_support"],
        "local_candidate_endpoint": local_endpoint,
        "phenology_mean": out["phenology_mean"],
        "phenology_peak": out["phenology_peak"],
        "phenology_mean_candidate_endpoint": pheno_mean_endpoint,
        "phenology_peak_candidate_endpoint": pheno_peak_endpoint,
        "phenology_candidate_endpoint": pheno_endpoint,
        "phenology_selector": out["phenology_selector"],
        "phenology_group_gate": out["phenology_group_gate"],
        "parent_phenology_group_gate": out["parent_phenology_group_gate"],
        "candidate_error": out["candidate_error"],
        "candidate_error_gate": out["candidate_error_gate"],
        "candidate_risk": out["candidate_risk"],
        "candidate_index": out["candidate_index"],
        "phenology_memory": out["experts"][:, :, 3],
        "phenology_memory_support": out["expert_support"][:, :, 3],
        "phenology_memory_dispersion": center_dispersion,
        "phenology_memory_gate": out["phenology_group_gate"],
        "phenology_expert_dispersion": center_dispersion,
        "phenology_fit_amplitude": fit_amplitude,
        "phenology_fit_phase_shift": fit_phase,
        "solver_phenology_support": solver_support,
        "solver_phenology_dispersion": solver_dispersion,
        "log_variance": out["log_variance"],
        "difficulty": out["difficulty"],
        "continuation": out["continuation"],
        "l1_lambda": out["l1_lambda"],
        "prior_strength": out["prior_strength"],
        "reliability": out["reliability"],
        "st_diagnostics": {"iterations": zero_iterations},
        "l1_diagnostics": {"iterations": zero_iterations},
    }


def hierarchical_uncertainty_components(
    result: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Decompose physical-scale uncertainty for the hierarchical fusion."""
    reference = result["prediction"]
    physical_scale = torch.tensor(
        (7.0, 1.0), dtype=reference.dtype, device=reference.device
    ).view(1, 2, 1, 1, 1)
    data = torch.exp(0.5 * result["log_variance"]).clamp_max(2.0) * physical_scale
    local_values = result["experts"][:, :, :3]
    local_weights = result["local_gate_weights"]
    local_center = result["local_fused"].unsqueeze(2)
    local_variance = (
        local_weights * (local_values - local_center).square()
    ).sum(dim=2).clamp_min(0.0)
    alpha = result["phenology_group_gate"].clamp(0.0, 1.0)
    within_local = torch.sqrt(((1.0 - alpha) * local_variance).clamp_min(0.0))
    between_groups = torch.sqrt(
        (
            alpha
            * (1.0 - alpha)
            * (result["phenology_memory"] - result["local_fused"]).square()
        ).clamp_min(0.0)
    )
    candidate_risk = result.get("candidate_error")
    if candidate_risk is None:
        selected_risk = torch.zeros_like(reference)
    else:
        selected_risk_normalized = torch.where(
            alpha >= 0.5,
            candidate_risk[:, :, 1],
            candidate_risk[:, :, 0],
        )
        selected_risk = selected_risk_normalized * physical_scale
    # The solver term represents only the accepted ST-Tensor+L1 adjustment;
    # QA observation projection is already part of the data-consistent prior.
    solver = (result["prediction"] - result["data_consistent_prior"]).abs()
    total = torch.sqrt(
        data.square()
        + selected_risk.square()
        + within_local.square()
        + between_groups.square()
        + solver.square()
    )
    return {
        "data": data,
        "selected_candidate_risk": selected_risk,
        "within_local": within_local,
        "between_groups": between_groups,
        "solver": solver,
        "total": total,
    }


def select_prepared_frontend(
    prepared_frontend: dict[str, object],
    batch_indices: torch.Tensor,
    years: int,
) -> dict[str, object]:
    """Select batch members from frontend tensors flattened as [B*Y,...]."""
    indices = batch_indices.to(dtype=torch.long)
    year_offsets = torch.arange(int(years), device=indices.device, dtype=torch.long)
    flat_indices = (indices[:, None] * int(years) + year_offsets[None, :]).reshape(-1)

    def select(value: object) -> object:
        if torch.is_tensor(value):
            return value.index_select(0, flat_indices.to(value.device))
        return value

    flat_out = prepared_frontend.get("flat_out")
    if not isinstance(flat_out, dict):
        raise TypeError("prepared_frontend['flat_out'] must be a dictionary.")
    return {
        "flat_experts": select(prepared_frontend["flat_experts"]),
        "flat_expert_support": select(prepared_frontend["flat_expert_support"]),
        "phenology_expert_dispersion": prepared_frontend["phenology_expert_dispersion"].index_select(
            0, indices.to(prepared_frontend["phenology_expert_dispersion"].device)
        ),
        "phenology_fit_amplitude": prepared_frontend["phenology_fit_amplitude"].index_select(
            0, indices.to(prepared_frontend["phenology_fit_amplitude"].device)
        ),
        "phenology_fit_phase_shift": prepared_frontend["phenology_fit_phase_shift"].index_select(
            0, indices.to(prepared_frontend["phenology_fit_phase_shift"].device)
        ),
        "solver_phenology_reference": prepared_frontend["solver_phenology_reference"].index_select(
            0, indices.to(prepared_frontend["solver_phenology_reference"].device)
        ),
        "solver_phenology_support": prepared_frontend["solver_phenology_support"].index_select(
            0, indices.to(prepared_frontend["solver_phenology_support"].device)
        ),
        "solver_phenology_dispersion": prepared_frontend["solver_phenology_dispersion"].index_select(
            0, indices.to(prepared_frontend["solver_phenology_dispersion"].device)
        ),
        "flat_out": {key: select(value) for key, value in flat_out.items()},
    }


def run_window(
    model: torch.nn.Module,
    x_window: torch.Tensor,
    center_index: int | torch.Tensor,
    solver_config: SolverConfig,
    core_size: int = 8,
    training_solver: bool = False,
    expert_mask: torch.Tensor | list[float] | tuple[float, ...] | None = None,
    route_mode: str | None = None,
    route_min_fraction: float | None = None,
    route_max_fraction: float | None = None,
    route_thresholds: torch.Tensor | list[float] | tuple[float, ...] | None = None,
    route_max_selected_fraction: float = 0.80,
    route_score_weights: tuple[float, float, float] = (0.50, 0.30, 0.20),
    domain_mask: torch.Tensor | None = None,
    budget_levels: torch.Tensor | list[int] | tuple[int, ...] | None = None,
    force_full_budget: bool = False,
    disable_solver: bool = False,
    structural_cell_blend: bool = False,
    return_budget_features: bool = False,
    prepared_frontend: dict[str, object] | None = None,
    return_prepared_frontend: bool = False,
    production_only: bool = False,
) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
    """Run a fixed multi-year window. x_window is [B,Y,C,T,H,W,S]."""
    if production_only and (training_solver or model.training or torch.is_grad_enabled()):
        raise RuntimeError("Production-only execution requires gradient-free evaluation.")
    if production_only and (return_budget_features or return_prepared_frontend or expert_mask is not None):
        raise ValueError("Production-only execution does not support training/ablation outputs.")
    if x_window.ndim != 7:
        raise ValueError(f"Expected [B,Y,C,T,H,W,S], got {tuple(x_window.shape)}")
    b, years, c, t, h, w, sensors = x_window.shape
    flat_x = x_window.reshape(b * years, c, t, h, w, sensors)
    if prepared_frontend is None:
        prepared_frontend = prepare_window_frontend(
            model, x_window, return_budget_features=return_budget_features,
            production_only=production_only,
        )
    if prepared_frontend is not None:
        expected_shape = (b * years, 2, len(EXPERT_NAMES), t, h, w)
        flat_experts = prepared_frontend["flat_experts"]
        flat_expert_support = prepared_frontend["flat_expert_support"]
        base_flat_out = prepared_frontend["flat_out"]
        if not torch.is_tensor(flat_experts) or tuple(flat_experts.shape) != expected_shape:
            raise ValueError("prepared_frontend does not match the current input window.")
        flat_out = dict(base_flat_out)
        if return_budget_features and "budget_features" not in flat_out:
            raise ValueError("Prepared frontend has no budget features.")

    if expert_mask is not None:
        mask = torch.as_tensor(expert_mask, dtype=flat_x.dtype, device=flat_x.device)
        if mask.numel() != flat_experts.shape[2] or not bool((mask > 0).any()):
            raise ValueError("expert_mask must select at least one available expert.")
        mask = mask.view(1, 1, -1, 1, 1, 1)
        gates = flat_out["gate_weights"] * mask
        gates = gates / gates.sum(dim=2, keepdim=True).clamp_min(1e-8)
        flat_out["gate_weights"] = gates
        masked_fused = (gates * flat_experts).sum(dim=2)
        masked_baseline = masked_fused.mean(dim=2, keepdim=True)
        masked_amplitude_prior = masked_baseline + flat_out["amplitude_gain"] * (
            masked_fused - masked_baseline
        )
        masked_prior = masked_amplitude_prior + flat_out["residual"]
        flat_out["fused"] = masked_fused
        flat_out["seasonal_baseline"] = masked_baseline
        flat_out["amplitude_prior"] = masked_amplitude_prior
        flat_out["prior"] = torch.stack(
            [masked_prior[:, 0].clamp(0.0, 7.0), masked_prior[:, 1].clamp(0.0, 1.0)], dim=1
        )

    base_prior = flat_out["prior"].reshape(b, years, 2, t, h, w)
    phenology_reference = prepared_frontend["solver_phenology_reference"]
    if tuple(phenology_reference.shape) != (b, years, 2, t, h, w):
        raise ValueError("Prepared solver phenology reference has an invalid shape.")
    reliability = flat_out["reliability"].reshape(b, years, 2, t, h, w, sensors)
    observations = prepared_frontend.get("flat_observations")
    if observations is None:
        observations = observed_si(flat_x, flat_out["reliability"])
    obs_flat, obs_weight_flat, anchor_flat, anchor_mask_flat = observations
    observed = obs_flat.reshape(b, years, 2, t, h, w)
    obs_weight_window = obs_weight_flat.reshape(b, years, 2, t, h, w)
    usable_mask_window = obs_weight_window > 0.0
    anchor = anchor_flat.reshape(b, years, 2, t, h, w)
    anchor_mask_window = anchor_mask_flat.reshape(b, years, 2, t, h, w)

    if bool(getattr(model.config, "use_phenology_memory", False)):
        raise RuntimeError(
            "Post-fusion phenology memory is retired in V56. "
            "Use the explicit PHENO expert instead."
        )
    memory = base_prior
    memory_support = torch.zeros_like(base_prior)
    memory_dispersion = torch.zeros_like(base_prior)
    memory_gate = torch.zeros_like(base_prior)
    prior = base_prior

    # Routing only needs QA weights and masks, not normalized solver values.
    obs_weight_n = obs_weight_window.permute(0, 2, 4, 5, 3, 1).contiguous()
    anchor_mask = anchor_mask_window.permute(0, 2, 4, 5, 3, 1).contiguous()
    weight_core, _ = _to_cores(obs_weight_n, core_size)
    anchor_mask_core, core_shape = _to_cores(anchor_mask, core_size)
    ncore = core_shape[-2] * core_shape[-1]

    if torch.is_tensor(center_index):
        center_idx = center_index.to(device=x_window.device, dtype=torch.long).reshape(b)
    else:
        center_idx = torch.full((b,), int(center_index), dtype=torch.long, device=x_window.device)

    difficulty = flat_out["difficulty"].reshape(b, years, 2).amax(dim=1)
    continuation = flat_out["continuation"].reshape(b, years, 2).mean(dim=1)
    solver_demand = (0.6 * difficulty + 0.4 * continuation).clamp(0.0, 1.0)
    prior_strength = flat_out["prior_strength"].reshape(b, years, 2).mean(dim=1)
    l1_lambda = flat_out["l1_lambda"].reshape(b, years, 2).mean(dim=1)
    repeat = lambda z: z[:, :, None].expand(b, 2, ncore).reshape(-1)

    if domain_mask is None:
        domain = _vegetation_domain_from_lc(x_window, center_idx)
    else:
        domain = domain_mask.to(device=x_window.device, dtype=torch.bool)
        if domain.shape != (b, h, w):
            raise ValueError(f"Expected domain_mask {(b, h, w)}, got {tuple(domain.shape)}")
    domain_n = domain[:, None, :, :, None, None].expand(b, 2, h, w, t, years)
    domain_core, _ = _to_cores(domain_n, core_size)
    needs_solver = _core_needs_solver(anchor_mask_core, domain_core, center_idx, b, ncore)
    routed = route_mode is not None
    budget_logits_core = None
    budget_class = None
    budget_feature_core = None
    adaptive_budget = None
    llm_budget = None
    solver_benefit_probability = None
    solver_structural_eligible = None
    solver_structural_cells = None

    use_cell_selective_routing = bool(structural_cell_blend) or (
        route_mode == "learned_budget"
        and bool(getattr(model.config, "budget_structural_cell_selective", False))
    )
    if use_cell_selective_routing:
        solver_structural_cells = structural_gap_cell_mask(
            usable_mask_window,
            domain,
            center_idx,
            model.config.budget_structural_min_run_steps,
            model.config.budget_structural_max_patch_missing,
            model.config.budget_structural_max_support_deficit,
        )
        nr, nc = h // core_size, w // core_size
        solver_structural_eligible = solver_structural_cells.reshape(
            b, 2, t, nr, core_size, nc, core_size
        ).permute(0, 1, 3, 5, 2, 4, 6).reshape(b, 2, nr * nc, -1).any(dim=-1)

    if disable_solver:
        # Stage-1 control calibration only needs the differentiable expert
        # frontend and the data-consistent prior. A zero budget avoids the
        # expensive ST-Tensor/L1 iterations and prevents solver errors from
        # contaminating gamma/alpha checkpoint selection.
        st_budget = torch.zeros_like(needs_solver, dtype=torch.long)
        selected = torch.zeros_like(needs_solver, dtype=torch.bool)
        route_scores = torch.zeros_like(needs_solver, dtype=prior.dtype)
    elif force_full_budget:
        st_budget = torch.where(
            needs_solver,
            torch.full_like(needs_solver, int(solver_config.max_iterations), dtype=torch.long),
            torch.zeros_like(needs_solver, dtype=torch.long),
        )
        selected = needs_solver.clone()
        route_scores = selected.to(prior.dtype)
    elif route_mode == "learned_budget":
        if training_solver:
            raise ValueError("Learned solver budgeting is an inference-only option.")
        if "budget_logits" not in flat_out:
            raise RuntimeError("The checkpoint does not contain an LLM solver budget head.")
        levels = torch.as_tensor(
            budget_levels if budget_levels is not None else (0, 3, 5, 8, 13, 24),
            dtype=torch.long,
            device=prior.device,
        )
        if levels.numel() != flat_out["budget_logits"].shape[2]:
            raise ValueError("budget_levels must match the number of LLM budget classes.")
        budget_logits_core = _aggregate_budget_logits(flat_out["budget_logits"], b, years, core_size)
        budget_context_core = _budget_gap_context(
            usable_mask_window, anchor_mask_window, domain, center_idx, core_size
        )
        if solver_structural_eligible is None:
            solver_structural_eligible = solver_structural_eligibility(
                budget_context_core,
                model.config.budget_structural_min_core_missing,
                model.config.budget_structural_min_run_fraction,
                model.config.budget_structural_max_patch_missing,
                model.config.budget_structural_max_support_deficit,
            )
        budget_logits_core = model.add_budget_context(
            budget_logits_core, budget_context_core
        )
        probabilities = torch.softmax(budget_logits_core, dim=-1)
        global_quantiles = torch.as_tensor(
            model.config.budget_quantiles,
            dtype=probabilities.dtype,
            device=probabilities.device,
        ).view(1, 2, 1)
        global_need_thresholds = torch.as_tensor(
            model.config.budget_need_thresholds,
            dtype=probabilities.dtype,
            device=probabilities.device,
        ).view(1, 2, 1)
        quantiles = global_quantiles.expand(b, 2, ncore).clone()
        need_thresholds = global_need_thresholds.expand(b, 2, ncore).clone()
        pft_codes = tuple(model.config.budget_pft_codes)
        pft_quantiles = model.config.budget_pft_quantiles
        pft_need_thresholds = model.config.budget_pft_need_thresholds
        if pft_codes and len(pft_quantiles) == 2 and all(
            len(row) == len(pft_codes) for row in pft_quantiles
        ) and len(pft_need_thresholds) == 2 and all(
            len(row) == len(pft_codes) for row in pft_need_thresholds
        ):
            dominant_pft = extract_budget_pft(
                x_window, center_idx, core_size, pft_codes
            )
            pft_q = torch.as_tensor(
                pft_quantiles, dtype=probabilities.dtype, device=probabilities.device
            )
            pft_need = torch.as_tensor(
                pft_need_thresholds,
                dtype=probabilities.dtype,
                device=probabilities.device,
            )
            for pi, code in enumerate(pft_codes):
                use = dominant_pft == int(code)
                quantiles = torch.where(
                    use[:, None, :], pft_q[:, pi].view(1, 2, 1), quantiles
                )
                need_thresholds = torch.where(
                    use[:, None, :],
                    pft_need[:, pi].view(1, 2, 1),
                    need_thresholds,
                )
        need_probability = 1.0 - probabilities[..., 0]
        solver_benefit_probability = need_probability
        conditional = probabilities[..., 1:] / need_probability.unsqueeze(-1).clamp_min(1e-8)
        positive_class = (
            conditional.cumsum(dim=-1) < quantiles.unsqueeze(-1)
        ).sum(dim=-1)
        positive_class = positive_class.clamp_max(levels.numel() - 2) + 1
        budget_class = torch.where(
            need_probability >= need_thresholds, positive_class, 0
        )
        if not 0.0 <= float(route_max_selected_fraction) <= 1.0:
            raise ValueError("route_max_selected_fraction must be between 0 and 1.")
        max_selected = int(math.floor(float(route_max_selected_fraction) * ncore))
        candidate = (
            (budget_class > 0)
            & needs_solver.reshape(b, 2, ncore)
            & solver_structural_eligible
        )
        if max_selected < ncore:
            keep = torch.zeros_like(candidate)
            for batch_index in range(b):
                for variable_index in range(2):
                    indices = torch.nonzero(
                        candidate[batch_index, variable_index], as_tuple=False
                    ).flatten()
                    if indices.numel() == 0 or max_selected <= 0:
                        continue
                    count = min(max_selected, int(indices.numel()))
                    scores = need_probability[batch_index, variable_index, indices]
                    chosen = indices[torch.topk(scores, k=count, largest=True).indices]
                    keep[batch_index, variable_index, chosen] = True
            budget_class = torch.where(keep, budget_class, torch.zeros_like(budget_class))
        llm_budget = levels[budget_class].reshape(-1)
        adaptive_budget = torch.round(
            solver_config.min_iterations
            + repeat(solver_demand)
            * (solver_config.max_iterations - solver_config.min_iterations)
        ).to(torch.long)
        # The learned head predicts a prior-relative solver-benefit decision
        # and a conditional upper budget. Budget zero preserves the
        # hierarchical phenology prior before the shared final QA projection and skips
        # both ST-Tensor and L1.
        st_budget = torch.minimum(adaptive_budget, llm_budget)
        st_budget = torch.where(needs_solver, st_budget, torch.zeros_like(st_budget))
        selected = needs_solver & (st_budget > 0)
        route_scores = (
            probabilities * levels.to(probabilities.dtype).view(1, 1, 1, -1)
        ).sum(dim=-1).reshape(-1) / max(float(solver_config.max_iterations), 1.0)
    elif routed:
        if training_solver:
            raise ValueError("LLM solver routing is an inference-only production option.")
        thresholds = (
            torch.as_tensor(route_thresholds, dtype=prior.dtype, device=prior.device)
            if route_thresholds is not None else None
        )
        selected, route_scores, needs_solver = _route_solver_cores(
            anchor_mask_core,
            weight_core,
            domain_core,
            solver_demand,
            center_idx,
            b,
            ncore,
            route_mode,
            route_min_fraction,
            route_max_fraction,
            thresholds,
            route_max_selected_fraction,
            route_score_weights,
        )
        full_budget = torch.round(
            solver_config.min_iterations
            + repeat(solver_demand) * (solver_config.max_iterations - solver_config.min_iterations)
        ).to(torch.long)
        st_budget = torch.where(
            needs_solver,
            torch.where(selected, full_budget, torch.zeros_like(full_budget)),
            torch.zeros_like(full_budget),
        )
    else:
        if training_solver:
            base_budget = torch.full_like(
                needs_solver, int(solver_config.training_iterations), dtype=torch.long
            )
        else:
            base_budget = torch.round(
                solver_config.min_iterations
                + repeat(solver_demand) * (solver_config.max_iterations - solver_config.min_iterations)
            ).to(torch.long)
        st_budget = torch.where(needs_solver, base_budget, torch.zeros_like(base_budget))
        selected = needs_solver.clone()
        route_scores = (
            st_budget.to(prior.dtype) / max(float(solver_config.max_iterations), 1.0)
        )

    def gather_year(tensor: torch.Tensor) -> torch.Tensor:
        # Keep the batch dimension explicit when edge windows have different centres.
        view = center_idx.view(b, 1, *([1] * (tensor.ndim - 2)))
        index = view.expand(b, 1, *tensor.shape[2:])
        return tensor.gather(1, index).squeeze(1)
    # Only cell-selective routing guarantees that no selected core means the
    # exact center prior before the unchanged observation projection.
    zero_budget_fastpath = (
        production_only and use_cell_selective_routing
        and not bool(selected.any()) and not bool((st_budget > 0).any())
    )
    if zero_budget_fastpath:
        center_prediction = gather_year(prior)
        st_diag = {"iterations": torch.zeros_like(st_budget), "budget": st_budget}
        l1_diag = {"iterations": torch.zeros_like(st_budget), "budget": torch.zeros_like(st_budget)}
    else:
        prior_n = prior.permute(0, 2, 4, 5, 3, 1).contiguous()
        reference_n = phenology_reference.permute(0, 2, 4, 5, 3, 1).contiguous()
        observed_n = observed.permute(0, 2, 4, 5, 3, 1).contiguous()
        anchor_n = anchor.permute(0, 2, 4, 5, 3, 1).contiguous()
        scale = variable_scale(prior_n, prior_n.ndim)
        prior_n = prior_n / scale
        reference_n = reference_n / scale
        observed_n = observed_n / scale
        anchor_n = anchor_n / scale

        prior_core, core_shape = _to_cores(prior_n, core_size)
        observed_core, _ = _to_cores(observed_n, core_size)
        anchor_core, _ = _to_cores(anchor_n, core_size)
        reference_core, _ = _to_cores(reference_n, core_size)
        if bool(getattr(solver_config, "residual_phenology_mode", False)):
            # The numerical solver operates on a bounded residual encoding. This
            # keeps its [0,1] assumptions while moving seasonal amplitude and peak
            # timing into the explicit PHENO reference rather than asking the
            # low-rank/L1 stages to reconstruct the absolute vegetation cycle.
            solver_observed_core = (
                0.5 + 0.5 * (observed_core - reference_core)
            ).clamp(0.0, 1.0)
            solver_prior_core = (
                0.5 + 0.5 * (prior_core - reference_core)
            ).clamp(0.0, 1.0)
            solver_anchor_core = (
                0.5 + 0.5 * (anchor_core - reference_core)
            ).clamp(0.0, 1.0)
        else:
            solver_observed_core = observed_core
            solver_prior_core = prior_core
            solver_anchor_core = anchor_core

        if not training_solver and not torch.is_grad_enabled():
            # Encoding/decoding and final QA acceptance stay in this pipeline.
            refinement = refine_encoded_core(
                observed=solver_observed_core,
                observation_weight=weight_core,
                prior=solver_prior_core,
                prior_strength=repeat(prior_strength),
                anchor_value=solver_anchor_core,
                anchor_mask=anchor_mask_core,
                difficulty=repeat(solver_demand),
                l1_lambda=repeat(l1_lambda),
                st_budget=st_budget,
                config=solver_config,
                execution_fastpath=production_only,
                validate=False,
            )
            st_core = refinement.st_values
            filtered_core = refinement.values
            st_diag = refinement.st_diagnostics
            l1_diag = refinement.l1_diagnostics
            st_diag["routed"] = selected
            st_diag["needs_solver"] = needs_solver
            if adaptive_budget is not None:
                st_diag["adaptive_budget"] = adaptive_budget
            if llm_budget is not None:
                st_diag["llm_budget"] = llm_budget
            l1_diag["routed"] = selected
            l1_diag["needs_solver"] = needs_solver
        else:
            st_context = torch.no_grad() if training_solver else nullcontext()
            with st_context:
                st_core, st_diag = adaptive_sttensor(
                    solver_observed_core,
                    weight_core,
                    solver_prior_core,
                    repeat(prior_strength),
                    solver_anchor_core,
                    anchor_mask_core,
                    repeat(solver_demand),
                    solver_config,
                    training=training_solver,
                    iteration_budget=st_budget,
                    execution_fastpath=production_only,
                )
            st_diag["routed"] = selected
            st_diag["needs_solver"] = needs_solver
            if adaptive_budget is not None:
                st_diag["adaptive_budget"] = adaptive_budget
            if llm_budget is not None:
                st_diag["llm_budget"] = llm_budget

            n, pixels, _t, _y = st_core.shape
            sequence = st_core.permute(0, 1, 3, 2).reshape(n, pixels, _y * _t)
            prior_sequence = solver_prior_core.permute(0, 1, 3, 2).reshape_as(sequence)
            curvature_protection = _phenology_curvature_protection(prior_sequence)
            observed_sequence = solver_observed_core.permute(0, 1, 3, 2).reshape_as(sequence)
            observation_weight_sequence = weight_core.permute(0, 1, 3, 2).reshape_as(sequence)
            anchor_sequence = solver_anchor_core.permute(0, 1, 3, 2).reshape_as(sequence)
            anchor_mask_sequence = anchor_mask_core.permute(0, 1, 3, 2).reshape_as(sequence)
            positive = st_budget > 0
            l1_budget = torch.ceil(
                st_budget.to(prior.dtype)
                / max(float(solver_config.max_iterations), 1.0)
                * float(solver_config.l1_max_iterations)
            ).to(torch.long)
            l1_budget = torch.where(
                positive,
                l1_budget.clamp(min=int(solver_config.l1_min_iterations)),
                torch.zeros_like(l1_budget),
            )
            l1_context = torch.no_grad() if training_solver else nullcontext()
            with l1_context:
                filtered, l1_diag = adaptive_l1_trend_filter(
                    sequence,
                    observed_sequence,
                    observation_weight_sequence,
                    anchor_sequence,
                    anchor_mask_sequence,
                    repeat(l1_lambda),
                    repeat(solver_demand),
                    solver_config,
                    training=training_solver,
                    iteration_budget=l1_budget,
                    curvature_protection=curvature_protection,
                    execution_fastpath=production_only,
                )
            l1_diag["routed"] = selected
            l1_diag["needs_solver"] = needs_solver
            filtered_core = filtered.reshape(n, pixels, _y, _t).permute(0, 1, 3, 2)
        final_n = _from_cores(filtered_core, core_shape, core_size)
        if bool(getattr(solver_config, "residual_phenology_mode", False)):
            final_n = reference_n + 2.0 * (final_n - 0.5)
        final = (final_n * variable_scale(final_n, final_n.ndim)).permute(0, 1, 5, 4, 2, 3)
        final = torch.stack([final[:, 0].clamp(0.0, 7.0), final[:, 1].clamp(0.0, 1.0)], dim=1)
        if not production_only:
            st_n = _from_cores(st_core, core_shape, core_size)
            if bool(getattr(solver_config, "residual_phenology_mode", False)):
                st_n = reference_n + 2.0 * (st_n - 0.5)
            st_full = (st_n * variable_scale(st_n, st_n.ndim)).permute(0, 1, 5, 4, 2, 3)
            st_full = torch.stack(
                [st_full[:, 0].clamp(0.0, 7.0), st_full[:, 1].clamp(0.0, 1.0)], dim=1
            )
            center_st_prediction = gather_year(st_full.permute(0, 2, 1, 3, 4, 5))
        center_prediction = gather_year(final.permute(0, 2, 1, 3, 4, 5))
    st_diag["routed"], st_diag["needs_solver"] = selected, needs_solver
    l1_diag["routed"], l1_diag["needs_solver"] = selected, needs_solver
    center_prior = gather_year(prior)
    if not production_only:
        center_solver_reference = gather_year(phenology_reference)
    center_base_prior = gather_year(base_prior)
    if not production_only:
        center_memory = gather_year(memory)
        center_memory_support = gather_year(memory_support)
        center_memory_dispersion = gather_year(memory_dispersion)
        center_memory_gate = gather_year(memory_gate)
        center_fused = gather_year(flat_out["fused"].reshape(b, years, 2, t, h, w))
    center_amplitude_prior = gather_year(
        flat_out["amplitude_prior"].reshape(b, years, 2, t, h, w)
    )
    center_amplitude_gain = gather_year(
        flat_out["amplitude_gain"].reshape(b, years, 2, t, h, w)
    )
    # Candidate reliability must be calibrated against the exact endpoints
    # that enter the final prior.  The inherited decoder adds a residual after
    # expert fusion, so raw Local/PHENO values are not valid counterfactual
    # endpoints whenever that residual is nonzero.
    center_residual = center_base_prior - center_amplitude_prior

    def prior_endpoint(candidate: torch.Tensor) -> torch.Tensor:
        seasonal = candidate.mean(dim=2, keepdim=True)
        amplitude = seasonal + center_amplitude_gain * (candidate - seasonal)
        endpoint = amplitude + center_residual
        return torch.stack(
            [endpoint[:, 0].clamp(0.0, 7.0), endpoint[:, 1].clamp(0.0, 1.0)],
            dim=1,
        )
    center_observed = gather_year(observed)
    center_observation_weight = gather_year(obs_weight_window)
    center_anchor = gather_year(anchor)
    center_anchor_mask = gather_year(anchor_mask_window)
    center_data_consistent_prior = apply_observation_fidelity(
        center_prior,
        center_observed,
        center_observation_weight,
        center_anchor,
        center_anchor_mask,
        solver_config.data_fidelity_gain,
    )
    center_log_variance = gather_year(flat_out["log_variance"].reshape(b, years, 2, t, h, w))
    expert_center = gather_year(
        flat_experts.reshape(b, years, 2, len(EXPERT_NAMES), t, h, w)
    )
    expert_support_center = gather_year(
        flat_expert_support.reshape(b, years, 2, len(EXPERT_NAMES), t, h, w)
    )
    gate_center = gather_year(
        flat_out["gate_weights"].reshape(b, years, 2, len(EXPERT_NAMES), t, h, w)
    )
    local_gate_center = gather_year(
        flat_out["local_gate_weights"].reshape(b, years, 2, 3, t, h, w)
    )
    local_fused_center = gather_year(
        flat_out["local_fused"].reshape(b, years, 2, t, h, w)
    )
    local_support_center = gather_year(
        flat_out["local_support"].reshape(b, years, 2, t, h, w)
    )
    phenology_mean_center = gather_year(
        flat_out["phenology_mean"].reshape(b, years, 2, t, h, w)
    )
    phenology_peak_center = gather_year(
        flat_out["phenology_peak"].reshape(b, years, 2, t, h, w)
    )
    local_candidate_endpoint = prior_endpoint(local_fused_center)
    if not production_only:
        phenology_mean_candidate_endpoint = prior_endpoint(phenology_mean_center)
        phenology_peak_candidate_endpoint = prior_endpoint(phenology_peak_center)
    phenology_candidate_endpoint = prior_endpoint(expert_center[:, :, 3])
    phenology_selector_center = gather_year(
        flat_out["phenology_selector"].reshape(b, years, 2, t, h, w)
    )
    phenology_group_gate_center = gather_year(
        flat_out["phenology_group_gate"].reshape(b, years, 2, t, h, w)
    )
    if not production_only:
        parent_phenology_group_gate_center = gather_year(
            flat_out["parent_phenology_group_gate"].reshape(b, years, 2, t, h, w)
        )
    candidate_error_center = gather_year(
        flat_out["candidate_error"].reshape(b, years, 2, 2, t, h, w)
    )
    if not production_only:
        candidate_error_gate_center = gather_year(
            flat_out["candidate_error_gate"].reshape(b, years, 2, t, h, w)
        )
        candidate_risk_center = gather_year(
            flat_out["candidate_risk"].reshape(b, years, 2, 3, t, h, w)
        )
        candidate_index_center = gather_year(
            flat_out["candidate_index"].reshape(b, years, 2, t, h, w)
        )
    pre_solver_acceptance_center = gather_year(
        flat_out["solver_acceptance"].reshape(b, years, 2, t, h, w)
    ).clamp(0.0, 1.0)
    phenology_expert_dispersion = gather_year(
        prepared_frontend["phenology_expert_dispersion"]
    )
    if not production_only:
        phenology_fit_amplitude = gather_year(
            prepared_frontend["phenology_fit_amplitude"]
        )
        phenology_fit_phase_shift = gather_year(
            prepared_frontend["phenology_fit_phase_shift"]
        )
    solver_phenology_support = gather_year(
        prepared_frontend["solver_phenology_support"]
    )
    solver_phenology_dispersion = gather_year(
        prepared_frontend["solver_phenology_dispersion"]
    )
    # Compatibility aliases now expose the explicit fourth expert. They no
    # longer represent a post-fusion correction branch.
    center_memory = expert_center[:, :, 3]
    center_memory_support = expert_support_center[:, :, 3]
    center_memory_dispersion = phenology_expert_dispersion
    center_memory_gate = phenology_group_gate_center
    solver_structural_fill_mask = None
    if solver_structural_cells is not None:
        nr, nc = h // core_size, w // core_size
        selected_map = selected.reshape(b, 2, nr, nc)
        selected_map = selected_map.repeat_interleave(core_size, dim=2).repeat_interleave(
            core_size, dim=3
        )
        solver_structural_fill_mask = (
            solver_structural_cells & selected_map[:, :, None, :, :]
        )
        center_prediction = torch.where(
            solver_structural_fill_mask, center_prediction, center_prior
        )
        if not production_only:
            center_st_prediction = torch.where(
                solver_structural_fill_mask, center_st_prediction, center_prior
            )
    center_prediction = apply_observation_fidelity(
        center_prediction,
        center_observed,
        center_observation_weight,
        center_anchor,
        center_anchor_mask,
        solver_config.data_fidelity_gain,
    )
    center_solver_candidate = center_prediction
    nr, nc = h // core_size, w // core_size
    solver_budget_fraction = (
        st_budget.reshape(b, 2, nr, nc).to(center_prediction.dtype)
        / max(float(solver_config.max_iterations), 1.0)
    )
    solver_budget_fraction = solver_budget_fraction.repeat_interleave(
        core_size, dim=2
    ).repeat_interleave(core_size, dim=3)
    solver_budget_fraction = solver_budget_fraction[:, :, None].expand(
        b, 2, t, h, w
    )
    solver_acceptance_center = model.refine_solver_acceptance(
        center_data_consistent_prior,
        center_solver_candidate,
        phenology_peak_center,
        expert_center[:, :, 3],
        local_fused_center,
        solver_phenology_support,
        solver_phenology_dispersion,
        phenology_selector_center,
        phenology_group_gate_center,
        pre_solver_acceptance_center,
        solver_budget_fraction,
    )
    solver_acceptance_center, solver_peak_guard = apply_candidate_relative_peak_guard(
        solver_acceptance_center,
        phenology_group_gate_center,
        phenology_candidate_endpoint,
        local_candidate_endpoint,
        center_solver_candidate,
        center_data_consistent_prior,
        solver_budget_fraction,
        bool(getattr(model.config, "solver_peak_safe_routing", False)),
        float(getattr(model.config, "solver_peak_safe_margin_lai", 0.0)),
    )
    center_prediction = center_data_consistent_prior + solver_acceptance_center * (
        center_solver_candidate - center_data_consistent_prior
    )
    center_prediction = torch.stack(
        [center_prediction[:, 0].clamp(0.0, 7.0), center_prediction[:, 1].clamp(0.0, 1.0)],
        dim=1,
    )
    if production_only:
        return {
            "prediction": center_prediction,
            "data_consistent_prior": center_data_consistent_prior,
            "solver_candidate": center_solver_candidate,
            "solver_acceptance": solver_acceptance_center,
            "solver_peak_guard": solver_peak_guard,
            "experts": expert_center,
            "expert_support": expert_support_center,
            "local_gate_weights": local_gate_center,
            "local_fused": local_fused_center,
            "phenology_group_gate": phenology_group_gate_center,
            "phenology_memory": center_memory,
            "candidate_error": candidate_error_center,
            "log_variance": center_log_variance,
            "difficulty": difficulty,
            "solver_structural_fill_mask": solver_structural_fill_mask,
            "solver_route_fraction": selected.to(prior.dtype).reshape(b, 2, ncore).mean(dim=-1),
            "solver_route_score": route_scores.reshape(b, 2, ncore),
            "solver_active_fraction": needs_solver.to(prior.dtype).reshape(b, 2, ncore).mean(dim=-1),
            "solver_budget_fraction": st_budget.to(prior.dtype).reshape(b, 2, ncore).mean(dim=-1)
            / max(float(solver_config.max_iterations), 1.0),
            "st_diagnostics": st_diag,
            "l1_diagnostics": l1_diag,
            "production_zero_budget": zero_budget_fastpath,
        }

    center_st_prediction = apply_observation_fidelity(
        center_st_prediction,
        center_observed,
        center_observation_weight,
        center_anchor,
        center_anchor_mask,
        solver_config.data_fidelity_gain,
    )
    if return_budget_features:
        if "budget_features" not in flat_out:
            raise RuntimeError("Budget features requested from a model without a budget head.")
        budget_feature_core = _aggregate_budget_features(
            flat_out["budget_features"], b, years, core_size
        )
    result = {
        "prediction": center_prediction,
        "solver_candidate": center_solver_candidate,
        "solver_acceptance": solver_acceptance_center,
        "solver_peak_guard": solver_peak_guard,
        "pre_solver_acceptance": pre_solver_acceptance_center,
        "solver_budget_fraction_map": solver_budget_fraction,
        "st_prediction": center_st_prediction,
        "prior": center_prior,
        "data_consistent_prior": center_data_consistent_prior,
        "solver_reference": center_solver_reference,
        "base_prior": center_base_prior,
        "phenology_memory": center_memory,
        "phenology_memory_support": center_memory_support,
        "phenology_memory_dispersion": center_memory_dispersion,
        "phenology_memory_gate": center_memory_gate,
        "fused": center_fused,
        "amplitude_prior": center_amplitude_prior,
        "amplitude_gain": center_amplitude_gain,
        "observed": center_observed,
        "observation_weight": center_observation_weight,
        "anchor_value": center_anchor,
        "anchor_mask": center_anchor_mask,
        "full_window": final,
        "experts": expert_center,
        "expert_support": expert_support_center,
        "gate_weights": gate_center,
        "local_gate_weights": local_gate_center,
        "local_fused": local_fused_center,
        "local_candidate_endpoint": local_candidate_endpoint,
        "local_support": local_support_center,
        "phenology_mean": phenology_mean_center,
        "phenology_peak": phenology_peak_center,
        "phenology_mean_candidate_endpoint": phenology_mean_candidate_endpoint,
        "phenology_peak_candidate_endpoint": phenology_peak_candidate_endpoint,
        "phenology_candidate_endpoint": phenology_candidate_endpoint,
        "phenology_selector": phenology_selector_center,
        "phenology_group_gate": phenology_group_gate_center,
        "parent_phenology_group_gate": parent_phenology_group_gate_center,
        "candidate_error": candidate_error_center,
        "candidate_error_gate": candidate_error_gate_center,
        "candidate_risk": candidate_risk_center,
        "candidate_index": candidate_index_center,
        "phenology_expert_dispersion": phenology_expert_dispersion,
        "phenology_fit_amplitude": phenology_fit_amplitude,
        "phenology_fit_phase_shift": phenology_fit_phase_shift,
        "solver_phenology_support": solver_phenology_support,
        "solver_phenology_dispersion": solver_phenology_dispersion,
        "log_variance": center_log_variance,
        "difficulty": difficulty,
        "continuation": continuation,
        "solver_demand": solver_demand,
        "l1_lambda": l1_lambda,
        "prior_strength": prior_strength,
        "solver_route_fraction": selected.to(prior.dtype).reshape(b, 2, ncore).mean(dim=-1),
        "solver_route_score": route_scores.reshape(b, 2, ncore),
        "solver_active_fraction": needs_solver.to(prior.dtype).reshape(b, 2, ncore).mean(dim=-1),
        "solver_budget_fraction": st_budget.to(prior.dtype).reshape(b, 2, ncore).mean(dim=-1)
        / max(float(solver_config.max_iterations), 1.0),
        "st_diagnostics": st_diag,
        "l1_diagnostics": l1_diag,
    }
    if budget_logits_core is not None:
        result["budget_logits_core"] = budget_logits_core
        result["budget_class"] = budget_class
    if solver_benefit_probability is not None:
        result["solver_benefit_probability"] = solver_benefit_probability
    if solver_structural_eligible is not None:
        result["solver_structural_eligible"] = solver_structural_eligible
    if solver_structural_fill_mask is not None:
        result["solver_structural_fill_mask"] = solver_structural_fill_mask
    if budget_feature_core is not None:
        result["budget_features_core"] = budget_feature_core
        result["budget_context_core"] = _budget_gap_context(
            usable_mask_window, anchor_mask_window, domain, center_idx, core_size
        )
    if return_prepared_frontend:
        result["prepared_frontend"] = {
            "flat_experts": flat_experts,
            "flat_expert_support": flat_expert_support,
            "phenology_expert_dispersion": prepared_frontend[
                "phenology_expert_dispersion"
            ],
            "phenology_fit_amplitude": prepared_frontend[
                "phenology_fit_amplitude"
            ],
            "phenology_fit_phase_shift": prepared_frontend[
                "phenology_fit_phase_shift"
            ],
            "solver_phenology_reference": prepared_frontend[
                "solver_phenology_reference"
            ],
            "solver_phenology_support": prepared_frontend[
                "solver_phenology_support"
            ],
            "solver_phenology_dispersion": prepared_frontend[
                "solver_phenology_dispersion"
            ],
            "flat_out": base_flat_out,
        }
    return result
