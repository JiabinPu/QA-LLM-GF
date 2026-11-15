"""Fast sensor-independent local and interannual phenology experts."""

from __future__ import annotations

from functools import lru_cache

import torch


MIN_WEIGHT = 0.02
EXPERT_NAMES = ("LIN", "QA_IDW", "CST", "PHENO")


def _shift_time_no_wrap(value: torch.Tensor, offset: int) -> torch.Tensor:
    """Shift a [B,V,T,H,W] series without wrapping year-end values."""
    shifted = torch.zeros_like(value)
    time = value.shape[2]
    if abs(int(offset)) >= time:
        return shifted
    if offset >= 0:
        shifted[:, :, : time - offset] = value[:, :, offset:]
    else:
        shifted[:, :, -offset:] = value[:, :, : time + offset]
    return shifted


def _fit_visible_phenology_template(
    mean_value: torch.Tensor,
    peak_value: torch.Tensor,
    support: torch.Tensor,
    dispersion: torch.Tensor,
    visible_value: torch.Tensor,
    visible_confidence: torch.Tensor,
    phase_radius: int,
    amplitude_min: float,
    amplitude_max: float,
    minimum_fit_steps: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Align and scale a cross-year template using visible target-year shoulders.

    The withheld target cells have already been removed from visible_confidence.
    Consequently, this fit uses operationally available evidence only and cannot
    leak a validation target into its own reconstruction.
    """
    scale = torch.tensor(
        (7.0, 1.0), dtype=mean_value.dtype, device=mean_value.device
    ).view(1, 2, 1, 1, 1)
    offsets = tuple(range(-int(phase_radius), int(phase_radius) + 1))
    shifted_mean = torch.stack(
        [_shift_time_no_wrap(mean_value, offset) for offset in offsets], dim=2
    )
    shifted_peak = torch.stack(
        [_shift_time_no_wrap(peak_value, offset) for offset in offsets], dim=2
    )
    shifted_support = torch.stack(
        [_shift_time_no_wrap(support, offset) for offset in offsets], dim=2
    )
    shifted_dispersion = torch.stack(
        [_shift_time_no_wrap(dispersion, offset) for offset in offsets], dim=2
    )

    fit_weight = (visible_confidence.unsqueeze(2) * shifted_support).clamp(0.0, 1.0)
    template_n = shifted_mean / scale.unsqueeze(2)
    visible_n = visible_value.unsqueeze(2) / scale.unsqueeze(2)
    fit_mass = fit_weight.sum(dim=3, keepdim=True)
    fit_steps = (fit_weight > 0.0).sum(dim=3)
    template_mean = (fit_weight * template_n).sum(dim=3, keepdim=True) / fit_mass.clamp_min(1e-8)
    visible_mean = (fit_weight * visible_n).sum(dim=3, keepdim=True) / fit_mass.clamp_min(1e-8)
    centered_template = template_n - template_mean
    centered_visible = visible_n - visible_mean
    variance = (fit_weight * centered_template.square()).sum(dim=3, keepdim=True)
    covariance = (fit_weight * centered_template * centered_visible).sum(dim=3, keepdim=True)
    # Ridge shrinkage toward unit amplitude prevents sparse shoulders from
    # turning a poorly conditioned regression into an extreme annual scaling.
    ridge = 0.10 * variance + 1e-4 * fit_mass
    candidate_amplitude = ((covariance + ridge) / (variance + ridge).clamp_min(1e-8)).clamp(
        float(amplitude_min), float(amplitude_max)
    )
    can_scale = (fit_steps.unsqueeze(3) >= int(minimum_fit_steps)) & (variance > 1e-6)
    candidate_amplitude = torch.where(
        can_scale, candidate_amplitude, torch.ones_like(candidate_amplitude)
    )
    candidate_intercept_n = torch.where(
        can_scale,
        visible_mean - candidate_amplitude * template_mean,
        torch.zeros_like(visible_mean),
    )
    fitted_n = candidate_amplitude * template_n + candidate_intercept_n
    fit_error = (fit_weight * (visible_n - fitted_n).square()).sum(dim=3) / fit_mass.squeeze(3).clamp_min(1e-8)
    eligible = fit_steps >= int(minimum_fit_steps)
    fit_error = fit_error.masked_fill(~eligible, float("inf"))
    best_shift = fit_error.argmin(dim=2)
    any_eligible = eligible.any(dim=2)
    zero_shift_index = offsets.index(0)
    zero_error = fit_error[:, :, zero_shift_index]
    best_error = fit_error.gather(2, best_shift.unsqueeze(2)).squeeze(2)
    relative_gain = (zero_error - best_error) / zero_error.clamp_min(1e-8)
    # A one-composite shift is accepted only when visible target-year evidence
    # gives a material improvement. This avoids phase jitter from sparse QA.
    phase_supported = relative_gain >= 0.05
    best_shift = torch.where(
        any_eligible & phase_supported,
        best_shift,
        torch.full_like(best_shift, zero_shift_index),
    )

    def choose(stack: torch.Tensor) -> torch.Tensor:
        index = best_shift[:, :, None, None].expand(
            -1, -1, 1, stack.shape[3], -1, -1
        )
        return stack.gather(2, index).squeeze(2)

    aligned_mean = choose(shifted_mean)
    aligned_peak = choose(shifted_peak)
    aligned_support = choose(shifted_support)
    aligned_dispersion = choose(shifted_dispersion)

    parameter_index = best_shift.unsqueeze(2).unsqueeze(3).expand(
        -1, -1, 1, 1, aligned_mean.shape[-2], aligned_mean.shape[-1]
    )
    amplitude = candidate_amplitude.gather(2, parameter_index).squeeze(2)
    intercept_n = candidate_intercept_n.gather(2, parameter_index).squeeze(2)
    intercept = intercept_n * scale
    fitted_mean = amplitude * aligned_mean + intercept
    fitted_peak = amplitude * aligned_peak + intercept
    fitted_peak = torch.maximum(fitted_mean, fitted_peak)
    fitted_mean[:, 0].clamp_(0.0, 7.0)
    fitted_mean[:, 1].clamp_(0.0, 1.0)
    fitted_peak[:, 0].clamp_(0.0, 7.0)
    fitted_peak[:, 1].clamp_(0.0, 1.0)
    phase_values = torch.as_tensor(
        offsets, dtype=mean_value.dtype, device=mean_value.device
    )[best_shift]
    phase_values = phase_values[:, :, None].expand_as(fitted_mean)
    amplitude_values = amplitude.expand_as(fitted_mean)
    return (
        fitted_mean,
        fitted_peak,
        aligned_support,
        aligned_dispersion,
        amplitude_values,
        phase_values,
    )


def _channels(x: torch.Tensor, variable: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    value_ch, valid_ch, qa_ch = ((0, 2, 4) if variable == 0 else (1, 3, 5))
    return x[:, value_ch], x[:, valid_ch] > 0.5, x[:, qa_ch].clamp(0.0, 1.0)


def _weighted_sensor_composite(
    values: torch.Tensor,
    valid: torch.Tensor,
    qa: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    weights = valid.to(values.dtype) * qa.clamp_min(MIN_WEIGHT)
    denom = weights.sum(dim=-1)
    fused = (values * weights).sum(dim=-1) / denom.clamp_min(torch.finfo(values.dtype).tiny)
    return fused, denom > 0, denom


def _profile_mean(values: torch.Tensor, valid: torch.Tensor, dim: int) -> torch.Tensor:
    weights = valid.to(values.dtype)
    return (values * weights).sum(dim=dim, keepdim=True) / weights.sum(dim=dim, keepdim=True).clamp_min(1.0)


def _linear(series: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    b, t, h, w = series.shape
    profiles = h * w
    values = series.permute(0, 2, 3, 1).reshape(b, profiles, t)
    observed = valid.permute(0, 2, 3, 1).reshape(b, profiles, t)
    idx = torch.arange(t, device=series.device, dtype=torch.long).view(1, 1, t)
    prev_idx = torch.cummax(torch.where(observed, idx, torch.full_like(idx, -1)), dim=2).values
    next_idx = torch.flip(
        torch.cummin(torch.flip(torch.where(observed, idx, torch.full_like(idx, t)), dims=[2]), dim=2).values,
        dims=[2],
    )
    prev_val = torch.gather(values, 2, prev_idx.clamp(0, t - 1))
    next_val = torch.gather(values, 2, next_idx.clamp(0, t - 1))
    both = (prev_idx >= 0) & (next_idx < t)
    alpha = (idx.to(values.dtype) - prev_idx.to(values.dtype)) / (
        next_idx.to(values.dtype) - prev_idx.to(values.dtype)
    ).clamp_min(1.0)
    interp = prev_val + alpha * (next_val - prev_val)
    fallback = _profile_mean(values, observed, dim=2)
    out = torch.where(
        both,
        interp,
        torch.where(prev_idx >= 0, prev_val, torch.where(next_idx < t, next_val, fallback.expand_as(values))),
    )
    out = torch.where(observed, values, out)
    distance_scale = max(float(t) / 6.0, 1.0)
    span = (next_idx - prev_idx).clamp_min(1).to(values.dtype)
    prev_distance = (idx - prev_idx).clamp_min(0).to(values.dtype)
    next_distance = (next_idx - idx).clamp_min(0).to(values.dtype)
    two_sided = 1.0 / (1.0 + span / distance_scale)
    one_sided = torch.where(
        prev_idx >= 0,
        0.5 / (1.0 + prev_distance / distance_scale),
        torch.where(next_idx < t, 0.5 / (1.0 + next_distance / distance_scale), torch.zeros_like(values)),
    )
    support = torch.where(observed, torch.ones_like(values), torch.where(both, two_sided, one_sided))
    shape = (b, h, w, t)
    return (
        out.reshape(shape).permute(0, 3, 1, 2).contiguous(),
        support.reshape(shape).permute(0, 3, 1, 2).contiguous().clamp(0.0, 1.0),
    )


@lru_cache(maxsize=16)
def _idw_kernel_cpu(t: int, power: float) -> torch.Tensor:
    idx = torch.arange(t, dtype=torch.float32)
    return 1.0 / (torch.abs(idx[:, None] - idx[None, :]) + 1.0).pow(power)


def _idw_all_sensors(
    values: torch.Tensor,
    valid: torch.Tensor,
    qa: torch.Tensor,
    power: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    b, t, h, w, s = values.shape
    weights = valid.to(values.dtype) * qa.clamp_min(MIN_WEIGHT)
    kernel = _idw_kernel_cpu(t, power).to(device=values.device, dtype=values.dtype)
    weighted = (values * weights).permute(1, 0, 2, 3, 4).reshape(t, -1)
    weight_2d = weights.permute(1, 0, 2, 3, 4).reshape(t, -1)
    numerator = (kernel @ weighted).reshape(t, b, h, w, s).sum(dim=-1).permute(1, 0, 2, 3)
    denominator = (kernel @ weight_2d).reshape(t, b, h, w, s).sum(dim=-1).permute(1, 0, 2, 3)
    composite, composite_valid, _ = _weighted_sensor_composite(values, valid, qa)
    fallback = _profile_mean(composite, composite_valid, dim=1)
    estimate = torch.where(
        denominator > 0,
        numerator / denominator.clamp_min(1e-8),
        fallback.expand_as(composite),
    )
    support = (1.0 - torch.exp(-0.5 * denominator)).clamp(0.0, 1.0)
    return estimate, support


def build_si_experts(
    x: torch.Tensor,
    return_support: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Return [B, variable, expert, time, row, col] in physical units."""
    if x.ndim != 6:
        raise ValueError(f"Expected [B,C,T,H,W,S], got {tuple(x.shape)}")
    b = x.shape[0]
    values = torch.stack([x[:, 0], x[:, 1]], dim=1).flatten(0, 1)
    valid = torch.stack([x[:, 2] > 0.5, x[:, 3] > 0.5], dim=1).flatten(0, 1)
    qa = torch.stack([x[:, 4], x[:, 5]], dim=1).flatten(0, 1).clamp(0.0, 1.0)
    same_day, same_day_valid, same_day_weight = _weighted_sensor_composite(values, valid, qa)
    linear, linear_support = _linear(same_day, same_day_valid)
    idw, idw_support = _idw_all_sensors(values, valid, qa)
    cst = torch.where(same_day_valid, same_day, idw)
    same_day_support = (1.0 - torch.exp(-same_day_weight)).clamp(0.0, 1.0)
    cst_support = torch.where(same_day_valid, same_day_support, 0.75 * idw_support)
    out = torch.stack([linear, idw, cst], dim=1).reshape(b, 2, 3, *linear.shape[1:])
    support = torch.stack([linear_support, idw_support, cst_support], dim=1).reshape(
        b, 2, 3, *linear_support.shape[1:]
    )
    out[:, 0].clamp_(0.0, 7.0)
    out[:, 1].clamp_(0.0, 1.0)
    return (out, support) if return_support else out


def build_phenology_expert(
    x_window: torch.Tensor,
    fallback: torch.Tensor,
    phase_radius: int = 1,
    fit_visible_template: bool = True,
    amplitude_min: float = 0.50,
    amplitude_max: float = 2.00,
    minimum_fit_steps: int = 3,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Build QA/PFT-consistent phenology candidates with bounded phase tolerance.

    Parameters
    ----------
    x_window
        Input tensor shaped [B,Y,C,T,H,W,S]. The center-year withheld values
        have already been removed by the dataset, so the expert cannot see the
        validation target.
    fallback
        A local estimate shaped [B,Y,2,T,H,W]. It is returned wherever no
        cross-year evidence exists, while zero support prevents PHENO from
        receiving a meaningful gate advantage there.

    Returns
    -------
    mean_value, peak_value, support, dispersion
        Parent-compatible fusion expert fields.
    fit_amplitude, fit_phase_shift
        Visible-evidence affine-fit diagnostics.
    fitted_mean, fitted_peak, fitted_support, fitted_dispersion
        Solver-reference fields. Fitting never replaces the validated fusion
        expert.
    """
    if x_window.ndim != 7:
        raise ValueError(f"Expected [B,Y,C,T,H,W,S], got {tuple(x_window.shape)}")
    phase_radius = int(phase_radius)
    if phase_radius < 0 or phase_radius > 3:
        raise ValueError("phase_radius must be between 0 and 3 composites.")
    b, years, channels, time, height, width, sensors = x_window.shape
    expected = (b, years, 2, time, height, width)
    if tuple(fallback.shape) != expected:
        raise ValueError(f"Expected fallback {expected}, got {tuple(fallback.shape)}")

    values = torch.stack((x_window[:, :, 0], x_window[:, :, 1]), dim=2)
    valid = torch.stack((x_window[:, :, 2] > 0.5, x_window[:, :, 3] > 0.5), dim=2)
    qa = torch.stack((x_window[:, :, 4], x_window[:, :, 5]), dim=2).clamp(0.0, 1.0)
    candidate = torch.stack(
        (x_window[:, :, 6] > 0.5, x_window[:, :, 7] > 0.5), dim=2
    ) & valid
    saturated = (x_window[:, :, 18] > 0.5).unsqueeze(2).expand_as(candidate)
    strict = candidate & ~saturated
    sensor_weight = candidate.to(values.dtype) * qa.clamp_min(MIN_WEIGHT)

    # Saturated retrievals remain usable. If unsaturated and saturated
    # retrievals coexist, every unsaturated candidate receives higher weight.
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
    year_confidence = 1.0 - torch.prod(
        1.0 - sensor_weight.clamp(0.0, 1.0), dim=-1
    )

    if channels >= 27:
        one_hot = x_window[:, :, 19:27, 0, :, :, 0]
        pft_support, pft_index = one_hot.max(dim=2)
        pft_known = pft_support > 0.5
    else:
        pft_index = torch.zeros(
            (b, years, height, width), dtype=torch.long, device=x_window.device
        )
        pft_known = torch.ones_like(pft_index, dtype=torch.bool)

    mean_estimate = fallback.clone()
    peak_estimate = fallback.clone()
    support = torch.zeros_like(fallback)
    dispersion = torch.ones_like(fallback)
    legacy_mean = fallback.clone()
    legacy_peak = fallback.clone()
    legacy_support = torch.zeros_like(fallback)
    legacy_dispersion = torch.ones_like(fallback)
    fit_amplitude = torch.ones_like(fallback)
    fit_phase_shift = torch.zeros_like(fallback)
    year_index = torch.arange(years, dtype=values.dtype, device=values.device)
    physical_scale = torch.tensor(
        (7.0, 1.0), dtype=values.dtype, device=values.device
    ).view(1, 2, 1, 1, 1)

    for target_year in range(years):
        distance = 1.0 / (1.0 + torch.abs(year_index - float(target_year)))
        # The cross-year template must be independent of the target year. Its
        # still-visible observations are used once, downstream, to fit phase
        # and amplitude; local experts and QA projection retain the direct
        # current-year cross-sensor evidence.
        distance[target_year] = 0.0
        distance = distance.view(1, years, 1, 1, 1, 1)
        pft_match = (
            (pft_index == pft_index[:, target_year : target_year + 1])
            & pft_known
            & pft_known[:, target_year : target_year + 1]
        ).unsqueeze(2).unsqueeze(3)
        same_doy_denominator = torch.zeros_like(year_denom[:, target_year])
        same_doy_numerator = torch.zeros_like(same_doy_denominator)
        phase_denominator = torch.zeros_like(same_doy_denominator)
        phase_numerator = torch.zeros_like(same_doy_denominator)
        phase_second_moment = torch.zeros_like(same_doy_denominator)
        peak_value = torch.full_like(same_doy_denominator, float("-inf"))
        evidence_complement = torch.ones_like(same_doy_denominator)
        for offset in range(-phase_radius, phase_radius + 1):
            phase_weight = 1.0 / (1.0 + abs(offset))
            if offset >= 0:
                target_slice = slice(0, time - offset)
                source_slice = slice(offset, time)
            else:
                target_slice = slice(-offset, time)
                source_slice = slice(0, time + offset)
            shifted_value = year_value[:, :, :, source_slice]
            shifted_confidence = year_confidence[:, :, :, source_slice]
            source_weight = (
                shifted_confidence
                * distance
                * pft_match.to(values.dtype)
                * phase_weight
            )
            phase_denominator[:, :, target_slice] += source_weight.sum(dim=1)
            phase_numerator[:, :, target_slice] += (
                shifted_value * source_weight
            ).sum(dim=1)
            phase_second_moment[:, :, target_slice] += (
                shifted_value.square() * source_weight
            ).sum(dim=1)
            if offset == 0:
                same_doy_denominator = source_weight.sum(dim=1)
                same_doy_numerator = (shifted_value * source_weight).sum(dim=1)
            phase_peak = shifted_value.masked_fill(
                source_weight <= 0.0, float("-inf")
            ).amax(dim=1)
            peak_value[:, :, target_slice] = torch.maximum(
                peak_value[:, :, target_slice], phase_peak
            )
            evidence_complement[:, :, target_slice] *= torch.prod(
                1.0 - source_weight.clamp(0.0, 1.0), dim=1
            )
        has_evidence = phase_denominator > 0.0
        has_same_doy = same_doy_denominator > 0.0
        same_doy_value = same_doy_numerator / same_doy_denominator.clamp_min(1e-8)
        candidate_value = torch.where(
            has_same_doy, same_doy_value, fallback[:, target_year]
        )
        phase_center = phase_numerator / phase_denominator.clamp_min(1e-8)
        peak_value = torch.where(has_evidence, peak_value, fallback[:, target_year])
        variance = (
            phase_second_moment / phase_denominator.clamp_min(1e-8)
            - phase_center.square()
        ).clamp_min(0.0)
        normalized_dispersion = (
            torch.sqrt(variance.clamp_min(0.0)) / physical_scale
        ).clamp(0.0, 1.0)
        evidence = 1.0 - evidence_complement
        mean_estimate[:, target_year] = candidate_value
        peak_estimate[:, target_year] = peak_value
        support[:, target_year] = torch.where(
            has_evidence, evidence, torch.zeros_like(evidence)
        )
        dispersion[:, target_year] = torch.where(
            has_evidence, normalized_dispersion, torch.ones_like(normalized_dispersion)
        )
        legacy_mean[:, target_year] = mean_estimate[:, target_year]
        legacy_peak[:, target_year] = peak_estimate[:, target_year]
        legacy_support[:, target_year] = support[:, target_year]
        legacy_dispersion[:, target_year] = dispersion[:, target_year]

        if fit_visible_template:
            (
                mean_estimate[:, target_year],
                peak_estimate[:, target_year],
                support[:, target_year],
                dispersion[:, target_year],
                fit_amplitude[:, target_year],
                fit_phase_shift[:, target_year],
            ) = _fit_visible_phenology_template(
                mean_estimate[:, target_year],
                peak_estimate[:, target_year],
                support[:, target_year],
                dispersion[:, target_year],
                year_value[:, target_year],
                year_confidence[:, target_year],
                phase_radius,
                amplitude_min,
                amplitude_max,
                minimum_fit_steps,
            )

    fitted_mean = mean_estimate
    fitted_peak = peak_estimate
    fitted_support = support
    fitted_dispersion = dispersion
    legacy_mean[:, :, 0].clamp_(0.0, 7.0)
    legacy_mean[:, :, 1].clamp_(0.0, 1.0)
    legacy_peak[:, :, 0].clamp_(0.0, 7.0)
    legacy_peak[:, :, 1].clamp_(0.0, 1.0)
    return (
        legacy_mean,
        legacy_peak,
        legacy_support.clamp(0.0, 1.0),
        legacy_dispersion,
        fit_amplitude,
        fit_phase_shift,
        fitted_mean,
        fitted_peak,
        fitted_support.clamp(0.0, 1.0),
        fitted_dispersion,
    )


def build_window_experts(
    x_window: torch.Tensor,
    phenology_phase_radius: int = 1,
    fit_visible_phenology_template: bool = True,
    phenology_amplitude_min: float = 0.50,
    phenology_amplitude_max: float = 2.00,
    phenology_minimum_fit_steps: int = 3,
    prepared_local: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Return LIN, QA-IDW, CST, and PHENO for a multi-year window."""
    if x_window.ndim != 7:
        raise ValueError(f"Expected [B,Y,C,T,H,W,S], got {tuple(x_window.shape)}")
    b, years, channels, time, height, width, sensors = x_window.shape
    flat_x = x_window.reshape(b * years, channels, time, height, width, sensors)
    if prepared_local is None:
        local, local_support = build_si_experts(flat_x, return_support=True)
    else:
        if torch.is_grad_enabled():
            raise RuntimeError("Prepared local experts are inference-only.")
        local, local_support = prepared_local
        expected = (b * years, 2, 3, time, height, width)
        if tuple(local.shape) != expected or tuple(local_support.shape) != expected:
            raise ValueError("Prepared local experts have incompatible geometry.")
    local = local.reshape(b, years, 2, 3, time, height, width)
    local_support = local_support.reshape(b, years, 2, 3, time, height, width)
    (
        pheno_mean,
        pheno_peak,
        pheno_support,
        pheno_dispersion,
        pheno_fit_amplitude,
        pheno_fit_phase_shift,
        solver_pheno_mean,
        solver_pheno_peak,
        solver_pheno_support,
        solver_pheno_dispersion,
    ) = build_phenology_expert(
        x_window,
        local[:, :, :, 0],
        phase_radius=phenology_phase_radius,
        fit_visible_template=fit_visible_phenology_template,
        amplitude_min=phenology_amplitude_min,
        amplitude_max=phenology_amplitude_max,
        minimum_fit_steps=phenology_minimum_fit_steps,
    )
    experts = torch.cat((local, pheno_mean.unsqueeze(3)), dim=3)
    support = torch.cat((local_support, pheno_support.unsqueeze(3)), dim=3)
    return (
        experts.reshape(b * years, 2, len(EXPERT_NAMES), time, height, width),
        support.reshape(b * years, 2, len(EXPERT_NAMES), time, height, width),
        {
            "phenology_mean": pheno_mean.reshape(b * years, 2, time, height, width),
            "phenology_peak": pheno_peak.reshape(b * years, 2, time, height, width),
            "phenology_dispersion": pheno_dispersion,
            "phenology_fit_amplitude": pheno_fit_amplitude,
            "phenology_fit_phase_shift": pheno_fit_phase_shift,
            "solver_phenology_mean": solver_pheno_mean.reshape(
                b * years, 2, time, height, width
            ),
            "solver_phenology_peak": solver_pheno_peak.reshape(
                b * years, 2, time, height, width
            ),
            "solver_phenology_support": solver_pheno_support,
            "solver_phenology_dispersion": solver_pheno_dispersion,
        },
    )


def observed_si(
    x: torch.Tensor,
    reliability: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build soft observations and graded anchors for the latent SI tensor.

    SCF_QC=1 observations participate in the fused anchor value, but only an
    unsaturated candidate activates the exact solver constraint. When both
    states coexist, every unsaturated candidate has a strictly greater fusion
    weight than every saturated candidate.
    """
    values = torch.stack([x[:, 0], x[:, 1]], dim=1)
    valid = torch.stack([x[:, 2] > 0.5, x[:, 3] > 0.5], dim=1)
    qa = torch.stack([x[:, 4], x[:, 5]], dim=1).clamp(0.0, 1.0)
    candidates = torch.stack([x[:, 6] > 0.5, x[:, 7] > 0.5], dim=1) & valid
    saturated = (x[:, 18] > 0.5).unsqueeze(1).expand_as(candidates)
    strict = candidates & ~saturated
    # Learned reliability may modulate a QA-qualified observation but may not
    # silently remove it. This keeps the LLM in a QA-reasoning role while the
    # product remains data-consistent.
    rel = torch.ones_like(qa) if reliability is None else 0.5 + 0.5 * reliability.clamp(0.0, 1.0)
    weights = valid.to(values.dtype) * qa.clamp_min(MIN_WEIGHT) * rel
    denom = weights.sum(dim=-1)
    obs = (values * weights).sum(dim=-1) / denom.clamp_min(1e-8)
    anchor_weights = candidates.to(values.dtype) * weights
    # The ordering correction is only active when the two retrieval states
    # coexist. It does not introduce a fixed physical mixing coefficient.
    inf = torch.full_like(anchor_weights, float("inf"))
    weakest_strict = torch.where(strict, anchor_weights, inf).amin(
        dim=-1, keepdim=True
    )
    has_strict = strict.any(dim=-1, keepdim=True)
    saturation_cap = weakest_strict * (1.0 - 1e-4)
    anchor_weights = torch.where(
        saturated & candidates & has_strict,
        torch.minimum(anchor_weights, saturation_cap),
        anchor_weights,
    )
    anchor_denom = anchor_weights.sum(dim=-1)
    anchor_value = (values * anchor_weights).sum(dim=-1) / anchor_denom.clamp_min(1e-8)
    # Independent-evidence combination is invariant to the number of active
    # sensors and avoids reducing a valid single-sensor observation by 1/3.
    confidence = 1.0 - torch.prod(1.0 - weights.clamp(0.0, 1.0), dim=-1)
    exact_anchor = strict.any(dim=-1)
    return obs, confidence, anchor_value, exact_anchor
