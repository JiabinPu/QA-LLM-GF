"""Shared saturation-aware QA and StdDev calibration."""

from __future__ import annotations

from pathlib import Path

import numpy as np


QUALITY_SCORE_VERSION = "saturation_semantic_std_v2"
ANCHOR_THRESHOLD = 0.90
WARM_START_CHECKPOINT_VERSIONS = (
    "QA-LLM-GF-V36-saturation-aware-data-consistent",
)
TRAINING_CHECKPOINT_VERSION = "QA-LLM-GF-independent-full-controller"
BUDGET_CHECKPOINT_VERSION = "QA-LLM-GF-independent-full-budget"

_PARAMS = {
    "LAI": {"std_scale": 0.1, "value_floor": 0.2, "uncertainty_cap": 2.0},
    "FPAR": {"std_scale": 0.01, "value_floor": 0.05, "uncertainty_cap": 2.0},
}


def _version_text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    if isinstance(value, np.ndarray) and value.size == 1:
        return _version_text(value.reshape(-1)[0])
    return str(value or "")


def validate_checkpoint_protocol(checkpoint: dict, *, budget: bool) -> None:
    """Reject checkpoints produced by a different QA/fidelity protocol."""
    actual = _version_text(checkpoint.get("version", ""))
    accepted = (BUDGET_CHECKPOINT_VERSION,) if budget else (TRAINING_CHECKPOINT_VERSION,)
    if actual not in accepted:
        raise RuntimeError(
            f"Incompatible checkpoint protocol: expected one of {accepted!r}, got {actual!r}."
        )
    protocol = checkpoint.get("quality_protocol", {})
    if _version_text(protocol.get("quality_score_version", "")) != QUALITY_SCORE_VERSION:
        raise RuntimeError("Checkpoint does not contain the saturation-aware QA protocol.")
    if abs(float(protocol.get("anchor_threshold", -1.0)) - ANCHOR_THRESHOLD) > 1e-8:
        raise RuntimeError("Checkpoint anchor threshold is incompatible with the active protocol.")
    if _version_text(protocol.get("candidate_endpoint_contract", "")) != (
        "counterfactual_expert_after_inherited_amplitude_transform_and_decoder_"
        "residual_before_observation_projection"
    ):
        raise RuntimeError("Checkpoint does not contain the required candidate-endpoint contract.")
    if _version_text(protocol.get("withheld_target_confidence_role", "")) != (
        "independent_sensor_evidence_union_not_fixed_three_sensor_division"
    ):
        raise RuntimeError("Checkpoint uses the obsolete mission-era target weighting rule.")
    phase_radius = int(checkpoint.get("model_config", {}).get("phenology_phase_radius", -1))
    if phase_radius != 1:
        raise RuntimeError(
            "The active protocol requires phenology_phase_radius=1 for cross-tile comparability."
        )
    model_config = checkpoint.get("model_config", {})
    if model_config.get('active_variable') not in ('LAI', 'FPAR'):
        raise RuntimeError('Checkpoint must identify its independent target variable.')
    solver_config = checkpoint.get("solver_config", {})
    if bool(model_config.get("use_direct_three_candidate_routing", False)):
        raise RuntimeError("The active protocol prohibits direct three-candidate routing.")
    if not bool(model_config.get("use_candidate_error_routing", False)):
        raise RuntimeError("The active protocol requires Local-versus-PHENO candidate-risk routing.")
    if not bool(model_config.get("use_absolute_candidate_risk_routing", False)):
        raise RuntimeError("The active protocol requires absolute candidate-risk selection.")
    if not bool(model_config.get("hard_candidate_risk_selection", False)):
        raise RuntimeError("The active protocol requires hard risk selection with Local tie-breaking.")
    risk_margin = float(model_config.get("candidate_risk_margin", -1.0))
    if not 0.0 <= risk_margin <= 0.10:
        raise RuntimeError("The candidate-risk margin is invalid.")
    temperature = float(model_config.get("candidate_risk_temperature", -1.0))
    if not 0.0 < temperature <= 0.25:
        raise RuntimeError("The candidate-risk temperature is invalid.")
    if bool(model_config.get("use_amplitude_restoration", False)):
        raise RuntimeError(
            "The active protocol requires amplitude restoration to remain disabled; candidate "
            "endpoints are learned within the current architecture."
        )
    if not bool(model_config.get("fit_visible_phenology_template", False)):
        raise RuntimeError("The active protocol requires the visible-shoulder phenology template fit.")
    if bool(model_config.get("use_solver_acceptance_gate", False)) or bool(
        model_config.get("use_post_solver_acceptance_gate", False)
    ):
        raise RuntimeError(
            "The active protocol prohibits uncalibrated eta heads; Module 2B budget selection is "
            "the sole solver-acceptance decision."
        )
    if not bool(solver_config.get("residual_phenology_mode", False)):
        raise RuntimeError("The active protocol requires residual-domain ST-Tensor/L1.")
    if budget:
        if not bool(model_config.get("use_two_stage_budget_head", False)):
            raise RuntimeError(
                "Budgeting requires independent benefit and conditional-budget heads."
            )
        if not bool(model_config.get("solver_peak_safe_routing", False)):
            raise RuntimeError(
                "Budgeting requires candidate-relative LAI peak-safe acceptance."
            )


def validate_warm_start_protocol(checkpoint: dict) -> None:
    """Require the tile-specific Stage-A foundation used by the frozen controller."""
    actual = _version_text(checkpoint.get("version", ""))
    if actual not in WARM_START_CHECKPOINT_VERSIONS:
        raise RuntimeError(
            "The frozen controller requires the accepted Stage-A foundation as its "
            f"warm start; expected one of {WARM_START_CHECKPOINT_VERSIONS!r}, got {actual!r}."
        )
    protocol = checkpoint.get("quality_protocol", {})
    if _version_text(protocol.get("quality_score_version", "")) != QUALITY_SCORE_VERSION:
        raise RuntimeError("Warm-start checkpoint uses a different compact QA protocol.")
    if abs(float(protocol.get("anchor_threshold", -1.0)) - ANCHOR_THRESHOLD) > 1e-8:
        raise RuntimeError("Warm-start checkpoint uses a different anchor threshold.")


def validate_checkpoint_tile(checkpoint: dict, tile_root: Path) -> None:
    """Reject a controller or budget checkpoint trained for another tile."""
    recorded_root_text = str(checkpoint.get("args", {}).get("root", "")).strip()
    if not recorded_root_text:
        raise RuntimeError("Checkpoint does not record its training tile root.")
    recorded_root = Path(recorded_root_text).resolve()
    requested_root = Path(tile_root).resolve()
    if recorded_root != requested_root:
        raise RuntimeError(
            "Cross-tile checkpoint rejected: "
            f"checkpoint root={recorded_root}, requested root={requested_root}."
        )


def quality_protocol_metadata() -> dict[str, object]:
    return {
        "quality_score_version": QUALITY_SCORE_VERSION,
        "anchor_threshold": ANCHOR_THRESHOLD,
        "phenology_phase_radius_composites": 1,
        "stddev_feature_unsaturated": "log1p(min(std_physical/max(abs(value),floor),2))/log(3)",
        "stddev_feature_saturated": "log1p(min(stddev_decoded,2))/log(3)",
        "hard_qa_role": "continuous_soft_observation",
        "candidate_anchor_role": "strict_unsaturated_or_saturated_main_algorithm_observation",
        "withheld_target_confidence_role": "independent_sensor_evidence_union_not_fixed_three_sensor_division",
        "exact_anchor_role": "strict_unsaturated_candidate_only",
        "saturated_anchor_role": "quality_weighted_soft_anchor_candidate",
        "phenology_support_role": "hard_availability_plus_parent_gate_prior_and_explicit_LLM_candidate_error_context",
        "phenology_dispersion_role": "separate_LLM_routing_context_and_uncertainty_evidence",
        "phenology_candidate_role": "parent_preserving_LLM_selected_legacy_mean_to_peak_continuum",
        "phenology_phase_role": "solver_only_visible_shoulder_alignment_with_affine_error_and_five_percent_gain_guard",
        "phenology_amplitude_role": "solver_only_ridge_regularized_visible_shoulder_fit_without_withheld_target_leakage",
        "hierarchical_group_role": "hard_absolute_LLM_candidate_risk_selection_without_parent_alpha_or_value_averaging_with_Local_tie_breaking_and_explicit_normalized_risk_margin",
        "candidate_error_role": "value_bin_balanced_normalized_residual_consistent_endpoint_error_prediction_weighted_by_squared_candidate_separation",
        "candidate_endpoint_contract": "counterfactual_expert_after_inherited_amplitude_transform_and_decoder_residual_before_observation_projection",
        "candidate_error_training_objective": "from_scratch_selected_variable_reconstruction_and_original_auxiliary_losses_with_endpoint_risk_supervision",
        "solver_domain_role": "ST_Tensor_and_L1_on_bounded_residual_relative_to_optional_fitted_PHENO_while_preserving_the_parent_fusion_prior",
        "solver_acceptance_role": "independent_benefit_then_conditional_budget_on_structural_cells_with_candidate_relative_LAI_peak_safe_acceptance",
        "continuous_control_supervision": "joint_PHENO_mean_peak_selector_and_candidate_endpoint_risk_calibration_balanced_by_physical_value_bin",
        "continuous_control_target_domain": "complete_observation_gaps_without_visible_sensor_evidence",
        "model_selection_role": "validation_complete_gap_normalized_RMSE_and_selected_variable_high_value_RMSE_without_inherited_model",
        "control_calibration_role": "separate_LAI_FPAR_eight_tile_from_scratch_controller_then_Module02C_budget_calibration",
        "uncertainty_calibration_role": "selected_LLM_candidate_risk_plus_data_expert_and_solver_components_then_module2b_validation_complete_gap_scale_calibration_checkpoint_hash_bound",
    }


def normalized_semantic_stddev(
    value: np.ndarray,
    std_dn: np.ndarray,
    valid: np.ndarray,
    scf_qc: np.ndarray,
    variable: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode StdDev according to the retrieval state.

    For SCF_QC=0, StdDev is converted to physical units and divided by the
    retrieved value. For SCF_QC=1, the decoded StdDev is already a normalized
    saturation dispersion and must not be divided a second time.
    """
    params = _PARAMS[variable]
    available = valid & (std_dn != 255)
    saturated = scf_qc.astype(np.uint8) == 1
    std_decoded = std_dn.astype(np.float32) * float(params["std_scale"])
    uncertainty = np.zeros_like(std_decoded, dtype=np.float32)

    unsaturated_available = available & ~saturated
    denominator = np.maximum(
        np.abs(value).astype(np.float32), float(params["value_floor"])
    )
    np.divide(
        std_decoded,
        denominator,
        out=uncertainty,
        where=unsaturated_available,
    )
    uncertainty[available & saturated] = std_decoded[available & saturated]

    cap = float(params["uncertainty_cap"])
    normalized = np.log1p(np.clip(uncertainty, 0.0, cap)) / np.log1p(cap)
    normalized[~available] = 0.0
    return std_decoded, normalized.astype(np.float32)


def calibrated_quality(
    stored_score: np.ndarray,
    value: np.ndarray,
    std_dn: np.ndarray,
    hard_valid: np.ndarray,
    scf_qc: np.ndarray,
    variable: str,
    score_version: object = "",
    anchor_threshold: float = ANCHOR_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read a V36 score and derive graded anchor candidates.

    Strict unsaturated observations must pass the anchor threshold. Saturated
    main-algorithm observations remain candidates even though their QA prior is
    lower; fusion treats them as weighted soft constraints.
    """
    version = _version_text(score_version).lower()
    if version != QUALITY_SCORE_VERSION:
        raise RuntimeError(
            f"Compact quality protocol {version!r} is incompatible with "
            f"{QUALITY_SCORE_VERSION!r}. Rerun Module01_Build_Intermediate_Data.m."
        )
    stored = stored_score.astype(np.float32) / 100.0
    _, std_normalized = normalized_semantic_stddev(
        value, std_dn, hard_valid, scf_qc, variable
    )
    quality = np.clip(stored, 0.0, 1.0).astype(np.float32)
    quality[~hard_valid] = 0.0
    saturated = scf_qc.astype(np.uint8) == 1
    candidate = hard_valid & (
        (saturated & (quality > 0.0))
        | (~saturated & (quality >= float(anchor_threshold)))
    )
    return quality, std_normalized, candidate
