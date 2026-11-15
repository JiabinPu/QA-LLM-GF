"""LLM4HRS-style QA controller with a low-cost pixel decoder."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .experts import EXPERT_NAMES, build_si_experts
from .utils import variable_scale


@dataclass
class ControllerConfig:
    active_variable: str | None = None
    input_channels: int = 27
    num_sensors: int = 3
    num_experts: int = len(EXPERT_NAMES)
    backbone: str = "gpt2"
    gpt_model: str = "gpt2"
    gpt_layers: int = 1
    context_dim: int = 64
    local_dim: int = 16
    spatial_grid: int = 4
    dropout: float = 0.1
    residual_scale_lai: float = 0.25
    residual_scale_fpar: float = 0.02
    use_amplitude_restoration: bool = False
    amplitude_log_gain_limit: float = 0.50
    use_phenology_memory: bool = False
    use_hierarchical_phenology_gate: bool = False
    hierarchy_hidden_dim: int = 32
    phenology_phase_radius: int = 1
    fit_visible_phenology_template: bool = False
    phenology_amplitude_min: float = 0.50
    phenology_amplitude_max: float = 2.00
    phenology_minimum_fit_steps: int = 3
    phenology_selector_bias: float = 0.00
    use_solver_acceptance_gate: bool = False
    use_post_solver_acceptance_gate: bool = False
    solver_acceptance_bias: float = 0.00
    use_candidate_error_routing: bool = False
    use_direct_three_candidate_routing: bool = False
    use_absolute_candidate_risk_routing: bool = False
    hard_candidate_risk_selection: bool = False
    candidate_risk_margin: float = 0.01
    candidate_risk_temperature: float = 0.08
    memory_logit_bias: float = -1.50
    memory_support_scale_init: float = 1.00
    memory_dispersion_scale_init: float = 2.00
    train_gpt_mlp: bool = False
    use_budget_head: bool = False
    use_two_stage_budget_head: bool = False
    num_budget_levels: int = 6
    budget_quantiles: tuple[float, float] = (0.5, 0.5)
    budget_need_thresholds: tuple[float, float] = (0.0, 0.0)
    budget_context_dim: int = 4
    budget_pft_codes: tuple[int, ...] = ()
    budget_pft_quantiles: tuple[tuple[float, ...], tuple[float, ...]] = ((), ())
    budget_pft_need_thresholds: tuple[tuple[float, ...], tuple[float, ...]] = ((), ())
    budget_structural_min_core_missing: float = 0.50
    budget_structural_min_run_fraction: float = 3.0 / 46.0
    budget_structural_min_run_steps: int = 3
    budget_structural_max_patch_missing: float = 0.80
    budget_structural_max_support_deficit: float = 0.95
    budget_structural_cell_selective: bool = False
    solver_peak_safe_routing: bool = False
    solver_peak_safe_margin_lai: float = 0.0


class TinyBackbone(nn.Module):
    """Development fallback; the scientific model uses a pretrained GPT-2."""

    def __init__(self, width: int = 128) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=4,
            dim_feedforward=width * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.width = width

    def forward(self, inputs_embeds: torch.Tensor):
        return type("TinyOutput", (), {"last_hidden_state": self.encoder(inputs_embeds)})


class QALLMAdaptiveController(nn.Module):
    """Contextual QA reasoning, expert gating, reliability, and solver control."""

    def __init__(self, config: ControllerConfig) -> None:
        super().__init__()
        self.config = config
        flat_channels = config.input_channels * config.num_sensors

        if config.backbone == "gpt2":
            try:
                from transformers import GPT2Model
            except ImportError as exc:
                raise ImportError("Install transformers to use the GPT-2 QA-LLM backbone.") from exc
            self.backbone = GPT2Model.from_pretrained(
                config.gpt_model,
                output_attentions=False,
                output_hidden_states=False,
                local_files_only=True,
            )
            self.backbone.h = self.backbone.h[: int(config.gpt_layers)]
            self.backbone.config.use_cache = False
            width = int(self.backbone.config.n_embd)
            for name, parameter in self.backbone.named_parameters():
                trainable = "ln" in name or "wpe" in name
                if config.train_gpt_mlp and "mlp" in name:
                    trainable = True
                parameter.requires_grad = trainable
        elif config.backbone == "tiny":
            width = 128
            self.backbone = TinyBackbone(width)
        else:
            raise ValueError(f"Unsupported backbone: {config.backbone!r}")

        self.temporal_input = nn.Sequential(
            nn.LayerNorm(flat_channels * 2),
            nn.Linear(flat_channels * 2, width),
        )
        self.spatial_input = nn.Sequential(
            nn.LayerNorm(flat_channels),
            nn.Linear(flat_channels, width),
        )
        self.temporal_output = nn.Sequential(
            nn.LayerNorm(width * 2),
            nn.Linear(width * 2, config.context_dim),
            nn.GELU(),
        )
        self.spatial_output = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, config.context_dim // 2),
            nn.GELU(),
        )
        self.local_encoder = nn.Sequential(
            nn.Conv2d(flat_channels, config.local_dim, kernel_size=1),
            nn.GELU(),
        )

        decoder_in = config.local_dim + config.context_dim + config.context_dim // 2
        output_channels = 2 * config.num_experts + 2 + 2 * config.num_sensors + 2
        self.pixel_head = nn.Sequential(
            nn.Conv2d(decoder_in, 48, kernel_size=1),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Conv2d(48, output_channels, kernel_size=1),
        )
        self.amplitude_head = (
            nn.Conv2d(decoder_in, 2, kernel_size=1)
            if config.use_amplitude_restoration else None
        )
        if self.amplitude_head is not None:
            nn.init.zeros_(self.amplitude_head.weight)
            nn.init.zeros_(self.amplitude_head.bias)
        self.memory_head = (
            nn.Conv2d(decoder_in, 2, kernel_size=1)
            if config.use_phenology_memory else None
        )
        if self.memory_head is not None:
            nn.init.zeros_(self.memory_head.weight)
            nn.init.constant_(self.memory_head.bias, float(config.memory_logit_bias))
            support_init = max(float(config.memory_support_scale_init), 1e-4)
            dispersion_init = max(float(config.memory_dispersion_scale_init), 1e-4)
            self.memory_support_scale_raw = nn.Parameter(
                torch.full((2,), math.log(math.expm1(support_init)))
            )
            self.memory_dispersion_scale_raw = nn.Parameter(
                torch.full((2,), math.log(math.expm1(dispersion_init)))
            )
        else:
            self.register_parameter("memory_support_scale_raw", None)
            self.register_parameter("memory_dispersion_scale_raw", None)
        hierarchy_in = decoder_in + 14

        def hierarchy_head(initial_bias: float) -> nn.Sequential:
            hidden = max(int(config.hierarchy_hidden_dim), 8)
            head = nn.Sequential(
                nn.Conv2d(hierarchy_in, hidden, kernel_size=1),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Conv2d(hidden, 2, kernel_size=1),
            )
            nn.init.zeros_(head[-1].weight)
            nn.init.constant_(head[-1].bias, float(initial_bias))
            return head

        self.phenology_selector_head = (
            hierarchy_head(config.phenology_selector_bias)
            if config.use_hierarchical_phenology_gate else None
        )
        self.group_gate_correction_head = (
            hierarchy_head(0.0)
            if config.use_hierarchical_phenology_gate else None
        )
        self.candidate_error_head = None
        if config.use_candidate_error_routing:
            hidden = max(int(config.hierarchy_hidden_dim), 8)
            self.candidate_error_head = nn.Sequential(
                nn.Conv2d(hierarchy_in + 10, hidden, kernel_size=1),
                nn.GELU(),
                nn.Conv2d(hidden, 4, kernel_size=1),
            )
            # PHENO must earn selection through calibrated lower risk; it is
            # not mixed into Local merely because its support is nonzero.
            nn.init.zeros_(self.candidate_error_head[-1].weight)
            if config.use_absolute_candidate_risk_routing:
                if config.hard_candidate_risk_selection:
                    # Equal risks are calibrated-neutral; the strict '<'
                    # comparison below resolves ties to Local.
                    initial_raw = math.log(math.expm1(0.20))
                    nn.init.constant_(
                        self.candidate_error_head[-1].bias, initial_raw
                    )
                else:
                    local_raw = math.log(math.expm1(0.05))
                    pheno_raw = math.log(math.expm1(0.75))
                    initial_bias = torch.tensor(
                        (local_raw, pheno_raw, local_raw, pheno_raw),
                        dtype=self.candidate_error_head[-1].bias.dtype,
                    )
                    with torch.no_grad():
                        self.candidate_error_head[-1].bias.copy_(initial_bias)
            else:
                initial_error = 0.10
                initial_raw = math.log(math.expm1(initial_error))
                nn.init.constant_(self.candidate_error_head[-1].bias, initial_raw)
        self.candidate_rank_head = None
        if config.use_direct_three_candidate_routing:
            hidden = max(int(config.hierarchy_hidden_dim), 8)
            self.candidate_rank_head = nn.Sequential(
                nn.Conv2d(hierarchy_in + 12, hidden, kernel_size=1),
                nn.GELU(),
                nn.Conv2d(hidden, 6, kernel_size=1),
            )
            # Equal initial risks make Local the deterministic tie-break. The
            # head must learn a lower risk before either PHENO endpoint enters.
            nn.init.zeros_(self.candidate_rank_head[-1].weight)
            initial_raw = math.log(math.expm1(0.20))
            nn.init.constant_(self.candidate_rank_head[-1].bias, initial_raw)
        self.solver_acceptance_head = (
            hierarchy_head(config.solver_acceptance_bias)
            if config.use_hierarchical_phenology_gate
            and config.use_solver_acceptance_gate
            else None
        )
        self.post_solver_acceptance_head = None
        if config.use_post_solver_acceptance_gate:
            hidden = max(int(config.hierarchy_hidden_dim), 8)
            self.post_solver_acceptance_head = nn.Sequential(
                nn.Conv2d(26, hidden, kernel_size=1),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Conv2d(hidden, 2, kernel_size=1),
            )
            # Zero residual preserves the inherited pre-solver acceptance until
            # the candidate-aware head has been calibrated.
            nn.init.zeros_(self.post_solver_acceptance_head[-1].weight)
            nn.init.zeros_(self.post_solver_acceptance_head[-1].bias)
        budget_classes = (
            config.num_budget_levels - 1
            if config.use_two_stage_budget_head else config.num_budget_levels
        )
        if config.use_budget_head and budget_classes < 1:
            raise ValueError("A solver budget head requires at least two budget levels.")
        self.budget_head = (
            nn.Conv2d(decoder_in, 2 * budget_classes, kernel_size=1)
            if config.use_budget_head else None
        )
        self.budget_benefit_head = (
            nn.Conv2d(decoder_in, 2, kernel_size=1)
            if config.use_budget_head and config.use_two_stage_budget_head else None
        )
        self.budget_context_weight = (
            nn.Parameter(torch.zeros(2, budget_classes, config.budget_context_dim))
            if config.use_budget_head else None
        )
        self.budget_benefit_context_weight = (
            nn.Parameter(torch.zeros(2, config.budget_context_dim))
            if config.use_budget_head and config.use_two_stage_budget_head else None
        )
        self.solver_head = nn.Sequential(
            nn.LayerNorm(config.context_dim),
            nn.Linear(config.context_dim, 48),
            nn.GELU(),
            nn.Linear(48, 8),
        )

    def config_dict(self) -> dict:
        return asdict(self.config)

    def prepare_annual_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Inference-only fields whose reductions do not depend on window peers."""
        if self.training or torch.is_grad_enabled():
            raise RuntimeError("Annual feature reuse is inference-only.")
        b, c, t, h, w, s = x.shape
        flat = x.permute(0, 2, 1, 5, 3, 4).reshape(b, t, c * s, h, w)
        return {
            "temporal_mean": flat.mean(dim=(-1, -2)),
            "temporal_std": flat.var(dim=(-1, -2), unbiased=False).clamp_min(0.0).sqrt(),
            "local": self.local_encoder(flat.reshape(b * t, c * s, h, w)).reshape(
                b, t, self.config.local_dim, h, w
            ),
        }

    def _temporal_context(self, x: torch.Tensor, prepared: dict | None = None) -> torch.Tensor:
        b, c, t, h, w, s = x.shape
        if prepared is None:
            flat = x.permute(0, 2, 1, 5, 3, 4).reshape(b, t, c * s, h, w)
            mean = flat.mean(dim=(-1, -2))
            std = flat.var(dim=(-1, -2), unbiased=False).clamp_min(0.0).sqrt()
        else:
            mean, std = prepared["temporal_mean"], prepared["temporal_std"]
            if mean.shape != (b, t, c * s) or std.shape != mean.shape:
                raise ValueError("Annual temporal statistics have incompatible geometry.")
        token = self.temporal_input(torch.cat([mean, std], dim=-1))
        backward_token = torch.flip(token, dims=[1])
        encoded = self.backbone(inputs_embeds=torch.cat([token, backward_token], dim=0)).last_hidden_state
        forward, backward = encoded.split(b, dim=0)
        backward = torch.flip(backward, dims=[1])
        return self.temporal_output(torch.cat([forward, backward], dim=-1))

    def _spatial_context(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _t, h, w, s = x.shape
        spatial = x.mean(dim=2).permute(0, 1, 4, 2, 3).reshape(b, c * s, h, w)
        pooled = F.adaptive_avg_pool2d(spatial, (self.config.spatial_grid, self.config.spatial_grid))
        token = pooled.flatten(2).transpose(1, 2)
        encoded = self.backbone(inputs_embeds=self.spatial_input(token)).last_hidden_state
        context = self.spatial_output(encoded).transpose(1, 2).reshape(
            b,
            self.config.context_dim // 2,
            self.config.spatial_grid,
            self.config.spatial_grid,
        )
        return F.interpolate(context, size=(h, w), mode="bilinear", align_corners=False)

    def forward(
        self,
        x: torch.Tensor,
        experts: torch.Tensor | tuple[torch.Tensor, ...] | None = None,
        return_budget_features: bool = False,
        prepared_annual: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        if x.ndim != 6:
            raise ValueError(f"Expected [B,C,T,H,W,S], got {tuple(x.shape)}")
        b, c, t, h, w, s = x.shape
        if c != self.config.input_channels or s != self.config.num_sensors:
            raise ValueError(
                f"Expected C={self.config.input_channels}, S={self.config.num_sensors}; got C={c}, S={s}."
            )
        if experts is None:
            expert_values, expert_support = build_si_experts(x, return_support=True)
            if self.config.num_experts == len(EXPERT_NAMES):
                # A single-year call has no cross-year phenology evidence.
                # Duplicate LIN as a numerically safe value and assign zero
                # PHENO support; multi-year scientific runs replace it with the
                # actual PHENO expert in prepare_window_frontend().
                expert_values = torch.cat(
                    (expert_values, expert_values[:, :, :1]), dim=2
                )
                expert_support = torch.cat(
                    (expert_support, torch.zeros_like(expert_support[:, :, :1])), dim=2
                )
            phenology_peak = expert_values[:, :, -1]
            phenology_dispersion = torch.ones_like(phenology_peak)
        elif isinstance(experts, tuple):
            if len(experts) == 2:
                expert_values, expert_support = experts
                phenology_peak = expert_values[:, :, -1]
                phenology_dispersion = torch.ones_like(phenology_peak)
            elif len(experts) == 4:
                expert_values, expert_support, phenology_peak, phenology_dispersion = experts
            else:
                raise ValueError("experts tuple must contain 2 or 4 tensors.")
        else:
            expert_values = experts
            expert_support = torch.ones_like(expert_values)
            phenology_peak = expert_values[:, :, -1]
            phenology_dispersion = torch.ones_like(phenology_peak)
        if expert_values.shape[2] != self.config.num_experts:
            raise ValueError(
                f"Controller expects {self.config.num_experts} experts, got "
                f"{expert_values.shape[2]}. Multi-year runs must use build_window_experts()."
            )
        if prepared_annual is not None and (self.training or torch.is_grad_enabled()):
            raise RuntimeError("Annual feature reuse is inference-only.")
        temporal = self._temporal_context(x, prepared_annual)
        spatial = self._spatial_context(x)
        if prepared_annual is None:
            local = x.permute(0, 2, 1, 5, 3, 4).reshape(b * t, c * s, h, w)
            local = self.local_encoder(local)
        else:
            local = prepared_annual["local"]
            if local.shape != (b, t, self.config.local_dim, h, w):
                raise ValueError("Annual local features have incompatible geometry.")
            local = local.reshape(b * t, self.config.local_dim, h, w)
        temporal_map = temporal.reshape(b * t, self.config.context_dim, 1, 1).expand(-1, -1, h, w)
        spatial_map = spatial[:, None].expand(-1, t, -1, -1, -1).reshape(
            b * t, self.config.context_dim // 2, h, w
        )
        decoder_features = torch.cat([local, temporal_map, spatial_map], dim=1)
        raw = self.pixel_head(decoder_features)
        raw = raw.reshape(b, t, raw.shape[1], h, w).permute(0, 2, 1, 3, 4)

        k = self.config.num_experts
        gate_logits = raw[:, : 2 * k].reshape(b, 2, k, t, h, w)
        residual_scale = torch.tensor(
            [self.config.residual_scale_lai, self.config.residual_scale_fpar],
            dtype=x.dtype,
            device=x.device,
        ).view(1, 2, 1, 1, 1)
        residual = torch.tanh(raw[:, 2 * k : 2 * k + 2]) * residual_scale
        available = expert_support > 0.0
        no_available = ~available.any(dim=2, keepdim=True)
        available = available.clone()
        available[:, :, :1] |= no_available
        support_logits = torch.log(expert_support.clamp_min(1e-6))
        if self.phenology_selector_head is not None:
            local_available = available[:, :, :3].clone()
            no_local = ~local_available.any(dim=2, keepdim=True)
            local_available[:, :, :1] |= no_local
            local_logits = (
                gate_logits[:, :, :3] + support_logits[:, :, :3]
            ).masked_fill(~local_available, -1e4)
            local_gate_weights = torch.softmax(local_logits, dim=2)
            local_fused = (local_gate_weights * expert_values[:, :, :3]).sum(dim=2)
            local_support = (
                local_gate_weights * expert_support[:, :, :3]
            ).sum(dim=2).clamp(0.0, 1.0)
            phenology_mean = expert_values[:, :, 3]
            physical_scale = torch.tensor(
                (7.0, 1.0), dtype=x.dtype, device=x.device
            ).view(1, 2, 1, 1, 1)

            def as_decoder_channels(value: torch.Tensor) -> torch.Tensor:
                return value.permute(0, 2, 1, 3, 4).reshape(b * t, 2, h, w)

            hierarchy_features = torch.cat(
                [
                    decoder_features,
                    as_decoder_channels(local_fused / physical_scale),
                    as_decoder_channels(phenology_mean / physical_scale),
                    as_decoder_channels(phenology_peak / physical_scale),
                    as_decoder_channels((phenology_peak - phenology_mean) / physical_scale),
                    as_decoder_channels(local_support),
                    as_decoder_channels(expert_support[:, :, 3]),
                    as_decoder_channels(phenology_dispersion.clamp(0.0, 1.0)),
                ],
                dim=1,
            )
            correction_raw = self._group_correction(hierarchy_features)
            group_correction = correction_raw.reshape(b, t, 2, h, w).permute(
                0, 2, 1, 3, 4
            )
            # With a zero correction head this reproduces the old flat
            # four-expert gate exactly, while exposing an explicit group-level
            # decision for subsequent learning.
            local_group_logit = torch.logsumexp(local_logits, dim=2)
            group_logit = (
                gate_logits[:, :, 3]
                + torch.log(expert_support[:, :, 3].clamp_min(1e-6))
                - local_group_logit
                + group_correction
            )
            parent_phenology_group_gate = torch.sigmoid(group_logit)
            local_delta = (
                expert_values[:, :, :3] - local_fused.unsqueeze(2)
            ) / physical_scale.unsqueeze(2)
            local_dispersion = (
                local_gate_weights * local_delta.square()
            ).sum(dim=2).clamp_min(0.0)
            # The exact zero-variance case is common before task-head
            # calibration. sqrt(0) has an unbounded derivative and produced
            # NaN gradients in genuine from-scratch multi-tile training.
            local_dispersion = (local_dispersion + 1e-8).sqrt()
            candidate_risk = torch.zeros(
                (b, 2, 3, t, h, w), dtype=x.dtype, device=x.device
            )
            candidate_index = torch.zeros(
                (b, 2, t, h, w), dtype=torch.long, device=x.device
            )
            if self.candidate_rank_head is not None:
                rank_features = torch.cat(
                    [
                        hierarchy_features,
                        as_decoder_channels((phenology_mean - local_fused) / physical_scale),
                        as_decoder_channels((phenology_peak - local_fused) / physical_scale),
                        as_decoder_channels(
                            (phenology_mean - local_fused).abs() / physical_scale
                        ),
                        as_decoder_channels(
                            (phenology_peak - local_fused).abs() / physical_scale
                        ),
                        as_decoder_channels(local_dispersion),
                        as_decoder_channels(residual / physical_scale),
                    ],
                    dim=1,
                )
                rank_raw = self.candidate_rank_head(rank_features)
                candidate_risk = F.softplus(
                    rank_raw.reshape(b, t, 2, 3, h, w).permute(0, 2, 3, 1, 4, 5)
                ).clamp_min(1e-4)
                pheno_available = expert_support[:, :, 3] > 0.0
                has_local_evidence = (expert_support[:, :, :3] > 0.0).any(dim=2)
                candidate_available = torch.stack(
                    (has_local_evidence, pheno_available, pheno_available), dim=2
                )
                rank_score = candidate_risk.clone()
                rank_score[:, :, 1:] = (
                    rank_score[:, :, 1:] + float(self.config.candidate_risk_margin)
                )
                rank_score = rank_score.masked_fill(~candidate_available, float("inf"))
                candidate_index = rank_score.argmin(dim=2)
                # Keep a best-PHENO indicator for the residual solver reference,
                # even where Local wins the pre-solver candidate decision.
                phenology_selector = (
                    candidate_risk[:, :, 2] < candidate_risk[:, :, 1]
                ).to(x.dtype)
                adaptive_phenology = torch.where(
                    phenology_selector > 0.5, phenology_peak, phenology_mean
                )
                phenology_group_gate = (candidate_index > 0).to(x.dtype)
                selected_pheno_risk = torch.where(
                    phenology_selector > 0.5,
                    candidate_risk[:, :, 2],
                    candidate_risk[:, :, 1],
                )
                candidate_error = torch.stack(
                    (candidate_risk[:, :, 0], selected_pheno_risk), dim=2
                )
                candidate_error_gate = phenology_group_gate
            else:
                selector_raw = self.phenology_selector_head(hierarchy_features)
                phenology_selector = torch.sigmoid(
                    selector_raw.reshape(b, t, 2, h, w).permute(0, 2, 1, 3, 4)
                )
                adaptive_phenology = phenology_mean + phenology_selector * (
                    phenology_peak - phenology_mean
                )
            expert_values = torch.cat(
                (expert_values[:, :, :3], adaptive_phenology.unsqueeze(2)), dim=2
            )
            if self.candidate_rank_head is None and self.candidate_error_head is not None:
                reliability_features = torch.cat(
                    [
                        hierarchy_features,
                        as_decoder_channels(adaptive_phenology / physical_scale),
                        as_decoder_channels(
                            (adaptive_phenology - local_fused) / physical_scale
                        ),
                        as_decoder_channels(
                            (adaptive_phenology - local_fused).abs() / physical_scale
                        ),
                        as_decoder_channels(local_dispersion),
                        as_decoder_channels(residual / physical_scale),
                    ],
                    dim=1,
                )
                error_raw = self.candidate_error_head(reliability_features)
                candidate_error = F.softplus(
                    error_raw.reshape(b, t, 2, 2, h, w).permute(0, 2, 3, 1, 4, 5)
                ).clamp_min(1e-4)
                local_error = candidate_error[:, :, 0]
                phenology_error = candidate_error[:, :, 1]
                if self.config.use_absolute_candidate_risk_routing:
                    if self.config.hard_candidate_risk_selection:
                        # The risk head is supervised directly, so routing does
                        # not need a reconstruction gradient through this
                        # decision. A normalized minimum advantage prevents
                        # numerical near-ties from flipping large map areas.
                        candidate_error_gate = (
                            phenology_error
                            + float(self.config.candidate_risk_margin)
                            < local_error
                        ).to(local_error.dtype)
                    else:
                        temperature = max(
                            float(self.config.candidate_risk_temperature), 1e-4
                        )
                        candidate_error_gate = torch.sigmoid(
                            (local_error - phenology_error) / temperature
                        )
                    phenology_group_gate = candidate_error_gate
                else:
                    candidate_error_gate = local_error / (
                        local_error + phenology_error
                    ).clamp_min(1e-6)
                    parent_logit = torch.logit(
                        parent_phenology_group_gate.clamp(1e-4, 1.0 - 1e-4)
                    )
                    error_log_odds = torch.logit(
                        candidate_error_gate.clamp(1e-4, 1.0 - 1e-4)
                    )
                    phenology_group_gate = torch.sigmoid(
                        parent_logit + error_log_odds
                    )
            elif self.candidate_rank_head is None:
                candidate_error = torch.zeros(
                    (b, 2, 2, t, h, w), dtype=x.dtype, device=x.device
                )
                candidate_error_gate = torch.full_like(
                    parent_phenology_group_gate, 0.5
                )
                phenology_group_gate = parent_phenology_group_gate
            pheno_available = expert_support[:, :, 3] > 0.0
            has_local_evidence = (expert_support[:, :, :3] > 0.0).any(dim=2)
            phenology_group_gate = torch.where(
                pheno_available,
                torch.where(
                    has_local_evidence,
                    phenology_group_gate,
                    torch.ones_like(phenology_group_gate),
                ),
                torch.zeros_like(phenology_group_gate),
            )
            gate_weights = torch.cat(
                (
                    (1.0 - phenology_group_gate).unsqueeze(2) * local_gate_weights,
                    phenology_group_gate.unsqueeze(2),
                ),
                dim=2,
            )
            if self.solver_acceptance_head is not None:
                solver_acceptance_raw = self.solver_acceptance_head(hierarchy_features)
                solver_acceptance = torch.sigmoid(
                    solver_acceptance_raw.reshape(b, t, 2, h, w).permute(
                        0, 2, 1, 3, 4
                    )
                )
            else:
                solver_acceptance = torch.ones_like(phenology_group_gate)
        else:
            gated_logits = (gate_logits + support_logits).masked_fill(~available, -1e4)
            gate_weights = torch.softmax(gated_logits, dim=2)
            phenology_mean = expert_values[:, :, 3]
            phenology_selector = torch.zeros_like(phenology_mean)
            phenology_group_gate = gate_weights[:, :, 3]
            local_gate_weights = gate_weights[:, :, :3]
            local_gate_weights = local_gate_weights / local_gate_weights.sum(
                dim=2, keepdim=True
            ).clamp_min(1e-8)
            local_fused = (local_gate_weights * expert_values[:, :, :3]).sum(dim=2)
            local_support = (
                local_gate_weights * expert_support[:, :, :3]
            ).sum(dim=2).clamp(0.0, 1.0)
            solver_acceptance = torch.ones_like(phenology_group_gate)
            parent_phenology_group_gate = phenology_group_gate
            candidate_error = torch.zeros(
                (b, 2, 2, t, h, w), dtype=x.dtype, device=x.device
            )
            candidate_error_gate = torch.full_like(phenology_group_gate, 0.5)
            candidate_risk = torch.zeros(
                (b, 2, 3, t, h, w), dtype=x.dtype, device=x.device
            )
            candidate_index = torch.zeros(
                (b, 2, t, h, w), dtype=torch.long, device=x.device
            )
        fused = (gate_weights * expert_values).sum(dim=2)
        seasonal_baseline = fused.mean(dim=2, keepdim=True)
        if self.amplitude_head is not None:
            amplitude_raw = self.amplitude_head(decoder_features)
            amplitude_raw = amplitude_raw.reshape(b, t, 2, h, w).permute(
                0, 2, 1, 3, 4
            )
            amplitude_gain = torch.exp(
                float(self.config.amplitude_log_gain_limit) * torch.tanh(amplitude_raw)
            )
        else:
            amplitude_gain = torch.ones_like(fused)
        amplitude_prior = seasonal_baseline + amplitude_gain * (fused - seasonal_baseline)
        prior = amplitude_prior + residual
        prior = torch.stack([prior[:, 0].clamp(0.0, 7.0), prior[:, 1].clamp(0.0, 1.0)], dim=1)
        if self.memory_head is not None:
            memory_logits = self.memory_head(decoder_features)
            memory_logits = memory_logits.reshape(b, t, 2, h, w).permute(
                0, 2, 1, 3, 4
            )
        else:
            memory_logits = torch.full_like(prior, -20.0)

        offset = 2 * k + 2
        reliability = torch.sigmoid(raw[:, offset : offset + 2 * s]).reshape(b, 2, s, t, h, w)
        reliability = reliability.permute(0, 1, 3, 4, 5, 2).contiguous()
        log_variance = raw[:, offset + 2 * s : offset + 2 * s + 2].clamp(-8.0, 4.0)

        patch_context = temporal.mean(dim=1)
        controls = self.solver_head(patch_context).reshape(b, 2, 4)
        difficulty = torch.sigmoid(controls[:, :, 0])
        prior_strength = 0.05 + 1.95 * torch.sigmoid(controls[:, :, 1])
        l1_lambda = 0.002 + 0.048 * torch.sigmoid(controls[:, :, 2])
        continuation = torch.sigmoid(controls[:, :, 3])

        normalized_prior = prior / variable_scale(prior, prior.ndim)
        result = {
            "prior": prior,
            "normalized_prior": normalized_prior,
            "residual": residual,
            "fused": fused,
            "amplitude_prior": amplitude_prior,
            "amplitude_gain": amplitude_gain,
            "seasonal_baseline": seasonal_baseline,
            "memory_logits": memory_logits,
            "experts": expert_values,
            "expert_support": expert_support,
            "gate_logits": gate_logits,
            "gate_weights": gate_weights,
            "local_gate_weights": local_gate_weights,
            "local_fused": local_fused,
            "local_support": local_support,
            "phenology_mean": phenology_mean,
            "phenology_peak": phenology_peak,
            "phenology_selector": phenology_selector,
            "phenology_group_gate": phenology_group_gate,
            "parent_phenology_group_gate": parent_phenology_group_gate,
            "candidate_error": candidate_error,
            "candidate_error_gate": candidate_error_gate,
            "candidate_risk": candidate_risk,
            "candidate_index": candidate_index,
            "solver_acceptance": solver_acceptance,
            "reliability": reliability,
            "log_variance": log_variance,
            "difficulty": difficulty,
            "prior_strength": prior_strength,
            "l1_lambda": l1_lambda,
            "continuation": continuation,
            "temporal_context": temporal,
            "spatial_context": spatial,
        }
        if self.budget_head is not None:
            budget_raw = self.budget_head(decoder_features)
            if self.config.use_two_stage_budget_head:
                if self.budget_benefit_head is None:
                    raise RuntimeError("Two-stage budgeting has no benefit head.")
                positive_logits = budget_raw.reshape(
                    b, t, 2, self.config.num_budget_levels - 1, h, w
                ).permute(0, 2, 3, 1, 4, 5).contiguous()
                benefit_logit = self.budget_benefit_head(decoder_features).reshape(
                    b, t, 2, h, w
                ).permute(0, 2, 1, 3, 4).contiguous()
                # The shared composer expects the class axis last. Restore the
                # pipeline contract [B,V,L,T,H,W] after composing probabilities.
                composed = self._compose_budget_logits(
                    benefit_logit,
                    positive_logits.permute(0, 1, 3, 4, 5, 2),
                )
                result["budget_logits"] = composed.permute(0, 1, 5, 2, 3, 4)
                result["budget_benefit_logit"] = benefit_logit
                result["budget_positive_logits"] = positive_logits
            else:
                result["budget_logits"] = budget_raw.reshape(
                    b, t, 2, self.config.num_budget_levels, h, w
                ).permute(0, 2, 3, 1, 4, 5).contiguous()
            if return_budget_features:
                result["budget_features"] = decoder_features.reshape(
                    b, t, decoder_features.shape[1], h, w
                ).permute(0, 2, 1, 3, 4).contiguous()
        return result

    def _group_correction(self, features: torch.Tensor) -> torch.Tensor:
        head = self.group_gate_correction_head
        if (getattr(self, "_fold_constant_heads", False) and not self.training
                and not torch.is_grad_enabled() and features.device.type == "cpu"
                and features.dtype == torch.float32 and not torch.is_autocast_enabled("cpu")):
            last = head[-1]
            if (isinstance(last, nn.Conv2d) and last.kernel_size == (1, 1)
                    and int(torch.count_nonzero(last.weight)) == 0
                    and last.bias is not None and int(torch.count_nonzero(last.bias)) == 0):
                # Scalar extrema avoid allocating a full-size isfinite mask.
                low, high = torch.aminmax(features)
                if bool(torch.isfinite(low) & torch.isfinite(high)):
                    return features.new_zeros((features.shape[0], last.out_channels, *features.shape[2:]))
        return head(features)

    def refine_solver_acceptance(
        self,
        data_consistent_prior: torch.Tensor,
        solver_candidate: torch.Tensor,
        phenology_peak: torch.Tensor,
        adaptive_phenology: torch.Tensor,
        local_fused: torch.Tensor,
        phenology_support: torch.Tensor,
        phenology_dispersion: torch.Tensor,
        phenology_selector: torch.Tensor,
        phenology_group_gate: torch.Tensor,
        pre_solver_acceptance: torch.Tensor,
        solver_budget_fraction: torch.Tensor,
    ) -> torch.Tensor:
        """Accept an actual ST/L1 candidate using LLM context and candidate deltas."""
        if self.post_solver_acceptance_head is None:
            return pre_solver_acceptance.clamp(0.0, 1.0)
        tensors = (
            data_consistent_prior, solver_candidate, phenology_peak,
            adaptive_phenology, local_fused, phenology_support,
            phenology_dispersion, phenology_selector, phenology_group_gate,
            pre_solver_acceptance, solver_budget_fraction,
        )
        if any(value.shape != data_consistent_prior.shape for value in tensors):
            raise ValueError("Post-solver acceptance inputs must share [B,2,T,H,W].")
        scale = variable_scale(data_consistent_prior, data_consistent_prior.ndim)
        delta = (solver_candidate - data_consistent_prior) / scale
        features = torch.cat(
            (
                data_consistent_prior / scale,
                solver_candidate / scale,
                delta,
                delta.abs(),
                phenology_peak / scale,
                adaptive_phenology / scale,
                local_fused / scale,
                phenology_support.clamp(0.0, 1.0),
                phenology_dispersion.clamp(0.0, 1.0),
                phenology_selector.clamp(0.0, 1.0),
                phenology_group_gate.clamp(0.0, 1.0),
                pre_solver_acceptance.clamp(0.0, 1.0),
                solver_budget_fraction.clamp(0.0, 1.0),
            ),
            dim=1,
        )
        b, _c, t, h, w = features.shape
        flat = features.permute(0, 2, 1, 3, 4).reshape(b * t, 26, h, w)
        correction = self.post_solver_acceptance_head(flat)
        correction = correction.reshape(b, t, 2, h, w).permute(0, 2, 1, 3, 4)
        base_logit = torch.logit(pre_solver_acceptance.clamp(1e-4, 1.0 - 1e-4))
        return torch.sigmoid(base_logit + correction)

    def memory_gate_coefficients(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return positive learned coefficients for support and dispersion."""
        if self.memory_support_scale_raw is None or self.memory_dispersion_scale_raw is None:
            reference = next(self.parameters())
            zero = torch.zeros(2, dtype=reference.dtype, device=reference.device)
            return zero, zero
        return (
            F.softplus(self.memory_support_scale_raw),
            F.softplus(self.memory_dispersion_scale_raw),
        )

    def add_budget_context(
        self, logits: torch.Tensor, context: torch.Tensor | None
    ) -> torch.Tensor:
        if context is None:
            return logits
        if self.budget_context_weight is None:
            raise RuntimeError("This controller was created without budget context weights.")
        if logits.ndim == 3:
            adjustment = torch.einsum("nvd,vld->nvl", context, self.budget_context_weight)
            benefit_adjustment = (
                torch.einsum("nvd,vd->nv", context, self.budget_benefit_context_weight)
                if self.config.use_two_stage_budget_head else None
            )
        elif logits.ndim == 4:
            adjustment = torch.einsum("bvnd,vld->bvnl", context, self.budget_context_weight)
            benefit_adjustment = (
                torch.einsum("bvnd,vd->bvn", context, self.budget_benefit_context_weight)
                if self.config.use_two_stage_budget_head else None
            )
        else:
            raise ValueError(f"Unsupported budget-logit shape: {tuple(logits.shape)}")
        if not self.config.use_two_stage_budget_head:
            return logits + adjustment
        if benefit_adjustment is None or self.budget_benefit_context_weight is None:
            raise RuntimeError("Two-stage budgeting has no benefit-context weights.")
        benefit_logit = torch.logsumexp(logits[..., 1:], dim=-1) - logits[..., 0]
        return self._compose_budget_logits(
            benefit_logit + benefit_adjustment,
            logits[..., 1:] + adjustment,
        )

    @staticmethod
    def _compose_budget_logits(
        benefit_logit: torch.Tensor, positive_logits: torch.Tensor
    ) -> torch.Tensor:
        """Compose independent benefit and conditional-budget probabilities."""
        log_no_solver = F.logsigmoid(-benefit_logit).unsqueeze(-1)
        log_need_solver = F.logsigmoid(benefit_logit).unsqueeze(-1)
        log_positive = F.log_softmax(positive_logits, dim=-1)
        return torch.cat((log_no_solver, log_need_solver + log_positive), dim=-1)

    def budget_logits_from_core_features(
        self, features: torch.Tensor, context: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Apply the lightweight budget head to detached [N,D] core features."""
        if self.budget_head is None:
            raise RuntimeError("This controller was created without a solver budget head.")
        weight = self.budget_head.weight[:, :, 0, 0]
        raw = F.linear(features, weight, self.budget_head.bias)
        if self.config.use_two_stage_budget_head:
            if self.budget_benefit_head is None:
                raise RuntimeError("Two-stage budgeting has no benefit head.")
            positive = raw.reshape(features.shape[0], 2, self.config.num_budget_levels - 1)
            benefit_weight = self.budget_benefit_head.weight[:, :, 0, 0]
            benefit = F.linear(
                features, benefit_weight, self.budget_benefit_head.bias
            ).reshape(features.shape[0], 2)
            logits = self._compose_budget_logits(benefit, positive)
        else:
            logits = raw.reshape(features.shape[0], 2, self.config.num_budget_levels)
        return self.add_budget_context(logits, context)


def build_controller(config: dict | ControllerConfig) -> QALLMAdaptiveController:
    return QALLMAdaptiveController(config if isinstance(config, ControllerConfig) else ControllerConfig(**config))
