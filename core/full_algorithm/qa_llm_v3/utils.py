from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch


VARIABLES = ("LAI", "FPAR")
EXPERTS = ("LIN", "QA_IDW", "CST", "PHENO")
SENSOR_NAMES = ("Terra_MODIS", "Aqua_MODIS", "SNPP_VIIRS")
PHYSICAL_SCALE = (7.0, 1.0)


def parse_items(text: str | list[str] | tuple[str, ...]) -> list[str]:
    if isinstance(text, (list, tuple)):
        return [str(x).strip() for x in text if str(x).strip()]
    return [x.strip() for x in str(text).split(",") if x.strip()]


def parse_ints(text: str | list[int] | tuple[int, ...]) -> list[int]:
    if isinstance(text, (list, tuple)):
        return sorted({int(x) for x in text})
    out: list[int] = []
    for raw in str(text).split(","):
        item = raw.strip()
        if not item:
            continue
        if ":" not in item:
            out.append(int(item))
            continue
        values = [int(x) for x in item.split(":") if x.strip()]
        if len(values) == 2:
            start, stop = values
            step = 1 if stop >= start else -1
        elif len(values) == 3:
            start, step, stop = values
        else:
            raise ValueError(f"Invalid integer selection: {item!r}")
        out.extend(range(start, stop + (1 if step > 0 else -1), step))
    return sorted(set(out))


def resolve_device(name: str) -> torch.device:
    if name not in {"cpu", "cuda", "mps", "auto"}:
        raise ValueError(f"Unknown tensor device: {name}")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    if name == "cuda" or (name == "auto" and torch.cuda.is_available()):
        return torch.device("cuda")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Apple MPS was requested but is unavailable.")
    if name == "mps" or (name == "auto" and torch.backends.mps.is_available()):
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def variable_scale(ref: torch.Tensor, ndim: int | None = None) -> torch.Tensor:
    dims = int(ndim if ndim is not None else ref.ndim)
    shape = [1, 2] + [1] * max(dims - 2, 0)
    return torch.tensor(PHYSICAL_SCALE, dtype=ref.dtype, device=ref.device).view(*shape)


def saturation_aware_target_weights(
    x_window: torch.Tensor,
    center_index: torch.Tensor,
    target_mask_sensor: torch.Tensor,
    target_weight_sensor: torch.Tensor,
    saturated_weight_floor: float,
    target_saturated_sensor: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply one target-weight protocol in training, calibration, and evaluation."""
    if target_mask_sensor.shape != target_weight_sensor.shape:
        raise ValueError("Target masks and weights must have identical shapes.")
    if x_window.ndim != 7 or target_mask_sensor.ndim != 6:
        raise ValueError("Expected window [B,Y,C,T,H,W,S] and target [B,2,T,H,W,S].")
    floor = float(saturated_weight_floor)
    if not 0.0 <= floor <= 1.0:
        raise ValueError("saturated_weight_floor must be between 0 and 1.")
    if floor <= 0.0:
        return target_weight_sensor
    if target_saturated_sensor is None:
        batch = torch.arange(x_window.shape[0], device=x_window.device)
        center = center_index.to(device=x_window.device, dtype=torch.long).reshape(-1)
        saturated = (x_window[batch, center, 18] > 0.5).unsqueeze(1)
        saturated_target = target_mask_sensor.to(torch.bool) & saturated
    else:
        if target_saturated_sensor.shape != target_mask_sensor.shape:
            raise ValueError("target_saturated_sensor must match target masks.")
        saturated_target = (
            target_mask_sensor.to(torch.bool)
            & target_saturated_sensor.to(device=target_mask_sensor.device, dtype=torch.bool)
        )
    floor_tensor = torch.full_like(target_weight_sensor, floor)
    return torch.where(
        saturated_target,
        torch.maximum(target_weight_sensor, floor_tensor),
        target_weight_sensor,
    )


def split_target_domains(
    target_mask: torch.Tensor,
    observation_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split withheld targets by whether another usable sensor remains visible."""
    if target_mask.shape != observation_weight.shape:
        raise ValueError(
            "target_mask and observation_weight must have identical shapes: "
            f"{tuple(target_mask.shape)} != {tuple(observation_weight.shape)}"
        )
    visible = observation_weight > 0.0
    complete_gap = target_mask & ~visible
    partial_sensor = target_mask & visible
    if bool((complete_gap & partial_sensor).any()):
        raise RuntimeError("Target-domain masks must be mutually exclusive.")
    if not torch.equal(complete_gap | partial_sensor, target_mask):
        raise RuntimeError("Target-domain masks must exhaust the withheld targets.")
    return complete_gap, partial_sensor


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def checkpoint_args(checkpoint: dict) -> dict:
    value = checkpoint.get("args", {})
    return dict(value) if isinstance(value, dict) else vars(value)
