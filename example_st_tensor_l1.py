"""Execute one synthetic ST-Tensor/L1 core without loading a learned model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

# Standalone execution from a source checkout needs no editable installation.
_CORE = Path(__file__).resolve().parent / "core" / "full_algorithm"
sys.path.insert(0, str(_CORE))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    args = parser.parse_args()

    # Set PYTORCH_ENABLE_MPS_FALLBACK=1 in the launching environment when needed.
    # The module and this example do not change that process-wide setting.
    import torch

    from st_tensor_l1 import production_config, refine_core

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; use --device cpu or --device auto.")
    if device == "mps" and not torch.backends.mps.is_available():
        parser.error("Apple MPS is unavailable; use --device cpu or --device auto.")

    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        pixel = torch.arange(64, dtype=torch.float32, device=device).view(1, 64, 1, 1)
        time = torch.arange(46, dtype=torch.float32, device=device).view(1, 1, 46, 1)
        year = torch.arange(5, dtype=torch.float32, device=device).view(1, 1, 1, 5)
        phase = time * 0.19 + pixel * 0.07 + year * 0.11
        prior = (0.45 + 0.22 * torch.sin(phase)).contiguous()
        reference = (0.45 + 0.20 * torch.sin(time * 0.19 + year * 0.11)).expand_as(prior).clone()
        observed = (0.48 + 0.29 * torch.sin(phase + 0.35)).clamp(0.0, 1.0)
        weight = (0.15 + 0.65 * ((pixel + time + year) % 4 == 0)).expand_as(prior).clone()
        anchor_mask = torch.zeros_like(prior, dtype=torch.bool)
        anchor_mask[0, 0, 0, 0] = True
        anchor_mask[0, 31, 23, 2] = True
        anchor_mask[0, 63, 45, 4] = True

        result = refine_core(
            observed=observed,
            observation_weight=weight,
            prior=prior,
            prior_strength=torch.tensor([0.8], dtype=prior.dtype, device=device),
            anchor_value=observed.clone(),
            anchor_mask=anchor_mask,
            difficulty=torch.tensor([0.6], dtype=prior.dtype, device=device),
            l1_lambda=torch.tensor([0.025], dtype=prior.dtype, device=device),
            st_budget=torch.tensor([3], dtype=torch.long, device=device),
            phenology_reference=reference,
            config=production_config(),
        )
        if device == "mps":
            torch.mps.synchronize()
        elif device == "cuda":
            torch.cuda.synchronize()

        print(json.dumps({
            "fixture": "synthetic_component_example",
            "device": device,
            "shape": list(result.values.shape),
            "st_budget": result.st_diagnostics["budget"].cpu().tolist(),
            "st_iterations": result.st_diagnostics["iterations"].cpu().tolist(),
            "l1_budget": result.l1_diagnostics["budget"].cpu().tolist(),
            "l1_iterations": result.l1_diagnostics["iterations"].cpu().tolist(),
        }, sort_keys=True))
    finally:
        torch.set_num_threads(previous_threads)


if __name__ == "__main__":
    main()
