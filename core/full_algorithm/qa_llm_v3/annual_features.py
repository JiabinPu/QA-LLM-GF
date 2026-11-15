"""Short-lived annual feature reuse within one spatial production batch."""
from __future__ import annotations

import torch

from .experts import build_si_experts


class AnnualFeatureCache:
    """Cache only window-independent quantities, never PHENO or learned gates.

    Discard this object after each spatial batch. Active pixel membership is
    part of every key; LC changes cannot reuse a different model batch. Never
    reuse it across validation masking scenarios: these are production inputs.
    """
    def __init__(self):
        self.entries = {}
        self.hits = self.loads = 0
        self.model_id = None

    def prepare(self, model, x, years, active):
        if model.training or torch.is_grad_enabled() or x.device.type != "cpu":
            raise RuntimeError("Annual feature reuse currently requires CPU inference.")
        if x.ndim != 7 or x.shape[1] != len(years) or x.shape[0] != len(active):
            raise ValueError("Annual cache input geometry does not match its keys.")
        if self.model_id is not None and self.model_id != id(model):
            raise ValueError("A feature cache cannot be shared between models.")
        self.model_id = id(model)
        selected = []
        for position, year in enumerate(years):
            key = (int(year), tuple(active))
            if key not in self.entries:
                value = x[:, position].contiguous()
                local, support = build_si_experts(value, return_support=True)
                self.entries[key] = dict(model=model.prepare_annual_features(value),
                                         local=local, support=support)
                self.loads += 1
            else:
                self.hits += 1
            selected.append(self.entries[key])
        def combine(values):
            return torch.stack(values, dim=1).flatten(0, 1)
        return {
            "model": {key: combine([item["model"][key] for item in selected])
                      for key in selected[0]["model"]},
            "local_experts": (combine([item["local"] for item in selected]),
                              combine([item["support"] for item in selected])),
        }


def bounded_year_groups(years, all_years, size, window_size):
    """Bound RAM and HDF5 handles while retaining adjacent-year overlap."""
    from .product import year_window
    if not 1 <= int(size) <= 3:
        raise ValueError("year-group-size must be in 1..3.")
    ordered = sorted(set(map(int, years)))
    group = []
    for year in ordered:
        if group and (len(group) == int(size) or year != group[-1] + 1):
            yield {y: year_window(all_years, y, window_size)[0] for y in group}
            group = []
        group.append(year)
    if group:
        yield {y: year_window(all_years, y, window_size)[0] for y in group}


def cache_fitted_group_size(requested, window_size, cache_mb, source_bytes):
    """Avoid crossing a full-block cache boundary when concurrent workers share RAM."""
    if cache_mb <= 0 or source_bytes <= 0:
        return int(requested)
    source_capacity = int(cache_mb * 1024**2 // source_bytes)
    if source_capacity >= window_size + 1:
        return min(int(requested),source_capacity-int(window_size)+1)
    return int(requested)
