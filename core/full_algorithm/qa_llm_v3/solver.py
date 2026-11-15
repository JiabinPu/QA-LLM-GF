"""Compatibility imports for the independent ST-Tensor/L1 module.

Existing scientific callers and training checkpoints retain this module path.
The compiled numerical implementation preserves the original solver operations.
"""

from st_tensor_l1 import SolverConfig, adaptive_l1_trend_filter, adaptive_sttensor
from st_tensor_l1._kernels import (
    _adaptive_sttensor_inference,
    _fold,
    _l1_system,
    _logdet_shrink,
    _low_rank_svd,
    _second_difference,
    _unfold,
)

__all__ = ["SolverConfig", "adaptive_sttensor", "adaptive_l1_trend_filter"]
