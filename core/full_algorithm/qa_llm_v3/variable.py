"""Single-target isolation while retaining the original solver tensor layout.

The inactive slot is padding, not a second supervised task. Keeping original
slot positions preserves LAI/FPAR-specific scales and solver bounds.
"""
from __future__ import annotations

import numpy as np


def variable_index(variable: str) -> int:
    if variable not in ('LAI', 'FPAR'):
        raise ValueError('A separate LAI or FPAR variable is required.')
    return ('LAI', 'FPAR').index(variable)


def isolate_token(token: dict, variable: str | None) -> dict:
    if variable is None:
        return token
    inactive = ('fpar' if variable_index(variable) == 0 else 'lai')
    result = dict(token)
    for key in (f'{inactive}_dn', f'{inactive}_std_dn'):
        result[key] = np.full_like(token[key], 255)
    for key in (f'flag_{inactive}', f'score_{inactive}'):
        result[key] = np.zeros_like(token[key])
    return result


def assert_single_batch(batch: dict, variable: str) -> None:
    inactive = 1 - variable_index(variable)
    x = batch['x_window']
    for channel in (inactive, 2 + inactive, 4 + inactive, 6 + inactive, 16 + inactive):
        if x[:, :, channel].count_nonzero().item():
            raise ValueError(f'Inactive variable input is not zero: channel {channel}')
    for name in ('target_mask', 'target_weight', 'target_saturated'):
        if batch[name][:, inactive].count_nonzero().item():
            raise ValueError(f'Inactive variable contributes to {name}')
