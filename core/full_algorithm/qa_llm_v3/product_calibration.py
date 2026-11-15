"""Checked, one-time application of Module03B to raw model uncertainty."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def sha256(path):
    with Path(path).open('rb') as handle:
        digest = hashlib.sha256()
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


class ProductCalibration:
    def __init__(self, folder, variable, checkpoint):
        folder = Path(folder)
        report_path = folder / 'calibration_report.json'
        report = json.loads(report_path.read_text(encoding='utf-8'))
        table_path = folder / 'product_uncertainty_calibration.csv'
        if (report.get('status'), report.get('variable')) != ('complete', variable):
            raise ValueError('Wrong or incomplete uncertainty calibration')
        if report.get('ground_data_used') is not False or report.get('deployment_allowed') is not True:
            raise ValueError('Calibration is not eligible for product application')
        if report.get('checkpoint_sha256') != sha256(checkpoint):
            raise ValueError('Calibration belongs to another budget checkpoint')
        self.table_hash = sha256(table_path)
        self.report_hash = sha256(report_path)
        if report.get('calibration_sha256') != self.table_hash:
            raise ValueError('Calibration table changed after verification')
        table = pd.read_csv(table_path)
        keys = list(zip(table.QAMode, table.PFTCode))
        expected = {(m, p) for m in range(1, 8) for p in range(1, 9)}
        if len(keys) != 56 or set(keys) != expected or set(table.Variable) != {variable}:
            raise ValueError('Expected exactly 56 unique variable/mode/PFT calibration rows')
        values = table[['ScaleFactor', 'Quantile68', 'Quantile95']].to_numpy(float)
        if not np.isfinite(values).all() or (values[:, 0] <= 0).any() or (values[:, 1:] < 0).any() or (values[:, 2] < values[:, 1]).any():
            raise ValueError('Invalid calibration factors or interval quantiles')
        self.lookup = np.ones((8, 9), dtype=np.float32)
        for row in table.itertuples():
            self.lookup[int(row.QAMode), int(row.PFTCode)] = row.ScaleFactor

    def apply_raw(self, sigma, mode, pft):
        # This API accepts raw hierarchical sigma only, never an encoded product.
        if sigma.shape != mode.shape or pft.shape != (sigma.shape[0], *sigma.shape[-2:]):
            raise ValueError('Uncertainty/mode/PFT shape mismatch')
        if not torch.all((pft >= 0) & (pft <= 8)):
            raise ValueError('Unexpected PFT code')
        lut = torch.as_tensor(self.lookup, device=sigma.device, dtype=sigma.dtype)
        return sigma * lut[mode.long(), pft[:, None].long()]
