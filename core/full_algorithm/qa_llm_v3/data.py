"""Windowed multi-year dataset backed by the existing compact Module 1 HDF5 files."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from qg_dataset import DatasetOptions, QGLAIFPARDataset, bit

from .quality import calibrated_quality
from .variable import isolate_token


def build_input(
    token: dict[str, np.ndarray],
    year: int,
    sensor_start_years: Sequence[int],
    lc_type3: np.ndarray | None,
    lc_codes: Sequence[int],
    preallocate: bool = False,
    active_variable: str | None = None,
) -> np.ndarray:
    token = isolate_token(token, active_variable)
    lai_dn = token["lai_dn"]
    fpar_dn = token["fpar_dn"]
    score_lai = token["score_lai"]
    score_fpar = token["score_fpar"]
    flag_lai = token["flag_lai"]
    flag_fpar = token["flag_fpar"]
    lai_std_dn = token["lai_std_dn"]
    fpar_std_dn = token["fpar_std_dn"]
    scf_qc = token["scf_qc"]
    doy = token["doy"]
    h, w, t, sensors = lai_dn.shape

    active = np.asarray([year >= int(start) for start in sensor_start_years[:sensors]], dtype=bool)
    active = np.broadcast_to(active.reshape(1, 1, 1, sensors), (h, w, t, sensors))
    hard_lai, anchor_lai, valid_lai = bit(flag_lai, 1), bit(flag_lai, 2), bit(flag_lai, 3)
    hard_fpar, anchor_fpar, valid_fpar = bit(flag_fpar, 1), bit(flag_fpar, 2), bit(flag_fpar, 3)
    valid_lai_input = active & hard_lai & valid_lai & (lai_dn != 255)
    valid_fpar_input = active & hard_fpar & valid_fpar & (fpar_dn != 255)
    anchor_lai = active & anchor_lai & valid_lai & (lai_dn != 255)
    anchor_fpar = active & anchor_fpar & valid_fpar & (fpar_dn != 255)

    lai = lai_dn.astype(np.float32) * 0.1
    fpar = fpar_dn.astype(np.float32) * 0.01
    lai[~valid_lai_input] = 0.0
    fpar[~valid_fpar_input] = 0.0
    score_version = token.get("quality_score_version", "")
    qa_lai, lai_std, anchor_lai = calibrated_quality(
        score_lai, lai, lai_std_dn, valid_lai_input, scf_qc, "LAI", score_version
    )
    qa_fpar, fpar_std, anchor_fpar = calibrated_quality(
        score_fpar, fpar, fpar_std_dn, valid_fpar_input, scf_qc, "FPAR", score_version
    )
    qa_lai[~active] = 0.0
    qa_fpar[~active] = 0.0
    lai_std[~valid_lai_input] = 0.0
    fpar_std[~valid_fpar_input] = 0.0
    sensor_id = np.arange(sensors, dtype=np.float32) / max(sensors - 1, 1)
    sensor_id = np.broadcast_to(sensor_id.reshape(1, 1, 1, sensors), (h, w, t, sensors))
    doy = np.asarray(doy[:t], dtype=np.float32).reshape(1, 1, t, 1)
    doy_sin = np.broadcast_to(np.sin(2 * np.pi * (doy - 1) / 365.0), (h, w, t, sensors)).astype(np.float32)
    doy_cos = np.broadcast_to(np.cos(2 * np.pi * (doy - 1) / 365.0), (h, w, t, sensors)).astype(np.float32)

    primary = bit(flag_lai, 4) | bit(flag_fpar, 4)
    backup = bit(flag_lai, 5) | bit(flag_fpar, 5)
    atmosphere = bit(flag_lai, 6) | bit(flag_fpar, 6)
    land = bit(flag_lai, 7) | bit(flag_fpar, 7)
    cloud = bit(flag_lai, 8) | bit(flag_fpar, 8)
    channels = [
        lai,
        fpar,
        valid_lai_input.astype(np.float32),
        valid_fpar_input.astype(np.float32),
        qa_lai,
        qa_fpar,
        anchor_lai.astype(np.float32),
        anchor_fpar.astype(np.float32),
        primary.astype(np.float32),
        backup.astype(np.float32),
        atmosphere.astype(np.float32),
        land.astype(np.float32),
        cloud.astype(np.float32),
        sensor_id,
        doy_sin,
        doy_cos,
        lai_std.astype(np.float32),
        fpar_std.astype(np.float32),
        (scf_qc == 1).astype(np.float32),
    ]
    if preallocate:
        x = np.empty((h, w, t, sensors, len(channels) + len(lc_codes)), dtype=np.float32)
        for index, values in enumerate(channels):
            x[..., index] = values
    else:
        x = np.stack(channels, axis=-1)
    if lc_codes:
        if lc_type3 is None:
            raise ValueError("LC_Type_3 context is required but missing.")
        lc = np.asarray(lc_type3, dtype=np.uint8)
        if preallocate:
            for index, code in enumerate(lc_codes, start=len(channels)):
                x[..., index] = (lc == int(code)).reshape(h, w, 1, 1)
        else:
            one_hot = [
                np.broadcast_to((lc == int(code)).reshape(h, w, 1, 1), (h, w, t, sensors)).astype(np.float32)
                for code in lc_codes
            ]
            x = np.concatenate([x, np.stack(one_hot, axis=-1)], axis=-1)
    return np.transpose(x, (4, 2, 0, 1, 3)).astype(np.float32, copy=not preallocate)


class WindowedQADataset(Dataset):
    """Use a center-year mask with neighboring years from the same spatial patch."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        scenarios: Sequence[str],
        window_years: int,
        sensor_start_years: Sequence[int],
        options: DatasetOptions,
        max_samples_per_scenario: int = 0,
        sample_reserve_fraction: float = 0.0,
        seed: int = 20260720,
        optimize_io_order: bool = False,
    ) -> None:
        root = Path(root).resolve()
        manifest_dir = root / "Module01_Intermediate" / "PythonDataset"
        self.base = QGLAIFPARDataset(
            root=root,
            token_index_csv=manifest_dir / "token_index_python.csv",
            mask_manifest_csv=manifest_dir / f"mask_manifest_{split}.csv",
            split=split,
            scenarios=scenarios,
            options=options,
        )
        denial_start_year = {
            "sensor_denial_MOD": int(sensor_start_years[0]),
            "sensor_denial_MYD": int(sensor_start_years[1]),
            "sensor_denial_VNP": int(sensor_start_years[2]),
        }
        manifest = self.base.mask_manifest
        for scenario, start_year in denial_start_year.items():
            invalid = (manifest["Scenario"].astype(str) == scenario) & (
                manifest["Year"].astype(int) < start_year
            )
            manifest = manifest.loc[~invalid]
        self.base.mask_manifest = manifest.reset_index(drop=True)

        self.target_samples_per_scenario = max(0, int(max_samples_per_scenario))
        if max_samples_per_scenario > 0:
            parts = []
            group_columns = ["Scenario"]
            if "Tile" in self.base.mask_manifest.columns:
                group_columns.insert(0, "Tile")
            for _key, group in self.base.mask_manifest.groupby(group_columns, sort=False):
                requested = int(max_samples_per_scenario)
                reserve = int(math.ceil(requested * max(float(sample_reserve_fraction), 0.0)))
                n = min(requested + reserve, len(group))
                parts.append(group.sample(n=n, random_state=int(seed)) if len(group) > n else group)
            self.base.mask_manifest = pd.concat(parts, ignore_index=True).reset_index(drop=True)
        sampling_groups = ["Scenario"]
        if "Tile" in self.base.mask_manifest.columns:
            sampling_groups.insert(0, "Tile")
        self.base.mask_manifest["_ScenarioSamplePosition"] = (
            self.base.mask_manifest.groupby(sampling_groups, sort=False).cumcount()
        )
        if optimize_io_order:
            # Preserve the sampled rows but cluster nearby token windows so the
            # compact-data and annual-LC HDF5 caches are reused across batches.
            order = [
                column for column in ("TokenRow", "Scenario", "Seed")
                if column in self.base.mask_manifest.columns
            ]
            self.base.mask_manifest = self.base.mask_manifest.sort_values(
                order, kind="stable"
            ).reset_index(drop=True)

        self.window_years = max(1, int(window_years))
        if self.window_years % 2 == 0:
            raise ValueError("window_years must be odd.")
        self.sensor_start_years = tuple(int(x) for x in sensor_start_years)
        self.lc_codes = tuple(int(x) for x in options.lc_codes) if options.use_lc_context else ()
        self.token_lookup: dict[tuple[str, int, int, int, int, int], int] = {}
        self.spatial_years: dict[tuple[str, int, int, int, int], list[int]] = {}
        for idx, row in self.base.token_index.iterrows():
            spatial = (
                str(row["Tile"]),
                int(row["BlockRow"]),
                int(row["BlockCol"]),
                int(row["GlobalRowStart"]),
                int(row["GlobalColStart"]),
            )
            year = int(row["Year"])
            self.token_lookup[spatial + (year,)] = int(idx)
            self.spatial_years.setdefault(spatial, []).append(year)
        for key in self.spatial_years:
            self.spatial_years[key] = sorted(set(self.spatial_years[key]))

    def __len__(self) -> int:
        return len(self.base)

    def close(self) -> None:
        self.base.close()

    def _window(self, years: list[int], center_year: int) -> tuple[list[int], int]:
        position = years.index(center_year)
        start = max(0, position - self.window_years // 2)
        start = min(start, max(len(years) - self.window_years, 0))
        selected = years[start : start + self.window_years]
        return selected, selected.index(center_year)

    def __getitem__(self, index: int) -> dict:
        center = self.base[index]
        mask_row = self.base.mask_manifest.iloc[index]
        token_idx = int(mask_row["TokenRow"]) - 1
        center_row = self.base.token_index.iloc[token_idx]
        spatial = (
            str(center_row["Tile"]),
            int(center_row["BlockRow"]),
            int(center_row["BlockCol"]),
            int(center_row["GlobalRowStart"]),
            int(center_row["GlobalColStart"]),
        )
        years, center_index = self._window(self.spatial_years[spatial], int(center_row["Year"]))
        inputs = []
        for year in years:
            row = self.base.token_index.iloc[self.token_lookup[spatial + (year,)]]
            token = self.base._read_token_patch(row)
            lc = None
            if self.base.options.use_lc_context:
                r0 = int(row["LocalRowStart"]) - 1
                c0 = int(row["LocalColStart"]) - 1
                patch = int(row["PatchSize"])
                lc = self.base._read_lc_type3_patch(row, r0, r0 + patch, c0, c0 + patch)
            inputs.append(build_input(token, year, self.sensor_start_years, lc, self.lc_codes,
                                      active_variable=self.base.options.active_variable))
        inputs[center_index] = center["x"].numpy()
        return {
            "x_window": torch.as_tensor(np.stack(inputs), dtype=torch.float32),
            "center_index": torch.tensor(center_index, dtype=torch.long),
            "target": center["y"],
            "target_mask": center["target_mask"],
            "target_weight": center["loss_weight"],
            "target_saturated": center["target_saturated"],
            "scenario": str(mask_row["Scenario"]),
            "tile": str(center_row["Tile"]),
            "token_row": int(mask_row["TokenRow"]),
            "scenario_position": torch.tensor(
                int(mask_row["_ScenarioSamplePosition"]), dtype=torch.long
            ),
            "year": torch.tensor(int(center_row["Year"]), dtype=torch.long),
            "block_row": torch.tensor(spatial[1], dtype=torch.long),
            "block_col": torch.tensor(spatial[2], dtype=torch.long),
        }
