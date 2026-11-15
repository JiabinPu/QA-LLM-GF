"""On-demand QA-aware LAI/FPAR token dataset for one selected tile."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from qa_llm_v3.quality import calibrated_quality
from qa_llm_v3.variable import isolate_token, variable_index


CHANNEL_NAMES = [
    "LAI",
    "FPAR",
    "M_hard_LAI",
    "M_hard_FPAR",
    "QA_LAI",
    "QA_FPAR",
    "M_anchor_LAI",
    "M_anchor_FPAR",
    "PrimaryAlgorithm",
    "BackupAlgorithm",
    "AtmosphereOK",
    "LandOK",
    "CloudOK",
    "SensorID",
    "DOY_sin",
    "DOY_cos",
    "LAI_StdDev_norm",
    "FPAR_StdDev_norm",
    "SaturationFlag",
]

TARGET_NAMES = ["LAI", "FPAR"]
DEFAULT_LC_CODES = tuple(range(1, 9))


@dataclass
class DatasetOptions:
    active_variable: Optional[str] = None
    channels_first: bool = True
    set_invalid_to_zero: bool = True
    return_meta: bool = True
    min_anchor_ratio: float = 0.0
    h5_cache_size: int = 6
    token_cache_size: int = 0
    use_lc_context: bool = False
    lc_mask_dir: Optional[str | Path] = None
    lc_codes: Sequence[int] = DEFAULT_LC_CODES


class QGLAIFPARDataset(Dataset):
    """Read compact HDF5 patches and generate masked targets on demand."""

    def __init__(
        self,
        root: str | Path,
        token_index_csv: str | Path,
        mask_manifest_csv: str | Path,
        split: Optional[str] = None,
        scenarios: Optional[Sequence[str]] = None,
        options: Optional[DatasetOptions] = None,
    ) -> None:
        self.root = Path(root).resolve()
        self.token_index_csv = Path(token_index_csv).resolve()
        self.mask_manifest_csv = Path(mask_manifest_csv).resolve()
        self.token_index = pd.read_csv(self.token_index_csv)
        self.mask_manifest = pd.read_csv(self.mask_manifest_csv)
        self.options = options or DatasetOptions()
        self.lc_codes = tuple(int(x) for x in self.options.lc_codes)
        self.channel_names = CHANNEL_NAMES + (
            [f"LC_Type_3_{code:02d}" for code in self.lc_codes]
            if self.options.use_lc_context
            else []
        )
        self.lc_mask_dir = (
            Path(self.options.lc_mask_dir).resolve()
            if self.options.lc_mask_dir is not None and str(self.options.lc_mask_dir).strip()
            else self.root / "Module00_Landcover" / "Masks"
        )

        if split is not None and "Split" in self.mask_manifest.columns:
            self.mask_manifest = self.mask_manifest[self.mask_manifest["Split"].astype(str) == split]

        if scenarios is not None:
            keep = set(map(str, scenarios))
            self.mask_manifest = self.mask_manifest[
                self.mask_manifest["Scenario"].astype(str).isin(keep)
            ]

        if self.options.min_anchor_ratio > 0 and {"TokenRow"}.issubset(self.mask_manifest.columns):
            token_rows = self.mask_manifest["TokenRow"].astype(int).to_numpy() - 1
            ok = (token_rows >= 0) & (token_rows < len(self.token_index))
            anchor_sum = np.zeros(len(self.mask_manifest), dtype=np.float32)
            valid_rows = token_rows[ok]
            anchor_sum[ok] = (
                self.token_index.iloc[valid_rows]["AnchorRatioLAI"].to_numpy(dtype=np.float32)
                + self.token_index.iloc[valid_rows]["AnchorRatioFPAR"].to_numpy(dtype=np.float32)
            )
            self.mask_manifest = self.mask_manifest[
                ok & (anchor_sum >= float(self.options.min_anchor_ratio))
            ]

        self.mask_manifest = self.mask_manifest.reset_index(drop=True)
        if len(self.mask_manifest) == 0:
            raise ValueError("No dataset rows remain after split/scenario/quality filtering.")

        self._h5_cache: OrderedDict[str, h5py.File] = OrderedDict()
        self._token_cache: OrderedDict[tuple[str, int, int, int, int], dict[str, np.ndarray]] = OrderedDict()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_h5_cache"] = OrderedDict()
        state["_token_cache"] = OrderedDict()
        return state

    def close(self) -> None:
        for f in self._h5_cache.values():
            try:
                f.close()
            except Exception:
                pass
        self._h5_cache.clear()
        self._token_cache.clear()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __len__(self) -> int:
        return len(self.mask_manifest)

    def __getitem__(self, idx: int) -> dict:
        mask_row = self.mask_manifest.iloc[idx]
        token_row_idx = int(mask_row["TokenRow"]) - 1
        if token_row_idx < 0 or token_row_idx >= len(self.token_index):
            raise IndexError(f"Invalid TokenRow: {token_row_idx + 1}")

        token_row = self.token_index.iloc[token_row_idx]
        token = self._read_token_patch(token_row)
        sample = self._apply_mask(token, mask_row)

        x = sample["x"]
        y = sample["y"]
        target_mask = sample["target_mask"]
        loss_weight = sample["loss_weight"]
        target_saturated = sample["target_saturated"]

        if self.options.channels_first:
            x = np.transpose(x, (4, 2, 0, 1, 3))
            y = np.transpose(y, (4, 2, 0, 1, 3))
            target_mask = np.transpose(target_mask, (4, 2, 0, 1, 3))
            loss_weight = np.transpose(loss_weight, (4, 2, 0, 1, 3))
            target_saturated = np.transpose(target_saturated, (4, 2, 0, 1, 3))

        out = {
            "x": torch.as_tensor(x, dtype=torch.float32),
            "y": torch.as_tensor(y, dtype=torch.float32),
            "target_mask": torch.as_tensor(target_mask, dtype=torch.bool),
            "loss_weight": torch.as_tensor(loss_weight, dtype=torch.float32),
            "target_saturated": torch.as_tensor(target_saturated, dtype=torch.bool),
        }
        if self.options.return_meta:
            out["meta"] = {
                "source_file": str(token_row["SourceFile"]),
                "scenario": str(mask_row["Scenario"]),
                "seed": int(mask_row["Seed"]),
                "year": int(mask_row["Year"]),
                "block_row": int(mask_row["BlockRow"]),
                "block_col": int(mask_row["BlockCol"]),
                "patch_row": int(mask_row["PatchRow"]),
                "patch_col": int(mask_row["PatchCol"]),
                "channel_names": self.channel_names,
                "target_names": TARGET_NAMES,
            }
        return out

    def _read_token_patch(self, row: pd.Series) -> dict[str, np.ndarray]:
        src = self._resolve_source_file(row)
        r0 = int(row["LocalRowStart"]) - 1
        c0 = int(row["LocalColStart"]) - 1
        patch = int(row["PatchSize"])
        r1 = r0 + patch
        c1 = c0 + patch
        key = (str(src), r0, r1, c0, c1)

        if self.options.token_cache_size > 0 and key in self._token_cache:
            self._token_cache.move_to_end(key)
            return self._token_cache[key]

        if self.options.h5_cache_size <= 0:
            with h5py.File(src, "r") as f:
                token = read_token_arrays(f, r0, r1, c0, c1)
        else:
            token = read_token_arrays(self._get_h5_file(src), r0, r1, c0, c1)
        if self.options.use_lc_context:
            token["lc_type3"] = self._read_lc_type3_patch(row, r0, r1, c0, c1)

        if self.options.token_cache_size > 0:
            self._token_cache[key] = token
            self._token_cache.move_to_end(key)
            while len(self._token_cache) > int(self.options.token_cache_size):
                self._token_cache.popitem(last=False)
        return token

    def _get_h5_file(self, src: Path) -> h5py.File:
        key = str(src)
        if key in self._h5_cache:
            self._h5_cache.move_to_end(key)
            return self._h5_cache[key]
        f = h5py.File(src, "r")
        self._h5_cache[key] = f
        self._h5_cache.move_to_end(key)
        while len(self._h5_cache) > int(self.options.h5_cache_size):
            _old, old_file = self._h5_cache.popitem(last=False)
            try:
                old_file.close()
            except Exception:
                pass
        return f

    def _resolve_source_file(self, row: pd.Series) -> Path:
        src = Path(str(row["SourceFile"]))
        if src.exists():
            return src
        candidate = (
            self.root
            / "Module01_Intermediate"
            / "CompactStacks"
            / str(row["Tile"])
            / f"{int(row['Year']):04d}"
            / src.name
        )
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Compact source not found: {src}\nTried: {candidate}")

    def _resolve_lc_file(self, row: pd.Series) -> Path:
        if "LCSourceFile" in row.index:
            value = str(row["LCSourceFile"]).strip()
            if value and value.lower() != "nan":
                source = Path(value)
                if source.exists():
                    return source
                raise FileNotFoundError(
                    f"Indexed land-cover file does not exist: {source}"
                )
        tile = str(row["Tile"])
        year = int(row["Year"])
        candidates = [
            self.lc_mask_dir / f"LC_Type_3_vegetation_mask_{tile}_{year:04d}.h5",
            self.lc_mask_dir / f"vegetation_mask_{tile}_{year:04d}.h5",
            self.lc_mask_dir / f"{tile}_{year:04d}_vegetation_mask.h5",
            self.lc_mask_dir / f"{year:04d}.h5",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        tried = "\n".join(str(x) for x in candidates)
        raise FileNotFoundError(f"Land-cover file for {tile} {year} was not found. Tried:\n{tried}")

    def _read_lc_type3_patch(self, row: pd.Series, r0: int, r1: int, c0: int, c1: int) -> np.ndarray:
        src = self._resolve_lc_file(row)
        inline = (
            "LCInline" in row.index
            and str(row["LCInline"]).strip().lower() not in {"", "0", "false", "nan"}
        )
        if inline:
            with h5py.File(src, "r") as f:
                if "/LC_Type_3" not in f:
                    raise KeyError(f"Missing inline /LC_Type_3 in station file: {src}")
                return read_lc_grid_patch(f["/LC_Type_3"], r0, r1, c0, c1, f)
        # Compact observations use block-local coordinates, whereas the annual
        # LC file covers the complete 2400 x 2400 MODIS tile.
        global_r0 = int(row["GlobalRowStart"]) - 1
        global_c0 = int(row["GlobalColStart"]) - 1
        patch_size = int(row["PatchSize"])
        with h5py.File(src, "r") as f:
            if "/LC_Type_3" not in f:
                raise KeyError(f"Missing /LC_Type_3 in land-cover file: {src}")
            return read_lc_grid_patch(
                f["/LC_Type_3"],
                global_r0,
                global_r0 + patch_size,
                global_c0,
                global_c0 + patch_size,
                f,
            )

    def _apply_mask(self, token: dict[str, np.ndarray], mask_row: pd.Series) -> dict[str, np.ndarray]:
        token = isolate_token(token, self.options.active_variable)
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

        h, w, t, s = lai_dn.shape
        m_hard_lai = bit(flag_lai, 1)
        m_anchor_lai = bit(flag_lai, 2)
        valid_lai = bit(flag_lai, 3)
        primary_lai = bit(flag_lai, 4)
        backup_lai = bit(flag_lai, 5)
        atmosphere_lai = bit(flag_lai, 6)
        land_lai = bit(flag_lai, 7)
        cloud_lai = bit(flag_lai, 8)

        m_hard_fpar = bit(flag_fpar, 1)
        m_anchor_fpar = bit(flag_fpar, 2)
        valid_fpar = bit(flag_fpar, 3)
        primary_fpar = bit(flag_fpar, 4)
        backup_fpar = bit(flag_fpar, 5)
        atmosphere_fpar = bit(flag_fpar, 6)
        land_fpar = bit(flag_fpar, 7)
        cloud_fpar = bit(flag_fpar, 8)

        lai = lai_dn.astype(np.float32) * 0.1
        fpar = fpar_dn.astype(np.float32) * 0.01
        valid_lai_input = m_hard_lai & valid_lai & (lai_dn != 255)
        valid_fpar_input = m_hard_fpar & valid_fpar & (fpar_dn != 255)
        if self.options.set_invalid_to_zero:
            lai[~valid_lai_input] = 0.0
            fpar[~valid_fpar_input] = 0.0
        else:
            lai[~valid_lai_input] = np.nan
            fpar[~valid_fpar_input] = np.nan

        score_version = token.get("quality_score_version", "")
        qa_lai, lai_std_norm, m_anchor_lai = calibrated_quality(
            score_lai, lai, lai_std_dn, valid_lai_input, scf_qc, "LAI", score_version
        )
        qa_fpar, fpar_std_norm, m_anchor_fpar = calibrated_quality(
            score_fpar, fpar, fpar_std_dn, valid_fpar_input, scf_qc, "FPAR", score_version
        )

        sensor_id = np.zeros((1, 1, 1, s), dtype=np.float32)
        if s > 1:
            sensor_id[0, 0, 0, :] = np.arange(s, dtype=np.float32) / float(s - 1)
        sensor_id = np.broadcast_to(sensor_id, (h, w, t, s))

        if len(doy) < t:
            doy = np.pad(doy, (0, t - len(doy)), constant_values=np.nan)
        doy = doy[:t].reshape(1, 1, t, 1)
        doy_sin = np.broadcast_to(np.sin(2 * np.pi * (doy - 1) / 365.0).astype(np.float32), (h, w, t, s))
        doy_cos = np.broadcast_to(np.cos(2 * np.pi * (doy - 1) / 365.0).astype(np.float32), (h, w, t, s))

        primary = primary_lai | primary_fpar
        backup = backup_lai | backup_fpar
        atmosphere = atmosphere_lai | atmosphere_fpar
        land = land_lai | land_fpar
        cloud = cloud_lai | cloud_fpar

        x = np.stack(
            [
                lai,
                fpar,
                m_hard_lai.astype(np.float32),
                m_hard_fpar.astype(np.float32),
                qa_lai,
                qa_fpar,
                m_anchor_lai.astype(np.float32),
                m_anchor_fpar.astype(np.float32),
                primary.astype(np.float32),
                backup.astype(np.float32),
                atmosphere.astype(np.float32),
                land.astype(np.float32),
                cloud.astype(np.float32),
                sensor_id,
                doy_sin,
                doy_cos,
                lai_std_norm.astype(np.float32),
                fpar_std_norm.astype(np.float32),
                (scf_qc == 1).astype(np.float32),
            ],
            axis=-1,
        )
        vegetation_domain = None
        if self.options.use_lc_context:
            if "lc_type3" not in token:
                raise KeyError("LC context was requested but lc_type3 was not loaded.")
            lc = np.asarray(token["lc_type3"], dtype=np.uint8)
            vegetation_domain = np.isin(lc, np.asarray(self.lc_codes, dtype=np.uint8)).reshape(h, w, 1, 1)
            lc_channels = [
                np.broadcast_to((lc == int(code)).astype(np.float32).reshape(h, w, 1, 1), (h, w, t, s))
                for code in self.lc_codes
            ]
            if lc_channels:
                x = np.concatenate([x, np.stack(lc_channels, axis=-1)], axis=-1)

        scenario = str(mask_row["Scenario"])
        rng = np.random.default_rng(int(mask_row["Seed"]))
        mask_lai, mask_fpar = make_eval_masks(
            m_anchor_lai,
            m_anchor_fpar,
            scenario,
            rng,
            lai=lai,
            fpar=fpar,
            hard_lai=valid_lai_input,
            hard_fpar=valid_fpar_input,
        )
        if vegetation_domain is not None:
            mask_lai &= vegetation_domain
            mask_fpar &= vegetation_domain

        for ch in [0, 2, 4, 6, 16]:
            x[..., ch][mask_lai] = 0.0
        for ch in [1, 3, 5, 7, 17]:
            x[..., ch][mask_fpar] = 0.0
        # A complete retrieval gap cannot retain QA semantics from the hidden
        # observation. Sensor identity, DOY, and LC remain valid external context.
        shared_gap = mask_lai & mask_fpar
        if self.options.active_variable is not None:
            shared_gap = (mask_lai, mask_fpar)[variable_index(self.options.active_variable)]
        for ch in [8, 9, 10, 11, 12, 18]:
            x[..., ch][shared_gap] = 0.0

        target = np.stack([lai, fpar], axis=-1).astype(np.float32)
        target_mask = np.stack([mask_lai, mask_fpar], axis=-1)
        target_saturated = np.stack(
            [(scf_qc == 1) & mask_lai, (scf_qc == 1) & mask_fpar], axis=-1
        )
        loss_weight = np.stack([qa_lai * mask_lai, qa_fpar * mask_fpar], axis=-1).astype(np.float32)
        return {
            "x": x.astype(np.float32),
            "y": target,
            "target_mask": target_mask,
            "loss_weight": loss_weight,
            "target_saturated": target_saturated,
        }


def read_matlab_h5_patch(dataset: h5py.Dataset, r0: int, r1: int, c0: int, c1: int) -> np.ndarray:
    arr = dataset[:, :, c0:c1, r0:r1]
    return np.ascontiguousarray(np.transpose(arr, (3, 2, 1, 0)))


def read_lc_grid_patch(dataset: h5py.Dataset, r0: int, r1: int, c0: int, c1: int, root_h5: h5py.File) -> np.ndarray:
    # LC masks are written by MATLAB. HDF5 reverses the two logical MATLAB
    # dimensions as seen by h5py, so legacy files without an explicit h5py
    # attribute must default to physical col,row storage.
    order = str(dataset.attrs.get(
        "python_h5py_dimension_order",
        root_h5.attrs.get("python_h5py_dimension_order", "col,row"),
    )).lower().replace(" ", "")
    if order.startswith("col,row"):
        arr = dataset[c0:c1, r0:r1]
        return np.ascontiguousarray(arr.T)
    arr = dataset[r0:r1, c0:c1]
    return np.ascontiguousarray(arr)


def read_optional_patch(f: h5py.File, path: str, r0: int, r1: int, c0: int, c1: int, ref) -> np.ndarray:
    if path not in f:
        return np.full_like(ref, 255, dtype=np.uint8)
    return read_matlab_h5_patch(f[path], r0, r1, c0, c1)


def read_token_arrays(f: h5py.File, r0: int, r1: int, c0: int, c1: int) -> dict[str, np.ndarray]:
    if "/qa/SCF_QC" not in f:
        raise RuntimeError(
            f"Missing /qa/SCF_QC in {f.filename}. "
            "Rerun Module01_Build_Intermediate_Data.m for the frozen QA protocol."
        )
    lai_dn = read_matlab_h5_patch(f["/obs/LAI_DN"], r0, r1, c0, c1)
    fpar_dn = read_matlab_h5_patch(f["/obs/FPAR_DN"], r0, r1, c0, c1)
    score_lai = read_matlab_h5_patch(f["/score/LAI"], r0, r1, c0, c1)
    score_fpar = read_matlab_h5_patch(f["/score/FPAR"], r0, r1, c0, c1)
    flag_lai = read_matlab_h5_patch(f["/flag/LAI"], r0, r1, c0, c1)
    flag_fpar = read_matlab_h5_patch(f["/flag/FPAR"], r0, r1, c0, c1)
    lai_std_dn = read_optional_patch(f, "/uncert/LAI_STD_DN", r0, r1, c0, c1, lai_dn)
    fpar_std_dn = read_optional_patch(f, "/uncert/FPAR_STD_DN", r0, r1, c0, c1, fpar_dn)
    scf_qc = read_matlab_h5_patch(f["/qa/SCF_QC"], r0, r1, c0, c1)
    doy = np.asarray(f.attrs.get("DOY", np.arange(1, 362, 8))).astype(np.float32)
    return {
        "lai_dn": lai_dn,
        "fpar_dn": fpar_dn,
        "score_lai": score_lai,
        "score_fpar": score_fpar,
        "flag_lai": flag_lai,
        "flag_fpar": flag_fpar,
        "lai_std_dn": lai_std_dn,
        "fpar_std_dn": fpar_std_dn,
        "scf_qc": scf_qc,
        "doy": doy,
        "quality_score_version": f.attrs.get("quality_score_version", ""),
    }


def bit(arr: np.ndarray, bit_index_1based: int) -> np.ndarray:
    return (arr.astype(np.uint8) & np.uint8(1 << (bit_index_1based - 1))) > 0


def make_eval_masks(
    truth_lai: np.ndarray,
    truth_fpar: np.ndarray,
    scenario: str,
    rng: np.random.Generator,
    lai: np.ndarray | None = None,
    fpar: np.ndarray | None = None,
    hard_lai: np.ndarray | None = None,
    hard_fpar: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    mask_lai = np.zeros_like(truth_lai, dtype=bool)
    mask_fpar = np.zeros_like(truth_fpar, dtype=bool)

    if scenario == "high_value_holdout":
        if lai is None or fpar is None or hard_lai is None or hard_fpar is None:
            raise ValueError("high_value_holdout requires values and hard-QA masks.")
        # Hide all hard-QA-qualified high values, including anchors and usable
        # non-anchor observations. Lower bounds define the audit domain;
        # reporting code further separates progressively more extreme values.
        mask_lai = hard_lai & np.isfinite(lai) & (lai >= 2.0)
        mask_fpar = hard_fpar & np.isfinite(fpar) & (fpar >= 0.5)
    elif scenario.startswith("random"):
        ratio = 0.10 if scenario.endswith("10") else 0.30 if scenario.endswith("30") else 0.20
        mask_lai = random_mask(truth_lai, ratio, rng)
        mask_fpar = random_mask(truth_fpar, ratio, rng)
    elif scenario.startswith("temporal_block"):
        length = int(scenario.split("_")[-1])
        t = truth_lai.shape[2]
        start = 0 if t <= length else int(rng.integers(0, t - length + 1))
        sl = slice(start, min(start + length, t))
        mask_lai[:, :, sl, :] = truth_lai[:, :, sl, :]
        mask_fpar[:, :, sl, :] = truth_fpar[:, :, sl, :]
    elif scenario == "spatial_block":
        h, w = truth_lai.shape[:2]
        block = max(4, round(min(h, w) / 3))
        r0 = int(rng.integers(0, h - block + 1))
        c0 = int(rng.integers(0, w - block + 1))
        rs = slice(r0, r0 + block)
        cs = slice(c0, c0 + block)
        mask_lai[rs, cs, :, :] = truth_lai[rs, cs, :, :]
        mask_fpar[rs, cs, :, :] = truth_fpar[rs, cs, :, :]
    elif scenario.startswith("sensor_denial"):
        sensor = {"MOD": 0, "MYD": 1, "VNP": 2}
        key = scenario.split("_")[-1]
        ss = sensor.get(key, 0)
        if ss < truth_lai.shape[3]:
            mask_lai[:, :, :, ss] = truth_lai[:, :, :, ss]
            mask_fpar[:, :, :, ss] = truth_fpar[:, :, :, ss]
    else:
        raise ValueError(f"Unknown scenario: {scenario}")
    return mask_lai, mask_fpar


def random_mask(truth: np.ndarray, ratio: float, rng: np.random.Generator) -> np.ndarray:
    flat_truth = np.asarray(truth, dtype=bool).reshape(-1)
    out = np.zeros(flat_truth.shape, dtype=bool)
    idx = np.flatnonzero(flat_truth)
    if idx.size == 0:
        return out.reshape(truth.shape)
    n = max(1, int(round(idx.size * float(ratio))))
    chosen = rng.choice(idx, size=min(n, idx.size), replace=False)
    out[chosen] = True
    return out.reshape(truth.shape)
