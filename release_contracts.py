"""Public preflight contracts; no model inference, fitting, or asset changes."""
from pathlib import Path

from release_common import ROOT, confined_path, load_json, verify_manifest


FIELDS = ("obs/LAI_DN", "obs/FPAR_DN", "score/LAI", "score/FPAR",
          "flag/LAI", "flag/FPAR", "qa/SCF_QC", "uncert/LAI_STD_DN", "uncert/FPAR_STD_DN")


def check_annual_file(path, size, year):
    import h5py
    import numpy as np
    from qa_llm_v3.quality import QUALITY_SCORE_VERSION

    with h5py.File(path, "r") as handle:
        for name in FIELDS:
            if name not in handle or handle[name].shape != (3, 46, size, size) or handle[name].dtype != np.uint8:
                raise ValueError(f"Invalid observation schema: {path.name}/{name}")
        if "LC_Type_3" not in handle or handle["LC_Type_3"].shape != (size, size) or handle["LC_Type_3"].dtype != np.uint8:
            raise ValueError(f"Invalid annual landcover: {path.name}")
        if not np.array_equal(handle.attrs.get("DOY"), np.arange(1, 362, 8)):
            raise ValueError(f"Wrong 46 composite dates: {path.name}")
        if handle.attrs.get("year") != year or handle.attrs.get("quality_score_version") != QUALITY_SCORE_VERSION:
            raise ValueError(f"Year or QA protocol mismatch: {path.name}")
        if handle.attrs.get("observation_h5py_order") != "sensor,time,col,row":
            raise ValueError(f"Undeclared observation orientation: {path.name}")
        if handle.attrs.get("python_h5py_dimension_order") != "col,row":
            raise ValueError(f"Undeclared LC orientation: {path.name}")


def training_contract(folder):
    import numpy as np
    import pandas as pd

    folder = Path(folder).resolve()
    manifest = verify_manifest(folder)
    registry = load_json(ROOT / "core" / "tile_registry.json")
    if manifest.get("role") != "training_tutorial" or not set(manifest["tiles"]) <= set(registry["training"]):
        raise ValueError("Tutorial training must contain only declared training tiles")
    index = pd.read_csv(folder / "token_index.csv")
    required = {"TokenRow", "Tile", "Split", "Year", "GlobalRowStart", "GlobalColStart", "PatchSize", "SourceFile", "LCSourceFile"}
    if index.empty or not required <= set(index.columns):
        raise ValueError("Incomplete tutorial token index")
    if not np.array_equal(index.TokenRow.to_numpy(), np.arange(1, len(index) + 1)):
        raise ValueError("TokenRow must be contiguous and one-based")
    if set(index.Split) != {"train", "val", "test"} or set(index.Tile) != set(manifest["tiles"]):
        raise ValueError("Tutorial tile/split inventory mismatch")
    keys = ["Tile", "GlobalRowStart", "GlobalColStart"]
    for _, group in index.groupby(keys):
        if group.Split.nunique() != 1 or len(group) != len(manifest["years"]) or set(group.Year) != set(manifest["years"]):
            raise ValueError("Spatial split leakage, duplicate years, or missing context")
    locations = index.drop_duplicates(keys)
    # Check actual patch extents, not just equality of their origins.
    for tile, group in locations.groupby("Tile"):
        rows = list(group.itertuples())
        for i, left in enumerate(rows):
            for right in rows[i + 1:]:
                overlap = (abs(left.GlobalRowStart - right.GlobalRowStart) < left.PatchSize
                           and abs(left.GlobalColStart - right.GlobalColStart) < left.PatchSize)
                if overlap and left.Split != right.Split:
                    raise ValueError(f"Overlapping train/validation/test patches in {tile}")
    for row in index.itertuples():
        if row.PatchSize != 40:
            raise ValueError("Tutorial expects native 40x40 patches")
        for column in ("SourceFile", "LCSourceFile"):
            name = getattr(row, column)
            if name not in manifest["files"]:
                raise ValueError(f"Token source is absent from asset manifest: {name}")
            confined_path(folder, name)
        check_annual_file(folder / row.SourceFile, 40, int(row.Year))
    masks = {}
    for split in ("train", "val", "test"):
        table = pd.read_csv(folder / f"mask_manifest_{split}.csv")
        ids = pd.to_numeric(table.TokenRow, errors="raise").to_numpy()
        if not len(ids) or not np.isfinite(ids).all() or (ids != np.floor(ids)).any() or (ids < 1).any() or (ids > len(index)).any():
            raise ValueError(f"Out-of-range/nonintegral mask TokenRow: {split}")
        selected = index.iloc[ids.astype(int) - 1]
        if set(selected.Split) != {split} or not set(selected.Year) <= set(manifest["center_years"]):
            raise ValueError(f"Wrong masking split or center year: {split}")
        if table.duplicated(["TokenRow", "Scenario"]).any():
            raise ValueError(f"Duplicate masking cases: {split}")
        masks[split] = table
    return manifest, index, masks


def production_contract(folder):
    folder = Path(folder).resolve()
    manifest = verify_manifest(folder)
    if manifest.get("role") != "production_region" or manifest.get("years") != list(range(2001, 2025)):
        raise ValueError("Expected the complete 2001-2024 production example")
    if (manifest.get("height"), manifest.get("width"), manifest.get("native_patch_size"), manifest.get("core_size")) != (200, 200, 40, 8):
        raise ValueError("Expected a native, aligned 200x200 region, 40x40 patches and 8x8 cores")
    if any((manifest[key] - 1) % 40 for key in ("origin_row_1based", "origin_col_1based")):
        raise ValueError("Crop is not aligned with the original scientific patch grid")
    for year in manifest["years"]:
        check_annual_file(folder / f"{year}.h5", 200, year)
    return manifest


def model_contract(variable):
    from qa_llm_v3.quality import validate_checkpoint_protocol
    from qa_llm_v3.product_calibration import ProductCalibration
    from safetensors import safe_open

    folder = ROOT / "models" / variable
    verify_manifest(folder)
    verify_manifest(ROOT / "assets" / "gpt2_first_block")
    meta = load_json(folder / "model.json")
    if meta.get("role") != "frozen_scientific_model" or meta.get("variable") != variable or meta["model_config"].get("active_variable") != variable:
        raise ValueError(f"Wrong scientific model identity: {variable}")
    if meta.get("full_state") is not True or meta.get("optimizer_included") is not False:
        raise ValueError("Production requires full exported state without optimizer state")
    validate_checkpoint_protocol(meta, budget=True)
    settings = meta["budget_calibration_args"]
    levels = meta["budget_policy"]["levels"]
    if levels != sorted(set(levels)) or levels[0] != 0 or levels[-1] != meta["solver_config"]["max_iterations"]:
        raise ValueError("Budget levels and solver are inconsistent")
    if settings["core_size"] != 8 or settings["window_years"] != 5 or not 0 <= settings["policy_max_selected_fraction"] <= 1:
        raise ValueError("Unexpected production window/core/routing settings")
    with safe_open(str(folder / "model.safetensors"), framework="pt", device="cpu") as state:
        if "backbone.wte.weight" not in state.keys() or not any(k.startswith("backbone.h.0.") for k in state.keys()):
            raise ValueError("Frozen backbone tensors are missing")
    ProductCalibration(folder, variable, folder / "model.safetensors")
    return meta
