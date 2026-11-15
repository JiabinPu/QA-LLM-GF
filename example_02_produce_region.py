"""Frozen-model production on an aligned 200 x 200, 2001-2024 example region."""
import argparse
from pathlib import Path

from release_common import ROOT, write_json, load_frozen_model, load_calibration, output_directory
from release_contracts import production_contract, model_contract


def run_patch(model, solver, meta, calibration, handles, years, center, row, col, variable, device):
    import numpy as np
    import torch
    from qg_dataset import read_token_arrays, read_lc_grid_patch
    from qa_llm_v3.data import build_input
    from qa_llm_v3.pipeline import run_window
    from qa_llm_v3.product import compose_layers, year_window
    from qa_llm_v3.utils import parse_ints
    settings = meta["budget_calibration_args"]
    window, center_index = year_window(years, center, int(settings["window_years"]))
    inputs, cover = [], []
    for year in window:
        handle = handles[year]
        token = read_token_arrays(handle, row, row+40, col, col+40)
        lc = read_lc_grid_patch(handle["LC_Type_3"], row, row+40, col, col+40, handle)
        cover.append(lc)
        inputs.append(build_input(token, year, parse_ints(settings["sensor_start_years"]),
            lc, range(1, 9), active_variable=variable))
    pft = cover[center_index]
    vegetation = np.isin(pft, range(1, 9))
    if not vegetation.any():
        return np.full((46, 40, 40, 3), 65535, np.uint16)
    from portable_runtime import transfer_preserving_layout
    from qa_llm_v3.uncertainty_core import center_landcover
    x = transfer_preserving_layout(torch.as_tensor(np.stack(inputs)[None], dtype=torch.float32), device)
    mask = torch.as_tensor(vegetation[None], device=device)
    with torch.inference_mode():
        result = run_window(model, x, center_index, solver, core_size=int(settings["core_size"]),
            training_solver=False, route_mode="learned_budget",
            route_max_selected_fraction=float(settings["policy_max_selected_fraction"]),
            budget_levels=meta["budget_policy"]["levels"], domain_mask=mask,
            structural_cell_blend=True)
        layers = compose_layers(result, x[:, center_index], mask, int(settings["core_size"]),
            variable=variable, product_calibration=calibration,
            pft=center_landcover(x, torch.full((1,), center_index, device=device), parse_ints(settings["lc_codes"]))[0])
    return layers[0]


def initialize_part(path, size):
    import h5py
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.attrs["complete"] = 0
        handle.create_dataset("layers", shape=(46, size, size, 3), dtype="uint16", fillvalue=65535,
            chunks=(1, 40, 40, 3), compression="gzip", compression_opts=1, shuffle=True)


def merge_year(output, year, size, manifest, smoke):
    import h5py
    names = (("Lai", "Lai_Uncertainty", "Lai_QA"), ("Fpar", "Fpar_Uncertainty", "Fpar_QA"))
    import numpy as np
    sources = [output / "Parts" / v / f"{year}.h5" for v in ("LAI", "FPAR")]
    if not all(p.is_file() for p in sources):
        return 0
    target = output / str(year)
    target.mkdir(parents=True, exist_ok=True)
    with h5py.File(sources[0], "r") as lai, h5py.File(sources[1], "r") as fpar:
        for handle in (lai, fpar):
            if handle.attrs.get("complete") != 1 or handle["layers"].shape != (46, size, size, 3):
                return 0
        for ti, doy in enumerate(range(1, 362, 8)):
            path = target / f"QA_LLM_GF_{manifest['tile']}_{year}{doy:03d}.h5"
            with h5py.File(path, "w") as handle:
                for source, datasets in zip((lai, fpar), names):
                    array = source["layers"][ti]
                    for layer, name in enumerate(datasets):
                        # Reverse row/column storage so MATLAB h5read is native row,col.
                        d = handle.create_dataset(name, data=array[..., layer].T, fillvalue=np.uint16(65535),
                            compression="gzip", compression_opts=1, shuffle=True)
                        d.attrs.update(fill_value=65535, matlab_dimension_order="row,col",
                                       python_h5py_dimension_order="col,row")
                        if not name.endswith("_QA"):
                            d.attrs["scale_factor"] = 0.001
                        else:
                            d.attrs["bit_layout"] = "0:2 source;3:4 usable;5:6 strict;7:8 saturated;9:15 confidence"
                handle.attrs.update(complete=1, tile=manifest["tile"], year=year, doy=doy,
                    rows=size, cols=size, example_smoke=bool(smoke),
                    origin_row_1based=manifest["origin_row_1based"],
                    origin_col_1based=manifest["origin_col_1based"],
                    matlab_dimension_order="row,col", python_h5py_dimension_order="col,row",
                    physical_scale_factor=0.001, physical_fill_value=65535,
                    uncertainty_application_count=1,
                    uncertainty_quantity="calibrated sigma; not a confidence-interval half width")
                handle.flush()
    return 46


def main():
    from production_cli import parser, configure
    from portable_runtime import produce as produce_paired
    p = parser("region")
    p.add_argument("--variable", choices=("LAI", "FPAR", "both"), default="both")
    a = p.parse_args()
    try:
        config = configure(a, "region")
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        p.error(str(exc))
    if a.check_only:
        return
    if a.variable == "both":
        produce_paired(config)
    else:
        if a.device == "cpu_gpu" or a.replay_map:
            p.error("Mixed/replayed production records paired LAI+FPAR; use --variable both")
        # Retain the original single-variable teaching interface. It uses a
        # single resolved device and separate Parts, not the paired task queue.
        produce(a, Path(config["output"]), Path(config["source"]), config["source_identity"], config["years"], config["height"])


def produce(a, output, data, manifest, years, size):
    import h5py
    import torch
    from qa_llm_v3.utils import resolve_device
    handles = {}
    try:
        torch.set_num_threads(a.threads)
        device = resolve_device(a.device)
        for year in manifest["years"]:
            handles[year] = h5py.File(data / f"{year}.h5", "r")
        for variable in (("LAI", "FPAR") if a.variable == "both" else (a.variable,)):
            model, solver, meta = load_frozen_model(variable, device)
            calibration = load_calibration(variable)
            for year in years:
                path = output / "Parts" / variable / f"{year}.h5"
                initialize_part(path, size)
                with h5py.File(path, "r+") as part:
                    for row in range(0, size, 40):
                        for col in range(0, size, 40):
                            values = run_patch(model, solver, meta, calibration, handles,
                                manifest["years"], year, row, col, variable, device)
                            part["layers"][:, row:row+40, col:col+40] = values
                            count = (row//40)*(size//40) + col//40 + 1
                            print(f"{variable}/{year}: {count}/{(size//40)**2} patches", flush=True)
                    part.attrs["complete"] = 1
            del model
        counts = {str(year): merge_year(output, year, size, manifest, a.smoke) for year in years}
        write_json(output / "product_summary.json", dict(requested_years=years, variable=a.variable,
            final_file_counts=counts, size=size, smoke=a.smoke,
            final_complete=all(n == 46 for n in counts.values())))
        print(f"Finished: {output}; final yearly file counts: {counts}")
    finally:
        for handle in handles.values():
            handle.close()


if __name__ == "__main__":
    main()
