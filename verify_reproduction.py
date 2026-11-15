"""Compare six uint16 layers to the frozen Mac-runtime small-sample references.

This is not a comparison to the deleted full-tile products or old Windows data.
CPU and MPS each have their own reference. Nothing trains or runs a whole tile.
"""
import argparse
import hashlib
from pathlib import Path

from release_common import ROOT, load_json, sha256, write_json, output_directory


def verify(device, output=None):
    import numpy as np
    import torch
    from qa_llm_v3.utils import resolve_device
    from portable_runtime import InputPreparer, ModelRuntime, LAYERS, environment
    folder = ROOT / "data" / "mac_reference"
    manifest = load_json(folder / "manifest.json")
    backend = resolve_device(device).type
    if backend not in manifest["profiles"]:
        raise RuntimeError(f"No measured {backend} reference is included; CUDA requires verification on CUDA hardware")
    profile = manifest["profiles"][backend]
    path = folder / profile["file"]
    if sha256(path) != profile["sha256"]:
        raise RuntimeError("The reference fixture checksum differs")
    for name, expected in manifest["model_hashes"].items():
        if sha256(ROOT / name) != expected:
            raise RuntimeError(f"Reference model/calibration asset differs: {name}")
    from release_contracts import production_contract
    production_contract(ROOT / "data" / "production_region")
    runtime = ModelRuntime(backend, 1)
    preparer = InputPreparer(ROOT / "data" / "production_region", "region", runtime.settings)
    rows = []
    try:
        with np.load(path, allow_pickle=False) as reference:
            for i, case in enumerate(manifest["cases"]):
                prepared = preparer.prepare(case["year"], case["row"], case["col"])
                values, stats = runtime.infer(prepared)
                expected = reference[f"case_{i}"]
                if values.shape != expected.shape or values.dtype != expected.dtype:
                    raise RuntimeError("Reference/output shape or dtype differs")
                differences = {}
                for li, name in enumerate(LAYERS):
                    a, b = values[..., li], expected[..., li]
                    delta = np.abs(a.astype(np.int32) - b.astype(np.int32))
                    differences[name] = dict(different_values=int(np.count_nonzero(a != b)), max_dn_difference=int(delta.max()))
                diagnostic_match = stats["solver_core_diagnostics"] == profile["solver_diagnostics"][i]
                row = dict(**case, layers=differences, actual_solver_diagnostics_equal=diagnostic_match,
                           all_six_layers_exact=bool(np.array_equal(values, expected)))
                rows.append(row)
                print(f"{backend}: case {i + 1}/{len(manifest['cases'])}, six-layer DN exact={row['all_six_layers_exact']}, solver counters exact={diagnostic_match}", flush=True)
    finally:
        preparer.close()
    report = dict(ok=all(r["all_six_layers_exact"] and r["actual_solver_diagnostics_equal"] for r in rows),
                  reference_scope=manifest["reference_scope"], backend=backend, cases=rows,
                  environment=environment(), reference_environment=profile["environment"],
                  reference_fixture_sha256=profile["sha256"])
    target = output_directory(output or ROOT / "runs" / f"reproduction_{backend}.json")
    write_json(target, report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    result = verify(a.device, a.output)
    if not result["ok"]:
        raise SystemExit("Reference comparison failed; see the per-layer report")


if __name__ == "__main__":
    main()
