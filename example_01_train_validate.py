"""Small-data demonstration of the ORIGINAL separate-variable learning chain."""
import argparse
from pathlib import Path
import subprocess
import sys
import shutil

from release_common import ROOT, load_json, write_json, verify_manifest, output_directory, execute_job
from release_contracts import training_contract


def prepare_run(data, output, *, validated=None):
    import pandas as pd
    manifest, index, masks = validated or training_contract(data)
    target = output / "Module01_Intermediate" / "PythonDataset"
    index = index.copy()
    for column in ("SourceFile", "LCSourceFile"):
        index[column] = index[column].map(lambda value: str((data / value).resolve()))
    target.mkdir(parents=True, exist_ok=True)
    # Disjoint spatial groups remain disjoint across all neighboring years.
    keys = ["Tile", "BlockRow", "BlockCol", "GlobalRowStart", "GlobalColStart"]
    if index.groupby(keys).Split.nunique().max() != 1:
        raise ValueError("Spatial train/validation/test leakage")
    index.to_csv(target / "token_index_python.csv", index=False)
    lc_summary = output / "Module00_Landcover" / "Summaries" / "landcover_mask_summary.csv"
    lc_summary.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(data / "landcover_mask_summary.csv", lc_summary)
    for split in ("train", "val", "test"):
        table = masks[split]
        rows = index.iloc[table.TokenRow.astype(int).to_numpy() - 1]
        if set(rows.Split) != {split}:
            raise ValueError(f"Wrong spatial split: {split}")
        table.to_csv(target / f"mask_manifest_{split}.csv", index=False)
    for variable in ("LAI", "FPAR"):
        write_json(output / "Module02A_TrainingInputs" / variable / "input_contract.json",
                   dict(variable=variable, tiles=manifest["tiles"], role="tutorial_only"))
    write_json(output / "tutorial_input.json", dict(role="tutorial_not_paper_reproduction",
               tiles=manifest["tiles"]))


def command(module, options):
    result = [sys.executable, "-u", "-m", module]
    for key, value in options.items():
        if value is not None:
            result += ["--" + key.replace("_", "-"), str(value)]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variable", choices=("LAI", "FPAR", "both"), default="both")
    parser.add_argument("--stage", choices=("prepare", "train", "budget", "evaluate", "uncertainty", "all"), default="prepare")
    parser.add_argument("--output", type=Path, default=ROOT / "runs" / "tutorial")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--budget-epochs", type=int, default=3)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="One epoch and one batch for training; minimal budget sample; separate output")
    args = parser.parse_args()
    if min(args.epochs, args.budget_epochs, args.threads) < 1:
        parser.error("Epochs and thread count must be positive")
    if args.smoke and args.output == ROOT / "runs" / "tutorial":
        args.output = ROOT / "runs" / "tutorial_smoke"
    try:
        args.output = output_directory(args.output)
    except ValueError as exc:
        parser.error(str(exc))
    data = ROOT / "data" / "training_tutorial"
    verify_manifest(ROOT / "assets" / "gpt2_first_block")
    validated = training_contract(data)
    if args.check_only:
        print(f"Validated tutorial assets; no output written: {args.output}")
    else:
        prepare_run(data, args.output, validated=validated)
    if args.stage == "prepare":
        print(f"Tutorial inputs {'checked' if args.check_only else 'prepared'}: {args.output}")
        return
    stages = ("train", "budget", "evaluate", "uncertainty") if args.stage == "all" else (args.stage,)
    for variable in (("LAI", "FPAR") if args.variable == "both" else (args.variable,)):
        for stage in stages:
            if not args.check_only:
                required = {
                    "budget": [args.output / "Module02B_Training" / variable / "best.pt"],
                    "evaluate": [args.output / "Module02C_BudgetCalibration" / variable / "best_budget.pt"],
                    "uncertainty": [args.output / "Module03A_Evaluation" / variable / split / "evaluation_manifest.json" for split in ("val", "test")],
                }.get(stage, [])
                for path in required:
                    if not path.is_file():
                        raise FileNotFoundError(f"{variable}/{stage} requires an earlier stage: {path}")
            if stage == "train":
                options = load_json(ROOT / "config" / "training_settings.json")
                options.update(root=args.output, variable=variable, epochs=args.epochs,
                    phenology_calibration_epochs=args.epochs, device=args.device,
                    torch_threads=args.threads, print_every=1,
                    gpt_model=ROOT / "assets" / "gpt2_first_block")
                if args.smoke:
                    options.update(epochs=1, phenology_calibration_epochs=1,
                                   max_train_batches=1, max_val_batches=1)
                jobs = [command("qa_llm_v3.train", options)]
            elif stage == "budget":
                options = load_json(ROOT / "config" / "budget_settings.json")
                options.update(root=args.output, variable=variable, epochs=args.budget_epochs,
                    base_checkpoint=args.output / "Module02B_Training" / variable / "best.pt",
                    output_dir=args.output / "Module02C_BudgetCalibration" / variable,
                    lc_mask_dir=args.output / "Module00_Landcover" / "Masks",
                    device=args.device, torch_threads=args.threads, print_every=1,
                    overwrite_oracle_cache=1)
                if args.smoke:
                    options.update(epochs=1, train_samples_per_scenario=1, val_samples_per_scenario=1)
                jobs = [command("qa_llm_v3.calibrate_budget", options)]
            elif stage == "evaluate":
                jobs = [command("qa_llm_v3.evaluation", dict(root=args.output, variable=variable,
                    checkpoint=args.output / "Module02C_BudgetCalibration" / variable / "best_budget.pt",
                    output_dir=args.output / "Module03A_Evaluation" / variable / split,
                    split=split, device=args.device, torch_threads=args.threads,
                    batch_size=1, print_every=1)) for split in ("val", "test")]
            else:
                jobs = [command("qa_llm_v3.uncertainty", dict(root=args.output, variable=variable))]
            for job in jobs:
                print(subprocess.list2cmdline(job) if args.check_only else f"{variable}: {stage}", flush=True)
                if not args.check_only:
                    execute_job(job)
    print("Tutorial only: do not replace released scientific models with these small-data checkpoints.")


if __name__ == "__main__":
    main()
