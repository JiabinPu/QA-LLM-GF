# QA-LLM-GF: portable CPU and GPU examples

## Getting the complete runnable package

Download the complete **v1.0.2** package from the [latest release](https://github.com/JiabinPu/QA-LLM-GF/releases/latest). It includes the platform-specific ST-Tensor/L1 binaries, fixed LAI/FPAR weights, local GPT-2 initialization assets, and example data. Extract it and follow this README or `START_HERE_zh.md`; the supplied solver binaries run in place without a separate solver installation.

The ST/L1 `api` and `_kernels` implementations are distributed only as compiled extensions in this version. The public interfaces and launch examples remain available, but their source implementations are not included. GitHub's automatically generated source archive is not the complete runtime package: obtain the matching binaries and `models/`, `assets/`, and `data/` from the release. `SHA256SUMS.txt` records the release attachment checksums.

This package retains the fixed models and scientific settings of the Mac production workflow. Production needs no retraining, model server or API key. Author-code, model and derived-data licences remain pending; see `LICENSE_PENDING.md`.

## Install

Use **CPython 3.12** and a compatible platform-specific PyTorch installation.
The binary distribution provides these targets:

| Operating system | Architecture | Binary target |
|---|---|---|
| macOS | Apple Silicon arm64 | CPython 3.12 |
| Linux | x86_64 | CPython 3.12; built and CPU-tested on Ubuntu 22.04 / glibc 2.35 |
| Windows | x86_64 | CPython 3.12 |

Intel Mac and other Python versions or architectures are outside this binary
support list. Unsupported targets or missing binaries produce a clear error;
the loader does not use a Python implementation, build from source, or download
a substitute.

From the extracted complete package, create an environment and install its
runtime dependencies:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate with `.venv\Scripts\activate`. The complete package
contains the native subdirectories for the listed targets and selects the
matching binary locally. For the solver alone, download its matching CPython
3.12 wheel and install it with `python -m pip install /path/to/downloaded.whl`;
no solver source build is required. Install an appropriate PyTorch build first
when device selection matters. Apple GPU execution uses MPS, not CUDA.
`requirements-macos-arm64.lock.txt` records a Mac environment, not a universal
cross-platform lock file.

Keep `models/` and `assets/gpt2_first_block/` for production. The model constructor
loads local GPT-2 initialization before the complete frozen safetensors state.
Numerical work does not require MATLAB, GDAL, SciPy or an external source checkout.

## ST-Tensor/L1 module

The module is located in `core/full_algorithm/st_tensor_l1/` and requires PyTorch,
but no learned model weights when used independently. `api` and `_kernels` are
compiled-only; `qa_llm_v3.solver` keeps the original import interfaces compatible.
Gradient-free inference uses the joint encoded-core interface. Model routing,
normalization, final QA projection and six-layer encoding remain in the caller.

See [ST_TENSOR_L1.md](ST_TENSOR_L1.md) for the interfaces, or run the supplied
synthetic example from the complete package:

```sh
python example_st_tensor_l1.py --device cpu
```

This binary-only distribution does not retract source already published in
older releases or Git history. Compilation is a distribution choice, not a
claim that inspection or reverse engineering is impossible.

## Choose production resources

Paired LAI+FPAR production accepts `--device cpu`, `mps`, `cuda`, `auto`, or
`cpu_gpu`. The default `auto` uses five CPU owners plus five GPU owners when
a supported GPU is available, and five CPU owners otherwise. Explicit `cpu_gpu`
also allows CPU-only fallback; `execution_resources.json` records requested and
effective resources and the fallback reason. Explicit `cpu` forces CPU-only
execution. Explicit `mps`/`cuda` requires that accelerator and fails if unavailable.

`--gpu-device auto` selects the accelerator for joint mode. Five GPU owners
share one physical GPU; they do not mean five GPUs. Each numerical owner uses
one CPU thread. Reduce owner counts for smaller machines; five plus five is a
resource default, not a universal performance optimum.

Set a lane's worker count to zero to disable it, but both counts cannot be zero.
When `auto`/`cpu_gpu` has no GPU and the requested CPU count is zero, it starts
one fallback CPU owner and records that exception. Forced CPU mode requires a
positive CPU count; explicit MPS/CUDA requires a positive GPU count. Training
and the legacy single-variable tutorial select one device, rather than the
paired production coordinator's owner lanes.

The runtime retains FP32, one numerical thread per owner and the adopted
input stride/offset layout during device transfer. MPS uses CPU fallback for
unsupported operations. There is no AMP, alternative LayerNorm/GELU, or change
to solver formulas, thresholds, routing, frozen weights or six-layer encoding.
CPU and GPU can still round differently near numerical boundaries. Replay
therefore preserves each patch's backend rather than assuming CPU and MPS DN
are interchangeable.

## Example 1: tutorial training and validation

```sh
python example_01_train_validate.py --variable LAI --stage all
python example_01_train_validate.py --variable FPAR --stage all
```

For a small execution example:

```sh
python example_01_train_validate.py --variable both --stage all --smoke
```

Stages are `prepare`, `train`, `budget`, `evaluate`, `uncertainty`, and `all`.
`prepare` organizes indices, masks and contracts for already compacted teaching
data; it does not ingest original MODIS/VIIRS HDF archives. Training retains
its own device options; the paired CPU/GPU owner parameters are for frozen-model
production.

Training data contain eight aligned 40 x 40 locations from h23v04: four train,
two validation and two test. Every location retains 2011-2015 context, with 2013
as target year. Spatial membership is preserved across neighboring years.
Selection used input coverage, not output error. The source-tile PFT support
summary is retained. Only language-pretrained GPT-2 initializes this tutorial;
no historical fitted QA-LLM checkpoint is loaded.

Defaults are two controller epochs and three budget-head epochs. Smoke uses
one epoch and minimal training/validation batches and budget samples, then
evaluates the supplied validation/test cases. Regular output is `runs/tutorial/`;
smoke output is `runs/tutorial_smoke/`.

| Stage | Results below the tutorial output folder |
|---|---|
| Train | `Module02B_Training/<variable>/training_log.csv`, `best.pt` |
| Budget | `Module02C_BudgetCalibration/<variable>/best_budget.pt`, metrics |
| Evaluate | `Module03A_Evaluation/<variable>/val` and `test` |
| Uncertainty | `Module03B_UncertaintyCalibration/<variable>` |

The tutorial demonstrates the learning chain, not paper-level eight-tile
accuracy. Its fitted checkpoints do not replace the released production
models. Training uses five masking scenarios; evaluation uses eight. References
are withheld satellite observations, not field truth. Test data neither select
epochs nor fit calibration. Validation is reused for model selection, budgets
and uncertainty fitting, so it is not a fresh independent conformal split.
Original deployment criteria remain in force.

Loss and physical/normalized RMSE tables are retained. LAI normalization divides
by 7 and FPAR by 1. `R2` is `1-SSE/SST(reference)`; `PearsonR2` is squared correlation.

## Example 2: bundled production crop

The teaching input is an aligned 200 x 200 region of **h27v06**, retaining
2001-2024. The Mac reference described below uses four patches from this same
bundled input.

```sh
python example_02_produce_region.py --check-only
python example_02_produce_region.py --device cpu --smoke --output runs/smoke_cpu
python example_02_produce_region.py --device mps --smoke --output runs/smoke_mps
python example_02_produce_region.py --device auto --cpu-workers 5 --gpu-workers 5 --threads 1
```

Explicit MPS requires an available Apple GPU. `--smoke` targets 2013 and the
first aligned 40 x 40 patch; without `--output` its destination is
`runs/production_smoke_<device>/` (for example, `production_smoke_auto`).
Full paired execution covers 25 patches and 24 target
years, writing 46 final H5 files per year under `Products/<year>/`: 1,104 files
in total. Its default root is `runs/production_200x200_<device>/`. `--years 2013` limits the target year while retaining neighboring
input context. Complete example execution is optional and is not triggered by
`--check-only`.

The legacy separate-variable teaching interface remains available:

```sh
python example_02_produce_region.py --variable LAI --device cpu --output runs/single_variable
python example_02_produce_region.py --variable FPAR --device cpu --output runs/single_variable
```

These commands must run sequentially. This legacy interface uses one device
(`cpu/mps/cuda/auto`), writes variable files under `Parts/`, and merges final
files under `<output>/<year>/` when both variables are ready. It does not
support `cpu_gpu` or `--replay-map` and does not produce the paired solver
statistics. It may overwrite its same-named teaching results. Keep its output
separate from the paired interface's output.

## Example 3: native tile from prepared external inputs

Supply a tile directory containing native compact stacks, annual landcover and
the token index. Replace `<tile-directory>` below; complete tile inputs are not
bundled.

```sh
python example_03_produce_tile.py --data-root <tile-directory> --years 2001:2024 --device auto --cpu-workers 5 --gpu-workers 5 --threads 1 --output runs/native_tile --check-only
python example_03_produce_tile.py --data-root <tile-directory> --device auto --cpu-workers 5 --gpu-workers 5 --threads 1 --output runs/native_tile_smoke --smoke
```

`--check-only` validates inputs/settings without inference or output writes.
`--smoke` targets 2013 and its first native 40 x 40 patch. Remove `--check-only`
from the first command only when intending to process all requested tile years.
The input layout is:

```text
<tile-directory>/
  Module00_Landcover/Masks/LC_Type_3_vegetation_mask_<tile>_<year>.h5
  Module01_Intermediate/
    CompactStacks/<tile>/<year>/CompactQA_<tile>_<year>_R01_C01.h5
    PythonDataset/token_index_python.csv
```

Retain every year's 16 block files, all 24 context years (2001-2024), annual
landcover and the complete native index. A 2400 x 2400 tile has 3,600 native
40 x 40 patches per target year and 8 x 8 solver cores. Inputs must follow the
QA, StdDev, sensor and orientation contract in `MODEL_AND_DATA_CARD.md`.
This entry consumes compact inputs; it is not a raw satellite-HDF preprocessor.
It uses the code and frozen models inside this package. Default tile outputs are
`runs/production_<tile>_<device>/`, or
`runs/production_<tile>_smoke_<device>/` with `--smoke`. Device-specific defaults
keep CPU/MPS/auto examples from accidentally sharing a cache.

## Paired outputs, resume and replay

The default `--variable both` region interface and native-tile interface retain
an output lock, cached patches and integrity checks. Version 1.0.2 does not
calculate a code/model/input/environment run fingerprint and does not use one
to block production or resume. Tile, target years, patch identity, geometry and
required data structure are still checked; file SHA checks, patch checksums and
final-product seals remain in force. These integrity checks are distinct from
a run fingerprint. Structural compatibility does not certify that changed
inputs or settings are scientifically interchangeable; keep the version and
data records needed for a controlled comparison. Do not run multiple writers
into one output directory. Recorded GPU patches retain their backend identity.

`backend_assignments.jsonl` durably records each patch backend before dispatch.
`execution_manifest.json` retains completed patch backends and six-layer
checksums. To repeat the same patch/device mapping into a fresh destination:

```sh
python example_02_produce_region.py --smoke --device auto --output runs/paired_smoke
python example_02_produce_region.py --smoke --device auto --replay-map runs/paired_smoke/execution_manifest.json --output runs/paired_smoke_replay
```

Keep the same input and target scope and make every recorded backend available.
Replay rejects an unavailable required backend rather than changing its device.
A new dynamic run may assign patches differently from an earlier run; use the
saved map when the exact backend assignment matters.

Paired outputs include:

| Location | Contents |
|---|---|
| `Patches/<year>/` | Six-layer native patch data, actual solver diagnostics and checksums |
| `Products/<year>/` | 46 final H5s and `annual_complete.json` content seal |
| `Products/solver_by_year_variable.csv` | Year/variable solver statistics |
| `Products/solver_total_by_variable.csv` | Variable totals |
| `Products/solver_budget_distribution.csv` | Applied budgets and actual iteration distributions |
| `Products/solver_statistics_notes.txt` | Counting units, denominator and interpretation |
| `execution_resources.json` | Requested/effective owners and fallback |
| `product_summary.json` | Completion and assembly-verification summary |

There is no solver runtime CSV. Actual activation uses native-core iteration
counters, not QA mode-7 pixel coverage or a candidate fraction. Read the notes
before comparing statistics across scopes. Cached patch content is validated,
and assembly reads all six layers back from final H5s and compares them with
their source patches before sealing each year.

Final H5 datasets are `/Lai`, `/Fpar`, `/Lai_Uncertainty`, `/Fpar_Uncertainty`,
`/Lai_QA`, and `/Fpar_QA`, all uint16. Physical estimates and uncertainties use
scale 0.001; exclude fill 65535 first. QA is unscaled. Nonvegetated pixels are
fill-valued. Uncertainty is calibrated sigma, not a 95% interval half-width.
MATLAB reads row,col directly; h5py sees col,row storage.

## Verification scope

Binary-release validation is specific to its Python, platform and device
profile. The records below describe earlier source releases; they are not new
v1.0.2 binary test results. Local CPU/MPS evidence is available for those
recorded Mac profiles. CUDA has not been measured locally, and no statement
below establishes testing of every distributed platform binary.

### Historical v1.0.1 source-wrapper verification

The numerical ST/L1 kernels are unchanged from v1.0.0. All 54 CPU/MPS unit
tests passed. The four CPU reference cases still match all six DN layers and
solver diagnostics exactly. Direct v1.0.0 versus v1.0.1 comparison on the
current MPS environment also matches all six layers and diagnostics for the
four cases. One explicitly forced positive-budget 2013 window additionally
matches on CPU and MPS, including intermediate predictions and solver counts;
this test is not a measurement of natural controller activation.

The stored MPS reference was generated on macOS 26.5.1. On the current macOS
27.2.1, both versions show the same small differences from that reference,
including a packed QA value. The reference files are retained unchanged, so
`verify_reproduction.py --device mps` remains strict and reports these
differences. See `solver_encapsulation_verification.json` for the actual
environment, comparison scope and per-layer counts. This is a same-environment
refactor check, not a claim of universal cross-platform bitwise equivalence.

### Historical v1.0.0 Mac reproduction scope

Run the bundled reference checks without training or whole-tile production:

```sh
python verify_reproduction.py --device cpu --output runs/reproduction_cpu.json
python verify_reproduction.py --device mps --output runs/reproduction_mps.json
```

CPU and MPS each passed four fixed h27v06 cases: `(2013,0,0)`, `(2013,0,40)`,
`(2001,0,0)`, `(2024,0,0)`, where coordinates are zero-based native patch origins.
All six uint16 layers matched their same-backend reference with zero differing
values and zero maximum DN difference. Recorded solver diagnostics also matched.
The references are `data/mac_reference/reference_cpu.npz` and
`reference_mps.npz`, generated afresh with fixed bundled inputs, frozen weights
and the preserved local Mac runtime. They are not old Windows DN and do not
require the original project at verification time.

A separate 10-patch integration used five CPU and five MPS owners, with five
patches executed by each backend. All 46 assembled H5 files passed six-layer
patch-by-patch readback checks. Final replay retained the complete backend map
and all 10 six-layer patch checksums; the three solver CSVs retained every
numeric value, excluding run-fingerprint metadata. A subsequent resume reused
all 10 cached patches with zero new inference and verified all 46 sealed H5s.
This is a small-scope integration check, not a speed benchmark or complete
200 x 200/whole-tile production.

An additional packaging-time native-tile comparison used h23v04 at
`(2013,0,0)` on CPU and MPS. Preprocessed values and strides, all six DN layers
and actual solver diagnostics matched the frozen local runtime. That external
native-tile input is not bundled; the included reproducibility fixtures remain
the four h27v06 cases above.

The complete-window h27v06 and native h23v04 comparison cases all have zero
solver budgets. Four separate
native-core component tests passed on CPU/MPS with positive ST=3 and L1=2
budgets. The MPS ST test executed three iterations (six QR and nine SVD calls),
with the expected Torch SVD CPU-fallback warning; L1 executed two iterations
(one Cholesky and two solves). These component checks do not establish
end-to-end acceptance of a positively routed full window.

A GPU-unavailable `auto` run also passed a one-patch CPU-only smoke and all 46
H5 assembly checks. All 40 unit tests passed: 32 CPU-only owner/layout/statistics
checks plus the four CPU/MPS positive-budget component checks described above.
`environment_tested.json` records these measured scopes. CUDA remains untested;
none of these checks is a speed benchmark.

The original complete Mac product H5s were deleted. These new fixed-input
checks are therefore not a regression against those deleted annual products.
External SI TIFFs are not substitutes for the six-layer QA-LLM-GF reference.
No retraining or full-tile production was performed for this candidate; the
checks establish only their recorded numerical/software scope, not scientific
accuracy or universal cross-platform bitwise equivalence.

## MATLAB

```matlab
pythonExe = 'python';
Example01_Train_Validate('LAI', 'all', pythonExe);
Example02_Produce_Region('both', 'smoke', pythonExe, 'runs/matlab_smoke', 'auto', 5, 5);
```

`Example03_Produce_Tile.m` launches native compact-tile production; its default
mode checks inputs before inference. These MATLAB entry points delegate all
numerical work to Python; `Read_Example_Product.m` reads the final H5s. No
in-process `pyenv`, MATLAB GPU toolbox or Mapping Toolbox is needed.

## Package contents

`core/` contains the scientific interfaces, pipeline and compiled ST/L1 binaries;
`portable_runtime.py` handles devices, transfers and paired scheduling; `config/` holds tutorial settings; `models/`
and `assets/` contain frozen local resources; `data/` contains teaching inputs
and the small Mac reference; `third_party/` holds notices. Results are created
under `runs/` unless another destination is selected. The distribution contains
no local virtual environment, full-tile archive, old console logs, personal
absolute paths or external source checkout.

## v1.0.2 compiled-module verification

All 64 local CPU/MPS unit tests passed. The compiled solver was also checked against the original Python implementation in 14 component and gradient cases on each local backend, with 119 tensors compared per backend and zero differences. The actual GitHub-built Mac wheel passed the MPS component check and the positive-budget CPU/MPS window check on this Mac. Three build hosts (macOS arm64, Ubuntu 22.04 x86_64 and Windows x86_64) each passed 14 same-environment CPU comparisons. CUDA is optional and was not tested here.

Four fixed-model cases preserve all six DN layers and actual solver diagnostics on the same backend. A forced positive-budget 2013 window also preserves intermediate outputs and diagnostics; it is not a natural activation-rate or speed measurement. A real one-patch CPU integration produced and read back all 46 final H5 files, matched the reference DN, and then resumed with zero new inference. New production records contain no run fingerprint. See `binary_module_verification.json` for the scope. The historical MPS reference remains strict and is unchanged.

