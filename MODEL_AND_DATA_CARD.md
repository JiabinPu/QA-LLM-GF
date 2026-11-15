# Model, data and execution card

Status: v1.0.2 compiled-solver distribution; author-code, model and derived-data
licences remain pending. See `LICENSE_PENDING.md` and `third_party/`. Successful
software checks do not assign redistribution rights or establish scientific
accuracy. No complete-tile production or retraining is required to install or
inspect this candidate.

## Binary solver distribution

The ST-Tensor/L1 `api` and `_kernels` implementations are supplied only as
compiled extensions. The public interface, launchers and surrounding pipeline
remain available, but the core source implementations are not part of this
release. This distribution requires CPython 3.12 and a compatible PyTorch build.
Binary targets are macOS Apple Silicon arm64, Linux x86_64 and Windows x86_64;
Intel Mac and other Python versions or architectures are outside this support
list. The Linux wheel was built and CPU-tested on Ubuntu 22.04 x86_64 / glibc 2.35. Other Linux distributions are unverified; the generic linux_x86_64 tag is not a manylinux compatibility claim.

The complete package carries the matching native subdirectories and runs in
place; the solver alone can be installed from the matching standalone wheel.
No source build is needed or provided. Unsupported targets or missing binaries
raise a clear error without switching to Python source or downloading another
implementation. CPU operation and optional MPS/CUDA interfaces are retained.
Available local measured evidence covers its recorded CPU/MPS profiles; CUDA
has not been tested locally. A platform binary's existence does not itself
establish a successful execution test on that platform.

Source previously published in old releases or Git history cannot be retracted
by this packaging change. Compiled distribution is not a guarantee against
inspection or reverse engineering.

## Model identity

Independent LAI and FPAR controllers use separate numerical heads/weights and a
language-pretrained GPT-2 first-block backbone. Their eight training tiles and
two independent application tiles are recorded in `core/tile_registry.json`;
the latter do not enter model fitting or calibration. The scientific source
checkpoint identities remain in `models/*/model.json`.

The scientific numerical implementation, frozen safetensors and calibration
coefficients are inherited from the Mac production workflow. Runtime adaptation supplies
portable paths, device selection, explicit input-layout transfer and CPU/GPU
owner scheduling. It does not retrain weights or revise quality definitions,
candidate selection, learned budgets, solver equations, fidelity or encoding.

Safetensors contain the complete fitted state, including backbone tensors,
without optimizer state. The local GPT-2 initialization asset remains necessary
for the current model constructor before the full state is loaded. It also
initializes tutorial training. It is language-pretrained initialization, not a
historical fitted QA-LLM checkpoint. Model, metadata, calibration CSV/report
and their manifests must remain together.

The controller receives numerical remote-sensing evidence rather than natural
language prompts. It selects Local/PHENO evidence and allocates bounded
ST-Tensor+L1 iterations. Numerical solvers and observation fidelity remain part
of inference and evaluation. These examples do not demonstrate that linguistic
knowledge itself causes an accuracy improvement.

## Inputs and teaching data

The bundled training tutorial uses eight spatially disjoint 40 x 40 locations
from h23v04, retaining 2011-2015 context with 2013 as the center year. The spatial
split is four training locations, two validation and two test. Selection used
input coverage, not output error. The source-tile PFT support summary is retained.
This subset illustrates the training chain, not the full population used for
published model fitting.

The bundled production input is a native aligned 200 x 200 region of **h27v06**
for 2001-2024. It retains all dates and pre-operational/unavailable sensor slots.
The four Mac reproduction cases also use this same bundled h27v06 input;
there is no separate h25v06 reference input in this package.
Neither teaching data set is a raw MODIS/VIIRS HDF archive.

`example_01_train_validate.py --stage prepare` assembles indices and masking
contracts for the supplied compact tutorial data. `example_03_produce_tile.py`
consumes externally prepared native compact stacks and annual landcover.
Neither entry point promises raw-HDF ingestion or supplies a complete tile.

Annual compact observations have h5py order [sensor, time, col, row], equivalent
to MATLAB [row, col, time, sensor]. Sensors are Terra, Aqua and VIIRS, with
DOY 1:8:361 (46 composite start dates) per year.

| Dataset | Storage | Decoding / role |
|---|---|---|
| `obs/LAI_DN` | uint8 | x0.1 LAI; 255 unavailable |
| `obs/FPAR_DN` | uint8 | x0.01 FPAR; 255 unavailable |
| `uncert/LAI_STD_DN` | uint8 | State-aware StdDev, x0.1 before semantic normalization |
| `uncert/FPAR_STD_DN` | uint8 | State-aware StdDev, x0.01 before semantic normalization |
| `score/LAI`, `score/FPAR` | uint8 | Stored semantic quality /100 |
| `flag/LAI`, `flag/FPAR` | uint8 | bit0 hard-usable;1 stored anchor;2 valid;3 primary;4 backup;5 atmosphere;6 land;7 cloud |
| `qa/SCF_QC` | uint8 | Retrieval state, including main nonsaturated/saturated distinction |
| `LC_Type_3` | uint8 | Annual native PFT; vegetation codes 1-8; h5py col,row |

Read `quality.py` before changing semantic quality rules. Saturated StdDev is
already normalized in the adopted protocol and must not again be divided by
the retrieved variable. Unsaturated StdDev uses the relative-value transform.
The engine recomputes candidate definitions; a stored anchor flag alone is not
the final saturation-aware anchor decision. Backup retrievals remain in compact
storage but are not admitted as model anchors.

The inactive variable's numerical slots are isolated using the same function
as independent-variable training. Packing LAI and FPAR in one archive does not
mean joint model fitting. External SI products and ground references are not
inference inputs. Ground data do not fit the models or uncertainty calibration.

For native tile production, preserve the 2400 x 2400 grid, 40 x 40 patches,
8 x 8 solver cores, annual vegetation masks and all 24 context years. A target
year uses the adopted retrospective five-year window; requested output years
must not discard neighboring-year input context. Landcover orientation and
vegetation definitions must agree with the compact inputs.

## Numerical execution and reproduction

Production supports CPU, Apple MPS, CUDA and concurrent CPU/GPU owners.
Its default `auto` coordinator starts CPU and GPU lanes when a supported GPU
is available, with five CPU and five GPU owners by default; without a GPU it
uses CPU-only execution with five CPU workers by default. Explicit `cpu_gpu`
also permits CPU-only fallback and records the requested and effective backend
in its execution profile. Explicit `cpu` forces CPU-only operation; explicit
`mps` or `cuda` fails if that requested device is unavailable. A zero CPU or GPU
owner count disables that lane; both counts cannot be zero. The automatic
fallback exception is no available GPU with a requested CPU count of zero:
`auto`/`cpu_gpu` starts one CPU worker and records requested zero, effective one
and the reason. Forced CPU mode requires a positive CPU count; explicit
MPS/CUDA mode requires a positive GPU count.

Five GPU owners share one selected physical accelerator; they are not five
physical GPUs. Each owner uses one numerical CPU thread by default. Training
and single-tensor automatic selection use one device, whereas the production
coordinator can combine owner lanes. Hardware, available memory, device choice,
fallback and owner counts belong in run metadata rather than the scientific
model configuration.

The Mac path uses FP32, preserves the adopted input stride layout during device
transfer, and enables CPU fallback for unsupported MPS operations. It does not
use mixed precision or substitute alternative LayerNorm/GELU, QR/SVD or routing
implementations. Concurrent owners receive separate patches. CPU and accelerator
arithmetic can still produce different final DN near boundaries, so CPU results
must be checked against CPU references and MPS results against MPS references.
No cross-platform bitwise-equivalence guarantee is made.

### Historical source-release validation

The checks in this subsection describe the earlier source releases and their
recorded environments. They are not new validation of the v1.0.2 compiled
extensions. In particular, Mac CPU/MPS records do not establish execution on
the Linux or Windows binary targets or on CUDA. Read any v1.0.2 validation
report using its stated Python, platform, device and sample scope.

The earlier reproduction baseline is a fixed-input small sample from
the frozen local Mac runtime and frozen weights. The source is the bundled
h27v06 teaching crop. CPU and MPS references are stored separately in
`data/mac_reference/reference_cpu.npz` and `reference_mps.npz`, with source,
model and fixture identities in their manifest. Verification does not require
the original production checkout.

CPU and MPS verification each passed four native 40 x 40 cases at zero-based
origins: `(2013,0,0)`, `(2013,0,40)`, `(2001,0,0)`, and `(2024,0,0)`. All six
uint16 layers had zero differing DN values and maximum DN difference zero
against their same-backend reference. Actual recorded solver diagnostics also
matched. These four complete-window cases all have zero solver budgets.
A separate packaging-time native h23v04 case `(2013,0,0)` also matched the frozen
local runtime on CPU and MPS: preprocessing values and strides, all six DN
layers and solver diagnostics. Its solver budgets are also zero. This external
native-tile input is not bundled; the shipped fixtures cover h27v06.

A separate 10-patch integration used five CPU and five MPS owners, executing
five patches on each backend. All 46 assembled H5s passed patch-by-patch
six-layer readback checks. Final replay retained all backend assignments and
the 10 six-layer patch checksums. All numeric fields in the three solver CSVs
matched, excluding run-fingerprint metadata. Resume reused all 10 cached
patches with zero new inference and verified the 46 sealed H5s. A
GPU-unavailable `auto` configuration also passed a one-patch CPU-only smoke and
its 46 H5 assembly checks. These are small-scope integration checks, not a
throughput benchmark or a whole teaching crop/tile run.

Four positive-budget native-core component tests passed on CPU/MPS with ST=3
and L1=2 budgets. On MPS, ST actually performed three iterations, six QR and
nine SVD calls with Torch's expected SVD CPU-fallback warning; L1 performed
two iterations, one Cholesky and two solves. This verifies component execution,
not complete-window positive-route acceptance. All 40 unit tests passed: these
four CPU/MPS component checks plus 32 CPU-only owner-selection, layout and
solver-statistics checks.

`verify_reproduction.py --device cpu` and `--device mps` compare the package
with the corresponding included reference; `--output` selects the JSON report.
Measured environment and scope are recorded in `environment_tested.json`.
CUDA remains untested. No full-tile production or retraining was performed.

The original complete Mac product H5s were deleted, so these fixed-input
checks are not a regression against those deleted annual products or old
Windows DN. External SI TIFFs are comparison products, not interchangeable
six-layer QA-LLM-GF references. The measured sample does not establish exact
agreement for all production tiles or scientific accuracy. Reference fixtures
do not contain personal absolute paths or old console logs.

## Outputs, QA and uncertainty

Final H5 files contain `/Lai`, `/Fpar`, `/Lai_Uncertainty`, `/Fpar_Uncertainty`,
`/Lai_QA`, and `/Fpar_QA`, all uint16. Physical quantities and uncertainty use
scale 0.001; QA is unscaled. Exclude fill 65535 before scaling or decoding bits.
Nonvegetated pixels remain fill-valued. MATLAB reads row,col directly; h5py
sees col,row storage. Uncertainty is a calibrated sigma, not a 95% interval
half-width, and calibration is applied exactly once.

QA bits 0-2 encode source mode; 3-4 usable sensor count; 5-6 strict candidate
count; 7-8 saturated candidate count; 9-15 confidence 0-100. Source modes are:

1. strict-only candidate evidence;
2. saturated-only candidate evidence;
3. mixed strict and saturated evidence;
4. other usable soft observations;
5. no current usable observations, Local-selected reconstruction;
6. no current usable observations, PHENO-selected reconstruction;
7. structural ST-Tensor+L1 routing.

Mode 7 is a pixel/period coverage label. It does not by itself measure actual
solver activation, iterations, runtime saving or a nonzero final adjustment.
Actual ST/L1 activation requires native-core diagnostics with a stated valid
core denominator and variable slot. It cannot be inferred from QA coverage or
a candidate-region fraction. Do not combine these statistics under one label.

Uncertainty follows the adopted satellite-reference residual model rather than
demonstrated ground-truth error for each pixel. Validation data are reused for
model selection, budget and uncertainty calibration, so they are not a fresh
independent conformal split. No finite-sample distribution-free coverage claim
is made. Test data neither select epochs nor fit calibration.

## Limits and use

This is retrospective reconstruction, with preceding and following years
available to phenology. It is not causal forecasting or a guarantee after
disturbance or after all MODIS observations disappear. Candidate smoothing,
inter-sensor inconsistency, high values and large gaps still require inspection
with support and uncertainty diagnostics.

The public examples demonstrate execution and numerical reproduction within
a declared scope. They do not independently validate the whole world, every
biome or a new instrument. Tutorial scores are not paper accuracy estimates.
Use verified manuscript tables for scientific accuracy; a complete product
deposit needs its own coverage/QC inventory and citation.

Output directories are separate from installed source, models and inputs.
Paired LAI+FPAR production and the native-tile entry retain an output lock and
checksummed patch cache. Version 1.0.2 does not compute a run fingerprint from
code, model, input or environment and does not use one to block production or
resume. Tile, target years, patch identities, geometry and required data
structure are still checked. File-level SHA validation, patch checksums and
final-product integrity seals remain; they are integrity checks rather than
run fingerprints. Structural compatibility does not demonstrate that changed
inputs or settings are scientifically interchangeable, so users retain the
version and data records needed for controlled comparisons. Backend assignments
are durable before dispatch in `backend_assignments.jsonl`, and
`execution_manifest.json` provides the map for `--replay-map` into a compatible
fresh destination.
Unavailable recorded backends cannot silently become another backend on resume
or replay. Dynamic new runs may distribute patches differently.

Paired final H5s are under `Products/<year>/`, sealed with
`annual_complete.json` after six-layer patch-by-patch readback. Three tables
are under `Products/`: `solver_by_year_variable.csv`,
`solver_total_by_variable.csv`, and `solver_budget_distribution.csv`, with
`solver_statistics_notes.txt` defining units and denominators. There is no
solver runtime CSV. Actual activation statistics come from the recorded native
core diagnostics, independently of QA mode-7 pixel coverage.

The retained separate-variable teaching interface uses one `cpu/mps/cuda/auto`
device, writes `Parts/` and final `<output>/<year>/`, may replace same-named
teaching results, and does not produce paired solver statistics. It does not
support `cpu_gpu` or replay maps. Keep its output separate from paired runs.
Do not run concurrent writers into any one output directory.
