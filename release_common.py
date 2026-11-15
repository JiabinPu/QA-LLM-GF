"""Portable paths and model loading for the public examples."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
CORE = ROOT / "core" / "full_algorithm"
sys.path.insert(0, str(CORE))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# Must precede the first torch import, including imports by preflight contracts.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
for _thread_name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                     "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_thread_name] = "1"


def sha256(path):
    """Identify model/data assets, not executable code."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def confined_path(folder, name):
    folder = Path(folder).resolve()
    path = (folder / name).resolve()
    if folder not in path.parents:
        raise ValueError(f"Path leaves its asset directory: {name}")
    return path


def output_directory(path):
    path = Path(path).expanduser().resolve()
    runs = (ROOT / "runs").resolve()
    if path == ROOT or (ROOT in path.parents and path != runs and runs not in path.parents):
        raise ValueError("Use runs/ or a separate output directory, never installed code/assets/data")
    return path


def execute_job(job):
    env = dict(os.environ, PYTHONPATH=str(CORE), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    subprocess.run(job, cwd=ROOT, env=env, check=True)


def verify_manifest(folder):
    """Verify the content, not only the presence, of every declared asset."""
    folder = Path(folder).resolve()
    manifest = load_json(folder / "manifest.json")
    for name, expected in manifest["files"].items():
        path = confined_path(folder, name)
        if not path.is_file():
            raise FileNotFoundError(f"Missing asset: {name}")
        if sha256(path) != expected:
            raise ValueError(f"Asset SHA256 mismatch: {folder.name}/{name}")
    return manifest


def load_frozen_model(variable, device):
    """Load the complete fitted state from local safetensors."""
    from safetensors.torch import load_file
    from qa_llm_v3.model import ControllerConfig, QALLMAdaptiveController
    from qa_llm_v3.quality import validate_checkpoint_protocol
    from qa_llm_v3.solver import SolverConfig

    folder = ROOT / "models" / variable
    meta = load_json(folder / "model.json")
    if meta["role"] != "frozen_scientific_model" or meta["model_config"]["active_variable"] != variable:
        raise ValueError("The production example requires the released scientific model")
    validate_checkpoint_protocol(meta, budget=True)
    config = dict(meta["model_config"])
    config["gpt_model"] = str(ROOT / "assets" / "gpt2_first_block")
    model = QALLMAdaptiveController(ControllerConfig(**config))
    model.load_state_dict(load_file(str(folder / "model.safetensors")), strict=True)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, SolverConfig(**meta["solver_config"]), meta


def load_calibration(variable):
    from qa_llm_v3.product_calibration import ProductCalibration
    folder = ROOT / "models" / variable
    return ProductCalibration(folder, variable, folder / "model.safetensors")


def make_manifest(folder, **metadata):
    """Asset-export helper; no code inventory or runtime code checks."""
    folder = Path(folder)
    files = {p.relative_to(folder).as_posix(): sha256(p)
             for p in sorted(folder.rglob("*"))
             if p.is_file() and p.name != "manifest.json" and "__pycache__" not in p.parts}
    write_json(folder / "manifest.json", dict(metadata, files=files))
