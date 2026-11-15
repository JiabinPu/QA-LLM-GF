"""Self-contained ST-Tensor/L1 component regression checks.

Only PyTorch, the bundled compiled modules and the public pipeline helper are
needed. Fixtures are synthetic native-size cores, not a controller evaluation
or a full-tile production run. Core Python source files are not required.
"""
from __future__ import annotations

import ast
from dataclasses import asdict, replace
from importlib.machinery import ExtensionFileLoader
import inspect
import os
import pickle
from pathlib import Path
import sys
import unittest
from unittest import mock

sys.dont_write_bytecode = True
_ROOT = Path(__file__).resolve().parents[1]
_CORE = _ROOT / "core" / "full_algorithm"
sys.path.insert(0, str(_CORE))

import torch

from st_tensor_l1 import (
    SolverConfig, adaptive_l1_trend_filter, adaptive_sttensor,
    production_config, refine_core, refine_encoded_core,
)
from st_tensor_l1 import _kernels, api
from qa_llm_v3 import solver as compatibility_solver


def _pipeline_curvature():
    # Read the actual helper without importing any model, data, or HF modules.
    path = _CORE / "qa_llm_v3" / "pipeline.py"
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                and n.name == "_phenology_curvature_protection")
    scope = {"torch": torch}
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    exec(compile(module, str(path), "exec"), scope)
    return scope[node.name]


_CURVATURE = _pipeline_curvature()


def _fixture(device="cpu", noncontiguous=False):
    core = torch.arange(3, dtype=torch.float32).view(3, 1, 1, 1)
    pixel = torch.arange(64, dtype=torch.float32).view(1, 64, 1, 1)
    time = torch.arange(46, dtype=torch.float32).view(1, 1, 46, 1)
    year = torch.arange(5, dtype=torch.float32).view(1, 1, 1, 5)
    phase = 0.23 * time + 0.061 * pixel + 0.13 * year + 0.17 * core
    prior = 0.46 + 0.23 * torch.sin(phase)
    observed = (0.49 + 0.29 * torch.sin(phase + 0.28)).clamp(0, 1)
    reference = 0.44 + 0.18 * torch.sin(phase - 0.16)
    weight = (0.12 + 0.72 * ((pixel + time + year + core) % 4 == 0)).expand_as(prior).clone()
    mask = torch.zeros_like(prior, dtype=torch.bool)
    anchors = observed.clone()
    for index, value in (
        ((0, 0, 0, 0), 0.0), ((0, 31, 23, 2), 1.0),
        ((1, 63, 45, 4), 0.375), ((2, 4, 5, 1), 0.0),
        ((2, 33, 26, 3), 1.0),
    ):
        mask[index] = True
        anchors[index] = value
    values = dict(
        observed=observed, observation_weight=weight, prior=prior,
        prior_strength=torch.tensor([0.8, 0.7, 0.9]),
        anchor_value=anchors, anchor_mask=mask,
        difficulty=torch.tensor([0.25, 0.6, 0.8]),
        l1_lambda=torch.tensor([0.02, 0.025, 0.03]),
        st_budget=torch.tensor([0, 3, 8], dtype=torch.long),
        phenology_reference=reference,
    )
    values = {key: value.to(device) for key, value in values.items()}
    if noncontiguous:
        for key, value in values.items():
            if value.ndim == 4:
                values[key] = value.transpose(1, 2).contiguous().transpose(1, 2)
    return values


def _legacy_core_sequence(inputs, config, fastpath=True):
    """Literal old ST -> year-major L1 chain, using unchanged raw kernels."""
    with torch.no_grad():
        st, st_diag = _kernels.adaptive_sttensor(
            inputs["observed"], inputs["observation_weight"], inputs["prior"],
            inputs["prior_strength"], inputs["anchor_value"], inputs["anchor_mask"],
            inputs["difficulty"], config, training=False,
            iteration_budget=inputs["st_budget"], execution_fastpath=fastpath,
        )
        n, pixels, times, years = st.shape
        def sequence(value):
            return value.permute(0, 1, 3, 2).reshape(n, pixels, years * times)
        signal = sequence(st)
        protection = _CURVATURE(sequence(inputs["prior"]))
        l1_budget = torch.ceil(
            inputs["st_budget"].to(inputs["prior"].dtype)
            / max(float(config.max_iterations), 1.0)
            * float(config.l1_max_iterations)
        ).to(torch.long)
        l1_budget = torch.where(
            inputs["st_budget"] > 0,
            l1_budget.clamp(min=int(config.l1_min_iterations)),
            torch.zeros_like(l1_budget),
        )
        filtered, l1_diag = _kernels.adaptive_l1_trend_filter(
            signal, sequence(inputs["observed"]), sequence(inputs["observation_weight"]),
            sequence(inputs["anchor_value"]), sequence(inputs["anchor_mask"]),
            inputs["l1_lambda"], inputs["difficulty"], config, training=False,
            iteration_budget=l1_budget, curvature_protection=protection,
            execution_fastpath=fastpath,
        )
        return filtered.reshape(n, pixels, years, times).permute(0, 1, 3, 2), st, st_diag, l1_diag


def _encoded(inputs, config):
    values = {key: value for key, value in inputs.items() if key != "phenology_reference"}
    if config.residual_phenology_mode:
        reference = inputs["phenology_reference"]
        for key in ("observed", "prior", "anchor_value"):
            values[key] = (0.5 + 0.5 * (values[key] - reference)).clamp(0, 1)
    return values


class EncapsulatedSolverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def assert_tensor_identical(self, actual, expected):
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertEqual(actual.device, expected.device)
        self.assertEqual(actual.shape, expected.shape)
        self.assertTrue(torch.equal(actual, expected), f"Mismatch in {tuple(actual.shape)}")

    def assert_diagnostics_identical(self, actual, expected):
        self.assertEqual(set(actual), set(expected))
        for name, value in expected.items():
            with self.subTest(diagnostic=name):
                self.assert_tensor_identical(actual[name], value)

    def check_encoded(self, config=None, device="cpu", noncontiguous=False,
                      zero_budget=False, fastpath=True):
        config = production_config() if config is None else config
        inputs = _encoded(_fixture(device, noncontiguous), config)
        if noncontiguous:
            # Residual arithmetic may materialize; explicitly preserve a strided fixture.
            for key, value in inputs.items():
                if value.ndim == 4:
                    inputs[key] = value.transpose(1, 2).contiguous().transpose(1, 2)
            self.assertFalse(inputs["prior"].is_contiguous())
        if zero_budget:
            inputs["st_budget"] = torch.zeros_like(inputs["st_budget"])
        original = {k: (v.clone(), v.stride(), v.data_ptr()) for k, v in inputs.items()}
        expected = _legacy_core_sequence(inputs, config, fastpath)
        actual = refine_encoded_core(**inputs, config=config, execution_fastpath=fastpath)
        if device == "mps":
            torch.mps.synchronize()
        self.assert_tensor_identical(actual.values, expected[0])
        self.assert_tensor_identical(actual.st_values, expected[1])
        self.assert_diagnostics_identical(actual.st_diagnostics, expected[2])
        self.assert_diagnostics_identical(actual.l1_diagnostics, expected[3])
        self.assertFalse(actual.values.requires_grad)
        for name, value in inputs.items():
            self.assert_tensor_identical(value, original[name][0])
            self.assertEqual(value.stride(), original[name][1])
            self.assertEqual(value.data_ptr(), original[name][2])
        self.check_statistics(actual)
        return actual

    def check_statistics(self, result):
        st = result.st_diagnostics
        l1 = result.l1_diagnostics
        expected = {
            "CoreCount": st["iterations"].numel(),
            "STBudgetPositiveCoreCount": int((st["budget"] > 0).sum()),
            "L1BudgetPositiveCoreCount": int((l1["budget"] > 0).sum()),
            "STStartedCoreCount": int((st["iterations"] > 0).sum()),
            "L1StartedCoreCount": int((l1["iterations"] > 0).sum()),
            "EitherStartedCoreCount": int(((st["iterations"] > 0) | (l1["iterations"] > 0)).sum()),
            "STIterationSum": int(st["iterations"].sum()),
            "L1IterationSum": int(l1["iterations"].sum()),
        }
        stats = result.statistics()
        for name, value in expected.items():
            self.assertEqual(stats[name], value, name)
        for prefix, diag in (("ST", st), ("L1", l1)):
            for field, suffix in (("budget", "BudgetDistribution"), ("iterations", "IterationDistribution")):
                keys, counts = torch.unique(diag[field].cpu(), return_counts=True)
                self.assertEqual(stats[prefix + suffix], dict(zip(keys.tolist(), counts.tolist())))

    def test_core_implementations_load_only_from_compiled_extensions(self):
        package = _CORE / "st_tensor_l1"
        for module in (api, _kernels):
            with self.subTest(module=module.__name__):
                self.assertIsInstance(module.__loader__, ExtensionFileLoader)
                self.assertIn(Path(module.__file__).suffix, (".so", ".pyd"))
                with self.assertRaises((OSError, TypeError)):
                    inspect.getsource(module)
        for function in (adaptive_sttensor, adaptive_l1_trend_filter,
                         refine_core, refine_encoded_core):
            with self.subTest(function=function.__name__):
                with self.assertRaises((OSError, TypeError)):
                    inspect.getsource(function)
        leaked = [str(path.relative_to(package)) for path in package.rglob("*")
                  if path.is_file() and path.name.split(".")[0] in ("api", "_kernels")
                  and path.suffix in (".py", ".pyc", ".pyx", ".c", ".cpp")]
        self.assertEqual(leaked, [])

    def test_config_dataclass_pickle_roundtrip(self):
        for config in (SolverConfig(), production_config()):
            with self.subTest(max_iterations=config.max_iterations):
                restored = pickle.loads(pickle.dumps(config, protocol=pickle.HIGHEST_PROTOCOL))
                self.assertIs(type(restored), type(config))
                self.assertEqual(asdict(restored), asdict(config))

    def test_legacy_solver_imports_are_exact_aliases(self):
        for name in ("SolverConfig", "adaptive_sttensor", "adaptive_l1_trend_filter"):
            self.assertIs(getattr(compatibility_solver, name), getattr(_kernels, name))

    def test_explicit_production_profile_and_generic_defaults(self):
        expected = dict(min_iterations=3, max_iterations=24, training_iterations=2,
                        tolerance=0.001, beta=2.0, beta_growth=1.12, epsilon=0.0001,
                        rank_cap=8, oversample=2, alpha_update_every=5,
                        l1_min_iterations=1, l1_max_iterations=5, l1_rho=1.0,
                        data_fidelity_gain=12.0, l1_data_fidelity_gain=8.0,
                        l1_peak_relaxation=0.7, residual_phenology_mode=True)
        self.assertEqual(asdict(production_config()), expected)
        self.assertEqual(SolverConfig().max_iterations, 40)
        self.assertFalse(SolverConfig().residual_phenology_mode)
        changed = production_config()
        changed.max_iterations = 1
        self.assertEqual(production_config().max_iterations, 24)

    def test_encoded_mixed_budgets(self):
        result = self.check_encoded()
        self.assertEqual(result.st_diagnostics["budget"].tolist(), [0, 3, 8])
        self.assertEqual(result.l1_diagnostics["budget"].tolist(), [0, 1, 2])

    def test_encoded_noncontiguous_inputs(self):
        self.check_encoded(noncontiguous=True)

    def test_encoded_without_fastpaths(self):
        self.check_encoded(fastpath=False)

    def test_encoded_zero_budget_skips_factorizations(self):
        with mock.patch.object(torch.linalg, "svd", side_effect=AssertionError("Unexpected SVD")), \
             mock.patch.object(torch.linalg, "qr", side_effect=AssertionError("Unexpected QR")), \
             mock.patch.object(torch.linalg, "cholesky", side_effect=AssertionError("Unexpected factorization")):
            result = self.check_encoded(zero_budget=True)
        self.assertEqual(result.statistics()["STIterationSum"], 0)
        self.assertEqual(result.statistics()["L1IterationSum"], 0)

    def test_full_interface_residual_and_absolute_modes(self):
        for residual in (True, False):
            with self.subTest(residual=residual):
                config = replace(production_config(), residual_phenology_mode=residual)
                inputs = _fixture()
                expected = _legacy_core_sequence(_encoded(inputs, config), config)
                values, st_values = expected[:2]
                if residual:
                    reference = inputs["phenology_reference"]
                    values = reference + 2.0 * (values - 0.5)
                    st_values = reference + 2.0 * (st_values - 0.5)
                actual = refine_core(**inputs, config=config)
                self.assert_tensor_identical(actual.values, values.clamp(0, 1))
                self.assert_tensor_identical(actual.st_values, st_values.clamp(0, 1))
                self.assert_diagnostics_identical(actual.st_diagnostics, expected[2])
                self.assert_diagnostics_identical(actual.l1_diagnostics, expected[3])

    def test_all_production_allowances_keep_l1_mapping(self):
        inputs = _encoded(_fixture(), production_config())
        inputs = {key: torch.cat([value, value], dim=0) for key, value in inputs.items()}
        inputs["st_budget"] = torch.tensor([0, 3, 5, 8, 13, 24])
        expected = _legacy_core_sequence(inputs, production_config())
        result = refine_encoded_core(**inputs)
        self.assert_tensor_identical(result.values, expected[0])
        self.assert_diagnostics_identical(result.st_diagnostics, expected[2])
        self.assert_diagnostics_identical(result.l1_diagnostics, expected[3])
        self.assertEqual(result.l1_diagnostics["budget"].tolist(), [0, 1, 2, 2, 3, 5])

    def test_l1_allowance_depends_on_assigned_budget_not_st_early_stop(self):
        result = self.check_encoded(config=replace(production_config(), tolerance=10.0))
        self.assertEqual(result.st_diagnostics["iterations"].tolist(), [0, 3, 3])
        self.assertEqual(result.st_diagnostics["budget"].tolist(), [0, 3, 8])
        self.assertEqual(result.l1_diagnostics["budget"].tolist(), [0, 1, 2])

    def test_encoded_preserves_anchor_extremes(self):
        config = replace(production_config(), residual_phenology_mode=False)
        inputs = _encoded(_fixture(), config)
        result = refine_encoded_core(**inputs, config=config)
        self.assert_tensor_identical(result.st_values[inputs["anchor_mask"]],
                                     inputs["anchor_value"][inputs["anchor_mask"]])
        self.assert_tensor_identical(result.values[inputs["anchor_mask"]],
                                     inputs["anchor_value"][inputs["anchor_mask"]])

    def test_joint_interface_restores_external_gradient_mode(self):
        inputs = _encoded(_fixture(), production_config())
        with torch.enable_grad():
            self.assertTrue(torch.is_grad_enabled())
            result = refine_encoded_core(**inputs)
            self.assertFalse(result.values.requires_grad)
            self.assertTrue(torch.is_grad_enabled())

    def test_full_residual_interface_requires_reference(self):
        inputs = _fixture()
        inputs.pop("phenology_reference")
        with self.assertRaises(ValueError):
            refine_core(**inputs)

    @unittest.skipUnless(os.environ.get("QALLM_TEST_GPU") == "1" and torch.backends.mps.is_available(),
                         "Set QALLM_TEST_GPU=1 on an MPS-capable process")
    def test_mps_encoded_matches_same_backend_legacy_chain(self):
        if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "1":
            self.skipTest("MPS component comparison uses native CPU fallback")
        self.check_encoded(device="mps")


if __name__ == "__main__":
    unittest.main()
