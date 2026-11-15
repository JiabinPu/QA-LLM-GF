"""Positive-budget component execution, separate from frozen-model coverage.

The four h27v06 real-data smoke cases currently choose zero solver budgets.
These deterministic synthetic fixtures exercise one native 8x8 core with
positive ST/L1 budgets; they do not establish that a real-data model routes
to either solver. No learned budget, threshold, model or product is changed.

CPU always runs, using one numerical thread. Opt in to actual Apple MPS with
QALLM_TEST_GPU=1; absence of MPS is an ordinary unittest skip. Native Torch
operators are observed, never replaced by CPU implementations. MPS fallback
warnings are printed as evidence when the installed Torch emits them; an
operator supported natively in a future Torch need not emit such a warning.
This is component executability coverage, not CPU/GPU bitwise equivalence.
"""
from __future__ import annotations

from contextlib import ExitStack
import json
import os
from pathlib import Path
import sys
import unittest
from unittest import mock
import warnings

for _thread_name in (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_name] = "1"
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
sys.dont_write_bytecode = True
_PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PACKAGE / "core" / "full_algorithm"))

import torch

from qa_llm_v3 import solver

torch.set_num_threads(1)
_GPU_REQUESTED = os.environ.get("QALLM_TEST_GPU") == "1"
_MPS_AVAILABLE = _GPU_REQUESTED and torch.backends.mps.is_available()
_MPS_SKIP = "Set QALLM_TEST_GPU=1 and provide an available Apple MPS device"


def _st_fixture(device):
    """Normalized observations and a distinct prior, one 64x46x5 core."""
    pixel = torch.arange(64, dtype=torch.float32).view(1, 64, 1, 1)
    time = torch.arange(46, dtype=torch.float32).view(1, 1, 46, 1)
    year = torch.arange(5, dtype=torch.float32).view(1, 1, 1, 5)
    phase = time * 0.19 + pixel * 0.07 + year * 0.11
    prior = (0.45 + 0.22 * torch.sin(phase)).contiguous()
    observed = (0.48 + 0.29 * torch.sin(phase + 0.35)).clamp(0, 1)
    weight = (0.15 + 0.65 * ((pixel + time + year) % 4 == 0)).expand_as(prior).clone()
    anchor_mask = torch.zeros_like(prior, dtype=torch.bool)
    anchor_value = observed.clone()
    for index, value in (((0, 0, 0, 0), 0.0), ((0, 31, 23, 2), 1.0),
                         ((0, 63, 45, 4), 0.375)):
        anchor_mask[index] = True
        anchor_value[index] = value
    values = dict(observed=observed, observation_weight=weight, prior=prior,
                  prior_strength=torch.tensor([0.8]), anchor_value=anchor_value,
                  anchor_mask=anchor_mask, difficulty=torch.tensor([0.6]),
                  iteration_budget=torch.tensor([3], dtype=torch.long))
    return {name: value.to(device) for name, value in values.items()}


def _l1_fixture(device):
    """Normalized 46-date signal with curvature and three strict anchors."""
    pixel = torch.arange(64, dtype=torch.float32).view(1, 64, 1)
    time = torch.arange(46, dtype=torch.float32).view(1, 1, 46)
    phase = time * 0.2 + pixel * 0.03
    signal = (0.48 + 0.24 * torch.sin(phase)
              + 0.08 * ((time % 5) == 0)).clamp(0, 1)
    observed = (0.45 + 0.28 * torch.sin(phase + 0.25)).clamp(0, 1)
    weight = (0.1 + 0.7 * ((pixel + time) % 3 == 0)).expand_as(signal).clone()
    anchor_mask = torch.zeros_like(signal, dtype=torch.bool)
    anchor_value = observed.clone()
    for index, value in (((0, 0, 0), 0.0), ((0, 31, 23), 1.0), ((0, 63, 45), 0.375)):
        anchor_mask[index] = True
        anchor_value[index] = value
    values = dict(signal=signal, observed=observed, observation_weight=weight,
                  anchor_value=anchor_value, anchor_mask=anchor_mask,
                  l1_lambda=torch.tensor([0.025]), difficulty=torch.tensor([0.6]),
                  iteration_budget=torch.tensor([2], dtype=torch.long))
    return {name: value.to(device) for name, value in values.items()}


class SolverExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if Path(solver.__file__).resolve() != _PACKAGE / "core/full_algorithm/qa_llm_v3/solver.py":
            raise RuntimeError("Component tests must load this release's solver")

    def _execute(self, kind, device):
        self.assertEqual(torch.get_num_threads(), 1)
        if device == "mps":
            self.assertEqual(os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"), "1",
                             "MPS component execution requires the release's native fallback profile")
        fixture = _st_fixture(device) if kind == "ST" else _l1_fixture(device)
        source = fixture["prior" if kind == "ST" else "signal"]
        self.assertEqual(tuple(source.shape), (1, 64, 46, 5) if kind == "ST" else (1, 64, 46))
        operation_names = ("qr", "svd") if kind == "ST" else ("cholesky", "cholesky_solve")
        operations = {}
        observed_devices = {name: [] for name in operation_names}

        def observe(name, native):
            def wrapped(*args, **kwargs):
                observed_devices[name].append(args[0].device.type)
                return native(*args, **kwargs)
            return wrapped

        with ExitStack() as stack, warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for name in operation_names:
                owner = torch if name == "cholesky_solve" else torch.linalg
                native = getattr(owner, name)
                operations[name] = stack.enter_context(mock.patch.object(
                    owner, name, side_effect=observe(name, native)))
            with torch.inference_mode():
                function = solver.adaptive_sttensor if kind == "ST" else solver.adaptive_l1_trend_filter
                output, diagnostic = function(**fixture, config=solver.SolverConfig(),
                                              training=False, execution_fastpath=False)
            if device == "mps":
                torch.mps.synchronize()

        # These counters come from actual positive-budget solver execution.
        actual = diagnostic["iterations"].detach().cpu()
        budget = diagnostic["budget"].detach().cpu()
        self.assertTrue(bool((actual > 0).all()))
        self.assertTrue(bool((actual <= budget).all()))
        self.assertTrue(torch.equal(budget, fixture["iteration_budget"].cpu()))
        self.assertEqual(output.device.type, device)
        self.assertEqual(output.dtype, torch.float32)
        self.assertEqual(output.shape, source.shape)
        self.assertFalse(output.requires_grad)
        value = output.detach().cpu()
        mask = fixture["anchor_mask"].cpu()
        self.assertTrue(bool(torch.isfinite(value).all()))
        self.assertTrue(bool(((value >= 0) & (value <= 1)).all()))
        self.assertTrue(torch.equal(value[mask], fixture["anchor_value"].cpu()[mask]))
        self.assertTrue(bool((value[~mask] != source.cpu()[~mask]).any()),
                        "A positive-budget test must do more than return the initial prior")
        for name in operation_names:
            self.assertGreater(operations[name].call_count, 0, name)
            self.assertEqual(set(observed_devices[name]), {device}, name)
        if kind == "ST":
            # Two projected modes require QR; the five-year mode uses direct SVD.
            self.assertEqual(operations["qr"].call_count, 2 * int(actual.sum()))
            self.assertEqual(operations["svd"].call_count, 3 * int(actual.sum()))
            self.assertTrue(bool(torch.isfinite(diagnostic["relative_change"].cpu()).all()))
            self.assertTrue(bool(torch.isfinite(diagnostic["alpha"].cpu()).all()))
        else:
            self.assertEqual(operations["cholesky"].call_count, 1)
            self.assertEqual(operations["cholesky_solve"].call_count, int(actual.sum()))
        if device == "mps":
            # Native unsupported-op fallback moves operands internally; the
            # wrappers only establish the requested device, not kernel residency.
            print(json.dumps(dict(component=kind, device=device, shape=list(output.shape),
                                  iterations=actual.tolist(), budget=budget.tolist(),
                                  native_operator_calls={name: op.call_count for name, op in operations.items()},
                                  native_fallback_enabled=True,
                                  native_warnings=[str(w.message) for w in caught]), sort_keys=True), flush=True)

    def test_cpu_st_positive_budget_executes_qr_svd(self):
        self._execute("ST", "cpu")

    def test_cpu_l1_positive_budget_executes_cholesky_solve(self):
        self._execute("L1", "cpu")

    @unittest.skipUnless(_MPS_AVAILABLE, _MPS_SKIP)
    def test_mps_st_positive_budget_native_fallback_enabled(self):
        self._execute("ST", "mps")

    @unittest.skipUnless(_MPS_AVAILABLE, _MPS_SKIP)
    def test_mps_l1_positive_budget_native_fallback_enabled(self):
        self._execute("L1", "mps")


if __name__ == "__main__":
    unittest.main()
