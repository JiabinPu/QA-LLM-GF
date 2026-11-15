"""CPU-only checks for owner selection and layout-preserving tensor transfer.

Availability is injected: CUDA/MPS cases test scheduling metadata only. No
model is constructed, no accelerator is queried or used, and no data are read.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys
import unittest
from unittest import mock

for _thread_limit in (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_limit] = "1"
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from portable_runtime import select_owners, transfer_preserving_layout

torch.set_num_threads(1)


class OwnerSelectionTests(unittest.TestCase):
    NONE = {"cuda": False, "mps": False}
    MPS = {"cuda": False, "mps": True}
    CUDA = {"cuda": True, "mps": False}
    BOTH = {"cuda": True, "mps": True}

    def test_auto_and_joint_mode_use_five_cpu_and_five_gpu_owners(self):
        for mode in ("auto", "cpu_gpu"):
            for available, expected_gpu in ((self.MPS, "mps"), (self.CUDA, "cuda")):
                with self.subTest(mode=mode, gpu=expected_gpu):
                    owners, reason = select_owners(mode, 5, 5, available=available)
                    self.assertEqual(owners, ["cpu"] * 5 + [expected_gpu] * 5)
                    self.assertIsNone(reason)

    def test_auto_prefers_cuda_when_both_accelerators_are_declared(self):
        owners, reason = select_owners("auto", 2, 3, available=self.BOTH)
        self.assertEqual(owners, ["cpu", "cpu", "cuda", "cuda", "cuda"])
        self.assertIsNone(reason)

    def test_gpu_device_can_select_mps_instead_of_available_cuda(self):
        for mode in ("auto", "cpu_gpu"):
            with self.subTest(mode=mode):
                owners, reason = select_owners(mode, 2, 3, "mps", self.BOTH)
                self.assertEqual(owners, ["cpu"] * 2 + ["mps"] * 3)
                self.assertIsNone(reason)

    def test_no_gpu_keeps_requested_cpu_workers_and_records_fallback(self):
        for mode in ("auto", "cpu_gpu"):
            for available in (self.NONE, {}):
                with self.subTest(mode=mode, available=available):
                    owners, reason = select_owners(mode, 5, 5, available=available)
                    self.assertEqual(owners, ["cpu"] * 5)
                    self.assertIsInstance(reason, str)
                    self.assertTrue(reason.strip())

    def test_missing_requested_gpu_does_not_substitute_another_gpu(self):
        for mode in ("auto", "cpu_gpu"):
            with self.subTest(mode=mode):
                owners, reason = select_owners(mode, 3, 2, "cuda", self.MPS)
                self.assertEqual(owners, ["cpu"] * 3)
                self.assertIsInstance(reason, str)
                self.assertTrue(reason.strip())

    def test_no_gpu_with_zero_requested_cpu_starts_one_fallback_cpu(self):
        for mode in ("auto", "cpu_gpu"):
            with self.subTest(mode=mode):
                owners, reason = select_owners(mode, 0, 5, available=self.NONE)
                self.assertEqual(owners, ["cpu"])
                self.assertIsInstance(reason, str)
                self.assertIn("fallback", reason.lower())

    def test_zero_cpu_lane_is_disabled_when_gpu_is_available(self):
        for mode in ("auto", "cpu_gpu"):
            with self.subTest(mode=mode):
                owners, reason = select_owners(mode, 0, 4, available=self.MPS)
                self.assertEqual(owners, ["mps"] * 4)
                self.assertIsNone(reason)

    def test_zero_gpu_lane_uses_cpu_without_a_fallback_reason(self):
        for mode in ("auto", "cpu_gpu"):
            for available in (self.NONE, self.MPS):
                with self.subTest(mode=mode, available=available):
                    owners, reason = select_owners(mode, 4, 0, available=available)
                    self.assertEqual(owners, ["cpu"] * 4)
                    self.assertIsNone(reason)

    def test_cpu_mode_forces_cpu_even_if_accelerators_are_available(self):
        owners, reason = select_owners("cpu", 5, 5, available=self.BOTH)
        self.assertEqual(owners, ["cpu"] * 5)
        self.assertIsNone(reason)

    def test_cpu_mode_rejects_zero_cpu_workers(self):
        with self.assertRaises(ValueError):
            select_owners("cpu", 0, 5, available=self.BOTH)

    def test_strict_accelerator_mode_uses_only_requested_backend(self):
        for mode in ("mps", "cuda"):
            with self.subTest(mode=mode):
                owners, reason = select_owners(mode, 5, 3, available=self.BOTH)
                self.assertEqual(owners, [mode] * 3)
                self.assertIsNone(reason)

    def test_strict_accelerator_mode_never_falls_back(self):
        for mode, available in (("mps", self.CUDA), ("cuda", self.MPS)):
            with self.subTest(mode=mode):
                with self.assertRaisesRegex(RuntimeError, mode):
                    select_owners(mode, 5, 5, available=available)

    def test_strict_accelerator_mode_rejects_zero_gpu_workers(self):
        for mode in ("mps", "cuda"):
            with self.subTest(mode=mode):
                with self.assertRaises(ValueError):
                    select_owners(mode, 5, 0, available=self.BOTH)

    def test_both_lanes_zero_is_invalid_in_every_mode(self):
        for mode in ("cpu", "mps", "cuda", "auto", "cpu_gpu"):
            with self.subTest(mode=mode):
                with self.assertRaises(ValueError):
                    select_owners(mode, 0, 0, available=self.BOTH)

    def test_worker_counts_outside_zero_to_ten_are_invalid(self):
        for cpu, gpu in ((-1, 5), (5, -1), (11, 5), (5, 11)):
            with self.subTest(cpu=cpu, gpu=gpu):
                with self.assertRaises(ValueError):
                    select_owners("auto", cpu, gpu, available=self.BOTH)

    def test_ten_workers_per_lane_is_the_accepted_boundary(self):
        owners, reason = select_owners("auto", 10, 10, available=self.MPS)
        self.assertEqual(owners, ["cpu"] * 10 + ["mps"] * 10)
        self.assertIsNone(reason)

    def test_unknown_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            select_owners("unrecognized", 1, 1, available=self.BOTH)

    def test_injected_cuda_availability_does_not_query_accelerators(self):
        forbidden = AssertionError("Owner selection queried hardware despite injected availability")
        with mock.patch.object(torch.cuda, "is_available", side_effect=forbidden), \
                mock.patch.object(torch.backends.mps, "is_available", side_effect=forbidden):
            owners, reason = select_owners("auto", 5, 5, available=self.CUDA)
        self.assertEqual(owners, ["cpu"] * 5 + ["cuda"] * 5)
        self.assertIsNone(reason)


class LayoutTransferTests(unittest.TestCase):
    def assert_preserved(self, source, target_device):
        result = transfer_preserving_layout(source, torch.device(target_device))
        self.assertEqual(result.device.type, "cpu")
        self.assertEqual(result.dtype, source.dtype)
        self.assertEqual(tuple(result.shape), tuple(source.shape))
        self.assertEqual(result.stride(), source.stride())
        self.assertEqual(result.storage_offset(), source.storage_offset())
        torch.testing.assert_close(result, source, rtol=0, atol=0)
        return result

    def test_noncontiguous_offset_view_preserved_by_cpu_noop(self):
        source = torch.arange(512, dtype=torch.float32).reshape(8, 8, 8)
        source = source[1:7:2, 1:7:2, 2:6].permute(2, 0, 1).unsqueeze(0)
        self.assertFalse(source.is_contiguous())
        self.assertGreater(source.storage_offset(), 0)
        self.assert_preserved(source, "cpu")

    def test_real_cpu_copy_restores_gapped_strides_and_nonzero_offset(self):
        source = torch.arange(512, dtype=torch.float32).reshape(8, 8, 8)
        source = source[1:7:2, 1:7:2, 2:6].permute(2, 0, 1).unsqueeze(0)
        # cpu:0 requests a real CPU copy. Ordinary .to(..., preserve_format)
        # compacts this gapped view and resets its storage offset, exercising
        # the restoration path without an accelerator or a mocked tensor.
        ordinary_copy = source.to(device="cpu:0", memory_format=torch.preserve_format)
        self.assertIsNot(ordinary_copy, source)
        self.assertNotEqual((ordinary_copy.stride(), ordinary_copy.storage_offset()),
                            (source.stride(), source.storage_offset()))
        result = self.assert_preserved(source, "cpu:0")
        self.assertIsNot(result, source)

    def test_singleton_axis_strides_are_not_normalized(self):
        storage = torch.arange(64, dtype=torch.float32)
        source = torch.as_strided(storage, (1, 3, 1, 4), (777, 4, 333, 1), 7)
        self.assert_preserved(source, "cpu:0")

    def test_expanded_zero_stride_view_preserves_aliasing_and_values(self):
        source = torch.as_strided(torch.arange(64, dtype=torch.float32),
                                  (2, 3, 4), (0, 4, 1), 5)
        result = self.assert_preserved(source, "cpu:0")
        self.assertEqual(result.stride()[0], 0)
        torch.testing.assert_close(result[0], result[1], rtol=0, atol=0)

    def test_multiple_broadcast_axes_with_offset_are_preserved(self):
        source = torch.as_strided(torch.arange(32, dtype=torch.float32),
                                  (3, 2, 1), (0, 0, 21), 9)
        self.assert_preserved(source, "cpu:0")

    def test_transposed_integer_view_keeps_dtype_and_values(self):
        source = torch.arange(64, dtype=torch.int16).reshape(8, 8)[1:7:2, 2:7].T
        self.assertFalse(source.is_contiguous())
        self.assert_preserved(source, "cpu:0")

    def test_scalar_with_storage_offset_is_preserved(self):
        source = torch.arange(16, dtype=torch.float32)[7]
        self.assertEqual(source.ndim, 0)
        self.assertGreater(source.storage_offset(), 0)
        self.assert_preserved(source, "cpu:0")


class SmallCPUResourceTests(unittest.TestCase):
    def test_four_core_cpu_only_uses_two_owners(self):
        from portable_runtime import fit_cpu_budget
        self.assertEqual(fit_cpu_budget(['cpu'] * 5, 4)[0], ['cpu'] * 2)

    def test_one_core_still_runs_without_gpu(self):
        from portable_runtime import fit_cpu_budget
        self.assertEqual(fit_cpu_budget(['cpu'] * 5, 1)[0], ['cpu'])

    def test_two_owner_budget_retains_both_available_lanes(self):
        from portable_runtime import fit_cpu_budget
        self.assertEqual(fit_cpu_budget(['cpu'] * 5 + ['mps'] * 5, 4)[0], ['mps', 'cpu'])

    def test_this_mac_budget_preserves_five_plus_five(self):
        from portable_runtime import fit_cpu_budget
        owners = ['cpu'] * 5 + ['mps'] * 5
        self.assertEqual(fit_cpu_budget(owners, 15), (owners, None))


class CacheScopeTests(unittest.TestCase):
    """Structural cache identities and DN integrity, without model inference."""

    def config(self, output):
        return dict(
            output=str(output), source="unused_fixture", kind="region", tile="h25v06",
            years=[2013], coords=[(0, 0)], height=40, width=40,
            row_offset=0, col_offset=0, origin_row_1based=1, origin_col_1based=1,
            device="cpu", cpu_workers=1, gpu_workers=0, gpu_device="auto",
            threads=1, cache_mb=1, replay_map=None,
            source_identity={"old_input_hash": "not-used-for-resume"},
        )

    def patch(self, output, *, tile="h25v06", legacy=True):
        import hashlib
        import json
        import h5py
        import numpy as np
        from portable_runtime import patch_path
        from solver_diagnostics import empty_core_diagnostics
        prepared = {"vegetation": np.zeros((40, 40), dtype=bool)}
        values = (np.arange(46 * 40 * 40 * 6, dtype=np.uint32) % 65000).astype(np.uint16)
        values = values.reshape(46, 40, 40, 6)
        record = dict(year=2013, row=0, col=0, backend="cpu",
                      layer_sha256=hashlib.sha256(values.tobytes()).hexdigest(),
                      solver_core_diagnostics={v: empty_core_diagnostics(v, prepared)
                                               for v in ("LAI", "FPAR")})
        path = patch_path(output, 2013, 0, 0)
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w") as h:
            h.create_dataset("layers", data=values, compression="lzf")
            h.attrs.update(complete=1, tile=tile, year=2013, row=0, col=0,
                           record=json.dumps(record))
            if legacy:
                h.attrs["run_fingerprint"] = "obsolete-run-identity"
        return path, record, values

    def test_legacy_run_metadata_does_not_gate_resumption(self):
        import copy
        from portable_runtime import _run_scope, _check_resume_scope
        config = self.config("unused")
        old = _run_scope(config)
        old.update(schema="qa-llm-gf-portable-production-v1", code={"old.py": "hash"},
                   assets={"model": "hash"}, environment={"torch": "old-version"},
                   threads=100, source_identity={"files": "old-input"},
                   fingerprint="old", run_fingerprint="old")
        snapshot = copy.deepcopy(old)
        _check_resume_scope(old, config)
        self.assertEqual(old, snapshot)
        self.assertNotIn("fingerprint", _run_scope(config))
        self.assertNotIn("source_identity", _run_scope(config))
        self.assertNotIn("environment", _run_scope(config))
        for field, value in (("tile", "h12v03"), ("kind", "tile"),
                             ("years", [2014]), ("coords", [[0, 40]]),
                             ("dtype", "float64"), ("core_size", 4)):
            changed = copy.deepcopy(old)
            changed[field] = value
            with self.subTest(field=field), self.assertRaisesRegex(RuntimeError, "structure"):
                _check_resume_scope(changed, config)
        changed = copy.deepcopy(old)
        changed["output_geometry"]["height"] = 80
        with self.assertRaises(RuntimeError):
            _check_resume_scope(changed, config)

    def test_legacy_patch_fingerprint_is_ignored_without_mutating_the_cache(self):
        import tempfile
        from pathlib import Path
        from portable_runtime import _read_patch
        with tempfile.TemporaryDirectory() as folder:
            path, expected, _ = self.patch(Path(folder))
            old = path.read_bytes()
            self.assertEqual(_read_patch(path, (2013, 0, 0), tile="h25v06"), expected)
            self.assertEqual(path.read_bytes(), old)
            with self.assertRaises(RuntimeError):
                _read_patch(path, (2013, 0, 0), tile="h12v03")
            with self.assertRaises(RuntimeError):
                _read_patch(path, (2014, 0, 0), tile="h25v06")

    def test_patch_dn_corruption_still_fails(self):
        import tempfile
        from pathlib import Path
        import h5py
        from portable_runtime import _read_patch
        with tempfile.TemporaryDirectory() as folder:
            path, _, _ = self.patch(Path(folder))
            with h5py.File(path, "r+") as h:
                h["layers"][0, 0, 0, 0] = 123
            with self.assertRaisesRegex(RuntimeError, "checksum"):
                _read_patch(path, (2013, 0, 0), tile="h25v06")

    def test_new_worker_patch_has_no_run_fingerprint(self):
        import queue
        import tempfile
        from pathlib import Path
        import h5py
        import numpy as np
        import portable_runtime as runtime
        from solver_diagnostics import empty_core_diagnostics
        prepared = {"vegetation": np.zeros((40, 40), dtype=bool)}
        values = np.zeros((46, 40, 40, 6), dtype=np.uint16)
        stats = dict(backend="cpu", solver_core_diagnostics={
            v: empty_core_diagnostics(v, prepared) for v in ("LAI", "FPAR")})
        with tempfile.TemporaryDirectory() as folder:
            config = self.config(folder)
            tasks, results = queue.Queue(), queue.Queue()
            tasks.put([(2013, 0, 0)])
            tasks.put(None)
            with mock.patch.object(runtime, "ModelRuntime") as model, \
                    mock.patch.object(runtime, "InputPreparer") as preparer:
                model.return_value.infer.return_value = values, stats
                preparer.return_value.prepare.return_value = prepared
                runtime._worker(config, "cpu", tasks, results)
            events = []
            while not results.empty():
                events.append(results.get())
            self.assertFalse(any(event[0] == "error" for event in events), events)
            with h5py.File(runtime.patch_path(folder, 2013, 0, 0), "r") as h:
                self.assertNotIn("run_fingerprint", h.attrs)
                self.assertNotIn("fingerprint", h.attrs)
                self.assertEqual(h.attrs["tile"], "h25v06")

    def test_input_scope_does_not_hash_files_or_return_their_inventory(self):
        import json
        import tempfile
        from pathlib import Path
        import portable_runtime as runtime
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "h25v06"
            source.mkdir()
            metadata = dict(tile="h25v06", years=list(range(2001, 2025)),
                            height=200, width=200, native_patch_size=40, core_size=8,
                            origin_row_1based=1, origin_col_1based=1,
                            files={"input.h5": "obsolete-input-hash"})
            (source / "manifest.json").write_text(json.dumps(metadata))
            with mock.patch.object(runtime, "sha256", side_effect=AssertionError("Input hashing is disabled")):
                region = runtime.source_identity(source, "region")
                self.assertNotIn("files", region)
                self.assertEqual(region["tile"], "h25v06")
                compact = source / "Module01_Intermediate" / "CompactStacks" / source.name
                landcover = source / "Module00_Landcover" / "Masks"
                index = source / "Module01_Intermediate" / "PythonDataset" / "token_index_python.csv"
                index.parent.mkdir(parents=True)
                index.touch()
                landcover.mkdir(parents=True)
                for year in range(2001, 2025):
                    annual = compact / str(year)
                    annual.mkdir(parents=True)
                    for block in range(16):
                        (annual / f"fixture_{block}.h5").touch()
                    (landcover / f"LC_Type_3_vegetation_mask_h25v06_{year}.h5").touch()
                native = runtime.source_identity(source, "tile")
                self.assertEqual(native["height"], 2400)
                self.assertNotIn("files", native)
                self.assertNotIn("token_index_sha256", native)

    def test_resume_legacy_cache_without_code_model_or_environment_identity(self):
        import json
        import tempfile
        from pathlib import Path
        import h5py
        import portable_runtime as runtime
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            config = self.config(output)
            patch, _, _ = self.patch(output)
            patch_before = patch.read_bytes()
            old = runtime._run_scope(config)
            old.update(schema="qa-llm-gf-portable-production-v1", code={"changed": "code"},
                       assets={"changed": "weights"}, environment={"changed": "environment"},
                       source_identity={"changed": "input"}, run_fingerprint="legacy-value",
                       threads=200)
            manifest = output / "run_manifest.json"
            manifest.write_text(json.dumps(old))
            manifest_before = manifest.read_bytes()
            with mock.patch.object(runtime, "environment", side_effect=AssertionError("Environment identity requested")), \
                    mock.patch.object(runtime, "ModelRuntime", side_effect=AssertionError("Cached patch must not rerun")):
                summary = runtime.produce(config)
            self.assertEqual(summary["resumed_patch_count"], 1)
            self.assertNotIn("run_fingerprint", summary)
            self.assertEqual(patch.read_bytes(), patch_before)
            self.assertEqual(manifest.read_bytes(), manifest_before)
            for name in ("execution_manifest.json", "product_summary.json", "status.json"):
                value = json.loads((output / name).read_text())
                self.assertNotIn("run_fingerprint", value)
                self.assertNotIn("fingerprint", value)
            product = output / "Products" / "2013"
            seal_path = product / "annual_complete.json"
            seal = json.loads(seal_path.read_text())
            self.assertNotIn("run_fingerprint", seal)
            self.assertEqual(len(seal["files"]), 46)
            for name in seal["files"]:
                with h5py.File(product / name, "r") as h:
                    self.assertNotIn("run_fingerprint", h.attrs)
            # Inherited seals retain their bytes, even with obsolete provenance.
            seal["run_fingerprint"] = "a-different-legacy-value"
            seal.pop("tile")
            seal.pop("output_geometry")
            seal.pop("patch_origins")
            seal_path.write_text(json.dumps(seal))
            old_seal = seal_path.read_bytes()
            runtime.assemble_year(config, 2013, [(0, 0)], [])
            self.assertEqual(seal_path.read_bytes(), old_seal)
            wrong_geometry = dict(config, height=80)
            with self.assertRaisesRegex(RuntimeError, "geometry"):
                runtime.assemble_year(wrong_geometry, 2013, [(0, 0)], [])
            product_file = product / next(iter(seal["files"]))
            with h5py.File(product_file, "r+") as h:
                h["Lai"][0, 0] = int(h["Lai"][0, 0]) + 1
            with self.assertRaisesRegex(RuntimeError, "Annual product changed"):
                runtime.assemble_year(config, 2013, [(0, 0)], [])


if __name__ == "__main__":
    unittest.main()
