"""Small actual-diagnostic fixtures only; no model, GPU or log directory."""
from __future__ import annotations

import copy
import csv
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from solver_diagnostics import (aggregate, capture_core_diagnostics,
                                empty_core_diagnostics, validate_core_diagnostics,
                                write_solver_statistics)


def diagnostic(variable, valid, *, selected=(), st=None, l1=None,
               st_budget=None, l1_budget=None, controller_budget=None):
    prepared = {'vegetation': np.zeros((40, 40), dtype=bool)}
    for core in valid:
        prepared['vegetation'][8 * (core // 5), 8 * (core % 5)] = True
    if not valid:
        return empty_core_diagnostics(variable, prepared)
    offset = 0 if variable == 'LAI' else 25
    route = np.zeros(50, dtype=bool)
    needs = np.zeros(50, dtype=bool)
    for core in selected:
        route[offset + core] = needs[offset + core] = True
    counters = {}
    for prefix, iterations, budgets in (('st', st or {}, st_budget or {}), ('l1', l1 or {}, l1_budget or {})):
        applied = np.zeros(50, dtype=np.int64)
        actual = np.zeros(50, dtype=np.int64)
        predicted = np.zeros(50, dtype=np.int64)
        for core in selected:
            applied[offset + core] = budgets.get(core, 3 if prefix == 'st' else 1)
            predicted[offset + core] = (controller_budget or {}).get(core, 3)
        for core, count in iterations.items():
            actual[offset + core] = count
        counters[prefix + '_diagnostics'] = dict(routed=route, needs_solver=needs,
                                               budget=applied, iterations=actual, llm_budget=predicted)
    return capture_core_diagnostics(variable, prepared, counters)


def record(year=2013, row=0, col=0, *, valid=(0,), backend='cpu', lai=None, fpar=None):
    return dict(year=year, row=row, col=col, backend=backend,
                solver_core_diagnostics={'LAI': lai or diagnostic('LAI', valid),
                                         'FPAR': fpar or diagnostic('FPAR', valid)})


def dictionaries(table):
    return [dict(zip(table['headers'], values)) for values in table['rows']]


class SolverDiagnosticsTests(unittest.TestCase):
    def test_selection_positive_budget_and_actual_start_are_distinct(self):
        lai = diagnostic('LAI', (0, 1), selected=(0, 1), st={0: 2}, l1={0: 1})
        self.assertEqual(lai['SelectedCoreYears'], 2)
        self.assertEqual(lai['STStartedCoreYears'], 1)
        self.assertEqual(lai['L1StartedCoreYears'], 1)
        self.assertEqual(lai['EitherStartedCoreYears'], 1)
        self.assertEqual(lai['Distributions']['STAppliedBudget']['3'], 2)
        self.assertEqual(lai['Distributions']['STActualIterations']['0'], 1)
        rows = aggregate([record(valid=(0, 1), lai=lai)], [2013])['metadata']['total_by_variable']
        self.assertEqual(rows['LAI']['STActivationPct'], 50)
        self.assertEqual(rows['LAI']['SelectionPct'], 100)
        self.assertEqual(rows['FPAR']['STStartedCoreYears'], 0)

    def test_truncated_budget_is_not_replaced_by_a_controller_band(self):
        diag = diagnostic('LAI', (0,), selected=(0,), st={0: 2},
                          st_budget={0: 7}, controller_budget={0: 13})
        self.assertEqual(diag['Distributions']['ControllerSTBudget']['13'], 1)
        self.assertEqual(diag['Distributions']['STAppliedBudget']['7'], 1)
        self.assertEqual(diag['Distributions']['STActualIterations']['2'], 1)

    def test_cuda_backend_is_metadata_only_not_a_claim_of_gpu_execution(self):
        item = record(backend='cuda', lai=diagnostic('LAI', (0,), selected=(0,), st={0: 1}))
        package = aggregate([item], [2013])
        self.assertEqual(package['metadata']['total_by_variable']['LAI']['STStartedCoreYears'], 1)
        inconsistent = copy.deepcopy(item)
        inconsistent['assigned_backend'] = 'mps'
        with self.assertRaises(ValueError):
            aggregate([inconsistent], [2013])
        retry_on_cpu = copy.deepcopy(item)
        retry_on_cpu['backend'] = 'cpu'
        with self.assertRaisesRegex(ValueError, 'conflict'):
            aggregate([item, retry_on_cpu], [2013])

    def test_identical_retry_deduplicates_but_conflicts_fail(self):
        item = record(lai=diagnostic('LAI', (0,), selected=(0,), st={0: 1}))
        retry = copy.deepcopy(item)
        retry['seconds'] = 999  # Timing cannot change scientific deduplication.
        package = aggregate([item, retry], [2013])
        self.assertEqual(package['metadata']['covered_patch_count'], 1)
        self.assertEqual(package['metadata']['duplicate_patch_records_ignored'], 1)
        self.assertEqual(package['metadata']['duplicate_records_ignored'], 2)
        self.assertEqual(package['metadata']['total_by_variable']['LAI']['STStartedCoreYears'], 1)
        changed_backend = copy.deepcopy(item)
        changed_backend['backend'] = 'mps'
        with self.assertRaisesRegex(ValueError, 'conflict'):
            aggregate([item, changed_backend], [2013])
        changed_observation = record(lai=diagnostic('LAI', (0,), selected=(0,), st={0: 2}))
        with self.assertRaisesRegex(ValueError, 'conflict'):
            aggregate([item, changed_observation], [2013])

    def test_no_vegetation_has_no_imputed_controller_prediction(self):
        item = record(valid=())
        for variable in ('LAI', 'FPAR'):
            diag = item['solver_core_diagnostics'][variable]
            self.assertTrue(diag['InferenceSkipped'])
            self.assertEqual(diag['CoreData']['ControllerSTBudget'], [None] * 25)
            self.assertEqual(sum(diag['Distributions']['ControllerSTBudget'].values()), 0)
        result = aggregate([item], [2013])
        for value in result['metadata']['total_by_variable'].values():
            self.assertEqual(value['ValidCoreYears'], 0)
            self.assertIsNone(value['STActivationPct'])
        with tempfile.TemporaryDirectory() as folder:
            written = write_solver_statistics([item], folder, [2013])
            self.assertEqual(set(written['file_sha256']), {'solver_by_year_variable.csv',
                             'solver_total_by_variable.csv', 'solver_budget_distribution.csv',
                             'solver_statistics_notes.txt'})
            with (Path(folder) / 'solver_total_by_variable.csv').open(newline='') as stream:
                for row in csv.DictReader(stream):
                    self.assertEqual(row['STActivationPct'], '')
            # Identical statistics are idempotent, while a changed result must
            # not silently replace already published summaries.
            self.assertEqual(written, write_solver_statistics([item], folder, [2013]))
            with self.assertRaisesRegex(RuntimeError, 'differ'):
                write_solver_statistics([record()], folder, [2013])

    def test_total_percentage_uses_summed_core_denominators(self):
        active = record(year=2013, valid=(0,), lai=diagnostic('LAI', (0,), selected=(0,), st={0: 1}))
        inactive = record(year=2014, valid=tuple(range(9)))
        package = aggregate([active, inactive], [2014, 2013], tile='h25v06', run_fingerprint='fixture')
        annual = dictionaries(package['tables']['solver_by_year_variable.csv'])
        self.assertEqual(len(annual), 4)
        lai_annual = [row for row in annual if row['Variable'] == 'LAI']
        self.assertEqual([row['STActivationPct'] for row in lai_annual], [100.0, 0.0])
        totals = dictionaries(package['tables']['solver_total_by_variable.csv'])
        self.assertEqual(len(totals), 2)
        self.assertEqual(totals[0]['ValidCoreYears'], 10)
        self.assertEqual(totals[0]['STActivationPct'], 10.0)
        self.assertNotEqual(totals[0]['STActivationPct'], 50.0)
        rows = dictionaries(package['tables']['solver_budget_distribution.csv'])
        zero = [row for row in rows if row['Scope'] == 'TOTAL' and row['Variable'] == 'LAI'
                and row['Distribution'] == 'STActualIterations' and row['Value'] == 0]
        self.assertEqual(zero[0]['CoreYears'], 9)

    def test_corrupt_counters_missing_variable_and_wrong_scope_reject(self):
        corrupt = diagnostic('LAI', (0,), selected=(0,), st={0: 1})
        corrupt['STStartedCoreYears'] = 0
        with self.assertRaises(ValueError):
            validate_core_diagnostics(corrupt)
        missing = record()
        del missing['solver_core_diagnostics']['FPAR']
        with self.assertRaises(ValueError):
            aggregate([missing], [2013])
        with self.assertRaises(ValueError):
            aggregate([record(row=1)], [2013])
        with self.assertRaises(ValueError):
            aggregate([record(year=2014)], [2013])
        inconsistent = record()
        inconsistent['assigned_backend'] = 'mps'
        with self.assertRaises(ValueError):
            aggregate([inconsistent], [2013])

    def test_legacy_fingerprint_values_do_not_gate_statistics_or_deduplication(self):
        left = record()
        left['run_fingerprint'] = 'old-run-a'
        right = copy.deepcopy(left)
        right['run_fingerprint'] = 'old-run-b'
        package = aggregate([left, right], [2013], run_fingerprint='ignored-caller-value')
        self.assertEqual(package['metadata']['covered_patch_count'], 1)
        self.assertEqual(package['metadata']['duplicate_patch_records_ignored'], 1)
        self.assertNotIn('run_fingerprint', package['metadata'])
        for table in package['tables'].values():
            self.assertNotIn('RunFingerprint', table['headers'])
            self.assertNotIn('fingerprint', table['headers'])

    def test_new_statistics_have_no_run_fingerprint_fields(self):
        with tempfile.TemporaryDirectory() as folder:
            written = write_solver_statistics([record()], folder, [2013],
                                              tile='h25v06', run_fingerprint='legacy-argument')
            self.assertNotIn('run_fingerprint', written)
            for name in written['file_sha256']:
                text = (Path(folder) / name).read_text()
                self.assertNotIn('RunFingerprint', text)
                self.assertNotIn('Run fingerprint:', text)
                self.assertNotIn('legacy-argument', text)

    def test_legacy_statistics_compare_without_fingerprint_and_keep_their_bytes(self):
        import hashlib
        import io
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            first = write_solver_statistics([record()], output, [2013], tile='h25v06')
            for name in first['file_sha256']:
                path = output / name
                if path.suffix == '.csv':
                    rows = list(csv.reader(io.StringIO(path.read_text())))
                    rows[0].insert(1, 'RunFingerprint')
                    for row in rows[1:]:
                        row.insert(1, 'unrelated-old-fingerprint')
                    with path.open('w', newline='') as stream:
                        csv.writer(stream).writerows(rows)
                else:
                    path.write_text(path.read_text() + 'Run fingerprint: unrelated-old-fingerprint\n')
            old_bytes = {name: (output / name).read_bytes() for name in first['file_sha256']}
            second = write_solver_statistics([record()], output, [2013], tile='h25v06',
                                             run_fingerprint='new-ignored-value')
            for name, content in old_bytes.items():
                self.assertEqual((output / name).read_bytes(), content)
                self.assertEqual(second['file_sha256'][name], hashlib.sha256(content).hexdigest())
            # Scientific counters still protect against conflicting summaries.
            path = output / 'solver_by_year_variable.csv'
            rows = list(csv.reader(io.StringIO(path.read_text())))
            rows[1][rows[0].index('ValidCoreYears')] = '999'
            with path.open('w', newline='') as stream:
                csv.writer(stream).writerows(rows)
            with self.assertRaisesRegex(RuntimeError, 'differ'):
                write_solver_statistics([record()], output, [2013], tile='h25v06')


if __name__ == '__main__':
    unittest.main()
