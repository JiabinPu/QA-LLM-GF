"""Calibrate from Module03A residuals, with no second model inference."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .uncertainty_core import (SampleStore, MODES, write_csv, fit_calibration,
    evaluate_test, reliability_table, build_test_decisions, update_report_decision)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect(folder, variable, split, expected_tiles):
    report = json.loads((folder/'evaluation_manifest.json').read_text(encoding='utf-8'))
    if (report['Status'], report['Variable'], report['Split']) != ('complete', variable, split):
        raise ValueError(f'Incomplete or wrong evaluation: {folder}')
    index_file = folder/'uncertainty_residual_index.csv'
    if report.get('ResidualSchema') != 'full_chain_raw_sigma_v1' or sha(index_file) != report.get('ResidualIndexSHA256'):
        raise ValueError('Residual index missing, changed, or from an old evaluation')
    index = pd.read_csv(index_file)
    if set(index.Tile) != set(expected_tiles) or set(index.Variable) != {variable}:
        raise ValueError('Residual tile/variable inventory mismatch')
    store = SampleStore()
    for number, row in enumerate(index.itertuples(), 1):
        path = (folder/row.File).resolve()
        if folder.resolve() not in path.parents:
            raise ValueError('Residual path leaves evaluation directory')
        if sha(path) != row.SHA256:
            raise ValueError(f'Residual file changed: {path}')
        with np.load(path, allow_pickle=False) as data:
            error, sigma, weight, mode, pft = [data[name] for name in ('error', 'sigma', 'weight', 'mode', 'pft')]
            if any(a.shape != (row.N,) for a in (error, sigma, weight, mode, pft)):
                raise ValueError(f'Residual size mismatch: {path}')
            if not (np.isfinite(error).all() and np.isfinite(sigma).all() and np.isfinite(weight).all()
                    and (error >= 0).all() and (sigma > 0).all() and (weight > 0).all()
                    and np.isin(mode, MODES).all() and np.isin(pft, range(1, 9)).all()):
                raise ValueError(f'Invalid residual values: {path}')
            for m, c in set(zip(mode.tolist(), pft.tolist())):
                use = (mode == m) & (pft == c)
                store.add((variable, int(m), int(c)), (str(row.Tile), str(row.Scenario)),
                          error[use], sigma[use], weight[use])
        if number % 100 == 0 or number == len(index):
            print(f'{variable} {split} residuals: {number}/{len(index)} ({number/len(index):.1%})', flush=True)
    if not store.groups:
        raise ValueError(f'No calibration evidence: {folder}')
    return store, report


def run(args):
    root = args.root.resolve()
    output = root/'Module03B_UncertaintyCalibration'/args.variable
    contract = json.loads((root/'Module02A_TrainingInputs'/args.variable/'input_contract.json').read_text(encoding='utf-8'))
    if contract['variable'] != args.variable:
        raise ValueError('Wrong variable contract')
    args.tiles = ','.join(contract['tiles'])
    evaluation = root/'Module03A_Evaluation'/args.variable
    validation, val_report = collect(evaluation/'val', args.variable, 'val', contract['tiles'])
    rows, support = fit_calibration(args, validation)
    del validation
    output.mkdir(parents=True, exist_ok=True)
    table = output/'product_uncertainty_calibration.csv'
    write_csv(table, rows)
    write_csv(output/'validation_calibration_support.csv', support)
    fitted_hash = sha(table)
    # Fitting is complete before any test errors are read.
    test, test_report = collect(evaluation/'test', args.variable, 'test', contract['tiles'])
    if val_report['CheckpointSHA256'] != test_report['CheckpointSHA256']:
        raise ValueError('Validation and test were generated with different budget models')
    test_rows = evaluate_test(test, rows)
    reliability = reliability_table(args, test, rows)
    write_csv(output/'test_uncertainty_metrics.csv', test_rows)
    write_csv(output/'test_uncertainty_reliability_bins.csv', reliability)
    primary = [r for r in test_rows if r['Grouping'] == 'ALL' and r['Weighting'] == 'tile_scenario_equal']
    decisions = build_test_decisions(primary, args)
    report = dict(variable=args.variable, status='complete', checkpoint_sha256=val_report['CheckpointSHA256'],
                  calibration_sha256=fitted_hash, fit_split='val', verification_split='test',
                  ground_data_used=False, point_estimates_changed=False, inference_runs=0,
                  validation_manifest_sha256=sha(evaluation/'val'/'evaluation_manifest.json'),
                  test_manifest_sha256=sha(evaluation/'test'/'evaluation_manifest.json'),
                  interval_interpretation='weighted empirical residual quantiles; no distribution-free finite-sample guarantee',
                  validation_reuse='val also supports model and budget selection; calibration is not a fresh independent calibration split',
                  source_sigma='raw_full_chain_hierarchical_sigma_not_Module02C_scale_recommendation',
                  application='sigma_calibrated=raw_sigma*ScaleFactor; interval_halfwidth=calibrated_sigma*Quantile68_or_95',
                  fallback='mode_pft then mode then observed/reconstructed domain then variable; support recorded in table')
    update_report_decision(report, decisions, args)
    if sha(table) != fitted_hash:
        raise RuntimeError('Calibration table changed during test evaluation')
    (output/'calibration_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'{args.variable}: uncertainty calibration complete without model inference: {output}', flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--variable', choices=('LAI', 'FPAR'), required=True)
    p.add_argument('--lc-codes', default='1,2,3,4,5,6,7,8')
    p.add_argument('--min-stratum-pixels', type=int, default=5000)
    p.add_argument('--min-stratum-tiles', type=int, default=3)
    p.add_argument('--scale-min', type=float, default=0.05)
    p.add_argument('--scale-max', type=float, default=20.0)
    p.add_argument('--reliability-bins', type=int, default=10)
    p.add_argument('--reliability-max-points', type=int, default=500000)
    p.add_argument('--coverage68-tolerance', type=float, default=0.03)
    p.add_argument('--coverage95-tolerance', type=float, default=0.02)
    p.add_argument('--deployment-undercoverage68', type=float, default=0.03)
    p.add_argument('--deployment-undercoverage95', type=float, default=0.02)
    args = p.parse_args()
    if not 0 < args.scale_min < args.scale_max or args.min_stratum_pixels < 1 or args.min_stratum_tiles < 1:
        p.error('Invalid scale/support settings')
    if args.reliability_bins < 1 or args.reliability_max_points < 1:
        p.error('Invalid reliability settings')
    for name, maximum in [('coverage68_tolerance', .68), ('coverage95_tolerance', .95),
                          ('deployment_undercoverage68', .68), ('deployment_undercoverage95', .95)]:
        if not 0 <= getattr(args, name) < maximum:
            p.error('Invalid coverage tolerance')
    return args


if __name__ == '__main__':
    run(parse_args())
