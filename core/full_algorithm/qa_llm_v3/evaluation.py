"""Independent-variable, single-pass full-budget-chain evaluation.

Metrics use all common finite withheld references, not the plotting sample.
No selection, fitting or checkpoint update occurs here.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from qg_dataset import DatasetOptions
from .data import WindowedQADataset
from .experts import EXPERT_NAMES
from .model import ControllerConfig, QALLMAdaptiveController
from .pipeline import (aggregate_withheld_target, prepare_window_frontend, run_window,
                       hierarchical_uncertainty_components)
from .quality import validate_checkpoint_protocol, validate_checkpoint_tile
from .solver import SolverConfig
from .utils import parse_ints, resolve_device, saturation_aware_target_weights
from .variable import assert_single_batch, variable_index
from .uncertainty_core import center_state


SCENARIOS = ('random_10', 'random_30', 'temporal_block_3', 'temporal_block_6',
             'spatial_block', 'sensor_denial_MOD', 'sensor_denial_MYD', 'sensor_denial_VNP')
BINS = {'LAI': (0, 2, 3, 4, 5, float('inf')),
        'FPAR': (0, 0.5, 0.7, 0.8, 0.9, float('inf'))}


class Moments:
    def __init__(self):
        self.n = 0
        self.s = np.zeros(9, dtype=np.float64)

    def update(self, pred, ref, weight):
        p, r, w = (np.asarray(a, dtype=np.float64) for a in (pred, ref, weight))
        use = np.isfinite(p) & np.isfinite(r) & np.isfinite(w) & (w > 0)
        p, r, w = p[use], r[use], w[use]
        self.n += len(p)
        e = p - r
        self.s += np.array([w.sum(), (w*p).sum(), (w*r).sum(), (w*p*p).sum(),
                            (w*r*r).sum(), (w*p*r).sum(), (w*e*e).sum(),
                            (w*np.abs(e)).sum(), (w*e).sum()])

    def row(self, scale):
        w, sp, sr, spp, srr, spr, se2, sea, se = self.s
        if w <= 0:
            return None
        vp, vr, cov = max(spp - sp*sp/w, 0), max(srr - sr*sr/w, 0), spr - sp*sr/w
        bias, rmse = se/w, np.sqrt(max(se2/w, 0))
        r = float(np.clip(cov / np.sqrt(vp*vr), -1, 1)) if vp > 0 and vr > 0 else np.nan
        slope = cov/vp if vp > 0 else np.nan
        return dict(N=self.n, WeightSum=w, RMSE=rmse, MAE=sea/w, Bias=bias,
                    ubRMSE=np.sqrt(max(se2/w-bias*bias, 0)), R=r, PearsonR2=r*r,
                    R2=1-se2/vr if vr > 0 else np.nan,
                    Slope_ref_on_pred=slope, Intercept_ref_on_pred=sr/w-slope*sp/w,
                    NRMSE=rmse/scale, NMAE=sea/w/scale, NBias=bias/scale)


def load_model(path, root, variable, device):
    cp = torch.load(path, map_location=device, weights_only=False)
    validate_checkpoint_protocol(cp, budget=True)
    validate_checkpoint_tile(cp, root)
    if cp['model_config'].get('active_variable') != variable:
        raise ValueError(f'Expected a {variable} budget checkpoint')
    model = QALLMAdaptiveController(ControllerConfig(**cp['model_config'])).to(device)
    result = model.load_state_dict(cp['model_state'], strict=False)
    missing = [k for k in result.missing_keys if k in dict(model.named_parameters())
               and dict(model.named_parameters())[k].requires_grad]
    if result.unexpected_keys or missing:
        raise ValueError(f'Invalid checkpoint: missing={missing}, unexpected={result.unexpected_keys}')
    model.eval()
    return model, SolverConfig(**cp['solver_config']), cp


def evaluate(args, *, dataset_root=None, dataset_factory=None):
    torch.set_num_threads(args.torch_threads)
    device = resolve_device(args.device)
    root, output = args.root.resolve(), args.output_dir.resolve()
    vi = variable_index(args.variable)
    scale = 7.0 if vi == 0 else 1.0
    model, solver, checkpoint = load_model(args.checkpoint, root, args.variable, device)
    config = checkpoint['budget_calibration_args']
    levels = checkpoint['budget_policy']['levels']
    if not levels or levels[0] != 0 or levels[-1] != solver.max_iterations:
        raise ValueError('Budget levels are inconsistent with the saved solver.')
    # Use the settings that were actually calibrated, not a fresh default threshold.
    route_options = dict(core_size=int(config['core_size']), route_mode='learned_budget',
                         budget_levels=levels, structural_cell_blend=True,
                         route_max_selected_fraction=float(config['policy_max_selected_fraction']))
    dataset = (dataset_factory or WindowedQADataset)(
        Path(dataset_root).resolve() if dataset_root is not None else root,
        args.split, SCENARIOS, int(config['window_years']),
        parse_ints(config['sensor_start_years']),
        DatasetOptions(active_variable=args.variable, return_meta=False, use_lc_context=True,
                       lc_codes=parse_ints(config['lc_codes']), h5_cache_size=8, token_cache_size=8),
        max_samples_per_scenario=0, optimize_io_order=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    statistics = defaultdict(Moments)
    windows, coverage, uncertainty_rows, plotting, residual_index = [], [], [], [], []
    histogram = defaultdict(lambda: np.zeros((100, 100), dtype=np.int64))
    edges = np.linspace(0, scale, 101)
    rng = np.random.default_rng(20260720)
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    (output / 'uncertainty_residuals').mkdir(exist_ok=True)
    summary = dict(Status='running', Variable=args.variable, Split=args.split,
                   Reference='synthetic satellite reference', CheckpointSHA256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                   RouteOptions=route_options, Ablations=bool(args.ablations))
    (output / 'evaluation_manifest.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    batches = 0
    try:
        for index, batch in enumerate(loader, 1):
            if args.max_batches and index > args.max_batches:
                break
            assert_single_batch(batch, args.variable)
            x, centers = batch['x_window'].to(device), batch['center_index'].to(device)
            mask_sensor = batch['target_mask'].to(device)
            weights = saturation_aware_target_weights(
                x, centers, mask_sensor, batch['target_weight'].to(device),
                float(config['saturated_target_weight_floor']), batch['target_saturated'].to(device))
            reference, mask, weight = aggregate_withheld_target(batch['target'].to(device), mask_sensor, weights)
            with torch.inference_mode():
                frontend = prepare_window_frontend(model, x)
                result = run_window(model, x, centers, solver, prepared_frontend=frontend, **route_options)
                predictions = {('IDW' if name == 'QA_IDW' else name): result['experts'][:, vi, ei]
                               for ei, name in enumerate(EXPERT_NAMES)}
                predictions['PHENO_PEAK'] = result['phenology_peak'][:, vi]
                predictions['QA_LLM_GF'] = result['prediction'][:, vi]
                if args.ablations:
                    for name, expert_mask in [('TWO_EXPERT_INFERENCE_ABLATION', [0, 1, 1, 0]),
                                              ('THREE_EXPERT_INFERENCE_ABLATION', [1, 1, 1, 0])]:
                        ablation = run_window(model, x, centers, solver, expert_mask=expert_mask,
                                              prepared_frontend=frontend, **route_options)
                        predictions[name] = ablation['prediction'][:, vi]
                sigma = hierarchical_uncertainty_components(result)['total'][:, vi].cpu().numpy()
                modes, _, _ = center_state(x, centers, result, parse_ints(config['lc_codes']))
                modes = modes[:, vi].cpu().numpy().astype(np.uint8)
            pred = {k: v.detach().cpu().numpy() for k, v in predictions.items()}
            ref, valid, weights_np = reference[:, vi].cpu().numpy(), mask[:, vi].cpu().numpy(), weight[:, vi].cpu().numpy()
            visible = result['observation_weight'][:, vi].detach().cpu().numpy() > 0
            sat = (batch['target_saturated'][:, vi] & batch['target_mask'][:, vi]).any(dim=-1).numpy()
            unsat = (~batch['target_saturated'][:, vi] & batch['target_mask'][:, vi]).any(dim=-1).numpy()
            for bi, scenario in enumerate(batch['scenario']):
                tile = str(batch['tile'][bi])
                identity = dict(Tile=tile, Scenario=str(scenario), Variable=args.variable,
                                Year=int(batch['year'][bi]), TokenRow=int(batch['token_row'][bi]))
                eligible = valid[bi] & np.isfinite(ref[bi]) & (weights_np[bi] > 0)
                common = eligible.copy()
                for name, values in pred.items():
                    finite = np.isfinite(values[bi])
                    coverage.append(dict(**identity, Method=name, ReferenceN=int(eligible.sum()),
                                         FiniteN=int((eligible & finite).sum())))
                    common &= finite
                domains = {'complete_gap': ~visible[bi], 'partial_sensor': visible[bi],
                           'saturated_only': sat[bi] & ~unsat[bi], 'unsaturated_only': unsat[bi] & ~sat[bi],
                           'mixed': sat[bi] & unsat[bi]}
                lc_scores = x[bi, int(centers[bi]), 19:].mean(dim=(1, 4))
                codes = np.asarray(parse_ints(config['lc_codes']))
                pft = codes[lc_scores.argmax(dim=0).cpu().numpy()]
                pft[lc_scores.max(dim=0).values.cpu().numpy() <= 0] = 0
                for code in codes:
                    domains[f'PFT_{code}'] = np.broadcast_to(pft == code, common.shape)
                for lower, upper in zip(BINS[args.variable][:-1], BINS[args.variable][1:]):
                    domains[f'reference_{lower:g}_{upper:g}'] = (ref[bi] >= lower) & (ref[bi] < upper)
                groups = [('ALL', 'ALL', 'all', common), (tile, 'ALL', 'all', common),
                          ('ALL', str(scenario), 'all', common), (tile, str(scenario), 'all', common)]
                for domain, domain_mask in domains.items():
                    groups += [('ALL', 'ALL', domain, common & domain_mask),
                               (tile, 'ALL', domain, common & domain_mask),
                               ('ALL', str(scenario), domain, common & domain_mask)]
                positions = np.flatnonzero(common)
                selected = rng.choice(positions, min(16, len(positions)), replace=False)
                for name, values in pred.items():
                    for tile_group, scenario_group, domain, use in groups:
                        if not use.any():
                            continue
                        p, r = values[bi][use], ref[bi][use]
                        for weighting, w in [('sample_equal', np.ones(len(p))), ('qa_weighted', weights_np[bi][use])]:
                            statistics[(tile_group, scenario_group, domain, name, weighting)].update(p, r, w)
                    moment = Moments()
                    moment.update(values[bi][common], ref[bi][common], np.ones(int(common.sum())))
                    row = moment.row(scale)
                    if row is not None:
                        windows.append(dict(**identity, Method=name, **row))
                    histogram[(tile, name)] += np.histogram2d(values[bi][common], ref[bi][common], bins=(edges, edges))[0].astype(np.int64)
                    for position in selected:
                        plotting.append(dict(**identity, Method=name, FlatIndex=int(position),
                                             Estimate=float(values[bi].flat[position]), Reference=float(ref[bi].flat[position])))
                use_sigma = common & np.isfinite(sigma[bi]) & (sigma[bi] > 0)
                error = np.abs(pred['QA_LLM_GF'][bi] - ref[bi])
                uncertainty_rows.append(dict(**identity, N=int(use_sigma.sum()),
                    Covered68=int((use_sigma & (error <= sigma[bi])).sum()),
                    Covered95=int((use_sigma & (error <= 1.96*sigma[bi])).sum()),
                    SigmaSum=float(sigma[bi][use_sigma].sum()),
                    Note='raw_full_chain_sigma_not_final_product_calibration'))
                # Calibration uses final-method availability, not competitor availability.
                residual_use = (eligible & np.isfinite(pred['QA_LLM_GF'][bi]) & np.isfinite(sigma[bi])
                                & (sigma[bi] > 0) & np.broadcast_to(pft > 0, eligible.shape))
                residual_file = f'uncertainty_residuals/window_{len(residual_index):06d}.npz'
                np.savez_compressed(output/residual_file,
                    error=error[residual_use].astype(np.float32), sigma=sigma[bi][residual_use].astype(np.float32),
                    weight=weights_np[bi][residual_use].astype(np.float32),
                    mode=modes[bi][residual_use],
                    pft=np.broadcast_to(pft, eligible.shape)[residual_use].astype(np.uint8))
                residual_index.append(dict(**identity, File=residual_file, N=int(residual_use.sum()),
                    SHA256=hashlib.sha256((output/residual_file).read_bytes()).hexdigest()))
            batches += 1
            if index % args.print_every == 0 or index == len(loader):
                total = min(len(loader), args.max_batches or len(loader))
                print(f'{args.variable} {args.split}: {index}/{total} ({index/max(total,1):.1%}), '
                      f'elapsed={time.perf_counter()-started:.1f}s', flush=True)
    finally:
        dataset.close()
    if not windows:
        raise RuntimeError('No common valid evaluation targets. No successful report produced.')
    rows = [dict(Tile=k[0], Scenario=k[1], Domain=k[2], Method=k[3], Weighting=k[4],
                 Variable=args.variable, Split=args.split, **v.row(scale)) for k, v in sorted(statistics.items())]
    pd.DataFrame(rows).to_csv(output / 'metrics.csv', index=False)
    pd.DataFrame(residual_index).to_csv(output/'uncertainty_residual_index.csv', index=False)
    for name, data in [('window_metrics', windows), ('coverage', coverage),
                       ('uncertainty_counts', uncertainty_rows), ('plot_sample', plotting)]:
        pd.DataFrame(data).to_csv(output / (name + '.csv'), index=False)
    np.savez_compressed(output / 'scatter_histograms.npz', edges=edges,
                        **{tile+'__'+method: h for (tile, method), h in histogram.items()})
    summary.update(Status='partial' if args.max_batches and batches < len(loader) else 'complete',
                   ResidualSchema='full_chain_raw_sigma_v1',
                   ResidualIndexSHA256=hashlib.sha256((output/'uncertainty_residual_index.csv').read_bytes()).hexdigest(),
                   Batches=batches, TotalBatches=len(loader), ElapsedSec=time.perf_counter()-started,
                   MetricsBasis='all common finite withheld references; sample_equal and qa_weighted reported separately',
                   R2Definition='1-SSE/SST(reference); PearsonR2 is separately named',
                   PlotSample='16 random common pixels per window; NOT a uniform pooled sample; NOT used for metrics',
                   HistogramRange=[0, scale], HistogramNote='outside-range values excluded from histogram, retained in metrics',
                   AblationNote='inference expert restriction only; not independently retrained two/three-expert models')
    (output / 'evaluation_manifest.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(f'Full-chain evaluation {summary["Status"]}: {output}', flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--variable', choices=('LAI', 'FPAR'), required=True)
    p.add_argument('--split', choices=('val', 'test'), default='test')
    p.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    p.add_argument('--torch-threads', type=int, default=4)
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--max-batches', type=int, default=0)
    p.add_argument('--print-every', type=int, default=5)
    p.add_argument('--ablations', action='store_true')
    args = p.parse_args()
    if args.batch_size < 1 or args.max_batches < 0 or args.print_every < 1 or args.torch_threads < 1:
        p.error('Invalid batch/thread/progress settings')
    return args


if __name__ == '__main__':
    evaluate(parse_args())
