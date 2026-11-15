"""Actual per-core solver diagnostics and post-inference CSV summaries.

The native-core capture/validation functions below are copied unchanged from
the local production implementation. No model, solver, LocalProduction import,
timing table, training, or product mutation is performed by this module.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
from pathlib import Path
import uuid

_CORE_DIAGNOSTICS_SCHEMA = 'qa-llm-gf-core-solver-diagnostics-v1'
_CORE_ORDER = 'row-major; row_offset=8*(index//5); col_offset=8*(index%5)'
_CONTROLLER_LEVELS = (0, 3, 5, 8, 13, 24)


def _diagnostic_array(value):
    """Copy a diagnostic view to CPU without changing any solver tensor."""
    import numpy as np
    if hasattr(value, 'detach') and hasattr(value, 'cpu'):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _diagnostic_valid_cores(variable, prepared):
    import numpy as np
    if variable not in ('LAI', 'FPAR'):
        raise ValueError('A separate LAI or FPAR diagnostic is required')
    mask = _diagnostic_array(prepared['vegetation'])
    if mask.shape != (40, 40) or mask.dtype != np.dtype('bool'):
        raise ValueError('Diagnostic vegetation must be a boolean 40x40 native patch')
    return mask.reshape(5, 8, 5, 8).any(axis=(1, 3)).reshape(25).tolist()


def _diagnostic_slot(variable, value, label, *, boolean=False, maximum=None, levels=None):
    import numpy as np
    array = _diagnostic_array(value)
    if array.shape not in {(50,), (1, 2, 25)}:
        raise ValueError(f'{label} must have shape (50,) or (1,2,25)')
    if boolean:
        if array.dtype != np.dtype('bool'):
            raise ValueError(f'{label} must contain genuine boolean diagnostics')
    else:
        if not np.issubdtype(array.dtype, np.integer):
            raise ValueError(f'{label} must contain integer diagnostics, never floats/NaN/Inf')
        if np.any(array < 0) or (maximum is not None and np.any(array > maximum)):
            raise ValueError(f'{label} is outside its declared diagnostic bounds')
        if levels is not None and not np.isin(array, levels).all():
            raise ValueError(f'{label} is outside the frozen controller budget levels')
    slot = 0 if variable == 'LAI' else 1
    return array.reshape(1, 2, 25)[0, slot].tolist()


def _diagnostic_histogram(values, valid, bins):
    counts = {str(value): 0 for value in bins}
    for value, keep in zip(values, valid):
        if keep:
            counts[str(value)] += 1
    return counts


def _diagnostics_from_core_data(variable, core, skipped):
    valid, selected = core['Valid'], core['Selected']
    st_iterations, l1_iterations = core['STIterations'], core['L1Iterations']
    st_started = [value > 0 for value in st_iterations]
    l1_started = [value > 0 for value in l1_iterations]
    count = lambda values: sum(bool(value) and keep for value, keep in zip(values, valid))
    result = dict(Variable=variable, Schema=_CORE_DIAGNOSTICS_SCHEMA,
                  CoreOrder=_CORE_ORDER, CoreSize=8, InferenceSkipped=skipped,
                  ValidCoreYears=sum(valid), SelectedCoreYears=count(selected),
                  STStartedCoreYears=count(st_started), L1StartedCoreYears=count(l1_started),
                  EitherStartedCoreYears=count([a or b for a, b in zip(st_started, l1_started)]),
                  STIterationSum=sum(value for value, keep in zip(st_iterations, valid) if keep),
                  L1IterationSum=sum(value for value, keep in zip(l1_iterations, valid) if keep),
                  CoreData=core)
    result['Distributions'] = {
        'ControllerSTBudget': _diagnostic_histogram(core['ControllerSTBudget'], valid, _CONTROLLER_LEVELS),
        'STAppliedBudget': _diagnostic_histogram(core['STBudget'], valid, range(25)),
        'L1AppliedBudget': _diagnostic_histogram(core['L1Budget'], valid, range(6)),
        'STActualIterations': _diagnostic_histogram(st_iterations, valid, range(25)),
        'L1ActualIterations': _diagnostic_histogram(l1_iterations, valid, range(6)),
    }
    return result


def empty_core_diagnostics(variable, prepared):
    """Record a real no-vegetation bypass; no controller prediction was made."""
    valid = _diagnostic_valid_cores(variable, prepared)
    if any(valid):
        raise ValueError('Empty diagnostics require a genuinely empty vegetation domain')
    core = dict(Valid=valid, Selected=[False] * 25, NeedsSolver=[False] * 25,
                STBudget=[0] * 25, L1Budget=[0] * 25,
                STIterations=[0] * 25, L1Iterations=[0] * 25,
                ControllerSTBudget=[None] * 25)
    return validate_core_diagnostics(_diagnostics_from_core_data(variable, core, True))


def capture_core_diagnostics(variable, prepared, result):
    """Capture returned solver counters, not routing proxies or pixel QA modes."""
    valid = _diagnostic_valid_cores(variable, prepared)
    if not any(valid):
        return empty_core_diagnostics(variable, prepared)
    st, l1 = result['st_diagnostics'], result['l1_diagnostics']
    if not isinstance(st, dict) or not isinstance(l1, dict):
        raise ValueError('Both solvers must return diagnostic dictionaries')
    selected = _diagnostic_slot(variable, st['routed'], 'ST.routed', boolean=True)
    needs = _diagnostic_slot(variable, st['needs_solver'], 'ST.needs_solver', boolean=True)
    if (selected != _diagnostic_slot(variable, l1['routed'], 'L1.routed', boolean=True)
            or needs != _diagnostic_slot(variable, l1['needs_solver'], 'L1.needs_solver', boolean=True)):
        raise ValueError('ST and L1 returned conflicting routing diagnostics')
    core = dict(Valid=valid, Selected=selected, NeedsSolver=needs,
                STBudget=_diagnostic_slot(variable, st['budget'], 'ST.budget', maximum=24),
                L1Budget=_diagnostic_slot(variable, l1['budget'], 'L1.budget', maximum=5),
                STIterations=_diagnostic_slot(variable, st['iterations'], 'ST.iterations', maximum=24),
                L1Iterations=_diagnostic_slot(variable, l1['iterations'], 'L1.iterations', maximum=5),
                ControllerSTBudget=_diagnostic_slot(variable, st['llm_budget'], 'ST.llm_budget',
                                                    maximum=24, levels=_CONTROLLER_LEVELS))
    return validate_core_diagnostics(_diagnostics_from_core_data(variable, core, False))


def validate_core_diagnostics(diag):
    """Recompute every count and histogram from saved native-core observations."""
    if not isinstance(diag, dict):
        raise ValueError('Core diagnostics must be a dictionary')
    variable, skipped = diag.get('Variable'), diag.get('InferenceSkipped')
    if variable not in ('LAI', 'FPAR') or type(skipped) is not bool:
        raise ValueError('Core diagnostics need a variable and an explicit bypass flag')
    core = diag.get('CoreData')
    fields = {'Valid', 'Selected', 'NeedsSolver', 'STBudget', 'L1Budget',
              'STIterations', 'L1Iterations', 'ControllerSTBudget'}
    if not isinstance(core, dict) or set(core) != fields:
        raise ValueError('Core diagnostics lack their complete raw observations')
    for name, values in core.items():
        if not isinstance(values, list) or len(values) != 25:
            raise ValueError(f'{name} must preserve all 25 native cores')
        if name in {'Valid', 'Selected', 'NeedsSolver'}:
            if any(type(value) is not bool for value in values):
                raise ValueError(f'{name} must preserve genuine boolean observations')
        elif name == 'ControllerSTBudget' and skipped:
            if any(value is not None for value in values):
                raise ValueError('An unexecuted controller has no observed budget prediction')
        else:
            maximum = 5 if name.startswith('L1') else 24
            if any(type(value) is not int or not 0 <= value <= maximum for value in values):
                raise ValueError(f'{name} must preserve bounded integer observations')
            if name == 'ControllerSTBudget' and any(value not in _CONTROLLER_LEVELS for value in values):
                raise ValueError('Controller budget observations are outside their frozen levels')
    if skipped and (any(core['Valid']) or any(core['Selected']) or any(core['NeedsSolver'])
                    or any(core['STBudget']) or any(core['L1Budget'])
                    or any(core['STIterations']) or any(core['L1Iterations'])):
        raise ValueError('A no-vegetation bypass must have no executed valid core')
    if any(value > budget for value, budget in zip(core['STIterations'], core['STBudget'])):
        raise ValueError('ST actual iterations cannot exceed their applied budget')
    if any(value > budget for value, budget in zip(core['L1Iterations'], core['L1Budget'])):
        raise ValueError('L1 actual iterations cannot exceed their applied budget')
    expected = _diagnostics_from_core_data(variable, core, skipped)
    count_fields = {'CoreSize', 'ValidCoreYears', 'SelectedCoreYears', 'STStartedCoreYears',
                    'L1StartedCoreYears', 'EitherStartedCoreYears', 'STIterationSum', 'L1IterationSum'}
    if any(type(diag.get(name)) is not int or diag[name] < 0 for name in count_fields):
        raise ValueError('Core diagnostic totals must be nonnegative integers')
    distributions = diag.get('Distributions')
    if not isinstance(distributions, dict) or set(distributions) != set(expected['Distributions']):
        raise ValueError('Core diagnostic distributions are incomplete')
    for name, histogram in distributions.items():
        if (not isinstance(histogram, dict) or set(histogram) != set(expected['Distributions'][name])
                or any(type(value) is not int or value < 0 for value in histogram.values())):
            raise ValueError(f'{name} must preserve the complete integer histogram')
        if sum(histogram.values()) != expected['ValidCoreYears']:
            raise ValueError(f'{name} histogram does not cover the valid-core domain')
    if diag != expected:
        raise ValueError('Core diagnostic counts or distributions disagree with their raw observations')
    return diag


VARIABLES = ('LAI', 'FPAR')
_COUNT_FIELDS = ('ValidCoreYears', 'SelectedCoreYears', 'STStartedCoreYears',
                 'L1StartedCoreYears', 'EitherStartedCoreYears',
                 'STIterationSum', 'L1IterationSum')
_PERCENTAGES = ('STActivationPct', 'L1ActivationPct', 'EitherActivationPct', 'SelectionPct')
_BINS = {
    'ControllerSTBudget': _CONTROLLER_LEVELS,
    'STAppliedBudget': tuple(range(25)),
    'L1AppliedBudget': tuple(range(6)),
    'STActualIterations': tuple(range(25)),
    'L1ActualIterations': tuple(range(6)),
}


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _years(years):
    if (not isinstance(years, (list, tuple)) or not years
            or any(type(year) is not int or year not in range(2001, 2025) for year in years)
            or len(set(years)) != len(years)):
        raise ValueError('Use distinct target years from 2001 through 2024')
    return sorted(years)


def _provenance(records, name, explicit):
    supplied = [record[name] for record in records if name in record]
    if any(not isinstance(value, str) or not value for value in supplied):
        raise ValueError(f'{name} must be a nonempty string when provided')
    values = set(supplied)
    if explicit is not None:
        if not isinstance(explicit, str) or not explicit:
            raise ValueError(f'{name} must be a nonempty string')
        values.add(explicit)
    if len(values) > 1:
        raise ValueError(f'Conflicting {name} values in solver statistics')
    return next(iter(values), '')


def _accumulator():
    return dict(**{name: 0 for name in _COUNT_FIELDS}, CompletedPatches=0,
                Distributions={name: {str(value): 0 for value in bins}
                               for name, bins in _BINS.items()})


def _add(target, source):
    for name in _COUNT_FIELDS:
        target[name] += source[name]
    target['CompletedPatches'] += source.get('CompletedPatches', 1)
    for name, bins in _BINS.items():
        for value in bins:
            target['Distributions'][name][str(value)] += source['Distributions'][name][str(value)]


def _checked_metrics(value):
    valid = value['ValidCoreYears']
    for name, histogram in value['Distributions'].items():
        if sum(histogram.values()) != valid:
            raise ValueError(f'{name} distribution does not cover its valid-core domain')
    for prefix in ('ST', 'L1'):
        histogram = value['Distributions'][prefix + 'ActualIterations']
        if sum(int(key) * count for key, count in histogram.items()) != value[prefix + 'IterationSum']:
            raise ValueError(f'{prefix} iteration distribution disagrees with its actual iteration sum')
        if sum(count for key, count in histogram.items() if int(key) > 0) != value[prefix + 'StartedCoreYears']:
            raise ValueError(f'{prefix} iteration distribution disagrees with its actual started count')
    if not (max(value['STStartedCoreYears'], value['L1StartedCoreYears'])
            <= value['EitherStartedCoreYears']
            <= min(valid, value['STStartedCoreYears'] + value['L1StartedCoreYears'])):
        raise ValueError('Recorded solver union is outside its valid-core domain')
    return {name: (100.0 * value[numerator] / valid if valid else None)
            for name, numerator in zip(_PERCENTAGES, ('STStartedCoreYears', 'L1StartedCoreYears',
                                                       'EitherStartedCoreYears', 'SelectedCoreYears'))}


def aggregate(records, years, *, tile=None, run_fingerprint=None):
    """Return actual solver tables and metadata without a run fingerprint.

    Each record is {year,row,col,backend,solver_core_diagnostics:{LAI,FPAR}}.
    Both variable diagnostics are mandatory, although either can be a real
    no-vegetation bypass. Region/partial samples are supported and explicitly
    counted; this function does not certify full-tile completeness. Origins
    retain the full-tile zero-based 40-pixel native grid.

    Repeated (year,row,col) records count once only if actual diagnostics and
    backend/provided claim provenance agree. Retry/timing fields are ignored.
    The tile keyword identifies the spatial scope. The legacy run_fingerprint
    keyword is accepted but ignored, including on inherited patch records.
    """
    years = _years(years)
    records = list(records)
    if any(not isinstance(record, dict) for record in records):
        raise ValueError('Solver statistics need dictionary patch records')
    tile = _provenance(records, 'tile', tile)
    annual = {(year, variable): _accumulator() for year in years for variable in VARIABLES}
    seen, duplicate_patch_count = {}, 0
    for record in records:
        year, row, col = (record.get(name) for name in ('year', 'row', 'col'))
        if (type(year) is not int or year not in years
                or type(row) is not int or row not in range(0, 2400, 40)
                or type(col) is not int or col not in range(0, 2400, 40)):
            raise ValueError('Solver diagnostic patch is outside the configured native scope')
        backend = record.get('backend')
        if backend not in {'cpu', 'mps', 'cuda'} or record.get('assigned_backend', backend) != backend:
            raise ValueError('Solver diagnostics need the actual consistent CPU, MPS or CUDA backend')
        diagnostics = record.get('solver_core_diagnostics')
        if not isinstance(diagnostics, dict) or set(diagnostics) != set(VARIABLES):
            raise ValueError('Each completed patch must record both actual variable diagnostics')
        for variable in VARIABLES:
            diagnostic = validate_core_diagnostics(diagnostics[variable])
            if diagnostic['Variable'] != variable:
                raise ValueError('Solver diagnostic is stored under the wrong variable')
        # Scope identity is inferred/checked globally above. Preserve optional
        # durable claim provenance in conflict detection without requiring a
        # local scheduler or importing the original production implementation.
        signature = _canonical(dict(diagnostics=diagnostics, backend=backend,
                                    claim_id=record.get('claim_id'),
                                    claim_digest=record.get('claim_digest')))
        key = year, row, col
        if key in seen:
            if seen[key] != signature:
                raise ValueError('Repeated patch diagnostics conflict with observations, backend or claim')
            duplicate_patch_count += 1
            continue
        seen[key] = signature
        for variable in VARIABLES:
            _add(annual[year, variable], diagnostics[variable])

    status = 'recorded_actual_diagnostics'
    metrics_headers = list(_COUNT_FIELDS) + list(_PERCENTAGES)
    tails = ['CompletedPatches', 'Status']
    annual_headers = ['Tile', 'Year', 'Variable'] + metrics_headers + tails
    total_headers = ['Tile', 'Variable'] + metrics_headers + tails
    distribution_headers = ['Tile', 'Scope', 'Year', 'Variable', 'Distribution',
                            'Value', 'CoreYears', 'Status']
    annual_rows, total_rows, distribution_rows = [], [], []
    totals = {variable: _accumulator() for variable in VARIABLES}

    def metric_values(value):
        metrics = _checked_metrics(value)
        return [value[name] for name in _COUNT_FIELDS] + [metrics[name] for name in _PERCENTAGES]

    def distributions(scope, year, variable, value):
        for name, bins in _BINS.items():
            for number in bins:
                distribution_rows.append([tile, scope, year, variable, name, number,
                                          value['Distributions'][name][str(number)], status])

    for year in years:
        for variable in VARIABLES:
            value = annual[year, variable]
            annual_rows.append([tile, year, variable] + metric_values(value)
                               + [value['CompletedPatches'], status])
            distributions('YEAR', year, variable, value)
            _add(totals[variable], value)
    total_metadata = {}
    for variable in VARIABLES:
        value = totals[variable]
        total_rows.append([tile, variable] + metric_values(value)
                          + [value['CompletedPatches'], status])
        distributions('TOTAL', None, variable, value)
        total_metadata[variable] = dict(**{name: value[name] for name in _COUNT_FIELDS},
                                       **_checked_metrics(value), CompletedPatches=value['CompletedPatches'],
                                       Status=status)
    patch_count_by_year = {str(year): sum(identity[0] == year for identity in seen) for year in years}
    return dict(tables={
        'solver_by_year_variable.csv': dict(headers=annual_headers, rows=annual_rows),
        'solver_total_by_variable.csv': dict(headers=total_headers, rows=total_rows),
        'solver_budget_distribution.csv': dict(headers=distribution_headers, rows=distribution_rows),
    }, metadata=dict(schema='qa-llm-gf-recorded-solver-statistics-v1', tile=tile,
                     target_years=years,
                     covered_patch_count=len(seen), covered_patch_count_by_year=patch_count_by_year,
                     duplicate_records_ignored=2 * duplicate_patch_count,
                     duplicate_patch_records_ignored=duplicate_patch_count,
                     count_unit='core_target_year', duplicate_count_unit='patch_variable_record',
                     core_size=8, total_by_variable=total_metadata, status=status,
                     full_native_grid_by_year={year: count == 3600 for year, count in patch_count_by_year.items()}))


def _same_statistics_without_legacy_fingerprint(path, expected):
    """Compare inherited statistics while leaving their original bytes intact."""
    old = path.read_text(encoding='utf-8')
    if path.suffix == '.csv':
        rows = list(csv.reader(io.StringIO(old)))
        if not rows:
            return False
        keep = [i for i, name in enumerate(rows[0])
                if name.casefold().replace('_', '') not in {'runfingerprint', 'fingerprint'}]
        if any(len(row) != len(rows[0]) for row in rows):
            return False
        normalized = [[row[i] for i in keep] for row in rows]
        return normalized == list(csv.reader(io.StringIO(expected)))
    def lines(value):
        return [line for line in value.splitlines()
                if not line.strip().casefold().startswith(('run fingerprint:', 'fingerprint:'))]
    return lines(old) == lines(expected)


def write_solver_statistics(records, output, years, *, tile=None, run_fingerprint=None):
    """Write three CSVs and notes directly into output; return metadata/hashes.

    Existing equivalent legacy summaries are accepted with their old fingerprint
    column/notes ignored, without changing those files. Conflicting files and symlinks
    are rejected before writing any summary, so this API never silently
    replaces a previously published result. Empty denominators serialize as
    blank CSV cells. No runtime or performance table is emitted.
    """
    package = aggregate(records, years, tile=tile, run_fingerprint=run_fingerprint)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if output.is_symlink():
        raise ValueError('Solver-statistics output must not be a symlink')
    metadata = package['metadata']
    texts = {}
    for name, table in package['tables'].items():
        stream = io.StringIO(newline='')
        writer = csv.writer(stream)
        writer.writerow(table['headers'])
        writer.writerows(table['rows'])
        texts[name] = stream.getvalue()
    texts['solver_statistics_notes.txt'] = (
        f"{metadata['tile'] or 'Unspecified tile'} real solver diagnostics, target years {metadata['target_years']}.\n"
        "Each active variable call records all 25 row-major 8x8 cores in solver_core_diagnostics.\n"
        "ValidCoreYears counts center-year vegetation-domain cores; neighboring years and DOYs are not extra core-years.\n"
        "ST/L1 starts are observed diagnostic iterations >0; EitherStarted is their per-core union.\n"
        "Selected is counted separately; QA mode, routing fractions and positive budgets are not actual activation counters.\n"
        "ControllerSTBudget records the observed six-level cap (0,3,5,8,13,24).\n"
        "STAppliedBudget allows every actual integer0-24, including difficulty-truncated caps; L1AppliedBudget uses0-5.\n"
        "STActualIterations and L1ActualIterations retain observed zero iterations for every valid core.\n"
        "No-vegetation patches record a real bypass with zero valid cores and no imputed controller prediction.\n"
        "Repeated target-year/native-patch origins are deduplicated before both variable totals; conflicts are rejected.\n"
        "duplicate_records_ignored retains the original patch-variable unit; duplicate_patch_records_ignored counts patch records.\n"
        "Annual and overall ratios use summed numerators/summed valid-core denominators; a zero denominator is blank.\n"
        "Region/partial samples are allowed: CompletedPatches and full_native_grid_by_year identify the actual coverage.\n"
        "Statistics contain no runtime table and no claimed speedup from activation percentages.\n"
    )
    for name, value in texts.items():
        path = output / name
        if path.exists() or path.is_symlink():
            if (path.is_symlink() or not path.is_file()
                    or not _same_statistics_without_legacy_fingerprint(path, value)):
                raise RuntimeError(f'Previously published solver statistics differ: {path}')
    seals = {}
    for name, value in texts.items():
        path = output / name
        if not path.exists():
            temporary = output / f'.{name}.{os.getpid()}.{uuid.uuid4().hex}.tmp'
            try:
                with temporary.open('x', encoding='utf-8', newline='') as stream:
                    stream.write(value)
                    stream.flush()
                    os.fsync(stream.fileno())
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        seals[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return dict(metadata, file_sha256=seals)

