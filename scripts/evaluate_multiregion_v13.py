"""V13 CPU evaluation of fixed one/three-crop pools and saved shortlist choices.

No model is executed and no epoch, seed, crop or candidate is selected here.
``evaluate_records`` accepts the complete cases, flat saved predictions and
pipeline records; each budgeted prediction explicitly declares c1 or c3.
The sealed V12 statistics implementation supplies canonical point scoring and
the source-stratified screenshot-cluster bootstrap, without changing its code.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import html
import json
import math
from pathlib import Path

import numpy as np
from shapely.geometry import Point, box

import evaluate_crossfit_candidates as base
from clickagent.action_benchmark import sha256, write_json
from clickagent.spatial_geometry import shape_from_geometry


OUT = Path('runs/multiregion_v13')
CONDITIONS = ('c1', 'c3')
BUDGETS = (4, 8, 16, 32)
BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 20261313
EXPECTED_COUNTS = {'test': 192, 'supplement': 6}
HEAD_NAMES = tuple(f'{arm}_{seed}_val' for arm in ('c1_top1', 'c3_top1', 'c3_top4') for seed in ('s1', 's2')) + tuple(
    f'c3_{objective}_{seed}_loso' for objective in ('top1', 'top4') for seed in ('s1', 's2'))


def _require(value, message):
    if not value:
        raise ValueError(message)


def _jsonable(value):
    """Match persisted JSON key semantics, without rounding metric values."""
    if isinstance(value, dict):
        _require(all(isinstance(k, str) or type(k) is int for k in value), 'Unexpected JSON key')
        converted = {str(k): _jsonable(v) for k, v in value.items()}
        _require(len(converted) == len(value), 'Colliding JSON keys')
        return converted
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _ci(draws):
    return np.quantile(draws, [.025, .975]).tolist()


def _condition(row):
    condition = row.get('condition')
    if row.get('budget') is not None:
        _require(condition in CONDITIONS, 'Every budgeted method needs explicit c1/c3 condition')
    else:
        _require(condition in (*CONDITIONS, None), 'Invalid reference condition')
    return condition


def _comparison_specs(names):
    """Fixed scientific comparisons; other pairs are descriptive only."""
    pairs = []
    def add(a, b, role):
        if a in names and b in names and a != b and not any({a, b} == {x, y} for x, y, _ in pairs):
            pairs.append((a, b, role))
    for seed in ('s1', 's2'):
        add(f'c3_top1_{seed}_loso__K16', f'c1_top1_{seed}_val__K16',
            'primary' if seed == 's1' else 'robustness')
        for suffix in ('', '_top4_gui'):
            def name(arm, selector='val', k=16):
                return f'{arm}_{seed}_{selector}{suffix}__K{k}'
            add(name('c3_top1'), name('c1_top1'), 'search_ablation')
            if suffix:
                add(name('c3_top4', 'loso'), name('c1_top1'), 'exploratory_overall_compression')
            for selector in ('val', 'loso'):
                add(name('c3_top4', selector), name('c3_top1', selector),
                    'compression_objective_' + selector + ('_primary' if seed == 's1' else '_robustness'))
            for objective in ('top1', 'top4'):
                add(name('c3_' + objective, 'loso'), name('c3_' + objective), 'selection_protocol')
        for arm in ('c1_top1', 'c3_top1', 'c3_top4'):
            for k in (4, 8, 32):
                if arm != 'c1_top1':
                    add(f'{arm}_{seed}_val__K{k}', f'c1_top1_{seed}_val__K{k}', 'secondary_budget')
    for name in names:
        if name.endswith('__K16') and not name.startswith('gui_'):
            condition = 'c1' if name.startswith('c1_') else 'c3' if name.startswith('c3_') else None
            if condition:
                add(name, f'gui_{condition}__K16', 'same_pool_verifier_reference')
                if '_top4_gui__K16' in name:
                    add(name, f'gui_{condition}__K4', 'verification_count_proxy')
            add(name, 'qwen_native_completed_lax', 'native_reference')
    add('gui_c3__K16', 'gui_c1__K16', 'verifier_search_reference')
    return pairs


def independent_geometry_audit(cases, details):
    """Independently check all saved pool/chosen points with Shapely covers.

    Canonical point_in_geometry remains the metric. Differences at its existing
    1e-12 boundary tolerance are disclosed, never silently converted to misses.
    """
    checked = 0; differences = []
    for case, detail in zip(cases, details):
        shape = shape_from_geometry(case['target_geometry']); size = np.asarray(case['image_size'])
        points = {tuple(p) for budgets in detail['pools'].values() for pool in budgets.values() for p in pool['points']}
        points.update(tuple(m['point']) for m in detail['methods'].values() if m['point'] is not None)
        for point in sorted(points):
            checked += 1
            canonical = base.point_hit(list(point), case)
            independent = bool(shape.covers(Point(*(np.asarray(point) / size))))
            if canonical != independent:
                differences.append(dict(sample_id=case['sample_id'], point=list(point), canonical=canonical,
                    shapely_covers=independent, canonical_numerical_boundary_tolerance=1e-12))
    return dict(unique_case_points_checked=checked, numerical_boundary_differences=differences,
                headline_metric_unchanged=True)


def _paired(cases, matrix, names, bundle, specs):
    result = []
    for method, reference, role in specs:
        a, b = names.index(method), names.index(reference)
        delta = matrix[:, a].astype(int) - matrix[:, b].astype(int)
        row = dict(method=method, reference=reference, role=role, n=len(cases),
                   wins=int((delta > 0).sum()), losses=int((delta < 0).sum()),
                   both_hit=int(((matrix[:, a] == 1) & (matrix[:, b] == 1)).sum()),
                   both_miss=int(((matrix[:, a] == 0) & (matrix[:, b] == 0)).sum()),
                   delta=float(delta.mean()), ci95=_ci(bundle['overall'][:, a] - bundle['overall'][:, b]),
                   by_source={})
        for source, draws in bundle['by_source'].items():
            ix = [i for i, c in enumerate(cases) if base._source(c) == source]
            d = delta[ix]
            row['by_source'][source] = dict(n=len(ix), wins=int((d > 0).sum()), losses=int((d < 0).sum()),
                delta=float(d.mean()), ci95=_ci(draws[:, a] - draws[:, b]))
        result.append(row)
    return result


def _validate_scores_and_shortlists(cases, predictions, pipelines, gui_records):
    """Authenticate saved decisions against their actual pool and saved scores."""
    case_map = {c['sample_id']: c for c in cases}
    pipeline_map = {p['sample_id']: p for p in pipelines}
    by_token = {}
    for row in predictions:
        token = (row['sample_id'], row['method'], row.get('budget'))
        _require(token not in by_token, 'Repeated prediction')
        by_token[token] = row
    for row in predictions:
        sid = row['sample_id']; budget = row.get('budget'); condition = _condition(row)
        _require(sid in case_map, 'Unknown prediction case')
        if budget is None:
            continue
        _require(budget in BUDGETS, 'Unexpected candidate budget')
        pool = pipeline_map[sid]['pools'][condition][str(budget)]
        points = [p['point'] for p in pool]
        if 'scores' in row:
            scores = np.asarray(row['scores'], dtype=np.float64)
            _require(scores.shape == (len(pool),) and np.isfinite(scores).all(), 'Invalid candidate scores')
            if 'retained_points' not in row:
                expected = points[int(scores.argmax())] if points else None
                _require(row['point'] == expected, 'Saved head point differs from stable score argmax')
        if row['method'].endswith('_top4_gui'):
            _require(budget == 16, 'Top4 GUI policy must use K16')
            parent_token = (sid, row['method'][:-len('_top4_gui')], 16)
            _require(parent_token in by_token, 'Shortlist has no saved parent head')
            parent = by_token[parent_token]
            _require(parent['condition'] == condition and 'scores' in parent, 'Shortlist parent condition/scores differ')
            scores = np.asarray(parent['scores'], dtype=np.float64)
            order = np.argsort(-scores, kind='stable')[:4]
            expected = [points[int(i)] for i in order]
            _require(row.get('retained_points') == expected and row.get('n_verified', len(expected)) == len(expected),
                     'Retained points differ from stable K16 Top4')
        if gui_records is not None and (row['method'].startswith('gui_') or row['method'].endswith('_top4_gui')):
            records = gui_records[sid]
            retained = row.get('retained_points', points)
            values = []
            for point in retained:
                key = ','.join(map(str, point))
                _require(key in records, 'Saved GUI point score missing')
                record = records[key]
                _require(record.get('point', point) == point and math.isfinite(float(record['log_odds'])),
                         'Invalid/misaligned GUI point score')
                values.append(float(record['log_odds']))
            expected = retained[int(np.argmax(values))] if retained else None
            _require(row['point'] == expected, 'Saved GUI choice differs from shared point log-odds')
            if row['method'].startswith('gui_'):
                _require(row['method'] == 'gui_' + condition, 'GUI method and condition disagree')


def _crop_details(case, pipeline):
    """Posthoc geometry descriptions, never inputs to crop/candidate selection."""
    views = pipeline.get('local_views', [])
    _require(isinstance(views, list) and len(views) <= 3, 'Expected zero to three local views')
    shape = shape_from_geometry(case['target_geometry'])
    w, h = case['image_size']; items = []
    for i, view in enumerate(views):
        transform = view['transform']; coordinates = transform['box']
        a = np.asarray(coordinates, dtype=float)
        _require(a.shape == (4,) and np.isfinite(a).all() and 0 <= a[0] < a[2] <= w and 0 <= a[1] < a[3] <= h,
                 'Invalid crop box')
        crop = box(a[0] / w, a[1] / h, a[2] / w, a[3] / h)
        overlap = shape.intersection(crop).area
        items.append(dict(index=i, box=coordinates, target_positive_area_overlap=bool(overlap > 0),
            target_area_coverage=float(overlap / shape.area)))
    first = bool(items and items[0]['target_positive_area_overlap'])
    any_view = any(v['target_positive_area_overlap'] for v in items)
    return dict(n_views=len(items), views=items, first_crop_overlaps=first, any_crop_overlaps=any_view,
                additional_crop_new_overlap=any_view and not first)


def evaluate_records(cases, predictions, pipelines, *, gui_records=None, bootstrap_reps=BOOTSTRAP_REPS,
                     bootstrap_seed=BOOTSTRAP_SEED, cohort='unspecified'):
    """Evaluate all supplied cases; each observed method must cover every case.

    Pipeline pools are ``{c1:{'16':[...]},c3:{'16':[...]}}``. Inputs may use
    any aligned subset of the predefined budgets. CLI completeness/count checks
    are separate from this import API, so synthetic and validation cohorts work.
    """
    cases, predictions, pipelines = list(cases), list(predictions), list(pipelines)
    _require(bool(cases) and bool(predictions), 'Cases and predictions must not be empty')
    ids = [c['sample_id'] for c in cases]
    _require(len(set(ids)) == len(ids), 'Repeated case id')
    _require(len(pipelines) == len(cases) and {p['sample_id'] for p in pipelines} == set(ids),
             'Pipelines must cover all cases exactly once')
    pipeline_map = {p['sample_id']: p for p in pipelines}
    _require(all(set(p['pools']) == set(CONDITIONS) for p in pipelines), 'Both c1/c3 pools are required')
    descriptors = {}
    for row in predictions:
        name = base.method_key(row['method'], row.get('budget'))
        condition = _condition(row)
        _require(name not in descriptors or descriptors[name] == condition, 'Method changes condition across cases')
        descriptors[name] = condition
    _validate_scores_and_shortlists(cases, predictions, pipelines, gui_records)
    merged = {sid: dict(sample_id=sid, source=base._source(cases[i]), dataset=cases[i].get('dataset'),
        screenshot_cluster=base._cluster(cases[i]), pools={}, methods={}) for i, sid in enumerate(ids)}
    methods = {}; pools = {}; scored = []
    for condition in (*CONDITIONS, None):
        rows = [r for r in predictions if _condition(r) == condition]
        pool_rows = [dict(sample_id=sid, pools=pipeline_map[sid]['pools'][condition] if condition else {}) for sid in ids]
        # Validate and describe a condition's pool even when no method uses it.
        supplied = rows or [dict(sample_id=sid, method='_pool_validation_only', budget=None, point=None) for sid in ids]
        result = base.evaluate_records(cases, supplied, pool_rows, bootstrap_reps=bootstrap_reps,
                                       bootstrap_seed=bootstrap_seed)
        if condition:
            pools[condition] = result['summary']['pools']
        if rows:
            methods.update({m: dict(v, condition=condition) for m, v in result['summary']['methods'].items()})
        for detail in result['details']:
            if condition:
                merged[detail['sample_id']]['pools'][condition] = detail['pools']
            if rows:
                for name, value in detail['methods'].items():
                    merged[detail['sample_id']]['methods'][name] = dict(value, condition=condition)
    names = sorted(methods); details = [merged[sid] for sid in ids]
    for name in names:
        values = [d['methods'][name] for d in details]
        for source, metric in methods[name]['by_source'].items():
            ix = [i for i, c in enumerate(cases) if base._source(c) == source]
            metric['failure_counts'] = dict(Counter(values[i]['failure_stage'] for i in ix))
            if methods[name]['pool_budget'] is not None:
                available = sum(values[i]['pool_oracle'] for i in ix)
                metric.update(pool_available=available,
                    conditional_selection_accuracy=metric['hits'] / available if available else None,
                    selector_missed_correct_point=sum(values[i]['pool_oracle'] and not values[i]['hit'] for i in ix))
    matrix = np.asarray([[d['methods'][m]['hit'] for m in names] for d in details], dtype=bool)
    bundle = base._bootstrap_bundle(matrix, cases, bootstrap_reps, bootstrap_seed)
    pairs = _paired(cases, matrix, names, bundle, _comparison_specs(names))
    for pair in pairs:
        for field, method, reference in (('method_losses_breakdown', pair['method'], pair['reference']),
                                          ('reference_losses_breakdown', pair['reference'], pair['method'])):
            pair[field] = dict(Counter(d['methods'][method]['failure_stage'] for d in details
                if not d['methods'][method]['hit'] and d['methods'][reference]['hit']))
    prediction_map = {(r['sample_id'], base.method_key(r['method'], r.get('budget'))): r for r in predictions}
    top4_names = []
    for name in names:
        values = [d['methods'][name] for d in details]
        source_rows = [prediction_map[(sid, name)] for sid in ids]
        if methods[name]['pool_budget'] != 16:
            continue
        has_scores = ['scores' in r and 'retained_points' not in r for r in source_rows]
        _require(not any(has_scores) or all(has_scores), 'K16 scores must cover every case')
        has_retained = ['retained_points' in r for r in source_rows]
        _require(not any(has_retained) or all(has_retained), 'Retained-set metadata must cover every case')
        if not any(has_scores) and not any(has_retained):
            continue
        condition = methods[name]['condition']; observations = []; counts = []
        for case, detail, row, value in zip(cases, details, source_rows, values):
            points = detail['pools'][condition]['16']['points']
            retained = row['retained_points'] if 'retained_points' in row else [points[int(j)] for j in
                np.argsort(-np.asarray(row['scores']), kind='stable')[:4]]
            _require(len(retained) == min(4, len(points)), 'A Top4 set must retain exactly min(4,N) points')
            hit = any(base.point_hit(p, case) for p in retained)
            value.update(top4_points=retained, top4_oracle=hit)
            observations.append(hit); counts.append(len(retained))
            if 'retained_points' in row:
                value['shortlist_failure_stage'] = ('hit' if value['hit'] else 'empty_pool' if not points else
                    'nonempty_pool_without_correct_point' if not value['pool_oracle'] else
                    'compression_dropped_all_correct_points' if not hit else 'gui_misselected_reachable_shortlist')
        oracle = sum(observations); available = methods[name]['pool_available']; hits = methods[name]['hits']
        top4 = dict(oracle=oracle, oracle_accuracy=oracle / len(cases), parent_pool_oracle=available,
            conditional_retention=oracle / available if available else None,
            compression_misses=available - oracle, total_candidates=sum(counts),
            mean_candidates=float(np.mean(counts)), candidate_count_distribution=dict(Counter(counts)),
            definition='At least one of stable score-Top4 fixed points hits any valid target region',
            cost_scope='verification image count proxy; not measured latency', by_source={})
        for source in sorted({base._source(c) for c in cases}):
            ix = [i for i, c in enumerate(cases) if base._source(c) == source]
            available_s = sum(values[i]['pool_oracle'] for i in ix)
            oracle_s = sum(observations[i] for i in ix)
            top4['by_source'][source] = dict(n=len(ix), oracle=oracle_s, parent_pool_oracle=available_s,
                conditional_retention=oracle_s / available_s if available_s else None,
                compression_misses=available_s - oracle_s)
            if any(has_retained):
                hits_s = sum(values[i]['hit'] for i in ix)
                top4['by_source'][source].update(gui_hits=hits_s, gui_misselections=oracle_s - hits_s,
                    gui_conditional_accuracy=hits_s / oracle_s if oracle_s else None,
                    failure_counts=dict(Counter(values[i]['shortlist_failure_stage'] for i in ix)))
        if any(has_retained):
            _require(hits <= oracle <= available, 'Shortlist/parent oracle ordering violated')
            top4.update(gui_conditional_accuracy=hits / oracle if oracle else None,
                        gui_misselections=oracle - hits,
                        failure_counts=dict(Counter(v['shortlist_failure_stage'] for v in values)))
            _require(sum(top4['failure_counts'].values()) == len(cases), 'Shortlist failure partition incomplete')
        methods[name]['top4'] = top4; top4_names.append(name)
    top4_pairs = []
    if top4_names:
        top4_matrix = np.asarray([[d['methods'][m]['top4_oracle'] for m in top4_names] for d in details], dtype=bool)
        top4_bundle = base._bootstrap_bundle(top4_matrix, cases, bootstrap_reps, bootstrap_seed)
        for j, name in enumerate(top4_names):
            methods[name]['top4']['ci95'] = _ci(top4_bundle['overall'][:, j])
            for source, draws in top4_bundle['by_source'].items():
                methods[name]['top4']['by_source'][source]['ci95'] = _ci(draws[:, j])
        top4_pairs = _paired(cases, top4_matrix, top4_names, top4_bundle, _comparison_specs(top4_names))
    for detail, case in zip(details, cases):
        detail['crop_geometry'] = _crop_details(case, pipeline_map[case['sample_id']])
        for name, value in detail['methods'].items():
            scored.append(dict(sample_id=detail['sample_id'], source=detail['source'], method_key=name, **value))
    coverage = {}
    for budget in sorted(set(pools['c1']) & set(pools['c3']), key=int):
        a = np.asarray([d['pools']['c1'][budget]['point_oracle'] for d in details], bool)
        b = np.asarray([d['pools']['c3'][budget]['point_oracle'] for d in details], bool)
        coverage[budget] = dict(c1_oracle=int(a.sum()), c3_oracle=int(b.sum()),
            gained=int((b & ~a).sum()), lost=int((a & ~b).sum()), both=int((a & b).sum()),
            neither=int((~a & ~b).sum()), delta=float(b.mean() - a.mean()))
    crops = [d['crop_geometry'] for d in details]
    summary = dict(complete=True, version=13, cohort=cohort, n=len(cases), methods=methods, pools=pools,
        primary=dict(method='c3_top1_s1_loso__K16', reference='c1_top1_s1_val__K16',
            available=all(m in names for m in ('c3_top1_s1_loso__K16', 'c1_top1_s1_val__K16'))),
        robustness=dict(method='c3_top1_s2_loso__K16', reference='c1_top1_s2_val__K16',
            available=all(m in names for m in ('c3_top1_s2_loso__K16', 'c1_top1_s2_val__K16'))),
        paired_comparisons=pairs, paired_top4_oracle=top4_pairs, coverage_changes=coverage,
        crop_geometry=dict(n=len(cases), actual_view_counts=dict(Counter(c['n_views'] for c in crops)),
            first_crop_overlap=sum(c['first_crop_overlaps'] for c in crops),
            any_crop_overlap=sum(c['any_crop_overlaps'] for c in crops),
            additional_crop_new_overlap=sum(c['additional_crop_new_overlap'] for c in crops),
            definition='Posthoc positive-area intersection with any valid target; not used for selecting crops'),
        bootstrap=bundle['metadata'], metric='All-case any-region integer point hit rate',
        point_metric='Canonical public_evaluation.point_in_geometry; any valid bbox/polygon, existing 1e-12 normalized numerical tolerance, no added pixel radius or distance reward',
        denominator='Every supplied case, including empty pools, unavailable views and null outputs',
        budget_scope='Fixed final point and verification-count budgets; three crops require additional upstream Qwen work',
        loso_scope='Scorer-only source-heldout CV selects one common epoch, not an independent test score. Frozen upstream saw these sources and used their validation labels for upstream selection; not end-to-end LODO.',
        inference_scope='Saved Top1 and Top4/GUI choices only; null is auxiliary and not an action-permission gate',
        audit_scope='Record, pool, saved-score and metric validation; freeze and training provenance require the separate auditor',
        test_selection_used=False if cohort in EXPECTED_COUNTS else None)
    audit = dict(complete=True, version=13, cohort=cohort, n=len(cases), methods=len(names),
        predictions=len(predictions), each_method_has_all_cases=True, declared_condition_pools_checked=True,
        head_argmax_replayed=True, top4_stable_order_replayed=True,
        shared_gui_argmax_replayed=gui_records is not None,
        independent_geometry=independent_geometry_audit(cases, details), model_execution=False,
        scope='Saved-record completeness, decision replay and independent point geometry only; no raw-head/training provenance audit')
    return _jsonable(dict(summary=summary, details=details, scored_predictions=scored, audit=audit))


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def _rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8-sig').splitlines() if line.strip()]


def _key(sid):
    return hashlib.sha256(sid.encode()).hexdigest()[:24]


def load_records(root, split):
    """Load complete per-case artifacts, retaining exactly their saved choices."""
    root = Path(root); case_path = root / 'data' / (split + '.jsonl'); cases = _rows(case_path)
    _require(len(cases) == EXPECTED_COUNTS[split], f'Incomplete {split} cohort')
    paths = [case_path]; predictions = []; pipelines = []; gui_records = {}
    for case in cases:
        sid = case['sample_id']; stem = _key(sid)
        pp = root / 'pipeline' / split / (stem + '.json')
        hp = root / 'predictions' / split / (stem + '.json')
        gp = root / 'gui_reference' / split / (stem + '.json')
        pipeline, head, gui = _read(pp), _read(hp), _read(gp); paths.extend((pp, hp, gp))
        _require(all(r['sample_id'] == sid for r in (pipeline, head, gui)), 'Artifact/sample mismatch')
        _require(pipeline['complete'] and pipeline['ground_truth_in_model_inputs'] is False,
                 'Incomplete/GT-dependent candidate pipeline')
        from clickagent.multiregion_proposals import build_c1_c3_pools
        _require(pipeline['pools'] == build_c1_c3_pools(pipeline['global_result'], pipeline['local_views']),
                 'Saved candidate pool differs from frozen decoded-point trace')
        _require(head['pipeline_sha256'] == sha256(pp), 'Head used a different pipeline')
        expected_heads = {(name, budget) for name in HEAD_NAMES for budget in BUDGETS} | {
            ('global_top1', None), ('qwen_native_completed_lax', None)}
        expected_gui = {('gui_' + c, k) for c in CONDITIONS for k in (4, 16)} | {
            (name + '_top4_gui', 16) for name in HEAD_NAMES}
        for artifact, expected in ((head, expected_heads), (gui, expected_gui)):
            tokens = [(r['method'], r.get('budget')) for r in artifact['predictions']]
            _require(len(tokens) == len(expected) and set(tokens) == expected, 'Missing/repeated registered prediction method')
        required_gui = {','.join(map(str, p['point'])) for c in CONDITIONS for p in pipeline['pools'][c]['16']}
        _require(set(gui['records']) == required_gui, 'GUI records must cover exactly the shared K16 union')
        native_path = root / 'native' / split / (stem + '.json'); native = _read(native_path); paths.append(native_path)
        head_map = {(r['method'], r.get('budget')): r for r in head['predictions']}
        _require(native['ground_truth_in_model_inputs'] is False and
                 head_map[('qwen_native_completed_lax', None)]['point'] == native['completed_lax_point'],
                 'Native decoder coordinates changed or used GT')
        _require(head_map[('global_top1', None)]['point'] == pipeline['global_result']['point'],
                 'Global Top1 coordinates changed')
        for artifact in (head, gui):
            _require(artifact.get('ground_truth_in_model_inputs') is False, 'Missing GT-free prediction declaration')
            for row in artifact['predictions']:
                _require(row.get('sample_id', sid) == sid, 'Nested prediction/sample mismatch')
                predictions.append(dict(row, sample_id=sid))
        pipelines.append(pipeline); gui_records[sid] = gui['records']
    return cases, predictions, pipelines, gui_records, paths


def selection_overview(root):
    """Describe fixed selection evidence; never select another epoch here."""
    path = Path(root) / 'models/selection.json'; value = _read(path)
    _require(value['complete'] and value['fresh_test_read'] is False and value['seed_selected_by_test'] is False,
             'Missing/invalid pretest model selection')
    _require(set(value['selections']) == set(HEAD_NAMES), 'Registered selected heads are incomplete')
    overview = dict(models={name: {k: row[k] for k in ('condition', 'objective', 'policy', 'seed', 'epoch', 'validation')}
        for name, row in value['selections'].items()}, cross_validation=value['cross_validation'],
        scope='Pretest source-heldout CV hyperparameter selection; not an independent validation/test claim',
        upstream_saw_held_sources_and_validation_labels=True)
    for name, row in value['selections'].items():
        if row['policy'] == 'loso':
            _require(row['epoch'] == value['cross_validation'][row['objective']]['selected_epoch'],
                     'S1/S2 LOSO aliases do not use the fixed shared objective epoch')
    return overview, path


def _percent(value):
    return '—' if value is None else f'{100 * value:.1f}%'


def render_report(summary, out):
    """HTML and exportable scientific SVG/PNG; no screenshot/image inference."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out = Path(out); out.mkdir(parents=True, exist_ok=True); esc = html.escape
    primary = {m: v for m, v in summary['methods'].items() if v['pool_budget'] in (None, 16)}
    names = list(primary)
    fig, ax = plt.subplots(figsize=(12, max(4, .30 * len(names) + 1)))
    ax.barh(np.arange(len(names)), [100 * primary[m]['accuracy'] for m in names], color='#3677a8')
    for i, m in enumerate(names):
        ax.plot(np.asarray(primary[m]['ci95']) * 100, [i, i], color='#243246', lw=1)
    ax.set(yticks=np.arange(len(names)), yticklabels=names, xlim=(0, 100), xlabel='Any-region point hit rate (%)')
    ax.invert_yaxis(); ax.grid(axis='x', alpha=.2); fig.tight_layout()
    for ext in ('svg', 'png'):
        fig.savefig(out / ('accuracy.' + ext), dpi=150)
    plt.close(fig)
    rows = []
    for name, m in summary['methods'].items():
        t = m.get('top4', {})
        rows.append(f'<tr><td>{esc(name)}</td><td>{esc(str(m["condition"]))}</td><td>{m["hits"]}/{m["n"]}</td>'
            f'<td>{_percent(m["accuracy"])}</td><td>{m.get("pool_available", "—")}</td>'
            f'<td>{_percent(m.get("conditional_selection_accuracy"))}</td><td>{t.get("oracle", "—")}</td>'
            f'<td>{_percent(t.get("conditional_retention"))}</td><td>{t.get("gui_misselections", "—")}</td></tr>')
    pairs = ''.join(f'<tr><td>{esc(p["method"])}</td><td>{esc(p["reference"])}</td><td>{esc(p["role"])}</td>'
        f'<td>{p["wins"]}/{p["losses"]}</td><td>{100*p["delta"]:+.2f}</td>'
        f'<td>[{100*p["ci95"][0]:.2f}, {100*p["ci95"][1]:.2f}]</td></tr>' for p in summary['paired_comparisons'])
    domain_rows = ''.join(f'<tr><td>{esc(m)}</td><td>{esc(s)}</td><td>{v["hits"]}/{v["n"]}</td>'
        f'<td>{_percent(v["accuracy"])}</td></tr>' for m, metric in summary['methods'].items()
        for s, v in metric['by_source'].items())
    pool_rows = ''.join(f'<tr><td>{c}</td><td>{k}</td><td>{p["oracle"]}/{p["n"]}</td>'
        f'<td>{p["empty"]}</td><td>{esc(json.dumps(p["candidate_size_counts"]))}</td></tr>'
        for c, budgets in summary['pools'].items() for k, p in budgets.items())
    overlap = summary['crop_geometry']; cohort_label = '独立主测试' if summary['cohort'] == 'test' else '补充小样本，单列解释' if summary['cohort'] == 'supplement' else summary['cohort']
    page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>V13 多区域候选实验</title>
<style>body{{font:15px system-ui,sans-serif;color:#243246;max-width:1450px;margin:32px auto;padding:0 20px}}p{{line-height:1.65}}table{{border-collapse:collapse;font-size:12px;width:100%}}td,th{{border:1px solid #d8e1eb;padding:7px;text-align:left}}th{{background:#eef3f8}}.scroll{{overflow:auto}}img{{max-width:100%}}</style>
<h1>V13 多区域搜索与 Top4 压缩：{esc(cohort_label)}</h1>
<p>预定整体主比较为c3_top1_s1_loso对c1_top1_s1_val；同一对照的S2是稳健性。c3_val对c1_val为搜索消融，Top4训练及GUI后比较分别报告，不按测试挑选模型。</p>
<p>全部{summary['n']}题进入分母。点击点在任一有效标注区域内即命中，包含既有数值边界处理；不要求等于标注中心。c1/c3与各K均使用保存的真实候选池，空池计失败。</p>
<p>Top4 oracle表示保留四点中至少一个可点击真点；它与GUI最终选对是两个阶段。Top4训练没有GUI教师，也不代表已针对GUI易混淆负例优化。null为辅助信号，本实验没有测动作许可门控。</p>
<p>预算匹配指最终候选和验证图像计数。三个搜索crop需要更多上游Qwen计算；缓存GUI分数的重用不构成端到端延迟实测。配对区间按source分层、截图簇bootstrap {summary['bootstrap']['reps']}次，只反映冻结模型下的样本不确定性。</p>
<p>LOSO是评分器来源留出交叉验证，用宏平均曲线选择公共epoch；这些曲线不是独立测试分数。冻结上游已见过来源，且上游选模使用过相应验证标签，不能称全链路LODO。S1为固定主种子，S2单列，不按测试换模型。</p>
<img src="accuracy.svg" alt="K16和直接坐标命中率与95%区间">
<div class="scroll"><table><tr><th>方法</th><th>condition</th><th>命中</th><th>命中率</th><th>父池oracle</th><th>可达条件命中</th><th>Top4 oracle</th><th>可达条件保留</th><th>GUI可达误选</th></tr>{''.join(rows)}</table></div>
<h2>固定候选覆盖</h2><table><tr><th>condition</th><th>K</th><th>oracle</th><th>空池</th><th>实际点数分布</th></tr>{pool_rows}</table>
<p>第一crop与目标有面积交集{overlap['first_crop_overlap']}/{summary['n']}；任一crop有交集{overlap['any_crop_overlap']}/{summary['n']}；新增crop带来区域交集{overlap['additional_crop_new_overlap']}。这是事后标注分析，区域重叠不等于已解码出正确点击点。</p>
<h2>配对命中差值</h2><div class="scroll"><table><tr><th>方法</th><th>对照</th><th>角色</th><th>赢/输</th><th>差值pp</th><th>95%区间pp</th></tr>{pairs}</table></div>
<h2>来源分项</h2><div class="scroll"><table><tr><th>方法</th><th>source</th><th>命中</th><th>命中率</th></tr>{domain_rows}</table></div>
<p>完整Top4 oracle配对区间、各阶段失败和实际验证数量见<a href="summary.json">summary.json</a>；逐题池、点和crop重叠见<a href="details.json">details.json</a>。本页complete只说明输入记录与统计完整；实验隔离需另行审计。</p></html>'''
    (out / 'report.html').write_text(page, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=OUT)
    parser.add_argument('--split', choices=tuple(EXPECTED_COUNTS))
    args = parser.parse_args(); splits = [args.split] if args.split else list(EXPECTED_COUNTS)
    frozen_path = args.root / 'frozen.json'; frozen = _read(frozen_path)
    _require(frozen['complete'] and frozen['fresh_test_features_at_freeze'] == 0, 'Experiment is not frozen before testing')
    dependencies = [Path(__file__), Path(base.__file__)]
    sealed_sources = {Path(p).resolve(): digest for p, digest in frozen['files'].items()}
    for path in dependencies:
        _require(sealed_sources.get(path.resolve()) == sha256(path), 'Evaluation source was not sealed: ' + str(path))
    selection, selection_path = selection_overview(args.root)
    for split in splits:
        cases, predictions, pipelines, gui_records, paths = load_records(args.root, split)
        result = evaluate_records(cases, predictions, pipelines, gui_records=gui_records, cohort=split)
        paths.extend((frozen_path, selection_path))
        result['summary'].update(input_sha256={str(p): sha256(p) for p in paths}, evaluator_sha256=sha256(__file__),
            evaluator_dependencies={str(Path(base.__file__)): sha256(base.__file__)}, selection_overview=selection)
        dest = args.root / 'evaluation' / split
        write_json(dest / 'summary.json', result['summary']); write_json(dest / 'details.json', result['details'])
        (dest / 'scored_predictions.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n'
            for r in result['scored_predictions']), encoding='utf-8')
        result['audit'].update(input_sha256=result['summary']['input_sha256'], evaluator_sha256=sha256(__file__),
            evaluator_dependencies=result['summary']['evaluator_dependencies'], frozen_sha256=sha256(frozen_path),
            registered_method_completeness_checked=True, decoded_pool_trace_rebuilt=True,
            native_and_global_points_unchanged=True, sealed_evaluator_sources_verified=True,
            summary_sha256=sha256(dest / 'summary.json'), details_sha256=sha256(dest / 'details.json'),
            scored_predictions_sha256=sha256(dest / 'scored_predictions.jsonl'))
        write_json(dest / 'audit.json', result['audit'])
        render_report(result['summary'], dest)
        print(json.dumps(dict(complete=True, split=split, n=len(cases), methods=len(result['summary']['methods']))), flush=True)
    links = ''.join(f'<li><a href="{split}/report.html">{split} ({EXPECTED_COUNTS[split]}题，分别报告)</a></li>' for split in EXPECTED_COUNTS
        if (args.root / 'evaluation' / split / 'summary.json').exists())
    (args.root / 'evaluation' / 'report.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
        '<title>V13实验报告</title><body style="font:17px system-ui;max-width:1000px;margin:48px auto">'
        '<h1>V13 多区域搜索与候选压缩</h1><p>主测试与补充队列保持独立分母；训练/LOSO选模证据将在冻结后补充，'
        '此页不挑选epoch或种子。完整模型与数据隔离以独立审计为准。</p><ul>' + links + '</ul></body></html>', encoding='utf-8')


if __name__ == '__main__':
    main()
