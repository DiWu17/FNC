"""CPU statistics for frozen V12 predictions, without model execution.

``evaluate_records(cases, predictions, pools)`` returns ``summary``, ``details``
and ``scored_predictions``. A prediction is {sample_id, method, budget, point};
budget=None denotes an unrestricted coordinate decoder. A non-null budget
declares that the chosen point belongs to that exact pool. Different budgets
are independent sets, never slices reconstructed from another budget.

This module checks the supplied records, not the experiment's freeze protocol.
It does not require a particular cohort size or the presence of primary heads.
Every observed method must cover every supplied case, including null outputs.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import html
import itertools
import json
import math
from numbers import Real
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

from clickagent.public_evaluation import point_in_geometry


OUT = Path('runs/crossfit_candidates_v12')
BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 20261102
PRIMARY = ('oof_s1__K16', 'in_sample_s1__K16')
ROBUSTNESS = ('oof_s2__K16', 'in_sample_s2__K16')


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def method_key(method, budget):
    _require(isinstance(method, str) and bool(method.strip()), 'Invalid method name')
    _require(budget is None or (isinstance(budget, int) and not isinstance(budget, bool) and budget > 0),
             'budget must be a positive integer or null')
    return method if budget is None else f'{method}__K{budget}'


def _point(value, size, context):
    if value is None:
        return None
    _require(isinstance(value, (list, tuple, np.ndarray)) and len(value) == 2, f'{context}: invalid point')
    _require(all(isinstance(v, Real) and not isinstance(v, (bool, np.bool_)) and
                 math.isfinite(float(v)) and float(v) == int(v) for v in value),
             f'{context}: point must contain finite integer original-image pixels')
    point = [int(value[0]), int(value[1])]
    _require(0 <= point[0] < size[0] and 0 <= point[1] < size[1], f'{context}: out-of-image point')
    return point


def _geometry(value):
    if value is None:
        return None
    _require(isinstance(value, dict) and value.get('type') in ('bbox', 'polygon'), 'Invalid target geometry')
    if value['type'] == 'bbox':
        parts = []
        for coordinates in value.get('boxes', []):
            a = np.asarray(coordinates, dtype=np.float64)
            _require(a.shape == (4,) and np.isfinite(a).all() and ((a >= 0) & (a <= 1)).all()
                     and a[0] < a[2] and a[1] < a[3], 'Invalid normalized target bbox')
            parts.append(box(*a))
    else:
        parts = []
        for coordinates in value.get('polygons', []):
            a = np.asarray(coordinates, dtype=np.float64)
            _require(a.ndim == 2 and a.shape[0] >= 3 and a.shape[1] == 2 and np.isfinite(a).all()
                     and ((a >= 0) & (a <= 1)).all(), 'Invalid normalized target polygon')
            polygon = Polygon(a)
            _require(polygon.is_valid and polygon.area > 0, 'Invalid or zero-area target polygon')
            parts.append(polygon)
    _require(bool(parts), 'Target geometry requires at least one region')
    result = unary_union(parts)
    _require(result.is_valid and not result.is_empty and result.area > 0, 'Invalid target union')
    return value


def _hit(point, size, geometry):
    return bool(point is not None and geometry is not None and
                point_in_geometry([point[0] / size[0], point[1] / size[1]], geometry))


def point_hit(point, case):
    """Any valid region covers the point, including polygon/bbox boundaries."""
    return _hit(_point(point, case['image_size'], case['sample_id']),
                case['image_size'], _geometry(case['target_geometry']))


def _source(case):
    source = case.get('source') or case.get('dataset')
    _require(isinstance(source, str) and bool(source), 'A case needs source or dataset for stratification')
    return source


def _cluster(case):
    for name in ('bootstrap_cluster_id', 'screenshot_cluster_id', 'pixel_sha256',
                 'image_sha256', 'group_id', 'image_id', 'image'):
        if case.get(name):
            return str(case[name])
    return str(case['sample_id'])


def _bootstrap_bundle(matrix, cases, reps, seed):
    values = np.asarray(matrix, dtype=np.float64)
    _require(values.ndim == 2 and values.shape[0] == len(cases) and len(cases) > 0,
             'Bootstrap matrix/case alignment mismatch')
    _require(np.isfinite(values).all(), 'Nonfinite bootstrap observations')
    _require(isinstance(reps, int) and not isinstance(reps, bool) and reps > 0, 'Invalid bootstrap repetitions')
    strata = defaultdict(lambda: defaultdict(list))
    for i, case in enumerate(cases):
        strata[_source(case)][_cluster(case)].append(i)
    rng = np.random.default_rng(seed)
    numerator = np.zeros((reps, values.shape[1])); denominator = np.zeros(reps)
    source_samples = {}; cluster_counts = {}
    for source in sorted(strata):
        groups = [strata[source][k] for k in sorted(strata[source])]
        cluster_counts[source] = len(groups)
        hits = np.stack([values[g].sum(0) for g in groups])
        sizes = np.asarray([len(g) for g in groups])
        local_num = np.zeros_like(numerator); local_den = np.zeros(reps)
        # Bound the temporary [replicate,cluster,method] array even for train cohorts.
        for start in range(0, reps, 128):
            stop = min(start + 128, reps)
            draws = rng.integers(0, len(groups), size=(stop - start, len(groups)))
            local_num[start:stop] = hits[draws].sum(1)
            local_den[start:stop] = sizes[draws].sum(1)
        numerator += local_num; denominator += local_den
        source_samples[source] = local_num / local_den[:, None]
    return dict(overall=numerator / denominator[:, None], by_source=source_samples,
                metadata=dict(reps=reps, seed=int(seed), unit='screenshot cluster', strata='source',
                              cluster_counts=cluster_counts, total_clusters=sum(cluster_counts.values()),
                              interval='paired percentile 95%',
                              uncertainty='conditional on frozen trained models; not training-seed uncertainty'))


def bootstrap_accuracies(matrix, cases, reps=BOOTSTRAP_REPS, seed=BOOTSTRAP_SEED):
    """Source-stratified screenshot-cluster accuracy draws shared by all methods."""
    return _bootstrap_bundle(matrix, cases, reps, seed)['overall']


def _ci(samples):
    return np.quantile(samples, [.025, .975]).tolist()


def _pair_order(names):
    pairs = []
    def add(a, b, role):
        if a in names and b in names and a != b and not any({a, b} == {x, y} for x, y, _ in pairs):
            pairs.append((a, b, role))
    add(*PRIMARY, 'primary'); add(*ROBUSTNESS, 'robustness')
    for seed in ('s1', 's2'):
        for budget in (4, 8, 32):
            add(f'oof_{seed}__K{budget}', f'in_sample_{seed}__K{budget}', 'secondary_budget')
    for name in names:
        if name in ('qwen_native_completed_lax', 'native', 'qwen_native', 'gui_actor__K16'):
            add(PRIMARY[0], name, 'reference')
    for seed in ('s1', 's2'):
        add(f'oof_{seed}_top4_gui__K16', f'in_sample_{seed}_top4_gui__K16', 'secondary_shortlist')
        for arm in ('oof', 'in_sample'):
            shortlist = f'{arm}_{seed}_top4_gui__K16'
            for reference in ('gui_actor__K4', 'gui_actor__K16', f'{arm}_{seed}__K16'):
                add(shortlist, reference, 'secondary_shortlist')
    for a, b in itertools.combinations(names, 2):
        add(a, b, 'descriptive_pairwise')
    return pairs


def paired_analysis(cases, hit_matrix, method_names, *, reps=BOOTSTRAP_REPS,
                    seed=BOOTSTRAP_SEED, bootstrap=None):
    """Return paired wins/losses and cluster bootstrap differences for every pair.

    Rows of hit_matrix follow cases and columns follow method_names. Differences
    are method minus reference. Supplying an internally computed bootstrap
    bundle avoids repeating draws when this is called by evaluate_records.
    """
    matrix = np.asarray(hit_matrix)
    _require(matrix.shape == (len(cases), len(method_names)) and np.isin(matrix, [0, 1]).all(),
             'Expected aligned binary hit matrix')
    _require(len(set(method_names)) == len(method_names), 'Repeated paired method name')
    bundle = _bootstrap_bundle(matrix, cases, reps, seed) if bootstrap is None else bootstrap
    result = []
    for method, reference, role in _pair_order(method_names):
        a, b = method_names.index(method), method_names.index(reference)
        difference = matrix[:, a].astype(int) - matrix[:, b].astype(int)
        item = dict(method=method, reference=reference, role=role, n=len(cases),
                    wins=int((difference > 0).sum()), losses=int((difference < 0).sum()),
                    both_hit=int(((matrix[:, a] == 1) & (matrix[:, b] == 1)).sum()),
                    both_miss=int(((matrix[:, a] == 0) & (matrix[:, b] == 0)).sum()),
                    delta=float(difference.mean()), ci95=_ci(bundle['overall'][:, a] - bundle['overall'][:, b]),
                    by_source={})
        for source in sorted(bundle['by_source']):
            indices = [i for i, c in enumerate(cases) if _source(c) == source]
            diff = difference[indices]; draws = bundle['by_source'][source]
            item['by_source'][source] = dict(n=len(indices), wins=int((diff > 0).sum()),
                losses=int((diff < 0).sum()), delta=float(diff.mean()), ci95=_ci(draws[:, a] - draws[:, b]))
        result.append(item)
    return result


def evaluate_records(cases, predictions, pools, *, bootstrap_reps=BOOTSTRAP_REPS,
                     bootstrap_seed=BOOTSTRAP_SEED):
    """Score all cases and observed methods; no cohort/model count is hardcoded.

    Missing predictions are not silently dropped: each observed (method,budget)
    needs exactly one explicit row per case, using point=None for no output.
    Optional retained_points/n_verified metadata describes an already generated
    shortlist; it never changes predictions or constructs a new policy.
    """
    cases, predictions, pools = list(cases), list(predictions), list(pools)
    _require(bool(cases), 'No cases supplied')
    ids = [c['sample_id'] for c in cases]
    _require(all(isinstance(s, str) and s for s in ids) and len(set(ids)) == len(ids), 'Repeated/invalid case id')
    case_map = dict(zip(ids, cases)); geometry = {}; sizes = {}
    for c in cases:
        _source(c)
        size = c['image_size']
        _require(len(size) == 2 and all(isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in size),
                 'Invalid integer image_size')
        sizes[c['sample_id']] = size; geometry[c['sample_id']] = _geometry(c['target_geometry'])
    pool_map = {}
    for record in pools:
        sid = record['sample_id']
        _require(sid in case_map and sid not in pool_map, 'Unknown/repeated pool case')
        parsed = {}
        for budget, items in record['pools'].items():
            _require(isinstance(budget, str) and budget.isdigit() and int(budget) > 0 and str(int(budget)) == budget,
                     'Pool budgets must be canonical positive-integer strings')
            points = [_point(v['point'], sizes[sid], sid) for v in items]
            _require(all(p is not None for p in points) and len({tuple(p) for p in points}) == len(points),
                     'Null or repeated pool coordinate')
            _require(len(points) <= int(budget), 'Pool contains more points than its declared budget')
            labels = [_hit(p, sizes[sid], geometry[sid]) for p in points]
            parsed[budget] = dict(points=points, point_hits=labels, point_oracle=any(labels), n_points=len(points))
        pool_map[sid] = parsed
    _require(set(pool_map) == set(ids), 'Pool records must cover all supplied cases')
    budgets = sorted({b for p in pool_map.values() for b in p}, key=int)
    _require(all(set(v) == set(budgets) for v in pool_map.values()), 'Pool budgets differ between cases')
    pred_map = {}; descriptors = {}
    for row in predictions:
        sid = row['sample_id']; method, budget = row['method'], row.get('budget')
        _require(sid in case_map, 'Unknown prediction case')
        name = method_key(method, budget); token = (sid, name)
        _require(token not in pred_map, 'Repeated prediction for case/method/budget')
        _require(name not in descriptors or descriptors[name] == (method, budget), 'Method-key collision')
        point = _point(row['point'], sizes[sid], f'{sid}/{name}')
        if budget is not None:
            _require(str(budget) in pool_map[sid], 'Prediction references absent pool budget')
            _require(point is None or point in pool_map[sid][str(budget)]['points'], 'Selected point is not in declared pool')
            if 'scores' in row:
                scores = np.asarray(row['scores'], dtype=np.float64)
                points = pool_map[sid][str(budget)]['points']
                _require(scores.shape == (len(points),) and np.isfinite(scores).all(), 'Invalid saved candidate scores')
                if points and 'selected_index' in row:
                    selected = int(scores.argmax())
                    _require(row['selected_index'] == selected and point == points[selected],
                             'Saved argmax does not match the chosen point')
        value = dict(method=method, pool_budget=budget, point=point, hit=_hit(point, sizes[sid], geometry[sid]))
        if 'retained_points' in row:
            _require(budget is not None, 'A retained set needs a declared parent pool')
            retained = [_point(v, sizes[sid], f'{sid}/{name}/retained') for v in row['retained_points']]
            _require(all(v is not None and v in pool_map[sid][str(budget)]['points'] for v in retained)
                     and len({tuple(v) for v in retained}) == len(retained), 'Invalid retained candidate set')
            _require(point is None or point in retained, 'Chosen point is not in retained set')
            _require(row.get('n_verified', len(retained)) == len(retained), 'n_verified differs from retained size')
            value.update(retained_points=retained, n_verified=len(retained),
                         retained_point_oracle=any(_hit(v, sizes[sid], geometry[sid]) for v in retained))
        pred_map[token] = value; descriptors[name] = (method, budget)
    names = sorted(descriptors)
    _require(bool(names), 'No methods supplied')
    _require(len(pred_map) == len(cases) * len(names), 'Each observed method must include every case, including null outputs')
    details = []; scored = []
    for c in cases:
        sid = c['sample_id']; values = {}
        for name in names:
            v = dict(pred_map[(sid, name)]); budget = v['pool_budget']
            if budget is not None:
                pool = pool_map[sid][str(budget)]
                v.update(pool_oracle=pool['point_oracle'], n_points=pool['n_points'])
                v['failure_stage'] = ('hit' if v['hit'] else 'empty_pool' if pool['n_points'] == 0 else
                    'nonempty_pool_without_correct_point' if not pool['point_oracle'] else 'reachable_wrong_selection')
            else:
                v['failure_stage'] = 'hit' if v['hit'] else 'no_point' if v['point'] is None else 'wrong_point'
            values[name] = v
            scored.append(dict(sample_id=sid, source=_source(c), dataset=c.get('dataset'), method_key=name, **v))
        details.append(dict(sample_id=sid, source=_source(c), dataset=c.get('dataset'),
                            screenshot_cluster=_cluster(c), pools=pool_map[sid], methods=values))
    matrix = np.asarray([[d['methods'][m]['hit'] for m in names] for d in details], bool)
    oracle_matrix = np.asarray([[d['pools'][b]['point_oracle'] for b in budgets] for d in details], bool)
    bundle = _bootstrap_bundle(np.concatenate((matrix, oracle_matrix), 1), cases, bootstrap_reps, bootstrap_seed)
    method_bundle = dict(overall=bundle['overall'][:, :len(names)],
                         by_source={s: v[:, :len(names)] for s, v in bundle['by_source'].items()})
    sources = sorted({_source(c) for c in cases}); metrics = {}; pool_metrics = {}
    for j, name in enumerate(names):
        values = [d['methods'][name] for d in details]; budget = descriptors[name][1]
        hits = int(matrix[:, j].sum())
        m = dict(method=descriptors[name][0], pool_budget=budget, n=len(cases), hits=hits,
                 accuracy=hits / len(cases), ci95=_ci(bundle['overall'][:, j]),
                 points=sum(v['point'] is not None for v in values), abstentions=sum(v['point'] is None for v in values),
                 failure_counts=dict(Counter(v['failure_stage'] for v in values)), by_source={})
        if budget is not None:
            available = sum(v['pool_oracle'] for v in values)
            m.update(pool_available=available, conditional_selection_accuracy=hits / available if available else None,
                     empty_pools=sum(v['n_points'] == 0 for v in values),
                     nonempty_pool_without_correct_point=sum(v['n_points'] > 0 and not v['pool_oracle'] for v in values),
                     selector_missed_correct_point=sum(v['pool_oracle'] and not v['hit'] for v in values))
            _require(hits + m['empty_pools'] + m['nonempty_pool_without_correct_point'] +
                     m['selector_missed_correct_point'] == len(cases), 'Failure decomposition does not partition cases')
        if any('retained_points' in v for v in values):
            _require(all('retained_points' in v for v in values), 'Retained-set metadata must cover the entire method')
            counts = [v['n_verified'] for v in values]
            m['retained_set'] = dict(oracle=sum(v['retained_point_oracle'] for v in values),
                oracle_accuracy=sum(v['retained_point_oracle'] for v in values) / len(cases),
                total_candidates=sum(counts), mean_candidates=float(np.mean(counts)),
                candidate_count_distribution=dict(Counter(counts)),
                cost_scope='candidate image verification count proxy; not measured runtime or end-to-end speedup')
        for source in sources:
            ix = [i for i, c in enumerate(cases) if _source(c) == source]
            m['by_source'][source] = dict(n=len(ix), hits=int(matrix[ix, j].sum()),
                accuracy=float(matrix[ix, j].mean()), ci95=_ci(bundle['by_source'][source][:, j]))
        metrics[name] = m
    for j, budget in enumerate(budgets):
        values = [d['pools'][budget] for d in details]; ci_column = len(names) + j
        pool_metrics[budget] = dict(n=len(cases), oracle=sum(v['point_oracle'] for v in values),
            oracle_accuracy=float(oracle_matrix[:, j].mean()), ci95=_ci(bundle['overall'][:, ci_column]),
            empty=sum(v['n_points'] == 0 for v in values), total_points=sum(v['n_points'] for v in values),
            candidate_size_counts=dict(Counter(v['n_points'] for v in values)),
            by_source={s: dict(n=sum(d['source'] == s for d in details),
                oracle=sum(d['source'] == s and d['pools'][budget]['point_oracle'] for d in details)) for s in sources})
    comparisons = paired_analysis(cases, matrix, names, reps=bootstrap_reps,
                                  seed=bootstrap_seed, bootstrap=method_bundle)
    for comparison in comparisons:
        for field, method, reference in [('method_losses_breakdown', comparison['method'], comparison['reference']),
                                          ('reference_losses_breakdown', comparison['reference'], comparison['method'])]:
            losses = [d['methods'][method]['failure_stage'] for d in details
                      if not d['methods'][method]['hit'] and d['methods'][reference]['hit']]
            comparison[field] = dict(Counter(losses))
    summary = dict(complete=True, version=12, n=len(cases), methods=metrics, pools=pool_metrics,
        primary=dict(method=PRIMARY[0], reference=PRIMARY[1], available=all(m in names for m in PRIMARY)),
        robustness=dict(method=ROBUSTNESS[0], reference=ROBUSTNESS[1], available=all(m in names for m in ROBUSTNESS)),
        paired_comparisons=comparisons, bootstrap=bundle['metadata'],
        metric='Existing public_evaluation.point_in_geometry: integer original-image point inside any valid normalized bbox/polygon, inclusive edges with existing 1e-12 numerical boundary tolerance; no distance weighting or added pixel tolerance; every supplied case remains in the denominator.',
        completeness='Every observed method has one explicit prediction for every case; no fixed cohort/model count is enforced here.',
        audit_scope='Record/statistics validation only; freeze, training provenance and independent test selection require the separate experiment auditor.',
        cohort_splits=sorted({c.get('experiment_split', c.get('split', 'unknown')) for c in cases}),
        selection_used_this_cohort=(True if any(c.get('experiment_split', c.get('split')) == 'val' for c in cases)
                                   else False if all(c.get('experiment_split', c.get('split')) == 'test' for c in cases)
                                   else None))
    return dict(summary=summary, details=details, scored_predictions=scored)


def _format_percent(value):
    return '—' if value is None else f'{100 * value:.1f}%'


def render_report(summary, out):
    """Save standalone scientific SVG/PNG charts and HTML with local chart assets."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    methods = summary['methods']; names = list(methods)
    fig, ax = plt.subplots(figsize=(11, max(3, .35 * len(names) + 1)))
    y = np.arange(len(names)); means = [100 * methods[m]['accuracy'] for m in names]
    ax.barh(y, means, color=['#2768ac' if m == PRIMARY[0] else '#89a9ca' for m in names])
    for i, name in enumerate(names):
        lo, hi = np.asarray(methods[name]['ci95']) * 100
        ax.plot([lo, hi], [i, i], color='#222222', linewidth=1.2)
    ax.set(yticks=y, yticklabels=names, xlabel='Point hit rate (%)', xlim=(0, 100),
           title=f'All {summary["n"]} cases; screenshot-cluster 95% intervals')
    ax.invert_yaxis(); fig.tight_layout()
    for suffix in ('svg', 'png'): fig.savefig(out / f'accuracy.{suffix}', dpi=150)
    plt.close(fig)
    selected = [p for p in summary['paired_comparisons'] if p['role'] != 'descriptive_pairwise']
    if not selected: selected = summary['paired_comparisons'][:10]
    if selected:
        fig, ax = plt.subplots(figsize=(12, max(3, .4 * len(selected) + 1)))
        for i, pair in enumerate(selected):
            lo, hi = np.asarray(pair['ci95']) * 100
            ax.plot([lo, hi], [i, i], color='#326d9d'); ax.scatter(pair['delta'] * 100, i, color='#163e62')
        ax.axvline(0, color='#888888', linewidth=1)
        ax.set(yticks=np.arange(len(selected)),
               yticklabels=[p['method'] + ' - ' + p['reference'] for p in selected],
               xlabel='Paired point-hit difference (percentage points)', title='Shared cluster bootstrap; 95% intervals')
        ax.invert_yaxis(); fig.tight_layout()
        for suffix in ('svg', 'png'): fig.savefig(out / f'paired_delta.{suffix}', dpi=150)
        plt.close(fig)
    esc = html.escape
    rows = []
    for name, m in methods.items():
        ci = '–'.join(_format_percent(v) for v in m['ci95'])
        rows.append(f'<tr><td>{esc(name)}</td><td>{m["hits"]}/{m["n"]}</td><td>{_format_percent(m["accuracy"])}</td>'
                    f'<td>{ci}</td><td>{_format_percent(m.get("conditional_selection_accuracy"))}</td>'
                    f'<td>{m["abstentions"]}</td><td>{esc(json.dumps(m["failure_counts"]))}</td></tr>')
    pair_rows = []
    for p in summary['paired_comparisons']:
        ci = ' to '.join(f'{100 * v:+.2f}' for v in p['ci95'])
        pair_rows.append(f'<tr><td>{esc(p["method"])}</td><td>{esc(p["reference"])}</td><td>{esc(p["role"])}</td>'
                         f'<td>{p["wins"]}/{p["losses"]}</td><td>{100*p["delta"]:+.2f}</td><td>{ci}</td></tr>')
    source_rows = []
    for name, m in methods.items():
        for source, value in m['by_source'].items():
            source_rows.append(f'<tr><td>{esc(name)}</td><td>{esc(source)}</td><td>{value["hits"]}/{value["n"]}</td>'
                               f'<td>{_format_percent(value["accuracy"])}</td></tr>')
    retained_rows = []
    for name, m in methods.items():
        if 'retained_set' in m:
            r = m['retained_set']
            retained_rows.append(f'<tr><td>{esc(name)}</td><td>{r["oracle"]}/{m["n"]}</td>'
                                 f'<td>{m["hits"]}/{m["n"]}</td><td>{r["mean_candidates"]:.2f}</td>'
                                 f'<td>{r["total_candidates"]}</td></tr>')
    retained_table = ('<h2>已提供短名单的次要统计</h2><table><tr><th>方法</th><th>短名单点oracle</th>'
                      '<th>最终命中</th><th>平均验证候选数</th><th>总候选数</th></tr>' +
                      ''.join(retained_rows) + '</table>') if retained_rows else ''
    oracle = ', '.join(f'K{k}: {v["oracle"]}/{v["n"]}, empty {v["empty"]}' for k, v in summary['pools'].items())
    page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>V12 candidate evaluation</title>
<style>body{{font:15px system-ui,sans-serif;margin:32px auto;max-width:1400px;color:#233047;padding:0 20px}}h1,h2{{color:#173e64}}table{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{border:1px solid #d9e0e8;padding:7px;text-align:left}}th{{background:#eef3f8}}img{{max-width:100%}}.scroll{{overflow:auto}}p{{line-height:1.6}}</style>
<h1>V12 固定候选评测</h1><p>全部 {summary['n']} 题进入分母。命中任一有效标注区域即可，包含边界；不要求等于标注中心点。空池和无输出计失败。此报告不执行训练、挑选种子或重新构造候选。</p>
<p>主比较：{esc(PRIMARY[0])} vs {esc(PRIMARY[1])}；S2为稳健性，其他预算与参考为次要比较。{esc(oracle)}</p>
<p>区间按source分层、截图簇配对bootstrap {summary['bootstrap']['reps']}次。它反映冻结模型下的样本不确定性，不包含训练随机性。本脚本的complete只表示输入记录和统计完整，实验冻结与测试隔离需另行审计。</p>
<img src="accuracy.svg" alt="全部方法命中率与95%区间"><div class="scroll"><table><tr><th>方法</th><th>命中</th><th>命中率</th><th>95%区间</th><th>可达条件命中率</th><th>无输出</th><th>失败分解</th></tr>{''.join(rows)}</table></div>
<h2>配对比较</h2>{'<img src="paired_delta.svg" alt="配对差值与95%区间">' if selected else ''}
<div class="scroll"><table><tr><th>方法</th><th>对照</th><th>比较类型</th><th>赢/输</th><th>差值pp</th><th>95%区间pp</th></tr>{''.join(pair_rows)}</table></div>
<h2>来源切片</h2><div class="scroll"><table><tr><th>方法</th><th>source</th><th>命中</th><th>命中率</th></tr>{''.join(source_rows)}</table></div>
{retained_table}
<p>候选验证图像数量仅是计数代理，不等于实测延迟。详情和配对来源区间见 <a href="summary.json">summary.json</a>，逐题点、池及失败环节见 <a href="details.json">details.json</a>。</p></html>'''
    (out / 'report.html').write_text(page, encoding='utf-8')


def _read_jsonl(path):
    return [json.loads(s) for s in Path(path).read_text(encoding='utf-8-sig').splitlines() if s.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--pools', type=Path, required=True)
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--bootstrap-reps', type=int, default=BOOTSTRAP_REPS)
    parser.add_argument('--bootstrap-seed', type=int, default=BOOTSTRAP_SEED)
    args = parser.parse_args()
    inputs = [args.cases, args.predictions, args.pools]
    destinations = [args.out / n for n in ('summary.json', 'details.json', 'scored_predictions.jsonl', 'report.html')]
    _require(not {p.resolve() for p in inputs}.intersection(p.resolve() for p in destinations), 'Output would overwrite input')
    result = evaluate_records(_read_jsonl(args.cases), _read_jsonl(args.predictions), _read_jsonl(args.pools),
                              bootstrap_reps=args.bootstrap_reps, bootstrap_seed=args.bootstrap_seed)
    result['summary']['input_sha256'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    result['summary']['evaluator_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.out.mkdir(parents=True, exist_ok=True)
    for name in ('summary', 'details'):
        (args.out / (name + '.json')).write_text(json.dumps(result[name], ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (args.out / 'scored_predictions.jsonl').write_text(''.join(json.dumps(v, ensure_ascii=False) + '\n'
                                                    for v in result['scored_predictions']), encoding='utf-8')
    render_report(result['summary'], args.out)
    print(json.dumps(dict(complete=True, n=result['summary']['n'], methods=len(result['summary']['methods']),
                          out=str(args.out)), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
