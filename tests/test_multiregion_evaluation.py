import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from evaluate_multiregion_v13 import HEAD_NAMES, evaluate_records, render_report, selection_overview


def fixture():
    cases = [dict(sample_id=sid, source=source, pixel_sha256=sid, image_size=[100, 100],
                  target_geometry=dict(type='bbox', boxes=[target]))
             for sid, source, target in [('a', 'A', [.1, .1, .3, .3]),
                                         ('b', 'A', [.7, .7, .9, .9]),
                                         ('c', 'B', [.1, .1, .3, .3])]]
    p1 = [[[80, 80], [70, 70]], [[80, 80]], []]
    p3 = [[[80, 80], [70, 70], [60, 60], [50, 50], [40, 40], [20, 20]],
          [[20, 20], [80, 80]], []]
    pipelines = []
    predictions = []
    gui = {}
    for i, case in enumerate(cases):
        sid = case['sample_id']
        pools = {c: {str(k): [dict(point=p) for p in points[:k]] for k in (4, 16)}
                 for c, points in [('c1', p1[i]), ('c3', p3[i])]}
        pipelines.append(dict(sample_id=sid, pools=pools, local_views=[
            dict(transform=dict(box=[0, 0, 40, 40])), dict(transform=dict(box=[60, 60, 100, 100]))]))
        scores1 = [[2., 1.], [1.], []][i]
        scores3 = [[6., 5., 4., 3., 2., 1.], [2., 1.], []][i]
        scores4 = [[0., 5., 4., 3., 2., 2.5], [2., 1.], []][i]
        records = {','.join(map(str, p)): dict(point=p, log_odds=10. if p == ([20, 20] if i == 0 else [80, 80]) else 0.)
                   for p in p1[i] + p3[i]}
        gui[sid] = records
        for name, condition, scores in [('c1_top1_s1_val', 'c1', scores1),
                                         ('c3_top1_s1_val', 'c3', scores3),
                                         ('c3_top4_s1_val', 'c3', scores4)]:
            points = [v['point'] for v in pools[condition]['16']]
            predictions.append(dict(sample_id=sid, method=name, condition=condition, budget=16,
                                    point=points[int(np.argmax(scores))] if points else None, scores=scores))
            if condition == 'c3':
                kept = [points[int(j)] for j in np.argsort(-np.asarray(scores), kind='stable')[:4]]
                values = [records[','.join(map(str, p))]['log_odds'] for p in kept]
                predictions.append(dict(sample_id=sid, method=name + '_top4_gui', condition=condition,
                    budget=16, retained_points=kept, n_verified=len(kept),
                    point=kept[int(np.argmax(values))] if kept else None))
        for condition in ('c1', 'c3'):
            for budget in (4, 16):
                points = [v['point'] for v in pools[condition][str(budget)]]
                values = [records[','.join(map(str, p))]['log_odds'] for p in points]
                predictions.append(dict(sample_id=sid, method='gui_' + condition, condition=condition,
                    budget=budget, point=points[int(np.argmax(values))] if points else None))
        predictions.append(dict(sample_id=sid, method='qwen_native_completed_lax', budget=None,
                                point=[20, 20] if i == 0 else None))
    return cases, predictions, pipelines, gui


def evaluated():
    c, p, pools, g = fixture()
    return evaluate_records(c, p, pools, gui_records=g, bootstrap_reps=40, cohort='test')


def test_true_condition_pools_top4_and_complete_failure_partitions():
    result = evaluated(); s = result['summary']
    assert s['n'] == 3
    assert s['pools']['c1']['16']['oracle'] == 1
    assert s['pools']['c3']['16']['oracle'] == 2
    assert s['coverage_changes']['16'] == dict(c1_oracle=1, c3_oracle=2, gained=1, lost=0,
                                               both=1, neither=1, delta=pytest.approx(1 / 3))
    a = s['methods']['c3_top1_s1_val__K16']
    b = s['methods']['c3_top4_s1_val__K16']
    assert a['hits'] == b['hits'] == 0
    assert a['top4']['oracle'] == 1 and b['top4']['oracle'] == 2
    compressed = s['methods']['c3_top1_s1_val_top4_gui__K16']
    assert compressed['hits'] == 1
    assert compressed['top4']['failure_counts'] == dict(compression_dropped_all_correct_points=1,
                                                       hit=1, empty_pool=1)
    fixed = s['methods']['c3_top4_s1_val_top4_gui__K16']
    assert fixed['hits'] == 2 and fixed['top4']['gui_misselections'] == 0
    assert fixed['top4']['total_candidates'] == 6
    assert fixed['top4']['candidate_count_distribution'] == {'4': 1, '2': 1, '0': 1}
    assert result == json.loads(json.dumps(result))  # Persistence must not change dictionary-key types.


def test_search_new_coverage_and_posthoc_crop_overlap_are_separate():
    s = evaluated()['summary']
    assert s['coverage_changes']['16']['gained'] == 1
    assert s['crop_geometry']['additional_crop_new_overlap'] == 1
    assert s['crop_geometry']['first_crop_overlap'] == 2
    assert s['crop_geometry']['any_crop_overlap'] == 3
    assert 'not end-to-end LODO' in s['loso_scope']


def test_pairwise_bootstrap_is_paired_and_seed_reproducible():
    a, b = evaluated(), evaluated()
    assert a['summary']['paired_comparisons'] == b['summary']['paired_comparisons']
    pair = next(p for p in a['summary']['paired_top4_oracle']
                if p['method'] == 'c3_top4_s1_val__K16' and p['reference'] == 'c3_top1_s1_val__K16')
    assert pair['wins'] == 1 and pair['losses'] == 0 and pair['delta'] == pytest.approx(1 / 3)
    assert a['summary']['bootstrap']['strata'] == 'source'


@pytest.mark.parametrize('kind', ['missing_condition', 'changed_condition', 'wrong_argmax', 'wrong_retained',
                                 'wrong_gui', 'missing_row', 'out_of_pool', 'fractional', 'duplicate_pool'])
def test_corrupt_predictions_and_pools_rejected(kind):
    c, p, pools, g = fixture()
    if kind == 'missing_condition':
        p[0].pop('condition')
    elif kind == 'changed_condition':
        next(r for r in p if r['sample_id'] == 'b' and r['method'] == 'c1_top1_s1_val')['condition'] = 'c3'
    elif kind == 'wrong_argmax':
        p[0]['point'] = [70, 70]
    elif kind == 'wrong_retained':
        next(r for r in p if r['method'] == 'c3_top4_s1_val_top4_gui')['retained_points'].reverse()
    elif kind == 'wrong_gui':
        next(r for r in p if r['method'] == 'gui_c3' and r['budget'] == 16)['point'] = [80, 80]
    elif kind == 'missing_row':
        p.pop()
    elif kind == 'out_of_pool':
        p[0]['point'] = [99, 99]
    elif kind == 'fractional':
        p[-1]['point'] = [20.1, 20]
    elif kind == 'duplicate_pool':
        pools[0]['pools']['c1']['16'].append(copy.deepcopy(pools[0]['pools']['c1']['16'][0]))
    with pytest.raises((ValueError, KeyError)):
        evaluate_records(c, p, pools, gui_records=g, bootstrap_reps=10)


def test_inclusive_any_region_polygon_and_empty_denominator():
    c, p, pools, g = fixture()
    c[0]['target_geometry'] = dict(type='polygon', polygons=[[[.8, .8], [.9, .8], [.9, .9], [.8, .9]],
                                                            [[.1, .1], [.3, .1], [.3, .3], [.1, .3]]])
    r = evaluate_records(c, p, pools, gui_records=g, bootstrap_reps=10)
    assert r['details'][0]['methods']['c1_top1_s1_val__K16']['hit'] is True
    assert r['summary']['methods']['c1_top1_s1_val__K16']['n'] == 3
    assert r['summary']['methods']['c1_top1_s1_val__K16']['empty_pools'] == 1


def test_stable_top4_ties_use_original_pool_order():
    c, p, pools, g = fixture()
    parent = next(r for r in p if r['method'] == 'c3_top1_s1_val' and r['sample_id'] == 'a')
    parent['scores'] = [1.] * 6
    # Existing Top4 is the first four original candidates, already matching the tied order.
    r = evaluate_records(c, p, pools, gui_records=g, bootstrap_reps=10)
    assert r['details'][0]['methods']['c3_top1_s1_val__K16']['top4_points'] == [
        [80, 80], [70, 70], [60, 60], [50, 50]]


def test_predeclared_overall_primary_is_not_confused_with_search_ablation():
    c, p, pools, g = fixture()
    extra = []
    for row in p:
        if row['method'].startswith('c3_top1_s1_val'):
            value = copy.deepcopy(row); value['method'] = value['method'].replace('_val', '_loso'); extra.append(value)
    r = evaluate_records(c, p + extra, pools, gui_records=g, bootstrap_reps=10)
    assert r['summary']['primary']['available']
    primary = next(pair for pair in r['summary']['paired_comparisons'] if pair['role'] == 'primary')
    assert primary['method'] == 'c3_top1_s1_loso__K16'
    assert primary['reference'] == 'c1_top1_s1_val__K16'
    assert any(pair['role'] == 'search_ablation' for pair in r['summary']['paired_comparisons'])


def test_independent_geometry_discloses_existing_numerical_boundary_difference():
    c, p, pools, g = fixture()
    c[0]['target_geometry'] = dict(type='bbox', boxes=[[.8000000000005, .8, .9, .9]])
    r = evaluate_records(c, p, pools, gui_records=g, bootstrap_reps=10)
    assert r['details'][0]['methods']['c1_top1_s1_val__K16']['hit'] is True
    differences = r['audit']['independent_geometry']['numerical_boundary_differences']
    assert any(d['point'] == [80, 80] and d['canonical'] and not d['shapely_covers'] for d in differences)
    assert r['audit']['model_execution'] is False


def test_selection_overview_requires_both_seeds_to_use_shared_loso_epoch(tmp_path):
    value = dict(complete=True, fresh_test_read=False, seed_selected_by_test=False,
                 cross_validation={'top1': {'selected_epoch': 1}, 'top4': {'selected_epoch': 2}}, selections={})
    for name in HEAD_NAMES:
        condition, objective, seed, policy = name.split('_')
        epoch = value['cross_validation'][objective]['selected_epoch'] if policy == 'loso' else 0
        value['selections'][name] = dict(condition=condition, objective=objective, policy=policy,
                                        seed=seed, epoch=epoch, validation={})
    folder = tmp_path / 'models'; folder.mkdir(); path = folder / 'selection.json'
    path.write_text(json.dumps(value), encoding='utf-8')
    overview, _ = selection_overview(tmp_path)
    assert len(overview['models']) == 10
    value['selections']['c3_top1_s2_loso']['epoch'] = 7
    path.write_text(json.dumps(value), encoding='utf-8')
    with pytest.raises(ValueError, match='shared objective epoch'):
        selection_overview(tmp_path)


def test_report_uses_separate_search_compression_and_loso_scope(tmp_path):
    s = evaluated()['summary']; render_report(s, tmp_path)
    text = (tmp_path / 'report.html').read_text(encoding='utf-8')
    assert 'Top4 oracle' in text and '全链路LODO' in text and 'null为辅助信号' in text
    assert (tmp_path / 'accuracy.svg').exists() and (tmp_path / 'accuracy.png').stat().st_size > 0
