import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest


SPEC = importlib.util.spec_from_file_location('crossfit_eval_test_module', Path(__file__).parents[1] / 'scripts/evaluate_crossfit_candidates.py')
evaluation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluation)


def records():
    cases = [dict(sample_id=sid, source=source, pixel_sha256=pixel, image_size=[100, 100],
                  experiment_split='test', target_geometry=geometry) for sid, source, pixel, geometry in [
        ('a', 'A', 'same', {'type': 'bbox', 'boxes': [[.1, .1, .2, .2], [.8, .8, .9, .9]]}),
        ('b', 'A', 'other', {'type': 'bbox', 'boxes': [[.8, .8, .9, .9]]}),
        ('c', 'B', 'triangle', {'type': 'polygon', 'polygons': [[[.1, .1], [.2, .1], [.1, .2]]]}),
        ('d', 'B', 'empty', {'type': 'bbox', 'boxes': [[.1, .1, .2, .2]]}),
        ('e', 'A', 'same', {'type': 'bbox', 'boxes': [[.1, .1, .2, .2], [.8, .8, .9, .9]]}),
    ]]
    p16 = {'a': [[15, 15], [50, 50]], 'b': [[50, 50]], 'c': [[10, 10], [50, 50]], 'd': [], 'e': [[15, 15], [50, 50]]}
    p4 = {'a': [[85, 85]], 'b': [[85, 85]], 'c': [[10, 10]], 'd': [], 'e': [[85, 85]]}
    pools = [dict(sample_id=c['sample_id'], pools={str(k): [dict(point=p) for p in points[c['sample_id']]]
             for k, points in [(4, p4), (16, p16)]}) for c in cases]
    predictions = []
    for case in cases:
        sid = case['sample_id']
        for method in ('oof_s1', 'in_sample_s1'):
            point = p16[sid][0] if p16[sid] else None
            if sid == 'c' and method == 'in_sample_s1': point = [50, 50]
            predictions.append(dict(sample_id=sid, method=method, budget=16, point=point))
        predictions.append(dict(sample_id=sid, method='native', budget=None, point=[85, 85] if sid in ('a', 'b', 'e') else None))
    return cases, predictions, pools


def test_any_region_boundary_and_failure_partition():
    cases, predictions, pools = records()
    result = evaluation.evaluate_records(cases, predictions, pools, bootstrap_reps=100, bootstrap_seed=4)
    summary = result['summary']; primary = summary['paired_comparisons'][0]
    assert summary['methods']['oof_s1__K16']['hits'] == 3
    assert summary['methods']['in_sample_s1__K16']['hits'] == 2
    assert summary['methods']['native']['hits'] == 3
    assert summary['pools']['16']['oracle'] == 3 and summary['pools']['4']['oracle'] == 4
    assert primary['role'] == 'primary' and (primary['wins'], primary['losses']) == (1, 0)
    assert primary['reference_losses_breakdown'] == {'reachable_wrong_selection': 1}
    assert summary['methods']['oof_s1__K16']['failure_counts'] == {
        'hit': 3, 'nonempty_pool_without_correct_point': 1, 'empty_pool': 1}
    assert evaluation.point_hit([85, 85], cases[0])
    assert evaluation.point_hit([10, 10], cases[2])
    assert not evaluation.point_hit([50, 50], cases[0])


def test_cluster_bootstrap_reproducible_and_permutation_invariant():
    cases, predictions, pools = records()
    left = evaluation.evaluate_records(cases, predictions, pools, bootstrap_reps=100, bootstrap_seed=9)['summary']
    right = evaluation.evaluate_records(cases[::-1], predictions[::-1], pools[::-1], bootstrap_reps=100, bootstrap_seed=9)['summary']
    assert left == right
    assert left['bootstrap']['total_clusters'] == 4
    matrix = np.ones((len(cases), 2), bool)
    paired = evaluation.paired_analysis(cases, matrix, ['x', 'y'], reps=100, seed=9)[0]
    assert paired['ci95'] == [0., 0.] and paired['wins'] == paired['losses'] == 0


def test_validation_does_not_claim_unused_for_selection():
    cases, predictions, pools = records()
    for case in cases: case['experiment_split'] = 'val'
    assert evaluation.evaluate_records(cases, predictions, pools, bootstrap_reps=20)['summary']['selection_used_this_cohort'] is True


@pytest.mark.parametrize('bad', ['duplicate', 'missing', 'out_of_pool', 'noninteger', 'unknown', 'duplicate_pool'])
def test_bad_records_fail_closed(bad):
    cases, predictions, pools = records()
    if bad == 'duplicate': predictions.append(copy.deepcopy(predictions[0]))
    elif bad == 'missing': predictions.pop(0)
    elif bad == 'out_of_pool': predictions[0]['point'] = [2, 2]
    elif bad == 'noninteger': predictions[0]['point'] = [15.5, 15]
    elif bad == 'unknown': predictions[0]['sample_id'] = 'not-a-case'
    elif bad == 'duplicate_pool': pools[0]['pools']['16'].append(copy.deepcopy(pools[0]['pools']['16'][0]))
    with pytest.raises(ValueError): evaluation.evaluate_records(cases, predictions, pools, bootstrap_reps=20)


def test_retained_oracle_is_only_from_existing_points():
    cases, predictions, pools = records()
    for case in cases:
        sid = case['sample_id']; pool = next(p['pools']['16'] for p in pools if p['sample_id'] == sid)
        retained = [p['point'] for p in pool[:1]]
        predictions.append(dict(sample_id=sid, method='fixed_shortlist', budget=16,
                                point=retained[0] if retained else None, retained_points=retained, n_verified=len(retained)))
    result = evaluation.evaluate_records(cases, predictions, pools, bootstrap_reps=20)
    assert result['summary']['methods']['fixed_shortlist__K16']['retained_set']['oracle'] == 3
    predictions[-1]['retained_points'] = [[80, 80]]
    with pytest.raises(ValueError): evaluation.evaluate_records(cases, predictions, pools, bootstrap_reps=20)
