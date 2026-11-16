import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import audit_multiregion_v13 as audit
from clickagent.internal_candidate_scorer import InternalCandidateScorerConfig, internal_inputs_to_torch, subset_internal_inputs
from clickagent.multiregion_scorer import MultiRegionScorer, build_multiregion_inputs


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads(); torch.set_num_threads(4)
    yield
    torch.set_num_threads(previous)


def save(path, arrays):
    with Path(path).open('wb') as stream:
        np.savez_compressed(stream, **arrays)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def raw(seed, case, transform=None):
    rng = np.random.default_rng(seed); n = 6; t = 2
    arrays = dict(target_attention=rng.random((t, 96, n)).astype(np.float32) * .01,
        target_hidden=rng.normal(size=(t, 12288)).astype(np.float16),
        image_hidden=rng.normal(size=(n, 4096)).astype(np.float16),
        vision_hidden=rng.normal(size=(4*n, 1152)).astype(np.float16))
    meta = dict(sample_id=case['sample_id'], image_sha256=case['image_sha256'], grid_shape=[2, 3],
        image_size=transform['crop_size'] if transform else case['image_size'], own_token_forward=True,
        all_backbone_parameters_frozen=True, ground_truth_in_model_input=False,
        instruction_sha256=hashlib.sha256(case['instruction'].encode()).hexdigest())
    if transform:
        meta['transform'] = transform
    return arrays, meta


def cache_fixture(tmp_path, condition='c3', split='test'):
    case = dict(sample_id='toy', source='domain', dataset='domain', image_sha256='toy-image',
                image_size=[1000, 600], instruction='Click Save',
                target_geometry=dict(type='bbox', boxes=[[.05, .05, .2, .3]]))
    ga, gm = raw(1, case); gp = tmp_path / 'global.npz'
    save(gp, dict(ga, metadata=np.asarray(json.dumps(gm))))
    views = []; local_raw = []
    for i, coordinates in enumerate(([0, 0, 400, 400], [400, 100, 800, 500], [600, 300, 1000, 600])):
        transform = dict(box=coordinates, crop_size=[800, 2*(coordinates[3]-coordinates[1])], scale=2)
        a, m = raw(i+2, case, transform); lp = tmp_path / f'local_{i}.npz'
        save(lp, dict(a, metadata=np.asarray(json.dumps(m))))
        views.append(dict(feature_path=str(lp), feature_sha256=digest(lp), transform=transform))
        local_raw.append((a, m, transform))
    points = np.asarray([[100, 100], [500, 300], [900, 500]], np.int64)
    pipeline = dict(global_feature_path=str(gp), global_feature_sha256=digest(gp), local_views=views,
                    pools={c: {str(k): [dict(point=p) for p in points.tolist()] for k in audit.BUDGETS}
                           for c in ('c1', 'c3')})
    pp = tmp_path / 'pipeline.json'; pp.write_text(json.dumps(pipeline), encoding='utf-8')
    selected = local_raw[:1] if condition == 'c1' else local_raw
    inputs = build_multiregion_inputs(ga, gm, selected, points, case['image_size'])
    evidence = {str(pp): digest(pp), str(gp): digest(gp)}
    evidence.update({v['feature_path']: v['feature_sha256'] for v in views[:1 if condition == 'c1' else 3]})
    arrays = dict(sample_id=np.asarray('toy'), available=np.asarray(True), points=points,
        pipeline_sha256=np.asarray(digest(pp)), annotation_derived_coordinates=np.asarray(False),
        cache_fingerprint=np.asarray(audit.cache_fingerprint(case, condition, evidence, split in ('train', 'val'))))
    arrays.update({'i__'+name: value for name, value in inputs.items()})
    arrays.update({'pool_indices_'+str(k): np.arange(len(points), dtype=np.int64) for k in audit.BUDGETS})
    if split in ('train', 'val'):
        arrays['labels'] = np.asarray([1, 0, 0], np.float32)
    cp = tmp_path / 'cache.npz'; save(cp, arrays)
    return case, pipeline, pp, cp, evidence, arrays, inputs


def test_cv_partition_removes_entire_cross_source_near_component():
    train = [dict(sample_id='held', dataset='H', image_fold_group=0),
             dict(sample_id='cross', dataset='B', image_fold_group=0),
             dict(sample_id='allowed', dataset='B', image_fold_group=1)]
    val = [dict(sample_id='vH', dataset='H'), dict(sample_id='vB', dataset='B')]
    assert audit.cv_partition(train, val, 'H') == (['allowed'], ['vH'], ['cross'])


def test_near_hash_group_exclusion_cannot_split_matching_images():
    values = [dict(pixel_sha256='a', dhash256='0', image_fold_group=0),
              dict(pixel_sha256='b', dhash256='1', image_fold_group=1)]
    with pytest.raises(ValueError, match='Near image split'):
        audit.validate_components(values)
    values[1]['image_fold_group'] = 0
    assert audit.validate_components(values)['near_pairs_checked'] == 1


def history_fixture():
    ids = ['a', 'b']; eligible = {sid: {str(k): 1 for k in audit.BUDGETS} for sid in ids}
    schedules = [audit.expected_schedule(ids, eligible, ids, e) for e in range(3)]
    history = [dict(epoch=e, queries=len(plan), optimizer_steps=int(bool(plan)),
        schedule_sha256=audit.digest_object(plan), train_loss=.5 if plan else None,
        validation=dict(n=2, named=2, hits=1, top4_hits=1, oracle=2, loss=.5))
        for e, plan in enumerate(schedules)]
    return history, schedules


def test_actual_schedule_query_and_update_counts_are_recomputed():
    history, schedules = history_fixture()
    audit.validate_history(history, epochs=2, n_validation=2, schedules=schedules, accumulation=8)
    assert history[0]['queries'] == history[0]['optimizer_steps'] == 0
    assert audit.expected_schedule(['a', 'b'], {'a': {str(k): 1 for k in audit.BUDGETS},
        'b': {str(k): 0 for k in audit.BUDGETS}}, ['a', 'b'], 1) == [
        p for p in schedules[1] if p[0] == 'a']


@pytest.mark.parametrize('field,value', [('queries', 1), ('optimizer_steps', 2), ('schedule_sha256', 'bad')])
def test_false_training_history_is_rejected(field, value):
    history, schedules = history_fixture(); history[1][field] = value
    with pytest.raises(ValueError):
        audit.validate_history(history, epochs=2, n_validation=2, schedules=schedules, accumulation=8)


def test_fixed_replay_sample_is_order_invariant_and_never_filters_empty_outcomes():
    cases = [dict(sample_id=f'{source}{i}', source=source, point=None, oracle=False)
             for source in ('A', 'B') for i in range(5)]
    a = audit.replay_sample(cases, 2); b = audit.replay_sample(cases[::-1], 2)
    assert [c['sample_id'] for c in a] == [c['sample_id'] for c in b]
    assert len(a) == 4 and all(c['point'] is None for c in a)


def test_ledger_rejects_mutation_and_unsealed_sidecar(tmp_path):
    p = tmp_path / 'evidence'; p.write_text('original', encoding='utf-8'); ledger = audit.Ledger()
    old = ledger.need(p); p.write_text('changed longer', encoding='utf-8')
    with pytest.raises(ValueError, match='changed'):
        ledger.need(p, old)
    with pytest.raises(ValueError, match='not sealed'):
        audit.require_sealed(p, {'files': {}}, audit.Ledger())


def test_registered_tolerance_accepts_small_arithmetic_difference():
    points = np.asarray([[1, 1], [2, 2]])
    saved = dict(scores=[1., .5], selected_index=0, point=[1, 1], scoring_device='cuda:0')
    result = audit.compare_replay(np.asarray([1.00004, .5], np.float32), saved, points)
    assert result['selected_point_exact'] and 0 < result['max_abs_difference'] < 1e-4
    assert result['atol'] == 1e-4 and result['rtol'] == 1e-5
    with pytest.raises(ValueError, match='tolerance'):
        audit.compare_replay(np.asarray([1.01, .5]), saved, points)


def test_tolerance_cannot_excuse_changed_point_even_on_near_tie():
    saved = dict(scores=[1., 1.00001], selected_index=1, point=[2, 2])
    with pytest.raises(ValueError, match='argmax point'):
        audit.compare_replay(np.asarray([1.00001, 1.]), saved, np.asarray([[1, 1], [2, 2]]))


def test_top4_near_order_change_is_disclosed_without_substitution():
    saved = dict(scores=[5., 2., 1.99999, 1., 0.], selected_index=0, point=[0, 0])
    result = audit.compare_replay(np.asarray([5., 1.99999, 2., 1., 0.]), saved,
                                  np.asarray([[i, i] for i in range(5)]))
    assert result['selected_point_exact'] and result['top4_order_exact'] is False
    assert result['saved_top4_indices'] == [0, 1, 2, 3]


@pytest.mark.parametrize('condition', ['c1', 'c3'])
def test_raw_rebuild_and_real_cpu_forward_exact_on_all_cached_inputs(tmp_path, condition):
    case, pipeline, pp, cp, evidence, arrays, inputs = cache_fixture(tmp_path, condition)
    sizes = audit.check_cache(cp, case, condition, pipeline, pp, evidence, 'test', audit.Ledger(), digest(cp))
    assert sizes['16'] == 3
    actual, points, ix, proof = audit.rebuild_inputs(case, pipeline, condition, cp, audit.Ledger())
    assert set(proof['exact_input_arrays']) == set(inputs)
    assert not proof['empty_cache_has_no_i_arrays']
    model = MultiRegionScorer(InternalCandidateScorerConfig(width=16, head_width=2, dropout=0)).eval()
    with torch.inference_mode():
        out = model(**internal_inputs_to_torch(subset_internal_inputs(actual, ix)))
    s = out['scores'].numpy(); pick = int(s.argmax())
    saved = dict(scores=s.tolist(), selected_index=pick, point=points[ix[pick]].tolist())
    assert audit.compare_replay(s, saved, points[ix])['max_abs_difference'] == 0


@pytest.mark.parametrize('corruption', ['feature', 'test_labels', 'coordinates', 'train_label', 'input_label'])
def test_rehashed_corrupt_cache_cannot_bypass_independent_checks(tmp_path, corruption):
    split = 'train' if corruption == 'train_label' else 'test'
    case, pipeline, pp, cp, evidence, arrays, _ = cache_fixture(tmp_path, 'c3', split)
    if corruption == 'feature':
        arrays['i__query_hidden'] = arrays['i__query_hidden'] + np.float16(.25)
    elif corruption == 'test_labels':
        arrays['labels'] = np.ones(3, np.float32)
    elif corruption == 'coordinates':
        arrays['points'] = arrays['points'].copy(); arrays['points'][0, 0] += 1
    elif corruption == 'train_label':
        arrays['labels'] = 1-arrays['labels']
    else:
        arrays['i__labels'] = np.ones(3, np.float32)
    save(cp, arrays)
    with pytest.raises(ValueError):
        if corruption == 'feature':
            audit.rebuild_inputs(case, pipeline, 'c3', cp, audit.Ledger())
        else:
            audit.check_cache(cp, case, 'c3', pipeline, pp, evidence, split, audit.Ledger(), digest(cp))


def test_empty_fixed_sample_still_performs_real_zero_candidate_cpu_forward(tmp_path):
    case, pipeline, pp, cp, evidence, arrays, _ = cache_fixture(tmp_path, 'c3')
    for condition in ('c1', 'c3'):
        pipeline['pools'][condition] = {str(k): [] for k in audit.BUDGETS}
    pp.write_text(json.dumps(pipeline), encoding='utf-8'); evidence[str(pp)] = digest(pp)
    arrays = {k: v for k, v in arrays.items() if not k.startswith('i__')}
    arrays.update(points=np.empty((0, 2), np.int64), available=np.asarray(False),
        pipeline_sha256=np.asarray(digest(pp)),
        cache_fingerprint=np.asarray(audit.cache_fingerprint(case, 'c3', evidence, False)))
    for k in audit.BUDGETS:
        arrays['pool_indices_'+str(k)] = np.empty(0, np.int64)
    save(cp, arrays)
    audit.check_cache(cp, case, 'c3', pipeline, pp, evidence, 'test', audit.Ledger(), digest(cp))
    inputs, points, ix, proof = audit.rebuild_inputs(case, pipeline, 'c3', cp, audit.Ledger())
    assert proof['empty_cache_has_no_i_arrays'] and proof['exact_input_arrays'] == []
    model = MultiRegionScorer(InternalCandidateScorerConfig(width=16, head_width=2, dropout=0)).eval()
    with torch.inference_mode():
        out = model(**internal_inputs_to_torch(subset_internal_inputs(inputs, ix)))
    assert out['scores'].numel() == 0 and torch.isfinite(out['null_scores']).all()
    assert audit.compare_replay(out['scores'].numpy(), dict(scores=[], point=None), points[ix])['selected_point_exact']


def test_rehashed_raw_with_gt_flag_is_rejected_before_forward(tmp_path):
    case, pipeline, _, cp, _, _, _ = cache_fixture(tmp_path, 'c3')
    path = Path(pipeline['global_feature_path'])
    with np.load(path, allow_pickle=False) as z:
        values = {k: z[k].copy() for k in z.files}
    meta = json.loads(str(values['metadata'])); meta['ground_truth_in_model_input'] = True
    values['metadata'] = np.asarray(json.dumps(meta)); save(path, values)
    pipeline['global_feature_sha256'] = digest(path)
    with pytest.raises(ValueError, match='GT-free'):
        audit.rebuild_inputs(case, pipeline, 'c3', cp, audit.Ledger())
