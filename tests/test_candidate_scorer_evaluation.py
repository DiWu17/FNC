import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import evaluate_candidate_scorer as evaluation
import audit_candidate_scorer as audit
from clickagent.candidate_budget import build_pools, choose_point, coordinate_key


def dump(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def save(path, arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


@pytest.fixture
def toy(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation, 'OUT', tmp_path)
    monkeypatch.setattr(audit, 'OUT', tmp_path)
    monkeypatch.setattr(evaluation, 'EXPECTED_CASES', 3)
    dump(tmp_path / 'protocol.json', {'complete': True}); dump(tmp_path / 'frozen.json', {'complete': True, 'frozen_at': 0.})
    cohort = []; index_rows = []
    for n in range(3):
        sid = f'toy:{n}'; stem = evaluation.key(sid); image = tmp_path / f'image{n}.png'; image.write_bytes(str(n).encode())
        case = dict(sample_id=sid, dataset=['a', 'b', 'c'][n], experiment_split='test', image=str(image),
             image_sha256=evaluation.sha256(image), pixel_sha256='pixel' + str(n), image_size=[100, 100], instruction='target',
             target_geometry=dict(type='bbox', boxes=[[0, 0, .15, .15] if n == 0 else [.9, .9, 1., 1.]]))
        cohort.append(case)
        gp = dict(point=[10, 10], all_points=[[10, 10], [20, 20]], all_scores=[2., 1.], parent_indices=[0, 1], target_name='target') if n < 2 else dict(point=None, candidates=[], failure='no_name', target_name='')
        metadata = dict(sample_id=sid, image_sha256=case['image_sha256'], image_size=[100, 100], elapsed_seconds=1.,
                        instruction_sha256=hashlib.sha256(b'target').hexdigest(), own_token_forward=True,
                        all_backbone_parameters_frozen=True, ground_truth_in_model_input=False)
        raw = tmp_path / 'test/features' / (stem + '.npz'); save(raw, {'metadata': np.asarray(json.dumps(metadata))})
        pipeline = dict(sample_id=sid, global_result=gp, old_crop_result=None, old_crop_transform=None,
                        global_feature_path=str(raw), global_feature_sha256=evaluation.sha256(raw),
                        local_feature_path=None, local_feature_sha256=None, ground_truth_in_model_inputs=False)
        pipeline['pools'] = build_pools(pipeline)
        path = tmp_path / 'test/pipeline' / (stem + '.json'); dump(path, pipeline)
        methods = {}
        for name in evaluation.expected_head_methods():
            pool = pipeline['pools'][name.rsplit('_K', 1)[1]]
            if pool:
                scores = [1., 0.] if name.startswith(('internal_full_s1_', 'internal_full_s2_')) else [0., 1.]
                ix = int(np.argmax(scores)); null = 2. if n == 1 else -1.
                methods[name] = dict(point=pool[ix]['point'], selected_index=ix, scores=scores, null_score=null,
                                     null_would_reject=null > scores[ix], n_points=2, elapsed_cached_head_seconds=.01)
            else:
                methods[name] = dict(point=None, scores=[], null_score=None, n_points=0, elapsed_cached_head_seconds=0., failure='no_candidates')
        methods['old_critic_K16'] = dict(point=[10, 10] if n < 2 else None, scores=[1., 0.] if n < 2 else [])
        native_point = [[10, 10], [95, 95], None][n]
        methods['global_top1'] = dict(point=gp['point'])
        methods['qwen_native_completed_lax'] = dict(point=native_point)
        dump(tmp_path / 'predictions/test' / (stem + '.json'), dict(sample_id=sid, methods=methods, ground_truth_in_model_inputs=False))
        records = {coordinate_key(v['point']): dict(log_odds=float(i)) for i, v in enumerate(pipeline['pools']['16'])}
        dump(tmp_path / 'gui_reference/test' / (stem + '.json'), dict(sample_id=sid, records=records,
             choices={str(k): choose_point(pipeline['pools'][str(k)], records) for k in (4, 16)},
             batches=[dict(seconds=.5)] if n < 2 else [], ground_truth_in_model_inputs=False))
        dump(tmp_path / 'test/native' / (stem + '.json'), dict(completed_lax_point=native_point, elapsed_seconds=.4,
             generated_text=['CLICK 100 100', 'CLICK 960 960', 'WAIT'][n], terminated_by_eos=True))
        dump(tmp_path / 'ui_tars' / (stem + '.json'), dict(sample_id=sid, point=native_point, ground_truth_in_model_input=False))
        points = np.asarray([v['point'] for v in pipeline['pools']['32']], np.int64).reshape(-1, 2)
        arrays = dict(sample_id=np.asarray(sid), available=np.asarray(bool(len(points))), points=points,
                      annotation_derived_coordinates=np.asarray(False), pipeline_sha256=np.asarray(evaluation.sha256(path)))
        arrays.update({'pool_indices_' + str(k): np.arange(len(points), dtype=np.int64) for k in (4, 8, 16, 32)})
        internal_path = tmp_path / 'cache/internal/test' / (stem + '.npz')
        save(internal_path, dict(arrays, **({'i__signal': np.zeros((2, 1))} if n < 2 else {})))
        visual_path = tmp_path / 'cache/visual/test' / (stem + '.npz')
        if n < 2:
            encoding = tmp_path / 'vision_encodings' / (case['image_sha256'] + '.npz')
            save(encoding, dict(metadata=np.asarray(json.dumps(dict(image_sha256=case['image_sha256'], image_size=[100, 100],
                 all_visual_parameters_frozen=True, gui_actor_used=False, language_model_loaded=False, timing_seconds=dict(total=.2))))))
            arrays['visual_encoding_sha256'] = np.asarray(evaluation.sha256(encoding))
        save(visual_path, dict(arrays, **({'v__signal': np.zeros((2, 1))} if n < 2 else {})))
        index_rows.append(dict(sample_id=sid, path=str(internal_path), sha256=evaluation.sha256(internal_path),
             visual_path=str(visual_path), visual_sha256=evaluation.sha256(visual_path), available=n < 2, n_points=len(points),
             pipeline_path=str(path), pipeline_sha256=evaluation.sha256(path), raw_files={str(raw): evaluation.sha256(raw)}))
    (tmp_path / 'manifest.jsonl').write_text(''.join(json.dumps(c) + '\n' for c in cohort), encoding='utf-8')
    dump(tmp_path / 'cache/test_index.json', dict(complete=True, visual_complete=True, has_labels=False, split='test',
         real_candidate_pools=True, annotation_derived_coordinates=False, cases=index_rows))
    dump(tmp_path / 'predictions/test/completion.json', dict(complete=True, n=3, frozen_sha256=evaluation.sha256(tmp_path / 'frozen.json')))
    return tmp_path, cohort


def test_full_evaluation_and_all_choice_audit_include_every_case(toy, monkeypatch):
    root, cohort = toy
    predictions, details, summary = evaluation.evaluate_split('test')
    assert len(predictions) == 3 * 38 and len(details) == 3
    primary = summary['methods'][evaluation.PRIMARY]
    assert primary['hits'] == 1 and primary['empty_pools'] == 1
    assert primary['nonempty_pool_without_correct_point'] == 1 and primary['selector_missed_correct_point'] == 0
    assert primary['pool_available'] == 1 and primary['conditional_selection_accuracy'] == 1.
    assert summary['methods']['gui_actor_K16']['selector_missed_correct_point'] == 1
    assert summary['methods']['qwen_native_completed_lax']['hits'] == 2
    assert primary['optional_null_policy']['learned_rejections'] == 1
    comparison = next(p for p in summary['paired_comparisons'] if p['method'] == evaluation.PRIMARY and p['reference'] == 'gui_actor_K16')
    assert (comparison['wins'], comparison['losses'], comparison['ties']) == (1, 0, 2)
    monkeypatch.setattr(audit, 'replay', lambda *args: dict(complete=True, integer_points_exact=True))
    checked = audit.audit_split('test', {'frozen_at': 0.})
    assert checked['predictions_checked'] == 114 and checked['label_free_caches'] == 6 and checked['pools_checked'] == 12


@pytest.mark.parametrize('point,correct', [([0, 0], True), ([15, 15], True), ([16, 15], False), (None, False),
       ([100, 0], False), ([True, 2], False), ([1.5, 2], False), ([float('nan'), 0], False)])
def test_actual_region_metric_inclusive_edges_and_pixel_bounds(point, correct):
    case = dict(image_size=[100, 100], target_geometry=dict(type='bbox', boxes=[[0, 0, .15, .15]]))
    assert evaluation.point_hit(point, case) == audit.independent_hit(point, case) == correct


def test_union_regions_and_polygons():
    case = dict(image_size=[100, 100], target_geometry=dict(type='bbox', boxes=[[0, 0, .1, .1], [.8, .8, 1., 1.]]))
    assert evaluation.point_hit([90, 90], case)
    case['target_geometry'] = dict(type='polygon', polygons=[[[.2, .2], [.8, .2], [.5, .8]]])
    assert evaluation.point_hit([50, 20], case) and not evaluation.point_hit([0, 0], case)


def test_paired_cluster_bootstrap_uses_same_draws_and_is_seeded():
    cohort = [dict(sample_id=str(i), dataset='one', pixel_sha256='same' if i < 2 else 'other') for i in range(4)]
    matrix = np.array([[1, 1], [0, 0], [0, 0], [1, 1]])
    first = evaluation.bootstrap_accuracies(matrix, cohort, 50, 123)
    assert np.array_equal(first, evaluation.bootstrap_accuracies(matrix, cohort, 50, 123))
    assert np.array_equal(first[:, 0], first[:, 1]) and np.all(first == .5)


@pytest.mark.parametrize('corruption', ['wrong_choice', 'missing_method', 'missing_case', 'missing_ui'])
def test_missing_or_inconsistent_predictions_cannot_be_silently_omitted(toy, corruption):
    root, cohort = toy; stem = evaluation.key(cohort[0]['sample_id'])
    path = root / 'predictions/test' / (stem + '.json'); value = evaluation.read(path)
    if corruption == 'wrong_choice': value['methods'][evaluation.PRIMARY]['point'] = [20, 20]
    elif corruption == 'missing_method': value['methods'].pop(evaluation.PRIMARY)
    elif corruption == 'missing_case': path.unlink()
    else: (root / 'ui_tars' / (stem + '.json')).unlink()
    if corruption in ('wrong_choice', 'missing_method'): dump(path, value)
    with pytest.raises((AssertionError, FileNotFoundError)): evaluation.evaluate_split('test')


def test_test_cache_cannot_contain_labels_even_with_updated_sha(toy):
    root, cohort = toy; index_path = root / 'cache/test_index.json'; index = evaluation.read(index_path)
    row = index['cases'][0]; path = Path(row['path'])
    with np.load(path, allow_pickle=False) as z: arrays = {k: z[k].copy() for k in z.files}
    arrays['labels'] = np.zeros(2); save(path, arrays); row['sha256'] = evaluation.sha256(path); dump(index_path, index)
    with pytest.raises(AssertionError): audit.audit_split_inputs('test', cohort, {'frozen_at': 0.})
