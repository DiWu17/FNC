"""Synthetic fixtures only; no project manifests, features or model inference."""
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import analyze_candidate_scorer_compression as compression
from clickagent.candidate_budget import build_pools, choose_point


def toy_case(index=0, empty=False):
    sid = f'synthetic:{index}'
    target = 55 if index == 0 else 15
    case = dict(sample_id=sid, dataset='synthetic', pixel_sha256=str(index), image_size=[200, 100],
                target_geometry=dict(type='bbox', boxes=[[(target - 1) / 200, .09, (target + 1) / 200, .11]]))
    def result(y):
        return dict(all_points=[[10 * i + 5, y] for i in range(8)], all_scores=list(range(8, 0, -1)), parent_indices=list(range(8)))
    pipeline = dict(sample_id=sid, global_result={} if empty else result(10), old_crop_result=None if empty else result(20),
         old_crop_transform=None if empty else dict(box=[0, 0, 200, 100], scale=1, original_size=[200, 100]), ground_truth_in_model_inputs=False)
    pipeline['pools'] = build_pools(pipeline)
    points = [v['point'] for v in pipeline['pools']['16']]
    head = np.zeros(len(points)); gui = np.zeros(len(points))
    if not empty:
        if index == 0:
            head[[0, 5, 6, 7]] = [10, 9, 8, 7]
            gui[[0, 5, 15]] = [3, 8, 12]
        else:
            head[[0, 4, 5, 6]] = [10, 9, 8, 7]
            gui[[1, 4, 12]] = [8, 9, 12]
    records = {','.join(map(str, p)): dict(point=p, log_odds=float(gui[i])) for i, p in enumerate(points)}
    reference = dict(sample_id=sid, records=records, choices={str(k): choose_point(pipeline['pools'][str(k)], records) for k in (4, 16)},
                     ground_truth_in_model_inputs=False)
    prediction = dict(sample_id=sid, methods={}, ground_truth_in_model_inputs=False)
    audited = dict(sample_id=sid, pools={}, methods={})
    for k in (4, 16):
        pts = [v['point'] for v in pipeline['pools'][str(k)]]
        audited['pools'][str(k)] = dict(points=pts, point_hits=[compression.hit(p, case) for p in pts])
        point = reference['choices'][str(k)]['point']
        audited['methods']['gui_actor_K' + str(k)] = dict(point=point, hit=compression.hit(point, case))
    for name in compression.HEADS:
        ix = int(head.argmax()) if len(points) else None
        point = points[ix] if ix is not None else None
        prediction['methods'][name + '_K16'] = dict(scores=head.tolist(), n_points=len(points), point=point, selected_index=ix)
        audited['methods'][name + '_K16'] = dict(point=point, hit=compression.hit(point, case), scores=head.tolist())
    return case, pipeline, prediction, reference, audited


def test_ties_use_original_pool_order_in_both_stages():
    kept, winner = compression.shortlist_choice([1] * 6, [1] * 6)
    assert kept == [0, 1, 2, 3] and winner == 0
    kept, winner = compression.shortlist_choice([5, 6, 9, 8, 7, 10], [0] * 6)
    assert kept == [5, 2, 3, 4] and winner == 2  # not first in head order


def test_small_empty_and_nonfinite_pools():
    assert compression.shortlist_choice([], []) == ([], None)
    assert compression.shortlist_choice([2, 1], [0, 1]) == ([0, 1], 1)
    for head, gui in [([float('nan')], [0]), ([0], [float('inf')]), ([0, 1], [1])]:
        with pytest.raises(AssertionError): compression.shortlist_choice(head, gui)


def test_synthetic_compression_can_rescue_and_harm_without_changing_points(monkeypatch):
    monkeypatch.setattr(compression, 'N', 3)
    monkeypatch.setattr(compression, 'BOOTSTRAP_REPS', 200)
    inputs = [toy_case(0), toy_case(1), toy_case(2, empty=True)]
    details = [compression.case_analysis(*values) for values in inputs]
    first = details[0]
    assert first['methods'][compression.PRIMARY]['hit']
    assert not first['methods']['gui_actor_K4']['hit'] and not first['methods']['gui_actor_K16']['hit']
    assert not first['methods']['internal_full_s1_K16']['hit']
    assert first['shortlists']['internal_full_s1']['point_oracle'] and not first['quota4_oracle']
    second = details[1]
    assert second['methods']['gui_actor_K4']['hit'] and not second['methods'][compression.PRIMARY]['hit']
    assert second['source_k16_oracle'] and not second['shortlists']['internal_full_s1']['point_oracle']
    assert details[2]['source_k16_points'] == []
    summary = compression.summarize('test', [x[0] for x in inputs], details)
    assert summary['methods'][compression.PRIMARY]['n'] == 3
    assert summary['methods'][compression.PRIMARY]['hits'] == 1
    assert summary['methods'][compression.PRIMARY]['verification_view_proxy']['total'] == 8
    assert summary['methods']['gui_actor_K16']['verification_view_proxy']['total'] == 32
    stage = summary['stages']['internal_full_s1']
    assert stage['source_empty'] == stage['compression_dropped_all_correct'] == stage['hits'] == 1
    assert stage['equal_actual_views_descriptive']['n'] == 3
    pair = next(p for p in summary['paired_comparisons'] if p['method'] == compression.PRIMARY and p['reference'] == 'gui_actor_K4')
    assert (pair['wins'], pair['losses'], pair['ties']) == (1, 1, 1)
    assert len(summary['methods']) == 10 and len(summary['paired_comparisons']) == 12 and len(summary['oracle_paired_comparisons']) == 8


@pytest.mark.parametrize('kind', ['score_nan', 'missing_gui', 'wrong_head_point', 'wrong_gui_choice', 'wrong_audited_scores'])
def test_inconsistent_existing_scores_or_choices_are_rejected(kind):
    values = list(toy_case()); case, pipeline, prediction, gui, audited = values
    if kind == 'score_nan': prediction['methods']['internal_full_s1_K16']['scores'][0] = float('nan')
    elif kind == 'missing_gui': gui['records'].pop('5,10')
    elif kind == 'wrong_head_point': prediction['methods']['internal_full_s1_K16']['point'] = [55, 10]
    elif kind == 'wrong_gui_choice': gui['choices']['16']['point'] = [55, 10]
    elif kind == 'wrong_audited_scores': audited['methods']['internal_full_s1_K16']['scores'][0] = -999
    with pytest.raises((AssertionError, KeyError)): compression.case_analysis(*values)


def test_selections_do_not_accept_ground_truth():
    scores = [9, 8, 7, 6, 5]; gui = [0, 4, 2, 1, 99]
    result = compression.shortlist_choice(scores, gui)
    assert result == ([0, 1, 2, 3], 1)
    # The pure selector has no case/label parameter; relabeling cannot change it.
    for _ in range(3): assert compression.shortlist_choice(scores, gui) == result


def test_registration_is_fixed_and_before_test_only(tmp_path, monkeypatch):
    monkeypatch.setattr(compression, 'OUT', tmp_path)
    registration = compression.preregister()
    assert registration['original_primary_protocol'] is False
    assert registration['training_validation_progress_previously_inspected']
    assert registration['constants']['source_budget'] == 16 and registration['constants']['retained_candidates'] == 4
    assert compression.preregister() == registration
    changed = copy.deepcopy(registration); changed['script_sha256'] = 'wrong'
    with pytest.raises(AssertionError): compression.validate_preregistration(changed)


def test_existing_test_artifact_blocks_new_registration(tmp_path, monkeypatch):
    monkeypatch.setattr(compression, 'OUT', tmp_path)
    path = tmp_path / 'test/features/one.npz'; path.parent.mkdir(parents=True); path.write_bytes(b'no data used')
    with pytest.raises(AssertionError): compression.preregister()
    assert not (tmp_path / 'compression_preregistration.json').exists()


@pytest.mark.parametrize('audit_state', ['missing', 'incomplete', 'identity_mismatch'])
def test_formal_gate_fails_before_any_manifest_or_label_read(tmp_path, monkeypatch, audit_state):
    monkeypatch.setattr(compression, 'OUT', tmp_path)
    compression.preregister()
    if audit_state == 'incomplete': compression.write(tmp_path / 'final_audit.json', dict(complete=False, splits={}))
    if audit_state == 'identity_mismatch':
        compression.write(tmp_path / 'frozen.json', dict(complete=True))
        compression.write(tmp_path / 'protocol.json', dict(complete=True))
        compression.write(tmp_path / 'final_audit.json', dict(complete=True, splits={'test': {}, 'diagnostic': {}}, frozen_sha256='wrong', protocol_sha256='wrong'))
    calls = []
    def forbid(path): calls.append(str(path)); raise AssertionError('Labels must not be read')
    monkeypatch.setattr(compression, 'read_rows', forbid)
    with pytest.raises((AssertionError, FileNotFoundError)): compression.analyze()
    assert calls == [] and not (tmp_path / 'evaluation/compression/analysis.json').exists()


def test_stratified_bootstrap_is_paired_and_deterministic(monkeypatch):
    monkeypatch.setattr(compression, 'BOOTSTRAP_REPS', 50)
    cohort = [dict(sample_id=str(i), dataset='one', pixel_sha256='same' if i < 2 else 'other') for i in range(4)]
    matrix = np.array([[1, 1], [0, 0], [0, 0], [1, 1]])
    samples = compression.bootstrap(matrix, cohort)
    assert np.array_equal(samples, compression.bootstrap(matrix, cohort)) and (samples == .5).all()


def synthetic_audited_bundle(tmp_path, monkeypatch):
    """Create an explicitly synthetic audit handoff; never load real artifacts."""
    monkeypatch.setattr(compression, 'OUT', tmp_path)
    monkeypatch.setattr(compression, 'N', 3)
    monkeypatch.setattr(compression, 'BOOTSTRAP_REPS', 200)
    compression.preregister()
    all_cohorts = {}; evaluation_hashes = {}; summaries = {}
    names = {f'{v}_{s}_K{k}' for v in ('internal_full', 'internal_pooled', 'internal_global', 'visual')
             for s in ('s1', 's2') for k in (4, 8, 16, 32)}
    names |= {'old_critic_K16', 'global_top1', 'qwen_native_completed_lax', 'gui_actor_K4', 'gui_actor_K16', 'ui_tars'}
    for split in ('test', 'diagnostic'):
        cohort = []; details = []; inputs = {}
        for number in range(3):
            values = list(toy_case(number, empty=number == 2))
            for value in values: value['sample_id'] = split + ':' + value['sample_id']
            case, pipeline, prediction, gui, audited = values
            case.update(experiment_split=split, diagnostic_reuse=split == 'diagnostic')
            cohort.append(case); details.append(audited)
            stem = compression.key(case['sample_id'])
            paths = [tmp_path / split / 'pipeline' / (stem + '.json'), tmp_path / 'predictions' / split / (stem + '.json'),
                     tmp_path / 'gui_reference' / split / (stem + '.json')]
            for path, value in zip(paths, (pipeline, prediction, gui)):
                compression.write(path, value); inputs[str(path)] = compression.sha(path)
        folder = tmp_path / 'evaluation' / split
        methods = {name: dict(hits=sum(d['methods'].get(name, {}).get('hit', False) for d in details)) for name in names}
        summary = dict(complete=True, split=split, n=3, primary='internal_full_s1_K16', selection_used_this_cohort=False,
                       exploratory_reused_images=split == 'diagnostic', methods=methods, input_sha256=inputs)
        summaries[split] = summary; all_cohorts[split] = cohort
        compression.write(folder / 'details.json', details)
        (folder / 'predictions.jsonl').write_text('', encoding='utf-8')
    (tmp_path / 'manifest.jsonl').write_text(''.join(json.dumps(c) + '\n' for c in all_cohorts['test']), encoding='utf-8')
    (tmp_path / 'diagnostic_manifest.jsonl').write_text(''.join(json.dumps(c) + '\n' for c in all_cohorts['diagnostic']), encoding='utf-8')
    compression.write(tmp_path / 'protocol.json', dict(complete=True))
    compression.write(tmp_path / 'data_selection.json', dict(complete=True))
    files = {str(tmp_path / name): compression.sha(tmp_path / name) for name in ('protocol.json', 'manifest.jsonl', 'data_selection.json', 'diagnostic_manifest.jsonl')}
    models = {}
    for head in compression.HEADS:
        path = tmp_path / 'models' / (head + '.pt'); path.parent.mkdir(exist_ok=True); path.write_bytes(b'synthetic, never loaded')
        models[head] = dict(file=str(path), sha256=compression.sha(path)); files[str(path)] = compression.sha(path)
    compression.write(tmp_path / 'frozen.json', dict(complete=True, primary='internal_full_s1', primary_budget=16,
                      fresh_test_features_at_freeze=0, files=files, models=models))
    for split, summary in summaries.items():
        summary.update(frozen_sha256=compression.sha(tmp_path / 'frozen.json'), protocol_sha256=compression.sha(tmp_path / 'protocol.json'))
        folder = tmp_path / 'evaluation' / split; compression.write(folder / 'summary.json', summary)
        evaluation_hashes.update({str(folder / name): compression.sha(folder / name) for name in ('summary.json', 'details.json', 'predictions.jsonl')})
    compression.write(tmp_path / 'final_audit.json', dict(complete=True,
        frozen_sha256=compression.sha(tmp_path / 'frozen.json'), protocol_sha256=compression.sha(tmp_path / 'protocol.json'),
        source_sha256={str(compression.ROOT / 'scripts' / name): compression.sha(compression.ROOT / 'scripts' / name)
                       for name in ('evaluate_candidate_scorer.py', 'audit_candidate_scorer.py')},
        evaluation_sha256=evaluation_hashes,
        splits={s: dict(n=3, methods=38, predictions_checked=114, replay=dict(complete=True, integer_points_exact=True)) for s in summaries}))


def test_complete_synthetic_handoff_writes_separate_report_without_model_loading(tmp_path, monkeypatch):
    synthetic_audited_bundle(tmp_path, monkeypatch)
    compression.analyze()
    result = compression.read(tmp_path / 'evaluation/compression/analysis.json')
    assert result['complete'] and set(result['cohorts']) == {'test', 'diagnostic'}
    assert result['cohorts']['test']['methods'][compression.PRIMARY]['hits'] == 1
    assert result['cohorts']['diagnostic']['exploratory_reused_images']
    assert (tmp_path / 'evaluation/compression/report.html').exists()


def test_post_audit_score_file_change_is_rejected_before_labels(tmp_path, monkeypatch):
    synthetic_audited_bundle(tmp_path, monkeypatch)
    path = next((tmp_path / 'predictions/test').glob('*.json'))
    with path.open('a', encoding='utf-8') as stream: stream.write(' ')
    calls = []
    monkeypatch.setattr(compression, 'read_rows', lambda p: calls.append(p))
    with pytest.raises(AssertionError, match='Audited prediction/pool changed'): compression.analyze()
    assert not calls and not (tmp_path / 'evaluation/compression/analysis.json').exists()
