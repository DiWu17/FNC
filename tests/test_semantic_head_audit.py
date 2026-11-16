"""CPU fixtures for the final auditor; no real test feature/model is read."""
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import audit_semantic_head_experiment as audit


def _fixture(tmp_path, monkeypatch):
    out = tmp_path/'experiment'; base = tmp_path/'old'
    out.mkdir(); (out/'frozen_readouts.json').write_text('{}', encoding='utf-8')
    monkeypatch.setattr(audit, 'BASE', base)
    primary = 'semantic_hard_s1__quality'
    frozen = dict(models={primary: dict(selector='quality')})
    protocol = dict(primary=primary, manifest_sha256='fixture-manifest')
    virtual = {}; records = {}; all_rows = {}
    plain = ['v6_s1_ownfine', 'v6_s2_ownfine', 'phrase_local', 'qwen_native_completed_lax', 'ui_tars']
    for split in ('diagnostic', 'test'):
        group = []; rows = []
        root = out if split == 'test' else base
        for i in range(192):
            sid = f'{split}:{i}'
            case = dict(sample_id=sid, dataset=['ScreenSpot-v2', 'ScreenSpot-Pro', 'VenusBench-GD'][i % 3],
                image_size=[100, 100], target_geometry=dict(type='bbox', boxes=[[.1, .1, .4, .4]]))
            available = i != 0
            record = dict(case=case, metadata=dict(target_available=available, grid_shape=[1, 2]),
                parent_indices=np.array([0, 1]), frozen_points=np.array([[20, 20], [75, 20]]),
                label_fixed_point_hit=np.array([1., 0.]))
            group.append(record)
            def row(method, point, **extra):
                return dict(sample_id=sid, dataset=case['dataset'], method=method, point=point,
                    hit=bool(audit.score_point(point, case)), **extra)
            for method in ('v5_fixed', 'v6_s1_fixed', 'v6_s2_fixed', primary+'__combined'):
                extras = dict(selected_candidate_index=0, selected_parent_index=0, selected_region_overlap=True,
                    candidate_region_oracle=True, frozen_fine_point_oracle=True, region_top3=True,
                    region_top5=True, failure_stage='hit')
                if method == primary+'__combined' and available:
                    extras['candidates'] = [dict(point=[20, 20], own_fine_point=[22, 22],
                        parent_index=0, candidate_index=0), dict(point=[75, 20], own_fine_point=[76, 22],
                        parent_index=1, candidate_index=1)]
                rows.append(row(method, [20, 20] if available else None,
                    **(extras if available else dict(failure_stage='no_name'))))
            rows.append(row(primary+'__combined_ownfine', [22, 22] if available else None))
            rows.append(row(primary+'__combined_center', [25, 50] if available else None))
            for method in plain:
                point = [75, 20] if method == 'ui_tars' else [20, 20]
                if not available and method not in ('qwen_native_completed_lax', 'ui_tars'):
                    point = None
                rows.append(row(method, point))
            virtual[audit.path_key(root/'native'/f'{audit.key(sid)}.json')] = dict(completed_lax_point=[20, 20])
            virtual[audit.path_key(root/'ui_tars'/f'{audit.key(sid)}.json')] = dict(point=[75, 20])
        records[split] = group; all_rows[split] = rows
        folder = out/'evaluation'/split; folder.mkdir(parents=True)
        prediction_path = folder/'predictions.jsonl'
        prediction_path.write_text(''.join(json.dumps(row)+'\n' for row in rows), encoding='utf-8')
        methods = {}
        for method in sorted({row['method'] for row in rows}):
            selected = [row for row in rows if row['method'] == method]
            hit_count = sum(row['hit'] for row in selected)
            metric = dict(n=192, hits=hit_count, points=sum(row['point'] is not None for row in selected), accuracy=hit_count/192,
                datasets={dataset: dict(n=64, hits=sum(row['hit'] for row in selected if row['dataset'] == dataset))
                    for dataset in ('ScreenSpot-v2', 'ScreenSpot-Pro', 'VenusBench-GD')})
            if any('selected_region_overlap' in row for row in selected):
                metric.update(region_top1=191, region_top3=191, region_top5=191,
                    candidate_region_coverage=191, frozen_fine_point_oracle=191,
                    failure_stages=dict(Counter(row['failure_stage'] for row in selected)))
            methods[method] = metric
        summary = dict(primary=primary+'__combined', n=192, split=split, test_is_fresh=split == 'test',
            methods=methods, manifest_sha256=protocol['manifest_sha256'],
            frozen_sha256=audit.sha256(out/'frozen_readouts.json'), all_missing_outputs_retained=True,
            predictions_sha256=audit.sha256(prediction_path), prediction_count=len(rows))
        virtual[audit.path_key(folder/'summary.json')] = summary
    real_read = audit.read
    monkeypatch.setattr(audit, 'read', lambda filename: virtual[audit.path_key(filename)]
        if audit.path_key(filename) in virtual else real_read(filename))
    return out, protocol, frozen, records, virtual, all_rows


def test_auditor_replays_every_fixture_method_and_all_auxiliary_fields(tmp_path, monkeypatch):
    out, protocol, frozen, records, _, _ = _fixture(tmp_path, monkeypatch)
    result, lookup = audit.audit_predictions(out, protocol, frozen, records)
    for split in ('diagnostic', 'test'):
        assert result[split]['methods'] == 11
        assert result[split]['scored_points'] == len(lookup[split]) == 11*192


def test_auditor_rejects_corrupted_aggregate_and_mismatched_frozen_point(tmp_path, monkeypatch):
    out, protocol, frozen, records, virtual, rows = _fixture(tmp_path, monkeypatch)
    summary = virtual[audit.path_key(out/'evaluation/test/summary.json')]
    summary['methods']['v5_fixed']['hits'] -= 1
    with pytest.raises(AssertionError):
        audit.audit_predictions(out, protocol, frozen, records)
    summary['methods']['v5_fixed']['hits'] += 1
    # Keep point and hit internally consistent, but move off the sealed point.
    first = next(row for row in rows['test'] if row['method'] == protocol['primary']+'__combined' and row['point'] is not None)
    first['point'] = [21, 20]
    prediction_path = out/'evaluation/test/predictions.jsonl'
    prediction_path.write_text(''.join(json.dumps(row)+'\n' for row in rows['test']), encoding='utf-8')
    summary['predictions_sha256'] = audit.sha256(prediction_path)
    with pytest.raises(AssertionError):
        audit.audit_predictions(out, protocol, frozen, records)


def test_region_overlap_requires_positive_area_and_accepts_multiple_valid_regions():
    case = dict(target_geometry=dict(type='bbox', boxes=[[0., 0., .5, 1.]]))
    np.testing.assert_array_equal(audit.region_overlap(case, [1, 2], [0, 1]), [True, False])
    case = dict(target_geometry=dict(type='bbox', boxes=[[.1, .1, .2, .2], [.8, .8, .9, .9]]))
    np.testing.assert_array_equal(audit.region_overlap(case, [1, 2], [0, 1]), [True, True])
    case = dict(target_geometry=dict(type='polygon', polygons=[[[0., 0.], [.5, 0.], [0., 1.]]]))
    np.testing.assert_array_equal(audit.region_overlap(case, [1, 2], [0, 1]), [True, False])


def _numeric_fixture():
    case = dict(sample_id='synthetic-rounding', image_size=[100, 100],
        target_geometry=dict(type='bbox', boxes=[[.1, .755, .3, .765]]))
    record = dict(case=case, metadata=dict(grid_shape=[1, 1]), parent_indices=np.array([0]))
    cpu_offsets = torch.full((1, 4, 2), .5)
    cpu_offsets[0, 2] = torch.tensor([.4, .51])
    gpu_offsets = cpu_offsets.clone()
    gpu_offsets[0, 2, 1] = torch.nextafter(cpu_offsets[0, 2, 1], torch.tensor(1.))
    cpu = dict(child_logits=torch.tensor([[-10., -10., 10., -10.]]),
        offsets=cpu_offsets, candidate_lengths=[1])
    gpu = dict(cpu, offsets=gpu_offsets)
    canonical = audit.frozen_point_arrays(gpu, record, 0)
    record['frozen_points'] = canonical['points'].copy()
    record['frozen_point_features'] = canonical['features_float16'].copy()
    return record, cpu, gpu


def test_canonical_numeric_exception_reports_half_pixel_and_hit_oracle_effect():
    record, cpu, gpu = _numeric_fixture()
    events, batch = audit.verify_canonical_gpu_batch([record], cpu, gpu, 'test', 0)
    assert len(events) == 1 and batch['all_gpu_points_exact']
    event = events[0]
    assert event['cpu_point'] == [20, 75] and event['canonical_gpu_point'] == [20, 76]
    assert event['classified_as_half_pixel_rounding'] and event['hit_changed'] and event['point_oracle_changed']
    assert event['half_pixel_boundaries'][0]['continuous_delta_px'] < .00001
    assert event['canonical_all_13_float16_features_exact']


def test_canonical_numeric_replay_rejects_bad_point_and_any_float16_feature():
    record, cpu, gpu = _numeric_fixture()
    wrong = dict(record, frozen_points=record['frozen_points']+np.array([[0, 1]]))
    with pytest.raises(AssertionError, match='Canonical GPU point mismatch'):
        audit.verify_canonical_gpu_batch([wrong], cpu, gpu, 'test', 0)
    changed = record['frozen_point_features'].copy()
    changed[0, 7] = np.nextafter(changed[0, 7], np.float16(1.))
    wrong = dict(record, frozen_point_features=changed)
    with pytest.raises(AssertionError, match='Canonical GPU feature mismatch'):
        audit.verify_canonical_gpu_batch([wrong], cpu, gpu, 'test', 0)


def _repeat_fixture(monkeypatch):
    record, cpu, gpu = _numeric_fixture()
    real_arrays = audit.frozen_point_arrays
    midpoint = np.float32((float(np.float16(.304443359375))+float(np.float16(.3046875)))/2)
    lower = np.nextafter(midpoint, np.float32(-np.inf))
    def arrays(output, item, cursor):
        result = real_arrays(output, item, cursor)
        if 'synthetic_probability' in output:
            result['features'][0, 0] = np.float32(output['synthetic_probability'])
        if output.get('bad_coordinate_feature'):
            result['features'][0, 7] += .01
        result['features_float16'] = result['features'].astype(np.float16)
        return result
    monkeypatch.setattr(audit, 'frozen_point_arrays', arrays)
    cpu = dict(cpu, synthetic_probability=midpoint)
    gpu = dict(gpu, synthetic_probability=midpoint)
    record['frozen_point_features'] = arrays(gpu, record, 0)['features_float16']
    runs = [dict(gpu, synthetic_probability=value) for value in (lower, midpoint, lower, midpoint, lower)]
    return record, cpu, runs


def test_fixed_five_replays_require_exact_witness_and_explain_measured_float32_boundary(monkeypatch):
    record, cpu, runs = _repeat_fixture(monkeypatch)
    events, batch = audit.verify_canonical_gpu_repeats([record], cpu, runs, 'test', 0)
    assert len(events) == 1 and batch['fixed_gpu_replay_count'] == 5
    assert batch['whole_batch_13_column_exact_witness_runs'] == [1, 3]
    assert batch['interrun_not_bitstable'] and batch['all_integer_points_exact_every_run']
    assert [r['float16_feature_mismatch_elements'] for r in batch['runs']] == [1, 0, 1, 0, 1]
    boundary = batch['explained_float16_boundaries'][0]
    assert boundary['real_range_crosses_or_touches_midpoint'] and boundary['cached_pattern_observed']
    assert len(boundary['source_float32_runs']) == 5
    assert boundary['source_float32_max'] == boundary['float16_midpoint']


def test_fixed_five_replays_reject_absent_witness_and_unexplained_descriptor(monkeypatch):
    record, cpu, runs = _repeat_fixture(monkeypatch)
    with pytest.raises(AssertionError, match='exactly five'):
        audit.verify_canonical_gpu_repeats([record], cpu, runs[:4], 'test', 0)
    with pytest.raises(AssertionError, match='never reproduced'):
        audit.verify_canonical_gpu_repeats([record], cpu, [runs[0]]*5, 'test', 0)
    bad = list(runs); bad[0] = dict(runs[0], synthetic_probability=.1)
    with pytest.raises(AssertionError, match='exactly two|adjacent'):
        audit.verify_canonical_gpu_repeats([record], cpu, bad, 'test', 0)


def test_fixed_five_replays_reject_one_wrong_integer_or_coordinate_feature(monkeypatch):
    record, cpu, runs = _repeat_fixture(monkeypatch)
    bad = list(runs); offsets = runs[0]['offsets'].clone(); offsets[0, 2, 1] = .6
    bad[0] = dict(runs[0], offsets=offsets)
    with pytest.raises(AssertionError, match='point mismatch'):
        audit.verify_canonical_gpu_repeats([record], cpu, bad, 'test', 0)
    bad = list(runs); bad[0] = dict(runs[0], bad_coordinate_feature=True)
    with pytest.raises(AssertionError, match='coordinate-feature mismatch'):
        audit.verify_canonical_gpu_repeats([record], cpu, bad, 'test', 0)


def _componentwise_fixture(monkeypatch):
    record, cpu, runs = _repeat_fixture(monkeypatch)
    real_arrays = audit.frozen_point_arrays
    midpoint = np.float32(.3045654296875)
    lower = np.nextafter(midpoint, np.float32(-np.inf))
    def arrays(output, item, cursor):
        result = real_arrays(output, item, cursor)
        result['features'][0, 1] = np.float32(output.get('second_probability', midpoint))
        result['features_float16'] = result['features'].astype(np.float16)
        return result
    monkeypatch.setattr(audit, 'frozen_point_arrays', arrays)
    record['frozen_point_features'][0, 1] = np.float16(midpoint)
    runs = [dict(runs[i], synthetic_probability=lower if i in (0, 2) else midpoint,
        second_probability=midpoint if i in (0, 2) else lower) for i in range(5)]
    return record, cpu, runs


def test_componentwise_current_five_can_pass_without_a_joint_exact_run(monkeypatch):
    record, cpu, runs = _componentwise_fixture(monkeypatch)
    events, batch = audit.verify_canonical_gpu_repeats([record], cpu, runs, 'test', 0)
    assert batch['whole_batch_13_column_exact_witness_runs'] == [] and not batch['witness_exact']
    assert batch['all_13_columns_elementwise_reproduced']
    assert not events[0]['canonical_all_13_float16_features_exact']
    assert events[0]['canonical_all_13_float16_elements_reproduced']


def _retained_log_fixture(tmp_path, monkeypatch):
    record, cpu, runs = _componentwise_fixture(monkeypatch)
    _, measured = audit.verify_canonical_gpu_repeats([record], cpu, runs, 'test', 0)
    folder = tmp_path/'numeric_evidence'; folder.mkdir()
    log = folder/'failed_formal_fixed_five.log'
    log.write_text(''.join('CANONICAL GPU FIXED REPEAT '+json.dumps(dict(split='test', batch_start=0, **r))+'\n'
        for r in measured['runs']), encoding='utf-8')
    joint = dict(runs[0], synthetic_probability=np.float32(.3045654296875), second_probability=np.float32(.3045654296875))
    events, _ = audit.verify_canonical_gpu_batch([record], cpu, joint, 'test', 0)
    prior = folder/'prior_single_exact_witness.json'
    prior.write_text(json.dumps(dict(checkpoint_sha256='fine-fixture', cpu_numeric_exceptions=events)), encoding='utf-8')
    (tmp_path/'protocol.json').write_text(json.dumps(dict(fixed_fine_sha256='fine-fixture')), encoding='utf-8')
    point_file = folder/'bound_point_cache.bin'; point_file.write_bytes(b'unchanged fixed-point cache fixture')
    evidence = dict(evidence_kind='authenticated_preserved_fixed_five_log', split='test', batch_start=0,
        samples=[record['case']['sample_id']], n_candidates=1, provenance='Synthetic unit test only',
        files={str(log):audit.sha256(log), str(prior):audit.sha256(prior)},
        context_sha256_asserted_at_preservation={str(tmp_path/'protocol.json'):audit.sha256(tmp_path/'protocol.json')},
        point_cache_sha256_asserted_at_preservation={str(point_file):audit.sha256(point_file)})
    manifest = folder/'evidence_manifest.json'; manifest.write_text(json.dumps(evidence), encoding='utf-8')
    monkeypatch.setattr(audit, 'PRESERVED_NUMERIC_EVIDENCE_SHA256', audit.sha256(manifest))
    return record, cpu, point_file


def test_preserved_current_five_log_reports_inferred_bounds_without_fake_joint_witness(tmp_path, monkeypatch):
    record, cpu, _ = _retained_log_fixture(tmp_path, monkeypatch)
    events, batch = audit.verify_preserved_numeric_evidence(tmp_path, [record], cpu, 'test', 0)
    assert batch['evidence_kind'] == 'authenticated_preserved_fixed_five_log'
    assert batch['all_13_columns_elementwise_reproduced'] and not batch['whole_batch_13_exact']
    assert batch['fp32_matching_run_bounds_inferred'] and batch['no_new_gpu_draws']
    assert batch['whole_batch_13_column_exact_witness_runs'] == []
    assert not events[0]['canonical_all_13_float16_features_exact']
    for boundary in batch['explained_float16_boundaries']:
        assert boundary['source_float32_max'] is None
        assert any(v is None for v in boundary['source_float32_runs'])
        assert boundary['matching_run_fp32_values_inferred']


def test_preserved_current_five_log_rejects_changed_bound_cache(tmp_path, monkeypatch):
    record, cpu, point_file = _retained_log_fixture(tmp_path, monkeypatch)
    point_file.write_bytes(b'changed after evidence preservation')
    with pytest.raises(AssertionError, match='Artifact changed'):
        audit.verify_preserved_numeric_evidence(tmp_path, [record], cpu, 'test', 0)
