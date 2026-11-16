"""CPU-only final V11 audit: identities, all choices/metrics and fixed replays."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

from clickagent.action_benchmark import sha256, write_json
from clickagent.candidate_budget import BUDGETS, build_pools
from clickagent.public_evaluation import point_in_geometry
from candidate_scorer_protocol import OUT, SOURCES, SEEDS, VARIANTS, _object_sha, validate_model
import evaluate_candidate_scorer as evaluation
from point_aligned_data import read
from run_target_token_experiment import key, read_jsonl

SCORE_ATOL = 1e-6
SCORE_RTOL = 1e-5


def independent_hit(point, case):
    """Independently recompute final point hits, including bounds and edge union."""
    if point is None: return False
    if not isinstance(point, (list, tuple)) or len(point) != 2: return False
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v != round(v) for v in point): return False
    width, height = case['image_size']
    if not 0 <= point[0] < width or not 0 <= point[1] < height: return False
    x, y = point[0] / width, point[1] / height
    geometry = case['target_geometry']
    if geometry['type'] == 'bbox':
        return any(a - 1e-12 <= x <= c + 1e-12 and b - 1e-12 <= y <= d + 1e-12 for a, b, c, d in geometry['boxes'])
    return bool(point_in_geometry([x, y], geometry))


def _canonical(value):
    # JSON object keys are strings (e.g. candidate_size_counts), unlike Python Counters.
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


def verify_seal():
    frozen = read(OUT / 'frozen.json')
    assert frozen['complete'] and frozen['fresh_test_features_at_freeze'] == 0
    assert frozen['primary'] == 'internal_full_s1' and frozen['primary_budget'] == 16
    categories = {'frozen': frozen['files'], 'training': frozen['training_input_evidence']['files'],
                  'external': frozen['external_input_evidence']['files']}
    for name in ('training_input_evidence', 'external_input_evidence'):
        evidence = frozen[name]
        assert evidence['complete'] and evidence['checked_once_at_freeze']
        assert evidence['inventory_sha256'] == _object_sha(evidence['files'])
    all_files = {}
    for inventory in categories.values():
        for path, digest in inventory.items():
            normalized = os.path.normcase(os.path.abspath(path))
            assert all_files.setdefault(normalized, digest) == digest
    with ThreadPoolExecutor(max_workers=4) as pool:
        actual = list(pool.map(sha256, map(Path, all_files)))
    for (path, expected), digest in zip(all_files.items(), actual):
        assert digest == expected, ('Frozen input changed', path)
    sources = {p: sha256(Path(p)) for p in SOURCES}
    spec = read(OUT / 'protocol.json'); training = frozen['training_input_evidence']
    expected_models = {v + '_' + s for v in VARIANTS for s in ('s1', 's2')}
    assert set(frozen['models']) == expected_models
    for variant in VARIANTS:
        for suffix, seed in zip(('s1', 's2'), SEEDS):
            entry = frozen['models'][variant + '_' + suffix]
            identity = training['identities']['visual' if variant == 'visual' else 'internal']
            meta = validate_model(Path(entry['file']), variant, seed, spec, sources, identity, training['counts'])
            assert entry['selected'] == meta['selected'] and entry['sha256'] == meta['checkpoint_sha256']
    return frozen, {name + '_files_checked': len(values) for name, values in categories.items()}


def _scalar_json(archive):
    value = archive['metadata']
    assert value.ndim == 0 and value.dtype.kind in 'US'
    return json.loads(str(value))


def audit_split_inputs(split, cohort, frozen):
    index_path = OUT / 'cache' / (split + '_index.json'); index = read(index_path)
    assert index['complete'] and index['visual_complete'] and index['has_labels'] is False
    assert index['split'] == split and index['real_candidate_pools'] and index['annotation_derived_coordinates'] is False
    assert [r['sample_id'] for r in index['cases']] == [c['sample_id'] for c in cohort]
    assert len({r['sample_id'] for r in index['cases']}) == len(cohort)
    counts = dict(label_free_caches=0, raw_features_checked=0, pools_checked=0)
    raw_seen = set()
    for row, case in zip(index['cases'], cohort):
        assert sha256(Path(case['image'])) == case['image_sha256']
        pipeline_path = OUT / split / 'pipeline' / (key(case['sample_id']) + '.json')
        assert str(pipeline_path) == row['pipeline_path'] and sha256(pipeline_path) == row['pipeline_sha256']
        pipeline = read(pipeline_path)
        assert pipeline['sample_id'] == case['sample_id'] and pipeline['ground_truth_in_model_inputs'] is False
        assert pipeline['pools'] == build_pools(pipeline)
        counts['pools_checked'] += len(BUDGETS)
        raw_files = {pipeline['global_feature_path']: pipeline['global_feature_sha256']}
        if pipeline['local_feature_path']: raw_files[pipeline['local_feature_path']] = pipeline['local_feature_sha256']
        assert row['raw_files'] == raw_files
        for path, digest in raw_files.items():
            if path not in raw_seen:
                assert sha256(Path(path)) == digest
                with np.load(path, allow_pickle=False) as archive: meta = _scalar_json(archive)
                assert meta['sample_id'] == case['sample_id'] and meta['image_sha256'] == case['image_sha256']
                assert meta['own_token_forward'] and meta['all_backbone_parameters_frozen'] and meta['ground_truth_in_model_input'] is False
                assert meta['instruction_sha256'] == hashlib.sha256(case['instruction'].encode()).hexdigest()
                if path == pipeline['global_feature_path']: assert meta['image_size'] == case['image_size']
                else: assert meta['transform'] == pipeline['old_crop_transform']
                if split == 'test': assert Path(path).stat().st_mtime >= frozen['frozen_at']
                raw_seen.add(path); counts['raw_features_checked'] += 1
        common = None
        for visual in (False, True):
            path = Path(row['visual_path'] if visual else row['path'])
            assert sha256(path) == row['visual_sha256' if visual else 'sha256']
            with np.load(path, allow_pickle=False) as archive:
                assert 'labels' not in archive.files and not any('label' in k or 'target_geometry' in k for k in archive.files)
                assert str(archive['sample_id']) == case['sample_id'] and bool(archive['available']) == row['available']
                assert not bool(archive['annotation_derived_coordinates'])
                assert str(archive['pipeline_sha256']) == row['pipeline_sha256']
                points = archive['points']; assert points.shape == (row['n_points'], 2)
                assert np.issubdtype(points.dtype, np.integer) and len(points) == len(pipeline['pools']['32'])
                assert points.tolist() == [v['point'] for v in pipeline['pools']['32']]
                assert row['available'] == bool(len(points))
                for budget in BUDGETS:
                    indices = archive['pool_indices_' + str(budget)]
                    assert np.issubdtype(indices.dtype, np.integer) and indices.ndim == 1
                    assert ((indices >= 0) & (indices < len(points))).all() and len(np.unique(indices)) == len(indices)
                    assert points[indices].tolist() == [v['point'] for v in pipeline['pools'][str(budget)]]
                if common is None: common = points.copy()
                else: assert np.array_equal(common, points)
                if visual and row['available']:
                    encoding = OUT / 'vision_encodings' / (case['image_sha256'] + '.npz')
                    assert sha256(encoding) == str(archive['visual_encoding_sha256'])
                    with np.load(encoding, allow_pickle=False) as value: metadata = _scalar_json(value)
                    assert metadata['image_sha256'] == case['image_sha256'] and metadata['image_size'] == case['image_size']
                    assert metadata['all_visual_parameters_frozen'] and not metadata['gui_actor_used'] and not metadata['language_model_loaded']
                    if split == 'test': assert encoding.stat().st_mtime >= frozen['frozen_at']
            if split == 'test': assert path.stat().st_mtime >= frozen['frozen_at']
            counts['label_free_caches'] += 1
    completion = read(OUT / 'predictions' / split / 'completion.json')
    assert completion['complete'] and completion['n'] == len(cohort) and completion['frozen_sha256'] == sha256(OUT / 'frozen.json')
    return index, counts


def audit_native(point, result, case):
    from clickagent.public_diagnostics import parse_offline_lax_click
    complete = result['terminated_by_eos'] or result['generated_text'].endswith(('\n', '\r'))
    parsed = parse_offline_lax_click(result['generated_text']) if complete else None
    expected = [round(v * (size - 1) / 1000) for v, size in zip(parsed, case['image_size'])] if parsed else None
    assert point == expected == result['completed_lax_point']


def replay(split, cohort, index, count=2):
    """Deterministic label-blind sample; CPU only, integer choices remain exact."""
    import torch
    from clickagent.internal_candidate_scorer import load_internal_scorer
    from clickagent.point_critic import FrozenPointEncoder, load_point_critic
    from clickagent.visual_candidate_verifier import load_visual_verifier
    from cache_candidate_scorer import raw_arrays
    from point_aligned_data import archive_record
    from point_aligned_predict import load_predictors, predict_record
    from train_candidate_scorer import load_cache, subset
    torch.set_num_threads(4)
    by_id = {r['sample_id']: r for r in index['cases']}
    available = [c for c in cohort if by_id[c['sample_id']]['available']]
    selected = sorted(available, key=lambda c: hashlib.sha256(('v11_cpu_audit:' + c['sample_id']).encode()).hexdigest())[:count]
    assert len(selected) == min(count, len(available))
    heads = {'internal_full_s1': (load_internal_scorer(OUT / 'models/internal_full_s1.pt', 'cpu')[0].eval(), False),
             'visual_s1': (load_visual_verifier(OUT / 'models/visual_s1.pt', 'cpu')[0].eval(), True)}
    decoder = load_predictors(primary_only=True)
    old_encoder = FrozenPointEncoder(decoder['joint_connected_s1'][0])
    old_critic = load_point_critic(evaluation.OLD / 'models/point_critic_s1.pt')[0].eval()
    max_score_difference = 0.; raw_replays = 0; head_replays = 0; old_critic_replays = 0
    with torch.inference_mode():
        for case in selected:
            sid = case['sample_id']; row = by_id[sid]
            stored = read(OUT / 'predictions' / split / (key(sid) + '.json'))['methods']
            pipeline = read(OUT / split / 'pipeline' / (key(sid) + '.json'))
            for name, (model, visual) in heads.items():
                record = load_cache(Path(row['visual_path'] if visual else row['path']), visual)
                assert record['labels'] is None
                for budget in (4, 16, 32):
                    indices = record['pool_indices'][str(budget)]
                    output = model(**subset(record, indices, visual, 'cpu'))
                    scores = output['scores'].float().cpu().numpy()
                    expected = stored[name + '_K' + str(budget)]
                    assert np.allclose(scores, expected['scores'], atol=SCORE_ATOL, rtol=SCORE_RTOL)
                    null = float(output['null_scores'].reshape(-1)[0])
                    assert np.isclose(null, expected['null_score'], atol=SCORE_ATOL, rtol=SCORE_RTOL)
                    assert bool(null > scores.max()) == expected['null_would_reject']
                    selected_point = pipeline['pools'][str(budget)][int(scores.argmax())]['point']
                    assert selected_point == expected['point']
                    max_score_difference = max(max_score_difference, float(np.max(np.abs(scores - np.asarray(expected['scores'])))))
                    head_replays += 1
            arrays, metadata = raw_arrays(pipeline['global_feature_path'])
            state = old_encoder.prepare(arrays, metadata['grid_shape'], case['image_size'])
            points = [v['point'] for v in pipeline['pools']['16']]
            features = old_encoder.features(state, points).astype(np.float16).astype(np.float32)
            scores = old_critic(torch.from_numpy(features)).cpu().numpy()
            assert np.allclose(scores, stored['old_critic_K16']['scores'], atol=SCORE_ATOL, rtol=SCORE_RTOL)
            assert points[int(scores.argmax())] == stored['old_critic_K16']['point']
            old_critic_replays += 1
            del arrays, state, features
            replayed_pipeline = dict(pipeline)
            for raw_name, result_name in (('global_feature_path', 'global_result'), ('local_feature_path', 'old_crop_result')):
                if pipeline[raw_name] is None: continue
                with np.load(pipeline[raw_name], allow_pickle=False) as archive: metadata = _scalar_json(archive)
                public = dict(sample_id=sid, image_size=metadata['image_size'], image_sha256=case['image_sha256'])
                record = archive_record(pipeline[raw_name], public, labels=False)
                result = predict_record(record, decoder, only=['joint_connected_s1'], top_k=64)['joint_connected_s1']
                expected = pipeline[result_name]
                assert result['point'] == expected['point']
                assert result.get('all_points', []) == expected.get('all_points', [])
                assert result.get('parent_indices', []) == expected.get('parent_indices', [])
                assert np.allclose(result.get('all_scores', []), expected.get('all_scores', []), atol=SCORE_ATOL, rtol=SCORE_RTOL)
                replayed_pipeline[result_name] = result
                raw_replays += 1
            replayed_pools = build_pools(replayed_pipeline)
            for budget in BUDGETS:
                assert [v['point'] for v in replayed_pools[str(budget)]] == [v['point'] for v in pipeline['pools'][str(budget)]]
    return dict(complete=True, selection='first two nonempty sample IDs by sha256(v11_cpu_audit:sample_id), no annotations',
                case_ids=[c['sample_id'] for c in selected], heads=list(heads), budgets=[4, 16, 32],
                head_forwards=head_replays, raw_decoder_forwards=raw_replays, old_critic_forwards=old_critic_replays, integer_points_exact=True,
                score_atol=SCORE_ATOL, score_rtol=SCORE_RTOL, max_abs_head_score_difference=max_score_difference)


def audit_split(split, frozen):
    cohort = evaluation.cases(split)
    index, checks = audit_split_inputs(split, cohort, frozen)
    expected_rows, expected_details, expected_summary = evaluation.evaluate_split(split, write=False)
    dest = OUT / 'evaluation' / split
    saved_rows = read_jsonl(dest / 'predictions.jsonl'); saved_details = read(dest / 'details.json'); saved_summary = read(dest / 'summary.json')
    assert saved_rows == _canonical(expected_rows) and saved_details == _canonical(expected_details)
    assert saved_summary == _canonical(expected_summary), ('Evaluation summary mismatch', split)
    case_map = {c['sample_id']: c for c in cohort}; seen = set()
    for row in saved_rows:
        identity = (row['sample_id'], row['method']); assert identity not in seen; seen.add(identity)
        case = case_map[row['sample_id']]
        assert row['hit'] == independent_hit(row['point'], case)
        if 'optional_null_hit' in row: assert row['optional_null_hit'] == independent_hit(row['optional_null_point'], case)
        if row['method'] == 'qwen_native_completed_lax':
            root = evaluation.OLD if split == 'diagnostic' else OUT / split
            audit_native(row['point'], read(root / 'native' / (key(case['sample_id']) + '.json')), case)
    methods = len(saved_summary['methods'])
    assert len(saved_rows) == len(cohort) * methods == len(seen)
    assert methods == len(evaluation.expected_head_methods()) + 6
    for item in saved_details:
        case = case_map[item['sample_id']]
        for pool in item['pools'].values():
            assert pool['point_hits'] == [independent_hit(p, case) for p in pool['points']]
    checks.update(n=len(cohort), methods=methods, predictions_checked=len(saved_rows), replay=replay(split, cohort, index))
    return checks


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--split', choices=['test', 'diagnostic', 'both'], default='both')
    args = parser.parse_args()
    print('AUDIT V11 frozen inputs, sources and models', flush=True)
    frozen, counters = verify_seal(); splits = {}
    for split in ('test', 'diagnostic') if args.split == 'both' else (args.split,):
        print('AUDIT V11 ' + split + ' all choices, labels, metrics, CPU replay', flush=True)
        splits[split] = audit_split(split, frozen)
    evidence_files = [OUT / 'evaluation' / split / name for split in splits for name in ('summary.json', 'details.json', 'predictions.jsonl')]
    result = dict(complete=True, version=11, frozen_sha256=sha256(OUT / 'frozen.json'),
        protocol_sha256=sha256(OUT / 'protocol.json'), splits=splits, **counters,
        source_sha256={str(Path(__file__)): sha256(Path(__file__)),
                       str(Path(evaluation.__file__)): sha256(Path(evaluation.__file__))},
        evaluation_sha256={str(p): sha256(p) for p in evidence_files},
        warning='Small deterministic CPU replays supplement complete artifact/choice/metric auditing; they are not a replay of every model forward.')
    # A partial audit never masquerades as the two-cohort final audit.
    path = OUT / ('final_audit.json' if args.split == 'both' else 'audit_' + args.split + '.json')
    write_json(path, result)
    print('AUDIT V11 COMPLETE ' + str(path), flush=True)


if __name__ == '__main__': main()
