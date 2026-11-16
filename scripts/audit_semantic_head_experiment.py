"""Fail-closed CPU audit of the sealed semantic-head experiment.

This does not fit a readout, choose a checkpoint or change any predictions.
The independent fresh CPU replay uses only the two prespecified quality
checkpoints. Fixed fine points/features and hit labels are verified separately.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import re

import numpy as np
from shapely.geometry import box
from shapely.ops import unary_union
import torch

from clickagent.action_benchmark import sha256, write_json
from clickagent.attention_field_ranker import build_attention_field
from clickagent.candidate_ranker import build_candidate_inputs, decode_ranker, load_ranker, make_ranker_batch
from clickagent.model_baselines import parse_ui_tars_point
from clickagent.public_diagnostics import parse_offline_lax_click
from clickagent.semantic_head_experiment import load_readout, score_output
from clickagent.spatial_geometry import shape_from_geometry
from evaluate_conditioned_readouts import score_point
from index_gui_images import fingerprint
from run_target_token_experiment import key, read_jsonl
from semantic_head_data import load_input_record, load_records, make_batch
from train_semantic_head_experiment import selectors


BASE = Path('runs/candidate_ranker_v5')
# This pins evidence preserved AFTER the parent-confirmed failed formal run.
# It is not a claim that the evidence had a historical pre-execution signature.
PRESERVED_NUMERIC_EVIDENCE_SHA256 = '04ac7a6ab4999b2a7d0afda4309e0508f38290686abef3f4d150c9c0c3cbdbbd'
EXPECTED_INPUT_KEYS = {
    'query_hidden', 'image_hidden', 'vision_hidden', 'source_features', 'candidate_features',
    'source_token', 'source_head', 'source_candidate', 'source_case', 'candidate_case',
    'last_query', 'candidate_lengths', 'attention_field', 'field_token', 'field_candidate',
    'frozen_point_features',
}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def text_file(path):
    raw = Path(path).read_bytes()
    return raw.decode('utf-16') if raw.startswith((b'\xff\xfe', b'\xfe\xff')) else raw.decode('utf-8-sig')


def check_hashes(mapping):
    for filename, digest in mapping.items():
        assert sha256(Path(filename)) == digest, f'Artifact changed: {filename}'


def path_key(path):
    # Index/source rows already provide absolute paths. Matching their spelling
    # needs no per-row filesystem realpath call; actual file identity is checked
    # independently with decoded RGB and SHA below. Unknown aliases fail closed.
    return os.path.normcase(os.path.abspath(str(path))).casefold()


def region_overlap(case, grid, parents):
    target = shape_from_geometry(case['target_geometry'])
    gh, gw = map(int, grid)
    return np.asarray([target.intersection(box((int(p) % gw)/gw, (int(p)//gw)/gh,
        (int(p) % gw+1)/gw, (int(p)//gw+1)/gh)).area > 0 for p in parents], bool)


def _same_original_rows(actual, original):
    ignored = {'experiment_split', 'split', 'training_extension'}
    assert {c['sample_id'] for c in actual} == {c['sample_id'] for c in original}
    expected = {c['sample_id']: c for c in original}
    for c in actual:
        for name, value in expected[c['sample_id']].items():
            if name not in ignored:
                assert c[name] == value, (c['sample_id'], name)


def audit_selection(out, protocol, cases):
    selection = read(out/'data_selection.json')
    assert sha256(out/'data_selection.json') == protocol['data_selection_sha256']
    assert sha256(Path('scripts/prepare_semantic_head_data.py')) == protocol['data_preparation_sha256'] == selection['source_script_sha256']
    assert selection['model_outputs_used_for_selection'] is False
    assert not selection['gpu_used'] and not selection['network_used']
    assert selection['base_rows_content_preserved_except_split_and_training_extension_metadata']
    assert selection['status'] in ('ready', 'ready_with_unresolved_history_references')
    for entry in selection['input_sources']:
        assert sha256(Path(entry['path'])) == entry['sha256'], entry['path']
    for entry in selection['outputs'].values():
        assert sha256(Path(entry['path'])) == entry['sha256']
    assert len({c['sample_id'] for c in cases}) == len(cases)
    counts = Counter(c['experiment_split'] for c in cases)
    assert dict(counts) == selection['counts']
    assert counts['train'] == protocol['train_queries_expanded']
    assert counts['val'] == protocol['validation_queries'] == 172
    assert counts['test'] == counts['diagnostic'] == 192
    base = [c for c in cases if c['experiment_split'] == 'train' and not c.get('training_extension')]
    extra = [c for c in cases if c.get('training_extension')]
    val = [c for c in cases if c['experiment_split'] == 'val']
    diagnostic = [c for c in cases if c['experiment_split'] == 'diagnostic']
    test = [c for c in cases if c['experiment_split'] == 'test']
    assert len(base) == protocol['train_queries_base'] == 618
    assert len(extra) == protocol['train_queries_extra'] == selection['extra_queries']
    previous = read_jsonl(Path('runs/conditioned_v3/manifest.jsonl'))
    _same_original_rows(base, [c for c in previous if c['experiment_split'] == 'train'])
    _same_original_rows(val, [c for c in previous if c['experiment_split'] == 'val'])
    _same_original_rows(diagnostic, [c for c in read_jsonl(BASE/'manifest.jsonl') if c['experiment_split'] == 'test'])
    split_pixels = {name: {c['pixel_sha256'] for c in cases if c['experiment_split'] == name} for name in counts}
    names = list(split_pixels)
    for i, name in enumerate(names):
        for other in names[i+1:]:
            assert not split_pixels[name] & split_pixels[other], (name, other)
    assert dict(Counter(c['dataset'] for c in test)) == protocol['fresh_test_quotas']
    assert all(n == 64 for n in protocol['fresh_test_quotas'].values())

    # Rebuild historical exact/near exclusions from the sealed reference list.
    public = read('data/conditioned_v3/public_image_index.json')
    assert public['complete']
    pool = read_jsonl(Path('data/conditioned_v3/training_pool_b8ed555f5e93.jsonl'))
    cache = read_jsonl(out/'data_selection_image_hash_cache.jsonl')
    info = {path_key(r['image']): r for r in public['images']}
    for r in pool:
        info.setdefault(path_key(r['image']), r)
    info.update({path_key(r['image']): r for r in cache})
    history = [Path(e['path']) for e in selection['input_sources']
        if 'runs' in Path(e['path']).parts and 'manifest' in Path(e['path']).name and Path(e['path']).suffix == '.jsonl']
    assert len(history) == selection['historical_manifest_count']
    resolutions = selection['historical_source_manifest_resolutions']
    unresolved = {(e['manifest'], e['sample_id']): e for e in selection['unresolved_historical_image_references']}
    assert all(e['provided_hashes_used'] for e in unresolved.values()), 'History without usable exact and near hashes'
    historical_pixels = set(); historical_hashes = set()
    protected_pixels = {r['pixel_sha256'] for r in public['images']}
    protected_hashes = {int(r['dhash256'], 16) for r in public['images']}
    for path in history:
        source = resolutions.get(str(path))
        originals = None
        if source:
            assert sha256(Path(source['source_manifest'])) == source['sha256']
            originals = {r['sample_id']: r for r in read_jsonl(Path(source['source_manifest']))}
        for c in read_jsonl(path):
            if not c.get('image') and not c.get('image_path'):
                continue
            raw_path = Path(c.get('image', c.get('image_path')))
            resolved = raw_path if raw_path.is_absolute() else path.parent/raw_path
            entry = info.get(path_key(resolved))
            if entry is None and originals is not None:
                original = originals[c['sample_id']]
                assert original.get('image') == c.get('image')
                raw = Path(original.get('image', original.get('image_path')))
                old_source = Path(source['source_manifest'])
                resolved = raw if raw.is_absolute() else old_source.parent/raw
                entry = info.get(path_key(resolved))
            if entry is None and c.get('pixel_sha256') and c.get('dhash256'):
                entry = c
            assert entry is not None, ('Unresolved historical screenshot', str(path), c.get('sample_id'))
            historical_pixels.add(entry['pixel_sha256']); historical_hashes.add(int(entry['dhash256'], 16))
            if c.get('experiment_split', c.get('split', 'unknown')) != 'train':
                protected_pixels.add(entry['pixel_sha256']); protected_hashes.add(int(entry['dhash256'], 16))
    assert len(historical_pixels) == selection['historical_pixel_count']
    for c in val+diagnostic:
        protected_pixels.add(c['pixel_sha256']); protected_hashes.add(int(c['dhash256'], 16))
    assert len(protected_pixels) == selection['public_or_historical_heldout_protected_pixel_count']
    radius = protocol['fresh_test_near_exclusion_radius']
    assert radius == 4
    seen_pixels = historical_pixels | {c['pixel_sha256'] for c in cases if c['experiment_split'] != 'test'}
    seen_hashes = historical_hashes | {int(c['dhash256'], 16) for c in cases if c['experiment_split'] != 'test'}
    nearest = []
    for c in test:
        assert c['pixel_sha256'] not in seen_pixels
        distance = min((int(c['dhash256'], 16)^h).bit_count() for h in seen_hashes)
        assert distance > radius, ('Near duplicate fresh screenshot', c['sample_id'], distance)
        nearest.append(distance); seen_pixels.add(c['pixel_sha256']); seen_hashes.add(int(c['dhash256'], 16))
    pool_by_id = {c['sample_id']: c for c in pool}
    for c in extra:
        assert c['experiment_split'] == c['source_split'] == 'train'
        assert c['official_training_split'] and c['expected_actionable']
        original = pool_by_id[c['sample_id']]
        for name in ('instruction', 'target_geometry', 'image', 'image_sha256', 'pixel_sha256', 'dhash256'):
            assert c[name] == original[name], (c['sample_id'], name)
        assert c['pixel_sha256'] not in protected_pixels
        assert all((int(c['dhash256'], 16)^h).bit_count() > radius for h in protected_hashes)
        if c['training_extension_origin'] == 'new_official_train_image_pair':
            assert c['pixel_sha256'] not in historical_pixels
            assert all((int(c['dhash256'], 16)^h).bit_count() > radius for h in historical_hashes)
        else:
            assert c['training_extension_origin'] == 'existing_train_image'
            assert c['pixel_sha256'] in {r['pixel_sha256'] for r in base}
    filenames = sorted({os.path.abspath(c['image']) for c in cases})
    with ThreadPoolExecutor(max_workers=4) as pool:
        verified_images = dict(zip(filenames, pool.map(fingerprint, filenames)))
    for c in cases:
        filename = os.path.abspath(c['image'])
        actual = verified_images[filename]
        assert actual['file_sha256'] == c['image_sha256'] and actual['pixel_sha256'] == c['pixel_sha256']
        assert actual['dhash256'] == c['dhash256'] and actual['image_size'] == c['image_size']
    return dict(counts=dict(counts), extra_train_queries=len(extra), historical_pixels=len(historical_pixels),
        historical_manifests=len(history), unresolved_history_with_provided_hashes=len(unresolved),
        fresh_unique_pixels=len(split_pixels['test']), fresh_near_min_distance=min(nearest),
        actual_image_files_and_pixels_verified=len(verified_images))


def audit_frozen(out, protocol, frozen):
    assert protocol['primary'] == frozen['primary_model'] == 'semantic_hard_s1__quality'
    assert protocol['test_features_required_absent_until_freeze'] and not protocol['test_used_for_fitting_or_checkpoint_selection']
    assert frozen['fresh_test_features_at_freeze'] == 0 and not frozen['fresh_test_used_for_training_or_selection']
    assert protocol['manifest_sha256'] == frozen['manifest_sha256'] == sha256(out/'manifest.jsonl')
    assert protocol['fixed_fine_sha256'] == frozen['fixed_fine_sha256'] == sha256(Path(protocol['fixed_fine_checkpoint']))
    check_hashes(protocol['unchanged_artifact_sha256']); check_hashes(protocol['baseline_checkpoint_sha256']); check_hashes(frozen['sources'])
    identity = read(out/'training_identity.json')
    assert identity['sources'] == frozen['sources'] and identity['manifest_sha256'] == protocol['manifest_sha256']
    for source, digest in frozen['sources'].items():
        assert sha256(out/'source_snapshot'/Path(source).name) == digest, source
    assert len(protocol['jobs']) == 10 and len(frozen['models']) == 30
    assert protocol['epochs'] == 24 and protocol['training_queries_per_epoch'] == 600
    assert protocol['batch_size'] == 8 and protocol['updates_per_epoch'] == 75
    assert protocol['parameter_counts']['static'] == protocol['parameter_counts']['semantic']
    assert sha256(Path(protocol['head_prior'])) == read(BASE/'training_identity.json')['prior_sha256']
    prior = np.asarray(read(protocol['head_prior'])['weights'], np.float32)
    expected_prior = np.log1p(96*prior/max(float(prior.sum()), 1e-9))
    epochs = {}; expected_models = set()
    for job in protocol['jobs']:
        h = read(out/'models'/f'{job["name"]}_history.json')
        assert h['complete'] and h['job'] == job
        assert h['updates_per_epoch'] == 75
        assert h['parameters'] == protocol['parameter_counts'][job['architecture']]['total']
        assert h['core_parameters'] == protocol['parameter_counts'][job['architecture']]['core']
        history = h['history']
        assert [r['epoch'] for r in history] == list(range(25))
        assert h['fit_named_queries'] >= 600
        for row in history:
            expected = 600 if row['epoch'] else 0
            assert row['consumed_queries'] == expected and row['optimizer_steps'] == (75 if row['epoch'] else 0)
            for value in (row['val']['rank_loss'], row['val']['quality_loss']):
                assert np.isfinite(value)
        for rule in ('region', 'point', 'quality'):
            chosen = max(history, key=lambda row: (*selectors(row['val'])[rule], -row['epoch']))
            name = job['name']+'__'+rule; expected_models.add(name)
            entry = frozen['models'][name]
            assert entry['job'] == job and entry['selector'] == rule
            assert entry['epoch'] == h['checkpoints'][rule]['epoch'] == chosen['epoch']
            assert entry['sha256'] == h['checkpoints'][rule]['sha256'] == sha256(out/'models'/entry['file'])
            assert entry['validation'] == chosen['val']
            saved = torch.load(out/'models'/entry['file'], map_location='cpu', weights_only=True)
            np.testing.assert_array_equal(saved['state_dict']['core.head_prior'].numpy(), expected_prior)
            assert saved['architecture'] == job['architecture'] and saved['metadata']['job'] == job
            assert saved['metadata']['epoch'] == chosen['epoch'] and saved['metadata']['selector'] == rule
            assert saved['metadata']['validation'] == chosen['val'] and saved['metadata']['backbone_trainable_parameters'] == 0
            epochs[name] = chosen['epoch']
    assert set(frozen['models']) == expected_models
    feature_identity = read(out/'feature_identity.json')
    assert feature_identity['manifest_sha256'] == protocol['manifest_sha256']
    assert feature_identity['max_pixels'] == feature_identity['config']['max_pixels'] == protocol['max_pixels'] == 262144
    assert feature_identity['gt_in_inputs'] is False
    assert feature_identity['qwen_weight_identity_sha256'] == sha256(Path('data/conditioned_v3/qwen_weight_identity.json'))
    check_hashes(feature_identity['source_sha256'])
    for source, digest in feature_identity['source_sha256'].items():
        assert sha256(out/'feature_sources'/Path(source).name) == digest
    for stage in ('extra_train', 'test'):
        runtime = read(out/f'runtime_{stage}.json')
        assert runtime['all_parameters_frozen'] and runtime['trainable_parameter_count'] == 0 and not runtime['model_training']
        assert runtime['parameter_dtype_counts'] == {'torch.float16': runtime['parameter_count']}
        completion = read(out/f'features_{stage}_completion.json')
        assert completion['complete'] and completion['manifest_sha256'] == protocol['manifest_sha256']
        assert completion['n'] == (192 if stage == 'test' else protocol['train_queries_extra'])
    pin_log = out/'pin_backbone.log'
    assert 'PINNED data' in text_file(pin_log)
    pin = read('data/conditioned_v3/qwen_weight_identity.json')
    for name, entry in pin['files'].items():
        path = Path(pin['model_path'])/name
        assert path.stat().st_size == entry['bytes'] and path.stat().st_mtime_ns <= pin_log.stat().st_mtime_ns
    return epochs


def raw_path(out, case):
    if case.get('training_extension') or case['experiment_split'] == 'test':
        root = out
    elif case['experiment_split'] == 'diagnostic':
        root = BASE
    else:
        root = Path(read(BASE/'protocol.json')['raw_feature_root_trainval'])
    return root/'features'/f'{key(case["sample_id"])}.npz'


def audit_records(out, protocol, records_by_split):
    freeze_time = (out/'frozen_readouts.json').stat().st_mtime_ns
    sealed_inputs = read(out/'training_identity.json')['train_val_cache_inventory']
    expected_head_order = [f'layer_{layer}.head_{head}' for layer in (15, 28, 36) for head in range(32)]
    reconstructed = Counter(); named = Counter(); nonpeak = 0; source_tokens = 0
    for split, records in records_by_split.items():
        completion = read(out/f'point_cache_{split}_completion.json')
        assert completion['complete'] and completion['n'] == len(records)
        assert completion['manifest_sha256'] == protocol['manifest_sha256'] and completion['fine_sha256'] == protocol['fixed_fine_sha256']
        for r in records:
            c = r['case']; path = raw_path(out, c)
            point_path = out/'point_cache'/f'{key(c["sample_id"])}.npz'
            with np.load(point_path, allow_pickle=False) as point_archive:
                point_meta = json.loads(str(point_archive['metadata']))
                assert point_meta['sample_id'] == c['sample_id'] and point_meta['input_identity'] == r['cache_identity']
                assert point_meta['fine_sha256'] == protocol['fixed_fine_sha256'] and point_meta['manifest_sha256'] == protocol['manifest_sha256']
                assert point_meta['target_available'] == r['metadata']['target_available']
                if not point_meta['target_available']:
                    assert set(point_archive.files) == {'metadata'}
            if split in ('train', 'val'):
                sealed = sealed_inputs[c['sample_id']]
                assert sealed['point_cache_sha256'] == sha256(point_path)
                assert sealed['input_identity'] == r['cache_identity']
            if split == 'test':
                assert path.stat().st_mtime_ns >= freeze_time
            with np.load(path, allow_pickle=False) as archive:
                meta = json.loads(str(archive['metadata']))
                assert meta['sample_id'] == c['sample_id'] and meta['image_sha256'] == c['image_sha256']
                assert meta['pixel_sha256'] == c['pixel_sha256'] and meta['image_size'] == c['image_size']
                assert meta['own_token_forward'] and meta['ground_truth_in_model_input'] is False and meta['all_backbone_parameters_frozen']
                assert meta['target_available'] == r['metadata']['target_available']
                if not meta['target_available']:
                    continue
                assert meta['head_order'] == expected_head_order
                built = build_candidate_inputs(archive['target_attention'], archive['target_hidden'], archive['image_hidden'], archive['vision_hidden'], meta['grid_shape'])
                for name, array in built.items():
                    np.testing.assert_array_equal(array, r[name], err_msg=f'{c["sample_id"]}: {name}')
                field = build_attention_field(archive['target_attention'], built['parent_indices'])
                np.testing.assert_array_equal(field, r['attention_field'])
                assert len(meta['target_raw_indices']) == len(built['query_hidden']) > 0
                assert all(i > 0 for i in meta['target_raw_indices'])
                nonpeak += int((field[..., 5] == 0).sum()); source_tokens += len(built['query_hidden'])
            assert r['metadata']['source_archive_sha256'] == sha256(path)
            named[split] += 1; reconstructed[split] += 1
            with np.load(point_path, allow_pickle=False) as cached:
                point_meta = json.loads(str(cached['metadata']))
                assert point_meta['sample_id'] == c['sample_id'] and point_meta['labels_are_supervision_only']
                assert point_meta['fine_sha256'] == protocol['fixed_fine_sha256'] and point_meta['manifest_sha256'] == protocol['manifest_sha256']
                np.testing.assert_array_equal(cached['parent_indices'], r['parent_indices'])
                assert set(cached.files) == {'metadata', 'parent_indices', 'frozen_points', 'frozen_point_features', 'label_fixed_point_hit', 'label_hard_negative'}
            independent_region = region_overlap(c, r['metadata']['grid_shape'], r['parent_indices'])
            np.testing.assert_array_equal(independent_region, r['label_area'].sum(-1) > 0)
            np.testing.assert_array_equal(r['label_fixed_point_hit'], [score_point(p.tolist(), c) for p in r['frozen_points']])
            # Explicitly check the final model-input whitelist and labels=False.
            inputs, targets = make_batch([r], 'cpu', labels=False)
            assert targets is None and set(inputs) == EXPECTED_INPUT_KEYS
            assert inputs['frozen_point_features'].shape == (len(r['parent_indices']), 13)
            assert torch.isfinite(inputs['frozen_point_features']).all()
        assert completion['named'] == named[split]
        print('RAW AND LABEL INPUT AUDIT '+split+' '+str(named[split]), flush=True)
    return dict(raw_inputs_reconstructed=dict(reconstructed), named=dict(named),
        generated_name_tokens_checked=source_tokens, nonpeak_attention_entries=nonpeak)


def frozen_point_arrays(output, record, cursor):
    """Reconstruct canonical cache fields without using a target or hit label.

    ``output`` is CPU float32 even for a CUDA forward, matching the original
    cache writer's softmax, feature construction and pixel rounding exactly.
    Continuous coordinates are retained only for numeric diagnostics.
    """
    n = len(record['parent_indices']); c = record['case']
    logits = output['child_logits'][cursor:cursor+n]
    probability = logits.softmax(-1).numpy()
    # The original cache descriptors choose from softmax probabilities, while
    # decode_ranker itself chooses from raw logits. Usually identical, but keep
    # both exact operation paths for a possible float32 softmax tie.
    feature_child = probability.argmax(-1)
    all_offsets = output['offsets'][cursor:cursor+n].numpy()
    feature_offsets = all_offsets[np.arange(n), feature_child]
    child = logits.numpy().argmax(-1)
    offsets = all_offsets[np.arange(n), child]
    gh, gw = record['metadata']['grid_shape']; rr, cc = np.divmod(record['parent_indices'], gw)
    w, h = c['image_size']; continuous = np.empty((n, 2), np.float64)
    # Preserve decode_ranker's exact operation order, including divide before
    # multiplying by image size; half-pixel diagnostics need those last bits.
    for j in range(n):
        dx, dy = offsets[j]
        xy = [(int(cc[j])*2+int(child[j]) % 2+float(dx))/(2*gw),
              (int(rr[j])*2+int(child[j])//2+float(dy))/(2*gh)]
        continuous[j] = [xy[0]*w, xy[1]*h]
    points = np.clip(np.rint(continuous), 0, np.asarray([w-1, h-1])).astype(np.int64)
    entropy = -(probability*np.log(np.maximum(probability, 1e-12))).sum(-1)/np.log(4)
    xy = points/np.array(c['image_size'])[None]
    relative = xy*np.array([gw, gh])[None]-np.stack((cc, rr), -1)
    features = np.concatenate((probability, feature_offsets, entropy[:, None], xy, relative,
        np.tile([1/gw, 1/gh], (n, 1))), -1)
    return dict(points=points, features=features, features_float16=features.astype(np.float16),
        continuous=continuous, child=child, selected_offsets=offsets)


def verify_canonical_gpu_batch(group, cpu_output, gpu_output_cpu, split, batch_start, strict_features=True):
    """Reject any canonical mismatch; report CPU-only half-pixel exceptions.

    No integer-coordinate tolerance is used to accept a cache. Every CUDA point
    and every stored float16 feature in the entire original batch must match.
    """
    lengths = [len(r['parent_indices']) for r in group]
    assert list(cpu_output['candidate_lengths']) == list(gpu_output_cpu['candidate_lengths']) == lengths
    exceptions = []; cursor = 0
    for r in group:
        c = r['case']; n = len(r['parent_indices'])
        cpu = frozen_point_arrays(cpu_output, r, cursor)
        gpu = frozen_point_arrays(gpu_output_cpu, r, cursor)
        np.testing.assert_array_equal(gpu['points'], r['frozen_points'], err_msg='Canonical GPU point mismatch: '+c['sample_id'])
        if strict_features:
            np.testing.assert_array_equal(gpu['features_float16'], r['frozen_point_features'], err_msg='Canonical GPU feature mismatch: '+c['sample_id'])
        mismatch = np.flatnonzero(np.any(cpu['points'] != r['frozen_points'], axis=1))
        cpu_hits = np.asarray([score_point(p.tolist(), c) for p in cpu['points']], bool)
        canonical_hits = np.asarray([score_point(p.tolist(), c) for p in gpu['points']], bool)
        for ci in mismatch:
            ci = int(ci); axes = np.flatnonzero(cpu['points'][ci] != gpu['points'][ci])
            assert cpu['child'][ci] == gpu['child'][ci], 'CPU/GPU changed fine child, not merely rounding'
            boundaries = []
            for axis in axes:
                half = float(min(cpu['points'][ci, axis], gpu['points'][ci, axis])+.5)
                cpu_distance = abs(float(cpu['continuous'][ci, axis])-half)
                gpu_distance = abs(float(gpu['continuous'][ci, axis])-half)
                delta = abs(float(cpu['continuous'][ci, axis])-float(gpu['continuous'][ci, axis]))
                # These classify the exception; acceptance above is exact GPU
                # replay, never a one-pixel tolerance on the stored point.
                assert abs(int(cpu['points'][ci, axis])-int(gpu['points'][ci, axis])) == 1
                assert cpu_distance < .001 and gpu_distance < .001 and delta < .001
                boundaries.append(dict(axis='xy'[int(axis)], boundary=half,
                    cpu_distance_px=cpu_distance, canonical_gpu_distance_px=gpu_distance,
                    continuous_delta_px=delta))
            exceptions.append(dict(split=split, batch_start=batch_start,
                sample_id=c['sample_id'], candidate_index=ci, parent_index=int(r['parent_indices'][ci]),
                cpu_point=cpu['points'][ci].tolist(), canonical_gpu_point=gpu['points'][ci].tolist(),
                stored_point=r['frozen_points'][ci].tolist(),
                cpu_continuous_point=cpu['continuous'][ci].tolist(),
                canonical_gpu_continuous_point=gpu['continuous'][ci].tolist(),
                cpu_selected_offsets=cpu['selected_offsets'][ci].tolist(),
                canonical_gpu_selected_offsets=gpu['selected_offsets'][ci].tolist(),
                selected_child=int(gpu['child'][ci]), half_pixel_boundaries=boundaries,
                cpu_point_hit=bool(cpu_hits[ci]), canonical_gpu_point_hit=bool(canonical_hits[ci]),
                hit_changed=bool(cpu_hits[ci] != canonical_hits[ci]),
                cpu_point_oracle=bool(cpu_hits.any()), canonical_gpu_point_oracle=bool(canonical_hits.any()),
                point_oracle_changed=bool(cpu_hits.any() != canonical_hits.any()),
                classified_as_half_pixel_rounding=True, canonical_points_exact=True,
                canonical_all_13_float16_features_exact=bool(strict_features)))
        cursor += n
    assert exceptions, 'Canonical fallback was invoked without an integer difference'
    batch = dict(split=split, batch_start=batch_start, samples=[r['case']['sample_id'] for r in group],
        n_candidates=sum(lengths), all_gpu_points_exact=True, all_gpu_13_float16_features_exact=bool(strict_features))
    return exceptions, batch


def verify_canonical_gpu_repeats(group, cpu_output, gpu_outputs_cpu, split, batch_start):
    """Fixed five-run elementwise evidence and adjacent-FP16 reproducibility.

    All integer points and coordinate-derived feature columns 7:13 are exact
    in every run. Each descriptor must reproduce its cached bit pattern in at
    least one actual run; a simultaneous whole-batch witness is reported
    separately. Variation must cross/touch an adjacent FP16 midpoint.
    """
    assert len(gpu_outputs_cpu) == 5, 'Canonical reproducibility audit requires exactly five runs'
    lengths = [len(r['parent_indices']) for r in group]
    cache = np.concatenate([r['frozen_point_features'] for r in group])
    origins = [(r['case']['sample_id'], ci) for r in group for ci in range(len(r['parent_indices']))]
    raw = []; quantized = []; runs = []
    for run_index, output in enumerate(gpu_outputs_cpu):
        assert list(output['candidate_lengths']) == lengths
        cursor = 0; pieces = []; point_differences = []; coordinate_differences = []
        for r in group:
            arrays = frozen_point_arrays(output, r, cursor)
            for ci, axis in np.argwhere(arrays['points'] != r['frozen_points']):
                point_differences.append(dict(sample_id=r['case']['sample_id'], candidate_index=int(ci),
                    axis='xy'[int(axis)], actual=int(arrays['points'][ci, axis]), stored=int(r['frozen_points'][ci, axis]),
                    continuous=float(arrays['continuous'][ci, axis])))
            for ci, channel in np.argwhere(arrays['features_float16'][:, 7:] != r['frozen_point_features'][:, 7:]):
                channel = int(channel)+7
                coordinate_differences.append(dict(sample_id=r['case']['sample_id'], candidate_index=int(ci),
                    channel=channel, actual=float(arrays['features_float16'][ci, channel]),
                    stored=float(r['frozen_point_features'][ci, channel])))
            pieces.append(arrays['features']); cursor += len(r['parent_indices'])
        full = np.concatenate(pieces); q = full.astype(np.float16)
        raw.append(full[:, :7].astype(np.float32)); quantized.append(q)
        differences = np.argwhere(q != cache)
        runs.append(dict(run=run_index, integer_point_mismatches=len(point_differences),
            coordinate_feature_mismatches=len(coordinate_differences), point_mismatches=point_differences,
            coordinate_mismatches=coordinate_differences,
            float16_feature_mismatch_elements=len(differences), mismatches=[dict(
                sample_id=origins[int(ci)][0], candidate_index=origins[int(ci)][1], channel=int(ch),
                source_float32=float(full[ci, ch]), actual_float16=float(q[ci, ch]), cached_float16=float(cache[ci, ch]))
                for ci, ch in differences]))
        print('CANONICAL GPU FIXED REPEAT '+json.dumps(dict(split=split, batch_start=batch_start, **runs[-1])), flush=True)
    raw = np.stack(raw); quantized = np.stack(quantized)
    assert all(r['integer_point_mismatches'] == 0 for r in runs), 'Canonical GPU integer point mismatch in fixed five runs'
    assert all(r['coordinate_feature_mismatches'] == 0 for r in runs), 'Canonical GPU coordinate-feature mismatch in fixed five runs'
    witness = [i for i in range(5) if np.array_equal(quantized[i], cache)]
    assert np.all(np.any(quantized == cache[None], axis=0)), 'Cached descriptor element was never reproduced in the fixed five GPU runs'
    boundaries = []
    for ci, ch in np.argwhere(np.any(quantized != cache[None], axis=0)):
        ci = int(ci); ch = int(ch)
        assert ch < 7, 'Coordinate-derived columns are never allowed a quantization exception'
        observed = np.unique(quantized[:, ci, ch])
        assert len(observed) == 2, 'Descriptor must vary between exactly two adjacent FP16 values'
        low, high = map(float, observed)
        assert float(np.nextafter(np.float16(low), np.float16(np.inf))) == high, 'Descriptor difference exceeds one adjacent FP16 step'
        midpoint = (low+high)/2; values = raw[:, ci, ch]
        assert float(values.min()) <= midpoint <= float(values.max()), 'Measured float32 range does not cross/touch the FP16 midpoint'
        assert bool((observed == cache[ci, ch]).any()), 'Cached descriptor pattern was never reproduced'
        boundaries.append(dict(sample_id=origins[ci][0], candidate_index=origins[ci][1], channel=ch,
            source_float32_runs=[float(v) for v in values], source_float32_min=float(values.min()),
            source_float32_max=float(values.max()), float16_low=low, float16_high=high,
            float16_ulp=high-low, float16_midpoint=midpoint, cached_float16=float(cache[ci, ch]),
            per_run_float16=[float(v) for v in quantized[:, ci, ch]],
            real_range_crosses_or_touches_midpoint=True, cached_pattern_observed=True))
    reference = witness[0] if witness else 0
    events, batch = verify_canonical_gpu_batch(group, cpu_output, gpu_outputs_cpu[reference], split, batch_start, strict_features=bool(witness))
    batch.pop('all_gpu_13_float16_features_exact')
    bitstable = all(np.array_equal(quantized[0], quantized[i]) for i in range(1, 5))
    batch.update(fixed_gpu_replay_count=5, all_integer_points_exact_every_run=True,
        coordinate_columns_7_to_13_exact_every_run=True, witness_exact=True,
        whole_batch_13_column_exact_witness_runs=witness, interrun_features_bitstable=bitstable,
        interrun_not_bitstable=not bitstable, runs=runs, explained_float16_boundaries=boundaries,
        max_source_float32_descriptor_run_range=float((raw.max(0)-raw.min(0)).max()),
        no_adaptive_or_additional_retries=True)
    batch.update(witness_exact=bool(witness), whole_batch_13_exact=bool(witness),
        all_13_columns_elementwise_reproduced=True, descriptor_elements_reproduced=True,
        evidence_kind='direct_fixed_five_gpu_outputs', fp32_bound_kind='directly_measured_all_five_float32_values')
    for event in events:
        event['canonical_exact_witness_run'] = witness[0] if witness else None
        event['canonical_all_13_float16_features_exact'] = bool(witness)
        event['canonical_all_13_float16_elements_reproduced'] = True
        event['canonical_descriptor_interrun_bitstable'] = bitstable
    return events, batch


def verify_preserved_numeric_evidence(out, group, cpu_output, split, batch_start):
    """Authenticate the retained formal five-run componentwise evidence.

    Exact-cache elements were not printed with their source FP32 values. Their
    quantized outputs imply rounding-bin intervals, which we label as inferred
    bounds, never invented direct measurements. No new CUDA draws are made.
    """
    folder = out/'numeric_evidence'; manifest_path = folder/'evidence_manifest.json'
    assert sha256(manifest_path) == PRESERVED_NUMERIC_EVIDENCE_SHA256
    evidence = read(manifest_path)
    assert evidence['evidence_kind'] == 'authenticated_preserved_fixed_five_log'
    assert evidence['split'] == split and evidence['batch_start'] == batch_start
    assert evidence['samples'] == [r['case']['sample_id'] for r in group]
    assert evidence['n_candidates'] == sum(len(r['parent_indices']) for r in group)
    check_hashes(evidence['files'])
    # These are explicitly preservation-time assertions. The retained source,
    # logs and parent-authorized unchanged-state provenance are the execution
    # evidence; the following are not retroactive cryptographic commitments.
    check_hashes(evidence['context_sha256_asserted_at_preservation'])
    check_hashes(evidence['point_cache_sha256_asserted_at_preservation'])
    log_path = folder/'failed_formal_fixed_five.log'
    runs = [json.loads(line.split('CANONICAL GPU FIXED REPEAT ', 1)[1])
        for line in text_file(log_path).splitlines() if 'CANONICAL GPU FIXED REPEAT ' in line]
    assert len(runs) == 5 and [r['run'] for r in runs] == list(range(5))
    by_sid = {r['case']['sample_id']: r for r in group}; cells = defaultdict(dict)
    for run in runs:
        assert run['split'] == split and run['batch_start'] == batch_start
        assert run['integer_point_mismatches'] == run['coordinate_feature_mismatches'] == 0
        assert not run['point_mismatches'] and not run['coordinate_mismatches']
        assert run['float16_feature_mismatch_elements'] == len(run['mismatches'])
        for item in run['mismatches']:
            sid = item['sample_id']; ci = item['candidate_index']; ch = item['channel']
            assert sid in by_sid and 0 <= ci < len(by_sid[sid]['parent_indices']) and 0 <= ch < 7
            assert float(by_sid[sid]['frozen_point_features'][ci, ch]) == item['cached_float16']
            raw = item['source_float32']; actual = item['actual_float16']
            assert np.isfinite(raw) and float(np.float32(raw)) == raw
            assert float(np.float16(raw)) == actual and actual != item['cached_float16']
            key_ = (sid, ci, ch)
            assert run['run'] not in cells[key_]
            cells[key_][run['run']] = item
    boundaries = []
    for (sid, ci, ch), differences in sorted(cells.items()):
        cache = float(by_sid[sid]['frozen_point_features'][ci, ch])
        observed_runs = [i for i in range(5) if i not in differences]
        assert observed_runs, 'Cached element never appeared in retained CURRENT five runs'
        values = [differences[i]['actual_float16'] if i in differences else cache for i in range(5)]
        unique = sorted(set(values)); assert len(unique) == 2
        low, high = unique
        assert float(np.nextafter(np.float16(low), np.float16(np.inf))) == high
        midpoint = (low+high)/2
        previous = float(np.nextafter(np.float16(cache), np.float16(-np.inf)))
        following = float(np.nextafter(np.float16(cache), np.float16(np.inf)))
        cache_bin = [(previous+cache)/2, (cache+following)/2]
        intervals = []; direct = []
        for i in range(5):
            if i in differences:
                value = differences[i]['source_float32']; direct.append(value)
                intervals.append(dict(kind='directly_logged_float32', lower=value, upper=value))
            else:
                direct.append(None)
                intervals.append(dict(kind='inferred_from_exact_cached_float16_rounding_bin',
                    lower=cache_bin[0], upper=cache_bin[1], conservative_closed_bounds=True))
        # Observed low/high quantizer outputs prove their real source range
        # touches/crosses the common midpoint even though exact-cache FP32
        # source values were not logged. Do not substitute invented raw maxima.
        assert low in values and high in values
        for i, value in enumerate(values):
            interval = intervals[i]
            assert interval['upper'] <= midpoint if value == low else interval['lower'] >= midpoint
        minimum_bounds = [min(v['lower'] for v in intervals), min(v['upper'] for v in intervals)]
        maximum_bounds = [max(v['lower'] for v in intervals), max(v['upper'] for v in intervals)]
        boundaries.append(dict(sample_id=sid, candidate_index=ci, channel=ch,
            source_float32_runs=direct, source_float32_interval_runs=intervals,
            source_float32_min=minimum_bounds[0] if minimum_bounds[0] == minimum_bounds[1] else None,
            source_float32_max=maximum_bounds[0] if maximum_bounds[0] == maximum_bounds[1] else None,
            source_float32_min_enclosure=minimum_bounds, source_float32_max_enclosure=maximum_bounds,
            fp32_bound_kind='direct_mismatch_values_and_inferred_exact_cache_rounding_bins',
            matching_run_fp32_values_inferred=True, float16_low=low, float16_high=high,
            float16_ulp=high-low, float16_midpoint=midpoint, cached_float16=cache,
            per_run_float16=values, cached_pattern_observed_current_runs=observed_runs,
            real_range_crosses_or_touches_midpoint=True, midpoint_proof_kind='observed_adjacent_low_and_high_quantizer_outputs',
            cached_pattern_observed=True))
    witness = [r['run'] for r in runs if r['float16_feature_mismatch_elements'] == 0]
    prior_path = folder/'prior_single_exact_witness.json'; prior = read(prior_path)
    assert prior['checkpoint_sha256'] == read(out/'protocol.json')['fixed_fine_sha256']
    old_events = {(e['sample_id'], e['candidate_index']): e for e in prior['cpu_numeric_exceptions']}
    events = []; cursor = 0
    for r in group:
        c = r['case']; arrays = frozen_point_arrays(cpu_output, r, cursor)
        cpu_hits = [bool(score_point(p.tolist(), c)) for p in arrays['points']]
        stored_hits = [bool(score_point(p.tolist(), c)) for p in r['frozen_points']]
        for ci in np.flatnonzero(np.any(arrays['points'] != r['frozen_points'], axis=1)):
            ci = int(ci); old = old_events[c['sample_id'], ci]
            assert old['parent_index'] == int(r['parent_indices'][ci])
            assert old['cpu_point'] == arrays['points'][ci].tolist()
            assert old['canonical_gpu_point'] == r['frozen_points'][ci].tolist()
            event = dict(old)
            event.update(cpu_continuous_point=arrays['continuous'][ci].tolist(),
                cpu_point_hit=cpu_hits[ci], canonical_gpu_point_hit=stored_hits[ci],
                hit_changed=cpu_hits[ci] != stored_hits[ci], cpu_point_oracle=any(cpu_hits),
                canonical_gpu_point_oracle=any(stored_hits), point_oracle_changed=any(cpu_hits) != any(stored_hits),
                canonical_points_exact=True, canonical_all_13_float16_features_exact=bool(witness),
                canonical_all_13_float16_elements_reproduced=True,
                canonical_exact_witness_run=witness[0] if witness else None,
                canonical_descriptor_interrun_bitstable=False,
                canonical_gpu_continuous_point_evidence=dict(kind='separate_prior_single_exact_witness_execution',
                    path=str(prior_path), sha256=sha256(prior_path), not_current_formal_five_measurement=True),
                current_formal_five_gpu_continuous_values_were_logged=False)
            events.append(event)
        cursor += len(r['parent_indices'])
    assert events
    batch = dict(split=split, batch_start=batch_start, samples=evidence['samples'], n_candidates=evidence['n_candidates'],
        all_gpu_points_exact=True, fixed_gpu_replay_count=5, all_integer_points_exact_every_run=True,
        coordinate_columns_7_to_13_exact_every_run=True, witness_exact=bool(witness), whole_batch_13_exact=bool(witness),
        whole_batch_13_column_exact_witness_runs=witness, all_13_columns_elementwise_reproduced=True,
        descriptor_elements_reproduced=True, interrun_features_bitstable=not cells, interrun_not_bitstable=bool(cells),
        runs=runs, explained_float16_boundaries=boundaries, no_adaptive_or_additional_retries=True,
        evidence_kind='authenticated_preserved_fixed_five_log', fp32_bound_kind='direct_mismatch_values_and_inferred_exact_cache_rounding_bins',
        fp32_matching_run_bounds_inferred=True, log_path=str(log_path), log_sha256=sha256(log_path),
        evidence_manifest_sha256=PRESERVED_NUMERIC_EVIDENCE_SHA256,
        identity_binding_kind='asserted_at_post_execution_preservation_not_historical_precommit',
        provenance=evidence['provenance'], no_new_gpu_draws=True,
        prior_joint_witness_is_supplementary_only=True)
    return events, batch


def enrich_numeric_exceptions(exceptions, lookup, frozen):
    for event in exceptions:
        comparisons = {}
        for seed in ('s1', 's2'):
            method = f'semantic_hard_{seed}__quality__combined'
            row = lookup[event['split']][method, event['sample_id']]
            selected = row.get('selected_candidate_index')
            uses_exception = selected == event['candidate_index']
            comparisons[method] = dict(selected_candidate_index=selected,
                selects_differing_candidate=uses_exception, stored_point=row['point'], stored_hit=row['hit'],
                hypothetical_cpu_point_hit=event['cpu_point_hit'] if uses_exception else row['hit'])
        event['prespecified_primary_methods'] = comparisons
        event['primary_s1_selects_differing_candidate'] = comparisons[frozen['primary_model']+'__combined']['selects_differing_candidate']


@torch.inference_mode()
def replay_numeric_case(out, sample_id, split='test'):
    """Replay one original named batch, without running the full final audit."""
    protocol = read(out/'protocol.json'); frozen = read(out/'frozen_readouts.json')
    assert sha256(Path(protocol['fixed_fine_checkpoint'])) == protocol['fixed_fine_sha256']
    named = []
    for case in read_jsonl(out/'manifest.jsonl'):
        if case['experiment_split'] != split:
            continue
        cache_root = out if split == 'test' else BASE
        with np.load(cache_root/'candidate_cache'/f'{key(case["sample_id"])}.npz', allow_pickle=False) as archive:
            if json.loads(str(archive['metadata']))['target_available']:
                named.append(case)
    index = next(i for i, c in enumerate(named) if c['sample_id'] == sample_id)
    start = index//8*8; group = []; point_hashes = {}
    for case in named[start:start+8]:
        record = load_input_record(out, case)
        path = out/'point_cache'/f'{key(case["sample_id"])}.npz'; point_hashes[str(path)] = sha256(path)
        with np.load(path, allow_pickle=False) as archive:
            meta = json.loads(str(archive['metadata']))
            assert meta['input_identity'] == record['cache_identity'] and meta['fine_sha256'] == protocol['fixed_fine_sha256']
            for name in ('frozen_points', 'frozen_point_features', 'label_fixed_point_hit', 'label_hard_negative'):
                record[name] = archive[name].copy()
        group.append(record)
    cpu_model, _ = load_ranker(Path(protocol['fixed_fine_checkpoint']), 'cpu')
    inputs, labels = make_ranker_batch(group, 'cpu', labels=False); assert labels is None
    cpu_output = cpu_model(**inputs); del cpu_model
    if split == 'test' and start == 80 and (out/'numeric_evidence/evidence_manifest.json').exists():
        events, batch = verify_preserved_numeric_evidence(out, group, cpu_output, split, start)
    else:
        assert torch.cuda.is_available(), 'Numeric canonical replay requires cuda:0'
        gpu_model, _ = load_ranker(Path(protocol['fixed_fine_checkpoint']), 'cuda:0')
        inputs, labels = make_ranker_batch(group, 'cuda:0', labels=False); assert labels is None
        gpu_outputs = []
        for repeat in range(5):
            output = gpu_model(**inputs)
            gpu_outputs.append({k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in output.items()})
        del gpu_model
        events, batch = verify_canonical_gpu_repeats(group, cpu_output, gpu_outputs, split, start)
    lookup = {split: {(r['method'], r['sample_id']): r for r in read_jsonl(out/'evaluation'/split/'predictions.jsonl')}}
    enrich_numeric_exceptions(events, lookup, frozen)
    check_hashes(point_hashes)
    assert sha256(Path(protocol['fixed_fine_checkpoint'])) == protocol['fixed_fine_sha256']
    result = dict(complete=True, scope='Only one original cached eight-name batch, not the final full audit',
        requested_sample_id=sample_id, canonical_device='cuda:0',
        evidence_kind=batch['evidence_kind'], no_new_gpu_draws=batch.get('no_new_gpu_draws', False),
        checkpoint_sha256=protocol['fixed_fine_sha256'], batch=batch, cpu_numeric_exceptions=events,
        cache_files_unchanged=True, auditor_sha256=sha256(Path(__file__)))
    write_json(out/'numeric_replay_five.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


@torch.inference_mode()
def audit_fixed_points(out, protocol, records_by_split):
    model, _ = load_ranker(Path(protocol['fixed_fine_checkpoint']), 'cpu')
    groups = defaultdict(list)
    for r in records_by_split['train']:
        groups[r['case']['pixel_sha256']].append(r['case'])
    checked = Counter(); candidates = 0; gpu_model = None; numeric_exceptions = []; gpu_batches = []
    for split, records in records_by_split.items():
        named = [r for r in records if r['metadata']['target_available']]
        for start in range(0, len(named), 8):
            group = named[start:start+8]
            inputs, labels = make_ranker_batch(group, 'cpu', labels=False)
            assert labels is None
            output = model(**inputs); cursor = 0
            cpu_arrays = []
            for r in group:
                cpu_arrays.append(frozen_point_arrays(output, r, cursor)); cursor += len(r['parent_indices'])
            mismatched = any(not np.array_equal(a['points'], r['frozen_points']) for a, r in zip(cpu_arrays, group))
            if mismatched:
                if split == 'test' and start == 80 and (out/'numeric_evidence/evidence_manifest.json').exists():
                    events, batch = verify_preserved_numeric_evidence(out, group, output, split, start)
                else:
                    assert torch.cuda.is_available(), 'Canonical GPU replay is required for a CPU integer difference'
                    if gpu_model is None:
                        gpu_model, _ = load_ranker(Path(protocol['fixed_fine_checkpoint']), 'cuda:0')
                    gpu_inputs, gpu_labels = make_ranker_batch(group, 'cuda:0', labels=False)
                    assert gpu_labels is None
                    canonical_runs = []
                    for repeat in range(5):
                        gpu_output = gpu_model(**gpu_inputs)
                        canonical_runs.append({k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in gpu_output.items()})
                    events, batch = verify_canonical_gpu_repeats(group, output, canonical_runs, split, start)
                numeric_exceptions.extend(events); gpu_batches.append(batch)
                print('CANONICAL NUMERIC EVIDENCE '+json.dumps(dict(batch=batch, exceptions=events)), flush=True)
            cursor = 0
            for i, (r, arrays) in enumerate(zip(group, cpu_arrays)):
                c = r['case']; n = len(r['parent_indices'])
                decoded = np.empty((n, 2), np.int64)
                for pred in decode_ranker(output, r, i, top_k=n):
                    decoded[pred['candidate_index']] = pred['point']
                np.testing.assert_array_equal(decoded, arrays['points'])
                same = np.all(arrays['points'] == r['frozen_points'], axis=1)
                np.testing.assert_array_equal(arrays['points'][same], r['frozen_points'][same], err_msg=c['sample_id'])
                gh, gw = r['metadata']['grid_shape']; rr, cc = np.divmod(r['parent_indices'], gw)
                # CPU/GPU float32 reductions may differ before float16 storage.
                # No geometry is consulted in the feature reconstruction.
                np.testing.assert_allclose(r['frozen_point_features'][:, :7], arrays['features'][:, :7], rtol=.003, atol=.0007, err_msg=c['sample_id'])
                np.testing.assert_array_equal(r['frozen_point_features'][same, 7:], arrays['features_float16'][same, 7:])
                if split == 'train':
                    others = [o for o in groups[c['pixel_sha256']] if o['sample_id'] != c['sample_id']]
                    other_shape = unary_union([shape_from_geometry(o['target_geometry']) for o in others]) if others else None
                    hard = np.asarray([bool(other_shape is not None and other_shape.intersection(box(
                        col/gw, row/gh, (col+1)/gw, (row+1)/gh)).area > 0 and r['label_area'][j].sum() == 0)
                        for j, (row, col) in enumerate(zip(rr, cc))], bool)
                else:
                    hard = np.zeros(n, bool)
                np.testing.assert_array_equal(hard, r['label_hard_negative'])
                checked[split] += 1; candidates += n; cursor += n
        print('FROZEN POINT CPU AUDIT '+split+' '+str(checked[split]), flush=True)
    if gpu_model is not None:
        del gpu_model
        torch.cuda.empty_cache()
    return dict(fixed_point_cases=dict(checked), fixed_candidate_points=candidates,
        fixed_point_features_reconstructed_without_geometry=True, hard_negative_labels_recomputed=True,
        all_cpu_integer_points_exact=not numeric_exceptions,
        cpu_integer_difference_candidates=len(numeric_exceptions), cpu_numeric_exceptions=numeric_exceptions,
        canonical_gpu_exception_batches=gpu_batches,
        canonical_exception_acceptance='Fixed-five componentwise evidence: every point and coordinate column exact every run; every cached descriptor actually reproduced, variations only adjacent FP16 with directly measured or explicitly inferred quantization-bin midpoint evidence. Joint witness reported separately; no coordinate tolerance, new draws for retained batch, or adaptive retries.')


def audit_native_and_ui(out, records_by_split):
    count = 0
    for split in ('diagnostic', 'test'):
        root = out if split == 'test' else BASE
        ui_audit = read(root/'ui_tars/final_audit.json')
        ui_summary = read(root/'ui_tars/summary.json')
        assert ui_audit['complete'] and ui_summary['complete']
        assert ui_audit['same_test_images'] == ui_audit['coordinate_conversions_replayed'] == ui_summary['n'] == 192
        assert ui_audit['official_weight_sha_verification'] and ui_audit['trainable_parameters'] == 0
        assert ui_audit['source_hashes_match'] and ui_audit['max_pixels'] == 262144
        assert all(ui_audit['official_nonweight_git_blobs_match'].values()) and all(ui_audit['official_lfs_hashes_match'].values())
        ui_identity = read(root/'ui_tars/identity.json')
        assert ui_identity['manifest_sha256'] == sha256(root/'manifest.jsonl')
        assert ui_identity['driver_sha256'] == sha256(Path('scripts/run_ui_tars_baseline.py'))
        assert ui_identity['runtime_source_sha256'] == sha256(Path('src/clickagent/model_baselines.py'))
        for r in records_by_split[split]:
            c = r['case']; native = read(root/'native'/f'{key(c["sample_id"])}.json')
            assert native['sample_id'] == c['sample_id'] and native['ground_truth_in_model_input'] is False
            assert native['ground_truth_in_model_inputs'] is False and native['max_pixels'] == 262144
            assert native['image_size'] == c['image_size'] and native['instruction'] == c['instruction']
            assert native['generated_tokens'] <= 32
            complete = native['terminated_by_eos'] or native['generated_text'].endswith(('\n', '\r'))
            normalized = parse_offline_lax_click(native['generated_text']) if complete else None
            point = [round(v*(size-1)/1000) for v, size in zip(normalized, c['image_size'])] if normalized else None
            assert point == native['completed_lax_point']
            ui = read(root/'ui_tars'/f'{key(c["sample_id"])}.json')
            assert ui['sample_id'] == c['sample_id'] and ui['ground_truth_in_model_input'] is False
            ui_point = parse_ui_tars_point(ui['generated_text'], ui['processed_size'], ui['image_size']) if ui['terminated_by_eos'] else None
            assert ui_point == ui['point']
            count += 1
    return dict(native_completed_lax_replayed=count, official_ui_tars_audits_complete=True)


def audit_predictions(out, protocol, frozen, records_by_split):
    results = {}; lookup = {}
    expected_methods = {'v5_fixed', 'v6_s1_fixed', 'v6_s1_ownfine', 'v6_s2_fixed', 'v6_s2_ownfine',
        'phrase_local', 'qwen_native_completed_lax', 'ui_tars'}
    for name, entry in frozen['models'].items():
        modes = ['rank'] if entry['selector'] == 'region' else ['combined'] if entry['selector'] == 'quality' else ['rank', 'combined', 'quality']
        for mode in modes:
            expected_methods.update(name+'__'+mode+suffix for suffix in ('', '_ownfine', '_center'))
    for split in ('diagnostic', 'test'):
        folder = out/'evaluation'/split
        summary = read(folder/'summary.json'); rows = read_jsonl(folder/'predictions.jsonl')
        assert summary['primary'] == protocol['primary']+'__combined'
        assert summary['n'] == 192 and summary['split'] == split and summary['test_is_fresh'] == (split == 'test')
        assert summary['manifest_sha256'] == protocol['manifest_sha256'] and summary['frozen_sha256'] == sha256(out/'frozen_readouts.json')
        assert summary['all_missing_outputs_retained'] and summary['predictions_sha256'] == sha256(folder/'predictions.jsonl')
        by_id = {r['case']['sample_id']: r for r in records_by_split[split]}
        by = {}; counts = Counter(); hits = Counter(); points = Counter(); region_hits = Counter()
        region_fields = {name: Counter() for name in ('region_top3', 'region_top5', 'candidate_region_oracle', 'frozen_fine_point_oracle')}
        stages = defaultdict(Counter)
        region_cache = {sid: region_overlap(r['case'], r['metadata']['grid_shape'], r['parent_indices'])
            for sid, r in by_id.items() if r['metadata']['target_available']}
        for row in rows:
            sid = row['sample_id']; pair = (row['method'], sid)
            assert pair not in by and sid in by_id
            by[pair] = row; r = by_id[sid]; case = r['case']
            assert row['dataset'] == case['dataset'] and row['hit'] == bool(score_point(row['point'], case))
            counts[row['method']] += 1; hits[row['method']] += row['hit']; points[row['method']] += row['point'] is not None
            stages[row['method']][row.get('failure_stage')] += 1
            for field, counter in region_fields.items():
                counter[row['method']] += bool(row.get(field, False))
            if row['method'] in ('qwen_native_completed_lax', 'ui_tars'):
                source_root = out if split == 'test' else BASE
                folder_name = 'native' if row['method'] == 'qwen_native_completed_lax' else 'ui_tars'
                stored = read(source_root/folder_name/f'{key(sid)}.json')
                assert row['point'] == stored['completed_lax_point' if folder_name == 'native' else 'point']
            if 'selected_candidate_index' in row:
                ci = row['selected_candidate_index']; parent = int(r['parent_indices'][ci]); positive = region_cache[sid]
                assert row['selected_parent_index'] == parent
                assert row['point'] == r['frozen_points'][ci].tolist()
                assert row['selected_region_overlap'] == bool(positive[ci])
                assert row['candidate_region_oracle'] == bool(positive.any())
                assert row['frozen_fine_point_oracle'] == bool(r['label_fixed_point_hit'].any())
                if 'candidates' in row:
                    for candidate in row['candidates']:
                        cci = candidate['candidate_index']
                        assert candidate['parent_index'] == int(r['parent_indices'][cci])
                        assert candidate['point'] == r['frozen_points'][cci].tolist()
                    assert row['region_top3'] == any(positive[p['candidate_index']] for p in row['candidates'][:3])
                    assert row['region_top5'] == any(positive[p['candidate_index']] for p in row['candidates'][:5])
                else:
                    assert row['method'] in ('v5_fixed', 'v6_s1_fixed', 'v6_s2_fixed')
                    assert int(row['selected_region_overlap']) <= int(row['region_top3']) <= int(row['region_top5']) <= int(row['candidate_region_oracle'])
                stage = 'hit' if row['hit'] else 'fixed_fine_miss_after_intersecting_parent' if positive[ci] else 'ranking_miss_with_target_candidate' if positive.any() else 'no_target_peak_candidate'
                assert row['failure_stage'] == stage
                region_hits[row['method']] += bool(positive[ci])
            elif 'selected_parent_index' in row and row['point'] is not None:
                ci = np.flatnonzero(r['parent_indices'] == row['selected_parent_index'])
                assert len(ci) == 1 and row['selected_region_overlap'] == bool(region_cache[sid][ci[0]])
                region_hits[row['method']] += row['selected_region_overlap']
            if row['point'] is None and row['method'] not in ('qwen_native_completed_lax', 'ui_tars'):
                assert not r['metadata']['target_available']
        assert set(counts) == set(summary['methods']) == expected_methods
        assert summary['prediction_count'] == len(rows) == 192*len(expected_methods)
        for method in counts:
            metric = summary['methods'][method]
            assert counts[method] == metric['n'] == 192 and hits[method] == metric['hits'] and points[method] == metric['points']
            assert metric['accuracy'] == hits[method]/192
            if 'region_top1' in metric:
                assert metric['region_top1'] == region_hits[method]
                for field, summary_name in [('region_top3', 'region_top3'), ('region_top5', 'region_top5'),
                    ('candidate_region_oracle', 'candidate_region_coverage'), ('frozen_fine_point_oracle', 'frozen_fine_point_oracle')]:
                    assert metric[summary_name] == region_fields[field][method]
                # JSON object keys stringify None in older baseline summaries.
                expected_stages = {str(k) if k is not None else 'null': v for k, v in stages[method].items()}
                assert metric['failure_stages'] == expected_stages
            for dataset, sub in metric['datasets'].items():
                group = [r for r in rows if r['method'] == method and r['dataset'] == dataset]
                assert len(group) == sub['n'] == 64 and sum(r['hit'] for r in group) == sub['hits']
        # Pair the auxiliary outputs with their selected candidate metadata.
        for row in rows:
            if 'candidates' not in row:
                continue
            sid = row['sample_id']; r = by_id[sid]; parent = row['selected_parent_index']
            gh, gw = r['metadata']['grid_shape']; rr, cc = divmod(parent, gw); w, h = r['case']['image_size']
            center = [int(np.clip(np.rint((cc+.5)*w/gw), 0, w-1)), int(np.clip(np.rint((rr+.5)*h/gh), 0, h-1))]
            assert by[row['method']+'_center', sid]['point'] == center
            assert by[row['method']+'_ownfine', sid]['point'] == row['candidates'][0]['own_fine_point']
        results[split] = dict(methods=len(counts), scored_points=len(rows), n=192,
            prediction_sha256=summary['predictions_sha256'])
        lookup[split] = by
        print('ALL SCORES REPLAYED '+split+' '+str(len(rows)), flush=True)
    return results, lookup


@torch.inference_mode()
def replay_primary_cpu(out, frozen, records, by):
    named = [r for r in records if r['metadata']['target_available']]
    result = {}
    for seed in ('s1', 's2'):
        name = f'semantic_hard_{seed}__quality'; entry = frozen['models'][name]
        model, metadata = load_readout(out/'models'/entry['file'], 'cpu')
        assert metadata['selector'] == 'quality'
        method = name+'__combined'; hits = 0; regions = 0
        for start in range(0, len(named), 8):
            group = named[start:start+8]; inputs, labels = make_batch(group, 'cpu', labels=False)
            assert labels is None and set(inputs) == EXPECTED_INPUT_KEYS
            output = model(**inputs); scored = score_output(output, 'combined')
            for i, r in enumerate(group):
                pred = decode_ranker(scored, r, i, top_k=1)[0]
                sid = r['case']['sample_id']; actual = by[method, sid]; ci = pred['candidate_index']
                assert ci == actual['selected_candidate_index'] and pred['parent_index'] == actual['selected_parent_index']
                fixed = r['frozen_points'][ci].tolist()
                assert fixed == actual['point']
                hits += bool(score_point(fixed, r['case'])); regions += actual['selected_region_overlap']
        expected = [r for (m, _), r in by.items() if m == method]
        assert len(expected) == 192 and hits == sum(r['hit'] for r in expected)
        result[name] = dict(named=len(named), n=192, hits=hits, region_hits=regions,
            selected_parent_and_fixed_point_exact=True, checkpoint_sha256=entry['sha256'])
        del model
        print('CPU REPLAY '+name+' '+json.dumps(result[name]), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('runs/semantic_head_v7'))
    parser.add_argument('--numeric-case', help='Only fixed-five replay this original batch to numeric_replay_five.json')
    parser.add_argument('--numeric-split', choices=('diagnostic', 'test'), default='test')
    args = parser.parse_args(); out = args.output; torch.set_num_threads(4)
    if args.numeric_case:
        replay_numeric_case(out, args.numeric_case, args.numeric_split)
        return
    protocol = read(out/'protocol.json'); frozen = read(out/'frozen_readouts.json')
    cases = read_jsonl(out/'manifest.jsonl')
    print('AUDIT STAGE frozen checkpoints/sources', flush=True)
    epochs = audit_frozen(out, protocol, frozen)
    print('AUDIT STAGE selection/images', flush=True)
    selection = audit_selection(out, protocol, cases)
    records = {}
    for split in ('train', 'val', 'diagnostic', 'test'):
        print('AUDIT STAGE load records '+split, flush=True)
        records[split] = load_records(out, split)
    print('AUDIT STAGE raw records/labels', flush=True)
    arrays = audit_records(out, protocol, records)
    print('AUDIT STAGE fixed points CPU/canonical GPU exceptions', flush=True)
    points = audit_fixed_points(out, protocol, records)
    print('AUDIT STAGE external baselines', flush=True)
    baselines = audit_native_and_ui(out, records)
    print('AUDIT STAGE all predictions', flush=True)
    scores, lookup = audit_predictions(out, protocol, frozen, records)
    enrich_numeric_exceptions(points['cpu_numeric_exceptions'], lookup, frozen)
    print('AUDIT STAGE fresh primary CPU replay', flush=True)
    replay = replay_primary_cpu(out, frozen, records['test'], lookup['test'])
    test_log_path = out/'tests_integration_final.log'
    test_log = text_file(test_log_path)
    assert re.search(r'\b(?:errors?|failed)\b', test_log, re.IGNORECASE) is None
    passed = re.findall(r'(\d+) passed', test_log)
    assert passed and int(passed[-1]) >= 39
    # Recheck immutable artifacts after all CPU replay and image decoding.
    check_hashes(protocol['unchanged_artifact_sha256']); check_hashes(protocol['baseline_checkpoint_sha256']); check_hashes(frozen['sources'])
    result = dict(complete=True, primary=protocol['primary']+'__combined', selection=selection,
        validation_selected_epochs=epochs, checkpoints_verified=30, trajectories_verified=10,
        training_query_budget_per_epoch=600, optimizer_steps_per_epoch=75, epochs_per_trajectory=24,
        fresh_features_after_freeze=True, backbone_trainable_parameters=0,
        arrays=arrays, fixed_points=points, external_baselines=baselines, scores=scores,
        cpu_replay=replay, fresh_primary_cpu_gpu_exact=True, tests_passed=int(passed[-1]),
        tests_log=str(test_log_path), tests_log_sha256=sha256(test_log_path),
        previous_frozen_artifacts_unchanged=True, frozen_sha256=sha256(out/'frozen_readouts.json'),
        manifest_sha256=protocol['manifest_sha256'], protocol_sha256=sha256(out/'protocol.json'),
        auditor_sha256=sha256(Path(__file__)))
    write_json(out/'final_audit.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
