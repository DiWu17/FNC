"""Prespecification and model/source freeze for the V11 verifier replacement."""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np

from clickagent.action_benchmark import sha256, write_json
from point_aligned_data import read
from run_target_token_experiment import read_jsonl

OUT = Path('runs/candidate_scorer_v11')
SEEDS = (20261131, 20261132)
VARIANTS = ('internal_full', 'internal_pooled', 'internal_global', 'visual')
SOURCES = [
    'src/clickagent/internal_candidate_scorer.py',
    'src/clickagent/visual_candidate_verifier.py',
    'scripts/extract_candidate_scorer.py',
    'scripts/cache_candidate_scorer.py',
    'scripts/train_candidate_scorer.py',
    'scripts/run_candidate_scorer.py',
    'scripts/candidate_scorer_protocol.py',
    'scripts/visual_candidate_cache.py',
    'scripts/precache_candidate_visual.py',
    'scripts/precache_candidate_inputs.py',
    'scripts/prepare_candidate_scorer.py',
    'src/clickagent/candidate_budget.py',
    'src/clickagent/public_evaluation.py',
    'src/clickagent/point_quality_readout.py',
    'src/clickagent/semantic_head_experiment.py',
    'scripts/run_target_token_experiment.py',
]
EXPECTED_COUNTS = dict(train=873, val=172, test=192)
OLD_IDENTITY = Path('runs/context_verifier_v9/inference_identity.json')
QWEN_IDENTITY = Path('data/conditioned_v3/qwen_weight_identity.json')
RUNTIME_CONFIG = Path('runs/candidate_v2/identity.json')
REFERENCE_IDENTITIES = {
    'gui_verifier_identity_sha256': Path('models/GUI-Actor-Verifier-2B/clickagent_download_verified.json'),
    'ui_tars_identity_sha256': Path('models/UI-TARS-1.5-7B/clickagent_download_verified.json'),
}


def init():
    selection = read(OUT / 'data_selection.json')
    assert selection['complete']
    protocol = dict(
        version=11, name='frozen natural-target internal candidate scoring and dedicated visual verification',
        created_at=time.time(), primary='internal_full_s1, K16, fresh independent192',
        selection='epoch selected on original172 validation K16 point hits, tie lower validation loss; seed1 prespecified',
        primary_reference='same-pool GUI-Actor-Verifier-2B K16; no GUI-Actor parameters/features/teacher in new heads',
        counts=dict(Counter(c['experiment_split'] for c in read_jsonl(OUT / 'manifest.jsonl'))),
        diagnostic='old v9/v10 192 is reused, secondary only; never train or choose epochs from diagnostic',
        frozen_backbones=['Qwen3-VL-8B', 'v8 joint_connected_s1 spatial decoder', 'GUI-Actor reference'],
        candidates='unchanged v10 quotas: globalTop(K/2)+old independent cropTop(K/2); exact point dedup after quota, no refill',
        budgets=[4, 8, 16, 32], primary_budget=16,
        training='873 original train queries, real generated K32 candidate pools only; no GT coordinate injection',
        training_pool_sampling='seeded random budget from4/8/16/32; real prefix pool, with shuffled candidate order',
        supervision='actual point-in-region labels; multi-positive listwise with explicit none candidate and balanced point BCE',
        optimization=dict(epochs=16, learning_rate=0.0002, weight_decay=0.02, accumulation=8,
                          gradient_clip=1., seeds=list(SEEDS)),
        variants=list(VARIANTS), internal='raw target token states, complete per-head field, point-aligned full/local content; late interaction and candidate-set comparison',
        pooled_control='same internal architecture/parameters, token evidence homogenized before candidate interaction',
        global_control='same internal architecture/parameters, local view masked; global evidence unchanged',
        visual='separate frozen Qwen3 visual encoder, shared high-resolution full screenshot max1048576px; point-conditioned multi-scale ROI and target token state readout',
        new_head_training='both seeds for all four variants; no selection across variants using test',
        inference='primary always best nonempty candidate; optional none output reported separately, not an action-permission gate',
        set_scoring='score each budget with its own candidate set; set-dependent scores are never reused by slicing K32',
        references=['GUI-Actor K4/K16', 'old v9 point critic same K16', 'Qwen native completed lax', 'UI-TARS1.5-7B'],
        metric='any integer point inside a valid bbox/polygon is correct; all192 denominator; empty name/pool fails',
        uncertainty='paired stratified screenshot bootstrap intervals; subset counts; two seeds; no claim of equal total compute',
        timing='separate cached-head scoring, extra visual encoding and GUI-Actor verification; cached feature extraction cost disclosed',
        independent_test_extraction='only after all selected checkpoints, model inputs, sources and validation selections frozen',
        manifest_sha256=sha256(OUT / 'manifest.jsonl'), data_selection_sha256=sha256(OUT / 'data_selection.json'),
        diagnostic_manifest_sha256=sha256(OUT / 'diagnostic_manifest.jsonl'))
    path = OUT / 'protocol.json'
    if path.exists():
        old = read(path); protocol['created_at'] = old['created_at']; assert protocol == old
    else:
        write_json(path, protocol)
    print('V11 PROTOCOL SEALED', flush=True)


def _object_sha(value):
    return hashlib.sha256(json.dumps(value, separators=(',', ':')).encode()).hexdigest()


def _checked_file(path, expected, inventory):
    """Hash shared large inputs once during freeze, never on every test query."""
    name = os.path.normcase(os.path.abspath(path))
    if name not in inventory:
        inventory[name] = sha256(Path(path))
    actual = inventory[name]
    assert actual == expected, ('SHA256 mismatch', str(path), expected, actual)
    return actual


def assert_no_test_artifacts(manifest):
    roots = [OUT / 'test', OUT / 'predictions/test', OUT / 'cache/internal/test',
             OUT / 'cache/visual/test']
    artifacts = [str(p) for root in roots if root.exists() for p in root.rglob('*') if p.is_file()]
    artifacts += [str(p) for p in (OUT / 'cache').glob('test*') if p.is_file()]
    # Shared visual encodings live outside the split directories.
    artifacts += [str(p) for c in manifest if c['experiment_split'] == 'test'
                  for p in [OUT / 'vision_encodings' / (c['image_sha256'] + '.npz')] if p.exists()]
    assert not artifacts, ('Independent test artifacts exist before model freeze', artifacts[:20])


def _cache_common(archive, case, pipeline):
    from clickagent.public_evaluation import point_in_geometry
    assert str(archive['sample_id']) == case['sample_id']
    assert not bool(archive['annotation_derived_coordinates'])
    assert str(archive['pipeline_sha256']) == sha256(Path(pipeline['_path']))
    points = archive['points']
    assert points.ndim == 2 and points.shape[1] == 2 and np.issubdtype(points.dtype, np.integer)
    assert ((points >= 0) & (points < np.asarray(case['image_size']))).all()
    assert len({tuple(p) for p in points.tolist()}) == len(points)
    assert bool(archive['available']) == bool(len(points))
    assert points.tolist() == [r['point'] for r in pipeline['pools']['32']]
    labels = archive['labels']
    expected_labels = [point_in_geometry((p / np.asarray(case['image_size'])).tolist(), case['target_geometry']) for p in points]
    assert labels.shape == (len(points),) and np.array_equal(labels, np.asarray(expected_labels, np.float32))
    named16 = oracle16 = 0
    for budget in (4, 8, 16, 32):
        indices = archive['pool_indices_' + str(budget)]
        assert indices.ndim == 1 and np.issubdtype(indices.dtype, np.integer)
        assert len(np.unique(indices)) == len(indices) and ((indices >= 0) & (indices < len(points))).all()
        assert points[indices].tolist() == [r['point'] for r in pipeline['pools'][str(budget)]]
        if budget == 16:
            named16 = int(bool(len(indices))); oracle16 = int(bool(labels[indices].any()))
    return named16, oracle16


def validate_training_inputs(manifest, files):
    """Recheck actual caches, labels and upstream artifacts, including empty rows."""
    from clickagent.candidate_budget import build_pools
    inventory, identities, counts = {}, {False: {}, True: {}}, {}
    for split in ('train', 'val'):
        index_path = OUT / 'cache' / (split + '_index.json')
        index = read(index_path)
        assert index['complete'] and index['visual_complete'] and index['has_labels']
        assert index['split'] == split and index['real_candidate_pools']
        assert index['annotation_derived_coordinates'] is False
        expected = [c for c in manifest if c['experiment_split'] == split]
        assert [r['sample_id'] for r in index['cases']] == [c['sample_id'] for c in expected]
        counts[split] = dict(n=len(expected), named16=0, oracle16=0)
        files[str(index_path)] = sha256(index_path)
        for visual in (False, True):
            rows = [(r['sample_id'], r['available'], r['visual_sha256' if visual else 'sha256']) for r in index['cases']]
            identities[visual][split] = _object_sha(rows)
        for row, case in zip(index['cases'], expected):
            _checked_file(case['image'], case['image_sha256'], inventory)
            pipeline_path = Path(row['pipeline_path'])
            _checked_file(pipeline_path, row['pipeline_sha256'], inventory)
            pipeline = read(pipeline_path)
            assert pipeline['sample_id'] == case['sample_id'] and pipeline['ground_truth_in_model_inputs'] is False
            assert pipeline['pools'] == build_pools(pipeline), ('Pool policy changed', case['sample_id'])
            raw_files = {pipeline['global_feature_path']: pipeline['global_feature_sha256']}
            if pipeline['local_feature_path']:
                raw_files[pipeline['local_feature_path']] = pipeline['local_feature_sha256']
            assert raw_files == row['raw_files']
            for raw_path, digest in raw_files.items():
                _checked_file(raw_path, digest, inventory)
            pipeline['_path'] = str(pipeline_path)
            common = None
            for visual in (False, True):
                path = Path(row['visual_path'] if visual else row['path'])
                _checked_file(path, row['visual_sha256' if visual else 'sha256'], inventory)
                with np.load(path, allow_pickle=False) as archive:
                    named, oracle = _cache_common(archive, case, pipeline)
                    assert bool(archive['available']) == row['available'] and len(archive['points']) == row['n_points']
                    keys = ['sample_id', 'available', 'points', 'labels', 'annotation_derived_coordinates', 'pipeline_sha256']
                    keys += ['pool_indices_' + str(k) for k in (4, 8, 16, 32)]
                    actual_common = {k: archive[k].copy() for k in keys}
                    if common is None:
                        common = actual_common
                        counts[split]['named16'] += named; counts[split]['oracle16'] += oracle
                    else:
                        assert all(np.array_equal(common[k], actual_common[k]) for k in keys)
                    feature_keys = [k for k in archive.files if k.startswith('v__' if visual else 'i__')]
                    assert not any('label' in k or 'target_geometry' in k for k in feature_keys)
                    if row['available']:
                        assert feature_keys
                        if visual:
                            encoding_path = OUT / 'vision_encodings' / (case['image_sha256'] + '.npz')
                            _checked_file(encoding_path, str(archive['visual_encoding_sha256']), inventory)
                            with np.load(encoding_path, allow_pickle=False) as encoding:
                                meta = json.loads(str(encoding['metadata']))
                            assert meta['image_sha256'] == case['image_sha256'] and meta['image_size'] == case['image_size']
                            assert meta['all_visual_parameters_frozen'] and not meta['language_model_loaded'] and not meta['gui_actor_used']
                            assert meta['max_pixels'] == 1048576
        print(f'FREEZE TRAINING INPUTS {split}: {len(expected)} verified', flush=True)
    return dict(complete=True, checked_once_at_freeze=True, identities={'internal': identities[False], 'visual': identities[True]},
                counts=counts, files=inventory, inventory_sha256=_object_sha(inventory))


def validate_external_inputs(files):
    """Verify old immutable code/models and the actual local Qwen weight bytes."""
    old = read(OLD_IDENTITY); inventory = {}
    files[str(OLD_IDENTITY)] = sha256(OLD_IDENTITY)
    for group in ('sources', 'models'):
        for path, digest in old[group].items():
            _checked_file(path, digest, inventory); files[path] = digest
    _checked_file(QWEN_IDENTITY, old['qwen_identity_sha256'], inventory)
    files[str(QWEN_IDENTITY)] = old['qwen_identity_sha256']
    for name, path in REFERENCE_IDENTITIES.items():
        _checked_file(path, old[name], inventory); files[str(path)] = old[name]
    identity = read(QWEN_IDENTITY)
    model_root = Path(identity['model_path']).resolve()
    assert identity['files']
    for relative, entry in identity['files'].items():
        path = (model_root / relative).resolve()
        assert path.is_relative_to(model_root)
        assert path.stat().st_size == entry['bytes']
        _checked_file(path, entry['sha256'], inventory)
        # Small model configs/tokenizers remain in the cheap test-time seal.
        if not path.name.endswith('.safetensors'):
            files[str(path)] = entry['sha256']
    config = read(RUNTIME_CONFIG)['config']
    assert Path(config['model_path']).resolve() == model_root
    assert config['dtype'] == 'float16' and config['layers'] == [15, 28, 36]
    assert config['local_files_only'] is True and config['max_pixels'] == 262144 and config['no_image'] is False
    files[str(RUNTIME_CONFIG)] = sha256(RUNTIME_CONFIG)
    return dict(complete=True, checked_once_at_freeze=True, old_identity_sha256=sha256(OLD_IDENTITY),
                qwen_identity_sha256=sha256(QWEN_IDENTITY), runtime_config_sha256=sha256(RUNTIME_CONFIG),
                files=inventory, inventory_sha256=_object_sha(inventory))


def validate_model(path, variant, seed, protocol, sources, inputs, counts):
    import torch
    meta = read(path.with_suffix('.json'))
    assert meta['complete'] and meta['checkpoint_sha256'] == sha256(path)
    assert meta['variant'] == variant and meta['seed'] == seed
    assert meta['protocol_sha256'] == sha256(OUT / 'protocol.json')
    assert meta['source_identity'] == sources and meta['input_identity'] == inputs
    history = meta['history']
    assert [row['epoch'] for row in history] == list(range(protocol['optimization']['epochs'] + 1))
    best, selected = None, None
    for row in history:
        v = row['validation']
        assert v['n'] == counts['val']['n'] and v['named'] == counts['val']['named16'] and v['oracle'] == counts['val']['oracle16']
        assert isinstance(v['hits'], int) and 0 <= v['hits'] <= v['oracle']
        assert isinstance(v['refused'], int) and 0 <= v['refused'] <= v['named']
        assert math.isfinite(v['loss']) and math.isfinite(row['elapsed_seconds'])
        assert row['train_loss'] is None if row['epoch'] == 0 else isinstance(row['train_loss'], (int, float)) and math.isfinite(row['train_loss'])
        selector = (v['hits'], -v['loss'])
        if best is None or selector > best:
            best, selected = selector, row
    assert meta['selected'] == selected, ('Wrong validation selector', str(path))
    # CPU metadata inspection does not instantiate a model or execute a forward.
    payload = torch.load(path, map_location='cpu', weights_only=True)
    checkpoint = payload['metadata']
    for name in ('variant', 'seed', 'protocol_sha256', 'source_identity', 'input_identity', 'selected'):
        assert checkpoint[name] == meta[name], ('Checkpoint/sidecar mismatch', str(path), name)
    assert checkpoint['train_n'] == counts['train']['n'] and checkpoint['validation_n'] == counts['val']['n']
    assert checkpoint['primary_budget'] == 16 and checkpoint['pretrained_backbones_frozen'] is True
    assert checkpoint['training_uses_gui_actor'] is False and checkpoint['real_candidate_pools'] is True
    assert checkpoint['annotation_derived_points'] is False
    assert payload['state_dict'] and all(np.isfinite(t.detach().cpu().numpy()).all() for t in payload['state_dict'].values())
    return meta


def freeze():
    manifest = read_jsonl(OUT / 'manifest.jsonl'); protocol = read(OUT / 'protocol.json')
    assert dict(Counter(c['experiment_split'] for c in manifest)) == EXPECTED_COUNTS == protocol['counts']
    assert len({c['sample_id'] for c in manifest}) == len(manifest)
    assert protocol['optimization']['epochs'] == 16 and protocol['optimization']['seeds'] == list(SEEDS)
    assert protocol['variants'] == list(VARIANTS) and protocol['primary_budget'] == 16
    selection = read(OUT / 'data_selection.json')
    assert selection['complete'] and selection['retained_train_val_exact_unchanged'] is True
    assert selection['manifest_sha256'] == sha256(OUT / 'manifest.jsonl') == protocol['manifest_sha256']
    assert selection['diagnostic_manifest_sha256'] == sha256(OUT / 'diagnostic_manifest.jsonl') == protocol['diagnostic_manifest_sha256']
    assert protocol['data_selection_sha256'] == sha256(OUT / 'data_selection.json')
    for path, digest in selection['source_sha256'].items():
        assert sha256(Path(path)) == digest, path
    assert_no_test_artifacts(manifest)
    files = {str(OUT / p): sha256(OUT / p) for p in ['protocol.json', 'manifest.jsonl', 'data_selection.json', 'diagnostic_manifest.jsonl']}
    sources = {p: sha256(Path(p)) for p in SOURCES}; files.update(sources)
    external = validate_external_inputs(files)
    training = validate_training_inputs(manifest, files)
    models = {}
    for variant in VARIANTS:
        for suffix, seed in zip(('s1', 's2'), SEEDS):
            name = variant + '_' + suffix; path = OUT / 'models' / (name + '.pt')
            inputs = training['identities']['visual' if variant == 'visual' else 'internal']
            meta = validate_model(path, variant, seed, protocol, sources, inputs, training['counts'])
            models[name] = dict(file=str(path), sha256=sha256(path), selected=meta['selected'])
            files[str(path)] = sha256(path); files[str(path.with_suffix('.json'))] = sha256(path.with_suffix('.json'))
    # Detect source/model/index changes during the one-time input scan.
    for path, digest in files.items():
        assert sha256(Path(path)) == digest, path
    assert_no_test_artifacts(manifest)
    result = dict(complete=True, frozen_at=time.time(), fresh_test_features_at_freeze=0,
                  models=models, files=files, primary='internal_full_s1', primary_budget=16,
                  training_input_evidence=training, external_input_evidence=external)
    path = OUT / 'frozen.json'
    if path.exists():
        old = read(path); result['frozen_at'] = old['frozen_at']; assert result == old
    else:
        write_json(path, result)
    print('V11 MODELS AND INPUT PIPELINE FROZEN', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--stage', choices=['init', 'freeze'], required=True)
    args = p.parse_args(); init() if args.stage == 'init' else freeze()
