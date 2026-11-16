"""Strict freeze fixtures use tiny CPU files; no real models/data or CUDA."""
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import candidate_scorer_protocol as protocol
from clickagent.candidate_budget import build_pools


def dump(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def npz(path, **arrays):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    out = tmp_path / 'out'; out.mkdir()
    monkeypatch.setattr(protocol, 'OUT', out)
    monkeypatch.setattr(protocol, 'VARIANTS', ('internal_full', 'visual'))
    monkeypatch.setattr(protocol, 'EXPECTED_COUNTS', dict(train=2, val=1, test=1))
    source = tmp_path / 'new_source.py'; source.write_text('frozen source', encoding='utf-8')
    monkeypatch.setattr(protocol, 'SOURCES', [str(source)])
    old_source = tmp_path / 'old_source.py'; old_source.write_text('old source', encoding='utf-8')
    old_model = tmp_path / 'old.pt'; old_model.write_bytes(b'old checkpoint')
    model_root = tmp_path / 'qwen'; model_root.mkdir()
    (model_root / 'config.json').write_text('{}', encoding='utf-8')
    (model_root / 'weights.safetensors').write_bytes(b'tiny fake model bytes')
    qwen = tmp_path / 'qwen_identity.json'
    dump(qwen, dict(model_path=str(model_root), files={p.name: dict(sha256=protocol.sha256(p), bytes=p.stat().st_size)
                                                     for p in model_root.iterdir()}))
    old_identity = tmp_path / 'old_identity.json'
    dump(old_identity, dict(sources={str(old_source): protocol.sha256(old_source)},
                            models={str(old_model): protocol.sha256(old_model)}, qwen_identity_sha256=protocol.sha256(qwen)))
    runtime = tmp_path / 'runtime.json'
    dump(runtime, dict(config=dict(model_path=str(model_root), dtype='float16', layers=[15, 28, 36],
                                  local_files_only=True, max_pixels=262144, no_image=False)))
    monkeypatch.setattr(protocol, 'OLD_IDENTITY', old_identity)
    monkeypatch.setattr(protocol, 'QWEN_IDENTITY', qwen)
    monkeypatch.setattr(protocol, 'RUNTIME_CONFIG', runtime)
    monkeypatch.setattr(protocol, 'REFERENCE_IDENTITIES', {})
    manifest = []
    for number, split in enumerate(('train', 'train', 'val', 'test')):
        image = tmp_path / f'image{number}.png'; image.write_bytes(f'image{number}'.encode())
        manifest.append(dict(sample_id=f'case{number}', experiment_split=split, image=str(image),
                             image_sha256=protocol.sha256(image), image_size=[100, 100],
                             target_geometry=dict(type='bbox', boxes=[[0., 0., .3, .3]])))
    manifest_path = out / 'manifest.jsonl'
    manifest_path.write_text(''.join(json.dumps(c) + '\n' for c in manifest), encoding='utf-8')
    (out / 'diagnostic_manifest.jsonl').write_text('', encoding='utf-8')
    selection = dict(complete=True, retained_train_val_exact_unchanged=True, source_sha256={},
                     manifest_sha256=protocol.sha256(manifest_path),
                     diagnostic_manifest_sha256=protocol.sha256(out / 'diagnostic_manifest.jsonl'))
    dump(out / 'data_selection.json', selection)
    spec = dict(counts=dict(train=2, val=1, test=1), optimization=dict(epochs=16, seeds=list(protocol.SEEDS)),
                variants=list(protocol.VARIANTS), primary_budget=16,
                manifest_sha256=selection['manifest_sha256'], diagnostic_manifest_sha256=selection['diagnostic_manifest_sha256'],
                data_selection_sha256=protocol.sha256(out / 'data_selection.json'))
    dump(out / 'protocol.json', spec)
    for split in ('train', 'val'):
        rows = []
        for case in (c for c in manifest if c['experiment_split'] == split):
            available = case['sample_id'] != 'case1'
            sid = case['sample_id']; raw_path = out / 'raw' / (sid + '.npz')
            npz(raw_path, metadata=np.asarray('{}'), target_hidden=np.zeros((int(available), 2), np.float16))
            gp = dict(all_points=[[20, 10], [80, 40]], all_scores=[2., 1.], parent_indices=[0, 1]) if available else {}
            pipeline = dict(sample_id=sid, global_result=gp, old_crop_result=None, old_crop_transform=None,
                            global_feature_path=str(raw_path), global_feature_sha256=protocol.sha256(raw_path),
                            local_feature_path=None, local_feature_sha256=None, ground_truth_in_model_inputs=False)
            pipeline['pools'] = build_pools(pipeline)
            pipeline_path = out / 'pipeline' / (sid + '.json'); dump(pipeline_path, pipeline)
            points = np.asarray([r['point'] for r in pipeline['pools']['32']], np.int64).reshape(-1, 2)
            arrays = dict(sample_id=np.asarray(sid), available=np.asarray(available), points=points,
                          labels=np.asarray([1., 0.] if available else [], np.float32),
                          annotation_derived_coordinates=np.asarray(False), pipeline_sha256=np.asarray(protocol.sha256(pipeline_path)))
            arrays.update({'pool_indices_' + str(k): np.arange(len(points), dtype=np.int64) for k in (4, 8, 16, 32)})
            internal_path = out / 'cache/internal' / split / (sid + '.npz')
            npz(internal_path, **arrays, **({'i__signal': np.zeros((len(points), 2), np.float16)} if available else {}))
            visual_path = out / 'cache/visual' / split / (sid + '.npz')
            if available:
                encoding_path = out / 'vision_encodings' / (case['image_sha256'] + '.npz')
                npz(encoding_path, metadata=np.asarray(json.dumps(dict(image_sha256=case['image_sha256'], image_size=[100, 100],
                        all_visual_parameters_frozen=True, language_model_loaded=False, gui_actor_used=False, max_pixels=1048576))))
                arrays['visual_encoding_sha256'] = np.asarray(protocol.sha256(encoding_path))
            npz(visual_path, **arrays, **({'v__signal': np.zeros((len(points), 2), np.float16)} if available else {}))
            rows.append(dict(sample_id=sid, available=available, n_points=len(points), path=str(internal_path),
                             visual_path=str(visual_path), sha256=protocol.sha256(internal_path), visual_sha256=protocol.sha256(visual_path),
                             pipeline_path=str(pipeline_path), pipeline_sha256=protocol.sha256(pipeline_path),
                             raw_files={str(raw_path): protocol.sha256(raw_path)}))
        dump(out / 'cache' / (split + '_index.json'), dict(complete=True, visual_complete=True, has_labels=True,
             split=split, real_candidate_pools=True, annotation_derived_coordinates=False, cases=rows))
    for variant in protocol.VARIANTS:
        visual = variant == 'visual'; input_identity = {}
        for split in ('train', 'val'):
            rows = protocol.read(out / 'cache' / (split + '_index.json'))['cases']
            input_identity[split] = protocol._object_sha([(r['sample_id'], r['available'], r['visual_sha256' if visual else 'sha256']) for r in rows])
        for suffix, seed in zip(('s1', 's2'), protocol.SEEDS):
            history = []
            for epoch in range(17):
                # Epoch3 ties epoch2 exactly; epoch2 must win, despite later lower-loss zero-hit epochs.
                history.append(dict(epoch=epoch, validation=dict(n=1, named=1, hits=int(epoch in (1, 2, 3)), oracle=1,
                    refused=0, loss=.3 if epoch in (2, 3) else (.5 if epoch == 1 else .1)),
                    train_loss=None if epoch == 0 else .7, elapsed_seconds=float(epoch)))
            selected = history[2]
            metadata = dict(variant=variant, seed=seed, selected=selected, train_n=2, validation_n=1,
                            primary_budget=16, pretrained_backbones_frozen=True, training_uses_gui_actor=False,
                            real_candidate_pools=True, annotation_derived_points=False, input_identity=input_identity,
                            source_identity={str(source): protocol.sha256(source)}, protocol_sha256=protocol.sha256(out / 'protocol.json'))
            path = out / 'models' / (variant + '_' + suffix + '.pt'); path.parent.mkdir(exist_ok=True)
            torch.save(dict(metadata=metadata, state_dict={'weight': torch.ones(2)}, config={}), path)
            dump(path.with_suffix('.json'), dict(complete=True, selected=selected, history=history,
                 checkpoint_sha256=protocol.sha256(path), variant=variant, seed=seed,
                 input_identity=input_identity, source_identity=metadata['source_identity'], protocol_sha256=metadata['protocol_sha256']))
    return dict(out=out, manifest=manifest, source=source, model=out / 'models/internal_full_s1.pt', qwen_root=model_root)


def test_freeze_checks_real_inputs_and_keeps_large_files_out_of_per_query_seal(fixture):
    protocol.freeze()
    frozen = protocol.read(fixture['out'] / 'frozen.json')
    assert frozen['complete'] and len(frozen['models']) == 4
    assert frozen['models']['internal_full_s1']['selected']['epoch'] == 2
    assert frozen['training_input_evidence']['counts']['train'] == dict(n=2, named16=1, oracle16=1)
    assert frozen['training_input_evidence']['counts']['val'] == dict(n=1, named16=1, oracle16=1)
    assert frozen['training_input_evidence']['checked_once_at_freeze']
    assert str(fixture['qwen_root'] / 'weights.safetensors') not in frozen['files']
    assert all('/cache/internal/' not in p.replace('\\', '/') for p in frozen['files'])
    protocol.freeze()  # deterministic and idempotent before any test work


@pytest.mark.parametrize('kind', ['short_history', 'wrong_selector', 'variant', 'seed', 'protocol', 'source', 'input'])
def test_model_identity_and_selector_reject_corruption(fixture, kind):
    path = fixture['model'].with_suffix('.json'); meta = protocol.read(path)
    if kind == 'short_history': meta['history'].pop()
    elif kind == 'wrong_selector': meta['selected'] = meta['history'][3]
    elif kind in ('variant', 'seed'): meta[kind] = 'wrong'
    elif kind == 'protocol': meta['protocol_sha256'] = 'wrong'
    elif kind == 'source': meta['source_identity'] = {}
    elif kind == 'input': meta['input_identity'] = {}
    dump(path, meta)
    with pytest.raises(AssertionError): protocol.freeze()
    assert not (fixture['out'] / 'frozen.json').exists()


def test_checkpoint_metadata_must_match_sidecar(fixture):
    path = fixture['model']; payload = torch.load(path, map_location='cpu', weights_only=True)
    payload['metadata']['selected'] = dict(payload['metadata']['selected'], epoch=3)
    torch.save(payload, path)
    meta = protocol.read(path.with_suffix('.json')); meta['checkpoint_sha256'] = protocol.sha256(path)
    dump(path.with_suffix('.json'), meta)
    with pytest.raises(AssertionError, match='Checkpoint/sidecar mismatch'): protocol.freeze()


@pytest.mark.parametrize('target', ['raw', 'internal', 'visual', 'pipeline', 'qwen'])
def test_actual_bytes_checked_not_only_index_json(fixture, target):
    row = protocol.read(fixture['out'] / 'cache/train_index.json')['cases'][0]
    path = {'raw': next(iter(row['raw_files'])), 'internal': row['path'], 'visual': row['visual_path'],
            'pipeline': row['pipeline_path'], 'qwen': fixture['qwen_root'] / 'weights.safetensors'}[target]
    with Path(path).open('ab') as stream: stream.write(b'changed')
    with pytest.raises(AssertionError): protocol.freeze()


@pytest.mark.parametrize('relative', ['test/native/one.json', 'test/global/one.json', 'test/old_crop_features/one.npz',
                                    'cache/internal/test/one.npz', 'cache/visual/test/one.npz',
                                    'predictions/test/one.json', 'cache/test_index.json'])
def test_all_preexisting_test_artifacts_block_freeze(fixture, relative):
    path = fixture['out'] / relative; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b'x')
    with pytest.raises(AssertionError, match='test artifacts'): protocol.freeze()


def test_shared_test_visual_encoding_blocks_freeze(fixture):
    case = fixture['manifest'][-1]
    path = fixture['out'] / 'vision_encodings' / (case['image_sha256'] + '.npz'); path.write_bytes(b'x')
    with pytest.raises(AssertionError, match='test artifacts'): protocol.freeze()


def test_wrong_labels_rejected_even_after_updating_cache_sha(fixture):
    index_path = fixture['out'] / 'cache/train_index.json'; index = protocol.read(index_path)
    row = index['cases'][0]; path = Path(row['path'])
    with np.load(path, allow_pickle=False) as z: arrays = {k: z[k].copy() for k in z.files}
    arrays['labels'][:] = 0.; npz(path, **arrays); row['sha256'] = protocol.sha256(path); dump(index_path, index)
    with pytest.raises(AssertionError): protocol.freeze()
