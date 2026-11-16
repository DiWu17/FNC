"""V12 label-free whole-chain proposal extraction and unchanged V11 inputs."""
from __future__ import annotations
import argparse
import hashlib
import json
from functools import lru_cache
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from clickagent.action_benchmark import sha256, write_json
from clickagent.candidate_budget import build_pools
from clickagent.context_verifier import ContextFeatureRuntime
from clickagent.crossfit_proposer import predict_raw, load_crossfit_proposer
from clickagent.internal_candidate_scorer import build_internal_inputs
from clickagent.model_features import ExtractionConfig
from clickagent.point_aligned_inference import proposal_crop
from cache_candidate_scorer import raw_arrays, base_cache, save_npz
from run_point_aligned_inference import save_features, native
from crossfit_protocol import OUT, cases, key, read, raw_path, assert_sealed


def runtime_config():
    cfg = read('runs/candidate_v2/identity.json')['config']
    cfg.update(max_new_tokens=64, seed=20261131, conditions=('prefill',),
               offload_folder=str((OUT / 'offload').absolute()))
    cfg.update(layers=tuple(cfg['layers']),
               max_memory={int(k) if str(k).isdigit() else k: v for k, v in cfg['max_memory'].items()})
    return ExtractionConfig(**cfg)


def role_for(case, group):
    if group in ('val', 'test'):
        return 'full'
    fold = int(case['image_fold'])
    return 'fold_' + str(fold if group == 'oof' else (fold + 1) % 5)


def folder(group):
    return OUT / ('test' if group == 'test' else 'trainval') / group


def checked_proposers():
    result = {}
    for role in [f'fold_{i}' for i in range(5)] + ['full']:
        path = OUT / 'proposers' / (role + '.pt'); meta = read(path.with_suffix('.json'))
        assert meta['complete'], 'Finish every upstream role before extraction'
        assert meta['protocol_sha256'] == sha256(OUT / 'protocol.json')
        assert meta['checkpoint_sha256'] == sha256(path), role
        result[role] = meta['checkpoint_sha256']
    return result


def verify_raw(path, case, transform=None, legacy=False):
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z['metadata']))
    assert meta['sample_id'] == case['sample_id'] and meta['image_sha256'] == case['image_sha256']
    assert meta['image_size'] == case['image_size']
    assert meta['own_token_forward'] and meta['all_backbone_parameters_frozen']
    assert meta['ground_truth_in_model_input'] is False
    digest = hashlib.sha256(case['instruction'].encode()).hexdigest()
    if not legacy or 'instruction_sha256' in meta:
        assert meta['instruction_sha256'] == digest
    else:
        source = Path(path).parent.parent / 'manifest.jsonl'
        from crossfit_protocol import rows
        assert {r['sample_id']: r['instruction'] for r in rows(source)}[case['sample_id']] == case['instruction']
    if transform is not None:
        assert meta['transform'] == transform
    return meta


def public_case(case):
    return {k: case[k] for k in ('sample_id', 'image', 'image_size', 'image_sha256', 'instruction', 'dataset')}


def predict_global(models, case, group):
    dest = folder(group) / 'global' / (key(case['sample_id']) + '.json')
    role = role_for(case, group)
    path = raw_path(case) if group != 'test' else folder(group) / 'features' / (key(case['sample_id']) + '.npz')
    public = public_case(case)
    verify_raw(path, public, legacy=group != 'test')
    if not dest.exists():
        arrays, meta = raw_arrays(path)
        result = predict_raw(models[role], arrays, meta, public['image_size'], topk=64)
        result.update(sample_id=public['sample_id'], proposer_role=role,
            proposer_sha256=sha256(OUT / 'proposers' / (role + '.pt')),
            raw_feature_path=str(path), raw_feature_sha256=sha256(path))
        write_json(dest, result)
    result = read(dest)
    assert result['proposer_role'] == role and result['proposer_sha256'] == sha256(OUT / 'proposers' / (role + '.pt'))
    assert result['raw_feature_sha256'] == sha256(path)
    return result


def local_identity(case, transform):
    # Sharing requires identical pixels, instruction and exact crop transform.
    payload = dict(sample_id=case['sample_id'], image_sha256=case['image_sha256'],
                   instruction=case['instruction'], transform=transform)
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


@lru_cache(maxsize=1)
def legacy_identity():
    seal_path = Path('runs/candidate_scorer_v11/frozen.json')
    seal = read(seal_path)
    assert seal['complete']
    # These are the actual frozen model/feature contracts, not learned proposer
    # weights. Reusing a crop's four raw arrays cannot change the new crop.
    for name in ('conditioned_features', 'context_verifier', 'target_token', 'model_features'):
        p = 'src/clickagent/' + name + '.py'
        assert sha256(Path(p)) == seal['files'][p]
    identity = read(OUT / 'runtime_identity.json')
    assert identity['complete'] and identity['legacy_frozen_sha256'] == sha256(seal_path)
    return seal_path, seal['training_input_evidence']['files']


def identical_legacy_local(case, transform):
    path = Path('runs/candidate_scorer_v11/trainval/pipeline') / (key(case['sample_id']) + '.json')
    if not path.exists():
        return None
    _, evidence = legacy_identity()
    expected = evidence.get(os.path.normcase(os.path.abspath(path)))
    assert expected and sha256(path) == expected
    old = read(path)
    if old.get('old_crop_transform') != transform or not old.get('local_feature_path'):
        return None
    local = Path(old['local_feature_path'])
    assert evidence[os.path.normcase(os.path.abspath(local))] == sha256(local) == old['local_feature_sha256']
    verify_raw(local, dict(case, image_size=transform['crop_size']), transform)
    return local


def assemble(models, case, group, runtime=None):
    dest = folder(group) / 'pipeline' / (key(case['sample_id']) + '.json')
    if dest.exists():
        result = read(dest)
        assert result['proposer_role'] == role_for(case, group)
        assert result['proposer_sha256'] == sha256(OUT / 'proposers' / (result['proposer_role'] + '.pt'))
        assert result['pools'] == build_pools(result)
        return result
    public = public_case(case); role = role_for(case, group)
    gp = predict_global(models, case, group)
    result = dict(sample_id=public['sample_id'], proposer_role=role,
        proposer_sha256=gp['proposer_sha256'], global_result=gp,
        old_crop_result=None, old_crop_transform=None, local_feature_path=None,
        global_feature_path=gp['raw_feature_path'], global_feature_sha256=gp['raw_feature_sha256'],
        ground_truth_in_model_inputs=False)
    if gp['candidates']:
        with Image.open(public['image']) as im:
            crop, transform = proposal_crop(im.convert('RGB'), gp['candidates'][0]['parent_index'], gp['grid_shape'])
        local_path = OUT / ('test/local_features' if group == 'test' else 'trainval/local_features') / (local_identity(public, transform) + '.npz')
        legacy = identical_legacy_local(public, transform) if group != 'test' else None
        if legacy is not None:
            local_path = legacy
        if not local_path.exists():
            if runtime is None:
                raise RuntimeError('Frozen Qwen runtime required for new crop')
            arrays, meta = runtime.infer_spatial_features(crop, public['instruction'])
            meta.update(sample_id=public['sample_id'], image_sha256=public['image_sha256'], transform=transform,
                        instruction_sha256=hashlib.sha256(public['instruction'].encode()).hexdigest())
            save_features(local_path, arrays, meta)
        verify_raw(local_path, dict(public, image_size=list(crop.size)), transform)
        arrays, meta = raw_arrays(local_path)
        lp = predict_raw(models[role], arrays, meta, list(crop.size), topk=64)
        result.update(old_crop_result=lp, old_crop_transform=transform,
                      local_feature_path=str(local_path), local_feature_sha256=sha256(local_path))
        result['identical_legacy_raw_reused'] = legacy is not None
        if legacy is not None:
            old_pipeline = Path('runs/candidate_scorer_v11/trainval/pipeline') / (key(public['sample_id']) + '.json')
            contract = read(OUT / 'runtime_identity.json')
            result['legacy_raw_reuse_identity'] = dict(
                old_pipeline_path=str(old_pipeline), old_pipeline_sha256=sha256(old_pipeline),
                raw_path=str(local_path), raw_sha256=sha256(local_path),
                legacy_frozen_sha256=contract['legacy_frozen_sha256'],
                runtime_contract_sha256=contract['runtime_contract_sha256'],
                runtime_identity_sha256=sha256(OUT / 'runtime_identity.json'))
    result['pools'] = build_pools(result)
    write_json(dest, result)
    return result


def cache(group):
    upstream = checked_proposers()
    split = 'train' if group in ('oof', 'in_sample') else group
    entries = []; started = time.time()
    for i, case in enumerate(cases(split), 1):
        sid = key(case['sample_id']); pp = folder(group) / 'pipeline' / (sid + '.json'); pipeline = read(pp)
        assert pipeline['proposer_role'] == role_for(case, group)
        assert pipeline['proposer_sha256'] == upstream[pipeline['proposer_role']]
        assert pipeline['pools'] == build_pools(pipeline)
        path = OUT / 'cache' / group / (sid + '.npz')
        if not path.exists():
            public = public_case(case)
            arrays = base_cache(case if split in ('train', 'val') else public, pipeline, split)
            if bool(arrays['available']):
                ga, gm = raw_arrays(pipeline['global_feature_path'])
                la, lm = raw_arrays(pipeline['local_feature_path']) if pipeline['local_feature_path'] else (None, None)
                features = build_internal_inputs(ga, gm, la, lm, arrays['points'], public['image_size'], transform=pipeline['old_crop_transform'])
                assert not any('label' in k or 'target_geometry' in k for k in features)
                arrays.update({'i__' + k: v for k, v in features.items()})
                del ga, gm, la, lm, features
            arrays['pipeline_sha256'] = np.asarray(sha256(pp))
            save_npz(path, arrays)
        with np.load(path, allow_pickle=False) as z:
            assert ('labels' in z.files) == (split in ('train', 'val'))
            assert str(z['pipeline_sha256']) == sha256(pp)
            available = bool(z['available']); n = len(z['points'])
        evidence = {str(pp): sha256(pp), pipeline['global_feature_path']: pipeline['global_feature_sha256']}
        if pipeline['local_feature_path']:
            evidence[pipeline['local_feature_path']] = pipeline['local_feature_sha256']
        entries.append(dict(sample_id=case['sample_id'], path=str(path), sha256=sha256(path),
                            available=available, n_points=n, evidence=evidence))
        if i % 25 == 0:
            print(json.dumps(dict(stage='cache', group=group, i=i, elapsed=time.time()-started)), flush=True)
    write_json(OUT / 'cache' / (group + '_index.json'), dict(complete=True, group=group, cases=entries,
        has_labels=split in ('train', 'val'), annotation_derived_coordinates=False))


def main():
    p = argparse.ArgumentParser(); p.add_argument('--stage', choices=['global', 'local', 'cache'], required=True)
    p.add_argument('--group', choices=['in_sample', 'oof', 'val', 'test'], required=True)
    p.add_argument('--limit', type=int)
    args = p.parse_args(); torch.set_num_threads(4)
    if args.group == 'test':
        assert_sealed()
    checked_proposers()
    if args.stage == 'cache':
        cache(args.group); return
    split = 'train' if args.group in ('in_sample', 'oof') else args.group
    selected = cases(split)
    stage_folder = 'global' if args.stage == 'global' else 'pipeline'
    selected = [c for c in selected if not (folder(args.group) / stage_folder / (key(c['sample_id']) + '.json')).exists()]
    if args.limit:
        selected = selected[:args.limit]
    if not selected:
        return
    roles = sorted({role_for(c, args.group) for c in selected})
    models = {r: load_crossfit_proposer(OUT / 'proposers' / (r + '.pt'), 'cpu')[0].eval() for r in roles}
    if args.stage == 'local':
        assert read(OUT / 'runtime_identity.json')['complete']
    runtime = ContextFeatureRuntime(runtime_config()) if args.stage == 'local' else None
    if runtime:
        write_json(folder(args.group) / 'runtime.json', runtime.runtime_metadata)
    started = time.time()
    for i, case in enumerate(selected, 1):
        if args.group == 'test' and args.stage == 'local':
            public = public_case(case)
            assert sha256(Path(public['image'])) == public['image_sha256']
            with Image.open(public['image']) as im:
                image = im.convert('RGB')
            path = folder('test') / 'features' / (key(public['sample_id']) + '.npz')
            if not path.exists():
                arrays, meta = runtime.infer_spatial_features(image, public['instruction'])
                meta.update(sample_id=public['sample_id'], image_sha256=public['image_sha256'],
                    instruction_sha256=hashlib.sha256(public['instruction'].encode()).hexdigest())
                save_features(path, arrays, meta)
            native(runtime, image, public['instruction'], folder('test') / 'native' / (key(public['sample_id']) + '.json'))
        result = predict_global(models, case, args.group) if args.stage == 'global' else assemble(models, case, args.group, runtime)
        progress = dict(stage=args.stage, group=args.group, i=i, n=len(selected), elapsed=time.time()-started)
        write_json(folder(args.group) / ('progress_' + args.stage + '.json'), progress)
        print(json.dumps(progress), flush=True)


if __name__ == '__main__':
    main()
