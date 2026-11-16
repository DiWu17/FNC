"""V11 real candidate pools. Predictions accept only public, label-free inputs."""
from __future__ import annotations

import argparse
import hashlib
from functools import lru_cache
import json
from pathlib import Path
import time

import torch
import numpy as np
from PIL import Image

from clickagent.action_benchmark import sha256, write_json
from clickagent.candidate_budget import build_pools
from clickagent.context_verifier import ContextFeatureRuntime
from clickagent.model_features import ExtractionConfig
from clickagent.point_aligned_inference import proposal_crop
from point_aligned_data import archive_record, read
from point_aligned_predict import load_predictors, predict_record
from run_point_aligned_inference import save_features, native
from run_target_token_experiment import key, read_jsonl
from train_context_point_critic import raw_path

OUT = Path('runs/candidate_scorer_v11')
PRIMARY = 'joint_connected_s1'


def cases(split):
    if split == 'diagnostic':
        return read_jsonl(OUT / 'diagnostic_manifest.jsonl')
    return [c for c in read_jsonl(OUT / 'manifest.jsonl')
            if c['experiment_split'] in (('train', 'val') if split == 'trainval' else (split,))]


def public_case(case):
    return {k: case[k] for k in ('sample_id', 'image', 'image_size', 'image_sha256', 'instruction', 'dataset')}


def root_for(split):
    return OUT / ('diagnostic' if split == 'diagnostic' else 'test' if split == 'test' else 'trainval')


def full_path(case, split):
    if split == 'trainval':
        return raw_path(case)
    if split == 'diagnostic':
        return Path('runs/context_verifier_v9/features') / (key(case['sample_id']) + '.npz')
    return root_for(split) / 'features' / (key(case['sample_id']) + '.npz')


def runtime_config():
    cfg = read('runs/candidate_v2/identity.json')['config']
    cfg.update(max_new_tokens=64, seed=20261131, conditions=['prefill'],
               offload_folder=str((OUT / 'offload').absolute()))
    cfg.update(layers=tuple(cfg['layers']), conditions=tuple(cfg['conditions']),
               max_memory={int(k) if str(k).isdigit() else k: v for k, v in cfg['max_memory'].items()})
    return ExtractionConfig(**cfg)


@lru_cache(maxsize=2)
def source_instructions(path):
    return {r['sample_id']: r['instruction'] for r in read_jsonl(Path(path))}


def assert_feature_identity(path, case, transform=None, legacy=False):
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z['metadata']))
    assert meta['sample_id'] == case['sample_id'], str(path)
    assert meta['image_sha256'] == case['image_sha256'], str(path)
    assert meta['image_size'] == case['image_size'], str(path)
    assert meta['own_token_forward'] and meta['all_backbone_parameters_frozen']
    assert meta['ground_truth_in_model_input'] is False
    instruction_sha = hashlib.sha256(case['instruction'].encode()).hexdigest()
    if 'instruction_sha256' in meta:
        assert meta['instruction_sha256'] == instruction_sha
    else:
        assert legacy, 'New feature archives require an instruction hash'
        source = Path(path).parent.parent / 'manifest.jsonl'
        assert source_instructions(str(source))[case['sample_id']] == case['instruction']
    if transform is not None:
        assert meta['transform'] == transform
    return meta


def predict_global(models, case, split):
    root = root_for(split); dest = root / 'global' / (key(case['sample_id']) + '.json')
    assert_feature_identity(full_path(case, split), case, legacy=split == 'trainval')
    digest = sha256(full_path(case, split))
    old = read(dest) if dest.exists() else None
    if old is None or old.get('raw_feature_sha256') != digest:
        record = archive_record(full_path(case, split), case, labels=False)
        result = predict_record(record, models, only=[PRIMARY], top_k=64)[PRIMARY]
        result['raw_feature_sha256'] = digest
        write_json(dest, result)
    return read(dest)


def assemble(models, case, split, runtime=None):
    root = root_for(split); sid = key(case['sample_id']); dest = root / 'pipeline' / (sid + '.json')
    if dest.exists():
        old = read(dest)
        assert old['global_feature_sha256'] == sha256(Path(old['global_feature_path']))
        assert_feature_identity(old['global_feature_path'], case, legacy=split == 'trainval')
        if old['local_feature_path']:
            assert old['local_feature_sha256'] == sha256(Path(old['local_feature_path']))
            local_case = dict(case, image_size=old['old_crop_transform']['crop_size'])
            assert_feature_identity(old['local_feature_path'], local_case, old['old_crop_transform'])
        return old
    if split == 'diagnostic':
        old = read(Path('runs/context_verifier_v9/pipeline') / (sid + '.json'))
        result = {k: old[k] for k in ('global_result', 'old_crop_result', 'old_crop_transform')}
        local_path = Path('runs/context_verifier_v9/old_crop_features') / (sid + '.npz')
    else:
        gp = predict_global(models, case, split)
        result = dict(global_result=gp, old_crop_result=None, old_crop_transform=None)
        local_path = root / 'old_crop_features' / (sid + '.npz')
        if gp['candidates']:
            with Image.open(case['image']) as opened:
                crop, transform = proposal_crop(opened.convert('RGB'), gp['candidates'][0]['parent_index'], gp['grid_shape'])
            if not local_path.exists():
                if runtime is None:
                    raise RuntimeError('Local feature extraction needs the frozen Qwen runtime')
                arrays, meta = runtime.infer_spatial_features(crop, case['instruction'])
                assert meta['all_backbone_parameters_frozen'] and meta['own_token_forward']
                meta.update(sample_id=case['sample_id'], image_sha256=case['image_sha256'], transform=transform,
                            instruction_sha256=hashlib.sha256(case['instruction'].encode()).hexdigest())
                save_features(local_path, arrays, meta)
            local_case = dict(sample_id=case['sample_id'], image_size=list(crop.size), image_sha256=case['image_sha256'])
            assert_feature_identity(local_path, dict(local_case, instruction=case['instruction']), transform)
            local = archive_record(local_path, local_case, labels=False)
            assert local['metadata']['transform'] == transform
            result.update(old_crop_transform=transform,
                          old_crop_result=predict_record(local, models, only=[PRIMARY], top_k=64)[PRIMARY])
    result.update(sample_id=case['sample_id'], global_feature_path=str(full_path(case, split)),
                  local_feature_path=str(local_path) if result['old_crop_result'] is not None else None,
                  global_feature_sha256=sha256(full_path(case, split)),
                  local_feature_sha256=sha256(local_path) if result['old_crop_result'] is not None else None,
                  ground_truth_in_model_inputs=False)
    result['pools'] = build_pools(result)
    write_json(dest, result)
    return result


def assert_test_sealed():
    seal = read(OUT / 'frozen.json')
    assert seal['complete'] and seal['fresh_test_features_at_freeze'] == 0
    for path, digest in seal['files'].items():
        assert sha256(Path(path)) == digest, path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stage', required=True, choices=['global_trainval', 'local_trainval', 'diagnostic', 'test'])
    p.add_argument('--limit', type=int)
    args = p.parse_args(); torch.set_num_threads(4)
    split = 'trainval' if args.stage.endswith('trainval') else args.stage
    if split == 'test':
        assert_test_sealed()
    root = root_for(split)
    for folder in ('global', 'pipeline', 'features', 'old_crop_features', 'native'):
        (root / folder).mkdir(parents=True, exist_ok=True)
    rows = cases(split)
    models = load_predictors(primary_only=True)
    runtime = None; start = time.time()
    if args.stage == 'global_trainval':
        pending = [c for c in rows if not (root / 'global' / (key(c['sample_id']) + '.json')).exists()]
    else:
        pending = [c for c in rows if not (root / 'pipeline' / (key(c['sample_id']) + '.json')).exists()]
    if args.limit:
        pending = pending[:args.limit]
    if pending and args.stage in ('local_trainval', 'test'):
        runtime = ContextFeatureRuntime(runtime_config())
        write_json(root / 'runtime.json', runtime.runtime_metadata)
    for i, original in enumerate(pending, 1):
        case = public_case(original)
        if args.stage == 'global_trainval':
            result = predict_global(models, case, split)
        else:
            if split == 'test':
                assert sha256(Path(case['image'])) == case['image_sha256']
                with Image.open(case['image']) as opened:
                    image = opened.convert('RGB')
                path = full_path(case, split)
                if not path.exists():
                    arrays, meta = runtime.infer_spatial_features(image, case['instruction'])
                    assert meta['own_token_forward'] and meta['all_backbone_parameters_frozen']
                    assert meta['ground_truth_in_model_input'] is False
                    meta.update(sample_id=case['sample_id'], image_sha256=case['image_sha256'],
                                instruction_sha256=hashlib.sha256(case['instruction'].encode()).hexdigest())
                    save_features(path, arrays, meta)
                native(runtime, image, case['instruction'], root / 'native' / (key(case['sample_id']) + '.json'))
            result = assemble(models, case, split, runtime)
        progress = dict(stage=args.stage, i=i, pending=len(pending), sample_id=case['sample_id'],
                        elapsed_seconds=time.time()-start)
        write_json(root / ('progress_' + args.stage + '.json'), progress)
        print(json.dumps(progress), flush=True)
    folder = 'global' if args.stage == 'global_trainval' else 'pipeline'
    complete = all((root / folder / (key(c['sample_id']) + '.json')).exists() for c in rows)
    write_json(root / (args.stage + '_completion.json'), dict(complete=complete, n=len(rows),
               ground_truth_in_model_inputs=False, manifest_sha256=sha256(OUT / 'manifest.jsonl')))


if __name__ == '__main__':
    main()
