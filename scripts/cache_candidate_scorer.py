"""V11 candidate inputs, cached before fitting any head. Test inputs have no labels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from clickagent.action_benchmark import sha256, write_json
from clickagent.internal_candidate_scorer import build_internal_inputs
from clickagent.public_evaluation import point_in_geometry
from candidate_scorer_protocol import OUT
from extract_candidate_scorer import cases, root_for, assert_test_sealed
from point_aligned_data import read
from run_target_token_experiment import key


def raw_arrays(path):
    with np.load(path, allow_pickle=False) as z:
        metadata = json.loads(str(z['metadata']))
        arrays = {k: z[k].copy() for k in ('target_attention', 'target_hidden', 'image_hidden', 'vision_hidden')}
    return arrays, metadata


def save_npz(path, arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def base_cache(case, pipeline, split):
    pools = pipeline['pools']; points = np.asarray([v['point'] for v in pools['32']], np.int64).reshape(-1, 2)
    lookup = {tuple(p): i for i, p in enumerate(points.tolist())}
    result = dict(sample_id=np.asarray(case['sample_id']), available=np.asarray(bool(len(points))), points=points)
    for budget in (4, 8, 16, 32):
        result['pool_indices_' + str(budget)] = np.asarray([lookup[tuple(v['point'])] for v in pools[str(budget)]], np.int64)
    if split in ('train', 'val'):
        result['labels'] = np.asarray([point_in_geometry((p / np.asarray(case['image_size'])).tolist(), case['target_geometry'])
                                       for p in points], np.float32)
    result['annotation_derived_coordinates'] = np.asarray(False)
    return result


def internal(split):
    group = 'trainval' if split in ('train', 'val') else split
    rows = [c for c in cases(group) if group != 'trainval' or c['experiment_split'] == split]
    index_rows = []; started = time.time()
    for i, case in enumerate(rows, 1):
        sid = key(case['sample_id']); path = OUT / 'cache/internal' / split / (sid + '.npz')
        visual_path = OUT / 'cache/visual' / split / (sid + '.npz')
        pipeline_path = root_for(group) / 'pipeline' / (sid + '.json'); pipeline = read(pipeline_path)
        assert pipeline['global_feature_sha256'] == sha256(Path(pipeline['global_feature_path']))
        if pipeline['local_feature_path']:
            assert pipeline['local_feature_sha256'] == sha256(Path(pipeline['local_feature_path']))
        public = {k: case[k] for k in ('sample_id', 'image_size')}
        if not path.exists():
            arrays = base_cache(case if split in ('train', 'val') else public, pipeline, split)
            if bool(arrays['available']):
                ga, gm = raw_arrays(pipeline['global_feature_path'])
                la, lm = raw_arrays(pipeline['local_feature_path']) if pipeline['local_feature_path'] else (None, None)
                features = build_internal_inputs(ga, gm, la, lm, arrays['points'], public['image_size'],
                                                 transform=pipeline['old_crop_transform'])
                assert not any('label' in k or 'target_geometry' in k for k in features)
                arrays.update({'i__' + k: v for k, v in features.items()})
                del ga, gm, la, lm, features
            arrays['pipeline_sha256'] = np.asarray(sha256(pipeline_path))
            save_npz(path, arrays)
        with np.load(path, allow_pickle=False) as z:
            available = bool(z['available']); n = len(z['points'])
            assert ('labels' in z.files) == (split in ('train', 'val'))
            assert str(z['pipeline_sha256']) == sha256(pipeline_path)
        index_rows.append(dict(sample_id=case['sample_id'], path=str(path), visual_path=str(visual_path),
                               available=available, n_points=n, sha256=sha256(path),
                               pipeline_path=str(pipeline_path), pipeline_sha256=sha256(pipeline_path),
                               raw_files={p: sha256(Path(p)) for p in
                                          (pipeline['global_feature_path'], pipeline['local_feature_path']) if p}))
        if i % 25 == 0:
            print(json.dumps(dict(stage='internal_cache', split=split, i=i, n=len(rows), elapsed_seconds=time.time()-started)), flush=True)
    dest = OUT / 'cache' / (split + '_index.json')
    old = read(dest) if dest.exists() else {}
    previous = {r['sample_id']: r for r in old.get('cases', [])}
    visual_complete = bool(old.get('visual_complete'))
    for row in index_rows:
        prior = previous.get(row['sample_id'], {})
        visual_path = Path(row['visual_path'])
        if prior.get('visual_sha256') and visual_path.exists() and sha256(visual_path) == prior['visual_sha256']:
            row['visual_sha256'] = prior['visual_sha256']
        else:
            visual_complete = False
    write_json(dest, dict(complete=True, visual_complete=visual_complete, split=split,
                         cases=index_rows, has_labels=split in ('train', 'val'), real_candidate_pools=True,
                         annotation_derived_coordinates=False, elapsed_seconds=time.time()-started))


def visual(split, device):
    # The independent visual module supplies the frozen encoder and ROI builder.
    from visual_candidate_cache import cache_visual_split
    cache_visual_split(split, device)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--stage', choices=['internal', 'visual'], required=True)
    p.add_argument('--split', choices=['train', 'val', 'diagnostic', 'test'], required=True)
    p.add_argument('--device', default='cuda:1')
    args = p.parse_args(); torch.set_num_threads(4)
    if args.split == 'test':
        assert_test_sealed()
    internal(args.split) if args.stage == 'internal' else visual(args.split, args.device)
