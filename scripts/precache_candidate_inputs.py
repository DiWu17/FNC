"""Overlap CPU-only V11 input construction with train/val Qwen extraction.

Only a proper prefix of train/val is allowed. No indices are written: the
official cache stage subsequently verifies the completed files and builds the
whole-cohort index. All feature construction delegates to the same functions.
"""
import argparse
from pathlib import Path
import time

import numpy as np
import torch

from clickagent.action_benchmark import sha256, write_json
from clickagent.internal_candidate_scorer import build_internal_inputs
from clickagent.visual_candidate_verifier import build_visual_inputs, load_visual_encoding
from cache_candidate_scorer import base_cache, raw_arrays, save_npz
from candidate_scorer_protocol import OUT
from extract_candidate_scorer import cases, root_for
from point_aligned_data import read
from run_target_token_experiment import key


def cache_case(case):
    split = case['experiment_split']
    assert split in ('train', 'val')
    sid = key(case['sample_id'])
    pipeline_path = root_for('trainval') / 'pipeline' / (sid + '.json')
    if not pipeline_path.exists():
        return False
    pipeline = read(pipeline_path)
    internal_path = OUT / 'cache/internal' / split / (sid + '.npz')
    visual_path = OUT / 'cache/visual' / split / (sid + '.npz')
    if not internal_path.exists():
        for name in ('global', 'local'):
            path = pipeline[name + '_feature_path']
            if path:
                assert sha256(Path(path)) == pipeline[name + '_feature_sha256']
        arrays = base_cache(case, pipeline, split)
        if bool(arrays['available']):
            ga, gm = raw_arrays(pipeline['global_feature_path'])
            la, lm = raw_arrays(pipeline['local_feature_path']) if pipeline['local_feature_path'] else (None, None)
            features = build_internal_inputs(ga, gm, la, lm, arrays['points'], case['image_size'],
                                             transform=pipeline['old_crop_transform'])
            arrays.update({'i__' + k: v for k, v in features.items()})
            del ga, gm, la, lm, features
        arrays['pipeline_sha256'] = np.asarray(sha256(pipeline_path))
        save_npz(internal_path, arrays)
    with np.load(internal_path, allow_pickle=False) as z:
        assert str(z['pipeline_sha256']) == sha256(pipeline_path)
        assert 'labels' in z.files
        available = bool(z['available'])
        arrays = {k: z[k].copy() for k in z.files if not k.startswith('i__')}
    encoding_path = OUT / 'vision_encodings' / (case['image_sha256'] + '.npz')
    if not visual_path.exists() and (not available or encoding_path.exists()):
        if available:
            encoding = load_visual_encoding(encoding_path)
            assert encoding['metadata']['image_sha256'] == case['image_sha256']
            raw, _ = raw_arrays(pipeline['global_feature_path'])
            inputs = build_visual_inputs(encoding, arrays['points'], raw['target_hidden'])
            arrays.update({'v__' + k: v for k, v in inputs.items()})
            arrays['visual_encoding_sha256'] = np.asarray(sha256(encoding_path))
        save_npz(visual_path, arrays)
    return visual_path.exists()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=900)
    args = parser.parse_args()
    torch.set_num_threads(2)
    rows = cases('trainval')
    # Leave a substantial trailing extraction buffer to avoid competing writers
    # when the official full-cache stage starts. This is scheduling only.
    assert 0 < args.limit <= len(rows) - 100
    assert not (OUT / 'frozen.json').exists()
    selected = rows[:args.limit]
    pending = selected.copy()
    started = time.time()
    while pending:
        pending = [case for case in pending if not cache_case(case)]
        progress = dict(complete=not pending, prefix_size=len(selected),
                        complete_inputs=len(selected)-len(pending),
                        elapsed_seconds=time.time()-started, trainval_only=True,
                        writes_indices=False, additional_model_forwards=0)
        write_json(OUT / 'input_precache_progress.json', progress)
        print(progress, flush=True)
        if pending:
            time.sleep(10)
    write_json(OUT / 'input_precache_completion.json', progress)
