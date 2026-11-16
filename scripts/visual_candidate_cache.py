"""Extra frozen-vision features for the dedicated V11 visual verifier."""
from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import torch
from PIL import Image

from clickagent.action_benchmark import sha256, write_json
from clickagent.visual_candidate_verifier import (
    FrozenVisualEncoder, build_visual_inputs, save_visual_encoding, load_visual_encoding,
)
from candidate_scorer_protocol import OUT
from extract_candidate_scorer import cases, root_for
from point_aligned_data import read
from run_target_token_experiment import key


def cache_visual_split(split, device):
    from cache_candidate_scorer import save_npz, raw_arrays
    group = 'trainval' if split in ('train', 'val') else split
    rows = [c for c in cases(group) if group != 'trainval' or c['experiment_split'] == split]
    manifest = {c['sample_id']: c for c in rows}
    index_path = OUT / 'cache' / (split + '_index.json'); index = read(index_path)
    assert index['complete']
    # An optional image-only worker can run while Qwen collects local states.
    # Give its live trainval pass ownership rather than racing on the same NPZ.
    progress = OUT / 'visual_precache_progress.json'
    completion = OUT / 'visual_precache_completion.json'
    while split in ('train', 'val') and progress.exists() and not completion.exists():
        if time.time() - progress.stat().st_mtime > 120:
            break  # A stopped worker leaves resumable, individually atomic files.
        time.sleep(5)
    runtime = None; started = time.time()
    for i, row in enumerate(index['cases'], 1):
        case = manifest[row['sample_id']]; dest = Path(row['visual_path'])
        if not dest.exists():
            with np.load(row['path'], allow_pickle=False) as z:
                arrays = {k: z[k].copy() for k in z.files if not k.startswith('i__')}
            if row['available']:
                encoding_path = OUT / 'vision_encodings' / (case['image_sha256'] + '.npz')
                if not encoding_path.exists():
                    if runtime is None:
                        runtime = FrozenVisualEncoder.from_local('models/Qwen3-VL-8B-Instruct', device=device,
                                                                 dtype=torch.float16, max_pixels=1048576)
                    assert sha256(Path(case['image'])) == case['image_sha256']
                    with Image.open(case['image']) as opened:
                        encoding = runtime.encode(opened.convert('RGB'))
                    encoding['metadata']['image_sha256'] = case['image_sha256']
                    save_visual_encoding(encoding_path, encoding)
                encoding = load_visual_encoding(encoding_path)
                assert encoding['metadata']['image_sha256'] == case['image_sha256']
                pipeline = read(root_for(group) / 'pipeline' / (key(case['sample_id']) + '.json'))
                raw, _ = raw_arrays(pipeline['global_feature_path'])
                inputs = build_visual_inputs(encoding, arrays['points'], raw['target_hidden'])
                assert not any('label' in k or 'target_geometry' in k for k in inputs)
                arrays.update({'v__' + k: v for k, v in inputs.items()})
                arrays['visual_encoding_sha256'] = np.asarray(sha256(encoding_path))
                del encoding, raw, inputs
            save_npz(dest, arrays)
        row['visual_sha256'] = sha256(dest)
        if i % 10 == 0:
            progress = dict(stage='visual_cache', split=split, i=i, n=len(index['cases']), elapsed_seconds=time.time()-started)
            write_json(OUT / 'cache' / (split + '_visual_progress.json'), progress)
            print(progress, flush=True)
    index['visual_complete'] = True
    index['extra_visual_max_pixels'] = 1048576
    index['visual_cache_seconds'] = time.time()-started
    write_json(index_path, index)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(); p.add_argument('--split', required=True); p.add_argument('--device', default='cuda:1')
    args = p.parse_args(); cache_visual_split(args.split, args.device)
