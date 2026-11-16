"""Prepare train/validation image-only encodings independently of candidate extraction."""
import json
import argparse
from pathlib import Path
import time

import torch
from PIL import Image

from clickagent.action_benchmark import sha256, write_json
from clickagent.visual_candidate_verifier import FrozenVisualEncoder, save_visual_encoding, load_visual_encoding
from run_target_token_experiment import read_jsonl


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=['trainval', 'diagnostic'], default='trainval')
    parser.add_argument('--wait-trainval', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    root = Path('runs/candidate_scorer_v11')
    if args.wait_trainval:
        while not (root / 'visual_precache_completion.json').exists():
            time.sleep(5)
    if args.split == 'trainval':
        rows = [r for r in read_jsonl(root / 'manifest.jsonl') if r['experiment_split'] in ('train', 'val')]
        prefix = 'visual_precache'
    else:
        rows = read_jsonl(root / 'diagnostic_manifest.jsonl')
        prefix = 'visual_diagnostic_precache'
    images = {r['image_sha256']: r for r in rows}
    runtime = None; started = time.time()
    for i, (digest, case) in enumerate(sorted(images.items()), 1):
        dest = root / 'vision_encodings' / (digest + '.npz')
        if dest.exists():
            assert load_visual_encoding(dest)['metadata']['image_sha256'] == digest
            continue
        if runtime is None:
            runtime = FrozenVisualEncoder.from_local('models/Qwen3-VL-8B-Instruct', device='cuda:1',
                dtype=torch.float16, max_pixels=1048576)
        assert sha256(Path(case['image'])) == digest
        with Image.open(case['image']) as image:
            encoding = runtime.encode(image.convert('RGB'))
        encoding['metadata'].update(image_sha256=digest, offline_trainval_precache=args.split == 'trainval',
            offline_diagnostic_precache=args.split == 'diagnostic',
            concurrent_workload='frozen Qwen local trainval extraction; not an isolated latency benchmark')
        save_visual_encoding(dest, encoding)
        progress = dict(i=i, n=len(images), elapsed_seconds=time.time()-started)
        write_json(root / (prefix + '_progress.json'), progress)
        print(json.dumps(progress), flush=True)
    write_json(root / (prefix + '_completion.json'), dict(complete=True, n=len(images),
        elapsed_seconds=time.time()-started, training_validation_only=args.split == 'trainval',
        reused_diagnostic_only=args.split == 'diagnostic'))
