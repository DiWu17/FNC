"""Validate standalone vision extraction against archived full-VLM vision states."""
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from clickagent.action_benchmark import write_json
from clickagent.visual_candidate_verifier import FrozenVisualEncoder
from cache_candidate_scorer import raw_arrays
from run_target_token_experiment import read_jsonl
from train_context_point_critic import raw_path


def main():
    torch.set_num_threads(4)
    root = Path('runs/candidate_scorer_v11')
    rows = [c for c in read_jsonl(root / 'manifest.jsonl') if c['experiment_split'] == 'val'][:2]
    runtime = FrozenVisualEncoder.from_local('models/Qwen3-VL-8B-Instruct', device='cuda:1',
                                             dtype=torch.float16, max_pixels=262144)
    results = []
    for case in rows:
        raw, meta = raw_arrays(raw_path(case))
        with Image.open(case['image']) as im:
            encoding = runtime.encode(im.convert('RGB'))
        current = encoding['features'].reshape(-1, 1152).astype(np.float32)
        old = raw['vision_hidden'].astype(np.float32)
        assert current.shape == old.shape
        cosine = np.sum(current * old, -1) / np.maximum(np.linalg.norm(current, axis=-1) * np.linalg.norm(old, axis=-1), 1e-12)
        result = dict(sample_id=case['sample_id'], shape=list(old.shape),
                      mean_patch_cosine=float(cosine.mean()), minimum_patch_cosine=float(cosine.min()),
                      mean_abs_difference=float(np.abs(current-old).mean()), max_abs_difference=float(np.abs(current-old).max()),
                      bitwise_equal=bool(np.array_equal(current, old)))
        results.append(result); print(result, flush=True)
        assert result['mean_patch_cosine'] > .999 and result['minimum_patch_cosine'] > .99, result
    write_json(root / 'visual_encoder_alignment.json', dict(complete=True, validation_only=True, cases=results,
               conclusion='standalone visual extraction agrees with archived full-VLM visual patch states at matched pixels'))


if __name__ == '__main__':
    main()
