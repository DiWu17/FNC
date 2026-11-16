"""Small validation-only runtime check before V11 training or fresh extraction."""
import json
from pathlib import Path
import time

import numpy as np
import torch
from PIL import Image

from clickagent.action_benchmark import write_json
from clickagent.visual_candidate_verifier import FrozenVisualEncoder, save_visual_encoding
from run_target_token_experiment import read_jsonl


def main():
    torch.set_num_threads(4)
    root = Path('runs/candidate_scorer_v11')
    rows = [r for r in read_jsonl(root / 'manifest.jsonl') if r['experiment_split'] == 'val']
    rows = sorted(rows, key=lambda r: -(r['image_size'][0] * r['image_size'][1]))[:2]
    runtime = FrozenVisualEncoder.from_local('models/Qwen3-VL-8B-Instruct', device='cuda:1',
                                             dtype=torch.float16, max_pixels=1048576)
    results = []
    for row in rows:
        with Image.open(row['image']) as im:
            encoding = runtime.encode(im.convert('RGB'))
        assert np.isfinite(encoding['features']).all() and np.isfinite(encoding['merged_features']).all()
        assert encoding['metadata']['all_visual_parameters_frozen']
        assert not encoding['metadata']['language_model_loaded'] and not encoding['metadata']['gui_actor_used']
        encoding['metadata']['image_sha256'] = row['image_sha256']
        dest = root / 'vision_encodings' / (row['image_sha256'] + '.npz')
        save_visual_encoding(dest, encoding)
        result = dict(sample_id=row['sample_id'], original_size=row['image_size'],
                      detail_shape=list(encoding['features'].shape), semantic_shape=list(encoding['merged_features'].shape),
                      timings=encoding['metadata']['timing_seconds'],
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(1))
        results.append(result); print(json.dumps(result), flush=True)
    write_json(root / 'visual_runtime_smoke.json', dict(complete=True, validation_only=True, cases=results,
               model_load_seconds=runtime.load_seconds, time=time.time()))


if __name__ == '__main__':
    main()
