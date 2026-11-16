"""Validation-only raw-feature CLI versus separated candidate-cache replay."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

from clickagent.action_benchmark import sha256, write_json
from clickagent.attention_field_ranker import load_field_ranker, make_field_batch
from clickagent.candidate_ranker import decode_ranker
from run_target_token_experiment import key, read_jsonl


def main():
    out = Path('runs/attention_field_v6'); base = Path('runs/candidate_ranker_v5')
    torch.set_num_threads(4)
    candidates = [c for c in read_jsonl(out/'manifest.jsonl') if c['experiment_split'] == 'val']
    selected = None
    for c in candidates:
        path = base/'candidate_cache'/f'{key(c["sample_id"])}.npz'
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z['metadata']))
            if meta['target_available']:
                # Deliberately whitelist deployment inputs; do not copy supervision.
                names = ['query_hidden', 'image_hidden', 'vision_hidden', 'source_features', 'candidate_features',
                    'source_token', 'source_head', 'source_candidate', 'parent_indices']
                r = {name: z[name].copy() for name in names}; selected = c; break
    assert selected is not None
    r.update(metadata=meta, case=dict(image_size=meta['image_size']))
    sid = key(selected['sample_id'])
    with np.load(out/'field_cache'/f'{sid}.npz', allow_pickle=False) as z: r['attention_field'] = z['attention_field'].copy()
    source = Path('runs/conditioned_v3/features')/f'{sid}.npz'
    command = [sys.executable, 'scripts/predict_attention_field_ranker.py', '--features', str(source)]
    result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8', check=True)
    actual = json.loads(result.stdout)
    model, _ = load_field_ranker(out/'models/all.pt', 'cpu')
    with torch.inference_mode():
        inputs, _ = make_field_batch([r], 'cpu', labels=False)
        expected = decode_ranker(model(**inputs), r, top_k=1)[0]
    assert actual['proposed_point'] == expected['point']
    assert actual['proposals'][0]['parent_index'] == expected['parent_index']
    write_json(out/'evaluation/cli_smoke.json', dict(complete=True, validation_only=True,
        sample_id=selected['sample_id'], geometry_supplied=False, raw_and_cached_point_agree=True,
        point=expected['point'], source_sha256=sha256(source), cli_sha256=sha256(Path('scripts/predict_attention_field_ranker.py'))))
    print('CLI raw/cache validation agreement', expected['point'], flush=True)


if __name__ == '__main__': main()
