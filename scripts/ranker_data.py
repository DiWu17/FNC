"""Cache label-free candidate inputs separately from their supervision."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np

from clickagent.action_benchmark import sha256, write_json
from clickagent.candidate_ranker import build_candidate_inputs, candidate_supervision
from run_target_token_experiment import key, read_jsonl


def builder_digest():
    return hashlib.sha256(inspect.getsource(build_candidate_inputs).encode()).hexdigest()


def cache_candidates(out, split):
    protocol = json.loads((out/'protocol.json').read_text(encoding='utf-8'))
    cases = [c for c in read_jsonl(out/'manifest.jsonl') if c['experiment_split'] in (('train', 'val') if split == 'trainval' else ('test',))]
    if split == 'test':
        frozen = json.loads((out/'frozen_readouts.json').read_text(encoding='utf-8'))
        assert frozen['test_features_at_freeze'] == 0
        for entry in frozen['models'].values(): assert sha256(out/'models'/entry['file']) == entry['sha256']
    root = out if split == 'test' else Path(protocol['raw_feature_root_trainval'])
    folder = out/'candidate_cache'; folder.mkdir(exist_ok=True)
    for i, c in enumerate(cases):
        target = folder/f'{key(c["sample_id"])}.npz'
        if target.exists():
            with np.load(target, allow_pickle=False) as z: meta = json.loads(str(z['metadata']))
            assert meta['candidate_builder_digest'] == builder_digest()
            continue
        source = root/'features'/f'{key(c["sample_id"])}.npz'
        with np.load(source, allow_pickle=False) as z:
            meta = json.loads(str(z['metadata'])); arrays = {}
            assert meta['sample_id'] == c['sample_id'] and meta['image_sha256'] == c['image_sha256']
            assert meta['own_token_forward'] and not meta['ground_truth_in_model_input']
            if meta['target_available']:
                arrays = build_candidate_inputs(z['target_attention'], z['target_hidden'], z['image_hidden'], z['vision_hidden'], meta['grid_shape'])
                arrays.update(candidate_supervision(c['target_geometry'], meta['grid_shape'], arrays['parent_indices']))
        meta.update(candidate_builder_digest=builder_digest(), source_archive_sha256=sha256(source),
            candidate_source='Every own-name-token / raw-head argmax; unique parent cells, no GT injection',
            labels_separate_from_inputs=True)
        arrays['metadata'] = np.asarray(json.dumps(meta, ensure_ascii=False))
        temp = target.with_suffix('.tmp')
        with temp.open('wb') as stream: np.savez_compressed(stream, **arrays)
        temp.replace(target)
        if (i+1) % 100 == 0: print(f'Candidate cache {split}: {i+1}/{len(cases)}', flush=True)
    write_json(out/f'candidates_{split}_completion.json', dict(complete=True, n=len(cases), builder_digest=builder_digest()))


def load_records(out, split):
    cases = [c for c in read_jsonl(out/'manifest.jsonl') if c['experiment_split'] == split]
    result = []
    for c in cases:
        with np.load(out/'candidate_cache'/f'{key(c["sample_id"])}.npz', allow_pickle=False) as z:
            r = {k: z[k].copy() for k in z.files if k != 'metadata'}
            meta = json.loads(str(z['metadata']))
        assert meta['sample_id'] == c['sample_id'] and meta['candidate_builder_digest'] == builder_digest()
        r.update(case=c, metadata=meta); result.append(r)
    return result


def main():
    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, default=Path('runs/candidate_ranker_v5'))
    p.add_argument('--split', choices=['trainval', 'test'], default='trainval'); args = p.parse_args()
    cache_candidates(args.output, args.split)


if __name__ == '__main__': main()
