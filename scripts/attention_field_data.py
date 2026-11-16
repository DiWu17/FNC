"""Add full candidate attention fields without mutating any v5 cache or source."""
import argparse
import inspect
import hashlib
import json
from pathlib import Path

import numpy as np

from clickagent.action_benchmark import sha256, write_json
from clickagent.attention_field_ranker import build_attention_field
from ranker_data import load_records as load_base_records
from run_target_token_experiment import key, read_jsonl


def field_digest():
    return hashlib.sha256(inspect.getsource(build_attention_field).encode()).hexdigest()


def cache_fields(out, split):
    protocol = json.loads((out/'protocol.json').read_text(encoding='utf-8'))
    base = Path(protocol['base_run']); base_protocol = json.loads((base/'protocol.json').read_text())
    assert sha256(base/'manifest.jsonl') == protocol['manifest_sha256']
    cases = [c for c in read_jsonl(base/'manifest.jsonl') if c['experiment_split'] in
             (('train', 'val') if split == 'trainval' else ('test',))]
    if split == 'test':
        frozen = json.loads((out/'frozen_readouts.json').read_text())
        for entry in frozen['models'].values():
            assert sha256(out/'models'/entry['file']) == entry['sha256']
    folder = out/'field_cache'; folder.mkdir(exist_ok=True)
    for i, c in enumerate(cases):
        sid = key(c['sample_id']); target = folder/f'{sid}.npz'
        source = (base if split == 'test' else Path(base_protocol['raw_feature_root_trainval']))/'features'/f'{sid}.npz'
        base_cache = base/'candidate_cache'/f'{sid}.npz'
        if target.exists():
            with np.load(target, allow_pickle=False) as z: meta = json.loads(str(z['metadata']))
            assert meta['builder_sha256'] == field_digest() and meta['base_cache_sha256'] == sha256(base_cache)
            continue
        with np.load(base_cache, allow_pickle=False) as z:
            original = json.loads(str(z['metadata']))
            parents = z['parent_indices'] if original['target_available'] else None
        # Hash the source actually read; no reconstruction from labels or predictions.
        source_sha = sha256(source); assert source_sha == original['source_archive_sha256']
        arrays = {}
        if parents is not None:
            with np.load(source, allow_pickle=False) as z:
                a = z['target_attention']
                expected = np.unique(a.argmax(-1))
                np.testing.assert_array_equal(parents, expected)
                arrays['attention_field'] = build_attention_field(a, parents)
        meta = dict(sample_id=c['sample_id'], builder_sha256=field_digest(),
            base_cache_sha256=sha256(base_cache), raw_feature_sha256=source_sha,
            target_available=original['target_available'], labels_in_builder=False)
        arrays['metadata'] = np.asarray(json.dumps(meta))
        with target.open('wb') as stream: np.savez_compressed(stream, **arrays)
        if (i+1) % 100 == 0: print(f'Attention fields {split}: {i+1}/{len(cases)}', flush=True)
    write_json(out/f'fields_{split}_completion.json', dict(complete=True, n=len(cases), builder_sha256=field_digest()))


def load_records(out, split):
    protocol = json.loads((out/'protocol.json').read_text(encoding='utf-8'))
    records = load_base_records(Path(protocol['base_run']), split)
    for r in records:
        with np.load(out/'field_cache'/f'{key(r["case"]["sample_id"])}.npz', allow_pickle=False) as z:
            meta = json.loads(str(z['metadata']))
            assert meta['sample_id'] == r['case']['sample_id'] and meta['builder_sha256'] == field_digest()
            if r['metadata']['target_available']: r['attention_field'] = z['attention_field'].copy()
    return records


def main():
    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, default=Path('runs/attention_field_v6'))
    p.add_argument('--split', choices=['trainval', 'test'], default='trainval'); args = p.parse_args()
    cache_fields(args.output, args.split)


if __name__ == '__main__': main()
