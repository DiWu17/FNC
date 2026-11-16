"""References old features; caches new label-free inputs and frozen fine-point labels."""
import argparse
from collections import defaultdict
from functools import lru_cache
import json
from pathlib import Path

import numpy as np
import torch

from clickagent.action_benchmark import sha256, write_json
from clickagent.attention_field_ranker import build_attention_field, make_field_batch, subset_field_candidates
from clickagent.candidate_ranker import (build_candidate_inputs, candidate_supervision,
    decode_ranker, load_ranker, make_ranker_batch)
from clickagent.spatial_geometry import shape_from_geometry
from evaluate_conditioned_readouts import score_point
from run_target_token_experiment import key, read_jsonl
from ranker_data import builder_digest
from attention_field_data import field_digest


BASE = Path('runs/candidate_ranker_v5'); FIELD = Path('runs/attention_field_v6')


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


@lru_cache(maxsize=None)
def _digest_stat(path, size, modified):
    return sha256(Path(path))


def checked_digest(path):
    path = Path(path); stat = path.stat()
    return _digest_stat(str(path), stat.st_size, stat.st_mtime_ns)


def cases_for(out, split, regime='expanded'):
    rows = [c for c in read_jsonl(out/'manifest.jsonl') if c['experiment_split'] == split]
    return [c for c in rows if not c.get('training_extension')] if split == 'train' and regime == 'base' else rows


def load_input_record(out, case):
    name = key(case['sample_id'])+'.npz'; old = BASE/'candidate_cache'/name
    if old.exists() and not case.get('training_extension') and case['experiment_split'] != 'test':
        raw_root = Path('runs/conditioned_v3') if case['experiment_split'] in ['train', 'val'] else BASE
        raw = raw_root/'features'/name; field_path = FIELD/'field_cache'/name
        with np.load(old, allow_pickle=False) as z:
            meta = json.loads(str(z['metadata'])); r = {k: z[k].copy() for k in z.files if k != 'metadata'}
        assert meta['candidate_builder_digest'] == builder_digest()
        assert meta['source_archive_sha256'] == checked_digest(raw)
        identity = dict(candidate_sha256=checked_digest(old), raw_sha256=checked_digest(raw),
            candidate_builder_digest=builder_digest(), field_builder_digest=field_digest())
        if meta['target_available']:
            with np.load(field_path, allow_pickle=False) as z:
                field_meta = json.loads(str(z['metadata']))
                assert field_meta['sample_id'] == case['sample_id'] and field_meta['builder_sha256'] == field_digest()
                assert field_meta['base_cache_sha256'] == identity['candidate_sha256']
                assert field_meta['raw_feature_sha256'] == identity['raw_sha256']
                r['attention_field'] = z['attention_field'].copy()
            identity['field_sha256'] = checked_digest(field_path)
    else:
        folder = out/'candidate_cache'; folder.mkdir(exist_ok=True); cached = folder/name
        raw = out/'features'/name
        if cached.exists():
            with np.load(cached, allow_pickle=False) as z:
                meta = json.loads(str(z['metadata'])); r = {k: z[k].copy() for k in z.files if k != 'metadata'}
        else:
            with np.load(raw, allow_pickle=False) as z:
                meta = json.loads(str(z['metadata'])); r = {}
                if meta['target_available']:
                    r = build_candidate_inputs(z['target_attention'], z['target_hidden'], z['image_hidden'], z['vision_hidden'], meta['grid_shape'])
                    r['attention_field'] = build_attention_field(z['target_attention'], r['parent_indices'])
                    r.update(candidate_supervision(case['target_geometry'], meta['grid_shape'], r['parent_indices']))
            meta.update(source_archive_sha256=sha256(raw), labels_separate_from_inputs=True,
                manifest_sha256=sha256(out/'manifest.jsonl'), candidate_builder_digest=builder_digest(),
                field_builder_digest=field_digest())
            with cached.open('wb') as stream: np.savez_compressed(stream, metadata=np.asarray(json.dumps(meta)), **r)
        assert meta['manifest_sha256'] == checked_digest(out/'manifest.jsonl')
        assert meta['candidate_builder_digest'] == builder_digest() and meta['field_builder_digest'] == field_digest()
        assert meta['source_archive_sha256'] == checked_digest(raw)
        identity = dict(candidate_sha256=checked_digest(cached), raw_sha256=checked_digest(raw),
            candidate_builder_digest=builder_digest(), field_builder_digest=field_digest())
    assert meta['sample_id'] == case['sample_id'] and meta['image_sha256'] == case['image_sha256']
    assert meta['own_token_forward'] and not meta['ground_truth_in_model_input']
    r.update(case=case, metadata=meta, cache_identity=identity); return r


def load_records(out, split, regime='expanded', quality=True):
    records = [load_input_record(out, c) for c in cases_for(out, split, regime)]
    if quality:
        protocol = read(out/'protocol.json')
        for r in records:
            if not r['metadata']['target_available']: continue
            with np.load(out/'point_cache'/f'{key(r["case"]["sample_id"])}.npz', allow_pickle=False) as z:
                meta = json.loads(str(z['metadata']))
                assert meta['manifest_sha256'] == protocol['manifest_sha256']
                assert meta['fine_sha256'] == protocol['fixed_fine_sha256']
                assert meta['sample_id'] == r['case']['sample_id'] and meta['input_identity'] == r['cache_identity']
                np.testing.assert_array_equal(z['parent_indices'], r['parent_indices'])
                for name in ['frozen_points', 'frozen_point_features', 'label_fixed_point_hit', 'label_hard_negative']:
                    r[name] = z[name].copy()
    return records


def subset_record(record, keep):
    r = subset_field_candidates(record, keep)
    for name in ['frozen_points', 'frozen_point_features', 'label_fixed_point_hit', 'label_hard_negative']:
        r[name] = record[name][keep]
    return r


def make_batch(records, device, labels=True):
    inputs, targets = make_field_batch(records, device, labels=labels)
    inputs['frozen_point_features'] = torch.from_numpy(np.concatenate([r['frozen_point_features'] for r in records])).to(device)
    if labels:
        for name in ['label_fixed_point_hit', 'label_hard_negative']:
            targets[name] = torch.from_numpy(np.concatenate([r[name] for r in records])).to(device)
    return inputs, targets


@torch.inference_mode()
def cache_points(out, split, device, regime='expanded'):
    protocol = read(out/'protocol.json'); cases = cases_for(out, split, regime)
    folder = out/'point_cache'; folder.mkdir(exist_ok=True); pending = []; existing_named = 0
    for c in cases:
        dest = folder/f'{key(c["sample_id"])}.npz'
        if dest.exists():
            with np.load(dest, allow_pickle=False) as z: meta = json.loads(str(z['metadata']))
            assert meta['manifest_sha256'] == protocol['manifest_sha256'] and meta['fine_sha256'] == protocol['fixed_fine_sha256']
            assert meta['sample_id'] == c['sample_id']
            assert meta['input_identity'] == load_input_record(out, c)['cache_identity']
            existing_named += bool(meta.get('target_available', True))
        else: pending.append(c)
    records = [load_input_record(out, c) for c in pending]
    train_groups = defaultdict(list)
    for c in cases_for(out, 'train'): train_groups[c['pixel_sha256']].append(c)
    entry = read(BASE/'frozen_readouts.json')['models']['full']; path = BASE/'models'/entry['file']
    assert sha256(path) == protocol['fixed_fine_sha256'] == entry['sha256']
    model, _ = load_ranker(path, device)
    named = [r for r in records if r['metadata']['target_available']]
    for r in records:
        if not r['metadata']['target_available']:
            meta = dict(manifest_sha256=protocol['manifest_sha256'], fine_sha256=protocol['fixed_fine_sha256'],
                sample_id=r['case']['sample_id'], target_available=False, input_identity=r['cache_identity'])
            with (folder/f'{key(r["case"]["sample_id"])}.npz').open('wb') as stream:
                np.savez_compressed(stream, metadata=np.asarray(json.dumps(meta)))
    for start in range(0, len(named), 8):
        group = named[start:start+8]; inputs, _ = make_ranker_batch(group, device, labels=False); output = model(**inputs)
        cpu = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in output.items()}; cursor = 0
        for i, r in enumerate(group):
            c = r['case']; n = len(r['parent_indices']); dest = folder/f'{key(c["sample_id"])}.npz'
            logits = cpu['child_logits'][cursor:cursor+n]; probs = logits.softmax(-1).numpy()
            child = probs.argmax(-1); offsets = cpu['offsets'][cursor:cursor+n].numpy()[np.arange(n), child]
            points = np.empty((n, 2), np.int64)
            for pred in decode_ranker(cpu, r, i, top_k=n): points[pred['candidate_index']] = pred['point']
            gh, gw = r['metadata']['grid_shape']; rr, cc = np.divmod(r['parent_indices'], gw)
            xy = points/np.array(c['image_size'])[None]
            entropy = -(probs*np.log(np.maximum(probs, 1e-12))).sum(-1)/np.log(4)
            relative = xy*np.array([gw, gh])[None]-np.stack((cc, rr), -1)
            features = np.concatenate((probs, offsets, entropy[:, None], xy, relative, np.tile([1/gw, 1/gh], (n, 1))), -1).astype(np.float16)
            assert features.shape == (n, 13)
            quality = np.array([score_point(p.tolist(), c) for p in points], np.float32)
            hard = np.zeros(n, bool)
            if split == 'train':
                import shapely
                from shapely.ops import unary_union
                others = [o for o in train_groups[c['pixel_sha256']] if o['sample_id'] != c['sample_id']]
                if others:
                    other = unary_union([shape_from_geometry(o['target_geometry']) for o in others])
                    cells = shapely.box(cc/gw, rr/gh, (cc+1)/gw, (rr+1)/gh)
                    hard = (shapely.area(shapely.intersection(cells, other)) > 0) & (r['label_area'].sum(-1) == 0)
            meta = dict(manifest_sha256=protocol['manifest_sha256'], fine_sha256=protocol['fixed_fine_sha256'],
                sample_id=c['sample_id'], labels_are_supervision_only=True, target_available=True, input_identity=r['cache_identity'])
            with dest.open('wb') as stream:
                np.savez_compressed(stream, metadata=np.asarray(json.dumps(meta)), parent_indices=r['parent_indices'],
                    frozen_points=points, frozen_point_features=features, label_fixed_point_hit=quality,
                    label_hard_negative=hard.astype(np.float32))
            cursor += n
        print(f'POINT CACHE {split}: {min(start+8,len(named))}/{len(named)}', flush=True)
    suffix = '_base' if split == 'train' and regime == 'base' else ''
    write_json(out/f'point_cache_{split}{suffix}_completion.json', dict(complete=True, n=len(cases), named=existing_named+len(named),
        manifest_sha256=protocol['manifest_sha256'], fine_sha256=protocol['fixed_fine_sha256']))


def main():
    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, default=Path('runs/semantic_head_v7'))
    p.add_argument('--split', choices=['train','val','diagnostic','test'], required=True)
    p.add_argument('--regime', choices=['base','expanded'], default='expanded')
    p.add_argument('--device', default='cpu'); args = p.parse_args(); torch.set_num_threads(4)
    if args.split in ['test','diagnostic']: assert (args.output/'frozen_readouts.json').exists()
    cache_points(args.output, args.split, args.device, args.regime)


if __name__ == '__main__': main()
