"""Read-only adapters for v7 train/val and label-free v8/crop feature archives."""
import json
from pathlib import Path

import numpy as np
import torch

from clickagent.attention_field_ranker import build_attention_field, make_field_batch
from clickagent.candidate_ranker import build_candidate_inputs, candidate_supervision
from clickagent.action_benchmark import sha256
from run_target_token_experiment import key, read_jsonl
from semantic_head_data import load_records as load_v7_records


OUT=Path('runs/point_aligned_v8'); BASE=Path('runs/semantic_head_v7')


def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def training_records(split):
    records=load_v7_records(BASE,split)
    manifest={r['sample_id']:r for r in read_jsonl(OUT/'manifest.jsonl') if r['experiment_split']==split}
    assert {r['case']['sample_id'] for r in records}==set(manifest)
    assert all(r['case']==manifest[r['case']['sample_id']] for r in records)
    return records


def batch(records,device,labels=True):
    inputs,target=make_field_batch(records,device,labels)
    if labels:
        target['label_fixed_point_hit']=torch.from_numpy(np.concatenate([r['label_fixed_point_hit'] for r in records])).to(device)
    return inputs,target


def archive_record(path,case,labels=False):
    with np.load(path,allow_pickle=False) as z:
        meta=json.loads(str(z['metadata'])); r={}
        if meta['target_available']:
            r=build_candidate_inputs(z['target_attention'],z['target_hidden'],z['image_hidden'],z['vision_hidden'],meta['grid_shape'])
            r['attention_field']=build_attention_field(z['target_attention'],r['parent_indices'])
            if labels:r.update(candidate_supervision(case['target_geometry'],meta['grid_shape'],r['parent_indices']))
    r.update(case=case,metadata=meta)
    return r


def fresh_records(labels=False):
    cases=[r for r in read_jsonl(OUT/'manifest.jsonl') if r['experiment_split']=='test']
    return [archive_record(OUT/'features'/f'{key(c["sample_id"])}.npz',c,labels) for c in cases]


def frozen_source_identity():
    frozen=read(BASE/'frozen_readouts.json')
    for name,digest in frozen['sources'].items():assert sha256(Path(name))==digest,name
    return dict(v7_manifest_sha256=sha256(BASE/'manifest.jsonl'),v7_frozen_sha256=sha256(BASE/'frozen_readouts.json'),
        training_identity_sha256=sha256(BASE/'training_identity.json'),
        point_caches={p.name:sha256(p) for p in sorted((BASE/'point_cache').glob('*.npz'))},
        candidate_sources_unchanged=True)
