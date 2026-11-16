"""Meaningful pipeline protections and legacy replay for the V15 scaling run."""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import run_quality_readout_v15 as run


def test_training_cursor_resume_does_not_repeat_or_skip_queries():
    epoch=1;position=0;full=[]
    for _ in range(9):
        ix,epoch,position=run.next_batch(19,run.SEEDS[0],epoch,position)
        full.append(ix.tolist())
    assert all(sorted(sum(full[i:i+3],[]))==list(range(19)) for i in (0,3,6))
    epoch=1;position=0
    for _ in range(4):_,epoch,position=run.next_batch(19,run.SEEDS[0],epoch,position)
    resumed=[]
    for _ in range(5):
        ix,epoch,position=run.next_batch(19,run.SEEDS[0],epoch,position);resumed.append(ix.tolist())
    assert resumed==full[4:]


def test_labels_are_rejected_before_extraction_or_pipeline_lookup():
    extractor=object.__new__(run.Extractor)
    public={k:None for k in run.PUBLIC}
    with pytest.raises(ValueError,match='Labels in extraction input'):
        extractor.case(dict(public,target_geometry={'boxes':[]}))
    with pytest.raises(ValueError,match='public fields only'):
        extractor.view(dict(public,target_geometry={}),None)


@pytest.mark.parametrize('seed',run.SEEDS)
def test_fresh_model_is_identical_to_archived_initialization(seed):
    torch.manual_seed(seed);model=run.RetentionModel()
    initial=run.digest({k:hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest() for k,v in model.state_dict().items()})
    expected=run.read(run.OLD/f'models/aligned_s{run.SEEDS.index(seed)+1}.json')
    assert initial==expected['initial_state_sha256']
    assert sum(p.numel() for p in model.parameters())==22050


def test_cpu_replay_preserves_legacy_crop_points_and_extra_states(tmp_path,monkeypatch):
    torch.set_num_threads(4)
    c=run.rows(run.V13/'data/val.jsonl')[0]
    p=run.read(run.V13/f'pipeline/val/{run.key(c["sample_id"])}.json')
    extractor=object.__new__(run.Extractor)
    extractor.proposer=run.load_crossfit_proposer(run.V12/'proposers/full.pt','cpu')[0]
    views=[dict(feature_path=p['global_feature_path'],feature_sha256=p['global_feature_sha256'],transform=None)]+p['local_views']
    def replay(public,transform):
        v=next(v for v in views if v['transform']==transform)
        with np.load(v['feature_path'],allow_pickle=False) as z:
            meta=__import__('json').loads(str(z['metadata']))
            arrays={k:z[k].copy() for k in ('target_attention','target_hidden','image_hidden','vision_hidden')}
        return arrays,meta,{k:v[k] for k in ('feature_path','feature_sha256','transform')}
    extractor.view=replay
    monkeypatch.setattr(run,'OUT',tmp_path)
    got=extractor.case({k:c[k] for k in run.PUBLIC})
    assert got['pools']==p['pools']
    assert got['global_result']['all_points']==p['global_result']['all_points']
    assert [v['transform'] for v in got['local_views']]==[v['transform'] for v in p['local_views']]
    b=run.build_base(got,c['image_size'])
    old=next(e for e in run.old_entries('val') if e['sample_id']==c['sample_id'])
    with np.load(old['path'],allow_pickle=False) as z:
        for name in ('base','active','points','baseline'):np.testing.assert_array_equal(b[name],z[name])
    state=np.load(old['state_path'],allow_pickle=False);mask=np.load(old['mask_path'],allow_pickle=False)
    for j,v in enumerate(views):
        arrays,meta,_=replay(c,v['transform'])
        f,m=run.state_features(arrays,meta,b['points'],c['image_size'],v['transform'])
        np.testing.assert_array_equal(f,state[:,j,:f.shape[1]])
        np.testing.assert_array_equal(mask[:,j,:f.shape[1]],np.broadcast_to(m[:,None],f.shape[:2]))
