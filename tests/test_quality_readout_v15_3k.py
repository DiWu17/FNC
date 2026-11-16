"""Check selection, leakage barriers, authenticated cache reuse and raw replay."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import run_quality_readout_v15_3k as run


def test_selected_cohorts_are_balanced_nested_and_outcome_independent():
    rows=[dict(sample_id=f'{s}-{i}',dataset=s) for s in run.SOURCES for i in range(750)]
    large=run.select_subset(rows,700);small=run.select_subset(rows,30)
    assert len(large)==2100
    assert {r['sample_id'] for r in small}<={r['sample_id'] for r in large}
    assert all(sum(r['dataset']==s for r in large)==700 for s in run.SOURCES)
    changed=[dict(r,model_hit=False,n=0,cache_available=False) for r in reversed(rows)]
    assert [r['sample_id'] for r in run.select_subset(changed,700)]==[r['sample_id'] for r in large]
    with pytest.raises(ValueError,match='Insufficient'):run.select_subset(rows,751)


def test_exact_total_and_fixed_compute_budget():
    assert run.TOTAL==2973 and run.STEPS==2976
    assert run.SEEDS==(20261311,20261312,20261313)


def test_labels_rejected_before_any_extraction_or_lookup():
    extractor=object.__new__(run.Extractor);public={k:None for k in run.base.PUBLIC}
    with pytest.raises(ValueError,match='Labels in extraction input'):
        extractor.case(dict(public,target_geometry={}))
    with pytest.raises(ValueError,match='public fields only'):
        extractor.view(dict(public,target_geometry={}),None)


def test_reserve_extraction_requires_complete_freeze(tmp_path,monkeypatch):
    monkeypatch.setattr(run,'OUT',tmp_path)
    with pytest.raises(ValueError,match='Freeze all models'):run.extract('new_reserve')
    run.write(tmp_path/'models/frozen.json',dict(complete=False,models={}))
    with pytest.raises(ValueError,match='Incomplete model freeze'):run.extract('new_reserve')


def test_altered_cache_is_rejected(tmp_path):
    case=dict(sample_id='test')
    e=dict(case_identity=run.digest(case))
    for name,h in [('path','sha256'),('labels_path','labels_sha256'),('state_path','state_sha256'),('mask_path','mask_sha256'),('pipeline_path','pipeline_sha256')]:
        p=tmp_path/name;p.write_bytes(name.encode());e[name]=str(p);e[h]=run.sha(p)
    run.authenticate_entry(e,case)
    Path(e['state_path']).write_bytes(b'changed')
    with pytest.raises(ValueError,match='Cached feature drift'):run.authenticate_entry(e,case)


def test_seed_mean_bootstrap_keeps_fractional_successes():
    entries=[dict(source='same',group='a'),dict(source='same',group='b')]
    a=[[True,False],[False,False],[False,False]];b=[[False,False]]*3
    value=run.mean_seed_comparison(a,b,entries)
    assert value['delta']==pytest.approx(1/6)
    assert value['ci95'][1]==pytest.approx(1/3)


def test_raw_to_compact_replay_is_identical_to_existing_v15_cache():
    run.torch.set_num_threads(4)
    selected=run.base.rows(ROOT/'runs/quality_readout_v15_pilot/data/add90.jsonl')
    case=next(c for c in selected if run.read(run.base.OUT/f'features/new_train/cases/{run.key(c["sample_id"])}.json')['n']>0)
    e=run.read(run.base.OUT/f'features/new_train/cases/{run.key(case["sample_id"])}.json')
    pipe=run.read(e['pipeline_path']);b,labels,state,mask=run.compact(case,pipe,Path(e['pipeline_path']))
    with np.load(e['path'],allow_pickle=False) as z:
        for name in ('base','active','points','baseline'):np.testing.assert_array_equal(b[name],z[name])
    np.testing.assert_array_equal(labels,np.load(e['labels_path'],allow_pickle=False))
    np.testing.assert_array_equal(state,np.load(e['state_path'],allow_pickle=False))
    np.testing.assert_array_equal(mask,np.load(e['mask_path'],allow_pickle=False))


def test_pipeline_and_crop_planning_replay_legacy_without_new_forward(tmp_path,monkeypatch):
    run.torch.set_num_threads(4)
    c=run.base.rows(run.base.V13/'data/val.jsonl')[0]
    p=run.read(run.base.V13/f'pipeline/val/{run.key(c["sample_id"])}.json')
    extractor=object.__new__(run.Extractor)
    extractor.proposer=run.base.load_crossfit_proposer(run.base.V12/'proposers/full.pt','cpu')[0]
    views=[dict(feature_path=p['global_feature_path'],feature_sha256=p['global_feature_sha256'],transform=None)]+p['local_views']
    def replay(public,transform):
        v=next(v for v in views if v['transform']==transform)
        with np.load(v['feature_path'],allow_pickle=False) as z:
            meta=run.json.loads(str(z['metadata']));arrays={k:z[k].copy() for k in ('target_attention','target_hidden','image_hidden','vision_hidden')}
        return arrays,meta,{k:v[k] for k in ('feature_path','feature_sha256','transform')}
    extractor.view=replay;monkeypatch.setattr(run,'OUT',tmp_path)
    got,_=extractor.case({k:c[k] for k in run.base.PUBLIC})
    assert got['pools']==p['pools']
    assert got['global_result']['all_points']==p['global_result']['all_points']
    assert [v['transform'] for v in got['local_views']]==[v['transform'] for v in p['local_views']]
