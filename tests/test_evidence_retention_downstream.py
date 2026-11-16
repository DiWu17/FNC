import importlib.util
from pathlib import Path
import numpy as np
import pytest

SCRIPT=Path(__file__).resolve().parents[1]/'scripts/evaluate_evidence_retention_v14.py'
spec=importlib.util.spec_from_file_location('retention_downstream',SCRIPT)
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_worker_job_excludes_all_gt_and_analysis_fields():
    case=dict(sample_id='case',image='image.png',image_sha256='sha',instruction='click',
        target_geometry=object(),labels=object(),coverage=True)
    job=module.public_job(case,np.array([[1,2],[3,4]]))
    assert set(job)=={'sample_id','image','image_sha256','instruction','points','split'}
    assert job['points']==[[1,2],[3,4]]


def test_common_gui_scores_stable_ties_and_subsets():
    scores={(1,2):2.,(3,4):2.,(5,6):5.}
    assert module.choose_gui([[1,2],[3,4]],scores)==[1,2]
    assert module.choose_gui([[3,4],[1,2]],scores)==[3,4]
    assert module.choose_gui([[1,2],[5,6]],scores)==[5,6]
    assert module.choose_gui([],scores) is None
    with pytest.raises(ValueError):module.choose_gui([[7,8]],scores)
    with pytest.raises(ValueError):module.choose_gui([[1,2]],{(1,2):np.nan})


def test_all_downstream_pools_share_coordinate_universe_order():
    row=dict(points=[[1,1],[2,2],[3,3]],pools=dict(baseline=[2,0],new=[0,2]))
    assert module.policy_points(row,'baseline')==module.policy_points(row,'new')==[[1,1],[3,3]]


def test_pairing_preserves_gains_and_losses_instead_of_only_net_count():
    result=module.paired([True,True,False,False],[True,False,True,False],[],bootstrap=False)
    assert result['method_hits']==2 and result['reference_hits']==2
    assert result['wins']==1 and result['losses']==1 and result['delta']==0


def test_point_metric_retains_all_valid_regions_and_empty_output_failure():
    case=dict(image_size=[100,100],target_geometry=dict(type='bbox',boxes=[[.1,.1,.2,.2],[.8,.8,.9,.9]]))
    assert module.hit([10,20],case) and module.hit([85,85],case)
    assert not module.hit([50,50],case) and not module.hit(None,case)


def test_windows_sharing_retry_preserves_payload(monkeypatch,tmp_path):
    calls=[];payload={'scores':[1,2,3]}
    def writer(path,value):
        calls.append((path,value))
        if len(calls)<3:raise PermissionError('simulated transient sharing lock')
        return 'written'
    monkeypatch.setattr(module,'original_write',writer)
    monkeypatch.setattr(module.time,'sleep',lambda _:None)
    assert module.write(tmp_path/'result.json',payload)=='written'
    assert len(calls)==3 and all(value is payload for _,value in calls)


def test_permanent_write_failure_is_bounded(monkeypatch,tmp_path):
    count=[]
    def writer(path,value):
        count.append(1);raise PermissionError('permanent')
    monkeypatch.setattr(module,'original_write',writer)
    monkeypatch.setattr(module.time,'sleep',lambda _:None)
    with pytest.raises(PermissionError):module.write(tmp_path/'result.json',{})
    assert len(count)==20
