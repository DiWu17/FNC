"""CPU fake-runtime checks: caching only raw arrays, always rerun learned heads."""
import importlib.util
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from clickagent.multiregion_proposals import build_c1_c3_pools, plan_multiregion_crops


@pytest.fixture
def runner(monkeypatch, tmp_path):
    path = Path(__file__).resolve().parents[1]/"scripts/extract_multiregion_v13.py"
    spec = importlib.util.spec_from_file_location("multiregion_test_runner",path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    monkeypatch.setattr(module,"OUT",tmp_path/"v13")
    return module


def result(points,parents):
    return dict(point=points[0] if points else None,all_points=points,all_scores=[float(len(points)-i) for i in range(len(points))],
        parent_indices=parents,grid_shape=[20,20],candidates=[dict(parent_index=parents[0])] if parents else [])


def save_raw(path,meta):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("wb") as stream:
        np.savez(stream,metadata=np.asarray(json.dumps(meta)),**{k:np.zeros((1,),np.float16) for k in
            ("target_attention","target_hidden","image_hidden","vision_hidden")})


def fixture(runner,tmp_path,monkeypatch,local_has_name):
    image_path=tmp_path/"image.png";Image.new("RGB",(1000,1000)).save(image_path)
    case=dict(sample_id="sid",dataset="mock",instruction="Click settings",image=str(image_path),
        image_size=[1000,1000],image_sha256=runner.sha(image_path),image_fold=0,oof_provider="fold_0")
    global_result=result([[25,25],[975,25],[25,975]],[0,19,380])
    local_result=result([[20,20],[40,40]],[0,1]) if local_has_name else result([],[])
    ga=tmp_path/"global.npz"
    meta=dict(sample_id="sid",image_sha256=case['image_sha256'],image_size=[1000,1000],
        own_token_forward=True,all_backbone_parameters_frozen=True,ground_truth_in_model_input=False,
        instruction_sha256=runner.hashlib.sha256(case['instruction'].encode()).hexdigest(),kind="global")
    save_raw(ga,meta)
    calls=[]
    def predict(model,arrays,metadata,size,topk=64):
        calls.append(metadata['kind'])
        return global_result if metadata['kind']=='global' else local_result
    monkeypatch.setattr(runner,"predict_raw",predict)
    first=plan_multiregion_crops(global_result,case['image_size'])[0]['transform']
    old=dict(global_result=global_result,pools=build_c1_c3_pools(global_result,[dict(result=local_result,transform=first)])['c1'])
    class Frozen:
        models={"fold_0":None};model_sha={"fold_0":"model_sha"};runtime_identity_sha="runtime_sha"
        identity={"runtime_contract_sha256":"contract_sha"}
        def role(self,case,split):return "fold_0"
        def global_path(self,case,split):return ga
        def pipeline(self,case,group):return old
        def local_match(self,case,transform):return None
    return case,meta,global_result,local_result,Frozen(),calls


def test_no_name_local_raw_is_preserved_and_learned_outputs_rerun(runner,tmp_path,monkeypatch):
    case,meta,ga,la,frozen,calls=fixture(runner,tmp_path,monkeypatch,False)
    class Runtime:
        def __init__(self):self.calls=0
        def infer_spatial_features(self,image,instruction):
            assert instruction==case['instruction'];self.calls+=1
            local_meta=dict(meta,kind='local',image_size=list(image.size))
            return {k:np.zeros((1,),np.float16) for k in runner.RAW_ARRAY_KEYS},local_meta
    active=Runtime()
    pipeline=runner.extract_case(case,'train',frozen,lambda:active)
    assert active.calls==3 and pipeline['current_local_raw_forward_count']==3
    assert len(pipeline['local_views'])==3
    assert all(Path(v['feature_path']).exists() and not v['result']['all_points'] for v in pipeline['local_views'])
    assert calls==['global','local','local','local']
    assert all(not v['raw_reused'] and v['learned_output_recomputed'] for v in pipeline['local_views'])
    # A completed resume validates artifacts without running any model forward.
    assert runner.extract_case(case,'train',frozen,lambda:active)==pipeline
    assert active.calls==3


def test_authenticated_raw_reuse_never_copies_old_learned_result(runner,tmp_path,monkeypatch):
    case,meta,ga,la,frozen,calls=fixture(runner,tmp_path,monkeypatch,True)
    old_paths={}
    def match(case,transform):
        path=tmp_path/(str(transform['parent_index'])+'.npz')
        save_raw(path,dict(meta,kind='local',image_size=transform['crop_size'],transform=transform))
        old_paths[str(path)]=runner.sha(path)
        return dict(feature_path=str(path),feature_sha256=runner.sha(path),source='v12_authenticated_raw')
    frozen.local_match=match
    def forbidden_runtime():raise AssertionError('Raw reuse must not start Qwen')
    pipeline=runner.extract_case(case,'train',frozen,forbidden_runtime)
    assert pipeline['current_local_raw_forward_count']==0
    assert calls==['global','local','local','local']
    assert all(v['raw_reused'] and v['current_raw_forward_wall_seconds'] is None for v in pipeline['local_views'])
    assert all(runner.sha(p)==s for p,s in old_paths.items())
    assert len(pipeline['pools']['c1']['16'])<=16 and len(pipeline['pools']['c3']['16'])<=16


def test_fresh_test_requires_scorer_freeze(runner):
    with pytest.raises(FileNotFoundError):runner.assert_test_frozen()
