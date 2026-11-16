import inspect
import numpy as np
import pytest
import torch

from clickagent.evidence_retention import (BASE_DIM, STATE_DIM, RetentionModel, build_base,
    map_point, state_features, permute_evidence, select_indices, retention_loss)


def proposal(points):
    return dict(all_points=points,all_scores=list(range(len(points),0,-1)))


def test_universe_uses_every_existing_point_and_exact_coordinate_dedup():
    tr=dict(box=[10,10,90,90],scale=2,original_size=[100,100])
    pipe=dict(global_result=proposal([[15,15],[50,50],[60,60]]),
        local_views=[dict(result=proposal([[10,10],[20,20]]),transform=tr)],
        pools=dict(c3={'16':[dict(point=[15,15]),dict(point=[50,50]),dict(point=[20,20])]}))
    base=build_base(pipe,[100,100])
    assert base['points'].tolist()==[[15,15],[50,50],[60,60],[20,20]]
    assert base['baseline'].tolist()==[0,1,3]
    assert base['base'].shape==(4,BASE_DIM)
    assert base['base'][0,30]==.5
    assert 'target_geometry' not in inspect.signature(build_base).parameters
    assert 'target_geometry' not in inspect.signature(state_features).parameters


def test_matched_selection_has_fixed_count_stable_ties_and_canonical_order():
    assert select_indices(np.array([0,2,2,3]),2).tolist()==[1,3]
    assert select_indices(np.array([]),0).tolist()==[]
    with pytest.raises(ValueError):select_indices(np.array([1]),2)
    with pytest.raises(ValueError):select_indices(np.array([np.nan]),1)


def test_state_channels_keep_all_tokens_heads_and_original_crop_geometry():
    attention=np.zeros((2,96,4),np.float32);attention[0,:,0]=1;attention[1,:,3]=1
    arrays=dict(target_attention=attention,target_hidden=np.ones((2,12288),np.float16),image_hidden=np.ones((4,4096),np.float16))
    meta=dict(grid_shape=[2,2]);points=np.array([[10,10],[90,90],[50,50]])
    f,mask=state_features(arrays,meta,points,[100,100])
    assert f.shape==(3,2,STATE_DIM) and mask.all()
    assert (f[0,0,:96]>0).all() and (f[0,1,:96]==0).all()
    assert (f[1,1,:96]>0).all()
    tr=dict(box=[0,0,50,50],scale=2,original_size=[100,100])
    f,mask=state_features(arrays,meta,points,[100,100],tr)
    assert mask.tolist()==[True,False,False] and (f[~mask]==0).all()


def test_empty_name_creates_no_valid_internal_evidence():
    arrays=dict(target_attention=np.zeros((0,96,4)),target_hidden=np.zeros((0,12288)),image_hidden=np.ones((4,4096)))
    f,mask=state_features(arrays,dict(grid_shape=[2,2]),np.array([[5,5]]),[10,10])
    assert f.shape==(1,0,STATE_DIM) and not mask.any()


def test_permutation_preserves_each_view_mask_and_token_vectors():
    x=np.arange(5*4*2*STATE_DIM,dtype=np.float32).reshape(5,4,2,STATE_DIM)
    mask=np.ones((5,4,2),bool);mask[0,1]=False
    out=permute_evidence(x,mask,np.random.default_rng(3))
    assert np.array_equal(out[0,1],x[0,1])
    for v in range(4):
        ix=np.flatnonzero(mask[:,v].any(-1))
        assert sorted(map(tuple,out[ix,v].reshape(len(ix),-1)))==sorted(map(tuple,x[ix,v].reshape(len(ix),-1)))
    assert not np.array_equal(out,x)


@pytest.mark.parametrize('positive',[False,True])
def test_absent_views_have_finite_forward_and_gradients(positive):
    torch.manual_seed(1);m=RetentionModel()
    base=torch.randn(3,BASE_DIM);state=torch.zeros(3,4,2,STATE_DIM)
    mask=torch.zeros(3,4,2,dtype=torch.bool);mask[0,0]=True
    scores=m(base,state,mask)
    loss=retention_loss(scores,torch.tensor([float(positive),0,0]),m.null)
    loss.backward()
    assert torch.isfinite(scores).all() and torch.isfinite(loss)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())


def test_model_candidate_permutation_equivariance_and_rank_parameter_match():
    torch.manual_seed(3);m=RetentionModel().eval()
    x=torch.randn(5,BASE_DIM);active=torch.ones(5,4,dtype=torch.bool);order=torch.tensor([4,1,0,3,2])
    with torch.inference_mode():
        a=m(x,active=active);b=m(x[order],active=active[order])
    torch.testing.assert_close(b,a[order])
    assert sum(p.numel() for p in m.parameters())==sum(p.numel() for p in RetentionModel().parameters())
