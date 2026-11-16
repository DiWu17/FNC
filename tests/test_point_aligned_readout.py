import copy
import numpy as np
import pytest
import torch

from clickagent.attention_field_ranker import make_field_batch
from clickagent.candidate_ranker import decode_ranker
from clickagent.semantic_head_ranker import SemanticHeadRankerConfig
from clickagent.point_aligned_readout import (PointAlignedRanker, multi_positive_point_loss,
    all_candidate_points, save_aligned, load_aligned, aligned_loss)
from test_semantic_head_ranker import field_record


@pytest.fixture(autouse=True)
def threads():
    old=torch.get_num_threads();torch.set_num_threads(1);yield;torch.set_num_threads(old)


def make(connect):
    return PointAlignedRanker(SemanticHeadRankerConfig(width=8,head_width=8),np.ones(96)/96,connect).eval()


def test_controls_have_identical_forward_and_differ_only_in_fine_gradient_path():
    torch.manual_seed(9);a=make(False);b=make(True);b.load_state_dict(a.state_dict())
    torch.nn.init.normal_(a.spatial_fine[-1].weight,std=.1);b.load_state_dict(a.state_dict())
    inputs,_=make_field_batch([field_record()], 'cpu')
    oa,ob=a(**inputs),b(**inputs)
    for k in ('scores','offsets','child_logits'):torch.testing.assert_close(oa[k],ob[k],rtol=0,atol=0)
    oa['child_logits'].sum().backward();ob['child_logits'].sum().backward()
    assert a.field_projection[0].weight.grad is None
    assert b.field_projection[0].weight.grad.abs().sum()>0
    assert a.spatial_fine[-1].weight.grad.abs().sum()>0


def test_zero_adapter_and_hook_cleanup():
    m=make(True);r=field_record();inputs,_=make_field_batch([r],'cpu')
    out=m(**inputs)
    torch.testing.assert_close(out['child_logits'],out['base_child_logits'],rtol=0,atol=0)
    torch.testing.assert_close(out['offsets'],out['base_offsets'],rtol=1e-6,atol=1e-7)
    assert len(m.field_projection._forward_hooks)==0
    assert out['routed_spatial_features'].shape==(len(r['parent_indices']),8)


def test_point_loss_rewards_any_valid_point_and_null_only_when_no_valid_point():
    scores=torch.tensor([0.,1.,2.,4.,-2.],requires_grad=True)
    output=dict(scores=scores,null_scores=torch.tensor([0.,1.],requires_grad=True),candidate_lengths=[3,2])
    labels=torch.tensor([1.,0.,1.,0.,0.],requires_grad=True)
    loss=multi_positive_point_loss(output,labels);loss.backward()
    assert labels.grad is None
    assert scores.grad[0]<0 and scores.grad[2]<0 and scores.grad[1]>0
    assert output['null_scores'].grad[1]<0 and scores.grad[3]>0


def test_vector_decoder_matches_existing_decoder_candidate_order():
    m=make(True);records=[field_record(seed=11),field_record(seed=12,tokens=2)]
    inputs,_=make_field_batch(records,'cpu');out=m(**inputs)
    points=all_candidate_points(out,records)
    for i,r in enumerate(records):
        old=decode_ranker(out,r,i,top_k=len(r['parent_indices']))
        for c in old:assert c['point']==points[i][c['candidate_index']].tolist()


def test_joint_loss_reaches_spatial_adapter_and_route(tmp_path):
    m=make(True);torch.nn.init.normal_(m.spatial_fine[-1].weight,std=.1)
    r=field_record();inputs,target=make_field_batch([r],'cpu');out=m(**inputs)
    loss,pieces=aligned_loss(out,target,torch.tensor([1.]+[0.]*(len(r['parent_indices'])-1)),True)
    loss.backward()
    assert m.field_projection[0].weight.grad.abs().sum()>0
    assert m.spatial_fine[-1].weight.grad.abs().sum()>0
    assert all(np.isfinite(v) for v in pieces.values())
    save_aligned(tmp_path/'m.pt',m,dict(synthetic=True));loaded,meta=load_aligned(tmp_path/'m.pt')
    assert meta['synthetic'] and loaded.connect_fine
    torch.testing.assert_close(loaded(**inputs)['scores'],m(**inputs)['scores'],rtol=0,atol=0)
