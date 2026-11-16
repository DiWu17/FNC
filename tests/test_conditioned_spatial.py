import numpy as np
import pytest

from clickagent.spatial_geometry import spatial_supervision,decode_candidates
from clickagent.public_evaluation import point_in_geometry


def test_tiny_target_without_a_center_still_supervises_offsets():
    geometry={'type':'bbox','boxes':[[.09,.09,.11,.11]]}
    weights,offsets=spatial_supervision(geometry,(2,2))
    assert weights.tolist()==[[1.,0.],[0.,0.]]
    assert np.allclose(offsets[:,0,0],[.2,.2])
    proposal=decode_candidates([[10,0],[0,0]],offsets,(2,2),(100,100))[0]
    assert proposal['point']==[10,10]
    assert point_in_geometry(proposal['normalized_point'],geometry)
    center=decode_candidates([[10,0],[0,0]],offsets,(2,2),(100,100),center_only=True)[0]
    assert not point_in_geometry(center['normalized_point'],geometry)


def test_concave_polygon_labels_always_provide_a_valid_point():
    geometry={'type':'polygon','polygons':[[[.01,.01],[.8,.01],[.8,.1],[.1,.1],[.1,.8],[.01,.8]]]}
    weights,offsets=spatial_supervision(geometry,(4,4))
    assert weights.sum()==pytest.approx(1.)
    for row,col in zip(*np.nonzero(weights)):
        x,y=offsets[:,row,col]
        assert point_in_geometry([(col+x)/4,(row+y)/4],geometry)


def test_candidate_points_are_bounded_and_region_windows_are_diverse():
    values=np.arange(16).reshape(4,4)
    proposals=decode_candidates(values,np.full((2,4,4),.99),(4,4),(101,99),top_k=5)
    assert proposals[0]['patch_index']==15
    assert len({r['patch_index'] for r in proposals})==len(proposals)
    assert all(0<=r['point'][0]<101 and 0<=r['point'][1]<99 for r in proposals)


def make_inputs(variant='full'):
    import torch
    from clickagent.conditioned_grounder import ConditionedSpatialGrounder,SpatialConfig
    torch.manual_seed(11)
    config=SpatialConfig(image_dim=16,vision_dim=8,query_dim=24,query_layers=3,heads=4,width=16,variant=variant)
    model=ConditionedSpatialGrounder(config)
    attention=torch.rand(2,3,4,6)*.02
    query=torch.randn(2,3,24,requires_grad=True);image=torch.randn(2,6,16,requires_grad=True)
    vision=torch.randn(2,24,8,requires_grad=True)
    mask=torch.tensor([[True,True,False],[True,True,True]])
    return model,{'attention':attention,'query_hidden':query,'image_hidden':image,'vision_hidden':vision,'token_mask':mask,'grid_shape':[2,3]}


def test_semantic_and_image_features_have_a_trainable_path_to_spatial_scores():
    import torch
    model,inputs=make_inputs()
    with torch.no_grad():model.score_delta.weight.normal_(0,.05)
    output=model(**inputs)
    assert output['logits'].shape==(2,4,6) and output['offsets'].shape==(2,2,4,6)
    assert output['token_weights'][0,2]==0
    output['logits'][:,0,0].sum().backward()
    assert inputs['query_hidden'].grad.abs().sum()>0
    assert inputs['image_hidden'].grad.abs().sum()>0
    assert inputs['vision_hidden'].grad.abs().sum()>0


def test_no_mass_ablation_is_invariant_to_per_head_mass_changes():
    import torch
    model,inputs=make_inputs('no_mass');model.eval()
    with torch.no_grad():
        model.score_delta.weight.normal_(0,.05);model.head_delta[-1].weight.normal_(0,.02)
    first=model(**inputs)
    altered=dict(inputs,attention=inputs['attention']*torch.tensor([.2,.5,2.,3.])[None,None,:,None])
    second=model(**altered)
    assert not torch.allclose(first['head_mass'],second['head_mass'])
    assert torch.allclose(first['logits'],second['logits'],atol=2e-5)


def test_full_readout_can_use_absolute_mass_without_changing_spatial_shape():
    import torch
    model,inputs=make_inputs();model.eval()
    with torch.no_grad():
        model.score_delta.weight.normal_(0,.05);model.head_delta[-1].weight.normal_(0,.02)
    first=model(**inputs)
    altered=dict(inputs,attention=inputs['attention']*torch.tensor([.2,.5,2.,3.])[None,None,:,None])
    second=model(**altered)
    assert not torch.allclose(first['logits'],second['logits'],atol=1e-5)


def test_empty_names_are_failures_and_labels_do_not_enter_forward():
    model,inputs=make_inputs()
    import torch
    with pytest.raises(ValueError,match='empty target'):
        model(**dict(inputs,token_mask=torch.zeros_like(inputs['token_mask'])))
    with pytest.raises(TypeError):model(**inputs,target_geometry={'type':'bbox'})


def test_trainable_spatial_loss_and_checkpoint_round_trip(tmp_path):
    import torch
    from clickagent.conditioned_grounder import spatial_loss,save_spatial_checkpoint,load_spatial_checkpoint
    model,inputs=make_inputs()
    weights,offsets=spatial_supervision({'type':'bbox','boxes':[[.20,.20,.22,.22]]},(4,6))
    labels={'weights':torch.from_numpy(np.stack([weights]*2)),
        'offsets':torch.from_numpy(np.stack([offsets]*2))}
    threads=torch.get_num_threads();torch.set_num_threads(2)
    try:
        optimizer=torch.optim.AdamW(model.parameters(),lr=.01)
        initial=float(spatial_loss(model(**inputs),**labels)[0].detach())
        for _ in range(40):
            optimizer.zero_grad();loss,_=spatial_loss(model(**inputs),**labels)
            loss.backward();optimizer.step()
        model.eval()
        with torch.no_grad():
            expected=model(**inputs)
            final=float(spatial_loss(expected,**labels)[0])
        assert final<initial*.35
        assert expected['logits'].flatten(1).argmax(-1).tolist()==[1,1]
        checkpoint=tmp_path/'spatial.pt'
        save_spatial_checkpoint(checkpoint,model,{'training_steps':40})
        restored,metadata=load_spatial_checkpoint(checkpoint)
        assert metadata['training_steps']==40
        with torch.no_grad():
            actual=restored(**inputs)
            torch.testing.assert_close(actual['logits'],expected['logits'])
            torch.testing.assert_close(actual['offsets'],expected['offsets'])
    finally:torch.set_num_threads(threads)


def test_static_head_control_uses_only_first_token_attention_and_coarse_centers():
    import torch
    from clickagent.conditioned_grounder import spatial_loss
    model,inputs=make_inputs('static_96');model.eval()
    assert sum(p.numel() for p in model.parameters())==4
    output=model(**inputs)
    altered=dict(inputs,query_hidden=inputs['query_hidden']*17,image_hidden=inputs['image_hidden']*-3,
        vision_hidden=inputs['vision_hidden']*5)
    changed=inputs['attention'].clone();changed[:,1:]*=.01;altered['attention']=changed
    actual=model(**altered)
    assert output['grid_shape']==[2,3]
    torch.testing.assert_close(output['logits'],actual['logits'])
    assert torch.all(output['offsets']==.5)
    weights,offsets=spatial_supervision({'type':'bbox','boxes':[[.20,.20,.22,.22]]},(4,6))
    labels={'weights':torch.from_numpy(np.stack([weights]*2)),
        'offsets':torch.from_numpy(np.stack([offsets]*2))}
    loss,details=spatial_loss(output,**labels);assert details['offset_loss']==0
    expected=torch.nn.functional.cross_entropy(output['logits'].flatten(1),torch.tensor([0,0]))
    torch.testing.assert_close(loss,expected)
    loss.backward();assert model.base_head_weights.grad.abs().sum()>0


def test_no_attention_ablation_removes_all_explicit_attention_channels():
    import torch
    model,inputs=make_inputs('no_attention');model.eval()
    with torch.no_grad():
        model.score_delta.weight.normal_(0,.05);model.head_delta[-1].weight.normal_(0,.02)
        first=model(**inputs)
        altered=dict(inputs,attention=torch.rand_like(inputs['attention'])*.001)
        second=model(**altered)
    assert not torch.allclose(first['head_mass'],second['head_mass'])
    torch.testing.assert_close(first['logits'],second['logits'])
    torch.testing.assert_close(first['offsets'],second['offsets'])
