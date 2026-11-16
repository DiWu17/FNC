import numpy as np
from PIL import Image
import torch

from clickagent.context_verifier import split_image_positions,diverse_crops,overlap_iou,LocalSpatialRecorder
from clickagent.conditioned_features import unmerge_raster
from clickagent.point_aligned_inference import map_crop_point
from clickagent.point_critic import PointUtilityCritic,point_utility_loss,FrozenPointEncoder


def test_multi_image_positions_preserve_local_keys():
    ids=np.array([1,99,99,2,99,99,99,3])
    result=split_image_positions(ids,99,[[1,2,4],[1,2,6]],2)
    assert [v.tolist() for v in result]==[[1,2],[4,5,6]]


def test_diverse_rois_keep_first_crop_and_separate_other_region():
    image=Image.new('RGB',(1600,900));c=[{'parent_index':i} for i in [105,106,130]]
    proposals=diverse_crops(image,c,(10,20))
    assert len(proposals)==2 and proposals[0][1]['parent_index']==105
    assert overlap_iou(proposals[0][1]['box'],proposals[1][1]['box'])<=.3
    for crop,t in proposals:
        assert map_crop_point([0,0],t)==t['box'][:2]


def test_point_loss_supports_multiple_valid_points_and_all_negative_queries():
    s=torch.tensor([2.,1.,-2.,-.5,.3],requires_grad=True);y=torch.tensor([1.,1.,0.,0.,0.])
    loss=point_utility_loss(s,y,[3,2]);loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(s.grad).all()
    changed=point_utility_loss(torch.tensor([3.,2.,-2.,-.5,.3]),y,[3,2])
    assert changed<loss


def test_actual_point_changes_features_within_same_parent():
    encoder=FrozenPointEncoder.__new__(FrozenPointEncoder)
    state=dict(query_last=np.ones(32),query_mean=np.ones(32),image=np.ones((4,32)),
        vision=np.arange(4*4*32).reshape(4,4,32).astype(float),spectra=np.ones((4,1152)),
        grid_shape=(2,2),image_size=(200,200))
    f=encoder.features(state,[[10,10],[80,80]])
    assert f.shape==(2,1484) and not np.array_equal(f[0],f[1])
    # Same image/point source yields exactly the same features.
    np.testing.assert_array_equal(f,encoder.features(state,[[10,10],[80,80]]))


def test_critic_normalization_and_finite_scores():
    model=PointUtilityCritic(np.zeros(5),np.zeros(5));output=model(torch.ones(3,5))
    assert output.shape==(3,) and torch.isfinite(output).all()


def test_local_visual_hook_slices_second_image_before_unmerging():
    recorder=LocalSpatialRecorder.__new__(LocalSpatialRecorder)
    recorder.vision_offset=8;recorder.grid_shape=(1,2)
    full=torch.arange(32).reshape(16,2).float()
    recorder._vision_hook(None,(full,))
    np.testing.assert_array_equal(recorder.vision_hidden,unmerge_raster(full[8:].half().numpy(),(1,2)))


def test_symmetric_choice_matches_points_after_order_reversal(monkeypatch):
    monkeypatch.syspath_prepend('scripts')
    from run_context_verifier import symmetric_choice
    pool=[{'point':[10,20]},{'point':[70,80]}]
    fwd=dict(order=[[10,20],[70,80]],scores=[3.,1.,-2.],point=[10,20])
    rev=dict(order=[[70,80],[10,20]],scores=[1.,3.,-2.],point=[10,20])
    result=symmetric_choice(fwd,rev,pool)
    assert result['point']==[10,20] and not result['order_disagrees']
    fwd['scores'][-1]=8.;rev['scores'][-1]=8.
    assert symmetric_choice(fwd,rev,pool)['point'] is None
    assert symmetric_choice(fwd,rev,pool)['forced_point']==[10,20]
