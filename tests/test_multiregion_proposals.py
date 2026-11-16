from copy import deepcopy

from PIL import Image
import pytest

from clickagent.candidate_budget import build_pools
from clickagent.multiregion_proposals import (
    box_iou, build_c1_c3_pools, build_multiregion_pools, crop_transform,
    plan_multiregion_crops, public_raw_identity, round_robin,
)
from clickagent.point_aligned_inference import proposal_crop


def proposal(points, parents=None, grid=(20, 20)):
    parents = list(range(len(points))) if parents is None else parents
    return dict(all_points=points, all_scores=[float(len(points)-i) for i in range(len(points))],
                parent_indices=parents, grid_shape=list(grid), candidates=[dict(parent_index=parents[0])] if parents else [])


@pytest.mark.parametrize("size,grid,parent", [((800, 600), (12,16),0), ((800,600),(12,16),191), ((317,199),(4,6),17)])
def test_transform_is_exact_v12_geometry(size, grid, parent):
    image = Image.new("RGB",size)
    crop, old = proposal_crop(image,parent,grid)
    assert crop_transform(size,parent,grid) == old
    crop.close(); image.close()


def test_stable_diverse_regions_reject_overlap_with_every_previous_box():
    result = proposal([[25,25],[75,25],[975,25],[25,975],[975,975]], [0,1,19,380,399])
    result["target_geometry"] = object()  # irrelevant; no GT API argument exists
    selected = plan_multiregion_crops(result,[1000,1000])
    assert [r["transform"]["parent_index"] for r in selected] == [0,19,380]
    assert [r["global_source_rank"] for r in selected] == [1,3,4]
    assert all(box_iou(a["transform"]["box"],b["transform"]["box"]) <= .25
               for i,a in enumerate(selected) for b in selected[i+1:])
    assert len(plan_multiregion_crops(result,[100,100])) == 1


def test_iou_boundary_and_invalid_boxes():
    assert box_iou([0,0,10,10],[6,0,16,10]) == .25
    assert box_iou([0,0,10,10],[10,0,20,10]) == 0
    with pytest.raises(ValueError): box_iou([0,0,0,10],[0,0,1,1])


def test_round_robin_skips_empty_views_but_does_not_deduplicate():
    assert round_robin([["a","b","c"],[],["a","d"]],4) == ["a","a","b","d"]
    assert round_robin([[],[]]) == []
    assert round_robin([[1],[2]],0) == []


def test_c1_matches_v12_and_c3_is_nested_with_fixed_quota_before_dedup():
    global_result = proposal([[i+1,10] for i in range(20)])
    transform = dict(box=[0,0,100,100],scale=1,original_size=[1000,1000],crop_size=[100,100])
    local0 = proposal([[1,10]]+[[30+i,10] for i in range(19)])
    local1 = proposal([[60+i,10] for i in range(20)])
    local2 = proposal([[80+i,10] for i in range(20)])
    views=[dict(result=r,transform=deepcopy(transform)) for r in (local0,local1,local2)]
    pools=build_c1_c3_pools(global_result,views)
    assert pools['c1'] == build_pools(dict(global_result=global_result,old_crop_result=local0,old_crop_transform=transform))
    # K4 = global1/global2 + view0/rank1 duplicate + view1/rank1: exactly3.
    assert [v['point'] for v in pools['c3']['4']] == [[1,10],[2,10],[60,10]]
    previous=set()
    for k in ('4','8','16','32'):
        current={tuple(x['point']) for x in pools['c3'][k]}
        assert previous <= current and len(current) <= int(k)
        previous=current
    assert [x['source'] for x in pools['c3']['8']][-3:] == ['local_1','local_2','old_crop']


def test_no_name_means_no_crops_and_empty_pools():
    empty = proposal([],[])
    assert plan_multiregion_crops(empty,[1000,1000]) == []
    assert build_c1_c3_pools(empty,[]) == {c:{str(k):[] for k in (4,8,16,32)} for c in ('c1','c3')}


def test_raw_identity_binds_every_input_and_runtime():
    base=('sid','image_sha','click target',dict(box=[0,0,100,100],scale=2),'runtime_sha')
    value=public_raw_identity(*base)
    assert value==public_raw_identity(*base)
    for index,new in enumerate(('other_sid','other_image','different target',dict(box=[1,0,101,100],scale=2),'other_runtime')):
        changed=list(base);changed[index]=new
        assert public_raw_identity(*changed)!=value
