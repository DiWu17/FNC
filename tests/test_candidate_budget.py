import numpy as np
import pytest

from clickagent.candidate_budget import build_pools, ranked_points, choose_point, coordinate_key


def result(points, scores):
    return dict(all_points=points, all_scores=scores, parent_indices=list(range(len(points))))


def test_nested_source_quotas_preserve_two_plus_two_and_stable_rank_ties():
    g=result([[i,1] for i in range(20)],[0,0]+list(range(-1,-19,-1)))
    local=result([[i,30] for i in range(20)],list(range(20,0,-1)))
    transform=dict(box=[0,0,100,100],scale=1,original_size=[100,100])
    pools=build_pools(dict(global_result=g,old_crop_result=local,old_crop_transform=transform))
    assert [v['point'] for v in pools['4']]==[[0,1],[1,1],[0,30],[1,30]]
    for a,b in zip(['4','8','16'],['8','16','32']):
        assert {tuple(v['point']) for v in pools[a]} < {tuple(v['point']) for v in pools[b]}
    assert [sum(v['source']==s for v in pools['32']) for s in ['global','old_crop']]==[16,16]


def test_dedup_is_after_quota_without_backfill():
    r=result([[1,1],[2,2],[3,3]],[3,2,1])
    t=dict(box=[0,0,100,100],scale=1,original_size=[100,100])
    pools=build_pools(dict(global_result=r,old_crop_result=r,old_crop_transform=t))
    assert [v['point'] for v in pools['4']]==[[1,1],[2,2]]
    assert len(pools['8'])==3


def test_local_points_map_before_cross_source_dedup():
    r=result([[20,40],[40,60]],[2,1])
    t=dict(box=[10,20,60,70],scale=2,original_size=[100,100])
    items=ranked_points(r,'old_crop',t)
    assert items[0]['point']==[20,40] and items[1]['point']==[30,50]
    assert items[0]['local_point']==[20,40]


def test_shared_scores_can_reduce_accuracy_when_a_new_wrong_point_wins():
    g=result([[10,10],[20,20],[90,90]],[3,2,1])
    pools=build_pools(dict(global_result=g,old_crop_result=None,old_crop_transform=None))
    values={coordinate_key(v):dict(log_odds=s) for v,s in [([10,10],2),([20,20],1),([90,90],3)]}
    assert choose_point(pools['4'],values)['point']==[10,10]
    assert choose_point(pools['8'],values)['point']==[90,90]


def test_empty_and_nonfinite_inputs_are_explicit():
    assert choose_point([], {})['point'] is None
    assert all(not p for p in build_pools(dict(global_result={},old_crop_result=None)).values())
    with pytest.raises(ValueError):ranked_points(result([[1,2]],[np.nan]),'global')
    with pytest.raises(ValueError):ranked_points(result([[1.5,2]],[1]),'global')


def test_selection_has_stable_ties_without_annotation_access():
    pool=[dict(point=[1,2],source='global',source_rank=1),dict(point=[2,3],source='old_crop',source_rank=1)]
    records={'1,2':dict(log_odds=2),'2,3':dict(log_odds=2)}
    r=choose_point(pool,records)
    assert r['point']==[1,2] and r['margin']==0 and r['exact_top_ties']==2
