import numpy as np
import pytest
from PIL import Image
from clickagent.mvp_repair_v21 import make_pools,presentation_order,visual_inputs,combine_orders,valid_point


def native(points,anchor=None):
    return dict(point=points[0] if anchor is None else anchor,reference_result=dict(all_predictions=[dict(point=p) for p in points]))


def test_matching_candidate_count_and_mvp_anchor_for_all_budget_sizes():
    for n in [1,2,3]:
        out=make_pools(native([[10,10],[20,20],[90,90]][:n]),[[11,10],[12,10],[80,80]],[3,2,1],[100,100])
        assert len(out['native'])==len(out['augmented'])==n
        assert out['native'][0]==out['augmented'][0]==[10,10]
        if n>1:assert [11,10] in out['augmented']
        if n>2:assert [80,80] in out['augmented'] and [12,10] not in out['augmented']


def test_empty_ours_native_fill_and_invalid_mvp_retained():
    m=native([[10,10],[20,20],[90,90]])
    r=make_pools(m,[],[],[100,100]);assert r['native']==r['augmented']
    assert make_pools(dict(point=None),[[40,40]],[3],[100,100])['native']==[]
    assert make_pools(dict(point=[-1,20]),[[40,40]],[3],[100,100])['augmented']==[]
    assert not valid_point([np.nan,4],[100,100])


def test_native_duplicates_do_not_inflate_matched_budget():
    r=make_pools(native([[10,10],[10,10],[20,20]]),[[30,30],[70,70]],[1,2],[100,100])
    assert len(r['native'])==len(r['augmented'])==2
    assert r['augmented']==[[10,10],[70,70]]


def test_deterministic_order_reversal_and_candidate_alignment():
    pool=[[10,10],[20,20],[90,90]]
    order=presentation_order('id',3);assert order==presentation_order('id',3)
    reverse=presentation_order('id',3,True);assert reverse==order[::-1]
    canonical=np.array([1.,4.,2.,-2.])
    obs=[dict(order=o,scores=[canonical[j] for j in o]+[canonical[-1]]) for o in [order,reverse]]
    combined=combine_orders(pool,pool[0],obs)
    assert combined['point']==pool[1] and not combined['order_disagrees']
    assert not combined['fallback_to_mvp']


def test_N_fallback_is_identical_for_both_policies():
    for pool in [[[10,10],[20,20]],[[10,10],[90,90]]]:
        result=combine_orders(pool,pool[0],[dict(order=[0,1],scores=[1,2,8]),dict(order=[1,0],scores=[2,1,8])])
        assert result['fallback_to_mvp'] and result['point']==[10,10]


def test_visual_inputs_match_pixels_at_edges_without_annotation_geometry():
    image=Image.new('RGB',(1300,700),'white')
    pools=[[[0,0],[1299,699],[600,300]],[[0,0],[20,20],[1000,400]]]
    layouts=[]
    for pool in pools:
        images,messages,meta=visual_inputs(image,'Click Save',pool,[2,0,1])
        assert len(images)==4 and all(im.size==(512,512) for im in images)
        assert meta['input_pixels']==4*512*512
        assert 'Click Save' in messages[1]['content'][-1]['text']
        for crop in meta['crops']:
            x,y=crop['render']['marked_centers'][0]
            assert 0<=x<512 and 0<=y<512
        layouts.append([im.size for im in images])
    assert layouts[0]==layouts[1]
    with pytest.raises(TypeError):visual_inputs(image,'Click Save',pools[0],[0,1,2],target_geometry={})


def test_pool_selection_depends_on_scores_not_ground_truth_or_source():
    m=native([[10,10],[20,20],[90,90]])
    points=[[30,30],[70,70],[50,50]]
    a=make_pools(m,points,[3,2,1],[100,100]);b=make_pools(m,points,[1,2,3],[100,100])
    assert a['augmented']!=b['augmented']
    with pytest.raises(TypeError):make_pools(m,points,[1,2,3],[100,100],labels=[1,0,0])
