import copy

import numpy as np
from PIL import Image
import torch

from clickagent.point_aligned_inference import bounded_box,proposal_crop,map_crop_point,marked_crop,yes_no_score,unique_points


def test_crop_geometry_edges_and_full_small_image():
    assert bounded_box((1000,600),(995,5),(256,256))==[744,0,1000,256]
    assert bounded_box((100,50),(10,20),(768,768))==[0,0,100,50]
    image=Image.new('RGB',(1000,600),'white')
    crop,t=proposal_crop(image,0,(20,30))
    assert crop.size==(512,512) and t['box']==[0,0,256,256]
    assert map_crop_point([200,100],t)==[100,50]
    assert map_crop_point(None,t) is None
    assert map_crop_point([10000,-100],t)==[999,0]


def test_marker_keeps_point_center_and_crop_transforms():
    image=Image.new('RGB',(1000,1000),(30,40,50))
    marked,t=marked_crop(image,[990,10])
    assert t['box']==[232,0,1000,768] and t['local_point']==[758,10]
    assert marked.getpixel(tuple(t['local_point']))==(30,40,50)
    assert marked.getpixel((746,10))==(255,0,0)


def test_yes_no_probability_and_absolute_mass():
    logits=torch.tensor([2.,1.,0.,8.]);r=yes_no_score(logits,{'Yes':[0,1],'No':[2]})
    expected=(np.exp(2)+np.exp(1))/(np.exp(2)+np.exp(1)+1)
    assert abs(r['score']-expected)<1e-6
    assert r['yes_no_mass']<.01 and not np.isnan(r['score'])


def test_pool_deduplicates_integer_points_preserving_fixed_order():
    a=[dict(point=[1,2],source='g'),dict(point=[3,4],source='g')]
    b=[dict(point=[1,2],source='c'),dict(point=[5,6],source='c')]
    before=copy.deepcopy((a,b));result=unique_points(a,b,limit=4)
    assert [r['point'] for r in result]==[[1,2],[3,4],[5,6]]
    assert result[0]['source']=='g' and (a,b)==before


def test_readout_prediction_does_not_need_ground_truth():
    import sys
    sys.path.insert(0,'scripts')
    from point_aligned_predict import predict_record
    from test_semantic_head_ranker import field_record
    from clickagent.candidate_ranker import SemanticCandidateRanker,RankerConfig
    from clickagent.point_aligned_readout import PointAlignedRanker
    from clickagent.semantic_head_ranker import SemanticHeadRankerConfig
    torch.set_num_threads(1)
    r=field_record();r['case']={'image_size':[813,517]};r['metadata']['target_text']='example'
    for name in list(r):
        if name.startswith('label_'):del r[name]
    fixed=SemanticCandidateRanker(RankerConfig(width=8),np.zeros(96)).eval()
    own=PointAlignedRanker(SemanticHeadRankerConfig(width=8,head_width=8),np.zeros(96),True).eval()
    models={'v5_fixed':(fixed,'v5'),'own':(own,'v8_own')}
    result=predict_record(r,models)
    assert result['own']['point'] is not None and not result['own']['ground_truth_in_readout_inputs']
    assert len(result['own']['all_points'])==len(r['parent_indices'])
