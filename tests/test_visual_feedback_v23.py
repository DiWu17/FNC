import numpy as np
from PIL import Image
from clickagent.visual_feedback_v23 import (make_pool, visual_inputs, decode_coordinate,
                                           decode_selection, presentation)


def test_pool_keeps_argmax_and_prefers_diverse_points():
    points = [[100,100],[102,100],[500,400],[700,400]]
    assert make_pool(points,[4,3,2,1],(1000,800)) == [[100,100],[500,400],[700,400]]
    assert make_pool([],[],(1000,800)) == []


def test_views_have_identical_geometry_budget_and_marker_only_difference():
    image = Image.new('RGB',(1300,800),'white');pool=[[5,8],[900,700],[250,250]]
    a,ma,ga=visual_inputs(image,'target','case',pool,'local_relook')
    b,mb,gb=visual_inputs(image,'target','case',pool,'marked_refine')
    c,mc,gc=visual_inputs(image,'target','case',pool,'marked_verify')
    assert ma==mb  # local relook vs marked refine changes pixels only
    assert ga['crop_box']==gb['crop_box']==gc['crop_box']==[0,0,512,512]
    assert ga['order']==gb['order']==gc['order']
    assert ga['input_pixels']==gb['input_pixels']==gc['input_pixels']==524288
    assert all(im.size==(512,512) for im in a+b+c)
    assert not np.array_equal(np.asarray(a[0]),np.asarray(b[0]))
    assert all(np.array_equal(np.asarray(x),np.asarray(y)) for x,y in zip(b,c))
    for im in a+b+c+[image]:im.close()


def test_coordinates_use_original_image_and_require_complete_output():
    g=dict(pool=[[20,30]],shown=[[20,30]],labels=['A'],order=[0],original_size=[2001,1001])
    assert decode_coordinate('CLICK 500 250',True,g)['point']==[1000,250]
    assert decode_coordinate('CLICK 500 250',False,g)['point']==[20,30]
    for text in ['CLICK 1001 250','CLICK -1 2','CLICK 1 2 extra','(1,2)']:
        assert decode_coordinate(text,True,g)['fallback']


def test_keep_and_selection_follow_shuffled_labels_and_fallback():
    g=dict(pool=[[10,10],[40,40]],shown=[[40,40],[10,10]],labels=['A','B'],order=[1,0],original_size=[100,100])
    assert decode_coordinate('KEEP A',True,g)['point']==[40,40]
    assert decode_selection([3,2,1],g)['point']==[40,40]
    assert decode_selection([1,2,3],g)['point']==[10,10]
    assert decode_coordinate('WAIT',True,g)['point']==[10,10]


def test_ties_and_order_are_deterministic():
    pool=make_pool([[4,5],[50,60],[80,90]],[1,1,1],(100,100))
    assert pool[0]==[4,5]
    assert presentation('case',pool)==presentation('case',pool)


def test_small_image_and_edge_crop_preserve_bounds():
    image=Image.new('RGB',(90,40),'white')
    ims,_,g=visual_inputs(image,'target','small',[[89,39]],'marked_refine')
    assert g['crop_box']==[0,0,90,40]
    assert decode_coordinate('CLICK 1000 1000',True,g)['point']==[89,39]
    for im in ims+[image]:im.close()
