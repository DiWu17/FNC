import numpy as np
from clickagent.candidate_identity_v27 import (role_features,union_points,union_scores,retain,
    content_shuffle,probe_features,fit_residual,infer,DIM)


def test_exact_union_and_original_score_priority():
    p,ai,bi=union_points([[1,2],[3,4]],[[3,4],[5,6]])
    assert p.tolist()==[[1,2],[3,4],[5,6]]
    assert ai.tolist()==[0,1,-1] and bi.tolist()==[-1,0,1]
    assert union_scores([8,4],[20,3],ai,bi).tolist()==[8,4,3]
    p,ai,bi=union_points([],[])
    assert p.shape==(0,2) and len(union_scores([],[],ai,bi))==0
    x,b=probe_features(p,[100,100],[],ai,bi,np.zeros((0,3,48)),np.zeros((0,72)),np.zeros((0,3),bool),retain([],ai))
    assert x.shape==(0,DIM) and len(b)==0


def test_protected_budget_and_fill():
    ai=np.r_[np.arange(20),[-1]*20];s=np.arange(40,dtype=float)
    ix=retain(s,ai)
    assert len(ix)==16 and (ai[ix]>=0).sum()==8 and (ai[ix]<0).sum()==8
    assert len(retain([1,2,3],[0,1,-1]))==3
    assert retain(s,ai,protected=False).tolist()==list(range(24,40))


def test_fine_raster_and_crop_validity():
    # 1x1 coarse grid, four distinct fine vectors in raster order.
    v=np.eye(4,dtype=np.float32);a=np.ones((2,96,1),np.float32)
    pts=np.asarray([[10,10],[70,10],[10,70],[70,70],[110,10]])
    f,g,valid=role_features(a,v,[1,1],pts,[100,100])
    assert f.shape==(5,48) and g.shape==(5,72)
    assert valid.tolist()==[True,True,True,True,False]
    np.testing.assert_allclose(f,0,atol=1e-7)
    assert np.array_equal(g[4],np.zeros(72))
    _,_,valid=role_features(a,v,[1,1],pts,[200,200],{'box':[50,0,150,100]})
    assert valid.tolist()==[False,True,False,True,True]
    f,g,valid=role_features(a[:0],v,[1,1],pts,[100,100])
    assert not f.any() and not g.any() and not valid.any()


def test_label_free_feature_and_shuffle_contract():
    n=8;role=np.arange(n*3*48).reshape(n,3,48).astype(float)
    geom=np.zeros((n,72));valid=np.ones((n,3),bool);ai=np.r_[np.arange(4),[-1]*4]
    shuffled,_,vm,order=content_shuffle(role,geom,valid,ai,'id')
    np.testing.assert_array_equal(shuffled,role[order]);np.testing.assert_array_equal(vm,valid)
    assert set(order[:4])==set(range(4)) and set(order[4:])==set(range(4,8))
    p=np.c_[np.arange(n)*3,np.zeros(n)]
    x,b=probe_features(p,[100,100],np.arange(n),ai,np.arange(n),role,geom,valid,np.arange(n))
    assert x.shape==(n,DIM) and np.isfinite(x).all() and len(b)==n


def test_pairwise_fit_has_useful_supervision_and_zero_fallback():
    x=np.zeros((3,DIM));x[:,0]=[1,0,-1]
    rec=dict(x=x,base=np.zeros(3),y=np.asarray([True,False,False]),source='s')
    m=fit_residual([rec],.3);assert infer(m,rec).argmax()==0 and m['rankable_queries']==1
    empty=dict(rec,y=np.zeros(3,bool));m=fit_residual([empty],.3)
    assert not m['weight'].any() and m['rankable_queries']==0
