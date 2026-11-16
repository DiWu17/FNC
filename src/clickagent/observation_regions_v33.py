"""Label-free complementary regions from complete spatial evidence."""
import hashlib
import re
import numpy as np
from .grounded_composition_v28 import fixed_crop
from .reinspection_v31 import iou
from .target_token import _content_indices

ARMS=('nearby','name_fullpool','coordinate_mass','random_matched')

def grid_points(grid,size):
    h,w=map(int,grid[-2:]);yy,xx=np.mgrid[:h,:w]
    return np.c_[(xx.ravel()+.5)*size[0]/w,(yy.ravel()+.5)*size[1]/h]

def coordinate_distribution(arrays,meta):
    """Read actual own-forward p-value tokens; never retokenize the answer."""
    count=int(np.prod(meta['grid_shape']));missing=dict(indices=[],array_indices=[],available=False)
    if meta.get('point') is None:return np.zeros(count),missing
    match=re.search(r'<p>(\s*\d+\s*,\s*\d+\s*)</p>',meta['generated_text'])
    if match is None:return np.zeros(count),missing
    indices=_content_indices(meta['generated_prefixes'],match.start(1),match.end(1))
    attention=np.asarray(arrays['all_attention'],float)
    actual=[i for i in indices if 0<=i<len(attention)]
    if not actual:return np.zeros(count),missing
    headmaps=attention[np.asarray(actual)].mean(0)
    total=headmaps.sum(1,keepdims=True);valid=total[:,0]>1e-12
    if not valid.any():return np.zeros(count),missing
    probability=(headmaps[valid]/total[valid]).mean(0)
    assert len(probability)==count and np.isfinite(probability).all() and (probability>=0).all()
    return probability/probability.sum(),dict(indices=[i+1 for i in actual],array_indices=actual,available=True,heads=int(valid.sum()),generated_token_count=len(actual),index_convention='indices1based;array_indices0based')

def unique_support(points,scores):
    pp=np.asarray(points,float).reshape(-1,2);ss=np.asarray(scores,float).reshape(-1)
    assert len(pp)==len(ss) and np.isfinite(pp).all() and np.isfinite(ss).all()
    bypoint={}
    for point,score in zip(pp,ss):
        key=tuple(point);bypoint[key]=max(float(score),bypoint.get(key,-np.inf))
    if not bypoint:return np.empty((0,2)),np.empty(0)
    points=np.asarray(sorted(bypoint));scores=np.asarray([bypoint[tuple(q)] for q in points])
    z=(scores-scores.mean())/max(float(scores.std()),1e-6);weights=np.exp(np.clip(z-z.max(),-30,0));weights/=weights.sum()
    return points,weights

def box_mass(boxes,points,weights):
    boxes=np.asarray(boxes,float);points=np.asarray(points,float).reshape(-1,2);weights=np.asarray(weights,float)
    assert len(points)==len(weights)
    if not len(points):return np.zeros(len(boxes))
    values=[]
    for start in range(0,len(boxes),128):
        b=boxes[start:start+128]
        inside=(points[None,:,0]>=b[:,None,0])&(points[None,:,0]<b[:,None,2])&(points[None,:,1]>=b[:,None,1])&(points[None,:,1]<b[:,None,3])
        values.extend(inside@weights)
    return np.asarray(values)

def plan_regions(size,grid,first_crop,base_point,proposal,coordinate_weights,sample_id):
    """Same anchor set, geometry and overlap penalty for the two guided arms."""
    grid=grid_points(grid,size)
    support,weights=unique_support(proposal['all_points'],proposal['all_scores'])
    anchors=np.unique(np.concatenate([grid,support],axis=0),axis=0)
    transforms={}
    for point in anchors:
        tr=fixed_crop(point.tolist(),size);transforms.setdefault(tuple(tr['box']),tr)
    boxes=sorted(transforms);overlap=np.asarray([iou(box,first_crop['box']) for box in boxes])
    eligible=np.flatnonzero(overlap<=.25)
    if not len(eligible):eligible=np.flatnonzero(overlap==overlap.min())
    random_index=min(eligible,key=lambda j:hashlib.sha256((sample_id+'|v33_random|'+str(boxes[j])).encode()).hexdigest())
    def metadata(j,kind,mass=0.,fallback=False):
        return dict(transform=transforms[boxes[int(j)]],strategy=kind,support_mass=float(mass),first_crop_iou=float(overlap[j]),
            satisfies_iou_limit=bool(overlap[j]<=.25),eligible_boxes=len(eligible),all_anchor_boxes=len(boxes),
            original_support_points=len(proposal['all_points']),unique_support_points=len(support),fallback_to_matched_random=fallback,ground_truth_used=False)
    plans={'random_matched':metadata(random_index,'random_matched')}
    for kind,positions,massweights in [('name_fullpool',support,weights),('coordinate_mass',grid,np.asarray(coordinate_weights,float))]:
        if not len(massweights) or not (massweights>0).any():plans[kind]=metadata(random_index,kind,fallback=True);continue
        mass=box_mass(boxes,positions,massweights)
        j=max(eligible,key=lambda j:(float(mass[j]*(1-overlap[j])),-float(overlap[j]),-int(j)))
        plans[kind]=metadata(j,kind,mass[j])
    center=base_point if base_point is not None else [(first_crop['box'][0]+first_crop['box'][2])/2,(first_crop['box'][1]+first_crop['box'][3])/2]
    tr=fixed_crop(center,size)
    plans['nearby']=dict(transform=tr,strategy='nearby',first_crop_iou=iou(tr['box'],first_crop['box']),ground_truth_used=False)
    assert set(plans)==set(ARMS)
    return plans
