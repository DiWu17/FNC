"""Area-overlap supervision with valid within-cell continuous coordinates."""
from __future__ import annotations

import numpy as np


def shape_from_geometry(geometry):
    from shapely.geometry import box,Polygon
    from shapely.ops import unary_union
    if geometry['type']=='bbox':parts=[box(*b) for b in geometry['boxes']]
    else:parts=[Polygon(p) for p in geometry['polygons']]
    shape=unary_union(parts)
    if not shape.is_valid:shape=shape.buffer(0)
    if shape.is_empty or shape.area<=0:raise ValueError('target must have positive area')
    return shape


def spatial_supervision(geometry,grid_shape):
    """All intersecting cells supervise a location, including sub-cell targets.

    Classification labels are normalized intersection areas. Each positive cell
    gets a valid representative point of target AND cell intersection. There is
    no requirement for an existing grid center to hit. These fields are labels
    used by training only and are never inputs to the frozen model/readout.
    """
    from shapely.geometry import box
    gh,gw=map(int,grid_shape)
    if gh<1 or gw<1:raise ValueError('invalid grid')
    target=shape_from_geometry(geometry)
    weights=np.zeros((gh,gw),np.float32);offsets=np.zeros((2,gh,gw),np.float32)
    for row in range(gh):
        for col in range(gw):
            region=target.intersection(box(col/gw,row/gh,(col+1)/gw,(row+1)/gh))
            if not region.is_empty and region.area>0:
                point=region.representative_point()
                weights[row,col]=region.area
                offsets[:,row,col]=[point.x*gw-col,point.y*gh-row]
    if weights.sum()<=0:raise ValueError('target has no positive-area grid intersection')
    weights/=weights.sum()
    return weights,offsets


def decode_candidates(logits,offsets,grid_shape,image_size,top_k=5,center_only=False):
    """Rank diverse candidate regions, each with a learned continuous point.

    Regions are coarse proposal windows, not predicted element boundaries.
    Their normalized rectangle includes one neighboring cell in each direction.
    The score is a spatial proposal probability, not calibrated click safety.
    """
    gh,gw=map(int,grid_shape);w,h=map(int,image_size)
    values=np.asarray(logits,dtype=np.float64).reshape(gh,gw)
    offsets=np.asarray(offsets,dtype=np.float64).reshape(2,gh,gw)
    if not np.isfinite(values).all() or not np.isfinite(offsets).all():raise ValueError('nonfinite spatial output')
    probs=np.exp(values-values.max());probs/=probs.sum()
    order=np.argsort(-values.ravel(),kind='stable');chosen=[];result=[]
    for index in order:
        row,col=divmod(int(index),gw)
        if any(max(abs(row-a),abs(col-b))<=1 for a,b in chosen):continue
        chosen.append((row,col))
        delta=np.full(2,.5) if center_only else np.clip(offsets[:,row,col],0,1)
        xy=[(col+float(delta[0]))/gw,(row+float(delta[1]))/gh]
        point=[int(np.clip(np.rint(xy[0]*w),0,w-1)),int(np.clip(np.rint(xy[1]*h),0,h-1))]
        result.append({'point':point,'normalized_point':[point[0]/w,point[1]/h],
            'region':[max(0,col-1)/gw,max(0,row-1)/gh,min(gw,col+2)/gw,min(gh,row+2)/gh],
            'cell':[col/gw,row/gh,(col+1)/gw,(row+1)/gh],'patch_index':int(index),
            'offset':delta.tolist(),'spatial_probability':float(probs[row,col])})
        if len(result)>=top_k:break
    return result
