"""Cached attention-guided fine appearance and candidate competition probes.

All feature/retention functions are label-free. Container states are a weak
context anchor, not an executed arbitrary left-of/between relation predicate.
"""
from __future__ import annotations
import hashlib
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

ARMS=('prior','visual','pair_visual','full','shuffled')
RIDGE=(.03,.3,3.)
PRIOR=slice(0,10)
VISUAL=slice(10,106)
PEER=slice(106,202)
CONTEXT=slice(202,322)
DIM=322


def unit(x):
    x=np.asarray(x,np.float32)
    return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-8)


def group_heads(x):
    """Keep layer separation; average each four adjacent heads, 96->24."""
    assert x.shape[-1]==96
    return x.reshape(*x.shape[:-1],24,4).mean(-1)


def role_features(attention,vision,grid,points,size,transform=None):
    """Same-vision-space cosine to own-token attention-weighted templates.

    Each actual role token gets its own 96 templates BEFORE mean/min across
    tokens. Fine pixels stay on the native unmerged raster. No text-to-vision
    projection is learned or assumed. Uniform appearance is subtracted.
    """
    gh,gw=map(int,grid);v=np.asarray(vision,np.float32)
    assert v.ndim==2 and len(v)==4*gh*gw
    a=np.asarray(attention,np.float32);assert a.shape[1:]==(96,gh*gw)
    p=np.asarray(points,float).reshape(-1,2);n=len(p)
    out=np.zeros((n,48),np.float32);geom=np.zeros((n,72),np.float32)
    bounds=np.asarray([0,0,*size] if transform is None else transform['box'],float)
    xy=(p-bounds[:2])/(bounds[2:]-bounds[:2]);valid=((xy>=0)&(xy<1)).all(-1)
    if not len(a):return out,geom,np.zeros(n,bool)
    vf=unit(v).reshape(2*gh,2*gw,-1)
    coarse=vf.reshape(gh,2,gw,2,-1).mean((1,3)).reshape(gh*gw,-1)
    rel=a/np.maximum(a.sum(-1,keepdims=True),1e-12)
    proto=unit(rel@coarse)
    cell=np.clip(np.floor(xy*[2*gw,2*gh]).astype(int),[0,0],[2*gw-1,2*gh-1])
    center=vf[cell[:,1],cell[:,0]]
    uniform=unit(vf.reshape(-1,v.shape[-1]).mean(0,keepdims=True))[0]
    cosine=np.einsum('nd,thd->nth',center,proto,optimize=True)-(center@uniform)[:,None,None]
    out=np.concatenate([group_heads(cosine.mean(1)),group_heads(cosine.min(1))],-1)
    yy,xx=np.meshgrid((np.arange(gh)+.5)/gh,(np.arange(gw)+.5)/gw,indexing='ij')
    centers=rel.mean(0)@np.stack([xx.reshape(-1),yy.reshape(-1)],-1)
    delta=xy[:,None]-centers[None]
    geom=np.concatenate([group_heads(delta[...,0]),group_heads(delta[...,1]),
                         group_heads(np.linalg.norm(delta,axis=-1))],-1)
    out[~valid]=0;geom[~valid]=0
    assert np.isfinite(out).all() and np.isfinite(geom).all()
    return out.astype(np.float32),geom.astype(np.float32),valid


def union_points(a,b):
    points=[];lookup={};ai=[];bi=[]
    for kind,pp in enumerate([a,b]):
        for j,p in enumerate(np.asarray(pp).reshape(-1,2)):
            key=tuple(map(int,p))
            if key not in lookup:
                lookup[key]=len(points);points.append(list(key));ai.append(-1);bi.append(-1)
            (ai if kind==0 else bi)[lookup[key]]=j
    return np.asarray(points,np.int64).reshape(-1,2),np.asarray(ai,np.int64),np.asarray(bi,np.int64)


def union_scores(a,b,ai,bi):
    s=np.zeros(len(ai),np.float64)
    present=ai>=0;s[present]=np.asarray(a)[ai[present]]
    s[~present]=np.asarray(b)[bi[~present]]
    return s


def retain(scores,ai,budget=16,protected=True):
    scores=np.asarray(scores);ai=np.asarray(ai,np.int64)
    if not protected:return np.sort(np.argsort(-scores,kind='stable')[:budget])
    a=np.flatnonzero(ai>=0);b=np.flatnonzero(ai<0)
    a=a[np.argsort(-scores[a],kind='stable')];b=b[np.argsort(-scores[b],kind='stable')]
    chosen=list(a[:budget//2])+list(b[:budget-budget//2]);selected=set(chosen)
    for j in np.argsort(-scores,kind='stable'):
        if len(chosen)>=min(budget,len(scores)):break
        if int(j) not in selected:chosen.append(int(j));selected.add(int(j))
    return np.asarray(sorted(chosen),np.int64)


def content_shuffle(role,geometry,valid,ai,sample_id):
    """Within availability/origin strata; same features/missingness, wrong point."""
    groups={};order=np.arange(len(ai))
    for i in range(len(ai)):groups.setdefault((bool(ai[i]>=0),*valid[i].tolist()),[]).append(i)
    rng=np.random.default_rng(int(hashlib.sha256((sample_id+'v27_shuffle').encode()).hexdigest()[:16],16))
    for ids in groups.values():order[ids]=rng.permutation(ids)
    return role[order],geometry[order],valid.copy(),order


def probe_features(points,size,scores,ai,bi,role,geometry,valid,indices,arm='full'):
    assert arm in ARMS
    ix=np.asarray(indices);p=np.asarray(points)[ix];n=len(ix)
    s=np.asarray(scores,float);z=(s-s.mean())/max(float(s.std()),1e-6) if len(s) else s
    x=np.zeros((n,DIM),np.float64)
    x[:,0]=(ai[ix]<0);x[:,1]=(bi[ix]>=0)&(ai[ix]>=0)
    x[:,2]=np.log1p(len(s))/8;x[:,3]=np.log1p(n)/4
    x[:,4:7]=valid[ix];x[:,7]=z[ix]
    xy=p/np.asarray(size);dist=np.linalg.norm(xy[:,None]-xy[None],axis=-1)
    np.fill_diagonal(dist,np.inf)
    peer=np.argsort(dist,axis=1,kind='stable')[:,:min(3,max(0,n-1))]
    content=role[ix,:2].reshape(n,96)
    has=np.zeros(n,bool);contrast=np.zeros_like(content)
    if peer.shape[1]:
        close=dist[np.arange(n)[:,None],peer]<=.12
        has=close.any(1)
        other=(content[peer]*close[...,None]).sum(1)/np.maximum(close.sum(1)[:,None],1)
        contrast=np.where(has[:,None],content-other,0)
    x[:,8]=has;x[:,9]=np.minimum(dist.min(1),1) if n>1 else 1
    if arm!='prior':x[:,VISUAL]=content
    if arm in ('pair_visual','full','shuffled'):x[:,PEER]=contrast
    if arm in ('full','shuffled'):
        x[:,CONTEXT]=np.concatenate([role[ix,2],geometry[ix]],-1)
    assert np.isfinite(x).all()
    return x,z[ix]


def fit_residual(records,ridge):
    """Query-balanced positive-v-negative pairwise loss, training labels only."""
    xx=np.concatenate([r['x'] for r in records if len(r['x'])])
    mean=xx.mean(0);std=np.maximum(xx.std(0),.05)
    # Features constant over all fitting points have no identifying effect.
    std=np.where(xx.std(0)>1e-10,std,1.)
    dx=[];base=[];weights=[];sources=sorted({r['source'] for r in records})
    counts={src:sum(r['source']==src for r in records) for src in sources}
    for r in records:
        pos=np.flatnonzero(r['y']);neg=np.flatnonzero(~r['y'])
        if not len(pos) or not len(neg):continue
        pairs=np.asarray([(i,j) for i in pos for j in neg])
        d=(r['x'][pairs[:,0]]-r['x'][pairs[:,1]])/std
        dx.append(d);base.append(r['base'][pairs[:,0]]-r['base'][pairs[:,1]])
        weights.append(np.full(len(d),1/(len(sources)*counts[r['source']]*len(d))))
    if not dx:return dict(weight=np.zeros(DIM),mean=mean,std=std,iterations=0,rankable_queries=0,success=True)
    d=np.concatenate(dx);b=np.concatenate(base);w=np.concatenate(weights);w/=w.sum()
    def objective(theta):
        margins=b+d@theta;prob=expit(-margins)
        loss=float(np.dot(w,np.logaddexp(0,-margins))+.5*ridge*np.dot(theta,theta))
        grad=-(d.T@(w*prob))+ridge*theta
        return loss,grad
    result=minimize(objective,np.zeros(DIM),jac=True,method='L-BFGS-B',
                    options=dict(maxiter=200,ftol=1e-10,gtol=1e-6))
    assert np.isfinite(result.x).all() and np.isfinite(result.fun)
    gradient_max=float(np.abs(result.jac).max())
    assert result.success or gradient_max<1e-4, ('Residual optimizer did not converge',result.message,gradient_max)
    return dict(weight=result.x,mean=mean,std=std,iterations=int(result.nit),
                rankable_queries=len(dx),success=bool(result.success),loss=float(result.fun),
                gradient_max=gradient_max,message=str(result.message))


def infer(model,record):
    return record['base']+(record['x']-model['mean'])/model['std']@model['weight']
