"""V28: centered visual readout and explicit, grounded spatial predicates.

Inference routines accept public observations only. Annotation-dependent mining
and supervision live in the runner, outside feature generation.
"""
from __future__ import annotations
import re
import time
import hashlib
import numpy as np
from scipy.special import expit
from scipy.optimize import minimize
from .conditioned_features import ConditionedFeatureRuntime, SpatialContentRecorder
from .model_features import _sync_cuda
from .target_token import _content_indices

GLOBAL_PROMPT = (
    "Locate the exact GUI target requested in the screenshot. Reply only with this complete XML: "
    "<g><t>target visible name</t><p>x,y</p><a1>reference name</a1><b1>x1,y1,x2,y2</b1>"
    "<r1>relation</r1><a2>reference name</a2><b2>x1,y1,x2,y2</b2><r2>relation</r2></g>. "
    "All coordinates are integers 0..1000 relative to the displayed image. p is INSIDE the target. "
    "For each reference, b is the visible bounding box of that particular reference OBJECT, and r describes "
    "the target relative to that object: left, right, above, below, inside, near, or none. "
    "Use references only when the user's instruction actually specifies a reference or containment; "
    "do not invent one from incidental layout. For between use two reference objects and two directional relations. "
    "Name target and references in at most six words each. For an unused reference write NONE in a and b, "
    "and none in r. If the target is not identifiable write NONE in t and p. No explanations or action claims."
)
LOCAL_PROMPT = (
    "Locate the requested target in this cropped screenshot. Reply only with "
    "<g><t>visible target name</t><p>x,y</p></g>. Name it in at most six words. "
    "x,y are integers 0..1000 relative to THIS crop, and must be inside that target. "
    "If the target is absent or uncertain use NONE in both fields. Do not substitute another control. "
    "No explanations or action claims."
)
RELATIONS = ('left','right','above','below','inside','near')
RIDGES = (.03,.3,3.)
TOKEN_SLOTS = 4
VIS_DIM = (TOKEN_SLOTS+2)*96*2  # global and local, retaining every head


def unit(x):
    x=np.asarray(x,np.float32)
    return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-8)


def parse(text,prefixes):
    frame=re.fullmatch(r'\s*<g>(.*?)</g>\s*',text,re.S)
    result=dict(complete=bool(frame),target=None,point=None,anchors=[],roles={})
    if not frame:return result
    tags=list(re.finditer(r'<([a-z][a-z0-9]*)>([^<>]*)</\1>',frame.group(1)))
    # XML whitespace BETWEEN complete fields is valid, as is whitespace around
    # the outer frame. Still reject unparsed prose and duplicate fields.
    remainder=re.sub(r'<([a-z][a-z0-9]*)>([^<>]*)</\1>','',frame.group(1))
    if remainder.strip():result['complete']=False;return result
    fields={m.group(1):m for m in tags}
    if len(fields)!=len(tags) or not {'t','p'}<=fields.keys():result['complete']=False;return result
    def value(tag):return fields[tag].group(2).strip() if tag in fields else 'NONE'
    def coordinates(tag,n):
        raw=value(tag);parts=raw.split(',')
        if len(parts)!=n or not all(re.fullmatch(r'\s*\d{1,4}\s*',x) for x in parts):return None
        vv=[int(x) for x in parts]
        return vv if all(0<=x<=1000 for x in vv) else None
    for role,tag in [('target','t'),('anchor1','a1'),('anchor2','a2')]:
        if tag not in fields:continue
        m=fields[tag];start=frame.start(1)+m.start(2);end=frame.start(1)+m.end(2)
        while start<end and text[start].isspace():start+=1
        while end>start and text[end-1].isspace():end-=1
        result['roles'][role]=[i+1 for i in _content_indices(prefixes,start,end)] if value(tag).upper()!='NONE' else []
    result['target']=None if value('t').upper()=='NONE' else value('t')
    result['point']=coordinates('p',2) if result['target'] else None
    for j in [1,2]:
        box=coordinates(f'b{j}',4);relation=value(f'r{j}').lower()
        valid=bool(box and box[0]<box[2] and box[1]<box[3] and relation in RELATIONS and value(f'a{j}').upper()!='NONE')
        result['anchors'].append(dict(name=value(f'a{j}'),box=box if valid else None,relation=relation if valid else 'none',valid=valid))
    return result


class GroundedRuntime(ConditionedFeatureRuntime):
    def infer(self,image,instruction,prompt,max_new_tokens):
        import torch
        with self._lock,torch.inference_mode():
            _sync_cuda();started=time.perf_counter()
            inputs,positions,grid=self._prepare_image(image,instruction,prompt)
            length=int(inputs['input_ids'].shape[-1]);ids=[];prefixes=[];maps=[];queries=[];ended=False;text=''
            with SpatialContentRecorder(self.model,self.config.layers) as recorder:
                recorder.begin(positions,grid)
                output=self.model(**inputs,use_cache=True,logits_to_keep=1,output_attentions=False,output_hidden_states=False,return_dict=True)
                initial=recorder.snapshot();ih=recorder.image_hidden;vh=recorder.vision_hidden
                mask=inputs.get('attention_mask')
                for step in range(max_new_tokens):
                    token=int(output.logits[0,-1].argmax().item())
                    if token in self.eos_ids:ended=True;break
                    ids.append(token);text=self.processor.tokenizer.decode(ids,skip_special_tokens=True);prefixes.append(text)
                    output,mask=self._forward_token(token,output,mask,length,step)
                    own=recorder.snapshot();maps.append(own['raw_image_attention']);queries.append(own['hidden'])
                    if '</g>' in text:break
                del output
            parsed=parse(text,prefixes);p=int(np.prod(grid));h=len(initial['head_order']);d=initial['hidden'].size
            arrays=dict(all_attention=np.stack(maps).astype(np.float32) if maps else np.empty((0,h,p),np.float32),
                all_hidden=np.stack(queries).astype(np.float16) if queries else np.empty((0,d),np.float16),image_hidden=ih,vision_hidden=vh)
            _sync_cuda()
            meta=dict(**parsed,generated_text=text,generated_token_ids=ids,generated_prefixes=prefixes,
                grid_shape=list(grid),image_size=list(image.size),head_order=initial['head_order'],input_tokens=length,
                own_token_forward=True,terminated_by_eos=ended,token_cap_reached=len(ids)==max_new_tokens and not ended,
                system_prompt=prompt,max_new_tokens=max_new_tokens,elapsed_seconds=time.perf_counter()-started,
                ground_truth_in_model_input=False,all_backbone_parameters_frozen=not any(p.requires_grad for p in self.model.parameters()))
            return arrays,meta


def visual_features(attention,vision,grid,points,size,transform=None,mode='centered',seed=''):
    """Center BEFORE normalization/pooling; retain 96 heads and four token slots.

    Uniform and spatially shuffled attention have identical dimensionality.
    Views are kept separate as global/local at the aggregation boundary.
    """
    gh,gw=map(int,grid);v=np.asarray(vision,np.float32);a=np.asarray(attention,np.float32)
    assert v.shape[0]==4*gh*gw and a.shape[1:]==(96,gh*gw)
    n=len(points);out=np.zeros((n,(TOKEN_SLOTS+2)*96),np.float32)
    bounds=np.asarray([0,0,*size] if transform is None else transform['box'],float)
    xy=(np.asarray(points).reshape(-1,2)-bounds[:2])/(bounds[2:]-bounds[:2]);valid=((xy>=0)&(xy<1)).all(-1)
    if not len(a):return out,np.zeros(n,bool)
    if mode!='raw':v=v-v.mean(0,keepdims=True)
    vf=unit(v).reshape(2*gh,2*gw,-1)
    coarse=vf.reshape(gh,2,gw,2,-1).mean((1,3)).reshape(gh*gw,-1)
    rel=a/np.maximum(a.sum(-1,keepdims=True),1e-12)
    if mode=='uniform':rel=np.full_like(rel,1/(gh*gw))
    if mode=='shuffled':
        rng=np.random.default_rng(int(hashlib.sha256(seed.encode()).hexdigest()[:16],16));rel=rel[...,rng.permutation(gh*gw)]
    template=unit(rel@coarse)
    cell=np.clip(np.floor(xy*[2*gw,2*gh]).astype(int),[0,0],[2*gw-1,2*gh-1])
    vv=vf[cell[:,1],cell[:,0]]
    sim=np.einsum('nd,thd->nth',vv,template,optimize=True)
    for j in range(min(TOKEN_SLOTS,len(a))):out[:,j*96:(j+1)*96]=sim[:,j]
    out[:,TOKEN_SLOTS*96:(TOKEN_SLOTS+1)*96]=sim.mean(1)
    out[:,(TOKEN_SLOTS+1)*96:]=sim.min(1)
    out[~valid]=0
    assert np.isfinite(out).all()
    return out,valid


def relation_features(points,size,anchors):
    """Execute soft predicates against predicted OBJECT boxes, never centroids."""
    xy=np.asarray(points,float).reshape(-1,2)/np.asarray(size,float)
    out=np.zeros((len(xy),15),np.float32);probs=[]
    for j,a in enumerate(anchors[:2]):
        if not a.get('valid'):continue
        box=np.asarray(a['box'],float)/1000;x,y=xy.T;s=.03;r=a['relation']
        if r=='left':prob=expit((box[0]-x)/s)
        elif r=='right':prob=expit((x-box[2])/s)
        elif r=='above':prob=expit((box[1]-y)/s)
        elif r=='below':prob=expit((y-box[3])/s)
        elif r=='inside':prob=expit((x-box[0])/s)*expit((box[2]-x)/s)*expit((y-box[1])/s)*expit((box[3]-y)/s)
        else:
            delta=np.maximum(np.maximum(box[:2]-xy,xy-box[2:]),0);prob=np.exp(-np.sum(delta**2,axis=1)/(2*.15**2))
        out[:,j*7+RELATIONS.index(r)]=prob;out[:,j*7+6]=1;probs.append(prob)
    if probs:out[:,14]=np.prod(probs,axis=0)
    return out


def fixed_crop(point,size):
    w,h=map(int,size);x,y=point
    cw=min(w,640);ch=min(h,640)
    x0=int(np.clip(round(x-cw/2),0,w-cw));y0=int(np.clip(round(y-ch/2),0,h-ch))
    return dict(box=[x0,y0,x0+cw,y0+ch],original_size=[w,h],scale=2,crop_size=[cw*2,ch*2],ground_truth_used=False)


def pixel(point,size,transform=None):
    if point is None:return None
    p=np.asarray(point,float)/1000*(np.asarray(size)-1)
    if transform is not None:
        p=p/transform['scale']+np.asarray(transform['box'][:2])
        p=np.clip(np.rint(p),[0,0],np.asarray(transform['original_size'])-1)
    return np.rint(p).astype(int).tolist()


def identity(dim):return dict(weight=np.zeros(dim),mean=np.zeros(dim),std=np.ones(dim),success=True,rankable_queries=0)


def fit(records,ridge,hard=False):
    dim=records[0]['x'].shape[1]
    if ridge is None:return identity(dim)
    xx=np.concatenate([r['x'] for r in records if len(r['x'])]);sigma=xx.std(0)
    mean=xx.mean(0);std=np.where(sigma>1e-10,np.maximum(sigma,1e-4),1.)
    ds=[];bs=[];ws=[];sources=sorted({r['source'] for r in records});counts={s:sum(r['source']==s for r in records) for s in sources}
    for r in records:
        pos=np.flatnonzero(r['y']);neg=np.flatnonzero(~r['y'])
        if not len(pos) or not len(neg):continue
        pos=pos[np.argsort(-r['base'][pos],kind='stable')[:4]]
        if hard:
            dist=np.linalg.norm(r['points'][neg,None]/r['size']-r['points'][pos][None]/r['size'],axis=-1).min(1)
            close=neg[np.argsort(dist,kind='stable')[:8]]
            high=neg[np.argsort(-r['base'][neg],kind='stable')[:8]]
            neg=np.unique(np.r_[close,high])
        else:neg=neg[np.argsort(-r['base'][neg],kind='stable')[:16]]
        pi=np.repeat(pos,len(neg));ni=np.tile(neg,len(pos))
        ds.append(((r['x'][pi]-r['x'][ni])/std).astype(np.float64));bs.append(r['base'][pi]-r['base'][ni])
        ws.append(np.full(len(pi),1/(len(sources)*counts[r['source']]*len(pi))))
    if not ds:return identity(dim)
    d=np.concatenate(ds);b=np.concatenate(bs);w=np.concatenate(ws);w/=w.sum()
    def obj(theta):
        margin=b+d@theta
        return float(w@np.logaddexp(0,-margin)+ridge*(theta@theta)/2),-d.T@(w*expit(-margin))+ridge*theta
    result=minimize(obj,np.zeros(dim),jac=True,method='L-BFGS-B',options=dict(maxiter=160,ftol=1e-9,gtol=1e-5))
    assert np.isfinite(result.x).all() and (result.success or np.max(np.abs(result.jac))<1e-4),(result.message,np.max(np.abs(result.jac)))
    return dict(weight=result.x,mean=mean,std=std,success=bool(result.success),rankable_queries=len(ds),pairs=len(d),iterations=int(result.nit),loss=float(result.fun))


def infer(model,record):return record['base']+(record['x']-model['mean'])/model['std']@model['weight']
