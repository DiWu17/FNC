"""Candidate-directed reinspection and content-aware, exact-point decisions."""
from __future__ import annotations
import hashlib
import json
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

SHORT_GLOBAL_PROMPT = (
    "Locate the exact GUI element specified by the complete user instruction. "
    "Reply only with <g><t>visible target name</t><p>x,y</p></g>. "
    "Name it in at most six words. x,y are integers 0..1000 relative to the full screenshot "
    "and must be inside the requested target. Respect all positional, object and relational "
    "constraints in the instruction. If the target is absent or uncertain, write NONE in both fields. "
    "No reference boxes, explanations, or claims that an action was performed."
)
JUDGE_PROMPT = (
    "Evaluate two proposed GUI click points using the complete instruction and screenshot. "
    "The top panel is the full screenshot overview. Bottom LEFT and RIGHT panels are detailed "
    "crops centered on the two proposed clicks. Hollow green LEFT and purple RIGHT rings mark "
    "the proposed points; rings, panel borders and headings are added annotations, not GUI content. "
    "A point is correct only if it lies inside the specific element requested, respecting all "
    "instruction constraints. Both points may be correct or both may be wrong. "
    "Reply A if only LEFT is correct, B if only RIGHT is correct, C if both are correct, "
    "or D if neither is correct. Output exactly one letter. Do not execute an action."
)
MODES = ('geometry', 'content', 'combined', 'blurred', 'shuffled')

def key(s): return hashlib.sha256(s.encode()).hexdigest()[:24]
def unit(x):
    x=np.asarray(x,dtype=np.float64)
    return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)
def iou(a,b):
    a,b=np.asarray(a,float),np.asarray(b,float)
    intersection=float(np.prod(np.maximum(np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2]),0)))
    return intersection/max(float(np.prod(a[2:]-a[:2])+np.prod(b[2:]-b[:2])-intersection),1.)

def plan_diverse(points,scores,size,first_box):
    from .grounded_composition_v28 import fixed_crop
    points=np.asarray(points,float).reshape(-1,2);scores=np.asarray(scores,float)
    assert len(points)==len(scores) and np.isfinite(points).all() and np.isfinite(scores).all()
    if not len(points):
        points=np.asarray([[size[0]*x,size[1]*y] for x,y in [(0.25,0.25),(0.75,0.25),(0.25,0.75),(0.75,0.75)]])
        scores=np.zeros(4)
    order=np.argsort(-scores,kind='stable')[:64];points=points[order];scores=scores[order]
    z=(scores-scores.mean())/max(float(scores.std()),1e-6);w=np.exp(np.clip(z-z.max(),-30,0));w/=w.sum()
    plans=[]
    for j,p in enumerate(points):
        tr=fixed_crop(p.tolist(),size);x0,y0,x1,y1=tr['box']
        inside=((points[:,0]>=x0)&(points[:,0]<x1)&(points[:,1]>=y0)&(points[:,1]<y1))
        plans.append((tr,float(w[inside].sum()),iou(tr['box'],first_box),j))
    eligible=[p for p in plans if p[2]<=.25]
    chosen=max(eligible or plans,key=lambda p:(p[1]*(1-p[2]),-p[2],-p[3]))
    tr,support,overlap,j=chosen
    return tr,dict(candidate_count=len(points),support=support,first_crop_iou=overlap,
                   satisfies_iou_limit=bool(overlap<=.25),selected_center=points[j].tolist(),labels_used=False)

def proposal_observation(arrays,meta):
    idx=np.asarray(meta['roles'].get('target',[]),int)-1
    raw=dict(target_attention=arrays['all_attention'][idx],target_hidden=arrays['all_hidden'][idx],
             image_hidden=arrays['image_hidden'],vision_hidden=arrays['vision_hidden'])
    return raw,dict(meta,target_available=bool(len(idx)),target_complete=bool(meta['complete']),target_text=meta['target'])

def first_valid(*points):
    return next((list(p) for p in points if p is not None),None)

def consensus(points,preferred,threshold=14):
    groups=[]
    for p in points:
        if p is None:continue
        p=list(p)
        group=next((g for g in groups if all(np.max(np.abs(np.asarray(p)-q))<=threshold for q in g)),None)
        if group is None:groups.append([p])
        else:group.append(p)
    if not groups:return None
    group=max(groups,key=lambda g:(len(g),int(preferred is not None and list(preferred) in g)))
    return list(preferred) if preferred is not None and list(preferred) in group else group[0]

def eligible_pair(base,alternative):
    return base is not None and alternative is not None and list(base)!=list(alternative)

def judge_canvas(image,left,right,blur=False):
    """A single512x512 visual input;128 original-pixel local crops enlarged2x."""
    w,h=image.size;canvas=Image.new('RGB',(512,512),(235,235,235));geometry={}
    scale=min(512/w,256/h);rw,rh=max(1,round(w*scale)),max(1,round(h*scale));ox,oy=(512-rw)//2,(256-rh)//2
    with image.resize((rw,rh),Image.Resampling.LANCZOS) as im:
        if blur:im=im.filter(ImageFilter.GaussianBlur(8))
        canvas.paste(im,(ox,oy))
    points=[left,right]
    for j,p in enumerate(points):
        cw,ch=min(128,w),min(128,h);x0=int(np.clip(round(p[0]-cw/2),0,w-cw));y0=int(np.clip(round(p[1]-ch/2),0,h-ch))
        with image.crop((x0,y0,x0+cw,y0+ch)).resize((256,256),Image.Resampling.LANCZOS) as patch:
            if blur:patch=patch.filter(ImageFilter.GaussianBlur(8))
            canvas.paste(patch,(j*256,256))
        geometry[str(j)]=dict(box=[x0,y0,x0+cw,y0+ch],local_center=[j*256+(p[0]-x0)*256/cw,256+(p[1]-y0)*256/ch])
    draw=ImageDraw.Draw(canvas)
    try:font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',15)
    except OSError:font=ImageFont.load_default()
    for j,(p,label,color) in enumerate(zip(points,['LEFT','RIGHT'],[(0,165,20),(180,0,180)])):
        x,y=ox+p[0]*rw/w,oy+p[1]*rh/h
        draw.ellipse((x-5,y-5,x+5,y+5),outline=color,width=2)
        x,y=geometry[str(j)]['local_center'];draw.ellipse((x-8,y-8,x+8,y+8),outline=color,width=2)
        draw.rectangle((j*256,256,j*256+70,276),fill='white');draw.text((j*256+3,257),label,fill=color,font=font)
    draw.line((0,255,511,255),fill='black',width=2);draw.line((255,256,255,511),fill='black',width=2)
    return canvas,dict(overview_resize=[rw,rh],overview_offset=[ox,oy],candidates=[list(left),list(right)],
        local=geometry,blurred=blur,ground_truth_used=False,canvas_size=[512,512])

def pair_geometry(base,alternative,global_point,base_crop,diverse_crop,size,names):
    """Relative geometry/provenance only;no image/source IDs or absolute XY."""
    from .coordinate_calibration_v29 import name_agreement
    diagonal=max(float(np.linalg.norm(size)),1.)
    def describe(p,part):
        def edge(box):
            x0,y0,x1,y1=box;x,y=p
            return float(np.clip(min((x-x0)/max(x1-x0,1),(x1-x)/max(x1-x0,1),
                (y-y0)/max(y1-y0,1),(y1-y)/max(y1-y0,1))*2,0,1))
        d=np.asarray(p)-global_point if global_point is not None else np.zeros(2)
        return np.asarray([float(part=='base'),d[0]/diagonal,d[1]/diagonal,np.linalg.norm(d)/diagonal,
            edge(base_crop),edge(diverse_crop),name_agreement(names.get(part),names.get('global'))])
    return unit(describe(alternative,'alternative')-describe(base,'base'))

def joint_vector(forward,reverse):return unit(unit(forward)-unit(reverse))

def kernel(a,b,mode):
    geo=a['geometry']@b['geometry'].T
    if mode=='geometry':return geo
    field='blurred' if mode=='blurred' else 'content'
    content=a[field]@b[field].T
    return content if mode=='content' else .5*(geo+content)

def stack(records,mode='combined',salt='train'):
    value={field:np.asarray([r[field] for r in records],float) for field in ['geometry','content','blurred']}
    if mode=='shuffled' and len(records)>1:
        # Keep geometry and capacity; break query-to-joint-representation binding.
        order=sorted(range(len(records)),key=lambda i:key(records[i]['sample_id']+'|'+salt))
        permutation=np.arange(len(records))
        for i,j in enumerate(order):permutation[j]=order[(i+1)%len(order)]
        value['content']=value['content'][permutation]
    return value

def fit(records,mode,ridge=.01):
    assert records and all(r['eligible'] for r in records)
    x=stack(records,mode,'train');y=np.asarray([r['delta'] for r in records],float)
    counts={s:sum(r['source']==s for r in records) for s in {r['source'] for r in records}}
    weight=np.asarray([1/(len(counts)*counts[r['source']]) for r in records])
    k=kernel(x,x,mode);alpha=np.linalg.solve(k+np.diag(ridge/weight),y)
    assert np.isfinite(alpha).all()
    return dict(mode=mode,ridge=ridge,basis=x,alpha=alpha,training_ids=[r['sample_id'] for r in records],
        training_rgb=len(records),repairs=int((y>0).sum()),harms=int((y<0).sum()),ties=int((y==0).sum()),
        feature_dimension=len(records[0]['content']),geometry_dimension=len(records[0]['geometry']))

def scores(model,records):
    if not records:return np.empty(0)
    return kernel(stack(records,model['mode'],'evaluation'),model['basis'],model['mode'])@model['alpha']

def decide(base,alternative,utility,threshold=.1):
    if base is None:return first_valid(alternative),'missing_base_fallback'
    if not eligible_pair(base,alternative):return list(base),'keep_base'
    return (list(alternative),'learned_switch') if utility>threshold else (list(base),'keep_base')
