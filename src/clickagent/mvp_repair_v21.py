"""Public-only candidate construction and matched visual verification inputs."""
import hashlib
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

CHOICE_PROMPT = (
    "Identify the exact clickable GUI target described by the user's instruction. "
    "The first image is a screenshot overview with labeled proposed points. "
    "The following images are local crops, one per candidate. A hollow red ring "
    "marks the exact proposed click position; its center must be INSIDE the target. "
    "Markers and letters are added annotations, not original UI. Use the full "
    "instruction, the overview context, and all local crops. Choose the candidate "
    "whose ring center best matches the requested target, or N if none does or "
    "the target is ambiguous. Reply with exactly one candidate letter or N. "
    "Do not output coordinates or explanations."
)
PUBLIC_KEYS = ('sample_id','dataset','image','image_size','image_sha256','instruction','cohort','source','group','source_group')


def stable_key(value):
    return hashlib.sha256(value.encode()).hexdigest()[:24]


def valid_point(p, size):
    return isinstance(p,(list,tuple,np.ndarray)) and len(p)==2 and all(
        np.isfinite(v) and int(v)==v and 0<=v<limit for v,limit in zip(p,size))


def canonical(anchor, others):
    anchor=tuple(map(int,anchor))
    return [list(anchor)]+[list(p) for p in sorted({tuple(map(int,p)) for p in others}-{anchor})]


def make_pools(mvp_output, points, scores, size):
    """Match the exact distinct candidate count to MVP, keep its anchor always.

    First added point is the highest scored distinct point, including near-anchor
    refinements. The next favors a different14px neighborhood; native predictions
    fill missing slots. No labels, target geometry, or source identity accepted.
    """
    anchor=mvp_output.get('point')
    if not valid_point(anchor,size):
        return dict(native=[],augmented=[],anchor=None,our_additions=0)
    predictions=mvp_output.get('reference_result',{}).get('all_predictions',[])
    native=canonical(anchor,[p['point'] for p in predictions if valid_point(p.get('point'),size)])[:3]
    wanted=len(native);selected=[list(map(int,anchor))];new=0
    points=np.asarray(points).reshape(-1,2);scores=np.asarray(scores)
    assert len(points)==len(scores) and np.isfinite(scores).all()
    order=np.argsort(-scores,kind='stable')
    eligible=[list(map(int,points[i])) for i in order if valid_point(points[i],size) and list(map(int,points[i]))!=selected[0]]
    if wanted>1 and eligible:
        selected.append(eligible[0]);new+=1
    if wanted>2:
        distinct=[p for p in eligible if p not in selected]
        diverse=[p for p in distinct if all(max(abs(p[d]-q[d]) for d in [0,1])>14 for q in selected)]
        if diverse or distinct:
            selected.append((diverse or distinct)[0]);new+=1
    for p in native:
        if len(selected)==wanted:break
        if p not in selected:selected.append(p)
    augmented=canonical(anchor,selected[1:])
    assert len(native)==len(augmented)==wanted and native[0]==augmented[0]
    return dict(native=native,augmented=augmented,anchor=list(map(int,anchor)),our_additions=new)


def presentation_order(sample_id,n,reverse=False):
    seed=int(stable_key(sample_id+'|v21_order'),16)
    order=np.random.default_rng(seed).permutation(n).tolist()
    return list(reversed(order)) if reverse else order


def font(size):
    try:return ImageFont.truetype('arial.ttf',size)
    except OSError:return ImageFont.load_default(size=size)


def square_view(image,points,labels):
    """Fixed512x512 image; annotation geometry follows letterbox scaling."""
    w,h=image.size;scale=min(512/w,512/h)
    resized=image.resize((max(1,round(w*scale)),max(1,round(h*scale))),Image.Resampling.LANCZOS)
    canvas=Image.new('RGB',(512,512),(240,240,240));ox=(512-resized.width)//2;oy=(512-resized.height)//2
    canvas.paste(resized,(ox,oy));draw=ImageDraw.Draw(canvas);mapped=[]
    for i,(p,label) in enumerate(zip(points,labels)):
        x,y=ox+p[0]*resized.width/w,oy+p[1]*resized.height/h
        draw.ellipse((x-5,y-5,x+5,y+5),outline=(255,0,0),width=2)
        dx,dy=[(8,-20),(8,8),(-24,8)][i%3]
        tx,ty=max(0,min(490,x+dx)),max(0,min(490,y+dy))
        draw.rectangle((tx,ty,tx+18,ty+20),fill=(255,255,255));draw.text((tx+2,ty),label,font=font(17),fill=(180,0,0))
        mapped.append([x,y])
    return canvas,dict(source_size=[w,h],resized_size=list(resized.size),offset=[ox,oy],marked_centers=mapped)


def visual_inputs(image,instruction,pool,order):
    assert 1<=len(pool)<=3 and sorted(order)==list(range(len(pool)))
    shown=[pool[i] for i in order];labels=list('ABC')[:len(pool)]
    overview,overview_geometry=square_view(image,shown,labels);images=[overview];geometry=[]
    w,h=image.size
    for p,label in zip(shown,labels):
        # Fixed512-source-pixel local context, shifted at borders, no GT crop.
        cw,ch=min(512,w),min(512,h);x=max(0,min(w-cw,int(p[0])-cw//2));y=max(0,min(h-ch,int(p[1])-ch//2))
        crop=image.crop((x,y,x+cw,y+ch));marked,meta=square_view(crop,[[p[0]-x,p[1]-y]],[label]);images.append(marked)
        geometry.append(dict(box=[x,y,x+cw,y+ch],point=list(p),label=label,render=meta))
    content=[{'type':'text','text':'Full screenshot overview:'},{'type':'image'}]
    for label in labels:content += [{'type':'text','text':f'Candidate {label} local crop:'},{'type':'image'}]
    content.append({'type':'text','text':'User instruction: '+instruction+'\nValid replies: '+', '.join(labels+['N'])+'.'})
    messages=[{'role':'system','content':CHOICE_PROMPT},{'role':'user','content':content}]
    return images,messages,dict(order=order,labels=labels+['N'],overview=overview_geometry,crops=geometry,
                                input_pixels=(len(pool)+1)*512*512)


def combine_orders(pool,anchor,observations):
    assert len(observations)==2
    k=len(pool);aligned=[];choices=[]
    for obs in observations:
        order=obs['order'];scores=np.asarray(obs['scores'],np.float64)
        assert sorted(order)==list(range(k)) and scores.shape==(k+1,) and np.isfinite(scores).all()
        scores=scores-(np.max(scores)+np.log(np.exp(scores-np.max(scores)).sum()))
        canonical_scores=np.empty(k+1)
        for shown,original in enumerate(order):canonical_scores[original]=scores[shown]
        canonical_scores[-1]=scores[-1];aligned.append(canonical_scores);choices.append(int(canonical_scores.argmax()))
    scores=np.mean(aligned,0);choice=int(scores.argmax());fallback=choice==k
    return dict(point=list(anchor) if fallback else pool[choice],fallback_to_mvp=fallback,
                selected_candidate_index=None if fallback else choice,scores=scores.tolist(),
                order_disagrees=choices[0]!=choices[1],per_order_choice=choices)
