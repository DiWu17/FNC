"""Label-free crop geometry and a frozen, zero-shot marked-point verifier."""
import time

import numpy as np
from PIL import Image, ImageDraw
import torch


VERIFIER_PROMPT = (
    "Inspect the GUI screenshot and the user's target instruction. A hollow red circle marks one proposed "
    "click point. Decide whether the CENTER of that circle lies inside the intended visible clickable target. "
    "The circle is only a marker, not part of the GUI. Reply with exactly Yes or No. Do not output coordinates, "
    "execute anything, or claim that a click occurred. If the intended target cannot be identified here, reply No."
)


def bounded_box(size, center, span):
    """Integer crop, shifted at edges rather than padded with synthetic pixels."""
    result=[]
    for full, c, width in zip(size,center,span):
        width=min(full,max(1,int(round(width))))
        start=min(full-width,max(0,int(round(c-width/2))))
        result.append((start,start+width))
    return [result[0][0],result[1][0],result[0][1],result[1][1]]


def proposal_crop(image, parent_index, grid_shape, cells=4, minimum=256, scale=2):
    gh,gw=grid_shape;row,col=divmod(int(parent_index),gw)
    center=((col+.5)*image.width/gw,(row+.5)*image.height/gh)
    box=bounded_box(image.size,center,(max(minimum,cells*image.width/gw),max(minimum,cells*image.height/gh)))
    crop=image.crop(box);crop=crop.resize((crop.width*scale,crop.height*scale),Image.Resampling.LANCZOS)
    return crop,dict(box=box,scale=scale,original_size=list(image.size),crop_size=list(crop.size),
        parent_index=int(parent_index),grid_shape=list(grid_shape),ground_truth_used=False)


def map_crop_point(point, transform):
    if point is None:return None
    box=transform['box'];scale=transform['scale'];w,h=transform['original_size']
    return [int(np.clip(round(point[0]/scale+box[0]),0,w-1)),int(np.clip(round(point[1]/scale+box[1]),0,h-1))]


def marked_crop(image, point, side=768):
    box=bounded_box(image.size,point,(side,side));crop=image.crop(box)
    local=[point[0]-box[0],point[1]-box[1]]
    # A hollow circle preserves pixels at the actual point being verified.
    ImageDraw.Draw(crop).ellipse((local[0]-12,local[1]-12,local[0]+12,local[1]+12),outline=(255,0,0),width=3)
    return crop,dict(box=box,point=list(point),local_point=local,marker_radius=12,marker_width=3)


def answer_token_ids(tokenizer):
    result={}
    for name in ('Yes','No'):
        ids=set()
        for word in (name,name.lower(),name.upper()):
            for prefix in ('',' '):
                tokens=tokenizer.encode(prefix+word,add_special_tokens=False)
                if len(tokens)==1:ids.add(int(tokens[0]))
        assert ids, f'No single-token {name} variants'
        result[name]=sorted(ids)
    assert set(result['Yes']).isdisjoint(result['No'])
    return result


def yes_no_score(logits, answer_ids):
    logits=logits.float()
    yes=torch.logsumexp(logits[answer_ids['Yes']],0);no=torch.logsumexp(logits[answer_ids['No']],0)
    denom=torch.logaddexp(yes,no)
    return dict(score=float((yes-denom).exp()),yes_no_mass=float((denom-torch.logsumexp(logits,0)).exp()),
        yes_logit=float(yes),no_logit=float(no))


def verify_point(runtime,image,instruction,point):
    crop,transform=marked_crop(image,point);ids=answer_token_ids(runtime.processor.tokenizer)
    with runtime._lock,torch.inference_mode():
        torch.cuda.synchronize();started=time.perf_counter()
        inputs,positions,grid=runtime._prepare_image(crop,instruction,VERIFIER_PROMPT)
        output=runtime.model(**inputs,use_cache=False,logits_to_keep=1,output_attentions=False,output_hidden_states=False,return_dict=True)
        logits=output.logits[0,-1].detach().float().cpu();del output
        result=yes_no_score(logits,ids);top=logits.topk(5)
        result.update(transform=transform,answer_ids=ids,top5=[dict(id=int(i),text=runtime.processor.tokenizer.decode([int(i)]),logit=float(v)) for i,v in zip(top.indices,top.values)],
            elapsed_seconds=time.perf_counter()-started,grid_shape=list(grid),image_tokens=len(positions),
            instruction=instruction,system_prompt=VERIFIER_PROMPT,ground_truth_in_model_input=False,
            generated_coordinates=False,calibrated_probability=False)
        return result


def unique_points(*groups,limit=None):
    seen=set();result=[]
    for group in groups:
        for item in group:
            key=tuple(item['point'])
            if key not in seen:result.append(item);seen.add(key)
    return result if limit is None else result[:limit]
