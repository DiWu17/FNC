"""V9 geometry, contextual own-token collection, and comparable candidate choice.

The v8 modules remain sealed. No geometry label enters this module.
"""
from __future__ import annotations

import re
import time
import numpy as np
from PIL import Image, ImageDraw

from .conditioned_features import ConditionedFeatureRuntime, SpatialContentRecorder, unmerge_raster
from .model_features import _sync_cuda
from .point_aligned_inference import bounded_box, proposal_crop
from .target_token import TARGET_NAME_PROMPT, select_stages


CHOICE_PROMPT = (
    "Identify the GUI target in the user's instruction. The first image is the full screenshot overview. "
    "Subsequent images show proposed points in labeled local views. Each hollow red circle marks an exact "
    "point; the center must be inside the intended target, not merely near it. Compare ALL candidates. "
    "Select the best candidate, or N if none is correct. The markers and labels are annotations. "
    "Reply with exactly one candidate letter or N. Do not output coordinates or claim an action occurred."
)
CONTEXT_NOTE = (
    "Image 1 is an overview of the whole screen; its red rectangle identifies the region enlarged in image 2. "
    "Resolve the instruction using BOTH images, and locate the target within image 2. "
    "If the target is outside image 2 or cannot be identified there, use TARGET: NONE. "
    "An earlier tentative target name is: {name}. This name may be wrong; correct it using the images.\n"
    "User instruction: {instruction}"
)


def fit_pixels(image, pixels):
    scale=min(1.,np.sqrt(pixels/(image.width*image.height)))
    return image.resize((max(1,round(image.width*scale)),max(1,round(image.height*scale))),Image.Resampling.LANCZOS)


def overlap_iou(a,b):
    area=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    aa=(a[2]-a[0])*(a[3]-a[1]);bb=(b[2]-b[0])*(b[3]-b[1])
    return area/max(1,aa+bb-area)


def diverse_crops(image,candidates,grid_shape,maximum=2):
    selected=[]
    for item in candidates:
        crop,transform=proposal_crop(image,item['parent_index'],grid_shape)
        if any(overlap_iou(transform['box'],t['box'])>.3 for _,t in selected):continue
        selected.append((crop,transform))
        if len(selected)==maximum:break
    return selected


def split_image_positions(ids,image_token_id,grids,merge):
    positions=np.flatnonzero(np.asarray(ids)==image_token_id)
    counts=[int(np.prod(g))//(merge*merge) for g in grids]
    if sum(counts)!=len(positions):raise ValueError('multi-image tokens/grid mismatch')
    return np.split(positions,np.cumsum(counts)[:-1])


class LocalSpatialRecorder(SpatialContentRecorder):
    def __init__(self,model,layers,vision_offset):
        super().__init__(model,layers);self.vision_offset=vision_offset

    def _vision_hook(self,module,args):
        import torch
        count=int(np.prod(self.grid_shape))*4
        full=args[0]
        if full.shape[0]<self.vision_offset+count:raise ValueError('missing local vision patches')
        features=full[self.vision_offset:self.vision_offset+count].detach().to(dtype=torch.float16).cpu().numpy()
        self.vision_hidden=unmerge_raster(features,self.grid_shape,2)


class ContextFeatureRuntime(ConditionedFeatureRuntime):
    def prepare_views(self,images,text,system_prompt,labels=None):
        content=[]
        for i,img in enumerate(images):
            if labels:content.append({'type':'text','text':labels[i]})
            content.append({'type':'image'})
        content.append({'type':'text','text':text})
        messages=[{'role':'system','content':system_prompt},{'role':'user','content':content}]
        template=self.processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        inputs=self.processor(text=[template],images=[im.convert('RGB') for im in images],return_tensors='pt',padding=False)
        grids=inputs['image_grid_thw'].cpu().numpy()
        positions=split_image_positions(inputs['input_ids'][0].cpu().numpy(),self.model.config.image_token_id,
                                       grids,self.model.config.vision_config.spatial_merge_size)
        return inputs.to(self.device),positions,grids

    def infer_context_features(self,image,crop,instruction,transform,target_name,max_new_tokens=64):
        import torch
        with self._lock,torch.inference_mode():
            _sync_cuda();started=time.perf_counter()
            overview=image.copy();ImageDraw.Draw(overview).rectangle(transform['box'],outline=(255,0,0),width=max(2,round(image.width/500)))
            overview=fit_pixels(overview,65536)
            text=CONTEXT_NOTE.format(name=target_name,instruction=instruction)
            inputs,positions,grids=self.prepare_views([overview,crop],text,TARGET_NAME_PROMPT)
            grid=tuple((grids[1,1:]//2).tolist());input_length=int(inputs['input_ids'].shape[-1])
            vision_offset=int(np.prod(grids[0]));generated=[];prefixes=[];maps=[];queries=[];text='';ended=False
            with LocalSpatialRecorder(self.model,self.config.layers,vision_offset) as recorder:
                recorder.begin(positions[1],grid)
                outputs=self.model(**inputs,use_cache=True,logits_to_keep=1,output_attentions=False,output_hidden_states=False,return_dict=True)
                snap=recorder.snapshot();prefill_attention=snap['raw_image_attention'];prefill_hidden=snap['hidden']
                image_hidden=recorder.image_hidden;vision_hidden=recorder.vision_hidden;mask=inputs.get('attention_mask')
                assert image_hidden is not None and vision_hidden is not None
                for step in range(max_new_tokens):
                    token=int(outputs.logits[0,-1].argmax().item())
                    if token in self.eos_ids:ended=True;break
                    generated.append(token);text=self.processor.tokenizer.decode(generated,skip_special_tokens=True);prefixes.append(text)
                    outputs,mask=self._forward_token(token,outputs,mask,input_length,step)
                    snap=recorder.snapshot();maps.append(snap['raw_image_attention']);queries.append(snap['hidden'])
                    if re.search(r'(?:\A|\n)TARGET:[^\r\n]*\r?\n',text):break
                del outputs
            cap=len(generated)==max_new_tokens and not ended
            stages,parsed=select_stages(text,prefixes,ended_by_eos=ended,token_cap_reached=cap)
            indices=stages.get('target_mean',[]);p=int(np.prod(grid));heads=len(snap['head_order']);qdim=int(prefill_hidden.size)
            attention=np.stack([maps[i-1] for i in indices]) if indices else np.empty((0,heads,p),np.float32)
            query=np.stack([queries[i-1] for i in indices]) if indices else np.empty((0,qdim),np.float16)
            arrays=dict(target_attention=attention.astype(np.float32),target_hidden=query.astype(np.float16),
                image_hidden=image_hidden,vision_hidden=vision_hidden,prefill_attention=prefill_attention,prefill_hidden=prefill_hidden)
            _sync_cuda()
            meta=dict(**parsed,generated_text=text,generated_token_ids=generated,generated_prefixes=prefixes,stages=stages,
                target_raw_indices=indices,target_query_absolute_positions=[input_length+i-1 for i in indices],
                grid_shape=list(grid),image_size=list(crop.size),head_order=snap['head_order'],
                image_content_layer=15,image_content_positions=positions[1].tolist(),image_content_source='local image prefill positions after full overview',
                vision_content_source='local image slice of premerge vision features, restored to raster',
                vision_offset=vision_offset,all_image_grids=grids.tolist(),all_image_positions=[v.tolist() for v in positions],
                input_tokens=input_length,own_token_forward=True,ground_truth_in_model_input=False,
                terminated_by_eos=ended,token_cap_reached=cap,system_prompt=TARGET_NAME_PROMPT,
                context_note=CONTEXT_NOTE.format(name=target_name,instruction=instruction),
                elapsed_seconds=time.perf_counter()-started,all_backbone_parameters_frozen=not any(p.requires_grad for p in self.model.parameters()))
            return arrays,meta


def marked_candidate_view(image,point,label):
    box=bounded_box(image.size,point,(768,768));crop=image.crop(box)
    x,y=point[0]-box[0],point[1]-box[1]
    ImageDraw.Draw(crop).ellipse((x-12,y-12,x+12,y+12),outline=(255,0,0),width=3)
    # Let the SAME official processor resize these views as in v8 verify_point.
    return crop,dict(box=box,point=list(point),label=label,marker_radius=12,marker_width=3)


def compare_candidates(runtime,image,instruction,pool,reverse=False):
    import torch
    if not pool:return dict(point=None,order=[],scores=[],ground_truth_in_model_inputs=False)
    items=list(reversed(pool)) if reverse else list(pool)
    labels=[chr(ord('A')+i) for i in range(len(items))]
    overview=fit_pixels(image,65536);images=[overview];transforms=[]
    for label,item in zip(labels,items):
        view,transform=marked_candidate_view(image,item['point'],label);images.append(view);transforms.append(transform)
    text='User instruction: '+instruction+'\nValid replies: '+', '.join(labels+['N'])+'.'
    with runtime._lock,torch.inference_mode():
        _sync_cuda();started=time.perf_counter()
        inputs,positions,grids=runtime.prepare_views(images,text,CHOICE_PROMPT,['Full screenshot']+[f'Candidate {v}' for v in labels])
        output=runtime.model(**inputs,use_cache=False,logits_to_keep=1,output_attentions=False,output_hidden_states=False,return_dict=True)
        logits=output.logits[0,-1].detach().float().cpu();del output
        ids=[]
        for label in labels+['N']:
            variants=[]
            for prefix in ('',' '):
                token=runtime.processor.tokenizer.encode(prefix+label,add_special_tokens=False)
                if len(token)==1:variants.append(token[0])
            assert variants;ids.append(sorted(set(variants)))
        scores=torch.stack([torch.logsumexp(logits[v],0) for v in ids]);choice=int(scores.argmax());forced=int(scores[:-1].argmax())
        mass=float(torch.exp(torch.logsumexp(scores,0)-torch.logsumexp(logits,0)))
        _sync_cuda()
        return dict(point=items[choice]['point'] if choice<len(items) else None,forced_point=items[forced]['point'],
            order=[v['point'] for v in items],scores=scores.tolist(),labels=labels+['N'],token_ids=ids,
            choice=labels[choice] if choice<len(items) else 'N',choice_mass=mass,transforms=transforms,
            image_grids=grids.tolist(),per_candidate_pixel_budget=262144,
            elapsed_seconds=time.perf_counter()-started,reverse=reverse,
            generated_coordinates=False,ground_truth_in_model_inputs=False,calibrated_probability=False)
