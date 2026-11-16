"""Prompt/task interventions with actual generated-token own-forward readouts."""
from __future__ import annotations
import re
import time
import numpy as np
from .conditioned_features import ConditionedFeatureRuntime, SpatialContentRecorder
from .model_features import _sync_cuda
from .target_token import TARGET_NAME_PROMPT, select_stages, _content_indices

CONTEXT_PROMPT = (
    "Inspect the screenshot and resolve the exact GUI target referred to by the user's instruction. "
    "Use its visible container and relative location to distinguish similar or repeated controls. "
    "Reply with an ACTION line followed by one complete grounded description:\nACTION: CLICK or WAIT\n"
    "<ground><container>visible container</container><location>position</location>"
    "<target>visible target label or name</target></ground>\n"
    "Choose CLICK when the requested target is identifiable, visible and enabled; otherwise WAIT. "
    "Container is a visible panel or toolbar name in at most four words, or NONE. "
    "Location describes the target relative to the CURRENT DISPLAYED IMAGE, using exactly one of "
    "upper_left, upper_center, upper_right, middle_left, center, middle_right, "
    "lower_left, lower_center, lower_right, unknown. Do not describe a position relative to another object. "
    "Target must name that specific target in at most twelve words, preferably using its visible text; "
    "describe an unlabeled icon. If the target cannot be identified use NONE. "
    "Do not output coordinates, explanations, extra lines, or claim that an action was executed."
)
COORDINATE_PROMPT = (
    "Inspect the screenshot and locate the exact GUI target referred to by the user's instruction. "
    "Reply in exactly three lines:\nACTION: CLICK or WAIT\n"
    "TARGET: the target's visible label or a short concrete name\nPOINT: x, y\n"
    "Choose CLICK when the requested target is identifiable, visible and enabled; otherwise WAIT. "
    "TARGET names that specific target in at most twelve words; use its visible text or describe an unlabeled icon. "
    "On POINT give integer x and y in 0..1000 for a point inside that target, normalized to the displayed "
    "screenshot width and height. If the target cannot be identified use TARGET: NONE and POINT: NONE. "
    "Do not output explanations or extra lines, and never claim an action was executed."
)
PROMPTS = dict(name=TARGET_NAME_PROMPT, context=CONTEXT_PROMPT, coordinate=COORDINATE_PROMPT)
ARMS = ('name', 'context_name', 'coordinate_name', 'coordinate_mean', 'framed_phrase',
        'entity_mean', 'role_sum', 'role_product', 'location_gate')
ALL_ARMS = ARMS+('coordinate_separator',)


def completed_line(text, tag, ended_by_eos=False, capped=False):
    match = re.search(r'(?:\A|\n)'+re.escape(tag)+r':[ \t]*([^\r\n]+)(?:\r?\n|$)', text)
    complete = bool(match and (match.end()<len(text) or text.endswith(('\n','\r')) or ended_by_eos))
    if capped and not text.endswith(('\n','\r')):
        complete = False
    return match if complete else None


def select_readouts(text, prefixes, *, ended_by_eos, capped):
    stages, parsed = select_stages(text, prefixes, ended_by_eos=ended_by_eos, token_cap_reached=capped)
    result = dict(target=stages.get('target_mean', []), coordinate=[], separator=[],
                  container=[], location=[], phrase=[])
    frame=re.search(r'<ground>(.*?)</ground>',text,re.DOTALL)
    fields={}
    if frame:
        remainder=frame.group(1)
        for tag in ['container','location','target']:
            matches=list(re.finditer(r'<'+tag+r'>([^<>]*)</'+tag+r'>',frame.group(1)))
            if len(matches)!=1:fields={};break
            field=matches[0];value=field.group(1).strip();fields[tag]=value
            start=frame.start(1)+field.start(1);end=frame.start(1)+field.end(1)
            while start<end and text[start].isspace():start+=1
            while end>start and text[end-1].isspace():end-=1
            role='target' if tag=='target' else tag
            result[role]=[i+1 for i in _content_indices(prefixes,start,end)] if value.upper() not in ('','NONE','UNKNOWN','N/A') else []
            remainder=remainder.replace(field.group(0),'',1)
        if not fields or remainder.strip():
            fields={};result.update(target=[],container=[],location=[])
        else:
            result['phrase']=sorted(set(result['target']+result['container']+result['location']))
            parsed.update(target_text=fields['target'],target_available=bool(result['target']),target_complete=True)
    point = None
    match = completed_line(text, 'POINT', ended_by_eos, capped)
    if match:
        raw = match.group(1).strip()
        # Complete, unambiguous integer coordinates only. Optional balanced
        # parentheses/brackets and comma/space separators are registered upfront.
        if raw[:1] in ('(', '['):
            closing = ')' if raw[0]=='(' else ']'
            raw = raw[1:-1].strip() if raw.endswith(closing) else ''
        pair = re.fullmatch(r'(\d{1,4})(?:\s*,\s*|\s+)(\d{1,4})', raw)
        if pair and all(0<=int(v)<=1000 for v in pair.groups()):
            point = [int(v) for v in pair.groups()]
            start, end = match.span(1)
            while start<end and text[start].isspace():start+=1
            while end>start and text[end-1].isspace():end-=1
            result['coordinate'] = [i+1 for i in _content_indices(prefixes, start, end)]
            comma = text.find(',', start, end)
            if comma>=0: result['separator'] = [i+1 for i in _content_indices(prefixes, comma, comma+1)]
    return result, dict(**parsed, normalized_point=point, coordinate_available=bool(result['coordinate']),
                       point_line_complete=match is not None,ground_fields=fields,ground_complete=bool(fields))


def should_stop(text, prompt_kind):
    if prompt_kind=='context':return '</ground>' in text
    tag = 'POINT' if prompt_kind=='coordinate' else 'TARGET'
    return completed_line(text, tag) is not None


class PromptReadoutRuntime(ConditionedFeatureRuntime):
    def infer_trajectory(self, image, instruction, prompt_kind, max_new_tokens=64):
        import torch
        assert prompt_kind in PROMPTS
        with self._lock, torch.inference_mode():
            _sync_cuda(); started=time.perf_counter()
            inputs, positions, grid = self._prepare_image(image, instruction, PROMPTS[prompt_kind])
            input_length=int(inputs['input_ids'].shape[-1])
            generated=[]; prefixes=[]; maps=[]; queries=[]; ended=False; text=''
            with SpatialContentRecorder(self.model,self.config.layers) as recorder:
                recorder.begin(positions,grid)
                outputs=self.model(**inputs,use_cache=True,logits_to_keep=1,
                    output_attentions=False,output_hidden_states=False,return_dict=True)
                initial=recorder.snapshot(); image_hidden=recorder.image_hidden; vision_hidden=recorder.vision_hidden
                assert image_hidden is not None and vision_hidden is not None
                attention_mask=inputs.get('attention_mask')
                for step in range(max_new_tokens):
                    token_id=int(outputs.logits[0,-1].argmax().item())
                    if token_id in self.eos_ids:ended=True;break
                    generated.append(token_id)
                    text=self.processor.tokenizer.decode(generated,skip_special_tokens=True);prefixes.append(text)
                    # Always forward the actual sampled token before capturing
                    # its own query/hidden state, including coordinate tokens.
                    outputs,attention_mask=self._forward_token(token_id,outputs,attention_mask,input_length,step)
                    snapshot=recorder.snapshot()
                    maps.append(snapshot['raw_image_attention']);queries.append(snapshot['hidden'])
                    if should_stop(text,prompt_kind):break
                del outputs
            capped=len(generated)==max_new_tokens and not ended
            stages,parsed=select_readouts(text,prefixes,ended_by_eos=ended,capped=capped)
            p=int(np.prod(grid));heads=len(initial['head_order']);qdim=int(initial['hidden'].size)
            arrays=dict(all_attention=np.stack(maps).astype(np.float32) if maps else np.empty((0,heads,p),np.float32),
                all_hidden=np.stack(queries).astype(np.float16) if queries else np.empty((0,qdim),np.float16),
                image_hidden=image_hidden,vision_hidden=vision_hidden,
                prefill_attention=initial['raw_image_attention'].astype(np.float32),prefill_hidden=initial['hidden'].astype(np.float16))
            _sync_cuda()
            meta=dict(**parsed,prompt_kind=prompt_kind,system_prompt=PROMPTS[prompt_kind],generated_text=text,
                generated_token_ids=generated,generated_prefixes=prefixes,readout_stages=stages,
                grid_shape=list(grid),image_size=list(image.size),head_order=initial['head_order'],
                image_content_layer=15,image_content_positions=positions.tolist(),
                image_content_source='prefill decoder output at actual image-token positions',
                fine_grid_shape=[int(grid[0])*2,int(grid[1])*2],
                vision_content_source='last vision block before merger, restored native raster',
                input_tokens=input_length,own_token_forward=True,ground_truth_in_model_input=False,
                terminated_by_eos=ended,token_cap_reached=capped,max_new_tokens=max_new_tokens,
                elapsed_seconds=time.perf_counter()-started,
                all_backbone_parameters_frozen=not any(p.requires_grad for p in self.model.parameters()))
            return arrays,meta


def select_arrays(arrays, metadata, stage):
    assert stage in ('target','coordinate','separator','phrase','container','location')
    indices=metadata['readout_stages'][stage]
    assert all(1<=i<=len(metadata['generated_token_ids']) for i in indices)
    ix=np.asarray(indices,dtype=np.int64)-1
    selected=dict(target_attention=arrays['all_attention'][ix],target_hidden=arrays['all_hidden'][ix],
                  image_hidden=arrays['image_hidden'],vision_hidden=arrays['vision_hidden'])
    meta=dict(metadata)
    meta.update(target_available=bool(indices),target_complete=bool(indices),target_raw_indices=indices,
        target_query_absolute_positions=[metadata['input_tokens']+i-1 for i in indices],
        readout_stage=stage,readout_available=bool(indices))
    return selected,meta


LOCATION_CENTERS={
    'upper_left':(1/6,1/6),'upper_center':(.5,1/6),'upper_right':(5/6,1/6),
    'middle_left':(1/6,.5),'center':(.5,.5),'middle_right':(5/6,.5),
    'lower_left':(1/6,5/6),'lower_center':(.5,5/6),'lower_right':(5/6,5/6)}


def compose_arrays(arrays,metadata,operator):
    """Combine same-grid evidence before peak extraction/candidate pruning.

    The entity query and absolute image-attention mass are identical in every
    operator. Modifier maps are evidence, not calibrated probabilities. The
    location gate is an explicitly geometric alternative; raw attention products
    must not be interpreted as executing arbitrary spatial relation predicates.
    """
    assert operator in ('entity_mean','role_sum','role_product','location_gate')
    entity,meta=select_arrays(arrays,metadata,'target')
    if not meta['target_available']:return entity,meta
    mean=entity['target_attention'].mean(0,dtype=np.float32)
    query=entity['target_hidden'].astype(np.float32).mean(0).astype(np.float16)
    mass=mean.sum(-1,keepdims=True);p=mean.shape[-1]
    def normalize(a):
        total=a.sum(-1,keepdims=True)
        return np.divide(a,total,out=np.full_like(a,1/p),where=total>1e-12)
    base=normalize(mean);groups=[base];weights=[1.]
    for role in ['container','location']:
        indices=metadata['readout_stages'][role]
        if indices:
            modifier=arrays['all_attention'][np.asarray(indices)-1].mean(0,dtype=np.float32)
            groups.append(normalize(modifier));weights.append(.5)
    if operator=='role_sum':
        relative=sum(w*a for w,a in zip(weights,groups))/sum(weights)
    elif operator=='role_product':
        log=sum(w*np.log(np.maximum(a,1e-8)) for w,a in zip(weights,groups))
        relative=normalize(np.exp(log-log.max(-1,keepdims=True)))
    elif operator=='location_gate':
        location=metadata.get('ground_fields',{}).get('location','unknown').lower().strip()
        if location in LOCATION_CENTERS:
            gh,gw=metadata['grid_shape'];cx,cy=LOCATION_CENTERS[location]
            yy,xx=np.meshgrid((np.arange(gh)+.5)/gh,(np.arange(gw)+.5)/gw,indexing='ij')
            prior=np.exp(-((xx-cx)**2+(yy-cy)**2)/(2*.35**2)).reshape(1,-1)
            relative=normalize((base*prior).astype(np.float32))
        else:relative=base
    else:relative=base
    selected=dict(target_attention=(relative*mass)[None].astype(np.float32),target_hidden=query[None],
                  image_hidden=arrays['image_hidden'],vision_hidden=arrays['vision_hidden'])
    meta.update(readout_stage=operator,composition_operator=operator,composition_modifier_weight=.5,
                composition_query='same mean entity query for every operator',composition_before_candidate_pruning=True,
                source_role_indices={k:metadata['readout_stages'][k] for k in ['target','container','location']})
    return selected,meta


def normalized_to_pixel(point, size):
    if point is None:return None
    return [round(v*(s-1)/1000) for v,s in zip(point,size)]


def map_view_point(point, transform):
    if point is None:return None
    if transform is None:return list(point)
    return [int(round(point[d]/transform['scale']+transform['box'][d])) for d in (0,1)]


def coordinate_consensus(points):
    """Registered simple14px complete-link grouping, original view order ties."""
    groups=[]
    for index,p in enumerate(points):
        if p is None:continue
        found=next((g for g in groups if all(max(abs(p[d]-q[d]) for d in (0,1))<=14 for _,q in g)),None)
        if found is None:groups.append([(index,p)])
        else:found.append((index,p))
    if not groups:return None
    best=max(groups,key=lambda g:len(g))
    return list(best[0][1])
