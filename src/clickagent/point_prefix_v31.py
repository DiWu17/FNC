"""Stop unused XML tails,keeping the original prompt and actual generated IDs."""
import re
import time
import numpy as np

def point_prefix(text,prefixes):
    from .grounded_composition_v28 import parse
    if not re.match(r'\s*<g>',text):return None
    t=re.search(r'<t>[^<>]*</t>',text);p=re.search(r'<p>[^<>]*</p>',text)
    if t is None or p is None:return None
    end=max(t.end(),p.end());body=text[:end]
    parsed=parse(body+'</g>',prefixes)
    if not parsed['complete']:return None
    return dict(parsed,prefix_end=end,parser_only_closing_tag=True)

def infer_prefix(runtime,image,instruction,prompt,max_new_tokens):
    import torch
    from .grounded_composition_v28 import parse
    from .conditioned_features import SpatialContentRecorder
    from .model_features import _sync_cuda
    with runtime._lock,torch.inference_mode():
        _sync_cuda();started=time.perf_counter();inputs,positions,grid=runtime._prepare_image(image,instruction,prompt)
        length=int(inputs['input_ids'].shape[-1]);ids=[];prefixes=[];maps=[];queries=[];ended=False;text='';parsed=None
        with SpatialContentRecorder(runtime.model,runtime.config.layers) as recorder:
            recorder.begin(positions,grid)
            output=runtime.model(**inputs,use_cache=True,logits_to_keep=1,output_attentions=False,output_hidden_states=False,return_dict=True)
            initial=recorder.snapshot();ih=recorder.image_hidden;vh=recorder.vision_hidden;mask=inputs.get('attention_mask')
            for step in range(max_new_tokens):
                token=int(output.logits[0,-1].argmax().item())
                if token in runtime.eos_ids:ended=True;break
                ids.append(token);text=runtime.processor.tokenizer.decode(ids,skip_special_tokens=True);prefixes.append(text)
                output,mask=runtime._forward_token(token,output,mask,length,step)
                own=recorder.snapshot();maps.append(own['raw_image_attention']);queries.append(own['hidden'])
                parsed=point_prefix(text,prefixes)
                if parsed is not None or '</g>' in text:break
            del output
        original_parse=parse(text,prefixes)
        selected=parsed if parsed is not None else original_parse
        p=int(np.prod(grid));h=len(initial['head_order']);d=initial['hidden'].size
        arrays=dict(all_attention=np.stack(maps).astype(np.float32) if maps else np.empty((0,h,p),np.float32),
            all_hidden=np.stack(queries).astype(np.float16) if queries else np.empty((0,d),np.float16),image_hidden=ih,vision_hidden=vh)
        _sync_cuda()
        meta=dict(selected,generated_text=text,generated_token_ids=ids,generated_prefixes=prefixes,
            complete=original_parse['complete'],point_prefix_complete=parsed is not None,
            generated_frame_complete=original_parse['complete'],completion_kind='point_prefix' if parsed is not None else 'normal_end',
            parser_only_closing_tag=parsed is not None,synthetic_token_ids_added=False,
            grid_shape=list(grid),image_size=list(image.size),head_order=initial['head_order'],input_tokens=length,
            own_token_forward=True,terminated_by_eos=ended,token_cap_reached=len(ids)==max_new_tokens and not ended,
            system_prompt=prompt,max_new_tokens=max_new_tokens,elapsed_seconds=time.perf_counter()-started,
            ground_truth_in_model_input=False,all_backbone_parameters_frozen=not any(p.requires_grad for p in runtime.model.parameters()))
        return arrays,meta
