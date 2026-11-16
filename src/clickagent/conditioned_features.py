"""Frozen own-target-token features with aligned image-token content.

The historic collectors remain unchanged. Image content comes from decoder
layer 15's *prefill image positions*, not the last query and not a pooled image.
All target query rows are read only after forwarding their actual sampled ID.
"""
from __future__ import annotations

import re
import time

import numpy as np

from .action_model import RawSelectedLayerRecorder
from .model_features import _sync_cuda
from .target_token import TARGET_NAME_PROMPT,TargetTokenRuntime,select_stages


def unmerge_raster(features,grid_shape,merge=2):
    """Qwen3 vision order is block-row, block-col, inner-row, inner-col."""
    gh,gw=map(int,grid_shape)
    if features.shape[0]!=gh*gw*merge*merge:raise ValueError('unmerged vision token count differs from grid')
    d=features.shape[-1]
    return features.reshape(gh,gw,merge,merge,d).transpose(0,2,1,3,4).reshape(gh*merge*gw*merge,d).copy()


class SpatialContentRecorder(RawSelectedLayerRecorder):
    def __init__(self,model,layers,image_layer=15):
        super().__init__(model,layers)
        if image_layer not in layers:raise ValueError('image content layer must be recorded')
        self.image_layer=image_layer
        self.image_hidden=None
        self.vision_hidden=None

    def __enter__(self):
        super().__enter__()
        try:self.handles.append(self.model.model.visual.merger.register_forward_pre_hook(self._vision_hook))
        except BaseException:self.__exit__(None,None,None);raise
        return self

    def begin(self,image_positions,grid_shape):
        super().begin(image_positions,grid_shape)
        self.image_hidden=None
        self.vision_hidden=None

    def _vision_hook(self,module,args):
        import torch
        features=args[0].detach().to(dtype=torch.float16).cpu().numpy()
        self.vision_hidden=unmerge_raster(features,self.grid_shape,2)

    def _hidden_hook(self,layer):
        parent=super()._hidden_hook(layer)
        def hook(module,args,output):
            import torch
            parent(module,args,output)
            hidden=output[0] if isinstance(output,(tuple,list)) else output
            if layer==self.image_layer and hidden.shape[1]>1:
                indices=torch.as_tensor(self.image_positions,device=hidden.device,dtype=torch.long)
                if indices.numel()!=int(np.prod(self.grid_shape)) or int(indices.max())>=hidden.shape[1]:
                    raise RuntimeError('spatial content positions do not match prefill image grid')
                self.image_hidden=hidden[0].index_select(0,indices).detach().to(dtype=torch.float16).cpu().numpy().copy()
        return hook


class ConditionedFeatureRuntime(TargetTokenRuntime):
    def infer_spatial_features(self,image,instruction,max_new_tokens=64):
        import torch
        with self._lock,torch.inference_mode():
            _sync_cuda();started=time.perf_counter()
            inputs,positions,grid=self._prepare_image(image,instruction,TARGET_NAME_PROMPT)
            input_length=int(inputs['input_ids'].shape[-1])
            generated=[];prefixes=[];maps=[];queries=[];ended_by_eos=False;text=''
            with SpatialContentRecorder(self.model,self.config.layers) as recorder:
                recorder.begin(positions,grid)
                outputs=self.model(**inputs,use_cache=True,logits_to_keep=1,output_attentions=False,
                    output_hidden_states=False,return_dict=True)
                snapshot=recorder.snapshot();prefill_attention=snapshot['raw_image_attention']
                prefill_hidden=snapshot['hidden'];image_hidden=recorder.image_hidden
                vision_hidden=recorder.vision_hidden
                if image_hidden is None or vision_hidden is None:raise RuntimeError('no spatial image content captured')
                attention_mask=inputs.get('attention_mask')
                for step in range(max_new_tokens):
                    token_id=int(outputs.logits[0,-1].argmax().item())
                    if token_id in self.eos_ids:ended_by_eos=True;break
                    generated.append(token_id)
                    text=self.processor.tokenizer.decode(generated,skip_special_tokens=True);prefixes.append(text)
                    outputs,attention_mask=self._forward_token(token_id,outputs,attention_mask,input_length,step)
                    snapshot=recorder.snapshot()
                    maps.append(snapshot['raw_image_attention']);queries.append(snapshot['hidden'])
                    if re.search(r'(?:\A|\n)TARGET:[^\r\n]*\r?\n',text):break
                del outputs
            cap=len(generated)==max_new_tokens and not ended_by_eos
            stages,parsed=select_stages(text,prefixes,ended_by_eos=ended_by_eos,token_cap_reached=cap)
            indices=stages.get('target_mean',[])
            p=int(np.prod(grid));heads=len(snapshot['head_order']);qdim=int(prefill_hidden.size)
            attention=np.stack([maps[i-1] for i in indices]) if indices else np.empty((0,heads,p),np.float32)
            query=np.stack([queries[i-1] for i in indices]) if indices else np.empty((0,qdim),np.float32)
            arrays={'target_attention':attention.astype(np.float32),'target_hidden':query.astype(np.float16),
                'image_hidden':image_hidden,'vision_hidden':vision_hidden,'prefill_attention':prefill_attention.astype(np.float32),
                'prefill_hidden':prefill_hidden.astype(np.float16)}
            _sync_cuda()
            meta={**parsed,'generated_text':text,'generated_token_ids':generated,'generated_prefixes':prefixes,
                'stages':stages,'target_raw_indices':indices,'target_query_absolute_positions':[input_length+i-1 for i in indices],
                'grid_shape':list(grid),'image_size':list(image.size),'head_order':snapshot['head_order'],
                'image_content_layer':15,'image_content_positions':positions.tolist(),'image_content_source':'prefill decoder output at actual image-token positions',
                'fine_grid_shape':[int(grid[0])*2,int(grid[1])*2],
                'vision_content_source':'last vision block before 2x2 merger, restored from block order to full raster',
                'input_tokens':input_length,'own_token_forward':True,'ground_truth_in_model_input':False,
                'terminated_by_eos':ended_by_eos,'token_cap_reached':cap,'system_prompt':TARGET_NAME_PROMPT,
                'elapsed_seconds':time.perf_counter()-started,'all_backbone_parameters_frozen':not any(p.requires_grad for p in self.model.parameters())}
            return arrays,meta
