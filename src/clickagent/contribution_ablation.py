"""Input-only ablations for the unchanged22,050-parameter quality scorer.

No labels, target geometry or outcome-dependent keys enter this interface.
All variants retain candidate coordinates, public metadata, masks and shapes.
"""
from __future__ import annotations
import hashlib
import numpy as np
import torch

VARIANTS=('aligned','no_evidence','spatial_shuffle','attention_only','hidden_only','token_mean')


def permuted_indices(sample_id, valid_indices):
    payload=__import__('json').dumps([sample_id,'spatial_permutation'],sort_keys=True,ensure_ascii=False,separators=(',',':'))
    rng=np.random.default_rng(int(hashlib.sha256(payload.encode()).hexdigest()[:8],16))
    result=[]
    for indices in valid_indices:
        indices=np.asarray(indices,np.int64)
        result.append(indices[rng.permutation(len(indices))] if len(indices)>1 else indices.copy())
    return result


def transform_inputs(inputs, lengths, sample_ids, variant):
    if variant not in VARIANTS:raise ValueError('Unknown contribution ablation')
    if len(lengths)!=len(sample_ids) or sum(lengths)!=len(inputs['base']):raise ValueError('Query boundaries differ')
    if variant=='aligned':return inputs
    original=inputs['state'];mask=inputs['mask'];state=original.clone()
    if original.shape[:3]!=mask.shape or original.shape[-1]!=294:raise ValueError('State/mask shape mismatch')
    if variant=='no_evidence':state.zero_()
    elif variant=='attention_only':state[...,288:]=0
    elif variant=='hidden_only':state[...,:288]=0
    elif variant=='token_mean':
        weights=mask[...,None].float();mean=(original.float()*weights).sum(2,keepdim=True)/weights.sum(2,keepdim=True).clamp_min(1)
        state=torch.where(mask[...,None],mean.expand_as(original),torch.zeros_like(mean)).to(original.dtype)
    elif variant=='spatial_shuffle':
        start=0
        for sid,n in zip(sample_ids,lengths):
            valid=[torch.nonzero(mask[start:start+n,v].any(-1),as_tuple=False).flatten().cpu().numpy() for v in range(4)]
            orders=permuted_indices(sid,valid)
            for v,(indices,order) in enumerate(zip(valid,orders)):
                if len(indices)>1:
                    dest=torch.as_tensor(indices+start,device=state.device);source=torch.as_tensor(order+start,device=state.device)
                    state[dest,v]=original[source,v]
            start+=n
    return dict(inputs,state=state)
