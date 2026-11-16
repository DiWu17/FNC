"""CPU canonical, label-free predictors shared by v8 inference and replay audit."""
from pathlib import Path

import numpy as np
import torch

from clickagent.attention_field_ranker import load_field_ranker,make_field_batch
from clickagent.candidate_ranker import load_ranker,make_ranker_batch,decode_ranker
from clickagent.point_aligned_readout import load_aligned,all_candidate_points
from clickagent.semantic_head_experiment import load_readout,score_output
from point_aligned_data import OUT,BASE,read


OLD_MODELS={
    'v5_fixed':('v5','runs/candidate_ranker_v5/models/full.pt'),
    'v6_s1_fixed':('v6','runs/attention_field_v6/models/all.pt'),
    'v6_s2_fixed':('v6','runs/attention_field_v6/models/all_replica.pt'),
}
for suffix in ('s1','s2'):
    OLD_MODELS['v7_point_'+suffix]=('v7_rank',str(BASE/'models'/f'semantic_hard_{suffix}__point.pt'))
    OLD_MODELS['v7_point_'+suffix+'_ownfine']=('v7_own',str(BASE/'models'/f'semantic_hard_{suffix}__point.pt'))
    OLD_MODELS['v7_quality_'+suffix]=('v7_combined',str(BASE/'models'/f'semantic_hard_{suffix}__quality.pt'))


def load_predictors(primary_only=False):
    models={};frozen=read(OUT/'frozen_readouts.json')
    for name,entry in frozen['models'].items():
        if primary_only and name!=frozen['primary']:continue
        model,_=load_aligned(OUT/'models'/entry['file'],'cpu');models[name]=(model,'v8_'+entry['job']['decoder'])
    if not primary_only:
        for name,(kind,path) in OLD_MODELS.items():
            loader=load_ranker if kind=='v5' else load_field_ranker if kind=='v6' else load_readout
            model,_=loader(Path(path),'cpu');models[name]=(model,kind)
    # Fixed decoder is also needed by fixed controls, even for primary-only smoke.
    if 'v5_fixed' not in models:models['v5_fixed']=(load_ranker(Path(OLD_MODELS['v5_fixed'][1]),'cpu')[0],'v5')
    return models


def point_features(output,record,points):
    probs=output['child_logits'].softmax(-1).numpy();child=probs.argmax(-1)
    offsets=output['offsets'].numpy()[np.arange(len(child)),child]
    gh,gw=record['metadata']['grid_shape'];rr,cc=np.divmod(record['parent_indices'],gw)
    xy=points/np.asarray(record['case']['image_size']);relative=xy*np.asarray([gw,gh])-np.stack((cc,rr),-1)
    entropy=-(probs*np.log(np.maximum(probs,1e-12))).sum(-1)/np.log(4)
    return np.concatenate((probs,offsets,entropy[:,None],xy,relative,np.tile([1/gw,1/gh],(len(points),1))),-1).astype(np.float16)


@torch.inference_mode()
def predict_record(record,models,only=None,top_k=5):
    """No target_geometry access, no label tensors passed into any readout."""
    selected={k:v for k,v in models.items() if only is None or k in only}
    if not record['metadata']['target_available']:
        return {name:dict(point=None,candidates=[],failure='no_name',target_name=record['metadata'].get('target_text','')) for name in selected}
    base,_=make_ranker_batch([record],'cpu',labels=False)
    fixed_output=models['v5_fixed'][0](**base)
    fixed_points=all_candidate_points(fixed_output,[record])[0]
    field,_=make_field_batch([record],'cpu',labels=False)
    result={}
    for name,(model,kind) in selected.items():
        if kind=='v5':output=fixed_output
        elif kind.startswith('v7_'):
            inputs=dict(field,frozen_point_features=torch.from_numpy(point_features(fixed_output,record,fixed_points)))
            output=score_output(model(**inputs),'combined' if kind=='v7_combined' else 'rank')
        else:output=model(**field)
        points=all_candidate_points(output,[record])[0] if kind in ('v8_own','v7_own') else fixed_points
        candidates=decode_ranker(output,record,0,top_k=top_k)
        for c in candidates:c['point']=points[c['candidate_index']].tolist()
        result[name]=dict(point=candidates[0]['point'],candidates=candidates,
            all_points=points.tolist(),all_scores=output['scores'].tolist(),
            parent_indices=record['parent_indices'].tolist(),target_name=record['metadata']['target_text'],
            grid_shape=list(record['metadata']['grid_shape']),decoder='own' if kind in ('v8_own','v7_own') else 'frozen_v5',
            ground_truth_in_readout_inputs=False)
    return result
