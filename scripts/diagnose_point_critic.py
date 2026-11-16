"""Validation-only feature sensitivity at fixed weights and fixed candidates.

Mean replacements are diagnostics, NOT matched retraining or deployed policies.
Candidate generation remains image/name-attention based in every condition.
"""
import json
from pathlib import Path

import numpy as np
import torch
from clickagent.action_benchmark import sha256,write_json
from clickagent.point_critic import load_point_critic

OUT=Path('runs/context_verifier_v9')


@torch.inference_mode()
def main():
    torch.set_num_threads(4);data=[]
    for path in sorted((OUT/'critic_cache/val').glob('*.npz')):
        with np.load(path,allow_pickle=False) as z:
            if bool(z['available']):data.append((torch.from_numpy(z['features'].astype(np.float32)),z['labels'].astype(bool),str(z['sample_id'])))
    interventions={
        'attention_statistics_mean':[(320,1472)],
        'explicit_target_state_and_products_mean':[(0,64),(256,320)],
        'explicit_visual_content_and_products_mean':[(64,320)],
        'selected_subpatch_and_within_cell_offset_mean':[(224,256),(288,320),(1474,1478)],
        'coordinate_channels_mean':[(1472,1484)]}
    for name,channels in [('spatial_attention_values_mean',[0,1,2]),
                          ('image_mass_entropy_mean',[3,4]),('own_peak_flags_mean',[5])]:
        interventions[name]=[(320+block*576+head*6+channel,321+block*576+head*6+channel)
            for block in range(2) for head in range(96) for channel in channels]
    indices={name:torch.tensor([i for start,end in ranges for i in range(start,end)],dtype=torch.long)
             for name,ranges in interventions.items()}
    results={}
    for suffix in ['s1','s2']:
        model,_=load_point_critic(OUT/'models'/f'point_critic_{suffix}.pt');assert len(model.mean)==1484
        normal=0;rows={name:dict(hits=0,choice_changes=0,logit_mean_absolute_changes=[],cases=[]) for name in interventions}
        for x,y,sid in data:
            base=model(x);ci=int(base.argmax());normal+=int(y[ci])
            for name,index in indices.items():
                altered=x.clone()
                altered[:,index]=model.mean[index]
                changed=model(altered);other=int(changed.argmax());r=rows[name]
                r['hits']+=int(y[other]);r['choice_changes']+=int(ci!=other)
                r['logit_mean_absolute_changes'].append(float((base-changed).abs().mean()))
                r['cases'].append(dict(sample_id=sid,base_choice=ci,altered_choice=other,base_hit=bool(y[ci]),altered_hit=bool(y[other])))
        for r in rows.values():r['logit_mean_absolute_change']=float(np.mean(r.pop('logit_mean_absolute_changes')))
        results[suffix]=dict(named=len(data),total_validation=172,baseline_hits=normal,interventions=rows,
            checkpoint_sha256=sha256(OUT/'models'/f'point_critic_{suffix}.pt'))
    folder=OUT/'evaluation';folder.mkdir(exist_ok=True)
    write_json(folder/'point_critic_validation_sensitivity.json',dict(complete=True,split='original_validation_only',
        models=results,source_sha256=sha256(Path(__file__)),
        warning='Post-hoc fixed-weight mean replacement, not matched retraining. Attention still forms candidates and contains visual/target information. Removing explicit channels is not removing an input modality. No test outcome or policy selection.'))
    print(json.dumps({s:dict(baseline=v['baseline_hits'],ablations={k:dict(hits=r['hits'],choice_changes=r['choice_changes']) for k,r in v['interventions'].items()}) for s,v in results.items()}),flush=True)


if __name__=='__main__':main()
